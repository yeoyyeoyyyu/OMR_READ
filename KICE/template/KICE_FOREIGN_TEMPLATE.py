import os
import warnings
from collections import OrderedDict

import cv2
import tensorflow as tf

import KICE.common.KICE_PREPROCESS_TEMPLATE as pe
import KICE.common.KICE_CROP_RECOG as crop
import common.unicode as unicode
from common.OMR_S3_Conn import S3FileDelMove
from common.dbConnect import dblogger, dbReupld, GetLsnCd


def main(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, base_dir, fileNM, IMG, templateImages, gray, C):
    print('convert template')
    warnings.filterwarnings(action='ignore')
    model_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/models/'
    
    try:        
        name0Model      = tf.lite.Interpreter(model_dir + 'S1_Case018_augmentation.tflite')
        name1Model      = tf.lite.Interpreter(model_dir + 'S1_Case020_augmentation.tflite')
        idx0Model       = tf.lite.Interpreter(model_dir + 'S1_Case09_augmentation.tflite')
        case1Model      = tf.lite.Interpreter(model_dir + 'S1_Case09_augmentation.tflite')
        answerModel     = tf.lite.Interpreter(model_dir + 'S1_Case05_augmentation.tflite')
        longAnswerModel = tf.lite.Interpreter(model_dir + 'S1_Case09_augmentation.tflite')
        sexModel        = tf.lite.Interpreter(model_dir + 'S1_Sex_augmentation.tflite')
        select02Model   = tf.lite.Interpreter(model_dir + 'S1_Select02_augmentation.tflite')
        birthdayModel   = tf.lite.Interpreter(model_dir + 'S1_Case09_augmentation.tflite')
        select09Model   = tf.lite.Interpreter(model_dir + 'S1_ShortCase09_augmentation.tflite')
    except:
        dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, 'FAIL LOAD MODEL', None, 'failure')

    dbErrorList = ['FAIL LOAD IMG', 'FAIL TEMPLATE ROI', 'FAIL ROI MASKING', \
                   'FAIL CROP NAME REGION', 'FAIL CROP CODE REGION', 'FAIL CROP SELECT REGION', 'FAIL CROP ANS REGION', \
                   'FAIL RECOG NAME', 'FAIL RECOG CODE', 'FAIL CROP SELECT REGION', 'FAIL RECOG ANS']

    successFiles = 0
    dbError = 0
    lsn_cds = GetLsnCd(OMR_MST_CD)
    try:
        gray = cv2.resize(gray, dsize=(1100, 850), interpolation=cv2.INTER_AREA)
        # binary = pe.threshold(gray, BSTOR_CD, OMR_MST_CD)
        blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
        # _, binary = cv2.threshold(blur, C, 255, cv2.THRESH_BINARY)
        dbError += 1 # 0

        # 각 템플릿에 일치하는 곳의 좌상단 좌표와 검색된 템플릿 표시가 된 이미지 저장
        regionXY, templateImg = pe.setROI(OMR_MST_CD, gray, templateImages)
        dbError += 1 # 1

        regionXYWH = []
        regionXYWH.append((regionXY[1][0] + 1, regionXY[1][1] + 89, 202, 327))  # 수험번호
        regionXYWH.append((regionXY[2][0], regionXY[2][1] + 88, 241, 350))  # 이름

        regionXYWH.append((regionXY[3][0] + 80, regionXY[3][1] + 51, 22, 301))  # 선택과목
        regionXYWH.append((regionXY[0][0] + 37, regionXY[0][1] + 35, 113, 668))  # 답안20 (기준 : templateAns)
        regionXYWH.append((regionXY[0][0] + 211, regionXY[0][1] + 35, 111, 333))  # 답안30 (기준 : templateAns)
        dbError += 1  # 2

        """
        인식 시작
        s3 storage에 있는 h5 모델 이름들도 같이 전달: models
        """

        cropRecog = crop.CropRecognition(binary, base_dir, OMR_MST_CD, name0Model, name1Model, idx0Model,
                                         case1Model, answerModel, sexModel, longAnswerModel, select02Model, select09Model,
                                         birthdayModel)
        # cropName --> stdName
        nameRegion = cropRecog.cropRegion(binary, regionXYWH[1], 0, 0)
        dbError += 1  # 3

        # cropStudentCode --> stdCode
        codeRegion = cropRecog.cropRegion(binary, regionXYWH[0], 0, 0)
        dbError += 1  # 4

        select09Region = cropRecog.cropRegion(binary, regionXYWH[2], 0, 0)
        dbError += 1  # 5

        # ans20Region --> answerList
        ans20Region = cropRecog.cropRegion(binary, regionXYWH[3], 0, 0)
        ans30Region = cropRecog.cropRegion(binary, regionXYWH[4], 0, 0)
        dbError += 1  # 6

        '1. 이름 인식'
        stdName = []
        stdName, nameAccuracy = cropRecog.mapName(nameRegion)

        while 'none' in stdName:
            stdName.remove('none')
        if stdName == []:
            stdName = ['-1']

        stdName = unicode.join_jamos("".join(stdName))
        dbError += 1  # 7

        '2. 수험번호 인식'
        studentCode = []
        studentCode, codeAccuracy = cropRecog.mapCode(codeRegion)

        stdCode = "".join([str(_) for _ in studentCode])
        dbError += 1  # 8

        '3. 선택과목 인식'
        stdSelect, selectAccuracy = cropRecog.mapSelect09(select09Region)
        if int(stdSelect) not in lsn_cds:
            stdSelect = '-1'
        dbError += 1  # 9

        '4. 답안 인식'
        missCnt = 0
        answerList = []
        answerAccuracy = []

        ans, acc = cropRecog.mapAnswer(ans20Region, 20)
        answerList.extend(ans)
        answerAccuracy.append(acc)
        # print(answerList, answerAccuracy)

        ans, acc = cropRecog.mapAnswer(ans30Region, 10)
        answerList.extend(ans)
        answerAccuracy.append(acc)

        missCnt = answerList.count(-1)
        dbError += 1  # 10

        successFiles += 1

    except:
        LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + dbErrorList[dbError]
        dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')
        print(LOG_MSG)
        dbReupld(fileNM)

    if dbError == 11:
        print("stdName: ", stdName, nameAccuracy)
        print("stdCode: ", stdCode, codeAccuracy)
        print("stdSelect: ", stdSelect, selectAccuracy)
        print("answerList: ", answerList, sum(answerAccuracy) / len(answerAccuracy))
        print("answer miss(missCnt): ", missCnt)

        # json에 담기 위한 자료구조 생성
        jsonData = OrderedDict()

        # 저장 구조에 맞게 key, value 저장
        jsonData["omr_img"] = base_dir + fileNM
        jsonData["stdn_nm"] = stdName
        jsonData["exmn_no"] = stdCode
        jsonData["sex"] = '-1'
        jsonData["bthday"] = '-1-1-1-1-1-1-1-1'
        jsonData["lsn_cd"] = stdSelect
        jsonData["mark_no"] = answerList
        jsonData["err_cnt"] = missCnt

        try:
            # 인식 완료한 파일은 job_done/ 폴더로 이동하고 디렉토리 위치에서 delete
            S3FileDelMove(base_dir, fileNM, 1)
        except:
            LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + 'FAIL DELETE FILE'
            dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')

    else:
        jsonData = OrderedDict()
        jsonData["omr_img"] = base_dir + fileNM
        jsonData["stdn_nm"] = '-1'
        jsonData["exmn_no"] = '-1-1-1-1-1-1-1-1-1-1-1'
        jsonData["sex"] = '-1'
        jsonData["bthday"] = '-1-1-1-1-1-1-1-1'
        jsonData["lsn_cd"] = int(-1)
        jsonData["mark_no"] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, \
                               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

        try:
            # job_fail/ 로 보내고 지움
            S3FileDelMove(base_dir, fileNM, 0)
        except:
            LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + 'FAIL DELETE FILE'
            dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')

    return successFiles, jsonData