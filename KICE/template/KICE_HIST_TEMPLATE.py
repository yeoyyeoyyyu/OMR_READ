import os
import warnings
from collections import OrderedDict

import cv2
import tensorflow as tf

import KICE.common.KICE_PREPROCESS_TEMPLATE as pe
import KICE.common.KICE_CROP_RECOG as crop
import common.unicode as unicode
from common.OMR_S3_Conn import S3FileDelMove
from common.dbConnect import dblogger, dbReupld


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

    successFiles = 0
    dbError = 0

    dbErrorList = ['FAIL LOAD IMG', 'FAIL TEMPLATE ROI', 'FAIL ROI MASKING', 'FAIL CROP NAME REGION', \
                   'FAIL CROP CODE REGION', 'FAIL CROP BIRTHDAY REGION', 'FAIL CROP ANS REGION', 'FAIL CROP SEX REGION', \
                   'FAIL RECOG NAME', 'FAIL RECOG CODE', 'FAIL RECOG BIRTHDAY', 'FAIL RECOG ANS', 'FAIL RECOG SEX']

    try:
        gray = cv2.resize(gray, dsize=(1100, 850), interpolation=cv2.INTER_AREA)
        # binary = pe.threshold(gray, BSTOR_CD)
        blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
        dbError += 1 # 0
        print('c')
        # 각 템플릿에 일치하는 곳의 좌상단 좌표와 검색된 템플릿 표시가 된 이미지 저장
        regionXY, templateImg = pe.setROI(OMR_MST_CD, gray, templateImages)
        dbError += 1 # 1
        print('a')
        '''
        template 순서
        답안
        생년월일
        수험번호
        성별
        '''
        regionXYWH = []
        regionXYWH.append((regionXY[1][0] + 11, regionXY[1][1] + 74, 158, 169))  # 생일
        regionXYWH.append((regionXY[2][0], regionXY[2][1] + 89, 201, 323))  # 수험번호
        regionXYWH.append((regionXY[3][0], regionXY[3][1] + 345, 242, 350))  # 이름 (기준: templateSex)
        regionXYWH.append((regionXY[3][0] + 42, regionXY[3][1] + 43, 18, 51))  # 성별
        regionXYWH.append((regionXY[0][0] + 36, regionXY[0][1] + 33, 110, 663))  # 답안
        print('b')
        dbError += 1  # 2

        # 이미지 인식 클래스 호출. 파라미터: 인식에 사용되는 모델들
        cropRecog = crop.CropRecognition(binary, base_dir, OMR_MST_CD, name0Model, name1Model, idx0Model,
                                         case1Model, answerModel, sexModel, longAnswerModel, select02Model,
                                         select09Model, birthdayModel)

        # 이름
        nameRegion = cropRecog.cropRegion(binary, regionXYWH[2], 0, 0)
        dbError += 1  # 3
        # 수험번호
        codeRegion = cropRecog.cropRegion(binary, regionXYWH[1], 0, 0)
        dbError += 1  # 4
        # 생년월일
        birthdayRegion = cropRecog.cropRegion(binary, regionXYWH[0], 0, 0)
        dbError += 1  # 5
        # 답안
        ans20Region = cropRecog.cropRegion(binary, regionXYWH[4], 0, 0)
        dbError += 1  # 6
        # 성별
        sexRegion = cropRecog.cropRegion(binary, regionXYWH[3], 41, 120)
        dbError += 1  # 7

        '''
        1. 이름 인식
        '''
        # 인식한 이름과 정확도 저장
        stdName, nameAccuracy = cropRecog.mapName(nameRegion)

        while 'none' in stdName:
            stdName.remove('none')
            if stdName == []:
                stdName = ['-1']

        stdName = unicode.join_jamos("".join(stdName))
        dbError += 1  # 8

        '''
        2. 수험번호 인식
        '''
        studentCode, codeAccuracy = cropRecog.mapCode(codeRegion)
        stdCode = "".join([str(_) for _ in studentCode])
        dbError += 1  # 9

        '''
        3. 생년월일 인식
        '''
        studentBday, bdayAccuracy = cropRecog.mapBday(birthdayRegion)
        stdBday = "".join([str(_) for _ in studentBday])
        dbError += 1  # 10

        '''
        4. 답안 인식
        '''
        answerList, answerAccuracy = cropRecog.mapAnswer(ans20Region, 20)
        missCnt = answerList.count(-1)
        dbError += 1  # 11

        '''
        5. 성별인식
        '''
        stdSex, sexAccuracy = cropRecog.mapSex(sexRegion)
        dbError += 1  # 12

        successFiles += 1

    except:
        LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + dbErrorList[dbError]
        dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')
        print(LOG_MSG)
        dbReupld(fileNM)

    if dbError == 13:
        # 확인용 출력
        print("stdName: ", stdName, nameAccuracy)
        print("stdSex: ", stdSex, sexAccuracy)
        print("stdCode: ", stdCode, codeAccuracy)
        print("stdBday: ", stdBday, bdayAccuracy)
        # print("answerList: ", answerList, sum(answerAccuracy) / len(answerAccuracy))
        print("answerLIst: ", answerList)
        print("answer miss(missCnt): ", missCnt)

        # json에 key - value 저장
        jsonData = OrderedDict()
        jsonData["omr_img"] = base_dir + fileNM
        jsonData["stdn_nm"] = stdName
        jsonData["exmn_no"] = stdCode
        jsonData["sex"] = stdSex
        jsonData["bthday"] = stdBday
        jsonData["lsn_cd"] = 34
        jsonData["mark_no"] = answerList
        jsonData["err_cnt"] = missCnt

        try:
            S3FileDelMove(base_dir, fileNM, 1)
        except:
            LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + 'FAIL DELETE FILE'
            dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')

    else:
        jsonData = OrderedDict()
        jsonData["omr_img"] = base_dir + fileNM
        jsonData["stdn_nm"] = '-1'
        jsonData["exmn_no"] = '-1-1-1-1-1-1-1-1-1-1-1'  # 11개
        jsonData["sex"] = '-1'  # 1개
        jsonData["bthday"] = '-1-1-1-1-1-1-1-1'  # 8개
        jsonData["lsn_cd"] = 34
        jsonData["mark_no"] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

        try:
            S3FileDelMove(base_dir, fileNM, 0)
        except:
            LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + 'FAIL DELETE FILE'
            dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')

    return successFiles, jsonData