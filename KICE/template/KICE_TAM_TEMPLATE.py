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


    dbErrorList = ['FAIL LOAD IMG', 'FAIL TEMPALTE ROI', 'FAIL CHECK IMG', 'FAIL ROI MASKING', \
                   'FAIL CROP NAME REGION', 'FAIL CROP CODE REGION', 'FAIL CROP BIRTHDAY REGION', \
                   'FAIL CROP FIRST SELECT REGION', 'FAIL CROP ANS REGION', 'FAIL CROP SECOND SELECT REGION', \
                   'FAIL CROP SECOND ANS REGION', 'FAIL CROP SEX REGION', 'FAIL RECOG NAME', 'FAIL RECOG CODE', \
                   'FAIL RECOG BIRTHDAY', 'FAIL RECOG FIRST SELECT', 'FAIL RECOG FIRST ANS', 'FAIL RECOG SECOND SELECT', \
                   'FAIL RECOG SECOND ANS', 'FAIL RECOG SEX']

    successFiles = 0
    dbError = 0
    lsn_cds = GetLsnCd(OMR_MST_CD)
    try:
        gray = cv2.resize(gray, dsize=(1100, 850), interpolation=cv2.INTER_AREA)
        # binary = pe.threshold(gray, BSTOR_CD, OMR_MST_CD)
        blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        # binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
        dbError += 1 # 0

        # 각 템플릿에 일치하는 곳의 좌상단 좌표와 검색된 템플릿 표시가 된 이미지 저장
        regionXY, templateImg = pe.setROI(OMR_MST_CD, gray, templateImages)
        dbError += 1 # 1

        # checkIMG = check.checkImage(gray)
        # checkTop = checkIMG.check_top()
        # checkLeft = checkIMG.check_left(regionXY[1])
        #
        # if checkTop == False:
        #     raise Exception('FAIL CHECK TOP')
        # if checkLeft == False:
        #     raise Exception('FAIL CHECK LEFT')
        dbError += 1  # 2

        regionXYWH = []
# 7113077218-01
        regionXYWH.append((regionXY[1][0], regionXY[1][1] + 90, 202, 326))  # 수험번호
        regionXYWH.append((regionXY[2][0], regionXY[2][1] + 93, 242, 350))  # 이름

        regionXYWH.append((regionXY[0][0], regionXY[0][1] + 202, 42, 327))  # 1과목 선택 (기준 : templateAns)
        regionXYWH.append((regionXY[0][0] + 68, regionXY[0][1] + 38, 102, 658))  # 1과목 답안 (기준 : templateAns)

        regionXYWH.append((regionXY[0][0] + 261, regionXY[0][1] + 200, 40, 331))  # 2과목 선택 (기준 : templateAns)
        regionXYWH.append((regionXY[0][0] + 327, regionXY[0][1] + 38, 102, 658))  # 2과목 답안 (기준 : templateAns)

        dbError += 1  # 3

        '''
        인식 시작
        '''
        # ETOOS_CROP_RECOG.py CropRecognition 클래스 호출
        cropRecog = crop.CropRecognition(binary, base_dir, OMR_MST_CD, name0Model, name1Model, idx0Model,
                                         case1Model, answerModel, sexModel, longAnswerModel, select02Model, select09Model,
                                         birthdayModel)

        # 이름
        nameRegion = cropRecog.cropRegion(binary, regionXYWH[1], 0, 0)
        dbError += 1  # 4
        # 수험번호
        codeRegion = cropRecog.cropRegion(binary, regionXYWH[0], 0, 0)
        dbError += 1  # 5
        # 1과목 선택
        firstSelect = cropRecog.cropRegion(binary, regionXYWH[2], 0, 0)
        dbError += 1  # 6
        # 1과목 답안
        first20Region = cropRecog.cropRegion(binary, regionXYWH[3], 0, 0)
        dbError += 1  # 7
        # 2과목 선택
        secondSelect = cropRecog.cropRegion(binary, regionXYWH[4], 0, 0)
        dbError += 1  # 8
        # 2과목 답안
        second20Region = cropRecog.cropRegion(binary, regionXYWH[5], 0, 0)
        dbError += 1  # 9

        '''
        1. 이름 인식
        '''
        # 인식한 이름과 정확도 저장
        stdName = []
        stdName, nameAccuracy = cropRecog.mapName(nameRegion)
        while 'none' in stdName:
            stdName.remove('none')
        if stdName == []:
            stdName = ['-1']
        stdName = unicode.join_jamos("".join(stdName))
        dbError += 1  # 10

        '''
        2. 수험번호 인식
        '''
        # 인식한 수험번호와 정확도 저장
        studentCode = []
        studentCode, codeAccuracy = cropRecog.mapCode(codeRegion)
        stdCode = "".join([str(_) for _ in studentCode])
        dbError += 1  # 11

        '''
        4. 선택과목 인식
        '''
        missCnt = 0

        # 1과목 선택 영역 인식하여 저장
        firstSubject, acc = cropRecog.mapSelect(firstSelect)
        if int(firstSubject) not in lsn_cds:
            firstSubject = '-1'
        dbError += 1  # 12

        # 1과목 마킹 영역 인식하여 저장
        firstAnswerList = []
        firstAnswerAccuracy = []
        ans, acc = cropRecog.mapAnswer(first20Region, 20)
        firstAnswerList.extend(ans)
        firstAnswerAccuracy.append(acc)
        missCnt += firstAnswerList.count(-1)

        dbError += 1  # 13

        # 2과목 선택 영역 인식하여 저장
        secondSubject, acc = cropRecog.mapSelect(secondSelect)
        if int(secondSubject) not in lsn_cds:
            secondSubject = '-2'
        dbError += 1  # 14

        # 2과목 마킹 영역 인식하여 저장
        secondAnswerList = []
        secondAnswerAccuracy = []
        ans, acc = cropRecog.mapAnswer(second20Region, 20)
        secondAnswerList.extend(ans)
        secondAnswerAccuracy.append(acc)
        missCnt += secondAnswerList.count(-1)

        dbError += 1  # 15

        successFiles += 1

    except:
        LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + dbErrorList[dbError]
        dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')
        dbReupld(fileNM)

    if dbError == 16:
        # 확인용 프린트
        print("stdName: ", stdName, nameAccuracy)
        print("stdCode: ", stdCode, codeAccuracy)
        print("first sbj: ", firstSubject, firstAnswerList, sum(firstAnswerAccuracy) / len(firstAnswerAccuracy))
        print("second sbj: ", secondSubject, secondAnswerList, sum(secondAnswerAccuracy) / len(secondAnswerAccuracy))
        print("answer miss(missCnt): ", missCnt)

        # json에 담기 위한 자료구조 생성
        jsonData = OrderedDict()
        tam1 = OrderedDict()
        tam2 = OrderedDict()

        lsn_list = []

        tam1["lsn_cd"] = int(firstSubject)
        tam1["mark_no"] = firstAnswerList
        tam1["err_cnt"] = int(firstAnswerList.count(-1))
        tam1["lsn_seq"] = 1

        tam2["lsn_cd"] = int(secondSubject)
        tam2["mark_no"] = secondAnswerList
        tam2["err_cnt"] = int(secondAnswerList.count(-1))
        tam2["lsn_seq"] = 2

        lsn_list.append(tam1)
        lsn_list.append(tam2)

        # 저장 구조에 맞게 key, value 저장
        jsonData["omr_img"] = base_dir + fileNM
        jsonData["stdn_nm"] = stdName
        jsonData["exmn_no"] = stdCode
        jsonData["sex"] = '-1'
        jsonData["bthday"] = '-1-1-1-1-1-1-1-1'
        jsonData["lsn_list"] = lsn_list

        try:
            # 인식 완료한 파일은 job_done/ 폴더로 이동하고 디렉토리 위치에서 delete
            S3FileDelMove(base_dir, fileNM, 1)
        except:
            LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + 'FAIL DELETE FILE'
            dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')

    else:
        jsonData = OrderedDict()
        tam1 = OrderedDict()
        tam2 = OrderedDict()

        lsn_list = []

        tam1["lsn_cd"] = int(-1)
        tam1["mark_no"] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        tam1["lsn_seq"] = 1

        tam2["lsn_cd"] = int(-2)
        tam2["mark_no"] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        tam2["lsn_seq"] = 2

        lsn_list.append(tam1)
        lsn_list.append(tam2)

        jsonData["omr_img"] = base_dir + fileNM
        jsonData["stdn_nm"] = '-1'
        jsonData["exmn_no"] = '-1-1-1-1-1-1-1-1-1-1-1'
        jsonData["sex"] = '-1'
        jsonData["bthday"] = '-1-1-1-1-1-1-1-1'
        jsonData["lsn_list"] = lsn_list

        try:
            # job_fail/ 로 보내고 지움
            S3FileDelMove(base_dir, fileNM, 0)
        except:
            LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + 'FAIL DELETE FILE'
            dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')

    # ETOOS_MAIN.py에 인식한 json data 리턴(ETOOS_MAIN.py에서 S3에 업로드)
    return successFiles, jsonData