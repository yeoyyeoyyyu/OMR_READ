import os
import warnings
from collections import OrderedDict

import cv2
import tensorflow as tf

import KICE.common.KICE_PREPROCESS_TEMPLATE as pe
import KICE.common.KICE_CROP_RECOG as crop
import common.unicode as unicode
from common.OMR_S3_Conn import S3FileDelMove
from common.dbConnect import dblogger


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

    dbErrorList = ['FAIL LOAD IMG', 'FAIL TEMPLATE ROI', 'FAIL CHECK IMG', 'FAIL ROI MASKING', 'FAIL CROP NAME REGION', \
                   'FAIL CROP CODE REGION', 'FAIL CROP SHORT ANS REGION', 'FAIL CROP SELECT TYPE REGION',\
                   'FAIL CROP LONG ANS REGION', 'FAIL CROP SELECT ANS REGION', 'FAIL RECOG NAME', \
                   'FAIL RECOG CODE', 'FAIL RECOG SHORT ANS', 'FAIL RECOG LONG ANS', \
                   'FAIL RECOG SELECT ANS', 'FAIL RECOG SELECT TYPE']

    successFiles = 0
    dbError = 0

    try:
        gray = cv2.resize(gray, dsize=(1100, 850), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
        dbError += 1 # 0

        # 각 템플릿에 일치하는 곳의 좌상단 좌표와 검색된 템플릿 표시가 된 이미지 저장
        regionXY, templateImg = pe.setROI(OMR_MST_CD, gray, templateImages)
        dbError += 1 # 1

        # checkIMG = check.checkImage(gray)
        # checkTop = checkIMG.check_top()
        # checkLeft = checkIMG.check_left(regionXY[3])
        #
        # if checkTop == False:
        #     raise Exception('FAIL CHECK TOP')
        # if checkLeft == False:
        #     raise  Exception('FAIL CHECK LEFT')
        dbError += 1 # 2

        regionXYWH = []
        regionXYWH.append((regionXY[3][0], regionXY[3][1] + 88, 201, 326))  # 수험번호
        regionXYWH.append((regionXY[0][0], regionXY[0][1] + 394, 244, 351))  # 이름 (기준: template15)

        regionXYWH.append((regionXY[4][0] + 41, regionXY[4][1] + 74, 19, 122))  # 과목선택(3)

        # 공통과목 객관식
        regionXYWH.append((regionXY[0][0] + 40, regionXY[0][1] + 36, 110, 261))   # 답안08 (기준: template15)
        regionXYWH.append((regionXY[0][0] + 180, regionXY[0][1] + 35, 108, 233))  # 답안0915
        # 공통과목 주관식
        regionXYWH.append((regionXY[1][0], regionXY[1][1] + 36, 356, 327))        # 답안1620
        regionXYWH.append((regionXY[2][0] - 145, regionXY[2][1] + 35, 137, 327))  # 답안2122 (기준: template2328)

        # 선택과목 객관식
        regionXYWH.append((regionXY[2][0] + 40, regionXY[2][1] + 34, 111, 198))   # 답안2328
        # 선택과목 주관식
        regionXYWH.append((regionXY[2][0] + 155, regionXY[2][1] + 34, 133, 328))  # 답안2930 (기준: template2328)

        dbError += 1  # 3

        """
        인식 시작
        """
        cropRecog = crop.CropRecognition(binary, base_dir, OMR_MST_CD, name0Model, name1Model, idx0Model,
                                         case1Model, answerModel, sexModel, longAnswerModel, select02Model, select09Model,
                                         birthdayModel)
        # cropName --> stdName
        nameRegion = cropRecog.cropRegion(binary, regionXYWH[1], 0, 0)
        dbError += 1  # 4

        # cropStudentCode --> stdCode
        codeRegion = cropRecog.cropRegion(binary, regionXYWH[0], 0, 0)
        dbError += 1  # 5

        # ans20Region --> answerList
        ans9Region = cropRecog.cropRegion(binary, regionXYWH[3], 0, 0)
        ans15Region = cropRecog.cropRegion(binary, regionXYWH[4], 0, 0)
        dbError += 1  # 6

        select02Region = cropRecog.cropRegion(binary, regionXYWH[2], 17, 88)
        dbError += 1  # 7

        ans1620Region = cropRecog.cropRegion(binary, regionXYWH[5], 0, 0)
        ans2122Region = cropRecog.cropRegion(binary, regionXYWH[6], 0, 0)
        dbError += 1  # 8

        ans2328Region = cropRecog.cropRegion(binary, regionXYWH[7], 0, 0)
        ans2930Region = cropRecog.cropRegion(binary, regionXYWH[8], 0, 0)
        dbError += 1  # 9

        '1. 이름 인식'
        stdName = []
        stdName, nameAccuracy = cropRecog.mapName(nameRegion)
        while 'none' in stdName:
            stdName.remove('none')
        if stdName == []:
            stdName = ['-1']
        stdName = unicode.join_jamos("".join(stdName))
        dbError += 1  # 10

        '2. 수험번호 인식'
        studentCode = []
        studentCode, codeAccuracy = cropRecog.mapCode(codeRegion)
        stdCode = "".join([str(_) for _ in studentCode])
        dbError += 1  # 11

        answerList = []
        answerAccuracy = []
        '객관식 1번 ~ 9번 인식'
        ans, acc = cropRecog.mapShortAnswer(ans9Region, 8)
        answerList.extend(ans)
        answerAccuracy.append(acc)
        # print(answerList, answerAccuracy)
        '객관식 10번 ~ 15번 인식'
        ans, acc = cropRecog.mapShortAnswer(ans15Region, 7)
        answerList.extend(ans)
        answerAccuracy.append(acc)
        # print(answerList)
        dbError += 1  # 12

        '주관식 16번 ~ 20번 인식'
        longAnswerList = []
        longAnswerListtAccuracy = []
        ans, acc = cropRecog.mapLongAnswer(ans1620Region, 5)
        longAnswerList.extend(ans)
        longAnswerListtAccuracy.append(acc)
        '주관식 21번 ~ 22번 인식'
        ans, acc = cropRecog.mapLongAnswer(ans2122Region, 2)
        longAnswerList.extend(ans)
        longAnswerListtAccuracy.append(acc)
        dbError += 1  # 13

        '선택과목 객관식 23번 ~28번 인식'
        SanswerList = []
        SanswerAccuracy = []
        ans, acc = cropRecog.mapShortAnswer(ans2328Region, 6)
        SanswerList.extend(ans)
        SanswerAccuracy.append(acc)
        '선택과목 주관식 29번 ~ 30번 인식'
        ans, acc = cropRecog.mapLongAnswer(ans2930Region, 2)
        SanswerList.extend(ans)
        SanswerAccuracy.append(acc)

        answerList.extend(longAnswerList + SanswerList)
        missCnt = answerList.count(-1)
        dbError += 1  # 14

        '과목선택 인식'
        stdSelect02, stdAccuracy02 = cropRecog.mapSelect02(select02Region)
        # stdSelect = "가형" if stdSelect == "남" else "나형"
        dbError += 1  # 15

        successFiles += 1

    except:
        LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + dbErrorList[dbError]
        dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')
        print(LOG_MSG)

    if dbError == 16:
        print("stdName: ", stdName, nameAccuracy)
        print("stdCode: ", stdCode, codeAccuracy)
        print("selectOption: ", stdSelect02, stdAccuracy02)
        print("answerList: ", answerList, sum(answerAccuracy) / len(answerAccuracy))
        print("answer miss(missCnt): ", missCnt)

        jsonData = OrderedDict()

        jsonData["omr_img"] = base_dir + fileNM
        jsonData["stdn_nm"] = stdName
        jsonData["exmn_no"] = stdCode
        jsonData["sex"] = '-1'
        jsonData["bthday"] = '-1-1-1-1-1-1-1-1'
        jsonData["lsn_cd"] = stdSelect02
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