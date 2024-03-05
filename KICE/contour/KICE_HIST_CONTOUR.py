import os
import warnings
from multiprocessing import current_process
from collections import OrderedDict

import cv2
import tensorflow as tf

import KICE.common.KICE_PREPROCESS_CONTOUR as pe
import KICE.common.KICE_CROP_RECOG as crop
import KICE.template.KICE_HIST_TEMPLATE as templateHIST
import common.unicode as unicode #한글 자모 결합 오픈소스
from common.OMR_S3_Conn import S3FileDelMove
from common.dbConnect import dblogger, getThreshold, UpdateThreshold


def main(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, base_dir, fileNMs, templates, is_local):
    warnings.filterwarnings(action='ignore')
    model_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/models/'
    dbError = 0
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

    # 결과를 저장할 리스트
    TOT = []
    # 로그 남기기 용으로 성공한 파일 갯수
    successFiles = 0
    cvt = 0
    # 현재 인식 회수 확인 카운트
    cnt = 0
    pid = current_process().pid
    C = getThreshold(BSTOR_CD, OMR_MST_CD)

    # 모든 파일에 대하여 하나씩 반복문
    for fileNM in fileNMs:
        dbError = 0
        cnt += 1
        print('\n{0} / {1}, pid : {2}'.format(cnt, len(fileNMs), pid))
        try:
            # call class PreprocessingImage in ETOOS_PREPROCESS_IMG.py
            srcImg = pe.PreprocessingImage(base_dir, OMR_MST_CD)
            src, gray, contourBinary = srcImg.loadImage(fileNM)        
            dbError += 1 # 0
            
            # detect contour
            contours, _, Y, _, df = srcImg.setContour(10, gray, contourBinary)
            dbError += 1 # 1

            code = []
            sex = []
            name = []
            birthday = []
            answer20 = []
            sexIdx = 0
            nameIdx = 0

            for i in range(len(contours)):
                if code == [] and int(df['r'][i] * 100) in range(180, 220):
                    code = cv2.resize(contours[i], dsize=(205, 418), interpolation=cv2.INTER_AREA)

                elif sex == [] and int(df['r'][i] * 100) in range(150, 180):
                    sex = cv2.resize(contours[i], dsize=(63, 103), interpolation=cv2.INTER_AREA)
                    sexIdx = i

                elif name == [] and int(df['r'][i] * 100) in range(170, 200):
                    name = cv2.resize(contours[i], dsize=(244, 448), interpolation=cv2.INTER_AREA)
                    nameIdx = i

                elif birthday == [] and int(df['r'][i] * 100) in range(120, 150):
                    birthday = cv2.resize(contours[i], dsize=(183, 247), interpolation=cv2.INTER_AREA)

                elif answer20 == [] and int(df['r'][i] * 100) in range(470, 520):
                    answer20 = cv2.resize(contours[i], dsize=(144, 703), interpolation=cv2.INTER_AREA)

            if Y[sexIdx] > Y[nameIdx]:
                tmp = sex
                sex = name
                name = tmp
            dbError += 1 # 2

            name = name[90:435, 2:240]
            if C == 0:
                average = name.mean()
                C = int((300 - average) * 1.35)
                UpdateThreshold(OMR_MST_CD, BSTOR_CD, C)
            dbError += 1 # 3

            # 수험번호
            codeRegion = code[90:413, 2:202]
            codeRegion = cv2.GaussianBlur(codeRegion, ksize=(5, 5), sigmaX=0)
            codeRegion = cv2.adaptiveThreshold(codeRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, codeRegion = cv2.threshold(codeRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1  # 4

            # 생년월일
            birthdayRegion = birthday[75:245, 12:168]
            birthdayRegion = cv2.GaussianBlur(birthdayRegion, ksize=(5, 5), sigmaX=0)
            birthdayRegion = cv2.adaptiveThreshold(birthdayRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, birthdayRegion = cv2.threshold(birthdayRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1  # 5

            # 성별
            sexRegion = sex[40:98, 42:60]
            sexRegion = cv2.GaussianBlur(sexRegion, ksize=(5, 5), sigmaX=0)
            sexRegion = cv2.adaptiveThreshold(sexRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, sexRegion = cv2.threshold(sexRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1  # 6

            # 이름
            nameRegion = name
            nameRegion = cv2.GaussianBlur(nameRegion, ksize=(5, 5), sigmaX=0)
            nameRegion = cv2.adaptiveThreshold(nameRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, C)
            # _, nameRegion = cv2.threshold(nameRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1  # 7

            # 답안
            ans20Region = answer20[35:, 36:]
            ans20Region = cv2.GaussianBlur(ans20Region, ksize=(5, 5), sigmaX=0)
            ans20Region = cv2.adaptiveThreshold(ans20Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans20Region = cv2.threshold(ans20Region, C, 255, cv2.THRESH_BINARY)
            dbError += 1  # 8

            """
            인식 시작
            """            
            cropRecog = crop.CropRecognition(gray, base_dir, OMR_MST_CD, name0Model, name1Model, idx0Model,
                                             case1Model, answerModel, sexModel, longAnswerModel, select02Model, select09Model, birthdayModel)
            # 수험번호 인식
            studentCode, codeAccuracy = cropRecog.mapCode(codeRegion)
            # 리스트 형식의 수험번호를 하나의 string형식으로 변환
            stdCode = "".join([str(_) for _ in studentCode])
            dbError += 1  # 9
            
            # 생년월일 인식
            studentBday, bdayAccuracy = cropRecog.mapBday(birthdayRegion)
            # 리스트 형식의 생년월일을 하나의 string형식으로 변환
            stdBday = "".join([str(_) for _ in studentBday])
            dbError += 1  # 10

            # 성별 인식
            stdSex, sexAccuracy = cropRecog.mapSex(sexRegion)
            dbError += 1  # 11

            # 이름 인식
            stdName, nameAccuracy = cropRecog.mapName(nameRegion)
            # 이름 인식 예시: ["ㅇ", "ㅕ", "ㄴ", "ㅅ", "ㅓ", "ㅇ", "ㅈ", "ㅜ", "none", "none", "none", "none"]
            # 이 중 none 제거
            while 'none' in stdName:
                stdName.remove('none')
                if stdName == []:
                    stdName = ['-1']
            # none이 제거된 이름 리스트에서 한글 자모 오픈소스를 사용하여 완벽한 이름으로 결합
            stdName = unicode.join_jamos("".join(stdName))
            dbError += 1  # 12

            # 답안 인식
            answerList = []
            answerAccuracy = []
            # 1~20번 마킹 영역 인식하여 저장
            ans, acc = cropRecog.mapAnswer(ans20Region, 20)
            answerList.extend(ans)
            answerAccuracy.append(acc)

            # 1~45번 마킹 중 -1(미인식)인 갯수 저장
            missCnt = answerList.count(-1)
            dbError += 1  # 13

            successFiles += 1

        except:
            LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'CONVERT TEMPLATE ERROR NUM : ' + str(dbError) 
            dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')
            print('pid {0} / dbError {1}'.format(pid, dbError))
            cvt += 1

            successFlag, jsonData = templateHIST.main(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, base_dir, fileNM, src, templates, gray, C)
            TOT.append(jsonData)
            successFiles += successFlag

        if dbError == 14:
            print("파 일 명:", fileNM)
            print("수험번호:", stdCode)
            print("이    름:", stdName)
            print("답    안:", answerList)
            print("미 기 입:", missCnt)

            # json에 담기 위한 자료구조 생성
            jsonData = OrderedDict()

            # 저장 구조에 맞게 key, value 저장
            jsonData["omr_img"] = base_dir + fileNM
            jsonData["stdn_nm"] = stdName
            jsonData["exmn_no"] = stdCode
            jsonData["sex"] = stdSex
            jsonData["bthday"] = stdBday
            jsonData["lsn_cd"] = 34
            jsonData["mark_no"] = answerList
            jsonData["err_cnt"] = missCnt

            # 인식할 이미지가 여러장일 경우 리스트 형식으로 append 하며 저장해야하므로 리스트에 append
            TOT.append(jsonData)

            try:
                # 인식 완료한 파일은 job_done/ 폴더로 이동하고 디렉토리 위치에서 delete
                S3FileDelMove(base_dir, fileNM, 1)
            except:
                LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + 'FAIL DELETE FILE'
                dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')

    return successFiles, TOT, cvt