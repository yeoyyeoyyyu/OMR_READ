import os
import warnings
from multiprocessing import current_process
from collections import OrderedDict

import cv2
import tensorflow as tf

import KICE.common.KICE_PREPROCESS_CONTOUR as pe
import KICE.common.KICE_CROP_RECOG as crop
import KICE.template.KICE_FOREIGN_TEMPLATE as templateForeign
import common.unicode as unicode #한글 자모 결합 오픈소스
from common.OMR_S3_Conn import S3FileDelMove
from common.dbConnect import dblogger, getThreshold, UpdateThreshold, GetLsnCd


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
    lsn_cds = GetLsnCd(OMR_MST_CD)
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
            contours, X, _, _, df = srcImg.setContour(8, gray, contourBinary)
            dbError += 1 # 1

            code = []
            select = []
            name = []
            answer20 = []
            answer30 = []

            for i in range(len(contours)):
                if code == [] and int(df['r'][i] * 100) in range(180, 220):
                    code = cv2.resize(contours[i], dsize=(205, 421), interpolation=cv2.INTER_AREA)

                elif select == [] and int(df['r'][i] * 100) in range(330, 360):
                    select = cv2.resize(contours[i], dsize=(104, 357), interpolation=cv2.INTER_AREA)

                elif name == [] and int(df['r'][i] * 100) in range(170, 200):
                    name = cv2.resize(contours[i], dsize=(245, 452), interpolation=cv2.INTER_AREA)

                elif answer20 == [] and int(df['r'][i] * 100) in range(460, 510):
                    answer20 = cv2.resize(contours[i], dsize=(145, 709), interpolation=cv2.INTER_AREA)

                elif answer30 == [] and int(df['r'][i] * 100) in range(230, 280):
                    answer30 = cv2.resize(contours[i], dsize=(145, 373), interpolation=cv2.INTER_AREA)
            dbError += 1 # 2

            name = name[90:442,2:242]
            if C == 0:
                average = name.mean()
                C = int((300 - average) * 1.35)
                UpdateThreshold(OMR_MST_CD, BSTOR_CD, C)
            dbError += 1 # 3

            # 수험번호
            codeRegion = code[90:419, 2:202]
            codeRegion = cv2.GaussianBlur(codeRegion, ksize=(5, 5), sigmaX=0)
            codeRegion = cv2.adaptiveThreshold(codeRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, codeRegion = cv2.threshold(codeRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1  # 4

            selectRegion = select[52:354,83:101]
            selectRegion = cv2.GaussianBlur(selectRegion, ksize=(5, 5), sigmaX=0)
            selectRegion = cv2.adaptiveThreshold(selectRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, selectRegion = cv2.threshold(selectRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 5

            # 이름
            nameRegion = name
            nameRegion = cv2.GaussianBlur(nameRegion, ksize=(5, 5), sigmaX=0)
            nameRegion = cv2.adaptiveThreshold(nameRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, C)
            # _, nameRegion = cv2.threshold(nameRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1  # 6

            # 답안
            ans20Region = answer20[36:705,36:137]
            ans20Region = cv2.GaussianBlur(ans20Region, ksize=(5, 5), sigmaX=0)
            ans20Region = cv2.adaptiveThreshold(ans20Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans20Region = cv2.threshold(ans20Region, C, 255, cv2.THRESH_BINARY)
            dbError += 1  # 7

            ans30Region = answer30[36:370,36:137]
            ans30Region = cv2.GaussianBlur(ans30Region, ksize=(5, 5), sigmaX=0)
            ans30Region = cv2.adaptiveThreshold(ans30Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans30Region = cv2.threshold(ans30Region, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 8

            '''
            인식 시작                                          
            call class CropRecognition in ETOOS_CROP_RECOG.py 
            params = models                                  
            '''
            # ETOOS_CROP_RECOG.py의 CropRecognition 클래스 호출. 이 때 반복문 밖에서 로드한 모델을 인자로 넘겨줌
            cropRecog = crop.CropRecognition(gray, base_dir, OMR_MST_CD, name0Model, name1Model, idx0Model,
                                             case1Model, answerModel, sexModel, longAnswerModel, select02Model, select09Model, birthdayModel)
            
            # 수험번호 인식
            studentCode, codeAccuracy = cropRecog.mapCode(codeRegion)
            # 리스트 형식의 수험번호를 하나의 string형식으로 변환
            stdCode = "".join([str(_) for _ in studentCode])
            dbError += 1  # 9

            # 선택과목 인식
            stdSelect, selectAccuracy = cropRecog.mapSelect09(selectRegion)
            if int(stdSelect) not in lsn_cds:
                stdSelect = '-1'
            dbError += 1 # 10

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
            dbError += 1  # 11

            # 답안 인식
            missCnt = 0
            answerList = []
            answerAccuracy = []

            ans, acc = cropRecog.mapAnswer(ans20Region, 20)
            answerList.extend(ans)
            answerAccuracy.append(acc)

            ans, acc = cropRecog.mapAnswer(ans30Region, 10)
            answerList.extend(ans)
            answerAccuracy.append(acc)

            missCnt = answerList.count(-1)
            dbError += 1 # 12

            # 1~45번 마킹 중 -1(미인식)인 갯수 저장
            missCnt = answerList.count(-1)
            dbError += 1  # 13

            successFiles += 1

        except:
            LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'CONVERT TEMPLATE ERROR NUM : ' + str(dbError) 
            dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')
            print('pid {0} / dbError {1}'.format(pid, dbError))
            cvt += 1

            successFlag, jsonData = templateForeign.main(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, base_dir, fileNM, src, templates, gray, C)
            TOT.append(jsonData)
            successFiles += successFlag

        if dbError == 14:
            print("파 일 명:", fileNM)
            print("수험번호:", stdCode)
            print("이    름:", stdName)
            print("선택과목:", stdSelect)
            print("답    안:", answerList)
            print("미 기 입:", missCnt)

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

            # 인식할 이미지가 여러장일 경우 리스트 형식으로 append 하며 저장해야하므로 리스트에 append
            TOT.append(jsonData)

            try:
                # 인식 완료한 파일은 job_done/ 폴더로 이동하고 디렉토리 위치에서 delete
                S3FileDelMove(base_dir, fileNM, 1)
            except:
                LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + 'FAIL DELETE FILE'
                dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')

    return successFiles, TOT, cvt