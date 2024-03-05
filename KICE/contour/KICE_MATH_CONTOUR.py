import os
import warnings
from multiprocessing import current_process
from collections import OrderedDict

import cv2
import tensorflow as tf

import KICE.common.KICE_PREPROCESS_CONTOUR as pe
import KICE.common.KICE_CROP_RECOG as crop
import KICE.template.KICE_MATH_TEMPLATE as templateMATH
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
    # threshold_list = []
    for fileNM in fileNMs:
        dbError = 0
        cnt += 1
        print('\n{0} / {1}, pid : {2}'.format(cnt, len(fileNMs), pid))
        dbError = 0
        try:
            # call class PreprocessingImage in ETOOS_PREPROCESS_IMG.py
            srcImg = pe.PreprocessingImage(base_dir, OMR_MST_CD)
            src, gray, contourBinary = srcImg.loadImage(fileNM)        
            dbError += 1 # 0

            # detect contour
            contours, _, Y, _, df = srcImg.setContour(20, gray, contourBinary)
            dbError += 1 # 1

            code = []
            select = []
            name = []
            # 공통과목 객관식
            answer15 = []
            # 공통과목 주관식
            answer16 = []
            answer17 = []
            answer18 = []
            answer19 = []
            answer20 = []
            answer21 = []
            answer22 = []
            # 선택과목 객관식
            answer2328 = []
            # 선택과목 주관식
            answer29 = []
            answer30 = []

            idx = 0
            for i in range(len(contours)):
                if code == [] and int(df['r'][i] * 100) in range(185, 225):
                    code = cv2.resize(contours[i], dsize=(204, 419), interpolation=cv2.INTER_AREA)
                
                elif select == [] and int(df['r'][i] * 100) in range(340, 380):
                    select = cv2.resize(contours[i], dsize=(61, 220), interpolation=cv2.INTER_AREA)

                elif name == [] and int(df['r'][i] * 100) in range(163, 203):
                    name = cv2.resize(contours[i], dsize=(243, 445), interpolation=cv2.INTER_AREA)

                elif i > 1 and answer15 == [] and int(df['r'][i] * 100) in range(86, 126):
                    answer15 = cv2.resize(contours[i], dsize=(283, 301), interpolation=cv2.INTER_AREA)

                if name != [] and answer15 != []:
                    break

            idx = i + 1
            tmp = 0
            h, _ = gray.shape

            for j in range(idx, len(contours)):
                if tmp == 0 and Y[j] < h // 2:
                    tmp += 1

                elif answer16 == [] and Y[j] < h // 2:
                    answer16 = cv2.resize(contours[j], dsize=(64, 369), interpolation=cv2.INTER_AREA)

                elif answer17 == [] and Y[j] < h // 2:
                    answer17 = cv2.resize(contours[j], dsize=(64, 369), interpolation=cv2.INTER_AREA)

                elif answer18 == [] and Y[j] < h // 2:
                    answer18 = cv2.resize(contours[j], dsize=(64, 369), interpolation=cv2.INTER_AREA)

                elif answer19 == [] and Y[j] < h // 2:
                    answer19 = cv2.resize(contours[j], dsize=(64, 369), interpolation=cv2.INTER_AREA)

                elif answer20 == [] and Y[j] < h // 2:
                    answer20 = cv2.resize(contours[j], dsize=(64, 369), interpolation=cv2.INTER_AREA)

                elif answer21 == [] and Y[j] > h // 2:
                    answer21 = cv2.resize(contours[j], dsize=(64, 369), interpolation=cv2.INTER_AREA)

                elif answer22 == [] and Y[j] > h // 2:
                    answer22 = cv2.resize(contours[j], dsize=(64, 369), interpolation=cv2.INTER_AREA)

                elif answer2328 == [] and Y[j] > h // 2 and int(df['r'][j] * 100) in range(140, 180):
                    answer2328 = cv2.resize(contours[j], dsize=(149, 237), interpolation=cv2.INTER_AREA)

                elif answer29 == [] and Y[j] > h // 2:
                    answer29 = cv2.resize(contours[j], dsize=(64, 369), interpolation=cv2.INTER_AREA)

                elif answer30 == [] and Y[j] > h // 2:
                    answer30 = cv2.resize(contours[j], dsize=(64, 369), interpolation=cv2.INTER_AREA)
            dbError += 1 # 2

            name = name[85:435, 2:240]
            if C == 0:
                average = name.mean()
                C = int((300 - average) * 1.35)
                UpdateThreshold(OMR_MST_CD, BSTOR_CD, C)
            dbError += 1 # 3

            # 수험번호
            codeRegion = code[90:415, 2:200]
            codeRegion = cv2.GaussianBlur(codeRegion, ksize=(5, 5), sigmaX=0)
            codeRegion = cv2.adaptiveThreshold(codeRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, codeRegion = cv2.threshold(codeRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 4

            # 선택과목
            selectRegion = select[70:200, 40:60]
            selectRegion = cv2.GaussianBlur(selectRegion, ksize=(5, 5), sigmaX=0)
            selectRegion = cv2.adaptiveThreshold(selectRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, selectRegion = cv2.threshold(selectRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 5

            # 이름
            nameRegion = name
            nameRegion = cv2.GaussianBlur(nameRegion, ksize=(5, 5), sigmaX=0)
            nameRegion = cv2.adaptiveThreshold(nameRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, C)
            # _, nameRegion = cv2.threshold(nameRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 6

            # 공통과목 객관식 답안
            # ans08Region: 1~8번, ans15Region: 9~15번
            h, w = answer15.shape
            ans08Region = answer15[35:, 42:w//2]
            ans08Region = cv2.GaussianBlur(ans08Region, ksize=(5, 5), sigmaX=0)
            ans08Region = cv2.adaptiveThreshold(ans08Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans08Region = cv2.threshold(ans08Region, C, 255, cv2.THRESH_BINARY)

            ans15Region = answer15[35:270, w//2 + 40:w//2 + 139]
            ans15Region = cv2.GaussianBlur(ans15Region, ksize=(5, 5), sigmaX=0)
            ans15Region = cv2.adaptiveThreshold(ans15Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans15Region = cv2.threshold(ans15Region, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 7

            # 공통과목 주관식 답안
            ans16Region = answer16[35:370, 2:]
            ans16Region = cv2.GaussianBlur(ans16Region, ksize=(5, 5), sigmaX=0)
            ans16Region = cv2.adaptiveThreshold(ans16Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans16Region = cv2.threshold(ans16Region, C, 255, cv2.THRESH_BINARY)

            ans17Region = answer17[35:370, 2:]
            ans17Region = cv2.GaussianBlur(ans17Region, ksize=(5, 5), sigmaX=0)
            ans17Region = cv2.adaptiveThreshold(ans17Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans17Region = cv2.threshold(ans17Region, C, 255, cv2.THRESH_BINARY)

            ans18Region = answer18[35:370, 2:]
            ans18Region = cv2.GaussianBlur(ans18Region, ksize=(5, 5), sigmaX=0)
            ans18Region = cv2.adaptiveThreshold(ans18Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans18Region = cv2.threshold(ans18Region, C, 255, cv2.THRESH_BINARY)

            ans19Region = answer19[35:370, 2:]
            ans19Region = cv2.GaussianBlur(ans19Region, ksize=(5, 5), sigmaX=0)
            ans19Region = cv2.adaptiveThreshold(ans19Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans19Region = cv2.threshold(ans19Region, C, 255, cv2.THRESH_BINARY)

            ans20Region = answer20[35:370, 2:]
            ans20Region = cv2.GaussianBlur(ans20Region, ksize=(5, 5), sigmaX=0)
            ans20Region = cv2.adaptiveThreshold(ans20Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans20Region = cv2.threshold(ans20Region, C, 255, cv2.THRESH_BINARY)

            ans21Region = answer21[35:370, 2:]
            ans21Region = cv2.GaussianBlur(ans21Region, ksize=(5, 5), sigmaX=0)
            ans21Region = cv2.adaptiveThreshold(ans21Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans21Region = cv2.threshold(ans21Region, C, 255, cv2.THRESH_BINARY)

            ans22Region = answer22[35:370, 2:]
            ans22Region = cv2.GaussianBlur(ans22Region, ksize=(5, 5), sigmaX=0)
            ans22Region = cv2.adaptiveThreshold(ans22Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans22Region = cv2.threshold(ans22Region, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 8

            # 선택과목 객관식 답안
            ans2328Region = answer2328[38:, 41:143]
            ans2328Region = cv2.GaussianBlur(ans2328Region, ksize=(5, 5), sigmaX=0)
            ans2328Region = cv2.adaptiveThreshold(ans2328Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans2328Region = cv2.threshold(ans2328Region, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 9

            # 선택과목 주관식 답안
            ans29Region = answer29[35:370, 2:]
            ans29Region = cv2.GaussianBlur(ans29Region, ksize=(5, 5), sigmaX=0)
            ans29Region = cv2.adaptiveThreshold(ans29Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans29Region = cv2.threshold(ans29Region, C, 255, cv2.THRESH_BINARY)

            ans30Region = answer30[35:370, 2:]
            ans30Region = cv2.GaussianBlur(ans30Region, ksize=(5, 5), sigmaX=0)
            ans30Region = cv2.adaptiveThreshold(ans30Region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, ans30Region = cv2.threshold(ans30Region, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 10

            '''
            인식 시작                                          
            call class CropRecognition in KICE_CROP_RECOG.py 
            params = models                                  
            '''
            cropRecog = crop.CropRecognition(gray, base_dir, OMR_MST_CD, name0Model, name1Model, idx0Model,
                                             case1Model, answerModel, sexModel, longAnswerModel, select02Model, select09Model, birthdayModel)
            # 수험번호 인식
            studentCode, codeAccuracy = cropRecog.mapCode(codeRegion)
            # 리스트 형식의 수험번호를 하나의 string형식으로 변환
            stdCode = "".join([str(_) for _ in studentCode])
            dbError += 1  # 11

            # 선택과목 인식
            stdSelect, stdAccuracy = cropRecog.mapSelect02(selectRegion)
            dbError += 1  # 12

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
            dbError += 1  # 13

            # 답안 인식
            answerList = []
            answerAccuracy = []

            # 공통과목 객관식
            # 1~9번 마킹 영역 인식하여 저장
            ans, acc = cropRecog.mapShortAnswer(ans08Region, 8)
            answerList.extend(ans)
            answerAccuracy.append(acc)
            # 10~15번
            ans, acc = cropRecog.mapShortAnswer(ans15Region, 7)
            answerList.extend(ans)
            answerAccuracy.append(acc)
            dbError += 1 # 14

            # 공통과목 주관식
            # 16번
            ans, acc = cropRecog.mapLongAnswer(ans16Region, 1)
            contours, _ = cv2.findContours(ans16Region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            answerList.extend(ans)
            answerAccuracy.append(acc)
            # 17번
            ans, acc = cropRecog.mapLongAnswer(ans17Region, 1)
            contours, _ = cv2.findContours(ans17Region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ans17Region = cv2.cvtColor(ans17Region, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(ans17Region, contours, -1, (0, 0, 255), 1)
            if len(contours) > 4:
                ans[0] = -1
            answerList.extend(ans)
            answerAccuracy.append(acc)
            # 18번
            ans, acc = cropRecog.mapLongAnswer(ans18Region, 1)
            contours, _ = cv2.findContours(ans18Region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ans18Region = cv2.cvtColor(ans18Region, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(ans18Region, contours, -1, (0, 0, 255), 1)
            if len(contours) > 4:
                ans[0] = -1
            answerList.extend(ans)
            answerAccuracy.append(acc)
            # 19번
            ans, acc = cropRecog.mapLongAnswer(ans19Region, 1)
            contours, _ = cv2.findContours(ans19Region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ans19Region = cv2.cvtColor(ans19Region, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(ans19Region, contours, -1, (0, 0, 255), 1)
            if len(contours) > 4:
                ans[0] = -1
            answerList.extend(ans)
            answerAccuracy.append(acc)
            # 20번
            ans, acc = cropRecog.mapLongAnswer(ans20Region, 1)
            contours, _ = cv2.findContours(ans20Region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ans20Region = cv2.cvtColor(ans20Region, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(ans20Region, contours, -1, (0, 0, 255), 1)
            if len(contours) > 4:
                ans[0] = -1
            answerList.extend(ans)
            answerAccuracy.append(acc)
            # 21번
            ans, acc = cropRecog.mapLongAnswer(ans21Region, 1)
            contours, _ = cv2.findContours(ans21Region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ans21Region = cv2.cvtColor(ans21Region, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(ans21Region, contours, -1, (0, 0, 255), 1)
            if len(contours) > 4:
                ans[0] = -1
            answerList.extend(ans)
            answerAccuracy.append(acc)
            # 22번
            ans, acc = cropRecog.mapLongAnswer(ans22Region, 1)
            contours, _ = cv2.findContours(ans22Region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ans22Region = cv2.cvtColor(ans22Region, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(ans22Region, contours, -1, (0, 0, 255), 1)
            if len(contours) > 4:
                ans[0] = -1
            answerList.extend(ans)
            answerAccuracy.append(acc)
            dbError += 1  # 15

            # 선택과목 객관식
            ans, acc = cropRecog.mapShortAnswer(ans2328Region, 6)
            answerList.extend(ans)
            answerAccuracy.append(acc)

            # 선택과목 주관식
            # 29번
            ans, acc = cropRecog.mapLongAnswer(ans29Region, 1)
            contours, _ = cv2.findContours(ans29Region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ans29Region = cv2.cvtColor(ans29Region, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(ans29Region, contours, -1, (0, 0, 255), 1)
            if len(contours) > 4:
                ans[0] = -1
            answerList.extend(ans)
            answerAccuracy.append(acc)
            # 30번
            ans, acc = cropRecog.mapLongAnswer(ans30Region, 1)
            contours, _ = cv2.findContours(ans30Region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ans30Region = cv2.cvtColor(ans30Region, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(ans30Region, contours, -1, (0, 0, 255), 1)
            if len(contours) > 4:
                ans[0] = -1
            answerList.extend(ans)
            answerAccuracy.append(acc)

            missCnt = answerList.count(-1)
            dbError += 1  # 16

            successFiles += 1

        except:
            LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'CONVERT TEMPLATE ERROR NUM : ' + str(dbError) 
            dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')
            print('pid {0} / dbError {1}'.format(pid, dbError))
            cvt += 1
            successFlag, jsonData = templateMATH.main(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, base_dir, fileNM, src, templates, gray, C)
            TOT.append(jsonData)
            successFiles += successFlag

        if dbError == 17:
            # 확인용 프린트
            print("파 일 명:", fileNM)
            print("수험번호:", stdCode)
            print("과목코드:", stdSelect)
            print("이    름:", stdName)
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

    print('탬플릿 전환 된 이미지 수 : {}\n'.format(cvt))
    return successFiles, TOT, cvt