import os
import warnings
from multiprocessing import current_process
from collections import OrderedDict

import cv2
import tensorflow as tf

import KICE.common.KICE_PREPROCESS_CONTOUR as pe
import KICE.common.KICE_CROP_RECOG as crop
import KICE.template.KICE_TAM_TEMPLATE as templateTAM
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
            name = []
            firstSbj = []
            secondSbj = []

            for i in range(len(contours)):
                if code == [] and int(df['r'][i] * 100) in range(187, 230):
                    code = cv2.resize(contours[i], dsize=(204, 422), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('crop_images/code.jpg', code)

                elif name == [] and int(df['r'][i] * 100) in range(168, 208):
                    name = cv2.resize(contours[i], dsize=(243, 457), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('crop_images/name.jpg', name)

                elif firstSbj == [] and int(df['r'][i] * 100) in range(395, 460):
                    # print(df['r'][i])
                    firstSbj = cv2.resize(contours[i], dsize=(171, 701), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('crop_images/1st.jpg', firstSbj)

                elif secondSbj == [] and int(df['r'][i] * 100) in range(395, 460):
                    secondSbj = cv2.resize(contours[i], dsize=(171, 701), interpolation=cv2.INTER_AREA)
                    cv2.imwrite('crop_images/2nd.jpg', secondSbj)
            dbError += 1 # 2

            name = name[90:443, 2:240]
            if C == 0:
                average = name.mean()
                C = int((300 - average) * 1.35)
                UpdateThreshold(OMR_MST_CD, BSTOR_CD, C)
            dbError += 1 # 3

            # 수험번호
            codeRegion = code[90:415, 2:202]
            codeRegion = cv2.GaussianBlur(codeRegion, ksize=(5, 5), sigmaX=0)
            codeRegion = cv2.adaptiveThreshold(codeRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, codeRegion = cv2.threshold(codeRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 4

            # 이름
            nameRegion = name
            nameRegion = cv2.GaussianBlur(nameRegion, ksize=(5, 5), sigmaX=0)
            nameRegion = cv2.adaptiveThreshold(nameRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, C)
            # _, nameRegion = cv2.threshold(nameRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 5

            # 1과목 선택과목
            firstSelect = firstSbj[202:538, 2:42]
            firstSelect = cv2.GaussianBlur(firstSelect, ksize=(5, 5), sigmaX=0)
            firstSelect = cv2.adaptiveThreshold(firstSelect, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, firstSelect = cv2.threshold(firstSelect, C, 255, cv2.THRESH_BINARY)

            # 1과목 답안
            firstAnsRegion = firstSbj[38:, 68:]
            firstAnsRegion = cv2.GaussianBlur(firstAnsRegion, ksize=(5, 5), sigmaX=0)
            firstAnsRegion = cv2.adaptiveThreshold(firstAnsRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # cv2.imwrite('1st_select.jpg', firstAnsRegion)
            # _, firstAnsRegion = cv2.threshold(firstAnsRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 6

            # 2과목 선택과목
            secondSelect = secondSbj[202:538, 2:42]
            secondSelect = cv2.GaussianBlur(secondSelect, ksize=(5, 5), sigmaX=0)
            secondSelect = cv2.adaptiveThreshold(secondSelect, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # _, secondSelect = cv2.threshold(secondSelect, C, 255, cv2.THRESH_BINARY)

            # 2과목 답안
            secondAnsRegion = secondSbj[38:, 68:]
            secondAnsRegion = cv2.GaussianBlur(secondAnsRegion, ksize=(5, 5), sigmaX=0)
            secondAnsRegion = cv2.adaptiveThreshold(secondAnsRegion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, C)
            # cv2.imwrite('2nd_select.jpg', secondAnsRegion)
            # _, secondAnsRegion = cv2.threshold(secondAnsRegion, C, 255, cv2.THRESH_BINARY)
            dbError += 1 # 7


            '''
            인식 시작                                          
            call class CropRecognition in KICE_CROP_RECOG.py 
            params = models                                  
            '''
            # ETOOS_CROP_RECOG.py의 CropRecognition 클래스 호출. 이 때 반복문 밖에서 로드한 모델을 인자로 넘겨줌
            cropRecog = crop.CropRecognition(gray, base_dir, OMR_MST_CD, name0Model, name1Model, idx0Model,
                                             case1Model, answerModel, sexModel, longAnswerModel, select02Model, select09Model, birthdayModel)

            # 수험번호 인식
            studentCode = []
            studentCode, codeAccuracy = cropRecog.mapCode(codeRegion)
            stdCode = "".join([str(_) for _ in studentCode])
            dbError += 1  # 8

            # 이름 인식
            stdName = []
            stdName, nameAccuracy = cropRecog.mapName(nameRegion)
            while 'none' in stdName:
                stdName.remove('none')
            if stdName == []:
                stdName = ['-1']
            stdName = unicode.join_jamos("".join(stdName))
            dbError += 1  # 9

            # 1과목 선택과목 인식
            firstSelect, acc = cropRecog.mapSelect(firstSelect)
            if int(firstSelect) not in lsn_cds:
                firstSelect = '-1'

            # 1과목 답안 인식
            firstAnswerList = []
            firstAnswerAccuracy = []
            # 1~20번 마킹 영역 인식하여 저장
            ans, acc = cropRecog.mapAnswer(firstAnsRegion, 20)
            firstAnswerList.extend(ans)
            firstAnswerAccuracy.append(acc)
            firstMissCnt = firstAnswerList.count(-1)
            dbError += 1 # 10

            # 2과목 선택과목 인식
            secondSelect, acc = cropRecog.mapSelect(secondSelect)
            if int(secondSelect) not in lsn_cds:
                secondSelect = '-2'

            # 2과목 답안 인식
            secondAnswerList = []
            secondAnswerAccuracy = []
            # 1~20번 마킹 영역 인식하여 저장
            ans, acc = cropRecog.mapAnswer(secondAnsRegion, 20)
            secondAnswerList.extend(ans)
            secondAnswerAccuracy.append(acc)
            secondMissCnt = secondAnswerList.count(-1)
            dbError += 1  # 11

            successFiles += 1

        except:
            LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'CONVERT TEMPLATE ERROR NUM : ' + str(dbError) 
            dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')
            print('pid {0} / dbError {1}'.format(pid, dbError))
            cvt += 1

            successFlag, jsonData = templateTAM.main(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, base_dir, fileNM, src, templates, gray, C)
            TOT.append(jsonData)
            successFiles += successFlag

        if dbError == 12:
            print("파 일 명:", fileNM)
            print("수험번호:", stdCode)
            print("이    름:", stdName)
            print("제1 선택과목 : ", firstSelect, firstAnswerList)
            print("제1 선택과목 미기입 : ", firstMissCnt)
            print("제2 선택과목 : ", secondSelect, secondAnswerList)
            print("제2 선택과목 미기입 : ", secondMissCnt)

            # json에 담기 위한 자료구조 생성
            jsonData = OrderedDict()
            tam1 = OrderedDict()
            tam2 = OrderedDict()

            lsn_list = []

            tam1["lsn_cd"] = firstSelect
            tam1["mark_no"] = firstAnswerList
            tam1["err_cnt"] = int(firstMissCnt)
            tam1["lsn_seq"] = 1

            tam2["lsn_cd"] = secondSelect
            tam2["mark_no"] = secondAnswerList
            tam2["err_cnt"] = int(secondMissCnt)
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

            # 인식할 이미지가 여러장일 경우 리스트 형식으로 append 하며 저장해야하므로 리스트에 append
            TOT.append(jsonData)

            try:
                # 인식 완료한 파일은 job_done/ 폴더로 이동하고 디렉토리 위치에서 delete
                S3FileDelMove(base_dir, fileNM, 1)
            except:
                LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + 'FAIL DELETE FILE'
                dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')

    return successFiles, TOT, cvt