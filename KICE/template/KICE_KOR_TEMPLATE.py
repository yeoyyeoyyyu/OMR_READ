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

    # DB에 남길 에러 로그 메시지 리스트
    dbErrorList = ['FAIL LOAD IMG', 'FAIL TEMPLATE ROI', 'FAIL CHECK IMG', 'FAIL ROI MASKING', \
                   'FAIL CROP NAME REGION', 'FAIL CROP CODE REGION', 'FAIL CROP ANS REGION', 'FAIL CROP SELECT TYPE REGION', \
                   'FAIL RECOG NAME', 'FAIL RECOG CODE', 'FAIL RECOG ANS', 'FAIL RECOG SELECT TYPE']

    successFiles = 0
    dbError = 0

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

        # checkIMG = check.checkImage(gray)
        # checkTop = checkIMG.check_top()
        # checkLeft = checkIMG.check_left(regionXY[2])
        #
        # if checkTop == False:
        #     raise Exception('FAIL CHECK TOP')
        # if checkLeft == False:
        #     raise Exception('FAIL CHECK LEFT')
        dbError += 1  # 2

        regionXYWH = []

        # 검색된 템플릿 좌표로부터 읽어야 할 영역 좌표 계산하여 리스트에 append
        regionXYWH.append((regionXY[2][0], regionXY[2][1] + 88, 201, 324))  # 1 수험번호
        regionXYWH.append((regionXY[3][0], regionXY[3][1] + 90, 241, 350))  # 2 이름
        regionXYWH.append((regionXY[4][0] + 43, regionXY[4][1] + 69, 28, 82))  # 3 선택과목
        regionXYWH.append((regionXY[0][0] + 36, regionXY[0][1] + 38, 110, 666))  # 4 답안20    (기준: template34)
        regionXYWH.append((regionXY[0][0] + 177, regionXY[0][1] + 38, 110, 463))  # 5 답안34 (기준: template34)
        regionXYWH.append((regionXY[1][0] + 36, regionXY[1][1] + 39, 110, 363))  # 6 답안45  (기준: template45)
        dbError += 1  # 3

        """
        인식 시작
        """
        # ETOOS_CROP_RECOG.py의 CropRecognition 클래스 호출. 이 때 반복문 밖에서 로드한 모델을 인자로 넘겨줌
        cropRecog = crop.CropRecognition(binary, base_dir, OMR_MST_CD, name0Model, name1Model, idx0Model,
                                            case1Model, answerModel, sexModel, longAnswerModel, select02Model, select09Model, birthdayModel)
        # 이름영역 좌표를 인자로 넘겨 이름영역 이미지를 nameRegion에 저장
        nameRegion = cropRecog.cropRegion(binary, regionXYWH[1], 0, 0)
        dbError += 1  # 4

        # 수험번호 영역 좌표를 인자로 넘겨 수험번호 영역 이미지를 codeRegion에 저장
        codeRegion = cropRecog.cropRegion(binary, regionXYWH[0], 0, 0)
        dbError += 1  # 5

        # 답안 영역 좌표를 인자로 넘겨 답안 영역 이미지를 ans20Region 등에 저장
        # ans20Region: 1~20번, ans40Region: 21~34번, ans45Region: 35~45번 영역
        ans20Region = cropRecog.cropRegion(binary, regionXYWH[3], 0, 0)
        ans34Region = cropRecog.cropRegion(binary, regionXYWH[4], 0, 0)
        ans45Region = cropRecog.cropRegion(binary, regionXYWH[5], 0, 0)
        dbError += 1  # 6

        # 2021.01.29 선택과목 영역 좌표를 인자로 넘겨 성별 영역 이미지를 selectRegion 저장
        selectRegion = cropRecog.cropRegion(binary, regionXYWH[2], 0, 0)
        dbError += 1  # 7

        '''
        1. 이름 인식
        '''
        # 이름 영역 인식하여 인식한 이름과 정확도 저장
        stdName, nameAccuracy = cropRecog.mapName(nameRegion)
        # 이름 인식 예시: ["ㅇ", "ㅕ", "ㄴ", "ㅅ", "ㅓ", "ㅇ", "ㅈ", "ㅜ", "none", "none", "none", "none"]
        # 이 중 none 제거
        while 'none' in stdName:
            stdName.remove('none')
        if stdName == []:
            stdName = ['-1']
        # none이 제거된 이름 리스트에서 한글 자모 오픈소스를 사용하여 완벽한 이름으로 결합
        stdName = unicode.join_jamos("".join(stdName))
        dbError += 1  # 8

        '''
        2. 수험번호 인식
        '''
        # 수험번호 영역 인식하여 인식한 수험번호와 정확도 저장
        studentCode, codeAccuracy = cropRecog.mapCode(codeRegion)
        # 리스트 형식의 수험번호를 하나의 string형식으로 변환
        stdCode = "".join([str(_) for _ in studentCode])
        dbError += 1  # 9

        '''
        3. 답안 인식
        '''
        # 45개가 되는 마킹 답안을 담을 리스트와 정확도를 담을 리스트 생성
        answerList = []
        answerAccuracy = []
        # 1~20번 마킹 영역 인식하여 저장
        ans, acc = cropRecog.mapAnswer(ans20Region, 20)
        # 인식한 답안과 정확도를 리스트에 추가(ans와 answerList 모두 list형태라 append 아닌 extend 사용, 정확도는 숫자 하나 이므로 append)
        answerList.extend(ans)
        answerAccuracy.append(acc)
        # print(answerList, answerAccuracy)

        # 21~34번 마킹 영역 인식하여 저장
        ans, acc = cropRecog.mapAnswer(ans34Region, 14)
        answerList.extend(ans)
        answerAccuracy.append(acc)

        # 35~45번 마킹 영역 인식하여 저장
        ans, acc = cropRecog.mapAnswer(ans45Region, 11)
        answerList.extend(ans)
        answerAccuracy.append(acc)

        # 1~45번 마킹 중 -1(미인식)인 갯수 저장
        missCnt = answerList.count(-1)
        dbError += 1  # 10

        '''
        4. 선택과목 인식
        '''
        # 선택과목 영역 인식하여 인식한 성별과 정확도 저장
        stdSelect, stdAccuracy = cropRecog.mapSex(selectRegion)

        if stdSelect == '남':
            stdSelect = 39
        elif stdSelect == '여':
            stdSelect = 40
        else:
            stdSelect = -1

        dbError += 1  # 11

        successFiles += 1

    except:
        LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + dbErrorList[dbError]
        dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')
        print(LOG_MSG)
        dbReupld(fileNM)

    if dbError == 12:
        # 확인용 프린트
        print("stdName: ", stdName, nameAccuracy)
        print("stdCode: ", stdCode, codeAccuracy)
        print("selectOption: ", stdSelect, stdAccuracy)
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
        jsonData["lsn_cd"] = int(-1) # 2021.01.29 int(-1)로 수정
        jsonData["mark_no"] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, \
                               -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, \
                               -1, -1, -1, -1, -1]
        
        try:
            # job_fail/ 로 보내고 지움
            S3FileDelMove(base_dir, fileNM, 0)
        except:
            LOG_MSG = 'FILE NAME : ' + str(fileNM) + ' , ' + 'ERROR : ' + 'FAIL DELETE FILE'
            dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, LOG_MSG, None, 'failure')

    return successFiles, jsonData