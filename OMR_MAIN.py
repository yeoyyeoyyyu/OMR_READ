'''
OMR_MAIN.py
-TOS에 의해 호출되는 파일
-파라미터를 전달 받아 S3에서 이미지 파일명을 리스트에 저장
-리스트에 저장된 이미지 파일명을 프로세스 수에 맞게 2차원 리스트로 분리
-TOS에서 전달 받은 파라미터와 분리된 2차원 리스트를 각 프로세스에 전달
'''
import math
import time
import datetime
import multiprocessing as mp
import argparse

from OMR_MANAGER import OmrManager
from common.OMR_S3_Conn import exploreS3, S3RetUpload
from common.dbConnect import dblogger

class ParamError(Exception):
    def __str__(self):
        return "parameters error"

def check_dir(EXAM_CD, CMPM_CD, BSTOR_CD, OMR_MST_CD):
    """
    받아온 인자가 잘 들어온지 확인하여 ERROR 메시지 출력
    :param EXAM_CD    :시험 코드
    :param CMPM_CD    :회사 코드
    :param BSTOR_CD   :지점 코드
    :param OMR_MST_CD :OMR마스터 코드
    :param CLUSTER    :s3 BUCKET 분류
    """
    if EXAM_CD == 'fail' or  CMPM_CD == 'fail' or BSTOR_CD == 'fail' or OMR_MST_CD == 'fail':
        dblogger(EXAM_CD, CMPM_CD, BSTOR_CD, OMR_MST_CD, '인자 호출되지 않음', None, 'failure')
        raise ParamError()

if __name__ == '__main__':
    # 인식 소요시간 확인용
    process_start = time.time()
    # 시작 시간을 담는 용도
    start = datetime.datetime.now()
    tt = str(start.year) + str(start.month) + str(start.day) + "_" + str(start.hour) + str(start.minute)
    
    obj_parser = argparse.ArgumentParser(description='Get path for S3 jpg files and process OMR.')
    obj_parser.add_argument('path1', nargs='?', default='fail', type=str, help='시험') #omrtest
    obj_parser.add_argument('path2', nargs='?', default='fail', type=str, help='회사') #1
    obj_parser.add_argument('path3', nargs='?', default='fail', type=str, help='지점') #1
    obj_parser.add_argument('path4', nargs='?', default='fail', type=str, help='OMR마스터코드') #math
    args = obj_parser.parse_args()

    EXAM_CD = args.path1
    CMPN_CD = args.path2
    BSTOR_CD = args.path3
    OMR_MST_CD = args.path4
    
    # 받아온 파라미터들로 S3 접근 경로 생성
    base_dir = 'aido/omr/' + EXAM_CD + "/" + CMPN_CD + "/" + BSTOR_CD + "/" + OMR_MST_CD + "/"
    
    # 받아온 인자가 유효한지 판단
    check_dir(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD)
    dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, None, 'begin', None)

    # S3 bucket에서 업로드된 이미지 파일명과 탬플릿 파일명을 저장
    files, templates = exploreS3(OMR_MST_CD, base_dir)

    # 최대 프로세스 수와 파일명을 저장한 리스트의 길이를 비교하여 멀티프로세스 분할 수 정함
    max_process = mp.cpu_count() - 1
    file_count = len(files)
    print('len(flies), max_process :', file_count, max_process)    
    
    if file_count < max_process:
        max_process, file_count = file_count, max_process
        
    # 1차원의 리스트를 정해진 분할 수에 따라 2차원화
    n = math.ceil(len(files) / max_process)
    result = [files[i: i + n] for i in range(0, len(files), n)]        
    print(n, result)  
    
    # 프로세스들을 저장할 리스트
    procs = []      
    try:
        # 프로세스 관리할 객체 생성
        manager = mp.Manager()
        # 각 프로세스의 실행결과를 저장할 리스트
        result_list = manager.list()
        successFiles = manager.list()
        cvt_templates = manager.list()
        # 각 프로세스에 파라미터와 이미지명 전달
        for idx in range(max_process):
            proc = mp.Process(target=OmrManager,
                                args=(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, base_dir, result[idx], templates, result_list, successFiles, cvt_templates))
            procs.append(proc)
            proc.start()
            # 파일 리스트의 길이가 프로세스 분할 수보다 적을 때 반복문 조기 종료
            if n * (idx + 1) >= file_count:
                break
        # 모든 프로세스가 종료될 때까지 기다림
        for proc in procs:
            proc.join()
    except Exception as e:
        print(e)

    # 저장된 결과(Manager객체)를 리스트에 저장
    print('successfiles :', sum(successFiles))
    print('convert templates :', sum(cvt_templates))
    result_TOT = list(result_list)
    # print(type(result_TOT))
    # print(result_TOT)
    # 결과를 S3에 json파일로 업로드
    try:
        S3RetUpload(base_dir, "TOT_" + tt + ".json", result_TOT)
        S3RetUpload(base_dir, "TOT.json", result_TOT)
        print("json upload to S3")
    except:
        log_message = "S3 업로드 실패"
        print("fail json upload to S3")
        dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, log_message, None, "failure")

    log_message = "SUCCESS FILES:" + str(sum(successFiles)) + ", " + "COVERT TEMPLATE:" + str(sum(cvt_templates))
    dblogger(EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, log_message, None, "failure")
    end = time.time() - process_start
    print(end)