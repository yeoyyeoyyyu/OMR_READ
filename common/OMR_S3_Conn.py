"""
'OMR_S3_Conn' PURPOSE
- S3 연결 및 파일 접근과 read, write 관리
"""
import os
import json
import boto3
from configparser import ConfigParser

config = ConfigParser()
config.read('{}/conf.ini'.format(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

# ================================================================================
# s3 connection
# ================================================================================
REGION_NAME = config['S3']['region_name']
AWS_ACCESS_KEY_ID = config['S3']['aws_access_key_id']
AWS_SECRET_ACCESS_KEY = config['S3']['aws_secret_access_key']
BUCKET_NAME = config['S3']['bucket_name']

def ConnS3():
    """
    S3 커넥션
    :return    :s3 커넥션 모듈
    """
    s3 = boto3.resource(
        's3'
        , region_name=REGION_NAME
        , aws_access_key_id=AWS_ACCESS_KEY_ID
        , aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    return s3


def exploreS3(OMR_MST_CD, base_dir):
    '''
    :param OMR_MST_CD:
    :param base_dir: 접근하고자 하는 S3 경로
    :return:
    '''
    # s3 클라이언트 생성
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=REGION_NAME)
    # 가져와야 할 파일 이름과 템플릿 이름을 담을 리스트와 이를 도와주는 리스트 생성
    fileNMs = []
    dest = []
    templateNMs = []

    '''
    1. 인식 대상 파일 이름 저장
    '''
    prefix = base_dir

    # 원하는 bucket과 하위경로에 있는 object list
    obj_list = s3.list_objects(Bucket=BUCKET_NAME, Prefix=prefix)  # dict type

    # object list의 Contents를 가져옴
    contents_lst = obj_list['Contents']

    # Content list 출력
    for content in contents_lst:
        if 'job_done' in content['Key'] or \
                'job_fail' in content['Key'] or \
                'rot_images' in content['Key'] or \
                'check_fail' in content['Key']:
            # break
            continue
        tmp = content['Key']
        dest.append(tmp.split(OMR_MST_CD + '/')[-1])
    # print('dest : ', dest)
    for item in dest:
        if '.jpg' in item or \
                '.JPG' in item or \
                '.jpeg' in item or \
                '.JPEG' in item or \
                '.png' in item or \
                '.PNG' in item:
            fileNMs.append(item)

    # print(len(fileNMs), fileNMs)

    '''
    2. 탬플릿 이미지 이름 저장
    '''
    templatePath = 'templates/' + OMR_MST_CD + '/'
    # print(templatePath)
    prefix = templatePath
    # 원하는 bucket과 하위경로에 있는 object list
    obj_list = s3.list_objects(Bucket=BUCKET_NAME, Prefix=prefix)  # dict type

    # object list의 Contents를 가져옴
    contents_lst = obj_list['Contents']

    # Content list 출력
    for content in contents_lst:
        tmp = content['Key'].split(OMR_MST_CD + '/')[-1]
        if '.jpg' in tmp or \
                '.JPG' in tmp or \
                '.jpeg' in tmp or \
                '.JPEG' in tmp or \
                '.png' in tmp or \
                '.PNG' in tmp:
            templateNMs.append(tmp)
    # print(len(templateNMs), templateNMs)

    '''
    json 파일 존재 확인
    '''
    # '.json' 파일이 있는지 없는지 알아보기 위한 flag
    isthere = 0
    for item in dest:
        # 'TOT' 로 파일 이름이 시작되고, '_' 가 있어 뒤에는 이 전 저장한 결과의 시간이 있고,
        # 파일 형식이 '.json'인 파일이 있다면 그 파일을 현 위치에서 지우고 job_done/ 디렉토리로 이동
        if item.split("_")[0] == 'TOT' and item.split(".")[-1] == 'json':
            isthere = 1
            S3FileDelMove(base_dir, item, 1) # 마지막 인자가 1이라면 job_done/ 디렉토리로 옮기고 현 위치에서는 삭제
        # 파일 이름이 'TOT.json'이라면 바로 삭제
        if item == 'TOT.json':
            isthere = 1
            S3FileDelMove(base_dir, item, -1) # 마지막 인자가 -1이라면 현 위치에서 삭제, job_done/ 디렉토리로 옮기는 작업 없음
    # '.json' 파일이 아예 없다면(어떠한 이전 결과도 없다면) 메시지 출력
    if not isthere:
        print("No json file in " + "'" + base_dir + "'")

    return fileNMs, templateNMs


def S3GetFile(base_dir, fileNM):
    """
    base_dir 안의 파일을 read
    :param base_dir :로드할 경로(시험/회사/지점/OMR마스터코드)
    :param fileNM   :읽어야 할 파일 이름
    :return         :읽은 파일
    """

    # S3 커넥션
    s3 = ConnS3()
    # base_dir과 fileNM 경로 합쳐서 'edu-ai' 버킷 안에서 해당하는 경로의 파일 read
    ret = s3.Object('edu-ai', os.path.join(base_dir, fileNM)).get().get('Body').read()

    return ret


def S3FileDelMove(base_dir, delFileNM, DorM, newFileNM=None):
    """
    base_dir 안의 지워야 할 파일 이름을 받아 지움, 이 때 그냥 지울지, job_fail/이나 job_done/으로 보내고 지울지 인자로 받아옴
    :param base_dir     :탐색할 경로(시험/회사/지점/OMR마스터코드)
    :param delFileNM    :지워야 할 파일 이름
    :param DorM         :바로 지울지, job_fail/이나 job_done/으로 보내고 지울지 결정 인자
    :newFileNM          :로컬에서 인식할 때 변경된 파일 이름
    :return             :없음
    """
    # S3 커넥션
    s3 = ConnS3()

    # 인자가 1이면 job_done/ 으로 보내고 지움
    if DorM == 1:
        s3.Object(BUCKET_NAME, base_dir + "job_done/" + delFileNM).copy_from(
            CopySource="edu-ai" + "/" + base_dir + delFileNM)
        s3.Object(BUCKET_NAME, base_dir + delFileNM).delete()

    # 인자가 0이면 job_fail/ 로 보내고 지움
    if DorM == 0:
        s3.Object(BUCKET_NAME, base_dir + "job_fail/" + delFileNM).copy_from(
            CopySource="edu-ai" + "/" + base_dir + delFileNM)
        s3.Object(BUCKET_NAME, base_dir + delFileNM).delete()

    # 인자가 2이면 check_fail/ 로 보내고 지움
    if DorM == 2:
        s3.Object(BUCKET_NAME, base_dir + "check_fail/" + delFileNM).copy_from(
            CopySource="edu-ai" + "/" + base_dir + delFileNM)
        s3.Object(BUCKET_NAME, base_dir + delFileNM).delete()

    # 인자가 -1이면 바로 지움
    # TOT.json (시간 없는 TOT) 바로 지우기
    if DorM == -1:
        s3.Object(BUCKET_NAME, base_dir + delFileNM).delete()

    '''
    로컬에서 업로드 시 사용
    '''
    # 로컬에서 업로드한 이미지가 인식 성공했을 때 job_done/ 로 보내고 지움
    if DorM == 3:
        s3.Object(BUCKET_NAME, base_dir + "job_done/" + newFileNM).copy_from(
            CopySource="edu-ai" + "/" + base_dir + delFileNM)
        s3.Object(BUCKET_NAME, base_dir + delFileNM).delete()

    # 로컬에서 업로드한 이미지가 인식 실패했을 때 job_fail/ 로 보내고 지움
    if DorM == 4:
        s3.Object(BUCKET_NAME, base_dir + "job_fail/" + newFileNM).copy_from(
            CopySource="edu-ai" + "/" + base_dir + delFileNM)
        s3.Object(BUCKET_NAME, base_dir + delFileNM).delete()


def S3RetUpload(base_dir, retFileNM, ret):
    """
    결과 json 파일 이름으로 결과 json 파일을 S3에 저장
    :param base_dir     :탐색할 경로(시험/회사/지점/OMR마스터코드)
    :param retFileNM    :받아온 업로드 할 파일 이름
    :param ret          :받아온 파일
    :return             :없음
    """
    # S3 커넥션
    s3 = ConnS3()
    # json 파일 S3에 업로드
    s3.Object(BUCKET_NAME, base_dir + retFileNM).put(Body=json.dumps(ret, ensure_ascii=False, indent="\t"))


def S3IMGUpload(base_dir, retFileNM, ret):
    """
    올바르게 회전한 이미지 파일 S3에 저장
    :param base_dir     :탐색할 경로(시험/회사/지점/OMR마스터코드)
    :param retFileNM    :받아온 업로드 할 파일 이름
    :param ret          :받아온 파일
    :return             :없음
    """

    # S3 커넥션
    s3 = ConnS3()
    # json 파일 S3에 업로드
    s3.Object(BUCKET_NAME, base_dir + retFileNM).put(Body=bytes(ret))