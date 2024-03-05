import os
import cv2
import numpy as np
from common.OMR_S3_Conn import S3GetFile
from configparser import ConfigParser
config = ConfigParser()
# 두단계 상위의 OMR_MULTI/conf.ini를 리드
config.read('{}/conf.ini'.format(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


def setROI(OMR_MST_CD, gray, templateImages):
    """
    ROI 지정을 위한 template 로드 및 좌표값 저장
    :param OMR_MST_CD       :인자로 받아온 오엠알 마스터 코드
    :param templateImages   :template 이미지 이름
    :return                 : 영역별 좌표값, 템플릿 영역이 마킹된 이미지
    """
    retImg = gray

    # 각 템플릿 영역의 좌상단 좌표를 저장할 리스트
    regionXY = []

    # 읽어야 할 템플릿 이미지만큼 반복문(regionTemplate에 각 반복문마다 다음 index의 템플릿 이미지 이름이 저장됨)
    for regionTemplate in templateImages:
        # S3로부터 이미지 읽어오기
        # OMR_MST_CD 에 맞는 templates 경로 지정하여 템플릿 이미지 읽어오기(순서가 중요)
        # 이미지(bytearray) 읽어오고, decode를 통해 이미지화
        path = config['S3']['first_dir'] + '/omr/templates/' + OMR_MST_CD + '/'
        template = S3GetFile(path, regionTemplate)
        template = cv2.imdecode(np.asarray(bytearray(template)), 0)

        # openCV의 메치템플릿을 사용하여 읽을 grayscale OMR 이미지에서 탐색
        res = cv2.matchTemplate(retImg, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 템플릿이 탐색된 위치의 좌상단 좌표 저장
        top_left = max_loc

        w, h = template.shape[::-1]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(retImg, top_left, bottom_right, 0, 2)

        # 좌상단 좌표를 리스트에 append
        regionXY.append(top_left)
    # cv2.imwrite('KICE/check/drawtemplate.jpg', retImg)
    return regionXY, retImg

def contrast_gray(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high-low)/np.maximum(10, high+low), high, low

def adjust_contrast_gray(img, target = 0.4):
    contrast, high, low = contrast_gray(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200./np.maximum(10, high-low)
        img = (img - low + 25)*ratio
        img = np.maximum(np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img
