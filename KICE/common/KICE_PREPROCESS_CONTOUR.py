import cv2
import numpy as np
import pandas as pd
import math
from common.OMR_S3_Conn import S3GetFile, S3IMGUpload #woo


# 기울인 이미지 고도화(오픈소스)
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


def deskew_img(img):
    # BGR2GRAY
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # asis 127
    thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)[1]

    # find largest contour
    contours, hr = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contr = contours[0]

    # find min rect
    rect = cv2.minAreaRect(contr)

    # get angle
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    print(angle)

    if np.abs(angle) < 7:
        return rotateImage(img, angle)
    else:
        return img


# --------------------------------------------------------------------------------------------
# 90, 180, 270도 회전 함수 추가
# --------------------------------------------------------------------------------------------
def rotate_90(cvImage):
    newImage = cvImage.copy()
    # 가로세로 길이 비교 후 90도 회전
    if newImage.shape[0] > newImage.shape[1]:
        newImage = cv2.rotate(newImage, cv2.ROTATE_90_CLOCKWISE)
    else:
        pass

    # 상단에 검정색이 있는지?
    dst_a = newImage[np.int(newImage.shape[0] / 40.0):np.int(newImage.shape[0] / 17.0), :]
    dst_b = newImage[newImage.shape[0] - np.int(newImage.shape[0] / 17.0):newImage.shape[0] - np.int(newImage.shape[0] / 40.0), :]

    # 아래영역 자른것의 평균값이 더 작다 <=> 더 어둡다 => 180도 회전
    if np.mean(dst_a) > np.mean(dst_b):
        newImage = cv2.rotate(newImage, cv2.ROTATE_180)
    else:
        pass

    return newImage

def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]


    rows, cols = img.shape[0], img.shape[1]

    # 회전변환
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    img_rot = cv2.warpAffine(img, M, (cols, rows))

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")

    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")

    # 원근변환
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img_crop = cv2.warpPerspective(img, M, (width, height))

    '''90도 회전 START'''
    if angle < -45:
        height, width = img_crop.shape
        center = (width / 2, height / 2)
        angle = 90
        scale = 1
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        radians = math.radians(angle)
        sin = math.sin(radians)
        cos = math.cos(radians)
        bound_w = int((height * scale * abs(sin)) + (width * scale * abs(cos)))
        bound_h = int((height * scale * abs(cos)) + (width * scale * abs(sin)))

        matrix[0, 2] += ((bound_w / 2) - center[0])
        matrix[1, 2] += ((bound_h / 2) - center[1])

        img_crop = cv2.warpAffine(img_crop, matrix, (bound_w, bound_h))
    '''90도 회전 END'''

    img_text_crop = img_crop[0:50, 0:img_crop.shape[1]]

    x = int(rect[0][0])

    return img_crop, img_text_crop, x

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

# -------------------------------------------------------------------
# Image preprocessing Class
# -------------------------------------------------------------------
class PreprocessingImage:
    """
    이미지 로드 및 프리프로세싱
    :param base_dir: 로드할 이미지 path
    """

    # loadImage에서 grayscale로 변환된 이미지 저장
    img = None

    def __init__(self, base_dir, OMR_MST_CD):
        self.base_dir = base_dir
        self.OMR_MST_CD = OMR_MST_CD

    def onMouse(self, event, x, y, flags, param):
        """
        테스트를 위한 openCV 처리
        - 윈도우 이미지 클릭 시 좌표 프린트
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)

    def loadImage(self, fileNM):
        """
        fileNM 파일 로드 및 전처리
        :param fileNM   :로드할 이미지 이름
        :return         :그레이 이미지, 전처리 된 이미지
        """
        # 인식할 파일 이름 프린트
        print(fileNM)

        # S3로부터 이미지 읽어올 때 이미지 읽는 방법
        img = S3GetFile(self.base_dir, fileNM)
        # 이미지(bytearray) 읽어오고, decode를 통해 이미지화
        img = cv2.imdecode(np.asarray(bytearray(img)), cv2.IMREAD_COLOR)

        # 90, 180, 270도 회전 함수 추가
        img = rotate_90(img)

        # S3 업로드를 위해 이미지 encode 처리하여 encode_img에 저장
        encode_img = cv2.imencode('.jpg', img)[1].tobytes()

        # encode_img 파일 base_dir/rot_images 하위에 rot_ prefix 붙여서 업로드
        S3IMGUpload(self.base_dir + 'rot_images/', 'rot_' + fileNM, encode_img)

        # 컬러 이미지를 gray scale로 바꿔줌(인식률 향상을 위한 전처리)
        src = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        src = cv2.resize(src, dsize=(1100, 850), interpolation=cv2.INTER_AREA)
        # gray 이미지를 저장
        self.img = src.copy()
        # s = adjust_contrast_gray(src)
        blur = cv2.GaussianBlur(src, (5, 5), sigmaX=5)
        # contourBinary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 5)

        src_f = src.astype(np.float32)
        blur_f = blur.astype(np.float32)

        s = 1.7 * src_f - 0.7 * blur_f
        s = np.clip(s, 0, 255).astype(np.uint8)

        contourBinary = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 5)

        return img, src, contourBinary

    def setContour(self, n, recogBinary, contourBinary):
        contours, hierarchy = cv2.findContours(contourBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:n]
        print('number of contours : ', len(contours))

        df = pd.DataFrame(contours).rename({0: 'contours'}, axis=1)

        # contours에 있는 것들을 cv2.minAreaRect해서 xywh에 저장
        df['xywh'] = df['contours'].apply(cv2.minAreaRect)
        df = sortDF(df)

        box = df['xywh'].apply(cv2.boxPoints)
        areas = []
        img_crops = []
        X = []
        Y = []

        for i in df.index:
            # 각 컨투어들의 xywh값
            rect = df['xywh'][i]
            area = int(rect[1][0] * rect[1][1])
            areas.append(area)

            # 기울기 반영하여 좌표잡음
            coords = box.loc[i]
            aa = np.int0(coords)
            cv2.drawContours(self.img, [aa], -1, (0, 0, 255), 2)

            # 기울기 반영되어 crop한 이미지 / x값 retrun
            img_crop, img_text_crop, x = crop_minAreaRect(recogBinary, df['xywh'][i])

            img_crops.append(img_crop)
            X.append(x)
            Y.append(int(rect[0][1]))

        # cv2.imwrite('KICE/check/img_contour.jpg', self.img)
        # 컨투어대로 잘린 이미지, x,y좌표, 면적 리턴
        return img_crops, X, Y, areas, df

def sortDF(df):
    areas = []
    X = []
    Y = []
    W = []
    H = []
    R = []

    for i in df.index:
        # 각 컨투어들의 xywh값
        rect = df['xywh'][i]
        areas.append(int(rect[1][0] * rect[1][1]))
        X.append(int(rect[0][0]))
        Y.append(int(rect[0][1]))

        if rect[2] < -45:
            w = int(rect[1][1])
            h = int(rect[1][0])
        else:
            w = int(rect[1][0])
            h = int(rect[1][1])
        R.append(round(h / w, 2))
        W.append(w)
        H.append(h)

    df['area'] = areas
    df['x'] = X
    df['y'] = Y
    df['w'] = W
    df['h'] = H
    df['r'] = R

    df = df.sort_values(by=['x', 'y', 'area'], ascending=True)
    df = df[(df['r'] > 0.8) & (df['r'] < 8)]
    df = df.reset_index(drop=True)

    return df