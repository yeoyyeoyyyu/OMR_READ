"""
'ETOOS_CROP_RECOG_KICE' PURPOSE
- 6월 평가원 시험 인식용
- 인식해야할 영역 크롭
- 인식해야 할 전체 영역부터 그 영역의 마킹 란까지 크롭
- 이름, 수험번호, 생년월일, 답안, 성별, 선택과목, 주관식 답안(수학) 에 맞게 각 모델을 통해 인식
- 즉, 이미지 자르고 인식하는 과정
"""
import cv2
import warnings
import numpy as np


class CropRecognition:
    """
    영역별 이미지 crop 및 인식
    :param targetImg    :자를 원본 이미지
    :param base_dir     :bucket directory
    :param models       :인식할 모델 리스트(name0Model, name1Model, idx0Model, case1Model, answerModel, sexModel, longAnswerModel)
    """

    def __init__(self, targetImg, base_dir, OMR_MST_CD, name0Model, name1Model, idx0Model, case1Model, answerModel,
                 sexModel, longAnswerModel, select02Model, select09Model, birthdayModel):
        self.targetImg = targetImg
        self.OMR_MST_CD = OMR_MST_CD
        self.name0Model = name0Model
        self.name1Model = name1Model
        self.idx0Model = idx0Model
        self.case1Model = case1Model
        self.answerModel = answerModel
        self.sexModel = sexModel
        self.longAnswerModel = longAnswerModel
        self.select02Model = select02Model  ## 2021.01.29
        self.select09Model = select09Model  ## 2021.06.01
        self.birthdayModel = birthdayModel
        self.base_dir = base_dir

    def onMouse(self, event, x, y, flags, param):
        """
        테스트를 위한 openCV 처리
        - 윈도우 이미지 클릭 시 좌표 프린트
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)

    def cropRegion(self, targetImg, xywh, cropX, cropY):
        """
        이미지에서 영역 자르기
        :param targetImg    :자를 이미지
        :param xywh         :잘라야 할 영역
        :param cropX, cropY :이미지 resize를 위한 X, Y 크기
        :return             :자른 이미지
        """
        warnings.filterwarnings(action='ignore')

        # targetImg(어레이 타입)에 x시작 좌표와 끝 좌표, y시작 좌표와 끝 좌표를 인자로 넣어주면 그 영역만 잘리게 되어 roiRegion에 값이 들어감
        roiRegion = targetImg[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]]

        # 이미지 복사
        img = roiRegion.copy()

        # 이미지 resize가 필요하지 않으면 0이므로 조건을 통해 resize
        if cropX > 0 and cropY > 0:
            img = cv2.resize(img, dsize=(cropX, cropY), interpolation=cv2.INTER_AREA)

        return img

    def mapName(self, regionImg):
        warnings.filterwarnings(action='ignore')

        # reset tensor
        self.name0Model.allocate_tensors()
        name0_input_detail = self.name0Model.get_input_details()
        name0_output_detail = self.name0Model.get_output_details()
        # print(name0_input_detail)
        self.name1Model.allocate_tensors()
        name1_input_detail = self.name1Model.get_input_details()
        name1_output_detail = self.name1Model.get_output_details()
        # print(name1_input_detail)

        # 이름 매핑 dictionary(자료형)
        # 0: 이름 자음, 1: 이름 모음, 2: 이름 받침
        nameDict0 = {0: 'ㄱ', 1: 'ㄱ', 2: 'ㄴ', 3: 'ㄷ', 4: 'ㄷ', 5: 'ㄹ', 6: 'ㅁ', 7: 'ㅂ', 8: 'ㅂ', 9: 'ㅅ', 10: 'ㅅ', \
                     11: 'ㅇ', 12: 'ㅈ', 13: 'ㅈ', 14: 'ㅊ', 15: 'ㅋ', 16: 'ㅌ', 17: 'ㅍ', 18: 'ㅎ', -1: 'none'}
        nameDict1 = {0: 'ㅏ', 1: 'ㅐ', 2: 'ㅏ', 3: 'ㅐ', 4: 'ㅓ', 5: 'ㅔ', 6: 'ㅕ', 7: 'ㅖ', 8: 'ㅗ', 9: 'ㅘ', 10: 'ㅘ', \
                     11: 'ㅚ', 12: 'ㅛ', 13: 'ㅜ', 14: 'ㅝ', 15: 'ㅝ', 16: 'ㅟ', 17: 'ㅠ', 18: 'ㅡ', 19: 'ㅢ', 20: 'ㅣ',
                     -1: 'none'}
        nameDict2 = {0: 'ㄱ', 1: 'ㄲ', 2: 'ㄴ', 3: 'ㄴ', 4: 'ㄷ', 5: 'ㄹ', 6: 'ㄹ', 7: 'ㄻ', 8: 'ㄼ', 9: 'ㅀ', 10: 'ㅁ', \
                     11: 'ㅂ', 12: 'ㅅ', 13: 'ㅆ', 14: 'ㅇ', 15: 'ㅈ', 16: 'ㅊ', 17: 'ㅋ', 18: 'ㅌ', 19: 'ㅍ', 20: 'ㅎ',
                     -1: 'none'}

        # name10Region: 이름 첫 글자의 자음 (전체 이름 영역 이미지와 자음 마킹란 영역과 모델 인식을 위한 resize)
        # name11Region: 이름 첫 글자의  모음 (전체 이름 영역 이미지와 모음 마킹란 영역과 모델 인식을 위한 resize)
        # name12Region: 이름 첫 글자의  받침 (전체 이름 영역 이미지와 받침 마킹란 영역과 모델 인식을 위한 resize)
        name00Region = np.array((self.cropRegion(regionImg, (0, 0, 20, 318), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)
        name01Region = np.array((self.cropRegion(regionImg, (20, 0, 20, 365), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)
        name02Region = np.array((self.cropRegion(regionImg, (40, 0, 20, 365), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)

        # name10Region: 이름 두번째 글자의 자음 (전체 이름 영역 이미지와 자음 마킹란 영역과 모델 인식을 위한 resize)
        # name11Region: 이름 두번째 글자의  모음 (전체 이름 영역 이미지와 모음 마킹란 영역과 모델 인식을 위한 resize)
        # name12Region: 이름 두번째 글자의  받침 (전체 이름 영역 이미지와 받침 마킹란 영역과 모델 인식을 위한 resize)
        name10Region = np.array((self.cropRegion(regionImg, (60, 0, 20, 318), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)
        name11Region = np.array((self.cropRegion(regionImg, (80, 0, 20, 365), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)
        name12Region = np.array((self.cropRegion(regionImg, (100, 0, 20, 365), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)

        # name20Region: 이름 세번째 글자의 자음 (전체 이름 영역 이미지와 자음 마킹란 영역과 모델 인식을 위한 resize)
        # name21Region: 이름 세번째 글자의  모음 (전체 이름 영역 이미지와 모음 마킹란 영역과 모델 인식을 위한 resize)
        # name22Region: 이름 세번째 글자의  받침 (전체 이름 영역 이미지와 받침 마킹란 영역과 모델 인식을 위한 resize)
        name20Region = np.array((self.cropRegion(regionImg, (120, 0, 20, 318), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)
        name21Region = np.array((self.cropRegion(regionImg, (140, 0, 20, 365), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)
        name22Region = np.array((self.cropRegion(regionImg, (160, 0, 20, 365), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)

        # name30Region: 이름 네번째 글자의 자음 (전체 이름 영역 이미지와 자음 마킹란 영역과 모델 인식을 위한 resize)
        # name31Region: 이름 네번째 글자의  모음 (전체 이름 영역 이미지와 모음 마킹란 영역과 모델 인식을 위한 resize)
        # name32Region: 이름 네번째 글자의  받침 (전체 이름 영역 이미지와 받침 마킹란 영역과 모델 인식을 위한 resize)
        name30Region = np.array((self.cropRegion(regionImg, (180, 0, 20, 318), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)
        name31Region = np.array((self.cropRegion(regionImg, (200, 0, 20, 365), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)
        name32Region = np.array((self.cropRegion(regionImg, (220, 0, 20, 365), 10, 238) / 256).reshape((1, 238, 10, 1)), dtype=np.float32)

        # 인식할 이름을 담을 어레이
        name = []
        acc = []
        # 성씨 인식(자음은 name0Model, 모음과 받침은 name1Model을 사용하여 각 인식 영역 이미지 인식)
        # 초성에 쌍자음(ㄲ, ㄸ, ㅃ, ㅆ, ㅉ)이 인식되면 일반 자음(ㄱ, ㄷ, ㅂ, ㅅ, ㅈ)으로 변경
        self.name0Model.set_tensor(name0_input_detail[0]['index'], name00Region)
        self.name0Model.invoke()
        result = self.name0Model.get_tensor(name0_output_detail[0]['index'])
        name.append(nameDict0[int(result.argmax() - 1)])
        acc.append(result.max())

        self.name1Model.set_tensor(name1_input_detail[0]['index'], name01Region)
        self.name1Model.invoke()
        result = self.name1Model.get_tensor(name1_output_detail[0]['index'])
        name.append(nameDict1[result.argmax() - 1])
        acc.append(result.max())
        
        self.name1Model.set_tensor(name1_input_detail[0]['index'], name02Region)
        self.name1Model.invoke()
        result = self.name1Model.get_tensor(name1_output_detail[0]['index'])
        name.append(nameDict2[result.argmax() - 1])
        acc.append(result.max())
        
        self.name0Model.set_tensor(name0_input_detail[0]['index'], name10Region)
        self.name0Model.invoke()
        result = self.name0Model.get_tensor(name0_output_detail[0]['index'])
        name.append(nameDict0[result.argmax() - 1])
        acc.append(result.max())
        
        self.name1Model.set_tensor(name1_input_detail[0]['index'], name11Region)
        self.name1Model.invoke()
        result = self.name1Model.get_tensor(name1_output_detail[0]['index'])
        name.append(nameDict1[result.argmax() - 1])
        acc.append(result.max())
        
        self.name1Model.set_tensor(name1_input_detail[0]['index'], name12Region)
        self.name1Model.invoke()
        result = self.name1Model.get_tensor(name1_output_detail[0]['index'])
        name.append(nameDict2[result.argmax() - 1])
        acc.append(result.max())
        
        self.name0Model.set_tensor(name0_input_detail[0]['index'], name20Region)
        self.name0Model.invoke()
        result = self.name0Model.get_tensor(name0_output_detail[0]['index'])
        name.append(nameDict0[result.argmax() - 1])
        acc.append(result.max())
        
        self.name1Model.set_tensor(name1_input_detail[0]['index'], name21Region)
        self.name1Model.invoke()
        result = self.name1Model.get_tensor(name1_output_detail[0]['index'])
        name.append(nameDict1[result.argmax() - 1])
        acc.append(result.max())
        
        self.name1Model.set_tensor(name1_input_detail[0]['index'], name22Region)
        self.name1Model.invoke()
        result = self.name1Model.get_tensor(name1_output_detail[0]['index'])
        name.append(nameDict2[result.argmax() - 1])
        acc.append(result.max())
        
        self.name0Model.set_tensor(name0_input_detail[0]['index'], name30Region)
        self.name0Model.invoke()
        result = self.name0Model.get_tensor(name0_output_detail[0]['index'])
        name.append(nameDict0[result.argmax() - 1])
        acc.append(result.max())
        
        self.name1Model.set_tensor(name1_input_detail[0]['index'], name31Region)
        self.name1Model.invoke()
        result = self.name1Model.get_tensor(name1_output_detail[0]['index'])
        name.append(nameDict1[result.argmax() - 1])
        acc.append(result.max())
        
        self.name1Model.set_tensor(name1_input_detail[0]['index'], name32Region)
        self.name1Model.invoke()
        result = self.name1Model.get_tensor(name1_output_detail[0]['index'])
        name.append(nameDict2[result.argmax() - 1])
        acc.append(result.max())
        
        if name[-3] == 'none' or name[-2] == 'none':            
            name[-3] = 'none'
            name[-2] = 'none'
            name[-1] = 'none'

        # print(name, np.mean(acc))
        return name, np.mean(acc)

    def mapCode(self, regionImg):
        """
        수험번호 인식
        :param regionImg    :수험번호 영역 이미지
        :return             :인식된 수험번호, 인식 정확도
        """
        warnings.filterwarnings(action='ignore')

        # 모델 인식을 위한 resize 크기
        codeX, codeY = 12, 264

        # y좌표와 h는 이미지의 높이
        y, h = 1, 331

        # 인식할 수험번호를 담을 어레이
        code = []
        acc = []
        # reset tensor
        self.idx0Model.allocate_tensors()
        input_detail = self.idx0Model.get_input_details()
        output_detail = self.idx0Model.get_output_details()

        # 6월 평가원 학교번호
        idx0Region = np.array((self.cropRegion(regionImg, (0, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        idx1Region = np.array((self.cropRegion(regionImg, (20, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        idx2Region = np.array((self.cropRegion(regionImg, (40, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        idx3Region = np.array((self.cropRegion(regionImg, (60, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        idx4Region = np.array((self.cropRegion(regionImg, (80, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)

        # 6월 평가원 반/번호
        idx5Region = np.array((self.cropRegion(regionImg, (120, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        idx6Region = np.array((self.cropRegion(regionImg, (140, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        idx7Region = np.array((self.cropRegion(regionImg, (160, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        idx8Region = np.array((self.cropRegion(regionImg, (180, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)

        img_list = [idx0Region, idx1Region, idx2Region, idx3Region, idx4Region, idx5Region, idx6Region, idx7Region, idx8Region]
    
        for img in img_list:
            # input data
            self.idx0Model.set_tensor(input_detail[0]['index'], img)
            # run
            self.idx0Model.invoke()
            # get output
            result = self.idx0Model.get_tensor(output_detail[0]['index'])
            # 예측 결과에서 -1 해줘야 실제 마킹값이 나옴
            code.append(result.argmax() - 1)
            acc.append(result.max())

        # print(code, np.mean(acc))
        return code, np.mean(acc)

    def mapBday(self, regionImg):
        """
        생년월일 인식
        :param regionImg: 생년월일 영역 이미지
        :return: 인식된 생년월일, 인식 정확도
        """
        warnings.filterwarnings(action='ignore')

        # 모델 인식을 위한 resize 크기
        codeX, codeY = 12, 264

        # y좌표와 h는 이미지의 높이
        y, h = 1, 330

        # 생년월일을 담을 어레이
        bday = []
        acc = []
        
        # tensor reset
        self.case1Model.allocate_tensors()
        input_detail = self.case1Model.get_input_details()
        output_detail = self.case1Model.get_output_details()

        # 각 년, 월, 일 마킹 영역 이미지를 yy0 등에 저장
        yy0 = np.array((self.cropRegion(regionImg, (0, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        yy1 = np.array((self.cropRegion(regionImg, (20, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        mm0 = np.array((self.cropRegion(regionImg, (60, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        mm1 = np.array((self.cropRegion(regionImg, (80, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        dd0 = np.array((self.cropRegion(regionImg, (120, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        dd1 = np.array((self.cropRegion(regionImg, (140, y, 20, h), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)

        img_list = [yy0, yy1, mm0, mm1, dd0, dd1]
        for img in img_list:
            # input data
            self.case1Model.set_tensor(input_detail[0]['index'], img)
            # run
            self.case1Model.invoke()
            # get output
            result = self.case1Model.get_tensor(output_detail[0]['index'])
            # 예측 결과에서 -1 해줘야 실제 마킹값이 나옴
            bday.append(result.argmax() - 1)
            acc.append(result.max())

        # print(bday, np.mean(acc))
        return bday, sum(acc) / len(acc)

    def mapShortAnswer(self, regionImg, nop):
        """
        객관식 답안 인식(인식할 답안 갯수를 동적으로 입력받아 인식)
        :param regionImg    :답안 영역 이미지
        :param nop          :인식할 답안 갯수
        :return             : 인식된 답안, 인식 정확도
        """
        warnings.filterwarnings(action='ignore')

        # 이미지 인식을 위한 resize 크기
        codeX, codeY = 120, 38

        # 자를 이미지의 시작 좌표와 크기
        x, y, w, h = 0, 0, 101, 33

        # 답안 리스트를 담을 어레이와 정확도를 담을 어레이
        answer = []
        acc = []

        self.answerModel.allocate_tensors()
        input_detail = self.answerModel.get_input_details()
        output_detail = self.answerModel.get_output_details()

        for i in range(nop):
            answerImg = np.array((self.cropRegion(regionImg, (x, y, w, h), codeX, codeY) / 256).reshape((1, codeY, codeX, 1)), dtype=np.float32)
            self.answerModel.set_tensor(input_detail[0]['index'], answerImg)
            self.answerModel.invoke()
            result = self.answerModel.get_tensor(output_detail[0]['index'])
            answer.append(int(result.argmax()))
            acc.append(result.max())
            y += 33
        answer = [-1 if i == 0 else i for i in answer] 
        return answer, np.mean(acc)

    def mapLongAnswer(self, regionImg, nop):
        """
        주관식 답안 인식(인식할 답안 갯수를 동적으로 입력받아 인식)
        :param regionImg    :답안 영역 이미지
        :param nop          :인식할 답안 갯수
        :return             :인식된 답안, 인식 정확도
        """
        warnings.filterwarnings(action='ignore')

        # 3문제 또는 6문제를 각 한 문제씩으로 자를 이미지의 시작 좌표와 크기
        x, y, w, h = 0, 0, 60, 334

        # 한 문항 이미지에서 인식할 각 자리 이미지의 시작 좌표와 크기
        dx, dy, dw, dh = 0, 0, 20, 334

        # 답안을 담을 어레이와 정확도를 담을 어레이
        answer = []
        acc = []

        self.longAnswerModel.allocate_tensors()
        input_detail = self.longAnswerModel.get_input_details()
        output_detail = self.longAnswerModel.get_output_details()

        # 들어온 이미지에서 문항의 갯수만큼 반복문
        for i in range(nop):
            # 이미지에서 한 문항의 크기만큼 이미지를 잘라줌
            answerImg = self.cropRegion(regionImg, (x, y, w, h), 0, 0)

            # 각 자릿수 인식을 담을 어레이
            digit = []

            # 최대 3자리인 주관식 답안이므로 3번 반복문
            for j in range(3):
                # 한 문항안에서 각 자릿수를 인식하기 위해 이미지를 잘라주고 모델 인식을 위해 resize한 후, longAnswerModel로 인식
                digitImg = np.array((self.cropRegion(answerImg, (dx, dy, dw, dh), 12, 264) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
                self.longAnswerModel.set_tensor(input_detail[0]['index'], digitImg)
                self.longAnswerModel.invoke()
                result = self.longAnswerModel.get_tensor(output_detail[0]['index'])
                pred = int(result.argmax())

                # 마킹 안했을 경우(-1 일 경우), "none" 저장
                digit.append("none" if pred - 1 == -1 else pred - 1)
                acc.append(result.max())
                # 자릿수 한칸 오른쪽으로 이동
                dx += 20
            # 한 문항이 끝나고 다음 문항으로 옮길 때의 공백을 위해 13만큼 오른쪽으로 좌표 이동
            if i == 1 or i == 2:
                x += 16 + w
            else:
                x += 13 + w
            dx = 0

            # 만약 예를 들어 ['2',  'none', 'none']
            # ['2',    '5' , 'none']
            # ['none', '5' , 'none']
            # 이런 결과들이 있다면 -1로 뱉어내야 하므로 처리
            if (digit[0] != 'none' or digit[1] != 'none') and (digit[1] == 'none' or digit[2] == 'none'):
                answer.append(int(-1))
            else:
                # 인식한 자릿수에 none이 있다면 모두 없앤 후 숫자만 담은 후 answer에 최종 append를 통해 답안 리스트 추가
                while 'none' in digit:
                    digit.remove('none')
                digit = "".join([str(_) for _ in digit])

                answer.append(int(digit) if digit else int(-1))

        return answer, np.mean(acc)

    def mapSex(self, regionImg):
        """
        성별 인식
        :param regionImg: 성별 영역 이미지
        :return: 인식된 성별, 인식 정확도
        """
        warnings.filterwarnings(action='ignore')

        # 이미지 인식을 위한 resize
        cropX, cropY = 24, 80  # 6월 평가원 용
        # cropX, cropY = 41, 120  # 6월 평가원 용

        # tensor reset
        self.sexModel.allocate_tensors()
        input_detail = self.sexModel.get_input_details()
        output_detail = self.sexModel.get_output_details()

        # 성별은 영역 자를 필요 없이 바로 resize
        img = cv2.resize(regionImg, dsize=(cropX, cropY), interpolation=cv2.INTER_AREA)
        img = np.array((img / 256).reshape(1, cropY, cropX, 1), dtype=np.float32)

        # input data
        self.sexModel.set_tensor(input_detail[0]['index'], img)
        # run
        self.sexModel.invoke()
        # get output
        result = self.sexModel.get_tensor(output_detail[0]['index'])
        sex = result.argmax()
        acc = result.max()

        if sex == 1:
            sex = '남'
        elif sex == 2:
            sex = '여'
        else:
            sex = '-1'

        return sex, acc

    def mapAnswer(self, regionImg, nop):
        """
        객관식 답안 인식(인식할 답안 갯수를 동적으로 입력받아 인식)
        :param regionImg    :답안 영역 이미지
        :param nop          :인식할 답안 갯수
        :return             : 인식된 답안, 인식 정확도
        """
        # print(regionImg.shape)
        # 이미지 인식을 위한 resize 크기
        codeX, codeY = 120, 38

        # 자를 이미지의 시작 좌표와 크기
        x, y, w, h = 0, 0, 101, 33

        # 답안 리스트를 담을 어레이와 정확도를 담을 어레이
        answer = []
        acc = []
        self.answerModel.allocate_tensors()
        input_detail = self.answerModel.get_input_details()
        output_detail = self.answerModel.get_output_details()   

        for i in range(nop):
            answerImg = np.array((self.cropRegion(regionImg, (x, y, w, h), codeX, codeY) / 256).reshape((1, 38, 120, 1)), dtype=np.float32)
            self.answerModel.set_tensor(input_detail[0]['index'], answerImg)
            self.answerModel.invoke()
            result = self.answerModel.get_tensor(output_detail[0]['index'])
            answer.append(int(result.argmax()))
            acc.append(result.max())
            # if i < 10:
            #     y += 33
            # else:
            #     y += 34
            y += 33
            if i % 4 == 0:
                y += 2
        answer = [-1 if i == 0 else i for i in answer] 
        # print(answer, np.mean(acc))
        return answer, np.mean(acc)

    def mapSelect(self, regionImg):
        """
        선택과목 인식(탐구과목)
        :param regionImg    : 선택과목 영역 이미지
        :return             : 인식된 선택과목, 인식 정확도
        """

        # 이미지 인식을 위한 resize
        codeX, codeY = 12, 264

        # 선택 과목을 담을 리스트(두자리)
        selectSubject = []
        self.case1Model.allocate_tensors()
        input_detail = self.case1Model.get_input_details()
        output_detail = self.case1Model.get_output_details()

        # 탐구 과목 마킹란 이미지 두자리를 각각 region0, region1에 잘라 저장
        region0 = np.array((self.cropRegion(regionImg, (0, 0, 21, 330), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)
        region1 = np.array((self.cropRegion(regionImg, (21, 0, 21, 330), codeX, codeY) / 256).reshape((1, 264, 12, 1)), dtype=np.float32)

        self.case1Model.set_tensor(input_detail[0]['index'], region0)
        self.case1Model.invoke()
        result = self.case1Model.get_tensor(output_detail[0]['index'])
        selectSubject.append(result.argmax() - 1)
        acc = result.max()
        
        self.case1Model.set_tensor(input_detail[0]['index'], region1)
        self.case1Model.invoke()
        result = self.case1Model.get_tensor(output_detail[0]['index'])
        selectSubject.append(result.argmax() - 1)
        acc = result.max()

        # string 형태로 리스트에서 각 값을 꺼내와 join
        selectSubject = "".join([str(_) for _ in selectSubject])
        if '-1' in selectSubject:
            selectSubject = '-1'

        return selectSubject, np.mean(acc)

    def mapSelect02(self, regionImg):  ## 2021.01.29
        """
        선택과목 (0, 1, 2) 세가지 인식 (고3 수학)
        :param regionImg: 선택과목 영역 이미지
        :return: 인식된 선택과목, 인식 정확도
        """
        #         plt.imshow(regionImg)
        #         plt.show()
        subject = {0: -1, 1: 36, 2: 37, 3:38}

        self.select02Model.allocate_tensors()
        input_detail = self.select02Model.get_input_details()
        output_detail = self.select02Model.get_output_details()

        cropX, cropY = 19, 120  # 6월 평가원 용도
        img = cv2.resize(regionImg, dsize=(cropX, cropY), interpolation=cv2.INTER_AREA)
        img = np.array((img / 255).reshape(1, cropY, cropX, 1), dtype=np.float32)  # 6월 평가원 용도

        self.select02Model.set_tensor(input_detail[0]['index'], img)
        self.select02Model.invoke()
        result = self.select02Model.get_tensor(output_detail[0]['index'])

        select02 = int(result.argmax())
        select02 = subject[select02]
        acc = result.max()        

        return select02, acc

    def mapSelect09(self, regionImg):   ## 2021.06.01
        """
        제2외국어/한문 선택과목 1~9
        :param regionImg: 선택과목 영역 이미지
        :return: 인식된 선택과목, 인식 정확도
        """
        selectList = {0: -1, 1: 7, 2: 8, 3: 9, 4: 42, 5: 43, 6: 44, 7: 45, 8: 47, 9: 46}

        cropX, cropY = 22, 300
        self.select09Model.allocate_tensors()
        input_detail = self.select09Model.get_input_details()
        output_detail = self.select09Model.get_output_details()

        img = cv2.resize(regionImg, dsize=(cropX, cropY), interpolation=cv2.INTER_AREA)
        img = np.array((img / 256).reshape(1, 300, 22, 1), dtype=np.float32)

        self.select09Model.set_tensor(input_detail[0]['index'], img)
        self.select09Model.invoke()
        result = self.select09Model.get_tensor(output_detail[0]['index'])
        
        select09 = selectList[int(result.argmax())]
        acc = result.max()

        return select09, acc