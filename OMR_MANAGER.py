'''
OMR_MANAGER.py
-OMR_MAIN.py에 의해 호출
-전달받은 파라미터들로 과목별 인식파일 호출
'''
import platform
from multiprocessing import current_process

# 평가원 모의고사 인식 파일
import KICE.contour.KICE_KOR_CONTOUR as KiceKor
import KICE.contour.KICE_MATH_CONTOUR as KiceMath
import KICE.contour.KICE_ENG_CONTOUR as KiceEng
import KICE.contour.KICE_HIST_CONTOUR as KiceHist
import KICE.contour.KICE_TAM_CONTOUR as KiceTam
import KICE.contour.KICE_FOREIGN_CONTOUR as KiceForeign


class OmrManager:
    def __init__(self, EXAM_CD, CMPN_CD, BSTOR_CD, OMR_MST_CD, base_dir, fileNMs, templates, result_list, successFiles, cvt_templates=None):
        self.EXAM_CD = EXAM_CD
        self.CMPN_CD = CMPN_CD
        self.BSTOR_CD = BSTOR_CD
        self.OMR_MST_CD = OMR_MST_CD
        self.base_dir = base_dir
        self.is_local = platform.system()
        self.fileNMs = fileNMs
        self.templates = templates
        self.result_list = result_list
        self.successFiles = successFiles
        self.cvt_templates = cvt_templates

        self.run()
        
    def run(self):
        # OMR_MST_CD
        # 10: 평가원 고3 국어 / 11: 평가원 고3 수학 / 12: 평가원 고3 영어 / 13: 평가원 고3 한국사 / 14: 평가원 고3 탐구 / 15: 평가원 고3 제2외국어
        
        omr_list = {
            '10': KiceKor.main, 
            '11': KiceMath.main,
            '12': KiceEng.main, 
            '13': KiceHist.main, 
            '14': KiceTam.main, 
            '15': KiceForeign.main}

        successFiles, result_TOT, cvt_templates = omr_list[self.OMR_MST_CD](self.EXAM_CD, self.CMPN_CD, self.BSTOR_CD, self.OMR_MST_CD, 
                                                                             self.base_dir, self.fileNMs, self.templates, self.is_local)
        
        self.successFiles.append(successFiles)
        self.result_list.extend(result_TOT)
        self.cvt_templates.append(cvt_templates)