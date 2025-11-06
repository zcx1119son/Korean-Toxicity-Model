# **🇰🇷 HateSpeech-Detector-NLP: 한국어 유해성 텍스트 2단계 분류 모델**

## **💡 프로젝트 개요 (Project Overview)**

본 프로젝트는 온라인상에서 발생하는 **혐오 발언, 비하, 스팸 등의 유해성 텍스트**를 자동으로 탐지하고 그 유형을 분류하는 **2단계 분류 (Two-Tier Classification)** AI 모델 개발을 목표로 합니다. 이 모델은 사용자 콘텐츠 관리 및 모더레이션 시스템의 핵심 기능을 수행합니다.

* **1차 분류:** 텍스트의 유해성(Toxic) 여부 판단 (유해 / 정상)  
* **2차 분류:** 유해 텍스트의 구체적인 유형 분류 (예: 성별, 지역, 연령, 정치 등 7가지)  
* **저장소 URL:** [https://github.com/zcx1119son/Korean-Toxicity-Model](https://www.google.com/search?q=https://github.com/zcx1119son/Korean-Toxicity-Model)

## **⚙️ 기술 스택 (Tech Stack)**

| 구분 | 기술 / 라이브러리 | 상세 내용 |
| :---- | :---- | :---- |
| **핵심 모델** | **BERT (KcBERT Fine-tuning)** | 한국어에 최적화된 BERT 모델 기반으로 파인튜닝하여 높은 성능 달성 |
| **딥러닝** | PyTorch, Hugging Face Transformers | 모델 구현 및 학습, 사전 학습 모델 로드 |
| **데이터 처리** | Pandas, Numpy, Scikit-learn | 데이터 전처리, 학습/검증 분할, 성능 평가 |
| **개발 환경** | Jupyter Notebook (.ipynb) | 실험 및 결과 도출 환경 |

## **✅ 최종 성능 및 결과**

BERT 기반의 2단계 분류 구조를 통해 유해성 판단 및 유형 분류에서 높은 성능을 달성했습니다.

| 분류 단계 | 모델 특징 | 평가 지표 (F1-Score 기준) |
| :---- | :---- | :---- |
| **1차 분류 (유해/정상)** | 이진 분류 모델 | **F1-Score: 0.95 이상** |
| **2차 분류 (유형 분류)** | 다중 분류 모델 | **F1-Score: 0.88 이상** |

### **💡 트러블 슈팅 및 기여 (Troubleshooting & Contribution)**

* **데이터 불균형 해결:** AI Hub 데이터셋의 클래스별 불균형 문제를 해소하기 위해 \*\*class\_weight\*\*를 계산하여 학습 과정에 적용함으로써, 소수 클래스에 대한 예측 성능을 크게 개선했습니다.  
* **효율적인 분류:** 2단계 계층적 분류 구조를 도입하여 모든 텍스트를 7가지 유형으로 분류하는 대신, 유해 텍스트에만 유형 분류 모델을 적용하여 **전체 시스템의 추론 시간 효율**을 높였습니다.

## **🛠️ 코드 및 재현 안내 (Reproduction Guide)**

프로젝트 코드, 대용량 모델 파일, 데이터셋 출처, 그리고 상세 보고서까지 **모든 자료가 아래 링크에 통합**되어 있습니다.

### **1\. 📂 코드 및 의존성 파일**

* **메인 코드 (.ipynb):** [Korean\_Toxic\_Text\_Classification.ipynb](https://www.google.com/search?q=https://github.com/zcx1119son/Korean-Toxicity-Model/blob/master/Korean_Toxic_Text_Classification.ipynb)  
* **환경 설정:** requirements.txt (프로젝트 재현을 위한 필수 라이브러리 목록)

### **2\. 💾 대용량 자료 다운로드 링크 (필수)**

| 자원 | 용량 | 링크 |
| :---- | :---- | :---- |
| **학습 데이터셋 (AI Hub)** | 대용량 | [AI Hub 다운로드 페이지 (국가기록물 대상)](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EA%B5%AD%EA%B0%80%EA%B8%B0%EB%A1%9D%EB%AC%BC&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71788) |
| **학습된 모델 파일 (772MB)** | **약 772MB** | [Google Drive 모델 파일 다운로드](https://www.google.com/search?q=https://drive.google.com/drive/folders/1dL8Y7zl4BddPBDbXeHF5BwB5hf-1N_b7?usp%3Dsharing) |
| **상세 PPT/PDF 자료** | 보조 보고서 | [Google Drive 상세 자료 링크](https://drive.google.com/drive/folders/1c32AJIo_1g993qb2vQyJK4hUfzu8aug4?usp=sharing) |

### **3\. 재현 단계 (Replication Steps)**

1. **Repository Clone:** 본 GitHub 저장소를 로컬로 복제합니다.  
2. **환경 설정:** requirements.txt를 사용하여 Python 환경을 설정합니다. (pip install \-r requirements.txt)  
3. **파일 다운로드:** 상단의 **AI Hub 데이터셋**과 **Google Drive 모델 파일**을 모두 다운로드합니다.  
4. **코드 실행:** Korean\_Toxic\_Text\_Classification.ipynb 파일을 열어 전처리, 모델 로드, 예측 결과를 순서대로 확인합니다.

© zcx1119son
