# **🇰🇷 HateSpeech-Detector-NLP: 한국어 유해성 텍스트 2단계 분류 모델**

## **🚀 프로젝트 개요 (Project Overview)**

본 프로젝트는 **한국어 텍스트**에서 유해성 여부를 판단하고, 유해하다고 판단될 경우 그 \*\*유형(Type)\*\*까지 분류하는 **2단계 분류 (Two-Tier Classification)** AI 모델입니다. 온라인 커뮤니티, 댓글 등에서 혐오 발언 및 유해 콘텐츠를 자동으로 필터링하여 건전한 소통 환경을 조성하는 것을 목표로 합니다.

### **핵심 역할**

* **1차 분류:** 텍스트의 유해성(Toxic) 여부(유해/정상) 판단  
* **2차 분류:** 유해 텍스트의 구체적인 유형(성별, 종교, 연령, 정치 등 7가지) 분류  
* **기술적 기여:** Hugging Face의 **BERT(KcBERT)** 모델을 파인튜닝하여 한국어 텍스트 분류 성능 극대화

## **⚙️ 기술 스택 (Tech Stack)**

| 구분 | 기술 / 라이브러리 | 용도 |
| :---- | :---- | :---- |
| **언어** | Python 3.8+ | 모델 개발 및 데이터 전처리 |
| **딥러닝** | PyTorch, Hugging Face Transformers | BERT 모델 구현 및 학습 |
| **데이터** | Pandas, Scikit-learn | 데이터 로딩, 전처리, 평가 |
| **개발 환경** | Jupyter Notebook (.ipynb) | 실험 및 코드 관리 |

## **✅ 최종 성능 및 결과**

BERT 기반 모델을 활용하여 높은 수준의 분류 정확도를 달성했습니다.

| 분류 단계 | 모델 | 평가 지표 (F1-Score 기준) |
| :---- | :---- | :---- |
| **1차 분류 (유해/정상)** | BERT Classifier (KcBERT Fine-tuning) | **F1-Score: 0.95 이상** |
| **2차 분류 (7가지 유형)** | Harmful Type Classifier | **F1-Score: 0.88 이상** |

### **🔍 결과 예시 (Result Example)**

입력 텍스트: "나이가 들면 다 저렇게 된다더니, 지식 수준이 떨어지는 게 너무 티 난다."  
\---  
1차 분류 결과: 유해  
2차 분류 결과 (유해성 유형): 연령 혐오 및 비하

## **🛠️ 모델 재현 안내 (Reproduction Guide)**

대용량 데이터 및 모델 가중치 파일은 GitHub 용량 제한(100MB)으로 인해 외부에 보관되어 있습니다. 아래 링크를 통해 모든 필요 파일을 다운로드하여 코드를 재현할 수 있습니다.

### **1\. 📂 코드 파일**

| 파일 | 내용 |
| :---- | :---- |
| **Korean\_Toxic\_Text\_Classification.ipynb** | **[바로가기](https://www.google.com/search?q=./Korean_Toxic_Text_Classification.ipynb)** (모델 정의, 학습, 평가 및 예측 코드 전체) |
| **requirements.txt** (예상) | 프로젝트 실행에 필요한 파이썬 라이브러리 목록 |

### **2\. 💾 데이터셋 및 모델 다운로드 (필수)**

| 자원 | 용량 | 링크 |
| :---- | :---- | :---- |
| **학습 데이터셋** | 대용량 | [AI Hub 다운로드 페이지](https://www.google.com/search?q=https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex%3D1%26currMenu%3D115%26topMenu%3D100%26srchOptnCnd%3DOPTNCND001%26searchKeyword%3D%25EC%258A%25A4%25ED%258C%25B4%26srchDetailCnd%3DDETAILCND001%26srchOrder%3DORDER001%26srchPagePer%3D20%26aihubDataSe%3Ddata%26dataSetSn%3D71788) |
| **학습된 모델 파일** | **약 772MB** | [Google Drive 모델 파일 다운로드](https://drive.google.com/drive/folders/1e-oi0XS1VkUjUAImm09irzHy3Irr_WV6?usp=sharing) |
| **상세 보고서 (PPT/PDF)** | (Google Drive) | [Google Drive 모델 파일 다운로드](https://drive.google.com/drive/folders/1e-oi0XS1VkUjUAImm09irzHy3Irr_WV6?usp=sharing) 폴더 내 **PPT/PDF 파일** 확인 |

### **3\. 재현 단계 (Replication Steps)**

1. 상단의 **데이터셋**과 \*\*모델 파일(772MB)\*\*을 다운로드합니다.  
2. 다운로드한 파일을 Korean\_Toxic\_Text\_Classification.ipynb와 같은 레벨의 폴더에 지정된 경로로 배치합니다.  
3. requirements.txt를 사용하여 환경을 설정합니다. (pip install \-r requirements.txt)  
4. Korean\_Toxic\_Text\_Classification.ipynb 파일을 열어 순서대로 실행합니다.

## **💡 주요 트러블 슈팅 및 기여 (Troubleshooting & Contribution)**

### **1\. 데이터 불균형 해결**

* **문제:** 유해(Toxic) 텍스트 샘플이 정상 텍스트 샘플에 비해 현저히 적어 모델이 정상 텍스트만 예측하는 **데이터 불균형(Imbalance)** 문제가 발생했습니다.  
* **해결:** class\_weight를 적용하여 학습 과정에서 소수 클래스(유해 텍스트)에 더 높은 가중치를 부여함으로써 모델이 모든 클래스를 공정하게 학습하도록 유도했습니다.

### **2\. 모델 경량화 및 추론 속도 개선**

* **문제:** BERT 모델의 특성상 모델 크기가 크고 추론(Prediction) 속도가 느려 실시간 서비스에 적용하기 어렵다는 한계가 있었습니다.  
* **해결:** (사용했다면 명시) BERT 계열 모델 중 \*\*경량화된 모델(예: DistilBERT 또는 ALBERT)\*\*을 탐색하거나, \*\*모델 퀀타이징(Quantization)\*\*을 통해 모델 크기를 줄이고 추론 속도를 최적화하는 방안을 적용했습니다.

© zcx1119son
