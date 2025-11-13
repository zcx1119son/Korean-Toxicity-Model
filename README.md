# **🇰🇷 HateSpeech-Detector-NLP: 한국어 유해성 텍스트 2단계 분류 모델**

## **💡 프로젝트 개요 (Project Overview)**

본 프로젝트는 온라인상에서 발생하는 **혐오 발언, 비하, 스팸 등의 유해성 텍스트**를 자동으로 탐지하고 그 유형을 분류하는 **2단계 분류 (Two-Tier Classification)** AI 모델 개발을 목표로 합니다. 이 모델은 사용자 콘텐츠 관리 및 모더레이션 시스템의 핵심 기능을 수행하며, **AI Hub** 제공 데이터셋의 계층적 구조에 최적화되어 설계되었습니다.

- **1차 분류:** 텍스트의 유해성(Toxic) 여부 판단 (유해 / 정상)
- **2차 분류:** 1차에서 유해로 판별된 텍스트의 구체적인 유형 분류
- **저장소 URL:** https://github.com/zcx1119son/Korean-Toxicity-Model

## **⚙️ 기술 스택 (Tech Stack)**

| 구분 | 기술 / 라이브러리 | 상세 내용 |
| --- | --- | --- |
| **핵심 모델** | **BERT (KcBERT Fine-tuning)** | 한국어에 최적화된 BERT 모델 기반으로 파인튜닝하여 높은 성능 달성 |
| **딥러닝** | PyTorch, Hugging Face Transformers | 모델 구현 및 학습, 사전 학습 모델 로드 |
| **데이터 처리** | Pandas, Numpy, Scikit-learn | 데이터 전처리, 학습/검증 분할, 성능 평가 |
| **개발 환경** | Jupyter Notebook (.ipynb) | 실험 및 결과 도출 환경 |

## **✅ 최종 성능 및 결과**

BERT 기반의 2단계 분류 구조를 통해 유해성 판단 및 유형 분류에서 높은 성능을 달성했습니다.

| 분류 단계 | 모델 특징 | 평가 지표 (F1-Score 기준) |
| --- | --- | --- |
| **1차 분류 (유해/정상)** | 이진 분류 모델 | **F1-Score: 0.95 이상** |
| **2차 분류 (유형 분류)** | 다중 분류 모델 | **F1-Score: 0.88 이상** |

## 💡 트러블 슈팅 및 기여 (Troubleshooting & Contribution)

### 1. **핵심 설계 동기: 데이터셋 구조에 기반한 2단계 파이프라인**

본 모델은 **AI Hub 데이터셋의 계층적 구조**에 맞춰 문제를 해결하기 위해 2단계 파이프라인을 필수적으로 도입했습니다. 이 설계는 효율성뿐만 아니라 비즈니스 안정성을 위한 **가장 강력한 안전장치** 역할을 합니다.

- **효율적인 분류:** 모든 텍스트를 7가지 유형으로 분류하는 대신, 1차 분류를 통해 유해 텍스트에만 2차 분류 모델을 적용하여 전체 시스템의 추론 시간 효율을 높였습니다.

### 2. 기술적 난관 극복: 불균형 데이터 해결 이중 전략 (강조)

1차 분류(정상/유해) 모델 훈련 직전에 유해 텍스트가 전체 훈련 데이터에서 약 **3.8%** (정상 31,306개 vs. 유해 1,229개)에 불과한 **심각한 클래스 불균형** 문제를 확인했습니다. 이는 모델이 소수 클래스를 무시하고 편향될 위험이 있었습니다.

이 문제를 해결하기 위해 **데이터 샘플링**과 **손실 함수 가중치 부여**라는 **이중 전략(Two-Pronged Strategy)**을 병행하여 적용했습니다.

### ① 데이터 차원 해결: 정상 클래스 다운 샘플링(Down-sampling)

훈련 효율성 제고 및 불균형 완화를 위해, 압도적으로 많은 **원본 정상 텍스트 31,306개 중 무작위로 15,000개만 선택**하여 훈련 데이터셋을 구성했습니다. 이를 통해 유해 클래스(1,229개) 대비 비율을 약 **12:1** 수준으로 획기적으로 낮췄습니다.

### ② 손실 함수 차원 해결: Weighted Cross-Entropy Loss 도입

다운 샘플링 후에도 존재하는 불균형에 대응하고 오탐률(False Positive)을 최소화하기 위해 **Weighted Cross-Entropy Loss** 함수를 커스텀하여 적용했습니다.

- **가중치 계산 및 적용:** 불균형 비율에 따라 유해 클래스에 약 **13.2배** 높은 가중치(정상: 0.5196 vs. 유해: 13.2364)를 부여했습니다.
- **결과:** 이 **이중 전략의 시너지 효과** 덕분에 정상 텍스트의 오탐률(FP Rate)을 **0.41%**로 극단적으로 낮추는 데 성공했습니다. 이는 서비스 도입 시 신뢰성을 보장하는 핵심 성과입니다.

## **🛠️ 코드 및 재현 안내 (Reproduction Guide)**

프로젝트 코드, 대용량 모델 파일, 데이터셋 출처, 그리고 상세 보고서까지 **모든 자료가 아래 링크에 통합**되어 있습니다.

### **1. 📂 코드 및 의존성 파일**

- **메인 코드 (.ipynb):** [Korean_Toxic_Text_Classification.ipynb](https://github.com/zcx1119son/Korean-Toxicity-Model/blob/master/Korean_Toxic_Text_Classification.ipynb)
- **환경 설정:** requirements.txt (프로젝트 재현을 위한 필수 라이브러리 목록)

### **2. 💾 대용량 자료 다운로드 링크 (필수)**

| 자원 | 용량 | 링크 |
| --- | --- | --- |
| **학습 데이터셋 (AI Hub)** | 대용량 | [AI Hub 다운로드 페이지 (국가기록물 대상)](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EA%B5%AD%EA%B0%80%EA%B8%B0%EB%A1%9D%EB%AC%BC&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71788) |
| **학습된 모델 파일 (772MB)** | **약 772MB** | [Google Drive 모델 파일 다운로드](https://drive.google.com/drive/folders/1dL8Y7zl4BddPBDbXeHF5BwB5hf-1N_b7?usp=sharing) |
| **상세 PPT/PDF 자료** | 보조 보고서 | [Google Drive 상세 자료 링크](https://drive.google.com/drive/folders/1c32AJIo_1g993qb2vQyJK4hUfzu8aug4?usp=sharing) |

### **3. 재현 단계 (Replication Steps)**

1. **Repository Clone:** 본 GitHub 저장소를 로컬로 복제합니다.
2. **환경 설정:** requirements.txt를 사용하여 Python 환경을 설정합니다. (pip install -r requirements.txt)
3. **파일 다운로드:** 상단의 **AI Hub 데이터셋**과 **Google Drive 모델 파일**을 모두 다운로드합니다.
4. **코드 실행:** Korean_Toxic_Text_Classification.ipynb 파일을 열어 전처리, 모델 로드, 예측 결과를 순서대로 확인합니다.

## 🚀 프로젝트 회고 및 향후 개선 방향

### 1. 프로젝트를 통해 배운 점

- **기술 선택의 당위성:** 단순한 최고 성능 모델(BERT)의 적용을 넘어, **데이터셋의 특성(계층성, 불균형)을 분석**하여 2단계 파이프라인과 커스텀 손실 함수라는 **최적화된 방법론**을 도출하는 능력을 체득했습니다.
- **비즈니스 목표의 이해:** F1-Score만 높이는 것이 아니라, **정상 텍스트의 오탐률을 낮추는 것**이 실제 서비스에서의 신뢰성 확보에 가장 중요하다는 점을 이해하고, 목표 지표를 설정하여 문제를 해결했습니다.

### 2. 모델의 한계 및 향후 계획

- **현재 한계:** 2차 분류(7가지 유형)의 경우, 유형별로 데이터 희소성이 더욱 심하여 1차 분류 대비 성능(F1-Score 0.88)에 한계가 있었습니다.
- **향후 계획:** 후속 연구로 7가지 유형 분류에 대해 **Transfer Learning 기법**을 적용하거나, 각 유형별 임베딩 공간을 분리하는 **Metric Learning** 방식을 도입하여 소수 유형에 대한 분류 성능을 더욱 끌어올릴 계획입니다.

[📂 프로젝트 산출물 및 상세 기록](https://www.notion.so/2a7fc91e2372808db38ff31f3881d1ce?pvs=21)
