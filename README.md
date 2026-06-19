# 📈 시계열 이상탐지 웹앱 (ts-anomaly-app)

다변량 시계열 CSV를 업로드하면 자동으로 분석해 **이상(anomaly)을 탐지**하고,
다양한 평가지표 대시보드로 탐지 결과의 적절성을 판단할 수 있는 Streamlit 앱입니다.

## 진행 현황 (2026-06-19 기준)

### ✅ 완료 (1~3단계)
- **1단계** — 임의의 다변량 CSV 업로드 & 미리보기, 시간/수치 컬럼 자동 인식, 시계열 그래프
- **2단계** — 견고한 로딩(라벨 자동 감지 · 결측 처리 · 잘못된 파일 가드) + **파일 내용 해시 기반 캐싱**(파일 변경 시 자동 재분석) + **자동 EDA**(기초통계 · 결측 · 상관)
- **3단계** — darts 이상탐지 코어: `LinearRegression+NormScorer`(메인) / `KMeansScorer`(옵션) + 평균 앙상블, 이상 지점·구간 하이라이트, 임계값(분위수) 슬라이더(기본형)

### ⏭️ 남은 할 일
- **배포 1차** (다음 세션 최우선) — GitHub 저장소 생성 → Streamlit Community Cloud 배포. 배포 준비 절차는 아래 "배포" 섹션 참고
- **4단계** — 다중 방법 **앙상블 강화 + 방법 간 비교 UI**(점수 오버레이, 방법별 탐지수, 일치도). Wasserstein 등 세 번째 방법 추가 검토
- **5단계 (차별화 최우선)** — 라벨 유무에 따른 **평가 대시보드 분기**
  - 라벨 有: AUC-ROC, Precision/Recall/F1, 혼동행렬, ROC/PR 곡선
  - 라벨 無: 이상 점수 분포, 탐지 비율, **대화형 임계값(민감도) 슬라이더 실시간 재평가**
- **6단계 (여유 시)** — 이상 시점 **드릴다운 + 변수별 기여도**, 결과(이상 목록) **CSV 다운로드**
- **7단계** — 마무리(예외처리 보강 · README) + 배포 최종 갱신

> 채점 우선순위: 기본기는 안정적으로 채우되 **'차별화'(특히 5단계 임계값 슬라이더·평가 분기, 그다음 4단계 비교/앙상블)** 에서 확실히 점수 확보.

## 기술 스택
Streamlit · darts · plotly · pandas · scikit-learn

## 로컬 실행 (Windows)
```powershell
# 기존 가상환경 사용
C:\Users\1204s\Desktop\ts-exam\venv\Scripts\activate
streamlit run app.py
```

## 샘플 데이터
`sample_data/` 에 데모용 CSV 2개가 있습니다.
- `sample_with_labels.csv` : 정답 라벨(`is_anomaly`) 포함
- `sample_no_labels.csv` : 라벨 없음

재생성: `python sample_data/_generate.py`

## 배포 (Streamlit Community Cloud)

의존성은 `requirements.txt`에 버전 고정, Python 버전은 `runtime.txt`(python-3.13).

### 다음 세션 배포 전 미리 준비할 것
1. **GitHub 계정** 로그인 가능 상태 (없으면 https://github.com 에서 가입)
2. **빈 저장소(repository) 1개** 준비 — 이름 예: `ts-anomaly-app`, **Private/Public 무관**, README/.gitignore 없이 비어 있게 생성
   (저장소 생성·푸시는 다음 세션에서 함께 진행 가능)
3. **Streamlit Community Cloud** 가입/로그인 — https://share.streamlit.io 에서 GitHub 계정으로 로그인
4. (선택) `gh` CLI 로그인 여부 확인: 터미널에서 `gh auth status`

### 배포 시 메인 파일 / 설정
- Main file path: `app.py`
- Python version: 3.13 (`runtime.txt`)
- darts가 무거우므로 첫 빌드는 수 분 소요될 수 있음. 메인 탐지기를 가벼운 LinearRegression으로 둔 이유.
