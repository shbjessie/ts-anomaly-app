"""시계열 이상탐지 웹앱 (Streamlit).

2단계: 견고한 로딩(자동 컬럼 인식·결측 처리·파일 가드) + 파일 해시 기반 캐싱
        + 자동 EDA(기초통계·결측·상관) + 라벨 자동 감지.
이후 단계에서 이상탐지 / 평가 대시보드 / 드릴다운을 추가한다.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src import detect, eda, io_utils, viz

# --------------------------------------------------------------------------- #
# 페이지 설정
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="시계열 이상탐지",
    page_icon="📈",
    layout="wide",
)

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLES = {
    "샘플 1 · 라벨 있음 (with labels)": SAMPLE_DIR / "sample_with_labels.csv",
    "샘플 2 · 라벨 없음 (no labels)": SAMPLE_DIR / "sample_no_labels.csv",
}

st.title("📈 시계열 이상탐지 대시보드")
st.caption(
    "다변량 시계열 CSV를 업로드하면 자동으로 분석합니다. "
    "(현재 2단계: 견고한 로딩 · 해시 캐싱 · 자동 EDA)"
)


# --------------------------------------------------------------------------- #
# 캐시: 파일 내용 해시를 키로 사용 -> 파일이 바뀌면 자동 재분석
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="📥 데이터 로딩·전처리 중...")
def load_cached(content_hash: str, _raw_bytes: bytes) -> io_utils.DataBundle:
    # _raw_bytes는 밑줄 접두사라 캐시 키에서 제외됨 -> 캐시 키 = content_hash
    return io_utils.load_bundle(content_hash, _raw_bytes)


@st.cache_data(show_spinner="🚨 이상탐지 수행 중...")
def detect_cached(
    content_hash: str,
    _bundle: io_utils.DataBundle,
    methods: tuple[str, ...],
    train_ratio: float,
    lags: int,
    kmeans_window: int,
    kmeans_k: int,
) -> detect.DetectionResult:
    # 캐시 키 = (파일 해시 + 탐지 파라미터). _bundle은 해시 제외.
    return detect.run_detection(
        _bundle,
        list(methods),
        train_ratio=train_ratio,
        lags=lags,
        kmeans_window=kmeans_window,
        kmeans_k=kmeans_k,
    )


# --------------------------------------------------------------------------- #
# 사이드바: 데이터 소스
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1. 데이터 입력")
    source_type = st.radio("데이터 소스", ["CSV 업로드", "데모 샘플 사용"], index=0)

    raw_bytes: bytes | None = None
    if source_type == "CSV 업로드":
        uploaded = st.file_uploader("CSV 파일 선택", type=["csv"])
        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
    else:
        sample_name = st.selectbox("샘플 선택", list(SAMPLES.keys()))
        raw_bytes = SAMPLES[sample_name].read_bytes()


if raw_bytes is None:
    st.info("⬅️ 왼쪽 사이드바에서 CSV를 업로드하거나 데모 샘플을 선택하세요.")
    st.stop()

# --------------------------------------------------------------------------- #
# 로딩 (해시 캐싱 + 가드)
# --------------------------------------------------------------------------- #
content_hash = io_utils.compute_hash(raw_bytes)
try:
    bundle = load_cached(content_hash, raw_bytes)
except io_utils.CsvLoadError as e:
    st.error(f"❌ 이 파일은 분석할 수 없습니다: {e}")
    st.stop()
except Exception as e:  # 예기치 못한 오류 최종 가드
    st.error(f"❌ 처리 중 오류가 발생했습니다: {e}")
    st.stop()

# --------------------------------------------------------------------------- #
# 분석 설정 요약
# --------------------------------------------------------------------------- #
st.subheader("데이터 개요")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("행 수", f"{bundle.n_rows:,}")
c2.metric("수치 변수", f"{len(bundle.numeric_cols)}")
c3.metric("시간 컬럼", bundle.time_col or "미검출")
c4.metric("라벨(정답)", bundle.label_col or "없음")
c5.metric("파일 해시", content_hash)

if bundle.label_col:
    st.success(
        f"✅ 라벨 컬럼 **`{bundle.label_col}`** 을(를) 감지했습니다. "
        "→ 이후 단계에서 정답 기반 평가(AUC/F1 등)를 제공합니다."
    )
else:
    st.info(
        "ℹ️ 라벨(정답) 컬럼이 없습니다. "
        "→ 이후 단계에서 임계값 슬라이더 기반 평가를 제공합니다."
    )

if bundle.time_col:
    st.caption(
        f"🕒 시간 범위: **{bundle.df[bundle.time_col].min()}** ~ "
        f"**{bundle.df[bundle.time_col].max()}**"
    )
if bundle.notes:
    with st.expander("🔧 자동 전처리 내역", expanded=False):
        for note in bundle.notes:
            st.write(f"- {note}")

# --------------------------------------------------------------------------- #
# 탭: 데이터 미리보기 / 자동 EDA
# --------------------------------------------------------------------------- #
tab_data, tab_eda, tab_detect = st.tabs(
    ["📋 데이터 & 그래프", "🔍 자동 EDA", "🚨 이상탐지"]
)

with tab_data:
    st.markdown("##### 표 미리보기 (상위 100행)")
    st.dataframe(bundle.df.head(100), use_container_width=True)

    st.markdown("##### 시계열 그래프")
    selected = st.multiselect(
        "표시할 변수 선택",
        options=bundle.numeric_cols,
        default=bundle.numeric_cols[: min(4, len(bundle.numeric_cols))],
    )
    if selected:
        fig = viz.line_chart(bundle.df, bundle.time_col, selected)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("표시할 변수를 한 개 이상 선택하세요.")

with tab_eda:
    cols = bundle.numeric_cols

    st.markdown("##### 1) 기초 통계")
    st.dataframe(eda.basic_stats(bundle.df, cols), use_container_width=True)

    st.markdown("##### 2) 결측치 현황 (원본 기준)")
    miss = eda.missing_summary(bundle.raw_df, cols)
    if miss["결측수"].sum() == 0:
        st.success("결측치가 없습니다.")
    else:
        mcol1, mcol2 = st.columns([1, 1.4])
        with mcol1:
            st.dataframe(miss, use_container_width=True)
        with mcol2:
            st.plotly_chart(viz.missing_bar(miss), use_container_width=True)
        st.caption("※ 분석용 데이터에서는 위 결측을 보간/채움으로 처리했습니다.")

    st.markdown("##### 3) 변수 간 상관관계")
    if len(cols) >= 2:
        st.plotly_chart(
            viz.correlation_heatmap(eda.correlation(bundle.df, cols)),
            use_container_width=True,
        )
    else:
        st.info("상관관계를 보려면 수치 변수가 2개 이상 필요합니다.")

with tab_detect:
    st.markdown("##### 탐지 설정")
    cset1, cset2 = st.columns([1.3, 1])
    with cset1:
        method_labels = st.multiselect(
            "탐지 방법 (여러 개 선택 시 앙상블)",
            options=list(detect.METHODS.keys()),
            default=list(detect.METHODS.keys()),
            format_func=lambda k: detect.METHODS[k],
        )
    with cset2:
        train_ratio = st.slider(
            "학습 구간 비율", 0.2, 0.8, 0.5, 0.05,
            help="앞쪽 이 비율만큼을 '정상'으로 보고 모델/스케일러를 학습합니다.",
        )

    with st.expander("⚙️ 고급 하이퍼파라미터", expanded=False):
        a1, a2, a3 = st.columns(3)
        lags = a1.number_input("예측 lags", 1, 168, 24, help="예측기가 참고할 과거 시점 수")
        kmeans_window = a2.number_input("KMeans window", 1, 96, 12)
        kmeans_k = a3.number_input("KMeans k(군집 수)", 2, 30, 6)

    if not method_labels:
        st.info("탐지 방법을 한 개 이상 선택하세요.")
        st.stop()

    result = detect_cached(
        content_hash, bundle, tuple(method_labels),
        train_ratio, int(lags), int(kmeans_window), int(kmeans_k),
    )

    if result.errors:
        for m, msg in result.errors.items():
            st.warning(f"'{detect.METHODS.get(m, m)}' 실행 실패: {msg}")
    if not result.used_methods:
        st.error("탐지에 성공한 방법이 없습니다. 파라미터를 조정해 보세요.")
        st.stop()

    # 어떤 점수로 이상 판정할지 (앙상블 또는 개별 방법)
    score_options = (["ensemble"] if len(result.used_methods) > 1 else []) + result.used_methods
    score_names = {"ensemble": "앙상블(평균)", **detect.METHODS}
    sc1, sc2 = st.columns([1, 1.4])
    with sc1:
        score_key = st.selectbox(
            "이상 판정에 사용할 점수",
            options=score_options,
            format_func=lambda k: score_names.get(k, k),
        )
    with sc2:
        quantile = st.slider(
            "임계값(분위수) — 이 분위수 이상을 이상으로 표시",
            0.80, 0.999, 0.95, 0.005,
            help="5단계에서 라벨/분포 기반 인터랙티브 평가로 확장됩니다.",
        )

    score = result.scores[score_key]
    flags = detect.flags_from_scores(score, quantile)
    thr = detect.threshold_value(score, quantile)

    n_flag = int(flags.sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("탐지된 이상 시점", f"{n_flag:,}")
    m2.metric("탐지 비율", f"{n_flag / len(flags) * 100:.2f}%")
    m3.metric("임계값(점수)", f"{thr:.3f}")

    st.markdown("##### 탐지 결과")
    overview_vars = st.multiselect(
        "함께 볼 변수",
        options=bundle.numeric_cols,
        default=bundle.numeric_cols[: min(3, len(bundle.numeric_cols))],
        key="detect_vars",
    )
    if overview_vars:
        fig = viz.anomaly_overview(
            bundle.df, bundle.time_col, overview_vars,
            score=score, flags=flags, threshold=thr,
            score_label=score_names.get(score_key, score_key),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("표시할 변수를 한 개 이상 선택하세요.")
