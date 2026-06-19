"""시계열 이상탐지 웹앱 (Streamlit).

2단계: 견고한 로딩(자동 컬럼 인식·결측 처리·파일 가드) + 파일 해시 기반 캐싱
        + 자동 EDA(기초통계·결측·상관) + 라벨 자동 감지.
이후 단계에서 이상탐지 / 평가 대시보드 / 드릴다운을 추가한다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src import detect, eda, evaluate, io_utils, viz

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
    "(로딩·캐싱 · 자동 EDA · 다중 방법 이상탐지 · 방법 비교/앙상블 · 라벨 유무별 평가 대시보드)"
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
tab_data, tab_eda, tab_detect, tab_eval = st.tabs(
    ["📋 데이터 & 그래프", "🔍 자동 EDA", "🚨 이상탐지", "📊 평가 대시보드"]
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

    used = result.used_methods
    multi = len(used) > 1
    score_names = {"ensemble": "앙상블(평균)", **detect.METHODS}

    # 공통 임계값(분위수) — 방법 비교와 최종 판정에 함께 적용
    quantile = st.slider(
        "임계값(분위수) — 각 점수의 이 분위수 이상을 이상으로 본다",
        0.80, 0.999, 0.95, 0.005,
        help="방법 비교·최종 판정에 공통 적용. 5단계에서 라벨/분포 기반 인터랙티브 평가로 확장됩니다.",
    )

    x_axis = bundle.df[bundle.time_col] if bundle.time_col else None
    per_flags = detect.per_method_flags(result.scores, used, quantile)

    # ----- 방법 비교 (방법이 2개 이상일 때) -----
    if multi:
        st.markdown("##### 🔬 방법 비교")
        st.caption(
            "방법별 이상 점수를 겹쳐 보고(아래), 같은 임계값에서 각 방법이 몇 개를 "
            "잡는지·서로 얼마나 일치하는지 비교합니다."
        )
        st.plotly_chart(
            viz.score_overlay(result.scores, used, score_names, x=x_axis),
            use_container_width=True,
        )

        comp_rows = []
        for m in used:
            nf = int(per_flags[m].sum())
            comp_rows.append({
                "방법": detect.METHODS.get(m, m),
                "탐지 수": nf,
                "탐지 비율(%)": round(nf / len(per_flags[m]) * 100, 2),
            })
        union_f = detect.combine_flags(per_flags, "union")
        inter_f = detect.combine_flags(per_flags, "intersection")
        for label, f in [("합집합(OR)", union_f), ("교집합(AND)", inter_f)]:
            comp_rows.append({
                "방법": label,
                "탐지 수": int(f.sum()),
                "탐지 비율(%)": round(int(f.sum()) / len(f) * 100, 2),
            })

        ccomp, cagree = st.columns([1.3, 1])
        with ccomp:
            st.dataframe(
                pd.DataFrame(comp_rows).set_index("방법"),
                use_container_width=True,
            )
        with cagree:
            if len(used) == 2:
                a, b = per_flags[used[0]], per_flags[used[1]]
                inter_n = int((a & b).sum())
                union_n = int((a | b).sum())
                st.metric("두 방법 모두 탐지 (교집합)", f"{inter_n:,}")
                st.metric("어느 한쪽만 탐지", f"{union_n - inter_n:,}")
                st.metric("일치도 (Jaccard)", f"{detect.jaccard(a, b):.3f}")
            st.caption("Jaccard = 교집합 / 합집합. 1에 가까울수록 두 방법이 같은 지점을 잡습니다.")

    # ----- 최종 판정 & 결과 -----
    st.markdown("##### 🚩 이상 판정")
    mode_options = (["mean", "union", "intersection"] if multi else []) + used
    mode_names = {
        "mean": "앙상블 · 평균 점수",
        "union": "앙상블 · 합집합(OR) — 민감",
        "intersection": "앙상블 · 교집합(AND) — 보수적",
        **detect.METHODS,
    }
    mode = st.selectbox(
        "판정 방식",
        options=mode_options,
        format_func=lambda k: mode_names.get(k, k),
        help=(
            "평균=점수 평균에 임계값 적용 · 합집합=한 방법이라도 잡으면 이상 · "
            "교집합=모든 방법이 잡아야 이상"
        ),
    )

    if mode == "mean":
        score = result.scores["ensemble"]
        flags = detect.flags_from_scores(score, quantile)
        thr = detect.threshold_value(score, quantile)
        score_label = "앙상블(평균) 점수"
    elif mode in ("union", "intersection"):
        flags = detect.combine_flags(per_flags, mode)
        score = result.scores["ensemble"]
        thr = None  # 방법별 개별 임계라 단일 임계선 없음
        score_label = "앙상블(평균) 점수 · 참고"
    else:  # 개별 방법
        score = result.scores[mode]
        flags = per_flags[mode]
        thr = detect.threshold_value(score, quantile)
        score_label = f"{detect.METHODS.get(mode, mode)} 점수"

    n_flag = int(flags.sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("탐지된 이상 시점", f"{n_flag:,}")
    m2.metric("탐지 비율", f"{n_flag / len(flags) * 100:.2f}%")
    m3.metric("임계값(점수)", f"{thr:.3f}" if thr is not None else "방법별 개별")

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
            score_label=score_label,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("표시할 변수를 한 개 이상 선택하세요.")

# --------------------------------------------------------------------------- #
# 평가 대시보드 탭 (라벨 유무로 분기 + 대화형 임계값 슬라이더)
#   detect 탭에서 만든 result / score_names 를 그대로 재사용한다.
#   (Streamlit은 탭 본문이 모두 같은 실행에서 돌므로 모듈 스코프 변수 공유 가능)
# --------------------------------------------------------------------------- #
with tab_eval:
    st.markdown("##### 📊 탐지 결과 평가")

    # 평가할 연속 점수 선택 (앙상블 평균 또는 개별 방법)
    eval_options = (["ensemble"] if len(result.used_methods) > 1 else []) + result.used_methods
    ce1, ce2 = st.columns([1, 1.4])
    with ce1:
        eval_key = st.selectbox(
            "평가할 이상 점수(연속값)",
            options=eval_options,
            format_func=lambda k: score_names.get(k, k),
            key="eval_score",
            help="ROC/PR·분포는 연속 점수 기준으로 계산됩니다.",
        )
    with ce2:
        eval_q = st.slider(
            "임계값(분위수) — 슬라이더를 움직이면 지표가 실시간 재계산",
            0.50, 0.999, 0.95, 0.005, key="eval_q",
        )
    eval_score = result.scores[eval_key]
    eval_thr = detect.threshold_value(eval_score, eval_q)

    # 배포 스테일/모듈 누락 방어: 평가 차트 함수가 모두 로드됐는지 확인.
    # (Streamlit Cloud가 푸시 후 서브모듈을 재임포트하지 않으면 옛 viz가 남아
    #  AttributeError가 날 수 있음 → raw 트레이스백 대신 안내 메시지로 대체)
    _need = ("roc_curve_fig", "pr_curve_fig", "confusion_fig", "score_distribution")
    _missing = [fn for fn in _need if not hasattr(viz, fn)]
    if _missing:
        st.error(
            "평가 대시보드 모듈이 최신 상태로 로드되지 않았습니다"
            f" (누락: {', '.join(_missing)}). "
            "배포 환경이라면 우측 하단 **Manage app → Reboot app** 으로 "
            "앱을 재시작해 주세요."
        )
        st.stop()

    if bundle.label_col:
        # ----------------------- 라벨 있음: 정답 기반 평가 ----------------------- #
        st.success(f"✅ 라벨 **`{bundle.label_col}`** 기준 정답 평가")
        y_true = bundle.df[bundle.label_col]
        met = evaluate.labeled_metrics(y_true, eval_score, eval_q)

        st.caption(
            f"평가 대상 {met['n']:,}개 시점 중 실제 이상(양성) {met['n_pos']:,}개 "
            f"({met['n_pos'] / met['n'] * 100:.2f}%) · 점수 NaN 구간 제외"
        )

        if met["single_class"]:
            st.warning(
                "라벨이 한 종류뿐이라(전부 정상 또는 전부 이상) AUC/곡선을 계산할 수 없습니다. "
                "아래 임계값 기반 지표만 표시합니다."
            )
        else:
            t1, t2, t3 = st.columns(3)
            t1.metric("ROC-AUC", f"{met['roc_auc']:.3f}")
            t2.metric("PR-AUC (AP)", f"{met['pr_auc']:.3f}")
            t3.metric("양성 비율(기준선)", f"{met['baseline']:.3f}")

        st.markdown(f"**현재 임계값(분위수 {eval_q:.3f}, 점수 {met['threshold']:.3f}) 기준**")
        p1, p2, p3 = st.columns(3)
        p1.metric("정밀도 (Precision)", f"{met['precision']:.3f}")
        p2.metric("재현율 (Recall)", f"{met['recall']:.3f}")
        p3.metric("F1", f"{met['f1']:.3f}")

        if not met["single_class"]:
            gcol1, gcol2 = st.columns(2)
            with gcol1:
                fpr, tpr = met["roc"]
                st.plotly_chart(viz.roc_curve_fig(fpr, tpr, met["roc_auc"]),
                                use_container_width=True)
            with gcol2:
                rec, prec = met["pr"]
                st.plotly_chart(viz.pr_curve_fig(rec, prec, met["pr_auc"], met["baseline"]),
                                use_container_width=True)

        dcol1, dcol2 = st.columns([1, 1.3])
        with dcol1:
            st.plotly_chart(viz.confusion_fig(met["cm"]), use_container_width=True)
        with dcol2:
            st.plotly_chart(
                viz.score_distribution(eval_score, eval_thr, labels=y_true),
                use_container_width=True,
            )
        st.caption(
            "💡 임계값(분위수)을 낮추면 재현율↑·정밀도↓, 높이면 그 반대입니다. "
            "ROC/PR-AUC는 임계값과 무관한 전체 성능 지표입니다."
        )

    else:
        # ----------------------- 라벨 없음: 분포 + 대화형 임계값 ----------------------- #
        st.info(
            "ℹ️ 정답 라벨이 없어 정확도 지표 대신 **점수 분포 + 대화형 임계값**으로 "
            "민감도를 직접 조절·평가합니다."
        )
        flags_eval = detect.flags_from_scores(eval_score, eval_q)
        n_flag = int(flags_eval.sum())
        valid_n = int(eval_score.notna().sum())

        u1, u2, u3 = st.columns(3)
        u1.metric("임계값(점수)", f"{eval_thr:.3f}")
        u2.metric("탐지된 이상 시점", f"{n_flag:,}")
        u3.metric("탐지 비율", f"{n_flag / valid_n * 100:.2f}%" if valid_n else "—")

        st.plotly_chart(
            viz.score_distribution(eval_score, eval_thr, labels=None),
            use_container_width=True,
        )
        st.caption(
            "💡 빨간 선(임계값) 오른쪽이 '이상'으로 판정됩니다. "
            "슬라이더를 올리면 더 보수적으로(이상 적게), 내리면 더 민감하게 탐지합니다."
        )
