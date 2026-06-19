"""plotly 기반 시각화 함수 모음.

1단계에서는 기본 시계열 라인 차트만 제공한다.
(이상 구간 하이라이트, ROC/PR, 기여도 차트는 이후 단계에서 추가.)
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def line_chart(df: pd.DataFrame, time_col: str | None, columns: list[str]) -> go.Figure:
    """선택한 수치 변수들을 시계열 라인 차트로 그린다.

    변수마다 스케일이 다를 수 있으므로 변수별 서브플롯(공유 x축)으로 표시한다.
    """
    x = df[time_col] if time_col else df.index

    n = len(columns)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=columns,
        vertical_spacing=0.04,
    )

    for i, col in enumerate(columns, start=1):
        fig.add_trace(
            go.Scatter(x=x, y=df[col], mode="lines", name=col),
            row=i,
            col=1,
        )

    fig.update_layout(
        height=max(260, 180 * n),
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """상관행렬 히트맵."""
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            text=corr.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(title="상관"),
        )
    )
    fig.update_layout(
        height=max(300, 60 * len(corr) + 120),
        margin=dict(l=60, r=20, t=30, b=40),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _contiguous_spans(flags: pd.Series) -> list[tuple[int, int]]:
    """True 구간의 (시작위치, 끝위치) 목록을 반환 (정수 위치 기준)."""
    spans: list[tuple[int, int]] = []
    arr = flags.to_numpy()
    start = None
    for i, v in enumerate(arr):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(arr) - 1))
    return spans


def anomaly_overview(
    df: pd.DataFrame,
    time_col: str | None,
    columns: list[str],
    score: pd.Series,
    flags: pd.Series,
    threshold: float | None = None,
    score_label: str = "이상 점수",
) -> go.Figure:
    """변수별 시계열(이상 지점 빨강 마커) + 하단 이상 점수 + 임계선.

    이상 구간은 옅은 빨강 음영으로 모든 서브플롯에 표시한다.
    """
    x = df[time_col] if time_col else df.index
    n = len(columns)
    rows = n + 1
    heights = [1.0] * n + [1.2]
    total = sum(heights)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=columns + [score_label],
        vertical_spacing=0.035,
        row_heights=[h / total for h in heights],
    )

    flag_idx = flags[flags].index

    for i, col in enumerate(columns, start=1):
        fig.add_trace(
            go.Scatter(x=x, y=df[col], mode="lines", line=dict(width=1),
                       name=col, showlegend=False),
            row=i, col=1,
        )
        if len(flag_idx):
            fig.add_trace(
                go.Scatter(
                    x=x.loc[flag_idx] if hasattr(x, "loc") else x[flag_idx],
                    y=df[col].loc[flag_idx],
                    mode="markers",
                    marker=dict(color="red", size=5),
                    name="이상",
                    showlegend=False,
                ),
                row=i, col=1,
            )

    # 이상 점수 + 임계선
    fig.add_trace(
        go.Scatter(x=x, y=score, mode="lines", line=dict(color="#1f77b4", width=1),
                   name=score_label, showlegend=False),
        row=rows, col=1,
    )
    # 단일 임계값이 있을 때만 임계선 표시 (합집합/교집합 판정은 방법별 임계라 생략)
    if threshold is not None and pd.notna(threshold):
        fig.add_hline(
            y=threshold, line=dict(color="red", dash="dash"),
            row=rows, col=1,
        )

    # 이상 구간 음영 (모든 서브플롯)
    spans = _contiguous_spans(flags)
    for s, e in spans:
        x0 = x.iloc[s] if hasattr(x, "iloc") else x[s]
        x1 = x.iloc[e] if hasattr(x, "iloc") else x[e]
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="red", opacity=0.08, line_width=0,
        )

    fig.update_layout(
        height=max(360, 170 * rows),
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def score_overlay(
    scores: pd.DataFrame,
    methods: list[str],
    score_names: dict[str, str],
    x=None,
) -> go.Figure:
    """방법별 [0,1] 이상 점수를 한 그래프에 겹쳐 표시한다(+ 앙상블 평균 점선).

    방법마다 점수의 높낮이·민감 구간이 어떻게 다른지 한눈에 비교한다.
    """
    if x is None:
        x = scores.index
    fig = go.Figure()
    for m in methods:
        if m in scores:
            fig.add_trace(
                go.Scatter(x=x, y=scores[m], mode="lines",
                           name=score_names.get(m, m), line=dict(width=1.2))
            )
    if "ensemble" in scores and len(methods) > 1:
        fig.add_trace(
            go.Scatter(x=x, y=scores["ensemble"], mode="lines",
                       name=score_names.get("ensemble", "앙상블(평균)"),
                       line=dict(width=1.8, color="black", dash="dot"))
        )
    fig.update_layout(
        height=320,
        yaxis_title="이상 점수 (0~1)",
        margin=dict(l=40, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def roc_curve_fig(fpr, tpr, auc_val: float) -> go.Figure:
    """ROC 곡선 (+ 무작위 대각선)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.3f})",
        line=dict(color="#1f77b4", width=2), fill="tozeroy", fillcolor="rgba(31,119,180,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="무작위(0.5)",
        line=dict(color="gray", dash="dash"),
    ))
    fig.update_layout(
        height=360, title="ROC 곡선",
        xaxis_title="거짓양성률 (FPR)", yaxis_title="재현율 (TPR)",
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(x=0.98, y=0.05, xanchor="right", yanchor="bottom"),
    )
    return fig


def pr_curve_fig(rec, prec, ap: float, baseline: float) -> go.Figure:
    """Precision-Recall 곡선 (+ 양성비율 기준선)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rec, y=prec, mode="lines", name=f"PR (AP={ap:.3f})",
        line=dict(color="#d62728", width=2),
    ))
    fig.add_hline(
        y=baseline, line=dict(color="gray", dash="dash"),
        annotation_text=f"기준선({baseline:.3f})", annotation_position="bottom right",
    )
    fig.update_layout(
        height=360, title="Precision-Recall 곡선",
        xaxis_title="재현율 (Recall)", yaxis_title="정밀도 (Precision)",
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
    )
    return fig


def confusion_fig(cm) -> go.Figure:
    """2x2 혼동행렬 히트맵. cm=[[TN,FP],[FN,TP]]."""
    x_lab = ["예측 정상(0)", "예측 이상(1)"]
    y_lab = ["실제 정상(0)", "실제 이상(1)"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=x_lab, y=y_lab,
        colorscale="Blues", showscale=False,
        text=cm, texttemplate="%{text}", textfont=dict(size=18),
    ))
    fig.update_layout(
        height=320, title="혼동행렬",
        margin=dict(l=60, r=20, t=40, b=40),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def score_distribution(
    score: pd.Series, threshold: float | None = None, labels: pd.Series | None = None
) -> go.Figure:
    """이상 점수 히스토그램 + 임계선. 라벨이 있으면 정상/이상 분포를 겹쳐 표시."""
    s = score.dropna()
    fig = go.Figure()
    if labels is not None:
        lab = labels.reindex(s.index)
        normal = s[lab == 0]
        anom = s[lab == 1]
        fig.add_trace(go.Histogram(
            x=normal, name="실제 정상(0)", opacity=0.65,
            marker_color="#1f77b4", nbinsx=40,
        ))
        fig.add_trace(go.Histogram(
            x=anom, name="실제 이상(1)", opacity=0.65,
            marker_color="#d62728", nbinsx=40,
        ))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Histogram(
            x=s, name="이상 점수", marker_color="#1f77b4", nbinsx=40,
        ))
    if threshold is not None and pd.notna(threshold):
        fig.add_vline(
            x=threshold, line=dict(color="red", dash="dash"),
            annotation_text="임계값", annotation_position="top",
        )
    fig.update_layout(
        height=340, title="이상 점수 분포",
        xaxis_title="이상 점수 (0~1)", yaxis_title="빈도",
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def missing_bar(missing_df: pd.DataFrame) -> go.Figure:
    """변수별 결측 비율 막대그래프."""
    fig = go.Figure(
        go.Bar(
            x=missing_df.index,
            y=missing_df["결측비율(%)"],
            marker_color="#d62728",
        )
    )
    fig.update_layout(
        height=300,
        yaxis_title="결측비율(%)",
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig
