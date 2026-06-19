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
    threshold: float,
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
