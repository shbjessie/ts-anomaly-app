"""자동 EDA (탐색적 데이터 분석) 요약.

업로드 즉시 데이터의 성격을 한눈에 보여주기 위한 기초 통계 · 결측 · 상관 요약.
"""
from __future__ import annotations

import pandas as pd


def basic_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """기초 통계량 표 (변수별 행)."""
    desc = df[cols].describe().T
    # 보기 좋은 한글 컬럼명
    desc = desc.rename(
        columns={
            "count": "개수",
            "mean": "평균",
            "std": "표준편차",
            "min": "최소",
            "25%": "25%",
            "50%": "중앙값",
            "75%": "75%",
            "max": "최대",
        }
    )
    # 결측/왜도 추가
    desc["결측수"] = df[cols].isna().sum()
    desc["왜도"] = df[cols].skew()
    return desc.round(3)


def missing_summary(raw_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """원본(결측 처리 전) 기준 변수별 결측 현황."""
    n = len(raw_df)
    miss = raw_df[cols].isna().sum()
    out = pd.DataFrame(
        {
            "결측수": miss,
            "결측비율(%)": (miss / n * 100).round(2),
        }
    )
    return out.sort_values("결측수", ascending=False)


def correlation(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """수치 변수 간 피어슨 상관행렬."""
    return df[cols].corr().round(3)
