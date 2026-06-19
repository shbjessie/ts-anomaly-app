"""darts 기반 이상탐지 코어.

두 가지 계열의 방법을 제공한다.
- forecast_norm : 가벼운 LinearRegression 예측기 + NormScorer (예측 잔차 기반)
                  → 배포가 가볍고, 규칙적 패턴의 이탈 탐지에 강함 (메인)
- kmeans        : KMeansScorer (윈도우 군집 거리 기반, 비예측)
                  → 라벨 없는 임의 CSV에도 잘 동작 (옵션)

여러 방법의 점수를 [0,1]로 정규화해 한 표로 모으고, 평균 앙상블 점수도 제공한다.
(방법 간 비교/앙상블 UI는 4단계, 임계값 인터랙션은 5단계에서 확장.)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

# darts 임포트는 모듈 로드 시 무거우므로 함수 안에서 지연 임포트한다.

# 방법 식별자 -> 사람이 읽는 이름
METHODS: dict[str, str] = {
    "forecast_norm": "예측기반 (LinearRegression + Norm)",
    "kmeans": "군집기반 (KMeans)",
}


@dataclass
class DetectionResult:
    scores: pd.DataFrame          # 인덱스=행 위치(0..n-1), 컬럼=방법명 + 'ensemble'
    used_methods: list[str]       # 실제 점수가 계산된 방법 식별자
    errors: dict[str, str]        # 실패한 방법 -> 사유
    params: dict                  # 사용한 하이퍼파라미터


def _minmax(arr: np.ndarray) -> np.ndarray:
    """NaN을 무시하고 [0,1]로 정규화."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-12:
        out = np.where(np.isfinite(arr), 0.0, np.nan)
        return out
    return (arr - lo) / (hi - lo)


def run_detection(
    bundle,
    methods: list[str],
    train_ratio: float = 0.5,
    lags: int = 24,
    kmeans_window: int = 12,
    kmeans_k: int = 6,
) -> DetectionResult:
    """선택한 방법들로 이상 점수를 계산한다.

    각 방법 점수는 [0,1]로 정규화되어 scores DataFrame에 들어가고,
    계산되지 않은 앞부분(lags/window 손실)은 NaN으로 둔다.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from darts import TimeSeries
        from darts.ad import ForecastingAnomalyModel, KMeansScorer, NormScorer
        from darts.dataprocessing.transformers import Scaler
        from darts.models import LinearRegressionModel

        vals = bundle.df[bundle.numeric_cols].to_numpy(dtype=float)
        n = len(vals)
        series = TimeSeries.from_values(vals)

        # 정규화(스케일링)는 train 구간에만 fit
        n_train = max(int(n * train_ratio), 2)
        train = series[:n_train]
        scaler = Scaler()
        scaler.fit(train)
        s_all = scaler.transform(series)
        s_train = scaler.transform(train)

        scores = pd.DataFrame(index=np.arange(n))
        used: list[str] = []
        errors: dict[str, str] = {}

        def _place(score_ts) -> np.ndarray:
            """score TimeSeries -> 전체 길이 배열(앞부분 NaN)."""
            full = np.full(n, np.nan)
            pos = np.asarray(score_ts.time_index).astype(int)
            full[pos] = score_ts.values().flatten()
            return full

        for m in methods:
            try:
                if m == "forecast_norm":
                    eff_lags = max(1, min(lags, n_train - 1))
                    model = LinearRegressionModel(lags=eff_lags)
                    fam = ForecastingAnomalyModel(model=model, scorer=NormScorer())
                    fam.fit(s_train, allow_model_training=True)
                    raw = _place(fam.score(s_all))
                elif m == "kmeans":
                    eff_w = max(1, min(kmeans_window, n_train // 2 or 1))
                    eff_k = max(2, min(kmeans_k, n_train - eff_w))
                    km = KMeansScorer(window=eff_w, k=eff_k)
                    km.fit(s_train)
                    raw = _place(km.score(s_all))
                else:
                    continue
                scores[m] = _minmax(raw)
                used.append(m)
            except Exception as e:  # 방법별 실패는 격리 (앱은 계속 동작)
                errors[m] = str(e)

        if used:
            # 앙상블: 사용된 방법들의 평균(겹치는 행만)
            scores["ensemble"] = scores[used].mean(axis=1, skipna=True)

    return DetectionResult(
        scores=scores,
        used_methods=used,
        errors=errors,
        params={
            "train_ratio": train_ratio,
            "lags": lags,
            "kmeans_window": kmeans_window,
            "kmeans_k": kmeans_k,
        },
    )


def flags_from_scores(score: pd.Series, quantile: float) -> pd.Series:
    """분위수 임계값으로 이상 여부(bool)를 만든다."""
    valid = score.dropna()
    if valid.empty:
        return pd.Series(False, index=score.index)
    thr = float(valid.quantile(quantile))
    return (score >= thr).fillna(False)


def threshold_value(score: pd.Series, quantile: float) -> float:
    valid = score.dropna()
    return float(valid.quantile(quantile)) if not valid.empty else float("nan")
