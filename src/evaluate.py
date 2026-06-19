"""5단계: 이상탐지 결과 평가 지표.

라벨(정답)이 있는 경우의 지도학습식 평가지표를 계산한다.
- 임계값과 무관한 지표: ROC-AUC, PR-AUC(Average Precision) + ROC/PR 곡선
- 임계값(분위수) 의존 지표: Precision / Recall / F1 / 혼동행렬

라벨이 없는 경우는 점수 분포 + 대화형 임계값 재평가(app 쪽 viz로 처리)로 대신한다.
sklearn 임포트는 호출 시점으로 지연한다(앱 초기 로딩 가볍게).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned(y_true: pd.Series, score: pd.Series):
    """score가 유한한 행만 골라 (y_true[int], score[float], mask)로 정렬해 반환.

    예측 lags/윈도우 손실로 앞부분 점수가 NaN인 행은 평가에서 제외한다.
    """
    sc_arr = score.to_numpy(dtype=float)
    mask = np.isfinite(sc_arr)
    yt = y_true.to_numpy()[mask].astype(int)
    sc = sc_arr[mask]
    return yt, sc, mask


def labeled_metrics(y_true: pd.Series, score: pd.Series, quantile: float) -> dict:
    """라벨 기반 평가지표 묶음.

    반환 dict 키:
      n, n_pos, single_class, threshold,
      precision, recall, f1, cm(2x2),
      (단일 클래스가 아닐 때) roc_auc, pr_auc, roc=(fpr,tpr), pr=(recall,precision), baseline
    """
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    yt, sc, _ = _aligned(y_true, score)
    out: dict = {"n": int(len(yt)), "n_pos": int(yt.sum())}
    single_class = len(np.unique(yt)) < 2 or len(sc) == 0
    out["single_class"] = single_class

    thr = float(np.quantile(sc, quantile)) if len(sc) else float("nan")
    out["threshold"] = thr
    pred = (sc >= thr).astype(int) if len(sc) else np.array([], dtype=int)

    # 임계값 의존 지표
    out["precision"] = float(precision_score(yt, pred, zero_division=0))
    out["recall"] = float(recall_score(yt, pred, zero_division=0))
    out["f1"] = float(f1_score(yt, pred, zero_division=0))
    out["cm"] = confusion_matrix(yt, pred, labels=[0, 1])

    # 임계값 무관 지표 (양/음성 둘 다 있어야 정의됨)
    if not single_class:
        out["roc_auc"] = float(roc_auc_score(yt, sc))
        out["pr_auc"] = float(average_precision_score(yt, sc))
        fpr, tpr, _ = roc_curve(yt, sc)
        prec, rec, _ = precision_recall_curve(yt, sc)  # 반환순서: precision, recall, thr
        out["roc"] = (fpr, tpr)
        out["pr"] = (rec[::-1], prec[::-1])  # recall 오름차순으로 정리
        out["baseline"] = float(yt.mean())  # PR 곡선 기준선(양성 비율)

    return out
