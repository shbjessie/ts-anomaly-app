"""데모용 다변량 시계열 CSV 2개를 생성한다.

- sample_with_labels.csv : 정답 라벨(is_anomaly) 포함
- sample_no_labels.csv   : 라벨 없음

발표 동영상/테스트용. 재현성을 위해 시드를 고정한다.
실행: python sample_data/_generate.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).resolve().parent


def make_series(n=1500, freq="h"):
    """3개 수치 변수 + 시간 인덱스를 가진 정상 패턴 생성."""
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    t = np.arange(n)

    # 일/주 주기 + 추세 + 노이즈
    temperature = (
        20
        + 5 * np.sin(2 * np.pi * t / 24)        # 하루 주기
        + 2 * np.sin(2 * np.pi * t / (24 * 7))  # 주 주기
        + rng.normal(0, 0.6, n)
    )
    pressure = (
        101.3
        + 0.5 * np.sin(2 * np.pi * t / 24 + 1.0)
        + 0.002 * t                              # 완만한 추세
        + rng.normal(0, 0.15, n)
    )
    vibration = (
        0.5
        + 0.2 * np.abs(np.sin(2 * np.pi * t / 12))
        + rng.normal(0, 0.05, n)
    )
    return idx, temperature, pressure, vibration


def inject_anomalies(temperature, pressure, vibration, n):
    """여러 유형의 이상을 주입하고 라벨을 만든다."""
    label = np.zeros(n, dtype=int)

    # 1) 점(point) 스파이크 - temperature
    for c in rng.choice(np.arange(100, n - 100), size=8, replace=False):
        temperature[c] += rng.choice([-1, 1]) * rng.uniform(8, 14)
        label[c] = 1

    # 2) 구간(collective) 이상 - pressure 급락 구간
    s = 600
    temperature[s:s + 20] += 3
    pressure[s:s + 20] -= 2.0
    label[s:s + 20] = 1

    # 3) 분산 증가(variance) 이상 - vibration
    s2 = 1000
    vibration[s2:s2 + 30] += rng.normal(0, 0.4, 30)
    label[s2:s2 + 30] = 1

    return temperature, pressure, vibration, label


def main():
    n = 1500
    idx, temp, pres, vib = make_series(n)

    # 라벨 있는 버전
    temp_a, pres_a, vib_a, label = inject_anomalies(temp.copy(), pres.copy(), vib.copy(), n)
    df_lab = pd.DataFrame(
        {
            "timestamp": idx,
            "temperature": np.round(temp_a, 3),
            "pressure": np.round(pres_a, 3),
            "vibration": np.round(vib_a, 3),
            "is_anomaly": label,
        }
    )
    # 임의 CSV 견고성 테스트용으로 결측치 약간 주입
    miss = rng.choice(np.arange(n), size=12, replace=False)
    df_lab.loc[miss, "pressure"] = np.nan
    df_lab.to_csv(OUT / "sample_with_labels.csv", index=False)

    # 라벨 없는 버전 (다른 시드 패턴, 이상 포함하되 라벨 미제공)
    idx2, temp2, pres2, vib2 = make_series(n)
    temp2, pres2, vib2, _ = inject_anomalies(temp2, pres2, vib2, n)
    df_nolab = pd.DataFrame(
        {
            "time": idx2,                 # 일부러 다른 컬럼명("time") 사용 -> 자동인식 테스트
            "sensor_a": np.round(temp2, 3),
            "sensor_b": np.round(pres2, 3),
            "sensor_c": np.round(vib2, 3),
        }
    )
    df_nolab.to_csv(OUT / "sample_no_labels.csv", index=False)

    print(f"생성 완료:")
    print(f"  - sample_with_labels.csv : {df_lab.shape}, 이상 {int(label.sum())}개")
    print(f"  - sample_no_labels.csv   : {df_nolab.shape}")


if __name__ == "__main__":
    main()
