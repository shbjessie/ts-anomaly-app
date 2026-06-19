"""CSV 입출력 · 컬럼 자동 인식 · 결측 처리 · 해시 캐싱 유틸.

2단계: 임의의 CSV에 견고하도록 다음을 제공한다.
- 파일 내용 해시(요구사항: 파일이 바뀌면 재분석 -> 해시 기반 캐싱)
- 타임스탬프 / 수치 / 라벨(정답) 컬럼 자동 인식
- 결측치 처리
- 잘못된 파일 가드 (빈 파일, 파싱 실패, 수치 컬럼 없음 등)
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import pandas as pd


# --------------------------------------------------------------------------- #
# 컬럼 이름 힌트
# --------------------------------------------------------------------------- #
_TIME_NAME_HINTS = (
    "time", "timestamp", "date", "datetime", "ds", "ts", "기간", "날짜", "시간",
)
_LABEL_NAME_HINTS = (
    "anomaly", "label", "is_anomaly", "target", "ground_truth", "groundtruth",
    "fault", "outlier", "abnormal", "y", "정답", "이상", "라벨",
)


class CsvLoadError(Exception):
    """CSV를 분석에 사용할 수 없을 때 발생시키는 예외."""


@dataclass
class DataBundle:
    """로딩·전처리 결과 묶음."""

    df: pd.DataFrame                 # 결측 처리까지 끝난 분석용 데이터
    raw_df: pd.DataFrame             # 시간정렬만 된 원본(결측 그대로) - EDA 결측 리포트용
    time_col: str | None
    numeric_cols: list[str]
    label_col: str | None
    content_hash: str
    n_rows: int
    n_missing_filled: int = 0        # 결측 처리로 채운 셀 수
    notes: list[str] = field(default_factory=list)  # 사용자에게 보여줄 처리 메모


# --------------------------------------------------------------------------- #
# 해시
# --------------------------------------------------------------------------- #
def compute_hash(raw_bytes: bytes) -> str:
    """파일 내용 바이트에서 짧은 SHA-256 해시를 만든다 (캐시 키)."""
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 컬럼 자동 인식
# --------------------------------------------------------------------------- #
def detect_time_column(df: pd.DataFrame) -> str | None:
    """타임스탬프로 보이는 컬럼명을 추정한다."""
    candidates: list[tuple[float, bool, str]] = []  # (파싱성공률, 이름힌트, 컬럼명)

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            return col
        # 순수 숫자형은 시간으로 오인하지 않도록 제외 (단, 이름 힌트가 있으면 허용)
        name_hint = any(h in str(col).lower() for h in _TIME_NAME_HINTS)
        if pd.api.types.is_numeric_dtype(s) and not name_hint:
            continue
        parsed = pd.to_datetime(s, errors="coerce")
        success = parsed.notna().mean()
        if success >= 0.7:
            candidates.append((success, name_hint, col))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return candidates[0][2]


def _is_binary_series(s: pd.Series) -> bool:
    """결측 제외 후 값이 {0,1} (또는 불리언) 두 종류뿐인지."""
    vals = s.dropna().unique()
    if len(vals) == 0 or len(vals) > 2:
        return False
    try:
        as_set = set(int(v) for v in vals)
    except (ValueError, TypeError):
        return set(vals) <= {True, False}
    return as_set <= {0, 1}


def detect_label_column(df: pd.DataFrame, exclude: list[str] | None = None) -> str | None:
    """정답(이상 여부) 라벨 컬럼을 추정한다.

    조건: 이진(0/1·bool) 컬럼이면서, 이름 힌트가 있으면 가장 우선.
    이름 힌트가 전혀 없으면 자동 채택하지 않는다(오탐 방지).
    """
    exclude = set(exclude or [])
    hinted: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if not _is_binary_series(df[col]):
            continue
        if any(h in str(col).lower() for h in _LABEL_NAME_HINTS):
            hinted.append(col)
    return hinted[0] if hinted else None


def detect_numeric_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """분석에 쓸 수치형 컬럼 목록."""
    exclude = set(exclude or [])
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


# --------------------------------------------------------------------------- #
# 결측 처리
# --------------------------------------------------------------------------- #
def fill_missing(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """수치 컬럼의 결측치를 시간 보간(선형) + 앞뒤 채움으로 메운다."""
    out = df.copy()
    before = int(out[cols].isna().sum().sum())
    if before:
        out[cols] = (
            out[cols]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
    after = int(out[cols].isna().sum().sum())
    return out, before - after


# --------------------------------------------------------------------------- #
# 메인 파이프라인
# --------------------------------------------------------------------------- #
def read_csv(source) -> pd.DataFrame:
    """업로드 파일 객체 또는 경로 문자열에서 DataFrame을 읽는다."""
    return pd.read_csv(source)


def load_bundle(content_hash: str, raw_bytes: bytes) -> DataBundle:
    """원본 바이트 -> 전처리 완료 DataBundle.

    content_hash는 캐시 키 용도(앱에서 st.cache_data와 함께 사용).
    잘못된 파일은 CsvLoadError로 막는다.
    """
    notes: list[str] = []

    # --- 파싱 가드 ---
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise CsvLoadError(f"CSV 파싱에 실패했습니다: {e}") from e

    if df.empty or df.shape[1] == 0:
        raise CsvLoadError("빈 CSV이거나 열이 없습니다.")

    # 완전 중복 행 제거
    dup = int(df.duplicated().sum())
    if dup:
        df = df.drop_duplicates().reset_index(drop=True)
        notes.append(f"완전 중복 행 {dup}개 제거")

    # --- 시간 컬럼 ---
    time_col = detect_time_column(df)
    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        bad_time = int(df[time_col].isna().sum())
        if bad_time:
            df = df.dropna(subset=[time_col])
            notes.append(f"시간 파싱 불가 행 {bad_time}개 제외")
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        notes.append("시간 컬럼을 찾지 못해 행 순서를 인덱스로 사용")

    # --- 라벨 컬럼 ---
    label_col = detect_label_column(df, exclude=[time_col] if time_col else [])

    # --- 수치 컬럼 (시간/라벨 제외) ---
    exclude = [c for c in (time_col, label_col) if c]
    numeric_cols = detect_numeric_columns(df, exclude=exclude)
    if not numeric_cols:
        raise CsvLoadError(
            "분석 가능한 수치형 컬럼을 찾지 못했습니다. "
            "수치 데이터가 포함된 CSV인지 확인하세요."
        )

    raw_df = df.copy()  # 결측 그대로 보존 (EDA 리포트용)

    # --- 결측 처리 (분석용 df) ---
    clean, filled = fill_missing(df, numeric_cols)
    if filled:
        notes.append(f"수치 결측 {filled}개 셀을 보간/채움 처리")

    return DataBundle(
        df=clean,
        raw_df=raw_df,
        time_col=time_col,
        numeric_cols=numeric_cols,
        label_col=label_col,
        content_hash=content_hash,
        n_rows=len(clean),
        n_missing_filled=filled,
        notes=notes,
    )
