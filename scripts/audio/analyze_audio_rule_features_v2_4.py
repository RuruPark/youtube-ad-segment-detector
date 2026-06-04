#!/usr/bin/env python3
"""추출된 label 기반 오디오 feature를 분석해 오디오 단서 규칙 설계에 사용한다."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
TASK_NAME = "analyze_audio_rule_features"
PREFERRED_VERSION = "v2_4"
DATA_AUDIO_DIR = PROJECT_ROOT / "data/audio"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_AUDIO_DIR = PROJECT_ROOT / "scripts/audio"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"
BACKUPS_DIR = PROJECT_ROOT / "backups"
PREFERRED_FEATURE_FILE = DATA_AUDIO_DIR / "audio_labeled_segment_features_v2_4.csv"

SEGMENT_TYPES = ["random_non_ad_30s", "pre_ad_10s", "ad_full", "post_ad_10s"]
PAIRWISE_COMPARISONS = [
    ("ad_full_vs_random_non_ad_30s", "ad_full", "random_non_ad_30s"),
    ("ad_full_vs_pre_ad_10s", "ad_full", "pre_ad_10s"),
    ("ad_full_vs_post_ad_10s", "ad_full", "post_ad_10s"),
    ("pre_ad_10s_vs_post_ad_10s", "pre_ad_10s", "post_ad_10s"),
]
DELTA_TYPES = [
    ("ad_minus_random", "ad_full", "random_non_ad_30s"),
    ("ad_minus_pre", "ad_full", "pre_ad_10s"),
    ("post_minus_ad", "post_ad_10s", "ad_full"),
    ("post_minus_pre", "post_ad_10s", "pre_ad_10s"),
]
REQUIRED_COLUMNS = [
    "video_id",
    "ad_interval_id",
    "segment_type",
    "segment_start_sec",
    "segment_end_sec",
    "segment_duration_sec",
    "feature_status",
]
EXCLUDED_METADATA_COLUMNS = {
    "version",
    "source_label_file",
    "source_label_modified_time",
    "video_path",
    "video_title",
    "ad_interval_id",
    "video_id",
    "segment_id",
    "segment_type",
    "segment_start_sec",
    "segment_end_sec",
    "segment_duration_sec",
    "segment_start_mmss",
    "segment_end_mmss",
    "sampling_status",
    "sampling_warning",
    "feature_status",
    "feature_error",
    "audio_available",
    "audio_stream_index",
    "decoded_sample_rate",
    "decoded_num_samples",
    "decoded_duration_sec",
    "decoded_duration_diff_sec",
    "random_seed",
    "random_attempt_count",
    "random_boundary_buffer_sec",
    "source_ad_start_sec",
    "source_ad_end_sec",
    "source_ad_duration_sec",
    "video_duration_sec",
    "context_truncated",
    "context_overlaps_other_ad",
}
FORBIDDEN_LATEST_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".wav",
    ".mp3",
    ".m4a",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
}


class TaskLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp} {message}\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def iso_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def readable_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {sec:.1f}s"
    if minutes:
        return f"{minutes}m {sec:.1f}s"
    return f"{sec:.1f}s"


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return value
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def version_tuple(version: str) -> Tuple[int, ...]:
    nums = re.findall(r"\d+", version or "")
    return tuple(int(x) for x in nums) if nums else tuple()


def extract_version(path: Path) -> str:
    match = re.search(r"(v\d+(?:_\d+)*)", path.name)
    if match:
        return match.group(1)
    return f"latest_audio_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def old_project_snapshot() -> Dict[str, Any]:
    if not OLD_PROJECT_ROOT.exists():
        return {"path": str(OLD_PROJECT_ROOT), "exists": False, "file_count": 0, "metadata_digest": None}
    entries: List[str] = []
    file_count = 0
    for dirpath, _, filenames in os.walk(OLD_PROJECT_ROOT):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            rel = path.relative_to(OLD_PROJECT_ROOT).as_posix()
            entries.append(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}")
            file_count += 1
    import hashlib

    digest = hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()
    return {
        "path": str(OLD_PROJECT_ROOT),
        "exists": True,
        "file_count": file_count,
        "metadata_digest": digest,
        "snapshot_type": "relative_path_size_mtime_ns",
    }


def discover_feature_file() -> Tuple[Path, str, str]:
    if PREFERRED_FEATURE_FILE.exists():
        return PREFERRED_FEATURE_FILE, PREFERRED_VERSION, "v2_4 preferred feature CSV exists"
    candidates = sorted(DATA_AUDIO_DIR.glob("audio_labeled_segment_features_*.csv"))
    if not candidates:
        raise RuntimeError("No valid audio_labeled_segment_features file found")
    candidates.sort(key=lambda p: (version_tuple(extract_version(p)), p.stat().st_mtime), reverse=True)
    selected = candidates[0]
    return selected, extract_version(selected), "fallback to highest version or newest feature CSV"


def output_paths(version: str) -> Dict[str, Path]:
    return {
        "profile": DATA_AUDIO_DIR / f"audio_rule_segment_type_profile_{version}.csv",
        "pairwise": DATA_AUDIO_DIR / f"audio_rule_pairwise_feature_comparison_{version}.csv",
        "paired_delta": DATA_AUDIO_DIR / f"audio_rule_paired_context_delta_{version}.csv",
        "paired_delta_summary": DATA_AUDIO_DIR / f"audio_rule_paired_context_delta_summary_{version}.csv",
        "thresholds": DATA_AUDIO_DIR / f"audio_rule_candidate_thresholds_{version}.csv",
        "recommendations": DATA_AUDIO_DIR / f"audio_rule_feature_recommendations_{version}.csv",
        "normalized_features": DATA_AUDIO_DIR / f"audio_rule_video_normalized_features_{version}.csv",
        "normalized_pairwise": DATA_AUDIO_DIR / f"audio_rule_video_normalized_pairwise_comparison_{version}.csv",
        "report": REPORTS_DIR / f"analyze_audio_rule_features_{version}_report.json",
        "summary": REPORTS_DIR / f"analyze_audio_rule_features_{version}_summary.md",
        "run_log": LOGS_DIR / f"analyze_audio_rule_features_{version}_run_log.txt",
        "script": SCRIPTS_AUDIO_DIR / f"analyze_audio_rule_features_{version}.py",
    }


def backup_existing(paths: Dict[str, Path], version: str, timestamp: str) -> Optional[Path]:
    targets = list(paths.values()) + [LATEST_DIR / "README_latest_files.md"]
    existing = [p for p in targets if p.exists()]
    if not existing:
        return None
    backup_dir = BACKUPS_DIR / f"analyze_audio_rule_features_{version}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        if path.is_file():
            shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def feature_group(feature: str) -> str:
    if feature.startswith("audio_rms") or feature.startswith("audio_log_energy") or feature in {
        "audio_peak_amplitude",
        "audio_mean_abs_amplitude",
    }:
        return "Energy / Amplitude"
    if feature in {"silence_ratio", "low_energy_ratio"}:
        return "Silence / Low Energy"
    if feature.startswith("zero_crossing_rate"):
        return "Zero Crossing"
    if feature.startswith("spectral_flux"):
        return "Spectral Flux"
    if feature.startswith("spectral_"):
        return "Spectral"
    if feature.startswith("mfcc_"):
        return "MFCC"
    if feature.startswith("onset_"):
        return "Onset"
    if feature == "tempo_estimate":
        return "Optional"
    return "Other Audio"


def select_features(df: pd.DataFrame) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, float], Dict[str, int], List[str]]:
    numeric_candidates: List[str] = []
    excluded: List[Dict[str, Any]] = []
    nan_ratio: Dict[str, float] = {}
    inf_count: Dict[str, int] = {}
    constant_features: List[str] = []
    cleaned = df.copy()
    for col in df.columns:
        if col in EXCLUDED_METADATA_COLUMNS:
            excluded.append({"feature_name": col, "reason": "metadata_or_status_column"})
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        non_null = numeric.notna().sum()
        if non_null == 0:
            excluded.append({"feature_name": col, "reason": "non_numeric_or_all_nan"})
            continue
        numeric_candidates.append(col)
        arr = numeric.to_numpy(dtype=float)
        inf_count[col] = int(np.isinf(arr).sum())
        if inf_count[col]:
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            cleaned[col] = numeric
        nan_ratio[col] = float(numeric.isna().mean())
        finite = numeric[np.isfinite(numeric)]
        if nan_ratio[col] >= 0.5:
            excluded.append({"feature_name": col, "reason": f"nan_ratio_ge_0.5 ({nan_ratio[col]:.3f})"})
            continue
        if len(finite) == 0:
            excluded.append({"feature_name": col, "reason": "no_finite_values"})
            continue
        if finite.nunique(dropna=True) <= 1:
            constant_features.append(col)
            excluded.append({"feature_name": col, "reason": "constant_feature"})
            continue
    analyzed = [col for col in numeric_candidates if col not in {x["feature_name"] for x in excluded if x["reason"] != "metadata_or_status_column"}]
    return numeric_candidates, analyzed, excluded, nan_ratio, inf_count, constant_features


def profile_by_segment(df: pd.DataFrame, features: Sequence[str], version: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for segment_type in SEGMENT_TYPES:
        group = df[df["segment_type"] == segment_type]
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce")
            finite = values[np.isfinite(values)]
            q25 = finite.quantile(0.25) if len(finite) else np.nan
            q75 = finite.quantile(0.75) if len(finite) else np.nan
            rows.append(
                {
                    "version": version,
                    "segment_type": segment_type,
                    "feature_name": feature,
                    "feature_group": feature_group(feature),
                    "count": int(len(group)),
                    "valid_count": int(len(finite)),
                    "missing_count": int(values.isna().sum()),
                    "missing_ratio": safe_float(values.isna().mean()) if len(values) else np.nan,
                    "mean": safe_float(finite.mean()) if len(finite) else np.nan,
                    "std": safe_float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                    "median": safe_float(finite.median()) if len(finite) else np.nan,
                    "q25": safe_float(q25),
                    "q75": safe_float(q75),
                    "iqr": safe_float(q75 - q25) if np.isfinite(q25) and np.isfinite(q75) else np.nan,
                    "min": safe_float(finite.min()) if len(finite) else np.nan,
                    "max": safe_float(finite.max()) if len(finite) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return safe_float((a.mean() - b.mean()) / math.sqrt(pooled))


def effect_level(effect: float) -> str:
    if not np.isfinite(effect):
        return "unknown"
    mag = abs(effect)
    if mag >= 0.8:
        return "large"
    if mag >= 0.5:
        return "medium"
    if mag >= 0.2:
        return "small"
    return "weak"


def robust_effect(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    iqr_a = a.quantile(0.75) - a.quantile(0.25)
    iqr_b = b.quantile(0.75) - b.quantile(0.25)
    pooled_iqr = np.nanmean([iqr_a, iqr_b])
    if not np.isfinite(pooled_iqr) or pooled_iqr == 0:
        return np.nan
    return safe_float((a.median() - b.median()) / pooled_iqr)


def interpretation_hint(feature: str, direction: str, comparison: str) -> str:
    group = feature_group(feature)
    if direction == "weak_or_unclear":
        return "차이가 작거나 방향성이 약해 단독 rule보다는 낮은 우선순위로 해석"
    if group == "Energy / Amplitude":
        cue = "음량/에너지"
    elif group == "Silence / Low Energy":
        cue = "무음 또는 저에너지 비율"
    elif group in {"Spectral", "Spectral Flux"}:
        cue = "스펙트럼 질감 또는 편집/BGM 변화"
    elif group == "Onset":
        cue = "onset/변화 강도"
    elif group == "MFCC":
        cue = "음색 계열 보조 단서"
    else:
        cue = group
    return f"{comparison}에서 {cue} 차이를 보이는 탐색적 audio cue"


def pairwise_comparison(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for comparison_name, seg_a, seg_b in PAIRWISE_COMPARISONS:
        a_df = df[df["segment_type"] == seg_a]
        b_df = df[df["segment_type"] == seg_b]
        for feature in features:
            a = pd.to_numeric(a_df[feature], errors="coerce")
            b = pd.to_numeric(b_df[feature], errors="coerce")
            a_f = a[np.isfinite(a)]
            b_f = b[np.isfinite(b)]
            mean_a = a_f.mean() if len(a_f) else np.nan
            mean_b = b_f.mean() if len(b_f) else np.nan
            median_a = a_f.median() if len(a_f) else np.nan
            median_b = b_f.median() if len(b_f) else np.nan
            mean_diff = mean_a - mean_b if np.isfinite(mean_a) and np.isfinite(mean_b) else np.nan
            median_diff = median_a - median_b if np.isfinite(median_a) and np.isfinite(median_b) else np.nan
            relative = mean_diff / abs(mean_b) if np.isfinite(mean_diff) and np.isfinite(mean_b) and abs(mean_b) > 1e-12 else np.nan
            d = cohen_d(a, b)
            r = robust_effect(a, b)
            if not np.isfinite(d) or abs(d) < 0.2:
                direction = "weak_or_unclear"
            elif mean_diff > 0:
                direction = "higher_in_a"
            else:
                direction = "lower_in_a"
            rows.append(
                {
                    "comparison_name": comparison_name,
                    "feature_name": feature,
                    "feature_group": feature_group(feature),
                    "segment_type_a": seg_a,
                    "segment_type_b": seg_b,
                    "n_a": int(len(a_f)),
                    "n_b": int(len(b_f)),
                    "mean_a": safe_float(mean_a),
                    "mean_b": safe_float(mean_b),
                    "median_a": safe_float(median_a),
                    "median_b": safe_float(median_b),
                    "mean_diff_a_minus_b": safe_float(mean_diff),
                    "median_diff_a_minus_b": safe_float(median_diff),
                    "relative_mean_diff": safe_float(relative),
                    "cohen_d": safe_float(d),
                    "effect_size_level": effect_level(d),
                    "robust_effect_size": safe_float(r),
                    "direction": direction,
                    "abs_effect_rank": np.nan,
                    "missing_ratio_a": safe_float(a.isna().mean()) if len(a) else np.nan,
                    "missing_ratio_b": safe_float(b.isna().mean()) if len(b) else np.nan,
                    "interpretation_hint": interpretation_hint(feature, direction, comparison_name),
                }
            )
    out = pd.DataFrame(rows)
    out["abs_cohen_d"] = out["cohen_d"].abs()
    out["abs_effect_rank"] = out.groupby("comparison_name")["abs_cohen_d"].rank(method="first", ascending=False)
    out = out.drop(columns=["abs_cohen_d"])
    return out


def paired_delta(df: pd.DataFrame, features: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    index_cols = ["ad_interval_id", "video_id"]
    rows: List[Dict[str, Any]] = []
    for ad_interval_id, group in df.groupby("ad_interval_id"):
        video_id = group["video_id"].iloc[0]
        by_type = {row["segment_type"]: row for _, row in group.iterrows()}
        for delta_type, seg_a, seg_b in DELTA_TYPES:
            if seg_a not in by_type or seg_b not in by_type:
                continue
            row_a = by_type[seg_a]
            row_b = by_type[seg_b]
            for feature in features:
                value_a = safe_float(row_a.get(feature))
                value_b = safe_float(row_b.get(feature))
                delta = value_a - value_b if np.isfinite(value_a) and np.isfinite(value_b) else np.nan
                rel = delta / abs(value_b) if np.isfinite(delta) and np.isfinite(value_b) and abs(value_b) > 1e-12 else np.nan
                direction = "positive" if np.isfinite(delta) and delta > 0 else "negative" if np.isfinite(delta) and delta < 0 else "zero_or_missing"
                rows.append(
                    {
                        "ad_interval_id": ad_interval_id,
                        "video_id": video_id,
                        "feature_name": feature,
                        "feature_group": feature_group(feature),
                        "delta_type": delta_type,
                        "segment_type_a": seg_a,
                        "segment_type_b": seg_b,
                        "value_a": value_a,
                        "value_b": value_b,
                        "delta_a_minus_b": safe_float(delta),
                        "abs_delta": safe_float(abs(delta)) if np.isfinite(delta) else np.nan,
                        "relative_delta": safe_float(rel),
                        "delta_direction": direction,
                    }
                )
    delta_df = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []
    for (delta_type, feature), group in delta_df.groupby(["delta_type", "feature_name"]):
        vals = pd.to_numeric(group["delta_a_minus_b"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        abs_vals = vals.abs()
        q25 = vals.quantile(0.25) if len(vals) else np.nan
        q75 = vals.quantile(0.75) if len(vals) else np.nan
        std = vals.std(ddof=1) if len(vals) > 1 else np.nan
        paired = vals.mean() / std if np.isfinite(std) and std > 0 else np.nan
        summary_rows.append(
            {
                "delta_type": delta_type,
                "feature_name": feature,
                "feature_group": feature_group(feature),
                "pair_count": int(len(vals)),
                "delta_mean": safe_float(vals.mean()) if len(vals) else np.nan,
                "delta_std": safe_float(std),
                "delta_median": safe_float(vals.median()) if len(vals) else np.nan,
                "delta_q25": safe_float(q25),
                "delta_q75": safe_float(q75),
                "abs_delta_mean": safe_float(abs_vals.mean()) if len(abs_vals) else np.nan,
                "abs_delta_median": safe_float(abs_vals.median()) if len(abs_vals) else np.nan,
                "positive_delta_ratio": safe_float((vals > 0).mean()) if len(vals) else np.nan,
                "negative_delta_ratio": safe_float((vals < 0).mean()) if len(vals) else np.nan,
                "paired_effect_size": safe_float(paired),
                "interpretation_hint": "같은 ad_interval_id 내부 context 차이를 줄인 탐색적 변화량 단서",
            }
        )
    return delta_df, pd.DataFrame(summary_rows)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else np.nan
    balanced = np.nanmean([recall, specificity])
    return {
        "in_sample_balanced_accuracy": safe_float(balanced),
        "in_sample_precision": safe_float(precision),
        "in_sample_recall": safe_float(recall),
        "in_sample_f1": safe_float(f1),
    }


def threshold_analysis(df: pd.DataFrame, features: Sequence[str], pairwise: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["segment_type"].isin(["ad_full", "random_non_ad_30s"])].copy()
    y = (subset["segment_type"] == "ad_full").astype(int).to_numpy()
    effect_map = pairwise[pairwise["comparison_name"] == "ad_full_vs_random_non_ad_30s"].set_index("feature_name")
    rows: List[Dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(subset[feature], errors="coerce")
        valid = np.isfinite(values.to_numpy(dtype=float))
        x = values.to_numpy(dtype=float)[valid]
        y_valid = y[valid]
        pos_count = int((y_valid == 1).sum())
        neg_count = int((y_valid == 0).sum())
        base = {
            "feature_name": feature,
            "feature_group": feature_group(feature),
            "direction": "unclear",
            "best_threshold": np.nan,
            "threshold_rule": "",
            "in_sample_balanced_accuracy": np.nan,
            "in_sample_precision": np.nan,
            "in_sample_recall": np.nan,
            "in_sample_f1": np.nan,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "selected_by_effect_size": False,
            "selected_by_threshold_quality": False,
            "caveat": "in-sample exploratory threshold; not final rule",
        }
        if pos_count < 3 or neg_count < 3 or len(np.unique(x)) < 2:
            base["caveat"] += "; insufficient valid variation"
            rows.append(base)
            continue
        quantiles = np.unique(np.nanquantile(x, np.linspace(0.05, 0.95, 19)))
        candidates = np.unique(np.concatenate([quantiles, np.unique(x)]))
        best: Optional[Dict[str, Any]] = None
        for threshold in candidates:
            for direction, pred in [
                ("higher_indicates_ad", (x >= threshold).astype(int)),
                ("lower_indicates_ad", (x <= threshold).astype(int)),
            ]:
                metrics = binary_metrics(y_valid, pred)
                candidate = {
                    **base,
                    **metrics,
                    "direction": direction,
                    "best_threshold": safe_float(threshold),
                    "threshold_rule": f"{feature} {'>=' if direction == 'higher_indicates_ad' else '<='} {threshold:.6g}",
                }
                if best is None:
                    best = candidate
                else:
                    key = (
                        candidate["in_sample_balanced_accuracy"],
                        candidate["in_sample_f1"] if np.isfinite(candidate["in_sample_f1"]) else -1,
                        -abs(candidate["best_threshold"] - np.nanmedian(x)),
                    )
                    best_key = (
                        best["in_sample_balanced_accuracy"],
                        best["in_sample_f1"] if np.isfinite(best["in_sample_f1"]) else -1,
                        -abs(best["best_threshold"] - np.nanmedian(x)),
                    )
                    if key > best_key:
                        best = candidate
        assert best is not None
        effect = safe_float(effect_map.loc[feature, "cohen_d"]) if feature in effect_map.index else np.nan
        best["selected_by_effect_size"] = bool(np.isfinite(effect) and abs(effect) >= 0.5)
        best["selected_by_threshold_quality"] = bool(best["in_sample_balanced_accuracy"] >= 0.65)
        rows.append(best)
    return pd.DataFrame(rows)


def video_normalized_features(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for feature in features:
        norm_col = f"{feature}__video_robust_z"
        out[norm_col] = np.nan
        for video_id, group in df.groupby("video_id"):
            values = pd.to_numeric(group[feature], errors="coerce")
            finite = values[np.isfinite(values)]
            if len(finite) < 2:
                continue
            median = finite.median()
            iqr = finite.quantile(0.75) - finite.quantile(0.25)
            scale = iqr if np.isfinite(iqr) and iqr > 0 else finite.std(ddof=1)
            if not np.isfinite(scale) or scale == 0:
                continue
            out.loc[group.index, norm_col] = (values - median) / scale
    return out


def recommendation_text(feature: str, usage: str, direction: str) -> str:
    group = feature_group(feature)
    if usage == "exclude":
        return "결측이 많거나 안정성이 낮아 이번 rule 후보에서는 제외하는 것이 좋다."
    if group == "Energy / Amplitude":
        return f"{feature}는 광고/비광고 간 음량 또는 에너지 차이를 나타내므로 audio cue score의 기본 신호로 사용할 수 있다."
    if group == "Silence / Low Energy":
        return f"{feature}는 무음/저에너지 구간 비율을 나타내므로 광고 구간의 끊김 없는 설명, BGM, 편집 밀도 차이를 보는 단서로 사용할 수 있다."
    if group == "Spectral Flux":
        return f"{feature}는 프레임 간 스펙트럼 변화량을 나타내므로 BGM, 효과음, 편집 전환 변화의 보조 단서로 사용할 수 있다."
    if group == "Spectral":
        return f"{feature}는 음색/주파수 질감 차이를 나타내므로 단독 threshold보다 다른 audio cue와 결합해 쓰는 것이 적절하다."
    if group == "Onset":
        return f"{feature}는 onset 또는 변화 강도 계열로, 광고 구간의 편집 밀도나 강조 구간 단서로 사용할 수 있다."
    if group == "MFCC":
        return f"{feature}는 해석 가능성이 낮은 음색 계열 feature이므로 단독 rule보다는 보조 score에 낮은 가중치로 사용하는 것이 적절하다."
    return f"{feature}는 방향성이 확인된 경우 보조 audio cue로 검토할 수 있다."


def build_recommendations(
    features: Sequence[str],
    pairwise: pd.DataFrame,
    delta_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    excluded: List[Dict[str, Any]],
) -> pd.DataFrame:
    pair_index = pairwise.set_index(["comparison_name", "feature_name"])
    threshold_index = thresholds.set_index("feature_name") if not thresholds.empty else pd.DataFrame()
    excluded_reasons = {item["feature_name"]: item["reason"] for item in excluded if item["reason"] != "metadata_or_status_column"}
    rows: List[Dict[str, Any]] = []
    for feature in features:
        evidence = "ad_vs_random"
        selected = pair_index.loc[("ad_full_vs_random_non_ad_30s", feature)] if ("ad_full_vs_random_non_ad_30s", feature) in pair_index.index else None
        usage = "low_priority"
        direction = "unclear"
        effect = np.nan
        if selected is not None:
            effect = safe_float(selected["cohen_d"])
            if np.isfinite(effect) and abs(effect) >= 0.5:
                usage = "use_for_ad_likelihood"
                direction = "higher_indicates_ad" if effect > 0 else "lower_indicates_ad"
        start_row = pair_index.loc[("ad_full_vs_pre_ad_10s", feature)] if ("ad_full_vs_pre_ad_10s", feature) in pair_index.index else None
        end_row = pair_index.loc[("ad_full_vs_post_ad_10s", feature)] if ("ad_full_vs_post_ad_10s", feature) in pair_index.index else None
        if usage == "low_priority" and start_row is not None and np.isfinite(start_row["cohen_d"]) and abs(start_row["cohen_d"]) >= 0.5:
            usage = "use_for_start_context_change"
            evidence = "ad_vs_pre"
            effect = safe_float(start_row["cohen_d"])
            direction = "higher_transition_change" if effect > 0 else "lower_transition_change"
        if usage == "low_priority" and end_row is not None and np.isfinite(end_row["cohen_d"]) and abs(end_row["cohen_d"]) >= 0.5:
            usage = "use_for_end_context_change"
            evidence = "ad_vs_post"
            effect = safe_float(end_row["cohen_d"])
            direction = "higher_transition_change" if effect > 0 else "lower_transition_change"
        delta_rows = delta_summary[delta_summary["feature_name"] == feature]
        if usage == "low_priority" and not delta_rows.empty:
            best_delta = delta_rows.iloc[delta_rows["paired_effect_size"].abs().fillna(0).argmax()]
            if np.isfinite(best_delta["paired_effect_size"]) and abs(best_delta["paired_effect_size"]) >= 0.5:
                usage = "use_for_general_audio_change"
                evidence = "paired_delta"
                effect = safe_float(best_delta["paired_effect_size"])
                direction = "higher_transition_change" if effect > 0 else "lower_transition_change"
        if feature in excluded_reasons:
            usage = "exclude"
            evidence = "missing_or_unstable"
            direction = "unclear"
        threshold = threshold_index.loc[feature] if isinstance(threshold_index, pd.DataFrame) and feature in threshold_index.index else None
        candidate_threshold = safe_float(threshold["best_threshold"]) if threshold is not None else np.nan
        threshold_rule = str(threshold["threshold_rule"]) if threshold is not None else ""
        selected_by_threshold = bool(threshold is not None and threshold["selected_by_threshold_quality"])
        if usage == "low_priority" and selected_by_threshold:
            usage = "use_for_ad_likelihood"
            evidence = "ad_vs_random"
            direction = str(threshold["direction"])
        if usage.startswith("use_for") and np.isfinite(effect) and abs(effect) >= 0.8:
            confidence = "high"
        elif usage.startswith("use_for") and (np.isfinite(effect) and abs(effect) >= 0.5 or selected_by_threshold):
            confidence = "medium"
        elif usage == "exclude":
            confidence = "low"
        else:
            confidence = "low"
        caution = "sample size is small; threshold is in-sample exploratory candidate, not final detector performance"
        if feature_group(feature) == "MFCC" and usage != "exclude":
            caution += "; MFCC is lower interpretability and should be auxiliary"
            if confidence == "high":
                confidence = "medium"
        rows.append(
            {
                "feature_name": feature,
                "feature_group": feature_group(feature),
                "recommended_usage": usage,
                "primary_evidence": evidence,
                "direction": direction,
                "effect_size": safe_float(effect),
                "effect_size_level": effect_level(effect),
                "candidate_threshold": candidate_threshold,
                "threshold_rule": threshold_rule,
                "confidence_level": confidence,
                "caution": caution,
                "plain_korean_interpretation": recommendation_text(feature, usage, direction),
            }
        )
    for feature, reason in excluded_reasons.items():
        if feature in features:
            continue
        rows.append(
            {
                "feature_name": feature,
                "feature_group": feature_group(feature),
                "recommended_usage": "exclude",
                "primary_evidence": "missing_or_unstable",
                "direction": "unclear",
                "effect_size": np.nan,
                "effect_size_level": "unknown",
                "candidate_threshold": np.nan,
                "threshold_rule": "",
                "confidence_level": "low",
                "caution": reason,
                "plain_korean_interpretation": "결측이 많거나 constant feature라 rule 후보에서 제외한다.",
            }
        )
    return pd.DataFrame(rows)


def top_features(pairwise: pd.DataFrame, comparison: str, n: int = 10) -> List[Dict[str, Any]]:
    rows = pairwise[pairwise["comparison_name"] == comparison].copy()
    rows = rows.sort_values("cohen_d", key=lambda s: s.abs(), ascending=False).head(n)
    keep = ["feature_name", "feature_group", "cohen_d", "effect_size_level", "direction", "mean_diff_a_minus_b"]
    return rows[keep].to_dict("records")


def top_recommended(recommendations: pd.DataFrame, n: int = 15) -> List[Dict[str, Any]]:
    order = {"high": 0, "medium": 1, "low": 2}
    rec = recommendations[recommendations["recommended_usage"] != "exclude"].copy()
    rec["_order"] = rec["confidence_level"].map(order).fillna(9)
    rec["_abs_effect"] = rec["effect_size"].abs()
    rec = rec.sort_values(["_order", "_abs_effect"], ascending=[True, False]).head(n)
    keep = ["feature_name", "feature_group", "recommended_usage", "direction", "effect_size", "confidence_level", "threshold_rule"]
    return rec[keep].to_dict("records")


def write_markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows_"

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            if not np.isfinite(value):
                return ""
            return f"{value:.4g}"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def segment_korean_summary(profile: pd.DataFrame, segment_type: str) -> str:
    seg = profile[profile["segment_type"] == segment_type].copy()
    if seg.empty:
        return "- 데이터 없음"
    energy = seg[seg["feature_name"].isin(["audio_rms_mean", "audio_log_energy_mean", "silence_ratio", "spectral_flux_max", "onset_strength_max"])]
    parts = []
    for _, row in energy.iterrows():
        parts.append(f"{row['feature_name']} median={row['median']:.4g}")
    return "- " + "; ".join(parts)


def write_summary(
    path: Path,
    report: Dict[str, Any],
    profile: pd.DataFrame,
    pairwise: pd.DataFrame,
    delta_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> str:
    cols = ["feature_name", "feature_group", "cohen_d", "effect_size_level", "direction", "mean_diff_a_minus_b"]
    ad_random = top_features(pairwise, "ad_full_vs_random_non_ad_30s", 10)
    ad_pre = top_features(pairwise, "ad_full_vs_pre_ad_10s", 10)
    ad_post = top_features(pairwise, "ad_full_vs_post_ad_10s", 10)
    pre_post = top_features(pairwise, "pre_ad_10s_vs_post_ad_10s", 10)
    threshold_rows = thresholds.sort_values("in_sample_balanced_accuracy", ascending=False).head(10).to_dict("records")
    rec_rows = top_recommended(recommendations, 15)
    high_priority = recommendations[recommendations["confidence_level"] == "high"]["feature_name"].head(12).tolist()
    medium_priority = recommendations[recommendations["confidence_level"] == "medium"]["feature_name"].head(12).tolist()
    low_or_exclude = recommendations[recommendations["recommended_usage"].isin(["low_priority", "exclude"])]["feature_name"].head(12).tolist()
    sub_lines = [
        f"- {name}: {result.get('status')} (warnings={len(result.get('warnings', []))}, errors={len(result.get('errors', []))})"
        for name, result in report.get("sub_agent_results", {}).items()
    ]
    output_lines = [f"- {p}" for p in report.get("output_files", [])]
    text = "\n".join(
        [
            "# Analyze Audio Rule Features Summary",
            "",
            "## 1. 작업 개요",
            "이번 작업은 기존 labeled audio feature CSV를 읽어 rule-based detector feature selection support를 위한 audio cue analysis를 수행한 것이다.",
            "raw audio 재추출, ffmpeg decode, wav 생성은 수행하지 않았다.",
            "결과는 labeled audio feature analysis이며 final detector 성능으로 해석하지 않는다.",
            "",
            "## 2. 사용한 입력 파일",
            f"- selected feature CSV: {report.get('selected_feature_file')}",
            f"- selected feature modified time: {report.get('selected_feature_file_modified_time')}",
            f"- sampling plan CSV: {report.get('input_files', {}).get('sampling_plan')}",
            f"- group summary CSV: {report.get('input_files', {}).get('group_summary')}",
            f"- previous extraction report: {report.get('input_files', {}).get('extraction_report')}",
            "",
            "## 3. 입력 데이터 개요",
            f"- total rows: {report.get('input_row_count')}",
            f"- segment_type counts: {report.get('segment_type_counts')}",
            f"- video count: {report.get('video_count')}",
            f"- ad interval count: {report.get('ad_interval_count')}",
            f"- numeric feature count: {report.get('numeric_feature_count')}",
            f"- analyzed feature count: {report.get('analyzed_feature_count')}",
            f"- excluded feature count: {report.get('excluded_feature_count')}",
            "",
            "## 4. Segment Type별 오디오 특성 요약",
            "### random_non_ad_30s",
            segment_korean_summary(profile, "random_non_ad_30s"),
            "### pre_ad_10s",
            segment_korean_summary(profile, "pre_ad_10s"),
            "### ad_full",
            segment_korean_summary(profile, "ad_full"),
            "### post_ad_10s",
            segment_korean_summary(profile, "post_ad_10s"),
            "",
            "## 5. 핵심 비교 결과",
            "### ad_full vs random_non_ad_30s",
            write_markdown_table(ad_random, cols),
            "",
            "### ad_full vs pre_ad_10s",
            write_markdown_table(ad_pre, cols),
            "",
            "### ad_full vs post_ad_10s",
            write_markdown_table(ad_post, cols),
            "",
            "### pre_ad_10s vs post_ad_10s",
            write_markdown_table(pre_post, cols),
            "",
            "## 6. Paired Delta 결과",
            "같은 ad_interval_id 안에서 ad_full, pre/post context, random_non_ad_30s 간 차이를 계산했다.",
            "현재 ad_full은 광고 전체 평균이므로 ad_minus_pre와 post_minus_ad는 정밀 boundary 변화가 아니라 광고 주변 context와 광고 전체의 차이로 해석해야 한다.",
            write_markdown_table(
                delta_summary.sort_values("paired_effect_size", key=lambda s: s.abs(), ascending=False)
                .head(10)
                .to_dict("records"),
                ["delta_type", "feature_name", "feature_group", "pair_count", "delta_median", "positive_delta_ratio", "paired_effect_size"],
            ),
            "",
            "## 7. Candidate Threshold 결과",
            "아래 threshold는 ad_full=positive, random_non_ad_30s=negative로 둔 train/test split 없는 in-sample exploratory threshold 후보이다.",
            "final rule 또는 final 성능으로 해석하지 않는다.",
            write_markdown_table(
                threshold_rows,
                ["feature_name", "feature_group", "direction", "threshold_rule", "in_sample_balanced_accuracy", "in_sample_f1", "selected_by_effect_size", "selected_by_threshold_quality"],
            ),
            "",
            "## 8. Rule-based Detector 반영 제안",
            "1. audio_ad_likelihood_score 후보: ad_full vs random_non_ad_30s에서 차이가 큰 energy, silence, spectral/onset 계열 feature를 video-level robust z-score로 변환해 방향성을 반영한다.",
            "2. audio_context_change_score 후보: abs(ad_full - pre_ad_10s), abs(post_ad_10s - ad_full) 같은 paired delta를 보조 단서로 사용한다.",
            "3. audio_silence_transition_score 후보: silence_ratio, low_energy_ratio가 광고 구간에서 어떻게 달라지는지 방향성을 반영한다.",
            "4. audio_music_or_texture_change_score 후보: spectral_flux, spectral_flatness, spectral_centroid 계열을 BGM/편집 질감 변화 단서로 둔다.",
            "5. audio_onset_or_flux_score 후보: onset_strength와 spectral_flux_max를 편집 밀도 변화 보조 cue로 둔다.",
            "",
            "Pseudo-rule 초안:",
            "```text",
            "IF top energy/silence/spectral cue is strong relative to the same-video baseline",
            "AND context delta around the labeled ad interval is directionally consistent",
            "THEN raise audio_ad_likelihood_score",
            "```",
            "audio cue 단독으로 최종 판단하지 않고 visual/OCR/scene cue와 결합하는 것이 적절하다.",
            "",
            "## 9. 우선 적용할 Feature 추천",
            "### 추천 feature",
            write_markdown_table(rec_rows, ["feature_name", "feature_group", "recommended_usage", "direction", "effect_size", "confidence_level", "threshold_rule"]),
            "",
            f"- high priority: {high_priority}",
            f"- medium priority: {medium_priority}",
            f"- low priority/exclude examples: {low_or_exclude}",
            "",
            "## 10. 한계",
            "- sample size가 작아 effect size는 탐색적 효과 크기로만 해석해야 한다.",
            "- random_non_ad_30s가 전체 비광고 구간을 대표하지 않을 수 있다.",
            "- 현재 ad_full은 광고 전체 평균이라 광고 시작 초반/종료 직전의 boundary 변화를 직접 보지는 못한다.",
            "- train/test split 없이 threshold를 탐색했으므로 final 성능으로 해석하면 안 된다.",
            "",
            "## 11. 다음 작업 제안",
            "- ad_start_first_10s, ad_end_last_10s feature를 추가 추출한다.",
            "- 5초 window-level audio cue score를 생성한다.",
            "- visual/OCR/scene cue와 결합한 rule score를 설계한다.",
            "- 실제 boundary 후보에서 audio cue가 false positive 감소에 도움이 되는지 검증한다.",
            "",
            "## 12. Sub Agent 검증 결과",
            *sub_lines,
            "",
            "## 13. 생성 파일 목록",
            *output_lines,
            "",
            f"- old_project_modified: {report.get('old_project_modified')}",
            f"- input_feature_file_modified: {report.get('input_feature_file_modified')}",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")
    return text


def forbidden_latest_files() -> List[str]:
    if not LATEST_DIR.exists():
        return []
    forbidden: List[str] = []
    for path in LATEST_DIR.rglob("*"):
        if path.is_dir() and path.name in {"cache", "tmp"}:
            forbidden.append(str(path))
        elif path.is_file() and path.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES:
            forbidden.append(str(path))
    return forbidden


def update_latest(paths: Dict[str, Path], report: Dict[str, Any], backup_dir: Optional[Path]) -> List[str]:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    include_keys = {
        "profile",
        "pairwise",
        "paired_delta_summary",
        "thresholds",
        "recommendations",
        "normalized_pairwise",
        "report",
        "summary",
        "run_log",
        "script",
    }
    allowed_names = {paths[key].name for key in include_keys if key in paths} | {"README_latest_files.md"}
    archived: List[str] = []
    cleanup_dir = (backup_dir or (BACKUPS_DIR / f"{TASK_NAME}_{report['version']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")) / "latest_for_chatgpt_previous"
    for item in list(LATEST_DIR.iterdir()):
        if item.name in allowed_names:
            continue
        cleanup_dir.mkdir(parents=True, exist_ok=True)
        dest = cleanup_dir / item.name
        if dest.exists():
            dest = cleanup_dir / f"{dest.stem}_{datetime.now().strftime('%H%M%S')}{dest.suffix}"
        shutil.move(str(item), str(dest))
        archived.append(str(dest))
    if archived:
        report["latest_for_chatgpt_archived_previous_files"] = archived
    copied: List[str] = []
    for key in include_keys:
        path = paths[key]
        if path.exists() and path.suffix.lower() not in FORBIDDEN_LATEST_SUFFIXES:
            dest = LATEST_DIR / path.name
            shutil.copy2(path, dest)
            copied.append(str(dest))
    readme = LATEST_DIR / "README_latest_files.md"
    lines = [
        "# Latest Files",
        "",
        f"- task_name: {TASK_NAME}",
        f"- version: {report['version']}",
        f"- selected feature file: {report['selected_feature_file']}",
        f"- selected feature file modified time: {report['selected_feature_file_modified_time']}",
        f"- old_project_modified: {str(report.get('old_project_modified')).lower()}",
        "- forbidden media/model/cache files included: false",
        "",
        "This bundle is for rule-based audio cue analysis and labeled audio feature analysis.",
        "It is not a final detector performance claim.",
        "",
        "## Files",
        f"- {paths['profile'].name}: segment type feature profile",
        f"- {paths['pairwise'].name}: pairwise effect-size comparison",
        f"- {paths['paired_delta_summary'].name}: interval-level paired delta summary",
        f"- {paths['thresholds'].name}: exploratory single-feature threshold candidates",
        f"- {paths['recommendations'].name}: rule feature recommendations",
        f"- {paths['normalized_pairwise'].name}: video-level normalized pairwise comparison",
        f"- {paths['report'].name}: detailed report JSON",
        f"- {paths['summary'].name}: human-readable summary",
        f"- {paths['run_log'].name}: run log",
        f"- {paths['script'].name}: reproducible analysis script",
        "- README_latest_files.md: this manifest",
    ]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copied.append(str(readme))
    return sorted(copied)


def validate_input(df: pd.DataFrame, report: Dict[str, Any], feature_mtime_before: int, feature_file: Path) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
    found = set(df.get("segment_type", pd.Series(dtype=str)).dropna().astype(str))
    missing_types = set(SEGMENT_TYPES) - found
    if missing_types:
        errors.append(f"missing core segment types: {sorted(missing_types)}")
    if report["version"] == "v2_4" and report["input_row_count"] != 88:
        warnings.append("v2_4 row count differs from expected 88")
    if (df["feature_status"] != "success").any():
        warnings.append("some feature_status rows are not success")
    if feature_file.stat().st_mtime_ns != feature_mtime_before:
        errors.append("input feature file mtime changed during task")
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {"status": status, "warnings": warnings, "errors": errors}


def validate_stats(report: Dict[str, Any], profile: pd.DataFrame, pairwise: pd.DataFrame, delta_summary: pd.DataFrame) -> Dict[str, Any]:
    warnings: List[str] = ["sample size is small; effect sizes are exploratory"]
    errors: List[str] = []
    if report["analyzed_feature_count"] < 5:
        errors.append("too few analyzed numeric audio features")
    bad_metadata = [f for f in report["analyzed_features"] if f in EXCLUDED_METADATA_COLUMNS]
    if bad_metadata:
        errors.append(f"metadata columns included as features: {bad_metadata}")
    if profile.empty or pairwise.empty or delta_summary.empty:
        errors.append("one or more statistical outputs are empty")
    if pairwise["cohen_d"].notna().sum() < max(1, len(pairwise) // 3):
        errors.append("most effect sizes are NaN")
    if report["inf_count_by_feature"] and any(v > 0 for v in report["inf_count_by_feature"].values()):
        warnings.append("inf values were replaced with NaN before analysis")
    if "tempo_estimate" in [x["feature_name"] for x in report["excluded_features"]]:
        warnings.append("tempo_estimate excluded because it is mostly NaN")
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {"status": status, "warnings": warnings, "errors": errors}


def validate_recommendations(recommendations: pd.DataFrame, thresholds: pd.DataFrame, summary_text: str) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    if recommendations.empty:
        errors.append("feature recommendation output is empty")
    if not {"recommended_usage", "primary_evidence", "direction"}.issubset(recommendations.columns):
        errors.append("recommendation evidence columns missing")
    if "in-sample exploratory threshold" not in summary_text:
        errors.append("threshold caveat missing from summary")
    banned = ["최종 광고 탐지 성능", "최종 모델 성능", "audio만으로 광고 탐지 성공", "광고 탐지 정확도 확정"]
    found = [phrase for phrase in banned if phrase in summary_text]
    if found:
        errors.append(f"final-performance banned phrases found: {found}")
    high = (recommendations["confidence_level"] == "high").sum() if "confidence_level" in recommendations else 0
    if high == 0:
        warnings.append("no high-confidence rule feature recommendations")
    if thresholds["selected_by_threshold_quality"].sum() == 0:
        warnings.append("no threshold candidate passed exploratory quality filter")
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {"status": status, "warnings": warnings, "errors": errors}


def validate_interpretation(summary_text: str) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    required = ["작업 개요", "핵심 비교 결과", "한계", "다음 작업 제안", "Sub Agent"]
    missing = [item for item in required if item not in summary_text]
    if missing:
        errors.append(f"summary missing required sections: {missing}")
    if "audio cue analysis" not in summary_text:
        errors.append("summary does not describe result as audio cue analysis")
    if "ad_full은 광고 전체 평균" not in summary_text:
        errors.append("ad_full whole-ad-average limitation missing")
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {"status": status, "warnings": warnings, "errors": errors}


def validate_output_safety(report: Dict[str, Any], paths: Dict[str, Path]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    for key, path in paths.items():
        if not path.exists():
            errors.append(f"missing output {key}: {path}")
    forbidden = forbidden_latest_files()
    if forbidden:
        errors.append(f"forbidden files found in latest_for_chatgpt: {forbidden}")
    if report.get("input_feature_file_modified"):
        errors.append("input feature file modified")
    if report.get("old_project_modified"):
        errors.append("old project modified")
    if report.get("backup_dir"):
        warnings.append(f"existing files were backed up to {report['backup_dir']}")
    if report.get("latest_for_chatgpt_archived_previous_files"):
        warnings.append("stale latest_for_chatgpt files were archived")
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {"status": status, "warnings": warnings, "errors": errors}


def main() -> int:
    start_epoch = time.time()
    start_time = now_iso()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    selected_feature_file, version, reason = discover_feature_file()
    paths = output_paths(version)
    for directory in [DATA_AUDIO_DIR, REPORTS_DIR, LOGS_DIR, SCRIPTS_AUDIO_DIR, LATEST_DIR, BACKUPS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_existing(paths, version, timestamp)
    logger = TaskLogger(paths["run_log"])
    warnings: List[str] = []
    errors: List[str] = []
    if backup_dir:
        warnings.append(f"existing outputs backed up to {backup_dir}")

    logger.log("[STEP 01] Start task and create old project snapshot")
    logger.log(f"Task start time: {start_time}")
    old_before = old_project_snapshot()

    logger.log("[STEP 02] Discover latest labeled audio feature file")
    selected_mtime = iso_from_timestamp(selected_feature_file.stat().st_mtime)
    feature_mtime_before = selected_feature_file.stat().st_mtime_ns
    logger.log(f"Selected feature file: {selected_feature_file}")
    logger.log(f"Selected feature reason: {reason}")

    logger.log("[STEP 03] Load feature CSV and validate segment types")
    df = pd.read_csv(selected_feature_file, encoding="utf-8-sig")
    segment_counts = df["segment_type"].value_counts(dropna=False).to_dict()
    logger.log(f"Input row count: {len(df)}")
    logger.log(f"Segment type counts: {segment_counts}")
    missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_required:
        errors.append(f"missing required columns: {missing_required}")

    logger.log("[STEP 04] Identify numeric audio feature columns")
    numeric_features, analyzed_features, excluded_features, nan_ratio, inf_count, constant_features = select_features(df)
    for col in analyzed_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    logger.log(f"Numeric feature count: {len(numeric_features)}")
    logger.log(f"Analyzed feature count: {len(analyzed_features)}")
    logger.log(f"Excluded feature count: {len(excluded_features)}")
    if "tempo_estimate" in [x["feature_name"] for x in excluded_features]:
        warnings.append("tempo_estimate excluded from main ranking because it is mostly NaN")

    logger.log("[STEP 05] Build segment-type profiles")
    profile = profile_by_segment(df, analyzed_features, version)
    profile.to_csv(paths["profile"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 06] Compute pairwise comparisons and effect sizes")
    pairwise = pairwise_comparison(df, analyzed_features)
    pairwise.to_csv(paths["pairwise"], index=False, encoding="utf-8-sig")
    logger.log(f"Pairwise comparison row count: {len(pairwise)}")

    logger.log("[STEP 07] Compute interval-level paired deltas")
    delta_df, delta_summary = paired_delta(df, analyzed_features)
    delta_df.to_csv(paths["paired_delta"], index=False, encoding="utf-8-sig")
    delta_summary.to_csv(paths["paired_delta_summary"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 08] Generate candidate threshold analysis")
    thresholds = threshold_analysis(df, analyzed_features, pairwise)
    thresholds.to_csv(paths["thresholds"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 09] Generate rule feature recommendations")
    norm_df = video_normalized_features(df, analyzed_features)
    norm_df.to_csv(paths["normalized_features"], index=False, encoding="utf-8-sig")
    norm_features = [f"{feature}__video_robust_z" for feature in analyzed_features if f"{feature}__video_robust_z" in norm_df.columns]
    norm_pairwise = pairwise_comparison(norm_df, norm_features)
    norm_pairwise.to_csv(paths["normalized_pairwise"], index=False, encoding="utf-8-sig")
    recommendations = build_recommendations(analyzed_features, pairwise, delta_summary, thresholds, excluded_features)
    recommendations.to_csv(paths["recommendations"], index=False, encoding="utf-8-sig")
    logger.log(f"Rule recommendation row count: {len(recommendations)}")

    report: Dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "task_name": TASK_NAME,
        "version": version,
        "start_time": start_time,
        "end_time": None,
        "actual_runtime_seconds": None,
        "actual_runtime_readable": None,
        "input_files": {
            "selected_feature_file": str(selected_feature_file),
            "sampling_plan": str(DATA_AUDIO_DIR / f"audio_labeled_segment_sampling_plan_{version}.csv"),
            "group_summary": str(DATA_AUDIO_DIR / f"audio_labeled_segment_group_summary_{version}.csv"),
            "probe_metadata": str(DATA_AUDIO_DIR / f"audio_probe_metadata_{version}.csv"),
            "extraction_report": str(REPORTS_DIR / f"extract_labeled_audio_features_{version}_report.json"),
            "ad_interval_segments": str(PROJECT_ROOT / f"data/segments/ad_interval_segments_{version}.csv"),
        },
        "output_files": [str(p) for p in paths.values()],
        "generated_files": [str(p) for p in paths.values()],
        "latest_for_chatgpt_files": [],
        "missing_input_files": [],
        "missing_required_columns": missing_required,
        "warnings": warnings,
        "errors": errors,
        "old_project_modified": None,
        "old_project_snapshot_before": old_before,
        "old_project_snapshot_after": None,
        "sub_agent_results": {},
        "selected_feature_file": str(selected_feature_file),
        "selected_feature_file_modified_time": selected_mtime,
        "selected_feature_reason": reason,
        "input_row_count": int(len(df)),
        "segment_type_counts": {str(k): int(v) for k, v in segment_counts.items()},
        "video_count": int(df["video_id"].nunique()),
        "ad_interval_count": int(df["ad_interval_id"].nunique()),
        "expected_row_count": 88 if version == "v2_4" else None,
        "actual_row_count": int(len(df)),
        "numeric_feature_count": int(len(numeric_features)),
        "analyzed_feature_count": int(len(analyzed_features)),
        "excluded_feature_count": int(len(excluded_features)),
        "analyzed_features": analyzed_features,
        "excluded_features": excluded_features,
        "feature_groups": {feature: feature_group(feature) for feature in analyzed_features},
        "nan_ratio_by_feature": nan_ratio,
        "inf_count_by_feature": inf_count,
        "constant_features": constant_features,
        "profile_output_path": str(paths["profile"]),
        "pairwise_comparison_output_path": str(paths["pairwise"]),
        "paired_delta_output_path": str(paths["paired_delta"]),
        "paired_delta_summary_output_path": str(paths["paired_delta_summary"]),
        "candidate_threshold_output_path": str(paths["thresholds"]),
        "feature_recommendation_output_path": str(paths["recommendations"]),
        "video_normalized_output_path": str(paths["normalized_features"]),
        "video_normalized_pairwise_output_path": str(paths["normalized_pairwise"]),
        "top_ad_vs_random_features": top_features(pairwise, "ad_full_vs_random_non_ad_30s", 10),
        "top_pre_vs_ad_features": top_features(pairwise, "ad_full_vs_pre_ad_10s", 10),
        "top_ad_vs_post_features": top_features(pairwise, "ad_full_vs_post_ad_10s", 10),
        "recommended_rule_features": top_recommended(recommendations, 15),
        "interpretation_note": "audio cue analysis and rule-based detector feature selection support; not final detector performance",
        "limitations": [
            "small labeled sample size",
            "random_non_ad_30s may not represent all non-ad content",
            "ad_full is whole-ad average, not start/end boundary-local audio",
            "threshold analysis is in-sample exploratory without train/test split",
        ],
        "next_steps": [
            "extract ad_start_first_10s and ad_end_last_10s features",
            "build 5-second window-level audio cue scores",
            "combine audio cue with visual/OCR/scene cues",
            "test whether audio cue reduces false positives on boundary candidates",
        ],
        "latest_for_chatgpt_forbidden_files_found": [],
        "input_feature_file_modified": False,
        "backup_dir": str(backup_dir) if backup_dir else "",
    }

    logger.log("[STEP 10] Run Sub Agent validations")
    input_result = validate_input(df, report, feature_mtime_before, selected_feature_file)
    stats_result = validate_stats(report, profile, pairwise, delta_summary)
    report["sub_agent_results"] = {
        "input_schema_validation": input_result,
        "statistical_analysis_validation": stats_result,
    }
    summary_text = write_summary(paths["summary"], report, profile, pairwise, delta_summary, thresholds, recommendations)
    report["sub_agent_results"]["rule_recommendation_validation"] = validate_recommendations(recommendations, thresholds, summary_text)
    report["sub_agent_results"]["interpretation_report_validation"] = validate_interpretation(summary_text)

    old_after = old_project_snapshot()
    report["old_project_snapshot_after"] = old_after
    report["old_project_modified"] = old_before != old_after
    report["input_feature_file_modified"] = selected_feature_file.stat().st_mtime_ns != feature_mtime_before
    if report["old_project_modified"]:
        errors.append("old project modified during task")
    if report["input_feature_file_modified"]:
        errors.append("input feature file modified during task")

    save_json(paths["report"], report)
    logger.log("[STEP 11] Update latest_for_chatgpt with allowed files only")
    report["latest_for_chatgpt_files"] = update_latest(paths, report, backup_dir)
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden_latest_files()
    report["sub_agent_results"]["output_safety_validation"] = validate_output_safety(report, paths)

    for name, result in report["sub_agent_results"].items():
        logger.log(f"Sub Agent {name}: {result['status']} warnings={len(result.get('warnings', []))} errors={len(result.get('errors', []))}")
        for warning in result.get("warnings", []):
            warning_text = f"{name}: {warning}"
            if warning_text not in warnings:
                warnings.append(warning_text)
        for error in result.get("errors", []):
            error_text = f"{name}: {error}"
            if error_text not in errors:
                errors.append(error_text)
    report["warnings"] = warnings
    report["errors"] = errors

    summary_text = write_summary(paths["summary"], report, profile, pairwise, delta_summary, thresholds, recommendations)
    end_time = now_iso()
    runtime = time.time() - start_epoch
    report["end_time"] = end_time
    report["actual_runtime_seconds"] = round(runtime, 3)
    report["actual_runtime_readable"] = readable_seconds(runtime)
    save_json(paths["report"], report)
    report["latest_for_chatgpt_files"] = update_latest(paths, report, backup_dir)
    save_json(paths["report"], report)
    shutil.copy2(paths["report"], LATEST_DIR / paths["report"].name)
    shutil.copy2(paths["summary"], LATEST_DIR / paths["summary"].name)

    logger.log(f"Task end time: {end_time}")
    logger.log(f"Actual runtime: {report['actual_runtime_readable']}")
    logger.log(f"Warnings: {warnings}")
    logger.log(f"Errors: {errors}")
    logger.log("[STEP 12] Print human-readable final summary")
    shutil.copy2(paths["run_log"], LATEST_DIR / paths["run_log"].name)

    status = "실패" if any(r["status"] == "FAIL" for r in report["sub_agent_results"].values()) or errors else "조건부 성공" if any(r["status"] == "WARN" for r in report["sub_agent_results"].values()) or warnings else "성공"
    print("\n# Audio Rule Feature Analysis Result", flush=True)
    print(f"- status: {status}", flush=True)
    print(f"- selected_feature_file: {selected_feature_file}", flush=True)
    print(f"- version: {version}", flush=True)
    print(f"- rows: {len(df)}, analyzed_features: {len(analyzed_features)}, excluded_features: {len(excluded_features)}", flush=True)
    print(f"- pairwise_rows: {len(pairwise)}, recommendation_rows: {len(recommendations)}", flush=True)
    print(f"- report: {paths['report']}", flush=True)
    print(f"- summary: {paths['summary']}", flush=True)
    print(f"- errors: {errors}", flush=True)
    return 1 if errors or any(r["status"] == "FAIL" for r in report["sub_agent_results"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
