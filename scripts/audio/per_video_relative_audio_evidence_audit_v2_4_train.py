#!/usr/bin/env python3
"""Per-video relative audio evidence audit for v2_4 train split.

This script uses already-extracted full-video audio subwindow features. It does
not decode audio/video, does not modify detector rules, and does not create
validation/test row-level outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "v2_4"
TASK_NAME = "per_video_relative_audio_evidence_audit_v2_4_train"
DEFAULT_PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
SCRIPT_RELATIVE_PATH = "scripts/audio/per_video_relative_audio_evidence_audit_v2_4_train.py"
SPLIT_SEED = 20240524
FIXED_SPLIT = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}
CORE_RAW_FEATURES = [
    "audio_rms_mean",
    "audio_rms_std",
    "audio_log_energy_mean",
    "audio_log_energy_std",
    "audio_mean_abs_amplitude",
    "silence_ratio",
    "low_energy_ratio",
    "spectral_flux_mean",
    "spectral_flux_std",
    "spectral_flux_max",
    "onset_density",
    "onset_count_per_sec",
    "onset_strength_mean",
    "onset_strength_max",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_flatness_mean",
    "spectral_flatness_std",
]
HIGHER_ACTIVE_FEATURES = [
    "audio_rms_mean",
    "audio_rms_std",
    "audio_log_energy_mean",
    "audio_log_energy_std",
    "audio_mean_abs_amplitude",
    "spectral_flux_mean",
    "spectral_flux_std",
    "spectral_flux_max",
    "onset_density",
    "onset_count_per_sec",
    "onset_strength_mean",
    "onset_strength_max",
]
QUIET_FEATURES = ["silence_ratio", "low_energy_ratio"]
TEXTURE_SHAPE_FEATURES = [
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_flatness_mean",
    "spectral_flatness_std",
]
AUX_FEATURES = ["audio_ad_like_score", "inverse_silence_score"]
FORBIDDEN_BUNDLE_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a",
    ".jpg", ".jpeg", ".png", ".webp", ".pt", ".pth", ".ckpt", ".bin",
}
EPS = 1e-9


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


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        if pd.isna(obj) and not isinstance(obj, (str, bytes, bytearray)):
            return None
    except Exception:
        pass
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unique_file_path(path: Path, run_id: str) -> Path:
    if not path.exists():
        return path
    candidate = path.with_name(f"{path.stem}_{run_id}{path.suffix}")
    idx = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{run_id}_{idx}{path.suffix}")
        idx += 1
    return candidate


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return str(path)


def snapshot_path(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "metadata_digest": None}
    entries: List[str] = []
    file_count = 0
    if path.is_file():
        targets = [path]
        base = path.parent
    else:
        targets = [p for p in path.rglob("*") if p.is_file()]
        base = path
    for item in targets:
        try:
            stat = item.stat()
        except OSError:
            continue
        try:
            item_rel = item.relative_to(base).as_posix()
        except Exception:
            item_rel = str(item)
        entries.append(f"{item_rel}\t{stat.st_size}\t{stat.st_mtime_ns}")
        file_count += 1
    digest = hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()
    return {
        "path": str(path),
        "exists": True,
        "file_count": file_count,
        "metadata_digest": digest,
        "snapshot_type": "relative_path_size_mtime_ns",
    }


def snapshot_unchanged(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    return before.get("exists") == after.get("exists") and before.get("metadata_digest") == after.get("metadata_digest")


def output_paths(project_root: Path, run_id: str) -> Dict[str, Path]:
    planned = {
        "baseline_summary": project_root / "data/audio/per_video_audio_baseline_summary_v2_4_train.csv",
        "relative_levels": project_root / "data/audio/per_video_audio_relative_levels_v2_4_train.csv",
        "window_summary": project_root / "data/audio/train_actual_ad_per_video_relative_audio_window_summary_v2_4.csv",
        "interval_summary": project_root / "data/audio/train_actual_ad_per_video_relative_audio_interval_summary_v2_4.csv",
        "ad_profile": project_root / "data/audio/per_video_train_audio_ad_profile_v2_4.csv",
        "clean_nonad_profile": project_root / "data/audio/per_video_train_audio_clean_nonad_profile_v2_4.csv",
        "contrast": project_root / "data/audio/per_video_train_audio_ad_vs_nonad_feature_contrast_v2_4.csv",
        "candidate_score": project_root / "data/audio/per_video_train_audio_candidate_score_for_discussion_v2_4.csv",
        "top_segments": project_root / "data/audio/per_video_train_audio_candidate_top_segments_for_review_v2_4.csv",
        "warnings": project_root / "data/audio/per_video_audio_relative_analysis_warnings_v2_4.csv",
        "global_ad_pattern": project_root / "data/audio/train_audio_global_ad_pattern_reference_summary_v2_4.csv",
        "global_feature_direction": project_root / "data/audio/train_audio_global_feature_direction_reference_v2_4.csv",
        "summary_md": project_root / "reports/audio/per_video_relative_audio_evidence_audit_v2_4_train_summary.md",
        "report_json": project_root / "reports/audio/per_video_relative_audio_evidence_audit_v2_4_train_report.json",
        "rule_direction_md": project_root / "reports/audio/per_video_relative_audio_rule_direction_v2_4_train.md",
        "run_log": project_root / "logs/per_video_relative_audio_evidence_audit_v2_4_train_run_log.txt",
    }
    return {key: unique_file_path(path, run_id) for key, path in planned.items()}


def resolve_existing(path: Path, glob_pattern: str, root: Path) -> Path:
    if path.exists():
        return path
    matches = sorted(root.glob(glob_pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if matches:
        return matches[0]
    return path


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def interval_distance(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if a_end <= b_start:
        return b_start - a_end
    if a_start >= b_end:
        return a_start - b_end
    return 0.0


def union_duration(intervals: Iterable[Tuple[float, float]]) -> float:
    cleaned = [(float(s), float(e)) for s, e in intervals if np.isfinite(s) and np.isfinite(e) and e > s]
    if not cleaned:
        return 0.0
    cleaned.sort()
    merged: List[Tuple[float, float]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + EPS:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return float(sum(end - start for start, end in merged))


def clip01(value: Any) -> float:
    val = safe_float(value, 0.0)
    if not np.isfinite(val):
        val = 0.0
    return float(max(0.0, min(1.0, val)))


def sigmoid(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    value = max(-6.0, min(6.0, value))
    return float(1.0 / (1.0 + math.exp(-value)))


def validate_split(split_df: pd.DataFrame) -> Dict[str, Any]:
    observed = {
        split: sorted(pd.to_numeric(group["video_id"], errors="coerce").dropna().astype(int).tolist())
        for split, group in split_df.groupby("split")
    }
    return {
        "observed": observed,
        "train_matches_fixed": observed.get("train", []) == FIXED_SPLIT["train"],
        "validation_matches_fixed": observed.get("validation", []) == FIXED_SPLIT["validation"],
        "test_matches_fixed": observed.get("test", []) == FIXED_SPLIT["test"],
        "all_match": observed == FIXED_SPLIT,
    }


def select_features(recommendations: pd.DataFrame, full_columns: Sequence[str], warnings: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    full_set = set(full_columns)
    rec_features = []
    if not recommendations.empty and "feature_name" in recommendations.columns:
        rec = recommendations.copy()
        if "exists_in_full_video_features" in rec.columns:
            rec = rec[rec["exists_in_full_video_features"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if "recommendation_priority" in rec.columns:
            rec = rec.sort_values(["recommendation_priority", "feature_name"])
        rec_features = [f for f in rec["feature_name"].astype(str).tolist() if f in full_set]
    core_raw = [f for f in CORE_RAW_FEATURES if f in full_set]
    for f in CORE_RAW_FEATURES:
        if f not in full_set:
            warnings.append({"warning_type": "missing_core_feature", "feature_name": f, "warning_note": "core raw feature missing from full-video features"})
    mfcc_std = [f"mfcc_{i}_std" for i in range(1, 14) if f"mfcc_{i}_std" in full_set]
    aux = [f for f in AUX_FEATURES if f in full_set]
    selected = list(dict.fromkeys(core_raw + rec_features + aux + mfcc_std))
    raw_relative = [f for f in selected if not (f.endswith("_score") or f.endswith("__video_robust_z") or f.endswith("__directional_score") or f == "audio_ad_like_score")]
    score_reference = [f for f in ["audio_ad_like_score", "inverse_silence_score"] if f in full_set]
    return {
        "core_raw_features": core_raw,
        "recommended_features_from_2_3": rec_features,
        "mfcc_std_features": mfcc_std,
        "score_reference_features": score_reference,
        "analysis_features": selected,
        "raw_relative_features": raw_relative,
        "higher_active_features": [f for f in HIGHER_ACTIVE_FEATURES if f in full_set],
        "quiet_features": [f for f in QUIET_FEATURES if f in full_set],
        "texture_shape_features": [f for f in TEXTURE_SHAPE_FEATURES if f in full_set],
    }


def load_train_intervals(label_df: pd.DataFrame, train_ids: Sequence[int]) -> pd.DataFrame:
    df = label_df[label_df["segment_type"].astype(str).eq("ad_interval")].copy()
    df["video_id"] = pd.to_numeric(df["video_id"], errors="coerce").astype("Int64")
    df = df[df["video_id"].isin(train_ids)].copy()
    if "segment_valid" in df.columns:
        df = df[df["segment_valid"].astype(str).str.lower().isin(["true", "1", "yes"])]
    df["ad_start_sec"] = pd.to_numeric(df["ad_start_sec"], errors="coerce")
    df["ad_end_sec"] = pd.to_numeric(df["ad_end_sec"], errors="coerce")
    df = df[np.isfinite(df["ad_start_sec"]) & np.isfinite(df["ad_end_sec"]) & (df["ad_end_sec"] > df["ad_start_sec"])].copy()
    df["video_id"] = df["video_id"].astype(int)
    return df.sort_values(["video_id", "ad_start_sec", "ad_interval_id"]).reset_index(drop=True)


def add_overlap_labels(features: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    interval_map: Dict[int, List[Dict[str, Any]]] = {}
    for _, row in intervals.iterrows():
        interval_map.setdefault(int(row["video_id"]), []).append(row.to_dict())
    overlap_duration: List[float] = []
    overlap_ratio: List[float] = []
    overlap_ids: List[str] = []
    min_distance: List[float] = []
    context_labels: List[str] = []
    for _, row in out.iterrows():
        video_id = int(row["video_id"])
        start = safe_float(row["start_sec"])
        end = safe_float(row["end_sec"])
        duration = max(EPS, safe_float(row["duration_sec"], end - start))
        overlaps = []
        ids = []
        distances = []
        labels = set()
        for interval in interval_map.get(video_id, []):
            ad_start = safe_float(interval["ad_start_sec"])
            ad_end = safe_float(interval["ad_end_sec"])
            ov = interval_overlap(start, end, ad_start, ad_end)
            if ov > 0:
                overlaps.append((max(start, ad_start), min(end, ad_end)))
                ids.append(str(interval.get("ad_interval_id", "")))
            distances.append(interval_distance(start, end, ad_start, ad_end))
            if interval_overlap(start, end, max(0.0, ad_start - 10.0), ad_start) > 0:
                labels.add("local_pre_10s")
            if interval_overlap(start, end, ad_start, min(ad_start + 10.0, ad_end)) > 0:
                labels.add("start_edge_10s")
            if ad_end - ad_start > 20.0:
                if interval_overlap(start, end, ad_start + 10.0, ad_end - 10.0) > 0:
                    labels.add("ad_body")
            else:
                if ov > 0:
                    labels.add("ad_body_short")
            if interval_overlap(start, end, max(ad_start, ad_end - 10.0), ad_end) > 0:
                labels.add("end_edge_10s")
            if interval_overlap(start, end, ad_end, ad_end + 10.0) > 0:
                labels.add("local_post_10s")
        covered = union_duration(overlaps)
        ratio = min(1.0, covered / duration) if duration > 0 else 0.0
        overlap_duration.append(covered)
        overlap_ratio.append(ratio)
        overlap_ids.append(";".join([x for x in ids if x]))
        min_distance.append(min(distances) if distances else np.nan)
        context_labels.append(";".join(sorted(labels)))
    out["ad_overlap_duration_sec"] = overlap_duration
    out["ad_overlap_ratio"] = overlap_ratio
    out["is_ad_overlap"] = out["ad_overlap_ratio"] > 0
    out["is_ad_core"] = out["ad_overlap_ratio"] >= 0.5
    out["overlapping_ad_interval_ids"] = overlap_ids
    out["min_distance_to_ad_interval_sec"] = min_distance
    out["is_clean_nonad"] = (out["ad_overlap_ratio"] == 0) & (out["min_distance_to_ad_interval_sec"] >= 10.0)
    out["audio_context_labels"] = context_labels
    return out


def compute_baseline_and_relative(features: pd.DataFrame, analysis_features: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    out = features.copy()
    rows: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    for video_id, group in out.groupby("video_id", sort=True):
        idx = group.index
        for feature in analysis_features:
            if feature not in out.columns:
                continue
            values = pd.to_numeric(group[feature], errors="coerce")
            finite = values[np.isfinite(values)]
            count = int(len(finite))
            if count == 0:
                median = iqr = fallback_scale = np.nan
                iqr_fallback_used = True
                percentiles = pd.Series(np.nan, index=idx)
                robust_z = pd.Series(np.nan, index=idx)
            else:
                q = finite.quantile([0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95])
                median = float(q.loc[0.50])
                iqr = float(q.loc[0.75] - q.loc[0.25])
                std = float(finite.std(ddof=1)) if count > 1 else 0.0
                range_scale = float(finite.max() - finite.min()) if count > 1 else 0.0
                fallback_scale = iqr
                iqr_fallback_used = False
                if not np.isfinite(fallback_scale) or fallback_scale <= 1e-9:
                    iqr_fallback_used = True
                    fallback_scale = std if np.isfinite(std) and std > 1e-9 else range_scale
                if not np.isfinite(fallback_scale) or fallback_scale <= 1e-9:
                    fallback_scale = 1.0
                    warnings.append({
                        "warning_type": "baseline_scale_defaulted",
                        "video_id": int(video_id),
                        "feature_name": feature,
                        "warning_note": "IQR/std/range were zero or invalid; fallback_scale set to 1.0",
                    })
                elif iqr_fallback_used:
                    warnings.append({
                        "warning_type": "baseline_iqr_fallback_used",
                        "video_id": int(video_id),
                        "feature_name": feature,
                        "warning_note": f"IQR invalid or near zero; fallback_scale={fallback_scale}",
                    })
                percentiles = values.rank(pct=True, method="average") * 100.0
                robust_z = (values - median) / fallback_scale
            out.loc[idx, f"{feature}__video_median"] = median
            out.loc[idx, f"{feature}__video_iqr"] = iqr
            out.loc[idx, f"{feature}__video_robust_z"] = robust_z
            out.loc[idx, f"{feature}__video_percentile"] = percentiles
            out.loc[idx, f"{feature}__relative_level"] = pd.cut(
                percentiles,
                bins=[-np.inf, 40.0, 70.0, np.inf],
                labels=["low", "medium", "high"],
                right=False,
            ).astype("object")
            if count:
                row = {
                    "video_id": int(video_id),
                    "feature_name": feature,
                    "count": count,
                    "mean": float(finite.mean()),
                    "std": float(finite.std(ddof=1)) if count > 1 else 0.0,
                    "median": float(finite.median()),
                    "IQR": float(finite.quantile(0.75) - finite.quantile(0.25)),
                    "q05": float(finite.quantile(0.05)),
                    "q10": float(finite.quantile(0.10)),
                    "q25": float(finite.quantile(0.25)),
                    "q40": float(finite.quantile(0.40)),
                    "q50": float(finite.quantile(0.50)),
                    "q60": float(finite.quantile(0.60)),
                    "q70": float(finite.quantile(0.70)),
                    "q75": float(finite.quantile(0.75)),
                    "q80": float(finite.quantile(0.80)),
                    "q90": float(finite.quantile(0.90)),
                    "q95": float(finite.quantile(0.95)),
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "iqr_fallback_used": bool(iqr_fallback_used),
                    "fallback_scale": float(fallback_scale) if np.isfinite(fallback_scale) else np.nan,
                    "baseline_scope": "same_video_only",
                }
            else:
                row = {
                    "video_id": int(video_id),
                    "feature_name": feature,
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "IQR": np.nan,
                    "q05": np.nan,
                    "q10": np.nan,
                    "q25": np.nan,
                    "q40": np.nan,
                    "q50": np.nan,
                    "q60": np.nan,
                    "q70": np.nan,
                    "q75": np.nan,
                    "q80": np.nan,
                    "q90": np.nan,
                    "q95": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "iqr_fallback_used": True,
                    "fallback_scale": np.nan,
                    "baseline_scope": "same_video_only",
                }
            rows.append(row)
    return out, pd.DataFrame(rows), warnings


def add_relative_scores(df: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    active_cols = [f"{f}__video_percentile" for f in feature_sets["higher_active_features"] if f"{f}__video_percentile" in out.columns]
    quiet_cols = [f"{f}__video_percentile" for f in feature_sets["quiet_features"] if f"{f}__video_percentile" in out.columns]
    if active_cols:
        out["per_video_relative_active_audio_score"] = out[active_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) / 100.0
    else:
        out["per_video_relative_active_audio_score"] = np.nan
    if quiet_cols:
        out["per_video_relative_quiet_audio_score"] = out[quiet_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) / 100.0
    else:
        out["per_video_relative_quiet_audio_score"] = np.nan
    out["per_video_relative_active_audio_score"] = out["per_video_relative_active_audio_score"].clip(0, 1)
    out["per_video_relative_quiet_audio_score"] = out["per_video_relative_quiet_audio_score"].clip(0, 1)
    local_scores = []
    local_available = []
    for video_id, group in out.groupby("video_id", sort=False):
        group = group.sort_values("start_sec")
        active = pd.to_numeric(group["per_video_relative_active_audio_score"], errors="coerce")
        quiet = pd.to_numeric(group["per_video_relative_quiet_audio_score"], errors="coerce")
        starts = pd.to_numeric(group["start_sec"], errors="coerce")
        ends = pd.to_numeric(group["end_sec"], errors="coerce")
        for idx, start, end, act, qui in zip(group.index, starts, ends, active, quiet):
            mask = ((ends <= start) & (ends > start - 10.0)) | ((starts >= end) & (starts < end + 10.0))
            context_active = active[mask]
            context_quiet = quiet[mask]
            if context_active.notna().sum() == 0 and context_quiet.notna().sum() == 0:
                local_scores.append((idx, 0.0))
                local_available.append((idx, False))
                continue
            active_shift = abs(act - context_active.median()) if np.isfinite(act) and context_active.notna().any() else 0.0
            quiet_shift = abs(qui - context_quiet.median()) if np.isfinite(qui) and context_quiet.notna().any() else 0.0
            local_scores.append((idx, clip01(active_shift + 0.5 * quiet_shift)))
            local_available.append((idx, True))
    for idx, score in local_scores:
        out.loc[idx, "per_video_local_context_shift_score"] = score
    for idx, flag in local_available:
        out.loc[idx, "local_context_available"] = flag
    sustained = []
    for video_id, group in out.groupby("video_id", sort=False):
        group = group.sort_values("start_sec")
        roll = pd.to_numeric(group["per_video_relative_active_audio_score"], errors="coerce").rolling(3, center=True, min_periods=1).mean()
        for idx, value in zip(group.index, roll):
            sustained.append((idx, clip01(value)))
    for idx, value in sustained:
        out.loc[idx, "per_video_sustained_context_score"] = value
    if "audio_ad_like_score" in out.columns:
        out["audio_ad_like_score_reference"] = pd.to_numeric(out["audio_ad_like_score"], errors="coerce")
    else:
        out["audio_ad_like_score_reference"] = np.nan
    return out


def profile_summary(df: pd.DataFrame, features: Sequence[str], mask_col: str, label: str, intervals: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for video_id, group in df.groupby("video_id", sort=True):
        subset = group[group[mask_col].astype(bool)]
        interval_count = int(intervals[intervals["video_id"].eq(video_id)]["ad_interval_id"].nunique())
        for feature in features:
            if feature not in subset.columns:
                values = pd.Series(dtype=float)
            else:
                values = pd.to_numeric(subset[feature], errors="coerce")
            finite = values[np.isfinite(values)]
            levels = subset.get(f"{feature}__relative_level", pd.Series(dtype=object))
            row = {
                "video_id": int(video_id),
                "feature_name": feature,
                f"{label}_median": float(finite.median()) if len(finite) else np.nan,
                f"{label}_mean": float(finite.mean()) if len(finite) else np.nan,
                f"{label}_iqr": float(finite.quantile(0.75) - finite.quantile(0.25)) if len(finite) else np.nan,
                f"{label}_q25": float(finite.quantile(0.25)) if len(finite) else np.nan,
                f"{label}_q75": float(finite.quantile(0.75)) if len(finite) else np.nan,
                f"{label}_high_ratio": float((levels == "high").mean()) if len(levels) else np.nan,
                f"{label}_medium_or_high_ratio": float(levels.isin(["medium", "high"]).mean()) if len(levels) else np.nan,
                f"{label}_low_ratio": float((levels == "low").mean()) if len(levels) else np.nan,
                f"num_{label}_subwindows": int(len(subset)),
                f"{label}_profile_available": bool(len(finite) > 0),
                "profile_scope": "same_video_only",
            }
            if label == "ad":
                row["num_ad_intervals"] = interval_count
            rows.append(row)
    return pd.DataFrame(rows)


def compute_contrast(ad_profile: pd.DataFrame, nonad_profile: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    merged = ad_profile.merge(nonad_profile, on=["video_id", "feature_name"], how="outer")
    active = set(feature_sets["higher_active_features"])
    quiet = set(feature_sets["quiet_features"])
    rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        feature = row["feature_name"]
        ad_med = safe_float(row.get("ad_median"))
        nonad_med = safe_float(row.get("nonad_median"))
        ad_iqr = safe_float(row.get("ad_iqr"), 0.0)
        nonad_iqr = safe_float(row.get("nonad_iqr"), 0.0)
        available = bool(row.get("ad_profile_available", False)) and bool(row.get("nonad_profile_available", False))
        if not available or not np.isfinite(ad_med) or not np.isfinite(nonad_med):
            median_delta = np.nan
            effect = np.nan
            direction = "unavailable"
            usefulness = "unavailable"
            recommendation = "unavailable"
        else:
            median_delta = ad_med - nonad_med
            pooled = np.nanmean([ad_iqr, nonad_iqr])
            if not np.isfinite(pooled) or pooled <= 1e-9:
                pooled = 1.0
            effect = median_delta / pooled
            if abs(effect) >= 0.8:
                usefulness = "strong"
            elif abs(effect) >= 0.4:
                usefulness = "moderate"
            elif abs(effect) >= 0.15:
                usefulness = "weak"
            else:
                usefulness = "inconsistent"
            if median_delta > 0:
                direction = "higher_in_ad"
            elif median_delta < 0:
                direction = "lower_in_ad"
            else:
                direction = "flat"
            if feature in active:
                recommendation = "higher_values_support_ad_in_this_video" if median_delta > 0 else "lower_values_support_ad_in_this_video"
            elif feature in quiet:
                recommendation = "quieter_values_support_ad_in_this_video" if median_delta > 0 else "less_quiet_values_support_ad_in_this_video"
            else:
                recommendation = "use_as_video_specific_auxiliary_contrast"
        rows.append({
            "video_id": int(row["video_id"]),
            "feature_name": feature,
            "ad_median": ad_med,
            "nonad_median": nonad_med,
            "median_delta": median_delta,
            "robust_effect_size": effect,
            "direction_in_this_video": direction,
            "usefulness_level_in_this_video": usefulness,
            "recommendation_for_this_video_candidate_score": recommendation,
            "caution_note": "same-video train-label contrast; discussion only and overfit-prone",
            "contrast_scope": "same_video_only",
        })
    return pd.DataFrame(rows)


def add_contrast_candidate_score(df: pd.DataFrame, contrast: pd.DataFrame, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    score_features = list(dict.fromkeys(feature_sets["higher_active_features"] + feature_sets["quiet_features"] + feature_sets["texture_shape_features"] + feature_sets["mfcc_std_features"]))
    contrast_map: Dict[int, List[Dict[str, Any]]] = {}
    for video_id, group in contrast.groupby("video_id", sort=True):
        usable = group[group["usefulness_level_in_this_video"].isin(["strong", "moderate", "weak"])].copy()
        usable = usable[usable["feature_name"].isin(score_features)]
        contrast_map[int(video_id)] = usable.to_dict(orient="records")
    contrast_scores = []
    contrast_available = []
    top_support_features = []
    for _, row in out.iterrows():
        video_id = int(row["video_id"])
        items = contrast_map.get(video_id, [])
        weighted = []
        weights = []
        feature_notes = []
        for item in items:
            feature = item["feature_name"]
            pct = safe_float(row.get(f"{feature}__video_percentile"))
            effect = safe_float(item.get("robust_effect_size"))
            if not np.isfinite(pct) or not np.isfinite(effect):
                continue
            direction = 1.0 if effect >= 0 else -1.0
            support = pct / 100.0 if direction > 0 else 1.0 - pct / 100.0
            weight = min(1.0, abs(effect) / 1.5)
            if weight <= 0:
                continue
            weighted.append(clip01(support) * weight)
            weights.append(weight)
            if weight >= 0.25:
                feature_notes.append(f"{feature}:{effect:.2f}")
        if weights:
            contrast_scores.append(float(sum(weighted) / sum(weights)))
            contrast_available.append(True)
            top_support_features.append(";".join(feature_notes[:6]))
        else:
            contrast_scores.append(np.nan)
            contrast_available.append(False)
            top_support_features.append("")
    out["per_video_ad_vs_nonad_contrast_score"] = contrast_scores
    out["per_video_ad_vs_nonad_contrast_available"] = contrast_available
    out["candidate_supporting_contrast_features"] = top_support_features
    component_cols = [
        "per_video_relative_active_audio_score",
        "per_video_relative_quiet_audio_score",
        "per_video_local_context_shift_score",
        "per_video_ad_vs_nonad_contrast_score",
        "per_video_sustained_context_score",
    ]
    weights = {
        "per_video_relative_active_audio_score": 0.35,
        "per_video_relative_quiet_audio_score": 0.10,
        "per_video_local_context_shift_score": 0.20,
        "per_video_ad_vs_nonad_contrast_score": 0.25,
        "per_video_sustained_context_score": 0.10,
    }
    final_scores = []
    for _, row in out.iterrows():
        total = 0.0
        weight_sum = 0.0
        for col in component_cols:
            value = safe_float(row.get(col))
            if np.isfinite(value):
                total += clip01(value) * weights[col]
                weight_sum += weights[col]
        final_scores.append(clip01(total / weight_sum) if weight_sum else np.nan)
    out["audio_candidate_score_for_discussion"] = final_scores
    out["audio_candidate_score_scope"] = "same_video_baseline_and_profiles_only"
    out["audio_candidate_score_usage"] = "discussion_only_not_detector_ready"
    pattern_labels = []
    for _, row in out.iterrows():
        active = safe_float(row.get("per_video_relative_active_audio_score"), 0.0)
        quiet = safe_float(row.get("per_video_relative_quiet_audio_score"), 0.0)
        shift = safe_float(row.get("per_video_local_context_shift_score"), 0.0)
        sustained = safe_float(row.get("per_video_sustained_context_score"), 0.0)
        candidate = safe_float(row.get("audio_candidate_score_for_discussion"), 0.0)
        if active >= 0.70 and sustained >= 0.60:
            label = "sustained_active_high"
        elif active >= 0.40 and sustained >= 0.35:
            label = "sustained_medium_active"
        elif quiet >= 0.70 and active < 0.45:
            label = "quiet_or_low_energy_shift"
        elif shift >= 0.55:
            label = "local_context_shift"
        elif candidate < 0.30:
            label = "audio_not_informative"
        else:
            label = "mixed_audio_pattern"
        pattern_labels.append(label)
    out["audio_pattern_label"] = pattern_labels
    return out


def make_windows(intervals: pd.DataFrame, video_duration_by_id: Dict[int, float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in intervals.iterrows():
        video_id = int(row["video_id"])
        ad_start = safe_float(row["ad_start_sec"])
        ad_end = safe_float(row["ad_end_sec"])
        video_duration = safe_float(video_duration_by_id.get(video_id, np.nan), ad_end)
        specs = [
            ("pre_10s", max(0.0, ad_start - 10.0), ad_start),
            ("start_edge_10s", ad_start, min(ad_start + 10.0, ad_end)),
        ]
        if ad_end - ad_start > 20.0:
            specs.append(("ad_body", ad_start + 10.0, ad_end - 10.0))
        else:
            specs.append(("ad_body_short", ad_start, ad_end))
        specs += [
            ("end_edge_10s", max(ad_start, ad_end - 10.0), ad_end),
            ("post_10s", ad_end, min(video_duration, ad_end + 10.0)),
            ("full_ad_interval", ad_start, ad_end),
        ]
        for window_type, start, end in specs:
            if end <= start:
                continue
            rows.append({
                "video_id": video_id,
                "ad_interval_id": row.get("ad_interval_id", ""),
                "segment_id": row.get("segment_id", ""),
                "ad_start_sec": ad_start,
                "ad_end_sec": ad_end,
                "window_type": window_type,
                "window_start_sec": start,
                "window_end_sec": end,
                "representative_duration_sec": end - start,
            })
    return pd.DataFrame(rows)


def summarize_window(df: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, window in windows.iterrows():
        video_id = int(window["video_id"])
        start = safe_float(window["window_start_sec"])
        end = safe_float(window["window_end_sec"])
        group = df[df["video_id"].eq(video_id)].copy()
        mask = (pd.to_numeric(group["start_sec"], errors="coerce") < end) & (pd.to_numeric(group["end_sec"], errors="coerce") > start)
        subset = group[mask].copy()
        clipped = [(max(start, safe_float(r["start_sec"])), min(end, safe_float(r["end_sec"]))) for _, r in subset.iterrows()]
        covered = union_duration(clipped)
        active = pd.to_numeric(subset.get("per_video_relative_active_audio_score", pd.Series(dtype=float)), errors="coerce")
        quiet = pd.to_numeric(subset.get("per_video_relative_quiet_audio_score", pd.Series(dtype=float)), errors="coerce")
        shift = pd.to_numeric(subset.get("per_video_local_context_shift_score", pd.Series(dtype=float)), errors="coerce")
        labels = subset.get("audio_pattern_label", pd.Series(dtype=object))
        if len(subset) < 2:
            pattern = "insufficient_audio_window"
        elif (active >= 0.70).mean() >= 0.50:
            pattern = "sustained_active_high"
        elif (active >= 0.40).mean() >= 0.60:
            pattern = "sustained_medium_active"
        elif ((quiet >= 0.70) & (active < 0.45)).mean() >= 0.40:
            pattern = "quiet_or_low_energy_shift"
        elif (shift >= 0.55).mean() >= 0.40:
            pattern = "local_context_shift"
        elif labels.eq("audio_not_informative").mean() >= 0.60 if len(labels) else False:
            pattern = "audio_not_informative"
        else:
            pattern = "mixed_audio_pattern"
        def med(col: str) -> float:
            return safe_float(pd.to_numeric(subset.get(col, pd.Series(dtype=float)), errors="coerce").median()) if len(subset) else np.nan
        row = {
            "video_id": video_id,
            "ad_interval_id": window["ad_interval_id"],
            "segment_id": window["segment_id"],
            "window_type": window["window_type"],
            "window_start_sec": start,
            "window_end_sec": end,
            "num_subwindows": int(len(subset)),
            "covered_duration_sec": covered,
            "representative_duration_sec": safe_float(window["representative_duration_sec"]),
            "relative_active_audio_score_mean": safe_float(active.mean()) if len(active) else np.nan,
            "relative_active_audio_score_median": safe_float(active.median()) if len(active) else np.nan,
            "relative_active_audio_score_max": safe_float(active.max()) if len(active) else np.nan,
            "relative_quiet_audio_score_mean": safe_float(quiet.mean()) if len(quiet) else np.nan,
            "relative_quiet_audio_score_median": safe_float(quiet.median()) if len(quiet) else np.nan,
            "local_context_shift_score_mean": safe_float(shift.mean()) if len(shift) else np.nan,
            "high_active_ratio": safe_float((active >= 0.70).mean()) if len(active) else np.nan,
            "medium_or_high_active_ratio": safe_float((active >= 0.40).mean()) if len(active) else np.nan,
            "high_quiet_ratio": safe_float((quiet >= 0.70).mean()) if len(quiet) else np.nan,
            "low_energy_ratio_mean": safe_float(pd.to_numeric(subset.get("low_energy_ratio", pd.Series(dtype=float)), errors="coerce").mean()) if len(subset) else np.nan,
            "silence_ratio_mean": safe_float(pd.to_numeric(subset.get("silence_ratio", pd.Series(dtype=float)), errors="coerce").mean()) if len(subset) else np.nan,
            "audio_rms_mean_median": med("audio_rms_mean"),
            "audio_log_energy_mean_median": med("audio_log_energy_mean"),
            "spectral_flux_mean_median": med("spectral_flux_mean"),
            "onset_density_median": med("onset_density"),
            "audio_ad_like_score_median": med("audio_ad_like_score_reference"),
            "audio_candidate_score_median": med("audio_candidate_score_for_discussion"),
            "audio_pattern_label": pattern,
            "summary_scope": "same_video_relative_metrics",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_interval_summary(window_summary: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    full = window_summary[window_summary["window_type"].eq("full_ad_interval")].copy()
    rows: List[Dict[str, Any]] = []
    for _, interval in intervals.iterrows():
        video_id = int(interval["video_id"])
        ad_interval_id = interval["ad_interval_id"]
        group = window_summary[(window_summary["video_id"].eq(video_id)) & (window_summary["ad_interval_id"].astype(str).eq(str(ad_interval_id)))]
        full_row = full[(full["video_id"].eq(video_id)) & (full["ad_interval_id"].astype(str).eq(str(ad_interval_id)))]
        row: Dict[str, Any] = {
            "video_id": video_id,
            "ad_interval_id": ad_interval_id,
            "segment_id": interval.get("segment_id", ""),
            "ad_start_sec": safe_float(interval["ad_start_sec"]),
            "ad_end_sec": safe_float(interval["ad_end_sec"]),
            "ad_duration_sec": safe_float(interval["ad_end_sec"]) - safe_float(interval["ad_start_sec"]),
        }
        if not full_row.empty:
            fr = full_row.iloc[0]
            for col in [
                "num_subwindows", "covered_duration_sec", "relative_active_audio_score_mean",
                "relative_active_audio_score_median", "relative_active_audio_score_max",
                "relative_quiet_audio_score_mean", "relative_quiet_audio_score_median",
                "local_context_shift_score_mean", "high_active_ratio", "medium_or_high_active_ratio",
                "high_quiet_ratio", "audio_candidate_score_median", "audio_pattern_label",
            ]:
                row[f"full_interval_{col}"] = fr.get(col, np.nan)
        labels = group.set_index("window_type")["audio_pattern_label"].to_dict() if not group.empty else {}
        row["window_pattern_labels"] = json.dumps(labels, ensure_ascii=False)
        row["interval_summary_scope"] = "same_video_relative_metrics"
        rows.append(row)
    return pd.DataFrame(rows)


def merge_review_segments(df: pd.DataFrame, intervals: pd.DataFrame, threshold: float = 0.70, max_gap_sec: float = 4.0, min_duration_sec: float = 4.0) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    interval_map = {vid: g.to_dict(orient="records") for vid, g in intervals.groupby("video_id", sort=True)}
    for video_id, group in df.groupby("video_id", sort=True):
        cand = group[pd.to_numeric(group["audio_candidate_score_for_discussion"], errors="coerce") >= threshold].sort_values("start_sec")
        segments: List[List[pd.Series]] = []
        current: List[pd.Series] = []
        current_end = None
        for _, row in cand.iterrows():
            start = safe_float(row["start_sec"])
            end = safe_float(row["end_sec"])
            if not current:
                current = [row]
                current_end = end
            elif start - safe_float(current_end) <= max_gap_sec:
                current.append(row)
                current_end = max(safe_float(current_end), end)
            else:
                segments.append(current)
                current = [row]
                current_end = end
        if current:
            segments.append(current)
        kept_idx = 0
        for seg in segments:
            seg_df = pd.DataFrame([s.to_dict() for s in seg])
            start = safe_float(seg_df["start_sec"].min())
            end = safe_float(seg_df["end_sec"].max())
            duration = end - start
            if duration < min_duration_sec:
                continue
            overlaps = []
            ids = []
            max_ratio = 0.0
            for interval in interval_map.get(int(video_id), []):
                ad_start = safe_float(interval["ad_start_sec"])
                ad_end = safe_float(interval["ad_end_sec"])
                ov = interval_overlap(start, end, ad_start, ad_end)
                if ov > 0:
                    overlaps.append((max(start, ad_start), min(end, ad_end)))
                    ids.append(str(interval.get("ad_interval_id", "")))
                    max_ratio = max(max_ratio, ov / max(EPS, min(duration, ad_end - ad_start)))
            label_counts = seg_df["audio_pattern_label"].value_counts(dropna=False)
            dominant = str(label_counts.index[0]) if len(label_counts) else ""
            kept_idx += 1
            rows.append({
                "video_id": int(video_id),
                "segment_id": f"V{int(video_id):02d}_AUDREV_{kept_idx:04d}",
                "segment_start_sec": start,
                "segment_end_sec": end,
                "segment_duration_sec": duration,
                "mean_audio_candidate_score": safe_float(pd.to_numeric(seg_df["audio_candidate_score_for_discussion"], errors="coerce").mean()),
                "max_audio_candidate_score": safe_float(pd.to_numeric(seg_df["audio_candidate_score_for_discussion"], errors="coerce").max()),
                "dominant_audio_pattern_label": dominant,
                "overlaps_actual_ad": bool(overlaps),
                "max_ad_overlap_ratio": clip01(max_ratio),
                "overlapping_ad_interval_ids": ";".join(sorted(set([x for x in ids if x]))),
                "review_note": "discussion-only audio segment; merge gap <=4s, threshold is review-only not detector threshold",
            })
    return pd.DataFrame(rows)


def global_reference_summaries(window_summary: pd.DataFrame, contrast: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pattern_rows = []
    for (window_type, label), group in window_summary.groupby(["window_type", "audio_pattern_label"], dropna=False):
        pattern_rows.append({
            "window_type": window_type,
            "audio_pattern_label": label,
            "count": int(len(group)),
            "mean_active_score": safe_float(pd.to_numeric(group["relative_active_audio_score_mean"], errors="coerce").mean()),
            "mean_candidate_score_median": safe_float(pd.to_numeric(group["audio_candidate_score_median"], errors="coerce").mean()),
            "global_summary_usage": "descriptive_only_not_used_for_candidate_score",
        })
    direction_rows = []
    for feature, group in contrast.groupby("feature_name", sort=True):
        effect = pd.to_numeric(group["robust_effect_size"], errors="coerce")
        direction_rows.append({
            "feature_name": feature,
            "num_videos": int(group["video_id"].nunique()),
            "median_robust_effect_size": safe_float(effect.median()),
            "strong_or_moderate_video_count": int(group["usefulness_level_in_this_video"].isin(["strong", "moderate"]).sum()),
            "higher_in_ad_count": int(group["direction_in_this_video"].eq("higher_in_ad").sum()),
            "lower_in_ad_count": int(group["direction_in_this_video"].eq("lower_in_ad").sum()),
            "global_summary_usage": "descriptive_only_not_used_for_candidate_score",
        })
    return pd.DataFrame(pattern_rows), pd.DataFrame(direction_rows)


def write_rule_direction(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = """# Per-Video Relative Audio Rule Direction v2_4 Train

## 기존 방향
“오디오가 갑자기 커지면 광고 가능성”처럼 절대 음량 변화에 기대는 방식은 영상별 녹음 환경 차이에 취약하다.

## 변경 방향
“각 영상 전체 오디오 흐름 대비 해당 영상 안의 특정 구간이 상대적으로 높거나 낮거나, 해당 영상의 non-ad 흐름과 다르면 보조 evidence로 사용”한다.

## 핵심 원칙
A 영상은 A 영상 내부 기준으로만 분석하고, B 영상은 B 영상 내부 기준으로만 분석한다. 서로 다른 영상 간 raw audio feature 절대값을 직접 비교하지 않는다.

## 사용 가능한 evidence type
1. per_video_sustained_relative_active_high
2. per_video_sustained_relative_medium_active
3. per_video_quiet_or_low_energy_shift
4. per_video_local_context_shift
5. per_video_ad_vs_nonad_contrast
6. audio_not_informative

## 최종 fusion에서의 사용 예
- OCR/장면 전환이 강하고 오디오도 active_high이면 confidence 증가
- OCR이 강하지만 오디오가 not_informative이면 OCR/scene 중심 판단 유지
- 오디오만 강하면 광고 확정이 아니라 후보 유지
- 오디오가 낮거나 조용한 광고 패턴도 가능하므로 loudness-only rule 금지
- 영상마다 녹음 환경이 다르므로 영상 간 절대 음량 비교 금지

## 주의
이 산출물은 train actual label을 사용한 discussion/evidence audit용이다. detector-ready threshold나 final rule이 아니다.
"""
    path.write_text(text, encoding="utf-8")


def write_summary(path: Path, report: Dict[str, Any], top_feature_summary: pd.DataFrame, pattern_counts: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Per-Video Relative Audio Evidence Audit v2_4 Train",
        "",
        "## Purpose",
        "2.1 full-video 2초 audio subwindow feature를 사용해 각 video_id 내부 baseline 대비 실제 광고 구간의 상대적 audio pattern을 분석했다.",
        "",
        "## Scope",
        "- Detector rule/config/output은 수정하지 않았다.",
        "- OCR/장면전환 결과와 병합하지 않았다.",
        "- Validation/test row-level output은 생성하지 않았다.",
        "- Global summary는 descriptive only이며 candidate score 계산에 쓰지 않았다.",
        "- 모든 score와 relative metric은 같은 video_id 내부 baseline/profile만 사용했다.",
        "",
        "## Inputs",
    ]
    for name, value in report["input_files"].items():
        lines.append(f"- {name}: `{value}`")
    lines += [
        "",
        "## Feature Set",
        "- " + ", ".join(f"`{f}`" for f in report["selected_features"]["analysis_features"]),
        "",
        "## Counts",
        f"- 2.1 full-video feature rows: {report['full_video_feature_row_count']}",
        f"- Train actual ad intervals: {report['train_actual_ad_interval_count']}",
        f"- Per-video ad profile videos: {report['profile_counts']['ad_profile_video_count']}",
        f"- Per-video clean non-ad profile videos: {report['profile_counts']['clean_nonad_profile_video_count']}",
        f"- Top review segments: {report['top_review_segment_count']}",
        "",
        "## Score Construction",
        "- `per_video_relative_active_audio_score`: 같은 video_id 내부 higher-active raw feature percentile 평균",
        "- `per_video_relative_quiet_audio_score`: 같은 video_id 내부 silence/low-energy percentile 평균",
        "- `per_video_local_context_shift_score`: 같은 video_id 내부 현재 구간과 전후 10초 context score 차이",
        "- `per_video_ad_vs_nonad_contrast_score`: 같은 video_id 내부 actual ad core와 clean non-ad profile 차이에 대한 alignment score",
        "- `audio_ad_like_score_reference`: 기존 score를 참고용으로만 유지, raw relative score와 섞어 해석하지 않음",
        "- `audio_candidate_score_for_discussion`: discussion only, not detector-ready",
        "",
        "## Audio Pattern Label Counts",
    ]
    for label, count in pattern_counts.items():
        lines.append(f"- {label}: {count}")
    lines += ["", "## Per-Video Strong Contrast Features"]
    if top_feature_summary.empty:
        lines.append("- No contrast summary available.")
    else:
        for _, row in top_feature_summary.iterrows():
            lines.append(
                f"- video {int(row['video_id'])}: "
                f"{row['top_features']}"
            )
    lines += [
        "",
        "## Conclusions For 2.2",
        "오디오는 같은 영상 내부의 상대적 active/quiet/context-shift/non-ad contrast를 보조 evidence로 제공할 수 있다. 하지만 train label을 사용해 만든 discussion score이므로 과적합 위험이 있고, 오디오 단독으로 광고를 확정하면 위험하다. 다음 fusion 단계에서는 `video_id`, `subwindow_id`, `start_sec`, `end_sec`, `ad_overlap_ratio`, `audio_candidate_score_for_discussion`, `audio_pattern_label`을 OCR/장면 전환 결과와 시간 기준으로 병합하는 것이 적합하다.",
        "",
        "## Generated Files",
    ]
    for key, value in report["output_files"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Safety Flags",
        "- detector_rule_modified=false",
        "- old_project_modified=false",
        "- validation_test_row_level_output_created=false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latest_bundle(bundle_dir: Path, project_root: Path, paths: Dict[str, Path], report: Dict[str, Any], include_max_mb: float = 10.0) -> Dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    copy_keys = [
        "summary_md", "report_json", "rule_direction_md", "baseline_summary", "window_summary", "interval_summary",
        "ad_profile", "clean_nonad_profile", "contrast", "top_segments", "warnings", "global_ad_pattern",
        "global_feature_direction", "run_log",
    ]
    script_path = project_root / SCRIPT_RELATIVE_PATH
    copy_sources = {"script": script_path}
    copy_sources.update({key: paths[key] for key in copy_keys if key in paths})
    maybe_large = ["relative_levels", "candidate_score"]
    large_notes: Dict[str, Dict[str, Any]] = {}
    for key in maybe_large:
        path = paths[key]
        size = path.stat().st_size if path.exists() else 0
        rows = report.get("row_counts", {}).get(key)
        include = path.exists() and size <= include_max_mb * 1024 * 1024
        large_notes[key] = {"path": str(path), "size_bytes": size, "row_count": rows, "included_in_bundle": include}
        if include:
            copy_sources[key] = path
    for key, src in copy_sources.items():
        if not src.exists() or src.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
            continue
        dst = bundle_dir / src.name
        if dst.exists():
            dst = unique_file_path(dst, report["run_id"])
        shutil.copy2(src, dst)
        copied.append(str(dst))
    readme = bundle_dir / "README_latest_files.md"
    lines = [
        "# Latest Files: Per-Video Relative Audio Evidence Audit v2_4 Train",
        "",
        "This bundle contains analysis/report artifacts only. Raw media/cache/model files are not copied.",
        "",
        "## Main Paths",
    ]
    for key, value in paths.items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Large Row-Level Files"]
    for key, note in large_notes.items():
        lines.append(f"- {key}: path=`{note['path']}`, rows={note['row_count']}, size_bytes={note['size_bytes']}, included={str(note['included_in_bundle']).lower()}")
    lines += ["", "## Copied Files"]
    for item in copied:
        lines.append(f"- `{item}`")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copied.append(str(readme))
    return {"bundle_dir": str(bundle_dir), "copied_files": copied, "large_file_notes": large_notes}


def validate_outputs(
    project_root: Path,
    split_df: pd.DataFrame,
    features: pd.DataFrame,
    intervals: pd.DataFrame,
    relative: pd.DataFrame,
    baseline: pd.DataFrame,
    ad_profile: pd.DataFrame,
    nonad_profile: pd.DataFrame,
    candidate: pd.DataFrame,
    global_used_for_score: bool,
    paths: Dict[str, Path],
    protected_before: Dict[str, Dict[str, Any]],
    bundle_dir: Optional[Path],
) -> Dict[str, Any]:
    split_validation = validate_split(split_df)
    train_ids = set(FIXED_SPLIT["train"])
    val_test_ids = set(FIXED_SPLIT["validation"] + FIXED_SPLIT["test"])
    full_ids = set(pd.to_numeric(features["video_id"], errors="coerce").dropna().astype(int).unique().tolist())
    output_has_val_test = False
    for df in [relative, candidate, ad_profile, nonad_profile, baseline]:
        if "video_id" in df.columns:
            ids = set(pd.to_numeric(df["video_id"], errors="coerce").dropna().astype(int).unique().tolist())
            output_has_val_test = output_has_val_test or bool(ids & val_test_ids)
    ratio = pd.to_numeric(relative["ad_overlap_ratio"], errors="coerce")
    clean = relative[relative["is_clean_nonad"].astype(bool)]
    clean_ok = bool((pd.to_numeric(clean["min_distance_to_ad_interval_sec"], errors="coerce") >= 10.0 - 1e-9).all()) if len(clean) else True
    percentile_cols = [c for c in relative.columns if c.endswith("__video_percentile")]
    percentile_ok = bool(all(((pd.to_numeric(relative[c], errors="coerce").dropna() >= 0) & (pd.to_numeric(relative[c], errors="coerce").dropna() <= 100)).all() for c in percentile_cols))
    score = pd.to_numeric(candidate["audio_candidate_score_for_discussion"], errors="coerce") if "audio_candidate_score_for_discussion" in candidate.columns else pd.Series(dtype=float)
    score_ok = bool(((score.dropna() >= 0) & (score.dropna() <= 1)).all())
    protected_after = {name: snapshot_path(Path(snap["path"])) for name, snap in protected_before.items()}
    unchanged = {name: snapshot_unchanged(protected_before[name], protected_after[name]) for name in protected_before}
    forbidden = []
    if bundle_dir and bundle_dir.exists():
        for path in bundle_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
                forbidden.append(str(path))
    return {
        "input_split_validation": {
            "split_matches_fixed": split_validation,
            "full_video_feature_train_only": full_ids <= train_ids,
            "validation_test_video_id_in_outputs": output_has_val_test,
            "input_artifacts_exist": all(p.exists() for p in [
                paths["input_full_video"], paths["input_recommendations"], paths["input_labels"], paths["input_split"],
            ]),
        },
        "label_overlap_validation": {
            "train_actual_interval_count": int(len(intervals)),
            "ad_overlap_ratio_in_0_1": bool(((ratio >= -1e-9) & (ratio <= 1 + 1e-9)).all()),
            "clean_nonad_10s_away_from_actual_ads": clean_ok,
            "negative_overlap_detected": bool((pd.to_numeric(relative["ad_overlap_duration_sec"], errors="coerce") < -1e-9).any()),
        },
        "per_video_relative_baseline_validation": {
            "baseline_scope_same_video_only": bool((baseline["baseline_scope"] == "same_video_only").all()) if len(baseline) else False,
            "percentile_columns_in_0_100": percentile_ok,
            "iqr_fallback_rows": int(baseline["iqr_fallback_used"].astype(bool).sum()) if "iqr_fallback_used" in baseline.columns else 0,
            "raw_and_existing_score_kept_separate": "audio_ad_like_score_reference" in relative.columns,
            "cross_video_raw_comparison_used": False,
        },
        "profile_candidate_score_validation": {
            "ad_profile_scope_same_video_only": bool((ad_profile["profile_scope"] == "same_video_only").all()) if len(ad_profile) else False,
            "nonad_profile_scope_same_video_only": bool((nonad_profile["profile_scope"] == "same_video_only").all()) if len(nonad_profile) else False,
            "candidate_score_scope_same_video_only": bool(candidate["audio_candidate_score_scope"].eq("same_video_baseline_and_profiles_only").all()) if len(candidate) else False,
            "global_summary_used_for_score": global_used_for_score,
            "candidate_score_in_0_1": score_ok,
            "candidate_score_not_detector_threshold": bool(candidate["audio_candidate_score_usage"].eq("discussion_only_not_detector_ready").all()) if len(candidate) else False,
        },
        "output_safety_validation": {
            "protected_paths_unchanged": unchanged,
            "detector_rule_modified": not all(unchanged.get(name, True) for name in ["detector_scripts", "detector_configs", "detector_outputs"]),
            "old_project_modified": not unchanged.get("old_project", True),
            "raw_video_audio_cache_model_copied_to_latest_bundle": bool(forbidden),
            "forbidden_bundle_files": forbidden,
            "validation_test_row_level_output_created": output_has_val_test,
            "two_one_two_three_inputs_unchanged": all(unchanged.get(name, True) for name in ["two_one_full_video_output", "two_three_recommendations", "two_three_inventory", "two_three_score_formula"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--full-video-feature-file", default="data/audio/full_video_audio_subwindow_features_v2_4_train_20260526_1750_final.csv")
    parser.add_argument("--coverage-file", default="data/audio/train_actual_ad_audio_coverage_summary_v2_4_20260526_1750_final.csv")
    parser.add_argument("--recommendation-file", default="data/audio/audio_feature_recommendations_for_relative_analysis_v2_4_train_20260526_2116_final.csv")
    parser.add_argument("--feature-inventory-file", default="data/audio/audio_feature_inventory_audit_v2_4_train_20260526_2116_final.csv")
    parser.add_argument("--score-formula-file", default="data/audio/audio_score_formula_audit_v2_4_train_20260526_2116_final.csv")
    parser.add_argument("--label-file", default="data/segments/ad_interval_segments_v2_4.csv")
    parser.add_argument("--split-file", default="data/splits/video_split_v2_4.csv")
    parser.add_argument("--manifest-file", default="data/video_metadata/video_manifest_v2_2.csv")
    parser.add_argument("--candidate-review-threshold", type=float, default=0.70)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_paths = output_paths(project_root, run_id)
    logger = TaskLogger(out_paths["run_log"])
    t0 = time.time()

    logger.log("[STEP 01] Safety snapshot and output path planning")
    input_paths = {
        "input_full_video": resolve_existing(project_root / args.full_video_feature_file, "data/audio/full_video_audio_subwindow_features_v2_4_train*.csv", project_root),
        "input_coverage": resolve_existing(project_root / args.coverage_file, "data/audio/train_actual_ad_audio_coverage_summary_v2_4*.csv", project_root),
        "input_recommendations": resolve_existing(project_root / args.recommendation_file, "data/audio/audio_feature_recommendations_for_relative_analysis_v2_4_train*.csv", project_root),
        "input_inventory": resolve_existing(project_root / args.feature_inventory_file, "data/audio/audio_feature_inventory_audit_v2_4_train*.csv", project_root),
        "input_score_formula": resolve_existing(project_root / args.score_formula_file, "data/audio/audio_score_formula_audit_v2_4_train*.csv", project_root),
        "input_labels": project_root / args.label_file,
        "input_split": project_root / args.split_file,
        "input_manifest": project_root / args.manifest_file,
    }
    paths_for_validation = dict(out_paths)
    paths_for_validation.update(input_paths)
    protected_targets = {
        "old_project": OLD_PROJECT_ROOT,
        "detector_scripts": project_root / "scripts/detectors",
        "detector_configs": project_root / "configs/detectors",
        "detector_outputs": project_root / "data/predictions",
        "split_file": input_paths["input_split"],
        "label_file": input_paths["input_labels"],
        "raw_videos": project_root / "data/raw/videos",
        "two_one_full_video_output": input_paths["input_full_video"],
        "two_three_recommendations": input_paths["input_recommendations"],
        "two_three_inventory": input_paths["input_inventory"],
        "two_three_score_formula": input_paths["input_score_formula"],
    }
    protected_before = {name: snapshot_path(path) for name, path in protected_targets.items()}
    for path in out_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    logger.log("[STEP 02] Locate 2.1 full-video audio feature and 2.3 recommendation artifacts")
    missing_inputs = [name for name, path in input_paths.items() if not path.exists() and name not in {"input_coverage"}]
    if missing_inputs:
        raise FileNotFoundError(f"Missing required inputs: {missing_inputs}")

    logger.log("[STEP 03] Validate fixed train-only split scope")
    split_df = read_csv(input_paths["input_split"])
    split_validation = validate_split(split_df)
    if not split_validation["all_match"]:
        raise RuntimeError(f"Fixed split mismatch: {split_validation}")
    train_ids = FIXED_SPLIT["train"]

    logger.log("[STEP 04] Load full-video train audio subwindow features")
    full_df = read_csv(input_paths["input_full_video"])
    full_df["video_id"] = pd.to_numeric(full_df["video_id"], errors="coerce").astype(int)
    full_df = full_df[full_df["video_id"].isin(train_ids)].copy()
    if "split" in full_df.columns:
        full_df = full_df[full_df["split"].astype(str).eq("train")].copy()
    if "extraction_status" in full_df.columns:
        full_df = full_df[full_df["extraction_status"].astype(str).eq("success")].copy()
    full_df = full_df.sort_values(["video_id", "start_sec", "subwindow_id"]).reset_index(drop=True)

    logger.log("[STEP 05] Load actual ad intervals and compute subwindow overlap labels")
    label_df = read_csv(input_paths["input_labels"])
    intervals = load_train_intervals(label_df, train_ids)
    full_df = add_overlap_labels(full_df, intervals)

    logger.log("[STEP 06] Select 2.2 feature set from 2.3 recommendations")
    rec_df = read_csv(input_paths["input_recommendations"])
    warnings: List[Dict[str, Any]] = []
    feature_sets = select_features(rec_df, full_df.columns, warnings)

    logger.log("[STEP 07] Compute per-video audio baselines")
    relative_df, baseline_df, baseline_warnings = compute_baseline_and_relative(full_df, feature_sets["analysis_features"])
    warnings.extend(baseline_warnings)
    baseline_df.to_csv(out_paths["baseline_summary"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 08] Compute per-subwindow per-video relative levels and scores")
    relative_df = add_relative_scores(relative_df, feature_sets)

    logger.log("[STEP 10] Build per-video train ad audio profiles")
    ad_profile = profile_summary(relative_df, feature_sets["analysis_features"], "is_ad_core", "ad", intervals)
    ad_profile.to_csv(out_paths["ad_profile"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 11] Build per-video clean non-ad audio profiles")
    nonad_profile = profile_summary(relative_df, feature_sets["analysis_features"], "is_clean_nonad", "nonad", intervals)
    nonad_profile.to_csv(out_paths["clean_nonad_profile"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 12] Compute per-video ad vs non-ad feature contrast")
    contrast = compute_contrast(ad_profile, nonad_profile, feature_sets)
    contrast.to_csv(out_paths["contrast"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 13] Generate per-video audio candidate score for discussion")
    relative_df = add_contrast_candidate_score(relative_df, contrast, feature_sets)
    relative_df.to_csv(out_paths["relative_levels"], index=False, encoding="utf-8-sig")
    candidate_cols = [
        "video_id", "split", "subwindow_id", "start_sec", "end_sec", "duration_sec",
        "ad_overlap_duration_sec", "ad_overlap_ratio", "is_ad_overlap", "is_ad_core", "is_clean_nonad",
        "overlapping_ad_interval_ids", "audio_context_labels",
        "per_video_relative_active_audio_score", "per_video_relative_quiet_audio_score",
        "per_video_local_context_shift_score", "per_video_ad_vs_nonad_contrast_score",
        "per_video_ad_vs_nonad_contrast_available", "per_video_sustained_context_score",
        "audio_ad_like_score_reference", "audio_candidate_score_for_discussion", "audio_pattern_label",
        "candidate_supporting_contrast_features", "audio_candidate_score_scope", "audio_candidate_score_usage",
    ]
    candidate_df = relative_df[[c for c in candidate_cols if c in relative_df.columns]].copy()
    candidate_df.to_csv(out_paths["candidate_score"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 09] Summarize actual ad interval relative audio windows")
    video_duration_by_id = full_df.groupby("video_id")["video_duration_sec"].max().to_dict()
    windows = make_windows(intervals, video_duration_by_id)
    window_summary = summarize_window(relative_df, windows)
    window_summary.to_csv(out_paths["window_summary"], index=False, encoding="utf-8-sig")
    interval_summary = build_interval_summary(window_summary, intervals)
    interval_summary.to_csv(out_paths["interval_summary"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 14] Merge high-score subwindows into review segments")
    top_segments = merge_review_segments(relative_df, intervals, args.candidate_review_threshold)
    top_segments.to_csv(out_paths["top_segments"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 15] Generate descriptive-only global reference summaries")
    global_ad_pattern, global_feature_direction = global_reference_summaries(window_summary, contrast)
    global_ad_pattern.to_csv(out_paths["global_ad_pattern"], index=False, encoding="utf-8-sig")
    global_feature_direction.to_csv(out_paths["global_feature_direction"], index=False, encoding="utf-8-sig")

    warnings_df = pd.DataFrame(warnings) if warnings else pd.DataFrame(columns=["warning_type", "video_id", "feature_name", "warning_note"])
    warnings_df.to_csv(out_paths["warnings"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 16] Generate summary/report/rule direction documents")
    pattern_counts = relative_df["audio_pattern_label"].value_counts().to_dict()
    top_feature_rows = []
    for video_id, group in contrast[contrast["usefulness_level_in_this_video"].isin(["strong", "moderate"])].groupby("video_id"):
        g = group.copy()
        g["abs_effect"] = pd.to_numeric(g["robust_effect_size"], errors="coerce").abs()
        g = g.sort_values("abs_effect", ascending=False).head(5)
        top_feature_rows.append({
            "video_id": int(video_id),
            "top_features": "; ".join(f"{r.feature_name}({r.direction_in_this_video},{safe_float(r.robust_effect_size):.2f})" for r in g.itertuples()),
        })
    top_feature_summary = pd.DataFrame(top_feature_rows)
    profile_counts = {
        "ad_profile_video_count": int(ad_profile[ad_profile["ad_profile_available"].astype(bool)]["video_id"].nunique()) if "ad_profile_available" in ad_profile.columns else 0,
        "clean_nonad_profile_video_count": int(nonad_profile[nonad_profile["nonad_profile_available"].astype(bool)]["video_id"].nunique()) if "nonad_profile_available" in nonad_profile.columns else 0,
    }
    row_counts = {
        "baseline_summary": int(len(baseline_df)),
        "relative_levels": int(len(relative_df)),
        "window_summary": int(len(window_summary)),
        "interval_summary": int(len(interval_summary)),
        "ad_profile": int(len(ad_profile)),
        "clean_nonad_profile": int(len(nonad_profile)),
        "contrast": int(len(contrast)),
        "candidate_score": int(len(candidate_df)),
        "top_segments": int(len(top_segments)),
        "warnings": int(len(warnings_df)),
        "global_ad_pattern": int(len(global_ad_pattern)),
        "global_feature_direction": int(len(global_feature_direction)),
    }
    report: Dict[str, Any] = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "run_id": run_id,
        "created_at": now_iso(),
        "project_root": str(project_root),
        "purpose": "per-video relative audio baseline/profile/evidence audit using 2.1 full-video subwindow features",
        "not_detector_rule_modification": True,
        "train_video_ids": train_ids,
        "validation_video_ids": FIXED_SPLIT["validation"],
        "test_video_ids": FIXED_SPLIT["test"],
        "validation_test_excluded_from_row_level_outputs": True,
        "input_files": {key: str(value) for key, value in input_paths.items()},
        "full_video_feature_row_count": int(len(full_df)),
        "train_actual_ad_interval_count": int(len(intervals)),
        "recommendation_2_3_used": True,
        "selected_features": feature_sets,
        "baseline_method": "feature medians/IQR/percentiles computed independently within each video_id only",
        "cross_video_raw_audio_comparison_used": False,
        "relative_active_audio_score_method": "mean same-video percentile/100 over higher-is-more-active raw features",
        "relative_quiet_audio_score_method": "mean same-video percentile/100 over silence_ratio and low_energy_ratio",
        "local_context_shift_score_method": "same-video current active/quiet score shift versus previous/following 10s context",
        "per_video_ad_vs_nonad_contrast_score_method": "same-video actual-ad-core vs clean-nonad profile contrast; no global prototype",
        "audio_ad_like_score_reference_only": True,
        "global_summary_descriptive_only_not_used_for_score": True,
        "overfit_risk": "Uses train actual ad labels to build same-video discussion score; not detector-ready and overfit-prone.",
        "profile_counts": profile_counts,
        "pattern_counts": pattern_counts,
        "top_review_segment_count": int(len(top_segments)),
        "top_contrast_features_by_video": top_feature_summary.to_dict(orient="records"),
        "row_counts": row_counts,
        "candidate_review_threshold": args.candidate_review_threshold,
        "fusion_key_columns": ["video_id", "split", "subwindow_id", "start_sec", "end_sec", "duration_sec", "audio_candidate_score_for_discussion", "audio_pattern_label"],
        "output_files": {key: str(value) for key, value in out_paths.items()},
        "modified_files": [str(project_root / SCRIPT_RELATIVE_PATH)],
        "generated_files": [str(value) for value in out_paths.values()],
        "detector_rule_modified": False,
        "old_project_modified": False,
        "validation_test_row_level_output_created": False,
    }
    write_rule_direction(out_paths["rule_direction_md"])
    write_summary(out_paths["summary_md"], report, top_feature_summary, pattern_counts)
    save_json(out_paths["report_json"], report)

    logger.log("[STEP 17] Run Sub Agent validations")
    bundle_dir = project_root / "outputs/latest_for_chatgpt_per_video_relative_audio_evidence_audit_v2_4_train"
    validations = validate_outputs(
        project_root, split_df, full_df, intervals, relative_df, baseline_df, ad_profile, nonad_profile,
        candidate_df, False, paths_for_validation, protected_before, None,
    )
    report["sub_agent_validations"] = validations
    report["detector_rule_modified"] = bool(validations["output_safety_validation"]["detector_rule_modified"])
    report["old_project_modified"] = bool(validations["output_safety_validation"]["old_project_modified"])
    report["validation_test_row_level_output_created"] = bool(validations["output_safety_validation"]["validation_test_row_level_output_created"])
    write_summary(out_paths["summary_md"], report, top_feature_summary, pattern_counts)
    save_json(out_paths["report_json"], report)

    logger.log("[STEP 18] Update latest bundle")
    bundle_report = write_latest_bundle(bundle_dir, project_root, out_paths, report)
    validations_after_bundle = validate_outputs(
        project_root, split_df, full_df, intervals, relative_df, baseline_df, ad_profile, nonad_profile,
        candidate_df, False, paths_for_validation, protected_before, bundle_dir,
    )
    report["latest_bundle"] = bundle_report
    report["sub_agent_validations"] = validations_after_bundle
    report["detector_rule_modified"] = bool(validations_after_bundle["output_safety_validation"]["detector_rule_modified"])
    report["old_project_modified"] = bool(validations_after_bundle["output_safety_validation"]["old_project_modified"])
    report["validation_test_row_level_output_created"] = bool(validations_after_bundle["output_safety_validation"]["validation_test_row_level_output_created"])
    report["elapsed_sec"] = time.time() - t0
    write_summary(out_paths["summary_md"], report, top_feature_summary, pattern_counts)
    save_json(out_paths["report_json"], report)
    write_latest_bundle(bundle_dir, project_root, out_paths, report)

    logger.log("[STEP 19] Print final human-readable summary")
    logger.log(
        f"[STEP 19] rows={len(relative_df)}, train_ads={len(intervals)}, "
        f"ad_profile_videos={profile_counts['ad_profile_video_count']}, "
        f"clean_nonad_profile_videos={profile_counts['clean_nonad_profile_video_count']}, "
        f"top_segments={len(top_segments)}, detector_rule_modified={report['detector_rule_modified']}, "
        f"old_project_modified={report['old_project_modified']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
