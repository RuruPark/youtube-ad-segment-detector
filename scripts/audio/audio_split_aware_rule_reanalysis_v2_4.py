#!/usr/bin/env python3
"""v2.4 video split 기준 leakage를 방지하며 train-only 오디오 규칙을 재분석한다."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
TASK_NAME = "audio_split_aware_rule_reanalysis"
VERSION = "v2_4"
SPLIT_SEED = 20240524
SPLIT_POLICY = "video_id_deterministic_shuffle"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_a"
DATA_AUDIO_DIR = PROJECT_ROOT / "data/audio"
REPORTS_AUDIO_DIR = PROJECT_ROOT / "reports/audio"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_AUDIO_DIR = PROJECT_ROOT / "scripts/audio"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SPLIT_FILE = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"
BACKUPS_DIR = PROJECT_ROOT / "backups"

FIXED_SPLIT = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}
INPUT_AUDIO_FILES = {
    "labeled": DATA_AUDIO_DIR / "audio_labeled_segment_features_v2_4.csv",
    "edge": DATA_AUDIO_DIR / "audio_ad_edge_5s_10s_features_v2_4.csv",
    "subwindow": DATA_AUDIO_DIR / "audio_ad_edge_persistence_subwindow_features_v2_4.csv",
    "context_scores": DATA_AUDIO_DIR / "audio_ad_edge_persistence_context_scores_v2_4.csv",
    "boundary_scores": DATA_AUDIO_DIR / "audio_ad_edge_boundary_persistence_scores_v2_4.csv",
}
EXPLORATORY_REFERENCE_FILES = {
    "feature_recommendations": DATA_AUDIO_DIR / "audio_rule_feature_recommendations_v2_4.csv",
    "candidate_thresholds": DATA_AUDIO_DIR / "audio_rule_candidate_thresholds_v2_4.csv",
    "persistence_rule_config": CONFIGS_DIR / "audio_persistence_rule_config_v2_4.json",
    "rule_design_md": PROJECT_ROOT / "reports/audio_persistence_rule_design_v2_4.md",
    "edge_persistence_summary": PROJECT_ROOT / "reports/extract_audio_ad_edge_persistence_v2_4_summary.md",
}
ANALYSIS_SCRIPT = SCRIPTS_AUDIO_DIR / "analyze_audio_rule_features_v2_4.py"
PERSISTENCE_SCRIPT = SCRIPTS_AUDIO_DIR / "extract_audio_ad_edge_persistence_v2_4.py"

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
METADATA_EXCLUDES = {
    "version",
    "source_label_file",
    "source_label_modified_time",
    "video_path",
    "video_title",
    "ad_interval_id",
    "video_id",
    "video_id_int",
    "segment_id",
    "subwindow_id",
    "segment_type",
    "boundary_type",
    "context_type",
    "parent_segment_type",
    "subwindow_index",
    "segment_start_sec",
    "segment_end_sec",
    "segment_duration_sec",
    "subwindow_start_sec",
    "subwindow_end_sec",
    "subwindow_duration_sec",
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
    "ad_start_sec",
    "ad_end_sec",
    "ad_duration_sec",
    "video_duration_sec",
    "context_truncated",
    "context_overlaps_other_ad",
    "relative_start_from_ad_start_sec",
    "relative_end_from_ad_start_sec",
    "relative_start_from_ad_end_sec",
    "relative_end_from_ad_end_sec",
    "relative_start_to_boundary_sec",
    "relative_end_to_boundary_sec",
    "split",
    "split_seed",
    "split_source",
    "split_policy",
    "leakage_guard",
}
HIGHER_FEATURES = [
    "onset_density",
    "onset_count_per_sec",
    "spectral_flux_mean",
    "onset_strength_mean",
    "spectral_flux_max",
    "onset_strength_max",
    "audio_log_energy_mean",
    "audio_rms_mean",
    "audio_mean_abs_amplitude",
]
LOWER_FEATURES = [
    "silence_ratio",
    "low_energy_ratio",
    "spectral_flatness_std",
    "spectral_bandwidth_mean",
    "spectral_centroid_std",
    "zero_crossing_rate_std",
]
SCORE_WEIGHTS = {
    "onset_density_score": 0.30,
    "flux_onset_score": 0.25,
    "energy_score": 0.20,
    "inverse_silence_score": 0.15,
    "spectral_texture_score": 0.10,
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


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_module(ANALYSIS_SCRIPT, "audio_rule_analysis_v2_4")
PERSISTENCE = load_module(PERSISTENCE_SCRIPT, "audio_edge_persistence_v2_4")


def paths() -> Dict[str, Path]:
    return {
        "labeled_with_split": DATA_AUDIO_DIR / "audio_labeled_segment_features_v2_4_with_split.csv",
        "edge_with_split": DATA_AUDIO_DIR / "audio_ad_edge_5s_10s_features_v2_4_with_split.csv",
        "subwindow_with_split": DATA_AUDIO_DIR / "audio_ad_edge_persistence_subwindow_features_v2_4_with_split.csv",
        "context_scores_with_split": DATA_AUDIO_DIR / "audio_ad_edge_persistence_context_scores_v2_4_with_split.csv",
        "boundary_scores_with_split": DATA_AUDIO_DIR / "audio_ad_edge_boundary_persistence_scores_v2_4_with_split.csv",
        "profile_train": DATA_AUDIO_DIR / "audio_rule_segment_type_profile_v2_4_train_only.csv",
        "pairwise_train": DATA_AUDIO_DIR / "audio_rule_pairwise_feature_comparison_v2_4_train_only.csv",
        "paired_delta_train": DATA_AUDIO_DIR / "audio_rule_paired_context_delta_summary_v2_4_train_only.csv",
        "edge_pairwise_train": DATA_AUDIO_DIR / "audio_ad_edge_5s_10s_pairwise_comparison_v2_4_train_only.csv",
        "edge_delta_train": DATA_AUDIO_DIR / "audio_ad_edge_5s_10s_delta_summary_v2_4_train_only.csv",
        "thresholds_train": DATA_AUDIO_DIR / "audio_rule_candidate_thresholds_v2_4_train_only.csv",
        "recommendations_train": DATA_AUDIO_DIR / "audio_rule_feature_recommendations_v2_4_train_only.csv",
        "validation_audit": DATA_AUDIO_DIR / "audio_rule_validation_application_audit_v2_4_train_only.csv",
        "validation_summary": DATA_AUDIO_DIR / "audio_rule_validation_summary_v2_4_train_only.csv",
        "config_train": CONFIGS_DIR / "audio_persistence_rule_config_v2_4_train_only.json",
        "summary_md": REPORTS_AUDIO_DIR / "audio_split_aware_rule_reanalysis_v2_4.md",
        "report_json": REPORTS_AUDIO_DIR / "audio_split_aware_rule_reanalysis_v2_4_report.json",
        "run_log": LOGS_DIR / "audio_split_aware_rule_reanalysis_v2_4_run_log.txt",
        "script": SCRIPTS_AUDIO_DIR / "audio_split_aware_rule_reanalysis_v2_4.py",
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def readable_seconds(seconds: float) -> str:
    minutes, sec = divmod(float(seconds), 60.0)
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
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def old_project_snapshot() -> Dict[str, Any]:
    if not OLD_PROJECT_ROOT.exists():
        return {"path": str(OLD_PROJECT_ROOT), "exists": False, "file_count": 0, "metadata_digest": None}
    entries: List[str] = []
    count = 0
    for dirpath, _, filenames in os.walk(OLD_PROJECT_ROOT):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            rel = path.relative_to(OLD_PROJECT_ROOT).as_posix()
            entries.append(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}")
            count += 1
    import hashlib

    digest = hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()
    return {
        "path": str(OLD_PROJECT_ROOT),
        "exists": True,
        "file_count": count,
        "metadata_digest": digest,
        "snapshot_type": "relative_path_size_mtime_ns",
    }


def backup_existing(out: Dict[str, Path], timestamp: str) -> Optional[Path]:
    targets = list(out.values()) + [LATEST_DIR / "README_latest_files.md"]
    existing = [path for path in targets if path.exists()]
    if not existing:
        return None
    backup_dir = BACKUPS_DIR / f"audio_split_aware_rule_reanalysis_v2_4_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        if path.is_file():
            shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_video_id(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def create_fixed_split_file(path: Path) -> None:
    rows = []
    for split, ids in FIXED_SPLIT.items():
        for vid in ids:
            rows.append(
                {
                    "version": VERSION,
                    "video_id": vid,
                    "split": split,
                    "split_method": SPLIT_POLICY,
                    "split_seed": SPLIT_SEED,
                    "note": "generated from fixed user-provided v2_4 split list",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def load_or_create_split() -> Tuple[pd.DataFrame, bool]:
    generated = False
    if not SPLIT_FILE.exists():
        create_fixed_split_file(SPLIT_FILE)
        generated = True
    split_df = read_csv(SPLIT_FILE)
    split_df["video_id_int"] = split_df["video_id"].map(normalize_video_id)
    return split_df, generated


def split_map(split_df: pd.DataFrame) -> Dict[int, str]:
    return {int(row["video_id_int"]): str(row["split"]) for _, row in split_df.dropna(subset=["video_id_int"]).iterrows()}


def attach_split(df: pd.DataFrame, split_lookup: Dict[int, str]) -> Tuple[pd.DataFrame, int, List[Any]]:
    out = df.copy()
    out["video_id_int"] = out["video_id"].map(normalize_video_id)
    out["split"] = out["video_id_int"].map(split_lookup).fillna("unknown")
    out["split_seed"] = SPLIT_SEED
    out["split_source"] = SPLIT_FILE.name
    out["split_policy"] = SPLIT_POLICY
    out["leakage_guard"] = True
    unknown = out[out["split"].eq("unknown")]["video_id"].dropna().unique().tolist()
    return out, int(out["split"].eq("unknown").sum()), unknown


def add_density(df: pd.DataFrame, duration_col: str = "segment_duration_sec") -> pd.DataFrame:
    out = df.copy()
    if "onset_density" not in out.columns and "onset_count" in out.columns and duration_col in out.columns:
        duration = pd.to_numeric(out[duration_col], errors="coerce").replace(0, np.nan)
        onset = pd.to_numeric(out["onset_count"], errors="coerce")
        out["onset_density"] = onset / duration
        out["onset_count_per_sec"] = out["onset_density"]
    if "onset_count_per_sec" not in out.columns and "onset_density" in out.columns:
        out["onset_count_per_sec"] = out["onset_density"]
    return out


def feature_group(feature: str) -> str:
    return ANALYSIS.feature_group(feature)


def select_train_features(df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, float], List[str]]:
    features: List[str] = []
    excluded: List[Dict[str, Any]] = []
    nan_ratio: Dict[str, float] = {}
    constant: List[str] = []
    for col in df.columns:
        if col in METADATA_EXCLUDES or col.endswith("__video_robust_z") or col.endswith("__directional_score"):
            excluded.append({"feature_name": col, "reason": "metadata_or_split_column"})
            continue
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() == 0:
            excluded.append({"feature_name": col, "reason": "non_numeric_or_all_nan"})
            continue
        ratio = float(values.isna().mean())
        nan_ratio[col] = ratio
        if ratio >= 0.5:
            excluded.append({"feature_name": col, "reason": f"nan_ratio_ge_0.5 ({ratio:.3f})"})
            continue
        finite = values[np.isfinite(values)]
        if finite.nunique(dropna=True) <= 1:
            constant.append(col)
            excluded.append({"feature_name": col, "reason": "constant_feature"})
            continue
        features.append(col)
    return features, excluded, nan_ratio, constant


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else np.nan
    return {
        "train_balanced_accuracy": safe_float(np.nanmean([recall, spec])),
        "train_precision": safe_float(precision),
        "train_recall": safe_float(recall),
        "train_f1": safe_float(f1),
    }


def train_thresholds(train_df: pd.DataFrame, features: Sequence[str], pairwise: pd.DataFrame) -> pd.DataFrame:
    subset = train_df[train_df["segment_type"].isin(["ad_full", "random_non_ad_30s"])].copy()
    y = (subset["segment_type"] == "ad_full").astype(int).to_numpy()
    effect = pairwise[pairwise["comparison_name"].eq("ad_full_vs_random_non_ad_30s")].set_index("feature_name")
    rows = []
    for feature in features:
        x_all = pd.to_numeric(subset[feature], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x_all)
        x = x_all[valid]
        yv = y[valid]
        pos = int((yv == 1).sum())
        neg = int((yv == 0).sum())
        base = {
            "version": VERSION,
            "split_basis": "train_only",
            "split_seed": SPLIT_SEED,
            "feature_name": feature,
            "feature_group": feature_group(feature),
            "direction": "unclear",
            "best_threshold": np.nan,
            "threshold_rule": "",
            "train_balanced_accuracy": np.nan,
            "train_precision": np.nan,
            "train_recall": np.nan,
            "train_f1": np.nan,
            "train_positive_count": pos,
            "train_negative_count": neg,
            "selected_by_effect_size": False,
            "selected_by_threshold_quality": False,
            "caveat": "train-only threshold candidate; validation/test not used for selection",
            "leakage_guard_note": "validation audit only; test not used for rule design",
        }
        if pos < 2 or neg < 2 or len(np.unique(x)) < 2:
            base["caveat"] += "; insufficient train variation"
            rows.append(base)
            continue
        candidates = np.unique(np.nanquantile(x, np.linspace(0.05, 0.95, 19)))
        best: Optional[Dict[str, Any]] = None
        for thr in candidates:
            for direction, pred in [
                ("higher_indicates_ad", (x >= thr).astype(int)),
                ("lower_indicates_ad", (x <= thr).astype(int)),
            ]:
                row = {**base, **metrics(yv, pred)}
                row["direction"] = direction
                row["best_threshold"] = safe_float(thr)
                row["threshold_rule"] = f"{feature} {'>=' if direction == 'higher_indicates_ad' else '<='} {thr:.6g}"
                if best is None or (row["train_balanced_accuracy"], row["train_f1"] if np.isfinite(row["train_f1"]) else -1) > (
                    best["train_balanced_accuracy"],
                    best["train_f1"] if np.isfinite(best["train_f1"]) else -1,
                ):
                    best = row
        assert best is not None
        d = safe_float(effect.loc[feature, "cohen_d"]) if feature in effect.index else np.nan
        best["selected_by_effect_size"] = bool(np.isfinite(d) and abs(d) >= 0.5)
        best["selected_by_threshold_quality"] = bool(best["train_balanced_accuracy"] >= 0.65)
        rows.append(best)
    return pd.DataFrame(rows)


def effect_level(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    mag = abs(value)
    if mag >= 0.8:
        return "large"
    if mag >= 0.5:
        return "medium"
    if mag >= 0.2:
        return "small"
    return "weak"


def recommendation_text(feature: str, usage: str) -> str:
    group = feature_group(feature)
    if usage == "exclude":
        return "결측/상수/해석 불안정으로 train-only rule 후보에서 제외한다."
    if group == "Onset":
        return f"{feature}는 train split에서 광고 구간의 onset persistence 또는 변화 밀도를 보는 audio cue 후보이다."
    if group == "Spectral Flux":
        return f"{feature}는 train split에서 BGM/효과음/편집 변화 지속성을 보는 보조 cue 후보이다."
    if group == "Energy / Amplitude":
        return f"{feature}는 train split에서 음량/에너지 차이를 보는 rule-based detector support feature 후보이다."
    if group == "Silence / Low Energy":
        return f"{feature}는 낮을수록 광고스러운 끊김 없는 audio pattern을 볼 수 있는 inverse cue 후보이다."
    return f"{feature}는 train-only feature recommendation의 보조 audio cue 후보이다."


def build_recommendations(features: Sequence[str], pairwise: pd.DataFrame, delta_summary: pd.DataFrame, thresholds: pd.DataFrame, excluded: List[Dict[str, Any]]) -> pd.DataFrame:
    pair_index = pairwise.set_index(["comparison_name", "feature_name"])
    thr_index = thresholds.set_index("feature_name")
    rows = []
    for feature in features:
        usage = "low_priority"
        evidence = "ad_vs_random"
        direction = "unclear"
        effect = np.nan
        if ("ad_full_vs_random_non_ad_30s", feature) in pair_index.index:
            row = pair_index.loc[("ad_full_vs_random_non_ad_30s", feature)]
            effect = safe_float(row["cohen_d"])
            if np.isfinite(effect) and abs(effect) >= 0.5:
                usage = "use_for_ad_likelihood"
                direction = "higher_indicates_ad" if effect > 0 else "lower_indicates_ad"
        for comp, candidate_usage, ev in [
            ("ad_full_vs_pre_ad_10s", "use_for_start_context_change", "ad_vs_pre"),
            ("ad_full_vs_post_ad_10s", "use_for_end_context_change", "ad_vs_post"),
        ]:
            if usage == "low_priority" and (comp, feature) in pair_index.index:
                row = pair_index.loc[(comp, feature)]
                d = safe_float(row["cohen_d"])
                if np.isfinite(d) and abs(d) >= 0.5:
                    usage = candidate_usage
                    evidence = ev
                    direction = "higher_transition_change" if d > 0 else "lower_transition_change"
                    effect = d
        if usage == "low_priority":
            deltas = delta_summary[delta_summary["feature_name"].eq(feature)]
            if not deltas.empty:
                best = deltas.iloc[deltas["paired_effect_size"].abs().fillna(0).argmax()]
                d = safe_float(best["paired_effect_size"])
                if np.isfinite(d) and abs(d) >= 0.5:
                    usage = "use_for_general_audio_change"
                    evidence = "paired_delta"
                    direction = "higher_transition_change" if d > 0 else "lower_transition_change"
                    effect = d
        thr = thr_index.loc[feature] if feature in thr_index.index else None
        if usage == "low_priority" and thr is not None and bool(thr["selected_by_threshold_quality"]):
            usage = "use_for_ad_likelihood"
            direction = str(thr["direction"])
        threshold_rule = str(thr["threshold_rule"]) if thr is not None else ""
        candidate_threshold = safe_float(thr["best_threshold"]) if thr is not None else np.nan
        confidence = "high" if usage.startswith("use_for") and np.isfinite(effect) and abs(effect) >= 0.8 else "medium" if usage.startswith("use_for") else "low"
        rows.append(
            {
                "version": VERSION,
                "split_basis": "train_only",
                "split_seed": SPLIT_SEED,
                "feature_name": feature,
                "feature_group": feature_group(feature),
                "recommended_usage": usage,
                "primary_evidence": evidence,
                "direction": direction,
                "train_effect_size": safe_float(effect),
                "effect_size_level": effect_level(effect),
                "candidate_threshold": candidate_threshold,
                "threshold_rule": threshold_rule,
                "confidence_level": confidence,
                "validation_status": "not_checked",
                "caution": "train-only recommendation; validation audit only; test not used",
                "plain_korean_interpretation": recommendation_text(feature, usage),
                "leakage_guard_note": "computed from train split only",
            }
        )
    for item in excluded:
        if item["reason"] == "metadata_or_split_column":
            continue
        rows.append(
            {
                "version": VERSION,
                "split_basis": "train_only",
                "split_seed": SPLIT_SEED,
                "feature_name": item["feature_name"],
                "feature_group": feature_group(item["feature_name"]),
                "recommended_usage": "exclude",
                "primary_evidence": "missing_or_unstable",
                "direction": "unclear",
                "train_effect_size": np.nan,
                "effect_size_level": "unknown",
                "candidate_threshold": np.nan,
                "threshold_rule": "",
                "confidence_level": "low",
                "validation_status": "not_checked",
                "caution": item["reason"],
                "plain_korean_interpretation": "train split에서 결측이 많거나 안정적이지 않아 제외한다.",
                "leakage_guard_note": "computed from train split only",
            }
        )
    return pd.DataFrame(rows)


def validation_audit(validation_df: pd.DataFrame, thresholds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset = validation_df[validation_df["segment_type"].isin(["ad_full", "random_non_ad_30s"])].copy()
    audit_rows = []
    for _, thr in thresholds.iterrows():
        feature = thr["feature_name"]
        if feature not in subset.columns or not np.isfinite(safe_float(thr["best_threshold"])):
            continue
        direction = str(thr["direction"])
        threshold = safe_float(thr["best_threshold"])
        for _, row in subset.iterrows():
            value = safe_float(row.get(feature))
            if not np.isfinite(value):
                rule_pass = False
                note = "missing_validation_value"
            elif direction == "higher_indicates_ad":
                rule_pass = value >= threshold
                note = ""
            elif direction == "lower_indicates_ad":
                rule_pass = value <= threshold
                note = ""
            else:
                rule_pass = False
                note = "unclear_train_direction"
            expected_positive = row["segment_type"] == "ad_full"
            consistent = bool(rule_pass == expected_positive)
            audit_rows.append(
                {
                    "version": VERSION,
                    "split_basis": "train_only_rule_applied_to_validation",
                    "video_id": int(row["video_id_int"]),
                    "ad_interval_id": row.get("ad_interval_id", ""),
                    "segment_type": row.get("segment_type", ""),
                    "boundary_type": "",
                    "feature_name": feature,
                    "train_threshold": threshold,
                    "validation_value": value,
                    "rule_pass": bool(rule_pass),
                    "expected_direction_from_train": direction,
                    "validation_direction_consistent": consistent,
                    "validation_note": note or "validation_application_audit_only",
                }
            )
    audit = pd.DataFrame(audit_rows)
    summary_rows = []
    if not audit.empty:
        for feature, group in audit.groupby("feature_name"):
            pass_ratio = group["rule_pass"].mean()
            consistency = group["validation_direction_consistent"].mean()
            recommendation = "keep" if consistency >= 0.65 else "review_threshold" if consistency >= 0.5 else "unstable"
            train_dir = group["expected_direction_from_train"].iloc[0]
            summary_rows.append(
                {
                    "feature_name": feature,
                    "train_direction": train_dir,
                    "validation_sample_count": int(len(group)),
                    "validation_pass_count": int(group["rule_pass"].sum()),
                    "validation_pass_ratio": safe_float(pass_ratio),
                    "validation_direction_consistency": safe_float(consistency),
                    "validation_review_recommendation": recommendation,
                    "caution": "validation audit only; threshold not adjusted automatically",
                }
            )
    return audit, pd.DataFrame(summary_rows)


def train_config(thresholds: pd.DataFrame, context_scores: pd.DataFrame, recommendations: pd.DataFrame, split_df: pd.DataFrame) -> Dict[str, Any]:
    selected = recommendations[recommendations["recommended_usage"].ne("exclude")].sort_values(
        ["confidence_level", "train_effect_size"], ascending=[True, False]
    )
    threshold_map = {
        row["feature_name"]: {
            "direction": row["direction"],
            "threshold": safe_float(row["best_threshold"]),
            "threshold_rule": row["threshold_rule"],
        }
        for _, row in thresholds.iterrows()
        if bool(row.get("selected_by_threshold_quality", False)) or bool(row.get("selected_by_effect_size", False))
    }
    train_context = context_scores[context_scores["split"].eq("train")].copy()
    positive_context = train_context[train_context["context_type"].isin(["start_post_10s", "end_pre_10s"])]
    random_context = train_context[train_context["context_type"].eq("random_non_ad_30s")]
    if not positive_context.empty and not random_context.empty:
        score_thr = safe_float(np.nanmean([positive_context["audio_ad_like_score_median"].quantile(0.25), random_context["audio_ad_like_score_median"].quantile(0.75)]))
        ratio_thr = safe_float(max(0.5, min(0.8, positive_context["ad_like_ratio"].quantile(0.25))))
        consecutive_thr = safe_float(max(4.0, min(8.0, positive_context["max_consecutive_ad_like_sec"].quantile(0.25))))
    else:
        score_thr = 0.55
        ratio_thr = 0.6
        consecutive_thr = 6.0
    split_lists = {split: sorted(split_df[split_df["split"].eq(split)]["video_id_int"].dropna().astype(int).tolist()) for split in ["train", "validation", "test"]}
    return {
        "version": VERSION,
        "split_basis": "train_only",
        "split_seed": SPLIT_SEED,
        "split_file": str(SPLIT_FILE),
        "train_video_ids": split_lists["train"],
        "validation_video_ids": split_lists["validation"],
        "test_video_ids": split_lists["test"],
        "input_feature_files": {k: str(v) for k, v in INPUT_AUDIO_FILES.items()},
        "exploratory_reference_files": {k: str(v) for k, v in EXPLORATORY_REFERENCE_FILES.items()},
        "selected_features": selected["feature_name"].head(20).tolist(),
        "feature_directions": {
            **{feature: "higher_indicates_ad" for feature in HIGHER_FEATURES},
            **{feature: "lower_indicates_ad" for feature in LOWER_FEATURES},
        },
        "feature_weights": SCORE_WEIGHTS,
        "train_only_thresholds": threshold_map,
        "audio_ad_like_score_formula": "0.30*onset_density_score + 0.25*flux_onset_score + 0.20*energy_score + 0.15*inverse_silence_score + 0.10*spectral_texture_score",
        "persistence_window_sec": 10,
        "subwindow_size_sec": 2,
        "subwindow_stride_sec": 2,
        "start_boundary_rule": {
            "post_10s_audio_ad_like_ratio_min": ratio_thr,
            "post_10s_max_consecutive_ad_like_sec_min": consecutive_thr,
            "post_10s_median_score_min": score_thr,
            "post_10s_median_score_minus_pre_10s_median_score_min": 0.05,
        },
        "end_boundary_rule": {
            "pre_10s_audio_ad_like_ratio_min": ratio_thr,
            "pre_10s_max_consecutive_ad_like_sec_min": consecutive_thr,
            "pre_10s_median_score_min": score_thr,
            "pre_10s_median_score_minus_post_10s_median_score_min": 0.05,
        },
        "validation_usage_policy": "validation_application_audit_only_no_automatic_threshold_adjustment",
        "test_usage_policy": "test_not_used_for_rule_design_preserved_for_final_evaluation",
        "leakage_guard_notes": [
            "thresholds, recommendations, and config are computed from train split only",
            "validation is used only for application audit",
            "test rows are not used for threshold, recommendation, or config",
        ],
        "caveats": [
            "train video count is small",
            "thresholds are exploratory candidates",
            "full-data v2_4 outputs are preserved only as exploratory_full_v2_4 references",
        ],
    }


def forbidden_latest_files() -> List[str]:
    if not LATEST_DIR.exists():
        return []
    forbidden = []
    for path in LATEST_DIR.rglob("*"):
        if path.is_dir() and path.name in {"cache", "tmp"}:
            forbidden.append(str(path))
        elif path.is_file() and path.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES:
            forbidden.append(str(path))
    return forbidden


def update_latest(out: Dict[str, Path], report: Dict[str, Any], backup_dir: Optional[Path]) -> List[str]:
    include_keys = [
        "labeled_with_split",
        "edge_with_split",
        "context_scores_with_split",
        "boundary_scores_with_split",
        "thresholds_train",
        "recommendations_train",
        "validation_summary",
        "config_train",
        "summary_md",
        "report_json",
        "run_log",
        "script",
    ]
    allowed = {out[k].name for k in include_keys} | {"README_latest_files.md"}
    archived = []
    cleanup_dir = (backup_dir or BACKUPS_DIR / f"{TASK_NAME}_latest_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}") / "latest_for_chatgpt_a_previous"
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for item in list(LATEST_DIR.iterdir()):
        if item.name in allowed:
            continue
        cleanup_dir.mkdir(parents=True, exist_ok=True)
        dest = cleanup_dir / item.name
        if dest.exists():
            dest = cleanup_dir / f"{dest.stem}_{datetime.now().strftime('%H%M%S')}{dest.suffix}"
        shutil.move(str(item), str(dest))
        archived.append(str(dest))
    if archived:
        report["latest_for_chatgpt_a_archived_previous_files"] = archived
    copied = []
    skipped = []
    for key in include_keys:
        path = out[key]
        if not path.exists():
            continue
        if path.stat().st_size > 5_000_000:
            skipped.append(str(path))
            continue
        shutil.copy2(path, LATEST_DIR / path.name)
        copied.append(str(LATEST_DIR / path.name))
    report["latest_for_chatgpt_a_skipped_large_files"] = skipped
    readme = LATEST_DIR / "README_latest_files.md"
    lines = [
        "# Latest Files",
        "",
        f"- task_name: {TASK_NAME}",
        f"- version: {VERSION}",
        f"- split_seed: {SPLIT_SEED}",
        f"- split_file: {SPLIT_FILE}",
        "- output bundle path: outputs/latest_for_chatgpt_a",
        "- validation: audit only",
        "- test: not used for rule design",
        f"- old_project_modified: {str(report.get('old_project_modified')).lower()}",
        "",
        "This bundle contains leakage-aware train-only audio rule reanalysis outputs.",
        "Existing full-data v2_4 audio rule outputs are exploratory_full_v2_4 references only.",
        "",
        "## Included Files",
    ]
    for item in sorted(copied):
        lines.append(f"- {Path(item).name}")
    if skipped:
        lines.append("")
        lines.append("## Large Files Not Copied")
        for item in skipped:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("No media, frame image, model, cache, tmp, or raw video files are included.")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copied.append(str(readme))
    return sorted(copied)


def write_table(rows: Sequence[Dict[str, Any]], cols: Sequence[str]) -> str:
    if not rows:
        return "_No rows_"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append("" if not np.isfinite(value) else f"{value:.4g}")
            else:
                vals.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(path: Path, report: Dict[str, Any], thresholds: pd.DataFrame, recs: pd.DataFrame, validation_summary: pd.DataFrame) -> str:
    top_recs = recs[recs["recommended_usage"].ne("exclude")].copy()
    top_recs["_abs"] = pd.to_numeric(top_recs["train_effect_size"], errors="coerce").abs()
    top_recs = top_recs.sort_values(["confidence_level", "_abs"], ascending=[True, False]).head(12).to_dict("records")
    top_thr = thresholds.sort_values("train_balanced_accuracy", ascending=False).head(12).to_dict("records")
    val_keep = validation_summary[validation_summary["validation_review_recommendation"].eq("keep")].head(10).to_dict("records") if not validation_summary.empty else []
    val_review = validation_summary[validation_summary["validation_review_recommendation"].ne("keep")].head(10).to_dict("records") if not validation_summary.empty else []
    sub_lines = [
        f"- {name}: {result.get('status')} (warnings={len(result.get('warnings', []))}, errors={len(result.get('errors', []))})"
        for name, result in report.get("sub_agent_results", {}).items()
    ]
    outputs = [f"- {p}" for p in report.get("output_files", [])]
    text = "\n".join(
        [
            "# Audio Split-aware Rule Reanalysis v2_4",
            "",
            "## 1. 작업 개요",
            "OCR 작업에서 고정된 v2_4 video_id split을 audio rule 분석에도 반영했다.",
            "기존 audio feature CSV는 raw/precomputed feature로 유지했고, 기존 full-data threshold/recommendation/config는 exploratory_full_v2_4 reference로 보존했다.",
            "이번 산출물은 split_aware_train_only_v2_4 기준이며, train split만 사용해 threshold/recommendation/config를 재산출했다.",
            "",
            "## 2. 사용한 Split",
            f"- split_seed: {SPLIT_SEED}",
            f"- train video_id: {report['train_video_ids']}",
            f"- validation video_id: {report['validation_video_ids']}",
            f"- test video_id: {report['test_video_ids']}",
            f"- split file: {SPLIT_FILE}",
            "",
            "## 3. 입력 파일",
            *[f"- audio feature {k}: {v}" for k, v in INPUT_AUDIO_FILES.items()],
            *[f"- exploratory reference {k}: {v}" for k, v in EXPLORATORY_REFERENCE_FILES.items()],
            "",
            "## 4. Split Join 결과",
            f"- split row count by file: {report['split_row_count_by_file']}",
            f"- unknown split row count by file: {report['split_join_unknown_row_count_by_file']}",
            "",
            "## 5. Train-only 분석 결과",
            f"- train numeric feature count: {report['train_numeric_feature_count']}",
            f"- train threshold count: {report['train_threshold_count']}",
            f"- train recommendation count: {report['train_recommendation_count']}",
            "### Train-only threshold candidates",
            write_table(top_thr, ["feature_name", "direction", "threshold_rule", "train_balanced_accuracy", "train_f1"]),
            "### Train-only feature recommendations",
            write_table(top_recs, ["feature_name", "recommended_usage", "direction", "train_effect_size", "confidence_level", "threshold_rule"]),
            "",
            "## 6. Train-only Audio Rule Config",
            "기존 audio_ad_like_score 공식은 유지했지만 threshold/config 값은 train split row만 사용해 산출했다.",
            "feature weight는 validation/test를 보지 않고 기존 exploratory 설계를 보수적으로 유지했다.",
            f"- config: {report['train_only_config_file']}",
            "",
            "## 7. Validation Application Audit",
            "validation은 train-only threshold를 적용해 보는 audit only 용도이며, threshold 자동 조정은 수행하지 않았다.",
            "### consistent_on_validation 후보",
            write_table(val_keep, ["feature_name", "train_direction", "validation_sample_count", "validation_pass_ratio", "validation_direction_consistency", "validation_review_recommendation"]),
            "### review 필요 후보",
            write_table(val_review, ["feature_name", "train_direction", "validation_sample_count", "validation_pass_ratio", "validation_direction_consistency", "validation_review_recommendation"]),
            "",
            "## 8. Test Split 보호",
            "- test_used_for_threshold_design=false",
            "- test_used_for_feature_recommendation=false",
            "- test_used_for_rule_config=false",
            "- test_split_preserved_for_final_evaluation=true",
            "test split은 rule 설계, threshold 조정, validation audit에 사용하지 않았다.",
            "",
            "## 9. 생성 파일 목록",
            *outputs,
            "",
            "## 10. 한계",
            "- train video 수가 12개라 sample size가 작다.",
            "- validation video 수가 3개라 조정 판단도 조심해야 한다.",
            "- test는 아직 보지 않았으므로 최종 일반화 성능은 알 수 없다.",
            "- threshold는 train-only exploratory candidate다.",
            "",
            "## 11. 다음 작업 제안",
            "- OCR feature/threshold/rule도 같은 split 기준으로 산출한다.",
            "- scene + audio + OCR fusion rule은 train split 기준으로 설계한다.",
            "- validation은 조정 후보 검토용으로만 사용한다.",
            "- test는 최종 평가에서만 사용한다.",
            "",
            "## 12. Sub Agent 검증 결과",
            *sub_lines,
            "",
            f"- old_project_modified: {report.get('old_project_modified')}",
            f"- input_audio_feature_files_modified: {report.get('input_audio_feature_files_modified')}",
            f"- exploratory_reference_files_modified: {report.get('exploratory_reference_files_modified')}",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")
    return text


def validate(report: Dict[str, Any], out: Dict[str, Path], summary_text: str) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    errors: List[str] = []
    if report["train_video_ids"] != FIXED_SPLIT["train"] or report["validation_video_ids"] != FIXED_SPLIT["validation"] or report["test_video_ids"] != FIXED_SPLIT["test"]:
        errors.append("split video_id lists do not match fixed v2_4 split")
    if report["unknown_split_video_ids"]:
        warnings.append("some video_id values mapped to unknown split")
    results["split_integrity_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = []
    errors = []
    if report["validation_used_for_adjustment"]:
        errors.append("validation marked as used for adjustment")
    if report["test_used_for_threshold_design"] or report["test_used_for_feature_recommendation"] or report["test_used_for_rule_config"]:
        errors.append("test split used in rule design")
    if "exploratory_full_v2_4" not in summary_text:
        errors.append("exploratory full-data distinction missing")
    if report["validation_audit_row_count"] and report["validation_unstable_feature_count"]:
        warnings.append("validation audit found unstable or review-needed features")
    results["leakage_guard_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = ["train sample size is small; thresholds are exploratory"]
    errors = []
    if report["train_threshold_count"] == 0 or report["train_recommendation_count"] == 0:
        errors.append("train-only threshold/recommendation output is empty")
    if "onset_density" not in report["train_analyzed_features"]:
        errors.append("onset_density missing from train features")
    if report["constant_features_train"]:
        warnings.append("some train features are constant and excluded")
    if any(v for v in report.get("inf_count_by_feature_train", {}).values()):
        warnings.append("inf values were detected and converted to NaN")
    results["train_only_audio_analysis_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = []
    errors = []
    if not out["config_train"].exists():
        errors.append("train-only config missing")
    if report["validation_used_for_adjustment"]:
        errors.append("validation adjusted threshold/config")
    if report["test_audit_generated"]:
        errors.append("test audit generated")
    if report["validation_unstable_feature_count"]:
        warnings.append("some validation features need review; no automatic adjustment applied")
    results["validation_audit_config_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = []
    errors = []
    missing = [str(path) for path in out.values() if not path.exists()]
    if missing:
        errors.append(f"missing outputs: {missing}")
    forbidden = forbidden_latest_files()
    if forbidden:
        errors.append(f"forbidden files in latest_for_chatgpt_a: {forbidden}")
    if report["input_audio_feature_files_modified"] or report["exploratory_reference_files_modified"] or report["old_project_modified"]:
        errors.append("safety modification flag is true")
    if report["backup_dir"]:
        warnings.append("existing outputs or latest README were backed up before overwrite")
    results["output_safety_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}
    return results


def forbidden_latest_files() -> List[str]:
    if not LATEST_DIR.exists():
        return []
    forbidden = []
    for path in LATEST_DIR.rglob("*"):
        if path.is_dir() and path.name in {"cache", "tmp"}:
            forbidden.append(str(path))
        elif path.is_file() and path.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES:
            forbidden.append(str(path))
    return forbidden


def main() -> int:
    start_epoch = time.time()
    start_time = now_iso()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = paths()
    for directory in [DATA_AUDIO_DIR, REPORTS_AUDIO_DIR, LOGS_DIR, SCRIPTS_AUDIO_DIR, CONFIGS_DIR, LATEST_DIR, SPLIT_FILE.parent, BACKUPS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_existing(out, timestamp)
    logger = TaskLogger(out["run_log"])
    warnings: List[str] = []
    errors: List[str] = []
    if backup_dir:
        warnings.append(f"existing outputs backed up to {backup_dir}")

    logger.log("[STEP 01] Start task and create old project snapshot")
    logger.log(f"Task start time: {start_time}")
    old_before = old_project_snapshot()
    input_mtimes = {str(path): path.stat().st_mtime_ns for path in list(INPUT_AUDIO_FILES.values()) + list(EXPLORATORY_REFERENCE_FILES.values()) if path.exists()}

    logger.log("[STEP 02] Load or create v2_4 video split file")
    split_df, generated_split = load_or_create_split()
    split_lookup = split_map(split_df)
    logger.log(f"Split file: {SPLIT_FILE}")
    for split in ["train", "validation", "test"]:
        logger.log(f"{split} video_ids: {sorted(split_df[split_df['split'].eq(split)]['video_id_int'].dropna().astype(int).tolist())}")

    logger.log("[STEP 03] Validate split integrity and video_id coverage")
    audio_dfs = {name: read_csv(path) for name, path in INPUT_AUDIO_FILES.items()}
    split_counts_by_file: Dict[str, Dict[str, int]] = {}
    unknown_counts: Dict[str, int] = {}
    unknown_ids: List[Any] = []

    logger.log("[STEP 04] Attach split column to audio feature files")
    with_split: Dict[str, pd.DataFrame] = {}
    with_split_path_map = {
        "labeled": out["labeled_with_split"],
        "edge": out["edge_with_split"],
        "subwindow": out["subwindow_with_split"],
        "context_scores": out["context_scores_with_split"],
        "boundary_scores": out["boundary_scores_with_split"],
    }
    for name, df in audio_dfs.items():
        joined, unknown_count, ids = attach_split(df, split_lookup)
        with_split[name] = joined
        joined.to_csv(with_split_path_map[name], index=False, encoding="utf-8-sig")
        split_counts_by_file[name] = {str(k): int(v) for k, v in joined["split"].value_counts(dropna=False).to_dict().items()}
        unknown_counts[name] = unknown_count
        unknown_ids.extend(ids)
        logger.log(f"{name}: rows={len(joined)} split_counts={split_counts_by_file[name]} unknown_rows={unknown_count}")

    logger.log("[STEP 05] Mark existing full-data rule outputs as exploratory references")
    exploratory_modified_before = {str(path): path.stat().st_mtime_ns for path in EXPLORATORY_REFERENCE_FILES.values() if path.exists()}

    logger.log("[STEP 06] Recompute train-only segment feature profiles")
    labeled = add_density(with_split["labeled"], "segment_duration_sec")
    labeled.to_csv(out["labeled_with_split"], index=False, encoding="utf-8-sig")
    labeled_train = labeled[labeled["split"].eq("train")].copy()
    train_features, excluded, nan_ratio, constant_features = select_train_features(labeled_train)
    for feature in train_features:
        labeled_train[feature] = pd.to_numeric(labeled_train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    profile = ANALYSIS.profile_by_segment(labeled_train, train_features, VERSION)
    profile["split_basis"] = "train_only"
    profile["split_seed"] = SPLIT_SEED
    profile.to_csv(out["profile_train"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 07] Recompute train-only pairwise comparisons and effect sizes")
    pairwise = ANALYSIS.pairwise_comparison(labeled_train, train_features)
    pairwise.insert(0, "split_seed", SPLIT_SEED)
    pairwise.insert(0, "split_basis", "train_only")
    pairwise.insert(0, "version", VERSION)
    pairwise.to_csv(out["pairwise_train"], index=False, encoding="utf-8-sig")
    _, delta_summary = ANALYSIS.paired_delta(labeled_train, train_features)
    delta_summary.insert(0, "split_seed", SPLIT_SEED)
    delta_summary.insert(0, "split_basis", "train_only")
    delta_summary.insert(0, "version", VERSION)
    delta_summary.to_csv(out["paired_delta_train"], index=False, encoding="utf-8-sig")

    edge = add_density(with_split["edge"], "segment_duration_sec")
    edge.to_csv(out["edge_with_split"], index=False, encoding="utf-8-sig")
    edge_train = edge[edge["split"].eq("train")].copy()
    edge_pairwise, edge_delta = PERSISTENCE.edge_pairwise(edge_train, labeled_train)
    edge_pairwise.insert(0, "split_seed", SPLIT_SEED)
    edge_pairwise.insert(0, "split_basis", "train_only")
    edge_pairwise.insert(0, "version", VERSION)
    edge_delta.insert(0, "split_seed", SPLIT_SEED)
    edge_delta.insert(0, "split_basis", "train_only")
    edge_delta.insert(0, "version", VERSION)
    edge_pairwise.to_csv(out["edge_pairwise_train"], index=False, encoding="utf-8-sig")
    edge_delta.to_csv(out["edge_delta_train"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 08] Recompute train-only candidate thresholds")
    thresholds = train_thresholds(labeled_train, train_features, pairwise)
    thresholds.to_csv(out["thresholds_train"], index=False, encoding="utf-8-sig")
    logger.log(f"Train-only threshold row count: {len(thresholds)}")

    logger.log("[STEP 09] Recompute train-only feature recommendations")
    recommendations = build_recommendations(train_features, pairwise, delta_summary, thresholds, excluded)
    audit, val_summary = validation_audit(labeled[labeled["split"].eq("validation")].copy(), thresholds)
    validation_status_map = {
        row["feature_name"]: "consistent_on_validation" if row["validation_review_recommendation"] == "keep" else "validation_review_needed"
        for _, row in val_summary.iterrows()
    }
    recommendations["validation_status"] = recommendations["feature_name"].map(validation_status_map).fillna("not_checked")
    recommendations.to_csv(out["recommendations_train"], index=False, encoding="utf-8-sig")
    logger.log(f"Train-only recommendation row count: {len(recommendations)}")

    logger.log("[STEP 10] Rebuild train-only audio persistence rule config")
    context_scores = with_split["context_scores"].copy()
    config = train_config(thresholds, context_scores, recommendations, split_df)
    save_json(out["config_train"], config)

    logger.log("[STEP 11] Apply train-only rules to validation split for audit only")
    audit.to_csv(out["validation_audit"], index=False, encoding="utf-8-sig")
    val_summary.to_csv(out["validation_summary"], index=False, encoding="utf-8-sig")
    logger.log(f"Validation application audit row count: {len(audit)}")

    logger.log("[STEP 12] Do not use test split for rule design")
    test_audit_generated = False
    logger.log("test_used_for_threshold_design=false; test_used_for_feature_recommendation=false; test_used_for_rule_config=false")

    old_after = old_project_snapshot()
    input_modified = any(path.exists() and path.stat().st_mtime_ns != input_mtimes.get(str(path), path.stat().st_mtime_ns) for path in INPUT_AUDIO_FILES.values())
    exploratory_modified = any(path.exists() and path.stat().st_mtime_ns != exploratory_modified_before.get(str(path), path.stat().st_mtime_ns) for path in EXPLORATORY_REFERENCE_FILES.values())
    report: Dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "task_name": TASK_NAME,
        "version": VERSION,
        "start_time": start_time,
        "end_time": None,
        "actual_runtime_seconds": None,
        "actual_runtime_readable": None,
        "input_files": {**{k: str(v) for k, v in INPUT_AUDIO_FILES.items()}, **{"split_file": str(SPLIT_FILE)}},
        "output_files": [str(p) for p in out.values()],
        "generated_files": [str(p) for p in out.values()],
        "latest_for_chatgpt_files": [],
        "missing_input_files": [str(path) for path in list(INPUT_AUDIO_FILES.values()) + list(EXPLORATORY_REFERENCE_FILES.values()) if not path.exists()],
        "missing_required_columns": [],
        "warnings": warnings,
        "errors": errors,
        "old_project_modified": old_before != old_after,
        "old_project_snapshot_before": old_before,
        "old_project_snapshot_after": old_after,
        "sub_agent_results": {},
        "split_file": str(SPLIT_FILE),
        "generated_split_file": generated_split,
        "split_seed": SPLIT_SEED,
        "split_policy": SPLIT_POLICY,
        "train_video_ids": sorted(split_df[split_df["split"].eq("train")]["video_id_int"].dropna().astype(int).tolist()),
        "validation_video_ids": sorted(split_df[split_df["split"].eq("validation")]["video_id_int"].dropna().astype(int).tolist()),
        "test_video_ids": sorted(split_df[split_df["split"].eq("test")]["video_id_int"].dropna().astype(int).tolist()),
        "train_video_count": int((split_df["split"] == "train").sum()),
        "validation_video_count": int((split_df["split"] == "validation").sum()),
        "test_video_count": int((split_df["split"] == "test").sum()),
        "unknown_split_video_ids": sorted(set(map(str, unknown_ids))),
        "split_join_unknown_row_count_by_file": unknown_counts,
        "split_row_count_by_file": split_counts_by_file,
        "raw_feature_files_kept": [str(v) for v in INPUT_AUDIO_FILES.values()],
        "exploratory_full_data_files_preserved": {k: str(v) for k, v in EXPLORATORY_REFERENCE_FILES.items()},
        "train_only_files_generated": [str(out[k]) for k in ["profile_train", "pairwise_train", "paired_delta_train", "edge_pairwise_train", "edge_delta_train", "thresholds_train", "recommendations_train", "config_train"]],
        "validation_used_for_adjustment": False,
        "validation_used_for_audit_only": True,
        "test_used_for_threshold_design": False,
        "test_used_for_feature_recommendation": False,
        "test_used_for_rule_config": False,
        "test_split_preserved_for_final_evaluation": True,
        "test_audit_generated": test_audit_generated,
        "train_row_count_by_file": {k: int(v["split"].eq("train").sum()) for k, v in with_split.items()},
        "validation_row_count_by_file": {k: int(v["split"].eq("validation").sum()) for k, v in with_split.items()},
        "test_row_count_by_file": {k: int(v["split"].eq("test").sum()) for k, v in with_split.items()},
        "train_numeric_feature_count": int(len(train_features)),
        "train_analyzed_features": train_features,
        "train_threshold_count": int(len(thresholds)),
        "train_recommendation_count": int(len(recommendations)),
        "validation_audit_row_count": int(len(audit)),
        "validation_unstable_feature_count": int((val_summary["validation_review_recommendation"].ne("keep")).sum()) if not val_summary.empty else 0,
        "excluded_features": excluded,
        "nan_ratio_by_feature_train": nan_ratio,
        "inf_count_by_feature_train": {feature: int(np.isinf(pd.to_numeric(labeled_train[feature], errors="coerce")).sum()) for feature in train_features},
        "constant_features_train": constant_features,
        "with_split_output_files": [str(with_split_path_map[k]) for k in with_split_path_map],
        "train_only_threshold_file": str(out["thresholds_train"]),
        "train_only_recommendation_file": str(out["recommendations_train"]),
        "train_only_config_file": str(out["config_train"]),
        "validation_audit_file": str(out["validation_audit"]),
        "summary_md": str(out["summary_md"]),
        "run_log": str(out["run_log"]),
        "script": str(out["script"]),
        "latest_for_chatgpt_forbidden_files_found": [],
        "input_audio_feature_files_modified": bool(input_modified),
        "exploratory_reference_files_modified": bool(exploratory_modified),
        "old_project_modified": old_before != old_after,
        "backup_dir": str(backup_dir) if backup_dir else "",
    }
    if report["old_project_modified"]:
        errors.append("old project modified during task")
    if input_modified:
        errors.append("input audio feature files modified during task")
    if exploratory_modified:
        errors.append("exploratory reference files modified during task")

    logger.log("[STEP 13] Run Sub Agent validations")
    summary_text = write_summary(out["summary_md"], report, thresholds, recommendations, val_summary)
    save_json(out["report_json"], report)
    report["sub_agent_results"] = validate(report, out, summary_text)
    for name, result in report["sub_agent_results"].items():
        logger.log(f"Sub Agent {name}: {result['status']} warnings={len(result.get('warnings', []))} errors={len(result.get('errors', []))}")
        for warning in result.get("warnings", []):
            warnings.append(f"{name}: {warning}")
        for error in result.get("errors", []):
            errors.append(f"{name}: {error}")
    report["warnings"] = warnings
    report["errors"] = errors
    summary_text = write_summary(out["summary_md"], report, thresholds, recommendations, val_summary)
    save_json(out["report_json"], report)

    logger.log("[STEP 14] Update latest_for_chatgpt with allowed files only")
    report["latest_for_chatgpt_files"] = update_latest(out, report, backup_dir)
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden_latest_files()
    end_time = now_iso()
    runtime = time.time() - start_epoch
    report["end_time"] = end_time
    report["actual_runtime_seconds"] = round(runtime, 3)
    report["actual_runtime_readable"] = readable_seconds(runtime)
    save_json(out["report_json"], report)
    shutil.copy2(out["report_json"], LATEST_DIR / out["report_json"].name)
    shutil.copy2(out["summary_md"], LATEST_DIR / out["summary_md"].name)

    logger.log(f"Task end time: {end_time}")
    logger.log(f"Actual runtime: {report['actual_runtime_readable']}")
    logger.log(f"Warnings: {warnings}")
    logger.log(f"Errors: {errors}")
    logger.log("[STEP 15] Print human-readable final summary")
    shutil.copy2(out["run_log"], LATEST_DIR / out["run_log"].name)

    status = "실패" if errors or any(r["status"] == "FAIL" for r in report["sub_agent_results"].values()) else "조건부 성공" if warnings or any(r["status"] == "WARN" for r in report["sub_agent_results"].values()) else "성공"
    print("\n# Audio Split-aware Rule Reanalysis Result", flush=True)
    print(f"- status: {status}", flush=True)
    print(f"- split_seed: {SPLIT_SEED}", flush=True)
    print(f"- train/validation/test videos: {report['train_video_ids']} / {report['validation_video_ids']} / {report['test_video_ids']}", flush=True)
    print(f"- train features: {len(train_features)}, thresholds: {len(thresholds)}, recommendations: {len(recommendations)}", flush=True)
    print(f"- validation audit rows: {len(audit)}", flush=True)
    print(f"- report: {out['report_json']}", flush=True)
    print(f"- summary: {out['summary_md']}", flush=True)
    print(f"- latest_bundle: {LATEST_DIR}", flush=True)
    print(f"- errors: {errors}", flush=True)
    return 1 if errors or any(r["status"] == "FAIL" for r in report["sub_agent_results"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
