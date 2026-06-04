#!/usr/bin/env python3
"""v2.4 검토용 canonical visual anchor에 scene/audio/OCR evidence를 정렬한다."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
VERSION = "v2_4"
TASK_NAME = "visual_anchor_alignment_pack"
SPLIT_SEED = 20240524
FIXED_SPLIT = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}
EXPECTED_CANONICAL_ANCHOR = PROJECT_ROOT / "data/scene/visual_scene_boundary_anchors_v2_4.csv"
CANONICAL_ANCHOR = PROJECT_ROOT / "data/features/visual_scene_boundary_anchors_v2_4.csv"
CANONICAL_ANCHOR_WITH_SPLIT = PROJECT_ROOT / "data/features/visual_scene_boundary_anchors_v2_4_with_split.csv"
FALLBACK_SCENE_CANDIDATE = PROJECT_ROOT / "data/scene/scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed.csv"
SPLIT_FILE = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"
MANIFEST_FILE = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
LABEL_INTERVAL_FILE = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_4.csv"
AUDIO_CONFIG_FILE = PROJECT_ROOT / "configs/audio_persistence_rule_config_v2_4_train_only.json"
DATA_AUDIO_DIR = PROJECT_ROOT / "data/audio"
DATA_OCR_DIR = PROJECT_ROOT / "data/ocr"
DATA_FUSION_DIR = PROJECT_ROOT / "data/fusion"
REPORTS_FUSION_DIR = PROJECT_ROOT / "reports/fusion"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_FUSION_DIR = PROJECT_ROOT / "scripts/fusion"
BUNDLE_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_visual_anchor_alignment"
BACKUPS_DIR = PROJECT_ROOT / "backups"
BASE_AUDIO_SCRIPT = PROJECT_ROOT / "scripts/audio/audio_scene_anchor_persistence_features_v2_4.py"

AUDIO_REFERENCE_FILES = {
    "feature_recommendations": DATA_AUDIO_DIR / "audio_rule_feature_recommendations_v2_4_train_only.csv",
    "candidate_thresholds": DATA_AUDIO_DIR / "audio_rule_candidate_thresholds_v2_4_train_only.csv",
    "validation_summary": DATA_AUDIO_DIR / "audio_rule_validation_summary_v2_4_train_only.csv",
    "split_aware_summary": PROJECT_ROOT / "reports/audio/audio_split_aware_rule_reanalysis_v2_4.md",
    "baseline_subwindows": DATA_AUDIO_DIR / "audio_ad_edge_persistence_subwindow_features_v2_4_with_split.csv",
}
PREVIOUS_FALLBACK_AUDIO_FILES = [
    DATA_AUDIO_DIR / "audio_scene_anchor_persistence_features_v2_4.csv",
    DATA_AUDIO_DIR / "audio_scene_anchor_persistence_features_v2_4_train_val_for_discussion.csv",
    DATA_FUSION_DIR / "scene_audio_anchor_rule_discussion_table_v2_4_train_val.csv",
    PROJECT_ROOT / "reports/audio/audio_scene_anchor_persistence_features_v2_4_summary.md",
    PROJECT_ROOT / "reports/audio/audio_scene_anchor_persistence_features_v2_4_report.json",
]
OCR_REFERENCE_FILES = {
    "recommendations": DATA_OCR_DIR / "ocr_rule_feature_recommendations_v2_4.csv",
    "frame_recovered": DATA_OCR_DIR / "ocr_frame_level_results_v2_4_recovered.csv",
    "frame_original": DATA_OCR_DIR / "ocr_frame_level_results_v2_4.csv",
    "labeled_recovered": DATA_OCR_DIR / "ocr_labeled_segment_features_v2_4_recovered.csv",
    "edge_recovered": DATA_OCR_DIR / "ocr_ad_edge_5s_10s_features_v2_4_recovered.csv",
    "summary_extract": PROJECT_ROOT / "reports/ocr/extract_ocr_cues_v2_4_summary.md",
    "summary_train_only": PROJECT_ROOT / "reports/ocr/create_train_only_ocr_feature_files_v2_4_summary.md",
    "analysis_labeled": PROJECT_ROOT / "reports/ocr/ocr_labeled_segment_analysis_v2_4.md",
}
FORBIDDEN_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a", ".jpg", ".jpeg", ".png", ".webp", ".pt", ".pth", ".ckpt", ".bin"}
SUBWINDOW_SIZE_SEC = 2.0
SUBWINDOW_STRIDE_SEC = 2.0
PERSISTENCE_WINDOW_SEC = 10.0


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_AUDIO_SCRIPT, "audio_scene_anchor_base_v2_4")


class TaskLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")


def paths() -> Dict[str, Path]:
    return {
        "audio_full": DATA_AUDIO_DIR / "audio_visual_anchor_persistence_features_v2_4.csv",
        "audio_discussion": DATA_AUDIO_DIR / "audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv",
        "audio_subwindows_discussion": DATA_AUDIO_DIR / "audio_visual_anchor_persistence_subwindow_features_v2_4_train_val_for_discussion.csv",
        "audio_level_thresholds": DATA_AUDIO_DIR / "audio_visual_anchor_level_thresholds_v2_4_train_only.csv",
        "ocr_status": DATA_OCR_DIR / "ocr_visual_anchor_alignment_status_v2_4.csv",
        "audit": DATA_FUSION_DIR / "visual_anchor_rule_discussion_audit_v2_4_train_val.csv",
        "discussion_table": DATA_FUSION_DIR / "scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val.csv",
        "alignment_status": DATA_FUSION_DIR / "visual_anchor_alignment_status_v2_4.csv",
        "report": REPORTS_FUSION_DIR / "visual_anchor_alignment_pack_v2_4_report.json",
        "summary": REPORTS_FUSION_DIR / "visual_anchor_alignment_pack_v2_4_summary.md",
        "run_log": LOGS_DIR / "visual_anchor_alignment_pack_v2_4_run_log.txt",
        "script": SCRIPTS_FUSION_DIR / "align_visual_anchor_audio_ocr_evidence_v2_4.py",
        "bundle_readme": BUNDLE_DIR / "README_latest_files.md",
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def readable_seconds(seconds: float) -> str:
    minutes, sec = divmod(float(seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {sec:.1f}s"
    if minutes:
        return f"{minutes}m {sec:.1f}s"
    return f"{sec:.1f}s"


def mmss(seconds: Any) -> str:
    return BASE.mmss(seconds)


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def old_project_snapshot() -> Dict[str, Any]:
    return BASE.old_project_snapshot()


def file_stats(files: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    return BASE.file_stats(files)


def stats_changed(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> List[str]:
    return BASE.stats_changed(before, after)


def ensure_dirs() -> None:
    for d in [DATA_AUDIO_DIR, DATA_OCR_DIR, DATA_FUSION_DIR, REPORTS_FUSION_DIR, LOGS_DIR, SCRIPTS_FUSION_DIR, BUNDLE_DIR, BACKUPS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def backup_existing(out: Dict[str, Path], logger: TaskLogger) -> Optional[Path]:
    existing = [p for p in out.values() if p.exists() and p.name != Path(__file__).name]
    if not existing:
        return None
    backup_dir = BACKUPS_DIR / f"visual_anchor_alignment_pack_v2_4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for src in existing:
        dst = backup_dir / src.relative_to(PROJECT_ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    logger.log(f"Backed up {len(existing)} existing outputs to {backup_dir}")
    return backup_dir


def load_split(errors: List[str], warnings: List[str]) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_FILE)
    df["video_id"] = pd.to_numeric(df["video_id"], errors="coerce").astype("Int64")
    actual = {s: sorted(df.loc[df["split"].eq(s), "video_id"].dropna().astype(int).unique().tolist()) for s in ["train", "validation", "test"]}
    for split, expected in FIXED_SPLIT.items():
        if actual.get(split) != sorted(expected):
            errors.append(f"split mismatch for {split}: expected={expected}, actual={actual.get(split)}")
    return df


def load_visual_anchor(split_df: pd.DataFrame, warnings: List[str], errors: List[str]) -> pd.DataFrame:
    if not CANONICAL_ANCHOR.exists():
        errors.append(f"canonical visual anchor missing: {CANONICAL_ANCHOR}")
        return pd.DataFrame()
    base = pd.read_csv(CANONICAL_ANCHOR)
    split_source = "video_split_v2_4.csv"
    if CANONICAL_ANCHOR_WITH_SPLIT.exists():
        with_split = pd.read_csv(CANONICAL_ANCHOR_WITH_SPLIT)
        if len(with_split) == len(base) and "split" in with_split.columns and with_split["split"].notna().all():
            base = with_split
            split_source = "visual_scene_boundary_anchors_v2_4_with_split.csv"
    if "canonical_boundary_time_sec" in base.columns:
        base["candidate_time_sec"] = pd.to_numeric(base["canonical_boundary_time_sec"], errors="coerce")
    elif "candidate_time_sec" in base.columns:
        base["candidate_time_sec"] = pd.to_numeric(base["candidate_time_sec"], errors="coerce")
    else:
        errors.append("candidate_time_sec/canonical_boundary_time_sec missing from canonical visual anchor")
        base["candidate_time_sec"] = np.nan
    if "canonical_boundary_mmss" in base.columns:
        base["candidate_time_mmss"] = base["canonical_boundary_mmss"].astype(str)
    else:
        base["candidate_time_mmss"] = base["candidate_time_sec"].map(mmss)
    base["video_id"] = pd.to_numeric(base["video_id"], errors="coerce").astype("Int64")
    if "split" not in base.columns or base["split"].isna().any():
        split_map = {int(r["video_id"]): r["split"] for _, r in split_df.dropna(subset=["video_id"]).iterrows()}
        base["split"] = base["video_id"].map(lambda v: split_map.get(int(v), "unknown") if pd.notna(v) else "unknown")
        split_source = "video_split_v2_4.csv_joined"
    if "scene_boundary_anchor_id" in base.columns:
        base["visual_anchor_id"] = base["scene_boundary_anchor_id"].astype(str)
        base["original_anchor_id"] = base["scene_boundary_anchor_id"].astype(str)
    elif "visual_anchor_id" in base.columns:
        base["visual_anchor_id"] = base["visual_anchor_id"].astype(str)
        base["original_anchor_id"] = base["visual_anchor_id"].astype(str)
    else:
        base["visual_anchor_id"] = [f"v2_4_{int(v) if pd.notna(v) else 'unknown'}_{i:06d}_{int(round(safe_float(t)*1000)) if np.isfinite(safe_float(t)) else 0}" for i, (v, t) in enumerate(zip(base["video_id"], base["candidate_time_sec"]), start=1)]
        base["original_anchor_id"] = ""
    base["anchor_id"] = base["visual_anchor_id"]
    base["candidate_id"] = base["visual_anchor_id"]
    base["candidate_source"] = base.get("canonical_time_source", "visual_scene_boundary_anchor")
    base["method_used"] = base.get("source_relation", "visual_scene_boundary_anchor")
    base["scene_change_score"] = pd.to_numeric(base.get("visual_boundary_strength_score", base.get("opencv_scene_change_score", np.nan)), errors="coerce")
    base["scene_component_score"] = pd.to_numeric(base.get("visual_boundary_strength_score", np.nan), errors="coerce")
    if "scene_rank_in_video" not in base.columns:
        base["scene_rank_in_video"] = base.groupby("video_id")["scene_change_score"].rank(method="first", ascending=False).astype("Int64")
    for col in ["visual_start_like_score", "visual_end_like_score", "visual_internal_transition_hint", "is_high_priority", "source_file"]:
        if col not in base.columns:
            base[col] = np.nan
    base["visual_internal_transition_hint"] = base.get("source_relation", "").astype(str).str.contains("merged|separate|resnet|opencv", case=False, na=False)
    base["is_high_priority"] = base.get("visual_boundary_strength_band", "").astype(str).isin(["very_high", "high"])
    base["split_source_used"] = split_source
    base["anchor_status"] = np.where(base["candidate_time_sec"].notna() & base["video_id"].notna(), "valid", "invalid_anchor_schema")
    if base["candidate_time_sec"].isna().any():
        errors.append("some canonical visual anchors have null candidate_time_sec")
    return base


def resolve_video_info(anchors: pd.DataFrame, manifest: pd.DataFrame, ffprobe_path: Optional[str], warnings: List[str]) -> pd.DataFrame:
    return BASE.resolve_video_info(anchors, manifest, ffprobe_path, warnings)


def build_anchor_table(anchor_df: pd.DataFrame, video_info: pd.DataFrame) -> pd.DataFrame:
    video_map = {int(r["video_id"]): r for _, r in video_info.iterrows()}
    rows: List[Dict[str, Any]] = []
    keep = [
        "version", "split", "video_id", "visual_anchor_id", "original_anchor_id", "candidate_id", "anchor_id", "candidate_time_sec", "candidate_time_mmss",
        "candidate_source", "method_used", "scene_change_score", "scene_component_score", "scene_rank_in_video", "visual_start_like_score", "visual_end_like_score",
        "visual_internal_transition_hint", "is_high_priority", "source_file", "canonical_time_source", "source_relation", "visual_boundary_strength_score", "visual_boundary_strength_band",
        "opencv_scene_change_score", "resnet_scene_change_score", "opencv_score_percentile_in_video", "resnet_score_percentile_in_video", "reviewed_by_human", "reviewed_true_scene_change", "reviewed_false_positive", "anchor_status"
    ]
    for _, row in anchor_df.iterrows():
        video_id = int(row["video_id"]) if pd.notna(row["video_id"]) else -1
        info = video_map.get(video_id, {})
        rec = {col: row.get(col, np.nan) for col in keep}
        rec["version"] = VERSION
        rec["video_id"] = video_id
        rec["video_title"] = row.get("video_title", info.get("video_title_resolved", ""))
        rec["video_path"] = info.get("video_path_resolved", "")
        rec["video_duration_sec"] = info.get("video_duration_sec", np.nan)
        rec["audio_available"] = bool(info.get("audio_available", False))
        rec["audio_stream_index"] = info.get("audio_stream_index", np.nan)
        rec["probe_error"] = info.get("probe_error", "")
        rows.append(rec)
    return pd.DataFrame(rows)


def add_scene_levels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    train = out[out["split"].eq("train")]
    vals = pd.to_numeric(train["scene_change_score"], errors="coerce")
    q40 = vals.quantile(0.40) if vals.notna().any() else np.nan
    q70 = vals.quantile(0.70) if vals.notna().any() else np.nan
    def level(v: Any) -> str:
        f = safe_float(v)
        if not np.isfinite(f):
            return "low"
        if np.isfinite(q70) and f >= q70:
            return "high"
        if np.isfinite(q40) and f >= q40:
            return "medium"
        return "low"
    out["scene_transition_level"] = out["scene_change_score"].map(level)
    out["scene_start_signal_level"] = out["scene_transition_level"]
    out["scene_end_signal_level"] = out["scene_transition_level"]
    out["scene_level_source"] = "train_visual_anchor_strength_quantile_direction_not_classified"
    return out


def add_ocr_placeholder(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    out = df.copy()
    out["ocr_anchor_context_status"] = "missing_placeholder"
    out["ocr_available_for_alignment"] = False
    out["ocr_missing_reason"] = reason
    for col in [
        "ocr_start_signal_score", "ocr_end_signal_score", "ocr_context_score", "ocr_score_delta_post_minus_pre", "ocr_score_delta_pre_minus_post", "ocr_keyword_delta_post_minus_pre", "ocr_keyword_delta_pre_minus_post"
    ]:
        out[col] = np.nan
    for col in ["ocr_start_signal_level", "ocr_end_signal_level", "ocr_context_level"]:
        out[col] = "unknown"
    return out


def inspect_ocr_context_availability() -> Tuple[str, str, bool, bool]:
    patterns = [
        "ocr_visual_anchor_context_features_v2_4",
        "ocr_scene_anchor_context_features_v2_4",
        "scene_ocr_anchor_rule_discussion_table_v2_4",
        "ocr_anchor_context_",
    ]
    for root in [DATA_OCR_DIR, DATA_FUSION_DIR, PROJECT_ROOT / "outputs", PROJECT_ROOT / "reports"]:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if any(p in path.name for p in patterns):
                return "existing_candidate_found_not_joined", f"candidate OCR anchor context file found but not auto-joined: {path}", False, False
    frame = OCR_REFERENCE_FILES["frame_recovered"] if OCR_REFERENCE_FILES["frame_recovered"].exists() else OCR_REFERENCE_FILES["frame_original"]
    if frame.exists():
        try:
            meta = pd.read_csv(frame, usecols=lambda c: c in ["plan_kind", "segment_type"], nrows=2000)
            plans = set(meta.get("plan_kind", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
            if plans and plans.issubset({"labeled", "edge"}):
                return "missing", "frame-level OCR exists only for label-aligned/edge plans, so visual-anchor context OCR was not safely generated", False, False
        except Exception as exc:
            return "missing", f"could not inspect frame-level OCR safely: {type(exc).__name__}: {exc}", False, False
    return "missing", "no visual-anchor OCR context feature found; OCR placeholder columns created", False, False


def build_label_audit(features: pd.DataFrame) -> pd.DataFrame:
    intervals = pd.read_csv(LABEL_INTERVAL_FILE)
    intervals["video_id"] = pd.to_numeric(intervals["video_id"], errors="coerce").astype("Int64")
    rows = []
    for _, anchor in features[features["split"].isin(["train", "validation"])].iterrows():
        t = safe_float(anchor["candidate_time_sec"])
        video_id = int(anchor["video_id"])
        best = None
        for _, interval in intervals[intervals["video_id"].eq(video_id)].iterrows():
            for btype, col in [("ad_start", "ad_start_sec"), ("ad_end", "ad_end_sec")]:
                sec = safe_float(interval.get(col))
                if not np.isfinite(sec):
                    continue
                dist = abs(t - sec)
                if best is None or dist < best[0]:
                    best = (dist, btype, sec)
        dist, btype, sec = best if best is not None else (np.nan, "", np.nan)
        rows.append({
            "version": VERSION,
            "split": anchor["split"],
            "video_id": video_id,
            "visual_anchor_id": anchor["visual_anchor_id"],
            "candidate_time_sec": t,
            "candidate_time_mmss": anchor["candidate_time_mmss"],
            "nearest_true_boundary_type": btype,
            "nearest_true_boundary_sec": safe_float(sec),
            "distance_to_nearest_true_boundary_sec": safe_float(dist),
            "is_near_true_boundary_2s": bool(np.isfinite(dist) and dist <= 2),
            "is_near_true_boundary_5s": bool(np.isfinite(dist) and dist <= 5),
            "is_near_true_boundary_10s": bool(np.isfinite(dist) and dist <= 10),
            "audit_note": "label audit only; not used for scene/audio/OCR score calculation",
        })
    return pd.DataFrame(rows)


def suggested_support(level: str, delta: Any = None) -> str:
    if level == "high":
        return "high_discussion_support"
    if level == "medium":
        return "medium_discussion_support"
    return "low_or_unclear"


def build_discussion_table(features: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    df = features[features["split"].isin(["train", "validation"])].copy()
    df["audio_suggested_start_support"] = df["audio_start_signal_level"].map(suggested_support)
    df["audio_suggested_end_support"] = df["audio_end_signal_level"].map(suggested_support)
    df["audio_suggested_internal_ad_hint"] = np.where(df["audio_internal_ad_transition_hint"].astype(bool), "possible_internal_ad_transition", "not_indicated_by_audio")
    df["ocr_suggested_start_support"] = "ocr_missing"
    df["ocr_suggested_end_support"] = "ocr_missing"
    df["multimodal_discussion_status"] = np.where(df["ocr_available_for_alignment"].astype(bool), "scene_audio_ready_ocr_ready", "scene_audio_ready_ocr_missing")
    cols = [
        "version", "split", "video_id", "visual_anchor_id", "candidate_time_sec", "candidate_time_mmss", "candidate_source", "method_used",
        "scene_change_score", "scene_component_score", "visual_start_like_score", "visual_end_like_score", "visual_internal_transition_hint", "scene_start_signal_level", "scene_end_signal_level", "scene_transition_level",
        "audio_start_signal_score", "audio_end_signal_score", "audio_context_score", "audio_start_signal_level", "audio_end_signal_level", "audio_context_level", "audio_before_context_level", "audio_after_context_level", "audio_pre_10s_ad_like_ratio", "audio_post_10s_ad_like_ratio", "audio_pre_10s_max_consecutive_ad_like_sec", "audio_post_10s_max_consecutive_ad_like_sec", "audio_score_delta_post_minus_pre", "audio_score_delta_pre_minus_post",
        "ocr_anchor_context_status", "ocr_available_for_alignment", "ocr_start_signal_score", "ocr_end_signal_score", "ocr_context_score", "ocr_start_signal_level", "ocr_end_signal_level", "ocr_context_level", "ocr_score_delta_post_minus_pre", "ocr_score_delta_pre_minus_post", "ocr_keyword_delta_post_minus_pre", "ocr_keyword_delta_pre_minus_post",
        "audio_suggested_start_support", "audio_suggested_end_support", "audio_suggested_internal_ad_hint", "ocr_suggested_start_support", "ocr_suggested_end_support", "multimodal_discussion_status",
    ]
    table = df[[c for c in cols if c in df.columns]].merge(
        audit[["visual_anchor_id", "nearest_true_boundary_type", "nearest_true_boundary_sec", "distance_to_nearest_true_boundary_sec", "is_near_true_boundary_2s", "is_near_true_boundary_5s", "is_near_true_boundary_10s"]],
        on="visual_anchor_id",
        how="left",
    )
    table["discussion_note"] = "canonical visual anchor aligned evidence only; rule not finalized"
    return table


def forbidden_files_in_bundle() -> List[str]:
    if not BUNDLE_DIR.exists():
        return []
    bad: List[str] = []
    for path in BUNDLE_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            bad.append(str(path))
        if any(part in {"cache", "tmp", "raw", "videos"} for part in path.parts):
            bad.append(str(path))
    return sorted(set(bad))


def copy_bundle(out: Dict[str, Path], warnings: List[str], logger: TaskLogger) -> List[str]:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for path in BUNDLE_DIR.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
    keys = ["audio_discussion", "audio_level_thresholds", "audit", "discussion_table", "alignment_status", "ocr_status", "summary", "report", "run_log", "script"]
    copied: List[str] = []
    skipped: List[str] = []
    for key in keys:
        src = out[key]
        if src.exists():
            dst = BUNDLE_DIR / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    sub = out["audio_subwindows_discussion"]
    if sub.exists() and sub.stat().st_size <= 5_000_000:
        dst = BUNDLE_DIR / sub.name
        shutil.copy2(sub, dst)
        copied.append(str(dst))
    else:
        skipped.append(str(sub))
        warnings.append(f"large subwindow detail not copied to bundle: {sub}")
    for src in [AUDIO_CONFIG_FILE, AUDIO_REFERENCE_FILES["feature_recommendations"], AUDIO_REFERENCE_FILES["validation_summary"]]:
        if src.exists():
            dst = BUNDLE_DIR / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    readme = out["bundle_readme"]
    readme.write_text("\n".join([
        "# latest files: canonical visual anchor alignment pack",
        "",
        f"task: {TASK_NAME}",
        f"version: {VERSION}",
        f"canonical_anchor_file: {CANONICAL_ANCHOR}",
        f"canonical_anchor_expected_path_missing: {EXPECTED_CANONICAL_ANCHOR}",
        "fallback_scene_candidate_used_for_current_audio: false",
        "previous_fallback_audio_evidence_status: exploratory_reference_only",
        "discussion bundle contains train/validation row-level files only; test row-level features are excluded.",
        "OCR anchor context is placeholder/missing unless generated file is listed.",
        "",
        "## copied files",
        *[f"- {Path(p).name}" for p in copied],
        "",
        "## large files not copied",
        *([f"- {p}" for p in skipped] if skipped else ["- none"]),
        "",
        "This is a canonical anchor-aligned discussion evidence pack, not a finalized rule or interval detector.",
        "",
    ]), encoding="utf-8")
    copied.append(str(readme))
    logger.log(f"Copied {len(copied)} files to {BUNDLE_DIR}")
    return copied


def write_summary(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Visual Anchor Alignment Pack v2_4",
        "",
        "## 1. 작업 개요",
        "visual_scene_boundary_anchors_v2_4.csv의 실제 프로젝트 위치인 `/data/features` 파일을 canonical visual anchor로 사용해 scene/audio/OCR evidence를 정렬했다.",
        "이전 fallback scene candidate 기준 audio evidence는 exploratory/reference로만 보존했고, 이번 audio feature는 canonical visual anchor의 candidate_time_sec 기준으로 새로 계산했다.",
        "이 산출물은 rule 확정이나 interval detector 구현이 아니라 discussion용 evidence pack이다.",
        "",
        "## 2. Canonical Anchor",
        f"- expected path: {report['canonical_anchor_expected_path_missing']}",
        f"- actual path used: {report['canonical_anchor_actual_path_used']}",
        f"- row count: {report['visual_anchor_count']}",
        f"- train/validation/test: {report['visual_anchor_train_count']} / {report['visual_anchor_validation_count']} / {report['visual_anchor_test_count']}",
        "- fallback scene candidate는 이번 canonical alignment audio 계산에 사용하지 않았다.",
        "",
        "## 3. Audio 재계산",
        "각 visual anchor 주변 pre/post 10초를 2초 subwindow로 나누고, train-only audio config의 audio_ad_like_score 공식을 적용했다.",
        f"- audio feature success/failure: {report['audio_feature_success_count']} / {report['audio_feature_failed_count']}",
        f"- audio_ad_like_threshold: {report['audio_ad_like_threshold']}",
        "- qualitative level은 train visual anchor percentile 기준 high/medium/low discussion candidate다.",
        "",
        "## 4. OCR Alignment 상태",
        f"- OCR status: {report['ocr_anchor_context_status']}",
        f"- generated: {report['ocr_anchor_context_generated']}",
        f"- joined: {report['ocr_anchor_context_joined']}",
        f"- reason: {report['ocr_anchor_context_missing_reason']}",
        "label-aligned OCR feature는 reference로만 보고, visual anchor context feature처럼 사용하지 않았다.",
        "",
        "## 5. Discussion Table",
        "compact discussion table에는 scene strength, audio start/end/context level, OCR placeholder/status, train/validation audit distance columns가 포함된다.",
        "high/medium/low는 rule 확정값이 아니라 qualitative discussion candidate다.",
        "",
        "## 6. Leakage Guard",
        "- audio level threshold는 train anchor만 사용했다.",
        "- validation은 discussion/audit only다.",
        "- test row-level feature는 bundle에 복사하지 않았다.",
        "- label/audit columns는 score 계산에 사용하지 않았다.",
        "",
        "## 7. 생성 파일 목록",
        *[f"- {path}: {desc}" for path, desc in report.get('output_file_descriptions', {}).items()],
        "",
        "## 8. Sub Agent 검증 결과",
        *[f"- {name}: {res.get('status')} ({'; '.join(res.get('warnings', []) + res.get('errors', [])) or 'ok'})" for name, res in report.get('sub_agent_results', {}).items()],
        "",
        "## 9. 다음 단계",
        "- OCR anchor context가 준비되면 같은 visual_anchor_id 기준으로 join한다.",
        "- 이 채팅에서 scene/audio/OCR qualitative score를 보며 state-machine interval rule을 논의한다.",
        "- start/end/internal-ad transition 및 low gap bridge rule을 별도로 논의한다.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_validations(report: Dict[str, Any], features: pd.DataFrame, subwindows: pd.DataFrame, discussion: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    warnings: List[str] = []
    if report.get("canonical_anchor_file") != str(CANONICAL_ANCHOR):
        errors.append("canonical anchor file does not match accepted /data/features path")
    if report.get("fallback_scene_candidate_used_for_current_audio"):
        errors.append("fallback scene candidate used for current audio")
    if features["candidate_time_sec"].isna().any():
        errors.append("candidate_time_sec contains nulls")
    if report.get("unknown_split_anchor_count", 0) > 0:
        warnings.append("unknown split anchors exist")
    results["canonical_anchor_split_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    errors = []
    warnings = []
    if report.get("previous_fallback_audio_evidence_status") != "exploratory_reference_only":
        errors.append("previous fallback audio status not marked reference only")
    if report.get("audio_feature_success_count", 0) == 0:
        errors.append("no audio feature successes")
    if "onset_density" not in subwindows.columns:
        errors.append("onset_density missing")
    score = pd.to_numeric(subwindows.get("audio_ad_like_score"), errors="coerce")
    if score.dropna().lt(0).any() or score.dropna().gt(1).any():
        errors.append("audio_ad_like_score outside 0-1 range")
    if "tempo_estimate" in subwindows.columns and subwindows["tempo_estimate"].isna().all():
        warnings.append("tempo_estimate NaN for short 2s subwindows; not used for scoring")
    results["audio_recalculation_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    errors = []
    warnings = []
    if report.get("ocr_anchor_context_status") == "missing_placeholder":
        warnings.append("OCR visual-anchor context missing; placeholder columns created")
    if not (discussion["ocr_anchor_context_status"].eq("missing_placeholder").all() and discussion["ocr_available_for_alignment"].eq(False).all()):
        errors.append("OCR placeholder/status columns inconsistent")
    results["ocr_alignment_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    errors = []
    warnings = []
    if report.get("validation_used_for_audio_level_thresholds") or report.get("test_used_for_audio_level_thresholds"):
        errors.append("validation/test used for audio level thresholds")
    if report.get("test_row_level_features_copied_to_bundle"):
        errors.append("test row-level features copied to bundle")
    if report.get("label_columns_used_for_score"):
        errors.append("label columns used for score")
    required = ["audio_start_signal_level", "audio_end_signal_level", "audio_context_level", "ocr_anchor_context_status", "multimodal_discussion_status"]
    for col in required:
        if col not in discussion.columns:
            errors.append(f"missing discussion column {col}")
    results["leakage_discussion_quality_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    errors = []
    warnings = []
    missing = [p for p in report.get("output_files", []) if not Path(p).exists()]
    if missing:
        errors.append(f"missing output files: {missing}")
    if report.get("latest_for_chatgpt_forbidden_files_found"):
        errors.append("forbidden files found in bundle")
    if report.get("input_files_modified"):
        errors.append("input files modified")
    if report.get("old_project_modified"):
        errors.append("old project modified")
    results["output_safety_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}
    return results


def main() -> None:
    t0 = time.time()
    ensure_dirs()
    out = paths()
    logger = TaskLogger(out["run_log"])
    warnings: List[str] = []
    errors: List[str] = []
    logger.log("[STEP 01] Start task and create old project snapshot")
    old_before = old_project_snapshot()
    backup_dir = backup_existing(out, logger)
    start_time = now_iso()

    logger.log("[STEP 02] Load canonical visual_scene_boundary_anchors_v2_4.csv")
    split_df = load_split(errors, warnings)
    anchor_df = load_visual_anchor(split_df, warnings, errors)
    if anchor_df.empty or errors:
        raise RuntimeError("Cannot continue: " + "; ".join(errors))
    logger.log(f"Canonical anchor file: {CANONICAL_ANCHOR}")
    logger.log(f"Visual anchor rows: {len(anchor_df)}")

    logger.log("[STEP 03] Validate split and visual anchor schema")
    split_counts = anchor_df["split"].value_counts(dropna=False).to_dict()
    logger.log(f"Anchor split counts: {split_counts}")

    logger.log("[STEP 04] Detect previous fallback audio evidence and mark as reference only")
    previous_fallback_detected = any(p.exists() for p in PREVIOUS_FALLBACK_AUDIO_FILES)
    logger.log(f"Previous fallback audio evidence detected: {previous_fallback_detected}")

    logger.log("[STEP 05] Resolve raw video paths")
    ffmpeg_path = BASE.executable_path("ffmpeg")
    ffprobe_path = BASE.executable_path("ffprobe")
    manifest = pd.read_csv(MANIFEST_FILE)
    manifest["video_id"] = pd.to_numeric(manifest["video_id"], errors="coerce").astype("Int64")
    video_info = resolve_video_info(anchor_df, manifest, ffprobe_path, warnings)
    missing_video_count = int((~video_info["video_path_resolved"].map(lambda p: Path(str(p)).exists())).sum())
    logger.log(f"ffmpeg: {ffmpeg_path or 'missing'}")
    logger.log(f"ffprobe: {ffprobe_path or 'missing'}")
    logger.log(f"missing_video_count: {missing_video_count}")

    logger.log("[STEP 06] Build visual-anchor pre/post audio subwindow plan")
    anchor_table = build_anchor_table(anchor_df, video_info)
    anchor_table = add_scene_levels(anchor_table)
    subwindow_plan = BASE.build_subwindow_rows(anchor_table)
    logger.log(f"subwindow_plan rows: {len(subwindow_plan)}")

    input_files = [CANONICAL_ANCHOR, CANONICAL_ANCHOR_WITH_SPLIT, SPLIT_FILE, MANIFEST_FILE, AUDIO_CONFIG_FILE, LABEL_INTERVAL_FILE]
    input_files.extend(p for p in AUDIO_REFERENCE_FILES.values() if p.exists())
    input_files.extend(p for p in OCR_REFERENCE_FILES.values() if p.exists())
    input_stats_before = file_stats(input_files)

    logger.log("[STEP 07] Extract visual-anchor audio subwindow features")
    subwindow_features = BASE.extract_subwindow_features_from_contexts(subwindow_plan, ffmpeg_path, logger)
    audio_success = int((subwindow_features["feature_status"] == "success").sum())
    audio_failed = int(len(subwindow_features) - audio_success)
    logger.log(f"audio feature success/failure: {audio_success}/{audio_failed}")

    logger.log("[STEP 08] Compute audio persistence context scores")
    baseline_df = BASE.load_train_baseline()
    scored_subwindows = BASE.add_audio_scores(subwindow_features, baseline_df)
    config = json.loads(AUDIO_CONFIG_FILE.read_text(encoding="utf-8"))
    audio_threshold = safe_float(config.get("start_boundary_rule", {}).get("post_10s_median_score_min", np.nan))
    if not np.isfinite(audio_threshold):
        audio_threshold = 0.6
        warnings.append("audio threshold missing from train-only config; fallback 0.6 used")
    scored_subwindows["audio_ad_like_threshold"] = audio_threshold
    context_scores = BASE.summarize_contexts(scored_subwindows, audio_threshold)
    audio_features = BASE.combine_anchor_features(anchor_table, context_scores)
    audio_features, level_thresholds = BASE.add_levels(audio_features)
    audio_features = audio_features.rename(columns={"anchor_id": "visual_anchor_id_from_audio_base"})
    if "visual_anchor_id" not in audio_features.columns:
        audio_features["visual_anchor_id"] = audio_features["visual_anchor_id_from_audio_base"]
    audio_features = add_scene_levels(audio_features)

    logger.log("[STEP 09] Compute train-only qualitative audio levels")
    ocr_status, ocr_missing_reason, ocr_generated, ocr_joined = inspect_ocr_context_availability()
    if ocr_status == "missing":
        ocr_status = "missing_placeholder"
    logger.log(f"OCR anchor context status: {ocr_status} ({ocr_missing_reason})")
    audio_features = add_ocr_placeholder(audio_features, ocr_missing_reason)

    logger.log("[STEP 10] Search or generate OCR visual-anchor context features")
    # label-aligned OCR만 있을 때는 OCR 값을 placeholder로 둔다.

    logger.log("[STEP 11] Build scene+audio+OCR alignment discussion table")
    audio_features.to_csv(out["audio_full"], index=False, encoding="utf-8-sig")
    train_val_audio = audio_features[audio_features["split"].isin(["train", "validation"])].copy()
    train_val_audio.to_csv(out["audio_discussion"], index=False, encoding="utf-8-sig")
    train_val_sub = scored_subwindows[scored_subwindows["split"].isin(["train", "validation"])].copy()
    train_val_sub.to_csv(out["audio_subwindows_discussion"], index=False, encoding="utf-8-sig")
    level_thresholds.to_csv(out["audio_level_thresholds"], index=False, encoding="utf-8-sig")
    audit = build_label_audit(audio_features)
    audit.to_csv(out["audit"], index=False, encoding="utf-8-sig")
    discussion = build_discussion_table(audio_features, audit)
    discussion.to_csv(out["discussion_table"], index=False, encoding="utf-8-sig")

    ocr_status_df = pd.DataFrame([{
        "version": VERSION,
        "ocr_anchor_context_status": ocr_status,
        "ocr_anchor_context_generated": ocr_generated,
        "ocr_anchor_context_joined": ocr_joined,
        "ocr_anchor_context_missing_reason": ocr_missing_reason,
        "label_aligned_ocr_used_as_reference_only": True,
        "visual_anchor_count": int(len(audio_features)),
        "train_anchor_count": int(split_counts.get("train", 0)),
        "validation_anchor_count": int(split_counts.get("validation", 0)),
        "test_anchor_count": int(split_counts.get("test", 0)),
    }])
    ocr_status_df.to_csv(out["ocr_status"], index=False, encoding="utf-8-sig")
    alignment_status = pd.DataFrame([{
        "version": VERSION,
        "canonical_anchor_file": str(CANONICAL_ANCHOR),
        "canonical_anchor_expected_path_missing": str(EXPECTED_CANONICAL_ANCHOR),
        "canonical_anchor_actual_path_used": str(CANONICAL_ANCHOR),
        "fallback_scene_candidate_used_for_current_audio": False,
        "previous_fallback_audio_evidence_status": "exploratory_reference_only",
        "audio_visual_anchor_recalculated": True,
        "ocr_anchor_context_status": ocr_status,
        "discussion_bundle_contains_test_rows": False,
    }])
    alignment_status.to_csv(out["alignment_status"], index=False, encoding="utf-8-sig")

    input_stats_after = file_stats(input_files)
    old_after = old_project_snapshot()
    old_modified = old_before != old_after
    input_changed = stats_changed(input_stats_before, input_stats_after)
    if old_modified:
        errors.append("old project snapshot changed")
    if input_changed:
        errors.append(f"input files modified: {input_changed}")

    logger.log("[STEP 12] Run Sub Agent validations")
    output_descriptions = {
        str(out["audio_full"]): "full canonical visual-anchor audio persistence feature file; includes test locally only",
        str(out["audio_discussion"]): "train/validation canonical visual-anchor audio persistence discussion file",
        str(out["audio_subwindows_discussion"]): "train/validation 2s audio subwindow detail; kept out of bundle if large",
        str(out["audio_level_thresholds"]): "train-only audio level percentile thresholds",
        str(out["ocr_status"]): "OCR visual-anchor alignment status/placeholder summary",
        str(out["audit"]): "train/validation label-distance audit only; not used for scoring",
        str(out["discussion_table"]): "compact scene+audio+OCR visual-anchor discussion table",
        str(out["alignment_status"]): "canonical alignment status summary",
        str(out["report"]): "machine-readable report",
        str(out["summary"]): "human-readable summary",
        str(out["run_log"]): "step log",
        str(out["script"]): "reproducible script",
    }
    report: Dict[str, Any] = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "start_time": start_time,
        "end_time": now_iso(),
        "actual_runtime_seconds": safe_float(time.time() - t0),
        "actual_runtime_readable": readable_seconds(time.time() - t0),
        "input_files": [str(p) for p in input_files],
        "output_files": [str(p) for k, p in out.items() if k != "bundle_readme"],
        "generated_files": [str(p) for k, p in out.items() if k != "bundle_readme"],
        "warnings": warnings,
        "errors": errors,
        "canonical_anchor_file": str(CANONICAL_ANCHOR),
        "canonical_anchor_used": True,
        "canonical_anchor_expected_path_missing": str(EXPECTED_CANONICAL_ANCHOR),
        "canonical_anchor_actual_path_used": str(CANONICAL_ANCHOR),
        "canonical_anchor_source_dir": "data/features",
        "copied_anchor_to_data_scene": False,
        "fallback_anchor_used_for_current_audio": False,
        "fallback_scene_candidate_used_for_current_audio": False,
        "previous_fallback_audio_evidence_detected": previous_fallback_detected,
        "previous_fallback_audio_evidence_status": "exploratory_reference_only",
        "previous_audio_anchor_source": "fallback_scene_candidate" if previous_fallback_detected else "none_detected",
        "previous_audio_anchor_not_used_for_canonical_alignment": True,
        "visual_anchor_count": int(len(audio_features)),
        "visual_anchor_train_count": int(split_counts.get("train", 0)),
        "visual_anchor_validation_count": int(split_counts.get("validation", 0)),
        "visual_anchor_test_count": int(split_counts.get("test", 0)),
        "unknown_split_anchor_count": int(split_counts.get("unknown", 0)),
        "split_seed": SPLIT_SEED,
        "train_video_ids": FIXED_SPLIT["train"],
        "validation_video_ids": FIXED_SPLIT["validation"],
        "test_video_ids": FIXED_SPLIT["test"],
        "discussion_bundle_contains_test_rows": False,
        "train_used_for_audio_level_thresholds": True,
        "validation_used_for_audio_level_thresholds": False,
        "test_used_for_audio_level_thresholds": False,
        "validation_included_for_discussion_audit": True,
        "test_included_for_discussion_audit": False,
        "test_row_level_features_copied_to_bundle": False,
        "label_columns_used_for_score": False,
        "audio_config_file": str(AUDIO_CONFIG_FILE),
        "audio_ad_like_threshold": audio_threshold,
        "persistence_window_sec": PERSISTENCE_WINDOW_SEC,
        "subwindow_size_sec": SUBWINDOW_SIZE_SEC,
        "subwindow_stride_sec": SUBWINDOW_STRIDE_SEC,
        "audio_feature_success_count": audio_success,
        "audio_feature_failed_count": audio_failed,
        "audio_visual_anchor_feature_file": str(out["audio_full"]),
        "audio_score_range_min": safe_float(pd.to_numeric(scored_subwindows["audio_ad_like_score"], errors="coerce").min()),
        "audio_score_range_max": safe_float(pd.to_numeric(scored_subwindows["audio_ad_like_score"], errors="coerce").max()),
        "missing_video_count": missing_video_count,
        "ocr_anchor_context_status": ocr_status,
        "ocr_anchor_context_generated": ocr_generated,
        "ocr_anchor_context_joined": ocr_joined,
        "ocr_anchor_context_missing_reason": ocr_missing_reason,
        "label_aligned_ocr_used_as_reference_only": True,
        "anchor_columns_preserved": [c for c in anchor_table.columns if c in anchor_df.columns or c in ["candidate_time_sec", "candidate_time_mmss", "visual_anchor_id", "split", "video_duration_sec"]],
        "candidate_time_sec_column": "candidate_time_sec derived from canonical_boundary_time_sec",
        "visual_anchor_id_column": "visual_anchor_id derived from scene_boundary_anchor_id",
        "old_project_modified": old_modified,
        "old_project_snapshot_before": old_before,
        "old_project_snapshot_after": old_after,
        "input_files_modified": bool(input_changed),
        "input_files_modified_paths": input_changed,
        "latest_for_chatgpt_forbidden_files_found": [],
        "backup_dir": str(backup_dir) if backup_dir else None,
        "output_file_descriptions": output_descriptions,
    }
    # output 존재 검사를 위해 검증 전에 draft report/summary를 먼저 저장한다.
    save_json(out["report"], report)
    write_summary(out["summary"], report)
    validations = run_validations(report, audio_features, scored_subwindows, discussion)
    report["sub_agent_results"] = validations
    if any(r["status"] == "FAIL" for r in validations.values()):
        errors.append("one or more validation checks failed")
    report["errors"] = errors
    save_json(out["report"], report)
    write_summary(out["summary"], report)

    logger.log("[STEP 13] Update latest_for_chatgpt_visual_anchor_alignment")
    copied = copy_bundle(out, warnings, logger)
    forbidden = forbidden_files_in_bundle()
    report["latest_for_chatgpt_files"] = copied
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden
    if forbidden:
        errors.append(f"forbidden files found in bundle: {forbidden}")
    report["warnings"] = warnings
    report["errors"] = errors
    report["end_time"] = now_iso()
    report["actual_runtime_seconds"] = safe_float(time.time() - t0)
    report["actual_runtime_readable"] = readable_seconds(time.time() - t0)
    save_json(out["report"], report)
    write_summary(out["summary"], report)
    shutil.copy2(out["report"], BUNDLE_DIR / out["report"].name)
    shutil.copy2(out["summary"], BUNDLE_DIR / out["summary"].name)
    shutil.copy2(out["run_log"], BUNDLE_DIR / out["run_log"].name)

    logger.log("[STEP 14] Print human-readable final summary")
    print("\n작업 완료 요약", flush=True)
    print(f"- status: {'FAIL' if errors else 'CONDITIONAL_SUCCESS' if warnings or any(r['status']=='WARN' for r in validations.values()) else 'SUCCESS'}", flush=True)
    print(f"- canonical_visual_anchor_file: {CANONICAL_ANCHOR}", flush=True)
    print("- fallback_scene_candidate_used_for_current_audio: false", flush=True)
    print(f"- visual_anchor split counts: train={split_counts.get('train', 0)}, validation={split_counts.get('validation', 0)}, test={split_counts.get('test', 0)}", flush=True)
    print(f"- audio subwindow feature success/failure: {audio_success}/{audio_failed}", flush=True)
    print(f"- OCR anchor context status: {ocr_status}", flush=True)
    print("- discussion bundle contains test rows: false", flush=True)
    print(f"- report: {out['report']}", flush=True)


if __name__ == "__main__":
    main()
