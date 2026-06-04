#!/usr/bin/env python3
"""v2.4 검토용 scene anchor 오디오 persistence evidence pack을 만든다."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
VERSION = "v2_4"
TASK_NAME = "audio_scene_anchor_persistence_features"
SPLIT_SEED = 20240524
SPLIT_FILE = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"
FIXED_SPLIT = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}

DATA_AUDIO_DIR = PROJECT_ROOT / "data/audio"
DATA_FUSION_DIR = PROJECT_ROOT / "data/fusion"
DATA_SCENE_DIR = PROJECT_ROOT / "data/scene"
DATA_REVIEW_DIR = PROJECT_ROOT / "data/review"
REPORTS_AUDIO_DIR = PROJECT_ROOT / "reports/audio"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_AUDIO_DIR = PROJECT_ROOT / "scripts/audio"
CONFIGS_DIR = PROJECT_ROOT / "configs"
BUNDLE_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_audio_scene_anchor"
BACKUPS_DIR = PROJECT_ROOT / "backups"

PREFERRED_ANCHOR_FILE = DATA_SCENE_DIR / "visual_scene_boundary_anchors_v2_4.csv"
ANCHOR_FALLBACKS = [
    DATA_SCENE_DIR / "visual_scene_boundary_anchors_v2_4.csv",
    DATA_SCENE_DIR / "scene_boundary_anchors_v2_4.csv",
    DATA_SCENE_DIR / "scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed.csv",
    DATA_SCENE_DIR / "scene_candidates_v2_4_merged_ffmpeg_fallback.csv",
    DATA_SCENE_DIR / "scene_candidate_boundary_audit_v2_4_merged_ffmpeg_fallback.csv",
]
MANIFEST_FILE = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
LABEL_INTERVAL_FILE = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_4.csv"
CONFIG_FILE = CONFIGS_DIR / "audio_persistence_rule_config_v2_4_train_only.json"
REFERENCE_FILES = {
    "feature_recommendations": DATA_AUDIO_DIR / "audio_rule_feature_recommendations_v2_4_train_only.csv",
    "candidate_thresholds": DATA_AUDIO_DIR / "audio_rule_candidate_thresholds_v2_4_train_only.csv",
    "validation_summary": DATA_AUDIO_DIR / "audio_rule_validation_summary_v2_4_train_only.csv",
    "split_aware_summary": REPORTS_AUDIO_DIR / "audio_split_aware_rule_reanalysis_v2_4.md",
    "labeled_features_with_split": DATA_AUDIO_DIR / "audio_labeled_segment_features_v2_4_with_split.csv",
    "edge_features_with_split": DATA_AUDIO_DIR / "audio_ad_edge_5s_10s_features_v2_4_with_split.csv",
    "boundary_persistence_with_split": DATA_AUDIO_DIR / "audio_ad_edge_boundary_persistence_scores_v2_4_with_split.csv",
    "subwindow_features_with_split": DATA_AUDIO_DIR / "audio_ad_edge_persistence_subwindow_features_v2_4_with_split.csv",
}
HELPER_SCRIPT = SCRIPTS_AUDIO_DIR / "extract_labeled_audio_features_v2_4.py"
PERSISTENCE_SCRIPT = SCRIPTS_AUDIO_DIR / "extract_audio_ad_edge_persistence_v2_4.py"

SAMPLE_RATE = 16000
SUBWINDOW_SIZE_SEC = 2.0
SUBWINDOW_STRIDE_SEC = 2.0
PERSISTENCE_WINDOW_SEC = 10.0
FORBIDDEN_SUFFIXES = {
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


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPER = load_module(HELPER_SCRIPT, "audio_feature_helper_v2_4")
PERSISTENCE = load_module(PERSISTENCE_SCRIPT, "audio_persistence_helper_v2_4")
FEATURE_COLUMNS = list(HELPER.FEATURE_COLUMNS)


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


def paths() -> Dict[str, Path]:
    return {
        "full_features": DATA_AUDIO_DIR / "audio_scene_anchor_persistence_features_v2_4.csv",
        "discussion_features": DATA_AUDIO_DIR / "audio_scene_anchor_persistence_features_v2_4_train_val_for_discussion.csv",
        "discussion_subwindows": DATA_AUDIO_DIR / "audio_scene_anchor_persistence_subwindow_features_v2_4_train_val_for_discussion.csv",
        "level_thresholds": DATA_AUDIO_DIR / "audio_scene_anchor_level_thresholds_v2_4_train_only.csv",
        "audit": DATA_AUDIO_DIR / "audio_scene_anchor_rule_discussion_audit_v2_4_train_val.csv",
        "discussion_table": DATA_FUSION_DIR / "scene_audio_anchor_rule_discussion_table_v2_4_train_val.csv",
        "report": REPORTS_AUDIO_DIR / "audio_scene_anchor_persistence_features_v2_4_report.json",
        "summary": REPORTS_AUDIO_DIR / "audio_scene_anchor_persistence_features_v2_4_summary.md",
        "run_log": LOGS_DIR / "audio_scene_anchor_persistence_features_v2_4_run_log.txt",
        "script": SCRIPTS_AUDIO_DIR / "audio_scene_anchor_persistence_features_v2_4.py",
        "bundle_readme": BUNDLE_DIR / "README_latest_files.md",
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


def mmss(seconds: Any) -> str:
    if seconds is None or pd.isna(seconds):
        return ""
    sec = max(0, int(round(float(seconds))))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


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
    if not OLD_PROJECT_ROOT.exists():
        return {"path": str(OLD_PROJECT_ROOT), "exists": False, "file_count": 0, "metadata_digest": None}
    import hashlib

    entries: List[str] = []
    file_count = 0
    for dirpath, _, filenames in os.walk(OLD_PROJECT_ROOT):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            file_count += 1
            rel = path.relative_to(OLD_PROJECT_ROOT)
            entries.append(f"{rel}|{stat.st_size}|{int(stat.st_mtime)}")
    digest = hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()
    return {"path": str(OLD_PROJECT_ROOT), "exists": True, "file_count": file_count, "metadata_digest": digest}


def file_stats(paths_to_check: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for path in paths_to_check:
        if path.exists():
            stat = path.stat()
            stats[str(path)] = {"exists": True, "size": stat.st_size, "mtime": stat.st_mtime}
        else:
            stats[str(path)] = {"exists": False, "size": None, "mtime": None}
    return stats


def stats_changed(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> List[str]:
    changed = []
    for path, stat_before in before.items():
        stat_after = after.get(path, {})
        if stat_before != stat_after:
            changed.append(path)
    return changed


def ensure_dirs() -> None:
    for directory in [DATA_AUDIO_DIR, DATA_FUSION_DIR, REPORTS_AUDIO_DIR, LOGS_DIR, SCRIPTS_AUDIO_DIR, BUNDLE_DIR, BACKUPS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def backup_existing(outputs: Sequence[Path], logger: TaskLogger) -> Optional[Path]:
    existing = [path for path in outputs if path.exists()]
    if not existing:
        return None
    backup_dir = BACKUPS_DIR / f"audio_scene_anchor_persistence_features_v2_4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        dest = backup_dir / path.relative_to(PROJECT_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    logger.log(f"Backed up {len(existing)} existing output files to {backup_dir}")
    return backup_dir


def executable_path(name: str) -> Optional[str]:
    path = shutil.which(name)
    if path:
        return path
    conda_candidate = Path(".venv/bin") / name
    if conda_candidate.exists():
        return str(conda_candidate)
    return None


def load_split(warnings: List[str]) -> pd.DataFrame:
    if not SPLIT_FILE.exists():
        rows = []
        for split, video_ids in FIXED_SPLIT.items():
            for video_id in video_ids:
                rows.append(
                    {
                        "version": VERSION,
                        "video_id": video_id,
                        "split": split,
                        "split_method": "video_id_deterministic_shuffle_no_ocr_text_or_quality_used",
                        "split_seed": SPLIT_SEED,
                    }
                )
        df = pd.DataFrame(rows)
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SPLIT_FILE, index=False, encoding="utf-8-sig")
        warnings.append("split file missing; generated from fixed user-provided split list")
        return df
    df = pd.read_csv(SPLIT_FILE)
    df["video_id"] = pd.to_numeric(df["video_id"], errors="coerce").astype("Int64")
    return df


def validate_split(split_df: pd.DataFrame, errors: List[str], warnings: List[str]) -> Dict[str, Any]:
    actual = {
        split: sorted(pd.to_numeric(split_df.loc[split_df["split"].eq(split), "video_id"], errors="coerce").dropna().astype(int).unique().tolist())
        for split in ["train", "validation", "test"]
    }
    for split, expected in FIXED_SPLIT.items():
        if actual.get(split) != sorted(expected):
            errors.append(f"split mismatch for {split}: expected={expected}, actual={actual.get(split)}")
    all_ids = [vid for values in actual.values() for vid in values]
    if len(all_ids) != len(set(all_ids)):
        errors.append("split video_id overlap detected")
    if "split_seed" in split_df.columns:
        seeds = set(pd.to_numeric(split_df["split_seed"], errors="coerce").dropna().astype(int).unique().tolist())
        if seeds and seeds != {SPLIT_SEED}:
            warnings.append(f"split_seed differs from expected {SPLIT_SEED}: {sorted(seeds)}")
    return actual


def discover_anchor_file(warnings: List[str]) -> Tuple[Path, bool, str]:
    if PREFERRED_ANCHOR_FILE.exists():
        return PREFERRED_ANCHOR_FILE, False, "preferred visual_scene_boundary_anchors_v2_4.csv exists"
    for candidate in ANCHOR_FALLBACKS:
        if candidate.exists():
            warnings.append(f"preferred visual anchor file missing; using fallback anchor source: {candidate}")
            return candidate, True, "preferred missing; fallback file with candidate_time_sec used"
    search_roots = [
        DATA_SCENE_DIR,
        DATA_REVIEW_DIR,
        PROJECT_ROOT / "outputs/latest_for_chatgpt",
        PROJECT_ROOT / "outputs/latest_for_chatgpt_a",
        PROJECT_ROOT / "outputs/latest_for_chatgpt_audio_scene_anchor",
        PROJECT_ROOT / "reports",
    ]
    patterns = ["visual_scene_boundary_anchors_", "scene_boundary_anchors_", "scene_candidates_v2_4", "scene_candidate_boundary_audit_v2_4"]
    candidates: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if any(pattern in path.name for pattern in patterns):
                candidates.append(path)
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        warnings.append(f"using latest fallback anchor candidate from search: {candidates[0]}")
        return candidates[0], True, "searched fallback candidate_time_sec source"
    raise FileNotFoundError("No scene anchor or candidate_time_sec fallback file found")


def load_anchor_file(anchor_file: Path, fallback_anchor_source: bool, errors: List[str]) -> pd.DataFrame:
    df = pd.read_csv(anchor_file)
    if "candidate_time_sec" not in df.columns:
        for alt in ["boundary_sec", "scene_time_sec", "time_sec", "timestamp_sec", "candidate_sec"]:
            if alt in df.columns:
                df["candidate_time_sec"] = pd.to_numeric(df[alt], errors="coerce")
                break
    if "candidate_time_sec" not in df.columns:
        errors.append("candidate_time_sec could not be recovered from scene anchor file")
        return df
    df["candidate_time_sec"] = pd.to_numeric(df["candidate_time_sec"], errors="coerce")
    df["video_id"] = pd.to_numeric(df.get("video_id"), errors="coerce").astype("Int64")
    if "candidate_id" not in df.columns:
        source_col = "merged_candidate_id" if "merged_candidate_id" in df.columns else None
        df["candidate_id"] = df[source_col].astype(str) if source_col else [f"scene_anchor_{idx + 1:06d}" for idx in range(len(df))]
    df["anchor_id"] = VERSION + "_" + df["video_id"].astype(str) + "_" + df["candidate_id"].astype(str)
    df["candidate_time_mmss"] = df["candidate_time_sec"].map(mmss)
    df["scene_anchor_type"] = "fallback_scene_candidate_anchor" if fallback_anchor_source else "visual_scene_boundary_anchor"
    if "scene_rank_in_video" not in df.columns:
        score = pd.to_numeric(df.get("scene_change_score", pd.Series(np.nan, index=df.index)), errors="coerce")
        df["scene_rank_in_video"] = score.groupby(df["video_id"]).rank(method="first", ascending=False).astype("Int64")
    for col in ["scene_component_score", "visual_start_like_score", "visual_end_like_score", "visual_internal_transition_hint", "is_high_priority"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_FILE)
    manifest["video_id"] = pd.to_numeric(manifest["video_id"], errors="coerce").astype("Int64")
    return manifest


def resolve_video_info(anchors: pd.DataFrame, manifest: pd.DataFrame, ffprobe_path: Optional[str], warnings: List[str]) -> pd.DataFrame:
    manifest_map = {int(row["video_id"]): row for _, row in manifest.dropna(subset=["video_id"]).iterrows()}
    rows = []
    for video_id in sorted(pd.to_numeric(anchors["video_id"], errors="coerce").dropna().astype(int).unique().tolist()):
        source = manifest_map.get(video_id)
        anchor_rows = anchors[anchors["video_id"].astype("Int64").eq(video_id)]
        video_path = ""
        video_title = ""
        duration = np.nan
        if source is not None:
            video_path = str(source.get("video_path", "") or "")
            video_title = str(source.get("video_title", source.get("video_name", "")) or "")
            duration = safe_float(source.get("duration_sec", source.get("video_duration_sec", np.nan)))
        if not video_path and "video_path" in anchor_rows.columns:
            video_path = str(anchor_rows["video_path"].dropna().astype(str).iloc[0]) if anchor_rows["video_path"].notna().any() else ""
        if not video_title and "video_title" in anchor_rows.columns and anchor_rows["video_title"].notna().any():
            video_title = str(anchor_rows["video_title"].dropna().astype(str).iloc[0])
        path_obj = Path(video_path)
        if video_path and not path_obj.is_absolute():
            path_obj = PROJECT_ROOT / video_path
        if (not video_path or not path_obj.exists()) and "video_filename" in anchor_rows.columns and anchor_rows["video_filename"].notna().any():
            filename = str(anchor_rows["video_filename"].dropna().astype(str).iloc[0])
            candidate = PROJECT_ROOT / "data/raw/videos" / filename
            if candidate.exists():
                path_obj = candidate
                video_path = str(candidate)
        if not path_obj.exists():
            warnings.append(f"video path missing for video_id={video_id}: {video_path}")
        probe = probe_video(ffprobe_path, path_obj) if ffprobe_path and path_obj.exists() else {}
        if np.isfinite(safe_float(probe.get("duration_sec"))):
            duration = safe_float(probe.get("duration_sec"))
        rows.append(
            {
                "video_id": video_id,
                "video_title_resolved": video_title,
                "video_path_resolved": str(path_obj) if video_path else "",
                "video_duration_sec": duration,
                "audio_available": bool(probe.get("audio_available", False)),
                "audio_stream_index": probe.get("audio_stream_index", np.nan),
                "probe_error": probe.get("probe_error", ""),
            }
        )
    return pd.DataFrame(rows)


def probe_video(ffprobe_path: Optional[str], video_path: Path) -> Dict[str, Any]:
    if not ffprobe_path:
        return {"audio_available": False, "probe_error": "ffprobe missing"}
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        return {"audio_available": False, "probe_error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"audio_available": False, "probe_error": proc.stderr.strip() or f"ffprobe_exit_{proc.returncode}"}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"audio_available": False, "probe_error": f"json_decode_failed: {exc}"}
    duration = safe_float(data.get("format", {}).get("duration"))
    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    stream_index = audio_streams[0].get("index") if audio_streams else np.nan
    return {
        "duration_sec": duration,
        "audio_available": bool(audio_streams),
        "audio_stream_index": stream_index,
        "probe_error": "",
    }


def build_anchor_table(anchor_df: pd.DataFrame, split_df: pd.DataFrame, video_info: pd.DataFrame) -> pd.DataFrame:
    split_map = {
        int(row["video_id"]): row["split"]
        for _, row in split_df.dropna(subset=["video_id"]).iterrows()
    }
    video_map = {int(row["video_id"]): row for _, row in video_info.iterrows()}
    records = []
    preserve_cols = [
        "version",
        "video_id",
        "candidate_id",
        "anchor_id",
        "candidate_time_sec",
        "candidate_time_mmss",
        "candidate_source",
        "scene_anchor_type",
        "scene_change_score",
        "scene_component_score",
        "scene_rank_in_video",
        "method_used",
        "source_file",
        "is_high_priority",
        "visual_start_like_score",
        "visual_end_like_score",
        "visual_internal_transition_hint",
        "video_title",
        "video_filename",
        "video_path",
        "nearest_ad_boundary_type",
        "nearest_ad_boundary_sec",
        "distance_to_nearest_ad_boundary_sec",
    ]
    for _, row in anchor_df.iterrows():
        video_id = int(row["video_id"]) if pd.notna(row["video_id"]) else -1
        info = video_map.get(video_id, {})
        record: Dict[str, Any] = {col: row.get(col, np.nan) for col in preserve_cols}
        record["version"] = VERSION
        record["video_id"] = video_id
        record["split"] = split_map.get(video_id, "unknown")
        record["video_title"] = row.get("video_title", info.get("video_title_resolved", ""))
        record["video_path"] = info.get("video_path_resolved", row.get("video_path", ""))
        record["video_duration_sec"] = info.get("video_duration_sec", np.nan)
        record["audio_available"] = info.get("audio_available", False)
        record["audio_stream_index"] = info.get("audio_stream_index", np.nan)
        record["probe_error"] = info.get("probe_error", "")
        records.append(record)
    return pd.DataFrame(records)


def build_subwindow_rows(anchor_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, anchor in anchor_table.iterrows():
        t = safe_float(anchor["candidate_time_sec"])
        duration = safe_float(anchor.get("video_duration_sec"))
        if not np.isfinite(t) or not np.isfinite(duration):
            continue
        contexts = [
            ("pre_10s", max(0.0, t - PERSISTENCE_WINDOW_SEC), min(t, duration), -PERSISTENCE_WINDOW_SEC, 0.0),
            ("post_10s", max(0.0, t), min(t + PERSISTENCE_WINDOW_SEC, duration), 0.0, PERSISTENCE_WINDOW_SEC),
        ]
        for context_type, context_start, context_end, rel_start_base, _ in contexts:
            cursor = context_start
            idx = 0
            while cursor < context_end - 1e-6 and idx < 5:
                end = min(cursor + SUBWINDOW_SIZE_SEC, context_end)
                sub_duration = end - cursor
                status = "planned" if sub_duration > 0 else "invalid_subwindow"
                warning = ""
                if sub_duration < SUBWINDOW_SIZE_SEC - 1e-6:
                    warning = "context_truncated_or_short_subwindow"
                rows.append(
                    {
                        "subwindow_id": f"{anchor['anchor_id']}_{context_type}_{idx:02d}",
                        "version": VERSION,
                        "split": anchor["split"],
                        "video_id": anchor["video_id"],
                        "candidate_id": anchor["candidate_id"],
                        "anchor_id": anchor["anchor_id"],
                        "candidate_time_sec": t,
                        "candidate_time_mmss": mmss(t),
                        "context_type": context_type,
                        "subwindow_index": idx,
                        "subwindow_start_sec": cursor,
                        "subwindow_end_sec": end,
                        "subwindow_duration_sec": sub_duration,
                        "relative_start_to_anchor_sec": cursor - t,
                        "relative_end_to_anchor_sec": end - t,
                        "video_path": anchor["video_path"],
                        "video_duration_sec": duration,
                        "audio_available": anchor.get("audio_available", False),
                        "audio_stream_index": anchor.get("audio_stream_index", np.nan),
                        "sampling_status": status,
                        "sampling_warning": warning,
                    }
                )
                cursor += SUBWINDOW_STRIDE_SEC
                idx += 1
    return pd.DataFrame(rows)


def add_empty_features(record: Dict[str, Any]) -> Dict[str, Any]:
    record.update(
        {
            "feature_status": "not_run",
            "feature_error": "",
            "decoded_sample_rate": np.nan,
            "decoded_num_samples": 0,
            "decoded_duration_sec": np.nan,
            "decoded_duration_diff_sec": np.nan,
        }
    )
    for col in FEATURE_COLUMNS:
        record[col] = np.nan
    return record


def extract_subwindow_features_from_contexts(
    subwindow_plan: pd.DataFrame,
    ffmpeg_path: Optional[str],
    logger: TaskLogger,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    grouped = subwindow_plan.groupby(["anchor_id", "context_type"], sort=False, dropna=False)
    total_groups = len(grouped)
    processed_groups = 0
    for (anchor_id, context_type), group in grouped:
        processed_groups += 1
        group_sorted = group.sort_values("subwindow_index")
        context_start = safe_float(group_sorted["subwindow_start_sec"].min())
        context_end = safe_float(group_sorted["subwindow_end_sec"].max())
        context_duration = context_end - context_start
        video_path = str(group_sorted["video_path"].iloc[0])
        can_decode = (
            ffmpeg_path is not None
            and Path(video_path).exists()
            and bool(group_sorted["audio_available"].iloc[0])
            and np.isfinite(context_start)
            and np.isfinite(context_duration)
            and context_duration > 0
        )
        audio = np.array([], dtype=np.float32)
        sr = SAMPLE_RATE
        decode_error = ""
        if can_decode:
            audio, sr, decode_error = HELPER.decode_audio_segment(ffmpeg_path, video_path, context_start, context_duration)
        elif ffmpeg_path is None:
            decode_error = "ffmpeg missing"
        elif not Path(video_path).exists():
            decode_error = "video path missing"
        elif not bool(group_sorted["audio_available"].iloc[0]):
            decode_error = "audio stream unavailable"
        else:
            decode_error = "invalid context duration"
        for _, row in group_sorted.iterrows():
            record = add_empty_features(row.to_dict())
            if row.get("sampling_status") != "planned":
                record["feature_status"] = row.get("sampling_status", "sampling_not_planned")
                record["feature_error"] = row.get("sampling_warning", "")
                records.append(record)
                continue
            if decode_error or audio.size == 0:
                record["feature_status"] = "decode_failed"
                record["feature_error"] = decode_error or "empty decoded context"
                records.append(record)
                continue
            start_offset = max(0, int(round((safe_float(row["subwindow_start_sec"]) - context_start) * sr)))
            end_offset = max(start_offset, int(round((safe_float(row["subwindow_end_sec"]) - context_start) * sr)))
            clip = audio[start_offset:min(end_offset, audio.size)]
            if clip.size == 0:
                record["feature_status"] = "empty_subwindow_audio"
                record["feature_error"] = "subwindow slice empty after context decode"
                records.append(record)
                continue
            decoded_duration = clip.size / float(sr)
            record["decoded_sample_rate"] = sr
            record["decoded_num_samples"] = int(clip.size)
            record["decoded_duration_sec"] = decoded_duration
            record["decoded_duration_diff_sec"] = decoded_duration - safe_float(row["subwindow_duration_sec"])
            try:
                record.update(HELPER.compute_audio_features(clip, sr))
                record["feature_status"] = "success"
            except Exception as exc:
                record["feature_status"] = "feature_compute_failed"
                record["feature_error"] = f"{type(exc).__name__}: {exc}"
            records.append(record)
        if processed_groups % 100 == 0:
            logger.log(f"[STEP 08] Extracted scene-anchor audio contexts {processed_groups}/{total_groups}")
    features = pd.DataFrame(records)
    duration = pd.to_numeric(features["subwindow_duration_sec"], errors="coerce").replace(0, np.nan)
    onset = pd.to_numeric(features["onset_count"], errors="coerce")
    features["onset_density"] = onset / duration
    features["onset_count_per_sec"] = onset / duration
    features["spectral_flux_mean_per_sec_proxy"] = pd.to_numeric(features["spectral_flux_mean"], errors="coerce")
    features["onset_strength_mean_per_sec_proxy"] = pd.to_numeric(features["onset_strength_mean"], errors="coerce")
    return features


def load_train_baseline() -> pd.DataFrame:
    baseline_path = REFERENCE_FILES["subwindow_features_with_split"]
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        if "split" in df.columns:
            df = df[df["split"].eq("train")].copy()
        if "onset_density" not in df.columns:
            duration = pd.to_numeric(df.get("subwindow_duration_sec", df.get("segment_duration_sec")), errors="coerce").replace(0, np.nan)
            df["onset_density"] = pd.to_numeric(df.get("onset_count"), errors="coerce") / duration
        if "onset_count_per_sec" not in df.columns:
            df["onset_count_per_sec"] = df["onset_density"]
        return df
    fallback_path = REFERENCE_FILES["labeled_features_with_split"]
    df = pd.read_csv(fallback_path)
    df = df[df["split"].eq("train")].copy() if "split" in df.columns else df
    duration = pd.to_numeric(df.get("segment_duration_sec"), errors="coerce").replace(0, np.nan)
    df["onset_density"] = pd.to_numeric(df.get("onset_count"), errors="coerce") / duration
    df["onset_count_per_sec"] = df["onset_density"]
    return df


def add_audio_scores(subwindow_features: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    scored = PERSISTENCE.add_ad_like_scores(subwindow_features, baseline_df)
    return scored


def max_consecutive_true(flags: Sequence[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize_contexts(scored_subwindows: pd.DataFrame, ad_like_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (anchor_id, context_type), group in scored_subwindows.groupby(["anchor_id", "context_type"], dropna=False):
        scores = pd.to_numeric(group["audio_ad_like_score"], errors="coerce")
        valid = scores[np.isfinite(scores)]
        flags = (scores >= ad_like_threshold).fillna(False).astype(bool).tolist()
        if len(valid):
            q25 = valid.quantile(0.25)
            q75 = valid.quantile(0.75)
            median = valid.median()
            std = valid.std(ddof=1) if len(valid) > 1 else 0.0
            consistency = 1.0 - min(1.0, safe_float(std) / max(abs(safe_float(median)), 1e-6))
        else:
            q25 = q75 = median = std = consistency = np.nan
        rows.append(
            {
                "anchor_id": anchor_id,
                "context_type": context_type,
                "subwindow_count": int(len(group)),
                "valid_subwindow_count": int(len(valid)),
                "score_mean": safe_float(valid.mean()) if len(valid) else np.nan,
                "score_median": safe_float(median),
                "score_p25": safe_float(q25),
                "score_p75": safe_float(q75),
                "score_min": safe_float(valid.min()) if len(valid) else np.nan,
                "score_max": safe_float(valid.max()) if len(valid) else np.nan,
                "score_std": safe_float(std),
                "ad_like_ratio": safe_float(sum(flags) / len(flags)) if flags else np.nan,
                "max_consecutive_ad_like_sec": safe_float(max_consecutive_true(flags) * SUBWINDOW_SIZE_SEC),
                "score_consistency": safe_float(consistency),
            }
        )
    return pd.DataFrame(rows)


def context_record(contexts: pd.DataFrame, anchor_id: str, context_type: str) -> Dict[str, Any]:
    row = contexts[(contexts["anchor_id"].eq(anchor_id)) & (contexts["context_type"].eq(context_type))]
    if row.empty:
        return {
            "subwindow_count": 0,
            "valid_subwindow_count": 0,
            "score_mean": np.nan,
            "score_median": np.nan,
            "score_p25": np.nan,
            "score_p75": np.nan,
            "score_min": np.nan,
            "score_max": np.nan,
            "score_std": np.nan,
            "ad_like_ratio": np.nan,
            "max_consecutive_ad_like_sec": np.nan,
            "score_consistency": np.nan,
        }
    return row.iloc[0].to_dict()


def combine_anchor_features(anchor_table: pd.DataFrame, context_scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, anchor in anchor_table.iterrows():
        pre = context_record(context_scores, str(anchor["anchor_id"]), "pre_10s")
        post = context_record(context_scores, str(anchor["anchor_id"]), "post_10s")
        pre_med = safe_float(pre["score_median"])
        post_med = safe_float(post["score_median"])
        pre_ratio = safe_float(pre["ad_like_ratio"])
        post_ratio = safe_float(post["ad_like_ratio"])
        pre_consec = safe_float(pre["max_consecutive_ad_like_sec"])
        post_consec = safe_float(post["max_consecutive_ad_like_sec"])
        start_delta = post_med - pre_med if np.isfinite(post_med) and np.isfinite(pre_med) else np.nan
        end_delta = pre_med - post_med if np.isfinite(post_med) and np.isfinite(pre_med) else np.nan
        before_context_score = (
            0.50 * pre_med + 0.30 * pre_ratio + 0.20 * min(pre_consec / PERSISTENCE_WINDOW_SEC, 1.0)
            if all(np.isfinite(x) for x in [pre_med, pre_ratio, pre_consec])
            else np.nan
        )
        after_context_score = (
            0.50 * post_med + 0.30 * post_ratio + 0.20 * min(post_consec / PERSISTENCE_WINDOW_SEC, 1.0)
            if all(np.isfinite(x) for x in [post_med, post_ratio, post_consec])
            else np.nan
        )
        start_signal_score = (
            0.30 * post_ratio
            + 0.25 * min(post_consec / PERSISTENCE_WINDOW_SEC, 1.0)
            + 0.20 * post_med
            + 0.15 * safe_float(post["score_p25"])
            + 0.10 * max(start_delta, 0.0)
            if all(np.isfinite(x) for x in [post_ratio, post_consec, post_med, safe_float(post["score_p25"])])
            else np.nan
        )
        end_signal_score = (
            0.30 * pre_ratio
            + 0.25 * min(pre_consec / PERSISTENCE_WINDOW_SEC, 1.0)
            + 0.20 * pre_med
            + 0.15 * safe_float(pre["score_p25"])
            + 0.10 * max(end_delta, 0.0)
            if all(np.isfinite(x) for x in [pre_ratio, pre_consec, pre_med, safe_float(pre["score_p25"])])
            else np.nan
        )
        context_score = np.nanmax([before_context_score, after_context_score]) if np.isfinite(before_context_score) or np.isfinite(after_context_score) else np.nan
        internal_hint = bool(
            np.isfinite(before_context_score)
            and np.isfinite(after_context_score)
            and before_context_score >= 0.45
            and after_context_score >= 0.45
        )
        record = anchor.to_dict()
        for prefix, ctx in [("audio_pre_10s", pre), ("audio_post_10s", post)]:
            record[f"{prefix}_subwindow_count"] = ctx["subwindow_count"]
            record[f"{prefix}_valid_subwindow_count"] = ctx["valid_subwindow_count"]
            record[f"{prefix}_score_mean"] = ctx["score_mean"]
            record[f"{prefix}_score_median"] = ctx["score_median"]
            record[f"{prefix}_score_p25"] = ctx["score_p25"]
            record[f"{prefix}_score_p75"] = ctx["score_p75"]
            record[f"{prefix}_score_min"] = ctx["score_min"]
            record[f"{prefix}_score_max"] = ctx["score_max"]
            record[f"{prefix}_score_std"] = ctx["score_std"]
            record[f"{prefix}_ad_like_ratio"] = ctx["ad_like_ratio"]
            record[f"{prefix}_max_consecutive_ad_like_sec"] = ctx["max_consecutive_ad_like_sec"]
            record[f"{prefix}_score_consistency"] = ctx["score_consistency"]
        record["audio_score_delta_post_minus_pre"] = safe_float(start_delta)
        record["audio_score_delta_pre_minus_post"] = safe_float(end_delta)
        record["audio_ad_like_ratio_delta_post_minus_pre"] = safe_float(post_ratio - pre_ratio) if np.isfinite(post_ratio) and np.isfinite(pre_ratio) else np.nan
        record["audio_ad_like_ratio_delta_pre_minus_post"] = safe_float(pre_ratio - post_ratio) if np.isfinite(post_ratio) and np.isfinite(pre_ratio) else np.nan
        record["audio_start_signal_score"] = safe_float(start_signal_score)
        record["audio_end_signal_score"] = safe_float(end_signal_score)
        record["audio_before_context_score"] = safe_float(before_context_score)
        record["audio_after_context_score"] = safe_float(after_context_score)
        record["audio_context_score"] = safe_float(context_score)
        record["audio_internal_ad_transition_hint"] = internal_hint
        rows.append(record)
    return pd.DataFrame(rows)


def percentile_against_train(value: float, train_values: pd.Series) -> float:
    finite = pd.to_numeric(train_values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if not np.isfinite(value) or len(finite) == 0:
        return np.nan
    return safe_float((finite <= value).mean())


def level_from_percentile(percentile: float) -> str:
    if not np.isfinite(percentile):
        return "low"
    if percentile >= 0.70:
        return "high"
    if percentile >= 0.40:
        return "medium"
    return "low"


def add_levels(features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = features.copy()
    score_cols = {
        "audio_start_signal": "audio_start_signal_score",
        "audio_end_signal": "audio_end_signal_score",
        "audio_context": "audio_context_score",
        "audio_before_context": "audio_before_context_score",
        "audio_after_context": "audio_after_context_score",
    }
    train = out[out["split"].eq("train")]
    threshold_rows = []
    for prefix, score_col in score_cols.items():
        train_values = pd.to_numeric(train[score_col], errors="coerce")
        finite = train_values[np.isfinite(train_values)]
        q40 = safe_float(finite.quantile(0.40)) if len(finite) else np.nan
        q70 = safe_float(finite.quantile(0.70)) if len(finite) else np.nan
        out[f"{prefix}_train_percentile"] = out[score_col].map(lambda v, vals=train_values: percentile_against_train(safe_float(v), vals))
        out[f"{prefix}_level_by_train_quantile"] = out[f"{prefix}_train_percentile"].map(level_from_percentile)
        threshold_rows.append(
            {
                "version": VERSION,
                "split_basis": "train_only",
                "split_seed": SPLIT_SEED,
                "score_name": score_col,
                "train_count": int(len(finite)),
                "train_q40_medium_min": q40,
                "train_q70_high_min": q70,
                "level_rule": "high if train_percentile >= 0.70; medium if >= 0.40; else low",
                "leakage_guard_note": "validation/test not used for level threshold calculation",
            }
        )
    out["audio_start_signal_level"] = out["audio_start_signal_level_by_train_quantile"]
    out["audio_end_signal_level"] = out["audio_end_signal_level_by_train_quantile"]
    out["audio_before_context_level"] = out["audio_before_context_level_by_train_quantile"]
    out["audio_after_context_level"] = out["audio_after_context_level_by_train_quantile"]
    out["audio_context_level"] = out["audio_context_level_by_train_quantile"]
    out["audio_level_source"] = "train_only_config_and_train_anchor_quantile"
    return out, pd.DataFrame(threshold_rows)


def build_label_audit(features: pd.DataFrame, label_file: Path) -> pd.DataFrame:
    intervals = pd.read_csv(label_file)
    intervals["video_id"] = pd.to_numeric(intervals["video_id"], errors="coerce").astype("Int64")
    rows: List[Dict[str, Any]] = []
    for _, anchor in features[features["split"].isin(["train", "validation"])].iterrows():
        video_id = int(anchor["video_id"])
        t = safe_float(anchor["candidate_time_sec"])
        video_intervals = intervals[intervals["video_id"].eq(video_id)]
        nearest_type = ""
        nearest_sec = np.nan
        distance = np.nan
        best = None
        for _, interval in video_intervals.iterrows():
            for boundary_type, sec_col in [("ad_start", "ad_start_sec"), ("ad_end", "ad_end_sec")]:
                sec = safe_float(interval.get(sec_col))
                if not np.isfinite(sec):
                    continue
                dist = abs(t - sec)
                if best is None or dist < best[0]:
                    best = (dist, boundary_type, sec)
        if best is not None:
            distance, nearest_type, nearest_sec = best
        rows.append(
            {
                "version": VERSION,
                "split": anchor["split"],
                "video_id": video_id,
                "candidate_id": anchor["candidate_id"],
                "anchor_id": anchor["anchor_id"],
                "candidate_time_sec": t,
                "candidate_time_mmss": anchor["candidate_time_mmss"],
                "nearest_true_boundary_sec": safe_float(nearest_sec),
                "nearest_true_boundary_type": nearest_type,
                "distance_to_nearest_true_boundary_sec": safe_float(distance),
                "is_near_true_boundary_2s": bool(np.isfinite(distance) and distance <= 2),
                "is_near_true_boundary_5s": bool(np.isfinite(distance) and distance <= 5),
                "is_near_true_boundary_10s": bool(np.isfinite(distance) and distance <= 10),
                "audio_start_signal_score": anchor["audio_start_signal_score"],
                "audio_start_signal_level": anchor["audio_start_signal_level"],
                "audio_end_signal_score": anchor["audio_end_signal_score"],
                "audio_end_signal_level": anchor["audio_end_signal_level"],
                "audio_context_score": anchor["audio_context_score"],
                "audio_context_level": anchor["audio_context_level"],
                "audit_note": "label audit only; labels were not used for audio score or level calculation",
            }
        )
    return pd.DataFrame(rows)


def build_discussion_table(features: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    discussion = features[features["split"].isin(["train", "validation"])].copy()
    audit_cols = [
        "anchor_id",
        "nearest_true_boundary_sec",
        "nearest_true_boundary_type",
        "distance_to_nearest_true_boundary_sec",
        "is_near_true_boundary_2s",
        "is_near_true_boundary_5s",
        "is_near_true_boundary_10s",
    ]
    compact_cols = [
        "version",
        "split",
        "video_id",
        "candidate_id",
        "anchor_id",
        "candidate_time_sec",
        "candidate_time_mmss",
        "candidate_source",
        "scene_anchor_type",
        "scene_change_score",
        "scene_rank_in_video",
        "audio_pre_10s_score_median",
        "audio_post_10s_score_median",
        "audio_pre_10s_ad_like_ratio",
        "audio_post_10s_ad_like_ratio",
        "audio_pre_10s_max_consecutive_ad_like_sec",
        "audio_post_10s_max_consecutive_ad_like_sec",
        "audio_score_delta_post_minus_pre",
        "audio_score_delta_pre_minus_post",
        "audio_start_signal_score",
        "audio_start_signal_level",
        "audio_end_signal_score",
        "audio_end_signal_level",
        "audio_before_context_score",
        "audio_before_context_level",
        "audio_after_context_score",
        "audio_after_context_level",
        "audio_context_score",
        "audio_context_level",
        "audio_internal_ad_transition_hint",
    ]
    table = discussion[[col for col in compact_cols if col in discussion.columns]].merge(
        audit[[col for col in audit_cols if col in audit.columns]], on="anchor_id", how="left"
    )
    table["discussion_note"] = "scene anchor audio evidence only; OCR score not included and rule is not finalized"
    return table


def forbidden_files_in_bundle() -> List[str]:
    if not BUNDLE_DIR.exists():
        return []
    forbidden = []
    for path in BUNDLE_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(str(path))
        if any(part in {"cache", "tmp", "raw", "videos"} for part in path.parts):
            forbidden.append(str(path))
    return sorted(set(forbidden))


def copy_bundle(files: Dict[str, Path], logger: TaskLogger, warnings: List[str]) -> List[str]:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for path in BUNDLE_DIR.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
    copy_keys = [
        "discussion_features",
        "discussion_subwindows",
        "level_thresholds",
        "audit",
        "discussion_table",
        "summary",
        "report",
        "run_log",
        "script",
    ]
    reference_copy = [
        CONFIG_FILE,
        REFERENCE_FILES["feature_recommendations"],
        REFERENCE_FILES["validation_summary"],
    ]
    copied: List[str] = []
    skipped: List[str] = []
    for key in copy_keys:
        src = files[key]
        if not src.exists():
            continue
        if key == "discussion_subwindows" and src.stat().st_size > 5_000_000:
            skipped.append(str(src))
            warnings.append(f"large discussion subwindow file not copied to bundle: {src}")
            continue
        dest = BUNDLE_DIR / src.name
        shutil.copy2(src, dest)
        copied.append(str(dest))
    for src in reference_copy:
        if src.exists():
            dest = BUNDLE_DIR / src.name
            shutil.copy2(src, dest)
            copied.append(str(dest))
    readme = files["bundle_readme"]
    readme.write_text(
        "\n".join(
            [
                "# latest files: audio scene anchor evidence pack",
                "",
                f"- task: {TASK_NAME}",
                f"- version: {VERSION}",
                f"- split_seed: {SPLIT_SEED}",
                f"- train video_id: {FIXED_SPLIT['train']}",
                f"- validation video_id: {FIXED_SPLIT['validation']}",
                f"- test video_id: {FIXED_SPLIT['test']}",
                "",
                "This bundle contains train/validation row-level scene anchor audio evidence for discussion.",
                "It does not contain test row-level feature files.",
                "This is not a final rule or final interval detector output.",
                "",
                "## Copied files",
                *[f"- {Path(path).name}" for path in copied],
                "",
                "## Large files kept outside bundle",
                *([f"- {path}" for path in skipped] if skipped else ["- none"]),
                "",
                "## Safety",
                "- forbidden media/model/cache files: not included",
                "- OCR score: not generated in this task",
                "- old_project_modified: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    copied.append(str(readme))
    logger.log(f"Copied {len(copied)} files to {BUNDLE_DIR}")
    return copied


def write_summary(path: Path, report: Dict[str, Any], top_counts: Dict[str, Any]) -> None:
    lines = [
        "# Scene Anchor Audio Persistence Evidence Pack v2_4",
        "",
        "## 1. 작업 개요",
        "이번 작업은 visual scene anchor의 candidate_time_sec를 기준으로 pre_10s/post_10s audio persistence feature를 생성한 audio evidence pack이다.",
        "label-aligned audio feature가 아니라 detector inference 구조에 맞춘 scene anchor 기준 feature이며, rule 확정이나 scene/audio/OCR fusion 확정 작업이 아니다.",
        "",
        "## 2. 입력 파일",
        f"- scene anchor file: {report['anchor_file']}",
        f"- split file: {report['split_file']}",
        f"- train-only audio config: {report['config_file']}",
        "- reference files:",
        *[f"  - {path}" for path in report.get("reference_files", [])],
        "",
        "## 3. Split 적용",
        f"- train video_id: {report['train_video_ids']}",
        f"- validation video_id: {report['validation_video_ids']}",
        f"- test video_id: {report['test_video_ids']}",
        f"- train/validation/test anchor count: {report['train_anchor_count']} / {report['validation_anchor_count']} / {report['test_anchor_count']}",
        "- discussion bundle에는 train/validation row만 포함했다.",
        "- test row-level feature는 discussion bundle에서 제외했다.",
        "",
        "## 4. 생성 Feature 설명",
        "- pre_10s: anchor 이전 10초를 2초 subwindow로 분할한 context.",
        "- post_10s: anchor 이후 10초를 2초 subwindow로 분할한 context.",
        "- ad_like_ratio: context 안에서 train-only config threshold 이상인 subwindow 비율.",
        "- max_consecutive_ad_like_sec: ad-like subwindow가 연속으로 지속된 최대 초 수.",
        "- start/end signal score: post 증가 또는 pre 감소 방향성을 보는 discussion용 audio cue.",
        "- before/after/context score: anchor 앞뒤 audio context가 광고스럽게 유지되는지 보는 보조 score.",
        "- qualitative level: train-only anchor score percentile 기준 high/medium/low discussion candidate.",
        "",
        "## 5. Rule 논의에 사용할 관찰 포인트",
        "- anchor 이후 10초 score 상승: start 후보 보조 근거로 볼 수 있다.",
        "- anchor 이전 10초가 높고 이후 하락: end 후보 보조 근거로 볼 수 있다.",
        "- before/after 둘 다 높음: 광고 내부 scene transition 가능성을 논의할 수 있다.",
        "- after low가 지속됨: end 이후 일반 context 가능성을 논의할 수 있다.",
        "",
        "## 6. 한계",
        "- audio 단독 판단 금지.",
        "- OCR/scene과 결합 전 최종 결정 금지.",
        "- qualitative level은 train-only 기반 discussion candidate다.",
        "- validation은 audit/discussion only이며 level threshold 계산에 쓰지 않았다.",
        "- test는 최종 평가 전까지 보호하며 bundle에 row-level feature를 넣지 않았다.",
        "",
        "## 7. 생성 파일 목록",
        *[f"- {path}: {desc}" for path, desc in report.get("output_file_descriptions", {}).items()],
        "",
        "## 8. Sub Agent 검증 결과",
        *[f"- {name}: {result.get('status')} ({'; '.join(result.get('warnings', []) + result.get('errors', [])) or 'ok'})" for name, result in report.get("sub_agent_results", {}).items()],
        "",
        "## 9. 다음 단계 제안",
        "- OCR scene anchor context feature가 생성되면 같은 anchor_id/candidate_time_sec 기준으로 join한다.",
        "- scene + audio + OCR qualitative context score를 비교한다.",
        "- state-machine interval detector rule은 이 evidence pack을 보고 별도 논의한다.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_validations(report: Dict[str, Any], features: pd.DataFrame, subwindows: pd.DataFrame, bundle_files: List[str]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    errors: List[str] = []
    if report["split_file"] != str(SPLIT_FILE):
        errors.append("unexpected split file")
    if report["train_video_ids"] != FIXED_SPLIT["train"] or report["validation_video_ids"] != FIXED_SPLIT["validation"] or report["test_video_ids"] != FIXED_SPLIT["test"]:
        errors.append("fixed split lists do not match")
    if not Path(report["anchor_file"]).exists():
        errors.append("anchor file missing")
    if "candidate_time_sec" not in report.get("candidate_time_sec_column", ""):
        errors.append("candidate_time_sec not recovered")
    if report.get("unknown_split_anchor_count", 0) > 0:
        warnings.append("some anchors have unknown split")
    results["input_split_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = []
    errors = []
    if report.get("feature_success_count", 0) == 0:
        errors.append("no subwindow feature extraction successes")
    if "onset_density" not in subwindows.columns:
        errors.append("onset_density missing")
    score = pd.to_numeric(subwindows.get("audio_ad_like_score"), errors="coerce")
    if len(score.dropna()) and ((score.dropna() < 0).any() or (score.dropna() > 1).any()):
        errors.append("audio_ad_like_score outside 0-1 range")
    if report.get("feature_failed_count", 0) > 0:
        warnings.append("some subwindow feature extraction rows failed")
    results["audio_feature_extraction_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = []
    errors = []
    if not report.get("train_used_for_level_thresholds"):
        errors.append("train not used for level thresholds")
    if report.get("validation_used_for_level_thresholds") or report.get("test_used_for_level_thresholds"):
        errors.append("validation/test used for level thresholds")
    bundle_paths = [Path(path) for path in bundle_files]
    if any("test" in path.name and "row" in path.name for path in bundle_paths):
        errors.append("possible test row-level file copied")
    if report.get("test_row_level_features_copied_to_bundle"):
        errors.append("test row-level features copied to bundle")
    if report.get("label_columns_used_for_score"):
        errors.append("label/audit columns used for score")
    results["leakage_guard_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = []
    errors = []
    required_cols = [
        "audio_start_signal_score",
        "audio_end_signal_score",
        "audio_context_score",
        "audio_start_signal_level",
        "audio_end_signal_level",
        "audio_context_level",
    ]
    discussion = features[features["split"].isin(["train", "validation"])]
    if discussion.empty:
        errors.append("discussion table is empty")
    for col in required_cols:
        if col not in features.columns:
            errors.append(f"missing required discussion score column: {col}")
    level_counts = discussion["audio_context_level"].value_counts(dropna=False).to_dict() if "audio_context_level" in discussion else {}
    if len(level_counts) <= 1:
        warnings.append("audio_context_level distribution is concentrated in one level")
    if report.get("validation_anchor_count", 0) < 20:
        warnings.append("validation anchor count is modest")
    results["discussion_file_quality_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = []
    errors = []
    missing_outputs = [path for path in report.get("output_files", []) if not Path(path).exists()]
    if missing_outputs:
        errors.append(f"missing output files: {missing_outputs}")
    if report.get("latest_for_chatgpt_forbidden_files_found"):
        errors.append("forbidden files found in latest bundle")
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
    output_candidates = list(out.values())
    backup_dir = backup_existing(output_candidates, logger)

    start_time = now_iso()
    logger.log("[STEP 02] Load and validate v2_4 video split")
    split_df = load_split(warnings)
    split_lists = validate_split(split_df, errors, warnings)

    logger.log("[STEP 03] Load scene anchor file")
    anchor_file, fallback_anchor_source, anchor_reason = discover_anchor_file(warnings)
    anchor_df = load_anchor_file(anchor_file, fallback_anchor_source, errors)
    logger.log(f"Selected anchor file: {anchor_file} ({anchor_reason})")
    logger.log(f"Anchor row count: {len(anchor_df)}")

    logger.log("[STEP 04] Load video manifest and resolve raw video paths")
    manifest = load_manifest()
    ffmpeg_path = executable_path("ffmpeg")
    ffprobe_path = executable_path("ffprobe")
    video_info = resolve_video_info(anchor_df, manifest, ffprobe_path, warnings)
    missing_video_count = int((~video_info["video_path_resolved"].map(lambda p: Path(str(p)).exists())).sum())
    logger.log(f"ffmpeg: {ffmpeg_path or 'missing'}")
    logger.log(f"ffprobe: {ffprobe_path or 'missing'}")
    logger.log(f"Missing video count: {missing_video_count}")

    logger.log("[STEP 05] Load train-only audio config and reference files")
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    ad_like_threshold = safe_float(
        config.get("start_boundary_rule", {}).get("post_10s_median_score_min", np.nan)
    )
    if not np.isfinite(ad_like_threshold):
        ad_like_threshold = 0.6
        warnings.append("audio_ad_like_threshold missing in config; fallback 0.6 used")
    input_paths = [SPLIT_FILE, anchor_file, MANIFEST_FILE, CONFIG_FILE, LABEL_INTERVAL_FILE]
    input_paths.extend(path for path in REFERENCE_FILES.values() if path.exists())
    input_stats_before = file_stats(input_paths)

    logger.log("[STEP 06] Build scene anchor table and pre/post subwindow plan")
    anchor_table = build_anchor_table(anchor_df, split_df, video_info)
    subwindow_plan = build_subwindow_rows(anchor_table)
    logger.log(f"Subwindow plan row count: {len(subwindow_plan)}")

    logger.log("[STEP 07] Extract scene-anchor subwindow audio features")
    subwindow_features = extract_subwindow_features_from_contexts(subwindow_plan, ffmpeg_path, logger)
    feature_success = int((subwindow_features["feature_status"] == "success").sum())
    feature_failed = int(len(subwindow_features) - feature_success)
    logger.log(f"Subwindow feature success/failure: {feature_success}/{feature_failed}")

    logger.log("[STEP 08] Compute audio_ad_like_score using train-only config baseline")
    baseline_df = load_train_baseline()
    scored_subwindows = add_audio_scores(subwindow_features, baseline_df)
    scored_subwindows["audio_ad_like_threshold"] = ad_like_threshold
    score_min = safe_float(pd.to_numeric(scored_subwindows["audio_ad_like_score"], errors="coerce").min())
    score_max = safe_float(pd.to_numeric(scored_subwindows["audio_ad_like_score"], errors="coerce").max())
    logger.log(f"audio_ad_like_score range: {score_min:.3f} - {score_max:.3f}")

    logger.log("[STEP 09] Compute persistence metrics and anchor-level signal scores")
    context_scores = summarize_contexts(scored_subwindows, ad_like_threshold)
    features = combine_anchor_features(anchor_table, context_scores)
    features, level_thresholds = add_levels(features)

    logger.log("[STEP 10] Write full and train/validation discussion outputs")
    out["full_features"].parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out["full_features"], index=False, encoding="utf-8-sig")
    train_val_features = features[features["split"].isin(["train", "validation"])].copy()
    train_val_features.to_csv(out["discussion_features"], index=False, encoding="utf-8-sig")
    train_val_subwindows = scored_subwindows[scored_subwindows["split"].isin(["train", "validation"])].copy()
    train_val_subwindows.to_csv(out["discussion_subwindows"], index=False, encoding="utf-8-sig")
    level_thresholds.to_csv(out["level_thresholds"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 11] Build train/validation label audit and compact discussion table")
    audit = build_label_audit(features, LABEL_INTERVAL_FILE)
    audit.to_csv(out["audit"], index=False, encoding="utf-8-sig")
    discussion_table = build_discussion_table(features, audit)
    out["discussion_table"].parent.mkdir(parents=True, exist_ok=True)
    discussion_table.to_csv(out["discussion_table"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 12] Draft report and summary")
    split_counts = features["split"].value_counts(dropna=False).to_dict()
    output_file_descriptions = {
        str(out["full_features"]): "full project-level scene anchor audio persistence features; local only, includes test rows",
        str(out["discussion_features"]): "train/validation scene anchor audio persistence features for discussion",
        str(out["discussion_subwindows"]): "train/validation 2s subwindow audio feature and score detail",
        str(out["level_thresholds"]): "train-only percentile thresholds for qualitative levels",
        str(out["audit"]): "train/validation label-distance audit only; not used for scoring",
        str(out["discussion_table"]): "compact scene+audio anchor discussion table",
        str(out["report"]): "machine-readable run report",
        str(out["summary"]): "human-readable summary",
        str(out["run_log"]): "step-by-step run log",
        str(out["script"]): "reproducible script",
    }
    input_stats_after = file_stats(input_paths)
    old_after = old_project_snapshot()
    old_modified = old_before != old_after
    input_changed = stats_changed(input_stats_before, input_stats_after)
    if old_modified:
        errors.append("old project snapshot changed")
    if input_changed:
        errors.append(f"input files modified: {input_changed}")
    report: Dict[str, Any] = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "start_time": start_time,
        "end_time": now_iso(),
        "actual_runtime_seconds": safe_float(time.time() - t0),
        "actual_runtime_readable": readable_seconds(time.time() - t0),
        "input_files": [str(path) for path in input_paths],
        "reference_files": [str(path) for path in REFERENCE_FILES.values() if path.exists()],
        "output_files": [str(path) for key, path in out.items() if key != "bundle_readme"],
        "generated_files": [str(path) for key, path in out.items() if key != "bundle_readme"],
        "warnings": warnings,
        "errors": errors,
        "split_file": str(SPLIT_FILE),
        "split_seed": SPLIT_SEED,
        "train_video_ids": FIXED_SPLIT["train"],
        "validation_video_ids": FIXED_SPLIT["validation"],
        "test_video_ids": FIXED_SPLIT["test"],
        "train_anchor_count": int(split_counts.get("train", 0)),
        "validation_anchor_count": int(split_counts.get("validation", 0)),
        "test_anchor_count": int(split_counts.get("test", 0)),
        "unknown_split_anchor_count": int(split_counts.get("unknown", 0)),
        "discussion_bundle_contains_test_rows": False,
        "train_used_for_level_thresholds": True,
        "validation_used_for_level_thresholds": False,
        "test_used_for_level_thresholds": False,
        "validation_included_for_discussion_audit": True,
        "test_included_for_discussion_audit": False,
        "test_row_level_features_copied_to_bundle": False,
        "label_columns_used_for_score": False,
        "config_file": str(CONFIG_FILE),
        "audio_ad_like_threshold": ad_like_threshold,
        "persistence_window_sec": PERSISTENCE_WINDOW_SEC,
        "subwindow_size_sec": SUBWINDOW_SIZE_SEC,
        "subwindow_stride_sec": SUBWINDOW_STRIDE_SEC,
        "feature_success_count": feature_success,
        "feature_failed_count": feature_failed,
        "missing_video_count": missing_video_count,
        "anchor_file": str(anchor_file),
        "fallback_anchor_source": fallback_anchor_source,
        "anchor_selection_reason": anchor_reason,
        "anchor_count": int(len(anchor_df)),
        "candidate_time_sec_column": "candidate_time_sec",
        "anchor_columns_preserved": [col for col in anchor_table.columns if col in anchor_df.columns or col in {"split", "video_duration_sec", "audio_available"}],
        "score_range_min": score_min,
        "score_range_max": score_max,
        "split_anchor_counts": {str(k): int(v) for k, v in split_counts.items()},
        "subwindow_row_count": int(len(scored_subwindows)),
        "train_val_discussion_row_count": int(len(train_val_features)),
        "train_val_subwindow_row_count": int(len(train_val_subwindows)),
        "audit_row_count": int(len(audit)),
        "discussion_table_row_count": int(len(discussion_table)),
        "old_project_modified": old_modified,
        "old_project_snapshot_before": old_before,
        "old_project_snapshot_after": old_after,
        "input_files_modified": bool(input_changed),
        "input_files_modified_paths": input_changed,
        "backup_dir": str(backup_dir) if backup_dir else None,
        "output_file_descriptions": output_file_descriptions,
        "ffmpeg_path": ffmpeg_path,
        "ffprobe_path": ffprobe_path,
        "latest_for_chatgpt_forbidden_files_found": [],
    }
    save_json(out["report"], report)
    write_summary(out["summary"], report, {})

    logger.log("[STEP 13] Update latest_for_chatgpt_audio_scene_anchor bundle")
    bundle_files = copy_bundle(out, logger, warnings)
    forbidden = forbidden_files_in_bundle()
    report["latest_for_chatgpt_files"] = bundle_files
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden
    if forbidden:
        errors.append(f"forbidden files found in bundle: {forbidden}")
    report["warnings"] = warnings
    report["errors"] = errors

    logger.log("[STEP 14] Run Sub Agent style validations")
    validations = run_validations(report, features, scored_subwindows, bundle_files)
    report["sub_agent_results"] = validations
    if any(result["status"] == "FAIL" for result in validations.values()):
        errors.append("one or more validation checks failed")
    report["errors"] = errors
    report["end_time"] = now_iso()
    report["actual_runtime_seconds"] = safe_float(time.time() - t0)
    report["actual_runtime_readable"] = readable_seconds(time.time() - t0)
    save_json(out["report"], report)
    write_summary(out["summary"], report, {})
    copy_bundle(out, logger, warnings)
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden_files_in_bundle()
    report["warnings"] = warnings
    save_json(out["report"], report)
    shutil.copy2(out["report"], BUNDLE_DIR / out["report"].name)
    shutil.copy2(out["summary"], BUNDLE_DIR / out["summary"].name)
    logger.log("[STEP 15] Print human-readable final summary")
    logger.log(f"Actual runtime: {report['actual_runtime_readable']}")
    logger.log(f"Output report: {out['report']}")
    print("\n작업 완료 요약", flush=True)
    print(f"- status: {'FAIL' if errors else 'CONDITIONAL_SUCCESS' if warnings else 'SUCCESS'}", flush=True)
    print(f"- anchor_file: {anchor_file}", flush=True)
    print(f"- anchor_count: {len(anchor_df)} split_counts={split_counts}", flush=True)
    print(f"- subwindow_features success/fail: {feature_success}/{feature_failed}", flush=True)
    print(f"- discussion rows train+validation: {len(train_val_features)}", flush=True)
    print(f"- discussion bundle contains test rows: false", flush=True)
    print(f"- report: {out['report']}", flush=True)


if __name__ == "__main__":
    main()
