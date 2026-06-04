#!/usr/bin/env python3
"""광고 경계 주변 오디오 feature를 추출하고 persistence 기반 오디오 단서 규칙을 설계한다."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
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
TASK_NAME = "extract_audio_ad_edge_persistence"
DATA_AUDIO_DIR = PROJECT_ROOT / "data/audio"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_AUDIO_DIR = PROJECT_ROOT / "scripts/audio"
CONFIGS_DIR = PROJECT_ROOT / "configs"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_a"
BACKUPS_DIR = PROJECT_ROOT / "backups"
RAW_VIDEO_DIR = PROJECT_ROOT / "data/raw/videos"

LABEL_FILE = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_4.csv"
MANIFEST_FILE = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
PREVIOUS_FILES = {
    "labeled_features": DATA_AUDIO_DIR / "audio_labeled_segment_features_v2_4.csv",
    "candidate_thresholds": DATA_AUDIO_DIR / "audio_rule_candidate_thresholds_v2_4.csv",
    "feature_recommendations": DATA_AUDIO_DIR / "audio_rule_feature_recommendations_v2_4.csv",
    "pairwise": DATA_AUDIO_DIR / "audio_rule_pairwise_feature_comparison_v2_4.csv",
    "paired_delta_summary": DATA_AUDIO_DIR / "audio_rule_paired_context_delta_summary_v2_4.csv",
    "normalized_pairwise": DATA_AUDIO_DIR / "audio_rule_video_normalized_pairwise_comparison_v2_4.csv",
    "analysis_summary": REPORTS_DIR / "analyze_audio_rule_features_v2_4_summary.md",
    "analysis_report": REPORTS_DIR / "analyze_audio_rule_features_v2_4_report.json",
    "previous_sampling_plan": DATA_AUDIO_DIR / "audio_labeled_segment_sampling_plan_v2_4.csv",
}
HELPER_SCRIPT = SCRIPTS_AUDIO_DIR / "extract_labeled_audio_features_v2_4.py"

SAMPLE_RATE = 16000
FRAME_LENGTH = 2048
HOP_LENGTH = 512
SUBWINDOW_SIZE_SEC = 2.0
SUBWINDOW_STRIDE_SEC = 2.0
RANDOM_SEED = 20260524

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
SCORE_COMPONENTS = {
    "onset_density_score": ["onset_density", "onset_count_per_sec"],
    "flux_onset_score": ["spectral_flux_mean", "onset_strength_mean", "spectral_flux_max", "onset_strength_max"],
    "energy_score": ["audio_log_energy_mean", "audio_rms_mean", "audio_mean_abs_amplitude"],
    "inverse_silence_score": ["silence_ratio", "low_energy_ratio"],
    "spectral_texture_score": ["spectral_flatness_std", "spectral_bandwidth_mean", "spectral_centroid_std"],
}
SCORE_WEIGHTS = {
    "onset_density_score": 0.30,
    "flux_onset_score": 0.25,
    "energy_score": 0.20,
    "inverse_silence_score": 0.15,
    "spectral_texture_score": 0.10,
}


def load_helper() -> Any:
    spec = importlib.util.spec_from_file_location("audio_feature_helper_v2_4", HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper script: {HELPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPER = load_helper()
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


def output_paths() -> Dict[str, Path]:
    return {
        "edge_plan": DATA_AUDIO_DIR / "audio_ad_edge_5s_10s_sampling_plan_v2_4.csv",
        "edge_features": DATA_AUDIO_DIR / "audio_ad_edge_5s_10s_features_v2_4.csv",
        "subwindow_plan": DATA_AUDIO_DIR / "audio_ad_edge_persistence_subwindow_plan_v2_4.csv",
        "subwindow_features": DATA_AUDIO_DIR / "audio_ad_edge_persistence_subwindow_features_v2_4.csv",
        "score_components": DATA_AUDIO_DIR / "audio_ad_edge_ad_like_score_components_v2_4.csv",
        "context_scores": DATA_AUDIO_DIR / "audio_ad_edge_persistence_context_scores_v2_4.csv",
        "boundary_scores": DATA_AUDIO_DIR / "audio_ad_edge_boundary_persistence_scores_v2_4.csv",
        "edge_pairwise": DATA_AUDIO_DIR / "audio_ad_edge_5s_10s_pairwise_comparison_v2_4.csv",
        "edge_delta_summary": DATA_AUDIO_DIR / "audio_ad_edge_5s_10s_delta_summary_v2_4.csv",
        "rule_design_csv": DATA_AUDIO_DIR / "audio_persistence_rule_design_v2_4.csv",
        "rule_config": CONFIGS_DIR / "audio_persistence_rule_config_v2_4.json",
        "rule_design_md": REPORTS_DIR / "audio_persistence_rule_design_v2_4.md",
        "report": REPORTS_DIR / "extract_audio_ad_edge_persistence_v2_4_report.json",
        "summary": REPORTS_DIR / "extract_audio_ad_edge_persistence_v2_4_summary.md",
        "run_log": LOGS_DIR / "extract_audio_ad_edge_persistence_v2_4_run_log.txt",
        "script": SCRIPTS_AUDIO_DIR / "extract_audio_ad_edge_persistence_v2_4.py",
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def iso_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


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
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
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


def backup_existing(paths: Dict[str, Path], timestamp: str) -> Optional[Path]:
    targets = list(paths.values()) + [LATEST_DIR / "README_latest_files.md"]
    existing = [path for path in targets if path.exists()]
    if not existing:
        return None
    backup_dir = BACKUPS_DIR / f"extract_audio_ad_edge_persistence_v2_4_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        if path.is_file():
            shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def executable_path(name: str) -> Optional[str]:
    path = shutil.which(name)
    if path:
        return path
    candidate = Path(".venv/bin") / name
    return str(candidate) if candidate.exists() else None


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def canonical_intervals(label_file: Path) -> pd.DataFrame:
    df = read_csv(label_file)
    if "segment_type" in df.columns:
        df = df[df["segment_type"].astype(str).str.lower().eq("ad_interval")].copy()
    out = pd.DataFrame()
    out["ad_interval_id"] = df.get("ad_interval_id", pd.Series([f"{VERSION}_{i}" for i in range(len(df))])).astype(str)
    out["video_id"] = df["video_id"].astype(str)
    out["video_title"] = df.get("video_title", "")
    out["video_filename"] = df.get("video_filename", "")
    out["video_path"] = df.get("video_path", "")
    out["ad_start_sec"] = pd.to_numeric(df["ad_start_sec"], errors="coerce")
    out["ad_end_sec"] = pd.to_numeric(df["ad_end_sec"], errors="coerce")
    out["ad_duration_sec"] = out["ad_end_sec"] - out["ad_start_sec"]
    out["validation_status"] = np.where(
        out["video_id"].notna() & (out["ad_start_sec"] < out["ad_end_sec"]) & (out["ad_duration_sec"] > 0),
        "valid",
        "invalid_interval",
    )
    out["validation_warning"] = ""
    duplicated = out.duplicated(["video_id", "ad_start_sec", "ad_end_sec"], keep=False)
    out.loc[duplicated, "validation_warning"] = "duplicate_interval"
    return out.reset_index(drop=True)


def resolve_path(value: Any) -> Optional[Path]:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_video_paths(intervals: pd.DataFrame, manifest_file: Path) -> pd.DataFrame:
    manifest = read_csv(manifest_file) if manifest_file.exists() else pd.DataFrame()
    manifest_by_id = {}
    if not manifest.empty and "video_id" in manifest.columns:
        manifest_by_id = {str(row["video_id"]): row for _, row in manifest.iterrows()}
    records = []
    for _, row in intervals.iterrows():
        video_id = str(row["video_id"])
        manifest_row = manifest_by_id.get(video_id)
        candidates: List[Path] = []
        label_path = resolve_path(row.get("video_path"))
        if label_path:
            candidates.append(label_path)
        if manifest_row is not None and "video_path" in manifest_row.index:
            mpath = resolve_path(manifest_row.get("video_path"))
            if mpath:
                candidates.append(mpath)
        filename_values = [row.get("video_filename")]
        if manifest_row is not None and "video_filename" in manifest_row.index:
            filename_values.append(manifest_row.get("video_filename"))
        for value in filename_values:
            if value is not None and not pd.isna(value) and str(value).strip():
                candidates.append(RAW_VIDEO_DIR / str(value).strip())
        resolved = None
        method = "unresolved"
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                method = "path_or_filename_match"
                break
        if resolved is None:
            for value in filename_values:
                if value is None or pd.isna(value) or not str(value).strip():
                    continue
                matches = list(RAW_VIDEO_DIR.glob(str(value).strip()))
                if matches:
                    resolved = matches[0]
                    method = "glob_match"
                    break
        manifest_duration = np.nan
        if manifest_row is not None and "duration_sec" in manifest_row.index:
            manifest_duration = safe_float(manifest_row.get("duration_sec"))
        records.append(
            {
                "video_id": video_id,
                "video_path": str(resolved) if resolved else "",
                "video_path_exists": bool(resolved and resolved.exists()),
                "video_path_resolution_method": method,
                "manifest_duration_sec": manifest_duration,
            }
        )
    return pd.DataFrame(records).drop_duplicates("video_id")


def probe_videos(video_paths: pd.DataFrame, ffprobe_path: Optional[str]) -> pd.DataFrame:
    records = []
    for _, row in video_paths.iterrows():
        path = Path(row["video_path"]) if row["video_path"] else None
        record = {
            "video_id": str(row["video_id"]),
            "video_path": row["video_path"],
            "path_exists": bool(path and path.exists()),
            "manifest_duration_sec": row.get("manifest_duration_sec", np.nan),
            "probe_status": "not_run",
            "probe_error": "",
            "video_duration_sec": row.get("manifest_duration_sec", np.nan),
            "audio_available": False,
            "audio_stream_index": np.nan,
            "audio_codec_name": "",
            "audio_sample_rate": np.nan,
        }
        if not record["path_exists"]:
            record["probe_status"] = "path_missing"
            record["probe_error"] = "video path missing"
        elif not ffprobe_path:
            record["probe_status"] = "ffprobe_missing"
            record["probe_error"] = "ffprobe executable not found"
        else:
            cmd = [
                ffprobe_path,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode:
                record["probe_status"] = "probe_failed"
                record["probe_error"] = proc.stderr.strip() or proc.stdout.strip()
            else:
                data = json.loads(proc.stdout)
                record["probe_status"] = "success"
                record["video_duration_sec"] = safe_float(data.get("format", {}).get("duration"))
                audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
                if audio_streams:
                    stream = audio_streams[0]
                    record["audio_available"] = True
                    record["audio_stream_index"] = stream.get("index", np.nan)
                    record["audio_codec_name"] = stream.get("codec_name", "")
                    record["audio_sample_rate"] = safe_float(stream.get("sample_rate"))
                else:
                    record["probe_error"] = "no_audio_stream"
        records.append(record)
    return pd.DataFrame(records)


def build_edge_plan(intervals: pd.DataFrame, video_paths: pd.DataFrame, probe_df: pd.DataFrame, label_mtime: str) -> pd.DataFrame:
    path_by_video = {str(row["video_id"]): row for _, row in video_paths.iterrows()}
    probe_by_video = {str(row["video_id"]): row for _, row in probe_df.iterrows()}
    rows: List[Dict[str, Any]] = []
    segment_defs = [
        ("ad_start_first_5s", lambda s, e, d: (s, min(s + 5.0, e, d))),
        ("ad_start_first_10s", lambda s, e, d: (s, min(s + 10.0, e, d))),
        ("ad_start_5to10s", lambda s, e, d: (min(s + 5.0, e), min(s + 10.0, e, d))),
        ("ad_end_last_5s", lambda s, e, d: (max(s, e - 5.0, 0.0), min(e, d))),
        ("ad_end_last_10s", lambda s, e, d: (max(s, e - 10.0, 0.0), min(e, d))),
        ("ad_end_minus10to_minus5s", lambda s, e, d: (max(s, e - 10.0, 0.0), max(s, e - 5.0, 0.0))),
    ]
    for _, interval in intervals.iterrows():
        video_id = str(interval["video_id"])
        path_row = path_by_video.get(video_id)
        probe = probe_by_video.get(video_id)
        video_path = path_row.get("video_path", "") if path_row is not None else ""
        video_duration = safe_float(probe.get("video_duration_sec")) if probe is not None else np.nan
        ad_start = safe_float(interval["ad_start_sec"])
        ad_end = safe_float(interval["ad_end_sec"])
        ad_duration = safe_float(interval["ad_duration_sec"])
        short5 = ad_duration < 5.0
        short10 = ad_duration < 10.0
        for segment_type, fn in segment_defs:
            start, end = fn(ad_start, ad_end, video_duration if np.isfinite(video_duration) else ad_end)
            duration = end - start if np.isfinite(start) and np.isfinite(end) else np.nan
            status = "planned" if np.isfinite(duration) and duration > 0 else "invalid_segment"
            warnings: List[str] = []
            if segment_type == "ad_start_5to10s" and (not np.isfinite(duration) or duration <= 0):
                status = "too_short_for_5to10s"
                warnings.append("duration_non_positive")
            if segment_type == "ad_end_minus10to_minus5s" and (not np.isfinite(duration) or duration <= 0):
                status = "too_short_for_minus10to_minus5s"
                warnings.append("duration_non_positive")
            truncated_by_ad = False
            if segment_type in {"ad_start_first_5s", "ad_end_last_5s"} and duration < 5.0 - 1e-6:
                truncated_by_ad = True
            if segment_type in {"ad_start_first_10s", "ad_start_5to10s", "ad_end_last_10s", "ad_end_minus10to_minus5s"} and duration < (10.0 if "first_10s" in segment_type or "last_10s" in segment_type else 5.0) - 1e-6:
                truncated_by_ad = True
            if truncated_by_ad:
                warnings.append("segment_truncated_by_ad_duration")
            truncated_by_video = bool(np.isfinite(video_duration) and end > video_duration + 1e-6)
            if truncated_by_video:
                warnings.append("segment_truncated_by_video_boundary")
            start_end_overlap = short10
            rows.append(
                {
                    "segment_id": f"{VERSION}_{interval['ad_interval_id']}_{segment_type}",
                    "version": VERSION,
                    "source_label_file": str(LABEL_FILE),
                    "source_label_modified_time": label_mtime,
                    "video_id": video_id,
                    "video_title": interval.get("video_title", ""),
                    "video_path": video_path,
                    "video_duration_sec": video_duration,
                    "ad_interval_id": interval["ad_interval_id"],
                    "ad_start_sec": ad_start,
                    "ad_end_sec": ad_end,
                    "ad_duration_sec": ad_duration,
                    "segment_type": segment_type,
                    "segment_start_sec": start,
                    "segment_end_sec": end,
                    "segment_duration_sec": duration,
                    "segment_start_mmss": mmss(start),
                    "segment_end_mmss": mmss(end),
                    "relative_start_from_ad_start_sec": start - ad_start if np.isfinite(start) else np.nan,
                    "relative_end_from_ad_start_sec": end - ad_start if np.isfinite(end) else np.nan,
                    "relative_start_from_ad_end_sec": start - ad_end if np.isfinite(start) else np.nan,
                    "relative_end_from_ad_end_sec": end - ad_end if np.isfinite(end) else np.nan,
                    "sampling_status": status,
                    "sampling_warning": ";".join(warnings),
                    "short_ad_under_5s": bool(short5),
                    "short_ad_under_10s": bool(short10),
                    "start_end_edge_overlap": bool(start_end_overlap),
                    "segment_truncated_by_ad_duration": bool(truncated_by_ad),
                    "segment_truncated_by_video_boundary": bool(truncated_by_video),
                }
            )
    return pd.DataFrame(rows)


def build_subwindow_plan(intervals: pd.DataFrame, video_paths: pd.DataFrame, probe_df: pd.DataFrame) -> pd.DataFrame:
    path_by_video = {str(row["video_id"]): row for _, row in video_paths.iterrows()}
    probe_by_video = {str(row["video_id"]): row for _, row in probe_df.iterrows()}
    rows: List[Dict[str, Any]] = []

    def add_windows(
        interval: pd.Series,
        boundary_type: str,
        context_type: str,
        parent_segment_type: str,
        context_start: float,
        context_end: float,
        boundary_sec: float,
        video_duration: float,
        video_path: str,
    ) -> None:
        clipped_start = max(0.0, context_start)
        clipped_end = min(video_duration, context_end) if np.isfinite(video_duration) else context_end
        idx = 0
        cursor = clipped_start
        while cursor < clipped_end - 1e-6:
            end = min(cursor + SUBWINDOW_SIZE_SEC, clipped_end)
            duration = end - cursor
            status = "planned" if duration > 0 else "invalid_subwindow"
            warning = []
            if duration < 2.0 - 1e-6:
                warning.append("reliability_warning_short_subwindow")
            rows.append(
                {
                    "subwindow_id": f"{VERSION}_{interval['ad_interval_id']}_{context_type}_{idx:02d}",
                    "version": VERSION,
                    "video_id": str(interval["video_id"]),
                    "video_path": video_path,
                    "video_duration_sec": video_duration,
                    "ad_interval_id": interval["ad_interval_id"],
                    "boundary_type": boundary_type,
                    "context_type": context_type,
                    "parent_segment_type": parent_segment_type,
                    "subwindow_index": idx,
                    "subwindow_start_sec": cursor,
                    "subwindow_end_sec": end,
                    "subwindow_duration_sec": duration,
                    "relative_start_to_boundary_sec": cursor - boundary_sec,
                    "relative_end_to_boundary_sec": end - boundary_sec,
                    "sampling_status": status,
                    "sampling_warning": ";".join(warning),
                }
            )
            idx += 1
            cursor += SUBWINDOW_STRIDE_SEC

    for _, interval in intervals.iterrows():
        video_id = str(interval["video_id"])
        path_row = path_by_video.get(video_id)
        probe = probe_by_video.get(video_id)
        video_path = path_row.get("video_path", "") if path_row is not None else ""
        video_duration = safe_float(probe.get("video_duration_sec")) if probe is not None else np.nan
        ad_start = safe_float(interval["ad_start_sec"])
        ad_end = safe_float(interval["ad_end_sec"])
        add_windows(interval, "ad_start", "start_pre_10s", "pre_ad_10s", ad_start - 10.0, ad_start, ad_start, video_duration, video_path)
        add_windows(interval, "ad_start", "start_post_10s", "ad_start_first_10s", ad_start, min(ad_start + 10.0, ad_end), ad_start, video_duration, video_path)
        add_windows(interval, "ad_end", "end_pre_10s", "ad_end_last_10s", max(ad_start, ad_end - 10.0), ad_end, ad_end, video_duration, video_path)
        add_windows(interval, "ad_end", "end_post_10s", "post_ad_10s", ad_end, min(ad_end + 10.0, video_duration if np.isfinite(video_duration) else ad_end + 10.0), ad_end, video_duration, video_path)

    random_plan = PREVIOUS_FILES["previous_sampling_plan"]
    if random_plan.exists():
        plan = read_csv(random_plan)
        random_rows = plan[plan["segment_type"].astype(str).eq("random_non_ad_30s")]
        for _, row in random_rows.iterrows():
            duration = safe_float(row["segment_duration_sec"])
            if not np.isfinite(duration) or duration <= 0 or row.get("sampling_status") not in {"planned", "sampled"}:
                continue
            start = safe_float(row["segment_start_sec"])
            end = safe_float(row["segment_end_sec"])
            video_duration = safe_float(row.get("video_duration_sec"))
            cursor = start
            idx = 0
            while cursor < end - 1e-6:
                sw_end = min(cursor + SUBWINDOW_SIZE_SEC, end)
                sw_duration = sw_end - cursor
                rows.append(
                    {
                        "subwindow_id": f"{VERSION}_{row['ad_interval_id']}_random_non_ad_30s_{idx:02d}",
                        "version": VERSION,
                        "video_id": str(row["video_id"]),
                        "video_path": row["video_path"],
                        "video_duration_sec": video_duration,
                        "ad_interval_id": row["ad_interval_id"],
                        "boundary_type": "random_non_ad_baseline",
                        "context_type": "random_non_ad_30s",
                        "parent_segment_type": "random_non_ad_30s",
                        "subwindow_index": idx,
                        "subwindow_start_sec": cursor,
                        "subwindow_end_sec": sw_end,
                        "subwindow_duration_sec": sw_duration,
                        "relative_start_to_boundary_sec": np.nan,
                        "relative_end_to_boundary_sec": np.nan,
                        "sampling_status": "planned",
                        "sampling_warning": "reliability_warning_short_subwindow" if sw_duration < 2.0 - 1e-6 else "",
                    }
                )
                idx += 1
                cursor += SUBWINDOW_STRIDE_SEC
    return pd.DataFrame(rows)


def add_empty_feature_columns(record: Dict[str, Any]) -> Dict[str, Any]:
    record.update(
        {
            "feature_status": "not_run",
            "feature_error": "",
            "audio_available": False,
            "audio_stream_index": np.nan,
            "decoded_sample_rate": np.nan,
            "decoded_num_samples": 0,
            "decoded_duration_sec": np.nan,
            "decoded_duration_diff_sec": np.nan,
        }
    )
    for col in FEATURE_COLUMNS:
        record[col] = np.nan
    return record


def extract_features(plan_df: pd.DataFrame, id_col: str, start_col: str, duration_col: str, ffmpeg_path: Optional[str], probe_df: pd.DataFrame, logger: TaskLogger, label: str) -> pd.DataFrame:
    probe_by_video = {str(row["video_id"]): row for _, row in probe_df.iterrows()}
    records: List[Dict[str, Any]] = []
    for idx, row in plan_df.iterrows():
        record = add_empty_feature_columns(row.to_dict())
        probe = probe_by_video.get(str(row["video_id"]))
        if probe is not None:
            record["audio_available"] = bool(probe.get("audio_available", False))
            record["audio_stream_index"] = probe.get("audio_stream_index", np.nan)
        if row.get("sampling_status") != "planned":
            record["feature_status"] = row.get("sampling_status", "sampling_not_planned")
            record["feature_error"] = row.get("sampling_warning", "")
            records.append(record)
            continue
        if not ffmpeg_path:
            record["feature_status"] = "ffmpeg_missing"
            record["feature_error"] = "ffmpeg executable not found"
            records.append(record)
            continue
        if not record["audio_available"]:
            record["feature_status"] = "audio_stream_unavailable"
            record["feature_error"] = "no audio stream from ffprobe"
            records.append(record)
            continue
        video_path = str(row.get("video_path", ""))
        start = safe_float(row.get(start_col))
        duration = safe_float(row.get(duration_col))
        if not video_path or not Path(video_path).exists() or not np.isfinite(start) or not np.isfinite(duration) or duration <= 0:
            record["feature_status"] = "invalid_segment_or_video_path"
            record["feature_error"] = "invalid path/start/duration"
            records.append(record)
            continue
        audio, sr, err = HELPER.decode_audio_segment(ffmpeg_path, video_path, start, duration)
        if err or audio.size == 0:
            record["feature_status"] = "decode_failed"
            record["feature_error"] = err or "empty decoded audio"
            records.append(record)
            continue
        decoded_duration = audio.size / float(sr)
        record["decoded_sample_rate"] = sr
        record["decoded_num_samples"] = int(audio.size)
        record["decoded_duration_sec"] = decoded_duration
        record["decoded_duration_diff_sec"] = decoded_duration - duration
        try:
            record.update(HELPER.compute_audio_features(audio, sr))
            record["feature_status"] = "success"
        except Exception as exc:
            record["feature_status"] = "feature_compute_failed"
            record["feature_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        if (idx + 1) % 100 == 0:
            logger.log(f"[STEP {'08' if label == 'edge' else '09'}] Extracted {label} audio features for {idx + 1}/{len(plan_df)} rows")
    out = pd.DataFrame(records)
    add_duration_normalized_features(out, duration_col)
    return out


def add_duration_normalized_features(df: pd.DataFrame, duration_col: str) -> None:
    duration = pd.to_numeric(df[duration_col], errors="coerce").replace(0, np.nan)
    onset = pd.to_numeric(df["onset_count"], errors="coerce")
    df["onset_density"] = onset / duration
    df["onset_count_per_sec"] = onset / duration
    df["spectral_flux_mean_per_sec_proxy"] = pd.to_numeric(df["spectral_flux_mean"], errors="coerce")
    df["onset_strength_mean_per_sec_proxy"] = pd.to_numeric(df["onset_strength_mean"], errors="coerce")


def sigmoid(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").clip(-6, 6)
    return 1.0 / (1.0 + np.exp(-x))


def add_video_robust_scores(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for feature in features:
        z_col = f"{feature}__video_robust_z"
        out[z_col] = np.nan
        for video_id, group in out.groupby("video_id"):
            values = pd.to_numeric(group[feature], errors="coerce")
            finite = values[np.isfinite(values)]
            if len(finite) < 2:
                continue
            median = finite.median()
            iqr = finite.quantile(0.75) - finite.quantile(0.25)
            scale = iqr if np.isfinite(iqr) and iqr > 0 else finite.std(ddof=1)
            if not np.isfinite(scale) or scale == 0:
                continue
            out.loc[group.index, z_col] = (values - median) / scale
    return out


def add_ad_like_scores(df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    all_features = list(dict.fromkeys(HIGHER_FEATURES + LOWER_FEATURES))
    score_df = add_video_robust_scores(df, all_features)
    baseline_z = add_video_robust_scores(baseline_df, all_features)
    global_stats: Dict[str, Tuple[float, float]] = {}
    for feature in all_features:
        values = pd.to_numeric(baseline_z.get(f"{feature}__video_robust_z", pd.Series(dtype=float)), errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) >= 3:
            global_stats[feature] = (finite.median(), finite.quantile(0.75) - finite.quantile(0.25))
        else:
            raw = pd.to_numeric(baseline_df.get(feature, pd.Series(dtype=float)), errors="coerce")
            finite_raw = raw[np.isfinite(raw)]
            global_stats[feature] = (finite_raw.median() if len(finite_raw) else 0.0, finite_raw.std(ddof=1) if len(finite_raw) > 1 else 1.0)
    for feature in all_features:
        z_col = f"{feature}__video_robust_z"
        if z_col not in score_df.columns:
            score_df[z_col] = np.nan
        raw = pd.to_numeric(score_df[feature], errors="coerce")
        fallback_median, fallback_scale = global_stats.get(feature, (0.0, 1.0))
        if not np.isfinite(fallback_scale) or fallback_scale == 0:
            fallback_scale = 1.0
        score_df[z_col] = score_df[z_col].fillna((raw - fallback_median) / fallback_scale)
        directional = score_df[z_col] if feature in HIGHER_FEATURES else -score_df[z_col]
        score_df[f"{feature}__directional_score"] = sigmoid(directional)
    for component, features in SCORE_COMPONENTS.items():
        cols = [f"{feature}__directional_score" for feature in features if f"{feature}__directional_score" in score_df.columns]
        score_df[component] = score_df[cols].mean(axis=1, skipna=True) if cols else np.nan
    total = pd.Series(0.0, index=score_df.index)
    weight_sum = pd.Series(0.0, index=score_df.index)
    for component, weight in SCORE_WEIGHTS.items():
        vals = pd.to_numeric(score_df[component], errors="coerce")
        total = total + vals.fillna(0.0) * weight
        weight_sum = weight_sum + vals.notna().astype(float) * weight
    score_df["audio_ad_like_score"] = (total / weight_sum.replace(0, np.nan)).clip(0, 1)
    return score_df


def threshold_on_onset_density(previous_features: pd.DataFrame) -> Dict[str, Any]:
    df = previous_features[previous_features["segment_type"].isin(["ad_full", "random_non_ad_30s"])].copy()
    df["onset_density"] = pd.to_numeric(df["onset_count"], errors="coerce") / pd.to_numeric(df["segment_duration_sec"], errors="coerce").replace(0, np.nan)
    df = df[np.isfinite(df["onset_density"])]
    if df.empty or df["segment_type"].nunique() < 2:
        return {"feature_name": "onset_density", "best_threshold": 0.0, "balanced_accuracy": np.nan, "threshold_source": "fallback_missing_labeled_density"}
    y = (df["segment_type"] == "ad_full").astype(int).to_numpy()
    x = df["onset_density"].to_numpy(dtype=float)
    best = None
    for thr in np.unique(np.nanquantile(x, np.linspace(0.05, 0.95, 19))):
        pred = (x >= thr).astype(int)
        tp = ((y == 1) & (pred == 1)).sum()
        tn = ((y == 0) & (pred == 0)).sum()
        fp = ((y == 0) & (pred == 1)).sum()
        fn = ((y == 1) & (pred == 0)).sum()
        recall = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        bal = np.nanmean([recall, spec])
        cand = (bal, thr, recall, spec)
        if best is None or cand[0] > best[0]:
            best = cand
    return {
        "feature_name": "onset_density",
        "best_threshold": safe_float(best[1]),
        "balanced_accuracy": safe_float(best[0]),
        "recall": safe_float(best[2]),
        "specificity": safe_float(best[3]),
        "threshold_source": "ad_full_vs_random_non_ad_30s onset_count / segment_duration_sec",
    }


def compute_context_scores(score_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (ad_interval_id, context_type), group in score_df.groupby(["ad_interval_id", "context_type"], dropna=False):
        scores = pd.to_numeric(group["audio_ad_like_score"], errors="coerce")
        valid = scores[np.isfinite(scores)]
        flags = (scores >= threshold).fillna(False).astype(bool).tolist()
        max_run = 0
        current = 0
        for flag in flags:
            if flag:
                current += 1
                max_run = max(max_run, current)
            else:
                current = 0
        if len(valid):
            q25 = valid.quantile(0.25)
            q75 = valid.quantile(0.75)
            std = valid.std(ddof=1) if len(valid) > 1 else 0.0
            median = valid.median()
            consistency = 1.0 - min(1.0, safe_float(std) / max(abs(safe_float(median)), 1e-6))
        else:
            q25 = q75 = std = median = np.nan
            consistency = np.nan
        sorted_group = group.sort_values("subwindow_index")
        if len(sorted_group) >= 2:
            x = pd.to_numeric(sorted_group["subwindow_index"], errors="coerce")
            y = pd.to_numeric(sorted_group["audio_ad_like_score"], errors="coerce")
            mask = np.isfinite(x) & np.isfinite(y)
            slope = np.polyfit(x[mask], y[mask], 1)[0] if mask.sum() >= 2 else np.nan
        else:
            slope = np.nan
        rows.append(
            {
                "version": VERSION,
                "ad_interval_id": ad_interval_id,
                "video_id": group["video_id"].iloc[0],
                "boundary_type": group["boundary_type"].iloc[0],
                "context_type": context_type,
                "subwindow_count": int(len(group)),
                "valid_subwindow_count": int(len(valid)),
                "audio_ad_like_score_mean": safe_float(valid.mean()) if len(valid) else np.nan,
                "audio_ad_like_score_median": safe_float(median),
                "audio_ad_like_score_p25": safe_float(q25),
                "audio_ad_like_score_p75": safe_float(q75),
                "audio_ad_like_score_min": safe_float(valid.min()) if len(valid) else np.nan,
                "audio_ad_like_score_max": safe_float(valid.max()) if len(valid) else np.nan,
                "audio_ad_like_score_std": safe_float(std),
                "ad_like_subwindow_count": int(sum(flags)),
                "ad_like_ratio": safe_float(sum(flags) / len(flags)) if flags else np.nan,
                "max_consecutive_ad_like_count": int(max_run),
                "max_consecutive_ad_like_sec": safe_float(max_run * SUBWINDOW_SIZE_SEC),
                "score_consistency": safe_float(consistency),
                "score_iqr": safe_float(q75 - q25) if np.isfinite(q25) and np.isfinite(q75) else np.nan,
                "score_slope": safe_float(slope),
                "high_onset_density_ratio": safe_float((pd.to_numeric(group["onset_density"], errors="coerce") >= group.attrs.get("onset_density_threshold", 0)).mean()),
                "high_flux_ratio": safe_float((pd.to_numeric(group["spectral_flux_mean__directional_score"], errors="coerce") >= 0.6).mean()),
                "low_silence_ratio": safe_float((pd.to_numeric(group["silence_ratio__directional_score"], errors="coerce") >= 0.6).mean()),
            }
        )
    return pd.DataFrame(rows)


def boundary_scores(context_scores: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    context_map = {(row["ad_interval_id"], row["context_type"]): row for _, row in context_scores.iterrows()}
    interval_map = {row["ad_interval_id"]: row for _, row in intervals.iterrows()}
    rows = []
    for ad_interval_id, interval in interval_map.items():
        for boundary_type, pre_context, post_context, boundary_sec in [
            ("ad_start", "start_pre_10s", "start_post_10s", interval["ad_start_sec"]),
            ("ad_end", "end_pre_10s", "end_post_10s", interval["ad_end_sec"]),
        ]:
            pre = context_map.get((ad_interval_id, pre_context))
            post = context_map.get((ad_interval_id, post_context))
            if pre is None or post is None:
                continue
            pre_med = safe_float(pre["audio_ad_like_score_median"])
            post_med = safe_float(post["audio_ad_like_score_median"])
            pre_ratio = safe_float(pre["ad_like_ratio"])
            post_ratio = safe_float(post["ad_like_ratio"])
            pre_consec = safe_float(pre["max_consecutive_ad_like_sec"])
            post_consec = safe_float(post["max_consecutive_ad_like_sec"])
            if boundary_type == "ad_start":
                score_delta = post_med - pre_med
                ratio_delta = post_ratio - pre_ratio
                score = (
                    0.30 * post_ratio
                    + 0.25 * min(post_consec / 10.0, 1.0)
                    + 0.20 * post_med
                    + 0.15 * safe_float(post["audio_ad_like_score_p25"])
                    + 0.10 * max(score_delta, 0.0)
                )
                start_score = score
                end_score = np.nan
                pattern = "start_like" if score >= 0.6 and score_delta > 0 else "weak_or_unclear"
                hint = "광고 시작 후 10초의 ad-like persistence와 pre→post 증가를 보는 start boundary audio cue"
            else:
                score_delta = pre_med - post_med
                ratio_delta = pre_ratio - post_ratio
                score = (
                    0.30 * pre_ratio
                    + 0.25 * min(pre_consec / 10.0, 1.0)
                    + 0.20 * pre_med
                    + 0.15 * safe_float(pre["audio_ad_like_score_p25"])
                    + 0.10 * max(score_delta, 0.0)
                )
                start_score = np.nan
                end_score = score
                pattern = "end_like" if score >= 0.6 and score_delta > 0 else "weak_or_unclear"
                hint = "광고 종료 전 10초 persistence와 종료 후 감소를 보는 end boundary audio cue"
            rows.append(
                {
                    "version": VERSION,
                    "video_id": interval["video_id"],
                    "ad_interval_id": ad_interval_id,
                    "boundary_type": boundary_type,
                    "boundary_sec": safe_float(boundary_sec),
                    "boundary_mmss": mmss(boundary_sec),
                    "pre_context_type": pre_context,
                    "post_context_type": post_context,
                    "pre_context_score_median": pre_med,
                    "post_context_score_median": post_med,
                    "pre_context_score_p25": safe_float(pre["audio_ad_like_score_p25"]),
                    "post_context_score_p25": safe_float(post["audio_ad_like_score_p25"]),
                    "pre_ad_like_ratio": pre_ratio,
                    "post_ad_like_ratio": post_ratio,
                    "pre_max_consecutive_ad_like_sec": pre_consec,
                    "post_max_consecutive_ad_like_sec": post_consec,
                    "score_delta_for_boundary_direction": safe_float(score_delta),
                    "ratio_delta_for_boundary_direction": safe_float(ratio_delta),
                    "audio_start_persistence_score": safe_float(start_score),
                    "audio_end_persistence_score": safe_float(end_score),
                    "audio_boundary_persistence_score": safe_float(score),
                    "boundary_audio_pattern": pattern,
                    "interpretation_hint": hint,
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
    pooled = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return safe_float((a.mean() - b.mean()) / math.sqrt(pooled))


def robust_effect(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    pooled_iqr = np.nanmean([a.quantile(0.75) - a.quantile(0.25), b.quantile(0.75) - b.quantile(0.25)])
    if not np.isfinite(pooled_iqr) or pooled_iqr == 0:
        return np.nan
    return safe_float((a.median() - b.median()) / pooled_iqr)


def edge_pairwise(edge_features: pd.DataFrame, previous_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([previous_features, edge_features], ignore_index=True, sort=False)
    features = [
        "onset_density",
        "onset_count_per_sec",
        "spectral_flux_mean",
        "onset_strength_mean",
        "spectral_flux_max",
        "onset_strength_max",
        "audio_log_energy_mean",
        "audio_rms_mean",
        "audio_mean_abs_amplitude",
        "silence_ratio",
        "low_energy_ratio",
        "spectral_flatness_std",
        "spectral_bandwidth_mean",
        "spectral_centroid_std",
        "zero_crossing_rate_std",
    ]
    if "onset_density" not in combined.columns:
        combined["onset_density"] = pd.to_numeric(combined["onset_count"], errors="coerce") / pd.to_numeric(combined["segment_duration_sec"], errors="coerce").replace(0, np.nan)
        combined["onset_count_per_sec"] = combined["onset_density"]
    comparisons = [
        ("pre_ad_10s_vs_ad_start_first_5s", "pre_ad_10s", "ad_start_first_5s"),
        ("pre_ad_10s_vs_ad_start_first_10s", "pre_ad_10s", "ad_start_first_10s"),
        ("ad_start_first_5s_vs_ad_start_5to10s", "ad_start_first_5s", "ad_start_5to10s"),
        ("ad_end_last_5s_vs_post_ad_10s", "ad_end_last_5s", "post_ad_10s"),
        ("ad_end_last_10s_vs_post_ad_10s", "ad_end_last_10s", "post_ad_10s"),
        ("ad_end_minus10to_minus5s_vs_ad_end_last_5s", "ad_end_minus10to_minus5s", "ad_end_last_5s"),
        ("ad_start_first_10s_vs_random_non_ad_30s", "ad_start_first_10s", "random_non_ad_30s"),
        ("ad_end_last_10s_vs_random_non_ad_30s", "ad_end_last_10s", "random_non_ad_30s"),
    ]
    rows = []
    for name, a_type, b_type in comparisons:
        a_df = combined[combined["segment_type"].eq(a_type)]
        b_df = combined[combined["segment_type"].eq(b_type)]
        for feature in features:
            a = pd.to_numeric(a_df[feature], errors="coerce")
            b = pd.to_numeric(b_df[feature], errors="coerce")
            af = a[np.isfinite(a)]
            bf = b[np.isfinite(b)]
            d = cohen_d(a, b)
            mean_diff = af.mean() - bf.mean() if len(af) and len(bf) else np.nan
            direction = "higher_in_a" if np.isfinite(d) and d >= 0.2 else "lower_in_a" if np.isfinite(d) and d <= -0.2 else "weak_or_unclear"
            rows.append(
                {
                    "comparison_name": name,
                    "feature_name": feature,
                    "feature_group": feature_group(feature),
                    "segment_type_a": a_type,
                    "segment_type_b": b_type,
                    "n_a": int(len(af)),
                    "n_b": int(len(bf)),
                    "mean_a": safe_float(af.mean()) if len(af) else np.nan,
                    "mean_b": safe_float(bf.mean()) if len(bf) else np.nan,
                    "median_a": safe_float(af.median()) if len(af) else np.nan,
                    "median_b": safe_float(bf.median()) if len(bf) else np.nan,
                    "mean_diff": safe_float(mean_diff),
                    "median_diff": safe_float(af.median() - bf.median()) if len(af) and len(bf) else np.nan,
                    "cohen_d": safe_float(d),
                    "robust_effect_size": robust_effect(a, b),
                    "direction": direction,
                    "interpretation_hint": f"{a_type}와 {b_type}의 boundary-adjacent audio feature 차이",
                }
            )
    pairwise = pd.DataFrame(rows)
    delta_rows = []
    for (name, feature), group in pairwise.groupby(["comparison_name", "feature_name"]):
        delta_rows.append(
            {
                "comparison_name": name,
                "feature_name": feature,
                "feature_group": feature_group(feature),
                "cohen_d": safe_float(group["cohen_d"].iloc[0]),
                "robust_effect_size": safe_float(group["robust_effect_size"].iloc[0]),
                "median_diff": safe_float(group["median_diff"].iloc[0]),
                "direction": group["direction"].iloc[0],
            }
        )
    return pairwise, pd.DataFrame(delta_rows)


def feature_group(feature: str) -> str:
    if feature in {"onset_density", "onset_count_per_sec"} or feature.startswith("onset_"):
        return "Onset"
    if feature.startswith("spectral_flux"):
        return "Spectral Flux"
    if feature.startswith("audio_"):
        return "Energy / Amplitude"
    if feature in {"silence_ratio", "low_energy_ratio"}:
        return "Silence / Low Energy"
    if feature.startswith("spectral_"):
        return "Spectral"
    if feature.startswith("zero_crossing"):
        return "Zero Crossing"
    if feature.startswith("mfcc"):
        return "MFCC"
    return "Other"


def build_rule_design(thresholds: Dict[str, Any], context_scores: pd.DataFrame, boundary_df: pd.DataFrame) -> pd.DataFrame:
    median_score = safe_float(context_scores["audio_ad_like_score_median"].median()) if not context_scores.empty else 0.6
    ad_like_threshold = thresholds["audio_ad_like_score_threshold"]
    delta_threshold = safe_float(boundary_df["score_delta_for_boundary_direction"].quantile(0.25)) if not boundary_df.empty else 0.05
    delta_threshold = max(delta_threshold, 0.05) if np.isfinite(delta_threshold) else 0.05
    rows = [
        {
            "rule_id": "audio_start_persistence_v2_4",
            "boundary_type": "ad_start",
            "rule_name": "start post-10s ad-like persistence",
            "rule_purpose": "scene candidate 이후 10초 동안 광고스러운 오디오 패턴이 유지되는지 확인",
            "required_context": "pre_10s and post_10s around candidate time t",
            "feature_or_score": "audio_ad_like_score, ad_like_ratio, max_consecutive_ad_like_sec, score_delta",
            "direction": "post_10s high and post_10s > pre_10s",
            "suggested_threshold": f"post_ad_like_ratio >= 0.6; post_max_consecutive_ad_like_sec >= 6; post_median_score >= {ad_like_threshold:.3f}; post-pre median delta >= {delta_threshold:.3f}",
            "threshold_source": "labeled v2_4 edge/subwindow exploratory score distribution",
            "persistence_condition": "post_10s_audio_ad_like_ratio >= 0.6 AND post_10s_max_consecutive_ad_like_sec >= 6",
            "delta_condition": f"post_10s_median_score - pre_10s_median_score >= {delta_threshold:.3f}",
            "confidence_level": "medium",
            "caution": "exploratory rule candidate; combine with visual/OCR/scene cue; not final performance",
            "plain_korean_interpretation": "광고 시작 후보 t 이후 10초 동안 광고스러운 오디오가 여러 2초 subwindow에서 지속되면 시작 boundary 신뢰도를 올린다.",
        },
        {
            "rule_id": "audio_end_persistence_v2_4",
            "boundary_type": "ad_end",
            "rule_name": "end pre-10s persistence and post drop",
            "rule_purpose": "scene candidate 이전 10초는 광고스럽고 이후 10초에서 낮아지는지 확인",
            "required_context": "pre_10s and post_10s around candidate time t",
            "feature_or_score": "audio_ad_like_score, ad_like_ratio, max_consecutive_ad_like_sec, score_delta",
            "direction": "pre_10s high and pre_10s > post_10s",
            "suggested_threshold": f"pre_ad_like_ratio >= 0.6; pre_max_consecutive_ad_like_sec >= 6; pre_median_score >= {ad_like_threshold:.3f}; pre-post median delta >= {delta_threshold:.3f}",
            "threshold_source": "labeled v2_4 edge/subwindow exploratory score distribution",
            "persistence_condition": "pre_10s_audio_ad_like_ratio >= 0.6 AND pre_10s_max_consecutive_ad_like_sec >= 6",
            "delta_condition": f"pre_10s_median_score - post_10s_median_score >= {delta_threshold:.3f}",
            "confidence_level": "medium",
            "caution": "exploratory rule candidate; combine with visual/OCR/scene cue; not final performance",
            "plain_korean_interpretation": "광고 종료 후보 t 이전 10초가 광고스럽게 지속되고 t 이후 낮아지면 종료 boundary 신뢰도를 올린다.",
        },
    ]
    return pd.DataFrame(rows)


def write_rule_config(path: Path, input_files: Dict[str, Any], thresholds: Dict[str, Any]) -> None:
    config = {
        "version": VERSION,
        "created_at": now_iso(),
        "input_files": input_files,
        "selected_features": HIGHER_FEATURES + LOWER_FEATURES,
        "feature_directions": {
            **{feature: "higher_indicates_ad" for feature in HIGHER_FEATURES},
            **{feature: "lower_indicates_ad" for feature in LOWER_FEATURES},
        },
        "feature_weights": SCORE_WEIGHTS,
        "score_components": SCORE_COMPONENTS,
        "ad_like_score_formula": "0.30*onset_density_score + 0.25*flux_onset_score + 0.20*energy_score + 0.15*inverse_silence_score + 0.10*spectral_texture_score",
        "persistence_window_sec": 10,
        "subwindow_size_sec": SUBWINDOW_SIZE_SEC,
        "subwindow_stride_sec": SUBWINDOW_STRIDE_SEC,
        "start_boundary_rule": {
            "post_10s_audio_ad_like_ratio_min": 0.6,
            "post_10s_max_consecutive_ad_like_sec_min": 6,
            "post_10s_median_score_min": thresholds["audio_ad_like_score_threshold"],
            "post_minus_pre_median_delta_min": thresholds["score_delta_threshold"],
        },
        "end_boundary_rule": {
            "pre_10s_audio_ad_like_ratio_min": 0.6,
            "pre_10s_max_consecutive_ad_like_sec_min": 6,
            "pre_10s_median_score_min": thresholds["audio_ad_like_score_threshold"],
            "pre_minus_post_median_delta_min": thresholds["score_delta_threshold"],
        },
        "default_thresholds": thresholds,
        "threshold_source": "v2_4 labeled ad interval edge/subwindow exploratory analysis",
        "caveats": [
            "Exploratory rule candidate, not final detector performance.",
            "Audio cue must be combined with visual/OCR/scene-change evidence.",
            "Scene candidate false-positive reduction needs separate validation.",
            "2s subwindows can be noisy.",
        ],
        "recommended_next_integration": "Compute the same pre/post 10s persistence score around scene candidates and fuse with scene/OCR/visual cue scores.",
    }
    save_json(path, config)


def write_rule_design_md(path: Path, rule_design: pd.DataFrame, thresholds: Dict[str, Any]) -> None:
    lines = [
        "# Audio Persistence Rule Design v2_4",
        "",
        "This document describes persistence-based audio cue design for rule-based detector support.",
        "It is an exploratory rule candidate, not final detector performance.",
        "",
        "## Audio Ad-like Score",
        "- onset_density_score: onset_density, onset_count_per_sec",
        "- flux_onset_score: spectral_flux_mean, onset_strength_mean, spectral_flux_max, onset_strength_max",
        "- energy_score: audio_log_energy_mean, audio_rms_mean, audio_mean_abs_amplitude",
        "- inverse_silence_score: inverse direction of silence_ratio and low_energy_ratio",
        "- spectral_texture_score: inverse direction of spectral_flatness_std, spectral_bandwidth_mean, spectral_centroid_std",
        "",
        f"Suggested exploratory audio_ad_like_score threshold: {thresholds['audio_ad_like_score_threshold']:.3f}",
        "",
        "## Rules",
    ]
    for _, row in rule_design.iterrows():
        lines += [
            f"### {row['rule_id']}",
            f"- purpose: {row['rule_purpose']}",
            f"- condition: {row['persistence_condition']}",
            f"- delta: {row['delta_condition']}",
            f"- caution: {row['caution']}",
            f"- interpretation: {row['plain_korean_interpretation']}",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def forbidden_latest_files() -> List[str]:
    if not LATEST_DIR.exists():
        return []
    forbidden = []
    for path in LATEST_DIR.rglob("*"):
        if path.is_dir() and path.name in {"cache", "tmp"}:
            forbidden.append(str(path))
        if path.is_file() and path.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES:
            forbidden.append(str(path))
    return forbidden


def update_latest(paths: Dict[str, Path], report: Dict[str, Any], backup_dir: Optional[Path]) -> List[str]:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    include_keys = {
        "edge_plan",
        "edge_features",
        "context_scores",
        "boundary_scores",
        "edge_pairwise",
        "edge_delta_summary",
        "rule_design_csv",
        "rule_config",
        "rule_design_md",
        "report",
        "summary",
        "run_log",
        "script",
        "score_components",
        "subwindow_features",
    }
    allowed = {paths[k].name for k in include_keys if k in paths} | {"README_latest_files.md"}
    archived = []
    cleanup_dir = (backup_dir or BACKUPS_DIR / f"{TASK_NAME}_latest_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}") / "latest_for_chatgpt_a_previous"
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
    for key in include_keys:
        path = paths[key]
        if not path.exists() or path.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES:
            continue
        if path.stat().st_size > 5_000_000 and key in {"score_components", "subwindow_features"}:
            report.setdefault("latest_for_chatgpt_a_skipped_large_files", []).append(str(path))
            continue
        dest = LATEST_DIR / path.name
        shutil.copy2(path, dest)
        copied.append(str(dest))
    readme = LATEST_DIR / "README_latest_files.md"
    lines = [
        "# Latest Files",
        "",
        f"- task_name: {TASK_NAME}",
        f"- version: {VERSION}",
        f"- selected label file: {LABEL_FILE}",
        "- latest path override: outputs/latest_for_chatgpt_a",
        f"- old_project_modified: {str(report.get('old_project_modified')).lower()}",
        "- forbidden media/model/cache files included: false",
        "",
        "This bundle is for persistence-based audio cue design and boundary-adjacent audio feature analysis.",
        "It is not a final detector performance claim.",
        "",
        "## Included Files",
    ]
    for path in sorted(copied):
        lines.append(f"- {Path(path).name}")
    if report.get("latest_for_chatgpt_a_skipped_large_files"):
        lines.append("")
        lines.append("## Large Files Not Copied")
        for path in report["latest_for_chatgpt_a_skipped_large_files"]:
            lines.append(f"- {path}")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copied.append(str(readme))
    return sorted(copied)


def write_markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows_"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                vals.append("" if not np.isfinite(value) else f"{value:.4g}")
            else:
                vals.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(path: Path, report: Dict[str, Any], edge_pairwise_df: pd.DataFrame, boundary_df: pd.DataFrame) -> str:
    start_rows = edge_pairwise_df[edge_pairwise_df["comparison_name"].isin(["pre_ad_10s_vs_ad_start_first_5s", "pre_ad_10s_vs_ad_start_first_10s", "ad_start_first_5s_vs_ad_start_5to10s"])].copy()
    end_rows = edge_pairwise_df[edge_pairwise_df["comparison_name"].isin(["ad_end_last_5s_vs_post_ad_10s", "ad_end_last_10s_vs_post_ad_10s", "ad_end_minus10to_minus5s_vs_ad_end_last_5s"])].copy()
    start_top = start_rows.sort_values("cohen_d", key=lambda s: s.abs(), ascending=False).head(10).to_dict("records")
    end_top = end_rows.sort_values("cohen_d", key=lambda s: s.abs(), ascending=False).head(10).to_dict("records")
    start_score_mean = safe_float(boundary_df[boundary_df["boundary_type"].eq("ad_start")]["audio_boundary_persistence_score"].mean())
    end_score_mean = safe_float(boundary_df[boundary_df["boundary_type"].eq("ad_end")]["audio_boundary_persistence_score"].mean())
    sub_lines = [
        f"- {name}: {result.get('status')} (warnings={len(result.get('warnings', []))}, errors={len(result.get('errors', []))})"
        for name, result in report.get("sub_agent_results", {}).items()
    ]
    outputs = [f"- {p}" for p in report.get("output_files", [])]
    text = "\n".join(
        [
            "# Extract Audio Ad Edge Persistence Summary",
            "",
            "## 1. 작업 개요",
            "v2_4 광고 interval 라벨을 기준으로 광고 시작/종료 edge 5초·10초 audio edge feature extraction을 수행했다.",
            "또한 10초 context를 2초 subwindow로 나누어 persistence-based audio cue design을 수행했다.",
            "이번 결과는 final detector 성능이 아니라 rule-based detector support feature와 exploratory rule candidate 설계용이다.",
            "",
            "## 2. 사용한 입력 파일",
            f"- 광고 interval 라벨: {LABEL_FILE}",
            f"- video manifest: {MANIFEST_FILE}",
            f"- raw video directory: {RAW_VIDEO_DIR}",
            *[f"- previous {k}: {v}" for k, v in PREVIOUS_FILES.items() if Path(v).exists()],
            "",
            "## 3. 생성한 Edge Segment",
            "- ad_start_first_5s: 광고 시작 직후 5초, 즉각적인 오디오 변화 확인",
            "- ad_start_first_10s: 광고 시작 이후 10초, 광고스러운 패턴 지속 여부 확인",
            "- ad_start_5to10s: 시작 후 5~10초, 첫 5초 이후에도 패턴이 유지되는지 확인",
            "- ad_end_last_5s: 광고 종료 직전 5초, 종료 직전 오디오 특성 확인",
            "- ad_end_last_10s: 광고 종료 직전 10초, 종료 직전 광고스러운 패턴 지속 확인",
            "- ad_end_minus10to_minus5s: 종료 10~5초 전, 마지막 5초 이전에도 패턴이 유지되는지 확인",
            "",
            "## 4. Edge Feature Extraction 결과",
            f"- interval count: {report.get('ad_interval_count')}",
            f"- edge segment count: {report.get('actual_edge_segment_count')}",
            f"- edge feature success/failure: {report.get('edge_feature_success_count')} / {report.get('edge_feature_failed_count')}",
            f"- short under 5s / under 10s: {report.get('short_ad_count_under_5s')} / {report.get('short_ad_count_under_10s')}",
            f"- duration truncated rows: {report.get('duration_truncated_count')}",
            "",
            "## 5. Persistence Subwindow 분석",
            "10초 context를 2초 단위 subwindow로 나눈 이유는 평균값 하나가 아니라 광고스러운 오디오 패턴이 일정 시간 지속되는지 보기 위해서다.",
            f"- subwindow count: {report.get('actual_subwindow_count')}",
            f"- subwindow feature success/failure: {report.get('subwindow_feature_success_count')} / {report.get('subwindow_feature_failed_count')}",
            f"- persistence context score rows: {report.get('persistence_context_score_count')}",
            f"- boundary persistence score rows: {report.get('boundary_persistence_score_count')}",
            "",
            "## 6. 5초 vs 10초 분석 결과",
            "광고 시작 쪽은 pre_ad_10s 대비 start_first_5s/start_first_10s에서 onset_density, spectral_flux, onset_strength 계열 변화가 주요 후보로 확인된다.",
            "광고 종료 쪽은 end_last_10s/end_last_5s와 post_ad_10s 차이를 통해 종료 후 ad-like score가 낮아지는지 보는 구조가 더 적합하다.",
            "ad_start_5to10s와 first_5s 비교, ad_end_minus10to_minus5s와 last_5s 비교는 광고스러운 오디오가 첫/마지막 5초에만 튀는지 또는 10초 안에서 유지되는지를 보는 persistence 보조 근거다.",
            "",
            "### Start-side top differences",
            write_markdown_table(start_top, ["comparison_name", "feature_name", "feature_group", "cohen_d", "direction", "median_diff"]),
            "",
            "### End-side top differences",
            write_markdown_table(end_top, ["comparison_name", "feature_name", "feature_group", "cohen_d", "direction", "median_diff"]),
            "",
            "## 7. 주요 Audio Cue 결과",
            "- onset_density/onset_count_per_sec: segment duration 영향을 줄이기 위해 onset_count보다 우선 사용한다.",
            "- spectral_flux_mean/onset_strength_mean: 2초 subwindow에서 BGM/효과음/편집 변화 지속성을 보는 핵심 cue다.",
            "- audio_log_energy_mean/audio_rms_mean: 음량/에너지 보조 cue로 사용한다.",
            "- silence_ratio/low_energy_ratio: 낮을수록 광고성 설명/BGM이 끊김 없이 지속될 가능성을 보는 inverse cue다.",
            "- spectral texture features: 해석 주의가 필요하므로 낮은 가중치로 둔다.",
            "",
            "## 8. Persistence Rule 초안",
            f"- average start boundary persistence score: {start_score_mean:.3f}",
            f"- average end boundary persistence score: {end_score_mean:.3f}",
            f"- exploratory ad-like threshold: {report.get('suggested_start_rule', {}).get('audio_ad_like_score_threshold')}",
            "- start rule: candidate t 이후 10초의 ad_like_ratio, max_consecutive_ad_like_sec, median/p25 score가 높고 pre→post score delta가 양수이면 start audio confidence를 올린다.",
            "- end rule: candidate t 이전 10초의 ad_like_ratio, max_consecutive_ad_like_sec, median/p25 score가 높고 pre→post score가 낮아지면 end audio confidence를 올린다.",
            "- audio_ad_like_score = 0.30 onset_density + 0.25 flux/onset + 0.20 energy + 0.15 inverse silence + 0.10 spectral texture component 초안.",
            "",
            "## 9. 추후 Scene Candidate 적용 방식",
            "scene candidate time t를 기준으로 start 후보는 t 이후 10초가 광고스럽게 지속되는지 본다.",
            "end 후보는 t 이전 10초가 광고스럽고 t 이후 10초에서 낮아지는지 본다.",
            "audio cue는 단독 판단이 아니라 visual/OCR/scene-change cue와 결합해야 한다.",
            "",
            "## 10. 한계",
            "- sample size가 작아 threshold는 exploratory candidate다.",
            "- scene candidate 전체 적용 전 false positive 감소 효과는 별도 검증이 필요하다.",
            "- audio cue 단독 판단은 금지한다.",
            "- 2초 subwindow는 짧아 noisy할 수 있다.",
            "- 광고 길이가 짧은 경우 edge segment overlap이 가능하다.",
            "",
            "## 11. 다음 작업 제안",
            "- scene candidate 주변 10초 persistence score 계산",
            "- audio persistence score와 scene_change_score 결합",
            "- OCR/visual cue와 rule fusion",
            "- false positive 감소 여부 검증",
            "",
            "## 12. Sub Agent 검증 결과",
            *sub_lines,
            "",
            "## 13. 생성 파일 목록",
            *outputs,
            "",
            f"- old_project_modified: {report.get('old_project_modified')}",
            f"- input_label_file_modified: {report.get('input_label_file_modified')}",
            f"- input_feature_files_modified: {report.get('input_feature_files_modified')}",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")
    return text


def validate_all(report: Dict[str, Any], paths: Dict[str, Path], summary_text: str) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    errors: List[str] = []
    if not Path(report["selected_label_file"]).exists():
        errors.append("selected label file missing")
    if report["ad_interval_count"] <= 0:
        errors.append("no ad intervals")
    if report["audio_stream_success_count"] < max(1, int(report["video_count"] * 0.8)):
        warnings.append("some videos did not validate with audio stream")
    if report["input_label_file_modified"] or report["input_feature_files_modified"]:
        errors.append("input file modification detected")
    results["input_schema_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = []
    errors = []
    if report["actual_edge_segment_count"] != report["expected_edge_segment_count"]:
        errors.append("edge segment count does not match expected")
    if report["edge_sampling_invalid_count"]:
        errors.append("edge sampling contains invalid rows")
    if report["duration_truncated_count"]:
        warnings.append("some edge segments were truncated by ad duration or video boundary")
    results["edge_sampling_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = []
    errors = []
    if report["edge_feature_success_count"] == 0 or report["subwindow_feature_success_count"] == 0:
        errors.append("feature extraction produced no successful rows")
    if not report["onset_density_available"]:
        errors.append("onset_density missing")
    if report["score_range_min"] < -1e-6 or report["score_range_max"] > 1 + 1e-6:
        errors.append("audio_ad_like_score out of 0-1 range")
    if report["subwindow_feature_failed_count"]:
        warnings.append("some subwindow feature rows failed")
    warnings.append("tempo_estimate remains NaN/unstable for short edge/subwindow segments")
    results["audio_feature_normalization_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = []
    errors = []
    if report["persistence_context_score_count"] == 0 or report["boundary_persistence_score_count"] == 0:
        errors.append("persistence outputs are empty")
    if "final detector 성능이 아니라" not in summary_text:
        errors.append("exploratory caveat missing")
    if report["persistence_threshold_source"] == "fallback_default_0.6":
        warnings.append("persistence threshold used fallback default")
    warnings.append("subwindow count is small per context; persistence rule is exploratory")
    results["persistence_rule_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}

    warnings = []
    errors = []
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        errors.append(f"missing outputs: {missing}")
    forbidden = forbidden_latest_files()
    if forbidden:
        errors.append(f"forbidden latest_for_chatgpt_a files: {forbidden}")
    if report["old_project_modified"]:
        errors.append("old project modified")
    if report["backup_dir"]:
        warnings.append("existing outputs were backed up or latest README was updated")
    results["output_safety_validation"] = {"status": "FAIL" if errors else ("WARN" if warnings else "PASS"), "warnings": warnings, "errors": errors}
    return results


def main() -> int:
    start_epoch = time.time()
    start_time = now_iso()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = output_paths()
    for directory in [DATA_AUDIO_DIR, REPORTS_DIR, LOGS_DIR, SCRIPTS_AUDIO_DIR, CONFIGS_DIR, LATEST_DIR, BACKUPS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_existing(paths, timestamp)
    logger = TaskLogger(paths["run_log"])
    warnings: List[str] = []
    errors: List[str] = []
    if backup_dir:
        warnings.append(f"existing outputs backed up to {backup_dir}")
    logger.log("[STEP 01] Start task and create old project snapshot")
    logger.log(f"Task start time: {start_time}")
    old_before = old_project_snapshot()

    input_mtimes = {str(path): path.stat().st_mtime_ns for path in [LABEL_FILE, MANIFEST_FILE, *PREVIOUS_FILES.values()] if path.exists()}
    logger.log("[STEP 02] Discover latest valid ad interval label file")
    if not LABEL_FILE.exists():
        raise RuntimeError(f"Required v2_4 label file missing: {LABEL_FILE}")
    intervals = canonical_intervals(LABEL_FILE)
    label_mtime = iso_from_timestamp(LABEL_FILE.stat().st_mtime)
    logger.log(f"Selected label file: {LABEL_FILE}")
    logger.log("Selected label reason: v2_4 preferred ad_interval_segments file exists")
    logger.log(f"Input interval row count: {len(intervals)}")

    logger.log("[STEP 03] Load previous audio feature/rule analysis files")
    missing_previous = [str(path) for path in PREVIOUS_FILES.values() if not path.exists()]
    previous_features = read_csv(PREVIOUS_FILES["labeled_features"])
    for key, path in PREVIOUS_FILES.items():
        logger.log(f"Previous input {key}: {path} exists={path.exists()}")
    if missing_previous:
        warnings.append(f"missing optional previous files: {missing_previous}")

    logger.log("[STEP 04] Resolve raw video paths from manifest")
    video_paths = resolve_video_paths(intervals, MANIFEST_FILE)

    logger.log("[STEP 07] Validate ffmpeg/ffprobe and audio streams")
    ffmpeg_path = executable_path("ffmpeg")
    ffprobe_path = executable_path("ffprobe")
    probe_df = probe_videos(video_paths, ffprobe_path)
    audio_stream_success = int(probe_df["audio_available"].sum())
    audio_stream_failed = int(len(probe_df) - audio_stream_success)
    failed_video_ids = probe_df.loc[~probe_df["audio_available"], "video_id"].astype(str).tolist()
    logger.log(f"Video/audio stream success count: {audio_stream_success}; failed count: {audio_stream_failed}")

    logger.log("[STEP 05] Build 5s/10s ad edge segment sampling plan")
    edge_plan = build_edge_plan(intervals, video_paths, probe_df, label_mtime)
    edge_plan.to_csv(paths["edge_plan"], index=False, encoding="utf-8-sig")
    logger.log(f"Edge segment row count: {len(edge_plan)}")

    logger.log("[STEP 06] Build 2s subwindow plan for persistence analysis")
    subwindow_plan = build_subwindow_plan(intervals, video_paths, probe_df)
    subwindow_plan.to_csv(paths["subwindow_plan"], index=False, encoding="utf-8-sig")
    logger.log(f"Subwindow row count: {len(subwindow_plan)}")

    logger.log("[STEP 08] Extract audio features for edge segments")
    edge_features = extract_features(edge_plan, "segment_id", "segment_start_sec", "segment_duration_sec", ffmpeg_path, probe_df, logger, "edge")
    edge_features.to_csv(paths["edge_features"], index=False, encoding="utf-8-sig")
    edge_success = int((edge_features["feature_status"] == "success").sum())
    edge_failed = int(len(edge_features) - edge_success)
    logger.log(f"Edge feature success/failure count: {edge_success}/{edge_failed}")

    logger.log("[STEP 09] Extract subwindow features for persistence analysis")
    subwindow_features = extract_features(subwindow_plan, "subwindow_id", "subwindow_start_sec", "subwindow_duration_sec", ffmpeg_path, probe_df, logger, "subwindow")
    subwindow_features.to_csv(paths["subwindow_features"], index=False, encoding="utf-8-sig")
    sub_success = int((subwindow_features["feature_status"] == "success").sum())
    sub_failed = int(len(subwindow_features) - sub_success)
    logger.log(f"Subwindow feature success/failure count: {sub_success}/{sub_failed}")

    logger.log("[STEP 10] Compute duration-normalized features")
    if "onset_density" not in previous_features.columns:
        previous_features["onset_density"] = pd.to_numeric(previous_features["onset_count"], errors="coerce") / pd.to_numeric(previous_features["segment_duration_sec"], errors="coerce").replace(0, np.nan)
        previous_features["onset_count_per_sec"] = previous_features["onset_density"]
    onset_threshold = threshold_on_onset_density(previous_features)

    logger.log("[STEP 11] Compute ad-like audio scores")
    baseline_for_scoring = pd.concat(
        [
            previous_features[previous_features["segment_type"].isin(["ad_full", "random_non_ad_30s"])],
            edge_features.rename(columns={"segment_duration_sec": "segment_duration_sec"}),
        ],
        ignore_index=True,
        sort=False,
    )
    score_components = add_ad_like_scores(
        pd.concat(
            [
                edge_features.assign(row_source="edge_segment"),
                subwindow_features.assign(row_source="persistence_subwindow"),
            ],
            ignore_index=True,
            sort=False,
        ),
        baseline_for_scoring,
    )
    score_components.to_csv(paths["score_components"], index=False, encoding="utf-8-sig")
    sub_scores = score_components[score_components["row_source"].eq("persistence_subwindow")].copy()

    random_scores = sub_scores[sub_scores["context_type"].eq("random_non_ad_30s")]["audio_ad_like_score"]
    ad_context_scores = sub_scores[sub_scores["context_type"].isin(["start_post_10s", "end_pre_10s"])]["audio_ad_like_score"]
    if random_scores.notna().sum() >= 5 and ad_context_scores.notna().sum() >= 5:
        threshold = safe_float(np.nanmean([random_scores.quantile(0.75), ad_context_scores.quantile(0.25)]))
        threshold_source = "mean(random_non_ad_subwindow_p75, ad_edge_subwindow_p25)"
    else:
        threshold = 0.6
        threshold_source = "fallback_default_0.6"
        warnings.append("ad-like persistence threshold used fallback 0.6")
    threshold = safe_float(np.clip(threshold, 0.35, 0.75)) if np.isfinite(threshold) else 0.6

    logger.log("[STEP 12] Compute persistence metrics")
    sub_scores.attrs["onset_density_threshold"] = onset_threshold["best_threshold"]
    context_scores = compute_context_scores(sub_scores, threshold)
    context_scores.to_csv(paths["context_scores"], index=False, encoding="utf-8-sig")
    boundary_df = boundary_scores(context_scores, intervals)
    boundary_df.to_csv(paths["boundary_scores"], index=False, encoding="utf-8-sig")
    logger.log(f"Persistence context score row count: {len(context_scores)}")
    logger.log(f"Boundary persistence score row count: {len(boundary_df)}")

    logger.log("[STEP 13] Generate persistence-based rule design")
    thresholds = {
        "onset_density_threshold": onset_threshold["best_threshold"],
        "onset_density_threshold_source": onset_threshold["threshold_source"],
        "audio_ad_like_score_threshold": threshold,
        "audio_ad_like_score_threshold_source": threshold_source,
        "score_delta_threshold": safe_float(max(boundary_df["score_delta_for_boundary_direction"].quantile(0.25), 0.05)) if not boundary_df.empty else 0.05,
    }
    rule_design = build_rule_design(thresholds, context_scores, boundary_df)
    rule_design.to_csv(paths["rule_design_csv"], index=False, encoding="utf-8-sig")
    write_rule_config(paths["rule_config"], {**{k: str(v) for k, v in PREVIOUS_FILES.items()}, "label_file": str(LABEL_FILE), "manifest_file": str(MANIFEST_FILE)}, thresholds)
    write_rule_design_md(paths["rule_design_md"], rule_design, thresholds)
    logger.log(f"Rule design row count: {len(rule_design)}")

    edge_pairwise_df, edge_delta_summary = edge_pairwise(edge_features, previous_features)
    edge_pairwise_df.to_csv(paths["edge_pairwise"], index=False, encoding="utf-8-sig")
    edge_delta_summary.to_csv(paths["edge_delta_summary"], index=False, encoding="utf-8-sig")

    output_list = [str(path) for path in paths.values()]
    input_feature_modified = any(path.exists() and path.stat().st_mtime_ns != input_mtimes.get(str(path), path.stat().st_mtime_ns) for path in PREVIOUS_FILES.values())
    label_modified = LABEL_FILE.stat().st_mtime_ns != input_mtimes.get(str(LABEL_FILE), LABEL_FILE.stat().st_mtime_ns)
    old_after = old_project_snapshot()
    old_modified = old_before != old_after
    report: Dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "task_name": TASK_NAME,
        "version": VERSION,
        "start_time": start_time,
        "end_time": None,
        "actual_runtime_seconds": None,
        "actual_runtime_readable": None,
        "input_files": {
            "selected_label_file": str(LABEL_FILE),
            "video_manifest": str(MANIFEST_FILE),
            "raw_video_directory": str(RAW_VIDEO_DIR),
            "previous_audio_feature_rule_files": {k: str(v) for k, v in PREVIOUS_FILES.items()},
        },
        "output_files": output_list,
        "generated_files": output_list,
        "latest_for_chatgpt_files": [],
        "missing_input_files": missing_previous,
        "missing_required_columns": [],
        "warnings": warnings,
        "errors": errors,
        "old_project_modified": old_modified,
        "old_project_snapshot_before": old_before,
        "old_project_snapshot_after": old_after,
        "sub_agent_results": {},
        "selected_label_file": str(LABEL_FILE),
        "selected_label_file_modified_time": label_mtime,
        "selected_feature_files": {k: str(v) for k, v in PREVIOUS_FILES.items() if v.exists()},
        "ad_interval_count": int(len(intervals)),
        "video_count": int(intervals["video_id"].nunique()),
        "expected_edge_segment_count": int(len(intervals) * 6),
        "actual_edge_segment_count": int(len(edge_plan)),
        "expected_subwindow_count": int(len(intervals) * 20 + len(previous_features[previous_features["segment_type"].eq("random_non_ad_30s")]) * 15),
        "actual_subwindow_count": int(len(subwindow_plan)),
        "ffmpeg_available": bool(ffmpeg_path),
        "ffprobe_available": bool(ffprobe_path),
        "audio_stream_success_count": audio_stream_success,
        "audio_stream_failed_count": audio_stream_failed,
        "failed_video_ids": failed_video_ids,
        "edge_feature_success_count": edge_success,
        "edge_feature_failed_count": edge_failed,
        "subwindow_feature_success_count": sub_success,
        "subwindow_feature_failed_count": sub_failed,
        "duration_truncated_count": int(edge_plan["segment_truncated_by_ad_duration"].sum() + edge_plan["segment_truncated_by_video_boundary"].sum()),
        "short_ad_count_under_5s": int(intervals["ad_duration_sec"].lt(5).sum()),
        "short_ad_count_under_10s": int(intervals["ad_duration_sec"].lt(10).sum()),
        "edge_sampling_invalid_count": int((edge_plan["sampling_status"] != "planned").sum()),
        "ad_like_score_feature_count": int(len(HIGHER_FEATURES) + len(LOWER_FEATURES)),
        "persistence_context_score_count": int(len(context_scores)),
        "boundary_persistence_score_count": int(len(boundary_df)),
        "pairwise_comparison_row_count": int(len(edge_pairwise_df)),
        "rule_design_row_count": int(len(rule_design)),
        "selected_audio_cue_features": HIGHER_FEATURES + LOWER_FEATURES,
        "selected_persistence_metrics": ["ad_like_ratio", "max_consecutive_ad_like_sec", "audio_ad_like_score_median", "audio_ad_like_score_p25", "score_delta"],
        "suggested_start_rule": {
            "audio_ad_like_score_threshold": threshold,
            "post_ad_like_ratio_min": 0.6,
            "post_max_consecutive_ad_like_sec_min": 6,
            "post_minus_pre_median_delta_min": thresholds["score_delta_threshold"],
        },
        "suggested_end_rule": {
            "audio_ad_like_score_threshold": threshold,
            "pre_ad_like_ratio_min": 0.6,
            "pre_max_consecutive_ad_like_sec_min": 6,
            "pre_minus_post_median_delta_min": thresholds["score_delta_threshold"],
        },
        "nan_ratio_by_feature": {col: safe_float(score_components[col].isna().mean()) for col in ["audio_ad_like_score", "onset_density", "spectral_flux_mean", "silence_ratio"] if col in score_components.columns},
        "inf_count_by_feature": {col: int(np.isinf(pd.to_numeric(score_components[col], errors="coerce")).sum()) for col in ["audio_ad_like_score", "onset_density", "spectral_flux_mean", "silence_ratio"] if col in score_components.columns},
        "constant_features": [],
        "onset_density_available": "onset_density" in edge_features.columns and "onset_density" in subwindow_features.columns,
        "score_range_min": safe_float(score_components["audio_ad_like_score"].min()),
        "score_range_max": safe_float(score_components["audio_ad_like_score"].max()),
        "ad_like_ratio_range_min": safe_float(context_scores["ad_like_ratio"].min()),
        "ad_like_ratio_range_max": safe_float(context_scores["ad_like_ratio"].max()),
        "persistence_threshold_source": threshold_source,
        "latest_for_chatgpt_forbidden_files_found": [],
        "input_label_file_modified": label_modified,
        "input_feature_files_modified": bool(input_feature_modified),
        "backup_dir": str(backup_dir) if backup_dir else "",
    }
    if old_modified:
        errors.append("old project modified during task")
    if label_modified:
        errors.append("input label file modified during task")
    if input_feature_modified:
        errors.append("input feature/rule file modified during task")

    logger.log("[STEP 14] Run Sub Agent validations")
    summary_text = write_summary(paths["summary"], report, edge_pairwise_df, boundary_df)
    save_json(paths["report"], report)
    report["sub_agent_results"] = validate_all(report, paths, summary_text)
    for name, result in report["sub_agent_results"].items():
        logger.log(f"Sub Agent {name}: {result['status']} warnings={len(result.get('warnings', []))} errors={len(result.get('errors', []))}")
        for warning in result.get("warnings", []):
            warnings.append(f"{name}: {warning}")
        for error in result.get("errors", []):
            errors.append(f"{name}: {error}")
    report["warnings"] = warnings
    report["errors"] = errors
    summary_text = write_summary(paths["summary"], report, edge_pairwise_df, boundary_df)
    save_json(paths["report"], report)

    logger.log("[STEP 15] Update latest_for_chatgpt with allowed files only")
    report["latest_for_chatgpt_files"] = update_latest(paths, report, backup_dir)
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden_latest_files()
    end_time = now_iso()
    runtime = time.time() - start_epoch
    report["end_time"] = end_time
    report["actual_runtime_seconds"] = round(runtime, 3)
    report["actual_runtime_readable"] = readable_seconds(runtime)
    save_json(paths["report"], report)
    shutil.copy2(paths["report"], LATEST_DIR / paths["report"].name)
    shutil.copy2(paths["summary"], LATEST_DIR / paths["summary"].name)
    shutil.copy2(paths["run_log"], LATEST_DIR / paths["run_log"].name)

    logger.log(f"Task end time: {end_time}")
    logger.log(f"Actual runtime: {report['actual_runtime_readable']}")
    logger.log(f"Warnings: {warnings}")
    logger.log(f"Errors: {errors}")
    logger.log("[STEP 16] Print human-readable final summary")
    shutil.copy2(paths["run_log"], LATEST_DIR / paths["run_log"].name)

    status = "실패" if errors or any(r["status"] == "FAIL" for r in report["sub_agent_results"].values()) else "조건부 성공" if warnings or any(r["status"] == "WARN" for r in report["sub_agent_results"].values()) else "성공"
    print("\n# Audio Ad Edge Persistence Result", flush=True)
    print(f"- status: {status}", flush=True)
    print(f"- selected_label_file: {LABEL_FILE}", flush=True)
    print(f"- version: {VERSION}", flush=True)
    print(f"- edge_segments: {len(edge_plan)}, edge_feature_success/fail: {edge_success}/{edge_failed}", flush=True)
    print(f"- subwindows: {len(subwindow_plan)}, subwindow_feature_success/fail: {sub_success}/{sub_failed}", flush=True)
    print(f"- persistence_context_scores: {len(context_scores)}, boundary_scores: {len(boundary_df)}, rule_rows: {len(rule_design)}", flush=True)
    print(f"- report: {paths['report']}", flush=True)
    print(f"- summary: {paths['summary']}", flush=True)
    print(f"- latest_bundle: {LATEST_DIR}", flush=True)
    print(f"- errors: {errors}", flush=True)
    return 1 if errors or any(r["status"] == "FAIL" for r in report["sub_agent_results"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
