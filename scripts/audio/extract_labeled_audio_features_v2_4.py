#!/usr/bin/env python3
"""광고/비광고 segment 분석을 위한 label 기반 low-level 오디오 feature를 추출한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import wave
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.fftpack import dct
except Exception:  # pragma: no cover - scipy is expected in the cv env.
    dct = None


PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
TASK_NAME = "extract_labeled_audio_features"
MANIFEST_PATH = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
RAW_VIDEO_DIR = PROJECT_ROOT / "data/raw/videos"
DATA_AUDIO_DIR = PROJECT_ROOT / "data/audio"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_AUDIO_DIR = PROJECT_ROOT / "scripts/audio"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"
BACKUPS_DIR = PROJECT_ROOT / "backups"
SCRIPT_PATH = SCRIPTS_AUDIO_DIR / "extract_labeled_audio_features_v2_4.py"

SAMPLE_RATE = 16000
FRAME_LENGTH = 2048
HOP_LENGTH = 512
RANDOM_SEED = 20260524
RANDOM_ATTEMPTS_PER_BUFFER = 500
SILENCE_RMS_THRESHOLD = 0.01
LOW_ENERGY_RMS_THRESHOLD = 0.02
RANDOM_SEGMENT_DURATION_SEC = 30.0
CONTEXT_SEC = 10.0
BOUNDARY_BUFFERS_SEC = [30.0, 15.0, 5.0]

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

FEATURE_COLUMNS = [
    "audio_rms_mean",
    "audio_rms_std",
    "audio_rms_min",
    "audio_rms_max",
    "audio_log_energy_mean",
    "audio_log_energy_std",
    "audio_peak_amplitude",
    "audio_mean_abs_amplitude",
    "silence_ratio",
    "low_energy_ratio",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "spectral_flux_mean",
    "spectral_flux_std",
    "spectral_flux_max",
]
FEATURE_COLUMNS += [f"mfcc_{i}_mean" for i in range(1, 14)]
FEATURE_COLUMNS += [f"mfcc_{i}_std" for i in range(1, 14)]
FEATURE_COLUMNS += [
    "onset_count",
    "onset_strength_mean",
    "onset_strength_std",
    "onset_strength_max",
    "tempo_estimate",
]

SUMMARY_FEATURES = [
    "audio_rms_mean",
    "audio_log_energy_mean",
    "silence_ratio",
    "zero_crossing_rate_mean",
    "spectral_centroid_mean",
    "spectral_flatness_mean",
    "spectral_flux_mean",
    "spectral_flux_max",
    "onset_count",
    "onset_strength_max",
]


class TaskLogger:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self.log_path: Optional[Path] = None

    def set_path(self, path: Path) -> None:
        self.log_path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        for line in self.lines:
            self._append(line)

    def log(self, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"{timestamp} {message}"
        print(message, flush=True)
        self.lines.append(line)
        if self.log_path is not None:
            self._append(line)

    def _append(self, line: str) -> None:
        assert self.log_path is not None
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


LOGGER = TaskLogger()


def iso_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def readable_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {sec:.1f}s"
    if minutes:
        return f"{minutes}m {sec:.1f}s"
    return f"{sec:.1f}s"


def mmss(seconds: Any) -> str:
    if seconds is None or pd.isna(seconds):
        return ""
    sec = max(0, int(round(float(seconds))))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_time_to_seconds(value: Any) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return float("nan")
    if len(nums) == 2:
        return nums[0] * 60.0 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
    return float("nan")


def version_tuple(version: str) -> Tuple[int, ...]:
    nums = re.findall(r"\d+", version or "")
    return tuple(int(x) for x in nums) if nums else tuple()


def extract_version(path: Path) -> str:
    match = re.search(r"(v\d+(?:_\d+)*)", path.name)
    if match:
        return match.group(1)
    return f"latest_label_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def old_project_snapshot() -> Dict[str, Any]:
    root = OLD_PROJECT_ROOT
    if not root.exists():
        return {
            "path": str(root),
            "exists": False,
            "file_count": 0,
            "metadata_digest": None,
        }
    rel_entries: List[str] = []
    file_count = 0
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            rel = path.relative_to(root).as_posix()
            rel_entries.append(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}")
            file_count += 1
    rel_entries.sort()
    digest = hashlib.sha256("\n".join(rel_entries).encode("utf-8")).hexdigest()
    return {
        "path": str(root),
        "exists": True,
        "file_count": file_count,
        "metadata_digest": digest,
        "snapshot_type": "relative_path_size_mtime_ns",
    }


def excluded_path(path: Path) -> bool:
    excluded_parts = {"backups", "reports", "logs", "cache", "tmp"}
    parts = set(path.parts)
    if excluded_parts & parts:
        return True
    return "outputs" in parts and "latest_for_chatgpt" in parts


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table type: {suffix}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lstrip("\ufeff") for col in df.columns]
    return df


def find_col(columns: Sequence[str], names: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def detect_interval_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    columns = list(df.columns)
    video_col = find_col(columns, ["video_id", "label_mapping_video_id"])
    start_col = find_col(columns, ["ad_start_sec", "start_sec", "ad_start", "ad_start_time"])
    end_col = find_col(columns, ["ad_end_sec", "end_sec", "ad_end", "ad_end_time"])
    mode = "seconds"
    if start_col is None or end_col is None:
        start_col = find_col(columns, ["ad_start_mmss", "start_mmss"])
        end_col = find_col(columns, ["ad_end_mmss", "end_mmss"])
        mode = "mmss"
    return video_col, start_col, end_col, mode


def interval_priority(path: Path) -> int:
    name = path.name.lower()
    if path.parent == PROJECT_ROOT / "data/segments" and re.match(r"ad_interval_segments_.*\.csv$", name):
        return 0
    if "ad_interval" in name:
        return 1
    if "label_interval_context_segments" in name:
        return 2
    if "ad_label" in name:
        return 3
    if "human_review" in name:
        return 4
    return 9


def discover_label_candidates() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    roots = [PROJECT_ROOT / "data/segments", PROJECT_ROOT / "data/review"]
    paths: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"} and not excluded_path(path):
                paths.append(path)
    candidates: List[Dict[str, Any]] = []
    for path in paths:
        info: Dict[str, Any] = {
            "path": str(path),
            "modified_time": iso_from_timestamp(path.stat().st_mtime),
            "modified_time_epoch": path.stat().st_mtime,
            "priority": interval_priority(path),
            "version": extract_version(path),
            "valid": False,
            "reason": "",
        }
        try:
            df = normalize_columns(read_table(path))
            video_col, start_col, end_col, mode = detect_interval_columns(df)
            info.update(
                {
                    "row_count": int(len(df)),
                    "columns": list(df.columns),
                    "video_col": video_col,
                    "start_col": start_col,
                    "end_col": end_col,
                    "time_parse_mode": mode,
                }
            )
            if video_col and start_col and end_col:
                starts = df[start_col].map(parse_time_to_seconds)
                ends = df[end_col].map(parse_time_to_seconds)
                valid_rows = int(((starts < ends) & df[video_col].notna()).sum())
                info["valid_interval_rows"] = valid_rows
                if valid_rows > 0:
                    info["valid"] = True
                    info["reason"] = "video_id and ad start/end columns are recoverable"
                else:
                    info["reason"] = "start/end columns detected but no valid rows"
            else:
                info["reason"] = "missing recoverable video_id/start/end columns"
        except Exception as exc:
            info["reason"] = f"read_failed: {type(exc).__name__}: {exc}"
        candidates.append(info)

    valid = [c for c in candidates if c["valid"]]
    if not valid:
        raise RuntimeError("No valid ad labeling file found")
    valid.sort(
        key=lambda c: (
            c["priority"],
            -float(c["modified_time_epoch"]),
            tuple(-x for x in version_tuple(c["version"])),
            c["path"],
        )
    )
    selected = valid[0]
    return candidates, selected


def canonicalize_intervals(path: Path, version: str) -> pd.DataFrame:
    raw = normalize_columns(read_table(path))
    video_col, start_col, end_col, mode = detect_interval_columns(raw)
    if not (video_col and start_col and end_col):
        raise RuntimeError("Selected label file does not have recoverable interval columns")

    df = raw.copy()
    if "segment_type" in df.columns and path.name.startswith("ad_interval_segments_"):
        df = df[df["segment_type"].astype(str).str.lower().isin(["ad_interval", "ad_full", "ad"])].copy()
    if df.empty:
        df = raw.copy()

    starts = df[start_col].map(parse_time_to_seconds)
    ends = df[end_col].map(parse_time_to_seconds)
    out = pd.DataFrame()
    out["video_id"] = df[video_col].astype(str).str.strip()
    out["ad_start_sec"] = starts.astype(float)
    out["ad_end_sec"] = ends.astype(float)
    out["ad_duration_sec"] = out["ad_end_sec"] - out["ad_start_sec"]
    out["source_file"] = str(path)
    if "source_label_row_index" in df.columns:
        out["source_row_index"] = df["source_label_row_index"]
    else:
        out["source_row_index"] = df.index.astype(int)

    if "ad_interval_id" in df.columns:
        out["ad_interval_id"] = df["ad_interval_id"].astype(str)
    else:
        out["ad_interval_id"] = [
            f"{version}_{vid}_{idx}" for vid, idx in zip(out["video_id"], out["source_row_index"])
        ]
    optional_map = {
        "video_title": ["video_title", "label_mapping_video_title"],
        "channel_name": ["channel_name"],
        "ad_start_mmss": ["ad_start_mmss", "start_mmss"],
        "ad_end_mmss": ["ad_end_mmss", "end_mmss"],
        "ad_type": ["ad_type"],
        "video_path": ["video_path"],
        "video_filename": ["video_filename"],
    }
    for canonical, choices in optional_map.items():
        col = find_col(list(df.columns), choices)
        if col:
            out[canonical] = df[col]
        else:
            out[canonical] = ""
    if "ad_start_mmss" not in out or out["ad_start_mmss"].astype(str).str.len().eq(0).all():
        out["ad_start_mmss"] = out["ad_start_sec"].map(mmss)
    if "ad_end_mmss" not in out or out["ad_end_mmss"].astype(str).str.len().eq(0).all():
        out["ad_end_mmss"] = out["ad_end_sec"].map(mmss)

    out["interval_valid"] = (
        out["video_id"].notna()
        & out["video_id"].astype(str).str.len().gt(0)
        & np.isfinite(out["ad_start_sec"])
        & np.isfinite(out["ad_end_sec"])
        & (out["ad_start_sec"] < out["ad_end_sec"])
        & (out["ad_duration_sec"] > 0)
    )
    out["interval_warning"] = ""
    out.loc[~out["interval_valid"], "interval_warning"] = "invalid_or_non_positive_ad_interval"
    if out.duplicated(["video_id", "ad_start_sec", "ad_end_sec"]).any():
        out.loc[out.duplicated(["video_id", "ad_start_sec", "ad_end_sec"], keep=False), "interval_warning"] = (
            out["interval_warning"].astype(str) + ";duplicate_interval"
        ).str.strip(";")
    out["time_parse_mode"] = mode
    ordered = [
        "ad_interval_id",
        "video_id",
        "ad_start_sec",
        "ad_end_sec",
        "ad_duration_sec",
        "video_title",
        "channel_name",
        "ad_start_mmss",
        "ad_end_mmss",
        "ad_type",
        "source_file",
        "source_row_index",
        "video_path",
        "video_filename",
        "interval_valid",
        "interval_warning",
        "time_parse_mode",
    ]
    return out[ordered]


def resolve_relative_or_absolute(path_value: Any) -> Optional[Path]:
    if path_value is None or pd.isna(path_value):
        return None
    text = str(path_value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        return pd.DataFrame()
    return normalize_columns(pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig"))


def resolve_video_paths(intervals: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    manifest_by_id: Dict[str, pd.Series] = {}
    if not manifest.empty and "video_id" in manifest.columns:
        for _, row in manifest.iterrows():
            manifest_by_id[str(row["video_id"]).strip()] = row

    resolved_rows: List[Dict[str, Any]] = []
    for _, row in intervals.iterrows():
        video_id = str(row["video_id"]).strip()
        manifest_row = manifest_by_id.get(video_id)
        video_path: Optional[Path] = None
        resolution_method = ""
        candidates: List[Path] = []

        label_path = resolve_relative_or_absolute(row.get("video_path"))
        if label_path is not None:
            candidates.append(label_path)
        if manifest_row is not None and "video_path" in manifest_row.index:
            manifest_path = resolve_relative_or_absolute(manifest_row.get("video_path"))
            if manifest_path is not None:
                candidates.append(manifest_path)
        for candidate in candidates:
            if candidate.exists():
                video_path = candidate
                resolution_method = "existing_video_path"
                break

        filename_values = [row.get("video_filename")]
        if manifest_row is not None and "video_filename" in manifest_row.index:
            filename_values.append(manifest_row.get("video_filename"))
        if video_path is None:
            for value in filename_values:
                if value is None or pd.isna(value) or not str(value).strip():
                    continue
                candidate = RAW_VIDEO_DIR / str(value).strip()
                if candidate.exists():
                    video_path = candidate
                    resolution_method = "raw_video_dir_filename"
                    break
        if video_path is None:
            for value in filename_values:
                if value is None or pd.isna(value) or not str(value).strip():
                    continue
                matches = list(RAW_VIDEO_DIR.glob(str(value).strip()))
                if matches:
                    video_path = matches[0]
                    resolution_method = "raw_video_dir_glob"
                    break

        manifest_duration = np.nan
        manifest_title = ""
        if manifest_row is not None:
            if "duration_sec" in manifest_row.index:
                manifest_duration = parse_time_to_seconds(manifest_row.get("duration_sec"))
            if "video_title" in manifest_row.index:
                manifest_title = manifest_row.get("video_title")

        resolved_rows.append(
            {
                "video_id": video_id,
                "resolved_video_path": str(video_path) if video_path else "",
                "video_path_exists": bool(video_path and video_path.exists()),
                "video_path_resolution_method": resolution_method or "unresolved",
                "manifest_duration_sec": manifest_duration,
                "manifest_video_title": manifest_title,
            }
        )
    return pd.DataFrame(resolved_rows)


def executable_path(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    fallback = Path(".venv/bin") / name
    if fallback.exists():
        return str(fallback)
    return None


def run_command_json(cmd: Sequence[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout or f"exit_code={proc.returncode}").strip()
    try:
        return json.loads(proc.stdout), ""
    except Exception as exc:
        return None, f"json_parse_failed: {type(exc).__name__}: {exc}"


def probe_videos(video_rows: pd.DataFrame, ffprobe_path: Optional[str]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    unique = video_rows.drop_duplicates("video_id")
    for _, row in unique.iterrows():
        path_text = row.get("resolved_video_path", "")
        path = Path(path_text) if path_text else None
        record: Dict[str, Any] = {
            "video_id": row["video_id"],
            "video_path": path_text,
            "path_exists": bool(path and path.exists()),
            "manifest_duration_sec": row.get("manifest_duration_sec", np.nan),
            "probe_status": "not_run",
            "probe_error": "",
            "ffprobe_duration_sec": np.nan,
            "video_duration_sec": row.get("manifest_duration_sec", np.nan),
            "audio_available": False,
            "audio_stream_index": np.nan,
            "audio_codec_name": "",
            "audio_sample_rate": np.nan,
        }
        if not record["path_exists"]:
            record["probe_status"] = "path_missing"
            record["probe_error"] = "resolved video path missing"
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
            data, err = run_command_json(cmd)
            if data is None:
                record["probe_status"] = "probe_failed"
                record["probe_error"] = err
            else:
                record["probe_status"] = "success"
                try:
                    record["ffprobe_duration_sec"] = float(data.get("format", {}).get("duration", np.nan))
                    record["video_duration_sec"] = record["ffprobe_duration_sec"]
                except Exception:
                    pass
                audio_streams = [
                    stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"
                ]
                if audio_streams:
                    stream = audio_streams[0]
                    record["audio_available"] = True
                    record["audio_stream_index"] = stream.get("index", np.nan)
                    record["audio_codec_name"] = stream.get("codec_name", "")
                    try:
                        record["audio_sample_rate"] = float(stream.get("sample_rate", np.nan))
                    except Exception:
                        record["audio_sample_rate"] = np.nan
                else:
                    record["probe_error"] = "no_audio_stream"
        records.append(record)
    return pd.DataFrame(records)


def overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return float(a_start) < float(b_end) and float(a_end) > float(b_start)


def clip_interval(start: float, end: float, duration: float) -> Tuple[float, float]:
    return max(0.0, start), min(float(duration), end)


def valid_interval_pair(start: Any, end: Any) -> bool:
    return (
        start is not None
        and end is not None
        and np.isfinite(start)
        and np.isfinite(end)
        and float(start) < float(end)
    )


def sample_random_segment(
    video_id: str,
    video_duration: float,
    ad_intervals: List[Tuple[float, float]],
    context_intervals: List[Tuple[float, float]],
    used_random: List[Tuple[float, float]],
    rng: random.Random,
) -> Tuple[Optional[float], Optional[float], str, str, int, Optional[float]]:
    if not np.isfinite(video_duration) or video_duration < RANDOM_SEGMENT_DURATION_SEC:
        return None, None, "random_sampling_failed", "video_duration_short_or_unknown", 0, None
    max_start = video_duration - RANDOM_SEGMENT_DURATION_SEC
    total_attempts = 0
    for buffer_sec in BOUNDARY_BUFFERS_SEC:
        boundary_intervals: List[Tuple[float, float]] = []
        for ad_start, ad_end in ad_intervals:
            boundary_intervals.append(clip_interval(ad_start - buffer_sec, ad_start + buffer_sec, video_duration))
            boundary_intervals.append(clip_interval(ad_end - buffer_sec, ad_end + buffer_sec, video_duration))
        avoid = ad_intervals + context_intervals + boundary_intervals + used_random
        for _ in range(RANDOM_ATTEMPTS_PER_BUFFER):
            total_attempts += 1
            start = rng.uniform(0.0, max_start)
            end = start + RANDOM_SEGMENT_DURATION_SEC
            if any(overlaps(start, end, avoid_start, avoid_end) for avoid_start, avoid_end in avoid):
                continue
            warning = "" if buffer_sec == BOUNDARY_BUFFERS_SEC[0] else f"boundary_buffer_relaxed_to_{int(buffer_sec)}s"
            return start, end, "sampled", warning, total_attempts, buffer_sec
    return None, None, "random_sampling_failed", "no_valid_same_video_non_ad_window", total_attempts, None


def build_sampling_plan(
    intervals: pd.DataFrame,
    resolved_paths: pd.DataFrame,
    probe_df: pd.DataFrame,
    version: str,
    label_file: Path,
    label_mtime: str,
) -> pd.DataFrame:
    path_by_video = {
        str(row["video_id"]): row
        for _, row in resolved_paths.drop_duplicates("video_id").iterrows()
    }
    probe_by_video = {str(row["video_id"]): row for _, row in probe_df.iterrows()}

    valid_intervals_by_video: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    context_by_video: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for _, row in intervals.iterrows():
        video_id = str(row["video_id"])
        probe = probe_by_video.get(video_id)
        video_duration = float(probe["video_duration_sec"]) if probe is not None else np.nan
        if not row["interval_valid"]:
            continue
        ad_start = float(row["ad_start_sec"])
        ad_end = float(row["ad_end_sec"])
        valid_intervals_by_video[video_id].append((ad_start, ad_end))
        if np.isfinite(video_duration):
            context_by_video[video_id].append(clip_interval(ad_start - CONTEXT_SEC, ad_start, video_duration))
            context_by_video[video_id].append(clip_interval(ad_end, ad_end + CONTEXT_SEC, video_duration))

    rng = random.Random(RANDOM_SEED)
    used_random_by_video: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    rows: List[Dict[str, Any]] = []

    for interval_idx, row in intervals.iterrows():
        video_id = str(row["video_id"])
        probe = probe_by_video.get(video_id)
        resolved = path_by_video.get(video_id)
        video_path = ""
        video_duration = np.nan
        video_title = row.get("video_title", "")
        if resolved is not None:
            video_path = resolved.get("resolved_video_path", "")
            if not video_title:
                video_title = resolved.get("manifest_video_title", "")
        if probe is not None:
            video_duration = probe.get("video_duration_sec", np.nan)
        if not np.isfinite(video_duration):
            video_duration = resolved.get("manifest_duration_sec", np.nan) if resolved is not None else np.nan

        ad_start = float(row["ad_start_sec"]) if np.isfinite(row["ad_start_sec"]) else np.nan
        ad_end = float(row["ad_end_sec"]) if np.isfinite(row["ad_end_sec"]) else np.nan
        ad_duration = float(row["ad_duration_sec"]) if np.isfinite(row["ad_duration_sec"]) else np.nan
        ad_interval_id = str(row["ad_interval_id"])

        def base(segment_type: str) -> Dict[str, Any]:
            return {
                "segment_id": f"{version}_{ad_interval_id}_{segment_type}",
                "version": version,
                "source_label_file": str(label_file),
                "source_label_modified_time": label_mtime,
                "video_id": video_id,
                "video_title": video_title,
                "video_path": video_path,
                "video_duration_sec": video_duration,
                "ad_interval_id": ad_interval_id,
                "source_ad_start_sec": ad_start,
                "source_ad_end_sec": ad_end,
                "source_ad_duration_sec": ad_duration,
                "segment_type": segment_type,
                "segment_start_sec": np.nan,
                "segment_end_sec": np.nan,
                "segment_duration_sec": np.nan,
                "segment_start_mmss": "",
                "segment_end_mmss": "",
                "sampling_status": "not_planned",
                "sampling_warning": "",
                "random_seed": RANDOM_SEED,
                "random_attempt_count": 0,
                "context_truncated": False,
                "context_overlaps_other_ad": False,
                "random_boundary_buffer_sec": np.nan,
            }

        def finalize_segment(seg: Dict[str, Any], start: float, end: float, warning: str = "") -> Dict[str, Any]:
            seg["segment_start_sec"] = start
            seg["segment_end_sec"] = end
            seg["segment_duration_sec"] = end - start if valid_interval_pair(start, end) else np.nan
            seg["segment_start_mmss"] = mmss(start)
            seg["segment_end_mmss"] = mmss(end)
            seg["sampling_status"] = "planned" if valid_interval_pair(start, end) else "invalid_segment_bounds"
            seg["sampling_warning"] = warning
            if np.isfinite(video_duration):
                if start < -1e-6 or end > video_duration + 1e-6:
                    seg["sampling_status"] = "invalid_segment_out_of_video_bounds"
                    seg["sampling_warning"] = (warning + ";segment_out_of_video_bounds").strip(";")
            return seg

        if not row["interval_valid"]:
            for segment_type in ["ad_full", "pre_ad_10s", "post_ad_10s", "random_non_ad_30s"]:
                seg = base(segment_type)
                seg["sampling_status"] = "invalid_ad_interval"
                seg["sampling_warning"] = row.get("interval_warning", "invalid_ad_interval")
                rows.append(seg)
            continue

        rows.append(finalize_segment(base("ad_full"), ad_start, ad_end))

        pre_start = max(0.0, ad_start - CONTEXT_SEC)
        pre_end = ad_start
        pre_warning_parts: List[str] = []
        if pre_end - pre_start < CONTEXT_SEC - 1e-6:
            pre_warning_parts.append("context_truncated")
        other_ads = [
            iv for iv in valid_intervals_by_video[video_id] if not (abs(iv[0] - ad_start) < 1e-6 and abs(iv[1] - ad_end) < 1e-6)
        ]
        pre_overlap = any(overlaps(pre_start, pre_end, s, e) for s, e in other_ads)
        if pre_overlap:
            pre_warning_parts.append("context_overlaps_other_ad")
        pre = finalize_segment(base("pre_ad_10s"), pre_start, pre_end, ";".join(pre_warning_parts))
        pre["context_truncated"] = pre_end - pre_start < CONTEXT_SEC - 1e-6
        pre["context_overlaps_other_ad"] = pre_overlap
        rows.append(pre)

        post_start = ad_end
        post_end = min(video_duration, ad_end + CONTEXT_SEC) if np.isfinite(video_duration) else ad_end + CONTEXT_SEC
        post_warning_parts: List[str] = []
        if post_end - post_start < CONTEXT_SEC - 1e-6:
            post_warning_parts.append("context_truncated")
        post_overlap = any(overlaps(post_start, post_end, s, e) for s, e in other_ads)
        if post_overlap:
            post_warning_parts.append("context_overlaps_other_ad")
        post = finalize_segment(base("post_ad_10s"), post_start, post_end, ";".join(post_warning_parts))
        post["context_truncated"] = post_end - post_start < CONTEXT_SEC - 1e-6
        post["context_overlaps_other_ad"] = post_overlap
        rows.append(post)

        random_seg = base("random_non_ad_30s")
        r_start, r_end, status, warning, attempts, used_buffer = sample_random_segment(
            video_id,
            float(video_duration) if np.isfinite(video_duration) else np.nan,
            valid_intervals_by_video[video_id],
            context_by_video[video_id],
            used_random_by_video[video_id],
            rng,
        )
        random_seg["random_attempt_count"] = attempts
        random_seg["random_boundary_buffer_sec"] = used_buffer if used_buffer is not None else np.nan
        if r_start is not None and r_end is not None:
            random_seg = finalize_segment(random_seg, r_start, r_end, warning)
            random_seg["sampling_status"] = status
            used_random_by_video[video_id].append((r_start, r_end))
        else:
            random_seg["sampling_status"] = status
            random_seg["sampling_warning"] = warning
        rows.append(random_seg)

    return pd.DataFrame(rows)


def read_wav_from_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample_width={sample_width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False), sample_rate


def decode_audio_segment(
    ffmpeg_path: str,
    video_path: str,
    start_sec: float,
    duration_sec: float,
) -> Tuple[np.ndarray, int, str]:
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-t",
        f"{duration_sec:.6f}",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "wav",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True)
    except Exception as exc:
        return np.array([], dtype=np.float32), SAMPLE_RATE, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        return np.array([], dtype=np.float32), SAMPLE_RATE, err or f"ffmpeg_exit_{proc.returncode}"
    try:
        audio, sr = read_wav_from_bytes(proc.stdout)
    except Exception as exc:
        return np.array([], dtype=np.float32), SAMPLE_RATE, f"wav_decode_failed: {type(exc).__name__}: {exc}"
    return audio, sr, ""


def frame_audio(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.size == 0:
        return np.empty((0, frame_length), dtype=np.float32)
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))
    remainder = (audio.size - frame_length) % hop_length
    if remainder:
        audio = np.pad(audio, (0, hop_length - remainder))
    n_frames = 1 + (audio.size - frame_length) // hop_length
    shape = (n_frames, frame_length)
    strides = (audio.strides[0] * hop_length, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    min_mel = hz_to_mel(np.array([0.0]))[0]
    max_mel = hz_to_mel(np.array([sr / 2.0]))[0]
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, min(center, filters.shape[1])):
            filters[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, min(right, filters.shape[1])):
            filters[i - 1, j] = (right - j) / max(right - center, 1)
    return filters


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def compute_audio_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    features = {col: float("nan") for col in FEATURE_COLUMNS}
    if audio.size == 0:
        return features
    eps = 1e-10
    frames = frame_audio(audio.astype(np.float32, copy=False), FRAME_LENGTH, HOP_LENGTH)
    if frames.size == 0:
        return features
    frame_energy = np.mean(frames * frames, axis=1)
    rms = np.sqrt(frame_energy + eps)
    log_energy = np.log10(frame_energy + eps)
    features["audio_rms_mean"] = safe_float(np.mean(rms))
    features["audio_rms_std"] = safe_float(np.std(rms))
    features["audio_rms_min"] = safe_float(np.min(rms))
    features["audio_rms_max"] = safe_float(np.max(rms))
    features["audio_log_energy_mean"] = safe_float(np.mean(log_energy))
    features["audio_log_energy_std"] = safe_float(np.std(log_energy))
    features["audio_peak_amplitude"] = safe_float(np.max(np.abs(audio)))
    features["audio_mean_abs_amplitude"] = safe_float(np.mean(np.abs(audio)))
    features["silence_ratio"] = safe_float(np.mean(rms <= SILENCE_RMS_THRESHOLD))
    features["low_energy_ratio"] = safe_float(np.mean(rms <= LOW_ENERGY_RMS_THRESHOLD))

    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    features["zero_crossing_rate_mean"] = safe_float(np.mean(zcr))
    features["zero_crossing_rate_std"] = safe_float(np.std(zcr))

    window = np.hanning(FRAME_LENGTH).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1)
    magnitude = np.abs(spectrum).astype(np.float32) + eps
    power = magnitude * magnitude
    freqs = np.fft.rfftfreq(FRAME_LENGTH, 1.0 / sr).astype(np.float32)
    mag_sum = np.sum(magnitude, axis=1) + eps
    centroid = np.sum(magnitude * freqs[None, :], axis=1) / mag_sum
    bandwidth = np.sqrt(np.sum(magnitude * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / mag_sum)
    cumulative = np.cumsum(magnitude, axis=1)
    roll_threshold = 0.85 * cumulative[:, -1]
    roll_idx = np.argmax(cumulative >= roll_threshold[:, None], axis=1)
    rolloff = freqs[roll_idx]
    flatness = np.exp(np.mean(np.log(power + eps), axis=1)) / (np.mean(power, axis=1) + eps)
    features["spectral_centroid_mean"] = safe_float(np.mean(centroid))
    features["spectral_centroid_std"] = safe_float(np.std(centroid))
    features["spectral_bandwidth_mean"] = safe_float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = safe_float(np.std(bandwidth))
    features["spectral_rolloff_mean"] = safe_float(np.mean(rolloff))
    features["spectral_rolloff_std"] = safe_float(np.std(rolloff))
    features["spectral_flatness_mean"] = safe_float(np.mean(flatness))
    features["spectral_flatness_std"] = safe_float(np.std(flatness))

    norm_mag = magnitude / mag_sum[:, None]
    if norm_mag.shape[0] > 1:
        flux = np.sqrt(np.sum(np.diff(norm_mag, axis=0) ** 2, axis=1))
    else:
        flux = np.array([0.0], dtype=np.float32)
    features["spectral_flux_mean"] = safe_float(np.mean(flux))
    features["spectral_flux_std"] = safe_float(np.std(flux))
    features["spectral_flux_max"] = safe_float(np.max(flux))

    if dct is not None:
        filters = mel_filterbank(sr, FRAME_LENGTH, 40)
        mel_energy = np.maximum(np.dot(power, filters.T), eps)
        log_mel = np.log(mel_energy)
        mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :13]
        for idx in range(13):
            features[f"mfcc_{idx + 1}_mean"] = safe_float(np.mean(mfcc[:, idx]))
            features[f"mfcc_{idx + 1}_std"] = safe_float(np.std(mfcc[:, idx]))

    onset_strength = flux
    if onset_strength.size:
        threshold = np.median(onset_strength) + np.std(onset_strength)
        if onset_strength.size >= 3:
            peaks = (
                (onset_strength[1:-1] > onset_strength[:-2])
                & (onset_strength[1:-1] >= onset_strength[2:])
                & (onset_strength[1:-1] > threshold)
            )
            onset_count = int(np.sum(peaks))
        else:
            onset_count = int(np.sum(onset_strength > threshold))
        features["onset_count"] = float(onset_count)
        features["onset_strength_mean"] = safe_float(np.mean(onset_strength))
        features["onset_strength_std"] = safe_float(np.std(onset_strength))
        features["onset_strength_max"] = safe_float(np.max(onset_strength))
    features["tempo_estimate"] = float("nan")
    return features


def extract_features(plan_df: pd.DataFrame, probe_df: pd.DataFrame, ffmpeg_path: Optional[str]) -> pd.DataFrame:
    probe_by_video = {str(row["video_id"]): row for _, row in probe_df.iterrows()}
    records: List[Dict[str, Any]] = []
    for idx, row in plan_df.iterrows():
        record = row.to_dict()
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
        record.update({col: np.nan for col in FEATURE_COLUMNS})
        video_id = str(row["video_id"])
        probe = probe_by_video.get(video_id)
        if probe is not None:
            record["audio_available"] = bool(probe.get("audio_available", False))
            record["audio_stream_index"] = probe.get("audio_stream_index", np.nan)
        if row.get("sampling_status") not in {"planned", "sampled"}:
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
            record["feature_error"] = "no audio stream from probe"
            records.append(record)
            continue
        video_path = str(row.get("video_path", ""))
        if not video_path or not Path(video_path).exists():
            record["feature_status"] = "video_path_missing"
            record["feature_error"] = "video path missing"
            records.append(record)
            continue
        start = safe_float(row.get("segment_start_sec"))
        duration = safe_float(row.get("segment_duration_sec"))
        if not np.isfinite(start) or not np.isfinite(duration) or duration <= 0:
            record["feature_status"] = "invalid_segment_duration"
            record["feature_error"] = "segment start/duration invalid"
            records.append(record)
            continue
        audio, sr, err = decode_audio_segment(ffmpeg_path, video_path, start, duration)
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
            features = compute_audio_features(audio, sr)
            record.update(features)
            record["feature_status"] = "success"
        except Exception as exc:
            record["feature_status"] = "feature_compute_failed"
            record["feature_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        if (idx + 1) % 10 == 0:
            LOGGER.log(f"[STEP 06] Extracted audio features for {idx + 1}/{len(plan_df)} segments")
    return pd.DataFrame(records)


def write_group_summary(features_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for segment_type, group in features_df.groupby("segment_type", dropna=False):
        for feature in SUMMARY_FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce") if feature in group.columns else pd.Series(dtype=float)
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "segment_type": segment_type,
                    "feature": feature,
                    "count": int(len(group)),
                    "valid_count": int(len(finite)),
                    "mean": safe_float(finite.mean()) if len(finite) else np.nan,
                    "std": safe_float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                    "median": safe_float(finite.median()) if len(finite) else np.nan,
                    "min": safe_float(finite.min()) if len(finite) else np.nan,
                    "max": safe_float(finite.max()) if len(finite) else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    return summary


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return float("nan")
    return safe_float((a.mean() - b.mean()) / math.sqrt(pooled))


def key_comparisons(features_df: pd.DataFrame) -> List[Dict[str, Any]]:
    pairs = [
        ("ad_full", "random_non_ad_30s"),
        ("pre_ad_10s", "ad_full"),
        ("post_ad_10s", "ad_full"),
        ("pre_ad_10s", "post_ad_10s"),
    ]
    rows: List[Dict[str, Any]] = []
    for left, right in pairs:
        left_df = features_df[features_df["segment_type"] == left]
        right_df = features_df[features_df["segment_type"] == right]
        for feature in SUMMARY_FEATURES:
            left_vals = pd.to_numeric(left_df[feature], errors="coerce")
            right_vals = pd.to_numeric(right_df[feature], errors="coerce")
            left_finite = left_vals[np.isfinite(left_vals)]
            right_finite = right_vals[np.isfinite(right_vals)]
            rows.append(
                {
                    "comparison": f"{left} vs {right}",
                    "feature": feature,
                    "left_valid_count": int(len(left_finite)),
                    "right_valid_count": int(len(right_finite)),
                    "mean_difference": safe_float(left_finite.mean() - right_finite.mean())
                    if len(left_finite) and len(right_finite)
                    else np.nan,
                    "median_difference": safe_float(left_finite.median() - right_finite.median())
                    if len(left_finite) and len(right_finite)
                    else np.nan,
                    "cohens_d": cohen_d(left_vals, right_vals),
                }
            )
    return rows


def nan_inf_sanity(features_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
    nan_ratio: Dict[str, float] = {}
    inf_count: Dict[str, int] = {}
    out_of_range: Dict[str, int] = {}
    for col in FEATURE_COLUMNS:
        if col not in features_df.columns:
            continue
        values = pd.to_numeric(features_df[col], errors="coerce")
        nan_ratio[col] = safe_float(values.isna().mean())
        inf_count[col] = int(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).sum())
        if col in {"silence_ratio", "low_energy_ratio"}:
            out_of_range[col] = int(((values < 0) | (values > 1)).sum())
    return nan_ratio, inf_count, out_of_range


def validate_latest_label(intervals: pd.DataFrame, report: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    status = "PASS"
    if intervals.empty:
        errors.append("canonical interval table is empty")
    if not {"video_id", "ad_start_sec", "ad_end_sec"}.issubset(intervals.columns):
        errors.append("canonical required columns missing")
    invalid_count = int((~intervals["interval_valid"]).sum()) if "interval_valid" in intervals else len(intervals)
    if invalid_count:
        warnings.append(f"{invalid_count} interval rows are invalid or non-positive")
    if report.get("version") == "v2_4" and report.get("ad_interval_count") != 22:
        warnings.append("v2_4 selected but interval row count differs from expected ~22")
    selected_reason = report.get("selected_label_reason", "")
    if not selected_reason:
        warnings.append("selected label reason is empty")
    if errors:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    return {"status": status, "warnings": warnings, "errors": errors}


def validate_sampling_plan(plan: pd.DataFrame, intervals: pd.DataFrame) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    expected = len(intervals) * 4
    if len(plan) != expected:
        errors.append(f"planned row count {len(plan)} != expected {expected}")
    expected_types = {"ad_full", "pre_ad_10s", "post_ad_10s", "random_non_ad_30s"}
    found_types = set(plan["segment_type"].dropna().astype(str))
    missing = expected_types - found_types
    if missing:
        errors.append(f"missing segment types: {sorted(missing)}")
    valid_rows = plan[plan["sampling_status"].isin(["planned", "sampled"])]
    invalid_bounds = valid_rows[
        ~(pd.to_numeric(valid_rows["segment_start_sec"], errors="coerce") < pd.to_numeric(valid_rows["segment_end_sec"], errors="coerce"))
    ]
    if not invalid_bounds.empty:
        errors.append(f"{len(invalid_bounds)} valid sampling rows have start >= end")
    out_of_bounds = valid_rows[
        (pd.to_numeric(valid_rows["segment_start_sec"], errors="coerce") < -1e-6)
        | (
            pd.to_numeric(valid_rows["segment_end_sec"], errors="coerce")
            > pd.to_numeric(valid_rows["video_duration_sec"], errors="coerce") + 1e-6
        )
    ]
    if not out_of_bounds.empty:
        errors.append(f"{len(out_of_bounds)} valid sampling rows exceed video duration")
    random_rows = plan[plan["segment_type"] == "random_non_ad_30s"]
    ad_by_video: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for _, row in intervals[intervals["interval_valid"]].iterrows():
        ad_by_video[str(row["video_id"])].append((float(row["ad_start_sec"]), float(row["ad_end_sec"])))
    random_overlap_count = 0
    random_non30_count = 0
    for _, row in random_rows[random_rows["sampling_status"] == "sampled"].iterrows():
        start = float(row["segment_start_sec"])
        end = float(row["segment_end_sec"])
        if abs((end - start) - RANDOM_SEGMENT_DURATION_SEC) > 1e-3:
            random_non30_count += 1
        if any(overlaps(start, end, a, b) for a, b in ad_by_video[str(row["video_id"])]):
            random_overlap_count += 1
    if random_overlap_count:
        errors.append(f"{random_overlap_count} random_non_ad_30s rows overlap ad intervals")
    if random_non30_count:
        warnings.append(f"{random_non30_count} sampled random rows are not exactly 30 seconds")
    if plan["context_truncated"].astype(bool).any():
        warnings.append("some pre/post context windows are truncated by video boundaries")
    if plan["context_overlaps_other_ad"].astype(bool).any():
        warnings.append("some pre/post context windows overlap another ad interval")
    relaxed = random_rows["sampling_warning"].astype(str).str.contains("boundary_buffer_relaxed", na=False).sum()
    if relaxed:
        warnings.append(f"{int(relaxed)} random rows used relaxed boundary buffer")
    failed = (random_rows["sampling_status"] == "random_sampling_failed").sum()
    if failed:
        warnings.append(f"{int(failed)} random rows failed same-video sampling")
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {"status": status, "warnings": warnings, "errors": errors}


def validate_audio_features(features_df: pd.DataFrame, report: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    if not report.get("ffmpeg_available"):
        errors.append("ffmpeg unavailable")
    if not report.get("ffprobe_available"):
        errors.append("ffprobe unavailable")
    success_count = int((features_df["feature_status"] == "success").sum())
    if success_count == 0:
        errors.append("no successful audio feature rows")
    elif success_count < max(1, int(len(features_df) * 0.5)):
        warnings.append("less than half of planned segments extracted successfully")
    missing_core = [col for col in SUMMARY_FEATURES if col not in features_df.columns]
    if missing_core:
        errors.append(f"missing core feature columns: {missing_core}")
    for col in ["silence_ratio", "low_energy_ratio"]:
        vals = pd.to_numeric(features_df[col], errors="coerce") if col in features_df else pd.Series(dtype=float)
        bad = ((vals < 0) | (vals > 1)).sum()
        if bad:
            errors.append(f"{col} has {int(bad)} out-of-range rows")
    inf_counts = report.get("inf_count_by_feature", {})
    high_inf = {k: v for k, v in inf_counts.items() if v}
    if high_inf:
        errors.append(f"inf values present in feature columns: {high_inf}")
    if report.get("audio_stream_failed_count", 0):
        warnings.append("some videos have no usable audio stream or failed probe")
    if report.get("decoded_segment_failed_count", 0):
        warnings.append("some segment audio decode/feature extraction rows failed")
    warnings.append("tempo_estimate left as NaN in scipy/numpy fallback")
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {"status": status, "warnings": warnings, "errors": errors}


def validate_analysis(features_df: pd.DataFrame, summary_df: pd.DataFrame, summary_text: str) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    expected_types = {"ad_full", "pre_ad_10s", "post_ad_10s", "random_non_ad_30s"}
    summary_types = set(summary_df["segment_type"].dropna().astype(str)) if not summary_df.empty else set()
    if not expected_types.issubset(summary_types):
        errors.append("group summary missing one or more segment types")
    banned = [
        "final ad detection performance",
        "final model performance",
        "audio-only detected ad boundary",
        "confirmed ad detection accuracy",
    ]
    text_lower = summary_text.lower()
    banned_found = [phrase for phrase in banned if phrase in text_lower]
    if banned_found:
        errors.append(f"final-performance-like claim found: {banned_found}")
    if "audio cue analysis" not in text_lower:
        warnings.append("summary does not explicitly include audio cue analysis phrase")
    valid_counts = features_df.groupby("segment_type")["feature_status"].apply(lambda x: int((x == "success").sum())).to_dict()
    low_valid = {k: v for k, v in valid_counts.items() if v < 2}
    if low_valid:
        warnings.append(f"some segment types have low valid counts: {low_valid}")
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {"status": status, "warnings": warnings, "errors": errors}


def forbidden_latest_files() -> List[str]:
    if not LATEST_DIR.exists():
        return []
    forbidden: List[str] = []
    for path in LATEST_DIR.rglob("*"):
        if path.is_dir():
            if path.name in {"cache", "tmp"}:
                forbidden.append(str(path))
            continue
        if path.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES:
            forbidden.append(str(path))
    return forbidden


def validate_output_safety(report: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    for path in report.get("output_files", []):
        if not Path(path).exists():
            errors.append(f"required output missing: {path}")
    forbidden = forbidden_latest_files()
    if forbidden:
        errors.append(f"forbidden files found in latest_for_chatgpt: {forbidden}")
    if report.get("input_label_file_modified"):
        errors.append("input label file modified")
    if report.get("old_project_modified"):
        errors.append("old project modified")
    backup_dir = report.get("backup_dir")
    if backup_dir:
        warnings.append(f"existing files were backed up to {backup_dir}")
    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    return {"status": status, "warnings": warnings, "errors": errors}


def backup_existing_targets(version: str, timestamp: str) -> Optional[Path]:
    targets = [
        DATA_AUDIO_DIR / f"audio_labeled_segment_sampling_plan_{version}.csv",
        DATA_AUDIO_DIR / f"audio_labeled_segment_features_{version}.csv",
        DATA_AUDIO_DIR / f"audio_probe_metadata_{version}.csv",
        DATA_AUDIO_DIR / f"audio_labeled_segment_group_summary_{version}.csv",
        REPORTS_DIR / f"extract_labeled_audio_features_{version}_report.json",
        REPORTS_DIR / f"extract_labeled_audio_features_{version}_summary.md",
        LOGS_DIR / f"extract_labeled_audio_features_{version}_run_log.txt",
        SCRIPT_PATH,
        LATEST_DIR / "README_latest_files.md",
    ]
    existing = [path for path in targets if path.exists()]
    if not existing:
        return None
    backup_dir = BACKUPS_DIR / f"extract_labeled_audio_features_{version}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        if path == SCRIPT_PATH:
            # 현재 script가 방금 실행되었으므로 이번 run 이전 파일일 때만 보존한다.
            continue
        shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def copy_latest(files: Dict[str, Path], version: str, report: Dict[str, Any]) -> List[str]:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    allowed_names = {path.name for path in files.values()} | {"README_latest_files.md"}
    archived_previous: List[str] = []
    cleanup_root_text = report.get("backup_dir") or str(
        BACKUPS_DIR / f"extract_labeled_audio_features_{version}_latest_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    cleanup_dir = Path(cleanup_root_text) / "latest_for_chatgpt_previous"
    for item in list(LATEST_DIR.iterdir()):
        if item.name in allowed_names:
            continue
        cleanup_dir.mkdir(parents=True, exist_ok=True)
        destination = cleanup_dir / item.name
        if destination.exists():
            stem = destination.stem
            suffix = destination.suffix
            counter = 1
            while destination.exists():
                destination = cleanup_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.move(str(item), str(destination))
        archived_previous.append(str(destination))
    if archived_previous:
        report.setdefault("latest_for_chatgpt_archived_previous_files", []).extend(archived_previous)

    copied: List[str] = []
    for key, path in files.items():
        if not path.exists():
            continue
        if path.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES:
            continue
        destination = LATEST_DIR / path.name
        shutil.copy2(path, destination)
        copied.append(str(destination))
    readme_path = LATEST_DIR / "README_latest_files.md"
    lines = [
        "# Latest Files",
        "",
        f"- task_name: {TASK_NAME}",
        f"- version: {version}",
        f"- selected_label_file: {report.get('selected_label_file')}",
        f"- selected_label_file_modified_time: {report.get('selected_label_file_modified_time')}",
        f"- old_project_modified: {str(report.get('old_project_modified')).lower()}",
        "- forbidden_media_or_model_files_included: false",
        "",
        "This bundle is for audio cue analysis and labeled audio feature extraction.",
        "It is not a final ad detection performance or final model performance claim.",
        "",
        "## Files",
    ]
    descriptions = {
        "sampling_plan": "segment sampling plan for ad_full, pre_ad_10s, post_ad_10s, and random_non_ad_30s",
        "features": "low-level audio feature table by labeled segment",
        "probe": "video/audio stream probe metadata",
        "group_summary": "segment_type group summary statistics",
        "report": "machine-readable task report JSON",
        "summary": "human-readable task summary",
        "run_log": "step-by-step run log",
        "script": "reproducible extraction script",
    }
    for key, path in files.items():
        if path.exists():
            lines.append(f"- {path.name}: {descriptions.get(key, key)}")
    lines.extend(
        [
            "- README_latest_files.md: this manifest",
            "",
            "No mp4, wav, mp3, m4a, image, model, cache, tmp, or raw video files are included.",
        ]
    )
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copied.append(str(readme_path))
    return copied


def write_summary_md(
    path: Path,
    report: Dict[str, Any],
    group_summary: pd.DataFrame,
    sub_agent_results: Dict[str, Any],
) -> str:
    def simple_markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        columns = list(df.columns)

        def fmt(value: Any) -> str:
            if value is None or pd.isna(value):
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.6g}"
            return str(value).replace("|", "\\|")

        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
        return "\n".join(lines)

    def table_for_feature(feature: str) -> str:
        rows = group_summary[group_summary["feature"] == feature].copy()
        if rows.empty:
            return ""
        rows = rows[["segment_type", "valid_count", "mean", "median", "std", "min", "max"]]
        return simple_markdown_table(rows)

    comparison_lines = []
    for item in report.get("key_comparisons", [])[:12]:
        comparison_lines.append(
            f"- {item['comparison']} / {item['feature']}: "
            f"mean_diff={item['mean_difference']}, median_diff={item['median_difference']}, cohens_d={item['cohens_d']}"
        )
    if not comparison_lines:
        comparison_lines.append("- No stable comparison rows were available.")

    output_lines = [f"- {p}" for p in report.get("output_files", [])]
    latest_lines = [f"- {p}" for p in report.get("latest_for_chatgpt_files", [])]
    subagent_lines = [
        f"- {name}: {result.get('status')} (warnings={len(result.get('warnings', []))}, errors={len(result.get('errors', []))})"
        for name, result in sub_agent_results.items()
    ]
    warning_lines = [f"- {w}" for w in report.get("warnings", [])] or ["- None"]
    error_lines = [f"- {e}" for e in report.get("errors", [])] or ["- None"]
    segment_counts = report.get("segment_count_by_type", {})
    segment_count_lines = [f"- {k}: {v}" for k, v in sorted(segment_counts.items())]

    text = "\n".join(
        [
            "# Extract Labeled Audio Features Summary",
            "",
            "## 1. Task Overview",
            "This run uses the most recently selected valid ad interval labeling file as the source of truth for labeled audio feature extraction.",
            "It is label-based audio cue analysis, not a scene-candidate-based workflow.",
            "The goal is ad/non-ad audio feature comparison and boundary refinement support signal exploration.",
            "",
            "## 2. Selected Input Files",
            f"- selected label file: {report.get('selected_label_file')}",
            f"- selected label modified time: {report.get('selected_label_file_modified_time')}",
            f"- manifest file: {MANIFEST_PATH}",
            f"- raw video directory: {RAW_VIDEO_DIR}",
            "",
            "## 3. Generated Files",
            *output_lines,
            "",
            "## latest_for_chatgpt Files",
            *latest_lines,
            "",
            "## 4. Key Row Counts",
            f"- ad_interval_count: {report.get('ad_interval_count')}",
            f"- planned_segment_count: {report.get('planned_segment_count')}",
            f"- feature_success_count: {report.get('decoded_segment_success_count')}",
            f"- feature_failed_count: {report.get('decoded_segment_failed_count')}",
            f"- random_non_ad_success_count: {report.get('random_non_ad_success_count')}",
            f"- random_non_ad_failed_count: {report.get('random_non_ad_failed_count')}",
            "",
            "## Segment Count By Type",
            *segment_count_lines,
            "",
            "## 5. Main Audio Feature Summary",
            "### audio_rms_mean",
            table_for_feature("audio_rms_mean"),
            "",
            "### audio_log_energy_mean",
            table_for_feature("audio_log_energy_mean"),
            "",
            "### silence_ratio",
            table_for_feature("silence_ratio"),
            "",
            "### spectral_centroid_mean",
            table_for_feature("spectral_centroid_mean"),
            "",
            "### ad_full / random / pre / post comparisons",
            *comparison_lines,
            "",
            "## 6. Sub Agent Validation Results",
            *subagent_lines,
            "",
            "## 7. Warnings And Errors",
            "### Warnings",
            *warning_lines,
            "",
            "### Errors",
            *error_lines,
            "",
            "## 8. Interpretation Notes",
            "This is an intermediate multimodal feature candidate analysis.",
            "The results should be interpreted as audio cue analysis for labeled segments, not as a final detection result.",
            "Audio-only boundary detection or final accuracy claims are intentionally not made here.",
            "",
            "## 9. Suggested Next Steps",
            "- Extend high-signal features to 5-second windows around boundaries.",
            "- Combine useful audio cues with scene candidates in a later separate workflow.",
            "- Keep ASR or keyword features as a separate task.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return text


def save_json(path: Path, data: Dict[str, Any]) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            value = float(obj)
            return None if math.isnan(value) or math.isinf(value) else value
        if isinstance(obj, float):
            return None if math.isnan(obj) or math.isinf(obj) else obj
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    path.write_text(json.dumps(convert(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimated-runtime", default="20-40 minutes")
    args = parser.parse_args()

    start_epoch = time.time()
    start_time = utc_now_iso()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    warnings: List[str] = []
    errors: List[str] = []

    for directory in [DATA_AUDIO_DIR, REPORTS_DIR, LOGS_DIR, SCRIPTS_AUDIO_DIR, LATEST_DIR, BACKUPS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    LOGGER.log("[STEP 01] Start task and create old project snapshot")
    LOGGER.log(f"Task start time: {start_time}")
    old_before = old_project_snapshot()

    LOGGER.log("[STEP 02] Discover latest valid ad labeling file")
    candidates, selected = discover_label_candidates()
    selected_label_file = Path(selected["path"])
    version = extract_version(selected_label_file)
    label_mtime = selected["modified_time"]
    label_mtime_ns_before = selected_label_file.stat().st_mtime_ns

    run_log_path = LOGS_DIR / f"extract_labeled_audio_features_{version}_run_log.txt"
    LOGGER.set_path(run_log_path)
    LOGGER.log(f"Selected label file: {selected_label_file}")
    LOGGER.log(f"Selected label reason: priority={selected['priority']}; {selected['reason']}; mtime={label_mtime}; version={version}")

    backup_dir = backup_existing_targets(version, timestamp)
    if backup_dir:
        LOGGER.log(f"Backed up existing same-name outputs to {backup_dir}")
        warnings.append(f"existing outputs backed up to {backup_dir}")

    output_files = {
        "sampling_plan": DATA_AUDIO_DIR / f"audio_labeled_segment_sampling_plan_{version}.csv",
        "features": DATA_AUDIO_DIR / f"audio_labeled_segment_features_{version}.csv",
        "probe": DATA_AUDIO_DIR / f"audio_probe_metadata_{version}.csv",
        "group_summary": DATA_AUDIO_DIR / f"audio_labeled_segment_group_summary_{version}.csv",
        "report": REPORTS_DIR / f"extract_labeled_audio_features_{version}_report.json",
        "summary": REPORTS_DIR / f"extract_labeled_audio_features_{version}_summary.md",
        "run_log": run_log_path,
        "script": SCRIPT_PATH,
    }

    intervals = canonicalize_intervals(selected_label_file, version)
    input_row_count = len(intervals)
    invalid_interval_count = int((~intervals["interval_valid"]).sum())
    if invalid_interval_count:
        warnings.append(f"{invalid_interval_count} interval rows are invalid and kept with failure status")
    if version == "v2_4" and input_row_count != 22:
        warnings.append(f"v2_4 row count is {input_row_count}, expected approximately 22")
    LOGGER.log(f"Input interval row count: {input_row_count}")

    LOGGER.log("[STEP 03] Load video manifest and resolve raw video paths")
    manifest = load_manifest()
    missing_input_files: List[str] = []
    if manifest.empty:
        missing_input_files.append(str(MANIFEST_PATH))
        warnings.append("video manifest missing or empty")
    resolved_paths = resolve_video_paths(intervals, manifest)
    unresolved_video_ids = sorted(resolved_paths.loc[~resolved_paths["video_path_exists"], "video_id"].dropna().unique().tolist())
    if unresolved_video_ids:
        warnings.append(f"unresolved raw video paths for video_ids={unresolved_video_ids}")

    LOGGER.log("[STEP 05] Validate ffmpeg/ffprobe and audio streams")
    ffmpeg_path = executable_path("ffmpeg")
    ffprobe_path = executable_path("ffprobe")
    if not ffmpeg_path:
        errors.append("ffmpeg executable not found")
    if not ffprobe_path:
        errors.append("ffprobe executable not found")
    probe_df = probe_videos(resolved_paths, ffprobe_path)
    probe_df.to_csv(output_files["probe"], index=False, encoding="utf-8-sig")
    audio_success = int(probe_df["audio_available"].sum()) if not probe_df.empty else 0
    audio_failed = int(len(probe_df) - audio_success)
    LOGGER.log(f"Video/audio stream success count: {audio_success}; failed count: {audio_failed}")

    LOGGER.log("[STEP 04] Build labeled audio segment sampling plan")
    plan_df = build_sampling_plan(intervals, resolved_paths, probe_df, version, selected_label_file, label_mtime)
    plan_df.to_csv(output_files["sampling_plan"], index=False, encoding="utf-8-sig")
    LOGGER.log(f"Segment plan row count: {len(plan_df)}")

    LOGGER.log("[STEP 06] Extract audio features by segment")
    features_df = extract_features(plan_df, probe_df, ffmpeg_path)
    features_df.to_csv(output_files["features"], index=False, encoding="utf-8-sig")
    feature_success = int((features_df["feature_status"] == "success").sum())
    feature_failed = int(len(features_df) - feature_success)
    LOGGER.log(f"Feature extraction success count: {feature_success}; failure count: {feature_failed}")

    LOGGER.log("[STEP 07] Generate summary statistics")
    group_summary = write_group_summary(features_df, output_files["group_summary"])
    comparisons = key_comparisons(features_df)
    nan_ratio, inf_count, out_of_range = nan_inf_sanity(features_df)

    random_rows = plan_df[plan_df["segment_type"] == "random_non_ad_30s"]
    segment_count_by_type = plan_df["segment_type"].value_counts(dropna=False).to_dict()
    failed_video_ids = sorted(
        probe_df.loc[~probe_df["audio_available"], "video_id"].astype(str).tolist()
    )
    failed_reason_by_video = {
        str(row["video_id"]): row.get("probe_error", "")
        for _, row in probe_df.loc[~probe_df["audio_available"]].iterrows()
    }

    report: Dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "task_name": TASK_NAME,
        "version": version,
        "estimated_runtime": args.estimated_runtime,
        "start_time": start_time,
        "end_time": None,
        "actual_runtime_seconds": None,
        "actual_runtime_readable": None,
        "selected_label_file": str(selected_label_file),
        "selected_label_file_modified_time": label_mtime,
        "label_file_candidates": candidates,
        "selected_label_reason": f"Selected by priority data/segments/ad_interval_segments_*.csv first, then latest modified time and version suffix. priority={selected['priority']}; {selected['reason']}",
        "input_files": [str(selected_label_file), str(MANIFEST_PATH)],
        "output_files": [str(path) for path in output_files.values()],
        "generated_files": [str(path) for path in output_files.values()],
        "latest_for_chatgpt_files": [],
        "missing_input_files": missing_input_files,
        "missing_required_columns": [],
        "warnings": warnings,
        "errors": errors,
        "old_project_modified": None,
        "old_project_snapshot_before": old_before,
        "old_project_snapshot_after": None,
        "sub_agent_results": {},
        "ad_interval_count": int(input_row_count),
        "video_count": int(intervals["video_id"].nunique()),
        "planned_segment_count": int(len(plan_df)),
        "expected_planned_segment_count": int(input_row_count * 4),
        "segment_count_by_type": {str(k): int(v) for k, v in segment_count_by_type.items()},
        "random_non_ad_success_count": int((random_rows["sampling_status"] == "sampled").sum()),
        "random_non_ad_failed_count": int((random_rows["sampling_status"] == "random_sampling_failed").sum()),
        "ffmpeg_available": bool(ffmpeg_path),
        "ffmpeg_path": ffmpeg_path,
        "ffprobe_available": bool(ffprobe_path),
        "ffprobe_path": ffprobe_path,
        "audio_stream_success_count": audio_success,
        "audio_stream_failed_count": audio_failed,
        "failed_video_ids": failed_video_ids,
        "failed_reason_by_video": failed_reason_by_video,
        "decoded_segment_success_count": feature_success,
        "decoded_segment_failed_count": feature_failed,
        "feature_columns": FEATURE_COLUMNS,
        "nan_ratio_by_feature": nan_ratio,
        "inf_count_by_feature": inf_count,
        "out_of_range_feature_counts": out_of_range,
        "silence_ratio_min": safe_float(pd.to_numeric(features_df["silence_ratio"], errors="coerce").min()),
        "silence_ratio_max": safe_float(pd.to_numeric(features_df["silence_ratio"], errors="coerce").max()),
        "low_energy_ratio_min": safe_float(pd.to_numeric(features_df["low_energy_ratio"], errors="coerce").min()),
        "low_energy_ratio_max": safe_float(pd.to_numeric(features_df["low_energy_ratio"], errors="coerce").max()),
        "group_summary_path": str(output_files["group_summary"]),
        "key_comparisons": comparisons,
        "interpretation_note": "audio cue analysis and labeled audio feature extraction for exploratory boundary refinement support signal; no final performance claim",
        "latest_for_chatgpt_forbidden_files_found": [],
        "input_label_file_modified": False,
        "old_project_modified": None,
        "backup_dir": str(backup_dir) if backup_dir else "",
        "audio_settings": {
            "sample_rate": SAMPLE_RATE,
            "mono": True,
            "frame_length": FRAME_LENGTH,
            "hop_length": HOP_LENGTH,
            "random_seed": RANDOM_SEED,
            "silence_rms_threshold": SILENCE_RMS_THRESHOLD,
            "low_energy_rms_threshold": LOW_ENERGY_RMS_THRESHOLD,
            "tempo_estimate": "NaN; not computed in scipy/numpy fallback",
        },
    }

    LOGGER.log("[STEP 08] Run Sub Agent validations")
    sub_agent_results = {
        "latest_label_schema_validation": validate_latest_label(intervals, report),
        "sampling_plan_validation": validate_sampling_plan(plan_df, intervals),
        "audio_feature_validation": validate_audio_features(features_df, report),
    }

    LOGGER.log("[STEP 09] Update latest_for_chatgpt with allowed files only")
    report["latest_for_chatgpt_files"] = copy_latest(output_files, version, report)
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden_latest_files()

    label_mtime_ns_after = selected_label_file.stat().st_mtime_ns
    report["input_label_file_modified"] = label_mtime_ns_before != label_mtime_ns_after
    if report["input_label_file_modified"]:
        errors.append("input label file modified during task")

    old_after = old_project_snapshot()
    report["old_project_snapshot_after"] = old_after
    report["old_project_modified"] = old_before != old_after
    if report["old_project_modified"]:
        errors.append("old project snapshot changed during task")

    # summary/latest side effect가 생긴 뒤 analysis와 safety validator를 추가한다.
    provisional_summary = write_summary_md(output_files["summary"], report, group_summary, sub_agent_results)
    sub_agent_results["analysis_interpretation_validation"] = validate_analysis(
        features_df, group_summary, provisional_summary
    )
    save_json(output_files["report"], report)
    sub_agent_results["output_safety_validation"] = validate_output_safety(report)
    report["sub_agent_results"] = sub_agent_results

    for name, result in sub_agent_results.items():
        LOGGER.log(
            f"Sub Agent {name}: {result['status']} "
            f"warnings={len(result.get('warnings', []))} errors={len(result.get('errors', []))}"
        )
        for warning in result.get("warnings", []):
            if warning not in warnings:
                warnings.append(f"{name}: {warning}")
        for error in result.get("errors", []):
            if error not in errors:
                errors.append(f"{name}: {error}")

    report["warnings"] = warnings
    report["errors"] = errors

    final_summary = write_summary_md(output_files["summary"], report, group_summary, sub_agent_results)
    # 갱신된 report와 summary를 latest copy로 다시 반영한다.
    report["latest_for_chatgpt_files"] = copy_latest(output_files, version, report)
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden_latest_files()

    end_epoch = time.time()
    end_time = utc_now_iso()
    report["end_time"] = end_time
    report["actual_runtime_seconds"] = round(end_epoch - start_epoch, 3)
    report["actual_runtime_readable"] = readable_seconds(end_epoch - start_epoch)
    report["warnings"] = warnings
    report["errors"] = errors

    save_json(output_files["report"], report)
    # runtime과 validation 결과가 들어간 뒤 final report를 복사한다.
    report["latest_for_chatgpt_files"] = copy_latest(output_files, version, report)
    save_json(output_files["report"], report)

    LOGGER.log(f"Task end time: {end_time}")
    LOGGER.log(f"Actual runtime: {report['actual_runtime_readable']}")
    LOGGER.log(f"Warnings: {warnings}")
    LOGGER.log(f"Errors: {errors}")
    LOGGER.log("[STEP 10] Print final JSON result")

    final_json = {
        "task_name": TASK_NAME,
        "version": version,
        "selected_label_file": str(selected_label_file),
        "selected_label_file_modified_time": label_mtime,
        "input_files": [str(selected_label_file), str(MANIFEST_PATH)],
        "output_files": [str(path) for path in output_files.values()],
        "key_counts": {
            "ad_interval_count": int(input_row_count),
            "planned_segment_count": int(len(plan_df)),
            "feature_success_count": feature_success,
            "feature_failed_count": feature_failed,
            "random_non_ad_success_count": int((random_rows["sampling_status"] == "sampled").sum()),
            "random_non_ad_failed_count": int((random_rows["sampling_status"] == "random_sampling_failed").sum()),
            "video_count": int(intervals["video_id"].nunique()),
            "audio_stream_success_count": audio_success,
            "audio_stream_failed_count": audio_failed,
        },
        "sub_agent_status": {name: result["status"] for name, result in sub_agent_results.items()},
        "summary_path": str(output_files["summary"]),
        "report_path": str(output_files["report"]),
        "log_path": str(output_files["run_log"]),
        "old_project_modified": bool(report["old_project_modified"]),
        "latest_for_chatgpt_forbidden_files_found": report["latest_for_chatgpt_forbidden_files_found"],
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(final_json, ensure_ascii=False, indent=2), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
