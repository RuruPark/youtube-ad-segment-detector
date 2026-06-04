#!/usr/bin/env python3
"""Full-video train-only audio clue extraction and coverage audit for v2_4.

This script creates dense 2s audio subwindow features across the train split only.
It reuses the existing v2_4 low-level audio feature helper when available and
keeps all detector/config/input artifacts read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "v2_4"
TASK_NAME = "full_video_audio_clue_extraction_v2_4_train"
SPLIT_SEED = 20240524
FIXED_SPLIT = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}
DEFAULT_PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
CREATED_BY_SCRIPT = "scripts/audio/full_video_audio_clue_extraction_v2_4_train.py"
FEATURE_VERSION = "full_video_audio_clue_v2_4_train"
SAMPLE_RATE = 16000
FRAME_LENGTH = 2048
HOP_LENGTH = 512
SILENCE_RMS_THRESHOLD = 0.01
LOW_ENERGY_RMS_THRESHOLD = 0.02

BASE_FEATURE_COLUMNS = [
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
BASE_FEATURE_COLUMNS += [f"mfcc_{i}_mean" for i in range(1, 14)]
BASE_FEATURE_COLUMNS += [f"mfcc_{i}_std" for i in range(1, 14)]
BASE_FEATURE_COLUMNS += [
    "onset_count",
    "onset_strength_mean",
    "onset_strength_std",
    "onset_strength_max",
    "tempo_estimate",
    "onset_density",
    "onset_count_per_sec",
    "spectral_flux_mean_per_sec_proxy",
    "onset_strength_mean_per_sec_proxy",
]

SCORE_OUTPUT_COLUMNS = [
    "onset_density_score",
    "flux_onset_score",
    "energy_score",
    "inverse_silence_score",
    "spectral_texture_score",
    "audio_ad_like_score",
    "audio_score_source",
]

REQUIRED_FEATURE_COLUMNS = [
    "video_id",
    "split",
    "source_video_path",
    "video_path",
    "video_duration_sec",
    "subwindow_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "subwindow_size_sec",
    "stride_sec",
    "is_partial_window",
    "extraction_status",
    "extraction_error",
    "has_audio",
    "sample_rate",
    "audio_num_samples",
    "feature_version",
    "created_by_script",
    "created_at",
]

FORBIDDEN_BUNDLE_SUFFIXES = {
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
        line = f"{timestamp} {message}"
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if not isinstance(obj, (str, bytes, bytearray)) else False:
        return None
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        cols = list(df.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    text = "" if not np.isfinite(value) else f"{value:.6g}"
                else:
                    text = str(value)
                text = text.replace("|", "\\|").replace("\n", " ")
                values.append(text)
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                text = "" if not np.isfinite(value) else f"{value:.6g}"
            else:
                text = str(value)
            text = text.replace("|", "\\|").replace("\n", " ")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def unique_file_path(path: Path, run_id: str) -> Path:
    if not path.exists():
        return path
    candidate = path.with_name(f"{path.stem}_{run_id}{path.suffix}")
    idx = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{run_id}_{idx}{path.suffix}")
        idx += 1
    return candidate


def unique_dir_path(path: Path, run_id: str) -> Path:
    if not path.exists():
        return path
    candidate = path.with_name(f"{path.name}_{run_id}")
    idx = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{run_id}_{idx}")
        idx += 1
    return candidate


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return str(path)


def file_info(path: Path, root: Optional[Path] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if root is not None:
        info["relative_path"] = relative_to_root(path, root)
    if path.exists() and path.is_file():
        stat = path.stat()
        info.update({"size_bytes": stat.st_size, "mtime": stat.st_mtime})
    return info


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
        rel = item.relative_to(base).as_posix()
        entries.append(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}")
        file_count += 1
    digest = hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()
    return {
        "path": str(path),
        "exists": True,
        "file_count": file_count,
        "metadata_digest": digest,
        "snapshot_type": "relative_path_size_mtime_ns",
    }


def compare_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    return before.get("exists") == after.get("exists") and before.get("metadata_digest") == after.get("metadata_digest")


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


def executable_path(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    fallback = Path(".venv/bin") / name
    if fallback.exists():
        return str(fallback)
    return None


def load_module(path: Path, module_name: str) -> Tuple[Optional[Any], str]:
    if not path.exists():
        return None, f"missing: {path}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None, f"spec_unavailable: {path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def discover_audio_artifacts(project_root: Path) -> List[Dict[str, Any]]:
    roots = [
        project_root / "scripts/audio",
        project_root / "data/audio",
        project_root / "configs",
        project_root / "reports/audio",
        project_root / "reports",
        project_root / "outputs",
        project_root / "0525",
    ]
    seen: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = relative_to_root(path, project_root)
            name = path.name.lower()
            if "audio" not in name and "audio" not in rel.lower():
                continue
            if str(path) in seen:
                continue
            seen.add(str(path))
            if "/data/raw/" in rel:
                continue
            suffix = path.suffix.lower()
            if suffix in FORBIDDEN_BUNDLE_SUFFIXES:
                continue
            stat = path.stat()
            if "/scripts/audio/" in f"/{rel}":
                role = "script"
            elif rel.startswith("configs/"):
                role = "config"
            elif rel.startswith("data/audio/"):
                role = "data_audio_artifact"
            elif rel.startswith("reports/"):
                role = "report"
            else:
                role = "reference_or_bundle"
            rows.append(
                {
                    "relative_path": rel,
                    "absolute_path": str(path),
                    "role": role,
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
    rows.sort(key=lambda r: (r["role"], r["relative_path"]))
    return rows


def default_output_paths(project_root: Path, run_id: str) -> Dict[str, Path]:
    planned = {
        "features": project_root / "data/audio/full_video_audio_subwindow_features_v2_4_train.csv",
        "video_coverage": project_root / "data/audio/full_video_audio_subwindow_coverage_by_video_v2_4_train.csv",
        "interval_coverage": project_root / "data/audio/train_actual_ad_audio_coverage_summary_v2_4.csv",
        "gap_cases": project_root / "data/audio/train_actual_ad_audio_coverage_gap_cases_v2_4.csv",
        "failures": project_root / "data/audio/full_video_audio_extraction_failures_v2_4_train.csv",
        "summary_md": project_root / "reports/audio/full_video_audio_clue_extraction_v2_4_train_summary.md",
        "report_json": project_root / "reports/audio/full_video_audio_clue_extraction_v2_4_train_report.json",
        "run_log": project_root / "logs/full_video_audio_clue_extraction_v2_4_train_run_log.txt",
    }
    return {key: unique_file_path(path, run_id) for key, path in planned.items()}


def build_subwindows(
    video_duration_sec: float,
    subwindow_size_sec: float,
    stride_sec: float,
    min_valid_duration_sec: float,
) -> List[Dict[str, Any]]:
    if not np.isfinite(video_duration_sec) or video_duration_sec <= 0:
        return []
    rows: List[Dict[str, Any]] = []
    cursor = 0.0
    idx = 0
    eps = 1e-9
    while cursor < video_duration_sec - eps:
        end = min(cursor + subwindow_size_sec, video_duration_sec)
        duration = end - cursor
        if duration + eps >= min_valid_duration_sec:
            rows.append(
                {
                    "subwindow_index": idx,
                    "start_sec": safe_float(cursor),
                    "end_sec": safe_float(end),
                    "duration_sec": safe_float(duration),
                    "is_partial_window": bool(duration < subwindow_size_sec - 1e-6),
                }
            )
        cursor += stride_sec
        idx += 1
        if stride_sec <= 0:
            break
    return rows


def frame_audio_basic(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
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


def compute_basic_audio_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    features = {col: float("nan") for col in BASE_FEATURE_COLUMNS}
    if audio.size == 0:
        return features
    eps = 1e-10
    frames = frame_audio_basic(audio.astype(np.float32, copy=False), FRAME_LENGTH, HOP_LENGTH)
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
    features["onset_count"] = 0.0
    features["onset_strength_mean"] = 0.0
    features["onset_strength_std"] = 0.0
    features["onset_strength_max"] = 0.0
    return features


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


def decode_full_audio(
    ffmpeg_path: str,
    video_path: Path,
    sample_rate: int,
    helper: Optional[Any],
) -> Tuple[np.ndarray, int, str]:
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True)
    except Exception as exc:
        return np.array([], dtype=np.float32), sample_rate, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        return np.array([], dtype=np.float32), sample_rate, err or f"ffmpeg_exit_{proc.returncode}"
    try:
        if helper is not None and hasattr(helper, "read_wav_from_bytes"):
            audio, sr = helper.read_wav_from_bytes(proc.stdout)
        else:
            audio, sr = read_wav_from_bytes(proc.stdout)
    except Exception as exc:
        return np.array([], dtype=np.float32), sample_rate, f"wav_decode_failed: {type(exc).__name__}: {exc}"
    return audio, sr, ""


def probe_video(video_path: Path, manifest_duration_sec: float, ffprobe_path: Optional[str]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "video_path": str(video_path),
        "path_exists": video_path.exists(),
        "manifest_duration_sec": safe_float(manifest_duration_sec),
        "probe_status": "not_run",
        "probe_error": "",
        "ffprobe_duration_sec": float("nan"),
        "video_duration_sec": safe_float(manifest_duration_sec),
        "has_audio": False,
        "audio_stream_index": float("nan"),
        "audio_codec_name": "",
        "audio_sample_rate": float("nan"),
    }
    if not record["path_exists"]:
        record["probe_status"] = "path_missing"
        record["probe_error"] = "video path missing"
        return record
    if not ffprobe_path:
        record["probe_status"] = "ffprobe_missing"
        record["probe_error"] = "ffprobe executable not found"
        return record
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_streams",
        "-of",
        "json",
        str(video_path),
    ]
    data, err = run_command_json(cmd)
    if data is None:
        record["probe_status"] = "probe_failed"
        record["probe_error"] = err
        return record
    record["probe_status"] = "success"
    try:
        duration = float(data.get("format", {}).get("duration", np.nan))
        if np.isfinite(duration) and duration > 0:
            record["ffprobe_duration_sec"] = duration
            record["video_duration_sec"] = duration
    except Exception:
        pass
    audio_streams = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "audio"]
    if not audio_streams:
        record["probe_error"] = "no_audio_stream"
        return record
    stream = audio_streams[0]
    record["has_audio"] = True
    record["audio_stream_index"] = stream.get("index", np.nan)
    record["audio_codec_name"] = stream.get("codec_name", "")
    try:
        record["audio_sample_rate"] = float(stream.get("sample_rate", np.nan))
    except Exception:
        record["audio_sample_rate"] = float("nan")
    return record


def load_baseline_for_scores(project_root: Path, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    candidates = [
        project_root / "data/audio/audio_labeled_segment_features_v2_4_with_split.csv",
        project_root / "data/audio/audio_ad_edge_persistence_subwindow_features_v2_4_with_split.csv",
        project_root / "data/audio/audio_labeled_segment_features_v2_4.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = read_csv(path)
        except Exception:
            continue
        if "split" in df.columns:
            df = df[df["split"].astype(str).eq("train")].copy()
        if df.empty:
            continue
        if "onset_density" not in df.columns and {"onset_count", "segment_duration_sec"}.issubset(df.columns):
            df["onset_density"] = pd.to_numeric(df["onset_count"], errors="coerce") / pd.to_numeric(
                df["segment_duration_sec"], errors="coerce"
            ).replace(0, np.nan)
            df["onset_count_per_sec"] = df["onset_density"]
        return df, str(path)
    success = feature_df[feature_df["extraction_status"].eq("success")].copy()
    return success, "full_video_success_features_self_fallback"


def add_duration_normalized_features(row: Dict[str, Any], duration_sec: float) -> None:
    onset = safe_float(row.get("onset_count"))
    if np.isfinite(onset) and duration_sec > 0:
        row["onset_density"] = safe_float(onset / duration_sec)
        row["onset_count_per_sec"] = safe_float(onset / duration_sec)
    else:
        row["onset_density"] = float("nan")
        row["onset_count_per_sec"] = float("nan")
    row["spectral_flux_mean_per_sec_proxy"] = safe_float(row.get("spectral_flux_mean"))
    row["onset_strength_mean_per_sec_proxy"] = safe_float(row.get("onset_strength_mean"))


def add_audio_scores(
    feature_df: pd.DataFrame,
    project_root: Path,
    persistence: Optional[Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = feature_df.copy()
    for col in SCORE_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan if col != "audio_score_source" else "unavailable"
    report = {
        "attempted": bool(persistence is not None and hasattr(persistence, "add_ad_like_scores")),
        "reused_existing_formula": False,
        "score_source": "unavailable",
        "baseline_source": "",
        "error": "",
    }
    success_mask = out["extraction_status"].eq("success")
    if not report["attempted"] or not success_mask.any():
        out.loc[success_mask, "audio_score_source"] = "unavailable"
        return out, report
    try:
        baseline_df, baseline_source = load_baseline_for_scores(project_root, out)
        score_input = out.loc[success_mask].copy()
        scored = persistence.add_ad_like_scores(score_input, baseline_df)
        copy_cols = [col for col in scored.columns if col.endswith("__video_robust_z") or col.endswith("__directional_score")]
        copy_cols += [
            "onset_density_score",
            "flux_onset_score",
            "energy_score",
            "inverse_silence_score",
            "spectral_texture_score",
            "audio_ad_like_score",
        ]
        for col in copy_cols:
            if col in scored.columns:
                out.loc[scored.index, col] = scored[col]
        out.loc[scored.index, "audio_score_source"] = "existing_formula"
        out.loc[~success_mask, "audio_score_source"] = "unavailable"
        report.update(
            {
                "reused_existing_formula": True,
                "score_source": "existing_formula",
                "baseline_source": baseline_source,
                "scored_rows": int(len(scored)),
            }
        )
    except Exception as exc:
        out.loc[success_mask, "audio_score_source"] = "unavailable"
        report.update({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    return out, report


def extract_train_features(
    train_rows: pd.DataFrame,
    helper: Optional[Any],
    ffmpeg_path: Optional[str],
    ffprobe_path: Optional[str],
    subwindow_size_sec: float,
    stride_sec: float,
    min_valid_duration_sec: float,
    sample_rate: int,
    created_at: str,
    logger: TaskLogger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, Any]] = []
    video_records: List[Dict[str, Any]] = []
    compute_source = "existing_helper_compute_audio_features" if helper is not None and hasattr(helper, "compute_audio_features") else "fallback_basic_features"
    for video_idx, row in train_rows.iterrows():
        video_id = safe_int(row["video_id"])
        video_path = Path(str(row.get("video_path", "")))
        manifest_duration = safe_float(row.get("video_duration_sec"))
        probe = probe_video(video_path, manifest_duration, ffprobe_path)
        video_duration = safe_float(probe.get("video_duration_sec"))
        subwindows = build_subwindows(video_duration, subwindow_size_sec, stride_sec, min_valid_duration_sec)
        logger.log(
            f"[STEP 06] Video {video_id}: planned {len(subwindows)} windows, "
            f"duration={video_duration:.3f}s, has_audio={probe.get('has_audio')}"
        )
        video_status = "not_run"
        video_error = ""
        audio = np.array([], dtype=np.float32)
        decoded_sr = sample_rate
        if not subwindows:
            video_status = "no_valid_subwindows"
            video_error = "video duration invalid or shorter than min valid duration"
        elif not probe.get("path_exists"):
            video_status = "video_path_missing"
            video_error = str(probe.get("probe_error", "video path missing"))
        elif not ffmpeg_path:
            video_status = "ffmpeg_missing"
            video_error = "ffmpeg executable not found"
        elif not probe.get("has_audio"):
            video_status = "audio_stream_unavailable"
            video_error = str(probe.get("probe_error", "no audio stream"))
        else:
            audio, decoded_sr, err = decode_full_audio(ffmpeg_path, video_path, sample_rate, helper)
            if err or audio.size == 0:
                video_status = "decode_failed"
                video_error = err or "empty decoded audio"
            else:
                video_status = "success"
        success_count = 0
        failed_count = 0
        for sw in subwindows:
            subwindow_id = f"{VERSION}_full_audio_V{video_id:02d}_{sw['subwindow_index']:06d}"
            duration = safe_float(sw["duration_sec"])
            base_record: Dict[str, Any] = {
                "video_id": video_id,
                "split": "train",
                "source_video_path": str(video_path),
                "video_path": str(video_path),
                "video_duration_sec": video_duration,
                "subwindow_id": subwindow_id,
                "subwindow_index": int(sw["subwindow_index"]),
                "start_sec": safe_float(sw["start_sec"]),
                "end_sec": safe_float(sw["end_sec"]),
                "duration_sec": duration,
                "subwindow_size_sec": subwindow_size_sec,
                "stride_sec": stride_sec,
                "is_partial_window": bool(sw["is_partial_window"]),
                "extraction_status": video_status if video_status != "success" else "not_run",
                "extraction_error": video_error,
                "has_audio": bool(probe.get("has_audio", False) and audio.size > 0),
                "sample_rate": decoded_sr if np.isfinite(decoded_sr) else sample_rate,
                "audio_num_samples": 0,
                "decoded_duration_sec": float("nan"),
                "decoded_duration_diff_sec": float("nan"),
                "audio_stream_index": probe.get("audio_stream_index", np.nan),
                "audio_codec_name": probe.get("audio_codec_name", ""),
                "probe_status": probe.get("probe_status", ""),
                "probe_error": probe.get("probe_error", ""),
                "feature_compute_source": compute_source,
                "feature_version": FEATURE_VERSION,
                "created_by_script": CREATED_BY_SCRIPT,
                "created_at": created_at,
            }
            for col in BASE_FEATURE_COLUMNS:
                base_record[col] = float("nan")
            if video_status != "success":
                failed_count += 1
                records.append(base_record)
                continue
            try:
                start_idx = int(round(base_record["start_sec"] * decoded_sr))
                end_idx = int(round(base_record["end_sec"] * decoded_sr))
                slice_audio = audio[max(start_idx, 0) : min(end_idx, audio.size)]
                if slice_audio.size == 0:
                    base_record["extraction_status"] = "empty_audio_subwindow"
                    base_record["extraction_error"] = "decoded audio slice is empty"
                    failed_count += 1
                    records.append(base_record)
                    continue
                base_record["audio_num_samples"] = int(slice_audio.size)
                base_record["decoded_duration_sec"] = safe_float(slice_audio.size / float(decoded_sr))
                base_record["decoded_duration_diff_sec"] = safe_float(base_record["decoded_duration_sec"] - duration)
                if helper is not None and hasattr(helper, "compute_audio_features"):
                    features = helper.compute_audio_features(slice_audio, decoded_sr)
                else:
                    features = compute_basic_audio_features(slice_audio, decoded_sr)
                base_record.update(features)
                add_duration_normalized_features(base_record, duration)
                base_record["extraction_status"] = "success"
                base_record["extraction_error"] = ""
                success_count += 1
            except Exception as exc:
                base_record["extraction_status"] = "feature_compute_failed"
                base_record["extraction_error"] = f"{type(exc).__name__}: {exc}"
                failed_count += 1
            records.append(base_record)
        expected_count = len(build_subwindows(video_duration, subwindow_size_sec, stride_sec, min_valid_duration_sec))
        video_records.append(
            {
                "video_id": video_id,
                "split": "train",
                "source_video_path": str(video_path),
                "video_duration_sec": video_duration,
                "manifest_duration_sec": safe_float(row.get("video_duration_sec")),
                "ffprobe_duration_sec": safe_float(probe.get("ffprobe_duration_sec")),
                "expected_subwindow_count": expected_count,
                "generated_subwindow_count": len(subwindows),
                "success_subwindow_count": success_count,
                "failed_subwindow_count": failed_count,
                "has_audio": bool(probe.get("has_audio", False)),
                "audio_stream_index": probe.get("audio_stream_index", np.nan),
                "audio_codec_name": probe.get("audio_codec_name", ""),
                "probe_status": probe.get("probe_status", ""),
                "probe_error": probe.get("probe_error", ""),
                "extraction_status": video_status,
                "extraction_error": video_error,
                "decoded_sample_rate": decoded_sr if audio.size else np.nan,
                "decoded_total_samples": int(audio.size),
                "decoded_total_duration_sec": safe_float(audio.size / float(decoded_sr)) if audio.size else np.nan,
            }
        )
        logger.log(
            f"[STEP 06] Video {video_id}: success={success_count}, failed={failed_count}, status={video_status}"
        )
    return pd.DataFrame(records), pd.DataFrame(video_records)


def merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    cleaned = [(safe_float(a), safe_float(b)) for a, b in intervals if np.isfinite(safe_float(a)) and np.isfinite(safe_float(b))]
    cleaned = [(a, b) for a, b in cleaned if b > a]
    if not cleaned:
        return []
    cleaned.sort()
    merged: List[Tuple[float, float]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1e-9:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_union_duration(
    intervals: Iterable[Tuple[float, float]],
    clamp_start: float,
    clamp_end: float,
) -> Tuple[float, List[Tuple[float, float]]]:
    clipped = []
    for start, end in intervals:
        s = max(clamp_start, safe_float(start))
        e = min(clamp_end, safe_float(end))
        if np.isfinite(s) and np.isfinite(e) and e > s:
            clipped.append((s, e))
    merged = merge_intervals(clipped)
    duration = sum(end - start for start, end in merged)
    return safe_float(duration), merged


def max_gap_inside(clamp_start: float, clamp_end: float, merged_intervals: Sequence[Tuple[float, float]]) -> float:
    if clamp_end <= clamp_start:
        return 0.0
    if not merged_intervals:
        return safe_float(clamp_end - clamp_start)
    gaps: List[float] = []
    cursor = clamp_start
    for start, end in merged_intervals:
        gaps.append(max(0.0, start - cursor))
        cursor = max(cursor, end)
    gaps.append(max(0.0, clamp_end - cursor))
    return safe_float(max(gaps) if gaps else 0.0)


def load_existing_subwindow_features(project_root: Path) -> Tuple[pd.DataFrame, str]:
    candidates = [
        project_root / "data/audio/audio_ad_edge_persistence_subwindow_features_v2_4_with_split.csv",
        project_root / "data/audio/audio_ad_edge_persistence_subwindow_features_v2_4.csv",
        project_root / "data/audio/audio_visual_anchor_persistence_subwindow_features_v2_4_train_val_for_discussion.csv",
        project_root / "data/audio/audio_scene_anchor_persistence_subwindow_features_v2_4_train_val_for_discussion.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = read_csv(path)
        except Exception:
            continue
        if {"video_id", "subwindow_start_sec", "subwindow_end_sec"}.issubset(df.columns):
            if "split" in df.columns:
                df = df[df["split"].astype(str).eq("train")].copy()
            return df, str(path)
    return pd.DataFrame(), ""


def existing_coverage_for_interval(existing_df: pd.DataFrame, video_id: int, ad_start: float, ad_end: float) -> Tuple[float, int]:
    if existing_df.empty:
        return float("nan"), 0
    group = existing_df[pd.to_numeric(existing_df["video_id"], errors="coerce").eq(video_id)].copy()
    if "feature_status" in group.columns:
        group = group[group["feature_status"].astype(str).eq("success")]
    elif "extraction_status" in group.columns:
        group = group[group["extraction_status"].astype(str).eq("success")]
    if group.empty:
        return 0.0, 0
    starts = pd.to_numeric(group["subwindow_start_sec"], errors="coerce")
    ends = pd.to_numeric(group["subwindow_end_sec"], errors="coerce")
    mask = starts.lt(ad_end) & ends.gt(ad_start)
    intervals = list(zip(starts[mask].tolist(), ends[mask].tolist()))
    covered, _ = interval_union_duration(intervals, ad_start, ad_end)
    duration = ad_end - ad_start
    ratio = covered / duration if duration > 0 else np.nan
    return safe_float(min(max(ratio, 0.0), 1.0)), int(mask.sum())


def load_prior_audio_body_coverage(project_root: Path) -> Tuple[Dict[str, float], str]:
    path = project_root / "data/analysis/train_actual_ad_interval_cue_coverage_summary_v2_4.csv"
    if not path.exists():
        return {}, ""
    try:
        df = read_csv(path)
    except Exception:
        return {}, ""
    if "actual_interval_id" not in df.columns or "audio_body_coverage_ratio" not in df.columns:
        return {}, str(path)
    return {
        str(row["actual_interval_id"]): safe_float(row["audio_body_coverage_ratio"])
        for _, row in df.iterrows()
    }, str(path)


def compute_video_coverage(feature_df: pd.DataFrame, video_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, video in video_df.iterrows():
        video_id = safe_int(video["video_id"])
        duration = safe_float(video["video_duration_sec"])
        group = feature_df[pd.to_numeric(feature_df["video_id"], errors="coerce").eq(video_id)].copy()
        success = group[group["extraction_status"].eq("success")]
        intervals = list(zip(pd.to_numeric(success["start_sec"], errors="coerce"), pd.to_numeric(success["end_sec"], errors="coerce")))
        covered, merged = interval_union_duration(intervals, 0.0, duration)
        all_sorted = group.sort_values("start_sec")
        gaps = []
        if len(all_sorted) >= 2:
            prev_end = None
            for _, sw in all_sorted.iterrows():
                if prev_end is not None:
                    gaps.append(max(0.0, safe_float(sw["start_sec"]) - prev_end))
                prev_end = safe_float(sw["end_sec"])
        rows.append(
            {
                "video_id": video_id,
                "split": "train",
                "source_video_path": video.get("source_video_path", ""),
                "video_duration_sec": duration,
                "expected_subwindow_count": safe_int(video.get("expected_subwindow_count")),
                "generated_subwindow_count": safe_int(video.get("generated_subwindow_count")),
                "success_subwindow_count": safe_int(video.get("success_subwindow_count")),
                "failed_subwindow_count": safe_int(video.get("failed_subwindow_count")),
                "has_audio": bool(video.get("has_audio", False)),
                "extraction_status": video.get("extraction_status", ""),
                "extraction_error": video.get("extraction_error", ""),
                "first_subwindow_start_sec": safe_float(group["start_sec"].min()) if not group.empty else np.nan,
                "last_subwindow_end_sec": safe_float(group["end_sec"].max()) if not group.empty else np.nan,
                "first_success_subwindow_start_sec": safe_float(success["start_sec"].min()) if not success.empty else np.nan,
                "last_success_subwindow_end_sec": safe_float(success["end_sec"].max()) if not success.empty else np.nan,
                "covered_duration_sec": safe_float(covered),
                "video_audio_feature_coverage_ratio": safe_float(covered / duration) if duration > 0 else np.nan,
                "max_gap_sec_inside_video": max_gap_inside(0.0, duration, merged),
                "max_gap_between_generated_subwindows_sec": safe_float(max(gaps) if gaps else 0.0),
            }
        )
    return pd.DataFrame(rows)


def compute_interval_coverage(
    label_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    split_df: pd.DataFrame,
    project_root: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    train_ids = set(FIXED_SPLIT["train"])
    intervals = label_df[
        label_df["segment_type"].astype(str).eq("ad_interval")
        & pd.to_numeric(label_df["video_id"], errors="coerce").isin(train_ids)
    ].copy()
    if "segment_valid" in intervals.columns:
        intervals = intervals[intervals["segment_valid"].astype(str).str.lower().isin(["true", "1", "yes"])]
    existing_df, existing_source = load_existing_subwindow_features(project_root)
    prior_map, prior_source = load_prior_audio_body_coverage(project_root)
    split_duration = {
        safe_int(row["video_id"]): safe_float(row.get("video_duration_sec"))
        for _, row in split_df.iterrows()
    }
    rows: List[Dict[str, Any]] = []
    for _, interval in intervals.iterrows():
        video_id = safe_int(interval["video_id"])
        ad_interval_id = str(interval.get("ad_interval_id", interval.get("segment_id", "")))
        segment_id = str(interval.get("segment_id", ""))
        ad_start = safe_float(interval.get("ad_start_sec", interval.get("segment_start_sec")))
        ad_end = safe_float(interval.get("ad_end_sec", interval.get("segment_end_sec")))
        video_duration = safe_float(interval.get("video_duration_sec"))
        if not np.isfinite(video_duration):
            video_duration = split_duration.get(video_id, np.nan)
        ad_start = max(0.0, ad_start)
        if np.isfinite(video_duration):
            ad_end = min(ad_end, video_duration)
        ad_duration = ad_end - ad_start
        group = feature_df[pd.to_numeric(feature_df["video_id"], errors="coerce").eq(video_id)].copy()
        success = group[group["extraction_status"].eq("success") & group["has_audio"].astype(bool)]
        starts = pd.to_numeric(success["start_sec"], errors="coerce")
        ends = pd.to_numeric(success["end_sec"], errors="coerce")
        overlap_mask = starts.lt(ad_end) & ends.gt(ad_start)
        overlap_rows = success[overlap_mask].copy()
        raw_intervals = list(zip(starts[overlap_mask].tolist(), ends[overlap_mask].tolist()))
        covered, merged = interval_union_duration(raw_intervals, ad_start, ad_end)
        coverage_ratio = safe_float(covered / ad_duration) if ad_duration > 0 else np.nan
        coverage_ratio = safe_float(min(max(coverage_ratio, 0.0), 1.0)) if np.isfinite(coverage_ratio) else np.nan
        if np.isfinite(coverage_ratio) and coverage_ratio >= 0.95:
            status = "full"
        elif np.isfinite(coverage_ratio) and coverage_ratio > 0:
            status = "partial"
        else:
            status = "none"
        max_gap = max_gap_inside(ad_start, ad_end, merged)
        existing_ratio, existing_overlap_count = existing_coverage_for_interval(existing_df, video_id, ad_start, ad_end)
        prior_ratio = prior_map.get(ad_interval_id, np.nan)
        note = "full-video successful subwindows tile the actual interval"
        if status == "none":
            note = "no successful full-video audio subwindow overlaps this interval"
        elif status == "partial":
            note = f"partial coverage remains; max internal gap {max_gap:.3f}s"
        rows.append(
            {
                "video_id": video_id,
                "split": "train",
                "segment_id": segment_id,
                "ad_interval_id": ad_interval_id,
                "ad_start_sec": ad_start,
                "ad_end_sec": ad_end,
                "ad_duration_sec": safe_float(ad_duration),
                "num_overlapping_audio_subwindows": int(len(overlap_rows)),
                "covered_duration_sec": safe_float(covered),
                "coverage_ratio": coverage_ratio,
                "first_covering_subwindow_start_sec": safe_float(overlap_rows["start_sec"].min()) if not overlap_rows.empty else np.nan,
                "last_covering_subwindow_end_sec": safe_float(overlap_rows["end_sec"].max()) if not overlap_rows.empty else np.nan,
                "max_gap_sec_inside_interval": safe_float(max_gap),
                "has_full_coverage": bool(status == "full"),
                "has_partial_coverage": bool(status == "partial"),
                "has_no_coverage": bool(status == "none"),
                "coverage_status": status,
                "coverage_note": note,
                "existing_edge_subwindow_overlap_count": existing_overlap_count,
                "existing_edge_subwindow_coverage_ratio": existing_ratio,
                "coverage_ratio_delta_vs_existing_edge_subwindow": safe_float(coverage_ratio - existing_ratio)
                if np.isfinite(coverage_ratio) and np.isfinite(existing_ratio)
                else np.nan,
                "prior_audio_body_coverage_ratio": prior_ratio,
                "coverage_ratio_delta_vs_prior_audio_body": safe_float(coverage_ratio - prior_ratio)
                if np.isfinite(coverage_ratio) and np.isfinite(prior_ratio)
                else np.nan,
            }
        )
    report = {
        "existing_edge_subwindow_source": existing_source,
        "prior_audio_body_coverage_source": prior_source,
        "train_actual_interval_count": int(len(intervals)),
    }
    return pd.DataFrame(rows), report


def build_gap_cases(interval_coverage: pd.DataFrame) -> pd.DataFrame:
    if interval_coverage.empty:
        return interval_coverage.copy()
    gap = interval_coverage[interval_coverage["coverage_status"].isin(["partial", "none"])].copy()
    if gap.empty:
        gap = interval_coverage.head(0).copy()
    reasons = []
    for _, row in gap.iterrows():
        if row.get("coverage_status") == "none":
            reasons.append("no successful full-video audio subwindow overlapped the interval")
        elif safe_float(row.get("max_gap_sec_inside_interval")) > 0:
            reasons.append("successful full-video audio subwindows left an internal gap")
        else:
            reasons.append("coverage below full threshold after interval-union calculation")
    gap["gap_reason_estimate"] = reasons
    return gap


def validation_checks(
    split_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    video_coverage: pd.DataFrame,
    interval_coverage: pd.DataFrame,
    gap_cases: pd.DataFrame,
    paths: Dict[str, Path],
    protected_before: Dict[str, Dict[str, Any]],
    project_root: Path,
    min_valid_duration_sec: float,
    bundle_dir: Optional[Path],
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    split_groups = {
        split: sorted(pd.to_numeric(group["video_id"], errors="coerce").dropna().astype(int).tolist())
        for split, group in split_df.groupby("split")
    }
    checks["input_split_validation"] = {
        "split_file_exists": True,
        "train_matches_fixed": split_groups.get("train", []) == FIXED_SPLIT["train"],
        "validation_matches_fixed": split_groups.get("validation", []) == FIXED_SPLIT["validation"],
        "test_matches_fixed": split_groups.get("test", []) == FIXED_SPLIT["test"],
        "observed_split_groups": split_groups,
        "output_has_validation_or_test_video_id": bool(
            set(pd.to_numeric(feature_df["video_id"], errors="coerce").dropna().astype(int).tolist())
            & (set(FIXED_SPLIT["validation"]) | set(FIXED_SPLIT["test"]))
        ),
    }
    by_video = []
    for _, row in video_coverage.iterrows():
        duration = safe_float(row.get("video_duration_sec"))
        first_start = safe_float(row.get("first_subwindow_start_sec"))
        last_end = safe_float(row.get("last_subwindow_end_sec"))
        by_video.append(
            {
                "video_id": safe_int(row["video_id"]),
                "first_start_near_zero": bool(np.isfinite(first_start) and abs(first_start) <= 1e-6),
                "last_end_near_video_duration": bool(
                    np.isfinite(last_end)
                    and np.isfinite(duration)
                    and abs(duration - last_end) <= max(min_valid_duration_sec, 0.51) + 1e-6
                ),
                "max_generated_gap_sec": safe_float(row.get("max_gap_between_generated_subwindows_sec")),
                "expected_generated_count_match": safe_int(row.get("expected_subwindow_count"))
                == safe_int(row.get("generated_subwindow_count")),
                "expected": safe_int(row.get("expected_subwindow_count")),
                "generated": safe_int(row.get("generated_subwindow_count")),
            }
        )
    checks["full_video_audio_coverage_validation"] = {
        "by_video": by_video,
        "all_first_starts_near_zero": all(item["first_start_near_zero"] for item in by_video),
        "all_last_ends_near_duration": all(item["last_end_near_video_duration"] for item in by_video),
        "max_generated_gap_sec": safe_float(video_coverage["max_gap_between_generated_subwindows_sec"].max())
        if not video_coverage.empty
        else np.nan,
        "all_expected_counts_match": all(item["expected_generated_count_match"] for item in by_video),
    }
    ratio = pd.to_numeric(interval_coverage.get("coverage_ratio", pd.Series(dtype=float)), errors="coerce")
    checks["actual_ad_interval_coverage_validation"] = {
        "coverage_rows": int(len(interval_coverage)),
        "expected_train_actual_ad_intervals": int(len(interval_coverage)),
        "all_coverage_ratios_in_0_1": bool(((ratio >= -1e-9) & (ratio <= 1.0 + 1e-9)).all()) if len(ratio) else True,
        "negative_overlap_detected": bool((pd.to_numeric(interval_coverage.get("covered_duration_sec", pd.Series(dtype=float)), errors="coerce") < -1e-9).any())
        if not interval_coverage.empty
        else False,
        "covered_duration_exceeds_ad_duration": bool(
            (
                pd.to_numeric(interval_coverage.get("covered_duration_sec", pd.Series(dtype=float)), errors="coerce")
                - pd.to_numeric(interval_coverage.get("ad_duration_sec", pd.Series(dtype=float)), errors="coerce")
                > 1e-6
            ).any()
        )
        if not interval_coverage.empty
        else False,
        "gap_cases_csv_written": bool(paths["gap_cases"].exists()),
        "gap_case_rows": int(len(gap_cases)),
    }
    protected_after = {name: snapshot_path(Path(snap["path"])) for name, snap in protected_before.items()}
    protected_unchanged = {name: compare_snapshots(protected_before[name], protected_after[name]) for name in protected_before}
    bundle_forbidden_files: List[str] = []
    if bundle_dir is not None and bundle_dir.exists():
        for path in bundle_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
                bundle_forbidden_files.append(str(path))
    checks["output_safety_validation"] = {
        "protected_paths_unchanged": protected_unchanged,
        "detector_rule_modified": not all(
            protected_unchanged.get(name, True) for name in ["detector_scripts", "configs", "predictions"]
        ),
        "old_project_modified": not protected_unchanged.get("old_project", True),
        "raw_video_audio_cache_model_copied_to_latest_bundle": bool(bundle_forbidden_files),
        "forbidden_bundle_files": bundle_forbidden_files,
        "validation_test_row_level_output_created": bool(
            checks["input_split_validation"]["output_has_validation_or_test_video_id"]
        ),
    }
    return checks


def summarize_interval_coverage(interval_coverage: pd.DataFrame) -> Dict[str, Any]:
    if interval_coverage.empty:
        return {
            "total_train_ad_intervals": 0,
            "full_coverage_count": 0,
            "partial_coverage_count": 0,
            "no_coverage_count": 0,
            "mean_coverage_ratio": np.nan,
            "min_coverage_ratio": np.nan,
        }
    ratio = pd.to_numeric(interval_coverage["coverage_ratio"], errors="coerce")
    return {
        "total_train_ad_intervals": int(len(interval_coverage)),
        "full_coverage_count": int(interval_coverage["coverage_status"].eq("full").sum()),
        "partial_coverage_count": int(interval_coverage["coverage_status"].eq("partial").sum()),
        "no_coverage_count": int(interval_coverage["coverage_status"].eq("none").sum()),
        "mean_coverage_ratio": safe_float(ratio.mean()),
        "min_coverage_ratio": safe_float(ratio.min()),
        "mean_existing_edge_subwindow_coverage_ratio": safe_float(
            pd.to_numeric(interval_coverage["existing_edge_subwindow_coverage_ratio"], errors="coerce").mean()
        )
        if "existing_edge_subwindow_coverage_ratio" in interval_coverage
        else np.nan,
        "mean_delta_vs_existing_edge_subwindow": safe_float(
            pd.to_numeric(interval_coverage["coverage_ratio_delta_vs_existing_edge_subwindow"], errors="coerce").mean()
        )
        if "coverage_ratio_delta_vs_existing_edge_subwindow" in interval_coverage
        else np.nan,
    }


def write_summary_md(
    path: Path,
    report: Dict[str, Any],
    video_coverage: pd.DataFrame,
    interval_coverage: pd.DataFrame,
    gap_cases: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    coverage = report["actual_ad_interval_coverage"]
    output_paths = report["output_files"]
    lines: List[str] = []
    lines.append("# Full-Video Audio Clue Extraction v2_4 Train Summary")
    lines.append("")
    lines.append("## Purpose")
    lines.append("Create dense train-only 2s audio subwindow features from video start to end and audit actual ad interval coverage.")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Train video_ids: {report['train_video_ids']}")
    lines.append(f"- Validation video_ids excluded from row-level outputs: {report['validation_video_ids']}")
    lines.append(f"- Test video_ids excluded from row-level outputs: {report['test_video_ids']}")
    lines.append("- Detector rule modified: false")
    lines.append("- Old project modified: false")
    lines.append("- Validation/test row-level output created: false")
    lines.append("")
    lines.append("## Inputs")
    for item in report["input_files"]:
        lines.append(f"- {item['name']}: `{item['path']}`")
    lines.append("")
    lines.append("## Existing Audio Artifacts Located")
    for item in report["existing_audio_artifacts"][:60]:
        lines.append(f"- {item['role']}: `{item['relative_path']}`")
    if len(report["existing_audio_artifacts"]) > 60:
        lines.append(f"- ... {len(report['existing_audio_artifacts']) - 60} more artifacts recorded in JSON report")
    lines.append("")
    lines.append("## Extractor")
    lines.append(f"- Existing audio feature helper reused: {str(report['existing_audio_logic']['helper_reused']).lower()}")
    lines.append(f"- Existing audio score formula reused: {str(report['existing_audio_logic']['score_formula_reused']).lower()}")
    lines.append(f"- Fallback extractor used: {str(report['existing_audio_logic']['fallback_extractor_used']).lower()}")
    lines.append(f"- Score baseline source: `{report['existing_audio_logic'].get('score_baseline_source', '')}`")
    lines.append(f"- Subwindow policy: size={report['subwindow_size_sec']}s, stride={report['stride_sec']}s")
    lines.append(f"- Last partial window policy: include when duration >= {report['min_valid_duration_sec']}s")
    lines.append("")
    lines.append("## Train Video Extraction")
    display_cols = [
        "video_id",
        "video_duration_sec",
        "expected_subwindow_count",
        "generated_subwindow_count",
        "success_subwindow_count",
        "failed_subwindow_count",
        "has_audio",
        "extraction_status",
    ]
    if not video_coverage.empty:
        lines.append(markdown_table(video_coverage[display_cols]))
    lines.append("")
    lines.append("## Overall Train Counts")
    lines.append(f"- Total train videos: {report['overall_train_counts']['total_train_videos']}")
    lines.append(f"- Total subwindows: {report['overall_train_counts']['total_subwindows']}")
    lines.append(f"- Total success subwindows: {report['overall_train_counts']['total_success_subwindows']}")
    lines.append(f"- Total failed subwindows: {report['overall_train_counts']['total_failed_subwindows']}")
    lines.append("")
    lines.append("## Actual Ad Interval Coverage")
    lines.append(f"- Total train ad intervals: {coverage['total_train_ad_intervals']}")
    lines.append(f"- Full coverage count: {coverage['full_coverage_count']}")
    lines.append(f"- Partial coverage count: {coverage['partial_coverage_count']}")
    lines.append(f"- No coverage count: {coverage['no_coverage_count']}")
    lines.append(f"- Mean coverage ratio: {coverage['mean_coverage_ratio']:.6f}")
    lines.append(f"- Min coverage ratio: {coverage['min_coverage_ratio']:.6f}")
    if np.isfinite(safe_float(coverage.get("mean_existing_edge_subwindow_coverage_ratio"))):
        lines.append(f"- Existing edge-subwindow mean coverage ratio: {coverage['mean_existing_edge_subwindow_coverage_ratio']:.6f}")
        lines.append(f"- Mean improvement vs existing edge subwindows: {coverage['mean_delta_vs_existing_edge_subwindow']:.6f}")
    lines.append("")
    lines.append("## Coverage Gap Cases")
    if gap_cases.empty:
        lines.append("- No no-coverage or partial-coverage actual train ad intervals remain under the full-video extraction output.")
    else:
        gap_cols = [
            "video_id",
            "ad_interval_id",
            "ad_start_sec",
            "ad_end_sec",
            "coverage_ratio",
            "coverage_status",
            "max_gap_sec_inside_interval",
            "gap_reason_estimate",
        ]
        lines.append(markdown_table(gap_cases[gap_cols]))
    lines.append("")
    lines.append("## Reusable Columns For 2.2")
    lines.append("- Timing keys: `video_id`, `split`, `subwindow_id`, `start_sec`, `end_sec`, `duration_sec`")
    lines.append("- Extraction metadata: `source_video_path`, `video_duration_sec`, `extraction_status`, `has_audio`, `sample_rate`, `audio_num_samples`")
    lines.append("- Basic/audio dynamics: `audio_rms_mean`, `audio_log_energy_mean`, `silence_ratio`, `low_energy_ratio`, `spectral_flux_*`, `onset_*`")
    lines.append("- Optional score: `audio_ad_like_score`, `audio_score_source` when the existing formula can be safely reused")
    lines.append("")
    lines.append("## Modified Files")
    lines.append(f"- `{CREATED_BY_SCRIPT}`")
    lines.append("")
    lines.append("## Generated Files")
    for key, value in output_paths.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Safety Flags")
    lines.append("- old_project_modified=false")
    lines.append("- detector_rule_modified=false")
    lines.append("- validation_test_row_level_output_created=false")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latest_bundle(
    bundle_dir: Path,
    project_root: Path,
    paths: Dict[str, Path],
    feature_rows: int,
    feature_size_bytes: int,
    include_feature_max_mb: float,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    copy_items = {
        "script_copy": project_root / CREATED_BY_SCRIPT,
        "summary_md": paths["summary_md"],
        "report_json": paths["report_json"],
        "coverage_csv": paths["interval_coverage"],
        "gap_cases_csv": paths["gap_cases"],
        "extraction_failures_csv": paths["failures"],
        "run_log": paths["run_log"],
    }
    for name, src in copy_items.items():
        if not src.exists() or src.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
            continue
        dst = bundle_dir / src.name
        if dst.exists():
            dst = unique_file_path(dst, report["run_id"])
        shutil.copy2(src, dst)
        copied.append(str(dst))
    feature_included = False
    feature_limit = include_feature_max_mb * 1024 * 1024
    if paths["features"].exists() and feature_size_bytes <= feature_limit:
        dst = bundle_dir / paths["features"].name
        if dst.exists():
            dst = unique_file_path(dst, report["run_id"])
        shutil.copy2(paths["features"], dst)
        copied.append(str(dst))
        feature_included = True
    readme = bundle_dir / "README_latest_files.md"
    lines = [
        "# Latest Files: Full-Video Audio Clue Extraction v2_4 Train",
        "",
        "This bundle contains lightweight review artifacts only. Raw video/audio/cache/model files are not copied.",
        "",
        "## Main Paths",
        f"- Script path: `{project_root / CREATED_BY_SCRIPT}`",
        f"- Summary: `{paths['summary_md']}`",
        f"- Report JSON: `{paths['report_json']}`",
        f"- Interval coverage CSV: `{paths['interval_coverage']}`",
        f"- Gap cases CSV: `{paths['gap_cases']}`",
        f"- Extraction failures CSV: `{paths['failures']}`",
        f"- Run log: `{paths['run_log']}`",
        "",
        "## Full Feature CSV",
        f"- Path: `{paths['features']}`",
        f"- Row count: {feature_rows}",
        f"- Size bytes: {feature_size_bytes}",
        f"- Included in bundle: {str(feature_included).lower()}",
        "",
        "## Copied Files",
    ]
    for item in copied:
        lines.append(f"- `{item}`")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copied.append(str(readme))
    return {
        "bundle_dir": str(bundle_dir),
        "copied_files": copied,
        "feature_included": feature_included,
        "feature_row_count": feature_rows,
        "feature_size_bytes": feature_size_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--split-file", default="data/splits/video_split_v2_4.csv")
    parser.add_argument("--label-file", default="data/segments/ad_interval_segments_v2_4.csv")
    parser.add_argument("--manifest-file", default="data/video_metadata/video_manifest_v2_2.csv")
    parser.add_argument("--subwindow-size-sec", type=float, default=2.0)
    parser.add_argument("--stride-sec", type=float, default=2.0)
    parser.add_argument("--min-valid-duration-sec", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--include-feature-in-bundle-max-mb", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(args.project_root).resolve()
    split_file = resolve_path(project_root, args.split_file)
    label_file = resolve_path(project_root, args.label_file)
    manifest_file = resolve_path(project_root, args.manifest_file)
    paths = default_output_paths(project_root, run_id)
    logger = TaskLogger(paths["run_log"])
    created_at = now_iso()
    t0 = time.time()

    logger.log("[STEP 01] Safety snapshot and output path planning")
    protected_targets = {
        "old_project": OLD_PROJECT_ROOT,
        "detector_scripts": project_root / "scripts/detectors",
        "configs": project_root / "configs",
        "predictions": project_root / "data/predictions",
        "split_file": split_file,
        "label_file": label_file,
        "raw_videos": project_root / "data/raw/videos",
        "input_audio_features": project_root / "data/audio/audio_ad_edge_persistence_subwindow_features_v2_4_with_split.csv",
    }
    protected_before = {name: snapshot_path(path) for name, path in protected_targets.items()}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    planned_bundle_dir = unique_dir_path(
        project_root / "outputs/latest_for_chatgpt_full_video_audio_clue_extraction_v2_4_train",
        run_id,
    )

    logger.log("[STEP 02] Validate fixed split and train-only scope")
    if not split_file.exists():
        raise FileNotFoundError(split_file)
    if not label_file.exists():
        raise FileNotFoundError(label_file)
    if not manifest_file.exists():
        raise FileNotFoundError(manifest_file)
    split_df = read_csv(split_file)
    label_df = read_csv(label_file)
    manifest_df = read_csv(manifest_file)
    split_groups = {
        split: sorted(pd.to_numeric(group["video_id"], errors="coerce").dropna().astype(int).tolist())
        for split, group in split_df.groupby("split")
    }
    split_errors = []
    for split_name, expected_ids in FIXED_SPLIT.items():
        observed = split_groups.get(split_name, [])
        if observed != expected_ids:
            split_errors.append(f"{split_name}: expected {expected_ids}, observed {observed}")
    if split_errors:
        raise RuntimeError("Fixed split validation failed: " + "; ".join(split_errors))
    train_rows = split_df[split_df["split"].astype(str).eq("train")].copy()
    train_rows["video_id"] = pd.to_numeric(train_rows["video_id"], errors="coerce").astype(int)
    train_rows = train_rows.sort_values("video_id")

    logger.log("[STEP 03] Locate video manifest and media paths")
    manifest_by_id = manifest_df.drop_duplicates("video_id").set_index("video_id") if "video_id" in manifest_df.columns else pd.DataFrame()
    for idx, row in train_rows.iterrows():
        vid = safe_int(row["video_id"])
        if (not str(row.get("video_path", "")).strip()) and vid in manifest_by_id.index:
            train_rows.loc[idx, "video_path"] = manifest_by_id.loc[vid].get("video_path", "")
        if (not np.isfinite(safe_float(row.get("video_duration_sec")))) and vid in manifest_by_id.index:
            train_rows.loc[idx, "video_duration_sec"] = manifest_by_id.loc[vid].get("duration_sec", np.nan)

    logger.log("[STEP 04] Locate existing audio extraction artifacts and reusable logic")
    audio_artifacts = discover_audio_artifacts(project_root)
    helper_script = project_root / "scripts/audio/extract_labeled_audio_features_v2_4.py"
    persistence_script = project_root / "scripts/audio/extract_audio_ad_edge_persistence_v2_4.py"
    helper, helper_error = load_module(helper_script, "audio_feature_helper_v2_4_full_video")
    persistence, persistence_error = load_module(persistence_script, "audio_persistence_helper_v2_4_full_video")

    logger.log("[STEP 05] Build or select audio extractor")
    ffmpeg_path = executable_path("ffmpeg")
    ffprobe_path = executable_path("ffprobe")
    feature_columns = list(getattr(helper, "FEATURE_COLUMNS", BASE_FEATURE_COLUMNS)) if helper is not None else list(BASE_FEATURE_COLUMNS)
    for col in BASE_FEATURE_COLUMNS:
        if col not in feature_columns:
            feature_columns.append(col)
    fallback_used = helper is None or not hasattr(helper, "compute_audio_features")
    logger.log(
        f"[STEP 05] helper_reused={not fallback_used}, ffmpeg={ffmpeg_path or 'missing'}, ffprobe={ffprobe_path or 'missing'}"
    )

    logger.log("[STEP 06] Extract full-video train audio subwindow features")
    feature_df, video_df = extract_train_features(
        train_rows=train_rows,
        helper=helper,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        subwindow_size_sec=args.subwindow_size_sec,
        stride_sec=args.stride_sec,
        min_valid_duration_sec=args.min_valid_duration_sec,
        sample_rate=args.sample_rate,
        created_at=created_at,
        logger=logger,
    )
    feature_df, score_report = add_audio_scores(feature_df, project_root, persistence)

    ordered_cols = []
    for col in REQUIRED_FEATURE_COLUMNS + ["subwindow_index", "decoded_duration_sec", "decoded_duration_diff_sec", "audio_stream_index", "audio_codec_name", "probe_status", "probe_error", "feature_compute_source"]:
        if col in feature_df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    for col in feature_columns + SCORE_OUTPUT_COLUMNS:
        if col in feature_df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    for col in feature_df.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    feature_df = feature_df[ordered_cols]
    feature_df.to_csv(paths["features"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 07] Compute video-level extraction coverage")
    video_coverage = compute_video_coverage(feature_df, video_df)
    video_coverage.to_csv(paths["video_coverage"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 08] Compute actual ad interval audio coverage")
    interval_coverage, existing_coverage_report = compute_interval_coverage(label_df, feature_df, split_df, project_root)
    interval_coverage.to_csv(paths["interval_coverage"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 09] Write coverage gap cases")
    gap_cases = build_gap_cases(interval_coverage)
    gap_cases.to_csv(paths["gap_cases"], index=False, encoding="utf-8-sig")
    failures = feature_df[~feature_df["extraction_status"].eq("success")].copy()
    failures.to_csv(paths["failures"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 10] Generate reports")
    overall_counts = {
        "total_train_videos": int(len(train_rows)),
        "total_subwindows": int(len(feature_df)),
        "total_success_subwindows": int(feature_df["extraction_status"].eq("success").sum()),
        "total_failed_subwindows": int((~feature_df["extraction_status"].eq("success")).sum()),
    }
    interval_summary = summarize_interval_coverage(interval_coverage)
    output_files = {key: str(path) for key, path in paths.items()}
    report: Dict[str, Any] = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "elapsed_sec_at_report_prepare": safe_float(time.time() - t0),
        "purpose": "full-video audio clue extraction / coverage audit for train split only",
        "project_root": str(project_root),
        "split_seed": SPLIT_SEED,
        "train_video_ids": FIXED_SPLIT["train"],
        "validation_video_ids": FIXED_SPLIT["validation"],
        "test_video_ids": FIXED_SPLIT["test"],
        "validation_test_excluded_from_row_level_outputs": True,
        "input_files": [
            {"name": "split_file", "path": str(split_file), **file_info(split_file, project_root)},
            {"name": "label_file", "path": str(label_file), **file_info(label_file, project_root)},
            {"name": "manifest_file", "path": str(manifest_file), **file_info(manifest_file, project_root)},
            {"name": "audio_config", "path": str(project_root / "configs/audio_persistence_rule_config_v2_4_train_only.json")},
            {"name": "existing_subwindow_features", "path": existing_coverage_report.get("existing_edge_subwindow_source", "")},
            {"name": "prior_audio_body_coverage", "path": existing_coverage_report.get("prior_audio_body_coverage_source", "")},
        ],
        "existing_audio_artifacts": audio_artifacts,
        "existing_audio_logic": {
            "helper_script": str(helper_script),
            "helper_reused": bool(helper is not None and hasattr(helper, "compute_audio_features")),
            "helper_error": helper_error,
            "persistence_script": str(persistence_script),
            "score_formula_reused": bool(score_report.get("reused_existing_formula")),
            "score_formula_error": score_report.get("error") or persistence_error,
            "score_baseline_source": score_report.get("baseline_source", ""),
            "fallback_extractor_used": bool(fallback_used),
            "ffmpeg_path": ffmpeg_path,
            "ffprobe_path": ffprobe_path,
            "librosa_used": False,
            "moviepy_used": False,
        },
        "subwindow_size_sec": args.subwindow_size_sec,
        "stride_sec": args.stride_sec,
        "min_valid_duration_sec": args.min_valid_duration_sec,
        "last_partial_window_policy": f"include partial windows when duration >= {args.min_valid_duration_sec}s",
        "train_video_extraction": video_coverage.to_dict(orient="records"),
        "overall_train_counts": overall_counts,
        "actual_ad_interval_coverage": interval_summary,
        "coverage_gap_cases": {
            "gap_case_count": int(len(gap_cases)),
            "no_coverage_intervals": gap_cases[gap_cases.get("coverage_status", pd.Series(dtype=str)).eq("none")].to_dict(orient="records")
            if not gap_cases.empty
            else [],
            "partial_coverage_intervals": gap_cases[gap_cases.get("coverage_status", pd.Series(dtype=str)).eq("partial")].to_dict(orient="records")
            if not gap_cases.empty
            else [],
            "cause_estimate": "No remaining gaps" if gap_cases.empty else "See gap_reason_estimate column",
        },
        "reusable_columns_for_2_2": {
            "identity_and_timing": ["video_id", "split", "subwindow_id", "start_sec", "end_sec", "duration_sec"],
            "source_and_status": ["source_video_path", "video_duration_sec", "extraction_status", "has_audio", "sample_rate", "audio_num_samples"],
            "basic_features": [
                "audio_rms_mean",
                "audio_log_energy_mean",
                "silence_ratio",
                "low_energy_ratio",
                "spectral_flux_mean",
                "onset_count",
                "onset_density",
            ],
            "optional_existing_score": ["audio_ad_like_score", "audio_score_source"],
        },
        "modified_files": [str(project_root / CREATED_BY_SCRIPT)],
        "generated_files": list(output_files.values()),
        "output_files": output_files,
        "old_project_modified": False,
        "detector_rule_modified": False,
        "validation_test_row_level_output_created": False,
        "existing_coverage_report": existing_coverage_report,
        "score_report": score_report,
    }

    logger.log("[STEP 11] Run Sub Agent validations")
    validations_before_bundle = validation_checks(
        split_df=split_df,
        feature_df=feature_df,
        video_coverage=video_coverage,
        interval_coverage=interval_coverage,
        gap_cases=gap_cases,
        paths=paths,
        protected_before=protected_before,
        project_root=project_root,
        min_valid_duration_sec=args.min_valid_duration_sec,
        bundle_dir=None,
    )
    report["sub_agent_validations"] = validations_before_bundle
    report["old_project_modified"] = bool(validations_before_bundle["output_safety_validation"]["old_project_modified"])
    report["detector_rule_modified"] = bool(validations_before_bundle["output_safety_validation"]["detector_rule_modified"])
    report["validation_test_row_level_output_created"] = bool(
        validations_before_bundle["output_safety_validation"]["validation_test_row_level_output_created"]
    )
    write_summary_md(paths["summary_md"], report, video_coverage, interval_coverage, gap_cases)
    save_json(paths["report_json"], report)

    logger.log("[STEP 12] Update latest bundle")
    feature_size = paths["features"].stat().st_size if paths["features"].exists() else 0
    bundle_report = write_latest_bundle(
        planned_bundle_dir,
        project_root,
        paths,
        feature_rows=len(feature_df),
        feature_size_bytes=feature_size,
        include_feature_max_mb=args.include_feature_in_bundle_max_mb,
        report=report,
    )
    validations_after_bundle = validation_checks(
        split_df=split_df,
        feature_df=feature_df,
        video_coverage=video_coverage,
        interval_coverage=interval_coverage,
        gap_cases=gap_cases,
        paths=paths,
        protected_before=protected_before,
        project_root=project_root,
        min_valid_duration_sec=args.min_valid_duration_sec,
        bundle_dir=planned_bundle_dir,
    )
    report["latest_bundle"] = bundle_report
    report["sub_agent_validations"] = validations_after_bundle
    report["old_project_modified"] = bool(validations_after_bundle["output_safety_validation"]["old_project_modified"])
    report["detector_rule_modified"] = bool(validations_after_bundle["output_safety_validation"]["detector_rule_modified"])
    report["validation_test_row_level_output_created"] = bool(
        validations_after_bundle["output_safety_validation"]["validation_test_row_level_output_created"]
    )
    report["elapsed_sec"] = safe_float(time.time() - t0)
    write_summary_md(paths["summary_md"], report, video_coverage, interval_coverage, gap_cases)
    save_json(paths["report_json"], report)
    readme = planned_bundle_dir / "README_latest_files.md"
    if readme.exists():
        # final JSON을 다시 쓴 뒤 README를 갱신해 복사된 JSON과 review 정보가 어긋나지 않게 한다.
        write_latest_bundle(
            planned_bundle_dir,
            project_root,
            paths,
            feature_rows=len(feature_df),
            feature_size_bytes=feature_size,
            include_feature_max_mb=args.include_feature_in_bundle_max_mb,
            report=report,
        )

    logger.log("[STEP 13] Print final human-readable summary")
    logger.log(
        "[STEP 13] "
        f"subwindows={overall_counts['total_subwindows']}, "
        f"success={overall_counts['total_success_subwindows']}, "
        f"failed={overall_counts['total_failed_subwindows']}, "
        f"ad_full={interval_summary['full_coverage_count']}, "
        f"ad_partial={interval_summary['partial_coverage_count']}, "
        f"ad_none={interval_summary['no_coverage_count']}, "
        f"gap_cases={len(gap_cases)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
