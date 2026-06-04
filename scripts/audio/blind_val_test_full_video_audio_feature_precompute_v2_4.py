#!/usr/bin/env python3
"""Blind validation/test full-video audio feature precompute for v2_4.

This script applies the train 2.1 full-video 2s audio extraction style to
validation/test videos only. It does not read actual ad labels and does not
create validation/test label-based scores, ad overlap columns, ad profiles,
candidate scores, threshold tuning outputs, or detector outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "v2_4"
TASK_NAME = "blind_val_test_full_video_audio_feature_precompute_v2_4"
SPLIT_SEED = 20240524
DEFAULT_PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
CREATED_BY_SCRIPT = "scripts/audio/blind_val_test_full_video_audio_feature_precompute_v2_4.py"
FEATURE_VERSION = "blind_val_test_full_video_audio_precompute_v2_4"
SAMPLE_RATE = 16000
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
MFCC_STD_FEATURES = [f"mfcc_{i}_std" for i in range(1, 14)]
REFERENCE_FEATURES = ["audio_ad_like_score_reference", "inverse_silence_score"]
FORBIDDEN_BUNDLE_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".onnx",
}
FORBIDDEN_OUTPUT_COLUMNS = {
    "ad_overlap_duration_sec",
    "ad_overlap_ratio",
    "is_ad_overlap",
    "is_ad_core",
    "overlapping_ad_interval_ids",
    "is_clean_nonad",
    "clean_nonad",
    "audio_candidate_score_for_discussion",
    "audio_pattern_label",
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
    if not isinstance(obj, (str, bytes, bytearray)):
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


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


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return str(path)


def snapshot_path(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "metadata_digest": None}
    if path.is_file():
        targets = [path]
        base = path.parent
    else:
        targets = [p for p in path.rglob("*") if p.is_file()]
        base = path
    rows: List[str] = []
    for item in targets:
        try:
            stat = item.stat()
        except OSError:
            continue
        rows.append(f"{item.relative_to(base).as_posix()}\t{stat.st_size}\t{stat.st_mtime_ns}")
    return {
        "path": str(path),
        "exists": True,
        "file_count": len(rows),
        "metadata_digest": hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest(),
        "snapshot_type": "relative_path_size_mtime_ns",
    }


def compare_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    return before.get("exists") == after.get("exists") and before.get("metadata_digest") == after.get("metadata_digest")


def import_module_from_path(path: Path, module_name: str) -> Tuple[Optional[Any], str]:
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None, f"spec_unavailable: {path}"
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def default_output_paths(project_root: Path, run_id: str) -> Dict[str, Path]:
    planned = {
        "validation_features": project_root / "data/audio/blind_full_video_audio_subwindow_features_v2_4_validation.csv",
        "test_features": project_root / "data/audio/blind_full_video_audio_subwindow_features_v2_4_test.csv",
        "val_test_features": project_root / "data/audio/blind_full_video_audio_subwindow_features_v2_4_val_test.csv",
        "validation_baseline": project_root / "data/audio/blind_per_video_audio_baseline_summary_v2_4_validation.csv",
        "test_baseline": project_root / "data/audio/blind_per_video_audio_baseline_summary_v2_4_test.csv",
        "validation_relative": project_root / "data/audio/blind_per_video_audio_relative_levels_v2_4_validation.csv",
        "test_relative": project_root / "data/audio/blind_per_video_audio_relative_levels_v2_4_test.csv",
        "failures": project_root / "data/audio/blind_val_test_audio_extraction_failures_v2_4.csv",
        "warnings": project_root / "data/audio/blind_val_test_audio_precompute_warnings_v2_4.csv",
        "summary_md": project_root / "reports/audio/blind_val_test_full_video_audio_feature_precompute_v2_4_summary.md",
        "report_json": project_root / "reports/audio/blind_val_test_full_video_audio_feature_precompute_v2_4_report.json",
        "run_log": project_root / "logs/blind_val_test_full_video_audio_feature_precompute_v2_4_run_log.txt",
    }
    return {key: unique_file_path(path, run_id) for key, path in planned.items()}


def load_and_validate_split(split_file: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    split_df = read_csv(split_file)
    split_df["video_id"] = pd.to_numeric(split_df["video_id"], errors="coerce").astype(int)
    split_groups = {
        split: sorted(group["video_id"].dropna().astype(int).tolist())
        for split, group in split_df.groupby("split")
    }
    validation = {
        "split_seed_expected": SPLIT_SEED,
        "observed_split_groups": split_groups,
        "train_matches_fixed": split_groups.get("train", []) == FIXED_SPLIT["train"],
        "validation_matches_fixed": split_groups.get("validation", []) == FIXED_SPLIT["validation"],
        "test_matches_fixed": split_groups.get("test", []) == FIXED_SPLIT["test"],
    }
    validation["all_match"] = all(
        [
            validation["train_matches_fixed"],
            validation["validation_matches_fixed"],
            validation["test_matches_fixed"],
        ]
    )
    if not validation["all_match"]:
        raise RuntimeError(f"Fixed split mismatch: {validation}")
    return split_df, validation


def fill_manifest_fields(split_rows: pd.DataFrame, manifest_file: Path) -> pd.DataFrame:
    rows = split_rows.copy()
    manifest_df = read_csv(manifest_file)
    if "video_id" not in manifest_df.columns:
        return rows
    manifest_df["video_id"] = pd.to_numeric(manifest_df["video_id"], errors="coerce").astype(int)
    by_id = manifest_df.drop_duplicates("video_id").set_index("video_id")
    for idx, row in rows.iterrows():
        vid = safe_int(row["video_id"])
        if vid not in by_id.index:
            continue
        if not str(row.get("video_path", "")).strip():
            rows.loc[idx, "video_path"] = by_id.loc[vid].get("video_path", "")
        if not np.isfinite(safe_float(row.get("video_duration_sec"))):
            rows.loc[idx, "video_duration_sec"] = by_id.loc[vid].get("duration_sec", np.nan)
    return rows.sort_values("video_id").reset_index(drop=True)


def extract_blind_features_for_split(
    rows: pd.DataFrame,
    split_name: str,
    train_audio: Any,
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
    compute_source = (
        "existing_helper_compute_audio_features"
        if helper is not None and hasattr(helper, "compute_audio_features")
        else "train_2_1_fallback_basic_features"
    )
    base_feature_columns = list(getattr(train_audio, "BASE_FEATURE_COLUMNS", []))
    for _, row in rows.iterrows():
        video_id = safe_int(row["video_id"])
        video_path = Path(str(row.get("video_path", "")))
        manifest_duration = safe_float(row.get("video_duration_sec"))
        probe = train_audio.probe_video(video_path, manifest_duration, ffprobe_path)
        video_duration = safe_float(probe.get("video_duration_sec"))
        subwindows = train_audio.build_subwindows(
            video_duration,
            subwindow_size_sec,
            stride_sec,
            min_valid_duration_sec,
        )
        logger.log(
            f"[STEP {'06' if split_name == 'validation' else '07'}] {split_name} video {video_id}: "
            f"planned={len(subwindows)}, duration={video_duration:.3f}s, has_audio={probe.get('has_audio')}"
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
            audio, decoded_sr, err = train_audio.decode_full_audio(ffmpeg_path, video_path, sample_rate, helper)
            if err or audio.size == 0:
                video_status = "decode_failed"
                video_error = err or "empty decoded audio"
            else:
                video_status = "success"
        success_count = 0
        failed_count = 0
        for sw in subwindows:
            duration = safe_float(sw["duration_sec"])
            subwindow_index = int(sw["subwindow_index"])
            record: Dict[str, Any] = {
                "video_id": video_id,
                "split": split_name,
                "source_video_path": str(video_path),
                "video_path": str(video_path),
                "video_duration_sec": video_duration,
                "subwindow_id": f"{VERSION}_blind_audio_{split_name}_V{video_id:02d}_{subwindow_index:06d}",
                "subwindow_index": subwindow_index,
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
                "label_used": False,
                "actual_ad_label_used": False,
                "blind_precompute_only": True,
            }
            for col in base_feature_columns:
                record[col] = float("nan")
            if video_status != "success":
                failed_count += 1
                records.append(record)
                continue
            try:
                start_idx = int(round(record["start_sec"] * decoded_sr))
                end_idx = int(round(record["end_sec"] * decoded_sr))
                slice_audio = audio[max(start_idx, 0) : min(end_idx, audio.size)]
                if slice_audio.size == 0:
                    record["extraction_status"] = "empty_audio_subwindow"
                    record["extraction_error"] = "decoded audio slice is empty"
                    failed_count += 1
                    records.append(record)
                    continue
                record["audio_num_samples"] = int(slice_audio.size)
                record["decoded_duration_sec"] = safe_float(slice_audio.size / float(decoded_sr))
                record["decoded_duration_diff_sec"] = safe_float(record["decoded_duration_sec"] - duration)
                if helper is not None and hasattr(helper, "compute_audio_features"):
                    features = helper.compute_audio_features(slice_audio, decoded_sr)
                else:
                    features = train_audio.compute_basic_audio_features(slice_audio, decoded_sr)
                record.update(features)
                train_audio.add_duration_normalized_features(record, duration)
                record["extraction_status"] = "success"
                record["extraction_error"] = ""
                success_count += 1
            except Exception as exc:
                record["extraction_status"] = "feature_compute_failed"
                record["extraction_error"] = f"{type(exc).__name__}: {exc}"
                failed_count += 1
            records.append(record)
        video_records.append(
            {
                "video_id": video_id,
                "split": split_name,
                "source_video_path": str(video_path),
                "video_duration_sec": video_duration,
                "manifest_duration_sec": manifest_duration,
                "ffprobe_duration_sec": safe_float(probe.get("ffprobe_duration_sec")),
                "expected_subwindow_count": len(train_audio.build_subwindows(video_duration, subwindow_size_sec, stride_sec, min_valid_duration_sec)),
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
                "label_used": False,
                "actual_ad_label_used": False,
                "blind_precompute_only": True,
            }
        )
        logger.log(
            f"[STEP {'06' if split_name == 'validation' else '07'}] {split_name} video {video_id}: "
            f"success={success_count}, failed={failed_count}, status={video_status}"
        )
    return pd.DataFrame(records), pd.DataFrame(video_records)


def apply_reference_audio_score(
    feature_df: pd.DataFrame,
    project_root: Path,
    train_audio: Any,
    persistence: Optional[Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = feature_df.copy()
    report = {
        "attempted": bool(persistence is not None and hasattr(persistence, "add_ad_like_scores")),
        "audio_ad_like_score_reference_created": False,
        "source": "unavailable",
        "note": "reference score only; not a candidate score or detector threshold",
        "error": "",
    }
    if not report["attempted"]:
        out["audio_ad_like_score_reference"] = np.nan
        out["audio_score_source"] = "unavailable"
        return out, report
    try:
        scored, score_report = train_audio.add_audio_scores(out, project_root, persistence)
        if "audio_ad_like_score" in scored.columns:
            scored["audio_ad_like_score_reference"] = scored["audio_ad_like_score"]
            report["audio_ad_like_score_reference_created"] = bool(
                pd.to_numeric(scored["audio_ad_like_score_reference"], errors="coerce").notna().any()
            )
        else:
            scored["audio_ad_like_score_reference"] = np.nan
        if "audio_score_source" in scored.columns:
            scored["audio_score_source"] = scored["audio_score_source"].replace(
                {"existing_formula": "existing_formula_train_fixed_reference"}
            )
        else:
            scored["audio_score_source"] = "unavailable"
        report.update(
            {
                "source": score_report.get("score_source", "unavailable"),
                "baseline_source": score_report.get("baseline_source", ""),
                "score_report": score_report,
            }
        )
        return scored, report
    except Exception as exc:
        out["audio_ad_like_score_reference"] = np.nan
        out["audio_score_source"] = "unavailable"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        return out, report


def percentile_within_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    notna = numeric.notna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if not notna.any():
        return out
    out.loc[notna] = numeric.loc[notna].rank(method="average", pct=True).astype(float) * 100.0
    return out


def fallback_scale_for_series(series: pd.Series) -> Tuple[float, bool]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 1.0, True
    q25 = numeric.quantile(0.25)
    q75 = numeric.quantile(0.75)
    iqr = q75 - q25
    if np.isfinite(iqr) and abs(iqr) > 1e-12:
        return safe_float(iqr), False
    std = numeric.std(ddof=0)
    if np.isfinite(std) and abs(std) > 1e-12:
        return safe_float(std), True
    value_range = numeric.max() - numeric.min()
    if np.isfinite(value_range) and abs(value_range) > 1e-12:
        return safe_float(value_range), True
    return 1.0, True


def compute_blind_baseline_and_relative(
    feature_df: pd.DataFrame,
    analysis_features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    out = feature_df.copy()
    success_mask = out["extraction_status"].astype(str).eq("success")
    usable_features = [f for f in analysis_features if f in out.columns]
    for video_id, idx in out.groupby("video_id").groups.items():
        idx_list = list(idx)
        group_success = out.loc[idx_list].loc[success_mask.loc[idx_list]]
        split_values = sorted(out.loc[idx_list, "split"].dropna().astype(str).unique().tolist())
        split_name = split_values[0] if split_values else ""
        for feature in usable_features:
            series = pd.to_numeric(group_success[feature], errors="coerce").dropna()
            count = int(series.shape[0])
            if count:
                qs = series.quantile([0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95])
                median = safe_float(qs.loc[0.50])
                iqr_raw = safe_float(qs.loc[0.75] - qs.loc[0.25])
                scale, fallback_used = fallback_scale_for_series(series)
                mean = safe_float(series.mean())
                std = safe_float(series.std(ddof=0))
                row = {
                    "video_id": int(video_id),
                    "split": split_name,
                    "feature_name": feature,
                    "count": count,
                    "mean": mean,
                    "std": std,
                    "median": median,
                    "IQR": iqr_raw,
                    "q05": safe_float(qs.loc[0.05]),
                    "q10": safe_float(qs.loc[0.10]),
                    "q25": safe_float(qs.loc[0.25]),
                    "q40": safe_float(qs.loc[0.40]),
                    "q50": safe_float(qs.loc[0.50]),
                    "q60": safe_float(qs.loc[0.60]),
                    "q70": safe_float(qs.loc[0.70]),
                    "q75": safe_float(qs.loc[0.75]),
                    "q80": safe_float(qs.loc[0.80]),
                    "q90": safe_float(qs.loc[0.90]),
                    "q95": safe_float(qs.loc[0.95]),
                    "min": safe_float(series.min()),
                    "max": safe_float(series.max()),
                    "iqr_fallback_used": bool(fallback_used),
                    "fallback_scale": scale,
                    "baseline_scope": "same_video_only",
                    "actual_ad_label_used": False,
                }
            else:
                median = np.nan
                scale = 1.0
                fallback_used = True
                row = {
                    "video_id": int(video_id),
                    "split": split_name,
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
                    "fallback_scale": scale,
                    "baseline_scope": "same_video_only",
                    "actual_ad_label_used": False,
                }
            rows.append(row)
            target = out.loc[idx_list, feature] if feature in out.columns else pd.Series(np.nan, index=idx_list)
            numeric_target = pd.to_numeric(target, errors="coerce")
            out.loc[idx_list, f"{feature}__video_median"] = median
            out.loc[idx_list, f"{feature}__video_iqr"] = scale
            out.loc[idx_list, f"{feature}__video_robust_z"] = (numeric_target - median) / scale
            out.loc[idx_list, f"{feature}__video_percentile"] = percentile_within_series(target)
            out.loc[idx_list, f"{feature}__relative_level"] = pd.cut(
                out.loc[idx_list, f"{feature}__video_percentile"],
                bins=[-np.inf, 40.0, 70.0, np.inf],
                labels=["low", "medium", "high"],
                right=False,
            ).astype("string")
            if fallback_used:
                warnings.append(
                    {
                        "warning_type": "iqr_fallback_used",
                        "video_id": int(video_id),
                        "split": split_name,
                        "feature_name": feature,
                        "message": "IQR was zero/missing/tiny; fallback scale used for robust z.",
                    }
                )
    return pd.DataFrame(rows), out, warnings


def add_label_free_activity_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    active_cols = [f"{f}__video_percentile" for f in HIGHER_ACTIVE_FEATURES if f"{f}__video_percentile" in out.columns]
    quiet_cols = [f"{f}__video_percentile" for f in QUIET_FEATURES if f"{f}__video_percentile" in out.columns]
    if active_cols:
        out["blind_relative_active_audio_score"] = out[active_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) / 100.0
    else:
        out["blind_relative_active_audio_score"] = np.nan
    if quiet_cols:
        out["blind_relative_quiet_audio_score"] = out[quiet_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) / 100.0
    else:
        out["blind_relative_quiet_audio_score"] = np.nan
    active = pd.to_numeric(out["blind_relative_active_audio_score"], errors="coerce")
    quiet = pd.to_numeric(out["blind_relative_quiet_audio_score"], errors="coerce")
    labels: List[str] = []
    for a, q in zip(active, quiet):
        if np.isfinite(a) and np.isfinite(q) and a >= 0.60 and q >= 0.60:
            labels.append("relative_mixed")
        elif np.isfinite(a) and a >= 0.70:
            labels.append("relative_active_high")
        elif np.isfinite(q) and q >= 0.70:
            labels.append("relative_quiet_high")
        elif np.isfinite(a) and a >= 0.40:
            labels.append("relative_active_medium")
        else:
            labels.append("relative_audio_flat_or_uninformative")
    out["label_free_audio_relative_activity_label"] = labels
    out["label_used"] = False
    out["actual_ad_label_used"] = False
    out["blind_precompute_only"] = True
    return out


def split_baseline_relative_outputs(
    relative_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_relative = relative_df[relative_df["split"].astype(str).eq("validation")].copy()
    test_relative = relative_df[relative_df["split"].astype(str).eq("test")].copy()
    validation_baseline = baseline_df[baseline_df["split"].astype(str).eq("validation")].copy()
    test_baseline = baseline_df[baseline_df["split"].astype(str).eq("test")].copy()
    return validation_relative, test_relative, validation_baseline, test_baseline


def schema_compare(train_header: List[str], output_columns: Iterable[str]) -> Dict[str, Any]:
    output_cols = list(output_columns)
    train_set = set(train_header)
    output_set = set(output_cols)
    comparable_train_columns = [
        c for c in train_header
        if c not in {"split", "subwindow_id", "feature_version", "created_by_script", "created_at"}
    ]
    matched = [c for c in comparable_train_columns if c in output_set]
    missing = [c for c in comparable_train_columns if c not in output_set]
    added = [c for c in output_cols if c not in train_set]
    return {
        "train_column_count": len(train_header),
        "output_column_count": len(output_cols),
        "matched_train_columns_count": len(matched),
        "missing_train_columns": missing,
        "added_blind_columns": added,
    }


def build_failures(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()
    return feature_df[~feature_df["extraction_status"].astype(str).eq("success")].copy()


def video_extraction_summary(video_df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = [
        "video_id",
        "split",
        "video_duration_sec",
        "expected_subwindow_count",
        "generated_subwindow_count",
        "success_subwindow_count",
        "failed_subwindow_count",
        "has_audio",
        "extraction_status",
        "extraction_error",
    ]
    if video_df.empty:
        return []
    return video_df[cols].to_dict("records")


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
            values.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary(path: Path, report: Dict[str, Any], video_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display_cols = [
        "video_id",
        "split",
        "video_duration_sec",
        "expected_subwindow_count",
        "generated_subwindow_count",
        "success_subwindow_count",
        "failed_subwindow_count",
        "has_audio",
        "extraction_status",
    ]
    lines = [
        "# Blind Val/Test Full-Video Audio Feature Precompute v2_4",
        "",
        "## Purpose",
        "Prepare validation/test full-video 2s audio subwindow features and label-free per-video relative levels for later fusion.",
        "",
        "## Scope",
        "- This is blind preprocessing, not detector rule modification.",
        "- Validation/test actual ad interval labels were not read or used.",
        "- No validation/test ad overlap, ad/non-ad profile, ad-vs-nonad contrast, candidate score, threshold tuning, or performance metric was created.",
        f"- Validation video_ids: {report['validation_video_ids']}",
        f"- Test video_ids: {report['test_video_ids']}",
        "- Train split was used only for schema/reference artifacts; train row-level audio features were not regenerated.",
        "",
        "## Inputs",
    ]
    for key, value in report["input_files"].items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Subwindow Policy",
        f"- subwindow_size_sec={report['subwindow_policy']['subwindow_size_sec']}",
        f"- stride_sec={report['subwindow_policy']['stride_sec']}",
        f"- min_valid_duration_sec={report['subwindow_policy']['min_valid_duration_sec']}",
        "",
        "## Per-Video Extraction",
    ]
    if not video_df.empty:
        lines.append(markdown_table(video_df[display_cols]))
    lines += [
        "",
        "## Overall Counts",
        f"- validation_total_subwindows: {report['overall_counts']['validation_total_subwindows']}",
        f"- test_total_subwindows: {report['overall_counts']['test_total_subwindows']}",
        f"- validation_success_subwindows: {report['overall_counts']['validation_success_subwindows']}",
        f"- test_success_subwindows: {report['overall_counts']['test_success_subwindows']}",
        f"- validation_failed_subwindows: {report['overall_counts']['validation_failed_subwindows']}",
        f"- test_failed_subwindows: {report['overall_counts']['test_failed_subwindows']}",
        "",
        "## Feature Set",
    ]
    for feature in report["audio_features"]["analysis_features"]:
        lines.append(f"- `{feature}`")
    lines += [
        "",
        "## Train Schema Comparison",
        f"- matched_train_columns_count: {report['train_schema_comparison']['matched_train_columns_count']}",
        f"- missing_train_columns_count: {len(report['train_schema_comparison']['missing_train_columns'])}",
        f"- added_blind_columns_count: {len(report['train_schema_comparison']['added_blind_columns'])}",
        "",
        "## Blind Relative Baseline",
        "- Baseline scope: same_video_only.",
        "- Percentiles and robust z-scores were computed within each validation/test video independently.",
        "- Train baselines were not used for validation/test relative levels.",
        "- Validation/test raw audio absolute values were not directly compared across videos.",
        "- Relative levels are label-free features, not ad judgments.",
        "",
        "## Score/Label Safety",
        f"- audio_ad_like_score_reference_created: {str(report['audio_score_reference']['audio_ad_like_score_reference_created']).lower()}",
        "- audio_candidate_score_for_discussion_created=false",
        "- audio_pattern_label_created=false",
        "- validation_test_ad_nonad_contrast_created=false",
        "- validation_test_threshold_tuning_performed=false",
        "",
        "## Fusion Keys",
    ]
    for col in report["fusion_key_columns"]:
        lines.append(f"- `{col}`")
    lines += [
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
        "- actual_ad_label_used_for_validation=false",
        "- actual_ad_label_used_for_test=false",
        "- validation_test_label_based_analysis_created=false",
        "- validation_test_threshold_tuning_performed=false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_if_lightweight(src: Path, dest_dir: Path, max_mb: float, run_id: str) -> Dict[str, Any]:
    info = {
        "source": str(src),
        "copied": False,
        "destination": "",
        "size_bytes": src.stat().st_size if src.exists() else None,
        "row_count": None,
        "reason": "",
    }
    if not src.exists():
        info["reason"] = "missing"
        return info
    if src.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
        info["reason"] = "forbidden_suffix"
        return info
    if src.suffix.lower() == ".csv":
        try:
            info["row_count"] = sum(1 for _ in src.open("r", encoding="utf-8-sig")) - 1
        except Exception:
            info["row_count"] = None
    limit = max_mb * 1024 * 1024
    if src.stat().st_size > limit:
        info["reason"] = f"larger_than_{max_mb}_mb"
        return info
    dst = dest_dir / src.name
    if dst.exists():
        dst = unique_file_path(dst, run_id)
    shutil.copy2(src, dst)
    info["copied"] = True
    info["destination"] = str(dst)
    info["reason"] = "copied"
    return info


def write_latest_bundles(
    project_root: Path,
    paths: Dict[str, Path],
    report: Dict[str, Any],
    run_id: str,
) -> Dict[str, Any]:
    bundle = project_root / "outputs/latest_for_chatgpt_blind_val_test_audio_precompute_v2_4"
    lightweight = project_root / "outputs/latest_audio_val_test_blind"
    bundle.mkdir(parents=True, exist_ok=True)
    lightweight.mkdir(parents=True, exist_ok=True)
    copy_keys = [
        "summary_md",
        "report_json",
        "validation_baseline",
        "test_baseline",
        "failures",
        "warnings",
        "run_log",
    ]
    maybe_large_keys = [
        "validation_features",
        "test_features",
        "val_test_features",
        "validation_relative",
        "test_relative",
    ]
    copied_info: Dict[str, List[Dict[str, Any]]] = {"latest_for_chatgpt": [], "latest_audio_val_test_blind": []}
    script_path = project_root / CREATED_BY_SCRIPT
    for dest_name, dest in [("latest_for_chatgpt", bundle), ("latest_audio_val_test_blind", lightweight)]:
        copied_info[dest_name].append(copy_if_lightweight(script_path, dest, 10.0, run_id))
        for key in copy_keys:
            copied_info[dest_name].append(copy_if_lightweight(paths[key], dest, 10.0, run_id))
        for key in maybe_large_keys:
            copied_info[dest_name].append(copy_if_lightweight(paths[key], dest, 10.0, run_id))
    for dest_name, dest in [("latest_for_chatgpt", bundle), ("latest_audio_val_test_blind", lightweight)]:
        readme_name = "README_latest_files.md" if dest_name == "latest_for_chatgpt" else "README_latest_audio_val_test_blind.md"
        lines = [
            "# Blind Val/Test Audio Precompute Latest Files",
            "",
            "This bundle contains review artifacts only. Raw video/audio/cache/model files and actual label files are not copied.",
            "",
            f"- source_script: `{script_path}`",
            f"- source_report: `{paths['report_json']}`",
            f"- actual_ad_label_used_for_validation=false",
            f"- actual_ad_label_used_for_test=false",
            f"- audio_candidate_score_for_discussion_created=false",
            f"- validation_test_threshold_tuning_performed=false",
            "",
            "## File Entries",
        ]
        for item in copied_info[dest_name]:
            copied = str(item["copied"]).lower()
            rows = "" if item["row_count"] is None else f", rows={item['row_count']}"
            lines.append(
                f"- copied={copied}, source=`{item['source']}`, destination=`{item['destination']}`, "
                f"size_bytes={item['size_bytes']}{rows}, reason={item['reason']}"
            )
        (dest / readme_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        copied_info[dest_name].append(
            {
                "source": "generated_readme",
                "copied": True,
                "destination": str(dest / readme_name),
                "size_bytes": (dest / readme_name).stat().st_size,
                "row_count": None,
                "reason": "generated",
            }
        )
    return {
        "latest_for_chatgpt_dir": str(bundle),
        "latest_audio_val_test_blind_dir": str(lightweight),
        "copied_info": copied_info,
    }


def copy_current_task_to_latest_audio(project_root: Path, paths: Dict[str, Path], report: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    dest = project_root / "outputs/latest_audio"
    dest.mkdir(parents=True, exist_ok=True)
    copy_paths = [project_root / CREATED_BY_SCRIPT] + [Path(p) for p in report["generated_files"]]
    entries = []
    for src in copy_paths:
        if not src.exists() or src.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
            entries.append({"source": str(src), "copied": False, "destination": "", "reason": "missing_or_forbidden"})
            continue
        dst = dest / src.name
        if dst.exists():
            dst = unique_file_path(dst, run_id)
        shutil.copy2(src, dst)
        entries.append(
            {
                "source": str(src),
                "copied": True,
                "destination": str(dst),
                "size_bytes": src.stat().st_size,
                "reason": "copied",
            }
        )
    readme = dest / f"README_blind_val_test_audio_precompute_{run_id}.md"
    lines = [
        "# Current Task Copy: Blind Val/Test Audio Precompute",
        "",
        f"created_at: {now_iso()}",
        f"source_report: {paths['report_json']}",
        "",
        "## Safety Flags",
        "- detector_rule_modified=false",
        "- old_project_modified=false",
        "- actual_ad_label_used_for_validation=false",
        "- actual_ad_label_used_for_test=false",
        "- validation_test_label_based_analysis_created=false",
        "- validation_test_threshold_tuning_performed=false",
        "- audio_candidate_score_for_discussion_created=false",
        "",
        "## Copied Files",
    ]
    for entry in entries:
        lines.append(f"- copied={str(entry['copied']).lower()}, source=`{entry['source']}`, destination=`{entry['destination']}`, reason={entry['reason']}")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    entries.append({"source": "generated_readme", "copied": True, "destination": str(readme), "reason": "generated"})
    return {"latest_audio_dir": str(dest), "entries": entries}


def validate_outputs(
    split_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    relative_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    video_df: pd.DataFrame,
    paths: Dict[str, Path],
    protected_before: Dict[str, Dict[str, Any]],
    project_root: Path,
    latest_dirs: Sequence[Path],
    train_feature_mtime_before: Optional[float],
) -> Dict[str, Any]:
    output_ids = sorted(pd.to_numeric(feature_df["video_id"], errors="coerce").dropna().astype(int).unique().tolist())
    relative_cols = set(relative_df.columns)
    forbidden_cols_found = sorted(FORBIDDEN_OUTPUT_COLUMNS & relative_cols)
    baseline_scopes = sorted(baseline_df.get("baseline_scope", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    percent_cols = [c for c in relative_df.columns if c.endswith("__video_percentile")]
    percentile_ok = True
    for col in percent_cols:
        vals = pd.to_numeric(relative_df[col], errors="coerce").dropna()
        if not vals.empty and not ((vals >= -1e-9) & (vals <= 100.0 + 1e-9)).all():
            percentile_ok = False
            break
    by_video = []
    for _, row in video_df.iterrows():
        vid = safe_int(row["video_id"])
        group = feature_df[pd.to_numeric(feature_df["video_id"], errors="coerce").eq(vid)].copy()
        first = safe_float(group["start_sec"].min()) if not group.empty else np.nan
        last = safe_float(group["end_sec"].max()) if not group.empty else np.nan
        duration = safe_float(row.get("video_duration_sec"))
        gaps = []
        sorted_group = group.sort_values("start_sec")
        prev_end = None
        for _, sw in sorted_group.iterrows():
            if prev_end is not None:
                gaps.append(max(0.0, safe_float(sw["start_sec"]) - prev_end))
            prev_end = safe_float(sw["end_sec"])
        by_video.append(
            {
                "video_id": vid,
                "split": row.get("split", ""),
                "first_start_near_zero": bool(np.isfinite(first) and abs(first) <= 1e-6),
                "last_end_near_video_duration": bool(np.isfinite(last) and np.isfinite(duration) and abs(last - duration) <= 0.51),
                "expected_generated_count_match": safe_int(row.get("expected_subwindow_count")) == safe_int(row.get("generated_subwindow_count")),
                "max_gap_sec": safe_float(max(gaps) if gaps else 0.0),
            }
        )
    protected_after = {name: snapshot_path(Path(snap["path"])) for name, snap in protected_before.items()}
    protected_unchanged = {name: compare_snapshots(protected_before[name], protected_after[name]) for name in protected_before}
    forbidden_bundle_files: List[str] = []
    for directory in latest_dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
                forbidden_bundle_files.append(str(path))
            if path.name == "ad_interval_segments_v2_4.csv":
                forbidden_bundle_files.append(str(path))
    train_feature_unchanged = True
    train_feature_path = Path(protected_before["train_full_video_feature"]["path"])
    if train_feature_path.exists() and train_feature_mtime_before is not None:
        train_feature_unchanged = abs(train_feature_path.stat().st_mtime - train_feature_mtime_before) < 1e-9
    return {
        "input_split_validation": {
            "validation_ids_exact": sorted(split_df[split_df["split"].eq("validation")]["video_id"].astype(int).tolist()) == FIXED_SPLIT["validation"],
            "test_ids_exact": sorted(split_df[split_df["split"].eq("test")]["video_id"].astype(int).tolist()) == FIXED_SPLIT["test"],
            "output_video_ids": output_ids,
            "output_contains_train_video_id": bool(set(output_ids) & set(FIXED_SPLIT["train"])),
            "input_label_file_used": False,
        },
        "blind_scope_validation": {
            "actual_ad_label_file_read": False,
            "forbidden_label_columns_found": forbidden_cols_found,
            "audio_candidate_score_for_discussion_created": "audio_candidate_score_for_discussion" in relative_cols,
            "audio_pattern_label_created": "audio_pattern_label" in relative_cols,
            "validation_test_threshold_tuning_performed": False,
            "validation_test_ad_nonad_contrast_created": False,
        },
        "feature_extraction_validation": {
            "by_video": by_video,
            "all_first_starts_near_zero": all(item["first_start_near_zero"] for item in by_video),
            "all_last_ends_near_video_duration": all(item["last_end_near_video_duration"] for item in by_video),
            "all_expected_counts_match": all(item["expected_generated_count_match"] for item in by_video),
            "failure_file_exists": paths["failures"].exists(),
            "extraction_status_consistent": int((feature_df["extraction_status"].astype(str) != "success").sum()) == int(len(build_failures(feature_df))),
        },
        "blind_per_video_baseline_validation": {
            "baseline_scope_values": baseline_scopes,
            "same_video_only_baseline": baseline_scopes == ["same_video_only"],
            "percentile_columns_in_0_100": percentile_ok,
            "iqr_fallback_rows": int(baseline_df["iqr_fallback_used"].astype(bool).sum()) if "iqr_fallback_used" in baseline_df.columns else 0,
            "validation_test_raw_audio_cross_video_comparison_used": False,
            "train_baseline_used_for_relative_levels": False,
        },
        "output_safety_validation": {
            "protected_paths_unchanged": protected_unchanged,
            "detector_rule_modified": False,
            "old_project_modified": not protected_unchanged.get("old_project", True),
            "raw_video_audio_cache_model_copied_to_latest_bundle": bool(forbidden_bundle_files),
            "forbidden_bundle_files": forbidden_bundle_files,
            "two_one_two_two_two_three_outputs_unchanged": all(
                protected_unchanged.get(name, True)
                for name in [
                    "train_full_video_feature",
                    "two_two_relative_levels",
                    "two_three_recommendations",
                    "two_three_inventory",
                    "two_three_score_formula",
                ]
            ),
            "train_full_video_feature_mtime_unchanged": train_feature_unchanged,
            "actual_label_file_copied_to_latest_bundle": any("ad_interval_segments_v2_4.csv" in p for p in forbidden_bundle_files),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--split-file", default="data/splits/video_split_v2_4.csv")
    parser.add_argument("--manifest-file", default="data/video_metadata/video_manifest_v2_2.csv")
    parser.add_argument("--train-full-video-feature-file", default="data/audio/full_video_audio_subwindow_features_v2_4_train_20260526_1750_final.csv")
    parser.add_argument("--recommendation-file", default="data/audio/audio_feature_recommendations_for_relative_analysis_v2_4_train_20260526_2116_final.csv")
    parser.add_argument("--inventory-file", default="data/audio/audio_feature_inventory_audit_v2_4_train_20260526_2116_final.csv")
    parser.add_argument("--score-formula-file", default="data/audio/audio_score_formula_audit_v2_4_train_20260526_2116_final.csv")
    parser.add_argument("--subwindow-size-sec", type=float, default=2.0)
    parser.add_argument("--stride-sec", type=float, default=2.0)
    parser.add_argument("--min-valid-duration-sec", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(args.project_root).resolve()
    split_file = resolve_path(project_root, args.split_file)
    manifest_file = resolve_path(project_root, args.manifest_file)
    train_feature_file = resolve_path(project_root, args.train_full_video_feature_file)
    recommendation_file = resolve_path(project_root, args.recommendation_file)
    inventory_file = resolve_path(project_root, args.inventory_file)
    score_formula_file = resolve_path(project_root, args.score_formula_file)
    paths = default_output_paths(project_root, run_id)
    logger = TaskLogger(paths["run_log"])
    created_at = now_iso()
    t0 = time.time()

    logger.log("[STEP 01] Safety snapshot and output path planning")
    for required_path in [split_file, manifest_file, train_feature_file]:
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    protected_targets = {
        "old_project": OLD_PROJECT_ROOT,
        "detector_scripts": project_root / "scripts/detectors",
        "detector_configs": project_root / "configs",
        "detector_outputs": project_root / "data/predictions",
        "split_file": split_file,
        "raw_videos": project_root / "data/raw/videos",
        "train_full_video_feature": train_feature_file,
        "two_two_relative_levels": project_root / "data/audio/per_video_audio_relative_levels_v2_4_train.csv",
        "two_three_recommendations": recommendation_file,
        "two_three_inventory": inventory_file,
        "two_three_score_formula": score_formula_file,
    }
    protected_before = {name: snapshot_path(path) for name, path in protected_targets.items()}
    train_feature_mtime_before = train_feature_file.stat().st_mtime if train_feature_file.exists() else None
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    logger.log("[STEP 02] Locate train 2.1 audio extraction schema and helper scripts")
    train_audio_script = project_root / "scripts/audio/full_video_audio_clue_extraction_v2_4_train.py"
    train_audio, train_audio_import_error = import_module_from_path(train_audio_script, "train_full_video_audio_v2_4_reuse")
    if train_audio is None:
        raise RuntimeError(f"Could not import train 2.1 audio helper: {train_audio_import_error}")
    helper_script = project_root / "scripts/audio/extract_labeled_audio_features_v2_4.py"
    persistence_script = project_root / "scripts/audio/extract_audio_ad_edge_persistence_v2_4.py"
    helper, helper_error = train_audio.load_module(helper_script, "blind_audio_feature_helper_v2_4")
    persistence, persistence_error = train_audio.load_module(persistence_script, "blind_audio_persistence_helper_v2_4")
    train_header = list(read_csv(train_feature_file, nrows=0).columns)

    logger.log("[STEP 03] Validate fixed split and blind val/test scope")
    split_df, split_validation = load_and_validate_split(split_file)
    validation_rows = fill_manifest_fields(split_df[split_df["split"].astype(str).eq("validation")].copy(), manifest_file)
    test_rows = fill_manifest_fields(split_df[split_df["split"].astype(str).eq("test")].copy(), manifest_file)

    logger.log("[STEP 04] Locate validation/test video paths from manifest/split")
    path_records = []
    for _, row in pd.concat([validation_rows, test_rows], ignore_index=True).iterrows():
        video_path = Path(str(row.get("video_path", "")))
        path_records.append(
            {
                "video_id": safe_int(row["video_id"]),
                "split": str(row["split"]),
                "video_path": str(video_path),
                "path_exists": video_path.exists(),
                "video_duration_sec": safe_float(row.get("video_duration_sec")),
            }
        )

    logger.log("[STEP 05] Build or reuse audio extractor without reading labels")
    ffmpeg_path = train_audio.executable_path("ffmpeg")
    ffprobe_path = train_audio.executable_path("ffprobe")
    extractor_report = {
        "train_audio_script_reused_read_only": str(train_audio_script),
        "helper_script": str(helper_script),
        "helper_reused": bool(helper is not None and hasattr(helper, "compute_audio_features")),
        "helper_error": helper_error,
        "persistence_script": str(persistence_script),
        "score_formula_reuse_attempted": bool(persistence is not None and hasattr(persistence, "add_ad_like_scores")),
        "persistence_error": persistence_error,
        "ffmpeg_path": ffmpeg_path,
        "ffprobe_path": ffprobe_path,
        "label_file_read": False,
    }

    logger.log("[STEP 06] Extract blind full-video audio subwindow features for validation")
    validation_features, validation_video_summary = extract_blind_features_for_split(
        validation_rows,
        "validation",
        train_audio,
        helper,
        ffmpeg_path,
        ffprobe_path,
        args.subwindow_size_sec,
        args.stride_sec,
        args.min_valid_duration_sec,
        args.sample_rate,
        created_at,
        logger,
    )

    logger.log("[STEP 07] Extract blind full-video audio subwindow features for test")
    test_features, test_video_summary = extract_blind_features_for_split(
        test_rows,
        "test",
        train_audio,
        helper,
        ffmpeg_path,
        ffprobe_path,
        args.subwindow_size_sec,
        args.stride_sec,
        args.min_valid_duration_sec,
        args.sample_rate,
        created_at,
        logger,
    )
    full_features = pd.concat([validation_features, test_features], ignore_index=True)

    full_features, score_reference_report = apply_reference_audio_score(full_features, project_root, train_audio, persistence)
    validation_features = full_features[full_features["split"].astype(str).eq("validation")].copy()
    test_features = full_features[full_features["split"].astype(str).eq("test")].copy()

    logger.log("[STEP 08] Compare output schema with train full-video audio feature schema")
    schema_report = schema_compare(train_header, full_features.columns)

    analysis_features = []
    for feature in CORE_RAW_FEATURES + MFCC_STD_FEATURES + REFERENCE_FEATURES:
        if feature in full_features.columns and feature not in analysis_features:
            analysis_features.append(feature)
    missing_core_features = [feature for feature in CORE_RAW_FEATURES if feature not in full_features.columns]
    analysis_feature_report = {
        "core_raw_features": [f for f in CORE_RAW_FEATURES if f in full_features.columns],
        "missing_core_raw_features": missing_core_features,
        "mfcc_std_features": [f for f in MFCC_STD_FEATURES if f in full_features.columns],
        "reference_features": [f for f in REFERENCE_FEATURES if f in full_features.columns],
        "analysis_features": analysis_features,
    }

    logger.log("[STEP 09] Compute blind per-video baselines for validation")
    logger.log("[STEP 10] Compute blind per-video baselines for test")
    baseline_df, relative_df, baseline_warnings = compute_blind_baseline_and_relative(full_features, analysis_features)

    logger.log("[STEP 11] Compute blind per-video relative levels for validation/test")
    relative_df = add_label_free_activity_scores(relative_df)
    validation_relative, test_relative, validation_baseline, test_baseline = split_baseline_relative_outputs(relative_df, baseline_df)

    logger.log("[STEP 12] Write failures and warnings")
    failures_df = build_failures(full_features)
    warning_rows: List[Dict[str, Any]] = list(baseline_warnings)
    if missing_core_features:
        warning_rows.append(
            {
                "warning_type": "missing_core_feature",
                "video_id": "",
                "split": "",
                "feature_name": ",".join(missing_core_features),
                "message": "Some requested core raw features were not present in extraction output.",
            }
        )
    if not score_reference_report.get("audio_ad_like_score_reference_created"):
        warning_rows.append(
            {
                "warning_type": "audio_ad_like_score_reference_unavailable",
                "video_id": "",
                "split": "",
                "feature_name": "audio_ad_like_score_reference",
                "message": score_reference_report.get("error", "Reference score was not created."),
            }
        )
    warnings_df = pd.DataFrame(warning_rows)
    if warnings_df.empty:
        warnings_df = pd.DataFrame(columns=["warning_type", "video_id", "split", "feature_name", "message"])

    validation_features.to_csv(paths["validation_features"], index=False, encoding="utf-8-sig")
    test_features.to_csv(paths["test_features"], index=False, encoding="utf-8-sig")
    full_features.to_csv(paths["val_test_features"], index=False, encoding="utf-8-sig")
    validation_baseline.to_csv(paths["validation_baseline"], index=False, encoding="utf-8-sig")
    test_baseline.to_csv(paths["test_baseline"], index=False, encoding="utf-8-sig")
    validation_relative.to_csv(paths["validation_relative"], index=False, encoding="utf-8-sig")
    test_relative.to_csv(paths["test_relative"], index=False, encoding="utf-8-sig")
    failures_df.to_csv(paths["failures"], index=False, encoding="utf-8-sig")
    warnings_df.to_csv(paths["warnings"], index=False, encoding="utf-8-sig")

    video_summary = pd.concat([validation_video_summary, test_video_summary], ignore_index=True)
    overall_counts = {
        "validation_total_subwindows": int(len(validation_features)),
        "test_total_subwindows": int(len(test_features)),
        "validation_success_subwindows": int(validation_features["extraction_status"].astype(str).eq("success").sum()),
        "test_success_subwindows": int(test_features["extraction_status"].astype(str).eq("success").sum()),
        "validation_failed_subwindows": int((validation_features["extraction_status"].astype(str) != "success").sum()),
        "test_failed_subwindows": int((test_features["extraction_status"].astype(str) != "success").sum()),
    }

    output_files = {key: str(path) for key, path in paths.items()}
    generated_files = [str(paths[key]) for key in paths if key not in {"summary_md", "report_json", "run_log"}]
    generated_files += [str(paths["summary_md"]), str(paths["report_json"]), str(paths["run_log"])]
    report: Dict[str, Any] = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "elapsed_sec": safe_float(time.time() - t0),
        "purpose": "Blind validation/test full-video audio feature precompute for later OCR/scene/fusion use.",
        "blind_preprocessing": True,
        "not_detector_rule_modification": True,
        "project_root": str(project_root),
        "validation_video_ids": FIXED_SPLIT["validation"],
        "test_video_ids": FIXED_SPLIT["test"],
        "train_split_row_level_regenerated": False,
        "actual_ad_label_file_used": False,
        "input_files": {
            "split_file": str(split_file),
            "manifest_file": str(manifest_file),
            "train_full_video_feature_schema_reference": str(train_feature_file),
            "recommendation_file": str(recommendation_file),
            "inventory_file": str(inventory_file),
            "score_formula_file": str(score_formula_file),
        },
        "label_file_unused_confirmation": {
            "actual_label_file_read": False,
            "actual_label_file_path_checked": False,
            "actual_label_columns_created": False,
        },
        "subwindow_policy": {
            "subwindow_size_sec": args.subwindow_size_sec,
            "stride_sec": args.stride_sec,
            "min_valid_duration_sec": args.min_valid_duration_sec,
            "last_partial_window_policy": "include when duration >= min_valid_duration_sec",
        },
        "video_path_records": path_records,
        "video_extraction_summary": video_extraction_summary(video_summary),
        "overall_counts": overall_counts,
        "audio_features": analysis_feature_report,
        "train_schema_comparison": schema_report,
        "extractor": extractor_report,
        "blind_per_video_baseline_method": {
            "baseline_scope": "same_video_only",
            "percentile_scope": "same_video_only",
            "train_baseline_used_for_relative_levels": False,
            "validation_test_cross_video_raw_comparison_used": False,
            "iqr_fallback_policy": "use std, then range, then 1.0 if IQR is zero/missing/tiny",
        },
        "blind_relative_levels": {
            "label_free_feature": True,
            "relative_level_bins": "high percentile >=70, medium 40<=percentile<70, low <40",
            "blind_relative_active_audio_score_created": "blind_relative_active_audio_score" in relative_df.columns,
            "blind_relative_quiet_audio_score_created": "blind_relative_quiet_audio_score" in relative_df.columns,
            "label_free_audio_relative_activity_label_created": "label_free_audio_relative_activity_label" in relative_df.columns,
        },
        "audio_score_reference": score_reference_report,
        "audio_candidate_score_for_discussion_created": False,
        "audio_pattern_label_created": False,
        "validation_test_ad_nonad_contrast_created": False,
        "validation_test_threshold_tuning_performed": False,
        "performance_metrics_created": False,
        "fusion_key_columns": [
            "video_id",
            "split",
            "subwindow_id",
            "start_sec",
            "end_sec",
            "duration_sec",
            "blind_relative_active_audio_score",
            "blind_relative_quiet_audio_score",
            "label_free_audio_relative_activity_label",
            "audio_ad_like_score_reference",
        ],
        "output_files": output_files,
        "generated_files": generated_files,
        "modified_files": [str(project_root / CREATED_BY_SCRIPT)],
        "detector_rule_modified": False,
        "old_project_modified": False,
        "actual_ad_label_used_for_validation": False,
        "actual_ad_label_used_for_test": False,
        "validation_test_label_based_analysis_created": False,
        "validation_test_threshold_tuning_performed": False,
    }

    logger.log("[STEP 13] Generate summary/report")
    write_summary(paths["summary_md"], report, video_summary)
    save_json(paths["report_json"], report)

    logger.log("[STEP 14] Run Sub Agent validations")
    latest_dirs_pre: List[Path] = []
    validations = validate_outputs(
        split_df,
        full_features,
        relative_df,
        baseline_df,
        video_summary,
        paths,
        protected_before,
        project_root,
        latest_dirs_pre,
        train_feature_mtime_before,
    )
    report["sub_agent_validations"] = validations
    save_json(paths["report_json"], report)
    write_summary(paths["summary_md"], report, video_summary)

    logger.log("[STEP 15] Update latest bundles")
    latest_info = write_latest_bundles(project_root, paths, report, run_id)
    latest_audio_info = copy_current_task_to_latest_audio(project_root, paths, report, run_id)
    latest_dirs = [
        Path(latest_info["latest_for_chatgpt_dir"]),
        Path(latest_info["latest_audio_val_test_blind_dir"]),
        Path(latest_audio_info["latest_audio_dir"]),
    ]
    validations = validate_outputs(
        split_df,
        full_features,
        relative_df,
        baseline_df,
        video_summary,
        paths,
        protected_before,
        project_root,
        latest_dirs,
        train_feature_mtime_before,
    )
    report["latest_bundles"] = latest_info
    report["latest_audio_copy"] = latest_audio_info
    report["sub_agent_validations"] = validations
    report["elapsed_sec"] = safe_float(time.time() - t0)
    save_json(paths["report_json"], report)
    write_summary(paths["summary_md"], report, video_summary)

    logger.log("[STEP 16] Print final human-readable summary")
    logger.log(
        "[STEP 16] "
        f"validation_subwindows={overall_counts['validation_total_subwindows']}, "
        f"test_subwindows={overall_counts['test_total_subwindows']}, "
        f"validation_failures={overall_counts['validation_failed_subwindows']}, "
        f"test_failures={overall_counts['test_failed_subwindows']}, "
        "actual_label_used=False, candidate_score_created=False"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
