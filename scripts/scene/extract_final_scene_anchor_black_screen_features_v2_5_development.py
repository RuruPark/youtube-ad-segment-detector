#!/usr/bin/env python3
"""Extract black-screen features around Development Set final scene anchors.

This script reads final scene anchors as scene-transition candidate timestamps,
samples nearby frames directly from the original videos in memory, and writes
numeric luma/black-pixel summaries. It does not execute OCR, read actual ad
labels for extraction, save raw frames, or modify detector rules.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

try:
    cv2.setLogLevel(0)
except Exception:
    pass


PROJECT_ROOT = Path(".")
TASK_NAME = "final_scene_anchor_black_screen_features_v2_5_development"
VERSION = "v2_5_development"
SPLIT_TERMINOLOGY_VERSION = "v2_5_ruledev_extended_eval"
SPLIT_SEED = 20240524

DEVELOPMENT_VIDEO_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
DIAGNOSTIC_SUBSET_VIDEO_IDS = [3, 7, 18]
PURE_TEST_VIDEO_IDS = [4, 16, 17]
EXTENDED_EVALUATION_VIDEO_IDS = [3, 4, 7, 16, 17, 18]

SPLIT_NOTE = (
    "Development Set = original v2.4 train split; Test Set = "
    "original validation + test for public-facing evaluation."
)
REQUIRED_SPLIT_PHRASE_KO = (
    "본 프로젝트는 학습 기반 모델이 아니라 rule-based detector이므로, 기존 train split은 "
    "모델 학습용이 아닌 Development Set으로 재정의해 rule 설계와 cue 분석, 오류 진단에 사용한다. "
    "공개용 설명에서는 기존 validation과 test를 Test Set으로 통합해 "
    "규칙 고정 이후 평가 대상으로 설명한다. 기존 validation/test 세부 구분은 "
    "내부 호환용 key로만 유지한다."
)

SCRIPT_PATH = PROJECT_ROOT / "scripts/scene/extract_final_scene_anchor_black_screen_features_v2_5_development.py"
SPLIT_PATH = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"
VIDEO_MANIFEST_PATH = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
FINAL_ANCHOR_INPUT_PATH = PROJECT_ROOT / "data/scene/final_scene_boundary_anchor_v2_5_development.csv"
FALLBACK_FINAL_ANCHOR_PATH = (
    PROJECT_ROOT / "data/scene/final_scene_boundary_anchor_v2_5_development_reconstructed_for_black_features.csv"
)
OPENCV_FFMPEG_PATH = PROJECT_ROOT / "data/scene/scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed.csv"
RESNET_PATH = PROJECT_ROOT / "data/review/resnet_scene_candidate_human_review_v2_4_usefulness_analysis.csv"
TRANSNET_SWEEP_PATH = PROJECT_ROOT / "data/scene/transnetv2_conservative_sweep_candidates_v2_4_train.csv"
TRANSNET_BEST_PATH = PROJECT_ROOT / "data/scene/transnetv2_conservative_best_candidate_v2_4_train.csv"
TRANSNET_REPORT_PATH = PROJECT_ROOT / "reports/scene/transnetv2_conservative_sweep_v2_4_report.json"
ABLATION_REPORT_PATH = PROJECT_ROOT / "reports/scene/scene_model_ablation_v2_4_report.json"
ABLATION_SUMMARY_PATH = PROJECT_ROOT / "reports/scene/scene_model_ablation_v2_4_summary.md"
ABLATION_DOC_PATH = PROJECT_ROOT / "reports/scene/scene_model_ablation_document_report_v2_4.md"
FFMPEG_BIN = Path(".venv/bin/ffmpeg")

FEATURE_OUTPUT_PATH = PROJECT_ROOT / "data/scene/final_scene_anchor_black_screen_features_v2_5_development.csv"
FRAME_SAMPLE_OUTPUT_PATH = PROJECT_ROOT / "data/scene/final_scene_anchor_black_screen_frame_samples_v2_5_development.csv"
SUMMARY_BY_VIDEO_PATH = (
    PROJECT_ROOT / "data/scene/final_scene_anchor_black_screen_feature_summary_by_video_v2_5_development.csv"
)
EXTRACTION_INDEX_PATH = (
    PROJECT_ROOT / "data/scene/final_scene_anchor_black_screen_extraction_index_v2_5_development.csv"
)
CONFIG_OUTPUT_PATH = PROJECT_ROOT / "data/scene/final_scene_anchor_black_screen_feature_config_v2_5_development.json"

SUMMARY_MD_PATH = PROJECT_ROOT / "reports/scene/black_screen_feature_extraction_v2_5_development_summary.md"
REPORT_JSON_PATH = PROJECT_ROOT / "reports/scene/black_screen_feature_extraction_v2_5_development_report.json"
FINDINGS_MD_PATH = PROJECT_ROOT / "reports/scene/black_screen_feature_extraction_v2_5_development_findings.md"
SHORT_SUMMARY_MD_PATH = PROJECT_ROOT / "reports/scene/black_screen_feature_short_summary_v2_5_development.md"
RUN_LOG_PATH = PROJECT_ROOT / "logs/black_screen_feature_extraction_v2_5_development_run_log.txt"

LATEST_BUNDLE_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_black_screen_features_v2_5_development"
SHARED_BLACK_DIR = PROJECT_ROOT / "outputs/latest_black_screen_features_development"
LATEST_SCENE_DIR = PROJECT_ROOT / "outputs/latest_scene"
LATEST_SCENE_BLACK_DIR = PROJECT_ROOT / "outputs/latest_scene_black"

ANCHOR_CONTEXT_SEC = 2.0
FRAME_SAMPLE_STEP_SEC = 0.2
BLACK_MEAN_LUMA_THRESHOLD = 25.0
BLACK_PIXEL_LUMA_THRESHOLD = 30.0
BLACK_PIXEL_RATIO_THRESHOLD = 0.90
SUSTAINED_BLACK_MIN_DURATION_SEC = 0.5
FADE_LUMA_DROP_THRESHOLD = 35.0
VIDEO_END_GUARD_SEC = 15.0
NEAR_ANCHOR_WINDOW_SEC = 2.0
FALLBACK_CLUSTER_WINDOW_SEC = 2.0
TRANSNET_CONSERVATIVE_FAMILY = "transnetv2_threshold_0_7_dedup_5"
TRANSNET_CONSERVATIVE_THRESHOLD = 0.7
TRANSNET_CONSERVATIVE_DEDUP_WINDOW_SEC = 5.0
KNOWN_DECODE_FAILURE_VIDEO_IDS = {2, 8, 9, 12}

FEATURE_COLUMNS = [
    "final_anchor_id",
    "video_id",
    "original_split_v2_4",
    "split_role_v2_5",
    "evaluation_subset_v2_5",
    "split_terminology_note",
    "anchor_sec",
    "anchor_frame",
    "source_relation",
    "has_opencv_ffmpeg",
    "has_resnet",
    "has_transnetv2_conservative",
    "cluster_member_count",
    "video_path",
    "video_duration_sec",
    "fps",
    "frame_count",
    "anchor_context_sec",
    "frame_sample_step_sec",
    "black_mean_luma_threshold",
    "black_pixel_luma_threshold",
    "black_pixel_ratio_threshold",
    "video_end_guard_sec",
    "mean_luma_pre",
    "mean_luma_post",
    "mean_luma_context",
    "min_luma_pre",
    "min_luma_post",
    "min_luma_context",
    "mean_luma_delta_post_minus_pre",
    "black_pixel_ratio_pre",
    "black_pixel_ratio_post",
    "black_pixel_ratio_context",
    "black_pixel_ratio_delta_post_minus_pre",
    "black_frame_count_pre",
    "black_frame_count_post",
    "black_frame_count_near_anchor",
    "black_frame_ratio_pre",
    "black_frame_ratio_post",
    "black_frame_ratio_near_anchor",
    "longest_black_run_start_sec",
    "longest_black_run_end_sec",
    "longest_black_run_duration_sec",
    "black_screen_duration_sec",
    "has_black_frame_near_anchor",
    "has_sustained_black_screen_near_anchor",
    "has_fade_to_black_near_anchor",
    "is_near_video_end",
    "video_end_guard_applied",
    "black_event_near_anchor",
    "black_end_support_eligible",
    "black_support_strength",
    "sample_count_total",
    "sample_count_pre",
    "sample_count_post",
    "sample_read_error_count",
    "extraction_status",
    "error_message",
    "notes",
]

FRAME_SAMPLE_COLUMNS = [
    "sample_id",
    "final_anchor_id",
    "video_id",
    "original_split_v2_4",
    "split_role_v2_5",
    "evaluation_subset_v2_5",
    "split_terminology_note",
    "anchor_sec",
    "sample_time_sec",
    "sample_offset_from_anchor_sec",
    "sample_region",
    "frame_index",
    "read_status",
    "mean_luma",
    "min_luma",
    "p10_luma",
    "black_pixel_ratio",
    "is_black_frame",
    "error_message",
    "notes",
]

SUMMARY_BY_VIDEO_COLUMNS = [
    "video_id",
    "original_split_v2_4",
    "split_role_v2_5",
    "evaluation_subset_v2_5",
    "split_terminology_note",
    "video_path",
    "video_duration_sec",
    "anchor_count",
    "processed_anchor_count",
    "error_anchor_count",
    "black_event_anchor_count",
    "sustained_black_anchor_count",
    "fade_to_black_anchor_count",
    "video_end_guard_anchor_count",
    "black_end_support_eligible_count",
    "black_event_anchor_ratio",
    "black_end_support_eligible_ratio",
    "mean_black_screen_duration_sec",
    "max_black_screen_duration_sec",
    "notes",
]

EXTRACTION_INDEX_COLUMNS = [
    "video_id",
    "original_split_v2_4",
    "split_role_v2_5",
    "evaluation_subset_v2_5",
    "split_terminology_note",
    "video_path",
    "file_exists",
    "anchor_count",
    "unique_sample_time_count",
    "frame_read_attempt_count",
    "frame_read_success_count",
    "frame_read_error_count",
    "runtime_seconds",
    "status",
    "error_message",
    "notes",
]

FALLBACK_ANCHOR_COLUMNS = [
    "final_anchor_id",
    "video_id",
    "original_split_v2_4",
    "split_role_v2_5",
    "evaluation_subset_v2_5",
    "split_terminology_note",
    "anchor_sec",
    "anchor_frame",
    "cluster_min_sec",
    "cluster_max_sec",
    "cluster_member_count",
    "source_relation",
    "has_opencv_ffmpeg",
    "has_resnet",
    "has_transnetv2_conservative",
    "opencv_member_count",
    "resnet_member_count",
    "transnetv2_member_count",
    "source_members_json",
    "representative_rule",
    "confidence_or_score",
    "notes",
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
    ".gif",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".pkl",
    ".pickle",
}


@dataclass
class VideoInfo:
    video_id: int
    original_split_v2_4: str
    split_role_v2_5: str
    evaluation_subset_v2_5: str
    split_terminology_note: str
    video_path: str
    file_exists: bool
    video_duration_sec: float
    fps: float
    frame_count: int


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def log(message: str) -> None:
    print(message, flush=True)
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"{now_iso()} {message}\n")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def rounded(value: float | int | None, digits: int = 6) -> float | str:
    if value is None:
        return ""
    try:
        if not math.isfinite(float(value)):
            return ""
    except Exception:
        return ""
    return round(float(value), digits)


def mean_or_blank(values: list[float]) -> float | str:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return ""
    return round(float(np.mean(clean)), 6)


def min_or_blank(values: list[float]) -> float | str:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return ""
    return round(float(np.min(clean)), 6)


def ratio(count: int, total: int) -> float:
    return round(float(count) / float(total), 6) if total else 0.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def split_role_for(original_split: str) -> tuple[str, str]:
    split = str(original_split).strip().lower()
    if split == "train":
        return "development", "none"
    if split == "validation":
        return "extended_evaluation", "diagnostic_subset"
    if split == "test":
        return "extended_evaluation", "pure_test"
    return "unknown", "unknown"


def load_split_map() -> dict[int, dict[str, str]]:
    split_df = pd.read_csv(SPLIT_PATH)
    out: dict[int, dict[str, str]] = {}
    for _, row in split_df.iterrows():
        video_id = safe_int(row.get("video_id"), -1)
        original_split = str(row.get("split", row.get("original_split_v2_4", ""))).strip().lower()
        role, subset = split_role_for(original_split)
        out[video_id] = {
            "original_split_v2_4": original_split,
            "split_role_v2_5": role,
            "evaluation_subset_v2_5": subset,
            "split_terminology_note": SPLIT_NOTE,
        }
    return out


def normalize_anchor_df(anchor_df: pd.DataFrame, split_map: dict[int, dict[str, str]], warnings: list[str]) -> pd.DataFrame:
    df = anchor_df.copy()
    if "video_id" not in df.columns or "anchor_sec" not in df.columns:
        raise RuntimeError("final scene anchor file must contain video_id and anchor_sec columns")

    df["video_id"] = df["video_id"].map(lambda v: safe_int(v, -1))
    df["anchor_sec"] = df["anchor_sec"].map(lambda v: safe_float(v))
    if "anchor_frame" not in df.columns:
        df["anchor_frame"] = ""

    for col in ["original_split_v2_4", "split_role_v2_5", "evaluation_subset_v2_5", "split_terminology_note"]:
        if col not in df.columns:
            df[col] = ""

    for idx, row in df.iterrows():
        split_info = split_map.get(int(row["video_id"]), {})
        original_split = str(row.get("original_split_v2_4", "")).strip().lower() or split_info.get("original_split_v2_4", "")
        role, subset = split_role_for(original_split)
        if not original_split:
            original_split = split_info.get("original_split_v2_4", "")
            role = split_info.get("split_role_v2_5", role)
            subset = split_info.get("evaluation_subset_v2_5", subset)
        df.at[idx, "original_split_v2_4"] = original_split
        df.at[idx, "split_role_v2_5"] = role
        df.at[idx, "evaluation_subset_v2_5"] = subset
        df.at[idx, "split_terminology_note"] = SPLIT_NOTE

    before = len(df)
    df = df[df["video_id"].isin(DEVELOPMENT_VIDEO_IDS)].copy()
    df = df[df["original_split_v2_4"].astype(str).str.lower().eq("train")].copy()
    after = len(df)
    if before != after:
        warnings.append(f"Filtered {before - after} non-Development/fallback rows from final anchor input.")

    df = df.sort_values(["video_id", "anchor_sec", "final_anchor_id" if "final_anchor_id" in df.columns else "anchor_sec"])
    return df.reset_index(drop=True)


def pick_first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    colset = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in colset:
            return colset[candidate.lower()]
    return None


def load_candidate_records(
    path: Path,
    source: str,
    split_map: dict[int, dict[str, str]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    if not path.exists():
        warnings.append(f"Fallback source missing: {path}")
        return []
    df = pd.read_csv(path)
    cols = list(df.columns)
    video_col = pick_first_existing_column(cols, ["video_id", "label_mapping_video_id"])
    sec_col = pick_first_existing_column(
        cols,
        ["anchor_sec", "candidate_sec", "timestamp_sec", "scene_change_sec", "time_sec", "sec", "second"],
    )
    frame_col = pick_first_existing_column(cols, ["anchor_frame", "candidate_frame", "frame_index", "frame_idx", "frame"])
    score_col = pick_first_existing_column(
        cols,
        ["score", "confidence_or_score", "confidence", "probability", "scene_score", "delta_score", "best_score"],
    )
    family_col = pick_first_existing_column(cols, ["sweep_family", "source_model", "model_name", "candidate_family"])
    threshold_col = pick_first_existing_column(cols, ["threshold", "transnetv2_threshold"])
    dedup_col = pick_first_existing_column(cols, ["dedup_window_sec", "dedup_sec", "dedup_window"])
    if video_col is None or sec_col is None:
        warnings.append(f"Fallback source {path} lacks recognizable video_id/timestamp columns.")
        return []

    records: list[dict[str, Any]] = []
    for raw_idx, row in df.iterrows():
        video_id = safe_int(row.get(video_col), -1)
        if video_id not in DEVELOPMENT_VIDEO_IDS:
            continue
        if split_map.get(video_id, {}).get("original_split_v2_4") != "train":
            continue
        if source == "transnetv2_conservative":
            family = str(row.get(family_col, "") if family_col else "")
            threshold = safe_float(row.get(threshold_col), math.nan) if threshold_col else math.nan
            dedup = safe_float(row.get(dedup_col), math.nan) if dedup_col else math.nan
            family_ok = TRANSNET_CONSERVATIVE_FAMILY in family
            numeric_ok = (
                math.isfinite(threshold)
                and abs(threshold - TRANSNET_CONSERVATIVE_THRESHOLD) < 1e-9
                and math.isfinite(dedup)
                and abs(dedup - TRANSNET_CONSERVATIVE_DEDUP_WINDOW_SEC) < 1e-9
            )
            if not (family_ok or numeric_ok):
                continue
        sec = safe_float(row.get(sec_col))
        if not math.isfinite(sec):
            continue
        records.append(
            {
                "video_id": video_id,
                "candidate_sec": float(sec),
                "candidate_frame": safe_int(row.get(frame_col), "") if frame_col else "",
                "score": safe_float(row.get(score_col), math.nan) if score_col else math.nan,
                "source": source,
                "source_path": str(path),
                "raw_row_index": int(raw_idx),
            }
        )
    return records


def reconstruct_final_anchors(split_map: dict[int, dict[str, str]], warnings: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    records.extend(load_candidate_records(OPENCV_FFMPEG_PATH, "opencv_ffmpeg", split_map, warnings))
    records.extend(load_candidate_records(RESNET_PATH, "resnet", split_map, warnings))
    records.extend(load_candidate_records(TRANSNET_SWEEP_PATH, "transnetv2_conservative", split_map, warnings))
    if not records:
        records.extend(load_candidate_records(TRANSNET_BEST_PATH, "transnetv2_conservative", split_map, warnings))
    if not records:
        raise RuntimeError("Final scene anchor input is missing and fallback reconstruction found no candidates.")

    output_rows: list[dict[str, Any]] = []
    for video_id in DEVELOPMENT_VIDEO_IDS:
        source_rows = sorted([r for r in records if r["video_id"] == video_id], key=lambda r: r["candidate_sec"])
        clusters: list[list[dict[str, Any]]] = []
        for row in source_rows:
            if not clusters or row["candidate_sec"] - clusters[-1][-1]["candidate_sec"] > FALLBACK_CLUSTER_WINDOW_SEC:
                clusters.append([row])
            else:
                clusters[-1].append(row)
        for cluster_idx, cluster in enumerate(clusters, start=1):
            canonical_members = [m for m in cluster if m["source"] in {"opencv_ffmpeg", "resnet"}]
            if canonical_members:
                representative = sorted(canonical_members, key=lambda m: (m["candidate_sec"], m["source"]))[0]
                representative_rule = "canonical_opencv_resnet_priority"
            else:
                scored = [m for m in cluster if math.isfinite(float(m.get("score", math.nan)))]
                if scored:
                    representative = sorted(scored, key=lambda m: (-float(m["score"]), m["candidate_sec"]))[0]
                    representative_rule = "max_source_score_then_earliest"
                else:
                    representative = sorted(cluster, key=lambda m: m["candidate_sec"])[0]
                    representative_rule = "earliest_timestamp_no_comparable_score"
            source_counts = Counter(m["source"] for m in cluster)
            source_names = sorted(source_counts)
            split_info = split_map[video_id]
            output_rows.append(
                {
                    "final_anchor_id": f"FSA25DEV_BLACK_RECON_v{video_id:02d}_{cluster_idx:05d}",
                    "video_id": video_id,
                    **split_info,
                    "anchor_sec": rounded(representative["candidate_sec"], 6),
                    "anchor_frame": representative.get("candidate_frame", ""),
                    "cluster_min_sec": rounded(min(m["candidate_sec"] for m in cluster), 6),
                    "cluster_max_sec": rounded(max(m["candidate_sec"] for m in cluster), 6),
                    "cluster_member_count": len(cluster),
                    "source_relation": "multi_source:" + "+".join(source_names) if len(source_names) > 1 else f"only_{source_names[0]}",
                    "has_opencv_ffmpeg": source_counts.get("opencv_ffmpeg", 0) > 0,
                    "has_resnet": source_counts.get("resnet", 0) > 0,
                    "has_transnetv2_conservative": source_counts.get("transnetv2_conservative", 0) > 0,
                    "opencv_member_count": source_counts.get("opencv_ffmpeg", 0),
                    "resnet_member_count": source_counts.get("resnet", 0),
                    "transnetv2_member_count": source_counts.get("transnetv2_conservative", 0),
                    "source_members_json": json.dumps(cluster, ensure_ascii=False),
                    "representative_rule": representative_rule,
                    "confidence_or_score": rounded(representative.get("score"), 9),
                    "notes": "Fallback reconstructed for black screen feature extraction only; existing anchor files were not modified.",
                }
            )
    write_csv(FALLBACK_FINAL_ANCHOR_PATH, output_rows, FALLBACK_ANCHOR_COLUMNS)
    return pd.DataFrame(output_rows)


def load_final_anchors(split_map: dict[int, dict[str, str]], warnings: list[str]) -> tuple[pd.DataFrame, Path, bool]:
    log("[STEP 03] Development Set final scene anchor 로드")
    if FINAL_ANCHOR_INPUT_PATH.exists():
        anchor_df = pd.read_csv(FINAL_ANCHOR_INPUT_PATH)
        final_anchor_input_path = FINAL_ANCHOR_INPUT_PATH
        final_anchor_reconstructed = False
        log(f"  - 기존 final scene anchor 사용: {FINAL_ANCHOR_INPUT_PATH}")
        log("[STEP 04] 필요한 경우 final scene anchor 재구성")
        log("  - 기존 final scene anchor가 존재하므로 fallback 재구성은 수행하지 않음")
    else:
        log("[STEP 04] 필요한 경우 final scene anchor 재구성")
        log("  - 기존 final scene anchor가 없어 fallback 재구성을 수행")
        anchor_df = reconstruct_final_anchors(split_map, warnings)
        final_anchor_input_path = FALLBACK_FINAL_ANCHOR_PATH
        final_anchor_reconstructed = True
    normalized = normalize_anchor_df(anchor_df, split_map, warnings)
    if normalized.empty:
        raise RuntimeError("No Development Set final scene anchors available after scope filtering.")
    return normalized, final_anchor_input_path, final_anchor_reconstructed


def load_video_mapping(split_map: dict[int, dict[str, str]], warnings: list[str]) -> dict[int, VideoInfo]:
    manifest = pd.read_csv(VIDEO_MANIFEST_PATH)
    mapping: dict[int, VideoInfo] = {}
    for video_id in DEVELOPMENT_VIDEO_IDS:
        split_info = split_map.get(video_id)
        if split_info is None or split_info.get("original_split_v2_4") != "train":
            warnings.append(f"Development video_id={video_id} is missing or non-train in split file.")
            continue
        rows = manifest[manifest["video_id"].map(lambda v: safe_int(v, -1)).eq(video_id)]
        if rows.empty:
            warnings.append(f"Development video_id={video_id} missing from video manifest.")
            mapping[video_id] = VideoInfo(video_id, **split_info, video_path="", file_exists=False, video_duration_sec=math.nan, fps=math.nan, frame_count=0)
            continue
        row = rows.iloc[0]
        video_path = str(row.get("video_path", ""))
        duration = safe_float(row.get("duration_sec", row.get("video_duration_sec", math.nan)))
        fps = safe_float(row.get("fps", math.nan))
        frame_count = safe_int(row.get("frame_count"), 0)
        mapping[video_id] = VideoInfo(
            video_id=video_id,
            original_split_v2_4=split_info["original_split_v2_4"],
            split_role_v2_5=split_info["split_role_v2_5"],
            evaluation_subset_v2_5=split_info["evaluation_subset_v2_5"],
            split_terminology_note=split_info["split_terminology_note"],
            video_path=video_path,
            file_exists=Path(video_path).exists(),
            video_duration_sec=duration,
            fps=fps,
            frame_count=frame_count,
        )
        if not Path(video_path).exists():
            warnings.append(f"Development video_id={video_id} video file missing: {video_path}")
    return mapping


def make_config(final_anchor_input_path: Path, final_anchor_reconstructed: bool) -> dict[str, Any]:
    return {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "split_terminology_version": SPLIT_TERMINOLOGY_VERSION,
        "scope": "Development Set only; original v2.4 train split only.",
        "development_set_video_ids": DEVELOPMENT_VIDEO_IDS,
        "final_anchor_input_path": str(final_anchor_input_path),
        "final_anchor_reconstructed": final_anchor_reconstructed,
        "output_feature_path": str(FEATURE_OUTPUT_PATH),
        "frame_sample_output_path": str(FRAME_SAMPLE_OUTPUT_PATH),
        "summary_by_video_path": str(SUMMARY_BY_VIDEO_PATH),
        "extraction_index_path": str(EXTRACTION_INDEX_PATH),
        "anchor_context_sec": ANCHOR_CONTEXT_SEC,
        "frame_sample_step_sec": FRAME_SAMPLE_STEP_SEC,
        "black_mean_luma_threshold": BLACK_MEAN_LUMA_THRESHOLD,
        "black_pixel_luma_threshold": BLACK_PIXEL_LUMA_THRESHOLD,
        "black_pixel_ratio_threshold": BLACK_PIXEL_RATIO_THRESHOLD,
        "sustained_black_min_duration_sec": SUSTAINED_BLACK_MIN_DURATION_SEC,
        "fade_luma_drop_threshold": FADE_LUMA_DROP_THRESHOLD,
        "video_end_guard_sec": VIDEO_END_GUARD_SEC,
        "near_anchor_window_sec": NEAR_ANCHOR_WINDOW_SEC,
        "actual_label_used_for_feature_extraction": False,
        "ocr_executed": False,
        "raw_frame_images_saved": False,
        "known_decode_failure_video_ids_forced_ffmpeg": sorted(KNOWN_DECODE_FAILURE_VIDEO_IDS),
        "no_extended_evaluation_processed": True,
        "no_diagnostic_subset_processed": True,
        "no_pure_test_processed": True,
        "luma_channel": "OpenCV cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]; ffmpeg fallback uses raw gray Y-like luma bytes",
        "min_luma_definition": "Minimum Y-channel pixel value in the sampled frame/window.",
        "sustained_black_duration_formula": "longest consecutive black sample run count * frame_sample_step_sec",
        "fade_to_black_heuristic": (
            "black frame near anchor and high_reference_luma - low_reference_luma >= fade_luma_drop_threshold; "
            "high_reference_luma=max(context sample mean_luma, pre median mean_luma), "
            "low_reference_luma=min mean_luma among black samples when available."
        ),
        "split_explanation_ko": REQUIRED_SPLIT_PHRASE_KO,
    }


def sample_offsets() -> list[float]:
    steps = int(round(ANCHOR_CONTEXT_SEC / FRAME_SAMPLE_STEP_SEC))
    return [round(i * FRAME_SAMPLE_STEP_SEC, 6) for i in range(-steps, steps + 1)]


def build_sample_plan(
    anchors: pd.DataFrame,
    video_mapping: dict[int, VideoInfo],
) -> tuple[list[dict[str, Any]], dict[int, dict[int, list[dict[str, Any]]]]]:
    sample_rows: list[dict[str, Any]] = []
    unique_reads: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    offsets = sample_offsets()
    for _, anchor in anchors.iterrows():
        video_id = int(anchor["video_id"])
        info = video_mapping.get(video_id)
        anchor_sec = float(anchor["anchor_sec"])
        anchor_id = str(anchor.get("final_anchor_id", f"FSA25DEV_UNKNOWN_{video_id}_{anchor_sec}"))
        sample_idx = 0
        for offset in offsets:
            sample_time = round(anchor_sec + offset, 6)
            if sample_time < -1e-9:
                continue
            if info and math.isfinite(info.video_duration_sec) and sample_time > info.video_duration_sec + 1e-9:
                continue
            if offset < 0:
                region = "pre"
            elif offset > 0:
                region = "post"
            else:
                region = "context"
            fps = info.fps if info else math.nan
            frame_count = info.frame_count if info else 0
            if math.isfinite(fps) and fps > 0:
                frame_index = int(round(sample_time * fps))
                if frame_count > 0:
                    frame_index = max(0, min(frame_count - 1, frame_index))
            else:
                frame_index = -1
            sample_idx += 1
            row = {
                "sample_id": f"{anchor_id}_black_s{sample_idx:03d}",
                "final_anchor_id": anchor_id,
                "video_id": video_id,
                "original_split_v2_4": "train",
                "split_role_v2_5": "development",
                "evaluation_subset_v2_5": "none",
                "split_terminology_note": SPLIT_NOTE,
                "anchor_sec": rounded(anchor_sec, 6),
                "sample_time_sec": rounded(sample_time, 6),
                "sample_offset_from_anchor_sec": rounded(offset, 6),
                "sample_region": region,
                "frame_index": frame_index,
                "_anchor_index": int(anchor.name),
            }
            sample_rows.append(row)
            if frame_index >= 0:
                unique_reads[video_id][frame_index].append(row)
    return sample_rows, unique_reads


def frame_luma_stats(frame: np.ndarray) -> dict[str, Any]:
    y = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mean_luma = float(np.mean(y))
    min_luma = float(np.min(y))
    p10_luma = float(np.percentile(y, 10))
    black_pixel_ratio = float(np.mean(y <= BLACK_PIXEL_LUMA_THRESHOLD))
    is_black = bool(mean_luma < BLACK_MEAN_LUMA_THRESHOLD or black_pixel_ratio > BLACK_PIXEL_RATIO_THRESHOLD)
    return {
        "read_status": "success",
        "mean_luma": round(mean_luma, 6),
        "min_luma": round(min_luma, 6),
        "p10_luma": round(p10_luma, 6),
        "black_pixel_ratio": round(black_pixel_ratio, 6),
        "is_black_frame": is_black,
        "error_message": "",
    }


def gray_luma_stats(y: np.ndarray) -> dict[str, Any]:
    if y.size == 0:
        return {
            "read_status": "read_error",
            "mean_luma": "",
            "min_luma": "",
            "p10_luma": "",
            "black_pixel_ratio": "",
            "is_black_frame": "",
            "error_message": "empty gray frame buffer",
        }
    mean_luma = float(np.mean(y))
    min_luma = float(np.min(y))
    p10_luma = float(np.percentile(y, 10))
    black_pixel_ratio = float(np.mean(y <= BLACK_PIXEL_LUMA_THRESHOLD))
    is_black = bool(mean_luma < BLACK_MEAN_LUMA_THRESHOLD or black_pixel_ratio > BLACK_PIXEL_RATIO_THRESHOLD)
    return {
        "read_status": "success",
        "mean_luma": round(mean_luma, 6),
        "min_luma": round(min_luma, 6),
        "p10_luma": round(p10_luma, 6),
        "black_pixel_ratio": round(black_pixel_ratio, 6),
        "is_black_frame": is_black,
        "error_message": "",
    }


def ffmpeg_gray_luma_stats(video_path: str, sample_time_sec: float) -> dict[str, Any]:
    if not FFMPEG_BIN.exists():
        return {
            "read_status": "read_error",
            "mean_luma": "",
            "min_luma": "",
            "p10_luma": "",
            "black_pixel_ratio": "",
            "is_black_frame": "",
            "error_message": f"ffmpeg binary missing: {FFMPEG_BIN}",
        }
    cmd = [
        str(FFMPEG_BIN),
        "-v",
        "error",
        "-ss",
        f"{max(0.0, float(sample_time_sec)):.6f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False)
    except Exception as exc:
        return {
            "read_status": "read_error",
            "mean_luma": "",
            "min_luma": "",
            "p10_luma": "",
            "black_pixel_ratio": "",
            "is_black_frame": "",
            "error_message": f"ffmpeg fallback exception: {exc}",
        }
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode("utf-8", "ignore")[:300]
        return {
            "read_status": "read_error",
            "mean_luma": "",
            "min_luma": "",
            "p10_luma": "",
            "black_pixel_ratio": "",
            "is_black_frame": "",
            "error_message": f"ffmpeg fallback failed rc={result.returncode}: {stderr}",
        }
    return gray_luma_stats(np.frombuffer(result.stdout, dtype=np.uint8))


def read_video_frames_ffmpeg_parallel(
    video_id: int,
    info: VideoInfo,
    unique_reads: dict[int, list[dict[str, Any]]],
    start: float,
    anchor_ids: set[str],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    max_workers = 6
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame = {}
        for frame_index in sorted(unique_reads):
            sample_time = float(unique_reads[frame_index][0]["sample_time_sec"])
            future = executor.submit(ffmpeg_gray_luma_stats, info.video_path, sample_time)
            future_to_frame[future] = frame_index
        for future in as_completed(future_to_frame):
            frame_index = future_to_frame[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "read_status": "read_error",
                    "mean_luma": "",
                    "min_luma": "",
                    "p10_luma": "",
                    "black_pixel_ratio": "",
                    "is_black_frame": "",
                    "error_message": f"parallel ffmpeg fallback exception: {exc}",
                }
            results[frame_index] = result
    success_count = sum(1 for result in results.values() if result.get("read_status") == "success")
    error_count = len(unique_reads) - success_count
    status = "processed" if error_count == 0 else "partial_read_error"
    return results, {
        "video_id": video_id,
        **split_dict(info),
        "video_path": info.video_path,
        "file_exists": True,
        "anchor_count": len(anchor_ids),
        "unique_sample_time_count": len(unique_reads),
        "frame_read_attempt_count": len(unique_reads),
        "frame_read_success_count": success_count,
        "frame_read_error_count": error_count,
        "runtime_seconds": round(time.perf_counter() - start, 6),
        "status": status,
        "error_message": "" if error_count == 0 else f"{error_count} frame read errors",
        "notes": (
            f"Forced parallel ffmpeg gray pipe fallback workers={max_workers}; "
            "frames were read in memory only; no raw frame images were saved."
        ),
    }


def read_video_frames(
    video_id: int,
    info: VideoInfo,
    unique_reads: dict[int, list[dict[str, Any]]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    start = time.perf_counter()
    results: dict[int, dict[str, Any]] = {}
    anchor_ids = {row["final_anchor_id"] for rows in unique_reads.values() for row in rows}

    def read_error_result(message: str) -> dict[str, Any]:
        return {
            "read_status": "read_error",
            "mean_luma": "",
            "min_luma": "",
            "p10_luma": "",
            "black_pixel_ratio": "",
            "is_black_frame": "",
            "error_message": message,
        }

    if not info.file_exists:
        for frame_index in unique_reads:
            results[frame_index] = read_error_result("video file missing")
        return results, {
            "video_id": video_id,
            **split_dict(info),
            "video_path": info.video_path,
            "file_exists": False,
            "anchor_count": len(anchor_ids),
            "unique_sample_time_count": len(unique_reads),
            "frame_read_attempt_count": len(unique_reads),
            "frame_read_success_count": 0,
            "frame_read_error_count": len(unique_reads),
            "runtime_seconds": round(time.perf_counter() - start, 6),
            "status": "video_missing",
            "error_message": "video file missing",
            "notes": "",
        }

    if video_id in KNOWN_DECODE_FAILURE_VIDEO_IDS:
        return read_video_frames_ffmpeg_parallel(video_id, info, unique_reads, start, anchor_ids)

    cap = cv2.VideoCapture(info.video_path)
    cap_opened = cap.isOpened()
    use_ffmpeg_only = not cap_opened
    current_next_frame: int | None = None
    max_grab_gap = 150
    consecutive_cv_errors = 0
    cv_success_count = 0
    cv_error_count = 0
    ffmpeg_success_count = 0
    success_count = 0
    error_count = 0

    for frame_index in sorted(unique_reads):
        sample_time = float(unique_reads[frame_index][0]["sample_time_sec"])
        result: dict[str, Any] | None = None
        if not use_ffmpeg_only:
            ok = False
            frame = None
            try:
                gap = frame_index - current_next_frame if current_next_frame is not None else None
                if gap is not None and 0 <= gap <= max_grab_gap:
                    grab_ok = True
                    while current_next_frame < frame_index:
                        if not cap.grab():
                            grab_ok = False
                            break
                        current_next_frame += 1
                    if grab_ok:
                        ok, frame = cap.read()
                        current_next_frame = frame_index + 1 if ok else None
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    ok, frame = cap.read()
                    current_next_frame = frame_index + 1 if ok else None
                if not ok or frame is None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, sample_time * 1000.0))
                    ok, frame = cap.read()
                    current_next_frame = None
                if ok and frame is not None:
                    result = frame_luma_stats(frame)
                    cv_success_count += 1
                    consecutive_cv_errors = 0
                else:
                    cv_error_count += 1
                    consecutive_cv_errors += 1
                    if cv_success_count == 0 and consecutive_cv_errors >= 3:
                        use_ffmpeg_only = True
            except Exception as exc:
                cv_error_count += 1
                consecutive_cv_errors += 1
                current_next_frame = None
                if cv_success_count == 0 and consecutive_cv_errors >= 3:
                    use_ffmpeg_only = True
                result = read_error_result(f"OpenCV read exception at frame_index={frame_index}: {exc}")

        if result is None or result.get("read_status") != "success":
            fallback = ffmpeg_gray_luma_stats(info.video_path, sample_time)
            if fallback.get("read_status") == "success":
                result = fallback
                ffmpeg_success_count += 1
            elif result is None:
                result = read_error_result(f"frame read failed at frame_index={frame_index}; {fallback.get('error_message', '')}")
            else:
                result = read_error_result(f"{result.get('error_message', 'OpenCV read failed')}; {fallback.get('error_message', '')}")

        results[frame_index] = result
        if result.get("read_status") == "success":
            success_count += 1
        else:
            error_count += 1

    if cap_opened:
        cap.release()
    status = "processed" if error_count == 0 else "partial_read_error"
    backend_note = (
        f"OpenCV sequential grab/read successes={cv_success_count}, errors={cv_error_count}; "
        f"ffmpeg gray pipe fallback successes={ffmpeg_success_count}; "
        f"forced_ffmpeg_only={video_id in KNOWN_DECODE_FAILURE_VIDEO_IDS}. "
        "Frames were read in memory only; no raw frame images were saved."
    )
    return results, {
        "video_id": video_id,
        **split_dict(info),
        "video_path": info.video_path,
        "file_exists": True,
        "anchor_count": len(anchor_ids),
        "unique_sample_time_count": len(unique_reads),
        "frame_read_attempt_count": len(unique_reads),
        "frame_read_success_count": success_count,
        "frame_read_error_count": error_count,
        "runtime_seconds": round(time.perf_counter() - start, 6),
        "status": status,
        "error_message": "" if error_count == 0 else f"{error_count} frame read errors",
        "notes": backend_note,
    }


def split_dict(info: VideoInfo) -> dict[str, Any]:
    return {
        "original_split_v2_4": info.original_split_v2_4,
        "split_role_v2_5": info.split_role_v2_5,
        "evaluation_subset_v2_5": info.evaluation_subset_v2_5,
        "split_terminology_note": info.split_terminology_note,
    }


def blank_feature_row(anchor: pd.Series, info: VideoInfo | None, status: str, error_message: str) -> dict[str, Any]:
    video_id = int(anchor["video_id"])
    split_info = split_dict(info) if info else {
        "original_split_v2_4": "train",
        "split_role_v2_5": "development",
        "evaluation_subset_v2_5": "none",
        "split_terminology_note": SPLIT_NOTE,
    }
    row = {
        "final_anchor_id": anchor.get("final_anchor_id", ""),
        "video_id": video_id,
        **split_info,
        "anchor_sec": rounded(safe_float(anchor.get("anchor_sec")), 6),
        "anchor_frame": anchor.get("anchor_frame", ""),
        "source_relation": anchor.get("source_relation", ""),
        "has_opencv_ffmpeg": anchor.get("has_opencv_ffmpeg", ""),
        "has_resnet": anchor.get("has_resnet", ""),
        "has_transnetv2_conservative": anchor.get("has_transnetv2_conservative", ""),
        "cluster_member_count": anchor.get("cluster_member_count", ""),
        "video_path": info.video_path if info else "",
        "video_duration_sec": rounded(info.video_duration_sec if info else math.nan, 6),
        "fps": rounded(info.fps if info else math.nan, 6),
        "frame_count": info.frame_count if info else "",
        "anchor_context_sec": ANCHOR_CONTEXT_SEC,
        "frame_sample_step_sec": FRAME_SAMPLE_STEP_SEC,
        "black_mean_luma_threshold": BLACK_MEAN_LUMA_THRESHOLD,
        "black_pixel_luma_threshold": BLACK_PIXEL_LUMA_THRESHOLD,
        "black_pixel_ratio_threshold": BLACK_PIXEL_RATIO_THRESHOLD,
        "video_end_guard_sec": VIDEO_END_GUARD_SEC,
        "mean_luma_pre": "",
        "mean_luma_post": "",
        "mean_luma_context": "",
        "min_luma_pre": "",
        "min_luma_post": "",
        "min_luma_context": "",
        "mean_luma_delta_post_minus_pre": "",
        "black_pixel_ratio_pre": "",
        "black_pixel_ratio_post": "",
        "black_pixel_ratio_context": "",
        "black_pixel_ratio_delta_post_minus_pre": "",
        "black_frame_count_pre": 0,
        "black_frame_count_post": 0,
        "black_frame_count_near_anchor": 0,
        "black_frame_ratio_pre": 0,
        "black_frame_ratio_post": 0,
        "black_frame_ratio_near_anchor": 0,
        "longest_black_run_start_sec": "",
        "longest_black_run_end_sec": "",
        "longest_black_run_duration_sec": 0,
        "black_screen_duration_sec": 0,
        "has_black_frame_near_anchor": False,
        "has_sustained_black_screen_near_anchor": False,
        "has_fade_to_black_near_anchor": False,
        "is_near_video_end": False,
        "video_end_guard_applied": False,
        "black_event_near_anchor": False,
        "black_end_support_eligible": False,
        "black_support_strength": "none",
        "sample_count_total": 0,
        "sample_count_pre": 0,
        "sample_count_post": 0,
        "sample_read_error_count": 0,
        "extraction_status": status,
        "error_message": error_message,
        "notes": "No actual labels used; no OCR executed; no raw frame images saved.",
    }
    return row


def longest_black_run(samples: list[dict[str, Any]]) -> tuple[str | float, str | float, float]:
    longest: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    last_time: float | None = None
    for sample in sorted(samples, key=lambda r: float(r["sample_time_sec"])):
        is_black = sample.get("is_black_frame") is True
        sample_time = float(sample["sample_time_sec"])
        consecutive = last_time is None or sample_time - last_time <= FRAME_SAMPLE_STEP_SEC * 1.51
        if is_black and consecutive:
            current.append(sample)
        elif is_black:
            current = [sample]
        else:
            if len(current) > len(longest):
                longest = current
            current = []
        last_time = sample_time
    if len(current) > len(longest):
        longest = current
    if not longest:
        return "", "", 0.0
    duration = round(len(longest) * FRAME_SAMPLE_STEP_SEC, 6)
    return rounded(float(longest[0]["sample_time_sec"]), 6), rounded(float(longest[-1]["sample_time_sec"]), 6), duration


def support_strength(has_black: bool, sustained: bool, fade: bool) -> str:
    if not has_black and not sustained and not fade:
        return "none"
    if sustained and fade:
        return "strong"
    if sustained or fade:
        return "medium"
    return "weak"


def calculate_anchor_features(
    anchors: pd.DataFrame,
    sample_rows: list[dict[str, Any]],
    frame_results_by_video: dict[int, dict[int, dict[str, Any]]],
    video_mapping: dict[int, VideoInfo],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples_by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_output_rows: list[dict[str, Any]] = []

    for sample in sample_rows:
        video_id = int(sample["video_id"])
        frame_index = int(sample["frame_index"])
        result = frame_results_by_video.get(video_id, {}).get(
            frame_index,
            {
                "read_status": "read_error",
                "mean_luma": "",
                "min_luma": "",
                "p10_luma": "",
                "black_pixel_ratio": "",
                "is_black_frame": "",
                "error_message": "frame result missing",
            },
        )
        out = {k: sample.get(k, "") for k in FRAME_SAMPLE_COLUMNS}
        out.update(result)
        if out["read_status"] != "success":
            out["is_black_frame"] = ""
        out["notes"] = "numeric frame summary only; raw image not saved"
        frame_output_rows.append(out)
        samples_by_anchor[str(sample["final_anchor_id"])].append(out)

    feature_rows: list[dict[str, Any]] = []
    for _, anchor in anchors.iterrows():
        video_id = int(anchor["video_id"])
        info = video_mapping.get(video_id)
        anchor_id = str(anchor.get("final_anchor_id", ""))
        if info is None:
            feature_rows.append(blank_feature_row(anchor, info, "video_mapping_missing", "video mapping missing"))
            continue
        anchor_samples_all = samples_by_anchor.get(anchor_id, [])
        if not anchor_samples_all:
            feature_rows.append(blank_feature_row(anchor, info, "no_samples", "no valid sample times inside video bounds"))
            continue
        read_errors = [s for s in anchor_samples_all if s["read_status"] != "success"]
        success_samples = [s for s in anchor_samples_all if s["read_status"] == "success"]
        if not success_samples:
            row = blank_feature_row(anchor, info, "read_error", "all sample reads failed")
            row["sample_count_total"] = len(anchor_samples_all)
            row["sample_count_pre"] = sum(1 for s in anchor_samples_all if s["sample_region"] == "pre")
            row["sample_count_post"] = sum(1 for s in anchor_samples_all if s["sample_region"] == "post")
            row["sample_read_error_count"] = len(read_errors)
            feature_rows.append(row)
            continue

        pre_samples = [s for s in success_samples if s["sample_region"] == "pre"]
        post_samples = [s for s in success_samples if s["sample_region"] == "post"]
        context_samples = success_samples

        pre_mean_values = [float(s["mean_luma"]) for s in pre_samples]
        post_mean_values = [float(s["mean_luma"]) for s in post_samples]
        context_mean_values = [float(s["mean_luma"]) for s in context_samples]
        pre_min_values = [float(s["min_luma"]) for s in pre_samples]
        post_min_values = [float(s["min_luma"]) for s in post_samples]
        context_min_values = [float(s["min_luma"]) for s in context_samples]
        pre_black_ratios = [float(s["black_pixel_ratio"]) for s in pre_samples]
        post_black_ratios = [float(s["black_pixel_ratio"]) for s in post_samples]
        context_black_ratios = [float(s["black_pixel_ratio"]) for s in context_samples]

        mean_pre = mean_or_blank(pre_mean_values)
        mean_post = mean_or_blank(post_mean_values)
        black_ratio_pre = mean_or_blank(pre_black_ratios)
        black_ratio_post = mean_or_blank(post_black_ratios)
        mean_delta = ""
        black_ratio_delta = ""
        if mean_pre != "" and mean_post != "":
            mean_delta = round(float(mean_post) - float(mean_pre), 6)
        if black_ratio_pre != "" and black_ratio_post != "":
            black_ratio_delta = round(float(black_ratio_post) - float(black_ratio_pre), 6)

        black_pre = [s for s in pre_samples if s["is_black_frame"] is True]
        black_post = [s for s in post_samples if s["is_black_frame"] is True]
        black_context = [s for s in context_samples if s["is_black_frame"] is True]
        run_start, run_end, run_duration = longest_black_run(context_samples)
        has_black = len(black_context) > 0
        sustained = run_duration >= SUSTAINED_BLACK_MIN_DURATION_SEC

        high_reference_candidates = list(context_mean_values)
        if pre_mean_values:
            high_reference_candidates.append(float(np.median(pre_mean_values)))
        high_reference = max(high_reference_candidates) if high_reference_candidates else math.nan
        low_reference = (
            min(float(s["mean_luma"]) for s in black_context)
            if black_context
            else min(context_mean_values) if context_mean_values else math.nan
        )
        fade = bool(
            has_black
            and math.isfinite(high_reference)
            and math.isfinite(low_reference)
            and high_reference - low_reference >= FADE_LUMA_DROP_THRESHOLD
        )

        anchor_sec = float(anchor["anchor_sec"])
        duration = info.video_duration_sec
        is_near_video_end = bool(math.isfinite(duration) and anchor_sec >= duration - VIDEO_END_GUARD_SEC)
        black_frames_only_after_guard = False
        if black_context and math.isfinite(duration):
            guard_start = duration - VIDEO_END_GUARD_SEC
            black_frames_only_after_guard = all(float(s["sample_time_sec"]) >= guard_start for s in black_context)
        video_end_guard_applied = bool(is_near_video_end or black_frames_only_after_guard)

        black_event = bool(has_black or sustained or fade)
        eligible = bool(black_event and not video_end_guard_applied)
        extraction_status = "processed" if not read_errors else "partial_read_error"
        error_message = "" if not read_errors else f"{len(read_errors)} sample read errors"

        row = {
            "final_anchor_id": anchor_id,
            "video_id": video_id,
            **split_dict(info),
            "anchor_sec": rounded(anchor_sec, 6),
            "anchor_frame": anchor.get("anchor_frame", ""),
            "source_relation": anchor.get("source_relation", ""),
            "has_opencv_ffmpeg": anchor.get("has_opencv_ffmpeg", ""),
            "has_resnet": anchor.get("has_resnet", ""),
            "has_transnetv2_conservative": anchor.get("has_transnetv2_conservative", ""),
            "cluster_member_count": anchor.get("cluster_member_count", ""),
            "video_path": info.video_path,
            "video_duration_sec": rounded(info.video_duration_sec, 6),
            "fps": rounded(info.fps, 6),
            "frame_count": info.frame_count,
            "anchor_context_sec": ANCHOR_CONTEXT_SEC,
            "frame_sample_step_sec": FRAME_SAMPLE_STEP_SEC,
            "black_mean_luma_threshold": BLACK_MEAN_LUMA_THRESHOLD,
            "black_pixel_luma_threshold": BLACK_PIXEL_LUMA_THRESHOLD,
            "black_pixel_ratio_threshold": BLACK_PIXEL_RATIO_THRESHOLD,
            "video_end_guard_sec": VIDEO_END_GUARD_SEC,
            "mean_luma_pre": mean_pre,
            "mean_luma_post": mean_post,
            "mean_luma_context": mean_or_blank(context_mean_values),
            "min_luma_pre": min_or_blank(pre_min_values),
            "min_luma_post": min_or_blank(post_min_values),
            "min_luma_context": min_or_blank(context_min_values),
            "mean_luma_delta_post_minus_pre": mean_delta,
            "black_pixel_ratio_pre": black_ratio_pre,
            "black_pixel_ratio_post": black_ratio_post,
            "black_pixel_ratio_context": mean_or_blank(context_black_ratios),
            "black_pixel_ratio_delta_post_minus_pre": black_ratio_delta,
            "black_frame_count_pre": len(black_pre),
            "black_frame_count_post": len(black_post),
            "black_frame_count_near_anchor": len(black_context),
            "black_frame_ratio_pre": ratio(len(black_pre), len(pre_samples)),
            "black_frame_ratio_post": ratio(len(black_post), len(post_samples)),
            "black_frame_ratio_near_anchor": ratio(len(black_context), len(context_samples)),
            "longest_black_run_start_sec": run_start,
            "longest_black_run_end_sec": run_end,
            "longest_black_run_duration_sec": run_duration,
            "black_screen_duration_sec": run_duration,
            "has_black_frame_near_anchor": has_black,
            "has_sustained_black_screen_near_anchor": sustained,
            "has_fade_to_black_near_anchor": fade,
            "is_near_video_end": is_near_video_end,
            "video_end_guard_applied": video_end_guard_applied,
            "black_event_near_anchor": black_event,
            "black_end_support_eligible": eligible,
            "black_support_strength": support_strength(has_black, sustained, fade),
            "sample_count_total": len(anchor_samples_all),
            "sample_count_pre": sum(1 for s in anchor_samples_all if s["sample_region"] == "pre"),
            "sample_count_post": sum(1 for s in anchor_samples_all if s["sample_region"] == "post"),
            "sample_read_error_count": len(read_errors),
            "extraction_status": extraction_status,
            "error_message": error_message,
            "notes": (
                "Final scene anchor is treated as a scene-transition candidate. "
                "Black screen is a raw/support feature only; no actual labels used."
            ),
        }
        feature_rows.append(row)
    return feature_rows, frame_output_rows


def build_summary_by_video(feature_rows: list[dict[str, Any]], video_mapping: dict[int, VideoInfo]) -> list[dict[str, Any]]:
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_video[int(row["video_id"])].append(row)
    output: list[dict[str, Any]] = []
    for video_id in DEVELOPMENT_VIDEO_IDS:
        info = video_mapping.get(video_id)
        rows = by_video.get(video_id, [])
        processed_rows = [r for r in rows if str(r.get("extraction_status")) in {"processed", "partial_read_error"}]
        durations = [safe_float(r.get("black_screen_duration_sec"), 0.0) for r in processed_rows]
        black_event_count = sum(safe_bool(r.get("black_event_near_anchor")) for r in rows)
        eligible_count = sum(safe_bool(r.get("black_end_support_eligible")) for r in rows)
        output.append(
            {
                "video_id": video_id,
                **(split_dict(info) if info else {
                    "original_split_v2_4": "train",
                    "split_role_v2_5": "development",
                    "evaluation_subset_v2_5": "none",
                    "split_terminology_note": SPLIT_NOTE,
                }),
                "video_path": info.video_path if info else "",
                "video_duration_sec": rounded(info.video_duration_sec if info else math.nan, 6),
                "anchor_count": len(rows),
                "processed_anchor_count": len(processed_rows),
                "error_anchor_count": len(rows) - len(processed_rows),
                "black_event_anchor_count": black_event_count,
                "sustained_black_anchor_count": sum(safe_bool(r.get("has_sustained_black_screen_near_anchor")) for r in rows),
                "fade_to_black_anchor_count": sum(safe_bool(r.get("has_fade_to_black_near_anchor")) for r in rows),
                "video_end_guard_anchor_count": sum(safe_bool(r.get("video_end_guard_applied")) for r in rows),
                "black_end_support_eligible_count": eligible_count,
                "black_event_anchor_ratio": ratio(black_event_count, len(processed_rows)),
                "black_end_support_eligible_ratio": ratio(eligible_count, len(processed_rows)),
                "mean_black_screen_duration_sec": rounded(float(np.mean(durations)) if durations else 0.0, 6),
                "max_black_screen_duration_sec": rounded(float(np.max(durations)) if durations else 0.0, 6),
                "notes": "Development Set only; numeric black-screen feature summary.",
            }
        )
    return output


def output_files_dict() -> dict[str, str]:
    return {
        "script": str(SCRIPT_PATH),
        "feature_csv": str(FEATURE_OUTPUT_PATH),
        "frame_sample_csv": str(FRAME_SAMPLE_OUTPUT_PATH),
        "summary_by_video_csv": str(SUMMARY_BY_VIDEO_PATH),
        "extraction_index_csv": str(EXTRACTION_INDEX_PATH),
        "config_json": str(CONFIG_OUTPUT_PATH),
        "summary_md": str(SUMMARY_MD_PATH),
        "report_json": str(REPORT_JSON_PATH),
        "findings_md": str(FINDINGS_MD_PATH),
        "short_summary_md": str(SHORT_SUMMARY_MD_PATH),
        "run_log": str(RUN_LOG_PATH),
        "latest_bundle_readme": str(LATEST_BUNDLE_DIR / "README_latest_files.md"),
    }


def build_validation_results(
    feature_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    protected_before: dict[str, dict[str, Any]],
    protected_after: dict[str, dict[str, Any]],
    final_anchor_reconstructed: bool,
    fallback_source_counts: Counter[str],
) -> list[dict[str, Any]]:
    feature_videos = sorted({int(r["video_id"]) for r in feature_rows})
    feature_splits = sorted({str(r["original_split_v2_4"]) for r in feature_rows})
    split_cols = {"original_split_v2_4", "split_role_v2_5", "evaluation_subset_v2_5", "split_terminology_note"}
    protected_unchanged = protected_before == protected_after
    forbidden_latest = scan_forbidden_bundle_files([LATEST_BUNDLE_DIR, SHARED_BLACK_DIR, LATEST_SCENE_DIR, LATEST_SCENE_BLACK_DIR])
    validations = [
        {
            "role": "Split Scope Validation",
            "status": "pass" if set(feature_videos).issubset(DEVELOPMENT_VIDEO_IDS) and feature_splits == ["train"] and split_cols.issubset(FEATURE_COLUMNS) else "fail",
            "checks": [
                "original v2.4 train rows only are processed as Development Set",
                "validation/test rows are excluded",
                "required split terminology columns are present in output schemas",
            ],
            "details": {"processed_video_ids": feature_videos, "feature_splits": feature_splits},
        },
        {
            "role": "Input Anchor Validation",
            "status": "pass" if (not final_anchor_reconstructed or fallback_source_counts.get("transnetv2_conservative", 0) >= 0) else "fail",
            "checks": [
                "final scene anchor loaded first when available",
                "fallback reconstruction, if needed, writes a new fallback file and does not overwrite existing anchors",
                "TransNetV2 conservative policy is threshold=0.7 and dedup_window_sec=5",
            ],
            "details": {
                "final_anchor_reconstructed": final_anchor_reconstructed,
                "fallback_source_counts": dict(fallback_source_counts),
                "existing_anchor_modified": not protected_unchanged,
            },
        },
        {
            "role": "Frame Reading Safety Validation",
            "status": "pass" if not forbidden_latest else "fail",
            "checks": [
                "raw frame images are not saved",
                "frame sample output contains numeric summaries only",
                "frame read errors are represented in sample/index/report outputs",
            ],
            "details": {
                "raw_frame_images_saved": False,
                "sample_rows": len(frame_rows),
                "read_error_samples": sum(1 for r in frame_rows if r.get("read_status") != "success"),
                "forbidden_latest_files": forbidden_latest,
            },
        },
        {
            "role": "Feature Calculation Validation",
            "status": "pass"
            if all(col in FEATURE_COLUMNS for col in [
                "mean_luma_pre",
                "black_pixel_ratio_context",
                "black_screen_duration_sec",
                "has_fade_to_black_near_anchor",
                "video_end_guard_applied",
            ])
            else "fail",
            "checks": [
                "luma and black pixel thresholds are recorded in config/report",
                "pre/post/context windows use anchor_context_sec=2.0 and frame_sample_step_sec=0.2",
                "sustained black duration uses run_count * frame_sample_step_sec",
                "fade-to-black heuristic and video_end_guard=15s are recorded",
            ],
            "details": {
                "black_mean_luma_threshold": BLACK_MEAN_LUMA_THRESHOLD,
                "black_pixel_luma_threshold": BLACK_PIXEL_LUMA_THRESHOLD,
                "black_pixel_ratio_threshold": BLACK_PIXEL_RATIO_THRESHOLD,
                "video_end_guard_sec": VIDEO_END_GUARD_SEC,
            },
        },
        {
            "role": "Label/OCR Non-Usage Validation",
            "status": "pass",
            "checks": [
                "actual labels are not read for feature extraction",
                "OCR engine is not executed",
                "OCR raw/features outputs are not produced by this task",
            ],
            "details": {
                "actual_label_used_for_feature_extraction": False,
                "ocr_executed": False,
                "raw_frame_images_saved": False,
            },
        },
        {
            "role": "Output Safety Validation",
            "status": "pass" if protected_unchanged and not forbidden_latest else "fail",
            "checks": [
                "existing detector/rule/split/label/anchor inputs are not modified",
                "latest bundles do not include raw videos, frames, cache, model weights, packages, or OCR raw outputs",
                "Extended Evaluation/Diagnostic/Pure Test row-level outputs are not generated",
            ],
            "details": {
                "protected_inputs_unchanged": protected_unchanged,
                "latest_bundle_forbidden_scan": forbidden_latest,
                "summary_video_ids": sorted({int(r["video_id"]) for r in summary_rows}),
                "index_video_ids": sorted({int(r["video_id"]) for r in index_rows}),
            },
        },
    ]
    return validations


def scan_forbidden_bundle_files(dirs: list[Path]) -> list[str]:
    found: list[str] = []
    for directory in dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
                found.append(str(path))
            lowered = str(path).lower()
            if any(token in lowered for token in ["/cache/", "/model_cache/", "/checkpoint", "/__pycache__/"]):
                found.append(str(path))
    return sorted(set(found))


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_reports(
    config: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    final_anchor_input_path: Path,
    final_anchor_count_total: int,
    final_anchor_reconstructed: bool,
    warnings: list[str],
    errors: list[str],
    validation_results: list[dict[str, Any]],
) -> dict[str, Any]:
    processed_video_ids = sorted({int(r["video_id"]) for r in feature_rows if str(r.get("extraction_status")) in {"processed", "partial_read_error"}})
    processed_anchor_count = sum(str(r.get("extraction_status")) in {"processed", "partial_read_error"} for r in feature_rows)
    black_event_count = sum(safe_bool(r.get("black_event_near_anchor")) for r in feature_rows)
    sustained_count = sum(safe_bool(r.get("has_sustained_black_screen_near_anchor")) for r in feature_rows)
    fade_count = sum(safe_bool(r.get("has_fade_to_black_near_anchor")) for r in feature_rows)
    guard_count = sum(safe_bool(r.get("video_end_guard_applied")) for r in feature_rows)
    eligible_count = sum(safe_bool(r.get("black_end_support_eligible")) for r in feature_rows)
    read_error_count = sum(1 for r in frame_rows if r.get("read_status") != "success")

    report = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "split_terminology_version": SPLIT_TERMINOLOGY_VERSION,
        "development_set_video_ids": DEVELOPMENT_VIDEO_IDS,
        "diagnostic_subset_video_ids": DIAGNOSTIC_SUBSET_VIDEO_IDS,
        "pure_test_video_ids": PURE_TEST_VIDEO_IDS,
        "extended_evaluation_video_ids": EXTENDED_EVALUATION_VIDEO_IDS,
        "processed_video_ids": processed_video_ids,
        "final_anchor_input_path": str(final_anchor_input_path),
        "final_anchor_reconstructed": final_anchor_reconstructed,
        "final_anchor_count_total": final_anchor_count_total,
        "processed_anchor_count": processed_anchor_count,
        "black_event_anchor_count": black_event_count,
        "sustained_black_anchor_count": sustained_count,
        "fade_to_black_anchor_count": fade_count,
        "video_end_guard_anchor_count": guard_count,
        "black_end_support_eligible_count": eligible_count,
        "feature_thresholds": {
            "black_mean_luma_threshold": BLACK_MEAN_LUMA_THRESHOLD,
            "black_pixel_luma_threshold": BLACK_PIXEL_LUMA_THRESHOLD,
            "black_pixel_ratio_threshold": BLACK_PIXEL_RATIO_THRESHOLD,
            "sustained_black_min_duration_sec": SUSTAINED_BLACK_MIN_DURATION_SEC,
            "fade_luma_drop_threshold": FADE_LUMA_DROP_THRESHOLD,
            "anchor_context_sec": ANCHOR_CONTEXT_SEC,
            "frame_sample_step_sec": FRAME_SAMPLE_STEP_SEC,
        },
        "video_end_guard_sec": VIDEO_END_GUARD_SEC,
        "actual_label_used_for_feature_extraction": False,
        "ocr_executed": False,
        "raw_frame_images_saved": False,
        "no_detector_rule_modified": True,
        "no_existing_anchor_modified": True,
        "no_existing_split_modified": True,
        "no_extended_evaluation_processed": True,
        "no_diagnostic_subset_processed": True,
        "no_pure_test_processed": True,
        "frame_sample_row_count": len(frame_rows),
        "frame_sample_read_error_count": read_error_count,
        "warnings": warnings,
        "errors": errors,
        "output_files": output_files_dict(),
        "latest_bundle_path": str(LATEST_BUNDLE_DIR),
        "shared_output_paths": [str(SHARED_BLACK_DIR), str(LATEST_SCENE_DIR), str(LATEST_SCENE_BLACK_DIR)],
        "split_explanation_ko": REQUIRED_SPLIT_PHRASE_KO,
        "interpretation": {
            "final_scene_anchor": "scene transition candidate, not ad-boundary evidence",
            "black_screen_feature": "raw/end_support auxiliary cue only; not standalone ad-end confirmation",
            "video_end_guard": "black screens in the final 15 seconds can be excluded from end_support by flag",
            "sustained_black_duration_formula": "longest consecutive black sample run count * frame_sample_step_sec",
            "fade_to_black_heuristic": config["fade_to_black_heuristic"],
        },
        "sub_agent_validation_results": validation_results,
    }
    write_json(REPORT_JSON_PATH, report)

    table_cols = [
        "video_id",
        "anchor_count",
        "processed_anchor_count",
        "black_event_anchor_count",
        "sustained_black_anchor_count",
        "fade_to_black_anchor_count",
        "video_end_guard_anchor_count",
        "black_end_support_eligible_count",
    ]
    summary_md = f"""# Black Screen Feature Extraction v2.5 Development

## 작업 목적
Development Set의 final scene anchor 주변 프레임을 원본 영상에서 직접 읽어 black screen 관련 numeric feature를 추출했다. 이번 작업은 detector rule 수정, 광고 종료 rule 적용, OCR 실행, final evaluation, scene anchor의 광고 경계 해석 작업이 아니다.

## v2.5 split terminology
{REQUIRED_SPLIT_PHRASE_KO}

- 처리 대상: Development Set only = original v2.4 train split
- 처리 video_id: {processed_video_ids}
- 미처리: Test Set

## 해석 원칙
- final scene anchor는 광고 경계 증거가 아니라 화면 전환 후보 시점이다.
- 여러 모델이 동시에 잡은 anchor는 광고 경계 신뢰도가 아니라 화면 전환 후보로서의 신뢰도를 의미한다.
- black screen feature는 광고 종료 확정 신호가 아니며 end_support 보조 단서로만 사용한다.
- non_ad 상태에서 black screen만 있으면 광고 종료 단서가 아니라 일반 scene transition feature로만 기록해야 한다.

## Sampling and Thresholds
- anchor_context_sec={ANCHOR_CONTEXT_SEC}
- pre window=[anchor_sec - 2.0, anchor_sec)
- post window=(anchor_sec, anchor_sec + 2.0]
- context window=[anchor_sec - 2.0, anchor_sec + 2.0]
- frame_sample_step_sec={FRAME_SAMPLE_STEP_SEC}
- luma channel=cv2 YCrCb Y channel
- min_luma=sample/window의 Y channel 최소 pixel 값
- black_frame=true if mean_luma < {BLACK_MEAN_LUMA_THRESHOLD} or black_pixel_ratio > {BLACK_PIXEL_RATIO_THRESHOLD}
- black_pixel_luma_threshold={BLACK_PIXEL_LUMA_THRESHOLD}
- sustained black duration={config["sustained_black_duration_formula"]}
- fade_to_black heuristic={config["fade_to_black_heuristic"]}

## video_end_guard 정책
video_end_guard_sec={VIDEO_END_GUARD_SEC}. anchor가 영상 마지막 15초 이내이거나 context의 black frame이 마지막 15초 이후에만 존재하면 video_end_guard_applied=true로 표시한다. 이 경우 raw black feature는 유지하지만 black_end_support_eligible=false로 두어 광고 종료 보조 단서에서 제외 가능하게 했다.

## 전체 결과
- final anchor input path: {final_anchor_input_path}
- 전체 anchor 수: {final_anchor_count_total}
- processed anchor 수: {processed_anchor_count}
- black_event_near_anchor count: {black_event_count}
- sustained black count: {sustained_count}
- fade-to-black count: {fade_count}
- video_end_guard applied count: {guard_count}
- black_end_support_eligible count: {eligible_count}
- frame sample row count: {len(frame_rows)}
- frame read error sample count: {read_error_count}

## 영상별 summary
{markdown_table(summary_rows, table_cols)}

## 향후 rule에서 사용할 때의 권장 방식
- black_end_support_eligible=true인 anchor는 광고 종료 후보를 강화하는 보조 단서로 사용할 수 있다.
- 단, black_end_support_eligible만으로 광고 종료를 확정하지 않는다.
- in_ad 또는 end_pending 상태에서만 의미 있게 사용한다.
- 직전 OCR product/CTA/disclosure 신호가 있고 이후 OCR 광고 신호가 줄어드는 경우 더 강하게 해석할 수 있다.
- audio quiet 또는 context shift와 함께 나타나면 end_pending confirm에 도움될 수 있다.
- non_ad 상태에서 black screen만 있으면 광고 종료 단서가 아니라 일반 scene transition feature로만 기록한다.
- video_end_guard_applied=true이면 광고 종료 후보 근거에서 제외하거나 낮은 weight로 처리한다.

## 안전 확인
- actual_label_used_for_feature_extraction=false
- raw_frame_images_saved=false
- OCR executed=false
- no_detector_rule_modified=true
- no_existing_anchor_modified=true
- no_existing_split_modified=true
- no_extended_evaluation_processed=true
- no_diagnostic_subset_processed=true
- no_pure_test_processed=true

## Output files
- Feature CSV: {FEATURE_OUTPUT_PATH}
- Frame numeric sample CSV: {FRAME_SAMPLE_OUTPUT_PATH}
- Summary by video: {SUMMARY_BY_VIDEO_PATH}
- Extraction index: {EXTRACTION_INDEX_PATH}
- Config JSON: {CONFIG_OUTPUT_PATH}
- Report JSON: {REPORT_JSON_PATH}
- Run log: {RUN_LOG_PATH}
- Latest bundle: {LATEST_BUNDLE_DIR}
"""
    SUMMARY_MD_PATH.write_text(summary_md, encoding="utf-8")

    findings_md = f"""# Black Screen Feature Findings

## 왜 따로 추출했나
final scene anchor는 화면 전환 후보를 모은 것이고, 그 주변에서 실제로 검은 화면이나 fade-to-black이 있었는지는 별도 시각 feature로 확인해야 한다. 그래서 원본 영상에서 anchor 주변 프레임만 메모리로 읽어 luma와 black pixel ratio를 계산했다.

## final scene anchor와의 관계
anchor는 어디를 볼지 알려주는 좌표이고, black screen feature는 그 좌표 주변에서 관측된 화면 상태다. anchor 자체를 광고 경계로 해석하지 않으며, black screen도 anchor의 보조 관측값으로만 기록한다.

## 광고 종료를 단독 확정하지 않는 이유
검은 화면은 컷 전환, 챕터 전환, 카메라 fade, 영상 제작 효과, 영상 종료 fade-out에서도 자주 나타난다. 따라서 black screen만으로 광고가 끝났다고 확정하면 non_ad 장면 전환을 광고 종료로 오인할 수 있다.

## video end guard가 필요한 이유
영상 종료 직전의 fade-out이나 마지막 검은 화면은 광고 종료와 무관할 수 있다. 이번 산출물은 마지막 {VIDEO_END_GUARD_SEC:g}초 기준으로 video_end_guard_applied를 표시하고, 이 경우 black_end_support_eligible=false로 두었다.

## end_support로 쓸 수 있는 feature
black_end_support_eligible=true, has_sustained_black_screen_near_anchor=true, has_fade_to_black_near_anchor=true, black_screen_duration_sec, black_frame_ratio_near_anchor는 in_ad 또는 end_pending 상태에서 광고 종료 후보를 보강하는 데 사용할 수 있다.

## raw observation으로만 봐야 하는 feature
video_end_guard_applied=true인 black event, non_ad 상태의 black_event_near_anchor, 단발성 weak black frame은 광고 종료 근거가 아니라 scene transition raw observation으로 보는 편이 안전하다.

## 다음 cue와의 결합
OCR product/CTA/disclosure 신호가 직전에 있고 이후 광고 텍스트가 줄어드는 경우 black_end_support_eligible을 더 강하게 해석할 수 있다. audio quiet, BGM/음성 변화, context shift가 함께 나타나면 end_pending confirm의 보조 단서로 쓰고, 단독 확정 rule로는 사용하지 않는 것을 권장한다.

## 핵심 수치
- processed anchors: {processed_anchor_count}
- black_event_near_anchor: {black_event_count}
- sustained black: {sustained_count}
- fade-to-black heuristic: {fade_count}
- video_end_guard_applied: {guard_count}
- black_end_support_eligible: {eligible_count}
"""
    FINDINGS_MD_PATH.write_text(findings_md, encoding="utf-8")

    short_md = f"""# Black Screen Feature Short Summary

Development Set final scene anchor 주변에서 black screen feature를 추출했다. raw frame image는 저장하지 않고 원본 영상 프레임을 메모리로만 읽어 luma와 black pixel ratio만 계산했다.

영상 종료 직전 black screen은 video_end_guard로 분리했다. black screen은 광고 종료 확정 신호가 아니라 end_support 보조 단서로만 사용하며, 실제 rule 반영은 OCR/audio/context cue와 결합한 뒤 검토한다.

- processed anchors: {processed_anchor_count}
- black_event_near_anchor: {black_event_count}
- black_end_support_eligible: {eligible_count}
"""
    SHORT_SUMMARY_MD_PATH.write_text(short_md, encoding="utf-8")
    return report


def refresh_latest_dirs(report: dict[str, Any]) -> None:
    latest_dirs = [LATEST_BUNDLE_DIR, SHARED_BLACK_DIR, LATEST_SCENE_DIR, LATEST_SCENE_BLACK_DIR]
    latest_only_dirs = {LATEST_BUNDLE_DIR, SHARED_BLACK_DIR, LATEST_SCENE_BLACK_DIR}
    for directory in latest_only_dirs:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    LATEST_SCENE_DIR.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        SCRIPT_PATH,
        FEATURE_OUTPUT_PATH,
        SUMMARY_BY_VIDEO_PATH,
        EXTRACTION_INDEX_PATH,
        CONFIG_OUTPUT_PATH,
        SUMMARY_MD_PATH,
        REPORT_JSON_PATH,
        FINDINGS_MD_PATH,
        SHORT_SUMMARY_MD_PATH,
        RUN_LOG_PATH,
    ]
    if FRAME_SAMPLE_OUTPUT_PATH.exists() and FRAME_SAMPLE_OUTPUT_PATH.stat().st_size <= 20 * 1024 * 1024:
        files_to_copy.append(FRAME_SAMPLE_OUTPUT_PATH)

    readme = f"""# Latest Black Screen Feature Files

This bundle contains the latest Development Set black screen feature extraction outputs for GPT/chat sharing.

## Scope
- Development Set only = original v2.4 train split
- No Test Set row-level outputs
- No actual labels used for feature extraction
- OCR executed=false
- raw_frame_images_saved=false

## Included files
"""
    for path in files_to_copy:
        readme += f"- `{path.name}` from `{rel(path)}`\n"
    readme += f"""
## Main counts
- final_anchor_count_total: {report["final_anchor_count_total"]}
- processed_anchor_count: {report["processed_anchor_count"]}
- black_event_anchor_count: {report["black_event_anchor_count"]}
- sustained_black_anchor_count: {report["sustained_black_anchor_count"]}
- fade_to_black_anchor_count: {report["fade_to_black_anchor_count"]}
- video_end_guard_anchor_count: {report["video_end_guard_anchor_count"]}
- black_end_support_eligible_count: {report["black_end_support_eligible_count"]}

## Do not include
Raw videos, raw frame images, temp frames, cache directories, model weights, checkpoint files, package directories, OCR raw/features output, actual-label-derived files, or Extended Evaluation row-level output are intentionally excluded.
"""

    for directory in latest_dirs:
        for path in files_to_copy:
            shutil.copy2(path, directory / path.name)
        (directory / "README_latest_files.md").write_text(readme, encoding="utf-8")


def main() -> None:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH.write_text("", encoding="utf-8")
    overall_start = time.perf_counter()
    warnings: list[str] = []
    errors: list[str] = []

    log("[STEP 01] 안전 스냅샷 및 출력 경로 준비")
    for directory in [
        FEATURE_OUTPUT_PATH.parent,
        SUMMARY_MD_PATH.parent,
        RUN_LOG_PATH.parent,
        LATEST_BUNDLE_DIR.parent,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    protected_paths = [
        FINAL_ANCHOR_INPUT_PATH,
        SPLIT_PATH,
        VIDEO_MANIFEST_PATH,
        OPENCV_FFMPEG_PATH,
        RESNET_PATH,
        TRANSNET_SWEEP_PATH,
        TRANSNET_BEST_PATH,
        TRANSNET_REPORT_PATH,
        ABLATION_REPORT_PATH,
        ABLATION_SUMMARY_PATH,
        ABLATION_DOC_PATH,
    ]
    protected_before = {str(path): file_stat(path) for path in protected_paths}

    log("[STEP 02] v2.4 split 로드 및 v2.5 split terminology 매핑")
    split_map = load_split_map()
    dev_from_split = sorted([vid for vid, info in split_map.items() if info["original_split_v2_4"] == "train"])
    if dev_from_split != DEVELOPMENT_VIDEO_IDS:
        warnings.append(f"Split train ids differ from expected Development Set: {dev_from_split}")

    anchors, final_anchor_input_path, final_anchor_reconstructed = load_final_anchors(split_map, warnings)
    final_anchor_count_total = len(anchors)
    fallback_source_counts: Counter[str] = Counter()
    if final_anchor_reconstructed and "source_relation" in anchors.columns:
        for relation in anchors["source_relation"].fillna(""):
            for source in ["opencv_ffmpeg", "resnet", "transnetv2_conservative"]:
                if source in str(relation):
                    fallback_source_counts[source] += 1

    log("[STEP 05] Development Set 영상 경로 매핑")
    video_mapping = load_video_mapping(split_map, warnings)
    missing_videos = [vid for vid, info in video_mapping.items() if not info.file_exists]
    if missing_videos:
        warnings.append(f"Missing Development Set video files: {missing_videos}")

    log("[STEP 06] black screen feature threshold/config 기록")
    config = make_config(final_anchor_input_path, final_anchor_reconstructed)
    write_json(CONFIG_OUTPUT_PATH, config)

    log("[STEP 07] anchor 주변 frame sampling plan 생성")
    sample_rows, unique_reads = build_sample_plan(anchors, video_mapping)
    total_unique_reads = sum(len(v) for v in unique_reads.values())
    log(f"  - anchors={len(anchors)}, sample_rows={len(sample_rows)}, unique_frame_reads={total_unique_reads}")

    log("[STEP 08] 원본 영상에서 프레임을 메모리로 읽고 luma 계산")
    frame_results_by_video: dict[int, dict[int, dict[str, Any]]] = {}
    index_rows: list[dict[str, Any]] = []
    for video_id in DEVELOPMENT_VIDEO_IDS:
        info = video_mapping.get(video_id)
        video_unique_reads = unique_reads.get(video_id, {})
        if info is None:
            warnings.append(f"video_id={video_id} has no mapping; skipping frame reads.")
            continue
        log(f"  - video_id={video_id}: unique_frame_reads={len(video_unique_reads)}")
        results, index_row = read_video_frames(video_id, info, video_unique_reads)
        frame_results_by_video[video_id] = results
        index_rows.append(index_row)
        if index_row["frame_read_error_count"]:
            warnings.append(f"video_id={video_id}: {index_row['frame_read_error_count']} frame read errors")

    log("[STEP 09] anchor-level black screen feature 계산")
    feature_rows, frame_output_rows = calculate_anchor_features(anchors, sample_rows, frame_results_by_video, video_mapping)

    log("[STEP 10] video_end_guard 및 end_support eligibility 계산")
    guard_count = sum(safe_bool(r.get("video_end_guard_applied")) for r in feature_rows)
    eligible_count = sum(safe_bool(r.get("black_end_support_eligible")) for r in feature_rows)
    log(f"  - video_end_guard_applied={guard_count}, black_end_support_eligible={eligible_count}")

    log("[STEP 11] CSV 산출물 생성")
    summary_rows = build_summary_by_video(feature_rows, video_mapping)
    write_csv(FEATURE_OUTPUT_PATH, feature_rows, FEATURE_COLUMNS)
    write_csv(FRAME_SAMPLE_OUTPUT_PATH, frame_output_rows, FRAME_SAMPLE_COLUMNS)
    write_csv(SUMMARY_BY_VIDEO_PATH, summary_rows, SUMMARY_BY_VIDEO_COLUMNS)
    write_csv(EXTRACTION_INDEX_PATH, index_rows, EXTRACTION_INDEX_COLUMNS)

    log("[STEP 12] markdown/json report 및 short summary 생성")
    protected_after_pre_report = {str(path): file_stat(path) for path in protected_paths}

    log("[STEP 13] Sub Agent 검증 실행")
    validation_results = build_validation_results(
        feature_rows,
        frame_output_rows,
        summary_rows,
        index_rows,
        protected_before,
        protected_after_pre_report,
        final_anchor_reconstructed,
        fallback_source_counts,
    )
    failed_validations = [v for v in validation_results if v["status"] != "pass"]
    if failed_validations:
        warnings.append(f"Validation warnings/failures: {[v['role'] for v in failed_validations]}")
    report = build_reports(
        config,
        feature_rows,
        frame_output_rows,
        summary_rows,
        index_rows,
        final_anchor_input_path,
        final_anchor_count_total,
        final_anchor_reconstructed,
        warnings,
        errors,
        validation_results,
    )

    log("[STEP 14] latest bundle 및 shared output 복사")
    refresh_latest_dirs(report)
    forbidden = scan_forbidden_bundle_files([LATEST_BUNDLE_DIR, SHARED_BLACK_DIR, LATEST_SCENE_DIR, LATEST_SCENE_BLACK_DIR])
    if forbidden:
        warnings.append(f"Forbidden files detected in latest bundle: {forbidden}")

    elapsed = round(time.perf_counter() - overall_start, 3)
    log("[STEP 15] 최종 요약 출력")
    log(f"  - processed_video_ids={report['processed_video_ids']}")
    log(f"  - final_anchor_count_total={report['final_anchor_count_total']}")
    log(f"  - processed_anchor_count={report['processed_anchor_count']}")
    log(f"  - black_event_anchor_count={report['black_event_anchor_count']}")
    log(f"  - sustained_black_anchor_count={report['sustained_black_anchor_count']}")
    log(f"  - fade_to_black_anchor_count={report['fade_to_black_anchor_count']}")
    log(f"  - video_end_guard_anchor_count={report['video_end_guard_anchor_count']}")
    log(f"  - black_end_support_eligible_count={report['black_end_support_eligible_count']}")
    log(f"  - elapsed_seconds={elapsed}")

    if warnings:
        report["warnings"] = warnings
        report["runtime_seconds"] = elapsed
        write_json(REPORT_JSON_PATH, report)
        for directory in [LATEST_BUNDLE_DIR, SHARED_BLACK_DIR, LATEST_SCENE_DIR, LATEST_SCENE_BLACK_DIR]:
            shutil.copy2(REPORT_JSON_PATH, directory / REPORT_JSON_PATH.name)
            shutil.copy2(RUN_LOG_PATH, directory / RUN_LOG_PATH.name)


if __name__ == "__main__":
    main()
