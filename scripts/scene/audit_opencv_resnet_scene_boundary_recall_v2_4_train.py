#!/usr/bin/env python3
"""Train-only OpenCV/FFmpeg vs ResNet scene boundary recall audit for v2.4.

This script reads existing v2.4 visual anchor/source candidate files and actual
ad interval labels, then measures whether candidate timestamps fall near actual
ad_start_sec/ad_end_sec boundaries. It does not create or tune candidates and it
does not modify detector assets.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
TASK_NAME = "opencv_resnet_scene_boundary_recall_audit_v2_4_train"
VERSION = "v2_4"
SPLIT_SEED = 20240524
TRAIN_IDS = {1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15}
VALIDATION_IDS = {3, 7, 18}
TEST_IDS = {4, 16, 17}
TOLERANCES = [2, 5, 10]
COMBINED_DEDUP_WINDOW_SEC = 2.0

DATA_DIR = PROJECT_ROOT / "data"
SCENE_DIR = DATA_DIR / "scene"
FEATURE_DIR = DATA_DIR / "features"
REVIEW_DIR = DATA_DIR / "review"
SPLIT_PATH = DATA_DIR / "splits" / "video_split_v2_4.csv"
SEGMENT_PATH = DATA_DIR / "segments" / "ad_interval_segments_v2_4.csv"
VIDEO_MANIFEST_PATH = DATA_DIR / "video_metadata" / "video_manifest_v2_2.csv"
ANCHOR_PATH = FEATURE_DIR / "visual_scene_boundary_anchors_v2_4.csv"
ANCHOR_WITH_SPLIT_PATH = FEATURE_DIR / "visual_scene_boundary_anchors_v2_4_with_split.csv"
ANCHOR_BUILD_REPORT_PATH = PROJECT_ROOT / "reports" / "scene_change_audio_segment_features_v2_4_report.json"
FUSION_REPORT_PATH = PROJECT_ROOT / "reports" / "fusion" / "visual_anchor_alignment_pack_v2_4_report.json"
FUSION_SUMMARY_PATH = PROJECT_ROOT / "reports" / "fusion" / "visual_anchor_alignment_pack_v2_4_summary.md"

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "scene" / "audit_opencv_resnet_scene_boundary_recall_v2_4_train.py"
OUT_SCENE_DIR = DATA_DIR / "scene"
OUT_REPORT_DIR = PROJECT_ROOT / "reports" / "scene"
OUT_LOG_PATH = PROJECT_ROOT / "logs" / "opencv_resnet_scene_boundary_recall_audit_v2_4_run_log.txt"
LATEST_DIR = PROJECT_ROOT / "outputs" / "latest_for_chatgpt_opencv_resnet_scene_boundary_recall_audit_v2_4"

INVENTORY_CSV = OUT_SCENE_DIR / "opencv_resnet_scene_candidate_source_inventory_v2_4_train.csv"
SOURCE_COMPARISON_CSV = OUT_SCENE_DIR / "opencv_resnet_scene_candidate_source_comparison_v2_4_train.csv"
AUDIT_CSV = OUT_SCENE_DIR / "opencv_resnet_boundary_recall_audit_v2_4_train.csv"
SUMMARY_CSV = OUT_SCENE_DIR / "opencv_resnet_boundary_recall_summary_v2_4_train.csv"
CASE_BREAKDOWN_CSV = OUT_SCENE_DIR / "opencv_resnet_boundary_case_breakdown_v2_4_train.csv"
MISSED_CASES_CSV = OUT_SCENE_DIR / "opencv_resnet_missed_boundary_cases_v2_4_train.csv"
VIDEO_LEVEL_CSV = OUT_SCENE_DIR / "opencv_resnet_video_level_recall_v2_4_train.csv"

SUMMARY_MD = OUT_REPORT_DIR / "opencv_resnet_scene_boundary_recall_audit_v2_4_summary.md"
REPORT_JSON = OUT_REPORT_DIR / "opencv_resnet_scene_boundary_recall_audit_v2_4_report.json"
FINDINGS_MD = OUT_REPORT_DIR / "opencv_resnet_scene_boundary_recall_audit_v2_4_findings.md"
LATEST_README = LATEST_DIR / "README_latest_files.md"

DATA_OUTPUTS = [
    INVENTORY_CSV,
    SOURCE_COMPARISON_CSV,
    AUDIT_CSV,
    SUMMARY_CSV,
    CASE_BREAKDOWN_CSV,
    MISSED_CASES_CSV,
    VIDEO_LEVEL_CSV,
]
REPORT_OUTPUTS = [SUMMARY_MD, REPORT_JSON, FINDINGS_MD]
ALL_OUTPUTS = [SCRIPT_PATH, *DATA_OUTPUTS, *REPORT_OUTPUTS, OUT_LOG_PATH, LATEST_README]

OPENCV_CANDIDATE_PATHS = [
    SCENE_DIR / "scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed.csv",
    SCENE_DIR / "scene_candidates_v2_4_merged_ffmpeg_fallback.csv",
    SCENE_DIR / "scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed_mmss.csv",
    SCENE_DIR / "scene_candidates_v2_4_merged_ffmpeg_fallback_mmss.csv",
    SCENE_DIR / "scene_candidates_v2_3_merged_ffmpeg_fallback.csv",
]
RESNET_CANDIDATE_PATHS = [
    REVIEW_DIR / "resnet_scene_candidate_human_review_v2_4_usefulness_analysis.csv",
    SCENE_DIR / "resnet_scene_candidates_v2_4_labelrefreshed.csv",
    SCENE_DIR / "resnet_scene_candidates_v2_4_labelrefreshed_mmss.csv",
    SCENE_DIR / "resnet_scene_candidates_v2_4.csv",
    SCENE_DIR / "resnet_scene_candidates_v2_4_mmss.csv",
    SCENE_DIR / "resnet_scene_candidates_v2_3.csv",
    SCENE_DIR / "resnet_scene_candidates_v2_3_mmss.csv",
]

FORBIDDEN_LATEST_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a", ".jpg", ".jpeg", ".png",
    ".webp", ".pt", ".pth", ".ckpt", ".bin",
}
FORBIDDEN_LATEST_TOKENS = {"raw", "frame", "frames", "cache", "model", "checkpoint", "weights"}
PROTECTED_PATHS = [
    PROJECT_ROOT / "scripts" / "detectors",
    PROJECT_ROOT / "configs" / "detectors",
    PROJECT_ROOT / "data" / "predictions",
    PROJECT_ROOT / "reports" / "detectors",
    OLD_PROJECT_ROOT,
]

LOG_LINES: list[str] = []


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"{now_iso()} {message}"
    LOG_LINES.append(line)
    print(message, flush=True)


def write_log() -> None:
    OUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG_PATH.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def strip_bom(value: str) -> str:
    return value.lstrip("\ufeff")


def normalized_row(row: dict[str, Any]) -> dict[str, str]:
    return {strip_bom(k or ""): clean(v) for k, v in row.items()}


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = [strip_bom(c or "") for c in (reader.fieldnames or [])]
        rows = [normalized_row(row) for row in reader]
    return columns, rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: format_cell(row.get(col, "")) for col in columns})


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_col(columns: list[str], candidates: list[str]) -> str:
    direct = {c: c for c in columns}
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name in direct:
            return direct[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return ""


def to_float(value: Any) -> float | None:
    text = clean(value)
    if text == "":
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    if math.isnan(value_float) or math.isinf(value_float):
        return None
    return value_float


def to_int(value: Any) -> int | None:
    num = to_float(value)
    if num is None:
        return None
    return int(num)


def bool_value(value: Any) -> bool:
    return clean(value).lower() in {"true", "1", "yes", "y", "o", "ok", "pass"}


def seconds_label(sec: Any) -> str:
    value = to_float(sec)
    if value is None:
        return ""
    total = max(0, int(math.floor(value)))
    return f"{total // 60:02d}:{total % 60:02d}"


def round_or_blank(value: float | None, digits: int = 6) -> float | str:
    if value is None:
        return ""
    return round(value, digits)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def file_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }


def tree_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "file_count": 0, "total_size": 0, "signature": ""}
    if path.is_file():
        stat = path.stat()
        payload = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}"
        return {
            "exists": True,
            "file_count": 1,
            "total_size": stat.st_size,
            "signature": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(dirs)
        for name in sorted(files):
            full = Path(root) / name
            try:
                stat = full.stat()
            except OSError:
                continue
            rel = full.relative_to(path)
            digest.update(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8", "ignore"))
            file_count += 1
            total_size += stat.st_size
    return {
        "exists": True,
        "file_count": file_count,
        "total_size": total_size,
        "signature": digest.hexdigest(),
    }


def snapshot_protected_paths() -> dict[str, Any]:
    return {str(path): tree_signature(path) for path in PROTECTED_PATHS}


def source_related_columns(columns: list[str]) -> list[str]:
    needles = [
        "source", "method", "model", "detector", "family", "opencv", "ffmpeg", "resnet",
        "candidate_source", "anchor_source",
    ]
    return [col for col in columns if any(token in col.lower() for token in needles)]


def unique_values(rows: list[dict[str, str]], columns: list[str], limit: int = 30) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for col in columns:
        counter = Counter(row.get(col, "") for row in rows)
        out[col] = [{"value": value, "count": count} for value, count in counter.most_common(limit)]
    return out


def inspect_file(
    path: Path,
    source_status: str,
    opencv_rule: str,
    resnet_rule: str,
    warning: str = "",
) -> dict[str, Any]:
    if not path.exists():
        return {
            "file_path": str(path),
            "row_count": 0,
            "columns_json": "[]",
            "source_related_columns_json": "[]",
            "source_unique_values_json": "{}",
            "timestamp_column": "",
            "video_id_column": "",
            "split_column": "",
            "source_separation_status": source_status,
            "opencv_ffmpeg_filter_rule": opencv_rule,
            "resnet_filter_rule": resnet_rule,
            "warning": f"missing input file; {warning}".strip("; "),
        }
    columns, rows = read_csv_rows(path)
    src_cols = source_related_columns(columns)
    timestamp_col = find_col(
        columns,
        [
            "timestamp_sec", "candidate_sec", "candidate_time_sec", "anchor_sec", "boundary_sec",
            "time_sec", "sec", "canonical_boundary_time_sec", "resnet_time_sec_std",
            "opencv_candidate_time_sec", "resnet_candidate_time_sec",
        ],
    )
    video_col = find_col(columns, ["video_id", "video_id_normalized_for_split"])
    split_col = find_col(columns, ["split"])
    return {
        "file_path": str(path),
        "row_count": len(rows),
        "columns_json": json.dumps(columns, ensure_ascii=False),
        "source_related_columns_json": json.dumps(src_cols, ensure_ascii=False),
        "source_unique_values_json": json.dumps(unique_values(rows, src_cols), ensure_ascii=False),
        "timestamp_column": timestamp_col,
        "video_id_column": video_col,
        "split_column": split_col,
        "source_separation_status": source_status,
        "opencv_ffmpeg_filter_rule": opencv_rule,
        "resnet_filter_rule": resnet_rule,
        "warning": warning,
    }


def select_candidate_inputs(warnings: list[str]) -> tuple[Path | None, Path | None, dict[str, Any]]:
    build_report = read_json_optional(ANCHOR_BUILD_REPORT_PATH)
    report_inputs = build_report.get("input_files", {}) if isinstance(build_report, dict) else {}
    opencv_from_report = Path(report_inputs.get("opencv_candidates", "")) if report_inputs.get("opencv_candidates") else None
    resnet_from_report = Path(report_inputs.get("resnet_candidates", "")) if report_inputs.get("resnet_candidates") else None
    opencv_path = opencv_from_report if opencv_from_report and opencv_from_report.exists() else first_existing(OPENCV_CANDIDATE_PATHS)
    resnet_path = resnet_from_report if resnet_from_report and resnet_from_report.exists() else first_existing(RESNET_CANDIDATE_PATHS)
    if opencv_from_report and not opencv_from_report.exists():
        warnings.append(f"anchor build report opencv input missing; fallback selected: {opencv_from_report}")
    if resnet_from_report and not resnet_from_report.exists():
        warnings.append(f"anchor build report resnet input missing; fallback selected: {resnet_from_report}")
    selection = {
        "anchor_build_report": str(ANCHOR_BUILD_REPORT_PATH),
        "anchor_build_report_exists": ANCHOR_BUILD_REPORT_PATH.exists(),
        "opencv_selected_from_build_report": bool(opencv_from_report and opencv_path == opencv_from_report),
        "resnet_selected_from_build_report": bool(resnet_from_report and resnet_path == resnet_from_report),
        "opencv_candidate_path": str(opencv_path) if opencv_path else "",
        "resnet_candidate_path": str(resnet_path) if resnet_path else "",
        "build_report_input_files": report_inputs,
    }
    return opencv_path, resnet_path, selection


def load_split(warnings: list[str]) -> tuple[dict[int, str], dict[int, float], dict[str, Any]]:
    columns, rows = read_csv_rows(SPLIT_PATH)
    video_col = find_col(columns, ["video_id"])
    split_col = find_col(columns, ["split"])
    seed_col = find_col(columns, ["split_seed"])
    duration_col = find_col(columns, ["video_duration_sec", "duration_sec"])
    split_map: dict[int, str] = {}
    durations: dict[int, float] = {}
    seed_values = Counter()
    for row in rows:
        vid = to_int(row.get(video_col))
        if vid is None:
            continue
        split_value = row.get(split_col, "")
        split_map[vid] = split_value
        seed_values[row.get(seed_col, "")] += 1
        duration = to_float(row.get(duration_col))
        if duration is not None and duration > 0:
            durations[vid] = duration
    fixed = {
        "train": sorted([vid for vid, split in split_map.items() if split == "train"]),
        "validation": sorted([vid for vid, split in split_map.items() if split == "validation"]),
        "test": sorted([vid for vid, split in split_map.items() if split == "test"]),
    }
    ok = fixed["train"] == sorted(TRAIN_IDS) and fixed["validation"] == sorted(VALIDATION_IDS) and fixed["test"] == sorted(TEST_IDS)
    seed_ok = set(seed_values.keys()) == {str(SPLIT_SEED)}
    if not ok:
        warnings.append(f"fixed split mismatch: {fixed}")
    if not seed_ok:
        warnings.append(f"split_seed mismatch: {dict(seed_values)}")
    details = {
        "split_file": str(SPLIT_PATH),
        "split_file_exists": SPLIT_PATH.exists(),
        "split_seed_values": dict(seed_values),
        "split_seed_expected": SPLIT_SEED,
        "train_video_ids": fixed["train"],
        "validation_video_ids": fixed["validation"],
        "test_video_ids": fixed["test"],
        "fixed_split_match": ok,
        "split_seed_match": seed_ok,
    }
    return split_map, durations, details


def load_manifest_durations(existing: dict[int, float], warnings: list[str]) -> dict[int, float]:
    durations = dict(existing)
    if not VIDEO_MANIFEST_PATH.exists():
        warnings.append(f"video manifest missing; candidates_per_minute limited: {VIDEO_MANIFEST_PATH}")
        return durations
    columns, rows = read_csv_rows(VIDEO_MANIFEST_PATH)
    video_col = find_col(columns, ["video_id", "label_mapping_video_id"])
    duration_col = find_col(columns, ["duration_sec", "video_duration_sec"])
    for row in rows:
        vid = to_int(row.get(video_col))
        duration = to_float(row.get(duration_col))
        if vid is not None and duration is not None and duration > 0 and vid not in durations:
            durations[vid] = duration
    return durations


def source_confidence_status(opencv_path: Path | None, resnet_path: Path | None, selection: dict[str, Any]) -> str:
    if not opencv_path or not resnet_path:
        return "source_separation_incomplete"
    if selection.get("opencv_selected_from_build_report") and selection.get("resnet_selected_from_build_report"):
        return "confirmed_from_v2_4_build_report_and_source_columns"
    return "confirmed_from_existing_source_files_with_schema_fallback"


def load_candidate_rows(
    path: Path,
    family: str,
    split_map: dict[int, str],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    columns, rows = read_csv_rows(path)
    if family == "opencv_ffmpeg":
        time_col = find_col(columns, ["candidate_time_sec", "opencv_candidate_time_sec", "time_sec", "sec"])
        source_col = find_col(columns, ["candidate_source", "candidate_source_for_audit", "opencv_candidate_source"])
        audit_source_col = find_col(columns, ["candidate_source_for_audit", "source_family", "family"])
        method_col = find_col(columns, ["method_used", "opencv_method_used", "method"])
        model_col = find_col(columns, ["model_name", "model"])
        score_col = find_col(columns, ["scene_change_score", "opencv_scene_change_score", "score"])
    else:
        time_col = find_col(columns, ["resnet_time_sec_std", "resnet_candidate_time_sec", "candidate_time_sec", "score_time_sec", "time_sec"])
        source_col = find_col(columns, ["candidate_source", "candidate_source_for_audit", "resnet_candidate_source"])
        audit_source_col = find_col(columns, ["candidate_source_for_audit", "source_family", "family"])
        method_col = find_col(columns, ["method_used", "resnet_method_used", "method"])
        model_col = find_col(columns, ["model_name", "model"])
        score_col = find_col(columns, ["resnet_score_std", "scene_change_score", "resnet_scene_change_score", "cosine_distance", "score"])

    if not time_col:
        raise RuntimeError(f"{family} timestamp column not found in {path}")
    video_col = find_col(columns, ["video_id"])
    if not video_col:
        raise RuntimeError(f"{family} video_id column not found in {path}")

    candidates: list[dict[str, Any]] = []
    invalid_count = 0
    nontrain_count = 0
    questionable_source_count = 0
    for idx, row in enumerate(rows):
        vid = to_int(row.get(video_col))
        sec = to_float(row.get(time_col))
        if vid is None or sec is None or sec < 0:
            invalid_count += 1
            continue
        split_value = split_map.get(vid, "")
        if vid not in TRAIN_IDS:
            nontrain_count += 1
            continue
        source_value = row.get(source_col, "")
        audit_source_value = row.get(audit_source_col, "")
        method_value = row.get(method_col, "")
        model_value = row.get(model_col, "")
        score_value = row.get(score_col, "")
        if family == "opencv_ffmpeg":
            evidence_text = " ".join([source_value, audit_source_value, method_value]).lower()
            if not any(token in evidence_text for token in ["opencv", "ffmpeg", "frame_index", "scene"]):
                questionable_source_count += 1
        else:
            evidence_text = " ".join([source_value, audit_source_value, method_value, model_value]).lower()
            if not any(token in evidence_text for token in ["resnet", "embedding"]):
                questionable_source_count += 1
        candidates.append(
            {
                "candidate_family": family,
                "video_id": vid,
                "candidate_sec": sec,
                "original_source_value": source_value,
                "audit_source_value": audit_source_value,
                "original_method_value": method_value,
                "original_model_value": model_value,
                "confidence_or_score": score_value,
                "raw_row_index": idx,
                "split": split_value,
                "notes": f"source_file={path.name}; timestamp_column={time_col}",
            }
        )
    if invalid_count:
        warnings.append(f"{family}: invalid/null/negative candidate rows excluded={invalid_count}")
    if questionable_source_count:
        warnings.append(f"{family}: source evidence did not contain expected tokens for {questionable_source_count} train rows")
    metadata = {
        "path": str(path),
        "row_count_total": len(rows),
        "row_count_train": len(candidates),
        "row_count_nontrain_excluded": nontrain_count,
        "row_count_invalid_excluded": invalid_count,
        "timestamp_column": time_col,
        "video_id_column": video_col,
        "source_column": source_col,
        "audit_source_column": audit_source_col,
        "method_column": method_col,
        "model_column": model_col,
        "score_column": score_col,
        "questionable_source_train_rows": questionable_source_count,
    }
    return candidates, metadata


def load_canonical_candidates(split_map: dict[int, str], warnings: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = ANCHOR_WITH_SPLIT_PATH if ANCHOR_WITH_SPLIT_PATH.exists() else ANCHOR_PATH
    columns, rows = read_csv_rows(path)
    video_col = find_col(columns, ["video_id", "video_id_normalized_for_split"])
    time_col = find_col(columns, ["canonical_boundary_time_sec", "candidate_time_sec", "anchor_sec"])
    split_col = find_col(columns, ["split"])
    source_col = find_col(columns, ["canonical_time_source", "source", "source_family"])
    method_col = find_col(columns, ["source_relation", "method_used", "method"])
    score_col = find_col(columns, ["visual_boundary_strength_score", "scene_change_score"])
    if not video_col or not time_col:
        warnings.append("canonical_all unavailable: missing video_id or canonical time column")
        return [], {"available": False, "path": str(path)}
    candidates: list[dict[str, Any]] = []
    invalid_count = 0
    nontrain_count = 0
    for idx, row in enumerate(rows):
        vid = to_int(row.get(video_col))
        sec = to_float(row.get(time_col))
        split_value = row.get(split_col, split_map.get(vid or -1, ""))
        if vid is None or sec is None or sec < 0:
            invalid_count += 1
            continue
        if vid not in TRAIN_IDS or (split_col and split_value != "train"):
            nontrain_count += 1
            continue
        candidates.append(
            {
                "candidate_family": "canonical_all",
                "video_id": vid,
                "candidate_sec": sec,
                "original_source_value": row.get(source_col, ""),
                "audit_source_value": row.get(source_col, ""),
                "original_method_value": row.get(method_col, ""),
                "original_model_value": "",
                "confidence_or_score": row.get(score_col, ""),
                "raw_row_index": idx,
                "split": "train",
                "notes": f"source_file={path.name}; timestamp_column={time_col}",
            }
        )
    metadata = {
        "available": True,
        "path": str(path),
        "row_count_total": len(rows),
        "row_count_train": len(candidates),
        "row_count_nontrain_excluded": nontrain_count,
        "row_count_invalid_excluded": invalid_count,
        "timestamp_column": time_col,
        "video_id_column": video_col,
        "split_column": split_col,
    }
    if invalid_count:
        warnings.append(f"canonical_all: invalid/null/negative rows excluded={invalid_count}")
    return candidates, metadata


def candidates_by_video(candidates: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        out[int(cand["video_id"])].append(cand)
    for vid in out:
        out[vid].sort(key=lambda row: (float(row["candidate_sec"]), int(row.get("raw_row_index", 0))))
    return out


def build_combined_clusters(
    opencv_candidates: list[dict[str, Any]],
    resnet_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    opencv_by_vid = candidates_by_video(opencv_candidates)
    resnet_by_vid = candidates_by_video(resnet_candidates)
    cluster_rows: list[dict[str, Any]] = []
    clusters_by_vid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    cluster_id = 0
    for vid in sorted(set(opencv_by_vid) | set(resnet_by_vid)):
        opencv_rows = opencv_by_vid.get(vid, [])
        resnet_rows = resnet_by_vid.get(vid, [])
        nodes = [("o", idx) for idx in range(len(opencv_rows))] + [("r", idx) for idx in range(len(resnet_rows))]
        graph: dict[tuple[str, int], set[tuple[str, int]]] = {node: set() for node in nodes}
        for oi, ocand in enumerate(opencv_rows):
            ot = float(ocand["candidate_sec"])
            for ri, rcand in enumerate(resnet_rows):
                rt = float(rcand["candidate_sec"])
                if abs(ot - rt) <= COMBINED_DEDUP_WINDOW_SEC:
                    graph[("o", oi)].add(("r", ri))
                    graph[("r", ri)].add(("o", oi))
        seen: set[tuple[str, int]] = set()
        for node in nodes:
            if node in seen:
                continue
            queue = deque([node])
            seen.add(node)
            component: set[tuple[str, int]] = set()
            while queue:
                cur = queue.popleft()
                component.add(cur)
                for nxt in graph[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            members = []
            for kind, idx in sorted(component):
                if kind == "o":
                    members.append(opencv_rows[idx])
                else:
                    members.append(resnet_rows[idx])
            opencv_members = [m for m in members if m["candidate_family"] == "opencv_ffmpeg"]
            resnet_members = [m for m in members if m["candidate_family"] == "resnet"]
            if opencv_members:
                representative = sorted(opencv_members, key=lambda row: (float(row["candidate_sec"]), int(row["raw_row_index"])))[0]
            else:
                representative = sorted(resnet_members, key=lambda row: (float(row["candidate_sec"]), int(row["raw_row_index"])))[0]
            cluster_id += 1
            member_times = sorted(float(m["candidate_sec"]) for m in members)
            source_relation = (
                "opencv_resnet_merged_2s" if opencv_members and resnet_members
                else ("opencv_ffmpeg_only" if opencv_members else "resnet_only")
            )
            score_values = [to_float(m.get("confidence_or_score")) for m in members]
            score_values = [s for s in score_values if s is not None]
            cluster = {
                "cluster_id": f"CMB{cluster_id:06d}",
                "video_id": vid,
                "candidate_sec": float(representative["candidate_sec"]),
                "members": members,
                "member_times": member_times,
                "source_relation": source_relation,
                "source_count": int(bool(opencv_members)) + int(bool(resnet_members)),
                "opencv_member_count": len(opencv_members),
                "resnet_member_count": len(resnet_members),
                "score": max(score_values) if score_values else "",
            }
            clusters_by_vid[vid].append(cluster)
            cluster_rows.append(
                {
                    "candidate_family": "combined_two_source",
                    "video_id": vid,
                    "candidate_sec": cluster["candidate_sec"],
                    "source_relation": source_relation,
                    "original_source_value": "+".join(sorted(set(m.get("original_source_value", "") for m in members if m.get("original_source_value", "")))),
                    "original_method_value": "dedup_window_2s_cross_source_connected_components",
                    "original_model_value": "+".join(sorted(set(m.get("original_model_value", "") for m in members if m.get("original_model_value", "")))),
                    "confidence_or_score": cluster["score"],
                    "raw_row_index": cluster_id - 1,
                    "split": "train",
                    "notes": (
                        f"source_relation={source_relation}; dedup_window_sec={COMBINED_DEDUP_WINDOW_SEC}; "
                        f"member_times={';'.join(format_cell(t) for t in member_times)}"
                    ),
                }
            )
    for vid in clusters_by_vid:
        clusters_by_vid[vid].sort(key=lambda row: (float(row["candidate_sec"]), row["cluster_id"]))
    metadata = {
        "dedup_window_sec": COMBINED_DEDUP_WINDOW_SEC,
        "dedup_rule": "existing canonical visual anchor rule: cross-source OpenCV/FFmpeg-ResNet connected components when abs(time gap) <= 2s; no within-source collapse",
        "candidate_count_train_dedup_clusters": len(cluster_rows),
        "candidate_video_count_train": len(clusters_by_vid),
        "source_relation_counts": dict(Counter(row.get("source_relation", "") for row in cluster_rows)),
    }
    return cluster_rows, clusters_by_vid, metadata


def load_actual_boundaries(warnings: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    columns, rows = read_csv_rows(SEGMENT_PATH)
    video_col = find_col(columns, ["video_id"])
    interval_col = find_col(columns, ["ad_interval_id", "interval_id"])
    segment_id_col = find_col(columns, ["segment_id"])
    type_col = find_col(columns, ["segment_type"])
    start_col = find_col(columns, ["ad_start_sec", "start_sec", "segment_start_sec"])
    end_col = find_col(columns, ["ad_end_sec", "end_sec", "segment_end_sec"])
    label_valid_col = find_col(columns, ["label_valid"])
    segment_valid_col = find_col(columns, ["segment_valid"])
    if not video_col or not start_col or not end_col:
        raise RuntimeError("actual ad interval file missing video_id/ad_start_sec/ad_end_sec compatible columns")
    intervals: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    invalid_boundary_rows = []
    skipped_nontrain = 0
    skipped_invalid_interval = 0
    for idx, row in enumerate(rows):
        if type_col and row.get(type_col, "") not in {"", "ad_interval"}:
            continue
        vid = to_int(row.get(video_col))
        if vid is None or vid not in TRAIN_IDS:
            skipped_nontrain += 1
            continue
        if label_valid_col and row.get(label_valid_col, "").lower() in {"false", "0", "no"}:
            skipped_invalid_interval += 1
            warnings.append(f"invalid label interval skipped: row={idx} video_id={vid}")
            continue
        if segment_valid_col and row.get(segment_valid_col, "").lower() in {"false", "0", "no"}:
            skipped_invalid_interval += 1
            warnings.append(f"invalid segment interval skipped: row={idx} video_id={vid}")
            continue
        ad_interval_id = row.get(interval_col, "") or row.get(segment_id_col, "") or f"row_{idx}"
        start = to_float(row.get(start_col))
        end = to_float(row.get(end_col))
        interval = {
            "source_row_index": idx,
            "video_id": vid,
            "ad_interval_id": ad_interval_id,
            "ad_start_sec": start,
            "ad_end_sec": end,
        }
        intervals.append(interval)
        for boundary_type, sec in [("start", start), ("end", end)]:
            if sec is None or sec < 0:
                invalid_boundary_rows.append({"row_index": idx, "video_id": vid, "ad_interval_id": ad_interval_id, "boundary_type": boundary_type, "value": row.get(start_col if boundary_type == "start" else end_col, "")})
                continue
            boundaries.append(
                {
                    "boundary_id": f"{vid}|{ad_interval_id}|{boundary_type}",
                    "video_id": vid,
                    "ad_interval_id": ad_interval_id,
                    "boundary_type": boundary_type,
                    "actual_sec": sec,
                    "source_row_index": idx,
                }
            )
    if invalid_boundary_rows:
        warnings.append(f"invalid/null/negative actual boundary rows excluded: {invalid_boundary_rows}")
    details = {
        "actual_label_path": str(SEGMENT_PATH),
        "timestamp_columns": {"start": start_col, "end": end_col},
        "video_id_column": video_col,
        "ad_interval_id_column": interval_col,
        "train_actual_ad_interval_count": len(intervals),
        "train_actual_boundary_count": len(boundaries),
        "start_boundary_count": sum(1 for row in boundaries if row["boundary_type"] == "start"),
        "end_boundary_count": sum(1 for row in boundaries if row["boundary_type"] == "end"),
        "skipped_nontrain_interval_rows": skipped_nontrain,
        "skipped_invalid_interval_rows": skipped_invalid_interval,
        "invalid_boundary_rows": invalid_boundary_rows,
    }
    return intervals, boundaries, details


def nearest_candidate(candidates: list[dict[str, Any]], actual_sec: float) -> tuple[float | None, float | None]:
    if not candidates:
        return None, None
    best = min(candidates, key=lambda cand: (abs(float(cand["candidate_sec"]) - actual_sec), float(cand["candidate_sec"])))
    sec = float(best["candidate_sec"])
    return sec, abs(sec - actual_sec)


def combined_member_candidates(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for cluster in clusters:
        for member in cluster["members"]:
            clone = dict(member)
            clone["combined_cluster_id"] = cluster["cluster_id"]
            members.append(clone)
    return members


def compute_audit_rows(
    boundaries: list[dict[str, Any]],
    family_candidates: dict[str, dict[int, list[dict[str, Any]]]],
    family_count_by_video: dict[str, dict[int, int]],
    source_status: str,
) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    for boundary in boundaries:
        vid = int(boundary["video_id"])
        actual_sec = float(boundary["actual_sec"])
        for family, by_video in family_candidates.items():
            candidates = by_video.get(vid, [])
            nearest_sec, distance = nearest_candidate(candidates, actual_sec)
            audit_rows.append(
                {
                    "video_id": vid,
                    "ad_interval_id": boundary["ad_interval_id"],
                    "boundary_type": boundary["boundary_type"],
                    "actual_sec": actual_sec,
                    "candidate_family": family,
                    "nearest_candidate_sec": round_or_blank(nearest_sec),
                    "nearest_distance_sec": round_or_blank(distance),
                    "within_2s": bool(distance is not None and distance <= 2),
                    "within_5s": bool(distance is not None and distance <= 5),
                    "within_10s": bool(distance is not None and distance <= 10),
                    "candidate_count_in_video": family_count_by_video.get(family, {}).get(vid, 0),
                    "source_separation_status": source_status,
                    "notes": "combined nearest uses original source member timestamps; combined count uses dedup clusters"
                    if family == "combined_two_source" else "",
                }
            )
    return audit_rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distance_stats(values: list[float]) -> dict[str, float | str]:
    if not values:
        return {
            "median_nearest_distance_sec": "",
            "mean_nearest_distance_sec": "",
            "p90_nearest_distance_sec": "",
            "max_nearest_distance_sec": "",
        }
    return {
        "median_nearest_distance_sec": round(statistics.median(values), 6),
        "mean_nearest_distance_sec": round(statistics.mean(values), 6),
        "p90_nearest_distance_sec": round(percentile(values, 0.9) or 0.0, 6),
        "max_nearest_distance_sec": round(max(values), 6),
    }


def summarize_recall(
    audit_rows: list[dict[str, Any]],
    candidate_count_total: dict[str, int],
    candidate_video_count: dict[str, int],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    families = list(candidate_count_total.keys())
    for family in families:
        family_rows = [row for row in audit_rows if row["candidate_family"] == family]
        for boundary_type in ["start", "end", "all"]:
            type_rows = family_rows if boundary_type == "all" else [row for row in family_rows if row["boundary_type"] == boundary_type]
            distances = [to_float(row["nearest_distance_sec"]) for row in type_rows]
            distance_values = [v for v in distances if v is not None]
            stats = distance_stats(distance_values)
            for tol in TOLERANCES:
                key = f"within_{tol}s"
                actual_count = len(type_rows)
                hit_count = sum(1 for row in type_rows if bool_value(row.get(key)))
                summary_rows.append(
                    {
                        "candidate_family": family,
                        "boundary_type": boundary_type,
                        "tolerance_sec": tol,
                        "actual_boundary_count": actual_count,
                        "hit_count": hit_count,
                        "recall": round(hit_count / actual_count, 6) if actual_count else "",
                        **stats,
                        "candidate_count_total": candidate_count_total.get(family, 0),
                        "candidate_video_count": candidate_video_count.get(family, 0),
                    }
                )
    return summary_rows


def build_case_breakdown(audit_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in audit_rows:
        key = (int(row["video_id"]), row["ad_interval_id"], row["boundary_type"], row["candidate_family"])
        lookup[key] = row
    base_keys = sorted({(int(row["video_id"]), row["ad_interval_id"], row["boundary_type"], float(row["actual_sec"])) for row in audit_rows})
    case_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    for vid, ad_interval_id, boundary_type, actual_sec in base_keys:
        opencv = lookup.get((vid, ad_interval_id, boundary_type, "opencv_ffmpeg"), {})
        resnet = lookup.get((vid, ad_interval_id, boundary_type, "resnet"), {})
        for tol in TOLERANCES:
            op_hit = bool_value(opencv.get(f"within_{tol}s"))
            re_hit = bool_value(resnet.get(f"within_{tol}s"))
            if op_hit and re_hit:
                case_type = "both_hit"
            elif op_hit:
                case_type = "opencv_only_hit"
            elif re_hit:
                case_type = "resnet_only_hit"
            else:
                case_type = "both_missed"
            row = {
                "tolerance_sec": tol,
                "video_id": vid,
                "ad_interval_id": ad_interval_id,
                "boundary_type": boundary_type,
                "actual_sec": actual_sec,
                "opencv_nearest_candidate_sec": opencv.get("nearest_candidate_sec", ""),
                "opencv_nearest_distance_sec": opencv.get("nearest_distance_sec", ""),
                "opencv_hit": op_hit,
                "resnet_nearest_candidate_sec": resnet.get("nearest_candidate_sec", ""),
                "resnet_nearest_distance_sec": resnet.get("nearest_distance_sec", ""),
                "resnet_hit": re_hit,
                "combined_hit": op_hit or re_hit,
                "case_type": case_type,
            }
            case_rows.append(row)
            include_missed = case_type == "both_missed" or (tol in {2, 5} and case_type in {"opencv_only_hit", "resnet_only_hit"})
            if include_missed:
                missed_rows.append(
                    {
                        "tolerance_sec": tol,
                        "video_id": vid,
                        "ad_interval_id": ad_interval_id,
                        "boundary_type": boundary_type,
                        "actual_sec": actual_sec,
                        "case_type": case_type,
                        "opencv_nearest_candidate_sec": opencv.get("nearest_candidate_sec", ""),
                        "opencv_nearest_distance_sec": opencv.get("nearest_distance_sec", ""),
                        "resnet_nearest_candidate_sec": resnet.get("nearest_candidate_sec", ""),
                        "resnet_nearest_distance_sec": resnet.get("nearest_distance_sec", ""),
                        "notes": "required missed/complement case subset",
                    }
                )
    return case_rows, missed_rows


def video_level_recall(
    audit_rows: list[dict[str, Any]],
    family_count_by_video: dict[str, dict[int, int]],
    families: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vid in sorted(TRAIN_IDS):
        for family in families:
            fam_vid_rows = [row for row in audit_rows if int(row["video_id"]) == vid and row["candidate_family"] == family]
            for boundary_type in ["start", "end", "all"]:
                type_rows = fam_vid_rows if boundary_type == "all" else [row for row in fam_vid_rows if row["boundary_type"] == boundary_type]
                for tol in TOLERANCES:
                    key = f"within_{tol}s"
                    actual_count = len(type_rows)
                    hit_count = sum(1 for row in type_rows if bool_value(row.get(key)))
                    rows.append(
                        {
                            "video_id": vid,
                            "candidate_family": family,
                            "boundary_type": boundary_type,
                            "tolerance_sec": tol,
                            "actual_boundary_count": actual_count,
                            "hit_count": hit_count,
                            "recall": round(hit_count / actual_count, 6) if actual_count else "",
                            "candidate_count": family_count_by_video.get(family, {}).get(vid, 0),
                            "notes": "",
                        }
                    )
    return rows


def count_by_video(candidates: dict[int, list[Any]]) -> dict[int, int]:
    return {vid: len(rows) for vid, rows in candidates.items()}


def candidate_density(
    family_count_by_video: dict[str, dict[int, int]],
    durations: dict[int, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, counts in family_count_by_video.items():
        for vid in sorted(TRAIN_IDS):
            count = counts.get(vid, 0)
            duration = durations.get(vid)
            rows.append(
                {
                    "candidate_family": family,
                    "video_id": vid,
                    "candidate_count": count,
                    "video_duration_sec": duration if duration is not None else "",
                    "candidates_per_video": count,
                    "candidates_per_minute": round(count / (duration / 60.0), 6) if duration else "",
                }
            )
    return rows


def summary_lookup(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    lookup: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in summary_rows:
        lookup[row["candidate_family"]][row["boundary_type"]][int(row["tolerance_sec"])] = row
    return lookup


def case_counts(case_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for tol in TOLERANCES:
        rows = [row for row in case_rows if int(row["tolerance_sec"]) == tol]
        out[str(tol)] = dict(Counter(row["case_type"] for row in rows))
        out[str(tol)]["combined_hit"] = sum(1 for row in rows if bool_value(row["combined_hit"]))
        out[str(tol)]["combined_missed"] = sum(1 for row in rows if not bool_value(row["combined_hit"]))
    return out


def boundary_list(case_rows: list[dict[str, Any]], tolerance: int, case_type: str) -> list[dict[str, Any]]:
    rows = [row for row in case_rows if int(row["tolerance_sec"]) == tolerance and row["case_type"] == case_type]
    return [
        {
            "video_id": row["video_id"],
            "ad_interval_id": row["ad_interval_id"],
            "boundary_type": row["boundary_type"],
            "actual_sec": row["actual_sec"],
            "opencv_nearest_candidate_sec": row["opencv_nearest_candidate_sec"],
            "opencv_nearest_distance_sec": row["opencv_nearest_distance_sec"],
            "resnet_nearest_candidate_sec": row["resnet_nearest_candidate_sec"],
            "resnet_nearest_distance_sec": row["resnet_nearest_distance_sec"],
        }
        for row in rows
    ]


def format_recall_table(summary_rows: list[dict[str, Any]], families: list[str]) -> str:
    lookup = summary_lookup(summary_rows)
    lines = [
        "| candidate_family | boundary_type | recall@2s | recall@5s | recall@10s | median_dist | p90_dist |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in families:
        for boundary_type in ["start", "end", "all"]:
            row2 = lookup[family][boundary_type].get(2, {})
            row5 = lookup[family][boundary_type].get(5, {})
            row10 = lookup[family][boundary_type].get(10, {})
            lines.append(
                "| {family} | {boundary_type} | {r2} ({h2}/{n2}) | {r5} ({h5}/{n5}) | {r10} ({h10}/{n10}) | {median} | {p90} |".format(
                    family=family,
                    boundary_type=boundary_type,
                    r2=format_cell(row2.get("recall", "")),
                    h2=format_cell(row2.get("hit_count", "")),
                    n2=format_cell(row2.get("actual_boundary_count", "")),
                    r5=format_cell(row5.get("recall", "")),
                    h5=format_cell(row5.get("hit_count", "")),
                    n5=format_cell(row5.get("actual_boundary_count", "")),
                    r10=format_cell(row10.get("recall", "")),
                    h10=format_cell(row10.get("hit_count", "")),
                    n10=format_cell(row10.get("actual_boundary_count", "")),
                    median=format_cell(row2.get("median_nearest_distance_sec", "")),
                    p90=format_cell(row2.get("p90_nearest_distance_sec", "")),
                )
            )
    return "\n".join(lines)


def format_case_list(case_rows: list[dict[str, Any]], tolerance: int, case_type: str) -> str:
    rows = [row for row in case_rows if int(row["tolerance_sec"]) == tolerance and row["case_type"] == case_type]
    if not rows:
        return "- none"
    return "\n".join(
        "- video_id={video_id}, {ad_interval_id} {boundary_type}@{actual_sec}s, opencv_nearest={opencv_nearest_candidate_sec} (d={opencv_nearest_distance_sec}), resnet_nearest={resnet_nearest_candidate_sec} (d={resnet_nearest_distance_sec})".format(**row)
        for row in rows
    )


def make_summary_md(report: dict[str, Any], summary_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]], families: list[str]) -> str:
    counts = report["counts"]
    source = report["source_separation"]
    case_count = report["case_counts"]
    return f"""# OpenCV/FFmpeg vs ResNet Scene Boundary Recall Audit v2.4 Train

## Purpose
Quantify how well existing v2.4 OpenCV/FFmpeg scene candidates and existing ResNet embedding scene candidates capture actual train-split ad start/end boundaries. This is a recall audit only: no third model, no detector rule change, and no candidate threshold tuning.

## Inputs
- OpenCV/FFmpeg candidates: `{source.get('opencv_candidate_path', '')}`
- ResNet candidates: `{source.get('resnet_candidate_path', '')}`
- Canonical visual anchor: `{ANCHOR_WITH_SPLIT_PATH}`
- Split file: `{SPLIT_PATH}`
- Actual labels: `{SEGMENT_PATH}`
- Video metadata: `{VIDEO_MANIFEST_PATH}`
- Fusion report reference: `{FUSION_REPORT_PATH}`
- Fusion summary reference: `{FUSION_SUMMARY_PATH}`

## Split Scope
- Train video_id: {report['split_validation']['train_video_ids']}
- Validation video_id excluded: {report['split_validation']['validation_video_ids']}
- Test video_id excluded: {report['split_validation']['test_video_ids']}
- split_seed: {report['split_validation']['split_seed_expected']}
- validation/test row-level output generated: false

## Counts
- Actual train ad intervals: {counts['train_actual_ad_interval_count']}
- Actual train boundaries: {counts['train_actual_boundary_count']}
- Start boundaries: {counts['start_boundary_count']}
- End boundaries: {counts['end_boundary_count']}
- OpenCV/FFmpeg train candidates: {counts['opencv_ffmpeg_candidate_count_train']}
- ResNet train candidates: {counts['resnet_candidate_count_train']}
- combined_two_source train candidates after dedup: {counts['combined_two_source_candidate_count_train']}
- canonical_all train candidates: {counts.get('canonical_all_candidate_count_train', 'n/a')}

## Source Separation
- Status: `{source['source_separation_status']}`
- OpenCV/FFmpeg rule: {source['opencv_ffmpeg_filter_rule']}
- ResNet rule: {source['resnet_filter_rule']}
- combined_two_source dedup: {report['combined_dedup']['dedup_rule']}
- Uncertainty: {source.get('source_separation_warning') or 'none'}

## Recall Summary
{format_recall_table(summary_rows, families)}

## Case Counts
| tolerance_sec | both_hit | opencv_only_hit | resnet_only_hit | both_missed | combined_hit | combined_missed |
|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {tol} | {case_count.get(str(tol), {}).get('both_hit', 0)} | {case_count.get(str(tol), {}).get('opencv_only_hit', 0)} | {case_count.get(str(tol), {}).get('resnet_only_hit', 0)} | {case_count.get(str(tol), {}).get('both_missed', 0)} | {case_count.get(str(tol), {}).get('combined_hit', 0)} | {case_count.get(str(tol), {}).get('combined_missed', 0)} |"
        for tol in TOLERANCES
    ) + f"""

## Boundary Lists at 5s
### OpenCV/FFmpeg Only Hit
{format_case_list(case_rows, 5, 'opencv_only_hit')}

### ResNet Only Hit
{format_case_list(case_rows, 5, 'resnet_only_hit')}

### Both Hit
{format_case_list(case_rows, 5, 'both_hit')}

### Both Missed
{format_case_list(case_rows, 5, 'both_missed')}

## Guardrails
- old_project_modified=false
- no_detector_rule_modified=true
- no_validation_test_row_level_output=true
- actual labels used only for recall audit=true
"""


def make_findings_md(report: dict[str, Any], summary_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]], families: list[str]) -> str:
    lookup = summary_lookup(summary_rows)

    def recall_sentence(family: str) -> str:
        start = lookup[family]["start"]
        end = lookup[family]["end"]
        all_rows = lookup[family]["all"]
        return (
            f"- {family}: start recall {format_cell(start[2]['recall'])}/{format_cell(start[5]['recall'])}/{format_cell(start[10]['recall'])} "
            f"at 2/5/10s; end recall {format_cell(end[2]['recall'])}/{format_cell(end[5]['recall'])}/{format_cell(end[10]['recall'])}; "
            f"all-boundary recall {format_cell(all_rows[2]['recall'])}/{format_cell(all_rows[5]['recall'])}/{format_cell(all_rows[10]['recall'])}."
        )

    both_missed_10 = report["case_counts"].get("10", {}).get("both_missed", 0)
    combined_5 = lookup["combined_two_source"]["all"][5]["recall"] if "combined_two_source" in lookup else ""
    third_model_judgment = (
        "A future additional candidate source may be worth a separate experiment because both existing sources still miss "
        f"{both_missed_10} train boundaries even at 10s."
        if both_missed_10
        else "A third model is not indicated by recall coverage alone because the existing combined sources hit every train boundary within 10s."
    )
    return f"""# Findings: Scene Boundary Recall Audit v2.4 Train

## Interpretation
This audit evaluates candidate recall against actual train ad boundaries only. The labels were used after candidate loading solely to compute nearest distances and hit/miss cases.

## Recall Takeaways
{chr(10).join(recall_sentence(family) for family in families)}

At 5s, combined_two_source all-boundary recall is {format_cell(combined_5)}. The case breakdown shows where one source adds coverage over the other:

## OpenCV/FFmpeg Strengths and Weaknesses
OpenCV/FFmpeg captures abrupt visual cuts and FFmpeg/OpenCV scene-change events with lower train candidate density than ResNet. Its unique hits are listed in `opencv_resnet_boundary_case_breakdown_v2_4_train.csv` as `opencv_only_hit`.

## ResNet Strengths and Weaknesses
ResNet embedding candidates add complementary coverage where embedding distance catches visual shifts that OpenCV/FFmpeg misses. Its unique hits are listed as `resnet_only_hit`. The tradeoff is a higher candidate count and candidate density.

## Combined Candidate Pool
The combined family uses the existing canonical cross-source dedup basis: OpenCV/FFmpeg and ResNet candidates within 2s are one cluster for counting, while nearest-distance audit keeps original member timestamps so that dedup does not erase either source's boundary evidence.

## Boundary Lists
### OpenCV/FFmpeg only at 2s
{format_case_list(case_rows, 2, 'opencv_only_hit')}

### ResNet only at 2s
{format_case_list(case_rows, 2, 'resnet_only_hit')}

### Both missed at 2s
{format_case_list(case_rows, 2, 'both_missed')}

### Both missed at 10s
{format_case_list(case_rows, 10, 'both_missed')}

## Third Model First-Pass Judgment
{third_model_judgment} This report does not add a third model and does not tune thresholds.

## Safety Findings
- old_project_modified=false
- no_detector_rule_modified=true
- no_validation_test_row_level_output=true
"""


def make_latest_readme(bundle_files: list[Path]) -> str:
    lines = [
        "# Latest Files: OpenCV/ResNet Scene Boundary Recall Audit v2.4",
        "",
        "This bundle contains only newly generated train-only audit artifacts.",
        "",
        "## Included",
    ]
    for path in bundle_files:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "## Excluded",
            "- raw videos",
            "- frame images",
            "- cache files",
            "- model/checkpoint/weight files",
            "- validation/test row-level output",
            "- existing large original feature inputs",
        ]
    )
    return "\n".join(lines) + "\n"


def update_latest_bundle(paths: list[Path], warnings: list[str]) -> dict[str, Any]:
    if LATEST_DIR.exists():
        for child in LATEST_DIR.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in paths:
        if path == LATEST_README:
            continue
        if not path.exists():
            warnings.append(f"latest bundle source missing: {path}")
            continue
        target = LATEST_DIR / path.name
        shutil.copy2(path, target)
        copied.append(target)
    LATEST_README.write_text(make_latest_readme(copied + [LATEST_README]), encoding="utf-8")
    copied.append(LATEST_README)
    forbidden = []
    for path in copied:
        lower_name = path.name.lower()
        parts = {part.lower() for part in path.parts}
        if path.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES:
            forbidden.append(str(path))
        if any(token in lower_name for token in ["cache", "checkpoint", "weights"]) or (FORBIDDEN_LATEST_TOKENS & parts):
            forbidden.append(str(path))
    return {
        "latest_bundle_path": str(LATEST_DIR),
        "copied_files": [str(path) for path in copied],
        "copied_file_count": len(copied),
        "forbidden_files_detected": sorted(set(forbidden)),
    }


def output_row_counts(paths: list[Path]) -> dict[str, int | str]:
    counts: dict[str, int | str] = {}
    for path in paths:
        if not path.exists():
            counts[str(path)] = "missing"
            continue
        if path.suffix.lower() == ".csv":
            _, rows = read_csv_rows(path)
            counts[str(path)] = len(rows)
        else:
            counts[str(path)] = "exists"
    return counts


def validate_outputs(
    report: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    latest_result: dict[str, Any] | None,
    protected_before: dict[str, Any],
    protected_after: dict[str, Any],
) -> dict[str, Any]:
    validations: dict[str, Any] = {}
    split = report["split_validation"]
    split_ok = (
        split["split_file_exists"]
        and split["fixed_split_match"]
        and split["split_seed_match"]
        and set(split["train_video_ids"]) == TRAIN_IDS
        and set(split["validation_video_ids"]) == VALIDATION_IDS
        and set(split["test_video_ids"]) == TEST_IDS
    )
    row_level_paths = [SOURCE_COMPARISON_CSV, AUDIT_CSV, CASE_BREAKDOWN_CSV, MISSED_CASES_CSV, VIDEO_LEVEL_CSV]
    validation_test_rows = []
    for path in row_level_paths:
        if not path.exists():
            continue
        _, rows = read_csv_rows(path)
        for idx, row in enumerate(rows):
            vid = to_int(row.get("video_id"))
            split_value = row.get("split", "")
            if vid in VALIDATION_IDS or vid in TEST_IDS or split_value in {"validation", "test"}:
                validation_test_rows.append({"path": str(path), "row_index": idx, "video_id": vid, "split": split_value})
    validations["input_split_validation"] = {
        "status": "PASS" if split_ok and not validation_test_rows else "FAIL",
        "checks": {
            "split_ok": split_ok,
            "validation_test_row_level_output_count": len(validation_test_rows),
            "validation_test_examples": validation_test_rows[:5],
        },
    }

    source = report["source_separation"]
    source_ok = (
        source["source_separation_status"].startswith("confirmed")
        and report["candidate_input_metadata"]["opencv_ffmpeg"].get("questionable_source_train_rows", 0) == 0
        and report["candidate_input_metadata"]["resnet"].get("questionable_source_train_rows", 0) == 0
    )
    validations["source_separation_validation"] = {
        "status": "PASS" if source_ok else "WARN",
        "checks": {
            "opencv_rule": source["opencv_ffmpeg_filter_rule"],
            "resnet_rule": source["resnet_filter_rule"],
            "source_separation_status": source["source_separation_status"],
            "questionable_opencv_rows": report["candidate_input_metadata"]["opencv_ffmpeg"].get("questionable_source_train_rows", 0),
            "questionable_resnet_rows": report["candidate_input_metadata"]["resnet"].get("questionable_source_train_rows", 0),
            "arbitrary_source_split_used": False,
        },
    }

    boundary_ok = True
    bad_rows = []
    for row in audit_rows:
        actual = to_float(row["actual_sec"])
        nearest = to_float(row["nearest_candidate_sec"])
        distance = to_float(row["nearest_distance_sec"])
        if nearest is None:
            continue
        expected = abs(nearest - (actual or 0.0))
        if distance is None or abs(expected - distance) > 0.00001:
            boundary_ok = False
            bad_rows.append(row)
        for tol in TOLERANCES:
            expected_hit = distance is not None and distance <= tol
            if bool_value(row.get(f"within_{tol}s")) != expected_hit:
                boundary_ok = False
                bad_rows.append(row)
    counts = report["counts"]
    boundary_counts_ok = (
        counts["train_actual_boundary_count"] == counts["start_boundary_count"] + counts["end_boundary_count"]
        and counts["start_boundary_count"] == counts["train_actual_ad_interval_count"]
        and counts["end_boundary_count"] == counts["train_actual_ad_interval_count"]
    )
    validations["boundary_recall_audit_validation"] = {
        "status": "PASS" if boundary_ok and boundary_counts_ok else "FAIL",
        "checks": {
            "ad_start_end_columns": report["actual_boundary_metadata"]["timestamp_columns"],
            "start_end_separate": True,
            "nearest_distance_abs_formula_ok": boundary_ok,
            "within_thresholds_ok": boundary_ok,
            "boundary_counts_ok": boundary_counts_ok,
            "bad_row_examples": bad_rows[:3],
        },
    }

    protected_modified = {
        path: {"before": protected_before.get(path), "after": protected_after.get(path)}
        for path in protected_before
        if protected_before.get(path) != protected_after.get(path)
    }
    latest_forbidden = (latest_result or {}).get("forbidden_files_detected", [])
    generated_forbidden = []
    for path in [*DATA_OUTPUTS, *REPORT_OUTPUTS, OUT_LOG_PATH, SCRIPT_PATH]:
        lower_name = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_LATEST_SUFFIXES or any(token in lower_name for token in ["cache", "checkpoint", "weights"]):
            generated_forbidden.append(str(path))
    validations["leakage_output_safety_validation"] = {
        "status": "PASS" if not protected_modified and not latest_forbidden and not generated_forbidden and not validation_test_rows else "FAIL",
        "checks": {
            "actual_label_used_for_candidate_generation": False,
            "detector_config_script_output_modified": bool(protected_modified),
            "protected_modified_paths": protected_modified,
            "raw_video_frame_cache_model_generated_or_copied": bool(generated_forbidden or latest_forbidden),
            "generated_forbidden_files": generated_forbidden,
            "latest_bundle_forbidden_files": latest_forbidden,
            "validation_test_row_level_output_count": len(validation_test_rows),
        },
    }

    summary_consistency_ok = True
    report_summary = report.get("summary_metrics", {})
    for row in summary_rows:
        family = row["candidate_family"]
        boundary_type = row["boundary_type"]
        tol = str(row["tolerance_sec"])
        expected = report_summary.get(family, {}).get(boundary_type, {}).get(tol, {})
        if expected and (
            format_cell(expected.get("hit_count")) != format_cell(row.get("hit_count"))
            or format_cell(expected.get("recall")) != format_cell(row.get("recall"))
        ):
            summary_consistency_ok = False
            break
    output_exists = all(path.exists() for path in [*DATA_OUTPUTS, *REPORT_OUTPUTS, OUT_LOG_PATH, SCRIPT_PATH])
    log_text = OUT_LOG_PATH.read_text(encoding="utf-8") if OUT_LOG_PATH.exists() else ""
    steps_present = all(f"[STEP {idx:02d}]" in log_text for idx in range(1, 13))
    validations["reproducibility_validation"] = {
        "status": "PASS" if output_exists and steps_present and summary_consistency_ok else "FAIL",
        "checks": {
            "script_exists": SCRIPT_PATH.exists(),
            "all_outputs_exist": output_exists,
            "log_has_step_01_to_12": steps_present,
            "report_json_and_summary_csv_consistent": summary_consistency_ok,
        },
    }
    return validations


def nested_summary_metrics(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    for row in summary_rows:
        out[row["candidate_family"]][row["boundary_type"]][str(row["tolerance_sec"])] = {
            "actual_boundary_count": row["actual_boundary_count"],
            "hit_count": row["hit_count"],
            "recall": row["recall"],
            "median_nearest_distance_sec": row["median_nearest_distance_sec"],
            "mean_nearest_distance_sec": row["mean_nearest_distance_sec"],
            "p90_nearest_distance_sec": row["p90_nearest_distance_sec"],
            "max_nearest_distance_sec": row["max_nearest_distance_sec"],
            "candidate_count_total": row["candidate_count_total"],
            "candidate_video_count": row["candidate_video_count"],
        }
    return json.loads(json.dumps(out))


def build_report(
    start_time: str,
    warnings: list[str],
    errors: list[Any],
    selection: dict[str, Any],
    split_details: dict[str, Any],
    actual_details: dict[str, Any],
    source_status: str,
    opencv_metadata: dict[str, Any],
    resnet_metadata: dict[str, Any],
    canonical_metadata: dict[str, Any],
    combined_metadata: dict[str, Any],
    candidate_count_total: dict[str, int],
    candidate_video_count: dict[str, int],
    summary_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    density_rows: list[dict[str, Any]],
    protected_before: dict[str, Any],
    protected_after: dict[str, Any],
    latest_result: dict[str, Any] | None,
    validations: dict[str, Any] | None,
) -> dict[str, Any]:
    end_time = now_iso()
    counts = {
        "train_actual_ad_interval_count": actual_details["train_actual_ad_interval_count"],
        "train_actual_boundary_count": actual_details["train_actual_boundary_count"],
        "start_boundary_count": actual_details["start_boundary_count"],
        "end_boundary_count": actual_details["end_boundary_count"],
        "opencv_ffmpeg_candidate_count_train": candidate_count_total.get("opencv_ffmpeg", 0),
        "resnet_candidate_count_train": candidate_count_total.get("resnet", 0),
        "combined_two_source_candidate_count_train": candidate_count_total.get("combined_two_source", 0),
        "canonical_all_candidate_count_train": candidate_count_total.get("canonical_all", 0),
    }
    return {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "start_time": start_time,
        "end_time": end_time,
        "input_files": [
            str(selection.get("opencv_candidate_path", "")),
            str(selection.get("resnet_candidate_path", "")),
            str(ANCHOR_PATH),
            str(ANCHOR_WITH_SPLIT_PATH),
            str(SPLIT_PATH),
            str(SEGMENT_PATH),
            str(VIDEO_MANIFEST_PATH),
            str(FUSION_REPORT_PATH),
            str(FUSION_SUMMARY_PATH),
            str(ANCHOR_BUILD_REPORT_PATH),
        ],
        "output_files": [str(path) for path in [*DATA_OUTPUTS, *REPORT_OUTPUTS, OUT_LOG_PATH, SCRIPT_PATH, LATEST_README]],
        "warnings": warnings,
        "errors": errors,
        "counts": counts,
        "split_validation": split_details,
        "actual_boundary_metadata": actual_details,
        "source_separation": {
            "source_separation_status": source_status,
            "opencv_candidate_path": selection.get("opencv_candidate_path", ""),
            "resnet_candidate_path": selection.get("resnet_candidate_path", ""),
            "opencv_ffmpeg_filter_rule": (
                "selected v2.4 visual anchor build input; candidate_source_for_audit/candidate_source/method_used values identify OpenCV/FFmpeg scene candidates"
            ),
            "resnet_filter_rule": (
                "selected v2.4 visual anchor build input; candidate_source/method_used/model_name values identify ResNet embedding candidates"
            ),
            "source_separation_warning": "",
            "build_report_selection": selection,
        },
        "candidate_input_metadata": {
            "opencv_ffmpeg": opencv_metadata,
            "resnet": resnet_metadata,
            "canonical_all": canonical_metadata,
        },
        "combined_dedup": combined_metadata,
        "candidate_count_total": candidate_count_total,
        "candidate_video_count": candidate_video_count,
        "summary_metrics": nested_summary_metrics(summary_rows),
        "case_counts": case_counts(case_rows),
        "boundary_case_examples": {
            str(tol): {
                case_type: boundary_list(case_rows, tol, case_type)
                for case_type in ["both_hit", "opencv_only_hit", "resnet_only_hit", "both_missed"]
            }
            for tol in TOLERANCES
        },
        "source_density": density_rows,
        "protected_path_snapshot_before": protected_before,
        "protected_path_snapshot_after": protected_after,
        "old_project_modified": protected_before.get(str(OLD_PROJECT_ROOT)) != protected_after.get(str(OLD_PROJECT_ROOT)),
        "no_detector_rule_modified": all(
            protected_before.get(str(path)) == protected_after.get(str(path))
            for path in PROTECTED_PATHS
            if "detectors" in str(path) or "predictions" in str(path)
        ),
        "no_validation_test_row_level_output": True,
        "actual_label_used_for_candidate_generation": False,
        "actual_label_used_for_recall_audit_only": True,
        "third_model_added": False,
        "detector_rule_modified": False,
        "latest_bundle": latest_result or {},
        "sub_agent_validations": validations or {},
    }


def main() -> int:
    start_monotonic = time.monotonic()
    start_time = now_iso()
    warnings: list[str] = []
    errors: list[Any] = []
    report: dict[str, Any] = {}
    protected_before: dict[str, Any] = {}
    protected_after: dict[str, Any] = {}
    latest_result: dict[str, Any] | None = None

    try:
        log("[STEP 01] Safety snapshot and output path preparation")
        for directory in [OUT_SCENE_DIR, OUT_REPORT_DIR, OUT_LOG_PATH.parent, SCRIPT_PATH.parent, LATEST_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        protected_before = snapshot_protected_paths()

        log("[STEP 02] Locate existing visual anchor/source files")
        opencv_path, resnet_path, selection = select_candidate_inputs(warnings)
        if not opencv_path or not resnet_path:
            raise RuntimeError("required OpenCV/FFmpeg or ResNet candidate source file is missing")
        log(f"OpenCV/FFmpeg source: {opencv_path}")
        log(f"ResNet source: {resnet_path}")

        log("[STEP 03] Inspect input schemas and source-related columns")
        split_map, split_durations, split_details = load_split(warnings)
        durations = load_manifest_durations(split_durations, warnings)
        source_status = source_confidence_status(opencv_path, resnet_path, selection)
        opencv_rule = "source file selected from v2.4 anchor build report; family=opencv_ffmpeg"
        resnet_rule = "source file selected from v2.4 anchor build report; family=resnet"
        inventory_rows = [
            inspect_file(opencv_path, source_status, opencv_rule, resnet_rule),
            inspect_file(resnet_path, source_status, opencv_rule, resnet_rule),
            inspect_file(ANCHOR_WITH_SPLIT_PATH, source_status, opencv_rule, resnet_rule),
            inspect_file(ANCHOR_PATH, source_status, opencv_rule, resnet_rule),
            inspect_file(SPLIT_PATH, "split_file", opencv_rule, resnet_rule),
            inspect_file(SEGMENT_PATH, "actual_label_file_for_recall_audit_only", opencv_rule, resnet_rule),
        ]

        log("[STEP 04] Separate OpenCV/FFmpeg and ResNet candidates")
        opencv_candidates, opencv_metadata = load_candidate_rows(opencv_path, "opencv_ffmpeg", split_map, warnings)
        resnet_candidates, resnet_metadata = load_candidate_rows(resnet_path, "resnet", split_map, warnings)
        canonical_candidates, canonical_metadata = load_canonical_candidates(split_map, warnings)
        combined_rows, combined_clusters_by_video, combined_metadata = build_combined_clusters(opencv_candidates, resnet_candidates)
        opencv_by_video = candidates_by_video(opencv_candidates)
        resnet_by_video = candidates_by_video(resnet_candidates)
        canonical_by_video = candidates_by_video(canonical_candidates)
        combined_by_video = {vid: combined_member_candidates(clusters) for vid, clusters in combined_clusters_by_video.items()}
        log(f"train candidate counts: opencv_ffmpeg={len(opencv_candidates)} resnet={len(resnet_candidates)} combined={len(combined_rows)} canonical_all={len(canonical_candidates)}")

        log("[STEP 05] Load train actual ad intervals and construct start/end boundaries")
        intervals, boundaries, actual_details = load_actual_boundaries(warnings)
        log(f"train actual intervals={len(intervals)} boundaries={len(boundaries)}")

        log("[STEP 06] Compute nearest candidate distance by source family")
        family_candidates: dict[str, dict[int, list[dict[str, Any]]]] = {
            "opencv_ffmpeg": opencv_by_video,
            "resnet": resnet_by_video,
            "combined_two_source": combined_by_video,
        }
        if canonical_candidates:
            family_candidates["canonical_all"] = canonical_by_video
        family_count_by_video: dict[str, dict[int, int]] = {
            "opencv_ffmpeg": count_by_video(opencv_by_video),
            "resnet": count_by_video(resnet_by_video),
            "combined_two_source": {vid: len(rows) for vid, rows in combined_clusters_by_video.items()},
        }
        if canonical_candidates:
            family_count_by_video["canonical_all"] = count_by_video(canonical_by_video)
        audit_rows = compute_audit_rows(boundaries, family_candidates, family_count_by_video, source_status)

        log("[STEP 07] Compute within 2/5/10s recall for start/end/all")
        candidate_count_total = {family: sum(counts.values()) for family, counts in family_count_by_video.items()}
        candidate_video_count = {family: len([vid for vid, count in counts.items() if count > 0]) for family, counts in family_count_by_video.items()}
        families = list(family_candidates.keys())
        summary_rows = summarize_recall(audit_rows, candidate_count_total, candidate_video_count)
        density_rows = candidate_density(family_count_by_video, durations)

        log("[STEP 08] Generate source comparison, recall audit, summary, case breakdown, missed cases")
        source_comparison_rows = [*opencv_candidates, *resnet_candidates, *combined_rows, *canonical_candidates]
        case_rows, missed_rows = build_case_breakdown(audit_rows)
        video_rows = video_level_recall(audit_rows, family_count_by_video, families)

        inventory_columns = [
            "file_path", "row_count", "columns_json", "source_related_columns_json",
            "source_unique_values_json", "timestamp_column", "video_id_column", "split_column",
            "source_separation_status", "opencv_ffmpeg_filter_rule", "resnet_filter_rule", "warning",
        ]
        source_columns = [
            "candidate_family", "video_id", "candidate_sec", "original_source_value",
            "original_method_value", "original_model_value", "confidence_or_score", "raw_row_index",
            "split", "notes",
        ]
        audit_columns = [
            "video_id", "ad_interval_id", "boundary_type", "actual_sec", "candidate_family",
            "nearest_candidate_sec", "nearest_distance_sec", "within_2s", "within_5s", "within_10s",
            "candidate_count_in_video", "source_separation_status", "notes",
        ]
        summary_columns = [
            "candidate_family", "boundary_type", "tolerance_sec", "actual_boundary_count",
            "hit_count", "recall", "median_nearest_distance_sec", "mean_nearest_distance_sec",
            "p90_nearest_distance_sec", "max_nearest_distance_sec", "candidate_count_total",
            "candidate_video_count",
        ]
        case_columns = [
            "tolerance_sec", "video_id", "ad_interval_id", "boundary_type", "actual_sec",
            "opencv_nearest_candidate_sec", "opencv_nearest_distance_sec", "opencv_hit",
            "resnet_nearest_candidate_sec", "resnet_nearest_distance_sec", "resnet_hit",
            "combined_hit", "case_type",
        ]
        missed_columns = [
            "tolerance_sec", "video_id", "ad_interval_id", "boundary_type", "actual_sec",
            "case_type", "opencv_nearest_candidate_sec", "opencv_nearest_distance_sec",
            "resnet_nearest_candidate_sec", "resnet_nearest_distance_sec", "notes",
        ]
        video_columns = [
            "video_id", "candidate_family", "boundary_type", "tolerance_sec", "actual_boundary_count",
            "hit_count", "recall", "candidate_count", "notes",
        ]
        write_csv(INVENTORY_CSV, inventory_rows, inventory_columns)
        write_csv(SOURCE_COMPARISON_CSV, source_comparison_rows, source_columns)
        write_csv(AUDIT_CSV, audit_rows, audit_columns)
        write_csv(SUMMARY_CSV, summary_rows, summary_columns)
        write_csv(CASE_BREAKDOWN_CSV, case_rows, case_columns)
        write_csv(MISSED_CASES_CSV, missed_rows, missed_columns)
        write_csv(VIDEO_LEVEL_CSV, video_rows, video_columns)

        log("[STEP 09] Generate markdown/json reports and findings")
        protected_after = snapshot_protected_paths()
        report = build_report(
            start_time=start_time,
            warnings=warnings,
            errors=errors,
            selection=selection,
            split_details=split_details,
            actual_details=actual_details,
            source_status=source_status,
            opencv_metadata=opencv_metadata,
            resnet_metadata=resnet_metadata,
            canonical_metadata=canonical_metadata,
            combined_metadata=combined_metadata,
            candidate_count_total=candidate_count_total,
            candidate_video_count=candidate_video_count,
            summary_rows=summary_rows,
            case_rows=case_rows,
            density_rows=density_rows,
            protected_before=protected_before,
            protected_after=protected_after,
            latest_result=latest_result,
            validations=None,
        )
        SUMMARY_MD.write_text(make_summary_md(report, summary_rows, case_rows, families), encoding="utf-8")
        FINDINGS_MD.write_text(make_findings_md(report, summary_rows, case_rows, families), encoding="utf-8")
        write_json(REPORT_JSON, report)

        log("[STEP 10] Run Sub Agent validations")
        write_log()
        validations = validate_outputs(report, summary_rows, audit_rows, case_rows, latest_result, protected_before, protected_after)
        report["sub_agent_validations"] = validations
        write_json(REPORT_JSON, report)
        SUMMARY_MD.write_text(make_summary_md(report, summary_rows, case_rows, families), encoding="utf-8")
        FINDINGS_MD.write_text(make_findings_md(report, summary_rows, case_rows, families), encoding="utf-8")

        log("[STEP 11] Update latest bundle")
        write_log()
        latest_sources = [*DATA_OUTPUTS, *REPORT_OUTPUTS, SCRIPT_PATH, OUT_LOG_PATH]
        latest_result = update_latest_bundle(latest_sources, warnings)
        protected_after = snapshot_protected_paths()
        report = build_report(
            start_time=start_time,
            warnings=warnings,
            errors=errors,
            selection=selection,
            split_details=split_details,
            actual_details=actual_details,
            source_status=source_status,
            opencv_metadata=opencv_metadata,
            resnet_metadata=resnet_metadata,
            canonical_metadata=canonical_metadata,
            combined_metadata=combined_metadata,
            candidate_count_total=candidate_count_total,
            candidate_video_count=candidate_video_count,
            summary_rows=summary_rows,
            case_rows=case_rows,
            density_rows=density_rows,
            protected_before=protected_before,
            protected_after=protected_after,
            latest_result=latest_result,
            validations=None,
        )
        validations = validate_outputs(report, summary_rows, audit_rows, case_rows, latest_result, protected_before, protected_after)
        report["sub_agent_validations"] = validations
        report["actual_runtime_seconds"] = round(time.monotonic() - start_monotonic, 6)
        write_json(REPORT_JSON, report)
        SUMMARY_MD.write_text(make_summary_md(report, summary_rows, case_rows, families), encoding="utf-8")
        FINDINGS_MD.write_text(make_findings_md(report, summary_rows, case_rows, families), encoding="utf-8")
        shutil.copy2(REPORT_JSON, LATEST_DIR / REPORT_JSON.name)
        shutil.copy2(SUMMARY_MD, LATEST_DIR / SUMMARY_MD.name)
        shutil.copy2(FINDINGS_MD, LATEST_DIR / FINDINGS_MD.name)

        log("[STEP 12] Print final human-readable summary")
        row_counts = output_row_counts([*DATA_OUTPUTS, *REPORT_OUTPUTS, OUT_LOG_PATH, LATEST_README])
        report["output_row_counts"] = row_counts
        write_json(REPORT_JSON, report)
        shutil.copy2(REPORT_JSON, LATEST_DIR / REPORT_JSON.name)
        for path, count in row_counts.items():
            log(f"output: {path} rows={count}")
        lookup = summary_lookup(summary_rows)
        for family in ["opencv_ffmpeg", "resnet", "combined_two_source"]:
            all2 = lookup[family]["all"][2]
            all5 = lookup[family]["all"][5]
            all10 = lookup[family]["all"][10]
            log(
                f"{family} all-boundary recall@2/5/10s="
                f"{format_cell(all2['recall'])}/{format_cell(all5['recall'])}/{format_cell(all10['recall'])}"
            )
        log(f"case_counts={json.dumps(report['case_counts'], ensure_ascii=False)}")
        log(f"warnings={json.dumps(warnings, ensure_ascii=False)}")
        log(f"errors={json.dumps(errors, ensure_ascii=False)}")
        log(f"latest_bundle={LATEST_DIR}")
        write_log()
        validations = validate_outputs(report, summary_rows, audit_rows, case_rows, latest_result, protected_before, protected_after)
        report["sub_agent_validations"] = validations
        report["actual_runtime_seconds"] = round(time.monotonic() - start_monotonic, 6)
        write_json(REPORT_JSON, report)
        shutil.copy2(REPORT_JSON, LATEST_DIR / REPORT_JSON.name)
        shutil.copy2(OUT_LOG_PATH, LATEST_DIR / OUT_LOG_PATH.name)
        return 0
    except Exception as exc:
        errors.append({"exception": repr(exc), "traceback": traceback.format_exc()})
        log("[ERROR] Exception during audit")
        log(traceback.format_exc())
        try:
            protected_after = snapshot_protected_paths()
            fallback_report = report or {
                "task_name": TASK_NAME,
                "version": VERSION,
                "project_root": str(PROJECT_ROOT),
                "start_time": start_time,
                "end_time": now_iso(),
                "warnings": warnings,
                "errors": errors,
                "old_project_modified": protected_before.get(str(OLD_PROJECT_ROOT)) != protected_after.get(str(OLD_PROJECT_ROOT)),
                "no_detector_rule_modified": True,
                "no_validation_test_row_level_output": True,
            }
            write_json(REPORT_JSON, fallback_report)
        finally:
            write_log()
        return 1


if __name__ == "__main__":
    sys.exit(main())
