#!/usr/bin/env python3
"""Inspect existing scene-boundary candidate inputs for a later OCR extraction.

This script is intentionally read-only for existing candidate/anchor/label/split
assets. It only writes the OCR input inventory, recommendation report, run log,
and latest-file bundles requested for this inspection task.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(".")
VERSION = "v2_4"
TASK_NAME = "scene_boundary_inputs_for_ocr_v2_4"

SEARCH_DIRS = [
    PROJECT_ROOT / "data/scene",
    PROJECT_ROOT / "data/features",
    PROJECT_ROOT / "data/review",
    PROJECT_ROOT / "reports/scene",
    PROJECT_ROOT / "reports/fusion",
    PROJECT_ROOT / "scripts/scene",
    PROJECT_ROOT / "scripts/fusion",
    PROJECT_ROOT / "scripts/ocr",
    PROJECT_ROOT / "scripts/review",
]

KEYWORDS = [
    "scene",
    "boundary",
    "visual_anchor",
    "opencv",
    "ffmpeg",
    "resnet",
    "canonical",
    "anchor",
]

OPENCV_FFMPEG_CANDIDATE = PROJECT_ROOT / "data/scene/scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed.csv"
RESNET_RAW_CANDIDATE = PROJECT_ROOT / "data/scene/resnet_scene_candidates_v2_4_labelrefreshed.csv"
RESNET_REVIEW_ANALYSIS = PROJECT_ROOT / "data/review/resnet_scene_candidate_human_review_v2_4_usefulness_analysis.csv"
CANONICAL_ANCHOR = PROJECT_ROOT / "data/features/visual_scene_boundary_anchors_v2_4.csv"
CANONICAL_ANCHOR_WITH_SPLIT = PROJECT_ROOT / "data/features/visual_scene_boundary_anchors_v2_4_with_split.csv"
SPLIT_FILE = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"

REFERENCE_REPORTS = [
    PROJECT_ROOT / "reports/fusion/visual_anchor_alignment_pack_v2_4_report.json",
    PROJECT_ROOT / "reports/fusion/visual_anchor_alignment_pack_v2_4_summary.md",
    PROJECT_ROOT / "reports/scene/opencv_resnet_scene_boundary_recall_audit_v2_4_report.json",
    PROJECT_ROOT / "reports/scene/opencv_resnet_scene_boundary_recall_audit_v2_4_summary.md",
    PROJECT_ROOT / "reports/scene/opencv_resnet_scene_boundary_recall_audit_v2_4_findings.md",
]

SCRIPT_PATH = PROJECT_ROOT / "scripts/ocr/inspect_scene_boundary_inputs_for_ocr_v2_4.py"
DATA_OCR_DIR = PROJECT_ROOT / "data/ocr"
REPORTS_OCR_DIR = PROJECT_ROOT / "reports/ocr"
LOGS_DIR = PROJECT_ROOT / "logs"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_scene_boundary_inputs_for_ocr_v2_4"
LATEST_OCR_DIR = PROJECT_ROOT / "outputs/latest_ocr"

INVENTORY_CSV = DATA_OCR_DIR / "scene_boundary_input_inventory_for_ocr_v2_4.csv"
RECOMMENDATION_CSV = DATA_OCR_DIR / "recommended_scene_boundary_input_for_ocr_v2_4.csv"
SUMMARY_MD = REPORTS_OCR_DIR / "scene_boundary_inputs_for_ocr_v2_4_summary.md"
REPORT_JSON = REPORTS_OCR_DIR / "scene_boundary_inputs_for_ocr_v2_4_report.json"
RUN_LOG = LOGS_DIR / "scene_boundary_inputs_for_ocr_v2_4_run_log.txt"
LATEST_README = LATEST_DIR / "README_latest_files.md"
LATEST_OCR_README = LATEST_OCR_DIR / "README_latest_files.md"

INVENTORY_COLUMNS = [
    "candidate_group",
    "file_path",
    "file_exists",
    "row_count",
    "likely_scope",
    "video_id_column",
    "timestamp_column",
    "split_column",
    "source_column",
    "method_column",
    "model_column",
    "score_column",
    "key_columns_json",
    "sample_rows_json",
    "usable_for_ocr_anchor",
    "reason",
    "notes",
]

RECOMMENDATION_COLUMNS = [
    "recommendation_rank",
    "usage_type",
    "recommended_file_path",
    "recommended_video_id_column",
    "recommended_timestamp_column",
    "recommended_split_column",
    "recommended_source_column",
    "expected_scope",
    "why_recommended",
    "caution",
    "example_ocr_usage",
]

KEY_COLUMN_CANDIDATES = [
    "scene_boundary_anchor_id",
    "video_id",
    "video_id_normalized_for_split",
    "split",
    "candidate_time_sec",
    "resnet_time_sec_std",
    "canonical_boundary_time_sec",
    "boundary_sec",
    "candidate_sec",
    "anchor_sec",
    "timestamp_sec",
    "candidate_source",
    "candidate_source_for_audit",
    "canonical_time_source",
    "method_used",
    "model_name",
    "source_relation",
    "has_opencv_ffmpeg_candidate",
    "has_resnet_candidate",
    "source_count",
    "scene_change_score",
    "resnet_score_std",
    "visual_boundary_strength_score",
    "opencv_candidate_time_sec",
    "resnet_candidate_time_sec",
    "opencv_candidate_times_sec_list",
    "resnet_candidate_times_sec_list",
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
FORBIDDEN_BUNDLE_TOKENS = {"raw", "frame", "frames", "cache", "model", "weights", "checkpoint"}


LOG_LINES: list[str] = []


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"{now_iso()} {message}"
    LOG_LINES.append(line)
    print(message, flush=True)


def flush_log() -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def strip_bom(value: str) -> str:
    return value.lstrip("\ufeff")


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text


def compact_value(value: Any, max_len: int = 180) -> str:
    text = clean_cell(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: format_for_csv(row.get(col, "")) for col in columns})


def format_for_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = [strip_bom(c or "") for c in (reader.fieldnames or [])]
        rows: list[dict[str, str]] = []
        for raw in reader:
            rows.append({strip_bom(k or ""): clean_cell(v) for k, v in raw.items()})
    return columns, rows


def find_col(columns: list[str], candidates: list[str]) -> str:
    direct = {c: c for c in columns}
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in direct:
            return direct[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return ""


def to_int(value: Any) -> int | None:
    text = clean_cell(value)
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def load_split_map() -> dict[int, str]:
    if not SPLIT_FILE.exists():
        return {}
    columns, rows = read_csv_rows(SPLIT_FILE)
    video_col = find_col(columns, ["video_id"])
    split_col = find_col(columns, ["split"])
    split_map: dict[int, str] = {}
    if not video_col or not split_col:
        return split_map
    for row in rows:
        video_id = to_int(row.get(video_col))
        if video_id is not None:
            split_map[video_id] = row.get(split_col, "")
    return split_map


def file_stats(paths: list[Path]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.exists():
            stat = path.stat()
            stats[str(path)] = {
                "exists": True,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        else:
            stats[str(path)] = {"exists": False, "size": None, "mtime_ns": None}
    return stats


def stats_changed(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    changed: list[str] = []
    for path, before_stat in before.items():
        after_stat = after.get(path)
        if before_stat != after_stat:
            changed.append(path)
    return changed


def discover_files() -> list[Path]:
    discovered: set[Path] = set()
    for directory in SEARCH_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            name = path.name.lower()
            if any(keyword in name for keyword in KEYWORDS):
                discovered.add(path)
    for path in [
        OPENCV_FFMPEG_CANDIDATE,
        RESNET_RAW_CANDIDATE,
        RESNET_REVIEW_ANALYSIS,
        CANONICAL_ANCHOR,
        CANONICAL_ANCHOR_WITH_SPLIT,
        *REFERENCE_REPORTS,
    ]:
        discovered.add(path)
    return sorted(discovered, key=lambda p: rel(p))


def classify_group(path: Path) -> str:
    name = path.name.lower()
    path_text = rel(path).lower()
    if "visual_scene_boundary_anchors_v2_4" in name:
        return "canonical_visual_anchor"
    if (
        "opencv_resnet" in name
        or "visual_anchor_alignment_pack" in name
        or "recall_audit" in name
        or "case_breakdown" in name
        or "missed_boundary" in name
        or "source_inventory" in name
        or "source_comparison" in name
        or "video_level_recall" in name
    ):
        return "opencv_resnet_audit_reference"
    if path == OPENCV_FFMPEG_CANDIDATE or (
        path_text.startswith("data/scene/")
        and ("opencv" in name or "ffmpeg" in name or "merged_ffmpeg_fallback" in name)
        and "audit" not in name
        and "summary" not in name
        and "comparison" not in name
        and "overlap" not in name
    ):
        return "opencv_ffmpeg_raw"
    if path == RESNET_RAW_CANDIDATE or path == RESNET_REVIEW_ANALYSIS or (
        "resnet_scene_candidate" in name
        and "summary" not in name
        and "audit" not in name
        and "comparison" not in name
    ):
        return "resnet_raw"
    return "unknown_scene_related"


def timestamp_priority(candidate_group: str, columns: list[str], path: Path) -> list[str]:
    if candidate_group == "canonical_visual_anchor":
        return [
            "canonical_boundary_time_sec",
            "anchor_sec",
            "candidate_time_sec",
            "timestamp_sec",
            "boundary_sec",
        ]
    if candidate_group == "resnet_raw" and path == RESNET_REVIEW_ANALYSIS:
        return [
            "resnet_time_sec_std",
            "resnet_candidate_time_sec",
            "candidate_time_sec",
            "timestamp_sec",
            "candidate_sec",
        ]
    if candidate_group == "resnet_raw":
        return [
            "candidate_time_sec",
            "resnet_time_sec_std",
            "resnet_candidate_time_sec",
            "timestamp_sec",
            "candidate_sec",
        ]
    if candidate_group == "opencv_ffmpeg_raw":
        return [
            "candidate_time_sec",
            "opencv_candidate_time_sec",
            "timestamp_sec",
            "candidate_sec",
            "boundary_sec",
        ]
    return [
        "timestamp_sec",
        "candidate_sec",
        "candidate_time_sec",
        "anchor_sec",
        "boundary_sec",
        "canonical_boundary_time_sec",
        "resnet_time_sec_std",
        "opencv_candidate_time_sec",
        "resnet_candidate_time_sec",
    ]


def detect_columns(path: Path, candidate_group: str, columns: list[str]) -> dict[str, str]:
    if candidate_group == "canonical_visual_anchor":
        source_candidates = ["source_relation", "canonical_time_source", "candidate_source", "source"]
        method_candidates = ["canonical_time_source", "method_used", "source_relation", "method"]
        score_candidates = ["visual_boundary_strength_score", "scene_change_score", "resnet_scene_change_score", "opencv_scene_change_score", "score"]
    elif candidate_group == "resnet_raw":
        source_candidates = ["candidate_source_for_audit", "candidate_source", "source_relation", "source"]
        method_candidates = ["method_used", "resnet_method_used", "method"]
        score_candidates = ["resnet_score_std", "scene_change_score", "resnet_scene_change_score", "cosine_distance", "score"]
    elif candidate_group == "opencv_ffmpeg_raw":
        source_candidates = ["candidate_source_for_audit", "candidate_source", "source_relation", "source"]
        method_candidates = ["method_used", "opencv_method_used", "method"]
        score_candidates = ["scene_change_score", "opencv_scene_change_score", "score"]
    else:
        source_candidates = ["source_relation", "candidate_source_for_audit", "candidate_source", "canonical_time_source", "source"]
        method_candidates = ["method_used", "canonical_time_source", "method", "source_relation"]
        score_candidates = [
            "visual_boundary_strength_score",
            "scene_change_score",
            "resnet_score_std",
            "opencv_scene_change_score",
            "resnet_scene_change_score",
            "score",
        ]
    return {
        "video_id_column": find_col(columns, ["video_id", "video_id_normalized_for_split"]),
        "timestamp_column": find_col(columns, timestamp_priority(candidate_group, columns, path)),
        "split_column": find_col(columns, ["split"]),
        "source_column": find_col(columns, source_candidates),
        "method_column": find_col(columns, method_candidates),
        "model_column": find_col(columns, ["model_name", "model"]),
        "score_column": find_col(columns, score_candidates),
    }


def count_values(rows: list[dict[str, str]], column: str, limit: int = 12) -> dict[str, int]:
    if not column:
        return {}
    counts = Counter(row.get(column, "") for row in rows)
    return dict(counts.most_common(limit))


def split_scope(
    path: Path,
    rows: list[dict[str, str]],
    video_col: str,
    split_col: str,
    split_map: dict[int, str],
) -> tuple[str, dict[str, int]]:
    counts: Counter[str] = Counter()
    if split_col:
        for row in rows:
            counts[row.get(split_col, "") or "blank"] += 1
    elif video_col and split_map:
        for row in rows:
            video_id = to_int(row.get(video_col))
            counts[split_map.get(video_id or -1, "unknown")] += 1

    split_counts = dict(counts)
    meaningful = {k: v for k, v in split_counts.items() if k not in {"", "blank", "unknown"} and v > 0}
    name = path.name.lower()
    if meaningful:
        has_train = meaningful.get("train", 0) > 0
        has_validation = meaningful.get("validation", 0) > 0
        has_test = meaningful.get("test", 0) > 0
        if has_train and has_validation and has_test:
            source = "explicit_split" if split_col else "inferred_from_video_id"
            return f"all_splits_{source}: train={meaningful.get('train', 0)}, validation={meaningful.get('validation', 0)}, test={meaningful.get('test', 0)}", split_counts
        if has_train and not has_validation and not has_test:
            source = "explicit_split" if split_col else "inferred_from_video_id"
            return f"train_only_{source}: train={meaningful.get('train', 0)}", split_counts
        return "partial_split_scope: " + json_dumps(meaningful), split_counts
    if "train_only" in name or name.endswith("_train.csv") or "_train." in name:
        return "train_only_inferred_from_filename", split_counts
    return "unknown_scope_no_split_column_or_mapping", split_counts


def sample_rows(rows: list[dict[str, str]], columns: list[str], max_rows: int = 3) -> list[dict[str, str]]:
    sample_columns = [col for col in KEY_COLUMN_CANDIDATES if col in columns]
    if not sample_columns:
        sample_columns = columns[:10]
    samples: list[dict[str, str]] = []
    for row in rows[:max_rows]:
        samples.append({col: compact_value(row.get(col, "")) for col in sample_columns})
    return samples


def inspect_csv(path: Path, candidate_group: str, split_map: dict[int, str]) -> dict[str, Any]:
    columns, rows = read_csv_rows(path)
    detected = detect_columns(path, candidate_group, columns)
    likely_scope, split_counts = split_scope(
        path,
        rows,
        detected["video_id_column"],
        detected["split_column"],
        split_map,
    )
    key_columns = [col for col in KEY_COLUMN_CANDIDATES if col in columns]
    source_counts = count_values(rows, detected["source_column"])
    method_counts = count_values(rows, detected["method_column"])
    model_counts = count_values(rows, detected["model_column"])
    score_col = detected["score_column"]
    timestamp_col = detected["timestamp_column"]

    usable, reason = ocr_usability(candidate_group, path, detected, likely_scope)
    notes_bits = [
        f"columns={len(columns)}",
        f"split_counts={json_dumps(split_counts)}",
    ]
    if source_counts:
        notes_bits.append(f"source_counts={json_dumps(source_counts)}")
    if method_counts:
        notes_bits.append(f"method_counts={json_dumps(method_counts)}")
    if model_counts:
        notes_bits.append(f"model_counts={json_dumps(model_counts)}")
    if score_col:
        notes_bits.append(f"score_column={score_col}")
    if timestamp_col:
        notes_bits.append(f"timestamp_column={timestamp_col}")

    return {
        "candidate_group": candidate_group,
        "file_path": str(path),
        "file_exists": True,
        "row_count": len(rows),
        "likely_scope": likely_scope,
        "video_id_column": detected["video_id_column"],
        "timestamp_column": timestamp_col,
        "split_column": detected["split_column"],
        "source_column": detected["source_column"],
        "method_column": detected["method_column"],
        "model_column": detected["model_column"],
        "score_column": score_col,
        "key_columns_json": json_dumps(key_columns),
        "sample_rows_json": json_dumps(sample_rows(rows, columns)),
        "usable_for_ocr_anchor": usable,
        "reason": reason,
        "notes": "; ".join(notes_bits),
    }


def inspect_non_csv(path: Path, candidate_group: str) -> dict[str, Any]:
    key_info: dict[str, Any] = {}
    if path.suffix.lower() == ".json" and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                key_info["top_level_keys"] = sorted(data.keys())[:30]
                for key in ["input_files", "source_separation", "canonical_anchor", "split_scope"]:
                    if key in data:
                        key_info[key] = data[key]
        except Exception as exc:  # pragma: no cover - defensive report path only
            key_info["json_error"] = str(exc)
    usable = "reference_only" if candidate_group == "opencv_resnet_audit_reference" else "no"
    reason = "Report/script reference; not a row-level OCR anchor input."
    return {
        "candidate_group": candidate_group,
        "file_path": str(path),
        "file_exists": path.exists(),
        "row_count": "",
        "likely_scope": "reference_file_not_row_level_csv",
        "video_id_column": "",
        "timestamp_column": "",
        "split_column": "",
        "source_column": "",
        "method_column": "",
        "model_column": "",
        "score_column": "",
        "key_columns_json": json_dumps(key_info),
        "sample_rows_json": "[]",
        "usable_for_ocr_anchor": usable,
        "reason": reason,
        "notes": f"suffix={path.suffix.lower() or '(none)'}",
    }


def ocr_usability(candidate_group: str, path: Path, detected: dict[str, str], likely_scope: str) -> tuple[str, str]:
    has_video_time = bool(detected["video_id_column"] and detected["timestamp_column"])
    if candidate_group == "canonical_visual_anchor" and path == CANONICAL_ANCHOR_WITH_SPLIT and has_video_time and detected["split_column"]:
        return (
            "yes_primary",
            "Canonical visual anchor with split and canonical timestamp; already merges OpenCV/FFmpeg and ResNet candidates within 2s.",
        )
    if candidate_group == "canonical_visual_anchor" and has_video_time and detected["split_column"]:
        return (
            "limited",
            "Canonical visual anchor has split and timestamp, but this is not the all-split primary with_split file.",
        )
    if candidate_group == "canonical_visual_anchor" and has_video_time:
        return (
            "yes_after_split_join",
            "Canonical visual anchor is row-level and deduplicated, but this version lacks an explicit split column.",
        )
    if candidate_group in {"opencv_ffmpeg_raw", "resnet_raw"} and has_video_time:
        return (
            "yes_debug_or_source_specific",
            "Usable for source-specific OCR/debug analysis, but less suitable as the primary anchor because cross-source duplicates are not collapsed.",
        )
    if candidate_group == "opencv_resnet_audit_reference":
        return ("reference_only", "Audit/report reference for validation, not the preferred OCR extraction anchor.")
    if "train_only" in likely_scope:
        return ("limited", "Train-only or audit-derived file; avoid as all-split OCR extraction input.")
    return ("no", "Missing row-level video_id/timestamp columns or not a candidate/anchor table.")


def inspect_file(path: Path, split_map: dict[int, str]) -> dict[str, Any]:
    candidate_group = classify_group(path)
    if not path.exists():
        return {
            "candidate_group": candidate_group,
            "file_path": str(path),
            "file_exists": False,
            "row_count": "",
            "likely_scope": "missing",
            "video_id_column": "",
            "timestamp_column": "",
            "split_column": "",
            "source_column": "",
            "method_column": "",
            "model_column": "",
            "score_column": "",
            "key_columns_json": "[]",
            "sample_rows_json": "[]",
            "usable_for_ocr_anchor": "no",
            "reason": "File does not exist.",
            "notes": "",
        }
    if path.suffix.lower() == ".csv":
        try:
            return inspect_csv(path, candidate_group, split_map)
        except Exception as exc:
            row = inspect_non_csv(path, candidate_group)
            row["reason"] = f"CSV inspection failed: {exc}"
            return row
    return inspect_non_csv(path, candidate_group)


def row_by_path(inventory: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    target = str(path)
    for row in inventory:
        if row["file_path"] == target:
            return row
    return {}


def build_recommendations(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical = row_by_path(inventory, CANONICAL_ANCHOR_WITH_SPLIT)
    opencv = row_by_path(inventory, OPENCV_FFMPEG_CANDIDATE)
    resnet_raw = row_by_path(inventory, RESNET_RAW_CANDIDATE)
    resnet_review = row_by_path(inventory, RESNET_REVIEW_ANALYSIS)
    canonical_no_split = row_by_path(inventory, CANONICAL_ANCHOR)
    audit_summary = row_by_path(inventory, PROJECT_ROOT / "reports/scene/opencv_resnet_scene_boundary_recall_audit_v2_4_summary.md")

    return [
        {
            "recommendation_rank": 1,
            "usage_type": "primary_ocr_anchor",
            "recommended_file_path": str(CANONICAL_ANCHOR_WITH_SPLIT),
            "recommended_video_id_column": canonical.get("video_id_column", "video_id"),
            "recommended_timestamp_column": "canonical_boundary_time_sec",
            "recommended_split_column": canonical.get("split_column", "split"),
            "recommended_source_column": "source_relation",
            "expected_scope": canonical.get("likely_scope", "all_splits_explicit_split"),
            "why_recommended": "Canonical visual anchor already combines OpenCV/FFmpeg and ResNet evidence using the existing 2s merge rule and includes split for train/validation/test handling.",
            "caution": "Use label/audit columns only for analysis. The anchor window N seconds should be set in the follow-up OCR extraction job, not fixed here.",
            "example_ocr_usage": (
                f"--anchor-file {CANONICAL_ANCHOR_WITH_SPLIT} "
                "--video-id-column video_id --timestamp-column canonical_boundary_time_sec "
                "--split-column split --source-column source_relation "
                "--near-anchor-interval-sec 1.0 --background-interval-sec 1.5 "
                "--anchor-window-sec <set_in_followup_ocr_extraction>"
            ),
        },
        {
            "recommendation_rank": 2,
            "usage_type": "source_specific_debug",
            "recommended_file_path": str(OPENCV_FFMPEG_CANDIDATE),
            "recommended_video_id_column": opencv.get("video_id_column", "video_id"),
            "recommended_timestamp_column": opencv.get("timestamp_column", "candidate_time_sec"),
            "recommended_split_column": opencv.get("split_column", "join_via_data/splits/video_split_v2_4.csv"),
            "recommended_source_column": opencv.get("source_column", "candidate_source_for_audit"),
            "expected_scope": opencv.get("likely_scope", "all_splits_inferred_from_video_id"),
            "why_recommended": "Use when OCR behavior must be inspected for OpenCV/FFmpeg-only visual scene candidates.",
            "caution": "No explicit split column in the raw table; join by video_id to the split file before split-aware OCR analysis.",
            "example_ocr_usage": (
                f"--candidate-file {OPENCV_FFMPEG_CANDIDATE} "
                "--video-id-column video_id --timestamp-column candidate_time_sec "
                "--split-map-file ./data/splits/video_split_v2_4.csv"
            ),
        },
        {
            "recommendation_rank": 3,
            "usage_type": "source_specific_debug",
            "recommended_file_path": str(RESNET_RAW_CANDIDATE),
            "recommended_video_id_column": resnet_raw.get("video_id_column", "video_id"),
            "recommended_timestamp_column": resnet_raw.get("timestamp_column", "candidate_time_sec"),
            "recommended_split_column": resnet_raw.get("split_column", "join_via_data/splits/video_split_v2_4.csv"),
            "recommended_source_column": resnet_raw.get("source_column", "candidate_source_for_audit"),
            "expected_scope": resnet_raw.get("likely_scope", "all_splits_inferred_from_video_id"),
            "why_recommended": "Use when OCR behavior must be inspected for raw ResNet embedding scene candidates.",
            "caution": "Higher candidate density and no cross-source 2s dedup; prefer canonical anchor for production OCR sampling.",
            "example_ocr_usage": (
                f"--candidate-file {RESNET_RAW_CANDIDATE} "
                "--video-id-column video_id --timestamp-column candidate_time_sec "
                "--split-map-file ./data/splits/video_split_v2_4.csv"
            ),
        },
        {
            "recommendation_rank": 4,
            "usage_type": "source_specific_debug",
            "recommended_file_path": str(RESNET_REVIEW_ANALYSIS),
            "recommended_video_id_column": resnet_review.get("video_id_column", "video_id"),
            "recommended_timestamp_column": resnet_review.get("timestamp_column", "resnet_time_sec_std"),
            "recommended_split_column": resnet_review.get("split_column", "join_via_data/splits/video_split_v2_4.csv"),
            "recommended_source_column": resnet_review.get("source_column", "candidate_source_for_audit"),
            "expected_scope": resnet_review.get("likely_scope", "all_splits_inferred_from_video_id"),
            "why_recommended": "Use when ResNet human-review/usefulness fields are needed next to candidate timestamps.",
            "caution": "Review-enriched audit table, not the cleanest raw extraction input; do not confuse review labels with OCR anchors.",
            "example_ocr_usage": (
                f"--candidate-file {RESNET_REVIEW_ANALYSIS} "
                "--video-id-column video_id --timestamp-column resnet_time_sec_std "
                "--split-map-file ./data/splits/video_split_v2_4.csv"
            ),
        },
        {
            "recommendation_rank": 5,
            "usage_type": "audit_reference_only",
            "recommended_file_path": str(PROJECT_ROOT / "reports/scene/opencv_resnet_scene_boundary_recall_audit_v2_4_summary.md"),
            "recommended_video_id_column": audit_summary.get("video_id_column", ""),
            "recommended_timestamp_column": audit_summary.get("timestamp_column", ""),
            "recommended_split_column": audit_summary.get("split_column", ""),
            "recommended_source_column": audit_summary.get("source_column", ""),
            "expected_scope": "train_only_recall_audit_reference",
            "why_recommended": "Use as evidence explaining why combined/canonical anchors are preferred over either source alone.",
            "caution": "Not a row-level OCR extraction file.",
            "example_ocr_usage": "Read as documentation only; do not pass this markdown report to OCR extraction.",
        },
        {
            "recommendation_rank": 6,
            "usage_type": "not_recommended",
            "recommended_file_path": str(CANONICAL_ANCHOR),
            "recommended_video_id_column": canonical_no_split.get("video_id_column", "video_id"),
            "recommended_timestamp_column": canonical_no_split.get("timestamp_column", "canonical_boundary_time_sec"),
            "recommended_split_column": canonical_no_split.get("split_column", "missing"),
            "recommended_source_column": "source_relation",
            "expected_scope": canonical_no_split.get("likely_scope", "all_splits_inferred_from_video_id"),
            "why_recommended": "Contains the same canonical anchors, but the with_split file is safer for split-aware OCR sampling.",
            "caution": "Prefer visual_scene_boundary_anchors_v2_4_with_split.csv unless a downstream script joins split separately.",
            "example_ocr_usage": "Use only after joining split by video_id; otherwise use the rank 1 file.",
        },
    ]


def parse_json_cell(row: dict[str, Any], key: str) -> Any:
    try:
        return json.loads(row.get(key, "") or "{}")
    except json.JSONDecodeError:
        return {}


def summarize_source_counts(row: dict[str, Any]) -> str:
    notes = row.get("notes", "")
    marker = "source_counts="
    if marker not in notes:
        return ""
    tail = notes.split(marker, 1)[1]
    if "; " in tail:
        tail = tail.split("; ", 1)[0]
    return tail


def build_report(inventory: list[dict[str, Any]], recommendations: list[dict[str, Any]], validations: dict[str, Any]) -> dict[str, Any]:
    rows = {row["file_path"]: row for row in inventory}
    canonical = rows.get(str(CANONICAL_ANCHOR_WITH_SPLIT), {})
    opencv = rows.get(str(OPENCV_FFMPEG_CANDIDATE), {})
    resnet_raw = rows.get(str(RESNET_RAW_CANDIDATE), {})
    resnet_review = rows.get(str(RESNET_REVIEW_ANALYSIS), {})
    canonical_no_split = rows.get(str(CANONICAL_ANCHOR), {})

    answers = {
        "opencv_ffmpeg_candidate_file": {
            "path": str(OPENCV_FFMPEG_CANDIDATE),
            "row_count": opencv.get("row_count", ""),
            "timestamp_column": opencv.get("timestamp_column", ""),
            "video_id_column": opencv.get("video_id_column", ""),
            "split_column": opencv.get("split_column", ""),
            "source_column": opencv.get("source_column", ""),
            "scope": opencv.get("likely_scope", ""),
        },
        "resnet_candidate_files": {
            "raw_path": str(RESNET_RAW_CANDIDATE),
            "raw_row_count": resnet_raw.get("row_count", ""),
            "raw_timestamp_column": resnet_raw.get("timestamp_column", ""),
            "review_analysis_path": str(RESNET_REVIEW_ANALYSIS),
            "review_analysis_row_count": resnet_review.get("row_count", ""),
            "review_analysis_timestamp_column": resnet_review.get("timestamp_column", ""),
            "scope": resnet_raw.get("likely_scope", resnet_review.get("likely_scope", "")),
        },
        "canonical_visual_anchor_files": {
            "primary_with_split_path": str(CANONICAL_ANCHOR_WITH_SPLIT),
            "primary_row_count": canonical.get("row_count", ""),
            "primary_timestamp_column": canonical.get("timestamp_column", ""),
            "primary_video_id_column": canonical.get("video_id_column", ""),
            "primary_split_column": canonical.get("split_column", ""),
            "primary_source_columns": [
                col
                for col in [
                    "canonical_time_source",
                    "source_relation",
                    "has_opencv_ffmpeg_candidate",
                    "has_resnet_candidate",
                    "source_count",
                ]
                if col in parse_json_cell(canonical, "key_columns_json")
            ],
            "no_split_path": str(CANONICAL_ANCHOR),
            "no_split_row_count": canonical_no_split.get("row_count", ""),
        },
        "recommended_ocr_input": {
            "primary_anchor_file": str(CANONICAL_ANCHOR_WITH_SPLIT),
            "video_id_column": "video_id",
            "anchor_timestamp_column": "canonical_boundary_time_sec",
            "split_column": "split",
            "source_column_for_analysis": "source_relation",
            "policy": "anchor timestamp +/- N seconds sampled every 1.0s; non-anchor background sampled every 1.5s; N is a follow-up OCR extraction parameter.",
        },
    }

    group_counts = Counter(row["candidate_group"] for row in inventory)
    output_files = [
        SCRIPT_PATH,
        INVENTORY_CSV,
        RECOMMENDATION_CSV,
        SUMMARY_MD,
        REPORT_JSON,
        RUN_LOG,
        LATEST_README,
        LATEST_OCR_README,
    ]

    return {
        "task": TASK_NAME,
        "version": VERSION,
        "generated_at": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "purpose": "Locate and inspect existing OpenCV/FFmpeg, ResNet, and canonical visual scene-boundary files for future OCR sampling. This task does not run OCR extraction.",
        "searched_paths": [str(path) for path in SEARCH_DIRS],
        "keywords": KEYWORDS,
        "split_file": str(SPLIT_FILE),
        "inventory_row_count": len(inventory),
        "inventory_group_counts": dict(group_counts),
        "answers": answers,
        "recommendations": recommendations,
        "key_input_rows": {
            "opencv_ffmpeg_raw": opencv,
            "resnet_raw": resnet_raw,
            "resnet_review_analysis": resnet_review,
            "canonical_visual_anchor_with_split": canonical,
            "canonical_visual_anchor_no_split": canonical_no_split,
        },
        "validations": validations,
        "safety": {
            "existing_candidate_anchor_label_split_detector_ocr_files_modified": False,
            "raw_video_frame_cache_model_files_created_or_modified": False,
            "transnetv2_conflict": False,
            "notes": "Only new OCR inspection outputs and latest bundles were written.",
        },
        "outputs": [str(path) for path in output_files],
    }


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_summary_md(report: dict[str, Any]) -> str:
    answers = report["answers"]
    opencv = answers["opencv_ffmpeg_candidate_file"]
    resnet = answers["resnet_candidate_files"]
    canonical = answers["canonical_visual_anchor_files"]
    recommended = answers["recommended_ocr_input"]
    validations = report["validations"]

    key_rows = [
        {
            "file": "OpenCV/FFmpeg raw",
            "path": opencv["path"],
            "rows": opencv["row_count"],
            "video_id": opencv["video_id_column"],
            "timestamp": opencv["timestamp_column"],
            "split": opencv["split_column"] or "(join by video_id)",
            "scope": opencv["scope"],
        },
        {
            "file": "ResNet raw",
            "path": resnet["raw_path"],
            "rows": resnet["raw_row_count"],
            "video_id": "video_id",
            "timestamp": resnet["raw_timestamp_column"],
            "split": "(join by video_id)",
            "scope": resnet["scope"],
        },
        {
            "file": "ResNet review/usefulness",
            "path": resnet["review_analysis_path"],
            "rows": resnet["review_analysis_row_count"],
            "video_id": "video_id",
            "timestamp": resnet["review_analysis_timestamp_column"],
            "split": "(join by video_id)",
            "scope": resnet["scope"],
        },
        {
            "file": "Canonical visual anchor",
            "path": canonical["primary_with_split_path"],
            "rows": canonical["primary_row_count"],
            "video_id": canonical["primary_video_id_column"],
            "timestamp": canonical["primary_timestamp_column"],
            "split": canonical["primary_split_column"],
            "scope": "train/validation/test all splits",
        },
        {
            "file": "Canonical visual anchor no split",
            "path": canonical["no_split_path"],
            "rows": canonical["no_split_row_count"],
            "video_id": "video_id",
            "timestamp": "canonical_boundary_time_sec",
            "split": "(join by video_id)",
            "scope": "all splits inferred by video_id; not primary",
        },
    ]

    lines = [
        "# Scene Boundary Inputs for OCR v2.4",
        "",
        "## 1. 작업 목적",
        "기존 OpenCV/FFmpeg 장면전환 후보, ResNet embedding 장면전환 후보, 두 후보를 2초 기준으로 병합한 canonical visual scene boundary anchor가 프로젝트 어디에 있고 어떤 column 구조인지 확인했다. 이번 작업은 OCR 추출 실행이 아니며 TransNetV2 작업, detector rule, 기존 candidate/anchor/label/split 파일 수정과 별개다.",
        "",
        "## 2. 탐색한 경로 목록",
    ]
    lines.extend([f"- {path}" for path in report["searched_paths"]])
    lines.extend(
        [
            "",
            "## 3. 발견한 핵심 파일",
            *md_table(key_rows, ["file", "path", "rows", "video_id", "timestamp", "split", "scope"]),
            "",
            "## 4. OCR 작업 추천",
            f"- primary anchor file: `{recommended['primary_anchor_file']}`",
            f"- video_id column: `{recommended['video_id_column']}`",
            f"- anchor timestamp column: `{recommended['anchor_timestamp_column']}`",
            f"- split column: `{recommended['split_column']}`",
            f"- source/relation column: `{recommended['source_column_for_analysis']}`",
            "- 추천 이유: canonical visual anchor는 OpenCV/FFmpeg와 ResNet 후보를 기존 2초 merge rule로 묶어 OCR 중복 추출을 줄일 수 있고, `_with_split` 파일은 train/validation/test split이 명시되어 있다.",
            "- OpenCV/FFmpeg와 ResNet을 따로 쓰는 경우: source별 debug, source별 recall/false-positive 분석, 또는 canonical merge 전후 비교가 필요할 때만 raw 후보를 함께 사용한다.",
            "",
            "## 5. OCR 연계 예시",
            "```bash",
            "python scripts/ocr/<followup_ocr_extraction_script>.py \\",
            f"  --anchor-file {recommended['primary_anchor_file']} \\",
            "  --video-id-column video_id \\",
            "  --timestamp-column canonical_boundary_time_sec \\",
            "  --split-column split \\",
            "  --source-column source_relation \\",
            "  --near-anchor-interval-sec 1.0 \\",
            "  --background-interval-sec 1.5 \\",
            "  --anchor-window-sec <set_in_followup_ocr_extraction>",
            "```",
            "",
            "OCR 추출 정책은 anchor timestamp ±N초 주변은 1초 간격, anchor 주변이 아닌 전체 구간은 1.5초 간격으로 기록한다. N초 값은 이번 작업에서 정하지 않고 후속 OCR extraction 작업에서 parameter로 지정한다.",
            "",
            "## 6. Column 메모",
            "- canonical timestamp: `canonical_boundary_time_sec`",
            "- canonical source 관련: `canonical_time_source`, `source_relation`, `has_opencv_ffmpeg_candidate`, `has_resnet_candidate`, `source_count`, `opencv_candidate_time_sec`, `resnet_candidate_time_sec`",
            "- OpenCV/FFmpeg raw timestamp/source: `candidate_time_sec`, `candidate_source_for_audit`, `candidate_source`, `method_used`, `scene_change_score`",
            "- ResNet raw timestamp/source: `candidate_time_sec`, `candidate_source_for_audit`, `candidate_source`, `method_used`, `model_name`, `scene_change_score`",
            "- ResNet review/usefulness timestamp/source: `resnet_time_sec_std`, `candidate_source`, `method_used`, `model_name`, `resnet_score_std`",
            "",
            "## 7. train/validation/test 사용 가능 여부",
            f"- canonical `_with_split`: {canonical['primary_row_count']} rows, split column 포함, train/validation/test 전체 사용 가능",
            f"- canonical no-split: {canonical['no_split_row_count']} rows, 같은 anchor이나 split column이 없어 `_with_split`보다 후순위",
            f"- OpenCV/FFmpeg raw: {opencv['row_count']} rows, split column 없음, `video_id`로 split file join 필요",
            f"- ResNet raw: {resnet['raw_row_count']} rows, split column 없음, `video_id`로 split file join 필요",
            "- recall audit report/data: train-only audit reference로 사용하고 OCR primary anchor로는 사용하지 않는다.",
            "",
            "## 8. 검증 결과",
        ]
    )
    for name, result in validations.items():
        status = result.get("status", "")
        detail = result.get("detail", "")
        lines.append(f"- {name}: {status} ({detail})")
    lines.extend(
        [
            "",
            "## 9. Safety",
            "- 기존 OpenCV/FFmpeg candidate 파일 수정 없음",
            "- 기존 ResNet candidate/review 파일 수정 없음",
            "- 기존 visual anchor 파일 수정 없음",
            "- 기존 OCR extraction/result 파일 수정 없음",
            "- detector rule, actual label/split 파일 수정 없음",
            "- raw video/frame/cache/model 파일 생성 또는 수정 없음",
            "- TransNetV2 작업과 충돌 없음",
            "",
            "## 10. 생성 파일",
            f"- {SCRIPT_PATH}",
            f"- {INVENTORY_CSV}",
            f"- {RECOMMENDATION_CSV}",
            f"- {SUMMARY_MD}",
            f"- {REPORT_JSON}",
            f"- {RUN_LOG}",
            f"- {LATEST_DIR}",
            f"- {LATEST_OCR_DIR}",
            "",
        ]
    )
    return "\n".join(lines)


def validate_outputs(inventory: list[dict[str, Any]], recommendations: list[dict[str, Any]], before_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    after_stats = file_stats(list(Path(path) for path in before_stats.keys()))
    changed_inputs = stats_changed(before_stats, after_stats)
    rows_by_path = {row["file_path"]: row for row in inventory}
    canonical = rows_by_path.get(str(CANONICAL_ANCHOR_WITH_SPLIT), {})
    opencv = rows_by_path.get(str(OPENCV_FFMPEG_CANDIDATE), {})
    resnet = rows_by_path.get(str(RESNET_RAW_CANDIDATE), {})

    validation: dict[str, dict[str, str]] = {}
    validation["file_discovery_validation"] = {
        "status": "PASS" if opencv.get("file_exists") and resnet.get("file_exists") and canonical.get("file_exists") else "FAIL",
        "detail": "OpenCV/FFmpeg, ResNet, and canonical anchor files found.",
    }
    schema_ok = (
        canonical.get("video_id_column") == "video_id"
        and canonical.get("timestamp_column") == "canonical_boundary_time_sec"
        and canonical.get("split_column") == "split"
        and opencv.get("timestamp_column") == "candidate_time_sec"
        and resnet.get("timestamp_column") == "candidate_time_sec"
    )
    validation["schema_validation"] = {
        "status": "PASS" if schema_ok else "FAIL",
        "detail": "Recommended timestamp/video/split columns match detected schema." if schema_ok else "Unexpected column detection; inspect inventory CSV.",
    }
    primary = recommendations[0] if recommendations else {}
    suitability_ok = (
        primary.get("usage_type") == "primary_ocr_anchor"
        and primary.get("recommended_file_path") == str(CANONICAL_ANCHOR_WITH_SPLIT)
        and primary.get("recommended_timestamp_column") == "canonical_boundary_time_sec"
    )
    validation["ocr_suitability_validation"] = {
        "status": "PASS" if suitability_ok else "FAIL",
        "detail": "Canonical with_split anchor selected for primary OCR; raw source files kept for debug.",
    }
    safety_ok = not changed_inputs
    validation["safety_validation"] = {
        "status": "PASS" if safety_ok else "FAIL",
        "detail": "No protected input file stats changed." if safety_ok else "Changed protected inputs: " + ", ".join(changed_inputs),
    }
    return validation


def write_report_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bundle_files() -> list[Path]:
    return [SCRIPT_PATH, INVENTORY_CSV, RECOMMENDATION_CSV, SUMMARY_MD, REPORT_JSON, RUN_LOG]


def is_forbidden_bundle_file(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    if path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
        return True
    if lower_parts & FORBIDDEN_BUNDLE_TOKENS:
        return True
    return False


def copy_bundle_file(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def write_latest_readme(path: Path, copied: list[Path], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "This bundle contains only the newly generated scene-boundary-input inspection files for the OCR follow-up.",
        "",
        "## Included",
    ]
    lines.extend([f"- {copied_path}" for copied_path in copied])
    lines.extend(
        [
            "",
            "## Excluded",
            "- raw video",
            "- frame image",
            "- cache",
            "- model weight",
            "- package directory",
            "- existing large feature originals",
            "- validation/test row-level OCR output",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def update_latest_bundles() -> dict[str, Any]:
    copied_latest: list[Path] = []
    copied_latest_ocr: list[Path] = []
    skipped: list[str] = []
    for src in bundle_files():
        if is_forbidden_bundle_file(src):
            skipped.append(str(src))
            continue
        copied_latest.append(copy_bundle_file(src, LATEST_DIR))
        copied_latest_ocr.append(copy_bundle_file(src, LATEST_OCR_DIR))
    write_latest_readme(LATEST_README, copied_latest, "Latest Files: Scene Boundary Inputs for OCR v2.4")
    write_latest_readme(LATEST_OCR_README, copied_latest_ocr, "Latest OCR Files: Scene Boundary Inputs for OCR v2.4")
    return {
        "latest_dir": str(LATEST_DIR),
        "latest_ocr_dir": str(LATEST_OCR_DIR),
        "copied_latest": [str(path) for path in copied_latest],
        "copied_latest_ocr": [str(path) for path in copied_latest_ocr],
        "skipped": skipped,
    }


def main() -> None:
    log("[STEP 01] Safety snapshot and output path preparation")
    DATA_OCR_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_OCR_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_OCR_DIR.mkdir(parents=True, exist_ok=True)
    protected_inputs = [
        OPENCV_FFMPEG_CANDIDATE,
        RESNET_RAW_CANDIDATE,
        RESNET_REVIEW_ANALYSIS,
        CANONICAL_ANCHOR,
        CANONICAL_ANCHOR_WITH_SPLIT,
        SPLIT_FILE,
        *REFERENCE_REPORTS,
    ]
    before_stats = file_stats(protected_inputs)
    split_map = load_split_map()

    log("[STEP 02] Search scene boundary related files")
    discovered = discover_files()
    log(f"Discovered {len(discovered)} scene/boundary/anchor related files in requested paths.")

    log("[STEP 03] Inspect OpenCV/FFmpeg candidate file schema")
    log("[STEP 04] Inspect ResNet candidate file schema")
    log("[STEP 05] Inspect canonical visual anchor file schema")
    log("[STEP 06] Inspect related reports and generation scripts")
    inventory = [inspect_file(path, split_map) for path in discovered]

    log("[STEP 07] Build OCR input inventory")
    inventory.sort(key=lambda row: (row["candidate_group"], row["file_path"]))
    write_csv(INVENTORY_CSV, inventory, INVENTORY_COLUMNS)

    log("[STEP 08] Select recommended OCR scene-boundary input")
    recommendations = build_recommendations(inventory)
    write_csv(RECOMMENDATION_CSV, recommendations, RECOMMENDATION_COLUMNS)

    log("[STEP 09] Generate CSV/report/log")
    provisional_validations = {
        "file_discovery_validation": {"status": "PENDING", "detail": "Computed after outputs are built."},
        "schema_validation": {"status": "PENDING", "detail": "Computed after recommendations are selected."},
        "ocr_suitability_validation": {"status": "PENDING", "detail": "Computed after recommendations are selected."},
        "safety_validation": {"status": "PENDING", "detail": "Computed after protected file stat comparison."},
    }
    report = build_report(inventory, recommendations, provisional_validations)
    SUMMARY_MD.write_text(build_summary_md(report), encoding="utf-8")
    write_report_json(REPORT_JSON, report)
    flush_log()

    log("[STEP 10] Run Sub Agent validations")
    validations = validate_outputs(inventory, recommendations, before_stats)
    report = build_report(inventory, recommendations, validations)
    SUMMARY_MD.write_text(build_summary_md(report), encoding="utf-8")
    write_report_json(REPORT_JSON, report)
    flush_log()

    log("[STEP 11] Update latest bundle")
    bundle_status = update_latest_bundles()
    report["bundle_status"] = bundle_status
    write_report_json(REPORT_JSON, report)
    SUMMARY_MD.write_text(build_summary_md(report), encoding="utf-8")
    flush_log()
    # bundle 상태와 step 11이 반영된 뒤 final report/log를 복사한다.
    update_latest_bundles()

    log("[STEP 12] Print final summary")
    answers = report["answers"]
    opencv = answers["opencv_ffmpeg_candidate_file"]
    resnet = answers["resnet_candidate_files"]
    canonical = answers["canonical_visual_anchor_files"]
    recommended = answers["recommended_ocr_input"]
    log(f"Recommended OCR anchor file: {recommended['primary_anchor_file']}")
    log(f"Recommended columns: video_id={recommended['video_id_column']}, timestamp={recommended['anchor_timestamp_column']}, split={recommended['split_column']}")
    log(f"OpenCV/FFmpeg raw: {opencv['path']} rows={opencv['row_count']}")
    log(f"ResNet raw: {resnet['raw_path']} rows={resnet['raw_row_count']}")
    log(f"ResNet review/usefulness: {resnet['review_analysis_path']} rows={resnet['review_analysis_row_count']}")
    log(f"Canonical merged anchor: {canonical['primary_with_split_path']} rows={canonical['primary_row_count']}")
    flush_log()
    update_latest_bundles()


if __name__ == "__main__":
    main()
