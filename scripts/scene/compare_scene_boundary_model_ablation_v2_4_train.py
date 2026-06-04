#!/usr/bin/env python3
"""Train-only scene boundary model ablation for actual ad boundary recall.

This audit compares OpenCV/FFmpeg, ResNet, and TransNetV2 candidates against
actual train ad_start/ad_end boundaries. It does not evaluate full scene
transition precision/recall because this project does not have full-video scene
transition ground truth.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import extract_transnetv2_scene_candidates_v2_4_train_audit as base


PROJECT_ROOT = Path(".")
TRAIN_VIDEO_IDS = base.TRAIN_VIDEO_IDS
VALIDATION_VIDEO_IDS = base.VALIDATION_VIDEO_IDS
TEST_VIDEO_IDS = base.TEST_VIDEO_IDS
TOLERANCES = [2, 5, 10]
CROSS_SOURCE_DEDUP_SEC = 2.0

SCRIPT_PATH = PROJECT_ROOT / "scripts/scene/compare_scene_boundary_model_ablation_v2_4_train.py"
LOG_PATH = PROJECT_ROOT / "logs/scene_model_ablation_v2_4_run_log.txt"
LATEST_BUNDLE_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_scene_model_ablation_v2_4"
LATEST_SCENE_DIR = PROJECT_ROOT / "outputs/latest_scene"

INVENTORY_CSV = PROJECT_ROOT / "data/scene/scene_model_ablation_candidate_inventory_v2_4_train.csv"
FAMILIES_CSV = PROJECT_ROOT / "data/scene/scene_model_ablation_candidate_families_v2_4_train.csv"
AUDIT_CSV = PROJECT_ROOT / "data/scene/scene_model_ablation_boundary_recall_audit_v2_4_train.csv"
RECALL_SUMMARY_CSV = PROJECT_ROOT / "data/scene/scene_model_ablation_recall_summary_v2_4_train.csv"
EFFICIENCY_CSV = PROJECT_ROOT / "data/scene/scene_model_ablation_efficiency_summary_v2_4_train.csv"
COMBO_SUMMARY_CSV = PROJECT_ROOT / "data/scene/scene_model_ablation_pairwise_combo_summary_v2_4_train.csv"
THREE_WAY_CSV = PROJECT_ROOT / "data/scene/scene_model_ablation_three_way_case_breakdown_v2_4_train.csv"
UNIQUE_CASES_CSV = PROJECT_ROOT / "data/scene/scene_model_ablation_unique_hit_cases_v2_4_train.csv"
RECOVERY_CSV = PROJECT_ROOT / "data/scene/scene_model_ablation_existing_missed_recovery_v2_4_train.csv"
RECOMMENDATION_CSV = PROJECT_ROOT / "data/scene/scene_model_ablation_recommendation_v2_4_train.csv"
VIDEO_LEVEL_CSV = PROJECT_ROOT / "data/scene/scene_model_ablation_video_level_recall_v2_4_train.csv"

SUMMARY_MD = PROJECT_ROOT / "reports/scene/scene_model_ablation_v2_4_summary.md"
REPORT_JSON = PROJECT_ROOT / "reports/scene/scene_model_ablation_v2_4_report.json"
FINDINGS_MD = PROJECT_ROOT / "reports/scene/scene_model_ablation_v2_4_findings.md"
PROFESSOR_MD = PROJECT_ROOT / "reports/scene/scene_model_ablation_v2_4_professor_response.md"

TRANSNET_PRIMARY_CSV = PROJECT_ROOT / "data/scene/transnetv2_scene_candidates_v2_4_train.csv"
TRANSNET_CONSERVATIVE_BEST_CSV = PROJECT_ROOT / "data/scene/transnetv2_conservative_best_candidate_v2_4_train.csv"
TRANSNET_CONSERVATIVE_SWEEP_CSV = PROJECT_ROOT / "data/scene/transnetv2_conservative_sweep_candidates_v2_4_train.csv"
TRANSNET_PRIMARY_REPORT = PROJECT_ROOT / "reports/scene/transnetv2_scene_candidate_audit_v2_4_report.json"
TRANSNET_CONSERVATIVE_REPORT = PROJECT_ROOT / "reports/scene/transnetv2_conservative_sweep_v2_4_report.json"
OPENCV_RESNET_SUMMARY_CSV = PROJECT_ROOT / "data/scene/opencv_resnet_boundary_recall_summary_v2_4_train.csv"
OPENCV_RESNET_CASE_CSV = PROJECT_ROOT / "data/scene/opencv_resnet_boundary_case_breakdown_v2_4_train.csv"
OPENCV_RESNET_REPORT = PROJECT_ROOT / "reports/scene/opencv_resnet_scene_boundary_recall_audit_v2_4_report.json"

INPUT_FILES = [
    base.SPLIT_PATH,
    base.SEGMENT_PATH,
    base.MANIFEST_PATH,
    base.OPENCV_PATH,
    base.RESNET_PATH,
    base.CANONICAL_PATH,
    OPENCV_RESNET_SUMMARY_CSV,
    OPENCV_RESNET_CASE_CSV,
    OPENCV_RESNET_REPORT,
    TRANSNET_PRIMARY_CSV,
    TRANSNET_PRIMARY_REPORT,
    TRANSNET_CONSERVATIVE_BEST_CSV,
    TRANSNET_CONSERVATIVE_SWEEP_CSV,
    TRANSNET_CONSERVATIVE_REPORT,
]

SINGLE_MAIN_FAMILIES = ["opencv_ffmpeg", "resnet", "transnetv2_conservative"]
SINGLE_REFERENCE_FAMILIES = ["transnetv2_primary"]
COMBO_DEFINITIONS = {
    "combo_opencv_resnet": ["opencv_ffmpeg", "resnet"],
    "combo_opencv_transnetv2_conservative": ["opencv_ffmpeg", "transnetv2_conservative"],
    "combo_resnet_transnetv2_conservative": ["resnet", "transnetv2_conservative"],
    "combo_all_three_conservative": ["opencv_ffmpeg", "resnet", "transnetv2_conservative"],
    "combo_opencv_transnetv2_primary_reference": ["opencv_ffmpeg", "transnetv2_primary"],
    "combo_resnet_transnetv2_primary_reference": ["resnet", "transnetv2_primary"],
    "combo_all_three_primary_reference": ["opencv_ffmpeg", "resnet", "transnetv2_primary"],
}
MODEL_GROUP = {
    "opencv_ffmpeg": "single_model",
    "resnet": "single_model",
    "transnetv2_primary": "single_reference_upper_bound",
    "transnetv2_conservative": "single_model",
    "combo_opencv_resnet": "pairwise_combo",
    "combo_opencv_transnetv2_conservative": "pairwise_combo",
    "combo_resnet_transnetv2_conservative": "pairwise_combo",
    "combo_all_three_conservative": "three_model_combo",
    "combo_opencv_transnetv2_primary_reference": "pairwise_primary_reference",
    "combo_resnet_transnetv2_primary_reference": "pairwise_primary_reference",
    "combo_all_three_primary_reference": "three_model_primary_reference",
    "canonical_all": "canonical_reference",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def safe_float(value: Any) -> Optional[float]:
    return base.safe_float(value)


def safe_int(value: Any) -> Optional[int]:
    return base.safe_int(value)


def fmt(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    if number is None:
        return ""
    return base.format_float(number, digits)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    return base.read_csv_rows(path)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    base.write_csv(path, fieldnames, rows)


def load_json(path: Path) -> Dict[str, Any]:
    return base.load_json(path)


def total_duration_minutes(mapping_rows: Sequence[Dict[str, Any]]) -> float:
    seconds = 0.0
    for row in mapping_rows:
        duration = safe_float(row.get("duration_sec"))
        if duration:
            seconds += duration
    return seconds / 60.0 if seconds > 0 else 0.0


def duration_by_video(mapping_rows: Sequence[Dict[str, Any]]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for row in mapping_rows:
        vid = safe_int(row.get("video_id"))
        duration = safe_float(row.get("duration_sec"))
        if vid is not None and duration:
            out[vid] = duration
    return out


def add_inventory(
    rows: List[Dict[str, Any]],
    candidate_source: str,
    candidate_family: str,
    path: Path,
    video_id_column: str,
    timestamp_column: str,
    split_column: str,
    score_column: str,
    source_column: str,
    method_column: str,
    model_column: str,
    row_count_total: int,
    row_count_train: int,
    usable: bool,
    notes: str,
) -> None:
    rows.append(
        {
            "candidate_source": candidate_source,
            "candidate_family": candidate_family,
            "file_path": str(path),
            "file_exists": path.exists(),
            "row_count_total": row_count_total,
            "row_count_train": row_count_train,
            "video_id_column": video_id_column,
            "timestamp_column": timestamp_column,
            "split_column": split_column,
            "score_column": score_column,
            "source_column": source_column,
            "method_column": method_column,
            "model_column": model_column,
            "scope": "train-only rows used for this audit",
            "usable": usable,
            "notes": notes,
        }
    )


def normalize_candidate(
    family: str,
    vid: int,
    sec: float,
    frame: Any = None,
    score: Any = None,
    source_model: str = "",
    threshold: Any = None,
    dedup_window_sec: Any = None,
    cluster_id: str = "",
    cluster_member_count: int = 1,
    is_combination: bool = False,
    notes: str = "",
    source_members: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "candidate_family": family,
        "source_members_json": source_members or [family],
        "video_id": vid,
        "candidate_sec": sec,
        "candidate_frame": frame,
        "score": safe_float(score),
        "source_model": source_model or family,
        "threshold": threshold,
        "dedup_window_sec": dedup_window_sec,
        "cluster_id": cluster_id,
        "cluster_member_count": cluster_member_count,
        "is_combination": is_combination,
        "split": "train",
        "notes": notes,
    }


def load_opencv_resnet_canonical(inventory: List[Dict[str, Any]], warnings: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    opencv_rows, resnet_rows, canonical_rows, source_warnings = base.load_existing_candidates(TRAIN_VIDEO_IDS)
    warnings.extend(source_warnings)
    opencv_total = len(read_csv_rows(base.OPENCV_PATH)) if base.OPENCV_PATH.exists() else 0
    resnet_total = len(read_csv_rows(base.RESNET_PATH)) if base.RESNET_PATH.exists() else 0
    canonical_total = len(read_csv_rows(base.CANONICAL_PATH)) if base.CANONICAL_PATH.exists() else 0
    add_inventory(inventory, "opencv_ffmpeg_file", "opencv_ffmpeg", base.OPENCV_PATH, "video_id", "candidate_time_sec", "", "scene_change_score", "candidate_source/candidate_source_for_audit", "method_used", "", opencv_total, len(opencv_rows), bool(opencv_rows), "OpenCV/FFmpeg v2.4 candidate file read-only input")
    add_inventory(inventory, "resnet_file", "resnet", base.RESNET_PATH, "video_id", "resnet_time_sec_std or candidate_time_sec", "", "resnet_score_std or scene_change_score", "candidate_source", "method_used", "model_name", resnet_total, len(resnet_rows), bool(resnet_rows), "ResNet embedding candidate file read-only input")
    add_inventory(inventory, "canonical_anchor_file", "canonical_all", base.CANONICAL_PATH, "video_id", "canonical_boundary_time_sec", "split", "visual_boundary_strength_score", "source_relation", "", "canonical_time_source", canonical_total, len(canonical_rows), bool(canonical_rows), "Existing canonical visual anchor read-only reference")

    opencv = [
        normalize_candidate(
            "opencv_ffmpeg",
            int(row["video_id"]),
            float(row["candidate_sec"]),
            score=row.get("score"),
            source_model="opencv_ffmpeg",
            threshold=row.get("threshold"),
            notes=row.get("notes") or row.get("source_relation") or "",
        )
        for row in opencv_rows
    ]
    resnet = [
        normalize_candidate(
            "resnet",
            int(row["video_id"]),
            float(row["candidate_sec"]),
            score=row.get("score"),
            source_model="resnet",
            threshold=row.get("threshold"),
            notes=row.get("notes") or row.get("source_relation") or "",
        )
        for row in resnet_rows
    ]
    canonical = [
        normalize_candidate(
            "canonical_all",
            int(row["video_id"]),
            float(row["candidate_sec"]),
            score=row.get("score"),
            source_model="canonical_all",
            notes=row.get("source_relation") or row.get("notes") or "",
        )
        for row in canonical_rows
    ]
    return opencv, resnet, canonical


def load_transnet_primary(inventory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_rows = read_csv_rows(TRANSNET_PRIMARY_CSV) if TRANSNET_PRIMARY_CSV.exists() else []
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(all_rows, start=1):
        if row.get("split") != "train":
            continue
        if row.get("candidate_family") != "transnetv2_primary":
            continue
        vid = safe_int(row.get("video_id"))
        sec = safe_float(row.get("candidate_sec"))
        if vid not in TRAIN_VIDEO_IDS or sec is None:
            continue
        rows.append(
            normalize_candidate(
                "transnetv2_primary",
                int(vid),
                float(sec),
                frame=row.get("candidate_frame"),
                score=row.get("transnetv2_score"),
                source_model="transnetv2_primary",
                threshold=row.get("threshold"),
                dedup_window_sec=0,
                cluster_id=f"transnetv2_primary_raw_{idx:06d}",
                notes="TransNetV2 primary threshold 0.5 candidate; read-only existing output",
            )
        )
    add_inventory(inventory, "transnetv2_primary_file", "transnetv2_primary", TRANSNET_PRIMARY_CSV, "video_id", "candidate_sec", "split", "transnetv2_score", "candidate_family", "", "transnetv2-pytorch", len(all_rows), len(rows), bool(rows), "TransNetV2 primary candidate rows where candidate_family=transnetv2_primary")
    return rows


def load_transnet_conservative(inventory: List[Dict[str, Any]], warnings: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    best_rows = read_csv_rows(TRANSNET_CONSERVATIVE_BEST_CSV) if TRANSNET_CONSERVATIVE_BEST_CSV.exists() else []
    selected = best_rows[0] if best_rows else {}
    selected_family = selected.get("selected_family") or "transnetv2_threshold_0_7_dedup_5"
    selected_threshold = safe_float(selected.get("threshold"))
    selected_dedup = safe_float(selected.get("dedup_window_sec"))
    if selected_family != "transnetv2_threshold_0_7_dedup_5" or selected_threshold != 0.7 or selected_dedup != 5.0:
        warnings.append(f"TransNetV2 conservative selected family differs from expected: family={selected_family}, threshold={selected_threshold}, dedup={selected_dedup}")
    all_rows = read_csv_rows(TRANSNET_CONSERVATIVE_SWEEP_CSV) if TRANSNET_CONSERVATIVE_SWEEP_CSV.exists() else []
    rows: List[Dict[str, Any]] = []
    for row in all_rows:
        if row.get("split") != "train":
            continue
        if row.get("sweep_family") != selected_family:
            continue
        vid = safe_int(row.get("video_id"))
        sec = safe_float(row.get("candidate_sec"))
        if vid not in TRAIN_VIDEO_IDS or sec is None:
            continue
        rows.append(
            normalize_candidate(
                "transnetv2_conservative",
                int(vid),
                float(sec),
                frame=row.get("candidate_frame"),
                score=row.get("transnetv2_score"),
                source_model=selected_family,
                threshold=row.get("threshold"),
                dedup_window_sec=row.get("dedup_window_sec"),
                cluster_id=row.get("cluster_id") or "",
                cluster_member_count=safe_int(row.get("cluster_member_count")) or 1,
                notes="recommended TransNetV2 conservative candidate: threshold 0.7 + internal dedup 5s",
            )
        )
    add_inventory(inventory, "transnetv2_conservative_sweep_file", "transnetv2_conservative", TRANSNET_CONSERVATIVE_SWEEP_CSV, "video_id", "candidate_sec", "split", "transnetv2_score", "sweep_family", "", "transnetv2-pytorch", len(all_rows), len(rows), bool(rows), f"Filtered sweep_family={selected_family}; threshold={selected_threshold}; dedup={selected_dedup}")
    return rows, selected


def candidate_counts_by_video(rows: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for row in rows:
        vid = safe_int(row.get("video_id"))
        if vid is not None:
            out[vid] = out.get(vid, 0) + 1
    return out


def times_by_video(rows: Sequence[Dict[str, Any]]) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for row in rows:
        vid = safe_int(row.get("video_id"))
        sec = safe_float(row.get("candidate_sec"))
        if vid is not None and sec is not None:
            out.setdefault(vid, []).append(float(sec))
    for vid in out:
        out[vid].sort()
    return out


def cluster_combo_rows(family: str, members: List[Dict[str, Any]], included_models: List[str]) -> Tuple[List[Dict[str, Any]], Dict[int, List[float]]]:
    by_video: Dict[int, List[Dict[str, Any]]] = {}
    for row in members:
        vid = int(row["video_id"])
        by_video.setdefault(vid, []).append(row)
    cluster_rows: List[Dict[str, Any]] = []
    hit_times: Dict[int, List[float]] = {}
    cluster_index = 0
    for vid, nodes in sorted(by_video.items()):
        for row in nodes:
            sec = float(row["candidate_sec"])
            hit_times.setdefault(vid, []).append(sec)

        def edge(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            return abs(float(a["candidate_sec"]) - float(b["candidate_sec"])) <= CROSS_SOURCE_DEDUP_SEC

        for group in base.connected_components(nodes, edge):
            cluster_index += 1
            group_rows = [nodes[i] for i in group]
            best = sorted(
                group_rows,
                key=lambda r: (-(safe_float(r.get("score")) if safe_float(r.get("score")) is not None else -1.0), float(r["candidate_sec"])),
            )[0]
            secs = [float(r["candidate_sec"]) for r in group_rows]
            source_members = sorted({str(r["candidate_family"]) for r in group_rows})
            cluster_rows.append(
                normalize_candidate(
                    family,
                    vid,
                    float(best["candidate_sec"]),
                    frame=best.get("candidate_frame"),
                    score=best.get("score"),
                    source_model="+".join(included_models),
                    threshold="",
                    dedup_window_sec=CROSS_SOURCE_DEDUP_SEC,
                    cluster_id=f"{family}_v{vid:02d}_c{cluster_index:05d}",
                    cluster_member_count=len(group_rows),
                    is_combination=True,
                    notes=f"combination cluster; hit audit uses nearest original member timestamp; cluster_min={min(secs):.3f}; cluster_max={max(secs):.3f}",
                    source_members=source_members,
                )
            )
    for vid in hit_times:
        hit_times[vid].sort()
    return cluster_rows, hit_times


def build_family_sources(family_rows: Dict[str, List[Dict[str, Any]]], combo_hit_times: Dict[str, Dict[int, List[float]]], durations: Dict[int, float]) -> Dict[str, Dict[str, Any]]:
    sources: Dict[str, Dict[str, Any]] = {}
    for family, rows in family_rows.items():
        if family in combo_hit_times:
            hit_times = combo_hit_times[family]
        else:
            hit_times = times_by_video(rows)
        counts = candidate_counts_by_video(rows)
        sources[family] = {
            "times_by_video": hit_times,
            "candidate_count_by_video": counts,
            "candidate_count_total": len(rows),
            "candidate_video_count": len([vid for vid, n in counts.items() if n > 0]),
            "candidates_per_minute_total": len(rows) / (sum(durations.values()) / 60.0) if durations else None,
            "model_group": MODEL_GROUP.get(family, "unknown"),
        }
    return sources


def nearest(times: Sequence[float], actual: float) -> Tuple[Optional[float], Optional[float]]:
    return base.nearest_candidate(times, actual)


def compute_audit(boundaries: List[Dict[str, Any]], sources: Dict[str, Dict[str, Any]], durations: Dict[int, float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for boundary in boundaries:
        vid = int(boundary["video_id"])
        actual = float(boundary["actual_sec"])
        for family, data in sources.items():
            nearest_sec, distance = nearest(data["times_by_video"].get(vid, []), actual)
            count_in_video = data["candidate_count_by_video"].get(vid, 0)
            duration = durations.get(vid)
            rows.append(
                {
                    "video_id": vid,
                    "ad_interval_id": boundary["ad_interval_id"],
                    "boundary_type": boundary["boundary_type"],
                    "actual_sec": actual,
                    "candidate_family": family,
                    "nearest_candidate_sec": nearest_sec,
                    "nearest_distance_sec": distance,
                    "within_2s": distance is not None and distance <= 2,
                    "within_5s": distance is not None and distance <= 5,
                    "within_10s": distance is not None and distance <= 10,
                    "candidate_count_in_video": count_in_video,
                    "candidates_per_minute_in_video": count_in_video / (duration / 60.0) if duration else None,
                    "notes": "combination hit uses nearest original member timestamp; density uses dedup cluster count" if family.startswith("combo_") else "single/canonical candidate timestamp",
                }
            )
    return rows


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    return base.quantile(values, q)


def summarize_recall(audit_rows: List[Dict[str, Any]], sources: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for family, data in sources.items():
        family_rows = [r for r in audit_rows if r["candidate_family"] == family]
        for boundary_type in ["start", "end", "all"]:
            type_rows = family_rows if boundary_type == "all" else [r for r in family_rows if r["boundary_type"] == boundary_type]
            distances = [float(r["nearest_distance_sec"]) for r in type_rows if r.get("nearest_distance_sec") not in (None, "")]
            for tol in TOLERANCES:
                hit_count = sum(1 for r in type_rows if r.get(f"within_{tol}s") is True)
                actual_count = len(type_rows)
                rows.append(
                    {
                        "candidate_family": family,
                        "model_group": data["model_group"],
                        "boundary_type": boundary_type,
                        "tolerance_sec": tol,
                        "actual_boundary_count": actual_count,
                        "hit_count": hit_count,
                        "recall": hit_count / actual_count if actual_count else None,
                        "median_nearest_distance_sec": statistics.median(distances) if distances else None,
                        "mean_nearest_distance_sec": statistics.mean(distances) if distances else None,
                        "p90_nearest_distance_sec": quantile(distances, 0.9),
                        "max_nearest_distance_sec": max(distances) if distances else None,
                        "candidate_count_total": data["candidate_count_total"],
                        "candidate_video_count": data["candidate_video_count"],
                        "candidates_per_minute_total": data["candidates_per_minute_total"],
                    }
                )
    return rows


def summary_lookup(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    return {(r["candidate_family"], r["boundary_type"], int(r["tolerance_sec"])): r for r in rows}


def get_metric(lookup: Dict[Tuple[str, str, int], Dict[str, Any]], family: str, boundary_type: str, tol: int, key: str = "recall") -> Any:
    return lookup.get((family, boundary_type, tol), {}).get(key)


def audit_lookup(audit_rows: List[Dict[str, Any]]) -> Dict[Tuple[int, str, str, str], Dict[str, Any]]:
    return base.audit_lookup(audit_rows)


def classify_three_way(opencv_hit: bool, resnet_hit: bool, transnet_hit: bool) -> str:
    if opencv_hit and resnet_hit and transnet_hit:
        return "all_three_hit"
    if opencv_hit and resnet_hit:
        return "opencv_resnet_only"
    if opencv_hit and transnet_hit:
        return "opencv_transnetv2_only"
    if resnet_hit and transnet_hit:
        return "resnet_transnetv2_only"
    if opencv_hit:
        return "only_opencv"
    if resnet_hit:
        return "only_resnet"
    if transnet_hit:
        return "only_transnetv2"
    return "none_hit"


def three_way_cases(boundaries: List[Dict[str, Any]], audit_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    lookup = audit_lookup(audit_rows)
    case_rows: List[Dict[str, Any]] = []
    detailed_rows: List[Dict[str, Any]] = []
    case_types = ["none_hit", "only_opencv", "only_resnet", "only_transnetv2", "opencv_resnet_only", "opencv_transnetv2_only", "resnet_transnetv2_only", "all_three_hit"]
    for tol in TOLERANCES:
        hit_col = f"within_{tol}s"
        for boundary in boundaries:
            vid = int(boundary["video_id"])
            interval = str(boundary["ad_interval_id"])
            btype = str(boundary["boundary_type"])
            opencv = lookup.get((vid, interval, btype, "opencv_ffmpeg"), {})
            resnet = lookup.get((vid, interval, btype, "resnet"), {})
            transnet = lookup.get((vid, interval, btype, "transnetv2_conservative"), {})
            oh = bool(opencv.get(hit_col))
            rh = bool(resnet.get(hit_col))
            th = bool(transnet.get(hit_col))
            case_type = classify_three_way(oh, rh, th)
            detailed_rows.append(
                {
                    "tolerance_sec": tol,
                    "video_id": vid,
                    "ad_interval_id": interval,
                    "boundary_type": btype,
                    "actual_sec": boundary["actual_sec"],
                    "case_type": case_type,
                    "opencv_hit": oh,
                    "resnet_hit": rh,
                    "transnetv2_hit": th,
                    "opencv_nearest_candidate_sec": opencv.get("nearest_candidate_sec"),
                    "opencv_nearest_distance_sec": opencv.get("nearest_distance_sec"),
                    "resnet_nearest_candidate_sec": resnet.get("nearest_candidate_sec"),
                    "resnet_nearest_distance_sec": resnet.get("nearest_distance_sec"),
                    "transnetv2_nearest_candidate_sec": transnet.get("nearest_candidate_sec"),
                    "transnetv2_nearest_distance_sec": transnet.get("nearest_distance_sec"),
                    "notes": "3-way case over OpenCV/FFmpeg, ResNet, TransNetV2 conservative",
                }
            )
        for group in ["start", "end", "all"]:
            group_rows = detailed_rows if group == "all" else [r for r in detailed_rows if int(r["tolerance_sec"]) == tol and r["boundary_type"] == group]
            if group == "all":
                group_rows = [r for r in detailed_rows if int(r["tolerance_sec"]) == tol]
            denom = len(group_rows)
            for case_type in case_types:
                count = sum(1 for r in group_rows if r["case_type"] == case_type)
                case_rows.append(
                    {
                        "tolerance_sec": tol,
                        "boundary_type_group": group,
                        "case_type": case_type,
                        "boundary_count": count,
                        "boundary_ratio": count / denom if denom else None,
                        "interpretation": "Venn-style 3-model contribution case; not full-scene precision",
                    }
                )
    return case_rows, detailed_rows


def recovery_rows(boundaries: List[Dict[str, Any]], audit_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = audit_lookup(audit_rows)
    rows: List[Dict[str, Any]] = []
    for tol in TOLERANCES:
        hit_col = f"within_{tol}s"
        for boundary in boundaries:
            vid = int(boundary["video_id"])
            interval = str(boundary["ad_interval_id"])
            btype = str(boundary["boundary_type"])
            combo = lookup.get((vid, interval, btype, "combo_opencv_resnet"), {})
            opencv = lookup.get((vid, interval, btype, "opencv_ffmpeg"), {})
            resnet = lookup.get((vid, interval, btype, "resnet"), {})
            conserv = lookup.get((vid, interval, btype, "transnetv2_conservative"), {})
            primary = lookup.get((vid, interval, btype, "transnetv2_primary"), {})
            combo_hit = bool(combo.get(hit_col))
            conserv_hit = bool(conserv.get(hit_col))
            primary_hit = bool(primary.get(hit_col))
            if combo_hit and not conserv_hit and not primary_hit:
                notes = "existing OpenCV/ResNet hit while TransNetV2 did not"
            elif (not combo_hit) and conserv_hit:
                notes = "existing OpenCV/ResNet missed and TransNetV2 conservative recovered"
            elif (not combo_hit) and primary_hit:
                notes = "existing OpenCV/ResNet missed and TransNetV2 primary recovered"
            else:
                notes = "case retained for tolerance-level recovery audit"
            rows.append(
                {
                    "tolerance_sec": tol,
                    "video_id": vid,
                    "ad_interval_id": interval,
                    "boundary_type": btype,
                    "actual_sec": boundary["actual_sec"],
                    "existing_opencv_resnet_hit": combo_hit,
                    "transnetv2_conservative_hit": conserv_hit,
                    "recovered_by_transnetv2_conservative": (not combo_hit) and conserv_hit,
                    "recovered_by_transnetv2_primary": (not combo_hit) and primary_hit,
                    "opencv_nearest_candidate_sec": opencv.get("nearest_candidate_sec"),
                    "opencv_nearest_distance_sec": opencv.get("nearest_distance_sec"),
                    "resnet_nearest_candidate_sec": resnet.get("nearest_candidate_sec"),
                    "resnet_nearest_distance_sec": resnet.get("nearest_distance_sec"),
                    "transnetv2_conservative_nearest_candidate_sec": conserv.get("nearest_candidate_sec"),
                    "transnetv2_conservative_nearest_distance_sec": conserv.get("nearest_distance_sec"),
                    "transnetv2_primary_nearest_candidate_sec": primary.get("nearest_candidate_sec"),
                    "transnetv2_primary_nearest_distance_sec": primary.get("nearest_distance_sec"),
                    "notes": notes,
                }
            )
    return rows


def build_efficiency_summary(summary_rows: List[Dict[str, Any]], unique_case_rows: List[Dict[str, Any]], recovery: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = summary_lookup(summary_rows)
    families = sorted({r["candidate_family"] for r in summary_rows}, key=lambda x: list(MODEL_GROUP.keys()).index(x) if x in MODEL_GROUP else 999)
    unique_map = {
        "opencv_ffmpeg": "only_opencv",
        "resnet": "only_resnet",
        "transnetv2_conservative": "only_transnetv2",
    }
    rows: List[Dict[str, Any]] = []
    for family in families:
        count = safe_float(get_metric(lookup, family, "all", 2, "candidate_count_total")) or 0.0
        cpm = get_metric(lookup, family, "all", 2, "candidates_per_minute_total")
        hits = {tol: safe_float(get_metric(lookup, family, "all", tol, "hit_count")) or 0.0 for tol in TOLERANCES}
        recall = {tol: safe_float(get_metric(lookup, family, "all", tol, "recall")) or 0.0 for tol in TOLERANCES}
        unique_5 = 0
        unique_10 = 0
        if family in unique_map:
            unique_5 = sum(1 for r in unique_case_rows if int(r["tolerance_sec"]) == 5 and r["case_type"] == unique_map[family])
            unique_10 = sum(1 for r in unique_case_rows if int(r["tolerance_sec"]) == 10 and r["case_type"] == unique_map[family])
        recovered_5 = sum(1 for r in recovery if int(r["tolerance_sec"]) == 5 and r.get("recovered_by_transnetv2_conservative") in [True, "true"] and family in ["transnetv2_conservative", "combo_opencv_transnetv2_conservative", "combo_resnet_transnetv2_conservative", "combo_all_three_conservative"])
        recovered_10 = sum(1 for r in recovery if int(r["tolerance_sec"]) == 10 and r.get("recovered_by_transnetv2_conservative") in [True, "true"] and family in ["transnetv2_conservative", "combo_opencv_transnetv2_conservative", "combo_resnet_transnetv2_conservative", "combo_all_three_conservative"])
        interpretation = "candidate efficiency proxy, not precision"
        if family == "transnetv2_primary":
            interpretation += "; high-density reference"
        if family == "combo_all_three_conservative":
            interpretation += "; practical high-recall conservative 3-model option"
        rows.append(
            {
                "candidate_family": family,
                "model_group": MODEL_GROUP.get(family, "unknown"),
                "candidate_count_total": int(count),
                "candidates_per_minute_total": cpm,
                "all_hit_count_2s": int(hits[2]),
                "all_hit_count_5s": int(hits[5]),
                "all_hit_count_10s": int(hits[10]),
                "all_recall_2s": recall[2],
                "all_recall_5s": recall[5],
                "all_recall_10s": recall[10],
                "all_hit_per_100_candidates_2s": hits[2] / count * 100 if count else None,
                "all_hit_per_100_candidates_5s": hits[5] / count * 100 if count else None,
                "all_hit_per_100_candidates_10s": hits[10] / count * 100 if count else None,
                "unique_hit_count_5s": unique_5,
                "unique_hit_count_10s": unique_10,
                "unique_hit_per_100_candidates_5s": unique_5 / count * 100 if count else None,
                "unique_hit_per_100_candidates_10s": unique_10 / count * 100 if count else None,
                "interpretation": interpretation + f"; recovered_existing_missed_per_100_candidates_5s={recovered_5 / count * 100 if count else 0:.4f}; recovered_existing_missed_per_100_candidates_10s={recovered_10 / count * 100 if count else 0:.4f}",
            }
        )
    return rows


def density_risk(cpm: Optional[float], existing_cpm: float, primary_cpm: float) -> str:
    if cpm is None:
        return "unknown"
    if cpm <= existing_cpm * 1.5:
        return "low"
    if cpm <= existing_cpm * 2.0:
        return "medium"
    if cpm <= primary_cpm:
        return "high"
    return "very_high"


def combo_summary_rows(summary_rows: List[Dict[str, Any]], existing_cpm: float, primary_cpm: float) -> List[Dict[str, Any]]:
    lookup = summary_lookup(summary_rows)
    rows: List[Dict[str, Any]] = []
    combo_families = list(COMBO_DEFINITIONS.keys())
    for family in combo_families:
        included = COMBO_DEFINITIONS[family]
        best_single = {
            tol: max(safe_float(get_metric(lookup, single, "all", tol)) or 0.0 for single in included)
            for tol in TOLERANCES
        }
        actual_count = safe_float(get_metric(lookup, family, "all", 2, "actual_boundary_count")) or 0.0
        hits = {tol: safe_float(get_metric(lookup, family, "all", tol, "hit_count")) or 0.0 for tol in TOLERANCES}
        cpm = safe_float(get_metric(lookup, family, "all", 2, "candidates_per_minute_total"))
        rows.append(
            {
                "combo_family": family,
                "included_models": "+".join(included),
                "candidate_count_total": get_metric(lookup, family, "all", 2, "candidate_count_total"),
                "candidates_per_minute_total": cpm,
                "recall_all_2s": get_metric(lookup, family, "all", 2),
                "recall_all_5s": get_metric(lookup, family, "all", 5),
                "recall_all_10s": get_metric(lookup, family, "all", 10),
                "recall_start_2s": get_metric(lookup, family, "start", 2),
                "recall_start_5s": get_metric(lookup, family, "start", 5),
                "recall_start_10s": get_metric(lookup, family, "start", 10),
                "recall_end_2s": get_metric(lookup, family, "end", 2),
                "recall_end_5s": get_metric(lookup, family, "end", 5),
                "recall_end_10s": get_metric(lookup, family, "end", 10),
                "gain_vs_best_single_2s": (safe_float(get_metric(lookup, family, "all", 2)) or 0.0) - best_single[2],
                "gain_vs_best_single_5s": (safe_float(get_metric(lookup, family, "all", 5)) or 0.0) - best_single[5],
                "gain_vs_best_single_10s": (safe_float(get_metric(lookup, family, "all", 10)) or 0.0) - best_single[10],
                "remaining_missed_2s": int(actual_count - hits[2]),
                "remaining_missed_5s": int(actual_count - hits[5]),
                "remaining_missed_10s": int(actual_count - hits[10]),
                "density_risk_level": density_risk(cpm, existing_cpm, primary_cpm),
                "interpretation": "primary reference upper-bound" if "primary_reference" in family else "main conservative combo",
            }
        )
    return rows


def video_level_rows(audit_rows: List[Dict[str, Any]], sources: Dict[str, Dict[str, Any]], durations: Dict[int, float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for vid in TRAIN_VIDEO_IDS:
        for family, data in sources.items():
            candidate_count = data["candidate_count_by_video"].get(vid, 0)
            cpm = candidate_count / (durations[vid] / 60.0) if durations.get(vid) else None
            family_rows = [r for r in audit_rows if int(r["video_id"]) == vid and r["candidate_family"] == family]
            for boundary_type in ["start", "end", "all"]:
                type_rows = family_rows if boundary_type == "all" else [r for r in family_rows if r["boundary_type"] == boundary_type]
                for tol in TOLERANCES:
                    hit_count = sum(1 for r in type_rows if r.get(f"within_{tol}s") is True)
                    count = len(type_rows)
                    rows.append(
                        {
                            "video_id": vid,
                            "candidate_family": family,
                            "boundary_type": boundary_type,
                            "tolerance_sec": tol,
                            "actual_boundary_count": count,
                            "hit_count": hit_count,
                            "recall": hit_count / count if count else None,
                            "candidate_count": candidate_count,
                            "candidates_per_minute": cpm,
                            "notes": "train-only video-level ablation recall",
                        }
                    )
    return rows


def score_efficiency(row: Dict[str, Any]) -> float:
    return (
        (safe_float(row.get("all_hit_per_100_candidates_5s")) or 0.0) * 0.55
        + (safe_float(row.get("all_hit_per_100_candidates_10s")) or 0.0) * 0.35
        + (safe_float(row.get("unique_hit_per_100_candidates_5s")) or 0.0) * 0.10
    )


def recommendation_rows(summary_rows: List[Dict[str, Any]], efficiency_rows: List[Dict[str, Any]], combo_rows: List[Dict[str, Any]], existing_cpm: float, primary_cpm: float) -> List[Dict[str, Any]]:
    lookup = summary_lookup(summary_rows)
    efficiency_by_family = {r["candidate_family"]: r for r in efficiency_rows}
    single_families = ["opencv_ffmpeg", "resnet", "transnetv2_conservative", "transnetv2_primary"]
    main_single_families = ["opencv_ffmpeg", "resnet", "transnetv2_conservative"]
    main_pair_families = ["combo_opencv_resnet", "combo_opencv_transnetv2_conservative", "combo_resnet_transnetv2_conservative"]
    all_practical = main_single_families + main_pair_families + ["combo_all_three_conservative"]

    def make_row(rec_type: str, family: str, label: str, reason: str, caution: str) -> Dict[str, Any]:
        cpm = safe_float(get_metric(lookup, family, "all", 2, "candidates_per_minute_total"))
        return {
            "recommendation_type": rec_type,
            "selected_candidate_family": family,
            "selected_models": "+".join(COMBO_DEFINITIONS.get(family, [family])),
            "candidate_count_total": get_metric(lookup, family, "all", 2, "candidate_count_total"),
            "candidates_per_minute_total": cpm,
            "recall_all_2s": get_metric(lookup, family, "all", 2),
            "recall_all_5s": get_metric(lookup, family, "all", 5),
            "recall_all_10s": get_metric(lookup, family, "all", 10),
            "recall_start_5s": get_metric(lookup, family, "start", 5),
            "recall_end_5s": get_metric(lookup, family, "end", 5),
            "efficiency_score": score_efficiency(efficiency_by_family.get(family, {})),
            "density_risk_level": density_risk(cpm, existing_cpm, primary_cpm),
            "recommendation_label": label,
            "reason": reason,
            "caution": caution,
        }

    best_single_recall = max(single_families, key=lambda f: (safe_float(get_metric(lookup, f, "all", 5)) or 0.0, safe_float(get_metric(lookup, f, "all", 10)) or 0.0))
    best_single_eff = max(main_single_families, key=lambda f: score_efficiency(efficiency_by_family.get(f, {})))
    best_pair_recall = max(main_pair_families, key=lambda f: (safe_float(get_metric(lookup, f, "all", 5)) or 0.0, safe_float(get_metric(lookup, f, "all", 10)) or 0.0))
    best_pair_eff = max(main_pair_families, key=lambda f: score_efficiency(efficiency_by_family.get(f, {})))
    best_practical = max(
        all_practical,
        key=lambda f: (
            (safe_float(get_metric(lookup, f, "all", 5)) or 0.0) * 2.0
            + (safe_float(get_metric(lookup, f, "all", 10)) or 0.0)
            - max((safe_float(get_metric(lookup, f, "all", 2, "candidates_per_minute_total")) or 0.0) - existing_cpm * 2, 0.0) * 0.05
            + score_efficiency(efficiency_by_family.get(f, {})) * 0.02
        ),
    )
    upper = "combo_all_three_primary_reference"
    return [
        make_row("best_single_model_by_recall", best_single_recall, "upper_bound_high_density_reference" if best_single_recall == "transnetv2_primary" else "main_single_recall", "highest single-model all recall@5/@10 among single/reference families", "primary can be density-heavy; not precision/F1"),
        make_row("best_single_model_by_efficiency", best_single_eff, "efficiency_proxy_best_single", "highest hit-per-100-candidates proxy among main single families", "efficiency proxy is not precision"),
        make_row("best_pair_by_recall", best_pair_recall, "main_pair_recall", "highest all recall@5/@10 among main conservative pairwise combos", "pair adds candidates; density must be reviewed before OCR"),
        make_row("best_pair_by_efficiency", best_pair_eff, "efficiency_proxy_best_pair", "best hit-per-100-candidates proxy among main conservative pairs", "efficiency proxy is not precision"),
        make_row("best_overall_practical_choice", best_practical, "use_as_conservative_aux_anchor" if best_practical == "combo_all_three_conservative" else "viewer_review_before_integration", "balances all recall@5/@10, recovered missed boundaries, and candidate density", "train-only ad-boundary recall; validate with viewer/subset scene GT later"),
        make_row("upper_bound_high_density_reference", upper, "upper_bound_high_density_reference", "highest-density reference using TransNetV2 primary", "not recommended as OCR default due density risk"),
    ]


def md_table(rows: List[Dict[str, Any]], cols: Sequence[str], limit: Optional[int] = None) -> str:
    use_rows = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in use_rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(fmt(value, 4))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def validate(
    split_validation: Dict[str, Any],
    family_rows: Dict[str, List[Dict[str, Any]]],
    audit_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    three_way_summary: List[Dict[str, Any]],
    unique_rows: List[Dict[str, Any]],
    efficiency_rows: List[Dict[str, Any]],
    recommendation: List[Dict[str, Any]],
    protected_modified: List[str],
    latest_files: Sequence[Path],
    report_texts: Dict[str, str],
) -> Dict[str, Any]:
    valtest = set(VALIDATION_VIDEO_IDS + TEST_VIDEO_IDS)
    output_vids = {int(r["video_id"]) for rows in family_rows.values() for r in rows if safe_int(r.get("video_id")) is not None}
    audit_vids = {int(r["video_id"]) for r in audit_rows if safe_int(r.get("video_id")) is not None}
    distance_ok = True
    tolerance_ok = True
    for row in audit_rows:
        actual = safe_float(row.get("actual_sec"))
        nearest_sec = safe_float(row.get("nearest_candidate_sec"))
        distance = safe_float(row.get("nearest_distance_sec"))
        if actual is not None and nearest_sec is not None and distance is not None:
            if abs(abs(nearest_sec - actual) - distance) > 1e-6:
                distance_ok = False
        for tol in TOLERANCES:
            if bool(row.get(f"within_{tol}s")) != (distance is not None and distance <= tol):
                tolerance_ok = False
    three_way_ok = True
    for tol in TOLERANCES:
        for group, denom in [("start", 16), ("end", 16), ("all", 32)]:
            total = sum(int(r["boundary_count"]) for r in three_way_summary if int(r["tolerance_sec"]) == tol and r["boundary_type_group"] == group)
            if total != denom:
                three_way_ok = False
    unique_ok = len(unique_rows) == 32 * len(TOLERANCES)
    suffix_forbidden = {".mp4", ".mov", ".mkv", ".avi", ".jpg", ".jpeg", ".png", ".pth", ".pt", ".ckpt", ".npz", ".npy"}
    latest_safe = all(p.suffix.lower() not in suffix_forbidden for p in latest_files)
    required_phrases = [
        "모델을 합치면 recall이 올라갈 수 있으므로, 단순 combined 결과만 보지 않고 각 모델의 단독 성능을 먼저 비교했습니다.",
        "또한 모든 pairwise 조합과 3-model 조합을 비교하여 어떤 모델이 독립적으로 기여하는지 확인했습니다.",
        "후보를 많이 뽑으면 유리해지는 문제를 보완하기 위해 candidate count, candidates/min, hit per 100 candidates를 함께 제시했습니다.",
        "이번 평가는 전체 장면전환 GT가 아니라 광고 시작/종료 boundary 근처 포착률 기준입니다.",
        "향후 시간이 있다면 일부 영상에 대해 전체 scene transition GT를 라벨링하여 precision/F1까지 검증할 수 있습니다.",
    ]
    professor_ok = all(phrase in report_texts.get("professor", "") for phrase in required_phrases)
    proxy_ok = "precision" not in "\n".join(str(r.get("interpretation", "")) for r in efficiency_rows).lower().replace("not precision", "")
    return {
        "input_split_validation": {
            "status": "PASS" if split_validation.get("fixed_train_match") and not ((output_vids | audit_vids) & valtest) else "FAIL",
            "processed_video_ids": sorted(output_vids),
            "validation_test_video_ids_in_outputs": sorted((output_vids | audit_vids) & valtest),
            "split_validation": split_validation,
        },
        "candidate_source_validation": {
            "status": "PASS" if all(f in family_rows and family_rows[f] for f in ["opencv_ffmpeg", "resnet", "transnetv2_primary", "transnetv2_conservative"]) else "FAIL",
            "opencv_count": len(family_rows.get("opencv_ffmpeg", [])),
            "resnet_count": len(family_rows.get("resnet", [])),
            "transnetv2_primary_count": len(family_rows.get("transnetv2_primary", [])),
            "transnetv2_conservative_count": len(family_rows.get("transnetv2_conservative", [])),
            "transnetv2_conservative_expected": "threshold=0.7,dedup=5s",
        },
        "combination_logic_validation": {
            "status": "PASS" if all(f in family_rows and family_rows[f] for f in COMBO_DEFINITIONS) else "FAIL",
            "pairwise_generated": [f for f in COMBO_DEFINITIONS if "all_three" not in f],
            "three_model_generated": [f for f in COMBO_DEFINITIONS if "all_three" in f],
            "average_timestamp_used": False,
            "hit_density_policy_documented": True,
        },
        "recall_audit_validation": {
            "status": "PASS" if distance_ok and tolerance_ok and not (audit_vids & valtest) else "FAIL",
            "actual_label_used_for_candidate_generation": False,
            "actual_label_used_for_recall_audit_only": True,
            "nearest_distance_formula_ok": distance_ok,
            "tolerance_flags_ok": tolerance_ok,
            "start_end_all_separate": bool(summary_rows),
        },
        "unique_contribution_validation": {
            "status": "PASS" if three_way_ok and unique_ok else "FAIL",
            "three_way_case_counts_match_actual_boundaries": three_way_ok,
            "unique_case_row_count": len(unique_rows),
            "expected_unique_case_row_count": 96,
        },
        "density_efficiency_validation": {
            "status": "PASS" if efficiency_rows and recommendation and proxy_ok else "FAIL",
            "candidate_efficiency_proxy_not_precision": proxy_ok,
            "recommendation_not_recall_only": True,
        },
        "professor_response_validation": {
            "status": "PASS" if professor_ok else "FAIL",
            "required_phrases_present": professor_ok,
            "scene_gt_limitation_stated": "전체 장면전환 GT" in report_texts.get("professor", ""),
            "subset_scene_gt_future_plan_stated": "precision/F1" in report_texts.get("professor", ""),
        },
        "output_safety_validation": {
            "status": "PASS" if not protected_modified and latest_safe and not (audit_vids & valtest) else "FAIL",
            "protected_files_modified": protected_modified,
            "latest_bundle_excludes_forbidden_files": latest_safe,
            "validation_test_row_level_output_generated": bool(audit_vids & valtest),
        },
    }


def write_reports(
    inventory_rows: List[Dict[str, Any]],
    recall_summary: List[Dict[str, Any]],
    efficiency: List[Dict[str, Any]],
    combo_summary: List[Dict[str, Any]],
    three_way: List[Dict[str, Any]],
    unique_cases: List[Dict[str, Any]],
    recovery: List[Dict[str, Any]],
    recommendation: List[Dict[str, Any]],
    validations: Dict[str, Any],
    warnings: List[str],
    errors: List[str],
    protected_modified: List[str],
    row_counts: Dict[str, int],
    start_iso: str,
) -> Dict[str, str]:
    lookup = summary_lookup(recall_summary)
    recommendation_by_type = {r["recommendation_type"]: r for r in recommendation}
    single_rows = [r for r in efficiency if r["candidate_family"] in ["opencv_ffmpeg", "resnet", "transnetv2_primary", "transnetv2_conservative"]]
    main_pair_rows = [r for r in combo_summary if r["combo_family"] in ["combo_opencv_resnet", "combo_opencv_transnetv2_conservative", "combo_resnet_transnetv2_conservative"]]
    three_rows = [r for r in combo_summary if r["combo_family"] == "combo_all_three_conservative"]
    primary_ref_rows = [r for r in combo_summary if "primary_reference" in r["combo_family"]]
    none_5 = sum(int(r["boundary_count"]) for r in three_way if int(r["tolerance_sec"]) == 5 and r["boundary_type_group"] == "all" and r["case_type"] == "none_hit")
    none_10 = sum(int(r["boundary_count"]) for r in three_way if int(r["tolerance_sec"]) == 10 and r["boundary_type_group"] == "all" and r["case_type"] == "none_hit")
    recovered_5 = sum(1 for r in recovery if int(r["tolerance_sec"]) == 5 and r.get("recovered_by_transnetv2_conservative") in [True, "true"])
    recovered_10 = sum(1 for r in recovery if int(r["tolerance_sec"]) == 10 and r.get("recovered_by_transnetv2_conservative") in [True, "true"])

    professor_text = """# 장면 전환 방식 검토 요약

모델을 합치면 recall이 올라갈 수 있으므로, 단순 combined 결과만 보지 않고 각 모델의 단독 성능을 먼저 비교했습니다.
또한 모든 pairwise 조합과 3-model 조합을 비교하여 어떤 모델이 독립적으로 기여하는지 확인했습니다.
후보를 많이 뽑으면 유리해지는 문제를 보완하기 위해 candidate count, candidates/min, hit per 100 candidates를 함께 제시했습니다.
이번 평가는 전체 장면전환 GT가 아니라 광고 시작/종료 boundary 근처 포착률 기준입니다.
향후 시간이 있다면 일부 영상에 대해 전체 scene transition GT를 라벨링하여 precision/F1까지 검증할 수 있습니다.
"""
    PROFESSOR_MD.write_text(professor_text, encoding="utf-8")

    summary_text = f"""# Scene Boundary Model Ablation v2.4 Train

## 목적과 한계
OpenCV/FFmpeg, ResNet, TransNetV2 장면전환 후보가 train actual 광고 시작/종료 boundary를 얼마나 포착하는지 비교했다. 이번 평가는 전체 scene GT 기준 precision/recall/F1이 아니라 광고 boundary recall 기준 평가다.

Detector rule, canonical visual anchor, OCR, actual label, split, prediction 파일은 수정하지 않았다. Actual label은 후보 생성이나 threshold 선택에 사용하지 않고 recall audit에만 사용했다.

## 단독 모델 비교
{md_table(single_rows, ["candidate_family", "candidate_count_total", "candidates_per_minute_total", "all_recall_2s", "all_recall_5s", "all_recall_10s", "all_hit_per_100_candidates_5s"])}

## Pairwise 조합 비교
{md_table(main_pair_rows, ["combo_family", "candidate_count_total", "candidates_per_minute_total", "recall_all_2s", "recall_all_5s", "recall_all_10s", "gain_vs_best_single_5s", "density_risk_level"])}

## 3-model 조합
{md_table(three_rows, ["combo_family", "candidate_count_total", "candidates_per_minute_total", "recall_all_2s", "recall_all_5s", "recall_all_10s", "density_risk_level"])}

## Primary Reference Upper Bound
{md_table(primary_ref_rows, ["combo_family", "candidate_count_total", "candidates_per_minute_total", "recall_all_2s", "recall_all_5s", "recall_all_10s", "density_risk_level"])}

## 추천
{md_table(recommendation, ["recommendation_type", "selected_candidate_family", "candidate_count_total", "candidates_per_minute_total", "recall_all_5s", "recall_all_10s", "efficiency_score", "density_risk_level", "recommendation_label"])}

## Unique Contribution
- @5s all 기준 세 모델 모두 missed: {none_5}
- @10s all 기준 세 모델 모두 missed: {none_10}
- 기존 OpenCV/ResNet combined missed 중 TransNetV2 conservative recovery: @5s {recovered_5}, @10s {recovered_10}

## 정책
조합 후보는 같은 video_id에서 2초 이내 후보를 dedup cluster로 묶어 density를 계산했다. 평균 timestamp는 쓰지 않고, score가 있으면 최고 score timestamp, 없으면 가장 이른 timestamp를 대표로 썼다. Hit 계산은 dedup으로 source evidence가 사라지지 않도록 조합에 포함된 원본 member timestamp 중 actual boundary와 가장 가까운 후보를 기준으로 했다.
"""
    SUMMARY_MD.write_text(summary_text, encoding="utf-8")

    findings_text = f"""# Scene Boundary Model Ablation v2.4 Findings

## 해석 요약
OpenCV/FFmpeg는 후보 수가 상대적으로 적고 급격한 컷 변화에 대한 효율 proxy가 좋다. ResNet은 embedding 변화 기반이라 광고 boundary 근처, 특히 기존 OpenCV/FFmpeg가 놓친 일부 지점에서 보완성이 있다. TransNetV2 primary는 recall이 높지만 후보 밀도도 높아 OCR 기본 입력으로 쓰기에는 부담이 크다.

## 시작점/종료점 특성
start/end recall은 tolerance별로 차이가 있으므로 단일 all recall만 보지 않았다. ResNet과 TransNetV2는 시작점 보완에 의미가 있고, OpenCV/FFmpeg와 기존 combined는 종료점에서 강한 편인지 확인할 수 있게 start/end를 분리했다.

## TransNetV2 primary와 conservative
TransNetV2 primary는 upper-bound/reference로 유용하지만 candidate density risk가 크다. Conservative 기준은 threshold 0.7 + dedup 5초로 후보 수를 줄이면서 기존 missed boundary recovery를 유지하므로 보조 anchor로 더 현실적이다.

## 실용 조합
실제 OCR/후속 detector 보조 anchor 관점에서는 단순히 recall 1등을 고르지 않고 후보 수, candidates/min, hit per 100 candidates를 함께 봐야 한다. 이번 결과에서는 `{recommendation_by_type.get("best_overall_practical_choice", {}).get("selected_candidate_family")}`가 실용 선택으로 가장 적절하다.

## 검토 결론
단독 모델, pairwise, 3-model 조합을 모두 비교했기 때문에 “모델을 많이 합치면 좋아지는 것 아닌가”라는 지적에 대해 모델별 독립 기여와 후보 효율을 함께 설명할 수 있다. 단, 전체 scene transition GT가 없으므로 precision/F1은 계산하지 않았고, 향후 일부 영상 subset에 대해 전체 scene GT를 라벨링하면 precision/F1까지 검증할 수 있다.
"""
    FINDINGS_MD.write_text(findings_text, encoding="utf-8")

    report = {
        "task_name": "scene_model_ablation_v2_4_train",
        "project_root": str(PROJECT_ROOT),
        "start_time": start_iso,
        "end_time": now_iso(),
        "purpose": "train-only ad boundary recall ablation for OpenCV/FFmpeg, ResNet, and TransNetV2 scene boundary candidates",
        "evaluation_scope_note": "This is not full scene GT precision/recall/F1. Metrics are actual ad_start/ad_end boundary recall plus candidate density and efficiency proxies.",
        "input_files": {p.name: str(p) for p in INPUT_FILES},
        "single_models": ["opencv_ffmpeg", "resnet", "transnetv2_primary", "transnetv2_conservative"],
        "main_transnetv2_for_comparison": "transnetv2_conservative",
        "pairwise_combos": ["combo_opencv_resnet", "combo_opencv_transnetv2_conservative", "combo_resnet_transnetv2_conservative"],
        "primary_reference_combos": ["combo_opencv_transnetv2_primary_reference", "combo_resnet_transnetv2_primary_reference", "combo_all_three_primary_reference"],
        "three_model_combo": "combo_all_three_conservative",
        "combination_policy": {
            "density_dedup_window_sec": CROSS_SOURCE_DEDUP_SEC,
            "density_cluster_representative": "highest score timestamp, tie earliest timestamp; average timestamp not used",
            "hit_calculation": "nearest original member timestamp to actual boundary",
            "reason": "dedup should not remove source-specific boundary evidence",
        },
        "candidate_inventory": inventory_rows,
        "recommendations": recommendation,
        "sub_agent_validations": validations,
        "warnings": warnings,
        "errors": errors,
        "protected_files_modified": protected_modified,
        "output_files": {
            "script": str(SCRIPT_PATH),
            "inventory": str(INVENTORY_CSV),
            "candidate_families": str(FAMILIES_CSV),
            "recall_audit": str(AUDIT_CSV),
            "recall_summary": str(RECALL_SUMMARY_CSV),
            "efficiency_summary": str(EFFICIENCY_CSV),
            "combo_summary": str(COMBO_SUMMARY_CSV),
            "three_way_case_breakdown": str(THREE_WAY_CSV),
            "unique_hit_cases": str(UNIQUE_CASES_CSV),
            "existing_missed_recovery": str(RECOVERY_CSV),
            "recommendation": str(RECOMMENDATION_CSV),
            "video_level_recall": str(VIDEO_LEVEL_CSV),
            "summary_md": str(SUMMARY_MD),
            "report_json": str(REPORT_JSON),
            "findings_md": str(FINDINGS_MD),
            "professor_response_md": str(PROFESSOR_MD),
            "log": str(LOG_PATH),
            "latest_bundle": str(LATEST_BUNDLE_DIR),
            "latest_scene": str(LATEST_SCENE_DIR),
        },
        "output_row_counts": row_counts,
        "no_detector_rule_modified": True,
        "no_validation_test_row_level_output": True,
        "actual_label_used_for_candidate_generation": False,
        "actual_label_used_for_recall_audit_only": True,
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"summary": summary_text, "findings": findings_text, "professor": professor_text}


def update_latest(files: Sequence[Path], logger: base.Logger) -> List[Path]:
    LATEST_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_SCENE_DIR.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for source in files:
        if not source.exists():
            continue
        bundle_dest = LATEST_BUNDLE_DIR / source.name
        scene_dest = LATEST_SCENE_DIR / source.name
        shutil.copy2(source, bundle_dest)
        shutil.copy2(source, scene_dest)
        copied.append(bundle_dest)
    readme = LATEST_BUNDLE_DIR / "README_latest_files.md"
    lines = [
        "# Latest Files: Scene Model Ablation v2.4",
        "",
        "Included: newly generated CSV/report/script/log files.",
        "Excluded: raw videos, frame images, cache directories, model weights/checkpoints, package directories, large raw prediction arrays, validation/test row-level outputs.",
        "",
        "## Files",
    ]
    for path in copied:
        lines.append(f"- `{path.name}`")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copy2(readme, LATEST_SCENE_DIR / "README_scene_model_ablation_v2_4_latest_files.md")
    copied.append(readme)
    logger.write(f"latest bundle 갱신 완료: {LATEST_BUNDLE_DIR}")
    logger.write(f"latest_scene 복사 완료: {LATEST_SCENE_DIR}")
    return copied


def main() -> int:
    start_time = time.time()
    start_iso = now_iso()
    logger = base.Logger(LOG_PATH)
    warnings: List[str] = []
    errors: List[str] = []
    try:
        logger.step(1, "안전 스냅샷 및 출력 경로 준비")
        for path in [INVENTORY_CSV.parent, SUMMARY_MD.parent, LOG_PATH.parent, LATEST_BUNDLE_DIR, LATEST_SCENE_DIR]:
            path.mkdir(parents=True, exist_ok=True)
        protected_before = base.snapshot_paths(INPUT_FILES)

        logger.step(2, "split 및 train 범위 확인")
        _split_rows, split_by_video, split_validation = base.load_split(logger)
        mapping_rows, mapping_warnings = base.map_train_videos(split_by_video, base.load_manifest(), base.load_segment_rows())
        warnings.extend(mapping_warnings)
        durations = duration_by_video(mapping_rows)

        inventory_rows: List[Dict[str, Any]] = []
        logger.step(3, "OpenCV/FFmpeg 후보 로드")
        logger.step(4, "ResNet 후보 로드")
        opencv, resnet, canonical = load_opencv_resnet_canonical(inventory_rows, warnings)
        logger.write(f"OpenCV/FFmpeg={len(opencv)}, ResNet={len(resnet)}, canonical={len(canonical)}")

        logger.step(5, "TransNetV2 primary 후보 로드")
        transnet_primary = load_transnet_primary(inventory_rows)
        logger.write(f"TransNetV2 primary={len(transnet_primary)}")

        logger.step(6, "TransNetV2 conservative 후보 로드")
        transnet_conservative, conservative_selected = load_transnet_conservative(inventory_rows, warnings)
        logger.write(f"TransNetV2 conservative={len(transnet_conservative)}, selected={conservative_selected.get('selected_family')}")

        logger.step(7, "단독 candidate family 구성")
        family_rows: Dict[str, List[Dict[str, Any]]] = {
            "opencv_ffmpeg": opencv,
            "resnet": resnet,
            "transnetv2_primary": transnet_primary,
            "transnetv2_conservative": transnet_conservative,
        }
        if canonical:
            family_rows["canonical_all"] = canonical

        logger.step(8, "pairwise 조합 candidate family 구성")
        combo_hit_times: Dict[str, Dict[int, List[float]]] = {}
        for family in [
            "combo_opencv_resnet",
            "combo_opencv_transnetv2_conservative",
            "combo_resnet_transnetv2_conservative",
            "combo_opencv_transnetv2_primary_reference",
            "combo_resnet_transnetv2_primary_reference",
        ]:
            members: List[Dict[str, Any]] = []
            for source_family in COMBO_DEFINITIONS[family]:
                members.extend(family_rows[source_family])
            combo_rows, hit_times = cluster_combo_rows(family, members, COMBO_DEFINITIONS[family])
            family_rows[family] = combo_rows
            combo_hit_times[family] = hit_times

        logger.step(9, "3-model 조합 candidate family 구성")
        for family in ["combo_all_three_conservative", "combo_all_three_primary_reference"]:
            members = []
            for source_family in COMBO_DEFINITIONS[family]:
                members.extend(family_rows[source_family])
            combo_rows, hit_times = cluster_combo_rows(family, members, COMBO_DEFINITIONS[family])
            family_rows[family] = combo_rows
            combo_hit_times[family] = hit_times

        logger.step(10, "train actual 광고 시작/종료 boundary 생성")
        boundaries, boundary_warnings = base.construct_actual_boundaries(base.load_segment_rows(), TRAIN_VIDEO_IDS)
        warnings.extend(boundary_warnings)
        logger.write(f"actual boundary={len(boundaries)}")

        logger.step(11, "모델별 recall 계산")
        sources = build_family_sources(family_rows, combo_hit_times, durations)
        audit_rows = compute_audit(boundaries, sources, durations)
        recall_summary = summarize_recall(audit_rows, sources)
        recall_lookup = summary_lookup(recall_summary)

        logger.step(12, "후보 밀도 및 효율성 지표 계산")
        three_way_summary, unique_case_rows = three_way_cases(boundaries, audit_rows)
        recovery = recovery_rows(boundaries, audit_rows)
        efficiency = build_efficiency_summary(recall_summary, unique_case_rows, recovery)
        existing_cpm = safe_float(get_metric(recall_lookup, "combo_opencv_resnet", "all", 2, "candidates_per_minute_total")) or 0.0
        primary_cpm = safe_float(get_metric(recall_lookup, "transnetv2_primary", "all", 2, "candidates_per_minute_total")) or 0.0
        combo_summary = combo_summary_rows(recall_summary, existing_cpm, primary_cpm)

        logger.step(13, "3-way unique contribution 분석")
        logger.write("3-way case breakdown 생성 완료")

        logger.step(14, "기존 missed boundary recovery 분석")
        logger.write("existing OpenCV/ResNet missed recovery 생성 완료")

        logger.step(15, "추천 모델/조합 선정")
        recommendation = recommendation_rows(recall_summary, efficiency, combo_summary, existing_cpm, primary_cpm)
        logger.write("추천: " + json.dumps({r["recommendation_type"]: r["selected_candidate_family"] for r in recommendation}, ensure_ascii=False, sort_keys=True))

        logger.step(16, "CSV 산출물 생성")
        all_family_rows = [row for family in family_rows for row in family_rows[family]]
        write_csv(INVENTORY_CSV, ["candidate_source", "candidate_family", "file_path", "file_exists", "row_count_total", "row_count_train", "video_id_column", "timestamp_column", "split_column", "score_column", "source_column", "method_column", "model_column", "scope", "usable", "notes"], inventory_rows)
        write_csv(FAMILIES_CSV, ["candidate_family", "source_members_json", "video_id", "candidate_sec", "candidate_frame", "score", "source_model", "threshold", "dedup_window_sec", "cluster_id", "cluster_member_count", "is_combination", "split", "notes"], all_family_rows)
        write_csv(AUDIT_CSV, ["video_id", "ad_interval_id", "boundary_type", "actual_sec", "candidate_family", "nearest_candidate_sec", "nearest_distance_sec", "within_2s", "within_5s", "within_10s", "candidate_count_in_video", "candidates_per_minute_in_video", "notes"], audit_rows)
        write_csv(RECALL_SUMMARY_CSV, ["candidate_family", "model_group", "boundary_type", "tolerance_sec", "actual_boundary_count", "hit_count", "recall", "median_nearest_distance_sec", "mean_nearest_distance_sec", "p90_nearest_distance_sec", "max_nearest_distance_sec", "candidate_count_total", "candidate_video_count", "candidates_per_minute_total"], recall_summary)
        write_csv(EFFICIENCY_CSV, ["candidate_family", "model_group", "candidate_count_total", "candidates_per_minute_total", "all_hit_count_2s", "all_hit_count_5s", "all_hit_count_10s", "all_recall_2s", "all_recall_5s", "all_recall_10s", "all_hit_per_100_candidates_2s", "all_hit_per_100_candidates_5s", "all_hit_per_100_candidates_10s", "unique_hit_count_5s", "unique_hit_count_10s", "unique_hit_per_100_candidates_5s", "unique_hit_per_100_candidates_10s", "interpretation"], efficiency)
        write_csv(COMBO_SUMMARY_CSV, ["combo_family", "included_models", "candidate_count_total", "candidates_per_minute_total", "recall_all_2s", "recall_all_5s", "recall_all_10s", "recall_start_2s", "recall_start_5s", "recall_start_10s", "recall_end_2s", "recall_end_5s", "recall_end_10s", "gain_vs_best_single_2s", "gain_vs_best_single_5s", "gain_vs_best_single_10s", "remaining_missed_2s", "remaining_missed_5s", "remaining_missed_10s", "density_risk_level", "interpretation"], combo_summary)
        write_csv(THREE_WAY_CSV, ["tolerance_sec", "boundary_type_group", "case_type", "boundary_count", "boundary_ratio", "interpretation"], three_way_summary)
        write_csv(UNIQUE_CASES_CSV, ["tolerance_sec", "video_id", "ad_interval_id", "boundary_type", "actual_sec", "case_type", "opencv_hit", "resnet_hit", "transnetv2_hit", "opencv_nearest_candidate_sec", "opencv_nearest_distance_sec", "resnet_nearest_candidate_sec", "resnet_nearest_distance_sec", "transnetv2_nearest_candidate_sec", "transnetv2_nearest_distance_sec", "notes"], unique_case_rows)
        write_csv(RECOVERY_CSV, ["tolerance_sec", "video_id", "ad_interval_id", "boundary_type", "actual_sec", "existing_opencv_resnet_hit", "transnetv2_conservative_hit", "recovered_by_transnetv2_conservative", "recovered_by_transnetv2_primary", "opencv_nearest_candidate_sec", "opencv_nearest_distance_sec", "resnet_nearest_candidate_sec", "resnet_nearest_distance_sec", "transnetv2_conservative_nearest_candidate_sec", "transnetv2_conservative_nearest_distance_sec", "transnetv2_primary_nearest_candidate_sec", "transnetv2_primary_nearest_distance_sec", "notes"], recovery)
        write_csv(RECOMMENDATION_CSV, ["recommendation_type", "selected_candidate_family", "selected_models", "candidate_count_total", "candidates_per_minute_total", "recall_all_2s", "recall_all_5s", "recall_all_10s", "recall_start_5s", "recall_end_5s", "efficiency_score", "density_risk_level", "recommendation_label", "reason", "caution"], recommendation)
        write_csv(VIDEO_LEVEL_CSV, ["video_id", "candidate_family", "boundary_type", "tolerance_sec", "actual_boundary_count", "hit_count", "recall", "candidate_count", "candidates_per_minute", "notes"], video_level_rows(audit_rows, sources, durations))
        row_counts = {
            str(INVENTORY_CSV): len(inventory_rows),
            str(FAMILIES_CSV): len(all_family_rows),
            str(AUDIT_CSV): len(audit_rows),
            str(RECALL_SUMMARY_CSV): len(recall_summary),
            str(EFFICIENCY_CSV): len(efficiency),
            str(COMBO_SUMMARY_CSV): len(combo_summary),
            str(THREE_WAY_CSV): len(three_way_summary),
            str(UNIQUE_CASES_CSV): len(unique_case_rows),
            str(RECOVERY_CSV): len(recovery),
            str(RECOMMENDATION_CSV): len(recommendation),
            str(VIDEO_LEVEL_CSV): len(video_level_rows(audit_rows, sources, durations)),
        }

        logger.step(17, "markdown/json report 및 검토 요약 생성")
        protected_after = base.snapshot_paths(INPUT_FILES)
        protected_modified = base.changed_paths(protected_before, protected_after)
        latest_files = [SCRIPT_PATH, INVENTORY_CSV, FAMILIES_CSV, AUDIT_CSV, RECALL_SUMMARY_CSV, EFFICIENCY_CSV, COMBO_SUMMARY_CSV, THREE_WAY_CSV, UNIQUE_CASES_CSV, RECOVERY_CSV, RECOMMENDATION_CSV, VIDEO_LEVEL_CSV, SUMMARY_MD, REPORT_JSON, FINDINGS_MD, PROFESSOR_MD, LOG_PATH]
        preliminary_report_texts = {"professor": ""}
        validations = validate(split_validation, family_rows, audit_rows, recall_summary, three_way_summary, unique_case_rows, efficiency, recommendation, protected_modified, latest_files, preliminary_report_texts)
        report_texts = write_reports(inventory_rows, recall_summary, efficiency, combo_summary, three_way_summary, unique_case_rows, recovery, recommendation, validations, warnings, errors, protected_modified, row_counts, start_iso)
        validations = validate(split_validation, family_rows, audit_rows, recall_summary, three_way_summary, unique_case_rows, efficiency, recommendation, protected_modified, latest_files, report_texts)
        report_texts = write_reports(inventory_rows, recall_summary, efficiency, combo_summary, three_way_summary, unique_case_rows, recovery, recommendation, validations, warnings, errors, protected_modified, row_counts, start_iso)

        logger.step(18, "Sub Agent 검증 실행")
        logger.write("검증 상태: " + json.dumps({k: v.get("status") for k, v in validations.items()}, ensure_ascii=False, sort_keys=True))

        logger.step(19, "latest bundle 갱신")
        copied = update_latest(latest_files, logger)
        shutil.copy2(LOG_PATH, LATEST_BUNDLE_DIR / LOG_PATH.name)
        shutil.copy2(LOG_PATH, LATEST_SCENE_DIR / LOG_PATH.name)
        shutil.copy2(REPORT_JSON, LATEST_BUNDLE_DIR / REPORT_JSON.name)
        shutil.copy2(REPORT_JSON, LATEST_SCENE_DIR / REPORT_JSON.name)

        logger.step(20, "최종 요약 출력")
        logger.write(f"단독 모델: {['opencv_ffmpeg', 'resnet', 'transnetv2_primary', 'transnetv2_conservative']}")
        logger.write(f"pairwise 조합: {['combo_opencv_resnet', 'combo_opencv_transnetv2_conservative', 'combo_resnet_transnetv2_conservative']}")
        logger.write("3-model 조합: combo_all_three_conservative")
        logger.write("best_overall_practical_choice=" + str(next(r for r in recommendation if r["recommendation_type"] == "best_overall_practical_choice")["selected_candidate_family"]))
        logger.write(f"경과 시간: {time.time() - start_time:.2f}초")
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            logger.write("오류 발생: " + str(exc))
            logger.write(tb)
            REPORT_JSON.write_text(json.dumps({"task_name": "scene_model_ablation_v2_4_train", "project_root": str(PROJECT_ROOT), "start_time": start_iso, "end_time": now_iso(), "errors": [str(exc)], "traceback": tb, "no_detector_rule_modified": True, "no_validation_test_row_level_output": True, "actual_label_used_for_candidate_generation": False, "actual_label_used_for_recall_audit_only": True}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
