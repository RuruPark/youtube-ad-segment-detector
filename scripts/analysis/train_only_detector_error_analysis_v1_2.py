#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train-only error analysis for state-machine detector v1.2.

This script does not run the detector, modify detector rules, extract features,
or use validation/test rows. It reads existing v1.2 train/validation outputs,
filters to train only, and compares train predictions/open candidates with train
actual ad intervals for debugging before any v1.3 rule design.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

TASK_NAME = "train_only_detector_error_analysis_v1_2"
VERSION = "train_only_error_analysis_v1_2"
PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
RUN_SPLIT = "train"
EXCLUDED_SPLITS = {"validation", "test"}
EXPECTED_SPLITS = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}
EXPECTED_SEED = "20240524"
FORBIDDEN_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".wav", ".mp3", ".m4a", ".jpg", ".jpeg", ".png", ".webp",
    ".parquet", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".onnx",
}
FORBIDDEN_DIR_PARTS = {"cache", "frames", "frame_images", "raw_video", "video_proxy", "model_cache", "tmp", "__pycache__"}
INPUT_FILES = {
    "split_file": "data/splits/video_split_v2_4.csv",
    "actual_label_file": "data/segments/ad_interval_segments_v2_4.csv",
    "v1_2_prediction_csv": "data/predictions/state_machine_interval_predictions_v1_2_train_val.csv",
    "v1_2_open_csv": "data/predictions/state_machine_open_interval_candidates_v1_2_train_val.csv",
    "v1_2_trace_csv": "data/predictions/state_machine_anchor_trace_v1_2_train_val.csv",
    "v1_2_audit_csv": "data/predictions/state_machine_detector_validation_audit_v1_2_train_val.csv",
    "v1_2_audio_only_review_csv": "data/predictions/state_machine_audio_only_start_review_candidates_v1_2_train_val.csv",
    "v1_2_long_ad_event_csv": "data/predictions/state_machine_long_ad_end_review_events_v1_2_train_val.csv",
    "v1_2_report_json": "reports/detectors/state_machine_interval_detector_v1_2_report.json",
    "v1_2_summary_md": "reports/detectors/state_machine_interval_detector_v1_2_summary.md",
    "v1_1_prediction_csv": "data/predictions/state_machine_interval_predictions_v1_1_train_val.csv",
    "v1_1_open_csv": "data/predictions/state_machine_open_interval_candidates_v1_1_train_val.csv",
    "v1_1_trace_csv": "data/predictions/state_machine_anchor_trace_v1_1_train_val.csv",
    "v1_1_report_json": "reports/detectors/state_machine_interval_detector_v1_1_report.json",
}
OUTPUT_FILES = {
    "script": "scripts/analysis/train_only_detector_error_analysis_v1_2.py",
    "video_summary": "data/analysis/train_only_detector_error_video_summary_v1_2.csv",
    "interval_overlap": "data/analysis/train_only_detector_error_interval_overlap_v1_2.csv",
    "open_summary": "data/analysis/train_only_detector_error_open_interval_summary_v1_2.csv",
    "trace_reason_summary": "data/analysis/train_only_detector_error_trace_reason_summary_v1_2.csv",
    "state_transition_summary": "data/analysis/train_only_detector_error_state_transition_summary_v1_2.csv",
    "worst_cases": "data/analysis/train_only_detector_error_worst_cases_v1_2.csv",
    "rule_issue_candidates": "data/analysis/train_only_detector_error_rule_issue_candidates_v1_2.csv",
    "trace_focus_windows": "data/analysis/train_only_detector_error_trace_focus_windows_v1_2.csv",
    "v1_1_vs_v1_2_train_comparison": "data/analysis/train_only_detector_v1_1_vs_v1_2_train_comparison.csv",
    "summary_md": "reports/analysis/train_only_detector_error_analysis_v1_2_summary.md",
    "report_json": "reports/analysis/train_only_detector_error_analysis_v1_2_report.json",
    "rule_diagnosis_md": "reports/analysis/train_only_detector_error_analysis_v1_2_rule_diagnosis.md",
    "run_log": "logs/train_only_detector_error_analysis_v1_2_run_log.txt",
    "latest_bundle": "outputs/latest_for_chatgpt_train_only_detector_error_analysis_v1_2",
}
VIDEO_SUMMARY_COLUMNS = [
    "video_id", "split", "video_duration_sec", "actual_interval_count", "actual_total_duration_sec", "actual_coverage_ratio",
    "closed_prediction_count", "predicted_total_duration_sec", "predicted_coverage_ratio", "open_interval_count", "open_total_duration_proxy_sec", "open_coverage_ratio",
    "predicted_plus_open_duration_sec", "predicted_plus_open_coverage_ratio", "actual_pred_overlap_duration_sec", "actual_pred_overlap_ratio_of_actual", "actual_pred_overlap_ratio_of_prediction",
    "false_positive_duration_sec", "false_positive_ratio_of_video", "missed_actual_duration_sec", "missed_actual_ratio_of_actual", "over_detection_ratio_vs_actual", "open_over_actual_ratio",
    "start_reason_top", "end_reason_top", "end_pending_timeout_count", "end_pending_cancel_count", "in_ad_maintain_count", "most_common_in_ad_maintain_reason",
    "audio_only_start_rejected_count", "long_ad_audio_low_end_count", "severity_level", "review_priority", "diagnosis_note",
]
INTERVAL_OVERLAP_COLUMNS = [
    "video_id", "split", "interval_type", "interval_id", "start_sec", "end_sec", "duration_sec", "best_matching_actual_id", "best_actual_overlap_sec",
    "best_actual_overlap_ratio_of_interval", "best_actual_overlap_ratio_of_actual", "false_positive_duration_sec", "matched_prediction_ids", "missed_actual_duration_sec", "issue_type", "reason", "review_note",
]
OPEN_SUMMARY_COLUMNS = [
    "video_id", "split", "open_candidate_id", "start_sec", "end_proxy_sec", "duration_proxy_sec", "video_duration_sec", "open_coverage_ratio", "overlap_with_actual_sec",
    "overlap_ratio_of_open", "actual_after_open_start_sec", "start_reason", "last_state", "nearest_end_pending_events_count", "end_pending_timeout_after_open_count",
    "end_pending_cancel_after_open_count", "possible_failure_mode", "review_priority", "diagnosis_note",
]
TRACE_REASON_COLUMNS = ["scope", "video_id", "category", "key", "count", "ratio_within_category", "note"]
STATE_TRANSITION_COLUMNS = ["scope", "video_id", "state_before", "state_after", "transition", "count", "ratio_within_scope", "note"]
WORST_CASE_COLUMNS = [
    "rank", "video_id", "split", "severity_level", "review_priority", "video_duration_sec", "actual_coverage_ratio", "predicted_coverage_ratio", "open_coverage_ratio",
    "predicted_plus_open_coverage_ratio", "false_positive_duration_sec", "missed_actual_duration_sec", "over_detection_ratio_vs_actual", "diagnosis_note",
]
RULE_ISSUE_COLUMNS = [
    "issue_id", "issue_name", "evidence_from_train", "affected_video_count", "example_video_ids", "severity", "suspected_rule_area", "suggested_direction", "caution", "requires_manual_review",
]
FOCUS_COLUMNS = [
    "video_id", "split", "focus_type", "focus_time_sec", "window_start_sec", "window_end_sec", "visual_anchor_id", "transition_time_anchor", "state_before", "state_after", "decision_type", "decision_reason",
    "audio_context_level", "audio_before_context_level", "audio_after_context_level", "ocr_context_level", "ocr_context_reliability_level", "c3_flag", "d2_flag", "product_repetition_continues", "pending_elapsed_sec_actual", "pending_timeout_trigger", "note",
]
COMPARISON_COLUMNS = [
    "video_id", "split", "v1_1_predicted_coverage_ratio", "v1_2_predicted_coverage_ratio", "v1_1_open_coverage_ratio", "v1_2_open_coverage_ratio",
    "v1_1_closed_prediction_count", "v1_2_closed_prediction_count", "v1_1_open_interval_count", "v1_2_open_interval_count", "v1_1_end_pending_timeout_count", "v1_2_end_pending_timeout_count",
    "change_summary", "got_better_or_worse", "note",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        f = float(value)
        if math.isnan(f):
            return default
        return f
    except Exception:
        return default


def fmt(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {"exists": True, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256_file(path)}


def append_log(path: Path, message: str) -> None:
    print(message, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{now_iso()} {message}\n")


def read_csv(path: Path, optional: bool = False) -> list[dict[str, str]]:
    if optional and not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f).fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def snapshot_tree(path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("relative_path\tsize\tmtime_ns\n")
        if not path.exists():
            return
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for name in sorted(filenames):
                p = Path(dirpath) / name
                try:
                    st = p.stat()
                    f.write(f"{p.relative_to(path).as_posix()}\t{st.st_size}\t{st.st_mtime_ns}\n")
                except FileNotFoundError:
                    continue


def union_intervals(intervals: list[dict[str, Any]], video_duration: float | None = None) -> list[dict[str, Any]]:
    clean = []
    for item in intervals:
        start = as_float(item.get("start_sec"), 0.0)
        end = as_float(item.get("end_sec"), 0.0)
        if video_duration is not None and video_duration > 0:
            start = max(0.0, min(start, video_duration))
            end = max(0.0, min(end, video_duration))
        if end > start:
            clean.append({**item, "start_sec": start, "end_sec": end, "duration_sec": end - start})
    clean.sort(key=lambda r: (r["start_sec"], r["end_sec"]))
    out: list[dict[str, Any]] = []
    for item in clean:
        if not out or item["start_sec"] > out[-1]["end_sec"]:
            out.append(dict(item))
        else:
            out[-1]["end_sec"] = max(out[-1]["end_sec"], item["end_sec"])
            out[-1]["duration_sec"] = out[-1]["end_sec"] - out[-1]["start_sec"]
            out[-1]["interval_id"] = f"{out[-1].get('interval_id', '')},{item.get('interval_id', '')}".strip(",")
    return out


def duration_sum(intervals: list[dict[str, Any]]) -> float:
    return sum(max(0.0, as_float(i.get("end_sec")) - as_float(i.get("start_sec"))) for i in intervals)


def overlap_len(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def overlap_with_union(interval: dict[str, Any], unioned: list[dict[str, Any]]) -> float:
    s = as_float(interval.get("start_sec"))
    e = as_float(interval.get("end_sec"))
    return sum(overlap_len(s, e, as_float(u.get("start_sec")), as_float(u.get("end_sec"))) for u in unioned)


def overlap_between_unions(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    total = 0.0
    for a in left:
        for b in right:
            total += overlap_len(as_float(a.get("start_sec")), as_float(a.get("end_sec")), as_float(b.get("start_sec")), as_float(b.get("end_sec")))
    return total


def ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0 else num / den


def safe_ratio_or_blank(num: float, den: float) -> float | str:
    return "" if den <= 0 else round(num / den, 6)


def top_counter_value(counter: Counter[str]) -> str:
    if not counter:
        return ""
    key, count = counter.most_common(1)[0]
    return f"{key}:{count}"


def split_rows(rows: list[dict[str, str]], split: str = RUN_SPLIT) -> list[dict[str, str]]:
    return [r for r in rows if r.get("split") == split]


def row_video_id(row: dict[str, Any]) -> str:
    return str(row.get("video_id", "")).strip()


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def is_ad_segment(row: dict[str, str], type_col: str | None) -> bool:
    if type_col is None:
        return True
    value = str(row.get(type_col, "")).strip().lower()
    if not value:
        return True
    excluded = {"non_ad", "random_non_ad", "post_ad", "pre_ad", "context", "non-ad", "not_ad"}
    if value in excluded:
        return False
    if "non" in value and "ad" in value:
        return False
    return value in {"ad_interval", "ad_full", "ad", "advertisement"} or value.endswith("_ad") or "ad_interval" in value


def select_actual_schema(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("Actual label file is empty")
    cols = set(rows[0].keys())
    video_col = "video_id" if "video_id" in cols else None
    start_col = next((c for c in ["ad_start_sec", "start_sec", "segment_start_sec", "interval_start_sec"] if c in cols), None)
    end_col = next((c for c in ["ad_end_sec", "end_sec", "segment_end_sec", "interval_end_sec"] if c in cols), None)
    type_col = next((c for c in ["segment_type", "interval_type", "label", "is_ad"] if c in cols), None)
    if video_col is None or start_col is None or end_col is None:
        raise RuntimeError(f"Could not infer actual interval schema: video={video_col}, start={start_col}, end={end_col}, type={type_col}")
    ad_like_count = sum(1 for r in rows if is_ad_segment(r, type_col))
    if ad_like_count == 0:
        raise RuntimeError(f"Actual interval schema inferred but no clear ad rows found using type column {type_col}")
    return {"video_col": video_col, "start_col": start_col, "end_col": end_col, "type_col": type_col, "ad_like_count": ad_like_count, "columns": sorted(cols)}


def parse_interval_rows(rows: list[dict[str, str]], video_ids: set[str], id_col: str, start_col: str, end_col: str, split_value: str, kind: str, warnings: list[str], reason_cols: list[str] | None = None, video_duration: dict[str, float] | None = None) -> dict[str, list[dict[str, Any]]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(rows, start=1):
        vid = row_video_id(row)
        if vid not in video_ids:
            continue
        start = as_float(row.get(start_col), float("nan"))
        end = as_float(row.get(end_col), float("nan"))
        dur = video_duration.get(vid, 0.0) if video_duration else 0.0
        if math.isnan(start) or math.isnan(end) or end <= start:
            warnings.append(f"invalid_{kind}_interval_excluded:video={vid},row={i},start={row.get(start_col)},end={row.get(end_col)}")
            continue
        if dur > 0:
            start = max(0.0, min(start, dur))
            end = max(0.0, min(end, dur))
        if end <= start:
            warnings.append(f"clipped_{kind}_interval_empty_excluded:video={vid},row={i}")
            continue
        reason = "+".join(str(row.get(c, "")) for c in (reason_cols or []) if row.get(c, ""))
        by_video[vid].append({
            "video_id": vid,
            "split": split_value,
            "interval_type": kind,
            "interval_id": row.get(id_col) or f"{kind}_{vid}_{i:04d}",
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
            "reason": reason,
            "source_row": row,
        })
    return by_video


def start_reason_for_open(open_row: dict[str, Any], trace_by_anchor: dict[tuple[str, str], dict[str, str]]) -> str:
    key = (row_video_id(open_row), str(open_row.get("start_anchor_id", "")))
    trace = trace_by_anchor.get(key)
    if not trace:
        return ""
    return trace.get("decision_reason", "")


def severity_for(actual_cov: float, pred_plus_open_cov: float, open_cov: float, actual_duration: float, pred_plus_open_duration: float, missed_ratio: float) -> str:
    if actual_duration <= 0 and pred_plus_open_duration > 0:
        return "no_actual_but_predicted"
    over_ratio = pred_plus_open_cov / actual_cov if actual_cov > 0 else 0.0
    if pred_plus_open_cov >= 0.5 or (actual_cov > 0 and over_ratio >= 3) or open_cov >= 0.3:
        return "severe_over_detection"
    if pred_plus_open_cov >= 0.25 or (actual_cov > 0 and over_ratio >= 2):
        return "moderate_over_detection"
    if missed_ratio >= 0.5:
        return "missed_actual"
    return "reasonable_or_needs_manual_review"


def priority_for(severity: str, open_cov: float) -> str:
    if severity in {"severe_over_detection", "no_actual_but_predicted"} or open_cov >= 0.3:
        return "high"
    if severity in {"moderate_over_detection", "missed_actual"}:
        return "medium"
    return "low"


def diagnose_video(row: dict[str, Any]) -> str:
    notes = []
    if row["severity_level"] in {"severe_over_detection", "no_actual_but_predicted"}:
        notes.append("predicted/open coverage is much larger than train actual coverage")
    if as_float(row["open_coverage_ratio"]) >= 0.3:
        notes.append("open interval covers a large part of the video")
    if as_float(row["false_positive_duration_sec"]) >= 300:
        notes.append("closed prediction contains large non-overlapping duration")
    if as_float(row["missed_actual_ratio_of_actual"]) >= 0.5:
        notes.append("large portion of actual ad is missed by closed prediction")
    if as_float(row["end_pending_timeout_count"]) > 0 or as_float(row["end_pending_cancel_count"]) > 0:
        notes.append("end_pending timeout/cancel behavior should be inspected")
    return "; ".join(notes) if notes else "manual review recommended"


def best_actual_match(interval: dict[str, Any], actual_intervals: list[dict[str, Any]]) -> tuple[str, float, float, float, dict[str, Any] | None]:
    best = None
    best_overlap = 0.0
    for actual in actual_intervals:
        ov = overlap_len(as_float(interval["start_sec"]), as_float(interval["end_sec"]), as_float(actual["start_sec"]), as_float(actual["end_sec"]))
        if ov > best_overlap:
            best_overlap = ov
            best = actual
    interval_dur = max(0.0, as_float(interval["end_sec"]) - as_float(interval["start_sec"]))
    actual_dur = max(0.0, as_float(best["end_sec"]) - as_float(best["start_sec"])) if best else 0.0
    return (str(best.get("interval_id", "")) if best else "", best_overlap, ratio(best_overlap, interval_dur), ratio(best_overlap, actual_dur), best)


def issue_for_prediction(interval: dict[str, Any], actual: dict[str, Any] | None, overlap_ratio: float, fp: float, is_open: bool = False) -> str:
    if is_open:
        return "open_unresolved_candidate"
    if actual is None or overlap_ratio < 0.1:
        return "false_positive_candidate"
    start = as_float(interval["start_sec"])
    end = as_float(interval["end_sec"])
    a_start = as_float(actual["start_sec"])
    a_end = as_float(actual["end_sec"])
    dur = end - start
    a_dur = a_end - a_start
    if dur > max(120.0, a_dur * 2.0) and fp > max(30.0, a_dur * 0.5):
        return "overlong_prediction"
    if start < a_start - 10:
        return "early_start"
    if end > a_end + 10:
        return "late_end"
    return "good_overlap" if overlap_ratio >= 0.5 else "false_positive_candidate"


def build_focus_events(train_ids: list[str], video_summary: list[dict[str, Any]], actual_by_video: dict[str, list[dict[str, Any]]], pred_by_video: dict[str, list[dict[str, Any]]], open_by_video: dict[str, list[dict[str, Any]]], trace_by_video: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    selected: set[str] = set()
    selected.update(str(r["video_id"]) for r in video_summary if r.get("review_priority") == "high")
    for key in ["predicted_plus_open_coverage_ratio", "open_coverage_ratio", "false_positive_duration_sec", "missed_actual_ratio_of_actual"]:
        top = sorted(video_summary, key=lambda r: as_float(r.get(key), 0.0), reverse=True)[:5]
        selected.update(str(r["video_id"]) for r in top)
    events = []
    for vid in sorted(selected, key=lambda x: int(float(x))):
        trace = trace_by_video.get(vid, [])
        first_in = next((t for t in trace if t.get("state_after") == "in_ad" and t.get("state_before") != "in_ad"), None)
        if first_in:
            events.append({"video_id": vid, "focus_type": "first_in_ad_entry", "time": as_float(first_in.get("transition_time_anchor")), "note": first_in.get("decision_reason", "")})
        for o in open_by_video.get(vid, []):
            events.append({"video_id": vid, "focus_type": "open_interval_start", "time": as_float(o.get("start_sec")), "note": str(o.get("interval_id", ""))})
        for t in trace:
            if t.get("decision_type") == "end_pending_timeout":
                events.append({"video_id": vid, "focus_type": "end_pending_timeout", "time": as_float(t.get("transition_time_anchor")), "note": t.get("decision_reason", "")})
            if t.get("decision_type") == "end_pending_cancelled":
                events.append({"video_id": vid, "focus_type": "end_pending_cancelled_by_continuity", "time": as_float(t.get("transition_time_anchor")), "note": t.get("decision_reason", "")})
        for a in actual_by_video.get(vid, []):
            events.append({"video_id": vid, "focus_type": "actual_ad_start", "time": as_float(a.get("start_sec")), "note": str(a.get("interval_id", ""))})
            events.append({"video_id": vid, "focus_type": "actual_ad_end", "time": as_float(a.get("end_sec")), "note": str(a.get("interval_id", ""))})
        for p in pred_by_video.get(vid, []):
            actual_id, ov, ov_ratio, _, best = best_actual_match(p, actual_by_video.get(vid, []))
            fp = max(0.0, as_float(p.get("duration_sec")) - ov)
            if issue_for_prediction(p, best, ov_ratio, fp, False) == "overlong_prediction" or fp >= 300:
                events.append({"video_id": vid, "focus_type": "overlong_prediction_start", "time": as_float(p.get("start_sec")), "note": str(p.get("interval_id", ""))})
                events.append({"video_id": vid, "focus_type": "overlong_prediction_end", "time": as_float(p.get("end_sec")), "note": str(p.get("interval_id", ""))})
    # video/type/time을 초 단위로 반올림해 focus event 중복을 제거한다.
    dedup = {}
    for e in events:
        key = (e["video_id"], e["focus_type"], round(e["time"], 1))
        dedup[key] = e
    return list(dedup.values())


def scan_forbidden_bundle(bundle: Path) -> list[str]:
    found = []
    if not bundle.exists():
        return found
    for path in bundle.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(bundle)
        parts = {part.lower() for part in rel.parts}
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or parts & FORBIDDEN_DIR_PARTS:
            found.append(str(rel))
    return found


def refresh_latest_bundle(root: Path, output_files: dict[str, str], log_path: Path) -> list[str]:
    append_log(log_path, "[STEP 14] Update latest bundle")
    bundle = root / output_files["latest_bundle"]
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    keys = [
        "video_summary", "interval_overlap", "open_summary", "trace_reason_summary", "state_transition_summary", "worst_cases", "rule_issue_candidates", "trace_focus_windows", "v1_1_vs_v1_2_train_comparison",
        "summary_md", "report_json", "rule_diagnosis_md", "run_log", "script",
    ]
    copied = []
    for key in keys:
        src = root / output_files[key]
        if src.exists():
            shutil.copy2(src, bundle / src.name)
            copied.append(src.name)
    readme = bundle / "README_latest_files.md"
    readme.write_text("""# Latest Files: Train-Only Detector Error Analysis v1.2

- train-only detector error analysis.
- Not detector implementation.
- Not a final performance report.
- Validation/test rows are excluded.
- No media/video/frame/cache/model/raw video/proxy/checkpoint files are copied.
- Actual labels are used only for train-only debugging/audit, not detector decisions.

## Files

""" + "\n".join(f"- `{name}`" for name in sorted(copied + ["README_latest_files.md"])) + "\n", encoding="utf-8")
    return scan_forbidden_bundle(bundle)


def backup_existing_targets(root: Path, timestamp: str, log_path: Path) -> dict[str, Any]:
    backup_dir = root / "backups" / f"train_only_detector_error_analysis_v1_2_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed = []
    target_rels = [rel for key, rel in OUTPUT_FILES.items() if key != "latest_bundle"]
    for rel in target_rels:
        p = root / rel
        if p.exists():
            dest = backup_dir / "existing_files" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            backed.append(rel)
    latest = root / OUTPUT_FILES["latest_bundle"]
    if latest.exists():
        dest = backup_dir / "existing_latest_for_chatgpt_train_only_detector_error_analysis_v1_2"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(latest, dest)
        backed.append(OUTPUT_FILES["latest_bundle"])
    manifest = {"timestamp": timestamp, "backup_dir": str(backup_dir), "backed_up_existing_targets": backed, "target_paths": target_rels + [OUTPUT_FILES["latest_bundle"]]}
    (backup_dir / "backup_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_log(log_path, f"[STEP 01] Existing analysis target backup completed: backed_up_count={len(backed)}, backup_dir={backup_dir}")
    return manifest


def validate_split(split_rows: list[dict[str, str]]) -> dict[str, Any]:
    observed: dict[str, list[int]] = defaultdict(list)
    seeds = set()
    durations: dict[str, float] = {}
    for row in split_rows:
        split = row.get("split", "")
        if split in EXPECTED_SPLITS:
            observed[split].append(int(float(row.get("video_id", -1))))
        seeds.add(str(row.get("split_seed", "")))
        durations[str(row.get("video_id", ""))] = as_float(row.get("video_duration_sec"), 0.0)
    observed_sorted = {k: sorted(v) for k, v in observed.items()}
    ok = observed_sorted == EXPECTED_SPLITS and seeds == {EXPECTED_SEED}
    return {"ok": ok, "observed": observed_sorted, "seed_values": sorted(seeds), "video_durations": durations}


def run_analysis(root: Path, timestamp: str) -> dict[str, Any]:
    if str(root.resolve()) != str(PROJECT_ROOT):
        raise RuntimeError(f"Refusing to run outside project root: {root}")
    log_path = root / OUTPUT_FILES["run_log"]
    warnings: list[str] = []
    errors: list[str] = []
    append_log(log_path, "[STEP 01] Safety snapshot and backup")
    old_before = root / "reports/analysis/old_project_snapshot_before_train_only_detector_error_analysis_v1_2.tsv"
    snapshot_tree(OLD_PROJECT_ROOT, old_before)
    backup_manifest = backup_existing_targets(root, timestamp, log_path)
    required_inputs = [
        "split_file", "actual_label_file", "v1_2_prediction_csv", "v1_2_open_csv", "v1_2_trace_csv", "v1_2_audit_csv", "v1_2_audio_only_review_csv", "v1_2_long_ad_event_csv", "v1_2_report_json", "v1_2_summary_md",
    ]
    input_stats_before = {key: file_stat(root / INPUT_FILES[key]) for key in required_inputs + ["v1_1_prediction_csv", "v1_1_open_csv", "v1_1_trace_csv", "v1_1_report_json"]}
    input_stats_path = root / "reports/analysis/train_only_detector_error_analysis_v1_2_input_file_stats_before.json"
    input_stats_path.parent.mkdir(parents=True, exist_ok=True)
    input_stats_path.write_text(json.dumps(input_stats_before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    missing = [key for key in required_inputs if not (root / INPUT_FILES[key]).exists()]
    if missing:
        raise RuntimeError(f"Missing required inputs: {missing}")

    append_log(log_path, "[STEP 02] Load inputs")
    split_rows = read_csv(root / INPUT_FILES["split_file"])
    split_check = validate_split(split_rows)
    if not split_check["ok"]:
        raise RuntimeError(f"Fixed split mismatch: {split_check}")
    video_duration = split_check["video_durations"]
    train_ids = [str(v) for v in EXPECTED_SPLITS["train"]]
    train_id_set = set(train_ids)
    actual_rows_all = read_csv(root / INPUT_FILES["actual_label_file"])
    pred_rows_all = read_csv(root / INPUT_FILES["v1_2_prediction_csv"])
    open_rows_all = read_csv(root / INPUT_FILES["v1_2_open_csv"])
    trace_rows_all = read_csv(root / INPUT_FILES["v1_2_trace_csv"])
    audit_rows_all = read_csv(root / INPUT_FILES["v1_2_audit_csv"])
    audio_review_all = read_csv(root / INPUT_FILES["v1_2_audio_only_review_csv"])
    long_events_all = read_csv(root / INPUT_FILES["v1_2_long_ad_event_csv"])
    v11_pred_all = read_csv(root / INPUT_FILES["v1_1_prediction_csv"], optional=True)
    v11_open_all = read_csv(root / INPUT_FILES["v1_1_open_csv"], optional=True)
    v11_trace_all = read_csv(root / INPUT_FILES["v1_1_trace_csv"], optional=True)
    pred_rows = [r for r in pred_rows_all if r.get("split") == RUN_SPLIT and row_video_id(r) in train_id_set]
    open_rows = [r for r in open_rows_all if r.get("split") == RUN_SPLIT and row_video_id(r) in train_id_set]
    trace_rows = [r for r in trace_rows_all if r.get("split") == RUN_SPLIT and row_video_id(r) in train_id_set]
    audit_rows = [r for r in audit_rows_all if r.get("split") == RUN_SPLIT and row_video_id(r) in train_id_set]
    audio_review_rows = [r for r in audio_review_all if r.get("split") == RUN_SPLIT and row_video_id(r) in train_id_set]
    long_event_rows = [r for r in long_events_all if r.get("split") == RUN_SPLIT and row_video_id(r) in train_id_set]
    excluded_rows_seen = {
        "v1_2_prediction_validation_test_rows_excluded": len([r for r in pred_rows_all if r.get("split") in EXCLUDED_SPLITS]),
        "v1_2_open_validation_test_rows_excluded": len([r for r in open_rows_all if r.get("split") in EXCLUDED_SPLITS]),
        "v1_2_trace_validation_test_rows_excluded": len([r for r in trace_rows_all if r.get("split") in EXCLUDED_SPLITS]),
        "v1_2_audit_validation_test_rows_excluded": len([r for r in audit_rows_all if r.get("split") in EXCLUDED_SPLITS]),
    }

    append_log(log_path, "[STEP 03] Extract train actual intervals")
    actual_schema = select_actual_schema(actual_rows_all)
    actual_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(actual_rows_all, start=1):
        vid = str(row.get(actual_schema["video_col"], "")).strip()
        if vid not in train_id_set:
            continue
        if not is_ad_segment(row, actual_schema["type_col"]):
            continue
        start = as_float(row.get(actual_schema["start_col"]), float("nan"))
        end = as_float(row.get(actual_schema["end_col"]), float("nan"))
        dur = video_duration.get(vid, 0.0)
        if math.isnan(start) or math.isnan(end) or end <= start:
            warnings.append(f"invalid_actual_interval_excluded:video={vid},row={i}")
            continue
        if dur > 0:
            start = max(0.0, min(start, dur))
            end = max(0.0, min(end, dur))
        if end <= start:
            warnings.append(f"clipped_actual_interval_empty_excluded:video={vid},row={i}")
            continue
        actual_by_video[vid].append({
            "video_id": vid,
            "split": RUN_SPLIT,
            "interval_type": "actual",
            "interval_id": row.get("ad_interval_id") or row.get("segment_id") or f"actual_{vid}_{i:04d}",
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
            "reason": str(row.get(actual_schema["type_col"], "actual_ad_interval")),
            "source_row": row,
        })
    actual_union_by_video = {vid: union_intervals(actual_by_video.get(vid, []), video_duration.get(vid, 0.0)) for vid in train_ids}
    if not any(actual_union_by_video.values()):
        raise RuntimeError("No train actual intervals extracted")

    append_log(log_path, "[STEP 04] Prepare predicted/open intervals")
    pred_by_video = parse_interval_rows(pred_rows, train_id_set, "prediction_id", "ad_start_sec", "ad_end_sec", RUN_SPLIT, "closed_prediction", warnings, ["start_reason", "end_reason"], video_duration)
    open_by_video = parse_interval_rows(open_rows, train_id_set, "open_candidate_id", "ad_start_sec", "last_anchor_sec", RUN_SPLIT, "open_candidate", warnings, ["open_state", "open_reason"], video_duration)
    pred_union_by_video = {vid: union_intervals(pred_by_video.get(vid, []), video_duration.get(vid, 0.0)) for vid in train_ids}
    open_union_by_video = {vid: union_intervals(open_by_video.get(vid, []), video_duration.get(vid, 0.0)) for vid in train_ids}
    combined_union_by_video = {vid: union_intervals(pred_by_video.get(vid, []) + open_by_video.get(vid, []), video_duration.get(vid, 0.0)) for vid in train_ids}
    trace_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    trace_by_anchor: dict[tuple[str, str], dict[str, str]] = {}
    for row in trace_rows:
        vid = row_video_id(row)
        trace_by_video[vid].append(row)
        trace_by_anchor[(vid, str(row.get("visual_anchor_id", "")))] = row
    for vid in trace_by_video:
        trace_by_video[vid].sort(key=lambda r: as_float(r.get("transition_time_anchor"), 0.0))

    append_log(log_path, "[STEP 05] Compute video-level metrics")
    video_summary: list[dict[str, Any]] = []
    for vid in train_ids:
        dur = video_duration.get(vid, 0.0)
        actual_total = duration_sum(actual_union_by_video.get(vid, []))
        pred_total = duration_sum(pred_union_by_video.get(vid, []))
        open_total = duration_sum(open_union_by_video.get(vid, []))
        combined_total = duration_sum(combined_union_by_video.get(vid, []))
        actual_pred_overlap = overlap_between_unions(actual_union_by_video.get(vid, []), pred_union_by_video.get(vid, []))
        fp = max(0.0, pred_total - actual_pred_overlap)
        missed = max(0.0, actual_total - actual_pred_overlap)
        actual_cov = ratio(actual_total, dur)
        pred_cov = ratio(pred_total, dur)
        open_cov = ratio(open_total, dur)
        combined_cov = ratio(combined_total, dur)
        missed_ratio = ratio(missed, actual_total)
        severity = severity_for(actual_cov, combined_cov, open_cov, actual_total, combined_total, missed_ratio)
        priority = priority_for(severity, open_cov)
        trace = trace_by_video.get(vid, [])
        in_ad_maintain_reasons = Counter(r.get("decision_reason", "") for r in trace if r.get("state_before") == "in_ad" and r.get("state_after") == "in_ad")
        row = {
            "video_id": vid,
            "split": RUN_SPLIT,
            "video_duration_sec": fmt(dur, 3),
            "actual_interval_count": len(actual_by_video.get(vid, [])),
            "actual_total_duration_sec": fmt(actual_total, 3),
            "actual_coverage_ratio": fmt(actual_cov),
            "closed_prediction_count": len(pred_by_video.get(vid, [])),
            "predicted_total_duration_sec": fmt(pred_total, 3),
            "predicted_coverage_ratio": fmt(pred_cov),
            "open_interval_count": len(open_by_video.get(vid, [])),
            "open_total_duration_proxy_sec": fmt(open_total, 3),
            "open_coverage_ratio": fmt(open_cov),
            "predicted_plus_open_duration_sec": fmt(combined_total, 3),
            "predicted_plus_open_coverage_ratio": fmt(combined_cov),
            "actual_pred_overlap_duration_sec": fmt(actual_pred_overlap, 3),
            "actual_pred_overlap_ratio_of_actual": fmt(ratio(actual_pred_overlap, actual_total)),
            "actual_pred_overlap_ratio_of_prediction": fmt(ratio(actual_pred_overlap, pred_total)),
            "false_positive_duration_sec": fmt(fp, 3),
            "false_positive_ratio_of_video": fmt(ratio(fp, dur)),
            "missed_actual_duration_sec": fmt(missed, 3),
            "missed_actual_ratio_of_actual": fmt(missed_ratio),
            "over_detection_ratio_vs_actual": safe_ratio_or_blank(combined_cov, actual_cov),
            "open_over_actual_ratio": safe_ratio_or_blank(open_cov, actual_cov),
            "start_reason_top": top_counter_value(Counter(p.get("source_row", {}).get("start_reason", "") for p in pred_by_video.get(vid, []))),
            "end_reason_top": top_counter_value(Counter(p.get("source_row", {}).get("end_reason", "") for p in pred_by_video.get(vid, []))),
            "end_pending_timeout_count": sum(1 for r in trace if r.get("decision_type") == "end_pending_timeout"),
            "end_pending_cancel_count": sum(1 for r in trace if r.get("decision_type") == "end_pending_cancelled"),
            "in_ad_maintain_count": sum(1 for r in trace if r.get("state_before") == "in_ad" and r.get("state_after") == "in_ad"),
            "most_common_in_ad_maintain_reason": top_counter_value(in_ad_maintain_reasons),
            "audio_only_start_rejected_count": sum(1 for r in audio_review_rows if row_video_id(r) == vid),
            "long_ad_audio_low_end_count": sum(1 for r in long_event_rows if row_video_id(r) == vid and is_true(r.get("end_confirmed_by_v1_2"))),
            "severity_level": severity,
            "review_priority": priority,
            "diagnosis_note": "",
        }
        row["diagnosis_note"] = diagnose_video(row)
        video_summary.append(row)
    write_csv(root / OUTPUT_FILES["video_summary"], video_summary, VIDEO_SUMMARY_COLUMNS)
    worst_cases = sorted(video_summary, key=lambda r: (0 if r["review_priority"] == "high" else 1 if r["review_priority"] == "medium" else 2, -as_float(r["predicted_plus_open_coverage_ratio"]), -as_float(r["false_positive_duration_sec"])))
    worst_rows = []
    for rank, row in enumerate(worst_cases, start=1):
        item = {"rank": rank, **row}
        worst_rows.append(item)
    write_csv(root / OUTPUT_FILES["worst_cases"], worst_rows, WORST_CASE_COLUMNS)

    append_log(log_path, "[STEP 06] Compute interval-level overlap")
    interval_overlap_rows: list[dict[str, Any]] = []
    for vid in train_ids:
        actual_union = actual_union_by_video.get(vid, [])
        pred_union = pred_union_by_video.get(vid, [])
        for p in pred_by_video.get(vid, []):
            actual_id, ov, ov_int, ov_actual, best = best_actual_match(p, actual_by_video.get(vid, []))
            fp = max(0.0, as_float(p.get("duration_sec")) - overlap_with_union(p, actual_union))
            issue = issue_for_prediction(p, best, ov_int, fp, False)
            interval_overlap_rows.append({
                "video_id": vid, "split": RUN_SPLIT, "interval_type": "closed_prediction", "interval_id": p.get("interval_id"), "start_sec": fmt(as_float(p.get("start_sec")), 3), "end_sec": fmt(as_float(p.get("end_sec")), 3), "duration_sec": fmt(as_float(p.get("duration_sec")), 3),
                "best_matching_actual_id": actual_id, "best_actual_overlap_sec": fmt(ov, 3), "best_actual_overlap_ratio_of_interval": fmt(ov_int), "best_actual_overlap_ratio_of_actual": fmt(ov_actual),
                "false_positive_duration_sec": fmt(fp, 3), "matched_prediction_ids": "", "missed_actual_duration_sec": "", "issue_type": issue, "reason": p.get("reason", ""), "review_note": "train-only overlap audit; closed prediction only",
            })
        for o in open_by_video.get(vid, []):
            actual_id, ov, ov_int, ov_actual, best = best_actual_match(o, actual_by_video.get(vid, []))
            fp = max(0.0, as_float(o.get("duration_sec")) - overlap_with_union(o, actual_union))
            interval_overlap_rows.append({
                "video_id": vid, "split": RUN_SPLIT, "interval_type": "open_candidate", "interval_id": o.get("interval_id"), "start_sec": fmt(as_float(o.get("start_sec")), 3), "end_sec": fmt(as_float(o.get("end_sec")), 3), "duration_sec": fmt(as_float(o.get("duration_sec")), 3),
                "best_matching_actual_id": actual_id, "best_actual_overlap_sec": fmt(ov, 3), "best_actual_overlap_ratio_of_interval": fmt(ov_int), "best_actual_overlap_ratio_of_actual": fmt(ov_actual),
                "false_positive_duration_sec": fmt(fp, 3), "matched_prediction_ids": "", "missed_actual_duration_sec": "", "issue_type": "open_unresolved_candidate", "reason": o.get("reason", ""), "review_note": "open interval is unresolved candidate, not final prediction",
            })
        for a in actual_by_video.get(vid, []):
            matched = [p.get("interval_id") for p in pred_by_video.get(vid, []) if overlap_len(as_float(a.get("start_sec")), as_float(a.get("end_sec")), as_float(p.get("start_sec")), as_float(p.get("end_sec"))) > 0]
            ov_pred = overlap_with_union(a, pred_union)
            missed = max(0.0, as_float(a.get("duration_sec")) - ov_pred)
            interval_overlap_rows.append({
                "video_id": vid, "split": RUN_SPLIT, "interval_type": "actual", "interval_id": a.get("interval_id"), "start_sec": fmt(as_float(a.get("start_sec")), 3), "end_sec": fmt(as_float(a.get("end_sec")), 3), "duration_sec": fmt(as_float(a.get("duration_sec")), 3),
                "best_matching_actual_id": a.get("interval_id"), "best_actual_overlap_sec": "", "best_actual_overlap_ratio_of_interval": "", "best_actual_overlap_ratio_of_actual": "",
                "false_positive_duration_sec": "", "matched_prediction_ids": ",".join(matched), "missed_actual_duration_sec": fmt(missed, 3), "issue_type": "missed_actual_candidate" if ratio(missed, as_float(a.get("duration_sec"))) >= 0.5 else "good_overlap", "reason": a.get("reason", ""), "review_note": "actual train label used only for debugging/audit",
            })
    write_csv(root / OUTPUT_FILES["interval_overlap"], interval_overlap_rows, INTERVAL_OVERLAP_COLUMNS)

    append_log(log_path, "[STEP 07] Compute open interval summary")
    open_summary_rows: list[dict[str, Any]] = []
    for vid in train_ids:
        dur = video_duration.get(vid, 0.0)
        actual_union = actual_union_by_video.get(vid, [])
        for o in open_by_video.get(vid, []):
            start = as_float(o.get("start_sec"))
            end = as_float(o.get("end_sec"))
            odur = max(0.0, end - start)
            overlap = overlap_with_union(o, actual_union)
            actual_after = 0.0
            for a in actual_union:
                actual_after += overlap_len(start, dur, as_float(a.get("start_sec")), as_float(a.get("end_sec")))
            trace_after = [t for t in trace_by_video.get(vid, []) if as_float(t.get("transition_time_anchor")) >= start]
            nearest_end_pending = [t for t in trace_after if "end_pending" in {t.get("state_before"), t.get("state_after")} or str(t.get("decision_type", "")).startswith("end_pending")]
            timeout_count = sum(1 for t in trace_after if t.get("decision_type") == "end_pending_timeout")
            cancel_count = sum(1 for t in trace_after if t.get("decision_type") == "end_pending_cancelled")
            in_ad_hold = sum(1 for t in trace_after if t.get("state_before") == "in_ad" and t.get("state_after") == "in_ad")
            overlap_ratio_open = ratio(overlap, odur)
            if overlap_ratio_open < 0.1 and actual_after <= 0:
                mode = "wrong_start"
            elif overlap > 0 and odur - overlap > max(60.0, overlap):
                mode = "missed_end"
            elif timeout_count + cancel_count == 0:
                mode = "lack_of_end_evidence"
            elif in_ad_hold > timeout_count + cancel_count:
                mode = "over_persistent_in_ad"
            else:
                mode = "needs_manual_review"
            priority = "high" if ratio(odur, dur) >= 0.3 or mode in {"wrong_start", "missed_end"} else "medium"
            open_summary_rows.append({
                "video_id": vid, "split": RUN_SPLIT, "open_candidate_id": o.get("interval_id"), "start_sec": fmt(start, 3), "end_proxy_sec": fmt(end, 3), "duration_proxy_sec": fmt(odur, 3), "video_duration_sec": fmt(dur, 3), "open_coverage_ratio": fmt(ratio(odur, dur)),
                "overlap_with_actual_sec": fmt(overlap, 3), "overlap_ratio_of_open": fmt(overlap_ratio_open), "actual_after_open_start_sec": fmt(actual_after, 3), "start_reason": start_reason_for_open(o.get("source_row", {}), trace_by_anchor), "last_state": o.get("source_row", {}).get("open_state", ""),
                "nearest_end_pending_events_count": len(nearest_end_pending), "end_pending_timeout_after_open_count": timeout_count, "end_pending_cancel_after_open_count": cancel_count, "possible_failure_mode": mode, "review_priority": priority,
                "diagnosis_note": "open candidate proxy coverage only; unresolved interval separated from closed predictions",
            })
    write_csv(root / OUTPUT_FILES["open_summary"], open_summary_rows, OPEN_SUMMARY_COLUMNS)

    append_log(log_path, "[STEP 08] Compute trace reason and state transition summary")
    trace_summary_rows: list[dict[str, Any]] = []
    state_transition_rows: list[dict[str, Any]] = []

    def add_counter_rows(scope: str, vid: str, category: str, counter: Counter[str], note: str) -> None:
        total = sum(counter.values())
        for key, count in counter.most_common():
            trace_summary_rows.append({"scope": scope, "video_id": vid, "category": category, "key": key, "count": count, "ratio_within_category": fmt(ratio(count, total)), "note": note})

    def summarize_trace(scope: str, vid: str, rows: list[dict[str, str]]) -> None:
        transitions = Counter(f"{r.get('state_before','')}->{r.get('state_after','')}" for r in rows)
        state_total = sum(transitions.values())
        for transition, count in transitions.most_common():
            before, after = transition.split("->", 1)
            state_transition_rows.append({"scope": scope, "video_id": vid, "state_before": before, "state_after": after, "transition": transition, "count": count, "ratio_within_scope": fmt(ratio(count, state_total)), "note": "train trace only"})
        add_counter_rows(scope, vid, "state_transition", transitions, "train trace only")
        add_counter_rows(scope, vid, "decision_type", Counter(r.get("decision_type", "") for r in rows), "train trace only")
        add_counter_rows(scope, vid, "decision_reason", Counter(r.get("decision_reason", "") for r in rows), "train trace only")
        add_counter_rows(scope, vid, "in_ad_maintain_reason", Counter(r.get("decision_reason", "") for r in rows if r.get("state_before") == "in_ad" and r.get("state_after") == "in_ad"), "in_ad->in_ad rows")
        add_counter_rows(scope, vid, "pending_timeout", Counter(r.get("decision_reason", "") for r in rows if r.get("decision_type") in {"start_pending_timeout", "end_pending_timeout", "start_pending_cancelled", "end_pending_cancelled"}), "pending timeout/cancel rows")
        flags = Counter()
        expected_flag_keys = [
            "c3_flag",
            "d2_flag",
            "product_repetition_continues",
            "audio_only_start_rejected",
            "long_ad_audio_low_end_rule_applied",
            "minimum_duration_hold_applied",
            "e2_exception_applied",
        ]
        for key in expected_flag_keys:
            flags[key] = 0
        for r in rows:
            if is_true(r.get("c3_flag")): flags["c3_flag"] += 1
            if is_true(r.get("d2_flag")): flags["d2_flag"] += 1
            if is_true(r.get("product_repetition_continues")): flags["product_repetition_continues"] += 1
            if is_true(r.get("audio_only_start_rejected")): flags["audio_only_start_rejected"] += 1
            if is_true(r.get("long_ad_audio_low_end_rule_applied")): flags["long_ad_audio_low_end_rule_applied"] += 1
            if is_true(r.get("minimum_duration_hold_applied")): flags["minimum_duration_hold_applied"] += 1
            if is_true(r.get("e2_exception_applied")): flags["e2_exception_applied"] += 1
        add_counter_rows(scope, vid, "conflict_flag", flags, "boolean trace flags; zero-count rows retained for audit completeness")
        add_counter_rows(scope, vid, "bridge", Counter({"low_gap_bridge_applied": sum(1 for r in rows if is_true(r.get("low_gap_bridge_applied")))}), "bridge flags; zero-count row retained when absent")
    summarize_trace("overall_train", "", trace_rows)
    for vid in train_ids:
        summarize_trace("video", vid, trace_by_video.get(vid, []))
    write_csv(root / OUTPUT_FILES["trace_reason_summary"], trace_summary_rows, TRACE_REASON_COLUMNS)
    write_csv(root / OUTPUT_FILES["state_transition_summary"], state_transition_rows, STATE_TRANSITION_COLUMNS)

    append_log(log_path, "[STEP 09] Build trace focus windows")
    focus_events = build_focus_events(train_ids, video_summary, actual_by_video, pred_by_video, open_by_video, trace_by_video)
    focus_rows: list[dict[str, Any]] = []
    for event in focus_events:
        vid = event["video_id"]
        focus_time = as_float(event["time"])
        w_start = max(0.0, focus_time - 30.0)
        w_end = focus_time + 30.0
        for t in trace_by_video.get(vid, []):
            tt = as_float(t.get("transition_time_anchor"))
            if w_start <= tt <= w_end:
                focus_rows.append({
                    "video_id": vid, "split": RUN_SPLIT, "focus_type": event["focus_type"], "focus_time_sec": fmt(focus_time, 3), "window_start_sec": fmt(w_start, 3), "window_end_sec": fmt(w_end, 3),
                    "visual_anchor_id": t.get("visual_anchor_id", ""), "transition_time_anchor": fmt(tt, 3), "state_before": t.get("state_before", ""), "state_after": t.get("state_after", ""), "decision_type": t.get("decision_type", ""), "decision_reason": t.get("decision_reason", ""),
                    "audio_context_level": t.get("audio_context_level", ""), "audio_before_context_level": t.get("audio_before_context_level", ""), "audio_after_context_level": t.get("audio_after_context_level", ""), "ocr_context_level": t.get("ocr_context_level", ""), "ocr_context_reliability_level": t.get("ocr_context_reliability_level", ""),
                    "c3_flag": t.get("c3_flag", ""), "d2_flag": t.get("d2_flag", ""), "product_repetition_continues": t.get("product_repetition_continues", ""), "pending_elapsed_sec_actual": t.get("pending_elapsed_sec_actual", ""), "pending_timeout_trigger": t.get("pending_timeout_trigger", ""), "note": event.get("note", ""),
                })
    write_csv(root / OUTPUT_FILES["trace_focus_windows"], focus_rows, FOCUS_COLUMNS)

    append_log(log_path, "[STEP 10] Build v1.1 vs v1.2 train comparison")
    v11_pred_train = [r for r in v11_pred_all if r.get("split") == RUN_SPLIT and row_video_id(r) in train_id_set]
    v11_open_train = [r for r in v11_open_all if r.get("split") == RUN_SPLIT and row_video_id(r) in train_id_set]
    v11_trace_train = [r for r in v11_trace_all if r.get("split") == RUN_SPLIT and row_video_id(r) in train_id_set]
    v11_pred_by_video = parse_interval_rows(v11_pred_train, train_id_set, "prediction_id", "ad_start_sec", "ad_end_sec", RUN_SPLIT, "closed_prediction", warnings, ["start_reason", "end_reason"], video_duration) if v11_pred_all else defaultdict(list)
    v11_open_by_video = parse_interval_rows(v11_open_train, train_id_set, "open_candidate_id", "ad_start_sec", "last_anchor_sec", RUN_SPLIT, "open_candidate", warnings, ["open_state", "open_reason"], video_duration) if v11_open_all else defaultdict(list)
    v11_trace_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in v11_trace_train:
        v11_trace_by_video[row_video_id(r)].append(r)
    comparison_rows: list[dict[str, Any]] = []
    if not (v11_pred_all and v11_open_all and v11_trace_all):
        warnings.append("optional_v1_1_comparison_inputs_missing_or_partial")
    for vid in train_ids:
        dur = video_duration.get(vid, 0.0)
        v11_pred_cov = ratio(duration_sum(union_intervals(v11_pred_by_video.get(vid, []), dur)), dur)
        v12_pred_cov = ratio(duration_sum(pred_union_by_video.get(vid, [])), dur)
        v11_open_cov = ratio(duration_sum(union_intervals(v11_open_by_video.get(vid, []), dur)), dur)
        v12_open_cov = ratio(duration_sum(open_union_by_video.get(vid, [])), dur)
        summary_bits = []
        if v12_open_cov < v11_open_cov:
            summary_bits.append("open coverage lower in v1.2")
        elif v12_open_cov > v11_open_cov:
            summary_bits.append("open coverage higher in v1.2")
        if v12_pred_cov > v11_pred_cov:
            summary_bits.append("closed predicted coverage higher in v1.2")
        elif v12_pred_cov < v11_pred_cov:
            summary_bits.append("closed predicted coverage lower in v1.2")
        got = "mixed_or_needs_review"
        if v12_open_cov < v11_open_cov and v12_pred_cov <= v11_pred_cov * 1.2:
            got = "reduced_open_coverage_candidate"
        elif v12_open_cov > v11_open_cov or v12_pred_cov > v11_pred_cov * 1.5:
            got = "higher_coverage_needs_review"
        comparison_rows.append({
            "video_id": vid, "split": RUN_SPLIT, "v1_1_predicted_coverage_ratio": fmt(v11_pred_cov), "v1_2_predicted_coverage_ratio": fmt(v12_pred_cov), "v1_1_open_coverage_ratio": fmt(v11_open_cov), "v1_2_open_coverage_ratio": fmt(v12_open_cov),
            "v1_1_closed_prediction_count": len(v11_pred_by_video.get(vid, [])), "v1_2_closed_prediction_count": len(pred_by_video.get(vid, [])), "v1_1_open_interval_count": len(v11_open_by_video.get(vid, [])), "v1_2_open_interval_count": len(open_by_video.get(vid, [])),
            "v1_1_end_pending_timeout_count": sum(1 for r in v11_trace_by_video.get(vid, []) if r.get("decision_type") == "end_pending_timeout"), "v1_2_end_pending_timeout_count": sum(1 for r in trace_by_video.get(vid, []) if r.get("decision_type") == "end_pending_timeout"),
            "change_summary": "; ".join(summary_bits) if summary_bits else "no major coverage count change", "got_better_or_worse": got, "note": "train-only audit comparison; not final performance claim",
        })
    write_csv(root / OUTPUT_FILES["v1_1_vs_v1_2_train_comparison"], comparison_rows, COMPARISON_COLUMNS)

    append_log(log_path, "[STEP 11] Build rule issue candidates")
    rule_issue_rows = build_rule_issue_candidates(video_summary, trace_summary_rows, open_summary_rows)
    write_csv(root / OUTPUT_FILES["rule_issue_candidates"], rule_issue_rows, RULE_ISSUE_COLUMNS)
    write_rule_diagnosis(root / OUTPUT_FILES["rule_diagnosis_md"], video_summary, rule_issue_rows)

    append_log(log_path, "[STEP 12] Generate report and summary")
    input_files_modified = []
    for key, before in input_stats_before.items():
        if before != file_stat(root / INPUT_FILES[key]):
            input_files_modified.append(INPUT_FILES[key])
    old_after = root / "reports/analysis/old_project_snapshot_after_train_only_detector_error_analysis_v1_2.tsv"
    snapshot_tree(OLD_PROJECT_ROOT, old_after)
    old_project_modified = old_before.exists() and old_before.read_text(encoding="utf-8") != old_after.read_text(encoding="utf-8")
    validation_test_output_count = count_validation_test_outputs(root)
    if validation_test_output_count:
        errors.append(f"validation/test row found in generated analysis outputs: {validation_test_output_count}")
    if input_files_modified:
        errors.append("input files modified during analysis")
    if old_project_modified:
        errors.append("old project modified")
    metric_summary = summarize_metrics(video_summary)
    report = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(root),
        "created_at": now_iso(),
        "input_files": INPUT_FILES,
        "output_files": OUTPUT_FILES,
        "train_video_ids": EXPECTED_SPLITS["train"],
        "excluded_validation_video_ids": EXPECTED_SPLITS["validation"],
        "excluded_test_video_ids": EXPECTED_SPLITS["test"],
        "actual_schema": actual_schema,
        "split_check": split_check,
        "row_counts": {
            "actual_label_rows_all": len(actual_rows_all),
            "actual_train_intervals": sum(len(v) for v in actual_by_video.values()),
            "v1_2_predictions_train": len(pred_rows),
            "v1_2_open_train": len(open_rows),
            "v1_2_trace_train": len(trace_rows),
            "v1_2_audit_train": len(audit_rows),
            "v1_2_audio_only_review_train": len(audio_review_rows),
            "v1_2_long_ad_events_train": len(long_event_rows),
            "excluded_rows_seen": excluded_rows_seen,
            "generated_focus_window_rows": len(focus_rows),
        },
        "metric_summaries": metric_summary,
        "worst_case_lists": {"top_10": worst_rows[:10]},
        "rule_issue_candidates": rule_issue_rows,
        "sub_agent_results": [],
        "warnings": warnings,
        "errors": errors,
        "old_project_modified": old_project_modified,
        "input_files_modified": input_files_modified,
        "latest_for_chatgpt_forbidden_files_found": [],
        "validation_test_output_count": validation_test_output_count,
        "backup_dir": backup_manifest["backup_dir"],
        "backup_manifest": str(Path(backup_manifest["backup_dir"]) / "backup_manifest.json"),
        "input_file_stats_before_path": str(input_stats_path),
        "old_project_snapshot_before": str(old_before),
        "old_project_snapshot_after": str(old_after),
        "no_detector_run": True,
        "no_rule_change": True,
        "no_threshold_tuning": True,
        "no_feature_extraction": True,
        "train_only_error_analysis": True,
        "final_performance_claim": False,
    }
    summary_md = build_summary(report, video_summary, rule_issue_rows, trace_summary_rows)
    (root / OUTPUT_FILES["report_json"]).parent.mkdir(parents=True, exist_ok=True)
    (root / OUTPUT_FILES["report_json"]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / OUTPUT_FILES["summary_md"]).write_text(summary_md, encoding="utf-8")

    append_log(log_path, "[STEP 13] Sub Agent validations pending external execution")
    forbidden = refresh_latest_bundle(root, OUTPUT_FILES, log_path)
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden
    if forbidden:
        report["errors"].append("forbidden files found in latest bundle")
    (root / OUTPUT_FILES["report_json"]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / OUTPUT_FILES["summary_md"]).write_text(build_summary(report, video_summary, rule_issue_rows, trace_summary_rows), encoding="utf-8")
    refresh_latest_bundle(root, OUTPUT_FILES, log_path)
    append_log(log_path, final_summary_text(report))
    return report


def count_validation_test_outputs(root: Path) -> int:
    count = 0
    for key in ["video_summary", "interval_overlap", "open_summary", "trace_reason_summary", "state_transition_summary", "worst_cases", "trace_focus_windows", "v1_1_vs_v1_2_train_comparison"]:
        path = root / OUTPUT_FILES[key]
        if not path.exists():
            continue
        for row in read_csv(path):
            split = row.get("split", "")
            if split in EXCLUDED_SPLITS:
                count += 1
            if row.get("scope") == "video" and str(row.get("video_id", "")) in {"3", "4", "7", "16", "17", "18"}:
                count += 1
    return count


def summarize_metrics(video_summary: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(video_summary)
    total_actual = sum(as_float(r["actual_total_duration_sec"]) for r in video_summary)
    total_pred = sum(as_float(r["predicted_total_duration_sec"]) for r in video_summary)
    total_open = sum(as_float(r["open_total_duration_proxy_sec"]) for r in video_summary)
    severe = [r for r in video_summary if r["severity_level"] == "severe_over_detection"]
    high_open = [r for r in video_summary if as_float(r["open_coverage_ratio"]) >= 0.3]
    fp_top = sorted(video_summary, key=lambda r: as_float(r["false_positive_duration_sec"]), reverse=True)[:5]
    missed_top = sorted(video_summary, key=lambda r: as_float(r["missed_actual_duration_sec"]), reverse=True)[:5]
    return {
        "train_video_count": n,
        "actual_total_duration_sec": round(total_actual, 3),
        "closed_predicted_total_duration_sec": round(total_pred, 3),
        "open_interval_total_duration_proxy_sec": round(total_open, 3),
        "mean_actual_coverage": round(sum(as_float(r["actual_coverage_ratio"]) for r in video_summary) / n, 6) if n else 0.0,
        "mean_predicted_coverage": round(sum(as_float(r["predicted_coverage_ratio"]) for r in video_summary) / n, 6) if n else 0.0,
        "mean_open_coverage": round(sum(as_float(r["open_coverage_ratio"]) for r in video_summary) / n, 6) if n else 0.0,
        "severe_over_detection_video_count": len(severe),
        "open_coverage_high_video_count": len(high_open),
        "false_positive_duration_top_videos": [{"video_id": r["video_id"], "false_positive_duration_sec": r["false_positive_duration_sec"]} for r in fp_top],
        "missed_actual_top_videos": [{"video_id": r["video_id"], "missed_actual_duration_sec": r["missed_actual_duration_sec"], "missed_actual_ratio_of_actual": r["missed_actual_ratio_of_actual"]} for r in missed_top],
        "worst_over_detection_videos": [{"video_id": r["video_id"], "predicted_plus_open_coverage_ratio": r["predicted_plus_open_coverage_ratio"], "severity_level": r["severity_level"]} for r in sorted(video_summary, key=lambda x: as_float(x["predicted_plus_open_coverage_ratio"]), reverse=True)[:5]],
    }


def build_rule_issue_candidates(video_summary: list[dict[str, Any]], trace_summary_rows: list[dict[str, Any]], open_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    severe_videos = [r for r in video_summary if r["review_priority"] == "high"]
    open_high = [r for r in video_summary if as_float(r["open_coverage_ratio"]) >= 0.3]
    high_fp = [r for r in video_summary if as_float(r["false_positive_duration_sec"]) >= 300]
    high_missed = [r for r in video_summary if as_float(r["missed_actual_ratio_of_actual"]) >= 0.5]
    timeout_videos = [r for r in video_summary if as_float(r["end_pending_timeout_count"]) > 0 or as_float(r["end_pending_cancel_count"]) > 0]
    in_ad_sticky = [r for r in video_summary if as_float(r["in_ad_maintain_count"]) >= 30 and r["review_priority"] in {"high", "medium"}]
    audio_rejected = [r for r in video_summary if as_float(r["audio_only_start_rejected_count"]) >= 10]
    product_count = sum(as_float(r["count"]) for r in trace_summary_rows if r.get("category") == "conflict_flag" and r.get("key") == "product_repetition_continues" and r.get("scope") == "overall_train")

    def vids(rows: list[dict[str, Any]]) -> str:
        return ",".join(str(r["video_id"]) for r in rows[:8])

    issues = [
        {
            "issue_id": "RI001",
            "issue_name": "start gate still too permissive",
            "evidence_from_train": f"high priority videos={len(severe_videos)}, high false-positive videos={len(high_fp)}",
            "affected_video_count": len(set(r["video_id"] for r in severe_videos + high_fp)),
            "example_video_ids": vids(severe_videos or high_fp),
            "severity": "high" if severe_videos else "medium",
            "suspected_rule_area": "start_confirmation",
            "suggested_direction": "v1.3 후보로 start confirmation이 actual 이후 맥락과 얼마나 어긋나는지 trace window에서 먼저 확인한다.",
            "caution": "이 분석 안에서는 threshold/rule을 바꾸지 않는다.",
            "requires_manual_review": "true",
        },
        {
            "issue_id": "RI002",
            "issue_name": "in_ad continuity too sticky",
            "evidence_from_train": f"sticky in_ad candidate videos={len(in_ad_sticky)}; common maintain reasons are summarized in trace_reason_summary",
            "affected_video_count": len(in_ad_sticky),
            "example_video_ids": vids(in_ad_sticky),
            "severity": "high" if len(in_ad_sticky) >= 3 else "medium",
            "suspected_rule_area": "in_ad_continuity",
            "suggested_direction": "OCR/product/D2/audio continuity가 종료 근거를 얼마나 덮는지 focus window로 검토한다.",
            "caution": "continuity 완화는 실제 광고 내부 유지 실패를 만들 수 있어 train viewer 확인이 필요하다.",
            "requires_manual_review": "true",
        },
        {
            "issue_id": "RI003",
            "issue_name": "end_pending returns or cancels too easily",
            "evidence_from_train": f"videos with end_pending timeout/cancel={len(timeout_videos)}",
            "affected_video_count": len(timeout_videos),
            "example_video_ids": vids(timeout_videos),
            "severity": "medium",
            "suspected_rule_area": "end_pending_timeout",
            "suggested_direction": "timeout/cancel 직후 in_ad 복귀가 긴 false positive/open으로 이어지는지 확인한다.",
            "caution": "long-ad 종료 rule은 120초 이후 후보로만 검토한다.",
            "requires_manual_review": "true",
        },
        {
            "issue_id": "RI004",
            "issue_name": "open interval remains unresolved for too long",
            "evidence_from_train": f"open coverage >=0.3 videos={len(open_high)}, open candidates={len(open_summary_rows)}",
            "affected_video_count": len(open_high),
            "example_video_ids": vids(open_high),
            "severity": "high" if open_high else "low",
            "suspected_rule_area": "open_interval_policy",
            "suggested_direction": "open interval은 final prediction처럼 해석하지 말고 unresolved 후보로 분리 유지하면서 종료 실패 원인을 분류한다.",
            "caution": "video-end hard close는 test 전에는 도입하지 않는다.",
            "requires_manual_review": "true",
        },
        {
            "issue_id": "RI005",
            "issue_name": "audio context medium/high may be too broad for continuity",
            "evidence_from_train": f"audio-only rejected-heavy videos={len(audio_rejected)}",
            "affected_video_count": len(audio_rejected),
            "example_video_ids": vids(audio_rejected),
            "severity": "medium",
            "suspected_rule_area": "audio_context_level",
            "suggested_direction": "audio context가 start/continuity support로 작동한 구간과 actual overlap을 비교한다.",
            "caution": "audio만으로 rule을 자동 조정하지 않는다.",
            "requires_manual_review": "true",
        },
        {
            "issue_id": "RI006",
            "issue_name": "product repetition may over-maintain ad state",
            "evidence_from_train": f"overall product_repetition_continues trace count={int(product_count)}",
            "affected_video_count": sum(1 for r in video_summary if "product" in str(r.get("most_common_in_ad_maintain_reason", ""))),
            "example_video_ids": ",".join(r["video_id"] for r in video_summary if "product" in str(r.get("most_common_in_ad_maintain_reason", ""))[:80]),
            "severity": "medium" if product_count else "low",
            "suspected_rule_area": "product_repetition",
            "suggested_direction": "product repetition 유지가 실제 광고 종료 이후에도 계속되는지 trace window에서 확인한다.",
            "caution": "OCR reliability low는 non-ad evidence로 해석하지 않는다.",
            "requires_manual_review": "true",
        },
        {
            "issue_id": "RI007",
            "issue_name": "lack of negative/non-ad recovery state",
            "evidence_from_train": f"missed_actual videos={len(high_missed)}, severe/high-priority videos={len(severe_videos)}",
            "affected_video_count": len(set(r["video_id"] for r in high_missed + severe_videos)),
            "example_video_ids": vids(high_missed + severe_videos),
            "severity": "medium",
            "suspected_rule_area": "ocr_reliability_handling",
            "suggested_direction": "non-ad recovery 후보 상태를 만들 필요가 있는지, end_pending 복귀 흐름을 중심으로 검토한다.",
            "caution": "negative recovery는 실제 광고 중간 끊김을 만들 수 있으므로 train worst case 확인 후 논의한다.",
            "requires_manual_review": "true",
        },
    ]
    return issues


def write_rule_diagnosis(path: Path, video_summary: list[dict[str, Any]], issues: list[dict[str, Any]]) -> None:
    top_worst = sorted(video_summary, key=lambda r: as_float(r["predicted_plus_open_coverage_ratio"]), reverse=True)[:10]
    path.parent.mkdir(parents=True, exist_ok=True)
    issue_lines = [f"- `{i['issue_id']}` {i['issue_name']}: {i['evidence_from_train']} / 후보 영역 `{i['suspected_rule_area']}`" for i in issues]
    worst_lines = [f"- video {r['video_id']}: predicted+open coverage={r['predicted_plus_open_coverage_ratio']}, actual coverage={r['actual_coverage_ratio']}, severity={r['severity_level']}" for r in top_worst]
    path.write_text(f"""# Train-Only Detector v1.2 Rule Diagnosis

이 문서는 detector v1.3 rule을 만들기 전 원인 분석용 메모입니다. detector 구현을 수정하지 않았고, validation/test는 분석하지 않았습니다. 아래 내용은 최종 성능 평가가 아니라 train-only debugging 후보입니다.

## 가장 심각한 over-detection 후보

{chr(10).join(worst_lines)}

## v1.3에서 검토할 rule issue 후보

{chr(10).join(issue_lines)}

## 해석 주의

- closed prediction과 open interval candidate는 분리해서 보아야 합니다.
- open interval은 unresolved candidate이며 final prediction이 아닙니다.
- actual label은 train-only error analysis에만 사용했습니다.
- rule/threshold 변경은 이 작업에서 수행하지 않았습니다.
- validation은 train 원인 유형을 사람이 확인한 뒤 다시 봐야 합니다.
- test는 아직 실행하지 말아야 합니다.
""", encoding="utf-8")


def build_summary(report: dict[str, Any], video_summary: list[dict[str, Any]], issues: list[dict[str, Any]], trace_summary_rows: list[dict[str, Any]]) -> str:
    metrics = report["metric_summaries"]
    worst = sorted(video_summary, key=lambda r: as_float(r["predicted_plus_open_coverage_ratio"]), reverse=True)[:10]
    worst_lines = [f"- video {r['video_id']}: actual={r['actual_coverage_ratio']}, closed={r['predicted_coverage_ratio']}, open={r['open_coverage_ratio']}, combined={r['predicted_plus_open_coverage_ratio']}, severity={r['severity_level']}" for r in worst]
    reason_top = [r for r in trace_summary_rows if r.get("scope") == "overall_train" and r.get("category") == "decision_reason"][:12]
    reason_lines = [f"- {r['key']}: {r['count']}" for r in reason_top]
    issue_lines = [f"- {i['issue_id']} {i['issue_name']} ({i['suspected_rule_area']}): {i['severity']}" for i in issues]
    warnings = report.get("warnings", [])
    errors = report.get("errors", [])
    status = "FAILURE" if errors else "CONDITIONAL_SUCCESS" if warnings else "SUCCESS"
    sub_lines = [f"- {r.get('name')}: `{r.get('status')}`" for r in report.get("sub_agent_results", [])] or ["- Pending external Sub Agent validation"]
    return f"""# Train-Only Detector Error Analysis v1.2 Summary

Generated at: `{report.get('created_at')}`

## 작업 개요

- 작업 상태: {status}
- 목적: detector v1.2 실패 원인을 train split과 train actual label만으로 정량 분석
- detector 재실행: false
- rule/config/threshold 수정: false
- feature extraction: false
- final performance claim: false

## Train-Only 사용 이유

v1.3 rule 후보를 만들기 전, train에서 과검출과 open interval이 어디서 생기는지 먼저 분해하기 위해 train 정답 라벨만 사용했습니다. Validation은 원인 유형을 사람이 확인하기 전까지 보류해야 하며, test는 계속 보호합니다.

## 핵심 수치

- train video count: `{metrics['train_video_count']}`
- actual total duration sec: `{metrics['actual_total_duration_sec']}`
- closed predicted total duration sec: `{metrics['closed_predicted_total_duration_sec']}`
- open interval total duration proxy sec: `{metrics['open_interval_total_duration_proxy_sec']}`
- mean actual coverage: `{metrics['mean_actual_coverage']}`
- mean predicted coverage: `{metrics['mean_predicted_coverage']}`
- mean open coverage: `{metrics['mean_open_coverage']}`
- severe over-detection video count: `{metrics['severe_over_detection_video_count']}`
- open coverage high video count: `{metrics['open_coverage_high_video_count']}`

## Worst Cases Top 10

{chr(10).join(worst_lines)}

## False Positive Duration Top Videos

{json_dumps(metrics['false_positive_duration_top_videos'])}

## Missed Actual Top Videos

{json_dumps(metrics['missed_actual_top_videos'])}

## 주요 Reason 분포

{chr(10).join(reason_lines)}

## Suspected Rule Issue 후보

{chr(10).join(issue_lines)}

## Sub Agent Validation Results

{chr(10).join(sub_lines)}

## Safety

- old_project_modified: `{str(report.get('old_project_modified')).lower()}`
- input_files_modified: `{str(bool(report.get('input_files_modified'))).lower()}`
- validation/test output count: `{report.get('validation_test_output_count')}`
- no_detector_run: true
- no_rule_change: true
- no_threshold_tuning: true
- train_only_error_analysis: true

## 다음 단계

1. worst train videos를 viewer로 확인한다.
2. `train_only_detector_error_trace_focus_windows_v1_2.csv`를 기준으로 원인을 확인한다.
3. 이후 v1.3 rule candidate를 작성한다.
4. train에서 먼저 수정 확인 후 validation으로 이동한다.
5. test는 아직 실행하지 않는다.
"""


def final_summary_text(report: dict[str, Any]) -> str:
    metrics = report["metric_summaries"]
    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    status = "FAILURE" if errors else "CONDITIONAL_SUCCESS" if warnings else "SUCCESS"
    top_issues = ", ".join(i["issue_id"] for i in report.get("rule_issue_candidates", [])[:5])
    sub = ", ".join(f"{r.get('name')}={r.get('status')}" for r in report.get("sub_agent_results", [])) or "pending external validation"
    worst = ", ".join(str(v["video_id"]) for v in metrics.get("worst_over_detection_videos", [])[:5])
    return f"""[STEP 15] Final human-readable summary
작업 상태: {status}
train video count: {metrics['train_video_count']}
severe over-detection video count: {metrics['severe_over_detection_video_count']}
worst over-detection videos: {worst}
mean actual coverage: {metrics['mean_actual_coverage']}
mean predicted coverage: {metrics['mean_predicted_coverage']}
mean open coverage: {metrics['mean_open_coverage']}
top suspected rule issues: {top_issues}
output paths: report={PROJECT_ROOT / OUTPUT_FILES['report_json']}, summary={PROJECT_ROOT / OUTPUT_FILES['summary_md']}
Sub Agent validation 결과: {sub}
old_project_modified: {str(report.get('old_project_modified')).lower()}
input_files_modified: {str(bool(report.get('input_files_modified'))).lower()}
validation/test output count: {report.get('validation_test_output_count')}
latest bundle path: {PROJECT_ROOT / OUTPUT_FILES['latest_bundle']}
warnings/errors: warnings={warnings}, errors={errors}
다음 단계 제안: worst train videos를 viewer로 확인; trace_focus_windows CSV를 기준으로 원인 확인; 이후 v1.3 rule candidate 작성; train에서 먼저 수정 확인; validation은 train 결과가 말이 된 뒤 다시 확인; test는 아직 실행하지 말 것
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-only error analysis for detector v1.2 outputs.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    report = run_analysis(root, args.timestamp)
    return 1 if report.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
