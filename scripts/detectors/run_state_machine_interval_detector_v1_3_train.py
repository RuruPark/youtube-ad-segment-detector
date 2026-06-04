#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State-machine interval detector v1.3 train-only adjustment candidate.

This is not a final detector. It runs only on train split rows, preserves v1.1/v1.2
artifacts, and uses actual labels only after detector execution for train-only audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

VERSION = "v1.3"
BASE_VERSION = "v1.2"
DETECTOR_ID = "state_machine_interval_detector_v1_3_train"
RUN_SPLITS = {"train"}
EXCLUDED_SPLITS = {"validation", "test"}
STATE_NAMES = {"non_ad", "start_pending", "in_ad", "end_pending"}
EXPECTED_SPLITS = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}
EXPECTED_SEED = "20240524"
BANNED_DECISION_PATTERNS = [
    "nearest_true_boundary", "distance_to_nearest_true_boundary", "is_near_true_boundary",
    "label", "true", "audit", "ground_truth", "gt", "actual", "ad_start_sec_from_label", "ad_end_sec_from_label",
]
DECISION_FEATURE_COLUMNS = [
    "candidate_time_sec", "audio_start_signal_level", "audio_end_signal_level", "audio_context_level", "audio_before_context_level", "audio_after_context_level",
    "audio_score_delta_post_minus_pre", "audio_score_delta_pre_minus_post", "ocr_start_signal_level", "ocr_end_signal_level", "ocr_context_level", "ocr_context_reliability_level",
    "ocr_score_delta_post_minus_pre", "ocr_score_delta_pre_minus_post", "ocr_pre_10s_product_brand_count", "ocr_post_10s_product_brand_count",
    "ocr_pre_10s_purchase_cta_count", "ocr_post_10s_purchase_cta_count", "possible_end_context_support",
]
FORBIDDEN_BUNDLE_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".wav", ".mp3", ".m4a", ".jpg", ".jpeg", ".png", ".webp", ".parquet", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".onnx"}
FORBIDDEN_BUNDLE_PARTS = {"cache", "frames", "frame_images", "raw_video", "video_proxy", "model_cache", "tmp", "__pycache__"}

PREDICTION_COLUMNS = ["prediction_id", "version", "base_version", "split", "video_id", "ad_start_sec", "ad_end_sec", "ad_duration_sec", "start_anchor_id", "end_anchor_id", "start_reason", "end_reason", "interval_status", "merged_from_prediction_ids", "merge_reason", "used_test_row", "decision_feature_columns_json", "audit_columns_used_for_decision", "v1_3_adjustment_flags_json"]
TRACE_COLUMNS = ["version", "base_version", "split", "video_id", "visual_anchor_id", "transition_time_anchor", "candidate_time_mmss", "state_before", "state_after", "decision_type", "decision_reason", "pending_start_time", "pending_start_source", "pending_end_time", "pending_elapsed_sec_actual", "pending_anchor_count_actual", "pending_timeout_trigger", "audio_start_signal_level", "audio_end_signal_level", "audio_context_level", "audio_before_context_level", "audio_after_context_level", "audio_score_delta_post_minus_pre", "ocr_start_signal_level", "ocr_end_signal_level", "ocr_context_level", "ocr_context_reliability_level", "c3_flag", "consecutive_c3_count", "d2_flag", "product_repetition_continues", "purchase_cta_or_link_context", "audio_positive_delta", "strong_ad_evidence", "ocr_only_continuity_count", "ocr_only_continuity_elapsed_sec", "no_strong_ad_evidence_count", "no_strong_ad_evidence_elapsed_sec", "early_disclosure_guard_applied", "disclosure_notice_rejected", "ocr_only_continuity_cap_applied", "weak_continuity_recovery_applied", "end_pending_timeout_no_strong_continuity_applied", "unresolved_long_open_guard_applied", "low_gap_bridge_applied", "bridge_gap_sec", "minimum_duration_hold_applied", "e2_exception_applied", "retroactive_start_applied", "used_for_decision_columns_json", "audit_columns_used_for_decision"]
OPEN_COLUMNS = ["open_candidate_id", "version", "base_version", "split", "video_id", "ad_start_sec", "last_anchor_sec", "start_anchor_id", "last_anchor_id", "open_state", "open_reason", "pending_end_time", "interval_status", "used_test_row"]
UNRESOLVED_COLUMNS = ["version", "split", "video_id", "unresolved_candidate_id", "start_sec", "last_anchor_sec", "duration_proxy_sec", "video_duration_sec", "duration_ratio", "start_anchor_id", "last_anchor_id", "start_reason", "unresolved_reason", "recent_strong_evidence_count", "recent_window_sec", "moved_to_unresolved_and_reset_state", "review_note"]
DISCLOSURE_COLUMNS = ["version", "split", "video_id", "visual_anchor_id", "transition_time_anchor", "candidate_time_mmss", "ocr_start_signal_level", "ocr_context_level", "ocr_context_reliability_level", "audio_start_signal_level", "audio_score_delta_post_minus_pre", "product_repetition_continues", "purchase_cta_or_link_context", "early_disclosure_guard_applied", "rejection_reason", "review_note"]
OCR_CAP_COLUMNS = ["version", "split", "video_id", "visual_anchor_id", "transition_time_anchor", "ocr_only_continuity_count", "ocr_only_continuity_elapsed_sec", "ocr_context_level", "ocr_context_reliability_level", "audio_context_level", "strong_ad_evidence", "pending_end_time", "event_reason"]
WEAK_RECOVERY_COLUMNS = ["version", "split", "video_id", "visual_anchor_id", "transition_time_anchor", "no_strong_ad_evidence_count", "no_strong_ad_evidence_elapsed_sec", "audio_context_level", "ocr_context_level", "ocr_context_reliability_level", "strong_ad_evidence", "pending_end_time", "event_reason"]
TRAIN_AUDIT_COLUMNS = ["version", "split", "video_id", "prediction_count", "open_interval_candidate_count", "unresolved_long_open_count", "disclosure_notice_rejected_count", "ocr_only_continuity_cap_event_count", "weak_continuity_recovery_event_count", "average_closed_duration_sec", "pending_start_timeout_count", "pending_end_timeout_count", "end_pending_timeout_no_strong_continuity_count", "start_reason_counts_json", "end_reason_counts_json", "state_transition_counts_json", "audit_columns_used_for_decision", "final_performance_claim"]
VIDEO_SUMMARY_COLUMNS = ["video_id", "split", "video_duration_sec", "actual_interval_count", "actual_total_duration_sec", "actual_coverage_ratio", "closed_prediction_count", "predicted_total_duration_sec", "predicted_coverage_ratio", "open_interval_count", "open_total_duration_proxy_sec", "open_coverage_ratio", "unresolved_long_open_count", "unresolved_total_duration_proxy_sec", "unresolved_coverage_ratio", "predicted_plus_open_plus_unresolved_duration_sec", "predicted_plus_open_plus_unresolved_coverage_ratio", "actual_pred_overlap_duration_sec", "actual_pred_overlap_ratio_of_actual", "actual_pred_overlap_ratio_of_prediction", "false_positive_duration_sec", "false_positive_ratio_of_video", "missed_actual_duration_sec", "missed_actual_ratio_of_actual", "over_detection_ratio_vs_actual", "severity_level", "review_priority", "diagnosis_note"]
INTERVAL_OVERLAP_COLUMNS = ["video_id", "split", "interval_type", "interval_id", "start_sec", "end_sec", "duration_sec", "best_matching_actual_id", "best_actual_overlap_sec", "best_actual_overlap_ratio_of_interval", "false_positive_duration_sec", "matched_prediction_ids", "missed_actual_duration_sec", "issue_type", "reason", "review_note"]
OPEN_SUMMARY_COLUMNS = ["video_id", "split", "open_candidate_id", "start_sec", "end_proxy_sec", "duration_proxy_sec", "video_duration_sec", "open_coverage_ratio", "overlap_with_actual_sec", "overlap_ratio_of_open", "start_reason", "last_state", "possible_failure_mode", "review_priority", "diagnosis_note"]
TRACE_REASON_COLUMNS = ["scope", "video_id", "category", "key", "count", "ratio_within_category", "note"]
WORST_COLUMNS = ["rank", "video_id", "split", "severity_level", "review_priority", "actual_coverage_ratio", "predicted_coverage_ratio", "open_coverage_ratio", "unresolved_coverage_ratio", "predicted_plus_open_plus_unresolved_coverage_ratio", "false_positive_duration_sec", "missed_actual_duration_sec", "diagnosis_note"]
RULE_ISSUE_COLUMNS = ["issue_id", "issue_name", "evidence_from_train", "affected_video_count", "example_video_ids", "severity", "suspected_rule_area", "suggested_direction", "caution", "requires_manual_review"]
COMPARISON_COLUMNS = ["video_id", "split", "actual_coverage_ratio", "v1_2_predicted_coverage_ratio", "v1_3_predicted_coverage_ratio", "v1_2_open_coverage_ratio", "v1_3_open_coverage_ratio", "v1_3_unresolved_coverage_ratio", "v1_2_predicted_plus_open_coverage_ratio", "v1_3_predicted_plus_open_plus_unresolved_coverage_ratio", "v1_2_closed_prediction_count", "v1_3_closed_prediction_count", "v1_2_open_interval_count", "v1_3_open_interval_count", "v1_3_unresolved_long_open_count", "v1_2_severity_level", "v1_3_severity_level", "got_better_or_worse", "review_priority", "comparison_note"]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_log(path: Path, message: str) -> None:
    print(message, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{now_iso()} {message}\n")


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
    return round(value, digits) if isinstance(value, float) else value


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def read_csv(path: Path, optional: bool = False) -> list[dict[str, str]]:
    if optional and not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


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


def normalize_level(value: Any) -> str:
    s = str(value if value is not None else "").strip().lower()
    return s if s in {"high", "medium", "low"} else "low_or_unknown"


def is_medium_or_high(value: Any) -> bool:
    return normalize_level(value) in {"medium", "high"}


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def row_time(row: dict[str, Any]) -> float:
    return as_float(row.get("candidate_time_sec"), 0.0)


def row_anchor_id(row: dict[str, Any]) -> str:
    return str(row.get("visual_anchor_id") or row.get("scene_boundary_anchor_id") or "")


def row_video_id(row: dict[str, Any]) -> str:
    return str(row.get("video_id", "")).strip()


def forbidden_match(column: str) -> bool:
    lower = column.lower()
    for p in BANNED_DECISION_PATTERNS:
        if p == "gt":
            if lower == "gt" or lower.startswith("gt_") or lower.endswith("_gt") or "_gt_" in lower:
                return True
        elif p in lower:
            return True
    return False


def banned_decision_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if forbidden_match(c)]


def product_repetition(row: dict[str, Any], warnings: list[str]) -> bool:
    if "ocr_pre_10s_product_brand_count" in row and "ocr_post_10s_product_brand_count" in row:
        return as_float(row.get("ocr_pre_10s_product_brand_count")) >= 1 and as_float(row.get("ocr_post_10s_product_brand_count")) >= 1
    for a, b in [("ocr_pre_10s_product_brand_score", "ocr_post_10s_product_brand_score"), ("ocr_pre_10s_product_brand_score_raw", "ocr_post_10s_product_brand_score_raw")]:
        if a in row and b in row:
            return as_float(row.get(a)) > 0 and as_float(row.get(b)) > 0
    if "product_repetition_columns_missing" not in warnings:
        warnings.append("product_repetition_columns_missing")
    return False


def cta_or_link_context(row: dict[str, Any], warnings: list[str]) -> bool:
    candidates = [
        "ocr_pre_10s_purchase_cta_count", "ocr_post_10s_purchase_cta_count",
        "ocr_pre_10s_purchase_cta_score", "ocr_post_10s_purchase_cta_score",
        "ocr_pre_10s_purchase_cta_score_raw", "ocr_post_10s_purchase_cta_score_raw",
        "ocr_pre_10s_link_count", "ocr_post_10s_link_count", "ocr_pre_10s_more_info_count", "ocr_post_10s_more_info_count",
    ]
    found = False
    for col in candidates:
        if col in row:
            found = True
            if as_float(row.get(col)) > 0:
                return True
    if not found and "purchase_cta_or_link_columns_missing" not in warnings:
        warnings.append("purchase_cta_or_link_columns_missing")
    return False


def row_flags(row: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    audio_context = normalize_level(row.get("audio_context_level"))
    audio_before = normalize_level(row.get("audio_before_context_level"))
    audio_after = normalize_level(row.get("audio_after_context_level"))
    ocr_context = normalize_level(row.get("ocr_context_level"))
    reliability = normalize_level(row.get("ocr_context_reliability_level"))
    ocr_start = normalize_level(row.get("ocr_start_signal_level"))
    audio_start = normalize_level(row.get("audio_start_signal_level"))
    c3 = audio_context == "high" and reliability == "high" and ocr_context == "low"
    d2 = audio_context == "low" and ocr_context == "high"
    known_low = reliability == "high" and ocr_context == "low" and audio_context == "low"
    product = product_repetition(row, warnings)
    cta = cta_or_link_context(row, warnings)
    delta_col_present = "audio_score_delta_post_minus_pre" in row
    audio_delta = as_float(row.get("audio_score_delta_post_minus_pre"), 0.0)
    audio_positive_delta = (audio_delta > 0 and is_medium_or_high(audio_start)) if delta_col_present else False
    audio_ocr_both_mh = is_medium_or_high(audio_context) and is_medium_or_high(ocr_context) and is_medium_or_high(reliability)
    start_combo = ocr_start == "high" and not (row_time(row) <= 30 and not product and not cta and not audio_positive_delta) and (product or cta or audio_positive_delta)
    strong = bool(product or cta or audio_ocr_both_mh or start_combo)
    return {
        "audio_start": audio_start, "audio_end": normalize_level(row.get("audio_end_signal_level")), "audio_context": audio_context, "audio_before": audio_before, "audio_after": audio_after,
        "ocr_start": ocr_start, "ocr_end": normalize_level(row.get("ocr_end_signal_level")), "ocr_context": ocr_context, "reliability": reliability,
        "c3": c3, "d2": d2, "known_low": known_low, "product": product, "cta": cta, "audio_positive_delta": audio_positive_delta,
        "audio_delta": audio_delta, "audio_delta_present": delta_col_present, "audio_ocr_both_mh": audio_ocr_both_mh, "strong": strong,
    }


def start_confirm_reason(flags: dict[str, Any], source: str) -> str | None:
    if (flags["product"] or flags["cta"]) and (is_medium_or_high(flags["ocr_start"]) or is_medium_or_high(flags["ocr_context"])):
        return "start_confirmed_by_disclosure_plus_product_or_cta"
    if flags["audio_positive_delta"] and (is_medium_or_high(flags["ocr_context"]) or is_medium_or_high(flags["ocr_start"])) and is_medium_or_high(flags["reliability"]):
        return "start_confirmed_by_audio_ocr_combined_rise"
    if is_medium_or_high(flags["ocr_context"]) and is_medium_or_high(flags["reliability"]) and (is_medium_or_high(flags["audio_context"]) or flags["product"] or flags["cta"]):
        return "start_confirmed_by_ocr_context_after_pending"
    if source == "ocr_unknown_audio_support" and is_medium_or_high(flags["reliability"]) and (is_medium_or_high(flags["ocr_start"]) or is_medium_or_high(flags["ocr_context"])):
        return "audio_only_start_guard_confirmed_by_reliable_ocr"
    return None


def end_support(row: dict[str, Any], flags: dict[str, Any]) -> tuple[bool, str]:
    reasons = []
    if is_medium_or_high(flags["audio_before"]) and flags["audio_after"] == "low":
        reasons.append("audio_before_medium_high_after_low")
    drop = as_float(row.get("ocr_score_delta_pre_minus_post")) > 0 or as_float(row.get("audio_score_delta_pre_minus_post")) > 0 or flags["audio_after"] == "low"
    if is_medium_or_high(flags["ocr_end"]) and drop:
        reasons.append("ocr_end_medium_high_with_post_drop")
    if flags["known_low"]:
        reasons.append("known_low_flow")
    if reasons:
        return True, "+".join(reasons)
    return False, ""


def close_interval(raw: list[dict[str, Any]], split: str, vid: str, start: dict[str, Any] | None, end_time: float, end_anchor: str, end_reason: str, flags: dict[str, Any] | None = None) -> None:
    if start is None or end_time <= float(start["time"]):
        return
    raw_id = f"raw_{len(raw)+1:06d}"
    raw.append({"raw_prediction_id": raw_id, "split": split, "video_id": vid, "ad_start_sec": round(float(start["time"]), 3), "ad_end_sec": round(end_time, 3), "ad_duration_sec": round(end_time - float(start["time"]), 3), "start_anchor_id": start.get("anchor_id", ""), "end_anchor_id": end_anchor, "start_reason": start.get("reason", ""), "end_reason": end_reason, "v1_3_adjustment_flags_json": json_dumps(flags or {})})


def merge_intervals(raw: list[dict[str, Any]], merge_gap_sec: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in sorted(raw, key=lambda r: (r["split"], int(float(r["video_id"])), float(r["ad_start_sec"]), float(r["ad_end_sec"]))):
        if not merged or merged[-1]["split"] != item["split"] or merged[-1]["video_id"] != item["video_id"]:
            m = dict(item); m["merged_from_prediction_ids"] = item["raw_prediction_id"]; m["merge_reason"] = "none"; merged.append(m); continue
        last = merged[-1]
        gap = float(item["ad_start_sec"]) - float(last["ad_end_sec"])
        if float(item["ad_start_sec"]) <= float(last["ad_end_sec"]) or gap <= merge_gap_sec:
            last["ad_end_sec"] = round(max(float(last["ad_end_sec"]), float(item["ad_end_sec"])), 3)
            last["ad_duration_sec"] = round(float(last["ad_end_sec"]) - float(last["ad_start_sec"]), 3)
            last["end_anchor_id"] = item["end_anchor_id"]; last["end_reason"] = item["end_reason"]
            last["merged_from_prediction_ids"] += "," + item["raw_prediction_id"]; last["merge_reason"] = "overlap_or_gap_le_10s"
        else:
            m = dict(item); m["merged_from_prediction_ids"] = item["raw_prediction_id"]; m["merge_reason"] = "none"; merged.append(m)
    out = []
    for i, item in enumerate(merged, start=1):
        row = dict(item)
        row.update({"prediction_id": f"smi_v1_3_{i:06d}", "version": VERSION, "base_version": BASE_VERSION, "interval_status": "closed", "used_test_row": "false", "decision_feature_columns_json": json_dumps(DECISION_FEATURE_COLUMNS), "audit_columns_used_for_decision": "false"})
        out.append(row)
    return out


def validate_split(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    rows = read_csv(root / config["input_paths"]["split_file"])
    obs: dict[str, list[int]] = defaultdict(list); seeds = set(); durations = {}
    for r in rows:
        s = r.get("split", "")
        if s in EXPECTED_SPLITS:
            obs[s].append(int(float(r.get("video_id", -1))))
        seeds.add(str(r.get("split_seed", "")))
        durations[str(r.get("video_id", ""))] = as_float(r.get("video_duration_sec"), 0.0)
    observed = {k: sorted(v) for k, v in obs.items()}
    ok = observed == EXPECTED_SPLITS and seeds == {EXPECTED_SEED}
    return {"ok": ok, "observed": observed, "seed_values": sorted(seeds), "video_durations": durations}


def pending_metrics(pending: dict[str, Any] | None, idx: int, t: float) -> tuple[Any, Any]:
    if not pending:
        return "", ""
    return idx - int(pending["index"]), round(t - float(pending["time"]), 3)


def timeout_trigger(count: int, elapsed: float, max_count: int, max_elapsed: float, inclusive: bool = False) -> str:
    count_hit = count >= max_count if inclusive else count > max_count
    duration_hit = elapsed > max_elapsed
    if count_hit and duration_hit: return "duration_and_anchor_count_exceeded"
    if count_hit: return "anchor_count_exceeded"
    if duration_hit: return "duration_exceeded"
    return "none"


def recent_strong_count(history: deque[tuple[float, bool]], t: float, anchor_window: int, sec_window: float) -> int:
    recent_by_anchor = list(history)[-anchor_window:]
    recent_by_time = [x for x in history if t - x[0] <= sec_window]
    merged = {(round(x[0], 3), x[1]) for x in recent_by_anchor + recent_by_time}
    return sum(1 for _, strong in merged if strong)


def add_unresolved(rows: list[dict[str, Any]], split: str, vid: str, current_start: dict[str, Any], last_row: dict[str, Any], video_duration: float, reason: str, recent_count: int, recent_window_sec: float) -> None:
    start = float(current_start["time"]); last = row_time(last_row)
    rows.append({"version": VERSION, "split": split, "video_id": vid, "unresolved_candidate_id": f"unresolved_{len(rows)+1:06d}", "start_sec": round(start, 3), "last_anchor_sec": round(last, 3), "duration_proxy_sec": round(max(0.0, last - start), 3), "video_duration_sec": round(video_duration, 3), "duration_ratio": round((max(0.0, last - start) / video_duration) if video_duration > 0 else 0.0, 6), "start_anchor_id": current_start.get("anchor_id", ""), "last_anchor_id": row_anchor_id(last_row), "start_reason": current_start.get("reason", ""), "unresolved_reason": reason, "recent_strong_evidence_count": recent_count, "recent_window_sec": recent_window_sec, "moved_to_unresolved_and_reset_state": "true", "review_note": "Unresolved failure candidate; not a closed prediction and not a normal open interval."})


def run_detector(root: Path, config: dict[str, Any], log_path: Path, dry_run: bool = False) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    bad = banned_decision_columns(DECISION_FEATURE_COLUMNS)
    if bad:
        raise RuntimeError(f"Forbidden decision columns: {bad}")
    split_check = validate_split(root, config)
    if not split_check["ok"]:
        raise RuntimeError(f"Fixed split mismatch: {split_check}")
    video_durations = split_check["video_durations"]
    rows_all = read_csv(root / config["input_paths"]["primary_detector_input"])
    rows = [r for r in rows_all if r.get("split") == "train"]
    excluded = sum(1 for r in rows_all if r.get("split") in EXCLUDED_SPLITS)
    if any(r.get("split") != "train" for r in rows):
        raise RuntimeError("Non-train row survived filter")
    rows.sort(key=lambda r: (int(float(r.get("video_id", 0) or 0)), row_time(r), row_anchor_id(r)))
    if dry_run:
        return {"dry_run": True, "detector_input_row_count": len(rows), "excluded_validation_test_rows": excluded}
    pending_rules = config["pending_rules"]
    start_max_count = int(pending_rules["start_pending_max_anchor_count"])
    start_max_elapsed = float(pending_rules["start_pending_max_duration_sec"])
    end_max_count = int(pending_rules["end_pending_max_anchor_count"])
    end_max_elapsed = float(pending_rules["end_pending_max_duration_sec"])
    min_duration = float(config["minimum_duration_prior"]["minimum_ad_duration_sec"])
    merge_gap = float(config["interval_merge"]["merge_gap_sec"])
    early_window = float(config["global_disclosure_guard"]["early_window_sec"])
    ocr_cap_count = int(config["ocr_only_continuity_cap"]["ocr_only_maintain_max_anchor_count"])
    ocr_cap_elapsed = float(config["ocr_only_continuity_cap"]["ocr_only_maintain_max_duration_sec"])
    weak_cap_count = int(config["weak_continuity_recovery"]["no_strong_ad_evidence_max_anchor_count"])
    weak_cap_elapsed = float(config["weak_continuity_recovery"]["no_strong_ad_evidence_max_duration_sec"])
    long_open_max = float(config["unresolved_long_open_guard"]["max_open_duration_sec"])
    long_open_ratio = float(config["unresolved_long_open_guard"]["max_open_video_ratio"])
    recent_anchor_window = int(config["unresolved_long_open_guard"]["recent_strong_evidence_anchor_window"])
    recent_sec_window = float(config["unresolved_long_open_guard"]["recent_strong_evidence_duration_sec"])

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r.get("split", ""), row_video_id(r))].append(r)

    raw_predictions: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    open_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    disclosure_rows: list[dict[str, Any]] = []
    ocr_cap_rows: list[dict[str, Any]] = []
    weak_rows: list[dict[str, Any]] = []

    for (split, vid), group in grouped.items():
        state = "non_ad"; current_start = None; pending_start = None; pending_end = None
        consecutive_c3 = 0
        ocr_only_count = 0; ocr_only_start_time = None; ocr_only_start_anchor = None
        weak_count = 0; weak_start_time = None; weak_start_anchor = None
        strong_history: deque[tuple[float, bool]] = deque(maxlen=50)
        for idx, row in enumerate(group):
            t = row_time(row); anchor = row_anchor_id(row); flags = row_flags(row, warnings)
            state_before = state; decision_type = "hold"; decision_reason = "no_state_change"; pending_trigger = "none"
            early_guard = False; disclosure_rejected = False; ocr_cap_applied = False; weak_applied = False; timeout_no_strong_applied = False; unresolved_applied = False
            min_hold = False; e2 = False; retro = False; bridge = False; bridge_gap: float | str = ""
            if flags["c3"]: consecutive_c3 += 1
            else: consecutive_c3 = 0
            strong_history.append((t, bool(flags["strong"])))
            pm_count, pm_elapsed = "", ""
            video_duration = video_durations.get(vid, 0.0)

            def reset_continuity() -> None:
                nonlocal ocr_only_count, ocr_only_start_time, ocr_only_start_anchor, weak_count, weak_start_time, weak_start_anchor
                ocr_only_count = 0; ocr_only_start_time = None; ocr_only_start_anchor = None
                weak_count = 0; weak_start_time = None; weak_start_anchor = None

            if state in {"in_ad", "end_pending"} and current_start is not None:
                current_duration = t - float(current_start["time"])
                recent_count = recent_strong_count(strong_history, t, recent_anchor_window, recent_sec_window)
                if (current_duration >= long_open_max or (video_duration > 0 and current_duration / video_duration >= long_open_ratio)) and recent_count == 0:
                    add_unresolved(unresolved_rows, split, vid, current_start, row, video_duration, config["unresolved_long_open_guard"]["reason"], recent_count, recent_sec_window)
                    state = "non_ad"; current_start = None; pending_start = None; pending_end = None; reset_continuity()
                    decision_type = "unresolved_long_open_guard"
                    decision_reason = config["unresolved_long_open_guard"]["reason"]
                    unresolved_applied = True

            if not unresolved_applied:
                if state == "non_ad":
                    if flags["c3"]:
                        decision_reason = "c3_no_start_candidate"
                    elif flags["ocr_start"] == "high":
                        if t <= early_window and not flags["product"] and not flags["cta"] and not flags["audio_positive_delta"]:
                            early_guard = True; disclosure_rejected = True
                            decision_type = "disclosure_notice_review"
                            decision_reason = "early_disclosure_notice_not_ad_start"
                            disclosure_rows.append({"version": VERSION, "split": split, "video_id": vid, "visual_anchor_id": anchor, "transition_time_anchor": round(t, 3), "candidate_time_mmss": row.get("candidate_time_mmss", ""), "ocr_start_signal_level": flags["ocr_start"], "ocr_context_level": flags["ocr_context"], "ocr_context_reliability_level": flags["reliability"], "audio_start_signal_level": flags["audio_start"], "audio_score_delta_post_minus_pre": flags["audio_delta"], "product_repetition_continues": str(flags["product"]).lower(), "purchase_cta_or_link_context": str(flags["cta"]).lower(), "early_disclosure_guard_applied": "true", "rejection_reason": "early_disclosure_notice_rejected", "review_note": "Early OCR start high rejected as disclosure-only review candidate unless product/CTA/audio-rise support exists."})
                        else:
                            state = "start_pending"; pending_start = {"time": t, "anchor_id": anchor, "index": idx, "source": "ocr_start_high", "reason": "ocr_start_high_pending"}
                            decision_type = "start_pending_entered"; decision_reason = "ocr_start_high_pending"
                            pm_count, pm_elapsed = pending_metrics(pending_start, idx, t)
                    elif flags["ocr_start"] == "medium":
                        state = "start_pending"; pending_start = {"time": t, "anchor_id": anchor, "index": idx, "source": "ocr_start_medium", "reason": "ocr_start_medium_pending"}
                        decision_type = "start_pending_entered"; decision_reason = "ocr_start_medium_pending"; pm_count, pm_elapsed = pending_metrics(pending_start, idx, t)
                    elif flags["reliability"] in {"low", "low_or_unknown"} and is_medium_or_high(flags["audio_start"]):
                        state = "start_pending"; pending_start = {"time": t, "anchor_id": anchor, "index": idx, "source": "ocr_unknown_audio_support", "reason": "ocr_unknown_audio_start_support_pending"}
                        decision_type = "start_pending_entered"; decision_reason = "ocr_unknown_audio_start_support_pending"; pm_count, pm_elapsed = pending_metrics(pending_start, idx, t)

                elif state == "start_pending":
                    assert pending_start is not None
                    pm_count, pm_elapsed = pending_metrics(pending_start, idx, t)
                    count = int(pm_count); elapsed = float(pm_elapsed)
                    if count > start_max_count or elapsed > start_max_elapsed:
                        state = "non_ad"; decision_type = "start_pending_timeout"; decision_reason = "start_pending_cancelled_no_strong_ad_evidence"; pending_trigger = timeout_trigger(count, elapsed, start_max_count, start_max_elapsed); pending_start = None
                    else:
                        source = pending_start.get("source", "")
                        reason = start_confirm_reason(flags, source)
                        if reason:
                            state = "in_ad"; current_start = {"time": pending_start["time"], "anchor_id": pending_start["anchor_id"], "reason": reason}
                            decision_type = "start_confirmed"; decision_reason = reason; retro = True; pending_trigger = "confirmed"; pending_start = None; reset_continuity()
                        else:
                            state = "non_ad"; decision_type = "start_pending_cancelled"
                            if source == "ocr_unknown_audio_support": decision_reason = "audio_only_start_rejected_due_to_ocr_unconfirmed"
                            elif source == "ocr_start_high" and t <= early_window: decision_reason = "early_disclosure_notice_rejected"
                            else: decision_reason = "start_pending_cancelled_no_strong_ad_evidence"
                            pending_start = None

                elif state == "in_ad":
                    assert current_start is not None
                    duration = t - float(current_start["time"])
                    e_support, e_reason = end_support(row, flags)
                    if flags["strong"]:
                        reset_continuity(); decision_reason = "in_ad_maintained_by_strong_ad_evidence"
                    else:
                        weak_count += 1
                        if weak_start_time is None: weak_start_time = t; weak_start_anchor = anchor
                        if is_medium_or_high(flags["ocr_context"]):
                            ocr_only_count += 1
                            if ocr_only_start_time is None: ocr_only_start_time = t; ocr_only_start_anchor = anchor
                        weak_elapsed = t - float(weak_start_time)
                        ocr_elapsed = 0.0 if ocr_only_start_time is None else t - float(ocr_only_start_time)
                        if flags["known_low"] or e_support or consecutive_c3 >= 2:
                            if duration < min_duration:
                                min_hold = True; decision_reason = "minimum_duration_hold_before_20s"
                            else:
                                pend_time = t
                                state = "end_pending"; pending_end = {"time": pend_time, "anchor_id": anchor, "index": idx, "reason": e_reason or "repeated_c3_end_pending"}
                                decision_type = "end_pending_entered"; decision_reason = pending_end["reason"]; pm_count, pm_elapsed = pending_metrics(pending_end, idx, t)
                        elif ocr_only_count >= ocr_cap_count or ocr_elapsed >= ocr_cap_elapsed:
                            pend_time = float(ocr_only_start_time if ocr_only_start_time is not None else t)
                            pend_anchor = str(ocr_only_start_anchor or anchor)
                            state = "end_pending"; pending_end = {"time": pend_time, "anchor_id": pend_anchor, "index": idx, "reason": "ocr_only_continuity_cap_reached"}
                            decision_type = "end_pending_entered"; decision_reason = "ocr_only_continuity_cap_reached"; ocr_cap_applied = True; pm_count, pm_elapsed = pending_metrics(pending_end, idx, t)
                            ocr_cap_rows.append({"version": VERSION, "split": split, "video_id": vid, "visual_anchor_id": anchor, "transition_time_anchor": round(t, 3), "ocr_only_continuity_count": ocr_only_count, "ocr_only_continuity_elapsed_sec": round(ocr_elapsed, 3), "ocr_context_level": flags["ocr_context"], "ocr_context_reliability_level": flags["reliability"], "audio_context_level": flags["audio_context"], "strong_ad_evidence": "false", "pending_end_time": round(pend_time, 3), "event_reason": "ocr_only_continuity_cap_reached"})
                        elif weak_count >= weak_cap_count or weak_elapsed >= weak_cap_elapsed:
                            pend_time = float(weak_start_time if weak_start_time is not None else t)
                            pend_anchor = str(weak_start_anchor or anchor)
                            state = "end_pending"; pending_end = {"time": pend_time, "anchor_id": pend_anchor, "index": idx, "reason": "weak_continuity_recovery_to_end_pending"}
                            decision_type = "end_pending_entered"; decision_reason = "weak_continuity_recovery_to_end_pending"; weak_applied = True; pm_count, pm_elapsed = pending_metrics(pending_end, idx, t)
                            weak_rows.append({"version": VERSION, "split": split, "video_id": vid, "visual_anchor_id": anchor, "transition_time_anchor": round(t, 3), "no_strong_ad_evidence_count": weak_count, "no_strong_ad_evidence_elapsed_sec": round(weak_elapsed, 3), "audio_context_level": flags["audio_context"], "ocr_context_level": flags["ocr_context"], "ocr_context_reliability_level": flags["reliability"], "strong_ad_evidence": "false", "pending_end_time": round(pend_time, 3), "event_reason": "weak_continuity_recovery_to_end_pending"})
                        else:
                            decision_reason = "in_ad_held_without_strong_evidence_pending_recovery"

                elif state == "end_pending":
                    assert current_start is not None and pending_end is not None
                    pm_count, pm_elapsed = pending_metrics(pending_end, idx, t)
                    count = int(pm_count); elapsed = float(pm_elapsed)
                    if flags["strong"]:
                        state = "in_ad"; decision_type = "end_pending_cancelled"; decision_reason = "end_pending_cancelled_by_strong_ad_evidence"; pending_trigger = "continuity_cancel"; pending_end = None; reset_continuity()
                    elif flags["known_low"]:
                        close_interval(raw_predictions, split, vid, current_start, float(pending_end["time"]), str(pending_end["anchor_id"]), "end_pending_confirmed_by_known_low_flow", {"v1_3_end_policy": "known_low"})
                        state = "non_ad"; current_start = None; pending_end = None; reset_continuity(); decision_type = "end_confirmed"; decision_reason = "end_pending_confirmed_by_known_low_flow"; pending_trigger = "confirmed"
                    elif count >= end_max_count or elapsed > end_max_elapsed:
                        close_interval(raw_predictions, split, vid, current_start, float(pending_end["time"]), str(pending_end["anchor_id"]), config["end_pending_timeout_policy"]["reason"], {"end_pending_timeout_no_strong_continuity_applied": True})
                        state = "non_ad"; current_start = None; pending_end = None; reset_continuity(); decision_type = "end_confirmed"; decision_reason = config["end_pending_timeout_policy"]["reason"]; pending_trigger = timeout_trigger(count, elapsed, end_max_count, end_max_elapsed, inclusive=True); timeout_no_strong_applied = True
                    else:
                        decision_reason = "end_pending_waiting_no_strong_continuity"

            if state not in STATE_NAMES:
                raise RuntimeError(f"Invalid state: {state}")
            trace_rows.append({"version": VERSION, "base_version": BASE_VERSION, "split": split, "video_id": vid, "visual_anchor_id": anchor, "transition_time_anchor": round(t, 3), "candidate_time_mmss": row.get("candidate_time_mmss", ""), "state_before": state_before, "state_after": state, "decision_type": decision_type, "decision_reason": decision_reason, "pending_start_time": "" if pending_start is None else round(float(pending_start["time"]), 3), "pending_start_source": "" if pending_start is None else pending_start.get("source", ""), "pending_end_time": "" if pending_end is None else round(float(pending_end["time"]), 3), "pending_elapsed_sec_actual": pm_elapsed, "pending_anchor_count_actual": pm_count, "pending_timeout_trigger": pending_trigger, "audio_start_signal_level": flags["audio_start"], "audio_end_signal_level": flags["audio_end"], "audio_context_level": flags["audio_context"], "audio_before_context_level": flags["audio_before"], "audio_after_context_level": flags["audio_after"], "audio_score_delta_post_minus_pre": flags["audio_delta"], "ocr_start_signal_level": flags["ocr_start"], "ocr_end_signal_level": flags["ocr_end"], "ocr_context_level": flags["ocr_context"], "ocr_context_reliability_level": flags["reliability"], "c3_flag": str(flags["c3"]).lower(), "consecutive_c3_count": consecutive_c3, "d2_flag": str(flags["d2"]).lower(), "product_repetition_continues": str(flags["product"]).lower(), "purchase_cta_or_link_context": str(flags["cta"]).lower(), "audio_positive_delta": str(flags["audio_positive_delta"]).lower(), "strong_ad_evidence": str(flags["strong"]).lower(), "ocr_only_continuity_count": ocr_only_count, "ocr_only_continuity_elapsed_sec": "" if ocr_only_start_time is None else round(t - float(ocr_only_start_time), 3), "no_strong_ad_evidence_count": weak_count, "no_strong_ad_evidence_elapsed_sec": "" if weak_start_time is None else round(t - float(weak_start_time), 3), "early_disclosure_guard_applied": str(early_guard).lower(), "disclosure_notice_rejected": str(disclosure_rejected).lower(), "ocr_only_continuity_cap_applied": str(ocr_cap_applied).lower(), "weak_continuity_recovery_applied": str(weak_applied).lower(), "end_pending_timeout_no_strong_continuity_applied": str(timeout_no_strong_applied).lower(), "unresolved_long_open_guard_applied": str(unresolved_applied).lower(), "low_gap_bridge_applied": str(bridge).lower(), "bridge_gap_sec": bridge_gap, "minimum_duration_hold_applied": str(min_hold).lower(), "e2_exception_applied": str(e2).lower(), "retroactive_start_applied": str(retro).lower(), "used_for_decision_columns_json": json_dumps(DECISION_FEATURE_COLUMNS), "audit_columns_used_for_decision": "false"})

        last = group[-1] if group else None
        if last is not None and current_start is not None:
            dur = row_time(last) - float(current_start["time"])
            recent_count = recent_strong_count(strong_history, row_time(last), recent_anchor_window, recent_sec_window)
            if (dur >= long_open_max or (video_durations.get(vid, 0.0) > 0 and dur / video_durations.get(vid, 0.0) >= long_open_ratio)) and recent_count == 0:
                add_unresolved(unresolved_rows, split, vid, current_start, last, video_durations.get(vid, 0.0), config["unresolved_long_open_guard"]["reason"], recent_count, recent_sec_window)
            else:
                open_rows.append({"open_candidate_id": f"open_{len(open_rows)+1:06d}", "version": VERSION, "base_version": BASE_VERSION, "split": split, "video_id": vid, "ad_start_sec": round(float(current_start["time"]), 3), "last_anchor_sec": round(row_time(last), 3), "start_anchor_id": current_start.get("anchor_id", ""), "last_anchor_id": row_anchor_id(last), "open_state": state, "open_reason": "close_open_interval_at_video_end_false", "pending_end_time": "" if pending_end is None else round(float(pending_end["time"]), 3), "interval_status": "open_candidate", "used_test_row": "false"})

    predictions = merge_intervals(raw_predictions, merge_gap)
    all_output_rows = predictions + trace_rows + open_rows + unresolved_rows + disclosure_rows + ocr_cap_rows + weak_rows
    if any(r.get("split") != "train" for r in all_output_rows):
        raise RuntimeError("Non-train output detected")
    if any(float(p["ad_start_sec"]) >= float(p["ad_end_sec"]) for p in predictions):
        raise RuntimeError("Closed prediction start >= end detected")
    outp = config["output_paths"]
    append_log(log_path, "[STEP 05] Writing v1.3 train detector outputs")
    write_csv(root / outp["prediction_csv"], predictions, PREDICTION_COLUMNS)
    write_csv(root / outp["trace_csv"], trace_rows, TRACE_COLUMNS)
    write_csv(root / outp["open_interval_csv"], open_rows, OPEN_COLUMNS)
    write_csv(root / outp["unresolved_long_open_csv"], unresolved_rows, UNRESOLVED_COLUMNS)
    write_csv(root / outp["disclosure_notice_review_csv"], disclosure_rows, DISCLOSURE_COLUMNS)
    write_csv(root / outp["ocr_only_continuity_cap_csv"], ocr_cap_rows, OCR_CAP_COLUMNS)
    write_csv(root / outp["weak_continuity_recovery_csv"], weak_rows, WEAK_RECOVERY_COLUMNS)
    audit_rows = build_train_audit(predictions, open_rows, unresolved_rows, disclosure_rows, ocr_cap_rows, weak_rows, trace_rows)
    write_csv(root / outp["audit_csv"], audit_rows, TRAIN_AUDIT_COLUMNS)
    return {"warnings": warnings, "errors": errors, "split_check": split_check, "excluded_validation_test_rows": excluded, "predictions": predictions, "trace_rows": trace_rows, "open_rows": open_rows, "unresolved_rows": unresolved_rows, "disclosure_rows": disclosure_rows, "ocr_cap_rows": ocr_cap_rows, "weak_rows": weak_rows, "audit_rows": audit_rows, "decision_feature_columns": DECISION_FEATURE_COLUMNS, "forbidden_decision_columns_detected": bad}


def build_train_audit(predictions, open_rows, unresolved_rows, disclosure_rows, ocr_cap_rows, weak_rows, trace_rows):
    vids = [str(v) for v in EXPECTED_SPLITS["train"]]
    out = []
    for vid in vids:
        p = [r for r in predictions if r["video_id"] == vid]
        tr = [r for r in trace_rows if r["video_id"] == vid]
        avg = sum(as_float(r["ad_duration_sec"]) for r in p) / len(p) if p else 0.0
        out.append({"version": VERSION, "split": "train", "video_id": vid, "prediction_count": len(p), "open_interval_candidate_count": sum(1 for r in open_rows if r["video_id"] == vid), "unresolved_long_open_count": sum(1 for r in unresolved_rows if r["video_id"] == vid), "disclosure_notice_rejected_count": sum(1 for r in disclosure_rows if r["video_id"] == vid), "ocr_only_continuity_cap_event_count": sum(1 for r in ocr_cap_rows if r["video_id"] == vid), "weak_continuity_recovery_event_count": sum(1 for r in weak_rows if r["video_id"] == vid), "average_closed_duration_sec": round(avg, 3), "pending_start_timeout_count": sum(1 for r in tr if r["decision_type"] == "start_pending_timeout"), "pending_end_timeout_count": sum(1 for r in tr if r["decision_reason"] == "end_pending_timeout_no_strong_continuity"), "end_pending_timeout_no_strong_continuity_count": sum(1 for r in tr if r["end_pending_timeout_no_strong_continuity_applied"] == "true"), "start_reason_counts_json": json_dumps(Counter(r["start_reason"] for r in p)), "end_reason_counts_json": json_dumps(Counter(r["end_reason"] for r in p)), "state_transition_counts_json": json_dumps(Counter(f"{r['state_before']}->{r['state_after']}" for r in tr)), "audit_columns_used_for_decision": "false", "final_performance_claim": "false"})
    return out

# 사후 검수 helper
def union_intervals(intervals, video_duration=None):
    clean=[]
    for item in intervals:
        s=as_float(item.get('start_sec')); e=as_float(item.get('end_sec'))
        if video_duration and video_duration>0:
            s=max(0.0,min(s,video_duration)); e=max(0.0,min(e,video_duration))
        if e>s: clean.append({**item,'start_sec':s,'end_sec':e,'duration_sec':e-s})
    clean.sort(key=lambda r:(r['start_sec'],r['end_sec']))
    out=[]
    for item in clean:
        if not out or item['start_sec']>out[-1]['end_sec']:
            out.append(dict(item))
        else:
            out[-1]['end_sec']=max(out[-1]['end_sec'], item['end_sec']); out[-1]['duration_sec']=out[-1]['end_sec']-out[-1]['start_sec']
            out[-1]['interval_id']=str(out[-1].get('interval_id',''))+','+str(item.get('interval_id',''))
    return out

def duration_sum(intervals): return sum(max(0.0, as_float(i.get('end_sec'))-as_float(i.get('start_sec'))) for i in intervals)
def overlap_len(a,b,c,d): return max(0.0, min(b,d)-max(a,c))
def overlap_with_union(interval, unioned): return sum(overlap_len(as_float(interval.get('start_sec')), as_float(interval.get('end_sec')), as_float(u.get('start_sec')), as_float(u.get('end_sec'))) for u in unioned)
def overlap_between(left,right): return sum(overlap_len(as_float(a.get('start_sec')),as_float(a.get('end_sec')),as_float(b.get('start_sec')),as_float(b.get('end_sec'))) for a in left for b in right)
def ratio(n,d): return 0.0 if d<=0 else n/d

def parse_actual(root, config, train_ids, durations, warnings):
    rows=read_csv(root/config['input_paths']['actual_label_file'])
    by=defaultdict(list)
    for i,r in enumerate(rows,1):
        vid=row_video_id(r)
        if vid not in train_ids: continue
        seg=str(r.get('segment_type','')).lower()
        if seg and seg not in {'ad_interval','ad_full','ad','advertisement'} and 'ad_interval' not in seg: continue
        s=as_float(r.get('ad_start_sec') or r.get('segment_start_sec'), float('nan')); e=as_float(r.get('ad_end_sec') or r.get('segment_end_sec'), float('nan'))
        if math.isnan(s) or math.isnan(e) or e<=s: warnings.append(f'invalid_actual_interval_excluded:{vid}:{i}'); continue
        dur=durations.get(vid,0.0)
        if dur>0: s=max(0,min(s,dur)); e=max(0,min(e,dur))
        if e>s: by[vid].append({'video_id':vid,'split':'train','interval_type':'actual','interval_id':r.get('ad_interval_id') or r.get('segment_id') or f'actual_{vid}_{i:04d}','start_sec':s,'end_sec':e,'duration_sec':e-s,'reason':seg or 'actual_ad_interval'})
    return by

def parse_pred_like(rows, train_ids, kind, id_col, start_col, end_col, durations, warnings):
    by=defaultdict(list)
    for i,r in enumerate(rows,1):
        vid=row_video_id(r)
        if vid not in train_ids or r.get('split')!='train': continue
        s=as_float(r.get(start_col), float('nan')); e=as_float(r.get(end_col), float('nan'))
        if math.isnan(s) or math.isnan(e) or e<=s: warnings.append(f'invalid_{kind}_excluded:{vid}:{i}'); continue
        dur=durations.get(vid,0.0)
        if dur>0: s=max(0,min(s,dur)); e=max(0,min(e,dur))
        if e>s: by[vid].append({'video_id':vid,'split':'train','interval_type':kind,'interval_id':r.get(id_col) or f'{kind}_{vid}_{i:04d}','start_sec':s,'end_sec':e,'duration_sec':e-s,'reason':r.get('start_reason') or r.get('open_reason') or r.get('unresolved_reason') or ''})
    return by

def severity(actual_cov, combined_cov, open_cov, unresolved_cov, actual_total, combined_total, missed_ratio):
    if actual_total<=0 and combined_total>0: return 'no_actual_but_predicted'
    over=combined_cov/actual_cov if actual_cov>0 else 0
    if combined_cov>=0.5 or over>=3 or open_cov>=0.3 or unresolved_cov>=0.3: return 'severe_over_detection'
    if combined_cov>=0.25 or over>=2: return 'moderate_over_detection'
    if missed_ratio>=0.5: return 'missed_actual'
    return 'reasonable_or_needs_manual_review'

def priority(sev, open_cov, unresolved_cov):
    if sev in {'severe_over_detection','no_actual_but_predicted'} or open_cov>=0.3 or unresolved_cov>=0.3: return 'high'
    if sev in {'moderate_over_detection','missed_actual'}: return 'medium'
    return 'low'

def build_v13_audit_analysis(root, config, detector_result, log_path):
    append_log(log_path, '[STEP 06] Train-only audit/error analysis for v1.3')
    warnings=detector_result['warnings']; outp=config['output_paths']; train_ids={str(v) for v in EXPECTED_SPLITS['train']}; durations=detector_result['split_check']['video_durations']
    actual_by=parse_actual(root,config,train_ids,durations,warnings)
    pred_rows=read_csv(root/outp['prediction_csv']); open_rows=read_csv(root/outp['open_interval_csv']); unresolved_rows=read_csv(root/outp['unresolved_long_open_csv']); trace_rows=read_csv(root/outp['trace_csv'])
    pred_by=parse_pred_like(pred_rows,train_ids,'closed_prediction','prediction_id','ad_start_sec','ad_end_sec',durations,warnings)
    open_by=parse_pred_like(open_rows,train_ids,'open_candidate','open_candidate_id','ad_start_sec','last_anchor_sec',durations,warnings)
    unresolved_by=parse_pred_like(unresolved_rows,train_ids,'unresolved_long_open','unresolved_candidate_id','start_sec','last_anchor_sec',durations,warnings)
    video_rows=[]; interval_rows=[]; open_summary=[]
    for vid in sorted(train_ids, key=lambda x:int(x)):
        dur=durations.get(vid,0.0)
        au=union_intervals(actual_by.get(vid,[]),dur); pu=union_intervals(pred_by.get(vid,[]),dur); ou=union_intervals(open_by.get(vid,[]),dur); uu=union_intervals(unresolved_by.get(vid,[]),dur); combined=union_intervals(pred_by.get(vid,[])+open_by.get(vid,[])+unresolved_by.get(vid,[]),dur)
        actual_total=duration_sum(au); pred_total=duration_sum(pu); open_total=duration_sum(ou); unresolved_total=duration_sum(uu); combined_total=duration_sum(combined)
        overlap=overlap_between(au,pu); fp=max(0,pred_total-overlap); missed=max(0,actual_total-overlap)
        ac=ratio(actual_total,dur); pc=ratio(pred_total,dur); oc=ratio(open_total,dur); uc=ratio(unresolved_total,dur); cc=ratio(combined_total,dur); mr=ratio(missed,actual_total)
        sev=severity(ac,cc,oc,uc,actual_total,combined_total,mr); pri=priority(sev,oc,uc)
        note=[]
        if sev in {'severe_over_detection','no_actual_but_predicted'}: note.append('combined detector coverage remains high versus train actual')
        if uc>0: note.append('long open moved to unresolved failure candidate')
        if mr>=0.5: note.append('missed actual increased/needs review')
        video_rows.append({'video_id':vid,'split':'train','video_duration_sec':round(dur,3),'actual_interval_count':len(actual_by.get(vid,[])),'actual_total_duration_sec':round(actual_total,3),'actual_coverage_ratio':round(ac,6),'closed_prediction_count':len(pred_by.get(vid,[])),'predicted_total_duration_sec':round(pred_total,3),'predicted_coverage_ratio':round(pc,6),'open_interval_count':len(open_by.get(vid,[])),'open_total_duration_proxy_sec':round(open_total,3),'open_coverage_ratio':round(oc,6),'unresolved_long_open_count':len(unresolved_by.get(vid,[])),'unresolved_total_duration_proxy_sec':round(unresolved_total,3),'unresolved_coverage_ratio':round(uc,6),'predicted_plus_open_plus_unresolved_duration_sec':round(combined_total,3),'predicted_plus_open_plus_unresolved_coverage_ratio':round(cc,6),'actual_pred_overlap_duration_sec':round(overlap,3),'actual_pred_overlap_ratio_of_actual':round(ratio(overlap,actual_total),6),'actual_pred_overlap_ratio_of_prediction':round(ratio(overlap,pred_total),6),'false_positive_duration_sec':round(fp,3),'false_positive_ratio_of_video':round(ratio(fp,dur),6),'missed_actual_duration_sec':round(missed,3),'missed_actual_ratio_of_actual':round(mr,6),'over_detection_ratio_vs_actual':'' if ac<=0 else round(cc/ac,6),'severity_level':sev,'review_priority':pri,'diagnosis_note':'; '.join(note) if note else 'manual review recommended'})
        for coll, typ in [(pred_by.get(vid,[]),'closed_prediction'),(open_by.get(vid,[]),'open_candidate'),(unresolved_by.get(vid,[]),'unresolved_long_open')]:
            for p in coll:
                ov=overlap_with_union(p, au); pdur=as_float(p.get('duration_sec')); fp_i=max(0,pdur-ov)
                issue='good_overlap' if ratio(ov,pdur)>=0.5 else ('open_unresolved_candidate' if typ!='closed_prediction' else 'false_positive_candidate')
                interval_rows.append({'video_id':vid,'split':'train','interval_type':typ,'interval_id':p.get('interval_id'),'start_sec':round(as_float(p.get('start_sec')),3),'end_sec':round(as_float(p.get('end_sec')),3),'duration_sec':round(pdur,3),'best_matching_actual_id':'','best_actual_overlap_sec':round(ov,3),'best_actual_overlap_ratio_of_interval':round(ratio(ov,pdur),6),'false_positive_duration_sec':round(fp_i,3),'matched_prediction_ids':'','missed_actual_duration_sec':'','issue_type':issue,'reason':p.get('reason',''),'review_note':'train-only audit; no final performance claim'})
        for a in actual_by.get(vid,[]):
            ov=overlap_with_union(a, pu); adur=as_float(a.get('duration_sec')); miss=max(0,adur-ov)
            interval_rows.append({'video_id':vid,'split':'train','interval_type':'actual','interval_id':a.get('interval_id'),'start_sec':round(as_float(a.get('start_sec')),3),'end_sec':round(as_float(a.get('end_sec')),3),'duration_sec':round(adur,3),'best_matching_actual_id':a.get('interval_id'),'best_actual_overlap_sec':'','best_actual_overlap_ratio_of_interval':'','false_positive_duration_sec':'','matched_prediction_ids':'','missed_actual_duration_sec':round(miss,3),'issue_type':'missed_actual_candidate' if ratio(miss,adur)>=0.5 else 'good_overlap','reason':a.get('reason',''),'review_note':'actual train label used only after detector run for audit'})
        for o in open_by.get(vid,[]):
            odur=as_float(o.get('duration_sec')); ov=overlap_with_union(o, au)
            open_summary.append({'video_id':vid,'split':'train','open_candidate_id':o.get('interval_id'),'start_sec':round(as_float(o.get('start_sec')),3),'end_proxy_sec':round(as_float(o.get('end_sec')),3),'duration_proxy_sec':round(odur,3),'video_duration_sec':round(dur,3),'open_coverage_ratio':round(ratio(odur,dur),6),'overlap_with_actual_sec':round(ov,3),'overlap_ratio_of_open':round(ratio(ov,odur),6),'start_reason':o.get('reason',''),'last_state':'','possible_failure_mode':'needs_manual_review','review_priority':'high' if ratio(odur,dur)>=0.3 else 'medium','diagnosis_note':'open candidate remains separate from unresolved long open'})
    write_csv(root/outp['analysis_video_summary_csv'], video_rows, VIDEO_SUMMARY_COLUMNS)
    write_csv(root/outp['analysis_interval_overlap_csv'], interval_rows, INTERVAL_OVERLAP_COLUMNS)
    write_csv(root/outp['analysis_open_summary_csv'], open_summary, OPEN_SUMMARY_COLUMNS)
    trace_summary=build_trace_summary(trace_rows)
    write_csv(root/outp['analysis_trace_reason_summary_csv'], trace_summary, TRACE_REASON_COLUMNS)
    worst=sorted(video_rows,key=lambda r:(0 if r['review_priority']=='high' else 1, -as_float(r['predicted_plus_open_plus_unresolved_coverage_ratio'])))
    write_csv(root/outp['analysis_worst_cases_csv'], [{'rank':i+1,**r} for i,r in enumerate(worst)], WORST_COLUMNS)
    issues=build_rule_issues(video_rows, detector_result, trace_summary)
    write_csv(root/outp['analysis_rule_issue_candidates_csv'], issues, RULE_ISSUE_COLUMNS)
    return {'video_rows':video_rows,'interval_rows':interval_rows,'open_summary':open_summary,'trace_summary':trace_summary,'worst_rows':worst,'rule_issues':issues}

def build_trace_summary(rows):
    out=[]
    def add(scope,vid,cat,counter,note):
        total=sum(counter.values())
        for k,c in counter.most_common(): out.append({'scope':scope,'video_id':vid,'category':cat,'key':k,'count':c,'ratio_within_category':round(ratio(c,total),6),'note':note})
    scopes=[('overall_train','',rows)] + [('video',vid,[r for r in rows if r.get('video_id')==vid]) for vid in sorted({r.get('video_id') for r in rows}, key=lambda x:int(x))]
    for scope,vid,rr in scopes:
        add(scope,vid,'state_transition',Counter(f"{r.get('state_before')}->{r.get('state_after')}" for r in rr),'train trace only')
        add(scope,vid,'decision_type',Counter(r.get('decision_type','') for r in rr),'train trace only')
        add(scope,vid,'decision_reason',Counter(r.get('decision_reason','') for r in rr),'train trace only')
        add(scope,vid,'in_ad_maintain_reason',Counter(r.get('decision_reason','') for r in rr if r.get('state_before')=='in_ad' and r.get('state_after')=='in_ad'),'in_ad maintain rows')
        flags=Counter({k:0 for k in ['strong_ad_evidence','disclosure_notice_rejected','ocr_only_continuity_cap_applied','weak_continuity_recovery_applied','end_pending_timeout_no_strong_continuity_applied','unresolved_long_open_guard_applied','c3_flag','d2_flag','product_repetition_continues','purchase_cta_or_link_context']})
        for r in rr:
            for k in list(flags.keys()):
                if is_true(r.get(k)): flags[k]+=1
        add(scope,vid,'v1_3_flags',flags,'boolean flag counts; zero rows retained')
    return out

def build_rule_issues(video_rows, det, trace_summary):
    high=[r for r in video_rows if r['review_priority']=='high']; missed=[r for r in video_rows if as_float(r['missed_actual_ratio_of_actual'])>=0.5]
    def vids(rows): return ','.join(r['video_id'] for r in rows[:8])
    return [
        {'issue_id':'V13_RI001','issue_name':'train over-detection reduced candidate still needs viewer review','evidence_from_train':f"high priority videos={len(high)}",'affected_video_count':len(high),'example_video_ids':vids(high),'severity':'high' if high else 'medium','suspected_rule_area':'in_ad_continuity','suggested_direction':'viewer/trace에서 strong evidence cap이 실제 광고를 과도하게 자르지 않는지 확인','caution':'train-only candidate이며 성능 개선 확정 아님','requires_manual_review':'true'},
        {'issue_id':'V13_RI002','issue_name':'missed actual risk from stricter start/continuity','evidence_from_train':f"missed actual ratio>=0.5 videos={len(missed)}",'affected_video_count':len(missed),'example_video_ids':vids(missed),'severity':'high' if len(missed)>=3 else 'medium','suspected_rule_area':'start_confirmation','suggested_direction':'disclosure guard와 weak recovery가 실제 광고 시작을 놓치는지 확인','caution':'validation 전 train worst case 확인 필요','requires_manual_review':'true'},
        {'issue_id':'V13_RI003','issue_name':'unresolved long open should remain failure bucket','evidence_from_train':f"unresolved candidates={len(det['unresolved_rows'])}",'affected_video_count':len(set(r['video_id'] for r in det['unresolved_rows'])),'example_video_ids':','.join(sorted(set(r['video_id'] for r in det['unresolved_rows']), key=lambda x:int(x))[:8]),'severity':'medium','suspected_rule_area':'open_interval_policy','suggested_direction':'unresolved를 prediction으로 보지 않고 별도 failure review로 유지','caution':'test 전 hard-close 금지','requires_manual_review':'true'},
    ]

def summarize_metrics(rows, det):
    n=len(rows)
    return {'train_video_count':n,'actual_total_duration_sec':round(sum(as_float(r['actual_total_duration_sec']) for r in rows),3),'closed_predicted_total_duration_sec':round(sum(as_float(r['predicted_total_duration_sec']) for r in rows),3),'open_interval_total_duration_proxy_sec':round(sum(as_float(r['open_total_duration_proxy_sec']) for r in rows),3),'unresolved_long_open_total_duration_proxy_sec':round(sum(as_float(r['unresolved_total_duration_proxy_sec']) for r in rows),3),'mean_actual_coverage':round(sum(as_float(r['actual_coverage_ratio']) for r in rows)/n,6) if n else 0,'mean_predicted_coverage':round(sum(as_float(r['predicted_coverage_ratio']) for r in rows)/n,6) if n else 0,'mean_open_coverage':round(sum(as_float(r['open_coverage_ratio']) for r in rows)/n,6) if n else 0,'mean_unresolved_coverage':round(sum(as_float(r['unresolved_coverage_ratio']) for r in rows)/n,6) if n else 0,'severe_over_detection_video_count':sum(1 for r in rows if r['severity_level']=='severe_over_detection'),'unresolved_long_open_count':len(det['unresolved_rows']),'disclosure_notice_rejected_count':len(det['disclosure_rows']),'ocr_only_continuity_cap_event_count':len(det['ocr_cap_rows']),'weak_continuity_recovery_event_count':len(det['weak_rows']),'in_ad_maintained_by_ocr_context_count':sum(1 for r in det['trace_rows'] if r['decision_reason']=='in_ad_maintained_by_ocr_context'),'worst_videos':[{'video_id':r['video_id'],'coverage':r['predicted_plus_open_plus_unresolved_coverage_ratio'],'severity':r['severity_level']} for r in sorted(rows,key=lambda x:as_float(x['predicted_plus_open_plus_unresolved_coverage_ratio']),reverse=True)[:5]]}

def build_comparison(root, config, v13_analysis):
    outp=config['output_paths']; v12=read_csv(root/'data/analysis/train_only_detector_error_video_summary_v1_2.csv')
    v12_by={r['video_id']:r for r in v12}; rows=[]
    for r in v13_analysis['video_rows']:
        vid=r['video_id']; b=v12_by.get(vid,{})
        b_cov=as_float(b.get('predicted_plus_open_coverage_ratio',0)); n_cov=as_float(r.get('predicted_plus_open_plus_unresolved_coverage_ratio',0))
        b_miss=as_float(b.get('missed_actual_ratio_of_actual',0)); n_miss=as_float(r.get('missed_actual_ratio_of_actual',0))
        if n_cov < b_cov - 0.05 and n_miss <= b_miss + 0.25: status='better_candidate'
        elif n_miss > b_miss + 0.4: status='worse_candidate'
        elif n_cov < b_cov - 0.05: status='mixed_needs_review'
        elif abs(n_cov-b_cov)<=0.05: status='unchanged'
        else: status='worse_candidate'
        rows.append({'video_id':vid,'split':'train','actual_coverage_ratio':r['actual_coverage_ratio'],'v1_2_predicted_coverage_ratio':b.get('predicted_coverage_ratio',''),'v1_3_predicted_coverage_ratio':r['predicted_coverage_ratio'],'v1_2_open_coverage_ratio':b.get('open_coverage_ratio',''),'v1_3_open_coverage_ratio':r['open_coverage_ratio'],'v1_3_unresolved_coverage_ratio':r['unresolved_coverage_ratio'],'v1_2_predicted_plus_open_coverage_ratio':b.get('predicted_plus_open_coverage_ratio',''),'v1_3_predicted_plus_open_plus_unresolved_coverage_ratio':r['predicted_plus_open_plus_unresolved_coverage_ratio'],'v1_2_closed_prediction_count':b.get('closed_prediction_count',''),'v1_3_closed_prediction_count':r['closed_prediction_count'],'v1_2_open_interval_count':b.get('open_interval_count',''),'v1_3_open_interval_count':r['open_interval_count'],'v1_3_unresolved_long_open_count':r['unresolved_long_open_count'],'v1_2_severity_level':b.get('severity_level',''),'v1_3_severity_level':r['severity_level'],'got_better_or_worse':status,'review_priority':r['review_priority'],'comparison_note':'train-only comparison; not final performance claim'})
    write_csv(root/outp['comparison_csv'], rows, COMPARISON_COLUMNS)
    return rows

def write_adjustment_note(path, metrics, comparison_summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# State Machine Interval Detector v1.3 Train Adjustment Note

## 1. v1.3를 만든 이유

v1.2 train-only error analysis에서 severe over-detection이 광범위하게 확인되어, v1.2를 보존한 상태로 train-only adjustment candidate를 만들었다. 이 문서는 최종 detector 확정이나 성능 개선 claim이 아니다.

## 2. train-only 분석에서 확인된 문제

v1.2는 실제 광고 coverage 대비 closed/open coverage가 과도했고, OCR context medium/high 단독 유지와 end_pending 복귀 흐름이 주요 의심 지점이었다.

## 3. 수정 1: global disclosure guard

초반 OCR start high를 즉시 광고 시작으로 확정하지 않고, 제품/CTA/audio rise 결합 근거가 없으면 disclosure notice review candidate로 분리했다.

## 4. 수정 2: OCR-only continuity cap

OCR context medium/high 단독 유지가 2 anchor 또는 20초를 넘으면 end_pending으로 이동하도록 했다.

## 5. 수정 3: weak continuity recovery

강한 광고 지속 근거가 반복적으로 없으면 end_pending으로 이동하도록 했다. OCR low reliability를 non-ad evidence로 사용한 것은 아니다.

## 6. 수정 4: end_pending timeout no strong continuity policy

end_pending timeout 시 strong_ad_evidence가 없으면 in_ad 복귀가 아니라 종료 확정 후보로 처리한다.

## 7. 수정 5: unresolved long open guard

너무 길고 최근 strong evidence가 없는 open 흐름은 closed prediction이나 일반 open이 아니라 unresolved long open failure candidate로 분리한다.

## 8. v1.3 train 결과 요약

- severe over-detection count: `{metrics['severe_over_detection_video_count']}`
- mean predicted coverage: `{metrics['mean_predicted_coverage']}`
- mean open coverage: `{metrics['mean_open_coverage']}`
- mean unresolved coverage: `{metrics['mean_unresolved_coverage']}`
- unresolved long open count: `{metrics['unresolved_long_open_count']}`
- disclosure notice rejected count: `{metrics['disclosure_notice_rejected_count']}`
- OCR-only continuity cap event count: `{metrics['ocr_only_continuity_cap_event_count']}`
- weak continuity recovery event count: `{metrics['weak_continuity_recovery_event_count']}`

## 9. v1.2 대비 변화

v1.2 대비 변화는 train-only comparison CSV에서 검토한다. `better_candidate`, `mixed_needs_review`, `worse_candidate`는 검토용 분류이며 최종 성능 주장이 아니다.

## 10. 남은 검토 포인트

train worst case를 viewer/trace로 확인해 stricter rule이 실제 광고를 과도하게 놓치지 않는지 확인해야 한다.

## 11. validation/test 보류

validation/test는 이번 작업에서 실행하지 않았다. train에서 원인 유형이 말이 된 뒤 validation 전용 실행으로 이동하고, test는 마지막에 한 번만 실행한다.
""", encoding='utf-8')

def build_summary(report):
    m=report['v1_3_metrics']; b=report['v1_2_baseline_metrics']; errors=report.get('errors',[]); warnings=report.get('warnings',[])
    status='FAILURE' if errors else 'CONDITIONAL_SUCCESS' if warnings else 'SUCCESS'
    sub='\n'.join(f"- {r.get('name')}: `{r.get('status')}`" for r in report.get('sub_agent_results',[])) or '- Pending external Sub Agent validation'
    return f"""# State Machine Interval Detector v1.3 Train Summary

Generated at: `{report.get('end_time')}`

## 작업 개요

- 작업 상태: {status}
- detector: v1.3 train-only adjustment candidate
- base: v1.2 preserved
- validation/test execution: false
- final performance claim: false

## 왜 v1.3가 필요한지

v1.2 train-only 분석에서 severe over-detection이 확인되어, start confirmation과 in_ad continuity, end_pending timeout, long open 처리를 보수적으로 바꾸는 후보를 구현했다.

## 핵심 수치

- train video count: `{m['train_video_count']}`
- v1.2 severe over-detection count: `{b.get('severe_over_detection_video_count')}`
- v1.3 severe over-detection count: `{m['severe_over_detection_video_count']}`
- v1.2 mean predicted/open coverage: `{b.get('mean_predicted_coverage')}` / `{b.get('mean_open_coverage')}`
- v1.3 mean predicted/open/unresolved coverage: `{m['mean_predicted_coverage']}` / `{m['mean_open_coverage']}` / `{m['mean_unresolved_coverage']}`
- actual coverage: `{m['mean_actual_coverage']}`
- unresolved long open count: `{m['unresolved_long_open_count']}`
- disclosure notice rejected count: `{m['disclosure_notice_rejected_count']}`
- OCR-only continuity cap event count: `{m['ocr_only_continuity_cap_event_count']}`
- weak continuity recovery event count: `{m['weak_continuity_recovery_event_count']}`

## Worst Videos

`{m['worst_videos']}`

## Sub Agent Validation Results

{sub}

## 남은 문제와 다음 단계

1. v1.3 train worst cases를 viewer/trace로 검토한다.
2. v1.3이 train에서 말이 되면 validation 전용 실행으로 이동한다.
3. validation 결과가 괜찮으면 freeze 후보로 검토한다.
4. test는 마지막에 한 번만 실행한다.
"""

def scan_forbidden_bundle(bundle):
    found=[]
    if not bundle.exists(): return found
    for p in bundle.rglob('*'):
        if not p.is_file(): continue
        rel=p.relative_to(bundle); parts={x.lower() for x in rel.parts}
        if p.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES or parts & FORBIDDEN_BUNDLE_PARTS: found.append(str(rel))
    return found

def refresh_latest(root, config, log_path):
    append_log(log_path, '[STEP 11] Update latest bundle')
    outp=config['output_paths']; bundle=root/outp['latest_bundle']
    if bundle.exists(): shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    files=[
        'configs/detectors/state_machine_interval_detector_v1_3_train_config.json','scripts/detectors/run_state_machine_interval_detector_v1_3_train.py',
        outp['prediction_csv'],outp['trace_csv'],outp['open_interval_csv'],outp['unresolved_long_open_csv'],outp['disclosure_notice_review_csv'],outp['ocr_only_continuity_cap_csv'],outp['weak_continuity_recovery_csv'],outp['audit_csv'],outp['comparison_csv'],
        outp['analysis_video_summary_csv'],outp['analysis_interval_overlap_csv'],outp['analysis_open_summary_csv'],outp['analysis_trace_reason_summary_csv'],outp['analysis_worst_cases_csv'],outp['analysis_rule_issue_candidates_csv'],
        outp['summary_md'],outp['report_json'],outp['adjustment_note_md'],outp['run_log']]
    copied=[]
    for rel in files:
        src=root/rel
        if src.exists(): shutil.copy2(src,bundle/src.name); copied.append(src.name)
    (bundle/'README_latest_files.md').write_text("""# Latest Files: State Machine Detector v1.3 Train

- v1.3 train-only adjustment candidate.
- Not final detector.
- v1.2 preserved.
- Validation/test not executed.
- Test row-level output excluded.
- No media/frame/cache/model/raw video copied.
- Only small output/config/script/report/log files are copied.

## Files

"""+'\n'.join(f"- `{x}`" for x in sorted(copied+['README_latest_files.md']))+'\n',encoding='utf-8')
    return scan_forbidden_bundle(bundle)

def count_non_train_outputs(root, config):
    outp=config['output_paths']; keys=['prediction_csv','trace_csv','open_interval_csv','unresolved_long_open_csv','disclosure_notice_review_csv','ocr_only_continuity_cap_csv','weak_continuity_recovery_csv','audit_csv','comparison_csv','analysis_video_summary_csv','analysis_interval_overlap_csv','analysis_open_summary_csv','analysis_trace_reason_summary_csv','analysis_worst_cases_csv']
    count=0
    for k in keys:
        p=root/outp[k]
        if p.exists():
            for r in read_csv(p):
                if r.get('split') in EXCLUDED_SPLITS: count+=1
                if r.get('scope')=='video' and str(r.get('video_id')) in {'3','4','7','16','17','18'}: count+=1
    return count

def compare_input_stats(root, config):
    p=root/config['metadata']['input_file_stats_before_path']
    before=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    modified=[]
    for rel,b in before.items():
        if file_stat(root/rel)!=b: modified.append(rel)
    return modified

def final_summary(report):
    m=report['v1_3_metrics']; b=report['v1_2_baseline_metrics']; errors=report.get('errors',[]); warnings=report.get('warnings',[])
    status='FAILURE' if errors else 'CONDITIONAL_SUCCESS' if warnings else 'SUCCESS'
    sub=', '.join(f"{r.get('name')}={r.get('status')}" for r in report.get('sub_agent_results',[])) or 'pending external validation'
    return f"""[STEP 12] Final human-readable summary
작업 상태: {status}
v1.3 detector config path: ./configs/detectors/state_machine_interval_detector_v1_3_train_config.json
v1.3 detector script path: ./scripts/detectors/run_state_machine_interval_detector_v1_3_train.py
v1.3 train prediction row count: {report['row_counts']['prediction_rows']}
v1.3 train trace row count: {report['row_counts']['trace_rows']}
v1.3 open interval candidate count: {report['row_counts']['open_interval_rows']}
v1.3 unresolved long open candidate count: {report['row_counts']['unresolved_long_open_rows']}
disclosure notice rejected count: {m['disclosure_notice_rejected_count']}
OCR-only continuity cap event count: {m['ocr_only_continuity_cap_event_count']}
weak continuity recovery event count: {m['weak_continuity_recovery_event_count']}
v1.2 vs v1.3 severe over-detection count: {b.get('severe_over_detection_video_count')} -> {m['severe_over_detection_video_count']}
v1.2 vs v1.3 mean predicted/open/unresolved coverage: pred {b.get('mean_predicted_coverage')} -> {m['mean_predicted_coverage']}, open {b.get('mean_open_coverage')} -> {m['mean_open_coverage']}, unresolved v1.3={m['mean_unresolved_coverage']}
worst videos 변화: v1.3 worst={m['worst_videos']}
Sub Agent validation 결과: {sub}
old_project_modified: {str(report.get('old_project_modified')).lower()}
input_data_files_modified: {str(bool(report.get('input_files_modified'))).lower()}
validation_output_count: {report.get('validation_output_count')}
test_row_level_output_count: {report.get('test_row_level_output_count')}
latest bundle path: ./outputs/latest_for_chatgpt_state_machine_detector_v1_3_train
warnings/errors: warnings={warnings}, errors={errors}
다음 단계 제안: v1.3 train worst cases를 viewer/trace로 확인; train 결과가 안정적이면 validation 전용 v1.3 실행 절차로 이동; validation은 아직 실행하지 말 것; test는 아직 실행하지 말 것
"""

def run_all(root, config, dry_run=False):
    log_path=root/config['output_paths']['run_log']
    det=run_detector(root,config,log_path,dry_run=dry_run)
    if dry_run: return det
    analysis=build_v13_audit_analysis(root,config,det,log_path)
    append_log(log_path,'[STEP 07] Build v1.2 vs v1.3 train comparison')
    comp=build_comparison(root,config,analysis)
    metrics=summarize_metrics(analysis['video_rows'],det)
    v12_report=json.loads((root/'reports/analysis/train_only_detector_error_analysis_v1_2_report.json').read_text(encoding='utf-8'))
    baseline=v12_report.get('metric_summaries',{})
    comparison_summary=Counter(r['got_better_or_worse'] for r in comp)
    append_log(log_path,'[STEP 08] Generate v1.3 adjustment note')
    write_adjustment_note(root/config['output_paths']['adjustment_note_md'],metrics,comparison_summary)
    append_log(log_path,'[STEP 09] Generate report and summary')
    old_after=root/'reports/detectors/old_project_snapshot_after_state_machine_detector_v1_3_train.tsv'
    snapshot_tree(Path('./_old_project_not_included'), old_after)
    old_before=root/config['metadata']['old_project_snapshot_before']
    old_modified=old_before.exists() and old_before.read_text(encoding='utf-8')!=old_after.read_text(encoding='utf-8')
    input_modified=compare_input_stats(root,config)
    non_train=count_non_train_outputs(root,config)
    report={'task_name':'state_machine_interval_detector_v1_3_train_adjustment_candidate','version':VERSION,'base_version':BASE_VERSION,'project_root':str(root),'run_scope':'train_only','end_time':now_iso(),'input_files':config['input_paths'],'output_files':config['output_paths'],'train_video_ids':EXPECTED_SPLITS['train'],'excluded_validation_video_ids':EXPECTED_SPLITS['validation'],'excluded_test_video_ids':EXPECTED_SPLITS['test'],'v1_2_baseline_metrics':baseline,'v1_3_metrics':metrics,'v1_2_vs_v1_3_comparison_summary':dict(comparison_summary),'rule_deltas':config.get('adjustment_scope',[]),'decision_feature_columns':DECISION_FEATURE_COLUMNS,'forbidden_decision_columns_check':det['forbidden_decision_columns_detected'],'event_counts':{'disclosure_notice_rejected_count':len(det['disclosure_rows']),'ocr_only_continuity_cap_event_count':len(det['ocr_cap_rows']),'weak_continuity_recovery_event_count':len(det['weak_rows']),'unresolved_long_open_count':len(det['unresolved_rows'])},'severe_over_detection_count':metrics['severe_over_detection_video_count'],'worst_videos':metrics['worst_videos'],'row_counts':{'prediction_rows':len(det['predictions']),'trace_rows':len(det['trace_rows']),'open_interval_rows':len(det['open_rows']),'unresolved_long_open_rows':len(det['unresolved_rows']),'disclosure_rows':len(det['disclosure_rows']),'ocr_cap_rows':len(det['ocr_cap_rows']),'weak_recovery_rows':len(det['weak_rows']),'audit_rows':len(det['audit_rows']),'comparison_rows':len(comp)},'sub_agent_results':[],'warnings':det['warnings'],'errors':det['errors'],'old_project_modified':old_modified,'input_files_modified':input_modified,'validation_output_count':non_train,'test_row_level_output_count':0,'latest_for_chatgpt_forbidden_files_found':[],'backup_dir':config['metadata']['backup_dir'],'backup_manifest':str(root/config['metadata']['backup_manifest']),'old_project_snapshot_before':str(old_before),'old_project_snapshot_after':str(old_after),'no_validation_run':True,'no_test_run':True,'no_feature_extraction':True,'no_threshold_tuning':True,'no_final_performance_claim':True}
    if old_modified: report['errors'].append('old project modified')
    if input_modified: report['errors'].append('input/reference files modified')
    if non_train: report['errors'].append(f'validation/test output rows found: {non_train}')
    summary=build_summary(report)
    (root/config['output_paths']['report_json']).parent.mkdir(parents=True,exist_ok=True)
    (root/config['output_paths']['report_json']).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/config['output_paths']['summary_md']).write_text(summary,encoding='utf-8')
    forbidden=refresh_latest(root,config,log_path)
    report['latest_for_chatgpt_forbidden_files_found']=forbidden
    if forbidden: report['errors'].append('forbidden files found in latest bundle')
    (root/config['output_paths']['report_json']).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (root/config['output_paths']['summary_md']).write_text(build_summary(report),encoding='utf-8')
    refresh_latest(root,config,log_path)
    append_log(log_path, final_summary(report))
    return report

def parse_args():
    p=argparse.ArgumentParser(description='Run v1.3 train-only state machine detector adjustment candidate.')
    p.add_argument('--project-root',default='.')
    p.add_argument('--config',default='configs/detectors/state_machine_interval_detector_v1_3_train_config.json')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--no-run',action='store_true')
    return p.parse_args()

def main():
    args=parse_args(); root=Path(args.project_root).resolve()
    if str(root)!='.': raise RuntimeError(f'Refusing root {root}')
    config_path=Path(args.config); config_path=config_path if config_path.is_absolute() else root/config_path
    config=json.loads(config_path.read_text(encoding='utf-8'))
    log_path=root/config['output_paths']['run_log']
    append_log(log_path,'[STEP 05] Run v1.3 detector on train only started')
    if args.no_run:
        validate_split(root,config); append_log(log_path,'[STEP 05] --no-run requested; config loaded successfully'); return 0
    report=run_all(root,config,dry_run=args.dry_run)
    if args.dry_run:
        append_log(log_path,f"[STEP 05] Dry run completed: {report}"); return 0
    return 1 if report.get('errors') else 0

if __name__ == '__main__':
    raise SystemExit(main())
