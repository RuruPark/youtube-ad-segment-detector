#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train/validation-only scene/audio/OCR state-machine interval detector v1.1.

This prototype consumes precomputed visual-anchor aligned scene/audio/OCR features.
It does not extract features, tune thresholds, use test rows for output, or use audit
/ label / nearest-true-boundary columns in detector decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

VERSION = "state_machine_interval_detector_v1_1"
RUN_SPLITS = {"train", "validation"}
TEST_SPLIT = "test"
STATE_NAMES = {"non_ad", "start_pending", "in_ad", "end_pending"}
BANNED_DECISION_PATTERNS = [
    "nearest_true_boundary",
    "distance_to_nearest_true_boundary",
    "is_near_true_boundary",
    "label",
    "true",
    "audit",
]
DECISION_FEATURE_COLUMNS = [
    "candidate_time_sec",
    "audio_start_signal_level",
    "audio_end_signal_level",
    "audio_context_level",
    "audio_before_context_level",
    "audio_after_context_level",
    "audio_score_delta_pre_minus_post",
    "audio_score_delta_post_minus_pre",
    "ocr_start_signal_level",
    "ocr_end_signal_level",
    "ocr_context_level",
    "ocr_context_reliability_level",
    "ocr_score_delta_pre_minus_post",
    "ocr_score_delta_post_minus_pre",
    "ocr_pre_10s_product_brand_count",
    "ocr_post_10s_product_brand_count",
    "possible_end_context_support",
]
PREDICTION_COLUMNS = [
    "prediction_id",
    "version",
    "split",
    "video_id",
    "ad_start_sec",
    "ad_end_sec",
    "ad_duration_sec",
    "start_anchor_id",
    "end_anchor_id",
    "start_reason",
    "end_reason",
    "interval_status",
    "merged_from_prediction_ids",
    "merge_reason",
    "retroactive_start_applied",
    "used_test_row",
    "decision_feature_columns_json",
    "audit_columns_used_for_decision",
]
TRACE_COLUMNS = [
    "version",
    "split",
    "video_id",
    "visual_anchor_id",
    "transition_time_anchor",
    "candidate_time_mmss",
    "state_before",
    "state_after",
    "decision_type",
    "decision_reason",
    "pending_start_time",
    "pending_end_time",
    "pending_anchor_count",
    "pending_elapsed_sec",
    "audio_start_signal_level",
    "audio_end_signal_level",
    "audio_context_level",
    "audio_before_context_level",
    "audio_after_context_level",
    "ocr_start_signal_level",
    "ocr_end_signal_level",
    "ocr_context_level",
    "ocr_context_reliability_level",
    "c3_flag",
    "consecutive_c3_count",
    "d2_flag",
    "product_repetition_continues",
    "low_gap_bridge_applied",
    "bridge_gap_sec",
    "minimum_duration_hold_applied",
    "e2_exception_applied",
    "long_ad_prior_active",
    "retroactive_start_applied",
    "used_for_decision_columns_json",
    "audit_columns_used_for_decision",
]
OPEN_COLUMNS = [
    "open_candidate_id",
    "version",
    "split",
    "video_id",
    "ad_start_sec",
    "last_anchor_sec",
    "start_anchor_id",
    "last_anchor_id",
    "open_state",
    "open_reason",
    "pending_end_time",
    "interval_status",
    "used_test_row",
]
AUDIT_COLUMNS = [
    "version",
    "split",
    "prediction_count",
    "open_interval_candidate_count",
    "average_duration_sec",
    "pending_start_timeout_count",
    "pending_end_timeout_count",
    "start_reason_counts_json",
    "end_reason_counts_json",
    "state_transition_counts_json",
    "optional_boundary_proximity_counts_json",
    "audit_only_columns_used_json",
    "final_performance_claim",
]
FORBIDDEN_BUNDLE_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a", ".jpg", ".jpeg", ".png", ".webp",
    ".pt", ".pth", ".ckpt", ".onnx", ".pkl", ".pickle", ".parquet"
}
FORBIDDEN_BUNDLE_PARTS = {"cache", "tmp", "__pycache__", "frames", "frame_images", "raw_video", "proxy", "checkpoint", "checkpoints"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(root: Path, path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def append_log(path: Path, message: str) -> None:
    print(message, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{now_iso()} {message}\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


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


def normalize_level(value: Any) -> str:
    s = str(value if value is not None else "").strip().lower()
    if s in {"high", "medium", "low"}:
        return s
    return "low_or_unknown"


def is_medium_or_high(value: Any) -> bool:
    return normalize_level(value) in {"medium", "high"}


def is_high(value: Any) -> bool:
    return normalize_level(value) == "high"


def is_low(value: Any) -> bool:
    return normalize_level(value) == "low"


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


def row_time(row: dict[str, Any]) -> float:
    return as_float(row.get("candidate_time_sec"), 0.0)


def row_anchor_id(row: dict[str, Any]) -> str:
    return str(row.get("visual_anchor_id") or row.get("scene_boundary_anchor_id") or "")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def has_banned_decision_columns(columns: list[str]) -> list[str]:
    bad = []
    for col in columns:
        lower = col.lower()
        if any(pattern in lower for pattern in BANNED_DECISION_PATTERNS):
            bad.append(col)
    return bad


def product_repetition(row: dict[str, Any], warnings: list[str]) -> bool:
    pre_col = "ocr_pre_10s_product_brand_count"
    post_col = "ocr_post_10s_product_brand_count"
    if pre_col in row and post_col in row:
        return as_float(row.get(pre_col), 0.0) >= 1 and as_float(row.get(post_col), 0.0) >= 1
    score_pre = "ocr_pre_10s_product_brand_score"
    score_post = "ocr_post_10s_product_brand_score"
    if score_pre in row and score_post in row:
        return as_float(row.get(score_pre), 0.0) > 0 and as_float(row.get(score_post), 0.0) > 0
    if "product_repetition_columns_missing" not in warnings:
        warnings.append("product_repetition_columns_missing")
    return False


def row_flags(row: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    audio_context = normalize_level(row.get("audio_context_level"))
    ocr_context = normalize_level(row.get("ocr_context_level"))
    reliability = normalize_level(row.get("ocr_context_reliability_level"))
    c3 = audio_context == "high" and reliability == "high" and ocr_context == "low"
    d2 = audio_context == "low" and ocr_context == "high"
    known_low = reliability == "high" and ocr_context == "low" and audio_context == "low"
    prod = product_repetition(row, warnings)
    return {
        "audio_start": normalize_level(row.get("audio_start_signal_level")),
        "audio_end": normalize_level(row.get("audio_end_signal_level")),
        "audio_context": audio_context,
        "audio_before": normalize_level(row.get("audio_before_context_level")),
        "audio_after": normalize_level(row.get("audio_after_context_level")),
        "ocr_start": normalize_level(row.get("ocr_start_signal_level")),
        "ocr_end": normalize_level(row.get("ocr_end_signal_level")),
        "ocr_context": ocr_context,
        "reliability": reliability,
        "c3": c3,
        "d2": d2,
        "known_low": known_low,
        "product_repetition": prod,
    }


def next_bridge(row: dict[str, Any], next_row: dict[str, Any] | None, warnings: list[str]) -> tuple[bool, float | str, str]:
    if not next_row:
        return False, "", ""
    gap = row_time(next_row) - row_time(row)
    if gap < 0:
        return False, "", ""
    nf = row_flags(next_row, warnings)
    next_audio_mh = is_medium_or_high(nf["audio_context"])
    next_ocr_mh = is_medium_or_high(nf["ocr_context"])
    if gap <= 10:
        if next_audio_mh or next_ocr_mh:
            if nf["c3"]:
                return True, round(gap, 3), "low_gap_bridge_le_10s_weak_c3_review"
            return True, round(gap, 3), "low_gap_bridge_le_10s_audio_or_ocr_medium_high"
    elif gap <= 20:
        if next_audio_mh and next_ocr_mh and not nf["c3"]:
            return True, round(gap, 3), "low_gap_bridge_10_to_20s_audio_and_ocr_medium_high"
    return False, "", ""


def end_support(row: dict[str, Any], flags: dict[str, Any]) -> tuple[bool, str]:
    reasons = []
    if is_medium_or_high(flags["audio_before"]) and flags["audio_after"] == "low":
        reasons.append("audio_before_medium_high_after_low")
    ocr_audio_drop = as_float(row.get("ocr_score_delta_pre_minus_post"), 0.0) > 0 or as_float(row.get("audio_score_delta_pre_minus_post"), 0.0) > 0 or flags["audio_after"] == "low"
    if is_medium_or_high(flags["ocr_end"]) and ocr_audio_drop:
        reasons.append("ocr_end_medium_high_with_post_drop")
    if flags["known_low"]:
        reasons.append("known_low_flow")
    if reasons:
        hint = normalize_level(row.get("possible_end_context_support"))
        if is_medium_or_high(hint):
            reasons.append(f"discussion_hint_{hint}")
        return True, "+".join(reasons)
    return False, ""


def timeout_end_pending(pending_end: dict[str, Any], duration: float, long_after: float) -> tuple[bool, str]:
    if duration >= long_after:
        if pending_end.get("low_count", 0) >= 2 or pending_end.get("c3_count", 0) >= 2:
            return True, "end_pending_timeout_after_120s_confirmed_by_repeated_low_or_c3"
        return False, "end_pending_timeout_after_120s_return_to_in_ad_no_repeated_low_or_c3"
    return False, "end_pending_timeout_before_120s_return_to_in_ad_no_confirming_low_flow"


def close_interval(raw_predictions: list[dict[str, Any]], split: str, video_id: str, start: dict[str, Any], end_time: float, end_anchor_id: str, end_reason: str, retro: bool) -> None:
    if start is None:
        return
    if end_time <= start["time"]:
        return
    raw_id = f"raw_{len(raw_predictions)+1:06d}"
    raw_predictions.append({
        "raw_prediction_id": raw_id,
        "split": split,
        "video_id": str(video_id),
        "ad_start_sec": round(float(start["time"]), 3),
        "ad_end_sec": round(float(end_time), 3),
        "ad_duration_sec": round(float(end_time) - float(start["time"]), 3),
        "start_anchor_id": start.get("anchor_id", ""),
        "end_anchor_id": end_anchor_id,
        "start_reason": start.get("reason", ""),
        "end_reason": end_reason,
        "retroactive_start_applied": str(bool(retro)).lower(),
    })


def merge_intervals(raw_predictions: list[dict[str, Any]], merge_gap_sec: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    raw_sorted = sorted(raw_predictions, key=lambda r: (r["split"], int(float(r["video_id"])), float(r["ad_start_sec"]), float(r["ad_end_sec"])))
    for raw in raw_sorted:
        if not merged or merged[-1]["split"] != raw["split"] or merged[-1]["video_id"] != raw["video_id"]:
            item = dict(raw)
            item["merged_from_prediction_ids"] = raw["raw_prediction_id"]
            item["merge_reason"] = "none"
            merged.append(item)
            continue
        last = merged[-1]
        gap = float(raw["ad_start_sec"]) - float(last["ad_end_sec"])
        if float(raw["ad_start_sec"]) <= float(last["ad_end_sec"]) or gap <= merge_gap_sec:
            last["ad_end_sec"] = max(float(last["ad_end_sec"]), float(raw["ad_end_sec"]))
            last["ad_end_sec"] = round(last["ad_end_sec"], 3)
            last["ad_duration_sec"] = round(float(last["ad_end_sec"]) - float(last["ad_start_sec"]), 3)
            last["end_anchor_id"] = raw["end_anchor_id"]
            last["end_reason"] = raw["end_reason"]
            last["merged_from_prediction_ids"] = f"{last['merged_from_prediction_ids']},{raw['raw_prediction_id']}"
            last["merge_reason"] = "overlap_or_gap_le_10s"
            if raw.get("retroactive_start_applied") == "true":
                last["retroactive_start_applied"] = "true"
        else:
            item = dict(raw)
            item["merged_from_prediction_ids"] = raw["raw_prediction_id"]
            item["merge_reason"] = "none"
            merged.append(item)
    out = []
    for idx, item in enumerate(merged, start=1):
        item = dict(item)
        item["prediction_id"] = f"smi_v1_1_{idx:06d}"
        item["version"] = VERSION
        item["interval_status"] = "closed"
        item["used_test_row"] = "false"
        item["decision_feature_columns_json"] = json_dumps(DECISION_FEATURE_COLUMNS)
        item["audit_columns_used_for_decision"] = "false"
        out.append(item)
    return out


def validate_split(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    split_rows = read_csv(root / config["input_paths"]["split_file"])
    observed: dict[str, list[int]] = defaultdict(list)
    seeds = set()
    durations: dict[str, float] = {}
    for row in split_rows:
        split = row.get("split", "")
        vid = int(row.get("video_id", -1))
        observed[split].append(vid)
        seeds.add(row.get("split_seed", ""))
        durations[str(vid)] = as_float(row.get("video_duration_sec"), 0.0)
    observed_sorted = {k: sorted(v) for k, v in observed.items()}
    expected = {
        "train": config["train_video_id"],
        "validation": config["validation_video_id"],
        "test": config["test_video_id"],
    }
    ok = observed_sorted == expected and seeds == {str(config["split_seed"])}
    return {"ok": ok, "observed": observed_sorted, "seed_values": sorted(seeds), "video_durations": durations}


def boundary_audit_for_split(rows: list[dict[str, Any]], split: str) -> dict[str, int]:
    split_rows = [r for r in rows if r.get("split") == split]
    counts = {
        "anchor_near_true_boundary_2s": 0,
        "anchor_near_true_boundary_5s": 0,
        "anchor_near_true_boundary_10s": 0,
    }
    for row in split_rows:
        if str(row.get("is_near_true_boundary_2s", "")).lower() == "true":
            counts["anchor_near_true_boundary_2s"] += 1
        if str(row.get("is_near_true_boundary_5s", "")).lower() == "true":
            counts["anchor_near_true_boundary_5s"] += 1
        if str(row.get("is_near_true_boundary_10s", "")).lower() == "true":
            counts["anchor_near_true_boundary_10s"] += 1
    return counts


def run_detector(root: Path, config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    bad_decision_cols = has_banned_decision_columns(DECISION_FEATURE_COLUMNS)
    if bad_decision_cols:
        raise RuntimeError(f"Forbidden audit/label columns in decision feature set: {bad_decision_cols}")
    split_check = validate_split(root, config)
    if not split_check["ok"]:
        raise RuntimeError(f"Fixed split mismatch: {split_check}")
    primary_path = root / config["input_paths"]["primary_detector_input"]
    rows_all = read_csv(primary_path)
    input_columns = list(rows_all[0].keys()) if rows_all else []
    required = [
        "split", "video_id", "visual_anchor_id", "candidate_time_sec", "candidate_time_mmss",
        "audio_start_signal_level", "audio_end_signal_level", "audio_context_level", "audio_before_context_level", "audio_after_context_level",
        "ocr_start_signal_level", "ocr_end_signal_level", "ocr_context_level", "ocr_context_reliability_level",
    ]
    missing_required = [col for col in required if col not in input_columns]
    if missing_required:
        raise RuntimeError(f"Primary input missing required columns: {missing_required}")
    rows = [r for r in rows_all if r.get("split") in RUN_SPLITS]
    test_rows_read = sum(1 for r in rows_all if r.get("split") == TEST_SPLIT)
    if test_rows_read:
        warnings.append(f"primary_input_contains_test_rows_filtered_out={test_rows_read}")
    if any(r.get("split") == TEST_SPLIT for r in rows):
        raise RuntimeError("Test row survived train/validation filter")
    rows.sort(key=lambda r: (int(float(r.get("video_id", 0) or 0)), as_float(r.get("candidate_time_sec"), 0.0), row_anchor_id(r)))
    if dry_run:
        return {"dry_run": True, "input_row_count": len(rows), "warnings": warnings, "errors": errors}

    pending_rules = config["pending_rules"]
    min_duration = float(config["minimum_duration_prior"]["minimum_ad_duration_sec"])
    allow_e2 = bool(config["minimum_duration_prior"].get("allow_e2_exception_within_min_duration", True))
    long_after = float(config["long_ad_prior"]["long_ad_sensitive_after_sec"])
    merge_gap = float(config["interval_merge"]["merge_gap_sec"])

    raw_predictions: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    open_rows: list[dict[str, Any]] = []
    debug_source_rows: list[dict[str, Any]] = []

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("split", ""), str(row.get("video_id", "")))].append(row)

    for (split, video_id), group in grouped.items():
        state = "non_ad"
        current_start: dict[str, Any] | None = None
        pending_start: dict[str, Any] | None = None
        pending_end: dict[str, Any] | None = None
        consecutive_c3 = 0
        for idx, row in enumerate(group):
            t = row_time(row)
            anchor = row_anchor_id(row)
            flags = row_flags(row, warnings)
            state_before = state
            decision_type = "hold"
            decision_reason = "no_state_change"
            pending_anchor_count: int | str = ""
            pending_elapsed: float | str = ""
            bridge_applied = False
            bridge_gap: float | str = ""
            min_hold = False
            e2 = False
            retro = False
            long_active = bool(current_start and (t - current_start["time"] >= long_after))
            next_row = group[idx + 1] if idx + 1 < len(group) else None
            product = flags["product_repetition"]
            c3 = flags["c3"]
            d2 = flags["d2"]
            if c3:
                consecutive_c3 += 1
            else:
                consecutive_c3 = 0

            if state == "non_ad":
                if c3:
                    decision_reason = "c3_no_start_candidate"
                elif flags["ocr_start"] == "high":
                    state = "in_ad"
                    current_start = {"time": t, "anchor_id": anchor, "reason": "ocr_start_high"}
                    decision_type = "start_confirmed"
                    decision_reason = "ocr_start_high"
                elif flags["ocr_start"] == "medium":
                    state = "start_pending"
                    pending_start = {"time": t, "anchor_id": anchor, "index": idx, "reason": "ocr_start_medium_pending"}
                    decision_type = "start_pending_entered"
                    decision_reason = "ocr_start_medium_pending"
                    pending_anchor_count = 0
                    pending_elapsed = 0
                elif flags["reliability"] in {"low", "low_or_unknown"} and is_medium_or_high(flags["audio_start"]):
                    state = "start_pending"
                    pending_start = {"time": t, "anchor_id": anchor, "index": idx, "reason": "ocr_unknown_audio_start_support_pending"}
                    decision_type = "start_pending_entered"
                    decision_reason = "ocr_unknown_audio_start_support_pending"
                    pending_anchor_count = 0
                    pending_elapsed = 0
                else:
                    state = "non_ad"

            elif state == "start_pending":
                assert pending_start is not None
                count = idx - int(pending_start["index"])
                elapsed = t - float(pending_start["time"])
                start_max_count = int(pending_rules["start_pending_max_anchor_count"])
                start_max_elapsed = float(pending_rules["start_pending_max_duration_sec"])
                pending_anchor_count = min(count, start_max_count)
                pending_elapsed = round(min(elapsed, start_max_elapsed), 3)
                if count > start_max_count or elapsed > start_max_elapsed:
                    state = "non_ad"
                    decision_type = "start_pending_timeout"
                    decision_reason = "start_pending_timeout_cancel_to_non_ad"
                    pending_start = None
                elif is_medium_or_high(flags["ocr_context"]) or is_medium_or_high(flags["audio_context"]):
                    state = "in_ad"
                    current_start = {"time": pending_start["time"], "anchor_id": pending_start["anchor_id"], "reason": "start_pending_confirmed_by_continuity"}
                    decision_type = "start_confirmed"
                    decision_reason = "start_pending_confirmed_by_continuity"
                    retro = True
                    pending_start = None
                elif flags["known_low"]:
                    state = "non_ad"
                    decision_type = "start_pending_cancelled"
                    decision_reason = "start_pending_cancelled_by_known_low_flow"
                    pending_start = None
                else:
                    state = "non_ad"
                    decision_type = "start_pending_timeout"
                    decision_reason = "start_pending_timeout_cancel_to_non_ad"
                    pending_start = None

            elif state == "in_ad":
                assert current_start is not None
                duration = t - float(current_start["time"])
                long_active = duration >= long_after
                bridge_applied, bridge_gap, bridge_reason = next_bridge(row, next_row, warnings) if flags["known_low"] else (False, "", "")
                e_support, e_reason = end_support(row, flags)
                if is_medium_or_high(flags["ocr_context"]):
                    decision_reason = "in_ad_maintained_by_ocr_context"
                elif product:
                    decision_reason = "in_ad_maintained_by_product_repetition"
                elif is_medium_or_high(flags["audio_context"]) and flags["reliability"] in {"low", "low_or_unknown"}:
                    decision_reason = "in_ad_maintained_by_audio_with_ocr_unknown"
                elif d2:
                    decision_reason = "in_ad_maintained_by_d2_audio_low_ocr_high"
                elif bridge_applied:
                    decision_reason = bridge_reason
                    decision_type = "low_gap_bridge"
                elif e_support or consecutive_c3 >= 2:
                    pending_reason = "repeated_c3_end_pending" if consecutive_c3 >= 2 and not e_support else "end_support_pending"
                    if duration < min_duration:
                        next_low = False
                        if allow_e2 and flags["known_low"] and next_row is not None:
                            nf = row_flags(next_row, warnings)
                            next_low = bool(nf["known_low"])
                        if next_low:
                            e2 = True
                        else:
                            min_hold = True
                            decision_reason = "minimum_duration_hold_before_20s"
                    if not min_hold:
                        state = "end_pending"
                        pending_end = {
                            "time": t,
                            "anchor_id": anchor,
                            "index": idx,
                            "reason": pending_reason,
                            "low_count": 1 if flags["known_low"] else 0,
                            "c3_count": 1 if c3 else 0,
                        }
                        decision_type = "end_pending_entered"
                        decision_reason = pending_reason if not e2 else f"{pending_reason}+e2_exception"
                        pending_anchor_count = 0
                        pending_elapsed = 0
                else:
                    decision_reason = "in_ad_no_end_support"

            elif state == "end_pending":
                assert pending_end is not None and current_start is not None
                count = idx - int(pending_end["index"])
                elapsed = t - float(pending_end["time"])
                end_max_count = int(pending_rules["end_pending_max_anchor_count"])
                end_max_elapsed = float(pending_rules["end_pending_max_duration_sec"])
                pending_anchor_count = min(count, end_max_count)
                pending_elapsed = round(min(elapsed, end_max_elapsed), 3)
                duration = float(pending_end["time"]) - float(current_start["time"])
                long_active = duration >= long_after
                if count > end_max_count or elapsed > end_max_elapsed:
                    confirm, reason = timeout_end_pending(pending_end, duration, long_after)
                    if confirm:
                        close_interval(raw_predictions, split, video_id, current_start, float(pending_end["time"]), str(pending_end["anchor_id"]), reason, False)
                        current_start = None
                        state = "non_ad"
                        decision_type = "end_confirmed"
                    else:
                        state = "in_ad"
                        decision_type = "end_pending_timeout"
                    decision_reason = reason
                    pending_end = None
                else:
                    if flags["known_low"]:
                        pending_end["low_count"] = int(pending_end.get("low_count", 0)) + 1
                    if c3:
                        pending_end["c3_count"] = int(pending_end.get("c3_count", 0)) + 1
                    if is_medium_or_high(flags["ocr_context"]) or product or d2:
                        state = "in_ad"
                        decision_type = "end_pending_cancelled"
                        decision_reason = "end_pending_cancelled_by_continuity"
                        pending_end = None
                    elif flags["reliability"] in {"low", "low_or_unknown"}:
                        if count >= int(pending_rules["end_pending_max_anchor_count"]):
                            confirm, reason = timeout_end_pending(pending_end, duration, long_after)
                            if confirm:
                                close_interval(raw_predictions, split, video_id, current_start, float(pending_end["time"]), str(pending_end["anchor_id"]), reason, False)
                                current_start = None
                                state = "non_ad"
                                decision_type = "end_confirmed"
                            else:
                                state = "in_ad"
                                decision_type = "end_pending_timeout"
                            decision_reason = reason
                            pending_end = None
                        else:
                            decision_reason = "end_pending_held_ocr_reliability_low"
                    elif flags["known_low"]:
                        close_interval(raw_predictions, split, video_id, current_start, float(pending_end["time"]), str(pending_end["anchor_id"]), "end_pending_confirmed_by_known_low_flow", False)
                        current_start = None
                        state = "non_ad"
                        decision_type = "end_confirmed"
                        decision_reason = "end_pending_confirmed_by_known_low_flow"
                        pending_end = None
                    else:
                        if count >= int(pending_rules["end_pending_max_anchor_count"]):
                            confirm, reason = timeout_end_pending(pending_end, duration, long_after)
                            if confirm:
                                close_interval(raw_predictions, split, video_id, current_start, float(pending_end["time"]), str(pending_end["anchor_id"]), reason, False)
                                current_start = None
                                state = "non_ad"
                                decision_type = "end_confirmed"
                            else:
                                state = "in_ad"
                                decision_type = "end_pending_timeout"
                            decision_reason = reason
                            pending_end = None
                        else:
                            decision_reason = "end_pending_waiting_for_confirmation"

            if state not in STATE_NAMES:
                raise RuntimeError(f"Invalid state generated: {state}")
            trace_rows.append({
                "version": VERSION,
                "split": split,
                "video_id": video_id,
                "visual_anchor_id": anchor,
                "transition_time_anchor": round(t, 3),
                "candidate_time_mmss": row.get("candidate_time_mmss", ""),
                "state_before": state_before,
                "state_after": state,
                "decision_type": decision_type,
                "decision_reason": decision_reason,
                "pending_start_time": "" if pending_start is None else round(float(pending_start["time"]), 3),
                "pending_end_time": "" if pending_end is None else round(float(pending_end["time"]), 3),
                "pending_anchor_count": pending_anchor_count,
                "pending_elapsed_sec": pending_elapsed,
                "audio_start_signal_level": flags["audio_start"],
                "audio_end_signal_level": flags["audio_end"],
                "audio_context_level": flags["audio_context"],
                "audio_before_context_level": flags["audio_before"],
                "audio_after_context_level": flags["audio_after"],
                "ocr_start_signal_level": flags["ocr_start"],
                "ocr_end_signal_level": flags["ocr_end"],
                "ocr_context_level": flags["ocr_context"],
                "ocr_context_reliability_level": flags["reliability"],
                "c3_flag": str(bool(c3)).lower(),
                "consecutive_c3_count": consecutive_c3,
                "d2_flag": str(bool(d2)).lower(),
                "product_repetition_continues": str(bool(product)).lower(),
                "low_gap_bridge_applied": str(bool(bridge_applied)).lower(),
                "bridge_gap_sec": bridge_gap,
                "minimum_duration_hold_applied": str(bool(min_hold)).lower(),
                "e2_exception_applied": str(bool(e2)).lower(),
                "long_ad_prior_active": str(bool(long_active)).lower(),
                "retroactive_start_applied": str(bool(retro)).lower(),
                "used_for_decision_columns_json": json_dumps(DECISION_FEATURE_COLUMNS),
                "audit_columns_used_for_decision": "false",
            })
            debug_source_rows.append(row)

        last_row = group[-1] if group else None
        if last_row is not None and current_start is not None:
            open_rows.append({
                "open_candidate_id": f"open_{len(open_rows)+1:06d}",
                "version": VERSION,
                "split": split,
                "video_id": video_id,
                "ad_start_sec": round(float(current_start["time"]), 3),
                "last_anchor_sec": round(row_time(last_row), 3),
                "start_anchor_id": current_start.get("anchor_id", ""),
                "last_anchor_id": row_anchor_id(last_row),
                "open_state": state,
                "open_reason": "close_open_interval_at_video_end_false",
                "pending_end_time": "" if pending_end is None else round(float(pending_end["time"]), 3),
                "interval_status": "open_candidate",
                "used_test_row": "false",
            })
        if pending_start is not None:
            # 확정되지 않은 시작 후보는 open interval로 만들지 않는다.
            pass

    predictions = merge_intervals(raw_predictions, merge_gap)
    if any(r.get("split") == TEST_SPLIT for r in predictions + trace_rows + open_rows):
        raise RuntimeError("Test split output detected")
    if any(float(r["ad_start_sec"]) >= float(r["ad_end_sec"]) for r in predictions):
        raise RuntimeError("Closed interval with start >= end detected")

    output_paths = config["output_paths"]
    write_csv(root / output_paths["prediction_csv"], predictions, PREDICTION_COLUMNS)
    write_csv(root / output_paths["trace_csv"], trace_rows, TRACE_COLUMNS)
    write_csv(root / output_paths["open_interval_csv"], open_rows, OPEN_COLUMNS)

    audit_rows: list[dict[str, Any]] = []
    for split in ["train", "validation"]:
        split_preds = [p for p in predictions if p.get("split") == split]
        split_open = [o for o in open_rows if o.get("split") == split]
        split_trace = [t for t in trace_rows if t.get("split") == split]
        avg = sum(float(p["ad_duration_sec"]) for p in split_preds) / len(split_preds) if split_preds else 0.0
        audit_rows.append({
            "version": VERSION,
            "split": split,
            "prediction_count": len(split_preds),
            "open_interval_candidate_count": len(split_open),
            "average_duration_sec": round(avg, 3),
            "pending_start_timeout_count": sum(1 for t in split_trace if t["decision_type"] == "start_pending_timeout"),
            "pending_end_timeout_count": sum(1 for t in split_trace if t["decision_type"] == "end_pending_timeout"),
            "start_reason_counts_json": json_dumps(Counter(p["start_reason"] for p in split_preds)),
            "end_reason_counts_json": json_dumps(Counter(p["end_reason"] for p in split_preds)),
            "state_transition_counts_json": json_dumps(Counter(f"{t['state_before']}->{t['state_after']}" for t in split_trace)),
            "optional_boundary_proximity_counts_json": json_dumps(boundary_audit_for_split(debug_source_rows, split)),
            "audit_only_columns_used_json": json_dumps(["nearest_true_boundary_type", "nearest_true_boundary_sec", "distance_to_nearest_true_boundary_sec", "is_near_true_boundary_2s", "is_near_true_boundary_5s", "is_near_true_boundary_10s"]),
            "final_performance_claim": "false",
        })
    write_csv(root / output_paths["audit_csv"], audit_rows, AUDIT_COLUMNS)

    input_after = {}
    input_paths_for_mod_check = [
        config["input_paths"]["split_file"],
        config["input_paths"]["primary_detector_input"],
        config["input_paths"]["compact_discussion_input"],
        config["input_paths"]["canonical_visual_anchor"],
        config["input_paths"]["canonical_visual_anchor_with_split"],
        config["input_paths"]["audio_context_features"],
        config["input_paths"]["ocr_context_features"],
    ]
    for relpath in input_paths_for_mod_check:
        input_after[relpath] = file_stat(root / relpath)

    metadata = config.get("metadata", {})
    before_stats_path = root / metadata.get("input_file_stats_before_path", "") if metadata.get("input_file_stats_before_path") else None
    input_data_files_modified: list[str] = []
    if before_stats_path and before_stats_path.exists():
        before_stats = json.loads(before_stats_path.read_text(encoding="utf-8"))
        for relpath in input_paths_for_mod_check:
            before = before_stats.get(relpath)
            after = input_after.get(relpath)
            if before and after and before != after:
                input_data_files_modified.append(relpath)

    prediction_counts = Counter(p["split"] for p in predictions)
    trace_counts = Counter(t["split"] for t in trace_rows)
    open_counts = Counter(o["split"] for o in open_rows)
    pending_timeout_counts = {
        "start_pending_timeout": sum(1 for t in trace_rows if t["decision_type"] == "start_pending_timeout"),
        "end_pending_timeout": sum(1 for t in trace_rows if t["decision_type"] == "end_pending_timeout"),
    }
    report = {
        "task_name": "state_machine_interval_detector_v1_1_train_val_prototype",
        "version": VERSION,
        "project_root": str(root),
        "start_time": metadata.get("task_start_time"),
        "end_time": now_iso(),
        "estimated_duration": metadata.get("estimated_duration"),
        "backup_dir": metadata.get("backup_dir"),
        "backed_up_files": metadata.get("backed_up_files", []),
        "input_file_stats_before_path": metadata.get("input_file_stats_before_path"),
        "split_check": split_check,
        "input_paths": config["input_paths"],
        "output_paths": config["output_paths"],
        "primary_input_row_count": len(rows_all),
        "detector_input_row_count": len(rows),
        "test_rows_filtered_from_primary_input": test_rows_read,
        "prediction_row_count": len(predictions),
        "trace_row_count": len(trace_rows),
        "open_interval_candidate_count": len(open_rows),
        "audit_row_count": len(audit_rows),
        "prediction_count_by_split": dict(prediction_counts),
        "trace_count_by_split": dict(trace_counts),
        "open_interval_count_by_split": dict(open_counts),
        "pending_timeout_counts": pending_timeout_counts,
        "decision_feature_columns": DECISION_FEATURE_COLUMNS,
        "forbidden_decision_columns_detected": bad_decision_cols,
        "audit_columns_used_for_decision": False,
        "nearest_true_boundary_usage": config["leakage_guard_settings"]["nearest_true_boundary_usage"],
        "validation_usage": config["leakage_guard_settings"]["validation_usage"],
        "threshold_level_keyword_rule_design": config["leakage_guard_settings"]["threshold_level_keyword_rule_design"],
        "no_detector_on_test": True,
        "test_row_level_output_count": 0,
        "no_test_prediction_trace_or_open_interval": True,
        "no_feature_extraction": True,
        "no_threshold_tuning": True,
        "final_performance_claim": False,
        "close_open_interval_at_video_end": False,
        "input_data_files_modified": input_data_files_modified,
        "sub_agent_validation_results": [],
        "latest_for_chatgpt_forbidden_files_found": [],
        "warnings": warnings,
        "errors": errors,
    }
    if input_data_files_modified:
        report["errors"].append("Input data feature files changed during detector run")
    summary = build_summary(report)
    (root / output_paths["report_json"]).parent.mkdir(parents=True, exist_ok=True)
    (root / output_paths["report_json"]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / output_paths["summary_md"]).write_text(summary, encoding="utf-8")
    return report


def build_summary(report: dict[str, Any]) -> str:
    warnings = report.get("warnings", [])
    errors = report.get("errors", [])
    subagent = report.get("sub_agent_validation_results", [])
    sub_lines = [f"- {r.get('name')}: `{r.get('status')}`" for r in subagent] if subagent else ["- Pending or not yet recorded"]
    warn_lines = [f"- {w}" for w in warnings] if warnings else ["- None"]
    err_lines = [f"- {e}" for e in errors] if errors else ["- None"]
    return f"""# State Machine Interval Detector v1.1 Summary

Generated at: `{report.get('end_time')}`

## Status

- 작업 상태: {'SUCCESS' if not errors else 'FAILURE'}
- detector scope: train/validation only
- detector implementation: prototype
- feature extraction: no
- threshold tuning: no
- test row-level output count: `{report.get('test_row_level_output_count')}`
- final performance claim: false

## Outputs

- prediction rows: `{report.get('prediction_row_count')}`
- trace rows: `{report.get('trace_row_count')}`
- open interval candidate rows: `{report.get('open_interval_candidate_count')}`
- audit rows: `{report.get('audit_row_count')}`
- prediction count by split: `{report.get('prediction_count_by_split')}`
- pending timeout counts: `{report.get('pending_timeout_counts')}`

## Leakage Guard

- Decision columns exclude nearest_true_boundary / label / true / audit columns.
- `nearest_true_boundary_usage = audit_only_not_scoring`
- Validation audit is post-hoc only and does not change config or thresholds.
- Test split is excluded from prediction, trace, open interval, and audit outputs.

## Sub Agent Validation Results

{chr(10).join(sub_lines)}

## Warnings

{chr(10).join(warn_lines)}

## Errors

{chr(10).join(err_lines)}

## Next Step

Review validation audit results, then either freeze the rule or create a detector v1.2 adjustment-candidate task. Do not run test yet.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run state-machine interval detector v1.1 on train/validation only.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/detectors/state_machine_interval_detector_v1_1_config.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/config without writing detector outputs.")
    parser.add_argument("--no-run", action="store_true", help="Load and validate config, then exit without detector execution.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    log_path = root / config["output_paths"]["run_log"]
    append_log(log_path, "[STEP 05] Run detector")
    if args.no_run:
        append_log(log_path, "[STEP 05] --no-run requested; config loaded successfully")
        return 0
    try:
        report = run_detector(root, config, dry_run=args.dry_run)
        append_log(log_path, f"[STEP 05] Detector completed: predictions={report.get('prediction_row_count')}, trace={report.get('trace_row_count')}, open={report.get('open_interval_candidate_count')}")
        if report.get("errors"):
            append_log(log_path, f"[STEP 05] Detector completed with errors: {report.get('errors')}")
            return 1
        return 0
    except Exception as exc:
        append_log(log_path, f"[STEP 05] Detector failed: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
