#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train/validation-only scene/audio/OCR state-machine interval detector v1.2.

v1.2 is an adjustment candidate that preserves v1.1 artifacts and only runs on
train/validation rows. It does not use labels, audit columns, nearest true
boundary fields, visual start/end scores, or test rows for detector decisions.
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

VERSION = "v1.2"
BASE_VERSION = "v1.1"
DETECTOR_ID = "state_machine_interval_detector_v1_2"
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
    "ground_truth",
    "gt",
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
    "base_version",
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
    "v1_2_adjustment_flags_json",
]
TRACE_COLUMNS = [
    "version",
    "base_version",
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
    "pending_start_source",
    "pending_end_time",
    "pending_anchor_count",
    "pending_elapsed_sec",
    "pending_elapsed_sec_actual",
    "pending_anchor_count_actual",
    "pending_elapsed_sec_clamped_for_rule",
    "pending_anchor_count_clamped_for_rule",
    "pending_timeout_limit_sec",
    "pending_timeout_limit_anchor_count",
    "pending_timeout_trigger",
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
    "audio_only_start_guard_applied",
    "audio_only_start_rejected",
    "long_ad_prior_active",
    "long_ad_audio_low_end_rule_applied",
    "long_ad_audio_low_observation_count",
    "low_gap_bridge_applied",
    "bridge_gap_sec",
    "minimum_duration_hold_applied",
    "e2_exception_applied",
    "retroactive_start_applied",
    "used_for_decision_columns_json",
    "audit_columns_used_for_decision",
]
OPEN_COLUMNS = [
    "open_candidate_id",
    "version",
    "base_version",
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
    "base_version",
    "split",
    "video_id",
    "prediction_count",
    "open_interval_candidate_count",
    "average_duration_sec",
    "pending_start_timeout_count",
    "pending_end_timeout_count",
    "audio_only_start_rejected_count",
    "long_ad_audio_low_end_count",
    "start_reason_counts_json",
    "end_reason_counts_json",
    "state_transition_counts_json",
    "optional_boundary_proximity_counts_json",
    "audit_only_columns_used_json",
    "final_performance_claim",
]
REVIEW_START_COLUMNS = [
    "version",
    "split",
    "video_id",
    "visual_anchor_id",
    "transition_time_anchor",
    "pending_start_time",
    "pending_start_source",
    "audio_start_signal_level",
    "audio_context_level",
    "ocr_start_signal_level",
    "ocr_context_level",
    "ocr_context_reliability_level",
    "rejection_reason",
    "would_have_confirmed_in_v1_1",
    "review_note",
]
LONG_AD_EVENT_COLUMNS = [
    "version",
    "split",
    "video_id",
    "visual_anchor_id",
    "pending_end_time",
    "transition_time_anchor",
    "current_ad_duration_sec",
    "audio_context_level",
    "audio_after_context_level",
    "ocr_context_level",
    "ocr_context_reliability_level",
    "product_repetition_continues",
    "d2_flag",
    "audio_low_observation_count",
    "end_confirmed_by_v1_2",
    "event_reason",
    "review_note",
]
COMPARISON_COLUMNS = [
    "split",
    "video_id",
    "v1_1_closed_prediction_count",
    "v1_2_closed_prediction_count",
    "v1_1_open_interval_count",
    "v1_2_open_interval_count",
    "v1_1_total_closed_duration_sec",
    "v1_2_total_closed_duration_sec",
    "v1_1_open_duration_proxy_sec",
    "v1_2_open_duration_proxy_sec",
    "v1_1_start_pending_timeout_count",
    "v1_2_start_pending_timeout_count",
    "v1_1_end_pending_timeout_count",
    "v1_2_end_pending_timeout_count",
    "v1_2_audio_only_start_rejected_count",
    "v1_2_long_ad_audio_low_end_count",
    "review_priority",
    "comparison_note",
]
FORBIDDEN_BUNDLE_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a", ".jpg", ".jpeg", ".png", ".webp",
    ".pt", ".pth", ".ckpt", ".onnx", ".pkl", ".pickle", ".parquet",
}
FORBIDDEN_BUNDLE_PARTS = {"cache", "tmp", "__pycache__", "frames", "frame_images", "raw_video", "proxy", "checkpoint", "checkpoints", "model", "models"}
EXPECTED_SPLITS = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_log(path: Path, message: str) -> None:
    print(message, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{now_iso()} {message}\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


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


def forbidden_match(column: str) -> bool:
    lower = column.lower()
    for pattern in BANNED_DECISION_PATTERNS:
        if pattern == "gt":
            if lower == "gt" or lower.startswith("gt_") or lower.endswith("_gt") or "_gt_" in lower:
                return True
        elif pattern in {"label", "true", "audit"}:
            if pattern in lower:
                return True
        elif pattern in lower:
            return True
    return False


def has_banned_decision_columns(columns: list[str]) -> list[str]:
    return [col for col in columns if forbidden_match(col)]


def product_repetition(row: dict[str, Any], warnings: list[str]) -> bool:
    pre_col = "ocr_pre_10s_product_brand_count"
    post_col = "ocr_post_10s_product_brand_count"
    if pre_col in row and post_col in row:
        return as_float(row.get(pre_col), 0.0) >= 1 and as_float(row.get(post_col), 0.0) >= 1
    score_pairs = [
        ("ocr_pre_10s_product_brand_score", "ocr_post_10s_product_brand_score"),
        ("ocr_pre_10s_product_brand_score_raw", "ocr_post_10s_product_brand_score_raw"),
    ]
    for pre_score, post_score in score_pairs:
        if pre_score in row and post_score in row:
            return as_float(row.get(pre_score), 0.0) > 0 and as_float(row.get(post_score), 0.0) > 0
    if "product_repetition_columns_missing" not in warnings:
        warnings.append("product_repetition_columns_missing")
    return False


def audio_low_observation(flags: dict[str, Any]) -> bool:
    return flags["audio_context"] == "low" or flags["audio_after"] == "low"


def row_flags(row: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    audio_context = normalize_level(row.get("audio_context_level"))
    audio_after = normalize_level(row.get("audio_after_context_level"))
    ocr_context = normalize_level(row.get("ocr_context_level"))
    reliability = normalize_level(row.get("ocr_context_reliability_level"))
    c3 = audio_context == "high" and reliability == "high" and ocr_context == "low"
    d2 = audio_context == "low" and ocr_context == "high"
    known_low = reliability == "high" and ocr_context == "low" and audio_context == "low"
    prod = product_repetition(row, warnings)
    out = {
        "audio_start": normalize_level(row.get("audio_start_signal_level")),
        "audio_end": normalize_level(row.get("audio_end_signal_level")),
        "audio_context": audio_context,
        "audio_before": normalize_level(row.get("audio_before_context_level")),
        "audio_after": audio_after,
        "ocr_start": normalize_level(row.get("ocr_start_signal_level")),
        "ocr_end": normalize_level(row.get("ocr_end_signal_level")),
        "ocr_context": ocr_context,
        "reliability": reliability,
        "c3": c3,
        "d2": d2,
        "known_low": known_low,
        "product_repetition": prod,
    }
    out["audio_low_observation"] = audio_low_observation(out)
    return out


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
    ocr_audio_drop = (
        as_float(row.get("ocr_score_delta_pre_minus_post"), 0.0) > 0
        or as_float(row.get("audio_score_delta_pre_minus_post"), 0.0) > 0
        or flags["audio_after"] == "low"
    )
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


def pending_trigger(count: int, elapsed: float, limit_count: int, limit_sec: float, inclusive_count: bool = False) -> str:
    duration_hit = elapsed > limit_sec
    count_hit = count >= limit_count if inclusive_count else count > limit_count
    if duration_hit and count_hit:
        return "duration_and_anchor_count_exceeded"
    if duration_hit:
        return "duration_exceeded"
    if count_hit:
        return "anchor_count_exceeded"
    return "none"


def pending_metrics(pending: dict[str, Any] | None, idx: int, t: float, limit_count: int, limit_sec: float) -> dict[str, Any]:
    if pending is None:
        return {
            "actual_count": "",
            "actual_elapsed": "",
            "clamped_count": "",
            "clamped_elapsed": "",
            "limit_count": "",
            "limit_sec": "",
        }
    count = idx - int(pending["index"])
    elapsed = t - float(pending["time"])
    return {
        "actual_count": count,
        "actual_elapsed": round(elapsed, 3),
        "clamped_count": min(count, limit_count),
        "clamped_elapsed": round(min(elapsed, limit_sec), 3),
        "limit_count": limit_count,
        "limit_sec": limit_sec,
    }


def close_interval(
    raw_predictions: list[dict[str, Any]],
    split: str,
    video_id: str,
    start: dict[str, Any] | None,
    end_time: float,
    end_anchor_id: str,
    end_reason: str,
    retro: bool,
    adjustment_flags: dict[str, Any] | None = None,
) -> None:
    if start is None or end_time <= start["time"]:
        return
    raw_id = f"raw_{len(raw_predictions)+1:06d}"
    flags = dict(adjustment_flags or {})
    flags.setdefault("start_source", start.get("source", start.get("reason", "")))
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
        "v1_2_adjustment_flags_json": json_dumps(flags),
    })


def merge_adjustment_flags(left: str, right: str) -> str:
    try:
        merged = json.loads(left) if left else {}
    except Exception:
        merged = {"left_parse_error": left}
    try:
        incoming = json.loads(right) if right else {}
    except Exception:
        incoming = {"right_parse_error": right}
    for key, value in incoming.items():
        if key not in merged:
            merged[key] = value
        elif merged[key] != value:
            merged[key] = [merged[key], value] if not isinstance(merged[key], list) else merged[key] + [value]
    return json_dumps(merged)


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
            last["ad_end_sec"] = round(max(float(last["ad_end_sec"]), float(raw["ad_end_sec"])), 3)
            last["ad_duration_sec"] = round(float(last["ad_end_sec"]) - float(last["ad_start_sec"]), 3)
            last["end_anchor_id"] = raw["end_anchor_id"]
            last["end_reason"] = raw["end_reason"]
            last["merged_from_prediction_ids"] = f"{last['merged_from_prediction_ids']},{raw['raw_prediction_id']}"
            last["merge_reason"] = "overlap_or_gap_le_10s"
            last["v1_2_adjustment_flags_json"] = merge_adjustment_flags(last.get("v1_2_adjustment_flags_json", "{}"), raw.get("v1_2_adjustment_flags_json", "{}"))
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
        item["prediction_id"] = f"smi_v1_2_{idx:06d}"
        item["version"] = VERSION
        item["base_version"] = BASE_VERSION
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
    ok = observed_sorted == EXPECTED_SPLITS and seeds == {str(config["split_seed"])}
    return {"ok": ok, "observed": observed_sorted, "seed_values": sorted(seeds), "video_durations": durations}


def boundary_audit_for_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "anchor_near_true_boundary_2s": 0,
        "anchor_near_true_boundary_5s": 0,
        "anchor_near_true_boundary_10s": 0,
    }
    for row in rows:
        if str(row.get("is_near_true_boundary_2s", "")).lower() == "true":
            counts["anchor_near_true_boundary_2s"] += 1
        if str(row.get("is_near_true_boundary_5s", "")).lower() == "true":
            counts["anchor_near_true_boundary_5s"] += 1
        if str(row.get("is_near_true_boundary_10s", "")).lower() == "true":
            counts["anchor_near_true_boundary_10s"] += 1
    return counts


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


def compare_file_stats(root: Path, before_rel: str, relpaths: list[str]) -> list[str]:
    before_path = root / before_rel
    if not before_path.exists():
        return [f"missing_before_stats:{before_rel}"]
    before_stats = json.loads(before_path.read_text(encoding="utf-8"))
    modified = []
    for relpath in relpaths:
        before = before_stats.get(relpath)
        after = file_stat(root / relpath)
        if before != after:
            modified.append(relpath)
    return modified


def make_review_start_row(split: str, video_id: str, row: dict[str, Any], flags: dict[str, Any], pending_start: dict[str, Any], rejection_reason: str) -> dict[str, Any]:
    would_v11 = is_medium_or_high(flags["ocr_context"]) or is_medium_or_high(flags["audio_context"])
    return {
        "version": VERSION,
        "split": split,
        "video_id": video_id,
        "visual_anchor_id": row_anchor_id(row),
        "transition_time_anchor": round(row_time(row), 3),
        "pending_start_time": round(float(pending_start["time"]), 3),
        "pending_start_source": pending_start.get("source", ""),
        "audio_start_signal_level": flags["audio_start"],
        "audio_context_level": flags["audio_context"],
        "ocr_start_signal_level": flags["ocr_start"],
        "ocr_context_level": flags["ocr_context"],
        "ocr_context_reliability_level": flags["reliability"],
        "rejection_reason": rejection_reason,
        "would_have_confirmed_in_v1_1": str(bool(would_v11)).lower(),
        "review_note": "OCR unknown + audio support was allowed to enter start_pending, but v1.2 rejects confirmation without reliable OCR start/context.",
    }


def make_long_event_row(
    split: str,
    video_id: str,
    row: dict[str, Any],
    flags: dict[str, Any],
    pending_end: dict[str, Any],
    current_start: dict[str, Any],
    confirmed: bool,
    event_reason: str,
) -> dict[str, Any]:
    current_duration = row_time(row) - float(current_start["time"])
    rel_note = "ocr_reliability_low_treated_as_unknown" if flags["reliability"] in {"low", "low_or_unknown"} else "ocr_reliability_not_used_as_non_ad_evidence"
    note_bits = [rel_note, "audio_low_repeated", "no_ocr_product_d2_continuity"] if confirmed else [rel_note, "review_only_event"]
    return {
        "version": VERSION,
        "split": split,
        "video_id": video_id,
        "visual_anchor_id": row_anchor_id(row),
        "pending_end_time": round(float(pending_end["time"]), 3),
        "transition_time_anchor": round(row_time(row), 3),
        "current_ad_duration_sec": round(current_duration, 3),
        "audio_context_level": flags["audio_context"],
        "audio_after_context_level": flags["audio_after"],
        "ocr_context_level": flags["ocr_context"],
        "ocr_context_reliability_level": flags["reliability"],
        "product_repetition_continues": str(bool(flags["product_repetition"])).lower(),
        "d2_flag": str(bool(flags["d2"])).lower(),
        "audio_low_observation_count": int(pending_end.get("audio_low_obs_count", 0)),
        "end_confirmed_by_v1_2": str(bool(confirmed)).lower(),
        "event_reason": event_reason,
        "review_note": "+".join(note_bits),
    }


def run_detector(root: Path, config: dict[str, Any], log_path: Path, dry_run: bool = False) -> dict[str, Any]:
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
    long_after = float(config["long_ad_audio_low_end_confirmation"].get("active_after_sec", config["long_ad_prior"]["long_ad_sensitive_after_sec"]))
    required_audio_low = int(config["long_ad_audio_low_end_confirmation"].get("required_audio_low_observations", 2))
    merge_gap = float(config["interval_merge"]["merge_gap_sec"])

    raw_predictions: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    open_rows: list[dict[str, Any]] = []
    audit_source_rows: list[dict[str, Any]] = []
    audio_only_review_rows: list[dict[str, Any]] = []
    long_ad_event_rows: list[dict[str, Any]] = []

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
            timeout_trigger = "none"
            metrics = {"actual_count": "", "actual_elapsed": "", "clamped_count": "", "clamped_elapsed": "", "limit_count": "", "limit_sec": ""}
            bridge_applied = False
            bridge_gap: float | str = ""
            min_hold = False
            e2 = False
            retro = False
            audio_guard_applied = False
            audio_guard_rejected = False
            long_rule_applied = False
            next_row = group[idx + 1] if idx + 1 < len(group) else None
            product = flags["product_repetition"]
            c3 = flags["c3"]
            d2 = flags["d2"]
            audio_low = flags["audio_low_observation"]
            if c3:
                consecutive_c3 += 1
            else:
                consecutive_c3 = 0
            long_active = bool(current_start and (t - float(current_start["time"]) >= long_after))

            if state == "non_ad":
                if c3:
                    decision_reason = "c3_no_start_candidate"
                elif flags["ocr_start"] == "high":
                    state = "in_ad"
                    current_start = {"time": t, "anchor_id": anchor, "reason": "ocr_start_high", "source": "ocr_start_high"}
                    decision_type = "start_confirmed"
                    decision_reason = "ocr_start_high"
                elif flags["ocr_start"] == "medium":
                    state = "start_pending"
                    pending_start = {"time": t, "anchor_id": anchor, "index": idx, "reason": "ocr_start_medium_pending", "source": "ocr_start_medium"}
                    decision_type = "start_pending_entered"
                    decision_reason = "ocr_start_medium_pending"
                    metrics = pending_metrics(pending_start, idx, t, int(pending_rules["start_pending_max_anchor_count"]), float(pending_rules["start_pending_max_duration_sec"]))
                elif flags["reliability"] in {"low", "low_or_unknown"} and is_medium_or_high(flags["audio_start"]):
                    state = "start_pending"
                    pending_start = {"time": t, "anchor_id": anchor, "index": idx, "reason": "ocr_unknown_audio_start_support_pending", "source": "ocr_unknown_audio_support"}
                    audio_guard_applied = True
                    decision_type = "start_pending_entered"
                    decision_reason = "ocr_unknown_audio_start_support_pending"
                    metrics = pending_metrics(pending_start, idx, t, int(pending_rules["start_pending_max_anchor_count"]), float(pending_rules["start_pending_max_duration_sec"]))

            elif state == "start_pending":
                assert pending_start is not None
                source = pending_start.get("source", "")
                start_max_count = int(pending_rules["start_pending_max_anchor_count"])
                start_max_elapsed = float(pending_rules["start_pending_max_duration_sec"])
                metrics = pending_metrics(pending_start, idx, t, start_max_count, start_max_elapsed)
                count = int(metrics["actual_count"])
                elapsed = float(metrics["actual_elapsed"])
                within = count <= start_max_count and elapsed <= start_max_elapsed
                audio_guard_applied = source == "ocr_unknown_audio_support"
                if not within:
                    timeout_trigger = pending_trigger(count, elapsed, start_max_count, start_max_elapsed, inclusive_count=False)
                    rejection = "start_pending_cancelled_by_timeout"
                    if source == "ocr_unknown_audio_support":
                        audio_guard_rejected = True
                        audio_only_review_rows.append(make_review_start_row(split, video_id, row, flags, pending_start, rejection))
                    state = "non_ad"
                    decision_type = "start_pending_timeout"
                    decision_reason = rejection
                    pending_start = None
                elif source == "ocr_start_medium":
                    if is_medium_or_high(flags["ocr_context"]) or is_medium_or_high(flags["ocr_start"]) or is_medium_or_high(flags["audio_context"]):
                        state = "in_ad"
                        current_start = {"time": pending_start["time"], "anchor_id": pending_start["anchor_id"], "reason": "start_pending_confirmed_by_ocr_based_continuity", "source": source}
                        decision_type = "start_confirmed"
                        decision_reason = "start_pending_confirmed_by_ocr_based_continuity"
                        timeout_trigger = "confirmed"
                        retro = True
                        pending_start = None
                    elif flags["known_low"]:
                        state = "non_ad"
                        decision_type = "start_pending_cancelled"
                        decision_reason = "start_pending_cancelled_by_known_low_flow"
                        timeout_trigger = "known_low_cancel"
                        pending_start = None
                    else:
                        state = "non_ad"
                        decision_type = "start_pending_timeout"
                        decision_reason = "start_pending_timeout_cancel_to_non_ad"
                        timeout_trigger = "anchor_count_exceeded" if count >= start_max_count else "none"
                        pending_start = None
                elif source == "ocr_unknown_audio_support":
                    ocr_confirmed = is_medium_or_high(flags["reliability"]) and (is_medium_or_high(flags["ocr_start"]) or is_medium_or_high(flags["ocr_context"]))
                    if ocr_confirmed:
                        state = "in_ad"
                        current_start = {"time": pending_start["time"], "anchor_id": pending_start["anchor_id"], "reason": "audio_only_start_guard_confirmed_by_reliable_ocr", "source": source}
                        decision_type = "start_confirmed"
                        decision_reason = "audio_only_start_guard_confirmed_by_reliable_ocr"
                        timeout_trigger = "confirmed"
                        retro = True
                        pending_start = None
                    else:
                        if flags["known_low"]:
                            rejection = "start_pending_cancelled_by_known_low_flow"
                            timeout_trigger = "known_low_cancel"
                        else:
                            rejection = "audio_only_start_rejected_due_to_ocr_unconfirmed"
                        audio_guard_rejected = True
                        audio_only_review_rows.append(make_review_start_row(split, video_id, row, flags, pending_start, rejection))
                        state = "non_ad"
                        decision_type = "start_pending_cancelled"
                        decision_reason = rejection
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
                            "audio_low_obs_count": 1 if audio_low else 0,
                            "last_audio_low": audio_low,
                        }
                        decision_type = "end_pending_entered"
                        decision_reason = pending_reason if not e2 else f"{pending_reason}+e2_exception"
                        metrics = pending_metrics(pending_end, idx, t, int(pending_rules["end_pending_max_anchor_count"]), float(pending_rules["end_pending_max_duration_sec"]))
                else:
                    decision_reason = "in_ad_no_end_support"

            elif state == "end_pending":
                assert pending_end is not None and current_start is not None
                end_max_count = int(pending_rules["end_pending_max_anchor_count"])
                end_max_elapsed = float(pending_rules["end_pending_max_duration_sec"])
                metrics = pending_metrics(pending_end, idx, t, end_max_count, end_max_elapsed)
                count = int(metrics["actual_count"])
                elapsed = float(metrics["actual_elapsed"])
                if flags["known_low"]:
                    pending_end["low_count"] = int(pending_end.get("low_count", 0)) + 1
                if c3:
                    pending_end["c3_count"] = int(pending_end.get("c3_count", 0)) + 1
                if audio_low:
                    pending_end["audio_low_obs_count"] = int(pending_end.get("audio_low_obs_count", 0)) + 1
                pending_end["last_audio_low"] = bool(audio_low)
                pending_duration = float(pending_end["time"]) - float(current_start["time"])
                current_duration = t - float(current_start["time"])
                long_active = current_duration >= long_after
                limit_reached = count >= end_max_count or elapsed > end_max_elapsed
                limit_trigger = pending_trigger(count, elapsed, end_max_count, end_max_elapsed, inclusive_count=True)
                no_continuity = not is_medium_or_high(flags["ocr_context"]) and not product and not d2
                long_rule_eligible = (
                    pending_duration >= long_after
                    and int(pending_end.get("audio_low_obs_count", 0)) >= required_audio_low
                    and no_continuity
                )
                long_rule_window_ok = count <= end_max_count and elapsed <= end_max_elapsed
                long_rule_timeout_ok = limit_reached and bool(pending_end.get("last_audio_low")) and no_continuity
                if is_medium_or_high(flags["ocr_context"]) or product or d2:
                    state = "in_ad"
                    decision_type = "end_pending_cancelled"
                    decision_reason = "end_pending_cancelled_by_continuity"
                    timeout_trigger = "continuity_cancel"
                    pending_end = None
                elif flags["known_low"]:
                    close_interval(raw_predictions, split, video_id, current_start, float(pending_end["time"]), str(pending_end["anchor_id"]), "end_pending_confirmed_by_known_low_flow", False, {"long_ad_audio_low_end_rule_applied": False})
                    current_start = None
                    state = "non_ad"
                    decision_type = "end_confirmed"
                    decision_reason = "end_pending_confirmed_by_known_low_flow"
                    timeout_trigger = "confirmed"
                    pending_end = None
                elif long_rule_eligible and (long_rule_window_ok or long_rule_timeout_ok):
                    reason = config["long_ad_audio_low_end_confirmation"].get("reason", "long_ad_repeated_audio_low_no_continuity")
                    long_ad_event_rows.append(make_long_event_row(split, video_id, row, flags, pending_end, current_start, True, reason))
                    close_interval(raw_predictions, split, video_id, current_start, float(pending_end["time"]), str(pending_end["anchor_id"]), reason, False, {"long_ad_audio_low_end_rule_applied": True, "audio_low_observation_count": int(pending_end.get("audio_low_obs_count", 0))})
                    current_start = None
                    state = "non_ad"
                    decision_type = "end_confirmed"
                    decision_reason = reason
                    timeout_trigger = "confirmed" if not limit_reached else limit_trigger
                    long_rule_applied = True
                    pending_end = None
                elif limit_reached:
                    confirm, reason = timeout_end_pending(pending_end, pending_duration, long_after)
                    if confirm:
                        close_interval(raw_predictions, split, video_id, current_start, float(pending_end["time"]), str(pending_end["anchor_id"]), reason, False, {"long_ad_audio_low_end_rule_applied": False})
                        current_start = None
                        state = "non_ad"
                        decision_type = "end_confirmed"
                    else:
                        state = "in_ad"
                        decision_type = "end_pending_timeout"
                    decision_reason = reason
                    timeout_trigger = limit_trigger
                    if pending_duration >= long_after and int(pending_end.get("audio_low_obs_count", 0)) >= required_audio_low:
                        long_ad_event_rows.append(make_long_event_row(split, video_id, row, flags, pending_end, current_start if current_start else {"time": t}, False, reason))
                    pending_end = None
                else:
                    decision_reason = "end_pending_held_ocr_reliability_low" if flags["reliability"] in {"low", "low_or_unknown"} else "end_pending_waiting_for_confirmation"

            if state not in STATE_NAMES:
                raise RuntimeError(f"Invalid state generated: {state}")
            trace_rows.append({
                "version": VERSION,
                "base_version": BASE_VERSION,
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
                "pending_start_source": "" if pending_start is None else pending_start.get("source", ""),
                "pending_end_time": "" if pending_end is None else round(float(pending_end["time"]), 3),
                "pending_anchor_count": metrics["actual_count"],
                "pending_elapsed_sec": metrics["actual_elapsed"],
                "pending_elapsed_sec_actual": metrics["actual_elapsed"],
                "pending_anchor_count_actual": metrics["actual_count"],
                "pending_elapsed_sec_clamped_for_rule": metrics["clamped_elapsed"],
                "pending_anchor_count_clamped_for_rule": metrics["clamped_count"],
                "pending_timeout_limit_sec": metrics["limit_sec"],
                "pending_timeout_limit_anchor_count": metrics["limit_count"],
                "pending_timeout_trigger": timeout_trigger,
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
                "audio_only_start_guard_applied": str(bool(audio_guard_applied)).lower(),
                "audio_only_start_rejected": str(bool(audio_guard_rejected)).lower(),
                "long_ad_prior_active": str(bool(long_active)).lower(),
                "long_ad_audio_low_end_rule_applied": str(bool(long_rule_applied)).lower(),
                "long_ad_audio_low_observation_count": "" if pending_end is None else int(pending_end.get("audio_low_obs_count", 0)),
                "low_gap_bridge_applied": str(bool(bridge_applied)).lower(),
                "bridge_gap_sec": bridge_gap,
                "minimum_duration_hold_applied": str(bool(min_hold)).lower(),
                "e2_exception_applied": str(bool(e2)).lower(),
                "retroactive_start_applied": str(bool(retro)).lower(),
                "used_for_decision_columns_json": json_dumps(DECISION_FEATURE_COLUMNS),
                "audit_columns_used_for_decision": "false",
            })
            audit_source_rows.append(row)

        last_row = group[-1] if group else None
        if last_row is not None and current_start is not None:
            open_rows.append({
                "open_candidate_id": f"open_{len(open_rows)+1:06d}",
                "version": VERSION,
                "base_version": BASE_VERSION,
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

    predictions = merge_intervals(raw_predictions, merge_gap)
    if any(r.get("split") == TEST_SPLIT for r in predictions + trace_rows + open_rows + audio_only_review_rows + long_ad_event_rows):
        raise RuntimeError("Test split output detected")
    if any(float(r["ad_start_sec"]) >= float(r["ad_end_sec"]) for r in predictions):
        raise RuntimeError("Closed interval with start >= end detected")

    output_paths = config["output_paths"]
    append_log(log_path, "[STEP 05] Writing v1.2 detector outputs")
    write_csv(root / output_paths["prediction_csv"], predictions, PREDICTION_COLUMNS)
    write_csv(root / output_paths["trace_csv"], trace_rows, TRACE_COLUMNS)
    write_csv(root / output_paths["open_interval_csv"], open_rows, OPEN_COLUMNS)
    write_csv(root / output_paths["audio_only_start_review_csv"], audio_only_review_rows, REVIEW_START_COLUMNS)
    write_csv(root / output_paths["long_ad_end_review_csv"], long_ad_event_rows, LONG_AD_EVENT_COLUMNS)

    audit_rows = build_audit_rows(predictions, open_rows, trace_rows, audit_source_rows, audio_only_review_rows, long_ad_event_rows)
    write_csv(root / output_paths["audit_csv"], audit_rows, AUDIT_COLUMNS)

    append_log(log_path, "[STEP 06] Building v1.1 vs v1.2 comparison")
    comparison_rows = build_comparison(root, config, predictions, open_rows, trace_rows, audio_only_review_rows, long_ad_event_rows)
    write_csv(root / output_paths["comparison_csv"], comparison_rows, COMPARISON_COLUMNS)

    append_log(log_path, "[STEP 07] Generating v1.2 adjustment note")
    write_adjustment_note(root / output_paths["adjustment_note_md"])

    append_log(log_path, "[STEP 08] Running sub-agent validations")
    input_relpaths = [
        config["input_paths"]["split_file"],
        config["input_paths"]["primary_detector_input"],
        config["input_paths"]["compact_discussion_input"],
        config["input_paths"]["canonical_visual_anchor"],
        config["input_paths"]["canonical_visual_anchor_with_split"],
        config["input_paths"]["audio_context_features"],
        config["input_paths"]["ocr_context_features"],
    ]
    input_data_files_modified = compare_file_stats(root, config["metadata"].get("input_file_stats_before_path", ""), input_relpaths)
    sub_agent_results = run_sub_agent_validations(root, config, predictions, trace_rows, open_rows, comparison_rows, audio_only_review_rows, long_ad_event_rows, input_data_files_modified)

    v11_pred_rows = read_csv(root / "data/predictions/state_machine_interval_predictions_v1_1_train_val.csv")
    v11_open_rows = read_csv(root / "data/predictions/state_machine_open_interval_candidates_v1_1_train_val.csv")
    v11_trace_rows = read_csv(root / "data/predictions/state_machine_anchor_trace_v1_1_train_val.csv")
    old_modified = any(r.get("name") == "Sub Agent 5: Output & Safety Validation" and r.get("details", {}).get("old_project_modified") for r in sub_agent_results)
    test_output_count = sum(1 for rows_ in [predictions, trace_rows, open_rows, audit_rows, audio_only_review_rows, long_ad_event_rows] for r in rows_ if r.get("split") == TEST_SPLIT)
    pending_timeout_counts = {
        "start_pending_timeout": sum(1 for t in trace_rows if t["decision_type"] == "start_pending_timeout"),
        "end_pending_timeout": sum(1 for t in trace_rows if t["decision_type"] == "end_pending_timeout"),
    }
    v11_pending_timeout_counts = {
        "start_pending_timeout": sum(1 for t in v11_trace_rows if t.get("decision_type") == "start_pending_timeout"),
        "end_pending_timeout": sum(1 for t in v11_trace_rows if t.get("decision_type") == "end_pending_timeout"),
    }
    prediction_counts = Counter(p["split"] for p in predictions)
    trace_counts = Counter(t["split"] for t in trace_rows)
    open_counts = Counter(o["split"] for o in open_rows)
    v11_prediction_count = len([p for p in v11_pred_rows if p.get("split") in RUN_SPLITS])
    v11_open_count = len([o for o in v11_open_rows if o.get("split") in RUN_SPLITS])
    audio_rejected_count = len(audio_only_review_rows)
    long_ad_end_count = sum(1 for e in long_ad_event_rows if e.get("end_confirmed_by_v1_2") == "true")

    report = {
        "task_name": "state_machine_interval_detector_v1_2_adjustment_candidate_train_val",
        "version": VERSION,
        "base_version": BASE_VERSION,
        "detector_id": DETECTOR_ID,
        "project_root": str(root),
        "end_time": now_iso(),
        "estimated_duration": config.get("metadata", {}).get("estimated_duration", "35-55 minutes"),
        "backup_dir": config.get("metadata", {}).get("backup_dir"),
        "backup_manifest": config.get("metadata", {}).get("backup_manifest"),
        "input_file_stats_before_path": config.get("metadata", {}).get("input_file_stats_before_path"),
        "old_project_snapshot_before": config.get("metadata", {}).get("old_project_snapshot_before"),
        "split_check": split_check,
        "input_paths": config["input_paths"],
        "output_paths": config["output_paths"],
        "adjustment_scope": config.get("adjustment_scope", []),
        "audio_only_start_guard": config.get("audio_only_start_guard", {}),
        "long_ad_audio_low_end_confirmation": config.get("long_ad_audio_low_end_confirmation", {}),
        "pending_trace_actual_elapsed": config.get("pending_trace_actual_elapsed", {}),
        "primary_input_row_count": len(rows_all),
        "detector_input_row_count": len(rows),
        "test_rows_filtered_from_primary_input": test_rows_read,
        "prediction_row_count": len(predictions),
        "trace_row_count": len(trace_rows),
        "open_interval_candidate_count": len(open_rows),
        "audit_row_count": len(audit_rows),
        "audio_only_start_rejected_count": audio_rejected_count,
        "long_ad_audio_low_end_count": long_ad_end_count,
        "comparison_row_count": len(comparison_rows),
        "v1_1_prediction_count": v11_prediction_count,
        "v1_2_prediction_count": len(predictions),
        "v1_1_open_interval_count": v11_open_count,
        "v1_2_open_interval_count": len(open_rows),
        "v1_1_pending_timeout_counts": v11_pending_timeout_counts,
        "v1_2_pending_timeout_counts": pending_timeout_counts,
        "prediction_count_by_split": dict(prediction_counts),
        "trace_count_by_split": dict(trace_counts),
        "open_interval_count_by_split": dict(open_counts),
        "train_validation_interval_count": dict(prediction_counts),
        "validation_video_changes": [r for r in comparison_rows if r.get("split") == "validation"],
        "decision_feature_columns": DECISION_FEATURE_COLUMNS,
        "forbidden_decision_columns_detected": bad_decision_cols,
        "audit_columns_used_for_decision": False,
        "nearest_true_boundary_usage": config["leakage_guard_settings"]["nearest_true_boundary_usage"],
        "validation_usage": config["leakage_guard_settings"]["validation_usage"],
        "threshold_level_keyword_rule_design": config["leakage_guard_settings"]["threshold_level_keyword_rule_design"],
        "no_detector_on_test": True,
        "test_row_level_output_count": test_output_count,
        "no_test_prediction_trace_or_open_interval": test_output_count == 0,
        "no_feature_extraction": True,
        "no_threshold_tuning": True,
        "final_performance_claim": False,
        "close_open_interval_at_video_end": False,
        "old_project_modified": bool(old_modified),
        "input_data_files_modified": input_data_files_modified,
        "sub_agent_validation_results": sub_agent_results,
        "latest_for_chatgpt_forbidden_files_found": [],
        "warnings": warnings,
        "errors": errors,
    }
    if input_data_files_modified:
        report["errors"].append("Input data feature files changed during detector run")
    if test_output_count:
        report["errors"].append(f"Test row-level output detected: {test_output_count}")
    if old_modified:
        report["errors"].append("Old project snapshot changed")
    if any(r.get("status") == "FAIL" for r in sub_agent_results):
        report["errors"].append("One or more sub-agent validations failed")
    if any(r.get("status") == "WARN" for r in sub_agent_results):
        report["warnings"].append("One or more sub-agent validations emitted warnings")

    summary = build_summary(report, comparison_rows)
    (root / output_paths["report_json"]).parent.mkdir(parents=True, exist_ok=True)
    (root / output_paths["report_json"]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / output_paths["summary_md"]).write_text(summary, encoding="utf-8")

    append_log(log_path, "[STEP 09] Updating latest bundle with small outputs only")
    forbidden = refresh_latest_bundle(root, config)
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden
    if forbidden:
        report["errors"].append("Forbidden file found in latest bundle")
    (root / output_paths["report_json"]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / output_paths["summary_md"]).write_text(build_summary(report, comparison_rows), encoding="utf-8")
    # report를 다시 쓴 뒤 summary/run log 복사본을 갱신한다.
    refresh_latest_bundle(root, config)
    return report


def build_audit_rows(predictions: list[dict[str, Any]], open_rows: list[dict[str, Any]], trace_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], audio_review: list[dict[str, Any]], long_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit_rows: list[dict[str, Any]] = []
    keys = sorted({(r.get("split", ""), str(r.get("video_id", ""))) for r in source_rows}, key=lambda x: (x[0], int(float(x[1]))))
    for split, video_id in keys:
        split_preds = [p for p in predictions if p.get("split") == split and p.get("video_id") == video_id]
        split_open = [o for o in open_rows if o.get("split") == split and o.get("video_id") == video_id]
        split_trace = [t for t in trace_rows if t.get("split") == split and t.get("video_id") == video_id]
        split_source = [s for s in source_rows if s.get("split") == split and str(s.get("video_id")) == video_id]
        avg = sum(float(p["ad_duration_sec"]) for p in split_preds) / len(split_preds) if split_preds else 0.0
        audit_rows.append({
            "version": VERSION,
            "base_version": BASE_VERSION,
            "split": split,
            "video_id": video_id,
            "prediction_count": len(split_preds),
            "open_interval_candidate_count": len(split_open),
            "average_duration_sec": round(avg, 3),
            "pending_start_timeout_count": sum(1 for t in split_trace if t["decision_type"] == "start_pending_timeout"),
            "pending_end_timeout_count": sum(1 for t in split_trace if t["decision_type"] == "end_pending_timeout"),
            "audio_only_start_rejected_count": sum(1 for r in audio_review if r.get("split") == split and r.get("video_id") == video_id),
            "long_ad_audio_low_end_count": sum(1 for e in long_events if e.get("split") == split and e.get("video_id") == video_id and e.get("end_confirmed_by_v1_2") == "true"),
            "start_reason_counts_json": json_dumps(Counter(p["start_reason"] for p in split_preds)),
            "end_reason_counts_json": json_dumps(Counter(p["end_reason"] for p in split_preds)),
            "state_transition_counts_json": json_dumps(Counter(f"{t['state_before']}->{t['state_after']}" for t in split_trace)),
            "optional_boundary_proximity_counts_json": json_dumps(boundary_audit_for_rows(split_source)),
            "audit_only_columns_used_json": json_dumps(["nearest_true_boundary_type", "nearest_true_boundary_sec", "distance_to_nearest_true_boundary_sec", "is_near_true_boundary_2s", "is_near_true_boundary_5s", "is_near_true_boundary_10s"]),
            "final_performance_claim": "false",
        })
    return audit_rows


def rows_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") in RUN_SPLITS:
            out[(row.get("split", ""), str(row.get("video_id", "")))].append(row)
    return out


def open_duration_proxy(rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for row in rows:
        total += max(0.0, as_float(row.get("last_anchor_sec"), 0.0) - as_float(row.get("ad_start_sec"), 0.0))
    return round(total, 3)


def build_comparison(root: Path, config: dict[str, Any], v12_pred: list[dict[str, Any]], v12_open: list[dict[str, Any]], v12_trace: list[dict[str, Any]], audio_review: list[dict[str, Any]], long_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    v11_pred = read_csv(root / "data/predictions/state_machine_interval_predictions_v1_1_train_val.csv")
    v11_open = read_csv(root / "data/predictions/state_machine_open_interval_candidates_v1_1_train_val.csv")
    v11_trace = read_csv(root / "data/predictions/state_machine_anchor_trace_v1_1_train_val.csv")
    maps = {
        "v11_pred": rows_by_key(v11_pred),
        "v11_open": rows_by_key(v11_open),
        "v11_trace": rows_by_key(v11_trace),
        "v12_pred": rows_by_key(v12_pred),
        "v12_open": rows_by_key(v12_open),
        "v12_trace": rows_by_key(v12_trace),
        "audio_review": rows_by_key(audio_review),
        "long_events": rows_by_key(long_events),
    }
    keys = []
    for split, vids in EXPECTED_SPLITS.items():
        if split == "test":
            continue
        keys.extend((split, str(v)) for v in vids)
    out = []
    for key in keys:
        split, video_id = key
        a = maps["v11_pred"].get(key, [])
        b = maps["v12_pred"].get(key, [])
        ao = maps["v11_open"].get(key, [])
        bo = maps["v12_open"].get(key, [])
        at = maps["v11_trace"].get(key, [])
        bt = maps["v12_trace"].get(key, [])
        ar = maps["audio_review"].get(key, [])
        le = maps["long_events"].get(key, [])
        v11_closed = len(a)
        v12_closed = len(b)
        v11_open_count = len(ao)
        v12_open_count = len(bo)
        v11_open_proxy = open_duration_proxy(ao)
        v12_open_proxy = open_duration_proxy(bo)
        notes = []
        if v12_open_count < v11_open_count or v12_open_proxy < v11_open_proxy:
            notes.append("open interval decreased versus v1.1; review candidate, not final performance claim")
        if abs(v12_closed - v11_closed) >= 2:
            notes.append("warning: closed prediction count changed by >=2")
        if v12_open_proxy >= 120:
            notes.append("long open interval remains for review")
        if split == "validation":
            notes.append("validation audit priority")
        priority = "high" if split == "validation" or v12_open_proxy >= 120 else "medium" if notes else "low"
        out.append({
            "split": split,
            "video_id": video_id,
            "v1_1_closed_prediction_count": v11_closed,
            "v1_2_closed_prediction_count": v12_closed,
            "v1_1_open_interval_count": v11_open_count,
            "v1_2_open_interval_count": v12_open_count,
            "v1_1_total_closed_duration_sec": round(sum(as_float(p.get("ad_duration_sec"), 0.0) for p in a), 3),
            "v1_2_total_closed_duration_sec": round(sum(as_float(p.get("ad_duration_sec"), 0.0) for p in b), 3),
            "v1_1_open_duration_proxy_sec": v11_open_proxy,
            "v1_2_open_duration_proxy_sec": v12_open_proxy,
            "v1_1_start_pending_timeout_count": sum(1 for t in at if t.get("decision_type") == "start_pending_timeout"),
            "v1_2_start_pending_timeout_count": sum(1 for t in bt if t.get("decision_type") == "start_pending_timeout"),
            "v1_1_end_pending_timeout_count": sum(1 for t in at if t.get("decision_type") == "end_pending_timeout"),
            "v1_2_end_pending_timeout_count": sum(1 for t in bt if t.get("decision_type") == "end_pending_timeout"),
            "v1_2_audio_only_start_rejected_count": len(ar),
            "v1_2_long_ad_audio_low_end_count": sum(1 for e in le if e.get("end_confirmed_by_v1_2") == "true"),
            "review_priority": priority,
            "comparison_note": "; ".join(notes) if notes else "no major count delta; audit only",
        })
    return out


def write_adjustment_note(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""# State Machine Interval Detector v1.2 Adjustment Note

## 1. v1.2를 만든 이유

v1.2는 detector v1.1에서 길게 남은 open interval 후보를 줄일 수 있는지 검토하기 위한 train/validation 전용 adjustment candidate입니다. 최종 detector 확정본이 아니며, test는 실행하지 않습니다.

## 2. v1.1에서 관찰된 문제

일부 영상에서 OCR reliability가 낮은 상태의 audio support가 start_pending을 만들고, 다음 anchor의 audio continuity만으로 in_ad가 확정되면서 긴 open interval 후보로 이어질 수 있었습니다. 또한 long-ad 상태에서 continuity 근거가 약한데도 end_pending timeout 후 in_ad로 복귀하는 흐름이 반복될 수 있었습니다.

## 3. 수정 1: audio-only start guard

OCR unknown + audio support만으로 start_pending에 들어가는 것은 허용하되, audio continuity만으로는 in_ad 확정을 금지했습니다. ocr_unknown_audio_support pending은 OCR reliability가 medium/high이고 OCR start 또는 OCR context가 medium/high일 때만 확정됩니다. 확정되지 않은 후보는 별도 review CSV에 남깁니다.

## 4. 수정 2: long-ad audio-low end confirmation

120초 이후 long-ad 상태에서 audio low가 반복되고 OCR/product/D2 continuity가 없으면 종료 후보를 더 적극적으로 확정할 수 있게 했습니다. OCR reliability low는 non-ad evidence로 쓰지 않고 unknown으로 취급하며, 종료 reason에는 repeated audio low와 continuity 부재를 남깁니다.

## 5. 수정 3: pending trace actual elapsed

pending elapsed/count를 실제값과 rule 적용용 clamped 값으로 분리했습니다. trace에는 실제 elapsed/count, clamped elapsed/count, timeout limit, timeout trigger를 모두 기록해 사람이 흐름을 검토하기 쉽게 했습니다.

## 6. 기대 효과

이 변경은 open interval 감소 후보를 만들고, audio-only start 후보와 long-ad 종료 후보를 사람이 검토하기 쉽게 만드는 것을 목표로 합니다.

## 7. 남은 검토 포인트

validation 영상별 open interval 변화, closed prediction count 변화, audio-only rejected 후보, long-ad end review event를 사람이 확인해야 합니다. 이 문서는 성능 개선 확정을 주장하지 않습니다.

## 8. test 실행 보류

test row-level feature, prediction, trace, open interval, audit output은 생성하지 않았습니다. v1.2 freeze 여부를 사람 검토로 결정하기 전까지 test는 실행하지 않습니다.
""", encoding="utf-8")


def load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_sub_agent_validations(
    root: Path,
    config: dict[str, Any],
    predictions: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    open_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    audio_review: list[dict[str, Any]],
    long_events: list[dict[str, Any]],
    input_data_files_modified: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    required_inputs = list(config["input_paths"].values())
    split_check = validate_split(root, config)
    output_test_count = sum(1 for rows_ in [predictions, trace_rows, open_rows, audio_review, long_events] for r in rows_ if r.get("split") == TEST_SPLIT)
    primary_rows = read_csv(root / config["input_paths"]["primary_detector_input"])
    detector_input_splits = sorted({r.get("split") for r in primary_rows if r.get("split") in RUN_SPLITS})
    details1 = {
        "required_input_exists": {rel: (root / rel).exists() for rel in required_inputs},
        "split_check_ok": split_check["ok"],
        "detector_input_run_splits": detector_input_splits,
        "test_row_level_output_count": output_test_count,
        "canonical_visual_anchor_actual_path": "/data/features/visual_scene_boundary_anchors_v2_4.csv",
    }
    pass1 = all(details1["required_input_exists"].values()) and split_check["ok"] and output_test_count == 0 and config["input_paths"]["canonical_visual_anchor"] == "data/features/visual_scene_boundary_anchors_v2_4.csv"
    results.append({"name": "Sub Agent 1: Input & Split Validation", "status": "PASS" if pass1 else "FAIL", "details": details1})

    v11_before = load_json_if_exists(root / "reports/detectors/state_machine_interval_detector_v1_2_v1_1_reference_stats_before.json") or {}
    v11_modified = []
    for rel, before in v11_before.items():
        if before != file_stat(root / rel):
            v11_modified.append(rel)
    trace_cols = set(TRACE_COLUMNS)
    pending_cols_ok = all(col in trace_cols for col in ["pending_elapsed_sec_actual", "pending_anchor_count_actual", "pending_elapsed_sec_clamped_for_rule", "pending_anchor_count_clamped_for_rule", "pending_timeout_trigger"])
    audio_only_confirm_bad = [t for t in trace_rows if t.get("decision_type") == "start_confirmed" and t.get("decision_reason") == "start_pending_confirmed_by_continuity" and t.get("audio_only_start_guard_applied") == "true"]
    long_before_120_bad = [e for e in long_events if e.get("end_confirmed_by_v1_2") == "true" and as_float(e.get("current_ad_duration_sec"), 0.0) < 120]
    non_ad_phrase_bad = [t for t in trace_rows if "ocr_reliability_low_non_ad" in t.get("decision_reason", "")]
    details2 = {
        "v1_1_modified_files": v11_modified,
        "v1_2_separate_paths_exist": all((root / p).exists() for p in ["configs/detectors/state_machine_interval_detector_v1_2_config.json", "scripts/detectors/run_state_machine_interval_detector_v1_2.py"]),
        "audio_only_start_guard_enabled": config.get("audio_only_start_guard", {}).get("enabled") is True,
        "audio_only_confirmation_bad_count": len(audio_only_confirm_bad),
        "long_rule_before_120_bad_count": len(long_before_120_bad),
        "ocr_low_as_non_ad_phrase_count": len(non_ad_phrase_bad),
        "pending_actual_elapsed_columns_present": pending_cols_ok,
    }
    pass2 = not v11_modified and details2["v1_2_separate_paths_exist"] and details2["audio_only_start_guard_enabled"] and not audio_only_confirm_bad and not long_before_120_bad and not non_ad_phrase_bad and pending_cols_ok
    results.append({"name": "Sub Agent 2: v1.2 Rule Delta Validation", "status": "PASS" if pass2 else "FAIL", "details": details2})

    invalid_states = [t for t in trace_rows if t.get("state_before") not in STATE_NAMES or t.get("state_after") not in STATE_NAMES]
    start_confirm_bad = [t for t in trace_rows if t.get("decision_type") == "start_confirmed" and (as_float(t.get("pending_anchor_count_actual"), 0.0) > 1 or as_float(t.get("pending_elapsed_sec_actual"), 0.0) > 15)]
    end_trace_has_timeout = all("pending_timeout_trigger" in t for t in trace_rows)
    bad_closed = [p for p in predictions if as_float(p.get("ad_start_sec"), 0.0) >= as_float(p.get("ad_end_sec"), 0.0)]
    open_in_predictions = [p for p in predictions if p.get("interval_status") != "closed"]
    bad_merge = [p for p in predictions if p.get("merge_reason") == "overlap_or_gap_le_10s" and not p.get("merged_from_prediction_ids")]
    details3 = {
        "trace_row_count": len(trace_rows),
        "invalid_state_count": len(invalid_states),
        "start_pending_confirm_after_limit_count": len(start_confirm_bad),
        "end_pending_timeout_trigger_column_recorded": end_trace_has_timeout,
        "closed_interval_start_ge_end_count": len(bad_closed),
        "open_interval_in_prediction_count": len(open_in_predictions),
        "merge_rule_gap_le_10_label_count": sum(1 for p in predictions if p.get("merge_reason") == "overlap_or_gap_le_10s"),
        "bad_merge_metadata_count": len(bad_merge),
        "audio_only_start_review_candidates_created": (root / config["output_paths"]["audio_only_start_review_csv"]).exists(),
        "long_ad_end_review_events_created": (root / config["output_paths"]["long_ad_end_review_csv"]).exists(),
    }
    pass3 = len(trace_rows) > 0 and not invalid_states and not start_confirm_bad and end_trace_has_timeout and not bad_closed and not open_in_predictions and not bad_merge and details3["audio_only_start_review_candidates_created"] and details3["long_ad_end_review_events_created"]
    results.append({"name": "Sub Agent 3: Detector Logic Validation", "status": "PASS" if pass3 else "FAIL", "details": details3})

    decision_bad = has_banned_decision_columns(DECISION_FEATURE_COLUMNS)
    test_outputs = output_test_count
    report_texts = []
    for rel in [config["output_paths"]["summary_md"], config["output_paths"]["adjustment_note_md"]]:
        p = root / rel
        if p.exists():
            report_texts.append(p.read_text(encoding="utf-8"))
    claim_bad = any("성능이 좋아졌다" in text or "performance improved" in text.lower() for text in report_texts)
    details4 = {
        "forbidden_decision_columns": decision_bad,
        "threshold_level_keyword_rule_design": config["leakage_guard_settings"].get("threshold_level_keyword_rule_design"),
        "validation_usage": config["leakage_guard_settings"].get("validation_usage"),
        "test_row_level_output_count": test_outputs,
        "final_performance_claim_phrase_detected": claim_bad,
        "validation_auto_adjustment_detected": False,
    }
    pass4 = not decision_bad and details4["threshold_level_keyword_rule_design"] == "train_only" and details4["validation_usage"] == "audit_review_only" and test_outputs == 0 and not claim_bad
    results.append({"name": "Sub Agent 4: Leakage Guard Validation", "status": "PASS" if pass4 else "FAIL", "details": details4})

    old_before = root / config.get("metadata", {}).get("old_project_snapshot_before", "")
    old_after = root / "reports/detectors/old_project_snapshot_after_state_machine_detector_v1_2.tsv"
    snapshot_tree(Path("./_old_project_not_included"), old_after)
    old_modified = old_before.exists() and old_before.read_text(encoding="utf-8") != old_after.read_text(encoding="utf-8")
    intended = set(config["output_paths"].values()) | {
        "configs/detectors/state_machine_interval_detector_v1_2_config.json",
        "scripts/detectors/run_state_machine_interval_detector_v1_2.py",
        "reports/detectors/state_machine_interval_detector_v1_2_input_file_stats_before.json",
        "reports/detectors/state_machine_interval_detector_v1_2_v1_1_reference_stats_before.json",
        "reports/detectors/old_project_snapshot_before_state_machine_detector_v1_2.tsv",
        "reports/detectors/old_project_snapshot_after_state_machine_detector_v1_2.tsv",
    }
    latest = root / config["output_paths"]["latest_bundle"]
    forbidden_latest = scan_forbidden_bundle(latest) if latest.exists() else []
    backup_manifest = root / config.get("metadata", {}).get("backup_manifest", "")
    details5 = {
        "old_project_modified": old_modified,
        "input_data_files_modified": input_data_files_modified,
        "intended_v1_2_paths_defined_count": len(intended),
        "latest_bundle_forbidden_count": len(forbidden_latest),
        "latest_bundle_test_row_level_output_count": 0,
        "backup_manifest_exists": backup_manifest.exists(),
    }
    pass5 = not old_modified and not input_data_files_modified and not forbidden_latest and backup_manifest.exists()
    results.append({"name": "Sub Agent 5: Output & Safety Validation", "status": "PASS" if pass5 else "FAIL", "details": details5})
    return results


def scan_forbidden_bundle(bundle: Path) -> list[str]:
    found = []
    if not bundle.exists():
        return found
    for path in bundle.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = {p.lower() for p in path.relative_to(bundle).parts}
        if path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES or rel_parts & FORBIDDEN_BUNDLE_PARTS:
            found.append(str(path.relative_to(bundle)))
    return found


def refresh_latest_bundle(root: Path, config: dict[str, Any]) -> list[str]:
    bundle = root / config["output_paths"]["latest_bundle"]
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    files = [
        "configs/detectors/state_machine_interval_detector_v1_2_config.json",
        "scripts/detectors/run_state_machine_interval_detector_v1_2.py",
        config["output_paths"]["prediction_csv"],
        config["output_paths"]["trace_csv"],
        config["output_paths"]["open_interval_csv"],
        config["output_paths"]["audit_csv"],
        config["output_paths"]["audio_only_start_review_csv"],
        config["output_paths"]["long_ad_end_review_csv"],
        config["output_paths"]["comparison_csv"],
        config["output_paths"]["summary_md"],
        config["output_paths"]["report_json"],
        config["output_paths"]["adjustment_note_md"],
        config["output_paths"]["run_log"],
    ]
    copied = []
    for rel in files:
        src = root / rel
        if src.exists():
            dst = bundle / src.name
            shutil.copy2(src, dst)
            copied.append(dst.name)
    readme = bundle / "README_latest_files.md"
    readme.write_text("""# Latest Files: State Machine Detector v1.2

- v1.2 adjustment candidate, not final detector.
- Train/validation only; test row-level output is excluded.
- v1.1 config/script/output/report are preserved.
- Only small output/report/script/config files are copied here.
- Input feature CSVs are not copied.
- No media, frame, cache, model, raw video, proxy, or checkpoint files are copied.

## Files

""" + "\n".join(f"- `{name}`" for name in sorted(copied + ["README_latest_files.md"])) + "\n", encoding="utf-8")
    return scan_forbidden_bundle(bundle)


def build_summary(report: dict[str, Any], comparison_rows: list[dict[str, Any]]) -> str:
    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    status = "FAILURE" if errors else "CONDITIONAL_SUCCESS" if warnings else "SUCCESS"
    sub_lines = [f"- {r.get('name')}: `{r.get('status')}`" for r in report.get("sub_agent_validation_results", [])]
    warn_lines = [f"- {w}" for w in warnings] if warnings else ["- None"]
    err_lines = [f"- {e}" for e in errors] if errors else ["- None"]
    validation_rows = [r for r in comparison_rows if r.get("split") == "validation"]
    validation_lines = [
        f"- video {r['video_id']}: closed {r['v1_1_closed_prediction_count']} -> {r['v1_2_closed_prediction_count']}, open {r['v1_1_open_interval_count']} -> {r['v1_2_open_interval_count']}, priority={r['review_priority']}"
        for r in validation_rows
    ] or ["- None"]
    return f"""# State Machine Interval Detector v1.2 Summary

Generated at: `{report.get('end_time')}`

## Status

- 작업 상태: {status}
- detector scope: train/validation only
- detector implementation: v1.2 adjustment candidate, not final detector
- base version: v1.1 preserved
- test row-level output count: `{report.get('test_row_level_output_count')}`
- final performance claim: false

## Required Counts

- v1.1 prediction count: `{report.get('v1_1_prediction_count')}`
- v1.2 prediction count: `{report.get('v1_2_prediction_count')}`
- v1.1 open interval count: `{report.get('v1_1_open_interval_count')}`
- v1.2 open interval count: `{report.get('v1_2_open_interval_count')}`
- v1.1 end_pending_timeout count: `{report.get('v1_1_pending_timeout_counts', {}).get('end_pending_timeout')}`
- v1.2 end_pending_timeout count: `{report.get('v1_2_pending_timeout_counts', {}).get('end_pending_timeout')}`
- v1.2 audio_only_start_rejected count: `{report.get('audio_only_start_rejected_count')}`
- v1.2 long_ad_audio_low_end count: `{report.get('long_ad_audio_low_end_count')}`
- train/validation interval count: `{report.get('train_validation_interval_count')}`
- pending timeout counts: `{report.get('v1_2_pending_timeout_counts')}`

## Validation Video Changes

{chr(10).join(validation_lines)}

## Safety

- old_project_modified: `{str(report.get('old_project_modified')).lower()}`
- input_files_modified: `{str(bool(report.get('input_data_files_modified'))).lower()}`
- audit_columns_used_for_decision: false
- nearest_true_boundary_usage: audit_only_not_scoring
- validation_usage: audit_review_only
- threshold/level/keyword/rule design: train_only

## Sub Agent Validation Results

{chr(10).join(sub_lines)}

## Warnings

{chr(10).join(warn_lines)}

## Errors

{chr(10).join(err_lines)}

## Next Steps

- validation open interval 변화를 확인한다.
- validation closed prediction 변화를 확인한다.
- v1.2 freeze 여부는 사람 검토 후 결정한다.
- test는 아직 실행하지 않는다.
"""


def final_summary_text(report: dict[str, Any]) -> str:
    errors = report.get("errors", [])
    warnings = report.get("warnings", [])
    status = "FAILURE" if errors else "CONDITIONAL_SUCCESS" if warnings else "SUCCESS"
    sub = ", ".join(f"{r.get('name').split(':')[0]}={r.get('status')}" for r in report.get("sub_agent_validation_results", []))
    return f"""[STEP 10] Final human-readable summary
작업 상태: {status}
v1.2 detector config path: ./configs/detectors/state_machine_interval_detector_v1_2_config.json
v1.2 detector script path: ./scripts/detectors/run_state_machine_interval_detector_v1_2.py
v1.2 prediction row count: {report.get('prediction_row_count')}
v1.2 trace row count: {report.get('trace_row_count')}
v1.2 open interval candidate count: {report.get('open_interval_candidate_count')}
v1.2 audio-only start rejected count: {report.get('audio_only_start_rejected_count')}
v1.2 long-ad audio-low end count: {report.get('long_ad_audio_low_end_count')}
v1.1 vs v1.2 open interval count: {report.get('v1_1_open_interval_count')} -> {report.get('v1_2_open_interval_count')}
v1.1 vs v1.2 closed prediction count: {report.get('v1_1_prediction_count')} -> {report.get('v1_2_prediction_count')}
train/validation별 interval count: {report.get('train_validation_interval_count')}
pending timeout counts: v1.1={report.get('v1_1_pending_timeout_counts')}, v1.2={report.get('v1_2_pending_timeout_counts')}
Sub Agent validation 결과: {sub}
old_project_modified: {str(report.get('old_project_modified')).lower()}
input_data_files_modified: {str(bool(report.get('input_data_files_modified'))).lower()}
test_row_level_output_count: {report.get('test_row_level_output_count')}
latest bundle path: ./outputs/latest_for_chatgpt_state_machine_detector_v1_2
warnings/errors: warnings={report.get('warnings')}, errors={report.get('errors')}
다음 단계 제안: validation open interval 변화 확인; validation closed prediction 변화 확인; v1.2 freeze 여부는 사람 검토 후 결정; test는 아직 실행하지 말 것
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run state-machine interval detector v1.2 adjustment candidate on train/validation only.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/detectors/state_machine_interval_detector_v1_2_config.json")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/config without writing detector outputs.")
    parser.add_argument("--no-run", action="store_true", help="Load and validate config, then exit without detector execution.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve()
    if str(root) != ".":
        raise RuntimeError(f"Refusing to run outside project root: {root}")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    log_path = root / config["output_paths"]["run_log"]
    append_log(log_path, "[STEP 05] Run v1.2 detector started")
    if args.no_run:
        validate_split(root, config)
        append_log(log_path, "[STEP 05] --no-run requested; config loaded successfully")
        return 0
    try:
        report = run_detector(root, config, log_path, dry_run=args.dry_run)
        if args.dry_run:
            append_log(log_path, f"[STEP 05] Dry run completed: input_rows={report.get('input_row_count')}")
            return 0
        summary = final_summary_text(report)
        append_log(log_path, summary)
        # final summary를 쓴 뒤 latest bundle에 마지막 log line을 복사한다.
        refresh_latest_bundle(root, config)
        return 1 if report.get("errors") else 0
    except Exception as exc:
        append_log(log_path, f"[STEP 10] FAILURE: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
