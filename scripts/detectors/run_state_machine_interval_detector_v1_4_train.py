
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train split 기준 규칙 기반 상태 전이 탐지기를 실행한다.

private media, label, OCR 원문, 실행 산출물은 저장소에 포함하지 않는다.
전체 실행에는 ``data/`` 아래 private feature CSV가 필요하며, script/config 연결만 확인할 때는 ``--no-run``을 사용한다.
"""
# v1.4 train-only 조정 후보 탐지기.
# train split row만 처리하며, 신뢰도가 높은 닫힌 구간과 review 후보를 분리한다.

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

VERSION = "v1.4"
BASE_VERSION = "v1.3"
DETECTOR_ID = "state_machine_interval_detector_v1_4_train"
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
    "candidate_time_sec", "audio_start_signal_level", "audio_end_signal_level", "audio_context_level",
    "audio_before_context_level", "audio_after_context_level", "audio_score_delta_post_minus_pre",
    "audio_score_delta_pre_minus_post", "audio_pre_10s_ad_like_ratio", "audio_post_10s_ad_like_ratio",
    "ocr_start_signal_level", "ocr_end_signal_level", "ocr_context_level", "ocr_context_reliability_level",
    "ocr_keyword_delta_post_minus_pre", "ocr_score_delta_post_minus_pre", "ocr_score_delta_pre_minus_post",
    "ocr_pre_10s_ad_disclosure_count", "ocr_post_10s_ad_disclosure_count",
    "ocr_pre_10s_purchase_cta_count", "ocr_post_10s_purchase_cta_count",
    "ocr_pre_10s_discount_promo_count", "ocr_post_10s_discount_promo_count",
    "ocr_pre_10s_product_brand_count", "ocr_post_10s_product_brand_count",
    "ocr_pre_10s_ad_disclosure_score", "ocr_post_10s_ad_disclosure_score",
    "ocr_pre_10s_purchase_cta_score", "ocr_post_10s_purchase_cta_score",
    "ocr_pre_10s_product_brand_score", "ocr_post_10s_product_brand_score",
]
FORBIDDEN_BUNDLE_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".wav", ".mp3", ".m4a", ".jpg", ".jpeg", ".png", ".webp", ".parquet", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".onnx"}
FORBIDDEN_BUNDLE_PARTS = {"cache", "frames", "frame_images", "raw_video", "video_proxy", "model_cache", "tmp", "__pycache__"}

PREDICTION_COLUMNS = ["prediction_id", "version", "base_version", "split", "video_id", "ad_start_sec", "ad_end_sec", "ad_duration_sec", "start_anchor_id", "end_anchor_id", "start_reason", "end_reason", "start_confidence_level", "end_confidence_level", "hard_evidence_count", "support_evidence_count", "weak_context_count", "hard_evidence_density_per_60s", "max_weak_span_sec", "interval_status", "merged_from_prediction_ids", "merge_reason", "used_test_row", "decision_feature_columns_json", "audit_columns_used_for_decision", "v1_4_adjustment_flags_json"]
REVIEW_ONLY_COLUMNS = ["candidate_id", "version", "split", "video_id", "start_sec", "end_sec", "duration_sec", "start_anchor_id", "end_anchor_id", "start_reason", "end_reason", "start_confidence_level", "end_confidence_level", "hard_evidence_count", "support_evidence_count", "weak_context_count", "hard_evidence_density_per_60s", "max_weak_span_sec", "review_only_reasons", "interval_status", "review_note"]
TRACE_COLUMNS = ["version", "base_version", "split", "video_id", "visual_anchor_id", "transition_time_anchor", "candidate_time_mmss", "state_before", "state_after", "decision_type", "decision_reason", "pending_start_time", "pending_start_source", "pending_end_time", "pending_elapsed_sec_actual", "pending_anchor_count_actual", "pending_timeout_trigger", "audio_start_signal_level", "audio_end_signal_level", "audio_context_level", "audio_before_context_level", "audio_after_context_level", "audio_score_delta_post_minus_pre", "ocr_start_signal_level", "ocr_end_signal_level", "ocr_context_level", "ocr_context_reliability_level", "product_repetition_continues", "single_product_or_brand_mention", "cta_or_link_context", "audio_rise", "ocr_post_increase", "c3_flag", "d2_flag", "hard_evidence_flag", "support_evidence_flag", "weak_context_flag", "hard_evidence_reasons", "support_evidence_reasons", "weak_context_reasons", "last_hard_evidence_time", "sec_since_last_hard_evidence", "support_only_count", "support_only_elapsed_sec", "start_confirm_pattern", "start_confidence_level", "end_confidence_level", "early_disclosure_guard_applied", "interval_confidence_filter_applied", "moved_to_review_only", "unresolved_long_open_guard_applied", "rejected_long_open_guard_applied", "used_for_decision_columns_json", "audit_columns_used_for_decision"]
OPEN_COLUMNS = ["open_candidate_id", "version", "base_version", "split", "video_id", "ad_start_sec", "last_anchor_sec", "duration_proxy_sec", "video_duration_sec", "duration_ratio", "start_anchor_id", "last_anchor_id", "open_reason", "recent_hard_evidence_count", "hard_evidence_density_per_60s", "interval_status", "used_test_row", "review_note"]
UNRESOLVED_COLUMNS = ["version", "split", "video_id", "unresolved_candidate_id", "start_sec", "last_anchor_sec", "duration_proxy_sec", "video_duration_sec", "duration_ratio", "start_anchor_id", "last_anchor_id", "start_reason", "unresolved_reason", "recent_hard_evidence_count", "recent_window_sec", "hard_evidence_density_per_60s", "moved_to_unresolved_and_reset_state", "review_note"]
REJECTED_COLUMNS = ["version", "split", "video_id", "rejected_candidate_id", "start_sec", "last_anchor_sec", "duration_proxy_sec", "video_duration_sec", "duration_ratio", "start_anchor_id", "last_anchor_id", "start_reason", "rejected_reason", "recent_hard_evidence_count", "recent_window_sec", "hard_evidence_density_per_60s", "moved_to_rejected_and_reset_state", "review_note"]
DISCLOSURE_COLUMNS = ["version", "split", "video_id", "visual_anchor_id", "transition_time_anchor", "candidate_time_mmss", "ocr_start_signal_level", "ocr_context_level", "ocr_context_reliability_level", "audio_start_signal_level", "audio_score_delta_post_minus_pre", "product_repetition_continues", "cta_or_link_context", "early_disclosure_guard_applied", "rejection_reason", "review_note"]
EVIDENCE_TIER_COLUMNS = ["version", "split", "video_id", "visual_anchor_id", "transition_time_anchor", "state_before", "state_after", "hard_evidence_flag", "support_evidence_flag", "weak_context_flag", "hard_evidence_reasons", "support_evidence_reasons", "weak_context_reasons", "start_confirm_pattern", "evidence_tier_summary"]
FRESH_TIMEOUT_COLUMNS = ["version", "split", "video_id", "visual_anchor_id", "transition_time_anchor", "current_ad_start_sec", "last_hard_evidence_time", "sec_since_last_hard_evidence", "support_only_count", "support_only_elapsed_sec", "event_reason", "pending_end_time"]
CONF_FILTER_COLUMNS = ["version", "split", "video_id", "raw_interval_id", "start_sec", "end_sec", "duration_sec", "start_confidence_level", "end_confidence_level", "hard_evidence_count", "support_evidence_count", "weak_context_count", "hard_evidence_density_per_60s", "max_weak_span_sec", "filter_result", "review_only_reasons", "output_id"]
TRAIN_AUDIT_COLUMNS = ["version", "split", "video_id", "high_confidence_prediction_count", "review_only_candidate_count", "open_interval_candidate_count", "unresolved_long_open_count", "rejected_long_open_count", "interval_confidence_filter_event_count", "hard_evidence_trace_count", "support_evidence_trace_count", "weak_context_trace_count", "pending_start_timeout_count", "pending_end_timeout_count", "start_reason_counts_json", "end_reason_counts_json", "state_transition_counts_json", "audit_columns_used_for_decision", "final_performance_claim"]
VIDEO_SUMMARY_COLUMNS = ["video_id", "split", "video_duration_sec", "actual_interval_count", "actual_total_duration_sec", "actual_coverage_ratio", "high_confidence_prediction_count", "high_confidence_total_duration_sec", "high_confidence_coverage_ratio", "review_only_candidate_count", "review_only_total_duration_sec", "review_only_coverage_ratio", "open_interval_count", "open_total_duration_proxy_sec", "open_coverage_ratio", "unresolved_long_open_count", "unresolved_total_duration_proxy_sec", "unresolved_coverage_ratio", "rejected_long_open_count", "rejected_total_duration_proxy_sec", "rejected_coverage_ratio", "total_candidate_duration_sec", "total_candidate_coverage_ratio", "actual_high_conf_overlap_duration_sec", "actual_high_conf_overlap_ratio_of_actual", "actual_high_conf_overlap_ratio_of_prediction", "high_conf_false_positive_duration_sec", "high_conf_false_positive_ratio_of_video", "missed_actual_duration_sec", "missed_actual_ratio_of_actual", "review_candidate_overlap_with_actual_sec", "severity_level_high_confidence", "candidate_overcoverage_level", "review_priority", "diagnosis_note"]
INTERVAL_OVERLAP_COLUMNS = ["video_id", "split", "interval_type", "interval_id", "start_sec", "end_sec", "duration_sec", "best_matching_actual_id", "best_actual_overlap_sec", "best_actual_overlap_ratio_of_interval", "false_positive_duration_sec", "matched_prediction_ids", "missed_actual_duration_sec", "issue_type", "reason", "review_note"]
OPEN_SUMMARY_COLUMNS = ["video_id", "split", "candidate_kind", "candidate_id", "start_sec", "end_proxy_sec", "duration_proxy_sec", "video_duration_sec", "coverage_ratio", "overlap_with_actual_sec", "overlap_ratio_of_candidate", "start_reason", "last_state", "possible_failure_mode", "review_priority", "diagnosis_note"]
TRACE_REASON_COLUMNS = ["scope", "video_id", "category", "key", "count", "ratio_within_category", "note"]
WORST_COLUMNS = ["rank", "video_id", "split", "severity_level_high_confidence", "candidate_overcoverage_level", "review_priority", "actual_coverage_ratio", "high_confidence_coverage_ratio", "review_only_coverage_ratio", "open_coverage_ratio", "unresolved_coverage_ratio", "rejected_coverage_ratio", "total_candidate_coverage_ratio", "high_conf_false_positive_duration_sec", "missed_actual_duration_sec", "diagnosis_note"]
RULE_ISSUE_COLUMNS = ["issue_id", "issue_name", "evidence_from_train", "affected_video_count", "example_video_ids", "severity", "suspected_rule_area", "suggested_direction", "caution", "requires_manual_review"]
COMPARISON_COLUMNS = ["video_id", "split", "actual_coverage_ratio", "v1_3_closed_coverage_ratio", "v1_4_high_confidence_coverage_ratio", "v1_4_review_only_coverage_ratio", "v1_3_open_coverage_ratio", "v1_4_open_coverage_ratio", "v1_3_unresolved_coverage_ratio", "v1_4_unresolved_coverage_ratio", "v1_4_rejected_long_open_coverage_ratio", "v1_3_total_candidate_coverage_ratio", "v1_4_total_candidate_coverage_ratio", "v1_3_closed_prediction_count", "v1_4_high_confidence_prediction_count", "v1_4_review_only_candidate_count", "v1_3_severity_level", "v1_4_severity_level_high_confidence", "v1_4_candidate_overcoverage_level", "got_better_or_worse", "review_priority", "comparison_note"]


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
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except Exception:
        return default


def ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return max(0.0, num / den)


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
            writer.writerow({col: row.get(col, "") for col in columns})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def mmss(sec: float) -> str:
    s = max(0, int(round(sec)))
    return f"{s // 60:02d}:{s % 60:02d}"


def forbidden_match(column: str) -> bool:
    lower = column.lower()
    for pat in BANNED_DECISION_PATTERNS:
        if pat == "gt":
            if lower == "gt" or lower.startswith("gt_") or lower.endswith("_gt") or "_gt_" in lower:
                return True
        elif pat in lower:
            return True
    return False


def banned_decision_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if forbidden_match(c)]


def any_positive(row: dict[str, Any], cols: list[str]) -> tuple[bool, bool]:
    found = False
    for c in cols:
        if c in row:
            found = True
            if as_float(row.get(c)) > 0:
                return True, True
    return False, found


def product_flags(row: dict[str, Any], warnings: list[str]) -> tuple[bool, bool]:
    pre_cols = ["ocr_pre_10s_product_brand_count", "ocr_pre_10s_product_brand_score", "ocr_pre_10s_product_brand_score_raw"]
    post_cols = ["ocr_post_10s_product_brand_count", "ocr_post_10s_product_brand_score", "ocr_post_10s_product_brand_score_raw"]
    pre, pre_found = any_positive(row, pre_cols)
    post, post_found = any_positive(row, post_cols)
    if not (pre_found or post_found) and "product_columns_missing" not in warnings:
        warnings.append("product_columns_missing")
    return pre and post, pre or post


def cta_or_link(row: dict[str, Any], warnings: list[str]) -> bool:
    cols = [
        "ocr_pre_10s_purchase_cta_count", "ocr_post_10s_purchase_cta_count",
        "ocr_pre_10s_purchase_cta_score", "ocr_post_10s_purchase_cta_score",
        "ocr_pre_10s_purchase_cta_score_raw", "ocr_post_10s_purchase_cta_score_raw",
        "ocr_pre_10s_link_count", "ocr_post_10s_link_count", "ocr_pre_10s_more_info_count", "ocr_post_10s_more_info_count",
    ]
    val, found = any_positive(row, cols)
    if not found and "cta_or_link_columns_missing" not in warnings:
        warnings.append("cta_or_link_columns_missing")
    return val


def cue_flags(row: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    audio_start = normalize_level(row.get("audio_start_signal_level"))
    audio_end = normalize_level(row.get("audio_end_signal_level"))
    audio_context = normalize_level(row.get("audio_context_level"))
    audio_before = normalize_level(row.get("audio_before_context_level"))
    audio_after = normalize_level(row.get("audio_after_context_level"))
    ocr_start = normalize_level(row.get("ocr_start_signal_level"))
    ocr_end = normalize_level(row.get("ocr_end_signal_level"))
    ocr_context = normalize_level(row.get("ocr_context_level"))
    reliability = normalize_level(row.get("ocr_context_reliability_level"))
    product_rep, single_product = product_flags(row, warnings)
    cta = cta_or_link(row, warnings)
    disclosure_count, _ = any_positive(row, ["ocr_pre_10s_ad_disclosure_count", "ocr_post_10s_ad_disclosure_count", "ocr_pre_10s_ad_disclosure_score", "ocr_post_10s_ad_disclosure_score", "ocr_pre_10s_ad_disclosure_score_raw", "ocr_post_10s_ad_disclosure_score_raw"])
    discount, _ = any_positive(row, ["ocr_pre_10s_discount_promo_count", "ocr_post_10s_discount_promo_count", "ocr_pre_10s_discount_promo_score", "ocr_post_10s_discount_promo_score", "ocr_pre_10s_discount_promo_score_raw", "ocr_post_10s_discount_promo_score_raw"])
    explicit_disclosure = ocr_start == "high" or disclosure_count
    weak_disclosure = disclosure_count or ocr_start in {"medium", "high"}
    audio_delta = as_float(row.get("audio_score_delta_post_minus_pre"), 0.0)
    audio_pre_ratio = as_float(row.get("audio_pre_10s_ad_like_ratio"), 0.0)
    audio_post_ratio = as_float(row.get("audio_post_10s_ad_like_ratio"), 0.0)
    audio_rise = audio_delta > 0 or (is_medium_or_high(audio_start) and audio_post_ratio > audio_pre_ratio)
    ocr_delta = as_float(row.get("ocr_score_delta_post_minus_pre"), 0.0)
    ocr_keyword_delta = as_float(row.get("ocr_keyword_delta_post_minus_pre"), 0.0)
    ocr_post_increase = ocr_delta > 0 or ocr_keyword_delta > 0
    audio_high = audio_context == "high" or audio_after == "high"
    ocr_high = ocr_context == "high"
    reliability_high = reliability == "high"
    reliability_mh = reliability in {"medium", "high"}
    d2 = audio_context == "low" and ocr_context == "high"
    c3 = audio_context == "high" and reliability == "high" and ocr_context == "low"
    known_low = reliability == "high" and ocr_context == "low" and audio_context == "low"
    hard_reasons = []
    if explicit_disclosure and (product_rep or cta):
        hard_reasons.append("explicit_disclosure_plus_product_or_cta")
    if product_rep and cta:
        hard_reasons.append("product_repetition_plus_cta_or_link")
    if product_rep and audio_high:
        hard_reasons.append("product_repetition_plus_audio_high")
    if ocr_high and audio_high and reliability_high:
        hard_reasons.append("ocr_high_plus_audio_high_plus_reliability_high")
    support_reasons = []
    if is_medium_or_high(ocr_context):
        support_reasons.append("ocr_context_medium_high")
    if is_medium_or_high(audio_context):
        support_reasons.append("audio_context_medium_high")
    if d2:
        support_reasons.append("d2")
    if single_product:
        support_reasons.append("single_product_or_brand_mention")
    if weak_disclosure:
        support_reasons.append("weak_disclosure")
    weak_reasons = []
    if reliability in {"low", "low_or_unknown"}:
        weak_reasons.append("ocr_reliability_low")
    if audio_context == "medium" and not is_medium_or_high(ocr_context) and not product_rep and not cta:
        weak_reasons.append("audio_medium_alone")
    if ocr_context == "medium" and not is_medium_or_high(audio_context) and not product_rep and not cta:
        weak_reasons.append("ocr_medium_alone")
    if discount and not product_rep and not cta:
        weak_reasons.append("discount_or_benefit_word_alone")
    patterns = []
    if explicit_disclosure and (product_rep or cta):
        patterns.append("explicit_disclosure_plus_product_or_cta")
    if product_rep and audio_rise:
        patterns.append("product_repetition_plus_audio_rise")
    if cta and audio_rise:
        patterns.append("cta_or_link_plus_audio_rise")
    if ocr_post_increase and audio_rise and reliability_mh and bool(hard_reasons):
        patterns.append("ocr_post_increase_plus_audio_post_increase_plus_reliability_medium_high")
    end_reasons = []
    if is_medium_or_high(audio_before) and audio_after == "low":
        end_reasons.append("audio_before_medium_high_after_low")
    drop = as_float(row.get("ocr_score_delta_pre_minus_post")) > 0 or as_float(row.get("audio_score_delta_pre_minus_post")) > 0 or audio_after == "low"
    if is_medium_or_high(ocr_end) and drop:
        end_reasons.append("ocr_end_medium_high_with_post_drop")
    if known_low:
        end_reasons.append("known_low_flow")
    return {
        "audio_start": audio_start, "audio_end": audio_end, "audio_context": audio_context, "audio_before": audio_before, "audio_after": audio_after,
        "ocr_start": ocr_start, "ocr_end": ocr_end, "ocr_context": ocr_context, "reliability": reliability,
        "product_repetition": product_rep, "single_product": single_product, "cta_or_link": cta, "explicit_disclosure": explicit_disclosure,
        "weak_disclosure": weak_disclosure, "audio_rise": audio_rise, "ocr_post_increase": ocr_post_increase, "audio_delta": audio_delta,
        "audio_high": audio_high, "ocr_high": ocr_high, "reliability_high": reliability_high, "reliability_mh": reliability_mh,
        "d2": d2, "c3": c3, "known_low": known_low, "hard": bool(hard_reasons), "support": bool(support_reasons), "weak": bool(weak_reasons),
        "hard_reasons": hard_reasons, "support_reasons": support_reasons, "weak_reasons": weak_reasons,
        "start_patterns": patterns, "start_confirm_pattern": patterns[0] if patterns else "", "end_reasons": end_reasons,
    }


def start_confidence(pattern: str) -> str:
    if pattern in {"explicit_disclosure_plus_product_or_cta", "product_repetition_plus_audio_rise", "cta_or_link_plus_audio_rise"}:
        return "high"
    if pattern:
        return "medium"
    return "weak"


def pending_metrics(pending: dict[str, Any] | None, idx: int, t: float) -> tuple[str, str]:
    if not pending:
        return "", ""
    return str(idx - int(pending["index"])), str(round(t - float(pending["time"]), 3))


def timeout_trigger(count: int, elapsed: float, max_count: int, max_elapsed: float) -> str:
    count_hit = count > max_count
    duration_hit = elapsed > max_elapsed
    if count_hit and duration_hit:
        return "duration_and_anchor_count_exceeded"
    if count_hit:
        return "anchor_count_exceeded"
    if duration_hit:
        return "duration_exceeded"
    return "none"


def validate_split(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    rows = read_csv(root / config["input_paths"]["split_file"])
    observed: dict[str, list[int]] = defaultdict(list)
    seeds = set()
    durations = {}
    for row in rows:
        split = row.get("split", "")
        if split in EXPECTED_SPLITS:
            observed[split].append(int(float(row.get("video_id", -1))))
        seeds.add(str(row.get("split_seed", "")))
        durations[str(row.get("video_id", ""))] = as_float(row.get("video_duration_sec"), 0.0)
    observed_sorted = {k: sorted(v) for k, v in observed.items()}
    ok = observed_sorted == EXPECTED_SPLITS and seeds == {EXPECTED_SEED}
    return {"ok": ok, "observed": observed_sorted, "seed_values": sorted(seeds), "video_durations": durations}


def new_interval(start_time: float, anchor: str, reason: str, confidence: str, pattern: str) -> dict[str, Any]:
    return {
        "time": start_time, "anchor_id": anchor, "reason": reason, "start_confidence_level": confidence, "start_confirm_pattern": pattern,
        "hard_evidence_count": 0, "support_evidence_count": 0, "weak_context_count": 0, "last_hard_evidence_time": "", "hard_times": [],
        "weak_span_start": None, "max_weak_span_sec": 0.0, "support_only_span_start": None, "max_support_only_span_sec": 0.0,
    }


def update_interval_evidence(interval: dict[str, Any], cues: dict[str, Any], t: float) -> None:
    if cues["hard"]:
        interval["hard_evidence_count"] += 1
        interval["last_hard_evidence_time"] = t
        interval["hard_times"].append(t)
    if cues["support"]:
        interval["support_evidence_count"] += 1
    if cues["weak"]:
        interval["weak_context_count"] += 1
    if cues["weak"] and not cues["hard"]:
        if interval["weak_span_start"] is None:
            interval["weak_span_start"] = t
        interval["max_weak_span_sec"] = max(interval["max_weak_span_sec"], t - float(interval["weak_span_start"]))
    else:
        interval["weak_span_start"] = None
    if cues["support"] and not cues["hard"]:
        if interval["support_only_span_start"] is None:
            interval["support_only_span_start"] = t
        interval["max_support_only_span_sec"] = max(interval["max_support_only_span_sec"], t - float(interval["support_only_span_start"]))
    else:
        interval["support_only_span_start"] = None


def hard_density(interval: dict[str, Any], end_time: float) -> float:
    dur = max(0.0, end_time - float(interval["time"]))
    if dur <= 0:
        return 0.0
    return float(interval.get("hard_evidence_count", 0)) / (dur / 60.0)


def recent_hard_count(interval: dict[str, Any], t: float, window: float) -> int:
    return sum(1 for ht in interval.get("hard_times", []) if t - float(ht) <= window)


def close_raw(raw: list[dict[str, Any]], split: str, vid: str, current: dict[str, Any] | None, end_time: float, end_anchor: str, end_reason: str, end_conf: str, end_pattern: str) -> None:
    if current is None or end_time <= float(current["time"]):
        return
    dur = end_time - float(current["time"])
    raw_id = f"raw_v1_4_{len(raw)+1:06d}"
    raw.append({
        "raw_interval_id": raw_id, "split": split, "video_id": vid,
        "start_sec": round(float(current["time"]), 3), "end_sec": round(end_time, 3), "duration_sec": round(dur, 3),
        "start_anchor_id": current.get("anchor_id", ""), "end_anchor_id": end_anchor,
        "start_reason": current.get("reason", ""), "end_reason": end_reason,
        "start_confidence_level": current.get("start_confidence_level", "weak"), "end_confidence_level": end_conf,
        "hard_evidence_count": int(current.get("hard_evidence_count", 0)), "support_evidence_count": int(current.get("support_evidence_count", 0)), "weak_context_count": int(current.get("weak_context_count", 0)),
        "hard_evidence_density_per_60s": round(hard_density(current, end_time), 6), "max_weak_span_sec": round(float(current.get("max_weak_span_sec", 0.0)), 3),
        "max_support_only_span_sec": round(float(current.get("max_support_only_span_sec", 0.0)), 3), "start_confirm_pattern": current.get("start_confirm_pattern", ""), "end_confirm_pattern": end_pattern,
    })


def filter_and_merge_intervals(raw: list[dict[str, Any]], merge_gap: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    high_raw = []
    review_rows = []
    event_rows = []
    for item in raw:
        reasons = []
        if item.get("start_confidence_level") not in {"medium", "high"} or item.get("end_confidence_level") not in {"medium", "high"}:
            reasons.append("weak_start_or_end")
        if int(item.get("hard_evidence_count", 0)) < 1:
            reasons.append("no_hard_evidence")
        if as_float(item.get("max_weak_span_sec")) > 30:
            reasons.append("long_weak_span")
        if as_float(item.get("duration_sec")) > 180 and as_float(item.get("hard_evidence_density_per_60s")) < 0.5:
            reasons.append("low_hard_evidence_density")
            reasons.append("overlong_closed_candidate")
        if reasons:
            cid = f"review_v1_4_{len(review_rows)+1:06d}"
            review_rows.append({
                "candidate_id": cid, "version": VERSION, "split": item["split"], "video_id": item["video_id"], "start_sec": item["start_sec"], "end_sec": item["end_sec"], "duration_sec": item["duration_sec"],
                "start_anchor_id": item["start_anchor_id"], "end_anchor_id": item["end_anchor_id"], "start_reason": item["start_reason"], "end_reason": item["end_reason"],
                "start_confidence_level": item["start_confidence_level"], "end_confidence_level": item["end_confidence_level"], "hard_evidence_count": item["hard_evidence_count"], "support_evidence_count": item["support_evidence_count"], "weak_context_count": item["weak_context_count"], "hard_evidence_density_per_60s": item["hard_evidence_density_per_60s"], "max_weak_span_sec": item["max_weak_span_sec"], "review_only_reasons": "+".join(reasons), "interval_status": "review_only_closed_candidate", "review_note": "Closed interval did not pass v1.4 confidence filter; train-only review candidate.",
            })
            out_id = cid
            result = "review_only"
        else:
            high_raw.append(dict(item))
            out_id = item["raw_interval_id"]
            result = "high_confidence"
        event_rows.append({"version": VERSION, "split": item["split"], "video_id": item["video_id"], "raw_interval_id": item["raw_interval_id"], "start_sec": item["start_sec"], "end_sec": item["end_sec"], "duration_sec": item["duration_sec"], "start_confidence_level": item["start_confidence_level"], "end_confidence_level": item["end_confidence_level"], "hard_evidence_count": item["hard_evidence_count"], "support_evidence_count": item["support_evidence_count"], "weak_context_count": item["weak_context_count"], "hard_evidence_density_per_60s": item["hard_evidence_density_per_60s"], "max_weak_span_sec": item["max_weak_span_sec"], "filter_result": result, "review_only_reasons": "+".join(reasons), "output_id": out_id})
    merged: list[dict[str, Any]] = []
    for item in sorted(high_raw, key=lambda r: (int(float(r["video_id"])), float(r["start_sec"]), float(r["end_sec"]))):
        if not merged or merged[-1]["video_id"] != item["video_id"]:
            m = dict(item); m["merged_from_prediction_ids"] = item["raw_interval_id"]; m["merge_reason"] = "none"; merged.append(m); continue
        last = merged[-1]
        gap = float(item["start_sec"]) - float(last["end_sec"])
        if float(item["start_sec"]) <= float(last["end_sec"]) or gap <= merge_gap:
            last["end_sec"] = round(max(float(last["end_sec"]), float(item["end_sec"])), 3)
            last["duration_sec"] = round(float(last["end_sec"]) - float(last["start_sec"]), 3)
            last["end_anchor_id"] = item["end_anchor_id"]
            last["end_reason"] = item["end_reason"]
            last["hard_evidence_count"] = int(last["hard_evidence_count"]) + int(item["hard_evidence_count"])
            last["support_evidence_count"] = int(last["support_evidence_count"]) + int(item["support_evidence_count"])
            last["weak_context_count"] = int(last["weak_context_count"]) + int(item["weak_context_count"])
            last["max_weak_span_sec"] = max(as_float(last["max_weak_span_sec"]), as_float(item["max_weak_span_sec"]))
            last["hard_evidence_density_per_60s"] = round(int(last["hard_evidence_count"]) / max(0.001, (float(last["duration_sec"]) / 60.0)), 6)
            last["merged_from_prediction_ids"] += "," + item["raw_interval_id"]
            last["merge_reason"] = "overlap_or_gap_le_10s"
        else:
            m = dict(item); m["merged_from_prediction_ids"] = item["raw_interval_id"]; m["merge_reason"] = "none"; merged.append(m)
    pred_rows = []
    for i, item in enumerate(merged, start=1):
        pred_rows.append({
            "prediction_id": f"smi_v1_4_{i:06d}", "version": VERSION, "base_version": BASE_VERSION, "split": item["split"], "video_id": item["video_id"], "ad_start_sec": item["start_sec"], "ad_end_sec": item["end_sec"], "ad_duration_sec": item["duration_sec"],
            "start_anchor_id": item["start_anchor_id"], "end_anchor_id": item["end_anchor_id"], "start_reason": item["start_reason"], "end_reason": item["end_reason"], "start_confidence_level": item["start_confidence_level"], "end_confidence_level": item["end_confidence_level"], "hard_evidence_count": item["hard_evidence_count"], "support_evidence_count": item["support_evidence_count"], "weak_context_count": item["weak_context_count"], "hard_evidence_density_per_60s": item["hard_evidence_density_per_60s"], "max_weak_span_sec": item["max_weak_span_sec"], "interval_status": "high_confidence_closed", "merged_from_prediction_ids": item["merged_from_prediction_ids"], "merge_reason": item["merge_reason"], "used_test_row": "false", "decision_feature_columns_json": json_dumps(DECISION_FEATURE_COLUMNS), "audit_columns_used_for_decision": "false", "v1_4_adjustment_flags_json": json_dumps({"interval_confidence_filter": "passed"}),
        })
    return pred_rows, review_rows, event_rows


def add_start_review(rows: list[dict[str, Any]], split: str, vid: str, row: dict[str, Any], cues: dict[str, Any], reason: str, early: bool) -> None:
    rows.append({"version": VERSION, "split": split, "video_id": vid, "visual_anchor_id": row_anchor_id(row), "transition_time_anchor": round(row_time(row), 3), "candidate_time_mmss": row.get("candidate_time_mmss", ""), "ocr_start_signal_level": cues["ocr_start"], "ocr_context_level": cues["ocr_context"], "ocr_context_reliability_level": cues["reliability"], "audio_start_signal_level": cues["audio_start"], "audio_score_delta_post_minus_pre": cues["audio_delta"], "product_repetition_continues": str(cues["product_repetition"]).lower(), "cta_or_link_context": str(cues["cta_or_link"]).lower(), "early_disclosure_guard_applied": str(early).lower(), "rejection_reason": reason, "review_note": "v1.4 strict start confirmation review-only start candidate; not a closed prediction."})


def add_open_split(open_rows: list[dict[str, Any]], unresolved_rows: list[dict[str, Any]], rejected_rows: list[dict[str, Any]], split: str, vid: str, current: dict[str, Any], last_row: dict[str, Any], video_duration: float, cfg: dict[str, Any]) -> None:
    start = float(current["time"])
    last = row_time(last_row)
    dur = max(0.0, last - start)
    density = hard_density(current, last)
    recent_window = float(cfg["unresolved_if_no_hard_evidence_recent_sec"])
    recent_count = recent_hard_count(current, last, recent_window)
    duration_ratio = ratio(dur, video_duration)
    base = {"split": split, "video_id": vid, "start_anchor_id": current.get("anchor_id", ""), "last_anchor_id": row_anchor_id(last_row), "duration_proxy_sec": round(dur, 3), "video_duration_sec": round(video_duration, 3), "duration_ratio": round(duration_ratio, 6), "recent_hard_evidence_count": recent_count, "recent_window_sec": recent_window, "hard_evidence_density_per_60s": round(density, 6), "start_reason": current.get("reason", "")}
    if dur >= float(cfg["rejected_if_duration_sec"]) and recent_count == 0 and density < float(cfg["hard_evidence_density_min_per_60s"]):
        rejected_rows.append({"version": VERSION, "video_id": vid, "rejected_candidate_id": f"rejected_{len(rejected_rows)+1:06d}", "start_sec": round(start, 3), "last_anchor_sec": round(last, 3), "rejected_reason": "rejected_long_open_low_hard_evidence_density", "moved_to_rejected_and_reset_state": "true", "review_note": "Rejected/review-only long open failure candidate; not a prediction.", **base})
    elif dur >= float(cfg["unresolved_if_duration_sec"]) and recent_count == 0:
        unresolved_rows.append({"version": VERSION, "video_id": vid, "unresolved_candidate_id": f"unresolved_{len(unresolved_rows)+1:06d}", "start_sec": round(start, 3), "last_anchor_sec": round(last, 3), "unresolved_reason": "unresolved_long_open_no_recent_hard_evidence", "moved_to_unresolved_and_reset_state": "true", "review_note": "Unresolved long open candidate; not a prediction.", **base})
    else:
        open_rows.append({"open_candidate_id": f"open_v1_4_{len(open_rows)+1:06d}", "version": VERSION, "base_version": BASE_VERSION, "video_id": vid, "ad_start_sec": round(start, 3), "last_anchor_sec": round(last, 3), "open_reason": "open_interval_candidate_recent_or_short_hard_evidence_context", "interval_status": "open_interval_candidate", "used_test_row": "false", "review_note": "Open candidate only; not a closed prediction.", **base})


def run_detector(root: Path, config: dict[str, Any], log_path: Path, dry_run: bool = False) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    bad = banned_decision_columns(DECISION_FEATURE_COLUMNS)
    if bad:
        raise RuntimeError(f"Forbidden decision columns: {bad}")
    split_check = validate_split(root, config)
    if not split_check["ok"]:
        raise RuntimeError(f"Fixed split mismatch: {split_check}")
    rows_all = read_csv(root / config["input_paths"]["primary_detector_input"])
    rows = [r for r in rows_all if r.get("split") == "train"]
    excluded = sum(1 for r in rows_all if r.get("split") in EXCLUDED_SPLITS)
    if any(r.get("split") != "train" for r in rows):
        raise RuntimeError("Non-train row survived filter")
    rows.sort(key=lambda r: (int(float(r.get("video_id", 0) or 0)), row_time(r), row_anchor_id(r)))
    if dry_run:
        return {"dry_run": True, "detector_input_row_count": len(rows), "excluded_validation_test_rows": excluded, "split_check": split_check}
    append_log(log_path, "[STEP 04] Implement v1.4 evidence tier logic")
    append_log(log_path, "[STEP 05] Implement v1.4 state transition delta")
    pending_rules = config["pending_rules"]
    start_max_count = int(pending_rules["start_pending_max_anchor_count"])
    start_max_elapsed = float(pending_rules["start_pending_max_duration_sec"])
    end_max_count = int(pending_rules["end_pending_max_anchor_count"])
    end_max_elapsed = float(pending_rules["end_pending_max_duration_sec"])
    min_duration = float(config["minimum_duration_prior"]["minimum_ad_duration_sec"])
    merge_gap = float(config["interval_merge"]["merge_gap_sec"])
    early_window = float(config["global_disclosure_guard"]["early_window_sec"])
    support_limit_count = int(config["evidence_tiers"]["support_evidence_max_anchor_count"])
    support_limit_sec = float(config["evidence_tiers"]["support_evidence_max_duration_sec"])
    hard_stale_sec = float(config["hard_evidence_freshness_timeout"]["max_sec_since_last_hard_evidence"])
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("split", ""), row_video_id(row))].append(row)
    raw_closed: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    open_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    disclosure_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    fresh_rows: list[dict[str, Any]] = []
    video_durations = split_check["video_durations"]
    for (split, vid), group in grouped.items():
        state = "non_ad"
        pending_start: dict[str, Any] | None = None
        pending_end: dict[str, Any] | None = None
        current: dict[str, Any] | None = None
        support_only_count = 0
        support_only_start_time: float | None = None
        consecutive_c3 = 0
        for idx, row in enumerate(group):
            t = row_time(row)
            anchor = row_anchor_id(row)
            cues = cue_flags(row, warnings)
            state_before = state
            decision_type = "hold"
            decision_reason = "no_state_change"
            pending_trigger = "none"
            early_guard = False
            moved_review = False
            unresolved_applied = False
            rejected_applied = False
            interval_filter_applied = False
            end_conf = ""
            pm_count, pm_elapsed = "", ""
            if cues["c3"]:
                consecutive_c3 += 1
            else:
                consecutive_c3 = 0
            if state == "non_ad":
                if cues["c3"]:
                    decision_reason = "c3_no_start_candidate"
                elif cues["ocr_start"] in {"high", "medium"} or (cues["reliability"] in {"low", "low_or_unknown"} and is_medium_or_high(cues["audio_start"])):
                    if cues["ocr_start"] == "high" and t <= early_window and not cues["product_repetition"] and not cues["cta_or_link"] and not cues["audio_rise"]:
                        early_guard = True
                        decision_type = "disclosure_notice_review"
                        decision_reason = "early_disclosure_notice_not_ad_start"
                        add_start_review(disclosure_rows, split, vid, row, cues, "early_disclosure_notice_rejected", True)
                    else:
                        if cues["ocr_start"] == "high": source = "ocr_start_high"
                        elif cues["ocr_start"] == "medium": source = "ocr_start_medium"
                        else: source = "ocr_unknown_audio_support"
                        pending_start = {"time": t, "anchor_id": anchor, "index": idx, "source": source, "reason": f"{source}_pending"}
                        state = "start_pending"
                        decision_type = "start_pending_entered"
                        decision_reason = f"{source}_pending"
                        pm_count, pm_elapsed = pending_metrics(pending_start, idx, t)
            elif state == "start_pending":
                assert pending_start is not None
                pm_count, pm_elapsed = pending_metrics(pending_start, idx, t)
                count = int(pm_count); elapsed = float(pm_elapsed)
                if count > start_max_count or elapsed > start_max_elapsed:
                    pending_trigger = timeout_trigger(count, elapsed, start_max_count, start_max_elapsed)
                    decision_type = "start_pending_timeout"
                    decision_reason = "start_pending_not_confirmed_by_required_pattern"
                    add_start_review(disclosure_rows, split, vid, row, cues, decision_reason, False)
                    pending_start = None; state = "non_ad"
                elif cues["start_confirm_pattern"]:
                    pattern = cues["start_confirm_pattern"]
                    conf = start_confidence(pattern)
                    reason = f"start_confirmed_by_{pattern}"
                    current = new_interval(float(pending_start["time"]), str(pending_start["anchor_id"]), reason, conf, pattern)
                    update_interval_evidence(current, cues, t)
                    if cues["hard"]:
                        support_only_count = 0; support_only_start_time = None
                    else:
                        support_only_count = 1 if cues["support"] else 0
                        support_only_start_time = t if cues["support"] else None
                    state = "in_ad"
                    decision_type = "start_confirmed"
                    decision_reason = reason
                    pending_trigger = "confirmed"
                    pending_start = None
                else:
                    decision_type = "start_pending_cancelled"
                    decision_reason = "start_pending_not_confirmed_by_required_pattern"
                    add_start_review(disclosure_rows, split, vid, row, cues, decision_reason, False)
                    pending_start = None; state = "non_ad"
            elif state == "in_ad":
                assert current is not None
                update_interval_evidence(current, cues, t)
                interval_duration = t - float(current["time"])
                last_hard = current.get("last_hard_evidence_time", "")
                sec_since_last = "" if last_hard == "" else t - float(last_hard)
                if cues["known_low"] and interval_duration >= min_duration:
                    pending_end = {"time": t, "anchor_id": anchor, "index": idx, "reason": "known_low_flow"}
                    state = "end_pending"; decision_type = "end_pending_entered"; decision_reason = "known_low_flow"; pm_count, pm_elapsed = pending_metrics(pending_end, idx, t)
                elif cues["hard"]:
                    support_only_count = 0; support_only_start_time = None
                    decision_type = "maintain_in_ad"; decision_reason = "in_ad_maintained_by_hard_evidence"
                elif last_hard == "" and interval_duration > hard_stale_sec:
                    pending_end = {"time": t, "anchor_id": anchor, "index": idx, "reason": "hard_evidence_stale_enter_end_pending"}
                    state = "end_pending"; decision_type = "end_pending_entered"; decision_reason = "hard_evidence_stale_enter_end_pending"; pm_count, pm_elapsed = pending_metrics(pending_end, idx, t)
                    fresh_rows.append({"version": VERSION, "split": split, "video_id": vid, "visual_anchor_id": anchor, "transition_time_anchor": round(t,3), "current_ad_start_sec": round(float(current["time"]),3), "last_hard_evidence_time": "", "sec_since_last_hard_evidence": "", "support_only_count": support_only_count, "support_only_elapsed_sec": "", "event_reason": decision_reason, "pending_end_time": round(t,3)})
                elif last_hard != "" and float(sec_since_last) > hard_stale_sec:
                    pending_end = {"time": t, "anchor_id": anchor, "index": idx, "reason": "hard_evidence_stale_enter_end_pending"}
                    state = "end_pending"; decision_type = "end_pending_entered"; decision_reason = "hard_evidence_stale_enter_end_pending"; pm_count, pm_elapsed = pending_metrics(pending_end, idx, t)
                    fresh_rows.append({"version": VERSION, "split": split, "video_id": vid, "visual_anchor_id": anchor, "transition_time_anchor": round(t,3), "current_ad_start_sec": round(float(current["time"]),3), "last_hard_evidence_time": round(float(last_hard),3), "sec_since_last_hard_evidence": round(float(sec_since_last),3), "support_only_count": support_only_count, "support_only_elapsed_sec": "", "event_reason": decision_reason, "pending_end_time": round(t,3)})
                elif cues["support"]:
                    if support_only_start_time is None:
                        support_only_start_time = t
                    support_only_count += 1
                    support_elapsed = t - float(support_only_start_time)
                    if support_only_count <= support_limit_count and support_elapsed <= support_limit_sec:
                        decision_type = "maintain_in_ad"; decision_reason = "in_ad_maintained_by_short_support_evidence"
                    else:
                        pending_time = float(support_only_start_time)
                        pending_end = {"time": pending_time, "anchor_id": anchor, "index": idx, "reason": "support_evidence_expired_enter_end_pending"}
                        state = "end_pending"; decision_type = "end_pending_entered"; decision_reason = "support_evidence_expired_enter_end_pending"; pm_count, pm_elapsed = pending_metrics(pending_end, idx, t)
                        fresh_rows.append({"version": VERSION, "split": split, "video_id": vid, "visual_anchor_id": anchor, "transition_time_anchor": round(t,3), "current_ad_start_sec": round(float(current["time"]),3), "last_hard_evidence_time": "" if last_hard == "" else round(float(last_hard),3), "sec_since_last_hard_evidence": "" if last_hard == "" else round(float(sec_since_last),3), "support_only_count": support_only_count, "support_only_elapsed_sec": round(support_elapsed,3), "event_reason": "support_only_repeated_enter_end_pending", "pending_end_time": round(pending_time,3)})
                else:
                    pending_end = {"time": t, "anchor_id": anchor, "index": idx, "reason": "no_ad_evidence_enter_end_pending"}
                    state = "end_pending"; decision_type = "end_pending_entered"; decision_reason = "weak_context_not_used_for_continuity" if cues["weak"] else "no_ad_evidence_enter_end_pending"; pm_count, pm_elapsed = pending_metrics(pending_end, idx, t)
            elif state == "end_pending":
                assert current is not None and pending_end is not None
                update_interval_evidence(current, cues, t)
                pm_count, pm_elapsed = pending_metrics(pending_end, idx, t)
                count = int(pm_count); elapsed = float(pm_elapsed)
                if cues["hard"]:
                    state = "in_ad"
                    decision_type = "end_pending_cancelled"
                    decision_reason = "end_pending_cancelled_by_hard_evidence"
                    pending_trigger = "continuity_cancel_by_hard_evidence"
                    pending_end = None
                    support_only_count = 0; support_only_start_time = None
                elif cues["known_low"]:
                    close_raw(raw_closed, split, vid, current, float(pending_end["time"]), anchor, "end_pending_confirmed_by_known_low_flow", "high", "known_low_flow")
                    state = "non_ad"; current = None; pending_end = None; support_only_count = 0; support_only_start_time = None
                    decision_type = "end_confirmed"; decision_reason = "end_pending_confirmed_by_known_low_flow"; pending_trigger = "confirmed"
                    end_conf = "high"
                elif count > end_max_count or elapsed > end_max_elapsed:
                    pending_trigger = timeout_trigger(count, elapsed, end_max_count, end_max_elapsed)
                    close_raw(raw_closed, split, vid, current, float(pending_end["time"]), str(pending_end.get("anchor_id") or anchor), "end_pending_timeout_without_hard_evidence", "medium", "timeout_without_hard_evidence")
                    state = "non_ad"; current = None; pending_end = None; support_only_count = 0; support_only_start_time = None
                    decision_type = "end_confirmed"; decision_reason = "end_pending_timeout_without_hard_evidence"; end_conf = "medium"
                else:
                    decision_type = "end_pending_wait"
                    decision_reason = "end_pending_waiting_without_hard_evidence"
            state_after = state
            sec_since_last = ""
            last_hard_trace = ""
            if current is not None and current.get("last_hard_evidence_time", "") != "":
                last_hard_trace = round(float(current["last_hard_evidence_time"]), 3)
                sec_since_last = round(max(0.0, t - float(current["last_hard_evidence_time"])), 3)
            support_elapsed_trace = "" if support_only_start_time is None else round(t - float(support_only_start_time), 3)
            evidence_summary = "hard" if cues["hard"] else ("support" if cues["support"] else ("weak" if cues["weak"] else "none"))
            trace_rows.append({"version": VERSION, "base_version": BASE_VERSION, "split": split, "video_id": vid, "visual_anchor_id": anchor, "transition_time_anchor": round(t,3), "candidate_time_mmss": row.get("candidate_time_mmss", mmss(t)), "state_before": state_before, "state_after": state_after, "decision_type": decision_type, "decision_reason": decision_reason, "pending_start_time": "" if pending_start is None else round(float(pending_start["time"]),3), "pending_start_source": "" if pending_start is None else pending_start.get("source", ""), "pending_end_time": "" if pending_end is None else round(float(pending_end["time"]),3), "pending_elapsed_sec_actual": pm_elapsed, "pending_anchor_count_actual": pm_count, "pending_timeout_trigger": pending_trigger, "audio_start_signal_level": cues["audio_start"], "audio_end_signal_level": cues["audio_end"], "audio_context_level": cues["audio_context"], "audio_before_context_level": cues["audio_before"], "audio_after_context_level": cues["audio_after"], "audio_score_delta_post_minus_pre": cues["audio_delta"], "ocr_start_signal_level": cues["ocr_start"], "ocr_end_signal_level": cues["ocr_end"], "ocr_context_level": cues["ocr_context"], "ocr_context_reliability_level": cues["reliability"], "product_repetition_continues": str(cues["product_repetition"]).lower(), "single_product_or_brand_mention": str(cues["single_product"]).lower(), "cta_or_link_context": str(cues["cta_or_link"]).lower(), "audio_rise": str(cues["audio_rise"]).lower(), "ocr_post_increase": str(cues["ocr_post_increase"]).lower(), "c3_flag": str(cues["c3"]).lower(), "d2_flag": str(cues["d2"]).lower(), "hard_evidence_flag": str(cues["hard"]).lower(), "support_evidence_flag": str(cues["support"]).lower(), "weak_context_flag": str(cues["weak"]).lower(), "hard_evidence_reasons": "+".join(cues["hard_reasons"]), "support_evidence_reasons": "+".join(cues["support_reasons"]), "weak_context_reasons": "+".join(cues["weak_reasons"]), "last_hard_evidence_time": last_hard_trace, "sec_since_last_hard_evidence": sec_since_last, "support_only_count": support_only_count, "support_only_elapsed_sec": support_elapsed_trace, "start_confirm_pattern": cues["start_confirm_pattern"], "start_confidence_level": start_confidence(cues["start_confirm_pattern"]) if cues["start_confirm_pattern"] else "", "end_confidence_level": end_conf, "early_disclosure_guard_applied": str(early_guard).lower(), "interval_confidence_filter_applied": str(interval_filter_applied).lower(), "moved_to_review_only": str(moved_review).lower(), "unresolved_long_open_guard_applied": str(unresolved_applied).lower(), "rejected_long_open_guard_applied": str(rejected_applied).lower(), "used_for_decision_columns_json": json_dumps(DECISION_FEATURE_COLUMNS), "audit_columns_used_for_decision": "false"})
            evidence_rows.append({"version": VERSION, "split": split, "video_id": vid, "visual_anchor_id": anchor, "transition_time_anchor": round(t,3), "state_before": state_before, "state_after": state_after, "hard_evidence_flag": str(cues["hard"]).lower(), "support_evidence_flag": str(cues["support"]).lower(), "weak_context_flag": str(cues["weak"]).lower(), "hard_evidence_reasons": "+".join(cues["hard_reasons"]), "support_evidence_reasons": "+".join(cues["support_reasons"]), "weak_context_reasons": "+".join(cues["weak_reasons"]), "start_confirm_pattern": cues["start_confirm_pattern"], "evidence_tier_summary": evidence_summary})
        if current is not None and group:
            add_open_split(open_rows, unresolved_rows, rejected_rows, split, vid, current, group[-1], video_durations.get(vid, 0.0), config["long_open_candidate_split"])
    pred_rows, review_rows, filter_rows = filter_and_merge_intervals(raw_closed, merge_gap)
    # 실행 결과에 confidence filter가 있으면 trace row에 표시한다.
    for tr in trace_rows:
        tr["interval_confidence_filter_applied"] = "true"
    outp = config["output_paths"]
    append_log(log_path, "[STEP 06] Run v1.4 detector on train only")
    write_csv(root / outp["prediction_csv"], pred_rows, PREDICTION_COLUMNS)
    write_csv(root / outp["review_only_interval_csv"], review_rows, REVIEW_ONLY_COLUMNS)
    write_csv(root / outp["trace_csv"], trace_rows, TRACE_COLUMNS)
    write_csv(root / outp["open_interval_csv"], open_rows, OPEN_COLUMNS)
    write_csv(root / outp["unresolved_long_open_csv"], unresolved_rows, UNRESOLVED_COLUMNS)
    write_csv(root / outp["rejected_long_open_csv"], rejected_rows, REJECTED_COLUMNS)
    write_csv(root / outp["disclosure_notice_review_csv"], disclosure_rows, DISCLOSURE_COLUMNS)
    write_csv(root / outp["evidence_tier_events_csv"], evidence_rows, EVIDENCE_TIER_COLUMNS)
    write_csv(root / outp["fresh_evidence_timeout_events_csv"], fresh_rows, FRESH_TIMEOUT_COLUMNS)
    write_csv(root / outp["interval_confidence_filter_events_csv"], filter_rows, CONF_FILTER_COLUMNS)
    for name, rr in [("prediction", pred_rows), ("review", review_rows), ("trace", trace_rows), ("open", open_rows), ("unresolved", unresolved_rows), ("rejected", rejected_rows)]:
        if any(r.get("split") != "train" for r in rr):
            raise RuntimeError(f"Non-train row in {name} output")
    return {"warnings": warnings, "errors": errors, "split_check": split_check, "excluded_validation_test_rows": excluded, "detector_input_row_count": len(rows), "pred_rows": pred_rows, "review_rows": review_rows, "trace_rows": trace_rows, "open_rows": open_rows, "unresolved_rows": unresolved_rows, "rejected_rows": rejected_rows, "disclosure_rows": disclosure_rows, "evidence_rows": evidence_rows, "fresh_rows": fresh_rows, "filter_rows": filter_rows, "raw_closed_rows": raw_closed}


def union_intervals(items: list[dict[str, Any]], duration_limit: float | None = None) -> list[tuple[float, float]]:
    vals = []
    for item in items:
        s = as_float(item.get("start_sec", item.get("ad_start_sec", 0.0)))
        e = as_float(item.get("end_sec", item.get("ad_end_sec", item.get("last_anchor_sec", 0.0))))
        if duration_limit and duration_limit > 0:
            s = max(0.0, min(s, duration_limit)); e = max(0.0, min(e, duration_limit))
        if e > s:
            vals.append((s, e))
    vals.sort()
    merged: list[tuple[float, float]] = []
    for s, e in vals:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def duration_sum(intervals: list[tuple[float, float]]) -> float:
    return sum(max(0.0, e - s) for s, e in intervals)


def overlap_between(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    total = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def parse_actual(root: Path, config: dict[str, Any], train_ids: set[str], durations: dict[str, float], warnings: list[str]) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(read_csv(root / config["input_paths"]["actual_label_file"]), start=1):
        vid = row_video_id(row)
        if vid not in train_ids:
            continue
        seg = (row.get("segment_type") or row.get("label") or "").lower()
        is_ad = "ad_interval" in seg or seg in {"ad", "ad_full", "advertisement"} or str(row.get("is_ad", "")).lower() in {"1", "true", "yes"}
        if any(x in seg for x in ["non_ad", "random_non_ad", "post_ad", "pre_ad"]):
            is_ad = False
        if not is_ad:
            continue
        s = as_float(row.get("ad_start_sec") or row.get("segment_start_sec"), float("nan"))
        e = as_float(row.get("ad_end_sec") or row.get("segment_end_sec"), float("nan"))
        if math.isnan(s) or math.isnan(e) or e <= s:
            warnings.append(f"invalid_actual_interval_excluded:{vid}:{i}"); continue
        dur = durations.get(vid, 0.0)
        if dur > 0:
            s = max(0.0, min(s, dur)); e = max(0.0, min(e, dur))
        if e > s:
            by[vid].append({"video_id": vid, "split": "train", "interval_type": "actual", "interval_id": row.get("ad_interval_id") or row.get("segment_id") or f"actual_{vid}_{i:04d}", "start_sec": s, "end_sec": e, "duration_sec": e - s, "reason": seg or "actual_ad_interval"})
    return by


def pred_like(rows: list[dict[str, Any]], kind: str, id_col: str, start_col: str, end_col: str) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, r in enumerate(rows, start=1):
        vid = row_video_id(r)
        s = as_float(r.get(start_col), float("nan")); e = as_float(r.get(end_col), float("nan"))
        if math.isnan(s) or math.isnan(e) or e <= s:
            continue
        by[vid].append({"video_id": vid, "split": "train", "interval_type": kind, "interval_id": r.get(id_col) or f"{kind}_{vid}_{i:04d}", "start_sec": s, "end_sec": e, "duration_sec": e - s, "reason": r.get("start_reason") or r.get("open_reason") or r.get("unresolved_reason") or r.get("rejected_reason") or r.get("review_only_reasons") or ""})
    return by


def severity_high(actual_cov: float, pred_cov: float, actual_total: float, pred_total: float, missed_ratio: float) -> str:
    if actual_total <= 0 and pred_total > 0: return "no_actual_but_predicted"
    over = pred_cov / actual_cov if actual_cov > 0 else 0.0
    if pred_cov >= 0.5 or over >= 3: return "severe_over_detection"
    if pred_cov >= 0.25 or over >= 2: return "moderate_over_detection"
    if missed_ratio >= 0.5: return "missed_actual"
    return "reasonable_or_needs_manual_review"


def candidate_level(actual_cov: float, total_cov: float, open_cov: float, unresolved_cov: float, rejected_cov: float, actual_total: float, total: float) -> str:
    if actual_total <= 0 and total > 0: return "no_actual_but_candidate"
    over = total_cov / actual_cov if actual_cov > 0 else 0.0
    if total_cov >= 0.5 or over >= 3 or open_cov + unresolved_cov + rejected_cov >= 0.3: return "severe_candidate_overcoverage"
    if total_cov >= 0.25 or over >= 2: return "moderate_candidate_overcoverage"
    return "candidate_needs_manual_review"


def build_audit(root: Path, config: dict[str, Any], det: dict[str, Any], log_path: Path) -> dict[str, Any]:
    append_log(log_path, "[STEP 07] Train-only audit/error analysis for v1.4")
    outp = config["output_paths"]
    warnings = det["warnings"]
    durations = det["split_check"]["video_durations"]
    train_ids = {str(x) for x in EXPECTED_SPLITS["train"]}
    actual_by = parse_actual(root, config, train_ids, durations, warnings)
    high_by = pred_like(det["pred_rows"], "high_confidence_prediction", "prediction_id", "ad_start_sec", "ad_end_sec")
    review_by = pred_like(det["review_rows"], "review_only_closed_candidate", "candidate_id", "start_sec", "end_sec")
    open_by = pred_like(det["open_rows"], "open_candidate", "open_candidate_id", "ad_start_sec", "last_anchor_sec")
    unresolved_by = pred_like(det["unresolved_rows"], "unresolved_long_open", "unresolved_candidate_id", "start_sec", "last_anchor_sec")
    rejected_by = pred_like(det["rejected_rows"], "rejected_long_open", "rejected_candidate_id", "start_sec", "last_anchor_sec")
    video_rows = []
    interval_rows = []
    open_summary = []
    for vid in sorted(train_ids, key=lambda x: int(x)):
        dur = durations.get(vid, 0.0)
        au = union_intervals(actual_by.get(vid, []), dur)
        hu = union_intervals(high_by.get(vid, []), dur)
        ru = union_intervals(review_by.get(vid, []), dur)
        ou = union_intervals(open_by.get(vid, []), dur)
        uu = union_intervals(unresolved_by.get(vid, []), dur)
        ju = union_intervals(rejected_by.get(vid, []), dur)
        total_u = union_intervals(high_by.get(vid, []) + review_by.get(vid, []) + open_by.get(vid, []) + unresolved_by.get(vid, []) + rejected_by.get(vid, []), dur)
        actual_total = duration_sum(au); high_total = duration_sum(hu); review_total = duration_sum(ru); open_total = duration_sum(ou); unresolved_total = duration_sum(uu); rejected_total = duration_sum(ju); total_candidate = duration_sum(total_u)
        overlap_high = overlap_between(au, hu); fp_high = max(0.0, high_total - overlap_high); missed = max(0.0, actual_total - overlap_high)
        review_overlap = overlap_between(au, ru)
        ac = ratio(actual_total, dur); hc = ratio(high_total, dur); rc = ratio(review_total, dur); oc = ratio(open_total, dur); uc = ratio(unresolved_total, dur); jc = ratio(rejected_total, dur); tc = ratio(total_candidate, dur)
        mr = ratio(missed, actual_total)
        sev = severity_high(ac, hc, actual_total, high_total, mr)
        cand = candidate_level(ac, tc, oc, uc, jc, actual_total, total_candidate)
        pri = "high" if sev in {"severe_over_detection", "no_actual_but_predicted"} or cand.startswith("severe") else ("medium" if sev in {"moderate_over_detection", "missed_actual"} or cand.startswith("moderate") else "low")
        note = []
        if hc < ac and mr >= 0.5: note.append("high-confidence output may be too strict; inspect review-only overlap")
        if rc > 0: note.append("review-only closed candidates separated from predictions")
        if jc > 0: note.append("rejected long open candidates separated")
        if cand.startswith("severe"): note.append("candidate overcoverage remains high for manual review")
        video_rows.append({"video_id": vid, "split": "train", "video_duration_sec": round(dur,3), "actual_interval_count": len(actual_by.get(vid, [])), "actual_total_duration_sec": round(actual_total,3), "actual_coverage_ratio": round(ac,6), "high_confidence_prediction_count": len(high_by.get(vid, [])), "high_confidence_total_duration_sec": round(high_total,3), "high_confidence_coverage_ratio": round(hc,6), "review_only_candidate_count": len(review_by.get(vid, [])), "review_only_total_duration_sec": round(review_total,3), "review_only_coverage_ratio": round(rc,6), "open_interval_count": len(open_by.get(vid, [])), "open_total_duration_proxy_sec": round(open_total,3), "open_coverage_ratio": round(oc,6), "unresolved_long_open_count": len(unresolved_by.get(vid, [])), "unresolved_total_duration_proxy_sec": round(unresolved_total,3), "unresolved_coverage_ratio": round(uc,6), "rejected_long_open_count": len(rejected_by.get(vid, [])), "rejected_total_duration_proxy_sec": round(rejected_total,3), "rejected_coverage_ratio": round(jc,6), "total_candidate_duration_sec": round(total_candidate,3), "total_candidate_coverage_ratio": round(tc,6), "actual_high_conf_overlap_duration_sec": round(overlap_high,3), "actual_high_conf_overlap_ratio_of_actual": round(ratio(overlap_high, actual_total),6), "actual_high_conf_overlap_ratio_of_prediction": round(ratio(overlap_high, high_total),6), "high_conf_false_positive_duration_sec": round(fp_high,3), "high_conf_false_positive_ratio_of_video": round(ratio(fp_high,dur),6), "missed_actual_duration_sec": round(missed,3), "missed_actual_ratio_of_actual": round(mr,6), "review_candidate_overlap_with_actual_sec": round(review_overlap,3), "severity_level_high_confidence": sev, "candidate_overcoverage_level": cand, "review_priority": pri, "diagnosis_note": "; ".join(note) if note else "manual review recommended"})
        for coll, typ in [(high_by.get(vid, []), "high_confidence_prediction"), (review_by.get(vid, []), "review_only_closed_candidate"), (open_by.get(vid, []), "open_candidate"), (unresolved_by.get(vid, []), "unresolved_long_open"), (rejected_by.get(vid, []), "rejected_long_open")]:
            for p in coll:
                pdur = as_float(p["duration_sec"]); ov = overlap_between(union_intervals([p], dur), au); fp = max(0.0, pdur - ov)
                issue = "good_overlap" if ratio(ov, pdur) >= 0.5 else ("review_only_candidate" if typ == "review_only_closed_candidate" else ("open_unresolved_or_rejected_candidate" if typ != "high_confidence_prediction" else "false_positive_candidate"))
                interval_rows.append({"video_id": vid, "split": "train", "interval_type": typ, "interval_id": p["interval_id"], "start_sec": round(as_float(p["start_sec"]),3), "end_sec": round(as_float(p["end_sec"]),3), "duration_sec": round(pdur,3), "best_matching_actual_id": "", "best_actual_overlap_sec": round(ov,3), "best_actual_overlap_ratio_of_interval": round(ratio(ov,pdur),6), "false_positive_duration_sec": round(fp,3), "matched_prediction_ids": "", "missed_actual_duration_sec": "", "issue_type": issue, "reason": p.get("reason", ""), "review_note": "train-only audit; no final performance claim"})
        for a in actual_by.get(vid, []):
            adur = as_float(a["duration_sec"]); ov = overlap_between(union_intervals([a], dur), hu); miss = max(0.0, adur - ov)
            interval_rows.append({"video_id": vid, "split": "train", "interval_type": "actual", "interval_id": a["interval_id"], "start_sec": round(as_float(a["start_sec"]),3), "end_sec": round(as_float(a["end_sec"]),3), "duration_sec": round(adur,3), "best_matching_actual_id": a["interval_id"], "best_actual_overlap_sec": "", "best_actual_overlap_ratio_of_interval": "", "false_positive_duration_sec": "", "matched_prediction_ids": "", "missed_actual_duration_sec": round(miss,3), "issue_type": "missed_actual_candidate" if ratio(miss, adur) >= 0.5 else "good_overlap", "reason": a.get("reason", "actual"), "review_note": "actual train label used only after detector run for audit"})
        for coll, kind in [(open_by.get(vid, []), "open_candidate"), (unresolved_by.get(vid, []), "unresolved_long_open"), (rejected_by.get(vid, []), "rejected_long_open")]:
            for o in coll:
                odur = as_float(o["duration_sec"]); ov = overlap_between(union_intervals([o], dur), au)
                open_summary.append({"video_id": vid, "split": "train", "candidate_kind": kind, "candidate_id": o["interval_id"], "start_sec": round(as_float(o["start_sec"]),3), "end_proxy_sec": round(as_float(o["end_sec"]),3), "duration_proxy_sec": round(odur,3), "video_duration_sec": round(dur,3), "coverage_ratio": round(ratio(odur,dur),6), "overlap_with_actual_sec": round(ov,3), "overlap_ratio_of_candidate": round(ratio(ov,odur),6), "start_reason": o.get("reason", ""), "last_state": "", "possible_failure_mode": "rejected_or_unresolved_long_candidate" if kind != "open_candidate" else "needs_manual_review", "review_priority": "high" if ratio(odur,dur) >= 0.3 else "medium", "diagnosis_note": "candidate separated from high-confidence closed predictions"})
    trace_summary = build_trace_summary(det["trace_rows"])
    worst = sorted(video_rows, key=lambda r: (0 if r["review_priority"] == "high" else 1, -as_float(r["total_candidate_coverage_ratio"]), -as_float(r["missed_actual_ratio_of_actual"])))
    issues = build_rule_issues(video_rows, det, trace_summary)
    write_csv(root / outp["analysis_video_summary_csv"], video_rows, VIDEO_SUMMARY_COLUMNS)
    write_csv(root / outp["analysis_interval_overlap_csv"], interval_rows, INTERVAL_OVERLAP_COLUMNS)
    write_csv(root / outp["analysis_open_summary_csv"], open_summary, OPEN_SUMMARY_COLUMNS)
    write_csv(root / outp["analysis_trace_reason_summary_csv"], trace_summary, TRACE_REASON_COLUMNS)
    write_csv(root / outp["analysis_worst_cases_csv"], [{"rank": i + 1, **r} for i, r in enumerate(worst)], WORST_COLUMNS)
    write_csv(root / outp["analysis_rule_issue_candidates_csv"], issues, RULE_ISSUE_COLUMNS)
    audit_rows = []
    for vid in sorted(train_ids, key=lambda x: int(x)):
        tr = [r for r in det["trace_rows"] if r.get("video_id") == vid]
        audit_rows.append({"version": VERSION, "split": "train", "video_id": vid, "high_confidence_prediction_count": len(high_by.get(vid, [])), "review_only_candidate_count": len(review_by.get(vid, [])), "open_interval_candidate_count": len(open_by.get(vid, [])), "unresolved_long_open_count": len(unresolved_by.get(vid, [])), "rejected_long_open_count": len(rejected_by.get(vid, [])), "interval_confidence_filter_event_count": len([r for r in det["filter_rows"] if r.get("video_id") == vid]), "hard_evidence_trace_count": sum(1 for r in tr if is_true(r.get("hard_evidence_flag"))), "support_evidence_trace_count": sum(1 for r in tr if is_true(r.get("support_evidence_flag"))), "weak_context_trace_count": sum(1 for r in tr if is_true(r.get("weak_context_flag"))), "pending_start_timeout_count": sum(1 for r in tr if r.get("decision_type") in {"start_pending_timeout", "start_pending_cancelled"}), "pending_end_timeout_count": sum(1 for r in tr if "timeout" in r.get("decision_reason", "")), "start_reason_counts_json": json_dumps(Counter(r.get("start_reason", "") for r in det["pred_rows"] if r.get("video_id") == vid)), "end_reason_counts_json": json_dumps(Counter(r.get("end_reason", "") for r in det["pred_rows"] if r.get("video_id") == vid)), "state_transition_counts_json": json_dumps(Counter(f"{r.get('state_before')}->{r.get('state_after')}" for r in tr)), "audit_columns_used_for_decision": "false", "final_performance_claim": "false"})
    write_csv(root / outp["audit_csv"], audit_rows, TRAIN_AUDIT_COLUMNS)
    return {"video_rows": video_rows, "interval_rows": interval_rows, "open_summary": open_summary, "trace_summary": trace_summary, "worst_rows": worst, "rule_issues": issues, "audit_rows": audit_rows}


def build_trace_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    def add(scope: str, vid: str, category: str, counter: Counter[str], note: str) -> None:
        total = sum(counter.values())
        for key, count in counter.most_common():
            out.append({"scope": scope, "video_id": vid, "category": category, "key": key, "count": count, "ratio_within_category": round(ratio(count,total),6), "note": note})
    scopes = [("overall_train", "", rows)] + [("video", vid, [r for r in rows if r.get("video_id") == vid]) for vid in sorted({r.get("video_id") for r in rows}, key=lambda x: int(x))]
    for scope, vid, rr in scopes:
        add(scope, vid, "state_transition", Counter(f"{r.get('state_before')}->{r.get('state_after')}" for r in rr), "train trace only")
        add(scope, vid, "decision_reason", Counter(r.get("decision_reason", "") for r in rr), "train trace only")
        add(scope, vid, "evidence_tier", Counter("hard" if is_true(r.get("hard_evidence_flag")) else ("support" if is_true(r.get("support_evidence_flag")) else ("weak" if is_true(r.get("weak_context_flag")) else "none")) for r in rr), "hard/support/weak tier summary")
        add(scope, vid, "hard_evidence_reason", Counter(x for r in rr for x in str(r.get("hard_evidence_reasons", "")).split("+") if x), "hard evidence reason counts")
        add(scope, vid, "support_evidence_reason", Counter(x for r in rr for x in str(r.get("support_evidence_reasons", "")).split("+") if x), "support evidence reason counts")
        add(scope, vid, "v1_4_event", Counter({"fresh_evidence_timeout": sum(1 for r in rr if r.get("decision_reason") in {"hard_evidence_stale_enter_end_pending", "support_evidence_expired_enter_end_pending"}), "weak_context_not_used_for_continuity": sum(1 for r in rr if r.get("decision_reason") == "weak_context_not_used_for_continuity"), "end_pending_timeout_without_hard_evidence": sum(1 for r in rr if r.get("decision_reason") == "end_pending_timeout_without_hard_evidence")}), "event counts")
    return out


def build_rule_issues(video_rows: list[dict[str, Any]], det: dict[str, Any], trace_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missed = [r for r in video_rows if as_float(r["missed_actual_ratio_of_actual"]) >= 0.5]
    candidate_high = [r for r in video_rows if str(r["candidate_overcoverage_level"]).startswith("severe")]
    high_fp = [r for r in video_rows if as_float(r["high_conf_false_positive_duration_sec"]) > 60]
    def vids(rows: list[dict[str, Any]]) -> str:
        return ",".join(str(r["video_id"]) for r in rows[:8]) if rows else "none"
    return [
        {"issue_id": "V14_RI001", "issue_name": "high-confidence may be too strict and move actual ads to review-only", "evidence_from_train": f"missed_actual_ratio>=0.5 videos={len(missed)}; review_only_count={len(det['review_rows'])}", "affected_video_count": len(missed), "example_video_ids": vids(missed), "severity": "high" if len(missed) >= 3 else "medium", "suspected_rule_area": "strict_start_confirmation", "suggested_direction": "viewer에서 review-only 후보가 실제 광고를 포함하는지 확인하고, hard pattern을 완화할지 결정", "caution": "actual label은 audit-only이며 validation/test 전 train 검토 필요", "requires_manual_review": "true"},
        {"issue_id": "V14_RI002", "issue_name": "candidate overcoverage remains in review/open buckets", "evidence_from_train": f"severe candidate-overcoverage videos={len(candidate_high)}; open/unresolved/rejected={len(det['open_rows'])}/{len(det['unresolved_rows'])}/{len(det['rejected_rows'])}", "affected_video_count": len(candidate_high), "example_video_ids": vids(candidate_high), "severity": "high" if candidate_high else "medium", "suspected_rule_area": "long_open_candidate_split", "suggested_direction": "review-only/open/rejected bucket이 사람이 검토 가능한 크기인지 확인", "caution": "candidate bucket은 prediction이 아니므로 성능 claim에 쓰지 말 것", "requires_manual_review": "true"},
        {"issue_id": "V14_RI003", "issue_name": "remaining high-confidence false positives need boundary review", "evidence_from_train": f"high_conf_false_positive>60s videos={len(high_fp)}", "affected_video_count": len(high_fp), "example_video_ids": vids(high_fp), "severity": "medium", "suspected_rule_area": "interval_confidence_filter", "suggested_direction": "hard evidence density 조건과 end confidence를 trace로 검토", "caution": "너무 강한 filter는 missed actual을 늘릴 수 있음", "requires_manual_review": "true"},
    ]


def summarize_metrics(rows: list[dict[str, Any]], det: dict[str, Any]) -> dict[str, Any]:
    n = len(rows) or 1
    return {"train_video_count": len(rows), "actual_total_duration_sec": round(sum(as_float(r["actual_total_duration_sec"]) for r in rows),3), "high_confidence_total_duration_sec": round(sum(as_float(r["high_confidence_total_duration_sec"]) for r in rows),3), "review_only_total_duration_sec": round(sum(as_float(r["review_only_total_duration_sec"]) for r in rows),3), "open_interval_total_duration_proxy_sec": round(sum(as_float(r["open_total_duration_proxy_sec"]) for r in rows),3), "unresolved_long_open_total_duration_proxy_sec": round(sum(as_float(r["unresolved_total_duration_proxy_sec"]) for r in rows),3), "rejected_long_open_total_duration_proxy_sec": round(sum(as_float(r["rejected_total_duration_proxy_sec"]) for r in rows),3), "mean_actual_coverage": round(sum(as_float(r["actual_coverage_ratio"]) for r in rows)/n,6), "mean_high_confidence_coverage": round(sum(as_float(r["high_confidence_coverage_ratio"]) for r in rows)/n,6), "mean_review_only_coverage": round(sum(as_float(r["review_only_coverage_ratio"]) for r in rows)/n,6), "mean_open_coverage": round(sum(as_float(r["open_coverage_ratio"]) for r in rows)/n,6), "mean_unresolved_coverage": round(sum(as_float(r["unresolved_coverage_ratio"]) for r in rows)/n,6), "mean_rejected_coverage": round(sum(as_float(r["rejected_coverage_ratio"]) for r in rows)/n,6), "severe_over_detection_count_high_confidence": sum(1 for r in rows if r["severity_level_high_confidence"] == "severe_over_detection"), "severe_candidate_overcoverage_count": sum(1 for r in rows if str(r["candidate_overcoverage_level"]).startswith("severe")), "high_confidence_prediction_count": len(det["pred_rows"]), "review_only_candidate_count": len(det["review_rows"]), "open_interval_count": len(det["open_rows"]), "unresolved_long_open_count": len(det["unresolved_rows"]), "rejected_long_open_count": len(det["rejected_rows"]), "interval_confidence_filter_event_count": len(det["filter_rows"]), "hard_evidence_trace_count": sum(1 for r in det["trace_rows"] if is_true(r.get("hard_evidence_flag"))), "support_evidence_trace_count": sum(1 for r in det["trace_rows"] if is_true(r.get("support_evidence_flag"))), "weak_context_trace_count": sum(1 for r in det["trace_rows"] if is_true(r.get("weak_context_flag"))), "fresh_evidence_timeout_event_count": len(det["fresh_rows"]), "worst_videos": [r["video_id"] for r in sorted(rows, key=lambda x: as_float(x["total_candidate_coverage_ratio"]), reverse=True)[:5]]}


def build_comparison(root: Path, config: dict[str, Any], analysis: dict[str, Any]) -> list[dict[str, Any]]:
    append_log(root / config["output_paths"]["run_log"], "[STEP 08] Build v1.3 vs v1.4 train comparison")
    v13 = read_csv(root / "data/analysis/v1_3_train_video_result_table_for_review.csv")
    v13_by = {row_video_id(r): r for r in v13}
    rows = []
    for r in analysis["video_rows"]:
        vid = str(r["video_id"]); b = v13_by.get(vid, {})
        v13_closed = as_float(b.get("closed_coverage_ratio")); v13_open = as_float(b.get("open_coverage_ratio")); v13_unres = as_float(b.get("unresolved_coverage_ratio")); v13_total = as_float(b.get("predicted_plus_open_plus_unresolved_coverage_ratio"));
        v14_high = as_float(r["high_confidence_coverage_ratio"]); v14_total = as_float(r["total_candidate_coverage_ratio"])
        v13_miss = as_float(b.get("missed_actual_ratio")); v14_miss = as_float(r["missed_actual_ratio_of_actual"])
        if v14_high < v13_closed - 0.05 and v14_miss <= v13_miss + 0.25:
            status = "better_candidate"
        elif v14_miss > v13_miss + 0.4:
            status = "worse_candidate"
        elif v14_high < v13_closed - 0.05:
            status = "mixed_needs_review"
        elif abs(v14_high - v13_closed) <= 0.05:
            status = "unchanged"
        else:
            status = "worse_candidate"
        rows.append({"video_id": vid, "split": "train", "actual_coverage_ratio": r["actual_coverage_ratio"], "v1_3_closed_coverage_ratio": b.get("closed_coverage_ratio", ""), "v1_4_high_confidence_coverage_ratio": r["high_confidence_coverage_ratio"], "v1_4_review_only_coverage_ratio": r["review_only_coverage_ratio"], "v1_3_open_coverage_ratio": b.get("open_coverage_ratio", ""), "v1_4_open_coverage_ratio": r["open_coverage_ratio"], "v1_3_unresolved_coverage_ratio": b.get("unresolved_coverage_ratio", ""), "v1_4_unresolved_coverage_ratio": r["unresolved_coverage_ratio"], "v1_4_rejected_long_open_coverage_ratio": r["rejected_coverage_ratio"], "v1_3_total_candidate_coverage_ratio": b.get("predicted_plus_open_plus_unresolved_coverage_ratio", ""), "v1_4_total_candidate_coverage_ratio": r["total_candidate_coverage_ratio"], "v1_3_closed_prediction_count": b.get("closed_prediction_count", ""), "v1_4_high_confidence_prediction_count": r["high_confidence_prediction_count"], "v1_4_review_only_candidate_count": r["review_only_candidate_count"], "v1_3_severity_level": b.get("severity_level", ""), "v1_4_severity_level_high_confidence": r["severity_level_high_confidence"], "v1_4_candidate_overcoverage_level": r["candidate_overcoverage_level"], "got_better_or_worse": status, "review_priority": r["review_priority"], "comparison_note": "train-only candidate comparison; not a final performance claim"})
    write_csv(root / config["output_paths"]["comparison_csv"], rows, COMPARISON_COLUMNS)
    return rows


def write_docs(root: Path, config: dict[str, Any], report: dict[str, Any]) -> None:
    append_log(root / config["output_paths"]["run_log"], "[STEP 09] Generate v1.4 adjustment note")
    m = report["v1_4_metrics"]
    b = report["v1_3_baseline_metrics"]
    note = f"""# State Machine Interval Detector v1.4 Adjustment Note

## 1. v1.4를 만든 이유

v1.3 train-only 결과에서 closed/open/unresolved 후보가 여전히 과도하게 넓게 남았다. v1.4는 약한 후보를 버리는 것이 아니라, high-confidence closed prediction과 review-only candidate를 분리하기 위한 train-only adjustment candidate다.

## 2. v1.3 train 결과에서 확인된 문제

v1.3 severe over-detection count는 `{b.get('severe_over_detection_count')}`였고, closed/open/unresolved coverage가 실제 광고 coverage보다 컸다. 특히 start gate와 in_ad continuity가 여전히 넓게 작동했다.

## 3. 수정 1: strict start confirmation

OCR/audio medium rise alone, OCR context alone, audio context alone으로 start를 확정하지 않는다. 지정된 product/CTA/audio/증가 패턴 중 하나가 있어야 start_pending이 in_ad로 확정된다.

## 4. 수정 2: evidence tier split

각 anchor의 근거를 hard_evidence, support_evidence, weak_context로 분리한다. OCR/audio medium 단독과 D2는 hard가 아니라 support이며, weak_context는 in_ad 유지 근거가 아니다.

## 5. 수정 3: hard evidence freshness timeout

마지막 hard evidence가 없거나 30초를 넘게 stale해지면 end_pending으로 이동한다.

## 6. 수정 4: end_pending cancel only by hard evidence

end_pending은 hard evidence가 나올 때만 취소된다. support evidence, D2, 단일 product mention만으로는 in_ad 복귀를 허용하지 않는다.

## 7. 수정 5: interval confidence filter

닫힌 interval도 바로 prediction이 되지 않는다. start/end confidence, hard evidence count, weak span, hard evidence density를 확인해 high-confidence prediction과 review-only closed candidate로 분리한다.

## 8. 수정 6: long open candidate split

open, unresolved long open, rejected long open을 별도 CSV로 분리한다. rejected long open은 prediction이 아니라 약한 장기 후보 review bucket이다.

## 9. v1.4 train 결과 요약

- train video count: `{m['train_video_count']}`
- high-confidence prediction count: `{m['high_confidence_prediction_count']}`
- review-only candidate count: `{m['review_only_candidate_count']}`
- open/unresolved/rejected count: `{m['open_interval_count']}/{m['unresolved_long_open_count']}/{m['rejected_long_open_count']}`
- mean actual coverage: `{m['mean_actual_coverage']}`
- mean high-confidence coverage: `{m['mean_high_confidence_coverage']}`
- mean review-only coverage: `{m['mean_review_only_coverage']}`
- mean open/unresolved/rejected coverage: `{m['mean_open_coverage']}/{m['mean_unresolved_coverage']}/{m['mean_rejected_coverage']}`

## 10. v1.3 대비 변화

v1.3 대비 변화는 train-only comparison CSV에서 확인한다. 이 변화는 검토용 candidate 변화이며 성능 개선 확정이 아니다.

## 11. 남은 검토 포인트

high-confidence prediction이 실제 광고와 너무 멀어지지 않았는지, review-only bucket에 실제 광고가 과도하게 이동했는지 viewer/trace로 확인해야 한다.

## 12. validation/test 보류

validation/test는 이번 작업에서 실행하지 않았다. train 검토 후 validation 전용 실행으로 이동하고, test는 마지막에 한 번만 실행해야 한다.
"""
    write_text(root / config["output_paths"]["adjustment_note_md"], note)
    append_log(root / config["output_paths"]["run_log"], "[STEP 10] Generate report and summary")
    summary = f"""# State Machine Interval Detector v1.4 Train Summary

## 작업 개요

v1.4는 train-only adjustment candidate이며, v1.3을 보존한 상태로 high-confidence prediction과 review-only candidate를 분리했다.

## 핵심 수치

- train video count: `{m['train_video_count']}`
- v1.3 severe over-detection count: `{b.get('severe_over_detection_count')}`
- v1.4 severe over-detection count based on high-confidence: `{m['severe_over_detection_count_high_confidence']}`
- v1.3 closed/open/unresolved coverage: `{b.get('mean_closed_coverage')}` / `{b.get('mean_open_coverage')}` / `{b.get('mean_unresolved_coverage')}`
- v1.4 high-confidence/review/open/unresolved/rejected coverage: `{m['mean_high_confidence_coverage']}` / `{m['mean_review_only_coverage']}` / `{m['mean_open_coverage']}` / `{m['mean_unresolved_coverage']}` / `{m['mean_rejected_coverage']}`
- actual coverage: `{m['mean_actual_coverage']}`
- review-only candidate count: `{m['review_only_candidate_count']}`
- rejected long open count: `{m['rejected_long_open_count']}`
- interval confidence filter event count: `{m['interval_confidence_filter_event_count']}`
- hard/support/weak evidence counts: `{m['hard_evidence_trace_count']}` / `{m['support_evidence_trace_count']}` / `{m['weak_context_trace_count']}`

## 남은 문제

review-only와 rejected bucket은 prediction이 아니라 검토 대상이다. train에서 실제 광고가 review-only로 많이 이동했는지 확인해야 한다.

## 다음 단계

1. v1.4 train worst cases viewer/trace 검토
2. high-confidence prediction이 실제 광고와 너무 멀어지지 않았는지 확인
3. review-only candidate에 실제 광고가 많이 들어갔는지 확인
4. v1.4가 train에서 말이 되면 validation 전용 실행
5. test는 마지막에 한 번만 실행
"""
    write_text(root / config["output_paths"]["summary_md"], summary)


def baseline_v13(root: Path) -> dict[str, Any]:
    rows = read_csv(root / "data/analysis/v1_3_train_video_result_table_for_review.csv")
    n = len(rows) or 1
    return {"train_video_count": len(rows), "severe_over_detection_count": sum(1 for r in rows if r.get("severity_level") == "severe_over_detection"), "mean_actual_coverage": round(sum(as_float(r.get("actual_coverage_ratio")) for r in rows)/n,6), "mean_closed_coverage": round(sum(as_float(r.get("closed_coverage_ratio")) for r in rows)/n,6), "mean_open_coverage": round(sum(as_float(r.get("open_coverage_ratio")) for r in rows)/n,6), "mean_unresolved_coverage": round(sum(as_float(r.get("unresolved_coverage_ratio")) for r in rows)/n,6), "closed_prediction_count": sum(int(float(r.get("closed_prediction_count",0) or 0)) for r in rows), "open_interval_count": sum(int(float(r.get("open_interval_count",0) or 0)) for r in rows), "unresolved_long_open_count": sum(int(float(r.get("unresolved_long_open_count",0) or 0)) for r in rows)}


def scan_forbidden(directory: Path) -> list[str]:
    found = []
    for p in directory.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(directory)
        parts = {x.lower() for x in rel.parts}
        if p.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES or parts & FORBIDDEN_BUNDLE_PARTS:
            found.append(str(rel))
    return found


def refresh_latest(root: Path, config: dict[str, Any]) -> list[str]:
    append_log(root / config["output_paths"]["run_log"], "[STEP 12] Update latest bundle")
    outp = config["output_paths"]
    bundle = root / outp["latest_bundle"]
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    rels = [
        "configs/detectors/state_machine_interval_detector_v1_4_train_config.json",
        "scripts/detectors/run_state_machine_interval_detector_v1_4_train.py",
        outp["prediction_csv"], outp["review_only_interval_csv"], outp["trace_csv"], outp["open_interval_csv"], outp["unresolved_long_open_csv"], outp["rejected_long_open_csv"], outp["disclosure_notice_review_csv"], outp["evidence_tier_events_csv"], outp["fresh_evidence_timeout_events_csv"], outp["interval_confidence_filter_events_csv"], outp["audit_csv"], outp["comparison_csv"], outp["analysis_video_summary_csv"], outp["analysis_interval_overlap_csv"], outp["analysis_open_summary_csv"], outp["analysis_trace_reason_summary_csv"], outp["analysis_worst_cases_csv"], outp["analysis_rule_issue_candidates_csv"], outp["summary_md"], outp["report_json"], outp["adjustment_note_md"], outp["run_log"],
    ]
    for rel in rels:
        src = root / rel
        if src.exists():
            shutil.copy2(src, bundle / src.name)
    forbidden = scan_forbidden(bundle)
    readme = "# v1.4 Train-only Detector Adjustment Candidate Latest Files\n\n- v1.4 train-only adjustment candidate.\n- Not final detector.\n- v1.3 preserved.\n- High-confidence prediction and review-only candidate separated.\n- Validation/test not executed.\n- Test row-level output excluded.\n- No media/frame/cache/model/raw video copied.\n\n## Files\n"
    for p in sorted(bundle.iterdir()):
        if p.is_file():
            readme += f"- {p.name}\n"
    readme += "- README_latest_files.md\n"
    write_text(bundle / "README_latest_files.md", readme)
    if forbidden:
        raise RuntimeError(f"Forbidden files in latest bundle: {forbidden}")
    return forbidden


def input_modified(root: Path, config: dict[str, Any]) -> list[str]:
    stats_path = root / "reports/detectors/state_machine_interval_detector_v1_4_train_input_file_stats_before.json"
    if not stats_path.exists():
        return []
    before = json.loads(stats_path.read_text(encoding="utf-8"))
    changed = []
    for rel, old in before.items():
        p = root / rel
        now = file_stat(p)
        if now != old:
            changed.append(rel)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=os.environ.get("YASD_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))
    parser.add_argument("--config", default="configs/detectors/state_machine_interval_detector_v1_4_train_config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    log_path = root / config["output_paths"]["run_log"]
    append_log(log_path, "[STEP 03] Implement v1.4 train detector script")
    append_log(log_path, f"[STEP 03] script ready: scripts/detectors/run_state_machine_interval_detector_v1_4_train.py")
    if args.no_run:
        append_log(log_path, "[STEP 03] no-run requested; detector execution skipped")
        return 0
    try:
        det = run_detector(root, config, log_path, dry_run=args.dry_run)
        if args.dry_run:
            print(json.dumps(det, ensure_ascii=False, indent=2))
            return 0
        analysis = build_audit(root, config, det, log_path)
        comparison = build_comparison(root, config, analysis)
        v13 = baseline_v13(root)
        v14_metrics = summarize_metrics(analysis["video_rows"], det)
        old_after = root / "reports/detectors/old_project_snapshot_after_state_machine_detector_v1_4_train.tsv"
        snapshot_tree(Path("./_old_project_not_included"), old_after)
        old_before = root / "reports/detectors/old_project_snapshot_before_state_machine_detector_v1_4_train.tsv"
        old_modified = old_before.exists() and old_after.exists() and old_before.read_text(encoding="utf-8") != old_after.read_text(encoding="utf-8")
        latest_forbidden = []
        report = {"task_name": "state_machine_interval_detector_v1_4_train", "version": VERSION, "base_version": BASE_VERSION, "project_root": str(root), "run_scope": "train_only", "input_files": config["input_paths"], "output_files": config["output_paths"], "train_video_ids": EXPECTED_SPLITS["train"], "excluded_validation_video_ids": EXPECTED_SPLITS["validation"], "excluded_test_video_ids": EXPECTED_SPLITS["test"], "v1_3_baseline_metrics": v13, "v1_4_metrics": v14_metrics, "v1_3_vs_v1_4_comparison_summary": dict(Counter(r["got_better_or_worse"] for r in comparison)), "rule_deltas": config["adjustment_scope"], "decision_feature_columns": DECISION_FEATURE_COLUMNS, "forbidden_decision_columns_check": banned_decision_columns(DECISION_FEATURE_COLUMNS), "event_counts": {"evidence_tier_events": len(det["evidence_rows"]), "fresh_evidence_timeout_events": len(det["fresh_rows"]), "interval_confidence_filter_events": len(det["filter_rows"])}, "severe_over_detection_count": v14_metrics["severe_over_detection_count_high_confidence"], "worst_videos": v14_metrics["worst_videos"], "sub_agent_results": [], "warnings": det["warnings"], "errors": det["errors"], "old_project_modified": old_modified, "input_files_modified": input_modified(root, config), "latest_for_chatgpt_forbidden_files_found": latest_forbidden, "no_validation_run": True, "no_test_run": True, "no_feature_extraction": True, "no_threshold_tuning": True, "no_final_performance_claim": True, "validation_output_count": 0, "test_row_level_output_count": 0, "backup_dir": config.get("metadata", {}).get("backup_dir", "")}
        write_docs(root, config, report)
        write_text(root / config["output_paths"]["report_json"], json.dumps(report, ensure_ascii=False, indent=2))
        latest_forbidden = refresh_latest(root, config)
        report["latest_for_chatgpt_forbidden_files_found"] = latest_forbidden
        write_text(root / config["output_paths"]["report_json"], json.dumps(report, ensure_ascii=False, indent=2))
        # 최신 scan 결과가 반영된 뒤 final report를 복사한다.
        bundle = root / config["output_paths"]["latest_bundle"]
        shutil.copy2(root / config["output_paths"]["report_json"], bundle / Path(config["output_paths"]["report_json"]).name)
        shutil.copy2(root / config["output_paths"]["run_log"], bundle / Path(config["output_paths"]["run_log"]).name)
        append_log(log_path, "[STEP 13] Final human-readable summary")
        status = "SUCCESS" if not report["errors"] and not report["input_files_modified"] and not old_modified else "CONDITIONAL_SUCCESS"
        print("작업 상태:", status)
        print("v1.4 detector config path:", root / "configs/detectors/state_machine_interval_detector_v1_4_train_config.json")
        print("v1.4 detector script path:", root / "scripts/detectors/run_state_machine_interval_detector_v1_4_train.py")
        print("v1.4 train high-confidence prediction row count:", len(det["pred_rows"]))
        print("v1.4 review-only closed candidate count:", len(det["review_rows"]))
        print("v1.4 train trace row count:", len(det["trace_rows"]))
        print("v1.4 open/unresolved/rejected count:", len(det["open_rows"]), len(det["unresolved_rows"]), len(det["rejected_rows"]))
        print("interval confidence filter event count:", len(det["filter_rows"]))
        print("hard/support/weak evidence counts:", v14_metrics["hard_evidence_trace_count"], v14_metrics["support_evidence_trace_count"], v14_metrics["weak_context_trace_count"])
        print("v1.3 vs v1.4 severe over-detection count:", v13.get("severe_over_detection_count"), "->", v14_metrics["severe_over_detection_count_high_confidence"])
        print("v1.3 coverage closed/open/unresolved:", v13.get("mean_closed_coverage"), v13.get("mean_open_coverage"), v13.get("mean_unresolved_coverage"))
        print("v1.4 coverage high/review/open/unresolved/rejected:", v14_metrics["mean_high_confidence_coverage"], v14_metrics["mean_review_only_coverage"], v14_metrics["mean_open_coverage"], v14_metrics["mean_unresolved_coverage"], v14_metrics["mean_rejected_coverage"])
        print("worst videos:", v14_metrics["worst_videos"])
        print("old_project_modified:", old_modified)
        print("input_data_files_modified:", report["input_files_modified"])
        print("validation_output_count: 0")
        print("test_row_level_output_count: 0")
        print("latest bundle path:", root / config["output_paths"]["latest_bundle"])
        print("warnings/errors:", report["warnings"], report["errors"])
        print("다음 단계: v1.4 train worst cases를 viewer/trace로 확인하고, validation/test는 아직 실행하지 말 것")
        return 0
    except Exception as exc:
        append_log(log_path, f"[FAILURE] {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
