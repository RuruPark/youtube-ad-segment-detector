#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(".")
VERSION = "v2.1a"
DETECTOR_ID = "state_machine_interval_detector_v2_1a_experimental"
DEV_IDS = {1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15}

FORBIDDEN_PATTERNS = [
    "label",
    "true",
    "actual",
    "gt",
    "ground_truth",
    "audit",
    "nearest_true_boundary",
    "distance_to_nearest_true_boundary",
    "is_near_true_boundary",
    "ad_overlap",
    "is_ad_overlap",
    "is_ad_core",
    "is_clean_nonad",
    "overlapping_ad_interval_ids",
    "per_video_ad_vs_nonad_contrast_score",
    "audio_candidate_score_for_discussion",
]

DECISION_FEATURE_COLUMNS = [
    "scene_anchor_role",
    "scene_transition_reliability_score",
    "ocr_hard_disclosure_subtype_count_v2_1a",
    "ocr_fuzzy_disclosure_count_v2_1a",
    "ocr_negative_guard_count_v2_1a",
    "ocr_product_cta_link_count_v2_1a",
    "ocr_product_cta_link_post_increase_flag_v2_1a",
    "ocr_timeline_recent_hard_evidence_flag",
    "ocr_start_signal_level_v2_0",
    "ocr_end_signal_level_v2_0",
    "ocr_context_level_v2_0",
    "ocr_context_reliability_level",
    "audio_pre_relative_active_score",
    "audio_post_relative_active_score",
    "audio_context_relative_active_score",
    "audio_pre_relative_quiet_score",
    "audio_post_relative_quiet_score",
    "audio_context_relative_quiet_score",
    "audio_relative_local_shift_score",
    "audio_relative_sustained_context_score",
    "audio_start_signal_level_v2_0",
    "audio_end_signal_level_v2_0",
    "audio_context_level_v2_0",
    "audio_not_informative_flag",
    "opening_disclosure_guard_applied",
    "opening_disclosure_confirmed_by_later_product_cta",
    "hard_evidence_flag_v2_1a",
    "start_support_flag",
    "continuity_support_flag",
    "end_support_flag",
    "weak_context_flag_v2_1a",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return default if math.isnan(out) else out
    except Exception:
        return default


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def has_forbidden(columns: list[str]) -> list[str]:
    bad: list[str] = []
    for col in columns:
        low = col.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern == "gt":
                if low == "gt" or low.startswith("gt_") or low.endswith("_gt") or "_gt_" in low:
                    bad.append(col)
                    break
            elif pattern in low:
                bad.append(col)
                break
    return sorted(set(bad))


def parse_json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return []


def disclosure_window_sec(video_duration: float) -> float:
    return min(45.0, max(30.0, 0.02 * video_duration))


def sum_cols(row: dict[str, Any], names: list[str]) -> float:
    return sum(fnum(row.get(name)) for name in names)


def level_active(value: Any) -> bool:
    return str(value).strip().lower() in {"medium", "high"}


def build_ocr_restoration_map(frame_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    restored: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in frame_rows:
        try:
            vid = int(float(row.get("video_id") or 0))
        except Exception:
            continue
        if vid not in DEV_IDS:
            continue
        anchor = row.get("nearest_anchor_id") or (row.get("anchor_ids_joined") or "").split(";")[0]
        if not anchor:
            continue
        hit_count = fnum(row.get("corrected_ad_disclosure_hit_count") or row.get("ad_disclosure_keyword_count"))
        if hit_count <= 0:
            continue
        confidence = (row.get("matched_keyword_confidence") or row.get("matched_keyword_rules") or "").lower()
        if "typo" in confidence:
            restored[anchor]["typo"] += hit_count
        elif "proximity" in confidence or "near" in confidence:
            restored[anchor]["proximity"] += hit_count
        elif "fuzzy" in confidence:
            restored[anchor]["fuzzy"] += hit_count
        elif "exact" in confidence or confidence:
            restored[anchor]["exact"] += hit_count
        else:
            restored[anchor]["unknown"] += hit_count
    return {anchor: dict(counts) for anchor, counts in restored.items()}


def enhance_rows_with_v21a_flags(
    fusion_rows: list[dict[str, str]],
    restoration_map: dict[str, dict[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str, list[str]]:
    restoration_map = restoration_map or {}
    enhanced: list[dict[str, Any]] = []
    support_audit: list[dict[str, Any]] = []
    opening_audit: list[dict[str, Any]] = []
    subtype_audit: list[dict[str, Any]] = []
    warnings: list[str] = []

    subtype_cols = [
        "corrected_pre_ad_disclosure_exact_count",
        "corrected_post_ad_disclosure_exact_count",
        "corrected_pre_ad_disclosure_typo_count",
        "corrected_post_ad_disclosure_typo_count",
        "corrected_pre_ad_disclosure_proximity_count",
        "corrected_post_ad_disclosure_proximity_count",
        "corrected_pre_ad_disclosure_fuzzy_review_count",
        "corrected_post_ad_disclosure_fuzzy_review_count",
    ]
    has_subtype_cols = all(col in (fusion_rows[0] if fusion_rows else {}) for col in subtype_cols)
    subtype_sum = sum(sum_cols(row, subtype_cols) for row in fusion_rows) if has_subtype_cols else 0.0
    restoration_status = "restored"
    if not has_subtype_cols or subtype_sum == 0:
        if restoration_map:
            restoration_status = "restored"
            warnings.append("Fusion subtype columns were missing/zero; OCR frame aggregate restoration was used where anchor matches existed.")
        else:
            restoration_status = "fallback"
            warnings.append("Fusion subtype columns were missing/zero and OCR subtype restoration was unavailable; subtype_missing_fallback mode used.")
    else:
        warnings.append("OCR disclosure subtype counts were already present in v2.0 fusion input; restoration audit uses fusion subtype counts.")

    for row in fusion_rows:
        out = dict(row)
        vid = int(float(row.get("video_id") or 0))
        t = fnum(row.get("candidate_time_sec"))
        video_duration = fnum(row.get("video_duration_sec"))
        anchor_id = row.get("final_scene_anchor_id", "")
        restored = restoration_map.get(anchor_id, {})

        pre_exact = fnum(row.get("corrected_pre_ad_disclosure_exact_count"))
        post_exact = fnum(row.get("corrected_post_ad_disclosure_exact_count"))
        pre_typo = fnum(row.get("corrected_pre_ad_disclosure_typo_count"))
        post_typo = fnum(row.get("corrected_post_ad_disclosure_typo_count"))
        pre_prox = fnum(row.get("corrected_pre_ad_disclosure_proximity_count"))
        post_prox = fnum(row.get("corrected_post_ad_disclosure_proximity_count"))
        pre_fuzzy = fnum(row.get("corrected_pre_ad_disclosure_fuzzy_review_count"))
        post_fuzzy = fnum(row.get("corrected_post_ad_disclosure_fuzzy_review_count"))
        if (pre_exact + post_exact + pre_typo + post_typo + pre_prox + post_prox + pre_fuzzy + post_fuzzy) == 0 and restored:
            post_exact = restored.get("exact", 0.0)
            post_typo = restored.get("typo", 0.0)
            post_prox = restored.get("proximity", 0.0)
            post_fuzzy = restored.get("fuzzy", 0.0) + restored.get("unknown", 0.0)

        hard_subtype_count = pre_exact + post_exact + pre_typo + post_typo + pre_prox + post_prox
        fuzzy_count = pre_fuzzy + post_fuzzy
        post_hard_subtype_count = post_exact + post_typo + post_prox
        pre_hard_subtype_count = pre_exact + pre_typo + pre_prox
        negative_guard_count = sum_cols(
            row,
            ["corrected_pre_negative_guard_count", "corrected_post_negative_guard_count"],
        )
        pre_product_cta_link = sum_cols(
            row,
            [
                "corrected_pre_brand_product_count",
                "corrected_pre_purchase_cta_count",
                "corrected_pre_link_more_info_count",
                "corrected_pre_promotion_discount_count",
                "corrected_pre_sponsor_count",
            ],
        )
        post_product_cta_link = sum_cols(
            row,
            [
                "corrected_post_brand_product_count",
                "corrected_post_purchase_cta_count",
                "corrected_post_link_more_info_count",
                "corrected_post_promotion_discount_count",
                "corrected_post_sponsor_count",
            ],
        )
        product_cta_link_total = pre_product_cta_link + post_product_cta_link
        product_cta_types = sum(
            1
            for name in [
                "brand_product",
                "purchase_cta",
                "link_more_info",
                "promotion_discount",
                "sponsor",
            ]
            if fnum(row.get(f"corrected_pre_{name}_count")) + fnum(row.get(f"corrected_post_{name}_count")) > 0
        )
        product_or_cta_post_increase = post_product_cta_link > pre_product_cta_link
        disclosure_post_increase = post_hard_subtype_count > pre_hard_subtype_count
        relative_audio_active = (
            level_active(row.get("audio_start_signal_level_v2_0"))
            or level_active(row.get("audio_context_level_v2_0"))
            or fnum(row.get("audio_post_relative_active_score")) >= 0.55
            or fnum(row.get("audio_context_relative_active_score")) >= 0.55
        )
        relative_audio_quiet = (
            level_active(row.get("audio_end_signal_level_v2_0"))
            or fnum(row.get("audio_post_relative_quiet_score")) >= 0.55
            or fnum(row.get("audio_context_relative_quiet_score")) >= 0.55
        )
        timeline_hard = truth(row.get("ocr_timeline_recent_hard_evidence_flag"))
        ocr_context_level = str(row.get("ocr_context_level_v2_0") or "").lower()
        ocr_context_reliable = str(row.get("ocr_context_reliability_level") or "").lower() in {"medium", "high"}
        context_valid_ratio = fnum(row.get("ocr_context_valid_frame_ratio"))
        coverage_high_low_ocr = context_valid_ratio >= 0.65 and ocr_context_level in {"none", "low", ""}
        opening_window = disclosure_window_sec(video_duration)
        in_opening = t <= opening_window or truth(row.get("is_in_opening_disclosure_window"))
        confirmed_later = truth(row.get("opening_disclosure_confirmed_by_later_product_cta"))
        opening_notice = truth(row.get("opening_disclosure_notice_flag")) or truth(row.get("opening_disclosure_guard_applied"))
        opening_disclosure_only = bool(in_opening and (hard_subtype_count > 0 or opening_notice) and not confirmed_later)
        fuzzy_only = fuzzy_count > 0 and hard_subtype_count == 0
        negative_suppressed = negative_guard_count > 0

        exact_typo_prox_with_context = hard_subtype_count > 0 and (
            product_cta_link_total > 0 or timeline_hard or disclosure_post_increase or product_or_cta_post_increase
        )
        hard_evidence = (
            (truth(row.get("hard_evidence_flag_v2_0")) or exact_typo_prox_with_context)
            and not fuzzy_only
            and not negative_suppressed
            and not opening_disclosure_only
        )

        weak_disclosure_with_product = (
            (fuzzy_count > 0 or hard_subtype_count > 0)
            and product_cta_link_total > 0
            and not hard_evidence
            and not negative_suppressed
            and not opening_disclosure_only
        )
        ocr_post_increase_audio = (
            (disclosure_post_increase or product_or_cta_post_increase)
            and relative_audio_active
            and not hard_evidence
            and not negative_suppressed
            and not opening_disclosure_only
        )
        anchor_after_product_cta_increase = (
            product_or_cta_post_increase
            and product_cta_link_total > 0
            and not hard_evidence
            and not negative_suppressed
            and not opening_disclosure_only
        )
        timeline_recent_start = (
            timeline_hard
            and product_cta_link_total > 0
            and not hard_evidence
            and not negative_suppressed
            and not opening_disclosure_only
        )
        start_support = bool(
            weak_disclosure_with_product
            or ocr_post_increase_audio
            or anchor_after_product_cta_increase
            or timeline_recent_start
        )

        continuity_support = bool(
            product_cta_link_total >= 2
            or (product_cta_link_total > 0 and relative_audio_active)
            or (ocr_context_reliable and ocr_context_level in {"medium", "high"})
            or (timeline_hard and not opening_disclosure_only and not negative_suppressed)
        )
        ad_text_drop = (pre_hard_subtype_count + pre_product_cta_link) > (post_hard_subtype_count + post_product_cta_link + 1)
        no_recent_ocr_timeline_hard = not timeline_hard and product_cta_link_total == 0 and hard_subtype_count == 0
        ocr_end_context_available = bool(
            ad_text_drop
            or coverage_high_low_ocr
            or hard_subtype_count > 0
            or product_cta_link_total > 0
            or timeline_hard
            or ocr_context_reliable
        )
        audio_alone_end_support = bool(relative_audio_quiet and not ocr_end_context_available)
        audio_quiet_with_ocr_context = bool(relative_audio_quiet and ocr_end_context_available)
        end_support = bool(
            ad_text_drop
            or coverage_high_low_ocr
            or audio_quiet_with_ocr_context
            or no_recent_ocr_timeline_hard
        )
        weak_context = bool(truth(row.get("weak_context_flag_v2_0")) or fuzzy_only or opening_disclosure_only)
        ocr_any = hard_subtype_count + fuzzy_count + product_cta_link_total > 0 or timeline_hard
        audio_only_support = bool(relative_audio_active and not ocr_any)
        scene_only_support = bool(
            not hard_evidence
            and not start_support
            and not continuity_support
            and not end_support
            and fnum(row.get("scene_transition_reliability_score")) > 0
        )

        start_reasons = []
        if weak_disclosure_with_product:
            start_reasons.append("weak_disclosure_with_product_cta_link")
        if ocr_post_increase_audio:
            start_reasons.append("ocr_post_increase_with_relative_audio_active")
        if anchor_after_product_cta_increase:
            start_reasons.append("anchor_after_product_cta_link_increase")
        if timeline_recent_start:
            start_reasons.append("ocr_timeline_recent_hard_evidence_near_start")
        continuity_reasons = []
        if product_cta_link_total >= 2:
            continuity_reasons.append("product_brand_cta_link_repetition")
        if product_cta_link_total > 0 and relative_audio_active:
            continuity_reasons.append("relative_audio_active_with_ocr_support")
        if ocr_context_reliable and ocr_context_level in {"medium", "high"}:
            continuity_reasons.append("ocr_timeline_medium_high_context")
        if timeline_hard and not opening_disclosure_only and not negative_suppressed:
            continuity_reasons.append("recent_ocr_hard_timeline")
        end_reasons = []
        if ad_text_drop:
            end_reasons.append("ocr_ad_text_drop")
        if coverage_high_low_ocr:
            end_reasons.append("ocr_coverage_high_ocr_low")
        if relative_audio_quiet:
            end_reasons.append("relative_quiet_low_energy_shift")
        if no_recent_ocr_timeline_hard:
            end_reasons.append("no_recent_ocr_timeline_hard_evidence")

        out.update(
            {
                "ocr_disclosure_exact_count_v2_1a": pre_exact + post_exact,
                "ocr_disclosure_typo_count_v2_1a": pre_typo + post_typo,
                "ocr_disclosure_proximity_count_v2_1a": pre_prox + post_prox,
                "ocr_fuzzy_disclosure_count_v2_1a": fuzzy_count,
                "ocr_hard_disclosure_subtype_count_v2_1a": hard_subtype_count,
                "ocr_post_hard_disclosure_subtype_count_v2_1a": post_hard_subtype_count,
                "ocr_negative_guard_count_v2_1a": negative_guard_count,
                "ocr_product_cta_link_count_v2_1a": product_cta_link_total,
                "ocr_product_cta_timeline_type_count_v2_1a": product_cta_types + int(timeline_hard),
                "ocr_disclosure_post_increase_flag_v2_1a": disclosure_post_increase,
                "ocr_product_cta_link_post_increase_flag_v2_1a": product_or_cta_post_increase,
                "relative_audio_active_flag_v2_1a": relative_audio_active,
                "relative_audio_quiet_flag_v2_1a": relative_audio_quiet,
                "audio_alone_end_support_flag_v2_1a": audio_alone_end_support,
                "opening_disclosure_window_sec_v2_1a": opening_window,
                "opening_disclosure_only_flag_v2_1a": opening_disclosure_only,
                "negative_guard_suppressed_disclosure_flag_v2_1a": negative_suppressed,
                "fuzzy_only_disclosure_flag_v2_1a": fuzzy_only,
                "hard_evidence_flag_v2_1a": hard_evidence,
                "start_support_flag": start_support,
                "continuity_support_flag": continuity_support,
                "end_support_flag": end_support,
                "weak_context_flag_v2_1a": weak_context,
                "audio_only_support_flag_v2_1a": audio_only_support,
                "scene_only_support_flag_v2_1a": scene_only_support,
                "start_support_reasons_v2_1a": ";".join(start_reasons),
                "continuity_support_reasons_v2_1a": ";".join(continuity_reasons),
                "end_support_reasons_v2_1a": ";".join(end_reasons),
                "decision_feature_columns_json_v2_1a": json.dumps(DECISION_FEATURE_COLUMNS, ensure_ascii=False),
                "forbidden_decision_columns_found_v2_1a": json.dumps(has_forbidden(DECISION_FEATURE_COLUMNS), ensure_ascii=False),
                "scene_used_for_ad_likelihood_directly_v2_1a": False,
            }
        )
        enhanced.append(out)

        subtype_audit.append(
            {
                "video_id": vid,
                "final_scene_anchor_id": anchor_id,
                "candidate_time_sec": f"{t:.6f}",
                "fusion_subtype_available": has_subtype_cols,
                "restoration_map_available": bool(restored),
                "exact_count": pre_exact + post_exact,
                "typo_count": pre_typo + post_typo,
                "proximity_count": pre_prox + post_prox,
                "fuzzy_review_count": fuzzy_count,
                "negative_guard_count": negative_guard_count,
                "fuzzy_only_hard_evidence_allowed": False,
                "negative_guard_suppressed": negative_suppressed,
                "restoration_status": restoration_status,
            }
        )
        support_audit.append(
            {
                "video_id": vid,
                "final_scene_anchor_id": anchor_id,
                "candidate_time_sec": f"{t:.6f}",
                "hard_evidence_flag_v2_1a": hard_evidence,
                "start_support_flag": start_support,
                "continuity_support_flag": continuity_support,
                "end_support_flag": end_support,
                "weak_context_flag_v2_1a": weak_context,
                "start_support_reasons_v2_1a": ";".join(start_reasons),
                "continuity_support_reasons_v2_1a": ";".join(continuity_reasons),
                "end_support_reasons_v2_1a": ";".join(end_reasons),
                "audio_only_support_flag_v2_1a": audio_only_support,
                "scene_only_support_flag_v2_1a": scene_only_support,
                "end_support_allowed_to_enter_start_pending": False,
                "audio_support_alone_allowed_to_enter_start_pending": False,
                "audio_support_alone_allowed_to_confirm_end": False,
                "audio_alone_end_support_flag_v2_1a": audio_alone_end_support,
                "scene_only_allowed_to_enter_start_pending": False,
            }
        )
        if in_opening or opening_notice or opening_disclosure_only:
            opening_audit.append(
                {
                    "video_id": vid,
                    "final_scene_anchor_id": anchor_id,
                    "candidate_time_sec": f"{t:.6f}",
                    "video_duration_sec": f"{video_duration:.6f}",
                    "opening_disclosure_window_sec": f"{opening_window:.6f}",
                    "is_in_opening_disclosure_window": in_opening,
                    "opening_disclosure_notice_flag": opening_notice,
                    "opening_disclosure_confirmed_by_later_product_cta": confirmed_later,
                    "opening_disclosure_only_flag_v2_1a": opening_disclosure_only,
                    "hard_start_allowed": False if opening_disclosure_only else hard_evidence,
                    "guard_action": "opening_notice_review_only" if opening_disclosure_only else "not_opening_only",
                }
            )

    if not opening_audit:
        for row in enhanced[: min(100, len(enhanced))]:
            opening_audit.append(
                {
                    "video_id": row.get("video_id"),
                    "final_scene_anchor_id": row.get("final_scene_anchor_id"),
                    "candidate_time_sec": row.get("candidate_time_sec"),
                    "video_duration_sec": row.get("video_duration_sec"),
                    "opening_disclosure_window_sec": f"{fnum(row.get('opening_disclosure_window_sec_v2_1a')):.6f}",
                    "is_in_opening_disclosure_window": fnum(row.get("candidate_time_sec")) <= fnum(row.get("opening_disclosure_window_sec_v2_1a")),
                    "opening_disclosure_notice_flag": False,
                    "opening_disclosure_confirmed_by_later_product_cta": False,
                    "opening_disclosure_only_flag_v2_1a": False,
                    "hard_start_allowed": truth(row.get("hard_evidence_flag_v2_1a")),
                    "guard_action": "candidate_audit_no_opening_disclosure_guard_rows_detected",
                }
            )
        warnings.append("Opening disclosure guard count was zero; candidate audit rows were emitted for inspection.")

    return enhanced, subtype_audit, opening_audit, support_audit, restoration_status, warnings


def candidate_to_row(config: dict[str, Any], candidate: dict[str, Any], status: str, idx: int) -> dict[str, Any]:
    duration = max(0.0, candidate.get("end_sec", 0.0) - candidate.get("start_sec", 0.0))
    return {
        "candidate_id": candidate.get("candidate_id") or f"{config['config_id']}_C{idx:05d}",
        "config_id": config["config_id"],
        "config_family": config["config_family"],
        "version": VERSION,
        "detector_id": DETECTOR_ID,
        "original_split_v2_4": "train",
        "split_role_v2_5": "development",
        "evaluation_subset_v2_5": "none",
        "video_id": candidate.get("video_id"),
        "video_duration_sec": f"{candidate.get('video_duration_sec', 0.0):.6f}",
        "ad_start_sec": f"{candidate.get('start_sec', 0.0):.6f}",
        "ad_end_sec": f"{candidate.get('end_sec', 0.0):.6f}",
        "ad_duration_sec": f"{duration:.6f}",
        "start_anchor_id": candidate.get("start_anchor_id"),
        "end_anchor_id": candidate.get("end_anchor_id"),
        "start_reason": candidate.get("start_reason"),
        "end_reason": candidate.get("end_reason"),
        "start_confidence_level": candidate.get("start_confidence_level"),
        "end_confidence_level": candidate.get("end_confidence_level"),
        "hard_evidence_count": candidate.get("hard_evidence_count", 0),
        "start_support_count": candidate.get("start_support_count", 0),
        "continuity_support_count": candidate.get("continuity_support_count", 0),
        "end_support_count": candidate.get("end_support_count", 0),
        "support_evidence_count": candidate.get("support_evidence_count", 0),
        "weak_context_count": candidate.get("weak_context_count", 0),
        "ocr_hard_count": candidate.get("ocr_hard_count", 0),
        "ocr_timeline_recent_hard_count": candidate.get("ocr_timeline_recent_hard_count", 0),
        "product_cta_timeline_types": candidate.get("product_cta_timeline_types", 0),
        "audio_relative_support_count": candidate.get("audio_relative_support_count", 0),
        "hard_evidence_density_per_60s": f"{candidate.get('hard_evidence_density_per_60s', 0.0):.6f}",
        "max_weak_span_sec": f"{candidate.get('max_weak_span_sec', 0.0):.6f}",
        "interval_status": status,
        "interval_ad_score": f"{candidate.get('interval_ad_score', 0.0):.6f}",
        "interval_score_density": f"{candidate.get('interval_score_density', 0.0):.6f}",
        "video_relative_rank_score": f"{candidate.get('video_relative_rank_score', 0.0):.6f}",
        "start_strength_score": f"{candidate.get('start_strength_score', 0.0):.6f}",
        "continuity_strength_score": f"{candidate.get('continuity_strength_score', 0.0):.6f}",
        "end_quality_score": f"{candidate.get('end_quality_score', 0.0):.6f}",
        "hard_evidence_density_score": f"{candidate.get('hard_evidence_density_score', 0.0):.6f}",
        "ocr_timeline_consistency_score": f"{candidate.get('ocr_timeline_consistency_score', 0.0):.6f}",
        "audio_relative_support_score": f"{candidate.get('audio_relative_support_score', 0.0):.6f}",
        "weak_span_penalty": f"{candidate.get('weak_span_penalty', 0.0):.6f}",
        "overlong_penalty": f"{candidate.get('overlong_penalty', 0.0):.6f}",
        "opening_disclosure_only_penalty": f"{candidate.get('opening_disclosure_only_penalty', 0.0):.6f}",
        "fuzzy_only_penalty": f"{candidate.get('fuzzy_only_penalty', 0.0):.6f}",
        "audio_only_penalty": f"{candidate.get('audio_only_penalty', 0.0):.6f}",
        "scene_only_penalty": f"{candidate.get('scene_only_penalty', 0.0):.6f}",
        "duration_excess_penalty": f"{candidate.get('duration_excess_penalty', 0.0):.6f}",
        "long_candidate_low_density_penalty": f"{candidate.get('long_candidate_low_density_penalty', 0.0):.6f}",
        "video_prediction_ratio_before_budget_guard": f"{candidate.get('video_prediction_ratio_before_budget_guard', 0.0):.6f}",
        "video_prediction_ratio_after_budget_guard": f"{candidate.get('video_prediction_ratio_after_budget_guard', 0.0):.6f}",
        "budget_guard_action": candidate.get("budget_guard_action", ""),
        "ultra_high_confidence": str(bool(candidate.get("ultra_high_confidence"))).lower(),
        "failure_or_review_reason": candidate.get("failure_or_review_reason", ""),
        "decision_feature_columns_json": json.dumps(DECISION_FEATURE_COLUMNS, ensure_ascii=False),
        "forbidden_decision_columns_found": json.dumps(has_forbidden(DECISION_FEATURE_COLUMNS), ensure_ascii=False),
        "audit_columns_used_for_decision": "false",
        "actual_label_used_for_decision": "false",
    }


def score_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    duration = max(1.0, candidate["end_sec"] - candidate["start_sec"])
    anchor_count = max(1, candidate.get("anchor_count", 1))
    hard_count = candidate.get("hard_evidence_count", 0)
    hard_density = hard_count / max(1.0, duration / 60.0)
    start_strength = clip(
        (1.0 if candidate.get("start_confirmed_by_hard") else 0.62)
        + min(0.18, 0.04 * candidate.get("start_support_count", 0))
    )
    continuity_strength = clip(
        (
            hard_count
            + 0.50 * candidate.get("continuity_support_count", 0)
            + 0.30 * candidate.get("start_support_count", 0)
        )
        / anchor_count
    )
    end_reason = candidate.get("end_reason", "")
    if "end_support" in end_reason:
        end_quality = 0.86
    elif "freshness" in end_reason:
        end_quality = 0.68
    elif "support_cap" in end_reason:
        end_quality = 0.58
    else:
        end_quality = 0.35
    hard_density_score = clip(hard_density / 4.0)
    timeline_score = clip(
        (candidate.get("ocr_timeline_recent_hard_count", 0) + 0.50 * candidate.get("ocr_hard_count", 0))
        / max(1.0, anchor_count)
    )
    audio_score = clip(candidate.get("audio_relative_support_count", 0) / anchor_count)

    weights = config["weights"]
    weight_sum = sum(float(v) for v in weights.values()) or 1.0
    base = (
        weights["start_strength_score"] * start_strength
        + weights["continuity_strength_score"] * continuity_strength
        + weights["end_quality_score"] * end_quality
        + weights["hard_evidence_density_score"] * hard_density_score
        + weights["ocr_timeline_consistency_score"] * timeline_score
        + weights["audio_relative_support_score"] * audio_score
    ) / weight_sum

    max_weak = candidate.get("max_weak_span_sec", 0.0)
    weak_penalty = config["penalties"]["weak_span_penalty"] if max_weak > config["thresholds"]["max_weak_span_sec"] else 0.0
    long_low_density = duration >= config["long_candidate"]["long_candidate_duration_sec"] and hard_density_score < config["long_candidate"]["long_candidate_density_min"]
    overlong_penalty = config["penalties"]["overlong_penalty"] if duration >= config["long_candidate"]["long_candidate_duration_sec"] else 0.0
    opening_penalty = config["penalties"]["opening_disclosure_only_penalty"] if candidate.get("opening_disclosure_only") else 0.0
    fuzzy_penalty = config["penalties"]["fuzzy_only_penalty"] if candidate.get("fuzzy_only") else 0.0
    audio_penalty = config["penalties"]["audio_only_penalty"] if candidate.get("audio_only") else 0.0
    scene_penalty = config["penalties"]["scene_only_penalty"] if candidate.get("scene_only") else 0.0
    duration_excess_penalty = (
        config["penalties"]["duration_excess_penalty"]
        if duration > max(240.0, 0.18 * candidate.get("video_duration_sec", 0.0))
        else 0.0
    )
    long_density_penalty = config["penalties"]["long_candidate_low_density_penalty"] if long_low_density else 0.0
    interval_score = clip(
        base
        - weak_penalty
        - overlong_penalty
        - opening_penalty
        - fuzzy_penalty
        - audio_penalty
        - scene_penalty
        - duration_excess_penalty
        - long_density_penalty
    )
    score_density = clip(interval_score / max(1.0, duration / 60.0))
    rank_score = clip(0.65 * interval_score + 0.35 * score_density)

    ultra_cfg = config["ultra_high_confidence"]
    ultra = (
        interval_score >= ultra_cfg["ultra_min_interval_ad_score"]
        and score_density >= ultra_cfg["ultra_min_interval_score_density"]
        and hard_count >= ultra_cfg["ultra_min_hard_evidence_count"]
        and hard_density >= ultra_cfg["ultra_min_hard_evidence_density_per_60s"]
        and candidate.get("ocr_hard_count", 0) >= ultra_cfg["ultra_min_ocr_hard_count"]
        and candidate.get("product_cta_timeline_types", 0) >= ultra_cfg["ultra_min_product_cta_timeline_types"]
        and not candidate.get("opening_disclosure_only")
        and not candidate.get("fuzzy_only")
        and not candidate.get("audio_only")
        and not candidate.get("scene_only")
    )

    candidate.update(
        {
            "start_strength_score": start_strength,
            "continuity_strength_score": continuity_strength,
            "end_quality_score": end_quality,
            "hard_evidence_density_score": hard_density_score,
            "ocr_timeline_consistency_score": timeline_score,
            "audio_relative_support_score": audio_score,
            "weak_span_penalty": weak_penalty,
            "overlong_penalty": overlong_penalty,
            "opening_disclosure_only_penalty": opening_penalty,
            "fuzzy_only_penalty": fuzzy_penalty,
            "audio_only_penalty": audio_penalty,
            "scene_only_penalty": scene_penalty,
            "duration_excess_penalty": duration_excess_penalty,
            "long_candidate_low_density_penalty": long_density_penalty,
            "interval_ad_score": interval_score,
            "interval_score_density": score_density,
            "video_relative_rank_score": rank_score,
            "hard_evidence_density_per_60s": hard_density,
            "ultra_high_confidence": ultra,
            "long_low_density": long_low_density,
        }
    )
    return candidate


def new_candidate(row: dict[str, Any], start_reason: str, start_conf: str, start_pending_support: bool = False) -> dict[str, Any]:
    t = fnum(row.get("candidate_time_sec"))
    return {
        "video_id": int(float(row.get("video_id") or 0)),
        "video_duration_sec": fnum(row.get("video_duration_sec")),
        "start_sec": t,
        "end_sec": t,
        "start_anchor_id": row.get("final_scene_anchor_id"),
        "end_anchor_id": row.get("final_scene_anchor_id"),
        "start_reason": start_reason,
        "end_reason": "",
        "start_confidence_level": start_conf,
        "end_confidence_level": "",
        "hard_evidence_count": int(truth(row.get("hard_evidence_flag_v2_1a"))),
        "start_support_count": int(truth(row.get("start_support_flag")) or start_pending_support),
        "continuity_support_count": int(truth(row.get("continuity_support_flag"))),
        "end_support_count": int(truth(row.get("end_support_flag"))),
        "support_evidence_count": int(truth(row.get("start_support_flag"))) + int(truth(row.get("continuity_support_flag"))) + int(truth(row.get("end_support_flag"))),
        "weak_context_count": int(truth(row.get("weak_context_flag_v2_1a"))),
        "anchor_count": 1,
        "ocr_hard_count": int(fnum(row.get("ocr_hard_disclosure_subtype_count_v2_1a")) > 0),
        "ocr_timeline_recent_hard_count": int(truth(row.get("ocr_timeline_recent_hard_evidence_flag"))),
        "product_cta_timeline_types": int(fnum(row.get("ocr_product_cta_timeline_type_count_v2_1a"))),
        "audio_relative_support_count": int(truth(row.get("relative_audio_active_flag_v2_1a"))),
        "max_weak_span_sec": 0.0,
        "weak_span_start_sec": t if truth(row.get("weak_context_flag_v2_1a")) else None,
        "opening_disclosure_only": truth(row.get("opening_disclosure_only_flag_v2_1a")),
        "fuzzy_only": truth(row.get("fuzzy_only_disclosure_flag_v2_1a")),
        "audio_only": truth(row.get("audio_only_support_flag_v2_1a")),
        "scene_only": truth(row.get("scene_only_support_flag_v2_1a")),
        "start_confirmed_by_hard": truth(row.get("hard_evidence_flag_v2_1a")),
        "forbidden_decision_columns_found": has_forbidden(DECISION_FEATURE_COLUMNS),
    }


def absorb_row(candidate: dict[str, Any], row: dict[str, Any]) -> None:
    t = fnum(row.get("candidate_time_sec"))
    candidate["end_sec"] = t
    candidate["end_anchor_id"] = row.get("final_scene_anchor_id")
    candidate["hard_evidence_count"] += int(truth(row.get("hard_evidence_flag_v2_1a")))
    candidate["start_support_count"] += int(truth(row.get("start_support_flag")))
    candidate["continuity_support_count"] += int(truth(row.get("continuity_support_flag")))
    candidate["end_support_count"] += int(truth(row.get("end_support_flag")))
    candidate["support_evidence_count"] += int(truth(row.get("start_support_flag"))) + int(truth(row.get("continuity_support_flag"))) + int(truth(row.get("end_support_flag")))
    candidate["weak_context_count"] += int(truth(row.get("weak_context_flag_v2_1a")))
    candidate["anchor_count"] += 1
    candidate["ocr_hard_count"] += int(fnum(row.get("ocr_hard_disclosure_subtype_count_v2_1a")) > 0)
    candidate["ocr_timeline_recent_hard_count"] += int(truth(row.get("ocr_timeline_recent_hard_evidence_flag")))
    candidate["product_cta_timeline_types"] = max(
        candidate["product_cta_timeline_types"],
        int(fnum(row.get("ocr_product_cta_timeline_type_count_v2_1a"))),
    )
    candidate["audio_relative_support_count"] += int(truth(row.get("relative_audio_active_flag_v2_1a")))
    candidate["opening_disclosure_only"] = candidate["opening_disclosure_only"] or truth(row.get("opening_disclosure_only_flag_v2_1a"))
    candidate["fuzzy_only"] = candidate["fuzzy_only"] or truth(row.get("fuzzy_only_disclosure_flag_v2_1a"))
    candidate["audio_only"] = candidate["audio_only"] and truth(row.get("audio_only_support_flag_v2_1a"))
    candidate["scene_only"] = candidate["scene_only"] and truth(row.get("scene_only_support_flag_v2_1a"))
    if truth(row.get("weak_context_flag_v2_1a")):
        if candidate.get("weak_span_start_sec") is None:
            candidate["weak_span_start_sec"] = t
        candidate["max_weak_span_sec"] = max(candidate["max_weak_span_sec"], t - candidate["weak_span_start_sec"])
    else:
        candidate["weak_span_start_sec"] = None


def run_detector_for_config(enhanced_rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in enhanced_rows:
        vid = int(float(row.get("video_id") or 0))
        if vid in DEV_IDS:
            by_video[vid].append(row)
    for rows in by_video.values():
        rows.sort(key=lambda item: fnum(item.get("candidate_time_sec")))

    raw_candidates: list[dict[str, Any]] = []
    review_only: list[dict[str, Any]] = []
    open_candidates: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    max_since_hard = config["freshness_support_cap"]["max_sec_since_last_hard_evidence"]
    support_max_duration = config["freshness_support_cap"]["support_only_max_duration_sec"]
    support_max_anchors = config["freshness_support_cap"]["support_only_max_anchor_count"]

    for vid in sorted(by_video):
        state = "non_ad"
        current: dict[str, Any] | None = None
        pending: dict[str, Any] | None = None
        last_hard_time: float | None = None
        support_only_start_time: float | None = None
        support_only_anchor_count = 0
        rows = by_video[vid]
        for anchor_index, row in enumerate(rows):
            t = fnum(row.get("candidate_time_sec"))
            before = state
            hard = truth(row.get("hard_evidence_flag_v2_1a"))
            start_support = truth(row.get("start_support_flag"))
            continuity_support = truth(row.get("continuity_support_flag"))
            end_support = truth(row.get("end_support_flag"))
            weak = truth(row.get("weak_context_flag_v2_1a"))
            opening_only = truth(row.get("opening_disclosure_only_flag_v2_1a"))
            audio_only = truth(row.get("audio_only_support_flag_v2_1a"))
            scene_only = truth(row.get("scene_only_support_flag_v2_1a"))
            decision = "hold"

            if hard:
                last_hard_time = t
                support_only_start_time = None
                support_only_anchor_count = 0
            elif start_support or continuity_support:
                if support_only_start_time is None:
                    support_only_start_time = t
                support_only_anchor_count += 1

            if state == "non_ad":
                if opening_only:
                    review_only.append(
                        {
                            "video_id": vid,
                            "start_sec": t,
                            "end_sec": t,
                            "video_duration_sec": fnum(row.get("video_duration_sec")),
                            "reason": "opening_disclosure_notice_review_only",
                            "review_note": "opening disclosure alone is not a hard start",
                        }
                    )
                    decision = "opening_disclosure_review_only"
                elif hard:
                    current = new_candidate(row, "hard_evidence_start", "high")
                    state = "in_ad"
                    decision = "start_in_ad_by_hard_evidence"
                elif start_support and not audio_only and not scene_only:
                    pending = {
                        "row": row,
                        "time": t,
                        "anchor_index": anchor_index,
                        "support_count": 1,
                    }
                    state = "start_pending"
                    decision = "enter_start_pending_by_start_support"
                elif end_support:
                    decision = "end_support_ignored_for_start"
                elif audio_only:
                    decision = "audio_only_ignored_for_start"
                elif scene_only:
                    decision = "scene_only_ignored_for_start"

            elif state == "start_pending":
                if pending is None:
                    state = "non_ad"
                else:
                    elapsed = t - pending["time"]
                    anchor_elapsed = anchor_index - pending["anchor_index"] + 1
                    if hard:
                        current = new_candidate(pending["row"], "start_support_confirmed_by_hard_evidence", "medium", True)
                        absorb_row(current, row)
                        current["start_confirmed_by_hard"] = True
                        state = "in_ad"
                        pending = None
                        decision = "confirm_start_pending_by_hard_evidence"
                    elif end_support:
                        review_only.append(
                            {
                                "video_id": vid,
                                "start_sec": pending["time"],
                                "end_sec": t,
                                "video_duration_sec": fnum(row.get("video_duration_sec")),
                                "reason": "start_pending_cancelled_by_end_support",
                                "review_note": "end support is not allowed to confirm a start",
                            }
                        )
                        pending = None
                        state = "non_ad"
                        decision = "cancel_start_pending_by_end_support"
                    elif elapsed > support_max_duration or anchor_elapsed > support_max_anchors:
                        review_only.append(
                            {
                                "video_id": vid,
                                "start_sec": pending["time"],
                                "end_sec": t,
                                "video_duration_sec": fnum(row.get("video_duration_sec")),
                                "reason": "start_pending_timeout_review_only",
                                "review_note": "start support was not confirmed by hard evidence within cap",
                            }
                        )
                        pending = None
                        state = "non_ad"
                        decision = "cancel_start_pending_timeout"
                    else:
                        pending["support_count"] += int(start_support)
                        decision = "hold_start_pending"

            elif state == "in_ad":
                if current is None:
                    state = "non_ad"
                else:
                    absorb_row(current, row)
                    no_hard_elapsed = t - (last_hard_time if last_hard_time is not None else current["start_sec"])
                    support_elapsed = t - (support_only_start_time if support_only_start_time is not None else t)
                    if hard or continuity_support:
                        decision = "continue_in_ad_by_hard_or_continuity"
                    if (
                        (end_support and not hard)
                        or no_hard_elapsed > max_since_hard
                        or (support_elapsed > support_max_duration and support_only_anchor_count > support_max_anchors)
                    ):
                        pending = {
                            "time": t,
                            "anchor_id": row.get("final_scene_anchor_id"),
                            "reason": "end_support" if end_support else ("freshness_timeout" if no_hard_elapsed > max_since_hard else "support_cap"),
                        }
                        state = "end_pending"
                        decision = f"enter_end_pending_{pending['reason']}"

            elif state == "end_pending":
                if current is None or pending is None:
                    state = "non_ad"
                elif hard:
                    absorb_row(current, row)
                    state = "in_ad"
                    pending = None
                    decision = "cancel_end_pending_by_hard_evidence"
                elif continuity_support and not end_support:
                    absorb_row(current, row)
                    state = "in_ad"
                    pending = None
                    decision = "cancel_end_pending_by_continuity_support"
                else:
                    elapsed = t - pending["time"]
                    if end_support or elapsed >= min(20.0, max_since_hard):
                        current["end_sec"] = pending["time"]
                        current["end_anchor_id"] = pending["anchor_id"]
                        current["end_reason"] = f"{pending['reason']}_confirmed"
                        current["end_confidence_level"] = "high" if pending["reason"] == "end_support" else "medium"
                        raw_candidates.append(score_candidate(current, config))
                        current = None
                        pending = None
                        state = "non_ad"
                        last_hard_time = None
                        support_only_start_time = None
                        support_only_anchor_count = 0
                        decision = "confirm_end_pending"
                    else:
                        decision = "hold_end_pending"

            if len(trace_rows) < 5000:
                trace_rows.append(
                    {
                        "config_id": config["config_id"],
                        "config_family": config["config_family"],
                        "video_id": vid,
                        "final_scene_anchor_id": row.get("final_scene_anchor_id"),
                        "transition_time_anchor": row.get("transition_time_anchor"),
                        "candidate_time_sec": f"{t:.6f}",
                        "state_before": before,
                        "state_after": state,
                        "decision_type": decision,
                        "hard_evidence_flag_v2_1a": str(hard).lower(),
                        "start_support_flag": str(start_support).lower(),
                        "continuity_support_flag": str(continuity_support).lower(),
                        "end_support_flag": str(end_support).lower(),
                        "weak_context_flag_v2_1a": str(weak).lower(),
                        "opening_disclosure_only_flag_v2_1a": str(opening_only).lower(),
                        "actual_label_used_for_decision": "false",
                        "decision_feature_columns_json": json.dumps(DECISION_FEATURE_COLUMNS, ensure_ascii=False),
                        "forbidden_decision_columns_found": json.dumps(has_forbidden(DECISION_FEATURE_COLUMNS), ensure_ascii=False),
                    }
                )

        if current is not None:
            last_row = rows[-1]
            open_candidates.append(
                {
                    "config_id": config["config_id"],
                    "config_family": config["config_family"],
                    "version": VERSION,
                    "detector_id": DETECTOR_ID,
                    "video_id": vid,
                    "video_duration_sec": f"{fnum(last_row.get('video_duration_sec')):.6f}",
                    "ad_start_sec": f"{current['start_sec']:.6f}",
                    "last_anchor_sec": f"{fnum(last_row.get('candidate_time_sec')):.6f}",
                    "duration_proxy_sec": f"{max(0.0, fnum(last_row.get('candidate_time_sec')) - current['start_sec']):.6f}",
                    "start_anchor_id": current.get("start_anchor_id"),
                    "last_anchor_id": last_row.get("final_scene_anchor_id"),
                    "open_reason": "video_end_open_candidate",
                    "actual_label_used_for_decision": "false",
                }
            )

    for idx, candidate in enumerate(raw_candidates, start=1):
        candidate["candidate_id"] = f"{config['config_id']}_RAW_{idx:05d}"

    raw_rows = [candidate_to_row(config, candidate, "raw_closed_candidate", idx) for idx, candidate in enumerate(raw_candidates, 1)]
    pre_budget: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(raw_candidates, 1):
        thresholds = config["thresholds"]
        long_demote = (
            config["long_candidate"]["force_demote_long_low_density_candidate"]
            and candidate.get("long_low_density")
        )
        reasons = []
        if candidate["interval_ad_score"] < thresholds["min_interval_ad_score"]:
            reasons.append("below_min_interval_ad_score")
        if candidate["interval_score_density"] < thresholds["min_interval_score_density"]:
            reasons.append("below_min_interval_score_density")
        if candidate["hard_evidence_count"] < thresholds["min_hard_evidence_count"]:
            reasons.append("below_min_hard_evidence_count")
        if candidate["hard_evidence_density_per_60s"] < thresholds["min_hard_evidence_density_per_60s"]:
            reasons.append("below_min_hard_evidence_density_per_60s")
        if candidate["max_weak_span_sec"] > thresholds["max_weak_span_sec"]:
            reasons.append("weak_span_exceeds_limit")
        if candidate.get("forbidden_decision_columns_found"):
            reasons.append("forbidden_decision_columns_found")
        if candidate.get("opening_disclosure_only"):
            reasons.append("opening_disclosure_only")
        if candidate.get("fuzzy_only"):
            reasons.append("fuzzy_only")
        if candidate.get("audio_only"):
            reasons.append("audio_only")
        if candidate.get("scene_only"):
            reasons.append("scene_only")
        if long_demote:
            reasons.append("long_candidate_low_density_force_demote")
        if reasons:
            candidate["failure_or_review_reason"] = ";".join(reasons)
            review_rows.append(candidate_to_row(config, candidate, "review_only_candidate", idx))
        else:
            pre_budget.append(candidate)

    for item in review_only:
        review_rows.append(
            {
                "candidate_id": f"{config['config_id']}_REV_{len(review_rows) + 1:05d}",
                "config_id": config["config_id"],
                "config_family": config["config_family"],
                "version": VERSION,
                "detector_id": DETECTOR_ID,
                "original_split_v2_4": "train",
                "split_role_v2_5": "development",
                "evaluation_subset_v2_5": "none",
                "video_id": item.get("video_id"),
                "video_duration_sec": f"{item.get('video_duration_sec', 0.0):.6f}",
                "ad_start_sec": f"{item.get('start_sec', 0.0):.6f}",
                "ad_end_sec": f"{item.get('end_sec', 0.0):.6f}",
                "ad_duration_sec": f"{max(0.0, item.get('end_sec', 0.0) - item.get('start_sec', 0.0)):.6f}",
                "interval_status": "review_only_candidate",
                "failure_or_review_reason": item.get("reason", ""),
                "review_note": item.get("review_note", ""),
                "actual_label_used_for_decision": "false",
                "decision_feature_columns_json": json.dumps(DECISION_FEATURE_COLUMNS, ensure_ascii=False),
                "forbidden_decision_columns_found": json.dumps(has_forbidden(DECISION_FEATURE_COLUMNS), ensure_ascii=False),
            }
        )

    predictions, pruned, budget_events = apply_budget_guard(pre_budget, config)
    prediction_rows = [candidate_to_row(config, candidate, "prediction", idx) for idx, candidate in enumerate(predictions, 1)]
    pruned_rows = [candidate_to_row(config, candidate, "overprediction_pruned_review", idx) for idx, candidate in enumerate(pruned, 1)]
    return {
        "raw_candidates": raw_rows,
        "predictions": prediction_rows,
        "review_only": review_rows,
        "overprediction_pruned_review": pruned_rows,
        "open_candidates": open_candidates,
        "budget_guard_events": budget_events,
        "trace_sample": trace_rows,
    }


def apply_budget_guard(candidates: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_video[int(candidate["video_id"])].append(candidate)
    kept: list[dict[str, Any]] = []
    pruned: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    budget = config["budget_guard"]
    for vid in sorted(by_video):
        rows = by_video[vid]
        video_duration = rows[0].get("video_duration_sec", 0.0) or 0.0
        before_duration = sum(max(0.0, row["end_sec"] - row["start_sec"]) for row in rows)
        before_ratio = safe_ratio(before_duration, video_duration)
        target_duration = budget["target_prediction_ratio_after_pruning"] * video_duration
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                bool(row.get("ultra_high_confidence")),
                row.get("video_relative_rank_score", 0.0),
                row.get("interval_score_density", 0.0),
                row.get("interval_ad_score", 0.0),
            ),
            reverse=True,
        )
        kept_duration = 0.0
        for row in sorted_rows:
            duration = max(0.0, row["end_sec"] - row["start_sec"])
            action = "kept_no_budget_guard_needed"
            should_keep = True
            if before_ratio > budget["soft_overprediction_ratio"]:
                if row.get("ultra_high_confidence"):
                    action = "kept_ultra_high_confidence"
                    should_keep = True
                elif before_ratio > budget["hard_overprediction_ratio"] and kept_duration + duration > target_duration:
                    action = "demoted_hard_budget_guard_to_review"
                    should_keep = False
                elif kept_duration + duration > target_duration:
                    action = "demoted_soft_budget_guard_to_review"
                    should_keep = False
                else:
                    action = "kept_within_budget_target"
            if should_keep:
                kept_duration += duration
                row["budget_guard_action"] = action
                row["video_prediction_ratio_before_budget_guard"] = before_ratio
                row["video_prediction_ratio_after_budget_guard"] = safe_ratio(kept_duration, video_duration)
                kept.append(row)
            else:
                row["budget_guard_action"] = action
                row["video_prediction_ratio_before_budget_guard"] = before_ratio
                row["video_prediction_ratio_after_budget_guard"] = safe_ratio(kept_duration, video_duration)
                row["failure_or_review_reason"] = action
                pruned.append(row)
            events.append(
                {
                    "config_id": config["config_id"],
                    "config_family": config["config_family"],
                    "video_id": vid,
                    "candidate_id": row.get("candidate_id"),
                    "video_duration_sec": f"{video_duration:.6f}",
                    "candidate_duration_sec": f"{duration:.6f}",
                    "prediction_ratio_before_budget_guard": f"{before_ratio:.6f}",
                    "prediction_ratio_after_event": f"{safe_ratio(kept_duration, video_duration):.6f}",
                    "soft_overprediction_ratio": budget["soft_overprediction_ratio"],
                    "hard_overprediction_ratio": budget["hard_overprediction_ratio"],
                    "target_prediction_ratio_after_pruning": budget["target_prediction_ratio_after_pruning"],
                    "budget_guard_action": action,
                    "ultra_high_confidence": str(bool(row.get("ultra_high_confidence"))).lower(),
                    "consistency_failure": str(action == "kept_ultra_high_confidence" and not row.get("ultra_high_confidence")).lower(),
                    "actual_label_used_for_decision": "false",
                }
            )
    return kept, pruned, events


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one v2.1a experimental state machine detector config.")
    parser.add_argument("--fusion-input", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ocr-frame-csv")
    args = parser.parse_args()

    fusion_rows = read_csv(Path(args.fusion_input))
    restoration_map: dict[str, dict[str, float]] = {}
    if args.ocr_frame_csv and Path(args.ocr_frame_csv).exists():
        restoration_map = build_ocr_restoration_map(read_csv(Path(args.ocr_frame_csv)))
    enhanced, subtype_audit, opening_audit, support_audit, status, warnings = enhance_rows_with_v21a_flags(fusion_rows, restoration_map)
    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    outputs = run_detector_for_config(enhanced, config)
    out_dir = Path(args.output_dir)
    for name, rows in outputs.items():
        write_csv(out_dir / f"{name}.csv", rows)
    write_csv(out_dir / "ocr_disclosure_subtype_restoration_audit_v2_1a.csv", subtype_audit)
    write_csv(out_dir / "opening_disclosure_guard_audit_v2_1a.csv", opening_audit)
    write_csv(out_dir / "support_flag_split_audit_v2_1a.csv", support_audit)
    (out_dir / "detector_run_metadata.json").write_text(
        json.dumps({"ocr_subtype_restoration_status": status, "warnings": warnings}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
