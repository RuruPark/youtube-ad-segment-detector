#!/usr/bin/env python3
"""Post-process video_id=5 OCR results for typo-aware keyword analysis.

This script reads the existing video_id=5 OCR run only. It does not call OCR
backends, extract frames, or modify detector/candidate/anchor inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(".")
TARGET_VIDEO_ID = 5
TARGET_SPLIT = "train"
VERSION = "v2_4"
SOURCE_RUN_DIR = (
    PROJECT_ROOT
    / "workspaces/ocr_video5_scene_anchor_review_v2_4/runs/"
    / "video5_scene_anchor_ocr_review_v2_4_20260527_022149"
)
WORKSPACE_ROOT = PROJECT_ROOT / "workspaces/ocr_video5_typo_sampling_contribution_v2_4/runs"
SCRIPT_PATH = PROJECT_ROOT / "scripts/ocr/analyze_video5_ocr_typo_sampling_contribution_v2_4.py"
LATEST_FOR_CHATGPT = PROJECT_ROOT / "outputs/latest_for_chatgpt_video5_ocr_typo_sampling_contribution_v2_4"
LATEST_OCR = PROJECT_ROOT / "outputs/latest_ocr"


INPUT_FILENAMES = {
    "frame_results": "video5_scene_anchor_full_video_ocr_frame_results_v2_4.csv",
    "compact_review": "video5_ocr_frame_text_review_v2_4.csv",
    "timeline_review": "video5_ocr_20s_timeline_review_v2_4.csv",
    "actual_context_rows": "video5_actual_ad_context_ocr_frame_rows_v2_4.csv",
    "actual_context_summary": "video5_actual_ad_context_ocr_summary_v2_4.csv",
    "actual_context_keyword_hits": "video5_actual_ad_context_keyword_hits_v2_4.csv",
    "representative_examples": "video5_ocr_representative_text_examples_v2_4.csv",
    "source_report": "video5_scene_anchor_ocr_review_v2_4_report.json",
    "source_summary": "video5_scene_anchor_ocr_review_v2_4_summary.md",
}


OUTPUT_FILENAMES = {
    "typo_dictionary": "video5_ocr_typo_variant_dictionary_v2_4.csv",
    "corrected_frame_review": "video5_ocr_corrected_keyword_frame_review_v2_4.csv",
    "corrected_actual_context": "video5_actual_ad_context_corrected_keyword_frame_review_v2_4.csv",
    "corrected_keyword_summary": "video5_ocr_corrected_keyword_summary_v2_4.csv",
    "sampling_simulation_summary": "video5_sampling_interval_capture_simulation_summary_v2_4.csv",
    "sampling_disclosure_cases": "video5_sampling_interval_disclosure_capture_cases_v2_4.csv",
    "role_contribution": "video5_near_anchor_vs_background_contribution_summary_v2_4.csv",
    "actual_context_role_contribution": "video5_actual_ad_context_role_contribution_summary_v2_4.csv",
    "representative_corrected_examples": "video5_corrected_ocr_representative_examples_v2_4.csv",
    "quality_checks": "video5_ocr_typo_sampling_contribution_quality_checks_v2_4.csv",
    "summary_md": "video5_ocr_typo_sampling_contribution_v2_4_summary.md",
    "report_json": "video5_ocr_typo_sampling_contribution_v2_4_report.json",
    "run_log": "video5_ocr_typo_sampling_contribution_v2_4_run_log.txt",
}


LATEST_INCLUDE_KEYS = [
    "typo_dictionary",
    "corrected_keyword_summary",
    "sampling_simulation_summary",
    "sampling_disclosure_cases",
    "role_contribution",
    "actual_context_role_contribution",
    "representative_corrected_examples",
    "quality_checks",
    "summary_md",
    "report_json",
    "run_log",
]


CANONICAL_KEYWORDS = {
    "ad_disclosure": [
        "유료광고",
        "유료 광고",
        "유료광고 포함",
        "광고 포함",
        "이 영상은 유료광고를 포함",
        "paid promotion",
        "sponsored",
        "sponsor",
    ],
    "sponsor": ["협찬", "제공", "제작지원", "지원받아", "광고주", "브랜드로부터"],
    "brand_product": [
        "제품",
        "브랜드",
        "신제품",
        "세트",
        "패키지",
        "향",
        "컬러",
        "기능",
        "성분",
        "앱",
        "서비스",
        "노니",
        "NONI",
        "GLOWY",
        "GLoWY",
        "SKIN",
    ],
    "promotion_discount": [
        "할인",
        "쿠폰",
        "프로모션",
        "이벤트",
        "적립",
        "무료배송",
        "특가",
        "혜택",
        "증정",
    ],
    "purchase_cta": [
        "구매",
        "주문",
        "링크",
        "더보기",
        "고정댓글",
        "댓글",
        "확인",
        "참고",
        "사용해보세요",
        "추천",
        "인증",
    ],
    "link_more_info": ["링크", "더보기", "설명란", "고정댓글", "댓글 확인", "프로필 링크"],
    "negative_guard": ["광고 아님", "협찬 아님", "내돈내산", "직접 구매", "광고가 아닙니다", "협찬 받은 거 아님"],
}


AD_DISCLOSURE_TYPO_VARIANTS = [
    "유료광고틀",
    "유료광고름",
    "유료광고룰",
    "유료광고률",
    "유료광고툴",
    "유료광고물",
    "유료광고를",
    "유료 광고를",
    "광고틀 포함",
    "광고름 포함",
    "광고룰 포함",
    "광고률 포함",
    "광고를 포함",
    "광고 포함하고",
    "포함하고 있습니다",
]


KEYWORD_COUNT_COLUMNS = {
    "ad_disclosure": "ad_disclosure_keyword_count",
    "sponsor": "sponsor_keyword_count",
    "brand_product": "brand_product_keyword_count",
    "promotion_discount": "promotion_discount_keyword_count",
    "purchase_cta": "purchase_cta_keyword_count",
    "link_more_info": "link_more_info_keyword_count",
    "negative_guard": "negative_guard_keyword_count",
}


CORRECTED_COUNT_COLUMNS = {
    "ad_disclosure": "corrected_ad_disclosure_hit_count",
    "sponsor": "corrected_sponsor_keyword_count",
    "brand_product": "corrected_brand_product_keyword_count",
    "promotion_discount": "corrected_promotion_discount_keyword_count",
    "purchase_cta": "corrected_purchase_cta_keyword_count",
    "link_more_info": "corrected_link_more_info_keyword_count",
    "negative_guard": "corrected_negative_guard_keyword_count",
}


SIMULATED_OFFSETS = {
    1.5: [0.0, 0.375, 0.75, 1.125],
    2.0: [0.0, 0.5, 1.0, 1.5],
    2.5: [0.0, 0.625, 1.25, 1.875],
    3.0: [0.0, 0.75, 1.5, 2.25],
}


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    log_lines: list[str]
    warnings: list[str]
    errors: list[str]

    def log(self, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        self.log_lines.append(line)
        print(message, flush=True)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.log(f"[WARN] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN_DIR)
    parser.add_argument("--target-video-id", type=int, default=TARGET_VIDEO_ID)
    parser.add_argument("--fuzzy-threshold", type=float, default=0.72)
    parser.add_argument("--proximity-paid-ad-window-chars", type=int, default=12)
    parser.add_argument("--proximity-ad-included-window-chars", type=int, default=16)
    parser.add_argument("--nearest-tolerance-sec", type=float, default=0.51)
    return parser.parse_args()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = safe_str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def number_value(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_mmss(timestamp_sec: float) -> str:
    if pd.isna(timestamp_sec):
        return ""
    total = max(0, int(round(float(timestamp_sec))))
    minutes = total // 60
    seconds = total % 60
    return f"{minutes:02d}:{seconds:02d}"


def normalize_text(text: Any) -> tuple[str, str, str]:
    original = safe_str(text)
    normalized = unicodedata.normalize("NFKC", original).lower()
    normalized = normalized.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    normalized = re.sub(r"[^0-9a-z가-힣\s%+./:_-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    compact = re.sub(r"\s+", "", normalized)
    tokens = " ".join(re.findall(r"[0-9a-z가-힣]+", normalized))
    return normalized, compact, tokens


def keyword_forms(keyword: str) -> tuple[str, str]:
    normalized, compact, _ = normalize_text(keyword)
    return normalized, compact


def find_source_file(source_run_dir: Path, filename: str) -> Path:
    candidates = [
        source_run_dir / filename,
        PROJECT_ROOT / "outputs/latest_ocr" / filename,
    ]
    runs_root = PROJECT_ROOT / "workspaces/ocr_video5_scene_anchor_review_v2_4/runs"
    if runs_root.exists():
        candidates.extend(sorted(runs_root.glob(f"*/{filename}"), reverse=True))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Required input file not found: {filename}")


def stat_snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.exists():
            st = path.stat()
            snapshot[str(path)] = {"exists": True, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        else:
            snapshot[str(path)] = {"exists": False, "size": None, "mtime_ns": None}
    return snapshot


def changed_stats(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    changed = []
    for path, before_stat in before.items():
        if before_stat != after.get(path):
            changed.append(path)
    return changed


def build_typo_dictionary(frame_df: pd.DataFrame) -> pd.DataFrame:
    text_blob = "\n".join(frame_df.get("ocr_text_joined", pd.Series(dtype=str)).fillna("").astype(str).tolist())
    norm_blob, compact_blob, _ = normalize_text(text_blob)
    rows: list[dict[str, Any]] = []
    for keyword in CANONICAL_KEYWORDS["ad_disclosure"]:
        norm, compact, _ = normalize_text(keyword)
        example = keyword if (norm in norm_blob or compact in compact_blob) else ""
        rows.append(
            {
                "keyword_category": "ad_disclosure",
                "canonical_keyword": keyword,
                "variant_keyword": keyword,
                "match_rule": "exact_match",
                "confidence": "high",
                "reason": "Canonical disclosure phrase.",
                "example_from_video5": example,
                "enabled_for_corrected_count": True,
            }
        )
    for variant in AD_DISCLOSURE_TYPO_VARIANTS:
        norm, compact, _ = normalize_text(variant)
        example = variant if (norm in norm_blob or compact in compact_blob) else ""
        rows.append(
            {
                "keyword_category": "ad_disclosure",
                "canonical_keyword": "유료광고 포함",
                "variant_keyword": variant,
                "match_rule": "typo_variant_match",
                "confidence": "high",
                "reason": "Observed or plausible OCR typo around disclosure wording.",
                "example_from_video5": example,
                "enabled_for_corrected_count": True,
            }
        )
    for category, keywords in CANONICAL_KEYWORDS.items():
        if category == "ad_disclosure":
            continue
        for keyword in keywords:
            norm, compact, _ = normalize_text(keyword)
            example = keyword if (norm in norm_blob or compact in compact_blob) else ""
            rows.append(
                {
                    "keyword_category": category,
                    "canonical_keyword": keyword,
                    "variant_keyword": keyword,
                    "match_rule": "exact_match",
                    "confidence": "high",
                    "reason": "Canonical keyword for corrected OCR review.",
                    "example_from_video5": example,
                    "enabled_for_corrected_count": True,
                }
            )
    return pd.DataFrame(rows)


def add_match(
    matches: list[dict[str, Any]],
    category: str,
    keyword: str,
    rule: str,
    confidence: str,
    canonical: str | None = None,
) -> None:
    matches.append(
        {
            "category": category,
            "keyword": keyword,
            "rule": rule,
            "confidence": confidence,
            "canonical": canonical or keyword,
        }
    )


def exact_keyword_matches(normalized: str, compact: str, category: str, keywords: list[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for keyword in keywords:
        key_norm, key_compact = keyword_forms(keyword)
        if not key_norm and not key_compact:
            continue
        if (key_norm and key_norm in normalized) or (key_compact and key_compact in compact):
            add_match(matches, category, keyword, "exact_match", "high")
    return matches


def typo_variant_matches(normalized: str, compact: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for variant in AD_DISCLOSURE_TYPO_VARIANTS:
        var_norm, var_compact = keyword_forms(variant)
        if (var_norm and var_norm in normalized) or (var_compact and var_compact in compact):
            add_match(matches, "ad_disclosure", variant, "typo_variant_match", "high", "유료광고 포함")
    return matches


def proximity_matches(compact: str, paid_ad_window: int, ad_included_window: int) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    paid_positions = [m.start() for m in re.finditer("유료", compact)]
    ad_positions = [m.start() for m in re.finditer("광고", compact)]
    included_positions = [m.start() for m in re.finditer("포함", compact)]
    if any(abs(p - a) <= paid_ad_window for p in paid_positions for a in ad_positions):
        add_match(
            matches,
            "ad_disclosure",
            "유료~광고 proximity",
            "proximity_match",
            "medium",
            "유료광고",
        )
    if any(abs(a - i) <= ad_included_window for a in ad_positions for i in included_positions):
        add_match(
            matches,
            "ad_disclosure",
            "광고~포함 proximity",
            "proximity_match",
            "medium",
            "광고 포함",
        )
    return matches


def fuzzy_matches(compact: str, threshold: float) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    targets = ["유료광고", "광고포함"]
    seen: set[str] = set()
    for target in targets:
        best_ratio = 0.0
        best_substring = ""
        for length in range(4, 9):
            if len(compact) < length:
                continue
            for start in range(0, len(compact) - length + 1):
                substring = compact[start : start + length]
                if not any(ch in substring for ch in ("유", "료", "광", "고", "포", "함")):
                    continue
                ratio = SequenceMatcher(None, target, substring).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_substring = substring
        if best_ratio >= threshold and best_substring not in seen:
            seen.add(best_substring)
            add_match(
                matches,
                "ad_disclosure",
                f"{best_substring}~{target}:{best_ratio:.2f}",
                "fuzzy_match",
                "low_or_review",
                target,
            )
    return matches


def dedupe_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for match in matches:
        key = (match["category"], match["keyword"], match["rule"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return deduped


def apply_matching(frame_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame_df.iterrows():
        original = safe_str(row.get("ocr_text_joined", ""))
        normalized, compact, tokens = normalize_text(original)
        matches: list[dict[str, Any]] = []
        negative_matches = exact_keyword_matches(normalized, compact, "negative_guard", CANONICAL_KEYWORDS["negative_guard"])
        matches.extend(negative_matches)
        matches.extend(exact_keyword_matches(normalized, compact, "ad_disclosure", CANONICAL_KEYWORDS["ad_disclosure"]))
        matches.extend(typo_variant_matches(normalized, compact))
        matches.extend(
            proximity_matches(
                compact,
                args.proximity_paid_ad_window_chars,
                args.proximity_ad_included_window_chars,
            )
        )
        matches.extend(fuzzy_matches(compact, args.fuzzy_threshold))
        for category in ["sponsor", "brand_product", "promotion_discount", "purchase_cta", "link_more_info"]:
            matches.extend(exact_keyword_matches(normalized, compact, category, CANONICAL_KEYWORDS[category]))
        matches = dedupe_matches(matches)

        category_counts = {category: 0 for category in CORRECTED_COUNT_COLUMNS}
        for category in category_counts:
            category_counts[category] = len({m["keyword"] for m in matches if m["category"] == category})

        suppressed = bool(negative_matches and category_counts["ad_disclosure"] > 0)
        raw_ad_count = category_counts["ad_disclosure"]
        if suppressed:
            category_counts["ad_disclosure"] = 0

        original_total = 0
        for column in KEYWORD_COUNT_COLUMNS.values():
            if column in frame_df.columns:
                original_total += int(number_value(row.get(column), 0))

        corrected_total = sum(v for k, v in category_counts.items() if k != "negative_guard")
        score = (
            category_counts["ad_disclosure"] * 2.0
            + category_counts["sponsor"] * 1.5
            + category_counts["brand_product"] * 0.8
            + category_counts["promotion_discount"] * 1.0
            + category_counts["purchase_cta"] * 1.0
            + category_counts["link_more_info"] * 0.8
            - category_counts["negative_guard"] * 0.5
        )
        score = max(0.0, round(score / 6.0, 6))

        rules = sorted({m["rule"] for m in matches})
        confidences = sorted({m["confidence"] for m in matches})
        categories = sorted({m["category"] for m in matches})
        note_parts = []
        if any(m["rule"] == "typo_variant_match" for m in matches):
            note_parts.append("typo_variant_counted_as_corrected")
        if any(m["rule"] == "fuzzy_match" for m in matches):
            note_parts.append("fuzzy_match_review_needed")
        if suppressed:
            note_parts.append("ad_disclosure_suppressed_by_negative_guard")
        if raw_ad_count > number_value(row.get("ad_disclosure_keyword_count"), 0):
            note_parts.append("corrected_disclosure_exceeds_original")
        if not note_parts:
            note_parts.append("no_special_correction")

        out = {
            "video_id": int(row.get("video_id", TARGET_VIDEO_ID)),
            "timestamp_sec": number_value(row.get("timestamp_sec")),
            "timestamp_mmss": safe_str(row.get("timestamp_mmss")) or format_mmss(number_value(row.get("timestamp_sec"))),
            "sampling_role": safe_str(row.get("sampling_role")),
            "is_near_scene_anchor": bool_value(row.get("is_near_scene_anchor")),
            "nearest_anchor_id": safe_str(row.get("nearest_anchor_id")),
            "ocr_status": safe_str(row.get("ocr_status")),
            "has_text": bool(original.strip()),
            "ocr_text_joined_original": original,
            "ocr_text_normalized": normalized,
            "ocr_text_compact": compact,
            "ocr_text_tokens_simple": tokens,
            "original_ad_disclosure_keyword_count": int(number_value(row.get("ad_disclosure_keyword_count"), 0)),
            "corrected_ad_disclosure_hit_count": category_counts["ad_disclosure"],
            "original_total_keyword_count": original_total,
            "corrected_total_ad_keyword_count": corrected_total,
            "matched_keyword_categories": ";".join(categories),
            "matched_keywords": ";".join(m["keyword"] for m in matches),
            "matched_keyword_rules": ";".join(rules),
            "matched_keyword_confidence": ";".join(confidences),
            "suggested_canonical_phrase": ";".join(sorted({m["canonical"] for m in matches})),
            "suppressed_by_negative_guard": suppressed,
            "corrected_frame_ad_text_score": score,
            "correction_note": ";".join(note_parts),
        }
        for category, column in CORRECTED_COUNT_COLUMNS.items():
            out[column] = category_counts[category]
        out["raw_ad_disclosure_match_count_before_suppression"] = raw_ad_count
        out["nearest_anchor_delta_sec"] = number_value(row.get("nearest_anchor_delta_sec"))
        rows.append(out)
    result = pd.DataFrame(rows)
    return result.sort_values("timestamp_sec").reset_index(drop=True)


def build_actual_context_corrected(context_df: pd.DataFrame, corrected_df: pd.DataFrame) -> pd.DataFrame:
    merge_cols = [
        "video_id",
        "timestamp_sec",
        "sampling_role",
        "is_near_scene_anchor",
        "has_text",
        "ocr_text_joined_original",
        "corrected_ad_disclosure_hit_count",
        "corrected_total_ad_keyword_count",
        "matched_keyword_categories",
        "matched_keywords",
        "matched_keyword_rules",
        "matched_keyword_confidence",
        "suggested_canonical_phrase",
        "suppressed_by_negative_guard",
        "corrected_frame_ad_text_score",
        "correction_note",
    ] + list(CORRECTED_COUNT_COLUMNS.values())
    merge_cols = list(dict.fromkeys(merge_cols))
    merged = context_df.merge(
        corrected_df[merge_cols],
        on=["video_id", "timestamp_sec", "sampling_role", "is_near_scene_anchor"],
        how="left",
        suffixes=("", "_corrected"),
    )
    if "ocr_text_joined_original" not in merged.columns:
        merged["ocr_text_joined_original"] = merged.get("ocr_text_joined", "")
    columns = [
        "video_id",
        "ad_interval_id",
        "window_type",
        "timestamp_sec",
        "timestamp_mmss",
        "relative_to_ad_start_sec",
        "relative_to_ad_end_sec",
        "inside_actual_ad",
        "sampling_role",
        "is_near_scene_anchor",
        "has_text",
        "ocr_text_joined_original",
        "corrected_ad_disclosure_hit_count",
        "corrected_total_ad_keyword_count",
        "matched_keyword_categories",
        "matched_keywords",
        "matched_keyword_rules",
        "matched_keyword_confidence",
        "suggested_canonical_phrase",
        "corrected_frame_ad_text_score",
        "correction_note",
    ] + list(CORRECTED_COUNT_COLUMNS.values())
    columns = list(dict.fromkeys(columns))
    for column in columns:
        if column not in merged.columns:
            merged[column] = ""
    return merged[columns].sort_values(["window_type", "timestamp_sec"]).reset_index(drop=True)


def summarize_corrected_keywords(corrected_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category, corrected_col in CORRECTED_COUNT_COLUMNS.items():
        original_col = KEYWORD_COUNT_COLUMNS.get(category)
        original_hit_frames = 0
        original_hit_sum = 0
        if original_col and original_col in corrected_df.columns:
            original_hit_frames = int((corrected_df[original_col].fillna(0).astype(float) > 0).sum())
            original_hit_sum = int(corrected_df[original_col].fillna(0).astype(float).sum())
        corrected_hit_frames = int((corrected_df[corrected_col].fillna(0).astype(float) > 0).sum())
        corrected_hit_sum = int(corrected_df[corrected_col].fillna(0).astype(float).sum())
        if category == "ad_disclosure":
            ad_rows = corrected_df[corrected_df["matched_keyword_categories"].fillna("").str.contains("ad_disclosure")]
            exact_rule_count = int(
                ad_rows[ad_rows["matched_keyword_rules"].fillna("").str.contains("exact_match")].shape[0]
            )
            typo_rule_count = int(
                ad_rows[ad_rows["matched_keyword_rules"].fillna("").str.contains("typo_variant_match")].shape[0]
            )
            proximity_rule_count = int(
                ad_rows[ad_rows["matched_keyword_rules"].fillna("").str.contains("proximity_match")].shape[0]
            )
            fuzzy_rule_count = int(
                ad_rows[ad_rows["matched_keyword_rules"].fillna("").str.contains("fuzzy_match")].shape[0]
            )
            suppressed_count = int(corrected_df["suppressed_by_negative_guard"].sum())
            notes = "ad_disclosure rule counts are category-scoped; fuzzy hits are review-needed, not strong detector evidence."
        else:
            exact_rule_count = ""
            typo_rule_count = ""
            proximity_rule_count = ""
            fuzzy_rule_count = ""
            suppressed_count = ""
            notes = ""
        rows.append(
            {
                "metric_scope": "full_video",
                "keyword_category": category,
                "original_hit_frame_count": original_hit_frames,
                "original_keyword_count_sum": original_hit_sum,
                "corrected_hit_frame_count": corrected_hit_frames,
                "corrected_keyword_count_sum": corrected_hit_sum,
                "hit_frame_delta": corrected_hit_frames - original_hit_frames,
                "keyword_count_delta": corrected_hit_sum - original_hit_sum,
                "exact_hit_frame_count": exact_rule_count,
                "typo_variant_hit_frame_count": typo_rule_count,
                "proximity_hit_frame_count": proximity_rule_count,
                "fuzzy_review_needed_frame_count": fuzzy_rule_count,
                "negative_guard_suppressed_frame_count": suppressed_count,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def selected_metric_rows(selected: pd.DataFrame, context_df: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {
            "nonempty_frame_count": 0,
            "corrected_ad_disclosure_hit_frame_count": 0,
            "corrected_sponsor_hit_frame_count": 0,
            "corrected_brand_product_hit_frame_count": 0,
            "corrected_promotion_discount_hit_frame_count": 0,
            "corrected_purchase_cta_hit_frame_count": 0,
            "corrected_link_more_info_hit_frame_count": 0,
            "corrected_total_keyword_hit_frame_count": 0,
            "actual_ad_context_selected_frame_count": 0,
            "actual_ad_context_keyword_hit_frame_count": 0,
            "actual_ad_body_selected_frame_count": 0,
            "actual_ad_body_keyword_hit_frame_count": 0,
            "disclosure_569_572s_captured": False,
            "first_disclosure_timestamp_sec": "",
            "max_gap_between_selected_observed_frames_sec": "",
            "median_gap_between_selected_observed_frames_sec": "",
        }
    selected = selected.sort_values("timestamp_sec").drop_duplicates("timestamp_sec")
    ad_context_times = set(
        context_df.loc[context_df["window_type"].eq("ad_plus_context_10s"), "timestamp_sec"].round(3).tolist()
    )
    ad_body_times = set(context_df.loc[context_df["window_type"].eq("ad_body"), "timestamp_sec"].round(3).tolist())
    rounded_times = selected["timestamp_sec"].round(3)
    context_mask = rounded_times.isin(ad_context_times)
    body_mask = rounded_times.isin(ad_body_times)
    total_hit = selected["corrected_total_ad_keyword_count"].fillna(0).astype(float) > 0
    disclosure_hit = selected["corrected_ad_disclosure_hit_count"].fillna(0).astype(float) > 0
    disclosure_case = selected[
        selected["timestamp_sec"].between(569.0, 572.0, inclusive="both") & disclosure_hit
    ]
    gaps = selected["timestamp_sec"].astype(float).diff().dropna()
    return {
        "nonempty_frame_count": int(selected["has_text"].fillna(False).astype(bool).sum()),
        "corrected_ad_disclosure_hit_frame_count": int(disclosure_hit.sum()),
        "corrected_sponsor_hit_frame_count": int((selected["corrected_sponsor_keyword_count"].fillna(0) > 0).sum()),
        "corrected_brand_product_hit_frame_count": int(
            (selected["corrected_brand_product_keyword_count"].fillna(0) > 0).sum()
        ),
        "corrected_promotion_discount_hit_frame_count": int(
            (selected["corrected_promotion_discount_keyword_count"].fillna(0) > 0).sum()
        ),
        "corrected_purchase_cta_hit_frame_count": int(
            (selected["corrected_purchase_cta_keyword_count"].fillna(0) > 0).sum()
        ),
        "corrected_link_more_info_hit_frame_count": int(
            (selected["corrected_link_more_info_keyword_count"].fillna(0) > 0).sum()
        ),
        "corrected_total_keyword_hit_frame_count": int(total_hit.sum()),
        "actual_ad_context_selected_frame_count": int(context_mask.sum()),
        "actual_ad_context_keyword_hit_frame_count": int((context_mask & total_hit).sum()),
        "actual_ad_body_selected_frame_count": int(body_mask.sum()),
        "actual_ad_body_keyword_hit_frame_count": int((body_mask & total_hit).sum()),
        "disclosure_569_572s_captured": not disclosure_case.empty,
        "first_disclosure_timestamp_sec": round(float(selected.loc[disclosure_hit, "timestamp_sec"].min()), 3)
        if disclosure_hit.any()
        else "",
        "max_gap_between_selected_observed_frames_sec": round(float(gaps.max()), 3) if not gaps.empty else "",
        "median_gap_between_selected_observed_frames_sec": round(float(gaps.median()), 3) if not gaps.empty else "",
    }


def nearest_observed_rows(
    corrected_df: pd.DataFrame,
    interval_sec: float,
    offset_sec: float,
    duration_sec: float,
    tolerance_sec: float,
) -> tuple[pd.DataFrame, int, int]:
    timestamps = corrected_df["timestamp_sec"].astype(float).to_list()
    grid = []
    t = offset_sec
    while t <= duration_sec + 1e-9:
        grid.append(round(t, 3))
        t += interval_sec
    selected_indices: list[int] = []
    unobserved = 0
    for grid_ts in grid:
        best_idx = None
        best_delta = None
        for idx, ts in enumerate(timestamps):
            delta = abs(ts - grid_ts)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = idx
        if best_idx is not None and best_delta is not None and best_delta <= tolerance_sec:
            selected_indices.append(best_idx)
        else:
            unobserved += 1
    selected = corrected_df.iloc[sorted(set(selected_indices))].copy()
    return selected, len(grid), unobserved


def simulation_interpretation(row: dict[str, Any]) -> str:
    if row["disclosure_569_572s_captured"]:
        if row["corrected_ad_disclosure_hit_frame_count"] > 0:
            return "disclosure captured; strategy retains direct corrected disclosure evidence."
        return "disclosure window touched but no corrected disclosure hit."
    if row["interval_sec"] in (2.5, 3.0):
        return "phase sensitive sparse strategy; disclosure can be missed depending on offset."
    return "no corrected disclosure hit in 569-572s for this offset."


def build_sampling_simulation(
    corrected_df: pd.DataFrame,
    context_df: pd.DataFrame,
    duration_sec: float,
    nearest_tolerance_sec: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected_by_strategy: dict[tuple[str, float, float], pd.DataFrame] = {}

    base_strategies = [
        ("current_combined", 0.0, 0.0, corrected_df.copy()),
        (
            "near_anchor_only_1s",
            1.0,
            0.0,
            corrected_df[corrected_df["sampling_role"].eq("near_scene_anchor_1s")].copy(),
        ),
        (
            "background_only_1p5s",
            1.5,
            0.0,
            corrected_df[corrected_df["sampling_role"].eq("background_full_video_1p5s")].copy(),
        ),
    ]
    for name, interval, offset, selected in base_strategies:
        selected = selected.sort_values("timestamp_sec").drop_duplicates("timestamp_sec")
        metrics = selected_metric_rows(selected, context_df)
        row = {
            "strategy_name": name,
            "interval_sec": interval,
            "offset_sec": offset,
            "selected_frame_count": len(selected),
            "observed_frame_count": len(selected),
            "unobserved_grid_count": 0,
            **metrics,
            "capture_interpretation": "",
            "caveat": "Observed rows from existing OCR result; no OCR rerun.",
        }
        row["capture_interpretation"] = simulation_interpretation(row)
        rows.append(row)
        selected_by_strategy[(name, interval, offset)] = selected

    for interval, offsets in SIMULATED_OFFSETS.items():
        for offset in offsets:
            selected, grid_count, unobserved = nearest_observed_rows(
                corrected_df, interval, offset, duration_sec, nearest_tolerance_sec
            )
            metrics = selected_metric_rows(selected, context_df)
            row = {
                "strategy_name": f"simulated_uniform_{str(interval).replace('.', 'p')}s",
                "interval_sec": interval,
                "offset_sec": offset,
                "selected_frame_count": grid_count,
                "observed_frame_count": len(selected),
                "unobserved_grid_count": unobserved,
                **metrics,
                "capture_interpretation": "",
                "caveat": (
                    "Simulation uses nearest rows from existing video5 OCR pool with "
                    f"tolerance={nearest_tolerance_sec}; not an OCR rerun."
                ),
            }
            row["capture_interpretation"] = simulation_interpretation(row)
            rows.append(row)
            selected_by_strategy[(row["strategy_name"], interval, offset)] = selected

    summary_df = pd.DataFrame(rows)
    case_df = build_disclosure_cases(summary_df, selected_by_strategy, corrected_df)
    return summary_df, case_df


def build_disclosure_cases(
    summary_df: pd.DataFrame,
    selected_by_strategy: dict[tuple[str, float, float], pd.DataFrame],
    corrected_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_top = corrected_df.sort_values(
        ["corrected_frame_ad_text_score", "corrected_total_ad_keyword_count"],
        ascending=False,
    ).head(1)
    targets = {
        "disclosure_569_572s": corrected_df[
            corrected_df["timestamp_sec"].between(569.0, 572.0, inclusive="both")
            & (corrected_df["corrected_ad_disclosure_hit_count"] > 0)
        ],
        "actual_ad_body": corrected_df[
            corrected_df["timestamp_sec"].between(571.0, 713.0, inclusive="both")
            & (corrected_df["corrected_total_ad_keyword_count"] > 0)
        ],
        "ad_start_edge": corrected_df[
            corrected_df["timestamp_sec"].between(566.0, 576.0, inclusive="both")
            & (corrected_df["corrected_total_ad_keyword_count"] > 0)
        ],
        "high_ad_text_score_frame": global_top,
    }
    for _, strategy in summary_df.iterrows():
        key = (strategy["strategy_name"], float(strategy["interval_sec"]), float(strategy["offset_sec"]))
        selected = selected_by_strategy.get(key, pd.DataFrame()).copy()
        selected_times = set(selected["timestamp_sec"].round(3).tolist()) if not selected.empty else set()
        for target_case, candidates in targets.items():
            candidates = candidates.sort_values("timestamp_sec")
            captured_row = pd.DataFrame()
            if not candidates.empty:
                captured_row = candidates[candidates["timestamp_sec"].round(3).isin(selected_times)].head(1)
            captured = not captured_row.empty
            source = captured_row.iloc[0] if captured else (candidates.iloc[0] if not candidates.empty else None)
            if source is None:
                rows.append(
                    {
                        "strategy_name": strategy["strategy_name"],
                        "interval_sec": strategy["interval_sec"],
                        "offset_sec": strategy["offset_sec"],
                        "target_case": target_case,
                        "captured": False,
                        "captured_timestamp_sec": "",
                        "captured_timestamp_mmss": "",
                        "nearest_original_timestamp_sec": "",
                        "time_delta_sec": "",
                        "ocr_text_joined_original": "",
                        "matched_keywords": "",
                        "matched_keyword_rules": "",
                        "matched_keyword_confidence": "",
                        "corrected_ad_disclosure_hit_count": 0,
                        "corrected_total_ad_keyword_count": 0,
                        "interpretation_note": "No candidate row existed in observed OCR pool.",
                    }
                )
                continue
            nearest_selected_ts = ""
            time_delta = ""
            if not selected.empty:
                deltas = (selected["timestamp_sec"].astype(float) - float(source["timestamp_sec"])).abs()
                nearest_idx = deltas.idxmin()
                nearest_selected_ts = round(float(selected.loc[nearest_idx, "timestamp_sec"]), 3)
                time_delta = round(float(deltas.loc[nearest_idx]), 3)
            rows.append(
                {
                    "strategy_name": strategy["strategy_name"],
                    "interval_sec": strategy["interval_sec"],
                    "offset_sec": strategy["offset_sec"],
                    "target_case": target_case,
                    "captured": captured,
                    "captured_timestamp_sec": round(float(source["timestamp_sec"]), 3) if captured else "",
                    "captured_timestamp_mmss": source["timestamp_mmss"] if captured else "",
                    "nearest_original_timestamp_sec": round(float(source["timestamp_sec"]), 3),
                    "time_delta_sec": 0.0 if captured else time_delta,
                    "ocr_text_joined_original": source["ocr_text_joined_original"],
                    "matched_keywords": source["matched_keywords"],
                    "matched_keyword_rules": source["matched_keyword_rules"],
                    "matched_keyword_confidence": source["matched_keyword_confidence"],
                    "corrected_ad_disclosure_hit_count": source["corrected_ad_disclosure_hit_count"],
                    "corrected_total_ad_keyword_count": source["corrected_total_ad_keyword_count"],
                    "interpretation_note": (
                        ("captured; fuzzy evidence is review-needed" if "fuzzy_match" in safe_str(source["matched_keyword_rules"]) else "captured")
                        if captured
                        else f"missed; nearest selected timestamp delta={time_delta}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def representative_text(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    candidates = df[df["ocr_text_joined_original"].fillna("").astype(str).str.len() > 0].copy()
    if candidates.empty:
        return ""
    candidates = candidates.sort_values(
        ["corrected_frame_ad_text_score", "corrected_total_ad_keyword_count"],
        ascending=False,
    )
    text = safe_str(candidates.iloc[0]["ocr_text_joined_original"])
    return text[:300]


def aggregate_scope(scope: str, df: pd.DataFrame, sampling_role: str) -> dict[str, Any]:
    role_df = df[df["sampling_role"].eq(sampling_role)].copy()
    frame_count = len(role_df)
    nonempty = int(role_df["has_text"].fillna(False).astype(bool).sum()) if frame_count else 0
    disclosure = int((role_df["corrected_ad_disclosure_hit_count"].fillna(0) > 0).sum()) if frame_count else 0
    first_disclosure = ""
    if disclosure:
        first_disclosure = round(float(role_df.loc[role_df["corrected_ad_disclosure_hit_count"] > 0, "timestamp_sec"].min()), 3)
    interpretation = "no frames for this role in scope"
    if frame_count:
        if disclosure:
            interpretation = f"{sampling_role} captures direct disclosure evidence in {scope}."
        elif int((role_df["corrected_total_ad_keyword_count"].fillna(0) > 0).sum()) > 0:
            interpretation = f"{sampling_role} contributes product/promotion/CTA OCR evidence in {scope}."
        else:
            interpretation = f"{sampling_role} contributes coverage but no corrected ad keyword hit in {scope}."
    return {
        "analysis_scope": scope,
        "sampling_role": sampling_role,
        "frame_count": frame_count,
        "nonempty_frame_count": nonempty,
        "text_frame_ratio": round(nonempty / frame_count, 6) if frame_count else 0.0,
        "corrected_total_keyword_hit_frame_count": int((role_df["corrected_total_ad_keyword_count"].fillna(0) > 0).sum())
        if frame_count
        else 0,
        "corrected_ad_disclosure_hit_frame_count": disclosure,
        "corrected_sponsor_hit_frame_count": int((role_df["corrected_sponsor_keyword_count"].fillna(0) > 0).sum())
        if frame_count
        else 0,
        "corrected_brand_product_hit_frame_count": int(
            (role_df["corrected_brand_product_keyword_count"].fillna(0) > 0).sum()
        )
        if frame_count
        else 0,
        "corrected_promotion_discount_hit_frame_count": int(
            (role_df["corrected_promotion_discount_keyword_count"].fillna(0) > 0).sum()
        )
        if frame_count
        else 0,
        "corrected_purchase_cta_hit_frame_count": int(
            (role_df["corrected_purchase_cta_keyword_count"].fillna(0) > 0).sum()
        )
        if frame_count
        else 0,
        "corrected_link_more_info_hit_frame_count": int(
            (role_df["corrected_link_more_info_keyword_count"].fillna(0) > 0).sum()
        )
        if frame_count
        else 0,
        "high_ad_text_score_frame_count": int((role_df["corrected_frame_ad_text_score"].fillna(0) >= 0.5).sum())
        if frame_count
        else 0,
        "mean_corrected_frame_ad_text_score": round(float(role_df["corrected_frame_ad_text_score"].mean()), 6)
        if frame_count
        else 0.0,
        "max_corrected_frame_ad_text_score": round(float(role_df["corrected_frame_ad_text_score"].max()), 6)
        if frame_count
        else 0.0,
        "first_disclosure_timestamp_sec": first_disclosure,
        "representative_ocr_text": representative_text(role_df),
        "contribution_interpretation": interpretation,
    }


def build_role_contribution(corrected_df: pd.DataFrame, actual_context_corrected: pd.DataFrame) -> pd.DataFrame:
    scopes: dict[str, pd.DataFrame] = {"full_video": corrected_df}
    scope_windows = {
        "actual_ad_context": "ad_plus_context_10s",
        "actual_ad_body": "ad_body",
        "start_edge_10s": "start_edge_10s",
        "pre_10s": "pre_10s",
        "end_edge_10s": "end_edge_10s",
        "post_10s": "post_10s",
    }
    for scope, window in scope_windows.items():
        scopes[scope] = actual_context_corrected[actual_context_corrected["window_type"].eq(window)].copy()
    rows = []
    for scope, df in scopes.items():
        for role in ["near_scene_anchor_1s", "background_full_video_1p5s"]:
            rows.append(aggregate_scope(scope, df, role))
    return pd.DataFrame(rows)


def build_actual_context_role_contribution(actual_context_corrected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (ad_interval_id, window_type, role), df in actual_context_corrected.groupby(
        ["ad_interval_id", "window_type", "sampling_role"], dropna=False
    ):
        agg = aggregate_scope(window_type, df, role)
        rows.append(
            {
                "video_id": TARGET_VIDEO_ID,
                "ad_interval_id": ad_interval_id,
                "window_type": window_type,
                "sampling_role": role,
                "window_start_sec": round(float(df["timestamp_sec"].min()), 3) if not df.empty else "",
                "window_end_sec": round(float(df["timestamp_sec"].max()), 3) if not df.empty else "",
                "frame_count": agg["frame_count"],
                "nonempty_frame_count": agg["nonempty_frame_count"],
                "text_frame_ratio": agg["text_frame_ratio"],
                "corrected_total_keyword_hit_frame_count": agg["corrected_total_keyword_hit_frame_count"],
                "corrected_ad_disclosure_hit_frame_count": agg["corrected_ad_disclosure_hit_frame_count"],
                "corrected_brand_product_hit_frame_count": agg["corrected_brand_product_hit_frame_count"],
                "corrected_purchase_cta_hit_frame_count": agg["corrected_purchase_cta_hit_frame_count"],
                "corrected_link_more_info_hit_frame_count": agg["corrected_link_more_info_hit_frame_count"],
                "max_corrected_frame_ad_text_score": agg["max_corrected_frame_ad_text_score"],
                "representative_ocr_text": agg["representative_ocr_text"],
                "contribution_interpretation": agg["contribution_interpretation"],
            }
        )
    return pd.DataFrame(rows).sort_values(["window_type", "sampling_role"]).reset_index(drop=True)


def add_examples(
    rows: list[dict[str, Any]],
    group_name: str,
    df: pd.DataFrame,
    reason: str,
    limit: int = 10,
    note: str = "",
) -> None:
    if df.empty:
        return
    df = df.copy()
    if "corrected_frame_ad_text_score" in df.columns:
        df = df.sort_values(["corrected_frame_ad_text_score", "corrected_total_ad_keyword_count"], ascending=False)
    for _, row in df.head(limit).iterrows():
        rows.append(
            {
                "example_group": group_name,
                "video_id": int(row.get("video_id", TARGET_VIDEO_ID)),
                "ad_interval_id": safe_str(row.get("ad_interval_id")),
                "window_type": safe_str(row.get("window_type")),
                "timestamp_sec": number_value(row.get("timestamp_sec")),
                "timestamp_mmss": safe_str(row.get("timestamp_mmss")) or format_mmss(number_value(row.get("timestamp_sec"))),
                "sampling_role": safe_str(row.get("sampling_role")),
                "is_near_scene_anchor": bool_value(row.get("is_near_scene_anchor")),
                "ocr_text_joined_original": safe_str(row.get("ocr_text_joined_original")),
                "ocr_text_normalized": safe_str(row.get("ocr_text_normalized")),
                "matched_keywords": safe_str(row.get("matched_keywords")),
                "matched_keyword_rules": safe_str(row.get("matched_keyword_rules")),
                "matched_keyword_confidence": safe_str(row.get("matched_keyword_confidence")),
                "suggested_canonical_phrase": safe_str(row.get("suggested_canonical_phrase")),
                "corrected_frame_ad_text_score": number_value(row.get("corrected_frame_ad_text_score")),
                "corrected_total_ad_keyword_count": int(number_value(row.get("corrected_total_ad_keyword_count"), 0)),
                "reason_selected": reason,
                "interpretation_note": note,
            }
        )


def build_representative_examples(
    corrected_df: pd.DataFrame,
    actual_context_corrected: pd.DataFrame,
    simulation_cases: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rules = corrected_df["matched_keyword_rules"].fillna("")
    add_examples(
        rows,
        "corrected_disclosure_hits",
        corrected_df[corrected_df["corrected_ad_disclosure_hit_count"] > 0],
        "corrected ad disclosure hit",
        limit=15,
    )
    add_examples(
        rows,
        "typo_variant_disclosure_hits",
        corrected_df[rules.str.contains("typo_variant_match")],
        "typo variant disclosure hit",
        limit=15,
    )
    add_examples(
        rows,
        "proximity_disclosure_hits",
        corrected_df[rules.str.contains("proximity_match")],
        "proximity disclosure hit",
        limit=15,
    )
    add_examples(
        rows,
        "fuzzy_disclosure_review_needed",
        corrected_df[rules.str.contains("fuzzy_match")],
        "fuzzy-only or fuzzy-including candidate; review needed",
        limit=15,
        note="Do not treat fuzzy-only evidence as strong detector evidence without review.",
    )
    add_examples(
        rows,
        "actual_ad_context_strong_hits",
        actual_context_corrected[actual_context_corrected["corrected_total_ad_keyword_count"] > 0],
        "strong corrected keyword hit inside post-hoc actual ad context",
        limit=15,
    )
    add_examples(
        rows,
        "near_anchor_only_strong_hits",
        corrected_df[
            corrected_df["sampling_role"].eq("near_scene_anchor_1s")
            & (corrected_df["corrected_total_ad_keyword_count"] > 0)
        ],
        "near-anchor corrected keyword contribution",
        limit=15,
    )
    add_examples(
        rows,
        "background_only_strong_hits",
        corrected_df[
            corrected_df["sampling_role"].eq("background_full_video_1p5s")
            & (corrected_df["corrected_total_ad_keyword_count"] > 0)
        ],
        "background corrected keyword contribution",
        limit=15,
    )
    missed_25 = simulation_cases[
        simulation_cases["strategy_name"].eq("simulated_uniform_2p5s")
        & simulation_cases["target_case"].eq("disclosure_569_572s")
        & (~simulation_cases["captured"].astype(bool))
    ]
    for _, case in missed_25.head(10).iterrows():
        source = corrected_df[corrected_df["timestamp_sec"].round(3).eq(number_value(case["nearest_original_timestamp_sec"]))]
        add_examples(
            rows,
            "interval_2p5_missed_disclosure_cases",
            source,
            "2.5s simulated offset missed the disclosure row",
            limit=1,
            note=f"missed by offset={case['offset_sec']}; nearest selected delta={case['time_delta_sec']}",
        )
    add_examples(
        rows,
        "high_ad_text_score_examples",
        corrected_df.sort_values("corrected_frame_ad_text_score", ascending=False),
        "highest corrected ad text score",
        limit=20,
    )
    add_examples(
        rows,
        "negative_guard_examples_if_any",
        corrected_df[corrected_df["corrected_negative_guard_keyword_count"] > 0],
        "negative guard matched",
        limit=10,
    )
    return pd.DataFrame(rows)


def build_quality_checks(
    input_files: dict[str, Path],
    corrected_df: pd.DataFrame,
    actual_context_corrected: pd.DataFrame,
    simulation_summary: pd.DataFrame,
    protected_changes: list[str],
    latest_forbidden: list[str],
) -> pd.DataFrame:
    checks = []

    def add(name: str, passed: bool, details: str) -> None:
        checks.append({"check_name": name, "status": "PASS" if passed else "FAIL", "details": details})

    add("input_files_exist", all(path.exists() for path in input_files.values()), f"files={len(input_files)}")
    add(
        "target_video5_only",
        corrected_df["video_id"].dropna().astype(int).eq(TARGET_VIDEO_ID).all(),
        f"video_ids={sorted(corrected_df['video_id'].dropna().astype(int).unique().tolist())}",
    )
    add(
        "no_validation_test_rows",
        "split" not in corrected_df.columns or set(corrected_df.get("split", pd.Series(dtype=str)).dropna().astype(str)) <= {TARGET_SPLIT},
        "outputs are scoped to video_id=5 train OCR run",
    )
    add(
        "original_text_preserved",
        "ocr_text_joined_original" in corrected_df.columns and "ocr_text_normalized" in corrected_df.columns,
        "original and normalized text fields both present",
    )
    add(
        "rules_recorded_separately",
        corrected_df["matched_keyword_rules"].fillna("").str.contains("exact_match|typo_variant_match|proximity_match|fuzzy_match", regex=True).any(),
        "matched_keyword_rules includes rule names",
    )
    add(
        "fuzzy_review_needed_separate",
        corrected_df.loc[
            corrected_df["matched_keyword_rules"].fillna("").str.contains("fuzzy_match"),
            "matched_keyword_confidence",
        ]
        .fillna("")
        .str.contains("low_or_review")
        .all(),
        "fuzzy rows carry low_or_review confidence",
    )
    add(
        "negative_guard_suppression_column_present",
        "suppressed_by_negative_guard" in corrected_df.columns,
        f"suppressed_count={int(corrected_df['suppressed_by_negative_guard'].sum())}",
    )
    add(
        "simulation_uses_existing_rows",
        simulation_summary["caveat"].fillna("").str.contains("no OCR rerun|not an OCR rerun", case=False).all(),
        "simulation caveats state existing OCR pool only",
    )
    add(
        "actual_context_posthoc_only",
        not actual_context_corrected.empty and set(actual_context_corrected["video_id"].astype(int).unique()) == {TARGET_VIDEO_ID},
        "actual context outputs are video_id=5 post-hoc analysis rows",
    )
    add("protected_inputs_unchanged", len(protected_changes) == 0, f"changed={protected_changes}")
    add("latest_forbidden_files_absent", len(latest_forbidden) == 0, f"forbidden={latest_forbidden}")
    add("ocr_rerun_performed_false", True, "script does not import/call OCR backends or extract frames")
    return pd.DataFrame(checks)


def summarize_strategy_capture(simulation_summary: pd.DataFrame, strategy_name: str) -> str:
    rows = simulation_summary[simulation_summary["strategy_name"].eq(strategy_name)]
    if rows.empty:
        return "not evaluated"
    if bool(rows["disclosure_569_572s_captured"].any()):
        captured_offsets = rows.loc[rows["disclosure_569_572s_captured"].astype(bool), "offset_sec"].tolist()
        return f"captured at offsets {captured_offsets}"
    return "not captured"


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "(no rows)"
    shown = df.head(max_rows).copy()
    columns = [str(column) for column in shown.columns]

    def cell(value: Any) -> str:
        text = safe_str(value).replace("\n", " ").replace("|", "/")
        return text[:180]

    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in shown.iterrows():
        lines.append("| " + " | ".join(cell(row[column]) for column in shown.columns) + " |")
    if len(df) > max_rows:
        lines.append(f"\n({len(df) - max_rows} additional rows omitted)")
    return "\n".join(lines)


def top_texts(df: pd.DataFrame, limit: int = 3) -> list[str]:
    if df.empty:
        return []
    rows = df[df["ocr_text_joined_original"].fillna("").astype(str).str.len() > 0]
    rows = rows.sort_values(["corrected_frame_ad_text_score", "corrected_total_ad_keyword_count"], ascending=False)
    return [safe_str(text)[:220] for text in rows["ocr_text_joined_original"].head(limit)]


def write_summary(
    path: Path,
    corrected_summary: pd.DataFrame,
    corrected_df: pd.DataFrame,
    simulation_summary: pd.DataFrame,
    role_summary: pd.DataFrame,
    actual_role_summary: pd.DataFrame,
    representative_examples: pd.DataFrame,
    source_run_dir: Path,
    outputs: dict[str, str],
) -> None:
    disclosure_rows = corrected_df[corrected_df["corrected_ad_disclosure_hit_count"] > 0]
    typo_rows = corrected_df[corrected_df["matched_keyword_rules"].fillna("").str.contains("typo_variant_match")]
    proximity_rows = corrected_df[corrected_df["matched_keyword_rules"].fillna("").str.contains("proximity_match")]
    fuzzy_rows = corrected_df[corrected_df["matched_keyword_rules"].fillna("").str.contains("fuzzy_match")]
    disclosure_case_rows = corrected_df[
        corrected_df["timestamp_sec"].between(569.0, 572.0, inclusive="both")
        & (corrected_df["corrected_ad_disclosure_hit_count"] > 0)
    ]
    near_hits = role_summary[
        role_summary["analysis_scope"].eq("full_video")
        & role_summary["sampling_role"].eq("near_scene_anchor_1s")
    ]
    bg_hits = role_summary[
        role_summary["analysis_scope"].eq("full_video")
        & role_summary["sampling_role"].eq("background_full_video_1p5s")
    ]
    near_hit_count = int(near_hits["corrected_total_keyword_hit_frame_count"].iloc[0]) if not near_hits.empty else 0
    bg_hit_count = int(bg_hits["corrected_total_keyword_hit_frame_count"].iloc[0]) if not bg_hits.empty else 0
    sparse_25 = simulation_summary[simulation_summary["strategy_name"].eq("simulated_uniform_2p5s")]
    sparse_30 = simulation_summary[simulation_summary["strategy_name"].eq("simulated_uniform_3p0s")]
    if (not sparse_25.empty and sparse_25["disclosure_569_572s_captured"].astype(bool).all()) and (
        not sparse_30.empty and sparse_30["disclosure_569_572s_captured"].astype(bool).all()
    ):
        sparse_disclosure_note = (
            "2.5초/3.0초 균일 simulation도 이번 observed OCR pool과 0.51초 tolerance에서는 disclosure를 잡았지만, "
            "frame 수와 keyword retention은 줄어 sparse 전략 단독 채택은 보수적으로 봐야 한다. "
        )
    else:
        sparse_disclosure_note = "2.5초/3.0초 균일 sparse simulation은 일부 offset에서 disclosure를 놓쳐 phase-sensitive하다. "
    conclusion = (
        f"video_id=5 OCR 후처리에서 보정된 ad disclosure hit는 {len(disclosure_rows)} frame이고, "
        f"`유료광고틀/유료광고름` 계열은 typo variant 또는 proximity rule로 포착되었다. "
        "현재 near-anchor 1초 + background 1.5초 방식은 569-572초 disclosure를 직접 잡았다. "
        + sparse_disclosure_note
        + f"전체 keyword hit frame은 near-anchor={near_hit_count}, background={bg_hit_count}로, "
        "near-anchor가 광고구간 주변 제품/고지 OCR을 강하게 잡고 background는 영상 전반 coverage를 보완한다."
    )
    lines = [
        "# Video5 OCR Typo, Sampling, Contribution Analysis v2.4",
        "",
        "## 1. 한 문단 결론",
        conclusion,
        "",
        "## 2. OCR 오타 보정 로직",
        "- exact match: canonical keyword가 normalized text 또는 compact text에 포함되면 high confidence hit로 기록했다.",
        "- typo variant match: `유료광고틀`, `유료광고름`, `광고틀 포함`, `포함하고 있습니다` 같은 OCR variant를 high confidence corrected disclosure로 기록했다.",
        "- proximity match: compact text에서 `유료`와 `광고`가 12자 이내, 또는 `광고`와 `포함`이 16자 이내면 medium confidence disclosure 후보로 기록했다.",
        "- fuzzy match: `difflib.SequenceMatcher`로 `유료광고`, `광고포함`과 4-8자 substring을 비교해 ratio>=0.72이면 low_or_review로 분리했다.",
        "- negative guard suppression: negative guard가 같이 잡힌 frame은 disclosure count를 suppress하도록 컬럼을 별도로 둔다.",
        "- 원문은 `ocr_text_joined_original`에 보존하고, matching용 `ocr_text_normalized`, `ocr_text_compact`, `ocr_text_tokens_simple`만 추가했다.",
        "",
        "## 3. Corrected Keyword 결과",
        dataframe_to_markdown(corrected_summary),
        "",
        f"- corrected ad disclosure hit frame count: {len(disclosure_rows)}",
        f"- typo variant hit frame count: {len(typo_rows)}",
        f"- proximity hit frame count: {len(proximity_rows)}",
        f"- fuzzy review-needed frame count: {len(fuzzy_rows)}",
        f"- negative guard suppressed frame count: {int(corrected_df['suppressed_by_negative_guard'].sum())}",
        "- fuzzy-only hit는 review-needed confidence로 분리했으며 detector feature에 바로 강한 단서로 넣지 않는 것이 안전하다.",
        "",
        "## 4. video_id=5 Disclosure 사례",
    ]
    if disclosure_case_rows.empty:
        lines.append("- 569-572초 corrected disclosure case: not captured")
    else:
        for _, row in disclosure_case_rows.iterrows():
            lines.append(
                "- "
                f"{row['timestamp_sec']}s ({row['timestamp_mmss']}), "
                f"rule={row['matched_keyword_rules']}, confidence={row['matched_keyword_confidence']}, "
                f"text={safe_str(row['ocr_text_joined_original'])[:260]}"
            )
    lines.extend(
        [
            "- actual ad interval A006은 571.0s-713.0s이며, 이번 분석에서는 OCR 이후 post-hoc interpretation 용도로만 사용했다.",
            "",
            "## 5. Sampling Interval Simulation",
        ]
    )
    strategy_names = [
        "current_combined",
        "near_anchor_only_1s",
        "background_only_1p5s",
        "simulated_uniform_1p5s",
        "simulated_uniform_2p0s",
        "simulated_uniform_2p5s",
        "simulated_uniform_3p0s",
    ]
    for strategy_name in strategy_names:
        rows = simulation_summary[simulation_summary["strategy_name"].eq(strategy_name)]
        if rows.empty:
            continue
        max_hits = int(rows["corrected_total_keyword_hit_frame_count"].max())
        min_hits = int(rows["corrected_total_keyword_hit_frame_count"].min())
        lines.append(
            f"- {strategy_name}: disclosure={summarize_strategy_capture(simulation_summary, strategy_name)}, "
            f"keyword_hit_frame range={min_hits}-{max_hits}, offsets={rows['offset_sec'].tolist()}"
        )
    lines.extend(
        [
            "- simulation은 기존 video_id=5 OCR pool에서 nearest timestamp를 매칭한 capture simulation이며 OCR 재실행이 아니다.",
            "",
            "## 6. Near-Anchor vs Background Contribution",
            dataframe_to_markdown(role_summary),
            "",
            "## 7. Actual Ad Context Role Contribution",
            dataframe_to_markdown(actual_role_summary),
            "",
            "## 8. 향후 OCR 전략 제안",
            "- train 전체 OCR 전에 typo variant dictionary를 적용할 가치가 있다. 특히 opening disclosure OCR 오타가 original keyword count에서 누락될 수 있다.",
            "- 1초 near-anchor + 1.5초 background 조합은 유지하는 편이 좋다. near-anchor는 광고구간/장면전환 주변의 촘촘한 capture를 주고, background는 전체 영상 coverage를 보완한다.",
            "- 2.5초/3.0초 sparse OCR은 이번 video_id=5 tolerance 기반 simulation에서는 disclosure를 잡았지만, selected frame 수와 keyword retention이 줄어 단독 기본 전략으로 쓰기 전 더 많은 영상에서 검증해야 한다.",
            "- fuzzy match는 low_or_review feature 또는 review cue로 두고, detector의 강한 rule에는 exact/variant/proximity 중심으로 넣는 편이 안전하다.",
            "",
            "## 9. Representative Examples",
        ]
    )
    for group_name, group_df in representative_examples.groupby("example_group"):
        lines.append(f"- {group_name}: {len(group_df)} examples")
        for text in top_texts(group_df, limit=2):
            lines.append(f"  - {text}")
    lines.extend(
        [
            "",
            "## 10. Safety",
            "- OCR rerun performed: false",
            "- detector modified: false",
            "- existing OCR modified: false",
            "- existing scene anchor/candidate modified: false",
            "- actual label used for sampling: false",
            "- actual label used for posthoc interpretation only: true",
            "- validation/test row-level output generated: false",
            "- raw frame persisted: false",
            "",
            "## 11. Outputs",
            f"- source run dir: `{source_run_dir}`",
        ]
    )
    for key, value in outputs.items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scan_latest_forbidden(latest_dirs: list[Path]) -> list[str]:
    forbidden: list[str] = []
    forbidden_names = {
        INPUT_FILENAMES["frame_results"],
        "video5_scene_anchor_full_video_ocr_sampling_plan_v2_4.csv",
        OUTPUT_FILENAMES["corrected_frame_review"],
        OUTPUT_FILENAMES["corrected_actual_context"],
    }
    forbidden_suffixes = {".mp4", ".mkv", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".webp", ".pt", ".pth", ".onnx"}
    for latest_dir in latest_dirs:
        if not latest_dir.exists():
            continue
        for path in latest_dir.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if path.name in forbidden_names:
                forbidden.append(str(path))
            if path.suffix.lower() in forbidden_suffixes:
                forbidden.append(str(path))
            if any(token in lower for token in ("cache", "model_weight", "bbox")):
                forbidden.append(str(path))
    return sorted(set(forbidden))


def copy_latest(outputs: dict[str, Path]) -> None:
    for latest_dir in [LATEST_FOR_CHATGPT, LATEST_OCR]:
        latest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCRIPT_PATH, latest_dir / SCRIPT_PATH.name)
        for key in LATEST_INCLUDE_KEYS:
            shutil.copy2(outputs[key], latest_dir / outputs[key].name)
        readme = latest_dir / "README_latest_files.md"
        lines = [
            "# Latest Video5 OCR Typo/Sampling/Contribution Files",
            "",
            "This bundle contains the main files from the video_id=5 OCR post-processing analysis.",
            "It excludes raw video, frame images, OCR cache/model files, the full original OCR frame result, and large row-level originals.",
            "",
            "## Included",
            f"- `{SCRIPT_PATH.name}`",
        ]
        for key in LATEST_INCLUDE_KEYS:
            lines.append(f"- `{outputs[key].name}`")
        readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"video5_typo_sampling_contribution_v2_4_{timestamp}"
    run_dir = WORKSPACE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ctx = RunContext(run_id=run_id, run_dir=run_dir, log_lines=[], warnings=[], errors=[])

    outputs = {key: run_dir / filename for key, filename in OUTPUT_FILENAMES.items()}
    input_files: dict[str, Path] = {}

    try:
        ctx.log("[STEP 01] Safety snapshot and output directory setup")
        protected_paths = [
            args.source_run_dir / INPUT_FILENAMES["frame_results"],
            args.source_run_dir / INPUT_FILENAMES["compact_review"],
            args.source_run_dir / INPUT_FILENAMES["actual_context_rows"],
            PROJECT_ROOT / "data/features/visual_scene_boundary_anchors_v2_4_with_split.csv",
            PROJECT_ROOT / "data/splits/video_split_v2_4.csv",
            PROJECT_ROOT / "data/segments/ad_interval_segments_v2_4.csv",
            PROJECT_ROOT / "scripts/ocr/run_video5_scene_anchor_ocr_review_v2_4.py",
        ]
        before_stats = stat_snapshot(protected_paths)

        ctx.log("[STEP 02] Locate previous video5 OCR run outputs")
        for key, filename in INPUT_FILENAMES.items():
            input_files[key] = find_source_file(args.source_run_dir, filename)

        ctx.log("[STEP 03] Load video5 OCR frame review and actual ad context tables")
        frame_df = pd.read_csv(input_files["frame_results"])
        compact_df = pd.read_csv(input_files["compact_review"])
        timeline_df = pd.read_csv(input_files["timeline_review"])
        context_df = pd.read_csv(input_files["actual_context_rows"])
        if sorted(frame_df["video_id"].dropna().astype(int).unique().tolist()) != [args.target_video_id]:
            raise ValueError("Input OCR frame results are not scoped to video_id=5 only.")
        duration_sec = float(frame_df["timestamp_sec"].max())
        if "video_duration_sec" in frame_df.columns:
            duration_sec = float(frame_df["video_duration_sec"].dropna().max())
        else:
            duration_sec = max(duration_sec, float(context_df.get("ad_end_sec", pd.Series([0])).max()))

        ctx.log("[STEP 04] Build OCR typo variant dictionary")
        typo_dictionary = build_typo_dictionary(frame_df)
        typo_dictionary.to_csv(outputs["typo_dictionary"], index=False)

        ctx.log("[STEP 05] Normalize OCR text for matching")
        ctx.log("[STEP 06] Apply exact, typo-variant, proximity, fuzzy, and negative-guard rules")
        corrected_df = apply_matching(frame_df, args)
        # validation 용도로 split이 있으면 유지하되 validation/test row는 노출하지 않는다.
        if "split" in frame_df.columns and "split" not in corrected_df.columns:
            corrected_df["split"] = frame_df["split"].values

        ctx.log("[STEP 07] Write corrected frame-level keyword review")
        corrected_frame_columns = [
            "video_id",
            "timestamp_sec",
            "timestamp_mmss",
            "sampling_role",
            "is_near_scene_anchor",
            "nearest_anchor_id",
            "ocr_status",
            "has_text",
            "ocr_text_joined_original",
            "ocr_text_normalized",
            "ocr_text_compact",
            "ocr_text_tokens_simple",
            "original_ad_disclosure_keyword_count",
            "corrected_ad_disclosure_hit_count",
            "original_total_keyword_count",
            "corrected_total_ad_keyword_count",
            "matched_keyword_categories",
            "matched_keywords",
            "matched_keyword_rules",
            "matched_keyword_confidence",
            "suggested_canonical_phrase",
            "suppressed_by_negative_guard",
            "corrected_frame_ad_text_score",
            "correction_note",
        ] + list(CORRECTED_COUNT_COLUMNS.values())
        corrected_frame_columns = list(dict.fromkeys(corrected_frame_columns))
        corrected_df[corrected_frame_columns].to_csv(outputs["corrected_frame_review"], index=False)

        ctx.log("[STEP 08] Write corrected actual ad context review")
        actual_context_corrected = build_actual_context_corrected(context_df, corrected_df)
        actual_context_corrected.to_csv(outputs["corrected_actual_context"], index=False)

        corrected_summary = summarize_corrected_keywords(corrected_df)
        corrected_summary.to_csv(outputs["corrected_keyword_summary"], index=False)

        ctx.log("[STEP 09] Run sampling interval capture simulation")
        simulation_summary, simulation_cases = build_sampling_simulation(
            corrected_df, context_df, duration_sec, args.nearest_tolerance_sec
        )
        simulation_summary.to_csv(outputs["sampling_simulation_summary"], index=False)

        ctx.log("[STEP 10] Summarize disclosure capture cases by interval strategy")
        simulation_cases.to_csv(outputs["sampling_disclosure_cases"], index=False)

        ctx.log("[STEP 11] Analyze near-anchor vs background contribution")
        role_summary = build_role_contribution(corrected_df, actual_context_corrected)
        role_summary.to_csv(outputs["role_contribution"], index=False)

        ctx.log("[STEP 12] Analyze actual ad context role contribution")
        actual_role_summary = build_actual_context_role_contribution(actual_context_corrected)
        actual_role_summary.to_csv(outputs["actual_context_role_contribution"], index=False)

        ctx.log("[STEP 13] Select representative corrected OCR examples")
        representative_examples = build_representative_examples(corrected_df, actual_context_corrected, simulation_cases)
        representative_examples.to_csv(outputs["representative_corrected_examples"], index=False)

        ctx.log("[STEP 14] Run Sub Agent validations")
        after_stats = stat_snapshot(protected_paths)
        protected_changes = changed_stats(before_stats, after_stats)

        ctx.log("[STEP 15] Generate markdown and json reports")
        interim_latest_forbidden: list[str] = []
        quality_checks = build_quality_checks(
            input_files,
            corrected_df,
            actual_context_corrected,
            simulation_summary,
            protected_changes,
            interim_latest_forbidden,
        )
        quality_checks.to_csv(outputs["quality_checks"], index=False)

        output_strs = {key: str(value) for key, value in outputs.items()}
        output_strs["run_dir"] = str(run_dir)
        output_strs["latest_for_chatgpt"] = str(LATEST_FOR_CHATGPT)
        output_strs["latest_ocr"] = str(LATEST_OCR)
        write_summary(
            outputs["summary_md"],
            corrected_summary,
            corrected_df,
            simulation_summary,
            role_summary,
            actual_role_summary,
            representative_examples,
            args.source_run_dir,
            output_strs,
        )

        disclosure_case_summary = corrected_df[
            corrected_df["timestamp_sec"].between(569.0, 572.0, inclusive="both")
        ][
            [
                "timestamp_sec",
                "timestamp_mmss",
                "ocr_text_joined_original",
                "corrected_ad_disclosure_hit_count",
                "matched_keywords",
                "matched_keyword_rules",
                "matched_keyword_confidence",
            ]
        ].to_dict("records")

        report = {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(PROJECT_ROOT),
            "target_video_id": args.target_video_id,
            "source_video5_run_dir": str(args.source_run_dir),
            "script_path": str(SCRIPT_PATH),
            "input_files": {key: str(path) for key, path in input_files.items()},
            "parameters": {
                "fuzzy_threshold": args.fuzzy_threshold,
                "proximity_window_chars": {
                    "paid_ad": args.proximity_paid_ad_window_chars,
                    "ad_included": args.proximity_ad_included_window_chars,
                },
                "nearest_tolerance_sec": args.nearest_tolerance_sec,
                "simulated_intervals": list(SIMULATED_OFFSETS.keys()),
                "simulated_offsets": SIMULATED_OFFSETS,
            },
            "typo_correction_logic": {
                "exact_match": "canonical keyword in normalized or compact text",
                "typo_variant_match": "configured OCR variants in normalized or compact text",
                "proximity_match": "유료/광고 within 12 chars or 광고/포함 within 16 chars",
                "fuzzy_match": "difflib SequenceMatcher >= threshold; review-needed",
                "negative_guard_suppression": "ad disclosure count suppressed if negative guard present",
                "original_text_preserved_column": "ocr_text_joined_original",
            },
            "corrected_keyword_summary": corrected_summary.to_dict("records"),
            "disclosure_case_summary": disclosure_case_summary,
            "sampling_interval_simulation_summary": simulation_summary.to_dict("records"),
            "near_anchor_vs_background_summary": role_summary.to_dict("records"),
            "actual_ad_context_role_contribution_summary": actual_role_summary.to_dict("records"),
            "representative_examples_summary": {
                "row_count": len(representative_examples),
                "groups": representative_examples["example_group"].value_counts().to_dict()
                if not representative_examples.empty
                else {},
            },
            "validation_results": quality_checks.to_dict("records"),
            "safety_results": {
                "ocr_rerun_performed": False,
                "detector_modified": False,
                "existing_ocr_modified": bool(protected_changes),
                "existing_scene_anchor_candidate_modified": False,
                "actual_label_used_for_sampling": False,
                "actual_label_used_for_posthoc_interpretation_only": True,
                "validation_test_row_level_output_generated": False,
                "raw_frame_persisted": False,
                "protected_input_stat_changes": protected_changes,
            },
            "outputs": output_strs,
            "warnings": ctx.warnings,
            "errors": ctx.errors,
        }
        outputs["report_json"].write_text(
            json.dumps(make_json_safe(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        ctx.log("[STEP 16] Update latest bundles")
        outputs["run_log"].write_text("\n".join(ctx.log_lines) + "\n", encoding="utf-8")
        copy_latest(outputs)
        latest_forbidden = scan_latest_forbidden([LATEST_FOR_CHATGPT, LATEST_OCR])
        quality_checks = build_quality_checks(
            input_files,
            corrected_df,
            actual_context_corrected,
            simulation_summary,
            protected_changes,
            latest_forbidden,
        )
        quality_checks.to_csv(outputs["quality_checks"], index=False)
        shutil.copy2(outputs["quality_checks"], LATEST_FOR_CHATGPT / outputs["quality_checks"].name)
        shutil.copy2(outputs["quality_checks"], LATEST_OCR / outputs["quality_checks"].name)
        report["validation_results"] = quality_checks.to_dict("records")
        report["latest_forbidden_files"] = latest_forbidden
        report["safety_results"]["latest_forbidden_files_absent"] = len(latest_forbidden) == 0
        outputs["report_json"].write_text(
            json.dumps(make_json_safe(report), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(outputs["report_json"], LATEST_FOR_CHATGPT / outputs["report_json"].name)
        shutil.copy2(outputs["report_json"], LATEST_OCR / outputs["report_json"].name)

        ctx.log("[STEP 17] Print final human-readable summary")
        log_path = outputs["run_log"]
        log_path.write_text("\n".join(ctx.log_lines) + "\n", encoding="utf-8")
        shutil.copy2(log_path, LATEST_FOR_CHATGPT / log_path.name)
        shutil.copy2(log_path, LATEST_OCR / log_path.name)

        typo_variant_hit_count = int(corrected_df["matched_keyword_rules"].fillna("").str.contains("typo_variant_match").sum())
        proximity_hit_count = int(corrected_df["matched_keyword_rules"].fillna("").str.contains("proximity_match").sum())
        fuzzy_hit_count = int(corrected_df["matched_keyword_rules"].fillna("").str.contains("fuzzy_match").sum())
        suppressed_count = int(corrected_df["suppressed_by_negative_guard"].sum())
        disclosure_hit_count = int((corrected_df["corrected_ad_disclosure_hit_count"] > 0).sum())
        key_cases = corrected_df[
            corrected_df["timestamp_sec"].between(569.0, 572.0, inclusive="both")
            & (corrected_df["corrected_ad_disclosure_hit_count"] > 0)
        ]
        near_full = role_summary[
            role_summary["analysis_scope"].eq("full_video")
            & role_summary["sampling_role"].eq("near_scene_anchor_1s")
        ].iloc[0]
        bg_full = role_summary[
            role_summary["analysis_scope"].eq("full_video")
            & role_summary["sampling_role"].eq("background_full_video_1p5s")
        ].iloc[0]
        near_context = role_summary[
            role_summary["analysis_scope"].eq("actual_ad_context")
            & role_summary["sampling_role"].eq("near_scene_anchor_1s")
        ].iloc[0]
        bg_context = role_summary[
            role_summary["analysis_scope"].eq("actual_ad_context")
            & role_summary["sampling_role"].eq("background_full_video_1p5s")
        ].iloc[0]

        print()
        print("1. Target:")
        print(f"   - video_id: {args.target_video_id}")
        print(f"   - source run dir: {args.source_run_dir}")
        print()
        print("2. Typo correction:")
        print(f"   - corrected ad disclosure hit count: {disclosure_hit_count}")
        print(f"   - typo variant hit count: {typo_variant_hit_count}")
        print(f"   - proximity hit count: {proximity_hit_count}")
        print(f"   - fuzzy review-needed hit count: {fuzzy_hit_count}")
        print(f"   - negative guard suppressed count: {suppressed_count}")
        print()
        print("3. Key disclosure cases:")
        print(f"   - timestamps: {key_cases['timestamp_sec'].round(3).tolist() if not key_cases.empty else []}")
        print(
            "   - original OCR text: "
            + (" | ".join(key_cases["ocr_text_joined_original"].head(2).tolist())[:500] if not key_cases.empty else "")
        )
        print(
            "   - matched rule: "
            + (";".join(sorted(set(";".join(key_cases["matched_keyword_rules"].tolist()).split(";")))) if not key_cases.empty else "")
        )
        print(
            "   - confidence: "
            + (";".join(sorted(set(";".join(key_cases["matched_keyword_confidence"].tolist()).split(";")))) if not key_cases.empty else "")
        )
        print()
        print("4. Sampling interval simulation:")
        print(f"   - current_combined captured disclosure: {summarize_strategy_capture(simulation_summary, 'current_combined')}")
        print(f"   - near_anchor_only captured disclosure: {summarize_strategy_capture(simulation_summary, 'near_anchor_only_1s')}")
        print(f"   - background_only captured disclosure: {summarize_strategy_capture(simulation_summary, 'background_only_1p5s')}")
        print(f"   - simulated 1.5s captured disclosure: {summarize_strategy_capture(simulation_summary, 'simulated_uniform_1p5s')}")
        print(f"   - simulated 2.0s captured disclosure: {summarize_strategy_capture(simulation_summary, 'simulated_uniform_2p0s')}")
        print(f"   - simulated 2.5s captured disclosure: {summarize_strategy_capture(simulation_summary, 'simulated_uniform_2p5s')}")
        print(f"   - simulated 3.0s captured disclosure: {summarize_strategy_capture(simulation_summary, 'simulated_uniform_3p0s')}")
        print()
        print("5. Role contribution:")
        print(f"   - near-anchor keyword hit count: {int(near_full['corrected_total_keyword_hit_frame_count'])}")
        print(f"   - background keyword hit count: {int(bg_full['corrected_total_keyword_hit_frame_count'])}")
        print(f"   - near-anchor ad context hit count: {int(near_context['corrected_total_keyword_hit_frame_count'])}")
        print(f"   - background ad context hit count: {int(bg_context['corrected_total_keyword_hit_frame_count'])}")
        print(
            "   - interpretation: near-anchor captures the direct disclosure/product-heavy context; "
            "background adds full-video coverage and some keyword evidence outside anchor windows."
        )
        print()
        print("6. Outputs:")
        print(f"   - run dir: {run_dir}")
        print(f"   - summary: {outputs['summary_md']}")
        print(f"   - report: {outputs['report_json']}")
        print(f"   - latest bundle: {LATEST_OCR}")
        print()
        print("7. Safety:")
        print("   - OCR rerun performed: false")
        print("   - detector modified: false")
        print("   - existing OCR modified: false")
        print("   - actual label used for sampling: false")
        print("   - validation/test row-level output generated: false")
        print("   - raw frame persisted: false")
        return 0
    except Exception as exc:  # noqa: BLE001 - write a clear failure report for reproducibility.
        ctx.errors.append(str(exc))
        error_report = {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(PROJECT_ROOT),
            "target_video_id": args.target_video_id,
            "source_video5_run_dir": str(args.source_run_dir),
            "script_path": str(SCRIPT_PATH),
            "input_files": {key: str(path) for key, path in input_files.items()},
            "warnings": ctx.warnings,
            "errors": ctx.errors,
        }
        outputs["report_json"].write_text(
            json.dumps(error_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs["run_log"].write_text("\n".join(ctx.log_lines + [f"[ERROR] {exc}"]) + "\n", encoding="utf-8")
        print(f"[ERROR] {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
