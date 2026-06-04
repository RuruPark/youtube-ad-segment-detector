#!/usr/bin/env python3
"""Build Development Set ad-context OCR review sheets for v2.5.

This script is post-hoc documentation/review preparation. It reads an existing
Development Set OCR frame result and actual ad intervals, then filters OCR rows
inside [ad_start_sec - 5, ad_end_sec]. It does not run OCR, does not call an OCR
engine, does not create frame images, and does not modify detector artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
SOURCE_OCR_RUN_ID = "final_scene_anchor_ocr_v2_5_development_20260527_050904"
SOURCE_OCR_RUN_DIR = (
    DEFAULT_PROJECT_ROOT
    / "workspaces/ocr_final_scene_anchor_v2_5_development/runs"
    / SOURCE_OCR_RUN_ID
)
SCRIPT_RELATIVE_PATH = "scripts/ocr/build_development_ad_context_ocr_review_sheet_v2_5.py"
DEVELOPMENT_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
EXTENDED_EVALUATION_IDS = [3, 7, 18, 4, 16, 17]
DIAGNOSTIC_IDS = [3, 7, 18]
PURE_TEST_IDS = [4, 16, 17]
FORBIDDEN_BUNDLE_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".bin",
}
LOW_CONFIDENCE_THRESHOLD_DEFAULT = 0.50


class TaskLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_float(value: Any) -> float:
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if not isinstance(obj, (str, bytes, bytearray)):
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
    return obj


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snapshot_path(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "file_count": 0, "metadata_digest": None}
    if path.is_file():
        targets = [path]
        base = path.parent
    else:
        targets = [p for p in path.rglob("*") if p.is_file()]
        base = path
    rows: List[str] = []
    for item in targets:
        try:
            stat = item.stat()
        except OSError:
            continue
        rows.append(f"{item.relative_to(base).as_posix()}\t{stat.st_size}\t{stat.st_mtime_ns}")
    return {
        "path": str(path),
        "exists": True,
        "file_count": len(rows),
        "metadata_digest": hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest(),
    }


def compare_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
    return before.get("exists") == after.get("exists") and before.get("metadata_digest") == after.get("metadata_digest")


def locate_file(exact: Path, fallback_roots: Sequence[Path], filename: str) -> Path:
    if exact.exists():
        return exact
    matches: List[Path] = []
    for root in fallback_roots:
        if root.exists():
            matches.extend(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not locate {filename}; checked {exact} and fallback roots {fallback_roots}")
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def timestamp_mmss(value: Any) -> str:
    seconds = safe_float(value)
    if not np.isfinite(seconds):
        return ""
    seconds = max(0.0, seconds)
    minute = int(seconds // 60)
    sec = int(round(seconds - minute * 60))
    if sec >= 60:
        minute += 1
        sec -= 60
    return f"{minute:02d}:{sec:02d}"


def ensure_columns(df: pd.DataFrame, columns: Iterable[str], default: Any = "") -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = default
    return out


def as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def split_keywords(text: Any) -> List[str]:
    if pd.isna(text):
        return []
    raw = str(text).replace("|", ";").replace(",", ";")
    return [item.strip() for item in raw.split(";") if item.strip()]


def top_values(series: pd.Series, max_items: int = 8) -> str:
    counts: Dict[str, int] = {}
    for value in series.dropna().astype(str):
        for item in split_keywords(value):
            counts[item] = counts.get(item, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return "; ".join(f"{key}:{value}" for key, value in ordered[:max_items])


def first_nonempty_text(series: pd.Series, max_len: int = 220) -> str:
    for value in series.fillna("").astype(str):
        text = value.strip()
        if text:
            return text[:max_len]
    return ""


def short_title(value: Any, max_len: int = 42) -> str:
    text = "" if pd.isna(value) else str(value)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    idx = 2
    while True:
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def build_output_paths(run_dir: Path) -> Dict[str, Path]:
    return {
        "full_review_sheet": run_dir / "development_ad_start_minus5_to_ad_end_ocr_review_sheet_v2_5.csv",
        "compact_review_sheet": run_dir / "development_ad_start_minus5_to_ad_end_ocr_review_compact_v2_5.csv",
        "markdown_report": run_dir / "development_ad_start_minus5_to_ad_end_ocr_review_sheet_v2_5.md",
        "interval_summary": run_dir / "development_ad_context_ocr_quality_summary_by_interval_v2_5.csv",
        "video_summary": run_dir / "development_ad_context_ocr_quality_summary_by_video_v2_5.csv",
        "keyword_hit_review": run_dir / "development_ad_context_ocr_keyword_hit_review_v2_5.csv",
        "disclosure_review": run_dir / "development_ad_context_ocr_disclosure_review_v2_5.csv",
        "empty_lowconf_review": run_dir / "development_ad_context_ocr_empty_lowconf_review_v2_5.csv",
        "representative_examples": run_dir / "development_ad_context_ocr_representative_examples_v2_5.csv",
        "quality_checks": run_dir / "development_ad_context_ocr_review_quality_checks_v2_5.csv",
        "json_report": run_dir / "development_ad_context_ocr_review_v2_5_report.json",
        "run_log": run_dir / "development_ad_context_ocr_review_v2_5_run_log.txt",
    }


def load_inputs(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    ocr_df = pd.read_csv(paths["ocr_frame_result"], encoding="utf-8-sig")
    label_df = pd.read_csv(paths["label_file"], encoding="utf-8-sig")
    split_v25_df = pd.read_csv(paths["split_v25_file"], encoding="utf-8-sig")
    manifest_df = pd.read_csv(paths["manifest_file"], encoding="utf-8-sig")
    if paths["ocr_report"].exists():
        with paths["ocr_report"].open("r", encoding="utf-8") as fh:
            ocr_report = json.load(fh)
    else:
        ocr_report = {}
    return ocr_df, label_df, split_v25_df, manifest_df, ocr_report


def build_development_intervals(label_df: pd.DataFrame, split_v25_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    dev_map = split_v25_df[split_v25_df["split_role_v2_5"].astype(str).eq("development")].copy()
    dev_map["video_id"] = pd.to_numeric(dev_map["video_id"], errors="coerce").astype(int)
    if "split_terminology_note" not in dev_map.columns:
        dev_map["split_terminology_note"] = (
            "Development Set = original v2.4 train split; used for rule design, cue analysis, "
            "and error diagnosis, not ML model training."
        )
    dev_ids = sorted(dev_map["video_id"].tolist())
    if dev_ids != DEVELOPMENT_IDS:
        raise RuntimeError(f"Development Set mismatch: expected {DEVELOPMENT_IDS}, observed {dev_ids}")
    intervals = label_df[label_df["segment_type"].astype(str).eq("ad_interval")].copy()
    intervals["video_id"] = pd.to_numeric(intervals["video_id"], errors="coerce").astype(int)
    intervals = intervals[intervals["video_id"].isin(DEVELOPMENT_IDS)].copy()
    if "segment_valid" in intervals.columns:
        intervals = intervals[intervals["segment_valid"].astype(str).str.lower().isin(["true", "1", "yes"])]
    intervals["ad_start_sec"] = pd.to_numeric(intervals["ad_start_sec"], errors="coerce")
    intervals["ad_end_sec"] = pd.to_numeric(intervals["ad_end_sec"], errors="coerce")
    intervals = intervals[intervals["ad_end_sec"].gt(intervals["ad_start_sec"])].copy()
    intervals["ad_duration_sec"] = intervals["ad_end_sec"] - intervals["ad_start_sec"]
    intervals["review_window_start_sec"] = (intervals["ad_start_sec"] - 5.0).clip(lower=0.0)
    intervals["review_window_end_sec"] = intervals["ad_end_sec"]
    intervals["review_window_duration_sec"] = intervals["review_window_end_sec"] - intervals["review_window_start_sec"]
    keep_cols = [
        "video_id",
        "segment_id",
        "ad_interval_id",
        "ad_start_sec",
        "ad_end_sec",
        "ad_duration_sec",
        "review_window_start_sec",
        "review_window_end_sec",
        "review_window_duration_sec",
        "video_title",
    ]
    intervals = intervals[keep_cols].copy()
    intervals = intervals.merge(
        dev_map[
            [
                "video_id",
                "original_split_v2_4",
                "split_role_v2_5",
                "evaluation_subset_v2_5",
                "split_terminology_note",
            ]
        ],
        on="video_id",
        how="left",
    )
    if "video_title" not in manifest_df.columns:
        return intervals
    manifest_titles = manifest_df[["video_id", "video_title"]].copy()
    manifest_titles["video_id"] = pd.to_numeric(manifest_titles["video_id"], errors="coerce").astype(int)
    intervals = intervals.merge(manifest_titles.rename(columns={"video_title": "manifest_video_title"}), on="video_id", how="left")
    intervals["video_title"] = intervals["video_title"].fillna(intervals["manifest_video_title"])
    intervals = intervals.drop(columns=["manifest_video_title"])
    return intervals.sort_values(["video_id", "ad_interval_id"]).reset_index(drop=True)


def filter_review_rows(ocr_df: pd.DataFrame, intervals: pd.DataFrame, source_run_id: str) -> pd.DataFrame:
    ocr = ocr_df.copy()
    ocr["video_id"] = pd.to_numeric(ocr["video_id"], errors="coerce").astype(int)
    ocr["timestamp_sec"] = pd.to_numeric(ocr["timestamp_sec"], errors="coerce")
    rows: List[pd.DataFrame] = []
    for _, interval in intervals.iterrows():
        video_id = safe_int(interval["video_id"])
        start = safe_float(interval["review_window_start_sec"])
        end = safe_float(interval["review_window_end_sec"])
        group = ocr[ocr["video_id"].eq(video_id) & ocr["timestamp_sec"].ge(start - 1e-9) & ocr["timestamp_sec"].le(end + 1e-9)].copy()
        for col in [
            "original_split_v2_4",
            "split_role_v2_5",
            "evaluation_subset_v2_5",
            "split_terminology_note",
            "video_title",
            "ad_interval_id",
            "segment_id",
            "ad_start_sec",
            "ad_end_sec",
            "ad_duration_sec",
            "review_window_start_sec",
            "review_window_end_sec",
        ]:
            group[col] = interval.get(col, "")
        rows.append(group)
    if rows:
        review = pd.concat(rows, ignore_index=True)
    else:
        review = pd.DataFrame(columns=list(ocr.columns) + list(intervals.columns))
    review = review.sort_values(["video_id", "ad_interval_id", "timestamp_sec"]).reset_index(drop=True)
    review.insert(0, "review_row_id", [f"DEV_OCR_REVIEW_{i + 1:06d}" for i in range(len(review))])
    review["run_id"] = source_run_id
    review["timestamp_mmss"] = review["timestamp_sec"].map(timestamp_mmss)
    review["relative_to_ad_start_sec"] = pd.to_numeric(review["timestamp_sec"], errors="coerce") - pd.to_numeric(review["ad_start_sec"], errors="coerce")
    review["relative_to_ad_end_sec"] = pd.to_numeric(review["timestamp_sec"], errors="coerce") - pd.to_numeric(review["ad_end_sec"], errors="coerce")
    review["phase_in_review_window"] = np.where(review["timestamp_sec"] < review["ad_start_sec"], "pre_start_5s", "ad_body")
    review["has_text"] = (
        pd.to_numeric(review.get("ocr_text_count", pd.Series(0, index=review.index)), errors="coerce").fillna(0).gt(0)
        | review.get("ocr_text_joined", pd.Series("", index=review.index)).fillna("").astype(str).str.strip().ne("")
    )
    review["reviewer_note"] = ""
    review["ocr_quality_manual_label"] = ""
    return review


def add_review_flags(review: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = review.copy()
    numeric_cols = [
        "ocr_mean_confidence",
        "ocr_text_count",
        "corrected_ad_disclosure_hit_count",
        "corrected_sponsor_keyword_count",
        "corrected_brand_product_keyword_count",
        "corrected_promotion_discount_keyword_count",
        "corrected_purchase_cta_keyword_count",
        "corrected_link_more_info_keyword_count",
        "corrected_negative_guard_keyword_count",
        "corrected_total_ad_keyword_count",
        "corrected_frame_ad_text_score",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    score = df["corrected_frame_ad_text_score"].dropna()
    if score.empty:
        high_score_threshold = 0.50
        high_score_threshold_source = "fallback_0.50_no_score_values"
    else:
        q90 = float(score.quantile(0.90))
        high_score_threshold = max(0.50, q90)
        high_score_threshold_source = "max_0.50_and_review_q90"
    low_conf_threshold = LOW_CONFIDENCE_THRESHOLD_DEFAULT

    def quick_flag(row: pd.Series) -> str:
        if safe_float(row.get("corrected_ad_disclosure_hit_count")) > 0:
            return "disclosure_hit"
        if safe_float(row.get("corrected_sponsor_keyword_count")) > 0:
            return "sponsor_hit"
        product_cta = (
            safe_float(row.get("corrected_brand_product_keyword_count"))
            + safe_float(row.get("corrected_purchase_cta_keyword_count"))
            + safe_float(row.get("corrected_link_more_info_keyword_count"))
        )
        if product_cta > 0:
            return "product_or_cta_hit"
        if safe_float(row.get("corrected_frame_ad_text_score")) >= high_score_threshold:
            return "high_ad_score"
        if str(row.get("ocr_status", "")) == "success_empty":
            return "empty"
        if str(row.get("ocr_status", "")) == "failed":
            return "failed"
        conf = safe_float(row.get("ocr_mean_confidence"))
        if np.isfinite(conf) and conf < low_conf_threshold:
            return "low_confidence"
        return "normal"

    df["quick_quality_flag"] = df.apply(quick_flag, axis=1)
    df["is_low_confidence"] = pd.to_numeric(df["ocr_mean_confidence"], errors="coerce").lt(low_conf_threshold)
    df["has_corrected_keyword_hit"] = pd.to_numeric(df["corrected_total_ad_keyword_count"], errors="coerce").fillna(0).gt(0)
    rules = df.get("matched_keyword_rules", pd.Series("", index=df.index)).fillna("").astype(str)
    categories = df.get("matched_keyword_categories", pd.Series("", index=df.index)).fillna("").astype(str)
    suggested = df.get("suggested_canonical_phrase", pd.Series("", index=df.index)).fillna("").astype(str)
    df["is_typo_variant_hit"] = rules.str.contains("typo_variant_match", case=False, regex=False)
    df["is_fuzzy_review_needed"] = rules.str.contains("fuzzy_match", case=False, regex=False)
    df["is_disclosure_related"] = (
        pd.to_numeric(df["corrected_ad_disclosure_hit_count"], errors="coerce").fillna(0).gt(0)
        | categories.str.contains("ad_disclosure", case=False, regex=False)
        | rules.str.contains("typo_variant_match", case=False, regex=False)
        | rules.str.contains("proximity_match", case=False, regex=False)
        | rules.str.contains("fuzzy_match", case=False, regex=False)
        | suggested.str.contains("유료광고|광고포함", regex=True)
    )

    def disclosure_type(row: pd.Series) -> str:
        rule_text = str(row.get("matched_keyword_rules", ""))
        types = []
        if safe_float(row.get("corrected_ad_disclosure_hit_count")) > 0:
            types.append("exact")
        if "typo_variant_match" in rule_text:
            types.append("typo_variant")
        if "proximity_match" in rule_text:
            types.append("proximity")
        if "fuzzy_match" in rule_text:
            types.append("fuzzy_review_needed")
        unique = list(dict.fromkeys(types))
        if len(unique) > 1:
            return "mixed"
        return unique[0] if unique else "mixed"

    df["disclosure_review_type"] = df.apply(disclosure_type, axis=1)
    metadata = {
        "high_score_threshold": high_score_threshold,
        "high_score_threshold_source": high_score_threshold_source,
        "low_confidence_threshold": low_conf_threshold,
    }
    return df, metadata


FULL_COLUMNS = [
    "review_row_id",
    "run_id",
    "original_split_v2_4",
    "split_role_v2_5",
    "evaluation_subset_v2_5",
    "video_id",
    "video_title",
    "ad_interval_id",
    "ad_start_sec",
    "ad_end_sec",
    "ad_duration_sec",
    "review_window_start_sec",
    "review_window_end_sec",
    "timestamp_sec",
    "timestamp_mmss",
    "relative_to_ad_start_sec",
    "relative_to_ad_end_sec",
    "phase_in_review_window",
    "sampling_role",
    "is_anchor_dense",
    "is_background_regular",
    "nearest_anchor_id",
    "nearest_anchor_time_sec",
    "nearest_anchor_delta_sec",
    "ocr_status",
    "has_text",
    "ocr_text_joined",
    "ocr_text_normalized",
    "ocr_text_count",
    "ocr_token_count",
    "ocr_char_count",
    "ocr_box_count",
    "ocr_mean_confidence",
    "ocr_min_confidence",
    "ocr_max_confidence",
    "ocr_text_area_ratio",
    "ad_disclosure_keyword_count",
    "sponsor_keyword_count",
    "brand_product_keyword_count",
    "promotion_discount_keyword_count",
    "purchase_cta_keyword_count",
    "link_more_info_keyword_count",
    "negative_guard_keyword_count",
    "total_ad_keyword_count",
    "corrected_ad_disclosure_hit_count",
    "corrected_sponsor_keyword_count",
    "corrected_brand_product_keyword_count",
    "corrected_promotion_discount_keyword_count",
    "corrected_purchase_cta_keyword_count",
    "corrected_link_more_info_keyword_count",
    "corrected_negative_guard_keyword_count",
    "corrected_total_ad_keyword_count",
    "matched_keyword_categories",
    "matched_keywords",
    "matched_keyword_rules",
    "matched_keyword_confidence",
    "suggested_canonical_phrase",
    "suppressed_by_negative_guard",
    "corrected_frame_ad_text_score",
    "reviewer_note",
    "ocr_quality_manual_label",
]


def build_full_sheet(review: pd.DataFrame) -> pd.DataFrame:
    return ensure_columns(review, FULL_COLUMNS, "")[FULL_COLUMNS].copy()


def build_compact_sheet(review: pd.DataFrame) -> pd.DataFrame:
    df = review.copy()
    df["video_title_short"] = df["video_title"].map(short_title)
    cols = [
        "video_id",
        "video_title_short",
        "ad_interval_id",
        "timestamp_mmss",
        "timestamp_sec",
        "relative_to_ad_start_sec",
        "phase_in_review_window",
        "sampling_role",
        "ocr_status",
        "has_text",
        "ocr_mean_confidence",
        "ocr_text_joined",
        "matched_keyword_categories",
        "matched_keywords",
        "matched_keyword_rules",
        "corrected_total_ad_keyword_count",
        "corrected_frame_ad_text_score",
        "quick_quality_flag",
        "reviewer_note",
    ]
    return ensure_columns(df, cols, "")[cols].copy()


def build_keyword_review(review: pd.DataFrame) -> pd.DataFrame:
    df = review.copy()
    df["video_title_short"] = df["video_title"].map(short_title)
    mask = (
        pd.to_numeric(df["corrected_total_ad_keyword_count"], errors="coerce").fillna(0).gt(0)
        | df.get("matched_keywords", pd.Series("", index=df.index)).fillna("").astype(str).str.strip().ne("")
    )
    out = df[mask].copy()
    out["review_reason"] = np.where(
        pd.to_numeric(out["corrected_total_ad_keyword_count"], errors="coerce").fillna(0).gt(0),
        "corrected_keyword_count_positive",
        "matched_keywords_present",
    )
    cols = [
        "video_id",
        "video_title_short",
        "ad_interval_id",
        "timestamp_mmss",
        "timestamp_sec",
        "relative_to_ad_start_sec",
        "phase_in_review_window",
        "sampling_role",
        "ocr_text_joined",
        "matched_keyword_categories",
        "matched_keywords",
        "matched_keyword_rules",
        "matched_keyword_confidence",
        "corrected_ad_disclosure_hit_count",
        "corrected_sponsor_keyword_count",
        "corrected_brand_product_keyword_count",
        "corrected_promotion_discount_keyword_count",
        "corrected_purchase_cta_keyword_count",
        "corrected_link_more_info_keyword_count",
        "corrected_frame_ad_text_score",
        "suggested_canonical_phrase",
        "review_reason",
        "reviewer_note",
    ]
    return ensure_columns(out, cols, "")[cols].copy()


def build_disclosure_review(review: pd.DataFrame) -> pd.DataFrame:
    df = review.copy()
    df["video_title_short"] = df["video_title"].map(short_title)
    out = df[df["is_disclosure_related"]].copy()
    cols = [
        "video_id",
        "video_title_short",
        "ad_interval_id",
        "ad_start_sec",
        "ad_end_sec",
        "timestamp_mmss",
        "timestamp_sec",
        "relative_to_ad_start_sec",
        "phase_in_review_window",
        "ocr_text_joined",
        "ocr_text_normalized",
        "matched_keywords",
        "matched_keyword_rules",
        "matched_keyword_confidence",
        "suggested_canonical_phrase",
        "corrected_ad_disclosure_hit_count",
        "suppressed_by_negative_guard",
        "disclosure_review_type",
        "reviewer_note",
    ]
    return ensure_columns(out, cols, "")[cols].copy()


def build_empty_lowconf_review(review: pd.DataFrame, low_conf_threshold: float) -> pd.DataFrame:
    df = review.copy()
    df["video_title_short"] = df["video_title"].map(short_title)
    text_count = pd.to_numeric(df["ocr_text_count"], errors="coerce").fillna(0)
    conf = pd.to_numeric(df["ocr_mean_confidence"], errors="coerce")
    mask = (
        df["ocr_status"].astype(str).eq("success_empty")
        | df["ocr_status"].astype(str).eq("failed")
        | conf.lt(low_conf_threshold)
        | (text_count.eq(0) & df["phase_in_review_window"].eq("ad_body"))
    )
    out = df[mask].copy()

    def reason(row: pd.Series) -> str:
        reasons = []
        if str(row.get("ocr_status")) == "success_empty":
            reasons.append("success_empty")
        if str(row.get("ocr_status")) == "failed":
            reasons.append("failed")
        conf_val = safe_float(row.get("ocr_mean_confidence"))
        if np.isfinite(conf_val) and conf_val < low_conf_threshold:
            reasons.append("low_confidence")
        if safe_float(row.get("ocr_text_count")) == 0 and row.get("phase_in_review_window") == "ad_body":
            reasons.append("zero_text_inside_ad_body")
        return ";".join(reasons) or "review_needed"

    out["review_reason"] = out.apply(reason, axis=1)
    cols = [
        "video_id",
        "video_title_short",
        "ad_interval_id",
        "timestamp_mmss",
        "timestamp_sec",
        "relative_to_ad_start_sec",
        "phase_in_review_window",
        "sampling_role",
        "ocr_status",
        "ocr_mean_confidence",
        "ocr_text_count",
        "ocr_text_joined",
        "nearest_anchor_id",
        "nearest_anchor_delta_sec",
        "review_reason",
        "reviewer_note",
    ]
    return ensure_columns(out, cols, "")[cols].copy()


def summarize_interval(group: pd.DataFrame) -> Dict[str, Any]:
    low_conf = group["is_low_confidence"].fillna(False)
    nonempty = group["has_text"].fillna(False)
    disclosure_sum = pd.to_numeric(group["corrected_ad_disclosure_hit_count"], errors="coerce").fillna(0).sum()
    fuzzy_count = int(group["is_fuzzy_review_needed"].fillna(False).sum())
    typo_count = int(group["is_typo_variant_hit"].fillna(False).sum())
    keyword_rows = int(group["has_corrected_keyword_hit"].fillna(False).sum())
    low_count = int(low_conf.sum())
    if disclosure_sum > 0 or fuzzy_count > 0 or low_count >= max(3, int(len(group) * 0.25)) or group["quick_quality_flag"].eq("high_ad_score").any():
        priority = "high"
    elif keyword_rows > 0:
        priority = "medium"
    else:
        priority = "low"
    disclosure_times = group.loc[pd.to_numeric(group["corrected_ad_disclosure_hit_count"], errors="coerce").fillna(0).gt(0), "timestamp_sec"]
    keyword_times = group.loc[group["has_corrected_keyword_hit"].fillna(False), "timestamp_sec"]
    return {
        "original_split_v2_4": first_nonempty_text(group["original_split_v2_4"], 80),
        "split_role_v2_5": first_nonempty_text(group["split_role_v2_5"], 80),
        "evaluation_subset_v2_5": first_nonempty_text(group["evaluation_subset_v2_5"], 80),
        "video_id": safe_int(group["video_id"].iloc[0]),
        "video_title": group["video_title"].iloc[0],
        "ad_interval_id": group["ad_interval_id"].iloc[0],
        "ad_start_sec": safe_float(group["ad_start_sec"].iloc[0]),
        "ad_end_sec": safe_float(group["ad_end_sec"].iloc[0]),
        "ad_duration_sec": safe_float(group["ad_duration_sec"].iloc[0]),
        "review_window_start_sec": safe_float(group["review_window_start_sec"].iloc[0]),
        "review_window_end_sec": safe_float(group["review_window_end_sec"].iloc[0]),
        "review_window_duration_sec": safe_float(group["review_window_end_sec"].iloc[0]) - safe_float(group["review_window_start_sec"].iloc[0]),
        "review_row_count": int(len(group)),
        "pre_start_5s_row_count": int(group["phase_in_review_window"].eq("pre_start_5s").sum()),
        "ad_body_row_count": int(group["phase_in_review_window"].eq("ad_body").sum()),
        "success_text_count": int(group["ocr_status"].astype(str).eq("success_text").sum()),
        "success_empty_count": int(group["ocr_status"].astype(str).eq("success_empty").sum()),
        "failed_count": int(group["ocr_status"].astype(str).eq("failed").sum()),
        "nonempty_count": int(nonempty.sum()),
        "nonempty_ratio": float(nonempty.mean()) if len(group) else np.nan,
        "mean_confidence": safe_float(pd.to_numeric(group["ocr_mean_confidence"], errors="coerce").mean()),
        "median_confidence": safe_float(pd.to_numeric(group["ocr_mean_confidence"], errors="coerce").median()),
        "low_confidence_count": low_count,
        "corrected_keyword_hit_row_count": keyword_rows,
        "corrected_total_ad_keyword_count_sum": safe_float(pd.to_numeric(group["corrected_total_ad_keyword_count"], errors="coerce").fillna(0).sum()),
        "corrected_ad_disclosure_hit_count_sum": safe_float(disclosure_sum),
        "corrected_sponsor_keyword_count_sum": safe_float(pd.to_numeric(group["corrected_sponsor_keyword_count"], errors="coerce").fillna(0).sum()),
        "corrected_brand_product_keyword_count_sum": safe_float(pd.to_numeric(group["corrected_brand_product_keyword_count"], errors="coerce").fillna(0).sum()),
        "corrected_promotion_discount_keyword_count_sum": safe_float(pd.to_numeric(group["corrected_promotion_discount_keyword_count"], errors="coerce").fillna(0).sum()),
        "corrected_purchase_cta_keyword_count_sum": safe_float(pd.to_numeric(group["corrected_purchase_cta_keyword_count"], errors="coerce").fillna(0).sum()),
        "corrected_link_more_info_keyword_count_sum": safe_float(pd.to_numeric(group["corrected_link_more_info_keyword_count"], errors="coerce").fillna(0).sum()),
        "corrected_negative_guard_keyword_count_sum": safe_float(pd.to_numeric(group["corrected_negative_guard_keyword_count"], errors="coerce").fillna(0).sum()),
        "typo_variant_hit_count": typo_count,
        "fuzzy_review_needed_count": fuzzy_count,
        "first_disclosure_timestamp_sec": safe_float(disclosure_times.min()) if len(disclosure_times) else np.nan,
        "first_keyword_timestamp_sec": safe_float(keyword_times.min()) if len(keyword_times) else np.nan,
        "representative_ocr_text": first_nonempty_text(group["ocr_text_joined"], 260),
        "top_matched_keywords": top_values(group.get("matched_keywords", pd.Series(dtype=str))),
        "suggested_review_priority": priority,
        "interval_quality_note": f"{priority}_priority; keyword_rows={keyword_rows}; disclosure_hits={int(disclosure_sum)}; low_conf_rows={low_count}",
    }


def build_interval_summary(review: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, interval in intervals.iterrows():
        group = review[
            review["video_id"].eq(interval["video_id"])
            & review["ad_interval_id"].astype(str).eq(str(interval["ad_interval_id"]))
        ].copy()
        if group.empty:
            empty = {
                "original_split_v2_4": interval.get("original_split_v2_4", "train"),
                "split_role_v2_5": interval.get("split_role_v2_5", "development"),
                "evaluation_subset_v2_5": interval.get("evaluation_subset_v2_5", "none"),
                "video_id": safe_int(interval["video_id"]),
                "video_title": interval.get("video_title", ""),
                "ad_interval_id": interval.get("ad_interval_id", ""),
                "ad_start_sec": safe_float(interval.get("ad_start_sec")),
                "ad_end_sec": safe_float(interval.get("ad_end_sec")),
                "ad_duration_sec": safe_float(interval.get("ad_duration_sec")),
                "review_window_start_sec": safe_float(interval.get("review_window_start_sec")),
                "review_window_end_sec": safe_float(interval.get("review_window_end_sec")),
                "review_window_duration_sec": safe_float(interval.get("review_window_duration_sec")),
                "review_row_count": 0,
                "pre_start_5s_row_count": 0,
                "ad_body_row_count": 0,
                "success_text_count": 0,
                "success_empty_count": 0,
                "failed_count": 0,
                "nonempty_count": 0,
                "nonempty_ratio": np.nan,
                "mean_confidence": np.nan,
                "median_confidence": np.nan,
                "low_confidence_count": 0,
                "corrected_keyword_hit_row_count": 0,
                "corrected_total_ad_keyword_count_sum": 0,
                "corrected_ad_disclosure_hit_count_sum": 0,
                "corrected_sponsor_keyword_count_sum": 0,
                "corrected_brand_product_keyword_count_sum": 0,
                "corrected_promotion_discount_keyword_count_sum": 0,
                "corrected_purchase_cta_keyword_count_sum": 0,
                "corrected_link_more_info_keyword_count_sum": 0,
                "corrected_negative_guard_keyword_count_sum": 0,
                "typo_variant_hit_count": 0,
                "fuzzy_review_needed_count": 0,
                "first_disclosure_timestamp_sec": np.nan,
                "first_keyword_timestamp_sec": np.nan,
                "representative_ocr_text": "",
                "top_matched_keywords": "",
                "suggested_review_priority": "high",
                "interval_quality_note": "WARN_no_ocr_rows_in_review_window",
            }
            rows.append(empty)
        else:
            rows.append(summarize_interval(group))
    return pd.DataFrame(rows)


def build_video_summary(interval_summary: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for video_id, intervals in interval_summary.groupby("video_id", sort=True):
        group = review[review["video_id"].eq(video_id)].copy()
        nonempty = group["has_text"].fillna(False) if not group.empty else pd.Series(dtype=bool)
        rows.append(
            {
                "original_split_v2_4": first_nonempty_text(intervals["original_split_v2_4"], 80),
                "split_role_v2_5": first_nonempty_text(intervals["split_role_v2_5"], 80),
                "evaluation_subset_v2_5": first_nonempty_text(intervals["evaluation_subset_v2_5"], 80),
                "video_id": int(video_id),
                "video_title": intervals["video_title"].iloc[0],
                "ad_interval_count": int(len(intervals)),
                "review_row_count": int(len(group)),
                "success_text_count": int(group["ocr_status"].astype(str).eq("success_text").sum()) if not group.empty else 0,
                "success_empty_count": int(group["ocr_status"].astype(str).eq("success_empty").sum()) if not group.empty else 0,
                "failed_count": int(group["ocr_status"].astype(str).eq("failed").sum()) if not group.empty else 0,
                "nonempty_count": int(nonempty.sum()) if not group.empty else 0,
                "nonempty_ratio": float(nonempty.mean()) if len(group) else np.nan,
                "mean_confidence": safe_float(pd.to_numeric(group.get("ocr_mean_confidence", pd.Series(dtype=float)), errors="coerce").mean()) if not group.empty else np.nan,
                "corrected_keyword_hit_row_count": int(group.get("has_corrected_keyword_hit", pd.Series(dtype=bool)).fillna(False).sum()) if not group.empty else 0,
                "corrected_total_ad_keyword_count_sum": safe_float(pd.to_numeric(group.get("corrected_total_ad_keyword_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not group.empty else 0,
                "corrected_ad_disclosure_hit_count_sum": safe_float(pd.to_numeric(group.get("corrected_ad_disclosure_hit_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not group.empty else 0,
                "low_confidence_count": int(group.get("is_low_confidence", pd.Series(dtype=bool)).fillna(False).sum()) if not group.empty else 0,
                "high_priority_interval_count": int(intervals["suggested_review_priority"].eq("high").sum()),
                "medium_priority_interval_count": int(intervals["suggested_review_priority"].eq("medium").sum()),
                "representative_ocr_text": first_nonempty_text(group.get("ocr_text_joined", pd.Series(dtype=str)), 260) if not group.empty else "",
                "top_matched_keywords": top_values(group.get("matched_keywords", pd.Series(dtype=str))) if not group.empty else "",
            }
        )
    return pd.DataFrame(rows)


def select_representative_examples(review: pd.DataFrame, max_per_group: int = 8) -> pd.DataFrame:
    df = review.copy()
    groups = {
        "disclosure_hit": df[pd.to_numeric(df["corrected_ad_disclosure_hit_count"], errors="coerce").fillna(0).gt(0)],
        "sponsor_hit": df[pd.to_numeric(df["corrected_sponsor_keyword_count"], errors="coerce").fillna(0).gt(0)],
        "product_or_cta_hit": df[
            (
                pd.to_numeric(df["corrected_brand_product_keyword_count"], errors="coerce").fillna(0)
                + pd.to_numeric(df["corrected_purchase_cta_keyword_count"], errors="coerce").fillna(0)
                + pd.to_numeric(df["corrected_link_more_info_keyword_count"], errors="coerce").fillna(0)
            ).gt(0)
        ],
        "high_ad_score": df[df["quick_quality_flag"].eq("high_ad_score")],
        "empty_but_inside_ad": df[df["ocr_status"].astype(str).eq("success_empty") & df["phase_in_review_window"].eq("ad_body")],
        "low_confidence_with_text": df[df["is_low_confidence"].fillna(False) & df["has_text"].fillna(False)],
        "fuzzy_review_needed": df[df["is_fuzzy_review_needed"].fillna(False)],
        "typo_variant_hit": df[df["is_typo_variant_hit"].fillna(False)],
    }
    rows: List[pd.DataFrame] = []
    base_cols = [
        "video_id",
        "video_title",
        "ad_interval_id",
        "timestamp_mmss",
        "timestamp_sec",
        "relative_to_ad_start_sec",
        "phase_in_review_window",
        "sampling_role",
        "ocr_status",
        "ocr_mean_confidence",
        "ocr_text_joined",
        "matched_keyword_categories",
        "matched_keywords",
        "matched_keyword_rules",
        "corrected_frame_ad_text_score",
        "quick_quality_flag",
    ]
    for reason, group in groups.items():
        if group.empty:
            continue
        selected = group.sort_values(["video_id", "ad_interval_id", "timestamp_sec"]).head(max_per_group).copy()
        selected["example_group"] = reason
        rows.append(selected[["example_group"] + base_cols])
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=["example_group"] + base_cols)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "없음"
    small = df.head(max_rows).copy()
    cols = list(small.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in small.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                text = "" if not np.isfinite(value) else f"{value:.4g}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|").replace("\n", " ")[:220])
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    report: Dict[str, Any],
    interval_summary: pd.DataFrame,
    representative_examples: pd.DataFrame,
) -> None:
    frame = report["frame_review_summary"]
    lines = [
        "# Development Set 광고 구간 OCR Review Sheet v2.5",
        "",
        "## 1. 한 문단 결론",
        (
            f"Development Set 광고구간 OCR 품질 확인용 review sheet를 생성했다. "
            f"대상은 {report['review_scope_summary']['video_count']}개 video, "
            f"{report['review_scope_summary']['ad_interval_count']}개 actual ad interval이며, "
            f"[ad_start_sec - 5초, ad_end_sec] 범위의 기존 OCR frame row {frame['total_review_row_count']}개가 포함됐다. "
            f"success_text {frame['success_text_count']}개, success_empty {frame['success_empty_count']}개, failed {frame['failed_count']}개이며, "
            f"우선 확인할 항목은 disclosure/product/CTA keyword hit, 높은 ad text score, empty/low-confidence row다."
        ),
        "",
        "## 2. v2.5 split terminology",
        (
            "본 프로젝트는 학습 기반 모델이 아니라 rule-based detector이므로, 기존 train split은 모델 학습용이 아닌 "
            "Development Set으로 재정의해 rule 설계와 cue 분석, 오류 진단에 사용한다. 데이터 수가 적은 문제를 보완하기 위해 "
            "공개용 설명에서는 기존 validation과 test를 Test Set으로 통합해 규칙 고정 이후 평가 대상으로 설명한다. "
            "기존 validation split은 내부 호환용 key로만 유지한다."
        ),
        "",
        "## 3. 입력 파일 요약",
        f"- OCR frame result path: `{report['source_ocr_frame_result']}`",
        f"- actual ad interval file path: {report['input_files']['label_file']}",
        f"- split file path: {report['input_files']['split_v25_file']}",
        f"- video manifest path: `{report['input_files']['manifest_file']}`",
        f"- OCR run_id: `{report['source_ocr_run_id']}`",
        f"- OCR backend summary: `{report['source_ocr_backend_summary']}`",
        "",
        "## 4. review 범위",
        "- Development Set only",
        "- window definition: `[ad_start_sec - 5, ad_end_sec]`",
        "- included phases: `pre_start_5s`, `ad_body`",
        "- excluded: post-ad window, Test Set",
        "",
        "## 5. review sheet 사용법",
        "- full review CSV는 모든 OCR/keyword/confidence column과 reviewer용 빈 column을 포함한다.",
        "- compact review CSV는 사람이 빠르게 훑어볼 핵심 column만 남겼다.",
        "- `reviewer_note`에는 사람이 직접 본 화면/OCR 오류 메모를 적는다.",
        "- `ocr_quality_manual_label` 권장 값: `good`, `acceptable`, `typo_but_usable`, `wrong_text`, `missed_text`, `empty_but_should_have_text`, `not_relevant`.",
        "- 우선 볼 column: `timestamp_mmss`, `relative_to_ad_start_sec`, `ocr_text_joined`, `matched_keywords`, `corrected_frame_ad_text_score`, `quick_quality_flag`.",
        "",
        "## 6. 전체 OCR 품질 요약",
        f"- total review row count: `{frame['total_review_row_count']}`",
        f"- success_text count: `{frame['success_text_count']}`",
        f"- success_empty count: `{frame['success_empty_count']}`",
        f"- failed count: `{frame['failed_count']}`",
        f"- nonempty ratio: `{frame['nonempty_ratio']}`",
        f"- average confidence: `{frame['average_confidence']}`",
        f"- low confidence row count: `{frame['low_confidence_row_count']}`",
        f"- empty row count: `{frame['empty_row_count']}`",
        f"- keyword hit row count: `{frame['keyword_hit_row_count']}`",
        "",
        "## 7. interval별 요약",
        markdown_table(
            interval_summary[
                [
                    "video_id",
                    "ad_interval_id",
                    "review_row_count",
                    "nonempty_ratio",
                    "corrected_keyword_hit_row_count",
                    "corrected_ad_disclosure_hit_count_sum",
                    "representative_ocr_text",
                    "empty_low_confidence_count_for_report",
                    "suggested_review_priority",
                ]
            ].rename(columns={"empty_low_confidence_count_for_report": "empty/low_conf_count"})
            if "empty_low_confidence_count_for_report" in interval_summary.columns
            else interval_summary[
                [
                    "video_id",
                    "ad_interval_id",
                    "review_row_count",
                    "nonempty_ratio",
                    "corrected_keyword_hit_row_count",
                    "corrected_ad_disclosure_hit_count_sum",
                    "representative_ocr_text",
                    "suggested_review_priority",
                ]
            ],
            max_rows=40,
        ),
        "",
        "## 8. 사람이 우선 확인할 row",
        markdown_table(
            representative_examples[
                [
                    "example_group",
                    "video_id",
                    "ad_interval_id",
                    "timestamp_mmss",
                    "relative_to_ad_start_sec",
                    "ocr_status",
                    "ocr_mean_confidence",
                    "ocr_text_joined",
                    "matched_keywords",
                    "matched_keyword_rules",
                    "corrected_frame_ad_text_score",
                ]
            ],
            max_rows=60,
        ),
        "",
        "## 9. OCR 품질 확인 관점",
        "- 광고 구간에서 실제 화면 텍스트가 OCR에 잘 잡혔는가?",
        "- `유료광고 포함`류 문구가 정상/오타 형태로 잡혔는가?",
        "- 제품명, 브랜드명, 구매/링크/더보기 문구가 잡혔는가?",
        "- empty frame은 실제로 텍스트가 없는 구간일 가능성이 높은가?",
        "- OCR confidence가 낮은데 중요한 텍스트가 있는 row가 있는가?",
        "- 새 OCR 모델 실험이 필요한 오류 유형이 보이는가?",
        "",
        "## 10. safety",
        "- OCR rerun performed: false",
        "- OCR engine called: false",
        "- actual label used for filtering/review only: true",
        "- actual label used for sampling: false",
        "- detector modified: false",
        "- existing OCR modified: false",
        "- Test Set processed: false",
        "- Test Set processed: false",
        "- raw frame persisted: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scan_forbidden_files(directory: Path) -> List[str]:
    if not directory.exists():
        return []
    forbidden = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
            forbidden.append(str(path))
    return forbidden


def copy_latest_files(paths: Dict[str, Path], project_root: Path, run_id: str) -> Dict[str, Any]:
    latest_chatgpt = project_root / "outputs/latest_for_chatgpt_development_ad_context_ocr_review_v2_5"
    latest_ocr = project_root / "outputs/latest_ocr"
    latest_chatgpt.mkdir(parents=True, exist_ok=True)
    latest_ocr.mkdir(parents=True, exist_ok=True)
    include_keys = [
        "script",
        "markdown_report",
        "json_report",
        "run_log",
        "compact_review_sheet",
        "interval_summary",
        "video_summary",
        "keyword_hit_review",
        "disclosure_review",
        "empty_lowconf_review",
        "representative_examples",
        "quality_checks",
    ]
    copied: Dict[str, List[str]] = {"latest_for_chatgpt": [], "latest_ocr": []}
    for dest_name, dest in [("latest_for_chatgpt", latest_chatgpt), ("latest_ocr", latest_ocr)]:
        for key in include_keys:
            src = paths[key]
            if not src.exists() or src.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
                continue
            dst = dest / src.name
            if dst.exists():
                dst = unique_path(dst)
            shutil.copy2(src, dst)
            copied[dest_name].append(str(dst))
        readme = dest / "README_development_ad_context_ocr_review_v2_5.md"
        lines = [
            "# Development Ad Context OCR Review v2.5",
            "",
            f"- run_id: `{run_id}`",
            "- OCR rerun performed: false",
            "- OCR engine called: false",
            "- actual label used for sampling: false",
            "- actual label used for review filtering: true",
            "- Extended Evaluation/Pure Test row-level output: false",
            "",
            "## Copied files",
        ]
        for item in copied[dest_name]:
            lines.append(f"- `{item}`")
        readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
        copied[dest_name].append(str(readme))
    return {
        "latest_for_chatgpt_dir": str(latest_chatgpt),
        "latest_ocr_dir": str(latest_ocr),
        "copied_files": copied,
        "forbidden_files": scan_forbidden_files(latest_chatgpt) + scan_forbidden_files(latest_ocr),
    }


def make_quality_checks(
    input_paths: Dict[str, Path],
    review: pd.DataFrame,
    intervals: pd.DataFrame,
    paths: Dict[str, Path],
    protected_before: Dict[str, Dict[str, Any]],
    protected_targets: Dict[str, Path],
    latest_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    checks: List[Dict[str, str]] = []

    def add(name: str, status: bool | str, detail: str) -> None:
        if isinstance(status, str):
            stat = status
        else:
            stat = "PASS" if status else "FAIL"
        checks.append({"check_name": name, "status": stat, "detail": detail})

    add("input_ocr_frame_result_exists", input_paths["ocr_frame_result"].exists(), str(input_paths["ocr_frame_result"]))
    add("input_label_file_exists", input_paths["label_file"].exists(), str(input_paths["label_file"]))
    output_ids = sorted(pd.to_numeric(review.get("video_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique().tolist())
    add("development_only_filter_passed", set(output_ids).issubset(set(DEVELOPMENT_IDS)), f"output_video_ids={output_ids}")
    add("extended_evaluation_excluded", not bool(set(output_ids) & set(EXTENDED_EVALUATION_IDS)), f"extended_overlap={sorted(set(output_ids) & set(EXTENDED_EVALUATION_IDS))}")
    add("pure_test_excluded", not bool(set(output_ids) & set(PURE_TEST_IDS)), f"pure_test_overlap={sorted(set(output_ids) & set(PURE_TEST_IDS))}")
    add("diagnostic_subset_excluded", not bool(set(output_ids) & set(DIAGNOSTIC_IDS)), f"diagnostic_overlap={sorted(set(output_ids) & set(DIAGNOSTIC_IDS))}")
    add("actual_label_used_for_sampling_false", True, "Existing OCR sample schedule was not changed; labels were only used after OCR for review filtering.")
    add("actual_label_used_for_review_filter_true", True, "Development actual ad intervals were used to define review windows.")
    expected_start = (pd.to_numeric(intervals["ad_start_sec"], errors="coerce") - 5.0).clip(lower=0.0)
    add("review_window_start_equals_ad_start_minus_5_clipped", np.allclose(pd.to_numeric(intervals["review_window_start_sec"], errors="coerce"), expected_start, equal_nan=False), "max(0, ad_start_sec - 5)")
    add("review_window_end_equals_ad_end", np.allclose(pd.to_numeric(intervals["review_window_end_sec"], errors="coerce"), pd.to_numeric(intervals["ad_end_sec"], errors="coerce"), equal_nan=False), "review_window_end_sec == ad_end_sec")
    if review.empty:
        add("all_review_rows_within_window", "WARN", "No review rows found.")
    else:
        in_window = (
            pd.to_numeric(review["timestamp_sec"], errors="coerce").ge(pd.to_numeric(review["review_window_start_sec"], errors="coerce") - 1e-9)
            & pd.to_numeric(review["timestamp_sec"], errors="coerce").le(pd.to_numeric(review["review_window_end_sec"], errors="coerce") + 1e-9)
        )
        add("all_review_rows_within_window", bool(in_window.all()), f"bad_rows={(~in_window).sum()}")
    add("output_has_v2_5_split_columns", all(c in review.columns for c in ["original_split_v2_4", "split_role_v2_5", "evaluation_subset_v2_5", "split_terminology_note"]), "v2.5 columns present")
    add("reviewer_note_columns_present", "reviewer_note" in review.columns, "reviewer_note")
    add("manual_label_columns_present", "ocr_quality_manual_label" in review.columns, "ocr_quality_manual_label")
    add("no_ocr_engine_called", True, "Script imports pandas only and reads existing OCR CSV; no OCR engine invocation.")
    protected_after = {name: snapshot_path(path) for name, path in protected_targets.items()}
    unchanged = {name: compare_snapshots(protected_before[name], protected_after[name]) for name in protected_before}
    add("no_existing_files_modified", all(unchanged.values()), json.dumps(unchanged, ensure_ascii=False))
    forbidden_run_files = scan_forbidden_files(paths["run_dir"])
    add("no_raw_frame_persisted", not forbidden_run_files, f"forbidden_run_files={forbidden_run_files}")
    latest_forbidden = [] if latest_info is None else latest_info.get("forbidden_files", [])
    add("latest_bundle_forbidden_file_scan_passed", not latest_forbidden, f"latest_forbidden_files={latest_forbidden}")
    return pd.DataFrame(checks), {"protected_unchanged": unchanged, "latest_forbidden_files": latest_forbidden, "forbidden_run_files": forbidden_run_files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--source-run-id", default=SOURCE_OCR_RUN_ID)
    parser.add_argument("--timestamp", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"development_ad_context_ocr_review_v2_5_{timestamp}"
    run_dir = project_root / "workspaces/ocr_quality_review_v2_5_development/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = build_output_paths(run_dir)
    paths["run_dir"] = run_dir
    paths["script"] = project_root / SCRIPT_RELATIVE_PATH
    logger = TaskLogger(paths["run_log"])
    t0 = time.time()

    logger.log("[STEP 01] Safety snapshot and output directory setup")
    source_run_dir = project_root / "workspaces/ocr_final_scene_anchor_v2_5_development/runs" / args.source_run_id
    input_paths = {
        "ocr_frame_result": locate_file(
            source_run_dir / "final_scene_anchor_ocr_frame_results_v2_5_development.csv",
            [
                project_root / "workspaces/ocr_final_scene_anchor_v2_5_development/runs",
                project_root / "outputs/latest_ocr",
            ],
            "final_scene_anchor_ocr_frame_results_v2_5_development.csv",
        ),
        "ocr_video_summary": locate_file(
            source_run_dir / "final_scene_anchor_ocr_video_summary_v2_5_development.csv",
            [
                project_root / "workspaces/ocr_final_scene_anchor_v2_5_development/runs",
                project_root / "outputs/latest_ocr",
            ],
            "final_scene_anchor_ocr_video_summary_v2_5_development.csv",
        ),
        "ocr_report": locate_file(
            source_run_dir / "final_scene_anchor_ocr_v2_5_development_report.json",
            [
                project_root / "workspaces/ocr_final_scene_anchor_v2_5_development/runs",
                project_root / "outputs/latest_ocr",
            ],
            "final_scene_anchor_ocr_v2_5_development_report.json",
        ),
        "label_file": project_root / "data/segments/ad_interval_segments_v2_4.csv",
        "split_file_v2_4": project_root / "data/splits/video_split_v2_4.csv",
        "split_v25_file": project_root / "data/splits/video_split_v2_5_ruledev_extended_eval.csv",
        "manifest_file": project_root / "data/video_metadata/video_manifest_v2_2.csv",
    }
    protected_targets = {
        "source_ocr_frame_result": input_paths["ocr_frame_result"],
        "source_ocr_video_summary": input_paths["ocr_video_summary"],
        "source_ocr_report": input_paths["ocr_report"],
        "label_file": input_paths["label_file"],
        "split_file_v2_4": input_paths["split_file_v2_4"],
        "split_v25_file": input_paths["split_v25_file"],
        "manifest_file": input_paths["manifest_file"],
        "scene_data": project_root / "data/scene",
        "detector_scripts": project_root / "scripts/detectors",
        "detector_outputs": project_root / "data/predictions",
        "raw_videos": project_root / "data/raw/videos",
        "old_project": OLD_PROJECT_ROOT,
    }
    protected_before = {name: snapshot_path(path) for name, path in protected_targets.items()}

    logger.log("[STEP 02] Locate Development Set OCR frame result")
    logger.log(f"[STEP 02] OCR frame result: {input_paths['ocr_frame_result']}")

    logger.log("[STEP 03] Load OCR frame result and source OCR report")
    ocr_df, label_df, split_v25_df, manifest_df, source_ocr_report = load_inputs(input_paths)
    source_ocr_backend_summary = ""
    if "ocr_backend_status" in source_ocr_report:
        source_ocr_backend_summary = json.dumps(source_ocr_report.get("ocr_backend_status"), ensure_ascii=False)
    elif "ocr_backend" in ocr_df.columns:
        backends = sorted(ocr_df["ocr_backend"].dropna().astype(str).unique().tolist())
        versions = sorted(ocr_df.get("ocr_engine_version", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        source_ocr_backend_summary = f"backend={backends}; version={versions}"

    logger.log("[STEP 04] Load split file, manifest, and actual ad intervals")
    logger.log("[STEP 05] Apply v2.5 split terminology mapping")
    intervals = build_development_intervals(label_df, split_v25_df, manifest_df)

    logger.log("[STEP 06] Build Development Set ad_start_minus5_to_ad_end review windows")
    logger.log("[STEP 07] Filter OCR rows inside review windows")
    review = filter_review_rows(ocr_df, intervals, args.source_run_id)
    review, threshold_metadata = add_review_flags(review)

    logger.log("[STEP 08] Build full frame-level human review sheet")
    full_sheet = build_full_sheet(review)

    logger.log("[STEP 09] Build compact human-readable review sheet")
    compact_sheet = build_compact_sheet(review)

    logger.log("[STEP 10] Build keyword hit and disclosure-focused review sheets")
    keyword_review = build_keyword_review(review)
    disclosure_review = build_disclosure_review(review)

    logger.log("[STEP 11] Build empty/low-confidence review sheet")
    empty_lowconf_review = build_empty_lowconf_review(review, threshold_metadata["low_confidence_threshold"])

    logger.log("[STEP 12] Build interval-level and video-level quality summaries")
    interval_summary = build_interval_summary(review, intervals)
    interval_summary["empty_low_confidence_count_for_report"] = interval_summary["success_empty_count"] + interval_summary["failed_count"] + interval_summary["low_confidence_count"]
    video_summary = build_video_summary(interval_summary, review)

    logger.log("[STEP 13] Select representative examples and priority review rows")
    representative_examples = select_representative_examples(review)

    full_sheet.to_csv(paths["full_review_sheet"], index=False, encoding="utf-8-sig")
    compact_sheet.to_csv(paths["compact_review_sheet"], index=False, encoding="utf-8-sig")
    interval_summary.to_csv(paths["interval_summary"], index=False, encoding="utf-8-sig")
    video_summary.to_csv(paths["video_summary"], index=False, encoding="utf-8-sig")
    keyword_review.to_csv(paths["keyword_hit_review"], index=False, encoding="utf-8-sig")
    disclosure_review.to_csv(paths["disclosure_review"], index=False, encoding="utf-8-sig")
    empty_lowconf_review.to_csv(paths["empty_lowconf_review"], index=False, encoding="utf-8-sig")
    representative_examples.to_csv(paths["representative_examples"], index=False, encoding="utf-8-sig")

    logger.log("[STEP 14] Run Sub Agent validations")
    quality_checks, safety_results = make_quality_checks(
        input_paths,
        review,
        intervals,
        paths,
        protected_before,
        protected_targets,
        latest_info=None,
    )
    quality_checks.to_csv(paths["quality_checks"], index=False, encoding="utf-8-sig")

    total_rows = int(len(review))
    status_counts = review["ocr_status"].astype(str).value_counts().to_dict() if not review.empty else {}
    nonempty_ratio = float(review["has_text"].mean()) if total_rows else np.nan
    frame_review_summary = {
        "total_review_row_count": total_rows,
        "success_text_count": int(status_counts.get("success_text", 0)),
        "success_empty_count": int(status_counts.get("success_empty", 0)),
        "failed_count": int(status_counts.get("failed", 0)),
        "nonempty_ratio": safe_float(nonempty_ratio),
        "average_confidence": safe_float(pd.to_numeric(review.get("ocr_mean_confidence", pd.Series(dtype=float)), errors="coerce").mean()) if total_rows else np.nan,
        "low_confidence_row_count": int(review.get("is_low_confidence", pd.Series(dtype=bool)).fillna(False).sum()) if total_rows else 0,
        "empty_row_count": int(review["ocr_status"].astype(str).eq("success_empty").sum()) if total_rows else 0,
        "keyword_hit_row_count": int(review.get("has_corrected_keyword_hit", pd.Series(dtype=bool)).fillna(False).sum()) if total_rows else 0,
    }
    report: Dict[str, Any] = {
        "run_id": run_id,
        "generated_at": now_iso(),
        "project_root": str(project_root),
        "script_path": str(paths["script"]),
        "source_ocr_run_id": args.source_run_id,
        "source_ocr_frame_result": str(input_paths["ocr_frame_result"]),
        "source_ocr_backend_summary": source_ocr_backend_summary,
        "target_split_role_v2_5": "development",
        "target_original_split_v2_4": "train",
        "target_video_ids": DEVELOPMENT_IDS,
        "input_files": {key: str(value) for key, value in input_paths.items()},
        "review_window_definition": {
            "window_name": "ad_start_minus5_to_ad_end",
            "review_window_start_sec": "max(0, ad_start_sec - 5)",
            "review_window_end_sec": "ad_end_sec",
            "post_ad_window_included": False,
        },
        "review_scope_summary": {
            "video_count": int(intervals["video_id"].nunique()),
            "ad_interval_count": int(len(intervals)),
            "development_set_only": True,
            "extended_evaluation_processed": False,
            "diagnostic_subset_processed": False,
            "pure_test_processed": False,
        },
        "frame_review_summary": frame_review_summary,
        "threshold_metadata": threshold_metadata,
        "interval_quality_summary": {
            "interval_count": int(len(interval_summary)),
            "high_priority_count": int(interval_summary["suggested_review_priority"].eq("high").sum()),
            "medium_priority_count": int(interval_summary["suggested_review_priority"].eq("medium").sum()),
            "low_priority_count": int(interval_summary["suggested_review_priority"].eq("low").sum()),
        },
        "video_quality_summary": {
            "video_count": int(len(video_summary)),
            "total_review_rows": int(video_summary["review_row_count"].sum()) if not video_summary.empty else 0,
        },
        "keyword_hit_summary": {"row_count": int(len(keyword_review))},
        "disclosure_review_summary": {"row_count": int(len(disclosure_review))},
        "empty_lowconf_summary": {"row_count": int(len(empty_lowconf_review))},
        "representative_examples_summary": {"row_count": int(len(representative_examples))},
        "validation_results": {},
        "safety_results": safety_results,
        "outputs": {key: str(value) for key, value in paths.items() if key != "run_dir"},
        "warnings": [],
        "errors": [],
        "safety_flags": {
            "ocr_rerun_performed": False,
            "ocr_engine_called": False,
            "actual_label_used_for_sampling": False,
            "actual_label_used_for_review_filtering": True,
            "detector_modified": False,
            "existing_ocr_modified": False,
            "extended_evaluation_processed": False,
            "pure_test_processed": False,
            "raw_frame_persisted": False,
        },
    }
    if total_rows == 0:
        report["warnings"].append("No OCR rows were found inside Development Set review windows.")
    zero_row_intervals = interval_summary[interval_summary["review_row_count"].eq(0)]["ad_interval_id"].astype(str).tolist()
    if zero_row_intervals:
        report["warnings"].append(f"Some intervals had no OCR rows in the review window: {zero_row_intervals}")

    logger.log("[STEP 15] Generate markdown/json reports")
    write_markdown_report(paths["markdown_report"], report, interval_summary, representative_examples)
    save_json(paths["json_report"], report)

    logger.log("[STEP 16] Update latest bundles")
    latest_info = copy_latest_files(paths, project_root, run_id)
    quality_checks, safety_results = make_quality_checks(
        input_paths,
        review,
        intervals,
        paths,
        protected_before,
        protected_targets,
        latest_info=latest_info,
    )
    quality_checks.to_csv(paths["quality_checks"], index=False, encoding="utf-8-sig")
    report["validation_results"] = {
        "quality_check_status_counts": quality_checks["status"].value_counts().to_dict(),
        "quality_checks_path": str(paths["quality_checks"]),
    }
    report["safety_results"] = safety_results
    report["latest_bundles"] = latest_info
    report["elapsed_sec"] = safe_float(time.time() - t0)
    save_json(paths["json_report"], report)
    write_markdown_report(paths["markdown_report"], report, interval_summary, representative_examples)
    # JSON 갱신 뒤 final report/markdown/check를 latest bundle에 동기화한다.
    for dest in [
        Path(latest_info["latest_for_chatgpt_dir"]),
        Path(latest_info["latest_ocr_dir"]),
    ]:
        for key in ["json_report", "markdown_report", "quality_checks"]:
            dst = dest / paths[key].name
            shutil.copy2(paths[key], dst if not dst.exists() else dst)

    logger.log("[STEP 17] Print final human-readable summary")
    logger.log(
        "[STEP 17] "
        f"status=success, run_id={run_id}, rows={total_rows}, "
        f"keyword_hits={len(keyword_review)}, disclosure_rows={len(disclosure_review)}, "
        f"empty_lowconf_rows={len(empty_lowconf_review)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
