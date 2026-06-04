#!/usr/bin/env python3
"""검토가 끝난 v2.4 OpenCV/FFmpeg 장면 후보 workbook을 분석한다."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")

INPUT_XLSX = PROJECT_ROOT / "data/review/scene_candidate_human_review_v2_4.xlsx"
OPTIONAL_CANDIDATE_CSV = PROJECT_ROOT / "data/review/scene_candidate_human_review_candidate_sheet_v2_4.csv"
OPTIONAL_BOUNDARY_CSV = PROJECT_ROOT / "data/review/scene_candidate_human_review_boundary_sheet_v2_4.csv"
OPTIONAL_PREV_REPORT = PROJECT_ROOT / "reports/scene_candidate_human_review_v2_4_report.json"
OPTIONAL_PREV_SUMMARY = PROJECT_ROOT / "reports/scene_candidate_human_review_v2_4_summary.md"
OPTIONAL_PREV_LOG = PROJECT_ROOT / "logs/scene_candidate_human_review_v2_4_run_log.txt"

REPORT_PATH = PROJECT_ROOT / "reports/scene_candidate_human_review_v2_4_analysis_report.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/scene_candidate_human_review_v2_4_analysis_summary.md"
LOG_PATH = PROJECT_ROOT / "logs/scene_candidate_human_review_v2_4_analysis_run_log.txt"
REVIEWED_ROWS_CSV = PROJECT_ROOT / "data/review/scene_candidate_human_review_v2_4_reviewed_rows.csv"
FALSE_POSITIVE_CSV = PROJECT_ROOT / "data/review/scene_candidate_human_review_v2_4_false_positive_summary.csv"
THRESHOLD_CSV = PROJECT_ROOT / "data/review/scene_candidate_human_review_v2_4_threshold_analysis.csv"
SCRIPT_PATH = PROJECT_ROOT / "scripts/review/analyze_scene_candidate_human_review_v2_4.py"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"
LATEST_README = LATEST_DIR / "README_latest_files.md"

RUN_LOG: list[str] = []
REQUIRED_SHEETS = ["scene_candidate_review", "ad_boundary_review"]
REQUIRED_REVIEW_COLUMNS = [
    "review_priority",
    "review_status",
    "is_true_scene_change",
    "scene_change_strength",
    "scene_change_type",
    "is_ad_boundary_related",
    "false_positive_type",
    "keep_as_boundary_candidate",
    "review_note",
    "reviewer",
    "reviewed_at",
]
MEANINGFUL_REVIEW_COLUMNS = [
    "is_true_scene_change",
    "is_ad_boundary_related",
    "keep_as_boundary_candidate",
    "review_note",
]
REQUIRED_BOUNDARY_COLUMNS = [
    "ad_interval_id",
    "video_id",
    "ad_start_sec",
    "ad_end_sec",
    "start_hit_3s",
    "start_hit_5s",
    "end_hit_3s",
    "end_hit_5s",
    "nearest_candidate_to_start_sec",
    "distance_to_start_candidate_sec",
    "nearest_candidate_to_end_sec",
    "distance_to_end_candidate_sec",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{now_iso()}] {message}"
    RUN_LOG.append(line)
    print(message, flush=True)


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def to_float(value: Any) -> float | None:
    text = clean(value)
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def to_bool(value: Any) -> bool:
    text = clean(value).lower()
    return text in {"true", "yes", "y", "1", "t"}


def rate(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return round(num / den, 6)


def readable_runtime(seconds: float) -> str:
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    return f"{minutes}분 {sec}초"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def old_project_snapshot() -> dict[str, Any]:
    if not OLD_PROJECT_ROOT.exists():
        return {"exists": False, "file_count": 0, "digest": ""}
    h = hashlib.sha256()
    file_count = 0
    for path in sorted(p for p in OLD_PROJECT_ROOT.rglob("*") if p.is_file()):
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(OLD_PROJECT_ROOT).as_posix()
        h.update(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode("utf-8", errors="replace"))
        file_count += 1
    return {"exists": True, "file_count": file_count, "digest": h.hexdigest()}


def norm_token(value: Any) -> str:
    text = clean(value).lower().replace("-", "_").replace(" ", "_")
    if text in {"", "nan", "none", "null", "not_reviewed"}:
        return "missing"
    if text in {"yes", "true", "y", "1", "t"}:
        return "true"
    if text in {"no", "false", "n", "0", "f"}:
        return "false"
    if text in {"uncertain", "unclear", "partial", "ambiguous", "maybe", "unknown"}:
        return "uncertain"
    return text


def scene_class(value: Any, reviewed: bool) -> str:
    token = norm_token(value)
    if token == "true":
        return "true"
    if token == "false":
        return "false"
    if token in {"uncertain", "missing"}:
        return "uncertain" if reviewed else "missing"
    return "uncertain"


def keep_class(value: Any, reviewed: bool) -> str:
    token = norm_token(value)
    if token == "true":
        return "true"
    if token == "false":
        return "false"
    if token in {"uncertain", "missing"}:
        return "uncertain" if reviewed else "missing"
    return "uncertain"


def ad_boundary_class(value: Any, reviewed: bool) -> str:
    token = norm_token(value)
    if token in {"ad_start", "ad_end", "true"}:
        return "true"
    if token in {"no", "false"}:
        return "false"
    if token in {"inside_ad", "near_ad_but_not_boundary", "uncertain", "missing"}:
        return "uncertain" if reviewed else "missing"
    return "uncertain"


def is_meaningful_review_value(value: Any) -> bool:
    token = norm_token(value)
    return token not in {"missing", "not_reviewed"}


def normalize_candidate_reviews(scene_df: pd.DataFrame) -> pd.DataFrame:
    df = scene_df.copy()
    for col in REQUIRED_REVIEW_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    review_status_norm = df["review_status"].map(lambda x: norm_token(x))
    has_meaningful = pd.Series(False, index=df.index)
    for col in MEANINGFUL_REVIEW_COLUMNS:
        has_meaningful = has_meaningful | df[col].map(is_meaningful_review_value)
    df["reviewed_effective"] = (review_status_norm == "reviewed") | has_meaningful
    df["scene_truth_class"] = [
        scene_class(value, bool(reviewed)) for value, reviewed in zip(df["is_true_scene_change"], df["reviewed_effective"])
    ]
    df["ad_boundary_related_class"] = [
        ad_boundary_class(value, bool(reviewed)) for value, reviewed in zip(df["is_ad_boundary_related"], df["reviewed_effective"])
    ]
    df["keep_boundary_class"] = [
        keep_class(value, bool(reviewed)) for value, reviewed in zip(df["keep_as_boundary_candidate"], df["reviewed_effective"])
    ]
    df["scene_change_score_num"] = pd.to_numeric(df.get("scene_change_score"), errors="coerce")
    df["score_percentile_num"] = pd.to_numeric(df.get("score_percentile_in_video"), errors="coerce")
    df["distance_to_nearest_ad_boundary_num"] = pd.to_numeric(df.get("distance_to_nearest_ad_boundary_sec"), errors="coerce")
    return df


def describe_series(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.count()),
        "min": round(float(s.min()), 6),
        "p25": round(float(s.quantile(0.25)), 6),
        "median": round(float(s.median()), 6),
        "mean": round(float(s.mean()), 6),
        "p75": round(float(s.quantile(0.75)), 6),
        "max": round(float(s.max()), 6),
    }


def count_dict(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.fillna("missing").map(lambda x: clean(x) or "missing").value_counts().sort_index().items()}


def class_counts(df: pd.DataFrame, col: str) -> dict[str, int]:
    counts = df[col].value_counts().to_dict()
    return {
        "true": int(counts.get("true", 0)),
        "false": int(counts.get("false", 0)),
        "uncertain": int(counts.get("uncertain", 0)),
    }


def rate_by_group(df: pd.DataFrame, group_col: str, class_col: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if group_col not in df.columns:
        return out
    for group_value, group in df.groupby(group_col, dropna=False):
        label = clean(group_value) or "missing"
        counts = class_counts(group, class_col)
        denom = counts["true"] + counts["false"] + counts["uncertain"]
        out[label] = {**counts, "reviewed_count": int(denom), "true_rate": rate(counts["true"], denom)}
    return out


def score_band_labels(reviewed_df: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    scores = reviewed_df["scene_change_score_num"].dropna()
    if scores.empty:
        return pd.Series(["missing"] * len(reviewed_df), index=reviewed_df.index), {}
    p25 = float(scores.quantile(0.25))
    p50 = float(scores.quantile(0.50))
    p75 = float(scores.quantile(0.75))

    def label(value: Any) -> str:
        if pd.isna(value):
            return "missing"
        value = float(value)
        if value <= p25:
            return "score <= p25"
        if value <= p50:
            return "p25 < score <= p50"
        if value <= p75:
            return "p50 < score <= p75"
        return "score > p75"

    return reviewed_df["scene_change_score_num"].map(label), {"p25": p25, "p50": p50, "p75": p75}


def score_percentile_band(value: Any) -> str:
    num = to_float(value)
    if num is None:
        return "missing"
    if num <= 50:
        return "<=50"
    if num <= 75:
        return "50-75"
    if num <= 90:
        return "75-90"
    return ">90"


def distance_band(value: Any) -> str:
    num = to_float(value)
    if num is None:
        return "missing"
    if num <= 2:
        return "<=2s_good"
    if num <= 5:
        return "3-5s_acceptable"
    return ">5s_missed"


def build_false_positive_summary(reviewed_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fp_df = reviewed_df.copy()
    fp_df["false_positive_type_clean"] = fp_df.get("false_positive_type", "").map(lambda x: clean(x) or "missing")
    fp_df.loc[fp_df["false_positive_type_clean"].isin(["nan", "None", ""]), "false_positive_type_clean"] = "missing"
    for fp_type, count in fp_df["false_positive_type_clean"].value_counts().sort_index().items():
        rows.append(
            {
                "group_type": "overall",
                "group_value": "all_reviewed",
                "false_positive_type": fp_type,
                "count": int(count),
                "reviewed_count": int(len(reviewed_df)),
                "rate": rate(int(count), int(len(reviewed_df))),
            }
        )
    for group_col, group_type in [("candidate_source", "candidate_source"), ("score_band", "score_band")]:
        if group_col not in fp_df.columns:
            continue
        for group_value, group in fp_df.groupby(group_col, dropna=False):
            group_label = clean(group_value) or "missing"
            denom = len(group)
            for fp_type, count in group["false_positive_type_clean"].value_counts().sort_index().items():
                rows.append(
                    {
                        "group_type": group_type,
                        "group_value": group_label,
                        "false_positive_type": fp_type,
                        "count": int(count),
                        "reviewed_count": int(denom),
                        "rate": rate(int(count), int(denom)),
                    }
                )
    return rows


def build_threshold_rows(reviewed_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_specs = [("overall_score_band", ["score_band"]), ("candidate_source", ["candidate_source"]), ("candidate_source_score_band", ["candidate_source", "score_band"]), ("score_percentile_band", ["score_percentile_band"])]
    for group_type, group_cols in group_specs:
        if any(col not in reviewed_df.columns for col in group_cols):
            continue
        for key, group in reviewed_df.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            group_value = " | ".join(clean(v) or "missing" for v in key)
            scene_counts = class_counts(group, "scene_truth_class")
            keep_counts = class_counts(group, "keep_boundary_class")
            denom = scene_counts["true"] + scene_counts["false"] + scene_counts["uncertain"]
            rows.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "reviewed_count": int(denom),
                    "true_scene_change_count": scene_counts["true"],
                    "false_scene_change_count": scene_counts["false"],
                    "uncertain_scene_change_count": scene_counts["uncertain"],
                    "true_scene_change_rate": rate(scene_counts["true"], denom),
                    "keep_true_count": keep_counts["true"],
                    "keep_false_count": keep_counts["false"],
                    "keep_uncertain_count": keep_counts["uncertain"],
                    "keep_rate": rate(keep_counts["true"], denom),
                    "score_mean": describe_series(group["scene_change_score_num"]).get("mean", ""),
                    "score_median": describe_series(group["scene_change_score_num"]).get("median", ""),
                    "score_min": describe_series(group["scene_change_score_num"]).get("min", ""),
                    "score_max": describe_series(group["scene_change_score_num"]).get("max", ""),
                }
            )
    return rows


def review_note_examples(reviewed_df: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for _, row in reviewed_df.iterrows():
        note = clean(row.get("review_note"))
        fp_type = clean(row.get("false_positive_type"))
        if not note:
            continue
        if row.get("scene_truth_class") != "false" and fp_type in {"", "not_false_positive"}:
            continue
        examples.append(
            {
                "video_id": clean(row.get("video_id")),
                "candidate_time_mmss": clean(row.get("candidate_time_mmss")),
                "review_priority": clean(row.get("review_priority")),
                "false_positive_type": fp_type,
                "scene_truth_class": clean(row.get("scene_truth_class")),
                "scene_change_score": to_float(row.get("scene_change_score")),
                "review_note": note[:240],
            }
        )
        if len(examples) >= limit:
            break
    return examples


def analyze_candidates(scene_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = normalize_candidate_reviews(scene_df)
    reviewed = normalized[normalized["reviewed_effective"]].copy()
    reviewed["score_band"], quartiles = score_band_labels(reviewed)
    normalized.loc[reviewed.index, "score_band"] = reviewed["score_band"]
    reviewed["score_percentile_band"] = reviewed["score_percentile_num"].map(score_percentile_band)
    normalized.loc[reviewed.index, "score_percentile_band"] = reviewed["score_percentile_band"]
    reviewed["distance_band"] = reviewed["distance_to_nearest_ad_boundary_num"].map(distance_band)
    normalized.loc[reviewed.index, "distance_band"] = reviewed["distance_band"]

    priority_counts = count_dict(normalized.get("review_priority", pd.Series(dtype=str)))
    reviewed_by_priority = count_dict(reviewed.get("review_priority", pd.Series(dtype=str)))
    scene_counts = class_counts(reviewed, "scene_truth_class")
    ad_counts = class_counts(reviewed, "ad_boundary_related_class")
    keep_counts = class_counts(reviewed, "keep_boundary_class")

    source_score_dist = {
        clean(source) or "missing": describe_series(group["scene_change_score_num"])
        for source, group in reviewed.groupby("candidate_source", dropna=False)
    } if "candidate_source" in reviewed.columns else {}
    metrics = {
        "candidate_total_count": int(len(normalized)),
        "reviewed_candidate_count": int(len(reviewed)),
        "reviewed_rate": rate(len(reviewed), len(normalized)),
        "high_priority_count": int(priority_counts.get("high", 0)),
        "high_priority_reviewed_count": int(reviewed_by_priority.get("high", 0)),
        "medium_priority_count": int(priority_counts.get("medium", 0)),
        "medium_priority_reviewed_count": int(reviewed_by_priority.get("medium", 0)),
        "low_priority_count": int(priority_counts.get("low", 0)),
        "low_priority_reviewed_count": int(reviewed_by_priority.get("low", 0)),
        "reviewed_by_priority": reviewed_by_priority,
        "priority_counts": priority_counts,
        "reviewed_true_scene_change_count": scene_counts["true"],
        "reviewed_false_scene_change_count": scene_counts["false"],
        "reviewed_uncertain_scene_change_count": scene_counts["uncertain"],
        "true_scene_change_rate_among_reviewed": rate(scene_counts["true"], len(reviewed)),
        "true_scene_change_rate_by_priority": rate_by_group(reviewed, "review_priority", "scene_truth_class"),
        "true_scene_change_rate_by_candidate_source": rate_by_group(reviewed, "candidate_source", "scene_truth_class"),
        "true_scene_change_rate_by_method_used": rate_by_group(reviewed, "method_used", "scene_truth_class"),
        "true_scene_change_rate_by_score_band": rate_by_group(reviewed, "score_band", "scene_truth_class"),
        "ad_boundary_related_true_count": ad_counts["true"],
        "ad_boundary_related_false_count": ad_counts["false"],
        "ad_boundary_related_uncertain_count": ad_counts["uncertain"],
        "ad_boundary_related_rate_among_reviewed": rate(ad_counts["true"], len(reviewed)),
        "is_ad_boundary_related_raw_counts": count_dict(reviewed.get("is_ad_boundary_related", pd.Series(dtype=str))),
        "keep_as_boundary_candidate_true_count": keep_counts["true"],
        "keep_as_boundary_candidate_false_count": keep_counts["false"],
        "keep_as_boundary_candidate_uncertain_count": keep_counts["uncertain"],
        "keep_rate_among_reviewed": rate(keep_counts["true"], len(reviewed)),
        "keep_rate_by_priority": rate_by_group(reviewed, "review_priority", "keep_boundary_class"),
        "keep_rate_by_distance_band": rate_by_group(reviewed, "distance_band", "keep_boundary_class"),
        "false_positive_type_counts": count_dict(reviewed.get("false_positive_type", pd.Series(dtype=str))),
        "false_positive_type_by_candidate_source": {
            clean(source) or "missing": count_dict(group.get("false_positive_type", pd.Series(dtype=str)))
            for source, group in reviewed.groupby("candidate_source", dropna=False)
        } if "candidate_source" in reviewed.columns else {},
        "false_positive_type_by_score_band": {
            clean(band) or "missing": count_dict(group.get("false_positive_type", pd.Series(dtype=str)))
            for band, group in reviewed.groupby("score_band", dropna=False)
        } if "score_band" in reviewed.columns else {},
        "review_note_examples": review_note_examples(reviewed, limit=10),
        "score_quartiles_for_bands": quartiles,
        "reviewed_score_distribution": describe_series(reviewed["scene_change_score_num"]),
        "true_scene_change_score_distribution": describe_series(reviewed[reviewed["scene_truth_class"] == "true"]["scene_change_score_num"]),
        "false_scene_change_score_distribution": describe_series(reviewed[reviewed["scene_truth_class"] == "false"]["scene_change_score_num"]),
        "candidate_source_score_distribution": source_score_dist,
        "score_percentile_true_scene_change_rate": rate_by_group(reviewed, "score_percentile_band", "scene_truth_class"),
    }
    fp_rows = build_false_positive_summary(reviewed)
    threshold_rows = build_threshold_rows(reviewed)
    return normalized, metrics, fp_rows, threshold_rows


def analyze_boundaries(boundary_df: pd.DataFrame) -> dict[str, Any]:
    df = boundary_df.copy()
    for col in REQUIRED_BOUNDARY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    ad_interval_count = int(len(df))
    total_boundaries = ad_interval_count * 2
    start_hit_3 = int(df["start_hit_3s"].map(to_bool).sum())
    start_hit_5 = int(df["start_hit_5s"].map(to_bool).sum())
    end_hit_3 = int(df["end_hit_3s"].map(to_bool).sum())
    end_hit_5 = int(df["end_hit_5s"].map(to_bool).sum())
    both_hit_5 = int(df["both_boundary_hit_5s"].map(to_bool).sum()) if "both_boundary_hit_5s" in df.columns else 0
    start_dist = pd.to_numeric(df["distance_to_start_candidate_sec"], errors="coerce")
    end_dist = pd.to_numeric(df["distance_to_end_candidate_sec"], errors="coerce")
    combined_dist = pd.concat([start_dist, end_dist], ignore_index=True)
    combined_band_counts = count_dict(combined_dist.map(distance_band))
    start_band_counts = count_dict(start_dist.map(distance_band))
    end_band_counts = count_dict(end_dist.map(distance_band))
    added = df[
        (df["video_id"].map(lambda x: clean(x) == "12"))
        & (pd.to_numeric(df["ad_start_sec"], errors="coerce") == 1160)
        & (pd.to_numeric(df["ad_end_sec"], errors="coerce") == 1206)
    ]
    added_found = not added.empty
    added_record: dict[str, Any] = {"found": False}
    if added_found:
        row = added.iloc[0]
        added_record = {
            "found": True,
            "ad_interval_id": clean(row.get("ad_interval_id")),
            "video_id": clean(row.get("video_id")),
            "ad_start_sec": to_float(row.get("ad_start_sec")),
            "ad_end_sec": to_float(row.get("ad_end_sec")),
            "ad_start_mmss": clean(row.get("ad_start_mmss")),
            "ad_end_mmss": clean(row.get("ad_end_mmss")),
            "start_hit_3s": to_bool(row.get("start_hit_3s")),
            "start_hit_5s": to_bool(row.get("start_hit_5s")),
            "end_hit_3s": to_bool(row.get("end_hit_3s")),
            "end_hit_5s": to_bool(row.get("end_hit_5s")),
            "nearest_candidate_to_start_sec": to_float(row.get("nearest_candidate_to_start_sec")),
            "distance_to_start_candidate_sec": to_float(row.get("distance_to_start_candidate_sec")),
            "nearest_candidate_to_end_sec": to_float(row.get("nearest_candidate_to_end_sec")),
            "distance_to_end_candidate_sec": to_float(row.get("distance_to_end_candidate_sec")),
            "start_distance_band": distance_band(row.get("distance_to_start_candidate_sec")),
            "end_distance_band": distance_band(row.get("distance_to_end_candidate_sec")),
        }
    return {
        "ad_interval_count": ad_interval_count,
        "expected_ad_interval_count": 22,
        "ad_interval_count_matches_expected": ad_interval_count == 22,
        "total_boundaries": total_boundaries,
        "expected_total_boundaries": 44,
        "total_boundaries_matches_expected": total_boundaries == 44,
        "start_hit_3s_count": start_hit_3,
        "start_hit_5s_count": start_hit_5,
        "end_hit_3s_count": end_hit_3,
        "end_hit_5s_count": end_hit_5,
        "boundary_hit_3s_count": start_hit_3 + end_hit_3,
        "boundary_hit_5s_count": start_hit_5 + end_hit_5,
        "boundary_hit_5s_rate": rate(start_hit_5 + end_hit_5, total_boundaries),
        "both_boundary_hit_5s_count": both_hit_5,
        "distance_to_start_candidate_sec_distribution": describe_series(start_dist),
        "distance_to_end_candidate_sec_distribution": describe_series(end_dist),
        "distance_to_any_boundary_sec_distribution": describe_series(combined_dist),
        "distance_band_counts_combined": combined_band_counts,
        "distance_band_counts_start": start_band_counts,
        "distance_band_counts_end": end_band_counts,
        "added_video_12_interval": added_record,
        "added_video_12_interval_found": added_found,
    }


def choose_threshold_recommendation(metrics: dict[str, Any], threshold_rows: list[dict[str, Any]]) -> dict[str, str]:
    fp_counts = metrics.get("false_positive_type_counts", {})
    reviewed_count = metrics.get("reviewed_candidate_count", 0) or 0
    type_fp_keys = {"camera_motion", "subtitle_change", "text_overlay_only", "lighting_change", "object_motion", "motion_only"}
    type_fp_count = sum(int(fp_counts.get(key, 0)) for key in type_fp_keys)
    type_fp_share = rate(type_fp_count, reviewed_count) or 0
    lower_rows = [row for row in threshold_rows if row["group_type"] == "overall_score_band" and row["group_value"] in {"score <= p25", "p25 < score <= p50"}]
    lower_false = sum(int(row["false_scene_change_count"]) for row in lower_rows)
    lower_true = sum(int(row["true_scene_change_count"]) for row in lower_rows)
    true_rate = metrics.get("true_scene_change_rate_among_reviewed") or 0
    high_stats = metrics.get("true_scene_change_rate_by_priority", {}).get("high", {})
    high_true_rate = high_stats.get("true_rate") or 0
    if type_fp_share >= 0.35:
        rec = "postprocess_first"
        title = "threshold보다 후처리 우선"
        reason = "false positive가 점수 자체보다 camera/subtitle/text/lighting/motion 유형에 더 강하게 의존한다."
    elif lower_false > lower_true:
        rec = "raise_slightly"
        title = "threshold 소폭 상향 또는 score percentile guard 추가"
        reason = "median 이하 score band에서 false scene-change 비중이 커 source별 또는 video별 percentile guard가 필요하다."
    elif true_rate >= 0.70 and high_true_rate >= 0.70:
        rec = "maintain"
        title = "threshold 유지"
        reason = "reviewed candidate와 high priority 후보에서 true scene-change 비율이 충분히 높아 threshold 상향은 유효 후보 손실 위험이 있다."
    else:
        rec = "postprocess_first"
        title = "threshold보다 후처리 우선"
        reason = "현재 threshold 이상 후보의 precision 개선은 raw threshold보다 유형 기반 후처리와 다른 evidence 결합이 더 적절하다."
    return {"code": rec, "label": title, "reason": reason}


def rows_to_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reviewed_rows_csv(path: Path, normalized_df: pd.DataFrame) -> None:
    reviewed = normalized_df[normalized_df["reviewed_effective"]].copy()
    columns = [
        "video_id",
        "video_title",
        "candidate_time_sec",
        "candidate_time_mmss",
        "review_priority",
        "scene_change_score",
        "score_percentile_in_video",
        "candidate_source",
        "method_used",
        "nearest_ad_boundary_type",
        "distance_to_nearest_ad_boundary_sec",
        "is_near_any_ad_boundary_5s",
        "review_status",
        "is_true_scene_change",
        "scene_truth_class",
        "is_ad_boundary_related",
        "ad_boundary_related_class",
        "keep_as_boundary_candidate",
        "keep_boundary_class",
        "false_positive_type",
        "scene_change_type",
        "scene_change_strength",
        "score_band",
        "score_percentile_band",
        "distance_band",
        "review_note",
        "reviewer",
        "reviewed_at",
    ]
    for col in columns:
        if col not in reviewed.columns:
            reviewed[col] = ""
    reviewed[columns].to_csv(path, index=False, encoding="utf-8-sig")


def latest_forbidden_files() -> list[str]:
    forbidden: list[str] = []
    if not LATEST_DIR.exists():
        return forbidden
    for path in LATEST_DIR.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix == ".mp4":
            forbidden.append(str(path))
        if suffix in {".png", ".jpg", ".jpeg", ".webp"} and "frame" in name:
            forbidden.append(str(path))
        if any(token in name for token in ["checkpoint", "cache"]) or (("model" in name) and suffix not in {".md", ".json", ".txt"}):
            forbidden.append(str(path))
    return forbidden


def copy_latest() -> list[str]:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        REPORT_PATH,
        SUMMARY_PATH,
        LOG_PATH,
        REVIEWED_ROWS_CSV,
        FALSE_POSITIVE_CSV,
        THRESHOLD_CSV,
        SCRIPT_PATH,
    ]
    copied = []
    for path in files:
        if path.exists():
            target = LATEST_DIR / path.name
            shutil.copy2(path, target)
            copied.append(str(target))
    readme = """# latest_for_chatgpt files

이번 작업명: analyze_scene_candidate_human_review_v2_4

scene_candidate_human_review_v2_4.xlsx의 수동 review 결과를 분석한 report, summary, log, 분석용 CSV, 분석 스크립트만 복사했다.
원본 review xlsx, mp4, frame image, model, checkpoint, cache 파일은 새로 복사하지 않는다.

복사 파일:

- `scene_candidate_human_review_v2_4_analysis_report.json`
- `scene_candidate_human_review_v2_4_analysis_summary.md`
- `scene_candidate_human_review_v2_4_analysis_run_log.txt`
- `scene_candidate_human_review_v2_4_reviewed_rows.csv`
- `scene_candidate_human_review_v2_4_false_positive_summary.csv`
- `scene_candidate_human_review_v2_4_threshold_analysis.csv`
- `analyze_scene_candidate_human_review_v2_4.py`
- `README_latest_files.md`
"""
    LATEST_README.write_text(readme, encoding="utf-8")
    copied.append(str(LATEST_README))
    return copied


def build_summary(report: dict[str, Any]) -> str:
    cm = report["candidate_metrics"]
    bm = report["boundary_metrics"]
    rec = report["threshold_recommendation"]
    high_scene = cm.get("true_scene_change_rate_by_priority", {}).get("high", {})
    high_keep = cm.get("keep_rate_by_priority", {}).get("high", {})
    medium_scene = cm.get("true_scene_change_rate_by_priority", {}).get("medium", {})
    medium_keep = cm.get("keep_rate_by_priority", {}).get("medium", {})
    examples = cm.get("review_note_examples", [])
    example_lines = "\n".join(
        f"- video_id={ex['video_id']}, {ex['candidate_time_mmss']}, {ex['false_positive_type']}: {ex['review_note']}"
        for ex in examples
    ) or "- 대표 review_note 예시가 충분하지 않음"
    return f"""# Scene Candidate Human Review v2.4 Analysis

## 작업 개요

`scene_candidate_human_review_v2_4.xlsx`의 수동 입력 review를 분석했다. 이 문서는 final 광고 탐지 성능 평가가 아니라 scene-change 후보 리뷰 및 boundary evidence audit 결과이다.

## 입력 파일

- Reviewed workbook: `{INPUT_XLSX}`
- Candidate companion CSV: `{OPTIONAL_CANDIDATE_CSV}` ({'found' if OPTIONAL_CANDIDATE_CSV.exists() else 'missing'})
- Boundary companion CSV: `{OPTIONAL_BOUNDARY_CSV}` ({'found' if OPTIONAL_BOUNDARY_CSV.exists() else 'missing'})

## 리뷰 완료 현황

- candidate total: {cm['candidate_total_count']}
- reviewed candidate: {cm['reviewed_candidate_count']} ({cm['reviewed_rate']})
- high priority: {cm['high_priority_count']} / reviewed {cm['high_priority_reviewed_count']}
- medium priority: {cm['medium_priority_count']} / reviewed {cm['medium_priority_reviewed_count']}
- low priority: {cm['low_priority_count']} / reviewed {cm['low_priority_reviewed_count']}

## High Priority 리뷰 결과

- high reviewed count: {high_scene.get('reviewed_count', 0)}
- high true scene-change: {high_scene.get('true', 0)}
- high false scene-change: {high_scene.get('false', 0)}
- high uncertain/missing scene-change: {high_scene.get('uncertain', 0)}
- high true scene-change rate: {high_scene.get('true_rate')}
- high keep as boundary candidate rate: {high_keep.get('true_rate')}

## Medium/추가 Sample 리뷰 결과

- medium reviewed count: {medium_scene.get('reviewed_count', 0)}
- medium true scene-change: {medium_scene.get('true', 0)}
- medium false scene-change: {medium_scene.get('false', 0)}
- medium uncertain/missing scene-change: {medium_scene.get('uncertain', 0)}
- medium true scene-change rate: {medium_scene.get('true_rate')}
- medium keep as boundary candidate rate: {medium_keep.get('true_rate')}

v2_4에서 추가된 video_id=12 주변 candidate는 reviewed row 정규화 기준에 포함했다.

## False Positive 유형

주요 false positive count는 `false_positive_type_counts`와 `{FALSE_POSITIVE_CSV}`에 기록했다.

대표 review_note 예시:

{example_lines}

## Score/Threshold 분석

- reviewed score distribution: {cm['reviewed_score_distribution']}
- true score distribution: {cm['true_scene_change_score_distribution']}
- false score distribution: {cm['false_scene_change_score_distribution']}
- score band threshold rows: `{THRESHOLD_CSV}`

권장: **{rec['label']}** (`{rec['code']}`)

근거: {rec['reason']}

주의: scene_candidate_review는 이미 threshold를 통과한 후보만 포함한다. threshold 아래에서 탈락한 true scene change를 보지 않았으므로 recall 개선 또는 threshold 하향 주장은 제한한다. source별 score scale이 다를 수 있어 global raw score threshold 단정은 피해야 한다.

## Ad Boundary Distance 분석

- ad interval count: {bm['ad_interval_count']} (expected 22)
- total boundaries: {bm['total_boundaries']} (expected 44)
- boundary hit 5s count: {bm['boundary_hit_5s_count']}
- boundary hit 5s rate: {bm['boundary_hit_5s_rate']}
- distance band combined: {bm['distance_band_counts_combined']}

## Video_id=12 추가 광고 구간 분석

{bm['added_video_12_interval']}

## 최종 권장사항

Scene-change 후보는 광고 boundary recall 보조 evidence로 유용하지만, 단독으로 광고 boundary를 확정하면 일반 장면 전환 오탐이 증가한다.

따라서 scene-change는 OCR/audio/ad-probability 변화와 결합하고, boundary ±5초 및 source별 score percentile 조건을 함께 사용하는 것이 적절하다.

`keep_as_boundary_candidate`는 일반 장면 전환 유지가 아니라 광고 start/end boundary evidence로 유지한다는 의미로 해석해야 한다.

## 주의사항

이 결과는 final performance claim이 아니라 현재 threshold 이상 scene-change 후보의 precision/오탐 유형 분석이다.
"""


def write_log(report: dict[str, Any]) -> None:
    lines = [
        "작업명: analyze_scene_candidate_human_review_v2_4",
        f"start_time: {report.get('start_time')}",
        f"end_time: {report.get('end_time')}",
        f"actual_runtime_seconds: {report.get('actual_runtime_seconds')}",
        f"actual_runtime_readable: {report.get('actual_runtime_readable')}",
        "",
        *RUN_LOG,
        "",
        "sub_agent_results:",
        json.dumps(report.get("sub_agent_results", {}), ensure_ascii=False, indent=2),
        "",
        "errors:",
        json.dumps(report.get("errors", []), ensure_ascii=False, indent=2),
        "",
    ]
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")


def final_output(report: dict[str, Any]) -> dict[str, Any]:
    cm = report["candidate_metrics"]
    bm = report["boundary_metrics"]
    return {
        "input_xlsx": str(INPUT_XLSX),
        "candidate_total_count": cm["candidate_total_count"],
        "reviewed_candidate_count": cm["reviewed_candidate_count"],
        "high_priority_count": cm["high_priority_count"],
        "high_priority_reviewed_count": cm["high_priority_reviewed_count"],
        "medium_priority_reviewed_count": cm["medium_priority_reviewed_count"],
        "true_scene_change_rate_among_reviewed": cm["true_scene_change_rate_among_reviewed"],
        "ad_boundary_related_rate_among_reviewed": cm["ad_boundary_related_rate_among_reviewed"],
        "keep_as_boundary_candidate_rate_among_reviewed": cm["keep_rate_among_reviewed"],
        "ad_interval_count": bm["ad_interval_count"],
        "total_boundaries": bm["total_boundaries"],
        "boundary_hit_5s_count": bm["boundary_hit_5s_count"],
        "boundary_hit_5s_rate": bm["boundary_hit_5s_rate"],
        "added_video_12_interval_found": bm["added_video_12_interval_found"],
        "threshold_recommendation": report["threshold_recommendation"]["code"],
        "summary_path": str(SUMMARY_PATH),
        "report_path": str(REPORT_PATH),
        "log_path": str(LOG_PATH),
        "old_project_modified": report["old_project_modified"],
        "errors": report["errors"],
    }


def generate() -> int:
    start = now_iso()
    start_mono = time.monotonic()
    errors: list[Any] = []
    warnings: list[Any] = []
    log(f"작업 시작: {start}")
    old_before = old_project_snapshot()
    workbook_sha_before = file_sha256(INPUT_XLSX) if INPUT_XLSX.exists() else ""
    workbook_mtime_before = INPUT_XLSX.stat().st_mtime_ns if INPUT_XLSX.exists() else None
    log("old project snapshot 및 input workbook hash 기록 완료")

    missing_optional = [
        str(path)
        for path in [OPTIONAL_CANDIDATE_CSV, OPTIONAL_BOUNDARY_CSV, OPTIONAL_PREV_REPORT, OPTIONAL_PREV_SUMMARY, OPTIONAL_PREV_LOG]
        if not path.exists()
    ]
    if not INPUT_XLSX.exists():
        errors.append({"missing_input_xlsx": str(INPUT_XLSX)})
        raise FileNotFoundError(INPUT_XLSX)

    xl = pd.ExcelFile(INPUT_XLSX)
    sheet_names = xl.sheet_names
    missing_sheets = [sheet for sheet in REQUIRED_SHEETS if sheet not in sheet_names]
    if missing_sheets:
        errors.append({"missing_required_sheets": missing_sheets})
        raise ValueError(f"Missing sheets: {missing_sheets}")
    scene_df = pd.read_excel(INPUT_XLSX, sheet_name="scene_candidate_review")
    boundary_df = pd.read_excel(INPUT_XLSX, sheet_name="ad_boundary_review")
    missing_review_columns = [col for col in REQUIRED_REVIEW_COLUMNS if col not in scene_df.columns]
    missing_boundary_columns = [col for col in REQUIRED_BOUNDARY_COLUMNS if col not in boundary_df.columns]
    if missing_review_columns:
        errors.append({"missing_review_columns": missing_review_columns})
        raise ValueError(f"Missing review columns: {missing_review_columns}")
    if missing_boundary_columns:
        errors.append({"missing_boundary_columns": missing_boundary_columns})
        raise ValueError(f"Missing boundary columns: {missing_boundary_columns}")

    normalized, candidate_metrics, fp_rows, threshold_rows = analyze_candidates(scene_df)
    boundary_metrics = analyze_boundaries(boundary_df)
    threshold_recommendation = choose_threshold_recommendation(candidate_metrics, threshold_rows)

    write_reviewed_rows_csv(REVIEWED_ROWS_CSV, normalized)
    rows_to_csv(FALSE_POSITIVE_CSV, fp_rows)
    rows_to_csv(THRESHOLD_CSV, threshold_rows)

    old_after = old_project_snapshot()
    workbook_sha_after = file_sha256(INPUT_XLSX)
    workbook_mtime_after = INPUT_XLSX.stat().st_mtime_ns
    old_project_modified = old_before != old_after
    input_workbook_modified = workbook_sha_before != workbook_sha_after or workbook_mtime_before != workbook_mtime_after
    if old_project_modified:
        errors.append({"old_project_modified": True})
    if input_workbook_modified:
        errors.append({"input_workbook_modified": True})

    latest_copied = copy_latest()
    latest_forbidden = latest_forbidden_files()
    if latest_forbidden:
        warnings.append({"latest_forbidden_files_existing": latest_forbidden})

    end = now_iso()
    runtime = round(time.monotonic() - start_mono, 3)
    report = {
        "project_root": str(PROJECT_ROOT),
        "input_xlsx": str(INPUT_XLSX),
        "optional_inputs_missing": missing_optional,
        "sheet_names": sheet_names,
        "start_time": start,
        "end_time": end,
        "actual_runtime_seconds": runtime,
        "actual_runtime_readable": readable_runtime(runtime),
        "candidate_metrics": candidate_metrics,
        "boundary_metrics": boundary_metrics,
        "threshold_recommendation": threshold_recommendation,
        "rule_based_detector_recommendations": [
            "Scene-change 후보는 광고 boundary recall 보조 evidence로 유용하지만, 단독으로 광고 boundary를 확정하면 일반 장면 전환 오탐이 증가한다.",
            "따라서 scene-change는 OCR/audio/ad-probability 변화와 결합하고, boundary ±5초 및 source별 score percentile 조건을 함께 사용하는 것이 적절하다.",
            "boundary ±5초 안의 후보라도 keep_as_boundary_candidate가 false이거나 false_positive_type이 camera/subtitle/lighting 계열이면 단독 evidence로 쓰지 않는다.",
        ],
        "interpretation_limits": [
            "scene_candidate_review는 이미 threshold를 통과한 후보만 포함한다.",
            "threshold 아래에서 탈락한 true scene change를 보지 않았으므로 recall 개선 또는 threshold 하향 주장은 제한한다.",
            "original_opencv_v2_3와 ffmpeg_fallback_failed4는 score scale이 다를 수 있어 source별 분석을 우선한다.",
            "이 결과는 final 광고 탐지 성능이 아니라 scene-change 후보 리뷰 및 boundary evidence audit 결과이다.",
        ],
        "output_paths": {
            "report": str(REPORT_PATH),
            "summary": str(SUMMARY_PATH),
            "log": str(LOG_PATH),
            "reviewed_rows_csv": str(REVIEWED_ROWS_CSV),
            "false_positive_summary_csv": str(FALSE_POSITIVE_CSV),
            "threshold_analysis_csv": str(THRESHOLD_CSV),
            "script": str(SCRIPT_PATH),
        },
        "latest_for_chatgpt_copied": latest_copied,
        "latest_forbidden_files": latest_forbidden,
        "sub_agent_results": {
            "workbook_validation": {"status": "PENDING", "details": "Sub Agent 1 not recorded yet."},
            "added_ad_interval_validation": {"status": "PENDING", "details": "Sub Agent 2 not recorded yet."},
            "review_analysis_validation": {"status": "PENDING", "details": "Sub Agent 3 not recorded yet."},
            "safety_validation": {"status": "PENDING", "details": "Sub Agent 4 not recorded yet."},
        },
        "old_project_snapshot_before": old_before,
        "old_project_snapshot_after": old_after,
        "old_project_modified": old_project_modified,
        "input_workbook_sha256_before": workbook_sha_before,
        "input_workbook_sha256_after": workbook_sha_after,
        "input_workbook_modified": input_workbook_modified,
        "warnings": warnings,
        "errors": errors,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(build_summary(report), encoding="utf-8")
    write_log(report)
    copy_latest()
    print(json.dumps(final_output(report), ensure_ascii=False, indent=2), flush=True)
    return 0 if not errors else 1


def record_sub_agent_results(results: list[list[str]]) -> int:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    for key, status, details in results:
        report.setdefault("sub_agent_results", {})[key] = {"status": status, "details": details}
        log(f"sub agent result recorded: {key}={status} - {details}")
    old_after = old_project_snapshot()
    report["old_project_snapshot_after"] = old_after
    report["old_project_modified"] = report.get("old_project_snapshot_before") != old_after
    workbook_sha_after = file_sha256(INPUT_XLSX)
    report["input_workbook_sha256_after"] = workbook_sha_after
    report["input_workbook_modified"] = report.get("input_workbook_sha256_before") != workbook_sha_after
    errors = report.setdefault("errors", [])
    errors[:] = [err for err in errors if "old_project_modified" not in err and "input_workbook_modified" not in err]
    if report["old_project_modified"]:
        errors.append({"old_project_modified": True})
    if report["input_workbook_modified"]:
        errors.append({"input_workbook_modified": True})
    report["latest_forbidden_files"] = latest_forbidden_files()
    end = now_iso()
    report["end_time"] = end
    try:
        seconds = round((datetime.fromisoformat(end) - datetime.fromisoformat(report["start_time"])).total_seconds(), 3)
        report["actual_runtime_seconds"] = seconds
        report["actual_runtime_readable"] = readable_runtime(seconds)
    except Exception:
        pass
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(build_summary(report), encoding="utf-8")
    write_log(report)
    copy_latest()
    print(json.dumps(final_output(report), ensure_ascii=False, indent=2), flush=True)
    return 0 if not report.get("errors") else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-sub-agent", nargs=3, action="append", metavar=("KEY", "STATUS", "DETAILS"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.record_sub_agent:
        return record_sub_agent_results(args.record_sub_agent)
    return generate()


if __name__ == "__main__":
    raise SystemExit(main())
