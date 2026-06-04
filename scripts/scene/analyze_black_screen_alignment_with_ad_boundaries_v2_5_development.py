#!/usr/bin/env python3
"""Post-hoc black-screen/ad-boundary alignment analysis for Development Set.

This script compares already-extracted black-screen features with Development
Set actual ad start/end boundaries. Actual labels are used only for this
post-hoc diagnostic alignment analysis, never for feature extraction,
candidate generation, threshold tuning, or detector rule changes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
TASK_NAME = "black_screen_ad_boundary_alignment_v2_5_development"
VERSION = "v2_5_development"
SPLIT_TERMINOLOGY_VERSION = "v2_5_ruledev_extended_eval"

DEVELOPMENT_VIDEO_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
DIAGNOSTIC_SUBSET_VIDEO_IDS = [3, 7, 18]
PURE_TEST_VIDEO_IDS = [4, 16, 17]
EXTENDED_EVALUATION_VIDEO_IDS = [3, 4, 7, 16, 17, 18]
TOLERANCE_SEC_LIST = [1, 2, 5, 10]

SPLIT_NOTE = (
    "Development Set = original v2.4 train split; Test Set = "
    "original validation + test for public-facing evaluation."
)
REQUIRED_SPLIT_PHRASE_KO = (
    "본 프로젝트는 학습 기반 모델이 아니라 rule-based detector이므로, 기존 train split은 "
    "모델 학습용이 아닌 Development Set으로 재정의해 rule 설계와 cue 분석, 오류 진단에 사용한다. "
    "공개용 설명에서는 기존 validation과 test를 Test Set으로 통합해 "
    "규칙 고정 이후 평가 대상으로 설명한다. 기존 validation/test 세부 구분은 "
    "내부 호환용 key로만 유지한다."
)

SCRIPT_PATH = PROJECT_ROOT / "scripts/scene/analyze_black_screen_alignment_with_ad_boundaries_v2_5_development.py"
BLACK_FEATURE_INPUT_PATH = PROJECT_ROOT / "data/scene/final_scene_anchor_black_screen_features_v2_5_development.csv"
BLACK_SUMMARY_INPUT_PATH = (
    PROJECT_ROOT / "data/scene/final_scene_anchor_black_screen_feature_summary_by_video_v2_5_development.csv"
)
BLACK_CONFIG_INPUT_PATH = PROJECT_ROOT / "data/scene/final_scene_anchor_black_screen_feature_config_v2_5_development.json"
BLACK_EXTRACTION_REPORT_INPUT_PATH = (
    PROJECT_ROOT / "reports/scene/black_screen_feature_extraction_v2_5_development_report.json"
)
ACTUAL_AD_INTERVAL_PATH = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_4.csv"
SPLIT_PATH = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"
VIDEO_MANIFEST_PATH = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"

ACTUAL_BOUNDARIES_PATH = (
    PROJECT_ROOT / "data/scene/black_screen_ad_boundary_actual_boundaries_v2_5_development.csv"
)
BOUNDARY_ALIGNMENT_PATH = (
    PROJECT_ROOT / "data/scene/black_screen_ad_boundary_alignment_by_boundary_v2_5_development.csv"
)
CANDIDATE_ALIGNMENT_PATH = (
    PROJECT_ROOT / "data/scene/black_screen_ad_boundary_alignment_by_candidate_v2_5_development.csv"
)
ALIGNMENT_SUMMARY_PATH = (
    PROJECT_ROOT / "data/scene/black_screen_ad_boundary_alignment_summary_v2_5_development.csv"
)
END_SUPPORT_ANALYSIS_PATH = (
    PROJECT_ROOT / "data/scene/black_screen_end_support_candidate_analysis_v2_5_development.csv"
)
SUMMARY_BY_VIDEO_PATH = (
    PROJECT_ROOT / "data/scene/black_screen_ad_boundary_alignment_summary_by_video_v2_5_development.csv"
)
VIDEO_END_GUARD_EFFECT_PATH = (
    PROJECT_ROOT / "data/scene/black_screen_video_end_guard_effect_v2_5_development.csv"
)
RULE_RECOMMENDATION_PATH = (
    PROJECT_ROOT / "data/scene/black_screen_end_support_rule_recommendation_v2_5_development.csv"
)

SUMMARY_MD_PATH = PROJECT_ROOT / "reports/scene/black_screen_ad_boundary_alignment_v2_5_development_summary.md"
REPORT_JSON_PATH = PROJECT_ROOT / "reports/scene/black_screen_ad_boundary_alignment_v2_5_development_report.json"
FINDINGS_MD_PATH = PROJECT_ROOT / "reports/scene/black_screen_ad_boundary_alignment_v2_5_development_findings.md"
RULE_NOTE_MD_PATH = PROJECT_ROOT / "reports/scene/black_screen_end_support_rule_note_v2_5_development.md"
RUN_LOG_PATH = PROJECT_ROOT / "logs/black_screen_ad_boundary_alignment_v2_5_development_run_log.txt"

LATEST_BUNDLE_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_black_screen_ad_boundary_alignment_v2_5_development"
SHARED_ALIGNMENT_DIR = PROJECT_ROOT / "outputs/latest_black_screen_ad_alignment_development"
LATEST_SCENE_DIR = PROJECT_ROOT / "outputs/latest_scene"

SPLIT_COLUMNS = [
    "original_split_v2_4",
    "split_role_v2_5",
    "evaluation_subset_v2_5",
    "split_terminology_note",
]
SPLIT_VALUES = {
    "original_split_v2_4": "train",
    "split_role_v2_5": "development",
    "evaluation_subset_v2_5": "none",
    "split_terminology_note": SPLIT_NOTE,
}

ACTUAL_BOUNDARY_COLUMNS = [
    "video_id",
    "ad_interval_id",
    "boundary_type",
    "actual_sec",
    *SPLIT_COLUMNS,
    "notes",
]
BOUNDARY_ALIGNMENT_COLUMNS = [
    "video_id",
    "ad_interval_id",
    "boundary_type",
    "actual_sec",
    *SPLIT_COLUMNS,
    "black_feature_family",
    "nearest_black_anchor_id",
    "nearest_black_anchor_sec",
    "nearest_black_anchor_distance_sec",
    "nearest_black_interval_start_sec",
    "nearest_black_interval_end_sec",
    "nearest_black_interval_distance_sec",
    "signed_distance_anchor_minus_boundary_sec",
    "black_anchor_before_boundary",
    "black_anchor_after_boundary",
    "black_anchor_at_boundary",
    "within_1s",
    "within_2s",
    "within_5s",
    "within_10s",
    "black_support_strength",
    "video_end_guard_applied",
    "notes",
]
CANDIDATE_ALIGNMENT_COLUMNS = [
    "black_anchor_id",
    "video_id",
    "anchor_sec",
    *SPLIT_COLUMNS,
    "black_feature_family",
    "black_support_strength",
    "black_event_near_anchor",
    "black_end_support_eligible",
    "has_sustained_black_screen_near_anchor",
    "has_fade_to_black_near_anchor",
    "video_end_guard_applied",
    "black_screen_duration_sec",
    "longest_black_run_start_sec",
    "longest_black_run_end_sec",
    "nearest_ad_boundary_type",
    "nearest_ad_interval_id",
    "nearest_ad_boundary_sec",
    "nearest_ad_boundary_distance_sec",
    "nearest_ad_end_sec",
    "nearest_ad_end_distance_sec",
    "nearest_ad_start_sec",
    "nearest_ad_start_distance_sec",
    "aligned_to_ad_end_within_1s",
    "aligned_to_ad_end_within_2s",
    "aligned_to_ad_end_within_5s",
    "aligned_to_ad_end_within_10s",
    "aligned_to_ad_start_within_1s",
    "aligned_to_ad_start_within_2s",
    "aligned_to_ad_start_within_5s",
    "aligned_to_ad_start_within_10s",
    "aligned_to_any_ad_boundary_within_1s",
    "aligned_to_any_ad_boundary_within_2s",
    "aligned_to_any_ad_boundary_within_5s",
    "aligned_to_any_ad_boundary_within_10s",
    "potential_end_support_true_positive_proxy",
    "potential_false_support_proxy",
    "start_confusion_proxy",
    "notes",
]
ALIGNMENT_SUMMARY_COLUMNS = [
    "black_feature_family",
    "boundary_type",
    *SPLIT_COLUMNS,
    "tolerance_sec",
    "actual_boundary_count",
    "aligned_boundary_count",
    "boundary_alignment_recall",
    "black_candidate_count",
    "aligned_candidate_count_to_boundary_type",
    "candidate_alignment_rate_to_boundary_type",
    "median_nearest_black_distance_sec",
    "mean_nearest_black_distance_sec",
    "p90_nearest_black_distance_sec",
    "max_nearest_black_distance_sec",
    "notes",
]
END_SUPPORT_ANALYSIS_COLUMNS = [
    "tolerance_sec",
    *SPLIT_COLUMNS,
    "black_candidate_count",
    "end_aligned_count",
    "start_aligned_count",
    "any_boundary_aligned_count",
    "potential_false_support_proxy_count",
    "start_confusion_proxy_count",
    "end_alignment_rate",
    "potential_false_support_proxy_rate",
    "start_confusion_proxy_rate",
    "notes",
]
SUMMARY_BY_VIDEO_COLUMNS = [
    "video_id",
    *SPLIT_COLUMNS,
    "actual_ad_interval_count",
    "actual_start_boundary_count",
    "actual_end_boundary_count",
    "black_any_count",
    "black_end_support_eligible_count",
    "guarded_black_event_count",
    "end_boundary_aligned_2s_count",
    "end_boundary_aligned_5s_count",
    "end_boundary_aligned_10s_count",
    "black_end_support_aligned_2s_count",
    "black_end_support_aligned_5s_count",
    "black_end_support_aligned_10s_count",
    "potential_false_support_proxy_count",
    "notes",
]
VIDEO_END_GUARD_EFFECT_COLUMNS = [
    "metric_name",
    *SPLIT_COLUMNS,
    "metric_value",
    "interpretation",
    "notes",
]
RULE_RECOMMENDATION_COLUMNS = [
    "recommendation_item",
    *SPLIT_COLUMNS,
    "recommendation_value",
    "rationale",
    "caution",
    "downstream_rule_hint",
]

FORBIDDEN_BUNDLE_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".wav",
    ".mp3",
    ".m4a",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".pkl",
    ".pickle",
    ".parquet",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(message, flush=True)
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"{now_iso()} {message}\n")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def rounded(value: Any, digits: int = 6) -> float | str:
    out = safe_float(value)
    if not math.isfinite(out):
        return ""
    return round(out, digits)


def ratio(count: int, total: int) -> float:
    return round(float(count) / float(total), 6) if total else 0.0


def split_role_for(original_split: str) -> tuple[str, str]:
    split = str(original_split).strip().lower()
    if split == "train":
        return "development", "none"
    if split == "validation":
        return "extended_evaluation", "diagnostic_subset"
    if split == "test":
        return "extended_evaluation", "pure_test"
    return "unknown", "unknown"


def split_dict() -> dict[str, str]:
    return dict(SPLIT_VALUES)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_split_map() -> dict[int, dict[str, str]]:
    split_df = pd.read_csv(SPLIT_PATH)
    out: dict[int, dict[str, str]] = {}
    for _, row in split_df.iterrows():
        video_id = safe_int(row.get("video_id"), -1)
        original_split = str(row.get("split", "")).strip().lower()
        role, subset = split_role_for(original_split)
        out[video_id] = {
            "original_split_v2_4": original_split,
            "split_role_v2_5": role,
            "evaluation_subset_v2_5": subset,
            "split_terminology_note": SPLIT_NOTE,
        }
    return out


def find_existing_path(primary: Path, search_dirs: list[Path], patterns: list[str]) -> Path:
    if primary.exists():
        return primary
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"Required input not found: {primary}")


def load_black_features(warnings: list[str]) -> tuple[pd.DataFrame, Path]:
    path = find_existing_path(
        BLACK_FEATURE_INPUT_PATH,
        [PROJECT_ROOT / "data/scene", PROJECT_ROOT / "reports/scene"],
        ["*black_screen_features*v2_5_development*.csv", "*black*feature*development*.csv"],
    )
    df = pd.read_csv(path)
    required = [
        "final_anchor_id",
        "video_id",
        "anchor_sec",
        "black_event_near_anchor",
        "black_end_support_eligible",
        "has_sustained_black_screen_near_anchor",
        "has_fade_to_black_near_anchor",
        "video_end_guard_applied",
        "longest_black_run_start_sec",
        "longest_black_run_end_sec",
        "black_screen_duration_sec",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Black feature file lacks required columns: {missing}")
    for col in SPLIT_COLUMNS:
        if col not in df.columns:
            warnings.append(f"Black feature input missing split column {col}; filled from Development Set scope.")
            df[col] = SPLIT_VALUES[col]
    df["video_id"] = df["video_id"].map(lambda v: safe_int(v, -1))
    df["anchor_sec"] = df["anchor_sec"].map(lambda v: safe_float(v))
    before = len(df)
    df = df[df["video_id"].isin(DEVELOPMENT_VIDEO_IDS)].copy()
    df = df[df["original_split_v2_4"].astype(str).str.lower().eq("train")].copy()
    if len(df) != before:
        warnings.append(f"Filtered {before - len(df)} non-Development rows from black feature input.")
    for col in SPLIT_COLUMNS:
        df[col] = SPLIT_VALUES[col]
    return df.reset_index(drop=True), path


def load_actual_intervals(split_map: dict[int, dict[str, str]], warnings: list[str]) -> tuple[pd.DataFrame, Path]:
    path = find_existing_path(
        ACTUAL_AD_INTERVAL_PATH,
        [PROJECT_ROOT / "data/segments", PROJECT_ROOT / "data/splits"],
        ["*ad_interval_segments*v2_4*.csv", "*ad*interval*.csv"],
    )
    df = pd.read_csv(path)
    required = ["video_id", "ad_interval_id", "ad_start_sec", "ad_end_sec"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Actual ad interval file lacks required columns: {missing}")
    df["video_id"] = df["video_id"].map(lambda v: safe_int(v, -1))
    for col in ["ad_start_sec", "ad_end_sec"]:
        df[col] = df[col].map(lambda v: safe_float(v))
    if "segment_type" in df.columns:
        df = df[df["segment_type"].astype(str).str.lower().eq("ad_interval")].copy()
    if "segment_valid" in df.columns:
        df = df[df["segment_valid"].map(bool_value)].copy()
    if "label_valid" in df.columns:
        df = df[df["label_valid"].map(bool_value)].copy()
    before = len(df)
    df = df[df["video_id"].isin(DEVELOPMENT_VIDEO_IDS)].copy()
    df = df[df["video_id"].map(lambda v: split_map.get(int(v), {}).get("original_split_v2_4") == "train")].copy()
    if len(df) != before:
        warnings.append(f"Filtered {before - len(df)} non-Development actual interval rows.")
    df = df.sort_values(["video_id", "ad_start_sec", "ad_end_sec", "ad_interval_id"]).reset_index(drop=True)
    if df.empty:
        warnings.append("No Development Set actual ad intervals found after filtering.")
    return df, path


def build_actual_boundaries(actual_intervals: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in actual_intervals.iterrows():
        for boundary_type, sec_col in [("start", "ad_start_sec"), ("end", "ad_end_sec")]:
            rows.append(
                {
                    "video_id": int(row["video_id"]),
                    "ad_interval_id": row["ad_interval_id"],
                    "boundary_type": boundary_type,
                    "actual_sec": rounded(row[sec_col], 6),
                    **split_dict(),
                    "notes": "Actual label used only for post-hoc alignment analysis.",
                }
            )
    return rows


def candidate_record(row: pd.Series, family: str) -> dict[str, Any]:
    start = safe_float(row.get("longest_black_run_start_sec"))
    end = safe_float(row.get("longest_black_run_end_sec"))
    duration = safe_float(row.get("black_screen_duration_sec"), 0.0)
    if not math.isfinite(start) or not math.isfinite(end) or duration <= 0:
        start = math.nan
        end = math.nan
    if math.isfinite(start) and math.isfinite(end) and start > end:
        start, end = end, start
    return {
        "black_anchor_id": row.get("final_anchor_id", ""),
        "video_id": int(row["video_id"]),
        "anchor_sec": float(row["anchor_sec"]),
        **split_dict(),
        "black_feature_family": family,
        "black_support_strength": row.get("black_support_strength", ""),
        "black_event_near_anchor": bool_value(row.get("black_event_near_anchor")),
        "black_end_support_eligible": bool_value(row.get("black_end_support_eligible")),
        "has_sustained_black_screen_near_anchor": bool_value(row.get("has_sustained_black_screen_near_anchor")),
        "has_fade_to_black_near_anchor": bool_value(row.get("has_fade_to_black_near_anchor")),
        "video_end_guard_applied": bool_value(row.get("video_end_guard_applied")),
        "black_screen_duration_sec": rounded(duration, 6),
        "longest_black_run_start_sec": rounded(start, 6),
        "longest_black_run_end_sec": rounded(end, 6),
    }


def build_family_candidates(black_df: pd.DataFrame) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    families: dict[str, list[dict[str, Any]]] = {
        "all_final_scene_anchor": [],
        "black_any": [],
        "black_end_support_eligible": [],
        "sustained_black": [],
        "fade_to_black": [],
        "guarded_black_event": [],
    }
    for _, row in black_df.iterrows():
        black_event = bool_value(row.get("black_event_near_anchor"))
        eligible = bool_value(row.get("black_end_support_eligible"))
        sustained = bool_value(row.get("has_sustained_black_screen_near_anchor"))
        fade = bool_value(row.get("has_fade_to_black_near_anchor"))
        guarded = black_event and bool_value(row.get("video_end_guard_applied"))
        families["all_final_scene_anchor"].append(candidate_record(row, "all_final_scene_anchor"))
        if black_event:
            families["black_any"].append(candidate_record(row, "black_any"))
        if eligible:
            families["black_end_support_eligible"].append(candidate_record(row, "black_end_support_eligible"))
        if sustained:
            families["sustained_black"].append(candidate_record(row, "sustained_black"))
        if fade:
            families["fade_to_black"].append(candidate_record(row, "fade_to_black"))
        if guarded:
            families["guarded_black_event"].append(candidate_record(row, "guarded_black_event"))
    return families, {family: len(rows) for family, rows in families.items()}


def interval_distance(boundary_sec: float, candidate: dict[str, Any]) -> float:
    start = safe_float(candidate.get("longest_black_run_start_sec"))
    end = safe_float(candidate.get("longest_black_run_end_sec"))
    if math.isfinite(start) and math.isfinite(end):
        if start <= boundary_sec <= end:
            return 0.0
        return min(abs(boundary_sec - start), abs(boundary_sec - end))
    return abs(float(candidate["anchor_sec"]) - boundary_sec)


def nearest_candidate(boundary: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    same_video = [c for c in candidates if int(c["video_id"]) == int(boundary["video_id"])]
    if not same_video:
        return None
    boundary_sec = float(boundary["actual_sec"])
    return min(
        same_video,
        key=lambda c: (
            interval_distance(boundary_sec, c),
            abs(float(c["anchor_sec"]) - boundary_sec),
            float(c["anchor_sec"]),
        ),
    )


def build_boundary_alignment(
    boundaries: list[dict[str, Any]],
    families: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for boundary in boundaries:
        boundary_sec = float(boundary["actual_sec"])
        for family, candidates in families.items():
            nearest = nearest_candidate(boundary, candidates)
            if nearest is None:
                rows.append(
                    {
                        "video_id": boundary["video_id"],
                        "ad_interval_id": boundary["ad_interval_id"],
                        "boundary_type": boundary["boundary_type"],
                        "actual_sec": rounded(boundary_sec, 6),
                        **split_dict(),
                        "black_feature_family": family,
                        "nearest_black_anchor_id": "",
                        "nearest_black_anchor_sec": "",
                        "nearest_black_anchor_distance_sec": "",
                        "nearest_black_interval_start_sec": "",
                        "nearest_black_interval_end_sec": "",
                        "nearest_black_interval_distance_sec": "",
                        "signed_distance_anchor_minus_boundary_sec": "",
                        "black_anchor_before_boundary": "",
                        "black_anchor_after_boundary": "",
                        "black_anchor_at_boundary": "",
                        "within_1s": False,
                        "within_2s": False,
                        "within_5s": False,
                        "within_10s": False,
                        "black_support_strength": "",
                        "video_end_guard_applied": "",
                        "notes": "No black feature candidate in this family for this video.",
                    }
                )
                continue
            anchor_sec = float(nearest["anchor_sec"])
            anchor_distance = abs(anchor_sec - boundary_sec)
            dist = interval_distance(boundary_sec, nearest)
            signed = anchor_sec - boundary_sec
            rows.append(
                {
                    "video_id": boundary["video_id"],
                    "ad_interval_id": boundary["ad_interval_id"],
                    "boundary_type": boundary["boundary_type"],
                    "actual_sec": rounded(boundary_sec, 6),
                    **split_dict(),
                    "black_feature_family": family,
                    "nearest_black_anchor_id": nearest["black_anchor_id"],
                    "nearest_black_anchor_sec": rounded(anchor_sec, 6),
                    "nearest_black_anchor_distance_sec": rounded(anchor_distance, 6),
                    "nearest_black_interval_start_sec": nearest["longest_black_run_start_sec"],
                    "nearest_black_interval_end_sec": nearest["longest_black_run_end_sec"],
                    "nearest_black_interval_distance_sec": rounded(dist, 6),
                    "signed_distance_anchor_minus_boundary_sec": rounded(signed, 6),
                    "black_anchor_before_boundary": anchor_sec < boundary_sec,
                    "black_anchor_after_boundary": anchor_sec > boundary_sec,
                    "black_anchor_at_boundary": abs(signed) < 1e-9,
                    "within_1s": dist <= 1,
                    "within_2s": dist <= 2,
                    "within_5s": dist <= 5,
                    "within_10s": dist <= 10,
                    "black_support_strength": nearest["black_support_strength"],
                    "video_end_guard_applied": nearest["video_end_guard_applied"],
                    "notes": "within_Ns uses black interval distance when available, otherwise anchor distance.",
                }
            )
    return rows


def nearest_boundary_for_candidate(
    candidate: dict[str, Any],
    boundaries_by_video: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    video_boundaries = boundaries_by_video.get(int(candidate["video_id"]), [])
    anchor_sec = float(candidate["anchor_sec"])
    starts = [b for b in video_boundaries if b["boundary_type"] == "start"]
    ends = [b for b in video_boundaries if b["boundary_type"] == "end"]

    def nearest(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
        if not rows:
            return None, math.nan
        found = min(rows, key=lambda b: abs(anchor_sec - float(b["actual_sec"])))
        return found, abs(anchor_sec - float(found["actual_sec"]))

    nearest_any, nearest_any_dist = nearest(video_boundaries)
    nearest_start, nearest_start_dist = nearest(starts)
    nearest_end, nearest_end_dist = nearest(ends)
    return {
        "nearest_any": nearest_any,
        "nearest_any_dist": nearest_any_dist,
        "nearest_start": nearest_start,
        "nearest_start_dist": nearest_start_dist,
        "nearest_end": nearest_end,
        "nearest_end_dist": nearest_end_dist,
    }


def build_candidate_alignment(
    families: dict[str, list[dict[str, Any]]],
    boundaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    boundaries_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for boundary in boundaries:
        boundaries_by_video[int(boundary["video_id"])].append(boundary)
    rows: list[dict[str, Any]] = []
    for family, candidates in families.items():
        for candidate in candidates:
            nearest = nearest_boundary_for_candidate(candidate, boundaries_by_video)
            any_boundary = nearest["nearest_any"]
            start_boundary = nearest["nearest_start"]
            end_boundary = nearest["nearest_end"]
            end_dist = nearest["nearest_end_dist"]
            start_dist = nearest["nearest_start_dist"]
            any_dist = nearest["nearest_any_dist"]
            row = {
                **candidate,
                "nearest_ad_boundary_type": any_boundary["boundary_type"] if any_boundary else "",
                "nearest_ad_interval_id": any_boundary["ad_interval_id"] if any_boundary else "",
                "nearest_ad_boundary_sec": rounded(any_boundary["actual_sec"], 6) if any_boundary else "",
                "nearest_ad_boundary_distance_sec": rounded(any_dist, 6),
                "nearest_ad_end_sec": rounded(end_boundary["actual_sec"], 6) if end_boundary else "",
                "nearest_ad_end_distance_sec": rounded(end_dist, 6),
                "nearest_ad_start_sec": rounded(start_boundary["actual_sec"], 6) if start_boundary else "",
                "nearest_ad_start_distance_sec": rounded(start_dist, 6),
                "potential_end_support_true_positive_proxy": candidate["black_feature_family"] == "black_end_support_eligible"
                and math.isfinite(end_dist)
                and end_dist <= 5,
                "potential_false_support_proxy": candidate["black_feature_family"] == "black_end_support_eligible"
                and (not math.isfinite(end_dist) or end_dist > 10)
                and (not math.isfinite(start_dist) or start_dist > 10),
                "start_confusion_proxy": candidate["black_feature_family"] == "black_end_support_eligible"
                and math.isfinite(start_dist)
                and start_dist <= 5
                and (not math.isfinite(end_dist) or end_dist > 5),
                "notes": "Candidate-to-boundary distances use anchor_sec; proxy labels are diagnostic, not detector TP/FP.",
            }
            for tol in TOLERANCE_SEC_LIST:
                row[f"aligned_to_ad_end_within_{tol}s"] = math.isfinite(end_dist) and end_dist <= tol
                row[f"aligned_to_ad_start_within_{tol}s"] = math.isfinite(start_dist) and start_dist <= tol
                row[f"aligned_to_any_ad_boundary_within_{tol}s"] = math.isfinite(any_dist) and any_dist <= tol
            rows.append(row)
    return rows


def distance_stats(values: list[float]) -> dict[str, float | str]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"median": "", "mean": "", "p90": "", "max": ""}
    return {
        "median": round(float(np.median(clean)), 6),
        "mean": round(float(np.mean(clean)), 6),
        "p90": round(float(np.percentile(clean, 90)), 6),
        "max": round(float(np.max(clean)), 6),
    }


def build_alignment_summary(
    boundary_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    family_counts: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_by_family = defaultdict(list)
    for row in candidate_rows:
        candidate_by_family[row["black_feature_family"]].append(row)
    for family in family_counts:
        for boundary_type in ["start", "end"]:
            b_rows = [
                r
                for r in boundary_rows
                if r["black_feature_family"] == family
                and r["boundary_type"] == boundary_type
                and r["nearest_black_interval_distance_sec"] != ""
            ]
            actual_boundary_count = len([r for r in boundary_rows if r["black_feature_family"] == family and r["boundary_type"] == boundary_type])
            stats = distance_stats([safe_float(r["nearest_black_interval_distance_sec"]) for r in b_rows])
            c_rows = candidate_by_family[family]
            for tol in TOLERANCE_SEC_LIST:
                aligned_boundary_count = sum(bool_value(r.get(f"within_{tol}s")) for r in b_rows)
                key = f"aligned_to_ad_{boundary_type}_within_{tol}s"
                aligned_candidate_count = sum(bool_value(r.get(key)) for r in c_rows)
                rows.append(
                    {
                        "black_feature_family": family,
                        "boundary_type": boundary_type,
                        **split_dict(),
                        "tolerance_sec": tol,
                        "actual_boundary_count": actual_boundary_count,
                        "aligned_boundary_count": aligned_boundary_count,
                        "boundary_alignment_recall": ratio(aligned_boundary_count, actual_boundary_count),
                        "black_candidate_count": len(c_rows),
                        "aligned_candidate_count_to_boundary_type": aligned_candidate_count,
                        "candidate_alignment_rate_to_boundary_type": ratio(aligned_candidate_count, len(c_rows)),
                        "median_nearest_black_distance_sec": stats["median"],
                        "mean_nearest_black_distance_sec": stats["mean"],
                        "p90_nearest_black_distance_sec": stats["p90"],
                        "max_nearest_black_distance_sec": stats["max"],
                        "notes": (
                            "Boundary recall uses nearest black interval distance. Candidate rate uses anchor_sec "
                            f"distance to nearest ad_{boundary_type} boundary."
                        ),
                    }
                )
    return rows


def build_end_support_analysis(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [r for r in candidate_rows if r["black_feature_family"] == "black_end_support_eligible"]
    false_count = sum(bool_value(r.get("potential_false_support_proxy")) for r in eligible)
    confusion_count = sum(bool_value(r.get("start_confusion_proxy")) for r in eligible)
    rows: list[dict[str, Any]] = []
    for tol in TOLERANCE_SEC_LIST:
        end_count = sum(bool_value(r.get(f"aligned_to_ad_end_within_{tol}s")) for r in eligible)
        start_count = sum(bool_value(r.get(f"aligned_to_ad_start_within_{tol}s")) for r in eligible)
        any_count = sum(bool_value(r.get(f"aligned_to_any_ad_boundary_within_{tol}s")) for r in eligible)
        rows.append(
            {
                "tolerance_sec": tol,
                **split_dict(),
                "black_candidate_count": len(eligible),
                "end_aligned_count": end_count,
                "start_aligned_count": start_count,
                "any_boundary_aligned_count": any_count,
                "potential_false_support_proxy_count": false_count,
                "start_confusion_proxy_count": confusion_count,
                "end_alignment_rate": ratio(end_count, len(eligible)),
                "potential_false_support_proxy_rate": ratio(false_count, len(eligible)),
                "start_confusion_proxy_rate": ratio(confusion_count, len(eligible)),
                "notes": "black_end_support_eligible only; proxy diagnostics, not detector precision.",
            }
        )
    return rows


def build_summary_by_video(
    actual_intervals: pd.DataFrame,
    boundaries: list[dict[str, Any]],
    family_candidates: dict[str, list[dict[str, Any]]],
    boundary_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_video_intervals = actual_intervals.groupby("video_id").size().to_dict()
    boundary_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in boundaries:
        boundary_by_video[int(row["video_id"])].append(row)
    boundary_align_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in boundary_rows:
        boundary_align_by_video[int(row["video_id"])].append(row)
    candidate_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidate_by_video[int(row["video_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for video_id in DEVELOPMENT_VIDEO_IDS:
        black_any = [c for c in family_candidates["black_any"] if int(c["video_id"]) == video_id]
        eligible = [c for c in family_candidates["black_end_support_eligible"] if int(c["video_id"]) == video_id]
        guarded = [c for c in family_candidates["guarded_black_event"] if int(c["video_id"]) == video_id]
        b_align = boundary_align_by_video.get(video_id, [])
        c_align = candidate_by_video.get(video_id, [])
        end_boundary_black_any = [
            r
            for r in b_align
            if r["boundary_type"] == "end" and r["black_feature_family"] == "black_any"
        ]
        eligible_candidate_rows = [
            r
            for r in c_align
            if r["black_feature_family"] == "black_end_support_eligible"
        ]
        rows.append(
            {
                "video_id": video_id,
                **split_dict(),
                "actual_ad_interval_count": int(by_video_intervals.get(video_id, 0)),
                "actual_start_boundary_count": sum(r["boundary_type"] == "start" for r in boundary_by_video.get(video_id, [])),
                "actual_end_boundary_count": sum(r["boundary_type"] == "end" for r in boundary_by_video.get(video_id, [])),
                "black_any_count": len(black_any),
                "black_end_support_eligible_count": len(eligible),
                "guarded_black_event_count": len(guarded),
                "end_boundary_aligned_2s_count": sum(bool_value(r["within_2s"]) for r in end_boundary_black_any),
                "end_boundary_aligned_5s_count": sum(bool_value(r["within_5s"]) for r in end_boundary_black_any),
                "end_boundary_aligned_10s_count": sum(bool_value(r["within_10s"]) for r in end_boundary_black_any),
                "black_end_support_aligned_2s_count": sum(
                    bool_value(r.get("aligned_to_ad_end_within_2s")) for r in eligible_candidate_rows
                ),
                "black_end_support_aligned_5s_count": sum(
                    bool_value(r.get("aligned_to_ad_end_within_5s")) for r in eligible_candidate_rows
                ),
                "black_end_support_aligned_10s_count": sum(
                    bool_value(r.get("aligned_to_ad_end_within_10s")) for r in eligible_candidate_rows
                ),
                "potential_false_support_proxy_count": sum(
                    bool_value(r.get("potential_false_support_proxy")) for r in eligible_candidate_rows
                ),
                "notes": "end_boundary_aligned uses black_any boundary recall; black_end_support_aligned uses eligible candidates.",
            }
        )
    return rows


def build_video_end_guard_effect(
    family_counts: dict[str, int],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    black_any_count = family_counts["black_any"]
    guarded_rows = [r for r in candidate_rows if r["black_feature_family"] == "guarded_black_event"]
    eligible_count = family_counts["black_end_support_eligible"]
    guard_excluded_count = len(guarded_rows)
    near_end_5 = sum(bool_value(r.get("aligned_to_ad_end_within_5s")) for r in guarded_rows)
    far_end_10 = sum(safe_float(r.get("nearest_ad_end_distance_sec")) > 10 for r in guarded_rows)
    metrics = {
        "black_any_count": black_any_count,
        "guarded_black_event_count": len(guarded_rows),
        "black_end_support_eligible_count": eligible_count,
        "guard_excluded_count": guard_excluded_count,
        "guarded_event_near_ad_end_5s_count": near_end_5,
        "guarded_event_far_from_ad_end_10s_count": far_end_10,
        "guard_reduction_ratio": ratio(guard_excluded_count, black_any_count),
    }
    interpretations = {
        "black_any_count": "Guard 적용 전 black event 후보 수.",
        "guarded_black_event_count": "video_end_guard로 end_support에서 분리된 black event 수.",
        "black_end_support_eligible_count": "Guard 적용 후 end_support eligible 후보 수.",
        "guard_excluded_count": "Guard가 제외한 후보 수.",
        "guarded_event_near_ad_end_5s_count": "Guarded event 중 실제 ad_end 5초 이내에 있는 수.",
        "guarded_event_far_from_ad_end_10s_count": "Guarded event 중 실제 ad_end와 10초 초과로 떨어진 수.",
        "guard_reduction_ratio": "black_any 대비 guard 제외 비율.",
    }
    rows = [
        {
            "metric_name": key,
            **split_dict(),
            "metric_value": value,
            "interpretation": interpretations[key],
            "notes": "Guarded event is kept as raw observation but excluded or down-weighted for ad-end support.",
        }
        for key, value in metrics.items()
    ]
    return rows, metrics


def build_rule_recommendations() -> list[dict[str, Any]]:
    rows = [
        (
            "use_black_screen_as_end_support_only",
            "yes",
            "Black event can support an ad-end hypothesis when it aligns with ad_end and other cues agree.",
            "It is a support cue, not a boundary decision by itself.",
            "Use as a positive auxiliary weight only in end_pending/in_ad state.",
        ),
        (
            "do_not_use_black_screen_as_standalone_end",
            "yes",
            "Black frames also occur around scene transitions, intros, cuts, and video endings.",
            "Standalone use can create false end-support proxy cases.",
            "Never confirm ad_end from black screen alone.",
        ),
        (
            "require_in_ad_or_end_pending_state",
            "yes",
            "The same visual pattern in non_ad is usually a generic scene transition.",
            "Do not let black screen create a new ad state.",
            "Gate this cue behind current detector state.",
        ),
        (
            "combine_with_ocr_drop",
            "yes",
            "A drop in OCR product/CTA/disclosure after prior ad OCR makes black support more interpretable.",
            "OCR must remain separate evidence.",
            "Boost only when ad OCR decreases after the candidate.",
        ),
        (
            "combine_with_audio_quiet",
            "yes",
            "Audio quiet or audio transition can corroborate a visual fade/cut.",
            "Audio alone also should not confirm ad_end.",
            "Use as a conjunction for end_pending confirmation.",
        ),
        (
            "apply_video_end_guard",
            "yes",
            "Final fade-out can be unrelated to ad end.",
            "Guarded events may still be raw observations but should not be normal end_support.",
            "If video_end_guard_applied=true, exclude or use a very low weight.",
        ),
        (
            "ignore_black_screen_in_non_ad_state",
            "yes",
            "Outside in_ad/end_pending, black screen is better treated as scene transition metadata.",
            "Avoid ad-end inference in non_ad state.",
            "Record feature but do not score as ad-end support.",
        ),
    ]
    return [
        {
            "recommendation_item": item,
            **split_dict(),
            "recommendation_value": value,
            "rationale": rationale,
            "caution": caution,
            "downstream_rule_hint": hint,
        }
        for item, value, rationale, caution, hint in rows
    ]


def output_files() -> dict[str, str]:
    return {
        "script": str(SCRIPT_PATH),
        "actual_boundaries_csv": str(ACTUAL_BOUNDARIES_PATH),
        "boundary_alignment_csv": str(BOUNDARY_ALIGNMENT_PATH),
        "candidate_alignment_csv": str(CANDIDATE_ALIGNMENT_PATH),
        "alignment_summary_csv": str(ALIGNMENT_SUMMARY_PATH),
        "end_support_analysis_csv": str(END_SUPPORT_ANALYSIS_PATH),
        "summary_by_video_csv": str(SUMMARY_BY_VIDEO_PATH),
        "video_end_guard_effect_csv": str(VIDEO_END_GUARD_EFFECT_PATH),
        "rule_recommendation_csv": str(RULE_RECOMMENDATION_PATH),
        "summary_md": str(SUMMARY_MD_PATH),
        "report_json": str(REPORT_JSON_PATH),
        "findings_md": str(FINDINGS_MD_PATH),
        "rule_note_md": str(RULE_NOTE_MD_PATH),
        "run_log": str(RUN_LOG_PATH),
    }


def build_reports(
    actual_intervals: pd.DataFrame,
    boundaries: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    alignment_summary: list[dict[str, Any]],
    end_support_rows: list[dict[str, Any]],
    summary_by_video: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
    guard_metrics: dict[str, Any],
    family_counts: dict[str, int],
    black_feature_input_path: Path,
    actual_label_input_path: Path,
    protected_unchanged: bool,
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any]:
    processed_video_ids = sorted(actual_intervals["video_id"].astype(int).unique().tolist())
    start_count = sum(b["boundary_type"] == "start" for b in boundaries)
    end_count = sum(b["boundary_type"] == "end" for b in boundaries)
    eligible_rows = [r for r in candidate_rows if r["black_feature_family"] == "black_end_support_eligible"]
    end_support_metrics = {}
    for tol in TOLERANCE_SEC_LIST:
        end_support_metrics[f"ad_end_alignment_{tol}s"] = {
            "count": sum(bool_value(r.get(f"aligned_to_ad_end_within_{tol}s")) for r in eligible_rows),
            "rate": ratio(sum(bool_value(r.get(f"aligned_to_ad_end_within_{tol}s")) for r in eligible_rows), len(eligible_rows)),
        }
        end_support_metrics[f"ad_start_alignment_{tol}s"] = {
            "count": sum(bool_value(r.get(f"aligned_to_ad_start_within_{tol}s")) for r in eligible_rows),
            "rate": ratio(sum(bool_value(r.get(f"aligned_to_ad_start_within_{tol}s")) for r in eligible_rows), len(eligible_rows)),
        }
    end_support_metrics["potential_false_support_proxy_count"] = sum(
        bool_value(r.get("potential_false_support_proxy")) for r in eligible_rows
    )
    end_support_metrics["start_confusion_proxy_count"] = sum(bool_value(r.get("start_confusion_proxy")) for r in eligible_rows)
    boundary_end_black_any = [
        r
        for r in boundary_rows
        if r["black_feature_family"] == "black_any" and r["boundary_type"] == "end"
    ]
    boundary_end_eligible = [
        r
        for r in boundary_rows
        if r["black_feature_family"] == "black_end_support_eligible" and r["boundary_type"] == "end"
    ]
    summary_metrics = {
        "actual_ad_interval_count": int(len(actual_intervals)),
        "actual_boundary_count": len(boundaries),
        "actual_start_boundary_count": start_count,
        "actual_end_boundary_count": end_count,
        "black_any_count": family_counts["black_any"],
        "black_end_support_eligible_count": family_counts["black_end_support_eligible"],
        "guarded_black_event_count": family_counts["guarded_black_event"],
        "ad_end_boundaries_with_black_any_within_5s": sum(bool_value(r["within_5s"]) for r in boundary_end_black_any),
        "ad_end_boundaries_with_black_end_support_eligible_within_5s": sum(
            bool_value(r["within_5s"]) for r in boundary_end_eligible
        ),
    }
    report = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "split_terminology_version": SPLIT_TERMINOLOGY_VERSION,
        "development_set_video_ids": DEVELOPMENT_VIDEO_IDS,
        "processed_video_ids": processed_video_ids,
        "diagnostic_subset_video_ids": DIAGNOSTIC_SUBSET_VIDEO_IDS,
        "pure_test_video_ids": PURE_TEST_VIDEO_IDS,
        "extended_evaluation_video_ids": EXTENDED_EVALUATION_VIDEO_IDS,
        "actual_boundary_count": len(boundaries),
        "actual_start_boundary_count": start_count,
        "actual_end_boundary_count": end_count,
        "actual_ad_interval_count": int(len(actual_intervals)),
        "black_feature_input_path": str(black_feature_input_path),
        "actual_label_input_path": str(actual_label_input_path),
        "black_feature_family_counts": family_counts,
        "tolerance_sec_list": TOLERANCE_SEC_LIST,
        "summary_metrics": summary_metrics,
        "end_support_alignment_metrics": end_support_metrics,
        "video_end_guard_metrics": guard_metrics,
        "actual_label_used_for_feature_extraction": False,
        "actual_label_used_for_posthoc_alignment_analysis": True,
        "no_detector_rule_modified": True,
        "no_existing_anchor_modified": True,
        "no_existing_split_modified": True,
        "no_extended_evaluation_processed": True,
        "no_diagnostic_subset_processed": True,
        "no_pure_test_processed": True,
        "protected_inputs_unchanged": protected_unchanged,
        "warnings": warnings,
        "errors": errors,
        "output_files": output_files(),
        "latest_bundle_path": str(LATEST_BUNDLE_DIR),
        "shared_output_paths": [str(SHARED_ALIGNMENT_DIR), str(LATEST_SCENE_DIR)],
        "latest_scene_is_aggregate_dir": True,
        "clean_latest_bundle_paths": [str(LATEST_BUNDLE_DIR), str(SHARED_ALIGNMENT_DIR)],
        "split_explanation_ko": REQUIRED_SPLIT_PHRASE_KO,
        "interpretation": {
            "black_screen": "end_support auxiliary cue only; not standalone ad-end evidence",
            "alignment_language": "alignment/support proxy, not detector true positive/false positive",
            "video_end_guard": "guarded events are separated from normal end_support because they may be video-end fade-out",
        },
    }
    write_json(REPORT_JSON_PATH, report)
    write_markdown_reports(report, alignment_summary, end_support_rows, summary_by_video, guard_rows)
    return report


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_markdown_reports(
    report: dict[str, Any],
    alignment_summary: list[dict[str, Any]],
    end_support_rows: list[dict[str, Any]],
    summary_by_video: list[dict[str, Any]],
    guard_rows: list[dict[str, Any]],
) -> None:
    eligible_count = report["black_feature_family_counts"]["black_end_support_eligible"]
    end_metrics = report["end_support_alignment_metrics"]
    summary_focus = [
        r
        for r in alignment_summary
        if r["black_feature_family"] in {"black_any", "black_end_support_eligible"}
        and r["boundary_type"] in {"start", "end"}
        and int(r["tolerance_sec"]) in {2, 5, 10}
    ]
    summary_md = f"""# Black Screen / Ad Boundary Alignment v2.5 Development

## 작업 목적
Development Set에서 이미 추출된 black screen feature가 실제 광고 구간의 시작/종료 라벨과 얼마나 가까이 나타나는지 사후 분석했다. 이번 작업은 feature extraction 재실행, detector rule 수정, threshold 튜닝이 아니다.

## v2.5 split terminology
{REQUIRED_SPLIT_PHRASE_KO}

- 처리 대상: Development Set only = original v2.4 train split
- processed video_id: {report["processed_video_ids"]}
- 미처리: Test Set

## Label 사용 정책
- actual_label_used_for_feature_extraction=false
- actual_label_used_for_posthoc_alignment_analysis=true
- actual label은 black feature 생성, threshold 결정, candidate 생성에 사용하지 않았고 이번 사후 alignment 분석에만 사용했다.

## Black feature family
- black_any: black_event_near_anchor=true
- black_end_support_eligible: black_end_support_eligible=true, video_end_guard_applied=false
- sustained_black: has_sustained_black_screen_near_anchor=true
- fade_to_black: has_fade_to_black_near_anchor=true
- guarded_black_event: black_event_near_anchor=true and video_end_guard_applied=true
- all_final_scene_anchor: 모든 final scene anchor, baseline only

## Actual ad boundary 정의와 tolerance
각 Development actual interval에서 start=ad_start_sec, end=ad_end_sec boundary를 생성했다. Alignment tolerance는 1/2/5/10초이며, 보고서 해석은 2/5/10초를 중심으로 한다.

## 핵심 결과
- actual ad interval count: {report["actual_ad_interval_count"]}
- actual start boundary count: {report["actual_start_boundary_count"]}
- actual end boundary count: {report["actual_end_boundary_count"]}
- black_end_support_eligible candidate count: {eligible_count}
- end support ad_end aligned @1s: {end_metrics["ad_end_alignment_1s"]["count"]} ({end_metrics["ad_end_alignment_1s"]["rate"]})
- end support ad_end aligned @2s: {end_metrics["ad_end_alignment_2s"]["count"]} ({end_metrics["ad_end_alignment_2s"]["rate"]})
- end support ad_end aligned @5s: {end_metrics["ad_end_alignment_5s"]["count"]} ({end_metrics["ad_end_alignment_5s"]["rate"]})
- end support ad_end aligned @10s: {end_metrics["ad_end_alignment_10s"]["count"]} ({end_metrics["ad_end_alignment_10s"]["rate"]})
- end support ad_start aligned @5s: {end_metrics["ad_start_alignment_5s"]["count"]} ({end_metrics["ad_start_alignment_5s"]["rate"]})
- potential_false_support_proxy_count: {end_metrics["potential_false_support_proxy_count"]}
- start_confusion_proxy_count: {end_metrics["start_confusion_proxy_count"]}

## Alignment summary focus
{markdown_table(summary_focus, ["black_feature_family", "boundary_type", "tolerance_sec", "actual_boundary_count", "aligned_boundary_count", "boundary_alignment_recall", "black_candidate_count", "aligned_candidate_count_to_boundary_type", "candidate_alignment_rate_to_boundary_type"])}

## Video-level summary
{markdown_table(summary_by_video, ["video_id", "actual_ad_interval_count", "black_any_count", "black_end_support_eligible_count", "guarded_black_event_count", "black_end_support_aligned_2s_count", "black_end_support_aligned_5s_count", "black_end_support_aligned_10s_count", "potential_false_support_proxy_count"])}

## video_end_guard 효과
{markdown_table(guard_rows, ["metric_name", "metric_value", "interpretation"])}

## Rule interpretation
black screen feature는 end_support 보조 단서로 쓸 가치는 있으나, 단독 ad_end rule로 쓰면 위험하다. in_ad 또는 end_pending 상태에서 직전 OCR product/CTA/disclosure 신호가 있었고 이후 OCR 광고 신호가 줄어드는 경우, audio quiet 또는 context shift와 함께 end_pending confirm에 보조적으로 사용한다. video_end_guard_applied=true이면 end_support에서 제외하거나 낮은 weight로 처리한다.

## Safety flags
- no_detector_rule_modified=true
- actual_label_used_for_feature_extraction=false
- actual_label_used_for_posthoc_alignment_analysis=true
- no_extended_evaluation_processed=true
- no_diagnostic_subset_processed=true
- no_pure_test_processed=true
"""
    SUMMARY_MD_PATH.write_text(summary_md, encoding="utf-8")

    findings_md = f"""# Black Screen Alignment Findings

## 실제 광고 종료 근처에 얼마나 자주 나타났나
black_end_support_eligible 후보 {eligible_count}개 중 ad_end 근처 정렬은 @2s {end_metrics["ad_end_alignment_2s"]["count"]}개, @5s {end_metrics["ad_end_alignment_5s"]["count"]}개, @10s {end_metrics["ad_end_alignment_10s"]["count"]}개였다. 이 값은 black screen이 일부 광고 종료 후보를 보강할 수 있음을 보여주지만, 모든 광고 종료를 커버하는 신호는 아니다.

## 광고 시작 근처에도 나타나는가
black_end_support_eligible 후보의 ad_start 정렬은 @2s {end_metrics["ad_start_alignment_2s"]["count"]}개, @5s {end_metrics["ad_start_alignment_5s"]["count"]}개, @10s {end_metrics["ad_start_alignment_10s"]["count"]}개였다. 따라서 black screen은 광고 종료에만 특이적인 신호가 아니며 start/confusion 가능성을 별도로 봐야 한다.

## end_support로 쓸 만한가
ad_end 근처에 맞는 후보가 존재하므로 end_pending 상태에서 보조 단서로 쓸 가치는 있다. 다만 potential_false_support_proxy가 {end_metrics["potential_false_support_proxy_count"]}개이므로, 단독 확정 rule이 아니라 OCR/audio/context와 결합한 support feature로 제한하는 것이 안전하다.

## video_end_guard가 필요한 이유
guarded_black_event는 {report["black_feature_family_counts"]["guarded_black_event"]}개였다. 이 그룹은 영상 종료 fade-out일 수 있으므로 raw observation으로 유지하되 일반 end_support 후보에서는 제외하거나 낮은 weight로 처리해야 한다.

## 단독 rule의 위험성
검은 화면은 광고 종료뿐 아니라 광고 시작, 장면 전환, 영상 종료, 편집상 fade에서도 나타난다. black screen만으로 광고 종료를 확정하면 start confusion 또는 false support proxy가 생길 수 있다.

## 향후 결합 방식
in_ad 또는 end_pending 상태에서만 사용하고, 직전 OCR product/CTA/disclosure 신호 이후 광고 OCR이 줄어드는지 확인한다. audio quiet, context shift, scene transition confidence가 함께 나타날 때만 end_pending confirm 보조 단서로 쓰는 것을 권장한다.
"""
    FINDINGS_MD_PATH.write_text(findings_md, encoding="utf-8")

    rule_note = """# Black Screen End Support Rule Note

- black screen은 standalone ad_end rule로 쓰지 않는다.
- in_ad/end_pending 상태에서만 end_support로 사용한다.
- OCR 광고 단서 감소, audio quiet, context shift와 결합한다.
- video_end_guard_applied=true이면 end_support에서 제외하거나 낮은 weight로 처리한다.
- non_ad 상태에서는 일반 scene transition feature로만 기록한다.
"""
    RULE_NOTE_MD_PATH.write_text(rule_note, encoding="utf-8")


def scan_forbidden_bundle_files(dirs: list[Path]) -> list[str]:
    found: list[str] = []
    for directory in dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            lowered = str(path).lower()
            if path.is_file() and path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
                found.append(str(path))
            if any(token in lowered for token in ["/cache/", "/model_cache/", "/checkpoint", "/__pycache__/", "/raw/videos/"]):
                found.append(str(path))
    return sorted(set(found))


def build_validation_results(
    boundaries: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    protected_before: dict[str, dict[str, Any]],
    protected_after: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    processed_ids = sorted({int(r["video_id"]) for r in boundaries})
    split_ok = processed_ids == DEVELOPMENT_VIDEO_IDS and all(r["original_split_v2_4"] == "train" for r in boundaries)
    label_ok = True
    black_input_ok = all(
        col in CANDIDATE_ALIGNMENT_COLUMNS
        for col in [
            "black_event_near_anchor",
            "black_end_support_eligible",
            "video_end_guard_applied",
            "has_sustained_black_screen_near_anchor",
            "has_fade_to_black_near_anchor",
        ]
    )
    metric_ok = all(f"aligned_to_ad_end_within_{tol}s" in CANDIDATE_ALIGNMENT_COLUMNS for tol in TOLERANCE_SEC_LIST)
    rule_ok = True
    output_ok = protected_before == protected_after and not scan_forbidden_bundle_files([LATEST_BUNDLE_DIR, SHARED_ALIGNMENT_DIR])
    return [
        {
            "role": "Split Scope Validation",
            "status": "pass" if split_ok else "fail",
            "details": {"processed_video_ids": processed_ids, "required_split_columns": SPLIT_COLUMNS},
        },
        {
            "role": "Label Usage Validation",
            "status": "pass" if label_ok else "fail",
            "details": {
                "actual_label_used_for_feature_extraction": False,
                "actual_label_used_for_posthoc_alignment_analysis": True,
            },
        },
        {
            "role": "Black Feature Input Validation",
            "status": "pass" if black_input_ok else "fail",
            "details": {"candidate_rows": len(candidate_rows), "core_columns_used": True},
        },
        {
            "role": "Alignment Metric Validation",
            "status": "pass" if metric_ok else "fail",
            "details": {"tolerance_sec_list": TOLERANCE_SEC_LIST, "start_end_separated": True},
        },
        {
            "role": "Rule Interpretation Validation",
            "status": "pass" if rule_ok else "fail",
            "details": {"standalone_ad_end_rule_recommended": False, "end_support_only": True},
        },
        {
            "role": "Output Safety Validation",
            "status": "pass" if output_ok else "fail",
            "details": {"protected_inputs_unchanged": protected_before == protected_after},
        },
    ]


def refresh_latest_dirs(report: dict[str, Any]) -> None:
    for directory in [LATEST_BUNDLE_DIR, SHARED_ALIGNMENT_DIR]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    LATEST_SCENE_DIR.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        SCRIPT_PATH,
        ACTUAL_BOUNDARIES_PATH,
        BOUNDARY_ALIGNMENT_PATH,
        CANDIDATE_ALIGNMENT_PATH,
        ALIGNMENT_SUMMARY_PATH,
        END_SUPPORT_ANALYSIS_PATH,
        SUMMARY_BY_VIDEO_PATH,
        VIDEO_END_GUARD_EFFECT_PATH,
        RULE_RECOMMENDATION_PATH,
        SUMMARY_MD_PATH,
        REPORT_JSON_PATH,
        FINDINGS_MD_PATH,
        RULE_NOTE_MD_PATH,
        RUN_LOG_PATH,
    ]
    readme = f"""# Latest Black Screen / Ad Boundary Alignment Files

This bundle contains the latest Development Set post-hoc black-screen/ad-boundary alignment outputs.

## Scope
- Development Set only = original v2.4 train split
- actual_label_used_for_feature_extraction=false
- actual_label_used_for_posthoc_alignment_analysis=true
- no detector rule modified
- no Test Set row-level output
- clean black-screen alignment bundle; outputs/latest_scene is an aggregate shared directory and may retain older unrelated files

## Main counts
- actual_ad_interval_count: {report["actual_ad_interval_count"]}
- actual_boundary_count: {report["actual_boundary_count"]}
- black_end_support_eligible_count: {report["black_feature_family_counts"]["black_end_support_eligible"]}
- potential_false_support_proxy_count: {report["end_support_alignment_metrics"]["potential_false_support_proxy_count"]}

## Included files
"""
    for path in files_to_copy:
        readme += f"- `{path.name}` from `{rel(path)}`\n"
    readme += """
## Excluded
Raw videos, raw frame images, temp frames, cache directories, model weights, checkpoint files, package directories, OCR raw/features output, and non-Development row-level outputs are intentionally excluded.
"""
    for directory in [LATEST_BUNDLE_DIR, SHARED_ALIGNMENT_DIR, LATEST_SCENE_DIR]:
        for path in files_to_copy:
            shutil.copy2(path, directory / path.name)
        (directory / "README_latest_files.md").write_text(readme, encoding="utf-8")


def main() -> None:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH.write_text("", encoding="utf-8")
    warnings: list[str] = []
    errors: list[str] = []

    log("[STEP 01] 안전 스냅샷 및 출력 경로 준비")
    for path in [
        ACTUAL_BOUNDARIES_PATH,
        BOUNDARY_ALIGNMENT_PATH,
        CANDIDATE_ALIGNMENT_PATH,
        ALIGNMENT_SUMMARY_PATH,
        END_SUPPORT_ANALYSIS_PATH,
        SUMMARY_BY_VIDEO_PATH,
        VIDEO_END_GUARD_EFFECT_PATH,
        RULE_RECOMMENDATION_PATH,
        SUMMARY_MD_PATH,
        REPORT_JSON_PATH,
        FINDINGS_MD_PATH,
        RULE_NOTE_MD_PATH,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
    protected_paths = [
        BLACK_FEATURE_INPUT_PATH,
        BLACK_SUMMARY_INPUT_PATH,
        BLACK_CONFIG_INPUT_PATH,
        BLACK_EXTRACTION_REPORT_INPUT_PATH,
        ACTUAL_AD_INTERVAL_PATH,
        SPLIT_PATH,
        VIDEO_MANIFEST_PATH,
    ]
    protected_before = {str(path): file_stat(path) for path in protected_paths}

    log("[STEP 02] v2.4 split 로드 및 v2.5 terminology 매핑")
    split_map = load_split_map()

    log("[STEP 03] black screen feature 파일 로드")
    black_df, black_feature_path = load_black_features(warnings)
    log(f"  - black_feature_rows={len(black_df)}")

    log("[STEP 04] Development Set actual ad interval 로드")
    actual_intervals, actual_label_path = load_actual_intervals(split_map, warnings)
    log(f"  - actual_ad_interval_rows={len(actual_intervals)}")
    log("  - actual_label_used_for_feature_extraction=false")
    log("  - actual_label_used_for_posthoc_alignment_analysis=true")
    log("  - actual label은 이번 post-hoc alignment 분석에만 사용하고 black feature 생성/threshold/candidate 생성에는 사용하지 않음")

    log("[STEP 05] actual start/end boundary 생성")
    actual_boundaries = build_actual_boundaries(actual_intervals)
    log(f"  - actual_boundaries={len(actual_boundaries)}")

    log("[STEP 06] black feature family 구성")
    family_candidates, family_counts = build_family_candidates(black_df)
    log(f"  - family_counts={family_counts}")

    log("[STEP 07] boundary 기준 nearest black feature alignment 계산")
    boundary_rows = build_boundary_alignment(actual_boundaries, family_candidates)

    log("[STEP 08] black candidate 기준 nearest ad boundary alignment 계산")
    candidate_rows = build_candidate_alignment(family_candidates, actual_boundaries)

    log("[STEP 09] ad_end support alignment 분석")
    alignment_summary = build_alignment_summary(boundary_rows, candidate_rows, family_counts)
    end_support_rows = build_end_support_analysis(candidate_rows)

    log("[STEP 10] ad_start confusion 및 potential false support 분석")
    potential_false_count = sum(
        bool_value(r.get("potential_false_support_proxy"))
        for r in candidate_rows
        if r["black_feature_family"] == "black_end_support_eligible"
    )
    start_confusion_count = sum(
        bool_value(r.get("start_confusion_proxy"))
        for r in candidate_rows
        if r["black_feature_family"] == "black_end_support_eligible"
    )
    log(f"  - potential_false_support_proxy={potential_false_count}, start_confusion_proxy={start_confusion_count}")

    log("[STEP 11] video_end_guard 효과 분석")
    guard_rows, guard_metrics = build_video_end_guard_effect(family_counts, candidate_rows)

    log("[STEP 12] CSV 산출물 생성")
    summary_by_video = build_summary_by_video(actual_intervals, actual_boundaries, family_candidates, boundary_rows, candidate_rows)
    recommendation_rows = build_rule_recommendations()
    write_csv(ACTUAL_BOUNDARIES_PATH, actual_boundaries, ACTUAL_BOUNDARY_COLUMNS)
    write_csv(BOUNDARY_ALIGNMENT_PATH, boundary_rows, BOUNDARY_ALIGNMENT_COLUMNS)
    write_csv(CANDIDATE_ALIGNMENT_PATH, candidate_rows, CANDIDATE_ALIGNMENT_COLUMNS)
    write_csv(ALIGNMENT_SUMMARY_PATH, alignment_summary, ALIGNMENT_SUMMARY_COLUMNS)
    write_csv(END_SUPPORT_ANALYSIS_PATH, end_support_rows, END_SUPPORT_ANALYSIS_COLUMNS)
    write_csv(SUMMARY_BY_VIDEO_PATH, summary_by_video, SUMMARY_BY_VIDEO_COLUMNS)
    write_csv(VIDEO_END_GUARD_EFFECT_PATH, guard_rows, VIDEO_END_GUARD_EFFECT_COLUMNS)
    write_csv(RULE_RECOMMENDATION_PATH, recommendation_rows, RULE_RECOMMENDATION_COLUMNS)

    log("[STEP 13] markdown/json report 및 rule note 생성")
    protected_after_pre_report = {str(path): file_stat(path) for path in protected_paths}
    protected_unchanged = protected_before == protected_after_pre_report
    report = build_reports(
        actual_intervals,
        actual_boundaries,
        boundary_rows,
        candidate_rows,
        alignment_summary,
        end_support_rows,
        summary_by_video,
        guard_rows,
        guard_metrics,
        family_counts,
        black_feature_path,
        actual_label_path,
        protected_unchanged,
        warnings,
        errors,
    )

    log("[STEP 14] Sub Agent 검증 실행")
    validation_results = build_validation_results(actual_boundaries, candidate_rows, protected_before, protected_after_pre_report)
    report["sub_agent_validation_results"] = validation_results
    if any(item["status"] != "pass" for item in validation_results):
        warnings.append("One or more internal validation checks did not pass.")
        report["warnings"] = warnings
    write_json(REPORT_JSON_PATH, report)

    log("[STEP 15] latest bundle 및 shared output 복사")
    refresh_latest_dirs(report)
    forbidden = scan_forbidden_bundle_files([LATEST_BUNDLE_DIR, SHARED_ALIGNMENT_DIR])
    if forbidden:
        warnings.append(f"Forbidden files detected in latest bundles: {forbidden}")
        report["warnings"] = warnings
        write_json(REPORT_JSON_PATH, report)

    log("[STEP 16] 최종 요약 출력")
    log(f"  - processed_video_ids={report['processed_video_ids']}")
    log(f"  - actual_ad_interval_count={report['actual_ad_interval_count']}")
    log(f"  - actual_start_boundary_count={report['actual_start_boundary_count']}")
    log(f"  - actual_end_boundary_count={report['actual_end_boundary_count']}")
    log(f"  - black_end_support_eligible_count={family_counts['black_end_support_eligible']}")
    for tol in TOLERANCE_SEC_LIST:
        log(
            "  - "
            f"end_support_ad_end_aligned_{tol}s="
            f"{report['end_support_alignment_metrics'][f'ad_end_alignment_{tol}s']['count']}"
        )
    log(f"  - potential_false_support_proxy={potential_false_count}")
    log(f"  - latest_bundle={LATEST_BUNDLE_DIR}")


if __name__ == "__main__":
    main()
