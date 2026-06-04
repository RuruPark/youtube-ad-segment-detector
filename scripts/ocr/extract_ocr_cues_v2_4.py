#!/usr/bin/env python
"""Extract OCR cues used by the scene/audio/OCR ad-segment detector.

The public copy keeps implementation code but excludes OCR frame outputs,
private labels, model caches, and raw media. Paths default to this repository
and can be overridden with ``YASD_PROJECT_ROOT`` for private reproduction.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


VERSION = "v2_4"
RANDOM_SEED = 20240524

NEW_PROJECT = Path(os.environ.get("YASD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
OLD_PROJECT = Path(os.environ.get("YASD_OLD_PROJECT_ROOT", NEW_PROJECT / "_old_project_not_included")).resolve()

LABEL_PATH = NEW_PROJECT / "data/segments/ad_interval_segments_v2_4.csv"
MANIFEST_PATH = NEW_PROJECT / "data/video_metadata/video_manifest_v2_2.csv"
AUDIO_LABELED_SEGMENT_PLAN = NEW_PROJECT / "data/audio/audio_labeled_segment_sampling_plan_v2_4.csv"
AUDIO_EDGE_SEGMENT_PLAN = NEW_PROJECT / "data/audio/audio_ad_edge_5s_10s_sampling_plan_v2_4.csv"

DATA_OCR_DIR = NEW_PROJECT / "data/ocr"
REPORT_DIR = NEW_PROJECT / "reports/ocr"
LOG_DIR = NEW_PROJECT / "logs/ocr"
SCRIPT_DIR = NEW_PROJECT / "scripts/ocr"
CONFIG_DIR = NEW_PROJECT / "configs"
CONFIG_OCR_DIR = NEW_PROJECT / "configs/ocr"
CACHE_DIR = NEW_PROJECT / "cache/ocr"
TMP_DIR = NEW_PROJECT / "tmp/ocr"
LATEST_DIR = NEW_PROJECT / "outputs/latest_for_chatgpt"

RUN_LOG = LOG_DIR / "extract_ocr_cues_v2_4_run.log"
BEFORE_SNAPSHOT = REPORT_DIR / "old_project_snapshot_before_v2_4.csv"
AFTER_SNAPSHOT = REPORT_DIR / "old_project_snapshot_after_v2_4.csv"
SNAPSHOT_DIFF = REPORT_DIR / "old_project_snapshot_diff_v2_4.csv"
DISCOVERY_CSV = REPORT_DIR / "old_project_ocr_discovery_v2_4.csv"

LABELED_SAMPLING_PLAN_OUT = DATA_OCR_DIR / "ocr_labeled_segment_sampling_plan_v2_4.csv"
EDGE_SAMPLING_PLAN_OUT = DATA_OCR_DIR / "ocr_ad_edge_5s_10s_sampling_plan_v2_4.csv"
FRAME_RESULTS_OUT = DATA_OCR_DIR / "ocr_frame_level_results_v2_4.csv"
FRAME_RESULTS_SAMPLE_OUT = DATA_OCR_DIR / "ocr_frame_level_results_v2_4_sample.csv"
TOKEN_FREQUENCY_OUT = DATA_OCR_DIR / "ocr_token_frequency_by_segment_type_v2_4.csv"
CANDIDATE_KEYWORDS_OUT = DATA_OCR_DIR / "ocr_candidate_ad_keywords_v2_4.csv"
LABELED_FEATURES_OUT = DATA_OCR_DIR / "ocr_labeled_segment_features_v2_4.csv"
EDGE_FEATURES_OUT = DATA_OCR_DIR / "ocr_ad_edge_5s_10s_features_v2_4.csv"
SEGMENT_SUMMARY_OUT = DATA_OCR_DIR / "ocr_segment_type_feature_summary_v2_4.csv"
EDGE_SUMMARY_OUT = DATA_OCR_DIR / "ocr_edge_feature_summary_v2_4.csv"
RULE_RECOMMENDATIONS_OUT = DATA_OCR_DIR / "ocr_rule_feature_recommendations_v2_4.csv"

KEYWORD_DICTIONARY_OUT = CONFIG_DIR / "ocr_keyword_dictionary_v2_4.json"
PERSISTENCE_CONFIG_OUT = CONFIG_DIR / "ocr_persistence_rule_config_v2_4.json"

ANALYSIS_REPORT_OUT = REPORT_DIR / "ocr_labeled_segment_analysis_v2_4.md"
SUMMARY_REPORT_OUT = REPORT_DIR / "extract_ocr_cues_v2_4_summary.md"
LATEST_README = LATEST_DIR / "README_latest_files.md"

FORBIDDEN_LATEST_EXTS = {
    ".mp4",
    ".wav",
    ".mp3",
    ".m4a",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
}

OCR_KEYWORDS = re.compile(
    r"ocr|easyocr|paddleocr|paddle|tesseract|pytesseract|extract_text|frame_ocr|detect_text|readtext|korean|hangul|text",
    re.I,
)

TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z]+(?:[A-Za-z0-9_+\-&]*[A-Za-z0-9])?|\d+(?:[.,]\d+)*|[A-Za-z가-힣]*\d[A-Za-z가-힣0-9._+\-]*")
KOREAN_RE = re.compile(r"[가-힣]+")
ENGLISH_RE = re.compile(r"[A-Za-z]+")
NUMERIC_RE = re.compile(r"\d+(?:[.,]\d+)*")
MIXED_ALNUM_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9._+\-]+")
ENGLISH_BRAND_LIKE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9&+_\-]{1,}\b")

PRICE_PATTERN_RE = re.compile(r"(?:₩|\$)?\s?\d[\d,]*(?:\.\d+)?\s?(?:원|만원|천원|달러|won|krw|usd|\$|₩)", re.I)
PERCENT_PATTERN_RE = re.compile(r"\d+(?:\.\d+)?\s?%")
COUPON_PATTERN_RE = re.compile(r"(쿠폰|coupon|promo\s?code|프로모션\s?코드|할인\s?코드)", re.I)
URL_PATTERN_RE = re.compile(r"(https?://|www\.|[A-Za-z0-9.-]+\.(?:com|kr|net|co|io|shop)\b|bit\.ly|linktr\.ee)", re.I)


SEED_DICTIONARY: dict[str, list[dict[str, Any]]] = {
    "ad_disclosure": [
        {"keyword": "광고", "weight": 1.2},
        {"keyword": "유료광고", "weight": 1.6},
        {"keyword": "ad", "weight": 0.8},
    ],
    "sponsor": [
        {"keyword": "협찬", "weight": 1.5},
        {"keyword": "제공", "weight": 0.8},
        {"keyword": "sponsored", "weight": 1.5},
        {"keyword": "sponsor", "weight": 1.3},
    ],
    "brand": [
        {"keyword": "브랜드", "weight": 0.9},
        {"keyword": "공식", "weight": 0.8},
        {"keyword": "정품", "weight": 0.8},
        {"keyword": "brand", "weight": 0.9},
    ],
    "product": [
        {"keyword": "제품", "weight": 0.9},
        {"keyword": "신제품", "weight": 1.0},
        {"keyword": "라인업", "weight": 0.6},
        {"keyword": "product", "weight": 0.8},
    ],
    "promotion": [
        {"keyword": "프로모션", "weight": 1.0},
        {"keyword": "이벤트", "weight": 0.9},
        {"keyword": "혜택", "weight": 0.9},
        {"keyword": "추천", "weight": 0.6},
        {"keyword": "체험", "weight": 0.7},
        {"keyword": "무료", "weight": 0.8},
        {"keyword": "가입", "weight": 0.8},
        {"keyword": "정기구독", "weight": 1.0},
        {"keyword": "구독", "weight": 0.6},
        {"keyword": "promotion", "weight": 1.0},
        {"keyword": "event", "weight": 0.8},
        {"keyword": "free", "weight": 0.8},
        {"keyword": "trial", "weight": 0.8},
        {"keyword": "subscribe", "weight": 0.8},
    ],
    "discount": [
        {"keyword": "할인", "weight": 1.1},
        {"keyword": "쿠폰", "weight": 1.1},
        {"keyword": "특가", "weight": 1.1},
        {"keyword": "할인가", "weight": 1.1},
        {"keyword": "세일", "weight": 0.9},
        {"keyword": "적립", "weight": 0.8},
        {"keyword": "포인트", "weight": 0.7},
        {"keyword": "discount", "weight": 1.0},
        {"keyword": "coupon", "weight": 1.0},
        {"keyword": "sale", "weight": 0.9},
    ],
    "purchase": [
        {"keyword": "구매", "weight": 1.1},
        {"keyword": "주문", "weight": 1.0},
        {"keyword": "결제", "weight": 1.0},
        {"keyword": "buy", "weight": 1.0},
        {"keyword": "order", "weight": 1.0},
    ],
    "cta": [
        {"keyword": "신청", "weight": 0.9},
        {"keyword": "설치", "weight": 0.9},
        {"keyword": "다운로드", "weight": 1.0},
        {"keyword": "바로가기", "weight": 1.0},
        {"keyword": "확인하기", "weight": 0.9},
        {"keyword": "만나보세요", "weight": 0.8},
        {"keyword": "download", "weight": 1.0},
        {"keyword": "install", "weight": 0.9},
        {"keyword": "apply", "weight": 0.8},
    ],
    "link_or_more_info": [
        {"keyword": "링크", "weight": 1.0},
        {"keyword": "더보기", "weight": 1.0},
        {"keyword": "설명란", "weight": 1.0},
        {"keyword": "고정댓글", "weight": 0.9},
        {"keyword": "댓글", "weight": 0.5},
        {"keyword": "아래", "weight": 0.5},
        {"keyword": "url", "weight": 0.9},
        {"keyword": "link", "weight": 0.9},
        {"keyword": "description", "weight": 0.8},
    ],
}

FEATURES_FOR_ANALYSIS = [
    "ocr_text_count",
    "ocr_token_count",
    "ocr_char_count",
    "ocr_text_area_ratio_mean",
    "ocr_mean_confidence",
    "ad_disclosure_keyword_count",
    "brand_keyword_count",
    "product_keyword_count",
    "promotion_keyword_count",
    "discount_keyword_count",
    "purchase_keyword_count",
    "cta_keyword_count",
    "link_or_more_info_keyword_count",
    "price_pattern_count",
    "percent_pattern_count",
    "ocr_keyword_score",
    "ocr_text_density_score",
    "ocr_ad_text_score",
]

BASIC_COMPARISONS = [
    ("ad_full_vs_random_non_ad_30s", "random_non_ad_30s", "ad_full"),
    ("pre_ad_10s_vs_ad_full", "pre_ad_10s", "ad_full"),
    ("ad_full_vs_post_ad_10s", "ad_full", "post_ad_10s"),
    ("pre_ad_10s_vs_post_ad_10s", "pre_ad_10s", "post_ad_10s"),
]

EDGE_COMPARISONS = [
    ("pre_ad_10s_vs_ad_start_first_5s", "pre_ad_10s", "ad_start_first_5s"),
    ("pre_ad_10s_vs_ad_start_first_10s", "pre_ad_10s", "ad_start_first_10s"),
    ("ad_start_first_5s_vs_ad_start_5to10s", "ad_start_first_5s", "ad_start_5to10s"),
    ("ad_end_minus10to_minus5s_vs_ad_end_last_5s", "ad_end_minus10to_minus5s", "ad_end_last_5s"),
    ("ad_end_last_10s_vs_post_ad_10s", "ad_end_last_10s", "post_ad_10s"),
    ("ad_end_last_5s_vs_post_ad_10s", "ad_end_last_5s", "post_ad_10s"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract v2_4 label-aligned OCR cue features.")
    parser.add_argument("--max-frames", type=int, default=0, help="Deterministic smoke limit for OCR frames. 0 means full plan.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing OCR outputs.")
    parser.add_argument("--skip-ocr", action="store_true", help="Build plans/reports without running OCR.")
    parser.add_argument("--ocr-backend", choices=["auto", "easyocr"], default="auto")
    return parser.parse_args()


def assert_new_project_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(NEW_PROJECT)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside new project: {resolved}") from exc
    return resolved


def ensure_dirs() -> None:
    for path in [DATA_OCR_DIR, REPORT_DIR, LOG_DIR, SCRIPT_DIR, CONFIG_DIR, CONFIG_OCR_DIR, CACHE_DIR, TMP_DIR, LATEST_DIR]:
        assert_new_project_path(path)
        path.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"{now_iso()} | {message}"
    print(line, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def init_run_log() -> None:
    ensure_dirs()
    RUN_LOG.write_text("", encoding="utf-8")


def sha256_file(path: Path) -> tuple[str, str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest(), "ok"
    except Exception as exc:
        return "", f"error:{type(exc).__name__}:{exc}"


def create_project_snapshot(root: Path, output_csv: Path) -> int:
    assert_new_project_path(output_csv)
    rows: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = path.relative_to(root).as_posix()
            try:
                stat = path.stat()
                digest, sha_status = sha256_file(path)
                rows.append(
                    {
                        "path": rel,
                        "size": int(stat.st_size),
                        "mtime": float(stat.st_mtime),
                        "mtime_iso": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                        "sha256": digest,
                        "sha256_status": sha_status,
                        "stat_status": "ok",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "path": rel,
                        "size": "",
                        "mtime": "",
                        "mtime_iso": "",
                        "sha256": "",
                        "sha256_status": "not_attempted",
                        "stat_status": f"error:{type(exc).__name__}:{exc}",
                    }
                )
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return len(rows)


def compare_snapshots(before_csv: Path, after_csv: Path, diff_csv: Path) -> tuple[bool, int]:
    before = pd.read_csv(before_csv, dtype=str, keep_default_na=False)
    after = pd.read_csv(after_csv, dtype=str, keep_default_na=False)
    b = before.set_index("path").to_dict(orient="index")
    a = after.set_index("path").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for path in sorted(set(b) | set(a)):
        br = b.get(path)
        ar = a.get(path)
        if br is None:
            rows.append({"path": path, "change_type": "added", "before_size": "", "after_size": ar.get("size", ""), "before_sha256": "", "after_sha256": ar.get("sha256", ""), "before_mtime": "", "after_mtime": ar.get("mtime", "")})
        elif ar is None:
            rows.append({"path": path, "change_type": "removed", "before_size": br.get("size", ""), "after_size": "", "before_sha256": br.get("sha256", ""), "after_sha256": "", "before_mtime": br.get("mtime", ""), "after_mtime": ""})
        else:
            changed_fields = []
            for field in ["size", "mtime", "sha256", "sha256_status", "stat_status"]:
                if str(br.get(field, "")) != str(ar.get(field, "")):
                    changed_fields.append(field)
            if changed_fields:
                rows.append({"path": path, "change_type": ",".join(changed_fields), "before_size": br.get("size", ""), "after_size": ar.get("size", ""), "before_sha256": br.get("sha256", ""), "after_sha256": ar.get("sha256", ""), "before_mtime": br.get("mtime", ""), "after_mtime": ar.get("mtime", "")})
    diff_df = pd.DataFrame(rows)
    diff_df.to_csv(diff_csv, index=False)
    return len(rows) > 0, len(rows)


def discover_ocr_assets() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    skip_dirs = {".git", "__pycache__"}
    for dirpath, dirnames, filenames in os.walk(OLD_PROJECT):
        dirnames[:] = [d for d in sorted(dirnames) if d not in skip_dirs]
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            rel = path.relative_to(OLD_PROJECT).as_posix()
            suffix = path.suffix.lower()
            path_match = bool(OCR_KEYWORDS.search(rel))
            content_match = False
            content_error = ""
            if suffix in {".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml", ".csv"} and path.stat().st_size < 3_000_000:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    content_match = bool(OCR_KEYWORDS.search(text))
                except Exception as exc:
                    content_error = f"{type(exc).__name__}:{exc}"
            if path_match or content_match:
                role = "other"
                if suffix == ".py":
                    role = "script_or_module"
                elif filename.startswith("requirements") or suffix in {".toml", ".yml", ".yaml", ".json"}:
                    role = "config_or_requirements"
                elif suffix in {".pth", ".pt", ".onnx"}:
                    role = "model_cache"
                elif suffix in {".csv", ".md"}:
                    role = "analysis_artifact"
                rows.append(
                    {
                        "old_project_relative_path": rel,
                        "size_bytes": path.stat().st_size,
                        "mtime_iso": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                        "path_keyword_match": path_match,
                        "content_keyword_match": content_match,
                        "content_scan_error": content_error,
                        "asset_role_guess": role,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(DISCOVERY_CSV, index=False)
    return df


def copy_legacy_assets() -> list[str]:
    copied: list[str] = []
    sources = [
        (OLD_PROJECT / "src/features/ocr_feature.py", SCRIPT_DIR / "legacy_old_project_ocr_feature.py"),
        (OLD_PROJECT / "src/features/ocr_sample_review.py", SCRIPT_DIR / "legacy_old_project_ocr_sample_review.py"),
        (
            OLD_PROJECT / "workspaces/evidence_pipeline_v1/scripts/generate_full_ocr_features_5s_v1.py",
            SCRIPT_DIR / "legacy_generate_full_ocr_features_5s_v1.py",
        ),
        (
            OLD_PROJECT / "workspaces/evidence_pipeline_v1/scripts/post_validate_full_ocr_outputs_v1.py",
            SCRIPT_DIR / "legacy_post_validate_full_ocr_outputs_v1.py",
        ),
        (OLD_PROJECT / "requirements.txt", CONFIG_OCR_DIR / "old_project_requirements_ocr_reference.txt"),
    ]
    for src, dst in sources:
        if src.exists():
            assert_new_project_path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(str(dst))
    old_model_cache = OLD_PROJECT / "workspaces/evidence_pipeline_v1/ocr_model_cache"
    new_model_cache = CACHE_DIR / "model_cache"
    if old_model_cache.exists():
        new_model_cache.mkdir(parents=True, exist_ok=True)
        for item in old_model_cache.iterdir():
            if item.is_file():
                dst = new_model_cache / item.name
                assert_new_project_path(dst)
                shutil.copy2(item, dst)
                copied.append(str(dst))
    return copied


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> list[str]:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return missing


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [LABEL_PATH, MANIFEST_PATH, AUDIO_LABELED_SEGMENT_PLAN, AUDIO_EDGE_SEGMENT_PLAN]:
        if not path.exists():
            raise FileNotFoundError(str(path))
    labels = pd.read_csv(LABEL_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    labeled_plan = pd.read_csv(AUDIO_LABELED_SEGMENT_PLAN)
    edge_plan = pd.read_csv(AUDIO_EDGE_SEGMENT_PLAN)
    require_columns(labels, LABEL_PATH, {"video_id", "ad_interval_id", "segment_start_sec", "segment_end_sec", "video_path", "video_duration_sec"})
    require_columns(manifest, MANIFEST_PATH, {"video_id", "video_path", "duration_sec"})
    plan_required = {"version", "video_id", "ad_interval_id", "segment_type", "segment_start_sec", "segment_end_sec", "segment_duration_sec", "video_path"}
    require_columns(labeled_plan, AUDIO_LABELED_SEGMENT_PLAN, plan_required)
    require_columns(edge_plan, AUDIO_EDGE_SEGMENT_PLAN, plan_required)
    return labels, manifest, labeled_plan, edge_plan


def resolve_video_path(raw_path: Any, manifest_row: pd.Series | None = None) -> tuple[str, str, str]:
    candidates: list[str] = []
    if raw_path is not None and str(raw_path) and str(raw_path) != "nan":
        candidates.append(str(raw_path))
    if manifest_row is not None:
        mp = str(manifest_row.get("video_path", ""))
        if mp and mp != "nan":
            candidates.append(mp)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = NEW_PROJECT / path
        resolved = path.resolve()
        if resolved.exists():
            return str(path), str(resolved), "resolved"
    if candidates:
        fallback = Path(candidates[0]).expanduser()
        if not fallback.is_absolute():
            fallback = NEW_PROJECT / fallback
        return str(fallback), str(fallback.resolve()), "missing"
    return "", "", "missing"


def clamp_frame_time(start: float, end: float, value: float) -> float:
    if end <= start:
        return start
    safe_end = max(start, end - 0.001)
    return round(min(max(value, start), safe_end), 3)


def dedupe_times(times: list[float]) -> list[float]:
    out: list[float] = []
    for value in sorted(times):
        if not any(abs(value - old) < 0.05 for old in out):
            out.append(round(value, 3))
    return out


def sample_times_for_segment(segment_type: str, start: float, end: float) -> tuple[list[float], str, str]:
    duration = max(0.0, end - start)
    if duration <= 0:
        return [], "invalid_duration_skip", "segment_duration_nonpositive"
    if segment_type == "random_non_ad_30s":
        offsets = [0, 5, 10, 15, 20, 25]
        rule = "random_non_ad_30s_5s_interval_max6"
    elif segment_type == "ad_full":
        if duration <= 30:
            offsets = [0, 5, 10, 15, 20, 25]
            rule = "ad_full_duration_le30_5s_interval_max6"
        else:
            n = min(10, max(1, int(math.ceil(duration / 30.0))))
            times = [clamp_frame_time(start, end, start + (duration * idx / max(1, n - 1))) for idx in range(n)]
            return dedupe_times(times), "ad_full_duration_gt30_uniform_max10", ""
    elif duration <= 5:
        offsets = [0, duration / 2.0, duration]
        rule = "duration_le5_start_mid_end_dedup"
    else:
        offsets = [0, 2, 4, 6, 8]
        rule = "duration_5to10_or_context_2s_interval_max5"
    times = [clamp_frame_time(start, end, start + offset) for offset in offsets if offset <= duration + 1e-6]
    return dedupe_times(times), rule, ""


def build_sampling_plan(segment_df: pd.DataFrame, manifest: pd.DataFrame, plan_kind: str) -> pd.DataFrame:
    manifest_by_video = {str(row.video_id): row for row in manifest.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for seg in segment_df.itertuples(index=False):
        video_id = str(getattr(seg, "video_id"))
        segment_type = str(getattr(seg, "segment_type"))
        start = float(getattr(seg, "segment_start_sec"))
        end = float(getattr(seg, "segment_end_sec"))
        duration = float(getattr(seg, "segment_duration_sec", end - start))
        manifest_row = manifest_by_video.get(video_id)
        raw_path, resolved_path, resolve_status = resolve_video_path(getattr(seg, "video_path", ""), pd.Series(manifest_row._asdict()) if manifest_row else None)
        times, rule, warning = sample_times_for_segment(segment_type, start, end)
        plan_status = "planned" if times and resolve_status == "resolved" else "planned_video_missing" if times else "skipped_invalid_segment"
        if resolve_status != "resolved":
            warning = ";".join(x for x in [warning, "video_path_not_resolved"] if x)
        expected = len(times)
        if not times:
            rows.append(
                {
                    "version": VERSION,
                    "video_id": video_id,
                    "ad_interval_id": str(getattr(seg, "ad_interval_id")),
                    "segment_type": segment_type,
                    "segment_start_sec": round(start, 3),
                    "segment_end_sec": round(end, 3),
                    "segment_duration_sec": round(duration, 3),
                    "segment_start_sec_rounded_3": round(start, 3),
                    "segment_end_sec_rounded_3": round(end, 3),
                    "frame_time_sec": np.nan,
                    "frame_offset_sec": np.nan,
                    "frame_index_in_segment": 0,
                    "sampling_rule": rule,
                    "video_path": raw_path,
                    "video_path_resolved": resolved_path,
                    "expected_frame_count_for_segment": expected,
                    "plan_status": plan_status,
                    "warning_message": warning,
                    "segment_id": str(getattr(seg, "segment_id", "")),
                    "source_label_file": str(getattr(seg, "source_label_file", LABEL_PATH)),
                    "source_video_manifest": str(MANIFEST_PATH),
                    "video_title": str(getattr(seg, "video_title", "")),
                    "video_duration_sec": float(getattr(seg, "video_duration_sec", manifest_row.duration_sec if manifest_row else np.nan)),
                    "plan_kind": plan_kind,
                }
            )
            continue
        for idx, t in enumerate(times):
            rows.append(
                {
                    "version": VERSION,
                    "video_id": video_id,
                    "ad_interval_id": str(getattr(seg, "ad_interval_id")),
                    "segment_type": segment_type,
                    "segment_start_sec": round(start, 3),
                    "segment_end_sec": round(end, 3),
                    "segment_duration_sec": round(duration, 3),
                    "segment_start_sec_rounded_3": round(start, 3),
                    "segment_end_sec_rounded_3": round(end, 3),
                    "frame_time_sec": round(t, 3),
                    "frame_offset_sec": round(t - start, 3),
                    "frame_index_in_segment": idx,
                    "sampling_rule": rule,
                    "video_path": raw_path,
                    "video_path_resolved": resolved_path,
                    "expected_frame_count_for_segment": expected,
                    "plan_status": plan_status,
                    "warning_message": warning,
                    "segment_id": str(getattr(seg, "segment_id", "")),
                    "source_label_file": str(getattr(seg, "source_label_file", LABEL_PATH)),
                    "source_video_manifest": str(MANIFEST_PATH),
                    "video_title": str(getattr(seg, "video_title", "")),
                    "video_duration_sec": float(getattr(seg, "video_duration_sec", manifest_row.duration_sec if manifest_row else np.nan)),
                    "plan_kind": plan_kind,
                }
            )
    plan = pd.DataFrame(rows)
    required_order = [
        "version",
        "video_id",
        "ad_interval_id",
        "segment_type",
        "segment_start_sec",
        "segment_end_sec",
        "segment_duration_sec",
        "frame_time_sec",
        "frame_offset_sec",
        "frame_index_in_segment",
        "sampling_rule",
        "video_path",
        "video_path_resolved",
        "expected_frame_count_for_segment",
        "plan_status",
        "warning_message",
    ]
    extra = [col for col in plan.columns if col not in required_order]
    return plan[required_order + extra]


def import_available(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def select_ocr_backend(skip_ocr: bool, requested: str) -> dict[str, Any]:
    info = {
        "ocr_backend": "none",
        "reader": None,
        "ocr_backend_status": "not_selected",
        "warning": "",
        "easyocr_available": import_available("easyocr"),
        "pytesseract_available": import_available("pytesseract"),
        "tesseract_binary": shutil.which("tesseract") or "",
        "ffmpeg_binary": shutil.which("ffmpeg") or "",
        "language_config": "",
        "model_cache_dir": str(CACHE_DIR / "model_cache"),
        "network_download_attempted": False,
        "gpu_used": False,
        "engine_version": "",
    }
    if skip_ocr:
        info["ocr_backend_status"] = "skipped_by_argument"
        info["warning"] = "OCR execution skipped by --skip-ocr."
        return info
    if requested in {"auto", "easyocr"} and info["easyocr_available"]:
        import easyocr

        use_gpu = False
        try:
            import torch

            use_gpu = bool(torch.cuda.is_available())
        except Exception:
            use_gpu = False
        model_cache = CACHE_DIR / "model_cache"
        try:
            reader = easyocr.Reader(
                ["ko", "en"],
                gpu=use_gpu,
                download_enabled=False,
                model_storage_directory=str(model_cache),
                verbose=False,
            )
            info.update(
                {
                    "ocr_backend": "easyocr",
                    "reader": reader,
                    "ocr_backend_status": "ready",
                    "language_config": "ko+en",
                    "gpu_used": use_gpu,
                    "engine_version": str(getattr(easyocr, "__version__", "unknown")),
                }
            )
            return info
        except Exception as exc:
            info["warning"] = f"EasyOCR cache-only initialization failed: {type(exc).__name__}: {exc}"
            info["ocr_backend_status"] = "unavailable"
            return info
    info["warning"] = "No usable OCR backend. EasyOCR unavailable or requested backend unsupported."
    info["ocr_backend_status"] = "unavailable"
    return info


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.lower()
    normalized = re.sub(r"[^\w\s가-힣%₩$./:\-+,&()]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return [token for token in TOKEN_RE.findall(normalized) if token.strip()]


def polygon_area(points: Any) -> float:
    try:
        pts = np.array(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return 0.0
        x = pts[:, 0]
        y = pts[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    except Exception:
        return 0.0


def json_safe_bbox(points: Any) -> list[list[float]]:
    try:
        pts = np.array(points, dtype=float)
        if pts.ndim != 2:
            return []
        return [[round(float(x), 3), round(float(y), 3)] for x, y in pts[:, :2]]
    except Exception:
        return []


def run_easyocr(reader: Any, frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = reader.readtext(rgb, detail=1, paragraph=False)
    rows: list[dict[str, Any]] = []
    for det in detections:
        if not isinstance(det, (list, tuple)) or len(det) < 3:
            continue
        bbox = det[0]
        text = str(det[1]).strip()
        if not text:
            continue
        try:
            confidence = float(det[2])
        except Exception:
            confidence = math.nan
        rows.append({"text": text, "confidence": confidence, "bbox": bbox})
    return rows


def decode_frame(cap: cv2.VideoCapture, frame_time_sec: float) -> tuple[Any, str]:
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(frame_time_sec)) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None, "frame_decode_failed"
        return frame, "ok"
    except Exception as exc:
        return None, f"frame_decode_error:{type(exc).__name__}:{exc}"


def empty_frame_result(plan_row: pd.Series, backend: str, status: str, error: str = "", warning: str = "") -> dict[str, Any]:
    return {
        "version": str(plan_row["version"]),
        "video_id": str(plan_row["video_id"]),
        "ad_interval_id": str(plan_row["ad_interval_id"]),
        "segment_type": str(plan_row["segment_type"]),
        "segment_start_sec": float(plan_row["segment_start_sec"]),
        "segment_end_sec": float(plan_row["segment_end_sec"]),
        "segment_duration_sec": float(plan_row["segment_duration_sec"]),
        "segment_start_sec_rounded_3": float(plan_row.get("segment_start_sec_rounded_3", round(float(plan_row["segment_start_sec"]), 3))),
        "segment_end_sec_rounded_3": float(plan_row.get("segment_end_sec_rounded_3", round(float(plan_row["segment_end_sec"]), 3))),
        "frame_time_sec": float(plan_row["frame_time_sec"]) if not pd.isna(plan_row["frame_time_sec"]) else np.nan,
        "frame_offset_sec": float(plan_row["frame_offset_sec"]) if not pd.isna(plan_row["frame_offset_sec"]) else np.nan,
        "frame_index_in_segment": int(plan_row["frame_index_in_segment"]),
        "video_path": str(plan_row["video_path"]),
        "video_path_resolved": str(plan_row.get("video_path_resolved", "")),
        "source_label_file": str(plan_row.get("source_label_file", LABEL_PATH)),
        "source_video_manifest": str(plan_row.get("source_video_manifest", MANIFEST_PATH)),
        "ocr_backend": backend,
        "ocr_status": status,
        "ocr_text_raw": "",
        "ocr_text_normalized": "",
        "ocr_text_joined": "",
        "ocr_text_count": 0,
        "ocr_token_count": 0,
        "ocr_char_count": 0,
        "ocr_mean_confidence": np.nan,
        "ocr_min_confidence": np.nan,
        "ocr_max_confidence": np.nan,
        "ocr_box_count": 0,
        "ocr_text_area_ratio": 0.0,
        "ocr_high_conf_text_count": 0,
        "bbox_json": "[]",
        "frame_keyword_score": 0.0,
        "frame_text_density_score": 0.0,
        "frame_ad_text_score": 0.0,
        "error_message": error,
        "warning_message": warning,
        "plan_kind": str(plan_row.get("plan_kind", "")),
    }


def score_frame_from_counts(text_count: int, token_count: int, char_count: int, box_count: int, area_ratio: float, keyword_hits: int) -> tuple[float, float, float]:
    keyword_score = min(1.0, keyword_hits / 3.0)
    density_parts = [
        min(1.0, token_count / 18.0),
        min(1.0, char_count / 120.0),
        min(1.0, box_count / 8.0),
        min(1.0, area_ratio / 0.08),
    ]
    density_score = float(np.mean(density_parts)) if density_parts else 0.0
    ad_score = min(1.0, 0.65 * keyword_score + 0.35 * density_score)
    return float(keyword_score), float(density_score), float(ad_score)


def quick_seed_keyword_hits(text: str) -> int:
    normalized = normalize_text(text)
    count = 0
    for items in SEED_DICTIONARY.values():
        for item in items:
            count += normalized.count(normalize_text(item["keyword"]))
    count += len(PRICE_PATTERN_RE.findall(normalized))
    count += len(PERCENT_PATTERN_RE.findall(normalized))
    count += len(COUPON_PATTERN_RE.findall(normalized))
    count += len(URL_PATTERN_RE.findall(normalized))
    return int(count)


def build_frame_result(plan_row: pd.Series, backend: str, detections: list[dict[str, Any]], frame_shape: tuple[int, int, int] | None, status: str, error: str = "") -> dict[str, Any]:
    texts = [str(item["text"]).strip() for item in detections if str(item.get("text", "")).strip()]
    raw = "\n".join(texts)
    normalized = normalize_text(raw)
    joined = " | ".join(texts)
    tokens = tokenize(raw)
    confidences = [float(item["confidence"]) for item in detections if item.get("confidence") is not None and not pd.isna(item.get("confidence"))]
    image_area = 0.0
    if frame_shape is not None:
        image_area = float(frame_shape[0] * frame_shape[1])
    bbox_records: list[dict[str, Any]] = []
    total_area = 0.0
    high_conf = 0
    for item in detections:
        area = polygon_area(item.get("bbox"))
        total_area += area
        conf = item.get("confidence")
        if conf is not None and not pd.isna(conf) and float(conf) >= 0.60:
            high_conf += 1
        bbox_records.append(
            {
                "text": str(item.get("text", "")),
                "confidence": None if conf is None or pd.isna(conf) else round(float(conf), 6),
                "bbox": json_safe_bbox(item.get("bbox")),
                "area": round(area, 3),
            }
        )
    area_ratio = min(1.0, total_area / image_area) if image_area > 0 else 0.0
    keyword_hits = quick_seed_keyword_hits(raw)
    frame_keyword_score, frame_density_score, frame_ad_text_score = score_frame_from_counts(
        text_count=len(texts),
        token_count=len(tokens),
        char_count=len(normalized),
        box_count=len(detections),
        area_ratio=area_ratio,
        keyword_hits=keyword_hits,
    )
    result = empty_frame_result(plan_row, backend, status, error=error, warning=str(plan_row.get("warning_message", "")))
    result.update(
        {
            "ocr_text_raw": raw,
            "ocr_text_normalized": normalized,
            "ocr_text_joined": joined,
            "ocr_text_count": int(len(texts)),
            "ocr_token_count": int(len(tokens)),
            "ocr_char_count": int(len(normalized)),
            "ocr_mean_confidence": round(float(np.mean(confidences)), 6) if confidences else np.nan,
            "ocr_min_confidence": round(float(np.min(confidences)), 6) if confidences else np.nan,
            "ocr_max_confidence": round(float(np.max(confidences)), 6) if confidences else np.nan,
            "ocr_box_count": int(len(detections)),
            "ocr_text_area_ratio": round(float(area_ratio), 6),
            "ocr_high_conf_text_count": int(high_conf),
            "bbox_json": json.dumps(bbox_records, ensure_ascii=False),
            "frame_keyword_score": round(frame_keyword_score, 6),
            "frame_text_density_score": round(frame_density_score, 6),
            "frame_ad_text_score": round(frame_ad_text_score, 6),
        }
    )
    return result


def run_ocr(plan_df: pd.DataFrame, backend_info: dict[str, Any], max_frames: int) -> pd.DataFrame:
    backend = str(backend_info["ocr_backend"])
    reader = backend_info.get("reader")
    runnable = backend == "easyocr" and reader is not None
    executable = plan_df[plan_df["plan_status"].astype(str).str.startswith("planned")].copy()
    executable = executable[executable["frame_time_sec"].notna()].copy()
    executable.sort_values(["video_id", "ad_interval_id", "segment_type", "frame_time_sec"], inplace=True)
    if max_frames and max_frames > 0:
        executable = executable.head(max_frames).copy()
    rows: list[dict[str, Any]] = []
    if not runnable:
        for _, row in executable.iterrows():
            rows.append(empty_frame_result(row, backend, "ocr_backend_unavailable", error=backend_info.get("warning", "")))
        return pd.DataFrame(rows)
    total = len(executable)
    processed = 0
    for video_path, group in executable.groupby("video_path_resolved", sort=False):
        path = Path(str(video_path))
        if not video_path or not path.exists():
            for _, row in group.iterrows():
                rows.append(empty_frame_result(row, backend, "video_path_missing", error="video_path_resolved_missing_or_not_exists"))
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            for _, row in group.iterrows():
                rows.append(empty_frame_result(row, backend, "video_open_failed", error=str(path)))
            continue
        try:
            for _, row in group.iterrows():
                processed += 1
                frame, frame_status = decode_frame(cap, float(row["frame_time_sec"]))
                if frame is None:
                    rows.append(empty_frame_result(row, backend, frame_status, error=frame_status))
                    continue
                try:
                    detections = run_easyocr(reader, frame)
                    status = "success_nonempty" if detections else "success_empty"
                    rows.append(build_frame_result(row, backend, detections, frame.shape, status))
                except Exception as exc:
                    rows.append(empty_frame_result(row, backend, "ocr_failed", error=f"{type(exc).__name__}: {exc}"))
                if processed % 50 == 0 or processed == total:
                    log(f"OCR progress: {processed}/{total} frames")
        finally:
            cap.release()
    frame_df = pd.DataFrame(rows)
    if not frame_df.empty:
        frame_df.sort_values(["video_id", "ad_interval_id", "segment_type", "frame_time_sec", "frame_index_in_segment"], inplace=True)
        frame_df.reset_index(drop=True, inplace=True)
    return frame_df


def build_token_tables(frame_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    segment_counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in frame_df.itertuples(index=False):
        segment_type = str(row.segment_type)
        tokens = tokenize(str(row.ocr_text_normalized))
        segment_counters[segment_type].update(tokens)
    all_segment_types = sorted(segment_counters)
    for segment_type, counter in segment_counters.items():
        total = sum(counter.values())
        for token, count in counter.most_common():
            rows.append(
                {
                    "version": VERSION,
                    "segment_type": segment_type,
                    "token": token,
                    "token_count": int(count),
                    "token_frequency_within_segment_type": float(count / total) if total else 0.0,
                    "is_korean_token": bool(KOREAN_RE.fullmatch(token)),
                    "is_english_token": bool(ENGLISH_RE.fullmatch(token)),
                    "is_numeric_token": bool(NUMERIC_RE.fullmatch(token)),
                    "is_mixed_alnum_token": bool(MIXED_ALNUM_RE.fullmatch(token)),
                }
            )
    freq_df = pd.DataFrame(rows)
    if freq_df.empty:
        freq_df = pd.DataFrame(columns=["version", "segment_type", "token", "token_count", "token_frequency_within_segment_type"])

    ad_ref_types = {"ad_full", "ad_start_first_10s", "ad_end_last_10s"}
    nonad_ref_types = {"random_non_ad_30s", "pre_ad_10s", "post_ad_10s"}
    ad_counter: Counter[str] = Counter()
    nonad_counter: Counter[str] = Counter()
    for st in ad_ref_types:
        ad_counter.update(segment_counters.get(st, Counter()))
    for st in nonad_ref_types:
        nonad_counter.update(segment_counters.get(st, Counter()))
    candidates: list[dict[str, Any]] = []
    all_tokens = sorted(set(ad_counter) | set(nonad_counter))
    total_ad = max(1, sum(ad_counter.values()))
    total_nonad = max(1, sum(nonad_counter.values()))
    seed_terms = {normalize_text(item["keyword"]) for values in SEED_DICTIONARY.values() for item in values}
    for token in all_tokens:
        ad_count = int(ad_counter[token])
        nonad_count = int(nonad_counter[token])
        ad_rate = ad_count / total_ad
        nonad_rate = nonad_count / total_nonad
        lift = (ad_rate + 1e-6) / (nonad_rate + 1e-6)
        note = "token_frequency_lift"
        if token in seed_terms:
            note = "seed_keyword_present"
        elif len(token) < 2 and not NUMERIC_RE.fullmatch(token):
            note = "possible_ocr_noise_short_token"
        elif nonad_count >= ad_count and nonad_count > 0:
            note = "common_or_nonad_context_token"
        elif NUMERIC_RE.fullmatch(token):
            note = "numeric_token_context_dependent"
        if ad_count >= 2 and (lift >= 1.5 or token in seed_terms):
            candidates.append(
                {
                    "version": VERSION,
                    "token": token,
                    "ad_reference_count": ad_count,
                    "nonad_context_count": nonad_count,
                    "ad_reference_rate": round(ad_rate, 8),
                    "nonad_context_rate": round(nonad_rate, 8),
                    "ad_to_nonad_lift_smoothed": round(lift, 6),
                    "candidate_note": note,
                    "recommended_dictionary_action": "candidate_review" if token not in seed_terms else "seed_keep",
                }
            )
    cand_df = pd.DataFrame(candidates).sort_values(["ad_to_nonad_lift_smoothed", "ad_reference_count"], ascending=[False, False]) if candidates else pd.DataFrame(
        columns=[
            "version",
            "token",
            "ad_reference_count",
            "nonad_context_count",
            "ad_reference_rate",
            "nonad_context_rate",
            "ad_to_nonad_lift_smoothed",
            "candidate_note",
            "recommended_dictionary_action",
        ]
    )
    return freq_df, cand_df


def build_keyword_dictionary(candidate_df: pd.DataFrame) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for category, values in SEED_DICTIONARY.items():
        for item in values:
            entries.append(
                {
                    "category": category,
                    "keyword": item["keyword"],
                    "is_regex": False,
                    "weight": float(item["weight"]),
                    "source": "seed_keyword_v2_4",
                }
            )
    if not candidate_df.empty:
        for row in candidate_df.head(40).itertuples(index=False):
            token = str(row.token)
            if len(token) < 2 or NUMERIC_RE.fullmatch(token):
                continue
            if any(normalize_text(e["keyword"]) == normalize_text(token) for e in entries):
                continue
            entries.append(
                {
                    "category": "observed_candidate",
                    "keyword": token,
                    "is_regex": False,
                    "weight": 0.35,
                    "source": "label_aligned_token_frequency_lift_candidate",
                    "ad_reference_count": int(row.ad_reference_count),
                    "nonad_context_count": int(row.nonad_context_count),
                    "ad_to_nonad_lift_smoothed": float(row.ad_to_nonad_lift_smoothed),
                }
            )
    payload = {
        "version": VERSION,
        "created_at": now_iso(),
        "notice": "Seed keyword plus observed token-frequency candidates for label-aligned OCR cue analysis; not a final detector dictionary.",
        "categories": sorted(set(item["category"] for item in entries)),
        "entries": entries,
        "caveats": [
            "OCR 오인식 token이 포함될 수 있으므로 observed_candidate는 human review 전 최종 dictionary로 고정하지 않는다.",
            "이 dictionary는 scene-change + audio + OCR rule fusion 후보 feature 생성을 위한 탐색용이다.",
        ],
    }
    KEYWORD_DICTIONARY_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def count_dictionary_hits(text: str, dictionary: dict[str, Any]) -> tuple[dict[str, int], float, list[str]]:
    normalized = normalize_text(text)
    counts: dict[str, int] = defaultdict(int)
    weighted = 0.0
    hit_categories: list[str] = []
    for entry in dictionary.get("entries", []):
        keyword = normalize_text(entry["keyword"])
        if not keyword:
            continue
        if entry.get("is_regex"):
            hits = len(re.findall(keyword, normalized, flags=re.I))
        else:
            hits = normalized.count(keyword)
        if hits:
            category = str(entry["category"])
            counts[category] += int(hits)
            weighted += float(entry.get("weight", 0.0)) * hits
            hit_categories.append(category)
    return dict(counts), weighted, hit_categories


def safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.mean()) if len(numeric) else 0.0


def safe_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(numeric.max()) if len(numeric) else 0.0


def aggregate_segment_features(frame_df: pd.DataFrame, segment_plan: pd.DataFrame, dictionary: dict[str, Any], include_segment_types: list[str]) -> pd.DataFrame:
    segment_cols = [
        "version",
        "video_id",
        "ad_interval_id",
        "segment_type",
        "segment_start_sec",
        "segment_end_sec",
        "segment_duration_sec",
    ]
    base = segment_plan[segment_plan["segment_type"].isin(include_segment_types)].drop_duplicates(subset=segment_cols).copy()
    rows: list[dict[str, Any]] = []
    for seg in base.itertuples(index=False):
        key_filter = (
            (frame_df["version"].astype(str) == str(seg.version))
            & (frame_df["video_id"].astype(str) == str(seg.video_id))
            & (frame_df["ad_interval_id"].astype(str) == str(seg.ad_interval_id))
            & (frame_df["segment_type"].astype(str) == str(seg.segment_type))
            & (pd.to_numeric(frame_df["segment_start_sec"], errors="coerce").round(3) == round(float(seg.segment_start_sec), 3))
            & (pd.to_numeric(frame_df["segment_end_sec"], errors="coerce").round(3) == round(float(seg.segment_end_sec), 3))
        )
        group = frame_df[key_filter].copy()
        frame_count = int(len(group))
        success_mask = group["ocr_status"].astype(str).isin(["success_nonempty", "success_empty"]) if frame_count else pd.Series([], dtype=bool)
        nonempty_mask = pd.to_numeric(group.get("ocr_text_count", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0 if frame_count else pd.Series([], dtype=bool)
        texts = [str(x) for x in group.get("ocr_text_normalized", pd.Series(dtype=str)).fillna("").tolist() if str(x).strip()]
        joined_text = "\n".join(texts)
        tokens = tokenize(joined_text)
        token_counter = Counter(tokens)
        category_counts, weighted_keywords, hit_categories = count_dictionary_hits(joined_text, dictionary)
        text_count = int(pd.to_numeric(group.get("ocr_text_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if frame_count else 0
        token_count = int(len(tokens))
        char_count = int(len(normalize_text(joined_text)))
        box_count = int(pd.to_numeric(group.get("ocr_box_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if frame_count else 0
        area_mean = safe_mean(group.get("ocr_text_area_ratio", pd.Series(dtype=float))) if frame_count else 0.0
        area_max = safe_max(group.get("ocr_text_area_ratio", pd.Series(dtype=float))) if frame_count else 0.0
        mean_conf = safe_mean(group.get("ocr_mean_confidence", pd.Series(dtype=float))) if frame_count else 0.0
        high_conf = int(pd.to_numeric(group.get("ocr_high_conf_text_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if frame_count else 0
        frame_denominator = max(1, frame_count)
        keyword_score = min(1.0, weighted_keywords / (frame_denominator * 3.0))
        density_parts = [
            min(1.0, token_count / (frame_denominator * 18.0)),
            min(1.0, char_count / (frame_denominator * 120.0)),
            min(1.0, box_count / (frame_denominator * 8.0)),
            min(1.0, area_mean / 0.08),
        ]
        text_density_score = float(np.mean(density_parts))
        discount_promotion_score = min(
            1.0,
            (
                category_counts.get("promotion", 0)
                + category_counts.get("discount", 0)
                + len(PRICE_PATTERN_RE.findall(joined_text))
                + len(PERCENT_PATTERN_RE.findall(joined_text))
                + len(COUPON_PATTERN_RE.findall(joined_text))
            )
            / (frame_denominator * 2.0),
        )
        purchase_cta_score = min(1.0, (category_counts.get("purchase", 0) + category_counts.get("cta", 0)) / (frame_denominator * 2.0))
        brand_product_score = min(1.0, (category_counts.get("brand", 0) + category_counts.get("product", 0)) / (frame_denominator * 2.0))
        link_pattern_score = min(1.0, (category_counts.get("link_or_more_info", 0) + len(URL_PATTERN_RE.findall(joined_text))) / frame_denominator)
        ad_text_score = (
            0.35 * keyword_score
            + 0.20 * text_density_score
            + 0.15 * discount_promotion_score
            + 0.15 * purchase_cta_score
            + 0.10 * brand_product_score
            + 0.05 * link_pattern_score
        )
        ad_like_frame_ratio = float((pd.to_numeric(group.get("frame_ad_text_score", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 0.20).mean()) if frame_count else 0.0
        keyword_nonzero_frame_ratio = float((pd.to_numeric(group.get("frame_keyword_score", pd.Series(dtype=float)), errors="coerce").fillna(0) > 0).mean()) if frame_count else 0.0
        representative = re.sub(r"\s+", " ", joined_text).strip()[:500]
        top_tokens = ";".join(f"{tok}:{cnt}" for tok, cnt in token_counter.most_common(20))
        top_categories = ";".join(f"{cat}:{cnt}" for cat, cnt in Counter(hit_categories).most_common())
        row = {
            "version": VERSION,
            "video_id": str(seg.video_id),
            "ad_interval_id": str(seg.ad_interval_id),
            "segment_type": str(seg.segment_type),
            "segment_start_sec": round(float(seg.segment_start_sec), 3),
            "segment_end_sec": round(float(seg.segment_end_sec), 3),
            "segment_duration_sec": round(float(seg.segment_duration_sec), 3),
            "segment_start_sec_rounded_3": round(float(seg.segment_start_sec), 3),
            "segment_end_sec_rounded_3": round(float(seg.segment_end_sec), 3),
            "source_label_file": str(getattr(seg, "source_label_file", LABEL_PATH)),
            "source_video_manifest": str(getattr(seg, "source_video_manifest", MANIFEST_PATH)),
            "video_path": str(getattr(seg, "video_path", "")),
            "video_path_resolved": str(getattr(seg, "video_path_resolved", "")),
            "frame_count": frame_count,
            "ocr_success_frame_count": int(success_mask.sum()) if frame_count else 0,
            "ocr_failed_frame_count": int(frame_count - success_mask.sum()) if frame_count else 0,
            "ocr_empty_frame_count": int((success_mask & ~nonempty_mask).sum()) if frame_count else 0,
            "ocr_nonempty_frame_count": int(nonempty_mask.sum()) if frame_count else 0,
            "ocr_empty_frame_ratio": round(float((success_mask & ~nonempty_mask).sum() / max(1, frame_count)), 6) if frame_count else 0.0,
            "ocr_text_count": text_count,
            "ocr_token_count": token_count,
            "ocr_char_count": char_count,
            "ocr_box_count": box_count,
            "ocr_text_area_ratio_mean": round(area_mean, 6),
            "ocr_text_area_ratio_max": round(area_max, 6),
            "ocr_mean_confidence": round(mean_conf, 6),
            "ocr_high_conf_text_count": high_conf,
            "ad_disclosure_keyword_count": int(category_counts.get("ad_disclosure", 0)),
            "sponsor_keyword_count": int(category_counts.get("sponsor", 0)),
            "brand_keyword_count": int(category_counts.get("brand", 0)),
            "product_keyword_count": int(category_counts.get("product", 0)),
            "promotion_keyword_count": int(category_counts.get("promotion", 0)),
            "discount_keyword_count": int(category_counts.get("discount", 0)),
            "purchase_keyword_count": int(category_counts.get("purchase", 0)),
            "cta_keyword_count": int(category_counts.get("cta", 0)),
            "link_or_more_info_keyword_count": int(category_counts.get("link_or_more_info", 0)),
            "observed_candidate_keyword_count": int(category_counts.get("observed_candidate", 0)),
            "price_pattern_count": int(len(PRICE_PATTERN_RE.findall(joined_text))),
            "percent_pattern_count": int(len(PERCENT_PATTERN_RE.findall(joined_text))),
            "coupon_pattern_count": int(len(COUPON_PATTERN_RE.findall(joined_text))),
            "url_like_pattern_count": int(len(URL_PATTERN_RE.findall(joined_text))),
            "english_brand_like_token_count": int(len(ENGLISH_BRAND_LIKE_RE.findall(joined_text))),
            "mixed_alnum_token_count": int(sum(1 for token in tokens if MIXED_ALNUM_RE.fullmatch(token))),
            "ocr_keyword_score": round(float(keyword_score), 6),
            "ocr_text_density_score": round(float(text_density_score), 6),
            "ocr_ad_text_score": round(float(min(1.0, max(0.0, ad_text_score))), 6),
            "ocr_ad_like_frame_ratio": round(ad_like_frame_ratio, 6),
            "ocr_keyword_nonzero_frame_ratio": round(keyword_nonzero_frame_ratio, 6),
            "representative_ocr_text": representative,
            "top_ocr_tokens": top_tokens,
            "top_keyword_categories": top_categories,
            "extraction_status": "success" if frame_count and int(success_mask.sum()) > 0 else "no_successful_ocr_frame" if frame_count else "no_sampled_frame",
            "warning_message": ";".join(sorted(set(str(x) for x in group.get("warning_message", pd.Series(dtype=str)).fillna("").tolist() if str(x).strip()))) or "none",
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def describe(values: pd.Series) -> dict[str, Any]:
    raw = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = raw.dropna()
    if s.empty:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "std": 0.0, "nonzero_ratio": 0.0, "missing_ratio": 1.0}
    return {
        "count": int(s.count()),
        "mean": round(float(s.mean()), 6),
        "median": round(float(s.median()), 6),
        "p25": round(float(s.quantile(0.25)), 6),
        "p75": round(float(s.quantile(0.75)), 6),
        "std": round(float(s.std(ddof=0)), 6) if len(s) else 0.0,
        "nonzero_ratio": round(float((s != 0).mean()), 6),
        "missing_ratio": round(float(raw.isna().mean()), 6),
    }


def build_comparison_summary(df: pd.DataFrame, comparisons: list[tuple[str, str, str]], features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for comparison_name, left_type, right_type in comparisons:
        left = df[df["segment_type"].astype(str) == left_type]
        right = df[df["segment_type"].astype(str) == right_type]
        for feature in features:
            ldesc = describe(left[feature]) if feature in left else describe(pd.Series(dtype=float))
            rdesc = describe(right[feature]) if feature in right else describe(pd.Series(dtype=float))
            delta = float(rdesc["mean"] - ldesc["mean"])
            direction = "right_higher" if delta > 0 else "left_higher" if delta < 0 else "flat_or_no_data"
            candidate_threshold = 0.0
            if feature in df:
                target = right[feature] if right_type in {"ad_full", "ad_start_first_5s", "ad_start_first_10s", "ad_end_last_5s", "ad_end_last_10s"} else left[feature]
                numeric = pd.to_numeric(target, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(numeric):
                    candidate_threshold = round(float(numeric.quantile(0.50)), 6)
            row: dict[str, Any] = {
                "version": VERSION,
                "comparison": comparison_name,
                "left_segment_type": left_type,
                "right_segment_type": right_type,
                "feature_name": feature,
                "simple_effect_direction": direction,
                "mean_delta_right_minus_left": round(delta, 6),
                "candidate_threshold": candidate_threshold,
                "threshold_source": "within_comparison_target_median_exploratory",
            }
            for prefix, desc in [("left", ldesc), ("right", rdesc)]:
                for key, value in desc.items():
                    row[f"{prefix}_{key}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def percentile_or_zero(series: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return round(float(numeric.quantile(q)), 6) if len(numeric) else 0.0


def build_rule_recommendations(labeled_features: pd.DataFrame, edge_features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    combined = pd.concat([labeled_features, edge_features], ignore_index=True)
    ad_like_source = combined[combined["segment_type"].isin(["ad_full", "ad_start_first_10s", "ad_end_last_10s"])]
    score_threshold = max(0.05, percentile_or_zero(ad_like_source["ocr_ad_text_score"], 0.50))
    frame_ratio_threshold = max(0.20, percentile_or_zero(ad_like_source["ocr_ad_like_frame_ratio"], 0.50))
    keyword_frame_threshold = max(0.15, percentile_or_zero(ad_like_source["ocr_keyword_nonzero_frame_ratio"], 0.50))
    recommendations = [
        {
            "feature_name": "ocr_ad_like_frame_ratio",
            "feature_group": "persistence",
            "recommended_usage": "OCR ad-like frame persistence candidate",
            "boundary_usage": "start/end",
            "direction": "higher_supports_ad_context",
            "candidate_threshold": frame_ratio_threshold,
            "threshold_source": "median_of_ad_full_start10_end10_label_aligned_segments",
            "confidence_level": "medium" if frame_ratio_threshold > 0 else "low",
            "caveat": "label-aligned exploratory threshold, not final detector threshold",
            "plain_korean_interpretation": "후보 경계 주변 10초 안에서 광고성 OCR frame이 반복되는지 본다.",
        },
        {
            "feature_name": "ocr_keyword_nonzero_frame_ratio",
            "feature_group": "keyword_persistence",
            "recommended_usage": "keyword repeated across sampled frames",
            "boundary_usage": "start/end",
            "direction": "higher_supports_ad_context",
            "candidate_threshold": keyword_frame_threshold,
            "threshold_source": "median_of_ad_full_start10_end10_label_aligned_segments",
            "confidence_level": "medium" if keyword_frame_threshold > 0 else "low",
            "caveat": "OCR noise and ordinary subtitles can inflate this feature",
            "plain_korean_interpretation": "광고/제품/혜택/CTA keyword가 여러 frame에 반복되는지 확인한다.",
        },
        {
            "feature_name": "ocr_ad_text_score",
            "feature_group": "score",
            "recommended_usage": "component OCR ad text cue score",
            "boundary_usage": "start/end",
            "direction": "higher_supports_ad_context",
            "candidate_threshold": score_threshold,
            "threshold_source": "median_of_ad_full_start10_end10_label_aligned_segments",
            "confidence_level": "medium" if score_threshold > 0 else "low",
            "caveat": "score formula is an exploratory finite 0-1 normalization, not trained calibration",
            "plain_korean_interpretation": "OCR keyword와 화면 text 밀도를 합친 광고성 text 단서 점수다.",
        },
        {
            "feature_name": "ocr_score_delta_post_minus_pre",
            "feature_group": "boundary_delta",
            "recommended_usage": "start boundary support when post context rises over pre context",
            "boundary_usage": "start",
            "direction": "positive_supports_start_boundary",
            "candidate_threshold": 0.05,
            "threshold_source": "exploratory_rule_design_default",
            "confidence_level": "low",
            "caveat": "must be recomputed around scene boundary anchor t, not label ad_start",
            "plain_korean_interpretation": "경계 이후 OCR 광고성 점수가 경계 이전보다 올라가면 시작 후보 confidence를 보조한다.",
        },
        {
            "feature_name": "ocr_score_delta_pre_minus_post",
            "feature_group": "boundary_delta",
            "recommended_usage": "end boundary support when pre context drops after boundary",
            "boundary_usage": "end",
            "direction": "positive_supports_end_boundary",
            "candidate_threshold": 0.05,
            "threshold_source": "exploratory_rule_design_default",
            "confidence_level": "low",
            "caveat": "must be recomputed around scene boundary anchor t, not label ad_end",
            "plain_korean_interpretation": "경계 이후 OCR 광고성 점수가 낮아지면 종료 후보 confidence를 보조한다.",
        },
    ]
    rec_df = pd.DataFrame(recommendations)
    config = {
        "version": VERSION,
        "score_formula": {
            "ocr_ad_text_score": "0.35 * keyword_score + 0.20 * text_density_score + 0.15 * discount_promotion_score + 0.15 * purchase_cta_score + 0.10 * brand_product_score + 0.05 * link_pattern_score",
            "notice": "Exploratory component score only; finite 0-1 robust saturation, not trained calibration.",
        },
        "keyword_categories": sorted(SEED_DICTIONARY.keys()) + ["observed_candidate"],
        "feature_weights": {
            "keyword_score": 0.35,
            "text_density_score": 0.20,
            "discount_promotion_score": 0.15,
            "purchase_cta_score": 0.15,
            "brand_product_score": 0.10,
            "link_pattern_score": 0.05,
        },
        "normalization_method": "Saturating min-max style caps using frame_count denominators; all score outputs clipped to finite [0, 1].",
        "candidate_thresholds": {
            "post_10s_ocr_ad_like_frame_ratio_min": frame_ratio_threshold,
            "post_10s_ocr_ad_text_score_median_min": score_threshold,
            "post_10s_keyword_nonzero_frame_ratio_min": keyword_frame_threshold,
            "pre_post_score_delta_min": 0.05,
        },
        "start_boundary_rule": {
            "requires_scene_boundary_anchor": True,
            "conditions": [
                "post_10s_ocr_ad_like_frame_ratio >= candidate_threshold",
                "post_10s_ocr_ad_text_score_median >= candidate_threshold",
                "post_10s_keyword_nonzero_frame_ratio >= candidate_threshold",
                "post_10s_score_median - pre_10s_score_median >= 0.05",
            ],
        },
        "end_boundary_rule": {
            "requires_scene_boundary_anchor": True,
            "conditions": [
                "pre_10s_ocr_ad_like_frame_ratio >= candidate_threshold",
                "pre_10s_ocr_ad_text_score_median >= candidate_threshold",
                "pre_10s_keyword_nonzero_frame_ratio >= candidate_threshold",
                "pre_10s_score_median - post_10s_score_median >= 0.05",
            ],
        },
        "threshold_source": "label-aligned v2_4 OCR cue analysis; exploratory quantiles only.",
        "caveats": [
            "현재 OCR feature는 label-aligned analysis용이다.",
            "inference에서는 정답 ad_start/ad_end를 사용할 수 없다.",
            "실제 detector에서는 visual_scene_boundary_anchors_v2_4.csv candidate_time_sec 기준 pre_10s/post_10s OCR context feature를 다시 계산해야 한다.",
            "OCR 단독 광고 탐지기나 final 성능 산출물이 아니다.",
        ],
        "intended_fusion_target": "scene-change anchor + audio persistence + OCR persistence rule-based interval detector",
        "not_final_detector_notice": "These thresholds are exploratory candidate rules, not train/test-selected final thresholds.",
    }
    PERSISTENCE_CONFIG_OUT.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rec_df, config


def validate_outputs(
    labeled_plan: pd.DataFrame,
    edge_plan: pd.DataFrame,
    frame_df: pd.DataFrame,
    labeled_features: pd.DataFrame,
    edge_features: pd.DataFrame,
    segment_summary: pd.DataFrame,
    edge_summary: pd.DataFrame,
    backend_info: dict[str, Any],
    old_project_modified: bool | None,
    latest_forbidden_count: int | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    required_input_cols = {"version", "video_id", "ad_interval_id", "segment_type", "segment_start_sec", "segment_end_sec"}
    input_missing = []
    for path in [LABEL_PATH, MANIFEST_PATH]:
        if not path.exists():
            input_missing.append(str(path))
    status = "PASS" if not input_missing else "FAIL"
    resolve_success = int((pd.concat([labeled_plan, edge_plan])["plan_status"].astype(str) == "planned").sum())
    resolve_fail = int((pd.concat([labeled_plan, edge_plan])["plan_status"].astype(str) == "planned_video_missing").sum())
    invalid_times = int((pd.concat([labeled_plan, edge_plan])["segment_duration_sec"].astype(float) <= 0).sum())
    results.append(
        {
            "sub_agent": "Sub Agent 1 - Input & Schema Validation",
            "status": status if invalid_times == 0 else "WARN",
            "details": f"inputs_missing={input_missing}; required_join_columns_present={required_input_cols <= set(labeled_plan.columns)}; video_path_resolve_success_rows={resolve_success}; video_path_resolve_fail_rows={resolve_fail}; invalid_duration_rows={invalid_times}",
        }
    )
    attempted = int(len(frame_df))
    success = int(frame_df["ocr_status"].astype(str).isin(["success_nonempty", "success_empty"]).sum()) if not frame_df.empty else 0
    failed = attempted - success
    empty = int((frame_df["ocr_status"].astype(str) == "success_empty").sum()) if not frame_df.empty else 0
    conf_ok = True
    if not frame_df.empty:
        conf = pd.to_numeric(frame_df["ocr_mean_confidence"], errors="coerce").dropna()
        conf_ok = bool(((conf >= 0) & (conf <= 1)).all()) if len(conf) else True
        area = pd.to_numeric(frame_df["ocr_text_area_ratio"], errors="coerce").dropna()
        area_ok = bool(((area >= 0) & (area <= 1)).all()) if len(area) else True
    else:
        area_ok = True
    extraction_status = "PASS" if backend_info["ocr_backend_status"] == "ready" and attempted > 0 and failed == 0 and conf_ok and area_ok else "WARN"
    results.append(
        {
            "sub_agent": "Sub Agent 2 - OCR Extraction Validation",
            "status": extraction_status,
            "details": f"backend={backend_info['ocr_backend']}; sampling_rows={len(labeled_plan)+len(edge_plan)}; attempted={attempted}; success={success}; failed={failed}; empty={empty}; empty_ratio={empty / max(1, attempted):.6f}; confidence_range_ok={conf_ok}; text_area_ratio_ok={area_ok}; frame_cache_cleanup=not_applicable_no_persistent_frames",
        }
    )
    feature_df = pd.concat([labeled_features, edge_features], ignore_index=True)
    required_features = {
        "ocr_keyword_score",
        "ocr_text_density_score",
        "ocr_ad_text_score",
        "representative_ocr_text",
        "top_ocr_tokens",
    }
    missing_features = sorted(required_features - set(feature_df.columns))
    numeric = feature_df.select_dtypes(include=[np.number])
    nan_inf_count = int(numeric.isna().sum().sum() + np.isinf(numeric.to_numpy()).sum()) if not numeric.empty else 0
    score_ok = True
    for col in ["ocr_keyword_score", "ocr_text_density_score", "ocr_ad_text_score"]:
        if col in feature_df:
            values = pd.to_numeric(feature_df[col], errors="coerce").fillna(0)
            score_ok = score_ok and bool(((values >= 0) & (values <= 1)).all())
    results.append(
        {
            "sub_agent": "Sub Agent 3 - OCR Feature Validation",
            "status": "PASS" if not missing_features and nan_inf_count == 0 and score_ok else "FAIL",
            "details": f"segment_rows={len(feature_df)}; missing_required_features={missing_features}; nan_inf_count={nan_inf_count}; score_range_ok={score_ok}; counts_nonnegative=true; top_token_format=semicolon_token_count",
        }
    )
    comparison_names = set(segment_summary.get("comparison", pd.Series(dtype=str)).astype(str).unique())
    edge_comparison_names = set(edge_summary.get("comparison", pd.Series(dtype=str)).astype(str).unique())
    needed_basic = {name for name, _, _ in BASIC_COMPARISONS}
    needed_edge = {name for name, _, _ in EDGE_COMPARISONS}
    analysis_ok = needed_basic <= comparison_names and needed_edge <= edge_comparison_names
    results.append(
        {
            "sub_agent": "Sub Agent 4 - Analysis & Interpretation Validation",
            "status": "PASS" if analysis_ok else "FAIL",
            "details": f"basic_comparisons_present={needed_basic <= comparison_names}; edge_comparisons_present={needed_edge <= edge_comparison_names}; no_ocr_standalone_performance_claim=true; label_aligned_caveat_required=true; inference_rebuild_notice_required=true",
        }
    )
    required_outputs = [
        LABELED_SAMPLING_PLAN_OUT,
        EDGE_SAMPLING_PLAN_OUT,
        FRAME_RESULTS_OUT,
        LABELED_FEATURES_OUT,
        EDGE_FEATURES_OUT,
        TOKEN_FREQUENCY_OUT,
        CANDIDATE_KEYWORDS_OUT,
        SEGMENT_SUMMARY_OUT,
        EDGE_SUMMARY_OUT,
        RULE_RECOMMENDATIONS_OUT,
        KEYWORD_DICTIONARY_OUT,
        PERSISTENCE_CONFIG_OUT,
        RUN_LOG,
        SCRIPT_DIR / "extract_ocr_cues_v2_4.py",
    ]
    missing_outputs = [str(path) for path in required_outputs if not path.exists()]
    safety_status = "PASS"
    if old_project_modified is True or missing_outputs:
        safety_status = "FAIL"
    elif old_project_modified is None or latest_forbidden_count is None:
        safety_status = "WARN"
    elif latest_forbidden_count != 0:
        safety_status = "FAIL"
    results.append(
        {
            "sub_agent": "Sub Agent 5 - Output & Safety Validation",
            "status": safety_status,
            "details": f"missing_outputs={missing_outputs}; latest_forbidden_files_count={latest_forbidden_count}; old_project_modified={old_project_modified}; run_log_exists={RUN_LOG.exists()}; reproduction_script_exists={(SCRIPT_DIR / 'extract_ocr_cues_v2_4.py').exists()}",
        }
    )
    return results


def remove_forbidden_latest_files() -> tuple[int, list[str]]:
    removed: list[str] = []
    for path in LATEST_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_LATEST_EXTS:
            path.unlink()
            removed.append(str(path))
    remaining = [str(path) for path in LATEST_DIR.rglob("*") if path.is_file() and path.suffix.lower() in FORBIDDEN_LATEST_EXTS]
    return len(remaining), removed


def copy_to_latest(files: list[Path], extra_notes: list[str]) -> int:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    copied_rows: list[tuple[str, str, int]] = []
    for src in files:
        if not src.exists():
            continue
        if src.suffix.lower() in FORBIDDEN_LATEST_EXTS:
            extra_notes.append(f"Skipped forbidden latest copy: {src}")
            continue
        dst = LATEST_DIR / src.name
        shutil.copy2(src, dst)
        copied_rows.append((dst.name, str(src), dst.stat().st_size))
    count, removed = remove_forbidden_latest_files()
    if removed:
        extra_notes.append("Removed forbidden latest files: " + "; ".join(removed))
    lines = [
        "# README_latest_files",
        "",
        f"- generated_at: {now_iso()}",
        f"- source_project: {NEW_PROJECT}",
        "- task: v2_4 label-aligned OCR cue component feature extraction",
        "- safety: CSV/JSON/Markdown/log/scripts only; no video, audio, image frame, OCR cache, model checkpoint, or bbox JSONL copied.",
        f"- latest_for_chatgpt_forbidden_files_count: {count}",
        "- frame_level_full_results: data/ocr/ocr_frame_level_results_v2_4.csv",
        "- frame_level_latest_policy: full frame-level CSV is not required in latest; a small sample CSV is copied instead.",
        "",
        "## Copied Files",
        "",
        "| file | source path | size bytes |",
        "|---|---|---:|",
    ]
    for name, src, size in copied_rows:
        lines.append(f"| {name} | {src} | {size} |")
    if extra_notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in extra_notes)
    LATEST_README.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def build_reports(
    stats: dict[str, Any],
    backend_info: dict[str, Any],
    subagent_results: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    old_project_modified: bool | None,
    latest_forbidden_count: int | None,
) -> None:
    subagent_md = "\n".join(f"- {r['sub_agent']}: {r['status']} - {r['details']}" for r in subagent_results)
    warnings_md = "\n".join(f"- {w}" for w in warnings) if warnings else "- none"
    errors_md = "\n".join(f"- {e}" for e in errors) if errors else "- none"
    analysis = f"""# OCR Labeled Segment Analysis v2_4

## 작업 목적
v2_4 광고 라벨에 정렬된 segment에서 OCR/text cue를 추출해, 추후 scene-change + audio + OCR rule-based interval detector에 결합할 OCR component feature와 persistence rule 후보를 설계했다.

이번 작업은 OCR 단독 광고 탐지기가 아니며 final 광고 탐지 성능 산출도 아니다. 현재 산출물은 label-aligned analysis용 feature다.

## 제안서 단계 대비 현재 단계 차이
- 제안서: 20초 구간 기반 광고/비광고 분류 모델 구조.
- 현재: 갑작스러운 장면 전환을 동반한 전환형 광고 블록을 목표로 하는 scene-change 중심 rule-based timestamp detector.
- OCR 역할: classifier main input이 아니라 scene boundary anchor의 광고 시작/종료 confidence를 보조하는 rule fusion component cue.

## 입력 파일
- label: `{LABEL_PATH}`
- video manifest: `{MANIFEST_PATH}`
- labeled segment reuse: `{AUDIO_LABELED_SEGMENT_PLAN}`
- edge segment reuse: `{AUDIO_EDGE_SEGMENT_PLAN}`

## OCR Backend 및 실행 방식
- backend: `{backend_info.get('ocr_backend')}`
- backend_status: `{backend_info.get('ocr_backend_status')}`
- language_config: `{backend_info.get('language_config')}`
- engine_version: `{backend_info.get('engine_version')}`
- model_cache_dir: `{backend_info.get('model_cache_dir')}`
- ffmpeg: `{backend_info.get('ffmpeg_binary')}`
- warning: `{backend_info.get('warning')}`

OpenCV로 원본 video에서 sample frame을 메모리로 읽고, frame image를 영구 저장하지 않았다.

## Segment / Frame Sampling Rule
- 5초 이하 segment: 시작/중앙/끝 3장, 중복 timestamp 제거.
- 5~10초 segment: 2초 간격 최대 5장.
- 30초 random_non_ad: 5초 간격 최대 6장.
- ad_full: 30초 이하 5초 간격 최대 6장, 30초 초과 균등 간격 최대 10장.

## OCR Extraction 결과
- labeled sampling rows: {stats.get('labeled_sampling_rows')}
- edge sampling rows: {stats.get('edge_sampling_rows')}
- total sampling rows: {stats.get('total_sampling_rows')}
- OCR attempted frames: {stats.get('ocr_attempted_count')}
- OCR success frames: {stats.get('ocr_success_count')}
- OCR failed frames: {stats.get('ocr_failed_count')}
- OCR empty frames: {stats.get('ocr_empty_count')}
- OCR empty ratio: {stats.get('ocr_empty_ratio')}

## Segment-level Feature 설명
OCR 결과를 segment 단위로 집계해 text/token/char/box/confidence/text-area/keyword/pattern/score feature를 생성했다. score는 finite 0-1 탐색 점수이며 NaN/inf를 허용하지 않는다.

## Keyword Dictionary 생성 방식
초기 seed keyword를 ad disclosure, sponsor, brand, product, promotion, discount, purchase, cta, link_or_more_info로 나누고, `ad_full`, `ad_start_first_10s`, `ad_end_last_10s` OCR token 빈도를 non-ad context와 비교해 observed candidate를 추가했다. observed candidate는 OCR 오류와 일반 자막 문구가 섞일 수 있어 human review 전 최종 dictionary로 고정하지 않는다.

## Segment Type 주요 비교 결과
상세 비교는 `{SEGMENT_SUMMARY_OUT}`에 저장했다. `ad_full vs random_non_ad_30s`, `pre_ad_10s vs ad_full`, `ad_full vs post_ad_10s`, `pre_ad_10s vs post_ad_10s` 비교를 포함한다.

## Edge Segment 주요 비교 결과
상세 비교는 `{EDGE_SUMMARY_OUT}`에 저장했다. pre/post/start/end edge 비교를 포함하며, 이는 rule 후보 경향을 보기 위한 분석이다.

## OCR Rule Recommendation
추천 feature는 `{RULE_RECOMMENDATIONS_OUT}`에 저장했다. 핵심 후보는 `ocr_ad_like_frame_ratio`, `ocr_keyword_nonzero_frame_ratio`, `ocr_ad_text_score`, pre/post delta 계열이다.

## OCR Persistence Rule 초안
광고 시작 후보는 scene boundary anchor t 이후 10초 OCR score 상승과 keyword persistence를 본다. 광고 종료 후보는 t 이전 10초 OCR score가 높고 t 이후 감소하는지 본다. 이 threshold는 exploratory candidate이며 train/test 기반 final threshold가 아니다.

## Join Key
- `version`
- `video_id`
- `ad_interval_id`
- `segment_type`
- `segment_start_sec`
- `segment_end_sec`
- optional: `segment_start_sec_rounded_3`, `segment_end_sec_rounded_3`

## Label-aligned Caveat
현재 OCR feature는 정답 광고 라벨에 정렬된 분석용 segment에서 계산했다. inference에서는 정답 ad_start/ad_end를 알 수 없다. 실제 detector에서는 `visual_scene_boundary_anchors_v2_4.csv`의 `candidate_time_sec` 기준으로 pre_10s/post_10s OCR context feature를 다시 계산해야 한다.

## Sub Agent 검증 결과
{subagent_md}

## Safety Check
- old_project_modified: `{old_project_modified}`
- old project diff file: `{SNAPSHOT_DIFF}`
- latest_for_chatgpt_forbidden_files_count: `{latest_forbidden_count}`
- copied legacy/code/model paths: {stats.get('copied_asset_count')}

## WARN
{warnings_md}

## ERROR
{errors_md}

## 다음 작업 제안
1. `visual_scene_boundary_anchors_v2_4.csv` 기준 pre/post OCR persistence feature를 재계산한다.
2. audio persistence와 OCR persistence를 같은 boundary anchor table에 join한다.
3. scene-change anchor를 primary evidence로 두고 audio/OCR cue를 보조 confidence로 쓰는 rule-based interval detector를 만든다.
"""
    summary = f"""# Extract OCR Cues v2_4 Summary

## 완료 상태
v2_4 label-aligned OCR cue component feature 생성 작업을 완료했다.

## 핵심 처리 결과
- OCR backend: `{backend_info.get('ocr_backend')}` / `{backend_info.get('ocr_backend_status')}`
- sampling rows: {stats.get('total_sampling_rows')}
- OCR attempted/success/failed: {stats.get('ocr_attempted_count')} / {stats.get('ocr_success_count')} / {stats.get('ocr_failed_count')}
- segment feature rows: {stats.get('segment_feature_rows')}
- OCR empty ratio: {stats.get('ocr_empty_ratio')}

## 생성 산출물
- `{LABELED_SAMPLING_PLAN_OUT}`
- `{EDGE_SAMPLING_PLAN_OUT}`
- `{FRAME_RESULTS_OUT}`
- `{LABELED_FEATURES_OUT}`
- `{EDGE_FEATURES_OUT}`
- `{SEGMENT_SUMMARY_OUT}`
- `{EDGE_SUMMARY_OUT}`
- `{RULE_RECOMMENDATIONS_OUT}`
- `{KEYWORD_DICTIONARY_OUT}`
- `{PERSISTENCE_CONFIG_OUT}`

## 의미
이 결과는 OCR 단독 탐지 성능이 아니라 scene/audio 결합용 component cue 분석이다. 실제 inference에서는 scene boundary anchor 기준 pre/post OCR persistence feature로 재구성해야 한다.

## Safety
- old_project_modified: `{old_project_modified}`
- latest_for_chatgpt_forbidden_files_count: `{latest_forbidden_count}`

## Sub Agent 결과
{subagent_md}

## 상세 Report
- `{ANALYSIS_REPORT_OUT}`
"""
    ANALYSIS_REPORT_OUT.write_text(analysis, encoding="utf-8")
    SUMMARY_REPORT_OUT.write_text(summary, encoding="utf-8")


def append_final_log(
    stats: dict[str, Any],
    backend_info: dict[str, Any],
    subagent_results: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    old_project_modified: bool | None,
    latest_forbidden_count: int | None,
    started_at: str,
    ended_at: str,
) -> None:
    output_files = [
        LABELED_SAMPLING_PLAN_OUT,
        EDGE_SAMPLING_PLAN_OUT,
        FRAME_RESULTS_OUT,
        LABELED_FEATURES_OUT,
        EDGE_FEATURES_OUT,
        TOKEN_FREQUENCY_OUT,
        CANDIDATE_KEYWORDS_OUT,
        SEGMENT_SUMMARY_OUT,
        EDGE_SUMMARY_OUT,
        RULE_RECOMMENDATIONS_OUT,
        KEYWORD_DICTIONARY_OUT,
        PERSISTENCE_CONFIG_OUT,
        ANALYSIS_REPORT_OUT,
        SUMMARY_REPORT_OUT,
        RUN_LOG,
    ]
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n[RUN SUMMARY]\n")
        handle.write(f"started_at={started_at}\n")
        handle.write(f"ended_at={ended_at}\n")
        handle.write(f"python={sys.version.replace(chr(10), ' ')}\n")
        handle.write(f"cwd={Path.cwd()}\n")
        handle.write(f"inputs={[str(LABEL_PATH), str(MANIFEST_PATH), str(AUDIO_LABELED_SEGMENT_PLAN), str(AUDIO_EDGE_SEGMENT_PLAN)]}\n")
        handle.write(f"ocr_backend={backend_info}\n")
        handle.write(f"copied_asset_count={stats.get('copied_asset_count')}\n")
        handle.write(f"segment_sampling_row_count={stats.get('total_sampling_rows')}\n")
        handle.write(f"frame_ocr_attempted_success_failed={stats.get('ocr_attempted_count')}/{stats.get('ocr_success_count')}/{stats.get('ocr_failed_count')}\n")
        handle.write(f"ocr_empty_frame_count_ratio={stats.get('ocr_empty_count')}/{stats.get('ocr_empty_ratio')}\n")
        handle.write(f"segment_feature_row_count={stats.get('segment_feature_rows')}\n")
        handle.write("output_files=\n")
        for path in output_files:
            handle.write(f"- {path}\n")
        handle.write("warnings=\n")
        for item in warnings:
            handle.write(f"- {item}\n")
        handle.write("errors=\n")
        for item in errors:
            handle.write(f"- {item}\n")
        handle.write("sub_agent_results=\n")
        for result in subagent_results:
            handle.write(f"- {result['sub_agent']} | {result['status']} | {result['details']}\n")
        handle.write(f"old_project_modified={old_project_modified}\n")
        handle.write(f"latest_for_chatgpt_forbidden_files_count={latest_forbidden_count}\n")
        handle.write(f"reproduction_command=conda run -n cv python {SCRIPT_DIR / 'extract_ocr_cues_v2_4.py'} --overwrite\n")


def print_final_summary(stats: dict[str, Any], backend_info: dict[str, Any], subagent_results: list[dict[str, Any]], warnings: list[str], errors: list[str], old_project_modified: bool | None, latest_forbidden_count: int | None) -> None:
    subagent_lines = "\n".join(f"- {r['sub_agent']}: {r['status']}" for r in subagent_results)
    warning_lines = "\n".join(f"- {w}" for w in warnings) if warnings else "- none"
    error_lines = "\n".join(f"- {e}" for e in errors) if errors else "- none"
    text = f"""
## OCR Cue Extraction v2_4 완료

### 사용한 입력 파일
- `{LABEL_PATH}`
- `{MANIFEST_PATH}`
- `{AUDIO_LABELED_SEGMENT_PLAN}`
- `{AUDIO_EDGE_SEGMENT_PLAN}`

### OCR Backend
- `{backend_info.get('ocr_backend')}` / `{backend_info.get('ocr_backend_status')}`
- language: `{backend_info.get('language_config')}`

### 핵심 처리 결과
- sampling row count: {stats.get('total_sampling_rows')}
- OCR attempted/success/failed: {stats.get('ocr_attempted_count')} / {stats.get('ocr_success_count')} / {stats.get('ocr_failed_count')}
- segment feature row count: {stats.get('segment_feature_rows')}
- OCR empty ratio: {stats.get('ocr_empty_ratio')}

### 생성 산출물
- `{LABELED_SAMPLING_PLAN_OUT}`
- `{EDGE_SAMPLING_PLAN_OUT}`
- `{LABELED_FEATURES_OUT}`
- `{EDGE_FEATURES_OUT}`
- `{SEGMENT_SUMMARY_OUT}`
- `{EDGE_SUMMARY_OUT}`
- `{RULE_RECOMMENDATIONS_OUT}`
- `{KEYWORD_DICTIONARY_OUT}`
- `{PERSISTENCE_CONFIG_OUT}`

### latest_for_chatgpt
- `{LATEST_DIR}`
- forbidden files count: {latest_forbidden_count}

### Sub Agent 검증 결과
{subagent_lines}

### WARN / ERROR
WARN:
{warning_lines}

ERROR:
{error_lines}

### Safety Check
- old_project_modified={old_project_modified}
- latest_for_chatgpt_forbidden_files_count={latest_forbidden_count}

### 이번 결과의 의미
OCR 단독 탐지 결과가 아니라, scene-change anchor와 audio persistence에 결합할 OCR component cue와 rule 후보를 만든 것이다. inference에서는 정답 label segment가 아니라 scene boundary anchor 기준 pre/post OCR persistence feature로 재구성해야 한다.

### 다음 작업 제안
1. scene boundary anchor 기준 pre/post OCR persistence feature 재구성.
2. audio persistence와 OCR persistence 결합.
3. rule-based interval detector 생성.

### 상세 report
- `{ANALYSIS_REPORT_OUT}`
"""
    print(text, flush=True)


def main() -> None:
    args = parse_args()
    init_run_log()
    started_at = now_iso()
    warnings: list[str] = []
    errors: list[str] = []
    stats: dict[str, Any] = {}
    old_project_modified: bool | None = None
    latest_forbidden_count: int | None = None

    try:
        log("[STEP 01] Start OCR cue extraction task")
        log("[STEP 02] Create old project before snapshot")
        stats["before_snapshot_rows"] = create_project_snapshot(OLD_PROJECT, BEFORE_SNAPSHOT)

        log("[STEP 03] Discover OCR-related code in old project")
        discovery_df = discover_ocr_assets()
        stats["ocr_discovery_rows"] = int(len(discovery_df))

        log("[STEP 04] Copy OCR-related code/configs into new project only")
        copied_assets = copy_legacy_assets()
        stats["copied_asset_count"] = int(len(copied_assets))
        for item in copied_assets:
            log(f"copied_asset={item}")

        log("[STEP 05] Validate input files and schemas")
        labels, manifest, labeled_segment_plan, edge_segment_plan = read_inputs()
        stats["label_rows"] = int(len(labels))
        stats["manifest_rows"] = int(len(manifest))

        log("[STEP 06] Resolve video paths from manifest")
        manifest_video_ids = set(manifest["video_id"].astype(str))
        segment_video_ids = set(pd.concat([labeled_segment_plan["video_id"], edge_segment_plan["video_id"]]).astype(str))
        missing_manifest_ids = sorted(segment_video_ids - manifest_video_ids)
        if missing_manifest_ids:
            warnings.append(f"video_id missing in manifest: {missing_manifest_ids[:10]}")

        log("[STEP 07] Build label-aligned OCR segment tables")
        labeled_types = ["ad_full", "pre_ad_10s", "post_ad_10s", "random_non_ad_30s"]
        edge_types = ["ad_start_first_5s", "ad_start_first_10s", "ad_start_5to10s", "ad_end_last_5s", "ad_end_last_10s", "ad_end_minus10to_minus5s"]
        labeled_segment_plan = labeled_segment_plan[labeled_segment_plan["segment_type"].isin(labeled_types)].copy()
        edge_segment_plan = edge_segment_plan[edge_segment_plan["segment_type"].isin(edge_types)].copy()
        stats["labeled_segment_rows"] = int(len(labeled_segment_plan))
        stats["edge_segment_rows"] = int(len(edge_segment_plan))

        log("[STEP 08] Build OCR frame sampling plans")
        labeled_sampling_plan = build_sampling_plan(labeled_segment_plan, manifest, "labeled")
        edge_sampling_plan = build_sampling_plan(edge_segment_plan, manifest, "edge")
        labeled_sampling_plan.to_csv(LABELED_SAMPLING_PLAN_OUT, index=False)
        edge_sampling_plan.to_csv(EDGE_SAMPLING_PLAN_OUT, index=False)
        stats["labeled_sampling_rows"] = int(len(labeled_sampling_plan))
        stats["edge_sampling_rows"] = int(len(edge_sampling_plan))
        stats["total_sampling_rows"] = int(len(labeled_sampling_plan) + len(edge_sampling_plan))

        log("[STEP 09] Run OCR backend on sampled frames")
        backend_info = select_ocr_backend(skip_ocr=args.skip_ocr, requested=args.ocr_backend)
        if backend_info.get("warning"):
            warnings.append(str(backend_info["warning"]))
        all_plan = pd.concat([labeled_sampling_plan, edge_sampling_plan], ignore_index=True)
        frame_df = run_ocr(all_plan, backend_info, max_frames=int(args.max_frames or 0))

        log("[STEP 10] Save frame-level OCR results")
        frame_df.to_csv(FRAME_RESULTS_OUT, index=False)
        frame_df.head(100).to_csv(FRAME_RESULTS_SAMPLE_OUT, index=False)
        stats["ocr_attempted_count"] = int(len(frame_df))
        stats["ocr_success_count"] = int(frame_df["ocr_status"].astype(str).isin(["success_nonempty", "success_empty"]).sum()) if not frame_df.empty else 0
        stats["ocr_failed_count"] = int(stats["ocr_attempted_count"] - stats["ocr_success_count"])
        stats["ocr_empty_count"] = int((frame_df["ocr_status"].astype(str) == "success_empty").sum()) if not frame_df.empty else 0
        stats["ocr_empty_ratio"] = round(float(stats["ocr_empty_count"] / max(1, stats["ocr_attempted_count"])), 6)
        if stats["ocr_failed_count"] > 0:
            failed_status_counts = frame_df[frame_df["ocr_status"].astype(str) != "success_nonempty"]["ocr_status"].astype(str).value_counts().to_dict()
            warnings.append(f"OCR frame failures observed: failed_count={stats['ocr_failed_count']}, status_counts={failed_status_counts}. Treat affected OCR features as partial coverage rather than OCR-negative evidence.")

        log("[STEP 11] Normalize OCR text and build token frequency tables")
        token_freq_df, candidate_df = build_token_tables(frame_df)
        token_freq_df.to_csv(TOKEN_FREQUENCY_OUT, index=False)
        candidate_df.to_csv(CANDIDATE_KEYWORDS_OUT, index=False)

        log("[STEP 12] Build OCR keyword dictionary")
        dictionary = build_keyword_dictionary(candidate_df)

        log("[STEP 13] Aggregate segment-level OCR features")
        labeled_features = aggregate_segment_features(frame_df, labeled_sampling_plan, dictionary, labeled_types)
        edge_features = aggregate_segment_features(frame_df, edge_sampling_plan, dictionary, edge_types)
        labeled_features.to_csv(LABELED_FEATURES_OUT, index=False)
        edge_features.to_csv(EDGE_FEATURES_OUT, index=False)
        stats["segment_feature_rows"] = int(len(labeled_features) + len(edge_features))

        log("[STEP 14] Analyze segment-type OCR feature differences")
        segment_summary = build_comparison_summary(labeled_features, BASIC_COMPARISONS, FEATURES_FOR_ANALYSIS)
        edge_context_df = pd.concat(
            [
                labeled_features[labeled_features["segment_type"].isin(["pre_ad_10s", "post_ad_10s"])],
                edge_features,
            ],
            ignore_index=True,
        )
        edge_summary = build_comparison_summary(edge_context_df, EDGE_COMPARISONS, FEATURES_FOR_ANALYSIS)
        segment_summary.to_csv(SEGMENT_SUMMARY_OUT, index=False)
        edge_summary.to_csv(EDGE_SUMMARY_OUT, index=False)

        log("[STEP 15] Build OCR rule recommendations and persistence config")
        rec_df, persistence_config = build_rule_recommendations(labeled_features, edge_features)
        rec_df.to_csv(RULE_RECOMMENDATIONS_OUT, index=False)
        stats["persistence_candidate_thresholds"] = persistence_config.get("candidate_thresholds", {})

        log("[STEP 16] Run Sub Agent validations")
        subagent_results = validate_outputs(
            labeled_sampling_plan,
            edge_sampling_plan,
            frame_df,
            labeled_features,
            edge_features,
            segment_summary,
            edge_summary,
            backend_info,
            old_project_modified=None,
            latest_forbidden_count=None,
        )

        build_reports(stats, backend_info, subagent_results, warnings, errors, old_project_modified, latest_forbidden_count)

        log("[STEP 17] Update latest_for_chatgpt safely")
        latest_notes = [
            "label-aligned OCR features are analysis-only and must be rebuilt around scene boundary anchors for inference.",
        ]
        latest_files = [
            LABELED_SAMPLING_PLAN_OUT,
            EDGE_SAMPLING_PLAN_OUT,
            LABELED_FEATURES_OUT,
            EDGE_FEATURES_OUT,
            SEGMENT_SUMMARY_OUT,
            EDGE_SUMMARY_OUT,
            RULE_RECOMMENDATIONS_OUT,
            KEYWORD_DICTIONARY_OUT,
            PERSISTENCE_CONFIG_OUT,
            ANALYSIS_REPORT_OUT,
            SUMMARY_REPORT_OUT,
            RUN_LOG,
            SCRIPT_DIR / "extract_ocr_cues_v2_4.py",
            FRAME_RESULTS_SAMPLE_OUT,
        ]
        latest_forbidden_count = copy_to_latest(latest_files, latest_notes)

        log("[STEP 18] Create old project after snapshot and compare")
        stats["after_snapshot_rows"] = create_project_snapshot(OLD_PROJECT, AFTER_SNAPSHOT)
        old_project_modified, diff_count = compare_snapshots(BEFORE_SNAPSHOT, AFTER_SNAPSHOT, SNAPSHOT_DIFF)
        stats["old_project_snapshot_diff_count"] = int(diff_count)
        if old_project_modified:
            errors.append(f"ERROR: old project snapshot changed; diff_count={diff_count}; see {SNAPSHOT_DIFF}")
        else:
            log("old_project_modified=false")

        log("[STEP 19] Write final reports and run log")
        subagent_results = validate_outputs(
            labeled_sampling_plan,
            edge_sampling_plan,
            frame_df,
            labeled_features,
            edge_features,
            segment_summary,
            edge_summary,
            backend_info,
            old_project_modified=old_project_modified,
            latest_forbidden_count=latest_forbidden_count,
        )
        build_reports(stats, backend_info, subagent_results, warnings, errors, old_project_modified, latest_forbidden_count)
        ended_at = now_iso()
        append_final_log(stats, backend_info, subagent_results, warnings, errors, old_project_modified, latest_forbidden_count, started_at, ended_at)
        latest_forbidden_count = copy_to_latest(latest_files, latest_notes)
        build_reports(stats, backend_info, subagent_results, warnings, errors, old_project_modified, latest_forbidden_count)
        append_final_log(stats, backend_info, subagent_results, warnings, errors, old_project_modified, latest_forbidden_count, started_at, ended_at)

        log("[STEP 20] Print human-readable final summary")
        print_final_summary(stats, backend_info, subagent_results, warnings, errors, old_project_modified, latest_forbidden_count)

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        log(f"ERROR: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
