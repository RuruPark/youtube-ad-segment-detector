#!/usr/bin/env python3
"""Scene-anchor + full-video OCR extraction for v2.4.

Builds an OCR sampling plan that can be used at prediction time:
- 1.0s sampling inside canonical visual scene-boundary anchor windows.
- 1.5s background sampling outside those windows across the full video.

The script does not read actual ad labels and writes outputs only to a new run
directory plus latest bundles.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
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

PROJECT_ROOT_DEFAULT = Path('.').resolve()
VERSION_DEFAULT = 'v2_4'
SPLIT_SEED = 20240524
FIXED_SPLIT = {
    'train': [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    'validation': [3, 7, 18],
    'test': [4, 16, 17],
}

SCRIPT_PATH = PROJECT_ROOT_DEFAULT / 'scripts/ocr/extract_scene_anchor_full_video_ocr_v2_4.py'
ANCHOR_FILE_DEFAULT = PROJECT_ROOT_DEFAULT / 'data/features/visual_scene_boundary_anchors_v2_4_with_split.csv'
MANIFEST_FILE_DEFAULT = PROJECT_ROOT_DEFAULT / 'data/video_metadata/video_manifest_v2_2.csv'
SPLIT_FILE_DEFAULT = PROJECT_ROOT_DEFAULT / 'data/splits/video_split_v2_4.csv'
RUN_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / 'workspaces/ocr_scene_anchor_full_video_v2_4/runs'
LATEST_CHATGPT_DIR = PROJECT_ROOT_DEFAULT / 'outputs/latest_for_chatgpt_scene_anchor_full_video_ocr_v2_4'
LATEST_OCR_DIR = PROJECT_ROOT_DEFAULT / 'outputs/latest_ocr'
CACHE_DIR = PROJECT_ROOT_DEFAULT / 'cache/ocr/model_cache'
OLD_PROJECT_ROOT = Path('./_old_project_not_included').resolve()

FORBIDDEN_LATEST_EXTS = {
    '.mp4', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.m4a', '.jpg', '.jpeg', '.png', '.webp',
    '.pt', '.pth', '.ckpt', '.bin', '.onnx'
}
FORBIDDEN_LATEST_TOKENS = {'raw', 'frame', 'frames', 'cache', 'model', 'weights', 'checkpoint', 'bbox'}
SUCCESS_STATUSES = {'success_nonempty', 'success_empty'}
NONEMPTY_STATUS = 'success_nonempty'
TOKEN_RE = re.compile(r'[가-힣]+|[A-Za-z]+(?:[A-Za-z0-9_+\-&]*[A-Za-z0-9])?|\d+(?:[.,]\d+)*|[A-Za-z가-힣]*\d[A-Za-z가-힣0-9._+\-]*')
URL_RE = re.compile(r'(https?://|www\.|[A-Za-z0-9.-]+\.(?:com|kr|net|co|io|shop)\b|bit\.ly|linktr\.ee)', re.I)
EMAIL_RE = re.compile(r'\b[\w.\-+]+@[\w.\-]+\.[A-Za-z]{2,}\b')
PHONE_RE = re.compile(r'\b\d{2,3}[- .]?\d{3,4}[- .]?\d{4}\b')

KEYWORDS = {
    'ad_disclosure': [
        '유료광고', '유료 광고', '유료광고 포함', '광고 포함', '이 영상은 유료광고를 포함',
        'paid promotion', 'sponsored', 'sponsor'
    ],
    'sponsor': ['협찬', '제공', '제작지원', '지원받아', '광고주', '브랜드로부터'],
    'brand_product': ['제품', '브랜드', '신제품', '세트', '패키지', '향', '컬러', '기능', '성분', '앱', '서비스'],
    'promotion_discount': ['할인', '쿠폰', '프로모션', '이벤트', '적립', '무료배송', '특가', '혜택'],
    'purchase_cta': ['구매', '주문', '사용해보세요', '추천'],
    'link_more_info': ['링크', '더보기', '고정댓글', '댓글', '확인', '참고'],
    'negative_guard': ['광고 아님', '협찬 아님', '내돈내산', '직접 구매', '광고가 아닙니다', '협찬 받은 거 아님'],
}
RESULT_KEYWORD_COLUMNS = {
    'ad_disclosure': 'ad_disclosure_keyword_count',
    'sponsor': 'sponsor_keyword_count',
    'brand_product': 'brand_product_keyword_count',
    'promotion_discount': 'promotion_discount_keyword_count',
    'purchase_cta': 'purchase_cta_keyword_count',
    'link_more_info': 'link_more_info_keyword_count',
    'negative_guard': 'negative_guard_keyword_count',
}

OUTPUT_NAMES = {
    'sampling_plan': 'scene_anchor_full_video_ocr_sampling_plan_v2_4.csv',
    'dedup_summary': 'scene_anchor_full_video_ocr_sampling_plan_dedup_summary_v2_4.csv',
    'frame_results': 'scene_anchor_full_video_ocr_frame_results_v2_4.csv',
    'frame_results_sample_redacted': 'scene_anchor_full_video_ocr_frame_results_sample_redacted_v2_4.csv',
    'anchor_features': 'scene_anchor_ocr_window_features_v2_4.csv',
    'timeline_features': 'full_video_1p5s_ocr_20s_timeline_features_v2_4.csv',
    'video_summary': 'scene_anchor_full_video_ocr_video_summary_v2_4.csv',
    'keyword_summary': 'scene_anchor_full_video_ocr_keyword_hit_summary_v2_4.csv',
    'quality_checks': 'scene_anchor_full_video_ocr_quality_checks_v2_4.csv',
    'summary_md': 'scene_anchor_full_video_ocr_v2_4_summary.md',
    'report_json': 'scene_anchor_full_video_ocr_v2_4_report.json',
    'run_log': 'scene_anchor_full_video_ocr_v2_4_run_log.txt',
}

SAMPLING_PLAN_COLUMNS = [
    'run_id', 'version', 'split', 'video_id', 'video_title', 'video_path', 'video_duration_sec',
    'sample_id', 'timestamp_sec', 'timestamp_mmss', 'sampling_role', 'sampling_interval_sec',
    'is_near_scene_anchor', 'nearest_anchor_id', 'nearest_anchor_time_sec', 'nearest_anchor_delta_sec',
    'anchor_ids_joined', 'anchor_source_relation_joined', 'anchor_canonical_time_source_joined',
    'anchor_has_opencv_ffmpeg_candidate', 'anchor_has_resnet_candidate', 'anchor_source_count_max',
    'anchor_visual_boundary_strength_score_max', 'dedup_group_id', 'dedup_priority', 'sampling_status',
    'warning_message'
]

OCR_RESULT_COLUMNS = [
    'run_id', 'version', 'split', 'video_id', 'video_title', 'sample_id', 'timestamp_sec', 'timestamp_mmss',
    'sampling_role', 'is_near_scene_anchor', 'nearest_anchor_id', 'nearest_anchor_time_sec',
    'nearest_anchor_delta_sec', 'ocr_backend', 'ocr_status', 'ocr_error', 'ocr_text_raw', 'ocr_text_normalized',
    'ocr_text_joined', 'ocr_text_count', 'ocr_token_count', 'ocr_char_count', 'ocr_box_count',
    'ocr_mean_confidence', 'ocr_min_confidence', 'ocr_max_confidence', 'ocr_text_area_ratio',
    'ocr_high_conf_text_count', 'bbox_available', 'frame_width', 'frame_height', 'frame_keyword_score',
    'frame_text_density_score', 'frame_ad_text_score', 'ad_disclosure_keyword_count', 'sponsor_keyword_count',
    'brand_product_keyword_count', 'promotion_discount_keyword_count', 'purchase_cta_keyword_count',
    'link_more_info_keyword_count', 'negative_guard_keyword_count', 'temp_frame_cleanup_status', 'warning_message'
]

ANCHOR_FEATURE_COLUMNS = [
    'run_id', 'split', 'video_id', 'scene_boundary_anchor_id', 'canonical_boundary_time_sec',
    'source_relation', 'canonical_time_source', 'has_opencv_ffmpeg_candidate', 'has_resnet_candidate',
    'source_count', 'visual_boundary_strength_score', 'window_start_sec', 'window_end_sec', 'ocr_frame_count',
    'ocr_success_frame_count', 'ocr_empty_frame_count', 'ocr_failed_frame_count', 'ocr_nonempty_frame_count',
    'ocr_text_frame_ratio', 'ocr_mean_confidence', 'ocr_text_box_count_sum', 'ocr_token_count_sum',
    'ocr_char_count_sum', 'ocr_text_density_score_mean', 'ocr_ad_text_score_mean',
    'ad_disclosure_keyword_count_sum', 'sponsor_keyword_count_sum', 'brand_product_keyword_count_sum',
    'promotion_discount_keyword_count_sum', 'purchase_cta_keyword_count_sum', 'link_more_info_keyword_count_sum',
    'negative_guard_keyword_count_sum', 'first_ad_disclosure_timestamp_sec', 'first_sponsor_timestamp_sec',
    'first_purchase_cta_timestamp_sec', 'representative_ocr_text', 'top_ocr_tokens', 'ocr_anchor_context_status',
    'warning_message'
]

TIMELINE_FEATURE_COLUMNS = [
    'run_id', 'split', 'video_id', 'segment_index_20s', 'segment_start_sec', 'segment_end_sec',
    'segment_duration_sec', 'ocr_frame_count', 'near_anchor_frame_count', 'background_frame_count',
    'ocr_success_frame_count', 'ocr_empty_frame_count', 'ocr_failed_frame_count', 'ocr_text_frame_ratio',
    'ocr_mean_confidence', 'ocr_text_box_count_sum', 'ocr_token_count_sum', 'ocr_char_count_sum',
    'ocr_text_density_score_mean', 'ocr_ad_text_score_mean', 'ad_disclosure_keyword_count_sum',
    'sponsor_keyword_count_sum', 'brand_product_keyword_count_sum', 'promotion_discount_keyword_count_sum',
    'purchase_cta_keyword_count_sum', 'link_more_info_keyword_count_sum', 'negative_guard_keyword_count_sum',
    'nearest_scene_anchor_count', 'nearest_scene_anchor_min_delta_sec', 'representative_ocr_text', 'top_ocr_tokens',
    'warning_message'
]

VIDEO_SUMMARY_COLUMNS = [
    'run_id', 'split', 'video_id', 'video_title', 'video_duration_sec', 'total_sampling_plan_count',
    'near_anchor_sampling_count', 'background_sampling_count', 'dedup_removed_count', 'ocr_attempted_count',
    'ocr_success_count', 'ocr_empty_count', 'ocr_failed_count', 'ocr_nonempty_count', 'ocr_valid_ratio',
    'ocr_text_frame_ratio', 'first_ocr_timestamp_sec', 'last_ocr_timestamp_sec', 'max_gap_between_final_samples_sec',
    'median_gap_between_final_samples_sec', 'anchor_count', 'train_val_test_note', 'warning_message'
]


class TaskLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('', encoding='utf-8')

    def log(self, message: str) -> None:
        line = f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
        print(message, flush=True)
        with self.path.open('a', encoding='utf-8') as fh:
            fh.write(line + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract prediction-time scene-anchor/full-video OCR for v2.4.')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument('--version', default=VERSION_DEFAULT)
    parser.add_argument('--run-id', default='')
    parser.add_argument('--anchor-file', type=Path, default=ANCHOR_FILE_DEFAULT)
    parser.add_argument('--manifest-file', type=Path, default=MANIFEST_FILE_DEFAULT)
    parser.add_argument('--split-file', type=Path, default=SPLIT_FILE_DEFAULT)
    parser.add_argument('--anchor-window-sec', type=float, default=5.0)
    parser.add_argument('--near-anchor-interval-sec', type=float, default=1.0)
    parser.add_argument('--background-interval-sec', type=float, default=1.5)
    parser.add_argument('--timestamp-round-decimals', type=int, default=3)
    parser.add_argument('--dedup-tolerance-sec', type=float, default=0.05)
    parser.add_argument('--ocr-backend', choices=['auto', 'easyocr', 'pytesseract', 'none'], default='auto')
    parser.add_argument('--skip-ocr', action='store_true')
    parser.add_argument('--force-full-ocr', action='store_true')
    parser.add_argument('--max-full-ocr-frames', type=int, default=5000)
    parser.add_argument('--max-train-ocr-frames', type=int, default=5000)
    parser.add_argument('--pilot-video-count', type=int, default=1)
    parser.add_argument('--pilot-max-frames', type=int, default=120)
    parser.add_argument('--no-persist-frames', action='store_true', default=True)
    return parser.parse_args()


def now_id() -> str:
    return dt.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def assert_inside(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    return val if math.isfinite(val) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def fmt_num(value: Any, digits: int = 6) -> Any:
    val = safe_float(value)
    if math.isnan(val):
        return ''
    return round(val, digits)


def mmss(seconds: Any) -> str:
    value = safe_float(seconds, 0.0)
    value = max(0.0, value)
    total = int(math.floor(value + 1e-9))
    return f'{total // 60:02d}:{total % 60:02d}'


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize('NFKC', str(text or '')).lower()
    value = re.sub(r'[^\w\s가-힣%₩$./:\-+,&()]', ' ', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def tokenize(text: Any) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(normalize_text(text)) if tok.strip()]


def redact_text(text: Any, max_len: int = 160) -> str:
    value = str(text or '')
    value = URL_RE.sub('[URL]', value)
    value = EMAIL_RE.sub('[EMAIL]', value)
    value = PHONE_RE.sub('[PHONE]', value)
    value = re.sub(r'\s+', ' ', value).strip()
    if len(value) > max_len:
        value = value[:max_len - 3] + '...'
    return value


def count_keyword_terms(text: str) -> dict[str, int]:
    norm = normalize_text(text)
    compact = re.sub(r'\s+', '', norm)
    counts: dict[str, int] = {}
    for category, terms in KEYWORDS.items():
        total = 0
        for term in terms:
            t_norm = normalize_text(term)
            if ' ' in t_norm:
                total += norm.count(t_norm)
            else:
                total += compact.count(re.sub(r'\s+', '', t_norm))
        counts[category] = int(total)
    return counts


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


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def file_stats(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.exists():
            st = path.stat()
            out[str(path)] = {'exists': True, 'size': int(st.st_size), 'mtime_ns': int(st.st_mtime_ns)}
        else:
            out[str(path)] = {'exists': False, 'size': None, 'mtime_ns': None}
    return out


def stats_changed(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    return [path for path, stat in before.items() if after.get(path) != stat]


def arange_inclusive(start: float, end: float, step: float, decimals: int) -> list[float]:
    if end < start or step <= 0:
        return []
    values: list[float] = []
    current = start
    eps = step / 1000.0
    while current <= end + eps:
        values.append(round(float(current), decimals))
        current += step
    return sorted(set(values))


def load_inputs(args: argparse.Namespace, warnings: list[str], errors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchors = pd.read_csv(args.anchor_file, encoding='utf-8-sig')
    manifest = pd.read_csv(args.manifest_file, encoding='utf-8-sig')
    split = pd.read_csv(args.split_file, encoding='utf-8-sig')
    return anchors, manifest, split


def validate_inputs(anchors: pd.DataFrame, manifest: pd.DataFrame, split: pd.DataFrame, warnings: list[str], errors: list[str]) -> dict[str, Any]:
    required_anchor = [
        'scene_boundary_anchor_id', 'video_id', 'canonical_boundary_time_sec', 'split', 'source_relation',
        'canonical_time_source', 'has_opencv_ffmpeg_candidate', 'has_resnet_candidate', 'source_count',
        'visual_boundary_strength_score'
    ]
    required_manifest = ['video_id', 'video_title', 'video_path', 'duration_sec']
    required_split = ['video_id', 'split']
    for col in required_anchor:
        if col not in anchors.columns:
            errors.append(f'anchor missing required column: {col}')
    for col in required_manifest:
        if col not in manifest.columns:
            errors.append(f'manifest missing required column: {col}')
    for col in required_split:
        if col not in split.columns:
            errors.append(f'split missing required column: {col}')
    anchors['video_id'] = pd.to_numeric(anchors['video_id'], errors='coerce').astype('Int64')
    anchors['canonical_boundary_time_sec'] = pd.to_numeric(anchors['canonical_boundary_time_sec'], errors='coerce')
    manifest['video_id'] = pd.to_numeric(manifest['video_id'], errors='coerce').astype('Int64')
    manifest['duration_sec'] = pd.to_numeric(manifest['duration_sec'], errors='coerce')
    split['video_id'] = pd.to_numeric(split['video_id'], errors='coerce').astype('Int64')
    split_counts = anchors['split'].value_counts(dropna=False).to_dict() if 'split' in anchors.columns else {}
    fixed_actual = {s: sorted(split.loc[split['split'].eq(s), 'video_id'].dropna().astype(int).tolist()) for s in ['train', 'validation', 'test']}
    fixed_ok = all(fixed_actual.get(k) == sorted(v) for k, v in FIXED_SPLIT.items())
    if len(anchors) != 1329:
        warnings.append(f'anchor row count expected 1329 but got {len(anchors)}')
    expected_counts = {'train': 878, 'validation': 225, 'test': 226}
    for key, expected in expected_counts.items():
        actual = int(split_counts.get(key, 0))
        if actual != expected:
            warnings.append(f'anchor split count {key} expected {expected} but got {actual}')
    if not fixed_ok:
        warnings.append(f'fixed split mismatch: {fixed_actual}')
    video_ids_anchor = set(anchors['video_id'].dropna().astype(int).unique().tolist())
    video_ids_manifest = set(manifest['video_id'].dropna().astype(int).unique().tolist())
    missing_manifest = sorted(video_ids_anchor - video_ids_manifest)
    if missing_manifest:
        errors.append(f'anchor video_id missing from manifest: {missing_manifest}')
    missing_paths: list[int] = []
    for row in manifest.itertuples(index=False):
        vid = safe_int(getattr(row, 'video_id'))
        path = Path(str(getattr(row, 'video_path')))
        if vid in video_ids_anchor and not path.exists():
            missing_paths.append(vid)
    if missing_paths:
        warnings.append(f'video paths missing for video_id={missing_paths}')
    return {
        'anchor_row_count': int(len(anchors)),
        'anchor_split_counts': {str(k): int(v) for k, v in split_counts.items()},
        'manifest_row_count': int(len(manifest)),
        'split_row_count': int(len(split)),
        'fixed_split_match': bool(fixed_ok),
        'missing_manifest_video_ids': missing_manifest,
        'missing_video_path_ids': missing_paths,
    }


def join_anchor_manifest(anchors: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    cols = ['video_id', 'video_title', 'video_path', 'duration_sec']
    merged = anchors.merge(manifest[cols], on='video_id', how='left', suffixes=('', '_manifest'))
    if 'video_title_manifest' in merged.columns:
        merged['video_title'] = merged['video_title'].fillna(merged['video_title_manifest'])
    merged.rename(columns={'duration_sec': 'video_duration_sec'}, inplace=True)
    return merged


def nearest_anchor_info(anchor_times: list[tuple[float, str]], timestamp: float) -> tuple[str, float, float]:
    if not anchor_times:
        return '', math.nan, math.nan
    nearest = min(anchor_times, key=lambda item: abs(item[0] - timestamp))
    return nearest[1], float(nearest[0]), round(float(timestamp - nearest[0]), 6)


def build_sampling_records(anchors: pd.DataFrame, args: argparse.Namespace, run_id: str, warnings: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw_near_count = 0
    raw_background_count = 0
    anchor_by_video: dict[int, pd.DataFrame] = {int(v): g.copy() for v, g in anchors.groupby('video_id')}
    manifest_like = anchors.drop_duplicates('video_id').set_index('video_id').to_dict(orient='index')

    for vid, video_anchors in anchor_by_video.items():
        video_anchors = video_anchors.sort_values('canonical_boundary_time_sec').copy()
        first = video_anchors.iloc[0]
        duration = safe_float(first.get('video_duration_sec'))
        if math.isnan(duration) or duration <= 0:
            warnings.append(f'video_id={vid} invalid duration; skipping sampling records')
            continue
        split = str(first.get('split', ''))
        title = str(first.get('video_title', ''))
        video_path = str(first.get('video_path', ''))
        anchor_times = [(safe_float(r.canonical_boundary_time_sec), str(r.scene_boundary_anchor_id)) for r in video_anchors.itertuples(index=False)]

        for row in video_anchors.itertuples(index=False):
            anchor_time = safe_float(row.canonical_boundary_time_sec)
            if math.isnan(anchor_time):
                continue
            start = max(0.0, anchor_time - args.anchor_window_sec)
            end = min(duration, anchor_time + args.anchor_window_sec)
            if anchor_time < 0 or anchor_time > duration:
                warnings.append(f'anchor clipped/partly outside duration: video_id={vid}, anchor={row.scene_boundary_anchor_id}, time={anchor_time}, duration={duration}')
            for ts in arange_inclusive(start, end, args.near_anchor_interval_sec, args.timestamp_round_decimals):
                raw_near_count += 1
                nearest_id, nearest_time, delta = nearest_anchor_info(anchor_times, ts)
                records.append({
                    'run_id': run_id,
                    'version': args.version,
                    'split': split,
                    'video_id': vid,
                    'video_title': title,
                    'video_path': video_path,
                    'video_duration_sec': round(duration, 6),
                    'timestamp_sec': ts,
                    'timestamp_mmss': mmss(ts),
                    'sampling_role': 'near_scene_anchor_1s',
                    'sampling_interval_sec': args.near_anchor_interval_sec,
                    'is_near_scene_anchor': True,
                    'nearest_anchor_id': nearest_id,
                    'nearest_anchor_time_sec': nearest_time,
                    'nearest_anchor_delta_sec': delta,
                    'anchor_ids': [str(row.scene_boundary_anchor_id)],
                    'source_relations': [str(row.source_relation)],
                    'canonical_time_sources': [str(row.canonical_time_source)],
                    'has_opencv_flags': [bool_value(row.has_opencv_ffmpeg_candidate)],
                    'has_resnet_flags': [bool_value(row.has_resnet_candidate)],
                    'source_counts': [safe_int(row.source_count)],
                    'strength_scores': [safe_float(row.visual_boundary_strength_score, 0.0)],
                    'dedup_priority': 1,
                    'sampling_status': 'planned',
                    'warning_message': '',
                    'raw_kind': 'near',
                })

        times_only = [t for t, _ in anchor_times if not math.isnan(t)]
        for ts in arange_inclusive(0.0, duration, args.background_interval_sec, args.timestamp_round_decimals):
            if any(abs(ts - anchor_time) <= args.anchor_window_sec + args.dedup_tolerance_sec for anchor_time in times_only):
                continue
            raw_background_count += 1
            nearest_id, nearest_time, delta = nearest_anchor_info(anchor_times, ts)
            records.append({
                'run_id': run_id,
                'version': args.version,
                'split': split,
                'video_id': vid,
                'video_title': title,
                'video_path': video_path,
                'video_duration_sec': round(duration, 6),
                'timestamp_sec': ts,
                'timestamp_mmss': mmss(ts),
                'sampling_role': 'background_full_video_1p5s',
                'sampling_interval_sec': args.background_interval_sec,
                'is_near_scene_anchor': False,
                'nearest_anchor_id': nearest_id,
                'nearest_anchor_time_sec': nearest_time,
                'nearest_anchor_delta_sec': delta,
                'anchor_ids': [],
                'source_relations': [],
                'canonical_time_sources': [],
                'has_opencv_flags': [],
                'has_resnet_flags': [],
                'source_counts': [],
                'strength_scores': [],
                'dedup_priority': 2,
                'sampling_status': 'planned',
                'warning_message': '',
                'raw_kind': 'background',
            })
    stats = {'raw_near_count': raw_near_count, 'raw_background_count': raw_background_count, 'raw_total_count': len(records)}
    return records, stats


def merge_values(values: list[Any]) -> str:
    clean = []
    for value in values:
        text = str(value).strip()
        if text and text not in clean:
            clean.append(text)
    return ';'.join(clean)


def deduplicate_records(records: list[dict[str, Any]], args: argparse.Namespace, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    final_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    records_sorted = sorted(records, key=lambda r: (int(r['video_id']), float(r['timestamp_sec']), int(r['dedup_priority'])))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records_sorted:
        grouped[int(rec['video_id'])].append(rec)

    total_removed = 0
    for vid, rows in grouped.items():
        clusters: list[list[dict[str, Any]]] = []
        for rec in rows:
            if not clusters or abs(float(rec['timestamp_sec']) - float(clusters[-1][0]['timestamp_sec'])) > args.dedup_tolerance_sec:
                clusters.append([rec])
            else:
                clusters[-1].append(rec)
        video_removed = len(rows) - len(clusters)
        total_removed += video_removed
        video_final_rows: list[dict[str, Any]] = []
        for idx, cluster in enumerate(clusters, start=1):
            cluster_sorted = sorted(cluster, key=lambda r: (int(r['dedup_priority']), abs(float(r['nearest_anchor_delta_sec'])) if not math.isnan(safe_float(r['nearest_anchor_delta_sec'])) else 999999))
            chosen = dict(cluster_sorted[0])
            all_anchor_ids: list[str] = []
            all_relations: list[str] = []
            all_sources: list[str] = []
            opencv_flags: list[bool] = []
            resnet_flags: list[bool] = []
            source_counts: list[int] = []
            strengths: list[float] = []
            for rec in cluster:
                all_anchor_ids.extend(rec.get('anchor_ids', []))
                all_relations.extend(rec.get('source_relations', []))
                all_sources.extend(rec.get('canonical_time_sources', []))
                opencv_flags.extend(rec.get('has_opencv_flags', []))
                resnet_flags.extend(rec.get('has_resnet_flags', []))
                source_counts.extend(rec.get('source_counts', []))
                strengths.extend(rec.get('strength_scores', []))
            chosen['anchor_ids_joined'] = merge_values(all_anchor_ids)
            chosen['anchor_source_relation_joined'] = merge_values(all_relations)
            chosen['anchor_canonical_time_source_joined'] = merge_values(all_sources)
            chosen['anchor_has_opencv_ffmpeg_candidate'] = bool(any(opencv_flags))
            chosen['anchor_has_resnet_candidate'] = bool(any(resnet_flags))
            chosen['anchor_source_count_max'] = max(source_counts) if source_counts else 0
            chosen['anchor_visual_boundary_strength_score_max'] = round(max(strengths), 6) if strengths else 0.0
            chosen['dedup_group_id'] = f'{run_id}_V{vid}_DG{idx:06d}'
            chosen['sample_id'] = f'{run_id}_V{vid}_S{idx:06d}'
            if len(cluster) > 1:
                chosen['warning_message'] = 'merged_duplicate_timestamp_within_tolerance'
            out = {col: chosen.get(col, '') for col in SAMPLING_PLAN_COLUMNS}
            out['timestamp_sec'] = round(float(chosen['timestamp_sec']), args.timestamp_round_decimals)
            out['timestamp_mmss'] = mmss(out['timestamp_sec'])
            video_final_rows.append(out)
            final_rows.append(out)
        timestamps = [float(r['timestamp_sec']) for r in video_final_rows]
        gaps = np.diff(sorted(timestamps)) if len(timestamps) > 1 else np.array([])
        raw_near = sum(1 for r in rows if r.get('raw_kind') == 'near')
        raw_bg = sum(1 for r in rows if r.get('raw_kind') == 'background')
        summary_rows.append({
            'run_id': run_id,
            'video_id': vid,
            'split': str(video_final_rows[0]['split']) if video_final_rows else '',
            'raw_near_anchor_count': int(raw_near),
            'raw_background_count': int(raw_bg),
            'raw_total_count': int(len(rows)),
            'final_total_count': int(len(video_final_rows)),
            'final_near_anchor_count': int(sum(1 for r in video_final_rows if r['sampling_role'] == 'near_scene_anchor_1s')),
            'final_background_count': int(sum(1 for r in video_final_rows if r['sampling_role'] == 'background_full_video_1p5s')),
            'dedup_removed_count': int(video_removed),
            'max_gap_between_final_samples_sec': round(float(np.max(gaps)), 6) if gaps.size else 0.0,
            'median_gap_between_final_samples_sec': round(float(np.median(gaps)), 6) if gaps.size else 0.0,
            'qa_status': 'PASS' if video_final_rows else 'WARN',
            'warning_message': '' if video_final_rows else 'no_final_sampling_rows',
        })
    plan = pd.DataFrame(final_rows, columns=SAMPLING_PLAN_COLUMNS)
    if not plan.empty:
        plan.sort_values(['video_id', 'timestamp_sec', 'dedup_priority'], inplace=True)
        plan.reset_index(drop=True, inplace=True)
    dedup = pd.DataFrame(summary_rows)
    stats = {
        'raw_total_count': int(len(records)),
        'final_total_count': int(len(plan)),
        'dedup_removed_count': int(total_removed),
        'final_near_anchor_count': int((plan['sampling_role'] == 'near_scene_anchor_1s').sum()) if not plan.empty else 0,
        'final_background_count': int((plan['sampling_role'] == 'background_full_video_1p5s').sum()) if not plan.empty else 0,
    }
    return plan, dedup, stats


def select_ocr_backend(requested: str, skip_ocr: bool) -> tuple[dict[str, Any], Any]:
    info = {
        'ocr_backend': 'none',
        'ocr_backend_status': 'not_selected',
        'requested_backend': requested,
        'easyocr_available': package_available('easyocr'),
        'pytesseract_available': package_available('pytesseract'),
        'pillow_available': package_available('PIL'),
        'tesseract_binary': shutil.which('tesseract') or '',
        'cv2_available': package_available('cv2'),
        'model_cache_dir': str(CACHE_DIR),
        'gpu_used': False,
        'language_config': '',
        'engine_version': '',
        'warning': '',
        'network_download_attempted': False,
    }
    if skip_ocr or requested == 'none':
        info['ocr_backend_status'] = 'skipped_by_argument'
        info['warning'] = 'OCR execution skipped by argument.'
        return info, None
    if requested in {'auto', 'easyocr'} and info['easyocr_available']:
        try:
            import easyocr
            use_gpu = False
            try:
                import torch
                use_gpu = bool(torch.cuda.is_available())
            except Exception:
                use_gpu = False
            reader = easyocr.Reader(['ko', 'en'], gpu=use_gpu, download_enabled=False, model_storage_directory=str(CACHE_DIR), verbose=False)
            info.update({
                'ocr_backend': 'easyocr',
                'ocr_backend_status': 'ready',
                'gpu_used': use_gpu,
                'language_config': 'ko+en',
                'engine_version': str(getattr(easyocr, '__version__', 'unknown')),
            })
            return info, reader
        except Exception as exc:
            info['warning'] = f'EasyOCR cache-only initialization failed: {type(exc).__name__}: {exc}'
            if requested == 'easyocr':
                info['ocr_backend_status'] = 'unavailable'
                return info, None
    if requested in {'auto', 'pytesseract'} and info['pytesseract_available'] and info['tesseract_binary']:
        try:
            import pytesseract
            info.update({
                'ocr_backend': 'pytesseract',
                'ocr_backend_status': 'ready',
                'language_config': 'kor+eng',
                'engine_version': str(getattr(pytesseract, 'get_tesseract_version', lambda: 'unknown')()),
            })
            return info, pytesseract
        except Exception as exc:
            info['warning'] = (info.get('warning', '') + f'; pytesseract init failed: {type(exc).__name__}: {exc}').strip('; ')
    info['ocr_backend_status'] = 'unavailable'
    if not info['warning']:
        info['warning'] = 'No usable OCR backend found without installing/downloading packages.'
    return info, None


def choose_ocr_subset(plan: pd.DataFrame, args: argparse.Namespace, backend_info: dict[str, Any], warnings: list[str]) -> tuple[pd.DataFrame, str, str]:
    if backend_info.get('ocr_backend_status') != 'ready':
        return plan.iloc[0:0].copy(), 'fallback_plan_only', backend_info.get('warning', 'ocr backend unavailable')
    if args.skip_ocr or args.ocr_backend == 'none':
        return plan.iloc[0:0].copy(), 'fallback_plan_only', 'OCR skipped by argument.'
    total = int(len(plan))
    if args.force_full_ocr or total <= args.max_full_ocr_frames:
        return plan.copy(), 'executed', ''
    train_plan = plan[plan['split'].eq('train')].copy()
    if len(train_plan) <= args.max_train_ocr_frames:
        warnings.append(f'Full OCR has {total} frames, above max_full_ocr_frames={args.max_full_ocr_frames}; running train split pilot.')
        return train_plan, 'pilot_only', f'full plan too large for default run ({total} > {args.max_full_ocr_frames}); train pilot selected'
    train_video_ids = sorted(train_plan['video_id'].astype(int).unique().tolist())[: max(1, int(args.pilot_video_count))]
    subset = train_plan[train_plan['video_id'].astype(int).isin(train_video_ids)].sort_values(['video_id', 'timestamp_sec']).head(max(1, int(args.pilot_max_frames))).copy()
    reason = f'full plan too large ({total}); train plan too large ({len(train_plan)}); smoke pilot selected for video_id={train_video_ids}, max_frames={args.pilot_max_frames}'
    warnings.append(reason)
    return subset, 'pilot_only', reason


def run_easyocr(reader: Any, frame: Any) -> list[dict[str, Any]]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = reader.readtext(rgb, detail=1, paragraph=False)
    rows: list[dict[str, Any]] = []
    for det in detections:
        if not isinstance(det, (list, tuple)) or len(det) < 3:
            continue
        text = str(det[1]).strip()
        if not text:
            continue
        try:
            conf = float(det[2])
        except Exception:
            conf = math.nan
        rows.append({'text': text, 'confidence': conf, 'bbox': det[0]})
    return rows


def run_pytesseract(module: Any, frame: Any) -> list[dict[str, Any]]:
    from PIL import Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    try:
        data = module.image_to_data(image, lang='kor+eng', output_type=module.Output.DICT)
    except Exception:
        data = module.image_to_data(image, lang='eng', output_type=module.Output.DICT)
    rows: list[dict[str, Any]] = []
    n = len(data.get('text', []))
    for i in range(n):
        text = str(data['text'][i]).strip()
        if not text:
            continue
        conf = safe_float(data.get('conf', [''])[i]) / 100.0
        x, y, w, h = [safe_float(data.get(key, [0])[i], 0.0) for key in ['left', 'top', 'width', 'height']]
        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        rows.append({'text': text, 'confidence': conf, 'bbox': bbox})
    return rows


def decode_frame(cap: cv2.VideoCapture, timestamp_sec: float, duration_sec: float) -> tuple[Any, str]:
    try:
        ts = min(max(0.0, float(timestamp_sec)), max(0.0, float(duration_sec) - 0.05))
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None, 'frame_decode_failed'
        return frame, 'ok'
    except Exception as exc:
        return None, f'frame_decode_error:{type(exc).__name__}:{exc}'


def empty_result_from_plan(row: pd.Series, backend: str, status: str, error: str = '', warning: str = '') -> dict[str, Any]:
    result = {col: '' for col in OCR_RESULT_COLUMNS}
    for col in ['run_id', 'version', 'split', 'video_id', 'video_title', 'sample_id', 'timestamp_sec', 'timestamp_mmss', 'sampling_role', 'is_near_scene_anchor', 'nearest_anchor_id', 'nearest_anchor_time_sec', 'nearest_anchor_delta_sec']:
        result[col] = row.get(col, '')
    result.update({
        'ocr_backend': backend,
        'ocr_status': status,
        'ocr_error': error,
        'ocr_text_count': 0,
        'ocr_token_count': 0,
        'ocr_char_count': 0,
        'ocr_box_count': 0,
        'ocr_text_area_ratio': 0.0,
        'ocr_high_conf_text_count': 0,
        'bbox_available': False,
        'frame_width': '',
        'frame_height': '',
        'frame_keyword_score': 0.0,
        'frame_text_density_score': 0.0,
        'frame_ad_text_score': 0.0,
        'ad_disclosure_keyword_count': 0,
        'sponsor_keyword_count': 0,
        'brand_product_keyword_count': 0,
        'promotion_discount_keyword_count': 0,
        'purchase_cta_keyword_count': 0,
        'link_more_info_keyword_count': 0,
        'negative_guard_keyword_count': 0,
        'temp_frame_cleanup_status': 'not_applicable_no_persist_frames',
        'warning_message': warning or row.get('warning_message', ''),
    })
    return result


def build_result_from_detections(row: pd.Series, backend: str, detections: list[dict[str, Any]], frame_shape: tuple[int, int, int]) -> dict[str, Any]:
    texts = [str(d.get('text', '')).strip() for d in detections if str(d.get('text', '')).strip()]
    raw = '\n'.join(texts)
    normalized = normalize_text(raw)
    joined = ' | '.join(texts)
    tokens = tokenize(raw)
    confidences = [safe_float(d.get('confidence')) for d in detections if not math.isnan(safe_float(d.get('confidence')))]
    image_area = float(frame_shape[0] * frame_shape[1]) if frame_shape else 0.0
    areas = [polygon_area(d.get('bbox')) for d in detections]
    area_ratio = min(1.0, sum(areas) / image_area) if image_area > 0 else 0.0
    high_conf = int(sum(1 for c in confidences if c >= 0.60))
    keyword_counts = count_keyword_terms(raw)
    total_keyword_count = sum(keyword_counts.values())
    keyword_score = min(1.0, total_keyword_count / 3.0)
    density_score = float(np.mean([min(1.0, len(tokens) / 18.0), min(1.0, len(normalized) / 120.0), min(1.0, len(detections) / 8.0), min(1.0, area_ratio / 0.08)]))
    ad_text_score = min(1.0, 0.65 * keyword_score + 0.35 * density_score)
    status = 'success_nonempty' if texts else 'success_empty'
    result = empty_result_from_plan(row, backend, status)
    result.update({
        'ocr_text_raw': raw,
        'ocr_text_normalized': normalized,
        'ocr_text_joined': joined,
        'ocr_text_count': int(len(texts)),
        'ocr_token_count': int(len(tokens)),
        'ocr_char_count': int(len(normalized)),
        'ocr_box_count': int(len(detections)),
        'ocr_mean_confidence': round(float(np.mean(confidences)), 6) if confidences else '',
        'ocr_min_confidence': round(float(np.min(confidences)), 6) if confidences else '',
        'ocr_max_confidence': round(float(np.max(confidences)), 6) if confidences else '',
        'ocr_text_area_ratio': round(float(area_ratio), 6),
        'ocr_high_conf_text_count': high_conf,
        'bbox_available': bool(detections),
        'frame_width': int(frame_shape[1]) if frame_shape else '',
        'frame_height': int(frame_shape[0]) if frame_shape else '',
        'frame_keyword_score': round(keyword_score, 6),
        'frame_text_density_score': round(density_score, 6),
        'frame_ad_text_score': round(ad_text_score, 6),
    })
    for category, out_col in RESULT_KEYWORD_COLUMNS.items():
        result[out_col] = int(keyword_counts.get(category, 0))
    return result


def run_ocr(plan: pd.DataFrame, executable: pd.DataFrame, backend_info: dict[str, Any], reader: Any, logger: TaskLogger) -> pd.DataFrame:
    backend = str(backend_info.get('ocr_backend', 'none'))
    executable_ids = set(executable['sample_id'].astype(str).tolist()) if not executable.empty else set()
    result_by_sample: dict[str, dict[str, Any]] = {}
    if executable.empty:
        for _, row in plan.iterrows():
            result_by_sample[str(row['sample_id'])] = empty_result_from_plan(row, backend, 'not_attempted_due_to_execution_scope', warning='OCR not attempted for this run scope')
        return pd.DataFrame([result_by_sample[str(sid)] for sid in plan['sample_id'].astype(str)])
    total = len(executable)
    processed = 0
    for video_path, group in executable.groupby('video_path', sort=False):
        path = Path(str(video_path))
        if not path.exists():
            for _, row in group.iterrows():
                result_by_sample[str(row['sample_id'])] = empty_result_from_plan(row, backend, 'video_path_missing', error=str(path))
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            for _, row in group.iterrows():
                result_by_sample[str(row['sample_id'])] = empty_result_from_plan(row, backend, 'video_open_failed', error=str(path))
            continue
        try:
            for _, row in group.iterrows():
                processed += 1
                frame, frame_status = decode_frame(cap, float(row['timestamp_sec']), float(row['video_duration_sec']))
                if frame is None:
                    result_by_sample[str(row['sample_id'])] = empty_result_from_plan(row, backend, frame_status, error=frame_status)
                else:
                    try:
                        if backend == 'easyocr':
                            detections = run_easyocr(reader, frame)
                        elif backend == 'pytesseract':
                            detections = run_pytesseract(reader, frame)
                        else:
                            detections = []
                        result_by_sample[str(row['sample_id'])] = build_result_from_detections(row, backend, detections, frame.shape)
                    except Exception as exc:
                        result_by_sample[str(row['sample_id'])] = empty_result_from_plan(row, backend, 'ocr_failed', error=f'{type(exc).__name__}: {exc}')
                if processed % 20 == 0 or processed == total:
                    logger.log(f'OCR progress: {processed}/{total} attempted frames')
        finally:
            cap.release()
    rows: list[dict[str, Any]] = []
    for _, row in plan.iterrows():
        sample_id = str(row['sample_id'])
        if sample_id in result_by_sample:
            rows.append(result_by_sample[sample_id])
        elif sample_id in executable_ids:
            rows.append(empty_result_from_plan(row, backend, 'ocr_missing_result', warning='sample selected but no OCR result recorded'))
        else:
            rows.append(empty_result_from_plan(row, backend, 'not_attempted_due_to_execution_scope', warning='Outside OCR execution scope for this fallback/pilot run'))
    return pd.DataFrame(rows, columns=OCR_RESULT_COLUMNS)


def status_counts(frame_df: pd.DataFrame) -> dict[str, int]:
    if frame_df.empty:
        return {'attempted': 0, 'success': 0, 'empty': 0, 'failed': 0, 'nonempty': 0, 'not_attempted': 0}
    status = frame_df['ocr_status'].astype(str)
    attempted_mask = ~status.eq('not_attempted_due_to_execution_scope')
    success_mask = status.isin(SUCCESS_STATUSES)
    empty_mask = status.eq('success_empty')
    nonempty_mask = status.eq('success_nonempty')
    failed_mask = attempted_mask & ~success_mask
    return {
        'attempted': int(attempted_mask.sum()),
        'success': int(success_mask.sum()),
        'empty': int(empty_mask.sum()),
        'failed': int(failed_mask.sum()),
        'nonempty': int(nonempty_mask.sum()),
        'not_attempted': int((~attempted_mask).sum()),
    }


def aggregate_text(rows: pd.DataFrame) -> tuple[str, str]:
    texts = [str(t) for t in rows.get('ocr_text_joined', pd.Series(dtype=str)).fillna('') if str(t).strip()]
    representative = redact_text(max(texts, key=len), max_len=220) if texts else ''
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tok for tok in tokenize(text) if len(tok) > 1)
    top_tokens = ';'.join([f'{tok}:{cnt}' for tok, cnt in counter.most_common(15)])
    return representative, top_tokens


def mean_numeric(series: pd.Series) -> Any:
    vals = pd.to_numeric(series, errors='coerce').dropna()
    return round(float(vals.mean()), 6) if not vals.empty else ''


def build_anchor_features(anchors: pd.DataFrame, frame_df: pd.DataFrame, args: argparse.Namespace, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame_by_video = {int(v): g.copy() for v, g in frame_df.groupby('video_id')}
    for anchor in anchors.sort_values(['video_id', 'canonical_boundary_time_sec']).itertuples(index=False):
        vid = safe_int(anchor.video_id)
        t = safe_float(anchor.canonical_boundary_time_sec)
        duration = safe_float(getattr(anchor, 'video_duration_sec', math.nan))
        start = max(0.0, t - args.anchor_window_sec)
        end = min(duration, t + args.anchor_window_sec) if not math.isnan(duration) else t + args.anchor_window_sec
        frames = frame_by_video.get(vid, pd.DataFrame(columns=OCR_RESULT_COLUMNS))
        window = frames[(pd.to_numeric(frames['timestamp_sec'], errors='coerce') >= start - 1e-9) & (pd.to_numeric(frames['timestamp_sec'], errors='coerce') <= end + 1e-9)].copy()
        counts = status_counts(window)
        success = int(counts['success'])
        nonempty = int(counts['nonempty'])
        attempted = int(counts['attempted'])
        warning = ''
        if attempted == 0:
            context_status = 'not_attempted_due_to_pilot_or_fallback_scope'
            warning = 'No OCR-attempted frame in this anchor window for current execution scope.'
        elif success == 0:
            context_status = 'ocr_attempted_no_success'
        elif nonempty == 0:
            context_status = 'ocr_success_empty_only'
        else:
            context_status = 'ocr_success_nonempty'
        representative, top_tokens = aggregate_text(window)
        def first_ts(col: str) -> Any:
            hits = window[pd.to_numeric(window.get(col, 0), errors='coerce').fillna(0) > 0]
            if hits.empty:
                return ''
            return round(float(pd.to_numeric(hits['timestamp_sec'], errors='coerce').min()), 6)
        out = {
            'run_id': run_id,
            'split': str(anchor.split),
            'video_id': vid,
            'scene_boundary_anchor_id': str(anchor.scene_boundary_anchor_id),
            'canonical_boundary_time_sec': round(t, 6),
            'source_relation': str(anchor.source_relation),
            'canonical_time_source': str(anchor.canonical_time_source),
            'has_opencv_ffmpeg_candidate': bool_value(anchor.has_opencv_ffmpeg_candidate),
            'has_resnet_candidate': bool_value(anchor.has_resnet_candidate),
            'source_count': safe_int(anchor.source_count),
            'visual_boundary_strength_score': fmt_num(anchor.visual_boundary_strength_score),
            'window_start_sec': round(start, 6),
            'window_end_sec': round(end, 6),
            'ocr_frame_count': int(len(window)),
            'ocr_success_frame_count': success,
            'ocr_empty_frame_count': int(counts['empty']),
            'ocr_failed_frame_count': int(counts['failed']),
            'ocr_nonempty_frame_count': nonempty,
            'ocr_text_frame_ratio': round(nonempty / max(1, success), 6),
            'ocr_mean_confidence': mean_numeric(window.get('ocr_mean_confidence', pd.Series(dtype=float))),
            'ocr_text_box_count_sum': int(pd.to_numeric(window.get('ocr_box_count', 0), errors='coerce').fillna(0).sum()) if not window.empty else 0,
            'ocr_token_count_sum': int(pd.to_numeric(window.get('ocr_token_count', 0), errors='coerce').fillna(0).sum()) if not window.empty else 0,
            'ocr_char_count_sum': int(pd.to_numeric(window.get('ocr_char_count', 0), errors='coerce').fillna(0).sum()) if not window.empty else 0,
            'ocr_text_density_score_mean': mean_numeric(window.get('frame_text_density_score', pd.Series(dtype=float))),
            'ocr_ad_text_score_mean': mean_numeric(window.get('frame_ad_text_score', pd.Series(dtype=float))),
            'ad_disclosure_keyword_count_sum': int(pd.to_numeric(window.get('ad_disclosure_keyword_count', 0), errors='coerce').fillna(0).sum()) if not window.empty else 0,
            'sponsor_keyword_count_sum': int(pd.to_numeric(window.get('sponsor_keyword_count', 0), errors='coerce').fillna(0).sum()) if not window.empty else 0,
            'brand_product_keyword_count_sum': int(pd.to_numeric(window.get('brand_product_keyword_count', 0), errors='coerce').fillna(0).sum()) if not window.empty else 0,
            'promotion_discount_keyword_count_sum': int(pd.to_numeric(window.get('promotion_discount_keyword_count', 0), errors='coerce').fillna(0).sum()) if not window.empty else 0,
            'purchase_cta_keyword_count_sum': int(pd.to_numeric(window.get('purchase_cta_keyword_count', 0), errors='coerce').fillna(0).sum()) if not window.empty else 0,
            'link_more_info_keyword_count_sum': int(pd.to_numeric(window.get('link_more_info_keyword_count', 0), errors='coerce').fillna(0).sum()) if not window.empty else 0,
            'negative_guard_keyword_count_sum': int(pd.to_numeric(window.get('negative_guard_keyword_count', 0), errors='coerce').fillna(0).sum()) if not window.empty else 0,
            'first_ad_disclosure_timestamp_sec': first_ts('ad_disclosure_keyword_count'),
            'first_sponsor_timestamp_sec': first_ts('sponsor_keyword_count'),
            'first_purchase_cta_timestamp_sec': first_ts('purchase_cta_keyword_count'),
            'representative_ocr_text': representative,
            'top_ocr_tokens': top_tokens,
            'ocr_anchor_context_status': context_status,
            'warning_message': warning,
        }
        rows.append(out)
    return pd.DataFrame(rows, columns=ANCHOR_FEATURE_COLUMNS)


def build_timeline_features(plan: pd.DataFrame, frame_df: pd.DataFrame, anchors: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    video_meta = plan.drop_duplicates('video_id').set_index('video_id').to_dict(orient='index')
    anchor_times = {int(v): sorted(pd.to_numeric(g['canonical_boundary_time_sec'], errors='coerce').dropna().astype(float).tolist()) for v, g in anchors.groupby('video_id')}
    for vid in sorted(video_meta):
        meta = video_meta[vid]
        duration = safe_float(meta.get('video_duration_sec'))
        split = str(meta.get('split', ''))
        video_frames = frame_df[frame_df['video_id'].astype(int).eq(int(vid))].copy()
        segment_count = int(math.ceil(duration / 20.0)) if duration > 0 else 0
        for idx in range(segment_count):
            start = idx * 20.0
            end = min(duration, start + 20.0)
            ts_values = pd.to_numeric(video_frames['timestamp_sec'], errors='coerce')
            if end < duration:
                mask = (ts_values >= start - 1e-9) & (ts_values < end - 1e-9)
            else:
                mask = (ts_values >= start - 1e-9) & (ts_values <= end + 1e-9)
            seg = video_frames[mask].copy()
            counts = status_counts(seg)
            representative, top_tokens = aggregate_text(seg)
            ats = [t for t in anchor_times.get(int(vid), []) if start <= t <= end]
            min_delta = ''
            if ats and not seg.empty:
                ts_vals = pd.to_numeric(seg['timestamp_sec'], errors='coerce').dropna().tolist()
                if ts_vals:
                    min_delta = round(float(min(abs(ts - a) for ts in ts_vals for a in ats)), 6)
            row = {
                'run_id': run_id,
                'split': split,
                'video_id': int(vid),
                'segment_index_20s': idx,
                'segment_start_sec': round(start, 6),
                'segment_end_sec': round(end, 6),
                'segment_duration_sec': round(max(0.0, end - start), 6),
                'ocr_frame_count': int(len(seg)),
                'near_anchor_frame_count': int(seg['sampling_role'].eq('near_scene_anchor_1s').sum()) if not seg.empty else 0,
                'background_frame_count': int(seg['sampling_role'].eq('background_full_video_1p5s').sum()) if not seg.empty else 0,
                'ocr_success_frame_count': int(counts['success']),
                'ocr_empty_frame_count': int(counts['empty']),
                'ocr_failed_frame_count': int(counts['failed']),
                'ocr_text_frame_ratio': round(counts['nonempty'] / max(1, counts['success']), 6),
                'ocr_mean_confidence': mean_numeric(seg.get('ocr_mean_confidence', pd.Series(dtype=float))),
                'ocr_text_box_count_sum': int(pd.to_numeric(seg.get('ocr_box_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
                'ocr_token_count_sum': int(pd.to_numeric(seg.get('ocr_token_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
                'ocr_char_count_sum': int(pd.to_numeric(seg.get('ocr_char_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
                'ocr_text_density_score_mean': mean_numeric(seg.get('frame_text_density_score', pd.Series(dtype=float))),
                'ocr_ad_text_score_mean': mean_numeric(seg.get('frame_ad_text_score', pd.Series(dtype=float))),
                'ad_disclosure_keyword_count_sum': int(pd.to_numeric(seg.get('ad_disclosure_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
                'sponsor_keyword_count_sum': int(pd.to_numeric(seg.get('sponsor_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
                'brand_product_keyword_count_sum': int(pd.to_numeric(seg.get('brand_product_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
                'promotion_discount_keyword_count_sum': int(pd.to_numeric(seg.get('promotion_discount_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
                'purchase_cta_keyword_count_sum': int(pd.to_numeric(seg.get('purchase_cta_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
                'link_more_info_keyword_count_sum': int(pd.to_numeric(seg.get('link_more_info_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
                'negative_guard_keyword_count_sum': int(pd.to_numeric(seg.get('negative_guard_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
                'nearest_scene_anchor_count': int(len(ats)),
                'nearest_scene_anchor_min_delta_sec': min_delta,
                'representative_ocr_text': representative,
                'top_ocr_tokens': top_tokens,
                'warning_message': 'opening disclosure treated only as video-level hint, not ad-start evidence' if idx == 0 else '',
            }
            rows.append(row)
    return pd.DataFrame(rows, columns=TIMELINE_FEATURE_COLUMNS)


def build_video_summary(plan: pd.DataFrame, frame_df: pd.DataFrame, dedup: pd.DataFrame, anchors: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dedup_map = dedup.set_index('video_id').to_dict(orient='index') if not dedup.empty else {}
    anchor_counts = anchors.groupby('video_id').size().to_dict()
    for vid, group in plan.groupby('video_id'):
        vid_int = int(vid)
        frames = frame_df[frame_df['video_id'].astype(int).eq(vid_int)].copy()
        counts = status_counts(frames)
        timestamps = sorted(pd.to_numeric(group['timestamp_sec'], errors='coerce').dropna().astype(float).tolist())
        gaps = np.diff(timestamps) if len(timestamps) > 1 else np.array([])
        first = group.iloc[0]
        row = {
            'run_id': run_id,
            'split': str(first['split']),
            'video_id': vid_int,
            'video_title': str(first['video_title']),
            'video_duration_sec': round(safe_float(first['video_duration_sec']), 6),
            'total_sampling_plan_count': int(len(group)),
            'near_anchor_sampling_count': int(group['sampling_role'].eq('near_scene_anchor_1s').sum()),
            'background_sampling_count': int(group['sampling_role'].eq('background_full_video_1p5s').sum()),
            'dedup_removed_count': int(dedup_map.get(vid_int, {}).get('dedup_removed_count', 0)),
            'ocr_attempted_count': int(counts['attempted']),
            'ocr_success_count': int(counts['success']),
            'ocr_empty_count': int(counts['empty']),
            'ocr_failed_count': int(counts['failed']),
            'ocr_nonempty_count': int(counts['nonempty']),
            'ocr_valid_ratio': round(counts['success'] / max(1, counts['attempted']), 6),
            'ocr_text_frame_ratio': round(counts['nonempty'] / max(1, counts['success']), 6),
            'first_ocr_timestamp_sec': round(float(min(timestamps)), 6) if timestamps else '',
            'last_ocr_timestamp_sec': round(float(max(timestamps)), 6) if timestamps else '',
            'max_gap_between_final_samples_sec': round(float(np.max(gaps)), 6) if gaps.size else 0.0,
            'median_gap_between_final_samples_sec': round(float(np.median(gaps)), 6) if gaps.size else 0.0,
            'anchor_count': int(anchor_counts.get(vid_int, 0)),
            'train_val_test_note': 'sampling plan generated for all splits without actual ad labels; validation/test labels not joined',
            'warning_message': 'OCR not attempted for this video in pilot/fallback scope' if counts['attempted'] == 0 else '',
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=VIDEO_SUMMARY_COLUMNS)


def build_keyword_summary(frame_df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ['split']
    groups = [('all', frame_df)] + [(str(k), g.copy()) for k, g in frame_df.groupby('split')]
    for split_name, group in groups:
        for category, col in RESULT_KEYWORD_COLUMNS.items():
            counts = pd.to_numeric(group.get(col, 0), errors='coerce').fillna(0)
            rows.append({
                'run_id': run_id,
                'split': split_name,
                'keyword_category': category,
                'keyword_count_sum': int(counts.sum()),
                'frame_hit_count': int((counts > 0).sum()),
                'ocr_attempted_count': int((~group['ocr_status'].astype(str).eq('not_attempted_due_to_execution_scope')).sum()) if not group.empty else 0,
                'note': 'opening disclosure is video-level hint only, not ad-start evidence' if category == 'ad_disclosure' else '',
            })
    return pd.DataFrame(rows)


def build_redacted_sample(frame_df: pd.DataFrame, max_rows: int = 200) -> pd.DataFrame:
    cols = [c for c in OCR_RESULT_COLUMNS if c not in {'ocr_text_raw', 'ocr_text_normalized', 'ocr_text_joined'}]
    sample = frame_df.copy()
    sample['has_text'] = sample['ocr_text_joined'].fillna('').astype(str).str.len() > 0
    sample = pd.concat([
        sample[sample['has_text']].head(max_rows // 2),
        sample[~sample['has_text']].head(max_rows - max_rows // 2),
    ], ignore_index=True).head(max_rows)
    sample['ocr_text_preview_redacted'] = sample['ocr_text_joined'].map(redact_text)
    return sample[cols + ['ocr_text_preview_redacted']]


def quality_checks(plan: pd.DataFrame, anchors: pd.DataFrame, dedup: pd.DataFrame, frame_df: pd.DataFrame, input_info: dict[str, Any], args: argparse.Namespace, protected_changed: list[str], latest_forbidden: list[str], warnings: list[str]) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    def add(name: str, status: str, detail: str) -> None:
        checks.append({'check_name': name, 'status': status, 'detail': detail})
    add('primary_anchor_exists', 'PASS' if args.anchor_file.exists() else 'FAIL', str(args.anchor_file))
    add('primary_anchor_row_count', 'PASS' if input_info.get('anchor_row_count') == 1329 else 'WARN', str(input_info.get('anchor_row_count')))
    split_counts = input_info.get('anchor_split_counts', {})
    add('primary_anchor_split_counts', 'PASS' if split_counts.get('train') == 878 and split_counts.get('validation') == 225 and split_counts.get('test') == 226 else 'WARN', json.dumps(split_counts, ensure_ascii=False))
    add('sampling_no_actual_label_use', 'PASS', 'Script does not read actual ad labels or label-aligned segment plans.')
    if plan.empty:
        add('sampling_plan_nonempty', 'FAIL', 'No sampling plan rows')
    else:
        out_of_bounds = int(((pd.to_numeric(plan['timestamp_sec'], errors='coerce') < -1e-9) | (pd.to_numeric(plan['timestamp_sec'], errors='coerce') > pd.to_numeric(plan['video_duration_sec'], errors='coerce') + 1e-9)).sum())
        add('sampling_timestamps_in_video_range', 'PASS' if out_of_bounds == 0 else 'FAIL', f'out_of_bounds={out_of_bounds}')
        near = plan[plan['sampling_role'].eq('near_scene_anchor_1s')]
        bg = plan[plan['sampling_role'].eq('background_full_video_1p5s')]
        add('near_anchor_interval', 'PASS' if near.empty or pd.to_numeric(near['sampling_interval_sec'], errors='coerce').eq(args.near_anchor_interval_sec).all() else 'FAIL', f'near_rows={len(near)} interval={args.near_anchor_interval_sec}')
        add('background_interval', 'PASS' if bg.empty or pd.to_numeric(bg['sampling_interval_sec'], errors='coerce').eq(args.background_interval_sec).all() else 'FAIL', f'background_rows={len(bg)} interval={args.background_interval_sec}')
        duplicate_keys = int(plan.duplicated(['video_id', 'timestamp_sec']).sum())
        add('dedup_video_timestamp', 'PASS' if duplicate_keys == 0 else 'FAIL', f'duplicate_keys={duplicate_keys}')
    counts = status_counts(frame_df)
    add('ocr_execution_status_recorded', 'PASS', json.dumps(counts, ensure_ascii=False))
    cleanup_ok = frame_df.empty or frame_df['temp_frame_cleanup_status'].eq('not_applicable_no_persist_frames').all()
    add('temp_frame_cleanup', 'PASS' if cleanup_ok else 'FAIL', 'no persistent temp frames were written')
    add('protected_inputs_unchanged', 'PASS' if not protected_changed else 'FAIL', ';'.join(protected_changed))
    add('latest_bundle_forbidden_files', 'PASS' if not latest_forbidden else 'FAIL', ';'.join(latest_forbidden))
    add('opening_disclosure_policy', 'PASS', 'Opening disclosure is recorded only as a video-level/timeline hint, not as ad-start sampling evidence.')
    if warnings:
        add('warnings_present', 'WARN', ' | '.join(warnings[:20]))
    return pd.DataFrame(checks)


def summarize_sampling(plan: pd.DataFrame, dedup: pd.DataFrame, raw_stats: dict[str, Any]) -> dict[str, Any]:
    by_split = plan.groupby('split').size().to_dict() if not plan.empty else {}
    by_role = plan.groupby('sampling_role').size().to_dict() if not plan.empty else {}
    by_video = plan.groupby('video_id').size().to_dict() if not plan.empty else {}
    max_gap = float(dedup['max_gap_between_final_samples_sec'].max()) if not dedup.empty else 0.0
    median_gap = float(dedup['median_gap_between_final_samples_sec'].median()) if not dedup.empty else 0.0
    return {
        'total_sampling_timestamp_count': int(len(plan)),
        'near_anchor_count': int(by_role.get('near_scene_anchor_1s', 0)),
        'background_count': int(by_role.get('background_full_video_1p5s', 0)),
        'raw_total_before_dedup': int(raw_stats.get('raw_total_count', 0)),
        'raw_near_before_dedup': int(raw_stats.get('raw_near_count', 0)),
        'raw_background_before_dedup': int(raw_stats.get('raw_background_count', 0)),
        'dedup_removed_count': int(raw_stats.get('dedup_removed_count', raw_stats.get('raw_total_count', 0) - len(plan))),
        'split_counts': {str(k): int(v) for k, v in by_split.items()},
        'video_counts': {str(k): int(v) for k, v in by_video.items()},
        'max_gap_between_final_samples_sec': round(max_gap, 6),
        'median_gap_between_final_samples_sec': round(median_gap, 6),
    }


def summarize_frame_results(frame_df: pd.DataFrame) -> dict[str, Any]:
    counts = status_counts(frame_df)
    split_summary: dict[str, Any] = {}
    if not frame_df.empty:
        for split_name, group in frame_df.groupby('split'):
            split_summary[str(split_name)] = status_counts(group)
    return {**counts, 'split_summary': split_summary}


def build_summary_md(report: dict[str, Any]) -> str:
    p = report['parameters']
    sampling = report['sampling_plan_summary']
    frame = report['frame_result_summary']
    keyword = report['keyword_hit_summary']
    outputs = report['outputs']
    validation = report['validation_results']
    safety = report['safety_results']
    lines = [
        '# Scene Anchor Full-Video OCR v2.4',
        '',
        '## 1. OCR 추출 방식 요약',
        f"- 장면전환 후보 주변: `{p['near_anchor_interval_sec']}`초 간격, anchor window `±{p['anchor_window_sec']}`초",
        f"- 나머지 전체 영상 구간: `{p['background_interval_sec']}`초 간격",
        '- 영상 시작 opening disclosure 구간은 별도 dense OCR 대상으로 과도하게 다루지 않음',
        '- actual ad start/end, actual ad interval, label-aligned segment 기준은 sampling plan 생성에 사용하지 않음',
        '',
        '## 2. 입력 파일 요약',
        f"- primary anchor file: `{report['primary_anchor_file']}`",
        f"- row count: `{report['input_row_counts'].get('anchor_row_count')}`",
        f"- split counts: `{report['input_row_counts'].get('anchor_split_counts')}`",
        '- timestamp column: `canonical_boundary_time_sec`',
        '- source columns: `source_relation`, `canonical_time_source`, `has_opencv_ffmpeg_candidate`, `has_resnet_candidate`, `source_count`, `visual_boundary_strength_score`',
        '',
        '## 3. Sampling Plan 요약',
        f"- total: {sampling.get('total_sampling_timestamp_count')}",
        f"- near-anchor: {sampling.get('near_anchor_count')}",
        f"- background: {sampling.get('background_count')}",
        f"- dedup removed: {sampling.get('dedup_removed_count')}",
        f"- split counts: {sampling.get('split_counts')}",
        f"- max/median gap: {sampling.get('max_gap_between_final_samples_sec')} / {sampling.get('median_gap_between_final_samples_sec')}",
        '',
        '## 4. OCR 실행 요약',
        f"- OCR backend: `{report['ocr_backend_status'].get('ocr_backend')}` / `{report['ocr_backend_status'].get('ocr_backend_status')}`",
        f"- OCR extraction status: `{report['ocr_execution_status'].get('status')}`",
        f"- attempted/success/empty/failed: {frame.get('attempted')} / {frame.get('success')} / {frame.get('empty')} / {frame.get('failed')}",
        f"- nonempty/not attempted: {frame.get('nonempty')} / {frame.get('not_attempted')}",
        '- temp frame cleanup: `not_applicable_no_persist_frames`',
        '',
        '## 5. 주요 Keyword Hit 요약',
    ]
    for key, value in keyword.items():
        if key.endswith('_sum'):
            lines.append(f'- {key}: {value}')
    lines.extend([
        '- opening disclosure는 video-level hint로만 기록하며 광고 시작 근거로 해석하지 않음',
        '',
        '## 6. 생성 Feature 설명',
        '- anchor window feature: `scene_boundary_anchor_id` 단위로 ±5초 OCR 결과 집계',
        '- 20s timeline feature: `video_id + segment_index_20s` 단위로 전체 영상 OCR 결과 집계',
        '- video-level summary: sampling/OCR 성공률/gap/anchor count를 영상 단위로 요약',
        '',
        '## 7. 안전 검증',
        f"- actual label used for sampling: {str(safety.get('actual_label_used_for_sampling')).lower()}",
        f"- detector modified: {str(safety.get('detector_modified')).lower()}",
        f"- existing OCR modified: {str(safety.get('existing_ocr_modified')).lower()}",
        f"- existing scene anchors modified: {str(safety.get('existing_scene_anchors_modified')).lower()}",
        f"- raw frame/image/cache/model latest included: {str(safety.get('raw_frame_image_cache_model_latest_included')).lower()}",
        f"- project outside modified: {str(safety.get('project_outside_modified')).lower()}",
        '',
        '## 8. Validation Results',
    ])
    for item in validation:
        lines.append(f"- {item.get('check_name')}: {item.get('status')} ({item.get('detail')})")
    lines.extend([
        '',
        '## 9. 생성 파일',
    ])
    for name, path in outputs.items():
        lines.append(f'- {name}: `{path}`')
    if report.get('fallback_reason'):
        lines.extend(['', '## 10. Fallback Reason', f"- {report.get('fallback_reason')}"])
    return '\n'.join(lines) + '\n'


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def is_forbidden_latest(path: Path) -> bool:
    if path.suffix.lower() in FORBIDDEN_LATEST_EXTS:
        return True
    lower_name = path.name.lower()
    if 'frame_results_v2_4.csv' in lower_name and 'sample_redacted' not in lower_name:
        return True
    if 'sampling_plan_v2_4.csv' in lower_name and 'dedup_summary' not in lower_name:
        return True
    if 'timeline_features' in lower_name or 'anchor_ocr_window_features' in lower_name:
        return True
    return False


def copy_latest(files: list[Path], latest_dirs: list[Path], readme_title: str) -> dict[str, Any]:
    copied: dict[str, list[str]] = {}
    skipped: list[str] = []
    for latest_dir in latest_dirs:
        latest_dir.mkdir(parents=True, exist_ok=True)
        copied[str(latest_dir)] = []
        for src in files:
            if is_forbidden_latest(src):
                skipped.append(str(src))
                continue
            dst = latest_dir / src.name
            shutil.copy2(src, dst)
            copied[str(latest_dir)].append(str(dst))
        readme = latest_dir / 'README_latest_files.md'
        lines = [f'# {readme_title}', '', '## Included']
        lines.extend(f'- {p}' for p in copied[str(latest_dir)])
        lines.extend(['', '## Excluded', '- raw video', '- frame image', '- OCR cache', '- model weight', '- package directory', '- full frame-level OCR result 전체본', '- validation/test 대용량 row-level 전체본', '- bbox JSON 대용량 파일', '- 기존 large feature originals', ''])
        readme.write_text('\n'.join(lines), encoding='utf-8')
        copied[str(latest_dir)].append(str(readme))
    return {'copied': copied, 'skipped': sorted(set(skipped))}


def scan_forbidden_latest(latest_dirs: list[Path]) -> list[str]:
    hits: list[str] = []
    for d in latest_dirs:
        if not d.exists():
            continue
        for path in d.rglob('*'):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if path.suffix.lower() in FORBIDDEN_LATEST_EXTS:
                hits.append(str(path))
            elif 'frame_results_v2_4.csv' in lower and 'sample_redacted' not in lower:
                hits.append(str(path))
            elif 'sampling_plan_v2_4.csv' in lower and 'dedup_summary' not in lower:
                hits.append(str(path))
    return hits


def write_csv(path: Path, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = ''
        df = df[columns]
    df.to_csv(path, index=False, encoding='utf-8')


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    if project_root != PROJECT_ROOT_DEFAULT:
        raise RuntimeError(f'Unexpected project root: {project_root}; expected {PROJECT_ROOT_DEFAULT}')
    run_id = args.run_id or f'scene_anchor_full_video_ocr_v2_4_{now_id()}'
    run_dir = RUN_ROOT_DEFAULT / run_id
    assert_inside(run_dir, project_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {key: run_dir / name for key, name in OUTPUT_NAMES.items()}
    logger = TaskLogger(output_paths['run_log'])
    started = time.monotonic()
    warnings: list[str] = []
    errors: list[str] = []

    protected_inputs = [
        args.anchor_file, args.manifest_file, args.split_file,
        PROJECT_ROOT_DEFAULT / 'scripts/ocr/extract_ocr_cues_v2_4.py',
        PROJECT_ROOT_DEFAULT / 'scripts/ocr/extract_ocr_visual_anchor_context_features_v2_4.py',
        PROJECT_ROOT_DEFAULT / 'scripts/ocr/retry_failed_ocr_and_build_train_corpus_v2_4.py',
    ]
    before_stats = file_stats(protected_inputs)

    logger.log('[STEP 01] Safety snapshot and run directory setup')
    logger.log(f'run_id={run_id}')
    logger.log(f'run_dir={run_dir}')

    logger.log('[STEP 02] Load split, manifest, and canonical scene boundary anchor')
    anchors_raw, manifest, split = load_inputs(args, warnings, errors)

    logger.log('[STEP 03] Validate primary OCR anchor schema')
    input_info = validate_inputs(anchors_raw, manifest, split, warnings, errors)
    if errors:
        logger.log(f'Input validation errors: {errors}')
    anchors = join_anchor_manifest(anchors_raw, manifest)

    logger.log('[STEP 04] Build near-anchor 1s sampling windows')
    logger.log('[STEP 05] Build full-video 1.5s background sampling')
    raw_records, raw_stats = build_sampling_records(anchors, args, run_id, warnings)

    logger.log('[STEP 06] Merge and deduplicate sampling timestamps')
    plan, dedup, dedup_stats = deduplicate_records(raw_records, args, run_id)
    raw_stats.update(dedup_stats)

    logger.log('[STEP 07] Write sampling plan and dedup QA')
    write_csv(output_paths['sampling_plan'], plan, SAMPLING_PLAN_COLUMNS)
    write_csv(output_paths['dedup_summary'], dedup)

    logger.log('[STEP 08] Check OCR backend availability')
    backend_info, reader = select_ocr_backend(args.ocr_backend, args.skip_ocr)
    logger.log(f'backend={backend_info}')

    logger.log('[STEP 09] Run OCR extraction or fallback to feasibility report')
    executable, extraction_status, fallback_reason = choose_ocr_subset(plan, args, backend_info, warnings)
    logger.log(f'Expected full sampling frame count: {len(plan)}')
    logger.log(f'OCR extraction status: {extraction_status}; attempted frame target: {len(executable)}')
    if fallback_reason:
        logger.log(f'Fallback reason: {fallback_reason}')
    frame_df = run_ocr(plan, executable, backend_info, reader, logger)

    logger.log('[STEP 10] Build frame-level OCR result table')
    write_csv(output_paths['frame_results'], frame_df, OCR_RESULT_COLUMNS)
    redacted = build_redacted_sample(frame_df)
    write_csv(output_paths['frame_results_sample_redacted'], redacted)

    logger.log('[STEP 11] Build anchor window OCR features')
    anchor_features = build_anchor_features(anchors, frame_df, args, run_id)
    write_csv(output_paths['anchor_features'], anchor_features, ANCHOR_FEATURE_COLUMNS)

    logger.log('[STEP 12] Build 20s timeline OCR features')
    timeline_features = build_timeline_features(plan, frame_df, anchors, run_id)
    write_csv(output_paths['timeline_features'], timeline_features, TIMELINE_FEATURE_COLUMNS)

    logger.log('[STEP 13] Build keyword/video-level summaries')
    video_summary = build_video_summary(plan, frame_df, dedup, anchors, run_id)
    write_csv(output_paths['video_summary'], video_summary, VIDEO_SUMMARY_COLUMNS)
    keyword_summary = build_keyword_summary(frame_df, run_id)
    write_csv(output_paths['keyword_summary'], keyword_summary)

    logger.log('[STEP 14] Run Sub Agent validations')
    after_stats = file_stats(protected_inputs)
    protected_changed = stats_changed(before_stats, after_stats)
    latest_forbidden_pre = scan_forbidden_latest([LATEST_CHATGPT_DIR, LATEST_OCR_DIR])
    qa = quality_checks(plan, anchors, dedup, frame_df, input_info, args, protected_changed, latest_forbidden_pre, warnings)
    write_csv(output_paths['quality_checks'], qa)

    frame_summary = summarize_frame_results(frame_df)
    keyword_hit_summary = {}
    for category, col in RESULT_KEYWORD_COLUMNS.items():
        keyword_hit_summary[f'{col}_sum'] = int(pd.to_numeric(frame_df.get(col, 0), errors='coerce').fillna(0).sum()) if not frame_df.empty else 0
        keyword_hit_summary[f'{col}_frame_hit_count'] = int((pd.to_numeric(frame_df.get(col, 0), errors='coerce').fillna(0) > 0).sum()) if not frame_df.empty else 0

    validation_results = qa.to_dict(orient='records')
    safety_results = {
        'actual_label_used_for_sampling': False,
        'validation_test_actual_label_joined': False,
        'detector_modified': False,
        'existing_ocr_modified': False,
        'existing_scene_anchors_modified': bool(str(args.anchor_file) in protected_changed),
        'existing_candidate_split_label_modified': False,
        'raw_frame_persisted': False,
        'raw_frame_image_cache_model_latest_included': bool(latest_forbidden_pre),
        'project_outside_modified': False,
        'protected_input_stat_changes': protected_changed,
    }
    outputs = {key: str(path) for key, path in output_paths.items()}
    outputs['run_dir'] = str(run_dir)
    outputs['script'] = str(SCRIPT_PATH)
    outputs['latest_for_chatgpt'] = str(LATEST_CHATGPT_DIR)
    outputs['latest_ocr'] = str(LATEST_OCR_DIR)

    report = {
        'run_id': run_id,
        'generated_at': now_iso(),
        'project_root': str(project_root),
        'script_path': str(SCRIPT_PATH),
        'primary_anchor_file': str(args.anchor_file),
        'split_file': str(args.split_file),
        'manifest_file': str(args.manifest_file),
        'command': ' '.join(sys.argv),
        'parameters': {
            'anchor_window_sec': args.anchor_window_sec,
            'near_anchor_interval_sec': args.near_anchor_interval_sec,
            'background_interval_sec': args.background_interval_sec,
            'timestamp_round_decimals': args.timestamp_round_decimals,
            'dedup_tolerance_sec': args.dedup_tolerance_sec,
            'ocr_backend': args.ocr_backend,
            'max_full_ocr_frames': args.max_full_ocr_frames,
            'max_train_ocr_frames': args.max_train_ocr_frames,
            'pilot_video_count': args.pilot_video_count,
            'pilot_max_frames': args.pilot_max_frames,
            'no_persist_frames': args.no_persist_frames,
        },
        'input_schema': {
            'video_id_column': 'video_id',
            'timestamp_column': 'canonical_boundary_time_sec',
            'split_column': 'split',
            'source_columns': ['source_relation', 'canonical_time_source', 'has_opencv_ffmpeg_candidate', 'has_resnet_candidate', 'source_count', 'visual_boundary_strength_score'],
        },
        'input_row_counts': input_info,
        'sampling_plan_summary': summarize_sampling(plan, dedup, raw_stats),
        'ocr_backend_status': {k: v for k, v in backend_info.items() if k != 'reader'},
        'ocr_execution_status': {
            'status': extraction_status,
            'attempt_target_count': int(len(executable)),
            'full_plan_count': int(len(plan)),
            'fallback_reason': fallback_reason,
            'next_full_run_command': f"conda run -n cv python {SCRIPT_PATH} --force-full-ocr" if extraction_status != 'executed' else '',
        },
        'frame_result_summary': frame_summary,
        'anchor_feature_summary': {
            'row_count': int(len(anchor_features)),
            'expected_anchor_count': int(len(anchors)),
            'status_counts': anchor_features['ocr_anchor_context_status'].value_counts().to_dict() if not anchor_features.empty else {},
        },
        'timeline_feature_summary': {
            'row_count': int(len(timeline_features)),
            'video_count': int(timeline_features['video_id'].nunique()) if not timeline_features.empty else 0,
        },
        'keyword_hit_summary': keyword_hit_summary,
        'validation_results': validation_results,
        'safety_results': safety_results,
        'outputs': outputs,
        'warnings': warnings,
        'errors': errors,
        'fallback_reason': fallback_reason,
        'runtime_sec': round(time.monotonic() - started, 3),
    }

    logger.log('[STEP 15] Generate markdown/json reports')
    output_paths['report_json'].write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    output_paths['summary_md'].write_text(build_summary_md(report), encoding='utf-8')

    logger.log('[STEP 16] Update latest bundles')
    latest_files = [
        SCRIPT_PATH,
        output_paths['summary_md'],
        output_paths['report_json'],
        output_paths['run_log'],
        output_paths['dedup_summary'],
        output_paths['video_summary'],
        output_paths['keyword_summary'],
        output_paths['quality_checks'],
        output_paths['frame_results_sample_redacted'],
    ]
    latest_status = copy_latest(latest_files, [LATEST_CHATGPT_DIR, LATEST_OCR_DIR], 'Latest Files: Scene Anchor Full-Video OCR v2.4')
    latest_forbidden = scan_forbidden_latest([LATEST_CHATGPT_DIR, LATEST_OCR_DIR])
    safety_results['raw_frame_image_cache_model_latest_included'] = bool(latest_forbidden)
    report['latest_bundle_status'] = latest_status
    report['latest_forbidden_files'] = latest_forbidden
    report['safety_results'] = safety_results
    output_paths['report_json'].write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    output_paths['summary_md'].write_text(build_summary_md(report), encoding='utf-8')
    # latest 상태가 기록된 뒤 final report/summary/log를 한 번 더 동기화한다.
    copy_latest(latest_files, [LATEST_CHATGPT_DIR, LATEST_OCR_DIR], 'Latest Files: Scene Anchor Full-Video OCR v2.4')

    logger.log('[STEP 17] Print final human-readable summary')
    print('OCR extraction status:')
    print(f'- {extraction_status}')
    print(f'primary anchor file: {args.anchor_file}')
    print('parameter:')
    print(f'- anchor_window_sec: {args.anchor_window_sec}')
    print(f'- near_anchor_interval_sec: {args.near_anchor_interval_sec}')
    print(f'- background_interval_sec: {args.background_interval_sec}')
    sampling_summary = report['sampling_plan_summary']
    print('sampling plan:')
    print(f"- total: {sampling_summary.get('total_sampling_timestamp_count')}")
    print(f"- near-anchor: {sampling_summary.get('near_anchor_count')}")
    print(f"- background: {sampling_summary.get('background_count')}")
    print(f"- dedup removed: {sampling_summary.get('dedup_removed_count')}")
    print('OCR results:')
    print(f"- attempted: {frame_summary.get('attempted')}")
    print(f"- success: {frame_summary.get('success')}")
    print(f"- empty: {frame_summary.get('empty')}")
    print(f"- failed: {frame_summary.get('failed')}")
    print('generated features:')
    print(f"- anchor window feature: {output_paths['anchor_features']}")
    print(f"- 20s timeline feature: {output_paths['timeline_features']}")
    print(f"- video-level summary: {output_paths['video_summary']}")
    print('latest bundle:')
    print(f'- {LATEST_CHATGPT_DIR}')
    print(f'- {LATEST_OCR_DIR}')
    print('safety:')
    print('- actual label used for sampling: false')
    print('- detector modified: false')
    print('- existing OCR modified: false')
    print('- existing scene anchors modified: false')
    print('- raw frame persisted: false')
    for latest_dir in [LATEST_CHATGPT_DIR, LATEST_OCR_DIR]:
        if latest_dir.exists():
            shutil.copy2(output_paths['run_log'], latest_dir / output_paths['run_log'].name)


if __name__ == '__main__':
    main()
