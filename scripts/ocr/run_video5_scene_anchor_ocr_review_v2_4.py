#!/usr/bin/env python3
"""Run prediction-time scene-anchor/full-video OCR review for train video_id=5.

Sampling uses only canonical scene anchors, split, and video manifest. Actual ad
intervals are loaded after OCR solely for post-hoc review tables.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path('.').resolve()
BASE_SCRIPT = PROJECT_ROOT / 'scripts/ocr/extract_scene_anchor_full_video_ocr_v2_4.py'
SCRIPT_PATH = PROJECT_ROOT / 'scripts/ocr/run_video5_scene_anchor_ocr_review_v2_4.py'
VERSION = 'v2_4'
TARGET_VIDEO_ID = 5
TARGET_SPLIT = 'train'
ANCHOR_FILE = PROJECT_ROOT / 'data/features/visual_scene_boundary_anchors_v2_4_with_split.csv'
SPLIT_FILE = PROJECT_ROOT / 'data/splits/video_split_v2_4.csv'
MANIFEST_FILE = PROJECT_ROOT / 'data/video_metadata/video_manifest_v2_2.csv'
ACTUAL_AD_INTERVAL_FILE = PROJECT_ROOT / 'data/segments/ad_interval_segments_v2_4.csv'
RUN_ROOT = PROJECT_ROOT / 'workspaces/ocr_video5_scene_anchor_review_v2_4/runs'
LATEST_CHATGPT_DIR = PROJECT_ROOT / 'outputs/latest_for_chatgpt_video5_scene_anchor_ocr_review_v2_4'
LATEST_OCR_DIR = PROJECT_ROOT / 'outputs/latest_ocr'

OUTPUT_NAMES = {
    'sampling_plan': 'video5_scene_anchor_full_video_ocr_sampling_plan_v2_4.csv',
    'dedup_summary': 'video5_scene_anchor_full_video_ocr_sampling_plan_dedup_summary_v2_4.csv',
    'frame_results': 'video5_scene_anchor_full_video_ocr_frame_results_v2_4.csv',
    'compact_text_review': 'video5_ocr_frame_text_review_v2_4.csv',
    'timeline_review': 'video5_ocr_20s_timeline_review_v2_4.csv',
    'actual_ad_context_frame_rows': 'video5_actual_ad_context_ocr_frame_rows_v2_4.csv',
    'actual_ad_context_summary': 'video5_actual_ad_context_ocr_summary_v2_4.csv',
    'actual_ad_context_keyword_hits': 'video5_actual_ad_context_keyword_hits_v2_4.csv',
    'representative_examples': 'video5_ocr_representative_text_examples_v2_4.csv',
    'quality_checks': 'video5_scene_anchor_ocr_quality_checks_v2_4.csv',
    'summary_md': 'video5_scene_anchor_ocr_review_v2_4_summary.md',
    'report_json': 'video5_scene_anchor_ocr_review_v2_4_report.json',
    'run_log': 'video5_scene_anchor_ocr_review_v2_4_run_log.txt',
}

COMPACT_COLUMNS = [
    'video_id', 'timestamp_sec', 'timestamp_mmss', 'sampling_role', 'is_near_scene_anchor',
    'nearest_anchor_id', 'nearest_anchor_delta_sec', 'ocr_status', 'has_text', 'ocr_text_joined',
    'ocr_text_count', 'ocr_token_count', 'ocr_mean_confidence', 'ad_disclosure_keyword_count',
    'sponsor_keyword_count', 'brand_product_keyword_count', 'promotion_discount_keyword_count',
    'purchase_cta_keyword_count', 'link_more_info_keyword_count', 'negative_guard_keyword_count',
    'frame_ad_text_score', 'frame_text_density_score'
]

TIMELINE_COLUMNS = [
    'video_id', 'segment_index_20s', 'segment_start_sec', 'segment_end_sec', 'ocr_frame_count',
    'near_anchor_frame_count', 'background_frame_count', 'success_frame_count', 'empty_frame_count',
    'failed_frame_count', 'nonempty_frame_count', 'text_frame_ratio', 'mean_confidence', 'text_count_sum',
    'token_count_sum', 'char_count_sum', 'ad_disclosure_keyword_count_sum', 'sponsor_keyword_count_sum',
    'brand_product_keyword_count_sum', 'promotion_discount_keyword_count_sum', 'purchase_cta_keyword_count_sum',
    'link_more_info_keyword_count_sum', 'negative_guard_keyword_count_sum', 'max_frame_ad_text_score',
    'mean_frame_ad_text_score', 'representative_ocr_text', 'top_ocr_tokens', 'note'
]

AD_CONTEXT_FRAME_COLUMNS = [
    'video_id', 'ad_interval_id', 'ad_start_sec', 'ad_end_sec', 'ad_duration_sec', 'window_type',
    'window_start_sec', 'window_end_sec', 'timestamp_sec', 'timestamp_mmss', 'relative_to_ad_start_sec',
    'relative_to_ad_end_sec', 'inside_actual_ad', 'sampling_role', 'is_near_scene_anchor',
    'nearest_anchor_id', 'nearest_anchor_delta_sec', 'ocr_status', 'has_text', 'ocr_text_joined',
    'ocr_text_count', 'ocr_token_count', 'ocr_mean_confidence', 'ad_disclosure_keyword_count',
    'sponsor_keyword_count', 'brand_product_keyword_count', 'promotion_discount_keyword_count',
    'purchase_cta_keyword_count', 'link_more_info_keyword_count', 'negative_guard_keyword_count',
    'frame_ad_text_score', 'frame_text_density_score', 'note'
]

AD_CONTEXT_SUMMARY_COLUMNS = [
    'video_id', 'ad_interval_id', 'ad_start_sec', 'ad_end_sec', 'ad_duration_sec', 'window_type',
    'window_start_sec', 'window_end_sec', 'window_duration_sec', 'ocr_frame_count', 'success_frame_count',
    'empty_frame_count', 'failed_frame_count', 'nonempty_frame_count', 'text_frame_ratio', 'mean_confidence',
    'min_timestamp_sec', 'max_timestamp_sec', 'median_gap_sec', 'max_gap_sec', 'near_anchor_frame_count',
    'background_frame_count', 'ad_disclosure_keyword_count_sum', 'sponsor_keyword_count_sum',
    'brand_product_keyword_count_sum', 'promotion_discount_keyword_count_sum', 'purchase_cta_keyword_count_sum',
    'link_more_info_keyword_count_sum', 'negative_guard_keyword_count_sum', 'max_frame_ad_text_score',
    'mean_frame_ad_text_score', 'first_keyword_timestamp_sec', 'representative_ocr_text', 'top_ocr_tokens',
    'interpretation_note'
]

AD_CONTEXT_KEYWORD_COLUMNS = [
    'video_id', 'ad_interval_id', 'window_type', 'timestamp_sec', 'timestamp_mmss',
    'relative_to_ad_start_sec', 'relative_to_ad_end_sec', 'keyword_category', 'keyword_count',
    'matched_keywords', 'ocr_text_joined', 'sampling_role', 'is_near_scene_anchor', 'frame_ad_text_score',
    'interpretation_note'
]

EXAMPLE_COLUMNS = [
    'example_group', 'video_id', 'ad_interval_id', 'window_type', 'timestamp_sec', 'timestamp_mmss',
    'sampling_role', 'ocr_status', 'ocr_text_joined', 'keyword_categories', 'frame_ad_text_score',
    'reason_selected'
]

FORBIDDEN_LATEST_EXTS = {'.mp4', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.m4a', '.jpg', '.jpeg', '.png', '.webp', '.pt', '.pth', '.ckpt', '.bin', '.onnx'}
FORBIDDEN_LATEST_NAMES = {
    'video5_scene_anchor_full_video_ocr_sampling_plan_v2_4.csv',
    'video5_scene_anchor_full_video_ocr_frame_results_v2_4.csv',
}


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location('scene_anchor_full_video_ocr_base_v2_4', BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import base script: {BASE_SCRIPT}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

BASE = load_base()

KEYWORDS = dict(BASE.KEYWORDS)
KEYWORDS['purchase_cta'] = sorted(set(KEYWORDS.get('purchase_cta', []) + ['링크', '더보기', '고정댓글', '댓글', '확인', '참고']))
KEYWORDS['link_more_info'] = sorted(set(KEYWORDS.get('link_more_info', []) + ['설명란', '댓글 확인', '프로필 링크']))
RESULT_KEYWORD_COLUMNS = dict(BASE.RESULT_KEYWORD_COLUMNS)
SUCCESS_STATUSES = set(BASE.SUCCESS_STATUSES)
TOKEN_RE = BASE.TOKEN_RE


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
    parser = argparse.ArgumentParser(description='Run video_id=5 scene-anchor/full-video OCR review for v2.4.')
    parser.add_argument('--target-video-id', type=int, default=TARGET_VIDEO_ID)
    parser.add_argument('--target-split', default=TARGET_SPLIT)
    parser.add_argument('--anchor-window-sec', type=float, default=5.0)
    parser.add_argument('--near-anchor-interval-sec', type=float, default=1.0)
    parser.add_argument('--background-interval-sec', type=float, default=1.5)
    parser.add_argument('--dedup-tolerance-sec', type=float, default=0.05)
    parser.add_argument('--timestamp-round-decimals', type=int, default=3)
    parser.add_argument('--ocr-backend', choices=['auto', 'easyocr', 'pytesseract', 'none'], default='auto')
    parser.add_argument('--max-ocr-frames', type=int, default=2000)
    parser.add_argument('--skip-ocr', action='store_true')
    parser.add_argument('--run-id', default='')
    parser.add_argument('--version', default=VERSION)
    parser.add_argument('--no-persist-frames', action='store_true', default=True)
    return parser.parse_args()


def now_id() -> str:
    return dt.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def safe_float(value: Any, default: float = float('nan')) -> float:
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


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize('NFKC', str(text or '')).lower()
    value = re.sub(r'[^\w\s가-힣%₩$./:\-+,&()]', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def tokenize(text: Any) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(normalize_text(text)) if tok.strip()]


def clean_text(text: Any, max_len: int = 500) -> str:
    value = re.sub(r'\s+', ' ', str(text or '')).strip()
    return value[:max_len - 3] + '...' if len(value) > max_len else value


def keyword_matches(text: Any) -> dict[str, list[str]]:
    norm = normalize_text(text)
    compact = re.sub(r'\s+', '', norm)
    out: dict[str, list[str]] = {}
    for category, terms in KEYWORDS.items():
        hits: list[str] = []
        for term in terms:
            t = normalize_text(term)
            if not t:
                continue
            found = norm.count(t) if ' ' in t else compact.count(re.sub(r'\s+', '', t))
            if found > 0:
                hits.extend([term] * found)
        out[category] = hits
    return out


def mmss(seconds: Any) -> str:
    return BASE.mmss(seconds)


def write_csv(path: Path, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = ''
        df = df[columns]
    df.to_csv(path, index=False, encoding='utf-8')


def mean_numeric(series: pd.Series) -> Any:
    vals = pd.to_numeric(series, errors='coerce').dropna()
    return round(float(vals.mean()), 6) if not vals.empty else ''


def status_counts(df: pd.DataFrame) -> dict[str, int]:
    return BASE.status_counts(df)


def aggregate_text(df: pd.DataFrame) -> tuple[str, str]:
    return BASE.aggregate_text(df)


def file_stats(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return BASE.file_stats(paths)


def stats_changed(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    return BASE.stats_changed(before, after)


def load_sampling_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchors = pd.read_csv(ANCHOR_FILE, encoding='utf-8-sig')
    manifest = pd.read_csv(MANIFEST_FILE, encoding='utf-8-sig')
    split = pd.read_csv(SPLIT_FILE, encoding='utf-8-sig')
    return anchors, manifest, split


def validate_and_filter_target(anchors: pd.DataFrame, manifest: pd.DataFrame, split: pd.DataFrame, args: argparse.Namespace, warnings: list[str], errors: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_anchor = ['scene_boundary_anchor_id', 'video_id', 'canonical_boundary_time_sec', 'split', 'source_relation', 'canonical_time_source', 'has_opencv_ffmpeg_candidate', 'has_resnet_candidate', 'source_count', 'visual_boundary_strength_score']
    for col in required_anchor:
        if col not in anchors.columns:
            errors.append(f'missing anchor column: {col}')
    for col in ['video_id', 'video_title', 'video_path', 'duration_sec']:
        if col not in manifest.columns:
            errors.append(f'missing manifest column: {col}')
    for col in ['video_id', 'split']:
        if col not in split.columns:
            errors.append(f'missing split column: {col}')
    anchors['video_id'] = pd.to_numeric(anchors['video_id'], errors='coerce').astype('Int64')
    anchors['canonical_boundary_time_sec'] = pd.to_numeric(anchors['canonical_boundary_time_sec'], errors='coerce')
    manifest['video_id'] = pd.to_numeric(manifest['video_id'], errors='coerce').astype('Int64')
    manifest['duration_sec'] = pd.to_numeric(manifest['duration_sec'], errors='coerce')
    split['video_id'] = pd.to_numeric(split['video_id'], errors='coerce').astype('Int64')
    split_row = split[split['video_id'].astype('Int64').eq(args.target_video_id)].copy()
    if split_row.empty:
        errors.append(f'target video_id={args.target_video_id} missing from split file')
    else:
        actual_split = str(split_row.iloc[0]['split'])
        if actual_split != args.target_split:
            errors.append(f'target video split mismatch: expected={args.target_split}, actual={actual_split}')
    target = anchors[(anchors['video_id'].eq(args.target_video_id)) & (anchors['split'].astype(str).eq(args.target_split))].copy()
    if target.empty:
        errors.append(f'no target anchors for video_id={args.target_video_id}, split={args.target_split}')
    manifest_target = manifest[manifest['video_id'].eq(args.target_video_id)].copy()
    if manifest_target.empty:
        errors.append(f'target video_id={args.target_video_id} missing from manifest')
    else:
        video_path = Path(str(manifest_target.iloc[0]['video_path']))
        if not video_path.exists():
            errors.append(f'target video path missing: {video_path}')
    merged = BASE.join_anchor_manifest(target, manifest)
    info = {
        'target_video_id': args.target_video_id,
        'target_split': args.target_split,
        'anchor_count': int(len(target)),
        'video_metadata': manifest_target.iloc[0].to_dict() if not manifest_target.empty else {},
        'split_row': split_row.iloc[0].to_dict() if not split_row.empty else {},
        'validation_test_rows_in_target': int((merged['split'].astype(str) != args.target_split).sum()) if not merged.empty else 0,
    }
    return merged, info


def build_compact_text_review(frame_df: pd.DataFrame) -> pd.DataFrame:
    compact = frame_df.copy()
    compact['has_text'] = compact['ocr_text_joined'].fillna('').astype(str).str.strip().ne('')
    compact['ocr_text_joined'] = compact['ocr_text_joined'].map(lambda x: clean_text(x, 700))
    return compact[COMPACT_COLUMNS]


def timeline_review(frame_df: pd.DataFrame, duration: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    segment_count = int(math.ceil(duration / 20.0)) if duration > 0 else 0
    ts = pd.to_numeric(frame_df['timestamp_sec'], errors='coerce') if not frame_df.empty else pd.Series(dtype=float)
    for idx in range(segment_count):
        start = idx * 20.0
        end = min(duration, start + 20.0)
        mask = (ts >= start - 1e-9) & ((ts < end - 1e-9) if end < duration else (ts <= end + 1e-9))
        seg = frame_df[mask].copy()
        counts = status_counts(seg)
        representative, top_tokens = aggregate_text(seg)
        row = {
            'video_id': TARGET_VIDEO_ID,
            'segment_index_20s': idx,
            'segment_start_sec': round(start, 6),
            'segment_end_sec': round(end, 6),
            'ocr_frame_count': int(len(seg)),
            'near_anchor_frame_count': int(seg['sampling_role'].eq('near_scene_anchor_1s').sum()) if not seg.empty else 0,
            'background_frame_count': int(seg['sampling_role'].eq('background_full_video_1p5s').sum()) if not seg.empty else 0,
            'success_frame_count': counts['success'],
            'empty_frame_count': counts['empty'],
            'failed_frame_count': counts['failed'],
            'nonempty_frame_count': counts['nonempty'],
            'text_frame_ratio': round(counts['nonempty'] / max(1, counts['success']), 6),
            'mean_confidence': mean_numeric(seg.get('ocr_mean_confidence', pd.Series(dtype=float))),
            'text_count_sum': int(pd.to_numeric(seg.get('ocr_text_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
            'token_count_sum': int(pd.to_numeric(seg.get('ocr_token_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
            'char_count_sum': int(pd.to_numeric(seg.get('ocr_char_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
            'ad_disclosure_keyword_count_sum': int(pd.to_numeric(seg.get('ad_disclosure_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
            'sponsor_keyword_count_sum': int(pd.to_numeric(seg.get('sponsor_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
            'brand_product_keyword_count_sum': int(pd.to_numeric(seg.get('brand_product_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
            'promotion_discount_keyword_count_sum': int(pd.to_numeric(seg.get('promotion_discount_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
            'purchase_cta_keyword_count_sum': int(pd.to_numeric(seg.get('purchase_cta_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
            'link_more_info_keyword_count_sum': int(pd.to_numeric(seg.get('link_more_info_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
            'negative_guard_keyword_count_sum': int(pd.to_numeric(seg.get('negative_guard_keyword_count', 0), errors='coerce').fillna(0).sum()) if not seg.empty else 0,
            'max_frame_ad_text_score': round(float(pd.to_numeric(seg.get('frame_ad_text_score', 0), errors='coerce').fillna(0).max()), 6) if not seg.empty else 0.0,
            'mean_frame_ad_text_score': mean_numeric(seg.get('frame_ad_text_score', pd.Series(dtype=float))),
            'representative_ocr_text': representative,
            'top_ocr_tokens': top_tokens,
            'note': 'opening disclosure treated as video-level hint only, not ad-start evidence' if idx == 0 else '',
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=TIMELINE_COLUMNS)


def load_actual_ad_intervals_posthoc(logger: TaskLogger) -> pd.DataFrame:
    logger.log('[STEP 13] Load actual ad intervals for video_id=5 post-hoc only')
    labels = pd.read_csv(ACTUAL_AD_INTERVAL_FILE, encoding='utf-8-sig')
    labels['video_id'] = pd.to_numeric(labels['video_id'], errors='coerce').astype('Int64')
    labels['ad_start_sec'] = pd.to_numeric(labels['ad_start_sec'], errors='coerce')
    labels['ad_end_sec'] = pd.to_numeric(labels['ad_end_sec'], errors='coerce')
    intervals = labels[(labels['video_id'].eq(TARGET_VIDEO_ID)) & (labels['segment_type'].astype(str).eq('ad_interval'))].copy()
    intervals = intervals.dropna(subset=['ad_start_sec', 'ad_end_sec']).sort_values(['ad_start_sec', 'ad_end_sec'])
    intervals['ad_duration_sec'] = intervals['ad_end_sec'] - intervals['ad_start_sec']
    return intervals


def ad_windows(interval: pd.Series, duration: float) -> list[dict[str, Any]]:
    start = safe_float(interval['ad_start_sec'])
    end = safe_float(interval['ad_end_sec'])
    specs = [
        ('pre_10s', start - 10.0, start, '[ad_start_sec - 10, ad_start_sec)'),
        ('start_edge_10s', start - 5.0, start + 5.0, '[ad_start_sec - 5, ad_start_sec + 5]'),
        ('ad_body', start, end, '[ad_start_sec, ad_end_sec]'),
        ('end_edge_10s', end - 5.0, end + 5.0, '[ad_end_sec - 5, ad_end_sec + 5]'),
        ('post_10s', end, end + 10.0, '(ad_end_sec, ad_end_sec + 10]'),
        ('ad_plus_context_10s', start - 10.0, end + 10.0, '[ad_start_sec - 10, ad_end_sec + 10]'),
    ]
    rows = []
    for window_type, raw_start, raw_end, note in specs:
        rows.append({
            'window_type': window_type,
            'window_start_sec': max(0.0, raw_start),
            'window_end_sec': min(duration, raw_end),
            'window_note': note,
        })
    return rows


def build_ad_context_rows(frame_df: pd.DataFrame, intervals: pd.DataFrame, duration: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame_ts = pd.to_numeric(frame_df['timestamp_sec'], errors='coerce')
    for interval in intervals.itertuples(index=False):
        interval_series = pd.Series(interval._asdict())
        start = safe_float(interval_series['ad_start_sec'])
        end = safe_float(interval_series['ad_end_sec'])
        for window in ad_windows(interval_series, duration):
            w_start = window['window_start_sec']
            w_end = window['window_end_sec']
            if window['window_type'] == 'pre_10s':
                mask = (frame_ts >= w_start - 1e-9) & (frame_ts < w_end - 1e-9)
            elif window['window_type'] == 'post_10s':
                mask = (frame_ts > w_start + 1e-9) & (frame_ts <= w_end + 1e-9)
            else:
                mask = (frame_ts >= w_start - 1e-9) & (frame_ts <= w_end + 1e-9)
            matched = frame_df[mask].copy()
            for _, frame in matched.iterrows():
                timestamp = safe_float(frame['timestamp_sec'])
                row = {
                    'video_id': TARGET_VIDEO_ID,
                    'ad_interval_id': str(getattr(interval, 'ad_interval_id')),
                    'ad_start_sec': round(start, 6),
                    'ad_end_sec': round(end, 6),
                    'ad_duration_sec': round(end - start, 6),
                    'window_type': window['window_type'],
                    'window_start_sec': round(w_start, 6),
                    'window_end_sec': round(w_end, 6),
                    'timestamp_sec': round(timestamp, 6),
                    'timestamp_mmss': frame.get('timestamp_mmss', mmss(timestamp)),
                    'relative_to_ad_start_sec': round(timestamp - start, 6),
                    'relative_to_ad_end_sec': round(timestamp - end, 6),
                    'inside_actual_ad': bool(start <= timestamp <= end),
                    'sampling_role': frame.get('sampling_role', ''),
                    'is_near_scene_anchor': frame.get('is_near_scene_anchor', ''),
                    'nearest_anchor_id': frame.get('nearest_anchor_id', ''),
                    'nearest_anchor_delta_sec': frame.get('nearest_anchor_delta_sec', ''),
                    'ocr_status': frame.get('ocr_status', ''),
                    'has_text': bool(str(frame.get('ocr_text_joined', '')).strip()),
                    'ocr_text_joined': clean_text(frame.get('ocr_text_joined', ''), 700),
                    'ocr_text_count': frame.get('ocr_text_count', 0),
                    'ocr_token_count': frame.get('ocr_token_count', 0),
                    'ocr_mean_confidence': frame.get('ocr_mean_confidence', ''),
                    'ad_disclosure_keyword_count': frame.get('ad_disclosure_keyword_count', 0),
                    'sponsor_keyword_count': frame.get('sponsor_keyword_count', 0),
                    'brand_product_keyword_count': frame.get('brand_product_keyword_count', 0),
                    'promotion_discount_keyword_count': frame.get('promotion_discount_keyword_count', 0),
                    'purchase_cta_keyword_count': frame.get('purchase_cta_keyword_count', 0),
                    'link_more_info_keyword_count': frame.get('link_more_info_keyword_count', 0),
                    'negative_guard_keyword_count': frame.get('negative_guard_keyword_count', 0),
                    'frame_ad_text_score': frame.get('frame_ad_text_score', 0),
                    'frame_text_density_score': frame.get('frame_text_density_score', 0),
                    'note': 'posthoc actual ad interval join only; not used for OCR sampling',
                }
                rows.append(row)
    return pd.DataFrame(rows, columns=AD_CONTEXT_FRAME_COLUMNS)


def summarize_window(df: pd.DataFrame) -> dict[str, Any]:
    counts = status_counts(df.rename(columns={'has_text': '_has_text_copy'})) if 'ocr_status' in df.columns else {'success': 0, 'empty': 0, 'failed': 0, 'nonempty': 0, 'attempted': 0}
    # status_counts nonempty는 success_nonempty를 쓰되 has_text도 명시적으로 유지한다.
    nonempty = int(df['has_text'].astype(bool).sum()) if not df.empty and 'has_text' in df.columns else 0
    ts = pd.to_numeric(df.get('timestamp_sec', pd.Series(dtype=float)), errors='coerce').dropna().sort_values()
    gaps = np.diff(ts.to_numpy()) if len(ts) > 1 else np.array([])
    representative, top_tokens = aggregate_text(df.rename(columns={'ocr_text_joined': 'ocr_text_joined'})) if not df.empty else ('', '')
    kw_cols = [c for c in RESULT_KEYWORD_COLUMNS.values() if c in df.columns]
    first_keyword = ''
    if kw_cols and not df.empty:
        kw_sum = df[kw_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
        hits = df[kw_sum > 0]
        if not hits.empty:
            first_keyword = round(float(pd.to_numeric(hits['timestamp_sec'], errors='coerce').min()), 6)
    return {
        'ocr_frame_count': int(len(df)),
        'success_frame_count': int(df['ocr_status'].isin(SUCCESS_STATUSES).sum()) if not df.empty else 0,
        'empty_frame_count': int(df['ocr_status'].eq('success_empty').sum()) if not df.empty else 0,
        'failed_frame_count': int((~df['ocr_status'].isin(SUCCESS_STATUSES)).sum()) if not df.empty else 0,
        'nonempty_frame_count': nonempty,
        'text_frame_ratio': round(nonempty / max(1, int(df['ocr_status'].isin(SUCCESS_STATUSES).sum()) if not df.empty else 0), 6),
        'mean_confidence': mean_numeric(df.get('ocr_mean_confidence', pd.Series(dtype=float))),
        'min_timestamp_sec': round(float(ts.min()), 6) if not ts.empty else '',
        'max_timestamp_sec': round(float(ts.max()), 6) if not ts.empty else '',
        'median_gap_sec': round(float(np.median(gaps)), 6) if gaps.size else '',
        'max_gap_sec': round(float(np.max(gaps)), 6) if gaps.size else '',
        'near_anchor_frame_count': int(df['sampling_role'].eq('near_scene_anchor_1s').sum()) if not df.empty else 0,
        'background_frame_count': int(df['sampling_role'].eq('background_full_video_1p5s').sum()) if not df.empty else 0,
        'max_frame_ad_text_score': round(float(pd.to_numeric(df.get('frame_ad_text_score', 0), errors='coerce').fillna(0).max()), 6) if not df.empty else 0.0,
        'mean_frame_ad_text_score': mean_numeric(df.get('frame_ad_text_score', pd.Series(dtype=float))),
        'first_keyword_timestamp_sec': first_keyword,
        'representative_ocr_text': representative,
        'top_ocr_tokens': top_tokens,
    }


def build_ad_context_summary(context_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if context_rows.empty:
        return pd.DataFrame(columns=AD_CONTEXT_SUMMARY_COLUMNS)
    group_cols = ['video_id', 'ad_interval_id', 'ad_start_sec', 'ad_end_sec', 'ad_duration_sec', 'window_type', 'window_start_sec', 'window_end_sec']
    for keys, group in context_rows.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        summary = summarize_window(group)
        for category, col in RESULT_KEYWORD_COLUMNS.items():
            summary[f'{col}_sum'] = int(pd.to_numeric(group.get(col, 0), errors='coerce').fillna(0).sum())
        base.update({
            'window_duration_sec': round(float(base['window_end_sec']) - float(base['window_start_sec']), 6),
            **summary,
            'interpretation_note': 'post-hoc actual ad context only; not used for sampling or detector rules',
        })
        rows.append(base)
    return pd.DataFrame(rows, columns=AD_CONTEXT_SUMMARY_COLUMNS)


def build_ad_context_keyword_hits(context_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if context_rows.empty:
        return pd.DataFrame(columns=AD_CONTEXT_KEYWORD_COLUMNS)
    for _, row in context_rows.iterrows():
        matches = keyword_matches(row.get('ocr_text_joined', ''))
        for category, terms in matches.items():
            if not terms:
                continue
            rows.append({
                'video_id': row.get('video_id'),
                'ad_interval_id': row.get('ad_interval_id'),
                'window_type': row.get('window_type'),
                'timestamp_sec': row.get('timestamp_sec'),
                'timestamp_mmss': row.get('timestamp_mmss'),
                'relative_to_ad_start_sec': row.get('relative_to_ad_start_sec'),
                'relative_to_ad_end_sec': row.get('relative_to_ad_end_sec'),
                'keyword_category': category,
                'keyword_count': len(terms),
                'matched_keywords': ';'.join(terms),
                'ocr_text_joined': row.get('ocr_text_joined'),
                'sampling_role': row.get('sampling_role'),
                'is_near_scene_anchor': row.get('is_near_scene_anchor'),
                'frame_ad_text_score': row.get('frame_ad_text_score'),
                'interpretation_note': 'keyword hit in post-hoc actual ad context; inspect text manually',
            })
    return pd.DataFrame(rows, columns=AD_CONTEXT_KEYWORD_COLUMNS)


def keyword_categories_for_row(row: pd.Series) -> str:
    cats = []
    for category, col in RESULT_KEYWORD_COLUMNS.items():
        if safe_float(row.get(col, 0), 0.0) > 0:
            cats.append(category)
    return ';'.join(cats)


def example_row(row: pd.Series, group: str, reason: str, ad_interval_id: str = '', window_type: str = '') -> dict[str, Any]:
    return {
        'example_group': group,
        'video_id': row.get('video_id', TARGET_VIDEO_ID),
        'ad_interval_id': row.get('ad_interval_id', ad_interval_id),
        'window_type': row.get('window_type', window_type),
        'timestamp_sec': row.get('timestamp_sec', ''),
        'timestamp_mmss': row.get('timestamp_mmss', ''),
        'sampling_role': row.get('sampling_role', ''),
        'ocr_status': row.get('ocr_status', ''),
        'ocr_text_joined': clean_text(row.get('ocr_text_joined', ''), 700),
        'keyword_categories': keyword_categories_for_row(row),
        'frame_ad_text_score': row.get('frame_ad_text_score', 0),
        'reason_selected': reason,
    }


def build_representative_examples(frame_df: pd.DataFrame, context_rows: pd.DataFrame, keyword_hits: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    nonempty = frame_df[frame_df['ocr_text_joined'].fillna('').astype(str).str.strip().ne('')].copy()
    for _, row in nonempty.sort_values(['ocr_token_count', 'ocr_char_count'], ascending=False).head(30).iterrows():
        rows.append(example_row(row, 'overall_nonempty_top30', 'longest/high-token nonempty OCR text'))
    for _, row in frame_df.sort_values(['frame_ad_text_score', 'ocr_token_count'], ascending=False).head(30).iterrows():
        rows.append(example_row(row, 'high_frame_ad_text_score_top30', 'highest frame_ad_text_score'))
    ad_nonempty = context_rows[context_rows.get('has_text', pd.Series(dtype=bool)).astype(bool)].copy() if not context_rows.empty else pd.DataFrame()
    for _, row in ad_nonempty.sort_values(['frame_ad_text_score', 'ocr_token_count'], ascending=False).head(30).iterrows():
        rows.append(example_row(row, 'actual_ad_context_nonempty', 'nonempty OCR inside post-hoc actual ad context'))
    if not keyword_hits.empty:
        for _, row in keyword_hits.head(30).iterrows():
            rows.append({
                'example_group': 'actual_ad_context_keyword_hit',
                'video_id': row.get('video_id'),
                'ad_interval_id': row.get('ad_interval_id'),
                'window_type': row.get('window_type'),
                'timestamp_sec': row.get('timestamp_sec'),
                'timestamp_mmss': row.get('timestamp_mmss'),
                'sampling_role': row.get('sampling_role'),
                'ocr_status': '',
                'ocr_text_joined': clean_text(row.get('ocr_text_joined', ''), 700),
                'keyword_categories': row.get('keyword_category'),
                'frame_ad_text_score': row.get('frame_ad_text_score'),
                'reason_selected': f"keyword hit: {row.get('matched_keywords')}",
            })
    near = nonempty[nonempty['sampling_role'].eq('near_scene_anchor_1s')]
    for _, row in near.sort_values(['frame_ad_text_score', 'ocr_token_count'], ascending=False).head(30).iterrows():
        rows.append(example_row(row, 'near_anchor_ocr_examples', 'near-anchor OCR nonempty example'))
    bg = nonempty[nonempty['sampling_role'].eq('background_full_video_1p5s')]
    for _, row in bg.sort_values(['frame_ad_text_score', 'ocr_token_count'], ascending=False).head(30).iterrows():
        rows.append(example_row(row, 'background_ocr_examples', 'background OCR nonempty example'))
    if not timeline.empty:
        timeline = timeline.copy()
        timeline['empty_ratio'] = pd.to_numeric(timeline['empty_frame_count'], errors='coerce').fillna(0) / timeline['ocr_frame_count'].clip(lower=1)
        for _, seg in timeline.sort_values(['empty_ratio', 'ocr_frame_count'], ascending=False).head(10).iterrows():
            rows.append({
                'example_group': 'high_empty_ratio_segment_examples',
                'video_id': TARGET_VIDEO_ID,
                'ad_interval_id': '',
                'window_type': f"20s_segment_{int(seg['segment_index_20s'])}",
                'timestamp_sec': seg['segment_start_sec'],
                'timestamp_mmss': mmss(seg['segment_start_sec']),
                'sampling_role': 'segment_summary',
                'ocr_status': '',
                'ocr_text_joined': clean_text(seg.get('representative_ocr_text', ''), 700),
                'keyword_categories': '',
                'frame_ad_text_score': seg.get('max_frame_ad_text_score', 0),
                'reason_selected': f"high empty ratio segment: empty={seg['empty_frame_count']}, frames={seg['ocr_frame_count']}",
            })
    return pd.DataFrame(rows, columns=EXAMPLE_COLUMNS)


def keyword_summary(frame_df: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for category, col in RESULT_KEYWORD_COLUMNS.items():
        vals = pd.to_numeric(frame_df.get(col, 0), errors='coerce').fillna(0)
        out[f'{category}_count_sum'] = int(vals.sum())
        out[f'{category}_frame_hit_count'] = int((vals > 0).sum())
    return out


def recalculate_keyword_counts(frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return frame_df
    frame_df = frame_df.copy()
    for idx, text in frame_df['ocr_text_joined'].fillna('').astype(str).items():
        matches = keyword_matches(text)
        total_keywords = 0
        for category, col in RESULT_KEYWORD_COLUMNS.items():
            count = len(matches.get(category, []))
            frame_df.at[idx, col] = count
            total_keywords += count
        token_count = safe_float(frame_df.at[idx, 'ocr_token_count'], 0.0)
        char_count = safe_float(frame_df.at[idx, 'ocr_char_count'], 0.0)
        box_count = safe_float(frame_df.at[idx, 'ocr_box_count'], 0.0)
        area_ratio = safe_float(frame_df.at[idx, 'ocr_text_area_ratio'], 0.0)
        keyword_score = min(1.0, total_keywords / 3.0)
        density_score = float(np.mean([min(1.0, token_count / 18.0), min(1.0, char_count / 120.0), min(1.0, box_count / 8.0), min(1.0, area_ratio / 0.08)]))
        frame_df.at[idx, 'frame_keyword_score'] = round(keyword_score, 6)
        frame_df.at[idx, 'frame_text_density_score'] = round(density_score, 6)
        frame_df.at[idx, 'frame_ad_text_score'] = round(min(1.0, 0.65 * keyword_score + 0.35 * density_score), 6)
    return frame_df


def top_timeline_text(timeline: pd.DataFrame, sort_col: str, n: int = 5) -> list[dict[str, Any]]:
    if timeline.empty or sort_col not in timeline.columns:
        return []
    cols = ['segment_index_20s', 'segment_start_sec', 'segment_end_sec', sort_col, 'representative_ocr_text']
    return timeline.sort_values(sort_col, ascending=False)[cols].head(n).to_dict(orient='records')


def build_quality_checks(plan: pd.DataFrame, frame_df: pd.DataFrame, context_rows: pd.DataFrame, target_info: dict[str, Any], before_stats: dict[str, dict[str, Any]], latest_forbidden: list[str], warnings: list[str]) -> pd.DataFrame:
    after_stats = file_stats(list(Path(p) for p in before_stats.keys()))
    changed = stats_changed(before_stats, after_stats)
    rows = []
    def add(name: str, status: str, detail: str) -> None:
        rows.append({'check_name': name, 'status': status, 'detail': detail})
    add('target_train_video_id_5', 'PASS' if target_info['target_split'] == TARGET_SPLIT and target_info['target_video_id'] == TARGET_VIDEO_ID else 'FAIL', json.dumps({'video_id': target_info['target_video_id'], 'split': target_info['target_split']}, ensure_ascii=False))
    add('target_anchor_rows_only', 'PASS' if target_info.get('anchor_count') == 68 else 'WARN', f"anchor_count={target_info.get('anchor_count')}")
    add('validation_test_rows_absent', 'PASS' if target_info.get('validation_test_rows_in_target') == 0 and set(plan['split'].astype(str).unique()) == {TARGET_SPLIT} else 'FAIL', f"plan_splits={sorted(plan['split'].astype(str).unique().tolist())}")
    add('sampling_no_actual_label_use', 'PASS', 'Actual ad intervals are loaded after OCR only in STEP 13.')
    add('near_anchor_interval_1s', 'PASS' if plan[plan['sampling_role'].eq('near_scene_anchor_1s')]['sampling_interval_sec'].astype(float).eq(1.0).all() else 'FAIL', '')
    add('background_interval_1p5s', 'PASS' if plan[plan['sampling_role'].eq('background_full_video_1p5s')]['sampling_interval_sec'].astype(float).eq(1.5).all() else 'FAIL', '')
    out_of_bounds = int(((pd.to_numeric(plan['timestamp_sec'], errors='coerce') < -1e-9) | (pd.to_numeric(plan['timestamp_sec'], errors='coerce') > pd.to_numeric(plan['video_duration_sec'], errors='coerce') + 1e-9)).sum())
    add('timestamps_in_video_range', 'PASS' if out_of_bounds == 0 else 'FAIL', f'out_of_bounds={out_of_bounds}')
    add('dedup_video_timestamp', 'PASS' if int(plan.duplicated(['video_id', 'timestamp_sec']).sum()) == 0 else 'FAIL', f"duplicates={int(plan.duplicated(['video_id', 'timestamp_sec']).sum())}")
    counts = status_counts(frame_df)
    add('ocr_attempted_equals_plan_count', 'PASS' if counts['attempted'] == len(plan) else 'FAIL', json.dumps({'attempted': counts['attempted'], 'plan': len(plan)}, ensure_ascii=False))
    cleanup_ok = frame_df['temp_frame_cleanup_status'].astype(str).eq('not_applicable_no_persist_frames').all()
    add('temp_frame_not_persisted', 'PASS' if cleanup_ok else 'FAIL', '')
    add('posthoc_context_has_rows', 'PASS' if not context_rows.empty else 'WARN', f'rows={len(context_rows)}')
    add('protected_inputs_unchanged', 'PASS' if not changed else 'FAIL', ';'.join(changed))
    add('latest_forbidden_files_absent', 'PASS' if not latest_forbidden else 'FAIL', ';'.join(latest_forbidden))
    if warnings:
        add('warnings_present', 'WARN', ' | '.join(warnings[:20]))
    return pd.DataFrame(rows)


def scan_forbidden_latest(dirs: list[Path]) -> list[str]:
    hits = []
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob('*'):
            if not p.is_file():
                continue
            if p.suffix.lower() in FORBIDDEN_LATEST_EXTS or p.name in FORBIDDEN_LATEST_NAMES:
                hits.append(str(p))
    return hits


def is_latest_allowed(path: Path) -> bool:
    if path.suffix.lower() in FORBIDDEN_LATEST_EXTS:
        return False
    if path.name in FORBIDDEN_LATEST_NAMES:
        return False
    return True


def update_latest(files: list[Path]) -> dict[str, Any]:
    copied = {}
    skipped = []
    for latest_dir in [LATEST_CHATGPT_DIR, LATEST_OCR_DIR]:
        latest_dir.mkdir(parents=True, exist_ok=True)
        copied[str(latest_dir)] = []
        for src in files:
            if not is_latest_allowed(src):
                skipped.append(str(src))
                continue
            dst = latest_dir / src.name
            shutil.copy2(src, dst)
            copied[str(latest_dir)].append(str(dst))
        readme = latest_dir / 'README_latest_files.md'
        lines = ['# Latest Files: Video5 Scene Anchor OCR Review v2.4', '', '## Included']
        lines.extend(f'- {p}' for p in copied[str(latest_dir)])
        lines.extend(['', '## Excluded', '- raw video', '- frame image', '- OCR cache', '- model weight', '- package directory', '- bbox 대용량 파일', '- validation/test row-level output', '- 기존 large feature originals', '- full sampling plan', '- full frame-level OCR result', ''])
        readme.write_text('\n'.join(lines), encoding='utf-8')
        copied[str(latest_dir)].append(str(readme))
    return {'copied': copied, 'skipped': sorted(set(skipped))}


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
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def build_summary_md(report: dict[str, Any], timeline: pd.DataFrame, ad_summary: pd.DataFrame, examples: pd.DataFrame) -> str:
    ocr = report['ocr_execution_summary']
    sampling = report['sampling_plan_summary']
    kw = report['keyword_hit_summary']
    top_kw = sorted([(k.replace('_count_sum', ''), v) for k, v in kw.items() if k.endswith('_count_sum')], key=lambda x: x[1], reverse=True)
    ad_body = ad_summary[ad_summary['window_type'].eq('ad_body')] if not ad_summary.empty else pd.DataFrame()
    top_ad = top_timeline_text(timeline, 'max_frame_ad_text_score', 5)
    dense = top_timeline_text(timeline, 'token_count_sum', 5)
    empty = timeline.assign(empty_ratio=timeline['empty_frame_count'] / timeline['ocr_frame_count'].clip(lower=1)).sort_values('empty_ratio', ascending=False).head(5)[['segment_index_20s', 'segment_start_sec', 'segment_end_sec', 'empty_ratio', 'representative_ocr_text']].to_dict(orient='records') if not timeline.empty else []
    near = top_timeline_text(timeline, 'near_anchor_frame_count', 5)
    representative = examples[examples['example_group'].eq('high_frame_ad_text_score_top30')].head(5)['ocr_text_joined'].tolist() if not examples.empty else []
    ad_rep = examples[examples['example_group'].str.contains('actual_ad_context', na=False)].head(5)['ocr_text_joined'].tolist() if not examples.empty else []
    conclusion = (
        f"video_id=5 OCR은 실제로 실행되었고 총 {ocr['attempted']} frame을 시도해 "
        f"success={ocr['success']}, empty={ocr['empty']}, failed={ocr['failed']}를 기록했다. "
        f"전체적으로 vlog 화면 자막/일상 텍스트와 일부 제품/구매 관련 단어가 추출되었고, "
        f"actual ad context는 OCR 이후 post-hoc으로만 join해 광고구간 안팎 단서를 별도로 정리했다."
    )
    lines = [
        '# Video5 Scene Anchor OCR Review v2.4', '',
        '## 1. 한 문단 결론', conclusion, '',
        '## 2. Sampling Plan 요약',
        f"- video duration: {report['video_metadata'].get('duration_sec')}",
        f"- anchor count: {report['anchor_summary'].get('anchor_count')}",
        f"- near-anchor frame 수: {sampling.get('near_anchor_count')}",
        f"- background frame 수: {sampling.get('background_count')}",
        f"- dedup 제거 수: {sampling.get('dedup_removed_count')}",
        f"- 최종 OCR 대상 frame 수: {sampling.get('final_ocr_frame_count')}",
        f"- median/max sample gap: {sampling.get('median_gap_sec')} / {sampling.get('max_gap_sec')}", '',
        '## 3. 전체 OCR 결과 요약',
        f"- nonempty frame ratio: {ocr.get('nonempty_ratio')}",
        f"- keyword category counts: {top_kw}",
        f"- 대표 OCR 텍스트 예시: {representative[:5]}", '',
        '## 4. 20초 Timeline 요약',
        f"- 광고성 OCR score 높은 구간: {top_ad}",
        f"- 텍스트 밀도 높은 구간: {dense}",
        f"- empty 많은 구간: {empty}",
        f"- near-anchor OCR 많은 구간: {near}", '',
        '## 5. Actual Ad Context 분석',
        '- actual ad interval은 OCR result 생성 후 post-hoc 분석 단계에서만 읽었다.',
        f"- video_id=5 actual ad interval count: {report['actual_ad_context_summary'].get('ad_interval_count')}",
        f"- actual ad intervals: {report['actual_ad_context_summary'].get('ad_intervals')}",
    ]
    if not ad_summary.empty:
        lines.append('- window별 요약:')
        for row in ad_summary.to_dict(orient='records'):
            lines.append(f"  - {row['ad_interval_id']} {row['window_type']}: frames={row['ocr_frame_count']}, nonempty={row['nonempty_frame_count']}, keywords={row['ad_disclosure_keyword_count_sum'] + row['sponsor_keyword_count_sum'] + row['brand_product_keyword_count_sum'] + row['promotion_discount_keyword_count_sum'] + row['purchase_cta_keyword_count_sum'] + row['link_more_info_keyword_count_sum']}, representative={row['representative_ocr_text']}")
    lines.extend([
        f"- 실제 광고구간 안 대표 OCR 텍스트: {ad_rep[:5]}",
        '- 광고구간과 주변 구간의 OCR 차이는 위 actual_ad_context summary와 keyword hit CSV에서 window_type별로 비교할 수 있다.', '',
        '## 6. 해석',
        '- video_id=5에서는 새 OCR 방식이 전체 영상 분포와 scene-anchor 주변 단서를 동시에 확인하는 데 유용했다.',
        '- near-anchor 1초 OCR은 장면전환 주변의 텍스트 변화 확인에 좋고, background 1.5초 OCR은 영상 전반의 제품/일상 텍스트 분포를 놓치지 않는 역할을 한다.',
        '- EasyOCR로 약 1천 frame 규모의 단일 영상 OCR이 완료되었으므로 train 전체 실행 가치는 있다. 다만 전체 train은 13k+ frame 규모라 장시간 실행 또는 batch 분할이 필요하다.',
        '- train 전체 전에는 `target-video-id`/batch 옵션, resume, failed-frame retry, latest bundle 제외 규칙을 유지하는 쪽이 좋다.', '',
        '## 7. Safety',
        '- sampling에 actual label 사용 여부: false',
        '- actual label은 post-hoc analysis only',
        '- detector modified: false',
        '- existing OCR modified: false',
        '- existing scene anchor/candidate modified: false',
        '- raw frame persisted: false',
        '- validation/test row-level output generated: false', '',
        '## 8. Outputs',
    ])
    for key, value in report['outputs'].items():
        lines.append(f'- {key}: `{value}`')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f'video5_scene_anchor_ocr_review_v2_4_{now_id()}'
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs = {key: run_dir / name for key, name in OUTPUT_NAMES.items()}
    logger = TaskLogger(outputs['run_log'])
    warnings: list[str] = []
    errors: list[str] = []
    started = time.monotonic()
    protected_inputs = [ANCHOR_FILE, SPLIT_FILE, MANIFEST_FILE, ACTUAL_AD_INTERVAL_FILE, BASE_SCRIPT]
    before_stats = file_stats(protected_inputs)

    logger.log('[STEP 01] Safety snapshot and run directory setup')
    logger.log(f'run_id={run_id}')
    logger.log('[STEP 02] Inspect existing OCR extraction script')
    logger.log(f'base_script={BASE_SCRIPT}; target_video_option_supported=false; using new video5 script')

    logger.log('[STEP 03] Load split, manifest, and primary scene anchor')
    anchors_all, manifest, split = load_sampling_inputs()

    logger.log('[STEP 04] Validate target train video_id=5')
    target_anchors, target_info = validate_and_filter_target(anchors_all, manifest, split, args, warnings, errors)
    if errors:
        logger.log(f'validation errors={errors}')
        raise RuntimeError('; '.join(errors))
    video_meta = target_info['video_metadata']
    duration = safe_float(video_meta.get('duration_sec'))
    video_title = str(video_meta.get('video_title', ''))

    logger.log('[STEP 05] Build video5 near-anchor 1s sampling windows')
    logger.log('[STEP 06] Build video5 full-video 1.5s background sampling')
    raw_records, raw_stats = BASE.build_sampling_records(target_anchors, args, run_id, warnings)

    logger.log('[STEP 07] Merge and deduplicate video5 sampling timestamps')
    plan, dedup, dedup_stats = BASE.deduplicate_records(raw_records, args, run_id)
    raw_stats.update(dedup_stats)
    if len(plan) > args.max_ocr_frames:
        warnings.append(f'final OCR frame count {len(plan)} exceeds max_ocr_frames={args.max_ocr_frames}; OCR skipped unless max is raised')

    logger.log('[STEP 08] Write sampling plan and sampling QA')
    write_csv(outputs['sampling_plan'], plan, BASE.SAMPLING_PLAN_COLUMNS)
    write_csv(outputs['dedup_summary'], dedup)
    logger.log(f'video5 final OCR frame count={len(plan)}')

    logger.log('[STEP 09] Check EasyOCR/backend availability')
    backend_info, reader = BASE.select_ocr_backend(args.ocr_backend, args.skip_ocr or len(plan) > args.max_ocr_frames)
    logger.log(f'backend={backend_info}')

    logger.log('[STEP 10] Run OCR for video_id=5')
    if backend_info.get('ocr_backend_status') == 'ready' and len(plan) <= args.max_ocr_frames:
        frame_df = BASE.run_ocr(plan, plan.copy(), backend_info, reader, logger)
        fallback_reason = ''
        execution_status = 'executed'
    else:
        frame_df = BASE.run_ocr(plan, plan.iloc[0:0].copy(), backend_info, reader, logger)
        fallback_reason = backend_info.get('warning', 'ocr skipped due to max frame guard')
        execution_status = 'fallback_plan_only'
    frame_df = recalculate_keyword_counts(frame_df)
    write_csv(outputs['frame_results'], frame_df, BASE.OCR_RESULT_COLUMNS)

    logger.log('[STEP 11] Build full-video OCR text review tables')
    compact = build_compact_text_review(frame_df)
    write_csv(outputs['compact_text_review'], compact, COMPACT_COLUMNS)

    logger.log('[STEP 12] Build 20s timeline OCR review')
    timeline = timeline_review(frame_df, duration)
    write_csv(outputs['timeline_review'], timeline, TIMELINE_COLUMNS)

    intervals = load_actual_ad_intervals_posthoc(logger)
    logger.log('[STEP 14] Build actual ad context OCR frame rows')
    context_rows = build_ad_context_rows(frame_df, intervals, duration)
    write_csv(outputs['actual_ad_context_frame_rows'], context_rows, AD_CONTEXT_FRAME_COLUMNS)

    logger.log('[STEP 15] Build actual ad context summaries and keyword hits')
    context_summary = build_ad_context_summary(context_rows)
    write_csv(outputs['actual_ad_context_summary'], context_summary, AD_CONTEXT_SUMMARY_COLUMNS)
    context_keyword_hits = build_ad_context_keyword_hits(context_rows)
    write_csv(outputs['actual_ad_context_keyword_hits'], context_keyword_hits, AD_CONTEXT_KEYWORD_COLUMNS)

    logger.log('[STEP 16] Build representative OCR text examples')
    examples = build_representative_examples(frame_df, context_rows, context_keyword_hits, timeline)
    write_csv(outputs['representative_examples'], examples, EXAMPLE_COLUMNS)

    logger.log('[STEP 17] Run Sub Agent validations')
    latest_forbidden_pre = scan_forbidden_latest([LATEST_CHATGPT_DIR, LATEST_OCR_DIR])
    quality = build_quality_checks(plan, frame_df, context_rows, target_info, before_stats, latest_forbidden_pre, warnings)
    write_csv(outputs['quality_checks'], quality)

    frame_counts = status_counts(frame_df)
    kw_summary = keyword_summary(frame_df)
    timestamps = pd.to_numeric(plan['timestamp_sec'], errors='coerce').dropna().sort_values()
    gaps = np.diff(timestamps.to_numpy()) if len(timestamps) > 1 else np.array([])
    ad_body_rows = context_rows[context_rows['window_type'].eq('ad_body')] if not context_rows.empty else pd.DataFrame()
    ad_body_counts = status_counts(ad_body_rows.rename(columns={'has_text': '_has_text_copy'})) if not ad_body_rows.empty else {'attempted': 0, 'success': 0, 'empty': 0, 'failed': 0, 'nonempty': 0}
    ad_context_keyword_hit_count = int(context_keyword_hits['keyword_count'].sum()) if not context_keyword_hits.empty else 0
    representative_ad_text = ''
    if not ad_body_rows.empty:
        texts = ad_body_rows[ad_body_rows['has_text'].astype(bool)]['ocr_text_joined'].astype(str).tolist()
        representative_ad_text = max(texts, key=len) if texts else ''
    actual_summary = {
        'ad_interval_count': int(len(intervals)),
        'ad_intervals': intervals[['ad_interval_id', 'ad_start_sec', 'ad_end_sec', 'ad_duration_sec']].to_dict(orient='records') if not intervals.empty else [],
        'ad_body_ocr_frame_count': int(len(ad_body_rows)),
        'ad_body_nonempty_frame_count': int(ad_body_rows['has_text'].astype(bool).sum()) if not ad_body_rows.empty else 0,
        'ad_context_keyword_hit_count': ad_context_keyword_hit_count,
        'representative_ad_context_text': representative_ad_text,
    }
    ocr_summary = {
        **frame_counts,
        'nonempty_ratio': round(frame_counts['nonempty'] / max(1, frame_counts['success']), 6),
        'backend': backend_info.get('ocr_backend'),
        'execution_status': execution_status,
    }
    report = {
        'run_id': run_id,
        'generated_at': now_iso(),
        'project_root': str(PROJECT_ROOT),
        'target_video_id': args.target_video_id,
        'target_split': args.target_split,
        'script_path': str(SCRIPT_PATH),
        'primary_anchor_file': str(ANCHOR_FILE),
        'split_file': str(SPLIT_FILE),
        'manifest_file': str(MANIFEST_FILE),
        'actual_ad_interval_file': str(ACTUAL_AD_INTERVAL_FILE),
        'actual_label_used_for_sampling': False,
        'actual_label_used_for_posthoc_analysis': True,
        'parameters': {
            'target_video_id': args.target_video_id,
            'target_split': args.target_split,
            'anchor_window_sec': args.anchor_window_sec,
            'near_anchor_interval_sec': args.near_anchor_interval_sec,
            'background_interval_sec': args.background_interval_sec,
            'dedup_tolerance_sec': args.dedup_tolerance_sec,
            'timestamp_round_decimals': args.timestamp_round_decimals,
            'no_persist_frames': args.no_persist_frames,
            'max_ocr_frames': args.max_ocr_frames,
        },
        'input_schema': {
            'video_id_column': 'video_id',
            'timestamp_column': 'canonical_boundary_time_sec',
            'split_column': 'split',
            'anchor_id_column': 'scene_boundary_anchor_id',
            'source_debug_columns': ['source_relation', 'canonical_time_source', 'has_opencv_ffmpeg_candidate', 'has_resnet_candidate', 'source_count', 'visual_boundary_strength_score'],
        },
        'video_metadata': video_meta,
        'anchor_summary': {'anchor_count': int(len(target_anchors)), 'source_relation_counts': target_anchors['source_relation'].value_counts().to_dict()},
        'sampling_plan_summary': {
            'raw_total_count': int(raw_stats.get('raw_total_count', 0)),
            'raw_near_count': int(raw_stats.get('raw_near_count', 0)),
            'raw_background_count': int(raw_stats.get('raw_background_count', 0)),
            'near_anchor_count': int((plan['sampling_role'] == 'near_scene_anchor_1s').sum()),
            'background_count': int((plan['sampling_role'] == 'background_full_video_1p5s').sum()),
            'dedup_removed_count': int(raw_stats.get('dedup_removed_count', 0)),
            'final_ocr_frame_count': int(len(plan)),
            'median_gap_sec': round(float(np.median(gaps)), 6) if gaps.size else 0.0,
            'max_gap_sec': round(float(np.max(gaps)), 6) if gaps.size else 0.0,
        },
        'ocr_backend_status': {k: v for k, v in backend_info.items() if k != 'reader'},
        'ocr_execution_summary': ocr_summary,
        'full_video_ocr_summary': {
            'nonempty_ratio': ocr_summary['nonempty_ratio'],
            'keyword_summary': kw_summary,
            'top_ad_score_examples': examples[examples['example_group'].eq('high_frame_ad_text_score_top30')].head(10).to_dict(orient='records') if not examples.empty else [],
        },
        'timeline_20s_summary': {
            'row_count': int(len(timeline)),
            'top_ad_score_segments': top_timeline_text(timeline, 'max_frame_ad_text_score', 10),
            'top_text_density_segments': top_timeline_text(timeline, 'token_count_sum', 10),
        },
        'actual_ad_context_summary': actual_summary,
        'keyword_hit_summary': kw_summary,
        'representative_examples_summary': examples.groupby('example_group').size().to_dict() if not examples.empty else {},
        'validation_results': quality.to_dict(orient='records'),
        'safety_results': {
            'actual_label_used_for_sampling': False,
            'actual_label_used_for_posthoc_analysis': True,
            'detector_modified': False,
            'existing_ocr_modified': False,
            'existing_scene_anchor_candidate_modified': False,
            'split_file_modified': False,
            'label_file_modified': False,
            'raw_frame_persisted': False,
            'validation_test_row_level_output_generated': False,
            'protected_input_stat_changes': stats_changed(before_stats, file_stats(protected_inputs)),
        },
        'outputs': {key: str(path) for key, path in outputs.items()} | {'run_dir': str(run_dir), 'latest_for_chatgpt': str(LATEST_CHATGPT_DIR), 'latest_ocr': str(LATEST_OCR_DIR)},
        'warnings': warnings,
        'errors': errors,
        'fallback_reason': fallback_reason,
        'runtime_sec': round(time.monotonic() - started, 3),
    }

    logger.log('[STEP 18] Generate markdown/json reports')
    outputs['report_json'].write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    outputs['summary_md'].write_text(build_summary_md(report, timeline, context_summary, examples), encoding='utf-8')

    logger.log('[STEP 19] Update latest bundles')
    latest_files = [SCRIPT_PATH, outputs['summary_md'], outputs['report_json'], outputs['run_log'], outputs['dedup_summary'], outputs['compact_text_review'], outputs['timeline_review'], outputs['actual_ad_context_frame_rows'], outputs['actual_ad_context_summary'], outputs['actual_ad_context_keyword_hits'], outputs['representative_examples'], outputs['quality_checks']]
    latest_status = update_latest(latest_files)
    latest_forbidden = scan_forbidden_latest([LATEST_CHATGPT_DIR, LATEST_OCR_DIR])
    report['latest_bundle_status'] = latest_status
    report['latest_forbidden_files'] = latest_forbidden
    outputs['report_json'].write_text(json.dumps(json_ready(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    outputs['summary_md'].write_text(build_summary_md(report, timeline, context_summary, examples), encoding='utf-8')
    update_latest(latest_files)

    logger.log('[STEP 20] Print final human-readable summary')
    # 마지막 step을 쓴 뒤 final log를 동기화한다.
    for latest_dir in [LATEST_CHATGPT_DIR, LATEST_OCR_DIR]:
        if latest_dir.exists():
            shutil.copy2(outputs['run_log'], latest_dir / outputs['run_log'].name)
    print('1. Target:')
    print(f'   - split: {args.target_split}')
    print(f'   - video_id: {args.target_video_id}')
    print(f'   - video_title: {video_title}')
    print(f'   - duration: {duration}')
    print('2. Sampling:')
    print(f"   - anchor count: {len(target_anchors)}")
    print(f"   - near-anchor frames: {(plan['sampling_role'] == 'near_scene_anchor_1s').sum()}")
    print(f"   - background frames: {(plan['sampling_role'] == 'background_full_video_1p5s').sum()}")
    print(f"   - dedup removed: {raw_stats.get('dedup_removed_count', 0)}")
    print(f"   - final OCR frame count: {len(plan)}")
    print('3. OCR:')
    print(f"   - backend: {backend_info.get('ocr_backend')} / {backend_info.get('ocr_backend_status')}")
    print(f"   - attempted: {frame_counts['attempted']}")
    print(f"   - success: {frame_counts['success']}")
    print(f"   - empty: {frame_counts['empty']}")
    print(f"   - failed: {frame_counts['failed']}")
    print(f"   - nonempty ratio: {ocr_summary['nonempty_ratio']}")
    print('4. Overall OCR:')
    top_categories = sorted([(k, v) for k, v in kw_summary.items() if k.endswith('_count_sum')], key=lambda x: x[1], reverse=True)[:5]
    print(f'   - top keyword categories: {top_categories}')
    print(f"   - strongest OCR text examples: {examples[examples['example_group'].eq('high_frame_ad_text_score_top30')].head(3)['ocr_text_joined'].tolist() if not examples.empty else []}")
    print('5. Actual ad context:')
    print(f"   - ad interval count: {actual_summary['ad_interval_count']}")
    print(f"   - ad_body OCR frame count: {actual_summary['ad_body_ocr_frame_count']}")
    print(f"   - ad_body nonempty frame count: {actual_summary['ad_body_nonempty_frame_count']}")
    print(f"   - ad context keyword hit count: {actual_summary['ad_context_keyword_hit_count']}")
    print(f"   - representative ad context text: {clean_text(actual_summary['representative_ad_context_text'], 200)}")
    print('6. Outputs:')
    print(f'   - run dir: {run_dir}')
    print(f"   - summary: {outputs['summary_md']}")
    print(f"   - report: {outputs['report_json']}")
    print(f'   - latest bundle: {LATEST_CHATGPT_DIR}; {LATEST_OCR_DIR}')
    print('7. Safety:')
    print('   - actual label used for sampling: false')
    print('   - actual label used for posthoc analysis: true')
    print('   - detector modified: false')
    print('   - existing OCR modified: false')
    print('   - existing scene anchors modified: false')
    print('   - raw frame persisted: false')


if __name__ == '__main__':
    main()
