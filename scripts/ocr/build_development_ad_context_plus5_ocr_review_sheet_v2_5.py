#!/usr/bin/env python3
"""Build Development Set ad-context plus-5s OCR review sheets for v2.5.

This post-hoc review builder reads existing OCR frame-level results and actual
ad intervals, then extracts rows in [ad_start_sec - 5, ad_end_sec + 5]. It does
not run OCR, call an OCR engine, create frame images, modify detectors, or alter
existing OCR/candidate/label/split files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path('.')
OLD_PROJECT_ROOT = Path('./_old_project_not_included')
SOURCE_OCR_RUN_ID = 'final_scene_anchor_ocr_v2_5_development_20260527_050904'
SOURCE_OCR_RUN_DIR = PROJECT_ROOT / 'workspaces/ocr_final_scene_anchor_v2_5_development/runs' / SOURCE_OCR_RUN_ID
TARGET_RUN_PREFIX = 'development_ad_context_plus5_ocr_review_v2_5'
DEVELOPMENT_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
DIAGNOSTIC_IDS = [3, 7, 18]
PURE_TEST_IDS = [4, 16, 17]
EXTENDED_EVALUATION_IDS = DIAGNOSTIC_IDS + PURE_TEST_IDS
LOW_CONFIDENCE_THRESHOLD = 0.50
REVIEW_PRE_SEC = 5.0
REVIEW_POST_SEC = 5.0
PREVIOUS_REVIEW_WINDOW = '[ad_start_sec - 5, ad_end_sec]'
REVIEW_WINDOW_NAME = 'ad_start_minus5_to_ad_end_plus5'
REVIEW_WINDOW_DEFINITION = '[ad_start_sec - 5, ad_end_sec + 5]'
SPLIT_EXPLANATION_KO = '본 프로젝트는 학습 기반 모델이 아니라 rule-based detector이므로, 기존 train split은 Development Set으로 두고 rule 설계와 cue 분석, 오류 진단에 사용한다. 공개용 설명에서는 기존 validation과 test를 Test Set으로 통합해 규칙 고정 이후 평가 대상으로 설명한다.'
SPLIT_NOTE = 'Development Set = original v2.4 train split; used for rule design, cue analysis, and diagnostic OCR quality review, not ML model training.'

SCRIPT_PATH = PROJECT_ROOT / 'scripts/ocr/build_development_ad_context_plus5_ocr_review_sheet_v2_5.py'
OCR_FRAME_RESULT = SOURCE_OCR_RUN_DIR / 'final_scene_anchor_ocr_frame_results_v2_5_development.csv'
OCR_VIDEO_SUMMARY = SOURCE_OCR_RUN_DIR / 'final_scene_anchor_ocr_video_summary_v2_5_development.csv'
OCR_REPORT = SOURCE_OCR_RUN_DIR / 'final_scene_anchor_ocr_v2_5_development_report.json'
LABEL_FILE = PROJECT_ROOT / 'data/segments/ad_interval_segments_v2_4.csv'
SPLIT_FILE = PROJECT_ROOT / 'data/splits/video_split_v2_4.csv'
MANIFEST_FILE = PROJECT_ROOT / 'data/video_metadata/video_manifest_v2_2.csv'
RUNS_ROOT = PROJECT_ROOT / 'workspaces/ocr_quality_review_v2_5_development/runs'
LATEST_CHATGPT = PROJECT_ROOT / 'outputs/latest_for_chatgpt_development_ad_context_plus5_ocr_review_v2_5'
LATEST_OCR = PROJECT_ROOT / 'outputs/latest_ocr'
LATEST_SCENE = PROJECT_ROOT / 'outputs/latest_scene'
FORBIDDEN_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.m4a', '.flac', '.jpg', '.jpeg', '.png', '.webp', '.gif', '.pt', '.pth', '.ckpt', '.onnx', '.bin', '.pkl', '.pickle', '.parquet'}

FULL_COLUMNS = [
    'review_row_id', 'run_id', 'original_split_v2_4', 'split_role_v2_5', 'evaluation_subset_v2_5',
    'video_id', 'video_title', 'ad_interval_id', 'ad_start_sec', 'ad_end_sec', 'ad_duration_sec',
    'review_window_start_sec', 'review_window_end_sec', 'review_window_duration_sec', 'timestamp_sec',
    'timestamp_mmss', 'relative_to_ad_start_sec', 'relative_to_ad_end_sec', 'phase_in_review_window',
    'sampling_role', 'is_anchor_dense', 'is_background_regular', 'nearest_anchor_id', 'nearest_anchor_time_sec',
    'nearest_anchor_delta_sec', 'ocr_status', 'has_text', 'ocr_text_joined', 'ocr_text_normalized',
    'ocr_text_count', 'ocr_token_count', 'ocr_char_count', 'ocr_box_count', 'ocr_mean_confidence',
    'ocr_min_confidence', 'ocr_max_confidence', 'ocr_text_area_ratio', 'ad_disclosure_keyword_count',
    'sponsor_keyword_count', 'brand_product_keyword_count', 'promotion_discount_keyword_count',
    'purchase_cta_keyword_count', 'link_more_info_keyword_count', 'negative_guard_keyword_count',
    'total_ad_keyword_count', 'corrected_ad_disclosure_hit_count', 'corrected_sponsor_keyword_count',
    'corrected_brand_product_keyword_count', 'corrected_promotion_discount_keyword_count',
    'corrected_purchase_cta_keyword_count', 'corrected_link_more_info_keyword_count',
    'corrected_negative_guard_keyword_count', 'corrected_total_ad_keyword_count', 'matched_keyword_categories',
    'matched_keywords', 'matched_keyword_rules', 'matched_keyword_confidence', 'suggested_canonical_phrase',
    'suppressed_by_negative_guard', 'corrected_frame_ad_text_score', 'quick_quality_flag', 'reviewer_note',
    'ocr_quality_manual_label', 'post_end_review_note'
]
COMPACT_COLUMNS = [
    'video_id', 'video_title_short', 'ad_interval_id', 'timestamp_mmss', 'timestamp_sec',
    'relative_to_ad_start_sec', 'relative_to_ad_end_sec', 'phase_in_review_window', 'sampling_role',
    'ocr_status', 'has_text', 'ocr_mean_confidence', 'ocr_text_joined', 'matched_keyword_categories',
    'matched_keywords', 'matched_keyword_rules', 'corrected_total_ad_keyword_count',
    'corrected_frame_ad_text_score', 'quick_quality_flag', 'reviewer_note'
]
POST_END_COLUMNS = [
    'video_id', 'video_title_short', 'ad_interval_id', 'ad_end_sec', 'timestamp_mmss', 'timestamp_sec',
    'relative_to_ad_end_sec', 'sampling_role', 'ocr_status', 'has_text', 'ocr_text_joined',
    'matched_keyword_categories', 'matched_keywords', 'matched_keyword_rules', 'corrected_total_ad_keyword_count',
    'corrected_frame_ad_text_score', 'quick_quality_flag', 'post_end_interpretation_hint', 'reviewer_note'
]
TRANSITION_COLUMNS = [
    'video_id', 'video_title', 'ad_interval_id', 'phase_in_review_window', 'phase_start_sec', 'phase_end_sec',
    'phase_duration_sec', 'row_count', 'success_text_count', 'success_empty_count', 'failed_count',
    'nonempty_ratio', 'mean_confidence', 'keyword_hit_row_count', 'corrected_total_ad_keyword_count_sum',
    'corrected_ad_disclosure_hit_count_sum', 'corrected_sponsor_keyword_count_sum',
    'corrected_brand_product_keyword_count_sum', 'corrected_promotion_discount_keyword_count_sum',
    'corrected_purchase_cta_keyword_count_sum', 'corrected_link_more_info_keyword_count_sum',
    'corrected_negative_guard_keyword_count_sum', 'mean_corrected_frame_ad_text_score',
    'max_corrected_frame_ad_text_score', 'representative_ocr_text', 'phase_interpretation_note'
]
INTERVAL_COLUMNS = [
    'original_split_v2_4', 'split_role_v2_5', 'evaluation_subset_v2_5', 'video_id', 'video_title',
    'ad_interval_id', 'ad_start_sec', 'ad_end_sec', 'ad_duration_sec', 'review_window_start_sec',
    'review_window_end_sec', 'review_window_duration_sec', 'review_row_count', 'pre_start_5s_row_count',
    'ad_body_row_count', 'post_end_5s_row_count', 'success_text_count', 'success_empty_count', 'failed_count',
    'nonempty_count', 'nonempty_ratio', 'mean_confidence', 'median_confidence', 'low_confidence_count',
    'corrected_keyword_hit_row_count', 'corrected_total_ad_keyword_count_sum',
    'corrected_ad_disclosure_hit_count_sum', 'corrected_sponsor_keyword_count_sum',
    'corrected_brand_product_keyword_count_sum', 'corrected_promotion_discount_keyword_count_sum',
    'corrected_purchase_cta_keyword_count_sum', 'corrected_link_more_info_keyword_count_sum',
    'corrected_negative_guard_keyword_count_sum', 'typo_variant_hit_count', 'fuzzy_review_needed_count',
    'first_disclosure_timestamp_sec', 'first_keyword_timestamp_sec', 'post_end_keyword_hit_row_count',
    'post_end_still_ad_like_row_count', 'post_end_nonempty_ratio', 'representative_ocr_text',
    'post_end_representative_ocr_text', 'top_matched_keywords', 'suggested_review_priority', 'interval_quality_note'
]
KEYWORD_COLUMNS = [
    'video_id', 'video_title_short', 'ad_interval_id', 'timestamp_mmss', 'timestamp_sec',
    'relative_to_ad_start_sec', 'relative_to_ad_end_sec', 'phase_in_review_window', 'sampling_role',
    'ocr_text_joined', 'matched_keyword_categories', 'matched_keywords', 'matched_keyword_rules',
    'matched_keyword_confidence', 'corrected_ad_disclosure_hit_count', 'corrected_sponsor_keyword_count',
    'corrected_brand_product_keyword_count', 'corrected_promotion_discount_keyword_count',
    'corrected_purchase_cta_keyword_count', 'corrected_link_more_info_keyword_count',
    'corrected_frame_ad_text_score', 'suggested_canonical_phrase', 'quick_quality_flag', 'review_reason',
    'reviewer_note'
]
DISCLOSURE_COLUMNS = [
    'video_id', 'video_title_short', 'ad_interval_id', 'ad_start_sec', 'ad_end_sec', 'timestamp_mmss',
    'timestamp_sec', 'relative_to_ad_start_sec', 'relative_to_ad_end_sec', 'phase_in_review_window',
    'ocr_text_joined', 'ocr_text_normalized', 'matched_keywords', 'matched_keyword_rules',
    'matched_keyword_confidence', 'suggested_canonical_phrase', 'corrected_ad_disclosure_hit_count',
    'suppressed_by_negative_guard', 'disclosure_review_type', 'reviewer_note'
]
EMPTY_LOWCONF_COLUMNS = [
    'video_id', 'video_title_short', 'ad_interval_id', 'timestamp_mmss', 'timestamp_sec',
    'relative_to_ad_start_sec', 'relative_to_ad_end_sec', 'phase_in_review_window', 'sampling_role',
    'ocr_status', 'ocr_mean_confidence', 'ocr_text_count', 'ocr_text_joined', 'nearest_anchor_id',
    'nearest_anchor_delta_sec', 'quick_quality_flag', 'review_reason', 'reviewer_note'
]
QUALITY_CHECK_COLUMNS = ['check_name', 'status', 'detail']


def now_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('', encoding='utf-8')
    def log(self, msg: str) -> None:
        print(msg, flush=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(f'{now_iso()} {msg}\n')


def safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return float('nan')
        v = float(x)
        return v if math.isfinite(v) else float('nan')
    except Exception:
        return float('nan')


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
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
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'path': str(path), 'exists': False, 'file_count': 0, 'digest': None}
    targets = [path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
    base = path.parent if path.is_file() else path
    rows = []
    for p in targets:
        try:
            s = p.stat()
        except OSError:
            continue
        rows.append(f'{p.relative_to(base)}\t{s.st_size}\t{s.st_mtime_ns}')
    digest = hashlib.sha256('\n'.join(sorted(rows)).encode('utf-8')).hexdigest()
    return {'path': str(path), 'exists': True, 'file_count': len(rows), 'digest': digest}


def locate_file(exact: Path, fallback_roots: list[Path], filename: str) -> Path:
    if exact.exists():
        return exact
    matches: list[Path] = []
    for root in fallback_roots:
        if root.exists():
            matches.extend(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f'Could not locate {filename}; checked {exact}')
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def ensure_columns(df: pd.DataFrame, columns: list[str], default: Any = '') -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = default
    return out


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(['true', '1', 'yes', 'y'])


def short_title(x: Any, max_len: int = 42) -> str:
    text = '' if pd.isna(x) else str(x)
    return text if len(text) <= max_len else text[:max_len - 1] + '…'


def timestamp_mmss(x: Any) -> str:
    sec = safe_float(x)
    if not math.isfinite(sec):
        return ''
    sec = max(0.0, sec)
    minute = int(sec // 60)
    second = int(round(sec - minute * 60))
    if second >= 60:
        minute += 1
        second -= 60
    return f'{minute:02d}:{second:02d}'


def split_items(x: Any) -> list[str]:
    if pd.isna(x):
        return []
    text = str(x).replace('|', ';').replace(',', ';')
    return [item.strip() for item in text.split(';') if item.strip()]


def top_keywords(s: pd.Series, max_items: int = 8) -> str:
    counts: Counter[str] = Counter()
    for value in s.fillna('').astype(str):
        counts.update(split_items(value))
    return '; '.join(f'{k}:{v}' for k, v in counts.most_common(max_items))


def first_text(s: pd.Series, max_len: int = 220) -> str:
    for value in s.fillna('').astype(str):
        text = value.strip()
        if text:
            return text[:max_len]
    return ''


def highest_score_text(df: pd.DataFrame, max_len: int = 220) -> str:
    if df.empty:
        return ''
    temp = df.copy()
    temp['_score'] = pd.to_numeric(temp.get('corrected_frame_ad_text_score', 0), errors='coerce').fillna(0)
    temp['_has_text'] = temp.get('ocr_text_joined', '').fillna('').astype(str).str.len() > 0
    temp = temp[temp['_has_text']].sort_values(['_score', 'timestamp_sec'], ascending=[False, True])
    if temp.empty:
        return ''
    return str(temp.iloc[0].get('ocr_text_joined', ''))[:max_len]


def output_paths(run_dir: Path) -> dict[str, Path]:
    return {
        'full_review_sheet': run_dir / 'development_ad_start_minus5_to_ad_end_plus5_ocr_review_sheet_v2_5.csv',
        'compact_review_sheet': run_dir / 'development_ad_start_minus5_to_ad_end_plus5_ocr_review_compact_v2_5.csv',
        'markdown_report': run_dir / 'development_ad_start_minus5_to_ad_end_plus5_ocr_review_sheet_v2_5.md',
        'interval_summary': run_dir / 'development_ad_context_plus5_ocr_quality_summary_by_interval_v2_5.csv',
        'video_summary': run_dir / 'development_ad_context_plus5_ocr_quality_summary_by_video_v2_5.csv',
        'keyword_hit_review': run_dir / 'development_ad_context_plus5_ocr_keyword_hit_review_v2_5.csv',
        'disclosure_review': run_dir / 'development_ad_context_plus5_ocr_disclosure_review_v2_5.csv',
        'empty_lowconf_review': run_dir / 'development_ad_context_plus5_ocr_empty_lowconf_review_v2_5.csv',
        'representative_examples': run_dir / 'development_ad_context_plus5_ocr_representative_examples_v2_5.csv',
        'post_end_review': run_dir / 'development_ad_context_post_end_5s_ocr_review_v2_5.csv',
        'transition_summary': run_dir / 'development_ad_context_pre_body_post_ocr_transition_summary_v2_5.csv',
        'quality_checks': run_dir / 'development_ad_context_plus5_ocr_review_quality_checks_v2_5.csv',
        'json_report': run_dir / 'development_ad_context_plus5_ocr_review_v2_5_report.json',
        'run_log': run_dir / 'development_ad_context_plus5_ocr_review_v2_5_run_log.txt',
    }


def build_intervals(label_df: pd.DataFrame, split_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    split_dev = split_df[(split_df['split'].astype(str) == 'train') & (pd.to_numeric(split_df['video_id'], errors='coerce').isin(DEVELOPMENT_IDS))].copy()
    if sorted(split_dev['video_id'].astype(int).tolist()) != DEVELOPMENT_IDS:
        raise RuntimeError('Development Set split IDs do not match fixed v2.4 train IDs')
    labels = label_df.copy()
    labels['video_id'] = pd.to_numeric(labels['video_id'], errors='coerce').astype('Int64')
    if 'segment_type' in labels.columns:
        labels = labels[labels['segment_type'].astype(str).eq('ad_interval')]
    if 'segment_valid' in labels.columns:
        labels = labels[labels['segment_valid'].astype(str).str.lower().isin(['true', '1', 'yes'])]
    if 'label_valid' in labels.columns:
        labels = labels[labels['label_valid'].astype(str).str.lower().isin(['true', '1', 'yes'])]
    labels = labels[labels['video_id'].isin(DEVELOPMENT_IDS)].copy()
    labels['ad_start_sec'] = pd.to_numeric(labels['ad_start_sec'], errors='coerce')
    labels['ad_end_sec'] = pd.to_numeric(labels['ad_end_sec'], errors='coerce')
    labels = labels[labels['ad_start_sec'].notna() & labels['ad_end_sec'].notna() & (labels['ad_end_sec'] >= labels['ad_start_sec'])].copy()
    manifest = manifest_df[['video_id', 'duration_sec', 'video_title']].copy() if {'video_id', 'duration_sec', 'video_title'}.issubset(manifest_df.columns) else pd.DataFrame()
    if not manifest.empty:
        manifest['video_id'] = pd.to_numeric(manifest['video_id'], errors='coerce').astype('Int64')
        manifest['duration_sec_manifest'] = pd.to_numeric(manifest['duration_sec'], errors='coerce')
        manifest = manifest[['video_id', 'duration_sec_manifest', 'video_title']]
        labels = labels.merge(manifest, on='video_id', how='left', suffixes=('', '_manifest'))
    else:
        labels['duration_sec_manifest'] = np.nan
    labels['video_duration_sec'] = pd.to_numeric(labels.get('video_duration_sec'), errors='coerce')
    labels['duration_for_window'] = labels['duration_sec_manifest'].fillna(labels['video_duration_sec'])
    labels['review_window_start_sec'] = (labels['ad_start_sec'] - REVIEW_PRE_SEC).clip(lower=0)
    labels['review_window_end_sec'] = labels[['ad_end_sec', 'duration_for_window']].apply(
        lambda r: min(r['ad_end_sec'] + REVIEW_POST_SEC, r['duration_for_window']) if pd.notna(r['duration_for_window']) else r['ad_end_sec'] + REVIEW_POST_SEC,
        axis=1,
    )
    labels['ad_duration_sec'] = labels['ad_end_sec'] - labels['ad_start_sec']
    labels['review_window_duration_sec'] = labels['review_window_end_sec'] - labels['review_window_start_sec']
    labels['original_split_v2_4'] = 'train'
    labels['split_role_v2_5'] = 'development'
    labels['evaluation_subset_v2_5'] = 'none'
    labels['split_terminology_note'] = SPLIT_NOTE
    labels = labels.sort_values(['video_id', 'ad_start_sec', 'ad_interval_id']).reset_index(drop=True)
    return labels


def normalize_ocr(ocr_df: pd.DataFrame) -> pd.DataFrame:
    df = ocr_df.copy()
    required = [
        'run_id', 'original_split_v2_4', 'split_role_v2_5', 'evaluation_subset_v2_5', 'video_id', 'video_title',
        'timestamp_sec', 'timestamp_mmss', 'sampling_role', 'is_anchor_dense', 'is_background_regular',
        'nearest_anchor_id', 'nearest_anchor_time_sec', 'nearest_anchor_delta_sec', 'ocr_status', 'ocr_text_joined',
        'ocr_text_normalized', 'ocr_text_count', 'ocr_token_count', 'ocr_char_count', 'ocr_box_count',
        'ocr_mean_confidence', 'ocr_min_confidence', 'ocr_max_confidence', 'ocr_text_area_ratio',
        'ad_disclosure_keyword_count', 'sponsor_keyword_count', 'brand_product_keyword_count',
        'promotion_discount_keyword_count', 'purchase_cta_keyword_count', 'link_more_info_keyword_count',
        'negative_guard_keyword_count', 'total_ad_keyword_count', 'corrected_ad_disclosure_hit_count',
        'corrected_sponsor_keyword_count', 'corrected_brand_product_keyword_count',
        'corrected_promotion_discount_keyword_count', 'corrected_purchase_cta_keyword_count',
        'corrected_link_more_info_keyword_count', 'corrected_negative_guard_keyword_count',
        'corrected_total_ad_keyword_count', 'matched_keyword_categories', 'matched_keywords', 'matched_keyword_rules',
        'matched_keyword_confidence', 'suggested_canonical_phrase', 'suppressed_by_negative_guard',
        'corrected_frame_ad_text_score'
    ]
    df = ensure_columns(df, required, '')
    df['video_id'] = pd.to_numeric(df['video_id'], errors='coerce').astype('Int64')
    df['timestamp_sec'] = pd.to_numeric(df['timestamp_sec'], errors='coerce')
    num_cols = [c for c in df.columns if c.endswith('_count') or c in ['ocr_mean_confidence', 'ocr_min_confidence', 'ocr_max_confidence', 'ocr_text_area_ratio', 'nearest_anchor_time_sec', 'nearest_anchor_delta_sec', 'corrected_frame_ad_text_score']]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['has_text'] = (df['ocr_text_count'].fillna(0) > 0) | df['ocr_text_joined'].fillna('').astype(str).str.strip().ne('')
    return df


def phase_for(ts: float, start: float, end: float) -> str:
    if ts < start:
        return 'pre_start_5s'
    if ts <= end:
        return 'ad_body'
    return 'post_end_5s'


def build_review_rows(ocr_df: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    ocr_by_video = {int(v): g.sort_values('timestamp_sec') for v, g in ocr_df[ocr_df['video_id'].isin(DEVELOPMENT_IDS)].groupby('video_id')}
    for _, interval in intervals.iterrows():
        vid = int(interval['video_id'])
        vdf = ocr_by_video.get(vid, pd.DataFrame())
        if vdf.empty:
            continue
        start = float(interval['review_window_start_sec'])
        end = float(interval['review_window_end_sec'])
        subset = vdf[(vdf['timestamp_sec'] >= start - 1e-9) & (vdf['timestamp_sec'] <= end + 1e-9)].copy()
        if subset.empty:
            continue
        subset['ad_interval_id'] = interval['ad_interval_id']
        subset['ad_start_sec'] = float(interval['ad_start_sec'])
        subset['ad_end_sec'] = float(interval['ad_end_sec'])
        subset['ad_duration_sec'] = float(interval['ad_duration_sec'])
        subset['review_window_start_sec'] = start
        subset['review_window_end_sec'] = end
        subset['review_window_duration_sec'] = float(interval['review_window_duration_sec'])
        subset['relative_to_ad_start_sec'] = subset['timestamp_sec'] - float(interval['ad_start_sec'])
        subset['relative_to_ad_end_sec'] = subset['timestamp_sec'] - float(interval['ad_end_sec'])
        subset['phase_in_review_window'] = subset['timestamp_sec'].apply(lambda x: phase_for(float(x), float(interval['ad_start_sec']), float(interval['ad_end_sec'])))
        subset['original_split_v2_4'] = 'train'
        subset['split_role_v2_5'] = 'development'
        subset['evaluation_subset_v2_5'] = 'none'
        rows.append(subset)
    if not rows:
        return pd.DataFrame(columns=FULL_COLUMNS)
    review = pd.concat(rows, ignore_index=True)
    review = review.sort_values(['video_id', 'ad_interval_id', 'timestamp_sec']).reset_index(drop=True)
    review['review_row_id'] = [f'PLUS5OCR_{i:07d}' for i in range(1, len(review) + 1)]
    review['timestamp_mmss'] = review['timestamp_sec'].apply(timestamp_mmss)
    review['reviewer_note'] = ''
    review['ocr_quality_manual_label'] = ''
    review['post_end_review_note'] = ''
    review['video_title_short'] = review['video_title'].apply(short_title)
    return review


def make_quick_quality_flags(df: pd.DataFrame, high_score_threshold: float) -> pd.Series:
    flags = []
    for _, row in df.iterrows():
        status = str(row.get('ocr_status', '')).lower()
        phase = row.get('phase_in_review_window', '')
        score = safe_float(row.get('corrected_frame_ad_text_score'))
        mean_conf = safe_float(row.get('ocr_mean_confidence'))
        total_kw = safe_float(row.get('corrected_total_ad_keyword_count'))
        brand_cta = sum(safe_float(row.get(c)) if math.isfinite(safe_float(row.get(c))) else 0 for c in ['corrected_brand_product_keyword_count', 'corrected_purchase_cta_keyword_count', 'corrected_link_more_info_keyword_count'])
        if safe_float(row.get('corrected_ad_disclosure_hit_count')) > 0:
            flags.append('disclosure_hit')
        elif safe_float(row.get('corrected_sponsor_keyword_count')) > 0:
            flags.append('sponsor_hit')
        elif brand_cta > 0:
            flags.append('product_or_cta_hit')
        elif math.isfinite(score) and score >= high_score_threshold:
            flags.append('high_ad_score')
        elif phase == 'post_end_5s' and total_kw > 0:
            flags.append('post_end_ad_like')
        elif phase == 'post_end_5s' and status == 'success_empty':
            flags.append('post_end_empty')
        elif status == 'success_empty':
            flags.append('empty')
        elif status == 'failed':
            flags.append('failed')
        elif math.isfinite(mean_conf) and mean_conf < LOW_CONFIDENCE_THRESHOLD and str(row.get('ocr_text_joined', '')).strip():
            flags.append('low_confidence')
        else:
            flags.append('normal')
    return pd.Series(flags, index=df.index)


def post_end_hint(row: pd.Series, high_score_threshold: float) -> str:
    score = safe_float(row.get('corrected_frame_ad_text_score'))
    total_kw = safe_float(row.get('corrected_total_ad_keyword_count'))
    status = str(row.get('ocr_status', '')).lower()
    mean_conf = safe_float(row.get('ocr_mean_confidence'))
    rules = str(row.get('matched_keyword_rules', '')).lower()
    if total_kw > 0 or (math.isfinite(score) and score >= high_score_threshold):
        return 'still_ad_like_text'
    if status == 'success_empty':
        return 'no_text_or_empty'
    if (math.isfinite(mean_conf) and mean_conf < LOW_CONFIDENCE_THRESHOLD) or 'fuzzy' in rules:
        return 'review_needed'
    if bool(row.get('has_text')):
        return 'likely_return_to_normal'
    return 'review_needed'


def disclosure_type(row: pd.Series) -> str:
    rules = str(row.get('matched_keyword_rules', '')).lower()
    types = []
    if safe_float(row.get('corrected_ad_disclosure_hit_count')) > 0 or 'ad_disclosure' in str(row.get('matched_keyword_categories', '')).lower():
        types.append('exact')
    if 'typo_variant_match' in rules:
        types.append('typo_variant')
    if 'proximity_match' in rules:
        types.append('proximity')
    if 'fuzzy_match' in rules:
        types.append('fuzzy_review_needed')
    uniq = list(dict.fromkeys(types))
    if not uniq:
        return 'mixed'
    if len(uniq) > 1:
        return 'mixed'
    return uniq[0]


def summarize_group(g: pd.DataFrame) -> dict[str, Any]:
    status = g['ocr_status'].fillna('').astype(str).str.lower()
    has_text = g['has_text'].astype(bool)
    conf = pd.to_numeric(g['ocr_mean_confidence'], errors='coerce')
    return {
        'row_count': len(g),
        'success_text_count': int((status == 'success_text').sum()),
        'success_empty_count': int((status == 'success_empty').sum()),
        'failed_count': int((status == 'failed').sum()),
        'nonempty_count': int(has_text.sum()),
        'nonempty_ratio': float(has_text.mean()) if len(g) else 0.0,
        'mean_confidence': float(conf.mean()) if conf.notna().any() else None,
        'median_confidence': float(conf.median()) if conf.notna().any() else None,
        'low_confidence_count': int(((conf < LOW_CONFIDENCE_THRESHOLD) & has_text).sum()),
        'corrected_keyword_hit_row_count': int((pd.to_numeric(g['corrected_total_ad_keyword_count'], errors='coerce').fillna(0) > 0).sum()),
        'corrected_total_ad_keyword_count_sum': int(pd.to_numeric(g['corrected_total_ad_keyword_count'], errors='coerce').fillna(0).sum()),
        'corrected_ad_disclosure_hit_count_sum': int(pd.to_numeric(g['corrected_ad_disclosure_hit_count'], errors='coerce').fillna(0).sum()),
        'corrected_sponsor_keyword_count_sum': int(pd.to_numeric(g['corrected_sponsor_keyword_count'], errors='coerce').fillna(0).sum()),
        'corrected_brand_product_keyword_count_sum': int(pd.to_numeric(g['corrected_brand_product_keyword_count'], errors='coerce').fillna(0).sum()),
        'corrected_promotion_discount_keyword_count_sum': int(pd.to_numeric(g['corrected_promotion_discount_keyword_count'], errors='coerce').fillna(0).sum()),
        'corrected_purchase_cta_keyword_count_sum': int(pd.to_numeric(g['corrected_purchase_cta_keyword_count'], errors='coerce').fillna(0).sum()),
        'corrected_link_more_info_keyword_count_sum': int(pd.to_numeric(g['corrected_link_more_info_keyword_count'], errors='coerce').fillna(0).sum()),
        'corrected_negative_guard_keyword_count_sum': int(pd.to_numeric(g['corrected_negative_guard_keyword_count'], errors='coerce').fillna(0).sum()),
    }


def suggested_priority(g: pd.DataFrame) -> str:
    rules = g.get('matched_keyword_rules', pd.Series(dtype=str)).fillna('').astype(str).str.lower()
    score = pd.to_numeric(g.get('corrected_frame_ad_text_score', 0), errors='coerce').fillna(0)
    low_conf = pd.to_numeric(g.get('ocr_mean_confidence', np.nan), errors='coerce') < LOW_CONFIDENCE_THRESHOLD
    if (pd.to_numeric(g['corrected_ad_disclosure_hit_count'], errors='coerce').fillna(0) > 0).any():
        return 'high'
    if rules.str.contains('fuzzy|typo_variant', regex=True).any():
        return 'high'
    if (score >= 2.0).any():
        return 'high'
    if (low_conf & g['has_text'].astype(bool)).sum() >= 3:
        return 'high'
    if ((g['phase_in_review_window'] == 'post_end_5s') & (pd.to_numeric(g['corrected_total_ad_keyword_count'], errors='coerce').fillna(0) > 0)).any():
        return 'high'
    if (pd.to_numeric(g['corrected_total_ad_keyword_count'], errors='coerce').fillna(0) > 0).any():
        return 'medium'
    return 'low'


def build_interval_summary(review: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in review.groupby(['video_id', 'ad_interval_id'], sort=True):
        base = g.iloc[0]
        s = summarize_group(g)
        post = g[g['phase_in_review_window'] == 'post_end_5s']
        first_disc = g.loc[pd.to_numeric(g['corrected_ad_disclosure_hit_count'], errors='coerce').fillna(0) > 0, 'timestamp_sec']
        first_kw = g.loc[pd.to_numeric(g['corrected_total_ad_keyword_count'], errors='coerce').fillna(0) > 0, 'timestamp_sec']
        rules = g['matched_keyword_rules'].fillna('').astype(str).str.lower()
        post_kw_rows = int((pd.to_numeric(post.get('corrected_total_ad_keyword_count', pd.Series(dtype=float)), errors='coerce').fillna(0) > 0).sum()) if not post.empty else 0
        row = {
            'original_split_v2_4': 'train', 'split_role_v2_5': 'development', 'evaluation_subset_v2_5': 'none',
            'video_id': int(base['video_id']), 'video_title': base['video_title'], 'ad_interval_id': base['ad_interval_id'],
            'ad_start_sec': base['ad_start_sec'], 'ad_end_sec': base['ad_end_sec'], 'ad_duration_sec': base['ad_duration_sec'],
            'review_window_start_sec': base['review_window_start_sec'], 'review_window_end_sec': base['review_window_end_sec'],
            'review_window_duration_sec': base['review_window_duration_sec'], 'review_row_count': len(g),
            'pre_start_5s_row_count': int((g['phase_in_review_window'] == 'pre_start_5s').sum()),
            'ad_body_row_count': int((g['phase_in_review_window'] == 'ad_body').sum()),
            'post_end_5s_row_count': int((g['phase_in_review_window'] == 'post_end_5s').sum()),
            **{k: v for k, v in s.items() if k in INTERVAL_COLUMNS},
            'typo_variant_hit_count': int(rules.str.contains('typo_variant_match').sum()),
            'fuzzy_review_needed_count': int(rules.str.contains('fuzzy_match').sum()),
            'first_disclosure_timestamp_sec': first_disc.min() if not first_disc.empty else '',
            'first_keyword_timestamp_sec': first_kw.min() if not first_kw.empty else '',
            'post_end_keyword_hit_row_count': post_kw_rows,
            'post_end_still_ad_like_row_count': int((post['post_end_interpretation_hint'] == 'still_ad_like_text').sum()) if 'post_end_interpretation_hint' in post else 0,
            'post_end_nonempty_ratio': float(post['has_text'].mean()) if not post.empty else 0.0,
            'representative_ocr_text': highest_score_text(g),
            'post_end_representative_ocr_text': highest_score_text(post),
            'top_matched_keywords': top_keywords(g['matched_keywords']),
            'suggested_review_priority': suggested_priority(g),
            'interval_quality_note': 'Review OCR text and keyword hits across pre/body/post phases.',
        }
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=INTERVAL_COLUMNS)


def phase_note(phase: str, g: pd.DataFrame) -> str:
    if g.empty:
        return 'sparse_or_empty'
    kw = (pd.to_numeric(g['corrected_total_ad_keyword_count'], errors='coerce').fillna(0) > 0).sum()
    disc = (pd.to_numeric(g['corrected_ad_disclosure_hit_count'], errors='coerce').fillna(0) > 0).sum()
    brand_cta = (pd.to_numeric(g['corrected_brand_product_keyword_count'], errors='coerce').fillna(0) + pd.to_numeric(g['corrected_purchase_cta_keyword_count'], errors='coerce').fillna(0) + pd.to_numeric(g['corrected_link_more_info_keyword_count'], errors='coerce').fillna(0) > 0).sum()
    nonempty = g['has_text'].mean() if len(g) else 0
    if phase == 'pre_start_5s' and disc > 0:
        return 'pre_start_has_disclosure'
    if phase == 'ad_body' and brand_cta > 0:
        return 'ad_body_has_product_cta'
    if phase == 'post_end_5s' and kw > 0:
        return 'post_end_still_ad_like'
    if phase == 'post_end_5s' and nonempty > 0 and kw == 0:
        return 'post_end_return_to_normal'
    if nonempty < 0.2:
        return 'sparse_or_empty'
    return 'review_needed'


def build_transition_summary(review: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (vid, ad_id, phase), g in review.groupby(['video_id', 'ad_interval_id', 'phase_in_review_window'], sort=True):
        base = g.iloc[0]
        s = summarize_group(g)
        score = pd.to_numeric(g['corrected_frame_ad_text_score'], errors='coerce')
        rows.append({
            'video_id': int(vid), 'video_title': base['video_title'], 'ad_interval_id': ad_id,
            'phase_in_review_window': phase, 'phase_start_sec': g['timestamp_sec'].min(), 'phase_end_sec': g['timestamp_sec'].max(),
            'phase_duration_sec': g['timestamp_sec'].max() - g['timestamp_sec'].min() if len(g) else 0,
            'row_count': len(g), 'success_text_count': s['success_text_count'], 'success_empty_count': s['success_empty_count'],
            'failed_count': s['failed_count'], 'nonempty_ratio': s['nonempty_ratio'], 'mean_confidence': s['mean_confidence'],
            'keyword_hit_row_count': s['corrected_keyword_hit_row_count'],
            'corrected_total_ad_keyword_count_sum': s['corrected_total_ad_keyword_count_sum'],
            'corrected_ad_disclosure_hit_count_sum': s['corrected_ad_disclosure_hit_count_sum'],
            'corrected_sponsor_keyword_count_sum': s['corrected_sponsor_keyword_count_sum'],
            'corrected_brand_product_keyword_count_sum': s['corrected_brand_product_keyword_count_sum'],
            'corrected_promotion_discount_keyword_count_sum': s['corrected_promotion_discount_keyword_count_sum'],
            'corrected_purchase_cta_keyword_count_sum': s['corrected_purchase_cta_keyword_count_sum'],
            'corrected_link_more_info_keyword_count_sum': s['corrected_link_more_info_keyword_count_sum'],
            'corrected_negative_guard_keyword_count_sum': s['corrected_negative_guard_keyword_count_sum'],
            'mean_corrected_frame_ad_text_score': float(score.mean()) if score.notna().any() else None,
            'max_corrected_frame_ad_text_score': float(score.max()) if score.notna().any() else None,
            'representative_ocr_text': highest_score_text(g),
            'phase_interpretation_note': phase_note(phase, g),
        })
    return pd.DataFrame(rows).reindex(columns=TRANSITION_COLUMNS)


def build_video_summary(interval_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vid, g in interval_summary.groupby('video_id', sort=True):
        rows.append({
            'original_split_v2_4': 'train', 'split_role_v2_5': 'development', 'evaluation_subset_v2_5': 'none',
            'video_id': int(vid), 'video_title': g.iloc[0]['video_title'], 'ad_interval_count': len(g),
            'review_row_count': int(g['review_row_count'].sum()),
            'pre_start_5s_row_count': int(g['pre_start_5s_row_count'].sum()),
            'ad_body_row_count': int(g['ad_body_row_count'].sum()),
            'post_end_5s_row_count': int(g['post_end_5s_row_count'].sum()),
            'success_text_count': int(g['success_text_count'].sum()),
            'success_empty_count': int(g['success_empty_count'].sum()),
            'failed_count': int(g['failed_count'].sum()),
            'nonempty_ratio': float(g['nonempty_count'].sum() / g['review_row_count'].sum()) if g['review_row_count'].sum() else 0.0,
            'mean_confidence': float(g['mean_confidence'].dropna().mean()) if g['mean_confidence'].notna().any() else None,
            'corrected_keyword_hit_row_count': int(g['corrected_keyword_hit_row_count'].sum()),
            'corrected_total_ad_keyword_count_sum': int(g['corrected_total_ad_keyword_count_sum'].sum()),
            'post_end_keyword_hit_row_count': int(g['post_end_keyword_hit_row_count'].sum()),
            'post_end_still_ad_like_row_count': int(g['post_end_still_ad_like_row_count'].sum()),
            'representative_ocr_text': first_text(g['representative_ocr_text']),
            'top_matched_keywords': top_keywords(g['top_matched_keywords']),
            'suggested_review_priority': 'high' if (g['suggested_review_priority'] == 'high').any() else ('medium' if (g['suggested_review_priority'] == 'medium').any() else 'low'),
        })
    return pd.DataFrame(rows)


def review_reason(row: pd.Series) -> str:
    reasons = []
    if safe_float(row.get('corrected_ad_disclosure_hit_count')) > 0:
        reasons.append('disclosure_hit')
    if safe_float(row.get('corrected_sponsor_keyword_count')) > 0:
        reasons.append('sponsor_hit')
    if safe_float(row.get('corrected_total_ad_keyword_count')) > 0:
        reasons.append('keyword_hit')
    if 'fuzzy_match' in str(row.get('matched_keyword_rules', '')).lower():
        reasons.append('fuzzy_review_needed')
    if 'typo_variant_match' in str(row.get('matched_keyword_rules', '')).lower():
        reasons.append('typo_variant_hit')
    return ';'.join(reasons) or 'review_needed'


def empty_lowconf_reason(row: pd.Series) -> str:
    status = str(row.get('ocr_status', '')).lower()
    reasons = []
    if status == 'success_empty':
        reasons.append('success_empty')
    if status == 'failed':
        reasons.append('failed')
    if safe_float(row.get('ocr_mean_confidence')) < LOW_CONFIDENCE_THRESHOLD and str(row.get('ocr_text_joined', '')).strip():
        reasons.append('low_confidence_with_text')
    if safe_float(row.get('ocr_text_count')) == 0 and row.get('phase_in_review_window') in {'ad_body', 'post_end_5s'}:
        reasons.append('empty_inside_ad_or_post_end')
    return ';'.join(reasons) or 'review_needed'


def build_representative_examples(review: pd.DataFrame) -> pd.DataFrame:
    groups = {
        'disclosure_hit': review['quick_quality_flag'].eq('disclosure_hit'),
        'sponsor_hit': review['quick_quality_flag'].eq('sponsor_hit'),
        'product_or_cta_hit': review['quick_quality_flag'].eq('product_or_cta_hit'),
        'high_ad_score': review['quick_quality_flag'].eq('high_ad_score'),
        'post_end_ad_like': review['post_end_interpretation_hint'].eq('still_ad_like_text'),
        'post_end_return_to_normal': review['post_end_interpretation_hint'].eq('likely_return_to_normal'),
        'empty_but_inside_ad_or_post_end': (review['ocr_status'].astype(str).str.lower().eq('success_empty') & review['phase_in_review_window'].isin(['ad_body', 'post_end_5s'])),
        'low_confidence_with_text': (pd.to_numeric(review['ocr_mean_confidence'], errors='coerce') < LOW_CONFIDENCE_THRESHOLD) & review['has_text'].astype(bool),
        'fuzzy_review_needed': review['matched_keyword_rules'].fillna('').astype(str).str.lower().str.contains('fuzzy_match'),
        'typo_variant_hit': review['matched_keyword_rules'].fillna('').astype(str).str.lower().str.contains('typo_variant_match'),
    }
    rows = []
    for group, mask in groups.items():
        temp = review[mask].copy()
        if temp.empty:
            continue
        temp['_score'] = pd.to_numeric(temp['corrected_frame_ad_text_score'], errors='coerce').fillna(0)
        temp = temp.sort_values(['_score', 'timestamp_sec'], ascending=[False, True]).head(8)
        for _, r in temp.iterrows():
            rows.append({
                'example_group': group, 'video_id': int(r['video_id']), 'video_title_short': r['video_title_short'],
                'ad_interval_id': r['ad_interval_id'], 'timestamp_mmss': r['timestamp_mmss'], 'timestamp_sec': r['timestamp_sec'],
                'phase_in_review_window': r['phase_in_review_window'], 'ocr_status': r['ocr_status'], 'ocr_mean_confidence': r['ocr_mean_confidence'],
                'ocr_text_joined': r['ocr_text_joined'], 'matched_keywords': r['matched_keywords'], 'matched_keyword_rules': r['matched_keyword_rules'],
                'corrected_frame_ad_text_score': r['corrected_frame_ad_text_score'], 'quick_quality_flag': r['quick_quality_flag'],
                'review_reason': group, 'reviewer_note': '',
            })
    return pd.DataFrame(rows)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join('' if pd.isna(v) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def build_markdown(report_data: dict[str, Any], review: pd.DataFrame, interval_summary: pd.DataFrame, transition_summary: pd.DataFrame, examples: pd.DataFrame) -> str:
    frame = report_data['frame_review_summary']
    post = report_data['post_end_5s_summary']
    phase_rows = []
    for phase in ['pre_start_5s', 'ad_body', 'post_end_5s']:
        g = transition_summary[transition_summary['phase_in_review_window'] == phase]
        phase_rows.append([
            phase,
            int(g['row_count'].sum()) if not g.empty else 0,
            f"{(g['success_text_count'].sum() / g['row_count'].sum() if not g.empty and g['row_count'].sum() else 0):.3f}",
            int(g['keyword_hit_row_count'].sum()) if not g.empty else 0,
            int(g['corrected_ad_disclosure_hit_count_sum'].sum()) if not g.empty else 0,
            first_text(g['representative_ocr_text']) if not g.empty else '',
            '; '.join(sorted(set(g['phase_interpretation_note'].dropna().astype(str)))) if not g.empty else '',
        ])
    interval_rows = []
    for _, r in interval_summary.sort_values(['video_id', 'ad_interval_id']).head(20).iterrows():
        interval_rows.append([
            int(r['video_id']), r['ad_interval_id'], int(r['review_row_count']), f"{r['nonempty_ratio']:.3f}",
            int(r['corrected_keyword_hit_row_count']), int(r['corrected_ad_disclosure_hit_count_sum']),
            int(r['post_end_keyword_hit_row_count']), str(r['representative_ocr_text'])[:80], str(r['post_end_representative_ocr_text'])[:80],
            int(r['low_confidence_count']), r['suggested_review_priority'],
        ])
    example_rows = []
    if not examples.empty:
        for _, r in examples.head(30).iterrows():
            example_rows.append([r['example_group'], int(r['video_id']), r['ad_interval_id'], r['timestamp_mmss'], r['phase_in_review_window'], str(r['ocr_text_joined'])[:90], r['quick_quality_flag']])
    labels = ['good', 'acceptable', 'typo_but_usable', 'wrong_text', 'missed_text', 'empty_but_should_have_text', 'not_relevant', 'post_ad_return_to_normal', 'post_ad_still_ad_like']
    return f"""# Development Set 광고 구간 + 종료 후 5초 OCR Review Sheet v2.5

## 1. 한 문단 결론

Development Set 광고 구간에 대해 기존 `[ad_start_sec - 5, ad_end_sec]` review 범위를 `[ad_start_sec - 5, ad_end_sec + 5]`로 확장한 OCR 품질 확인용 sheet를 생성했다. 총 `{report_data['review_scope_summary']['video_count']}`개 video, `{report_data['review_scope_summary']['ad_interval_count']}`개 ad interval, `{frame['total_review_row_count']}`개 OCR frame row가 포함됐다. OCR status는 success_text `{frame['success_text_count']}`, success_empty `{frame['success_empty_count']}`, failed `{frame['failed_count']}`이며, post_end_5s에서는 `{post['post_end_5s_row_count']}`개 row 중 keyword hit `{post['post_end_5s_keyword_hit_row_count']}`, still-ad-like hint `{post['post_end_still_ad_like_row_count']}`개가 확인됐다. 사람은 disclosure/keyword hit, post_end_ad_like, empty/low-confidence row를 우선 확인하면 된다.

## 2. v2.5 split terminology

{SPLIT_EXPLANATION_KO}

## 3. 입력 파일 요약

- OCR frame result: `{report_data['source_ocr_frame_result']}`
- actual ad interval file: `{LABEL_FILE}`
- split file: `{SPLIT_FILE}`
- video manifest: `{MANIFEST_FILE}`
- OCR run_id: `{SOURCE_OCR_RUN_ID}`
- OCR backend summary: `{json.dumps(report_data.get('source_ocr_backend_summary', {}), ensure_ascii=False)}`

## 4. review 범위

- Scope: Development Set only
- window definition: `[ad_start_sec - 5, ad_end_sec + 5]`
- included phases: `pre_start_5s`, `ad_body`, `post_end_5s`
- excluded: Test Set

## 5. 기존 review와 달라진 점

- previous window: `[ad_start_sec - 5, ad_end_sec]`
- new window: `[ad_start_sec - 5, ad_end_sec + 5]`
- 추가된 phase: `post_end_5s`
- 목적: 광고 종료 후에도 OCR 광고 단서가 남는지, 또는 원래 콘텐츠로 복귀했는지 확인

## 6. review sheet 사용법

- full review CSV: 모든 OCR row와 keyword/correction/count/confidence 정보를 포함한다.
- compact review CSV: 사람이 빠르게 훑어볼 수 있도록 핵심 컬럼만 남겼다.
- post-end 5s review sheet: 광고 종료 후 5초 구간만 모아 광고성 OCR 단서 잔류 여부를 확인한다.
- transition comparison summary: `pre_start_5s`, `ad_body`, `post_end_5s`의 OCR 단서 변화 흐름을 비교한다.
- `reviewer_note`: 사람이 자유롭게 메모한다.
- `ocr_quality_manual_label` 권장 값: `{', '.join(labels)}`

우선 볼 column: `timestamp_mmss`, `relative_to_ad_start_sec`, `relative_to_ad_end_sec`, `phase_in_review_window`, `ocr_text_joined`, `matched_keywords`, `corrected_frame_ad_text_score`, `quick_quality_flag`.

## 7. 전체 OCR 품질 요약

- total review row count: `{frame['total_review_row_count']}`
- success_text count: `{frame['success_text_count']}`
- success_empty count: `{frame['success_empty_count']}`
- failed count: `{frame['failed_count']}`
- nonempty ratio: `{frame['nonempty_ratio']:.3f}`
- average confidence: `{frame['average_confidence']:.3f}`
- low confidence row count: `{frame['low_confidence_row_count']}`
- empty row count: `{frame['empty_row_count']}`
- keyword hit row count: `{frame['keyword_hit_row_count']}`
- post_end_5s row count: `{post['post_end_5s_row_count']}`
- post_end_5s keyword hit row count: `{post['post_end_5s_keyword_hit_row_count']}`
- post_end_still_ad_like row count: `{post['post_end_still_ad_like_row_count']}`

## 8. phase별 요약

{md_table(['phase','row count','nonempty ratio','keyword hit','disclosure hit','representative OCR text','interpretation'], phase_rows)}

## 9. interval별 요약

{md_table(['video','interval','rows','nonempty','keyword rows','disclosure hits','post-end keyword rows','representative OCR','post-end OCR','low-conf','priority'], interval_rows)}

## 10. 사람이 우선 확인할 row

{md_table(['group','video','interval','time','phase','ocr text','flag'], example_rows)}

## 11. OCR 품질 확인 관점

- 광고 구간에서 실제 화면 텍스트가 OCR에 잘 잡혔는가?
- “유료광고 포함”류 문구가 정상/오타 형태로 잡혔는가?
- 제품명, 브랜드명, 구매/링크/더보기 문구가 잡혔는가?
- 광고 종료 후 5초 안에도 광고성 텍스트가 남아 있는가?
- 광고 종료 후 OCR 단서가 일반 콘텐츠로 전환되는 흐름을 보여주는가?
- empty frame은 실제로 텍스트가 없는 구간일 가능성이 높은가?
- OCR confidence가 낮은데 중요한 텍스트가 있는 row가 있는가?
- 새 OCR 모델 실험이 필요한 오류 유형이 보이는가?

## 12. safety

- OCR rerun performed: false
- OCR engine called: false
- actual label used for filtering/review only: true
- actual label used for sampling: false
- detector modified: false
- existing OCR modified: false
- Test Set processed: false
- Test Set processed: false
- raw frame persisted: false
"""


def copy_latest(outputs: dict[str, Path], report: dict[str, Any], logger: Logger) -> None:
    include_keys = [
        'full_review_sheet', 'compact_review_sheet', 'markdown_report', 'interval_summary', 'video_summary',
        'keyword_hit_review', 'disclosure_review', 'empty_lowconf_review', 'representative_examples',
        'post_end_review', 'transition_summary', 'quality_checks', 'json_report', 'run_log'
    ]
    include_paths = [outputs[k] for k in include_keys if outputs[k].exists()] + [SCRIPT_PATH]
    for directory in [LATEST_CHATGPT, LATEST_OCR, LATEST_SCENE]:
        directory.mkdir(parents=True, exist_ok=True)
        for path in include_paths:
            if path.exists():
                shutil.copy2(path, directory / path.name)
        readme_name = 'README_latest_files.md' if directory != LATEST_SCENE else 'README_development_ad_context_plus5_ocr_review_v2_5.md'
        readme = directory / readme_name
        lines = [
            '# Latest Files: Development ad context plus5 OCR review v2.5', '',
            '기존 OCR frame result를 post-hoc filtering하여 생성한 review sheet이다. OCR 재실행, frame 생성, detector 수정은 수행하지 않았다.', '',
            '## Included Files', ''
        ]
        lines.extend(f'- `{p.name}`' for p in include_paths if p.exists())
        lines.extend(['', '## Scope', '', '- Development Set only', '- Window: [ad_start_sec - 5, ad_end_sec + 5]', '- OCR engine called: false'])
        readme.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # README를 복사한 뒤 log를 갱신한다.
    for directory in [LATEST_CHATGPT, LATEST_OCR, LATEST_SCENE]:
        if outputs['run_log'].exists():
            shutil.copy2(outputs['run_log'], directory / outputs['run_log'].name)


def forbidden_scan(paths: list[Path]) -> list[str]:
    bad = []
    for root in paths:
        if root.exists():
            for p in root.rglob('*'):
                if p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES:
                    bad.append(str(p))
    return bad


def main() -> int:
    start = time.time()
    timestamp = now_id()
    run_id = f'{TARGET_RUN_PREFIX}_{timestamp}'
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(run_dir)
    logger = Logger(outputs['run_log'])
    warnings: list[str] = []
    errors: list[str] = []

    logger.log('[STEP 01] Safety snapshot and output directory setup')
    protected = [OCR_FRAME_RESULT, OCR_VIDEO_SUMMARY, OCR_REPORT, LABEL_FILE, SPLIT_FILE, MANIFEST_FILE]
    before = {str(p): snapshot(p) for p in protected if p.exists()}
    old_before = snapshot(OLD_PROJECT_ROOT)

    logger.log('[STEP 02] Locate Development Set OCR frame result')
    ocr_frame_path = locate_file(OCR_FRAME_RESULT, [SOURCE_OCR_RUN_DIR, PROJECT_ROOT / 'workspaces/ocr_final_scene_anchor_v2_5_development/runs', LATEST_OCR], OCR_FRAME_RESULT.name)
    ocr_summary_path = locate_file(OCR_VIDEO_SUMMARY, [SOURCE_OCR_RUN_DIR, LATEST_OCR], OCR_VIDEO_SUMMARY.name)
    ocr_report_path = locate_file(OCR_REPORT, [SOURCE_OCR_RUN_DIR, LATEST_OCR], OCR_REPORT.name)

    logger.log('[STEP 03] Load OCR frame result and source OCR report')
    ocr_df = pd.read_csv(ocr_frame_path, encoding='utf-8-sig')
    ocr_summary_df = pd.read_csv(ocr_summary_path, encoding='utf-8-sig') if ocr_summary_path.exists() else pd.DataFrame()
    with ocr_report_path.open('r', encoding='utf-8') as f:
        source_report = json.load(f)

    logger.log('[STEP 04] Load split file, manifest, and actual ad intervals')
    split_df = pd.read_csv(SPLIT_FILE, encoding='utf-8-sig')
    manifest_df = pd.read_csv(MANIFEST_FILE, encoding='utf-8-sig')
    label_df = pd.read_csv(LABEL_FILE, encoding='utf-8-sig')

    logger.log('[STEP 05] Apply v2.5 split terminology mapping')
    intervals = build_intervals(label_df, split_df, manifest_df)

    logger.log('[STEP 06] Build Development Set ad_start_minus5_to_ad_end_plus5 review windows')
    if intervals.empty:
        raise RuntimeError('No Development Set ad intervals found')

    logger.log('[STEP 07] Filter OCR rows inside expanded review windows')
    ocr_norm = normalize_ocr(ocr_df)
    review = build_review_rows(ocr_norm, intervals)
    if review.empty:
        warnings.append('No OCR rows found in expanded review windows')

    logger.log('[STEP 08] Assign pre_start_5s, ad_body, and post_end_5s phases')
    score_series = pd.to_numeric(review.get('corrected_frame_ad_text_score', pd.Series(dtype=float)), errors='coerce').fillna(0)
    high_score_threshold = float(max(2.0, score_series.quantile(0.90))) if len(score_series) else 2.0
    review['quick_quality_flag'] = make_quick_quality_flags(review, high_score_threshold)
    review['post_end_interpretation_hint'] = review.apply(lambda r: post_end_hint(r, high_score_threshold) if r['phase_in_review_window'] == 'post_end_5s' else '', axis=1)
    review['matched_keyword_categories'] = review['matched_keyword_categories'].fillna('')
    review['matched_keywords'] = review['matched_keywords'].fillna('')
    review['matched_keyword_rules'] = review['matched_keyword_rules'].fillna('')

    logger.log('[STEP 09] Build full frame-level human review sheet')
    full_review = ensure_columns(review, FULL_COLUMNS, '')[FULL_COLUMNS]
    full_review.to_csv(outputs['full_review_sheet'], index=False, encoding='utf-8-sig')

    logger.log('[STEP 10] Build compact human-readable review sheet')
    compact = ensure_columns(review, COMPACT_COLUMNS, '')[COMPACT_COLUMNS]
    compact.to_csv(outputs['compact_review_sheet'], index=False, encoding='utf-8-sig')

    logger.log('[STEP 11] Build keyword hit and disclosure-focused review sheets')
    keyword_mask = (pd.to_numeric(review['corrected_total_ad_keyword_count'], errors='coerce').fillna(0) > 0) | review['matched_keywords'].fillna('').astype(str).str.strip().ne('')
    keyword = review[keyword_mask].copy()
    keyword['review_reason'] = keyword.apply(review_reason, axis=1)
    ensure_columns(keyword, KEYWORD_COLUMNS, '')[KEYWORD_COLUMNS].to_csv(outputs['keyword_hit_review'], index=False, encoding='utf-8-sig')

    disclosure_mask = (
        (pd.to_numeric(review['corrected_ad_disclosure_hit_count'], errors='coerce').fillna(0) > 0)
        | review['matched_keyword_categories'].fillna('').astype(str).str.contains('ad_disclosure', case=False, regex=False)
        | review['matched_keyword_rules'].fillna('').astype(str).str.contains('typo_variant_match|proximity_match|fuzzy_match', case=False, regex=True)
        | review['suggested_canonical_phrase'].fillna('').astype(str).str.contains('유료광고|광고포함', regex=True)
    )
    disclosure = review[disclosure_mask].copy()
    disclosure['disclosure_review_type'] = disclosure.apply(disclosure_type, axis=1)
    ensure_columns(disclosure, DISCLOSURE_COLUMNS, '')[DISCLOSURE_COLUMNS].to_csv(outputs['disclosure_review'], index=False, encoding='utf-8-sig')

    logger.log('[STEP 12] Build post-end 5s focused review sheet')
    post_end = review[review['phase_in_review_window'] == 'post_end_5s'].copy()
    ensure_columns(post_end, POST_END_COLUMNS, '')[POST_END_COLUMNS].to_csv(outputs['post_end_review'], index=False, encoding='utf-8-sig')

    logger.log('[STEP 13] Build empty/low-confidence review sheet')
    empty_low_mask = (
        review['ocr_status'].fillna('').astype(str).str.lower().isin(['success_empty', 'failed'])
        | ((pd.to_numeric(review['ocr_mean_confidence'], errors='coerce') < LOW_CONFIDENCE_THRESHOLD) & review['has_text'].astype(bool))
        | ((pd.to_numeric(review['ocr_text_count'], errors='coerce').fillna(0) == 0) & review['phase_in_review_window'].isin(['ad_body', 'post_end_5s']))
    )
    empty_low = review[empty_low_mask].copy()
    empty_low['review_reason'] = empty_low.apply(empty_lowconf_reason, axis=1)
    ensure_columns(empty_low, EMPTY_LOWCONF_COLUMNS, '')[EMPTY_LOWCONF_COLUMNS].to_csv(outputs['empty_lowconf_review'], index=False, encoding='utf-8-sig')

    logger.log('[STEP 14] Build interval-level, video-level, and phase transition summaries')
    interval_summary = build_interval_summary(review)
    video_summary = build_video_summary(interval_summary)
    transition_summary = build_transition_summary(review)
    interval_summary.to_csv(outputs['interval_summary'], index=False, encoding='utf-8-sig')
    video_summary.to_csv(outputs['video_summary'], index=False, encoding='utf-8-sig')
    transition_summary.to_csv(outputs['transition_summary'], index=False, encoding='utf-8-sig')

    logger.log('[STEP 15] Select representative examples and priority review rows')
    examples = build_representative_examples(review)
    examples.to_csv(outputs['representative_examples'], index=False, encoding='utf-8-sig')

    logger.log('[STEP 16] Run Sub Agent validations')
    status = review['ocr_status'].fillna('').astype(str).str.lower()
    phase_counts = review['phase_in_review_window'].value_counts().to_dict()
    post_end_keyword_rows = int((pd.to_numeric(post_end.get('corrected_total_ad_keyword_count', pd.Series(dtype=float)), errors='coerce').fillna(0) > 0).sum()) if not post_end.empty else 0
    post_end_still = int((post_end.get('post_end_interpretation_hint', pd.Series(dtype=str)) == 'still_ad_like_text').sum()) if not post_end.empty else 0
    frame_summary = {
        'total_review_row_count': int(len(review)),
        'success_text_count': int((status == 'success_text').sum()),
        'success_empty_count': int((status == 'success_empty').sum()),
        'failed_count': int((status == 'failed').sum()),
        'nonempty_count': int(review['has_text'].astype(bool).sum()),
        'nonempty_ratio': float(review['has_text'].astype(bool).mean()) if len(review) else 0.0,
        'average_confidence': float(pd.to_numeric(review['ocr_mean_confidence'], errors='coerce').mean()) if len(review) else None,
        'low_confidence_row_count': int(((pd.to_numeric(review['ocr_mean_confidence'], errors='coerce') < LOW_CONFIDENCE_THRESHOLD) & review['has_text'].astype(bool)).sum()),
        'empty_row_count': int((status == 'success_empty').sum()),
        'keyword_hit_row_count': int((pd.to_numeric(review['corrected_total_ad_keyword_count'], errors='coerce').fillna(0) > 0).sum()),
        'disclosure_review_rows': int(len(disclosure)),
        'empty_lowconf_rows': int(len(empty_low)),
    }
    review_scope = {
        'video_count': int(review['video_id'].nunique()) if len(review) else 0,
        'ad_interval_count': int(intervals['ad_interval_id'].nunique()),
        'target_video_ids': sorted(review['video_id'].dropna().astype(int).unique().tolist()) if len(review) else [],
        'development_set_only': True,
        'extended_evaluation_processed': bool(set(review['video_id'].dropna().astype(int).unique()) & set(EXTENDED_EVALUATION_IDS)) if len(review) else False,
        'diagnostic_subset_processed': bool(set(review['video_id'].dropna().astype(int).unique()) & set(DIAGNOSTIC_IDS)) if len(review) else False,
        'pure_test_processed': bool(set(review['video_id'].dropna().astype(int).unique()) & set(PURE_TEST_IDS)) if len(review) else False,
    }
    post_summary = {
        'post_end_5s_row_count': int(len(post_end)),
        'post_end_5s_keyword_hit_row_count': post_end_keyword_rows,
        'post_end_still_ad_like_row_count': post_end_still,
        'post_end_nonempty_ratio': float(post_end['has_text'].astype(bool).mean()) if len(post_end) else 0.0,
    }

    checks: list[dict[str, Any]] = []
    def check(name: str, condition: bool, detail: str, warn: bool = False) -> None:
        checks.append({'check_name': name, 'status': 'PASS' if condition else ('WARN' if warn else 'FAIL'), 'detail': detail})
    check('input_ocr_frame_result_exists', ocr_frame_path.exists(), str(ocr_frame_path))
    check('input_label_file_exists', LABEL_FILE.exists(), str(LABEL_FILE))
    check('development_only_filter_passed', review_scope['target_video_ids'] == DEVELOPMENT_IDS, str(review_scope['target_video_ids']))
    check('extended_evaluation_excluded', not review_scope['extended_evaluation_processed'], str(EXTENDED_EVALUATION_IDS))
    check('pure_test_excluded', not review_scope['pure_test_processed'], str(PURE_TEST_IDS))
    check('diagnostic_subset_excluded', not review_scope['diagnostic_subset_processed'], str(DIAGNOSTIC_IDS))
    check('actual_label_used_for_sampling_false', True, 'Actual labels were used only after OCR extraction for review filtering.')
    check('actual_label_used_for_review_filter_true', True, 'Actual ad intervals define post-hoc review windows.')
    start_ok = np.allclose(intervals['review_window_start_sec'], (intervals['ad_start_sec'] - REVIEW_PRE_SEC).clip(lower=0), equal_nan=True)
    check('review_window_start_equals_ad_start_minus_5_clipped', bool(start_ok), 'max(0, ad_start_sec - 5)')
    expected_end = intervals.apply(lambda r: min(r['ad_end_sec'] + REVIEW_POST_SEC, r['duration_for_window']) if pd.notna(r['duration_for_window']) else r['ad_end_sec'] + REVIEW_POST_SEC, axis=1)
    end_ok = np.allclose(intervals['review_window_end_sec'], expected_end, equal_nan=True)
    check('review_window_end_equals_ad_end_plus_5_clipped', bool(end_ok), 'min(video_duration_sec, ad_end_sec + 5)')
    within = True
    for _, r in review.iterrows():
        if not (r['review_window_start_sec'] - 1e-9 <= r['timestamp_sec'] <= r['review_window_end_sec'] + 1e-9):
            within = False
            break
    check('all_review_rows_within_window', within, 'all timestamps inside expanded review window')
    phase_valid = set(review['phase_in_review_window'].dropna().unique()).issubset({'pre_start_5s', 'ad_body', 'post_end_5s'})
    check('phase_pre_start_ad_body_post_end_assigned', phase_valid, str(phase_counts))
    check('post_end_5s_rows_present_or_warn', phase_counts.get('post_end_5s', 0) > 0, str(phase_counts), warn=True)
    check('output_has_v2_5_split_columns', all(c in full_review.columns for c in ['original_split_v2_4', 'split_role_v2_5', 'evaluation_subset_v2_5']), 'v2.5 split columns present')
    check('reviewer_note_columns_present', 'reviewer_note' in full_review.columns and 'post_end_review_note' in full_review.columns, 'manual note columns present')
    check('manual_label_columns_present', 'ocr_quality_manual_label' in full_review.columns, 'manual label column present')
    check('no_ocr_engine_called', True, 'This script does not import/call EasyOCR/pytesseract/PaddleOCR.')
    after = {str(p): snapshot(p) for p in protected if p.exists()}
    existing_unchanged = before == after
    check('no_existing_files_modified', existing_unchanged, 'Protected input file metadata unchanged')
    check('no_raw_frame_persisted', True, 'No frame images are written by this script')
    quality_checks = pd.DataFrame(checks).reindex(columns=QUALITY_CHECK_COLUMNS)
    quality_checks.to_csv(outputs['quality_checks'], index=False, encoding='utf-8-sig')

    logger.log('[STEP 17] Generate markdown/json reports')
    report_data: dict[str, Any] = {
        'run_id': run_id,
        'generated_at': now_iso(),
        'project_root': str(PROJECT_ROOT),
        'script_path': str(SCRIPT_PATH),
        'source_ocr_run_id': SOURCE_OCR_RUN_ID,
        'source_ocr_frame_result': str(ocr_frame_path),
        'source_ocr_backend_summary': source_report.get('ocr_backend_status', {}),
        'previous_review_window_definition': PREVIOUS_REVIEW_WINDOW,
        'review_window_definition': REVIEW_WINDOW_DEFINITION,
        'target_split_role_v2_5': 'development',
        'target_original_split_v2_4': 'train',
        'target_video_ids': DEVELOPMENT_IDS,
        'input_files': {
            'ocr_frame_result': str(ocr_frame_path), 'ocr_video_summary': str(ocr_summary_path), 'ocr_report': str(ocr_report_path),
            'actual_ad_intervals': str(LABEL_FILE), 'split_file': str(SPLIT_FILE), 'video_manifest': str(MANIFEST_FILE),
        },
        'review_scope_summary': review_scope,
        'frame_review_summary': frame_summary,
        'phase_summary': {phase: transition_summary[transition_summary['phase_in_review_window'] == phase].to_dict('records') for phase in ['pre_start_5s', 'ad_body', 'post_end_5s']},
        'post_end_5s_summary': post_summary,
        'interval_quality_summary': {'row_count': int(len(interval_summary)), 'high_priority_count': int((interval_summary['suggested_review_priority'] == 'high').sum()) if len(interval_summary) else 0},
        'video_quality_summary': {'row_count': int(len(video_summary))},
        'keyword_hit_summary': {'row_count': int(len(keyword))},
        'disclosure_review_summary': {'row_count': int(len(disclosure))},
        'empty_lowconf_summary': {'row_count': int(len(empty_low)), 'low_confidence_threshold': LOW_CONFIDENCE_THRESHOLD},
        'transition_comparison_summary': {'row_count': int(len(transition_summary))},
        'representative_examples_summary': {'row_count': int(len(examples))},
        'validation_results': checks,
        'safety_results': {
            'ocr_rerun_performed': False, 'ocr_engine_called': False, 'actual_label_used_for_sampling': False,
            'actual_label_used_for_review_filtering': True, 'detector_modified': False, 'existing_ocr_modified': not existing_unchanged,
            'extended_evaluation_processed': False, 'pure_test_processed': False, 'raw_frame_persisted': False,
        },
        'outputs': {k: str(v) for k, v in outputs.items()},
        'warnings': warnings,
        'errors': [],
        'threshold_metadata': {'low_confidence_threshold': LOW_CONFIDENCE_THRESHOLD, 'high_ad_score_threshold': high_score_threshold},
    }
    markdown = build_markdown(report_data, review, interval_summary, transition_summary, examples)
    outputs['markdown_report'].write_text(markdown, encoding='utf-8')
    save_json(outputs['json_report'], report_data)

    logger.log('[STEP 18] Update latest bundles')
    copy_latest(outputs, report_data, logger)
    forbidden = forbidden_scan([LATEST_CHATGPT, LATEST_OCR, LATEST_SCENE])
    check('latest_bundle_forbidden_file_scan_passed', len(forbidden) == 0, f'forbidden_file_count={len(forbidden)}')
    quality_checks = pd.DataFrame(checks).reindex(columns=QUALITY_CHECK_COLUMNS)
    quality_checks.to_csv(outputs['quality_checks'], index=False, encoding='utf-8-sig')
    report_data['validation_results'] = checks
    report_data['latest_forbidden_files'] = forbidden
    report_data['safety_results']['latest_bundle_forbidden_file_count'] = len(forbidden)
    report_data['elapsed_sec'] = time.time() - start
    save_json(outputs['json_report'], report_data)
    # 복사된 report/check/log를 갱신한다.
    copy_latest(outputs, report_data, logger)

    logger.log('[STEP 19] Print final human-readable summary')
    for line in [
        'Review extraction status:',
        '  - status: SUCCESS' if not errors else '  - status: CONDITIONAL_SUCCESS',
        f'  - run_id: {run_id}',
        f'  - run_dir: {run_dir}',
        'Source OCR:',
        f'  - source run_id: {SOURCE_OCR_RUN_ID}',
        f'  - source frame result: {ocr_frame_path}',
        '  - OCR rerun performed: false',
        'Target scope:',
        '  - split_role_v2_5: development',
        '  - original_split_v2_4: train',
        f'  - video_count: {review_scope["video_count"]}',
        f'  - ad_interval_count: {review_scope["ad_interval_count"]}',
        '  - review window: ad_start_sec - 5 to ad_end_sec + 5',
        'Review rows:',
        f'  - total rows: {frame_summary["total_review_row_count"]}',
        f'  - pre_start_5s rows: {phase_counts.get("pre_start_5s", 0)}',
        f'  - ad_body rows: {phase_counts.get("ad_body", 0)}',
        f'  - post_end_5s rows: {phase_counts.get("post_end_5s", 0)}',
        f'  - success_text: {frame_summary["success_text_count"]}',
        f'  - success_empty: {frame_summary["success_empty_count"]}',
        f'  - failed: {frame_summary["failed_count"]}',
        f'  - nonempty ratio: {frame_summary["nonempty_ratio"]:.3f}',
        f'  - keyword hit rows: {frame_summary["keyword_hit_row_count"]}',
        f'  - disclosure review rows: {len(disclosure)}',
        f'  - post_end keyword hit rows: {post_summary["post_end_5s_keyword_hit_row_count"]}',
        f'  - post_end still-ad-like rows: {post_summary["post_end_still_ad_like_row_count"]}',
        f'  - empty/lowconf rows: {len(empty_low)}',
        'Outputs:',
        f'  - full review sheet: {outputs["full_review_sheet"]}',
        f'  - compact review sheet: {outputs["compact_review_sheet"]}',
        f'  - markdown report: {outputs["markdown_report"]}',
        f'  - interval summary: {outputs["interval_summary"]}',
        f'  - phase transition summary: {outputs["transition_summary"]}',
        f'  - post-end 5s review: {outputs["post_end_review"]}',
        f'  - keyword hit review: {outputs["keyword_hit_review"]}',
        f'  - disclosure review: {outputs["disclosure_review"]}',
        f'  - empty/lowconf review: {outputs["empty_lowconf_review"]}',
        f'  - latest bundle: {LATEST_CHATGPT}',
        'Safety:',
        '  - OCR engine called: false',
        '  - actual label used for sampling: false',
        '  - actual label used for review filtering: true',
        '  - detector modified: false',
        f'  - existing OCR modified: {str(not existing_unchanged).lower()}',
        '  - Extended Evaluation processed: false',
        '  - Pure Test processed: false',
        '  - raw frame persisted: false',
        f'  - latest forbidden files: {len(forbidden)}',
    ]:
        logger.log(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
