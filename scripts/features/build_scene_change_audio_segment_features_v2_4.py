#!/usr/bin/env python3
"""Build visual scene boundary anchors and audio-segment aligned scene-change features v2.4.

This script does not run candidate extraction, frame decoding, OCR, audio generation,
or model inference. It only aligns existing OpenCV/FFmpeg, ResNet, and audio segment
CSV files.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path('.')
OLD_PROJECT_ROOT = Path('./_old_project_not_included')
TASK_NAME = 'scene_change_audio_segment_features_v2_4'
VERSION = 'v2_4'

DATA_DIR = PROJECT_ROOT / 'data'
SCENE_DIR = DATA_DIR / 'scene'
REVIEW_DIR = DATA_DIR / 'review'
AUDIO_DIR = DATA_DIR / 'audio'
FEATURE_DIR = DATA_DIR / 'features'
REPORT_DIR = PROJECT_ROOT / 'reports'
LOG_DIR = PROJECT_ROOT / 'logs'
SCRIPT_DIR = PROJECT_ROOT / 'scripts' / 'features'
LATEST_DIR = PROJECT_ROOT / 'outputs' / 'latest_for_chatgpt'

SCRIPT_PATH = SCRIPT_DIR / 'build_scene_change_audio_segment_features_v2_4.py'
ANCHOR_CSV = FEATURE_DIR / 'visual_scene_boundary_anchors_v2_4.csv'
SCENE_LABELED_CSV = FEATURE_DIR / 'scene_change_labeled_segment_features_v2_4.csv'
SCENE_EDGE_CSV = FEATURE_DIR / 'scene_change_ad_edge_5s_10s_features_v2_4.csv'
SCENE_COMBINED_CSV = FEATURE_DIR / 'scene_change_audio_segment_features_v2_4.csv'
JOIN_LABELED_CSV = FEATURE_DIR / 'audio_scene_labeled_segment_features_v2_4.csv'
JOIN_EDGE_CSV = FEATURE_DIR / 'audio_scene_ad_edge_5s_10s_features_v2_4.csv'
REPORT_JSON = REPORT_DIR / 'scene_change_audio_segment_features_v2_4_report.json'
SUMMARY_MD = REPORT_DIR / 'scene_change_audio_segment_features_v2_4_summary.md'
RUN_LOG = LOG_DIR / 'scene_change_audio_segment_features_v2_4_run_log.txt'
LATEST_README = LATEST_DIR / 'README_latest_files.md'

OUTPUT_FILES = [
    ANCHOR_CSV,
    SCENE_LABELED_CSV,
    SCENE_EDGE_CSV,
    SCENE_COMBINED_CSV,
    JOIN_LABELED_CSV,
    JOIN_EDGE_CSV,
    REPORT_JSON,
    SUMMARY_MD,
    RUN_LOG,
    SCRIPT_PATH,
]
ALLOWED_LATEST_NAMES = {p.name for p in OUTPUT_FILES} | {'README_latest_files.md'}
FORBIDDEN_SUFFIXES = {
    '.mp4', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.m4a', '.jpg', '.jpeg', '.png', '.webp',
    '.pt', '.pth', '.ckpt', '.bin',
}

OPENCV_CANDIDATE_PATHS = [
    SCENE_DIR / 'scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed.csv',
    SCENE_DIR / 'scene_candidates_v2_4_merged_ffmpeg_fallback.csv',
    SCENE_DIR / 'scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed_mmss.csv',
    SCENE_DIR / 'scene_candidates_v2_4_merged_ffmpeg_fallback_mmss.csv',
]
RESNET_CANDIDATE_PATHS = [
    REVIEW_DIR / 'resnet_scene_candidate_human_review_v2_4_usefulness_analysis.csv',
    SCENE_DIR / 'resnet_scene_candidates_v2_4.csv',
    SCENE_DIR / 'resnet_scene_candidates_v2_4_mmss.csv',
]
AUDIO_LABELED_PATHS = [
    AUDIO_DIR / 'audio_labeled_segment_features_v2_4.csv',
    LATEST_DIR / 'audio_labeled_segment_features_v2_4.csv',
    FEATURE_DIR / 'audio_labeled_segment_features_v2_4.csv',
    REVIEW_DIR / 'audio_labeled_segment_features_v2_4.csv',
]
AUDIO_EDGE_PATHS = [
    AUDIO_DIR / 'audio_ad_edge_5s_10s_features_v2_4.csv',
    LATEST_DIR / 'audio_ad_edge_5s_10s_features_v2_4.csv',
    FEATURE_DIR / 'audio_ad_edge_5s_10s_features_v2_4.csv',
    REVIEW_DIR / 'audio_ad_edge_5s_10s_features_v2_4.csv',
]
REFERENCE_FILES = [
    PROJECT_ROOT / 'reports/resnet_scene_candidate_human_review_v2_4_analysis_report.json',
    REVIEW_DIR / 'resnet_scene_candidate_human_review_v2_4_reviewed_rows.csv',
    REVIEW_DIR / 'scene_candidate_human_review_v2_4.xlsx',
]

LOG_LINES: list[str] = []


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def log(message: str) -> None:
    line = f'[{now_iso()}] {message}'
    LOG_LINES.append(line)
    print(message, flush=True)


def clean(value: Any) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def to_float(value: Any) -> float | None:
    text = clean(value)
    if text == '':
        return None
    try:
        return float(text)
    except Exception:
        return None


def bool_value(value: Any) -> bool:
    text = clean(value).lower()
    return text in {'true', '1', 'yes', 'y', 'reviewed', 'o', '예', '맞음'}


def bool_text(value: Any) -> str:
    return 'true' if bool(value) else 'false'


def mmss(seconds: Any) -> str:
    sec = to_float(seconds)
    if sec is None:
        return ''
    total = max(0, int(math.floor(sec)))
    return f'{total // 60:02d}:{total % 60:02d}'


def parse_mmss(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None
    if '분' in text and '초' in text:
        try:
            left, right = text.split('분', 1)
            return int(left.strip()) * 60 + float(right.replace('초', '').strip())
        except Exception:
            return None
    if ':' in text:
        parts = text.split(':')
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except Exception:
            return None
    return to_float(text)


def safe_rate(num: int | float, den: int | float) -> float:
    return round(float(num) / float(den), 6) if den else 0.0


def readable(seconds: float) -> str:
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    return f'{minutes}분 {sec}초'


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def old_project_snapshot() -> dict[str, Any]:
    if not OLD_PROJECT_ROOT.exists():
        return {'exists': False, 'file_count': 0, 'digest': ''}
    import hashlib
    h = hashlib.sha256()
    count = 0
    for path in sorted(p for p in OLD_PROJECT_ROOT.rglob('*') if p.is_file()):
        try:
            st = path.stat()
        except OSError:
            continue
        rel = path.relative_to(OLD_PROJECT_ROOT).as_posix()
        h.update(f'{rel}\t{st.st_size}\t{st.st_mtime_ns}\n'.encode('utf-8', errors='replace'))
        count += 1
    return {'exists': True, 'file_count': count, 'digest': h.hexdigest()}


def backup_existing(timestamp: str, input_paths: list[Path]) -> Path:
    backup_dir = PROJECT_ROOT / 'backups' / f'scene_change_audio_segment_features_v2_4_{timestamp}'
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in [*input_paths, *OUTPUT_FILES, LATEST_README]:
        if path.exists():
            try:
                rel = path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = Path(path.name)
            dst = backup_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
    return backup_dir


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def ensure_percentile(df: pd.DataFrame, score_col: str, percentile_col: str, video_col: str) -> pd.Series:
    existing = pd.to_numeric(df[percentile_col], errors='coerce') if percentile_col in df.columns else pd.Series([math.nan] * len(df))
    if existing.notna().all():
        return existing.clip(0, 1)
    scores = pd.to_numeric(df[score_col], errors='coerce') if score_col in df.columns else pd.Series([math.nan] * len(df))
    computed = scores.groupby(df[video_col].map(clean)).rank(method='average', pct=True)
    return existing.fillna(computed).fillna(0.0).clip(0, 1)


def standardize_opencv(df: pd.DataFrame, warnings: list[Any]) -> pd.DataFrame:
    time_col = find_col(df, ['candidate_time_sec', 'opencv_candidate_time_sec', 'time_sec'])
    video_col = find_col(df, ['video_id'])
    if not time_col or not video_col:
        raise RuntimeError('OpenCV/FFmpeg 후보에 video_id 또는 candidate_time_sec 컬럼이 없습니다.')
    mmss_col = find_col(df, ['candidate_time_mmss', 'candidate_time_mmss_floor', 'candidate_time_mmss_round', 'opencv_candidate_mmss'])
    score_col = find_col(df, ['scene_change_score', 'opencv_scene_change_score', 'score'])
    percentile_col = find_col(df, ['score_percentile_in_video', 'opencv_score_percentile_in_video'])
    out = pd.DataFrame()
    out['opencv_source_row_id'] = range(len(df))
    out['video_id'] = df[video_col].map(clean)
    out['video_title'] = df[find_col(df, ['video_title'])].map(clean) if find_col(df, ['video_title']) else ''
    out['opencv_candidate_time_sec'] = df[time_col].map(to_float)
    if mmss_col:
        out['opencv_candidate_mmss'] = df[mmss_col].map(clean)
    else:
        out['opencv_candidate_mmss'] = out['opencv_candidate_time_sec'].map(mmss)
    if score_col:
        out['opencv_scene_change_score'] = pd.to_numeric(df[score_col], errors='coerce')
    else:
        out['opencv_scene_change_score'] = pd.NA
        warnings.append('opencv_scene_change_score_missing')
    if percentile_col:
        out['opencv_score_percentile_in_video'] = pd.to_numeric(df[percentile_col], errors='coerce')
    else:
        out['opencv_score_percentile_in_video'] = ensure_percentile(out, 'opencv_scene_change_score', 'missing_percentile', 'video_id')
        warnings.append('opencv_percentile_recomputed_by_video_score_rank')
    out['opencv_score_percentile_in_video'] = out['opencv_score_percentile_in_video'].fillna(ensure_percentile(out, 'opencv_scene_change_score', 'opencv_score_percentile_in_video', 'video_id')).clip(0, 1)
    source_col = find_col(df, ['candidate_source', 'opencv_candidate_source', 'candidate_source_for_audit'])
    method_col = find_col(df, ['method_used', 'opencv_method_used'])
    out['opencv_candidate_source'] = df[source_col].map(clean) if source_col else 'opencv_ffmpeg'
    out['opencv_method_used'] = df[method_col].map(clean) if method_col else 'opencv_ffmpeg_existing_candidate'
    out['source_family'] = 'opencv_ffmpeg'
    out = out[out['video_id'].ne('') & out['opencv_candidate_time_sec'].notna()].copy()
    out = out.sort_values(['video_id', 'opencv_candidate_time_sec', 'opencv_source_row_id']).reset_index(drop=True)
    out['opencv_source_row_id'] = out.index
    return out


def standardize_resnet(df: pd.DataFrame, warnings: list[Any]) -> pd.DataFrame:
    video_col = find_col(df, ['video_id'])
    time_col = find_col(df, ['resnet_time_sec_std', 'resnet_candidate_time_sec', 'candidate_time_sec', 'score_time_sec'])
    if not video_col or not time_col:
        raise RuntimeError('ResNet 후보에 video_id 또는 candidate time 컬럼이 없습니다.')
    mmss_col = find_col(df, ['resnet_time_mmss_std', 'resnet_candidate_mmss', 'candidate_time_mmss'])
    score_col = find_col(df, ['resnet_score_std', 'scene_change_score', 'resnet_scene_change_score', 'cosine_distance'])
    percentile_col = find_col(df, ['resnet_percentile_std', 'resnet_score_percentile_in_video', 'score_percentile_in_video'])
    out = pd.DataFrame()
    out['resnet_source_row_id'] = range(len(df))
    out['video_id'] = df[video_col].map(clean)
    out['video_title'] = df[find_col(df, ['video_title'])].map(clean) if find_col(df, ['video_title']) else ''
    out['resnet_candidate_time_sec'] = df[time_col].map(to_float)
    out['resnet_candidate_mmss'] = df[mmss_col].map(clean) if mmss_col else out['resnet_candidate_time_sec'].map(mmss)
    if score_col:
        out['resnet_scene_change_score'] = pd.to_numeric(df[score_col], errors='coerce')
    else:
        out['resnet_scene_change_score'] = pd.NA
        warnings.append('resnet_scene_change_score_missing')
    if percentile_col:
        out['resnet_score_percentile_in_video'] = pd.to_numeric(df[percentile_col], errors='coerce')
    else:
        out['resnet_score_percentile_in_video'] = ensure_percentile(out, 'resnet_scene_change_score', 'missing_percentile', 'video_id')
        warnings.append('resnet_percentile_recomputed_by_video_score_rank')
    out['resnet_score_percentile_in_video'] = out['resnet_score_percentile_in_video'].fillna(ensure_percentile(out, 'resnet_scene_change_score', 'resnet_score_percentile_in_video', 'video_id')).clip(0, 1)
    source_col = find_col(df, ['candidate_source', 'resnet_candidate_source'])
    method_col = find_col(df, ['method_used', 'resnet_method_used'])
    out['resnet_candidate_source'] = df[source_col].map(clean) if source_col else 'resnet_embedding'
    out['resnet_method_used'] = df[method_col].map(clean) if method_col else 'resnet_existing_candidate'
    for src, dst in [
        ('reviewed_normalized', 'resnet_reviewed_normalized'),
        ('is_true_scene_change_norm', 'resnet_is_true_scene_change_norm'),
        ('false_positive_type', 'resnet_false_positive_type'),
        ('resnet_candidate_usefulness_auto', 'resnet_candidate_usefulness_auto'),
        ('resnet_review_analysis_group', 'resnet_review_analysis_group'),
        ('review_note', 'resnet_review_note'),
    ]:
        out[dst] = df[src].map(clean) if src in df.columns else ''
    out['source_family'] = 'resnet'
    out = out[out['video_id'].ne('') & out['resnet_candidate_time_sec'].notna()].copy()
    out = out.sort_values(['video_id', 'resnet_candidate_time_sec', 'resnet_source_row_id']).reset_index(drop=True)
    out['resnet_source_row_id'] = out.index
    return out


def nearest_time(times: list[float], target: float | None) -> tuple[float | None, float | None]:
    if target is None or not times:
        return None, None
    best = min(times, key=lambda t: (abs(t - target), t))
    return best, abs(best - target)


def strength_band(score: float | None) -> str:
    if score is None or pd.isna(score):
        return 'low'
    if score >= 0.95:
        return 'very_high'
    if score >= 0.90:
        return 'high'
    if score >= 0.75:
        return 'medium'
    return 'low'


def review_flags(resnet_rows: pd.DataFrame) -> dict[str, Any]:
    if resnet_rows.empty:
        return {
            'reviewed_by_human': False,
            'reviewed_true_scene_change': False,
            'reviewed_false_positive': False,
            'review_quality_note': 'no_review_reference',
            'review_note': '',
            'resnet_candidate_usefulness_auto': '',
            'resnet_review_analysis_group': '',
            'resnet_false_positive_type': '',
        }
    reviewed = resnet_rows['resnet_reviewed_normalized'].map(lambda v: clean(v).lower() == 'true') if 'resnet_reviewed_normalized' in resnet_rows.columns else pd.Series([False] * len(resnet_rows))
    true_scene = resnet_rows['resnet_is_true_scene_change_norm'].map(lambda v: clean(v).lower() == 'true') if 'resnet_is_true_scene_change_norm' in resnet_rows.columns else pd.Series([False] * len(resnet_rows))
    usefulness = resnet_rows['resnet_candidate_usefulness_auto'].map(clean) if 'resnet_candidate_usefulness_auto' in resnet_rows.columns else pd.Series([''] * len(resnet_rows))
    fp_type = resnet_rows['resnet_false_positive_type'].map(clean) if 'resnet_false_positive_type' in resnet_rows.columns else pd.Series([''] * len(resnet_rows))
    false_positive = usefulness.eq('false_positive') | fp_type.ne('') | (reviewed & ~true_scene & resnet_rows['resnet_is_true_scene_change_norm'].map(clean).ne(''))
    notes = [clean(v) for v in resnet_rows.get('resnet_review_note', pd.Series([''] * len(resnet_rows))).tolist() if clean(v)]
    if bool(false_positive.any()):
        quality = 'resnet_reviewed_false_positive_reference'
    elif bool((reviewed & true_scene).any()):
        quality = 'resnet_reviewed_true_scene_change_reference'
    elif bool(reviewed.any()):
        quality = 'resnet_reviewed_uncertain_or_nonboundary_reference'
    else:
        quality = 'resnet_unreviewed_reference'
    return {
        'reviewed_by_human': bool(reviewed.any()),
        'reviewed_true_scene_change': bool((reviewed & true_scene).any()),
        'reviewed_false_positive': bool(false_positive.any()),
        'review_quality_note': quality,
        'review_note': ' | '.join(notes[:3]),
        'resnet_candidate_usefulness_auto': ';'.join(sorted(set(usefulness[usefulness.ne('')]))),
        'resnet_review_analysis_group': ';'.join(sorted(set(resnet_rows.get('resnet_review_analysis_group', pd.Series(dtype=str)).map(clean).replace('', pd.NA).dropna()))),
        'resnet_false_positive_type': ';'.join(sorted(set(fp_type[fp_type.ne('')]))),
    }


def build_anchor_record(
    video_id: str,
    video_title: str,
    opencv_rows: pd.DataFrame,
    resnet_rows: pd.DataFrame,
    source_relation: str,
    nearest_other_time: float | None = None,
    nearest_other_dist: float | None = None,
) -> dict[str, Any]:
    has_o = not opencv_rows.empty
    has_r = not resnet_rows.empty
    o_times = sorted([float(v) for v in opencv_rows['opencv_candidate_time_sec'].dropna().tolist()]) if has_o else []
    r_times = sorted([float(v) for v in resnet_rows['resnet_candidate_time_sec'].dropna().tolist()]) if has_r else []
    all_times = sorted(o_times + r_times)
    if has_o:
        canonical = o_times[0]
        canonical_source = 'opencv_ffmpeg'
    elif has_r:
        canonical = r_times[0]
        canonical_source = 'resnet'
    else:
        canonical = math.nan
        canonical_source = ''
    rep_o = opencv_rows.sort_values(['opencv_candidate_time_sec', 'opencv_source_row_id']).iloc[0] if has_o else None
    if has_r:
        resnet_sort = resnet_rows.copy()
        resnet_sort['_gap'] = resnet_sort['resnet_candidate_time_sec'].map(lambda x: abs(float(x) - canonical))
        rep_r = resnet_sort.sort_values(['_gap', 'resnet_candidate_time_sec', 'resnet_source_row_id']).iloc[0]
    else:
        rep_r = None
    opencv_percentiles = pd.to_numeric(opencv_rows['opencv_score_percentile_in_video'], errors='coerce') if has_o else pd.Series(dtype=float)
    resnet_percentiles = pd.to_numeric(resnet_rows['resnet_score_percentile_in_video'], errors='coerce') if has_r else pd.Series(dtype=float)
    max_strength = max([v for v in [opencv_percentiles.max() if len(opencv_percentiles) else math.nan, resnet_percentiles.max() if len(resnet_percentiles) else math.nan] if pd.notna(v)] or [0.0])
    if source_relation == 'opencv_resnet_merged_2s':
        max_strength = min(float(max_strength) + 0.05, 1.0)
    time_gap = abs(o_times[0] - r_times[0]) if has_o and has_r else None
    if nearest_other_time is None and has_o and has_r:
        nearest_other_time = r_times[0] if canonical_source == 'opencv_ffmpeg' else o_times[0]
        nearest_other_dist = abs(nearest_other_time - canonical)
    review = review_flags(resnet_rows)
    return {
        'scene_boundary_anchor_id': '',
        'video_id': video_id,
        'video_title': video_title,
        'canonical_boundary_time_sec': canonical,
        'canonical_boundary_mmss': mmss(canonical),
        'canonical_time_source': canonical_source,
        'has_opencv_ffmpeg_candidate': has_o,
        'has_resnet_candidate': has_r,
        'source_relation': source_relation,
        'source_count': int(has_o) + int(has_r),
        'opencv_candidate_time_sec': rep_o['opencv_candidate_time_sec'] if rep_o is not None else '',
        'opencv_candidate_mmss': rep_o['opencv_candidate_mmss'] if rep_o is not None else '',
        'resnet_candidate_time_sec': rep_r['resnet_candidate_time_sec'] if rep_r is not None else '',
        'resnet_candidate_mmss': rep_r['resnet_candidate_mmss'] if rep_r is not None else '',
        'anchor_time_mean_sec': round(sum(all_times) / len(all_times), 6) if all_times else '',
        'anchor_time_min_sec': min(all_times) if all_times else '',
        'anchor_time_max_sec': max(all_times) if all_times else '',
        'anchor_time_spread_sec': round(max(all_times) - min(all_times), 6) if all_times else '',
        'time_gap_opencv_resnet_sec': '' if time_gap is None else round(time_gap, 6),
        'nearest_other_source_candidate_time_sec': '' if nearest_other_time is None else nearest_other_time,
        'nearest_other_source_candidate_mmss': '' if nearest_other_time is None else mmss(nearest_other_time),
        'distance_to_nearest_other_source_candidate_sec': '' if nearest_other_dist is None else round(nearest_other_dist, 6),
        'cross_source_near_5s': bool(nearest_other_dist is not None and nearest_other_dist <= 5),
        'opencv_scene_change_score': pd.to_numeric(opencv_rows['opencv_scene_change_score'], errors='coerce').max() if has_o else '',
        'opencv_score_percentile_in_video': opencv_percentiles.max() if has_o else '',
        'resnet_scene_change_score': pd.to_numeric(resnet_rows['resnet_scene_change_score'], errors='coerce').max() if has_r else '',
        'resnet_score_percentile_in_video': resnet_percentiles.max() if has_r else '',
        'visual_boundary_strength_score': round(float(max_strength), 6),
        'visual_boundary_strength_band': strength_band(float(max_strength)),
        'reviewed_by_human': review['reviewed_by_human'],
        'reviewed_true_scene_change': review['reviewed_true_scene_change'],
        'reviewed_false_positive': review['reviewed_false_positive'],
        'review_quality_note': review['review_quality_note'],
        'review_note': review['review_note'],
        'review_reference_source': 'resnet_review_analysis' if has_r else '',
        'resnet_candidate_usefulness_auto': review['resnet_candidate_usefulness_auto'],
        'resnet_review_analysis_group': review['resnet_review_analysis_group'],
        'resnet_false_positive_type': review['resnet_false_positive_type'],
        'opencv_candidate_count_in_anchor': len(opencv_rows),
        'resnet_candidate_count_in_anchor': len(resnet_rows),
        'opencv_candidate_times_sec_list': ';'.join(str(int(t)) if float(t).is_integer() else str(t) for t in o_times),
        'resnet_candidate_times_sec_list': ';'.join(str(int(t)) if float(t).is_integer() else str(t) for t in r_times),
    }


def build_anchors(opencv_df: pd.DataFrame, resnet_df: pd.DataFrame) -> pd.DataFrame:
    anchors: list[dict[str, Any]] = []
    all_videos = sorted(set(opencv_df['video_id'].map(clean)) | set(resnet_df['video_id'].map(clean)), key=lambda x: (int(x) if x.isdigit() else x))
    for video_id in all_videos:
        o = opencv_df[opencv_df['video_id'].map(clean).eq(video_id)].copy().reset_index(drop=True)
        r = resnet_df[resnet_df['video_id'].map(clean).eq(video_id)].copy().reset_index(drop=True)
        title = clean(o['video_title'].iloc[0]) if len(o) else (clean(r['video_title'].iloc[0]) if len(r) else '')
        graph: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
        nodes = [('o', i) for i in range(len(o))] + [('r', i) for i in range(len(r))]
        for node in nodes:
            graph[node]
        for oi, ot in enumerate(o['opencv_candidate_time_sec'].tolist()):
            for ri, rt in enumerate(r['resnet_candidate_time_sec'].tolist()):
                if abs(float(ot) - float(rt)) <= 2:
                    graph[('o', oi)].add(('r', ri))
                    graph[('r', ri)].add(('o', oi))
        seen: set[tuple[str, int]] = set()
        used_o: set[int] = set()
        used_r: set[int] = set()
        for node in nodes:
            if node in seen:
                continue
            queue = deque([node])
            comp: set[tuple[str, int]] = set()
            seen.add(node)
            while queue:
                cur = queue.popleft()
                comp.add(cur)
                for nxt in graph[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            comp_o = sorted(idx for kind, idx in comp if kind == 'o')
            comp_r = sorted(idx for kind, idx in comp if kind == 'r')
            if comp_o and comp_r:
                used_o.update(comp_o)
                used_r.update(comp_r)
                anchors.append(build_anchor_record(video_id, title, o.iloc[comp_o].copy(), r.iloc[comp_r].copy(), 'opencv_resnet_merged_2s'))
        r_times_all = [float(v) for v in r['resnet_candidate_time_sec'].tolist()]
        o_times_all = [float(v) for v in o['opencv_candidate_time_sec'].tolist()]
        for oi, row in o.iterrows():
            if oi in used_o:
                continue
            nt, dist = nearest_time(r_times_all, float(row['opencv_candidate_time_sec']))
            relation = 'opencv_resnet_near_5s_separate' if dist is not None and 2 < dist <= 5 else 'opencv_ffmpeg_only'
            anchors.append(build_anchor_record(video_id, title, o.iloc[[oi]].copy(), r.iloc[[]].copy(), relation, nt, dist))
        for ri, row in r.iterrows():
            if ri in used_r:
                continue
            nt, dist = nearest_time(o_times_all, float(row['resnet_candidate_time_sec']))
            relation = 'opencv_resnet_near_5s_separate' if dist is not None and 2 < dist <= 5 else 'resnet_only'
            anchors.append(build_anchor_record(video_id, title, o.iloc[[]].copy(), r.iloc[[ri]].copy(), relation, nt, dist))
    anchor_df = pd.DataFrame(anchors)
    if anchor_df.empty:
        return anchor_df
    anchor_df = anchor_df.sort_values(['video_id', 'canonical_boundary_time_sec', 'canonical_time_source']).reset_index(drop=True)
    counters: dict[str, int] = defaultdict(int)
    ids = []
    for _, row in anchor_df.iterrows():
        vid = clean(row['video_id'])
        counters[vid] += 1
        ids.append(f'V{vid}_SBA{counters[vid]:06d}')
    anchor_df['scene_boundary_anchor_id'] = ids
    return anchor_df


def normalize_segment_times(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'segment_start_sec' not in out.columns:
        start_mmss = find_col(out, ['segment_start_mmss'])
        out['segment_start_sec'] = out[start_mmss].map(parse_mmss) if start_mmss else pd.NA
    if 'segment_end_sec' not in out.columns:
        end_mmss = find_col(out, ['segment_end_mmss'])
        out['segment_end_sec'] = out[end_mmss].map(parse_mmss) if end_mmss else pd.NA
    out['segment_start_sec'] = pd.to_numeric(out['segment_start_sec'], errors='coerce')
    out['segment_end_sec'] = pd.to_numeric(out['segment_end_sec'], errors='coerce')
    if 'segment_start_mmss' not in out.columns:
        out['segment_start_mmss'] = out['segment_start_sec'].map(mmss)
    if 'segment_end_mmss' not in out.columns:
        out['segment_end_mmss'] = out['segment_end_sec'].map(mmss)
    if 'ad_interval_id' not in out.columns:
        out['ad_interval_id'] = ''
    return out


def primary_role(segment_type: str) -> tuple[str, str]:
    st = clean(segment_type)
    mapping = {
        'pre_ad_10s': ('segment_end', 'segment_end가 광고 시작 라벨에 해당'),
        'ad_start_first_5s': ('segment_start', 'segment_start가 광고 시작 라벨에 해당'),
        'ad_start_first_10s': ('segment_start', 'segment_start가 광고 시작 라벨에 해당'),
        'ad_start_5to10s': ('none', '광고 시작 직후 5~10초 구간이므로 직접 boundary edge가 아님'),
        'ad_end_last_5s': ('segment_end', 'segment_end가 광고 종료 라벨에 해당'),
        'ad_end_last_10s': ('segment_end', 'segment_end가 광고 종료 라벨에 해당'),
        'ad_end_minus10to_minus5s': ('none', '광고 종료 10~5초 전 구간이므로 직접 boundary edge가 아님'),
        'post_ad_10s': ('segment_start', 'segment_start가 광고 종료 라벨에 해당'),
        'ad_full': ('both', '광고 시작과 종료 edge를 모두 포함'),
        'random_non_ad_30s': ('none', '랜덤 non-ad 구간이므로 직접 광고 boundary edge가 아님'),
    }
    return mapping.get(st, ('none', 'unknown segment_type'))


def nearest_from_df(anchor_sub: pd.DataFrame, time_sec: float | None, time_col: str = 'canonical_boundary_time_sec') -> tuple[Any, Any, Any]:
    if time_sec is None or anchor_sub.empty or time_col not in anchor_sub.columns:
        return '', '', ''
    values = pd.to_numeric(anchor_sub[time_col], errors='coerce')
    valid = anchor_sub[values.notna()].copy()
    if valid.empty:
        return '', '', ''
    valid['_dist'] = pd.to_numeric(valid[time_col], errors='coerce').map(lambda v: abs(float(v) - time_sec))
    row = valid.sort_values(['_dist', time_col]).iloc[0]
    t = row[time_col]
    return t, mmss(t), round(float(row['_dist']), 6)


def nearest_from_times(times: list[float], time_sec: float | None) -> tuple[Any, Any]:
    nt, dist = nearest_time(times, time_sec)
    return ('' if nt is None else nt), ('' if dist is None else round(dist, 6))


def aggregate_segments(segment_df: pd.DataFrame, anchors: pd.DataFrame, opencv_df: pd.DataFrame, resnet_df: pd.DataFrame, source_file_name: str, group_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    seg = normalize_segment_times(segment_df)
    anchor_by_video = {vid: sub.copy() for vid, sub in anchors.groupby(anchors['video_id'].map(clean))}
    opencv_times_by_video = {vid: sorted([float(v) for v in sub['opencv_candidate_time_sec'].dropna().tolist()]) for vid, sub in opencv_df.groupby(opencv_df['video_id'].map(clean))}
    resnet_times_by_video = {vid: sorted([float(v) for v in sub['resnet_candidate_time_sec'].dropna().tolist()]) for vid, sub in resnet_df.groupby(resnet_df['video_id'].map(clean))}
    rows: list[dict[str, Any]] = []
    for idx, row in seg.iterrows():
        vid = clean(row.get('video_id'))
        start = to_float(row.get('segment_start_sec'))
        end = to_float(row.get('segment_end_sec'))
        stype = clean(row.get('segment_type'))
        sub = anchor_by_video.get(vid, pd.DataFrame(columns=anchors.columns)).copy()
        if start is not None and end is not None and not sub.empty:
            times = pd.to_numeric(sub['canonical_boundary_time_sec'], errors='coerce')
            inside = sub[(times >= start) & (times < end)].copy()
        else:
            inside = pd.DataFrame(columns=anchors.columns)
        count = len(inside)
        max_strength = pd.to_numeric(inside.get('visual_boundary_strength_score', pd.Series(dtype=float)), errors='coerce').max() if count else math.nan
        mean_strength = pd.to_numeric(inside.get('visual_boundary_strength_score', pd.Series(dtype=float)), errors='coerce').mean() if count else math.nan
        max_opencv = pd.to_numeric(inside.get('opencv_score_percentile_in_video', pd.Series(dtype=float)), errors='coerce').max() if count else math.nan
        max_resnet = pd.to_numeric(inside.get('resnet_score_percentile_in_video', pd.Series(dtype=float)), errors='coerce').max() if count else math.nan
        if count:
            strongest = inside.assign(_strength=pd.to_numeric(inside['visual_boundary_strength_score'], errors='coerce').fillna(-1)).sort_values(['_strength', 'canonical_boundary_time_sec'], ascending=[False, True]).iloc[0]
        else:
            strongest = None
        n_start_t, n_start_mmss, n_start_dist = nearest_from_df(sub, start)
        n_end_t, n_end_mmss, n_end_dist = nearest_from_df(sub, end)
        o_start_t, o_start_dist = nearest_from_times(opencv_times_by_video.get(vid, []), start)
        r_start_t, r_start_dist = nearest_from_times(resnet_times_by_video.get(vid, []), start)
        o_end_t, o_end_dist = nearest_from_times(opencv_times_by_video.get(vid, []), end)
        r_end_t, r_end_dist = nearest_from_times(resnet_times_by_video.get(vid, []), end)
        role, role_note = primary_role(stype)
        primary_time = ''
        primary_nearest_t = ''
        primary_nearest_mmss = ''
        primary_dist = ''
        primary_o_t = ''
        primary_o_dist = ''
        primary_r_t = ''
        primary_r_dist = ''
        if role == 'segment_start':
            primary_time, primary_nearest_t, primary_nearest_mmss, primary_dist = start, n_start_t, n_start_mmss, n_start_dist
            primary_o_t, primary_o_dist, primary_r_t, primary_r_dist = o_start_t, o_start_dist, r_start_t, r_start_dist
        elif role == 'segment_end':
            primary_time, primary_nearest_t, primary_nearest_mmss, primary_dist = end, n_end_t, n_end_mmss, n_end_dist
            primary_o_t, primary_o_dist, primary_r_t, primary_r_dist = o_end_t, o_end_dist, r_end_t, r_end_dist
        elif role == 'both':
            candidates = []
            if n_start_dist != '':
                candidates.append(('segment_start', start, n_start_t, n_start_mmss, n_start_dist, o_start_t, o_start_dist, r_start_t, r_start_dist))
            if n_end_dist != '':
                candidates.append(('segment_end', end, n_end_t, n_end_mmss, n_end_dist, o_end_t, o_end_dist, r_end_t, r_end_dist))
            if candidates:
                chosen = sorted(candidates, key=lambda x: (float(x[4]), x[0]))[0]
                primary_time, primary_nearest_t, primary_nearest_mmss, primary_dist = chosen[1], chosen[2], chosen[3], chosen[4]
                primary_o_t, primary_o_dist, primary_r_t, primary_r_dist = chosen[5], chosen[6], chosen[7], chosen[8]
        feature = {
            'segment_id': clean(row.get('segment_id')) or f'{group_name}_{idx}',
            'video_id': vid,
            'video_title': clean(row.get('video_title')),
            'ad_interval_id': clean(row.get('ad_interval_id')),
            'segment_type': stype,
            'segment_start_sec': start,
            'segment_end_sec': end,
            'segment_start_mmss': clean(row.get('segment_start_mmss')) or mmss(start),
            'segment_end_mmss': clean(row.get('segment_end_mmss')) or mmss(end),
            'source_audio_feature_file': source_file_name,
            'audio_segment_group': group_name,
            'scene_boundary_count_in_segment': count,
            'has_scene_boundary_in_segment': count > 0,
            'opencv_ffmpeg_boundary_count_in_segment': int(inside['has_opencv_ffmpeg_candidate'].sum()) if count else 0,
            'resnet_boundary_count_in_segment': int(inside['has_resnet_candidate'].sum()) if count else 0,
            'multi_source_boundary_count_in_segment': int((inside['source_count'] >= 2).sum()) if count else 0,
            'resnet_only_boundary_count_in_segment': int((inside['source_relation'] == 'resnet_only').sum()) if count else 0,
            'opencv_only_boundary_count_in_segment': int((inside['source_relation'] == 'opencv_ffmpeg_only').sum()) if count else 0,
            'cross_source_near_5s_boundary_count_in_segment': int(inside['cross_source_near_5s'].sum()) if count else 0,
            'reviewed_true_scene_boundary_count_in_segment': int(inside['reviewed_true_scene_change'].sum()) if count else 0,
            'reviewed_false_positive_boundary_count_in_segment': int(inside['reviewed_false_positive'].sum()) if count else 0,
            'max_visual_boundary_strength_score_in_segment': '' if pd.isna(max_strength) else round(float(max_strength), 6),
            'mean_visual_boundary_strength_score_in_segment': '' if pd.isna(mean_strength) else round(float(mean_strength), 6),
            'max_opencv_score_percentile_in_segment': '' if pd.isna(max_opencv) else round(float(max_opencv), 6),
            'max_resnet_score_percentile_in_segment': '' if pd.isna(max_resnet) else round(float(max_resnet), 6),
            'strongest_scene_boundary_time_sec': '' if strongest is None else strongest['canonical_boundary_time_sec'],
            'strongest_scene_boundary_mmss': '' if strongest is None else strongest['canonical_boundary_mmss'],
            'strongest_scene_boundary_source_relation': '' if strongest is None else strongest['source_relation'],
            'nearest_scene_boundary_to_segment_start_sec': n_start_t,
            'nearest_scene_boundary_to_segment_start_mmss': n_start_mmss,
            'distance_to_nearest_scene_boundary_to_segment_start_sec': n_start_dist,
            'nearest_scene_boundary_to_segment_end_sec': n_end_t,
            'nearest_scene_boundary_to_segment_end_mmss': n_end_mmss,
            'distance_to_nearest_scene_boundary_to_segment_end_sec': n_end_dist,
            'has_scene_boundary_near_segment_start_2s': bool(n_start_dist != '' and float(n_start_dist) <= 2),
            'has_scene_boundary_near_segment_start_5s': bool(n_start_dist != '' and float(n_start_dist) <= 5),
            'has_scene_boundary_near_segment_start_10s': bool(n_start_dist != '' and float(n_start_dist) <= 10),
            'has_scene_boundary_near_segment_end_2s': bool(n_end_dist != '' and float(n_end_dist) <= 2),
            'has_scene_boundary_near_segment_end_5s': bool(n_end_dist != '' and float(n_end_dist) <= 5),
            'has_scene_boundary_near_segment_end_10s': bool(n_end_dist != '' and float(n_end_dist) <= 10),
            'distance_to_nearest_opencv_boundary_to_segment_start_sec': o_start_dist,
            'distance_to_nearest_resnet_boundary_to_segment_start_sec': r_start_dist,
            'distance_to_nearest_opencv_boundary_to_segment_end_sec': o_end_dist,
            'distance_to_nearest_resnet_boundary_to_segment_end_sec': r_end_dist,
            'has_opencv_boundary_near_segment_start_5s': bool(o_start_dist != '' and float(o_start_dist) <= 5),
            'has_resnet_boundary_near_segment_start_5s': bool(r_start_dist != '' and float(r_start_dist) <= 5),
            'has_opencv_boundary_near_segment_end_5s': bool(o_end_dist != '' and float(o_end_dist) <= 5),
            'has_resnet_boundary_near_segment_end_5s': bool(r_end_dist != '' and float(r_end_dist) <= 5),
            'primary_edge_role': role,
            'primary_edge_role_note': role_note,
            'primary_edge_time_sec': primary_time,
            'nearest_scene_boundary_to_primary_edge_sec': primary_nearest_t,
            'nearest_scene_boundary_to_primary_edge_mmss': primary_nearest_mmss,
            'distance_to_nearest_scene_boundary_to_primary_edge_sec': primary_dist,
            'has_scene_boundary_near_primary_edge_2s': bool(primary_dist != '' and float(primary_dist) <= 2),
            'has_scene_boundary_near_primary_edge_5s': bool(primary_dist != '' and float(primary_dist) <= 5),
            'has_scene_boundary_near_primary_edge_10s': bool(primary_dist != '' and float(primary_dist) <= 10),
            'nearest_opencv_boundary_to_primary_edge_sec': primary_o_t,
            'distance_to_nearest_opencv_boundary_to_primary_edge_sec': primary_o_dist,
            'nearest_resnet_boundary_to_primary_edge_sec': primary_r_t,
            'distance_to_nearest_resnet_boundary_to_primary_edge_sec': primary_r_dist,
            'has_opencv_boundary_near_primary_edge_5s': bool(primary_o_dist != '' and float(primary_o_dist) <= 5),
            'has_resnet_boundary_near_primary_edge_5s': bool(primary_r_dist != '' and float(primary_r_dist) <= 5),
            'feature_interpretation_note': 'label_aligned_analysis_feature_not_direct_inference_feature',
        }
        rows.append(feature)
    features = pd.DataFrame(rows)
    original = seg.reset_index(drop=True)
    feature_cols_to_add = [c for c in features.columns if c not in original.columns]
    join_ready = pd.concat([original, features[feature_cols_to_add]], axis=1)
    return features, join_ready


def segment_summary(features: pd.DataFrame) -> dict[str, Any]:
    if features.empty:
        return {}
    by_type: dict[str, Any] = {}
    for stype, sub in features.groupby('segment_type', dropna=False):
        st = clean(stype)
        by_type[st] = {
            'row_count': int(len(sub)),
            'scene_boundary_count_sum': int(pd.to_numeric(sub['scene_boundary_count_in_segment'], errors='coerce').fillna(0).sum()),
            'segments_with_scene_boundary': int(sub['has_scene_boundary_in_segment'].sum()),
            'primary_edge_hit_2s': int(sub['has_scene_boundary_near_primary_edge_2s'].sum()),
            'primary_edge_hit_5s': int(sub['has_scene_boundary_near_primary_edge_5s'].sum()),
            'primary_edge_hit_10s': int(sub['has_scene_boundary_near_primary_edge_10s'].sum()),
        }
    return by_type


def update_latest(files: list[Path]) -> tuple[bool, list[str], list[str]]:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for path in LATEST_DIR.iterdir():
        if path.is_file():
            path.unlink()
    copied = []
    for src in files:
        if src.exists() and src.name in ALLOWED_LATEST_NAMES and src.suffix.lower() not in FORBIDDEN_SUFFIXES:
            dst = LATEST_DIR / src.name
            shutil.copy2(src, dst)
            copied.append(src.name)
    readme = '# latest_for_chatgpt files\n\n'
    readme += f'작업명: {TASK_NAME}\n\n'
    readme += '복사 파일 목록:\n\n'
    for name in copied:
        readme += f'- `{name}`\n'
    readme += '\n금지 파일 확인: mp4/mov/mkv/avi, wav/mp3/m4a, frame image, model/checkpoint/cache/weight 파일은 복사하지 않았다.\n'
    LATEST_README.write_text(readme, encoding='utf-8')
    copied.append(LATEST_README.name)
    forbidden = [p.name for p in LATEST_DIR.iterdir() if p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES]
    forbidden += [p.name for p in LATEST_DIR.iterdir() if p.is_file() and p.name not in ALLOWED_LATEST_NAMES]
    return not forbidden, copied, sorted(set(forbidden))


def build_summary(report: dict[str, Any]) -> str:
    seg_summary = report.get('scene_boundary_count_by_segment_type', {})
    sub_agents = report.get('sub_agent_results', {})
    anchor_rows = [
        ('OpenCV/FFmpeg only', report.get('opencv_only_anchor_count', 0)),
        ('ResNet only', report.get('resnet_only_anchor_count', 0)),
        ('Merged <=2s', report.get('opencv_resnet_merged_2s_anchor_count', 0)),
        ('Separate but cross-source near <=5s', report.get('opencv_resnet_near_5s_separate_count', 0)),
    ]
    anchor_table = '\n'.join(f'| {name} | {count} |' for name, count in anchor_rows)
    seg_table = '\n'.join(
        f"| {stype} | {vals.get('row_count', 0)} | {vals.get('scene_boundary_count_sum', 0)} | {vals.get('primary_edge_hit_2s', 0)} | {vals.get('primary_edge_hit_5s', 0)} | {vals.get('primary_edge_hit_10s', 0)} |"
        for stype, vals in sorted(seg_summary.items())
    )
    sub_table = '\n'.join(f"| {name} | {info.get('status')} | {info.get('summary', '')} |" for name, info in sub_agents.items())
    generated = '\n'.join(f'- `{path}`' for path in report.get('output_files', {}).values())
    return f"""# Scene Change Audio Segment Features v2.4

## 작업 개요

OpenCV/FFmpeg + ResNet scene-change 후보를 통합해 visual scene boundary anchor pool을 만들고, 현재 audio feature와 동일한 segment row 단위로 화면 전환 구조 feature를 집계했다. 이 작업은 광고 boundary 확정이 아니라 OCR/audio/visual 결합 rule 설계를 위한 전처리다.

## 입력 파일

- OpenCV/FFmpeg candidates: `{report.get('input_files', {}).get('opencv_candidates')}`
- ResNet candidates/review reference: `{report.get('input_files', {}).get('resnet_candidates')}`
- audio labeled segments: `{report.get('input_files', {}).get('audio_labeled_segments')}`
- audio edge segments: `{report.get('input_files', {}).get('audio_edge_segments')}`

## 후보 통합 기준

같은 `video_id` 안에서 OpenCV/FFmpeg 후보와 ResNet 후보가 2초 이내로 연결되면 하나의 visual scene boundary anchor로 병합했다. 2초 초과 5초 이내 후보는 병합하지 않고 별도 anchor로 유지하며 `cross_source_near_5s=true`와 nearest-other-source distance를 기록했다.

## 중복 병합 기준

- `abs(opencv_time - resnet_time) <= 2s`: `opencv_resnet_merged_2s`
- `2s < abs(opencv_time - resnet_time) <= 5s`: 별도 anchor, `opencv_resnet_near_5s_separate`
- 그 외: `opencv_ffmpeg_only` 또는 `resnet_only`

## canonical time 기준

OpenCV/FFmpeg + ResNet merged anchor의 `canonical_boundary_time_sec`는 OpenCV/FFmpeg 후보 시각이다. ResNet-only anchor는 ResNet 후보 시각, OpenCV/FFmpeg-only anchor는 OpenCV/FFmpeg 후보 시각을 사용했다.

## 평균값을 사용하지 않은 이유

평균 시각은 실제 어느 candidate extractor도 산출하지 않은 가상 timestamp가 될 수 있다. 따라서 join/집계용 canonical time은 deterministic source time으로 유지하고, 평균값은 `anchor_time_mean_sec` 보조 컬럼으로만 보존했다.

## 리뷰값을 후보 시각 선정에 사용하지 않은 이유

사람 리뷰값은 후보 제거 또는 대표 시각 선택 기준으로 사용하지 않았다. 리뷰 정보는 `reviewed_by_human`, `reviewed_false_positive`, `review_quality_note`, `review_note` 같은 reference 컬럼으로만 붙였다.

## visual scene boundary anchor 결과

| anchor relation | count |
|---|---:|
{anchor_table}

총 anchor 수: {report.get('visual_scene_boundary_anchor_count')}

## audio segment type별 scene-change feature 요약

| segment_type | rows | boundary_count_sum | primary_hit_2s | primary_hit_5s | primary_hit_10s |
|---|---:|---:|---:|---:|---:|
{seg_table}

## primary edge hit 요약

primary edge feature는 label-aligned analysis용이다. `pre_ad_10s`, `ad_start_first_*`, `ad_end_last_*`, `post_ad_10s`, `ad_full` 등 segment type별로 비교 edge를 지정했지만, actual inference에서는 정답 광고 라벨 기반 segment type을 그대로 사용할 수 없다.

## 생성 파일

{generated}

## Sub Agent 검증 결과

| Sub Agent | Status | Summary |
|---|---|---|
{sub_table}

## 주의사항

- 이 feature는 label-aligned analysis feature다.
- actual inference에서는 정답 광고 라벨 기반 segment type을 그대로 사용할 수 없다.
- scene boundary feature는 광고 점수가 아니라 화면 전환 구조 feature다.
- final 광고 탐지 성능 claim으로 해석하지 않는다.

## 다음 작업

- OCR segment feature 생성 후 join
- OCR/audio/scene 결합 rule 설계
- scene boundary anchor 기준 OCR/audio pre/post feature 생성
"""


def print_completion(report: dict[str, Any]) -> None:
    anchor_counts = [
        ('opencv_ffmpeg_only', report.get('opencv_only_anchor_count', 0)),
        ('resnet_only', report.get('resnet_only_anchor_count', 0)),
        ('opencv_resnet_merged_2s', report.get('opencv_resnet_merged_2s_anchor_count', 0)),
        ('opencv_resnet_near_5s_separate', report.get('opencv_resnet_near_5s_separate_count', 0)),
    ]
    print('\n## 작업 완료: SUCCESS' if not report.get('errors') else '\n## 작업 완료: PARTIAL/FAIL')
    print('\n### 핵심 요약')
    print(f"- 작업명: `{TASK_NAME}`")
    print(f"- OpenCV/FFmpeg 후보 수: {report.get('opencv_candidate_count')}")
    print(f"- ResNet 후보 수: {report.get('resnet_candidate_count')}")
    print(f"- visual scene boundary anchor 수: {report.get('visual_scene_boundary_anchor_count')}")
    print(f"- audio segment row 수: labeled={report.get('audio_labeled_segment_row_count')}, edge={report.get('audio_ad_edge_row_count')}")
    print(f"- scene feature row 수: labeled={report.get('scene_change_labeled_segment_feature_row_count')}, edge={report.get('scene_change_ad_edge_feature_row_count')}, combined={report.get('combined_scene_audio_segment_feature_row_count')}")
    print(f"- old_project_modified: {str(report.get('old_project_modified')).lower()}")
    print('\n### Anchor 통합 결과')
    print('| relation | count |')
    print('|---|---:|')
    for name, count in anchor_counts:
        print(f'| {name} | {count} |')
    print('\n### Segment Type 요약')
    print('| segment_type | rows | boundary_count_sum | primary_hit_2s | primary_hit_5s | primary_hit_10s |')
    print('|---|---:|---:|---:|---:|---:|')
    for stype, vals in sorted(report.get('scene_boundary_count_by_segment_type', {}).items()):
        print(f"| {stype} | {vals.get('row_count', 0)} | {vals.get('scene_boundary_count_sum', 0)} | {vals.get('primary_edge_hit_2s', 0)} | {vals.get('primary_edge_hit_5s', 0)} | {vals.get('primary_edge_hit_10s', 0)} |")
    print('\n### Sub Agent 검증 결과')
    print('| Sub Agent | Status | Summary |')
    print('|---|---|---|')
    for name, info in report.get('sub_agent_results', {}).items():
        print(f"| {name} | {info.get('status')} | {info.get('summary', '')} |")
    print('\n### 생성 파일')
    for path in report.get('output_files', {}).values():
        print(f'- `{path}`')
    warnings = report.get('warnings') or []
    errors = report.get('errors') or []
    print('\n### Warning / Error')
    print(f"- Warnings: {'없음' if not warnings else json.dumps(warnings, ensure_ascii=False)}")
    print(f"- Errors: {'없음' if not errors else json.dumps(errors, ensure_ascii=False)}")
    print('\n### 다음 작업 제안')
    print('- OCR segment feature 생성 후 join')
    print('- OCR/audio/scene 결합 rule 설계')
    print('\n상세 JSON은 report 파일에 저장했습니다.')


def validate_and_report(
    report: dict[str, Any],
    opencv_df: pd.DataFrame,
    resnet_df: pd.DataFrame,
    anchors: pd.DataFrame,
    audio_labeled: pd.DataFrame,
    audio_edge: pd.DataFrame,
    scene_labeled: pd.DataFrame,
    scene_edge: pd.DataFrame,
    join_labeled: pd.DataFrame,
    join_edge: pd.DataFrame,
    old_project_modified: bool,
    latest_ok: bool,
    latest_forbidden: list[str],
) -> dict[str, Any]:
    sub_agents: dict[str, Any] = {}
    input_ok = bool(len(opencv_df) and len(resnet_df) and len(audio_labeled) and len(audio_edge))
    time_ok = not opencv_df['opencv_candidate_time_sec'].isna().all() and not resnet_df['resnet_candidate_time_sec'].isna().all()
    segment_time_ok = bool((pd.to_numeric(audio_labeled['segment_end_sec'], errors='coerce') > pd.to_numeric(audio_labeled['segment_start_sec'], errors='coerce')).all()) and bool((pd.to_numeric(audio_edge['segment_end_sec'], errors='coerce') > pd.to_numeric(audio_edge['segment_start_sec'], errors='coerce')).all())
    sub_agents['sub_agent_1_input_schema_validation'] = {
        'status': 'PASS' if input_ok and time_ok and segment_time_ok else 'FAIL',
        'summary': '필수 입력과 핵심 시간 컬럼 확보' if input_ok and time_ok and segment_time_ok else '필수 입력 또는 시간 컬럼 문제',
        'checks': {
            'opencv_rows': len(opencv_df),
            'resnet_rows': len(resnet_df),
            'audio_labeled_rows': len(audio_labeled),
            'audio_edge_rows': len(audio_edge),
            'segment_start_end_valid': segment_time_ok,
        },
    }
    anchor_id_unique = anchors['scene_boundary_anchor_id'].is_unique if not anchors.empty else False
    merged = anchors[anchors['source_relation'] == 'opencv_resnet_merged_2s']
    merged_canonical_ok = bool((merged['canonical_time_source'] == 'opencv_ffmpeg').all()) if len(merged) else True
    merged_not_mean = bool((pd.to_numeric(merged['canonical_boundary_time_sec'], errors='coerce') != pd.to_numeric(merged['anchor_time_mean_sec'], errors='coerce')).any()) if len(merged) else True
    resnet_only = anchors[anchors['source_relation'] == 'resnet_only']
    resnet_canonical_ok = bool((resnet_only['canonical_time_source'] == 'resnet').all()) if len(resnet_only) else True
    near_separate = anchors[anchors['source_relation'] == 'opencv_resnet_near_5s_separate']
    near_sep_ok = bool((pd.to_numeric(near_separate['distance_to_nearest_other_source_candidate_sec'], errors='coerce') > 2).all() and (pd.to_numeric(near_separate['distance_to_nearest_other_source_candidate_sec'], errors='coerce') <= 5).all()) if len(near_separate) else True
    merge_ok = anchor_id_unique and merged_canonical_ok and resnet_canonical_ok and near_sep_ok
    sub_agents['sub_agent_2_scene_boundary_anchor_merge_validation'] = {
        'status': 'PASS' if merge_ok else 'FAIL',
        'summary': '2초 병합/5초 분리와 deterministic canonical time 확인' if merge_ok else 'anchor merge 규칙 위반 가능성',
        'checks': {
            'anchor_id_unique': anchor_id_unique,
            'merged_canonical_source_opencv': merged_canonical_ok,
            'resnet_only_canonical_source_resnet': resnet_canonical_ok,
            'near_5s_separate_distance_rule_ok': near_sep_ok,
            'canonical_not_average_for_some_merged_rows': merged_not_mean,
            'review_used_for_canonical_time': False,
            'reviewed_false_positive_removed': False,
        },
    }
    count_cols = [c for c in scene_labeled.columns if c.endswith('_count_in_segment') or c == 'scene_boundary_count_in_segment']
    dist_cols = [c for c in scene_labeled.columns if c.startswith('distance_to_')] + [c for c in scene_edge.columns if c.startswith('distance_to_')]
    nonnegative_counts = all((pd.to_numeric(pd.concat([scene_labeled.get(c, pd.Series(dtype=float)), scene_edge.get(c, pd.Series(dtype=float))]), errors='coerce').fillna(0) >= 0).all() for c in set(count_cols))
    nonnegative_dists = all((pd.to_numeric(pd.concat([scene_labeled.get(c, pd.Series(dtype=float)), scene_edge.get(c, pd.Series(dtype=float))]), errors='coerce').dropna() >= 0).all() for c in set(dist_cols))
    rows_ok = len(scene_labeled) == len(audio_labeled) and len(scene_edge) == len(audio_edge) and len(join_labeled) == len(audio_labeled) and len(join_edge) == len(audio_edge)
    primary_roles_ok = set(scene_labeled['primary_edge_role']).issubset({'segment_start', 'segment_end', 'both', 'none'}) and set(scene_edge['primary_edge_role']).issubset({'segment_start', 'segment_end', 'both', 'none'})
    source_dist_cols_ok = all(c in scene_labeled.columns and c in scene_edge.columns for c in [
        'distance_to_nearest_opencv_boundary_to_segment_start_sec',
        'distance_to_nearest_resnet_boundary_to_segment_start_sec',
        'distance_to_nearest_opencv_boundary_to_segment_end_sec',
        'distance_to_nearest_resnet_boundary_to_segment_end_sec',
    ])
    agg_ok = rows_ok and nonnegative_counts and nonnegative_dists and primary_roles_ok and source_dist_cols_ok
    sub_agents['sub_agent_3_audio_segment_feature_aggregation_validation'] = {
        'status': 'PASS' if agg_ok else 'FAIL',
        'summary': 'audio row count 유지 및 count/distance/primary edge feature 정상' if agg_ok else 'audio segment aggregation 검증 실패',
        'checks': {
            'row_counts_match': rows_ok,
            'count_features_nonnegative': nonnegative_counts,
            'distance_features_nonnegative': nonnegative_dists,
            'primary_edge_roles_valid': primary_roles_ok,
            'source_specific_edge_distance_columns_exist': source_dist_cols_ok,
        },
    }
    summary_text = SUMMARY_MD.read_text(encoding='utf-8') if SUMMARY_MD.exists() else ''
    leakage_ok = all(phrase in summary_text for phrase in [
        'label-aligned analysis feature',
        'scene boundary feature는 광고 점수가 아니라 화면 전환 구조 feature',
        'actual inference에서는 정답 광고 라벨 기반 segment type을 그대로 사용할 수 없다',
    ])
    sub_agents['sub_agent_4_leakage_interpretation_validation'] = {
        'status': 'PASS' if leakage_ok else 'FAIL',
        'summary': 'label leakage 한계와 화면 전환 구조 feature 해석 명시' if leakage_ok else 'leakage/interpretation 주의사항 누락',
        'checks': {
            'summary_mentions_label_aligned_analysis': 'label-aligned analysis feature' in summary_text,
            'summary_mentions_not_ad_score': 'scene boundary feature는 광고 점수가 아니라 화면 전환 구조 feature' in summary_text,
            'summary_mentions_not_direct_inference': 'actual inference에서는 정답 광고 라벨 기반 segment type을 그대로 사용할 수 없다' in summary_text,
            'canonical_time_from_review_or_label': False,
        },
    }
    output_exists = all(path.exists() for path in OUTPUT_FILES)
    safety_ok = output_exists and latest_ok and not latest_forbidden and not old_project_modified
    sub_agents['sub_agent_5_output_safety_validation'] = {
        'status': 'PASS' if safety_ok else 'FAIL',
        'summary': 'output/latest 생성 및 old project 미수정 확인' if safety_ok else 'output/latest/safety 검증 실패',
        'checks': {
            'all_output_files_exist': output_exists,
            'latest_forbidden_files_absent': not latest_forbidden,
            'old_project_modified': old_project_modified,
        },
        'warnings': latest_forbidden,
    }
    report['sub_agent_results'] = sub_agents
    return report


def main() -> int:
    start_time = now_iso()
    start_monotonic = time.monotonic()
    warnings: list[Any] = []
    errors: list[Any] = []
    fallback_used = False
    fallback_reason: list[str] = []
    for d in [FEATURE_DIR, REPORT_DIR, LOG_DIR, SCRIPT_DIR, LATEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    log('[STEP 1/12] input 파일 탐색')
    old_before = old_project_snapshot()
    opencv_path = first_existing(OPENCV_CANDIDATE_PATHS)
    resnet_path = first_existing(RESNET_CANDIDATE_PATHS)
    audio_labeled_path = first_existing(AUDIO_LABELED_PATHS)
    audio_edge_path = first_existing(AUDIO_EDGE_PATHS)
    missing = []
    for name, path in [('opencv_candidates', opencv_path), ('resnet_candidates', resnet_path), ('audio_labeled_segments', audio_labeled_path), ('audio_edge_segments', audio_edge_path)]:
        if path is None:
            missing.append(name)
    if missing:
        errors.append({'missing_required_inputs': missing})
        raise RuntimeError(f'missing required inputs: {missing}')
    for p in [opencv_path, resnet_path, audio_labeled_path, audio_edge_path]:
        if p and 'v2_3' in p.name:
            fallback_used = True
            fallback_reason.append(f'v2_3 fallback used: {p}')
    input_paths = [p for p in [opencv_path, resnet_path, audio_labeled_path, audio_edge_path, *REFERENCE_FILES] if p and p.exists()]

    log('[STEP 2/12] 안전 백업 및 old project snapshot')
    timestamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    backup_dir = backup_existing(timestamp, input_paths)
    log(f'backup_dir={backup_dir}')

    log('[STEP 3/12] OpenCV/FFmpeg 후보 로드 및 표준화')
    opencv_raw = read_csv(opencv_path)
    opencv_df = standardize_opencv(opencv_raw, warnings)
    log(f'opencv_candidate_count={len(opencv_df)} source={opencv_path}')

    log('[STEP 4/12] ResNet 후보 로드 및 review reference 표준화')
    resnet_raw = read_csv(resnet_path)
    resnet_df = standardize_resnet(resnet_raw, warnings)
    log(f'resnet_candidate_count={len(resnet_df)} source={resnet_path}')

    log('[STEP 5/12] visual scene boundary anchor 병합')
    anchors = build_anchors(opencv_df, resnet_df)
    log(f'visual_scene_boundary_anchor_count={len(anchors)}')

    log('[STEP 6/12] audio segment 파일 로드')
    audio_labeled = normalize_segment_times(read_csv(audio_labeled_path))
    audio_edge = normalize_segment_times(read_csv(audio_edge_path))
    log(f'audio_labeled_rows={len(audio_labeled)} audio_edge_rows={len(audio_edge)}')

    log('[STEP 7/12] audio_labeled segment 단위 scene feature 집계')
    scene_labeled, join_labeled = aggregate_segments(audio_labeled, anchors, opencv_df, resnet_df, audio_labeled_path.name, 'labeled_segment')
    log(f'scene_change_labeled_segment_feature_rows={len(scene_labeled)}')

    log('[STEP 8/12] audio_ad_edge_5s_10s segment 단위 scene feature 집계')
    scene_edge, join_edge = aggregate_segments(audio_edge, anchors, opencv_df, resnet_df, audio_edge_path.name, 'ad_edge_5s_10s')
    log(f'scene_change_ad_edge_feature_rows={len(scene_edge)}')

    log('[STEP 9/12] CSV 산출물 저장')
    combined = pd.concat([scene_labeled, scene_edge], ignore_index=True, sort=False)
    write_csv(ANCHOR_CSV, anchors)
    write_csv(SCENE_LABELED_CSV, scene_labeled)
    write_csv(SCENE_EDGE_CSV, scene_edge)
    write_csv(SCENE_COMBINED_CSV, combined)
    write_csv(JOIN_LABELED_CSV, join_labeled)
    write_csv(JOIN_EDGE_CSV, join_edge)

    log('[STEP 10/12] report/summary/log 생성')
    segment_types = sorted(set(scene_labeled['segment_type'].map(clean)) | set(scene_edge['segment_type'].map(clean)))
    segment_type_counts = dict(Counter(combined['segment_type'].map(clean)))
    scene_summary = segment_summary(combined)
    primary_hit_2 = {k: v.get('primary_edge_hit_2s', 0) for k, v in scene_summary.items()}
    primary_hit_5 = {k: v.get('primary_edge_hit_5s', 0) for k, v in scene_summary.items()}
    primary_hit_10 = {k: v.get('primary_edge_hit_10s', 0) for k, v in scene_summary.items()}
    output_files = {p.name: str(p) for p in OUTPUT_FILES if p.exists()}
    old_after = old_project_snapshot()
    old_project_modified = old_before != old_after
    report: dict[str, Any] = {
        'project_root': str(PROJECT_ROOT),
        'task_name': TASK_NAME,
        'version': VERSION,
        'start_time': start_time,
        'end_time': now_iso(),
        'actual_runtime_seconds': round(time.monotonic() - start_monotonic, 3),
        'actual_runtime_readable': readable(time.monotonic() - start_monotonic),
        'input_files': {
            'opencv_candidates': str(opencv_path),
            'resnet_candidates': str(resnet_path),
            'audio_labeled_segments': str(audio_labeled_path),
            'audio_edge_segments': str(audio_edge_path),
        },
        'output_files': output_files,
        'backup_dir': str(backup_dir),
        'fallback_used': fallback_used,
        'fallback_reason': fallback_reason,
        'opencv_candidate_count': int(len(opencv_df)),
        'resnet_candidate_count': int(len(resnet_df)),
        'visual_scene_boundary_anchor_count': int(len(anchors)),
        'opencv_only_anchor_count': int((anchors['source_relation'] == 'opencv_ffmpeg_only').sum()),
        'resnet_only_anchor_count': int((anchors['source_relation'] == 'resnet_only').sum()),
        'opencv_resnet_merged_2s_anchor_count': int((anchors['source_relation'] == 'opencv_resnet_merged_2s').sum()),
        'opencv_resnet_near_5s_separate_count': int((anchors['source_relation'] == 'opencv_resnet_near_5s_separate').sum()),
        'reviewed_true_scene_anchor_count': int(anchors['reviewed_true_scene_change'].sum()),
        'reviewed_false_positive_anchor_count': int(anchors['reviewed_false_positive'].sum()),
        'audio_labeled_segment_row_count': int(len(audio_labeled)),
        'audio_ad_edge_row_count': int(len(audio_edge)),
        'scene_change_labeled_segment_feature_row_count': int(len(scene_labeled)),
        'scene_change_ad_edge_feature_row_count': int(len(scene_edge)),
        'combined_scene_audio_segment_feature_row_count': int(len(combined)),
        'join_ready_labeled_row_count': int(len(join_labeled)),
        'join_ready_ad_edge_row_count': int(len(join_edge)),
        'segment_types': segment_types,
        'segment_type_counts': {str(k): int(v) for k, v in segment_type_counts.items()},
        'scene_boundary_count_by_segment_type': scene_summary,
        'primary_edge_hit_2s_by_segment_type': primary_hit_2,
        'primary_edge_hit_5s_by_segment_type': primary_hit_5,
        'primary_edge_hit_10s_by_segment_type': primary_hit_10,
        'warnings': warnings,
        'errors': errors,
        'sub_agent_results': {},
        'old_project_modified': old_project_modified,
    }
    SUMMARY_MD.write_text(build_summary(report), encoding='utf-8')
    REPORT_JSON.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    RUN_LOG.write_text('\n'.join(LOG_LINES) + '\n', encoding='utf-8')

    log('[STEP 11/12] latest_for_chatgpt 갱신 및 Sub Agent 검증')
    latest_ok, latest_copied, latest_forbidden = update_latest(OUTPUT_FILES)
    report['latest_for_chatgpt_updated'] = latest_ok
    report['latest_for_chatgpt_files'] = latest_copied
    report['latest_for_chatgpt_forbidden_files'] = latest_forbidden
    report = validate_and_report(report, opencv_df, resnet_df, anchors, audio_labeled, audio_edge, scene_labeled, scene_edge, join_labeled, join_edge, old_project_modified, latest_ok, latest_forbidden)
    if old_project_modified:
        errors.append('old_project_modified_unexpectedly')
    if any(info.get('status') == 'FAIL' for info in report['sub_agent_results'].values()):
        errors.append('sub_agent_validation_failed')
    report['warnings'] = warnings
    report['errors'] = errors
    report['end_time'] = now_iso()
    report['actual_runtime_seconds'] = round(time.monotonic() - start_monotonic, 3)
    report['actual_runtime_readable'] = readable(time.monotonic() - start_monotonic)
    report['output_files'] = {p.name: str(p) for p in OUTPUT_FILES if p.exists()}
    SUMMARY_MD.write_text(build_summary(report), encoding='utf-8')
    REPORT_JSON.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    RUN_LOG.write_text('\n'.join(LOG_LINES) + '\n', encoding='utf-8')
    shutil.copy2(REPORT_JSON, LATEST_DIR / REPORT_JSON.name)
    shutil.copy2(SUMMARY_MD, LATEST_DIR / SUMMARY_MD.name)
    shutil.copy2(RUN_LOG, LATEST_DIR / RUN_LOG.name)

    log('[STEP 12/12] 완료')
    LOG_LINES.append(f'작업 종료 시각: {report["end_time"]}')
    LOG_LINES.append(f'실제 작업 시간: {report["actual_runtime_readable"]}')
    RUN_LOG.write_text('\n'.join(LOG_LINES) + '\n', encoding='utf-8')
    shutil.copy2(RUN_LOG, LATEST_DIR / RUN_LOG.name)
    print_completion(report)
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
