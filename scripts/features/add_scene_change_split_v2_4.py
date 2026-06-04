#!/usr/bin/env python3
"""Attach fixed v2.4 video_id splits to scene-change/audio-scene feature tables.

This script only adds split metadata and train-only filtered copies. It does not
modify rules, thresholds, scene candidate extraction, anchor generation, audio, OCR,
or model outputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path('.')
OLD_PROJECT_ROOT = Path('./_old_project_not_included')
TASK_NAME = 'scene_change_split_v2_4'
VERSION = 'v2_4'

FEATURE_DIR = PROJECT_ROOT / 'data' / 'features'
LATEST_DIR = PROJECT_ROOT / 'outputs' / 'latest_for_chatgpt'
REPORT_DIR = PROJECT_ROOT / 'reports'
LOG_DIR = PROJECT_ROOT / 'logs'
SCRIPT_DIR = PROJECT_ROOT / 'scripts' / 'features'

SCRIPT_PATH = SCRIPT_DIR / 'add_scene_change_split_v2_4.py'
VIDEO_SPLIT_CSV = FEATURE_DIR / 'video_split_v2_4.csv'
FILE_SUMMARY_CSV = FEATURE_DIR / 'scene_change_split_file_summary_v2_4.csv'
ANCHOR_SUMMARY_CSV = FEATURE_DIR / 'scene_change_anchor_split_summary_v2_4.csv'
SEGMENT_SUMMARY_CSV = FEATURE_DIR / 'scene_change_segment_split_summary_v2_4.csv'
TRAIN_RULE_SUMMARY_CSV = FEATURE_DIR / 'scene_change_train_only_rule_analysis_summary_v2_4.csv'
REPORT_JSON = REPORT_DIR / 'scene_change_split_v2_4_report.json'
SUMMARY_MD = REPORT_DIR / 'scene_change_split_v2_4_summary.md'
RUN_LOG = LOG_DIR / 'scene_change_split_v2_4_run_log.txt'
LATEST_README = LATEST_DIR / 'README_latest_files.md'

TRAIN_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
VALIDATION_IDS = [3, 7, 18]
TEST_IDS = [4, 16, 17]
SPLIT_SOURCE = 'fixed_video_id_split_v2_4'
SPLIT_NOTE = 'train/validation/test split assigned by video_id for leakage-safe rule analysis'

FILE_SPECS = {
    'visual_scene_boundary_anchors': {
        'required': True,
        'data_path': FEATURE_DIR / 'visual_scene_boundary_anchors_v2_4.csv',
        'latest_path': LATEST_DIR / 'visual_scene_boundary_anchors_v2_4.csv',
        'with_split': FEATURE_DIR / 'visual_scene_boundary_anchors_v2_4_with_split.csv',
        'train_only': FEATURE_DIR / 'visual_scene_boundary_anchors_v2_4_train_only.csv',
        'kind': 'anchor',
    },
    'scene_change_labeled_segment_features': {
        'required': False,
        'data_path': FEATURE_DIR / 'scene_change_labeled_segment_features_v2_4.csv',
        'latest_path': LATEST_DIR / 'scene_change_labeled_segment_features_v2_4.csv',
        'with_split': FEATURE_DIR / 'scene_change_labeled_segment_features_v2_4_with_split.csv',
        'train_only': FEATURE_DIR / 'scene_change_labeled_segment_features_v2_4_train_only.csv',
        'kind': 'segment',
    },
    'scene_change_ad_edge_5s_10s_features': {
        'required': False,
        'data_path': FEATURE_DIR / 'scene_change_ad_edge_5s_10s_features_v2_4.csv',
        'latest_path': LATEST_DIR / 'scene_change_ad_edge_5s_10s_features_v2_4.csv',
        'with_split': FEATURE_DIR / 'scene_change_ad_edge_5s_10s_features_v2_4_with_split.csv',
        'train_only': FEATURE_DIR / 'scene_change_ad_edge_5s_10s_features_v2_4_train_only.csv',
        'kind': 'segment',
    },
    'scene_change_audio_segment_features': {
        'required': False,
        'data_path': FEATURE_DIR / 'scene_change_audio_segment_features_v2_4.csv',
        'latest_path': LATEST_DIR / 'scene_change_audio_segment_features_v2_4.csv',
        'with_split': FEATURE_DIR / 'scene_change_audio_segment_features_v2_4_with_split.csv',
        'train_only': FEATURE_DIR / 'scene_change_audio_segment_features_v2_4_train_only.csv',
        'kind': 'segment',
    },
    'audio_scene_labeled_segment_features': {
        'required': False,
        'data_path': FEATURE_DIR / 'audio_scene_labeled_segment_features_v2_4.csv',
        'latest_path': LATEST_DIR / 'audio_scene_labeled_segment_features_v2_4.csv',
        'with_split': FEATURE_DIR / 'audio_scene_labeled_segment_features_v2_4_with_split.csv',
        'train_only': FEATURE_DIR / 'audio_scene_labeled_segment_features_v2_4_train_only.csv',
        'kind': 'segment',
    },
    'audio_scene_ad_edge_5s_10s_features': {
        'required': False,
        'data_path': FEATURE_DIR / 'audio_scene_ad_edge_5s_10s_features_v2_4.csv',
        'latest_path': LATEST_DIR / 'audio_scene_ad_edge_5s_10s_features_v2_4.csv',
        'with_split': FEATURE_DIR / 'audio_scene_ad_edge_5s_10s_features_v2_4_with_split.csv',
        'train_only': FEATURE_DIR / 'audio_scene_ad_edge_5s_10s_features_v2_4_train_only.csv',
        'kind': 'segment',
    },
}

SUMMARY_OUTPUTS = [VIDEO_SPLIT_CSV, FILE_SUMMARY_CSV, ANCHOR_SUMMARY_CSV, SEGMENT_SUMMARY_CSV, TRAIN_RULE_SUMMARY_CSV]
REPORT_OUTPUTS = [REPORT_JSON, SUMMARY_MD, RUN_LOG, SCRIPT_PATH]
OUTPUT_FILES = SUMMARY_OUTPUTS + [spec['with_split'] for spec in FILE_SPECS.values()] + [spec['train_only'] for spec in FILE_SPECS.values()] + REPORT_OUTPUTS
ALLOWED_LATEST_NAMES = {p.name for p in OUTPUT_FILES} | {'README_latest_files.md'}
FORBIDDEN_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.m4a', '.jpg', '.jpeg', '.png', '.webp', '.pt', '.pth', '.ckpt', '.bin'}

LOG_LINES: list[str] = []


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def log(message: str) -> None:
    LOG_LINES.append(f'[{now_iso()}] {message}')
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


def normalize_video_id(value: Any) -> int | None:
    text = clean(value)
    if text == '':
        return None
    try:
        return int(float(text))
    except Exception:
        digits = ''.join(ch for ch in text if ch.isdigit())
        if digits:
            return int(digits)
    return None


def readable(seconds: float) -> str:
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    return f'{minutes}분 {sec}초'


def file_hash(path: Path) -> str:
    if not path.exists():
        return ''
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def old_project_snapshot() -> dict[str, Any]:
    if not OLD_PROJECT_ROOT.exists():
        return {'exists': False, 'file_count': 0, 'digest': ''}
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


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig')


def build_split_mapping() -> tuple[dict[int, str], pd.DataFrame]:
    rows = []
    mapping: dict[int, str] = {}
    for split, ids in [('train', TRAIN_IDS), ('validation', VALIDATION_IDS), ('test', TEST_IDS)]:
        for vid in ids:
            mapping[vid] = split
            rows.append({'video_id': vid, 'split': split, 'split_source': SPLIT_SOURCE, 'split_note': SPLIT_NOTE})
    return mapping, pd.DataFrame(rows)


def discover_inputs(warnings: list[Any], errors: list[Any]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for key, spec in FILE_SPECS.items():
        path = spec['data_path'] if spec['data_path'].exists() else (spec['latest_path'] if spec['latest_path'].exists() else None)
        if path is None:
            if spec['required']:
                errors.append(f'missing_required_input:{key}')
            else:
                warnings.append(f'skipped_missing_recommended_input:{key}')
            continue
        if path == spec['latest_path']:
            warnings.append(f'latest_for_chatgpt_fallback_used:{key}')
        found[key] = path
    return found


def backup_existing(timestamp: str, input_paths: list[Path]) -> Path:
    backup_dir = PROJECT_ROOT / 'backups' / f'scene_change_split_v2_4_{timestamp}'
    backup_dir.mkdir(parents=True, exist_ok=True)
    candidates = [*input_paths, *OUTPUT_FILES, LATEST_README]
    for path in candidates:
        if path.exists():
            try:
                rel = path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = Path(path.name)
            dst = backup_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
    return backup_dir


def apply_split(df: pd.DataFrame, split_map: dict[int, str]) -> tuple[pd.DataFrame, list[int], list[str]]:
    warnings: list[str] = []
    if 'video_id' not in df.columns:
        raise RuntimeError('video_id column missing')
    out = df.copy()
    for col in ['split', 'is_train', 'is_validation', 'is_test', 'split_source', 'split_note']:
        if col in out.columns:
            existing_col = f'original_{col}_existing'
            suffix = 1
            while existing_col in out.columns:
                existing_col = f'original_{col}_existing_{suffix}'
                suffix += 1
            out[existing_col] = out[col]
            out = out.drop(columns=[col])
            warnings.append(f'preserved_existing_column:{col}->{existing_col}')
    normalized = out['video_id'].map(normalize_video_id)
    out['video_id_normalized_for_split'] = normalized
    out['split'] = normalized.map(split_map).fillna('unknown')
    out['is_train'] = out['split'].eq('train')
    out['is_validation'] = out['split'].eq('validation')
    out['is_test'] = out['split'].eq('test')
    out['split_source'] = SPLIT_SOURCE
    out['split_note'] = SPLIT_NOTE
    unknown_ids = sorted(set(int(v) for v in normalized[out['split'].eq('unknown')].dropna().tolist()))
    if out['split'].eq('unknown').any():
        warnings.append(f'unknown_video_id_rows:{int(out["split"].eq("unknown").sum())}')
    return out, unknown_ids, warnings


def split_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = Counter(df['split'].map(clean)) if 'split' in df.columns else Counter()
    return {key: int(counts.get(key, 0)) for key in ['train', 'validation', 'test', 'unknown']}


def summarize_anchor(anchor_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if anchor_df.empty:
        return pd.DataFrame(rows)
    for split, sub in anchor_df.groupby('split', dropna=False):
        strength = pd.to_numeric(sub.get('visual_boundary_strength_score', pd.Series(dtype=float)), errors='coerce')
        band = sub.get('visual_boundary_strength_band', pd.Series([''] * len(sub))).map(clean)
        relation = sub.get('source_relation', pd.Series([''] * len(sub))).map(clean)
        rows.append({
            'split': clean(split),
            'anchor_count': int(len(sub)),
            'opencv_ffmpeg_only_count': int(relation.eq('opencv_ffmpeg_only').sum()),
            'resnet_only_count': int(relation.eq('resnet_only').sum()),
            'opencv_resnet_merged_2s_count': int(relation.eq('opencv_resnet_merged_2s').sum()),
            'opencv_resnet_near_5s_separate_count': int(relation.eq('opencv_resnet_near_5s_separate').sum()),
            'reviewed_true_scene_anchor_count': int(sub.get('reviewed_true_scene_change', pd.Series([False] * len(sub))).astype(bool).sum()),
            'reviewed_false_positive_anchor_count': int(sub.get('reviewed_false_positive', pd.Series([False] * len(sub))).astype(bool).sum()),
            'mean_visual_boundary_strength_score': '' if strength.dropna().empty else round(float(strength.mean()), 6),
            'median_visual_boundary_strength_score': '' if strength.dropna().empty else round(float(strength.median()), 6),
            'very_high_strength_count': int(band.eq('very_high').sum()),
            'high_strength_count': int(band.eq('high').sum()),
            'medium_strength_count': int(band.eq('medium').sum()),
            'low_strength_count': int(band.eq('low').sum()),
        })
    return pd.DataFrame(rows).sort_values('split')


def summarize_segments(processed: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    target_keys = [
        'scene_change_labeled_segment_features',
        'scene_change_ad_edge_5s_10s_features',
        'scene_change_audio_segment_features',
        'audio_scene_labeled_segment_features',
        'audio_scene_ad_edge_5s_10s_features',
    ]
    for key in target_keys:
        info = processed.get(key)
        if not info:
            continue
        df = pd.read_csv(info['with_split_path'], low_memory=False)
        if 'segment_type' not in df.columns:
            continue
        for (split, segment_type), sub in df.groupby(['split', 'segment_type'], dropna=False):
            count_col = pd.to_numeric(sub.get('scene_boundary_count_in_segment', pd.Series([0] * len(sub))), errors='coerce').fillna(0)
            max_strength = pd.to_numeric(sub.get('max_visual_boundary_strength_score_in_segment', pd.Series(dtype=float)), errors='coerce')
            rows.append({
                'source_file_key': key,
                'split': clean(split),
                'segment_type': clean(segment_type),
                'row_count': int(len(sub)),
                'scene_boundary_count_sum': int(count_col.sum()),
                'segments_with_scene_boundary_count': int(sub.get('has_scene_boundary_in_segment', pd.Series([False] * len(sub))).astype(bool).sum()),
                'primary_edge_hit_2s_count': int(sub.get('has_scene_boundary_near_primary_edge_2s', pd.Series([False] * len(sub))).astype(bool).sum()),
                'primary_edge_hit_5s_count': int(sub.get('has_scene_boundary_near_primary_edge_5s', pd.Series([False] * len(sub))).astype(bool).sum()),
                'primary_edge_hit_10s_count': int(sub.get('has_scene_boundary_near_primary_edge_10s', pd.Series([False] * len(sub))).astype(bool).sum()),
                'mean_scene_boundary_count_in_segment': round(float(count_col.mean()), 6) if len(sub) else 0.0,
                'mean_max_visual_boundary_strength_score_in_segment': '' if max_strength.dropna().empty else round(float(max_strength.mean()), 6),
            })
    return pd.DataFrame(rows)


def summarize_train_rule(segment_summary: pd.DataFrame) -> pd.DataFrame:
    if segment_summary.empty:
        return pd.DataFrame()
    train = segment_summary[segment_summary['split'].eq('train')].copy()
    rows = []
    for _, row in train.iterrows():
        n = int(row.get('row_count', 0))
        rows.append({
            'feature_group': row.get('source_file_key'),
            'segment_type': row.get('segment_type'),
            'train_row_count': n,
            'train_scene_boundary_count_sum': int(row.get('scene_boundary_count_sum', 0)),
            'train_segments_with_scene_boundary_count': int(row.get('segments_with_scene_boundary_count', 0)),
            'train_primary_edge_hit_2s_count': int(row.get('primary_edge_hit_2s_count', 0)),
            'train_primary_edge_hit_5s_count': int(row.get('primary_edge_hit_5s_count', 0)),
            'train_primary_edge_hit_10s_count': int(row.get('primary_edge_hit_10s_count', 0)),
            'train_primary_edge_hit_2s_rate': round(float(row.get('primary_edge_hit_2s_count', 0)) / n, 6) if n else 0.0,
            'train_primary_edge_hit_5s_rate': round(float(row.get('primary_edge_hit_5s_count', 0)) / n, 6) if n else 0.0,
            'train_primary_edge_hit_10s_rate': round(float(row.get('primary_edge_hit_10s_count', 0)) / n, 6) if n else 0.0,
            'train_mean_max_visual_boundary_strength_score': row.get('mean_max_visual_boundary_strength_score_in_segment', ''),
            'interpretation_note': 'train-only descriptive summary; do not tune using validation/test',
        })
    return pd.DataFrame(rows)


def update_latest(files: list[Path]) -> tuple[bool, list[str], list[str]]:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for path in LATEST_DIR.iterdir():
        if path.is_file():
            path.unlink()
    copied = []
    for src in files:
        if src.exists() and src.name in ALLOWED_LATEST_NAMES and src.suffix.lower() not in FORBIDDEN_SUFFIXES:
            shutil.copy2(src, LATEST_DIR / src.name)
            copied.append(src.name)
    readme = '# latest_for_chatgpt files\n\n'
    readme += f'작업명: {TASK_NAME}\n\n복사 파일 목록:\n\n'
    for name in copied:
        readme += f'- `{name}`\n'
    readme += '\n금지 파일 확인: mp4/mov/mkv/avi, wav/mp3/m4a, frame image, model/checkpoint/cache/weight 파일은 복사하지 않았다.\n'
    LATEST_README.write_text(readme, encoding='utf-8')
    copied.append(LATEST_README.name)
    forbidden = [p.name for p in LATEST_DIR.iterdir() if p.is_file() and (p.suffix.lower() in FORBIDDEN_SUFFIXES or p.name not in ALLOWED_LATEST_NAMES)]
    return not forbidden, copied, sorted(set(forbidden))


def build_summary(report: dict[str, Any]) -> str:
    file_rows = report.get('file_row_counts', {})
    file_table = '\n'.join(
        f"| {key} | {vals.get('input_row_count', 0)} | {vals.get('with_split_row_count', 0)} | {vals.get('train_row_count', 0)} | {vals.get('validation_row_count', 0)} | {vals.get('test_row_count', 0)} | {vals.get('unknown_row_count', 0)} |"
        for key, vals in file_rows.items()
    )
    with_split_files = '\n'.join(f'- `{p}`' for p in report.get('with_split_outputs', []))
    train_files = '\n'.join(f'- `{p}`' for p in report.get('train_only_outputs', []))
    processed = '\n'.join(f'- `{k}`: `{v}`' for k, v in report.get('input_files', {}).items())
    sub_table = '\n'.join(f"| {name} | {info.get('status')} | {info.get('summary', '')} |" for name, info in report.get('sub_agent_results', {}).items())
    return f"""# Scene Change Split v2.4

## 작업 개요

v2_4 고정 video_id split을 scene-change 관련 feature table에 부착했다. 전체 feature 파일은 유지하고, 각 파일의 `with_split` 버전과 `train_only` 버전을 별도로 생성했다. 이 작업은 정보 누수 방지를 위한 metadata 부착 작업이며 rule/threshold 수정 작업이 아니다.

## split 기준

- split_source: `{SPLIT_SOURCE}`
- split_note: `{SPLIT_NOTE}`

## train/validation/test video_id 목록

- train: {TRAIN_IDS}
- validation: {VALIDATION_IDS}
- test: {TEST_IDS}

## 처리한 입력 파일

{processed}

## 생성한 with_split 파일

{with_split_files}

## 생성한 train_only 파일

{train_files}

## split별 row count 요약

| file_key | input_rows | with_split_rows | train | validation | test | unknown |
|---|---:|---:|---:|---:|---:|---:|
{file_table}

## train-only summary 설명

`scene_change_train_only_rule_analysis_summary_v2_4.csv`는 train split만 기준으로 만든 descriptive summary다. rule을 자동으로 만들거나 수정한 결과가 아니며, validation/test 값을 보고 threshold나 rule을 조정하지 않기 위한 안전장치다.

## 정보 누수 방지 주의사항

scene-change 후보 추출과 visual anchor 생성은 전체 영상에 대해 수행 가능하지만, 이후 rule/threshold 결정은 train split만 사용해야 한다. validation은 train에서 만든 rule의 조정용으로만 사용하고, test는 최종 평가 전까지 사용하지 않는다. validation/test 결과를 보고 rule을 다시 수정하면 leakage 위험이 있다.

## rule/threshold 미수정 확인

이번 작업은 split 컬럼 부착과 train-only 사본 생성만 수행했다. scene-change threshold, visual strength 기준, OpenCV/ResNet 병합 기준, 광고 boundary 판단 rule은 수정하지 않았다. OpenCV/FFmpeg 후보 추출, ResNet embedding 후보 추출, visual anchor 생성, audio/OCR 생성도 다시 수행하지 않았다.

## Sub Agent 검증 결과

| Sub Agent | Status | Summary |
|---|---|---|
{sub_table}

## 다음 작업

- OCR feature 생성 후 최종 audio/scene/OCR join
- join된 최종 feature table에 split 컬럼 유지
- rule 설계 시 train_only만 사용
- validation은 조정용, test는 최종 평가용으로 보존
"""


def print_completion(report: dict[str, Any]) -> None:
    print('\n## 작업 완료: SUCCESS' if not report.get('errors') else '\n## 작업 완료: PARTIAL/FAIL')
    print('\n### 핵심 요약')
    print(f"- 작업명: `{TASK_NAME}`")
    print(f"- 처리한 입력 파일 수: {len(report.get('processed_files', []))}")
    print(f"- 생성한 with_split 파일 수: {report.get('with_split_output_count')}")
    print(f"- 생성한 train_only 파일 수: {report.get('train_only_output_count')}")
    print(f"- train video_id: {TRAIN_IDS}")
    print(f"- validation video_id: {VALIDATION_IDS}")
    print(f"- test video_id: {TEST_IDS}")
    print(f"- old_project_modified: {str(report.get('old_project_modified')).lower()}")
    print('\n### 파일별 row count')
    print('| file_key | input | with_split | train | validation | test | unknown |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for key, vals in report.get('file_row_counts', {}).items():
        print(f"| {key} | {vals.get('input_row_count', 0)} | {vals.get('with_split_row_count', 0)} | {vals.get('train_row_count', 0)} | {vals.get('validation_row_count', 0)} | {vals.get('test_row_count', 0)} | {vals.get('unknown_row_count', 0)} |")
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
    print('- OCR feature 생성 후 최종 audio/scene/OCR join')
    print('- 최종 joined feature table에도 split 컬럼 유지')
    print('- rule 설계는 train_only 기준으로 진행')
    print('\n상세 JSON은 report 파일에 저장했습니다.')


def validate(report: dict[str, Any], input_hash_before: dict[str, str], input_hash_after: dict[str, str], latest_ok: bool, latest_forbidden: list[str]) -> dict[str, Any]:
    processed = report.get('processed_files', [])
    skipped = report.get('skipped_files', [])
    file_row_counts = report.get('file_row_counts', {})
    split_counts_by_file = report.get('split_counts_by_file', {})
    unknown_counts = {k: v.get('unknown', 0) for k, v in split_counts_by_file.items()}
    mapping_ids = TRAIN_IDS + VALIDATION_IDS + TEST_IDS
    mapping_ok = len(mapping_ids) == len(set(mapping_ids)) and all(report['split_mapping'].get(str(v)) == 'train' for v in TRAIN_IDS) and all(report['split_mapping'].get(str(v)) == 'validation' for v in VALIDATION_IDS) and all(report['split_mapping'].get(str(v)) == 'test' for v in TEST_IDS)
    all_video_id = all(v.get('video_id_column_exists', False) for v in report.get('processed_files_detail', {}).values())
    input_ok = 'visual_scene_boundary_anchors' in processed and len(processed) >= 1 and all_video_id
    sub_agents = {}
    sub_agents['sub_agent_1_input_schema_validation'] = {
        'status': 'PASS' if input_ok else 'FAIL',
        'summary': '필수 anchor와 처리 파일 video_id 컬럼 확보' if input_ok else '필수 입력 또는 video_id 컬럼 누락',
        'checks': {'processed_files': processed, 'skipped_files': skipped, 'all_processed_have_video_id': all_video_id},
    }
    total_unknown = sum(unknown_counts.values())
    split_status = 'PASS' if mapping_ok and total_unknown == 0 else ('WARN' if mapping_ok and total_unknown > 0 else 'FAIL')
    sub_agents['sub_agent_2_split_mapping_validation'] = {
        'status': split_status,
        'summary': 'split mapping 정확, unknown row 없음' if split_status == 'PASS' else 'unknown video_id 또는 mapping 문제 확인 필요',
        'checks': {'mapping_ok': mapping_ok, 'unknown_counts_by_file': unknown_counts},
    }
    row_ok = True
    train_ok = True
    for key, vals in file_row_counts.items():
        if vals.get('input_row_count') != vals.get('with_split_row_count'):
            row_ok = False
        if vals.get('train_only_non_train_rows', 0) != 0:
            train_ok = False
    inputs_unchanged = input_hash_before == input_hash_after
    sub_agents['sub_agent_3_output_row_count_train_only_validation'] = {
        'status': 'PASS' if row_ok and train_ok and inputs_unchanged else 'FAIL',
        'summary': 'row count 유지, train_only에 train 외 split 없음, 원본 hash 불변' if row_ok and train_ok and inputs_unchanged else 'row count/train_only/input hash 검증 실패',
        'checks': {'with_split_row_count_matches_input': row_ok, 'train_only_contains_only_train': train_ok, 'input_hash_unchanged': inputs_unchanged},
    }
    summary_text = SUMMARY_MD.read_text(encoding='utf-8') if SUMMARY_MD.exists() else ''
    leakage_ok = all(phrase in summary_text for phrase in ['rule/threshold 수정 작업이 아니다', 'validation/test 값을 보고 threshold나 rule을 조정하지', 'split 컬럼 부착과 train-only 사본 생성만 수행'])
    sub_agents['sub_agent_4_leakage_rule_modification_validation'] = {
        'status': 'PASS' if leakage_ok else 'FAIL',
        'summary': 'rule/threshold 미수정 및 validation/test leakage 주의사항 명시' if leakage_ok else 'leakage/rule 미수정 문구 누락',
        'checks': {'summary_contains_rule_not_modified': 'rule/threshold 수정 작업이 아니다' in summary_text, 'summary_contains_validation_test_warning': 'validation/test 값을 보고 threshold나 rule을 조정하지' in summary_text, 'scene_or_audio_regenerated': False},
    }
    expected_existing_or_skipped = True
    for key, spec in FILE_SPECS.items():
        if key in processed:
            expected_existing_or_skipped = expected_existing_or_skipped and spec['with_split'].exists() and spec['train_only'].exists()
        else:
            expected_existing_or_skipped = expected_existing_or_skipped and key in skipped
    output_exists = all(p.exists() for p in [VIDEO_SPLIT_CSV, FILE_SUMMARY_CSV, ANCHOR_SUMMARY_CSV, SEGMENT_SUMMARY_CSV, TRAIN_RULE_SUMMARY_CSV, REPORT_JSON, SUMMARY_MD, RUN_LOG, SCRIPT_PATH]) and expected_existing_or_skipped
    sub_agents['sub_agent_5_output_latest_safety_validation'] = {
        'status': 'PASS' if output_exists and latest_ok and not latest_forbidden and not report.get('old_project_modified') else 'FAIL',
        'summary': 'output/latest 생성 및 old project 미수정 확인' if output_exists and latest_ok and not latest_forbidden and not report.get('old_project_modified') else 'output/latest/safety 검증 실패',
        'checks': {'outputs_exist_or_skipped': output_exists, 'latest_forbidden_files_absent': not latest_forbidden, 'old_project_modified': report.get('old_project_modified')},
        'warnings': latest_forbidden,
    }
    report['sub_agent_results'] = sub_agents
    return report


def main() -> int:
    start_time = now_iso()
    start_monotonic = time.monotonic()
    warnings: list[Any] = []
    errors: list[Any] = []
    for d in [FEATURE_DIR, REPORT_DIR, LOG_DIR, SCRIPT_DIR, LATEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    log('[STEP 1/10] input feature 파일 탐색')
    inputs = discover_inputs(warnings, errors)
    if errors:
        raise RuntimeError(errors)
    input_hash_before = {key: file_hash(path) for key, path in inputs.items()}

    log('[STEP 2/10] 안전 백업 및 old project snapshot')
    old_before = old_project_snapshot()
    timestamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    backup_dir = backup_existing(timestamp, list(inputs.values()))
    log(f'backup_dir={backup_dir}')

    log('[STEP 3/10] fixed split mapping 생성')
    split_map, split_df = build_split_mapping()
    write_csv(VIDEO_SPLIT_CSV, split_df)
    log(f'video_split_rows={len(split_df)}')

    log('[STEP 4/10] feature 파일별 split 부착 및 train_only 생성')
    processed: dict[str, dict[str, Any]] = {}
    skipped_files = []
    skipped_outputs = []
    file_summary_rows = []
    for key, spec in FILE_SPECS.items():
        if key not in inputs:
            skipped_files.append(key)
            skipped_outputs.extend([str(spec['with_split']), str(spec['train_only'])])
            continue
        path = inputs[key]
        df = pd.read_csv(path, low_memory=False)
        detail = {'input_path': str(path), 'video_id_column_exists': 'video_id' in df.columns, 'row_count': len(df)}
        if 'video_id' not in df.columns:
            warnings.append(f'skipped_no_video_id:{key}')
            skipped_files.append(key)
            skipped_outputs.extend([str(spec['with_split']), str(spec['train_only'])])
            continue
        with_split, unknown_ids, local_warnings = apply_split(df, split_map)
        train_only = with_split[with_split['split'].eq('train')].copy()
        write_csv(spec['with_split'], with_split)
        write_csv(spec['train_only'], train_only)
        counts = split_counts(with_split)
        non_train_in_train_only = int((~train_only['split'].eq('train')).sum())
        file_summary_rows.append({
            'file_key': key,
            'input_path': str(path),
            'with_split_path': str(spec['with_split']),
            'train_only_path': str(spec['train_only']),
            'input_row_count': int(len(df)),
            'with_split_row_count': int(len(with_split)),
            'train_row_count': counts['train'],
            'validation_row_count': counts['validation'],
            'test_row_count': counts['test'],
            'unknown_row_count': counts['unknown'],
            'split_applied': True,
            'warnings': ';'.join(local_warnings),
        })
        processed[key] = {
            'input_path': str(path),
            'with_split_path': str(spec['with_split']),
            'train_only_path': str(spec['train_only']),
            'input_row_count': int(len(df)),
            'with_split_row_count': int(len(with_split)),
            'train_row_count': counts['train'],
            'validation_row_count': counts['validation'],
            'test_row_count': counts['test'],
            'unknown_row_count': counts['unknown'],
            'unknown_video_ids': unknown_ids,
            'train_only_non_train_rows': non_train_in_train_only,
        }
        detail.update({'with_split_rows': len(with_split), 'unknown_video_ids': unknown_ids})
        processed[key]['detail'] = detail
        log(f'{key}: input={len(df)} train={counts["train"]} validation={counts["validation"]} test={counts["test"]} unknown={counts["unknown"]}')

    log('[STEP 5/10] split별 요약 CSV 생성')
    file_summary = pd.DataFrame(file_summary_rows)
    write_csv(FILE_SUMMARY_CSV, file_summary)
    if 'visual_scene_boundary_anchors' in processed:
        anchor_with_split = pd.read_csv(processed['visual_scene_boundary_anchors']['with_split_path'], low_memory=False)
        anchor_summary = summarize_anchor(anchor_with_split)
    else:
        anchor_summary = pd.DataFrame()
    write_csv(ANCHOR_SUMMARY_CSV, anchor_summary)
    segment_summary = summarize_segments(processed)
    write_csv(SEGMENT_SUMMARY_CSV, segment_summary)
    train_rule_summary = summarize_train_rule(segment_summary)
    write_csv(TRAIN_RULE_SUMMARY_CSV, train_rule_summary)

    log('[STEP 6/10] report 초안 및 summary 생성')
    old_after = old_project_snapshot()
    old_project_modified = old_before != old_after
    input_hash_after = {key: file_hash(path) for key, path in inputs.items()}
    report: dict[str, Any] = {
        'project_root': str(PROJECT_ROOT),
        'task_name': TASK_NAME,
        'version': VERSION,
        'start_time': start_time,
        'end_time': now_iso(),
        'actual_runtime_seconds': round(time.monotonic() - start_monotonic, 3),
        'actual_runtime_readable': readable(time.monotonic() - start_monotonic),
        'input_files': {key: str(path) for key, path in inputs.items()},
        'output_files': {p.name: str(p) for p in OUTPUT_FILES if p.exists()},
        'backup_dir': str(backup_dir),
        'split_mapping': {str(k): v for k, v in split_map.items()},
        'processed_files': list(processed.keys()),
        'processed_files_detail': {key: val['detail'] for key, val in processed.items()},
        'skipped_files': skipped_files,
        'skipped_outputs': skipped_outputs,
        'file_row_counts': {key: {k: v for k, v in val.items() if k.endswith('_count') or k == 'input_row_count' or k == 'with_split_row_count' or k == 'train_only_non_train_rows'} for key, val in processed.items()},
        'split_counts_by_file': {key: {'train': val['train_row_count'], 'validation': val['validation_row_count'], 'test': val['test_row_count'], 'unknown': val['unknown_row_count']} for key, val in processed.items()},
        'unknown_video_id_by_file': {key: val['unknown_video_ids'] for key, val in processed.items()},
        'train_video_ids': TRAIN_IDS,
        'validation_video_ids': VALIDATION_IDS,
        'test_video_ids': TEST_IDS,
        'with_split_outputs': [val['with_split_path'] for val in processed.values()],
        'train_only_outputs': [val['train_only_path'] for val in processed.values()],
        'with_split_output_count': len(processed),
        'train_only_output_count': len(processed),
        'summary_output_count': 4,
        'warnings': warnings,
        'errors': errors,
        'sub_agent_results': {},
        'old_project_modified': old_project_modified,
    }
    SUMMARY_MD.write_text(build_summary(report), encoding='utf-8')
    REPORT_JSON.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    RUN_LOG.write_text('\n'.join(LOG_LINES) + '\n', encoding='utf-8')

    log('[STEP 7/10] latest_for_chatgpt 갱신')
    latest_ok, latest_copied, latest_forbidden = update_latest(OUTPUT_FILES)
    report['latest_for_chatgpt_updated'] = latest_ok
    report['latest_for_chatgpt_files'] = latest_copied
    report['latest_for_chatgpt_forbidden_files'] = latest_forbidden

    log('[STEP 8/10] Sub Agent 검증')
    report = validate(report, input_hash_before, input_hash_after, latest_ok, latest_forbidden)
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

    log('[STEP 9/10] 최종 report/summary/log 저장')
    SUMMARY_MD.write_text(build_summary(report), encoding='utf-8')
    REPORT_JSON.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    RUN_LOG.write_text('\n'.join(LOG_LINES) + '\n', encoding='utf-8')
    shutil.copy2(REPORT_JSON, LATEST_DIR / REPORT_JSON.name)
    shutil.copy2(SUMMARY_MD, LATEST_DIR / SUMMARY_MD.name)
    shutil.copy2(RUN_LOG, LATEST_DIR / RUN_LOG.name)

    log('[STEP 10/10] 완료')
    LOG_LINES.append(f'작업 종료 시각: {report["end_time"]}')
    LOG_LINES.append(f'실제 작업 시간: {report["actual_runtime_readable"]}')
    RUN_LOG.write_text('\n'.join(LOG_LINES) + '\n', encoding='utf-8')
    shutil.copy2(RUN_LOG, LATEST_DIR / RUN_LOG.name)
    print_completion(report)
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
