#!/usr/bin/env python3
"""Build the state-machine ad review viewer v1.3 train-only update.

This script packages existing detector v1.3 train-only outputs into a review
viewer. It does not run the detector, extract features, tune rules/thresholds,
generate predictions, re-encode media, or copy video files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path('.')
OLD_PROJECT_ROOT = Path('./_old_project_not_included')
TASK_NAME = 'state_machine_ad_review_viewer_v1_3_train_update'
VERSION_KEY = 'v1_3_train'
VERSION_DOT = 'v1.3'
BASE_VERSION_KEY = 'v1_2'
BASE_VERSION_DOT = 'v1.2'
SCOPE = 'train_only'
TRAIN_IDS = {1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15}
VALIDATION_IDS = {3, 7, 18}
TEST_IDS = {4, 16, 17}
NON_TRAIN_IDS = VALIDATION_IDS | TEST_IDS
SPLIT_SEED = '20240524'

OUTPUT_ROOT = PROJECT_ROOT / 'outputs/review'
V12_DIR = OUTPUT_ROOT / 'state_machine_ad_review_viewer_v1_2'
V13_DIR = OUTPUT_ROOT / 'state_machine_ad_review_viewer_v1_3_train'
CURRENT_DIR = OUTPUT_ROOT / 'state_machine_ad_review_viewer_current'
REGISTRY_PATH = OUTPUT_ROOT / 'state_machine_ad_review_viewer_versions.json'
LATEST_DIR = PROJECT_ROOT / 'outputs/latest_for_chatgpt_state_machine_review_viewer_v1_3_train_update'
REPORT_DIR = PROJECT_ROOT / 'reports/review'
LOG_DIR = PROJECT_ROOT / 'logs'
BACKUP_ROOT = PROJECT_ROOT / 'backups'
REPORT_PATH = REPORT_DIR / 'state_machine_ad_review_viewer_v1_3_train_update_report.json'
SUMMARY_PATH = REPORT_DIR / 'state_machine_ad_review_viewer_v1_3_train_update_summary.md'
LOG_PATH = LOG_DIR / 'state_machine_ad_review_viewer_v1_3_train_update_run_log.txt'
V13_MANIFEST_NAME = 'review_manifest_v1_3_train.json'
CURRENT_MANIFEST_NAME = 'review_manifest_current_train_val.json'

INPUTS = {
    'config': PROJECT_ROOT / 'configs/detectors/state_machine_interval_detector_v1_3_train_config.json',
    'closed_predictions': PROJECT_ROOT / 'data/predictions/state_machine_interval_predictions_v1_3_train.csv',
    'anchor_trace': PROJECT_ROOT / 'data/predictions/state_machine_anchor_trace_v1_3_train.csv',
    'open_interval_candidates': PROJECT_ROOT / 'data/predictions/state_machine_open_interval_candidates_v1_3_train.csv',
    'unresolved_long_open_candidates': PROJECT_ROOT / 'data/predictions/state_machine_unresolved_long_open_candidates_v1_3_train.csv',
    'disclosure_notice_review_candidates': PROJECT_ROOT / 'data/predictions/state_machine_disclosure_notice_review_candidates_v1_3_train.csv',
    'ocr_only_continuity_cap_events': PROJECT_ROOT / 'data/predictions/state_machine_ocr_only_continuity_cap_events_v1_3_train.csv',
    'weak_continuity_recovery_events': PROJECT_ROOT / 'data/predictions/state_machine_weak_continuity_recovery_events_v1_3_train.csv',
    'train_audit': PROJECT_ROOT / 'data/predictions/state_machine_detector_train_audit_v1_3.csv',
    'v1_2_vs_v1_3_comparison': PROJECT_ROOT / 'data/predictions/state_machine_v1_2_vs_v1_3_train_comparison.csv',
    'error_video_summary': PROJECT_ROOT / 'data/analysis/train_only_detector_error_video_summary_v1_3.csv',
    'error_interval_overlap': PROJECT_ROOT / 'data/analysis/train_only_detector_error_interval_overlap_v1_3.csv',
    'error_open_interval_summary': PROJECT_ROOT / 'data/analysis/train_only_detector_error_open_interval_summary_v1_3.csv',
    'error_trace_reason_summary': PROJECT_ROOT / 'data/analysis/train_only_detector_error_trace_reason_summary_v1_3.csv',
    'error_worst_cases': PROJECT_ROOT / 'data/analysis/train_only_detector_error_worst_cases_v1_3.csv',
    'error_rule_issue_candidates': PROJECT_ROOT / 'data/analysis/train_only_detector_error_rule_issue_candidates_v1_3.csv',
    'detector_report': PROJECT_ROOT / 'reports/detectors/state_machine_interval_detector_v1_3_train_report.json',
    'detector_summary': PROJECT_ROOT / 'reports/detectors/state_machine_interval_detector_v1_3_train_summary.md',
    'adjustment_note': PROJECT_ROOT / 'reports/detectors/state_machine_interval_detector_v1_3_adjustment_note.md',
    'actual_labels': PROJECT_ROOT / 'data/segments/ad_interval_segments_v2_4.csv',
    'split': PROJECT_ROOT / 'data/splits/video_split_v2_4.csv',
}
SCRIPT_FILES = {
    'build_v1_3_train': PROJECT_ROOT / 'scripts/review/build_state_machine_ad_review_viewer_v1_3_train.py',
    'serve_current': PROJECT_ROOT / 'scripts/review/serve_state_machine_ad_review_viewer_current.py',
    'switch_version': PROJECT_ROOT / 'scripts/review/switch_state_machine_review_viewer_version.py',
}
FORBIDDEN_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.wav', '.mp3', '.m4a', '.jpg', '.jpeg', '.png', '.webp', '.parquet', '.pkl', '.pickle', '.pt', '.pth', '.ckpt', '.onnx'}
FORBIDDEN_DIRECTORY_PARTS = {'cache', 'frames', 'frame_images', 'raw_video', 'video_proxy', 'model_cache', 'tmp', '__pycache__'}
ACTUAL_START_COLUMNS = ['ad_start_sec', 'start_sec', 'segment_start_sec', 'interval_start_sec']
ACTUAL_END_COLUMNS = ['ad_end_sec', 'end_sec', 'segment_end_sec', 'interval_end_sec']
ACTUAL_TYPE_COLUMNS = ['segment_type', 'interval_type', 'label', 'is_ad']
NON_AD_VALUES = {'non_ad', 'random_non_ad', 'post_ad', 'pre_ad', 'not_ad', 'background', 'context', 'negative', 'false', '0', 'no'}
AD_VALUES = {'ad', 'ad_interval', 'ad_full', 'advertisement', 'sponsored', 'sponsor', 'true', '1', 'yes'}

class StepLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lines: list[str] = []
        self.step_no = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def step(self, message: str) -> None:
        self.step_no += 1
        self.write(f'[STEP {self.step_no:02d}] {message}')
    def write(self, message: str) -> None:
        print(message)
        self.lines.append(message)
        self.path.write_text('\n'.join(self.lines) + '\n', encoding='utf-8')

def now_stamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')

def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def file_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'path': str(path), 'exists': False}
    stat = path.stat()
    line_count = None
    if path.is_file() and path.suffix.lower() in {'.csv', '.json', '.md', '.txt', '.py'}:
        with path.open('r', encoding='utf-8-sig', errors='replace') as handle:
            line_count = sum(1 for _ in handle)
    return {'path': str(path), 'exists': True, 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': sha256_file(path), 'line_count': line_count}

def snapshot_tree(root: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = ['relative_path\tsize_bytes\tmtime_ns\tkind']
    if not root.exists():
        output_path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
        return {'root': str(root), 'exists': False, 'file_count': 0, 'dir_count': 0, 'snapshot': str(output_path), 'sha256': sha256_file(output_path)}
    file_count = 0
    dir_count = 0
    for current, dirs, files in os.walk(root):
        dirs.sort(); files.sort()
        current_path = Path(current)
        for dirname in dirs:
            path = current_path / dirname
            try: stat = path.stat()
            except FileNotFoundError: continue
            rows.append(f'{path.relative_to(root)}\t0\t{stat.st_mtime_ns}\tdir')
            dir_count += 1
        for filename in files:
            path = current_path / filename
            try: stat = path.stat()
            except FileNotFoundError: continue
            rows.append(f'{path.relative_to(root)}\t{stat.st_size}\t{stat.st_mtime_ns}\tfile')
            file_count += 1
    output_path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    return {'root': str(root), 'exists': True, 'file_count': file_count, 'dir_count': dir_count, 'snapshot': str(output_path), 'sha256': sha256_file(output_path)}

def backup_path(path: Path, backup_root: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    dst = backup_root / rel(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, dst)
    else:
        shutil.copy2(path, dst)
    return {'source': str(path), 'backup': str(dst)}

def backup_targets(timestamp: str, logger: StepLogger) -> dict[str, Any]:
    backup_dir = BACKUP_ROOT / f'state_machine_ad_review_viewer_v1_3_train_update_{timestamp}'
    targets = [CURRENT_DIR, REGISTRY_PATH, V13_DIR, LATEST_DIR, REPORT_PATH, SUMMARY_PATH, LOG_PATH, SCRIPT_FILES['build_v1_3_train'], SCRIPT_FILES['serve_current'], SCRIPT_FILES['switch_version']]
    copied = []
    for path in targets:
        item = backup_path(path, backup_dir)
        if item:
            copied.append(item)
    logger.write(f'Backed up {len(copied)} overwrite target(s) to {backup_dir}')
    return {'backup_dir': str(backup_dir), 'copied': copied}

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or str(value).strip() == '': return default
        return int(float(str(value).strip()))
    except Exception:
        return default

def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or str(value).strip() == '': return default
        return float(str(value).strip())
    except Exception:
        return default

def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'ad'}

def norm_split(value: Any) -> str:
    return str(value or '').strip().lower()

def pick_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    normalized = {name.strip('\ufeff'): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None

def clamp_time(value: float | None, duration: float | None) -> float | None:
    if value is None:
        return None
    value = max(0.0, value)
    if duration and duration > 0:
        value = min(value, duration)
    return round(value, 3)

def clamp_interval(start: float | None, end: float | None, duration: float | None, source: str, video_id: int, warnings: list[str]) -> tuple[float, float] | None:
    if start is None or end is None:
        warnings.append(f'Missing interval boundary in {source} for video_id={video_id}')
        return None
    original = (start, end)
    start = max(0.0, start)
    if duration and duration > 0:
        end = min(end, duration)
    if start >= end:
        warnings.append(f'Invalid interval dropped in {source} for video_id={video_id}: start={original[0]}, end={original[1]}, duration={duration}')
        return None
    if original != (start, end):
        warnings.append(f'Interval clamped in {source} for video_id={video_id}: {original[0]}-{original[1]} -> {start}-{end}')
    return round(start, 3), round(end, 3)

def detect_actual_schema(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        fieldnames = list(csv.DictReader(handle).fieldnames or [])
    video_col = pick_column(fieldnames, ['video_id'])
    start_col = pick_column(fieldnames, ACTUAL_START_COLUMNS)
    end_col = pick_column(fieldnames, ACTUAL_END_COLUMNS)
    type_col = pick_column(fieldnames, ACTUAL_TYPE_COLUMNS)
    sample_ad = sample_non_ad = sample_unknown = 0
    for row in rows[:2000]:
        value = str(row.get(type_col, '') if type_col else '').strip().lower()
        if value in AD_VALUES or value.startswith('ad_'):
            sample_ad += 1
        elif value in NON_AD_VALUES:
            sample_non_ad += 1
        elif value:
            sample_unknown += 1
    return {'fieldnames': fieldnames, 'video_id_column': video_col, 'start_column': start_col, 'end_column': end_col, 'type_column': type_col, 'usable': bool(video_col and start_col and end_col), 'clear_ad_count_sample': sample_ad, 'non_ad_count_sample': sample_non_ad, 'unknown_type_count_sample': sample_unknown}

def is_actual_ad_row(row: dict[str, str], schema: dict[str, Any]) -> bool:
    type_col = schema.get('type_column')
    if not type_col:
        return True
    value = str(row.get(type_col) or '').strip().lower()
    if value in NON_AD_VALUES:
        return False
    if value in AD_VALUES or value.startswith('ad_'):
        return True
    if type_col == 'is_ad':
        return truthy(value)
    return False

def scan_forbidden(root: Path) -> list[str]:
    bad = []
    if not root.exists():
        return bad
    for path in root.rglob('*'):
        parts = {part.lower() for part in path.relative_to(root).parts}
        if parts & FORBIDDEN_DIRECTORY_PARTS:
            bad.append(str(path.relative_to(root)))
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            bad.append(str(path.relative_to(root)))
    return sorted(bad)

def split_counts(videos: list[dict[str, Any]], key: str | None = None) -> dict[str, int]:
    counts = {'train': 0}
    for video in videos:
        if key is None:
            counts['train'] += 1
        else:
            counts['train'] += len(video.get(key, []))
    return counts

def validate_train_rows(rows: list[dict[str, str]], name: str, errors: list[str]) -> None:
    for idx, row in enumerate(rows, start=2):
        split = norm_split(row.get('split'))
        video_id = to_int(row.get('video_id'))
        if split and split != 'train':
            errors.append(f'{name} row {idx} has non-train split={split}')
        if video_id in NON_TRAIN_IDS:
            errors.append(f'{name} row {idx} has validation/test video_id={video_id}')

def load_split_videos(rows: list[dict[str, str]], warnings: list[str], errors: list[str]) -> dict[int, dict[str, Any]]:
    observed = defaultdict(set)
    seeds = set()
    videos: dict[int, dict[str, Any]] = {}
    for row in rows:
        video_id = to_int(row.get('video_id'))
        split = norm_split(row.get('split'))
        if video_id is None:
            warnings.append('Split row without numeric video_id dropped')
            continue
        observed[split].add(video_id)
        if row.get('split_seed') not in (None, ''):
            seeds.add(str(row.get('split_seed')))
        if split != 'train':
            continue
        if video_id not in TRAIN_IDS:
            errors.append(f'Unexpected non-fixed train video_id={video_id}')
            continue
        duration = to_float(row.get('video_duration_sec'), 0.0) or 0.0
        video_path = str(row.get('video_path') or '').strip()
        path_obj = Path(video_path) if video_path else Path()
        playable = bool(video_path) and path_obj.exists() and path_obj.is_file()
        if not playable:
            warnings.append(f'Video file missing or not a file for video_id={video_id}: {video_path}')
        videos[video_id] = {'video_id': video_id, 'split': 'train', 'video_name': row.get('video_name') or row.get('video_title') or '', 'video_path': video_path, 'video_url': f'/media/{video_id}', 'video_duration_sec': round(duration, 3), 'playable': playable, 'playback_warning': '' if playable else 'Video file is missing on the remote server.', 'actual_intervals': [], 'predicted_intervals': [], 'open_interval_candidates': [], 'unresolved_long_open_candidates': [], 'disclosure_notice_review_candidates': [], 'ocr_only_continuity_cap_events': [], 'weak_continuity_recovery_events': [], 'trace_focus_or_summary': {}, 'train_error_summary': {}, 'counts': {}}
    if observed.get('train', set()) != TRAIN_IDS:
        errors.append(f'Fixed train split mismatch: observed={sorted(observed.get("train", set()))}, expected={sorted(TRAIN_IDS)}')
    if observed.get('validation', set()) != VALIDATION_IDS:
        errors.append(f'Fixed validation split mismatch: observed={sorted(observed.get("validation", set()))}, expected={sorted(VALIDATION_IDS)}')
    if observed.get('test', set()) != TEST_IDS:
        errors.append(f'Fixed test split mismatch: observed={sorted(observed.get("test", set()))}, expected={sorted(TEST_IDS)}')
    if seeds != {SPLIT_SEED}:
        errors.append(f'Unexpected split_seed values: {sorted(seeds)} expected={[SPLIT_SEED]}')
    return videos

def attach_actual(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]], schema: dict[str, Any], warnings: list[str]) -> int:
    count = 0
    for idx, row in enumerate(rows, start=2):
        if not is_actual_ad_row(row, schema):
            continue
        video_id = to_int(row.get(schema['video_id_column']))
        if video_id is None or video_id not in videos:
            continue
        video = videos[video_id]
        interval = clamp_interval(to_float(row.get(schema['start_column'])), to_float(row.get(schema['end_column'])), video['video_duration_sec'], 'ad_interval_segments_v2_4', video_id, warnings)
        if not interval:
            continue
        start, end = interval
        video['actual_intervals'].append({'actual_id': row.get('ad_interval_id') or row.get('segment_id') or f'actual_row_{idx}', 'start': start, 'end': end, 'source': 'ad_interval_segments_v2_4', 'segment_type': row.get(schema.get('type_column') or '', '')})
        count += 1
    return count

def attach_predictions(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]], warnings: list[str], errors: list[str]) -> int:
    count = 0
    for row in rows:
        if truthy(row.get('used_test_row')):
            errors.append(f'v1.3 prediction used_test_row=true: {row.get("prediction_id")}')
        video_id = to_int(row.get('video_id'))
        if video_id not in videos:
            continue
        if str(row.get('interval_status') or '').strip().lower() != 'closed':
            warnings.append(f'Non-closed prediction ignored for video_id={video_id}: {row.get("prediction_id")}')
            continue
        video = videos[video_id]
        interval = clamp_interval(to_float(row.get('ad_start_sec')), to_float(row.get('ad_end_sec')), video['video_duration_sec'], 'state_machine_interval_predictions_v1_3_train', video_id, warnings)
        if not interval:
            continue
        start, end = interval
        video['predicted_intervals'].append({'prediction_id': row.get('prediction_id') or f'pred_{count + 1:06d}', 'start': start, 'end': end, 'start_reason': row.get('start_reason') or '', 'end_reason': row.get('end_reason') or '', 'start_anchor_id': row.get('start_anchor_id') or '', 'end_anchor_id': row.get('end_anchor_id') or '', 'interval_status': 'closed', 'v1_3_adjustment_flags_json': row.get('v1_3_adjustment_flags_json') or '', 'source': 'state_machine_interval_predictions_v1_3_train'})
        count += 1
    return count

def attach_open(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]], warnings: list[str], errors: list[str]) -> int:
    count = 0
    for row in rows:
        if truthy(row.get('used_test_row')):
            errors.append(f'v1.3 open candidate used_test_row=true: {row.get("open_candidate_id")}')
        video_id = to_int(row.get('video_id'))
        if video_id not in videos:
            continue
        video = videos[video_id]
        interval = clamp_interval(to_float(row.get('ad_start_sec')), to_float(row.get('last_anchor_sec'), video['video_duration_sec']), video['video_duration_sec'], 'state_machine_open_interval_candidates_v1_3_train', video_id, warnings)
        if not interval:
            continue
        start, end = interval
        video['open_interval_candidates'].append({'candidate_id': row.get('open_candidate_id') or f'open_{count + 1:06d}', 'start': start, 'display_end': end, 'last_anchor_sec': clamp_time(to_float(row.get('last_anchor_sec')), video['video_duration_sec']), 'reason': row.get('open_reason') or '', 'open_state': row.get('open_state') or '', 'start_anchor_id': row.get('start_anchor_id') or '', 'last_anchor_id': row.get('last_anchor_id') or '', 'interval_status': 'open_candidate', 'is_final_prediction': False, 'source': 'state_machine_open_interval_candidates_v1_3_train'})
        count += 1
    return count

def attach_unresolved(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]], warnings: list[str]) -> int:
    count = 0
    for row in rows:
        video_id = to_int(row.get('video_id'))
        if video_id not in videos:
            continue
        video = videos[video_id]
        interval = clamp_interval(to_float(row.get('start_sec')), to_float(row.get('last_anchor_sec')), video['video_duration_sec'], 'state_machine_unresolved_long_open_candidates_v1_3_train', video_id, warnings)
        if not interval:
            continue
        start, end = interval
        video['unresolved_long_open_candidates'].append({'candidate_id': row.get('unresolved_candidate_id') or f'unresolved_{count + 1:06d}', 'start': start, 'display_end': end, 'duration_proxy_sec': to_float(row.get('duration_proxy_sec')), 'duration_ratio': to_float(row.get('duration_ratio')), 'start_anchor_id': row.get('start_anchor_id') or '', 'last_anchor_id': row.get('last_anchor_id') or '', 'start_reason': row.get('start_reason') or '', 'unresolved_reason': row.get('unresolved_reason') or '', 'review_note': row.get('review_note') or '', 'is_final_prediction': False, 'source': 'state_machine_unresolved_long_open_candidates_v1_3_train'})
        count += 1
    return count

def attach_point_events(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]], key: str, source: str, id_prefix: str) -> int:
    count = 0
    for idx, row in enumerate(rows, start=2):
        video_id = to_int(row.get('video_id'))
        if video_id not in videos:
            continue
        video = videos[video_id]
        time_sec = clamp_time(to_float(row.get('transition_time_anchor')) or to_float(row.get('pending_end_time')), video['video_duration_sec'])
        event = {'event_id': f'{id_prefix}_{idx:06d}', 'time': time_sec, 'start': time_sec, 'visual_anchor_id': row.get('visual_anchor_id') or '', 'event_reason': row.get('event_reason') or row.get('rejection_reason') or '', 'review_note': row.get('review_note') or '', 'source': source}
        for name in ['candidate_time_mmss', 'rejection_reason', 'early_disclosure_guard_applied', 'ocr_only_continuity_count', 'ocr_only_continuity_elapsed_sec', 'no_strong_ad_evidence_count', 'no_strong_ad_evidence_elapsed_sec', 'pending_end_time']:
            if name in row:
                event[name] = row.get(name)
        video[key].append(event)
        count += 1
    return count

def attach_trace(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]]) -> None:
    by_video: dict[int, dict[str, Any]] = defaultdict(lambda: {'trace_row_count': 0, 'first_anchor_sec': None, 'last_anchor_sec': None})
    for row in rows:
        video_id = to_int(row.get('video_id'))
        if video_id not in videos:
            continue
        t = to_float(row.get('candidate_time_sec') or row.get('transition_time_anchor') or row.get('anchor_time_sec') or row.get('time_sec'))
        item = by_video[video_id]
        item['trace_row_count'] += 1
        if t is not None:
            item['first_anchor_sec'] = t if item['first_anchor_sec'] is None else min(item['first_anchor_sec'], t)
            item['last_anchor_sec'] = t if item['last_anchor_sec'] is None else max(item['last_anchor_sec'], t)
    for video_id, item in by_video.items():
        for key in ['first_anchor_sec', 'last_anchor_sec']:
            if item[key] is not None:
                item[key] = round(item[key], 3)
        videos[video_id]['trace_focus_or_summary'] = item

def attach_error_summary(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]]) -> None:
    for row in rows:
        video_id = to_int(row.get('video_id'))
        if video_id in videos:
            videos[video_id]['train_error_summary'] = dict(row)

def finalize_manifest(videos_by_id: dict[int, dict[str, Any]], counts: dict[str, int], info_messages: list[str], generated_at: str) -> dict[str, Any]:
    videos = []
    for video in sorted(videos_by_id.values(), key=lambda x: x['video_id']):
        for key in ['actual_intervals', 'predicted_intervals', 'open_interval_candidates', 'unresolved_long_open_candidates', 'disclosure_notice_review_candidates', 'ocr_only_continuity_cap_events', 'weak_continuity_recovery_events']:
            video[key] = sorted(video[key], key=lambda item: float(item.get('start') or item.get('time') or item.get('end') or 0))
        video['counts'] = {'actual': len(video['actual_intervals']), 'predicted': len(video['predicted_intervals']), 'open': len(video['open_interval_candidates']), 'unresolved_long_open': len(video['unresolved_long_open_candidates']), 'disclosure_notice_rejected': len(video['disclosure_notice_review_candidates']), 'ocr_only_continuity_cap_events': len(video['ocr_only_continuity_cap_events']), 'weak_continuity_recovery_events': len(video['weak_continuity_recovery_events'])}
        videos.append(video)
    manifest_ids = {int(v['video_id']) for v in videos}
    return {'version': VERSION_KEY, 'viewer_version': VERSION_KEY, 'detector_version': VERSION_DOT, 'base_version': BASE_VERSION_DOT, 'scope': SCOPE, 'task': 'state_machine_ad_review_viewer', 'generated_at': generated_at, 'review_only': True, 'conditional_success_source': True, 'no_detector_run': True, 'no_feature_extraction': True, 'no_threshold_tuning': True, 'actual_label_usage': 'audit_ui_only_not_detector_decision', 'validation_included': False, 'test_included': False, 'split_policy': {'included_splits': ['train'], 'split_seed': int(SPLIT_SEED), 'fixed_train_video_ids': sorted(TRAIN_IDS), 'excluded_validation_video_ids': sorted(VALIDATION_IDS), 'excluded_test_video_ids': sorted(TEST_IDS), 'validation_included': bool(manifest_ids & VALIDATION_IDS), 'test_included': bool(manifest_ids & TEST_IDS)}, 'media_policy': {'video_files_copied': False, 'video_reencoding': False, 'thumbnail_generation': False, 'frame_extraction': False, 'server_serves_current_manifest_whitelist_only': True}, 'prediction_count': counts['prediction_count'], 'trace_row_count': counts['trace_row_count'], 'open_interval_count': counts['open_interval_count'], 'unresolved_long_open_count': counts['unresolved_long_open_count'], 'disclosure_notice_rejected_count': counts['disclosure_notice_rejected_count'], 'ocr_only_continuity_cap_event_count': counts['ocr_only_continuity_cap_event_count'], 'weak_continuity_recovery_event_count': counts['weak_continuity_recovery_event_count'], 'info_messages': info_messages, 'counts_by_split': {'videos': split_counts(videos), 'actual_intervals': split_counts(videos, 'actual_intervals'), 'predicted_intervals': split_counts(videos, 'predicted_intervals'), 'open_interval_candidates': split_counts(videos, 'open_interval_candidates'), 'unresolved_long_open_candidates': split_counts(videos, 'unresolved_long_open_candidates'), 'disclosure_notice_review_candidates': split_counts(videos, 'disclosure_notice_review_candidates'), 'ocr_only_continuity_cap_events': split_counts(videos, 'ocr_only_continuity_cap_events'), 'weak_continuity_recovery_events': split_counts(videos, 'weak_continuity_recovery_events')}, 'videos': videos}

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def merge_timeline_seekbar_index(index: str) -> str:
    index = index.replace('''        <div id="adTimeline" class="ad-timeline" role="img" aria-label="Actual, predicted, overlap, and open interval visualization">
          <div id="timelineLayers"></div>
          <div id="timeMarker" class="time-marker"></div>
        </div>''', '''        <div id="adTimeline" class="ad-timeline" role="slider" tabindex="0" aria-label="Video seek and advertisement distribution timeline" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0" aria-valuetext="00:00 / 00:00">
          <div id="timelineLayers"></div>
          <div id="timeMarker" class="time-marker"></div>
        </div>''')
    index = index.replace('''

      <input id="seekBar" class="seekbar" type="range" min="0" max="0" step="0.05" value="0" aria-label="Video seekbar">
''', '\n')
    return index


def merge_timeline_seekbar_style(style: str) -> str:
    style = style.replace('''select,
button,
input[type="range"] {
  font: inherit;
}''', '''select,
button {
  font: inherit;
}''')
    style = style.replace('''  overflow: hidden;
}

#timelineLayers,''', '''  overflow: hidden;
  cursor: pointer;
  touch-action: none;
  user-select: none;
}

.ad-timeline:focus-visible {
  outline: 3px solid rgba(31, 122, 224, 0.38);
  outline-offset: 3px;
}

#timelineLayers,''')
    style = style.replace('''#timelineLayers,
.timeline-layer {
  position: absolute;
  inset: 0;
}''', '''#timelineLayers,
.timeline-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
}''')
    style = style.replace('''
.seekbar {
  width: 100%;
  margin: 12px 0 10px;
  accent-color: var(--control);
}
''', '\n')
    return style


def merge_timeline_seekbar_app(app: str) -> str:
    app = app.replace('''  rafId: null
};''', '''  rafId: null,
  timelineDragging: false
};''')
    app = app.replace('''    "timeMarker",
    "seekBar",
    "playPauseButton",''', '''    "timeMarker",
    "playPauseButton",''')
    app = app.replace('''  el.seekBar.addEventListener("input", () => {
    el.videoPlayer.currentTime = Number(el.seekBar.value || 0);
    updatePlaybackUi();
  });
''', '')
    app = app.replace('''  el.playPauseButton.addEventListener("click", togglePlay);''', '''  el.adTimeline.addEventListener("pointerdown", beginTimelineSeek);
  el.adTimeline.addEventListener("keydown", handleTimelineKeydown);
  el.playPauseButton.addEventListener("click", togglePlay);''')
    app = app.replace('''  el.videoPlayer.src = video.video_url;
  el.videoPlayer.load();
  el.seekBar.max = String(durationOf(video));
  el.seekBar.value = "0";
  el.currentTimeText.textContent = formatTime(0);
  el.durationText.textContent = formatTime(durationOf(video));''', '''  el.videoPlayer.src = video.video_url;
  el.videoPlayer.load();
  el.currentTimeText.textContent = formatTime(0);
  el.durationText.textContent = formatTime(durationOf(video));
  setTimelineAria(0, durationOf(video));''')
    app = app.replace('''function syncDuration() {
  if (!state.currentVideo) {
    return;
  }
  const duration = Number.isFinite(el.videoPlayer.duration) ? el.videoPlayer.duration : durationOf(state.currentVideo);
  el.seekBar.max = String(duration);
  el.durationText.textContent = formatTime(duration);
}''', '''function syncDuration() {
  if (!state.currentVideo) {
    return;
  }
  const duration = timelineDuration();
  el.durationText.textContent = formatTime(duration);
  setTimelineAria(Number(el.videoPlayer.currentTime) || 0, duration);
}''')
    app = app.replace('''function seekRelative(delta) {
  seekTo((Number(el.videoPlayer.currentTime) || 0) + delta);
}

function seekTo(timeSec) {
  const duration = Number(el.seekBar.max) || durationOf(state.currentVideo);
  el.videoPlayer.currentTime = clamp(timeSec, 0, duration || timeSec);
  updatePlaybackUi();
}''', '''function seekRelative(delta) {
  seekTo((Number(el.videoPlayer.currentTime) || 0) + delta);
}

function timelineDuration() {
  const mediaDuration = Number.isFinite(el.videoPlayer.duration) ? Number(el.videoPlayer.duration) : 0;
  return mediaDuration > 0 ? mediaDuration : durationOf(state.currentVideo);
}

function beginTimelineSeek(event) {
  if (!state.currentVideo) {
    return;
  }
  event.preventDefault();
  state.timelineDragging = true;
  if (el.adTimeline.setPointerCapture) {
    el.adTimeline.setPointerCapture(event.pointerId);
  }
  seekFromTimelinePointer(event);

  const move = (moveEvent) => {
    if (state.timelineDragging) {
      seekFromTimelinePointer(moveEvent);
    }
  };
  const stop = (stopEvent) => {
    state.timelineDragging = false;
    if (el.adTimeline.releasePointerCapture) {
      try {
        el.adTimeline.releasePointerCapture(stopEvent.pointerId);
      } catch (error) {
        // Pointer capture can already be released by the browser.
      }
    }
    el.adTimeline.removeEventListener("pointermove", move);
    el.adTimeline.removeEventListener("pointerup", stop);
    el.adTimeline.removeEventListener("pointercancel", stop);
  };
  el.adTimeline.addEventListener("pointermove", move);
  el.adTimeline.addEventListener("pointerup", stop);
  el.adTimeline.addEventListener("pointercancel", stop);
}

function seekFromTimelinePointer(event) {
  const duration = timelineDuration();
  const rect = el.adTimeline.getBoundingClientRect();
  if (duration <= 0 || rect.width <= 0) {
    return;
  }
  const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
  seekTo(ratio * duration);
}

function handleTimelineKeydown(event) {
  if (!state.currentVideo) {
    return;
  }
  const duration = timelineDuration();
  const current = Number(el.videoPlayer.currentTime) || 0;
  if (duration <= 0) {
    return;
  }
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    seekTo(current - 5);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    seekTo(current + 5);
  } else if (event.key === "PageDown") {
    event.preventDefault();
    seekTo(current - 30);
  } else if (event.key === "PageUp") {
    event.preventDefault();
    seekTo(current + 30);
  } else if (event.key === "Home") {
    event.preventDefault();
    seekTo(0);
  } else if (event.key === "End") {
    event.preventDefault();
    seekTo(duration);
  }
}

function seekTo(timeSec) {
  const duration = timelineDuration();
  el.videoPlayer.currentTime = clamp(timeSec, 0, duration || timeSec);
  updatePlaybackUi();
}''')
    app = app.replace('''function updatePlaybackUi() {
  const current = Number(el.videoPlayer.currentTime) || 0;
  const duration = Number(el.seekBar.max) || durationOf(state.currentVideo);
  el.currentTimeText.textContent = formatTime(current);
  el.durationText.textContent = formatTime(duration);
  if (document.activeElement !== el.seekBar) {
    el.seekBar.value = String(clamp(current, 0, duration || current));
  }
  updateMarker(current);
  el.skipPredictedButton.disabled = !activeClosedPrediction();
}''', '''function updatePlaybackUi() {
  const current = Number(el.videoPlayer.currentTime) || 0;
  const duration = timelineDuration();
  el.currentTimeText.textContent = formatTime(current);
  el.durationText.textContent = formatTime(duration);
  setTimelineAria(current, duration);
  updateMarker(current);
  el.skipPredictedButton.disabled = !activeClosedPrediction();
}''')
    app = app.replace('''function updateMarker(current) {
  const duration = Number(el.seekBar.max) || durationOf(state.currentVideo);
  const percent = duration > 0 ? (clamp(current, 0, duration) / duration) * 100 : 0;
  el.timeMarker.style.left = `${percent}%`;
}

function formatTime(value) {''', '''function updateMarker(current) {
  const duration = timelineDuration();
  const percent = duration > 0 ? (clamp(current, 0, duration) / duration) * 100 : 0;
  el.timeMarker.style.left = `${percent}%`;
}

function setTimelineAria(current, duration) {
  el.adTimeline.setAttribute("aria-valuemin", "0");
  el.adTimeline.setAttribute("aria-valuemax", String(Math.max(0, duration || 0)));
  el.adTimeline.setAttribute("aria-valuenow", String(clamp(current || 0, 0, duration || 0)));
  el.adTimeline.setAttribute("aria-valuetext", `${formatTime(current)} / ${formatTime(duration)}`);
}

function formatTime(value) {''')
    return app

def make_static_files(manifest: dict[str, Any], logger: StepLogger) -> None:
    if not V12_DIR.exists():
        raise RuntimeError(f'v1.2 viewer folder missing: {V12_DIR}')
    V13_DIR.mkdir(parents=True, exist_ok=True)
    index = (V12_DIR / 'index.html').read_text(encoding='utf-8')
    index = index.replace('상태 전이 기반 유튜브 광고 탐지 뷰어 v1.2', '상태 전이 기반 유튜브 광고 탐지 뷰어 v1.3 train-only')
    index = index.replace('Audio-only start rejected', 'Disclosure notice rejected')
    index = index.replace('audioOnlyList', 'disclosureList')
    index = index.replace('Long-ad end review events', 'Unresolved long open')
    index = index.replace('longEventList', 'unresolvedList')
    index = index.replace('''      <select id="splitFilter">
        <option value="validation" selected>validation</option>
        <option value="train">train</option>
      </select>''', '''      <select id="splitFilter">
        <option value="train" selected>train</option>
      </select>''')
    if 'ocrCapList' not in index:
        marker = '      </div>\n    </section>\n  </main>'
        extra = '''      </div>
      <div class="review-event-columns">
        <section>
          <h2>OCR-only continuity cap events</h2>
          <div id="ocrCapList" class="interval-list"></div>
        </section>
        <section>
          <h2>Weak continuity recovery events</h2>
          <div id="weakRecoveryList" class="interval-list"></div>
        </section>
      </div>
    </section>
  </main>'''
        index = index.replace(marker, extra, 1)
    index = merge_timeline_seekbar_index(index)
    (V13_DIR / 'index.html').write_text(index, encoding='utf-8')
    style = (V12_DIR / 'style.css').read_text(encoding='utf-8')
    if '.segment-unresolved' not in style:
        style += '''

.segment-unresolved {
  background:
    repeating-linear-gradient(
      45deg,
      rgba(31, 122, 224, 0.72) 0,
      rgba(31, 122, 224, 0.72) 5px,
      rgba(31, 122, 224, 0.12) 5px,
      rgba(31, 122, 224, 0.12) 10px
    );
  border: 2px dashed rgba(15, 75, 155, 0.95);
}
'''
    style = merge_timeline_seekbar_style(style)
    (V13_DIR / 'style.css').write_text(style, encoding='utf-8')
    app = (V12_DIR / 'app.js').read_text(encoding='utf-8')
    app = app.replace('review_manifest_v1_2_train_val.json', V13_MANIFEST_NAME)
    app = app.replace('const currentLabel = state.manifest.viewer_version || state.manifest.version || "v1_2";\n  el.viewerMeta.textContent = `current viewer ${currentLabel} | detector ${state.manifest.detector_version || "v1.2"} | human review / audit only`;', 'el.viewerMeta.textContent = "Detector version: v1.3 train-only | current viewer v1_3_train | human review / audit only";')
    app = app.replace('audio-only rejected=${video.counts.audio_only_start_rejected || 0}, long-end events=${video.counts.long_ad_end_review_events || 0}', 'unresolved=${video.counts.unresolved_long_open || 0}, disclosure rejected=${video.counts.disclosure_notice_rejected || 0}, OCR cap=${video.counts.ocr_only_continuity_cap_events || 0}, weak recovery=${video.counts.weak_continuity_recovery_events || 0}')
    app = app.replace('const openCandidates = video.open_interval_candidates || [];\n  const openIntervals = openCandidates.map((item) => ({ start: item.start, end: item.display_end }));', 'const openCandidates = video.open_interval_candidates || [];\n  const unresolvedCandidates = video.unresolved_long_open_candidates || [];\n  const openIntervals = openCandidates.map((item) => ({ start: item.start, end: item.display_end }));\n  const unresolvedIntervals = unresolvedCandidates.map((item) => ({ start: item.start, end: item.display_end }));')
    app = app.replace('const openActualOverlapIntervals = el.toggleOpen.checked ? computeOverlaps(actualIntervals, openIntervals) : [];\n  const visibleOverlapIntervals = mergeIntervals([...closedOverlapIntervals, ...openActualOverlapIntervals]);', 'const openActualOverlapIntervals = el.toggleOpen.checked ? computeOverlaps(actualIntervals, openIntervals) : [];\n  const unresolvedActualOverlapIntervals = el.toggleOpen.checked ? computeOverlaps(actualIntervals, unresolvedIntervals) : [];\n  const visibleOverlapIntervals = mergeIntervals([...closedOverlapIntervals, ...openActualOverlapIntervals, ...unresolvedActualOverlapIntervals]);')
    app = app.replace('  if (el.toggleOpen.checked) {\n    addLayer("open", subtractIntervals(openIntervals, el.toggleOverlap.checked ? openActualOverlapIntervals : []), duration, "segment-open");\n  }', '  if (el.toggleOpen.checked) {\n    addLayer("open", subtractIntervals(openIntervals, el.toggleOverlap.checked ? openActualOverlapIntervals : []), duration, "segment-open");\n    addLayer("unresolved", subtractIntervals(unresolvedIntervals, el.toggleOverlap.checked ? unresolvedActualOverlapIntervals : []), duration, "segment-unresolved");\n  }')
    app = app.replace('"audioOnlyList"', '"disclosureList"').replace('"longEventList"', '"unresolvedList"')
    app = app.replace('    "disclosureList",\n    "unresolvedList"', '    "disclosureList",\n    "unresolvedList",\n    "ocrCapList",\n    "weakRecoveryList"')
    app = app.replace('renderEventList(el.disclosureList, video.audio_only_start_review_candidates || [], "audio-only rejected", (item) => item.visual_anchor_id || item.event_id || "audio-only rejected");', 'renderEventList(el.disclosureList, video.disclosure_notice_review_candidates || [], "disclosure rejected", (item) => item.visual_anchor_id || item.event_id || "disclosure rejected");')
    app = app.replace('renderEventList(el.unresolvedList, video.long_ad_end_review_events || [], "long-ad end", (item) => item.visual_anchor_id || item.event_id || "long-ad end event");', 'renderIntervalList(el.unresolvedList, video.unresolved_long_open_candidates || [], "unresolved long open", (item) => item.candidate_id || "unresolved long open", true);\n  renderEventList(el.ocrCapList, video.ocr_only_continuity_cap_events || [], "OCR cap event", (item) => item.visual_anchor_id || item.event_id || "OCR cap event");\n  renderEventList(el.weakRecoveryList, video.weak_continuity_recovery_events || [], "weak recovery", (item) => item.visual_anchor_id || item.event_id || "weak recovery");')
    app = app.replace('renderEventList(el.audioOnlyList, video.audio_only_start_review_candidates || [], "audio-only rejected", (item) => item.visual_anchor_id || item.event_id || "audio-only rejected");\n  renderEventList(el.longEventList, video.long_ad_end_review_events || [], "long-ad end", (item) => item.visual_anchor_id || item.event_id || "long-ad end event");', 'renderEventList(el.disclosureList, video.disclosure_notice_review_candidates || [], "disclosure rejected", (item) => item.visual_anchor_id || item.event_id || "disclosure rejected");\n  renderIntervalList(el.unresolvedList, video.unresolved_long_open_candidates || [], "unresolved long open", (item) => item.candidate_id || "unresolved long open", true);\n  renderEventList(el.ocrCapList, video.ocr_only_continuity_cap_events || [], "OCR cap event", (item) => item.visual_anchor_id || item.event_id || "OCR cap event");\n  renderEventList(el.weakRecoveryList, video.weak_continuity_recovery_events || [], "weak recovery", (item) => item.visual_anchor_id || item.event_id || "weak recovery");')
    app = app.replace('function renderIntervalList(container, intervals, kind, titleFn, isOpen) {\n  container.innerHTML = "";', 'function renderIntervalList(container, intervals, kind, titleFn, isOpen) {\n  if (!container) {\n    return;\n  }\n  container.innerHTML = "";')
    app = app.replace('function renderEventList(container, events, kind, titleFn) {\n  container.innerHTML = "";', 'function renderEventList(container, events, kind, titleFn) {\n  if (!container) {\n    return;\n  }\n  container.innerHTML = "";')
    app = app.replace('[el.actualList, el.predictedList, el.openList, el.audioOnlyList, el.longEventList].forEach((container) => {\n    container.innerHTML = "";\n  });', '[el.actualList, el.predictedList, el.openList, el.unresolvedList, el.disclosureList, el.ocrCapList, el.weakRecoveryList].forEach((container) => {\n    if (container) {\n      container.innerHTML = "";\n    }\n  });')
    app = app.replace('const trace = video.trace_summary || {};', 'const trace = video.trace_focus_or_summary || video.trace_summary || {};')
    app = app.replace('`trace anchors: ${trace.anchor_count || 0}`', '`trace rows: ${trace.trace_row_count || trace.anchor_count || 0}`')
    app = app.replace('[el.actualList, el.predictedList, el.openList, el.disclosureList, el.unresolvedList].forEach((container) => {', '[el.actualList, el.predictedList, el.openList, el.unresolvedList, el.disclosureList, el.ocrCapList, el.weakRecoveryList].forEach((container) => {')
    app = merge_timeline_seekbar_app(app)
    if 'seekBar' in app:
        raise RuntimeError('Merged timeline seekbar patch failed: seekBar reference remains in generated app.js')
    (V13_DIR / 'app.js').write_text(app, encoding='utf-8')
    readme = f'''# 상태 전이 기반 유튜브 광고 탐지 뷰어 v1.3 train-only

This is a train-only review viewer for detector v1.3 adjustment candidate outputs. It is not a final detector and not a validation/test viewer. It does not run the detector, extract features, tune rules/thresholds, generate predictions, re-encode video, extract frames, or copy media files.

## Version And Rollback

- v1.3 train-only viewer: `{rel(V13_DIR)}`.
- v1.2 rollback viewer is preserved at `{rel(V12_DIR)}`.
- Roll back current viewer to v1.2:

```bash
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2
```

- Switch current viewer back to v1.3 train-only:

```bash
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train
```

## Run Current Viewer From VS Code Remote-SSH

```bash
cd .
python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000
```

Forward port `8000` in VS Code and open `http://localhost:8000` locally.

## Safety

Video files are not copied. The current server serves only train `video_path` entries registered in the current manifest. Validation video IDs `3`, `7`, `18` and test video IDs `4`, `16`, `17` are excluded from the v1.3 train manifest and media whitelist. Unsupported codecs are not transformed.

## Timeline Colors

- Red: actual ad interval, shown for audit only.
- Blue: closed detector prediction interval.
- Purple: actual/predicted overlap and actual/open-candidate overlap.
- Blue dashed/translucent: open interval candidate.
- Blue dashed/translucent with separate list label: unresolved long open candidate.

## v1.3 Train Review Lists

- Open interval: not a final prediction and not used by skip.
- Unresolved long open: failure candidate separated from normal open intervals.
- Disclosure notice rejected: early disclosure-only start candidates rejected by v1.3 guard.
- OCR-only continuity cap events and weak continuity recovery events: header-only files are shown as events: 0.

Validation/test are intentionally excluded. Do not use this viewer to automatically tune rules or thresholds, and do not make final performance claims.
'''
    (V13_DIR / 'README_review_viewer.md').write_text(readme, encoding='utf-8')
    write_json(V13_DIR / V13_MANIFEST_NAME, manifest)
    logger.write(f'v1.3 train static viewer written: {V13_DIR}')

def registry_data(current_version: str = VERSION_KEY) -> dict[str, Any]:
    return {'current_version': current_version, 'available_versions': {'v1_2': {'viewer_dir': rel(V12_DIR), 'manifest': 'review_manifest_v1_2_train_val.json', 'detector_version': 'v1.2', 'scope': 'train_validation'}, 'v1_3_train': {'viewer_dir': rel(V13_DIR), 'manifest': V13_MANIFEST_NAME, 'detector_version': 'v1.3', 'base_version': 'v1.2', 'scope': 'train_only'}}, 'current_viewer_dir': rel(CURRENT_DIR), 'rollback_supported': True}

def copy_version_to_current(version: str, generated_at: str, logger: StepLogger) -> dict[str, Any]:
    registry = registry_data(version)
    info = registry['available_versions'][version]
    source_dir = PROJECT_ROOT / info['viewer_dir']
    manifest_name = info['manifest']
    if not source_dir.exists():
        raise RuntimeError(f'Versioned viewer folder missing: {source_dir}')
    backup = None
    if CURRENT_DIR.exists():
        dst = BACKUP_ROOT / f'state_machine_ad_review_viewer_current_before_{version}_{now_stamp()}'
        shutil.copytree(CURRENT_DIR, dst)
        backup = str(dst)
        shutil.rmtree(CURRENT_DIR)
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ['index.html', 'app.js', 'style.css']:
        shutil.copy2(source_dir / name, CURRENT_DIR / name)
    app_path = CURRENT_DIR / 'app.js'
    app = app_path.read_text(encoding='utf-8')
    app = app.replace(manifest_name, CURRENT_MANIFEST_NAME).replace('review_manifest_v1_2_train_val.json', CURRENT_MANIFEST_NAME).replace('review_manifest_v1_3_train.json', CURRENT_MANIFEST_NAME)
    app_path.write_text(app, encoding='utf-8')
    shutil.copy2(source_dir / manifest_name, CURRENT_DIR / CURRENT_MANIFEST_NAME)
    current_version = {'current_version': version, 'detector_version': info['detector_version'], 'base_version': info.get('base_version'), 'scope': info.get('scope'), 'rollback_supported': True, 'rollback_target': 'v1_2', 'validation_included': False, 'test_included': False, 'updated_at': generated_at}
    write_json(CURRENT_DIR / 'current_version.json', current_version)
    readme_current = f'''# Current State Machine Ad Review Viewer

Current viewer version: `{version}` ({info['detector_version']}, {info.get('scope')}).

## Run

```bash
cd .
python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000
```

Forward port `8000` in VS Code Remote-SSH and open `http://localhost:8000` locally.

## Rollback / Switch

```bash
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train
```

No media files are copied. Current media route serves only paths whitelisted in `review_manifest_current_train_val.json`.
'''
    (CURRENT_DIR / 'README_current_viewer.md').write_text(readme_current, encoding='utf-8')
    write_json(REGISTRY_PATH, registry)
    bad = scan_forbidden(CURRENT_DIR)
    if bad:
        raise RuntimeError(f'Forbidden files found in current viewer: {bad}')
    logger.write(f'Current viewer switched to {version}: {CURRENT_DIR}')
    return {'current_backup': backup, 'current_version_json': current_version}

def write_summary(report: dict[str, Any]) -> None:
    lines = ['# State Machine Ad Review Viewer v1.3 Train Update Summary', '', '## 작업 개요', '', 'Existing detector v1.3 train-only adjustment candidate outputs were packaged into a train-only review viewer. This is viewer/manifest work only and does not run detector decisions, feature extraction, rule/threshold tuning, or prediction generation.', '', '## Current Viewer 전환', '', f'- current viewer: `{CURRENT_DIR}`', '- current version: `v1_3_train`', f'- v1.3 train viewer: `{V13_DIR}`', '', '## v1.2 Rollback 방법', '', '```bash', 'python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2', '```', '', '## v1.3 재전환 방법', '', '```bash', 'python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train', '```', '', '## 서버 실행 방법', '', '```bash', 'cd .', 'python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000', '```', '', 'VS Code Remote-SSH Ports panel에서 `8000`을 forward하고 `http://localhost:8000` 접속.', '', '## 색상 의미', '', '- Red: actual ad interval.', '- Blue: closed detector prediction.', '- Purple: actual/predicted overlap and actual/open overlap.', '- Blue dashed/translucent: open and unresolved candidates, separated by list labels.', '', '## v1.3 항목 의미', '', '- Open interval: final prediction이 아닌 open candidate.', '- Unresolved long open: normal open과 분리한 unresolved failure candidate.', '- Disclosure notice rejected: disclosure-only start rejection candidate.', '- OCR-only continuity cap / weak continuity recovery: header-only이면 events: 0.', '', '## Validation/Test 보호 상태', '', f'- validation_included: `{report.get("validation_included")}`', f'- test_included: `{report.get("test_included")}`', '', '## Counts', '', f'- `{report.get("v1_3_counts")}`', '', '## Output Files', '']
    for path in report.get('output_files', []):
        lines.append(f'- `{path}`')
    lines.extend(['', '## Warnings / Errors', '', f'- info: `{report.get("info_messages") or []}`', f'- warnings: `{report.get("warnings") or []}`', f'- errors: `{report.get("errors") or []}`', ''])
    SUMMARY_PATH.write_text('\n'.join(lines), encoding='utf-8')

def copy_latest(report: dict[str, Any], logger: StepLogger) -> None:
    if LATEST_DIR.exists():
        shutil.rmtree(LATEST_DIR)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [(SUMMARY_PATH, LATEST_DIR / SUMMARY_PATH.name), (REPORT_PATH, LATEST_DIR / REPORT_PATH.name), (LOG_PATH, LATEST_DIR / LOG_PATH.name), (SCRIPT_FILES['build_v1_3_train'], LATEST_DIR / SCRIPT_FILES['build_v1_3_train'].name), (SCRIPT_FILES['serve_current'], LATEST_DIR / SCRIPT_FILES['serve_current'].name), (SCRIPT_FILES['switch_version'], LATEST_DIR / SCRIPT_FILES['switch_version'].name), (REGISTRY_PATH, LATEST_DIR / REGISTRY_PATH.name), (CURRENT_DIR / 'current_version.json', LATEST_DIR / 'current_version.json'), (V13_DIR / 'README_review_viewer.md', LATEST_DIR / 'README_review_viewer.md'), (CURRENT_DIR / 'README_current_viewer.md', LATEST_DIR / 'README_current_viewer.md'), (V13_DIR / 'index.html', LATEST_DIR / 'index.html'), (V13_DIR / 'app.js', LATEST_DIR / 'app.js'), (V13_DIR / 'style.css', LATEST_DIR / 'style.css'), (V13_DIR / V13_MANIFEST_NAME, LATEST_DIR / V13_MANIFEST_NAME), (CURRENT_DIR / CURRENT_MANIFEST_NAME, LATEST_DIR / CURRENT_MANIFEST_NAME)]
    for src, dst in pairs:
        if src.exists():
            shutil.copy2(src, dst)
    readme = ['# Latest Files: State Machine Review Viewer v1.3 Train Update', '', 'Small review/update artifacts only. No media/video/frame/cache/model/raw video/proxy/checkpoint files are included. Validation/test row-level viewer outputs are not included.', '', '## Files', '']
    for path in sorted(LATEST_DIR.iterdir()):
        if path.name == 'README_latest_files.md':
            continue
        readme.append(f'- `{path.name}` ({path.stat().st_size} bytes)')
    readme.extend(['', '## Commands', '', '```bash', 'python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2', 'python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train', 'python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000', '```', ''])
    (LATEST_DIR / 'README_latest_files.md').write_text('\n'.join(readme), encoding='utf-8')
    report['latest_for_chatgpt_forbidden_files_found'] = scan_forbidden(LATEST_DIR)
    logger.write(f'Latest bundle updated: {LATEST_DIR}')

def main() -> int:
    if Path.cwd().resolve() != PROJECT_ROOT:
        print(f'ERROR: run from {PROJECT_ROOT}', file=sys.stderr)
        return 2
    generated_at = now_iso()
    timestamp = now_stamp()
    warnings: list[str] = []
    errors: list[str] = []
    info_messages: list[str] = []
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger = StepLogger(LOG_PATH)
    logger.step('Safety snapshot and backup')
    logger.write(f'Project root: {PROJECT_ROOT}')
    logger.write(f'Old project snapshot-only root: {OLD_PROJECT_ROOT}')
    backup = backup_targets(timestamp, logger)
    old_before = snapshot_tree(OLD_PROJECT_ROOT, REPORT_DIR / f'old_project_snapshot_before_review_viewer_v1_3_train_update_{timestamp}.tsv')
    v12_before = snapshot_tree(V12_DIR, REPORT_DIR / f'v1_2_viewer_snapshot_before_v1_3_train_update_{timestamp}.tsv')
    input_stats_before = {name: file_stats(path) for name, path in INPUTS.items()}
    for name, item in input_stats_before.items():
        if not item.get('exists'):
            errors.append(f'Required input missing: {name} -> {item.get("path")}')
    logger.write(f'Input stats captured: {len(input_stats_before)}')
    logger.write(f'Forbidden suffixes: {sorted(FORBIDDEN_SUFFIXES)}')
    logger.write(f'Forbidden directory parts: {sorted(FORBIDDEN_DIRECTORY_PARTS)}')
    logger.step('Locate current/v1.2 viewer and rollback source')
    existing_v1_2_viewer_found = V12_DIR.exists()
    v1_2_rollback_supported = existing_v1_2_viewer_found and SCRIPT_FILES['switch_version'].exists() and SCRIPT_FILES['serve_current'].exists()
    current_version_before = None
    if (CURRENT_DIR / 'current_version.json').exists():
        current_version_before = read_json(CURRENT_DIR / 'current_version.json').get('current_version')
    if not v1_2_rollback_supported:
        errors.append('v1.2 rollback source missing: need v1.2 viewer folder, switch script, and current server')
    logger.write(f'existing_v1_2_viewer_found: {existing_v1_2_viewer_found}')
    logger.write(f'v1_2_rollback_supported: {v1_2_rollback_supported}')
    logger.write(f'current_version_before: {current_version_before}')
    if errors:
        logger.write('Major problem detected before v1.3 load')
        return 1
    logger.step('Load v1.3 outputs')
    rows = {name: read_csv(path) for name, path in INPUTS.items() if path.suffix.lower() == '.csv'}
    for name in ['closed_predictions', 'anchor_trace', 'open_interval_candidates', 'unresolved_long_open_candidates', 'disclosure_notice_review_candidates', 'ocr_only_continuity_cap_events', 'weak_continuity_recovery_events', 'train_audit', 'v1_2_vs_v1_3_comparison']:
        validate_train_rows(rows[name], name, errors)
        if len(rows[name]) == 0 and name in ['ocr_only_continuity_cap_events', 'weak_continuity_recovery_events']:
            info_messages.append(f'{name} is header-only; count=0')
    videos_by_id = load_split_videos(rows['split'], warnings, errors)
    actual_schema = detect_actual_schema(INPUTS['actual_labels'], rows['actual_labels'])
    if not actual_schema['usable']:
        errors.append('Actual label schema could not be resolved. Options: A) prediction-only viewer, B) schema audit only, C) user specifies actual interval columns.')
    logger.write(f'v1.3 prediction rows: {len(rows["closed_predictions"])}')
    logger.write(f'v1.3 trace rows: {len(rows["anchor_trace"])}')
    logger.write(f'v1.3 open rows: {len(rows["open_interval_candidates"])}')
    logger.write(f'v1.3 unresolved rows: {len(rows["unresolved_long_open_candidates"])}')
    logger.write(f'v1.3 disclosure rows: {len(rows["disclosure_notice_review_candidates"])}')
    logger.write(f'v1.3 OCR cap rows: {len(rows["ocr_only_continuity_cap_events"])}')
    logger.write(f'v1.3 weak recovery rows: {len(rows["weak_continuity_recovery_events"])}')
    logger.write(f'Actual schema: {actual_schema}')
    if errors:
        logger.write('Major problem detected while loading inputs')
        return 1
    logger.step('Build v1.3 train manifest')
    actual_count = attach_actual(videos_by_id, rows['actual_labels'], actual_schema, warnings)
    prediction_count = attach_predictions(videos_by_id, rows['closed_predictions'], warnings, errors)
    open_count = attach_open(videos_by_id, rows['open_interval_candidates'], warnings, errors)
    unresolved_count = attach_unresolved(videos_by_id, rows['unresolved_long_open_candidates'], warnings)
    disclosure_count = attach_point_events(videos_by_id, rows['disclosure_notice_review_candidates'], 'disclosure_notice_review_candidates', 'state_machine_disclosure_notice_review_candidates_v1_3_train', 'disclosure_rejected')
    ocr_cap_count = attach_point_events(videos_by_id, rows['ocr_only_continuity_cap_events'], 'ocr_only_continuity_cap_events', 'state_machine_ocr_only_continuity_cap_events_v1_3_train', 'ocr_cap')
    weak_count = attach_point_events(videos_by_id, rows['weak_continuity_recovery_events'], 'weak_continuity_recovery_events', 'state_machine_weak_continuity_recovery_events_v1_3_train', 'weak_recovery')
    attach_trace(videos_by_id, rows['anchor_trace'])
    attach_error_summary(videos_by_id, rows.get('error_video_summary', []))
    counts = {'prediction_count': prediction_count, 'trace_row_count': len(rows['anchor_trace']), 'open_interval_count': open_count, 'unresolved_long_open_count': unresolved_count, 'disclosure_notice_rejected_count': disclosure_count, 'ocr_only_continuity_cap_event_count': ocr_cap_count, 'weak_continuity_recovery_event_count': weak_count}
    manifest = finalize_manifest(videos_by_id, counts, info_messages, generated_at)
    if manifest['split_policy']['validation_included'] or manifest['split_policy']['test_included']:
        errors.append('Validation/test IDs included in v1.3 train manifest')
    logger.write(f'Attached counts actual/pred/open/unresolved/disclosure/ocr/weak: {actual_count}/{prediction_count}/{open_count}/{unresolved_count}/{disclosure_count}/{ocr_cap_count}/{weak_count}')
    if errors:
        logger.write('Manifest safety failed')
        return 1
    logger.step('Generate v1.3 train viewer static files')
    make_static_files(manifest, logger)
    logger.step('Update current viewer to v1.3 train')
    current_copy = copy_version_to_current('v1_3_train', generated_at, logger)
    logger.step('Update switch script')
    for name, path in SCRIPT_FILES.items():
        if not path.exists():
            errors.append(f'Required script missing: {path}')
        else:
            logger.write(f'Script exists: {path}')
    logger.step('Update current server if needed')
    if not SCRIPT_FILES['serve_current'].exists():
        errors.append(f'Current server script missing: {SCRIPT_FILES["serve_current"]}')
    else:
        logger.write(f'Current server script exists: {SCRIPT_FILES["serve_current"]}')
    logger.step('Generate report and summary')
    old_after = snapshot_tree(OLD_PROJECT_ROOT, REPORT_DIR / f'old_project_snapshot_after_review_viewer_v1_3_train_update_{timestamp}.tsv')
    v12_after = snapshot_tree(V12_DIR, REPORT_DIR / f'v1_2_viewer_snapshot_after_v1_3_train_update_{timestamp}.tsv')
    input_stats_after = {name: file_stats(Path(item['path'])) for name, item in input_stats_before.items()}
    videos = manifest['videos']
    output_files = [str(V13_DIR / 'index.html'), str(V13_DIR / 'app.js'), str(V13_DIR / 'style.css'), str(V13_DIR / V13_MANIFEST_NAME), str(V13_DIR / 'README_review_viewer.md'), str(CURRENT_DIR / 'index.html'), str(CURRENT_DIR / 'app.js'), str(CURRENT_DIR / 'style.css'), str(CURRENT_DIR / CURRENT_MANIFEST_NAME), str(CURRENT_DIR / 'current_version.json'), str(CURRENT_DIR / 'README_current_viewer.md'), str(REGISTRY_PATH), str(SCRIPT_FILES['build_v1_3_train']), str(SCRIPT_FILES['serve_current']), str(SCRIPT_FILES['switch_version']), str(REPORT_PATH), str(SUMMARY_PATH), str(LOG_PATH)]
    report: dict[str, Any] = {'task_name': TASK_NAME, 'version': 'v1_3_train_viewer_update', 'project_root': str(PROJECT_ROOT), 'generated_at': generated_at, 'input_files': input_stats_before, 'input_files_after': input_stats_after, 'output_files': output_files, 'backup': backup, 'current_backup': current_copy.get('current_backup'), 'existing_v1_2_viewer_found': existing_v1_2_viewer_found, 'v1_2_rollback_supported': v1_2_rollback_supported, 'current_version_before': current_version_before, 'current_version_after': 'v1_3_train', 'v1_3_counts': counts, 'video_count_by_split': split_counts(videos), 'actual_interval_count': actual_count, 'predicted_interval_count': prediction_count, 'open_interval_count': open_count, 'unresolved_long_open_count': unresolved_count, 'disclosure_notice_rejected_count': disclosure_count, 'ocr_only_continuity_cap_event_count': ocr_cap_count, 'weak_continuity_recovery_event_count': weak_count, 'playable_video_count': sum(1 for video in videos if video.get('playable')), 'missing_video_count': sum(1 for video in videos if not video.get('playable')), 'missing_videos': [{'video_id': video['video_id'], 'split': video['split'], 'video_path': video['video_path']} for video in videos if not video.get('playable')], 'validation_included': manifest['split_policy']['validation_included'], 'test_included': manifest['split_policy']['test_included'], 'validation_video_ids_excluded': sorted(VALIDATION_IDS), 'test_video_ids_excluded': sorted(TEST_IDS), 'old_project_modified': old_before.get('sha256') != old_after.get('sha256'), 'v1_2_viewer_modified': v12_before.get('sha256') != v12_after.get('sha256'), 'input_files_modified': input_stats_before != input_stats_after, 'latest_for_chatgpt_forbidden_files_found': [], 'info_messages': info_messages, 'warnings': warnings, 'errors': errors, 'actual_label_schema': actual_schema, 'detector_report_loaded': INPUTS['detector_report'].exists(), 'detector_summary_loaded': INPUTS['detector_summary'].exists(), 'detector_config_loaded': INPUTS['config'].exists(), 'no_detector_run': True, 'no_feature_extraction': True, 'no_threshold_tuning': True, 'no_rule_tuning': True, 'no_prediction_generation': True, 'actual_label_usage': 'audit_ui_only_not_detector_decision', 'validation_usage': 'excluded_train_only_viewer', 'old_project_snapshot_before': old_before, 'old_project_snapshot_after': old_after, 'v1_2_viewer_snapshot_before': v12_before, 'v1_2_viewer_snapshot_after': v12_after, 'sub_agent_validations': {}}
    report['status'] = 'CONDITIONAL_SUCCESS' if not errors else 'FAILURE'
    write_json(REPORT_PATH, report)
    write_summary(report)
    logger.write(f'Report written: {REPORT_PATH}')
    logger.write(f'Summary written: {SUMMARY_PATH}')
    logger.step('Sub Agent validations')
    logger.write('Independent Sub Agent validations are run by the orchestration step and appended to this report/summary.')
    logger.step('Update latest bundle')
    copy_latest(report, logger)
    if report['latest_for_chatgpt_forbidden_files_found']:
        report['errors'].append(f'Forbidden latest files: {report["latest_for_chatgpt_forbidden_files_found"]}')
        report['status'] = 'FAILURE'
    write_json(REPORT_PATH, report)
    write_summary(report)
    copy_latest(report, logger)
    write_json(REPORT_PATH, report)
    shutil.copy2(REPORT_PATH, LATEST_DIR / REPORT_PATH.name)
    shutil.copy2(SUMMARY_PATH, LATEST_DIR / SUMMARY_PATH.name)
    shutil.copy2(LOG_PATH, LATEST_DIR / LOG_PATH.name)
    logger.step('Final human-readable summary')
    final_lines = [f'작업 상태: {report["status"]}', f'current viewer version before/after: {current_version_before} -> v1_3_train', f'v1.3 train viewer path: {V13_DIR}', f'current viewer path: {CURRENT_DIR}', f'rollback supported: {v1_2_rollback_supported}', 'rollback command: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2', 'v1.3 재전환 command: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train', 'server command: python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000', 'local browser: VS Code Ports panel에서 8000 forward 후 http://localhost:8000', f'train video count: {len(videos)}', f'v1.3 counts: {counts}', f'old_project_modified: {report["old_project_modified"]}', f'input_files_modified: {report["input_files_modified"]}', f'validation_included: {report["validation_included"]}', f'test_included: {report["test_included"]}', f'latest bundle path: {LATEST_DIR}', f'info: {info_messages if info_messages else "None"}', f'warnings: {warnings if warnings else "None"}', f'errors: {errors if errors else "None"}', '다음 단계: 서버 실행 후 VS Code port 8000 forward, local browser에서 current viewer 확인, v1.3 train worst videos 10, 12, 9, 1, 13 먼저 검토, 필요 시 rollback script로 v1.2 복귀. Validation/test는 아직 실행하지 말 것.']
    logger.write('\n'.join(final_lines))
    return 0 if report['status'] != 'FAILURE' else 1

if __name__ == '__main__':
    raise SystemExit(main())
