#!/usr/bin/env python3
"""Build the state-machine ad review viewer v1.2 update.

This updates review artifacts from existing detector v1.2 outputs only. It does
not run the detector, extract features, tune rules/thresholds, generate
predictions, re-encode video, extract frames, or copy media files.
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
TASK_NAME = 'state_machine_ad_review_viewer_v1_2_update'
VERSION_KEY = 'v1_2'
VERSION_DOT = 'v1.2'
BASE_VERSION_KEY = 'v1_1'
BASE_VERSION_DOT = 'v1.1'
INCLUDED_SPLITS = {'train', 'validation'}
FIXED_SPLITS = {
    'train': {1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15},
    'validation': {3, 7, 18},
    'test': {4, 16, 17},
}
EXCLUDED_TEST_VIDEO_IDS = {4, 16, 17}
SPLIT_SEED = '20240524'

OUTPUT_ROOT = PROJECT_ROOT / 'outputs/review'
V11_DIR = OUTPUT_ROOT / 'state_machine_ad_review_viewer_v1_1'
V12_DIR = OUTPUT_ROOT / 'state_machine_ad_review_viewer_v1_2'
CURRENT_DIR = OUTPUT_ROOT / 'state_machine_ad_review_viewer_current'
REGISTRY_PATH = OUTPUT_ROOT / 'state_machine_ad_review_viewer_versions.json'
LATEST_DIR = PROJECT_ROOT / 'outputs/latest_for_chatgpt_state_machine_review_viewer_v1_2_update'
REPORT_DIR = PROJECT_ROOT / 'reports/review'
LOG_DIR = PROJECT_ROOT / 'logs'
BACKUP_ROOT = PROJECT_ROOT / 'backups'
REPORT_PATH = REPORT_DIR / 'state_machine_ad_review_viewer_v1_2_update_report.json'
SUMMARY_PATH = REPORT_DIR / 'state_machine_ad_review_viewer_v1_2_update_summary.md'
LOG_PATH = LOG_DIR / 'state_machine_ad_review_viewer_v1_2_update_run_log.txt'
V12_MANIFEST_NAME = 'review_manifest_v1_2_train_val.json'
CURRENT_MANIFEST_NAME = 'review_manifest_current_train_val.json'

V12_INPUTS = {
    'closed_predictions': PROJECT_ROOT / 'data/predictions/state_machine_interval_predictions_v1_2_train_val.csv',
    'open_interval_candidates': PROJECT_ROOT / 'data/predictions/state_machine_open_interval_candidates_v1_2_train_val.csv',
    'anchor_trace': PROJECT_ROOT / 'data/predictions/state_machine_anchor_trace_v1_2_train_val.csv',
    'validation_audit': PROJECT_ROOT / 'data/predictions/state_machine_detector_validation_audit_v1_2_train_val.csv',
    'audio_only_start_review_candidates': PROJECT_ROOT / 'data/predictions/state_machine_audio_only_start_review_candidates_v1_2_train_val.csv',
    'long_ad_end_review_events': PROJECT_ROOT / 'data/predictions/state_machine_long_ad_end_review_events_v1_2_train_val.csv',
    'v1_1_vs_v1_2_comparison': PROJECT_ROOT / 'data/predictions/state_machine_v1_1_vs_v1_2_comparison_train_val.csv',
    'detector_report': PROJECT_ROOT / 'reports/detectors/state_machine_interval_detector_v1_2_report.json',
    'detector_summary': PROJECT_ROOT / 'reports/detectors/state_machine_interval_detector_v1_2_summary.md',
    'adjustment_note': PROJECT_ROOT / 'reports/detectors/state_machine_interval_detector_v1_2_adjustment_note.md',
    'config': PROJECT_ROOT / 'configs/detectors/state_machine_interval_detector_v1_2_config.json',
    'actual_labels': PROJECT_ROOT / 'data/segments/ad_interval_segments_v2_4.csv',
    'split': PROJECT_ROOT / 'data/splits/video_split_v2_4.csv',
}
V11_ROLLBACK_INPUTS = {
    'closed_predictions': PROJECT_ROOT / 'data/predictions/state_machine_interval_predictions_v1_1_train_val.csv',
    'open_interval_candidates': PROJECT_ROOT / 'data/predictions/state_machine_open_interval_candidates_v1_1_train_val.csv',
    'anchor_trace': PROJECT_ROOT / 'data/predictions/state_machine_anchor_trace_v1_1_train_val.csv',
    'validation_audit': PROJECT_ROOT / 'data/predictions/state_machine_detector_validation_audit_v1_1_train_val.csv',
}
SCRIPT_FILES = {
    'build_v1_2': PROJECT_ROOT / 'scripts/review/build_state_machine_ad_review_viewer_v1_2.py',
    'serve_current': PROJECT_ROOT / 'scripts/review/serve_state_machine_ad_review_viewer_current.py',
    'switch_version': PROJECT_ROOT / 'scripts/review/switch_state_machine_review_viewer_version.py',
}

FORBIDDEN_SUFFIXES = {
    '.mp4', '.mov', '.mkv', '.avi', '.webm', '.wav', '.mp3', '.m4a', '.jpg', '.jpeg', '.png', '.webp',
    '.parquet', '.pkl', '.pickle', '.pt', '.pth', '.ckpt', '.onnx'
}
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
    backup_dir = BACKUP_ROOT / f'state_machine_ad_review_viewer_v1_2_update_{timestamp}'
    targets = [V12_DIR, CURRENT_DIR, LATEST_DIR, REGISTRY_PATH, REPORT_PATH, SUMMARY_PATH, LOG_PATH, SCRIPT_FILES['build_v1_2'], SCRIPT_FILES['serve_current'], SCRIPT_FILES['switch_version']]
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

def clamp_time(value: float | None, duration: float | None) -> float | None:
    if value is None:
        return None
    value = max(0.0, value)
    if duration and duration > 0:
        value = min(value, duration)
    return round(value, 3)

def detect_actual_schema(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        fieldnames = list(csv.DictReader(handle).fieldnames or [])
    video_col = pick_column(fieldnames, ['video_id'])
    start_col = pick_column(fieldnames, ACTUAL_START_COLUMNS)
    end_col = pick_column(fieldnames, ACTUAL_END_COLUMNS)
    type_col = pick_column(fieldnames, ACTUAL_TYPE_COLUMNS)
    sample_ad = 0; sample_non_ad = 0; sample_unknown = 0
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
    counts = {split: 0 for split in sorted(INCLUDED_SPLITS)}
    for video in videos:
        if key is None:
            counts[video['split']] += 1
        else:
            counts[video['split']] += len(video.get(key, []))
    return counts

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
        if split not in INCLUDED_SPLITS:
            continue
        if video_id in EXCLUDED_TEST_VIDEO_IDS:
            errors.append(f'Test video_id={video_id} appeared with included split={split}')
            continue
        duration = to_float(row.get('video_duration_sec'), 0.0) or 0.0
        video_path = str(row.get('video_path') or '').strip()
        path_obj = Path(video_path) if video_path else Path()
        playable = bool(video_path) and path_obj.exists() and path_obj.is_file()
        if not playable:
            warnings.append(f'Video file missing or not a file for video_id={video_id}: {video_path}')
        videos[video_id] = {'video_id': video_id, 'split': split, 'video_name': row.get('video_name') or row.get('video_title') or '', 'video_path': video_path, 'video_url': f'/media/{video_id}', 'video_duration_sec': round(duration, 3), 'playable': playable, 'playback_warning': '' if playable else 'Video file is missing on the remote server.', 'actual_intervals': [], 'predicted_intervals': [], 'open_interval_candidates': [], 'audio_only_start_review_candidates': [], 'long_ad_end_review_events': [], 'trace_summary': {}, 'counts': {}}
    for split, expected in FIXED_SPLITS.items():
        if observed.get(split, set()) != expected:
            errors.append(f'Fixed split mismatch for {split}: observed={sorted(observed.get(split, set()))}, expected={sorted(expected)}')
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
            errors.append(f'v1.2 prediction used_test_row=true: {row.get("prediction_id")}')
        split = norm_split(row.get('split'))
        video_id = to_int(row.get('video_id'))
        if split not in INCLUDED_SPLITS or video_id is None or video_id not in videos:
            continue
        if video_id in EXCLUDED_TEST_VIDEO_IDS:
            errors.append(f'Test video_id={video_id} found in v1.2 predictions')
            continue
        if str(row.get('interval_status') or '').strip().lower() != 'closed':
            warnings.append(f'Non-closed prediction ignored for video_id={video_id}: {row.get("prediction_id")}')
            continue
        video = videos[video_id]
        interval = clamp_interval(to_float(row.get('ad_start_sec')), to_float(row.get('ad_end_sec')), video['video_duration_sec'], 'state_machine_interval_predictions_v1_2_train_val', video_id, warnings)
        if not interval:
            continue
        start, end = interval
        video['predicted_intervals'].append({'prediction_id': row.get('prediction_id') or f'pred_{count + 1:06d}', 'start': start, 'end': end, 'start_reason': row.get('start_reason') or '', 'end_reason': row.get('end_reason') or '', 'start_anchor_id': row.get('start_anchor_id') or '', 'end_anchor_id': row.get('end_anchor_id') or '', 'interval_status': 'closed', 'v1_2_adjustment_flags_json': row.get('v1_2_adjustment_flags_json') or '', 'source': 'state_machine_interval_predictions_v1_2_train_val'})
        count += 1
    return count

def attach_open(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]], warnings: list[str], errors: list[str]) -> int:
    count = 0
    for row in rows:
        if truthy(row.get('used_test_row')):
            errors.append(f'v1.2 open candidate used_test_row=true: {row.get("open_candidate_id")}')
        split = norm_split(row.get('split'))
        video_id = to_int(row.get('video_id'))
        if split not in INCLUDED_SPLITS or video_id is None or video_id not in videos:
            continue
        if video_id in EXCLUDED_TEST_VIDEO_IDS:
            errors.append(f'Test video_id={video_id} found in v1.2 open candidates')
            continue
        video = videos[video_id]
        display_end = to_float(row.get('last_anchor_sec'), video['video_duration_sec'])
        interval = clamp_interval(to_float(row.get('ad_start_sec')), display_end, video['video_duration_sec'], 'state_machine_open_interval_candidates_v1_2_train_val', video_id, warnings)
        if not interval:
            continue
        start, end = interval
        video['open_interval_candidates'].append({'candidate_id': row.get('open_candidate_id') or f'open_{count + 1:06d}', 'start': start, 'display_end': end, 'last_anchor_sec': clamp_time(to_float(row.get('last_anchor_sec')), video['video_duration_sec']), 'reason': row.get('open_reason') or '', 'open_state': row.get('open_state') or '', 'start_anchor_id': row.get('start_anchor_id') or '', 'last_anchor_id': row.get('last_anchor_id') or '', 'interval_status': 'open_candidate', 'is_final_prediction': False, 'source': 'state_machine_open_interval_candidates_v1_2_train_val'})
        count += 1
    return count

def attach_audio_only(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]], errors: list[str]) -> int:
    count = 0
    for idx, row in enumerate(rows, start=2):
        split = norm_split(row.get('split'))
        video_id = to_int(row.get('video_id'))
        if split not in INCLUDED_SPLITS or video_id is None or video_id not in videos:
            continue
        if video_id in EXCLUDED_TEST_VIDEO_IDS:
            errors.append(f'Test video_id={video_id} found in audio-only rejected candidates')
            continue
        video = videos[video_id]
        time_sec = clamp_time(to_float(row.get('pending_start_time')) or to_float(row.get('transition_time_anchor')), video['video_duration_sec'])
        video['audio_only_start_review_candidates'].append({'event_id': f'audio_only_start_rejected_{idx:06d}', 'time': time_sec, 'start': time_sec, 'visual_anchor_id': row.get('visual_anchor_id') or '', 'transition_time_anchor': to_float(row.get('transition_time_anchor')), 'pending_start_time': to_float(row.get('pending_start_time')), 'pending_start_source': row.get('pending_start_source') or '', 'rejection_reason': row.get('rejection_reason') or '', 'would_have_confirmed_in_v1_1': truthy(row.get('would_have_confirmed_in_v1_1')), 'review_note': row.get('review_note') or '', 'source': 'state_machine_audio_only_start_review_candidates_v1_2_train_val'})
        count += 1
    return count

def attach_long_events(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]], errors: list[str]) -> int:
    count = 0
    for idx, row in enumerate(rows, start=2):
        split = norm_split(row.get('split'))
        video_id = to_int(row.get('video_id'))
        if split not in INCLUDED_SPLITS or video_id is None or video_id not in videos:
            continue
        if video_id in EXCLUDED_TEST_VIDEO_IDS:
            errors.append(f'Test video_id={video_id} found in long-ad end events')
            continue
        video = videos[video_id]
        time_sec = clamp_time(to_float(row.get('pending_end_time')) or to_float(row.get('transition_time_anchor')), video['video_duration_sec'])
        video['long_ad_end_review_events'].append({'event_id': f'long_ad_end_event_{idx:06d}', 'time': time_sec, 'start': time_sec, 'visual_anchor_id': row.get('visual_anchor_id') or '', 'pending_end_time': to_float(row.get('pending_end_time')), 'transition_time_anchor': to_float(row.get('transition_time_anchor')), 'current_ad_duration_sec': to_float(row.get('current_ad_duration_sec')), 'event_reason': row.get('event_reason') or '', 'end_confirmed_by_v1_2': truthy(row.get('end_confirmed_by_v1_2')), 'review_note': row.get('review_note') or '', 'source': 'state_machine_long_ad_end_review_events_v1_2_train_val'})
        count += 1
    return count

def attach_trace(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]]) -> None:
    by_video: dict[int, dict[str, Any]] = defaultdict(lambda: {'anchor_count': 0, 'first_anchor_sec': None, 'last_anchor_sec': None})
    for row in rows:
        split = norm_split(row.get('split'))
        video_id = to_int(row.get('video_id'))
        if split not in INCLUDED_SPLITS or video_id is None or video_id not in videos:
            continue
        t = to_float(row.get('candidate_time_sec') or row.get('anchor_time_sec') or row.get('time_sec') or row.get('timestamp_sec'))
        item = by_video[video_id]
        item['anchor_count'] += 1
        if t is not None:
            item['first_anchor_sec'] = t if item['first_anchor_sec'] is None else min(item['first_anchor_sec'], t)
            item['last_anchor_sec'] = t if item['last_anchor_sec'] is None else max(item['last_anchor_sec'], t)
    for video_id, item in by_video.items():
        for key in ['first_anchor_sec', 'last_anchor_sec']:
            if item[key] is not None:
                item[key] = round(item[key], 3)
        videos[video_id]['trace_summary'] = item

def finalize_manifest(videos_by_id: dict[int, dict[str, Any]], generated_at: str, comparison_count: int) -> dict[str, Any]:
    videos = []
    for video in sorted(videos_by_id.values(), key=lambda x: (x['split'], x['video_id'])):
        for key in ['actual_intervals', 'predicted_intervals', 'open_interval_candidates', 'audio_only_start_review_candidates', 'long_ad_end_review_events']:
            video[key] = sorted(video[key], key=lambda item: float(item.get('start') or item.get('time') or item.get('end') or 0))
        video['counts'] = {'actual': len(video['actual_intervals']), 'predicted': len(video['predicted_intervals']), 'open': len(video['open_interval_candidates']), 'audio_only_start_rejected': len(video['audio_only_start_review_candidates']), 'long_ad_end_review_events': len(video['long_ad_end_review_events'])}
        videos.append(video)
    manifest = {'version': VERSION_KEY, 'viewer_version': VERSION_KEY, 'detector_version': VERSION_DOT, 'base_version': BASE_VERSION_DOT, 'task': 'state_machine_ad_review_viewer', 'generated_at': generated_at, 'review_only': True, 'no_detector_run': True, 'no_feature_extraction': True, 'no_threshold_tuning': True, 'actual_label_usage': 'audit_ui_only_not_detector_decision', 'split_policy': {'included_splits': sorted(INCLUDED_SPLITS), 'split_seed': int(SPLIT_SEED), 'fixed_train_video_ids': sorted(FIXED_SPLITS['train']), 'fixed_validation_video_ids': sorted(FIXED_SPLITS['validation']), 'excluded_test_video_ids': sorted(EXCLUDED_TEST_VIDEO_IDS), 'test_included': False, 'validation_usage': 'audit_review_only'}, 'media_policy': {'video_files_copied': False, 'video_reencoding': False, 'thumbnail_generation': False, 'frame_extraction': False, 'server_serves_manifest_whitelist_only': True}, 'prediction_count': sum(len(v['predicted_intervals']) for v in videos), 'open_interval_count': sum(len(v['open_interval_candidates']) for v in videos), 'audio_only_start_rejected_count': sum(len(v['audio_only_start_review_candidates']) for v in videos), 'long_ad_audio_low_end_count': sum(len(v['long_ad_end_review_events']) for v in videos), 'comparison_row_count': comparison_count, 'counts_by_split': {'videos': split_counts(videos), 'actual_intervals': split_counts(videos, 'actual_intervals'), 'predicted_intervals': split_counts(videos, 'predicted_intervals'), 'open_interval_candidates': split_counts(videos, 'open_interval_candidates'), 'audio_only_start_review_candidates': split_counts(videos, 'audio_only_start_review_candidates'), 'long_ad_end_review_events': split_counts(videos, 'long_ad_end_review_events')}, 'videos': videos}
    manifest_ids = {int(v['video_id']) for v in videos}
    manifest['split_policy']['test_included'] = bool(manifest_ids & EXCLUDED_TEST_VIDEO_IDS)
    return manifest

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def make_static_files(manifest: dict[str, Any], logger: StepLogger) -> None:
    if not V11_DIR.exists():
        raise RuntimeError(f'v1.1 viewer folder missing: {V11_DIR}')
    V12_DIR.mkdir(parents=True, exist_ok=True)
    index = (V11_DIR / 'index.html').read_text(encoding='utf-8')
    index = index.replace('상태 전이 기반 유튜브 광고 탐지 뷰어 v1.1', '상태 전이 기반 유튜브 광고 탐지 뷰어 v1.2')
    index = index.replace('State Machine Ad Review Viewer v1.1', '상태 전이 기반 유튜브 광고 탐지 뷰어 v1.2')
    if 'audioOnlyList' not in index:
        marker = '      </div>\n    </section>\n  </main>'
        extra = '''      </div>
      <div class="review-event-columns">
        <section>
          <h2>Audio-only start rejected</h2>
          <div id="audioOnlyList" class="interval-list"></div>
        </section>
        <section>
          <h2>Long-ad end review events</h2>
          <div id="longEventList" class="interval-list"></div>
        </section>
      </div>
    </section>
  </main>'''
        index = index.replace(marker, extra, 1)
    (V12_DIR / 'index.html').write_text(index, encoding='utf-8')
    style = (V11_DIR / 'style.css').read_text(encoding='utf-8')
    if '.review-event-columns' not in style:
        style += '''

.review-event-columns {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 16px;
}

.review-event-columns h2 {
  margin: 0 0 8px;
  font-size: 16px;
}

@media (max-width: 820px) {
  .review-event-columns {
    grid-template-columns: 1fr;
  }
}
'''
    (V12_DIR / 'style.css').write_text(style, encoding='utf-8')
    app = (V11_DIR / 'app.js').read_text(encoding='utf-8')
    app = app.replace('review_manifest_v1_1_train_val.json', V12_MANIFEST_NAME)
    app = app.replace('el.viewerMeta.textContent = "human review / audit only";', 'const currentLabel = state.manifest.viewer_version || state.manifest.version || "v1_2";\n  el.viewerMeta.textContent = `current viewer ${currentLabel} | detector ${state.manifest.detector_version || "v1.2"} | human review / audit only`;')
    app = app.replace('actual=${video.counts.actual}, predicted=${video.counts.predicted}, open=${video.counts.open}', 'actual=${video.counts.actual}, predicted=${video.counts.predicted}, open=${video.counts.open}, audio-only rejected=${video.counts.audio_only_start_rejected || 0}, long-end events=${video.counts.long_ad_end_review_events || 0}')
    app = app.replace('''    "actualList",
    "predictedList",
    "openList"
''', '''    "actualList",
    "predictedList",
    "openList",
    "audioOnlyList",
    "longEventList"
''')
    app = app.replace('''  renderIntervalList(el.actualList, video.actual_intervals || [], "actual", (item) => item.actual_id || "actual");
  renderIntervalList(el.predictedList, video.predicted_intervals || [], "prediction", (item) => item.prediction_id || "prediction");
  renderIntervalList(el.openList, video.open_interval_candidates || [], "open", (item) => item.candidate_id || "open candidate", true);
}
''', '''  renderIntervalList(el.actualList, video.actual_intervals || [], "actual", (item) => item.actual_id || "actual");
  renderIntervalList(el.predictedList, video.predicted_intervals || [], "prediction", (item) => item.prediction_id || "prediction");
  renderIntervalList(el.openList, video.open_interval_candidates || [], "open", (item) => item.candidate_id || "open candidate", true);
  renderEventList(el.audioOnlyList, video.audio_only_start_review_candidates || [], "audio-only rejected", (item) => item.visual_anchor_id || item.event_id || "audio-only rejected");
  renderEventList(el.longEventList, video.long_ad_end_review_events || [], "long-ad end", (item) => item.visual_anchor_id || item.event_id || "long-ad end event");
}
''')
    insert_after = '''function renderIntervalList(container, intervals, kind, titleFn, isOpen) {
  container.innerHTML = "";
  if (!intervals.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "None";
    container.appendChild(empty);
    return;
  }
  intervals.forEach((interval) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "interval-item";
    const end = isOpen ? interval.display_end : interval.end;
    const label = document.createElement("strong");
    label.textContent = titleFn(interval);
    const time = document.createElement("span");
    time.textContent = `${formatTime(interval.start)} - ${formatTime(end)} (${kind})`;
    const reason = document.createElement("span");
    reason.textContent = interval.start_reason || interval.end_reason || interval.reason || interval.segment_type || "";
    button.append(label, time, reason);
    button.addEventListener("click", () => seekTo(Number(interval.start) || 0));
    container.appendChild(button);
  });
}
'''
    event_fn = '''function renderEventList(container, events, kind, titleFn) {
  container.innerHTML = "";
  if (!events.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "None";
    container.appendChild(empty);
    return;
  }
  events.forEach((event) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "interval-item";
    const timeSec = Number(event.start ?? event.time ?? 0) || 0;
    const label = document.createElement("strong");
    label.textContent = titleFn(event);
    const time = document.createElement("span");
    time.textContent = `${formatTime(timeSec)} (${kind})`;
    const reason = document.createElement("span");
    reason.textContent = event.rejection_reason || event.event_reason || event.review_note || "";
    button.append(label, time, reason);
    button.addEventListener("click", () => seekTo(timeSec));
    container.appendChild(button);
  });
}
'''
    if 'function renderEventList' not in app:
        app = app.replace(insert_after, insert_after + '\n' + event_fn)
    app = app.replace('[el.actualList, el.predictedList, el.openList].forEach((container) => {', '[el.actualList, el.predictedList, el.openList, el.audioOnlyList, el.longEventList].forEach((container) => {')
    (V12_DIR / 'app.js').write_text(app, encoding='utf-8')
    readme = f'''# 상태 전이 기반 유튜브 광고 탐지 뷰어 v1.2

This is the versioned review viewer for existing detector v1.2 outputs. It is a human review/audit tool, not a detector. It does not run feature extraction, rule tuning, threshold tuning, prediction generation, video re-encoding, thumbnail generation, frame extraction, OCR, audio, or scene feature recomputation.

## Version And Rollback

- This folder is the v1.2 versioned viewer: `{rel(V12_DIR)}`.
- The v1.1 viewer is preserved at `{rel(V11_DIR)}`.
- The current viewer defaults to v1.2 at `{rel(CURRENT_DIR)}`.
- Roll back current viewer to v1.1:

```bash
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_1
```

- Switch current viewer back to v1.2:

```bash
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2
```

## Run Current Viewer From VS Code Remote-SSH

```bash
cd .
python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000
```

Then forward port `8000` in the VS Code Ports panel and open `http://localhost:8000` locally.

## Media And Split Safety

Video files are not copied. The server serves only train/validation `video_path` entries registered in the current manifest. Test video IDs `4`, `16`, and `17` are excluded from manifests and media whitelist. Unsupported codecs are not transformed by the server.

## Timeline Colors

- Red: actual ad interval, shown for audit only.
- Blue: closed detector prediction interval.
- Purple: overlap between actual ad and closed prediction, including actual/open-candidate overlap when open candidates are visible.
- Blue dashed/translucent: open interval candidate. This is not a final prediction and is not used by the skip button; actual/open overlap is drawn above it in purple.

## v1.2 Review Lists

- Audio-only start rejected: candidates that v1.2 rejected because OCR confirmation was insufficient.
- Long-ad end review events: v1.2 review events for long ad end behavior; this file may legitimately be empty.

Validation is review-only. Do not use this viewer to automatically tune rules or thresholds, and do not make final performance claims from this viewer alone.
'''
    (V12_DIR / 'README_review_viewer.md').write_text(readme, encoding='utf-8')
    write_json(V12_DIR / V12_MANIFEST_NAME, manifest)
    logger.write(f'v1.2 static viewer written: {V12_DIR}')

def registry_data(current_version: str = VERSION_KEY) -> dict[str, Any]:
    return {'current_version': current_version, 'available_versions': {'v1_1': {'viewer_dir': rel(V11_DIR), 'manifest': 'review_manifest_v1_1_train_val.json', 'detector_version': 'v1.1'}, 'v1_2': {'viewer_dir': rel(V12_DIR), 'manifest': V12_MANIFEST_NAME, 'detector_version': 'v1.2', 'base_version': 'v1.1'}}, 'current_viewer_dir': rel(CURRENT_DIR), 'rollback_supported': True}

def copy_version_to_current(version: str, generated_at: str, logger: StepLogger) -> dict[str, Any]:
    data = registry_data(version)
    version_info = data['available_versions'][version]
    source_dir = PROJECT_ROOT / version_info['viewer_dir']
    manifest_name = version_info['manifest']
    if not source_dir.exists():
        raise RuntimeError(f'Versioned viewer folder missing: {source_dir}')
    if CURRENT_DIR.exists():
        dst = BACKUP_ROOT / f'state_machine_ad_review_viewer_current_before_{version}_{now_stamp()}'
        shutil.copytree(CURRENT_DIR, dst)
        backup = str(dst)
    else:
        backup = None
    if CURRENT_DIR.exists():
        shutil.rmtree(CURRENT_DIR)
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ['index.html', 'app.js', 'style.css']:
        shutil.copy2(source_dir / name, CURRENT_DIR / name)
    app_path = CURRENT_DIR / 'app.js'
    app = app_path.read_text(encoding='utf-8')
    app = app.replace(manifest_name, CURRENT_MANIFEST_NAME).replace('review_manifest_v1_1_train_val.json', CURRENT_MANIFEST_NAME).replace('review_manifest_v1_2_train_val.json', CURRENT_MANIFEST_NAME)
    replacement = f'el.viewerMeta.textContent = "current viewer {version} | detector {version_info.get("detector_version", version)} | human review / audit only";'
    app = app.replace('el.viewerMeta.textContent = "human review / audit only";', replacement)
    app = app.replace('const currentLabel = state.manifest.viewer_version || state.manifest.version || "v1_2";\n  el.viewerMeta.textContent = `current viewer ${currentLabel} | detector ${state.manifest.detector_version || "v1.2"} | human review / audit only`;', replacement)
    app_path.write_text(app, encoding='utf-8')
    shutil.copy2(source_dir / manifest_name, CURRENT_DIR / CURRENT_MANIFEST_NAME)
    current_version = {'current_version': version, 'detector_version': version_info.get('detector_version'), 'base_version': version_info.get('base_version', 'v1.1' if version == 'v1_2' else None), 'rollback_supported': True, 'rollback_target': 'v1_1' if version == 'v1_2' else 'v1_2', 'updated_at': generated_at}
    write_json(CURRENT_DIR / 'current_version.json', current_version)
    readme_current = f'''# Current State Machine Ad Review Viewer

Current viewer version: `{version}` ({version_info.get('detector_version')}).

## Run

```bash
cd .
python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000
```

Forward port `8000` in VS Code Remote-SSH and open `http://localhost:8000` locally.

## Rollback / Switch

```bash
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_1
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2
```

No media files are copied. Current media route serves only paths whitelisted in `review_manifest_current_train_val.json`.
'''
    (CURRENT_DIR / 'README_current_viewer.md').write_text(readme_current, encoding='utf-8')
    write_json(REGISTRY_PATH, data)
    bad = scan_forbidden(CURRENT_DIR)
    if bad:
        raise RuntimeError(f'Forbidden files found in current viewer: {bad}')
    logger.write(f'Current viewer switched to {version}: {CURRENT_DIR}')
    return {'current_backup': backup, 'current_version_json': current_version}

def write_summary(report: dict[str, Any]) -> None:
    lines = ['# State Machine Ad Review Viewer v1.2 Update Summary', '', '## 작업 개요', '', 'Existing detector v1.2 outputs were packaged into a versioned review viewer while preserving the v1.1 viewer for rollback. This is review/audit-only work and does not run detector decisions, feature extraction, rule/threshold tuning, or prediction generation.', '', '## v1.2로 최신화된 내용', '', f'- v1.2 viewer path: `{V12_DIR}`', f'- current viewer path: `{CURRENT_DIR}`', f'- prediction/open/audio-only rejected/long-ad event counts: `{report.get("v1_2_counts")}`', '', '## v1.1 Rollback 방법', '', '```bash', 'python scripts/review/switch_state_machine_review_viewer_version.py --version v1_1', '```', '', '## v1.2 재전환 방법', '', '```bash', 'python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2', '```', '', '## 서버 실행 방법', '', '```bash', 'cd .', 'python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000', '```', '', 'VS Code Remote-SSH Ports panel에서 `8000`을 forward한 뒤 로컬 브라우저에서 `http://localhost:8000` 접속.', '', '## 색상/시각화', '', '색상/시각화는 기존 최신 viewer 정책을 유지했다: actual red, closed prediction blue, overlap high-saturation purple, open candidate blue dashed/translucent. Actual/open-candidate overlap is drawn above open candidates in purple.', '', '## v1.2 추가 Review 항목', '', '- Open interval: final prediction이 아닌 open candidate.', '- Audio-only start rejected: OCR confirmation 부족으로 v1.2에서 reject된 start 후보.', '- Long-ad end review events: long-ad end 관련 review event. 현재 파일은 비어 있을 수 있다.', '', '## Test 보호 상태', '', f'- test_included: `{report.get("test_included")}`', f'- excluded test video IDs: `{report.get("test_video_ids_excluded")}`', '', '## Output Files', '']
    for path in report.get('output_files', []):
        lines.append(f'- `{path}`')
    lines.extend(['', '## Warnings / Errors', '', f'- warnings: `{report.get("warnings") or []}`', f'- errors: `{report.get("errors") or []}`', ''])
    SUMMARY_PATH.write_text('\n'.join(lines), encoding='utf-8')

def copy_latest(report: dict[str, Any], logger: StepLogger) -> None:
    if LATEST_DIR.exists():
        shutil.rmtree(LATEST_DIR)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    copy_pairs = [(SUMMARY_PATH, LATEST_DIR / SUMMARY_PATH.name), (REPORT_PATH, LATEST_DIR / REPORT_PATH.name), (LOG_PATH, LATEST_DIR / LOG_PATH.name), (SCRIPT_FILES['build_v1_2'], LATEST_DIR / SCRIPT_FILES['build_v1_2'].name), (SCRIPT_FILES['serve_current'], LATEST_DIR / SCRIPT_FILES['serve_current'].name), (SCRIPT_FILES['switch_version'], LATEST_DIR / SCRIPT_FILES['switch_version'].name), (REGISTRY_PATH, LATEST_DIR / REGISTRY_PATH.name), (CURRENT_DIR / 'current_version.json', LATEST_DIR / 'current_version.json'), (V12_DIR / 'README_review_viewer.md', LATEST_DIR / 'README_review_viewer.md'), (CURRENT_DIR / 'README_current_viewer.md', LATEST_DIR / 'README_current_viewer.md'), (V12_DIR / 'index.html', LATEST_DIR / 'index.html'), (V12_DIR / 'app.js', LATEST_DIR / 'app.js'), (V12_DIR / 'style.css', LATEST_DIR / 'style.css'), (V12_DIR / V12_MANIFEST_NAME, LATEST_DIR / V12_MANIFEST_NAME), (CURRENT_DIR / CURRENT_MANIFEST_NAME, LATEST_DIR / CURRENT_MANIFEST_NAME)]
    for src, dst in copy_pairs:
        if src.exists():
            shutil.copy2(src, dst)
    readme = ['# Latest Files: State Machine Review Viewer v1.2 Update', '', 'Small review/update artifacts only. No media/video/frame/cache/model/raw video/proxy/checkpoint files are included.', '', '## Files', '']
    for path in sorted(LATEST_DIR.iterdir()):
        if path.name == 'README_latest_files.md':
            continue
        readme.append(f'- `{path.name}` ({path.stat().st_size} bytes)')
    readme.extend(['', '## Commands', '', '```bash', 'python scripts/review/switch_state_machine_review_viewer_version.py --version v1_1', 'python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2', 'python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000', '```', ''])
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger = StepLogger(LOG_PATH)
    logger.step('Safety snapshot and backup')
    logger.write(f'Project root: {PROJECT_ROOT}')
    logger.write(f'Old project snapshot-only root: {OLD_PROJECT_ROOT}')
    backup = backup_targets(timestamp, logger)
    old_before = snapshot_tree(OLD_PROJECT_ROOT, REPORT_DIR / f'old_project_snapshot_before_review_viewer_v1_2_update_{timestamp}.tsv')
    v11_before = snapshot_tree(V11_DIR, REPORT_DIR / f'v1_1_viewer_snapshot_before_v1_2_update_{timestamp}.tsv')
    input_stats_before = {name: file_stats(path) for name, path in {**V12_INPUTS, **{f'v1_1_{k}': v for k, v in V11_ROLLBACK_INPUTS.items()}}.items()}
    for name, item in input_stats_before.items():
        if not item.get('exists'):
            errors.append(f'Required input missing: {name} -> {item.get("path")}')
    logger.write(f'Input stats captured: {len(input_stats_before)}')
    logger.write(f'Forbidden suffixes: {sorted(FORBIDDEN_SUFFIXES)}')
    logger.write(f'Forbidden directory parts: {sorted(FORBIDDEN_DIRECTORY_PARTS)}')
    logger.step('Locate existing viewer and v1.1 rollback source')
    existing_v1_1_viewer_found = V11_DIR.exists()
    v1_1_rollback_supported = existing_v1_1_viewer_found and all(path.exists() for path in V11_ROLLBACK_INPUTS.values())
    if not v1_1_rollback_supported:
        errors.append('v1.1 rollback source missing: need v1.1 viewer folder and v1.1 detector outputs')
    current_version_before = None
    if (CURRENT_DIR / 'current_version.json').exists():
        current_version_before = read_json(CURRENT_DIR / 'current_version.json').get('current_version')
    logger.write(f'Existing v1.1 viewer found: {existing_v1_1_viewer_found}')
    logger.write(f'v1.1 rollback supported: {v1_1_rollback_supported}')
    logger.write(f'current_version_before: {current_version_before}')
    if errors:
        logger.write('Major problem detected before v1.2 load')
        return 1
    logger.step('Load v1.2 outputs')
    rows = {name: read_csv(path) for name, path in V12_INPUTS.items() if path.suffix.lower() == '.csv'}
    split_rows = rows['split']
    actual_rows = rows['actual_labels']
    videos_by_id = load_split_videos(split_rows, warnings, errors)
    actual_schema = detect_actual_schema(V12_INPUTS['actual_labels'], actual_rows)
    if not actual_schema['usable']:
        errors.append('Actual label schema could not be resolved. Options: A) prediction-only v1.2 viewer, B) schema audit only, C) user specifies actual interval columns.')
    detector_report_loaded = V12_INPUTS['detector_report'].exists()
    detector_summary_loaded = V12_INPUTS['detector_summary'].exists()
    detector_config_loaded = V12_INPUTS['config'].exists()
    logger.write(f'v1.2 prediction rows: {len(rows["closed_predictions"])}')
    logger.write(f'v1.2 open candidate rows: {len(rows["open_interval_candidates"])}')
    logger.write(f'v1.2 audio-only rejected rows: {len(rows["audio_only_start_review_candidates"])}')
    logger.write(f'v1.2 long-ad event rows: {len(rows["long_ad_end_review_events"])}')
    logger.write(f'Actual schema: {actual_schema}')
    if errors:
        logger.write('Major problem detected while loading inputs')
        return 1
    logger.step('Build v1.2 manifest')
    actual_count = attach_actual(videos_by_id, actual_rows, actual_schema, warnings)
    prediction_count = attach_predictions(videos_by_id, rows['closed_predictions'], warnings, errors)
    open_count = attach_open(videos_by_id, rows['open_interval_candidates'], warnings, errors)
    audio_count = attach_audio_only(videos_by_id, rows['audio_only_start_review_candidates'], errors)
    long_count = attach_long_events(videos_by_id, rows['long_ad_end_review_events'], errors)
    attach_trace(videos_by_id, rows['anchor_trace'])
    manifest = finalize_manifest(videos_by_id, generated_at, len(rows['v1_1_vs_v1_2_comparison']))
    if manifest['split_policy']['test_included']:
        errors.append('Test video IDs included in v1.2 manifest')
    logger.write(f'Attached counts actual/pred/open/audio/long: {actual_count}/{prediction_count}/{open_count}/{audio_count}/{long_count}')
    if errors:
        logger.write('Manifest safety failed')
        return 1
    logger.step('Generate or update v1.2 viewer static files')
    make_static_files(manifest, logger)
    logger.step('Create current viewer as v1.2')
    current_copy = copy_version_to_current('v1_2', generated_at, logger)
    logger.step('Implement switch script')
    for name, path in SCRIPT_FILES.items():
        if not path.exists():
            errors.append(f'Required script missing after implementation phase: {path}')
        else:
            logger.write(f'Script exists: {path}')
    logger.step('Implement current server')
    if not SCRIPT_FILES['serve_current'].exists():
        errors.append(f'Current server script missing: {SCRIPT_FILES["serve_current"]}')
    else:
        logger.write(f'Current server script exists: {SCRIPT_FILES["serve_current"]}')
    logger.step('Generate report and summary')
    old_after = snapshot_tree(OLD_PROJECT_ROOT, REPORT_DIR / f'old_project_snapshot_after_review_viewer_v1_2_update_{timestamp}.tsv')
    v11_after = snapshot_tree(V11_DIR, REPORT_DIR / f'v1_1_viewer_snapshot_after_v1_2_update_{timestamp}.tsv')
    input_stats_after = {name: file_stats(Path(item['path'])) for name, item in input_stats_before.items()}
    videos = manifest['videos']
    output_files = [str(V12_DIR / 'index.html'), str(V12_DIR / 'app.js'), str(V12_DIR / 'style.css'), str(V12_DIR / V12_MANIFEST_NAME), str(V12_DIR / 'README_review_viewer.md'), str(CURRENT_DIR / 'index.html'), str(CURRENT_DIR / 'app.js'), str(CURRENT_DIR / 'style.css'), str(CURRENT_DIR / CURRENT_MANIFEST_NAME), str(CURRENT_DIR / 'current_version.json'), str(CURRENT_DIR / 'README_current_viewer.md'), str(REGISTRY_PATH), str(SCRIPT_FILES['build_v1_2']), str(SCRIPT_FILES['serve_current']), str(SCRIPT_FILES['switch_version']), str(REPORT_PATH), str(SUMMARY_PATH), str(LOG_PATH)]
    report: dict[str, Any] = {'task_name': TASK_NAME, 'version': VERSION_KEY, 'project_root': str(PROJECT_ROOT), 'generated_at': generated_at, 'input_files': input_stats_before, 'input_files_after': input_stats_after, 'output_files': output_files, 'backup': backup, 'current_backup': current_copy.get('current_backup'), 'existing_v1_1_viewer_found': existing_v1_1_viewer_found, 'v1_1_rollback_supported': v1_1_rollback_supported, 'current_version_before': current_version_before, 'current_version_after': 'v1_2', 'v1_2_counts': {'prediction_count': prediction_count, 'open_interval_count': open_count, 'audio_only_start_rejected_count': audio_count, 'long_ad_audio_low_end_count': long_count}, 'video_count_by_split': split_counts(videos), 'actual_interval_count_by_split': split_counts(videos, 'actual_intervals'), 'predicted_interval_count_by_split': split_counts(videos, 'predicted_intervals'), 'open_interval_count_by_split': split_counts(videos, 'open_interval_candidates'), 'audio_only_start_rejected_count_by_split': split_counts(videos, 'audio_only_start_review_candidates'), 'long_ad_end_review_event_count_by_split': split_counts(videos, 'long_ad_end_review_events'), 'playable_video_count': sum(1 for video in videos if video.get('playable')), 'missing_video_count': sum(1 for video in videos if not video.get('playable')), 'missing_videos': [{'video_id': video['video_id'], 'split': video['split'], 'video_path': video['video_path']} for video in videos if not video.get('playable')], 'test_included': manifest['split_policy']['test_included'], 'test_video_ids_excluded': sorted(EXCLUDED_TEST_VIDEO_IDS), 'old_project_modified': old_before.get('sha256') != old_after.get('sha256'), 'v1_1_viewer_modified': v11_before.get('sha256') != v11_after.get('sha256'), 'input_files_modified': input_stats_before != input_stats_after, 'latest_for_chatgpt_forbidden_files_found': [], 'warnings': warnings, 'errors': errors, 'actual_label_schema': actual_schema, 'detector_report_loaded': detector_report_loaded, 'detector_summary_loaded': detector_summary_loaded, 'detector_config_loaded': detector_config_loaded, 'no_detector_run': True, 'no_feature_extraction': True, 'no_threshold_tuning': True, 'no_rule_tuning': True, 'no_prediction_generation': True, 'actual_label_usage': 'audit_ui_only_not_detector_decision', 'validation_usage': 'audit_review_only', 'old_project_snapshot_before': old_before, 'old_project_snapshot_after': old_after, 'v1_1_viewer_snapshot_before': v11_before, 'v1_1_viewer_snapshot_after': v11_after, 'sub_agent_validations': {}}
    report['status'] = 'SUCCESS' if not errors else 'FAILURE'
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
    final_lines = [f'작업 상태: {report["status"]}', f'current viewer version before/after: {current_version_before} -> v1_2', f'v1.2 viewer path: {V12_DIR}', f'current viewer path: {CURRENT_DIR}', f'rollback supported: {v1_1_rollback_supported}', 'rollback command: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_1', 'v1.2 재전환 command: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2', 'server command: python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000', 'local browser: VS Code Ports panel에서 8000 forward 후 http://localhost:8000', f'video count by split: {report["video_count_by_split"]}', f'v1.2 prediction/open/audio-only rejected count: {report["v1_2_counts"]}', f'old_project_modified: {report["old_project_modified"]}', f'input_files_modified: {report["input_files_modified"]}', f'test_included: {report["test_included"]}', f'latest bundle path: {LATEST_DIR}', f'warnings: {warnings if warnings else "None"}', f'errors: {errors if errors else "None"}', '다음 단계: 서버 실행 후 VS Code port 8000 forward, local browser에서 current viewer 확인, validation video 18 먼저 검토, 필요 시 rollback script로 v1.1 복귀. Test는 아직 실행하지 말 것.']
    logger.write('\n'.join(final_lines))
    return 0 if report['status'] != 'FAILURE' else 1

if __name__ == '__main__':
    raise SystemExit(main())
