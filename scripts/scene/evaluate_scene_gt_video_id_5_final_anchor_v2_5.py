#!/usr/bin/env python3
"""Evaluate video_id=5 final scene anchors against manual scene-transition GT.

This is a single-video Development Set diagnostic evaluation. It does not modify
existing detectors, anchors, split files, candidates, OCR outputs, media, cache,
or model/package directories.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import openpyxl
except Exception as exc:  # pragma: no cover - reported at runtime
    openpyxl = None
    OPENPYXL_ERROR = exc
else:
    OPENPYXL_ERROR = None

PROJECT_ROOT = Path('.')
OLD_PROJECT_ROOT = Path('./_old_project_not_included')
TASK_NAME = 'scene_gt_eval_video_id_5_final_anchor_v2_5'
VERSION = 'v2_5_diagnostic_video5'
VIDEO_ID = 5
SPLIT_SEED = 20240524
DEVELOPMENT_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
DIAGNOSTIC_IDS = [3, 7, 18]
PURE_TEST_IDS = [4, 16, 17]
TOLERANCES = [0.5, 1.0, 2.0]
MATCHING_POLICY = 'greedy_nearest_distance'
SPLIT_TERMINOLOGY_NOTE = 'original v2.4 train split is redefined as Development Set for rule design, cue analysis, and diagnostic evaluation.'
REQUIRED_SPLIT_EXPLANATION = '본 프로젝트는 학습 기반 모델이 아니라 rule-based detector이므로, 기존 train split은 Development Set으로 두고 rule 설계와 cue 분석, 오류 진단에 사용한다. 공개용 설명에서는 기존 validation과 test를 Test Set으로 통합해 규칙 고정 이후 평가 대상으로 설명한다.'

GT_XLSX = PROJECT_ROOT / 'data/labels/scene_transition_gt_video_id_5_diagnostic_v2_5.xlsx'
SPLIT_PATH = PROJECT_ROOT / 'data/splits/video_split_v2_4.csv'
MANIFEST_PATH = PROJECT_ROOT / 'data/video_metadata/video_manifest_v2_2.csv'
OPENCV_PATH = PROJECT_ROOT / 'data/scene/scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed.csv'
RESNET_PATH = PROJECT_ROOT / 'data/review/resnet_scene_candidate_human_review_v2_4_usefulness_analysis.csv'
TRANSNET_SWEEP_PATH = PROJECT_ROOT / 'data/scene/transnetv2_conservative_sweep_candidates_v2_4_train.csv'
TRANSNET_BEST_PATH = PROJECT_ROOT / 'data/scene/transnetv2_conservative_best_candidate_v2_4_train.csv'
TRANSNET_REPORT_PATH = PROJECT_ROOT / 'reports/scene/transnetv2_conservative_sweep_v2_4_report.json'
FINAL_DEV_ANCHOR_PATH = PROJECT_ROOT / 'data/scene/final_scene_boundary_anchor_v2_5_development.csv'
CANONICAL_PATH = PROJECT_ROOT / 'data/features/visual_scene_boundary_anchors_v2_4_with_split.csv'
CANONICAL_FALLBACK_PATH = PROJECT_ROOT / 'data/features/visual_scene_boundary_anchors_v2_4.csv'
ABLATION_REPORT_PATH = PROJECT_ROOT / 'reports/scene/scene_model_ablation_v2_4_report.json'
ABLATION_SUMMARY_PATH = PROJECT_ROOT / 'reports/scene/scene_model_ablation_v2_4_summary.md'
ABLATION_DOC_PATH = PROJECT_ROOT / 'reports/scene/scene_model_ablation_document_report_v2_4.md'

SCRIPT_PATH = PROJECT_ROOT / 'scripts/scene/evaluate_scene_gt_video_id_5_final_anchor_v2_5.py'
GT_SECONDS_PATH = PROJECT_ROOT / 'data/labels/scene_transition_gt_video_id_5_diagnostic_v2_5_seconds.csv'
GT_VALIDATION_PATH = PROJECT_ROOT / 'data/labels/scene_transition_gt_video_id_5_diagnostic_v2_5_validation.csv'
CANDIDATE_FAMILIES_PATH = PROJECT_ROOT / 'data/scene/scene_gt_eval_video_id_5_candidate_families_v2_5.csv'
FINAL_ANCHOR_VIDEO5_PATH = PROJECT_ROOT / 'data/scene/final_three_model_scene_anchor_video_id_5_diagnostic_v2_5.csv'
MATCHES_PATH = PROJECT_ROOT / 'data/scene/scene_gt_eval_video_id_5_matches_v2_5.csv'
UNMATCHED_GT_PATH = PROJECT_ROOT / 'data/scene/scene_gt_eval_video_id_5_unmatched_gt_v2_5.csv'
FALSE_POSITIVE_PATH = PROJECT_ROOT / 'data/scene/scene_gt_eval_video_id_5_false_positive_candidates_v2_5.csv'
PERFORMANCE_SUMMARY_PATH = PROJECT_ROOT / 'data/scene/scene_gt_eval_video_id_5_performance_summary_v2_5.csv'
SOURCE_DIAGNOSTIC_PATH = PROJECT_ROOT / 'data/scene/scene_gt_eval_video_id_5_source_diagnostic_summary_v2_5.csv'
SUMMARY_MD_PATH = PROJECT_ROOT / 'reports/scene/scene_gt_eval_video_id_5_final_anchor_v2_5_summary.md'
REPORT_JSON_PATH = PROJECT_ROOT / 'reports/scene/scene_gt_eval_video_id_5_final_anchor_v2_5_report.json'
FINDINGS_MD_PATH = PROJECT_ROOT / 'reports/scene/scene_gt_eval_video_id_5_final_anchor_v2_5_findings.md'
RUN_LOG_PATH = PROJECT_ROOT / 'logs/scene_gt_eval_video_id_5_final_anchor_v2_5_run_log.txt'
LATEST_BUNDLE = PROJECT_ROOT / 'outputs/latest_for_chatgpt_scene_gt_eval_video_id_5_v2_5'
SHARED_DIR = PROJECT_ROOT / 'outputs/latest_scene_gt_video5'
LATEST_SCENE_DIR = PROJECT_ROOT / 'outputs/latest_scene'

TRANSNET_FAMILY = 'transnetv2_threshold_0_7_dedup_5'
TRANSNET_THRESHOLD = 0.7
TRANSNET_DEDUP_SEC = 5.0
FINAL_CLUSTER_WINDOW_SEC = 2.0
DUPLICATE_GT_WINDOW_SEC = 0.3

SPLIT_COLUMNS = {
    'original_split_v2_4': 'train',
    'split_role_v2_5': 'development',
    'evaluation_subset_v2_5': 'none',
    'split_terminology_note': SPLIT_TERMINOLOGY_NOTE,
}

GT_SECONDS_COLUMNS = [
    'video_id', 'scene_gt_id', 'original_split_v2_4', 'split_role_v2_5', 'evaluation_subset_v2_5',
    'split_terminology_note', 'boundary_mmss_original', 'boundary_sec', 'parse_status', 'parse_warning',
    'source_file', 'row_index', 'notes',
]
GT_VALIDATION_COLUMNS = ['validation_item', 'status', 'count', 'examples_json', 'notes']
CANDIDATE_COLUMNS = [
    'candidate_family', 'video_id', 'original_split_v2_4', 'split_role_v2_5', 'evaluation_subset_v2_5',
    'split_terminology_note', 'candidate_sec', 'candidate_frame', 'source_model', 'source_relation', 'score',
    'threshold', 'dedup_window_sec', 'cluster_id', 'cluster_member_count', 'source_members_json', 'notes',
]
FINAL_ANCHOR_COLUMNS = [
    'final_anchor_id', 'video_id', 'original_split_v2_4', 'split_role_v2_5', 'evaluation_subset_v2_5',
    'split_terminology_note', 'anchor_sec', 'anchor_frame', 'cluster_min_sec', 'cluster_max_sec',
    'cluster_member_count', 'source_relation', 'has_opencv_ffmpeg', 'has_resnet', 'has_transnetv2_conservative',
    'opencv_member_count', 'resnet_member_count', 'transnetv2_member_count', 'source_members_json',
    'representative_rule', 'confidence_or_score', 'notes',
]
MATCH_COLUMNS = [
    'tolerance_sec', 'candidate_family', 'video_id', 'scene_gt_id', 'gt_boundary_sec',
    'gt_boundary_mmss_original', 'matched_candidate_sec', 'distance_sec', 'match_status',
    'candidate_source_relation', 'candidate_cluster_id', 'matching_policy', 'notes',
]
UNMATCHED_GT_COLUMNS = [
    'tolerance_sec', 'candidate_family', 'video_id', 'scene_gt_id', 'gt_boundary_sec',
    'gt_boundary_mmss_original', 'nearest_candidate_sec', 'nearest_candidate_distance_sec', 'false_negative', 'notes',
]
FP_COLUMNS = [
    'tolerance_sec', 'candidate_family', 'video_id', 'candidate_sec', 'candidate_source_relation',
    'candidate_cluster_id', 'nearest_gt_sec', 'nearest_gt_distance_sec', 'false_positive', 'notes',
]
PERFORMANCE_COLUMNS = [
    'candidate_family', 'tolerance_sec', 'gt_count', 'candidate_count', 'true_positive_count',
    'false_positive_count', 'false_negative_count', 'precision', 'recall', 'f1', 'candidates_per_minute',
    'false_positives_per_minute', 'median_match_distance_sec', 'mean_match_distance_sec',
    'p90_match_distance_sec', 'max_match_distance_sec', 'matching_policy', 'notes',
]
SOURCE_DIAGNOSTIC_COLUMNS = ['diagnostic_metric', 'candidate_family', 'tolerance_sec', 'metric_value', 'interpretation', 'notes']

FORBIDDEN_SUFFIXES = {
    '.mp4', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.m4a', '.jpg', '.jpeg', '.png', '.webp', '.gif',
    '.pt', '.pth', '.ckpt', '.onnx', '.pkl', '.pickle', '.parquet',
}


def now_iso() -> str:
    return datetime_now().astimezone().isoformat(timespec='seconds')


def datetime_now() -> dt.datetime:
    return dt.datetime.now()


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def log(message: str) -> None:
    print(message, flush=True)
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(f'{now_iso()} {message}\n')


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'null'}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    f = safe_float(value)
    if f is None:
        return default
    return int(f)


def fmt(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and math.isnan(value):
        return ''
    return f'{float(value):.{digits}f}'.rstrip('0').rstrip('.')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'path': str(path), 'exists': False}
    st = path.stat()
    return {'path': str(path), 'exists': True, 'size_bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'sha256': sha256(path)}


def snapshot_tree(root: Path) -> list[str]:
    if not root.exists():
        return []
    rows = []
    for path in root.rglob('*'):
        if path.is_file():
            st = path.stat()
            rows.append(f'{path.relative_to(root)}\t{st.st_size}\t{st.st_mtime_ns}')
    return sorted(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_csv_with_columns(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, '') for col in columns})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(lines)


def backup_existing_outputs() -> tuple[Path, list[str]]:
    backup_dir = PROJECT_ROOT / 'backups' / f'scene_gt_eval_video_id_5_v2_5_{datetime_now().strftime("%Y%m%d_%H%M%S")}'
    targets = [
        GT_SECONDS_PATH, GT_VALIDATION_PATH, CANDIDATE_FAMILIES_PATH, FINAL_ANCHOR_VIDEO5_PATH, MATCHES_PATH,
        UNMATCHED_GT_PATH, FALSE_POSITIVE_PATH, PERFORMANCE_SUMMARY_PATH, SOURCE_DIAGNOSTIC_PATH,
        SUMMARY_MD_PATH, REPORT_JSON_PATH, FINDINGS_MD_PATH, RUN_LOG_PATH,
    ]
    copied = []
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in targets:
        if path.exists():
            dst = backup_dir / rel(path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied.append(rel(path))
    for directory in [LATEST_BUNDLE, SHARED_DIR]:
        if directory.exists() and any(directory.iterdir()):
            dst = backup_dir / rel(directory)
            shutil.copytree(directory, dst, dirs_exist_ok=True)
            copied.append(rel(directory))
    return backup_dir, copied


def normalize_header(value: Any) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower()).strip('_')


def parse_boundary_time(value: Any) -> tuple[float | None, str, str]:
    if value is None:
        return None, 'invalid', 'empty boundary value'
    if isinstance(value, dt.datetime):
        sec = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000
        return sec, 'parsed', 'datetime interpreted as time-of-day seconds'
    if isinstance(value, dt.time):
        sec = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1_000_000
        return sec, 'parsed', 'time object interpreted as seconds from 00:00'
    if isinstance(value, dt.timedelta):
        return value.total_seconds(), 'parsed', 'timedelta interpreted as seconds'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        if math.isnan(f):
            return None, 'invalid', 'numeric NaN'
        if f < 0:
            return None, 'invalid', 'negative numeric boundary'
        if 0 < f < 1:
            return f * 86400.0, 'parsed', 'numeric value below 1 interpreted as Excel day fraction'
        return f, 'parsed', 'numeric value interpreted as seconds'
    text = str(value).strip()
    if not text:
        return None, 'invalid', 'empty boundary string'
    lowered = text.lower().replace(' ', '')
    m = re.fullmatch(r'(\d+(?:\.\d+)?)m(\d+(?:\.\d+)?)s?', lowered)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2)), 'parsed', ''
    m = re.fullmatch(r'(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)', lowered)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2)), 'parsed', 'mm:ss interpreted as minute:second'
    m = re.fullmatch(r'(\d+(?:\.\d+)?):(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)', lowered)
    if m:
        return float(m.group(1)) * 3600 + float(m.group(2)) * 60 + float(m.group(3)), 'parsed', 'hh:mm:ss interpreted as seconds'
    if re.fullmatch(r'\d+(?:\.\d+)?', lowered):
        f = float(lowered)
        if f < 0:
            return None, 'invalid', 'negative boundary string'
        return f, 'parsed', 'numeric string interpreted as seconds'
    return None, 'invalid', f'unrecognized boundary format: {text}'


def load_gt_xlsx(duration_sec: float | None, warnings: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if openpyxl is None:
        raise RuntimeError(f'openpyxl import failed: {OPENPYXL_ERROR}')
    if not GT_XLSX.exists():
        raise RuntimeError(f'GT xlsx not found: {GT_XLSX}')
    wb = openpyxl.load_workbook(GT_XLSX, data_only=True, read_only=True)
    ws = wb.worksheets[0]
    raw_rows = list(ws.iter_rows(values_only=True))
    if not raw_rows:
        raise RuntimeError('GT xlsx is empty')
    headers = [normalize_header(v) for v in raw_rows[0]]
    header_to_index = {h: idx for idx, h in enumerate(headers) if h}
    video_col = next((c for c in ['video_id', 'videoid', 'id'] if c in header_to_index), None)
    gt_id_col = next((c for c in ['scene_gt_id', 'gt_id', 'scene_id', 'id'] if c in header_to_index), None)
    boundary_candidates = ['boundary_mmss', 'boundary_min_sec', 'boundary_time', 'boundary', 'time_mmss', 'mmss', 'boundary_sec']
    boundary_col = next((c for c in boundary_candidates if c in header_to_index), None)
    if boundary_col is None:
        raise RuntimeError(f'Cannot find boundary column in GT xlsx headers: {headers}')
    if video_col is None:
        warnings.append('video_id column missing; all GT rows assumed video_id=5')
    if gt_id_col is None:
        warnings.append('scene_gt_id column missing; generated from row index')

    parsed_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    non_video5 = 0
    assumed_video_id = 0
    for excel_row_index, row_values in enumerate(raw_rows[1:], start=2):
        row = list(row_values)
        def get(col: str | None) -> Any:
            if col is None:
                return None
            idx = header_to_index[col]
            return row[idx] if idx < len(row) else None
        raw_video = get(video_col)
        vid = safe_int(raw_video)
        if vid is None:
            vid = VIDEO_ID
            assumed_video_id += 1
        if vid != VIDEO_ID:
            non_video5 += 1
            invalid_rows.append({
                'video_id': vid,
                'scene_gt_id': get(gt_id_col) or f'ROW{excel_row_index}',
                **SPLIT_COLUMNS,
                'boundary_mmss_original': get(boundary_col),
                'boundary_sec': '',
                'parse_status': 'excluded',
                'parse_warning': f'non-video5 row excluded: video_id={vid}',
                'source_file': str(GT_XLSX),
                'row_index': excel_row_index,
                'notes': 'Only video_id=5 is evaluated.',
            })
            continue
        boundary_raw = get(boundary_col)
        boundary_sec, status, warning = parse_boundary_time(boundary_raw)
        scene_gt_id = get(gt_id_col) or f'S{excel_row_index - 1:04d}'
        if boundary_sec is None or boundary_sec < 0:
            invalid_rows.append({
                'video_id': vid,
                'scene_gt_id': scene_gt_id,
                **SPLIT_COLUMNS,
                'boundary_mmss_original': boundary_raw,
                'boundary_sec': '',
                'parse_status': 'invalid',
                'parse_warning': warning,
                'source_file': str(GT_XLSX),
                'row_index': excel_row_index,
                'notes': 'Invalid GT row excluded from evaluation.',
            })
            continue
        if duration_sec is not None and boundary_sec > duration_sec + 1e-6:
            invalid_rows.append({
                'video_id': vid,
                'scene_gt_id': scene_gt_id,
                **SPLIT_COLUMNS,
                'boundary_mmss_original': boundary_raw,
                'boundary_sec': fmt(boundary_sec),
                'parse_status': 'invalid',
                'parse_warning': f'boundary_sec exceeds video duration {duration_sec}',
                'source_file': str(GT_XLSX),
                'row_index': excel_row_index,
                'notes': 'Out-of-duration GT row excluded from evaluation.',
            })
            continue
        parsed_rows.append({
            'video_id': vid,
            'scene_gt_id': scene_gt_id,
            **SPLIT_COLUMNS,
            'boundary_mmss_original': boundary_raw,
            'boundary_sec': fmt(boundary_sec),
            'parse_status': status,
            'parse_warning': warning,
            'source_file': str(GT_XLSX),
            'row_index': excel_row_index,
            'notes': 'Used for video_id=5 single-video diagnostic scene GT evaluation.',
        })
    parsed_rows.sort(key=lambda r: (safe_float(r['boundary_sec'], 0.0), str(r['scene_gt_id'])))
    metadata = {
        'sheet_name': ws.title,
        'headers': headers,
        'video_column': video_col,
        'scene_gt_id_column': gt_id_col,
        'boundary_column': boundary_col,
        'assumed_video_id_rows': assumed_video_id,
        'non_video5_rows_excluded': non_video5,
        'xlsx_total_data_rows': max(len(raw_rows) - 1, 0),
    }
    return parsed_rows, invalid_rows, metadata


def duplicate_gt_examples(gt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    secs = [(idx, row, safe_float(row['boundary_sec'], 0.0)) for idx, row in enumerate(gt_rows)]
    for i in range(len(secs)):
        idx_i, row_i, sec_i = secs[i]
        for j in range(i + 1, len(secs)):
            idx_j, row_j, sec_j = secs[j]
            if abs(sec_i - sec_j) <= DUPLICATE_GT_WINDOW_SEC:
                examples.append({
                    'scene_gt_id_a': row_i['scene_gt_id'],
                    'scene_gt_id_b': row_j['scene_gt_id'],
                    'boundary_sec_a': sec_i,
                    'boundary_sec_b': sec_j,
                    'distance_sec': abs(sec_i - sec_j),
                })
            elif sec_j - sec_i > DUPLICATE_GT_WINDOW_SEC:
                break
    return examples


def build_gt_validation(gt_rows: list[dict[str, Any]], invalid_rows: list[dict[str, Any]], gt_meta: dict[str, Any], duration_sec: float | None) -> list[dict[str, Any]]:
    duplicate_examples = duplicate_gt_examples(gt_rows)
    secs = [safe_float(row['boundary_sec'], 0.0) for row in gt_rows]
    monotonic = all(secs[i] <= secs[i + 1] for i in range(len(secs) - 1))
    duration_ok = True if duration_sec is None else all(0 <= sec <= duration_sec for sec in secs)
    return [
        {'validation_item': 'input_file_exists', 'status': str(GT_XLSX.exists()).lower(), 'count': 1 if GT_XLSX.exists() else 0, 'examples_json': '[]', 'notes': str(GT_XLSX)},
        {'validation_item': 'required_columns_found', 'status': str(bool(gt_meta.get('boundary_column'))).lower(), 'count': 1 if gt_meta.get('boundary_column') else 0, 'examples_json': json_dumps(gt_meta), 'notes': 'Flexible boundary column detection applied.'},
        {'validation_item': 'parsed_gt_rows', 'status': 'ok' if gt_rows else 'failed', 'count': len(gt_rows), 'examples_json': json_dumps(gt_rows[:5]), 'notes': 'Rows used for evaluation.'},
        {'validation_item': 'invalid_gt_rows', 'status': 'ok' if not invalid_rows else 'warning', 'count': len(invalid_rows), 'examples_json': json_dumps(invalid_rows[:10]), 'notes': 'Invalid/excluded rows are not evaluated.'},
        {'validation_item': 'duplicate_or_near_duplicate_gt_rows', 'status': 'warning' if duplicate_examples else 'ok', 'count': len(duplicate_examples), 'examples_json': json_dumps(duplicate_examples[:20]), 'notes': f'Near duplicate window <= {DUPLICATE_GT_WINDOW_SEC}s; original rows preserved.'},
        {'validation_item': 'non_video5_rows_excluded', 'status': 'warning' if gt_meta.get('non_video5_rows_excluded') else 'ok', 'count': gt_meta.get('non_video5_rows_excluded', 0), 'examples_json': '[]', 'notes': 'Only video_id=5 is evaluated.'},
        {'validation_item': 'boundary_sec_monotonic_check', 'status': 'ok' if monotonic else 'warning', 'count': len(secs), 'examples_json': json_dumps(secs[:10]), 'notes': 'GT rows sorted by boundary_sec in output.'},
        {'validation_item': 'duration_range_check', 'status': 'ok' if duration_ok else 'warning', 'count': len(secs), 'examples_json': json_dumps({'duration_sec': duration_sec, 'min_gt': min(secs) if secs else None, 'max_gt': max(secs) if secs else None}), 'notes': 'Rows outside duration are excluded when duration is available.'},
    ]


def load_video5_metadata() -> dict[str, Any]:
    manifest_rows = read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else []
    split_rows = read_csv(SPLIT_PATH) if SPLIT_PATH.exists() else []
    manifest = next((row for row in manifest_rows if safe_int(row.get('video_id')) == VIDEO_ID), {})
    split = next((row for row in split_rows if safe_int(row.get('video_id')) == VIDEO_ID), {})
    return {
        'video_id': VIDEO_ID,
        'duration_sec': safe_float(manifest.get('duration_sec'), safe_float(split.get('video_duration_sec'))),
        'fps': safe_float(manifest.get('fps')),
        'frame_count': safe_int(manifest.get('frame_count')),
        'video_path': manifest.get('video_path') or split.get('video_path', ''),
        'original_split_v2_4': split.get('split', ''),
        'split_seed': safe_int(split.get('split_seed')),
        'video_title': manifest.get('video_title') or split.get('video_name', ''),
    }


def frame_for_sec(sec: float | None, fps: float | None) -> str:
    if sec is None or fps is None:
        return ''
    return str(int(round(sec * fps)))


def candidate_row(family: str, sec: float, fps: float | None, source_model: str, source_relation: str, score: float | None = None, threshold: float | None = None, dedup_window: float | None = None, cluster_id: str = '', cluster_member_count: int = 1, members: list[dict[str, Any]] | None = None, notes: str = '') -> dict[str, Any]:
    return {
        'candidate_family': family,
        'video_id': VIDEO_ID,
        **SPLIT_COLUMNS,
        'candidate_sec': fmt(sec),
        'candidate_frame': frame_for_sec(sec, fps),
        'source_model': source_model,
        'source_relation': source_relation,
        'score': fmt(score, 9),
        'threshold': fmt(threshold, 6),
        'dedup_window_sec': fmt(dedup_window, 6),
        'cluster_id': cluster_id,
        'cluster_member_count': cluster_member_count,
        'source_members_json': json_dumps(members or []),
        'notes': notes,
    }


def load_candidate_families(video_meta: dict[str, Any], warnings: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fps = safe_float(video_meta.get('fps'))
    rows_out: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    if OPENCV_PATH.exists():
        rows, cols = read_csv_with_columns(OPENCV_PATH)
        count = 0
        for idx, row in enumerate(rows):
            if safe_int(row.get('video_id')) != VIDEO_ID:
                continue
            sec = safe_float(row.get('candidate_time_sec'))
            if sec is None or sec < 0:
                continue
            score = safe_float(row.get('scene_change_score'))
            rows_out.append(candidate_row(
                'opencv_ffmpeg', sec, fps, row.get('candidate_source') or row.get('method_used') or 'opencv_ffmpeg',
                'single_source', score=score, threshold=safe_float(row.get('threshold')), cluster_id=row.get('merged_candidate_id', f'opencv_{idx}'),
                members=[{'source': 'opencv_ffmpeg', 'candidate_sec': sec, 'score': score, 'raw_row_index': idx}],
                notes='OpenCV/FFmpeg source candidate for video_id=5 diagnostic scene GT evaluation.',
            ))
            count += 1
        stats['opencv_ffmpeg'] = {'path': str(OPENCV_PATH), 'columns': cols, 'candidate_count': count}
    else:
        warnings.append(f'Missing OpenCV/FFmpeg candidate file: {OPENCV_PATH}')
        stats['opencv_ffmpeg'] = {'path': str(OPENCV_PATH), 'candidate_count': 0, 'missing': True}

    if RESNET_PATH.exists():
        rows, cols = read_csv_with_columns(RESNET_PATH)
        count = 0
        for idx, row in enumerate(rows):
            if safe_int(row.get('video_id')) != VIDEO_ID:
                continue
            sec = safe_float(row.get('candidate_time_sec'))
            if sec is None or sec < 0:
                continue
            score = safe_float(row.get('scene_change_score'), safe_float(row.get('cosine_distance')))
            rows_out.append(candidate_row(
                'resnet', sec, fps, row.get('model_name') or row.get('candidate_source') or 'resnet_embedding',
                'single_source', score=score, threshold=safe_float(row.get('threshold')), cluster_id=f"resnet_rank_{row.get('score_rank_in_video', idx)}",
                members=[{'source': 'resnet', 'candidate_sec': sec, 'score': score, 'raw_row_index': idx}],
                notes='ResNet embedding source candidate for video_id=5 diagnostic scene GT evaluation.',
            ))
            count += 1
        stats['resnet'] = {'path': str(RESNET_PATH), 'columns': cols, 'candidate_count': count}
    else:
        warnings.append(f'Missing ResNet candidate file: {RESNET_PATH}')
        stats['resnet'] = {'path': str(RESNET_PATH), 'candidate_count': 0, 'missing': True}

    if TRANSNET_SWEEP_PATH.exists():
        rows, cols = read_csv_with_columns(TRANSNET_SWEEP_PATH)
        count = 0
        for idx, row in enumerate(rows):
            if safe_int(row.get('video_id')) != VIDEO_ID:
                continue
            family = row.get('sweep_family', '')
            threshold = safe_float(row.get('threshold'))
            dedup = safe_float(row.get('dedup_window_sec'))
            if family != TRANSNET_FAMILY or threshold != TRANSNET_THRESHOLD or dedup != TRANSNET_DEDUP_SEC:
                continue
            sec = safe_float(row.get('candidate_sec'))
            if sec is None or sec < 0:
                continue
            score = safe_float(row.get('transnetv2_score'))
            rows_out.append(candidate_row(
                'transnetv2_conservative', sec, fps, family, 'single_source', score=score, threshold=threshold,
                dedup_window=dedup, cluster_id=row.get('cluster_id', f'transnet_{idx}'),
                cluster_member_count=safe_int(row.get('cluster_member_count'), 1) or 1,
                members=[{'source': 'transnetv2_conservative', 'candidate_sec': sec, 'score': score, 'raw_row_index': idx, 'cluster_id': row.get('cluster_id')}],
                notes='TransNetV2 conservative candidate: threshold=0.7, dedup=5s.',
            ))
            count += 1
        stats['transnetv2_conservative'] = {'path': str(TRANSNET_SWEEP_PATH), 'columns': cols, 'candidate_count': count, 'threshold': TRANSNET_THRESHOLD, 'dedup_window_sec': TRANSNET_DEDUP_SEC, 'sweep_family': TRANSNET_FAMILY}
    else:
        warnings.append(f'Missing TransNetV2 sweep file: {TRANSNET_SWEEP_PATH}')
        stats['transnetv2_conservative'] = {'path': str(TRANSNET_SWEEP_PATH), 'candidate_count': 0, 'missing': True}

    # 선택적으로 canonical OpenCV/ResNet 기준을 함께 기록한다.
    canonical_path = CANONICAL_PATH if CANONICAL_PATH.exists() else CANONICAL_FALLBACK_PATH
    if canonical_path.exists():
        rows, cols = read_csv_with_columns(canonical_path)
        count = 0
        for idx, row in enumerate(rows):
            if safe_int(row.get('video_id')) != VIDEO_ID:
                continue
            if row.get('split') not in {'', 'train'} and 'split' in row:
                continue
            sec = safe_float(row.get('canonical_boundary_time_sec'))
            if sec is None or sec < 0:
                continue
            score = safe_float(row.get('visual_boundary_strength_score'))
            rows_out.append(candidate_row(
                'opencv_resnet_canonical', sec, fps, row.get('canonical_time_source') or 'canonical_opencv_resnet',
                row.get('source_relation') or 'canonical', score=score, cluster_id=row.get('scene_boundary_anchor_id', f'canonical_{idx}'),
                cluster_member_count=safe_int(row.get('source_count'), 1) or 1,
                members=[{'source': 'opencv_resnet_canonical', 'candidate_sec': sec, 'score': score, 'scene_boundary_anchor_id': row.get('scene_boundary_anchor_id'), 'source_relation': row.get('source_relation')}],
                notes='Optional existing OpenCV/ResNet canonical anchor reference.',
            ))
            count += 1
        stats['opencv_resnet_canonical'] = {'path': str(canonical_path), 'columns': cols, 'candidate_count': count, 'optional': True}
    return rows_out, stats


def load_or_build_final_anchor(candidate_rows: list[dict[str, Any]], video_meta: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
    if FINAL_DEV_ANCHOR_PATH.exists():
        rows = read_csv(FINAL_DEV_ANCHOR_PATH)
        filtered = [row for row in rows if safe_int(row.get('video_id')) == VIDEO_ID]
        if filtered:
            out = []
            for idx, row in enumerate(filtered, start=1):
                out.append({
                    'final_anchor_id': row.get('final_anchor_id') or f'F3M_V05_{idx:05d}',
                    'video_id': VIDEO_ID,
                    **SPLIT_COLUMNS,
                    'anchor_sec': row.get('anchor_sec', ''),
                    'anchor_frame': row.get('anchor_frame', ''),
                    'cluster_min_sec': row.get('cluster_min_sec', ''),
                    'cluster_max_sec': row.get('cluster_max_sec', ''),
                    'cluster_member_count': row.get('cluster_member_count', ''),
                    'source_relation': row.get('source_relation', ''),
                    'has_opencv_ffmpeg': row.get('has_opencv_ffmpeg', ''),
                    'has_resnet': row.get('has_resnet', ''),
                    'has_transnetv2_conservative': row.get('has_transnetv2_conservative', ''),
                    'opencv_member_count': row.get('opencv_member_count', ''),
                    'resnet_member_count': row.get('resnet_member_count', ''),
                    'transnetv2_member_count': row.get('transnetv2_member_count', ''),
                    'source_members_json': row.get('source_members_json', '[]'),
                    'representative_rule': row.get('representative_rule', 'existing_final_anchor_reused'),
                    'confidence_or_score': row.get('confidence_or_score', ''),
                    'notes': 'Reused existing Development Set final scene anchor for video_id=5; copied to diagnostic output only.',
                })
            return out
        warnings.append(f'Existing final anchor file found but no video_id=5 rows: {FINAL_DEV_ANCHOR_PATH}')

    warnings.append('Existing final Development anchor unavailable; rebuilding video_id=5 final anchor from source candidates.')
    fps = safe_float(video_meta.get('fps'))
    source_rows = [row for row in candidate_rows if row['candidate_family'] in {'opencv_ffmpeg', 'resnet', 'transnetv2_conservative'}]
    source_rows = sorted(source_rows, key=lambda row: (safe_float(row['candidate_sec'], 0.0), row['candidate_family']))
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_max: float | None = None
    for row in source_rows:
        sec = safe_float(row['candidate_sec'], 0.0) or 0.0
        if not current:
            current = [row]
            current_max = sec
        elif current_max is not None and sec - current_max <= FINAL_CLUSTER_WINDOW_SEC:
            current.append(row)
            current_max = max(current_max, sec)
        else:
            clusters.append(current)
            current = [row]
            current_max = sec
    if current:
        clusters.append(current)

    final_rows = []
    for idx, cluster in enumerate(clusters, start=1):
        secs = [safe_float(row['candidate_sec'], 0.0) or 0.0 for row in cluster]
        sources = {row['candidate_family'] for row in cluster}
        source_counts = Counter(row['candidate_family'] for row in cluster)
        scored = [row for row in cluster if safe_float(row.get('score')) is not None]
        if scored:
            rep = sorted(scored, key=lambda row: (-(safe_float(row.get('score')) or 0.0), safe_float(row['candidate_sec'], 0.0) or 0.0))[0]
            anchor_sec = safe_float(rep['candidate_sec'], 0.0) or 0.0
            representative_rule = 'max_source_score_then_earliest'
            confidence = safe_float(rep.get('score'))
        else:
            rep = sorted(cluster, key=lambda row: safe_float(row['candidate_sec'], 0.0) or 0.0)[0]
            anchor_sec = safe_float(rep['candidate_sec'], 0.0) or 0.0
            representative_rule = 'earliest_timestamp_no_score_available'
            confidence = None
        relation = 'only_' + next(iter(sources)) if len(sources) == 1 else 'multi_source:' + '+'.join(sorted(sources))
        members = []
        for row in cluster:
            members.append({'source': row['candidate_family'], 'candidate_sec': row['candidate_sec'], 'score': row.get('score'), 'cluster_id': row.get('cluster_id')})
        final_rows.append({
            'final_anchor_id': f'F3M_V05_{idx:05d}',
            'video_id': VIDEO_ID,
            **SPLIT_COLUMNS,
            'anchor_sec': fmt(anchor_sec),
            'anchor_frame': frame_for_sec(anchor_sec, fps),
            'cluster_min_sec': fmt(min(secs)),
            'cluster_max_sec': fmt(max(secs)),
            'cluster_member_count': len(cluster),
            'source_relation': relation,
            'has_opencv_ffmpeg': str('opencv_ffmpeg' in sources).lower(),
            'has_resnet': str('resnet' in sources).lower(),
            'has_transnetv2_conservative': str('transnetv2_conservative' in sources).lower(),
            'opencv_member_count': source_counts.get('opencv_ffmpeg', 0),
            'resnet_member_count': source_counts.get('resnet', 0),
            'transnetv2_member_count': source_counts.get('transnetv2_conservative', 0),
            'source_members_json': json_dumps(members),
            'representative_rule': representative_rule,
            'confidence_or_score': fmt(confidence, 9),
            'notes': 'Rebuilt final three-model diagnostic anchor; no average timestamp used.',
        })
    return final_rows


def append_final_anchor_candidates(candidate_rows: list[dict[str, Any]], final_anchor_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(candidate_rows)
    for row in final_anchor_rows:
        out.append({
            'candidate_family': 'final_three_model_anchor',
            'video_id': VIDEO_ID,
            **SPLIT_COLUMNS,
            'candidate_sec': row.get('anchor_sec', ''),
            'candidate_frame': row.get('anchor_frame', ''),
            'source_model': 'opencv_ffmpeg+resnet+transnetv2_conservative',
            'source_relation': row.get('source_relation', ''),
            'score': row.get('confidence_or_score', ''),
            'threshold': '',
            'dedup_window_sec': fmt(FINAL_CLUSTER_WINDOW_SEC),
            'cluster_id': row.get('final_anchor_id', ''),
            'cluster_member_count': row.get('cluster_member_count', ''),
            'source_members_json': row.get('source_members_json', '[]'),
            'notes': 'Final three-model cluster representative timestamp used for scene GT matching.',
        })
    return out


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def nearest_value(value: float, candidates: list[float]) -> tuple[float | None, float | None]:
    if not candidates:
        return None, None
    nearest = min(candidates, key=lambda x: abs(x - value))
    return nearest, abs(nearest - value)


def evaluate_family(gt_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]], family: str, tolerance: float, duration_sec: float | None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    gt_items = [{'index': i, 'scene_gt_id': row['scene_gt_id'], 'sec': safe_float(row['boundary_sec'], 0.0) or 0.0, 'mmss': row.get('boundary_mmss_original', '')} for i, row in enumerate(gt_rows)]
    cand_items = [{'index': i, 'sec': safe_float(row['candidate_sec'], 0.0) or 0.0, 'relation': row.get('source_relation', ''), 'cluster_id': row.get('cluster_id', '')} for i, row in enumerate(cand_rows)]
    pairs = []
    for gt in gt_items:
        for cand in cand_items:
            dist = abs(cand['sec'] - gt['sec'])
            if dist <= tolerance + 1e-12:
                pairs.append((dist, gt['index'], cand['index'], gt, cand))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_gt: set[int] = set()
    matched_cand: set[int] = set()
    matches = []
    for dist, gt_idx, cand_idx, gt, cand in pairs:
        if gt_idx in matched_gt or cand_idx in matched_cand:
            continue
        matched_gt.add(gt_idx)
        matched_cand.add(cand_idx)
        matches.append({
            'tolerance_sec': fmt(tolerance, 3),
            'candidate_family': family,
            'video_id': VIDEO_ID,
            'scene_gt_id': gt['scene_gt_id'],
            'gt_boundary_sec': fmt(gt['sec']),
            'gt_boundary_mmss_original': gt['mmss'],
            'matched_candidate_sec': fmt(cand['sec']),
            'distance_sec': fmt(dist, 6),
            'match_status': 'tp_matched',
            'candidate_source_relation': cand['relation'],
            'candidate_cluster_id': cand['cluster_id'],
            'matching_policy': MATCHING_POLICY,
            'notes': 'One-to-one greedy nearest-distance scene transition GT match.',
        })
    candidate_secs = [item['sec'] for item in cand_items]
    gt_secs = [item['sec'] for item in gt_items]
    unmatched_gt = []
    for gt in gt_items:
        if gt['index'] in matched_gt:
            continue
        nearest, dist = nearest_value(gt['sec'], candidate_secs)
        unmatched_gt.append({
            'tolerance_sec': fmt(tolerance, 3),
            'candidate_family': family,
            'video_id': VIDEO_ID,
            'scene_gt_id': gt['scene_gt_id'],
            'gt_boundary_sec': fmt(gt['sec']),
            'gt_boundary_mmss_original': gt['mmss'],
            'nearest_candidate_sec': fmt(nearest),
            'nearest_candidate_distance_sec': fmt(dist, 6),
            'false_negative': 'true',
            'notes': 'GT boundary has no one-to-one matched candidate within tolerance.',
        })
    false_positive = []
    for cand in cand_items:
        if cand['index'] in matched_cand:
            continue
        nearest, dist = nearest_value(cand['sec'], gt_secs)
        false_positive.append({
            'tolerance_sec': fmt(tolerance, 3),
            'candidate_family': family,
            'video_id': VIDEO_ID,
            'candidate_sec': fmt(cand['sec']),
            'candidate_source_relation': cand['relation'],
            'candidate_cluster_id': cand['cluster_id'],
            'nearest_gt_sec': fmt(nearest),
            'nearest_gt_distance_sec': fmt(dist, 6),
            'false_positive': 'true',
            'notes': 'Candidate has no one-to-one matched GT boundary within tolerance.',
        })
    tp = len(matches)
    fp = len(false_positive)
    fn = len(unmatched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / len(gt_items) if gt_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    duration_min = duration_sec / 60 if duration_sec else None
    distances = [safe_float(row['distance_sec'], 0.0) or 0.0 for row in matches]
    summary = {
        'candidate_family': family,
        'tolerance_sec': fmt(tolerance, 3),
        'gt_count': len(gt_items),
        'candidate_count': len(cand_items),
        'true_positive_count': tp,
        'false_positive_count': fp,
        'false_negative_count': fn,
        'precision': fmt(precision, 9),
        'recall': fmt(recall, 9),
        'f1': fmt(f1, 9),
        'candidates_per_minute': fmt(len(cand_items) / duration_min, 9) if duration_min else '',
        'false_positives_per_minute': fmt(fp / duration_min, 9) if duration_min else '',
        'median_match_distance_sec': fmt(percentile(distances, 0.5), 9),
        'mean_match_distance_sec': fmt(sum(distances) / len(distances), 9) if distances else '',
        'p90_match_distance_sec': fmt(percentile(distances, 0.9), 9),
        'max_match_distance_sec': fmt(max(distances), 9) if distances else '',
        'matching_policy': MATCHING_POLICY,
        'notes': f'single-video diagnostic scene GT evaluation; matched_gt_ratio={fmt(recall, 9)}; unmatched_gt_count={fn}; unmatched_candidate_count={fp}',
    }
    return summary, matches, unmatched_gt, false_positive


def evaluate_all(gt_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], duration_sec: float | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_family[row['candidate_family']].append(row)
    ordered_families = [f for f in ['opencv_ffmpeg', 'resnet', 'transnetv2_conservative', 'final_three_model_anchor', 'opencv_resnet_canonical', 'transnetv2_primary'] if f in by_family]
    summaries = []
    all_matches = []
    all_unmatched = []
    all_fp = []
    for family in ordered_families:
        family_rows = sorted(by_family[family], key=lambda row: safe_float(row.get('candidate_sec'), 0.0) or 0.0)
        for tol in TOLERANCES:
            summary, matches, unmatched, fps = evaluate_family(gt_rows, family_rows, family, tol, duration_sec)
            summaries.append(summary)
            all_matches.extend(matches)
            all_unmatched.extend(unmatched)
            all_fp.extend(fps)
    return summaries, all_matches, all_unmatched, all_fp


def build_source_diagnostic(summary_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_tol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_tol[row['tolerance_sec']].append(row)
    for tol, rows in sorted(by_tol.items(), key=lambda item: safe_float(item[0], 0.0) or 0.0):
        for metric, column, interpretation in [
            ('best_precision_family', 'precision', 'Highest precision within this single-video diagnostic tolerance.'),
            ('best_recall_family', 'recall', 'Highest recall within this single-video diagnostic tolerance.'),
            ('best_f1_family', 'f1', 'Highest F1 within this single-video diagnostic tolerance.'),
        ]:
            best = max(rows, key=lambda r: safe_float(r[column], -1.0) or -1.0)
            out.append({'diagnostic_metric': metric, 'candidate_family': best['candidate_family'], 'tolerance_sec': tol, 'metric_value': best[column], 'interpretation': interpretation, 'notes': 'Do not generalize beyond video_id=5.'})
        final_row = next((row for row in rows if row['candidate_family'] == 'final_three_model_anchor'), None)
        if final_row:
            for metric, col in [('final_anchor_precision', 'precision'), ('final_anchor_recall', 'recall'), ('final_anchor_f1', 'f1'), ('final_anchor_false_positives_per_minute', 'false_positives_per_minute'), ('final_anchor_candidate_count', 'candidate_count')]:
                out.append({'diagnostic_metric': metric, 'candidate_family': 'final_three_model_anchor', 'tolerance_sec': tol, 'metric_value': final_row[col], 'interpretation': f'Final three-model anchor {col} at tolerance {tol}s.', 'notes': 'single-video diagnostic scene GT only'})
    counts = Counter(row['candidate_family'] for row in candidate_rows)
    for family, metric in [('opencv_ffmpeg', 'opencv_candidate_count'), ('resnet', 'resnet_candidate_count'), ('transnetv2_conservative', 'transnetv2_conservative_candidate_count')]:
        out.append({'diagnostic_metric': metric, 'candidate_family': family, 'tolerance_sec': 'all', 'metric_value': counts.get(family, 0), 'interpretation': f'{family} candidate count for video_id=5.', 'notes': 'Count before scene GT matching.'})
    out.append({'diagnostic_metric': 'single_video_diagnostic_warning', 'candidate_family': 'all', 'tolerance_sec': 'all', 'metric_value': 'video_id=5 only', 'interpretation': 'This is a single-video diagnostic GT evaluation, not dataset-level scene transition performance.', 'notes': 'Use for method diagnosis and presentation nuance only.'})
    return out


def best_by_tolerance(summary_rows: list[dict[str, Any]], column: str) -> dict[str, str]:
    out: dict[str, str] = {}
    by_tol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_tol[row['tolerance_sec']].append(row)
    for tol, rows in by_tol.items():
        best = max(rows, key=lambda r: safe_float(r[column], -1.0) or -1.0)
        out[tol] = best['candidate_family']
    return out


def make_reports(report: dict[str, Any], summary_rows: list[dict[str, Any]], unmatched_rows: list[dict[str, Any]], fp_rows: list[dict[str, Any]]) -> None:
    # summary.md에 넣을 간단한 성능 표를 만든다.
    rows_for_table = []
    for row in summary_rows:
        if row['candidate_family'] in {'opencv_ffmpeg', 'resnet', 'transnetv2_conservative', 'final_three_model_anchor'}:
            rows_for_table.append([row['candidate_family'], row['tolerance_sec'], row['candidate_count'], row['precision'], row['recall'], row['f1'], row['false_positives_per_minute']])
    table = markdown_table(['family', 'tol_s', 'candidates', 'precision', 'recall', 'f1', 'fp/min'], rows_for_table)
    final_rows = [row for row in summary_rows if row['candidate_family'] == 'final_three_model_anchor']
    final_table = markdown_table(['tol_s', 'precision', 'recall', 'f1', 'FP', 'FN', 'FP/min'], [[r['tolerance_sec'], r['precision'], r['recall'], r['f1'], r['false_positive_count'], r['false_negative_count'], r['false_positives_per_minute']] for r in final_rows])
    unmatched_examples = [row for row in unmatched_rows if row['candidate_family'] == 'final_three_model_anchor' and row['tolerance_sec'] == '1'][:10]
    fp_examples = [row for row in fp_rows if row['candidate_family'] == 'final_three_model_anchor' and row['tolerance_sec'] == '1'][:10]
    summary_md = f"""# Scene Transition GT Evaluation: video_id=5 Diagnostic v2.5

## 작업 목적

사용자가 직접 라벨링한 `video_id=5` 전체 화면전환 GT의 `boundary_mmss`를 초 단위로 변환하고, OpenCV/FFmpeg + ResNet + TransNetV2 conservative를 결합한 final three-model anchor가 실제 화면전환을 얼마나 포착하는지 평가했다.

## v2.5 split terminology

{REQUIRED_SPLIT_EXPLANATION}

이번 평가는 `video_id=5` 한 편에 대한 **Development Set 내부 single-video diagnostic scene GT evaluation**이다. Test Set은 처리하지 않았다.

## GT parsing

- 입력 GT: `{GT_XLSX}`
- 변환 CSV: `{GT_SECONDS_PATH}`
- parsed GT rows: `{report['gt_count']}`
- invalid/excluded GT rows: `{report['invalid_gt_row_count']}`

`11m33`, `11:33`, numeric seconds, Excel time/datetime value를 초 단위로 변환하도록 처리했다.

## Candidate families

평가 family: `{', '.join(report['candidate_families'])}`

Final three-model anchor는 OpenCV/FFmpeg, ResNet, TransNetV2 conservative(`threshold=0.7`, `dedup_window=5s`)를 결합한 후보이다. Matching은 one-to-one greedy nearest distance 방식으로 수행했다.

## Final anchor result

{final_table}

## All family comparison

{table}

## Examples

### Unmatched GT examples for final anchor @1s

```json
{json.dumps(unmatched_examples, ensure_ascii=False, indent=2)}
```

### False positive candidate examples for final anchor @1s

```json
{json.dumps(fp_examples, ensure_ascii=False, indent=2)}
```

## Interpretation guard

이 precision/recall/F1은 광고 boundary 기준이 아니라 `video_id=5` 전체 scene transition GT 기준이다. 한 편의 diagnostic 결과이므로 전체 데이터셋 scene transition 성능으로 일반화하지 않는다.

## Safety

- no_detector_rule_modified=true
- no_existing_anchor_modified=true
- no_existing_split_modified=true
- no_extended_evaluation_processed=true
- no_diagnostic_subset_processed=true
- no_pure_test_processed=true
"""
    findings_md = f"""# Findings: video_id=5 Scene Transition GT Diagnostic

## GT 변환 방식

수동 라벨의 `boundary_mmss` 값은 `0m04`, `11m33`, `11:33` 같은 분/초 표현 또는 numeric seconds, Excel time object로 들어올 수 있도록 파서가 처리했다. 변환 결과는 `{GT_SECONDS_PATH}`에 `boundary_sec`로 저장했다.

## Final 3-model anchor의 화면전환 포착

Final anchor는 video_id=5에서 `{report['final_anchor_count']}`개 cluster 후보를 만들었다. 이 후보는 OpenCV/FFmpeg, ResNet, TransNetV2 conservative 세 source를 결합한 것이다. 평가 기준은 광고 boundary recall이 아니라 사용자가 라벨링한 전체 화면전환 GT이다.

## Precision/Recall/F1 해석

이번 precision/recall/F1은 **single-video diagnostic scene GT 기준**이다. 후보가 많아지면 recall은 올라갈 수 있지만 false positive도 늘 수 있다. 따라서 final anchor 결과는 recall만 보지 않고 precision, F1, false positives per minute과 함께 해석해야 한다.

## TransNetV2 conservative의 의미

TransNetV2 primary는 후보 밀도가 높아 OCR/후속 처리를 무겁게 만들 수 있었기 때문에, 이번 최종 후보에는 `threshold=0.7`, `dedup=5s` conservative 기준을 사용했다. video_id=5 scene GT에서도 이 source가 기존 OpenCV/ResNet 후보가 놓친 전환을 보완하는지 확인하는 진단 자료로 볼 수 있다.

## 일반화 한계

video_id=5 한 편만 라벨링한 결과이므로 전체 데이터셋 scene transition 성능으로 주장하면 안 된다. 발표에서는 “사용자 라벨링 single-video diagnostic GT에서 final anchor의 precision/recall/F1을 확인했다”라고 표현하는 것이 안전하다.

## 발표 문장 예시

- “이번 평가는 광고 boundary 기준이 아니라, video_id=5 한 편에 대해 직접 라벨링한 전체 화면전환 GT 기준 diagnostic 평가입니다.”
- “Final three-model anchor는 recall을 높이는 대신 후보 수가 많아질 수 있으므로 precision과 FP/min을 함께 확인했습니다.”
- “이 결과는 전체 데이터셋 일반화 성능이 아니라, 향후 subset scene GT를 더 늘릴 가치가 있는지 판단하기 위한 보조 진단입니다.”
"""
    SUMMARY_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD_PATH.write_text(summary_md.rstrip() + '\n', encoding='utf-8')
    FINDINGS_MD_PATH.write_text(findings_md.rstrip() + '\n', encoding='utf-8')


def validate_outputs(gt_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], final_anchor_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    split_cols = ['original_split_v2_4', 'split_role_v2_5', 'evaluation_subset_v2_5', 'split_terminology_note']
    gt_split_ok = all(all(col in row and row[col] == SPLIT_COLUMNS[col] for col in split_cols) for row in gt_rows)
    cand_split_ok = all(all(col in row and row[col] == SPLIT_COLUMNS[col] for col in split_cols) for row in candidate_rows)
    final_split_ok = all(all(col in row and row[col] == SPLIT_COLUMNS[col] for col in split_cols) for row in final_anchor_rows)
    families = {row['candidate_family'] for row in candidate_rows}
    tolerances = {row['tolerance_sec'] for row in summary_rows}
    required_tols = {fmt(t, 3) for t in TOLERANCES}
    final_summary = [row for row in summary_rows if row['candidate_family'] == 'final_three_model_anchor']
    latest_forbidden = []
    for directory in [LATEST_BUNDLE, SHARED_DIR, LATEST_SCENE_DIR]:
        if directory.exists():
            for path in directory.rglob('*'):
                if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
                    latest_forbidden.append(str(path))
    return {
        'gt_parsing_validation': {
            'passed': bool(gt_rows) and report['gt_count'] == len(gt_rows) and GT_XLSX.exists(),
            'details': {'gt_count': len(gt_rows), 'invalid_gt_row_count': report['invalid_gt_row_count'], 'gt_input_exists': GT_XLSX.exists()},
        },
        'split_scope_validation': {
            'passed': gt_split_ok and cand_split_ok and final_split_ok and {safe_int(row['video_id']) for row in gt_rows} == {VIDEO_ID} and report['single_video_diagnostic_only'],
            'details': {'gt_split_ok': gt_split_ok, 'candidate_split_ok': cand_split_ok, 'final_split_ok': final_split_ok, 'video_id': VIDEO_ID},
        },
        'candidate_source_validation': {
            'passed': {'opencv_ffmpeg', 'resnet', 'transnetv2_conservative', 'final_three_model_anchor'}.issubset(families) and report['transnetv2_conservative_threshold'] == 0.7 and report['transnetv2_conservative_dedup_window_sec'] == 5.0 and not report['protected_files_modified'],
            'details': {'families': sorted(families), 'protected_files_modified': report['protected_files_modified']},
        },
        'matching_metric_validation': {
            'passed': required_tols.issubset(tolerances) and bool(final_summary) and all(row['matching_policy'] == MATCHING_POLICY for row in summary_rows),
            'details': {'tolerances': sorted(tolerances), 'matching_policy': MATCHING_POLICY},
        },
        'interpretation_validation': {
            'passed': report['single_video_diagnostic_only'] and report['not_dataset_level_scene_transition_claim'] and report['not_ad_boundary_recall_evaluation'],
            'details': {'single_video_diagnostic_only': report['single_video_diagnostic_only']},
        },
        'output_safety_validation': {
            'passed': not report['protected_files_modified'] and not latest_forbidden and report['no_detector_rule_modified'] and report['no_existing_anchor_modified'] and report['no_existing_split_modified'],
            'details': {'latest_forbidden_files': latest_forbidden, 'protected_files_modified': report['protected_files_modified']},
        },
    }


def copy_latest(paths: list[Path], report: dict[str, Any]) -> None:
    for directory in [LATEST_BUNDLE, SHARED_DIR, LATEST_SCENE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if path.exists():
                shutil.copy2(path, directory / path.name)
        readme_name = 'README_latest_files.md' if directory != LATEST_SCENE_DIR else 'README_scene_gt_eval_video_id_5_v2_5_latest_files.md'
        readme = directory / readme_name
        lines = [
            '# Latest Files: Scene GT Evaluation video_id=5 v2.5',
            '',
            'This bundle contains only new small diagnostic CSV/report/script/log files. It excludes raw videos, frame images, cache, model weights, package directories, OCR outputs, and non-Development row-level outputs.',
            '',
            '## Included Files',
            '',
        ]
        for path in paths:
            if path.exists():
                lines.append(f'- `{path.name}`')
        lines.extend([
            '',
            '## Scope',
            '',
            '- video_id=5 only',
            '- original_split_v2_4=train',
            '- split_role_v2_5=development',
            '- single-video diagnostic scene GT evaluation only',
            f"- no_detector_rule_modified: `{str(report.get('no_detector_rule_modified')).lower()}`",
        ])
        readme.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH.write_text('', encoding='utf-8')
    warnings: list[str] = []
    errors: list[str] = []

    log('[STEP 01] 안전 스냅샷 및 출력 경로 준비')
    backup_dir, backed_up = backup_existing_outputs()
    protected_inputs = [
        GT_XLSX, SPLIT_PATH, MANIFEST_PATH, OPENCV_PATH, RESNET_PATH, TRANSNET_SWEEP_PATH, TRANSNET_BEST_PATH,
        TRANSNET_REPORT_PATH, FINAL_DEV_ANCHOR_PATH, CANONICAL_PATH, CANONICAL_FALLBACK_PATH,
        ABLATION_REPORT_PATH, ABLATION_SUMMARY_PATH, ABLATION_DOC_PATH,
    ]
    protected_before = {str(path): file_stat(path) for path in protected_inputs if path.exists()}
    old_project_before = snapshot_tree(OLD_PROJECT_ROOT)

    log('[STEP 02] 수동 scene GT xlsx 로드')
    video_meta = load_video5_metadata()
    duration_sec = safe_float(video_meta.get('duration_sec'))
    fps = safe_float(video_meta.get('fps'))
    if video_meta.get('original_split_v2_4') != 'train':
        raise RuntimeError(f'video_id=5 is not original train split: {video_meta}')

    log('[STEP 03] boundary_mmss를 boundary_sec로 변환')
    gt_rows, invalid_gt_rows, gt_meta = load_gt_xlsx(duration_sec, warnings)
    if not gt_rows:
        raise RuntimeError('No valid parsed GT rows are available for video_id=5')

    log('[STEP 04] GT validation 및 중복/오류 점검')
    validation_rows = build_gt_validation(gt_rows, invalid_gt_rows, gt_meta, duration_sec)
    duplicate_count = next((row['count'] for row in validation_rows if row['validation_item'] == 'duplicate_or_near_duplicate_gt_rows'), 0)
    if duplicate_count:
        warnings.append(f'Near-duplicate GT boundary pairs found: {duplicate_count}; original GT rows are preserved for evaluation.')

    log('[STEP 05] v2.5 split terminology 적용')
    # SPLIT_COLUMNS를 통해 모든 신규 row에 split column을 붙인다.

    log('[STEP 06] OpenCV/FFmpeg 후보 로드')
    log('[STEP 07] ResNet 후보 로드')
    log('[STEP 08] TransNetV2 conservative 후보 로드')
    candidate_rows, source_stats = load_candidate_families(video_meta, warnings)
    source_counts = Counter(row['candidate_family'] for row in candidate_rows)
    required_sources_available = sum(1 for f in ['opencv_ffmpeg', 'resnet', 'transnetv2_conservative'] if source_counts.get(f, 0) > 0)
    if required_sources_available < 2:
        raise RuntimeError(f'Too few candidate sources available for evaluation: {dict(source_counts)}')

    log('[STEP 09] final three-model scene anchor 생성')
    final_anchor_rows = load_or_build_final_anchor(candidate_rows, video_meta, warnings)
    candidate_rows = append_final_anchor_candidates(candidate_rows, final_anchor_rows)

    log('[STEP 10] tolerance별 one-to-one scene GT matching 수행')
    log('[STEP 11] precision/recall/F1 및 FP per minute 계산')
    summary_rows, match_rows, unmatched_rows, fp_rows = evaluate_all(gt_rows, candidate_rows, duration_sec)

    log('[STEP 12] unmatched GT와 false positive candidate 분석')
    diagnostic_rows = build_source_diagnostic(summary_rows, candidate_rows)

    log('[STEP 13] CSV 산출물 생성')
    write_csv(GT_SECONDS_PATH, gt_rows + invalid_gt_rows, GT_SECONDS_COLUMNS)
    write_csv(GT_VALIDATION_PATH, validation_rows, GT_VALIDATION_COLUMNS)
    write_csv(CANDIDATE_FAMILIES_PATH, candidate_rows, CANDIDATE_COLUMNS)
    write_csv(FINAL_ANCHOR_VIDEO5_PATH, final_anchor_rows, FINAL_ANCHOR_COLUMNS)
    write_csv(MATCHES_PATH, match_rows, MATCH_COLUMNS)
    write_csv(UNMATCHED_GT_PATH, unmatched_rows, UNMATCHED_GT_COLUMNS)
    write_csv(FALSE_POSITIVE_PATH, fp_rows, FP_COLUMNS)
    write_csv(PERFORMANCE_SUMMARY_PATH, summary_rows, PERFORMANCE_COLUMNS)
    write_csv(SOURCE_DIAGNOSTIC_PATH, diagnostic_rows, SOURCE_DIAGNOSTIC_COLUMNS)

    protected_after = {str(path): file_stat(path) for path in protected_inputs if path.exists()}
    old_project_after = snapshot_tree(OLD_PROJECT_ROOT)
    protected_modified = protected_before != protected_after
    old_project_modified = old_project_before != old_project_after

    performance_summary_json = {}
    for row in summary_rows:
        performance_summary_json.setdefault(row['candidate_family'], {})[row['tolerance_sec']] = {
            'gt_count': safe_int(row['gt_count']),
            'candidate_count': safe_int(row['candidate_count']),
            'true_positive_count': safe_int(row['true_positive_count']),
            'false_positive_count': safe_int(row['false_positive_count']),
            'false_negative_count': safe_int(row['false_negative_count']),
            'precision': safe_float(row['precision']),
            'recall': safe_float(row['recall']),
            'f1': safe_float(row['f1']),
            'false_positives_per_minute': safe_float(row['false_positives_per_minute']),
            'candidates_per_minute': safe_float(row['candidates_per_minute']),
        }

    report: dict[str, Any] = {
        'task_name': TASK_NAME,
        'version': VERSION,
        'project_root': str(PROJECT_ROOT),
        'created_at': now_iso(),
        'gt_input_path': str(GT_XLSX),
        'gt_output_seconds_path': str(GT_SECONDS_PATH),
        'video_id': VIDEO_ID,
        'video_duration_sec': duration_sec,
        'video_fps': fps,
        'original_split_v2_4': 'train',
        'split_role_v2_5': 'development',
        'evaluation_subset_v2_5': 'none',
        'split_terminology_note': SPLIT_TERMINOLOGY_NOTE,
        'split_explanation_ko': REQUIRED_SPLIT_EXPLANATION,
        'gt_count': len(gt_rows),
        'invalid_gt_row_count': len(invalid_gt_rows),
        'duplicate_or_near_duplicate_gt_pair_count': int(duplicate_count or 0),
        'gt_metadata': gt_meta,
        'candidate_families': sorted(set(row['candidate_family'] for row in candidate_rows)),
        'source_candidate_counts': dict(source_counts),
        'candidate_source_stats': source_stats,
        'final_anchor_sources': ['opencv_ffmpeg', 'resnet', 'transnetv2_conservative'],
        'final_anchor_count': len(final_anchor_rows),
        'transnetv2_conservative_family': TRANSNET_FAMILY,
        'transnetv2_conservative_threshold': TRANSNET_THRESHOLD,
        'transnetv2_conservative_dedup_window_sec': TRANSNET_DEDUP_SEC,
        'tolerances_sec': TOLERANCES,
        'matching_policy': MATCHING_POLICY,
        'performance_summary': performance_summary_json,
        'best_precision_family_by_tolerance': best_by_tolerance(summary_rows, 'precision'),
        'best_recall_family_by_tolerance': best_by_tolerance(summary_rows, 'recall'),
        'best_f1_family_by_tolerance': best_by_tolerance(summary_rows, 'f1'),
        'single_video_diagnostic_only': True,
        'not_dataset_level_scene_transition_claim': True,
        'not_ad_boundary_recall_evaluation': True,
        'no_extended_evaluation_processed': True,
        'no_diagnostic_subset_processed': True,
        'no_pure_test_processed': True,
        'no_detector_rule_modified': True,
        'no_existing_anchor_modified': True,
        'no_existing_split_modified': True,
        'protected_files_modified': protected_modified,
        'old_project_modified': old_project_modified,
        'warnings': warnings,
        'errors': errors,
        'output_files': {
            'gt_seconds': str(GT_SECONDS_PATH),
            'gt_validation': str(GT_VALIDATION_PATH),
            'candidate_families': str(CANDIDATE_FAMILIES_PATH),
            'final_three_model_anchor': str(FINAL_ANCHOR_VIDEO5_PATH),
            'matches': str(MATCHES_PATH),
            'unmatched_gt': str(UNMATCHED_GT_PATH),
            'false_positive_candidates': str(FALSE_POSITIVE_PATH),
            'performance_summary': str(PERFORMANCE_SUMMARY_PATH),
            'source_diagnostic_summary': str(SOURCE_DIAGNOSTIC_PATH),
            'summary_md': str(SUMMARY_MD_PATH),
            'findings_md': str(FINDINGS_MD_PATH),
            'log': str(RUN_LOG_PATH),
        },
        'latest_bundle_path': str(LATEST_BUNDLE),
        'shared_dir': str(SHARED_DIR),
        'latest_scene_copy_dir': str(LATEST_SCENE_DIR),
        'backup_dir': str(backup_dir),
        'backed_up_files': backed_up,
    }

    log('[STEP 14] markdown/json report 및 findings 생성')
    make_reports(report, summary_rows, unmatched_rows, fp_rows)

    log('[STEP 15] Sub Agent 검증 실행')
    validations = validate_outputs(gt_rows, candidate_rows, final_anchor_rows, summary_rows, report)
    report['sub_agent_validations'] = validations
    report['status'] = 'SUCCESS' if all(v['passed'] for v in validations.values()) else 'CONDITIONAL_SUCCESS'
    write_json(REPORT_JSON_PATH, report)

    log('[STEP 16] latest bundle 및 latest_scene_gt_video5 복사')
    latest_paths = [
        GT_SECONDS_PATH, GT_VALIDATION_PATH, CANDIDATE_FAMILIES_PATH, FINAL_ANCHOR_VIDEO5_PATH,
        MATCHES_PATH, UNMATCHED_GT_PATH, FALSE_POSITIVE_PATH, PERFORMANCE_SUMMARY_PATH,
        SOURCE_DIAGNOSTIC_PATH, SUMMARY_MD_PATH, REPORT_JSON_PATH, FINDINGS_MD_PATH,
        SCRIPT_PATH, RUN_LOG_PATH,
    ]
    copy_latest(latest_paths, report)
    for directory in [LATEST_BUNDLE, SHARED_DIR, LATEST_SCENE_DIR]:
        if directory.exists():
            shutil.copy2(RUN_LOG_PATH, directory / RUN_LOG_PATH.name)

    log('[STEP 17] 최종 요약 출력')
    final_by_tol = performance_summary_json.get('final_three_model_anchor', {})
    for line in [
        f'작업 상태: {report["status"]}',
        f'GT xlsx path: {GT_XLSX}',
        f'parsed GT row count: {len(gt_rows)}',
        f'invalid GT row count: {len(invalid_gt_rows)}',
        f'평가 candidate family 목록: {sorted(set(row["candidate_family"] for row in candidate_rows))}',
        f'final three-model anchor count: {len(final_anchor_rows)}',
        f'final anchor metrics by tolerance: {json.dumps(final_by_tol, ensure_ascii=False, sort_keys=True)}',
        f'best precision family: {report["best_precision_family_by_tolerance"]}',
        f'best recall family: {report["best_recall_family_by_tolerance"]}',
        f'best F1 family: {report["best_f1_family_by_tolerance"]}',
        'single-video diagnostic evaluation only: true',
        f'latest bundle path: {LATEST_BUNDLE}',
        f'기존 detector/anchor/label/split 수정 없음: {str(not protected_modified).lower()}',
    ]:
        log(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
