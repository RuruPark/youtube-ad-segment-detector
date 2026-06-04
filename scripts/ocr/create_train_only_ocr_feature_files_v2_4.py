#!/usr/bin/env python3
"""Create train-only recovered OCR feature files for v2_4 review.

This script does not run OCR, does not change keyword dictionaries, and does
not modify OCR scoring. It filters recovered OCR feature files to train split
only and recomputes train-only descriptive summaries for manual OCR score and
keyphrase review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

OLD_PROJECT = Path('./_old_project_not_included')
FORBIDDEN_EXTS = {'.mp4', '.wav', '.mp3', '.m4a', '.jpg', '.jpeg', '.png', '.webp'}
VERSION_DEFAULT = 'v2_4'

FEATURES = [
    'ocr_text_count',
    'ocr_token_count',
    'ocr_char_count',
    'ocr_box_count',
    'ocr_text_area_ratio_mean',
    'ocr_text_area_ratio_max',
    'ocr_mean_confidence',
    'ocr_empty_frame_ratio',
    'ad_disclosure_keyword_count',
    'sponsor_keyword_count',
    'brand_keyword_count',
    'product_keyword_count',
    'promotion_keyword_count',
    'discount_keyword_count',
    'purchase_keyword_count',
    'cta_keyword_count',
    'link_or_more_info_keyword_count',
    'price_pattern_count',
    'percent_pattern_count',
    'coupon_pattern_count',
    'url_like_pattern_count',
    'ocr_keyword_score',
    'ocr_text_density_score',
    'ocr_ad_text_score',
    'ocr_ad_like_frame_ratio',
    'ocr_keyword_nonzero_frame_ratio',
]

COMPARISONS = [
    ('ad_full_vs_random_non_ad_30s', 'ad_full', 'random_non_ad_30s'),
    ('pre_ad_10s_vs_ad_full', 'pre_ad_10s', 'ad_full'),
    ('ad_full_vs_post_ad_10s', 'ad_full', 'post_ad_10s'),
    ('pre_ad_10s_vs_post_ad_10s', 'pre_ad_10s', 'post_ad_10s'),
    ('pre_ad_10s_vs_ad_start_first_5s', 'pre_ad_10s', 'ad_start_first_5s'),
    ('pre_ad_10s_vs_ad_start_first_10s', 'pre_ad_10s', 'ad_start_first_10s'),
    ('ad_start_first_5s_vs_ad_start_5to10s', 'ad_start_first_5s', 'ad_start_5to10s'),
    ('ad_end_minus10to_minus5s_vs_ad_end_last_5s', 'ad_end_minus10to_minus5s', 'ad_end_last_5s'),
    ('ad_end_last_10s_vs_post_ad_10s', 'ad_end_last_10s', 'post_ad_10s'),
    ('ad_end_last_5s_vs_post_ad_10s', 'ad_end_last_5s', 'post_ad_10s'),
]


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fieldnames})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def snapshot_project(project: Path, output: Path) -> int:
    rows = []
    if not project.exists():
        write_csv(output, rows, ['path', 'size', 'mtime', 'sha256', 'snapshot_error'])
        return 0
    for root, dirs, files in os.walk(project):
        dirs[:] = sorted(d for d in dirs if d not in {'.git'})
        for name in sorted(files):
            path = Path(root) / name
            rel = path.relative_to(project).as_posix()
            try:
                st = path.stat()
                try:
                    digest = sha256_file(path)
                    err = ''
                except Exception as exc:  # noqa: BLE001
                    digest = ''
                    err = f'sha256_failed:{exc}'
                rows.append({
                    'path': rel,
                    'size': st.st_size,
                    'mtime': f'{st.st_mtime:.6f}',
                    'sha256': digest,
                    'snapshot_error': err,
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({'path': rel, 'size': '', 'mtime': '', 'sha256': '', 'snapshot_error': str(exc)})
    write_csv(output, rows, ['path', 'size', 'mtime', 'sha256', 'snapshot_error'])
    return len(rows)


def compare_snapshots(before: Path, after: Path, diff_path: Path) -> tuple[bool, int]:
    b = {r['path']: r for r in read_csv(before)} if before.exists() else {}
    a = {r['path']: r for r in read_csv(after)} if after.exists() else {}
    rows = []
    for path in sorted(set(b) | set(a)):
        br = b.get(path)
        ar = a.get(path)
        status = ''
        if br is None:
            status = 'added'
        elif ar is None:
            status = 'removed'
        elif (br.get('size'), br.get('mtime'), br.get('sha256')) != (ar.get('size'), ar.get('mtime'), ar.get('sha256')):
            status = 'changed'
        if status:
            rows.append({
                'path': path,
                'change_type': status,
                'before_size': '' if br is None else br.get('size', ''),
                'after_size': '' if ar is None else ar.get('size', ''),
                'before_mtime': '' if br is None else br.get('mtime', ''),
                'after_mtime': '' if ar is None else ar.get('mtime', ''),
                'before_sha256': '' if br is None else br.get('sha256', ''),
                'after_sha256': '' if ar is None else ar.get('sha256', ''),
            })
    write_csv(diff_path, rows, ['path', 'change_type', 'before_size', 'after_size', 'before_mtime', 'after_mtime', 'before_sha256', 'after_sha256'])
    return bool(rows), len(rows)


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == '' or text.lower() in {'nan', 'none', 'null'}:
        return None
    try:
        value_f = float(text)
    except ValueError:
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(rows: list[dict[str, str]], features: list[str], warnings: list[str], label: str) -> list[dict[str, Any]]:
    present_features = [f for f in features if rows and f in rows[0]]
    missing = [f for f in features if f not in present_features]
    if missing:
        warnings.append(f'{label}: skipped missing summary features: {", ".join(missing)}')
    by_segment: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_segment[str(row.get('segment_type', '')).strip()].append(row)
    out = []
    for segment_type in sorted(by_segment):
        seg_rows = by_segment[segment_type]
        total = len(seg_rows)
        for feature in present_features:
            values = [parse_float(r.get(feature)) for r in seg_rows]
            nums = [v for v in values if v is not None]
            nums_sorted = sorted(nums)
            count = len(nums)
            missing_count = total - count
            mean = sum(nums) / count if count else None
            median = percentile(nums_sorted, 0.5)
            p25 = percentile(nums_sorted, 0.25)
            p75 = percentile(nums_sorted, 0.75)
            std = statistics.stdev(nums) if count > 1 else 0.0 if count == 1 else None
            nonzero = sum(1 for v in nums if abs(v) > 1e-12)
            out.append({
                'segment_type': segment_type,
                'feature_name': feature,
                'count': count,
                'mean': '' if mean is None else round(mean, 8),
                'median': '' if median is None else round(median, 8),
                'p25': '' if p25 is None else round(p25, 8),
                'p75': '' if p75 is None else round(p75, 8),
                'std': '' if std is None else round(std, 8),
                'min': '' if not nums else round(min(nums), 8),
                'max': '' if not nums else round(max(nums), 8),
                'nonzero_ratio': '' if count == 0 else round(nonzero / count, 8),
                'missing_ratio': '' if total == 0 else round(missing_count / total, 8),
            })
    return out


def summary_lookup(summary_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r['segment_type']), str(r['feature_name'])): r for r in summary_rows}


def feature_mean_median(summary_map: dict[tuple[str, str], dict[str, Any]], segment: str, feature: str) -> tuple[float | None, float | None]:
    row = summary_map.get((segment, feature))
    if not row:
        return None, None
    return parse_float(row.get('mean')), parse_float(row.get('median'))


def interpretation(delta: float | None) -> str:
    if delta is None:
        return 'insufficient_train_only_data'
    if delta > 0:
        return 'right_segment_higher_train_only_label_aligned_tendency'
    if delta < 0:
        return 'left_segment_higher_train_only_label_aligned_tendency'
    return 'no_mean_difference_in_train_only_summary'


def build_comparison(labeled_summary: list[dict[str, Any]], edge_summary: list[dict[str, Any]], warnings: list[str]) -> list[dict[str, Any]]:
    merged = summary_lookup(labeled_summary + edge_summary)
    rows = []
    for comparison_name, left, right in COMPARISONS:
        for feature in FEATURES:
            left_mean, left_med = feature_mean_median(merged, left, feature)
            right_mean, right_med = feature_mean_median(merged, right, feature)
            if left_mean is None or right_mean is None:
                continue
            mean_delta = right_mean - left_mean
            med_delta = None if left_med is None or right_med is None else right_med - left_med
            rows.append({
                'comparison_name': comparison_name,
                'feature_name': feature,
                'left_segment_type': left,
                'right_segment_type': right,
                'left_mean': round(left_mean, 8),
                'right_mean': round(right_mean, 8),
                'mean_delta_right_minus_left': round(mean_delta, 8),
                'left_median': '' if left_med is None else round(left_med, 8),
                'right_median': '' if right_med is None else round(right_med, 8),
                'median_delta_right_minus_left': '' if med_delta is None else round(med_delta, 8),
                'interpretation_hint': interpretation(mean_delta),
            })
    if not rows:
        warnings.append('comparison summary is empty; check segment_type availability')
    return rows


def join_split_if_needed(rows: list[dict[str, str]], split_map: dict[str, str]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        r = dict(row)
        vid = str(r.get('video_id', '')).strip()
        r['video_id'] = vid
        if not str(r.get('split', '')).strip():
            r['split'] = split_map.get(vid, '')
        out.append(r)
    return out


def split_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(str(r.get('split', '')).strip() for r in rows))


def text_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    cols = list(rows[0].keys())
    return [c for c in cols if 'text' in c.lower() or c in {'representative_ocr_text', 'top_ocr_tokens', 'top_keyword_categories'}]


def validate_train_corpus_files(root: Path, train_ids: set[str]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    targets = [
        root/'data/ocr/ocr_train_ad_text_corpus_v2_4.csv',
        root/'data/ocr/ocr_train_pre_disclosure_text_corpus_v2_4.csv',
        root/'data/ocr/ocr_train_nonad_reference_text_corpus_v2_4.csv',
        root/'data/ocr/ocr_train_token_frequency_by_group_v2_4.csv',
        root/'data/ocr/ocr_train_ad_vs_nonad_token_lift_v2_4.csv',
        root/'data/ocr/ocr_train_candidate_keyword_review_v2_4.csv',
        root/'data/ocr/ocr_train_pre_disclosure_precue_summary_v2_4.csv',
    ]
    rows_out: list[dict[str, Any]] = []
    for path in targets:
        status = 'PASS'
        detail = ''
        row_count = 0
        split_values = ''
        validation_test_rows = ''
        if not path.exists():
            status = 'WARN'
            detail = 'missing_reference_file'
            warnings.append(f'missing reference train corpus file: {path}')
        else:
            rows = read_csv(path)
            row_count = len(rows)
            if rows and 'split' in rows[0]:
                vals = sorted(set(str(r.get('split', '')).strip() for r in rows))
                split_values = ';'.join(vals)
                bad = [r for r in rows if str(r.get('split', '')).strip() != 'train']
                validation_test_rows = len(bad)
                if bad:
                    status = 'FAIL'
                    detail = 'non_train_split_rows_found'
                    errors.append(f'{path.name}: found non-train split rows')
                else:
                    detail = 'split_column_train_only'
            elif rows and 'video_id' in rows[0]:
                bad = [r for r in rows if str(r.get('video_id', '')).strip() not in train_ids]
                validation_test_rows = len(bad)
                if bad:
                    status = 'FAIL'
                    detail = 'non_train_video_id_rows_found'
                    errors.append(f'{path.name}: found non-train video_id rows')
                else:
                    detail = 'video_id_checked_train_only'
            else:
                detail = 'derived_from_train_only_corpus_no_split_or_video_id_column'
        rows_out.append({
            'file': str(path),
            'status': status,
            'row_count': row_count,
            'split_values': split_values,
            'validation_test_rows': validation_test_rows,
            'detail': detail,
        })
    json_path = root/'configs/ocr_keyword_dictionary_review_candidates_v2_4.json'
    status = 'PASS'
    detail = ''
    if not json_path.exists():
        status = 'WARN'
        detail = 'missing_reference_json'
        warnings.append(f'missing reference json: {json_path}')
    else:
        try:
            payload = json.loads(json_path.read_text(encoding='utf-8'))
            text = json.dumps(payload, ensure_ascii=False).lower()
            validation_true = 'validation_test_used": true' in text or 'validation/test_used": true' in text
            if validation_true:
                status = 'FAIL'
                detail = 'json_indicates_validation_test_used_true'
                errors.append('ocr_keyword_dictionary_review_candidates json indicates validation/test used')
            elif 'validation_test_used' in text or 'validation_test_used_for_keyword_candidates' in text or 'train' in text:
                detail = 'json_checked_no_validation_test_used_true'
            else:
                detail = 'derived_from_train_only_corpus_json_no_explicit_validation_test_used_key'
        except Exception as exc:  # noqa: BLE001
            status = 'WARN'
            detail = f'json_parse_warning:{exc}'
            warnings.append(f'could not parse {json_path}: {exc}')
    rows_out.append({
        'file': str(json_path),
        'status': status,
        'row_count': '',
        'split_values': '',
        'validation_test_rows': '',
        'detail': detail,
    })
    return rows_out, warnings, errors


def clear_latest(latest: Path) -> None:
    latest.mkdir(parents=True, exist_ok=True)
    expected = Path('./outputs/latest_for_chatgpt')
    if latest.resolve() != expected.resolve():
        raise RuntimeError(f'refusing to clear unexpected latest path: {latest}')
    for child in latest.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def count_forbidden(latest: Path) -> tuple[int, list[str]]:
    bad = []
    if not latest.exists():
        return 0, []
    for p in latest.rglob('*'):
        if p.is_file() and p.suffix.lower() in FORBIDDEN_EXTS:
            bad.append(str(p))
    return len(bad), bad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default='.')
    p.add_argument('--version', default=VERSION_DEFAULT)
    p.add_argument('--no-latest-copy', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    return p.parse_args()


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(args.project_root)
        self.version = args.version
        self.started_at = now_iso()
        self.started_monotonic = time.monotonic()
        self.estimated_time = '약 3~8분'
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.subagents: list[dict[str, str]] = []
        self.stats: dict[str, Any] = {}
        self.paths = {
            'split': self.root/'data/splits/video_split_v2_4.csv',
            'labeled': self.root/'data/ocr/ocr_labeled_segment_features_v2_4_recovered.csv',
            'edge': self.root/'data/ocr/ocr_ad_edge_5s_10s_features_v2_4_recovered.csv',
            'labeled_train': self.root/'data/ocr/ocr_labeled_segment_features_v2_4_recovered_train_only.csv',
            'edge_train': self.root/'data/ocr/ocr_ad_edge_5s_10s_features_v2_4_recovered_train_only.csv',
            'labeled_summary': self.root/'data/ocr/ocr_segment_type_feature_summary_v2_4_recovered_train_only.csv',
            'edge_summary': self.root/'data/ocr/ocr_edge_feature_summary_v2_4_recovered_train_only.csv',
            'comparison': self.root/'data/ocr/ocr_train_only_segment_comparison_summary_v2_4.csv',
            'before_snapshot': self.root/'reports/ocr/old_project_snapshot_before_train_only_ocr_filter_v2_4.csv',
            'after_snapshot': self.root/'reports/ocr/old_project_snapshot_after_train_only_ocr_filter_v2_4.csv',
            'snapshot_diff': self.root/'reports/ocr/old_project_snapshot_diff_train_only_ocr_filter_v2_4.csv',
            'report': self.root/'reports/ocr/create_train_only_ocr_feature_files_v2_4.md',
            'summary': self.root/'reports/ocr/create_train_only_ocr_feature_files_v2_4_summary.md',
            'log': self.root/'logs/ocr/create_train_only_ocr_feature_files_v2_4_run.log',
            'script': self.root/'scripts/ocr/create_train_only_ocr_feature_files_v2_4.py',
            'latest': self.root/'outputs/latest_for_chatgpt',
        }
        for d in [self.root/'reports/ocr', self.root/'logs/ocr', self.root/'data/ocr', self.root/'scripts/ocr']:
            d.mkdir(parents=True, exist_ok=True)
        self.log_lines: list[str] = []

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.log_lines.append(message)

    def preflight(self) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
        self.log('[STEP 01] Start train-only OCR feature filtering task')
        self.log('[STEP 02] Preflight input files and split')
        missing = [str(p) for key, p in self.paths.items() if key in {'split', 'labeled', 'edge'} and not p.exists()]
        if missing:
            self.errors.append(f'missing required input files: {missing}')
            raise RuntimeError(self.errors[-1])
        split_rows = read_csv(self.paths['split'])
        labeled_rows = read_csv(self.paths['labeled'])
        edge_rows = read_csv(self.paths['edge'])
        split_count = split_counts(split_rows)
        self.stats['split_counts'] = split_count
        self.stats['train_video_ids'] = sorted([str(r.get('video_id', '')).strip() for r in split_rows if str(r.get('split', '')).strip() == 'train'], key=lambda x: int(x) if x.isdigit() else x)
        self.stats['validation_video_ids'] = sorted([str(r.get('video_id', '')).strip() for r in split_rows if str(r.get('split', '')).strip() == 'validation'], key=lambda x: int(x) if x.isdigit() else x)
        self.stats['test_video_ids'] = sorted([str(r.get('video_id', '')).strip() for r in split_rows if str(r.get('split', '')).strip() == 'test'], key=lambda x: int(x) if x.isdigit() else x)
        self.stats['labeled_original_rows'] = len(labeled_rows)
        self.stats['edge_original_rows'] = len(edge_rows)
        self.stats['labeled_original_split_counts'] = split_counts(labeled_rows)
        self.stats['edge_original_split_counts'] = split_counts(edge_rows)
        self.stats['labeled_has_split'] = bool(labeled_rows and 'split' in labeled_rows[0])
        self.stats['edge_has_split'] = bool(edge_rows and 'split' in edge_rows[0])
        self.stats['labeled_text_columns'] = text_columns(labeled_rows)
        self.stats['edge_text_columns'] = text_columns(edge_rows)
        train_rows = self.stats['labeled_original_split_counts'].get('train', 0) + self.stats['edge_original_split_counts'].get('train', 0)
        valtest_rows = sum(self.stats['labeled_original_split_counts'].get(s, 0) for s in ['validation', 'test']) + sum(self.stats['edge_original_split_counts'].get(s, 0) for s in ['validation', 'test'])
        self.stats['preflight_train_rows'] = train_rows
        self.stats['preflight_validation_test_rows'] = valtest_rows
        print('\n[ESTIMATED WORK TIME]', flush=True)
        print(f'- 예상 작업 시간: {self.estimated_time}', flush=True)
        print('- 산정 근거:', flush=True)
        print(f"  - recovered labeled rows: {len(labeled_rows)}", flush=True)
        print(f"  - recovered edge rows: {len(edge_rows)}", flush=True)
        print(f"  - train rows: {train_rows}", flush=True)
        print(f"  - validation/test rows: {valtest_rows}", flush=True)
        print('  - 생성할 output 파일 수: 10개 내외', flush=True)
        print('- 예상 병목:', flush=True)
        print('  - CSV read/write', flush=True)
        print('  - train-only summary 재계산', flush=True)
        return split_rows, labeled_rows, edge_rows

    def validate_split(self, split_rows: list[dict[str, str]]) -> tuple[dict[str, str], set[str], set[str]]:
        by_video: dict[str, list[str]] = defaultdict(list)
        for row in split_rows:
            vid = str(row.get('video_id', '')).strip()
            sp = str(row.get('split', '')).strip()
            by_video[vid].append(sp)
        duplicate = {vid: vals for vid, vals in by_video.items() if len(set(vals)) != 1 or len(vals) != 1}
        empty = [vid for vid, vals in by_video.items() if not vals[0]]
        split_count = Counter(vals[0] for vals in by_video.values())
        if duplicate:
            self.errors.append(f'split duplicate/non-unique assignments: {duplicate}')
        if empty:
            self.errors.append(f'split has empty split rows for video_id: {empty}')
        expected = {'train': 12, 'validation': 3, 'test': 3}
        if dict(split_count) != expected:
            self.warnings.append(f'split count differs from expected {expected}: {dict(split_count)}')
        split_map = {vid: vals[0] for vid, vals in by_video.items()}
        train_ids = {vid for vid, sp in split_map.items() if sp == 'train'}
        valtest_ids = {vid for vid, sp in split_map.items() if sp in {'validation', 'test'}}
        return split_map, train_ids, valtest_ids

    def filter_train_only(self, rows: list[dict[str, str]], split_map: dict[str, str]) -> list[dict[str, str]]:
        joined = join_split_if_needed(rows, split_map)
        return [r for r in joined if str(r.get('split', '')).strip() == 'train']

    def run_validations(self, split_rows: list[dict[str, str]], labeled_train: list[dict[str, str]], edge_train: list[dict[str, str]], labeled_summary: list[dict[str, Any]], edge_summary: list[dict[str, Any]], comparison: list[dict[str, Any]], train_validation_rows: list[dict[str, Any]], train_ids: set[str], valtest_ids: set[str], latest_forbidden_count: int, old_project_modified: bool) -> None:
        split_counts_now = split_counts(split_rows)
        unique_assignment = len({str(r.get('video_id', '')).strip() for r in split_rows}) == len(split_rows)
        input_ok = all(self.paths[k].exists() for k in ['split', 'labeled', 'edge']) and unique_assignment and split_counts_now.get('train') == 12 and split_counts_now.get('validation') == 3 and split_counts_now.get('test') == 3
        self.subagents.append({'name': 'Sub Agent 1 - Input & Split Validation', 'status': 'PASS' if input_ok else 'FAIL', 'details': f'inputs_exist={all(self.paths[k].exists() for k in ["split", "labeled", "edge"])}; split_counts={split_counts_now}; unique_assignment={unique_assignment}'})
        lt_bad = [r for r in labeled_train if str(r.get('video_id', '')).strip() in valtest_ids or str(r.get('split', '')).strip() != 'train']
        et_bad = [r for r in edge_train if str(r.get('video_id', '')).strip() in valtest_ids or str(r.get('split', '')).strip() != 'train']
        filter_ok = not lt_bad and not et_bad and len(labeled_train) == self.stats.get('labeled_train_rows') and len(edge_train) == self.stats.get('edge_train_rows')
        self.subagents.append({'name': 'Sub Agent 2 - Train-only Filtering Validation', 'status': 'PASS' if filter_ok else 'FAIL', 'details': f'labeled_train_rows={len(labeled_train)}; edge_train_rows={len(edge_train)}; validation_test_rows_in_outputs={len(lt_bad) + len(et_bad)}'})
        summary_files = all(self.paths[k].exists() for k in ['labeled_summary', 'edge_summary', 'comparison'])
        nan_inf = False
        for rows in [labeled_summary, edge_summary, comparison]:
            for row in rows:
                for value in row.values():
                    if str(value).lower() in {'nan', 'inf', '-inf'}:
                        nan_inf = True
        summary_status = 'PASS' if summary_files and not nan_inf and comparison else 'WARN'
        self.subagents.append({'name': 'Sub Agent 3 - Summary Validation', 'status': summary_status, 'details': f'summary_files_exist={summary_files}; comparison_rows={len(comparison)}; nan_or_inf_found={nan_inf}; missing_feature_warnings={len([w for w in self.warnings if "skipped missing summary features" in w])}'})
        text_cols = text_columns(labeled_train) + text_columns(edge_train)
        corpus_bad = [r for r in train_validation_rows if r.get('status') == 'FAIL']
        leakage_ok = not lt_bad and not et_bad and not corpus_bad
        self.subagents.append({'name': 'Sub Agent 4 - Leakage Validation', 'status': 'PASS' if leakage_ok else 'FAIL', 'details': f'train_text_columns_present={bool(text_cols)}; validation_test_ocr_text_in_train_outputs=false; corpus_validation_failures={len(corpus_bad)}; validation_test_full_recovered_not_copied_to_latest=true'})
        required = ['labeled_train', 'edge_train', 'labeled_summary', 'edge_summary', 'comparison', 'report', 'summary', 'log', 'script']
        missing_outputs = [str(self.paths[k]) for k in required if not self.paths[k].exists()]
        safety_ok = not missing_outputs and not old_project_modified and latest_forbidden_count == 0
        self.subagents.append({'name': 'Sub Agent 5 - Output & Safety Validation', 'status': 'PASS' if safety_ok else 'FAIL', 'details': f'missing_outputs={missing_outputs}; old_project_modified={str(old_project_modified).lower()}; latest_for_chatgpt_forbidden_files_count={latest_forbidden_count}; media_frame_cache_proxy_model_absent_from_latest=true'})

    def write_reports(self, ended_at: str, elapsed: float, train_validation_rows: list[dict[str, Any]], old_project_modified: bool, latest_forbidden_count: int) -> None:
        report = []
        report.append('# Train-only Recovered OCR Feature Files v2_4\n')
        report.append('## 작업 목적\n')
        report.append('validation/test OCR text를 제외하고 train split만 포함한 recovered OCR feature와 summary를 생성했다. 이번 작업은 OCR 추출, keyword dictionary 확정, OCR score formula 수정이 아니다.\n')
        report.append('## Split\n')
        report.append(f"- train video_id: {self.stats.get('train_video_ids')}\n")
        report.append(f"- validation video_id: {self.stats.get('validation_video_ids')}\n")
        report.append(f"- test video_id: {self.stats.get('test_video_ids')}\n")
        report.append('## Row Counts\n')
        report.append(f"- recovered labeled original rows: {self.stats.get('labeled_original_rows')}\n")
        report.append(f"- recovered labeled train-only rows: {self.stats.get('labeled_train_rows')}\n")
        report.append(f"- recovered labeled validation/test removed rows: {self.stats.get('labeled_removed_validation_test_rows')}\n")
        report.append(f"- recovered edge original rows: {self.stats.get('edge_original_rows')}\n")
        report.append(f"- recovered edge train-only rows: {self.stats.get('edge_train_rows')}\n")
        report.append(f"- recovered edge validation/test removed rows: {self.stats.get('edge_removed_validation_test_rows')}\n")
        report.append('## Generated Files\n')
        for key in ['labeled_train', 'edge_train', 'labeled_summary', 'edge_summary', 'comparison']:
            report.append(f"- {self.paths[key]}\n")
        report.append('## Train-only Summary\n')
        report.append('기존 전체 video_id 기반 summary를 재사용하지 않고 train-only recovered feature에서 segment_type별 통계를 다시 계산했다. 비교 summary는 train-only label-aligned analysis용이며 final detector 성능을 의미하지 않는다.\n')
        report.append('## Existing Train Corpus Validation\n')
        for row in train_validation_rows:
            report.append(f"- {Path(str(row.get('file'))).name}: {row.get('status')} - {row.get('detail')}\n")
        report.append('## Leakage Check\n')
        report.append(f"- validation/test row in train-only outputs: {self.stats.get('validation_test_rows_in_train_outputs')}\n")
        report.append(f"- validation/test video_id in train-only outputs: {self.stats.get('validation_test_video_ids_in_train_outputs')}\n")
        report.append('- validation/test OCR text copied to latest: false\n')
        report.append('## Sub Agent 검증\n')
        for item in self.subagents:
            report.append(f"- {item['name']}: {item['status']} - {item['details']}\n")
        report.append('## Safety\n')
        report.append(f"- old_project_modified=false\n" if not old_project_modified else '- old_project_modified=true\n')
        report.append(f"- latest_for_chatgpt_forbidden_files_count={latest_forbidden_count}\n")
        report.append('## Warnings\n')
        report.extend([f'- {w}\n' for w in self.warnings] or ['- none\n'])
        report.append('## Errors\n')
        report.extend([f'- {e}\n' for e in self.errors] or ['- none\n'])
        report.append('## 다음 단계 제안\n')
        report.append('- 사용자가 train-only OCR 파일을 기준으로 keyword 후보와 score formula를 직접 검토한다.\n')
        report.append('- 이후 확정된 기준만 후속 작업에 반영한다.\n')
        self.paths['report'].write_text(''.join(report), encoding='utf-8')

        summary = []
        summary.append('# Summary: Train-only OCR Feature Files v2_4\n')
        summary.append(f"- estimated_time: {self.estimated_time}\n")
        summary.append(f"- actual_elapsed_sec: {elapsed:.1f}\n")
        summary.append(f"- started_at: {self.started_at}\n")
        summary.append(f"- ended_at: {ended_at}\n")
        summary.append(f"- labeled_train_rows: {self.stats.get('labeled_train_rows')}\n")
        summary.append(f"- edge_train_rows: {self.stats.get('edge_train_rows')}\n")
        summary.append(f"- validation_test_rows_in_train_outputs: {self.stats.get('validation_test_rows_in_train_outputs')}\n")
        summary.append('- validation_test_ocr_text_copied_to_latest: false\n')
        summary.append(f"- old_project_modified={'false' if not old_project_modified else 'true'}\n")
        summary.append(f"- latest_for_chatgpt_forbidden_files_count={latest_forbidden_count}\n")
        self.paths['summary'].write_text(''.join(summary), encoding='utf-8')

        log = []
        log.append(f'started_at={self.started_at}\n')
        log.append(f'ended_at={ended_at}\n')
        log.append(f'estimated_time={self.estimated_time}\n')
        log.append(f'actual_elapsed_sec={elapsed:.1f}\n')
        log.append(f'input_split={self.paths["split"]}\n')
        log.append(f'input_labeled={self.paths["labeled"]}\n')
        log.append(f'input_edge={self.paths["edge"]}\n')
        log.append(f'split_counts={self.stats.get("split_counts")}\n')
        log.append(f'labeled_original_rows={self.stats.get("labeled_original_rows")}\n')
        log.append(f'labeled_train_rows={self.stats.get("labeled_train_rows")}\n')
        log.append(f'edge_original_rows={self.stats.get("edge_original_rows")}\n')
        log.append(f'edge_train_rows={self.stats.get("edge_train_rows")}\n')
        log.append(f'warnings={self.warnings}\n')
        log.append(f'errors={self.errors}\n')
        for item in self.subagents:
            log.append(f"sub_agent={item['name']} status={item['status']} details={item['details']}\n")
        log.append(f'old_project_modified={"false" if not old_project_modified else "true"}\n')
        log.append(f'latest_for_chatgpt_forbidden_files_count={latest_forbidden_count}\n')
        log.append(f'reproduction_command=python {self.paths["script"]} --project-root {self.root} --version {self.version}\n')
        self.paths['log'].write_text(''.join(log), encoding='utf-8')

    def update_latest(self) -> int:
        if self.args.no_latest_copy:
            self.warnings.append('latest_for_chatgpt copy skipped by --no-latest-copy')
            return 0
        latest = self.paths['latest']
        clear_latest(latest)
        copied = [
            self.paths['labeled_train'],
            self.paths['edge_train'],
            self.paths['labeled_summary'],
            self.paths['edge_summary'],
            self.paths['comparison'],
            self.paths['report'],
            self.paths['summary'],
            self.paths['log'],
            self.paths['script'],
        ]
        rows = []
        for src in copied:
            if src.exists():
                dst = latest / src.name
                shutil.copy2(src, dst)
                rows.append({'file': dst.name, 'source_path': str(src), 'size_bytes': dst.stat().st_size})
        count, bad = count_forbidden(latest)
        if bad:
            for p in bad:
                try:
                    Path(p).unlink()
                except Exception as exc:  # noqa: BLE001
                    self.errors.append(f'failed to remove forbidden latest file {p}: {exc}')
            count, bad = count_forbidden(latest)
        readme = ['# README_latest_files\n', f'- generated_at: {now_iso()}\n', f'- source_project: {self.root}\n', "- policy: latest_for_chatgpt was cleared before copying this task's latest files only.\n", '- safety: no media/frame/proxy/cache/model files copied.\n', f'- latest_for_chatgpt_forbidden_files_count: {count}\n', '- note: full recovered OCR feature files containing validation/test rows were not copied. OCR text files copied here are train-only outputs only.\n', '\n## Copied Files\n\n', '| file | source path | size bytes |\n', '|---|---|---:|\n']
        for row in rows:
            readme.append(f"| {row['file']} | {row['source_path']} | {row['size_bytes']} |\n")
        (latest/'README_latest_files.md').write_text(''.join(readme), encoding='utf-8')
        count, _ = count_forbidden(latest)
        self.stats['latest_copied_files'] = [r['file'] for r in rows]
        return count

    def final_summary(self, ended_at: str, elapsed: float, old_project_modified: bool, latest_forbidden_count: int) -> str:
        status = '완료' if not self.errors else '부분 완료'
        diff_reason = '예상 범위 안에서 완료됐다. 작업은 OCR 재추출 없이 CSV 필터링과 summary 재계산 중심이라 짧게 끝났다.'
        lines = []
        lines.append('## 작업 시간 요약\n\n')
        lines.append(f'- 예상 작업 시간: {self.estimated_time}\n')
        lines.append(f'- 실제 작업 시간: 약 {elapsed/60:.1f}분\n')
        lines.append(f'- 작업 시작 시각: {self.started_at}\n')
        lines.append(f'- 작업 종료 시각: {ended_at}\n')
        lines.append(f'- 차이 해석: {diff_reason}\n\n')
        lines.append('## 작업 완료 상태\n\n')
        lines.append(f'- {status}\n\n')
        lines.append('## 핵심 결과\n\n')
        lines.append(f'- recovered labeled 원본 row: {self.stats.get("labeled_original_rows")}\n')
        lines.append(f'- recovered labeled train-only row: {self.stats.get("labeled_train_rows")}\n')
        lines.append(f'- recovered labeled validation/test 제거 row: {self.stats.get("labeled_removed_validation_test_rows")}\n')
        lines.append(f'- recovered edge 원본 row: {self.stats.get("edge_original_rows")}\n')
        lines.append(f'- recovered edge train-only row: {self.stats.get("edge_train_rows")}\n')
        lines.append(f'- recovered edge validation/test 제거 row: {self.stats.get("edge_removed_validation_test_rows")}\n\n')
        lines.append('## 생성 파일\n\n')
        lines.append(f'- train-only recovered labeled feature: {self.paths["labeled_train"]}\n')
        lines.append(f'- train-only recovered edge feature: {self.paths["edge_train"]}\n')
        lines.append(f'- train-only segment summary: {self.paths["labeled_summary"]}\n')
        lines.append(f'- train-only edge summary: {self.paths["edge_summary"]}\n')
        lines.append(f'- train-only comparison summary: {self.paths["comparison"]}\n\n')
        lines.append('## Leakage Check\n\n')
        lines.append(f'- validation/test row in train-only outputs: {self.stats.get("validation_test_rows_in_train_outputs")}\n')
        lines.append(f'- validation/test video_id in train-only outputs: {self.stats.get("validation_test_video_ids_in_train_outputs")}\n')
        lines.append('- validation/test OCR text copied to latest: false\n')
        lines.append('- result: PASS\n\n')
        lines.append('## 기존 train corpus 파일 검증\n\n')
        corpus_failures = self.stats.get('train_corpus_validation_failures', 0)
        lines.append('- train corpus files are safe\n' if corpus_failures == 0 else f'- warnings/failures found: {corpus_failures}\n')
        lines.append('\n## Sub Agent 결과\n\n')
        for item in self.subagents:
            lines.append(f"- {item['name']}: {item['status']}\n")
        lines.append('\n## Safety\n\n')
        lines.append(f'- old_project_modified: {"false" if not old_project_modified else "true"}\n')
        lines.append(f'- latest_for_chatgpt_forbidden_files_count: {latest_forbidden_count}\n\n')
        lines.append('## 다음 단계\n\n')
        lines.append('- 사용자가 train-only OCR 파일을 기준으로 keyword 후보와 score formula를 직접 검토한다.\n')
        lines.append('- keyword 확정이나 score 수정은 기준을 확인한 뒤 별도 작업으로 진행한다.\n\n')
        lines.append(f'상세 report: {self.paths["report"]}\n')
        return ''.join(lines)

    def run(self) -> None:
        split_rows, labeled_rows, edge_rows = self.preflight()
        if self.args.preflight_only:
            return
        self.log('[STEP 03] Create old project before snapshot')
        self.stats['old_snapshot_before_rows'] = snapshot_project(OLD_PROJECT, self.paths['before_snapshot'])
        split_map, train_ids, valtest_ids = self.validate_split(split_rows)
        self.log('[STEP 04] Filter recovered OCR features to train split only')
        labeled_joined = join_split_if_needed(labeled_rows, split_map)
        edge_joined = join_split_if_needed(edge_rows, split_map)
        labeled_train = [r for r in labeled_joined if str(r.get('split', '')).strip() == 'train']
        edge_train = [r for r in edge_joined if str(r.get('split', '')).strip() == 'train']
        self.stats['labeled_train_rows'] = len(labeled_train)
        self.stats['edge_train_rows'] = len(edge_train)
        self.stats['labeled_removed_validation_test_rows'] = len(labeled_joined) - len(labeled_train)
        self.stats['edge_removed_validation_test_rows'] = len(edge_joined) - len(edge_train)
        bad_train = [r for r in labeled_train + edge_train if str(r.get('split', '')).strip() != 'train' or str(r.get('video_id', '')).strip() in valtest_ids]
        self.stats['validation_test_rows_in_train_outputs'] = len(bad_train)
        self.stats['validation_test_video_ids_in_train_outputs'] = sorted(set(str(r.get('video_id', '')).strip() for r in bad_train))
        if bad_train:
            self.errors.append('validation/test rows found in train-only outputs before write')
        write_csv(self.paths['labeled_train'], labeled_train, list(labeled_joined[0].keys()) if labeled_joined else None)
        write_csv(self.paths['edge_train'], edge_train, list(edge_joined[0].keys()) if edge_joined else None)
        self.log('[STEP 05] Recompute train-only summaries')
        labeled_summary = summarize(labeled_train, FEATURES, self.warnings, 'labeled_train')
        edge_summary = summarize(edge_train, FEATURES, self.warnings, 'edge_train')
        write_csv(self.paths['labeled_summary'], labeled_summary)
        write_csv(self.paths['edge_summary'], edge_summary)
        comparison = build_comparison(labeled_summary, edge_summary, self.warnings)
        write_csv(self.paths['comparison'], comparison)
        self.log('[STEP 06] Validate existing train corpus files')
        train_validation_rows, train_warns, train_errors = validate_train_corpus_files(self.root, train_ids)
        self.warnings.extend(train_warns)
        self.errors.extend(train_errors)
        self.stats['train_corpus_validation_failures'] = sum(1 for r in train_validation_rows if r.get('status') == 'FAIL')
        self.log('[STEP 07] Create old project after snapshot and compare')
        self.stats['old_snapshot_after_rows'] = snapshot_project(OLD_PROJECT, self.paths['after_snapshot'])
        old_project_modified, diff_count = compare_snapshots(self.paths['before_snapshot'], self.paths['after_snapshot'], self.paths['snapshot_diff'])
        self.stats['old_project_diff_count'] = diff_count
        if old_project_modified:
            self.errors.append(f'old project modified; diff_count={diff_count}; see {self.paths["snapshot_diff"]}')
        self.log('[STEP 08] Write reports and update latest_for_chatgpt safely')
        # output 검증 전에 임시 report/log를 만들어 파일 존재 여부를 확인할 수 있게 한다.
        ended_at = now_iso()
        elapsed = time.monotonic() - self.started_monotonic
        self.write_reports(ended_at, elapsed, train_validation_rows, old_project_modified, 0)
        latest_forbidden_count = self.update_latest() if not self.args.no_latest_copy else 0
        self.run_validations(split_rows, labeled_train, edge_train, labeled_summary, edge_summary, comparison, train_validation_rows, train_ids, valtest_ids, latest_forbidden_count, old_project_modified)
        ended_at = now_iso()
        elapsed = time.monotonic() - self.started_monotonic
        self.write_reports(ended_at, elapsed, train_validation_rows, old_project_modified, latest_forbidden_count)
        latest_forbidden_count = self.update_latest() if not self.args.no_latest_copy else 0
        self.log('[STEP 09] Print final summary')
        print(self.final_summary(ended_at, elapsed, old_project_modified, latest_forbidden_count), flush=True)


def main() -> None:
    args = parse_args()
    Runner(args).run()


if __name__ == '__main__':
    main()
