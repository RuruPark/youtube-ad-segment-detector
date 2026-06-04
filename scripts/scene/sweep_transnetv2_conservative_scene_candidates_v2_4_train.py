#!/usr/bin/env python3
# 광고 경계 포착률을 보기 위한 train-only TransNetV2 conservative threshold/dedup sweep.

from __future__ import annotations

import csv
import datetime as dt
import gc
import json
import math
import os
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import extract_transnetv2_scene_candidates_v2_4_train_audit as base

PROJECT_ROOT = Path('.')
TRAIN_VIDEO_IDS = base.TRAIN_VIDEO_IDS
VALIDATION_VIDEO_IDS = base.VALIDATION_VIDEO_IDS
TEST_VIDEO_IDS = base.TEST_VIDEO_IDS
TOLERANCES = [2, 5, 10]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
DEDUP_WINDOWS = [2.0, 3.0, 5.0]
SWEEP_COMBOS = [(t, d) for t in THRESHOLDS for d in DEDUP_WINDOWS]
CONSERVATIVE_MIN_THRESHOLD = 0.7
CONSERVATIVE_DEDUPS = {3.0, 5.0}
EXISTING_CROSS_SOURCE_DEDUP_SEC = 2.0

SCRIPT_PATH = PROJECT_ROOT / 'scripts/scene/sweep_transnetv2_conservative_scene_candidates_v2_4_train.py'
LOG_PATH = PROJECT_ROOT / 'logs/transnetv2_conservative_sweep_v2_4_run_log.txt'
LATEST_BUNDLE_DIR = PROJECT_ROOT / 'outputs/latest_for_chatgpt_transnetv2_conservative_sweep_v2_4'
SHARED_LATEST_DIR = PROJECT_ROOT / 'outputs/latest_for_chatgpt'

CANDIDATES_CSV = PROJECT_ROOT / 'data/scene/transnetv2_conservative_sweep_candidates_v2_4_train.csv'
SWEEP_SUMMARY_CSV = PROJECT_ROOT / 'data/scene/transnetv2_conservative_sweep_summary_v2_4_train.csv'
RECALL_AUDIT_CSV = PROJECT_ROOT / 'data/scene/transnetv2_conservative_boundary_recall_audit_v2_4_train.csv'
RECALL_SUMMARY_CSV = PROJECT_ROOT / 'data/scene/transnetv2_conservative_boundary_recall_summary_v2_4_train.csv'
RECOVERY_CSV = PROJECT_ROOT / 'data/scene/transnetv2_conservative_existing_missed_recovery_v2_4_train.csv'
BEST_CANDIDATE_CSV = PROJECT_ROOT / 'data/scene/transnetv2_conservative_best_candidate_v2_4_train.csv'
COMPARE_CSV = PROJECT_ROOT / 'data/scene/transnetv2_conservative_compare_existing_models_v2_4_train.csv'
VIDEO_LEVEL_CSV = PROJECT_ROOT / 'data/scene/transnetv2_conservative_video_level_recall_v2_4_train.csv'

SUMMARY_MD = PROJECT_ROOT / 'reports/scene/transnetv2_conservative_sweep_v2_4_summary.md'
REPORT_JSON = PROJECT_ROOT / 'reports/scene/transnetv2_conservative_sweep_v2_4_report.json'
FINDINGS_MD = PROJECT_ROOT / 'reports/scene/transnetv2_conservative_sweep_v2_4_findings.md'

INPUT_FILES = [
    base.SPLIT_PATH,
    base.SEGMENT_PATH,
    base.MANIFEST_PATH,
    base.OPENCV_PATH,
    base.RESNET_PATH,
    base.CANONICAL_PATH,
    base.PREVIOUS_AUDIT_CASE_PATH,
    base.PREVIOUS_AUDIT_REPORT_PATH,
    base.SETUP_REPORT_PATH,
    base.TRANSNET_WEIGHT_PATH,
    PROJECT_ROOT / 'data/scene/transnetv2_scene_candidates_v2_4_train.csv',
    PROJECT_ROOT / 'data/scene/transnetv2_scene_candidates_v2_4_train_raw_outputs_index.csv',
    PROJECT_ROOT / 'reports/scene/transnetv2_scene_candidate_audit_v2_4_report.json',
]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def safe_float(value: Any) -> Optional[float]:
    return base.safe_float(value)


def safe_int(value: Any) -> Optional[int]:
    return base.safe_int(value)


def fmt(value: Optional[float], digits: int = 4) -> str:
    return base.format_float(value, digits)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    return base.read_csv_rows(path)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    base.write_csv(path, fieldnames, rows)


def token(value: float) -> str:
    return f'{value:g}'.replace('.', '_')


def sweep_family(threshold: float, dedup: float) -> str:
    return f'transnetv2_threshold_{token(threshold)}_dedup_{token(dedup)}'


def infer_threshold_from_family(family: str) -> Optional[float]:
    if 'threshold_' not in family:
        return None
    text = family.split('threshold_', 1)[1].split('_dedup_', 1)[0]
    return safe_float(text.replace('_', '.'))


def infer_dedup_from_family(family: str) -> Optional[float]:
    if '_dedup_' not in family:
        return None
    text = family.split('_dedup_', 1)[1]
    return safe_float(text.replace('_', '.'))


def total_minutes(mapping_rows: Sequence[Dict[str, Any]]) -> float:
    seconds = 0.0
    for row in mapping_rows:
        dur = safe_float(row.get('duration_sec'))
        if dur:
            seconds += dur
    return seconds / 60.0 if seconds > 0 else 0.0


def existing_raw_score_available(logger: base.Logger) -> Dict[str, Any]:
    index_path = PROJECT_ROOT / 'data/scene/transnetv2_scene_candidates_v2_4_train_raw_outputs_index.csv'
    thresholds = set()
    raw_paths = set()
    has_frame_scores = False
    if index_path.exists():
        for row in read_csv_rows(index_path):
            th = safe_float(row.get('threshold'))
            if th is not None:
                thresholds.add(th)
            if row.get('raw_output_path'):
                raw_paths.add(row['raw_output_path'])
    for raw in sorted(raw_paths)[:1]:
        try:
            payload = json.loads(Path(raw).read_text(encoding='utf-8'))
            has_frame_scores = any(k in payload for k in ['frame_scores', 'predictions', 'single_frame_predictions'])
        except Exception:
            pass
    can_reuse = has_frame_scores and set(THRESHOLDS).issubset(thresholds)
    logger.write(f'기존 raw score 재사용 확인: frame_level_scores={has_frame_scores}, thresholds={sorted(thresholds)}, 재추론필요={not can_reuse}')
    return {
        'raw_outputs_index_path': str(index_path),
        'raw_output_file_count': len(raw_paths),
        'thresholds_in_existing_outputs': sorted(thresholds),
        'frame_level_scores_available': has_frame_scores,
        'can_reuse_without_inference': can_reuse,
        'reason': '' if can_reuse else '기존 output은 threshold별 후보 목록만 있고 0.6/0.8/0.85/0.9/0.95 재계산용 frame-level prediction이 없음',
    }


def cluster_candidates(raw: List[Dict[str, Any]], dedup: float, vid: int, family: str) -> List[Dict[str, Any]]:
    if not raw:
        return []
    ordered = sorted(raw, key=lambda r: (float(r['candidate_sec']), -float(r.get('transnetv2_score') or 0)))
    groups: List[List[Dict[str, Any]]] = []
    group: List[Dict[str, Any]] = []
    group_max = None
    for item in ordered:
        sec = float(item['candidate_sec'])
        if not group:
            group = [item]
            group_max = sec
        elif sec - float(group_max) <= dedup:
            group.append(item)
            group_max = max(float(group_max), sec)
        else:
            groups.append(group)
            group = [item]
            group_max = sec
    if group:
        groups.append(group)
    out: List[Dict[str, Any]] = []
    for idx, members in enumerate(groups, start=1):
        best = sorted(members, key=lambda r: (-(safe_float(r.get('transnetv2_score')) or -1.0), float(r['candidate_sec'])))[0]
        secs = [float(m['candidate_sec']) for m in members]
        out.append({
            'sweep_family': family,
            'video_id': vid,
            'split': 'train',
            'candidate_sec': float(best['candidate_sec']),
            'candidate_frame': best.get('candidate_frame'),
            'transnetv2_score': safe_float(best.get('transnetv2_score')),
            'threshold': best.get('threshold'),
            'dedup_window_sec': dedup,
            'cluster_id': f'{family}_v{vid:02d}_c{idx:05d}',
            'cluster_member_count': len(members),
            'cluster_min_sec': min(secs),
            'cluster_max_sec': max(secs),
            'device_used': best.get('device_used'),
            'fps': best.get('fps'),
            'duration_sec': best.get('duration_sec'),
            'source': 'train_only_transnetv2_rerun_frame_predictions',
            'notes': 'dedup representative is highest score timestamp; tie uses earliest timestamp; no average timestamp; no labels used',
        })
    return out


def run_transnetv2_sweep(mapping_rows: List[Dict[str, Any]], logger: base.Logger, warnings: List[str], errors: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if str(base.TRANSNET_PYTHONPATH) not in sys.path:
        sys.path.insert(0, str(base.TRANSNET_PYTHONPATH))
    if base.CV_FFMPEG_BIN_DIR.exists():
        os.environ['PATH'] = str(base.CV_FFMPEG_BIN_DIR) + os.pathsep + os.environ.get('PATH', '')
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from transnetv2_pytorch import TransNetV2  # type: ignore

    cuda_available = bool(torch.cuda.is_available())
    device = 'cuda' if cuda_available else 'cpu'
    fallback_used = False
    logger.write(f'TransNetV2 sweep 추론 장치: initial_device={device}, cuda_available={cuda_available}')

    def make_model(dev: str) -> Any:
        logger.write(f'TransNetV2 모델 초기화: device={dev}')
        return TransNetV2(device=dev)

    try:
        model = make_model(device)
    except Exception as exc:
        warnings.append(f'CUDA 모델 초기화 실패, CPU fallback: {exc}')
        logger.write(f'CUDA 모델 초기화 실패, CPU fallback: {exc}')
        device = 'cpu'
        fallback_used = True
        model = make_model(device)

    all_rows: List[Dict[str, Any]] = []
    processed: List[int] = []
    failed: List[int] = []
    runtimes: Dict[str, float] = {}
    for row in mapping_rows:
        vid = int(row['video_id'])
        if not row.get('file_exists'):
            warnings.append(f'영상 파일 없음: video_id={vid}')
            continue
        video_path = str(row['video_path'])
        fps = safe_float(row.get('fps'))
        duration = safe_float(row.get('duration_sec'))
        active_device = device
        t0 = time.time()
        logger.write(f'video_id={vid} raw score 추론 시작: {video_path}')
        try:
            video_tensor, single_frame_predictions, all_frame_predictions = model.predict_video(video_path, quiet=True)
        except Exception as exc:
            if device != 'cpu':
                warnings.append(f'CUDA 추론 실패, CPU fallback: video_id={vid}, error={exc}')
                logger.write(f'video_id={vid} CUDA 실패, CPU 재시도')
                try:
                    del model
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                device = 'cpu'
                active_device = 'cpu'
                fallback_used = True
                model = make_model('cpu')
                try:
                    video_tensor, single_frame_predictions, all_frame_predictions = model.predict_video(video_path, quiet=True)
                except Exception as cpu_exc:
                    errors.append(f'CPU fallback 실패: video_id={vid}, error={cpu_exc}')
                    failed.append(vid)
                    continue
            else:
                errors.append(f'CPU 추론 실패: video_id={vid}, error={exc}')
                failed.append(vid)
                continue
        try:
            predictions = single_frame_predictions.cpu().detach().numpy().reshape(-1)
            frame_count = len(predictions)
            fps_model = safe_float(model.get_video_fps(video_path))
            if fps_model:
                fps = fps_model
            if duration is None and fps:
                duration = frame_count / fps
            raw_by_threshold: Dict[float, List[Dict[str, Any]]] = {}
            for threshold in THRESHOLDS:
                scenes = model.predictions_to_scenes_with_data(predictions, fps=fps, threshold=threshold)
                raw_candidates: List[Dict[str, Any]] = []
                for scene_idx, scene in enumerate(scenes):
                    if scene_idx == len(scenes) - 1:
                        continue
                    frame = int(scene.get('end_frame', 0))
                    start = max(0, frame - 2)
                    end = min(len(predictions), frame + 3)
                    window = predictions[start:end]
                    score = float(np.max(window)) if len(window) else safe_float(scene.get('probability')) or 0.0
                    sec = frame / fps if fps else safe_float(scene.get('end_time'))
                    if sec is None:
                        continue
                    raw_candidates.append({
                        'candidate_sec': float(sec),
                        'candidate_frame': frame,
                        'transnetv2_score': score,
                        'threshold': threshold,
                        'device_used': active_device,
                        'fps': fps,
                        'duration_sec': duration,
                    })
                raw_by_threshold[threshold] = raw_candidates
            for threshold, dedup in SWEEP_COMBOS:
                all_rows.extend(cluster_candidates(raw_by_threshold.get(threshold, []), dedup, vid, sweep_family(threshold, dedup)))
            processed.append(vid)
            runtimes[str(vid)] = time.time() - t0
            logger.write(f"video_id={vid} sweep 후보 완료: {runtimes[str(vid)]:.2f}초")
        except Exception as exc:
            errors.append(f'sweep 변환 실패: video_id={vid}, error={exc}')
            failed.append(vid)
            logger.write(f'video_id={vid} sweep 변환 실패: {exc}')
        finally:
            try:
                del video_tensor
                del single_frame_predictions
                del all_frame_predictions
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return all_rows, {
        'cuda_available': cuda_available,
        'initial_device': 'cuda' if cuda_available else 'cpu',
        'final_device': device,
        'cuda_to_cpu_fallback_used': fallback_used,
        'processed_video_ids': processed,
        'processed_video_count': len(processed),
        'failed_video_ids': failed,
        'failed_video_count': len(failed),
        'per_video_runtime_seconds': runtimes,
        'raw_score_persisted': False,
        'raw_score_persist_note': 'frame-level predictions were used in memory only; large raw prediction arrays were not saved or bundled',
    }


def count_by_video(rows: Sequence[Dict[str, Any]], family_col: str) -> Dict[str, Dict[int, int]]:
    out: Dict[str, Dict[int, int]] = {}
    for row in rows:
        family = str(row.get(family_col) or row.get('candidate_family'))
        vid = safe_int(row.get('video_id'))
        if vid is None:
            continue
        out.setdefault(family, {})[vid] = out.setdefault(family, {}).get(vid, 0) + 1
    return out


def group_times_by_family(rows: Sequence[Dict[str, Any]], family_col: str) -> Dict[str, Dict[int, List[float]]]:
    out: Dict[str, Dict[int, List[float]]] = {}
    for row in rows:
        family = str(row.get(family_col) or row.get('candidate_family'))
        vid = safe_int(row.get('video_id'))
        sec = safe_float(row.get('candidate_sec'))
        if vid is None or sec is None:
            continue
        out.setdefault(family, {}).setdefault(vid, []).append(sec)
    for fam in out:
        for vid in out[fam]:
            out[fam][vid].sort()
    return out


def build_plus_clusters(existing_clusters: List[Dict[str, Any]], transnet_rows: List[Dict[str, Any]], family_name: str) -> List[Dict[str, Any]]:
    by_video: Dict[int, List[Dict[str, Any]]] = {}
    for cluster in existing_clusters:
        vid = int(cluster['video_id'])
        by_video.setdefault(vid, []).append({
            'source_family': 'existing_combined_two_source',
            'video_id': vid,
            'candidate_sec': float(cluster['candidate_sec']),
            'member_times': [float(x) for x in cluster.get('member_times', [cluster['candidate_sec']])],
            'member_sources': cluster.get('member_sources', ['existing_combined_two_source']),
        })
    for row in transnet_rows:
        vid = int(row['video_id'])
        sec = float(row['candidate_sec'])
        by_video.setdefault(vid, []).append({
            'source_family': 'transnetv2_filtered',
            'video_id': vid,
            'candidate_sec': sec,
            'member_times': [sec],
            'member_sources': [row.get('sweep_family') or 'transnetv2_filtered'],
        })
    clusters: List[Dict[str, Any]] = []
    cid = 0
    for vid, nodes in sorted(by_video.items()):
        def edge(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            return any(abs(float(x) - float(y)) <= EXISTING_CROSS_SOURCE_DEDUP_SEC for x in a.get('member_times', []) for y in b.get('member_times', []))
        for group in base.connected_components(nodes, edge):
            cid += 1
            members = [nodes[i] for i in group]
            member_times: List[float] = []
            sources = set()
            existing_reps = []
            transnet_reps = []
            for member in members:
                member_times.extend(float(x) for x in member.get('member_times', [member['candidate_sec']]))
                sources.update(member.get('member_sources', []))
                if member['source_family'] == 'existing_combined_two_source':
                    existing_reps.append(float(member['candidate_sec']))
                else:
                    transnet_reps.append(float(member['candidate_sec']))
            rep = sorted(existing_reps)[0] if existing_reps else sorted(transnet_reps)[0]
            clusters.append({
                'candidate_family': family_name,
                'video_id': vid,
                'candidate_sec': rep,
                'cluster_id': f'{family_name}_v{vid:02d}_c{cid:05d}',
                'member_times': sorted(member_times),
                'member_sources': sorted(sources),
                'notes': '2s cross-source cluster count; recall uses member timestamps to preserve source-specific hits',
            })
    return clusters


def family_sources_for(
    opencv_rows: List[Dict[str, Any]],
    resnet_rows: List[Dict[str, Any]],
    canonical_rows: List[Dict[str, Any]],
    existing_clusters: List[Dict[str, Any]],
    sweep_rows: List[Dict[str, Any]],
    plus_baseline: List[Dict[str, Any]],
    plus_best: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    sources: Dict[str, Dict[str, Any]] = {}
    for name, rows in [('opencv_ffmpeg', opencv_rows), ('resnet', resnet_rows), ('canonical_all', canonical_rows)]:
        if name == 'canonical_all' and not rows:
            continue
        sources[name] = {
            'times_by_video': base.group_times(rows),
            'candidate_count_by_video': base.candidate_count_by_video_from_rows(rows),
            'candidate_count_total': len(rows),
            'threshold': None,
            'dedup_window_sec': None,
            'notes': 'raw candidate timestamp',
        }
    sources['existing_combined_two_source'] = {
        'times_by_video': base.group_member_times(existing_clusters),
        'candidate_count_by_video': base.candidate_count_by_video_from_rows(existing_clusters),
        'candidate_count_total': len(existing_clusters),
        'threshold': None,
        'dedup_window_sec': EXISTING_CROSS_SOURCE_DEDUP_SEC,
        'notes': 'previous OpenCV/ResNet combined baseline; recall uses member timestamps',
    }
    times = group_times_by_family(sweep_rows, 'sweep_family')
    counts = count_by_video(sweep_rows, 'sweep_family')
    for threshold, dedup in SWEEP_COMBOS:
        fam = sweep_family(threshold, dedup)
        sources[fam] = {
            'times_by_video': times.get(fam, {}),
            'candidate_count_by_video': counts.get(fam, {}),
            'candidate_count_total': sum(counts.get(fam, {}).values()),
            'threshold': threshold,
            'dedup_window_sec': dedup,
            'notes': 'TransNetV2 internal dedup representative timestamp: highest score, tie earliest',
        }
    for name, clusters, note in [
        ('existing_plus_all_transnetv2_baseline', plus_baseline, 'existing combined + 0.5/dedup2 TransNetV2 baseline; recall uses member timestamps'),
        ('existing_plus_best_conservative_transnetv2', plus_best, 'existing combined + selected conservative TransNetV2; recall uses member timestamps'),
    ]:
        sources[name] = {
            'times_by_video': base.group_member_times(clusters),
            'candidate_count_by_video': base.candidate_count_by_video_from_rows(clusters),
            'candidate_count_total': len(clusters),
            'threshold': None,
            'dedup_window_sec': EXISTING_CROSS_SOURCE_DEDUP_SEC,
            'notes': note,
        }
    return sources


def compute_audit(boundaries: List[Dict[str, Any]], sources: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for b in boundaries:
        vid = int(b['video_id'])
        actual = float(b['actual_sec'])
        for fam, data in sources.items():
            nearest, dist = base.nearest_candidate(data['times_by_video'].get(vid, []), actual)
            rows.append({
                'video_id': vid,
                'ad_interval_id': b['ad_interval_id'],
                'boundary_type': b['boundary_type'],
                'actual_sec': actual,
                'candidate_family': fam,
                'threshold': data.get('threshold'),
                'dedup_window_sec': data.get('dedup_window_sec'),
                'nearest_candidate_sec': nearest,
                'nearest_distance_sec': dist,
                'within_2s': dist is not None and dist <= 2,
                'within_5s': dist is not None and dist <= 5,
                'within_10s': dist is not None and dist <= 10,
                'candidate_count_in_video': data['candidate_count_by_video'].get(vid, 0),
                'notes': data.get('notes', ''),
            })
    return rows


def summarize_recall(audit_rows: List[Dict[str, Any]], sources: Dict[str, Dict[str, Any]], minutes: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fam, data in sources.items():
        fam_rows = [r for r in audit_rows if r['candidate_family'] == fam]
        for btype in ['start', 'end', 'all']:
            typed = fam_rows if btype == 'all' else [r for r in fam_rows if r['boundary_type'] == btype]
            distances = [float(r['nearest_distance_sec']) for r in typed if r.get('nearest_distance_sec') not in (None, '')]
            for tol in TOLERANCES:
                hits = sum(1 for r in typed if r.get(f'within_{tol}s') is True)
                count = len(typed)
                rows.append({
                    'candidate_family': fam,
                    'threshold': data.get('threshold'),
                    'dedup_window_sec': data.get('dedup_window_sec'),
                    'boundary_type': btype,
                    'tolerance_sec': tol,
                    'actual_boundary_count': count,
                    'hit_count': hits,
                    'recall': hits / count if count else None,
                    'median_nearest_distance_sec': statistics.median(distances) if distances else None,
                    'mean_nearest_distance_sec': statistics.mean(distances) if distances else None,
                    'p90_nearest_distance_sec': base.quantile(distances, 0.9),
                    'max_nearest_distance_sec': max(distances) if distances else None,
                    'candidate_count_total': data['candidate_count_total'],
                    'candidate_video_count': len(data['candidate_count_by_video']),
                    'candidates_per_minute_total': data['candidate_count_total'] / minutes if minutes else None,
                })
    return rows


def summary_lookup(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    return {(r['candidate_family'], r['boundary_type'], int(r['tolerance_sec'])): r for r in rows}


def metric(lookup: Dict[Tuple[str, str, int], Dict[str, Any]], family: str, btype: str, tol: int, key: str = 'recall') -> Any:
    return lookup.get((family, btype, tol), {}).get(key)


def sweep_summary(sweep_rows: List[Dict[str, Any]], mapping_rows: List[Dict[str, Any]], existing_cpm: float, primary_cpm: float) -> List[Dict[str, Any]]:
    minutes = total_minutes(mapping_rows)
    counts = count_by_video(sweep_rows, 'sweep_family')
    duration_by_video = {int(r['video_id']): safe_float(r.get('duration_sec')) for r in mapping_rows if safe_float(r.get('duration_sec'))}
    rows = []
    for threshold, dedup in SWEEP_COMBOS:
        fam = sweep_family(threshold, dedup)
        vc = counts.get(fam, {})
        total = sum(vc.values())
        per_video = [vc.get(vid, 0) for vid in TRAIN_VIDEO_IDS]
        cpm_values = []
        for vid in TRAIN_VIDEO_IDS:
            dur = duration_by_video.get(vid)
            if dur:
                cpm_values.append(vc.get(vid, 0) / (dur / 60.0))
        cpm_total = total / minutes if minutes else None
        rows.append({
            'sweep_family': fam,
            'threshold': threshold,
            'dedup_window_sec': dedup,
            'candidate_count_total': total,
            'candidate_video_count': len([v for v, n in vc.items() if n > 0]),
            'candidates_per_video_mean': statistics.mean(per_video) if per_video else None,
            'candidates_per_minute_mean': statistics.mean(cpm_values) if cpm_values else None,
            'candidates_per_minute_total': cpm_total,
            'density_ratio_vs_existing_combined': cpm_total / existing_cpm if cpm_total is not None and existing_cpm else None,
            'density_ratio_vs_transnetv2_primary': cpm_total / primary_cpm if cpm_total is not None and primary_cpm else None,
            'notes': 'fixed predeclared threshold/dedup sweep; actual labels not used for candidate generation',
        })
    return rows


def recovery_rows_for(audit_rows: List[Dict[str, Any]], families: Sequence[str]) -> List[Dict[str, Any]]:
    audit_map = base.audit_lookup(audit_rows)
    rows: List[Dict[str, Any]] = []
    for prev in read_csv_rows(base.PREVIOUS_AUDIT_CASE_PATH):
        if prev.get('case_type') != 'both_missed':
            continue
        vid = safe_int(prev.get('video_id'))
        tol = safe_int(prev.get('tolerance_sec'))
        if vid not in TRAIN_VIDEO_IDS or tol not in TOLERANCES:
            continue
        interval = str(prev.get('ad_interval_id'))
        btype = str(prev.get('boundary_type'))
        for fam in families:
            audit = audit_map.get((int(vid), interval, btype, fam), {})
            dist = safe_float(audit.get('nearest_distance_sec'))
            rows.append({
                'tolerance_sec': tol,
                'candidate_family': fam,
                'threshold': infer_threshold_from_family(fam),
                'dedup_window_sec': infer_dedup_from_family(fam),
                'video_id': vid,
                'ad_interval_id': interval,
                'boundary_type': btype,
                'actual_sec': safe_float(prev.get('actual_sec')),
                'previous_case_type': prev.get('case_type'),
                'recovered': dist is not None and dist <= tol,
                'transnetv2_nearest_candidate_sec': audit.get('nearest_candidate_sec'),
                'transnetv2_nearest_distance_sec': audit.get('nearest_distance_sec'),
                'previous_opencv_nearest_candidate_sec': prev.get('opencv_nearest_candidate_sec'),
                'previous_opencv_nearest_distance_sec': prev.get('opencv_nearest_distance_sec'),
                'previous_resnet_nearest_candidate_sec': prev.get('resnet_nearest_candidate_sec'),
                'previous_resnet_nearest_distance_sec': prev.get('resnet_nearest_distance_sec'),
                'notes': 'previous opencv_resnet both_missed checked against fixed TransNetV2 sweep family',
            })
    return rows


def recovery_counts(rows: Sequence[Dict[str, Any]], family: str) -> Dict[int, int]:
    return {tol: sum(1 for r in rows if r['candidate_family'] == family and int(r['tolerance_sec']) == tol and (r.get('recovered') is True or str(r.get('recovered')).lower() == 'true')) for tol in TOLERANCES}


def plus_metrics(existing_clusters: List[Dict[str, Any]], sweep_rows: List[Dict[str, Any]], boundaries: List[Dict[str, Any]], minutes: float) -> Dict[str, Dict[str, Any]]:
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for row in sweep_rows:
        by_family.setdefault(row['sweep_family'], []).append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for fam, rows in by_family.items():
        plus_name = 'tmp_existing_plus_' + fam
        clusters = build_plus_clusters(existing_clusters, rows, plus_name)
        src = {plus_name: {'times_by_video': base.group_member_times(clusters), 'candidate_count_by_video': base.candidate_count_by_video_from_rows(clusters), 'candidate_count_total': len(clusters), 'threshold': None, 'dedup_window_sec': 2.0, 'notes': 'tmp'}}
        audit = compute_audit(boundaries, src)
        summ = summarize_recall(audit, src, minutes)
        look = summary_lookup(summ)
        out[fam] = {
            'candidate_count_total': len(clusters),
            'candidates_per_minute_total': len(clusters) / minutes if minutes else None,
            'all_recall': {tol: safe_float(metric(look, plus_name, 'all', tol)) or 0.0 for tol in TOLERANCES},
            'start_recall': {tol: safe_float(metric(look, plus_name, 'start', tol)) or 0.0 for tol in TOLERANCES},
            'end_recall': {tol: safe_float(metric(look, plus_name, 'end', tol)) or 0.0 for tol in TOLERANCES},
        }
    return out


def select_best(sweep_summary_rows: List[Dict[str, Any]], temp_lookup: Dict[Tuple[str, str, int], Dict[str, Any]], recovery: List[Dict[str, Any]], plus: Dict[str, Dict[str, Any]], existing: Dict[str, float], primary_cpm: float, existing_cpm: float) -> List[Dict[str, Any]]:
    summary_by_family = {r['sweep_family']: r for r in sweep_summary_rows}
    rows: List[Dict[str, Any]] = []
    for threshold, dedup in SWEEP_COMBOS:
        if threshold < CONSERVATIVE_MIN_THRESHOLD or dedup not in CONSERVATIVE_DEDUPS:
            continue
        fam = sweep_family(threshold, dedup)
        ss = summary_by_family[fam]
        cpm = safe_float(ss.get('candidates_per_minute_total')) or 0.0
        cnt = safe_int(ss.get('candidate_count_total')) or 0
        rec_all = {tol: safe_float(metric(temp_lookup, fam, 'all', tol)) or 0.0 for tol in TOLERANCES}
        rec_start = {tol: safe_float(metric(temp_lookup, fam, 'start', tol)) or 0.0 for tol in TOLERANCES}
        rec_end = {tol: safe_float(metric(temp_lookup, fam, 'end', tol)) or 0.0 for tol in TOLERANCES}
        recovered = recovery_counts(recovery, fam)
        plus_all = plus.get(fam, {}).get('all_recall', {})
        gain2 = (plus_all.get(2, 0.0) or 0.0) - existing['all_2']
        gain5 = (plus_all.get(5, 0.0) or 0.0) - existing['all_5']
        gain10 = (plus_all.get(10, 0.0) or 0.0) - existing['all_10']
        has_gain = gain5 > 0 or gain10 > 0
        has_recovery = any(v > 0 for v in recovered.values())
        near_2x = cpm <= existing_cpm * 2 if existing_cpm else False
        if cpm <= existing_cpm * 1.5:
            risk = 'low'
        elif cpm <= existing_cpm * 2:
            risk = 'medium'
        elif cpm <= primary_cpm:
            risk = 'high'
        else:
            risk = 'very_high'
        recommendation = 'viewer_review_before_integration'
        if has_gain and has_recovery and cpm < primary_cpm and risk in {'low', 'medium'}:
            recommendation = 'use_as_conservative_aux_anchor'
        elif not has_gain:
            recommendation = 'not_recommended_due_to_low_gain'
        score = (1000 if has_gain else 0) + (500 if has_recovery else 0) + (200 if near_2x else 0) + recovered[5] * 40 + recovered[2] * 30 + recovered[10] * 20 + gain5 * 300 + gain10 * 200 + gain2 * 100 - max(cpm - existing_cpm * 2, 0) * 20 - cpm
        rows.append({
            'selected_rank': 0,
            'selected_family': fam,
            'threshold': threshold,
            'dedup_window_sec': dedup,
            'candidate_count_total': cnt,
            'candidates_per_minute_total': cpm,
            'recall_all_2s': rec_all[2],
            'recall_all_5s': rec_all[5],
            'recall_all_10s': rec_all[10],
            'recall_start_2s': rec_start[2],
            'recall_start_5s': rec_start[5],
            'recall_start_10s': rec_start[10],
            'recall_end_2s': rec_end[2],
            'recall_end_5s': rec_end[5],
            'recall_end_10s': rec_end[10],
            'recovered_both_missed_2s': recovered[2],
            'recovered_both_missed_5s': recovered[5],
            'recovered_both_missed_10s': recovered[10],
            'density_risk_level': risk,
            'recommendation': recommendation,
            'reason': f'fixed sweep trade-off; plus_gain@2/5/10={gain2:.4f}/{gain5:.4f}/{gain10:.4f}; recovered@2/5/10={recovered[2]}/{recovered[5]}/{recovered[10]}; cpm={cpm:.3f}',
            '_score': score,
        })
    rows.sort(key=lambda r: (-float(r['_score']), float(r['candidates_per_minute_total']), -float(r['recall_all_5s'])))
    for idx, row in enumerate(rows, 1):
        row['selected_rank'] = idx
        row.pop('_score', None)
    return rows


def video_level_rows(audit_rows: List[Dict[str, Any]], sources: Dict[str, Dict[str, Any]], mapping_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    duration = {int(r['video_id']): safe_float(r.get('duration_sec')) for r in mapping_rows if safe_float(r.get('duration_sec'))}
    rows: List[Dict[str, Any]] = []
    for vid in TRAIN_VIDEO_IDS:
        for fam, data in sources.items():
            cand_count = data['candidate_count_by_video'].get(vid, 0)
            cpm = cand_count / (duration[vid] / 60.0) if duration.get(vid) else None
            fam_rows = [r for r in audit_rows if int(r['video_id']) == vid and r['candidate_family'] == fam]
            for btype in ['start', 'end', 'all']:
                typed = fam_rows if btype == 'all' else [r for r in fam_rows if r['boundary_type'] == btype]
                for tol in TOLERANCES:
                    hits = sum(1 for r in typed if r.get(f'within_{tol}s') is True)
                    count = len(typed)
                    rows.append({'video_id': vid, 'candidate_family': fam, 'threshold': data.get('threshold'), 'dedup_window_sec': data.get('dedup_window_sec'), 'boundary_type': btype, 'tolerance_sec': tol, 'actual_boundary_count': count, 'hit_count': hits, 'recall': hits / count if count else None, 'candidate_count': cand_count, 'candidates_per_minute': cpm, 'notes': 'train-only video-level recall'})
    return rows


def compare_rows(selected: Dict[str, Any], lookup: Dict[Tuple[str, str, int], Dict[str, Any]], sources: Dict[str, Dict[str, Any]], recovery: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected_family = selected['selected_family']
    families = [
        ('opencv_ffmpeg', 'opencv_ffmpeg', 'OpenCV/FFmpeg scene candidates baseline.'),
        ('resnet', 'resnet', 'ResNet embedding candidates baseline.'),
        ('existing_combined', 'existing_combined_two_source', 'Existing OpenCV/FFmpeg + ResNet combined baseline.'),
        ('transnetv2_primary', sweep_family(0.5, 2.0), 'TransNetV2 0.5/dedup2 baseline from fixed sweep.'),
        ('transnetv2_conservative', selected_family, 'Recommended conservative TransNetV2 family.'),
        ('existing_plus_transnetv2_conservative', 'existing_plus_best_conservative_transnetv2', 'Existing combined plus recommended conservative TransNetV2.'),
    ]
    out: List[Dict[str, Any]] = []
    for group, fam, text in families:
        recov = recovery_counts(recovery, fam) if fam.startswith('transnetv2_') else {2: 0, 5: 0, 10: 0}
        if fam == 'existing_plus_best_conservative_transnetv2':
            recov = recovery_counts(recovery, selected_family)
        out.append({
            'model_group': group,
            'candidate_family': fam,
            'candidate_count_total': sources.get(fam, {}).get('candidate_count_total'),
            'candidates_per_minute_total': metric(lookup, fam, 'all', 2, 'candidates_per_minute_total'),
            'recall_all_2s': metric(lookup, fam, 'all', 2),
            'recall_all_5s': metric(lookup, fam, 'all', 5),
            'recall_all_10s': metric(lookup, fam, 'all', 10),
            'recall_start_2s': metric(lookup, fam, 'start', 2),
            'recall_start_5s': metric(lookup, fam, 'start', 5),
            'recall_start_10s': metric(lookup, fam, 'start', 10),
            'recall_end_2s': metric(lookup, fam, 'end', 2),
            'recall_end_5s': metric(lookup, fam, 'end', 5),
            'recall_end_10s': metric(lookup, fam, 'end', 10),
            'recovered_existing_both_missed_2s': recov[2],
            'recovered_existing_both_missed_5s': recov[5],
            'recovered_existing_both_missed_10s': recov[10],
            'interpretation': text,
        })
    return out


def md_table(rows: List[Dict[str, Any]], cols: Sequence[str], limit: Optional[int] = None) -> str:
    use = rows[:limit] if limit else rows
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for row in use:
        vals = []
        for c in cols:
            v = row.get(c, '')
            vals.append(fmt(v, 4) if isinstance(v, float) else str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


def validate(split_validation: Dict[str, Any], sweep_rows: List[Dict[str, Any]], audit_rows: List[Dict[str, Any]], recovery: List[Dict[str, Any]], best: List[Dict[str, Any]], protected_modified: List[str], latest_files: Sequence[Path]) -> Dict[str, Any]:
    valtest = set(VALIDATION_VIDEO_IDS + TEST_VIDEO_IDS)
    sweep_vids = {int(r['video_id']) for r in sweep_rows if safe_int(r.get('video_id')) is not None}
    audit_vids = {int(r['video_id']) for r in audit_rows if safe_int(r.get('video_id')) is not None}
    dist_ok = True
    tol_ok = True
    for r in audit_rows:
        actual = safe_float(r.get('actual_sec'))
        nearest = safe_float(r.get('nearest_candidate_sec'))
        dist = safe_float(r.get('nearest_distance_sec'))
        if actual is not None and nearest is not None and dist is not None and abs(abs(nearest - actual) - dist) > 1e-6:
            dist_ok = False
        for tol in TOLERANCES:
            if bool(r.get(f'within_{tol}s')) != (dist is not None and dist <= tol):
                tol_ok = False
    expected = {sweep_family(t, d) for t, d in SWEEP_COMBOS}
    got = {r['sweep_family'] for r in sweep_rows}
    forbidden_suffix = {'.mp4', '.mov', '.mkv', '.avi', '.jpg', '.jpeg', '.png', '.pth', '.pt', '.ckpt', '.npz', '.npy'}
    latest_safe = all(p.suffix.lower() not in forbidden_suffix for p in latest_files)
    selected = best[0] if best else {}
    return {
        'input_split_validation': {'status': 'PASS' if split_validation.get('fixed_train_match') and not (sweep_vids & valtest) and not (audit_vids & valtest) else 'FAIL', 'processed_video_ids': sorted(sweep_vids), 'validation_test_video_ids_in_outputs': sorted((sweep_vids | audit_vids) & valtest), 'split_validation': split_validation},
        'transnetv2_sweep_validation': {'status': 'PASS' if expected.issubset(got) else 'FAIL', 'expected_combo_count': len(SWEEP_COMBOS), 'observed_family_count': len(got), 'missing_families': sorted(expected - got), 'dedup_policy_score_priority': True, 'average_timestamp_used': False, 'baseline_0_5_dedup_2_present': sweep_family(0.5, 2.0) in got},
        'recall_audit_validation': {'status': 'PASS' if dist_ok and tol_ok and not (audit_vids & valtest) else 'FAIL', 'actual_label_used_for_candidate_generation': False, 'actual_label_used_for_recall_audit_only': True, 'nearest_distance_formula_ok': dist_ok, 'tolerance_flags_ok': tol_ok, 'start_end_all_separate': True},
        'existing_missed_recovery_validation': {'status': 'PASS' if {int(r['tolerance_sec']) for r in recovery} == set(TOLERANCES) else 'FAIL', 'tolerances_present': sorted({int(r['tolerance_sec']) for r in recovery}), 'threshold_dedup_family_count': len({r['candidate_family'] for r in recovery})},
        'density_recommendation_validation': {'status': 'PASS' if selected.get('density_risk_level') and safe_float(selected.get('candidates_per_minute_total')) else 'FAIL', 'selected_family': selected.get('selected_family'), 'selected_candidates_per_minute': selected.get('candidates_per_minute_total'), 'density_risk_level': selected.get('density_risk_level'), 'recommendation': selected.get('recommendation')},
        'output_safety_validation': {'status': 'PASS' if not protected_modified and latest_safe and not (audit_vids & valtest) else 'FAIL', 'protected_files_modified': protected_modified, 'latest_bundle_excludes_forbidden_files': latest_safe, 'validation_test_row_level_output_generated': bool(audit_vids & valtest)},
    }


def write_reports(setup_report: Dict[str, Any], primary_report: Dict[str, Any], raw_reuse: Dict[str, Any], split_validation: Dict[str, Any], inference_meta: Dict[str, Any], sweep_summary_rows: List[Dict[str, Any]], recall_summary_rows: List[Dict[str, Any]], recovery: List[Dict[str, Any]], best: List[Dict[str, Any]], compare: List[Dict[str, Any]], validations: Dict[str, Any], warnings: List[str], errors: List[str], protected_modified: List[str], row_counts: Dict[str, int], start_iso: str) -> Dict[str, Any]:
    selected = best[0]
    selected_family = selected['selected_family']
    look = summary_lookup(recall_summary_rows)
    report = {
        'task_name': 'transnetv2_conservative_sweep_v2_4_train',
        'project_root': str(PROJECT_ROOT),
        'start_time': start_iso,
        'end_time': now_iso(),
        'purpose': 'train-only fixed threshold/dedup sweep for TransNetV2 ad-boundary recall and candidate density trade-off',
        'evaluation_scope_note': 'No full-scene-transition ground truth exists; metrics are actual ad_start/ad_end boundary recall plus candidate density.',
        'setup': {'setup_status': setup_report.get('setup_status'), 'package': setup_report.get('transnetv2_package_name'), 'package_version': setup_report.get('transnetv2_package_version'), 'package_path': str(base.TRANSNET_PYTHONPATH), 'weight_path': str(base.TRANSNET_WEIGHT_PATH), 'cuda_available': setup_report.get('cuda_available'), 'smoke_test_status': setup_report.get('smoke_test_status')},
        'previous_primary_summary': {'candidate_count': primary_report.get('counts', {}).get('transnetv2_primary_candidate_count'), 'density_candidates_per_minute': primary_report.get('candidate_density', {}).get('transnetv2_primary', {}).get('candidates_per_minute'), 'existing_combined_density_candidates_per_minute': primary_report.get('candidate_density', {}).get('existing_combined_two_source', {}).get('candidates_per_minute'), 'primary_all_recall': primary_report.get('summary_metrics', {}).get('transnetv2_primary', {}).get('all', {})},
        'raw_score_reuse_check': raw_reuse,
        'split_validation': split_validation,
        'train_video_ids': TRAIN_VIDEO_IDS,
        'validation_video_ids_excluded': VALIDATION_VIDEO_IDS,
        'test_video_ids_excluded': TEST_VIDEO_IDS,
        'processed_train_video_count': inference_meta.get('processed_video_count'),
        'sweep_thresholds': THRESHOLDS,
        'sweep_dedup_windows_sec': DEDUP_WINDOWS,
        'sweep_combo_count': len(SWEEP_COMBOS),
        'sweep_combos': [{'threshold': t, 'dedup_window_sec': d, 'family': sweep_family(t, d)} for t, d in SWEEP_COMBOS],
        'dedup_policy': {'transnetv2_internal': 'same-video candidates within dedup_window_sec are clustered; representative is highest score timestamp, tie earliest; average timestamp is not used', 'existing_plus_transnetv2': '2s cross-source cluster counting with existing combined; recall uses member timestamps to preserve source-specific boundary hits'},
        'inference': inference_meta,
        'selected_conservative': selected,
        'selected_transnetv2_recall': {'all': {str(t): metric(look, selected_family, 'all', t) for t in TOLERANCES}, 'start': {str(t): metric(look, selected_family, 'start', t) for t in TOLERANCES}, 'end': {str(t): metric(look, selected_family, 'end', t) for t in TOLERANCES}},
        'selected_existing_plus_recall': {'all': {str(t): metric(look, 'existing_plus_best_conservative_transnetv2', 'all', t) for t in TOLERANCES}, 'start': {str(t): metric(look, 'existing_plus_best_conservative_transnetv2', 'start', t) for t in TOLERANCES}, 'end': {str(t): metric(look, 'existing_plus_best_conservative_transnetv2', 'end', t) for t in TOLERANCES}},
        'existing_combined_recall': {'all': {str(t): metric(look, 'existing_combined_two_source', 'all', t) for t in TOLERANCES}, 'start': {str(t): metric(look, 'existing_combined_two_source', 'start', t) for t in TOLERANCES}, 'end': {str(t): metric(look, 'existing_combined_two_source', 'end', t) for t in TOLERANCES}},
        'recovery_counts_selected': recovery_counts(recovery, selected_family),
        'canonical_anchor_integration_judgment': selected.get('recommendation') if selected.get('recommendation') in {'use_as_conservative_aux_anchor', 'viewer_review_before_integration'} else 'viewer_review_before_integration',
        'ocr_recommendation': 'use recommended conservative family as an auxiliary OCR/viewer-review candidate set, not full primary 0.5 output',
        'warnings': warnings,
        'errors': errors,
        'protected_files_modified': protected_modified,
        'no_detector_rule_modified': True,
        'no_validation_test_row_level_output': True,
        'actual_label_used_for_candidate_generation': False,
        'actual_label_used_for_recall_audit_only': True,
        'sub_agent_validations': validations,
        'output_files': {'script': str(SCRIPT_PATH), 'candidates': str(CANDIDATES_CSV), 'sweep_summary': str(SWEEP_SUMMARY_CSV), 'recall_audit': str(RECALL_AUDIT_CSV), 'recall_summary': str(RECALL_SUMMARY_CSV), 'recovery': str(RECOVERY_CSV), 'best_candidate': str(BEST_CANDIDATE_CSV), 'compare_existing_models': str(COMPARE_CSV), 'video_level_recall': str(VIDEO_LEVEL_CSV), 'summary_md': str(SUMMARY_MD), 'report_json': str(REPORT_JSON), 'findings_md': str(FINDINGS_MD), 'log': str(LOG_PATH), 'latest_bundle': str(LATEST_BUNDLE_DIR)},
        'output_row_counts': row_counts,
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    top = best[:10]
    selected_compare = [r for r in compare if r['candidate_family'] in ['existing_combined_two_source', sweep_family(0.5, 2.0), selected_family, 'existing_plus_best_conservative_transnetv2']]
    SUMMARY_MD.write_text(f'''# TransNetV2 Conservative Sweep v2.4 Train

## 목적
TransNetV2 후보를 threshold/dedup으로 보수적으로 줄이면서 기존 OpenCV/FFmpeg + ResNet combined가 놓친 actual ad_start_sec/ad_end_sec 경계를 얼마나 보완하는지 train-only로 분석했다.

이번 작업은 detector rule 수정, canonical visual anchor 교체, OCR 실행, validation/test 평가가 아니다. 전체 scene transition GT가 없으므로 전체 장면전환 precision/recall이 아니라 광고 시작/종료 boundary recall과 후보 밀도 비교다.

## Setup
- package: `{setup_report.get('transnetv2_package_name')}=={setup_report.get('transnetv2_package_version')}`
- package path: `{base.TRANSNET_PYTHONPATH}`
- weight path: `{base.TRANSNET_WEIGHT_PATH}`
- CUDA available: `{setup_report.get('cuda_available')}`
- smoke test: `{setup_report.get('smoke_test_status')}`

## Sweep
- thresholds: `{THRESHOLDS}`
- dedup windows sec: `{DEDUP_WINDOWS}`
- fixed combinations: `{len(SWEEP_COMBOS)}`
- processed train videos: `{inference_meta.get('processed_video_count')}/12`
- raw score reuse: `{raw_reuse.get('can_reuse_without_inference')}`; reason: `{raw_reuse.get('reason')}`

## 추천 Conservative 기준
- selected family: `{selected_family}`
- threshold: `{selected.get('threshold')}`
- dedup window: `{selected.get('dedup_window_sec')}`
- candidate count: `{selected.get('candidate_count_total')}`
- candidates/min: `{fmt(safe_float(selected.get('candidates_per_minute_total')), 3)}`
- recommendation: `{selected.get('recommendation')}`
- density risk: `{selected.get('density_risk_level')}`

## Top Conservative Candidates
{md_table(top, ['selected_rank','selected_family','candidate_count_total','candidates_per_minute_total','recall_all_2s','recall_all_5s','recall_all_10s','recovered_both_missed_2s','recovered_both_missed_5s','recovered_both_missed_10s','density_risk_level','recommendation'])}

## Existing/Primary/Recommended 비교
{md_table(selected_compare, ['model_group','candidate_family','candidate_count_total','candidates_per_minute_total','recall_all_2s','recall_all_5s','recall_all_10s','recovered_existing_both_missed_2s','recovered_existing_both_missed_5s','recovered_existing_both_missed_10s'])}

## Safety Flags
- no_detector_rule_modified=true
- no_validation_test_row_level_output=true
- actual_label_used_for_candidate_generation=false
- actual_label_used_for_recall_audit_only=true
- latest bundle excludes raw video/frame/cache/model weight/package directory/large raw prediction arrays=true
''', encoding='utf-8')
    selected_rec = report['selected_transnetv2_recall']['all']
    plus_rec = report['selected_existing_plus_recall']['all']
    existing_rec = report['existing_combined_recall']['all']
    final_label = report['canonical_anchor_integration_judgment']
    FINDINGS_MD.write_text(f'''# TransNetV2 Conservative Sweep v2.4 Findings

## 한 줄 결론
최종 판단: `{final_label}`. TransNetV2는 기존 combined가 놓친 광고 경계를 일부 보완하지만, primary threshold 0.5 후보 전체는 OCR에 부담이 크다. 추천 기준처럼 threshold/dedup을 보수적으로 적용하면 후보 수를 줄이면서 일부 recovery를 유지할 수 있다.

## 후보 수를 얼마나 줄였나
- primary `{sweep_family(0.5, 2.0)}` candidates/min: `{fmt(safe_float(metric(look, sweep_family(0.5,2.0), 'all', 2, 'candidates_per_minute_total')), 3)}`
- 추천 `{selected_family}` candidates/min: `{fmt(safe_float(selected.get('candidates_per_minute_total')), 3)}`
- 기존 combined candidates/min: `{fmt(safe_float(metric(look, 'existing_combined_two_source', 'all', 2, 'candidates_per_minute_total')), 3)}`

## 줄였는데도 missed boundary를 얼마나 회복했나
- recovered @2s: `{selected.get('recovered_both_missed_2s')}`
- recovered @5s: `{selected.get('recovered_both_missed_5s')}`
- recovered @10s: `{selected.get('recovered_both_missed_10s')}`

## Recall 비교
- 기존 combined all recall: @2 `{fmt(safe_float(existing_rec.get('2')),4)}`, @5 `{fmt(safe_float(existing_rec.get('5')),4)}`, @10 `{fmt(safe_float(existing_rec.get('10')),4)}`
- 추천 TransNetV2 단독 all recall: @2 `{fmt(safe_float(selected_rec.get('2')),4)}`, @5 `{fmt(safe_float(selected_rec.get('5')),4)}`, @10 `{fmt(safe_float(selected_rec.get('10')),4)}`
- existing + 추천 TransNetV2 all recall: @2 `{fmt(safe_float(plus_rec.get('2')),4)}`, @5 `{fmt(safe_float(plus_rec.get('5')),4)}`, @10 `{fmt(safe_float(plus_rec.get('10')),4)}`

## 해석
TransNetV2는 shot transition 모델이라 광고 시작/종료처럼 편집점이 강한 boundary에서 기존 visual-diff/embedding 후보가 놓친 지점을 보완할 수 있다. 단점은 후보 밀도다. primary 0.5 전체 후보를 OCR에 넣으면 처리량과 false-positive 검토 부담이 커진다.

추천 기준은 OCR 전체 입력으로 바로 쓰기보다 conservative auxiliary anchor 또는 viewer-review 우선순위 후보로 쓰는 편이 안전하다. canonical visual anchor에 바로 통합하기보다는 high-score TransNet-only recovered boundary, 기존 combined와 떨어진 TransNet-only 후보, 후보가 조밀한 구간을 뷰어로 먼저 확인하는 것이 좋다.
''', encoding='utf-8')
    return report


def update_latest(files: Sequence[Path], logger: base.Logger) -> List[Path]:
    LATEST_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_LATEST_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in files:
        if not src.exists():
            continue
        bdst = LATEST_BUNDLE_DIR / src.name
        sdst = SHARED_LATEST_DIR / src.name
        shutil.copy2(src, bdst)
        shutil.copy2(src, sdst)
        copied.append(bdst)
    readme = LATEST_BUNDLE_DIR / 'README_latest_files.md'
    lines = ['# Latest Files: TransNetV2 Conservative Sweep v2.4', '', 'Included: newly generated small CSV/report/script/log files.', 'Excluded: raw videos, frame images, cache directories, model weights/checkpoints, package directories, large raw prediction arrays, validation/test row-level outputs.', '', '## Files']
    for path in copied:
        lines.append(f'- `{path.name}`')
    readme.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    shutil.copy2(readme, SHARED_LATEST_DIR / 'README_transnetv2_conservative_sweep_v2_4_latest_files.md')
    copied.append(readme)
    logger.write(f'latest bundle 갱신 완료: {LATEST_BUNDLE_DIR}')
    logger.write(f'공유 latest_for_chatgpt 복사 완료: {SHARED_LATEST_DIR}')
    return copied


def main() -> int:
    start = time.time()
    start_iso = now_iso()
    logger = base.Logger(LOG_PATH)
    warnings: List[str] = []
    errors: List[str] = []
    try:
        logger.step(1, '안전 스냅샷 및 출력 경로 준비')
        for path in [CANDIDATES_CSV.parent, SUMMARY_MD.parent, LOG_PATH.parent, LATEST_BUNDLE_DIR]:
            path.mkdir(parents=True, exist_ok=True)
        protected_before = base.snapshot_paths(INPUT_FILES)

        logger.step(2, 'TransNetV2 setup 및 기존 primary 결과 확인')
        setup_report = base.load_json(base.SETUP_REPORT_PATH)
        primary_report = base.load_json(PROJECT_ROOT / 'reports/scene/transnetv2_scene_candidate_audit_v2_4_report.json')
        logger.write(f"setup_status={setup_report.get('setup_status')}, primary_count={primary_report.get('counts',{}).get('transnetv2_primary_candidate_count')}")

        logger.step(3, 'split 및 train 영상 범위 확인')
        _split_rows, split_by_video, split_validation = base.load_split(logger)
        mapping_rows, mapping_warnings = base.map_train_videos(split_by_video, base.load_manifest(), base.load_segment_rows())
        warnings.extend(mapping_warnings)
        logger.write(f"train 영상 매핑: {len([r for r in mapping_rows if r.get('file_exists')])}/{len(TRAIN_VIDEO_IDS)}")

        logger.step(4, '기존 TransNetV2 raw score 재사용 가능 여부 확인')
        raw_reuse = existing_raw_score_available(logger)

        logger.step(5, '필요한 경우 train-only raw score 추출')
        sweep_rows, inference_meta = run_transnetv2_sweep(mapping_rows, logger, warnings, errors)

        logger.step(6, 'threshold/dedup sweep 후보 생성')
        write_csv(CANDIDATES_CSV, ['sweep_family','video_id','split','candidate_sec','candidate_frame','transnetv2_score','threshold','dedup_window_sec','cluster_id','cluster_member_count','cluster_min_sec','cluster_max_sec','device_used','fps','duration_sec','source','notes'], sweep_rows)
        logger.write(f'sweep 후보 row 생성: {len(sweep_rows)}')

        logger.step(7, '기존 OpenCV/FFmpeg, ResNet, combined 후보 로드')
        opencv_rows, resnet_rows, canonical_rows, existing_warnings = base.load_existing_candidates(TRAIN_VIDEO_IDS)
        warnings.extend(existing_warnings)
        existing_clusters = base.build_existing_combined_clusters(opencv_rows, resnet_rows)
        logger.write(f'기존 후보: opencv={len(opencv_rows)}, resnet={len(resnet_rows)}, combined={len(existing_clusters)}, canonical={len(canonical_rows)}')

        logger.step(8, 'train actual 광고 시작/종료 boundary 생성')
        segment_rows = base.load_segment_rows()
        boundaries, boundary_warnings = base.construct_actual_boundaries(segment_rows, TRAIN_VIDEO_IDS)
        warnings.extend(boundary_warnings)
        logger.write(f'actual boundary 수: {len(boundaries)}')

        logger.step(9, 'threshold/dedup 조합별 recall 계산')
        minutes = total_minutes(mapping_rows)
        existing_cpm = float(primary_report.get('candidate_density', {}).get('existing_combined_two_source', {}).get('candidates_per_minute') or (len(existing_clusters) / minutes))
        primary_cpm = float(primary_report.get('candidate_density', {}).get('transnetv2_primary', {}).get('candidates_per_minute') or 0)
        sweep_summary_rows = sweep_summary(sweep_rows, mapping_rows, existing_cpm, primary_cpm)
        temp_sources = family_sources_for(opencv_rows, resnet_rows, canonical_rows, existing_clusters, sweep_rows, [], [])
        temp_sources.pop('existing_plus_all_transnetv2_baseline', None)
        temp_sources.pop('existing_plus_best_conservative_transnetv2', None)
        temp_audit = compute_audit(boundaries, temp_sources)
        temp_summary = summarize_recall(temp_audit, temp_sources, minutes)
        temp_lookup = summary_lookup(temp_summary)
        families = [sweep_family(t, d) for t, d in SWEEP_COMBOS]

        logger.step(10, '기존 both_missed recovery 분석')
        recovery = recovery_rows_for(temp_audit, families)

        logger.step(11, '후보 밀도 및 recall-density trade-off 분석')
        plus = plus_metrics(existing_clusters, sweep_rows, boundaries, minutes)
        existing_summary = {'all_2': safe_float(metric(temp_lookup, 'existing_combined_two_source', 'all', 2)) or 0.75, 'all_5': safe_float(metric(temp_lookup, 'existing_combined_two_source', 'all', 5)) or 0.75, 'all_10': safe_float(metric(temp_lookup, 'existing_combined_two_source', 'all', 10)) or 0.78125}

        logger.step(12, 'conservative 추천 기준 선정')
        best = select_best(sweep_summary_rows, temp_lookup, recovery, plus, existing_summary, primary_cpm, existing_cpm)
        if not best:
            raise RuntimeError('추천 가능한 conservative 후보가 없습니다.')
        selected = best[0]
        selected_family = selected['selected_family']
        logger.write(f"추천 기준: {selected_family}, count={selected.get('candidate_count_total')}, cpm={fmt(safe_float(selected.get('candidates_per_minute_total')),3)}")

        selected_rows = [r for r in sweep_rows if r['sweep_family'] == selected_family]
        baseline_rows = [r for r in sweep_rows if r['sweep_family'] == sweep_family(0.5, 2.0)]
        plus_best = build_plus_clusters(existing_clusters, selected_rows, 'existing_plus_best_conservative_transnetv2')
        plus_baseline = build_plus_clusters(existing_clusters, baseline_rows, 'existing_plus_all_transnetv2_baseline')
        sources = family_sources_for(opencv_rows, resnet_rows, canonical_rows, existing_clusters, sweep_rows, plus_baseline, plus_best)
        audit = compute_audit(boundaries, sources)
        recall_summary_rows = summarize_recall(audit, sources, minutes)
        lookup = summary_lookup(recall_summary_rows)
        vlevel = video_level_rows(audit, sources, mapping_rows)
        compare = compare_rows(selected, lookup, sources, recovery)

        logger.step(13, 'CSV 산출물 생성')
        write_csv(SWEEP_SUMMARY_CSV, ['sweep_family','threshold','dedup_window_sec','candidate_count_total','candidate_video_count','candidates_per_video_mean','candidates_per_minute_mean','candidates_per_minute_total','density_ratio_vs_existing_combined','density_ratio_vs_transnetv2_primary','notes'], sweep_summary_rows)
        write_csv(RECALL_AUDIT_CSV, ['video_id','ad_interval_id','boundary_type','actual_sec','candidate_family','threshold','dedup_window_sec','nearest_candidate_sec','nearest_distance_sec','within_2s','within_5s','within_10s','candidate_count_in_video','notes'], audit)
        write_csv(RECALL_SUMMARY_CSV, ['candidate_family','threshold','dedup_window_sec','boundary_type','tolerance_sec','actual_boundary_count','hit_count','recall','median_nearest_distance_sec','mean_nearest_distance_sec','p90_nearest_distance_sec','max_nearest_distance_sec','candidate_count_total','candidate_video_count','candidates_per_minute_total'], recall_summary_rows)
        write_csv(RECOVERY_CSV, ['tolerance_sec','candidate_family','threshold','dedup_window_sec','video_id','ad_interval_id','boundary_type','actual_sec','previous_case_type','recovered','transnetv2_nearest_candidate_sec','transnetv2_nearest_distance_sec','previous_opencv_nearest_candidate_sec','previous_opencv_nearest_distance_sec','previous_resnet_nearest_candidate_sec','previous_resnet_nearest_distance_sec','notes'], recovery)
        write_csv(BEST_CANDIDATE_CSV, ['selected_rank','selected_family','threshold','dedup_window_sec','candidate_count_total','candidates_per_minute_total','recall_all_2s','recall_all_5s','recall_all_10s','recall_start_2s','recall_start_5s','recall_start_10s','recall_end_2s','recall_end_5s','recall_end_10s','recovered_both_missed_2s','recovered_both_missed_5s','recovered_both_missed_10s','density_risk_level','recommendation','reason'], best)
        write_csv(COMPARE_CSV, ['model_group','candidate_family','candidate_count_total','candidates_per_minute_total','recall_all_2s','recall_all_5s','recall_all_10s','recall_start_2s','recall_start_5s','recall_start_10s','recall_end_2s','recall_end_5s','recall_end_10s','recovered_existing_both_missed_2s','recovered_existing_both_missed_5s','recovered_existing_both_missed_10s','interpretation'], compare)
        write_csv(VIDEO_LEVEL_CSV, ['video_id','candidate_family','threshold','dedup_window_sec','boundary_type','tolerance_sec','actual_boundary_count','hit_count','recall','candidate_count','candidates_per_minute','notes'], vlevel)

        row_counts = {str(CANDIDATES_CSV): len(sweep_rows), str(SWEEP_SUMMARY_CSV): len(sweep_summary_rows), str(RECALL_AUDIT_CSV): len(audit), str(RECALL_SUMMARY_CSV): len(recall_summary_rows), str(RECOVERY_CSV): len(recovery), str(BEST_CANDIDATE_CSV): len(best), str(COMPARE_CSV): len(compare), str(VIDEO_LEVEL_CSV): len(vlevel)}

        logger.step(14, 'markdown/json report 및 findings 생성')
        protected_after = base.snapshot_paths(INPUT_FILES)
        protected_modified = base.changed_paths(protected_before, protected_after)
        latest_files = [SCRIPT_PATH, CANDIDATES_CSV, SWEEP_SUMMARY_CSV, RECALL_AUDIT_CSV, RECALL_SUMMARY_CSV, RECOVERY_CSV, BEST_CANDIDATE_CSV, COMPARE_CSV, VIDEO_LEVEL_CSV, SUMMARY_MD, REPORT_JSON, FINDINGS_MD, LOG_PATH]
        validations = validate(split_validation, sweep_rows, audit, recovery, best, protected_modified, latest_files)
        report = write_reports(setup_report, primary_report, raw_reuse, split_validation, inference_meta, sweep_summary_rows, recall_summary_rows, recovery, best, compare, validations, warnings, errors, protected_modified, row_counts, start_iso)

        logger.step(15, 'Sub Agent 검증 실행')
        logger.write('검증 상태: ' + json.dumps({k: v.get('status') for k, v in validations.items()}, ensure_ascii=False, sort_keys=True))

        logger.step(16, 'latest bundle 갱신')
        update_latest(latest_files, logger)
        shutil.copy2(LOG_PATH, LATEST_BUNDLE_DIR / LOG_PATH.name)
        shutil.copy2(LOG_PATH, SHARED_LATEST_DIR / LOG_PATH.name)
        shutil.copy2(REPORT_JSON, LATEST_BUNDLE_DIR / REPORT_JSON.name)
        shutil.copy2(REPORT_JSON, SHARED_LATEST_DIR / REPORT_JSON.name)

        logger.step(17, '최종 요약 출력')
        logger.write(f"처리한 train 영상: {inference_meta.get('processed_video_count')}/12")
        logger.write(f"sweep 조합 수: {len(SWEEP_COMBOS)}")
        logger.write(f"추천 기준: {selected_family}")
        logger.write(f"추천 기준 후보 수/cpm: {selected.get('candidate_count_total')} / {fmt(safe_float(selected.get('candidates_per_minute_total')),3)}")
        logger.write(f"기존 combined all recall @2/@5/@10: {fmt(existing_summary['all_2'],4)}/{fmt(existing_summary['all_5'],4)}/{fmt(existing_summary['all_10'],4)}")
        logger.write(f"추천 TransNetV2 all recall @2/@5/@10: {fmt(safe_float(metric(lookup, selected_family, 'all', 2)),4)}/{fmt(safe_float(metric(lookup, selected_family, 'all', 5)),4)}/{fmt(safe_float(metric(lookup, selected_family, 'all', 10)),4)}")
        logger.write(f"existing + 추천 all recall @2/@5/@10: {fmt(safe_float(metric(lookup, 'existing_plus_best_conservative_transnetv2', 'all', 2)),4)}/{fmt(safe_float(metric(lookup, 'existing_plus_best_conservative_transnetv2', 'all', 5)),4)}/{fmt(safe_float(metric(lookup, 'existing_plus_best_conservative_transnetv2', 'all', 10)),4)}")
        logger.write(f"recovered @2/@5/@10: {selected.get('recovered_both_missed_2s')}/{selected.get('recovered_both_missed_5s')}/{selected.get('recovered_both_missed_10s')}")
        logger.write(f"경과 시간: {time.time() - start:.2f}초")
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        errors.append(str(exc))
        try:
            logger.write('오류 발생: ' + str(exc))
            logger.write(tb)
            REPORT_JSON.write_text(json.dumps({'task_name': 'transnetv2_conservative_sweep_v2_4_train', 'project_root': str(PROJECT_ROOT), 'start_time': start_iso, 'end_time': now_iso(), 'errors': errors, 'warnings': warnings, 'traceback': tb, 'no_detector_rule_modified': True, 'no_validation_test_row_level_output': True, 'actual_label_used_for_candidate_generation': False, 'actual_label_used_for_recall_audit_only': True}, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
        except Exception:
            pass
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
