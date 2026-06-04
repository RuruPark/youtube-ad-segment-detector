#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, math, shutil, statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path('.')
OLD_PROJECT_ROOT = Path('./_old_project_not_included')
TRAIN_IDS = {1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15}
VALIDATION_IDS = {3, 7, 18}
TEST_IDS = {4, 16, 17}
NON_TRAIN_IDS = VALIDATION_IDS | TEST_IDS
SPLIT_SEED = '20240524'
VERSION = 'v2.4'
TASK = 'train_actual_label_audio_ocr_cue_coverage_audit_v2_4'
FORBIDDEN_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.wav', '.mp3', '.m4a', '.jpg', '.jpeg', '.png', '.webp', '.parquet', '.pkl', '.pickle', '.pt', '.pth', '.ckpt', '.onnx'}
FORBIDDEN_PARTS = {'cache', 'frames', 'frame_images', 'raw_video', 'video_proxy', 'model_cache', 'tmp', '__pycache__'}

SPLIT_FILE = PROJECT_ROOT / 'data/splits/video_split_v2_4.csv'
ACTUAL_FILE = PROJECT_ROOT / 'data/segments/ad_interval_segments_v2_4.csv'
AUDIO_CANDIDATES = [
    ('audio_subwindow_with_split', PROJECT_ROOT / 'data/audio/audio_ad_edge_persistence_subwindow_features_v2_4_with_split.csv', 'subwindow'),
    ('audio_subwindow', PROJECT_ROOT / 'data/audio/audio_ad_edge_persistence_subwindow_features_v2_4.csv', 'subwindow'),
    ('audio_segment_with_split', PROJECT_ROOT / 'data/audio/audio_labeled_segment_features_v2_4_with_split.csv', 'segment'),
    ('audio_segment', PROJECT_ROOT / 'data/audio/audio_labeled_segment_features_v2_4.csv', 'segment'),
    ('audio_edge_with_split', PROJECT_ROOT / 'data/audio/audio_ad_edge_5s_10s_features_v2_4_with_split.csv', 'segment'),
    ('audio_edge', PROJECT_ROOT / 'data/audio/audio_ad_edge_5s_10s_features_v2_4.csv', 'segment'),
    ('audio_context_scores', PROJECT_ROOT / 'data/audio/audio_ad_edge_persistence_context_scores_v2_4.csv', 'segment'),
    ('audio_visual_anchor', PROJECT_ROOT / 'data/audio/audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv', 'anchor'),
]
OCR_CANDIDATES = [
    ('ocr_frame_recovered', PROJECT_ROOT / 'data/ocr/ocr_frame_level_results_v2_4_recovered.csv', 'frame'),
    ('ocr_frame', PROJECT_ROOT / 'data/ocr/ocr_frame_level_results_v2_4.csv', 'frame'),
    ('ocr_visual_anchor_frame', PROJECT_ROOT / 'data/ocr/ocr_visual_anchor_frame_results_v2_4.csv', 'frame'),
    ('ocr_labeled_sampling_plan', PROJECT_ROOT / 'data/ocr/ocr_labeled_segment_sampling_plan_v2_4.csv', 'sampling_plan'),
    ('ocr_edge_sampling_plan', PROJECT_ROOT / 'data/ocr/ocr_ad_edge_5s_10s_sampling_plan_v2_4.csv', 'sampling_plan'),
    ('ocr_labeled_features_recovered_train_only', PROJECT_ROOT / 'data/ocr/ocr_labeled_segment_features_v2_4_recovered_train_only.csv', 'segment'),
    ('ocr_labeled_features_recovered', PROJECT_ROOT / 'data/ocr/ocr_labeled_segment_features_v2_4_recovered.csv', 'segment'),
    ('ocr_labeled_features', PROJECT_ROOT / 'data/ocr/ocr_labeled_segment_features_v2_4.csv', 'segment'),
    ('ocr_edge_features_recovered_train_only', PROJECT_ROOT / 'data/ocr/ocr_ad_edge_5s_10s_features_v2_4_recovered_train_only.csv', 'segment'),
    ('ocr_edge_features_recovered', PROJECT_ROOT / 'data/ocr/ocr_ad_edge_5s_10s_features_v2_4_recovered.csv', 'segment'),
    ('ocr_edge_features', PROJECT_ROOT / 'data/ocr/ocr_ad_edge_5s_10s_features_v2_4.csv', 'segment'),
    ('ocr_visual_anchor_context_train_val', PROJECT_ROOT / 'data/ocr/ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv', 'anchor'),
    ('ocr_visual_anchor_context', PROJECT_ROOT / 'data/ocr/ocr_visual_anchor_context_features_v2_4.csv', 'anchor'),
]
ANCHOR_CANDIDATES = [PROJECT_ROOT / 'data/features/visual_scene_boundary_anchors_v2_4_with_split.csv', PROJECT_ROOT / 'data/features/visual_scene_boundary_anchors_v2_4.csv']
DETECTOR_FILES = {
    'high': PROJECT_ROOT / 'data/predictions/state_machine_interval_predictions_v1_4_train.csv',
    'review': PROJECT_ROOT / 'data/predictions/state_machine_review_only_interval_candidates_v1_4_train.csv',
    'open': PROJECT_ROOT / 'data/predictions/state_machine_open_interval_candidates_v1_4_train.csv',
    'unresolved': PROJECT_ROOT / 'data/predictions/state_machine_unresolved_long_open_candidates_v1_4_train.csv',
    'rejected': PROJECT_ROOT / 'data/predictions/state_machine_rejected_long_open_candidates_v1_4_train.csv',
}
OUT = {
    'script': PROJECT_ROOT / 'scripts/analysis/train_actual_label_audio_ocr_cue_coverage_audit_v2_4.py',
    'interval_summary': PROJECT_ROOT / 'data/analysis/train_actual_ad_interval_cue_coverage_summary_v2_4.csv',
    'audio': PROJECT_ROOT / 'data/analysis/train_actual_ad_window_audio_coverage_v2_4.csv',
    'ocr': PROJECT_ROOT / 'data/analysis/train_actual_ad_window_ocr_coverage_v2_4.csv',
    'anchor': PROJECT_ROOT / 'data/analysis/train_actual_ad_boundary_anchor_proximity_v2_4.csv',
    'failure': PROJECT_ROOT / 'data/analysis/train_actual_ad_cue_failure_cases_v2_4.csv',
    'detector_join': PROJECT_ROOT / 'data/analysis/train_actual_ad_cue_detector_join_diagnosis_v2_4.csv',
    'inventory': PROJECT_ROOT / 'data/analysis/train_actual_ad_cue_artifact_inventory_v2_4.csv',
    'summary': PROJECT_ROOT / 'reports/analysis/train_actual_label_audio_ocr_cue_coverage_audit_v2_4_summary.md',
    'report': PROJECT_ROOT / 'reports/analysis/train_actual_label_audio_ocr_cue_coverage_audit_v2_4_report.json',
    'findings': PROJECT_ROOT / 'reports/analysis/train_actual_label_audio_ocr_cue_coverage_audit_v2_4_findings.md',
    'log': PROJECT_ROOT / 'logs/train_actual_label_audio_ocr_cue_coverage_audit_v2_4_run_log.txt',
    'latest': PROJECT_ROOT / 'outputs/latest_for_chatgpt_train_actual_label_audio_ocr_cue_coverage_audit_v2_4',
}

def now() -> str: return datetime.now().astimezone().isoformat(timespec='seconds')
def stamp() -> str: return datetime.now().strftime('%Y%m%d_%H%M%S')
def log(msg: str) -> None:
    print(msg)
    OUT['log'].parent.mkdir(parents=True, exist_ok=True)
    with OUT['log'].open('a', encoding='utf-8') as f: f.write(msg + '\n')
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
def file_stats(path: Path) -> dict[str, Any]:
    if not path.exists(): return {'path': str(path), 'exists': False}
    st = path.stat(); return {'path': str(path), 'exists': True, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'sha256': sha256(path)}
def snapshot_dir(src: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True); rows = ['relative_path\tsize\tmtime_ns']
    if src.exists():
        for p in sorted(src.rglob('*')):
            if p.is_file():
                st = p.stat(); rows.append(f'{p.relative_to(src)}\t{st.st_size}\t{st.st_mtime_ns}')
    out.write_text('\n'.join(rows) + '\n', encoding='utf-8')
def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists(): return []
    with path.open(newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def read_header(path: Path) -> list[str]:
    if not path.exists(): return []
    with path.open(newline='', encoding='utf-8-sig') as f: return next(csv.reader(f), [])
def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore'); writer.writeheader()
        for row in rows: writer.writerow({k: to_cell(row.get(k, '')) for k in fieldnames})
def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
def to_cell(value: Any) -> str:
    if value is None: return ''
    if isinstance(value, (list, dict, tuple, set)): return json.dumps(value, ensure_ascii=False)
    return str(value)
def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == '': return default
        v = float(value); return default if math.isnan(v) else v
    except Exception: return default
def in01(v: float) -> float: return max(0.0, min(1.0, v))
def bval(value: Any) -> bool: return str(value).strip().lower() in {'true', '1', 'yes', 'y'}
def get_col(row: dict[str, Any], candidates: Iterable[str], default: Any = '') -> Any:
    for c in candidates:
        if c in row and row.get(c) not in (None, ''): return row.get(c)
    return default
def overlap_len(a_start: float, a_end: float, b_start: float, b_end: float) -> float: return max(0.0, min(a_end, b_end) - max(a_start, b_start))
def union_len(intervals: list[tuple[float, float]]) -> float:
    cleaned = sorted((s, e) for s, e in intervals if e > s)
    if not cleaned: return 0.0
    merged = [list(cleaned[0])]
    for s, e in cleaned[1:]:
        if s > merged[-1][1]: merged.append([s, e])
        else: merged[-1][1] = max(merged[-1][1], e)
    return sum(e - s for s, e in merged)
def max_gap(times: list[float]) -> float:
    if len(times) < 2: return 0.0 if times else -1.0
    times = sorted(times); return max(b - a for a, b in zip(times, times[1:]))
def level_from_ratio(ratio: float) -> str: return 'high' if ratio >= 0.8 else ('medium' if ratio >= 0.4 else 'low')
def mmss(sec: float) -> str:
    sec = max(0, int(round(sec))); return f'{sec//60:02d}:{sec%60:02d}'

def backup_outputs() -> dict[str, Any]:
    backup_dir = PROJECT_ROOT / 'backups' / f'{TASK}_{stamp()}'; backup_dir.mkdir(parents=True, exist_ok=True); copied = []
    for path in OUT.values():
        if path.exists():
            dst = backup_dir / path.relative_to(PROJECT_ROOT); dst.parent.mkdir(parents=True, exist_ok=True)
            if path.is_dir(): shutil.copytree(path, dst)
            else: shutil.copy2(path, dst)
            copied.append({'source': str(path), 'backup': str(dst)})
    write_json(backup_dir / 'backup_manifest.json', {'created_at': now(), 'copied': copied})
    return {'backup_dir': str(backup_dir), 'copied': copied}

def validate_split(split_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    by_video = {int(fnum(r.get('video_id'))): r for r in split_rows}
    train = {vid for vid, r in by_video.items() if r.get('split') == 'train'}; val = {vid for vid, r in by_video.items() if r.get('split') == 'validation'}; test = {vid for vid, r in by_video.items() if r.get('split') == 'test'}; seeds = {str(r.get('split_seed')) for r in split_rows}
    if train != TRAIN_IDS or val != VALIDATION_IDS or test != TEST_IDS or seeds != {SPLIT_SEED}: raise RuntimeError(f'fixed split mismatch train={sorted(train)} val={sorted(val)} test={sorted(test)} seeds={seeds}')
    return by_video

def extract_actual_intervals(actual_rows: list[dict[str, str]], split_meta: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    intervals = []; exclude_types = {'non_ad', 'random_non_ad', 'post_ad', 'pre_ad'}
    for r in actual_rows:
        vid = int(fnum(get_col(r, ['video_id']))); typ = str(get_col(r, ['segment_type', 'interval_type', 'label', 'is_ad'], '')).strip().lower()
        if vid not in TRAIN_IDS or typ in exclude_types: continue
        if typ and ('ad' not in typ) and typ not in {'true', '1', 'yes'}: continue
        start = fnum(get_col(r, ['ad_start_sec', 'start_sec', 'segment_start_sec', 'interval_start_sec'])); end = fnum(get_col(r, ['ad_end_sec', 'end_sec', 'segment_end_sec', 'interval_end_sec']))
        if end <= start: continue
        meta = split_meta[vid]; dur = fnum(meta.get('video_duration_sec')); start=max(0.0,start); end=min(end,dur) if dur else end
        intervals.append({'actual_interval_id': r.get('ad_interval_id') or r.get('\ufeffsegment_id') or f'actual_{vid}_{len(intervals)+1}', 'video_id': vid, 'split': 'train', 'ad_start_sec': start, 'ad_end_sec': end, 'ad_duration_sec': max(0.0,end-start), 'video_duration_sec': dur, 'source_segment_type': typ, 'video_name': meta.get('video_name') or r.get('video_title') or ''})
    if not intervals: raise RuntimeError('could not extract train actual ad intervals')
    return intervals

def make_windows(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]
    for it in intervals:
        dur=it['video_duration_sec']; s=it['ad_start_sec']; e=it['ad_end_sec']
        for name, ws, we in [('pre_10s',s-10,s),('start_edge_10s',s,s+10),('ad_body',s,e),('end_edge_10s',e-10,e),('post_10s',e,e+10)]:
            cs=max(0.0,ws); ce=min(dur,we) if dur else we
            out.append({**it,'window_type':name,'window_start_sec':cs,'window_end_sec':ce,'window_duration_sec':max(0.0,ce-cs),'window_mmss':f'{mmss(cs)}-{mmss(ce)}','window_warning':'short_or_empty_window' if ce-cs<1.0 else ''})
    return out

def artifact_inventory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows=[]; chosen={'audio':None,'audio_mode':None,'ocr':None,'ocr_mode':None,'anchor':None}
    for artifact_type, candidates in [('audio', AUDIO_CANDIDATES), ('ocr', OCR_CANDIDATES)]:
        for role, path, mode in candidates:
            exists=path.exists(); header=read_header(path) if exists else []; row_count=max(0,len(read_csv(path))) if exists else 0; has_split='split' in header; train_rows=0
            if exists:
                for r in read_csv(path):
                    vid=int(fnum(r.get('video_id')))
                    if vid in TRAIN_IDS and (not has_split or r.get('split') in ('','train')): train_rows+=1
            used='not_used'
            if artifact_type=='audio' and chosen['audio'] is None and exists:
                chosen['audio']=path; chosen['audio_mode']=mode; used='primary_audio_coverage_source'
            if artifact_type=='ocr' and chosen['ocr'] is None and exists:
                chosen['ocr']=path; chosen['ocr_mode']=mode; used='primary_ocr_coverage_source'
            rows.append({'artifact_type':artifact_type,'candidate_role':role,'path':str(path),'exists':exists,'row_count':row_count,'has_split':has_split,'train_row_count':train_rows,'mode':mode,'used_as':used,'column_count':len(header),'columns_json':header[:80],'note':'' if exists else 'missing_artifact'})
    for path in ANCHOR_CANDIDATES:
        exists=path.exists(); header=read_header(path) if exists else []; row_count=max(0,len(read_csv(path))) if exists else 0; has_split='split' in header; train_rows=0
        if exists:
            for r in read_csv(path):
                vid=int(fnum(r.get('video_id')))
                if vid in TRAIN_IDS and (not has_split or r.get('split') in ('','train')): train_rows+=1
        used='not_used'
        if chosen['anchor'] is None and exists: chosen['anchor']=path; used='primary_boundary_anchor_source'
        rows.append({'artifact_type':'visual_anchor','candidate_role':'boundary_anchor','path':str(path),'exists':exists,'row_count':row_count,'has_split':has_split,'train_row_count':train_rows,'mode':'anchor','used_as':used,'column_count':len(header),'columns_json':header[:80],'note':'' if exists else 'missing_artifact'})
    return rows, chosen

def compute_anchor_proximity(intervals, anchor_path):
    anchors_by_vid=defaultdict(list); source=''
    if anchor_path and Path(anchor_path).exists():
        source=str(anchor_path)
        for r in read_csv(Path(anchor_path)):
            vid=int(fnum(r.get('video_id')))
            if vid in TRAIN_IDS and (r.get('split') in ('','train') or 'split' not in r): anchors_by_vid[vid].append(fnum(get_col(r,['canonical_boundary_time_sec','candidate_time_sec','anchor_time_mean_sec'])))
    rows=[]
    for it in intervals:
        vid=it['video_id']; anchors=sorted(anchors_by_vid.get(vid,[])); s=it['ad_start_sec']; e=it['ad_end_sec']
        def nearest(x):
            if not anchors: return None, None
            t=min(anchors,key=lambda a:abs(a-x)); return t, abs(t-x)
        st,sd=nearest(s); et,ed=nearest(e)
        if sd is None or ed is None: status='missing_anchor_artifact'
        elif sd<=5 and ed<=5: status='good'
        elif (sd<=10 and ed<=10) or sd<=5 or ed<=5: status='partial'
        else: status='poor'
        rows.append({'actual_interval_id':it['actual_interval_id'],'video_id':vid,'split':'train','ad_start_sec':s,'ad_end_sec':e,'nearest_anchor_to_ad_start_sec':st,'distance_to_nearest_start_anchor_sec':sd,'nearest_anchor_to_ad_end_sec':et,'distance_to_nearest_end_anchor_sec':ed,'start_anchor_within_2s':sd is not None and sd<=2,'start_anchor_within_5s':sd is not None and sd<=5,'start_anchor_within_10s':sd is not None and sd<=10,'end_anchor_within_2s':ed is not None and ed<=2,'end_anchor_within_5s':ed is not None and ed<=5,'end_anchor_within_10s':ed is not None and ed<=10,'boundary_anchor_coverage_status':status,'anchor_source_file':source,'anchor_count_in_video':len(anchors)})
    return rows

def compute_audio(windows, chosen):
    path=chosen.get('audio'); mode=chosen.get('audio_mode'); rows_by_vid=defaultdict(list)
    if path and Path(path).exists():
        for r in read_csv(Path(path)):
            vid=int(fnum(r.get('video_id')))
            if vid in TRAIN_IDS and (r.get('split') in ('','train') or 'split' not in r): rows_by_vid[vid].append(r)
    out=[]
    for w in windows:
        vid=w['video_id']; ws=w['window_start_sec']; we=w['window_end_sec']; wd=w['window_duration_sec']; expected=max(1,math.ceil(wd/2.0)) if wd>0 else 0
        overlaps=[]; times=[]; valid=0; scores=[]; low_count=0; levels=Counter(); covered=[]
        for r in rows_by_vid.get(vid,[]):
            if mode=='subwindow': rs=fnum(get_col(r,['subwindow_start_sec','segment_start_sec','start_sec'])); re=fnum(get_col(r,['subwindow_end_sec','segment_end_sec','end_sec']))
            elif mode=='segment': rs=fnum(get_col(r,['segment_start_sec','start_sec','window_start_sec'])); re=fnum(get_col(r,['segment_end_sec','end_sec','window_end_sec']))
            else: t=fnum(get_col(r,['candidate_time_sec','transition_time_anchor','frame_time_sec'])); rs,re=t-0.5,t+0.5
            if overlap_len(ws,we,rs,re)<=0: continue
            overlaps.append(r); covered.append((max(ws,rs),min(we,re))); times.append((rs+re)/2)
            is_valid=str(r.get('feature_status','')).lower() in {'success',''} and str(r.get('audio_available','')).lower() in {'true','1','yes',''}
            if is_valid: valid+=1
            silence = fnum(r.get('silence_ratio'), default=None) if r.get('silence_ratio') not in (None,'') else None
            low_energy = fnum(r.get('low_energy_ratio'), default=None) if r.get('low_energy_ratio') not in (None,'') else None
            score = 1-in01(silence) if silence is not None else (1-in01(low_energy) if low_energy is not None else fnum(get_col(r,['audio_context_score','audio_start_signal_score','audio_pre_10s_score_mean','audio_post_10s_score_mean']),0))
            scores.append(score)
            if score>=0.6: levels['high']+=1
            elif score>=0.3: levels['medium']+=1
            else: levels['low']+=1; low_count+=1
        observed=len(overlaps); count_cov=in01(observed/expected) if expected else 0; dur_cov=in01(union_len(covered)/wd) if wd>0 else 0; cov=max(count_cov, dur_cov if mode=='segment' else count_cov)
        mean=statistics.mean(scores) if scores else None; med=statistics.median(scores) if scores else None; mh=(levels['medium']+levels['high'])/observed if observed else 0; hi=levels['high']/observed if observed else 0
        out.append({**{k:w[k] for k in ['actual_interval_id','video_id','split','window_type','window_start_sec','window_end_sec','window_duration_sec','window_mmss']},'audio_source_file':str(path) if path else '', 'audio_coverage_mode':mode or 'missing','fallback_used':mode!='subwindow','expected_subwindow_count':expected,'observed_subwindow_count':observed,'valid_subwindow_count':valid,'missing_subwindow_count':max(0,expected-observed),'subwindow_coverage_ratio':round(cov,6),'duration_overlap_coverage_ratio':round(dur_cov,6),'max_time_gap_between_audio_subwindows_sec':max_gap(times),'mean_audio_ad_like_score':round(mean,6) if mean is not None else '', 'median_audio_ad_like_score':round(med,6) if med is not None else '', 'high_score_subwindow_ratio':round(hi,6),'medium_or_high_score_subwindow_ratio':round(mh,6),'audio_low_ratio':round(low_count/observed,6) if observed else 1.0,'audio_evidence_density_level':level_from_ratio(cov),'audio_signal_level_distribution':dict(levels),'audio_note':'activity proxy uses inverse silence/low_energy when ad-like score is unavailable' if observed else 'no overlapping existing audio rows'})
    return out

DISCLOSURE_KW=['유료광고','광고','협찬','sponsor','sponsored','ad ']; CTA_KW=['구매','링크','쿠폰','할인','더보기','프로모션','이벤트','purchase','buy','link','coupon','discount']; PRODUCT_KW=['제품','브랜드','화장품','앰플','크림','세럼','brand','product']; DISCOUNT_KW=['할인','쿠폰','%','세일','특가','discount','sale']
def count_keywords(text, words):
    t=(text or '').lower(); return sum(t.count(w.lower()) for w in words)
def compute_ocr(windows, chosen):
    path=chosen.get('ocr'); mode=chosen.get('ocr_mode'); rows_by_vid=defaultdict(list)
    if path and Path(path).exists():
        for r in read_csv(Path(path)):
            vid=int(fnum(r.get('video_id')))
            if vid in TRAIN_IDS and (r.get('split') in ('','train') or 'split' not in r): rows_by_vid[vid].append(r)
    out=[]
    for w in windows:
        vid=w['video_id']; ws=w['window_start_sec']; we=w['window_end_sec']; wd=w['window_duration_sec']; expected=max(1,math.ceil(wd/2.0)) if wd>0 else 0
        overlaps=[]; times=[]; covered=[]; attempted=success=empty=failed=recovered=0; text_box=token=char=0; confs=[]; disclosure=cta=product=discount=keyword_score=0; ad_scores=[]
        for r in rows_by_vid.get(vid,[]):
            if mode in {'frame','sampling_plan'}: t=fnum(get_col(r,['frame_time_sec','candidate_time_sec'])); rs,re=t-0.25,t+0.25
            elif mode=='segment': rs=fnum(get_col(r,['segment_start_sec','start_sec'])); re=fnum(get_col(r,['segment_end_sec','end_sec']))
            else: t=fnum(get_col(r,['candidate_time_sec','frame_time_sec'])); rs,re=t-0.5,t+0.5
            if overlap_len(ws,we,rs,re)<=0: continue
            overlaps.append(r); times.append((rs+re)/2); covered.append((max(ws,rs),min(we,re)))
            if mode=='sampling_plan': continue
            attempted+=1; status=str(get_col(r,['ocr_status','extraction_status','plan_status'],'')).lower(); txt_count=int(fnum(get_col(r,['ocr_text_count','ocr_text_box_count','frame_count'],0))); tok=int(fnum(get_col(r,['ocr_token_count'],0))); ch=int(fnum(get_col(r,['ocr_char_count'],0)))
            text_box+=int(fnum(get_col(r,['ocr_box_count','ocr_text_box_count'],txt_count))); token+=tok; char+=ch
            if 'success' in status or status in {'','ok'}:
                success+=1
                if 'empty' in status or ch==0: empty+=1
            else: failed+=1
            if bval(r.get('retry_success')): recovered+=1
            if get_col(r,['ocr_mean_confidence'],'')!='': confs.append(fnum(get_col(r,['ocr_mean_confidence'])))
            text=' '.join(str(r.get(c,'')) for c in ['ocr_text_raw','ocr_text_normalized','ocr_text_joined','detected_text_raw','detected_text_normalized','detected_text_joined','representative_ocr_text'])
            disclosure+=count_keywords(text,DISCLOSURE_KW); cta+=count_keywords(text,CTA_KW); product+=count_keywords(text,PRODUCT_KW); discount+=count_keywords(text,DISCOUNT_KW); keyword_score+=fnum(get_col(r,['frame_keyword_score','ocr_keyword_score'],0)); ad_scores.append(fnum(get_col(r,['frame_ad_text_score','ocr_ad_text_score'],0)))
        planned=len(overlaps) if mode=='sampling_plan' else 0; valid=success if mode!='sampling_plan' else planned; valid_ratio=in01(valid/expected) if expected else 0; dur_cov=in01(union_len(covered)/wd) if wd>0 else 0
        if mode=='segment': valid_ratio=max(valid_ratio,dur_cov)
        text_ratio=in01((success-empty)/attempted) if attempted else 0; failed_ratio=in01(failed/attempted) if attempted else 0; kw=int(disclosure+cta+product+discount+keyword_score)
        text_level='high' if valid_ratio>=0.8 and (text_ratio>=0.5 or kw>0 or sum(ad_scores)>0) else ('medium' if valid_ratio>=0.4 and (text_ratio>0 or kw>0 or sum(ad_scores)>0) else 'low')
        out.append({**{k:w[k] for k in ['actual_interval_id','video_id','split','window_type','window_start_sec','window_end_sec','window_duration_sec','window_mmss']},'ocr_source_file':str(path) if path else '', 'ocr_coverage_mode':mode or 'missing','fallback_used':mode not in {'frame','sampling_plan'},'expected_frame_count':expected,'planned_frame_count':planned,'attempted_frame_count':attempted,'success_frame_count':success,'empty_frame_count':empty,'failed_frame_count':failed,'recovered_success_count':recovered,'ocr_attempt_coverage_ratio':round(in01(attempted/expected) if expected else 0,6),'ocr_success_or_empty_valid_ratio':round(valid_ratio,6),'ocr_success_text_ratio':round(text_ratio,6),'ocr_failed_ratio':round(failed_ratio,6),'max_time_gap_between_ocr_frames_sec':max_gap(times),'text_box_count_sum':text_box,'token_count_sum':token,'char_count_sum':char,'mean_confidence':round(statistics.mean(confs),6) if confs else '', 'ad_disclosure_count':disclosure,'purchase_cta_count':cta,'product_brand_count':product,'discount_promo_count':discount,'total_ad_keyword_count':kw,'ocr_evidence_density_level':level_from_ratio(valid_ratio),'ocr_text_signal_level':text_level,'ocr_note':'OCR empty/failed rows are coverage/quality signals only; not non-ad evidence' if overlaps else 'no overlapping existing OCR rows'})
    return out

def detector_intervals():
    out={k:defaultdict(list) for k in DETECTOR_FILES}
    for bucket,path in DETECTOR_FILES.items():
        if not path.exists(): continue
        for r in read_csv(path):
            vid=int(fnum(r.get('video_id')))
            if vid not in TRAIN_IDS or r.get('split') not in ('','train'): continue
            if bucket=='high': s=fnum(get_col(r,['ad_start_sec','start_sec'])); e=fnum(get_col(r,['ad_end_sec','end_sec']))
            else: s=fnum(get_col(r,['start_sec','ad_start_sec','pending_start_time'])); e=fnum(get_col(r,['end_sec','ad_end_sec','last_anchor_sec','end_proxy_sec','transition_time_anchor']))
            if e>s: out[bucket][vid].append((s,e))
    return out
def build_detector_join(intervals, detector):
    rows=[]
    for it in intervals:
        vid=it['video_id']; s=it['ad_start_sec']; e=it['ad_end_sec']; dur=max(0.001,e-s); ovs={}
        for bucket,by_vid in detector.items(): ovs[bucket]=in01(sum(overlap_len(s,e,ps,pe) for ps,pe in by_vid.get(vid,[]))/dur)
        if ovs.get('high',0)>=0.2: outcome='hit_by_high_confidence'
        elif ovs.get('review',0)>=0.2: outcome='covered_by_review_only'
        elif ovs.get('open',0)>=0.2 or ovs.get('unresolved',0)>=0.2 or ovs.get('rejected',0)>=0.2: outcome='covered_by_open_or_unresolved'
        else: outcome='no_detector_candidate'
        rows.append({'actual_interval_id':it['actual_interval_id'],'video_id':vid,'split':'train','ad_start_sec':s,'ad_end_sec':e,'ad_duration_sec':it['ad_duration_sec'],'high_confidence_overlap_ratio_of_actual':round(ovs.get('high',0),6),'review_only_overlap_ratio_of_actual':round(ovs.get('review',0),6),'open_overlap_ratio_of_actual':round(ovs.get('open',0),6),'unresolved_overlap_ratio_of_actual':round(ovs.get('unresolved',0),6),'rejected_overlap_ratio_of_actual':round(ovs.get('rejected',0),6),'detector_outcome_bucket':outcome,'audit_only_note':'detector v1.4 outputs joined post-hoc for diagnosis only'})
    return rows

def priority_for(cat): return 'high' if cat in {'both_sparse','cue_collected_and_strong_but_detector_missed','anchor_missing_near_boundary'} else ('medium' if cat in {'audio_good_ocr_sparse','audio_sparse_ocr_good','cue_collected_but_weak_signal'} else 'low')
def diagnosis_note(cats, outcome):
    if 'cue_collected_and_strong_but_detector_missed' in cats: return 'cue coverage/signal exists but v1.4 did not high-confidence cover the actual interval; rule/state-machine issue is plausible.'
    if 'both_sparse' in cats: return 'audio and OCR are sparse in existing artifacts; cue coverage or artifact availability is a likely blocker.'
    if 'anchor_missing_near_boundary' in cats: return 'nearest scene anchor is not close to at least one actual boundary; visual anchor recall may limit detector boundary localization.'
    if 'cue_collected_but_weak_signal' in cats: return 'coverage exists but ad-like cue signal is weak; feature score definition or rule thresholds need review.'
    if 'audio_good_ocr_sparse' in cats: return 'audio coverage is present but OCR coverage is sparse; OCR sampling/quality may be limiting.'
    if 'audio_sparse_ocr_good' in cats: return 'OCR coverage is present but audio coverage is sparse; audio subwindow feature coverage may be limiting.'
    return f'coverage diagnosis for detector outcome {outcome}'
def summarize_intervals(intervals, audio_rows, ocr_rows, anchor_rows, detector_rows):
    audio_by=defaultdict(list); ocr_by=defaultdict(list)
    for r in audio_rows: audio_by[r['actual_interval_id']].append(r)
    for r in ocr_rows: ocr_by[r['actual_interval_id']].append(r)
    anchor_by={r['actual_interval_id']:r for r in anchor_rows}; det_by={r['actual_interval_id']:r for r in detector_rows}; rows=[]; failures=[]
    for it in intervals:
        aid=it['actual_interval_id']; ar=audio_by.get(aid,[]); orows=ocr_by.get(aid,[]); body_audio=next((r for r in ar if r['window_type']=='ad_body'), ar[0] if ar else {}); body_ocr=next((r for r in orows if r['window_type']=='ad_body'), orows[0] if orows else {})
        audio_level=body_audio.get('audio_evidence_density_level','low'); ocr_level=body_ocr.get('ocr_evidence_density_level','low'); audio_signal=fnum(body_audio.get('medium_or_high_score_subwindow_ratio'),0); ocr_text_level=body_ocr.get('ocr_text_signal_level','low'); total_kw=int(fnum(body_ocr.get('total_ad_keyword_count'),0)); anchor_status=anchor_by.get(aid,{}).get('boundary_anchor_coverage_status','missing_anchor_artifact'); outcome=det_by.get(aid,{}).get('detector_outcome_bucket','no_detector_candidate')
        cats=[]
        if audio_level in {'high','medium'} and ocr_level in {'high','medium'}: cats.append('audio_coverage_good_ocr_coverage_good')
        if audio_level in {'high','medium'} and ocr_level=='low': cats.append('audio_good_ocr_sparse')
        if audio_level=='low' and ocr_level in {'high','medium'}: cats.append('audio_sparse_ocr_good')
        if audio_level=='low' and ocr_level=='low': cats.append('both_sparse')
        if anchor_status in {'poor','missing_anchor_artifact'}: cats.append('anchor_missing_near_boundary')
        coverage_good=audio_level in {'high','medium'} and ocr_level in {'high','medium'}; weak_signal=coverage_good and audio_signal<0.3 and ocr_text_level=='low' and total_kw==0; strong_signal=coverage_good and (audio_signal>=0.5 or ocr_text_level in {'medium','high'} or total_kw>0)
        if weak_signal: cats.append('cue_collected_but_weak_signal')
        if strong_signal and outcome in {'no_detector_candidate','covered_by_review_only','covered_by_open_or_unresolved'}: cats.append('cue_collected_and_strong_but_detector_missed')
        if body_audio.get('audio_coverage_mode')=='missing' or body_ocr.get('ocr_coverage_mode')=='missing': cats.append('insufficient_existing_artifact')
        if not cats: cats.append('insufficient_existing_artifact')
        row={'actual_interval_id':aid,'video_id':it['video_id'],'split':'train','ad_start_sec':it['ad_start_sec'],'ad_end_sec':it['ad_end_sec'],'ad_duration_sec':it['ad_duration_sec'],'audio_body_density_level':audio_level,'audio_body_coverage_ratio':body_audio.get('subwindow_coverage_ratio',''),'audio_body_medium_or_high_signal_ratio':body_audio.get('medium_or_high_score_subwindow_ratio',''),'ocr_body_density_level':ocr_level,'ocr_body_valid_ratio':body_ocr.get('ocr_success_or_empty_valid_ratio',''),'ocr_text_signal_level':ocr_text_level,'ocr_total_ad_keyword_count':total_kw,'boundary_anchor_coverage_status':anchor_status,'detector_outcome_bucket':outcome,'diagnosis_categories':cats,'primary_diagnosis':cats[0],'diagnosis_note':diagnosis_note(cats,outcome)}
        rows.append(row)
        for cat in cats:
            if cat!='audio_coverage_good_ocr_coverage_good': failures.append({**row,'failure_type':cat,'review_priority':priority_for(cat),'review_note':diagnosis_note([cat],outcome)})
    return rows, failures

def counts_by(rows, key):
    c=Counter()
    for r in rows: c[str(r.get(key,''))]+=1
    return dict(c)
def scan_forbidden(path: Path):
    bad=[]
    if not path.exists(): return bad
    for p in path.rglob('*'):
        rel=p.relative_to(path)
        if p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES: bad.append(str(rel))
        if any(part in FORBIDDEN_PARTS for part in rel.parts): bad.append(str(rel))
    return sorted(set(bad))
def update_latest():
    latest=OUT['latest']
    if latest.exists():
        backup=PROJECT_ROOT/'backups'/f'{TASK}_latest_{stamp()}'; backup.parent.mkdir(parents=True,exist_ok=True); shutil.copytree(latest,backup); shutil.rmtree(latest)
    latest.mkdir(parents=True,exist_ok=True)
    files=[OUT['interval_summary'],OUT['audio'],OUT['ocr'],OUT['anchor'],OUT['failure'],OUT['detector_join'],OUT['inventory'],OUT['summary'],OUT['report'],OUT['findings'],OUT['log'],OUT['script']]
    for src in files:
        if src.exists(): shutil.copy2(src, latest/src.name)
    readme=['# Latest files: train actual label audio/OCR cue coverage audit v2.4','','- train actual label audio/OCR cue coverage audit','- not detector implementation','- not rule tuning','- train-only','- validation/test excluded','- no media copied','','Files:']
    for p in sorted(latest.iterdir()):
        if p.is_file(): readme.append(f'- {p.name}')
    readme.append('- README_latest_files.md'); (latest/'README_latest_files.md').write_text('\n'.join(readme)+'\n',encoding='utf-8')
    return scan_forbidden(latest)

def write_reports(intervals,audio_rows,ocr_rows,anchor_rows,summary_rows,failure_rows,detector_rows,inventory,warnings,errors,backup,input_stats_before,input_stats_after,old_before,old_after,latest_bad):
    actual_total=sum(fnum(r['ad_duration_sec']) for r in intervals); audio_counts=counts_by([r for r in audio_rows if r['window_type']=='ad_body'],'audio_evidence_density_level'); ocr_counts=counts_by([r for r in ocr_rows if r['window_type']=='ad_body'],'ocr_evidence_density_level'); diag=Counter()
    for r in summary_rows:
        for c in r.get('diagnosis_categories',[]): diag[c]+=1
    video_summary=defaultdict(lambda:Counter())
    for r in summary_rows: video_summary[r['video_id']][r['primary_diagnosis']]+=1
    top_problem=sorted(failure_rows,key=lambda r:(0 if r['review_priority']=='high' else 1,-fnum(r.get('ad_duration_sec'))))[:10]
    report={'task_name':TASK,'version':VERSION,'project_root':str(PROJECT_ROOT),'generated_at':now(),'input_files':{'split':str(SPLIT_FILE),'actual':str(ACTUAL_FILE)},'output_files':{k:str(v) for k,v in OUT.items()},'train_video_ids':sorted(TRAIN_IDS),'excluded_validation_video_ids':sorted(VALIDATION_IDS),'excluded_test_video_ids':sorted(TEST_IDS),'row_counts':{'actual_intervals':len(intervals),'audio_window_rows':len(audio_rows),'ocr_window_rows':len(ocr_rows),'anchor_rows':len(anchor_rows),'summary_rows':len(summary_rows),'failure_rows':len(failure_rows),'detector_join_rows':len(detector_rows),'inventory_rows':len(inventory)},'metric_summary':{'train_actual_ad_interval_count':len(intervals),'train_actual_ad_total_duration_sec':round(actual_total,3),'audio_body_density_counts':audio_counts,'ocr_body_density_counts':ocr_counts,'diagnosis_counts':dict(diag),'both_sparse_interval_count':diag.get('both_sparse',0),'audio_good_ocr_sparse_count':diag.get('audio_good_ocr_sparse',0),'ocr_good_audio_sparse_count':diag.get('audio_sparse_ocr_good',0),'anchor_missing_near_boundary_count':diag.get('anchor_missing_near_boundary',0),'cue_collected_but_weak_signal_count':diag.get('cue_collected_but_weak_signal',0),'cue_collected_and_strong_but_detector_missed_count':diag.get('cue_collected_and_strong_but_detector_missed',0),'insufficient_existing_artifact_count':diag.get('insufficient_existing_artifact',0)},'coverage_thresholds':{'high':'ratio >= 0.8','medium':'ratio >= 0.4','low':'ratio < 0.4'},'artifact_inventory_summary':counts_by(inventory,'used_as'),'fallback_usage':{'audio_modes':counts_by(audio_rows,'audio_coverage_mode'),'ocr_modes':counts_by(ocr_rows,'ocr_coverage_mode')},'video_diagnosis_summary':{str(k):dict(v) for k,v in video_summary.items()},'worst_cue_coverage_intervals':top_problem,'sub_agent_results':[],'warnings':warnings,'errors':errors,'backup':backup,'old_project_modified':old_before.read_text()!=old_after.read_text() if old_before.exists() and old_after.exists() else None,'input_files_modified':[p for p,b in input_stats_before.items() if b!=input_stats_after.get(p)],'latest_for_chatgpt_forbidden_files_found':latest_bad,'no_detector_run':True,'no_rule_change':True,'no_threshold_tuning':True,'no_feature_extraction':True,'train_only_actual_label_cue_coverage_audit':True,'validation_test_output_count':0}
    write_json(OUT['report'],report)
    md=['# Train Actual Label Audio/OCR Cue Coverage Audit v2.4 Summary','','## 작업 개요','- train 실제 광고 라벨 구간 기준으로 audio/OCR cue coverage와 visual anchor proximity를 점검했습니다.','- detector 재실행, rule 수정, threshold tuning, feature extraction은 수행하지 않았습니다.','- validation/test는 제외했습니다.','','## 핵심 수치',f'- train actual ad interval count: {len(intervals)}',f'- train actual ad total duration: {actual_total:.1f} sec',f'- audio body coverage high/medium/low: {audio_counts.get("high",0)} / {audio_counts.get("medium",0)} / {audio_counts.get("low",0)}',f'- OCR body coverage high/medium/low: {ocr_counts.get("high",0)} / {ocr_counts.get("medium",0)} / {ocr_counts.get("low",0)}',f'- both sparse interval count: {diag.get("both_sparse",0)}',f'- audio good but OCR sparse count: {diag.get("audio_good_ocr_sparse",0)}',f'- OCR good but audio sparse count: {diag.get("audio_sparse_ocr_good",0)}',f'- anchor missing near boundary count: {diag.get("anchor_missing_near_boundary",0)}',f'- cue collected but weak signal count: {diag.get("cue_collected_but_weak_signal",0)}',f'- cue collected and strong but detector missed count: {diag.get("cue_collected_and_strong_but_detector_missed",0)}',f'- artifact missing/insufficient count: {diag.get("insufficient_existing_artifact",0)}','','## Video별 cue coverage summary']
    for vid in sorted(video_summary): md.append(f'- video {vid}: {dict(video_summary[vid])}')
    md+=['','## Worst cue coverage intervals']
    for r in top_problem[:10]: md.append(f'- video {r["video_id"]} {mmss(fnum(r["ad_start_sec"]))}-{mmss(fnum(r["ad_end_sec"]))}: {r["failure_type"]} / {r["review_note"]}')
    md+=['','## 다음 단계 추천','- cue coverage가 충분한데 놓친 구간은 rule/state-machine 설계 문제로 되돌아가서 확인합니다.','- cue coverage가 부족한 구간은 OCR/audio sampling 또는 기존 feature artifact 보강 필요성을 먼저 검토합니다.','- boundary anchor가 부족한 구간은 visual anchor candidate recall 보강을 검토합니다.','- validation/test는 아직 실행하지 않습니다.']
    OUT['summary'].parent.mkdir(parents=True,exist_ok=True); OUT['summary'].write_text('\n'.join(md)+'\n',encoding='utf-8')
    findings=f'''# Train Actual Label Audio/OCR Cue Coverage Audit v2.4 Findings

## 결론
- train 실제 광고 구간 {len(intervals)}개, 총 {actual_total:.1f}초를 기준으로 기존 audio/OCR cue coverage를 점검했습니다.
- audio body density high/medium/low는 {audio_counts.get('high',0)} / {audio_counts.get('medium',0)} / {audio_counts.get('low',0)}입니다.
- OCR body density high/medium/low는 {ocr_counts.get('high',0)} / {ocr_counts.get('medium',0)} / {ocr_counts.get('low',0)}입니다.
- cue가 수집되었는데 detector가 놓친 후보는 {diag.get('cue_collected_and_strong_but_detector_missed',0)}개로 집계되었습니다.
- 이 결과는 final performance claim이 아니라 train-only 원인 진단입니다.

## 1. Audio coverage
- audio subwindow artifact를 우선 사용했습니다.
- audio가 부족한 구간은 `train_actual_ad_window_audio_coverage_v2_4.csv`의 `audio_evidence_density_level=low` 행에서 확인할 수 있습니다.
- audio score가 명시적으로 없는 subwindow에서는 inverse silence/low-energy를 activity proxy로 사용했습니다.

## 2. OCR coverage
- OCR recovered frame-level artifact를 우선 사용했습니다.
- OCR success/empty/failed는 coverage/quality 신호로만 해석했으며 non-ad evidence로 사용하지 않았습니다.
- OCR text/keyword가 희박한 구간은 `ocr_text_signal_level=low`와 keyword count를 함께 확인해야 합니다.

## 3. Boundary anchor proximity
- actual ad start/end와 nearest visual scene anchor 거리를 계산했습니다.
- `boundary_anchor_coverage_status=poor` 또는 `missing_anchor_artifact`인 구간은 scene anchor recall 문제가 detector boundary localization을 어렵게 만들 수 있습니다.

## 4. Detector miss와 cue coverage 연결
- detector v1.4 high/review/open/unresolved/rejected output은 post-hoc audit join에만 사용했습니다.
- `cue_collected_and_strong_but_detector_missed`는 cue coverage가 있는데 high-confidence prediction으로 잡히지 않은 구간입니다.
- `both_sparse`는 rule보다 기존 cue artifact coverage 부족 가능성을 먼저 확인해야 하는 구간입니다.

## 5. 다음 조치 제안
- cue coverage가 충분하면 rule/state-machine 설계 문제로 돌아갑니다.
- cue coverage가 부족하면 OCR/audio sampling 보강 또는 feature artifact 재생성을 검토합니다.
- boundary anchor가 부족하면 visual anchor candidate 보강을 검토합니다.
- train에서 원인 유형을 확정하기 전 validation/test는 실행하지 않습니다.
'''
    OUT['findings'].write_text(findings,encoding='utf-8')

def final_print(report):
    ms=report['metric_summary']; lines=['[STEP 13] Final human-readable summary','작업 상태: SUCCESS' if not report['errors'] else '작업 상태: CONDITIONAL_SUCCESS',f'train actual ad interval count: {ms["train_actual_ad_interval_count"]}',f'audio coverage high/medium/low count: {ms["audio_body_density_counts"].get("high",0)} / {ms["audio_body_density_counts"].get("medium",0)} / {ms["audio_body_density_counts"].get("low",0)}',f'OCR coverage high/medium/low count: {ms["ocr_body_density_counts"].get("high",0)} / {ms["ocr_body_density_counts"].get("medium",0)} / {ms["ocr_body_density_counts"].get("low",0)}',f'both sparse count: {ms["both_sparse_interval_count"]}',f'anchor missing near boundary count: {ms["anchor_missing_near_boundary_count"]}',f'cue collected but weak signal count: {ms["cue_collected_but_weak_signal_count"]}',f'cue collected and strong but detector missed count: {ms["cue_collected_and_strong_but_detector_missed_count"]}',f'insufficient artifact count: {ms["insufficient_existing_artifact_count"]}',f'판단: rule 문제 가능성={ms["cue_collected_and_strong_but_detector_missed_count"]>0}, audio/OCR coverage 문제 가능성={ms["both_sparse_interval_count"]>0 or ms["audio_good_ocr_sparse_count"]>0 or ms["ocr_good_audio_sparse_count"]>0}, visual anchor recall 문제 가능성={ms["anchor_missing_near_boundary_count"]>0}',f'output summary: {OUT["summary"]}',f'output report: {OUT["report"]}',f'latest bundle: {OUT["latest"]}',f'old_project_modified: {report["old_project_modified"]}',f'input_files_modified: {report["input_files_modified"]}',f'validation/test output count: {report["validation_test_output_count"]}',f'warnings/errors: {report["warnings"]} / {report["errors"]}','다음 단계: cue coverage가 충분하면 rule 설계 문제로 돌아가고, coverage가 부족하면 OCR/audio sampling 보강을 먼저 검토하며, boundary anchor가 부족하면 visual anchor candidate 보강을 검토합니다. validation/test는 아직 실행하지 마십시오.']
    for line in lines: log(line)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--project-root',default=str(PROJECT_ROOT)); args=parser.parse_args(); root=Path(args.project_root).resolve()
    if root!=PROJECT_ROOT: raise SystemExit(f'project root mismatch: {root}')
    OUT['log'].parent.mkdir(parents=True,exist_ok=True); OUT['log'].write_text('',encoding='utf-8'); warnings=[]; errors=[]
    log('[STEP 01] Safety snapshot and backup'); backup=backup_outputs(); old_before=PROJECT_ROOT/'reports/analysis/train_actual_label_audio_ocr_cue_coverage_old_project_before.tsv'; old_after=PROJECT_ROOT/'reports/analysis/train_actual_label_audio_ocr_cue_coverage_old_project_after.tsv'; snapshot_dir(OLD_PROJECT_ROOT,old_before)
    candidate_inputs=[SPLIT_FILE,ACTUAL_FILE]+[p for _,p,_ in AUDIO_CANDIDATES]+[p for _,p,_ in OCR_CANDIDATES]+ANCHOR_CANDIDATES+list(DETECTOR_FILES.values()); input_stats_before={str(p):file_stats(p) for p in candidate_inputs if p.exists()}; split_meta=validate_split(read_csv(SPLIT_FILE))
    log('[STEP 02] Build artifact inventory'); inventory,chosen=artifact_inventory(); write_csv(OUT['inventory'],inventory,['artifact_type','candidate_role','path','exists','row_count','has_split','train_row_count','mode','used_as','column_count','columns_json','note'])
    if not chosen.get('audio'): warnings.append('no usable audio artifact found')
    if not chosen.get('ocr'): warnings.append('no usable OCR artifact found')
    if not chosen.get('anchor'): warnings.append('no visual anchor artifact found')
    log('[STEP 03] Load split and train actual intervals'); intervals=extract_actual_intervals(read_csv(ACTUAL_FILE),split_meta); windows=make_windows(intervals)
    log('[STEP 04] Compute boundary anchor proximity'); anchor_rows=compute_anchor_proximity(intervals,chosen.get('anchor')); write_csv(OUT['anchor'],anchor_rows,['actual_interval_id','video_id','split','ad_start_sec','ad_end_sec','nearest_anchor_to_ad_start_sec','distance_to_nearest_start_anchor_sec','nearest_anchor_to_ad_end_sec','distance_to_nearest_end_anchor_sec','start_anchor_within_2s','start_anchor_within_5s','start_anchor_within_10s','end_anchor_within_2s','end_anchor_within_5s','end_anchor_within_10s','boundary_anchor_coverage_status','anchor_source_file','anchor_count_in_video'])
    log('[STEP 05] Compute audio coverage for actual windows'); audio_rows=compute_audio(windows,chosen); write_csv(OUT['audio'],audio_rows,['actual_interval_id','video_id','split','window_type','window_start_sec','window_end_sec','window_duration_sec','window_mmss','audio_source_file','audio_coverage_mode','fallback_used','expected_subwindow_count','observed_subwindow_count','valid_subwindow_count','missing_subwindow_count','subwindow_coverage_ratio','duration_overlap_coverage_ratio','max_time_gap_between_audio_subwindows_sec','mean_audio_ad_like_score','median_audio_ad_like_score','high_score_subwindow_ratio','medium_or_high_score_subwindow_ratio','audio_low_ratio','audio_evidence_density_level','audio_signal_level_distribution','audio_note'])
    log('[STEP 06] Compute OCR coverage for actual windows'); ocr_rows=compute_ocr(windows,chosen); write_csv(OUT['ocr'],ocr_rows,['actual_interval_id','video_id','split','window_type','window_start_sec','window_end_sec','window_duration_sec','window_mmss','ocr_source_file','ocr_coverage_mode','fallback_used','expected_frame_count','planned_frame_count','attempted_frame_count','success_frame_count','empty_frame_count','failed_frame_count','recovered_success_count','ocr_attempt_coverage_ratio','ocr_success_or_empty_valid_ratio','ocr_success_text_ratio','ocr_failed_ratio','max_time_gap_between_ocr_frames_sec','text_box_count_sum','token_count_sum','char_count_sum','mean_confidence','ad_disclosure_count','purchase_cta_count','product_brand_count','discount_promo_count','total_ad_keyword_count','ocr_evidence_density_level','ocr_text_signal_level','ocr_note'])
    log('[STEP 07] Build interval-level cue coverage summary'); log('[STEP 08] Join detector v1.4 outcome for diagnosis'); detector_rows=build_detector_join(intervals,detector_intervals()); write_csv(OUT['detector_join'],detector_rows,['actual_interval_id','video_id','split','ad_start_sec','ad_end_sec','ad_duration_sec','high_confidence_overlap_ratio_of_actual','review_only_overlap_ratio_of_actual','open_overlap_ratio_of_actual','unresolved_overlap_ratio_of_actual','rejected_overlap_ratio_of_actual','detector_outcome_bucket','audit_only_note']); summary_rows,failure_rows=summarize_intervals(intervals,audio_rows,ocr_rows,anchor_rows,detector_rows); write_csv(OUT['interval_summary'],summary_rows,['actual_interval_id','video_id','split','ad_start_sec','ad_end_sec','ad_duration_sec','audio_body_density_level','audio_body_coverage_ratio','audio_body_medium_or_high_signal_ratio','ocr_body_density_level','ocr_body_valid_ratio','ocr_text_signal_level','ocr_total_ad_keyword_count','boundary_anchor_coverage_status','detector_outcome_bucket','diagnosis_categories','primary_diagnosis','diagnosis_note'])
    log('[STEP 09] Extract failure cases'); write_csv(OUT['failure'],failure_rows,['failure_type','review_priority','actual_interval_id','video_id','split','ad_start_sec','ad_end_sec','ad_duration_sec','audio_body_density_level','audio_body_coverage_ratio','ocr_body_density_level','ocr_body_valid_ratio','boundary_anchor_coverage_status','detector_outcome_bucket','diagnosis_categories','primary_diagnosis','review_note'])
    log('[STEP 10] Generate reports'); input_stats_after={str(p):file_stats(p) for p in candidate_inputs if p.exists()}; snapshot_dir(OLD_PROJECT_ROOT,old_after); write_reports(intervals,audio_rows,ocr_rows,anchor_rows,summary_rows,failure_rows,detector_rows,inventory,warnings,errors,backup,input_stats_before,input_stats_after,old_before,old_after,[])
    log('[STEP 12] Update latest bundle'); latest_bad=update_latest(); report=json.loads(OUT['report'].read_text(encoding='utf-8')); report['latest_for_chatgpt_forbidden_files_found']=latest_bad; write_json(OUT['report'],report); shutil.copy2(OUT['report'],OUT['latest']/OUT['report'].name); final_print(report); return 0 if not errors else 1
if __name__=='__main__': raise SystemExit(main())
