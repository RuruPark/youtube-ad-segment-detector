#!/usr/bin/env python3
"""v2.4 visual anchor 주변 OCR context evidence pack을 생성한다."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path('.')
OLD_PROJECT_ROOT = Path('./_old_project_not_included')
VERSION = 'v2_4'
TASK_NAME = 'ocr_visual_anchor_context_features'
SPLIT_SEED = 20240524
FIXED_SPLIT = {
    'train': [1,2,5,6,8,9,10,11,12,13,14,15],
    'validation': [3,7,18],
    'test': [4,16,17],
}
CANONICAL_ANCHOR = PROJECT_ROOT / 'data/features/visual_scene_boundary_anchors_v2_4.csv'
CANONICAL_ANCHOR_WITH_SPLIT = PROJECT_ROOT / 'data/features/visual_scene_boundary_anchors_v2_4_with_split.csv'
SPLIT_FILE = PROJECT_ROOT / 'data/splits/video_split_v2_4.csv'
MANIFEST_FILE = PROJECT_ROOT / 'data/video_metadata/video_manifest_v2_2.csv'
SCENE_AUDIO_TABLE = PROJECT_ROOT / 'data/fusion/scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val.csv'
AUDIO_FEATURE_TABLE = PROJECT_ROOT / 'data/audio/audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv'
AUDIO_CONFIG = PROJECT_ROOT / 'configs/audio_persistence_rule_config_v2_4_train_only.json'
LABEL_FILE = PROJECT_ROOT / 'data/segments/ad_interval_segments_v2_4.csv'
OCR_SCRIPT_REF = PROJECT_ROOT / 'scripts/ocr/extract_ocr_cues_v2_4.py'
DATA_OCR_DIR = PROJECT_ROOT / 'data/ocr'
DATA_FUSION_DIR = PROJECT_ROOT / 'data/fusion'
REPORTS_OCR_DIR = PROJECT_ROOT / 'reports/ocr'
LOGS_DIR = PROJECT_ROOT / 'logs'
SCRIPTS_OCR_DIR = PROJECT_ROOT / 'scripts/ocr'
BUNDLE_DIR = PROJECT_ROOT / 'outputs/latest_for_chatgpt_ocr_visual_anchor'
BACKUPS_DIR = PROJECT_ROOT / 'backups'
CACHE_DIR = PROJECT_ROOT / 'cache/ocr/model_cache'

OCR_REFERENCE_FILES = {
    'recommendations': DATA_OCR_DIR / 'ocr_rule_feature_recommendations_v2_4.csv',
    'extract_summary': REPORTS_OCR_DIR / 'extract_ocr_cues_v2_4_summary.md',
    'train_summary': REPORTS_OCR_DIR / 'create_train_only_ocr_feature_files_v2_4_summary.md',
    'labeled_analysis': REPORTS_OCR_DIR / 'ocr_labeled_segment_analysis_v2_4.md',
    'failed_retry_analysis': REPORTS_OCR_DIR / 'ocr_failed_retry_and_train_corpus_analysis_v2_4.md',
}
FORBIDDEN_SUFFIXES = {'.mp4','.mov','.mkv','.avi','.wav','.mp3','.m4a','.jpg','.jpeg','.png','.webp','.pt','.pth','.ckpt','.bin'}

KEYWORDS = {
    'ad_disclosure': ['광고','유료광고','유료 광고','협찬','제공','제작지원','지원','sponsor','sponsored','ad'],
    'purchase_cta': ['구매','구입','주문','결제','링크','더보기','더 보기','장바구니','바로가기','신청','가입','다운로드'],
    'discount_promo': ['할인','쿠폰','프로모션','이벤트','혜택','무료','체험','특가','세일','증정'],
    'product_brand': ['제품','브랜드','서비스','앱','어플','어플리케이션','공식몰','쇼핑몰','정기구독','구독'],
    'beauty_lifestyle_common': ['피부','화장품','크림','세럼','앰플','렌즈','영양제','다이어트','향수','의류'],
}
TOKEN_RE = re.compile(r'[가-힣]+|[A-Za-z]+(?:[A-Za-z0-9_+\-&]*[A-Za-z0-9])?|\d+(?:[.,]\d+)*|[A-Za-z가-힣]*\d[A-Za-z가-힣0-9._+\-]*')


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

REF = load_module(OCR_SCRIPT_REF, 'ocr_ref_v2_4')
FUSION_REF = load_module(PROJECT_ROOT / 'scripts/fusion/align_visual_anchor_audio_ocr_evidence_v2_4.py', 'fusion_ref_v2_4')

class TaskLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('', encoding='utf-8')
    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")

def paths() -> Dict[str, Path]:
    return {
        'frame_plan': DATA_OCR_DIR / 'ocr_visual_anchor_frame_sampling_plan_v2_4.csv',
        'frame_plan_discussion': DATA_OCR_DIR / 'ocr_visual_anchor_frame_sampling_plan_v2_4_train_val_for_discussion.csv',
        'frame_results': DATA_OCR_DIR / 'ocr_visual_anchor_frame_results_v2_4.csv',
        'frame_results_discussion': DATA_OCR_DIR / 'ocr_visual_anchor_frame_results_v2_4_train_val_for_discussion.csv',
        'box_results': DATA_OCR_DIR / 'ocr_visual_anchor_box_results_v2_4.csv',
        'context_features': DATA_OCR_DIR / 'ocr_visual_anchor_context_features_v2_4.csv',
        'context_features_discussion': DATA_OCR_DIR / 'ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv',
        'level_thresholds': DATA_OCR_DIR / 'ocr_visual_anchor_level_thresholds_v2_4_train_only.csv',
        'alignment_status': DATA_OCR_DIR / 'ocr_visual_anchor_alignment_status_v2_4.csv',
        'fusion_table': DATA_FUSION_DIR / 'scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val_with_ocr.csv',
        'report': REPORTS_OCR_DIR / 'ocr_visual_anchor_context_features_v2_4_report.json',
        'summary': REPORTS_OCR_DIR / 'ocr_visual_anchor_context_features_v2_4_summary.md',
        'run_log': LOGS_DIR / 'ocr_visual_anchor_context_features_v2_4_run_log.txt',
        'script': SCRIPTS_OCR_DIR / 'extract_ocr_visual_anchor_context_features_v2_4.py',
        'bundle_readme': BUNDLE_DIR / 'README_latest_files.md',
    }

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')

def safe_float(v: Any) -> float:
    try: v = float(v)
    except Exception: return float('nan')
    return v if np.isfinite(v) else float('nan')

def readable_seconds(seconds: float) -> str:
    minutes, sec = divmod(float(seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours: return f'{hours}h {minutes}m {sec:.1f}s'
    if minutes: return f'{minutes}m {sec:.1f}s'
    return f'{sec:.1f}s'

def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, dict): return {str(k): json_ready(v) for k,v in obj.items()}
    if isinstance(obj, list): return [json_ready(v) for v in obj]
    if isinstance(obj, tuple): return [json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating, float)):
        val=float(obj); return None if not np.isfinite(val) else val
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return obj

def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def old_project_snapshot() -> Dict[str, Any]:
    return FUSION_REF.old_project_snapshot()

def file_stats(files: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    return FUSION_REF.file_stats(files)

def stats_changed(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> List[str]:
    return FUSION_REF.stats_changed(before, after)

def mmss(seconds: Any) -> str:
    return FUSION_REF.mmss(seconds)

def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', str(text)).lower()
    normalized = re.sub(r'[^\w\s가-힣%₩$./:\-+,&()]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def compact_text(text: str) -> str:
    return re.sub(r'\s+', '', normalize_text(text))

def tokenize(text: str) -> List[str]:
    return [tok for tok in TOKEN_RE.findall(normalize_text(text)) if tok.strip()]

def polygon_area(points: Any) -> float:
    return REF.polygon_area(points)

def box_json(points: Any) -> list[list[float]]:
    return REF.json_safe_bbox(points)

def ensure_dirs() -> None:
    for d in [DATA_OCR_DIR, DATA_FUSION_DIR, REPORTS_OCR_DIR, LOGS_DIR, SCRIPTS_OCR_DIR, BUNDLE_DIR, BACKUPS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def backup_existing(out: Dict[str, Path], logger: TaskLogger) -> Optional[Path]:
    existing = [p for p in out.values() if p.exists()]
    if not existing: return None
    backup_dir = BACKUPS_DIR / f"ocr_visual_anchor_context_features_v2_4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for src in existing:
        dst = backup_dir / src.relative_to(PROJECT_ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    logger.log(f'Backed up {len(existing)} existing outputs to {backup_dir}')
    return backup_dir

def load_split(errors: List[str]) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_FILE)
    df['video_id'] = pd.to_numeric(df['video_id'], errors='coerce').astype('Int64')
    actual = {s: sorted(df.loc[df['split'].eq(s), 'video_id'].dropna().astype(int).unique().tolist()) for s in ['train','validation','test']}
    for split, expected in FIXED_SPLIT.items():
        if actual.get(split) != sorted(expected):
            errors.append(f'split mismatch for {split}: expected={expected}, actual={actual.get(split)}')
    return df

def load_anchors(split_df: pd.DataFrame, errors: List[str]) -> pd.DataFrame:
    if not CANONICAL_ANCHOR.exists():
        errors.append(f'canonical visual anchor missing: {CANONICAL_ANCHOR}')
        return pd.DataFrame()
    df = pd.read_csv(CANONICAL_ANCHOR_WITH_SPLIT if CANONICAL_ANCHOR_WITH_SPLIT.exists() else CANONICAL_ANCHOR)
    df['video_id'] = pd.to_numeric(df['video_id'], errors='coerce').astype('Int64')
    df['candidate_time_sec'] = pd.to_numeric(df.get('canonical_boundary_time_sec', df.get('candidate_time_sec')), errors='coerce')
    df['candidate_time_mmss'] = df.get('canonical_boundary_mmss', df['candidate_time_sec'].map(mmss)).astype(str)
    if 'scene_boundary_anchor_id' in df.columns:
        df['visual_anchor_id'] = df['scene_boundary_anchor_id'].astype(str)
    elif 'visual_anchor_id' in df.columns:
        df['visual_anchor_id'] = df['visual_anchor_id'].astype(str)
    else:
        df['visual_anchor_id'] = [f"v2_4_{int(v) if pd.notna(v) else 'unknown'}_{i:06d}_{int(round(safe_float(t)*1000)) if np.isfinite(safe_float(t)) else 0}" for i,(v,t) in enumerate(zip(df['video_id'], df['candidate_time_sec']), start=1)]
    if 'split' not in df.columns or df['split'].isna().any():
        split_map = {int(r['video_id']): r['split'] for _,r in split_df.dropna(subset=['video_id']).iterrows()}
        df['split'] = df['video_id'].map(lambda v: split_map.get(int(v), 'unknown') if pd.notna(v) else 'unknown')
    if df['candidate_time_sec'].isna().any(): errors.append('candidate_time_sec has nulls')
    if df['video_id'].isna().any(): errors.append('video_id has nulls')
    df['candidate_source'] = df.get('canonical_time_source', 'visual_scene_boundary_anchor')
    df['method_used'] = df.get('source_relation', 'visual_scene_boundary_anchor')
    return df

def resolve_video_paths(anchors: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_FILE)
    manifest['video_id'] = pd.to_numeric(manifest['video_id'], errors='coerce').astype('Int64')
    manifest_map = {int(r['video_id']): r for _,r in manifest.dropna(subset=['video_id']).iterrows()}
    rows=[]
    for vid in sorted(anchors['video_id'].dropna().astype(int).unique().tolist()):
        m=manifest_map.get(vid)
        path = str(m.get('video_path','')) if m is not None else ''
        dur = safe_float(m.get('duration_sec', np.nan)) if m is not None else np.nan
        title = str(m.get('video_title','')) if m is not None else ''
        if path and not Path(path).exists(): warnings.append(f'video path missing for video_id={vid}: {path}')
        rows.append({'video_id':vid, 'video_path':path, 'video_duration_sec':dur, 'video_title':title})
    return pd.DataFrame(rows)

def build_sampling_plan(anchors: pd.DataFrame, video_info: pd.DataFrame) -> pd.DataFrame:
    vmap = {int(r['video_id']): r for _,r in video_info.iterrows()}
    rows=[]
    for _,a in anchors.iterrows():
        vid=int(a['video_id']); t=safe_float(a['candidate_time_sec']); info=vmap.get(vid, {})
        dur=safe_float(info.get('video_duration_sec', np.nan)); path=str(info.get('video_path',''))
        for context, rels in [('pre_10s', [-10,-8,-6,-4,-2]), ('post_10s', [0,2,4,6,8])]:
            for idx, rel in enumerate(rels):
                ft = t + rel
                warn=[]; status='planned'
                if not np.isfinite(t) or not np.isfinite(dur) or not path:
                    status='invalid_anchor_or_video'; ft=np.nan
                else:
                    if ft < 0:
                        ft=0.0; warn.append('clipped_start')
                    if ft > dur:
                        ft=dur; warn.append('clipped_end')
                ft_round = round(safe_float(ft), 3) if np.isfinite(safe_float(ft)) else np.nan
                frame_id = f"v2_4_ocrframe_{vid}_{int(round(ft_round*1000)) if np.isfinite(ft_round) else 'nan'}"
                rows.append({
                    'frame_sample_id': frame_id,
                    'version': VERSION,
                    'split': a['split'],
                    'video_id': vid,
                    'visual_anchor_id': a['visual_anchor_id'],
                    'candidate_time_sec': t,
                    'candidate_time_mmss': a['candidate_time_mmss'],
                    'context_type': context,
                    'relative_time_to_anchor_sec': rel,
                    'frame_time_sec': ft_round,
                    'frame_time_mmss': mmss(ft_round),
                    'video_path': path,
                    'video_duration_sec': dur,
                    'sampling_status': status,
                    'sampling_warning': ';'.join(warn),
                })
    return pd.DataFrame(rows)

def unique_frame_plan(plan: pd.DataFrame) -> pd.DataFrame:
    cols = ['frame_sample_id','version','split','video_id','frame_time_sec','frame_time_mmss','video_path','video_duration_sec','sampling_status','sampling_warning']
    ded = plan[cols].drop_duplicates(subset=['video_id','frame_time_sec']).copy()
    ded.sort_values(['video_id','frame_time_sec'], inplace=True)
    return ded

def init_easyocr() -> Tuple[Any, Dict[str, Any]]:
    import easyocr
    use_gpu=False
    try:
        import torch
        use_gpu=bool(torch.cuda.is_available())
    except Exception:
        use_gpu=False
    reader = easyocr.Reader(['ko','en'], gpu=use_gpu, download_enabled=False, model_storage_directory=str(CACHE_DIR), verbose=False)
    return reader, {'ocr_engine':'easyocr', 'language_config':'ko+en', 'gpu_used':use_gpu, 'engine_version':str(getattr(easyocr,'__version__','unknown')), 'model_cache_dir':str(CACHE_DIR)}

def empty_frame(row: pd.Series, engine: str, status: str, error: str='') -> Dict[str, Any]:
    return {
        'version': VERSION, 'split': row.get('split',''), 'video_id': int(row['video_id']), 'frame_time_sec': safe_float(row['frame_time_sec']), 'frame_time_mmss': row.get('frame_time_mmss',''),
        'frame_sample_id': row['frame_sample_id'], 'ocr_engine': engine, 'ocr_status': status, 'ocr_error': error,
        'detected_text_raw':'', 'detected_text_normalized':'', 'detected_text_joined':'', 'ocr_text_box_count':0, 'ocr_token_count':0, 'ocr_char_count':0,
        'ocr_mean_confidence':np.nan, 'ocr_min_confidence':np.nan, 'ocr_max_confidence':np.nan, 'ocr_text_area_ratio':0.0, 'ocr_has_text':False,
        'bbox_available':False, 'frame_width':np.nan, 'frame_height':np.nan, 'ocr_high_conf_text_count':0, 'frame_keyword_score':0.0, 'frame_text_density_score':0.0, 'frame_ad_text_score':0.0,
    }

def keyword_counts(text: str) -> Dict[str, int]:
    norm = normalize_text(text); comp = compact_text(text)
    counts={}
    for cat, words in KEYWORDS.items():
        c=0
        for w in words:
            nw=normalize_text(w); cw=compact_text(w)
            c += norm.count(nw)
            if cw != nw:
                c += comp.count(cw)
        counts[cat]=int(c)
    return counts

def score_frame(raw: str, token_count: int, char_count: int, box_count: int, area_ratio: float) -> Tuple[float,float,float]:
    counts=keyword_counts(raw)
    total_kw=sum(counts.values())
    kw_score=min(1.0, total_kw/3.0)
    density=np.mean([min(1.0, token_count/18.0), min(1.0, char_count/120.0), min(1.0, box_count/8.0), min(1.0, area_ratio/0.08)])
    ad_score=min(1.0, 0.65*kw_score + 0.35*density)
    return float(kw_score), float(density), float(ad_score)

def run_ocr_frames(dedup: pd.DataFrame, reader: Any, engine_info: Dict[str, Any], logger: TaskLogger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows=[]; boxes=[]; total=len(dedup); processed=0
    for video_path, group in dedup.groupby('video_path', sort=False):
        p=Path(str(video_path))
        if not p.exists():
            for _,r in group.iterrows(): rows.append(empty_frame(r, engine_info['ocr_engine'], 'failed', 'video_path_missing'))
            continue
        cap=cv2.VideoCapture(str(p))
        if not cap.isOpened():
            for _,r in group.iterrows(): rows.append(empty_frame(r, engine_info['ocr_engine'], 'failed', 'video_open_failed'))
            continue
        try:
            for _,r in group.iterrows():
                processed += 1
                frame, dec_status = REF.decode_frame(cap, safe_float(r['frame_time_sec']))
                if frame is None:
                    rows.append(empty_frame(r, engine_info['ocr_engine'], 'failed', dec_status)); continue
                h,w = frame.shape[:2]
                try:
                    detections = REF.run_easyocr(reader, frame)
                except Exception as exc:
                    rows.append(empty_frame(r, engine_info['ocr_engine'], 'failed', f'{type(exc).__name__}: {exc}')); continue
                texts=[str(d.get('text','')).strip() for d in detections if str(d.get('text','')).strip()]
                raw='\n'.join(texts); norm=normalize_text(raw); joined=' | '.join(texts); tokens=tokenize(raw)
                conf=[safe_float(d.get('confidence')) for d in detections if np.isfinite(safe_float(d.get('confidence')))]
                total_area=0.0; high_conf=0
                for bi,d in enumerate(detections):
                    area=polygon_area(d.get('bbox')); total_area += area
                    cf=safe_float(d.get('confidence'))
                    if np.isfinite(cf) and cf>=0.6: high_conf += 1
                    boxes.append({'version':VERSION,'video_id':int(r['video_id']),'frame_sample_id':r['frame_sample_id'],'frame_time_sec':safe_float(r['frame_time_sec']),'box_index':bi,'text':str(d.get('text','')),'confidence':cf,'bbox_json':json.dumps(box_json(d.get('bbox')), ensure_ascii=False),'bbox_area':area})
                area_ratio=min(1.0, total_area/(h*w)) if h*w else 0.0
                fkw, fdens, fad = score_frame(raw, len(tokens), len(norm), len(detections), area_ratio)
                status='success' if texts else 'empty'
                rec=empty_frame(r, engine_info['ocr_engine'], status, '')
                rec.update({'detected_text_raw':raw,'detected_text_normalized':norm,'detected_text_joined':joined,'ocr_text_box_count':len(detections),'ocr_token_count':len(tokens),'ocr_char_count':len(norm),'ocr_mean_confidence':float(np.mean(conf)) if conf else np.nan,'ocr_min_confidence':float(np.min(conf)) if conf else np.nan,'ocr_max_confidence':float(np.max(conf)) if conf else np.nan,'ocr_text_area_ratio':area_ratio,'ocr_has_text':bool(texts),'bbox_available':bool(detections),'frame_width':w,'frame_height':h,'ocr_high_conf_text_count':high_conf,'frame_keyword_score':fkw,'frame_text_density_score':fdens,'frame_ad_text_score':fad})
                rows.append(rec)
                if processed % 100 == 0 or processed == total:
                    logger.log(f'[STEP 08] OCR progress: {processed}/{total} unique frames')
        finally:
            cap.release()
    return pd.DataFrame(rows), pd.DataFrame(boxes)

def safe_mean(series: pd.Series) -> float:
    s=pd.to_numeric(series, errors='coerce').dropna()
    return float(s.mean()) if len(s) else 0.0

def safe_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors='coerce').fillna(0).sum())

def context_side_features(group: pd.DataFrame, prefix: str) -> Dict[str, Any]:
    n=len(group)
    status=group.get('ocr_status', pd.Series(dtype=str)).astype(str)
    success=status.isin(['success','empty'])
    empty=status.eq('empty')
    failed=status.eq('failed')
    text_frames=pd.to_numeric(group.get('ocr_text_box_count', pd.Series(dtype=float)), errors='coerce').fillna(0)>0
    text='\n'.join(group.get('detected_text_normalized', pd.Series(dtype=str)).fillna('').astype(str).tolist())
    counts=keyword_counts(text)
    return {
        f'context_{prefix}_frame_count': int(n),
        f'context_{prefix}_ocr_success_count': int(success.sum()),
        f'context_{prefix}_ocr_empty_count': int(empty.sum()),
        f'context_{prefix}_ocr_failed_count': int(failed.sum()),
        f'ocr_{prefix}_10s_text_box_count': int(safe_sum(group.get('ocr_text_box_count', pd.Series(dtype=float)))),
        f'ocr_{prefix}_10s_token_count': int(safe_sum(group.get('ocr_token_count', pd.Series(dtype=float)))),
        f'ocr_{prefix}_10s_char_count': int(safe_sum(group.get('ocr_char_count', pd.Series(dtype=float)))),
        f'ocr_{prefix}_10s_text_frame_ratio': float(text_frames.mean()) if n else 0.0,
        f'ocr_{prefix}_10s_mean_confidence': safe_mean(group.get('ocr_mean_confidence', pd.Series(dtype=float))),
        f'ocr_{prefix}_10s_text_area_ratio_mean': safe_mean(group.get('ocr_text_area_ratio', pd.Series(dtype=float))),
        f'ocr_{prefix}_10s_text_density_score': safe_mean(group.get('frame_text_density_score', pd.Series(dtype=float))),
        f'ocr_{prefix}_10s_ad_text_score_raw_frame_mean': safe_mean(group.get('frame_ad_text_score', pd.Series(dtype=float))),
        f'ocr_{prefix}_10s_ad_disclosure_count': counts.get('ad_disclosure',0),
        f'ocr_{prefix}_10s_purchase_cta_count': counts.get('purchase_cta',0),
        f'ocr_{prefix}_10s_discount_promo_count': counts.get('discount_promo',0),
        f'ocr_{prefix}_10s_product_brand_count': counts.get('product_brand',0),
        f'ocr_{prefix}_10s_beauty_lifestyle_common_count': counts.get('beauty_lifestyle_common',0),
        f'ocr_{prefix}_10s_total_ad_keyword_count': int(sum(counts.values())),
    }

def train_scale_score(values: pd.Series, train_values: pd.Series) -> pd.Series:
    tv=pd.to_numeric(train_values, errors='coerce').replace([np.inf,-np.inf], np.nan).dropna()
    vals=pd.to_numeric(values, errors='coerce').replace([np.inf,-np.inf], np.nan)
    if tv.empty: return pd.Series(np.nan, index=values.index)
    lo=float(tv.quantile(0.05)); hi=float(tv.quantile(0.95))
    if not np.isfinite(hi-lo) or hi <= lo: hi=lo+1.0
    return ((vals-lo)/(hi-lo)).clip(0,1)

def percentile(value: Any, train_values: pd.Series) -> float:
    tv=pd.to_numeric(train_values, errors='coerce').replace([np.inf,-np.inf], np.nan).dropna()
    v=safe_float(value)
    if tv.empty or not np.isfinite(v): return np.nan
    return float((tv <= v).mean())

def level_from_pct(p: Any, coverage_low: bool=False) -> str:
    f=safe_float(p)
    if coverage_low and not np.isfinite(f): return 'unknown'
    if not np.isfinite(f): return 'unknown'
    if f>=0.70: return 'high'
    if f>=0.40: return 'medium'
    return 'low'

def build_context_features(anchor_df: pd.DataFrame, plan: pd.DataFrame, frame_results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merged=plan.merge(frame_results, on=['frame_sample_id','video_id','frame_time_sec'], how='left', suffixes=('','_ocr'))
    rows=[]
    for _,a in anchor_df.iterrows():
        aid=a['visual_anchor_id']
        rec={'version':VERSION,'split':a['split'],'video_id':int(a['video_id']),'visual_anchor_id':aid,'candidate_time_sec':safe_float(a['candidate_time_sec']),'candidate_time_mmss':a['candidate_time_mmss']}
        pre=merged[(merged['visual_anchor_id'].eq(aid)) & (merged['context_type'].eq('pre_10s'))]
        post=merged[(merged['visual_anchor_id'].eq(aid)) & (merged['context_type'].eq('post_10s'))]
        rec.update(context_side_features(pre, 'pre'))
        rec.update(context_side_features(post, 'post'))
        rows.append(rec)
    ctx=pd.DataFrame(rows)
    for side in ['pre','post']:
        prefix=f'ocr_{side}_10s'
        # component별 raw column
        ctx[f'{prefix}_ad_disclosure_score_raw']=pd.to_numeric(ctx[f'{prefix}_ad_disclosure_count'], errors='coerce')
        ctx[f'{prefix}_purchase_cta_score_raw']=pd.to_numeric(ctx[f'{prefix}_purchase_cta_count'], errors='coerce')
        ctx[f'{prefix}_discount_promo_score_raw']=pd.to_numeric(ctx[f'{prefix}_discount_promo_count'], errors='coerce')
        ctx[f'{prefix}_product_brand_score_raw']=pd.to_numeric(ctx[f'{prefix}_product_brand_count'], errors='coerce')
        ctx[f'{prefix}_text_density_score_raw']=pd.to_numeric(ctx[f'{prefix}_text_density_score'], errors='coerce')
    train=ctx[ctx['split'].eq('train')]
    thresholds=[]
    for side in ['pre','post']:
        prefix=f'ocr_{side}_10s'
        comps={
            'ad_disclosure': f'{prefix}_ad_disclosure_score_raw',
            'purchase_cta': f'{prefix}_purchase_cta_score_raw',
            'discount_promo': f'{prefix}_discount_promo_score_raw',
            'product_brand': f'{prefix}_product_brand_score_raw',
            'text_density': f'{prefix}_text_density_score_raw',
        }
        for comp,col in comps.items():
            score_col=f'{prefix}_{comp}_score'
            ctx[score_col]=train_scale_score(ctx[col], train[col])
            tv=pd.to_numeric(train[col], errors='coerce').dropna()
            thresholds.append({'version':VERSION,'split_basis':'train_only','split_seed':SPLIT_SEED,'score_name':score_col,'source_column':col,'train_count':int(len(tv)),'train_q05':float(tv.quantile(0.05)) if len(tv) else np.nan,'train_q95':float(tv.quantile(0.95)) if len(tv) else np.nan,'level_note':'component robust min/max from train only'})
        ctx[f'{prefix}_ad_text_score']=(0.30*ctx[f'{prefix}_ad_disclosure_score']+0.25*ctx[f'{prefix}_purchase_cta_score']+0.20*ctx[f'{prefix}_discount_promo_score']+0.15*ctx[f'{prefix}_product_brand_score']+0.10*ctx[f'{prefix}_text_density_score']).clip(0,1)
    ctx['ocr_keyword_delta_post_minus_pre']=pd.to_numeric(ctx['ocr_post_10s_total_ad_keyword_count'], errors='coerce')-pd.to_numeric(ctx['ocr_pre_10s_total_ad_keyword_count'], errors='coerce')
    ctx['ocr_keyword_delta_pre_minus_post']=-ctx['ocr_keyword_delta_post_minus_pre']
    ctx['ocr_text_density_delta_post_minus_pre']=pd.to_numeric(ctx['ocr_post_10s_text_density_score'], errors='coerce')-pd.to_numeric(ctx['ocr_pre_10s_text_density_score'], errors='coerce')
    ctx['ocr_text_density_delta_pre_minus_post']=-ctx['ocr_text_density_delta_post_minus_pre']
    ctx['ocr_char_count_delta_post_minus_pre']=pd.to_numeric(ctx['ocr_post_10s_char_count'], errors='coerce')-pd.to_numeric(ctx['ocr_pre_10s_char_count'], errors='coerce')
    ctx['ocr_char_count_delta_pre_minus_post']=-ctx['ocr_char_count_delta_post_minus_pre']
    ctx['ocr_score_delta_post_minus_pre']=ctx['ocr_post_10s_ad_text_score']-ctx['ocr_pre_10s_ad_text_score']
    ctx['ocr_score_delta_pre_minus_post']=-ctx['ocr_score_delta_post_minus_pre']
    ctx['ocr_start_signal_score']=(0.70*ctx['ocr_post_10s_ad_text_score']+0.30*ctx['ocr_score_delta_post_minus_pre'].clip(lower=0)).clip(0,1)
    ctx['ocr_end_signal_score']=(0.70*ctx['ocr_pre_10s_ad_text_score']+0.30*ctx['ocr_score_delta_pre_minus_post'].clip(lower=0)).clip(0,1)
    ctx['ocr_context_score']=pd.concat([ctx['ocr_pre_10s_ad_text_score'],ctx['ocr_post_10s_ad_text_score']], axis=1).max(axis=1)
    train=ctx[ctx['split'].eq('train')]
    for name,col in [('ocr_start_signal','ocr_start_signal_score'),('ocr_end_signal','ocr_end_signal_score'),('ocr_context','ocr_context_score')]:
        ctx[f'{name}_train_percentile']=ctx[col].map(lambda v, vals=train[col]: percentile(v, vals))
        ctx[f'{name}_level_by_train_quantile']=ctx[f'{name}_train_percentile'].map(level_from_pct)
        tv=pd.to_numeric(train[col], errors='coerce').dropna()
        thresholds.append({'version':VERSION,'split_basis':'train_only','split_seed':SPLIT_SEED,'score_name':col,'source_column':col,'train_count':int(len(tv)),'train_q40_medium_min':float(tv.quantile(0.40)) if len(tv) else np.nan,'train_q70_high_min':float(tv.quantile(0.70)) if len(tv) else np.nan,'level_note':'high if train percentile >=0.70; medium >=0.40; else low'})
    ctx['ocr_start_signal_level']=ctx['ocr_start_signal_level_by_train_quantile']
    ctx['ocr_end_signal_level']=ctx['ocr_end_signal_level_by_train_quantile']
    ctx['ocr_context_level']=ctx['ocr_context_level_by_train_quantile']
    ctx['ocr_anchor_context_status']=np.where((ctx['context_pre_frame_count']+ctx['context_post_frame_count'])>0, 'generated', 'missing')
    ctx['ocr_available_for_alignment']=ctx['ocr_anchor_context_status'].eq('generated')
    ctx['ocr_missing_reason']=''
    return ctx, pd.DataFrame(thresholds)

def support(level: str) -> str:
    return {'high':'high_discussion_support','medium':'medium_discussion_support','low':'low_or_unclear','unknown':'unknown'}.get(str(level),'unknown')

def update_fusion_table(scene_audio: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    drop_cols=[c for c in scene_audio.columns if c.startswith('ocr_') or c in ['multimodal_discussion_status','ocr_suggested_start_support','ocr_suggested_end_support','ocr_suggested_internal_ad_hint']]
    base=scene_audio.drop(columns=drop_cols, errors='ignore')
    join_cols=[c for c in ctx.columns if c not in ['version','split','video_id','candidate_time_sec','candidate_time_mmss']]
    out=base.merge(ctx[join_cols], on='visual_anchor_id', how='left')
    out['ocr_suggested_start_support']=out['ocr_start_signal_level'].map(support)
    out['ocr_suggested_end_support']=out['ocr_end_signal_level'].map(support)
    out['ocr_suggested_internal_ad_hint']=np.where(out['ocr_context_level'].isin(['high','medium']), 'possible_ocr_ad_context', 'not_indicated_by_ocr')
    out['multimodal_discussion_status']=np.where(out['ocr_anchor_context_status'].eq('generated'), 'scene_audio_ready_ocr_ready', np.where(out['ocr_available_for_alignment'].fillna(False), 'scene_audio_ready_ocr_partial', 'scene_audio_ready_ocr_missing'))
    return out

def forbidden_files() -> List[str]:
    bad=[]
    if BUNDLE_DIR.exists():
        for p in BUNDLE_DIR.rglob('*'):
            if p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES: bad.append(str(p))
            if any(part in {'cache','tmp','raw','videos'} for part in p.parts): bad.append(str(p))
    return sorted(set(bad))

def copy_bundle(out: Dict[str, Path], warnings: List[str], logger: TaskLogger) -> List[str]:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for p in BUNDLE_DIR.iterdir():
        if p.is_file() or p.is_symlink(): p.unlink()
    keys=['frame_plan_discussion','context_features_discussion','level_thresholds','alignment_status','fusion_table','summary','report','run_log','script']
    copied=[]; skipped=[]
    for k in keys:
        src=out[k]
        if src.exists():
            dst=BUNDLE_DIR/src.name; shutil.copy2(src,dst); copied.append(str(dst))
    fr=out['frame_results_discussion']
    if fr.exists() and fr.stat().st_size <= 5_000_000:
        dst=BUNDLE_DIR/fr.name; shutil.copy2(fr,dst); copied.append(str(dst))
    else:
        skipped.append(str(fr)); warnings.append(f'frame results discussion file not copied to bundle due size/text volume: {fr}')
    readme=out['bundle_readme']
    readme.write_text('\n'.join(['# OCR visual-anchor context evidence pack','',f'task: {TASK_NAME}',f'version: {VERSION}',f'canonical_anchor_file: {CANONICAL_ANCHOR}','fallback_scene_candidate_used_for_current_ocr: false','label_aligned_ocr_used_as_reference_only: true','test row-level OCR features are not included in this bundle.','', '## copied files', *[f'- {Path(x).name}' for x in copied], '', '## large files kept outside bundle', *([f'- {x}' for x in skipped] if skipped else ['- none']), '', 'This is an OCR visual-anchor evidence pack for discussion, not a final rule or interval detector.','']), encoding='utf-8')
    copied.append(str(readme)); logger.log(f'Copied {len(copied)} files to {BUNDLE_DIR}')
    return copied

def write_summary(path: Path, report: Dict[str,Any]) -> None:
    lines=['# OCR Visual-Anchor Context Features v2_4','', '## 1. 작업 개요', 'canonical visual anchor 기준 pre/post 10초 OCR context feature를 생성했다. label-aligned OCR이 아니라 detector inference 구조에 맞춘 OCR evidence pack이며 rule 확정이 아니다.','', '## 2. 입력 파일', f"- canonical visual anchor: {report['canonical_anchor_file']}", f"- scene+audio table: {report['input_scene_audio_discussion_table']}", f"- OCR source/engine: {report['ocr_engine']}", '', '## 3. OCR frame sampling', f"- sampling rows: {report['ocr_frame_sampling_row_count']}", f"- unique frame OCR rows: {report['ocr_frame_sample_count']}", f"- train/validation/test contexts: {report['visual_anchor_train_count']} / {report['visual_anchor_validation_count']} / {report['visual_anchor_test_count']}", '', '## 4. OCR feature 생성 결과', f"- OCR success/empty/failure: {report['ocr_frame_success_count']} / {report['ocr_frame_empty_count']} / {report['ocr_frame_failed_count']}", f"- OCR context feature rows: {report['ocr_context_feature_count']}", '- keyword, text density, OCR ad text score, high/medium/low level을 생성했다.', '', '## 5. scene+audio+OCR alignment', f"- join key: visual_anchor_id", f"- joined row count: {report['joined_row_count']}", f"- OCR missing rows: {report['ocr_missing_row_count']}", f"- multimodal status: {report['multimodal_discussion_status_counts']}", '', '## 6. Leakage guard', '- OCR level threshold는 train split만 사용했다.', '- validation은 discussion/audit only다.', '- test row-level feature는 bundle에서 제외했다.', '- label/audit 컬럼은 OCR score 계산에 사용하지 않았다.', '', '## 7. 한계', '- OCR 품질은 frame/자막/화면 상태에 따라 불안정할 수 있다.', '- OCR text가 없는 frame이 광고가 아님을 확정하지 않는다.', '- OCR 단독 판단 금지.', '- high/medium/low는 discussion용 후보다.', '- test는 최종 평가 전까지 보호한다.', '', '## 8. 생성 파일 목록', *[f"- {p}: {d}" for p,d in report.get('output_file_descriptions',{}).items()], '', '## 9. Sub Agent 검증 결과', *[f"- {n}: {r.get('status')} ({'; '.join(r.get('warnings', [])+r.get('errors', [])) or 'ok'})" for n,r in report.get('sub_agent_results',{}).items()], '', '## 10. 다음 단계', '- 이 채팅에서 scene/audio/OCR qualitative score를 함께 보며 state-machine interval rule을 논의한다.', '- start-like / end-like / internal-ad transition 구분과 low gap bridge rule을 논의한다.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')

def run_validations(report: Dict[str,Any], context: pd.DataFrame, fusion: pd.DataFrame) -> Dict[str,Dict[str,Any]]:
    res={}
    err=[]; warn=[]
    if not CANONICAL_ANCHOR.exists(): err.append('canonical visual anchor missing')
    if report.get('fallback_scene_candidate_used_for_current_ocr'): err.append('fallback scene candidate used')
    if report.get('visual_anchor_count',0)==0: err.append('no visual anchors')
    if not SCENE_AUDIO_TABLE.exists(): err.append('scene+audio discussion table missing')
    res['input_anchor_validation']={'status':'FAIL' if err else ('WARN' if warn else 'PASS'),'warnings':warn,'errors':err}
    err=[]; warn=[]
    if report.get('ocr_engine')!='easyocr': err.append('easyocr not used')
    if report.get('ocr_frame_failed_count',0)>0: warn.append('some OCR frames failed')
    if report.get('ocr_frame_empty_count',0)/max(1,report.get('ocr_frame_sample_count',1))>0.75: warn.append('OCR empty frame ratio high')
    if report.get('frame_images_copied_to_bundle'): err.append('frame images copied to bundle')
    res['ocr_extraction_validation']={'status':'FAIL' if err else ('WARN' if warn else 'PASS'),'warnings':warn,'errors':err}
    err=[]; warn=[]
    req=['ocr_pre_10s_total_ad_keyword_count','ocr_post_10s_total_ad_keyword_count','ocr_pre_10s_text_density_score','ocr_post_10s_text_density_score','ocr_start_signal_score','ocr_end_signal_score','ocr_context_score']
    for c in req:
        if c not in context.columns: err.append(f'missing context col {c}')
    for c in ['ocr_start_signal_score','ocr_end_signal_score','ocr_context_score']:
        vals=pd.to_numeric(context.get(c),errors='coerce').dropna()
        if len(vals) and (vals.lt(0).any() or vals.gt(1).any()): err.append(f'{c} outside 0-1')
    if report.get('validation_used_for_ocr_level_thresholds') or report.get('test_used_for_ocr_level_thresholds'): err.append('validation/test used for thresholds')
    if context['ocr_context_level'].value_counts().shape[0] <= 1: warn.append('OCR context level distribution concentrated')
    res['ocr_feature_level_validation']={'status':'FAIL' if err else ('WARN' if warn else 'PASS'),'warnings':warn,'errors':err}
    err=[]; warn=[]
    if not fusion['split'].isin(['train','validation']).all(): err.append('test/non-trainval in fusion discussion table')
    if report.get('label_columns_used_for_ocr_score'): err.append('label columns used for OCR score')
    if fusion['ocr_anchor_context_status'].isna().any(): warn.append('some OCR status missing after join')
    res['fusion_leakage_quality_validation']={'status':'FAIL' if err else ('WARN' if warn else 'PASS'),'warnings':warn,'errors':err}
    err=[]; warn=[]
    missing=[p for p in report.get('output_files',[]) if not Path(p).exists()]
    if missing: err.append(f'missing outputs: {missing}')
    if report.get('latest_for_chatgpt_forbidden_files_found'): err.append('forbidden files found')
    if report.get('input_files_modified'): err.append('input files modified')
    if report.get('old_project_modified'): err.append('old project modified')
    res['output_safety_validation']={'status':'FAIL' if err else ('WARN' if warn else 'PASS'),'warnings':warn,'errors':err}
    return res

def main() -> None:
    t0=time.time(); ensure_dirs(); out=paths(); logger=TaskLogger(out['run_log']); warnings=[]; errors=[]
    logger.log('[STEP 01] Start task and create old project snapshot')
    old_before=old_project_snapshot(); backup_dir=backup_existing(out, logger); start_time=now_iso()
    logger.log('[STEP 02] Load canonical visual anchors and split')
    split_df=load_split(errors); anchors=load_anchors(split_df, errors)
    if errors: raise RuntimeError('; '.join(errors))
    split_counts=anchors['split'].value_counts(dropna=False).to_dict(); logger.log(f"visual anchor rows={len(anchors)}, split_counts={split_counts}")
    logger.log('[STEP 03] Validate visual_anchor_id, video_id, candidate_time_sec')
    logger.log('[STEP 04] Resolve raw video paths from manifest')
    video_info=resolve_video_paths(anchors, warnings)
    missing_video=int((~video_info['video_path'].map(lambda p: Path(str(p)).exists())).sum())
    logger.log(f'missing_video_count={missing_video}')
    logger.log('[STEP 05] Inspect existing OCR scripts and OCR engine availability')
    reader, engine_info=init_easyocr(); logger.log(f"OCR engine={engine_info}")
    input_files=[CANONICAL_ANCHOR, CANONICAL_ANCHOR_WITH_SPLIT, SPLIT_FILE, MANIFEST_FILE, SCENE_AUDIO_TABLE, AUDIO_FEATURE_TABLE, AUDIO_CONFIG, OCR_SCRIPT_REF]
    input_files.extend(p for p in OCR_REFERENCE_FILES.values() if p.exists())
    input_stats_before=file_stats(input_files)
    logger.log('[STEP 06] Build visual-anchor OCR frame sampling plan')
    anchors2=anchors.merge(video_info, on='video_id', how='left')
    plan=build_sampling_plan(anchors2, video_info)
    plan.to_csv(out['frame_plan'], index=False, encoding='utf-8-sig')
    plan[plan['split'].isin(['train','validation'])].to_csv(out['frame_plan_discussion'], index=False, encoding='utf-8-sig')
    ded=unique_frame_plan(plan); logger.log(f'frame sampling rows={len(plan)}, unique OCR frames={len(ded)}')
    logger.log('[STEP 07] Extract OCR frames by OpenCV VideoCapture')
    logger.log('[STEP 08] Run OCR and generate frame-level OCR results')
    frame_results, box_results=run_ocr_frames(ded, reader, engine_info, logger)
    frame_results.to_csv(out['frame_results'], index=False, encoding='utf-8-sig')
    frame_results[frame_results['split'].isin(['train','validation'])].to_csv(out['frame_results_discussion'], index=False, encoding='utf-8-sig')
    box_results.to_csv(out['box_results'], index=False, encoding='utf-8-sig')
    success=int(frame_results['ocr_status'].eq('success').sum()); empty=int(frame_results['ocr_status'].eq('empty').sum()); failed=int(frame_results['ocr_status'].eq('failed').sum())
    logger.log(f'OCR success/empty/failure={success}/{empty}/{failed}')
    logger.log('[STEP 09] Build pre/post 10s OCR context features')
    context, thresholds=build_context_features(anchors2, plan, frame_results)
    context.to_csv(out['context_features'], index=False, encoding='utf-8-sig')
    context_disc=context[context['split'].isin(['train','validation'])].copy(); context_disc.to_csv(out['context_features_discussion'], index=False, encoding='utf-8-sig')
    thresholds.to_csv(out['level_thresholds'], index=False, encoding='utf-8-sig')
    status_df=pd.DataFrame([{'version':VERSION,'ocr_anchor_context_status':'generated','ocr_anchor_context_generated':True,'ocr_anchor_context_joined':True,'label_aligned_ocr_used_as_reference_only':True,'frame_sampling_rows':len(plan),'unique_frame_count':len(ded),'context_rows':len(context)}])
    status_df.to_csv(out['alignment_status'], index=False, encoding='utf-8-sig')
    logger.log('[STEP 10] Compute train-only OCR score levels')
    logger.log('[STEP 11] Join OCR with scene+audio discussion table')
    scene_audio=pd.read_csv(SCENE_AUDIO_TABLE)
    fusion=update_fusion_table(scene_audio, context_disc)
    fusion.to_csv(out['fusion_table'], index=False, encoding='utf-8-sig')
    logger.log(f'fusion joined rows={len(fusion)}')
    logger.log('[STEP 12] Build train/validation discussion bundle without test rows')
    input_stats_after=file_stats(input_files); old_after=old_project_snapshot(); old_modified=old_before!=old_after; input_changed=stats_changed(input_stats_before,input_stats_after)
    if old_modified: errors.append('old project snapshot changed')
    if input_changed: errors.append(f'input files modified: {input_changed}')
    output_descriptions={str(out['frame_plan']):'full visual-anchor OCR frame sampling plan; local only may include test', str(out['frame_plan_discussion']):'train/validation OCR frame sampling plan for discussion', str(out['frame_results']):'deduplicated full frame-level OCR results; local only may include test', str(out['frame_results_discussion']):'train/validation frame-level OCR results; copied only if small enough', str(out['context_features']):'full visual-anchor OCR context features; local only may include test', str(out['context_features_discussion']):'train/validation visual-anchor OCR context features', str(out['level_thresholds']):'train-only OCR level thresholds', str(out['alignment_status']):'OCR alignment status table', str(out['fusion_table']):'scene+audio+OCR discussion table train/validation with OCR', str(out['report']):'machine-readable report', str(out['summary']):'human-readable summary', str(out['run_log']):'step log', str(out['script']):'reproducible script'}
    report={'task_name':TASK_NAME,'version':VERSION,'project_root':str(PROJECT_ROOT),'start_time':start_time,'end_time':now_iso(),'actual_runtime_seconds':safe_float(time.time()-t0),'actual_runtime_readable':readable_seconds(time.time()-t0),'input_files':[str(p) for p in input_files],'output_files':[str(p) for k,p in out.items() if k!='bundle_readme'],'generated_files':[str(p) for k,p in out.items() if k!='bundle_readme'],'warnings':warnings,'errors':errors,'canonical_anchor_file':str(CANONICAL_ANCHOR),'canonical_anchor_actual_path_used':str(CANONICAL_ANCHOR),'canonical_anchor_source_dir':'data/features','fallback_scene_candidate_used_for_current_ocr':False,'label_aligned_ocr_used_as_reference_only':True,'visual_anchor_count':int(len(anchors)),'visual_anchor_train_count':int(split_counts.get('train',0)),'visual_anchor_validation_count':int(split_counts.get('validation',0)),'visual_anchor_test_count':int(split_counts.get('test',0)),'split_seed':SPLIT_SEED,'train_video_ids':FIXED_SPLIT['train'],'validation_video_ids':FIXED_SPLIT['validation'],'test_video_ids':FIXED_SPLIT['test'],'train_used_for_ocr_level_thresholds':True,'validation_used_for_ocr_level_thresholds':False,'test_used_for_ocr_level_thresholds':False,'validation_included_for_discussion_audit':True,'test_included_for_discussion_audit':False,'test_row_level_features_copied_to_bundle':False,'label_columns_used_for_ocr_score':False,'ocr_engine':engine_info.get('ocr_engine'),'ocr_engine_info':engine_info,'ocr_frame_sampling_row_count':int(len(plan)),'ocr_frame_sample_count':int(len(frame_results)),'ocr_frame_success_count':success,'ocr_frame_empty_count':empty,'ocr_frame_failed_count':failed,'ocr_context_feature_count':int(len(context)),'ocr_train_context_count':int((context['split']=='train').sum()),'ocr_validation_context_count':int((context['split']=='validation').sum()),'ocr_test_context_count':int((context['split']=='test').sum()),'ocr_context_train_val_bundle_count':int(len(context_disc)),'ocr_keyword_dictionary_source':'static_safe_seed_keywords_plus_train_only_level_scaling; validation/test text not used to add keywords','ocr_level_threshold_source':'train_split_visual_anchor_context_percentiles_only','ocr_alignment_status':'generated_and_joined','input_scene_audio_discussion_table':str(SCENE_AUDIO_TABLE),'output_scene_audio_ocr_discussion_table':str(out['fusion_table']),'joined_row_count':int(len(fusion)),'ocr_missing_row_count':int(fusion['ocr_anchor_context_status'].isna().sum()),'multimodal_discussion_status_counts':{str(k):int(v) for k,v in fusion['multimodal_discussion_status'].value_counts(dropna=False).to_dict().items()},'old_project_modified':old_modified,'old_project_snapshot_before':old_before,'old_project_snapshot_after':old_after,'input_files_modified':bool(input_changed),'input_files_modified_paths':input_changed,'latest_for_chatgpt_forbidden_files_found':[],'frame_images_copied_to_bundle':False,'backup_dir':str(backup_dir) if backup_dir else None,'output_file_descriptions':output_descriptions}
    report['ocr_unique_frame_count']=int(len(frame_results))
    report['discussion_bundle_contains_test_rows']=False
    save_json(out['report'], report); write_summary(out['summary'], report)
    logger.log('[STEP 13] Run Sub Agent validations')
    validations=run_validations(report, context, fusion); report['sub_agent_results']=validations
    if any(v['status']=='FAIL' for v in validations.values()): errors.append('one or more validation checks failed')
    report['errors']=errors; save_json(out['report'], report); write_summary(out['summary'], report)
    logger.log('[STEP 14] Update latest_for_chatgpt_ocr_visual_anchor')
    copied=copy_bundle(out, warnings, logger); forbidden=forbidden_files(); report['latest_for_chatgpt_files']=copied; report['latest_for_chatgpt_forbidden_files_found']=forbidden
    if forbidden: errors.append(f'forbidden files found in bundle: {forbidden}')
    report['warnings']=warnings; report['errors']=errors; report['end_time']=now_iso(); report['actual_runtime_seconds']=safe_float(time.time()-t0); report['actual_runtime_readable']=readable_seconds(time.time()-t0)
    save_json(out['report'], report); write_summary(out['summary'], report)
    shutil.copy2(out['report'], BUNDLE_DIR/out['report'].name); shutil.copy2(out['summary'], BUNDLE_DIR/out['summary'].name); shutil.copy2(out['run_log'], BUNDLE_DIR/out['run_log'].name)
    logger.log('[STEP 15] Print human-readable final summary')
    print('\n작업 완료 요약', flush=True)
    print(f"- status: {'FAIL' if errors else 'CONDITIONAL_SUCCESS' if warnings or any(v['status']=='WARN' for v in validations.values()) else 'SUCCESS'}", flush=True)
    print(f'- canonical visual anchor: {CANONICAL_ANCHOR}', flush=True)
    print(f"- OCR engine/source: {engine_info.get('ocr_engine')}", flush=True)
    print(f'- frame sampling rows: {len(plan)}, unique OCR frames: {len(frame_results)}', flush=True)
    print(f'- OCR success/empty/failure: {success}/{empty}/{failed}', flush=True)
    print(f'- context rows: {len(context)}, train_val rows: {len(context_disc)}', flush=True)
    print(f'- joined rows: {len(fusion)}', flush=True)
    print('- discussion bundle contains test rows: false', flush=True)
    print(f"- report: {out['report']}", flush=True)

if __name__ == '__main__':
    main()
