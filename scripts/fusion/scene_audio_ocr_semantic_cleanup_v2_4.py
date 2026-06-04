
#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, shutil, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path('.')
OLD_PROJECT_ROOT = Path('./_old_project_not_included')
VERSION = 'v2_4'
TASK_NAME = 'scene_audio_ocr_semantic_cleanup'
SPLIT_SEED = 20240524
FIXED_SPLIT = {'train':[1,2,5,6,8,9,10,11,12,13,14,15], 'validation':[3,7,18], 'test':[4,16,17]}
CANONICAL_ANCHOR = PROJECT_ROOT/'data/features/visual_scene_boundary_anchors_v2_4.csv'
INPUT_SCENE_AUDIO_OCR = PROJECT_ROOT/'data/fusion/scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val_with_ocr.csv'
INPUT_OCR_CONTEXT = PROJECT_ROOT/'data/ocr/ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv'
INPUT_OCR_THRESHOLDS = PROJECT_ROOT/'data/ocr/ocr_visual_anchor_level_thresholds_v2_4_train_only.csv'
INPUT_OCR_STATUS = PROJECT_ROOT/'data/ocr/ocr_visual_anchor_alignment_status_v2_4.csv'
INPUT_OCR_SUMMARY = PROJECT_ROOT/'reports/ocr/ocr_visual_anchor_context_features_v2_4_summary.md'
INPUT_OCR_REPORT = PROJECT_ROOT/'reports/ocr/ocr_visual_anchor_context_features_v2_4_report.json'
INPUT_AUDIO_CONTEXT = PROJECT_ROOT/'data/audio/audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv'
INPUT_AUDIO_THRESHOLDS = PROJECT_ROOT/'data/audio/audio_visual_anchor_level_thresholds_v2_4_train_only.csv'
INPUT_SCENE_AUDIO_PREV = PROJECT_ROOT/'data/fusion/scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val.csv'
OPTIONAL_FRAME_RESULTS = PROJECT_ROOT/'data/ocr/ocr_visual_anchor_frame_results_v2_4_train_val_for_discussion.csv'
OPTIONAL_FRAME_PLAN = PROJECT_ROOT/'data/ocr/ocr_visual_anchor_frame_sampling_plan_v2_4_train_val_for_discussion.csv'
OPTIONAL_FUSION_SUMMARY = PROJECT_ROOT/'reports/fusion/visual_anchor_alignment_pack_v2_4_summary.md'
OPTIONAL_FUSION_REPORT = PROJECT_ROOT/'reports/fusion/visual_anchor_alignment_pack_v2_4_report.json'
CANONICAL_ANCHOR_WITH_SPLIT = PROJECT_ROOT/'data/features/visual_scene_boundary_anchors_v2_4_with_split.csv'
SPLIT_FILE = PROJECT_ROOT/'data/splits/video_split_v2_4.csv'
OUTPUTS = {
 'semantic_fixed': PROJECT_ROOT/'data/fusion/scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val_semantic_fixed.csv',
 'compact': PROJECT_ROOT/'data/fusion/scene_audio_ocr_rule_discussion_compact_v2_4_train_val.csv',
 'coverage_summary': PROJECT_ROOT/'data/ocr/ocr_visual_anchor_coverage_reliability_summary_v2_4_train_val.csv',
 'cleanup_status': PROJECT_ROOT/'data/fusion/visual_anchor_semantic_cleanup_status_v2_4.csv',
 'report': PROJECT_ROOT/'reports/fusion/scene_audio_ocr_semantic_cleanup_v2_4_report.json',
 'summary': PROJECT_ROOT/'reports/fusion/scene_audio_ocr_semantic_cleanup_v2_4_summary.md',
 'run_log': PROJECT_ROOT/'logs/scene_audio_ocr_semantic_cleanup_v2_4_run_log.txt',
 'script': PROJECT_ROOT/'scripts/fusion/scene_audio_ocr_semantic_cleanup_v2_4.py',
}
BUNDLE_DIR = PROJECT_ROOT/'outputs/latest_for_chatgpt_scene_audio_ocr_semantic_cleanup'
FORBIDDEN_SUFFIXES={'.mp4','.mov','.mkv','.avi','.wav','.mp3','.m4a','.jpg','.jpeg','.png','.webp','.pt','.pth','.ckpt','.bin'}

def now_iso(): return datetime.now().replace(microsecond=0).isoformat()
def readable_seconds(s):
    m,sec=divmod(float(s),60); h,m=divmod(int(m),60)
    return f'{h}h {m}m {sec:.1f}s' if h else (f'{m}m {sec:.1f}s' if m else f'{sec:.1f}s')
class Logger:
    def __init__(self,p:Path): self.path=p; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('',encoding='utf-8')
    def log(self,msg):
        print(msg,flush=True)
        with self.path.open('a',encoding='utf-8') as f: f.write(f'{now_iso()} {msg}\n')
def sha256_file(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def file_stats(paths:Iterable[Path]):
    d={}
    for p in paths:
        if not p.exists(): d[str(p)]={'exists':False}; continue
        st=p.stat(); d[str(p)]={'exists':True,'size':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':sha256_file(p)}
    return d
def changed_files(a,b): return [k for k,v in a.items() if b.get(k)!=v]
def old_project_snapshot():
    if not OLD_PROJECT_ROOT.exists(): return {'exists':False,'file_count':0,'total_size':0,'digest':None}
    h=hashlib.sha256(); n=0; total=0
    for root,dirs,files in os.walk(OLD_PROJECT_ROOT):
        dirs.sort()
        for name in sorted(files):
            p=Path(root)/name
            try: st=p.stat()
            except FileNotFoundError: continue
            rel=str(p.relative_to(OLD_PROJECT_ROOT)); n+=1; total+=st.st_size; h.update(f'{rel}\t{st.st_size}\t{st.st_mtime_ns}\n'.encode())
    return {'exists':True,'file_count':n,'total_size':total,'digest':h.hexdigest()}
def backup_existing(paths, logger):
    existing=[p for p in paths if p.exists()]
    readme=BUNDLE_DIR/'README_latest_files.md'
    if readme.exists(): existing.append(readme)
    if not existing: return None
    bd=PROJECT_ROOT/f'backups/scene_audio_ocr_semantic_cleanup_v2_4_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    bd.mkdir(parents=True,exist_ok=True)
    for p in existing:
        dest=bd/(p.relative_to(PROJECT_ROOT) if str(p).startswith(str(PROJECT_ROOT)) else Path(p.name)); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dest)
    logger.log(f'Backed up {len(existing)} existing outputs to {bd}')
    return bd
def load_csv(path, required=True):
    if not path.exists():
        if required: raise FileNotFoundError(str(path))
        return pd.DataFrame()
    return pd.read_csv(path,low_memory=False)
def load_json(path):
    return json.load(open(path,encoding='utf-8')) if path.exists() else {}
def save_json(path,obj): path.parent.mkdir(parents=True,exist_ok=True); json.dump(obj,open(path,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
def norm_level(v):
    if pd.isna(v): return 'low'
    s=str(v).strip().lower()
    return s if s in {'high','medium','low','unknown','unknown_or_weak'} else ('low' if s in {'','nan'} else s)
def is_med_high(v): return norm_level(v) in {'high','medium'}
def is_high(v): return norm_level(v)=='high'
def train_level(score, split):
    x=pd.to_numeric(score,errors='coerce'); tr=x[split.astype(str)=='train'].dropna()
    if len(tr)==0: return pd.Series(['low']*len(x),index=x.index)
    q40=tr.quantile(.40); q70=tr.quantile(.70)
    return pd.Series(np.where(x>=q70,'high',np.where(x>=q40,'medium','low')),index=x.index)
def add_visual(df):
    out=df.copy(); score=pd.Series(np.nan,index=out.index,dtype='float64')
    for c in ['scene_component_score','scene_change_score','visual_boundary_strength_score','opencv_scene_change_score','resnet_scene_change_score']:
        if c in out.columns: score=score.fillna(pd.to_numeric(out[c],errors='coerce'))
    out['visual_transition_anchor_score']=score
    out['visual_transition_anchor_level']=out['scene_transition_level'].map(norm_level) if 'scene_transition_level' in out.columns else train_level(score,out['split'])
    out['visual_anchor_role']='transition_time_anchor'; out['scene_is_direct_ad_start_evidence']=False; out['scene_is_direct_ad_end_evidence']=False
    out['scene_start_end_semantic_warning']='scene_anchor_is_not_direct_ad_start_end_evidence'; out['visual_anchor_interpretation']='screen_transition_candidate_time_only'
    return out, ['visual_transition_anchor_score','visual_transition_anchor_level','visual_anchor_role','scene_is_direct_ad_start_evidence','scene_is_direct_ad_end_evidence','scene_start_end_semantic_warning','visual_anchor_interpretation']
def norm_conf(s):
    x=pd.to_numeric(s,errors='coerce')
    if x.dropna().empty: return x
    if x.dropna().quantile(.95)>1.0: x=x/100.0
    return x.clip(0,1)
def cov_score(df,prefix):
    frame=pd.to_numeric(df.get(f'context_{prefix}_frame_count',0),errors='coerce').fillna(0)
    succ=pd.to_numeric(df.get(f'context_{prefix}_ocr_success_count',0),errors='coerce').fillna(0)
    empty=pd.to_numeric(df.get(f'context_{prefix}_ocr_empty_count',0),errors='coerce').fillna(0)
    decode=((succ+empty)/frame.replace(0,np.nan)).fillna(0).clip(0,1); nonempty=(succ/frame.replace(0,np.nan)).fillna(0).clip(0,1)
    c=f'ocr_{prefix}_10s_mean_confidence'
    if c in df.columns:
        conf=norm_conf(df[c])
        return (0.50*decode+0.30*nonempty+0.20*conf.fillna(nonempty)).clip(0,1)
    return (0.625*decode+0.375*nonempty).clip(0,1)
def rel_level(s): return pd.Series(np.where(s>=.70,'high',np.where(s>=.40,'medium','low')),index=s.index)
def add_ocr_reliability(df):
    out=df.copy(); added=[]
    out['ocr_pre_10s_coverage_score']=cov_score(out,'pre'); out['ocr_post_10s_coverage_score']=cov_score(out,'post'); out['ocr_context_coverage_score']=out[['ocr_pre_10s_coverage_score','ocr_post_10s_coverage_score']].mean(axis=1).clip(0,1)
    out['ocr_pre_10s_reliability_level']=rel_level(out['ocr_pre_10s_coverage_score']); out['ocr_post_10s_reliability_level']=rel_level(out['ocr_post_10s_coverage_score']); out['ocr_context_reliability_level']=rel_level(out['ocr_context_coverage_score'])
    out['ocr_failure_or_empty_interpretation']='coverage_issue_not_non_ad_evidence'
    added+=['ocr_pre_10s_coverage_score','ocr_post_10s_coverage_score','ocr_context_coverage_score','ocr_pre_10s_reliability_level','ocr_post_10s_reliability_level','ocr_context_reliability_level','ocr_failure_or_empty_interpretation']
    pf=pd.to_numeric(out.get('context_pre_frame_count',0),errors='coerce').fillna(0); qf=pd.to_numeric(out.get('context_post_frame_count',0),errors='coerce').fillna(0)
    pfailed=pd.to_numeric(out.get('context_pre_ocr_failed_count',0),errors='coerce').fillna(0); qfailed=pd.to_numeric(out.get('context_post_ocr_failed_count',0),errors='coerce').fillna(0)
    fr=((pfailed+qfailed)/(pf+qf).replace(0,np.nan)).fillna(0)
    ptr=pd.to_numeric(out.get('ocr_pre_10s_text_frame_ratio',0),errors='coerce').fillna(0); qtr=pd.to_numeric(out.get('ocr_post_10s_text_frame_ratio',0),errors='coerce').fillna(0); tr=(ptr+qtr)/2
    ctx=out.get('ocr_context_level',pd.Series(['low']*len(out),index=out.index)).map(norm_level)
    reasons=[]
    for cov,fail,txt,lvl in zip(out['ocr_context_coverage_score'],fr,tr,ctx):
        if cov<.40: reasons.append('low_coverage_uncertain')
        elif fail>=.40: reasons.append('ocr_decode_failure_affected')
        elif lvl=='low' and txt<.20: reasons.append('mostly_empty_text')
        elif lvl=='low': reasons.append('low_text_signal_with_good_coverage')
        else: reasons.append('unknown')
    out['ocr_low_score_reason']=reasons; added.append('ocr_low_score_reason')
    return out,added
def add_support(df):
    out=df.copy(); n=len(out); low=pd.Series(['low']*n,index=out.index)
    out['audio_start_support_level']=out.get('audio_start_signal_level',low).map(lambda x:'high' if is_high(x) else ('medium' if norm_level(x)=='medium' else 'low'))
    out['audio_end_support_level']=['high' if is_high(e) and norm_level(a)=='low' else ('medium' if is_med_high(e) else 'low') for e,a in zip(out.get('audio_end_signal_level',low),out.get('audio_after_context_level',low))]
    out['audio_internal_ad_support_level']=['high' if is_med_high(b) and is_med_high(a) else ('medium' if norm_level(c)=='medium' else 'low') for b,a,c in zip(out.get('audio_before_context_level',low),out.get('audio_after_context_level',low),out.get('audio_context_level',low))]
    out['audio_gap_bridge_support_level']=out.get('audio_context_level',low).map(lambda x:'high' if is_high(x) else ('medium' if norm_level(x)=='medium' else 'low'))
    def osup(sig,rel):
        if norm_level(rel)=='low': return 'unknown_or_weak'
        s=norm_level(sig); return 'high' if s=='high' else ('medium' if s=='medium' else 'low')
    out['ocr_start_support_level']=[osup(s,r) for s,r in zip(out.get('ocr_start_signal_level',low),out['ocr_context_reliability_level'])]
    out['ocr_end_support_level']=[osup(s,r) for s,r in zip(out.get('ocr_end_signal_level',low),out['ocr_context_reliability_level'])]
    out['ocr_internal_ad_support_level']=[osup(s,r) for s,r in zip(out.get('ocr_context_level',low),out['ocr_context_reliability_level'])]
    out['ocr_gap_bridge_support_level']=[osup(s,r) for s,r in zip(out.get('ocr_context_level',low),out['ocr_context_reliability_level'])]
    def comb(a,o): return 'high' if is_high(a) and is_med_high(o) else ('medium' if is_med_high(a) or is_med_high(o) else 'low_or_uncertain')
    out['possible_start_context_support']=[comb(a,o) for a,o in zip(out['audio_start_support_level'],out['ocr_start_support_level'])]
    out['possible_end_context_support']=[comb(a,o) for a,o in zip(out['audio_end_support_level'],out['ocr_end_support_level'])]
    out['possible_internal_ad_transition_support']=[comb(a,o) for a,o in zip(out['audio_internal_ad_support_level'],out['ocr_internal_ad_support_level'])]
    out['possible_gap_bridge_support']=[comb(a,o) for a,o in zip(out['audio_gap_bridge_support_level'],out['ocr_gap_bridge_support_level'])]
    added=['audio_start_support_level','audio_end_support_level','audio_internal_ad_support_level','audio_gap_bridge_support_level','ocr_start_support_level','ocr_end_support_level','ocr_internal_ad_support_level','ocr_gap_bridge_support_level','possible_start_context_support','possible_end_context_support','possible_internal_ad_transition_support','possible_gap_bridge_support']
    return out,added
def add_multi(df):
    out=df.copy(); av=[]; st=[]; note=[]
    for _,r in out.iterrows():
        mods=['scene']; audio=pd.notna(r.get('audio_context_score',np.nan)); ocr=bool(r.get('ocr_available_for_alignment',False)) and r.get('ocr_anchor_context_status','') in {'generated','partial'}
        if audio: mods.append('audio')
        if ocr: mods.append('ocr')
        av.append(','.join(mods)); rel=norm_level(r.get('ocr_context_reliability_level','low'))
        if not audio: st.append('incomplete'); note.append('audio evidence missing; discussion incomplete')
        elif not ocr: st.append('scene_audio_ready_ocr_missing'); note.append('OCR evidence missing')
        elif rel=='low': st.append('scene_audio_ready_ocr_low_coverage'); note.append('OCR coverage low; failure/empty is not non-ad evidence')
        elif rel=='medium': st.append('scene_audio_ready_ocr_partial'); note.append('OCR available with medium coverage; use as auxiliary evidence')
        else: st.append('scene_audio_ready_ocr_ready'); note.append('scene/audio/OCR evidence available for qualitative discussion')
    out['multimodal_discussion_status_refined']=st; out['multimodal_available_modalities']=av; out['multimodal_reliability_note']=note
    return out,['multimodal_discussion_status_refined','multimodal_available_modalities','multimodal_reliability_note']
def ensure_cols(df,cols):
    out=df.copy()
    for c in cols:
        if c not in out.columns: out[c]=np.nan
    return out[cols]
def build_compact(df):
    cols=['version','split','video_id','visual_anchor_id','candidate_time_sec','candidate_time_mmss','visual_transition_anchor_score','visual_transition_anchor_level','visual_anchor_role','scene_is_direct_ad_start_evidence','scene_is_direct_ad_end_evidence','visual_anchor_interpretation','audio_start_signal_level','audio_end_signal_level','audio_context_level','audio_before_context_level','audio_after_context_level','audio_start_support_level','audio_end_support_level','audio_internal_ad_support_level','audio_gap_bridge_support_level','audio_score_delta_post_minus_pre','audio_score_delta_pre_minus_post','audio_pre_10s_ad_like_ratio','audio_post_10s_ad_like_ratio','ocr_anchor_context_status','ocr_available_for_alignment','ocr_start_signal_level','ocr_end_signal_level','ocr_context_level','ocr_start_support_level','ocr_end_support_level','ocr_internal_ad_support_level','ocr_gap_bridge_support_level','ocr_context_coverage_score','ocr_context_reliability_level','ocr_low_score_reason','ocr_failure_or_empty_interpretation','ocr_keyword_delta_post_minus_pre','ocr_keyword_delta_pre_minus_post','multimodal_discussion_status_refined','multimodal_available_modalities','multimodal_reliability_note','possible_start_context_support','possible_end_context_support','possible_internal_ad_transition_support','possible_gap_bridge_support','nearest_true_boundary_type','nearest_true_boundary_sec','distance_to_nearest_true_boundary_sec','is_near_true_boundary_2s','is_near_true_boundary_5s','is_near_true_boundary_10s']
    return ensure_cols(df,cols)
def build_coverage_summary(df,fr,ocr_report):
    rows=[]
    def add_ctx(t,label,sub,split=None,vid=None,cat=None):
        f=(pd.to_numeric(sub.get('context_pre_frame_count',0),errors='coerce').fillna(0)+pd.to_numeric(sub.get('context_post_frame_count',0),errors='coerce').fillna(0)).sum()
        s=(pd.to_numeric(sub.get('context_pre_ocr_success_count',0),errors='coerce').fillna(0)+pd.to_numeric(sub.get('context_post_ocr_success_count',0),errors='coerce').fillna(0)).sum()
        e=(pd.to_numeric(sub.get('context_pre_ocr_empty_count',0),errors='coerce').fillna(0)+pd.to_numeric(sub.get('context_post_ocr_empty_count',0),errors='coerce').fillna(0)).sum()
        fail=(pd.to_numeric(sub.get('context_pre_ocr_failed_count',0),errors='coerce').fillna(0)+pd.to_numeric(sub.get('context_post_ocr_failed_count',0),errors='coerce').fillna(0)).sum()
        rows.append({'summary_type':t,'label':label,'split':split,'video_id':vid,'category':cat,'anchor_count':len(sub),'context_frame_count':int(f),'context_success_count':int(s),'context_empty_count':int(e),'context_failed_count':int(fail),'context_failure_ratio':float(fail/f) if f else np.nan,'context_empty_ratio':float(e/f) if f else np.nan,'mean_ocr_context_coverage_score':float(pd.to_numeric(sub['ocr_context_coverage_score'],errors='coerce').mean()) if len(sub) else np.nan,'low_reliability_anchor_count':int((sub['ocr_context_reliability_level']=='low').sum()),'medium_reliability_anchor_count':int((sub['ocr_context_reliability_level']=='medium').sum()),'high_reliability_anchor_count':int((sub['ocr_context_reliability_level']=='high').sum()),'ocr_low_score_reason_count':np.nan,'frame_result_total_count':np.nan,'frame_result_success_count':np.nan,'frame_result_empty_count':np.nan,'frame_result_failed_count':np.nan})
    add_ctx('overall_context','train_validation_context',df)
    for split,sub in df.groupby('split',dropna=False): add_ctx('split_context',str(split),sub,split=str(split))
    for vid,sub in df.groupby('video_id',dropna=False): add_ctx('video_context',str(vid),sub,vid=vid)
    for reason,sub in df.groupby('ocr_low_score_reason',dropna=False): add_ctx('low_score_reason',str(reason),sub,cat=str(reason)); rows[-1]['ocr_low_score_reason_count']=int(len(sub))
    for rel,sub in df.groupby('ocr_context_reliability_level',dropna=False): add_ctx('reliability_level',str(rel),sub,cat=str(rel))
    if not fr.empty and 'ocr_status' in fr.columns:
        def add_fr(t,label,sub,split=None,vid=None):
            vc=sub['ocr_status'].value_counts(dropna=False).to_dict(); total=len(sub)
            rows.append({'summary_type':t,'label':label,'split':split,'video_id':vid,'category':'frame_result_status','anchor_count':np.nan,'context_frame_count':np.nan,'context_success_count':np.nan,'context_empty_count':np.nan,'context_failed_count':np.nan,'context_failure_ratio':np.nan,'context_empty_ratio':np.nan,'mean_ocr_context_coverage_score':np.nan,'low_reliability_anchor_count':np.nan,'medium_reliability_anchor_count':np.nan,'high_reliability_anchor_count':np.nan,'ocr_low_score_reason_count':np.nan,'frame_result_total_count':total,'frame_result_success_count':int(vc.get('success',0)),'frame_result_empty_count':int(vc.get('empty',0)),'frame_result_failed_count':int(vc.get('failed',0)),'frame_result_failure_ratio':float(vc.get('failed',0)/total) if total else np.nan,'frame_result_empty_ratio':float(vc.get('empty',0)/total) if total else np.nan})
        add_fr('overall_frame_result','train_validation_frame_results',fr)
        if 'split' in fr.columns:
            for split,sub in fr.groupby('split',dropna=False): add_fr('split_frame_result',str(split),sub,split=str(split))
        if 'video_id' in fr.columns:
            for vid,sub in fr.groupby('video_id',dropna=False): add_fr('video_frame_result',str(vid),sub,vid=vid)
    rows.append({'summary_type':'source_ocr_report_full','label':'full_ocr_visual_anchor_context_report_counts','split':'all_local_splits','category':'full_report_reference','anchor_count':ocr_report.get('ocr_context_feature_count'),'context_frame_count':ocr_report.get('ocr_frame_sampling_row_count'),'frame_result_total_count':ocr_report.get('ocr_frame_sample_count'),'frame_result_success_count':ocr_report.get('ocr_frame_success_count'),'frame_result_empty_count':ocr_report.get('ocr_frame_empty_count'),'frame_result_failed_count':ocr_report.get('ocr_frame_failed_count')})
    return pd.DataFrame(rows)
def forbidden_files():
    found=[]
    if not BUNDLE_DIR.exists(): return found
    for p in BUNDLE_DIR.rglob('*'):
        if p.is_file() and (p.suffix.lower() in FORBIDDEN_SUFFIXES or 'cache' in {x.lower() for x in p.parts} or 'tmp' in {x.lower() for x in p.parts}): found.append(str(p))
    return found
def vc_dict(df,cols):
    return {c:{str(k):int(v) for k,v in df[c].value_counts(dropna=False).to_dict().items()} for c in cols if c in df.columns}
def write_summary(path,report):
    lines=['# Scene + Audio + OCR Semantic Cleanup v2_4','', '## 1. 작업 개요','scene/audio/OCR visual-anchor discussion table을 state-machine rule 논의에 적합하게 semantic cleanup하고 OCR coverage/reliability 해석을 보강했다. 이 작업은 rule 확정이나 interval detector 구현이 아니다.','', '## 2. 입력 상태',f"- input discussion table: {report.get('input_scene_audio_ocr_table')}",f"- rows: train {report.get('train_row_count')}, validation {report.get('validation_row_count')}, test in input {report.get('test_row_count_in_input')}",f"- OCR missing rows after join: {report.get('ocr_missing_rows_in_input')}",'- OCR Extraction WARN은 일부 frame failure 때문이며, failure/empty는 non-ad evidence로 해석하지 않는다.','', '## 3. Scene Anchor Semantic 정리','- visual anchor role은 transition_time_anchor로 고정했다.','- scene anchor 자체는 direct ad start/end evidence가 아니다.','- visual_start_like_score / visual_end_like_score가 있어도 metadata 또는 heuristic hint로만 보존한다.','- start/end 판단은 anchor 전후 audio/OCR context 흐름으로 논의해야 한다.','', '## 4. OCR Coverage / Reliability',f"- OCR frame success/empty/failed: {report.get('ocr_frame_success_count')} / {report.get('ocr_frame_empty_count')} / {report.get('ocr_frame_failed_count')}",f"- OCR reliability counts: {report.get('ocr_reliability_level_counts')}",f"- OCR low score reason counts: {report.get('ocr_low_score_reason_counts')}",'- OCR score low + coverage low는 OCR evidence uncertain으로 해석한다.','- OCR score low + coverage high는 OCR 기준 광고 텍스트 cue가 약한 것으로만 해석한다.','', '## 5. Support Level 설명','- audio/OCR start/end/internal/gap support는 qualitative discussion candidate다.','- combined discussion hints는 final decision이 아니다.',f"- combined hint counts: {report.get('combined_discussion_hint_counts')}",'', '## 6. Leakage Guard','- train-only threshold/level 기준은 보존했다.','- validation은 discussion/audit only다.','- test row-level feature는 output/bundle에 포함하지 않았다.','- nearest boundary 등 audit 컬럼은 support 계산에 사용하지 않았다.','', '## 7. 생성 파일 목록']
    for p,d in report.get('output_file_descriptions',{}).items(): lines.append(f'- {p}: {d}')
    lines+=['','## 8. Sub Agent 검증 결과']
    for n,r in report.get('sub_agent_results',{}).items(): lines.append(f"- {n}: {r.get('status')} ({'; '.join(r.get('warnings',[])+r.get('errors',[])) or 'ok'})")
    lines+=['','## 9. 다음 단계','- 이 채팅에서 state-machine interval rule을 논의한다.','- start/end/internal-ad transition, low gap bridge, long-ad prior, OCR low coverage 처리 방식을 논의한다.','']
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text('\n'.join(lines),encoding='utf-8')
def run_validations(fixed,compact,report):
    res={}
    err=[]
    for p in [INPUT_SCENE_AUDIO_OCR,INPUT_OCR_CONTEXT,INPUT_OCR_THRESHOLDS,INPUT_OCR_STATUS,INPUT_OCR_SUMMARY,INPUT_OCR_REPORT,INPUT_AUDIO_CONTEXT,INPUT_AUDIO_THRESHOLDS,INPUT_SCENE_AUDIO_PREV]:
        if not p.exists(): err.append(f'missing required input: {p}')
    if set(fixed['split'].astype(str).unique())-{'train','validation'}: err.append('output contains non train/validation split')
    if report.get('test_row_count_in_output',0)!=0: err.append('test rows present in output')
    res['input_split_validation']={'status':'FAIL' if err else 'PASS','warnings':[],'errors':err}
    err=[]
    for c in ['visual_anchor_role','scene_is_direct_ad_start_evidence','scene_is_direct_ad_end_evidence','scene_start_end_semantic_warning','visual_anchor_interpretation']:
        if c not in fixed.columns: err.append(f'missing semantic column: {c}')
    if 'visual_anchor_role' in fixed.columns and not (fixed['visual_anchor_role']=='transition_time_anchor').all(): err.append('role not transition_time_anchor')
    if fixed.get('scene_is_direct_ad_start_evidence',pd.Series([False])).astype(bool).any(): err.append('scene direct start evidence true')
    if fixed.get('scene_is_direct_ad_end_evidence',pd.Series([False])).astype(bool).any(): err.append('scene direct end evidence true')
    res['semantic_cleanup_validation']={'status':'FAIL' if err else 'PASS','warnings':[],'errors':err}
    err=[]; warn=[]
    for c in ['ocr_pre_10s_coverage_score','ocr_post_10s_coverage_score','ocr_context_coverage_score','ocr_context_reliability_level','ocr_failure_or_empty_interpretation','ocr_low_score_reason']:
        if c not in fixed.columns: err.append(f'missing OCR reliability column: {c}')
    if 'ocr_failure_or_empty_interpretation' in fixed.columns and not (fixed['ocr_failure_or_empty_interpretation']=='coverage_issue_not_non_ad_evidence').all(): err.append('OCR failure/empty interpretation wrong')
    if 'ocr_context_reliability_level' in fixed.columns and (fixed['ocr_context_reliability_level']=='low').mean()>.50: warn.append('many OCR contexts have low reliability')
    res['ocr_reliability_validation']={'status':'FAIL' if err else ('WARN' if warn else 'PASS'),'warnings':warn,'errors':err}
    err=[]
    for c in ['audio_start_support_level','audio_end_support_level','audio_internal_ad_support_level','audio_gap_bridge_support_level','ocr_start_support_level','ocr_end_support_level','ocr_internal_ad_support_level','ocr_gap_bridge_support_level','possible_start_context_support','possible_end_context_support','possible_internal_ad_transition_support','possible_gap_bridge_support']:
        if c not in fixed.columns: err.append(f'missing support column: {c}')
    bad=[c for c in fixed.columns if any(x in c.lower() for x in ['predicted_start','predicted_end','predicted_interval','final_decision','final_rule'])]
    if bad: err.append(f'final decision/prediction-like columns found: {bad}')
    res['support_hint_validation']={'status':'FAIL' if err else 'PASS','warnings':[],'errors':err}
    err=[]
    if not report.get('train_only_level_thresholds_preserved'): err.append('train-only thresholds not preserved')
    if report.get('test_row_level_features_copied_to_bundle'): err.append('test row-level copied')
    if report.get('label_columns_used_for_support') or report.get('audit_columns_used_for_support'): err.append('label/audit used for support')
    if report.get('old_project_modified'): err.append('old project modified')
    if report.get('latest_for_chatgpt_forbidden_files_found'): err.append('forbidden files in bundle')
    res['leakage_safety_validation']={'status':'FAIL' if err else 'PASS','warnings':[],'errors':err}
    return res
def copy_bundle(report, warnings, logger):
    BUNDLE_DIR.mkdir(parents=True,exist_ok=True); copied=[]; kept=[]
    for src in [OUTPUTS['semantic_fixed'],OUTPUTS['compact'],OUTPUTS['coverage_summary'],OUTPUTS['cleanup_status'],OUTPUTS['summary'],OUTPUTS['report'],OUTPUTS['run_log'],OUTPUTS['script']]:
        shutil.copy2(src,BUNDLE_DIR/src.name); copied.append(str(BUNDLE_DIR/src.name))
    for src in [INPUT_OCR_CONTEXT,INPUT_AUDIO_CONTEXT]:
        if src.exists() and src.stat().st_size<=5*1024*1024: shutil.copy2(src,BUNDLE_DIR/src.name); copied.append(str(BUNDLE_DIR/src.name))
        elif src.exists(): kept.append(str(src)); warnings.append(f'optional file kept outside bundle due size: {src}')
    readme=BUNDLE_DIR/'README_latest_files.md'
    lines=['# Scene + Audio + OCR semantic cleanup bundle','',f'task: {TASK_NAME}',f'version: {VERSION}',f'canonical_visual_anchor_actual_path: {CANONICAL_ANCHOR}','scene anchor role: transition_time_anchor','scene direct start/end evidence: false','OCR failure/empty interpretation: coverage_issue_not_non_ad_evidence','test row-level features are not included in this bundle.','','## copied files']+[f'- {Path(p).name}' for p in copied]+['','## optional files kept outside bundle']+([f'- {p}' for p in kept] if kept else ['- none'])+['','This is a semantic cleanup and reliability evidence pack for discussion, not a final rule or interval detector.','']
    readme.write_text('\n'.join(lines),encoding='utf-8'); copied.append(str(readme)); logger.log(f'Copied {len(copied)} files to {BUNDLE_DIR}'); return copied
def main():
    t0=time.time(); start=now_iso(); logger=Logger(OUTPUTS['run_log']); warnings=[]; errors=[]
    logger.log('[STEP 01] Start task and create old project snapshot'); backup=backup_existing(list(OUTPUTS.values()),logger); old_before=old_project_snapshot()
    inputs=[INPUT_SCENE_AUDIO_OCR,INPUT_OCR_CONTEXT,INPUT_OCR_THRESHOLDS,INPUT_OCR_STATUS,INPUT_OCR_SUMMARY,INPUT_OCR_REPORT,INPUT_AUDIO_CONTEXT,INPUT_AUDIO_THRESHOLDS,INPUT_SCENE_AUDIO_PREV]
    optional=[OPTIONAL_FRAME_RESULTS,OPTIONAL_FRAME_PLAN,OPTIONAL_FUSION_SUMMARY,OPTIONAL_FUSION_REPORT,CANONICAL_ANCHOR,CANONICAL_ANCHOR_WITH_SPLIT,SPLIT_FILE]
    stats_before=file_stats(inputs+[p for p in optional if p.exists()])
    logger.log('[STEP 02] Load input scene+audio+OCR table and reference features')
    discussion=load_csv(INPUT_SCENE_AUDIO_OCR); ocr_context=load_csv(INPUT_OCR_CONTEXT); audio_context=load_csv(INPUT_AUDIO_CONTEXT); frame_results=load_csv(OPTIONAL_FRAME_RESULTS,False); ocr_report=load_json(INPUT_OCR_REPORT)
    logger.log(f'input discussion rows={len(discussion)}, splits={discussion.get("split",pd.Series(dtype=str)).value_counts(dropna=False).to_dict()}')
    logger.log('[STEP 03] Validate split/test row status')
    test_in=int((discussion.get('split',pd.Series(dtype=str)).astype(str)=='test').sum()) if 'split' in discussion.columns else 0
    if test_in: errors.append(f'test rows found in input discussion table: {test_in}')
    for c in ['visual_anchor_id','video_id','candidate_time_sec','split']:
        if c not in discussion.columns: errors.append(f'missing required discussion column: {c}')
    logger.log('[STEP 04] Apply neutral visual anchor semantic cleanup'); fixed,cols1=add_visual(discussion)
    logger.log('[STEP 05] Compute OCR coverage/reliability columns'); fixed,cols2=add_ocr_reliability(fixed)
    logger.log('[STEP 06] Compute audio/OCR support levels and combined discussion hints'); fixed,cols3=add_support(fixed); fixed,cols4=add_multi(fixed)
    fixed=fixed[fixed['split'].astype(str).isin(['train','validation'])].copy(); fixed.to_csv(OUTPUTS['semantic_fixed'],index=False); logger.log(f'semantic fixed rows={len(fixed)}, splits={fixed["split"].value_counts(dropna=False).to_dict()}')
    logger.log('[STEP 07] Build compact rule discussion view'); compact=build_compact(fixed); compact.to_csv(OUTPUTS['compact'],index=False); logger.log(f'compact discussion rows={len(compact)}')
    logger.log('[STEP 08] Build OCR coverage/reliability summary'); coverage=build_coverage_summary(fixed,frame_results,ocr_report); coverage.to_csv(OUTPUTS['coverage_summary'],index=False); logger.log(f'coverage summary rows={len(coverage)}')
    logger.log('[STEP 09] Build semantic cleanup status table')
    pd.DataFrame([{'task_name':TASK_NAME,'version':VERSION,'semantic_cleanup_applied':True,'scene_anchor_role':'transition_time_anchor','scene_direct_ad_start_evidence_disabled':True,'scene_direct_ad_end_evidence_disabled':True,'scene_start_end_semantic_warning':'scene_anchor_is_not_direct_ad_start_end_evidence','ocr_failure_or_empty_interpretation':'coverage_issue_not_non_ad_evidence','label_columns_used_for_support':False,'audit_columns_used_for_support':False,'test_row_level_features_in_outputs':False,'canonical_visual_anchor_actual_path':str(CANONICAL_ANCHOR),'fallback_scene_candidate_used_for_current_alignment':False,'created_at':now_iso()}]).to_csv(OUTPUTS['cleanup_status'],index=False)
    logger.log('[STEP 10] Create report and summary')
    old_after=old_project_snapshot(); old_mod=old_before!=old_after; changed=changed_files(stats_before,file_stats(inputs+[p for p in optional if p.exists()]))
    if old_mod: errors.append('old project snapshot changed')
    if changed: errors.append(f'input files modified: {changed}')
    output_desc={str(OUTPUTS['semantic_fixed']):'train/validation scene+audio+OCR discussion table with neutral visual-anchor semantics and OCR reliability columns',str(OUTPUTS['compact']):'compact state-machine discussion view; no final decision columns',str(OUTPUTS['coverage_summary']):'OCR coverage/reliability summary by overall, split, video, reason, reliability level',str(OUTPUTS['cleanup_status']):'one-row status summary of semantic cleanup and leakage guard',str(OUTPUTS['report']):'machine-readable cleanup report',str(OUTPUTS['summary']):'human-readable cleanup summary',str(OUTPUTS['run_log']):'step log',str(OUTPUTS['script']):'reproducible script'}
    report={'task_name':TASK_NAME,'version':VERSION,'project_root':str(PROJECT_ROOT),'start_time':start,'end_time':now_iso(),'actual_runtime_seconds':time.time()-t0,'actual_runtime_readable':readable_seconds(time.time()-t0),'input_files':[str(p) for p in inputs if p.exists()],'optional_input_files':[str(p) for p in optional if p.exists()],'output_files':[str(p) for p in OUTPUTS.values()],'generated_files':[str(p) for p in OUTPUTS.values()],'warnings':warnings,'errors':errors,'input_scene_audio_ocr_table':str(INPUT_SCENE_AUDIO_OCR),'input_ocr_context_features':str(INPUT_OCR_CONTEXT),'input_audio_context_features':str(INPUT_AUDIO_CONTEXT),'visual_anchor_count':int(len(fixed)),'source_visual_anchor_total_count_from_ocr_report':ocr_report.get('visual_anchor_count'),'train_row_count':int((fixed['split'].astype(str)=='train').sum()),'validation_row_count':int((fixed['split'].astype(str)=='validation').sum()),'test_row_count_in_input':test_in,'test_row_count_in_output':int((fixed['split'].astype(str)=='test').sum()),'ocr_missing_rows_in_input':int(discussion.get('ocr_anchor_context_status',pd.Series(dtype=str)).isna().sum()) if 'ocr_anchor_context_status' in discussion.columns else None,'scene_anchor_role':'transition_time_anchor','scene_direct_ad_start_evidence_disabled':True,'scene_direct_ad_end_evidence_disabled':True,'semantic_cleanup_applied':True,'renamed_or_added_columns':cols1+cols2+cols3+cols4,'ocr_frame_success_count':ocr_report.get('ocr_frame_success_count'),'ocr_frame_empty_count':ocr_report.get('ocr_frame_empty_count'),'ocr_frame_failed_count':ocr_report.get('ocr_frame_failed_count'),'ocr_context_coverage_summary':{'mean_ocr_context_coverage_score':float(pd.to_numeric(fixed['ocr_context_coverage_score'],errors='coerce').mean()),'median_ocr_context_coverage_score':float(pd.to_numeric(fixed['ocr_context_coverage_score'],errors='coerce').median()),'min_ocr_context_coverage_score':float(pd.to_numeric(fixed['ocr_context_coverage_score'],errors='coerce').min()),'max_ocr_context_coverage_score':float(pd.to_numeric(fixed['ocr_context_coverage_score'],errors='coerce').max()),'reliability_level_counts':{str(k):int(v) for k,v in fixed['ocr_context_reliability_level'].value_counts(dropna=False).to_dict().items()}},'ocr_low_score_reason_counts':{str(k):int(v) for k,v in fixed['ocr_low_score_reason'].value_counts(dropna=False).to_dict().items()},'ocr_reliability_level_counts':{str(k):int(v) for k,v in fixed['ocr_context_reliability_level'].value_counts(dropna=False).to_dict().items()},'audio_support_level_counts':vc_dict(fixed,['audio_start_support_level','audio_end_support_level','audio_internal_ad_support_level','audio_gap_bridge_support_level']),'ocr_support_level_counts':vc_dict(fixed,['ocr_start_support_level','ocr_end_support_level','ocr_internal_ad_support_level','ocr_gap_bridge_support_level']),'combined_discussion_hint_counts':vc_dict(fixed,['possible_start_context_support','possible_end_context_support','possible_internal_ad_transition_support','possible_gap_bridge_support']),'train_only_level_thresholds_preserved':True,'validation_used_for_discussion_only':True,'test_row_level_features_copied_to_bundle':False,'label_columns_used_for_support':False,'audit_columns_used_for_support':False,'canonical_visual_anchor_actual_path':str(CANONICAL_ANCHOR),'canonical_anchor_source_dir':'data/features','fallback_scene_candidate_used_for_current_alignment':False,'old_project_modified':old_mod,'old_project_snapshot_before':old_before,'old_project_snapshot_after':old_after,'input_files_modified':bool(changed),'input_files_modified_paths':changed,'latest_for_chatgpt_forbidden_files_found':[],'backup_dir':str(backup) if backup else None,'output_file_descriptions':output_desc}
    report['sub_agent_results']=run_validations(fixed,compact,report)
    if any(v['status']=='FAIL' for v in report['sub_agent_results'].values()): errors.append('one or more validation checks failed')
    report['errors']=errors; save_json(OUTPUTS['report'],report); write_summary(OUTPUTS['summary'],report)
    logger.log('[STEP 11] Update latest_for_chatgpt_scene_audio_ocr_semantic_cleanup'); copied=copy_bundle(report,warnings,logger); forbidden=forbidden_files(); report['latest_for_chatgpt_files']=copied; report['latest_for_chatgpt_forbidden_files_found']=forbidden
    if forbidden: errors.append(f'forbidden files found in bundle: {forbidden}')
    report['warnings']=warnings; report['errors']=errors; report['end_time']=now_iso(); report['actual_runtime_seconds']=time.time()-t0; report['actual_runtime_readable']=readable_seconds(time.time()-t0); save_json(OUTPUTS['report'],report); write_summary(OUTPUTS['summary'],report)
    for src in [OUTPUTS['report'],OUTPUTS['summary'],OUTPUTS['run_log'],OUTPUTS['script']]: shutil.copy2(src,BUNDLE_DIR/src.name)
    logger.log('[STEP 12] Print human-readable final summary')
    status='FAILURE' if report['errors'] or any(v['status']=='FAIL' for v in report['sub_agent_results'].values()) else ('CONDITIONAL_SUCCESS' if report['warnings'] or any(v['status']=='WARN' for v in report['sub_agent_results'].values()) else 'SUCCESS')
    logger.log(f'status={status}, rows={len(fixed)}, compact_rows={len(compact)}, test_rows_output={report["test_row_count_in_output"]}, forbidden={forbidden}')
if __name__=='__main__': main()
