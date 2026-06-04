#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, os, shutil, hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path('.')
VERSION='v2.0'
BASE_VERSION='v1.4'
DETECTOR_ID='state_machine_interval_detector_v2_0_development'
DEV_IDS={1,2,5,6,8,9,10,11,12,13,14,15}
FORBIDDEN_PATTERNS=['label','true','actual','gt','ground_truth','audit','nearest_true_boundary','distance_to_nearest_true_boundary','is_near_true_boundary','ad_overlap','is_ad_overlap','is_ad_core','is_clean_nonad','overlapping_ad_interval_ids','per_video_ad_vs_nonad_contrast_score','audio_candidate_score_for_discussion']
FORBIDDEN_SUFFIXES={'.mp4','.mov','.mkv','.avi','.webm','.wav','.mp3','.m4a','.jpg','.jpeg','.png','.webp','.parquet','.pkl','.pickle','.pt','.pth','.ckpt','.onnx','.npy','.npz','.zip','.tar','.gz','.7z'}
FORBIDDEN_DIRS={'cache','frames','frame_images','raw_video','raw_videos','videos','video_proxy','model_cache','checkpoint','checkpoints','tmp','__pycache__','.git','node_modules','venv','.venv'}
MAX_BUNDLE_BYTES=10*1024*1024
PATHS={
 'config':ROOT/'configs/detectors/state_machine_interval_detector_v2_0_development_config.json',
 'fusion_input':ROOT/'data/fusion/final_scene_audio_ocr_rule_input_v2_0_development.csv',
 'labels':ROOT/'data/segments/ad_interval_segments_v2_4.csv',
 'v14_predictions':ROOT/'data/predictions/state_machine_interval_predictions_v1_4_train.csv',
 'raw':ROOT/'data/predictions/state_machine_interval_raw_candidates_v2_0_development.csv',
 'pred':ROOT/'data/predictions/state_machine_interval_predictions_v2_0_development.csv',
 'review':ROOT/'data/predictions/state_machine_review_only_interval_candidates_v2_0_development.csv',
 'pruned':ROOT/'data/predictions/state_machine_overprediction_pruned_review_candidates_v2_0_development.csv',
 'trace':ROOT/'data/predictions/state_machine_anchor_trace_v2_0_development.csv',
 'open':ROOT/'data/predictions/state_machine_open_interval_candidates_v2_0_development.csv',
 'unresolved':ROOT/'data/predictions/state_machine_unresolved_long_open_candidates_v2_0_development.csv',
 'rejected':ROOT/'data/predictions/state_machine_rejected_long_open_candidates_v2_0_development.csv',
 'disclosure':ROOT/'data/predictions/state_machine_disclosure_notice_review_candidates_v2_0_development.csv',
 'evidence':ROOT/'data/predictions/state_machine_evidence_tier_events_v2_0_development.csv',
 'fresh':ROOT/'data/predictions/state_machine_fresh_evidence_timeout_events_v2_0_development.csv',
 'conf':ROOT/'data/predictions/state_machine_interval_confidence_filter_events_v2_0_development.csv',
 'budget':ROOT/'data/predictions/state_machine_video_level_budget_guard_events_v2_0_development.csv',
 'det_audit':ROOT/'data/predictions/state_machine_detector_development_audit_v2_0.csv',
 'video_summary':ROOT/'data/analysis/development_detector_error_video_summary_v2_0.csv',
 'interval_overlap':ROOT/'data/analysis/development_detector_error_interval_overlap_v2_0.csv',
 'trace_reason':ROOT/'data/analysis/development_detector_error_trace_reason_summary_v2_0.csv',
 'worst':ROOT/'data/analysis/development_detector_error_worst_cases_v2_0.csv',
 'issues':ROOT/'data/analysis/development_detector_rule_issue_candidates_v2_0.csv',
 'comparison':ROOT/'data/analysis/state_machine_v1_4_vs_v2_0_development_comparison.csv',
 'ratio':ROOT/'data/analysis/state_machine_v2_0_video_prediction_ratio_summary.csv',
 'summary':ROOT/'reports/detectors/state_machine_interval_detector_v2_0_development_summary.md',
 'report':ROOT/'reports/detectors/state_machine_interval_detector_v2_0_development_report.json',
 'note':ROOT/'reports/detectors/state_machine_interval_detector_v2_0_adjustment_note.md',
 'log':ROOT/'logs/state_machine_interval_detector_v2_0_development_run_log.txt',
 'bundle':ROOT/'outputs/latest_for_chatgpt_state_machine_detector_v2_0_development',
 'schema_audit':ROOT/'data/fusion/final_scene_audio_ocr_rule_input_v2_0_development_schema_audit.csv',
 'join_quality':ROOT/'data/fusion/final_scene_audio_ocr_rule_input_v2_0_development_join_quality.csv',
 'column_mapping':ROOT/'data/fusion/final_scene_audio_ocr_rule_input_v2_0_development_column_mapping.json',
 'rule_doc':ROOT/'reports/rules/scene_audio_ocr_state_machine_rule_v2_0.md',
 'rule_contract':ROOT/'reports/rules/scene_audio_ocr_state_machine_rule_v2_0_contract.json',
 'rule_plain':ROOT/'reports/rules/scene_audio_ocr_state_machine_rule_v2_0_plain_korean_summary.md',
 'rule_change':ROOT/'reports/rules/scene_audio_ocr_state_machine_rule_v2_0_vs_v1_4_change_summary.md',
 'fusion_script':ROOT/'scripts/fusion/build_final_scene_audio_ocr_rule_input_v2_0_development.py',
 'detector_script':ROOT/'scripts/detectors/run_state_machine_interval_detector_v2_0_development.py',
}
LOG=[]
def now_iso(): return datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
def log(msg):
    line=f'[{now_iso()}] {msg}'; LOG.append(line); print(line, flush=True)
    PATHS['log'].parent.mkdir(parents=True, exist_ok=True)
    with PATHS['log'].open('a',encoding='utf-8') as f: f.write(line+'\n')
def read_csv(path):
    if not path.exists(): return []
    with path.open('r',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def write_csv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore'); w.writeheader()
        for r in rows: w.writerow({c:r.get(c,'') for c in cols})
def write_json(path,obj): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def fnum(v,default=0.0):
    try:
        if v is None or v=='': return default
        x=float(v); return default if math.isnan(x) else x
    except Exception: return default
def truth(v): return str(v).strip().lower() in {'true','1','yes','y'}
def safe_ratio(n,d): return 0.0 if d<=0 else n/d
def clip(x): return max(0.0,min(1.0,x))
def has_forbidden(cols):
    bad=[]
    for c in cols:
        low=c.lower()
        for p in FORBIDDEN_PATTERNS:
            if p=='gt':
                if low=='gt' or low.startswith('gt_') or low.endswith('_gt') or '_gt_' in low: bad.append(c); break
            elif p in low: bad.append(c); break
    return sorted(set(bad))
def mmss(sec):
    s=max(0,int(round(sec))); return f'{s//60:02d}:{s%60:02d}'
def interval_overlap(a1,a2,b1,b2): return max(0.0,min(a2,b2)-max(a1,b1))
def mk_empty(path, cols): write_csv(path, [], cols)
BASE_COLS=['prediction_id','version','detector_id','base_version','original_split_v2_4','split_role_v2_5','evaluation_subset_v2_5','video_id','video_duration_sec','ad_start_sec','ad_end_sec','ad_duration_sec','start_anchor_id','end_anchor_id','start_reason','end_reason','start_confidence_level','end_confidence_level','hard_evidence_count','support_evidence_count','weak_context_count','hard_evidence_density_per_60s','max_weak_span_sec','interval_status','interval_ad_score','interval_score_density','video_relative_rank_score','start_strength_score','continuity_strength_score','end_quality_score','hard_evidence_density_score','ocr_timeline_consistency_score','audio_relative_support_score','weak_span_penalty','overlong_penalty','opening_disclosure_only_penalty','fuzzy_only_penalty','audio_only_penalty','scene_only_penalty','video_prediction_ratio_before_budget_guard','video_prediction_ratio_after_budget_guard','budget_guard_action','ultra_high_confidence','decision_feature_columns_json','forbidden_decision_columns_found','audit_columns_used_for_decision']

def score_candidate(c):
    dur=max(1.0,c['end']-c['start'])
    start_strength=0.9 if c['start_hard'] else (0.55 if c['start_support'] else 0.25)
    cont=clip((c['hard']*1.0+c['support']*0.45)/max(1,c['anchors']))
    endq=0.75 if c['end_reason']!='video_end_open' else 0.25
    density=clip(c['hard']/max(1.0,dur/60.0)/3.0)
    timeline=clip(c['timeline_hard']/max(1,c['anchors']))
    audio=clip(c['audio_support']/max(1,c['anchors']))
    weak_pen=0.12 if c['max_weak_span']>30 else 0.0
    overlong=0.10 if dur>180 and density<0.35 else 0.0
    opening=0.25 if c['opening_only'] else 0.0
    fuzzy=0.20 if c['fuzzy_only'] else 0.0
    audio_only=0.25 if c['audio_only'] else 0.0
    scene_only=0.30 if c['scene_only'] else 0.0
    score=clip(.30*start_strength+.25*cont+.15*endq+.15*density+.10*timeline+.05*audio-weak_pen-overlong-opening-fuzzy-audio_only-scene_only)
    density_score=clip(score/max(1.0,dur/60.0))
    rank=clip(.60*score+.40*density_score-(0.10 if dur>180 else 0.0))
    ultra = c['hard'] >= 3 and c['timeline_hard'] >= 2 and density >= 0.35 and score >= 0.78 and not (c['fuzzy_only'] or c['audio_only'] or c['scene_only'] or c['opening_only'])
    c['ultra'] = ultra
    c.update({'start_strength_score':start_strength,'continuity_strength_score':cont,'end_quality_score':endq,'hard_evidence_density_score':density,'ocr_timeline_consistency_score':timeline,'audio_relative_support_score':audio,'weak_span_penalty':weak_pen,'overlong_penalty':overlong,'opening_disclosure_only_penalty':opening,'fuzzy_only_penalty':fuzzy,'audio_only_penalty':audio_only,'scene_only_penalty':scene_only,'interval_ad_score':score,'interval_score_density':density_score,'video_relative_rank_score':rank,'hard_evidence_density_per_60s':c['hard']/max(1.0,dur/60.0)})
    return c

def row_to_base(c, idx, status):
    dur=max(0.0,c['end']-c['start'])
    return {'prediction_id':f'V20_{idx:05d}','version':VERSION,'detector_id':DETECTOR_ID,'base_version':BASE_VERSION,'original_split_v2_4':'train','split_role_v2_5':'development','evaluation_subset_v2_5':'none','video_id':c['video_id'],'video_duration_sec':f'{c["video_duration"]:.6f}','ad_start_sec':f'{c["start"]:.6f}','ad_end_sec':f'{c["end"]:.6f}','ad_duration_sec':f'{dur:.6f}','start_anchor_id':c['start_anchor'],'end_anchor_id':c['end_anchor'],'start_reason':c['start_reason'],'end_reason':c['end_reason'],'start_confidence_level':c['start_conf'],'end_confidence_level':c['end_conf'],'hard_evidence_count':c['hard'],'support_evidence_count':c['support'],'weak_context_count':c['weak'],'hard_evidence_density_per_60s':f'{c["hard_evidence_density_per_60s"]:.6f}','max_weak_span_sec':f'{c["max_weak_span"]:.6f}','interval_status':status,'interval_ad_score':f'{c["interval_ad_score"]:.6f}','interval_score_density':f'{c["interval_score_density"]:.6f}','video_relative_rank_score':f'{c["video_relative_rank_score"]:.6f}','start_strength_score':f'{c["start_strength_score"]:.6f}','continuity_strength_score':f'{c["continuity_strength_score"]:.6f}','end_quality_score':f'{c["end_quality_score"]:.6f}','hard_evidence_density_score':f'{c["hard_evidence_density_score"]:.6f}','ocr_timeline_consistency_score':f'{c["ocr_timeline_consistency_score"]:.6f}','audio_relative_support_score':f'{c["audio_relative_support_score"]:.6f}','weak_span_penalty':f'{c["weak_span_penalty"]:.6f}','overlong_penalty':f'{c["overlong_penalty"]:.6f}','opening_disclosure_only_penalty':f'{c["opening_disclosure_only_penalty"]:.6f}','fuzzy_only_penalty':f'{c["fuzzy_only_penalty"]:.6f}','audio_only_penalty':f'{c["audio_only_penalty"]:.6f}','scene_only_penalty':f'{c["scene_only_penalty"]:.6f}','video_prediction_ratio_before_budget_guard':'','video_prediction_ratio_after_budget_guard':'','budget_guard_action':'','ultra_high_confidence':str(c['ultra']).lower(),'decision_feature_columns_json':c['decision_cols'],'forbidden_decision_columns_found':json.dumps(c['forbidden_cols']),'audit_columns_used_for_decision':'false'}

def load_labels():
    rows=read_csv(PATHS['labels']); by=defaultdict(list)
    for r in rows:
        vid=int(float(r.get('video_id') or 0))
        if vid in DEV_IDS and r.get('segment_type')=='ad_interval':
            by[vid].append({'id':r.get('ad_interval_id') or r.get('segment_id'),'start':fnum(r.get('segment_start_sec') or r.get('ad_start_sec')),'end':fnum(r.get('segment_end_sec') or r.get('ad_end_sec'))})
    return by

def main():
    PATHS['log'].parent.mkdir(parents=True, exist_ok=True); PATHS['log'].write_text('', encoding='utf-8')
    log('STEP 11 run v2.0 detector Development Set only')
    cfg=json.loads(PATHS['config'].read_text(encoding='utf-8'))
    rows=read_csv(PATHS['fusion_input'])
    rows=[r for r in rows if int(float(r.get('video_id') or 0)) in DEV_IDS]
    if any(r.get('original_split_v2_4')!='train' or r.get('split_role_v2_5')!='development' or r.get('evaluation_subset_v2_5')!='none' for r in rows):
        raise SystemExit('Non-development row in decision input')
    by=defaultdict(list)
    for r in rows: by[int(float(r['video_id']))].append(r)
    for vid in by: by[vid].sort(key=lambda r:fnum(r['candidate_time_sec']))
    raw=[]; trace=[]; evidence=[]; disclosure=[]; fresh=[]; review=[]; open_rows=[]; unresolved=[]; rejected=[]
    interval_idx=0; review_idx=0
    for vid in sorted(by):
        state='non_ad'; pending=None; cur=None; last_hard=None; support_start_time=None; support_anchor_count=0; weak_start=None
        video_duration=fnum(by[vid][0].get('video_duration_sec'))
        for anchor_index,r in enumerate(by[vid]):
            t=fnum(r['candidate_time_sec']); aid=r['final_scene_anchor_id']; before=state
            hard=truth(r['hard_evidence_flag_v2_0']); support=truth(r['support_evidence_flag_v2_0']); weak=truth(r['weak_context_flag_v2_0'])
            opening=truth(r['opening_disclosure_notice_flag']) or (truth(r['opening_disclosure_guard_applied']) and not truth(r['opening_disclosure_confirmed_by_later_product_cta']))
            fuzzy=truth(r['ocr_fuzzy_only_review_flag'])
            audio_active=r.get('audio_start_signal_level_v2_0') in {'medium','high'}
            scene_only=(not hard and not support and not weak)
            audio_support = 1 if (r.get('audio_context_level_v2_0') in {'medium','high'} or r.get('audio_start_signal_level_v2_0') in {'medium','high'}) else 0
            timeline_hard = 1 if truth(r.get('ocr_timeline_recent_hard_evidence_flag')) else 0
            decision_cols=json.loads(r.get('decision_feature_columns_json') or '[]')
            bad=has_forbidden(decision_cols)
            reason='hold'
            if hard: last_hard=t; support_start_time=None; support_anchor_count=0
            elif support:
                if support_start_time is None: support_start_time=t
                support_anchor_count+=1
            if weak and weak_start is None: weak_start=t
            if not weak: weak_start=None
            if opening:
                disclosure.append({'candidate_id':f'DISC_{len(disclosure)+1:05d}','version':VERSION,'video_id':vid,'anchor_id':aid,'candidate_time_sec':f'{t:.6f}','reason':'opening_disclosure_notice_review_only','review_note':'opening disclosure alone is not hard start'})
            if state=='non_ad':
                if opening:
                    reason='opening_disclosure_review_only'
                elif hard:
                    state='in_ad'; cur={'video_id':vid,'video_duration':video_duration,'start':t,'start_anchor':aid,'start_reason':'hard_ocr_or_timeline_start','start_conf':'high','hard':1,'support':0,'weak':0,'anchors':1,'timeline_hard':timeline_hard,'audio_support':audio_support,'max_weak_span':0.0,'start_hard':True,'start_support':False,'opening_only':False,'fuzzy_only':False,'audio_only':False,'scene_only':False,'decision_cols':r.get('decision_feature_columns_json','[]'),'forbidden_cols':bad}; reason='start_direct_hard'
                elif support:
                    pending={'time':t,'anchor':aid,'idx':anchor_index,'reason':'support_start_pending','row':r}; state='start_pending'; reason='enter_start_pending_support'
                elif fuzzy or audio_active or scene_only:
                    review_idx+=1; review.append({'candidate_id':f'REV_{review_idx:05d}','version':VERSION,'video_id':vid,'start_sec':f'{t:.6f}','end_sec':f'{t:.6f}','duration_sec':'0','reason':'weak_or_audio_or_scene_only_start','review_note':'not enough hard evidence for start'}); reason='weak_start_review_only'
            elif state=='start_pending':
                elapsed=t-pending['time']; anchors=anchor_index-pending['idx']
                if hard and elapsed<=15 and anchors<=2:
                    state='in_ad'; cur={'video_id':vid,'video_duration':video_duration,'start':pending['time'],'start_anchor':pending['anchor'],'start_reason':'start_pending_confirmed_by_hard_evidence','start_conf':'medium','hard':1,'support':1,'weak':0,'anchors':2,'timeline_hard':timeline_hard,'audio_support':audio_support,'max_weak_span':0.0,'start_hard':True,'start_support':True,'opening_only':False,'fuzzy_only':False,'audio_only':False,'scene_only':False,'decision_cols':r.get('decision_feature_columns_json','[]'),'forbidden_cols':bad}; pending=None; reason='confirm_start_pending'
                elif elapsed>15 or anchors>2:
                    review_idx+=1; review.append({'candidate_id':f'REV_{review_idx:05d}','version':VERSION,'video_id':vid,'start_sec':f'{pending["time"]:.6f}','end_sec':f'{t:.6f}','duration_sec':f'{elapsed:.6f}','reason':'start_pending_timeout_review_only','review_note':'not confirmed within 15s or 2 anchors'})
                    state='non_ad'; pending=None; reason='cancel_start_pending_timeout'
            elif state=='in_ad':
                if cur is None: state='non_ad'
                else:
                    cur['anchors']+=1; cur['hard']+=int(hard); cur['support']+=int(support); cur['weak']+=int(weak); cur['timeline_hard']+=timeline_hard; cur['audio_support']+=audio_support; cur['forbidden_cols']=sorted(set(cur['forbidden_cols'])|set(bad))
                    if weak_start is not None: cur['max_weak_span']=max(cur['max_weak_span'],t-weak_start)
                    no_hard_elapsed=(t-(last_hard if last_hard is not None else cur['start']))
                    support_elapsed=(t-(support_start_time if support_start_time is not None else t))
                    end_support=(r.get('audio_end_signal_level_v2_0') in {'medium','high'} and r.get('ocr_end_signal_level_v2_0') in {'medium','high'})
                    if (no_hard_elapsed>30) or (support_elapsed>15 and support_anchor_count>=3) or end_support:
                        state='end_pending'; pending={'time':t,'anchor':aid,'idx':anchor_index,'reason':'freshness_or_low_flow_end_pending'}; reason='enter_end_pending'
            elif state=='end_pending':
                if hard:
                    state='in_ad'; pending=None; reason='cancel_end_pending_by_hard_evidence'
                else:
                    elapsed=t-pending['time']; end_support=(r.get('audio_end_signal_level_v2_0') in {'medium','high'} and (r.get('ocr_end_signal_level_v2_0') in {'medium','high'} or fnum(r.get('ocr_context_valid_frame_ratio'))>=0.8 and r.get('ocr_context_level_v2_0')=='low'))
                    if elapsed>=20 or end_support:
                        cur['end']=pending['time']; cur['end_anchor']=pending['anchor']; cur['end_reason']='end_pending_timeout_no_hard_evidence' if elapsed>=20 else 'known_low_flow_end_support'; cur['end_conf']='medium' if elapsed>=20 else 'high'
                        raw.append(score_candidate(cur)); interval_idx+=1
                        state='non_ad'; cur=None; pending=None; last_hard=None; support_start_time=None; support_anchor_count=0; reason='confirm_end'
            trace.append({'version':VERSION,'detector_id':DETECTOR_ID,'original_split_v2_4':'train','split_role_v2_5':'development','evaluation_subset_v2_5':'none','video_id':vid,'final_scene_anchor_id':aid,'transition_time_anchor':f'{t:.6f}','candidate_time_mmss':r.get('candidate_time_mmss'),'state_before':before,'state_after':state,'decision_type':reason,'hard_evidence_flag_v2_0':str(hard).lower(),'support_evidence_flag_v2_0':str(support).lower(),'weak_context_flag_v2_0':str(weak).lower(),'hard_evidence_reasons_v2_0':r.get('hard_evidence_reasons_v2_0'),'support_evidence_reasons_v2_0':r.get('support_evidence_reasons_v2_0'),'weak_context_reasons_v2_0':r.get('weak_context_reasons_v2_0'),'opening_disclosure_guard_applied':r.get('opening_disclosure_guard_applied'),'audio_start_signal_level_v2_0':r.get('audio_start_signal_level_v2_0'),'audio_end_signal_level_v2_0':r.get('audio_end_signal_level_v2_0'),'ocr_start_signal_level_v2_0':r.get('ocr_start_signal_level_v2_0'),'ocr_end_signal_level_v2_0':r.get('ocr_end_signal_level_v2_0'),'decision_feature_columns_json':r.get('decision_feature_columns_json'),'forbidden_decision_columns_found':json.dumps(bad),'audit_columns_used_for_decision':'false'})
            evidence.append({'version':VERSION,'video_id':vid,'final_scene_anchor_id':aid,'transition_time_anchor':f'{t:.6f}','hard_evidence_flag_v2_0':str(hard).lower(),'support_evidence_flag_v2_0':str(support).lower(),'weak_context_flag_v2_0':str(weak).lower(),'hard_evidence_reasons_v2_0':r.get('hard_evidence_reasons_v2_0'),'support_evidence_reasons_v2_0':r.get('support_evidence_reasons_v2_0'),'weak_context_reasons_v2_0':r.get('weak_context_reasons_v2_0'),'evidence_tier_summary_v2_0':r.get('evidence_tier_summary_v2_0')})
            if reason=='enter_end_pending': fresh.append({'version':VERSION,'video_id':vid,'anchor_id':aid,'event_time_sec':f'{t:.6f}','event_reason':'freshness_or_support_cap_timeout','max_sec_since_last_hard_evidence':'30','actual_label_used':'false'})
        if state in {'in_ad','end_pending'} and cur:
            last_t=fnum(by[vid][-1]['candidate_time_sec'])
            dur=video_duration or last_t
            open_rows.append({'open_candidate_id':f'OPEN_{len(open_rows)+1:05d}','version':VERSION,'video_id':vid,'ad_start_sec':f'{cur["start"]:.6f}','last_anchor_sec':f'{last_t:.6f}','duration_proxy_sec':f'{max(0,last_t-cur["start"]):.6f}','video_duration_sec':f'{dur:.6f}','duration_ratio':f'{safe_ratio(max(0,last_t-cur["start"]),dur):.6f}','start_anchor_id':cur['start_anchor'],'last_anchor_id':by[vid][-1]['final_scene_anchor_id'],'open_reason':'video_end_open_candidate','review_note':'not closed as prediction'})
            if max(0,last_t-cur['start'])>180:
                unresolved.append({'unresolved_candidate_id':f'UNRES_{len(unresolved)+1:05d}','version':VERSION,'video_id':vid,'start_sec':f'{cur["start"]:.6f}','last_anchor_sec':f'{last_t:.6f}','duration_proxy_sec':f'{max(0,last_t-cur["start"]):.6f}','unresolved_reason':'long_open_candidate_not_closed','review_note':'requires manual review'})
    raw_rows=[row_to_base(c,i+1,'raw_closed_candidate') for i,c in enumerate(raw)]
    pre=[]; conf_events=[]
    for i,c in enumerate(raw):
        ok=c['start_conf'] in {'medium','high'} and c['end_conf'] in {'medium','high'} and c['hard']>=1 and c['max_weak_span']<=30 and not (c['fuzzy_only'] or c['audio_only'] or c['scene_only'] or c['opening_only']) and not c['forbidden_cols'] and c['interval_ad_score']>=0.35
        result='passes_confidence_filter' if ok else 'demoted_to_review_only'
        row=row_to_base(c,i+1,'pre_budget_prediction' if ok else 'review_only_confidence_filter')
        conf_events.append({'version':VERSION,'video_id':c['video_id'],'raw_interval_id':row['prediction_id'],'start_sec':row['ad_start_sec'],'end_sec':row['ad_end_sec'],'filter_result':result,'interval_ad_score':row['interval_ad_score'],'review_only_reasons':'' if ok else 'confidence_filter_failed'})
        if ok: pre.append((c,row))
        else:
            review_idx+=1; rr={'candidate_id':f'REV_{review_idx:05d}','version':VERSION,'video_id':c['video_id'],'start_sec':row['ad_start_sec'],'end_sec':row['ad_end_sec'],'duration_sec':row['ad_duration_sec'],'reason':'confidence_filter_demoted','review_note':'raw candidate failed v2.0 interval confidence filter'}; review.append(rr)
    final=[]; pruned=[]; budget_events=[]; ratio_rows=[]
    by_pred=defaultdict(list)
    for c,row in pre: by_pred[c['video_id']].append((c,row))
    cfg=json.loads(PATHS['config'].read_text(encoding='utf-8'))
    soft=cfg['rule_contract']['video_level_overprediction_guard']['soft_overprediction_ratio']; hard_ratio=cfg['rule_contract']['video_level_overprediction_guard']['hard_overprediction_ratio']; target=cfg['rule_contract']['video_level_overprediction_guard']['target_prediction_ratio_after_pruning']
    for vid,items in sorted(by_pred.items()):
        dur=items[0][0]['video_duration'] or 1.0; raw_dur=sum(max(0,c['end']-c['start']) for c,_ in items); raw_ratio=safe_ratio(raw_dur,dur); guard=raw_ratio>soft
        kept=[]; demoted=[]; running=0.0
        sorted_items=sorted(items,key=lambda x:x[0]['video_relative_rank_score'],reverse=True)
        for c,row in sorted_items:
            cd=max(0,c['end']-c['start']); ultra=c['ultra'] or (c['hard']>=2 and c['interval_score_density']>=0.18 and c['interval_ad_score']>=0.65)
            if not guard:
                action='kept_no_budget_guard_needed'; kept.append((c,row,action)); running+=cd
            elif safe_ratio(running+cd,dur)<=target or ultra:
                action='kept_ultra_high_confidence' if ultra and safe_ratio(running+cd,dur)>target else 'kept_under_budget'; kept.append((c,row,action)); running+=cd
            else:
                action='demoted_low_density_long_candidate' if cd>120 and c['interval_score_density']<0.2 else 'demoted_by_video_overprediction_guard'; demoted.append((c,row,action))
        final_ratio=safe_ratio(running,dur)
        for c,row,action in kept:
            row=row.copy(); row['interval_status']='prediction'; row['video_prediction_ratio_before_budget_guard']=f'{raw_ratio:.6f}'; row['video_prediction_ratio_after_budget_guard']=f'{final_ratio:.6f}'; row['budget_guard_action']=action; final.append(row)
        for c,row,action in demoted:
            row=row.copy(); row['interval_status']='overprediction_pruned_review'; row['video_prediction_ratio_before_budget_guard']=f'{raw_ratio:.6f}'; row['video_prediction_ratio_after_budget_guard']=f'{final_ratio:.6f}'; row['budget_guard_action']=action; pruned.append(row)
            review_idx+=1; review.append({'candidate_id':f'REV_{review_idx:05d}','version':VERSION,'video_id':c['video_id'],'start_sec':row['ad_start_sec'],'end_sec':row['ad_end_sec'],'duration_sec':row['ad_duration_sec'],'reason':action,'review_note':'demoted by video-level overprediction guard'})
        budget_events.append({'version':VERSION,'video_id':vid,'video_duration_sec':f'{dur:.6f}','raw_prediction_total_duration_sec':f'{raw_dur:.6f}','raw_prediction_ratio':f'{raw_ratio:.6f}','final_prediction_total_duration_sec':f'{running:.6f}','final_prediction_ratio':f'{final_ratio:.6f}','soft_overprediction_ratio':soft,'hard_overprediction_ratio':hard_ratio,'target_prediction_ratio_after_pruning':target,'budget_guard_triggered':str(guard).lower(),'budget_guard_action':'applied' if guard else 'not_needed'})
        ratio_rows.append(budget_events[-1].copy())
    write_csv(PATHS['raw'], raw_rows, BASE_COLS)
    write_csv(PATHS['pred'], final, BASE_COLS)
    write_csv(PATHS['pruned'], pruned, BASE_COLS)
    write_csv(PATHS['review'], review, ['candidate_id','version','video_id','start_sec','end_sec','duration_sec','reason','review_note'])
    write_csv(PATHS['trace'], trace, ['version','detector_id','original_split_v2_4','split_role_v2_5','evaluation_subset_v2_5','video_id','final_scene_anchor_id','transition_time_anchor','candidate_time_mmss','state_before','state_after','decision_type','hard_evidence_flag_v2_0','support_evidence_flag_v2_0','weak_context_flag_v2_0','hard_evidence_reasons_v2_0','support_evidence_reasons_v2_0','weak_context_reasons_v2_0','opening_disclosure_guard_applied','audio_start_signal_level_v2_0','audio_end_signal_level_v2_0','ocr_start_signal_level_v2_0','ocr_end_signal_level_v2_0','decision_feature_columns_json','forbidden_decision_columns_found','audit_columns_used_for_decision'])
    write_csv(PATHS['open'], open_rows, ['open_candidate_id','version','video_id','ad_start_sec','last_anchor_sec','duration_proxy_sec','video_duration_sec','duration_ratio','start_anchor_id','last_anchor_id','open_reason','review_note'])
    write_csv(PATHS['unresolved'], unresolved, ['unresolved_candidate_id','version','video_id','start_sec','last_anchor_sec','duration_proxy_sec','unresolved_reason','review_note'])
    mk_empty(PATHS['rejected'], ['rejected_candidate_id','version','video_id','start_sec','last_anchor_sec','duration_proxy_sec','rejected_reason','review_note'])
    write_csv(PATHS['disclosure'], disclosure, ['candidate_id','version','video_id','anchor_id','candidate_time_sec','reason','review_note'])
    write_csv(PATHS['evidence'], evidence, ['version','video_id','final_scene_anchor_id','transition_time_anchor','hard_evidence_flag_v2_0','support_evidence_flag_v2_0','weak_context_flag_v2_0','hard_evidence_reasons_v2_0','support_evidence_reasons_v2_0','weak_context_reasons_v2_0','evidence_tier_summary_v2_0'])
    write_csv(PATHS['fresh'], fresh, ['version','video_id','anchor_id','event_time_sec','event_reason','max_sec_since_last_hard_evidence','actual_label_used'])
    write_csv(PATHS['conf'], conf_events, ['version','video_id','raw_interval_id','start_sec','end_sec','filter_result','interval_ad_score','review_only_reasons'])
    write_csv(PATHS['budget'], budget_events, ['version','video_id','video_duration_sec','raw_prediction_total_duration_sec','raw_prediction_ratio','final_prediction_total_duration_sec','final_prediction_ratio','soft_overprediction_ratio','hard_overprediction_ratio','target_prediction_ratio_after_pruning','budget_guard_triggered','budget_guard_action'])
    write_csv(PATHS['ratio'], ratio_rows, ['version','video_id','video_duration_sec','raw_prediction_total_duration_sec','raw_prediction_ratio','final_prediction_total_duration_sec','final_prediction_ratio','soft_overprediction_ratio','hard_overprediction_ratio','target_prediction_ratio_after_pruning','budget_guard_triggered','budget_guard_action'])
    labels=load_labels(); video_summary=[]; interval_rows=[]
    final_by_video=defaultdict(list)
    for r in final: final_by_video[int(float(r['video_id']))].append(r)
    for vid in sorted(DEV_IDS):
        dur=fnum((by.get(vid) or [{'video_duration_sec':0}])[0].get('video_duration_sec')) if by.get(vid) else 0.0
        labs=labels.get(vid,[]); preds=final_by_video.get(vid,[])
        actual_total=sum(max(0,l['end']-l['start']) for l in labs); pred_total=sum(fnum(p['ad_duration_sec']) for p in preds); overlap=0.0
        for p in preds:
            ps=fnum(p['ad_start_sec']); pe=fnum(p['ad_end_sec']); best=0; bid=''
            for l in labs:
                ov=interval_overlap(ps,pe,l['start'],l['end']); overlap+=ov; 
                if ov>best: best=ov; bid=l['id']
            interval_rows.append({'video_id':vid,'prediction_id':p['prediction_id'],'prediction_start_sec':p['ad_start_sec'],'prediction_end_sec':p['ad_end_sec'],'best_matching_actual_id':bid,'best_actual_overlap_sec':f'{best:.6f}','posthoc_audit_only':'true'})
        video_summary.append({'video_id':vid,'split_role_v2_5':'development','video_duration_sec':f'{dur:.6f}','actual_interval_count':len(labs),'actual_total_duration_sec':f'{actual_total:.6f}','prediction_count':len(preds),'prediction_total_duration_sec':f'{pred_total:.6f}','overlap_duration_sec':f'{overlap:.6f}','false_positive_duration_sec':f'{max(0,pred_total-overlap):.6f}','missed_actual_duration_sec':f'{max(0,actual_total-overlap):.6f}','actual_label_used_for_decision':'false','actual_label_used_for_posthoc_audit':'true','final_performance_claim':'none_development_posthoc_audit_only'})
    write_csv(PATHS['video_summary'], video_summary, ['video_id','split_role_v2_5','video_duration_sec','actual_interval_count','actual_total_duration_sec','prediction_count','prediction_total_duration_sec','overlap_duration_sec','false_positive_duration_sec','missed_actual_duration_sec','actual_label_used_for_decision','actual_label_used_for_posthoc_audit','final_performance_claim'])
    write_csv(PATHS['interval_overlap'], interval_rows, ['video_id','prediction_id','prediction_start_sec','prediction_end_sec','best_matching_actual_id','best_actual_overlap_sec','posthoc_audit_only'])
    cnt=Counter((r['decision_type'],r['state_before'],r['state_after']) for r in trace)
    write_csv(PATHS['trace_reason'], [{'scope':'development','key':'|'.join(k),'count':v} for k,v in cnt.items()], ['scope','key','count'])
    worst=sorted(video_summary,key=lambda r:fnum(r['false_positive_duration_sec'])+fnum(r['missed_actual_duration_sec']),reverse=True)[:20]
    write_csv(PATHS['worst'], [{'rank':i+1, **r} for i,r in enumerate(worst)], ['rank','video_id','split_role_v2_5','video_duration_sec','actual_interval_count','actual_total_duration_sec','prediction_count','prediction_total_duration_sec','overlap_duration_sec','false_positive_duration_sec','missed_actual_duration_sec','actual_label_used_for_decision','actual_label_used_for_posthoc_audit','final_performance_claim'])
    issues=[]
    if len(pruned)>0: issues.append({'issue_id':'V20_ISSUE_001','issue_name':'overprediction_guard_demotions','evidence_from_development':f'{len(pruned)} candidates demoted','suspected_rule_area':'video_level_budget_guard','suggested_direction':'review demoted intervals before changing thresholds','caution':'Development audit only'})
    if len(disclosure)>0: issues.append({'issue_id':'V20_ISSUE_002','issue_name':'opening_disclosure_review_cases','evidence_from_development':f'{len(disclosure)} opening disclosure notices','suspected_rule_area':'opening_disclosure_guard','suggested_direction':'inspect early product/CTA confirmation','caution':'do not use opening disclosure alone'})
    write_csv(PATHS['issues'], issues, ['issue_id','issue_name','evidence_from_development','suspected_rule_area','suggested_direction','caution'])
    comp=[]
    v14=read_csv(PATHS['v14_predictions'])
    v14_by=defaultdict(list)
    for r in v14:
        try: v14_by[int(float(r.get('video_id')))].append(r)
        except Exception: pass
    for vid in sorted(DEV_IDS):
        v14_count=len(v14_by.get(vid,[])); v14_dur=sum(fnum(r.get('ad_duration_sec')) for r in v14_by.get(vid,[])); v20_count=len(final_by_video.get(vid,[])); v20_dur=sum(fnum(r.get('ad_duration_sec')) for r in final_by_video.get(vid,[])); dur=fnum((by.get(vid) or [{'video_duration_sec':0}])[0].get('video_duration_sec')) if by.get(vid) else 0
        comp.append({'video_id':vid,'split_role_v2_5':'development','v1_4_prediction_count':v14_count,'v1_4_prediction_duration_sec':f'{v14_dur:.6f}','v1_4_prediction_ratio':f'{safe_ratio(v14_dur,dur):.6f}','v2_0_prediction_count':v20_count,'v2_0_prediction_duration_sec':f'{v20_dur:.6f}','v2_0_prediction_ratio':f'{safe_ratio(v20_dur,dur):.6f}','comparison_note':'existing v1.4 prediction vs v2.0 development prediction; no validation/test'})
    write_csv(PATHS['comparison'], comp, ['video_id','split_role_v2_5','v1_4_prediction_count','v1_4_prediction_duration_sec','v1_4_prediction_ratio','v2_0_prediction_count','v2_0_prediction_duration_sec','v2_0_prediction_ratio','comparison_note'])
    hard_count=sum(1 for r in evidence if truth(r['hard_evidence_flag_v2_0'])); support_count=sum(1 for r in evidence if truth(r['support_evidence_flag_v2_0'])); weak_count=sum(1 for r in evidence if truth(r['weak_context_flag_v2_0']))
    videos_over_soft=[int(e['video_id']) for e in budget_events if fnum(e['raw_prediction_ratio'])>soft]
    videos_over_hard=[int(e['video_id']) for e in budget_events if fnum(e['raw_prediction_ratio'])>hard_ratio]
    mean_raw=sum(fnum(e['raw_prediction_ratio']) for e in budget_events)/len(budget_events) if budget_events else 0
    mean_final=sum(fnum(e['final_prediction_ratio']) for e in budget_events)/len(budget_events) if budget_events else 0
    forbidden_decision=[]
    for r in rows:
        forbidden_decision.extend(has_forbidden(json.loads(r.get('decision_feature_columns_json') or '[]')))
    forbidden_decision=sorted(set(forbidden_decision))
    audit_rows=[{'version':VERSION,'detector_id':DETECTOR_ID,'split_role_v2_5':'development','development_video_ids_json':json.dumps(sorted(DEV_IDS)),'fusion_input_rows':len(rows),'raw_candidate_count':len(raw_rows),'prediction_count_before_budget_guard':len(pre),'prediction_count_after_budget_guard':len(final),'review_only_count':len(review),'overprediction_pruned_review_count':len(pruned),'open_count':len(open_rows),'unresolved_count':len(unresolved),'rejected_count':0,'hard_evidence_anchor_count':hard_count,'support_evidence_anchor_count':support_count,'weak_evidence_anchor_count':weak_count,'budget_guard_triggered_video_count':len(videos_over_soft),'forbidden_decision_columns_found':json.dumps(forbidden_decision),'actual_label_used_for_decision':'false','actual_label_used_for_posthoc_development_audit':'true','validation_output_count':0,'test_row_level_output_count':0,'Extended_Evaluation_processed':'false','Diagnostic_Subset_processed':'false','Pure_Test_processed':'false'}]
    write_csv(PATHS['det_audit'], audit_rows, list(audit_rows[0].keys()))
    summary=f"""# State Machine Interval Detector v2.0 Development Summary\n\n- detector_id: `{DETECTOR_ID}`\n- split scope: Development Set only (`original_split_v2_4=train`)\n- fusion_input_rows: {len(rows)}\n- raw_candidate_count: {len(raw_rows)}\n- prediction_count_before_budget_guard: {len(pre)}\n- prediction_count_after_budget_guard: {len(final)}\n- review_only_count: {len(review)}\n- overprediction_pruned_review_count: {len(pruned)}\n- open_count: {len(open_rows)}\n- unresolved_count: {len(unresolved)}\n- hard/support/weak anchor counts: {hard_count}/{support_count}/{weak_count}\n- budget_guard_triggered_video_count: {len(videos_over_soft)}\n- mean_raw_prediction_ratio: {mean_raw:.6f}\n- mean_final_prediction_ratio: {mean_final:.6f}\n\nActual label was not used for decision. It was used only after detector execution for Development Set post-hoc audit.\n"""
    PATHS['summary'].parent.mkdir(parents=True,exist_ok=True); PATHS['summary'].write_text(summary,encoding='utf-8')
    note="""# v2.0 Adjustment Note\n\nThis detector implements the v2.0 documented rule contract. Main adjustments from v1.4 are final scene anchor semantics, opening disclosure guard, OCR 20s timeline freshness, same-video relative audio support, interval scoring, and video-level overprediction guard. Validation/test rows were not processed.\n"""
    PATHS['note'].write_text(note,encoding='utf-8')
    # latest bundle 구성
    if PATHS['bundle'].exists(): shutil.rmtree(PATHS['bundle'])
    PATHS['bundle'].mkdir(parents=True,exist_ok=True)
    bundle_files=[PATHS['rule_doc'],PATHS['rule_contract'],PATHS['rule_plain'],PATHS['rule_change'],PATHS['config'],PATHS['fusion_script'],PATHS['detector_script'],PATHS['summary'],PATHS['report'],PATHS['note'],PATHS['schema_audit'],PATHS['join_quality'],PATHS['column_mapping'],PATHS['raw'],PATHS['pred'],PATHS['review'],PATHS['pruned'],PATHS['trace'],PATHS['evidence'],PATHS['budget'],PATHS['det_audit'],PATHS['video_summary'],PATHS['comparison'],PATHS['ratio'],PATHS['issues']]
    for src in bundle_files:
        if src.exists() and src.is_file() and src.stat().st_size<=MAX_BUNDLE_BYTES and src.suffix.lower() not in FORBIDDEN_SUFFIXES:
            shutil.copy2(src, PATHS['bundle']/src.name)
    bundle_readme=f"""# Latest Files: State Machine Detector v2.0 Development\n\n## 1. v2.0 rule purpose\nIntegrate final scene anchors, final scene-anchor OCR, 20s OCR timeline, and same-video relative audio into a Development Set only state-machine detector.\n\n## 2. Detector summary\n- raw candidates: {len(raw_rows)}\n- predictions before budget guard: {len(pre)}\n- predictions after budget guard: {len(final)}\n- review-only candidates: {len(review)}\n- overprediction-pruned review candidates: {len(pruned)}\n\n## 3. v1.4 대비 핵심 변경점\nScene source count is transition reliability only; OCR timeline freshness and opening disclosure guard are explicit; video-level overprediction guard demotes excess candidates.\n\n## 4. Output files\nSee copied CSV/JSON/MD files in this directory. Raw frame/audio/media/cache/model files are excluded.\n\n## 5. Viewer first files\n1. `{PATHS['pred'].name}`\n2. `{PATHS['review'].name}`\n3. `{PATHS['trace'].name}`\n4. `{PATHS['det_audit'].name}`\n\n## 6. Video-level overprediction guard\nTriggered videos: {videos_over_soft}\n\n## 7. Leakage/safety\nactual_label_used_for_decision=false; actual_label_used_for_posthoc_development_audit=true; validation/test rows processed=0.\n\n## 8. Next step\nAdd v2.0 outputs to the version-aware review viewer registry.\n"""
    (PATHS['bundle']/'README_latest_files.md').write_text(bundle_readme,encoding='utf-8')
    # 제외 대상 scan bundle 구성
    forbidden=[]
    for root,dirs,files in os.walk(PATHS['bundle']):
        rp=Path(root)
        for d in dirs:
            if d.lower() in FORBIDDEN_DIRS: forbidden.append(str(rp/d))
        for name in files:
            p=rp/name
            if p.suffix.lower() in FORBIDDEN_SUFFIXES or p.stat().st_size>MAX_BUNDLE_BYTES: forbidden.append(str(p))
    report={'run_id':datetime.now().strftime('%Y%m%d_%H%M%S'),'timestamp':now_iso(),'task_name':DETECTOR_ID,'project_root':str(ROOT),'rule_doc_path':str(PATHS['rule_doc']),'rule_doc_sha256':sha256(PATHS['rule_doc']),'rule_contract_path':str(PATHS['rule_contract']),'rule_contract_sha256':sha256(PATHS['rule_contract']),'rule_doc_created_before_detector':True,'detector_executed':True,'feature_extraction_executed':False,'OCR_extraction_executed':False,'audio_extraction_executed':False,'scene_extraction_executed':False,'threshold_tuning_executed':False,'evaluation_executed':False,'actual_label_used_for_decision':False,'actual_label_used_for_posthoc_development_audit':True,'Extended_Evaluation_processed':False,'Diagnostic_Subset_processed':False,'Pure_Test_processed':False,'validation_output_count':0,'test_row_level_output_count':0,'fusion_input_path':str(PATHS['fusion_input']),'fusion_input_rows':len(rows),'development_video_ids':sorted(DEV_IDS),'raw_candidate_count':len(raw_rows),'prediction_count_before_budget_guard':len(pre),'prediction_count_after_budget_guard':len(final),'review_only_count':len(review),'overprediction_pruned_review_count':len(pruned),'open_count':len(open_rows),'unresolved_count':len(unresolved),'rejected_count':0,'hard_support_weak_counts':f'{hard_count}/{support_count}/{weak_count}','budget_guard_triggered_video_count':len(videos_over_soft),'videos_over_soft_ratio':videos_over_soft,'videos_over_hard_ratio':videos_over_hard,'mean_raw_prediction_ratio':mean_raw,'mean_final_prediction_ratio':mean_final,'v1_4_vs_v2_0_summary':f'v1.4 existing predictions compared for {len(comp)} Development videos','forbidden_decision_columns_found':forbidden_decision,'label_derived_audio_columns_used_for_decision':False,'audio_candidate_score_used_for_decision':False,'latest_bundle':str(PATHS['bundle']),'latest_bundle_forbidden_scan_passed':not forbidden,'latest_bundle_forbidden_files_found':forbidden,'ready_for_viewer_version_registry':not forbidden,'ready_for_rule_review':not forbidden and not forbidden_decision,'warnings':[],'errors':[]}
    write_json(PATHS['report'], report)
    # report 생성 뒤 bundle에 다시 복사한다
    shutil.copy2(PATHS['report'], PATHS['bundle']/PATHS['report'].name)
    log('detector complete')
    print(json.dumps(report, ensure_ascii=False))
if __name__=='__main__': main()
