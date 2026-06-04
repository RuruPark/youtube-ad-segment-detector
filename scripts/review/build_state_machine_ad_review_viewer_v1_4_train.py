#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT=Path('.')
OLD_PROJECT_ROOT=Path('./_old_project_not_included')
TRAIN_IDS={1,2,5,6,8,9,10,11,12,13,14,15}; VALIDATION_IDS={3,7,18}; TEST_IDS={4,16,17}; NON_TRAIN_IDS=VALIDATION_IDS|TEST_IDS
SPLIT_SEED='20240524'; VERSION_KEY='v1_4_train'; VERSION_DOT='v1.4'; BASE_VERSION_DOT='v1.3'; SCOPE='train_only'; WORST_ORDER=[10,13,5,9,1]
OUT_ROOT=PROJECT_ROOT/'outputs/review'; V12_DIR=OUT_ROOT/'state_machine_ad_review_viewer_v1_2'; V13_DIR=OUT_ROOT/'state_machine_ad_review_viewer_v1_3_train'; V14_DIR=OUT_ROOT/'state_machine_ad_review_viewer_v1_4_train'; CURRENT_DIR=OUT_ROOT/'state_machine_ad_review_viewer_current'; REGISTRY_PATH=OUT_ROOT/'state_machine_ad_review_viewer_versions.json'
LATEST_DIR=PROJECT_ROOT/'outputs/latest_for_chatgpt_state_machine_review_viewer_v1_4_train_update'; REPORT_DIR=PROJECT_ROOT/'reports/review'; LOG_PATH=PROJECT_ROOT/'logs/state_machine_ad_review_viewer_v1_4_train_update_run_log.txt'; REPORT_PATH=REPORT_DIR/'state_machine_ad_review_viewer_v1_4_train_update_report.json'; SUMMARY_PATH=REPORT_DIR/'state_machine_ad_review_viewer_v1_4_train_update_summary.md'
V14_MANIFEST='review_manifest_v1_4_train.json'; CURRENT_MANIFEST='review_manifest_current_train_val.json'
BUILD_SCRIPT=PROJECT_ROOT/'scripts/review/build_state_machine_ad_review_viewer_v1_4_train.py'; SERVE_SCRIPT=PROJECT_ROOT/'scripts/review/serve_state_machine_ad_review_viewer_current.py'; SWITCH_SCRIPT=PROJECT_ROOT/'scripts/review/switch_state_machine_review_viewer_version.py'
FORBIDDEN_SUFFIXES={'.mp4','.mov','.mkv','.avi','.webm','.wav','.mp3','.m4a','.jpg','.jpeg','.png','.webp','.parquet','.pkl','.pickle','.pt','.pth','.ckpt','.onnx'}
FORBIDDEN_PARTS={'cache','frames','frame_images','raw_video','video_proxy','model_cache','tmp','__pycache__'}
INPUTS={
 'config':'configs/detectors/state_machine_interval_detector_v1_4_train_config.json','high':'data/predictions/state_machine_interval_predictions_v1_4_train.csv','review':'data/predictions/state_machine_review_only_interval_candidates_v1_4_train.csv','trace':'data/predictions/state_machine_anchor_trace_v1_4_train.csv','open':'data/predictions/state_machine_open_interval_candidates_v1_4_train.csv','unresolved':'data/predictions/state_machine_unresolved_long_open_candidates_v1_4_train.csv','rejected':'data/predictions/state_machine_rejected_long_open_candidates_v1_4_train.csv','disclosure':'data/predictions/state_machine_disclosure_notice_review_candidates_v1_4_train.csv','evidence':'data/predictions/state_machine_evidence_tier_events_v1_4_train.csv','fresh':'data/predictions/state_machine_fresh_evidence_timeout_events_v1_4_train.csv','filter':'data/predictions/state_machine_interval_confidence_filter_events_v1_4_train.csv','audit':'data/predictions/state_machine_detector_train_audit_v1_4.csv','comparison':'data/predictions/state_machine_v1_3_vs_v1_4_train_comparison.csv','video_summary':'data/analysis/train_only_detector_error_video_summary_v1_4.csv','interval_overlap':'data/analysis/train_only_detector_error_interval_overlap_v1_4.csv','open_summary':'data/analysis/train_only_detector_error_open_interval_summary_v1_4.csv','trace_reason':'data/analysis/train_only_detector_error_trace_reason_summary_v1_4.csv','worst':'data/analysis/train_only_detector_error_worst_cases_v1_4.csv','rule_issues':'data/analysis/train_only_detector_error_rule_issue_candidates_v1_4.csv','detector_report':'reports/detectors/state_machine_interval_detector_v1_4_train_report.json','detector_summary':'reports/detectors/state_machine_interval_detector_v1_4_train_summary.md','adjustment_note':'reports/detectors/state_machine_interval_detector_v1_4_adjustment_note.md','actual':'data/segments/ad_interval_segments_v2_4.csv','split':'data/splits/video_split_v2_4.csv'}

def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def stamp(): return datetime.now().strftime('%Y%m%d_%H%M%S')
def rel(p:Path):
    try: return str(p.relative_to(PROJECT_ROOT))
    except ValueError: return str(p)
def log(msg:str):
    print(msg); LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
    with LOG_PATH.open('a',encoding='utf-8') as f: f.write(msg+'\n')
def sha(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def stats(p:Path):
    if not p.exists(): return {'exists':False,'path':str(p)}
    st=p.stat(); return {'exists':True,'path':str(p),'size':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':sha(p)}
def snapshot(src:Path,out:Path):
    out.parent.mkdir(parents=True,exist_ok=True); rows=['relative_path\tsize\tmtime_ns']
    if src.exists():
        for dp,dn,fn in os.walk(src):
            dn.sort(); fn.sort()
            for name in fn:
                p=Path(dp)/name
                try: st=p.stat(); rows.append(f'{p.relative_to(src)}\t{st.st_size}\t{st.st_mtime_ns}')
                except FileNotFoundError: pass
    out.write_text('\n'.join(rows)+'\n',encoding='utf-8'); return {'path':str(out),'sha256':sha(out),'exists':src.exists()}
def read_csv(relpath:str):
    p=PROJECT_ROOT/relpath
    with p.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def read_json(path:Path): return json.loads(path.read_text(encoding='utf-8'))
def write_json(path:Path,data): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def to_i(v,default=None):
    try:
        if v is None or str(v).strip()=='': return default
        return int(float(str(v)))
    except Exception: return default
def to_f(v,default=None):
    try:
        if v is None or str(v).strip()=='': return default
        return float(str(v))
    except Exception: return default
def truth(v): return str(v).lower() in {'true','1','yes','y'}
def mmss(v):
    s=max(0,int(round(to_f(v,0) or 0))); return f'{s//60:02d}:{s%60:02d}'
def scan_forbidden(root:Path):
    bad=[]
    if not root.exists(): return bad
    for p in root.rglob('*'):
        parts={x.lower() for x in p.relative_to(root).parts}
        if parts & FORBIDDEN_PARTS: bad.append(str(p.relative_to(root)))
        elif p.is_file() and p.suffix.lower() in FORBIDDEN_SUFFIXES: bad.append(str(p.relative_to(root)))
    return sorted(set(bad))
def backup(ts):
    b=PROJECT_ROOT/'backups'/f'state_machine_ad_review_viewer_v1_4_train_update_{ts}'; copied=[]
    targets=[CURRENT_DIR,REGISTRY_PATH,V14_DIR,LATEST_DIR,REPORT_PATH,SUMMARY_PATH,LOG_PATH,BUILD_SCRIPT,SERVE_SCRIPT,SWITCH_SCRIPT]
    for src in targets:
        if src.exists():
            dst=b/rel(src); dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(src,dst) if src.is_dir() else shutil.copy2(src,dst)
            copied.append({'source':str(src),'backup':str(dst)})
    if not any(x['source']==str(CURRENT_DIR) for x in copied): raise RuntimeError('current viewer backup failed; current folder missing')
    b.mkdir(parents=True,exist_ok=True); write_json(b/'backup_manifest.json',copied)
    return {'backup_dir':str(b),'copied':copied}
def ensure_train(name, rows):
    bad=[]
    for r in rows:
        vid=to_i(r.get('video_id')); split=str(r.get('split') or '').lower()
        if vid in NON_TRAIN_IDS or (vid is not None and vid not in TRAIN_IDS) or (split and split!='train'): bad.append({'video_id':vid,'split':split})
    if bad: raise RuntimeError(f'{name} contains non-train rows: {bad[:5]}')
def load_split(rows):
    obs=defaultdict(list); seeds=set(); meta={}
    for r in rows:
        vid=to_i(r.get('video_id')); sp=str(r.get('split') or '').lower(); seeds.add(str(r.get('split_seed') or ''))
        if vid is None: continue
        obs[sp].append(vid)
        if sp=='train': meta[vid]={'video_id':vid,'split':'train','video_name':r.get('video_name') or '', 'video_path':r.get('video_path') or '', 'video_duration_sec':to_f(r.get('video_duration_sec'),0) or 0, 'playable': bool(r.get('video_path') and Path(r.get('video_path') or '').exists())}
    expected={'train':sorted(TRAIN_IDS),'validation':sorted(VALIDATION_IDS),'test':sorted(TEST_IDS)}; got={k:sorted(v) for k,v in obs.items()}
    if got!=expected or seeds!={SPLIT_SEED}: raise RuntimeError(f'split mismatch: {got}, seeds={sorted(seeds)}')
    return meta
def actual_map(rows,durs,warnings):
    out=defaultdict(list)
    for idx,r in enumerate(rows,1):
        vid=to_i(r.get('video_id'))
        if vid not in TRAIN_IDS: continue
        typ=str(r.get('segment_type') or r.get('interval_type') or r.get('label') or r.get('is_ad') or '').lower()
        if any(x in typ for x in ['non_ad','random_non_ad','post_ad','pre_ad']) or typ in {'false','0','no'}: continue
        if typ and not (typ in {'ad','ad_interval','ad_full','true','1','yes'} or typ.startswith('ad_')): continue
        s=to_f(r.get('ad_start_sec') or r.get('start_sec') or r.get('segment_start_sec') or r.get('interval_start_sec'))
        e=to_f(r.get('ad_end_sec') or r.get('end_sec') or r.get('segment_end_sec') or r.get('interval_end_sec'))
        item=interval_item(s,e,durs.get(vid),'actual',vid,warnings)
        if item: item.update({'id':r.get('ad_interval_id') or r.get('segment_id') or f'actual_{vid}_{idx:04d}','label':'actual_ad'}); out[vid].append(item)
    return out
def interval_item(s,e,dur,src,vid,warnings):
    if s is None or e is None: warnings.append(f'missing boundary {src} video={vid}'); return None
    os_,oe=s,e; s=max(0,s); e=min(e,dur) if dur and dur>0 else e
    if s>=e: warnings.append(f'invalid interval dropped {src} video={vid} {os_}-{oe}'); return None
    return {'start':round(s,3),'end':round(e,3),'duration':round(e-s,3),'mmss':f'{mmss(s)}-{mmss(e)}'}
def interval_map(rows,kind,id_col,start_cols,end_cols,durs,warnings):
    out=defaultdict(list)
    for idx,r in enumerate(rows,1):
        vid=to_i(r.get('video_id'))
        if vid not in TRAIN_IDS: continue
        s=next((to_f(r.get(c)) for c in start_cols if to_f(r.get(c)) is not None),None); e=next((to_f(r.get(c)) for c in end_cols if to_f(r.get(c)) is not None),None)
        item=interval_item(s,e,durs.get(vid),kind,vid,warnings)
        if item:
            item.update({'id':r.get(id_col) or f'{kind}_{vid}_{idx:04d}','source_row':r})
            for k in ['start_reason','end_reason','start_confidence_level','end_confidence_level','hard_evidence_count','support_evidence_count','weak_context_count','hard_evidence_density_per_60s','max_weak_span_sec','review_only_reasons','interval_status','open_reason','unresolved_reason','rejected_reason']:
                if k in r: item[k]=r.get(k)
            out[vid].append(item)
    return out
def point_map(rows,id_prefix,time_col):
    out=defaultdict(list)
    for idx,r in enumerate(rows,1):
        vid=to_i(r.get('video_id'))
        if vid not in TRAIN_IDS: continue
        t=to_f(r.get(time_col),0) or 0
        out[vid].append({'id':r.get('visual_anchor_id') or r.get('raw_interval_id') or f'{id_prefix}_{vid}_{idx:04d}','time':round(t,3),'mmss':mmss(t),'reason':r.get('rejection_reason') or r.get('event_reason') or r.get('filter_result') or r.get('evidence_tier_summary') or '', 'source_row':r})
    return out
def evidence_summary(rows):
    g=defaultdict(list); out={}
    for r in rows:
        vid=to_i(r.get('video_id'))
        if vid in TRAIN_IDS: g[vid].append(r)
    for vid,items in g.items():
        hard=sum(1 for r in items if truth(r.get('hard_evidence_flag'))); support=sum(1 for r in items if truth(r.get('support_evidence_flag'))); weak=sum(1 for r in items if truth(r.get('weak_context_flag')))
        out[vid]={'hard_evidence_count':hard,'support_evidence_count':support,'weak_context_count':weak,'top_hard_reasons':Counter(x for r in items for x in str(r.get('hard_evidence_reasons') or '').split('+') if x).most_common(5),'top_support_reasons':Counter(x for r in items for x in str(r.get('support_evidence_reasons') or '').split('+') if x).most_common(5),'top_weak_context_reasons':Counter(x for r in items for x in str(r.get('weak_context_reasons') or '').split('+') if x).most_common(5)}
    return out
def by_video(rows): return {to_i(r.get('video_id')):r for r in rows if to_i(r.get('video_id')) in TRAIN_IDS}
def sort_ids(ids,summaries):
    out=[v for v in WORST_ORDER if v in ids]
    rest=[v for v in sorted(ids, key=lambda vid:(0 if summaries.get(vid,{}).get('review_priority')=='high' else 1, -(to_f(summaries.get(vid,{}).get('total_candidate_coverage_ratio'),0) or 0), vid)) if v not in out]
    return out+rest
def count_map(m): return sum(len(v) for v in m.values())
def build_manifest(data,warnings):
    meta=load_split(data['split']); durs={vid:x['video_duration_sec'] for vid,x in meta.items()}
    actual=actual_map(data['actual'],durs,warnings)
    high=interval_map(data['high'],'high','prediction_id',['ad_start_sec'],['ad_end_sec'],durs,warnings)
    review=interval_map(data['review'],'review','candidate_id',['start_sec'],['end_sec'],durs,warnings)
    openm=interval_map(data['open'],'open','open_candidate_id',['ad_start_sec'],['last_anchor_sec'],durs,warnings)
    unresolved=interval_map(data['unresolved'],'unresolved','unresolved_candidate_id',['start_sec'],['last_anchor_sec'],durs,warnings)
    rejected=interval_map(data['rejected'],'rejected','rejected_candidate_id',['start_sec'],['last_anchor_sec'],durs,warnings)
    disclosure=point_map(data['disclosure'],'disc','transition_time_anchor')
    confidence=point_map(data['filter'],'conf','start_sec')
    evsum=evidence_summary(data['evidence']); summaries=by_video(data['video_summary']); comps=by_video(data['comparison']); audits=by_video(data['audit'])
    vids=[]
    for vid in sort_ids(sorted(TRAIN_IDS),summaries):
        info=meta[vid]; s=summaries.get(vid,{})
        vids.append({'video_id':vid,'split':'train','video_name':info['video_name'],'video_path':info['video_path'],'video_url':f'/media/{vid}','video_duration_sec':info['video_duration_sec'],'playable':info['playable'],'actual_intervals':actual.get(vid,[]),'high_confidence_predictions':high.get(vid,[]),'review_only_interval_candidates':review.get(vid,[]),'open_interval_candidates':openm.get(vid,[]),'unresolved_long_open_candidates':unresolved.get(vid,[]),'rejected_long_open_candidates':rejected.get(vid,[]),'disclosure_notice_review_candidates':disclosure.get(vid,[]),'interval_confidence_filter_events':confidence.get(vid,[]),'evidence_tier_summary':evsum.get(vid,{'hard_evidence_count':0,'support_evidence_count':0,'weak_context_count':0,'top_hard_reasons':[],'top_support_reasons':[],'top_weak_context_reasons':[]}),'train_error_summary':s,'comparison_summary':comps.get(vid,{}),'audit_summary':audits.get(vid,{}),'counts':{'actual_intervals':len(actual.get(vid,[])),'high_confidence_predictions':len(high.get(vid,[])),'review_only_interval_candidates':len(review.get(vid,[])),'open_interval_candidates':len(openm.get(vid,[])),'unresolved_long_open_candidates':len(unresolved.get(vid,[])),'rejected_long_open_candidates':len(rejected.get(vid,[])),'disclosure_notice_review_candidates':len(disclosure.get(vid,[])),'interval_confidence_filter_events':len(confidence.get(vid,[]))},'review_priority':s.get('review_priority','low'),'viewer_review_hint':('Worst-priority v1.4 train review video ' if vid in WORST_ORDER else 'Review video ')+str(vid)})
    det=data['detector_report']; m=det.get('v1_4_metrics',{})
    manifest={'viewer_version':VERSION_KEY,'detector_version':VERSION_DOT,'base_version':BASE_VERSION_DOT,'scope':SCOPE,'generated_at':now(),'project_root':str(PROJECT_ROOT),'manifest_name':V14_MANIFEST,'split_policy':{'split_seed':SPLIT_SEED,'included_splits':['train'],'validation_included':False,'test_included':False,'train_video_ids':sorted(TRAIN_IDS),'excluded_validation_video_ids':sorted(VALIDATION_IDS),'excluded_test_video_ids':sorted(TEST_IDS)},'meta':{'detector_version':VERSION_DOT,'base_version':BASE_VERSION_DOT,'scope':SCOPE,'high_confidence_prediction_count':count_map(high),'review_only_candidate_count':count_map(review),'trace_row_count':len(data['trace']),'open_interval_count':count_map(openm),'unresolved_long_open_count':count_map(unresolved),'rejected_long_open_count':count_map(rejected),'disclosure_notice_rejected_count':count_map(disclosure),'interval_confidence_filter_event_count':count_map(confidence),'hard_evidence_trace_count':int(m.get('hard_evidence_trace_count',0)),'support_evidence_trace_count':int(m.get('support_evidence_trace_count',0)),'weak_context_trace_count':int(m.get('weak_context_trace_count',0)),'validation_included':False,'test_included':False,'worst_videos':WORST_ORDER,'note':'v1.4 train-only adjustment candidate; high-confidence predictions and review-only candidates are separated.'},'color_policy':{'actual':'#ef4444','high_confidence_prediction':'#2563eb','overlap':'#a855f7','review_only':'#7c3aed','open':'#60a5fa','unresolved':'#93c5fd','rejected':'#6b7280','event':'#0f766e'},'videos':vids}
    ids={v['video_id'] for v in vids}
    if ids!=TRAIN_IDS or ids & NON_TRAIN_IDS: raise RuntimeError(f'manifest train ids invalid: {sorted(ids)}')
    return manifest
INDEX_HTML='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Detector v1.4 Train Review Viewer</title><link rel="stylesheet" href="style.css"></head><body><header class="topbar"><div><h1>Detector version: v1.4 train-only</h1><p id="viewerMeta">High-confidence prediction and review-only candidate buckets are separated.</p></div><div class="badges"><span>train-only</span><span>not final detector</span><span>validation/test excluded</span></div></header><main class="layout"><aside class="sidebar"><section class="panel"><h2>Videos</h2><select id="videoSelect"></select><div id="videoList" class="video-list"></div></section><section class="panel"><h2>Summary</h2><div id="summaryCards" class="cards"></div></section><section class="panel"><h2>Buckets</h2><div id="toggles" class="toggles"></div></section></aside><section class="workspace"><div class="player-wrap"><video id="video" controls preload="metadata"></video><div id="playableWarning" class="warning"></div></div><div class="timeline-wrap"><div id="timeline" class="timeline"></div><div id="timeMarker" class="time-marker"></div></div><div class="controls"><button id="skipBtn">Skip high-confidence prediction</button><span id="currentTime">00:00</span></div><section class="panel"><h2>Intervals and Events</h2><div id="lists" class="lists"></div></section><section class="panel"><h2>Evidence Tier Summary</h2><pre id="evidenceSummary"></pre></section></section></main><script src="app.js"></script></body></html>'''
STYLE_CSS='''*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f7f7f4;color:#1f2937}.topbar{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;padding:18px 24px;background:#111827;color:white}.topbar h1{margin:0 0 6px;font-size:22px}.topbar p{margin:0;color:#cbd5e1}.badges{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.badges span{border:1px solid #475569;border-radius:6px;padding:5px 8px;background:#1f2937;font-size:12px}.layout{display:grid;grid-template-columns:340px 1fr;gap:16px;padding:16px}.sidebar,.workspace{display:flex;flex-direction:column;gap:12px}.panel{background:white;border:1px solid #ddd8cf;border-radius:8px;padding:12px;box-shadow:0 1px 2px rgba(0,0,0,.04)}.panel h2{font-size:15px;margin:0 0 10px}.player-wrap{background:#111827;border-radius:8px;min-height:260px;display:flex;align-items:center;justify-content:center;position:relative}video{width:100%;max-height:62vh;background:#111827}.warning{position:absolute;left:12px;bottom:12px;background:#fef3c7;color:#92400e;padding:6px 8px;border-radius:6px;font-size:13px}.timeline-wrap{position:relative;background:white;border:1px solid #ddd8cf;border-radius:8px;padding:12px}.timeline{height:160px;position:relative}.lane{position:relative;height:22px;margin:5px 0;border-bottom:1px solid #eee}.lane-label{position:absolute;left:0;width:170px;font-size:12px;color:#475569;line-height:20px}.seg{position:absolute;height:16px;top:2px;border-radius:3px;cursor:pointer;min-width:2px}.actual{background:#ef4444}.high{background:#2563eb}.overlap{background:#a855f7;z-index:5}.review{background:rgba(124,58,237,.45);border:2px dashed #7c3aed}.open{background:rgba(96,165,250,.45);border:2px dashed #2563eb}.unresolved{background:rgba(147,197,253,.55);border:2px dashed #60a5fa}.rejected{background:rgba(107,114,128,.45);border:2px dashed #4b5563}.event{background:#0f766e;width:4px}.time-marker{position:absolute;top:12px;bottom:12px;width:2px;background:#111827;pointer-events:none}.controls{display:flex;gap:12px;align-items:center}.controls button{padding:8px 10px;border:1px solid #2563eb;background:#2563eb;color:white;border-radius:6px;cursor:pointer}.cards{display:grid;grid-template-columns:1fr 1fr;gap:8px}.card{border:1px solid #e5e7eb;border-radius:6px;padding:8px;background:#fafafa}.card b{display:block;font-size:12px;color:#6b7280}.video-list{display:flex;flex-direction:column;gap:6px;max-height:240px;overflow:auto}.video-item{border:1px solid #e5e7eb;background:#fff;border-radius:6px;padding:8px;text-align:left;cursor:pointer}.video-item.active{border-color:#2563eb;background:#eff6ff}.toggles{display:grid;grid-template-columns:1fr;gap:6px}.toggles label{font-size:13px}.lists{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.bucket h3{font-size:14px;margin:0 0 8px}.item{border:1px solid #e5e7eb;border-radius:6px;padding:7px;margin-bottom:6px;cursor:pointer;background:#fff}.item small{display:block;color:#6b7280;margin-top:3px}pre{white-space:pre-wrap;background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:10px;max-height:220px;overflow:auto}@media(max-width:900px){.layout{grid-template-columns:1fr}.topbar{flex-direction:column}.cards{grid-template-columns:1fr}}'''
APP_JS='''const MANIFEST_URL='review_manifest_current_train_val.json';const state={manifest:null,video:null,enabled:{actual:true,high:true,review:true,open:true,unresolved:true,rejected:true,events:true}};const buckets=[['actual','Actual Ads','actual_intervals'],['high','High-confidence predictions','high_confidence_predictions'],['review','Review-only closed candidates','review_only_interval_candidates'],['open','Open interval candidates','open_interval_candidates'],['unresolved','Unresolved long open candidates','unresolved_long_open_candidates'],['rejected','Rejected long open candidates','rejected_long_open_candidates'],['events','Disclosure / confidence events','disclosure_notice_review_candidates']];function fmt(sec){sec=Math.max(0,Math.round(Number(sec)||0));return String(Math.floor(sec/60)).padStart(2,'0')+':'+String(sec%60).padStart(2,'0')}function pct(v){return((Number(v)||0)*100).toFixed(1)+'%'}async function init(){state.manifest=await fetch(MANIFEST_URL,{cache:'no-store'}).then(r=>r.json());document.getElementById('viewerMeta').textContent=`Detector ${state.manifest.detector_version} ${state.manifest.scope} | videos ${state.manifest.videos.length} | validation/test excluded`;buildSelectors();buildToggles();selectVideo(state.manifest.videos[0].video_id)}function buildSelectors(){const sel=document.getElementById('videoSelect'),list=document.getElementById('videoList');sel.innerHTML='';list.innerHTML='';state.manifest.videos.forEach(v=>{const o=document.createElement('option');o.value=v.video_id;o.textContent=`video ${v.video_id} ${v.review_priority||''}`;sel.appendChild(o);const b=document.createElement('button');b.className='video-item';b.dataset.vid=v.video_id;b.innerHTML=`<b>video ${v.video_id}</b><br><small>${v.viewer_review_hint||''}</small>`;b.onclick=()=>selectVideo(v.video_id);list.appendChild(b)});sel.onchange=()=>selectVideo(sel.value)}function buildToggles(){const wrap=document.getElementById('toggles');wrap.innerHTML='';buckets.forEach(([key,label])=>{const l=document.createElement('label');l.innerHTML=`<input type="checkbox" checked data-key="${key}"> ${label}`;l.querySelector('input').onchange=e=>{state.enabled[key]=e.target.checked;renderAll()};wrap.appendChild(l)})}function selectVideo(id){state.video=state.manifest.videos.find(v=>String(v.video_id)===String(id));document.getElementById('videoSelect').value=state.video.video_id;document.querySelectorAll('.video-item').forEach(x=>x.classList.toggle('active',String(x.dataset.vid)===String(id)));const video=document.getElementById('video');video.src=state.video.playable?state.video.video_url:'';document.getElementById('playableWarning').textContent=state.video.playable?'':'Video path missing or not playable on server';video.ontimeupdate=updateMarker;renderAll()}function renderAll(){renderSummary();renderTimeline();renderLists();renderEvidence();updateMarker()}function renderSummary(){const s=state.video.train_error_summary||{},c=state.video.counts||{};const rows=[['high-confidence',c.high_confidence_predictions],['review-only',c.review_only_interval_candidates],['open',c.open_interval_candidates],['unresolved',c.unresolved_long_open_candidates],['rejected',c.rejected_long_open_candidates],['actual cov',pct(s.actual_coverage_ratio)],['high cov',pct(s.high_confidence_coverage_ratio)],['review/open/unres',`${pct(s.review_only_coverage_ratio)} / ${pct(s.open_coverage_ratio)} / ${pct(s.unresolved_coverage_ratio)}`],['severity',s.severity_level_high_confidence||''],['priority',state.video.review_priority||'']];document.getElementById('summaryCards').innerHTML=rows.map(([k,v])=>`<div class="card"><b>${k}</b>${v??''}</div>`).join('')}function addLane(container,label,items,cls){const lane=document.createElement('div');lane.className='lane';lane.innerHTML=`<div class="lane-label">${label}</div>`;const dur=Number(state.video.video_duration_sec)||1,w=Math.max(100,container.clientWidth-180);(items||[]).forEach(it=>{const s=Number(it.start??it.time),e=Number(it.end??it.time);const seg=document.createElement('div');seg.className=`seg ${cls}`;seg.style.left=(170+(s/dur)*w)+'px';seg.style.width=Math.max(3,((Math.max(e-s,1)/dur)*w))+'px';seg.title=`${label} ${fmt(s)}-${fmt(e)} ${it.id||''}`;seg.onclick=()=>seek(s);lane.appendChild(seg)});container.appendChild(lane)}function overlaps(a,b){const out=[];(a||[]).forEach(x=>(b||[]).forEach(y=>{const s=Math.max(Number(x.start),Number(y.start)),e=Math.min(Number(x.end),Number(y.end));if(e>s)out.push({start:s,end:e,id:'overlap'})}));return out}function renderTimeline(){const t=document.getElementById('timeline');t.innerHTML='';const v=state.video;if(!v)return;if(state.enabled.actual)addLane(t,'Actual Ads',v.actual_intervals,'actual');if(state.enabled.high)addLane(t,'High-confidence',v.high_confidence_predictions,'high');if(state.enabled.review)addLane(t,'Review-only',v.review_only_interval_candidates,'review');if(state.enabled.open)addLane(t,'Open',v.open_interval_candidates,'open');if(state.enabled.unresolved)addLane(t,'Unresolved',v.unresolved_long_open_candidates,'unresolved');if(state.enabled.rejected)addLane(t,'Rejected',v.rejected_long_open_candidates,'rejected');const ov=[...overlaps(v.actual_intervals,v.high_confidence_predictions),...overlaps(v.actual_intervals,v.review_only_interval_candidates),...overlaps(v.actual_intervals,v.open_interval_candidates),...overlaps(v.actual_intervals,v.unresolved_long_open_candidates)];addLane(t,'Actual/Candidate overlap',ov,'overlap');if(state.enabled.events)addLane(t,'Events',[...(v.disclosure_notice_review_candidates||[]),...(v.interval_confidence_filter_events||[])],'event')}function renderLists(){const lists=document.getElementById('lists');lists.innerHTML='';buckets.forEach(([key,label,field])=>{const items=state.video[field]||[];const div=document.createElement('div');div.className='bucket';div.innerHTML=`<h3>${label} (${items.length})</h3>`;items.forEach(it=>{const start=Number(it.start??it.time),end=Number(it.end??it.time);const item=document.createElement('div');item.className='item';item.innerHTML=`${fmt(start)}${end&&end!==start?' - '+fmt(end):''}<small>${it.id||''} ${it.start_reason||it.reason||it.review_only_reasons||''}</small>`;item.onclick=()=>seek(start);div.appendChild(item)});lists.appendChild(div)})}function renderEvidence(){document.getElementById('evidenceSummary').textContent=JSON.stringify(state.video.evidence_tier_summary,null,2)}function seek(sec){const video=document.getElementById('video');if(video.src){video.currentTime=Number(sec)||0;video.play().catch(()=>{})}}function updateMarker(){const video=document.getElementById('video'),marker=document.getElementById('timeMarker'),wrap=document.querySelector('.timeline-wrap');const dur=Number(state.video?.video_duration_sec)||video.duration||1,w=wrap.clientWidth-24;marker.style.left=(12+(video.currentTime/dur)*w)+'px';document.getElementById('currentTime').textContent=fmt(video.currentTime)}document.getElementById('skipBtn').onclick=()=>{const t=document.getElementById('video').currentTime;const pred=(state.video.high_confidence_predictions||[]).find(p=>t>=Number(p.start)&&t<Number(p.end));if(pred)seek(Number(pred.end)+0.2)};window.addEventListener('resize',renderTimeline);init();'''
def readme(versioned):
    title='v1.4 Train-only State Machine Ad Review Viewer' if versioned else 'Current State Machine Ad Review Viewer'
    return f'''# {title}\n\nThis viewer shows detector v1.4 train-only adjustment candidate results. It is not a final detector or a final performance report.\n\n## What To Check First\n- High-confidence predictions vs actual ads.\n- Whether actual ads moved into review-only candidates.\n- Open/unresolved/rejected candidates are review buckets, not predictions.\n- CTA+audio-rise hard-missing start note from v1.4 validation.\n- Worst videos first: 10, 13, 5, 9, 1.\n\n## Buckets And Colors\n- Actual Ads: red.\n- High-confidence predictions: blue.\n- Actual/candidate overlap: saturated purple.\n- Review-only closed candidates: purple dashed/translucent.\n- Open interval candidates: blue dashed/translucent.\n- Unresolved long open candidates: pale blue dashed.\n- Rejected long open candidates: gray dashed.\n\nSkip predicted ad applies only to high-confidence predictions.\n\n## Run Server\n```bash\ncd .\npython scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000\n```\nIn VS Code Remote-SSH, forward port 8000 in the Ports panel and open http://localhost:8000 locally.\n\n## Rollback\n```bash\npython scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train\npython scripts/review/switch_state_machine_review_viewer_version.py --version v1_2\n```\nRe-switch to v1.4:\n```bash\npython scripts/review/switch_state_machine_review_viewer_version.py --version v1_4_train\n```\nNo video/media/frame/cache/model/raw video files are copied. Validation/test videos are excluded.\n'''
def write_viewer(manifest):
    if V14_DIR.exists(): shutil.rmtree(V14_DIR)
    V14_DIR.mkdir(parents=True,exist_ok=True)
    (V14_DIR/'index.html').write_text(INDEX_HTML,encoding='utf-8'); (V14_DIR/'app.js').write_text(APP_JS.replace(CURRENT_MANIFEST,V14_MANIFEST),encoding='utf-8'); (V14_DIR/'style.css').write_text(STYLE_CSS,encoding='utf-8'); write_json(V14_DIR/V14_MANIFEST,manifest); (V14_DIR/'README_review_viewer.md').write_text(readme(True),encoding='utf-8')
    bad=scan_forbidden(V14_DIR)
    if bad: raise RuntimeError(f'forbidden v1.4 viewer files: {bad}')
def update_current(manifest):
    before=None
    if (CURRENT_DIR/'current_version.json').exists():
        try: before=read_json(CURRENT_DIR/'current_version.json').get('current_version')
        except Exception: pass
    if CURRENT_DIR.exists(): shutil.rmtree(CURRENT_DIR)
    CURRENT_DIR.mkdir(parents=True,exist_ok=True)
    (CURRENT_DIR/'index.html').write_text(INDEX_HTML,encoding='utf-8'); (CURRENT_DIR/'app.js').write_text(APP_JS,encoding='utf-8'); (CURRENT_DIR/'style.css').write_text(STYLE_CSS,encoding='utf-8'); write_json(CURRENT_DIR/CURRENT_MANIFEST,manifest); write_json(CURRENT_DIR/'current_version.json',{'current_version':VERSION_KEY,'detector_version':VERSION_DOT,'base_version':BASE_VERSION_DOT,'scope':SCOPE,'rollback_supported':True,'rollback_targets':['v1_3_train','v1_2'],'validation_included':False,'test_included':False,'updated_at':now()}); (CURRENT_DIR/'README_current_viewer.md').write_text(readme(False),encoding='utf-8')
    bad=scan_forbidden(CURRENT_DIR)
    if bad: raise RuntimeError(f'forbidden current viewer files: {bad}')
    return before
def update_registry():
    reg=read_json(REGISTRY_PATH) if REGISTRY_PATH.exists() else {'available_versions':{}}
    reg.setdefault('available_versions',{})
    if V12_DIR.exists(): reg['available_versions']['v1_2']={'viewer_dir':rel(V12_DIR),'manifest':'review_manifest_v1_2_train_val.json','detector_version':'v1.2','scope':'train_validation'}
    if V13_DIR.exists(): reg['available_versions']['v1_3_train']={'viewer_dir':rel(V13_DIR),'manifest':'review_manifest_v1_3_train.json','detector_version':'v1.3','base_version':'v1.2','scope':'train_only'}
    reg['available_versions']['v1_4_train']={'viewer_dir':rel(V14_DIR),'manifest':V14_MANIFEST,'detector_version':VERSION_DOT,'base_version':BASE_VERSION_DOT,'scope':SCOPE}
    reg['current_version']=VERSION_KEY; reg['current_viewer_dir']=rel(CURRENT_DIR); reg['rollback_supported']=True; write_json(REGISTRY_PATH,reg); return reg
def patch_scripts():
    text=SWITCH_SCRIPT.read_text(encoding='utf-8')
    text=text.replace("SUPPORTED_VERSIONS = {'v1_2', 'v1_3_train'}", "SUPPORTED_VERSIONS = {'v1_2', 'v1_3_train', 'v1_4_train'}")
    text=text.replace("'review_manifest_v1_3_train.json',\n    }", "'review_manifest_v1_3_train.json',\n        'review_manifest_v1_4_train.json',\n    }")
    text=text.replace("if version == 'v1_3_train':\n        validation_ids = ids & VALIDATION_VIDEO_IDS", "if version in {'v1_3_train', 'v1_4_train'}:\n        validation_ids = ids & VALIDATION_VIDEO_IDS")
    text=text.replace("f'v1_3_train current manifest must be train-only; validation_ids={sorted(validation_ids)}, splits={sorted(splits)}'", "f'{version} current manifest must be train-only; validation_ids={sorted(validation_ids)}, splits={sorted(splits)}'")
    text=text.replace("if manifest.get('detector_version') != 'v1.3' or manifest.get('scope') != 'train_only':\n            raise RuntimeError('v1_3_train manifest must declare detector_version=v1.3 and scope=train_only')", "expected_detector = 'v1.4' if version == 'v1_4_train' else 'v1.3'\n        if manifest.get('detector_version') != expected_detector or manifest.get('scope') != 'train_only':\n            raise RuntimeError(f'{version} manifest must declare detector_version={expected_detector} and scope=train_only')")
    text=text.replace("'rollback_target': 'v1_2' if version == 'v1_3_train' else 'v1_3_train',", "'rollback_target': 'v1_3_train' if version == 'v1_4_train' else ('v1_2' if version == 'v1_3_train' else 'v1_3_train'),")
    text=text.replace("if version == 'v1_3_train':\n        payload['validation_included'] = False\n        payload['test_included'] = False", "if version in {'v1_3_train', 'v1_4_train'}:\n        payload['validation_included'] = False\n        payload['test_included'] = False")
    SWITCH_SCRIPT.write_text(text,encoding='utf-8')
    st=SERVE_SCRIPT.read_text(encoding='utf-8')
    st=st.replace("server_version = 'StateMachineAdReviewViewerCurrent/1.3'", "server_version = 'StateMachineAdReviewViewerCurrent/1.4'")
    if 'v1_4_train' not in st:
        st=st.replace("print('Switch to v1.3 train-only: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train')", "print('Switch to v1.3 train-only: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train')\n    print('Switch to v1.4 train-only: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_4_train')")
    SERVE_SCRIPT.write_text(st,encoding='utf-8')
def latest():
    if LATEST_DIR.exists(): shutil.rmtree(LATEST_DIR)
    LATEST_DIR.mkdir(parents=True,exist_ok=True)
    files=[SUMMARY_PATH,REPORT_PATH,LOG_PATH,BUILD_SCRIPT,SERVE_SCRIPT,SWITCH_SCRIPT,REGISTRY_PATH,CURRENT_DIR/'current_version.json',V14_DIR/'README_review_viewer.md',CURRENT_DIR/'README_current_viewer.md',V14_DIR/'index.html',V14_DIR/'app.js',V14_DIR/'style.css',V14_DIR/V14_MANIFEST,CURRENT_DIR/CURRENT_MANIFEST]
    for src in files:
        if src.exists(): shutil.copy2(src,LATEST_DIR/src.name)
    readme='# Latest Files: v1.4 Train Review Viewer Update\n\nSmall files only. No media/video/frame/cache/model/raw video/proxy/checkpoint files copied. Validation/test row-level viewer outputs excluded.\n\n## Files\n'
    for p in sorted(LATEST_DIR.iterdir()):
        if p.is_file(): readme+=f'- {p.name}\n'
    readme+='- README_latest_files.md\n'; (LATEST_DIR/'README_latest_files.md').write_text(readme,encoding='utf-8')
    bad=scan_forbidden(LATEST_DIR)
    if bad: raise RuntimeError(f'latest forbidden files: {bad}')
    return bad
def summary_text(report,manifest):
    m=manifest['meta']
    return f"""# v1.4 Train Review Viewer Update Summary

Current viewer was switched to `v1_4_train`. This is a train-only review viewer for a v1.4 adjustment candidate, not a final detector report.

## Current Viewer
- before: `{report['current_version_before']}`
- after: `{report['current_version_after']}`
- v1.4 viewer: `{V14_DIR}`
- current viewer: `{CURRENT_DIR}`

## Review Focus
- High-confidence predictions vs actual ads.
- Whether actual ads moved into review-only candidates.
- Open/unresolved/rejected are review buckets, not predictions.
- CTA+audio-rise hard-missing start note from v1.4 validation.
- Worst videos first: 10, 13, 5, 9, 1.

## Counts
- train videos: `{len(manifest['videos'])}`
- high-confidence/review/open/unresolved/rejected: `{m['high_confidence_prediction_count']}` / `{m['review_only_candidate_count']}` / `{m['open_interval_count']}` / `{m['unresolved_long_open_count']}` / `{m['rejected_long_open_count']}`
- disclosure candidates: `{m['disclosure_notice_rejected_count']}`
- confidence filter events: `{m['interval_confidence_filter_event_count']}`
- hard/support/weak trace counts: `{m['hard_evidence_trace_count']}` / `{m['support_evidence_trace_count']}` / `{m['weak_context_trace_count']}`

## Commands
```bash
python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2
python scripts/review/switch_state_machine_review_viewer_version.py --version v1_4_train
```

Forward port 8000 in VS Code Remote-SSH and open http://localhost:8000 locally.

validation_included=false; test_included=false; no media copied.
"""

# v1.4 review 값은 기존 v1.3 viewer UI로 렌더링한다.
# 이후 rebuild에서 새 experimental bucket layout이 다시 생성되지 않도록 override를 끝부분에 둔다.
def _v13_ui_manifest(manifest):
    out=dict(manifest)
    by_vid={int(v['video_id']):v for v in manifest.get('videos',[])}
    videos=[]
    for vid in [1,2,5,6,8,9,10,11,12,13,14,15]:
        v=dict(by_vid[vid])
        high=[]
        for i,item in enumerate(v.get('high_confidence_predictions',[]),1):
            row=item.get('source_row') or {}
            high.append({'prediction_id': item.get('id') or row.get('prediction_id') or f'v1_4_prediction_{vid}_{i}', 'start': item.get('start') if item.get('start') is not None else row.get('ad_start_sec'), 'end': item.get('end') if item.get('end') is not None else row.get('ad_end_sec'), 'duration': item.get('duration') if item.get('duration') is not None else row.get('ad_duration_sec'), 'start_reason': item.get('start_reason') or row.get('start_reason') or '', 'end_reason': item.get('end_reason') or row.get('end_reason') or '', 'interval_status': item.get('interval_status') or row.get('interval_status') or 'high_confidence_closed', 'source_version': 'v1.4'})
        def as_open(items, kind):
            mapped=[]
            for i,item in enumerate(items,1):
                row=item.get('source_row') or {}
                start=item.get('start') if item.get('start') is not None else row.get('start_sec') or row.get('pending_start_time') or 0
                end=item.get('end') if item.get('end') is not None else item.get('display_end') or item.get('last_anchor_sec') or row.get('end_proxy_sec') or row.get('last_anchor_sec') or row.get('transition_time_anchor') or row.get('end_sec') or start
                mapped.append({'candidate_id': item.get('id') or row.get('candidate_id') or row.get('unresolved_candidate_id') or f'v1_4_{kind}_{vid}_{i}', 'start': start, 'display_end': end, 'reason': item.get('reason') or row.get('start_reason') or row.get('unresolved_reason') or kind, 'candidate_type': kind, 'source_version': 'v1.4'})
            return mapped
        def as_events(items, kind):
            mapped=[]
            for i,item in enumerate(items,1):
                row=item.get('source_row') or {}
                mapped.append({'event_id': item.get('id') or row.get('visual_anchor_id') or row.get('raw_interval_id') or f'v1_4_{kind}_{vid}_{i}', 'visual_anchor_id': row.get('visual_anchor_id') or item.get('id') or '', 'time': item.get('time') if item.get('time') is not None else row.get('transition_time_anchor') or row.get('start_sec') or 0, 'rejection_reason': item.get('reason') or row.get('rejection_reason') or row.get('filter_result') or kind, 'event_reason': row.get('event_reason') or item.get('reason') or '', 'review_note': row.get('review_note') or row.get('review_only_reasons') or '', 'source_version': 'v1.4'})
            return mapped
        open_items=as_open(v.get('open_interval_candidates',[]),'open')
        unresolved_items=as_open(v.get('unresolved_long_open_candidates',[]),'unresolved_long_open')
        disclosure_items=as_events(v.get('disclosure_notice_review_candidates',[]),'disclosure_notice_rejected')
        v.update({'split':'train', 'predicted_intervals': high, 'open_interval_candidates': open_items, 'unresolved_long_open_candidates': unresolved_items, 'disclosure_notice_review_candidates': disclosure_items, 'ocr_only_continuity_cap_events': [], 'weak_continuity_recovery_events': [], 'trace_focus_or_summary': {'trace_row_count': 0}, 'counts': {'actual': len(v.get('actual_intervals',[])), 'predicted': len(high), 'open': len(open_items), 'unresolved_long_open': len(unresolved_items), 'disclosure_notice_rejected': len(disclosure_items), 'ocr_only_continuity_cap_events': 0, 'weak_continuity_recovery_events': 0, 'v1_4_review_only_closed_candidates': len(v.get('review_only_interval_candidates',[])), 'v1_4_rejected_long_open_candidates': len(v.get('rejected_long_open_candidates',[])), 'v1_4_interval_confidence_filter_events': len(v.get('interval_confidence_filter_events',[]))}, 'v1_4_review_only_interval_candidates': v.get('review_only_interval_candidates',[]), 'v1_4_rejected_long_open_candidates': v.get('rejected_long_open_candidates',[]), 'v1_4_interval_confidence_filter_events': v.get('interval_confidence_filter_events',[])})
        videos.append(v)
    out.update({'version':'v1_4_train_using_v1_3_ui', 'task':'state_machine_ad_review_viewer_v1_4_train_current_uses_v1_3_ui', 'review_only': True, 'actual_label_usage':'audit_ui_only_not_detector_decision', 'no_detector_run': True, 'no_feature_extraction': True, 'no_threshold_tuning': True, 'validation_included': False, 'test_included': False, 'info_messages':['UI static files restored from v1.3 viewer; review values/manifest rows come from v1.4 train outputs.'], 'videos': videos})
    return out

def _v13_static(name, manifest_name=None, current_word='current viewer v1_4_train'):
    text=(V13_DIR/name).read_text(encoding='utf-8')
    if name == 'index.html':
        text=text.replace('상태 전이 기반 유튜브 광고 탐지 뷰어 v1.3 train-only','상태 전이 기반 유튜브 광고 탐지 뷰어 v1.4 train-only')
        text=text.replace('<p id="viewerMeta">human review / audit only</p>','<p id="viewerMeta">v1.3 viewer UI / v1.4 train review values / human review only</p>')
    if name == 'app.js' and manifest_name:
        text=text.replace('fetch("review_manifest_v1_3_train.json", { cache: "no-store" })',f'fetch("{manifest_name}", {{ cache: "no-store" }})')
        text=text.replace('Detector version: v1.3 train-only | current viewer v1_3_train | human review / audit only',f'Detector version: v1.4 train-only | {current_word} | v1.3 viewer UI | human review / audit only')
    return text

def _v13_ui_readme(title):
    return f'''# {title}\n\nThis viewer intentionally uses the v1.3 viewer UI/static layout.\n\n- Detector/review values: v1.4 train-only\n- UI baseline: v1.3 train viewer\n- Validation/test excluded\n- No detector run, no feature extraction, no threshold/rule tuning\n- Media files are not copied; video is served through `/media/<video_id>` from the manifest whitelist.\n\nRun current viewer:\n`python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000`\n\nRollback:\n`python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train`\n`python scripts/review/switch_state_machine_review_viewer_version.py --version v1_2`\n'''

def write_viewer(manifest):
    manifest=_v13_ui_manifest(manifest)
    if V14_DIR.exists(): shutil.rmtree(V14_DIR)
    V14_DIR.mkdir(parents=True,exist_ok=True)
    (V14_DIR/'index.html').write_text(_v13_static('index.html'),encoding='utf-8')
    (V14_DIR/'app.js').write_text(_v13_static('app.js',V14_MANIFEST,'versioned viewer v1_4_train'),encoding='utf-8')
    (V14_DIR/'style.css').write_text(_v13_static('style.css'),encoding='utf-8')
    write_json(V14_DIR/V14_MANIFEST,manifest)
    (V14_DIR/'README_review_viewer.md').write_text(_v13_ui_readme('State Machine Ad Review Viewer v1.4 Train'),encoding='utf-8')
    bad=scan_forbidden(V14_DIR)
    if bad: raise RuntimeError(f'forbidden v1.4 viewer files: {bad}')

def update_current(manifest):
    manifest=_v13_ui_manifest(manifest)
    before=None
    if (CURRENT_DIR/'current_version.json').exists():
        try: before=read_json(CURRENT_DIR/'current_version.json').get('current_version')
        except Exception: pass
    if CURRENT_DIR.exists(): shutil.rmtree(CURRENT_DIR)
    CURRENT_DIR.mkdir(parents=True,exist_ok=True)
    (CURRENT_DIR/'index.html').write_text(_v13_static('index.html'),encoding='utf-8')
    (CURRENT_DIR/'app.js').write_text(_v13_static('app.js',CURRENT_MANIFEST,'current viewer v1_4_train'),encoding='utf-8')
    (CURRENT_DIR/'style.css').write_text(_v13_static('style.css'),encoding='utf-8')
    write_json(CURRENT_DIR/CURRENT_MANIFEST,{**manifest,'manifest_name':CURRENT_MANIFEST})
    write_json(CURRENT_DIR/'current_version.json',{'current_version':VERSION_KEY,'detector_version':VERSION_DOT,'base_version':BASE_VERSION_DOT,'scope':SCOPE,'rollback_supported':True,'rollback_targets':['v1_3_train','v1_2'],'validation_included':False,'test_included':False,'updated_at':now()})
    (CURRENT_DIR/'README_current_viewer.md').write_text(_v13_ui_readme('Current State Machine Ad Review Viewer'),encoding='utf-8')
    bad=scan_forbidden(CURRENT_DIR)
    if bad: raise RuntimeError(f'forbidden current viewer files: {bad}')
    return before

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default=str(PROJECT_ROOT)); ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args()
    if Path(args.project_root).resolve()!=PROJECT_ROOT: print('bad root',file=sys.stderr); return 2
    LOG_PATH.parent.mkdir(parents=True,exist_ok=True); LOG_PATH.write_text('',encoding='utf-8'); warnings=[]; errors=[]
    try:
        log('[STEP 01] Safety snapshot and backup')
        ts=stamp(); old_before=snapshot(OLD_PROJECT_ROOT,REPORT_DIR/'old_project_snapshot_before_state_machine_ad_review_viewer_v1_4_train_update.tsv'); backup_info=backup(ts)
        input_before={k:stats(PROJECT_ROOT/v) for k,v in INPUTS.items()}; write_json(REPORT_DIR/'state_machine_ad_review_viewer_v1_4_train_input_file_stats_before.json',input_before)
        log('[STEP 02] Locate current/v1.3/v1.2 viewer and rollback source')
        if not CURRENT_DIR.exists(): raise RuntimeError('current viewer missing')
        if not V13_DIR.exists(): warnings.append('v1.3 train viewer missing')
        if not V12_DIR.exists(): warnings.append('v1.2 viewer missing')
        log('[STEP 03] Load v1.4 outputs')
        missing=[k for k,v in INPUTS.items() if not (PROJECT_ROOT/v).exists()]
        if missing: raise RuntimeError(f'missing inputs: {missing}')
        data={k:read_csv(v) for k,v in INPUTS.items() if (PROJECT_ROOT/v).suffix=='.csv'}; data['detector_report']=read_json(PROJECT_ROOT/INPUTS['detector_report'])
        for k in ['high','review','trace','open','unresolved','rejected','disclosure','evidence','filter','audit','comparison','video_summary']: ensure_train(k,data[k])
        log('[STEP 04] Build v1.4 train manifest')
        manifest=build_manifest(data,warnings)
        if args.dry_run: print(json.dumps({'videos':len(manifest['videos']),'meta':manifest['meta']},ensure_ascii=False,indent=2)); return 0
        log('[STEP 05] Generate v1.4 train viewer static files'); write_viewer(manifest)
        log('[STEP 06] Update current viewer to v1.4 train'); before=update_current(manifest)
        log('[STEP 07] Update switch script'); registry=update_registry(); patch_scripts()
        log('[STEP 08] Update current server if needed')
        log('[STEP 09] Generate report and summary')
        old_after=snapshot(OLD_PROJECT_ROOT,REPORT_DIR/'old_project_snapshot_after_state_machine_ad_review_viewer_v1_4_train_update.tsv')
        input_after={k:stats(PROJECT_ROOT/v) for k,v in INPUTS.items()}; input_mod=[k for k,v in input_after.items() if input_before.get(k)!=v]
        report={'task_name':'state_machine_ad_review_viewer_v1_4_train_update','version':'v1_4_train_viewer_update','project_root':str(PROJECT_ROOT),'generated_at':now(),'input_files':{k:str(PROJECT_ROOT/v) for k,v in INPUTS.items()},'output_files':{'v1_4_viewer_dir':str(V14_DIR),'current_viewer_dir':str(CURRENT_DIR),'registry':str(REGISTRY_PATH),'summary_md':str(SUMMARY_PATH),'report_json':str(REPORT_PATH),'run_log':str(LOG_PATH),'latest_bundle':str(LATEST_DIR)},'existing_v1_3_viewer_found':V13_DIR.exists(),'existing_v1_2_viewer_found':V12_DIR.exists(),'rollback_supported':True,'current_version_before':before,'current_version_after':VERSION_KEY,'counts':manifest['meta'],'video_count_by_split':{'train':len(manifest['videos']),'validation':0,'test':0},'actual_interval_count':sum(len(v['actual_intervals']) for v in manifest['videos']),'playable_video_count':sum(1 for v in manifest['videos'] if v['playable']),'missing_video_count':sum(1 for v in manifest['videos'] if not v['playable']),'validation_included':False,'test_included':False,'old_project_modified':old_before['sha256']!=old_after['sha256'],'input_files_modified':input_mod,'latest_for_chatgpt_forbidden_files_found':[],'warnings':warnings,'errors':errors,'no_detector_run':True,'no_feature_extraction':True,'no_threshold_tuning':True,'actual_label_usage':'audit_ui_only_not_detector_decision','backup':backup_info,'sub_agent_results':[]}
        write_json(REPORT_PATH,report); SUMMARY_PATH.write_text(summary_text(report,manifest),encoding='utf-8')
        log('[STEP 11] Update latest bundle'); bad=latest(); report['latest_for_chatgpt_forbidden_files_found']=bad; write_json(REPORT_PATH,report); shutil.copy2(REPORT_PATH,LATEST_DIR/REPORT_PATH.name); shutil.copy2(LOG_PATH,LATEST_DIR/LOG_PATH.name)
        log('[STEP 12] Final human-readable summary')
        print('작업 상태: SUCCESS'); print(f'current viewer version before/after: {before} -> {VERSION_KEY}'); print(f'v1.4 train viewer path: {V14_DIR}'); print(f'current viewer path: {CURRENT_DIR}'); print('rollback supported: true'); print('rollback command: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_3_train'); print('v1.4 re-switch command: python scripts/review/switch_state_machine_review_viewer_version.py --version v1_4_train'); print('server command: python scripts/review/serve_state_machine_ad_review_viewer_current.py --host 127.0.0.1 --port 8000'); print('local browser: forward port 8000 in VS Code, then open http://localhost:8000'); print(f'train video count: {len(manifest["videos"])}'); print(f"v1.4 counts: high={manifest['meta']['high_confidence_prediction_count']}, review={manifest['meta']['review_only_candidate_count']}, open={manifest['meta']['open_interval_count']}, unresolved={manifest['meta']['unresolved_long_open_count']}, rejected={manifest['meta']['rejected_long_open_count']}, disclosure={manifest['meta']['disclosure_notice_rejected_count']}, confidence_filter={manifest['meta']['interval_confidence_filter_event_count']}"); print(f"old_project_modified: {report['old_project_modified']}"); print(f"input_files_modified: {report['input_files_modified']}"); print('validation_included: false'); print('test_included: false'); print(f'latest bundle path: {LATEST_DIR}'); print(f'warnings/errors: {warnings} / {errors}')
        return 0
    except Exception as e:
        errors.append(str(e)); log(f'[FAILURE] {e}'); return 1
if __name__=='__main__': raise SystemExit(main())
