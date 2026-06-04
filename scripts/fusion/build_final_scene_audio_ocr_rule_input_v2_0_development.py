#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path('.')
VERSION = 'v2.0'
DETECTOR_ID = 'state_machine_interval_detector_v2_0_development'
DEV_IDS = {1,2,5,6,8,9,10,11,12,13,14,15}
FORBIDDEN_PATTERNS = ['label','true','actual','gt','ground_truth','audit','nearest_true_boundary','distance_to_nearest_true_boundary','is_near_true_boundary','ad_overlap','is_ad_overlap','is_ad_core','is_clean_nonad','overlapping_ad_interval_ids','per_video_ad_vs_nonad_contrast_score','audio_candidate_score_for_discussion']
PATHS = {
 'split': ROOT/'data/splits/video_split_v2_4.csv',
 'manifest': ROOT/'data/video_metadata/video_manifest_v2_2.csv',
 'scene': ROOT/'data/scene/final_scene_boundary_anchor_v2_5_development.csv',
 'ocr_anchor': ROOT/'workspaces/ocr_final_scene_anchor_v2_5_development/runs/final_scene_anchor_ocr_v2_5_development_20260527_050904/final_scene_anchor_ocr_anchor_window_features_v2_5_development.csv',
 'ocr_timeline': ROOT/'workspaces/ocr_final_scene_anchor_v2_5_development/runs/final_scene_anchor_ocr_v2_5_development_20260527_050904/final_scene_anchor_ocr_20s_timeline_features_v2_5_development.csv',
 'audio': ROOT/'data/audio/per_video_audio_relative_levels_v2_4_train.csv',
 'audio_discussion': ROOT/'data/audio/per_video_train_audio_candidate_score_for_discussion_v2_4.csv',
 'fusion_input': ROOT/'data/fusion/final_scene_audio_ocr_rule_input_v2_0_development.csv',
 'schema_audit': ROOT/'data/fusion/final_scene_audio_ocr_rule_input_v2_0_development_schema_audit.csv',
 'join_quality': ROOT/'data/fusion/final_scene_audio_ocr_rule_input_v2_0_development_join_quality.csv',
 'column_mapping': ROOT/'data/fusion/final_scene_audio_ocr_rule_input_v2_0_development_column_mapping.json',
}

def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, '') for c in columns})

def fnum(v: Any, default: float=0.0) -> float:
    try:
        if v is None or v == '': return default
        x = float(v)
        if math.isnan(x): return default
        return x
    except Exception:
        return default

def bval(v: Any) -> bool:
    return str(v).strip().lower() in {'true','1','yes','y'}

def level(score: float, high=0.65, med=0.35) -> str:
    if score >= high: return 'high'
    if score >= med: return 'medium'
    return 'low'

def reliability_level(score: float) -> str:
    if score >= 0.8: return 'high'
    if score >= 0.45: return 'medium'
    return 'low'

def mmss(sec: float) -> str:
    s = max(0, int(round(sec)))
    return f'{s//60:02d}:{s%60:02d}'

def p90(vals):
    vals = sorted([float(v) for v in vals if v is not None])
    if not vals: return 0.0
    idx = min(len(vals)-1, int(math.ceil(0.9*len(vals)))-1)
    return vals[idx]

def overlap(row, start, end):
    return fnum(row.get('start_sec')) < end and fnum(row.get('end_sec')) > start

def sum_col(rows, col):
    return sum(fnum(r.get(col)) for r in rows)

def max_col(rows, col):
    vals = [fnum(r.get(col)) for r in rows if r.get(col) not in (None, '')]
    return max(vals) if vals else 0.0

def avg_col(rows, col):
    vals = [fnum(r.get(col)) for r in rows if r.get(col) not in (None, '')]
    return sum(vals)/len(vals) if vals else 0.0

def count_by_video(rows):
    d=defaultdict(list)
    for r in rows:
        d[int(float(r.get('video_id') or 0))].append(r)
    return d

def forbidden(cols):
    bad=[]
    for c in cols:
        low=c.lower()
        for pat in FORBIDDEN_PATTERNS:
            if pat == 'gt':
                if low == 'gt' or low.startswith('gt_') or low.endswith('_gt') or '_gt_' in low:
                    bad.append(c); break
            elif pat in low:
                bad.append(c); break
    return sorted(set(bad))

def main():
    scene = read_csv(PATHS['scene'])
    ocr_anchor = read_csv(PATHS['ocr_anchor'])
    timeline = read_csv(PATHS['ocr_timeline'])
    audio = read_csv(PATHS['audio'])
    split_rows = read_csv(PATHS['split'])
    manifest = read_csv(PATHS['manifest']) if PATHS['manifest'].exists() else []
    duration_by_video = {}
    for r in split_rows:
        vid = int(float(r.get('video_id') or 0))
        if vid in DEV_IDS and r.get('split') == 'train':
            duration_by_video[vid] = fnum(r.get('video_duration_sec'))
    for r in manifest:
        vid = int(float(r.get('video_id') or 0))
        duration_by_video.setdefault(vid, fnum(r.get('duration_sec')))

    scene = [r for r in scene if int(float(r.get('video_id') or 0)) in DEV_IDS and r.get('original_split_v2_4') == 'train' and r.get('split_role_v2_5') == 'development']
    ocr_anchor = [r for r in ocr_anchor if int(float(r.get('video_id') or 0)) in DEV_IDS and r.get('original_split_v2_4') == 'train']
    timeline = [r for r in timeline if int(float(r.get('video_id') or 0)) in DEV_IDS and r.get('original_split_v2_4') == 'train']
    audio = [r for r in audio if int(float(r.get('video_id') or 0)) in DEV_IDS and r.get('split') == 'train']

    ocr_by_id = {r.get('scene_boundary_anchor_id'): r for r in ocr_anchor if r.get('scene_boundary_anchor_id')}
    ocr_by_video = count_by_video(ocr_anchor)
    timeline_by_video = count_by_video(timeline)
    audio_by_video = count_by_video(audio)
    for vid in audio_by_video:
        audio_by_video[vid].sort(key=lambda r: fnum(r.get('start_sec')))
    for vid in timeline_by_video:
        timeline_by_video[vid].sort(key=lambda r: fnum(r.get('segment_start_sec')))

    decision_cols = [
        'ocr_hard_disclosure_flag','ocr_product_cta_continuity_flag','ocr_timeline_recent_hard_evidence_flag',
        'ocr_start_signal_level_v2_0','ocr_end_signal_level_v2_0','ocr_context_level_v2_0','ocr_context_reliability_level',
        'audio_pre_relative_active_score','audio_post_relative_active_score','audio_context_relative_active_score',
        'audio_pre_relative_quiet_score','audio_post_relative_quiet_score','audio_context_relative_quiet_score',
        'audio_relative_local_shift_score','audio_relative_sustained_context_score','audio_start_signal_level_v2_0',
        'audio_end_signal_level_v2_0','audio_context_level_v2_0','audio_not_informative_flag',
        'opening_disclosure_guard_applied','opening_disclosure_confirmed_by_later_product_cta',
        'hard_evidence_flag_v2_0','support_evidence_flag_v2_0','weak_context_flag_v2_0'
    ]
    bad_decision = forbidden(decision_cols)
    rows=[]
    schema_warnings=set()
    ocr_exact=0; ocr_nearest=0; audio_nonempty=0; timeline_nonempty=0
    for s in sorted(scene, key=lambda r:(int(float(r.get('video_id') or 0)), fnum(r.get('anchor_sec')))):
        vid = int(float(s.get('video_id') or 0))
        t = fnum(s.get('anchor_sec'))
        final_id = s.get('final_anchor_id','')
        ocr = ocr_by_id.get(final_id)
        join_method='scene_boundary_anchor_id_exact'
        if ocr is None:
            candidates = ocr_by_video.get(vid, [])
            nearest = min(candidates, key=lambda r: abs(fnum(r.get('canonical_boundary_time_sec'))-t), default=None)
            if nearest is not None and abs(fnum(nearest.get('canonical_boundary_time_sec'))-t) <= 0.25:
                ocr = nearest; join_method='video_id_time_nearest_0_25s'; ocr_nearest += 1
            else:
                ocr = {}; join_method='missing'
        else:
            ocr_exact += 1
        dur = duration_by_video.get(vid, fnum(s.get('video_duration_sec'), 0.0))
        pre_tl = [r for r in timeline_by_video.get(vid, []) if fnum(r.get('segment_end_sec')) > max(0,t-20) and fnum(r.get('segment_start_sec')) < t]
        post_tl = [r for r in timeline_by_video.get(vid, []) if fnum(r.get('segment_start_sec')) < t+20 and fnum(r.get('segment_end_sec')) > t]
        context_tl = [r for r in timeline_by_video.get(vid, []) if fnum(r.get('segment_end_sec')) > max(0,t-30) and fnum(r.get('segment_start_sec')) < t+30]
        if context_tl: timeline_nonempty += 1
        audio_rows = audio_by_video.get(vid, [])
        pre_a = [r for r in audio_rows if overlap(r, max(0,t-10), t)]
        post_a = [r for r in audio_rows if overlap(r, t, t+10)]
        context_a = [r for r in audio_rows if overlap(r, max(0,t-10), t+10)]
        if context_a: audio_nonempty += 1

        anchor_disc = fnum(ocr.get('corrected_ad_disclosure_hit_count_sum'))
        anchor_sponsor = fnum(ocr.get('corrected_sponsor_keyword_count_sum'))
        anchor_product = fnum(ocr.get('corrected_brand_product_keyword_count_sum'))
        anchor_promo = fnum(ocr.get('corrected_promotion_discount_keyword_count_sum'))
        anchor_cta = fnum(ocr.get('corrected_purchase_cta_keyword_count_sum'))
        anchor_link = fnum(ocr.get('corrected_link_more_info_keyword_count_sum'))
        anchor_neg = fnum(ocr.get('corrected_negative_guard_keyword_count_sum'))
        pre_disc = sum_col(pre_tl,'corrected_ad_disclosure_hit_count_sum')
        post_disc = max(sum_col(post_tl,'corrected_ad_disclosure_hit_count_sum'), anchor_disc)
        pre_sponsor = sum_col(pre_tl,'corrected_sponsor_keyword_count_sum')
        post_sponsor = max(sum_col(post_tl,'corrected_sponsor_keyword_count_sum'), anchor_sponsor)
        pre_product = sum_col(pre_tl,'corrected_brand_product_keyword_count_sum')
        post_product = max(sum_col(post_tl,'corrected_brand_product_keyword_count_sum'), anchor_product)
        pre_promo = sum_col(pre_tl,'corrected_promotion_discount_keyword_count_sum')
        post_promo = max(sum_col(post_tl,'corrected_promotion_discount_keyword_count_sum'), anchor_promo)
        pre_cta = sum_col(pre_tl,'corrected_purchase_cta_keyword_count_sum')
        post_cta = max(sum_col(post_tl,'corrected_purchase_cta_keyword_count_sum'), anchor_cta)
        pre_link = sum_col(pre_tl,'corrected_link_more_info_keyword_count_sum')
        post_link = max(sum_col(post_tl,'corrected_link_more_info_keyword_count_sum'), anchor_link)
        pre_neg = sum_col(pre_tl,'corrected_negative_guard_keyword_count_sum')
        post_neg = max(sum_col(post_tl,'corrected_negative_guard_keyword_count_sum'), anchor_neg)
        pre_score = avg_col(pre_tl,'corrected_frame_ad_text_score_mean')
        post_score = max(avg_col(post_tl,'corrected_frame_ad_text_score_mean'), fnum(ocr.get('corrected_frame_ad_text_score_mean')))
        context_score = max(max_col(context_tl,'corrected_frame_ad_text_score_max'), fnum(ocr.get('corrected_frame_ad_text_score_max')))
        recent_hard_segments = [r for r in context_tl if fnum(r.get('corrected_ad_disclosure_hit_count_sum'))>0 and fnum(r.get('corrected_negative_guard_keyword_count_sum'))<=0 and (fnum(r.get('corrected_brand_product_keyword_count_sum'))>0 or fnum(r.get('corrected_purchase_cta_keyword_count_sum'))>0 or fnum(r.get('corrected_link_more_info_keyword_count_sum'))>0 or fnum(r.get('corrected_frame_ad_text_score_max'))>=0.55)]
        recent_hard = bool(recent_hard_segments)
        recent_time = ''
        if recent_hard_segments:
            recent_time = max(fnum(r.get('segment_start_sec')) for r in recent_hard_segments)
        valid_ratio = 0.0
        frame_count = fnum(ocr.get('ocr_frame_count'))
        if frame_count > 0:
            valid_ratio = (fnum(ocr.get('ocr_success_text_count')) + fnum(ocr.get('ocr_success_empty_count'))) / frame_count
        text_ratio = fnum(ocr.get('ocr_text_frame_ratio'))
        context_valid = max(valid_ratio, avg_col(context_tl,'ocr_text_frame_ratio'))
        reliability = reliability_level(context_valid if context_valid else text_ratio)
        exact = post_disc
        typo = 0.0; prox = 0.0; fuzzy = 0.0
        schema_warnings.add('ocr_disclosure_subtype_columns_missing_mapped_corrected_ad_disclosure_hit_count_to_exact')
        neg_total = pre_neg + post_neg
        has_product = pre_product + post_product > 0
        has_cta_link = pre_cta + post_cta + pre_link + post_link > 0
        post_increase = post_score > pre_score + 0.05 or context_score >= 0.55
        hard_disc = exact + typo + prox > 0 and neg_total <= 0 and (has_product or has_cta_link or post_increase or recent_hard)
        fuzzy_review = fuzzy > 0 and exact + typo + prox == 0
        product_cta = has_product and has_cta_link
        support_ocr = (pre_sponsor + post_sponsor > 0) or has_product or has_cta_link or (reliability in {'medium','high'} and context_score >= 0.25) or ((pre_promo+post_promo)>0 and (has_product or has_cta_link))
        weak_ocr = fuzzy_review or (pre_promo+post_promo>0 and not (has_product or has_cta_link)) or context_valid < 0.4 or frame_count == 0

        pre_active = p90([fnum(r.get('per_video_relative_active_audio_score')) for r in pre_a])
        post_active = p90([fnum(r.get('per_video_relative_active_audio_score')) for r in post_a])
        context_active = p90([fnum(r.get('per_video_relative_active_audio_score')) for r in context_a])
        pre_quiet = p90([fnum(r.get('per_video_relative_quiet_audio_score')) for r in pre_a])
        post_quiet = p90([fnum(r.get('per_video_relative_quiet_audio_score')) for r in post_a])
        context_quiet = p90([fnum(r.get('per_video_relative_quiet_audio_score')) for r in context_a])
        local_shift = max(avg_col(context_a,'per_video_local_context_shift_score'), post_active - pre_active, post_quiet - pre_quiet)
        sustained = p90([fnum(r.get('per_video_sustained_context_score')) for r in context_a])
        audio_not_info = not context_a or max(context_active, context_quiet, abs(local_shift), sustained) < 0.05
        act_label = 'not_informative' if audio_not_info else ('relative_active' if context_active >= context_quiet else 'relative_quiet')
        start_audio = level(max(post_active, context_active, max(0, local_shift)))
        end_audio = level(max(post_quiet, context_quiet, max(0, post_quiet-pre_quiet)))
        context_audio = level(max(context_active, sustained))
        before_audio = level(pre_active)
        after_audio = level(post_active)

        opening_win = min(45.0, max(30.0, 0.02*dur)) if dur else 30.0
        in_opening = t <= opening_win
        disclosure_present = exact + typo + prox > 0
        later_confirm = (post_product + post_cta + post_link > 0) or recent_hard or (post_active >= 0.65 and post_score > pre_score)
        opening_notice = in_opening and disclosure_present and not later_confirm
        opening_guard = opening_notice

        hard_reasons=[]; support_reasons=[]; weak_reasons=[]
        if hard_disc: hard_reasons.append('hard_ocr_disclosure_confirmed')
        if product_cta: hard_reasons.append('product_cta_continuity')
        if recent_hard: hard_reasons.append('ocr_timeline_recent_hard')
        if support_ocr: support_reasons.append('ocr_support_context')
        if start_audio in {'medium','high'} and (hard_disc or support_ocr): support_reasons.append('same_video_relative_audio_active_support')
        if end_audio in {'medium','high'} and reliability == 'high' and context_score < 0.2: support_reasons.append('audio_quiet_with_ocr_drop_end_support')
        if weak_ocr: weak_reasons.append('ocr_weak_or_unknown')
        if audio_not_info: weak_reasons.append('audio_not_informative')
        if opening_notice: weak_reasons.append('opening_disclosure_notice_review_only')
        hard_flag = bool(hard_reasons) and not opening_notice
        support_flag = bool(support_reasons)
        weak_flag = bool(weak_reasons)
        src_count = int(bval(s.get('has_opencv_ffmpeg'))) + int(bval(s.get('has_resnet'))) + int(bval(s.get('has_transnetv2_conservative')))
        scene_score = src_count / 3.0
        row = {
            'version': VERSION, 'detector_id': DETECTOR_ID, 'original_split_v2_4':'train', 'split_role_v2_5':'development', 'evaluation_subset_v2_5':'none',
            'video_id': vid, 'video_duration_sec': f'{dur:.6f}', 'final_scene_anchor_id': final_id, 'transition_time_anchor': f'{t:.6f}', 'candidate_time_sec': f'{t:.6f}', 'candidate_time_mmss': mmss(t),
            'final_anchor_source_count': src_count, 'has_opencv_ffmpeg': str(bval(s.get('has_opencv_ffmpeg'))).lower(), 'has_resnet': str(bval(s.get('has_resnet'))).lower(), 'has_transnetv2_conservative': str(bval(s.get('has_transnetv2_conservative'))).lower(),
            'scene_anchor_role':'transition_time_anchor', 'scene_transition_reliability_score': f'{scene_score:.6f}', 'scene_transition_reliability_level': reliability_level(scene_score), 'scene_is_direct_ad_start_evidence':'false','scene_is_direct_ad_end_evidence':'false','scene_used_for_ad_likelihood_directly':'false',
            'ocr_context_reliability_level': reliability, 'ocr_pre_valid_frame_ratio': f'{avg_col(pre_tl,"ocr_text_frame_ratio"):.6f}', 'ocr_post_valid_frame_ratio': f'{max(avg_col(post_tl,"ocr_text_frame_ratio"), text_ratio):.6f}', 'ocr_context_valid_frame_ratio': f'{context_valid:.6f}', 'ocr_pre_text_frame_ratio': f'{avg_col(pre_tl,"ocr_text_frame_ratio"):.6f}', 'ocr_post_text_frame_ratio': f'{max(avg_col(post_tl,"ocr_text_frame_ratio"), text_ratio):.6f}',
            'corrected_pre_ad_disclosure_exact_count': f'{pre_disc:.0f}', 'corrected_post_ad_disclosure_exact_count': f'{post_disc:.0f}', 'corrected_pre_ad_disclosure_typo_count': '0', 'corrected_post_ad_disclosure_typo_count': '0', 'corrected_pre_ad_disclosure_proximity_count': '0', 'corrected_post_ad_disclosure_proximity_count': '0', 'corrected_pre_ad_disclosure_fuzzy_review_count': '0', 'corrected_post_ad_disclosure_fuzzy_review_count':'0',
            'corrected_pre_brand_product_count': f'{pre_product:.0f}', 'corrected_post_brand_product_count': f'{post_product:.0f}', 'corrected_pre_purchase_cta_count': f'{pre_cta:.0f}', 'corrected_post_purchase_cta_count': f'{post_cta:.0f}', 'corrected_pre_link_more_info_count': f'{pre_link:.0f}', 'corrected_post_link_more_info_count': f'{post_link:.0f}', 'corrected_pre_promotion_discount_count': f'{pre_promo:.0f}', 'corrected_post_promotion_discount_count': f'{post_promo:.0f}', 'corrected_pre_sponsor_count': f'{pre_sponsor:.0f}', 'corrected_post_sponsor_count': f'{post_sponsor:.0f}', 'corrected_pre_negative_guard_count': f'{pre_neg:.0f}', 'corrected_post_negative_guard_count': f'{post_neg:.0f}',
            'ocr_hard_disclosure_flag': str(hard_disc).lower(), 'ocr_fuzzy_only_review_flag': str(fuzzy_review).lower(), 'ocr_product_cta_continuity_flag': str(product_cta).lower(), 'ocr_timeline_recent_hard_evidence_flag': str(recent_hard).lower(), 'ocr_timeline_recent_hard_evidence_time': recent_time,
            'ocr_start_signal_level_v2_0': 'high' if hard_disc else ('medium' if support_ocr else 'low'), 'ocr_end_signal_level_v2_0': 'medium' if reliability == 'high' and context_score < 0.2 and post_quiet >= 0.35 else 'low', 'ocr_context_level_v2_0': 'high' if context_score >= 0.55 else ('medium' if context_score >= 0.25 or support_ocr else 'low'),
            'audio_pre_relative_active_score': f'{pre_active:.6f}', 'audio_post_relative_active_score': f'{post_active:.6f}', 'audio_context_relative_active_score': f'{context_active:.6f}', 'audio_pre_relative_quiet_score': f'{pre_quiet:.6f}', 'audio_post_relative_quiet_score': f'{post_quiet:.6f}', 'audio_context_relative_quiet_score': f'{context_quiet:.6f}', 'audio_relative_local_shift_score': f'{local_shift:.6f}', 'audio_relative_sustained_context_score': f'{sustained:.6f}', 'audio_relative_activity_label': act_label,
            'audio_start_signal_level_v2_0': start_audio, 'audio_end_signal_level_v2_0': end_audio, 'audio_context_level_v2_0': context_audio, 'audio_before_context_level_v2_0': before_audio, 'audio_after_context_level_v2_0': after_audio, 'audio_not_informative_flag': str(audio_not_info).lower(),
            'opening_disclosure_window_sec': f'{opening_win:.6f}', 'is_in_opening_disclosure_window': str(in_opening).lower(), 'opening_disclosure_notice_flag': str(opening_notice).lower(), 'opening_disclosure_guard_applied': str(opening_guard).lower(), 'opening_disclosure_confirmed_by_later_product_cta': str(later_confirm).lower(),
            'hard_evidence_flag_v2_0': str(hard_flag).lower(), 'support_evidence_flag_v2_0': str(support_flag).lower(), 'weak_context_flag_v2_0': str(weak_flag).lower(), 'hard_evidence_reasons_v2_0': ';'.join(hard_reasons), 'support_evidence_reasons_v2_0': ';'.join(support_reasons), 'weak_context_reasons_v2_0': ';'.join(weak_reasons), 'evidence_tier_summary_v2_0': f'hard={len(hard_reasons)};support={len(support_reasons)};weak={len(weak_reasons)}',
            'decision_feature_columns_json': json.dumps(decision_cols, ensure_ascii=False), 'forbidden_decision_columns_found': json.dumps(bad_decision, ensure_ascii=False), 'label_derived_audio_columns_excluded': 'true', 'audit_columns_used_for_decision': 'false',
            'ocr_join_method': join_method, 'audio_window_row_count': len(context_a), 'timeline_window_segment_count': len(context_tl),
        }
        rows.append(row)
    columns = ['version','detector_id','original_split_v2_4','split_role_v2_5','evaluation_subset_v2_5','video_id','video_duration_sec','final_scene_anchor_id','transition_time_anchor','candidate_time_sec','candidate_time_mmss','final_anchor_source_count','has_opencv_ffmpeg','has_resnet','has_transnetv2_conservative','scene_anchor_role','scene_transition_reliability_score','scene_transition_reliability_level','scene_is_direct_ad_start_evidence','scene_is_direct_ad_end_evidence','scene_used_for_ad_likelihood_directly','ocr_context_reliability_level','ocr_pre_valid_frame_ratio','ocr_post_valid_frame_ratio','ocr_context_valid_frame_ratio','ocr_pre_text_frame_ratio','ocr_post_text_frame_ratio','corrected_pre_ad_disclosure_exact_count','corrected_post_ad_disclosure_exact_count','corrected_pre_ad_disclosure_typo_count','corrected_post_ad_disclosure_typo_count','corrected_pre_ad_disclosure_proximity_count','corrected_post_ad_disclosure_proximity_count','corrected_pre_ad_disclosure_fuzzy_review_count','corrected_post_ad_disclosure_fuzzy_review_count','corrected_pre_brand_product_count','corrected_post_brand_product_count','corrected_pre_purchase_cta_count','corrected_post_purchase_cta_count','corrected_pre_link_more_info_count','corrected_post_link_more_info_count','corrected_pre_promotion_discount_count','corrected_post_promotion_discount_count','corrected_pre_sponsor_count','corrected_post_sponsor_count','corrected_pre_negative_guard_count','corrected_post_negative_guard_count','ocr_hard_disclosure_flag','ocr_fuzzy_only_review_flag','ocr_product_cta_continuity_flag','ocr_timeline_recent_hard_evidence_flag','ocr_timeline_recent_hard_evidence_time','ocr_start_signal_level_v2_0','ocr_end_signal_level_v2_0','ocr_context_level_v2_0','audio_pre_relative_active_score','audio_post_relative_active_score','audio_context_relative_active_score','audio_pre_relative_quiet_score','audio_post_relative_quiet_score','audio_context_relative_quiet_score','audio_relative_local_shift_score','audio_relative_sustained_context_score','audio_relative_activity_label','audio_start_signal_level_v2_0','audio_end_signal_level_v2_0','audio_context_level_v2_0','audio_before_context_level_v2_0','audio_after_context_level_v2_0','audio_not_informative_flag','opening_disclosure_window_sec','is_in_opening_disclosure_window','opening_disclosure_notice_flag','opening_disclosure_guard_applied','opening_disclosure_confirmed_by_later_product_cta','hard_evidence_flag_v2_0','support_evidence_flag_v2_0','weak_context_flag_v2_0','hard_evidence_reasons_v2_0','support_evidence_reasons_v2_0','weak_context_reasons_v2_0','evidence_tier_summary_v2_0','decision_feature_columns_json','forbidden_decision_columns_found','label_derived_audio_columns_excluded','audit_columns_used_for_decision','ocr_join_method','audio_window_row_count','timeline_window_segment_count']
    write_csv(PATHS['fusion_input'], rows, columns)
    schema_rows=[]
    for col in columns:
        schema_rows.append({'column':col,'present':'true','decision_feature':str(col in decision_cols).lower(),'forbidden_if_decision':str(col in bad_decision).lower(),'notes':''})
    for warn in sorted(schema_warnings):
        schema_rows.append({'column':warn,'present':'false','decision_feature':'false','forbidden_if_decision':'false','notes':'warning'})
    write_csv(PATHS['schema_audit'], schema_rows, ['column','present','decision_feature','forbidden_if_decision','notes'])
    jq=[
        {'metric':'final_scene_anchor_rows','value':len(scene),'notes':'primary row count'},
        {'metric':'ocr_anchor_feature_rows','value':len(ocr_anchor),'notes':'Development Set rows'},
        {'metric':'ocr_exact_join_count','value':ocr_exact,'notes':'scene_boundary_anchor_id exact joins'},
        {'metric':'ocr_nearest_join_count','value':ocr_nearest,'notes':'fallback joins'},
        {'metric':'ocr_join_coverage','value':f'{(ocr_exact+ocr_nearest)/len(scene):.6f}' if scene else '0','notes':''},
        {'metric':'ocr_timeline_feature_rows','value':len(timeline),'notes':'Development Set 20s rows'},
        {'metric':'ocr_timeline_window_coverage','value':f'{timeline_nonempty/len(scene):.6f}' if scene else '0','notes':''},
        {'metric':'audio_relative_rows','value':len(audio),'notes':'Development Set audio subwindows'},
        {'metric':'audio_join_coverage','value':f'{audio_nonempty/len(scene):.6f}' if scene else '0','notes':'anchors with context audio window rows'},
        {'metric':'decision_forbidden_columns_found','value':json.dumps(bad_decision),'notes':'must be []'},
    ]
    write_csv(PATHS['join_quality'], jq, ['metric','value','notes'])
    mapping={'created_at':datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),'join_strategy':{'ocr_anchor':'scene_boundary_anchor_id exact; fallback video_id + canonical_boundary_time_sec within 0.25s','ocr_timeline':'video_id and 20s segment overlap around anchor','audio':'video_id and audio subwindow overlap around anchor pre/post/context windows'},'input_paths':{k:str(v) for k,v in PATHS.items() if k not in {'fusion_input','schema_audit','join_quality','column_mapping'}},'decision_feature_columns':decision_cols,'forbidden_decision_columns_found':bad_decision,'warnings':sorted(schema_warnings)}
    PATHS['column_mapping'].write_text(json.dumps(mapping, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'fusion_input_rows':len(rows),'ocr_join_coverage':(ocr_exact+ocr_nearest)/len(scene) if scene else 0,'audio_join_coverage':audio_nonempty/len(scene) if scene else 0}, ensure_ascii=False))
if __name__ == '__main__': main()
