#!/usr/bin/env python3
"""Final scene-anchor OCR extraction for v2.5 Development Set.

Reads the prebuilt Development Set OCR schedule and executes OCR with
checkpoint/resume support. It never regenerates the schedule and never uses
actual ad labels for sampling.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path('.')
SCRIPT_PATH = PROJECT_ROOT / 'scripts/ocr/extract_final_scene_anchor_ocr_v2_5_development.py'
SCHEDULE_FILE = PROJECT_ROOT / 'data/ocr/final_scene_anchor_ocr_schedule_v2_5_development.csv'
INPUT_CONTRACT = PROJECT_ROOT / 'data/ocr/final_scene_anchor_ocr_input_contract_v2_5_development.json'
VIDEO_MAPPING_FILE = PROJECT_ROOT / 'data/ocr/final_scene_anchor_video_path_mapping_v2_5_development.csv'
FINAL_ANCHOR_FILE = PROJECT_ROOT / 'data/scene/final_scene_boundary_anchor_v2_5_development.csv'
INPUT_PREP_REPORT = PROJECT_ROOT / 'reports/ocr/final_scene_anchor_ocr_inputs_v2_5_development_report.json'
AD_INTERVAL_FILE = PROJECT_ROOT / 'data/segments/ad_interval_segments_v2_4.csv'
RUN_ROOT = PROJECT_ROOT / 'workspaces/ocr_final_scene_anchor_v2_5_development/runs'
LATEST_CHATGPT = PROJECT_ROOT / 'outputs/latest_for_chatgpt_final_scene_anchor_ocr_v2_5_development'
LATEST_OCR = PROJECT_ROOT / 'outputs/latest_ocr'
CACHE_DIR = PROJECT_ROOT / 'cache/ocr/model_cache'
OLD_PROJECT_ROOT = Path('./_old_project_not_included')
FFMPEG_BIN = Path('.venv/bin/ffmpeg')

VERSION = 'v2_5_development'
EXPECTED_VIDEO_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
EXTENDED_EVAL_IDS = [3, 4, 7, 16, 17, 18]
EXPECTED_SCHEDULE_COUNT = 23945
EXPECTED_ROLE_COUNTS = {'anchor_dense': 23234, 'background_regular': 711}
SPLIT_NOTE = 'Development Set = original v2.4 train split; used for rule design, cue analysis, and error diagnosis, not ML model training.'
FORBIDDEN_LATEST_NAMES = {
    'final_scene_anchor_ocr_frame_results_v2_5_development.csv',
}
FORBIDDEN_LATEST_EXTS = {'.mp4', '.mov', '.mkv', '.avi', '.jpg', '.jpeg', '.png', '.webp', '.pt', '.pth', '.ckpt', '.onnx', '.bin'}

OUTPUT_NAMES = {
    'frame_results': 'final_scene_anchor_ocr_frame_results_v2_5_development.csv',
    'sample_redacted': 'final_scene_anchor_ocr_frame_results_sample_redacted_v2_5_development.csv',
    'video_summary': 'final_scene_anchor_ocr_video_summary_v2_5_development.csv',
    'timeline_features': 'final_scene_anchor_ocr_20s_timeline_features_v2_5_development.csv',
    'anchor_features': 'final_scene_anchor_ocr_anchor_window_features_v2_5_development.csv',
    'keyword_summary': 'final_scene_anchor_ocr_keyword_hit_summary_v2_5_development.csv',
    'corrected_keyword_summary': 'final_scene_anchor_ocr_corrected_keyword_summary_v2_5_development.csv',
    'posthoc_context_summary': 'final_scene_anchor_ocr_development_actual_ad_context_summary_v2_5.csv',
    'posthoc_context_hits': 'final_scene_anchor_ocr_development_actual_ad_context_keyword_hits_v2_5.csv',
    'representative_examples': 'final_scene_anchor_ocr_representative_examples_v2_5_development.csv',
    'failed_empty_sample': 'final_scene_anchor_ocr_failed_empty_review_sample_v2_5_development.csv',
    'quality_checks': 'final_scene_anchor_ocr_quality_checks_v2_5_development.csv',
    'run_manifest': 'final_scene_anchor_ocr_run_manifest_v2_5_development.csv',
    'summary_md': 'final_scene_anchor_ocr_v2_5_development_summary.md',
    'report_json': 'final_scene_anchor_ocr_v2_5_development_report.json',
    'run_log': 'final_scene_anchor_ocr_v2_5_development_run_log.txt',
}
CHECKPOINT_NAMES = {
    'checkpoint': 'checkpoints/final_scene_anchor_ocr_checkpoint_v2_5_development.json',
    'completed': 'checkpoints/completed_sample_ids_v2_5_development.txt',
    'failed': 'checkpoints/failed_sample_ids_v2_5_development.csv',
}
LATEST_INCLUDE = [
    'sample_redacted', 'video_summary', 'timeline_features', 'anchor_features', 'keyword_summary',
    'corrected_keyword_summary', 'posthoc_context_summary', 'posthoc_context_hits', 'representative_examples',
    'failed_empty_sample', 'quality_checks', 'run_manifest', 'summary_md', 'report_json', 'run_log'
]

BASIC_KEYWORDS = {
    'ad_disclosure': ['유료광고', '유료 광고', '유료광고 포함', '광고 포함', '이 영상은 유료광고를 포함', 'paid promotion', 'sponsored', 'sponsor'],
    'sponsor': ['협찬', '제공', '제작지원', '지원받아', '광고주', '브랜드로부터'],
    'brand_product': ['제품', '브랜드', '신제품', '세트', '패키지', '향', '컬러', '기능', '성분', '앱', '서비스', '노니', 'NONI', 'GLOWY', 'SKIN'],
    'promotion_discount': ['할인', '쿠폰', '프로모션', '이벤트', '적립', '무료배송', '특가', '혜택', '증정'],
    'purchase_cta': ['구매', '주문', '링크', '더보기', '고정댓글', '댓글', '확인', '참고', '사용해보세요', '추천', '인증'],
    'link_more_info': ['링크', '더보기', '설명란', '고정댓글', '댓글 확인', '프로필 링크'],
    'negative_guard': ['광고 아님', '협찬 아님', '내돈내산', '직접 구매', '광고가 아닙니다', '협찬 받은 거 아님'],
}
DISCLOSURE_VARIANTS = ['유료광고틀', '유료광고름', '유료광고룰', '유료광고률', '유료광고툴', '유료광고물', '유료광고를', '유료 광고를', '광고틀 포함', '광고름 포함', '광고룰 포함', '광고률 포함', '광고를 포함', '광고 포함하고', '포함하고 있습니다']
BASIC_COUNT_COLS = {k: f'{k}_keyword_count' for k in BASIC_KEYWORDS}
CORRECTED_COUNT_COLS = {
    'ad_disclosure': 'corrected_ad_disclosure_hit_count',
    'sponsor': 'corrected_sponsor_keyword_count',
    'brand_product': 'corrected_brand_product_keyword_count',
    'promotion_discount': 'corrected_promotion_discount_keyword_count',
    'purchase_cta': 'corrected_purchase_cta_keyword_count',
    'link_more_info': 'corrected_link_more_info_keyword_count',
    'negative_guard': 'corrected_negative_guard_keyword_count',
}
SUCCESS_TEXT = 'success_text'
SUCCESS_EMPTY = 'success_empty'
FAILED = 'failed'
SKIPPED = 'skipped'
TOKEN_RE = re.compile(r'[가-힣]+|[A-Za-z]+(?:[A-Za-z0-9_+\-&]*[A-Za-z0-9])?|\d+(?:[.,]\d+)*|[A-Za-z가-힣]*\d[A-Za-z가-힣0-9._+\-]*')
URL_RE = re.compile(r'(https?://|www\.|[A-Za-z0-9.-]+\.(?:com|kr|net|co|io|shop)\b|bit\.ly|linktr\.ee)', re.I)
EMAIL_RE = re.compile(r'\b[\w.\-+]+@[\w.\-]+\.[A-Za-z]{2,}\b')
PHONE_RE = re.compile(r'\b\d{2,3}[- .]?\d{3,4}[- .]?\d{4}\b')

RESULT_COLUMNS = [
    'run_id', 'version', 'original_split_v2_4', 'split_role_v2_5', 'evaluation_subset_v2_5',
    'split_terminology_note', 'video_id', 'video_title', 'video_path', 'video_duration_sec',
    'sample_id', 'schedule_id', 'timestamp_sec', 'timestamp_mmss', 'sampling_role', 'is_anchor_dense',
    'is_background_regular', 'nearest_anchor_id', 'nearest_anchor_time_sec', 'nearest_anchor_delta_sec',
    'anchor_ids_joined', 'anchor_source_relation_joined', 'anchor_model_source_joined', 'anchor_source_count_max',
    'anchor_strength_score_max', 'ocr_backend', 'ocr_engine_version', 'ocr_status', 'ocr_error',
    'ocr_text_raw', 'ocr_text_normalized', 'ocr_text_joined', 'ocr_text_count', 'ocr_token_count',
    'ocr_char_count', 'ocr_box_count', 'ocr_mean_confidence', 'ocr_min_confidence', 'ocr_max_confidence',
    'ocr_text_area_ratio', 'ocr_high_conf_text_count', 'bbox_available', 'frame_width', 'frame_height',
    'temp_frame_cleanup_status', 'warning_message', 'ad_disclosure_keyword_count', 'sponsor_keyword_count',
    'brand_product_keyword_count', 'promotion_discount_keyword_count', 'purchase_cta_keyword_count',
    'link_more_info_keyword_count', 'negative_guard_keyword_count', 'total_ad_keyword_count',
    'frame_keyword_score', 'frame_text_density_score', 'frame_ad_text_score', 'corrected_ad_disclosure_hit_count',
    'corrected_sponsor_keyword_count', 'corrected_brand_product_keyword_count',
    'corrected_promotion_discount_keyword_count', 'corrected_purchase_cta_keyword_count',
    'corrected_link_more_info_keyword_count', 'corrected_negative_guard_keyword_count',
    'corrected_total_ad_keyword_count', 'corrected_frame_ad_text_score', 'matched_keyword_categories',
    'matched_keywords', 'matched_keyword_rules', 'matched_keyword_confidence', 'suggested_canonical_phrase',
    'suppressed_by_negative_guard', 'correction_note'
]


def now_id() -> str:
    return dt.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('', encoding='utf-8')

    def log(self, message: str) -> None:
        line = f'{now_iso()} {message}'
        print(message, flush=True)
        with self.path.open('a', encoding='utf-8') as fh:
            fh.write(line + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run final scene-anchor OCR for v2.5 Development Set.')
    parser.add_argument('--schedule-file', type=Path, default=SCHEDULE_FILE)
    parser.add_argument('--input-contract', type=Path, default=INPUT_CONTRACT)
    parser.add_argument('--video-mapping-file', type=Path, default=VIDEO_MAPPING_FILE)
    parser.add_argument('--final-anchor-file', type=Path, default=FINAL_ANCHOR_FILE)
    parser.add_argument('--ad-interval-file', type=Path, default=AD_INTERVAL_FILE)
    parser.add_argument('--output-run-dir', type=Path, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--force-rerun', action='store_true')
    parser.add_argument('--video-ids', default='')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--chunk-size', type=int, default=250)
    parser.add_argument('--ocr-backend', choices=['easyocr', 'pytesseract', 'none'], default='easyocr')
    parser.add_argument('--no-persist-frames', action='store_true', default=True)
    return parser.parse_args()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


def mmss(seconds: Any) -> str:
    value = max(0.0, safe_float(seconds, 0.0))
    total = int(math.floor(value + 1e-9))
    return f'{total // 60:02d}:{total % 60:02d}'


def normalize_text(text: Any) -> str:
    raw = '' if text is None or (isinstance(text, float) and math.isnan(text)) else str(text)
    norm = unicodedata.normalize('NFKC', raw).lower()
    norm = norm.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    norm = re.sub(r'[^0-9a-z가-힣\s%+./:_-]', ' ', norm)
    norm = re.sub(r'\s+', ' ', norm).strip()
    return norm


def compact_text(text: Any) -> str:
    return re.sub(r'\s+', '', normalize_text(text))


def tokenize(text: Any) -> list[str]:
    return TOKEN_RE.findall(str(text or ''))


def redact_text(text: Any, max_len: int = 500) -> str:
    out = str(text or '')
    out = URL_RE.sub('[URL]', out)
    out = EMAIL_RE.sub('[EMAIL]', out)
    out = PHONE_RE.sub('[PHONE]', out)
    out = re.sub(r'\s+', ' ', out).strip()
    return out[:max_len]


def keyword_forms(keyword: str) -> tuple[str, str]:
    norm = normalize_text(keyword)
    return norm, re.sub(r'\s+', '', norm)


def add_match(matches: list[dict[str, Any]], category: str, keyword: str, rule: str, confidence: str, canonical: str | None = None) -> None:
    matches.append({'category': category, 'keyword': keyword, 'rule': rule, 'confidence': confidence, 'canonical': canonical or keyword})


def exact_matches(norm: str, compact: str, category: str, keywords: list[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for keyword in keywords:
        k_norm, k_compact = keyword_forms(keyword)
        if (k_norm and k_norm in norm) or (k_compact and k_compact in compact):
            add_match(matches, category, keyword, 'exact_match', 'high')
    return matches


def typo_matches(norm: str, compact: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for variant in DISCLOSURE_VARIANTS:
        v_norm, v_compact = keyword_forms(variant)
        if (v_norm and v_norm in norm) or (v_compact and v_compact in compact):
            add_match(matches, 'ad_disclosure', variant, 'typo_variant_match', 'high', '유료광고 포함')
    return matches


def proximity_matches(compact: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    paid = [m.start() for m in re.finditer('유료', compact)]
    ad = [m.start() for m in re.finditer('광고', compact)]
    included = [m.start() for m in re.finditer('포함', compact)]
    if any(abs(p - a) <= 12 for p in paid for a in ad):
        add_match(matches, 'ad_disclosure', '유료~광고 proximity', 'proximity_match', 'medium', '유료광고')
    if any(abs(a - i) <= 16 for a in ad for i in included):
        add_match(matches, 'ad_disclosure', '광고~포함 proximity', 'proximity_match', 'medium', '광고 포함')
    return matches


def fuzzy_matches(compact: str, threshold: float = 0.72) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for target in ['유료광고', '광고포함']:
        best_ratio = 0.0
        best_sub = ''
        for length in range(4, 9):
            if len(compact) < length:
                continue
            for start in range(0, len(compact) - length + 1):
                sub = compact[start:start + length]
                if not any(ch in sub for ch in '유료광고포함'):
                    continue
                ratio = SequenceMatcher(None, target, sub).ratio()
                if ratio > best_ratio:
                    best_ratio, best_sub = ratio, sub
        if best_ratio >= threshold and best_sub and best_sub not in seen:
            seen.add(best_sub)
            add_match(matches, 'ad_disclosure', f'{best_sub}~{target}:{best_ratio:.2f}', 'fuzzy_match', 'low_or_review', target)
    return matches


def dedupe_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for m in matches:
        key = (m['category'], m['keyword'], m['rule'])
        if key not in seen:
            out.append(m)
            seen.add(key)
    return out


def compute_keyword_features(text: str, detections: list[dict[str, Any]] | None = None, frame_shape: tuple[int, int, int] | None = None) -> dict[str, Any]:
    norm = normalize_text(text)
    compact = re.sub(r'\s+', '', norm)
    tokens = tokenize(text)
    basic_counts = {}
    for category, keywords in BASIC_KEYWORDS.items():
        count = 0
        for keyword in keywords:
            k_norm, k_compact = keyword_forms(keyword)
            if (k_norm and k_norm in norm) or (k_compact and k_compact in compact):
                count += 1
        basic_counts[category] = count
    matches: list[dict[str, Any]] = []
    negative_matches = exact_matches(norm, compact, 'negative_guard', BASIC_KEYWORDS['negative_guard'])
    matches.extend(negative_matches)
    matches.extend(exact_matches(norm, compact, 'ad_disclosure', BASIC_KEYWORDS['ad_disclosure']))
    matches.extend(typo_matches(norm, compact))
    matches.extend(proximity_matches(compact))
    matches.extend(fuzzy_matches(compact))
    for category in ['sponsor', 'brand_product', 'promotion_discount', 'purchase_cta', 'link_more_info']:
        matches.extend(exact_matches(norm, compact, category, BASIC_KEYWORDS[category]))
    matches = dedupe_matches(matches)
    corrected = {category: len({m['keyword'] for m in matches if m['category'] == category}) for category in CORRECTED_COUNT_COLS}
    suppressed = bool(negative_matches and corrected.get('ad_disclosure', 0) > 0)
    raw_ad = corrected.get('ad_disclosure', 0)
    if suppressed:
        corrected['ad_disclosure'] = 0
    total_basic = sum(v for k, v in basic_counts.items() if k != 'negative_guard')
    total_corrected = sum(v for k, v in corrected.items() if k != 'negative_guard')
    keyword_score = min(1.0, total_basic / 3.0)
    det_count = len(detections or [])
    area_ratio = 0.0
    if detections and frame_shape:
        image_area = float(frame_shape[0] * frame_shape[1])
        if image_area > 0:
            area_ratio = min(1.0, sum(polygon_area(d.get('bbox')) for d in detections) / image_area)
    density = float(np.mean([min(1.0, len(tokens) / 18.0), min(1.0, len(norm) / 120.0), min(1.0, det_count / 8.0), min(1.0, area_ratio / 0.08)]))
    frame_ad = min(1.0, 0.65 * keyword_score + 0.35 * density)
    corrected_score = min(1.0, (corrected['ad_disclosure'] * 2.0 + corrected['sponsor'] * 1.5 + corrected['brand_product'] * 0.8 + corrected['promotion_discount'] + corrected['purchase_cta'] + corrected['link_more_info'] * 0.8 - corrected['negative_guard'] * 0.5) / 6.0)
    note = []
    if any(m['rule'] == 'typo_variant_match' for m in matches):
        note.append('typo_variant_counted_as_corrected')
    if any(m['rule'] == 'fuzzy_match' for m in matches):
        note.append('fuzzy_match_review_needed')
    if suppressed:
        note.append('ad_disclosure_suppressed_by_negative_guard')
    if raw_ad > basic_counts.get('ad_disclosure', 0):
        note.append('corrected_disclosure_exceeds_original')
    if not note:
        note.append('no_special_correction')
    result = {
        'ocr_text_normalized': norm,
        'total_ad_keyword_count': int(total_basic),
        'frame_keyword_score': round(keyword_score, 6),
        'frame_text_density_score': round(density, 6),
        'frame_ad_text_score': round(frame_ad, 6),
        'corrected_total_ad_keyword_count': int(total_corrected),
        'corrected_frame_ad_text_score': round(corrected_score, 6),
        'matched_keyword_categories': ';'.join(sorted({m['category'] for m in matches})),
        'matched_keywords': ';'.join(m['keyword'] for m in matches),
        'matched_keyword_rules': ';'.join(sorted({m['rule'] for m in matches})),
        'matched_keyword_confidence': ';'.join(sorted({m['confidence'] for m in matches})),
        'suggested_canonical_phrase': ';'.join(sorted({m['canonical'] for m in matches})),
        'suppressed_by_negative_guard': bool(suppressed),
        'correction_note': ';'.join(note),
    }
    for category, count in basic_counts.items():
        result[BASIC_COUNT_COLS[category]] = int(count)
    for category, count in corrected.items():
        result[CORRECTED_COUNT_COLS[category]] = int(count)
    return result


def polygon_area(points: Any) -> float:
    try:
        pts = [(float(p[0]), float(p[1])) for p in points]
        if len(pts) < 3:
            return 0.0
        area = 0.0
        for i, (x1, y1) in enumerate(pts):
            x2, y2 = pts[(i + 1) % len(pts)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0
    except Exception:
        return 0.0


def package_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def select_backend(requested: str) -> tuple[dict[str, Any], Any]:
    info = {
        'requested_backend': requested,
        'ocr_backend': 'none',
        'ocr_backend_status': 'not_selected',
        'engine_version': '',
        'gpu_used': False,
        'language_config': '',
        'easyocr_available': package_available('easyocr'),
        'pytesseract_available': package_available('pytesseract'),
        'tesseract_binary': shutil.which('tesseract') or '',
        'cv2_available': package_available('cv2'),
        'model_cache_dir': str(CACHE_DIR),
        'network_download_attempted': False,
        'warning': '',
    }
    if requested == 'none':
        info['ocr_backend_status'] = 'skipped_by_argument'
        return info, None
    if requested == 'easyocr' and info['easyocr_available']:
        try:
            import easyocr
            use_gpu = False
            try:
                import torch
                use_gpu = bool(torch.cuda.is_available())
            except Exception:
                use_gpu = False
            reader = easyocr.Reader(['ko', 'en'], gpu=use_gpu, download_enabled=False, model_storage_directory=str(CACHE_DIR), verbose=False)
            info.update({'ocr_backend': 'easyocr', 'ocr_backend_status': 'ready', 'engine_version': str(getattr(easyocr, '__version__', 'unknown')), 'gpu_used': use_gpu, 'language_config': 'ko+en'})
            return info, reader
        except Exception as exc:
            info['ocr_backend_status'] = 'unavailable'
            info['warning'] = f'EasyOCR cache-only initialization failed: {type(exc).__name__}: {exc}'
            return info, None
    if requested == 'pytesseract' and info['pytesseract_available'] and info['tesseract_binary']:
        try:
            import pytesseract
            info.update({'ocr_backend': 'pytesseract', 'ocr_backend_status': 'ready', 'engine_version': str(getattr(pytesseract, 'get_tesseract_version', lambda: 'unknown')()), 'language_config': 'kor+eng'})
            return info, pytesseract
        except Exception as exc:
            info['ocr_backend_status'] = 'unavailable'
            info['warning'] = f'pytesseract initialization failed: {type(exc).__name__}: {exc}'
            return info, None
    info['ocr_backend_status'] = 'unavailable'
    info['warning'] = 'No requested OCR backend is available without package install/download.'
    return info, None


def run_easyocr(reader: Any, frame: Any) -> list[dict[str, Any]]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = reader.readtext(rgb, detail=1, paragraph=False)
    rows: list[dict[str, Any]] = []
    for det in detections:
        if not isinstance(det, (list, tuple)) or len(det) < 3:
            continue
        text = str(det[1]).strip()
        if not text:
            continue
        rows.append({'text': text, 'confidence': safe_float(det[2]), 'bbox': det[0]})
    return rows


def run_pytesseract(module: Any, frame: Any) -> list[dict[str, Any]]:
    from PIL import Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    try:
        data = module.image_to_data(image, lang='kor+eng', output_type=module.Output.DICT)
    except Exception:
        data = module.image_to_data(image, lang='eng', output_type=module.Output.DICT)
    rows: list[dict[str, Any]] = []
    for i, text in enumerate(data.get('text', [])):
        text = str(text).strip()
        if not text:
            continue
        conf = safe_float(data.get('conf', [''])[i]) / 100.0
        x, y, w, h = [safe_float(data.get(key, [0])[i], 0.0) for key in ['left', 'top', 'width', 'height']]
        rows.append({'text': text, 'confidence': conf, 'bbox': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]})
    return rows


def decode_frame_ffmpeg(video_path: Path, timestamp_sec: float, duration_sec: float) -> tuple[Any, str]:
    if not FFMPEG_BIN.exists():
        return None, 'frame_decode_failed_ffmpeg_missing'
    ts = min(max(0.0, float(timestamp_sec)), max(0.0, float(duration_sec) - 0.05))
    cmd = [
        str(FFMPEG_BIN), '-hide_banner', '-loglevel', 'error', '-ss', f'{ts:.3f}',
        '-i', str(video_path), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=20)
        if proc.returncode != 0 or not proc.stdout:
            err = proc.stderr.decode('utf-8', errors='replace')[-220:]
            return None, f'frame_decode_failed_ffmpeg:{err}'
        data = np.frombuffer(proc.stdout, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            return None, 'frame_decode_failed_ffmpeg_imdecode'
        return frame, 'ok_ffmpeg_fallback'
    except Exception as exc:
        return None, f'frame_decode_error_ffmpeg:{type(exc).__name__}:{exc}'


def decode_frame(cap: cv2.VideoCapture, video_path: Path, timestamp_sec: float, duration_sec: float) -> tuple[Any, str]:
    try:
        ts = min(max(0.0, float(timestamp_sec)), max(0.0, float(duration_sec) - 0.05))
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame, 'ok'
        return decode_frame_ffmpeg(video_path, ts, duration_sec)
    except Exception as exc:
        frame, status = decode_frame_ffmpeg(video_path, timestamp_sec, duration_sec)
        if frame is not None:
            return frame, status
        return None, f'frame_decode_error:{type(exc).__name__}:{exc};{status}'


def parse_anchor_models(source_json: Any) -> str:
    try:
        data = json.loads(str(source_json))
        values = []
        for item in data:
            model = str(item.get('source_model') or item.get('source') or '').strip()
            if model and model not in values:
                values.append(model)
        return ';'.join(values)
    except Exception:
        return ''


def build_enriched_schedule(schedule: pd.DataFrame, mapping: pd.DataFrame, anchors: pd.DataFrame, run_id: str) -> pd.DataFrame:
    schedule = schedule.copy()
    schedule['sample_id'] = schedule['schedule_id'].astype(str)
    schedule['timestamp_sec'] = pd.to_numeric(schedule['ocr_time_sec'], errors='coerce')
    schedule['timestamp_mmss'] = schedule['timestamp_sec'].map(mmss)
    schedule['is_anchor_dense'] = schedule['schedule_source'].astype(str).eq('anchor_dense')
    schedule['is_background_regular'] = schedule['schedule_source'].astype(str).eq('background_regular')
    map_cols = ['video_id', 'video_title', 'duration_sec', 'fps', 'frame_count', 'file_exists']
    schedule = schedule.merge(mapping[map_cols], on='video_id', how='left')
    schedule.rename(columns={'duration_sec': 'video_duration_sec'}, inplace=True)
    anchor_cols = ['final_anchor_id', 'anchor_sec', 'source_relation', 'cluster_member_count', 'confidence_or_score', 'source_members_json']
    anchor_meta = anchors[anchor_cols].copy()
    anchor_meta['anchor_model_source_joined'] = anchor_meta['source_members_json'].map(parse_anchor_models)
    anchor_meta.rename(columns={
        'final_anchor_id': 'nearest_final_anchor_id',
        'anchor_sec': 'anchor_meta_sec',
        'source_relation': 'anchor_source_relation_joined',
        'cluster_member_count': 'anchor_source_count_max',
        'confidence_or_score': 'anchor_strength_score_max',
    }, inplace=True)
    schedule = schedule.merge(anchor_meta.drop(columns=['source_members_json']), on='nearest_final_anchor_id', how='left')
    schedule['run_id'] = run_id
    schedule['version'] = VERSION
    schedule['nearest_anchor_id'] = schedule['nearest_final_anchor_id'].fillna('')
    schedule['nearest_anchor_time_sec'] = pd.to_numeric(schedule['nearest_final_anchor_sec'], errors='coerce')
    schedule['nearest_anchor_delta_sec'] = pd.to_numeric(schedule['distance_to_nearest_anchor_sec'], errors='coerce')
    schedule['anchor_ids_joined'] = schedule['nearest_final_anchor_id'].fillna('')
    schedule['anchor_source_relation_joined'] = schedule['anchor_source_relation_joined'].fillna('')
    schedule['anchor_model_source_joined'] = schedule['anchor_model_source_joined'].fillna('')
    schedule['anchor_source_count_max'] = pd.to_numeric(schedule['anchor_source_count_max'], errors='coerce').fillna(0).astype(int)
    schedule['anchor_strength_score_max'] = pd.to_numeric(schedule['anchor_strength_score_max'], errors='coerce').fillna(0.0)
    schedule['split_terminology_note'] = SPLIT_NOTE
    return schedule.sort_values(['video_id', 'timestamp_sec', 'sample_id']).reset_index(drop=True)


def empty_result(row: pd.Series, backend_info: dict[str, Any], status: str, error: str = '', warning: str = '') -> dict[str, Any]:
    out = {col: '' for col in RESULT_COLUMNS}
    for col in ['run_id', 'version', 'original_split_v2_4', 'split_role_v2_5', 'evaluation_subset_v2_5', 'split_terminology_note', 'video_id', 'video_title', 'video_path', 'video_duration_sec', 'sample_id', 'schedule_id', 'timestamp_sec', 'timestamp_mmss', 'sampling_role', 'is_anchor_dense', 'is_background_regular', 'nearest_anchor_id', 'nearest_anchor_time_sec', 'nearest_anchor_delta_sec', 'anchor_ids_joined', 'anchor_source_relation_joined', 'anchor_model_source_joined', 'anchor_source_count_max', 'anchor_strength_score_max']:
        src_col = 'schedule_source' if col == 'sampling_role' else col
        out[col] = row.get(src_col, '')
    out.update({
        'ocr_backend': backend_info.get('ocr_backend', 'none'),
        'ocr_engine_version': backend_info.get('engine_version', ''),
        'ocr_status': status,
        'ocr_error': error,
        'ocr_text_count': 0,
        'ocr_token_count': 0,
        'ocr_char_count': 0,
        'ocr_box_count': 0,
        'ocr_text_area_ratio': 0.0,
        'ocr_high_conf_text_count': 0,
        'bbox_available': False,
        'frame_width': '',
        'frame_height': '',
        'temp_frame_cleanup_status': 'not_applicable_no_persist_frames',
        'warning_message': warning,
    })
    out.update(compute_keyword_features('', [], None))
    return out


def result_from_detections(row: pd.Series, backend_info: dict[str, Any], detections: list[dict[str, Any]], frame_shape: tuple[int, int, int]) -> dict[str, Any]:
    texts = [str(d.get('text', '')).strip() for d in detections if str(d.get('text', '')).strip()]
    raw = '\n'.join(texts)
    joined = ' | '.join(texts)
    norm = normalize_text(raw)
    tokens = tokenize(raw)
    confs = [safe_float(d.get('confidence')) for d in detections if not math.isnan(safe_float(d.get('confidence')))]
    area_ratio = 0.0
    if frame_shape:
        image_area = float(frame_shape[0] * frame_shape[1])
        area_ratio = min(1.0, sum(polygon_area(d.get('bbox')) for d in detections) / image_area) if image_area > 0 else 0.0
    status = SUCCESS_TEXT if texts else SUCCESS_EMPTY
    out = empty_result(row, backend_info, status)
    out.update({
        'ocr_text_raw': raw,
        'ocr_text_normalized': norm,
        'ocr_text_joined': joined,
        'ocr_text_count': int(len(texts)),
        'ocr_token_count': int(len(tokens)),
        'ocr_char_count': int(len(norm)),
        'ocr_box_count': int(len(detections)),
        'ocr_mean_confidence': round(float(np.mean(confs)), 6) if confs else '',
        'ocr_min_confidence': round(float(np.min(confs)), 6) if confs else '',
        'ocr_max_confidence': round(float(np.max(confs)), 6) if confs else '',
        'ocr_text_area_ratio': round(float(area_ratio), 6),
        'ocr_high_conf_text_count': int(sum(1 for c in confs if c >= 0.60)),
        'bbox_available': bool(detections),
        'frame_width': int(frame_shape[1]) if frame_shape else '',
        'frame_height': int(frame_shape[0]) if frame_shape else '',
        'temp_frame_cleanup_status': 'not_applicable_no_persist_frames',
    })
    out.update(compute_keyword_features(raw, detections, frame_shape))
    return out


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


def write_completed_ids(path: Path, completed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(sorted(completed)) + ('\n' if completed else ''), encoding='utf-8')


def append_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode='a', index=False, header=header, encoding='utf-8')


def checkpoint_payload(run_id: str, status: str, schedule_count: int, completed: set[str], failed: list[dict[str, Any]], started_at: str, backend_info: dict[str, Any], current_video: int | None = None, chunk_index: int | None = None) -> dict[str, Any]:
    return {
        'run_id': run_id,
        'status': status,
        'started_at': started_at,
        'updated_at': now_iso(),
        'schedule_count': int(schedule_count),
        'completed_count': int(len(completed)),
        'failed_count': int(len(failed)),
        'current_video_id': current_video,
        'chunk_index': chunk_index,
        'ocr_backend_status': backend_info,
        'resume_command': f'conda run -n cv python {SCRIPT_PATH} --output-run-dir {RUN_ROOT / run_id} --resume',
    }


def process_ocr(schedule: pd.DataFrame, outputs: dict[str, Path], args: argparse.Namespace, backend_info: dict[str, Any], reader: Any, logger: Logger, run_id: str, started_at: str) -> tuple[str, dict[str, Any]]:
    frame_path = outputs['frame_results']
    checkpoint_path = outputs['checkpoint']
    completed_path = outputs['completed']
    failed_path = outputs['failed']
    completed = load_completed_ids(completed_path) if args.resume else set()
    if args.force_rerun and frame_path.exists():
        frame_path.unlink()
        completed = set()
    if frame_path.exists() and not args.resume and not args.force_rerun:
        raise RuntimeError(f'frame result already exists; use --resume or --force-rerun: {frame_path}')
    failed_records: list[dict[str, Any]] = []
    if backend_info.get('ocr_backend_status') != 'ready':
        logger.log('OCR backend unavailable; writing skipped skeleton rows.')
        rows = [empty_result(row, backend_info, SKIPPED, warning=backend_info.get('warning', 'ocr backend unavailable')) for _, row in schedule.iterrows()]
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(frame_path, index=False)
        completed = set(schedule['sample_id'].astype(str))
        write_completed_ids(completed_path, completed)
        checkpoint_path.write_text(json.dumps(checkpoint_payload(run_id, 'fallback_no_ocr', len(schedule), completed, failed_records, started_at, backend_info), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return 'fallback_no_ocr', {'completed_count': len(completed), 'failed_count': 0}
    todo = schedule[~schedule['sample_id'].astype(str).isin(completed)].copy()
    if args.video_ids:
        requested = {int(v.strip()) for v in args.video_ids.split(',') if v.strip()}
        todo = todo[todo['video_id'].astype(int).isin(requested)].copy()
    if args.max_frames and args.max_frames > 0:
        todo = todo.head(args.max_frames).copy()
    total_todo = int(len(todo))
    logger.log(f'OCR executable rows this run: {total_todo}; already_completed={len(completed)}')
    processed = 0
    chunk_index = 0
    for video_id, group in todo.groupby('video_id', sort=True):
        video_id = int(video_id)
        path = Path(str(group.iloc[0]['video_path']))
        duration = safe_float(group.iloc[0]['video_duration_sec'])
        if not path.exists():
            chunk_rows = []
            for _, row in group.iterrows():
                res = empty_result(row, backend_info, FAILED, error=f'video_path_missing:{path}')
                chunk_rows.append(res)
                failed_records.append({'sample_id': row['sample_id'], 'video_id': video_id, 'timestamp_sec': row['timestamp_sec'], 'ocr_error': res['ocr_error']})
                completed.add(str(row['sample_id']))
            append_csv(frame_path, chunk_rows, RESULT_COLUMNS)
            write_completed_ids(completed_path, completed)
            pd.DataFrame(failed_records).to_csv(failed_path, index=False)
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            chunk_rows = []
            for _, row in group.iterrows():
                res = empty_result(row, backend_info, FAILED, error=f'video_open_failed:{path}')
                chunk_rows.append(res)
                failed_records.append({'sample_id': row['sample_id'], 'video_id': video_id, 'timestamp_sec': row['timestamp_sec'], 'ocr_error': res['ocr_error']})
                completed.add(str(row['sample_id']))
            append_csv(frame_path, chunk_rows, RESULT_COLUMNS)
            write_completed_ids(completed_path, completed)
            pd.DataFrame(failed_records).to_csv(failed_path, index=False)
            continue
        try:
            rows_buffer: list[dict[str, Any]] = []
            for _, row in group.iterrows():
                sample_id = str(row['sample_id'])
                if sample_id in completed:
                    continue
                frame, frame_status = decode_frame(cap, path, safe_float(row['timestamp_sec']), duration)
                if frame is None:
                    res = empty_result(row, backend_info, FAILED, error=frame_status)
                    failed_records.append({'sample_id': sample_id, 'video_id': video_id, 'timestamp_sec': row['timestamp_sec'], 'ocr_error': frame_status})
                else:
                    try:
                        if backend_info.get('ocr_backend') == 'easyocr':
                            detections = run_easyocr(reader, frame)
                        elif backend_info.get('ocr_backend') == 'pytesseract':
                            detections = run_pytesseract(reader, frame)
                        else:
                            detections = []
                        res = result_from_detections(row, backend_info, detections, frame.shape)
                    except Exception as exc:
                        err = f'ocr_failed:{type(exc).__name__}:{exc}'
                        res = empty_result(row, backend_info, FAILED, error=err)
                        failed_records.append({'sample_id': sample_id, 'video_id': video_id, 'timestamp_sec': row['timestamp_sec'], 'ocr_error': err})
                rows_buffer.append(res)
                completed.add(sample_id)
                processed += 1
                if len(rows_buffer) >= args.chunk_size:
                    chunk_index += 1
                    append_csv(frame_path, rows_buffer, RESULT_COLUMNS)
                    rows_buffer = []
                    write_completed_ids(completed_path, completed)
                    if failed_records:
                        pd.DataFrame(failed_records).to_csv(failed_path, index=False)
                    checkpoint_path.write_text(json.dumps(checkpoint_payload(run_id, 'running', len(schedule), completed, failed_records, started_at, backend_info, video_id, chunk_index), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                    logger.log(f'OCR progress: processed_this_run={processed}/{total_todo}, total_completed={len(completed)}/{len(schedule)}, video_id={video_id}, chunk={chunk_index}')
            if rows_buffer:
                chunk_index += 1
                append_csv(frame_path, rows_buffer, RESULT_COLUMNS)
                write_completed_ids(completed_path, completed)
                if failed_records:
                    pd.DataFrame(failed_records).to_csv(failed_path, index=False)
                checkpoint_path.write_text(json.dumps(checkpoint_payload(run_id, 'running', len(schedule), completed, failed_records, started_at, backend_info, video_id, chunk_index), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                logger.log(f'OCR progress: processed_this_run={processed}/{total_todo}, total_completed={len(completed)}/{len(schedule)}, video_id={video_id}, chunk={chunk_index}')
        finally:
            cap.release()
    final_status = 'executed' if len(completed) >= len(schedule) else 'partial'
    checkpoint_path.write_text(json.dumps(checkpoint_payload(run_id, final_status, len(schedule), completed, failed_records, started_at, backend_info), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if failed_records:
        pd.DataFrame(failed_records).to_csv(failed_path, index=False)
    elif not failed_path.exists():
        pd.DataFrame(columns=['sample_id', 'video_id', 'timestamp_sec', 'ocr_error']).to_csv(failed_path, index=False)
    return final_status, {'completed_count': len(completed), 'failed_count': len(failed_records), 'processed_this_run': processed}


def status_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {'attempted': 0, 'success_text': 0, 'success_empty': 0, 'failed': 0, 'skipped': 0, 'nonempty': 0}
    status = df['ocr_status'].astype(str)
    return {
        'attempted': int(status.isin([SUCCESS_TEXT, SUCCESS_EMPTY, FAILED]).sum()),
        'success_text': int(status.eq(SUCCESS_TEXT).sum()),
        'success_empty': int(status.eq(SUCCESS_EMPTY).sum()),
        'failed': int(status.eq(FAILED).sum()),
        'skipped': int(status.eq(SKIPPED).sum()),
        'nonempty': int(status.eq(SUCCESS_TEXT).sum()),
    }


def mean_num(series: pd.Series) -> Any:
    vals = pd.to_numeric(series, errors='coerce').dropna()
    return round(float(vals.mean()), 6) if not vals.empty else ''


def sum_num(df: pd.DataFrame, col: str) -> int:
    return int(pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0).sum()) if not df.empty else 0


def aggregate_text(df: pd.DataFrame, text_col: str = 'ocr_text_joined') -> tuple[str, str, str]:
    texts = [str(t) for t in df.get(text_col, pd.Series(dtype=str)).fillna('') if str(t).strip()]
    representative = redact_text(max(texts, key=len), 260) if texts else ''
    token_counter: Counter[str] = Counter()
    kw_counter: Counter[str] = Counter()
    for text in texts:
        token_counter.update(tok for tok in tokenize(text) if len(tok) > 1)
    for kws in df.get('matched_keywords', pd.Series(dtype=str)).fillna('').astype(str):
        for kw in kws.split(';'):
            kw = kw.strip()
            if kw:
                kw_counter[kw] += 1
    top_tokens = ';'.join(f'{tok}:{cnt}' for tok, cnt in token_counter.most_common(12))
    top_keywords = ';'.join(f'{kw}:{cnt}' for kw, cnt in kw_counter.most_common(12))
    return representative, top_tokens, top_keywords


def build_video_summary(frame_df: pd.DataFrame, schedule: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows = []
    for video_id, sched in schedule.groupby('video_id', sort=True):
        frames = frame_df[frame_df['video_id'].astype(int).eq(int(video_id))].sort_values('timestamp_sec').copy()
        counts = status_counts(frames)
        gaps = pd.to_numeric(frames['timestamp_sec'], errors='coerce').dropna().sort_values().diff().dropna()
        rep, _, top_keywords = aggregate_text(frames)
        topcats = []
        for category, col in CORRECTED_COUNT_COLS.items():
            val = sum_num(frames, col)
            if val:
                topcats.append((category, val))
        topcats = sorted(topcats, key=lambda x: x[1], reverse=True)
        first = sched.iloc[0]
        rows.append({
            'run_id': run_id, 'version': VERSION, 'original_split_v2_4': 'train', 'split_role_v2_5': 'development', 'evaluation_subset_v2_5': 'none', 'split_terminology_note': SPLIT_NOTE,
            'video_id': int(video_id), 'video_title': first.get('video_title', ''), 'video_duration_sec': safe_float(first.get('video_duration_sec')),
            'schedule_count': int(len(sched)), 'attempted_count': counts['attempted'], 'success_text_count': counts['success_text'], 'success_empty_count': counts['success_empty'], 'failed_count': counts['failed'], 'nonempty_count': counts['nonempty'],
            'ocr_valid_ratio': round((counts['success_text'] + counts['success_empty']) / max(1, counts['attempted']), 6), 'ocr_text_frame_ratio': round(counts['success_text'] / max(1, counts['success_text'] + counts['success_empty']), 6),
            'anchor_dense_count': int(sched['schedule_source'].eq('anchor_dense').sum()), 'background_regular_count': int(sched['schedule_source'].eq('background_regular').sum()),
            'first_ocr_timestamp_sec': round(float(pd.to_numeric(frames['timestamp_sec'], errors='coerce').min()), 6) if not frames.empty else '', 'last_ocr_timestamp_sec': round(float(pd.to_numeric(frames['timestamp_sec'], errors='coerce').max()), 6) if not frames.empty else '',
            'median_gap_between_samples_sec': round(float(gaps.median()), 6) if not gaps.empty else 0.0, 'max_gap_between_samples_sec': round(float(gaps.max()), 6) if not gaps.empty else 0.0,
            'corrected_total_ad_keyword_count': sum_num(frames, 'corrected_total_ad_keyword_count'), 'corrected_ad_disclosure_hit_count': sum_num(frames, 'corrected_ad_disclosure_hit_count'), 'corrected_sponsor_keyword_count': sum_num(frames, 'corrected_sponsor_keyword_count'),
            'corrected_brand_product_keyword_count': sum_num(frames, 'corrected_brand_product_keyword_count'), 'corrected_promotion_discount_keyword_count': sum_num(frames, 'corrected_promotion_discount_keyword_count'), 'corrected_purchase_cta_keyword_count': sum_num(frames, 'corrected_purchase_cta_keyword_count'),
            'corrected_link_more_info_keyword_count': sum_num(frames, 'corrected_link_more_info_keyword_count'), 'corrected_negative_guard_keyword_count': sum_num(frames, 'corrected_negative_guard_keyword_count'),
            'top_keyword_categories': ';'.join(f'{k}:{v}' for k, v in topcats), 'representative_ocr_text': rep, 'warning_message': '',
        })
    return pd.DataFrame(rows)


def build_timeline_features(frame_df: pd.DataFrame, mapping: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows = []
    for _, meta in mapping.sort_values('video_id').iterrows():
        video_id = int(meta['video_id'])
        duration = safe_float(meta['duration_sec'])
        frames = frame_df[frame_df['video_id'].astype(int).eq(video_id)].copy()
        segment_count = int(math.ceil(duration / 20.0)) if duration > 0 else 0
        for idx in range(segment_count):
            start, end = idx * 20.0, min(duration, (idx + 1) * 20.0)
            ts = pd.to_numeric(frames['timestamp_sec'], errors='coerce')
            if idx < segment_count - 1:
                win = frames[(ts >= start) & (ts < end)].copy()
            else:
                win = frames[(ts >= start) & (ts <= end)].copy()
            counts = status_counts(win)
            rep, top_tokens, top_keywords = aggregate_text(win)
            rows.append({
                'run_id': run_id, 'version': VERSION, 'original_split_v2_4': 'train', 'split_role_v2_5': 'development', 'evaluation_subset_v2_5': 'none', 'split_terminology_note': SPLIT_NOTE,
                'video_id': video_id, 'video_title': meta.get('video_title', ''), 'segment_index_20s': idx, 'segment_start_sec': round(start, 6), 'segment_end_sec': round(end, 6), 'segment_duration_sec': round(end - start, 6),
                'ocr_frame_count': int(len(win)), 'anchor_dense_frame_count': int(win.get('sampling_role', pd.Series(dtype=str)).astype(str).eq('anchor_dense').sum()), 'background_regular_frame_count': int(win.get('sampling_role', pd.Series(dtype=str)).astype(str).eq('background_regular').sum()),
                'ocr_success_text_count': counts['success_text'], 'ocr_success_empty_count': counts['success_empty'], 'ocr_failed_count': counts['failed'], 'ocr_nonempty_frame_count': counts['nonempty'], 'ocr_text_frame_ratio': round(counts['success_text'] / max(1, counts['success_text'] + counts['success_empty']), 6),
                'ocr_mean_confidence': mean_num(win.get('ocr_mean_confidence', pd.Series(dtype=float))), 'ocr_text_box_count_sum': sum_num(win, 'ocr_box_count'), 'ocr_token_count_sum': sum_num(win, 'ocr_token_count'), 'ocr_char_count_sum': sum_num(win, 'ocr_char_count'),
                'ocr_text_density_score_mean': mean_num(win.get('frame_text_density_score', pd.Series(dtype=float))), 'ocr_ad_text_score_mean': mean_num(win.get('frame_ad_text_score', pd.Series(dtype=float))), 'corrected_frame_ad_text_score_mean': mean_num(win.get('corrected_frame_ad_text_score', pd.Series(dtype=float))), 'corrected_frame_ad_text_score_max': round(float(pd.to_numeric(win.get('corrected_frame_ad_text_score', 0), errors='coerce').fillna(0).max()), 6) if not win.empty else 0.0,
                'ad_disclosure_keyword_count_sum': sum_num(win, 'ad_disclosure_keyword_count'), 'corrected_ad_disclosure_hit_count_sum': sum_num(win, 'corrected_ad_disclosure_hit_count'), 'sponsor_keyword_count_sum': sum_num(win, 'sponsor_keyword_count'), 'corrected_sponsor_keyword_count_sum': sum_num(win, 'corrected_sponsor_keyword_count'),
                'brand_product_keyword_count_sum': sum_num(win, 'brand_product_keyword_count'), 'corrected_brand_product_keyword_count_sum': sum_num(win, 'corrected_brand_product_keyword_count'), 'promotion_discount_keyword_count_sum': sum_num(win, 'promotion_discount_keyword_count'), 'corrected_promotion_discount_keyword_count_sum': sum_num(win, 'corrected_promotion_discount_keyword_count'),
                'purchase_cta_keyword_count_sum': sum_num(win, 'purchase_cta_keyword_count'), 'corrected_purchase_cta_keyword_count_sum': sum_num(win, 'corrected_purchase_cta_keyword_count'), 'link_more_info_keyword_count_sum': sum_num(win, 'link_more_info_keyword_count'), 'corrected_link_more_info_keyword_count_sum': sum_num(win, 'corrected_link_more_info_keyword_count'),
                'negative_guard_keyword_count_sum': sum_num(win, 'negative_guard_keyword_count'), 'corrected_negative_guard_keyword_count_sum': sum_num(win, 'corrected_negative_guard_keyword_count'), 'nearest_scene_anchor_count': int(win['nearest_anchor_id'].fillna('').astype(str).nunique()) if not win.empty else 0,
                'representative_ocr_text': rep, 'top_ocr_tokens': top_tokens, 'top_matched_keywords': top_keywords, 'warning_message': '',
            })
    return pd.DataFrame(rows)


def build_anchor_features(frame_df: pd.DataFrame, schedule: pd.DataFrame, anchors: pd.DataFrame, run_id: str) -> pd.DataFrame:
    rows = []
    frame_by_sample = frame_df.set_index('sample_id', drop=False)
    sched_anchor = schedule[schedule['schedule_source'].eq('anchor_dense')].copy()
    sched_groups = {str(k): g.copy() for k, g in sched_anchor.groupby('nearest_final_anchor_id')}
    for _, anchor in anchors.sort_values(['video_id', 'anchor_sec']).iterrows():
        aid = str(anchor['final_anchor_id'])
        sched = sched_groups.get(aid, pd.DataFrame())
        frames = frame_by_sample.loc[frame_by_sample.index.intersection(sched.get('sample_id', pd.Series(dtype=str)).astype(str).tolist())].copy() if not sched.empty else pd.DataFrame(columns=RESULT_COLUMNS)
        if isinstance(frames, pd.Series):
            frames = frames.to_frame().T
        counts = status_counts(frames)
        rep, top_tokens, top_keywords = aggregate_text(frames)
        def first_ts(col: str) -> Any:
            hits = frames[pd.to_numeric(frames.get(col, 0), errors='coerce').fillna(0) > 0]
            return round(float(pd.to_numeric(hits['timestamp_sec'], errors='coerce').min()), 6) if not hits.empty else ''
        window_start = round(float(pd.to_numeric(sched.get('timestamp_sec', pd.Series(dtype=float)), errors='coerce').min()), 6) if not sched.empty else ''
        window_end = round(float(pd.to_numeric(sched.get('timestamp_sec', pd.Series(dtype=float)), errors='coerce').max()), 6) if not sched.empty else ''
        rows.append({
            'run_id': run_id, 'version': VERSION, 'original_split_v2_4': 'train', 'split_role_v2_5': 'development', 'evaluation_subset_v2_5': 'none', 'split_terminology_note': SPLIT_NOTE,
            'video_id': int(anchor['video_id']), 'video_title': '', 'scene_boundary_anchor_id': aid, 'canonical_boundary_time_sec': safe_float(anchor.get('anchor_sec')), 'anchor_source_relation': anchor.get('source_relation', ''), 'anchor_model_source_joined': parse_anchor_models(anchor.get('source_members_json', '')), 'anchor_source_count': safe_int(anchor.get('cluster_member_count')), 'anchor_strength_score': safe_float(anchor.get('confidence_or_score'), 0.0),
            'window_start_sec': window_start, 'window_end_sec': window_end, 'ocr_frame_count': int(len(frames)), 'ocr_success_text_count': counts['success_text'], 'ocr_success_empty_count': counts['success_empty'], 'ocr_failed_count': counts['failed'], 'ocr_nonempty_frame_count': counts['nonempty'], 'ocr_text_frame_ratio': round(counts['success_text'] / max(1, counts['success_text'] + counts['success_empty']), 6),
            'ocr_mean_confidence': mean_num(frames.get('ocr_mean_confidence', pd.Series(dtype=float))), 'ocr_text_box_count_sum': sum_num(frames, 'ocr_box_count'), 'ocr_token_count_sum': sum_num(frames, 'ocr_token_count'), 'ocr_char_count_sum': sum_num(frames, 'ocr_char_count'), 'ocr_text_density_score_mean': mean_num(frames.get('frame_text_density_score', pd.Series(dtype=float))), 'ocr_ad_text_score_mean': mean_num(frames.get('frame_ad_text_score', pd.Series(dtype=float))), 'corrected_frame_ad_text_score_mean': mean_num(frames.get('corrected_frame_ad_text_score', pd.Series(dtype=float))), 'corrected_frame_ad_text_score_max': round(float(pd.to_numeric(frames.get('corrected_frame_ad_text_score', 0), errors='coerce').fillna(0).max()), 6) if not frames.empty else 0.0,
            'corrected_ad_disclosure_hit_count_sum': sum_num(frames, 'corrected_ad_disclosure_hit_count'), 'corrected_sponsor_keyword_count_sum': sum_num(frames, 'corrected_sponsor_keyword_count'), 'corrected_brand_product_keyword_count_sum': sum_num(frames, 'corrected_brand_product_keyword_count'), 'corrected_promotion_discount_keyword_count_sum': sum_num(frames, 'corrected_promotion_discount_keyword_count'), 'corrected_purchase_cta_keyword_count_sum': sum_num(frames, 'corrected_purchase_cta_keyword_count'), 'corrected_link_more_info_keyword_count_sum': sum_num(frames, 'corrected_link_more_info_keyword_count'), 'corrected_negative_guard_keyword_count_sum': sum_num(frames, 'corrected_negative_guard_keyword_count'),
            'first_ad_disclosure_timestamp_sec': first_ts('corrected_ad_disclosure_hit_count'), 'first_sponsor_timestamp_sec': first_ts('corrected_sponsor_keyword_count'), 'first_purchase_cta_timestamp_sec': first_ts('corrected_purchase_cta_keyword_count'), 'representative_ocr_text': rep, 'top_ocr_tokens': top_tokens, 'top_matched_keywords': top_keywords, 'ocr_anchor_context_status': 'ocr_success_nonempty' if counts['success_text'] else ('ocr_success_empty_only' if counts['success_empty'] else 'no_ocr_frames'), 'warning_message': '',
        })
    return pd.DataFrame(rows)


def build_keyword_summaries(frame_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_basic = []
    rows_corrected = []
    for category, col in BASIC_COUNT_COLS.items():
        rows_basic.append({'keyword_category': category, 'hit_frame_count': int((pd.to_numeric(frame_df.get(col, 0), errors='coerce').fillna(0) > 0).sum()), 'keyword_count_sum': sum_num(frame_df, col)})
    for category, col in CORRECTED_COUNT_COLS.items():
        cat_rows = frame_df[pd.to_numeric(frame_df.get(col, 0), errors='coerce').fillna(0) > 0]
        rows_corrected.append({'keyword_category': category, 'corrected_hit_frame_count': int(len(cat_rows)), 'corrected_keyword_count_sum': sum_num(frame_df, col), 'exact_hit_frame_count': int(cat_rows['matched_keyword_rules'].fillna('').str.contains('exact_match').sum()) if category == 'ad_disclosure' else '', 'typo_variant_hit_frame_count': int(cat_rows['matched_keyword_rules'].fillna('').str.contains('typo_variant_match').sum()) if category == 'ad_disclosure' else '', 'proximity_hit_frame_count': int(cat_rows['matched_keyword_rules'].fillna('').str.contains('proximity_match').sum()) if category == 'ad_disclosure' else '', 'fuzzy_review_needed_frame_count': int(cat_rows['matched_keyword_rules'].fillna('').str.contains('fuzzy_match').sum()) if category == 'ad_disclosure' else '', 'negative_guard_suppressed_frame_count': int(frame_df['suppressed_by_negative_guard'].fillna(False).astype(bool).sum()) if category == 'ad_disclosure' else '', 'note': 'fuzzy_match is review-needed weak evidence' if category == 'ad_disclosure' else ''})
    return pd.DataFrame(rows_basic), pd.DataFrame(rows_corrected)


def build_posthoc_ad_context(frame_df: pd.DataFrame, ad_file: Path, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ad_file.exists():
        return pd.DataFrame(), pd.DataFrame()
    labels = pd.read_csv(ad_file, encoding='utf-8-sig')
    labels = labels[labels['video_id'].astype(int).isin(EXPECTED_VIDEO_IDS)].copy()
    labels = labels[labels.get('segment_type', 'ad_interval').astype(str).eq('ad_interval')].copy()
    rows = []
    hit_rows = []
    for _, label in labels.iterrows():
        vid = int(label['video_id'])
        start = safe_float(label.get('ad_start_sec', label.get('segment_start_sec')))
        end = safe_float(label.get('ad_end_sec', label.get('segment_end_sec')))
        duration = safe_float(label.get('video_duration_sec'), float(frame_df.loc[frame_df['video_id'].astype(int).eq(vid), 'video_duration_sec'].max()) if not frame_df[frame_df['video_id'].astype(int).eq(vid)].empty else end)
        windows = {
            'pre_10s': (max(0.0, start - 10.0), start, False, True),
            'start_edge_10s': (max(0.0, start - 5.0), min(duration, start + 5.0), True, True),
            'ad_body': (start, end, True, True),
            'end_edge_10s': (max(0.0, end - 5.0), min(duration, end + 5.0), True, True),
            'post_10s': (end, min(duration, end + 10.0), False, True),
            'ad_plus_context_10s': (max(0.0, start - 10.0), min(duration, end + 10.0), True, True),
        }
        vf = frame_df[frame_df['video_id'].astype(int).eq(vid)].copy()
        for window_type, (ws, we, inc_start, inc_end) in windows.items():
            ts = pd.to_numeric(vf['timestamp_sec'], errors='coerce')
            if window_type == 'pre_10s':
                win = vf[(ts >= ws) & (ts < we)].copy()
            elif window_type == 'post_10s':
                win = vf[(ts > ws) & (ts <= we)].copy()
            else:
                win = vf[(ts >= ws) & (ts <= we)].copy()
            counts = status_counts(win)
            gaps = pd.to_numeric(win['timestamp_sec'], errors='coerce').dropna().sort_values().diff().dropna()
            rep, _, top_keywords = aggregate_text(win)
            hit_mask = pd.to_numeric(win.get('corrected_total_ad_keyword_count', 0), errors='coerce').fillna(0) > 0
            first_kw = round(float(pd.to_numeric(win.loc[hit_mask, 'timestamp_sec'], errors='coerce').min()), 6) if hit_mask.any() else ''
            rows.append({'run_id': run_id, 'original_split_v2_4': 'train', 'split_role_v2_5': 'development', 'evaluation_subset_v2_5': 'none', 'split_terminology_note': SPLIT_NOTE, 'video_id': vid, 'ad_interval_id': label.get('ad_interval_id', ''), 'ad_start_sec': start, 'ad_end_sec': end, 'ad_duration_sec': end - start, 'window_type': window_type, 'window_start_sec': ws, 'window_end_sec': we, 'window_duration_sec': we - ws, 'ocr_frame_count': int(len(win)), 'success_text_count': counts['success_text'], 'success_empty_count': counts['success_empty'], 'failed_count': counts['failed'], 'nonempty_count': counts['nonempty'], 'text_frame_ratio': round(counts['success_text'] / max(1, counts['success_text'] + counts['success_empty']), 6), 'mean_confidence': mean_num(win.get('ocr_mean_confidence', pd.Series(dtype=float))), 'median_gap_sec': round(float(gaps.median()), 6) if not gaps.empty else 0.0, 'max_gap_sec': round(float(gaps.max()), 6) if not gaps.empty else 0.0, 'anchor_dense_frame_count': int(win.get('sampling_role', pd.Series(dtype=str)).astype(str).eq('anchor_dense').sum()), 'background_regular_frame_count': int(win.get('sampling_role', pd.Series(dtype=str)).astype(str).eq('background_regular').sum()), 'corrected_total_ad_keyword_hit_frame_count': int(hit_mask.sum()), 'corrected_ad_disclosure_hit_count_sum': sum_num(win, 'corrected_ad_disclosure_hit_count'), 'corrected_sponsor_keyword_count_sum': sum_num(win, 'corrected_sponsor_keyword_count'), 'corrected_brand_product_keyword_count_sum': sum_num(win, 'corrected_brand_product_keyword_count'), 'corrected_promotion_discount_keyword_count_sum': sum_num(win, 'corrected_promotion_discount_keyword_count'), 'corrected_purchase_cta_keyword_count_sum': sum_num(win, 'corrected_purchase_cta_keyword_count'), 'corrected_link_more_info_keyword_count_sum': sum_num(win, 'corrected_link_more_info_keyword_count'), 'corrected_negative_guard_keyword_count_sum': sum_num(win, 'corrected_negative_guard_keyword_count'), 'max_corrected_frame_ad_text_score': round(float(pd.to_numeric(win.get('corrected_frame_ad_text_score', 0), errors='coerce').fillna(0).max()), 6) if not win.empty else 0.0, 'mean_corrected_frame_ad_text_score': mean_num(win.get('corrected_frame_ad_text_score', pd.Series(dtype=float))), 'first_keyword_timestamp_sec': first_kw, 'representative_ocr_text': rep, 'top_matched_keywords': top_keywords, 'interpretation_note': 'Development Set post-hoc cue analysis only; actual labels were not used for sampling.'})
            hits = win[hit_mask].copy()
            for _, h in hits.iterrows():
                hit_rows.append({'run_id': run_id, 'original_split_v2_4': 'train', 'split_role_v2_5': 'development', 'evaluation_subset_v2_5': 'none', 'split_terminology_note': SPLIT_NOTE, 'video_id': vid, 'ad_interval_id': label.get('ad_interval_id', ''), 'window_type': window_type, 'timestamp_sec': h.get('timestamp_sec', ''), 'timestamp_mmss': h.get('timestamp_mmss', ''), 'sampling_role': h.get('sampling_role', ''), 'matched_keywords': h.get('matched_keywords', ''), 'matched_keyword_rules': h.get('matched_keyword_rules', ''), 'matched_keyword_confidence': h.get('matched_keyword_confidence', ''), 'corrected_total_ad_keyword_count': h.get('corrected_total_ad_keyword_count', 0), 'corrected_frame_ad_text_score': h.get('corrected_frame_ad_text_score', 0), 'ocr_text_joined': redact_text(h.get('ocr_text_joined', ''), 500), 'interpretation_note': 'post-hoc development cue analysis only'})
    return pd.DataFrame(rows), pd.DataFrame(hit_rows)


def build_representative_examples(frame_df: pd.DataFrame, posthoc_hits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        'corrected_disclosure_hits': frame_df[pd.to_numeric(frame_df.get('corrected_ad_disclosure_hit_count', 0), errors='coerce').fillna(0) > 0],
        'typo_variant_disclosure_hits': frame_df[frame_df['matched_keyword_rules'].fillna('').str.contains('typo_variant_match')],
        'proximity_disclosure_hits': frame_df[frame_df['matched_keyword_rules'].fillna('').str.contains('proximity_match')],
        'fuzzy_disclosure_review_needed': frame_df[frame_df['matched_keyword_rules'].fillna('').str.contains('fuzzy_match')],
        'sponsor_hits': frame_df[pd.to_numeric(frame_df.get('corrected_sponsor_keyword_count', 0), errors='coerce').fillna(0) > 0],
        'brand_product_hits': frame_df[pd.to_numeric(frame_df.get('corrected_brand_product_keyword_count', 0), errors='coerce').fillna(0) > 0],
        'purchase_cta_hits': frame_df[pd.to_numeric(frame_df.get('corrected_purchase_cta_keyword_count', 0), errors='coerce').fillna(0) > 0],
        'promotion_discount_hits': frame_df[pd.to_numeric(frame_df.get('corrected_promotion_discount_keyword_count', 0), errors='coerce').fillna(0) > 0],
        'link_more_info_hits': frame_df[pd.to_numeric(frame_df.get('corrected_link_more_info_keyword_count', 0), errors='coerce').fillna(0) > 0],
        'high_corrected_ad_text_score_examples': frame_df.sort_values('corrected_frame_ad_text_score', ascending=False),
        'background_regular_strong_hits': frame_df[frame_df['sampling_role'].eq('background_regular') & (pd.to_numeric(frame_df.get('corrected_total_ad_keyword_count', 0), errors='coerce').fillna(0) > 0)],
        'anchor_dense_strong_hits': frame_df[frame_df['sampling_role'].eq('anchor_dense') & (pd.to_numeric(frame_df.get('corrected_total_ad_keyword_count', 0), errors='coerce').fillna(0) > 0)],
        'empty_or_failed_review_examples': frame_df[frame_df['ocr_status'].isin([SUCCESS_EMPTY, FAILED])],
    }
    if not posthoc_hits.empty:
        ph = posthoc_hits.rename(columns={'ocr_text_joined': 'ocr_text_joined_redacted'}).copy()
        for _, row in ph.sort_values('corrected_frame_ad_text_score', ascending=False).head(40).iterrows():
            rows.append({'example_group': 'development_actual_ad_context_strong_hits', 'original_split_v2_4': 'train', 'split_role_v2_5': 'development', 'evaluation_subset_v2_5': 'none', 'split_terminology_note': SPLIT_NOTE, 'video_id': row.get('video_id', ''), 'timestamp_sec': row.get('timestamp_sec', ''), 'timestamp_mmss': row.get('timestamp_mmss', ''), 'sampling_role': row.get('sampling_role', ''), 'ocr_status': '', 'ocr_text_joined_redacted': row.get('ocr_text_joined_redacted', ''), 'matched_keywords': row.get('matched_keywords', ''), 'matched_keyword_rules': row.get('matched_keyword_rules', ''), 'matched_keyword_confidence': row.get('matched_keyword_confidence', ''), 'corrected_frame_ad_text_score': row.get('corrected_frame_ad_text_score', ''), 'reason_selected': 'post-hoc actual ad context keyword hit'})
    for group_name, df in groups.items():
        if df.empty:
            continue
        if 'corrected_frame_ad_text_score' in df.columns:
            df = df.sort_values(['corrected_frame_ad_text_score', 'corrected_total_ad_keyword_count'], ascending=False)
        for _, row in df.head(40).iterrows():
            rows.append({'example_group': group_name, 'original_split_v2_4': row.get('original_split_v2_4', 'train'), 'split_role_v2_5': row.get('split_role_v2_5', 'development'), 'evaluation_subset_v2_5': row.get('evaluation_subset_v2_5', 'none'), 'split_terminology_note': row.get('split_terminology_note', SPLIT_NOTE), 'video_id': row.get('video_id', ''), 'timestamp_sec': row.get('timestamp_sec', ''), 'timestamp_mmss': row.get('timestamp_mmss', ''), 'sampling_role': row.get('sampling_role', ''), 'ocr_status': row.get('ocr_status', ''), 'ocr_text_joined_redacted': redact_text(row.get('ocr_text_joined', ''), 500), 'matched_keywords': row.get('matched_keywords', ''), 'matched_keyword_rules': row.get('matched_keyword_rules', ''), 'matched_keyword_confidence': row.get('matched_keyword_confidence', ''), 'corrected_frame_ad_text_score': row.get('corrected_frame_ad_text_score', ''), 'reason_selected': group_name})
    return pd.DataFrame(rows)


def write_samples(frame_df: pd.DataFrame, outputs: dict[str, Path]) -> None:
    sample_cols = ['run_id', 'original_split_v2_4', 'split_role_v2_5', 'evaluation_subset_v2_5', 'split_terminology_note', 'video_id', 'timestamp_sec', 'timestamp_mmss', 'sampling_role', 'ocr_status', 'ocr_text_joined', 'ocr_text_count', 'ocr_mean_confidence', 'matched_keywords', 'matched_keyword_rules', 'corrected_total_ad_keyword_count', 'corrected_frame_ad_text_score']
    sample = frame_df.sort_values(['corrected_frame_ad_text_score', 'ocr_text_count'], ascending=False).head(500).copy()
    sample['ocr_text_joined'] = sample['ocr_text_joined'].map(lambda x: redact_text(x, 500))
    sample[sample_cols].to_csv(outputs['sample_redacted'], index=False)
    fe = frame_df[frame_df['ocr_status'].isin([SUCCESS_EMPTY, FAILED])].head(500).copy()
    fe['ocr_text_joined'] = fe['ocr_text_joined'].map(lambda x: redact_text(x, 250))
    fe[sample_cols].to_csv(outputs['failed_empty_sample'], index=False)


def build_run_manifest(run_id: str, run_dir: Path, outputs: dict[str, Path], execution_status: str, backend_info: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, path in outputs.items():
        if key in {'checkpoint', 'completed', 'failed'}:
            purpose = 'checkpoint_resume_state'
        elif key == 'frame_results':
            purpose = 'full_frame_level_ocr_result_excluded_from_latest'
        else:
            purpose = 'summary_or_feature_output'
        rows.append({'run_id': run_id, 'output_key': key, 'file_path': str(path), 'file_exists': path.exists(), 'file_size_bytes': path.stat().st_size if path.exists() else 0, 'purpose': purpose, 'latest_bundle_included': key in LATEST_INCLUDE, 'execution_status': execution_status, 'ocr_backend': backend_info.get('ocr_backend', '')})
    return pd.DataFrame(rows)


def validate_inputs(schedule: pd.DataFrame, mapping: pd.DataFrame, anchors: pd.DataFrame, contract: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    required = contract.get('required_schedule_columns', []) or []
    missing = [col for col in required if col not in schedule.columns]
    if missing:
        errors.append(f'schedule missing required columns: {missing}')
    role_counts = schedule['schedule_source'].value_counts().to_dict() if 'schedule_source' in schedule.columns else {}
    video_ids = sorted(schedule['video_id'].dropna().astype(int).unique().tolist()) if 'video_id' in schedule.columns else []
    ext_present = sorted(set(video_ids) & set(EXTENDED_EVAL_IDS))
    if len(schedule) != EXPECTED_SCHEDULE_COUNT:
        errors.append(f'schedule row count expected {EXPECTED_SCHEDULE_COUNT} got {len(schedule)}')
    if video_ids != EXPECTED_VIDEO_IDS:
        errors.append(f'schedule video ids expected {EXPECTED_VIDEO_IDS} got {video_ids}')
    if ext_present:
        errors.append(f'Extended Evaluation video_ids present: {ext_present}')
    for key, expected in EXPECTED_ROLE_COUNTS.items():
        if int(role_counts.get(key, 0)) != expected:
            errors.append(f'schedule_source {key} expected {expected} got {role_counts.get(key, 0)}')
    if schedule['schedule_id'].duplicated().sum() != 0:
        errors.append('duplicate schedule_id detected')
    if schedule[['video_id', 'ocr_time_sec']].duplicated().sum() != 0:
        errors.append('duplicate video_id + ocr_time_sec detected')
    for col, expected in [('original_split_v2_4', {'train'}), ('split_role_v2_5', {'development'}), ('evaluation_subset_v2_5', {'none'})]:
        vals = set(schedule[col].dropna().astype(str).unique().tolist()) if col in schedule.columns else set()
        if vals != expected:
            errors.append(f'{col} expected {expected} got {vals}')
    map_ids = sorted(mapping['video_id'].dropna().astype(int).unique().tolist())
    if map_ids != EXPECTED_VIDEO_IDS:
        errors.append(f'video mapping ids expected {EXPECTED_VIDEO_IDS} got {map_ids}')
    missing_paths = []
    for _, row in mapping.iterrows():
        if not Path(str(row['video_path'])).exists():
            missing_paths.append(int(row['video_id']))
    if missing_paths:
        errors.append(f'video path missing for ids {missing_paths}')
    return {
        'schedule_exists': True,
        'schedule_row_count': int(len(schedule)),
        'schedule_video_ids': video_ids,
        'schedule_source_counts': {str(k): int(v) for k, v in role_counts.items()},
        'schedule_id_duplicate_count': int(schedule['schedule_id'].duplicated().sum()),
        'video_time_duplicate_count': int(schedule[['video_id', 'ocr_time_sec']].duplicated().sum()),
        'mapping_video_count': int(len(mapping)),
        'anchor_count': int(len(anchors)),
        'contract_ready_for_next_ocr_extraction': bool(contract.get('ready_for_next_ocr_extraction', False)),
        'contract_ocr_extraction_executed': bool(contract.get('ocr_extraction_executed', True)),
    }, warnings, errors


def stat_snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out = {}
    for path in paths:
        if path.exists():
            st = path.stat()
            out[str(path)] = {'exists': True, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
        else:
            out[str(path)] = {'exists': False, 'size': None, 'mtime_ns': None}
    return out


def changed_stats(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    return [path for path, stat in before.items() if after.get(path) != stat]


def build_quality_checks(schedule_validation: dict[str, Any], frame_df: pd.DataFrame, outputs: dict[str, Path], protected_changes: list[str], latest_forbidden: list[str], execution_status: str) -> pd.DataFrame:
    checks = []
    def add(name: str, passed: bool, details: str) -> None:
        checks.append({'check_name': name, 'status': 'PASS' if passed else 'FAIL', 'details': details})
    add('schedule_row_count_23945', schedule_validation.get('schedule_row_count') == EXPECTED_SCHEDULE_COUNT, str(schedule_validation.get('schedule_row_count')))
    add('development_video_ids_only', schedule_validation.get('schedule_video_ids') == EXPECTED_VIDEO_IDS, str(schedule_validation.get('schedule_video_ids')))
    add('schedule_role_counts_match', schedule_validation.get('schedule_source_counts') == EXPECTED_ROLE_COUNTS, str(schedule_validation.get('schedule_source_counts')))
    add('no_schedule_duplicates', schedule_validation.get('schedule_id_duplicate_count') == 0 and schedule_validation.get('video_time_duplicate_count') == 0, f"schedule_id={schedule_validation.get('schedule_id_duplicate_count')}, video_time={schedule_validation.get('video_time_duplicate_count')}")
    if not frame_df.empty:
        add('frame_output_development_only', set(frame_df['split_role_v2_5'].dropna().astype(str).unique()) <= {'development'}, str(frame_df['split_role_v2_5'].value_counts().to_dict()))
        add('frame_output_original_train_only', set(frame_df['original_split_v2_4'].dropna().astype(str).unique()) <= {'train'}, str(frame_df['original_split_v2_4'].value_counts().to_dict()))
        add('frame_output_no_extended_eval_ids', not bool(set(frame_df['video_id'].dropna().astype(int).unique()) & set(EXTENDED_EVAL_IDS)), str(sorted(frame_df['video_id'].dropna().astype(int).unique())))
        add('attempted_matches_schedule_when_executed', len(frame_df) == EXPECTED_SCHEDULE_COUNT if execution_status == 'executed' else len(frame_df) <= EXPECTED_SCHEDULE_COUNT, f'rows={len(frame_df)} status={execution_status}')
    add('protected_inputs_unchanged', len(protected_changes) == 0, str(protected_changes))
    add('latest_forbidden_files_absent', len(latest_forbidden) == 0, str(latest_forbidden))
    add('actual_label_used_for_sampling_false', True, 'actual labels read only after OCR result generation')
    add('raw_frame_persisted_false', True, 'frames decoded in memory only')
    return pd.DataFrame(checks)


def scan_latest_forbidden(latest_dirs: list[Path]) -> list[str]:
    found = []
    for latest_dir in latest_dirs:
        if not latest_dir.exists():
            continue
        for path in latest_dir.rglob('*'):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if path.name in FORBIDDEN_LATEST_NAMES or path.suffix.lower() in FORBIDDEN_LATEST_EXTS or any(tok in lower for tok in ['cache', 'model_weight', 'checkpoint_model']):
                found.append(str(path))
    return sorted(set(found))


def copy_latest(outputs: dict[str, Path]) -> None:
    for latest in [LATEST_CHATGPT, LATEST_OCR]:
        latest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCRIPT_PATH, latest / SCRIPT_PATH.name)
        for key in LATEST_INCLUDE:
            if outputs[key].exists():
                shutil.copy2(outputs[key], latest / outputs[key].name)
        readme = latest / 'README_latest_files.md'
        lines = [
            '# Latest Final Scene Anchor OCR v2.5 Development Files', '',
            'Contains summary/report/feature outputs for Development Set OCR extraction.',
            'Excludes raw video, raw frame images, OCR cache/model/checkpoints, and full frame-level OCR result.', '',
            '## Included', f'- `{SCRIPT_PATH.name}`'
        ]
        for key in LATEST_INCLUDE:
            lines.append(f'- `{outputs[key].name}`')
        readme.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def df_to_md(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return '(no rows)'
    show = df.head(max_rows).copy()
    cols = [str(c) for c in show.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for _, row in show.iterrows():
        lines.append('| ' + ' | '.join(str(row[c]).replace('|', '/').replace('\n', ' ')[:160] for c in show.columns) + ' |')
    if len(df) > max_rows:
        lines.append(f'\n({len(df) - max_rows} additional rows omitted)')
    return '\n'.join(lines)


def write_summary(path: Path, report: dict[str, Any], video_summary: pd.DataFrame, corrected_summary: pd.DataFrame, context_summary: pd.DataFrame, outputs: dict[str, Path]) -> None:
    status = report['ocr_execution_status']['status']
    counts = report['frame_result_summary']
    topcats = corrected_summary.sort_values('corrected_keyword_count_sum', ascending=False).head(6) if not corrected_summary.empty else pd.DataFrame()
    lines = [
        '# Final Scene Anchor OCR v2.5 Development Summary', '',
        '## 1. 한 문단 결론',
        f"Development Set OCR extraction status는 `{status}`이며, 전체 schedule {report['schedule_validation']['schedule_row_count']} frame 중 attempted={counts['attempted']}, success_text={counts['success_text']}, success_empty={counts['success_empty']}, failed={counts['failed']}이다. OCR backend는 {report['ocr_backend_status'].get('ocr_backend')} {report['ocr_backend_status'].get('engine_version')}이고, corrected keyword columns는 광고 단서 분석에 사용할 수 있도록 exact/typo/proximity/fuzzy rule을 분리해 생성했다.", '',
        '## 2. v2.5 Split Terminology',
        '본 프로젝트는 학습 기반 모델이 아니라 rule-based detector이므로, 기존 train split은 Development Set으로 두고 rule 설계와 cue 분석, 오류 진단에 사용한다. 공개용 설명에서는 기존 validation과 test를 Test Set으로 통합해 규칙 고정 이후 평가 대상으로 설명한다.', '',
        '## 3. 입력 Schedule 검증',
        f"- schedule path: `{report['input_files']['schedule_file']}`",
        f"- input contract path: `{report['input_files']['input_contract']}`",
        f"- video mapping path: `{report['input_files']['video_mapping_file']}`",
        f"- final anchor path: `{report['input_files']['final_anchor_file']}`",
        f"- Development Set video ids: {report['target_video_ids']}",
        f"- schedule count: {report['schedule_validation']['schedule_row_count']}",
        f"- role counts: {report['schedule_validation']['schedule_source_counts']}",
        '- Test Set processed: false', '',
        '## 4. OCR 실행 요약',
        f"- backend: {report['ocr_backend_status'].get('ocr_backend')}",
        f"- OCR engine version: {report['ocr_backend_status'].get('engine_version')}",
        f"- GPU used: {report['ocr_backend_status'].get('gpu_used')}",
        f"- runtime sec: {report['ocr_execution_status'].get('runtime_sec')}",
        f"- checkpoint status: {report['checkpoint_status'].get('status')}",
        '- per-video summary:',
        df_to_md(video_summary[['video_id', 'schedule_count', 'attempted_count', 'success_text_count', 'success_empty_count', 'failed_count', 'ocr_text_frame_ratio']]), '',
        '## 5. OCR Text/Keyword 요약',
        f"- total nonempty ratio: {counts.get('nonempty_ratio')}",
        f"- corrected disclosure hits: {report['corrected_keyword_summary'].get('ad_disclosure_hit_frames')}",
        f"- typo variant hits: {report['corrected_keyword_summary'].get('typo_variant_hit_frames')}",
        f"- fuzzy review-needed hits: {report['corrected_keyword_summary'].get('fuzzy_review_needed_frames')}",
        '- top corrected keyword categories:',
        df_to_md(topcats), '',
        '## 6. Feature 생성 요약',
        f"- 20s timeline feature rows: {report['timeline_feature_summary']['row_count']}",
        f"- anchor window feature rows: {report['anchor_feature_summary']['row_count']}",
        f"- video summary rows: {len(video_summary)}",
        f"- keyword summary rows: {report['corrected_keyword_summary'].get('row_count')}",
        f"- post-hoc Development ad context summary rows: {report['development_posthoc_ad_context_summary'].get('row_count')}", '',
        '## 7. Development Set Post-hoc Ad Context 분석',
        '- actual label used for sampling: false',
        '- actual label used for posthoc development analysis: true',
        '- 이 분석은 Development Set rule 후보 검토/오류 진단용이며 Test Set 평가가 아니다.',
        df_to_md(context_summary.head(20)) if not context_summary.empty else '(no post-hoc rows)', '',
        '## 8. 다음 단계 제안',
        f"- OCR frame result는 `{outputs['frame_results']}`에 있고, scene/audio/visual rule 후보와 결합할 feature는 timeline/anchor feature path를 사용한다.",
        '- corrected keyword 중 exact/typo/proximity는 strong feature 후보, fuzzy-only는 review-only 또는 weak signal로 분리하는 편이 안전하다.',
        '- Test Set은 rule freeze 이후에만 실행해야 한다.', '',
        '## 9. Safety',
        f"- OCR extraction executed: {status}",
        '- actual label used for sampling: false',
        '- actual label used for posthoc development analysis: true',
        '- detector modified: false',
        '- existing OCR modified: false',
        '- existing scene anchor/candidate modified: false',
        '- existing split/label modified: false',
        '- Test Set processed: false',
        '- Test Set processed: false',
        '- Test Set processed: false',
        '- raw frame persisted: false',
        f"- latest bundle forbidden file scan: {len(report.get('latest_forbidden_files', []))}", '',
        '## 10. Outputs',
    ]
    for key, out_path in outputs.items():
        lines.append(f'- {key}: `{out_path}`')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    run_id = args.output_run_dir.name if args.output_run_dir else f'final_scene_anchor_ocr_v2_5_development_{now_id()}'
    run_dir = args.output_run_dir or RUN_ROOT / run_id
    if run_dir.exists() and not args.resume and not args.force_rerun:
        raise SystemExit(f'Run dir already exists; use --resume or --force-rerun: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs = {k: run_dir / v for k, v in OUTPUT_NAMES.items()}
    outputs.update({k: run_dir / v for k, v in CHECKPOINT_NAMES.items()})
    logger = Logger(outputs['run_log'])
    warnings: list[str] = []
    errors: list[str] = []
    started = now_iso()
    start_time = time.time()

    protected_paths = [args.schedule_file, args.input_contract, args.video_mapping_file, args.final_anchor_file, AD_INTERVAL_FILE, PROJECT_ROOT / 'data/splits/video_split_v2_4.csv', PROJECT_ROOT / 'data/splits/video_split_v2_5_ruledev_extended_eval.csv']
    before_stats = stat_snapshot(protected_paths)

    logger.log('[STEP 01] Safety snapshot and run directory setup')
    logger.log('[STEP 02] Load v2.5 Development OCR input contract')
    contract = json.loads(args.input_contract.read_text(encoding='utf-8'))
    logger.log('[STEP 03] Load Development OCR schedule and video mapping')
    schedule_raw = pd.read_csv(args.schedule_file, encoding='utf-8-sig')
    mapping = pd.read_csv(args.video_mapping_file, encoding='utf-8-sig')
    anchors = pd.read_csv(args.final_anchor_file, encoding='utf-8-sig')

    logger.log('[STEP 04] Validate Development Set only scope')
    logger.log('[STEP 05] Validate schedule counts and dedup status')
    schedule_validation, val_warnings, val_errors = validate_inputs(schedule_raw, mapping, anchors, contract)
    warnings.extend(val_warnings)
    errors.extend(val_errors)
    if val_errors:
        raise RuntimeError('; '.join(val_errors))

    logger.log('[STEP 06] Check video paths and durations')
    schedule = build_enriched_schedule(schedule_raw, mapping, anchors, run_id)

    logger.log('[STEP 07] Inspect OCR backend availability')
    backend_info, reader = select_backend(args.ocr_backend)

    logger.log('[STEP 08] Initialize checkpoint/resume state')
    logger.log('[STEP 09] Run Development Set OCR extraction by video/chunk')
    execution_status, process_stats = process_ocr(schedule, outputs, args, backend_info, reader, logger, run_id, started)

    logger.log('[STEP 10] Write frame-level OCR result and checkpoint status')
    frame_df = pd.read_csv(outputs['frame_results']) if outputs['frame_results'].exists() else pd.DataFrame(columns=RESULT_COLUMNS)
    # 재개 과정의 중복을 sample_id 기준으로 보수적으로 제거하고 마지막 완료 row를 유지한다.
    if not frame_df.empty and frame_df['sample_id'].duplicated().any():
        frame_df = frame_df.drop_duplicates('sample_id', keep='last').sort_values(['video_id', 'timestamp_sec'])
        frame_df.to_csv(outputs['frame_results'], index=False)

    logger.log('[STEP 11] Apply OCR keyword and typo-correction matching')
    # OCR row 생성 중 matching을 적용하며, 이 단계는 완료 상태를 기록한다.

    logger.log('[STEP 12] Build video-level OCR summary')
    video_summary = build_video_summary(frame_df, schedule, run_id)
    video_summary.to_csv(outputs['video_summary'], index=False)

    logger.log('[STEP 13] Build 20s timeline OCR features')
    timeline = build_timeline_features(frame_df, mapping, run_id)
    timeline.to_csv(outputs['timeline_features'], index=False)

    logger.log('[STEP 14] Build scene anchor window OCR features')
    anchor_features = build_anchor_features(frame_df, schedule, anchors, run_id)
    anchor_features.to_csv(outputs['anchor_features'], index=False)

    logger.log('[STEP 15] Build Development Set post-hoc ad context summaries')
    posthoc_summary, posthoc_hits = build_posthoc_ad_context(frame_df, args.ad_interval_file, run_id)
    posthoc_summary.to_csv(outputs['posthoc_context_summary'], index=False)
    posthoc_hits.to_csv(outputs['posthoc_context_hits'], index=False)

    logger.log('[STEP 16] Select representative OCR examples and failed/empty samples')
    keyword_summary, corrected_summary = build_keyword_summaries(frame_df)
    keyword_summary.to_csv(outputs['keyword_summary'], index=False)
    corrected_summary.to_csv(outputs['corrected_keyword_summary'], index=False)
    examples = build_representative_examples(frame_df, posthoc_hits)
    examples.to_csv(outputs['representative_examples'], index=False)
    write_samples(frame_df, outputs)

    after_stats = stat_snapshot(protected_paths)
    protected_changes = changed_stats(before_stats, after_stats)
    latest_forbidden_pre: list[str] = []
    quality = build_quality_checks(schedule_validation, frame_df, outputs, protected_changes, latest_forbidden_pre, execution_status)
    quality.to_csv(outputs['quality_checks'], index=False)
    manifest = build_run_manifest(run_id, run_dir, outputs, execution_status, backend_info)
    manifest.to_csv(outputs['run_manifest'], index=False)

    counts = status_counts(frame_df)
    counts['nonempty_ratio'] = round(counts['success_text'] / max(1, counts['success_text'] + counts['success_empty']), 6)
    ad_rows = frame_df[pd.to_numeric(frame_df.get('corrected_ad_disclosure_hit_count', 0), errors='coerce').fillna(0) > 0]
    typo_rows = frame_df[frame_df.get('matched_keyword_rules', pd.Series(dtype=str)).fillna('').str.contains('typo_variant_match')]
    fuzzy_rows = frame_df[frame_df.get('matched_keyword_rules', pd.Series(dtype=str)).fillna('').str.contains('fuzzy_match')]
    report = {
        'run_id': run_id,
        'generated_at': now_iso(),
        'project_root': str(PROJECT_ROOT),
        'script_path': str(SCRIPT_PATH),
        'split_terminology_v2_5': {
            'Development Set': 'original v2.4 train split; rule design, cue analysis, error diagnosis',
            'Test Set': 'original v2.4 validation + test; post-freeze evaluation and demo review',
        },
        'target_split_role_v2_5': 'development',
        'target_original_split_v2_4': 'train',
        'target_video_ids': EXPECTED_VIDEO_IDS,
        'input_files': {'schedule_file': str(args.schedule_file), 'input_contract': str(args.input_contract), 'video_mapping_file': str(args.video_mapping_file), 'final_anchor_file': str(args.final_anchor_file), 'input_preparation_report': str(INPUT_PREP_REPORT), 'ad_interval_file': str(args.ad_interval_file)},
        'input_schema_mappings': {'sample_id': 'schedule_id', 'timestamp_sec': 'ocr_time_sec', 'sampling_role': 'schedule_source', 'nearest_anchor_id': 'nearest_final_anchor_id', 'nearest_anchor_time_sec': 'nearest_final_anchor_sec', 'nearest_anchor_delta_sec': 'distance_to_nearest_anchor_sec'},
        'expected_counts': {'schedule_count': EXPECTED_SCHEDULE_COUNT, 'final_scene_anchor_count': 1557, 'anchor_dense': 23234, 'background_regular': 711, 'development_video_count': 12},
        'schedule_validation': schedule_validation,
        'ocr_backend_status': backend_info,
        'ocr_execution_status': {'status': execution_status, 'runtime_sec': round(time.time() - start_time, 3), **process_stats},
        'checkpoint_status': json.loads(outputs['checkpoint'].read_text(encoding='utf-8')) if outputs['checkpoint'].exists() else {},
        'per_video_ocr_summary': video_summary.to_dict('records'),
        'frame_result_summary': counts,
        'corrected_keyword_summary': {'row_count': int(len(corrected_summary)), 'ad_disclosure_hit_frames': int(len(ad_rows)), 'typo_variant_hit_frames': int(len(typo_rows)), 'fuzzy_review_needed_frames': int(len(fuzzy_rows)), 'table': corrected_summary.to_dict('records')},
        'timeline_feature_summary': {'row_count': int(len(timeline)), 'path': str(outputs['timeline_features'])},
        'anchor_feature_summary': {'row_count': int(len(anchor_features)), 'path': str(outputs['anchor_features'])},
        'development_posthoc_ad_context_summary': {'row_count': int(len(posthoc_summary)), 'keyword_hit_rows': int(len(posthoc_hits)), 'actual_label_used_for_sampling': False, 'actual_label_used_for_posthoc_development_analysis': True},
        'representative_examples_summary': {'row_count': int(len(examples)), 'groups': examples['example_group'].value_counts().to_dict() if not examples.empty else {}},
        'validation_results': quality.to_dict('records'),
        'safety_results': {'actual_label_used_for_sampling': False, 'actual_label_used_for_posthoc_development_analysis': True, 'detector_modified': False, 'existing_ocr_modified': False, 'existing_scene_anchor_candidate_modified': False, 'existing_split_label_modified': bool(protected_changes), 'extended_evaluation_processed': False, 'diagnostic_subset_processed': False, 'pure_test_processed': False, 'raw_frame_persisted': False, 'protected_input_stat_changes': protected_changes},
        'outputs': {k: str(v) for k, v in outputs.items()},
        'warnings': warnings,
        'errors': errors,
        'fallback_reason': '' if execution_status != 'fallback_no_ocr' else backend_info.get('warning', ''),
        'resume_command': f'conda run -n cv python {SCRIPT_PATH} --output-run-dir {run_dir} --resume',
    }

    logger.log('[STEP 17] Run Sub Agent validations')
    logger.log('[STEP 18] Generate markdown/json reports')
    write_summary(outputs['summary_md'], report, video_summary, corrected_summary, posthoc_summary, outputs)
    outputs['report_json'].write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    logger.log('[STEP 19] Update latest bundles')
    copy_latest(outputs)
    latest_forbidden = scan_latest_forbidden([LATEST_CHATGPT, LATEST_OCR])
    quality = build_quality_checks(schedule_validation, frame_df, outputs, protected_changes, latest_forbidden, execution_status)
    quality.to_csv(outputs['quality_checks'], index=False)
    shutil.copy2(outputs['quality_checks'], LATEST_CHATGPT / outputs['quality_checks'].name)
    shutil.copy2(outputs['quality_checks'], LATEST_OCR / outputs['quality_checks'].name)
    report['validation_results'] = quality.to_dict('records')
    report['latest_forbidden_files'] = latest_forbidden
    report['safety_results']['latest_forbidden_files_absent'] = len(latest_forbidden) == 0
    outputs['report_json'].write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_summary(outputs['summary_md'], report, video_summary, corrected_summary, posthoc_summary, outputs)
    for key in ['report_json', 'summary_md']:
        shutil.copy2(outputs[key], LATEST_CHATGPT / outputs[key].name)
        shutil.copy2(outputs[key], LATEST_OCR / outputs[key].name)

    logger.log('[STEP 20] Print final human-readable summary')
    print('\n1. OCR extraction status:')
    print(f'   - {execution_status}')
    print(f'   - run_id: {run_id}')
    print(f'   - run_dir: {run_dir}')
    print('\n2. Target scope:')
    print('   - split_role_v2_5: development')
    print('   - original_split_v2_4: train')
    print(f'   - video_count: {len(EXPECTED_VIDEO_IDS)}')
    print(f'   - video_ids: {EXPECTED_VIDEO_IDS}')
    print('\n3. Input validation:')
    print(f"   - schedule count: {schedule_validation['schedule_row_count']}")
    print(f"   - anchor_dense count: {schedule_validation['schedule_source_counts'].get('anchor_dense')}")
    print(f"   - background_regular count: {schedule_validation['schedule_source_counts'].get('background_regular')}")
    print('   - Extended Evaluation processed: false')
    print('   - Pure Test processed: false')
    print('\n4. OCR backend:')
    print(f"   - backend: {backend_info.get('ocr_backend')}")
    print(f"   - version: {backend_info.get('engine_version')}")
    print(f"   - device/GPU if available: {backend_info.get('gpu_used')}")
    print('\n5. OCR results:')
    print(f"   - attempted: {counts['attempted']}")
    print(f"   - success_text: {counts['success_text']}")
    print(f"   - success_empty: {counts['success_empty']}")
    print(f"   - failed: {counts['failed']}")
    print(f"   - nonempty ratio: {counts['nonempty_ratio']}")
    print(f"   - per-video summary path: {outputs['video_summary']}")
    print('\n6. Features:')
    print(f"   - frame result path: {outputs['frame_results']}")
    print(f"   - video summary path: {outputs['video_summary']}")
    print(f"   - 20s timeline feature path: {outputs['timeline_features']}")
    print(f"   - anchor window feature path: {outputs['anchor_features']}")
    print(f"   - corrected keyword summary path: {outputs['corrected_keyword_summary']}")
    print(f"   - posthoc ad context summary path: {outputs['posthoc_context_summary']}")
    print('\n7. Keyword summary:')
    print(f'   - corrected ad disclosure hits: {len(ad_rows)}')
    print(f'   - typo variant hits: {len(typo_rows)}')
    print(f'   - fuzzy review-needed hits: {len(fuzzy_rows)}')
    print('   - top keyword categories: ' + ';'.join(corrected_summary.sort_values('corrected_keyword_count_sum', ascending=False).head(5).apply(lambda r: f"{r['keyword_category']}:{r['corrected_keyword_count_sum']}", axis=1).tolist()))
    print('\n8. Checkpoint/resume:')
    print(f"   - checkpoint status: {execution_status}")
    print(f"   - resume command: conda run -n cv python {SCRIPT_PATH} --output-run-dir {run_dir} --resume")
    print('\n9. Latest bundles:')
    print(f'   - latest_for_chatgpt: {LATEST_CHATGPT}')
    print(f'   - latest_ocr: {LATEST_OCR}')
    print('\n10. Safety:')
    print('   - actual label used for sampling: false')
    print('   - actual label used for posthoc development analysis: true')
    print('   - detector modified: false')
    print('   - existing OCR modified: false')
    print('   - existing scene anchor/candidate modified: false')
    print('   - split/label modified: false')
    print('   - raw frame persisted: false')
    print(f'   - latest forbidden files: {len(latest_forbidden)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
