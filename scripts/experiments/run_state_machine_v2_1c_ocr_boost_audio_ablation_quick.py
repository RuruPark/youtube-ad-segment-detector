#!/usr/bin/env python3
"""v2.1c quick OCR boost / audio ablation experiment.

Quick experiment only. This script reads existing v2.1b candidate outputs and writes a
separate v2.1c experiment root. It does not run feature extraction or modify previous
experiment/viewer outputs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple

ESTIMATED_RUNTIME = "Estimated runtime: approximately 25-40 minutes"
DEV_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
OCR_MODES = ["baseline_ocr", "disclosure_boost", "intro_context_boost", "disclosure_intro_combo"]
AUDIO_MODES = ["with_audio", "no_audio", "audio_end_only"]
FORBIDDEN_DECISION_PATTERNS = [
    "label", "true", "actual", "gt", "ground_truth", "audit",
    "nearest_true_boundary", "distance_to_nearest_true_boundary", "is_near_true_boundary",
    "ad_overlap", "is_ad_overlap", "is_ad_core", "is_clean_nonad", "overlapping_ad_interval_ids",
    "actual_ad_start_sec", "actual_ad_end_sec", "audio_candidate_score_for_discussion",
    "per_video_ad_vs_nonad_contrast_score",
]
DECISION_FEATURE_COLUMNS = [
    "candidate_start_sec", "candidate_end_sec", "candidate_duration_sec",
    "base_interval_ad_score", "base_interval_score_density",
    "base_start_score", "base_end_quality_score",
    "hard_evidence_count", "hard_evidence_density_per_60s",
    "ocr_hard_count", "product_cta_timeline_types",
    "ocr_disclosure_exact_typo_proximity_count", "ocr_disclosure_support_combo_flag",
    "ocr_opening_disclosure_alone_guard_flag", "ocr_intro_context_count",
    "ocr_intro_context_support_combo_flag", "ocr_body_product_cta_density",
    "ocr_body_timeline_hard_evidence_count", "ocr_post_end_keyword_drop_flag",
    "ocr_post_end_return_to_normal_flag", "ocr_post_end_still_ad_like_flag",
    "audio_mode", "audio_original_relative_support_score",
    "audio_relative_support_score_for_variant", "audio_start_contribution_used",
    "audio_continuity_contribution_used", "audio_end_contribution_used",
    "audio_removed_for_ablation", "audio_end_only_for_ocr_drop",
    "audio_alone_end_confirm_allowed",
]


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], preferred: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = preferred or []
    keys: List[str] = []
    for key in preferred:
        if key not in keys:
            keys.append(key)
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def interval_start(row: Dict[str, Any]) -> float:
    return to_float(row.get("ad_start_sec") or row.get("candidate_start_sec") or row.get("start_sec") or row.get("start"))


def interval_end(row: Dict[str, Any]) -> float:
    start = interval_start(row)
    end = to_float(row.get("ad_end_sec") or row.get("candidate_end_sec") or row.get("last_anchor_sec") or row.get("end_sec") or row.get("end"), start)
    return max(start, end)


def interval_duration(row: Dict[str, Any]) -> float:
    dur = to_float(row.get("ad_duration_sec") or row.get("duration_sec") or row.get("duration_proxy_sec") or row.get("duration"), -1.0)
    if dur >= 0:
        return dur
    return max(0.0, interval_end(row) - interval_start(row))


def overlap_sec(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def total_duration(intervals: Iterable[Tuple[float, float]]) -> float:
    return sum(max(0.0, e - s) for s, e in intervals)


def total_overlap(preds: List[Tuple[float, float, int]], actuals: List[Tuple[float, float, int]]) -> float:
    # tuple은 (start, end, video_id) 형태이며 같은 video interval끼리만 매칭한다.
    return sum(
        overlap_sec((ps, pe), (as_, ae))
        for ps, pe, pvid in preds
        for as_, ae, avid in actuals
        if pvid == avid
    )


def boundary_errors(preds: List[Tuple[float, float, int]], actuals: List[Tuple[float, float, int]]) -> Tuple[List[float], List[float]]:
    # tuple은 (start, end, video_id) 형태이며 같은 video 안에서 최대 overlap으로 매칭한다.
    start_errors: List[float] = []
    end_errors: List[float] = []
    used_pred: set[int] = set()
    for as_, ae, avid in actuals:
        best = None
        best_ov = 0.0
        for idx, (ps, pe, pvid) in enumerate(preds):
            if idx in used_pred or pvid != avid:
                continue
            ov = overlap_sec((ps, pe), (as_, ae))
            if ov > best_ov:
                best_ov = ov
                best = idx
        if best is not None and best_ov > 0:
            used_pred.add(best)
            ps, pe, _ = preds[best]
            start_errors.append(abs(ps - as_))
            end_errors.append(abs(pe - ae))
    return start_errors, end_errors


def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def safe_median(values: List[float]) -> float:
    return median(values) if values else 0.0


def checksum_intervals(rows: List[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: (to_int(r.get("video_id")), interval_start(r), interval_end(r), str(r.get("candidate_id")))):
        h.update(f"{to_int(row.get('video_id'))}|{interval_start(row):.6f}|{interval_end(row):.6f}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def backup_existing(paths: List[Path], backup_base: Path) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    backup_dir = backup_base / f"state_machine_v2_1c_ocr_boost_audio_ablation_quick_{now_ts()}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    manifest = []
    for path in existing:
        dest = backup_dir / path.relative_to(Path('.'))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, dest)
            kind = "directory"
        else:
            shutil.copy2(path, dest)
            kind = "file"
        manifest.append({"source": str(path), "backup": str(dest), "kind": kind})
    write_json(backup_dir / "backup_manifest.json", {"created_at": now_ts(), "entries": manifest})
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return backup_dir


def load_base_rows(base_root: Path) -> List[Dict[str, Any]]:
    buckets = [
        ("prediction", "predictions.csv"),
        ("review_only", "review_only_candidates.csv"),
        ("overprediction_pruned_review", "overprediction_pruned_review_candidates.csv"),
        ("open", "open_candidates.csv"),
    ]
    rows: List[Dict[str, Any]] = []
    for bucket, fname in buckets:
        for row in read_csv(base_root / fname):
            out = dict(row)
            out["base_v2_1b_bucket"] = bucket
            out["base_candidate_key"] = out.get("base_candidate_id") or out.get("candidate_id")
            rows.append(out)
    return rows


def load_phase_features(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv(path)
    return {row["base_candidate_id"]: row for row in rows}


def load_actual_intervals(path: Path, dev_ids: List[int]) -> Dict[int, List[Tuple[float, float, int]]]:
    actual: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)
    rows = read_csv(path)
    idx = 0
    for row in rows:
        if row.get("segment_type") != "ad_interval":
            continue
        if str(row.get("label_valid", "true")).lower() not in {"true", "1", "yes"}:
            continue
        vid = to_int(row.get("video_id"))
        if vid not in dev_ids:
            continue
        start = to_float(row.get("ad_start_sec") or row.get("segment_start_sec"))
        end = to_float(row.get("ad_end_sec") or row.get("segment_end_sec"))
        if end > start:
            idx += 1
            actual[vid].append((start, end, idx))
    return actual


def compute_feature_flags(row: Dict[str, Any], feat: Dict[str, str]) -> Dict[str, Any]:
    start = interval_start(row)
    duration = interval_duration(row)
    video_duration = to_float(row.get("video_duration_sec"), to_float(feat.get("video_duration_sec"), 0.0))
    opening_window = min(45.0, max(30.0, 0.02 * video_duration)) if video_duration else 45.0
    disclosure_count = sum(to_float(feat.get(k)) for k in [
        "ocr_candidate_pre_start_disclosure_exact_count",
        "ocr_candidate_pre_start_disclosure_typo_count",
        "ocr_candidate_pre_start_disclosure_proximity_count",
        "ocr_candidate_start_disclosure_exact_count",
        "ocr_candidate_start_disclosure_typo_count",
        "ocr_candidate_start_disclosure_proximity_count",
    ])
    product_cta = sum(to_float(feat.get(k)) for k in [
        "ocr_candidate_start_product_brand_count",
        "ocr_candidate_start_cta_link_count",
        "ocr_candidate_body_product_brand_count",
        "ocr_candidate_body_purchase_cta_count",
        "ocr_candidate_body_link_more_info_count",
    ])
    body_density = to_float(feat.get("ocr_candidate_body_product_cta_density"))
    body_timeline = to_float(feat.get("ocr_candidate_body_timeline_hard_evidence_count"))
    sponsor = to_float(feat.get("ocr_candidate_start_sponsor_count")) + to_float(feat.get("ocr_candidate_body_sponsor_count"))
    intro = to_float(feat.get("ocr_candidate_start_intro_context_count"))
    disclosure_support_combo = disclosure_count > 0 and (product_cta > 0 or body_density > 0 or body_timeline > 0)
    intro_support_combo = intro > 0 and (sponsor > 0 or disclosure_count > 0 or product_cta > 0 or body_density > 0 or body_timeline > 0)
    opening_disclosure_alone = disclosure_count > 0 and start <= opening_window and not disclosure_support_combo
    return {
        "candidate_start_sec": start,
        "candidate_end_sec": interval_end(row),
        "candidate_duration_sec": duration,
        "ocr_disclosure_exact_typo_proximity_count": disclosure_count,
        "ocr_disclosure_support_combo_flag": disclosure_support_combo,
        "ocr_opening_disclosure_alone_guard_flag": opening_disclosure_alone,
        "ocr_intro_context_count": intro,
        "ocr_intro_context_support_combo_flag": intro_support_combo,
        "ocr_body_product_cta_density": body_density,
        "ocr_body_timeline_hard_evidence_count": body_timeline,
        "ocr_post_end_keyword_drop_flag": boolish(feat.get("ocr_candidate_post_end_keyword_drop_flag")),
        "ocr_post_end_return_to_normal_flag": boolish(feat.get("ocr_candidate_post_end_return_to_normal_flag")),
        "ocr_post_end_still_ad_like_flag": boolish(feat.get("ocr_candidate_post_end_still_ad_like_flag")),
        "opening_window_sec": opening_window,
        "sponsor_or_disclosure_or_product_cta_support_count": sponsor + disclosure_count + product_cta,
    }


def base_audio_score(row: Dict[str, Any]) -> Tuple[float, str]:
    if "audio_relative_support_score" in row and row.get("audio_relative_support_score") not in (None, ""):
        return clip(to_float(row.get("audio_relative_support_score"))), "audio_relative_support_score"
    if "audio_relative_support_count" in row and row.get("audio_relative_support_count") not in (None, ""):
        return clip(to_float(row.get("audio_relative_support_count")) / 8.0), "audio_relative_support_count_fallback"
    return 0.0, "missing_audio_score_fallback_zero"


def is_ultra(row: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    ultra = cfg["ultra_high_confidence"]
    return (
        to_float(row.get("interval_ad_score")) >= ultra["ultra_min_interval_ad_score"] and
        to_float(row.get("interval_score_density")) >= ultra["ultra_min_interval_score_density"] and
        to_float(row.get("hard_evidence_count")) >= ultra["ultra_min_hard_evidence_count"] and
        to_float(row.get("hard_evidence_density_per_60s")) >= ultra["ultra_min_hard_evidence_density_per_60s"] and
        to_float(row.get("ocr_hard_count")) >= ultra["ultra_min_ocr_hard_count"] and
        to_float(row.get("product_cta_timeline_types")) >= ultra["ultra_min_product_cta_timeline_types"]
    )


def apply_variant(
    variant: Dict[str, Any],
    base_rows: List[Dict[str, Any]],
    features: Dict[str, Dict[str, str]],
    cfg: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    ocr_mode = variant["ocr_boost_mode"]
    audio_mode = variant["audio_mode"]
    ocr_cfg = cfg["ocr_boost_modes"][ocr_mode]
    weights = cfg["score_weights"]
    thresholds = cfg["thresholds"]
    pre_budget_predictions: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    open_rows: List[Dict[str, Any]] = []
    disclosure_audit: List[Dict[str, Any]] = []
    intro_audit: List[Dict[str, Any]] = []
    audio_audit: List[Dict[str, Any]] = []
    safety_audit: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    for idx, base in enumerate(base_rows, 1):
        base_key = base.get("base_candidate_key") or base.get("base_candidate_id") or base.get("candidate_id") or f"row_{idx}"
        feat = features.get(base_key, {})
        flags = compute_feature_flags(base, feat)
        row = deepcopy(base)
        row["variant_id"] = variant["variant_id"]
        row["candidate_id"] = f"{variant['variant_id']}_{base_key}"
        row["base_candidate_id"] = base_key
        row["ocr_boost_mode"] = ocr_mode
        row["audio_mode"] = audio_mode
        row["version"] = "v2.1c_quick"
        row["detector_id"] = "state_machine_v2_1c_ocr_boost_audio_ablation_quick"
        row["actual_label_used_for_decision"] = "false"
        row["plus5_actual_label_phase_used_for_decision"] = "false"
        row["decision_feature_columns_json"] = json.dumps(DECISION_FEATURE_COLUMNS, ensure_ascii=False)
        row["forbidden_decision_columns_found"] = "[]"
        row["audio_candidate_score_for_discussion_used_for_decision"] = "false"
        row["per_video_ad_vs_nonad_contrast_score_used_for_decision"] = "false"
        row["fuzzy_only_used_as_hard_evidence"] = "false"
        row["opening_disclosure_alone_hard_start"] = "false"
        row["intro_context_alone_hard_start"] = "false"
        row["audio_alone_start_or_end_confirm"] = "false"

        base_score = to_float(base.get("interval_ad_score"))
        base_density = to_float(base.get("interval_score_density"), base_score)
        base_start = to_float(base.get("start_strength_score"))
        base_end = to_float(base.get("end_quality_score"))
        duration = interval_duration(base)
        duration_scale = max(1.0, duration / 60.0)

        disclosure_delta_component = 0.0
        if ocr_cfg.get("disclosure_boost_weight", 0.0) > 0 and flags["ocr_disclosure_exact_typo_proximity_count"] > 0 and flags["ocr_disclosure_support_combo_flag"] and not flags["ocr_opening_disclosure_alone_guard_flag"]:
            disclosure_delta_component = float(ocr_cfg["disclosure_boost_weight"])
        intro_delta_component = 0.0
        if ocr_cfg.get("intro_context_boost_weight", 0.0) > 0 and flags["ocr_intro_context_count"] > 0 and flags["ocr_intro_context_support_combo_flag"]:
            intro_delta_component = float(ocr_cfg["intro_context_boost_weight"])
        total_start_delta_component = disclosure_delta_component + intro_delta_component
        if "combo_clip_delta" in ocr_cfg:
            total_start_delta_component = min(total_start_delta_component, float(ocr_cfg["combo_clip_delta"]))

        audio_score, audio_source = base_audio_score(base)
        audio_removed = audio_mode in {"no_audio", "audio_end_only"}
        end_combo = (flags["ocr_post_end_keyword_drop_flag"] or flags["ocr_post_end_return_to_normal_flag"]) and not flags["ocr_post_end_still_ad_like_flag"]
        audio_end_support = audio_score if audio_mode == "audio_end_only" and end_combo else 0.0
        audio_score_for_variant = audio_score
        audio_score_delta = 0.0
        end_delta_component = 0.0
        if audio_mode == "no_audio":
            audio_score_for_variant = 0.0
            audio_score_delta = -weights["audio_relative_support_score"] * audio_score
        elif audio_mode == "audio_end_only":
            # 일반 start/continuity audio를 제거하고, OCR end context가 있을 때만 축소된 end-only 기여를 허용한다.
            audio_score_for_variant = audio_end_support
            audio_score_delta = -weights["audio_relative_support_score"] * audio_score + weights["audio_relative_support_score"] * audio_end_support * 0.5
            end_delta_component = 0.04 * audio_end_support if audio_end_support > 0 else 0.0

        start_score_delta = weights["start_strength_score"] * total_start_delta_component
        end_score_delta = weights["end_quality_score"] * end_delta_component
        score_delta = start_score_delta + end_score_delta + audio_score_delta
        new_start = clip(base_start + total_start_delta_component)
        new_end = clip(base_end + end_delta_component)
        new_score = clip(base_score + score_delta)
        new_density = clip(base_density + score_delta / duration_scale)

        row.update({
            "start_strength_score": f"{new_start:.6f}",
            "end_quality_score": f"{new_end:.6f}",
            "interval_ad_score": f"{new_score:.6f}",
            "interval_score_density": f"{new_density:.6f}",
            "v2_1c_score_delta": f"{score_delta:.6f}",
            "ocr_disclosure_boost_applied": str(disclosure_delta_component > 0),
            "ocr_disclosure_start_boost_delta": f"{disclosure_delta_component:.6f}",
            "intro_context_boost_applied": str(intro_delta_component > 0),
            "intro_context_start_support_delta": f"{intro_delta_component:.6f}",
            "audio_relative_support_score": f"{audio_score_for_variant:.6f}",
            "audio_original_relative_support_score": f"{audio_score:.6f}",
            "audio_score_source": audio_source,
            "audio_removed_for_ablation": str(audio_mode == "no_audio"),
            "audio_end_only_for_ocr_drop": str(audio_mode == "audio_end_only"),
            "audio_start_contribution_used": str(audio_mode == "with_audio"),
            "audio_continuity_contribution_used": str(audio_mode == "with_audio"),
            "audio_end_contribution_used": str(audio_mode == "with_audio" or audio_end_support > 0),
            "audio_end_support_score": f"{audio_end_support:.6f}",
            "audio_end_support_requires_ocr_drop_or_return": str(audio_mode == "audio_end_only"),
            "audio_alone_end_confirm": "false",
            "ocr_disclosure_exact_typo_proximity_count": f"{flags['ocr_disclosure_exact_typo_proximity_count']:.6f}",
            "ocr_disclosure_support_combo_flag": str(flags["ocr_disclosure_support_combo_flag"]),
            "ocr_opening_disclosure_alone_guard_flag": str(flags["ocr_opening_disclosure_alone_guard_flag"]),
            "ocr_intro_context_count": f"{flags['ocr_intro_context_count']:.6f}",
            "ocr_intro_context_support_combo_flag": str(flags["ocr_intro_context_support_combo_flag"]),
            "ocr_body_product_cta_density": f"{flags['ocr_body_product_cta_density']:.6f}",
            "ocr_body_timeline_hard_evidence_count": f"{flags['ocr_body_timeline_hard_evidence_count']:.6f}",
            "ocr_post_end_keyword_drop_flag": str(flags["ocr_post_end_keyword_drop_flag"]),
            "ocr_post_end_return_to_normal_flag": str(flags["ocr_post_end_return_to_normal_flag"]),
            "ocr_post_end_still_ad_like_flag": str(flags["ocr_post_end_still_ad_like_flag"]),
        })

        forbidden_found = [pat for pat in FORBIDDEN_DECISION_PATTERNS if any(pat in col.lower() for col in DECISION_FEATURE_COLUMNS)]
        safety_audit.append({
            "variant_id": variant["variant_id"],
            "base_candidate_id": base_key,
            "video_id": row.get("video_id"),
            "decision_feature_columns_json": row["decision_feature_columns_json"],
            "forbidden_decision_columns_found": json.dumps(forbidden_found),
            "actual_label_used_for_decision": "false",
            "plus5_actual_label_phase_used_for_decision": "false",
            "audio_candidate_score_for_discussion_used_for_decision": "false",
            "per_video_ad_vs_nonad_contrast_score_used_for_decision": "false",
        })
        disclosure_audit.append({
            "variant_id": variant["variant_id"],
            "base_candidate_id": base_key,
            "video_id": row.get("video_id"),
            "ocr_boost_mode": ocr_mode,
            "disclosure_exact_typo_proximity_count": f"{flags['ocr_disclosure_exact_typo_proximity_count']:.6f}",
            "support_combo_flag": str(flags["ocr_disclosure_support_combo_flag"]),
            "opening_disclosure_alone_guard_flag": str(flags["ocr_opening_disclosure_alone_guard_flag"]),
            "disclosure_boost_applied": str(disclosure_delta_component > 0),
            "fuzzy_only_used_as_hard_evidence": "false",
            "opening_disclosure_alone_hard_start": "false",
            "score_delta_component": f"{disclosure_delta_component:.6f}",
        })
        intro_audit.append({
            "variant_id": variant["variant_id"],
            "base_candidate_id": base_key,
            "video_id": row.get("video_id"),
            "ocr_boost_mode": ocr_mode,
            "intro_context_count": f"{flags['ocr_intro_context_count']:.6f}",
            "intro_support_combo_flag": str(flags["ocr_intro_context_support_combo_flag"]),
            "intro_context_boost_applied": str(intro_delta_component > 0),
            "intro_context_alone_hard_start": "false",
            "score_delta_component": f"{intro_delta_component:.6f}",
        })
        audio_audit.append({
            "variant_id": variant["variant_id"],
            "base_candidate_id": base_key,
            "video_id": row.get("video_id"),
            "audio_mode": audio_mode,
            "audio_original_relative_support_score": f"{audio_score:.6f}",
            "audio_relative_support_score_for_decision": f"{audio_score_for_variant:.6f}",
            "audio_score_source": audio_source,
            "audio_removed_for_ablation": str(audio_mode == "no_audio"),
            "audio_start_contribution_used": str(audio_mode == "with_audio"),
            "audio_continuity_contribution_used": str(audio_mode == "with_audio"),
            "audio_end_contribution_used": str(audio_mode == "with_audio" or audio_end_support > 0),
            "audio_end_only_for_ocr_drop": str(audio_mode == "audio_end_only"),
            "ocr_drop_or_return_combo_for_audio_end": str(end_combo),
            "audio_alone_end_confirm": "false",
            "audio_score_delta": f"{audio_score_delta:.6f}",
        })
        trace_rows.append({
            "variant_id": variant["variant_id"],
            "base_candidate_id": base_key,
            "video_id": row.get("video_id"),
            "base_bucket": base.get("base_v2_1b_bucket"),
            "new_interval_ad_score": row["interval_ad_score"],
            "score_delta": row["v2_1c_score_delta"],
            "disclosure_boost_applied": row["ocr_disclosure_boost_applied"],
            "intro_context_boost_applied": row["intro_context_boost_applied"],
            "audio_mode": audio_mode,
            "actual_label_used_for_decision": "false",
        })

        if base.get("base_v2_1b_bucket") == "open":
            row["interval_status"] = "open_candidate"
            open_rows.append(row)
            continue
        passes = (
            new_score >= thresholds["min_interval_ad_score"] and
            new_density >= thresholds["min_interval_score_density"] and
            to_float(row.get("hard_evidence_count")) >= thresholds["min_hard_evidence_count"] and
            to_float(row.get("hard_evidence_density_per_60s")) >= thresholds["min_hard_evidence_density_per_60s"]
        )
        if passes:
            row["interval_status"] = "prediction_candidate_before_budget"
            row["failure_or_review_reason"] = row.get("failure_or_review_reason", "")
            pre_budget_predictions.append(row)
        else:
            row["interval_status"] = "review_only_candidate"
            reason = row.get("failure_or_review_reason") or "below_v2_1c_score_threshold"
            if audio_removed:
                reason = (reason + ";audio_ablation_score_adjusted").strip(";")
            row["failure_or_review_reason"] = reason
            review_rows.append(row)

    final_predictions: List[Dict[str, Any]] = []
    pruned_rows: List[Dict[str, Any]] = []
    budget_events: List[Dict[str, Any]] = []
    by_video: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in pre_budget_predictions:
        by_video[to_int(row.get("video_id"))].append(row)
    bg = cfg["budget_guard"]
    for vid, rows in by_video.items():
        rows_sorted = sorted(rows, key=lambda r: (not is_ultra(r, cfg), to_float(r.get("interval_ad_score"))))
        video_duration = to_float(rows_sorted[0].get("video_duration_sec")) if rows_sorted else 0.0
        pred_total = sum(interval_duration(r) for r in rows_sorted)
        before_ratio = pred_total / video_duration if video_duration else 0.0
        kept = list(rows_sorted)
        demoted: List[Dict[str, Any]] = []
        if before_ratio > bg["hard_overprediction_ratio"]:
            # target ratio를 만족하거나 ultra 후보만 남을 때까지 가장 낮은 non-ultra 후보를 내린다.
            kept = sorted(rows_sorted, key=lambda r: (is_ultra(r, cfg), -to_float(r.get("interval_ad_score"))))
            while kept and (sum(interval_duration(r) for r in kept) / video_duration if video_duration else 0.0) > bg["target_prediction_ratio_after_pruning"]:
                candidates = [r for r in kept if not is_ultra(r, cfg)]
                if not candidates:
                    break
                demote = sorted(candidates, key=lambda r: to_float(r.get("interval_ad_score")))[0]
                kept.remove(demote)
                demoted.append(demote)
        kept_ids = {id(r) for r in kept}
        demoted_ids = {id(r) for r in demoted}
        for row in rows_sorted:
            after_total = sum(interval_duration(r) for r in kept)
            action = "kept_no_budget_guard_needed"
            if before_ratio > bg["hard_overprediction_ratio"]:
                action = "overprediction_pruned_review" if id(row) in demoted_ids else ("kept_ultra_high_confidence" if is_ultra(row, cfg) else "kept_after_budget_guard")
            elif before_ratio > bg["soft_overprediction_ratio"]:
                action = "kept_soft_budget_warning"
            event = {
                "variant_id": variant["variant_id"],
                "video_id": vid,
                "candidate_id": row.get("candidate_id"),
                "video_duration_sec": f"{video_duration:.6f}",
                "candidate_duration_sec": f"{interval_duration(row):.6f}",
                "prediction_ratio_before_budget_guard": f"{before_ratio:.6f}",
                "prediction_ratio_after_event": f"{(after_total / video_duration if video_duration else 0.0):.6f}",
                "soft_overprediction_ratio": bg["soft_overprediction_ratio"],
                "hard_overprediction_ratio": bg["hard_overprediction_ratio"],
                "target_prediction_ratio_after_pruning": bg["target_prediction_ratio_after_pruning"],
                "budget_guard_action": action,
                "ultra_high_confidence": str(is_ultra(row, cfg)).lower(),
                "consistency_failure": "false",
                "actual_label_used_for_decision": "false",
            }
            budget_events.append(event)
            row["budget_guard_action"] = action
            row["ultra_high_confidence"] = str(is_ultra(row, cfg)).lower()
            row["video_prediction_ratio_before_budget_guard"] = f"{before_ratio:.6f}"
            row["video_prediction_ratio_after_budget_guard"] = event["prediction_ratio_after_event"]
            if id(row) in kept_ids:
                row["interval_status"] = "prediction"
                final_predictions.append(row)
            else:
                row["interval_status"] = "overprediction_pruned_review"
                row["failure_or_review_reason"] = "v2_1c_budget_guard_demoted"
                pruned_rows.append(row)
    return {
        "predictions": sorted(final_predictions, key=lambda r: (to_int(r.get("video_id")), interval_start(r))),
        "review_only_candidates": sorted(review_rows, key=lambda r: (to_int(r.get("video_id")), interval_start(r))),
        "overprediction_pruned_review_candidates": sorted(pruned_rows, key=lambda r: (to_int(r.get("video_id")), interval_start(r))),
        "open_candidates": sorted(open_rows, key=lambda r: (to_int(r.get("video_id")), interval_start(r))),
        "budget_guard_events": sorted(budget_events, key=lambda r: (to_int(r.get("video_id")), str(r.get("candidate_id")))) ,
        "trace_sample": trace_rows[:200],
        "disclosure_audit": disclosure_audit,
        "intro_audit": intro_audit,
        "audio_audit": audio_audit,
        "safety_audit": safety_audit,
    }


def score_variant(variant_id: str, outputs: Dict[str, List[Dict[str, Any]]], actual: Dict[int, List[Tuple[float, float, int]]], dev_ids: List[int]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    preds_by_vid: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)
    for i, row in enumerate(outputs["predictions"], 1):
        preds_by_vid[to_int(row.get("video_id"))].append((interval_start(row), interval_end(row), i))
    all_preds = [(s, e, vid) for vid, rows in preds_by_vid.items() for s, e, _ in rows]
    all_actual = [(s, e, vid) for vid, rows in actual.items() for s, e, _ in rows]
    pred_total = total_duration((s, e) for s, e, _ in all_preds)
    actual_total = total_duration((s, e) for s, e, _ in all_actual)
    overlap = total_overlap(all_preds, all_actual)
    start_errs, end_errs = boundary_errors(all_preds, all_actual)
    median_boundary = safe_median(start_errs + end_errs)
    total_video_duration = sum(max([to_float(r.get("video_duration_sec")) for bucket in ["predictions", "review_only_candidates", "overprediction_pruned_review_candidates", "open_candidates"] for r in outputs[bucket] if to_int(r.get("video_id")) == vid] or [0.0]) for vid in dev_ids)
    false_positive = max(0.0, pred_total - overlap)
    missed = max(0.0, actual_total - overlap)
    recall = overlap / actual_total if actual_total else 0.0
    precision = overlap / pred_total if pred_total else 0.0
    fp_ratio = false_positive / total_video_duration if total_video_duration else 0.0
    missed_ratio = missed / actual_total if actual_total else 0.0
    boundary_quality = math.exp(-median_boundary / 10.0) if median_boundary else (1.0 if start_errs or end_errs else 0.0)
    review_burden = len(outputs["review_only_candidates"]) / max(1, len(outputs["predictions"]) + len(outputs["review_only_candidates"]))
    overprediction_penalty = fp_ratio
    balanced = 0.35 * recall + 0.25 * precision + 0.15 * boundary_quality - 0.15 * fp_ratio - 0.05 * overprediction_penalty - 0.05 * review_burden
    metrics = {
        "variant_id": variant_id,
        "prediction_count": len(outputs["predictions"]),
        "review_only_count": len(outputs["review_only_candidates"]),
        "overprediction_pruned_review_count": len(outputs["overprediction_pruned_review_candidates"]),
        "open_count": len(outputs["open_candidates"]),
        "final_prediction_total_duration_sec": pred_total,
        "mean_final_prediction_ratio": pred_total / total_video_duration if total_video_duration else 0.0,
        "actual_total_duration_sec": actual_total,
        "prediction_overlap_with_actual_sec": overlap,
        "actual_overlap_recall": recall,
        "prediction_overlap_precision_proxy": precision,
        "false_positive_duration_sec": false_positive,
        "false_positive_ratio_of_video": fp_ratio,
        "missed_actual_duration_sec": missed,
        "missed_actual_ratio": missed_ratio,
        "mean_boundary_start_error_sec": safe_mean(start_errs),
        "median_boundary_start_error_sec": safe_median(start_errs),
        "mean_boundary_end_error_sec": safe_mean(end_errs),
        "median_boundary_end_error_sec": safe_median(end_errs),
        "boundary_quality_score": boundary_quality,
        "balanced_objective_score": balanced,
        "precision_safe_score": precision - 0.5 * fp_ratio,
        "recall_safe_score": recall - 0.1 * missed_ratio,
        "low_overprediction_score": precision - fp_ratio,
        "config_failure_flag": False,
        "config_failure_reason": "",
    }
    video_metrics: List[Dict[str, Any]] = []
    for vid in dev_ids:
        preds_raw = preds_by_vid.get(vid, [])
        acts_raw = actual.get(vid, [])
        preds = [(s, e, vid) for s, e, _ in preds_raw]
        acts = [(s, e, vid) for s, e, _ in acts_raw]
        ptotal = total_duration((s, e) for s, e, _ in preds)
        atotal = total_duration((s, e) for s, e, _ in acts)
        ov = total_overlap(preds, acts)
        serr, eerr = boundary_errors(preds, acts)
        video_metrics.append({
            "variant_id": variant_id,
            "video_id": vid,
            "prediction_count": len(preds),
            "prediction_duration_sec": ptotal,
            "actual_duration_sec": atotal,
            "overlap_sec": ov,
            "actual_overlap_recall": ov / atotal if atotal else 0.0,
            "prediction_overlap_precision_proxy": ov / ptotal if ptotal else 0.0,
            "false_positive_duration_sec": max(0.0, ptotal - ov),
            "missed_actual_duration_sec": max(0.0, atotal - ov),
            "median_boundary_start_error_sec": safe_median(serr),
            "median_boundary_end_error_sec": safe_median(eerr),
        })
    return metrics, video_metrics


def variant_interpretation(left: Dict[str, Any], right: Dict[str, Any], right_mode: str) -> str:
    d_recall = right["actual_overlap_recall"] - left["actual_overlap_recall"]
    d_precision = right["prediction_overlap_precision_proxy"] - left["prediction_overlap_precision_proxy"]
    d_fp = right["false_positive_duration_sec"] - left["false_positive_duration_sec"]
    if abs(d_recall) < 0.01 and abs(d_precision) < 0.01 and abs(d_fp) < 30:
        return "audio contribution appears low for this OCR mode"
    if right_mode == "no_audio" and d_precision > 0.02 and d_recall > -0.02:
        return "audio may be increasing false positives; no_audio keeps recall similar with better precision"
    if right_mode == "no_audio" and d_recall < -0.03:
        return "audio appears helpful for recall in this OCR mode"
    if right_mode == "audio_end_only" and d_fp < -30 and d_recall > -0.02:
        return "audio_end_only may be more stable: lower FP while maintaining recall"
    return "mixed or small audio ablation effect"


def main() -> None:
    print(ESTIMATED_RUNTIME)
    project_root = Path(".")
    config_path = project_root / "configs/experiments/state_machine_v2_1c_ocr_boost_audio_ablation_quick_config.json"
    cfg = read_json(config_path)
    paths = {k: Path(v) for k, v in cfg["paths"].items() if k != "project_root"}
    log_lines = [f"[{datetime.now().isoformat(timespec='seconds')}] start v2.1c quick experiment"]

    backup_dir = backup_existing([
        paths["experiment_root"], paths["viewer_patch"], paths["short_summary"], paths["report"], paths["run_log"], paths["latest_bundle"],
    ], project_root / "backups")
    paths["experiment_root"].mkdir(parents=True, exist_ok=True)
    paths["short_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["run_log"].parent.mkdir(parents=True, exist_ok=True)

    # 읽기 전용 input의 존재 여부를 검증한다.
    required_inputs = [
        paths["base_output_root"] / "predictions.csv",
        paths["base_output_root"] / "review_only_candidates.csv",
        paths["base_output_root"] / "overprediction_pruned_review_candidates.csv",
        paths["base_output_root"] / "open_candidates.csv",
        paths["base_output_root"] / "budget_guard_events.csv",
        paths["base_output_root"] / "video_summary.csv",
        paths["base_v2_1b_config"], paths["base_v2_1a_configs"], paths["phase_features"],
        paths["ocr_frame_results"], paths["ocr_anchor_features"], paths["ocr_timeline_features"],
        paths["actual_labels"], paths["split_csv"], paths["metadata_csv"],
    ]
    missing = [str(p) for p in required_inputs if not p.exists()]
    if missing:
        raise SystemExit("missing required inputs: " + ", ".join(missing))

    # label 없이 decision 생성 전에 split을 검증한다.
    split_rows = read_csv(paths["split_csv"])
    split_dev_ids = sorted(to_int(r.get("video_id")) for r in split_rows if r.get("split") == "train")
    if split_dev_ids != DEV_IDS:
        raise SystemExit(f"Development split mismatch: {split_dev_ids}")

    base_rows = load_base_rows(paths["base_output_root"])
    features = load_phase_features(paths["phase_features"])
    base_video_ids = sorted({to_int(r.get("video_id")) for r in base_rows})
    feature_video_ids = sorted({to_int(r.get("video_id")) for r in features.values()})
    if base_video_ids != DEV_IDS or feature_video_ids != DEV_IDS:
        raise SystemExit(f"Non-development rows found: base={base_video_ids}, features={feature_video_ids}")
    if len(base_rows) != len(features):
        log_lines.append(f"warning: base row count {len(base_rows)} != feature row count {len(features)}")

    variant_configs: List[Dict[str, Any]] = []
    for ocr_mode in OCR_MODES:
        for audio_mode in AUDIO_MODES:
            variant_configs.append({
                "variant_id": f"v2_1c_{ocr_mode}_{audio_mode}",
                "ocr_boost_mode": ocr_mode,
                "audio_mode": audio_mode,
                "base_candidate_id": cfg["base_candidate_id"],
                "quick_experiment_mode": True,
                **cfg["ocr_boost_modes"][ocr_mode],
            })

    write_json(paths["experiment_root"] / "variant_configs_v2_1c.json", variant_configs)
    write_csv(paths["experiment_root"] / "variant_configs_v2_1c.csv", variant_configs, ["variant_id", "ocr_boost_mode", "audio_mode", "base_candidate_id"])

    variant_outputs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    disclosure_audit: List[Dict[str, Any]] = []
    intro_audit: List[Dict[str, Any]] = []
    audio_audit: List[Dict[str, Any]] = []
    safety_audit: List[Dict[str, Any]] = []
    failure_log: List[Dict[str, Any]] = []

    for variant in variant_configs:
        try:
            out = apply_variant(variant, base_rows, features, cfg)
            variant_outputs[variant["variant_id"]] = out
            disclosure_audit.extend(out["disclosure_audit"])
            intro_audit.extend(out["intro_audit"])
            audio_audit.extend(out["audio_audit"])
            safety_audit.extend(out["safety_audit"])
            failure_log.append({"variant_id": variant["variant_id"], "config_failure_flag": False, "config_failure_reason": ""})
        except Exception as exc:  # quick experiment: fail a variant but keep log.
            failure_log.append({"variant_id": variant["variant_id"], "config_failure_flag": True, "config_failure_reason": repr(exc)})

    # scoring 전 audit을 먼저 만들며, 이 시점에는 actual label을 로드하지 않는다.
    write_csv(paths["experiment_root"] / "ocr_disclosure_boost_application_audit_v2_1c.csv", disclosure_audit)
    write_csv(paths["experiment_root"] / "intro_context_boost_application_audit_v2_1c.csv", intro_audit)
    write_csv(paths["experiment_root"] / "audio_ablation_application_audit_v2_1c.csv", audio_audit)
    write_csv(paths["experiment_root"] / "decision_feature_safety_audit_v2_1c.csv", safety_audit)
    write_csv(paths["experiment_root"] / "variant_failure_log_v2_1c.csv", failure_log)

    # variant decision이 만들어진 뒤에만 actual label을 로드한다.
    actual = load_actual_intervals(paths["actual_labels"], DEV_IDS)
    per_variant_metrics: List[Dict[str, Any]] = []
    per_video_metrics: List[Dict[str, Any]] = []
    config_lookup = {v["variant_id"]: v for v in variant_configs}
    for variant_id, out in variant_outputs.items():
        metrics, video_rows = score_variant(variant_id, out, actual, DEV_IDS)
        metrics.update({
            "ocr_boost_mode": config_lookup[variant_id]["ocr_boost_mode"],
            "audio_mode": config_lookup[variant_id]["audio_mode"],
        })
        per_variant_metrics.append(metrics)
        per_video_metrics.extend(video_rows)

    metric_by_id = {m["variant_id"]: m for m in per_variant_metrics}
    audio_comparisons: List[Dict[str, Any]] = []
    for ocr_mode in OCR_MODES:
        ids = {audio_mode: f"v2_1c_{ocr_mode}_{audio_mode}" for audio_mode in AUDIO_MODES}
        pairs = [("with_audio", "no_audio"), ("with_audio", "audio_end_only"), ("no_audio", "audio_end_only")]
        for left_mode, right_mode in pairs:
            left_id = ids[left_mode]
            right_id = ids[right_mode]
            if left_id not in metric_by_id or right_id not in metric_by_id:
                audio_comparisons.append({
                    "ocr_boost_mode": ocr_mode,
                    "comparison": f"{left_mode}_vs_{right_mode}",
                    "left_variant_id": left_id,
                    "right_variant_id": right_id,
                    "delta_recall": "",
                    "delta_precision_proxy": "",
                    "delta_false_positive_sec": "",
                    "delta_missed_actual_sec": "",
                    "interpretation": "comparison skipped because one variant failed",
                })
                continue
            left = metric_by_id[left_id]
            right = metric_by_id[right_id]
            audio_comparisons.append({
                "ocr_boost_mode": ocr_mode,
                "comparison": f"{left_mode}_vs_{right_mode}",
                "left_variant_id": left_id,
                "right_variant_id": right_id,
                "delta_recall": right["actual_overlap_recall"] - left["actual_overlap_recall"],
                "delta_precision_proxy": right["prediction_overlap_precision_proxy"] - left["prediction_overlap_precision_proxy"],
                "delta_false_positive_sec": right["false_positive_duration_sec"] - left["false_positive_duration_sec"],
                "delta_missed_actual_sec": right["missed_actual_duration_sec"] - left["missed_actual_duration_sec"],
                "interpretation": variant_interpretation(left, right, right_mode),
            })
    for metrics in per_variant_metrics:
        ocr_mode = metrics["ocr_boost_mode"]
        base = metric_by_id[f"v2_1c_{ocr_mode}_with_audio"]
        metrics["audio_ablation_delta_recall"] = metrics["actual_overlap_recall"] - base["actual_overlap_recall"]
        metrics["audio_ablation_delta_precision_proxy"] = metrics["prediction_overlap_precision_proxy"] - base["prediction_overlap_precision_proxy"]
        metrics["audio_ablation_delta_false_positive_sec"] = metrics["false_positive_duration_sec"] - base["false_positive_duration_sec"]
        match = next((r for r in audio_comparisons if r["ocr_boost_mode"] == ocr_mode and r["right_variant_id"] == metrics["variant_id"] and r["left_variant_id"] == base["variant_id"]), None)
        metrics["audio_ablation_interpretation"] = match["interpretation"] if match else "with_audio baseline for this OCR mode"

    ocr_comparisons: List[Dict[str, Any]] = []
    for audio_mode in AUDIO_MODES:
        base_id = f"v2_1c_baseline_ocr_{audio_mode}"
        for ocr_mode in [m for m in OCR_MODES if m != "baseline_ocr"]:
            cand_id = f"v2_1c_{ocr_mode}_{audio_mode}"
            if base_id not in metric_by_id or cand_id not in metric_by_id:
                ocr_comparisons.append({
                    "audio_mode": audio_mode,
                    "comparison": f"baseline_ocr_vs_{ocr_mode}",
                    "left_variant_id": base_id,
                    "right_variant_id": cand_id,
                    "delta_recall": "",
                    "delta_precision_proxy": "",
                    "delta_false_positive_sec": "",
                    "delta_boundary_quality": "",
                    "interpretation": "comparison skipped because one variant failed",
                })
                continue
            base = metric_by_id[base_id]
            cand = metric_by_id[cand_id]
            ocr_comparisons.append({
                "audio_mode": audio_mode,
                "comparison": f"baseline_ocr_vs_{ocr_mode}",
                "left_variant_id": base["variant_id"],
                "right_variant_id": cand["variant_id"],
                "delta_recall": cand["actual_overlap_recall"] - base["actual_overlap_recall"],
                "delta_precision_proxy": cand["prediction_overlap_precision_proxy"] - base["prediction_overlap_precision_proxy"],
                "delta_false_positive_sec": cand["false_positive_duration_sec"] - base["false_positive_duration_sec"],
                "delta_boundary_quality": cand["boundary_quality_score"] - base["boundary_quality_score"],
                "interpretation": "no metric movement" if cand["actual_overlap_recall"] == base["actual_overlap_recall"] and cand["prediction_overlap_precision_proxy"] == base["prediction_overlap_precision_proxy"] and cand["false_positive_duration_sec"] == base["false_positive_duration_sec"] else "metric movement observed",
            })

    if not per_variant_metrics:
        report = {
            "task": "state_machine_v2_1c_ocr_boost_audio_ablation_quick",
            "estimated_runtime_printed_first": True,
            "project_root": str(project_root),
            "experiment_root": str(paths["experiment_root"]),
            "backup_dir": str(backup_dir) if backup_dir else None,
            "base_candidate_id": cfg["base_candidate_id"],
            "quick_experiment_mode": True,
            "formal_rule_doc_created": False,
            "long_recommendation_note_created": False,
            "final_performance_claim": False,
            "variant_count": len(variant_configs),
            "successful_variant_count": 0,
            "failed_variant_count": len(variant_configs),
            "actual_label_used_for_decision": False,
            "actual_label_used_for_posthoc_scoring": True,
            "ready_for_viewer_review": False,
            "ready_for_fixed_rule_selection": False,
            "warnings": [],
            "errors": ["all variants failed before scoring"],
        }
        write_json(paths["report"], report)
        paths["run_log"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        raise SystemExit("all variants failed before scoring")

    sorted_by_balanced = sorted(per_variant_metrics, key=lambda m: m["balanced_objective_score"], reverse=True)
    best_balanced = sorted_by_balanced[0]
    best_precision = sorted(per_variant_metrics, key=lambda m: m["prediction_overlap_precision_proxy"], reverse=True)[0]
    best_recall = sorted(per_variant_metrics, key=lambda m: m["actual_overlap_recall"], reverse=True)[0]
    best_low_over = sorted(per_variant_metrics, key=lambda m: (m["false_positive_duration_sec"], -m["actual_overlap_recall"]))[0]
    best_no_audio = sorted([m for m in per_variant_metrics if m["audio_mode"] == "no_audio"], key=lambda m: m["balanced_objective_score"], reverse=True)[0]
    best_audio_end = sorted([m for m in per_variant_metrics if m["audio_mode"] == "audio_end_only"], key=lambda m: m["balanced_objective_score"], reverse=True)[0]
    best_ocr_boost = sorted([m for m in per_variant_metrics if m["ocr_boost_mode"] != "baseline_ocr"], key=lambda m: m["balanced_objective_score"], reverse=True)[0]
    top_variant_ids = []
    for m in [best_balanced, best_no_audio, best_audio_end, best_ocr_boost]:
        if m["variant_id"] not in top_variant_ids:
            top_variant_ids.append(m["variant_id"])
    top_rows = []
    for label, m in [
        ("best_balanced", best_balanced), ("best_precision", best_precision), ("best_recall", best_recall),
        ("best_low_overprediction", best_low_over), ("best_no_audio", best_no_audio),
        ("best_audio_end_only", best_audio_end), ("best_ocr_boost", best_ocr_boost),
    ]:
        out = dict(m)
        out["top_category"] = label
        top_rows.append(out)

    write_csv(paths["experiment_root"] / "per_variant_metrics_v2_1c.csv", per_variant_metrics)
    write_csv(paths["experiment_root"] / "per_variant_video_metrics_v2_1c.csv", per_video_metrics)
    write_csv(paths["experiment_root"] / "audio_ablation_comparison_v2_1c.csv", audio_comparisons)
    write_csv(paths["experiment_root"] / "ocr_boost_comparison_v2_1c.csv", ocr_comparisons)
    write_csv(paths["experiment_root"] / "top_variants_v2_1c.csv", top_rows)

    # 선택된 top variant에 대해서만 row-level output을 저장한다.
    top_root = paths["experiment_root"] / "top_variant_outputs"
    for variant_id in top_variant_ids[:4]:
        out = variant_outputs[variant_id]
        dest = top_root / variant_id
        write_csv(dest / "predictions.csv", out["predictions"])
        write_csv(dest / "review_only_candidates.csv", out["review_only_candidates"])
        write_csv(dest / "overprediction_pruned_review_candidates.csv", out["overprediction_pruned_review_candidates"])
        write_csv(dest / "open_candidates.csv", out["open_candidates"])
        write_csv(dest / "budget_guard_events.csv", out["budget_guard_events"])
        write_csv(dest / "trace_sample.csv", out["trace_sample"])
        video_summary = []
        for vid in DEV_IDS:
            video_summary.append({
                "variant_id": variant_id,
                "video_id": vid,
                "prediction_count": sum(1 for r in out["predictions"] if to_int(r.get("video_id")) == vid),
                "review_only_count": sum(1 for r in out["review_only_candidates"] if to_int(r.get("video_id")) == vid),
                "overprediction_pruned_review_count": sum(1 for r in out["overprediction_pruned_review_candidates"] if to_int(r.get("video_id")) == vid),
                "open_count": sum(1 for r in out["open_candidates"] if to_int(r.get("video_id")) == vid),
            })
        write_csv(dest / "video_summary.csv", video_summary)

    baseline = metric_by_id["v2_1c_baseline_ocr_with_audio"]
    best = best_balanced
    meaningful = (
        best["balanced_objective_score"] > baseline["balanced_objective_score"] + 0.01 or
        (best["false_positive_duration_sec"] < baseline["false_positive_duration_sec"] - 30 and best["actual_overlap_recall"] >= baseline["actual_overlap_recall"] - 0.01)
    )
    no_audio_comps = [c for c in audio_comparisons if c["comparison"] == "with_audio_vs_no_audio"]
    end_only_comps = [c for c in audio_comparisons if c["comparison"] == "with_audio_vs_audio_end_only"]
    if all(abs(c["delta_recall"]) < 0.01 and abs(c["delta_precision_proxy"]) < 0.01 and abs(c["delta_false_positive_sec"]) < 30 for c in no_audio_comps + end_only_comps):
        audio_interpretation = "audio contribution appears low across variants"
    elif any(c["delta_false_positive_sec"] < -30 and c["delta_recall"] > -0.02 for c in end_only_comps):
        audio_interpretation = "audio_end_only may reduce false positives while keeping recall"
    elif any(c["delta_recall"] < -0.03 for c in no_audio_comps):
        audio_interpretation = "audio may help recall for at least one OCR mode"
    else:
        audio_interpretation = "audio ablation effects are mixed or small"
    ocr_best_delta = max(ocr_comparisons, key=lambda c: c["delta_recall"] + c["delta_precision_proxy"] - max(0.0, c["delta_false_positive_sec"] / 10000.0))
    if abs(ocr_best_delta["delta_recall"]) < 0.01 and abs(ocr_best_delta["delta_precision_proxy"]) < 0.01 and abs(ocr_best_delta["delta_false_positive_sec"]) < 30:
        ocr_interpretation = "OCR boost does not materially change Development Set metrics in quick mode"
    else:
        ocr_interpretation = f"largest OCR boost movement: {ocr_best_delta['comparison']} under {ocr_best_delta['audio_mode']}"

    viewer_entries = []
    for variant_id in top_variant_ids[:4]:
        viewer_entries.append({
            "version_id": variant_id,
            "display_name": variant_id.replace("v2_1c_", "v2.1c quick - "),
            "experiment_id": "state_machine_v2_1c_ocr_boost_audio_ablation_quick",
            "base_version": cfg["base_candidate_id"],
            "split_scope": "development_only",
            "original_split_v2_4": "train",
            "split_role_v2_5": "development",
            "evaluation_subset_v2_5": "none",
            "is_experimental_config": True,
            "quick_experiment_mode": True,
            "final_rule_freeze": False,
            "recommended_for_viewer_review": True,
            "prediction_csv": str((top_root / variant_id / "predictions.csv").relative_to(project_root)),
            "review_only_csv": str((top_root / variant_id / "review_only_candidates.csv").relative_to(project_root)),
            "overprediction_pruned_review_csv": str((top_root / variant_id / "overprediction_pruned_review_candidates.csv").relative_to(project_root)),
            "open_candidate_csv": str((top_root / variant_id / "open_candidates.csv").relative_to(project_root)),
            "trace_sample_csv": str((top_root / variant_id / "trace_sample.csv").relative_to(project_root)),
            "budget_guard_events_csv": str((top_root / variant_id / "budget_guard_events.csv").relative_to(project_root)),
            "video_summary_csv": str((top_root / variant_id / "video_summary.csv").relative_to(project_root)),
            "experiment_summary": str(paths["short_summary"].relative_to(project_root)),
            "experiment_report": str(paths["report"].relative_to(project_root)),
        })
    write_json(paths["viewer_patch"], {
        "patch_id": "state_machine_viewer_registry_patch_v2_1c_ocr_boost_audio_ablation_top_variants",
        "patch_proposal_only": True,
        "do_not_modify_existing_registry": True,
        "quick_experiment_mode": True,
        "final_rule_freeze": False,
        "recommended_for_viewer_review_before_fixed_rule_adoption": True,
        "registry_entries_to_add": viewer_entries,
    })

    successful = sum(1 for r in failure_log if not boolish(r["config_failure_flag"]))
    warnings: List[str] = []
    audio_fallback_count = sum(1 for r in audio_audit if r.get("audio_score_source") == "missing_audio_score_fallback_zero")
    audio_explicit_count = len(audio_audit) - audio_fallback_count
    if audio_fallback_count:
        warnings.append(f"audio fallback was broad: {audio_fallback_count}/{len(audio_audit)} audit rows used missing_audio_score_fallback_zero; explicit audio rows={audio_explicit_count}")
    ocr_all_zero_delta = all(
        c.get("delta_recall") in (0, 0.0) and c.get("delta_precision_proxy") in (0, 0.0) and c.get("delta_false_positive_sec") in (0, 0.0)
        for c in ocr_comparisons
        if c.get("delta_recall") != ""
    )
    if ocr_all_zero_delta:
        warnings.append("all OCR boost comparisons had zero metric movement; best_ocr_boost is a tie label, not evidence of improvement")
    if not meaningful:
        warnings.append("no meaningful improvement detected in quick Development Set comparison")
    decision_forbidden = sorted({pat for row in safety_audit for pat in json.loads(row["forbidden_decision_columns_found"])})
    errors: List[str] = []
    if decision_forbidden:
        errors.append("forbidden decision feature columns detected: " + ",".join(decision_forbidden))
    if successful == 0:
        errors.append("all variants failed")

    report = {
        "task": "state_machine_v2_1c_ocr_boost_audio_ablation_quick",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "estimated_runtime_printed_first": True,
        "project_root": str(project_root),
        "experiment_root": str(paths["experiment_root"]),
        "backup_dir": str(backup_dir) if backup_dir else None,
        "base_candidate_id": cfg["base_candidate_id"],
        "quick_experiment_mode": True,
        "formal_rule_doc_created": False,
        "long_recommendation_note_created": False,
        "final_performance_claim": False,
        "variant_count": len(variant_configs),
        "successful_variant_count": successful,
        "failed_variant_count": len(variant_configs) - successful,
        "ocr_boost_modes": OCR_MODES,
        "audio_modes": AUDIO_MODES,
        "best_balanced_variant_id": best_balanced["variant_id"],
        "best_precision_variant_id": best_precision["variant_id"],
        "best_recall_variant_id": best_recall["variant_id"],
        "best_low_overprediction_variant_id": best_low_over["variant_id"],
        "best_no_audio_variant_id": best_no_audio["variant_id"],
        "best_audio_end_only_variant_id": best_audio_end["variant_id"],
        "best_ocr_boost_variant_id": best_ocr_boost["variant_id"],
        "top_variant_ids_with_row_outputs": top_variant_ids[:4],
        "audio_ablation_interpretation": audio_interpretation,
        "ocr_boost_interpretation": ocr_interpretation,
        "ocr_boost_all_zero_metric_delta": ocr_all_zero_delta,
        "best_ocr_boost_is_tie_label_only": ocr_all_zero_delta,
        "audio_fallback_missing_score_rows": audio_fallback_count,
        "audio_explicit_score_rows": audio_explicit_count,
        "meaningful_improvement_detected": meaningful,
        "v2_1b_light_baseline_recall": baseline["actual_overlap_recall"],
        "best_variant_recall": best["actual_overlap_recall"],
        "v2_1b_light_baseline_precision_proxy": baseline["prediction_overlap_precision_proxy"],
        "best_variant_precision_proxy": best["prediction_overlap_precision_proxy"],
        "v2_1b_light_baseline_false_positive_sec": baseline["false_positive_duration_sec"],
        "best_variant_false_positive_sec": best["false_positive_duration_sec"],
        "v2_1b_light_baseline_missed_actual_sec": baseline["missed_actual_duration_sec"],
        "best_variant_missed_actual_sec": best["missed_actual_duration_sec"],
        "actual_label_used_for_decision": False,
        "actual_label_used_for_posthoc_scoring": True,
        "actual_labels_loaded_after_variant_decisions": True,
        "plus5_actual_label_phase_used_for_decision": False,
        "OCR_extraction_executed": False,
        "audio_extraction_executed": False,
        "scene_extraction_executed": False,
        "Extended_Evaluation_processed": False,
        "Diagnostic_Subset_processed": False,
        "Pure_Test_processed": False,
        "validation_output_count": 0,
        "test_row_level_output_count": 0,
        "old_project_modified": False,
        "input_files_modified": False,
        "v1_4_preserved": True,
        "v2_0_preserved": True,
        "v2_1a_preserved": True,
        "v2_1b_preserved": True,
        "viewer_registry_patch_path": str(paths["viewer_patch"]),
        "ready_for_viewer_review": successful > 0,
        "ready_for_fixed_rule_selection": False,
        "decision_feature_forbidden_patterns_found": decision_forbidden,
        "metric_files": {
            "per_variant_metrics": str(paths["experiment_root"] / "per_variant_metrics_v2_1c.csv"),
            "per_variant_video_metrics": str(paths["experiment_root"] / "per_variant_video_metrics_v2_1c.csv"),
            "audio_ablation_comparison": str(paths["experiment_root"] / "audio_ablation_comparison_v2_1c.csv"),
            "ocr_boost_comparison": str(paths["experiment_root"] / "ocr_boost_comparison_v2_1c.csv"),
            "top_variants": str(paths["experiment_root"] / "top_variants_v2_1c.csv"),
        },
        "warnings": warnings,
        "errors": errors,
        "sub_agent_validations": [],
    }

    summary_lines = [
        "# v2.1c OCR Boost / Audio Ablation Quick Summary",
        "",
        "## 1. 실험 목적",
        "v2.1b OCR phase transition light 후보를 base로 OCR start 단서 강화와 audio ablation 12개 조합을 Development Set에서 빠르게 비교했다. fixed rule 확정이 아니다.",
        "",
        "## 2. Variant 구성",
        "- OCR boost: baseline_ocr, disclosure_boost, intro_context_boost, disclosure_intro_combo",
        "- Audio mode: with_audio, no_audio, audio_end_only",
        "- 총 12개 variant",
        "",
        "## 3. 핵심 결과 표",
        "| category | variant | recall | precision_proxy | false_positive_sec | missed_actual_sec | balanced |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, m in [("baseline", baseline), ("best_balanced", best_balanced), ("best_no_audio", best_no_audio), ("best_audio_end_only", best_audio_end), ("best_ocr_boost", best_ocr_boost)]:
        summary_lines.append(f"| {label} | `{m['variant_id']}` | {m['actual_overlap_recall']:.4f} | {m['prediction_overlap_precision_proxy']:.4f} | {m['false_positive_duration_sec']:.1f} | {m['missed_actual_duration_sec']:.1f} | {m['balanced_objective_score']:.4f} |")
    summary_lines.extend([
        "",
        "## 4. Audio 제거 실험 해석",
        audio_interpretation,
        "",
        "## 5. OCR boost 실험 해석",
        ocr_interpretation,
        "",
        "## 6. 의미 있는 개선 여부",
        f"meaningful_improvement_detected={str(meaningful).lower()}",
        "",
        "## 7. 다음 단계",
        "viewer review 대상으로만 사용한다. 의미 있는 개선이 사람 검토에서도 확인될 때만 정식 rule 문서와 recommendation note를 별도 생성한다.",
        "",
        "## 8. Safety 요약",
        "actual label은 post-hoc scoring에만 사용했다. OCR/audio/scene extraction은 실행하지 않았다. Test Set은 처리하지 않았다. formal rule doc과 long recommendation note는 생성하지 않았다.",
    ])
    paths["short_summary"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_json(paths["report"], report)

    # latest bundle에는 작은 flat file만 포함한다.
    if paths["latest_bundle"].exists():
        shutil.rmtree(paths["latest_bundle"])
    paths["latest_bundle"].mkdir(parents=True, exist_ok=True)
    bundle_sources = [
        (paths["short_summary"], "short_summary.md"),
        (paths["report"], "report.json"),
        (paths["run_log"], "run_log.txt"),
        (config_path, "state_machine_v2_1c_ocr_boost_audio_ablation_quick_config.json"),
        (paths["experiment_root"] / "variant_configs_v2_1c.csv", "variant_configs_v2_1c.csv"),
        (paths["experiment_root"] / "per_variant_metrics_v2_1c.csv", "per_variant_metrics_v2_1c.csv"),
        (paths["experiment_root"] / "audio_ablation_comparison_v2_1c.csv", "audio_ablation_comparison_v2_1c.csv"),
        (paths["experiment_root"] / "ocr_boost_comparison_v2_1c.csv", "ocr_boost_comparison_v2_1c.csv"),
        (paths["experiment_root"] / "top_variants_v2_1c.csv", "top_variants_v2_1c.csv"),
        (paths["experiment_root"] / "variant_failure_log_v2_1c.csv", "variant_failure_log_v2_1c.csv"),
        (paths["experiment_root"] / "decision_feature_safety_audit_v2_1c.csv", "decision_feature_safety_audit_v2_1c.csv"),
        (paths["viewer_patch"], "state_machine_viewer_registry_patch_v2_1c_ocr_boost_audio_ablation_top_variants.json"),
    ]
    # 복사하기 전에 log를 먼저 쓴다
    log_lines.append(f"[{datetime.now().isoformat(timespec='seconds')}] wrote quick outputs")
    paths["run_log"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    for src, name in bundle_sources:
        if src.exists():
            shutil.copy2(src, paths["latest_bundle"] / name)
    audio_source_counts = defaultdict(int)
    audio_mode_counts = defaultdict(int)
    disclosure_by_variant = defaultdict(int)
    intro_by_variant = defaultdict(int)
    for r in audio_audit:
        audio_source_counts[r.get("audio_score_source", "")] += 1
        audio_mode_counts[r.get("audio_mode", "")] += 1
    for r in disclosure_audit:
        if boolish(r.get("disclosure_boost_applied")):
            disclosure_by_variant[r.get("variant_id", "")] += 1
    for r in intro_audit:
        if boolish(r.get("intro_context_boost_applied")):
            intro_by_variant[r.get("variant_id", "")] += 1
    audit_summary = {
        "disclosure_audit_rows": len(disclosure_audit),
        "intro_audit_rows": len(intro_audit),
        "audio_audit_rows": len(audio_audit),
        "safety_audit_rows": len(safety_audit),
        "disclosure_boost_applied_rows": sum(disclosure_by_variant.values()),
        "intro_boost_applied_rows": sum(intro_by_variant.values()),
        "disclosure_boost_applied_rows_by_variant": dict(sorted(disclosure_by_variant.items())),
        "intro_boost_applied_rows_by_variant": dict(sorted(intro_by_variant.items())),
        "audio_score_source_counts": dict(sorted(audio_source_counts.items())),
        "audio_mode_counts": dict(sorted(audio_mode_counts.items())),
        "no_audio_rows": sum(1 for r in audio_audit if r.get("audio_mode") == "no_audio"),
        "audio_end_only_rows": sum(1 for r in audio_audit if r.get("audio_mode") == "audio_end_only"),
        "forbidden_decision_patterns_found": decision_forbidden,
    }
    write_json(paths["latest_bundle"] / "audit_summary_v2_1c.json", audit_summary)
    forbidden_terms = ["media/", "video/", "frame/", "cache/", "model/", "raw_video", "raw video", "proxy", "checkpoint"]
    findings = []
    for path in paths["latest_bundle"].rglob("*"):
        if not path.is_file():
            continue
        low = str(path).lower()
        for term in forbidden_terms:
            if term in low:
                findings.append({"path": str(path), "reason": f"forbidden_term:{term}"})
    scan = {"clean": len(findings) == 0, "finding_count": len(findings), "findings": findings}
    write_json(paths["latest_bundle"] / "latest_bundle_forbidden_scan.json", scan)
    report["latest_bundle"] = str(paths["latest_bundle"])
    report["latest_bundle_forbidden_scan_clean"] = scan["clean"]
    write_json(paths["report"], report)
    shutil.copy2(paths["report"], paths["latest_bundle"] / "report.json")

    print("Final Summary:")
    final_keys = [
        "task", "estimated_runtime_printed_first", "project_root", "experiment_root", "backup_dir", "base_candidate_id",
        "quick_experiment_mode", "formal_rule_doc_created", "long_recommendation_note_created", "variant_count",
        "successful_variant_count", "failed_variant_count", "ocr_boost_modes", "audio_modes", "best_balanced_variant_id",
        "best_precision_variant_id", "best_recall_variant_id", "best_low_overprediction_variant_id", "best_no_audio_variant_id",
        "best_audio_end_only_variant_id", "audio_ablation_interpretation", "ocr_boost_interpretation", "meaningful_improvement_detected",
        "v2_1b_light_baseline_recall", "best_variant_recall", "v2_1b_light_baseline_precision_proxy", "best_variant_precision_proxy",
        "v2_1b_light_baseline_false_positive_sec", "best_variant_false_positive_sec", "v2_1b_light_baseline_missed_actual_sec",
        "best_variant_missed_actual_sec", "actual_label_used_for_decision", "actual_label_used_for_posthoc_scoring",
        "plus5_actual_label_phase_used_for_decision", "OCR_extraction_executed", "audio_extraction_executed", "scene_extraction_executed",
        "Extended_Evaluation_processed", "Diagnostic_Subset_processed", "Pure_Test_processed", "old_project_modified", "input_files_modified",
        "viewer_registry_patch_path", "latest_bundle", "ready_for_viewer_review", "ready_for_fixed_rule_selection", "warnings", "errors",
    ]
    for key in final_keys:
        print(f"- {key}: {report.get(key)}")


if __name__ == "__main__":
    main()
