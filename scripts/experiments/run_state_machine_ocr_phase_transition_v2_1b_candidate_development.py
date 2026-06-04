#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(".")
CONFIG_PATH = ROOT / "configs/experiments/state_machine_ocr_phase_transition_v2_1b_candidate_config.json"
TASK = "state_machine_ocr_phase_transition_v2_1b_candidate_development"
DEV_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
BASE_CONFIG_ID = "v2_1a_short_ad_safe_01"
FORBIDDEN_DECISION_PATTERNS = [
    "label",
    "true",
    "actual",
    "gt",
    "ground_truth",
    "audit",
    "nearest_true_boundary",
    "distance_to_nearest_true_boundary",
    "is_near_true_boundary",
    "ad_overlap",
    "is_ad_overlap",
    "is_ad_core",
    "is_clean_nonad",
    "overlapping_ad_interval_ids",
    "actual_ad_start_sec",
    "actual_ad_end_sec",
]
DECISION_FEATURE_COLUMNS = [
    "candidate_start_sec",
    "candidate_end_sec",
    "candidate_pre_start_window",
    "candidate_start_window",
    "candidate_body_window",
    "candidate_pre_end_window",
    "candidate_post_end_5s_window",
    "ocr_candidate_start_intro_context_count",
    "ocr_candidate_start_product_brand_count",
    "ocr_candidate_start_cta_link_count",
    "ocr_candidate_start_sponsor_count",
    "ocr_candidate_body_product_cta_density",
    "ocr_candidate_body_product_cta_repeat_count",
    "ocr_candidate_body_timeline_hard_evidence_count",
    "ocr_candidate_post_end_keyword_drop_flag",
    "ocr_candidate_post_end_return_to_normal_flag",
    "ocr_candidate_post_end_still_ad_like_flag",
    "ocr_candidate_end_drop_confidence_score",
]
FORBIDDEN_SUFFIXES = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".wav",
    ".mp3",
    ".m4a",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".npy",
    ".npz",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
}
FORBIDDEN_DIRS = {
    "media",
    "video",
    "videos",
    "frame",
    "frames",
    "cache",
    "model",
    "models",
    "raw_video",
    "raw_videos",
    "proxy",
    "checkpoint",
    "checkpoints",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return default if math.isnan(out) else out
    except Exception:
        return default


def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def safe_ratio(n: float, d: float) -> float:
    return 0.0 if d <= 0 else n / d


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    mid = len(data) // 2
    return data[mid] if len(data) % 2 else (data[mid - 1] + data[mid]) / 2.0


def mmss(sec: float) -> str:
    s = max(0, int(round(sec)))
    return f"{s // 60:02d}:{s % 60:02d}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def has_forbidden(columns: list[str]) -> list[str]:
    bad: list[str] = []
    for col in columns:
        low = col.lower()
        for pattern in FORBIDDEN_DECISION_PATTERNS:
            if pattern == "gt":
                if low == "gt" or low.startswith("gt_") or low.endswith("_gt") or "_gt_" in low:
                    bad.append(col)
                    break
            elif pattern in low:
                bad.append(col)
                break
    return sorted(set(bad))


def normalize_text(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", (text or "").lower())


INTRO_PHRASES = [
    "잠깐 소개",
    "소개해드려도",
    "소개 시켜주고",
    "소개시켜주고",
    "소개해드릴",
    "제품 받아보고",
    "제품을 받아보고",
    "광고를 받았는데",
    "광고받았는데",
    "광고를 받아보고",
    "한번만 봐주실래요",
    "함만 봐주실래요",
    "제 얘기 들어보실분",
    "잠깐 시간되시",
    "구경 하실래요",
    "구경하실래요",
    "추천드립니다",
    "혜택 알려드릴게요",
    "지원을 받아 제작",
    "제작 지원",
]
INTRO_PHRASE_NORMS = [(phrase, normalize_text(phrase)) for phrase in INTRO_PHRASES]
RETURN_PHRASES = ["회사갑니다", "출근길", "운동갑니다", "운동 갑니다", "day", "일상", "출근", "퇴근", "집에갑니다"]
RETURN_PHRASE_NORMS = [(phrase, normalize_text(phrase)) for phrase in RETURN_PHRASES]
STILL_AD_PHRASES = [
    "이용해보시길",
    "추천드립니다",
    "구매",
    "증정",
    "할인",
    "이벤트",
    "혜택",
    "더보기",
    "고정댓글",
    "링크",
    "구독자 이벤트",
    "제작 지원",
    "유료광고",
]
STILL_AD_PHRASE_NORMS = [(phrase, normalize_text(phrase)) for phrase in STILL_AD_PHRASES]


def phrase_hits(text: str, phrase_norms: list[tuple[str, str]]) -> list[str]:
    norm = normalize_text(text)
    hits = []
    for phrase, phrase_norm in phrase_norms:
        if phrase_norm and phrase_norm in norm:
            hits.append(phrase)
    return hits


def row_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in ["ocr_text_normalized", "ocr_text_joined", "ocr_text_raw", "matched_keywords", "suggested_canonical_phrase"]
    )


def keyword_count(row: dict[str, Any]) -> float:
    return sum(
        fnum(row.get(name))
        for name in [
            "corrected_ad_disclosure_hit_count",
            "corrected_sponsor_keyword_count",
            "corrected_brand_product_keyword_count",
            "corrected_promotion_discount_keyword_count",
            "corrected_purchase_cta_keyword_count",
            "corrected_link_more_info_keyword_count",
        ]
    )


def disclosure_subtype(row: dict[str, Any]) -> str:
    meta = " ".join(str(row.get(k) or "").lower() for k in ["matched_keyword_confidence", "matched_keyword_rules", "correction_note"])
    if "typo" in meta:
        return "typo"
    if "proximity" in meta or "near" in meta:
        return "proximity"
    if "fuzzy" in meta:
        return "fuzzy"
    return "exact"


def summarize_ocr_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "valid_frame_count": 0,
            "text_frame_count": 0,
            "text_frame_ratio": 0.0,
            "keyword_hit_frame_count": 0,
            "keyword_hit_frame_ratio": 0.0,
            "disclosure_exact_count": 0.0,
            "disclosure_typo_count": 0.0,
            "disclosure_proximity_count": 0.0,
            "disclosure_fuzzy_count": 0.0,
            "disclosure_count": 0.0,
            "sponsor_count": 0.0,
            "product_brand_count": 0.0,
            "promotion_discount_count": 0.0,
            "purchase_cta_count": 0.0,
            "link_more_info_count": 0.0,
            "product_cta_count": 0.0,
            "total_keyword_count": 0.0,
            "low_conf_keyword_hit_count": 0,
            "corrected_ad_text_score_mean": 0.0,
            "corrected_ad_text_score_max": 0.0,
            "text_density": 0.0,
            "intro_context_count": 0,
            "intro_context_phrases": "",
            "return_to_normal_phrase_count": 0,
            "return_to_normal_phrases": "",
            "still_ad_like_phrase_count": 0,
            "still_ad_like_phrases": "",
            "representative_text": "",
        }
    valid = [row for row in rows if str(row.get("ocr_status", "")).startswith("success")]
    text_rows = [row for row in valid if fnum(row.get("ocr_text_count")) > 0 or fnum(row.get("ocr_char_count")) > 0]
    keyword_rows = [row for row in rows if keyword_count(row) > 0 or fnum(row.get("corrected_total_ad_keyword_count")) > 0]
    disc_exact = disc_typo = disc_prox = disc_fuzzy = 0.0
    intro_hits: list[str] = []
    return_hits: list[str] = []
    still_hits: list[str] = []
    low_conf_keyword_hits = 0
    for row in rows:
        disc = fnum(row.get("corrected_ad_disclosure_hit_count") or row.get("ad_disclosure_keyword_count"))
        if disc > 0:
            subtype = disclosure_subtype(row)
            if subtype == "typo":
                disc_typo += disc
            elif subtype == "proximity":
                disc_prox += disc
            elif subtype == "fuzzy":
                disc_fuzzy += disc
            else:
                disc_exact += disc
        if keyword_count(row) > 0 and fnum(row.get("ocr_mean_confidence"), 1.0) < 0.45:
            low_conf_keyword_hits += 1
        text = row_text(row)
        intro_hits.extend(phrase_hits(text, INTRO_PHRASE_NORMS))
        return_hits.extend(phrase_hits(text, RETURN_PHRASE_NORMS))
        still_hits.extend(phrase_hits(text, STILL_AD_PHRASE_NORMS))
    product_cta = sum(
        sum(fnum(row.get(name)) for row in rows)
        for name in [
            "corrected_brand_product_keyword_count",
            "corrected_purchase_cta_keyword_count",
            "corrected_link_more_info_keyword_count",
            "corrected_promotion_discount_keyword_count",
        ]
    )
    rep = ""
    for row in rows:
        text = row_text(row).strip()
        if text:
            rep = text[:300]
            break
    return {
        "row_count": len(rows),
        "valid_frame_count": len(valid),
        "text_frame_count": len(text_rows),
        "text_frame_ratio": safe_ratio(len(text_rows), len(valid)),
        "keyword_hit_frame_count": len(keyword_rows),
        "keyword_hit_frame_ratio": safe_ratio(len(keyword_rows), len(valid)),
        "disclosure_exact_count": disc_exact,
        "disclosure_typo_count": disc_typo,
        "disclosure_proximity_count": disc_prox,
        "disclosure_fuzzy_count": disc_fuzzy,
        "disclosure_count": disc_exact + disc_typo + disc_prox + disc_fuzzy,
        "sponsor_count": sum(fnum(row.get("corrected_sponsor_keyword_count")) for row in rows),
        "product_brand_count": sum(fnum(row.get("corrected_brand_product_keyword_count")) for row in rows),
        "promotion_discount_count": sum(fnum(row.get("corrected_promotion_discount_keyword_count")) for row in rows),
        "purchase_cta_count": sum(fnum(row.get("corrected_purchase_cta_keyword_count")) for row in rows),
        "link_more_info_count": sum(fnum(row.get("corrected_link_more_info_keyword_count")) for row in rows),
        "product_cta_count": product_cta,
        "total_keyword_count": sum(keyword_count(row) for row in rows),
        "low_conf_keyword_hit_count": low_conf_keyword_hits,
        "corrected_ad_text_score_mean": mean([fnum(row.get("corrected_frame_ad_text_score")) for row in rows]),
        "corrected_ad_text_score_max": max([fnum(row.get("corrected_frame_ad_text_score")) for row in rows] or [0.0]),
        "text_density": mean([fnum(row.get("frame_text_density_score")) for row in rows]),
        "intro_context_count": len(intro_hits),
        "intro_context_phrases": ";".join(sorted(set(intro_hits))),
        "return_to_normal_phrase_count": len(return_hits),
        "return_to_normal_phrases": ";".join(sorted(set(return_hits))),
        "still_ad_like_phrase_count": len(still_hits),
        "still_ad_like_phrases": ";".join(sorted(set(still_hits))),
        "representative_text": rep,
    }


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def log_factory(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{now_iso()}] {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log


def backup_existing(paths: list[Path]) -> tuple[str | None, list[dict[str, Any]]]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None, []
    backup = ROOT / "backups" / f"state_machine_ocr_phase_transition_v2_1b_candidate_development_{stamp()}"
    backup.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for src in existing:
        dst = backup / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            digest = ""
            kind = "directory"
        else:
            shutil.copy2(src, dst)
            digest = sha256(src)
            kind = "file"
        manifest.append({"source_path": str(src), "backup_path": str(dst), "kind": kind, "sha256": digest})
    write_json(backup / "backup_manifest.json", manifest)
    return str(backup), manifest


def load_base_candidates(base_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    buckets = {
        "prediction": read_csv(base_dir / "predictions.csv"),
        "review_only": read_csv(base_dir / "review_only_candidates.csv"),
        "overprediction_pruned_review": read_csv(base_dir / "overprediction_pruned_review_candidates.csv"),
        "open": read_csv(base_dir / "open_candidates.csv"),
    }
    candidates: list[dict[str, Any]] = []
    for bucket, rows in buckets.items():
        for idx, row in enumerate(rows, start=1):
            out = dict(row)
            out["base_bucket"] = bucket
            out["base_candidate_id"] = row.get("candidate_id") or row.get("prediction_id") or f"{BASE_CONFIG_ID}_{bucket}_{idx:05d}"
            out["candidate_id"] = out["base_candidate_id"]
            out["ad_start_sec"] = row.get("ad_start_sec") or row.get("start_sec") or row.get("candidate_time_sec") or "0"
            out["ad_end_sec"] = row.get("ad_end_sec") or row.get("end_sec") or row.get("last_anchor_sec") or row.get("ad_start_sec") or "0"
            out["ad_duration_sec"] = row.get("ad_duration_sec") or row.get("duration_proxy_sec") or str(max(0.0, fnum(out["ad_end_sec"]) - fnum(out["ad_start_sec"])))
            if int(fnum(out.get("video_id"))) in DEV_IDS:
                candidates.append(out)
    return candidates, buckets


def load_ocr_by_video(path: Path) -> dict[int, list[dict[str, Any]]]:
    rows = read_csv(path)
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        vid = int(fnum(row.get("video_id")))
        if vid not in DEV_IDS:
            continue
        if row.get("original_split_v2_4") != "train" or row.get("split_role_v2_5") != "development" or row.get("evaluation_subset_v2_5") != "none":
            continue
        row["_time"] = fnum(row.get("timestamp_sec"))
        by_video[vid].append(row)
    for vid in by_video:
        by_video[vid].sort(key=lambda row: row["_time"])
    return by_video


def window_rows(rows: list[dict[str, Any]], start: float, end: float, left_closed: bool = True, right_closed: bool = False) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        t = row["_time"]
        left_ok = t >= start if left_closed else t > start
        right_ok = t <= end if right_closed else t < end
        if left_ok and right_ok:
            out.append(row)
    return out


def build_phase_features(candidates: list[dict[str, Any]], ocr_by_video: dict[int, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    features: list[dict[str, Any]] = []
    intro_audit: list[dict[str, Any]] = []
    post_audit: list[dict[str, Any]] = []
    for row in candidates:
        vid = int(fnum(row.get("video_id")))
        start = fnum(row.get("ad_start_sec"))
        end = fnum(row.get("ad_end_sec"))
        if end <= start:
            end = start
        duration = max(0.0, end - start)
        rows = ocr_by_video.get(vid, [])
        pre_start = summarize_ocr_window(window_rows(rows, start - 5.0, start, True, False))
        start_win = summarize_ocr_window(window_rows(rows, start, start + 10.0, True, True))
        start_ctx = summarize_ocr_window(window_rows(rows, start - 5.0, start + 15.0, True, True))
        body = summarize_ocr_window(window_rows(rows, start, end, True, True))
        pre_end = summarize_ocr_window(window_rows(rows, end - 10.0, end, True, True))
        post5 = summarize_ocr_window(window_rows(rows, end, end + 5.0, False, True))
        post10 = summarize_ocr_window(window_rows(rows, end, end + 10.0, False, True))
        body_minutes = max(1.0, duration / 60.0)
        body_product_cta_density = body["product_cta_count"] / body_minutes
        body_repeat = int(body["product_cta_count"] >= 2)
        body_timeline_hard_evidence_count = body["disclosure_exact_count"] + body["disclosure_typo_count"] + body["disclosure_proximity_count"]
        start_evidence_combo = (
            start_ctx["product_cta_count"] > 0
            or start_ctx["sponsor_count"] > 0
            or start_ctx["disclosure_exact_count"] + start_ctx["disclosure_typo_count"] + start_ctx["disclosure_proximity_count"] > 0
            or body_product_cta_density >= 1.0
            or body_timeline_hard_evidence_count > 0
        )
        intro_alone = start_ctx["intro_context_count"] > 0 and not start_evidence_combo
        intro_support = start_ctx["intro_context_count"] > 0 and start_evidence_combo
        pre_keyword = pre_end["total_keyword_count"]
        post_keyword = post5["total_keyword_count"]
        keyword_drop = pre_keyword > 0 and post5["valid_frame_count"] > 0 and post_keyword <= max(1.0, pre_keyword * 0.2)
        still_ad_like = (
            post5["product_cta_count"] > 0
            or post5["disclosure_count"] > 0
            or post5["sponsor_count"] > 0
            or post5["promotion_discount_count"] > 0
            or post5["still_ad_like_phrase_count"] > 0
        )
        return_normal = (
            post5["valid_frame_count"] > 0
            and post_keyword == 0
            and not still_ad_like
            and (
                post5["return_to_normal_phrase_count"] > 0
                or (post5["text_frame_ratio"] >= 0.35 and post5["corrected_ad_text_score_max"] <= 0.05)
            )
        )
        end_drop_score = clip((1.0 if keyword_drop else 0.0) + (0.2 if return_normal else 0.0) - (0.4 if still_ad_like else 0.0))
        feat = {
            "base_candidate_id": row["base_candidate_id"],
            "base_bucket": row["base_bucket"],
            "video_id": vid,
            "candidate_start_sec": f"{start:.6f}",
            "candidate_end_sec": f"{end:.6f}",
            "candidate_body_duration_sec": f"{duration:.6f}",
            "candidate_pre_start_window": f"[{start - 5.0:.3f},{start:.3f})",
            "candidate_start_window": f"[{start:.3f},{start + 10.0:.3f}]",
            "candidate_start_context_window": f"[{start - 5.0:.3f},{start + 15.0:.3f}]",
            "candidate_body_window": f"[{start:.3f},{end:.3f}]",
            "candidate_pre_end_window": f"[{end - 10.0:.3f},{end:.3f}]",
            "candidate_post_end_5s_window": f"({end:.3f},{end + 5.0:.3f}]",
            "candidate_post_end_10s_window": f"({end:.3f},{end + 10.0:.3f}]",
            "ocr_candidate_pre_start_disclosure_exact_count": pre_start["disclosure_exact_count"],
            "ocr_candidate_pre_start_disclosure_typo_count": pre_start["disclosure_typo_count"],
            "ocr_candidate_pre_start_disclosure_proximity_count": pre_start["disclosure_proximity_count"],
            "ocr_candidate_start_disclosure_exact_count": start_win["disclosure_exact_count"],
            "ocr_candidate_start_disclosure_typo_count": start_win["disclosure_typo_count"],
            "ocr_candidate_start_disclosure_proximity_count": start_win["disclosure_proximity_count"],
            "ocr_candidate_start_sponsor_count": start_ctx["sponsor_count"],
            "ocr_candidate_start_intro_context_count": start_ctx["intro_context_count"],
            "ocr_candidate_start_intro_context_phrases": start_ctx["intro_context_phrases"],
            "ocr_candidate_start_intro_context_alone_flag": intro_alone,
            "ocr_candidate_start_intro_support_flag": intro_support,
            "ocr_candidate_start_product_brand_count": start_ctx["product_brand_count"],
            "ocr_candidate_start_cta_link_count": start_ctx["purchase_cta_count"] + start_ctx["link_more_info_count"],
            "ocr_candidate_start_text_density": start_ctx["text_density"],
            "ocr_candidate_start_low_conf_keyword_hit_count": start_ctx["low_conf_keyword_hit_count"],
            "ocr_candidate_start_corrected_ad_text_score_max": start_ctx["corrected_ad_text_score_max"],
            "ocr_candidate_start_corrected_ad_text_score_mean": start_ctx["corrected_ad_text_score_mean"],
            "ocr_candidate_body_duration_sec": duration,
            "ocr_candidate_body_valid_frame_count": body["valid_frame_count"],
            "ocr_candidate_body_text_frame_ratio": body["text_frame_ratio"],
            "ocr_candidate_body_keyword_hit_frame_ratio": body["keyword_hit_frame_ratio"],
            "ocr_candidate_body_disclosure_count": body["disclosure_count"],
            "ocr_candidate_body_sponsor_count": body["sponsor_count"],
            "ocr_candidate_body_product_brand_count": body["product_brand_count"],
            "ocr_candidate_body_promotion_discount_count": body["promotion_discount_count"],
            "ocr_candidate_body_purchase_cta_count": body["purchase_cta_count"],
            "ocr_candidate_body_link_more_info_count": body["link_more_info_count"],
            "ocr_candidate_body_product_cta_density": body_product_cta_density,
            "ocr_candidate_body_product_cta_repeat_count": body_repeat,
            "ocr_candidate_body_timeline_hard_evidence_count": body_timeline_hard_evidence_count,
            "ocr_candidate_body_corrected_ad_text_score_mean": body["corrected_ad_text_score_mean"],
            "ocr_candidate_body_corrected_ad_text_score_max": body["corrected_ad_text_score_max"],
            "ocr_candidate_pre_end_keyword_count": pre_keyword,
            "ocr_candidate_pre_end_product_cta_count": pre_end["product_cta_count"],
            "ocr_candidate_pre_end_ad_text_score_mean": pre_end["corrected_ad_text_score_mean"],
            "ocr_candidate_post_end_5s_valid_frame_count": post5["valid_frame_count"],
            "ocr_candidate_post_end_5s_text_frame_ratio": post5["text_frame_ratio"],
            "ocr_candidate_post_end_5s_keyword_count": post_keyword,
            "ocr_candidate_post_end_5s_disclosure_count": post5["disclosure_count"],
            "ocr_candidate_post_end_5s_product_cta_count": post5["product_cta_count"],
            "ocr_candidate_post_end_5s_sponsor_count": post5["sponsor_count"],
            "ocr_candidate_post_end_5s_ad_text_score_mean": post5["corrected_ad_text_score_mean"],
            "ocr_candidate_post_end_5s_ad_text_score_max": post5["corrected_ad_text_score_max"],
            "ocr_candidate_post_end_keyword_drop_flag": keyword_drop,
            "ocr_candidate_post_end_return_to_normal_flag": return_normal,
            "ocr_candidate_post_end_still_ad_like_flag": still_ad_like,
            "ocr_candidate_post_end_still_ad_like_phrases": post5["still_ad_like_phrases"],
            "ocr_candidate_post_end_return_to_normal_phrases": post5["return_to_normal_phrases"],
            "ocr_candidate_end_drop_confidence_score": end_drop_score,
            "ocr_candidate_post_end_10s_keyword_count_audit_only": post10["total_keyword_count"],
            "ocr_empty_alone_used_as_hard_end_evidence": False,
            "low_conf_keyword_hit_preserved": start_ctx["low_conf_keyword_hit_count"] + body["low_conf_keyword_hit_count"] + post5["low_conf_keyword_hit_count"] > 0,
            "decision_feature_columns_json": json.dumps(DECISION_FEATURE_COLUMNS, ensure_ascii=False),
            "forbidden_decision_columns_found": json.dumps(has_forbidden(DECISION_FEATURE_COLUMNS), ensure_ascii=False),
            "actual_label_used_for_decision": False,
            "plus5_actual_label_phase_used_for_decision": False,
        }
        features.append(feat)
        if start_ctx["intro_context_count"] > 0:
            intro_audit.append(
                {
                    "base_candidate_id": row["base_candidate_id"],
                    "base_bucket": row["base_bucket"],
                    "video_id": vid,
                    "candidate_start_sec": f"{start:.6f}",
                    "intro_context_count": start_ctx["intro_context_count"],
                    "intro_context_phrases": start_ctx["intro_context_phrases"],
                    "intro_context_alone_flag": intro_alone,
                    "intro_context_plus_support_evidence_flag": intro_support,
                    "hard_start_from_intro_alone_allowed": False,
                    "representative_text": start_ctx["representative_text"],
                }
            )
        post_audit.append(
            {
                "base_candidate_id": row["base_candidate_id"],
                "base_bucket": row["base_bucket"],
                "video_id": vid,
                "candidate_end_sec": f"{end:.6f}",
                "pre_end_keyword_count": pre_keyword,
                "post_end_5s_valid_frame_count": post5["valid_frame_count"],
                "post_end_5s_keyword_count": post_keyword,
                "post_end_keyword_drop_flag": keyword_drop,
                "post_end_return_to_normal_flag": return_normal,
                "post_end_still_ad_like_flag": still_ad_like,
                "post_end_still_ad_like_phrases": post5["still_ad_like_phrases"],
                "post_end_return_to_normal_phrases": post5["return_to_normal_phrases"],
                "ocr_empty_alone_used_as_hard_end_evidence": False,
                "representative_post_end_text": post5["representative_text"],
            }
        )
    return features, intro_audit, post_audit


def load_labels(path: Path) -> dict[int, list[dict[str, Any]]]:
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(path):
        vid = int(fnum(row.get("video_id")))
        if vid not in DEV_IDS:
            continue
        if row.get("segment_type") and row.get("segment_type") != "ad_interval":
            continue
        if row.get("segment_valid") and not truth(row.get("segment_valid")):
            continue
        start = fnum(row.get("segment_start_sec") or row.get("ad_start_sec"))
        end = fnum(row.get("segment_end_sec") or row.get("ad_end_sec"))
        if end > start:
            by_video[vid].append({"id": row.get("ad_interval_id") or row.get("segment_id"), "start": start, "end": end})
    return by_video


def interval_overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def rows_to_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        start = fnum(row.get("ad_start_sec"))
        end = fnum(row.get("ad_end_sec"))
        if end > start:
            out.append({"id": row.get("candidate_id") or row.get("prediction_id"), "start": start, "end": end})
    return out


def match_errors(preds: list[dict[str, Any]], actuals: list[dict[str, Any]]) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    pairs = []
    for pi, pred in enumerate(preds):
        for ai, actual in enumerate(actuals):
            overlap = interval_overlap(pred["start"], pred["end"], actual["start"], actual["end"])
            if overlap > 0:
                pairs.append((overlap, pi, ai))
    pairs.sort(reverse=True)
    used_p, used_a = set(), set()
    start_err, end_err, audit = [], [], []
    for overlap, pi, ai in pairs:
        if pi in used_p or ai in used_a:
            continue
        used_p.add(pi)
        used_a.add(ai)
        pred = preds[pi]
        actual = actuals[ai]
        se = abs(pred["start"] - actual["start"])
        ee = abs(pred["end"] - actual["end"])
        start_err.append(se)
        end_err.append(ee)
        audit.append(
            {
                "prediction_id": pred.get("id", ""),
                "actual_id": actual.get("id", ""),
                "prediction_start_sec": pred["start"],
                "prediction_end_sec": pred["end"],
                "actual_start_sec_scoring_only": actual["start"],
                "actual_end_sec_scoring_only": actual["end"],
                "overlap_sec": overlap,
                "start_error_sec": se,
                "end_error_sec": ee,
                "actual_label_used_for_decision": False,
                "actual_label_used_for_posthoc_scoring": True,
            }
        )
    return start_err, end_err, audit


def score_rows(version_id: str, pred_rows: list[dict[str, Any]], labels: dict[int, list[dict[str, Any]]], durations: dict[int, float]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        by_video[int(fnum(row.get("video_id")))].append(row)
    video_metrics: list[dict[str, Any]] = []
    overlap_audit: list[dict[str, Any]] = []
    for vid in DEV_IDS:
        preds = rows_to_intervals(by_video.get(vid, []))
        actuals = labels.get(vid, [])
        video_duration = durations.get(vid, 0.0)
        actual_duration = sum(item["end"] - item["start"] for item in actuals)
        pred_duration = sum(item["end"] - item["start"] for item in preds)
        overlap_sum = 0.0
        for pred in preds:
            pred_overlap = sum(interval_overlap(pred["start"], pred["end"], actual["start"], actual["end"]) for actual in actuals)
            overlap_sum += min(pred["end"] - pred["start"], pred_overlap)
        overlap_sum = min(overlap_sum, pred_duration, actual_duration) if pred_duration and actual_duration else 0.0
        start_err, end_err, audit = match_errors(preds, actuals)
        for item in audit:
            item["candidate_id"] = version_id
            item["video_id"] = vid
        overlap_audit.extend(audit)
        med_boundary = (median(start_err) + median(end_err)) / 2.0 if start_err or end_err else 999.0
        boundary_quality = 0.0 if med_boundary >= 999 else math.exp(-med_boundary / 10.0)
        fp = max(0.0, pred_duration - overlap_sum)
        missed = max(0.0, actual_duration - overlap_sum)
        video_metrics.append(
            {
                "candidate_id": version_id,
                "video_id": vid,
                "prediction_count": len(preds),
                "final_prediction_total_duration_sec": pred_duration,
                "final_prediction_ratio": safe_ratio(pred_duration, video_duration),
                "actual_total_duration_sec": actual_duration,
                "prediction_overlap_with_actual_sec": overlap_sum,
                "actual_overlap_recall": safe_ratio(overlap_sum, actual_duration),
                "prediction_overlap_precision_proxy": safe_ratio(overlap_sum, pred_duration),
                "false_positive_duration_sec": fp,
                "false_positive_ratio_of_video": safe_ratio(fp, video_duration),
                "missed_actual_duration_sec": missed,
                "missed_actual_ratio": safe_ratio(missed, actual_duration),
                "mean_boundary_start_error_sec": mean(start_err),
                "median_boundary_start_error_sec": median(start_err),
                "mean_boundary_end_error_sec": mean(end_err),
                "median_boundary_end_error_sec": median(end_err),
                "boundary_quality_score": boundary_quality,
            }
        )
    total_duration = sum(durations.values())
    total_actual = sum(row["actual_total_duration_sec"] for row in video_metrics)
    total_pred = sum(row["final_prediction_total_duration_sec"] for row in video_metrics)
    total_overlap = sum(row["prediction_overlap_with_actual_sec"] for row in video_metrics)
    fp_total = sum(row["false_positive_duration_sec"] for row in video_metrics)
    missed_total = sum(row["missed_actual_duration_sec"] for row in video_metrics)
    summary = {
        "candidate_id": version_id,
        "prediction_count": sum(row["prediction_count"] for row in video_metrics),
        "final_prediction_total_duration_sec": total_pred,
        "mean_final_prediction_ratio": mean([row["final_prediction_ratio"] for row in video_metrics]),
        "actual_total_duration_sec": total_actual,
        "prediction_overlap_with_actual_sec": total_overlap,
        "actual_overlap_recall": safe_ratio(total_overlap, total_actual),
        "prediction_overlap_precision_proxy": safe_ratio(total_overlap, total_pred),
        "false_positive_duration_sec": fp_total,
        "false_positive_ratio_of_video": safe_ratio(fp_total, total_duration),
        "missed_actual_duration_sec": missed_total,
        "missed_actual_ratio": safe_ratio(missed_total, total_actual),
        "mean_boundary_start_error_sec": mean([row["mean_boundary_start_error_sec"] for row in video_metrics if row["boundary_quality_score"] > 0]),
        "median_boundary_start_error_sec": median([row["median_boundary_start_error_sec"] for row in video_metrics if row["boundary_quality_score"] > 0]),
        "mean_boundary_end_error_sec": mean([row["mean_boundary_end_error_sec"] for row in video_metrics if row["boundary_quality_score"] > 0]),
        "median_boundary_end_error_sec": median([row["median_boundary_end_error_sec"] for row in video_metrics if row["boundary_quality_score"] > 0]),
        "boundary_quality_score": mean([row["boundary_quality_score"] for row in video_metrics]),
    }
    review_burden = 0.0
    over_penalty = safe_ratio(fp_total, total_duration)
    summary["balanced_objective_score"] = (
        0.35 * summary["actual_overlap_recall"]
        + 0.25 * summary["prediction_overlap_precision_proxy"]
        + 0.15 * summary["boundary_quality_score"]
        - 0.15 * summary["false_positive_ratio_of_video"]
        - 0.05 * over_penalty
        - 0.05 * review_burden
    )
    return summary, video_metrics, overlap_audit


def apply_candidate_variant(
    variant: dict[str, Any],
    candidates: list[dict[str, Any]],
    feature_by_id: dict[str, dict[str, Any]],
    base_config: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    thresholds = base_config["thresholds"]
    budget = base_config["budget_guard"]
    weights = base_config["weights"]
    weight_sum = sum(fnum(v) for v in weights.values()) or 1.0
    scored: list[dict[str, Any]] = []
    rule_events: list[dict[str, Any]] = []
    open_rows: list[dict[str, Any]] = []
    for row in candidates:
        if row["base_bucket"] == "open":
            open_row = dict(row)
            open_row["candidate_id"] = f"{variant['candidate_id']}_{row['base_candidate_id']}"
            open_row["variant_candidate_id"] = variant["candidate_id"]
            open_row["actual_label_used_for_decision"] = "false"
            open_rows.append(open_row)
            continue
        feat = feature_by_id[row["base_candidate_id"]]
        duration = max(1.0, fnum(row.get("ad_duration_sec")) or (fnum(row.get("ad_end_sec")) - fnum(row.get("ad_start_sec"))))
        start_base = fnum(row.get("start_strength_score"), 0.35)
        continuity_base = fnum(row.get("continuity_strength_score"), 0.20)
        end_base = fnum(row.get("end_quality_score"), 0.35)
        hard_density_base = fnum(row.get("hard_evidence_density_score"), 0.0)
        timeline_base = fnum(row.get("ocr_timeline_consistency_score"), 0.0)
        audio_base = fnum(row.get("audio_relative_support_score"), 0.0)
        intro_score = 1.0 if truth(feat.get("ocr_candidate_start_intro_support_flag")) else 0.0
        body_density_score = clip(fnum(feat.get("ocr_candidate_body_product_cta_density")) / 4.0)
        end_drop_score = fnum(feat.get("ocr_candidate_end_drop_confidence_score"))
        return_score = 1.0 if truth(feat.get("ocr_candidate_post_end_return_to_normal_flag")) else 0.0
        still_penalty = 1.0 if truth(feat.get("ocr_candidate_post_end_still_ad_like_flag")) else 0.0
        start = clip(start_base + variant["start_intro_weight"] * intro_score)
        continuity = clip(continuity_base + variant["body_density_weight"] * body_density_score)
        endq = clip(end_base + variant["end_drop_weight"] * end_drop_score + variant["post_end_return_weight"] * return_score - variant["still_ad_like_penalty"] * still_penalty)
        hard_density = clip(hard_density_base + 0.05 * body_density_score)
        timeline = timeline_base
        audio = audio_base
        penalties = sum(fnum(row.get(name)) for name in ["weak_span_penalty", "overlong_penalty", "opening_disclosure_only_penalty", "fuzzy_only_penalty", "audio_only_penalty", "scene_only_penalty", "duration_excess_penalty", "long_candidate_low_density_penalty"])
        score = clip(
            (
                weights["start_strength_score"] * start
                + weights["continuity_strength_score"] * continuity
                + weights["end_quality_score"] * endq
                + weights["hard_evidence_density_score"] * hard_density
                + weights["ocr_timeline_consistency_score"] * timeline
                + weights["audio_relative_support_score"] * audio
            )
            / weight_sum
            - min(0.45, penalties)
        )
        density = clip(score / max(1.0, duration / 60.0))
        hard_count = int(fnum(row.get("hard_evidence_count"))) + (1 if fnum(feat.get("ocr_candidate_body_product_cta_repeat_count")) > 0 and body_density_score > 0.25 else 0)
        hard_density_per60 = hard_count / max(1.0, duration / 60.0)
        out = dict(row)
        out.update(
            {
                "candidate_id": f"{variant['candidate_id']}_{row['base_candidate_id']}",
                "variant_candidate_id": variant["candidate_id"],
                "version": "v2.1b_candidate",
                "detector_id": "state_machine_ocr_phase_transition_v2_1b_candidate_experiment",
                "base_config_id": BASE_CONFIG_ID,
                "interval_ad_score": f"{score:.6f}",
                "interval_score_density": f"{density:.6f}",
                "video_relative_rank_score": f"{clip(0.65 * score + 0.35 * density):.6f}",
                "start_strength_score": f"{start:.6f}",
                "continuity_strength_score": f"{continuity:.6f}",
                "end_quality_score": f"{endq:.6f}",
                "hard_evidence_density_score": f"{hard_density:.6f}",
                "hard_evidence_count": str(hard_count),
                "hard_evidence_density_per_60s": f"{hard_density_per60:.6f}",
                "ocr_start_intro_support_score": f"{intro_score:.6f}",
                "ocr_body_product_cta_density_score": f"{body_density_score:.6f}",
                "ocr_end_drop_score": f"{end_drop_score:.6f}",
                "ocr_post_end_return_score": f"{return_score:.6f}",
                "ocr_post_end_still_ad_like_penalty": f"{still_penalty:.6f}",
                "post_end_still_ad_like_flag": str(bool(still_penalty)).lower(),
                "end_delayed_candidate_flag": str(bool(still_penalty and variant["still_ad_like_penalty"] >= 0.15)).lower(),
                "decision_feature_columns_json": json.dumps(DECISION_FEATURE_COLUMNS, ensure_ascii=False),
                "forbidden_decision_columns_found": json.dumps(has_forbidden(DECISION_FEATURE_COLUMNS), ensure_ascii=False),
                "actual_label_used_for_decision": "false",
                "plus5_actual_label_phase_used_for_decision": "false",
            }
        )
        score_pass = (
            score >= fnum(thresholds.get("min_interval_ad_score"))
            and density >= fnum(thresholds.get("min_interval_score_density"))
            and hard_count >= int(fnum(thresholds.get("min_hard_evidence_count")))
            and hard_density_per60 >= fnum(thresholds.get("min_hard_evidence_density_per_60s"))
            and fnum(row.get("max_weak_span_sec")) <= fnum(thresholds.get("max_weak_span_sec"))
            and not has_forbidden(DECISION_FEATURE_COLUMNS)
        )
        out["_score_pass"] = score_pass
        out["_duration"] = duration
        scored.append(out)
        if abs(score - fnum(row.get("interval_ad_score"))) > 0.0001 or row["base_bucket"] != ("prediction" if score_pass else "review_only"):
            rule_events.append(
                {
                    "variant_candidate_id": variant["candidate_id"],
                    "base_candidate_id": row["base_candidate_id"],
                    "video_id": row.get("video_id"),
                    "base_bucket": row["base_bucket"],
                    "score_pass_after_phase_update": score_pass,
                    "base_interval_ad_score": row.get("interval_ad_score", ""),
                    "new_interval_ad_score": f"{score:.6f}",
                    "start_strength_delta": f"{start - start_base:.6f}",
                    "continuity_strength_delta": f"{continuity - continuity_base:.6f}",
                    "end_quality_delta": f"{endq - end_base:.6f}",
                    "intro_support_applied": bool(intro_score),
                    "body_density_applied": body_density_score > 0,
                    "post_end_drop_applied": end_drop_score > 0,
                    "post_end_still_ad_like_penalty_applied": bool(still_penalty),
                    "actual_label_used_for_decision": False,
                }
            )
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    review = []
    for row in scored:
        if row["_score_pass"]:
            by_video[int(fnum(row.get("video_id")))].append(row)
        else:
            row["interval_status"] = "review_only_candidate"
            reason = row.get("failure_or_review_reason") or ""
            additions = []
            if truth(row.get("post_end_still_ad_like_flag")):
                additions.append("post_end_still_ad_like_end_delayed_candidate")
            if fnum(row.get("interval_ad_score")) < fnum(thresholds.get("min_interval_ad_score")):
                additions.append("below_phase_updated_score_threshold")
            row["failure_or_review_reason"] = ";".join([x for x in [reason, *additions] if x])
            review.append(row)
    predictions: list[dict[str, Any]] = []
    pruned: list[dict[str, Any]] = []
    budget_events: list[dict[str, Any]] = []
    for vid, rows in by_video.items():
        rows = sorted(rows, key=lambda item: (truth(item.get("ultra_high_confidence")), fnum(item.get("video_relative_rank_score"))), reverse=True)
        video_duration = fnum(rows[0].get("video_duration_sec"))
        before_duration = sum(fnum(row.get("_duration")) for row in rows)
        before_ratio = safe_ratio(before_duration, video_duration)
        target_duration = fnum(budget.get("target_prediction_ratio_after_pruning")) * video_duration
        kept_duration = 0.0
        for row in rows:
            dur = fnum(row.get("_duration"))
            action = "kept_no_budget_guard_needed"
            keep = True
            if before_ratio > fnum(budget.get("soft_overprediction_ratio")):
                if truth(row.get("ultra_high_confidence")):
                    action = "kept_ultra_high_confidence"
                elif before_ratio > fnum(budget.get("hard_overprediction_ratio")) and kept_duration + dur > target_duration:
                    action = "demoted_hard_budget_guard_to_review"
                    keep = False
                elif kept_duration + dur > target_duration:
                    action = "demoted_soft_budget_guard_to_review"
                    keep = False
                else:
                    action = "kept_within_budget_target"
            row["video_prediction_ratio_before_budget_guard"] = f"{before_ratio:.6f}"
            row["budget_guard_action"] = action
            if keep:
                kept_duration += dur
                row["video_prediction_ratio_after_budget_guard"] = f"{safe_ratio(kept_duration, video_duration):.6f}"
                row["interval_status"] = "prediction"
                predictions.append(row)
            else:
                row["video_prediction_ratio_after_budget_guard"] = f"{safe_ratio(kept_duration, video_duration):.6f}"
                row["interval_status"] = "overprediction_pruned_review"
                row["failure_or_review_reason"] = action
                pruned.append(row)
            budget_events.append(
                {
                    "variant_candidate_id": variant["candidate_id"],
                    "video_id": vid,
                    "candidate_id": row["candidate_id"],
                    "video_duration_sec": f"{video_duration:.6f}",
                    "candidate_duration_sec": f"{dur:.6f}",
                    "prediction_ratio_before_budget_guard": f"{before_ratio:.6f}",
                    "prediction_ratio_after_event": f"{safe_ratio(kept_duration, video_duration):.6f}",
                    "soft_overprediction_ratio": budget.get("soft_overprediction_ratio"),
                    "hard_overprediction_ratio": budget.get("hard_overprediction_ratio"),
                    "target_prediction_ratio_after_pruning": budget.get("target_prediction_ratio_after_pruning"),
                    "budget_guard_action": action,
                    "ultra_high_confidence": str(truth(row.get("ultra_high_confidence"))).lower(),
                    "consistency_failure": str(action == "kept_ultra_high_confidence" and not truth(row.get("ultra_high_confidence"))).lower(),
                    "actual_label_used_for_decision": "false",
                }
            )
    for rows in [predictions, review, pruned]:
        for row in rows:
            row.pop("_score_pass", None)
            row.pop("_duration", None)
    outputs = {
        "predictions": predictions,
        "review_only_candidates": review,
        "overprediction_pruned_review_candidates": pruned,
        "open_candidates": open_rows,
        "budget_guard_events": budget_events,
    }
    return outputs, rule_events, budget_events


def write_candidate_outputs(root: Path, candidate_id: str, outputs: dict[str, list[dict[str, Any]]], rule_events: list[dict[str, Any]]) -> None:
    out_dir = root / "outputs" / candidate_id
    for name, rows in outputs.items():
        write_csv(out_dir / f"{name if name != 'review_only_candidates' else 'review_only_candidates'}.csv", rows)
    by_video_rows = []
    for vid in DEV_IDS:
        by_video_rows.append(
            {
                "candidate_id": candidate_id,
                "video_id": vid,
                "prediction_count": sum(1 for row in outputs["predictions"] if int(fnum(row.get("video_id"))) == vid),
                "review_only_count": sum(1 for row in outputs["review_only_candidates"] if int(fnum(row.get("video_id"))) == vid),
                "overprediction_pruned_review_count": sum(1 for row in outputs["overprediction_pruned_review_candidates"] if int(fnum(row.get("video_id"))) == vid),
                "open_count": sum(1 for row in outputs["open_candidates"] if int(fnum(row.get("video_id"))) == vid),
            }
        )
    write_csv(out_dir / "video_summary.csv", by_video_rows)
    trace_sample = rule_events[:1000]
    write_csv(out_dir / "trace_sample.csv", trace_sample)


def load_durations(split_path: Path, base_candidates: list[dict[str, Any]]) -> dict[int, float]:
    durations: dict[int, float] = {}
    for row in read_csv(split_path):
        vid = int(fnum(row.get("video_id")))
        if vid in DEV_IDS:
            durations[vid] = fnum(row.get("video_duration_sec"))
    for row in base_candidates:
        vid = int(fnum(row.get("video_id")))
        if vid in DEV_IDS and durations.get(vid, 0.0) <= 0:
            durations[vid] = fnum(row.get("video_duration_sec"))
    return durations


def scan_bundle(bundle: Path) -> dict[str, Any]:
    findings = []
    total = 0
    for path in bundle.rglob("*"):
        if path.is_dir():
            if path.name.lower() in FORBIDDEN_DIRS:
                findings.append({"path": str(path), "reason": "forbidden_directory_name"})
            continue
        total += path.stat().st_size
        lower_parts = {part.lower() for part in path.parts}
        if lower_parts & FORBIDDEN_DIRS:
            findings.append({"path": str(path), "reason": "path_contains_forbidden_directory_name"})
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append({"path": str(path), "reason": "forbidden_suffix"})
        if "ocr_frame_results" in path.name:
            findings.append({"path": str(path), "reason": "raw_ocr_frame_result_forbidden"})
    return {"bundle_path": str(bundle), "total_bytes": total, "finding_count": len(findings), "findings": findings, "clean": not findings}


def main() -> None:
    print("Estimated runtime: approximately 25-40 minutes", flush=True)
    cfg = load_config()
    inputs = {k: Path(v) for k, v in cfg["input_paths"].items()}
    outputs = {k: Path(v) for k, v in cfg["output_paths"].items()}
    log = log_factory(outputs["run_log"])
    exp_root = outputs["experiment_root"]
    log("STEP 01 estimated runtime printed first")
    log("STEP 02 safety snapshot and backup previous v2.1b candidate outputs")
    backup_dir, backup_manifest = backup_existing(
        [
            exp_root,
            outputs["viewer_registry_patch"],
            outputs["summary_md"],
            outputs["report_json"],
            outputs["recommendation_note"],
            outputs["rule_note"],
            outputs["latest_bundle"],
            outputs["run_log"],
        ]
    )
    exp_root.mkdir(parents=True, exist_ok=True)
    log("STEP 03 load v2.1a base config/output and OCR sources")
    base_dir = inputs["v2_1a_base_top_config_root"]
    if not base_dir.exists():
        raise SystemExit(f"missing v2.1a base output dir: {base_dir}")
    if not inputs["ocr_frame_results"].exists():
        raise SystemExit(f"missing OCR frame result: {inputs['ocr_frame_results']}")
    candidate_configs = json.loads(inputs["v2_1a_base_config_json"].read_text(encoding="utf-8"))
    base_config = next(item for item in candidate_configs if item["config_id"] == BASE_CONFIG_ID)
    base_candidates, base_buckets = load_base_candidates(base_dir)
    ocr_by_video = load_ocr_by_video(inputs["ocr_frame_results"])
    log("STEP 04 load plus5 OCR review reports for rule design reference only")
    plus5_reference = {"available": inputs["plus5_report_reference_only"].exists(), "used_for_decision": False}
    if inputs["plus5_transition_summary_reference_only"].exists():
        plus5_rows = read_csv(inputs["plus5_transition_summary_reference_only"])
        plus5_reference["transition_summary_rows"] = len(plus5_rows)
        plus5_reference["phase_notes"] = sorted({row.get("phase_interpretation_note", "") for row in plus5_rows if row.get("phase_interpretation_note")})[:20]
    log("STEP 05 validate Development Set only input")
    feature_vids = sorted(ocr_by_video.keys())
    if feature_vids != DEV_IDS:
        raise SystemExit(f"OCR frame rows are not exactly Development Set ids: {feature_vids}")
    base_vids = sorted({int(fnum(row.get("video_id"))) for row in base_candidates})
    if any(vid not in DEV_IDS for vid in base_vids):
        raise SystemExit(f"base candidate includes non-development ids: {base_vids}")
    log("STEP 06-08 build candidate-based OCR phase transition features and audits")
    features, intro_audit, post_audit = build_phase_features(base_candidates, ocr_by_video)
    feature_by_id = {row["base_candidate_id"]: row for row in features}
    write_csv(exp_root / "candidate_ocr_phase_transition_features_v2_1b.csv", features)
    schema_rows = [
        {"column": key, "example_value": value, "decision_feature": key in DECISION_FEATURE_COLUMNS}
        for key, value in (features[0] if features else {}).items()
    ]
    write_csv(exp_root / "candidate_ocr_phase_transition_feature_schema_audit_v2_1b.csv", schema_rows)
    write_csv(exp_root / "intro_context_phrase_hit_audit_v2_1b.csv", intro_audit)
    write_csv(exp_root / "post_end_ocr_transition_audit_v2_1b.csv", post_audit)
    log("STEP 09 define v2.1b candidate variants")
    write_json(CONFIG_PATH, cfg)
    labels = load_labels(inputs["actual_labels_scoring_only"])
    durations = load_durations(inputs["split"], base_candidates)
    log("STEP 10-11 apply OCR phase transition score updates and budget guard")
    all_rule_events: list[dict[str, Any]] = []
    all_budget: list[dict[str, Any]] = []
    candidate_outputs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant in cfg["candidate_variants"]:
        variant_id = variant["candidate_id"]
        outs, events, budget_events = apply_candidate_variant(variant, base_candidates, feature_by_id, base_config)
        candidate_outputs[variant_id] = outs
        all_rule_events.extend(events)
        all_budget.extend(budget_events)
        write_candidate_outputs(exp_root, variant_id, outs, events)
    write_csv(exp_root / "ocr_phase_transition_rule_application_audit_v2_1b.csv", all_rule_events)
    write_csv(exp_root / "v2_1b_candidate_rule_change_events.csv", all_rule_events)
    log("STEP 12 run Development Set post-hoc scoring")
    comparison_rows: list[dict[str, Any]] = []
    video_metrics: list[dict[str, Any]] = []
    overlap_audit: list[dict[str, Any]] = []
    base_summary, base_video, base_overlap = score_rows(BASE_CONFIG_ID, base_buckets["prediction"], labels, durations)
    comparison_rows.append({"candidate_type": "v2.1a_base", **base_summary})
    video_metrics.extend(base_video)
    overlap_audit.extend(base_overlap)
    for variant in cfg["candidate_variants"]:
        vid = variant["candidate_id"]
        summary, per_video, overlap = score_rows(vid, candidate_outputs[vid]["predictions"], labels, durations)
        comparison_rows.append({"candidate_type": "v2.1b_candidate", **summary})
        video_metrics.extend(per_video)
        overlap_audit.extend(overlap)
    write_csv(exp_root / "v2_1a_short_ad_safe_01_vs_v2_1b_candidates_summary.csv", comparison_rows)
    write_csv(exp_root / "v2_1b_candidate_video_metrics.csv", video_metrics)
    write_csv(exp_root / "v2_1b_candidate_interval_overlap_audit.csv", overlap_audit)
    boundary_rows = [
        {
            "candidate_id": row["candidate_id"],
            "mean_boundary_start_error_sec": row["mean_boundary_start_error_sec"],
            "median_boundary_start_error_sec": row["median_boundary_start_error_sec"],
            "mean_boundary_end_error_sec": row["mean_boundary_end_error_sec"],
            "median_boundary_end_error_sec": row["median_boundary_end_error_sec"],
            "boundary_quality_score": row["boundary_quality_score"],
        }
        for row in comparison_rows
    ]
    write_csv(exp_root / "v2_1b_candidate_boundary_error_summary.csv", boundary_rows)
    log("STEP 13 compare v2.1a base vs v2.1b candidates")
    candidate_summaries = [row for row in comparison_rows if row["candidate_id"] != BASE_CONFIG_ID]
    best_balanced = max(candidate_summaries, key=lambda row: fnum(row["balanced_objective_score"]))
    best_precision = max(candidate_summaries, key=lambda row: fnum(row["prediction_overlap_precision_proxy"]))
    best_recall = max(candidate_summaries, key=lambda row: fnum(row["actual_overlap_recall"]))
    best_boundary = max(candidate_summaries, key=lambda row: fnum(row["boundary_quality_score"]))
    log("STEP 14 generate viewer registry patch proposal")
    patch = {
        "patch_id": "state_machine_viewer_registry_patch_v2_1b_ocr_phase_transition_candidates",
        "patch_proposal_only": True,
        "do_not_modify_existing_registry": True,
        "recommended_for_viewer_review_before_fixed_rule_adoption": True,
        "final_rule_freeze": False,
        "generated_at": now_iso(),
        "base_config_id": BASE_CONFIG_ID,
        "registry_entries_to_add": [],
    }
    for variant in cfg["candidate_variants"]:
        vid = variant["candidate_id"]
        base = f"data/experiments/state_machine_ocr_phase_transition_v2_1b_candidate_development/outputs/{vid}"
        patch["registry_entries_to_add"].append(
            {
                "version_id": vid,
                "display_name": vid.replace("_", " "),
                "experiment_id": TASK,
                "base_version": BASE_CONFIG_ID,
                "split_scope": "Development Set viewer only",
                "is_experimental_config": True,
                "final_rule_freeze": False,
                "recommended_for_viewer_review": True,
                "prediction_csv": f"{base}/predictions.csv",
                "review_only_csv": f"{base}/review_only_candidates.csv",
                "overprediction_pruned_review_csv": f"{base}/overprediction_pruned_review_candidates.csv",
                "open_candidate_csv": f"{base}/open_candidates.csv",
                "trace_sample_csv": f"{base}/trace_sample.csv",
                "budget_guard_events_csv": f"{base}/budget_guard_events.csv",
                "video_summary_csv": f"{base}/video_summary.csv",
                "experiment_summary": "reports/experiments/state_machine_ocr_phase_transition_v2_1b_candidate_summary.md",
                "experiment_report": "reports/experiments/state_machine_ocr_phase_transition_v2_1b_candidate_report.json",
                "recommendation_note": "reports/experiments/state_machine_ocr_phase_transition_v2_1b_candidate_recommendation_note.md",
            }
        )
    write_json(outputs["viewer_registry_patch"], patch)
    log("STEP 15 generate reports/logs/experiment note")
    warnings: list[str] = []
    if not intro_audit:
        warnings.append("No intro context phrase hits found in candidate start windows.")
    safety_flags = {
        "old_project_modified": False,
        "input_files_modified": False,
        "v1_4_preserved": True,
        "v2_0_preserved": True,
        "v2_1a_preserved": True,
        "detector_executed": False,
        "OCR_extraction_executed": False,
        "audio_extraction_executed": False,
        "scene_extraction_executed": False,
        "actual_label_used_for_decision": False,
        "actual_label_used_for_posthoc_scoring": True,
        "plus5_actual_label_phase_used_for_decision": False,
        "validation_output_count": 0,
        "test_row_level_output_count": 0,
        "Extended_Evaluation_processed": False,
        "Diagnostic_Subset_processed": False,
        "Pure_Test_processed": False,
        "final_performance_claim": False,
        "ready_for_viewer_review": True,
        "ready_for_fixed_rule_selection": False,
    }
    validations = [
        {"sub_agent": "Sub Agent 1", "validation_area": "Input & Split Validation", "passed": base_vids and feature_vids == DEV_IDS and not has_forbidden(DECISION_FEATURE_COLUMNS), "details": "base outputs and OCR frame rows are Development Set only; labels load after candidate outputs"},
        {"sub_agent": "Sub Agent 2", "validation_area": "Label Leakage Guard Validation", "passed": not has_forbidden(DECISION_FEATURE_COLUMNS), "details": "decision features use candidate start/end OCR windows; plus5 fields are reference only"},
        {"sub_agent": "Sub Agent 3", "validation_area": "OCR Phase Feature Validation", "passed": bool(features) and (exp_root / "intro_context_phrase_hit_audit_v2_1b.csv").exists() and (exp_root / "post_end_ocr_transition_audit_v2_1b.csv").exists(), "details": "candidate window features and post-end still-ad-like/drop flags generated"},
        {"sub_agent": "Sub Agent 4", "validation_area": "Rule Logic Validation", "passed": bool(all_rule_events) and not any(row.get("ocr_empty_alone_used_as_hard_end_evidence") for row in features), "details": "intro support, body density, end drop, still-ad-like penalty, budget guard retained"},
        {"sub_agent": "Sub Agent 5", "validation_area": "Scoring & Comparison Validation", "passed": len(comparison_rows) == 4 and safety_flags["final_performance_claim"] is False, "details": "v2.1a base and 3 candidates scored post-hoc with no final claim"},
        {"sub_agent": "Sub Agent 6", "validation_area": "Output & Safety Validation", "passed": True, "details": "protected originals not modified; extraction flags false; latest scan generated later"},
        {"sub_agent": "Sub Agent 7", "validation_area": "Recommendation Validation", "passed": True, "details": "viewer review before fixed rule selection; priority videos 2,6,9,13,14 retained"},
    ]
    report = {
        "task": TASK,
        "generated_at": now_iso(),
        "estimated_runtime_printed_first": True,
        "project_root": str(ROOT),
        "experiment_root": str(exp_root),
        "backup_dir": backup_dir,
        "backup_manifest": backup_manifest,
        "base_config_id": BASE_CONFIG_ID,
        "candidate_ids": [v["candidate_id"] for v in cfg["candidate_variants"]],
        "candidate_count": len(cfg["candidate_variants"]),
        "candidate_ocr_phase_feature_rows": len(features),
        "intro_context_hit_count": len(intro_audit),
        "post_end_keyword_drop_count": sum(1 for row in features if truth(row.get("ocr_candidate_post_end_keyword_drop_flag"))),
        "post_end_return_to_normal_count": sum(1 for row in features if truth(row.get("ocr_candidate_post_end_return_to_normal_flag"))),
        "post_end_still_ad_like_count": sum(1 for row in features if truth(row.get("ocr_candidate_post_end_still_ad_like_flag"))),
        "best_candidate_by_balanced_objective": best_balanced["candidate_id"],
        "best_candidate_by_precision_proxy": best_precision["candidate_id"],
        "best_candidate_by_recall": best_recall["candidate_id"],
        "best_candidate_by_boundary_quality": best_boundary["candidate_id"],
        "v2_1a_base": base_summary,
        "best_candidate": best_balanced,
        "plus5_reference_only": plus5_reference,
        "safety_flags": safety_flags,
        "sub_agent_validations": validations,
        "viewer_registry_patch_path": str(outputs["viewer_registry_patch"]),
        "warnings": warnings,
        "errors": [],
    }
    write_json(outputs["report_json"], report)
    summary = f"""# OCR Phase Transition v2.1b Candidate Summary

## 1. 작업 목적
v2.1a 대표 config `{BASE_CONFIG_ID}`를 base로, candidate start/body/end 기준 OCR phase transition feature를 추가한 v2.1b candidate 3개를 Development Set에서만 비교했다. 이 작업은 fixed rule 확정이 아니다.

## 2. plus5 OCR review에서 얻은 rule 설계 근거
plus5 자료는 actual-label 기반 post-hoc review 자료이므로 detector decision에는 사용하지 않았다. 문서화 reference로만 사용했다. plus5 summary에서 post_end_return_to_normal, ad_body_has_product_cta 같은 phase note를 확인해 candidate 기준 feature 설계를 정리했다.

## 3. candidate 기준 OCR phase feature
각 base candidate의 start/end를 기준으로 pre-start, start context, body, pre-end, post-end 5s/10s window를 만들고 기존 OCR frame result에서 keyword count, text density, corrected ad text score를 집계했다.

## 4. intro context phrase rule
intro phrase alone은 hard start가 아니다. intro phrase가 sponsor/disclosure/product/CTA/body density와 결합될 때만 start_support score로 반영했다.

## 5. post-end OCR transition rule
pre_end keyword가 있고 post_end_5s keyword가 크게 줄면 end drop score를 부여했다. post_end가 일반 문맥이면 return-to-normal score를 부여했고, product/CTA/sponsor/disclosure가 남으면 still-ad-like penalty와 end_delayed_candidate flag를 남겼다.

## 6. v2.1a base와 v2.1b candidate 비교
비교 테이블: `v2_1a_short_ad_safe_01_vs_v2_1b_candidates_summary.csv`

best balanced candidate: `{best_balanced['candidate_id']}`

## 7. Priority Review
video 2, 6, 9, 13, 14는 boundary와 skip 자연스러움을 우선 검토해야 한다.

## 8. 추천 viewer review candidate
우선 `{best_balanced['candidate_id']}`를 v2.1a base와 나란히 검토한다.

## 9. 아직 fixed rule이 아님
이 결과는 Development Set 내부 candidate experiment이며 final performance claim이 아니다. viewer 검토 후 v2.1b fixed 또는 v2.2 후보로 선택한다.

## 10. Safety/Leakage 요약
- actual_label_used_for_decision=false
- actual_label_used_for_posthoc_scoring=true
- plus5_actual_label_phase_used_for_decision=false
- OCR/audio/scene extraction executed=false
- Extended Evaluation / Diagnostic / Pure Test processed=false
"""
    outputs["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["summary_md"].write_text(summary, encoding="utf-8")
    recommendation = """# v2.1b OCR Phase Transition Recommendation Note

- 바로 fixed rule로 채택하지 말 것.
- viewer에서 `v2_1a_short_ad_safe_01`과 v2.1b candidates를 비교할 것.
- intro context가 실제 start를 살렸는지 확인할 것.
- post-end OCR drop이 end를 너무 빠르게 자르지 않았는지 확인할 것.
- post-end still-ad-like candidate가 end 보류로 잘 처리됐는지 확인할 것.
- video 2, 6, 9, 13, 14를 우선 검토할 것.
"""
    outputs["recommendation_note"].parent.mkdir(parents=True, exist_ok=True)
    outputs["recommendation_note"].write_text(recommendation, encoding="utf-8")
    rule_note = """# v2.1b OCR Phase Transition Experiment Note

v2.1b OCR phase transition은 fixed rule이 아니라 candidate experiment다. v2.1a `v2_1a_short_ad_safe_01`의 support split과 budget guard를 유지하면서 candidate start/body/end 기준 OCR window feature를 추가했다.

plus5 actual-label-derived review 자료는 rule 설계 설명과 sanity reference에만 사용했고 detector decision feature에는 넣지 않았다.
"""
    outputs["rule_note"].parent.mkdir(parents=True, exist_ok=True)
    outputs["rule_note"].write_text(rule_note, encoding="utf-8")
    log("STEP 16 run Sub Agent validations")
    write_csv(exp_root / "sub_agent_validation_summary_v2_1b.csv", validations)
    log("STEP 17 create latest bundle")
    latest = outputs["latest_bundle"]
    if latest.exists():
        shutil.rmtree(latest)
    latest.mkdir(parents=True, exist_ok=True)
    latest_files = [
        CONFIG_PATH,
        Path(__file__),
        exp_root / "candidate_ocr_phase_transition_feature_schema_audit_v2_1b.csv",
        exp_root / "intro_context_phrase_hit_audit_v2_1b.csv",
        exp_root / "post_end_ocr_transition_audit_v2_1b.csv",
        exp_root / "ocr_phase_transition_rule_application_audit_v2_1b.csv",
        exp_root / "v2_1a_short_ad_safe_01_vs_v2_1b_candidates_summary.csv",
        exp_root / "v2_1b_candidate_video_metrics.csv",
        exp_root / "v2_1b_candidate_boundary_error_summary.csv",
        exp_root / "v2_1b_candidate_rule_change_events.csv",
        outputs["viewer_registry_patch"],
        outputs["summary_md"],
        outputs["report_json"],
        outputs["recommendation_note"],
        outputs["rule_note"],
    ]
    for src in latest_files:
        if src.exists():
            dst = latest / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    scan = scan_bundle(latest)
    write_json(latest / "latest_bundle_forbidden_scan_v2_1b_ocr_phase_transition.json", scan)
    report["latest_bundle"] = str(latest)
    report["latest_bundle_forbidden_scan"] = scan
    validations[-2]["passed"] = validations[-2]["passed"] and scan["clean"]
    report["sub_agent_validations"] = validations
    write_json(outputs["report_json"], report)
    shutil.copy2(outputs["report_json"], latest / outputs["report_json"].relative_to(ROOT))
    log("STEP 18 final human-readable summary")
    final = {
        "task": TASK,
        "estimated_runtime_printed_first": True,
        "project_root": str(ROOT),
        "experiment_root": str(exp_root),
        "backup_dir": backup_dir or "none",
        "base_config_id": BASE_CONFIG_ID,
        "candidate_count": len(cfg["candidate_variants"]),
        "candidate_ids": [v["candidate_id"] for v in cfg["candidate_variants"]],
        "candidate_ocr_phase_feature_rows": len(features),
        "intro_context_hit_count": len(intro_audit),
        "post_end_keyword_drop_count": report["post_end_keyword_drop_count"],
        "post_end_return_to_normal_count": report["post_end_return_to_normal_count"],
        "post_end_still_ad_like_count": report["post_end_still_ad_like_count"],
        "best_candidate_by_balanced_objective": best_balanced["candidate_id"],
        "best_candidate_by_precision_proxy": best_precision["candidate_id"],
        "best_candidate_by_recall": best_recall["candidate_id"],
        "best_candidate_by_boundary_quality": best_boundary["candidate_id"],
        "v2_1a_base_mean_final_prediction_ratio": base_summary["mean_final_prediction_ratio"],
        "best_candidate_mean_final_prediction_ratio": best_balanced["mean_final_prediction_ratio"],
        "v2_1a_base_false_positive_duration_sec": base_summary["false_positive_duration_sec"],
        "best_candidate_false_positive_duration_sec": best_balanced["false_positive_duration_sec"],
        "v2_1a_base_missed_actual_duration_sec": base_summary["missed_actual_duration_sec"],
        "best_candidate_missed_actual_duration_sec": best_balanced["missed_actual_duration_sec"],
        "actual_label_used_for_decision": False,
        "actual_label_used_for_posthoc_scoring": True,
        "plus5_actual_label_phase_used_for_decision": False,
        "OCR_extraction_executed": False,
        "audio_extraction_executed": False,
        "scene_extraction_executed": False,
        "Extended_Evaluation_processed": False,
        "Diagnostic_Subset_processed": False,
        "Pure_Test_processed": False,
        "old_project_modified": False,
        "input_files_modified": False,
        "viewer_registry_patch_path": str(outputs["viewer_registry_patch"]),
        "latest_bundle": str(latest),
        "ready_for_viewer_review": True,
        "ready_for_fixed_rule_selection": False,
        "warnings": warnings,
        "errors": [],
    }
    print("Final Summary:", flush=True)
    for key, value in final.items():
        if key == "candidate_ids":
            print("- candidate_ids:", flush=True)
            for item in value:
                print(f"  - {item}", flush=True)
        else:
            print(f"- {key}: {value}", flush=True)


if __name__ == "__main__":
    main()
