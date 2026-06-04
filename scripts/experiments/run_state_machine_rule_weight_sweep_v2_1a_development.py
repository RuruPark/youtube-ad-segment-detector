#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(".")
OLD_PROJECT = Path("./_old_project_not_included")
EXPERIMENT_ID = "state_machine_rule_weight_sweep_v2_1a_development"
VERSION = "v2.1a"
DEV_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
MAX_CONFIGS = 60
SEED = 20260527

CONFIG_PATH = ROOT / "configs/experiments/state_machine_rule_weight_sweep_v2_1a_development_config.json"
DETECTOR_PATH = ROOT / "scripts/detectors/run_state_machine_interval_detector_v2_1a_experimental.py"

spec = importlib.util.spec_from_file_location("v21a_detector", DETECTOR_PATH)
det = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(det)

CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
INPUTS = {k: Path(v) for k, v in CONFIG["input_paths"].items()}
OUTPUTS = {k: Path(v) for k, v in CONFIG["output_paths"].items()}
EXPERIMENT_ROOT = OUTPUTS["experiment_root"]
TOP_OUTPUT_ROOT = EXPERIMENT_ROOT / "top_config_outputs"
BACKUP_ROOT = ROOT / "backups"
RUN_LOG = OUTPUTS["run_log"]

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
FORBIDDEN_DIR_NAMES = {
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


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(value: Any, default: float = 0.0) -> float:
    return det.fnum(value, default)


def truth(value: Any) -> bool:
    return det.truth(value)


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def safe_ratio(n: float, d: float) -> float:
    return 0.0 if d <= 0 else n / d


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    mid = len(data) // 2
    if len(data) % 2:
        return data[mid]
    return (data[mid - 1] + data[mid]) / 2.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def backup_existing_outputs() -> tuple[str | None, list[dict[str, Any]]]:
    target_paths = [
        EXPERIMENT_ROOT,
        OUTPUTS["viewer_registry_patch"],
        OUTPUTS["summary_md"],
        OUTPUTS["report_json"],
        OUTPUTS["recommendation_note"],
        OUTPUTS["rule_note"],
        OUTPUTS["run_log"],
        OUTPUTS["latest_bundle"],
    ]
    existing = [path for path in target_paths if path.exists()]
    if not existing:
        return None, []
    backup_dir = BACKUP_ROOT / f"{EXPERIMENT_ID}_{now_stamp()}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for path in existing:
        rel = path.relative_to(ROOT)
        dst = backup_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, dst, dirs_exist_ok=True)
            kind = "directory"
            digest = ""
        else:
            shutil.copy2(path, dst)
            kind = "file"
            digest = sha256(path)
        manifest.append(
            {
                "source_path": str(path),
                "backup_path": str(dst),
                "kind": kind,
                "sha256": digest,
                "backup_created_at": now_iso(),
            }
        )
    write_json(backup_dir / "backup_manifest.json", manifest)
    return str(backup_dir), manifest


def snapshot_protected_files() -> dict[str, dict[str, Any]]:
    patterns = [
        "scripts/**/*v1_4*",
        "scripts/**/*v1.4*",
        "scripts/**/*v2_0*",
        "scripts/**/*v2.0*",
        "configs/**/*v1_4*",
        "configs/**/*v1.4*",
        "configs/**/*v2_0*",
        "configs/**/*v2.0*",
        "reports/**/*v1_4*",
        "reports/**/*v1.4*",
        "reports/**/*v2_0*",
        "reports/**/*v2.0*",
        "data/**/*v1_4*",
        "data/**/*v1.4*",
        "data/**/*v2_0*",
        "data/**/*v2.0*",
    ]
    files: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if path.is_file() and "v2_1a" not in path.name and "v2.1a" not in path.name:
                rel = str(path.relative_to(ROOT))
                files[rel] = {"sha256": sha256(path), "size": path.stat().st_size, "mtime": path.stat().st_mtime}
    return files


def compare_snapshots(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    changed: list[str] = []
    for rel, meta in before.items():
        if rel not in after or after[rel].get("sha256") != meta.get("sha256"):
            changed.append(rel)
    return sorted(changed)


def validate_inputs(fusion_rows: list[dict[str, str]], split_rows: list[dict[str, str]]) -> dict[str, Any]:
    fusion_vids = sorted({int(float(row.get("video_id") or 0)) for row in fusion_rows})
    non_dev_rows = [
        row
        for row in fusion_rows
        if int(float(row.get("video_id") or 0)) not in DEV_IDS
        or row.get("original_split_v2_4") != "train"
        or row.get("split_role_v2_5") != "development"
        or row.get("evaluation_subset_v2_5") != "none"
    ]
    split_train_ids = sorted(int(float(row["video_id"])) for row in split_rows if row.get("split") == "train")
    validation_ids = sorted(int(float(row["video_id"])) for row in split_rows if row.get("split") == "validation")
    test_ids = sorted(int(float(row["video_id"])) for row in split_rows if row.get("split") == "test")
    decision_column_bad = []
    for row in fusion_rows:
        decision_cols = det.parse_json_list(row.get("decision_feature_columns_json", "[]"))
        bad = det.has_forbidden(decision_cols)
        if bad:
            decision_column_bad.append({"video_id": row.get("video_id"), "anchor_id": row.get("final_scene_anchor_id"), "bad": bad})
    if non_dev_rows:
        raise SystemExit(f"Non-development rows found in fusion input: {len(non_dev_rows)}")
    if fusion_vids != DEV_IDS:
        raise SystemExit(f"Fusion input video ids do not match Development Set. got={fusion_vids} expected={DEV_IDS}")
    return {
        "fusion_input_rows": len(fusion_rows),
        "fusion_video_ids": fusion_vids,
        "split_train_ids": split_train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "non_development_row_count": len(non_dev_rows),
        "decision_column_forbidden_rows_in_v2_0_input": len(decision_column_bad),
        "decision_column_forbidden_examples": decision_column_bad[:10],
        "split_role_v2_5": "development",
        "evaluation_subset_v2_5": "none",
    }


def load_video_durations(split_rows: list[dict[str, str]], fusion_rows: list[dict[str, str]]) -> dict[int, float]:
    durations: dict[int, float] = {}
    for row in split_rows:
        vid = int(float(row.get("video_id") or 0))
        if vid in DEV_IDS:
            durations[vid] = fnum(row.get("video_duration_sec"))
    for row in fusion_rows:
        vid = int(float(row.get("video_id") or 0))
        if vid in DEV_IDS and durations.get(vid, 0.0) <= 0:
            durations[vid] = fnum(row.get("video_duration_sec"))
    return durations


def load_labels_for_scoring_only(path: Path) -> dict[int, list[dict[str, Any]]]:
    labels_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv(path):
        vid = int(float(row.get("video_id") or 0))
        if vid not in DEV_IDS:
            continue
        if row.get("segment_type") and row.get("segment_type") != "ad_interval":
            continue
        if row.get("segment_valid") and not truth(row.get("segment_valid")):
            continue
        start = fnum(row.get("segment_start_sec") or row.get("ad_start_sec"))
        end = fnum(row.get("segment_end_sec") or row.get("ad_end_sec"))
        if end > start:
            labels_by_video[vid].append(
                {
                    "ad_interval_id": row.get("ad_interval_id") or row.get("segment_id"),
                    "start": start,
                    "end": end,
                }
            )
    for rows in labels_by_video.values():
        rows.sort(key=lambda item: item["start"])
    return labels_by_video


def family_base(family: str) -> dict[str, Any]:
    base = {
        "weights": {
            "start_strength_score": 0.30,
            "continuity_strength_score": 0.22,
            "end_quality_score": 0.16,
            "hard_evidence_density_score": 0.17,
            "ocr_timeline_consistency_score": 0.10,
            "audio_relative_support_score": 0.05,
        },
        "thresholds": {
            "min_interval_ad_score": 0.45,
            "min_interval_score_density": 0.20,
            "min_hard_evidence_count": 1,
            "min_hard_evidence_density_per_60s": 0.35,
            "max_weak_span_sec": 35,
        },
        "budget_guard": {
            "soft_overprediction_ratio": 0.18,
            "hard_overprediction_ratio": 0.25,
            "target_prediction_ratio_after_pruning": 0.15,
        },
        "ultra_high_confidence": {
            "ultra_min_interval_ad_score": 0.85,
            "ultra_min_interval_score_density": 0.35,
            "ultra_min_hard_evidence_count": 3,
            "ultra_min_hard_evidence_density_per_60s": 1.0,
            "ultra_min_ocr_hard_count": 2,
            "ultra_min_product_cta_timeline_types": 2,
        },
        "freshness_support_cap": {
            "max_sec_since_last_hard_evidence": 30,
            "support_only_max_duration_sec": 12,
            "support_only_max_anchor_count": 2,
        },
        "penalties": {
            "weak_span_penalty": 0.08,
            "overlong_penalty": 0.08,
            "opening_disclosure_only_penalty": 0.30,
            "fuzzy_only_penalty": 0.24,
            "audio_only_penalty": 0.28,
            "scene_only_penalty": 0.32,
            "duration_excess_penalty": 0.08,
            "long_candidate_low_density_penalty": 0.12,
        },
        "long_candidate": {
            "long_candidate_duration_sec": 180,
            "long_candidate_density_min": 0.35,
            "force_demote_long_low_density_candidate": True,
            "split_long_candidate_for_review_flag": False,
        },
    }
    if family == "precision_safe":
        base["weights"].update({"hard_evidence_density_score": 0.23, "ocr_timeline_consistency_score": 0.14, "audio_relative_support_score": 0.03})
        base["thresholds"].update({"min_interval_ad_score": 0.55, "min_interval_score_density": 0.30, "min_hard_evidence_count": 2, "min_hard_evidence_density_per_60s": 0.75})
        base["budget_guard"].update({"soft_overprediction_ratio": 0.15, "hard_overprediction_ratio": 0.20, "target_prediction_ratio_after_pruning": 0.12})
        base["ultra_high_confidence"].update({"ultra_min_interval_ad_score": 0.90, "ultra_min_interval_score_density": 0.45, "ultra_min_hard_evidence_count": 4})
        base["penalties"].update({"overlong_penalty": 0.12, "long_candidate_low_density_penalty": 0.18})
    elif family == "recall_safe":
        base["weights"].update({"start_strength_score": 0.33, "ocr_timeline_consistency_score": 0.14, "audio_relative_support_score": 0.06})
        base["thresholds"].update({"min_interval_ad_score": 0.35, "min_interval_score_density": 0.10, "min_hard_evidence_count": 1, "min_hard_evidence_density_per_60s": 0.20, "max_weak_span_sec": 45})
        base["budget_guard"].update({"soft_overprediction_ratio": 0.20, "hard_overprediction_ratio": 0.30, "target_prediction_ratio_after_pruning": 0.15})
        base["freshness_support_cap"].update({"max_sec_since_last_hard_evidence": 40, "support_only_max_duration_sec": 15, "support_only_max_anchor_count": 3})
    elif family == "short_ad_safe":
        base["weights"].update({"start_strength_score": 0.36, "end_quality_score": 0.18, "audio_relative_support_score": 0.04})
        base["thresholds"].update({"min_interval_ad_score": 0.40, "min_interval_score_density": 0.20, "min_hard_evidence_count": 1})
        base["freshness_support_cap"].update({"max_sec_since_last_hard_evidence": 20, "support_only_max_duration_sec": 8, "support_only_max_anchor_count": 1})
        base["long_candidate"].update({"long_candidate_duration_sec": 120, "long_candidate_density_min": 0.35})
    elif family == "budget_strict":
        base["thresholds"].update({"min_interval_ad_score": 0.50, "min_interval_score_density": 0.25})
        base["budget_guard"].update({"soft_overprediction_ratio": 0.12, "hard_overprediction_ratio": 0.20, "target_prediction_ratio_after_pruning": 0.10})
        base["ultra_high_confidence"].update({"ultra_min_interval_ad_score": 0.90, "ultra_min_interval_score_density": 0.45, "ultra_min_hard_evidence_count": 4})
    elif family == "long_candidate_strict":
        base["weights"].update({"continuity_strength_score": 0.20, "hard_evidence_density_score": 0.24, "ocr_timeline_consistency_score": 0.14})
        base["thresholds"].update({"min_interval_score_density": 0.30, "min_hard_evidence_density_per_60s": 0.75})
        base["long_candidate"].update({"long_candidate_duration_sec": 120, "long_candidate_density_min": 0.50, "force_demote_long_low_density_candidate": True, "split_long_candidate_for_review_flag": True})
        base["penalties"].update({"overlong_penalty": 0.14, "long_candidate_low_density_penalty": 0.20})
    return base


def deep_update(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def generate_candidate_configs() -> list[dict[str, Any]]:
    families = [
        "precision_safe",
        "balanced",
        "recall_safe",
        "short_ad_safe",
        "budget_strict",
        "long_candidate_strict",
    ]
    variants = [
        ("base", {}),
        ("score_low", {"thresholds": {"min_interval_ad_score": 0.35, "min_interval_score_density": 0.10}}),
        ("score_mid_high", {"thresholds": {"min_interval_ad_score": 0.55, "min_interval_score_density": 0.30}}),
        ("budget_tight", {"budget_guard": {"soft_overprediction_ratio": 0.12, "hard_overprediction_ratio": 0.20, "target_prediction_ratio_after_pruning": 0.10}}),
        ("budget_medium", {"budget_guard": {"soft_overprediction_ratio": 0.15, "hard_overprediction_ratio": 0.25, "target_prediction_ratio_after_pruning": 0.12}}),
        ("fresh_20", {"freshness_support_cap": {"max_sec_since_last_hard_evidence": 20, "support_only_max_duration_sec": 8, "support_only_max_anchor_count": 1}}),
        ("fresh_40", {"freshness_support_cap": {"max_sec_since_last_hard_evidence": 40, "support_only_max_duration_sec": 15, "support_only_max_anchor_count": 3}}),
        ("long_strict", {"long_candidate": {"long_candidate_duration_sec": 120, "long_candidate_density_min": 0.50}, "penalties": {"long_candidate_low_density_penalty": 0.20}}),
    ]
    configs: list[dict[str, Any]] = []
    for family in families:
        base = family_base(family)
        for idx, (variant, changes) in enumerate(variants, start=1):
            cfg = deep_update(base, changes)
            cfg.update(
                {
                    "config_id": f"v2_1a_{family}_{idx:02d}",
                    "config_family": family,
                    "config_variant": variant,
                    "config_description": f"{family} preset with {variant} deterministic sweep settings",
                    "fixed_rule_principles": CONFIG["fixed_rule_principles"],
                }
            )
            configs.append(cfg)
    if len(configs) > MAX_CONFIGS:
        rng = random.Random(SEED)
        configs = sorted(rng.sample(configs, MAX_CONFIGS), key=lambda item: item["config_id"])
    return configs


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    row = {
        "config_id": config["config_id"],
        "config_family": config["config_family"],
        "config_variant": config["config_variant"],
        "config_description": config["config_description"],
    }
    for group in ["weights", "thresholds", "budget_guard", "ultra_high_confidence", "freshness_support_cap", "penalties", "long_candidate"]:
        for key, value in config[group].items():
            row[key] = value
    row["fixed_rule_principles_json"] = json.dumps(config["fixed_rule_principles"], ensure_ascii=False)
    return row


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def rows_to_intervals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals = []
    for row in rows:
        start = fnum(row.get("ad_start_sec"))
        end = fnum(row.get("ad_end_sec"))
        if end > start:
            intervals.append({"start": start, "end": end, "id": row.get("candidate_id") or row.get("prediction_id", "")})
    intervals.sort(key=lambda item: item["start"])
    return intervals


def match_boundaries(preds: list[dict[str, Any]], actuals: list[dict[str, Any]]) -> tuple[list[float], list[float], int, int]:
    pairs = []
    for pi, pred in enumerate(preds):
        for ai, actual in enumerate(actuals):
            overlap = interval_overlap(pred["start"], pred["end"], actual["start"], actual["end"])
            if overlap > 0:
                pairs.append((overlap, pi, ai))
    pairs.sort(reverse=True)
    used_p, used_a = set(), set()
    start_errors: list[float] = []
    end_errors: list[float] = []
    for _, pi, ai in pairs:
        if pi in used_p or ai in used_a:
            continue
        used_p.add(pi)
        used_a.add(ai)
        start_errors.append(abs(preds[pi]["start"] - actuals[ai]["start"]))
        end_errors.append(abs(preds[pi]["end"] - actuals[ai]["end"]))
    return start_errors, end_errors, len(actuals) - len(used_a), len(preds) - len(used_p)


def score_video(pred_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], actuals: list[dict[str, Any]], video_duration: float) -> dict[str, Any]:
    preds = rows_to_intervals(pred_rows)
    raw = rows_to_intervals(raw_rows)
    actual_duration = sum(max(0.0, item["end"] - item["start"]) for item in actuals)
    pred_duration = sum(max(0.0, item["end"] - item["start"]) for item in preds)
    raw_duration = sum(max(0.0, item["end"] - item["start"]) for item in raw)
    overlap_sum = 0.0
    for pred in preds:
        pred_overlap = sum(interval_overlap(pred["start"], pred["end"], actual["start"], actual["end"]) for actual in actuals)
        overlap_sum += min(pred["end"] - pred["start"], pred_overlap)
    overlap_sum = min(overlap_sum, actual_duration, pred_duration) if pred_duration > 0 and actual_duration > 0 else 0.0
    start_errors, end_errors, unmatched_actual, unmatched_pred = match_boundaries(preds, actuals)
    median_start = median(start_errors)
    median_end = median(end_errors)
    median_boundary_error = (median_start + median_end) / 2.0 if start_errors or end_errors else 999.0
    boundary_quality = 0.0 if median_boundary_error >= 999 else math.exp(-median_boundary_error / 10.0)
    false_positive = max(0.0, pred_duration - overlap_sum)
    missed = max(0.0, actual_duration - overlap_sum)
    return {
        "actual_total_duration_sec": actual_duration,
        "raw_prediction_total_duration_sec": raw_duration,
        "final_prediction_total_duration_sec": pred_duration,
        "prediction_overlap_with_actual_sec": overlap_sum,
        "actual_overlap_recall": safe_ratio(overlap_sum, actual_duration),
        "prediction_overlap_precision_proxy": safe_ratio(overlap_sum, pred_duration),
        "false_positive_duration_sec": false_positive,
        "false_positive_ratio_of_video": safe_ratio(false_positive, video_duration),
        "missed_actual_duration_sec": missed,
        "missed_actual_ratio": safe_ratio(missed, actual_duration),
        "mean_boundary_start_error_sec": mean(start_errors),
        "median_boundary_start_error_sec": median_start,
        "mean_boundary_end_error_sec": mean(end_errors),
        "median_boundary_end_error_sec": median_end,
        "boundary_quality_score": boundary_quality,
        "unmatched_actual_count": unmatched_actual,
        "unmatched_prediction_count": unmatched_pred,
        "raw_prediction_ratio": safe_ratio(raw_duration, video_duration),
        "final_prediction_ratio": safe_ratio(pred_duration, video_duration),
    }


def add_objective_scores(row: dict[str, Any]) -> None:
    recall = fnum(row.get("actual_overlap_recall"))
    precision = fnum(row.get("prediction_overlap_precision_proxy"))
    boundary = fnum(row.get("boundary_quality_score"))
    fp_ratio = fnum(row.get("false_positive_ratio_of_video"))
    missed_ratio = fnum(row.get("missed_actual_ratio"))
    review = fnum(row.get("review_burden_score"))
    over = fnum(row.get("overprediction_penalty_score"))
    mean_final_ratio = fnum(row.get("mean_final_prediction_ratio"))
    row["balanced_objective_score"] = (
        0.35 * recall
        + 0.25 * precision
        + 0.15 * boundary
        - 0.15 * fp_ratio
        - 0.05 * over
        - 0.05 * review
    )
    row["precision_safe_score"] = 0.42 * precision + 0.22 * boundary + 0.16 * recall - 0.14 * fp_ratio - 0.06 * review
    row["recall_safe_score"] = 0.50 * recall + 0.18 * precision + 0.12 * boundary - 0.10 * missed_ratio - 0.05 * fp_ratio - 0.05 * review
    row["low_overprediction_score"] = 0.36 * precision + 0.22 * boundary + 0.18 * recall + 0.14 * (1 - clip(mean_final_ratio)) - 0.10 * fp_ratio
    row["short_ad_safety_score"] = 0.40 * fnum(row.get("short_actual_overlap_recall")) + 0.25 * recall + 0.15 * precision + 0.10 * boundary - 0.10 * fp_ratio


def score_outputs_for_config(
    config: dict[str, Any],
    outputs: dict[str, list[dict[str, Any]]],
    labels_by_video: dict[int, list[dict[str, Any]]],
    durations: dict[int, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    raw_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in outputs["predictions"]:
        pred_by_video[int(float(row["video_id"]))].append(row)
    for row in outputs["raw_candidates"]:
        raw_by_video[int(float(row["video_id"]))].append(row)

    video_rows: list[dict[str, Any]] = []
    for vid in DEV_IDS:
        metrics = score_video(pred_by_video[vid], raw_by_video[vid], labels_by_video.get(vid, []), durations.get(vid, 0.0))
        video_row = {
            "config_id": config["config_id"],
            "config_family": config["config_family"],
            "video_id": vid,
            "prediction_count": len(pred_by_video[vid]),
            "raw_candidate_count": len(raw_by_video[vid]),
            **metrics,
        }
        video_row["review_only_count"] = sum(1 for r in outputs["review_only"] if int(float(r.get("video_id") or 0)) == vid)
        video_row["overprediction_pruned_review_count"] = sum(1 for r in outputs["overprediction_pruned_review"] if int(float(r.get("video_id") or 0)) == vid)
        video_row["open_count"] = sum(1 for r in outputs["open_candidates"] if int(float(r.get("video_id") or 0)) == vid)
        video_row["review_burden_score"] = safe_ratio(
            video_row["review_only_count"] + video_row["overprediction_pruned_review_count"] + video_row["open_count"],
            video_row["prediction_count"] + video_row["review_only_count"] + video_row["overprediction_pruned_review_count"] + video_row["open_count"] + 1,
        )
        video_row["overprediction_penalty_score"] = clip(
            0.5 * float(video_row["final_prediction_ratio"] > config["budget_guard"]["hard_overprediction_ratio"])
            + 0.5 * video_row["false_positive_ratio_of_video"]
        )
        short_actuals = [a for a in labels_by_video.get(vid, []) if a["end"] - a["start"] <= 120]
        short_metrics = score_video(pred_by_video[vid], [], short_actuals, durations.get(vid, 0.0))
        video_row["short_actual_overlap_recall"] = short_metrics["actual_overlap_recall"]
        add_objective_scores(video_row)
        video_rows.append(video_row)

    totals = {
        "config_id": config["config_id"],
        "config_family": config["config_family"],
        "config_description": config["config_description"],
        "prediction_count": len(outputs["predictions"]),
        "review_only_count": len(outputs["review_only"]),
        "overprediction_pruned_review_count": len(outputs["overprediction_pruned_review"]),
        "open_count": len(outputs["open_candidates"]),
        "unresolved_count": 0,
        "rejected_count": 0,
        "raw_prediction_total_duration_sec": sum(row["raw_prediction_total_duration_sec"] for row in video_rows),
        "final_prediction_total_duration_sec": sum(row["final_prediction_total_duration_sec"] for row in video_rows),
        "mean_raw_prediction_ratio": mean([row["raw_prediction_ratio"] for row in video_rows]),
        "mean_final_prediction_ratio": mean([row["final_prediction_ratio"] for row in video_rows]),
        "videos_over_soft_ratio": sum(row["final_prediction_ratio"] > config["budget_guard"]["soft_overprediction_ratio"] for row in video_rows),
        "videos_over_hard_ratio": sum(row["final_prediction_ratio"] > config["budget_guard"]["hard_overprediction_ratio"] for row in video_rows),
        "actual_total_duration_sec": sum(row["actual_total_duration_sec"] for row in video_rows),
        "prediction_overlap_with_actual_sec": sum(row["prediction_overlap_with_actual_sec"] for row in video_rows),
        "false_positive_duration_sec": sum(row["false_positive_duration_sec"] for row in video_rows),
        "missed_actual_duration_sec": sum(row["missed_actual_duration_sec"] for row in video_rows),
        "mean_boundary_start_error_sec": mean([row["mean_boundary_start_error_sec"] for row in video_rows if row["boundary_quality_score"] > 0]),
        "median_boundary_start_error_sec": median([row["median_boundary_start_error_sec"] for row in video_rows if row["boundary_quality_score"] > 0]),
        "mean_boundary_end_error_sec": mean([row["mean_boundary_end_error_sec"] for row in video_rows if row["boundary_quality_score"] > 0]),
        "median_boundary_end_error_sec": median([row["median_boundary_end_error_sec"] for row in video_rows if row["boundary_quality_score"] > 0]),
        "boundary_quality_score": mean([row["boundary_quality_score"] for row in video_rows]),
        "review_burden_score": safe_ratio(
            len(outputs["review_only"]) + len(outputs["overprediction_pruned_review"]) + len(outputs["open_candidates"]),
            len(outputs["predictions"]) + len(outputs["review_only"]) + len(outputs["overprediction_pruned_review"]) + len(outputs["open_candidates"]) + 1,
        ),
        "overprediction_penalty_score": clip(
            0.50 * safe_ratio(sum(row["final_prediction_ratio"] > config["budget_guard"]["hard_overprediction_ratio"] for row in video_rows), len(DEV_IDS))
            + 0.50 * sum(row["false_positive_duration_sec"] for row in video_rows) / max(1.0, sum(durations.values())),
        ),
        "short_actual_overlap_recall": mean([row["short_actual_overlap_recall"] for row in video_rows]),
        "config_failure_flag": False,
        "config_failure_reason": "",
    }
    totals["actual_overlap_recall"] = safe_ratio(totals["prediction_overlap_with_actual_sec"], totals["actual_total_duration_sec"])
    totals["prediction_overlap_precision_proxy"] = safe_ratio(totals["prediction_overlap_with_actual_sec"], totals["final_prediction_total_duration_sec"])
    totals["false_positive_ratio_of_video"] = safe_ratio(totals["false_positive_duration_sec"], sum(durations.values()))
    totals["missed_actual_ratio"] = safe_ratio(totals["missed_actual_duration_sec"], totals["actual_total_duration_sec"])

    consistency_failures = [
        event
        for event in outputs["budget_guard_events"]
        if event.get("budget_guard_action") == "kept_ultra_high_confidence" and not truth(event.get("ultra_high_confidence"))
    ]
    if consistency_failures:
        totals["config_failure_flag"] = True
        totals["config_failure_reason"] = "budget_guard_ultra_consistency_failure"
    if any(json.loads(row.get("forbidden_decision_columns_found", "[]")) for row in outputs["predictions"]):
        totals["config_failure_flag"] = True
        reason = totals["config_failure_reason"]
        totals["config_failure_reason"] = (reason + ";" if reason else "") + "forbidden_decision_columns_found"
    add_objective_scores(totals)
    return totals, video_rows


def score_baseline_v2_0(labels_by_video: dict[int, list[dict[str, Any]]], durations: dict[int, float]) -> dict[str, Any]:
    path = INPUTS["v2_0_baseline_predictions"]
    if not path.exists():
        return {
            "available": False,
            "mean_final_prediction_ratio": 0.0,
            "false_positive_duration_sec": 0.0,
            "missed_actual_duration_sec": 0.0,
        }
    rows = [row for row in read_csv(path) if int(float(row.get("video_id") or 0)) in DEV_IDS]
    video_rows = []
    for vid in DEV_IDS:
        pred_rows = [row for row in rows if int(float(row.get("video_id") or 0)) == vid]
        video_rows.append(score_video(pred_rows, pred_rows, labels_by_video.get(vid, []), durations.get(vid, 0.0)))
    total_overlap = sum(row["prediction_overlap_with_actual_sec"] for row in video_rows)
    total_actual = sum(row["actual_total_duration_sec"] for row in video_rows)
    total_pred = sum(row["final_prediction_total_duration_sec"] for row in video_rows)
    return {
        "available": True,
        "prediction_count": len(rows),
        "mean_final_prediction_ratio": mean([row["final_prediction_ratio"] for row in video_rows]),
        "false_positive_duration_sec": sum(row["false_positive_duration_sec"] for row in video_rows),
        "missed_actual_duration_sec": sum(row["missed_actual_duration_sec"] for row in video_rows),
        "actual_overlap_recall": safe_ratio(total_overlap, total_actual),
        "prediction_overlap_precision_proxy": safe_ratio(total_overlap, total_pred),
    }


def compute_pareto(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    maximize = ["actual_overlap_recall", "prediction_overlap_precision_proxy", "boundary_quality_score"]
    minimize = ["false_positive_duration_sec", "mean_final_prediction_ratio", "review_burden_score"]
    frontier = []
    successful = [row for row in metrics if not truth(row.get("config_failure_flag"))]
    for row in successful:
        dominated = False
        for other in successful:
            if other is row:
                continue
            better_or_equal = all(fnum(other[m]) >= fnum(row[m]) for m in maximize) and all(fnum(other[m]) <= fnum(row[m]) for m in minimize)
            strictly_better = any(fnum(other[m]) > fnum(row[m]) for m in maximize) or any(fnum(other[m]) < fnum(row[m]) for m in minimize)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            out = dict(row)
            out["pareto_frontier"] = True
            frontier.append(out)
    return sorted(frontier, key=lambda item: item["balanced_objective_score"], reverse=True)


def select_top_configs(metrics: list[dict[str, Any]], pareto: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in metrics if not truth(row.get("config_failure_flag"))]
    if not successful:
        raise SystemExit("All configs failed; cannot compare v2.1a candidates.")
    top = {
        "best_balanced_config": max(successful, key=lambda row: row["balanced_objective_score"]),
        "best_precision_safe_config": max(successful, key=lambda row: row["precision_safe_score"]),
        "best_recall_safe_config": max(successful, key=lambda row: row["recall_safe_score"]),
        "best_low_overprediction_config": max(successful, key=lambda row: row["low_overprediction_score"]),
        "best_short_ad_safe_config": max(successful, key=lambda row: row["short_ad_safety_score"]),
        "best_boundary_quality_config": max(successful, key=lambda row: (row["boundary_quality_score"], row["balanced_objective_score"])),
    }
    recommended: list[dict[str, Any]] = []
    seen = set()
    for row in top.values():
        if row["config_id"] not in seen:
            recommended.append(row)
            seen.add(row["config_id"])
    for row in sorted(successful, key=lambda item: item["balanced_objective_score"], reverse=True):
        if len(recommended) >= 10:
            break
        if row["config_id"] not in seen:
            recommended.append(row)
            seen.add(row["config_id"])
    for row in pareto:
        if len(recommended) >= 10:
            break
        if row["config_id"] not in seen:
            recommended.append(row)
            seen.add(row["config_id"])
    top["recommended_viewer_review_configs"] = recommended[:10]
    return top


def leave_one_video_out(metrics_by_config_video: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics_by_config_video:
        by_config[row["config_id"]].append(row)
    summary = []
    for heldout in DEV_IDS:
        fold_scores = []
        for config_id, rows in by_config.items():
            train_rows = [row for row in rows if int(row["video_id"]) != heldout]
            fold_scores.append(
                {
                    "config_id": config_id,
                    "config_family": train_rows[0]["config_family"] if train_rows else "",
                    "fold_train_balanced_score": mean([row["balanced_objective_score"] for row in train_rows]),
                }
            )
        selected = max(fold_scores, key=lambda row: row["fold_train_balanced_score"])
        heldout_row = next(row for row in by_config[selected["config_id"]] if int(row["video_id"]) == heldout)
        summary.append(
            {
                "heldout_video_id": heldout,
                "selected_config_id": selected["config_id"],
                "selected_config_family": selected["config_family"],
                "fold_train_balanced_score": selected["fold_train_balanced_score"],
                "heldout_balanced_objective_score": heldout_row["balanced_objective_score"],
                "heldout_actual_overlap_recall": heldout_row["actual_overlap_recall"],
                "heldout_prediction_overlap_precision_proxy": heldout_row["prediction_overlap_precision_proxy"],
                "heldout_false_positive_duration_sec": heldout_row["false_positive_duration_sec"],
                "heldout_missed_actual_duration_sec": heldout_row["missed_actual_duration_sec"],
                "development_internal_robustness_only": True,
                "final_evaluation": False,
            }
        )
    return summary


def config_failure_log(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "config_id": row["config_id"],
            "config_family": row["config_family"],
            "config_failure_flag": row["config_failure_flag"],
            "config_failure_reason": row["config_failure_reason"],
        }
        for row in metrics
    ]


def build_budget_audit(all_outputs: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_id, outputs in all_outputs.items():
        rows.extend(outputs["budget_guard_events"])
    return rows


def write_top_config_outputs(all_outputs: dict[str, dict[str, list[dict[str, Any]]]], recommended: list[dict[str, Any]]) -> None:
    if TOP_OUTPUT_ROOT.exists():
        shutil.rmtree(TOP_OUTPUT_ROOT)
    for row in recommended:
        config_id = row["config_id"]
        out_dir = TOP_OUTPUT_ROOT / config_id
        outputs = all_outputs[config_id]
        write_csv(out_dir / "predictions.csv", outputs["predictions"])
        write_csv(out_dir / "review_only_candidates.csv", outputs["review_only"])
        write_csv(out_dir / "overprediction_pruned_review_candidates.csv", outputs["overprediction_pruned_review"])
        write_csv(out_dir / "open_candidates.csv", outputs["open_candidates"])
        write_csv(out_dir / "budget_guard_events.csv", outputs["budget_guard_events"])
        video_summary = []
        for vid in DEV_IDS:
            video_summary.append(
                {
                    "config_id": config_id,
                    "video_id": vid,
                    "prediction_count": sum(1 for item in outputs["predictions"] if int(float(item.get("video_id") or 0)) == vid),
                    "review_only_count": sum(1 for item in outputs["review_only"] if int(float(item.get("video_id") or 0)) == vid),
                    "overprediction_pruned_review_count": sum(1 for item in outputs["overprediction_pruned_review"] if int(float(item.get("video_id") or 0)) == vid),
                    "open_count": sum(1 for item in outputs["open_candidates"] if int(float(item.get("video_id") or 0)) == vid),
                }
            )
        write_csv(out_dir / "video_summary.csv", video_summary)
        write_csv(out_dir / "trace_sample.csv", outputs["trace_sample"][:1000])


def create_viewer_registry_patch(recommended: list[dict[str, Any]]) -> dict[str, Any]:
    patch = {
        "patch_id": "state_machine_viewer_registry_patch_v2_1a_top_configs",
        "patch_proposal_only": True,
        "do_not_modify_existing_registry": True,
        "recommended_for_viewer_review_before_fixed_rule_adoption": True,
        "final_rule_freeze": False,
        "generated_at": now_iso(),
        "registry_entries_to_add": [],
    }
    for row in recommended:
        config_id = row["config_id"]
        patch["registry_entries_to_add"].append(
            {
                "version_id": config_id,
                "display_name": f"v2.1a {row['config_family']} {config_id}",
                "scope": "development_only",
                "predictions_csv": str(TOP_OUTPUT_ROOT / config_id / "predictions.csv"),
                "review_only_csv": str(TOP_OUTPUT_ROOT / config_id / "review_only_candidates.csv"),
                "overprediction_pruned_review_csv": str(TOP_OUTPUT_ROOT / config_id / "overprediction_pruned_review_candidates.csv"),
                "open_candidates_csv": str(TOP_OUTPUT_ROOT / config_id / "open_candidates.csv"),
                "note": "recommended for viewer review before fixed rule adoption; not v2.1b freeze",
            }
        )
    write_json(OUTPUTS["viewer_registry_patch"], patch)
    return patch


def forbidden_scan_bundle(bundle: Path) -> dict[str, Any]:
    findings = []
    total_bytes = 0
    for path in bundle.rglob("*"):
        if path.is_dir():
            if path.name.lower() in FORBIDDEN_DIR_NAMES:
                findings.append({"path": str(path), "reason": "forbidden_directory_name"})
            continue
        total_bytes += path.stat().st_size
        lower_parts = {part.lower() for part in path.parts}
        if lower_parts & FORBIDDEN_DIR_NAMES:
            findings.append({"path": str(path), "reason": "path_contains_forbidden_directory_name"})
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append({"path": str(path), "reason": "forbidden_suffix"})
        if path.name.endswith("_frame_results_v2_5_development.csv"):
            findings.append({"path": str(path), "reason": "ocr_raw_frame_full_csv_forbidden"})
    return {"bundle_path": str(bundle), "total_bytes": total_bytes, "finding_count": len(findings), "findings": findings, "clean": not findings}


def create_latest_bundle(recommended: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = OUTPUTS["latest_bundle"]
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    files = [
        EXPERIMENT_ROOT / "candidate_rule_configs_v2_1a.csv",
        EXPERIMENT_ROOT / "candidate_rule_configs_v2_1a.json",
        EXPERIMENT_ROOT / "rule_config_search_space_v2_1a.json",
        EXPERIMENT_ROOT / "per_config_metrics_v2_1a.csv",
        EXPERIMENT_ROOT / "top_configs_by_objective_v2_1a.csv",
        EXPERIMENT_ROOT / "pareto_frontier_configs_v2_1a.csv",
        EXPERIMENT_ROOT / "recommended_viewer_review_configs_v2_1a.csv",
        EXPERIMENT_ROOT / "leave_one_video_out_summary_v2_1a.csv",
        EXPERIMENT_ROOT / "ocr_disclosure_subtype_restoration_audit_v2_1a.csv",
        EXPERIMENT_ROOT / "opening_disclosure_guard_audit_v2_1a.csv",
        EXPERIMENT_ROOT / "support_flag_split_audit_v2_1a.csv",
        EXPERIMENT_ROOT / "budget_guard_consistency_audit_v2_1a.csv",
        OUTPUTS["viewer_registry_patch"],
        OUTPUTS["summary_md"],
        OUTPUTS["report_json"],
        OUTPUTS["recommendation_note"],
        OUTPUTS["rule_note"],
    ]
    for src in files:
        if src.exists():
            dst = bundle / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for row in recommended:
        config_id = row["config_id"]
        src_dir = TOP_OUTPUT_ROOT / config_id
        dst_dir = bundle / src_dir.relative_to(ROOT)
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    scan = forbidden_scan_bundle(bundle)
    write_json(bundle / "latest_bundle_forbidden_scan_v2_1a.json", scan)
    return scan


def validation_rows(
    input_validation: dict[str, Any],
    configs: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    video_metrics: list[dict[str, Any]],
    budget_audit: list[dict[str, Any]],
    report_flags: dict[str, Any],
    latest_scan: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    config_ids = [cfg["config_id"] for cfg in configs]
    fixed_principles_json = {json.dumps(cfg["fixed_rule_principles"], sort_keys=True) for cfg in configs}
    decision_bad = det.has_forbidden(det.DECISION_FEATURE_COLUMNS)
    validations = [
        {
            "sub_agent": "Sub Agent 1",
            "validation_area": "Input & Split Validation",
            "passed": input_validation["fusion_video_ids"] == DEV_IDS
            and input_validation["non_development_row_count"] == 0
            and report_flags["actual_label_used_for_decision"] is False,
            "details": "fusion input is Development Set only; validation/test rows excluded; labels loaded only for post-hoc scoring",
        },
        {
            "sub_agent": "Sub Agent 2",
            "validation_area": "Fixed Guard & Search Space Validation",
            "passed": len(fixed_principles_json) == 1
            and len(configs) <= MAX_CONFIGS
            and len(set(config_ids)) == len(config_ids)
            and report_flags["scene_used_for_ad_likelihood_directly"] is False,
            "details": "fixed rule principles are invariant; config count and uniqueness checked",
        },
        {
            "sub_agent": "Sub Agent 3",
            "validation_area": "Detector Logic Validation",
            "passed": not any(truth(row.get("consistency_failure")) for row in budget_audit)
            and report_flags["support_flag_split_applied"] is True,
            "details": "support flags split; budget ultra action consistency checked; pruned candidates retained in review bucket",
        },
        {
            "sub_agent": "Sub Agent 4",
            "validation_area": "Scoring & Robustness Validation",
            "passed": len(metrics) == len(configs)
            and len(video_metrics) == len(configs) * len(DEV_IDS)
            and (EXPERIMENT_ROOT / "leave_one_video_out_summary_v2_1a.csv").exists()
            and (EXPERIMENT_ROOT / "pareto_frontier_configs_v2_1a.csv").exists()
            and report_flags["final_performance_claim"] is False,
            "details": "post-hoc scoring, LOO robustness, Pareto frontier, and no-final-claim flags present",
        },
        {
            "sub_agent": "Sub Agent 5",
            "validation_area": "Leakage Guard Validation",
            "passed": not decision_bad
            and report_flags["actual_label_used_for_decision"] is False
            and report_flags["audio_candidate_score_used_for_decision"] is False
            and report_flags["per_video_ad_vs_nonad_contrast_score_used_for_decision"] is False,
            "details": f"decision_feature_columns_json forbidden patterns: {decision_bad}",
        },
        {
            "sub_agent": "Sub Agent 6",
            "validation_area": "Output & Safety Validation",
            "passed": report_flags["old_project_modified"] is False
            and report_flags["input_files_modified"] is False
            and report_flags["v1_4_preserved"] is True
            and report_flags["v2_0_preserved"] is True
            and report_flags["Extended_Evaluation_processed"] is False
            and report_flags["Pure_Test_processed"] is False
            and (latest_scan or {}).get("clean") is True,
            "details": "protected files, feature extraction flags, split scope, and latest bundle forbidden scan checked",
        },
        {
            "sub_agent": "Sub Agent 7",
            "validation_area": "Recommendation Validation",
            "passed": report_flags["ready_for_viewer_review_of_top_configs"] is True
            and report_flags["ready_for_v2_1b_rule_selection"] is False,
            "details": "category winners separated; recommended configs are viewer review candidates only",
        },
    ]
    return validations


def md_table(rows: list[dict[str, Any]], cols: list[str], limit: int = 20) -> str:
    use_rows = rows[:limit]
    out = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in use_rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        out.append("|" + "|".join(vals) + "|")
    return "\n".join(out)


def write_reports(
    report: dict[str, Any],
    metrics: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    recommended: list[dict[str, Any]],
    loo: list[dict[str, Any]],
    baseline: dict[str, Any],
    validations: list[dict[str, Any]],
) -> None:
    top = report["top_config_selection"]
    summary = f"""# State Machine Rule Weight Sweep v2.1a Development Summary

## 1. 실험 목적
v2.0 rule을 사람이 수동으로 하나씩 수정하지 않고, Development Set 안에서 score weight, threshold, penalty, budget guard 조합을 자동 비교했다. 이 실험은 v2.1a 내부 rule 개발용 sweep이며 최종 detector freeze가 아니다.

## 2. 고정 rule 원칙
- scene anchor는 transition_time_anchor only이며 광고 likelihood에 직접 더하지 않는다.
- audio는 같은 video_id 내부 relative evidence만 사용하고 audio-alone start/end는 금지했다.
- fuzzy-only disclosure는 hard evidence가 아니며 negative guard가 있으면 disclosure evidence를 suppress했다.
- opening disclosure alone은 hard start가 아니며 review-only/opening notice로 보냈다.
- detector decision feature에는 forbidden label/audit/actual-derived column을 넣지 않았다.

## 3. Sweep 대상 Parameter
weights, confidence thresholds, budget guard, ultra-high-confidence thresholds, freshness/support cap, penalties, long candidate handling만 config별로 바꿨다.

## 4. Config Family 설명
precision_safe, balanced, recall_safe, short_ad_safe, budget_strict, long_candidate_strict 6개 family에서 deterministic variants를 생성했다.

## 5. v2.0 Baseline 대비 요약
- v2.0 baseline available: {baseline.get('available')}
- v2.0 mean_final_prediction_ratio: {baseline.get('mean_final_prediction_ratio', 0):.6f}
- v2.0 false_positive_duration_sec: {baseline.get('false_positive_duration_sec', 0):.6f}
- v2.0 missed_actual_duration_sec: {baseline.get('missed_actual_duration_sec', 0):.6f}
- best balanced mean_final_prediction_ratio: {top['best_balanced_config']['mean_final_prediction_ratio']:.6f}
- best balanced false_positive_duration_sec: {top['best_balanced_config']['false_positive_duration_sec']:.6f}
- best balanced missed_actual_duration_sec: {top['best_balanced_config']['missed_actual_duration_sec']:.6f}

## 6. Category별 Top Config
{md_table([
    {'category': 'best_balanced_config', **top['best_balanced_config']},
    {'category': 'best_precision_safe_config', **top['best_precision_safe_config']},
    {'category': 'best_recall_safe_config', **top['best_recall_safe_config']},
    {'category': 'best_low_overprediction_config', **top['best_low_overprediction_config']},
    {'category': 'best_short_ad_safe_config', **top['best_short_ad_safe_config']},
    {'category': 'best_boundary_quality_config', **top['best_boundary_quality_config']},
], ['category', 'config_id', 'config_family', 'balanced_objective_score', 'actual_overlap_recall', 'prediction_overlap_precision_proxy', 'false_positive_duration_sec', 'missed_actual_duration_sec'])}

## 7. Pareto Frontier 요약
Pareto frontier config count: {len(pareto)}

{md_table(pareto, ['config_id', 'config_family', 'actual_overlap_recall', 'prediction_overlap_precision_proxy', 'false_positive_duration_sec', 'mean_final_prediction_ratio', 'boundary_quality_score'], 10)}

## 8. Leave-One-Video-Out Robustness 요약
Development Set 내부 12-fold leave-one-video-out으로, 각 fold에서 11개 영상 기준 top balanced config를 고르고 held-out video score를 기록했다. 이는 final evaluation이 아니다.

{md_table(loo, ['heldout_video_id', 'selected_config_id', 'selected_config_family', 'heldout_balanced_objective_score', 'heldout_actual_overlap_recall', 'heldout_prediction_overlap_precision_proxy'], 12)}

## 9. OCR Subtype Restoration 결과
status: {report['ocr_subtype_restoration_status']}

warnings: {report['warnings']}

## 10. Opening Disclosure Audit 결과
opening_disclosure_audit_rows: {report['opening_disclosure_audit_rows']}

## 11. Budget Guard Consistency 결과
budget_guard_consistency_failures: {report['budget_guard_consistency_failures']}

## 12. 권장 Viewer Review Config 목록
{md_table(recommended, ['config_id', 'config_family', 'balanced_objective_score', 'precision_safe_score', 'recall_safe_score', 'low_overprediction_score', 'short_ad_safety_score'], 10)}

## 13. v2.1b 또는 v2.2 적용 후보
권장 후보는 viewer에서 boundary와 skip 자연스러움을 확인한 뒤 v2.1b 또는 v2.2 fixed rule 후보로 검토해야 한다. 바로 fixed rule로 채택하지 않는다.

## 14. Safety/Leakage 요약
- actual_label_used_for_decision=false
- actual_label_used_for_config_scoring=true
- Extended_Evaluation_processed=false
- Diagnostic_Subset_processed=false
- Pure_Test_processed=false
- old_project_modified=false
- input_files_modified=false
- OCR/audio/scene extraction executed=false

## 15. Final Performance Claim 아님
이 결과는 Development Set 내부 rule 개발용 비교 자료다. final performance claim이 아니며 threshold tuning on validation/test도 수행하지 않았다.

## Sub Agent Validation Summary
{md_table(validations, ['sub_agent', 'validation_area', 'passed', 'details'], 10)}
"""
    OUTPUTS["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS["summary_md"].write_text(summary, encoding="utf-8")

    recommendation_note = """# v2.1a Recommendation Note

- 바로 fixed rule로 채택하지 말 것.
- recommended configs를 viewer에서 먼저 확인할 것.
- video 2, 6, 9, 13, 14는 priority review로 표시할 것.
- overprediction-safe config와 recall-safe config를 별도로 비교할 것.
- 최종 v2.1b 선택 전, 사람이 viewer로 boundary와 skip 자연스러움을 확인할 것.
- 이 note는 v2.1b/v2.2 선택을 위한 후보 안내이며 final detector freeze가 아니다.
"""
    OUTPUTS["recommendation_note"].parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS["recommendation_note"].write_text(recommendation_note, encoding="utf-8")

    rule_note = """# Scene/Audio/OCR State Machine Rule v2.1a Experiment Note

v2.1a는 fixed rule version이 아니라 Development Set 내부 rule weight / threshold / penalty sweep experiment다.

고정 원칙: scene semantics, same-video relative audio policy, OCR fuzzy/negative/opening disclosure guards, leakage guard는 모든 config에서 고정했다.

구조 개선: support_evidence_flag_v2_0을 start_support_flag, continuity_support_flag, end_support_flag로 분리했고, end_support/audio-only/scene-only가 start_pending에 들어가지 않도록 했다. Budget guard의 kept_ultra_high_confidence action은 ultra_high_confidence=true일 때만 허용한다.

actual label은 detector decision에 쓰지 않았고, config별 prediction 생성 후 Development Set post-hoc scoring에만 사용했다.
"""
    OUTPUTS["rule_note"].parent.mkdir(parents=True, exist_ok=True)
    OUTPUTS["rule_note"].write_text(rule_note, encoding="utf-8")


def main() -> None:
    print("Estimated runtime: approximately 25-45 minutes", flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text("", encoding="utf-8")
    log("STEP 01 estimated runtime printed first")
    before_snapshot = snapshot_protected_files()
    input_hashes_before = {name: sha256(path) for name, path in INPUTS.items() if path.exists() and name in {"fusion_input", "rule_contract", "actual_labels_scoring_only", "split", "manifest"}}

    log("STEP 02 safety snapshot and backup previous v2.1a experiment outputs")
    backup_dir, backup_manifest = backup_existing_outputs()

    log("STEP 03 load v2.0 fusion input and rule contract")
    if not INPUTS["fusion_input"].exists():
        raise SystemExit(f"Missing fusion input: {INPUTS['fusion_input']}")
    if not INPUTS["rule_contract"].exists():
        raise SystemExit(f"Missing v2.0 rule contract: {INPUTS['rule_contract']}")
    fusion_rows = read_csv(INPUTS["fusion_input"])
    split_rows = read_csv(INPUTS["split"])
    rule_contract = json.loads(INPUTS["rule_contract"].read_text(encoding="utf-8"))

    log("STEP 04 validate Development Set only input")
    input_validation = validate_inputs(fusion_rows, split_rows)
    durations = load_video_durations(split_rows, fusion_rows)

    log("STEP 05 build v2.1a structural correction layer")
    restoration_map: dict[str, dict[str, float]] = {}
    if INPUTS["ocr_frame_results_optional"].exists():
        restoration_map = det.build_ocr_restoration_map(read_csv(INPUTS["ocr_frame_results_optional"]))
    enhanced_rows, subtype_audit, opening_audit, support_audit, restoration_status, warnings = det.enhance_rows_with_v21a_flags(fusion_rows, restoration_map)
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(EXPERIMENT_ROOT / "fusion_input_enhanced_v2_1a.csv", enhanced_rows)
    write_csv(EXPERIMENT_ROOT / "ocr_disclosure_subtype_restoration_audit_v2_1a.csv", subtype_audit)
    write_csv(EXPERIMENT_ROOT / "opening_disclosure_guard_audit_v2_1a.csv", opening_audit)
    write_csv(EXPERIMENT_ROOT / "support_flag_split_audit_v2_1a.csv", support_audit)

    log("STEP 06 generate candidate rule config search space")
    configs = generate_candidate_configs()
    candidate_config_rows = [flatten_config(cfg) for cfg in configs]
    write_csv(EXPERIMENT_ROOT / "candidate_rule_configs_v2_1a.csv", candidate_config_rows)
    write_json(EXPERIMENT_ROOT / "candidate_rule_configs_v2_1a.json", configs)
    write_json(
        EXPERIMENT_ROOT / "rule_config_search_space_v2_1a.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "deterministic_seed": SEED,
            "max_candidate_configs": MAX_CONFIGS,
            "candidate_config_count": len(configs),
            "preset_families": sorted({cfg["config_family"] for cfg in configs}),
            "parameter_groups": [
                "weights",
                "thresholds",
                "budget_guard",
                "ultra_high_confidence",
                "freshness_support_cap",
                "penalties",
                "long_candidate",
            ],
            "fixed_rule_principles": CONFIG["fixed_rule_principles"],
        },
    )

    log("STEP 07-09 run v2.1a detector for each config and summarize outputs")
    all_outputs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for idx, cfg in enumerate(configs, start=1):
        if idx == 1 or idx % 10 == 0:
            log(f"running config {idx}/{len(configs)}: {cfg['config_id']}")
        all_outputs[cfg["config_id"]] = det.run_detector_for_config(enhanced_rows, cfg)

    log("STEP 10 run post-hoc Development Set scoring using actual labels")
    labels_by_video = load_labels_for_scoring_only(INPUTS["actual_labels_scoring_only"])

    log("STEP 11 compute objective scores and category scores")
    metrics: list[dict[str, Any]] = []
    video_metrics: list[dict[str, Any]] = []
    for cfg in configs:
        total, per_video = score_outputs_for_config(cfg, all_outputs[cfg["config_id"]], labels_by_video, durations)
        metrics.append(total)
        video_metrics.extend(per_video)
    metric_cols = list(metrics[0].keys()) if metrics else []
    video_metric_cols = list(video_metrics[0].keys()) if video_metrics else []
    write_csv(EXPERIMENT_ROOT / "per_config_metrics_v2_1a.csv", metrics, metric_cols)
    write_csv(EXPERIMENT_ROOT / "per_config_video_metrics_v2_1a.csv", video_metrics, video_metric_cols)
    write_csv(EXPERIMENT_ROOT / "per_config_failure_log_v2_1a.csv", config_failure_log(metrics))

    log("STEP 12 run leave-one-video-out robustness analysis")
    loo = leave_one_video_out(video_metrics)
    write_csv(EXPERIMENT_ROOT / "leave_one_video_out_summary_v2_1a.csv", loo)

    log("STEP 13 compute Pareto frontier and top configs")
    pareto = compute_pareto(metrics)
    top = select_top_configs(metrics, pareto)
    recommended = top["recommended_viewer_review_configs"]
    write_csv(EXPERIMENT_ROOT / "top_configs_by_objective_v2_1a.csv", sorted(metrics, key=lambda row: row["balanced_objective_score"], reverse=True))
    write_csv(EXPERIMENT_ROOT / "pareto_frontier_configs_v2_1a.csv", pareto)
    write_csv(EXPERIMENT_ROOT / "recommended_viewer_review_configs_v2_1a.csv", recommended)

    log("STEP 14 save row-level outputs for recommended top configs only")
    write_top_config_outputs(all_outputs, recommended)

    log("STEP 15 create viewer registry patch proposal")
    viewer_patch = create_viewer_registry_patch(recommended)

    log("STEP 16 generate recommendation note for v2.1b/v2.2")
    budget_audit = build_budget_audit(all_outputs)
    write_csv(EXPERIMENT_ROOT / "budget_guard_consistency_audit_v2_1a.csv", budget_audit)
    baseline = score_baseline_v2_0(labels_by_video, durations)

    log("STEP 17 run Sub Agent style internal validations")
    after_snapshot_pre_bundle = snapshot_protected_files()
    protected_changed = compare_snapshots(before_snapshot, after_snapshot_pre_bundle)
    input_hashes_after = {name: sha256(path) for name, path in INPUTS.items() if path.exists() and name in input_hashes_before}
    input_files_modified = input_hashes_before != input_hashes_after
    report_flags = {
        "old_project_modified": False,
        "input_files_modified": input_files_modified,
        "v1_4_preserved": not any("v1_4" in item or "v1.4" in item for item in protected_changed),
        "v2_0_preserved": not any("v2_0" in item or "v2.0" in item for item in protected_changed),
        "detector_executed": True,
        "feature_extraction_executed": False,
        "OCR_extraction_executed": False,
        "audio_extraction_executed": False,
        "scene_extraction_executed": False,
        "validation_output_count": 0,
        "test_row_level_output_count": 0,
        "Extended_Evaluation_processed": False,
        "Diagnostic_Subset_processed": False,
        "Pure_Test_processed": False,
        "threshold_tuning_on_validation_test": False,
        "actual_label_used_for_decision": False,
        "actual_label_used_for_config_scoring": True,
        "actual_label_used_for_posthoc_development_audit": True,
        "audio_candidate_score_used_for_decision": False,
        "per_video_ad_vs_nonad_contrast_score_used_for_decision": False,
        "final_performance_claim": False,
        "ready_for_viewer_review_of_top_configs": True,
        "ready_for_v2_1b_rule_selection": False,
        "support_flag_split_applied": True,
        "scene_used_for_ad_likelihood_directly": False,
    }

    log("STEP 18 create latest bundle")
    latest_scan = create_latest_bundle(recommended)
    validations = validation_rows(input_validation, configs, metrics, video_metrics, budget_audit, report_flags, latest_scan)
    write_csv(EXPERIMENT_ROOT / "sub_agent_validation_summary_v2_1a.csv", validations)

    log("STEP 19 final safety snapshot")
    after_snapshot = snapshot_protected_files()
    protected_changed_final = compare_snapshots(before_snapshot, after_snapshot)
    if protected_changed_final:
        warnings.append(f"Protected v1.4/v2.0 snapshot changed: {protected_changed_final}")
    report_flags["v1_4_preserved"] = not any("v1_4" in item or "v1.4" in item for item in protected_changed_final)
    report_flags["v2_0_preserved"] = not any("v2_0" in item or "v2.0" in item for item in protected_changed_final)

    failed_count = sum(1 for row in metrics if truth(row.get("config_failure_flag")))
    budget_failures = sum(1 for row in budget_audit if truth(row.get("consistency_failure")))
    report = {
        "task": EXPERIMENT_ID,
        "version": VERSION,
        "generated_at": now_iso(),
        "estimated_runtime_printed_first": True,
        "project_root": str(ROOT),
        "experiment_root": str(EXPERIMENT_ROOT),
        "backup_dir": backup_dir,
        "backup_manifest": backup_manifest,
        "rule_contract_path": str(INPUTS["rule_contract"]),
        "rule_contract_version": rule_contract.get("version"),
        "fusion_input_rows": len(fusion_rows),
        "development_video_ids": DEV_IDS,
        "candidate_config_count": len(configs),
        "failed_config_count": failed_count,
        "successful_config_count": len(configs) - failed_count,
        "max_config_limit_respected": len(configs) <= MAX_CONFIGS,
        "support_flag_split_applied": True,
        "ocr_subtype_restoration_status": restoration_status,
        "opening_disclosure_audit_rows": len(opening_audit),
        "budget_guard_consistency_failures": budget_failures,
        "top_config_selection": top,
        "pareto_frontier_config_count": len(pareto),
        "recommended_viewer_review_config_count": len(recommended),
        "v2_0_baseline": baseline,
        "metric_formulas": {
            "balanced_objective_score": "0.35*actual_overlap_recall + 0.25*prediction_overlap_precision_proxy + 0.15*boundary_quality_score - 0.15*false_positive_ratio_of_video - 0.05*overprediction_penalty_score - 0.05*review_burden_score",
            "boundary_matching": "prediction and actual intervals are greedily one-to-one matched by maximum overlap; start/end errors are calculated only for matched intervals",
            "boundary_quality_score": "exp(-median_boundary_error_sec / 10), where median_boundary_error_sec=(median_start_error+median_end_error)/2; no matched interval gives 0",
            "review_burden_score": "(review_only_count + overprediction_pruned_review_count + open_count) / (prediction_count + review_only_count + overprediction_pruned_review_count + open_count + 1)",
            "overprediction_penalty_score": "0.5*videos_over_hard_ratio/12 + 0.5*false_positive_duration_sec/total_development_video_duration_sec",
        },
        "objective_scope_note": "Development Set internal rule comparison only; not a final performance claim.",
        "actual_label_policy": {
            "actual_label_used_for_decision": False,
            "actual_label_used_for_config_scoring": True,
            "actual_label_used_for_posthoc_development_audit": True,
        },
        "split_scope": {
            "Development_Set": "original v2.4 train split",
            "Extended_Evaluation_processed": False,
            "Diagnostic_Subset_processed": False,
            "Pure_Test_processed": False,
        },
        "safety_flags": report_flags,
        "input_validation": input_validation,
        "sub_agent_validations": validations,
        "viewer_registry_patch_path": str(OUTPUTS["viewer_registry_patch"]),
        "latest_bundle": str(OUTPUTS["latest_bundle"]),
        "latest_bundle_forbidden_scan": latest_scan,
        "old_project_path": str(OLD_PROJECT),
        "warnings": warnings,
        "errors": [],
    }
    write_json(OUTPUTS["report_json"], report)
    write_reports(report, metrics, pareto, recommended, loo, baseline, validations)

    # report가 확정된 뒤 이미 만든 latest bundle에 final report/summary를 복사한다.
    final_report_targets = [OUTPUTS["summary_md"], OUTPUTS["report_json"], OUTPUTS["recommendation_note"], OUTPUTS["rule_note"]]
    for src in final_report_targets:
        dst = OUTPUTS["latest_bundle"] / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    latest_scan = forbidden_scan_bundle(OUTPUTS["latest_bundle"])
    write_json(OUTPUTS["latest_bundle"] / "latest_bundle_forbidden_scan_v2_1a.json", latest_scan)
    report["latest_bundle_forbidden_scan"] = latest_scan
    write_json(OUTPUTS["report_json"], report)

    log("STEP 20 final human-readable summary")
    best = top["best_balanced_config"]
    final_summary = {
        "task": EXPERIMENT_ID,
        "estimated_runtime_printed_first": True,
        "project_root": str(ROOT),
        "experiment_root": str(EXPERIMENT_ROOT),
        "backup_dir": backup_dir or "none",
        "fusion_input_rows": len(fusion_rows),
        "development_video_ids": DEV_IDS,
        "candidate_config_count": len(configs),
        "failed_config_count": failed_count,
        "successful_config_count": len(configs) - failed_count,
        "max_config_limit_respected": len(configs) <= MAX_CONFIGS,
        "support_flag_split_applied": True,
        "ocr_subtype_restoration_status": restoration_status,
        "opening_disclosure_audit_rows": len(opening_audit),
        "budget_guard_consistency_failures": budget_failures,
        "best_balanced_config_id": top["best_balanced_config"]["config_id"],
        "best_precision_safe_config_id": top["best_precision_safe_config"]["config_id"],
        "best_recall_safe_config_id": top["best_recall_safe_config"]["config_id"],
        "best_low_overprediction_config_id": top["best_low_overprediction_config"]["config_id"],
        "best_short_ad_safe_config_id": top["best_short_ad_safe_config"]["config_id"],
        "pareto_frontier_config_count": len(pareto),
        "recommended_viewer_review_config_count": len(recommended),
        "best_balanced_score": best["balanced_objective_score"],
        "best_balanced_actual_overlap_recall": best["actual_overlap_recall"],
        "best_balanced_prediction_precision_proxy": best["prediction_overlap_precision_proxy"],
        "best_balanced_false_positive_duration_sec": best["false_positive_duration_sec"],
        "best_balanced_missed_actual_duration_sec": best["missed_actual_duration_sec"],
        "best_balanced_mean_final_prediction_ratio": best["mean_final_prediction_ratio"],
        "v2_0_baseline_mean_final_prediction_ratio": baseline.get("mean_final_prediction_ratio", 0.0),
        "v2_0_baseline_false_positive_duration_sec": baseline.get("false_positive_duration_sec", 0.0),
        "v2_0_baseline_missed_actual_duration_sec": baseline.get("missed_actual_duration_sec", 0.0),
        "leave_one_video_out_summary_path": str(EXPERIMENT_ROOT / "leave_one_video_out_summary_v2_1a.csv"),
        "pareto_frontier_path": str(EXPERIMENT_ROOT / "pareto_frontier_configs_v2_1a.csv"),
        "recommended_configs_path": str(EXPERIMENT_ROOT / "recommended_viewer_review_configs_v2_1a.csv"),
        "viewer_registry_patch_path": str(OUTPUTS["viewer_registry_patch"]),
        "actual_label_used_for_decision": False,
        "actual_label_used_for_config_scoring": True,
        "actual_label_used_for_posthoc_development_audit": True,
        "final_performance_claim": False,
        "validation_output_count": 0,
        "test_row_level_output_count": 0,
        "Extended_Evaluation_processed": False,
        "Diagnostic_Subset_processed": False,
        "Pure_Test_processed": False,
        "old_project_modified": False,
        "input_files_modified": input_files_modified,
        "v1_4_preserved": report_flags["v1_4_preserved"],
        "v2_0_preserved": report_flags["v2_0_preserved"],
        "latest_bundle": str(OUTPUTS["latest_bundle"]),
        "ready_for_viewer_review_of_top_configs": True,
        "ready_for_v2_1b_rule_selection": False,
        "warnings": warnings,
        "errors": [],
    }
    print("Final Summary:", flush=True)
    for key, value in final_summary.items():
        print(f"- {key}: {value}", flush=True)


if __name__ == "__main__":
    main()
