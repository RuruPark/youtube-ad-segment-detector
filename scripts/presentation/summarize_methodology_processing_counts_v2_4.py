#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize v2.4 methodology processing counts for presentation/report use.

This script reads existing CSV/JSON/MD artifacts only. It does not implement or run
an ad detector, re-extract features, tune thresholds, or expose test row-level
features in the latest bundle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
VERSION = "v2_4"
TASK_NAME = "methodology_processing_counts_v2_4"

SCRIPT_PATH = PROJECT_ROOT / "scripts/presentation/summarize_methodology_processing_counts_v2_4.py"
CSV_OUT = PROJECT_ROOT / "data/presentation/methodology_processing_counts_v2_4.csv"
SUMMARY_MD = PROJECT_ROOT / "reports/presentation/methodology_processing_counts_v2_4_summary.md"
REPORT_JSON = PROJECT_ROOT / "reports/presentation/methodology_processing_counts_v2_4_report.json"
SLIDE_MD = PROJECT_ROOT / "reports/presentation/methodology_processing_counts_v2_4_slide_text_ko.md"
RUN_LOG = PROJECT_ROOT / "logs/methodology_processing_counts_v2_4_run_log.txt"
SUB_AGENT_RESULTS = PROJECT_ROOT / "reports/presentation/methodology_processing_counts_v2_4_sub_agent_results.json"
BUNDLE_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_methodology_processing_counts_v2_4"
OLD_SNAPSHOT_BEFORE = PROJECT_ROOT / "reports/rules/old_project_snapshot_before_methodology_counts_v2_4.tsv"
OLD_SNAPSHOT_AFTER = PROJECT_ROOT / "reports/rules/old_project_snapshot_after_methodology_counts_v2_4.tsv"

FIXED_SPLIT = {
    "train": [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15],
    "validation": [3, 7, 18],
    "test": [4, 16, 17],
}
SPLIT_SEED = 20240524

SOURCE_FILES = {
    "split": "data/splits/video_split_v2_4.csv",
    "visual_anchor": "data/features/visual_scene_boundary_anchors_v2_4.csv",
    "visual_anchor_with_split": "data/features/visual_scene_boundary_anchors_v2_4_with_split.csv",
    "visual_alignment_report": "reports/fusion/visual_anchor_alignment_pack_v2_4_report.json",
    "visual_alignment_summary": "reports/fusion/visual_anchor_alignment_pack_v2_4_summary.md",
    "audio_features_full": "data/audio/audio_visual_anchor_persistence_features_v2_4.csv",
    "audio_features_train_val": "data/audio/audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv",
    "audio_subwindow_train_val": "data/audio/audio_visual_anchor_persistence_subwindow_features_v2_4_train_val_for_discussion.csv",
    "audio_config": "configs/audio_persistence_rule_config_v2_4_train_only.json",
    "ocr_context_full": "data/ocr/ocr_visual_anchor_context_features_v2_4.csv",
    "ocr_context_train_val": "data/ocr/ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv",
    "ocr_sampling_plan": "data/ocr/ocr_visual_anchor_frame_sampling_plan_v2_4.csv",
    "ocr_frame_results": "data/ocr/ocr_visual_anchor_frame_results_v2_4.csv",
    "ocr_frame_results_train_val": "data/ocr/ocr_visual_anchor_frame_results_v2_4_train_val_for_discussion.csv",
    "ocr_thresholds": "data/ocr/ocr_visual_anchor_level_thresholds_v2_4_train_only.csv",
    "ocr_report": "reports/ocr/ocr_visual_anchor_context_features_v2_4_report.json",
    "ocr_summary": "reports/ocr/ocr_visual_anchor_context_features_v2_4_summary.md",
    "semantic_cleanup_report": "reports/fusion/scene_audio_ocr_semantic_cleanup_v2_4_report.json",
    "semantic_cleanup_summary": "reports/fusion/scene_audio_ocr_semantic_cleanup_v2_4_summary.md",
}

OUTPUT_FILES = [
    SCRIPT_PATH,
    CSV_OUT,
    SUMMARY_MD,
    REPORT_JSON,
    SLIDE_MD,
    RUN_LOG,
]
BUNDLE_COPY_FILES = [CSV_OUT, SUMMARY_MD, REPORT_JSON, SLIDE_MD, RUN_LOG, SCRIPT_PATH]
FORBIDDEN_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a",
    ".jpg", ".jpeg", ".png", ".webp", ".pt", ".pth", ".ckpt", ".onnx",
    ".pkl", ".pickle", ".parquet",
}
FORBIDDEN_DIR_PARTS = {"cache", "tmp", "__pycache__", "frames", "frame_images", "raw_video", "proxy", "checkpoint", "checkpoints", "model", "models"}

CSV_COLUMNS = [
    "category",
    "metric_name",
    "value",
    "unit",
    "split_scope",
    "source_file",
    "source_field_or_calculation",
    "verified_status",
    "note",
    "slide_sentence_candidate",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def log(message: str) -> None:
    print(message, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{now_iso()} {message}\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {"exists": True, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256(path)}


def snapshot_old_project(path: Path) -> list[str]:
    rows = []
    if not OLD_PROJECT_ROOT.exists():
        path.write_text("MISSING_OLD_PROJECT\n", encoding="utf-8")
        return rows
    for p in OLD_PROJECT_ROOT.rglob("*"):
        if p.is_file():
            st = p.stat()
            rows.append(f"{p.relative_to(OLD_PROJECT_ROOT)}\t{st.st_size}\t{st.st_mtime_ns}")
    rows = sorted(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return rows


def backup_existing_outputs() -> tuple[Path, list[str]]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = PROJECT_ROOT / "backups" / f"methodology_processing_counts_v2_4_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed: list[str] = []
    for path in OUTPUT_FILES:
        if path.exists():
            dst = backup_dir / rel(path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            backed.append(rel(path))
    if BUNDLE_DIR.exists() and any(BUNDLE_DIR.iterdir()):
        dst = backup_dir / rel(BUNDLE_DIR)
        shutil.copytree(BUNDLE_DIR, dst)
        backed.append(rel(BUNDLE_DIR))
    return backup_dir, backed


def add_metric(rows: list[dict[str, Any]], category: str, metric_name: str, value: Any, unit: str, split_scope: str, source_file: str, calculation: str, status: str = "verified", note: str = "", slide_sentence: str = "") -> None:
    rows.append({
        "category": category,
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        "split_scope": split_scope,
        "source_file": source_file,
        "source_field_or_calculation": calculation,
        "verified_status": status,
        "note": note,
        "slide_sentence_candidate": slide_sentence,
    })


def discrepancy(discrepancies: list[dict[str, Any]], metric_name: str, expected_or_reported_value: Any, recalculated_value: Any, selected_value: Any, reason: str, note: str) -> None:
    if str(expected_or_reported_value) != str(recalculated_value):
        discrepancies.append({
            "metric_name": metric_name,
            "expected_or_reported_value": expected_or_reported_value,
            "recalculated_value": recalculated_value,
            "selected_value": selected_value,
            "selected_reason": reason,
            "source_conflict_note": note,
        })


def split_validate(split_rows: list[dict[str, str]]) -> dict[str, Any]:
    by_split: dict[str, list[int]] = {}
    seeds = sorted(set(r.get("split_seed", "") for r in split_rows))
    for r in split_rows:
        by_split.setdefault(r.get("split", ""), []).append(as_int(r.get("video_id"), -1))
    by_split = {k: sorted(v) for k, v in by_split.items()}
    ok = by_split == FIXED_SPLIT and seeds == [str(SPLIT_SEED)]
    return {"ok": ok, "observed": by_split, "seed_values": seeds}


def load_sub_agent_results() -> list[dict[str, Any]]:
    if not SUB_AGENT_RESULTS.exists():
        return []
    try:
        data = read_json(SUB_AGENT_RESULTS)
    except Exception:
        return []
    if isinstance(data, dict) and isinstance(data.get("sub_agent_validation_results"), list):
        return data["sub_agent_validation_results"]
    if isinstance(data, list):
        return data
    return []


def scan_forbidden_bundle_files() -> list[str]:
    found = []
    if not BUNDLE_DIR.exists():
        return found
    for path in BUNDLE_DIR.rglob("*"):
        if not path.is_file():
            continue
        rp = path.relative_to(BUNDLE_DIR)
        parts = {part.lower() for part in rp.parts}
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or parts & FORBIDDEN_DIR_PARTS:
            found.append(str(rp))
    return sorted(found)


def write_csv_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    start_time = now_iso()
    RUN_LOG.write_text("", encoding="utf-8")
    warnings: list[str] = []
    errors: list[str] = []
    discrepancies: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}

    log("[STEP 01] Safety snapshot and backup")
    if PROJECT_ROOT != Path.cwd():
        warnings.append(f"Script cwd is {Path.cwd()}, project root is {PROJECT_ROOT}")
    old_before = snapshot_old_project(OLD_SNAPSHOT_BEFORE)
    backup_dir, backed_files = backup_existing_outputs()
    input_stats_before: dict[str, Any] = {}
    for relpath in SOURCE_FILES.values():
        input_stats_before[relpath] = file_stat(PROJECT_ROOT / relpath)
    (PROJECT_ROOT / "reports/presentation/methodology_processing_counts_v2_4_input_file_stats_before.json").write_text(json.dumps(input_stats_before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("[STEP 02] Locate source artifacts")
    source_found: dict[str, str] = {}
    source_missing: dict[str, str] = {}
    for key, relpath in SOURCE_FILES.items():
        path = PROJECT_ROOT / relpath
        if path.exists():
            source_found[key] = str(path)
        else:
            source_missing[key] = str(path)
            warnings.append(f"Missing source artifact: {relpath}")
    if source_missing:
        warnings.append("Some optional source artifacts were missing; available canonical CSV/JSON sources were used.")

    split_rows = read_csv(PROJECT_ROOT / SOURCE_FILES["split"])
    split_check = split_validate(split_rows)
    if not split_check["ok"]:
        errors.append("Fixed split file does not match expected v2.4 split.")

    visual_report = read_json(PROJECT_ROOT / SOURCE_FILES["visual_alignment_report"])
    audio_config = read_json(PROJECT_ROOT / SOURCE_FILES["audio_config"])
    ocr_report = read_json(PROJECT_ROOT / SOURCE_FILES["ocr_report"])
    semantic_report = read_json(PROJECT_ROOT / SOURCE_FILES["semantic_cleanup_report"])

    log("[STEP 03] Compute visual/scene transition counts")
    visual_rows = read_csv(PROJECT_ROOT / SOURCE_FILES["visual_anchor"])
    visual_split_rows = read_csv(PROJECT_ROOT / SOURCE_FILES["visual_anchor_with_split"])
    visual_anchor_count = len(visual_rows)
    opencv_count = sum(as_int(r.get("opencv_candidate_count_in_anchor")) for r in visual_rows)
    resnet_count = sum(as_int(r.get("resnet_candidate_count_in_anchor")) for r in visual_rows)
    opencv_bool_count = sum(1 for r in visual_rows if boolish(r.get("has_opencv_ffmpeg_candidate")))
    resnet_bool_count = sum(1 for r in visual_rows if boolish(r.get("has_resnet_candidate")))
    visual_split_counts = Counter(r.get("split", "") for r in visual_split_rows)
    source_relation_counts = Counter(r.get("source_relation", "") for r in visual_rows)
    cross_source_near_5s_counts = Counter(str(r.get("cross_source_near_5s", "")) for r in visual_rows)
    selected.update({
        "opencv_count": opencv_count,
        "resnet_count": resnet_count,
        "visual_anchor_count": visual_anchor_count,
        "visual_train_count": visual_split_counts.get("train", 0),
        "visual_validation_count": visual_split_counts.get("validation", 0),
        "visual_test_count": visual_split_counts.get("test", 0),
        "canonical_anchor_actual_path": visual_report.get("canonical_anchor_actual_path_used", str(PROJECT_ROOT / SOURCE_FILES["visual_anchor"])),
        "fallback_scene_candidate_used": visual_report.get("fallback_scene_candidate_used_for_current_audio", False),
        "source_relation_counts": dict(source_relation_counts),
        "cross_source_near_5s_counts": dict(cross_source_near_5s_counts),
    })
    discrepancy(discrepancies, "visual_anchor_count_report_vs_csv", visual_report.get("visual_anchor_count"), visual_anchor_count, visual_anchor_count, "canonical CSV row count matches or overrides report", "visual anchor count from canonical CSV")
    discrepancy(discrepancies, "example_opencv_candidate_count", 707, opencv_count, opencv_count, "CSV sum of opencv_candidate_count_in_anchor", "User example checked against canonical CSV")
    discrepancy(discrepancies, "example_resnet_candidate_count", 1093, resnet_count, resnet_count, "CSV sum of resnet_candidate_count_in_anchor", "User example checked against canonical CSV")
    discrepancy(discrepancies, "example_visual_anchor_count", 1329, visual_anchor_count, visual_anchor_count, "canonical CSV row count", "User example checked against canonical CSV")
    visual_sentence = f"화면 전환 단계에서는 v2.4 기준 OpenCV/FFmpeg {opencv_count:,}개와 ResNet {resnet_count:,}개 후보를 통합해 총 {visual_anchor_count:,}개의 visual transition anchor를 생성했다."
    add_metric(metrics, "visual", "opencv_ffmpeg_candidate_count", opencv_count, "candidate", "all", SOURCE_FILES["visual_anchor"], "sum(opencv_candidate_count_in_anchor)", note=f"has_opencv_ffmpeg_candidate=True anchor count={opencv_bool_count}; duplicate candidates inside an anchor are counted by the *_count column.", slide_sentence=visual_sentence)
    add_metric(metrics, "visual", "resnet_candidate_count", resnet_count, "candidate", "all", SOURCE_FILES["visual_anchor"], "sum(resnet_candidate_count_in_anchor)", note=f"has_resnet_candidate=True anchor count={resnet_bool_count}; duplicate candidates inside an anchor are counted by the *_count column.", slide_sentence=visual_sentence)
    add_metric(metrics, "visual", "visual_anchor_count", visual_anchor_count, "anchor", "all", SOURCE_FILES["visual_anchor"], "row_count", slide_sentence=visual_sentence)
    for split in ["train", "validation", "test"]:
        add_metric(metrics, "visual", f"visual_anchor_{split}_count", visual_split_counts.get(split, 0), "anchor", split, SOURCE_FILES["visual_anchor_with_split"], "groupby(split).size")
    add_metric(metrics, "visual", "fallback_scene_candidate_used_for_current_alignment", str(bool(selected["fallback_scene_candidate_used"])).lower(), "boolean", "all", SOURCE_FILES["visual_alignment_report"], "fallback_scene_candidate_used_for_current_audio", note="False means canonical data/features visual anchor was used, not fallback scene candidate.")

    log("[STEP 04] Compute audio processing counts")
    audio_rows = read_csv(PROJECT_ROOT / SOURCE_FILES["audio_features_full"])
    audio_tv_rows = read_csv(PROJECT_ROOT / SOURCE_FILES["audio_features_train_val"])
    audio_sub_rows = read_csv(PROJECT_ROOT / SOURCE_FILES["audio_subwindow_train_val"])
    audio_anchor_count = len(audio_rows)
    audio_split_counts = Counter(r.get("split", "") for r in audio_rows)
    audio_tv_split_counts = Counter(r.get("split", "") for r in audio_tv_rows)
    audio_available_count = sum(1 for r in audio_rows if boolish(r.get("audio_available")))
    audio_probe_error_count = sum(1 for r in audio_rows if str(r.get("probe_error", "")).strip())
    audio_pre_sub_count = sum(as_int(r.get("audio_pre_10s_subwindow_count")) for r in audio_rows)
    audio_post_sub_count = sum(as_int(r.get("audio_post_10s_subwindow_count")) for r in audio_rows)
    audio_subwindow_total_from_context = audio_pre_sub_count + audio_post_sub_count
    audio_success_count = visual_report.get("audio_feature_success_count", audio_subwindow_total_from_context)
    audio_failed_count = visual_report.get("audio_feature_failed_count", audio_probe_error_count)
    audio_sub_feature_status = Counter(r.get("feature_status", "") for r in audio_sub_rows)
    audio_sub_sampling_status = Counter(r.get("sampling_status", "") for r in audio_sub_rows)
    discrepancy(discrepancies, "audio_feature_success_count_report_vs_csv_subwindow_sum", visual_report.get("audio_feature_success_count"), audio_subwindow_total_from_context, audio_success_count, "visual alignment report success count equals full audio context subwindow sum", "train/validation subwindow CSV has 10,984 rows because test row-level subwindows are not included there")
    audio_sentence = f"오디오 단계에서는 각 visual anchor 전후 10초를 2초 subwindow 단위로 분석해 총 {int(audio_success_count):,}개 audio subwindow feature를 성공적으로 처리했으며 실패는 {int(audio_failed_count):,}개였다."
    selected.update({
        "audio_anchor_count": audio_anchor_count,
        "audio_feature_success_count": int(audio_success_count),
        "audio_feature_failed_count": int(audio_failed_count),
        "audio_train_count": audio_split_counts.get("train", 0),
        "audio_validation_count": audio_split_counts.get("validation", 0),
        "audio_test_count": audio_split_counts.get("test", 0),
        "audio_train_val_discussion_count": len(audio_tv_rows),
        "audio_train_val_subwindow_rows": len(audio_sub_rows),
        "audio_ad_like_threshold": visual_report.get("audio_ad_like_threshold", audio_config.get("train_only_thresholds", {}).get("audio_ad_like_threshold")),
        "persistence_window_sec": visual_report.get("persistence_window_sec", audio_config.get("persistence_window_sec")),
        "subwindow_size_sec": visual_report.get("subwindow_size_sec", audio_config.get("subwindow_size_sec")),
        "subwindow_stride_sec": visual_report.get("subwindow_stride_sec", audio_config.get("subwindow_stride_sec")),
        "train_used_for_audio_level_thresholds": visual_report.get("train_used_for_audio_level_thresholds"),
        "validation_used_for_audio_level_thresholds": visual_report.get("validation_used_for_audio_level_thresholds"),
        "test_used_for_audio_level_thresholds": visual_report.get("test_used_for_audio_level_thresholds"),
        "audio_sub_feature_status_counts_train_val": dict(audio_sub_feature_status),
        "audio_sub_sampling_status_counts_train_val": dict(audio_sub_sampling_status),
    })
    add_metric(metrics, "audio", "audio_anchor_context_feature_count", audio_anchor_count, "anchor_context", "all", SOURCE_FILES["audio_features_full"], "row_count", note=f"audio_available True={audio_available_count}; probe_error non-empty={audio_probe_error_count}", slide_sentence=audio_sentence)
    add_metric(metrics, "audio", "audio_feature_success_count", int(audio_success_count), "subwindow_feature", "all", SOURCE_FILES["visual_alignment_report"], "audio_feature_success_count; cross-check=sum(pre/post subwindow_count) in audio feature CSV", slide_sentence=audio_sentence)
    add_metric(metrics, "audio", "audio_feature_failed_count", int(audio_failed_count), "subwindow_feature", "all", SOURCE_FILES["visual_alignment_report"], "audio_feature_failed_count; cross-check=probe_error count", slide_sentence=audio_sentence)
    add_metric(metrics, "audio", "audio_train_val_discussion_anchor_count", len(audio_tv_rows), "anchor_context", "train_validation", SOURCE_FILES["audio_features_train_val"], "row_count")
    add_metric(metrics, "audio", "audio_train_val_subwindow_feature_rows", len(audio_sub_rows), "subwindow_feature", "train_validation", SOURCE_FILES["audio_subwindow_train_val"], "row_count", note=f"feature_status counts={dict(audio_sub_feature_status)}")
    for split in ["train", "validation", "test"]:
        add_metric(metrics, "audio", f"audio_anchor_{split}_count", audio_split_counts.get(split, 0), "anchor_context", split, SOURCE_FILES["audio_features_full"], "groupby(split).size")
    add_metric(metrics, "audio", "pre_post_context_window_sec", selected["persistence_window_sec"], "sec", "all", SOURCE_FILES["visual_alignment_report"], "persistence_window_sec")
    add_metric(metrics, "audio", "subwindow_size_sec", selected["subwindow_size_sec"], "sec", "all", SOURCE_FILES["visual_alignment_report"], "subwindow_size_sec")
    add_metric(metrics, "audio", "subwindow_stride_sec", selected["subwindow_stride_sec"], "sec", "all", SOURCE_FILES["visual_alignment_report"], "subwindow_stride_sec")
    add_metric(metrics, "audio", "audio_ad_like_threshold", selected["audio_ad_like_threshold"], "score_threshold", "train_only", SOURCE_FILES["visual_alignment_report"], "audio_ad_like_threshold")

    log("[STEP 05] Compute OCR processing counts")
    ocr_plan_rows = read_csv(PROJECT_ROOT / SOURCE_FILES["ocr_sampling_plan"])
    ocr_frame_rows = read_csv(PROJECT_ROOT / SOURCE_FILES["ocr_frame_results"])
    ocr_frame_tv_rows = read_csv(PROJECT_ROOT / SOURCE_FILES["ocr_frame_results_train_val"])
    ocr_context_rows = read_csv(PROJECT_ROOT / SOURCE_FILES["ocr_context_full"])
    ocr_context_tv_rows = read_csv(PROJECT_ROOT / SOURCE_FILES["ocr_context_train_val"])
    ocr_frame_status = Counter(r.get("ocr_status", "") for r in ocr_frame_rows)
    ocr_frame_split_counts = Counter(r.get("split", "") for r in ocr_frame_rows)
    ocr_context_split_counts = Counter(r.get("split", "") for r in ocr_context_rows)
    ocr_context_tv_split_counts = Counter(r.get("split", "") for r in ocr_context_tv_rows)
    ocr_plan_split_counts = Counter(r.get("split", "") for r in ocr_plan_rows)
    ocr_unique_frame_count = len(set(r.get("frame_sample_id", "") for r in ocr_frame_rows))
    ocr_success = ocr_frame_status.get("success", 0)
    ocr_empty = ocr_frame_status.get("empty", 0)
    ocr_failed = ocr_frame_status.get("failed", 0)
    discrepancy(discrepancies, "ocr_frame_success_count_report_vs_csv", ocr_report.get("ocr_frame_success_count"), ocr_success, ocr_success, "OCR frame results CSV status count", "Machine-readable report and CSV should match")
    discrepancy(discrepancies, "ocr_frame_empty_count_report_vs_csv", ocr_report.get("ocr_frame_empty_count"), ocr_empty, ocr_empty, "OCR frame results CSV status count", "Machine-readable report and CSV should match")
    discrepancy(discrepancies, "ocr_frame_failed_count_report_vs_csv", ocr_report.get("ocr_frame_failed_count"), ocr_failed, ocr_failed, "OCR frame results CSV status count", "Machine-readable report and CSV should match")
    discrepancy(discrepancies, "ocr_context_count_report_vs_csv", ocr_report.get("ocr_context_feature_count"), len(ocr_context_rows), len(ocr_context_rows), "OCR context CSV row count", "Machine-readable report and CSV should match")
    ocr_sentence = f"OCR 단계에서는 visual anchor 기준 pre/post 10초 frame을 샘플링해 총 {len(ocr_frame_rows):,}개 frame OCR을 처리했고, success/empty/failed는 {ocr_success:,}/{ocr_empty:,}/{ocr_failed:,}개였으며 최종 OCR context feature {len(ocr_context_rows):,}개를 생성했다."
    selected.update({
        "ocr_visual_anchor_count": ocr_report.get("visual_anchor_count", len(ocr_context_rows)),
        "ocr_frame_sampling_row_count": len(ocr_plan_rows),
        "ocr_frame_sample_count": len(ocr_frame_rows),
        "ocr_unique_frame_count": ocr_unique_frame_count,
        "ocr_frame_success_count": ocr_success,
        "ocr_frame_empty_count": ocr_empty,
        "ocr_frame_failed_count": ocr_failed,
        "ocr_context_feature_count": len(ocr_context_rows),
        "ocr_train_context_count": ocr_context_split_counts.get("train", 0),
        "ocr_validation_context_count": ocr_context_split_counts.get("validation", 0),
        "ocr_test_context_count": ocr_context_split_counts.get("test", 0),
        "ocr_context_train_val_bundle_count": len(ocr_context_tv_rows),
        "joined_row_count": ocr_report.get("joined_row_count"),
        "ocr_missing_row_count": ocr_report.get("ocr_missing_row_count"),
        "ocr_reliability_level_counts": semantic_report.get("ocr_reliability_level_counts", {}),
        "ocr_low_score_reason_counts": semantic_report.get("ocr_low_score_reason_counts", {}),
        "train_used_for_ocr_level_thresholds": ocr_report.get("train_used_for_ocr_level_thresholds"),
        "validation_used_for_ocr_level_thresholds": ocr_report.get("validation_used_for_ocr_level_thresholds"),
        "test_used_for_ocr_level_thresholds": ocr_report.get("test_used_for_ocr_level_thresholds"),
        "label_aligned_ocr_used_as_reference_only": ocr_report.get("label_aligned_ocr_used_as_reference_only"),
        "ocr_plan_split_counts": dict(ocr_plan_split_counts),
        "ocr_frame_split_counts": dict(ocr_frame_split_counts),
        "ocr_context_split_counts": dict(ocr_context_split_counts),
        "ocr_context_train_val_split_counts": dict(ocr_context_tv_split_counts),
    })
    add_metric(metrics, "ocr", "ocr_frame_sampling_row_count", len(ocr_plan_rows), "frame_sample_plan_row", "all", SOURCE_FILES["ocr_sampling_plan"], "row_count", slide_sentence=ocr_sentence)
    add_metric(metrics, "ocr", "ocr_frame_sample_count", len(ocr_frame_rows), "ocr_frame_result", "all", SOURCE_FILES["ocr_frame_results"], "row_count", slide_sentence=ocr_sentence)
    add_metric(metrics, "ocr", "ocr_unique_frame_count", ocr_unique_frame_count, "unique_frame_sample_id", "all", SOURCE_FILES["ocr_frame_results"], "nunique(frame_sample_id)")
    add_metric(metrics, "ocr", "ocr_frame_success_count", ocr_success, "ocr_frame_result", "all", SOURCE_FILES["ocr_frame_results"], "count(ocr_status == success)", slide_sentence=ocr_sentence)
    add_metric(metrics, "ocr", "ocr_frame_empty_count", ocr_empty, "ocr_frame_result", "all", SOURCE_FILES["ocr_frame_results"], "count(ocr_status == empty)", slide_sentence=ocr_sentence)
    add_metric(metrics, "ocr", "ocr_frame_failed_count", ocr_failed, "ocr_frame_result", "all", SOURCE_FILES["ocr_frame_results"], "count(ocr_status == failed)", slide_sentence=ocr_sentence)
    add_metric(metrics, "ocr", "ocr_context_feature_count", len(ocr_context_rows), "anchor_context", "all", SOURCE_FILES["ocr_context_full"], "row_count", slide_sentence=ocr_sentence)
    for split in ["train", "validation", "test"]:
        add_metric(metrics, "ocr", f"ocr_context_{split}_count", ocr_context_split_counts.get(split, 0), "anchor_context", split, SOURCE_FILES["ocr_context_full"], "groupby(split).size")
    add_metric(metrics, "ocr", "ocr_context_train_val_bundle_count", len(ocr_context_tv_rows), "anchor_context", "train_validation", SOURCE_FILES["ocr_context_train_val"], "row_count")
    add_metric(metrics, "ocr", "joined_row_count", ocr_report.get("joined_row_count"), "joined_anchor_row", "train_validation", SOURCE_FILES["ocr_report"], "joined_row_count")
    add_metric(metrics, "ocr", "ocr_missing_row_count", ocr_report.get("ocr_missing_row_count"), "joined_anchor_row", "train_validation", SOURCE_FILES["ocr_report"], "ocr_missing_row_count")
    for level, count in selected["ocr_reliability_level_counts"].items():
        add_metric(metrics, "ocr", f"ocr_reliability_{level}_count", count, "anchor_context", "train_validation", SOURCE_FILES["semantic_cleanup_report"], f"ocr_reliability_level_counts.{level}")
    for reason, count in selected["ocr_low_score_reason_counts"].items():
        add_metric(metrics, "ocr", f"ocr_low_score_reason_{reason}_count", count, "anchor_context", "train_validation", SOURCE_FILES["semantic_cleanup_report"], f"ocr_low_score_reason_counts.{reason}")

    log("[STEP 06] Build presentation-ready text")
    fusion_sentence = f"결합 단계에서는 train/validation discussion table {selected['ocr_context_train_val_bundle_count']:,}개 anchor에 scene/audio/OCR evidence를 정렬해 rule 기반 검토에 활용했다."
    slide_text = f"""# 발표자료용 방법론 처리 수치 요약

- 화면 전환: v2.4 기준 OpenCV/FFmpeg {opencv_count:,}개와 ResNet {resnet_count:,}개 후보를 통합해 visual transition anchor {visual_anchor_count:,}개를 생성
- 오디오: 각 anchor 전후 10초를 2초 subwindow 단위로 분석해 audio subwindow feature {int(audio_success_count):,}개 처리, 실패 {int(audio_failed_count):,}개
- OCR: visual anchor 기준 OCR frame {len(ocr_frame_rows):,}개 처리, success/empty/failed {ocr_success:,}/{ocr_empty:,}/{ocr_failed:,}개, OCR context feature {len(ocr_context_rows):,}개 생성
- 결합: train/validation discussion table {selected['ocr_context_train_val_bundle_count']:,}개 anchor에 scene/audio/OCR evidence를 정렬하여 rule 기반 detector 검토에 활용
- 주의: visual scene anchor는 광고 시작/종료의 직접 evidence가 아니라 상태 변화 후보 시각인 transition_time_anchor
"""
    SLIDE_MD.parent.mkdir(parents=True, exist_ok=True)
    SLIDE_MD.write_text(slide_text, encoding="utf-8")

    log("[STEP 07] Generate CSV/summary/report")
    add_metric(metrics, "fusion", "train_validation_discussion_anchor_count", selected["ocr_context_train_val_bundle_count"], "anchor", "train_validation", SOURCE_FILES["semantic_cleanup_report"], "train_row_count + validation_row_count / joined_row_count", slide_sentence=fusion_sentence)
    write_csv_rows(CSV_OUT, metrics, CSV_COLUMNS)

    input_stats_after = {relpath: file_stat(PROJECT_ROOT / relpath) for relpath in SOURCE_FILES.values()}
    input_files_modified = [relpath for relpath, before in input_stats_before.items() if before != input_stats_after.get(relpath)]
    if input_files_modified:
        errors.append("One or more input files changed during processing count summary.")
    old_after = snapshot_old_project(OLD_SNAPSHOT_AFTER)
    old_project_modified = old_before != old_after
    if old_project_modified:
        errors.append("Old project snapshot before/after differs.")

    sub_agent_results = load_sub_agent_results()
    report = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "start_time": start_time,
        "end_time": now_iso(),
        "source_files_found": source_found,
        "source_files_missing": source_missing,
        "backup_dir": str(backup_dir),
        "backed_up_files": backed_files,
        "fixed_split_check": split_check,
        "selected_numbers": selected,
        "slide_sentences": {
            "visual": visual_sentence,
            "audio": audio_sentence,
            "ocr": ocr_sentence,
            "fusion": fusion_sentence,
        },
        "discrepancies": discrepancies,
        "discrepancy_count": len(discrepancies),
        "metric_csv": str(CSV_OUT),
        "summary_md": str(SUMMARY_MD),
        "slide_text_ko_md": str(SLIDE_MD),
        "no_detector_implementation": True,
        "no_detector_run": True,
        "no_feature_extraction": True,
        "no_threshold_tuning": True,
        "performance_claim": False,
        "test_row_level_features_copied_to_bundle": False,
        "old_project_snapshot_before": str(OLD_SNAPSHOT_BEFORE),
        "old_project_snapshot_after": str(OLD_SNAPSHOT_AFTER),
        "old_project_modified": old_project_modified,
        "input_file_stats_before": input_stats_before,
        "input_file_stats_after": input_stats_after,
        "input_files_modified": bool(input_files_modified),
        "input_files_modified_paths": input_files_modified,
        "sub_agent_validation_results": sub_agent_results,
        "latest_for_chatgpt_forbidden_files_found": [],
        "warnings": warnings,
        "errors": errors,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    discrepancy_lines = ["- None"] if not discrepancies else [f"- `{d['metric_name']}`: reported/example `{d['expected_or_reported_value']}` vs recalculated `{d['recalculated_value']}`; selected `{d['selected_value']}` ({d['selected_reason']})" for d in discrepancies]
    warning_lines = ["- None"] if not warnings else [f"- {w}" for w in warnings]
    error_lines = ["- None"] if not errors else [f"- {e}" for e in errors]
    sub_lines = ["- Pending or not recorded"] if not sub_agent_results else [f"- {r.get('name')}: `{r.get('status')}`" for r in sub_agent_results]
    summary = f"""# Methodology Processing Counts v2.4 Summary

Generated at: `{report['end_time']}`

## Scope

This is a presentation/report processing-count summary. It reads existing v2.4 artifacts only. It is not detector implementation, feature extraction, threshold tuning, rule modification, or a performance report.

## Presentation Sentences

- {visual_sentence}
- {audio_sentence}
- {ocr_sentence}
- {fusion_sentence}
- visual scene anchor는 광고 시작/종료의 직접 evidence가 아니라 상태 변화 후보 시각인 `transition_time_anchor`이다.

## A. 화면 전환 / Visual Anchor

- OpenCV/FFmpeg candidate count: `{opencv_count:,}` from `sum(opencv_candidate_count_in_anchor)`.
- ResNet candidate count: `{resnet_count:,}` from `sum(resnet_candidate_count_in_anchor)`.
- Final visual transition anchor count: `{visual_anchor_count:,}` from canonical CSV row count.
- Split anchor counts: train `{selected['visual_train_count']:,}`, validation `{selected['visual_validation_count']:,}`, test `{selected['visual_test_count']:,}`.
- Canonical anchor actual path: `{selected['canonical_anchor_actual_path']}`.
- Fallback scene candidate used for current alignment: `{selected['fallback_scene_candidate_used']}`.
- Integration notes: `source_relation` counts `{dict(source_relation_counts)}`; `cross_source_near_5s` counts `{dict(cross_source_near_5s_counts)}`.

## B. 오디오

- Audio anchor context feature rows: `{audio_anchor_count:,}`.
- Audio subwindow feature success/fail: `{int(audio_success_count):,}` / `{int(audio_failed_count):,}`.
- Train/validation discussion anchor count: `{len(audio_tv_rows):,}`.
- Train/validation subwindow feature rows available for discussion: `{len(audio_sub_rows):,}`.
- Context window: pre/post `{selected['persistence_window_sec']}` sec, subwindow size `{selected['subwindow_size_sec']}` sec, stride `{selected['subwindow_stride_sec']}` sec.
- `audio_ad_like_threshold`: `{selected['audio_ad_like_threshold']}`.
- Train-only audio level threshold use: train `{selected['train_used_for_audio_level_thresholds']}`, validation `{selected['validation_used_for_audio_level_thresholds']}`, test `{selected['test_used_for_audio_level_thresholds']}`.

## C. OCR

- OCR frame sampling plan rows: `{len(ocr_plan_rows):,}`.
- OCR frame result count: `{len(ocr_frame_rows):,}`; unique frame sample count `{ocr_unique_frame_count:,}`.
- OCR success/empty/failed: `{ocr_success:,}` / `{ocr_empty:,}` / `{ocr_failed:,}`.
- OCR context feature rows: `{len(ocr_context_rows):,}`.
- OCR context split counts: train `{selected['ocr_train_context_count']:,}`, validation `{selected['ocr_validation_context_count']:,}`, test `{selected['ocr_test_context_count']:,}`.
- Train/validation OCR context bundle count: `{len(ocr_context_tv_rows):,}`.
- Joined row count: `{selected['joined_row_count']}`, OCR missing row count: `{selected['ocr_missing_row_count']}`.
- OCR reliability counts on train/validation discussion table: `{selected['ocr_reliability_level_counts']}`.
- OCR low score reason counts: `{selected['ocr_low_score_reason_counts']}`.
- Train-only OCR threshold use: train `{selected['train_used_for_ocr_level_thresholds']}`, validation `{selected['validation_used_for_ocr_level_thresholds']}`, test `{selected['test_used_for_ocr_level_thresholds']}`.
- Label-aligned OCR used as reference only: `{selected['label_aligned_ocr_used_as_reference_only']}`.

## Discrepancies

{chr(10).join(discrepancy_lines)}

## Sub Agent Validation Results

{chr(10).join(sub_lines)}

## Safety

- old_project_modified: `{str(old_project_modified).lower()}`
- input_files_modified: `{str(bool(input_files_modified)).lower()}`
- test row-level features copied to latest bundle: `false`
- performance claim: `false`

## Warnings

{chr(10).join(warning_lines)}

## Errors

{chr(10).join(error_lines)}
"""
    SUMMARY_MD.write_text(summary, encoding="utf-8")

    log("[STEP 08] Sub Agent validations")
    if sub_agent_results:
        log("[STEP 08] External Sub Agent validation results loaded into report/summary")
    else:
        log("[STEP 08] External Sub Agent validation results not found yet")

    log("[STEP 09] Update latest bundle")
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for child in list(BUNDLE_DIR.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for path in BUNDLE_COPY_FILES:
        shutil.copy2(path, BUNDLE_DIR / path.name)
    readme = BUNDLE_DIR / "README_latest_files.md"
    readme.write_text(f"""# Latest Methodology Processing Counts v2.4

작업명: `{TASK_NAME}`

생성 시각: `{now_iso()}`

This is a presentation processing counts summary, not detector implementation or performance report. It contains only small generated outputs and does not include input feature source tables, media/frame/cache/model/raw video/proxy/checkpoint files, or test row-level feature CSVs.

## Files

| File | Role |
|---|---|
| `methodology_processing_counts_v2_4.csv` | metric table with source and calculation notes |
| `methodology_processing_counts_v2_4_summary.md` | human-readable count summary |
| `methodology_processing_counts_v2_4_report.json` | machine-readable report with selected values/discrepancies |
| `methodology_processing_counts_v2_4_slide_text_ko.md` | short Korean bullets for slides |
| `methodology_processing_counts_v2_4_run_log.txt` | run log |
| `summarize_methodology_processing_counts_v2_4.py` | reproducible summary script |

## Safety

- `no_detector_implementation=true`
- `no_detector_run=true`
- `no_feature_extraction=true`
- `no_threshold_tuning=true`
- `performance_claim=false`
- `test_row_level_features_copied_to_bundle=false`
- `old_project_modified={str(old_project_modified).lower()}`
- `input_files_modified={str(bool(input_files_modified)).lower()}`

## Next Step

발표자료 방법론 슬라이드에 `methodology_processing_counts_v2_4_slide_text_ko.md` 내용을 반영한다.
""", encoding="utf-8")

    forbidden = scan_forbidden_bundle_files()
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden
    report["latest_for_chatgpt_files"] = sorted(str(p.relative_to(BUNDLE_DIR)) for p in BUNDLE_DIR.rglob("*") if p.is_file())
    if forbidden:
        report["errors"].append("Forbidden files found in latest bundle.")
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(REPORT_JSON, BUNDLE_DIR / REPORT_JSON.name)
    shutil.copy2(RUN_LOG, BUNDLE_DIR / RUN_LOG.name)

    log("[STEP 10] Print final human-readable summary")
    status = "SUCCESS" if not report["errors"] else "FAILURE"
    print("\nMethodology processing counts summary")
    print(f"- status: {status}")
    print(f"- visual: {visual_sentence}")
    print(f"- audio: {audio_sentence}")
    print(f"- OCR: {ocr_sentence}")
    print(f"- discrepancy_count: {len(discrepancies)}")
    print(f"- warnings: {len(warnings)}")
    print(f"- errors: {report['errors'] if report['errors'] else 'none'}")
    print(f"- old_project_modified={str(old_project_modified).lower()}")
    print(f"- input_files_modified={str(bool(input_files_modified)).lower()}")
    print(f"- latest bundle: {BUNDLE_DIR}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
