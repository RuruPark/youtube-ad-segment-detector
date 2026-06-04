#!/usr/bin/env python3
"""Prepare Development Set final scene anchors and OCR input schedule.

This script combines existing OpenCV/FFmpeg, ResNet, and TransNetV2 conservative
scene-boundary candidates into a new OCR-ready anchor file. It does not execute
OCR, create frames, modify detector rules, or overwrite existing visual anchors.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
TASK_NAME = "final_scene_anchor_ocr_inputs_v2_5_development"
VERSION = "v2_5_development"
SPLIT_SEED = 20240524
DEVELOPMENT_VIDEO_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
DIAGNOSTIC_VIDEO_IDS = [3, 7, 18]
PURE_TEST_VIDEO_IDS = [4, 16, 17]
SPLIT_NOTE = (
    "Development Set = original v2.4 train split; Test Set = "
    "original validation + test for public-facing evaluation."
)
REQUIRED_SPLIT_PHRASE_KO = (
    "본 프로젝트는 학습 기반 모델이 아니라 rule-based detector이므로, 기존 train split은 "
    "모델 학습용이 아닌 Development Set으로 재정의해 rule 설계와 cue 분석, 오류 진단에 사용한다. "
    "공개용 설명에서는 기존 validation과 test를 Test Set으로 통합해 "
    "규칙 고정 이후 평가 대상으로 설명한다. 기존 validation/test 세부 구분은 "
    "내부 호환용 key로만 유지한다."
)

# 입력 경로.
SPLIT_PATH = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"
SPLIT_TERMINOLOGY_CONFIG = PROJECT_ROOT / "configs/splits/split_terminology_v2_5_ruledev_extended_eval.json"
VIDEO_MANIFEST_PATH = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
OPENCV_PATH = PROJECT_ROOT / "data/scene/scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed.csv"
RESNET_PATH = PROJECT_ROOT / "data/review/resnet_scene_candidate_human_review_v2_4_usefulness_analysis.csv"
CANONICAL_PATH = PROJECT_ROOT / "data/features/visual_scene_boundary_anchors_v2_4_with_split.csv"
CANONICAL_FALLBACK_PATH = PROJECT_ROOT / "data/features/visual_scene_boundary_anchors_v2_4.csv"
TRANSNET_SWEEP_PATH = PROJECT_ROOT / "data/scene/transnetv2_conservative_sweep_candidates_v2_4_train.csv"
TRANSNET_BEST_PATH = PROJECT_ROOT / "data/scene/transnetv2_conservative_best_candidate_v2_4_train.csv"
TRANSNET_REPORT_PATH = PROJECT_ROOT / "reports/scene/transnetv2_conservative_sweep_v2_4_report.json"
ABLATION_REPORT_PATH = PROJECT_ROOT / "reports/scene/scene_model_ablation_v2_4_report.json"
ABLATION_SUMMARY_PATH = PROJECT_ROOT / "reports/scene/scene_model_ablation_v2_4_summary.md"
ABLATION_DOC_PATH = PROJECT_ROOT / "reports/scene/scene_model_ablation_document_report_v2_4.md"

# 출력 경로.
SCRIPT_PATH = PROJECT_ROOT / "scripts/ocr/prepare_final_scene_anchor_ocr_inputs_v2_5_development.py"
FINAL_ANCHOR_PATH = PROJECT_ROOT / "data/scene/final_scene_boundary_anchor_v2_5_development.csv"
VIDEO_MAPPING_PATH = PROJECT_ROOT / "data/ocr/final_scene_anchor_video_path_mapping_v2_5_development.csv"
OCR_SCHEDULE_PATH = PROJECT_ROOT / "data/ocr/final_scene_anchor_ocr_schedule_v2_5_development.csv"
INPUT_CONTRACT_PATH = PROJECT_ROOT / "data/ocr/final_scene_anchor_ocr_input_contract_v2_5_development.json"
INTEGRATION_MANIFEST_PATH = PROJECT_ROOT / "data/ocr/final_scene_anchor_ocr_integration_manifest_v2_5_development.csv"
PREP_SUMMARY_PATH = PROJECT_ROOT / "data/ocr/final_scene_anchor_ocr_preparation_summary_v2_5_development.csv"
SUMMARY_MD_PATH = PROJECT_ROOT / "reports/ocr/final_scene_anchor_ocr_inputs_v2_5_development_summary.md"
REPORT_JSON_PATH = PROJECT_ROOT / "reports/ocr/final_scene_anchor_ocr_inputs_v2_5_development_report.json"
FINDINGS_MD_PATH = PROJECT_ROOT / "reports/ocr/final_scene_anchor_ocr_inputs_v2_5_development_findings.md"
NEXT_STEP_GUIDE_PATH = PROJECT_ROOT / "reports/ocr/final_scene_anchor_ocr_next_step_guide_v2_5_development.md"
RUN_LOG_PATH = PROJECT_ROOT / "logs/final_scene_anchor_ocr_inputs_v2_5_development_run_log.txt"
LATEST_BUNDLE_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_final_scene_anchor_ocr_inputs_v2_5_development"
SHARED_DIR = PROJECT_ROOT / "outputs/latest_ocr_inputs_development"
LATEST_SCENE_DIR = PROJECT_ROOT / "outputs/latest_scene"

# 정책 parameter.
ANCHOR_CLUSTER_WINDOW_SEC = 2.0
CROSS_SOURCE_DEDUP_WINDOW_SEC = 2.0
ANCHOR_CONTEXT_SEC = 10.0
ANCHOR_DENSE_STEP_SEC = 1.0
BACKGROUND_STEP_SEC = 1.5
DEDUP_TOLERANCE_SEC = 0.1
TRANSNET_CONSERVATIVE_FAMILY = "transnetv2_threshold_0_7_dedup_5"
TRANSNET_CONSERVATIVE_THRESHOLD = 0.7
TRANSNET_CONSERVATIVE_DEDUP_WINDOW_SEC = 5.0

FINAL_ANCHOR_COLUMNS = [
    "final_anchor_id", "video_id", "original_split_v2_4", "split_role_v2_5",
    "evaluation_subset_v2_5", "split_terminology_note", "anchor_sec", "anchor_frame",
    "cluster_min_sec", "cluster_max_sec", "cluster_member_count", "source_relation",
    "has_opencv_ffmpeg", "has_resnet", "has_transnetv2_conservative", "opencv_member_count",
    "resnet_member_count", "transnetv2_member_count", "source_members_json", "representative_rule",
    "confidence_or_score", "notes",
]
VIDEO_MAPPING_COLUMNS = [
    "video_id", "original_split_v2_4", "split_role_v2_5", "evaluation_subset_v2_5",
    "split_terminology_note", "video_title", "video_path", "file_exists", "duration_sec", "fps",
    "frame_count", "mapping_source", "notes",
]
OCR_SCHEDULE_COLUMNS = [
    "schedule_id", "video_id", "original_split_v2_4", "split_role_v2_5", "evaluation_subset_v2_5",
    "split_terminology_note", "ocr_time_sec", "ocr_frame_index", "schedule_source",
    "nearest_final_anchor_id", "nearest_final_anchor_sec", "distance_to_nearest_anchor_sec",
    "within_anchor_context", "anchor_context_sec", "anchor_dense_step_sec", "background_step_sec",
    "dedup_key", "video_path", "notes",
]
MANIFEST_COLUMNS = [
    "artifact_type", "artifact_path", "exists", "row_count", "purpose", "downstream_usage",
    "required_columns_json", "split_scope", "ready_for_ocr_extraction", "notes",
]
PREP_SUMMARY_COLUMNS = ["metric_name", "metric_value", "notes"]
FORBIDDEN_BUNDLE_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a",
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".pt", ".pth", ".ckpt",
    ".onnx", ".pkl", ".pickle", ".parquet",
}
FORBIDDEN_OUTPUT_NAMES = {
    "final_scene_anchor_ocr_raw_v2_5_development.csv",
    "final_scene_anchor_ocr_features_v2_5_development.csv",
}


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
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{now_iso()} {message}\n")


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path), "exists": True, "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "sha256": sha256(path),
    }


def snapshot_file_tree(root: Path) -> list[str]:
    rows: list[str] = []
    if not root.exists():
        return rows
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            rows.append(f"{path.relative_to(root)}\t{stat.st_size}\t{stat.st_mtime_ns}")
    return sorted(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_with_columns(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def backup_existing_outputs() -> tuple[Path, list[str]]:
    backup_dir = PROJECT_ROOT / "backups" / f"final_scene_anchor_ocr_inputs_v2_5_development_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    targets = [
        FINAL_ANCHOR_PATH, VIDEO_MAPPING_PATH, OCR_SCHEDULE_PATH, INPUT_CONTRACT_PATH,
        INTEGRATION_MANIFEST_PATH, PREP_SUMMARY_PATH, SUMMARY_MD_PATH, REPORT_JSON_PATH,
        FINDINGS_MD_PATH, NEXT_STEP_GUIDE_PATH, RUN_LOG_PATH,
    ]
    copied: list[str] = []
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in targets:
        if path.exists():
            dst = backup_dir / rel(path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            copied.append(rel(path))
    for directory in [LATEST_BUNDLE_DIR, SHARED_DIR]:
        if directory.exists() and any(directory.iterdir()):
            dst = backup_dir / rel(directory)
            shutil.copytree(directory, dst, dirs_exist_ok=True)
            copied.append(rel(directory))
    return backup_dir, copied


def load_split_rows() -> tuple[list[dict[str, str]], dict[int, dict[str, str]], dict[str, Any]]:
    rows = read_csv(SPLIT_PATH)
    by_split: dict[str, list[int]] = defaultdict(list)
    seed_values = set()
    for row in rows:
        vid = safe_int(row.get("video_id"))
        if vid is None:
            continue
        by_split[row.get("split", "")].append(vid)
        seed_values.add(str(row.get("split_seed", "")))
    observed = {key: sorted(value) for key, value in by_split.items()}
    expected = {
        "train": DEVELOPMENT_VIDEO_IDS,
        "validation": DIAGNOSTIC_VIDEO_IDS,
        "test": PURE_TEST_VIDEO_IDS,
    }
    split_check = {
        "valid": observed == expected and seed_values == {str(SPLIT_SEED)},
        "observed_split_video_ids": observed,
        "expected_split_video_ids": expected,
        "seed_values": sorted(seed_values),
        "row_count": len(rows),
    }
    dev_rows = [row for row in rows if safe_int(row.get("video_id")) in DEVELOPMENT_VIDEO_IDS and row.get("split") == "train"]
    return dev_rows, {int(row["video_id"]): row for row in rows if row.get("video_id")}, split_check


def load_manifest() -> dict[int, dict[str, str]]:
    if not VIDEO_MANIFEST_PATH.exists():
        return {}
    rows = read_csv(VIDEO_MANIFEST_PATH)
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        vid = safe_int(row.get("video_id"))
        if vid is not None:
            out[vid] = row
    return out


def build_video_mapping(dev_rows: list[dict[str, str]], manifest: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_row in sorted(dev_rows, key=lambda r: int(r["video_id"])):
        vid = int(split_row["video_id"])
        manifest_row = manifest.get(vid, {})
        video_path = manifest_row.get("video_path") or split_row.get("video_path", "")
        duration = safe_float(manifest_row.get("duration_sec"), safe_float(split_row.get("video_duration_sec")))
        fps = safe_float(manifest_row.get("fps"))
        frame_count = safe_int(manifest_row.get("frame_count"))
        if frame_count is None and duration is not None and fps is not None:
            frame_count = int(round(duration * fps))
        mapping_source = "video_manifest_v2_2" if manifest_row else "video_split_v2_4"
        rows.append({
            "video_id": vid,
            "original_split_v2_4": "train",
            "split_role_v2_5": "development",
            "evaluation_subset_v2_5": "none",
            "split_terminology_note": SPLIT_NOTE,
            "video_title": manifest_row.get("video_title") or split_row.get("video_name", ""),
            "video_path": video_path,
            "file_exists": str(Path(video_path).exists()).lower() if video_path else "false",
            "duration_sec": fmt_float(duration, 6),
            "fps": fmt_float(fps, 6),
            "frame_count": frame_count if frame_count is not None else "",
            "mapping_source": mapping_source,
            "notes": "Development Set only; no Test Set mapping generated.",
        })
    return rows


def candidate_frame(sec: float, fps: float | None) -> int | str:
    if fps is None:
        return ""
    return int(round(sec * fps))


def load_candidates(video_info: dict[int, dict[str, Any]], warnings: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_candidates: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    dev_ids = set(DEVELOPMENT_VIDEO_IDS)

    if OPENCV_PATH.exists():
        rows, columns = read_csv_with_columns(OPENCV_PATH)
        count = 0
        for idx, row in enumerate(rows):
            vid = safe_int(row.get("video_id"))
            sec = safe_float(row.get("candidate_time_sec"))
            if vid not in dev_ids or sec is None or sec < 0:
                continue
            fps = safe_float(video_info.get(vid, {}).get("fps"))
            all_candidates.append({
                "source": "opencv_ffmpeg", "video_id": vid, "candidate_sec": sec,
                "candidate_frame": candidate_frame(sec, fps), "score": safe_float(row.get("scene_change_score")),
                "source_model": row.get("candidate_source") or row.get("method_used") or "opencv_ffmpeg",
                "raw_row_index": idx, "source_path": str(OPENCV_PATH),
                "source_id": row.get("merged_candidate_id", ""),
            })
            count += 1
        stats["opencv_ffmpeg"] = {"path": str(OPENCV_PATH), "columns": columns, "development_candidate_count": count}
    else:
        warnings.append(f"OpenCV/FFmpeg candidate file missing: {OPENCV_PATH}")
        stats["opencv_ffmpeg"] = {"path": str(OPENCV_PATH), "development_candidate_count": 0, "missing": True}

    if RESNET_PATH.exists():
        rows, columns = read_csv_with_columns(RESNET_PATH)
        count = 0
        for idx, row in enumerate(rows):
            vid = safe_int(row.get("video_id"))
            sec = safe_float(row.get("candidate_time_sec"))
            if vid not in dev_ids or sec is None or sec < 0:
                continue
            fps = safe_float(video_info.get(vid, {}).get("fps"))
            all_candidates.append({
                "source": "resnet", "video_id": vid, "candidate_sec": sec,
                "candidate_frame": candidate_frame(sec, fps), "score": safe_float(row.get("scene_change_score"), safe_float(row.get("cosine_distance"))),
                "source_model": row.get("model_name") or row.get("candidate_source") or "resnet_embedding",
                "raw_row_index": idx, "source_path": str(RESNET_PATH),
                "source_id": row.get("score_rank_in_video", ""),
            })
            count += 1
        stats["resnet"] = {"path": str(RESNET_PATH), "columns": columns, "development_candidate_count": count}
    else:
        warnings.append(f"ResNet candidate file missing: {RESNET_PATH}")
        stats["resnet"] = {"path": str(RESNET_PATH), "development_candidate_count": 0, "missing": True}

    selected_family = TRANSNET_CONSERVATIVE_FAMILY
    if TRANSNET_BEST_PATH.exists():
        best_rows = read_csv(TRANSNET_BEST_PATH)
        if best_rows:
            selected_family = best_rows[0].get("selected_family", selected_family) or selected_family
            threshold = safe_float(best_rows[0].get("threshold"))
            dedup = safe_float(best_rows[0].get("dedup_window_sec"))
            if selected_family != TRANSNET_CONSERVATIVE_FAMILY or threshold != TRANSNET_CONSERVATIVE_THRESHOLD or dedup != TRANSNET_CONSERVATIVE_DEDUP_WINDOW_SEC:
                warnings.append("TransNetV2 best candidate file does not exactly match expected conservative criteria; strict sweep filter is still applied.")
    else:
        warnings.append(f"TransNetV2 best candidate file missing: {TRANSNET_BEST_PATH}")

    if TRANSNET_SWEEP_PATH.exists():
        rows, columns = read_csv_with_columns(TRANSNET_SWEEP_PATH)
        count = 0
        for idx, row in enumerate(rows):
            vid = safe_int(row.get("video_id"))
            sec = safe_float(row.get("candidate_sec"))
            threshold = safe_float(row.get("threshold"))
            dedup = safe_float(row.get("dedup_window_sec"))
            family = row.get("sweep_family", "")
            is_conservative = (
                family == TRANSNET_CONSERVATIVE_FAMILY
                and threshold == TRANSNET_CONSERVATIVE_THRESHOLD
                and dedup == TRANSNET_CONSERVATIVE_DEDUP_WINDOW_SEC
            )
            if vid not in dev_ids or sec is None or sec < 0 or not is_conservative:
                continue
            fps = safe_float(video_info.get(vid, {}).get("fps"), safe_float(row.get("fps")))
            all_candidates.append({
                "source": "transnetv2_conservative", "video_id": vid, "candidate_sec": sec,
                "candidate_frame": safe_int(row.get("candidate_frame"), safe_int(candidate_frame(sec, fps) if fps else None)),
                "score": safe_float(row.get("transnetv2_score")),
                "source_model": family, "raw_row_index": idx, "source_path": str(TRANSNET_SWEEP_PATH),
                "source_id": row.get("cluster_id", ""), "threshold": threshold, "dedup_window_sec": dedup,
            })
            count += 1
        stats["transnetv2_conservative"] = {
            "path": str(TRANSNET_SWEEP_PATH), "columns": columns, "development_candidate_count": count,
            "selected_family": selected_family, "threshold": TRANSNET_CONSERVATIVE_THRESHOLD,
            "dedup_window_sec": TRANSNET_CONSERVATIVE_DEDUP_WINDOW_SEC,
        }
    else:
        warnings.append(f"TransNetV2 conservative sweep candidate file missing: {TRANSNET_SWEEP_PATH}")
        stats["transnetv2_conservative"] = {"path": str(TRANSNET_SWEEP_PATH), "development_candidate_count": 0, "missing": True}

    return all_candidates, stats


def load_canonical_anchors() -> dict[int, list[dict[str, Any]]]:
    path = CANONICAL_PATH if CANONICAL_PATH.exists() else CANONICAL_FALLBACK_PATH
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return out
    rows = read_csv(path)
    for row in rows:
        vid = safe_int(row.get("video_id"))
        sec = safe_float(row.get("canonical_boundary_time_sec"))
        split = row.get("split", "train" if vid in DEVELOPMENT_VIDEO_IDS else "")
        if vid in DEVELOPMENT_VIDEO_IDS and sec is not None and sec >= 0 and split in {"", "train"}:
            out[vid].append({
                "video_id": vid, "canonical_sec": sec,
                "scene_boundary_anchor_id": row.get("scene_boundary_anchor_id", ""),
                "source_relation": row.get("source_relation", ""),
                "canonical_time_source": row.get("canonical_time_source", ""),
            })
    for values in out.values():
        values.sort(key=lambda item: item["canonical_sec"])
    return out


def cluster_candidates(candidates: list[dict[str, Any]], canonical: dict[int, list[dict[str, Any]]], video_info: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        by_video[int(cand["video_id"])].append(cand)

    final_rows: list[dict[str, Any]] = []
    for vid in DEVELOPMENT_VIDEO_IDS:
        video_candidates = sorted(by_video.get(vid, []), key=lambda item: (item["candidate_sec"], item["source"]))
        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_max: float | None = None
        for cand in video_candidates:
            sec = float(cand["candidate_sec"])
            if not current:
                current = [cand]
                current_max = sec
            elif current_max is not None and sec - current_max <= CROSS_SOURCE_DEDUP_WINDOW_SEC:
                current.append(cand)
                current_max = max(current_max, sec)
            else:
                clusters.append(current)
                current = [cand]
                current_max = sec
        if current:
            clusters.append(current)

        fps = safe_float(video_info.get(vid, {}).get("fps"))
        for idx, cluster in enumerate(clusters, start=1):
            secs = [float(member["candidate_sec"]) for member in cluster]
            cluster_min = min(secs)
            cluster_max = max(secs)
            sources = {str(member["source"]) for member in cluster}
            source_counts = Counter(str(member["source"]) for member in cluster)
            canonical_match = None
            for can in canonical.get(vid, []):
                can_sec = float(can["canonical_sec"])
                nearest_distance = min(abs(can_sec - sec) for sec in secs)
                if nearest_distance <= CROSS_SOURCE_DEDUP_WINDOW_SEC:
                    if canonical_match is None or nearest_distance < canonical_match["nearest_distance"]:
                        canonical_match = {**can, "nearest_distance": nearest_distance}

            representative_rule = ""
            anchor_sec: float
            confidence_or_score: float | None = None
            if canonical_match is not None:
                anchor_sec = float(canonical_match["canonical_sec"])
                representative_rule = "canonical_anchor_priority"
                numeric_scores = [member["score"] for member in cluster if member.get("score") is not None]
                confidence_or_score = max(numeric_scores) if numeric_scores else None
            else:
                scored_members = [member for member in cluster if member.get("score") is not None]
                if scored_members:
                    representative = sorted(scored_members, key=lambda item: (-float(item["score"]), float(item["candidate_sec"])))[0]
                    anchor_sec = float(representative["candidate_sec"])
                    confidence_or_score = float(representative["score"])
                    representative_rule = "max_source_score_then_earliest"
                else:
                    representative = sorted(cluster, key=lambda item: float(item["candidate_sec"]))[0]
                    anchor_sec = float(representative["candidate_sec"])
                    representative_rule = "earliest_timestamp_no_score_available"

            relation = "only_" + next(iter(sources)) if len(sources) == 1 else "multi_source:" + "+".join(sorted(sources))
            members_json = []
            for member in sorted(cluster, key=lambda item: (float(item["candidate_sec"]), item["source"])):
                members_json.append({
                    "source": member.get("source"),
                    "candidate_sec": member.get("candidate_sec"),
                    "candidate_frame": member.get("candidate_frame"),
                    "score": member.get("score"),
                    "source_model": member.get("source_model"),
                    "source_id": member.get("source_id"),
                    "raw_row_index": member.get("raw_row_index"),
                    "source_path": member.get("source_path"),
                })
            if canonical_match is not None:
                members_json.append({
                    "source": "canonical_opencv_resnet_anchor_for_representative_priority",
                    "candidate_sec": canonical_match.get("canonical_sec"),
                    "scene_boundary_anchor_id": canonical_match.get("scene_boundary_anchor_id"),
                    "source_relation": canonical_match.get("source_relation"),
                    "canonical_time_source": canonical_match.get("canonical_time_source"),
                })

            final_rows.append({
                "final_anchor_id": f"FSA25DEV_v{vid:02d}_{idx:05d}",
                "video_id": vid,
                "original_split_v2_4": "train",
                "split_role_v2_5": "development",
                "evaluation_subset_v2_5": "none",
                "split_terminology_note": SPLIT_NOTE,
                "anchor_sec": fmt_float(anchor_sec, 6),
                "anchor_frame": candidate_frame(anchor_sec, fps),
                "cluster_min_sec": fmt_float(cluster_min, 6),
                "cluster_max_sec": fmt_float(cluster_max, 6),
                "cluster_member_count": len(cluster),
                "source_relation": relation,
                "has_opencv_ffmpeg": str("opencv_ffmpeg" in sources).lower(),
                "has_resnet": str("resnet" in sources).lower(),
                "has_transnetv2_conservative": str("transnetv2_conservative" in sources).lower(),
                "opencv_member_count": source_counts.get("opencv_ffmpeg", 0),
                "resnet_member_count": source_counts.get("resnet", 0),
                "transnetv2_member_count": source_counts.get("transnetv2_conservative", 0),
                "source_members_json": json.dumps(members_json, ensure_ascii=False, sort_keys=True),
                "representative_rule": representative_rule,
                "confidence_or_score": fmt_float(confidence_or_score, 9),
                "notes": "OCR auxiliary anchor only; detector rules and existing visual anchors were not modified.",
            })
    return final_rows


def time_range(start: float, end: float, step: float) -> list[float]:
    times: list[float] = []
    if step <= 0:
        return times
    value = start
    # float rounding이 있어도 upper endpoint를 포함하도록 보정한다.
    while value <= end + 1e-9:
        times.append(round(value, 6))
        value += step
    return times


def nearest_anchor(time_sec: float, anchors: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float | None]:
    if not anchors:
        return None, None
    best = min(anchors, key=lambda row: abs(safe_float(row["anchor_sec"], 0.0) - time_sec))
    dist = abs(safe_float(best["anchor_sec"], 0.0) - time_sec)
    return best, dist


def build_ocr_schedule(final_anchors: list[dict[str, Any]], video_mapping: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in final_anchors:
        anchors_by_video[int(row["video_id"])].append(row)
    for rows in anchors_by_video.values():
        rows.sort(key=lambda row: safe_float(row["anchor_sec"], 0.0))

    schedule_rows: list[dict[str, Any]] = []
    by_dedup: dict[tuple[int, int], dict[str, Any]] = {}
    source_priority = {"anchor_dense": 0, "background_regular": 1}
    mapping_by_video = {int(row["video_id"]): row for row in video_mapping}

    def add_time(vid: int, time_sec: float, source: str) -> None:
        mapping = mapping_by_video[vid]
        duration = safe_float(mapping.get("duration_sec"))
        fps = safe_float(mapping.get("fps"))
        if duration is None or time_sec < 0 or time_sec > duration + 1e-6:
            return
        dedup_index = int(round(time_sec / DEDUP_TOLERANCE_SEC))
        key = (vid, dedup_index)
        anchors = anchors_by_video.get(vid, [])
        nearest, distance = nearest_anchor(time_sec, anchors)
        within = bool(distance is not None and distance <= ANCHOR_CONTEXT_SEC + 1e-9)
        if source == "background_regular" and within:
            return
        row = {
            "schedule_id": "",  # filled after sorting
            "video_id": vid,
            "original_split_v2_4": "train",
            "split_role_v2_5": "development",
            "evaluation_subset_v2_5": "none",
            "split_terminology_note": SPLIT_NOTE,
            "ocr_time_sec": fmt_float(time_sec, 6),
            "ocr_frame_index": candidate_frame(time_sec, fps),
            "schedule_source": source,
            "nearest_final_anchor_id": nearest.get("final_anchor_id", "") if nearest else "",
            "nearest_final_anchor_sec": nearest.get("anchor_sec", "") if nearest else "",
            "distance_to_nearest_anchor_sec": fmt_float(distance, 6),
            "within_anchor_context": str(within).lower(),
            "anchor_context_sec": fmt_float(ANCHOR_CONTEXT_SEC, 3),
            "anchor_dense_step_sec": fmt_float(ANCHOR_DENSE_STEP_SEC, 3),
            "background_step_sec": fmt_float(BACKGROUND_STEP_SEC, 3),
            "dedup_key": f"v{vid}_t{dedup_index}",
            "video_path": mapping.get("video_path", ""),
            "notes": "OCR schedule only; OCR engine not executed.",
        }
        existing = by_dedup.get(key)
        if existing is None:
            by_dedup[key] = row
            return
        existing_distance = safe_float(existing.get("distance_to_nearest_anchor_sec"), 10**9)
        new_distance = distance if distance is not None else 10**9
        should_replace = (
            source_priority[source] < source_priority[existing["schedule_source"]]
            or (source_priority[source] == source_priority[existing["schedule_source"]] and new_distance < (existing_distance or 10**9))
        )
        if should_replace:
            by_dedup[key] = row

    for mapping in video_mapping:
        vid = int(mapping["video_id"])
        if mapping.get("file_exists") != "true":
            continue
        duration = safe_float(mapping.get("duration_sec"))
        if duration is None or duration <= 0:
            continue
        for anchor in anchors_by_video.get(vid, []):
            anchor_sec = safe_float(anchor.get("anchor_sec"))
            if anchor_sec is None:
                continue
            start = max(0.0, anchor_sec - ANCHOR_CONTEXT_SEC)
            end = min(duration, anchor_sec + ANCHOR_CONTEXT_SEC)
            for time_sec in time_range(start, end, ANCHOR_DENSE_STEP_SEC):
                add_time(vid, time_sec, "anchor_dense")
        for time_sec in time_range(0.0, duration, BACKGROUND_STEP_SEC):
            add_time(vid, time_sec, "background_regular")

    sorted_rows = sorted(by_dedup.values(), key=lambda row: (int(row["video_id"]), safe_float(row["ocr_time_sec"], 0.0), row["schedule_source"]))
    for idx, row in enumerate(sorted_rows, start=1):
        row["schedule_id"] = f"OCRS25DEV_{idx:08d}"
    schedule_rows.extend(sorted_rows)
    return schedule_rows


def row_count(path: Path) -> int | str:
    if not path.exists():
        return ""
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return max(sum(1 for _ in f) - 1, 0)
    return ""


def build_contract(ready: bool, warnings: list[str], errors: list[str], next_script: str) -> dict[str, Any]:
    return {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "split_terminology_version": "v2.5_ruledev_extended_eval",
        "scope": "Development Set only (original v2.4 train split)",
        "development_set_video_ids": DEVELOPMENT_VIDEO_IDS,
        "final_anchor_path": str(FINAL_ANCHOR_PATH),
        "video_path_mapping_path": str(VIDEO_MAPPING_PATH),
        "ocr_schedule_path": str(OCR_SCHEDULE_PATH),
        "required_schedule_columns": OCR_SCHEDULE_COLUMNS,
        "schedule_time_column": "ocr_time_sec",
        "schedule_frame_column": "ocr_frame_index",
        "video_id_column": "video_id",
        "video_path_column": "video_path",
        "split_columns": ["original_split_v2_4", "split_role_v2_5", "evaluation_subset_v2_5", "split_terminology_note"],
        "schedule_policy": {
            "anchor_context_sec": ANCHOR_CONTEXT_SEC,
            "anchor_dense_step_sec": ANCHOR_DENSE_STEP_SEC,
            "background_step_sec": BACKGROUND_STEP_SEC,
            "dedup_tolerance_sec": DEDUP_TOLERANCE_SEC,
        },
        "final_anchor_sources": {
            "opencv_ffmpeg": str(OPENCV_PATH),
            "resnet": str(RESNET_PATH),
            "transnetv2_conservative": str(TRANSNET_SWEEP_PATH),
        },
        "transnetv2_conservative_threshold": TRANSNET_CONSERVATIVE_THRESHOLD,
        "transnetv2_conservative_dedup_window_sec": TRANSNET_CONSERVATIVE_DEDUP_WINDOW_SEC,
        "ocr_extraction_executed": False,
        "ready_for_next_ocr_extraction": ready,
        "next_recommended_script": next_script,
        "warnings": warnings,
        "errors": errors,
    }


def build_manifest(ready: bool) -> list[dict[str, Any]]:
    artifacts = [
        ("final_scene_anchor", FINAL_ANCHOR_PATH, "Final clustered scene boundary anchor for OCR scheduling.", "Use as scene-anchor reference, not as detector replacement.", FINAL_ANCHOR_COLUMNS),
        ("video_path_mapping", VIDEO_MAPPING_PATH, "Development Set video path and metadata mapping.", "Join OCR schedule to video files.", VIDEO_MAPPING_COLUMNS),
        ("ocr_schedule", OCR_SCHEDULE_PATH, "Primary OCR extractor input schedule.", "OCR extractor should seek these times in each video.", OCR_SCHEDULE_COLUMNS),
        ("input_contract", INPUT_CONTRACT_PATH, "Machine-readable contract for the next OCR step.", "Read before OCR extraction.", []),
        ("next_step_guide", NEXT_STEP_GUIDE_PATH, "Human-readable guide for next OCR task.", "Use as instruction handoff.", []),
    ]
    rows = []
    for artifact_type, path, purpose, usage, columns in artifacts:
        rows.append({
            "artifact_type": artifact_type,
            "artifact_path": str(path),
            "exists": str(path.exists()).lower(),
            "row_count": row_count(path),
            "purpose": purpose,
            "downstream_usage": usage,
            "required_columns_json": json.dumps(columns, ensure_ascii=False),
            "split_scope": "Development Set only",
            "ready_for_ocr_extraction": str(ready).lower(),
            "notes": "No OCR raw text/features generated in this preparation task.",
        })
    return rows


def build_summary_rows(video_mapping: list[dict[str, Any]], final_anchors: list[dict[str, Any]], schedule_rows: list[dict[str, Any]], ready: bool) -> list[dict[str, Any]]:
    opencv_only = sum(row["has_opencv_ffmpeg"] == "true" and row["has_resnet"] == "false" and row["has_transnetv2_conservative"] == "false" for row in final_anchors)
    resnet_only = sum(row["has_opencv_ffmpeg"] == "false" and row["has_resnet"] == "true" and row["has_transnetv2_conservative"] == "false" for row in final_anchors)
    trans_only = sum(row["has_opencv_ffmpeg"] == "false" and row["has_resnet"] == "false" and row["has_transnetv2_conservative"] == "true" for row in final_anchors)
    multi = len(final_anchors) - opencv_only - resnet_only - trans_only
    metrics = [
        ("development_video_count_expected", len(DEVELOPMENT_VIDEO_IDS), "Fixed v2.5 Development Set from original v2.4 train split."),
        ("development_video_count_mapped", sum(row["file_exists"] == "true" for row in video_mapping), "Mapped videos with existing files."),
        ("final_anchor_count_total", len(final_anchors), "Clustered final OCR anchors."),
        ("final_anchor_count_opencv_only", opencv_only, "Cluster contains only OpenCV/FFmpeg source."),
        ("final_anchor_count_resnet_only", resnet_only, "Cluster contains only ResNet source."),
        ("final_anchor_count_transnetv2_only", trans_only, "Cluster contains only TransNetV2 conservative source."),
        ("final_anchor_count_multi_source", multi, "Cluster contains at least two source families."),
        ("ocr_schedule_count_total", len(schedule_rows), "OCR-ready scheduled frame times."),
        ("ocr_anchor_dense_count", sum(row["schedule_source"] == "anchor_dense" for row in schedule_rows), "Dense schedule around final anchors."),
        ("ocr_background_regular_count", sum(row["schedule_source"] == "background_regular" for row in schedule_rows), "Regular background schedule outside anchor context."),
        ("schedule_ready_for_ocr", str(ready).lower(), "True when mapped videos and schedule rows are available."),
        ("ocr_extraction_executed", "false", "OCR is explicitly not executed in this task."),
        ("split_terminology_version", "v2.5_ruledev_extended_eval", "Development/Extended Evaluation terminology layer."),
        ("no_extended_evaluation_processed", "true", "Only original v2.4 train rows are processed."),
        ("no_diagnostic_subset_processed", "true", "Original validation split is not processed."),
        ("no_pure_test_processed", "true", "Original test split is not processed."),
    ]
    return [{"metric_name": name, "metric_value": value, "notes": notes} for name, value, notes in metrics]


def find_next_ocr_script() -> str:
    candidates = [
        PROJECT_ROOT / "scripts/ocr/extract_scene_anchor_full_video_ocr_v2_4.py",
        PROJECT_ROOT / "scripts/ocr/extract_ocr_visual_anchor_context_features_v2_4.py",
        PROJECT_ROOT / "scripts/ocr/extract_ocr_cues_v2_4.py",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def write_reports(report: dict[str, Any], summary_rows: list[dict[str, Any]], source_stats: dict[str, Any]) -> None:
    counts = {row["metric_name"]: row["metric_value"] for row in summary_rows}
    source_rows = [
        ["OpenCV/FFmpeg", source_stats.get("opencv_ffmpeg", {}).get("development_candidate_count", 0), str(OPENCV_PATH)],
        ["ResNet", source_stats.get("resnet", {}).get("development_candidate_count", 0), str(RESNET_PATH)],
        ["TransNetV2 conservative", source_stats.get("transnetv2_conservative", {}).get("development_candidate_count", 0), f"{TRANSNET_CONSERVATIVE_FAMILY}, threshold=0.7, dedup=5s"],
    ]
    schedule_table = markdown_table(
        ["metric", "value"],
        [["final anchor count", counts.get("final_anchor_count_total")], ["OCR schedule count", counts.get("ocr_schedule_count_total")], ["anchor_dense", counts.get("ocr_anchor_dense_count")], ["background_regular", counts.get("ocr_background_regular_count")]],
    )
    source_table = markdown_table(["source", "Development candidates", "input"], source_rows)
    summary_md = f"""# Final Scene Anchor OCR Inputs v2.5 Development Summary

## 작업 목적

OpenCV/FFmpeg, ResNet, TransNetV2 conservative 장면전환 후보를 결합한 신규 final scene anchor를 만들고, 다음 OCR 추출 작업에서 바로 읽을 수 있는 Development Set 전용 OCR-ready schedule과 input contract를 생성했다.

## v2.5 split terminology

{REQUIRED_SPLIT_PHRASE_KO}

이번 작업은 **Development Set only** 작업이다. Test Set은 처리하지 않았다.

## Final scene anchor 구성

{source_table}

TransNetV2 conservative 기준은 `threshold=0.7`, `dedup_window=5s`, `sweep_family={TRANSNET_CONSERVATIVE_FAMILY}`이다.

Final anchor는 같은 `video_id` 안에서 2초 이내 후보를 cluster로 묶었다. 평균 timestamp는 사용하지 않았고, 대표 timestamp는 기존 OpenCV/ResNet canonical anchor가 cluster에 있으면 canonical timestamp를 우선 사용했다. 없으면 source score가 가장 높은 timestamp를 사용하고, score 비교가 어려우면 가장 이른 timestamp를 사용했다.

## OCR-ready schedule 정책

- anchor 주변 범위: ±10초
- anchor 주변 OCR 간격: 1.0초
- anchor 외부 regular OCR 간격: 1.5초
- dedup tolerance: 0.1초
- 실제 OCR 실행: false

{schedule_table}

## 다음 OCR 작업에서 사용할 primary input

- `{OCR_SCHEDULE_PATH}`
- contract: `{INPUT_CONTRACT_PATH}`
- next-step guide: `{NEXT_STEP_GUIDE_PATH}`

## Safety

- existing detector/report/output 수정 없음
- existing visual anchor 수정 없음
- existing candidate file 수정 없음
- actual label 사용 없음
- OCR engine 실행 없음
- OCR raw/features output 생성 없음
- frame/cache/model/package 생성 없음
"""
    findings_md = f"""# Final Scene Anchor OCR Inputs v2.5 Development Findings

## 왜 기존 train을 Development Set이라고 부르는가

{REQUIRED_SPLIT_PHRASE_KO}

즉, 기존 `train`은 모델 weight 학습용이라기보다 rule 설계와 cue 분석을 위한 개발 영역이다. 그래서 새 산출물에서는 `original_split_v2_4=train`, `split_role_v2_5=development`, `evaluation_subset_v2_5=none`을 함께 기록했다.

## 왜 Test Set은 아직 처리하지 않았는가

이번 단계는 OCR 추출 직전 준비 단계이며, rule freeze 전 단서 고도화 작업이다. 따라서 기존 validation/test에 해당하는 Test Set의 row-level schedule은 만들지 않았다. Test Set은 규칙 고정 이후 최종 평가용으로 보호한다.

## 최종 anchor의 의미

최종 장면전환 anchor는 OpenCV/FFmpeg, ResNet, TransNetV2 conservative 세 source를 결합한 OCR 보조 anchor이다. detector rule에 자동 반영하지 않고, 기존 canonical visual anchor도 덮어쓰지 않는다.

## 왜 anchor 주변을 더 촘촘히 OCR하는가

광고 시작/종료 직전에는 고지 문구, 브랜드명, 할인/협찬 문구 같은 짧은 텍스트 cue가 빠르게 지나갈 수 있다. 그래서 final anchor 주변 ±10초는 1초 간격으로 촘촘히 schedule을 만들고, 나머지 구간은 1.5초 간격으로 유지해 OCR 비용을 줄였다.

## 왜 TransNetV2 primary가 아니라 conservative를 쓰는가

이전 실험에서 TransNetV2 primary는 recall은 높지만 후보 밀도가 높아 OCR 비용이 커지는 문제가 있었다. conservative 기준(`threshold=0.7`, `dedup=5s`)은 후보 수를 줄이면서도 기존 후보가 놓친 광고 경계를 보완하는 성격이 있어 OCR 보조 anchor로 더 적합하다.

## 다음 단계

다음 OCR 작업은 `{OCR_SCHEDULE_PATH}`를 primary input으로 읽으면 된다. 이번 작업은 OCR input preparation이며, OCR text/bbox/raw/features 결과는 생성하지 않았다.
"""
    next_guide = f"""# Next Step Guide: OCR Extraction from Final Scene Anchor Schedule

## Primary OCR input

OCR extractor가 읽을 primary input:

`{OCR_SCHEDULE_PATH}`

## Required columns

- `video_id`
- `video_path`
- `ocr_time_sec`
- `ocr_frame_index`
- `schedule_source`
- `nearest_final_anchor_id`
- `nearest_final_anchor_sec`
- `distance_to_nearest_anchor_sec`
- `original_split_v2_4`
- `split_role_v2_5`
- `evaluation_subset_v2_5`

## Scope

Development Set only. 이 schedule은 original v2.4 `train` split에 해당하는 12개 영상만 포함한다. Test Set schedule은 생성하지 않았다.

## Recommended OCR outputs for the next task

다음 OCR 실행 작업에서 저장할 권장 output:

- OCR raw result CSV
- OCR normalized feature CSV
- OCR run index CSV

이번 작업에서는 실제 OCR 실행을 하지 않았다. OCR engine 실행, frame image 생성, OCR raw text/bbox 결과 생성은 다음 작업에서 수행한다.

## Suggested extractor hook

기존 OCR script 후보:

`{report.get('next_recommended_script', '')}`

새 extractor는 위 script의 video seek/OCR 실행 부분을 재사용하되, 입력 schedule은 반드시 `{OCR_SCHEDULE_PATH}`를 사용하도록 연결하는 것이 좋다.
"""
    for path, text in [(SUMMARY_MD_PATH, summary_md), (FINDINGS_MD_PATH, findings_md), (NEXT_STEP_GUIDE_PATH, next_guide)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n", encoding="utf-8")


def validate_outputs(video_mapping: list[dict[str, Any]], final_anchors: list[dict[str, Any]], schedule_rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    split_columns = ["original_split_v2_4", "split_role_v2_5", "evaluation_subset_v2_5", "split_terminology_note"]
    output_sets = [video_mapping, final_anchors, schedule_rows]
    split_columns_present = all(all(col in (rows[0] if rows else {}) for col in split_columns) for rows in output_sets)
    schedule_vids = {int(row["video_id"]) for row in schedule_rows}
    anchor_vids = {int(row["video_id"]) for row in final_anchors}
    mapping_vids = {int(row["video_id"]) for row in video_mapping}
    invalid_schedule_times = []
    duration_by_video = {int(row["video_id"]): safe_float(row.get("duration_sec")) for row in video_mapping}
    for row in schedule_rows:
        vid = int(row["video_id"])
        t = safe_float(row.get("ocr_time_sec"))
        duration = duration_by_video.get(vid)
        if t is None or duration is None or t < -1e-9 or t > duration + 1e-6:
            invalid_schedule_times.append(row.get("schedule_id"))
    source_present = {
        "opencv_ffmpeg": any(row["has_opencv_ffmpeg"] == "true" for row in final_anchors),
        "resnet": any(row["has_resnet"] == "true" for row in final_anchors),
        "transnetv2_conservative": any(row["has_transnetv2_conservative"] == "true" for row in final_anchors),
    }
    forbidden_created = [str(PROJECT_ROOT / "data/ocr" / name) for name in FORBIDDEN_OUTPUT_NAMES if (PROJECT_ROOT / "data/ocr" / name).exists()]
    latest_forbidden = []
    for directory in [LATEST_BUNDLE_DIR, SHARED_DIR, LATEST_SCENE_DIR]:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
                latest_forbidden.append(str(path))
    return {
        "split_terminology_validation": {
            "passed": split_columns_present and schedule_vids <= set(DEVELOPMENT_VIDEO_IDS) and anchor_vids <= set(DEVELOPMENT_VIDEO_IDS) and mapping_vids == set(DEVELOPMENT_VIDEO_IDS),
            "details": {
                "split_columns_present": split_columns_present,
                "schedule_video_ids": sorted(schedule_vids),
                "anchor_video_ids": sorted(anchor_vids),
                "mapping_video_ids": sorted(mapping_vids),
                "diagnostic_or_pure_test_processed": bool((schedule_vids | anchor_vids | mapping_vids) & (set(DIAGNOSTIC_VIDEO_IDS) | set(PURE_TEST_VIDEO_IDS))),
            },
        },
        "final_scene_anchor_validation": {
            "passed": all(source_present.values()) and report["transnetv2_conservative_threshold"] == 0.7 and report["transnetv2_conservative_dedup_window_sec"] == 5.0,
            "details": {"source_present": source_present, "representative_policy_uses_average_timestamp": False},
        },
        "ocr_schedule_validation": {
            "passed": not invalid_schedule_times and bool(schedule_rows) and all(row.get("video_path") for row in schedule_rows),
            "details": {
                "invalid_schedule_time_count": len(invalid_schedule_times),
                "anchor_context_sec": ANCHOR_CONTEXT_SEC,
                "anchor_dense_step_sec": ANCHOR_DENSE_STEP_SEC,
                "background_step_sec": BACKGROUND_STEP_SEC,
                "dedup_tolerance_sec": DEDUP_TOLERANCE_SEC,
            },
        },
        "ocr_non_execution_validation": {
            "passed": not forbidden_created and report["ocr_extraction_executed"] is False,
            "details": {"forbidden_ocr_outputs_created": forbidden_created, "ocr_extraction_executed": report["ocr_extraction_executed"]},
        },
        "output_safety_validation": {
            "passed": not report["protected_files_modified"] and not latest_forbidden and not forbidden_created,
            "details": {"protected_files_modified": report["protected_files_modified"], "latest_forbidden_files": latest_forbidden},
        },
    }


def copy_latest(paths: list[Path], report: dict[str, Any]) -> None:
    for directory in [LATEST_BUNDLE_DIR, SHARED_DIR, LATEST_SCENE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        for path in paths:
            if path.exists():
                shutil.copy2(path, directory / path.name)
        readme_name = "README_latest_files.md" if directory != LATEST_SCENE_DIR else "README_final_scene_anchor_ocr_inputs_v2_5_development_latest_files.md"
        readme_path = directory / readme_name
        lines = [
            "# Latest Files: Final Scene Anchor OCR Inputs v2.5 Development",
            "",
            "이 폴더는 OCR 실행 직전 준비 산출물만 포함한다. raw video, frame image, cache, model weight, checkpoint, OCR raw/features output, Extended Evaluation row-level output은 포함하지 않는다.",
            "",
            "## Included Files",
            "",
        ]
        for path in paths:
            if path.exists():
                lines.append(f"- `{path.name}`")
        lines.extend([
            "",
            "## Scope",
            "",
            "- Development Set only",
            "- OCR extraction executed: false",
            f"- ready_for_next_ocr_extraction: `{str(report.get('ready_for_next_ocr_extraction')).lower()}`",
        ])
        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH.write_text("", encoding="utf-8")
    warnings: list[str] = []
    errors: list[str] = []

    log("[STEP 01] 안전 스냅샷 및 출력 경로 준비")
    backup_dir, backed_up = backup_existing_outputs()
    protected_inputs = [
        SPLIT_PATH, VIDEO_MANIFEST_PATH, OPENCV_PATH, RESNET_PATH, CANONICAL_PATH,
        CANONICAL_FALLBACK_PATH, TRANSNET_SWEEP_PATH, TRANSNET_BEST_PATH,
        TRANSNET_REPORT_PATH, ABLATION_REPORT_PATH, ABLATION_SUMMARY_PATH, ABLATION_DOC_PATH,
    ]
    protected_before = {str(path): file_stat(path) for path in protected_inputs if path.exists()}
    old_project_before = snapshot_file_tree(OLD_PROJECT_ROOT)

    log("[STEP 02] v2.4 split 로드 및 v2.5 split terminology 매핑")
    dev_rows, all_split_rows, split_check = load_split_rows()
    if not split_check["valid"]:
        errors.append("v2.4 split does not match fixed split definition")
        raise RuntimeError(json.dumps(split_check, ensure_ascii=False))
    if len(dev_rows) != len(DEVELOPMENT_VIDEO_IDS):
        warnings.append(f"Development Set split row count mismatch: {len(dev_rows)}")

    log("[STEP 03] Development Set 영상 경로 매핑")
    manifest = load_manifest()
    video_mapping = build_video_mapping(dev_rows, manifest)
    video_info = {int(row["video_id"]): row for row in video_mapping}
    mapped_video_count = sum(row["file_exists"] == "true" for row in video_mapping)
    if mapped_video_count == 0:
        raise RuntimeError("No Development Set video files could be mapped")
    if mapped_video_count < len(DEVELOPMENT_VIDEO_IDS):
        warnings.append(f"Only {mapped_video_count}/{len(DEVELOPMENT_VIDEO_IDS)} Development videos exist on disk")

    log("[STEP 04] OpenCV/FFmpeg 후보 로드")
    log("[STEP 05] ResNet 후보 로드")
    log("[STEP 06] TransNetV2 conservative 후보 로드")
    candidates, source_stats = load_candidates(video_info, warnings)
    source_counts = Counter(candidate["source"] for candidate in candidates)
    if sum(1 for source in ["opencv_ffmpeg", "resnet", "transnetv2_conservative"] if source_counts.get(source, 0) > 0) < 2:
        raise RuntimeError(f"Too few candidate sources available: {dict(source_counts)}")

    log("[STEP 07] 최종 scene anchor cluster 생성")
    canonical = load_canonical_anchors()
    final_anchors = cluster_candidates(candidates, canonical, video_info)

    log("[STEP 08] OCR-ready schedule 생성")
    schedule_rows = build_ocr_schedule(final_anchors, video_mapping)
    ready = bool(final_anchors and schedule_rows and mapped_video_count > 0)

    log("[STEP 09] OCR input contract 및 integration manifest 생성")
    next_script = find_next_ocr_script()
    contract = build_contract(ready, warnings, errors, next_script)
    summary_rows = build_summary_rows(video_mapping, final_anchors, schedule_rows, ready)
    write_csv(FINAL_ANCHOR_PATH, final_anchors, FINAL_ANCHOR_COLUMNS)
    write_csv(VIDEO_MAPPING_PATH, video_mapping, VIDEO_MAPPING_COLUMNS)
    write_csv(OCR_SCHEDULE_PATH, schedule_rows, OCR_SCHEDULE_COLUMNS)
    write_json(INPUT_CONTRACT_PATH, contract)
    manifest_rows = build_manifest(ready)
    write_csv(INTEGRATION_MANIFEST_PATH, manifest_rows, MANIFEST_COLUMNS)
    write_csv(PREP_SUMMARY_PATH, summary_rows, PREP_SUMMARY_COLUMNS)

    log("[STEP 10] markdown/json report 및 next-step guide 생성")
    protected_after = {str(path): file_stat(path) for path in protected_inputs if path.exists()}
    old_project_after = snapshot_file_tree(OLD_PROJECT_ROOT)
    protected_files_modified = protected_before != protected_after
    old_project_modified = old_project_before != old_project_after
    report: dict[str, Any] = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "created_at": now_iso(),
        "scope": "Development Set only (original v2.4 train split)",
        "split_seed": SPLIT_SEED,
        "development_set_video_ids": DEVELOPMENT_VIDEO_IDS,
        "diagnostic_subset_video_ids_not_processed": DIAGNOSTIC_VIDEO_IDS,
        "pure_test_set_video_ids_not_processed": PURE_TEST_VIDEO_IDS,
        "split_terminology_note": REQUIRED_SPLIT_PHRASE_KO,
        "split_check": split_check,
        "input_files": {"split": str(SPLIT_PATH), "opencv_ffmpeg": str(OPENCV_PATH), "resnet": str(RESNET_PATH), "transnetv2_conservative": str(TRANSNET_SWEEP_PATH), "canonical_reference": str(CANONICAL_PATH)},
        "source_stats": source_stats,
        "source_candidate_counts_loaded": dict(source_counts),
        "transnetv2_conservative_family": TRANSNET_CONSERVATIVE_FAMILY,
        "transnetv2_conservative_threshold": TRANSNET_CONSERVATIVE_THRESHOLD,
        "transnetv2_conservative_dedup_window_sec": TRANSNET_CONSERVATIVE_DEDUP_WINDOW_SEC,
        "final_anchor_policy": {
            "cluster_window_sec": CROSS_SOURCE_DEDUP_WINDOW_SEC,
            "mean_timestamp_used": False,
            "representative_priority": ["canonical_opencv_resnet_anchor", "max_source_score", "earliest_timestamp"],
        },
        "ocr_schedule_policy": {
            "anchor_context_sec": ANCHOR_CONTEXT_SEC,
            "anchor_dense_step_sec": ANCHOR_DENSE_STEP_SEC,
            "background_step_sec": BACKGROUND_STEP_SEC,
            "dedup_tolerance_sec": DEDUP_TOLERANCE_SEC,
        },
        "output_files": {
            "final_anchor": str(FINAL_ANCHOR_PATH),
            "video_path_mapping": str(VIDEO_MAPPING_PATH),
            "ocr_schedule": str(OCR_SCHEDULE_PATH),
            "input_contract": str(INPUT_CONTRACT_PATH),
            "integration_manifest": str(INTEGRATION_MANIFEST_PATH),
            "preparation_summary": str(PREP_SUMMARY_PATH),
            "summary_md": str(SUMMARY_MD_PATH),
            "findings_md": str(FINDINGS_MD_PATH),
            "next_step_guide": str(NEXT_STEP_GUIDE_PATH),
            "log": str(RUN_LOG_PATH),
            "latest_bundle": str(LATEST_BUNDLE_DIR),
            "shared_dir": str(SHARED_DIR),
            "latest_scene_copy_dir": str(LATEST_SCENE_DIR),
        },
        "output_row_counts": {
            "final_scene_anchor": len(final_anchors),
            "video_path_mapping": len(video_mapping),
            "ocr_schedule": len(schedule_rows),
            "integration_manifest": len(manifest_rows),
            "preparation_summary": len(summary_rows),
        },
        "development_video_count_expected": len(DEVELOPMENT_VIDEO_IDS),
        "development_video_count_mapped": mapped_video_count,
        "ocr_extraction_executed": False,
        "ready_for_next_ocr_extraction": ready,
        "next_recommended_script": next_script,
        "no_extended_evaluation_processed": True,
        "no_diagnostic_subset_processed": True,
        "no_pure_test_processed": True,
        "actual_label_used_for_schedule_generation": False,
        "detector_rule_modified": False,
        "existing_visual_anchor_modified": False,
        "protected_files_modified": protected_files_modified,
        "old_project_modified": old_project_modified,
        "backup_dir": str(backup_dir),
        "backed_up_files": backed_up,
        "warnings": warnings,
        "errors": errors,
    }
    write_reports(report, summary_rows, source_stats)

    log("[STEP 11] Sub Agent 검증 실행")
    validations = validate_outputs(video_mapping, final_anchors, schedule_rows, report)
    report["sub_agent_validations"] = validations
    report["status"] = "SUCCESS" if all(item["passed"] for item in validations.values()) else "CONDITIONAL_SUCCESS"
    write_json(REPORT_JSON_PATH, report)

    log("[STEP 12] latest bundle 및 latest_ocr_inputs_development 복사")
    latest_paths = [
        FINAL_ANCHOR_PATH, VIDEO_MAPPING_PATH, OCR_SCHEDULE_PATH, INPUT_CONTRACT_PATH,
        INTEGRATION_MANIFEST_PATH, PREP_SUMMARY_PATH, SUMMARY_MD_PATH, REPORT_JSON_PATH,
        FINDINGS_MD_PATH, NEXT_STEP_GUIDE_PATH, SCRIPT_PATH, RUN_LOG_PATH,
    ]
    copy_latest(latest_paths, report)
    # bundle step이 log에 기록된 뒤 final log를 복사한다.
    for directory in [LATEST_BUNDLE_DIR, SHARED_DIR, LATEST_SCENE_DIR]:
        if directory.exists():
            shutil.copy2(RUN_LOG_PATH, directory / RUN_LOG_PATH.name)

    log("[STEP 13] 최종 요약 출력")
    dense_count = sum(row["schedule_source"] == "anchor_dense" for row in schedule_rows)
    background_count = sum(row["schedule_source"] == "background_regular" for row in schedule_rows)
    final_summary = [
        f"작업 상태: {report['status']}",
        "처리 대상 split 표현: Development Set",
        f"Development Set video_id 목록: {DEVELOPMENT_VIDEO_IDS}",
        f"mapped Development Set 영상 수 / expected 12: {mapped_video_count}/12",
        f"final scene anchor count: {len(final_anchors)}",
        f"OCR schedule count: {len(schedule_rows)}",
        f"anchor_dense schedule count: {dense_count}",
        f"background_regular schedule count: {background_count}",
        "OCR extraction executed: false",
        f"ready_for_next_ocr_extraction: {str(ready).lower()}",
        f"final anchor path: {FINAL_ANCHOR_PATH}",
        f"OCR schedule path: {OCR_SCHEDULE_PATH}",
        f"OCR input contract path: {INPUT_CONTRACT_PATH}",
        f"next-step guide path: {NEXT_STEP_GUIDE_PATH}",
        f"latest bundle path: {LATEST_BUNDLE_DIR}",
        "Extended Evaluation/Diagnostic/Pure Test 미처리 확인: true",
        f"기존 detector/anchor/label/split 수정 없음: {str(not protected_files_modified).lower()}",
    ]
    for line in final_summary:
        log(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
