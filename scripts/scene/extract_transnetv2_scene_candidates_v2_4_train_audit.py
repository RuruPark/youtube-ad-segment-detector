#!/usr/bin/env python3
"""Train split 기준 TransNetV2 장면 후보를 추출하고 경계 포착률을 점검한다.

기존 visual anchor, label, split, detector config, prediction 파일은 수정하지 않는다.
정답 광고 구간은 후보 추출 뒤 포착률 점검에만 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import os
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(os.environ.get("YASD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
FIXED_SPLIT_SEED = "20240524"
TRAIN_VIDEO_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
VALIDATION_VIDEO_IDS = [3, 7, 18]
TEST_VIDEO_IDS = [4, 16, 17]
TOLERANCES = [2, 5, 10]
THRESHOLDS = [0.3, 0.5, 0.7]
PRIMARY_THRESHOLD = 0.5
DEDUP_WINDOW_SEC = 2.0

TRANSNET_PYTHONPATH = PROJECT_ROOT / "external/transnetv2/python"
TRANSNET_WEIGHT_PATH = (
    TRANSNET_PYTHONPATH
    / "transnetv2_pytorch"
    / "transnetv2-pytorch-weights.pth"
)
CV_FFMPEG_BIN_DIR = Path(".venv/bin")

SPLIT_PATH = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"
SEGMENT_PATH = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_4.csv"
MANIFEST_PATH = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
OPENCV_PATH = PROJECT_ROOT / "data/scene/scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed.csv"
RESNET_PATH = PROJECT_ROOT / "data/review/resnet_scene_candidate_human_review_v2_4_usefulness_analysis.csv"
CANONICAL_PATH = PROJECT_ROOT / "data/features/visual_scene_boundary_anchors_v2_4_with_split.csv"
PREVIOUS_AUDIT_CASE_PATH = PROJECT_ROOT / "data/scene/opencv_resnet_boundary_case_breakdown_v2_4_train.csv"
PREVIOUS_AUDIT_REPORT_PATH = PROJECT_ROOT / "reports/scene/opencv_resnet_scene_boundary_recall_audit_v2_4_report.json"
SETUP_REPORT_PATH = PROJECT_ROOT / "reports/scene/transnetv2_setup_check_v2_4_report.json"

DATA_SCENE_DIR = PROJECT_ROOT / "data/scene"
REPORT_SCENE_DIR = PROJECT_ROOT / "reports/scene"
SCRIPT_PATH = PROJECT_ROOT / "scripts/scene/extract_transnetv2_scene_candidates_v2_4_train_audit.py"
LOG_PATH = PROJECT_ROOT / "logs/transnetv2_scene_candidate_audit_v2_4_run_log.txt"
RAW_OUTPUT_DIR = DATA_SCENE_DIR / "transnetv2_raw_outputs_v2_4_train"
LATEST_BUNDLE_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_transnetv2_scene_candidate_audit_v2_4"
SHARED_LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"

VIDEO_MAPPING_CSV = DATA_SCENE_DIR / "transnetv2_video_path_mapping_v2_4_train.csv"
TRANSNET_CANDIDATES_CSV = DATA_SCENE_DIR / "transnetv2_scene_candidates_v2_4_train.csv"
RAW_INDEX_CSV = DATA_SCENE_DIR / "transnetv2_scene_candidates_v2_4_train_raw_outputs_index.csv"
SOURCE_COMPARISON_CSV = DATA_SCENE_DIR / "transnetv2_scene_candidate_source_comparison_v2_4_train.csv"
RECALL_AUDIT_CSV = DATA_SCENE_DIR / "transnetv2_boundary_recall_audit_v2_4_train.csv"
RECALL_SUMMARY_CSV = DATA_SCENE_DIR / "transnetv2_boundary_recall_summary_v2_4_train.csv"
CASE_BREAKDOWN_CSV = DATA_SCENE_DIR / "transnetv2_boundary_case_breakdown_v2_4_train.csv"
RECOVERED_CASES_CSV = DATA_SCENE_DIR / "transnetv2_existing_missed_recovered_cases_v2_4_train.csv"
VIDEO_LEVEL_RECALL_CSV = DATA_SCENE_DIR / "transnetv2_video_level_recall_v2_4_train.csv"

SUMMARY_MD = REPORT_SCENE_DIR / "transnetv2_scene_candidate_audit_v2_4_summary.md"
REPORT_JSON = REPORT_SCENE_DIR / "transnetv2_scene_candidate_audit_v2_4_report.json"
FINDINGS_MD = REPORT_SCENE_DIR / "transnetv2_scene_candidate_audit_v2_4_findings.md"

INPUT_FILES = [
    SPLIT_PATH,
    SEGMENT_PATH,
    MANIFEST_PATH,
    OPENCV_PATH,
    RESNET_PATH,
    CANONICAL_PATH,
    PREVIOUS_AUDIT_CASE_PATH,
    PREVIOUS_AUDIT_REPORT_PATH,
    SETUP_REPORT_PATH,
    TRANSNET_WEIGHT_PATH,
]


class Logger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def step(self, number: int, title: str) -> None:
        self.write(f"[STEP {number:02d}] {title}")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: csv_value(row.get(name)) for name in fieldnames})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if text == "":
            return None
        number = float(text)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def clean_video_id(value: Any) -> Optional[int]:
    number = safe_int(value)
    if number is None:
        return None
    return number


def format_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def candidate_family_for_threshold(threshold: float) -> str:
    if abs(threshold - PRIMARY_THRESHOLD) < 1e-9:
        return "transnetv2_primary"
    suffix = str(threshold).replace(".", "_")
    return f"transnetv2_threshold_{suffix}"


def snapshot_paths(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if path.exists():
            stat = path.stat()
            snapshot[str(path)] = {
                "exists": True,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        else:
            snapshot[str(path)] = {"exists": False, "size_bytes": None, "mtime_ns": None}
    return snapshot


def changed_paths(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> List[str]:
    changed: List[str] = []
    for path, old in before.items():
        new = after.get(path)
        if new != old:
            changed.append(path)
    return changed


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split(logger: Logger) -> Tuple[List[Dict[str, str]], Dict[int, Dict[str, str]], Dict[str, Any]]:
    rows = read_csv_rows(SPLIT_PATH)
    by_video: Dict[int, Dict[str, str]] = {}
    train_found: List[int] = []
    validation_found: List[int] = []
    test_found: List[int] = []
    seeds = set()
    for row in rows:
        vid = clean_video_id(row.get("video_id"))
        if vid is None:
            continue
        by_video[vid] = row
        if row.get("split") == "train":
            train_found.append(vid)
        if row.get("split") == "validation":
            validation_found.append(vid)
        if row.get("split") == "test":
            test_found.append(vid)
        if row.get("split_seed"):
            seeds.add(str(row.get("split_seed")))
    validation = {
        "split_file_exists": SPLIT_PATH.exists(),
        "split_seed_expected": FIXED_SPLIT_SEED,
        "split_seed_values": sorted(seeds),
        "train_video_ids_expected": TRAIN_VIDEO_IDS,
        "train_video_ids_found": sorted(train_found),
        "validation_video_ids_expected": VALIDATION_VIDEO_IDS,
        "validation_video_ids_found": sorted(validation_found),
        "test_video_ids_expected": TEST_VIDEO_IDS,
        "test_video_ids_found": sorted(test_found),
        "fixed_train_match": sorted(train_found) == TRAIN_VIDEO_IDS,
        "fixed_validation_match": sorted(validation_found) == VALIDATION_VIDEO_IDS,
        "fixed_test_match": sorted(test_found) == TEST_VIDEO_IDS,
    }
    logger.write(f"Split train videos: {sorted(train_found)}")
    return rows, by_video, validation


def load_manifest() -> Dict[int, Dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return {}
    rows = read_csv_rows(MANIFEST_PATH)
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        vid = clean_video_id(row.get("video_id"))
        if vid is not None:
            out[vid] = row
    return out


def load_segment_rows() -> List[Dict[str, str]]:
    return read_csv_rows(SEGMENT_PATH) if SEGMENT_PATH.exists() else []


def map_train_videos(
    split_by_video: Dict[int, Dict[str, str]],
    manifest_by_video: Dict[int, Dict[str, str]],
    segment_rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    segment_by_video: Dict[int, Dict[str, str]] = {}
    for row in segment_rows:
        vid = clean_video_id(row.get("video_id"))
        if vid in TRAIN_VIDEO_IDS and row.get("video_path"):
            segment_by_video.setdefault(vid, row)

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for vid in TRAIN_VIDEO_IDS:
        split_row = split_by_video.get(vid, {})
        manifest_row = manifest_by_video.get(vid, {})
        segment_row = segment_by_video.get(vid, {})
        path = split_row.get("video_path") or manifest_row.get("video_path") or segment_row.get("video_path") or ""
        mapping_source = "split.video_path" if split_row.get("video_path") else ""
        if not mapping_source and manifest_row.get("video_path"):
            mapping_source = "video_manifest.video_path"
        if not mapping_source and segment_row.get("video_path"):
            mapping_source = "ad_interval_segments.video_path"
        if not mapping_source:
            mapping_source = "not_found"
        exists = bool(path and Path(path).exists())
        if not exists:
            warnings.append(f"missing_video_path: video_id={vid}, path={path}")
        duration = (
            safe_float(manifest_row.get("duration_sec"))
            or safe_float(split_row.get("video_duration_sec"))
            or safe_float(segment_row.get("video_duration_sec"))
        )
        fps = safe_float(manifest_row.get("fps"))
        frame_count = safe_int(manifest_row.get("frame_count"))
        video_title = (
            split_row.get("video_name")
            or manifest_row.get("video_title")
            or segment_row.get("video_title")
            or ""
        )
        rows.append(
            {
                "video_id": vid,
                "split": split_row.get("split", "train"),
                "video_title": video_title,
                "video_path": path,
                "file_exists": exists,
                "duration_sec": duration,
                "fps": fps,
                "frame_count": frame_count,
                "mapping_source": mapping_source,
                "notes": "" if exists else "missing video file; skipped for inference",
            }
        )
    return rows, warnings


def infer_transnetv2(
    mapping_rows: List[Dict[str, Any]],
    logger: Logger,
    warnings: List[str],
    errors: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """TransNetV2 추론 결과를 후보 row와 raw output index로 반환한다."""
    candidate_rows: List[Dict[str, Any]] = []
    raw_index_rows: List[Dict[str, Any]] = []
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if str(TRANSNET_PYTHONPATH) not in sys.path:
        sys.path.insert(0, str(TRANSNET_PYTHONPATH))
    if CV_FFMPEG_BIN_DIR.exists():
        os.environ["PATH"] = str(CV_FFMPEG_BIN_DIR) + os.pathsep + os.environ.get("PATH", "")

    import numpy as np  # type: ignore
    import torch  # type: ignore
    from transnetv2_pytorch import TransNetV2  # type: ignore

    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    fallback_used = False
    logger.write(f"TransNetV2 initial device: {device}; torch.cuda.is_available={cuda_available}")

    model = None

    def make_model(target_device: str) -> Any:
        logger.write(f"Initializing TransNetV2 model on device={target_device}")
        return TransNetV2(device=target_device)

    try:
        model = make_model(device)
    except Exception as exc:
        warnings.append(f"TransNetV2 model initialization failed on {device}: {exc}")
        logger.write(f"Model initialization failed on {device}; falling back to CPU")
        device = "cpu"
        model = make_model(device)
        fallback_used = True

    raw_row_index = 0
    for row in mapping_rows:
        vid = int(row["video_id"])
        if not row.get("file_exists"):
            for threshold in THRESHOLDS:
                raw_index_rows.append(
                    {
                        "video_id": vid,
                        "video_path": row.get("video_path", ""),
                        "raw_output_path": "",
                        "output_format": "json",
                        "threshold": threshold,
                        "row_count": 0,
                        "device_used": "",
                        "runtime_seconds": 0,
                        "status": "SKIPPED_MISSING_VIDEO",
                        "error_message": "video file missing",
                    }
                )
            continue

        video_path = str(row["video_path"])
        raw_output_path = RAW_OUTPUT_DIR / f"video_{vid:02d}_transnetv2_raw_candidates.json"
        logger.write(f"Running TransNetV2 inference for video_id={vid} on {device}: {video_path}")
        start_time = time.time()
        status = "PASS"
        error_message = ""
        per_threshold: Dict[str, Dict[str, Any]] = {}
        fps = safe_float(row.get("fps"))
        duration = safe_float(row.get("duration_sec"))
        frame_count = safe_int(row.get("frame_count"))
        active_device = device

        try:
            video_tensor, single_frame_predictions, _all_frame_predictions = model.predict_video(video_path, quiet=True)
        except Exception as exc:
            if device != "cpu":
                warnings.append(f"CUDA inference failed for video_id={vid}; CPU fallback used: {exc}")
                logger.write(f"CUDA inference failed for video_id={vid}: {exc}; retrying on CPU")
                try:
                    del model
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                device = "cpu"
                active_device = "cpu"
                fallback_used = True
                model = make_model("cpu")
                try:
                    video_tensor, single_frame_predictions, _all_frame_predictions = model.predict_video(video_path, quiet=True)
                except Exception as cpu_exc:
                    runtime = time.time() - start_time
                    status = "FAIL"
                    error_message = traceback.format_exc()
                    errors.append(f"TransNetV2 inference failed for video_id={vid} on CUDA and CPU: {cpu_exc}")
                    logger.write(f"video_id={vid} failed on CUDA and CPU; continuing")
                    for threshold in THRESHOLDS:
                        raw_index_rows.append(
                            {
                                "video_id": vid,
                                "video_path": video_path,
                                "raw_output_path": str(raw_output_path),
                                "output_format": "json",
                                "threshold": threshold,
                                "row_count": 0,
                                "device_used": active_device,
                                "runtime_seconds": runtime,
                                "status": status,
                                "error_message": error_message,
                            }
                        )
                    continue
            else:
                runtime = time.time() - start_time
                status = "FAIL"
                error_message = traceback.format_exc()
                errors.append(f"TransNetV2 inference failed for video_id={vid} on CPU: {exc}")
                logger.write(f"video_id={vid} failed on CPU; continuing")
                for threshold in THRESHOLDS:
                    raw_index_rows.append(
                        {
                            "video_id": vid,
                            "video_path": video_path,
                            "raw_output_path": str(raw_output_path),
                            "output_format": "json",
                            "threshold": threshold,
                            "row_count": 0,
                            "device_used": active_device,
                            "runtime_seconds": runtime,
                            "status": status,
                            "error_message": error_message,
                        }
                    )
                continue

        try:
            predictions = single_frame_predictions.cpu().detach().numpy().reshape(-1)
            frame_count = int(len(predictions))
            fps_from_model = safe_float(model.get_video_fps(video_path))
            if fps_from_model:
                fps = fps_from_model
            if duration is None and fps:
                duration = frame_count / fps

            for threshold in THRESHOLDS:
                scenes = model.predictions_to_scenes_with_data(predictions, fps=fps, threshold=threshold)
                threshold_candidates: List[Dict[str, Any]] = []
                for scene_index, scene in enumerate(scenes):
                    if scene_index == len(scenes) - 1:
                        continue
                    boundary_frame = int(scene.get("end_frame", 0))
                    start_idx = max(0, boundary_frame - 2)
                    end_idx = min(len(predictions), boundary_frame + 3)
                    window = predictions[start_idx:end_idx]
                    score = float(np.max(window)) if len(window) else safe_float(scene.get("probability")) or 0.0
                    candidate_sec = (boundary_frame / fps) if fps else safe_float(scene.get("end_time"))
                    candidate = {
                        "candidate_frame": boundary_frame,
                        "candidate_sec": candidate_sec,
                        "transnetv2_score": score,
                        "threshold": threshold,
                        "source_scene_start_frame": scene.get("start_frame"),
                        "source_scene_end_frame": scene.get("end_frame"),
                        "source_scene_start_time": scene.get("start_time"),
                        "source_scene_end_time": scene.get("end_time"),
                    }
                    threshold_candidates.append(candidate)
                per_threshold[str(threshold)] = {
                    "scene_count": len(scenes),
                    "candidate_count": len(threshold_candidates),
                    "candidates": threshold_candidates,
                }
        except Exception as parse_exc:
            runtime = time.time() - start_time
            status = "FAIL"
            error_message = traceback.format_exc()
            errors.append(f"TransNetV2 output parsing failed for video_id={vid}: {parse_exc}")
            logger.write(f"video_id={vid} output parsing failed; continuing")
            for threshold in THRESHOLDS:
                raw_index_rows.append(
                    {
                        "video_id": vid,
                        "video_path": video_path,
                        "raw_output_path": str(raw_output_path),
                        "output_format": "json",
                        "threshold": threshold,
                        "row_count": 0,
                        "device_used": active_device,
                        "runtime_seconds": runtime,
                        "status": status,
                        "error_message": error_message,
                    }
                )
            continue
        finally:
            try:
                del video_tensor
                del single_frame_predictions
                del _all_frame_predictions
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        runtime = time.time() - start_time
        raw_payload = {
            "video_id": vid,
            "video_path": video_path,
            "device_used": active_device,
            "runtime_seconds": runtime,
            "fps": fps,
            "duration_sec": duration,
            "frame_count": frame_count,
            "threshold_policy": {
                "primary_threshold": PRIMARY_THRESHOLD,
                "optional_thresholds": [0.3, 0.7],
                "threshold_tuning_used": False,
            },
            "candidate_timestamp_rule": "scene end_frame for every TransNetV2 scene segment except the final tail segment",
            "thresholds": per_threshold,
        }
        raw_output_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        for threshold in THRESHOLDS:
            family = candidate_family_for_threshold(threshold)
            threshold_data = per_threshold.get(str(threshold), {"candidates": []})
            threshold_candidates = threshold_data.get("candidates", [])
            raw_index_rows.append(
                {
                    "video_id": vid,
                    "video_path": video_path,
                    "raw_output_path": str(raw_output_path),
                    "output_format": "json",
                    "threshold": threshold,
                    "row_count": len(threshold_candidates),
                    "device_used": active_device,
                    "runtime_seconds": runtime,
                    "status": status,
                    "error_message": error_message,
                }
            )
            for item in threshold_candidates:
                raw_row_index += 1
                candidate_rows.append(
                    {
                        "candidate_family": family,
                        "video_id": vid,
                        "split": "train",
                        "candidate_sec": item.get("candidate_sec"),
                        "candidate_frame": item.get("candidate_frame"),
                        "transnetv2_score": item.get("transnetv2_score"),
                        "threshold": threshold,
                        "source_output_path": str(raw_output_path),
                        "device_used": active_device,
                        "fps": fps,
                        "duration_sec": duration,
                        "raw_row_index": raw_row_index,
                        "notes": "candidate_sec=end_frame/fps; labels not used for extraction",
                    }
                )
        logger.write(
            f"video_id={vid} done in {runtime:.2f}s; primary candidates={per_threshold.get(str(PRIMARY_THRESHOLD), {}).get('candidate_count', 0)}"
        )

    inference_meta = {
        "cuda_available": cuda_available,
        "initial_device": "cuda" if cuda_available else "cpu",
        "final_device": device,
        "cuda_to_cpu_fallback_used": fallback_used,
        "processed_video_count": len([r for r in mapping_rows if r.get("file_exists")]),
        "failed_video_count": len([r for r in raw_index_rows if r.get("status") not in {"PASS", "SKIPPED_MISSING_VIDEO"}]),
    }
    return candidate_rows, raw_index_rows, inference_meta


def load_existing_candidates(train_ids: Sequence[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    train_set = set(train_ids)
    warnings: List[str] = []

    opencv_rows: List[Dict[str, Any]] = []
    if OPENCV_PATH.exists():
        for idx, row in enumerate(read_csv_rows(OPENCV_PATH), start=1):
            vid = clean_video_id(row.get("video_id"))
            sec = safe_float(row.get("candidate_time_sec"))
            if vid in train_set and sec is not None:
                opencv_rows.append(
                    {
                        "candidate_family": "opencv_ffmpeg",
                        "video_id": vid,
                        "candidate_sec": sec,
                        "score": safe_float(row.get("scene_change_score")),
                        "threshold": safe_float(row.get("threshold")),
                        "source_relation": row.get("candidate_source_for_audit") or row.get("candidate_source") or "opencv_ffmpeg",
                        "source_output_path": str(OPENCV_PATH),
                        "raw_row_index": idx,
                        "notes": row.get("method_used") or "",
                    }
                )
    else:
        warnings.append(f"missing OpenCV/FFmpeg candidate file: {OPENCV_PATH}")

    resnet_rows: List[Dict[str, Any]] = []
    if RESNET_PATH.exists():
        for idx, row in enumerate(read_csv_rows(RESNET_PATH), start=1):
            vid = clean_video_id(row.get("video_id"))
            sec = safe_float(row.get("resnet_time_sec_std"))
            if sec is None:
                sec = safe_float(row.get("candidate_time_sec"))
            if vid in train_set and sec is not None:
                resnet_rows.append(
                    {
                        "candidate_family": "resnet",
                        "video_id": vid,
                        "candidate_sec": sec,
                        "score": safe_float(row.get("resnet_score_std")) or safe_float(row.get("scene_change_score")),
                        "threshold": safe_float(row.get("threshold")),
                        "source_relation": row.get("candidate_source") or row.get("model_name") or "resnet",
                        "source_output_path": str(RESNET_PATH),
                        "raw_row_index": idx,
                        "notes": row.get("method_used") or row.get("model_name") or "",
                    }
                )
    else:
        warnings.append(f"missing ResNet candidate file: {RESNET_PATH}")

    canonical_rows: List[Dict[str, Any]] = []
    if CANONICAL_PATH.exists():
        for idx, row in enumerate(read_csv_rows(CANONICAL_PATH), start=1):
            vid = clean_video_id(row.get("video_id"))
            sec = safe_float(row.get("canonical_boundary_time_sec"))
            split = row.get("split")
            if vid in train_set and sec is not None and (split in {"train", ""} or split is None):
                canonical_rows.append(
                    {
                        "candidate_family": "canonical_all",
                        "video_id": vid,
                        "candidate_sec": sec,
                        "score": safe_float(row.get("visual_boundary_strength_score")),
                        "threshold": None,
                        "source_relation": row.get("source_relation") or row.get("canonical_time_source") or "canonical_all",
                        "source_output_path": str(CANONICAL_PATH),
                        "raw_row_index": idx,
                        "notes": "canonical visual anchor read-only input",
                    }
                )
    else:
        warnings.append(f"missing canonical anchor file: {CANONICAL_PATH}")

    return opencv_rows, resnet_rows, canonical_rows, warnings


def connected_components(nodes: List[Dict[str, Any]], edge_predicate: Any) -> List[List[int]]:
    parent = list(range(len(nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if edge_predicate(nodes[i], nodes[j]):
                union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def build_existing_combined_clusters(
    opencv_rows: List[Dict[str, Any]],
    resnet_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_video: Dict[int, List[Dict[str, Any]]] = {}
    for row in opencv_rows:
        item = dict(row)
        item["source_family"] = "opencv_ffmpeg"
        item["member_times"] = [float(row["candidate_sec"])]
        by_video.setdefault(int(row["video_id"]), []).append(item)
    for row in resnet_rows:
        item = dict(row)
        item["source_family"] = "resnet"
        item["member_times"] = [float(row["candidate_sec"])]
        by_video.setdefault(int(row["video_id"]), []).append(item)

    clusters: List[Dict[str, Any]] = []
    cluster_id = 0
    for vid, nodes in sorted(by_video.items()):
        def edge(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            if a["source_family"] == b["source_family"]:
                return False
            return abs(float(a["candidate_sec"]) - float(b["candidate_sec"])) <= DEDUP_WINDOW_SEC

        for group in connected_components(nodes, edge):
            members = [nodes[i] for i in group]
            cluster_id += 1
            source_set = sorted({m["source_family"] for m in members})
            member_times = sorted(float(m["candidate_sec"]) for m in members)
            if set(source_set) == {"opencv_ffmpeg", "resnet"}:
                relation = "opencv_resnet_merged_2s"
            elif source_set == ["opencv_ffmpeg"]:
                relation = "opencv_ffmpeg_only"
            else:
                relation = "resnet_only"
            representative = member_times[0]
            opencv_times = [float(m["candidate_sec"]) for m in members if m["source_family"] == "opencv_ffmpeg"]
            resnet_times = [float(m["candidate_sec"]) for m in members if m["source_family"] == "resnet"]
            if opencv_times and resnet_times:
                representative = min(opencv_times, key=lambda x: min(abs(x - y) for y in resnet_times))
            clusters.append(
                {
                    "cluster_id": f"existing_{cluster_id:06d}",
                    "candidate_family": "existing_combined_two_source",
                    "video_id": vid,
                    "candidate_sec": representative,
                    "member_times": member_times,
                    "member_sources": source_set,
                    "source_relation": relation,
                    "notes": f"2s cross-source dedup cluster; members={len(members)}",
                }
            )
    return clusters


def build_existing_plus_transnet_clusters(
    existing_clusters: List[Dict[str, Any]],
    transnet_primary_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_video: Dict[int, List[Dict[str, Any]]] = {}
    for cluster in existing_clusters:
        item = dict(cluster)
        item["source_family"] = "existing_combined_two_source"
        by_video.setdefault(int(item["video_id"]), []).append(item)
    for row in transnet_primary_rows:
        sec = safe_float(row.get("candidate_sec"))
        if sec is None:
            continue
        by_video.setdefault(int(row["video_id"]), []).append(
            {
                "candidate_family": "combined_existing_plus_transnetv2",
                "video_id": int(row["video_id"]),
                "candidate_sec": sec,
                "member_times": [sec],
                "member_sources": ["transnetv2_primary"],
                "source_family": "transnetv2_primary",
                "source_relation": "transnetv2_primary",
                "notes": "TransNetV2 primary member",
            }
        )

    clusters: List[Dict[str, Any]] = []
    cluster_id = 0
    for vid, nodes in sorted(by_video.items()):
        def edge(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            a_times = [float(x) for x in a.get("member_times", [a["candidate_sec"]])]
            b_times = [float(x) for x in b.get("member_times", [b["candidate_sec"]])]
            return any(abs(x - y) <= DEDUP_WINDOW_SEC for x in a_times for y in b_times)

        for group in connected_components(nodes, edge):
            members = [nodes[i] for i in group]
            cluster_id += 1
            member_times: List[float] = []
            source_set = set()
            existing_representatives: List[float] = []
            transnet_representatives: List[float] = []
            for member in members:
                member_times.extend(float(x) for x in member.get("member_times", [member["candidate_sec"]]))
                for source in member.get("member_sources", [member.get("source_family", "")]):
                    source_set.add(source)
                if member.get("source_family") == "existing_combined_two_source":
                    existing_representatives.append(float(member["candidate_sec"]))
                elif member.get("source_family") == "transnetv2_primary":
                    transnet_representatives.append(float(member["candidate_sec"]))
            member_times = sorted(member_times)
            if existing_representatives:
                representative = sorted(existing_representatives)[0]
                relation = "existing_cluster_with_or_without_transnetv2"
            else:
                representative = sorted(transnet_representatives)[0] if transnet_representatives else member_times[0]
                relation = "transnetv2_only_cluster"
            clusters.append(
                {
                    "cluster_id": f"existing_plus_transnet_{cluster_id:06d}",
                    "candidate_family": "combined_existing_plus_transnetv2",
                    "video_id": vid,
                    "candidate_sec": representative,
                    "member_times": member_times,
                    "member_sources": sorted(source_set),
                    "source_relation": relation,
                    "notes": f"2s all-source dedup cluster; members={len(members)}",
                }
            )
    return clusters


def construct_actual_boundaries(segment_rows: List[Dict[str, str]], train_ids: Sequence[int]) -> Tuple[List[Dict[str, Any]], List[str]]:
    train_set = set(train_ids)
    boundaries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for idx, row in enumerate(segment_rows, start=1):
        vid = clean_video_id(row.get("video_id"))
        if vid not in train_set:
            continue
        if row.get("segment_type") and row.get("segment_type") != "ad_interval":
            continue
        ad_interval_id = row.get("ad_interval_id") or row.get("segment_id") or f"row_{idx}"
        for boundary_type, column in [("start", "ad_start_sec"), ("end", "ad_end_sec")]:
            sec = safe_float(row.get(column))
            if sec is None or sec < 0:
                warnings.append(f"invalid actual boundary: video_id={vid}, interval={ad_interval_id}, column={column}, value={row.get(column)}")
                continue
            boundaries.append(
                {
                    "video_id": vid,
                    "ad_interval_id": ad_interval_id,
                    "boundary_type": boundary_type,
                    "actual_sec": sec,
                }
            )
    return boundaries, warnings


def group_times(rows: List[Dict[str, Any]]) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for row in rows:
        sec = safe_float(row.get("candidate_sec"))
        vid = clean_video_id(row.get("video_id"))
        if vid is not None and sec is not None:
            out.setdefault(vid, []).append(sec)
    for vid in out:
        out[vid].sort()
    return out


def group_member_times(cluster_rows: List[Dict[str, Any]]) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for row in cluster_rows:
        vid = clean_video_id(row.get("video_id"))
        if vid is None:
            continue
        times = [safe_float(x) for x in row.get("member_times", [])]
        for sec in times:
            if sec is not None:
                out.setdefault(vid, []).append(sec)
    for vid in out:
        out[vid].sort()
    return out


def candidate_count_by_video_from_rows(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for row in rows:
        vid = clean_video_id(row.get("video_id"))
        if vid is not None:
            counts[vid] = counts.get(vid, 0) + 1
    return counts


def nearest_candidate(times: Sequence[float], actual_sec: float) -> Tuple[Optional[float], Optional[float]]:
    if not times:
        return None, None
    best = min(times, key=lambda sec: (abs(sec - actual_sec), sec))
    return best, abs(best - actual_sec)


def build_family_sources(
    opencv_rows: List[Dict[str, Any]],
    resnet_rows: List[Dict[str, Any]],
    canonical_rows: List[Dict[str, Any]],
    transnet_rows: List[Dict[str, Any]],
    existing_clusters: List[Dict[str, Any]],
    existing_plus_transnet_clusters: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    family_sources: Dict[str, Dict[str, Any]] = {}
    family_sources["opencv_ffmpeg"] = {
        "times_by_video": group_times(opencv_rows),
        "candidate_count_by_video": candidate_count_by_video_from_rows(opencv_rows),
        "candidate_count_total": len(opencv_rows),
    }
    family_sources["resnet"] = {
        "times_by_video": group_times(resnet_rows),
        "candidate_count_by_video": candidate_count_by_video_from_rows(resnet_rows),
        "candidate_count_total": len(resnet_rows),
    }
    family_sources["existing_combined_two_source"] = {
        "times_by_video": group_member_times(existing_clusters),
        "candidate_count_by_video": candidate_count_by_video_from_rows(existing_clusters),
        "candidate_count_total": len(existing_clusters),
    }
    for threshold in THRESHOLDS:
        family = candidate_family_for_threshold(threshold)
        rows = [row for row in transnet_rows if row.get("candidate_family") == family]
        family_sources[family] = {
            "times_by_video": group_times(rows),
            "candidate_count_by_video": candidate_count_by_video_from_rows(rows),
            "candidate_count_total": len(rows),
        }
    family_sources["combined_existing_plus_transnetv2"] = {
        "times_by_video": group_member_times(existing_plus_transnet_clusters),
        "candidate_count_by_video": candidate_count_by_video_from_rows(existing_plus_transnet_clusters),
        "candidate_count_total": len(existing_plus_transnet_clusters),
    }
    if canonical_rows:
        family_sources["canonical_all"] = {
            "times_by_video": group_times(canonical_rows),
            "candidate_count_by_video": candidate_count_by_video_from_rows(canonical_rows),
            "candidate_count_total": len(canonical_rows),
        }
    return family_sources


def compute_recall_audit(
    boundaries: List[Dict[str, Any]],
    family_sources: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for boundary in boundaries:
        vid = int(boundary["video_id"])
        actual_sec = float(boundary["actual_sec"])
        for family, data in family_sources.items():
            times = data["times_by_video"].get(vid, [])
            nearest_sec, distance = nearest_candidate(times, actual_sec)
            rows.append(
                {
                    "video_id": vid,
                    "ad_interval_id": boundary["ad_interval_id"],
                    "boundary_type": boundary["boundary_type"],
                    "actual_sec": actual_sec,
                    "candidate_family": family,
                    "nearest_candidate_sec": nearest_sec,
                    "nearest_distance_sec": distance,
                    "within_2s": distance is not None and distance <= 2,
                    "within_5s": distance is not None and distance <= 5,
                    "within_10s": distance is not None and distance <= 10,
                    "candidate_count_in_video": data["candidate_count_by_video"].get(vid, 0),
                    "notes": "nearest over original member timestamps for combined families" if "combined" in family else "",
                }
            )
    return rows


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[int(pos)]
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def summarize_recall(
    audit_rows: List[Dict[str, Any]],
    family_sources: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    families = list(family_sources.keys())
    for family in families:
        family_rows = [row for row in audit_rows if row["candidate_family"] == family]
        for boundary_type in ["start", "end", "all"]:
            type_rows = family_rows if boundary_type == "all" else [row for row in family_rows if row["boundary_type"] == boundary_type]
            distances = [
                float(row["nearest_distance_sec"])
                for row in type_rows
                if row.get("nearest_distance_sec") not in (None, "")
            ]
            for tolerance in TOLERANCES:
                hit_col = f"within_{tolerance}s"
                hit_count = sum(1 for row in type_rows if row.get(hit_col) is True)
                actual_count = len(type_rows)
                summary_rows.append(
                    {
                        "candidate_family": family,
                        "boundary_type": boundary_type,
                        "tolerance_sec": tolerance,
                        "actual_boundary_count": actual_count,
                        "hit_count": hit_count,
                        "recall": (hit_count / actual_count) if actual_count else None,
                        "median_nearest_distance_sec": statistics.median(distances) if distances else None,
                        "mean_nearest_distance_sec": statistics.mean(distances) if distances else None,
                        "p90_nearest_distance_sec": quantile(distances, 0.9),
                        "max_nearest_distance_sec": max(distances) if distances else None,
                        "candidate_count_total": family_sources[family]["candidate_count_total"],
                        "candidate_video_count": len(family_sources[family]["candidate_count_by_video"]),
                    }
                )
    return summary_rows


def compute_video_level_recall(
    audit_rows: List[Dict[str, Any]],
    family_sources: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for vid in TRAIN_VIDEO_IDS:
        for family in family_sources.keys():
            family_video_rows = [
                row for row in audit_rows
                if int(row["video_id"]) == vid and row["candidate_family"] == family
            ]
            for boundary_type in ["start", "end", "all"]:
                type_rows = family_video_rows if boundary_type == "all" else [row for row in family_video_rows if row["boundary_type"] == boundary_type]
                for tolerance in TOLERANCES:
                    hit_col = f"within_{tolerance}s"
                    hit_count = sum(1 for row in type_rows if row.get(hit_col) is True)
                    count = len(type_rows)
                    rows.append(
                        {
                            "video_id": vid,
                            "candidate_family": family,
                            "boundary_type": boundary_type,
                            "tolerance_sec": tolerance,
                            "actual_boundary_count": count,
                            "hit_count": hit_count,
                            "recall": (hit_count / count) if count else None,
                            "candidate_count": family_sources[family]["candidate_count_by_video"].get(vid, 0),
                            "notes": "train-only video-level recall",
                        }
                    )
    return rows


def audit_lookup(audit_rows: List[Dict[str, Any]]) -> Dict[Tuple[int, str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[int, str, str, str], Dict[str, Any]] = {}
    for row in audit_rows:
        key = (
            int(row["video_id"]),
            str(row["ad_interval_id"]),
            str(row["boundary_type"]),
            str(row["candidate_family"]),
        )
        out[key] = row
    return out


def compute_case_breakdown(audit_rows: List[Dict[str, Any]], boundaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup = audit_lookup(audit_rows)
    rows: List[Dict[str, Any]] = []
    for boundary in boundaries:
        vid = int(boundary["video_id"])
        interval = str(boundary["ad_interval_id"])
        btype = str(boundary["boundary_type"])
        existing = lookup.get((vid, interval, btype, "existing_combined_two_source"), {})
        transnet = lookup.get((vid, interval, btype, "transnetv2_primary"), {})
        combined = lookup.get((vid, interval, btype, "combined_existing_plus_transnetv2"), {})
        for tolerance in TOLERANCES:
            hit_col = f"within_{tolerance}s"
            existing_hit = bool(existing.get(hit_col))
            transnet_hit = bool(transnet.get(hit_col))
            combined_hit = bool(combined.get(hit_col))
            if existing_hit and transnet_hit:
                case_type = "existing_and_transnetv2_hit"
            elif existing_hit:
                case_type = "existing_only_hit"
            elif transnet_hit:
                case_type = "transnetv2_only_hit"
            else:
                case_type = "both_missed"
            rows.append(
                {
                    "tolerance_sec": tolerance,
                    "video_id": vid,
                    "ad_interval_id": interval,
                    "boundary_type": btype,
                    "actual_sec": boundary["actual_sec"],
                    "existing_combined_hit": existing_hit,
                    "transnetv2_hit": transnet_hit,
                    "existing_plus_transnetv2_hit": combined_hit,
                    "existing_combined_nearest_candidate_sec": existing.get("nearest_candidate_sec"),
                    "existing_combined_nearest_distance_sec": existing.get("nearest_distance_sec"),
                    "transnetv2_nearest_candidate_sec": transnet.get("nearest_candidate_sec"),
                    "transnetv2_nearest_distance_sec": transnet.get("nearest_distance_sec"),
                    "case_type": case_type,
                }
            )
    return rows


def compute_recovered_cases(audit_rows: List[Dict[str, Any]], warnings: List[str]) -> List[Dict[str, Any]]:
    if not PREVIOUS_AUDIT_CASE_PATH.exists():
        warnings.append(f"missing previous opencv/resnet case breakdown: {PREVIOUS_AUDIT_CASE_PATH}")
        return []
    lookup = audit_lookup(audit_rows)
    rows: List[Dict[str, Any]] = []
    for prev in read_csv_rows(PREVIOUS_AUDIT_CASE_PATH):
        if prev.get("case_type") != "both_missed":
            continue
        vid = clean_video_id(prev.get("video_id"))
        if vid not in TRAIN_VIDEO_IDS:
            continue
        tolerance = safe_int(prev.get("tolerance_sec"))
        if tolerance not in TOLERANCES:
            continue
        interval = str(prev.get("ad_interval_id"))
        btype = str(prev.get("boundary_type"))
        transnet = lookup.get((int(vid), interval, btype, "transnetv2_primary"), {})
        distance = safe_float(transnet.get("nearest_distance_sec"))
        recovered = distance is not None and distance <= tolerance
        rows.append(
            {
                "tolerance_sec": tolerance,
                "video_id": vid,
                "ad_interval_id": interval,
                "boundary_type": btype,
                "actual_sec": safe_float(prev.get("actual_sec")),
                "previous_case_type": prev.get("case_type"),
                "transnetv2_recovered": recovered,
                "transnetv2_nearest_candidate_sec": transnet.get("nearest_candidate_sec"),
                "transnetv2_nearest_distance_sec": transnet.get("nearest_distance_sec"),
                "previous_opencv_nearest_candidate_sec": prev.get("opencv_nearest_candidate_sec"),
                "previous_opencv_nearest_distance_sec": prev.get("opencv_nearest_distance_sec"),
                "previous_resnet_nearest_candidate_sec": prev.get("resnet_nearest_candidate_sec"),
                "previous_resnet_nearest_distance_sec": prev.get("resnet_nearest_distance_sec"),
                "notes": "previous both_missed from opencv_resnet audit; TransNetV2 primary checked at same tolerance",
            }
        )
    return rows


def build_source_comparison_rows(
    opencv_rows: List[Dict[str, Any]],
    resnet_rows: List[Dict[str, Any]],
    canonical_rows: List[Dict[str, Any]],
    transnet_rows: List[Dict[str, Any]],
    existing_clusters: List[Dict[str, Any]],
    existing_plus_transnet_clusters: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source_rows in [opencv_rows, resnet_rows, canonical_rows]:
        for row in source_rows:
            rows.append(
                {
                    "candidate_family": row.get("candidate_family"),
                    "video_id": row.get("video_id"),
                    "candidate_sec": row.get("candidate_sec"),
                    "candidate_frame": "",
                    "score": row.get("score"),
                    "threshold": row.get("threshold"),
                    "source_relation": row.get("source_relation"),
                    "source_output_path": row.get("source_output_path"),
                    "notes": row.get("notes"),
                }
            )
    for row in transnet_rows:
        rows.append(
            {
                "candidate_family": row.get("candidate_family"),
                "video_id": row.get("video_id"),
                "candidate_sec": row.get("candidate_sec"),
                "candidate_frame": row.get("candidate_frame"),
                "score": row.get("transnetv2_score"),
                "threshold": row.get("threshold"),
                "source_relation": "transnetv2_scene_end_frame",
                "source_output_path": row.get("source_output_path"),
                "notes": row.get("notes"),
            }
        )
    for clusters in [existing_clusters, existing_plus_transnet_clusters]:
        for row in clusters:
            rows.append(
                {
                    "candidate_family": row.get("candidate_family"),
                    "video_id": row.get("video_id"),
                    "candidate_sec": row.get("candidate_sec"),
                    "candidate_frame": "",
                    "score": "",
                    "threshold": "",
                    "source_relation": row.get("source_relation"),
                    "source_output_path": "",
                    "notes": row.get("notes"),
                }
            )
    return rows


def nested_summary(summary_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in summary_rows:
        family = str(row["candidate_family"])
        btype = str(row["boundary_type"])
        tol = str(row["tolerance_sec"])
        out.setdefault(family, {}).setdefault(btype, {})[tol] = {
            key: row.get(key)
            for key in [
                "actual_boundary_count",
                "hit_count",
                "recall",
                "median_nearest_distance_sec",
                "mean_nearest_distance_sec",
                "p90_nearest_distance_sec",
                "max_nearest_distance_sec",
                "candidate_count_total",
                "candidate_video_count",
            ]
        }
    return out


def case_counts(case_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for row in case_rows:
        tol = str(row["tolerance_sec"])
        case_type = str(row["case_type"])
        out.setdefault(tol, {})
        out[tol][case_type] = out[tol].get(case_type, 0) + 1
    return out


def density_by_family(
    family_sources: Dict[str, Dict[str, Any]],
    mapping_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    duration_by_video = {
        int(row["video_id"]): safe_float(row.get("duration_sec"))
        for row in mapping_rows
        if safe_float(row.get("duration_sec")) is not None
    }
    total_minutes = sum(float(x) for x in duration_by_video.values()) / 60.0 if duration_by_video else None
    out: Dict[str, Dict[str, Any]] = {}
    for family, data in family_sources.items():
        counts = data["candidate_count_by_video"]
        candidate_total = int(data["candidate_count_total"])
        video_count = len(counts)
        out[family] = {
            "candidate_count_total": candidate_total,
            "candidate_video_count": video_count,
            "candidates_per_video": candidate_total / video_count if video_count else None,
            "candidates_per_minute": candidate_total / total_minutes if total_minutes else None,
        }
    return out


def get_metric(summary: Dict[str, Dict[str, Dict[str, Any]]], family: str, btype: str, tol: int, key: str = "recall") -> Any:
    return summary.get(family, {}).get(btype, {}).get(str(tol), {}).get(key)


def recall_line(summary: Dict[str, Dict[str, Dict[str, Any]]], family: str, btype: str) -> str:
    parts = []
    for tol in TOLERANCES:
        recall = get_metric(summary, family, btype, tol, "recall")
        hit = get_metric(summary, family, btype, tol, "hit_count")
        count = get_metric(summary, family, btype, tol, "actual_boundary_count")
        parts.append(f"@{tol}s {format_float(recall, 4)} ({hit}/{count})")
    return " / ".join(parts)


def markdown_recall_table(summary: Dict[str, Dict[str, Dict[str, Any]]], families: Sequence[str]) -> str:
    lines = ["| family | boundary_type | recall@2s | recall@5s | recall@10s |", "|---|---:|---:|---:|---:|"]
    for family in families:
        if family not in summary:
            continue
        for btype in ["start", "end", "all"]:
            cells = []
            for tol in TOLERANCES:
                recall = get_metric(summary, family, btype, tol, "recall")
                hit = get_metric(summary, family, btype, tol, "hit_count")
                count = get_metric(summary, family, btype, tol, "actual_boundary_count")
                cells.append(f"{format_float(recall, 4)} ({hit}/{count})")
            lines.append(f"| {family} | {btype} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(lines)


def rows_to_brief(rows: List[Dict[str, Any]], max_rows: int = 30) -> str:
    if not rows:
        return "- 없음"
    lines = []
    for row in rows[:max_rows]:
        pieces = [
            f"tol={row.get('tolerance_sec')}",
            f"video={row.get('video_id')}",
            f"interval={row.get('ad_interval_id')}",
            f"type={row.get('boundary_type')}",
            f"actual={format_float(safe_float(row.get('actual_sec')), 3)}",
        ]
        if "transnetv2_nearest_distance_sec" in row:
            pieces.append(f"tn_dist={format_float(safe_float(row.get('transnetv2_nearest_distance_sec')), 3)}")
        lines.append("- " + ", ".join(pieces))
    if len(rows) > max_rows:
        lines.append(f"- ... {len(rows) - max_rows}개 추가")
    return "\n".join(lines)


def run_validations(
    split_validation: Dict[str, Any],
    mapping_rows: List[Dict[str, Any]],
    transnet_rows: List[Dict[str, Any]],
    raw_index_rows: List[Dict[str, Any]],
    boundaries: List[Dict[str, Any]],
    audit_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    recovered_rows: List[Dict[str, Any]],
    report_summary: Dict[str, Dict[str, Dict[str, Any]]],
    protected_modified: List[str],
    latest_files: List[Path],
    warnings: List[str],
) -> Dict[str, Dict[str, Any]]:
    train_set = set(TRAIN_VIDEO_IDS)
    validation_test_set = set(VALIDATION_VIDEO_IDS + TEST_VIDEO_IDS)
    transnet_video_ids = {int(row["video_id"]) for row in transnet_rows if clean_video_id(row.get("video_id")) is not None}
    raw_index_video_ids = {int(row["video_id"]) for row in raw_index_rows if clean_video_id(row.get("video_id")) is not None}
    boundary_video_ids = {int(row["video_id"]) for row in boundaries}
    audit_video_ids = {int(row["video_id"]) for row in audit_rows}
    forbidden_latest_suffixes = {".mp4", ".mov", ".mkv", ".avi", ".jpg", ".jpeg", ".png", ".pth", ".pt", ".ckpt"}

    distance_ok = True
    tolerance_ok = True
    for row in audit_rows:
        actual = safe_float(row.get("actual_sec"))
        nearest = safe_float(row.get("nearest_candidate_sec"))
        distance = safe_float(row.get("nearest_distance_sec"))
        if nearest is not None and actual is not None and distance is not None:
            if abs(abs(nearest - actual) - distance) > 1e-6:
                distance_ok = False
        for tol in TOLERANCES:
            flag = row.get(f"within_{tol}s")
            expected = distance is not None and distance <= tol
            if bool(flag) != expected:
                tolerance_ok = False

    expected_both_missed_count = 0
    if PREVIOUS_AUDIT_CASE_PATH.exists():
        for row in read_csv_rows(PREVIOUS_AUDIT_CASE_PATH):
            vid = clean_video_id(row.get("video_id"))
            if vid in train_set and row.get("case_type") == "both_missed" and safe_int(row.get("tolerance_sec")) in TOLERANCES:
                expected_both_missed_count += 1

    latest_safe = all(path.suffix.lower() not in forbidden_latest_suffixes for path in latest_files)
    latest_safe = latest_safe and not any(
        "transnetv2_raw_outputs_v2_4_train" in Path(path).parts or "transnetv2-pytorch-weights" in str(path)
        for path in latest_files
    )

    return {
        "input_split_validation": {
            "status": "PASS" if split_validation.get("fixed_train_match") and split_validation.get("fixed_validation_match") and split_validation.get("fixed_test_match") and not (transnet_video_ids & validation_test_set) else "FAIL",
            "details": split_validation,
            "processed_video_ids": sorted(transnet_video_ids),
            "validation_test_video_ids_in_outputs": sorted((transnet_video_ids | raw_index_video_ids | audit_video_ids) & validation_test_set),
        },
        "video_mapping_validation": {
            "status": "PASS" if all(row.get("mapping_source") != "not_found" for row in mapping_rows) else "WARN",
            "mapped_video_count": len([row for row in mapping_rows if row.get("video_path")]),
            "existing_video_count": len([row for row in mapping_rows if row.get("file_exists")]),
            "missing_video_ids": [row["video_id"] for row in mapping_rows if not row.get("file_exists")],
            "mapping_sources": sorted({str(row.get("mapping_source")) for row in mapping_rows}),
        },
        "transnetv2_inference_validation": {
            "status": "PASS" if transnet_video_ids <= train_set and len(transnet_rows) > 0 else "FAIL",
            "package_path": str(TRANSNET_PYTHONPATH),
            "weight_path": str(TRANSNET_WEIGHT_PATH),
            "train_only_inference": transnet_video_ids <= train_set,
            "thresholds_present": sorted({safe_float(row.get("threshold")) for row in transnet_rows if safe_float(row.get("threshold")) is not None}),
            "frame_to_second_rule": "candidate_sec = candidate_frame / fps",
            "device_rows": sorted({str(row.get("device_used")) for row in transnet_rows}),
        },
        "boundary_recall_audit_validation": {
            "status": "PASS" if boundary_video_ids <= train_set and distance_ok and tolerance_ok else "FAIL",
            "actual_boundary_count": len(boundaries),
            "start_count": len([b for b in boundaries if b["boundary_type"] == "start"]),
            "end_count": len([b for b in boundaries if b["boundary_type"] == "end"]),
            "actual_label_used_for_candidate_generation": False,
            "actual_label_used_for_recall_audit_only": True,
            "nearest_distance_formula_ok": distance_ok,
            "tolerance_flags_ok": tolerance_ok,
        },
        "existing_missed_recovery_validation": {
            "status": "PASS" if len(recovered_rows) == expected_both_missed_count else "WARN",
            "previous_both_missed_rows_expected": expected_both_missed_count,
            "recovered_case_rows": len(recovered_rows),
            "report_csv_recall_summary_consistent": bool(report_summary) and len(summary_rows) > 0,
        },
        "output_safety_validation": {
            "status": "PASS" if not protected_modified and latest_safe and not (audit_video_ids & validation_test_set) else "FAIL",
            "protected_files_modified": protected_modified,
            "latest_bundle_file_count": len(latest_files),
            "latest_bundle_excludes_weight_raw_video_frame_cache": latest_safe,
            "validation_test_row_level_output_generated": bool(audit_video_ids & validation_test_set),
            "warnings": warnings,
        },
    }


def generate_reports(
    setup_report: Dict[str, Any],
    split_validation: Dict[str, Any],
    mapping_rows: List[Dict[str, Any]],
    boundaries: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    case_rows: List[Dict[str, Any]],
    recovered_rows: List[Dict[str, Any]],
    family_sources: Dict[str, Dict[str, Any]],
    density: Dict[str, Dict[str, Any]],
    inference_meta: Dict[str, Any],
    warnings: List[str],
    errors: List[str],
    protected_modified: List[str],
    validations: Dict[str, Dict[str, Any]],
    output_row_counts: Dict[str, int],
    start_time_iso: str,
    end_time_iso: str,
) -> Dict[str, Any]:
    summary = nested_summary(summary_rows)
    processed_video_count = len([row for row in mapping_rows if row.get("file_exists")])
    missing_mapping = [row for row in mapping_rows if not row.get("file_exists")]
    transnet_only_rows = [row for row in case_rows if row.get("case_type") == "transnetv2_only_hit"]
    both_missed_rows = [row for row in case_rows if row.get("case_type") == "both_missed"]
    recovered_true_rows = [row for row in recovered_rows if str(row.get("transnetv2_recovered")).lower() == "true" or row.get("transnetv2_recovered") is True]

    existing_cpm = density.get("existing_combined_two_source", {}).get("candidates_per_minute")
    transnet_cpm = density.get("transnetv2_primary", {}).get("candidates_per_minute")
    integration_value = "undetermined"
    if get_metric(summary, "combined_existing_plus_transnetv2", "all", 2, "recall") is not None:
        old_r2 = float(get_metric(summary, "existing_combined_two_source", "all", 2, "recall") or 0)
        new_r2 = float(get_metric(summary, "combined_existing_plus_transnetv2", "all", 2, "recall") or 0)
        old_r5 = float(get_metric(summary, "existing_combined_two_source", "all", 5, "recall") or 0)
        new_r5 = float(get_metric(summary, "combined_existing_plus_transnetv2", "all", 5, "recall") or 0)
        density_ratio = (transnet_cpm / existing_cpm) if existing_cpm and transnet_cpm is not None else None
        if (new_r2 > old_r2 or new_r5 > old_r5) and (density_ratio is None or density_ratio <= 2.5):
            integration_value = "worth_follow_up"
        elif new_r2 > old_r2 or new_r5 > old_r5:
            integration_value = "recall_gain_but_density_risk"
        else:
            integration_value = "limited_value_for_primary_anchor"

    report = {
        "task_name": "transnetv2_scene_candidate_audit_v2_4_train",
        "project_root": str(PROJECT_ROOT),
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "setup_status": setup_report.get("setup_status"),
        "transnetv2_package_name": setup_report.get("transnetv2_package_name", "transnetv2-pytorch"),
        "transnetv2_package_version": setup_report.get("transnetv2_package_version"),
        "transnetv2_package_path": str(TRANSNET_PYTHONPATH),
        "transnetv2_weight_path": str(TRANSNET_WEIGHT_PATH),
        "transnetv2_weight_matches_setup_report": str(setup_report.get("weight_path")) == str(TRANSNET_WEIGHT_PATH),
        "input_files": {path.name: str(path) for path in INPUT_FILES},
        "output_files": {
            "script": str(SCRIPT_PATH),
            "video_mapping": str(VIDEO_MAPPING_CSV),
            "transnet_candidates": str(TRANSNET_CANDIDATES_CSV),
            "raw_outputs_index": str(RAW_INDEX_CSV),
            "source_comparison": str(SOURCE_COMPARISON_CSV),
            "recall_audit": str(RECALL_AUDIT_CSV),
            "recall_summary": str(RECALL_SUMMARY_CSV),
            "case_breakdown": str(CASE_BREAKDOWN_CSV),
            "recovered_cases": str(RECOVERED_CASES_CSV),
            "video_level_recall": str(VIDEO_LEVEL_RECALL_CSV),
            "summary_md": str(SUMMARY_MD),
            "report_json": str(REPORT_JSON),
            "findings_md": str(FINDINGS_MD),
            "log": str(LOG_PATH),
            "latest_bundle": str(LATEST_BUNDLE_DIR),
            "shared_latest_for_chatgpt": str(SHARED_LATEST_DIR),
        },
        "split_validation": split_validation,
        "train_video_ids": TRAIN_VIDEO_IDS,
        "validation_video_ids_excluded": VALIDATION_VIDEO_IDS,
        "test_video_ids_excluded": TEST_VIDEO_IDS,
        "train_video_path_mapping": {
            "total_train_video_count": len(TRAIN_VIDEO_IDS),
            "processed_video_count": processed_video_count,
            "missing_video_count": len(missing_mapping),
            "missing_video_ids": [row["video_id"] for row in missing_mapping],
            "mapping_sources": sorted({str(row.get("mapping_source")) for row in mapping_rows}),
        },
        "counts": {
            "train_actual_ad_interval_count": len({(b["video_id"], b["ad_interval_id"]) for b in boundaries}),
            "train_actual_boundary_count": len(boundaries),
            "start_boundary_count": len([b for b in boundaries if b["boundary_type"] == "start"]),
            "end_boundary_count": len([b for b in boundaries if b["boundary_type"] == "end"]),
            "transnetv2_primary_candidate_count": family_sources.get("transnetv2_primary", {}).get("candidate_count_total", 0),
            "opencv_ffmpeg_candidate_count": family_sources.get("opencv_ffmpeg", {}).get("candidate_count_total", 0),
            "resnet_candidate_count": family_sources.get("resnet", {}).get("candidate_count_total", 0),
            "existing_combined_candidate_count": family_sources.get("existing_combined_two_source", {}).get("candidate_count_total", 0),
            "existing_plus_transnetv2_candidate_count": family_sources.get("combined_existing_plus_transnetv2", {}).get("candidate_count_total", 0),
            "canonical_all_candidate_count": family_sources.get("canonical_all", {}).get("candidate_count_total", 0),
        },
        "threshold_policy": {
            "primary_threshold": PRIMARY_THRESHOLD,
            "optional_sensitivity_thresholds": [0.3, 0.7],
            "threshold_tuning_used": False,
            "actual_label_used_for_threshold_selection": False,
        },
        "candidate_timestamp_rule": {
            "transnetv2": "scene end_frame for every detected scene segment except the final tail segment; candidate_sec=end_frame/fps",
            "combined_existing_plus_transnetv2": "2s cluster for counting; nearest-distance audit uses original member timestamps",
            "dedup_window_sec": DEDUP_WINDOW_SEC,
        },
        "inference": inference_meta,
        "summary_metrics": summary,
        "case_counts": case_counts(case_rows),
        "existing_both_missed_recovered_counts": {
            str(tol): {
                "previous_both_missed": len([r for r in recovered_rows if safe_int(r.get("tolerance_sec")) == tol]),
                "transnetv2_recovered": len([r for r in recovered_rows if safe_int(r.get("tolerance_sec")) == tol and (r.get("transnetv2_recovered") is True or str(r.get("transnetv2_recovered")).lower() == "true")]),
            }
            for tol in TOLERANCES
        },
        "candidate_density": density,
        "candidate_density_judgment": {
            "existing_candidates_per_minute": existing_cpm,
            "transnetv2_candidates_per_minute": transnet_cpm,
            "judgment": "density is acceptable for follow-up" if transnet_cpm is not None and existing_cpm is not None and transnet_cpm <= existing_cpm * 2.5 else "density needs review before canonical integration",
        },
        "canonical_integration_first_judgment": integration_value,
        "future_detector_work_needed": [
            "decide whether TransNetV2 candidate density needs score/cluster filtering before detector use",
            "define canonical visual anchor merge policy if TransNetV2 is adopted",
            "rerun train-only feature construction without using actual labels for candidate generation",
            "only after train policy freeze, evaluate validation/test without row-level leakage",
        ],
        "warnings": warnings,
        "errors": errors,
        "protected_files_modified": protected_modified,
        "no_detector_rule_modified": True,
        "no_validation_test_row_level_output": True,
        "actual_label_used_for_candidate_generation": False,
        "actual_label_used_for_recall_audit_only": True,
        "canonical_visual_anchor_replaced": False,
        "validation_test_evaluation_performed": False,
        "sub_agent_validations": validations,
        "output_row_counts": output_row_counts,
    }

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    families_for_report = [
        "transnetv2_primary",
        "existing_combined_two_source",
        "combined_existing_plus_transnetv2",
        "opencv_ffmpeg",
        "resnet",
        "canonical_all",
    ]
    recovered_by_tol = report["existing_both_missed_recovered_counts"]
    summary_md = f"""# TransNetV2 Scene Candidate Audit v2.4 Train

## 목적
TransNetV2로 train 영상의 shot transition 후보를 추출하고, 기존 OpenCV/FFmpeg + ResNet 후보가 실제 광고 시작/종료 경계를 놓친 지점을 보완하는지 train-only로 점검했다. Detector rule, canonical visual anchor, actual label, split 파일은 수정하지 않았다.

## Setup
- package: `{report['transnetv2_package_name']}=={report['transnetv2_package_version']}`
- package path: `{TRANSNET_PYTHONPATH}`
- weight path: `{TRANSNET_WEIGHT_PATH}`
- setup status: `{report['setup_status']}`
- initial device: `{inference_meta.get('initial_device')}`, final device: `{inference_meta.get('final_device')}`, CUDA fallback: `{inference_meta.get('cuda_to_cpu_fallback_used')}`

## Input/Split
- train video_id: `{TRAIN_VIDEO_IDS}`
- validation excluded: `{VALIDATION_VIDEO_IDS}`
- test excluded: `{TEST_VIDEO_IDS}`
- processed train videos: `{processed_video_count}/{len(TRAIN_VIDEO_IDS)}`
- missing train videos: `{len(missing_mapping)}`
- actual ad intervals: `{report['counts']['train_actual_ad_interval_count']}`
- actual boundaries: `{len(boundaries)}` (start `{report['counts']['start_boundary_count']}`, end `{report['counts']['end_boundary_count']}`)

## Candidate Counts
| family | count | videos | candidates/video | candidates/min |
|---|---:|---:|---:|---:|
"""
    for family in families_for_report:
        if family not in density:
            continue
        d = density[family]
        summary_md += f"| {family} | {d.get('candidate_count_total')} | {d.get('candidate_video_count')} | {format_float(safe_float(d.get('candidates_per_video')), 3)} | {format_float(safe_float(d.get('candidates_per_minute')), 3)} |\n"
    summary_md += f"""
## Recall Summary
{markdown_recall_table(summary, families_for_report)}

## Existing Both-Missed Recovery
| tolerance | previous both_missed | recovered by TransNetV2 |
|---:|---:|---:|
"""
    for tol in TOLERANCES:
        item = recovered_by_tol[str(tol)]
        summary_md += f"| {tol}s | {item['previous_both_missed']} | {item['transnetv2_recovered']} |\n"
    summary_md += f"""
## Key Case Lists
### TransNetV2 Only Hit
{rows_to_brief(transnet_only_rows)}

### Existing Both-Missed Recovered
{rows_to_brief(recovered_true_rows)}

### Existing + TransNetV2 Both Missed
{rows_to_brief(both_missed_rows)}

## Safety Flags
- no_detector_rule_modified=true
- no_validation_test_row_level_output=true
- actual_label_used_for_candidate_generation=false
- actual_label_used_for_recall_audit_only=true
- latest bundle excludes raw video/frame/cache/model weight/package directory=true
"""
    SUMMARY_MD.write_text(summary_md, encoding="utf-8")

    old_all_2 = get_metric(summary, "existing_combined_two_source", "all", 2, "recall")
    new_all_2 = get_metric(summary, "combined_existing_plus_transnetv2", "all", 2, "recall")
    old_all_5 = get_metric(summary, "existing_combined_two_source", "all", 5, "recall")
    new_all_5 = get_metric(summary, "combined_existing_plus_transnetv2", "all", 5, "recall")
    old_all_10 = get_metric(summary, "existing_combined_two_source", "all", 10, "recall")
    new_all_10 = get_metric(summary, "combined_existing_plus_transnetv2", "all", 10, "recall")
    findings_md = f"""# TransNetV2 Scene Candidate Audit v2.4 Findings

## 해석 요약
TransNetV2 primary threshold 0.5는 label tuning 없이 고정값으로 사용했다. 후보 생성 단계에서는 actual label을 전혀 읽지 않고, 모든 후보 추출이 끝난 뒤 train actual ad start/end boundary에 대해서만 nearest-distance recall audit을 수행했다.

## Recall 변화
- 기존 combined all recall: @2s `{format_float(safe_float(old_all_2), 4)}`, @5s `{format_float(safe_float(old_all_5), 4)}`, @10s `{format_float(safe_float(old_all_10), 4)}`
- existing + TransNetV2 all recall: @2s `{format_float(safe_float(new_all_2), 4)}`, @5s `{format_float(safe_float(new_all_5), 4)}`, @10s `{format_float(safe_float(new_all_10), 4)}`
- TransNetV2 start: {recall_line(summary, 'transnetv2_primary', 'start')}
- TransNetV2 end: {recall_line(summary, 'transnetv2_primary', 'end')}

## Existing Missed Recovery
기존 OpenCV/FFmpeg + ResNet audit에서 both_missed였던 boundary를 tolerance별로 다시 확인했다.
{rows_to_brief(recovered_true_rows, max_rows=50)}

## TransNetV2도 못 잡은 Boundary
{rows_to_brief(both_missed_rows, max_rows=50)}

## Candidate Density
- 기존 combined candidates/min: `{format_float(safe_float(existing_cpm), 3)}`
- TransNetV2 primary candidates/min: `{format_float(safe_float(transnet_cpm), 3)}`
- 판단: `{report['candidate_density_judgment']['judgment']}`

## Canonical 통합 1차 판단
`{integration_value}`. 이 판단은 train-only recall/density audit의 1차 결론이며, canonical visual anchor 교체나 detector rule 반영은 수행하지 않았다. 실제 반영 전에는 merge policy, 중복 cluster 정책, candidate density 제어 기준을 별도로 정해야 한다.
"""
    FINDINGS_MD.write_text(findings_md, encoding="utf-8")
    return report


def update_latest_bundles(files: Sequence[Path], logger: Logger) -> Tuple[List[Path], List[Path]]:
    LATEST_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_LATEST_DIR.mkdir(parents=True, exist_ok=True)
    copied_bundle: List[Path] = []
    copied_shared: List[Path] = []

    for source in files:
        if not source.exists():
            continue
        bundle_dest = LATEST_BUNDLE_DIR / source.name
        shared_dest = SHARED_LATEST_DIR / source.name
        shutil.copy2(source, bundle_dest)
        shutil.copy2(source, shared_dest)
        copied_bundle.append(bundle_dest)
        copied_shared.append(shared_dest)

    readme = LATEST_BUNDLE_DIR / "README_latest_files.md"
    lines = [
        "# Latest Files: TransNetV2 Scene Candidate Audit v2.4",
        "",
        "Included files are copies of newly generated small CSV/report/script/log artifacts.",
        "Excluded: raw videos, frame images, cache directories, model weights/checkpoints, package directories, raw per-video TransNetV2 JSON outputs, validation/test row-level outputs.",
        "",
        "## Files",
    ]
    for path in copied_bundle:
        lines.append(f"- `{path.name}`")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shared_readme = SHARED_LATEST_DIR / "README_transnetv2_scene_candidate_audit_v2_4_latest_files.md"
    shutil.copy2(readme, shared_readme)
    copied_bundle.append(readme)
    copied_shared.append(shared_readme)
    logger.write(f"Latest bundle updated: {LATEST_BUNDLE_DIR}")
    logger.write(f"Shared latest_for_chatgpt updated: {SHARED_LATEST_DIR}")
    return copied_bundle, copied_shared


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract TransNetV2 scene candidates and audit train ad boundary recall.")
    parser.add_argument("--skip-inference", action="store_true", help="Reuse existing raw outputs if available; intended only for debugging.")
    args = parser.parse_args()

    start_time = time.time()
    start_time_iso = now_iso()
    logger = Logger(LOG_PATH)
    warnings: List[str] = []
    errors: List[str] = []
    report: Dict[str, Any] = {}

    try:
        logger.step(1, "Safety snapshot and output path preparation")
        DATA_SCENE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_SCENE_DIR.mkdir(parents=True, exist_ok=True)
        RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LATEST_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        protected_snapshot_before = snapshot_paths(INPUT_FILES)

        logger.step(2, "Load TransNetV2 setup report and verify package/weight")
        setup_report = load_json(SETUP_REPORT_PATH)
        if not setup_report:
            warnings.append(f"setup report missing or empty: {SETUP_REPORT_PATH}")
        if not TRANSNET_WEIGHT_PATH.exists():
            errors.append(f"TransNetV2 weight missing: {TRANSNET_WEIGHT_PATH}")
        logger.write(f"Setup status: {setup_report.get('setup_status')}; weight exists={TRANSNET_WEIGHT_PATH.exists()}")

        logger.step(3, "Load split and identify train videos")
        split_rows, split_by_video, split_validation = load_split(logger)

        logger.step(4, "Map train video_id to video files")
        manifest_by_video = load_manifest()
        segment_rows = load_segment_rows()
        mapping_rows, mapping_warnings = map_train_videos(split_by_video, manifest_by_video, segment_rows)
        warnings.extend(mapping_warnings)
        write_csv(
            VIDEO_MAPPING_CSV,
            [
                "video_id",
                "split",
                "video_title",
                "video_path",
                "file_exists",
                "duration_sec",
                "fps",
                "frame_count",
                "mapping_source",
                "notes",
            ],
            mapping_rows,
        )
        logger.write(f"Mapped train videos: {len([r for r in mapping_rows if r.get('file_exists')])}/{len(TRAIN_VIDEO_IDS)}")

        logger.step(5, "Run TransNetV2 inference for train videos only")
        if args.skip_inference:
            raise RuntimeError("--skip-inference is not implemented for the official audit run")
        transnet_rows, raw_index_rows, inference_meta = infer_transnetv2(mapping_rows, logger, warnings, errors)

        logger.step(6, "Parse TransNetV2 outputs into candidate seconds")
        write_csv(
            TRANSNET_CANDIDATES_CSV,
            [
                "candidate_family",
                "video_id",
                "split",
                "candidate_sec",
                "candidate_frame",
                "transnetv2_score",
                "threshold",
                "source_output_path",
                "device_used",
                "fps",
                "duration_sec",
                "raw_row_index",
                "notes",
            ],
            transnet_rows,
        )
        write_csv(
            RAW_INDEX_CSV,
            [
                "video_id",
                "video_path",
                "raw_output_path",
                "output_format",
                "threshold",
                "row_count",
                "device_used",
                "runtime_seconds",
                "status",
                "error_message",
            ],
            raw_index_rows,
        )
        logger.write(f"TransNetV2 candidate rows written: {len(transnet_rows)}")

        logger.step(7, "Load existing OpenCV/FFmpeg, ResNet, and combined candidates")
        opencv_rows, resnet_rows, canonical_rows, existing_warnings = load_existing_candidates(TRAIN_VIDEO_IDS)
        warnings.extend(existing_warnings)
        existing_clusters = build_existing_combined_clusters(opencv_rows, resnet_rows)
        transnet_primary_rows = [row for row in transnet_rows if row.get("candidate_family") == "transnetv2_primary"]
        existing_plus_transnet_clusters = build_existing_plus_transnet_clusters(existing_clusters, transnet_primary_rows)
        source_comparison_rows = build_source_comparison_rows(
            opencv_rows,
            resnet_rows,
            canonical_rows,
            transnet_rows,
            existing_clusters,
            existing_plus_transnet_clusters,
        )
        write_csv(
            SOURCE_COMPARISON_CSV,
            [
                "candidate_family",
                "video_id",
                "candidate_sec",
                "candidate_frame",
                "score",
                "threshold",
                "source_relation",
                "source_output_path",
                "notes",
            ],
            source_comparison_rows,
        )
        logger.write(
            "Existing candidates loaded: "
            f"opencv={len(opencv_rows)}, resnet={len(resnet_rows)}, existing_combined={len(existing_clusters)}, canonical={len(canonical_rows)}"
        )

        logger.step(8, "Construct actual train ad start/end boundaries")
        boundaries, boundary_warnings = construct_actual_boundaries(segment_rows, TRAIN_VIDEO_IDS)
        warnings.extend(boundary_warnings)
        logger.write(f"Actual boundaries: {len(boundaries)}")

        logger.step(9, "Compute recall audit for TransNetV2 and combined candidates")
        family_sources = build_family_sources(
            opencv_rows,
            resnet_rows,
            canonical_rows,
            transnet_rows,
            existing_clusters,
            existing_plus_transnet_clusters,
        )
        audit_rows = compute_recall_audit(boundaries, family_sources)
        summary_rows = summarize_recall(audit_rows, family_sources)
        video_level_rows = compute_video_level_recall(audit_rows, family_sources)
        case_rows = compute_case_breakdown(audit_rows, boundaries)
        recovered_rows = compute_recovered_cases(audit_rows, warnings)

        logger.step(10, "Analyze recovery of existing both-missed boundaries")
        recovered_counts = {
            tol: len([row for row in recovered_rows if safe_int(row.get("tolerance_sec")) == tol and (row.get("transnetv2_recovered") is True or str(row.get("transnetv2_recovered")).lower() == "true")])
            for tol in TOLERANCES
        }
        logger.write(f"Recovered previous both_missed counts by tolerance: {recovered_counts}")

        logger.step(11, "Generate CSV outputs")
        write_csv(
            RECALL_AUDIT_CSV,
            [
                "video_id",
                "ad_interval_id",
                "boundary_type",
                "actual_sec",
                "candidate_family",
                "nearest_candidate_sec",
                "nearest_distance_sec",
                "within_2s",
                "within_5s",
                "within_10s",
                "candidate_count_in_video",
                "notes",
            ],
            audit_rows,
        )
        write_csv(
            RECALL_SUMMARY_CSV,
            [
                "candidate_family",
                "boundary_type",
                "tolerance_sec",
                "actual_boundary_count",
                "hit_count",
                "recall",
                "median_nearest_distance_sec",
                "mean_nearest_distance_sec",
                "p90_nearest_distance_sec",
                "max_nearest_distance_sec",
                "candidate_count_total",
                "candidate_video_count",
            ],
            summary_rows,
        )
        write_csv(
            CASE_BREAKDOWN_CSV,
            [
                "tolerance_sec",
                "video_id",
                "ad_interval_id",
                "boundary_type",
                "actual_sec",
                "existing_combined_hit",
                "transnetv2_hit",
                "existing_plus_transnetv2_hit",
                "existing_combined_nearest_candidate_sec",
                "existing_combined_nearest_distance_sec",
                "transnetv2_nearest_candidate_sec",
                "transnetv2_nearest_distance_sec",
                "case_type",
            ],
            case_rows,
        )
        write_csv(
            RECOVERED_CASES_CSV,
            [
                "tolerance_sec",
                "video_id",
                "ad_interval_id",
                "boundary_type",
                "actual_sec",
                "previous_case_type",
                "transnetv2_recovered",
                "transnetv2_nearest_candidate_sec",
                "transnetv2_nearest_distance_sec",
                "previous_opencv_nearest_candidate_sec",
                "previous_opencv_nearest_distance_sec",
                "previous_resnet_nearest_candidate_sec",
                "previous_resnet_nearest_distance_sec",
                "notes",
            ],
            recovered_rows,
        )
        write_csv(
            VIDEO_LEVEL_RECALL_CSV,
            [
                "video_id",
                "candidate_family",
                "boundary_type",
                "tolerance_sec",
                "actual_boundary_count",
                "hit_count",
                "recall",
                "candidate_count",
                "notes",
            ],
            video_level_rows,
        )

        output_row_counts = {
            str(VIDEO_MAPPING_CSV): len(mapping_rows),
            str(TRANSNET_CANDIDATES_CSV): len(transnet_rows),
            str(RAW_INDEX_CSV): len(raw_index_rows),
            str(SOURCE_COMPARISON_CSV): len(source_comparison_rows),
            str(RECALL_AUDIT_CSV): len(audit_rows),
            str(RECALL_SUMMARY_CSV): len(summary_rows),
            str(CASE_BREAKDOWN_CSV): len(case_rows),
            str(RECOVERED_CASES_CSV): len(recovered_rows),
            str(VIDEO_LEVEL_RECALL_CSV): len(video_level_rows),
        }

        logger.step(12, "Generate markdown/json reports and findings")
        density = density_by_family(family_sources, mapping_rows)
        protected_snapshot_after_outputs = snapshot_paths(INPUT_FILES)
        protected_modified = changed_paths(protected_snapshot_before, protected_snapshot_after_outputs)
        validations_placeholder: Dict[str, Dict[str, Any]] = {}
        report_summary = nested_summary(summary_rows)
        report = generate_reports(
            setup_report=setup_report,
            split_validation=split_validation,
            mapping_rows=mapping_rows,
            boundaries=boundaries,
            summary_rows=summary_rows,
            case_rows=case_rows,
            recovered_rows=recovered_rows,
            family_sources=family_sources,
            density=density,
            inference_meta=inference_meta,
            warnings=warnings,
            errors=errors,
            protected_modified=protected_modified,
            validations=validations_placeholder,
            output_row_counts=output_row_counts,
            start_time_iso=start_time_iso,
            end_time_iso=now_iso(),
        )

        logger.step(13, "Run Sub Agent validations")
        # 번들 복사 전에 검증하고, 최종 검증 결과를 반영해 report를 다시 쓴다.
        files_for_bundle = [
            SCRIPT_PATH,
            VIDEO_MAPPING_CSV,
            TRANSNET_CANDIDATES_CSV,
            RAW_INDEX_CSV,
            SOURCE_COMPARISON_CSV,
            RECALL_AUDIT_CSV,
            RECALL_SUMMARY_CSV,
            CASE_BREAKDOWN_CSV,
            RECOVERED_CASES_CSV,
            VIDEO_LEVEL_RECALL_CSV,
            SUMMARY_MD,
            REPORT_JSON,
            FINDINGS_MD,
            LOG_PATH,
        ]
        validations = run_validations(
            split_validation=split_validation,
            mapping_rows=mapping_rows,
            transnet_rows=transnet_rows,
            raw_index_rows=raw_index_rows,
            boundaries=boundaries,
            audit_rows=audit_rows,
            summary_rows=summary_rows,
            recovered_rows=recovered_rows,
            report_summary=report_summary,
            protected_modified=protected_modified,
            latest_files=files_for_bundle,
            warnings=warnings,
        )
        logger.write("Validation statuses: " + json.dumps({k: v.get("status") for k, v in validations.items()}, ensure_ascii=False, sort_keys=True))
        report = generate_reports(
            setup_report=setup_report,
            split_validation=split_validation,
            mapping_rows=mapping_rows,
            boundaries=boundaries,
            summary_rows=summary_rows,
            case_rows=case_rows,
            recovered_rows=recovered_rows,
            family_sources=family_sources,
            density=density,
            inference_meta=inference_meta,
            warnings=warnings,
            errors=errors,
            protected_modified=protected_modified,
            validations=validations,
            output_row_counts=output_row_counts,
            start_time_iso=start_time_iso,
            end_time_iso=now_iso(),
        )

        logger.step(14, "Update latest bundle")
        copied_bundle, copied_shared = update_latest_bundles(files_for_bundle, logger)
        # 번들 갱신 과정에서도 log가 추가되므로 최종 log/report를 한 번 더 복사한다.
        shutil.copy2(LOG_PATH, LATEST_BUNDLE_DIR / LOG_PATH.name)
        shutil.copy2(LOG_PATH, SHARED_LATEST_DIR / LOG_PATH.name)
        shutil.copy2(REPORT_JSON, LATEST_BUNDLE_DIR / REPORT_JSON.name)
        shutil.copy2(REPORT_JSON, SHARED_LATEST_DIR / REPORT_JSON.name)

        logger.step(15, "Print final human-readable summary")
        elapsed = time.time() - start_time
        summary = nested_summary(summary_rows)
        logger.write(f"Processed train videos: {report['train_video_path_mapping']['processed_video_count']}/{len(TRAIN_VIDEO_IDS)}")
        logger.write(f"TransNetV2 primary candidates: {report['counts']['transnetv2_primary_candidate_count']}")
        logger.write(f"TransNetV2 all recall: {recall_line(summary, 'transnetv2_primary', 'all')}")
        logger.write(f"Existing combined all recall: {recall_line(summary, 'existing_combined_two_source', 'all')}")
        logger.write(f"Existing + TransNetV2 all recall: {recall_line(summary, 'combined_existing_plus_transnetv2', 'all')}")
        logger.write(f"Canonical integration first judgment: {report['canonical_integration_first_judgment']}")
        logger.write(f"Output row counts: {json.dumps(output_row_counts, ensure_ascii=False, sort_keys=True)}")
        logger.write(f"Elapsed seconds: {elapsed:.2f}")
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        errors.append(str(exc))
        try:
            logger.write("ERROR: " + str(exc))
            logger.write(tb)
            partial_report = {
                "task_name": "transnetv2_scene_candidate_audit_v2_4_train",
                "project_root": str(PROJECT_ROOT),
                "start_time": start_time_iso,
                "end_time": now_iso(),
                "setup_status": load_json(SETUP_REPORT_PATH).get("setup_status"),
                "errors": errors,
                "warnings": warnings,
                "traceback": tb,
                "no_detector_rule_modified": True,
                "no_validation_test_row_level_output": True,
                "actual_label_used_for_candidate_generation": False,
                "actual_label_used_for_recall_audit_only": True,
                "latest_bundle_path": str(LATEST_BUNDLE_DIR),
            }
            REPORT_JSON.write_text(json.dumps(partial_report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
