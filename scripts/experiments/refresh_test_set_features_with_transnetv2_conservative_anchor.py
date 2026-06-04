#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import gc
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
TEST_IDS = [3, 4, 7, 16, 17, 18]
TEST_ID_SET = set(TEST_IDS)
DEVELOPMENT_IDS = {1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15}
TEST_SPLITS = {"validation", "test"}
TEST_NOTE = (
    "test set = Test Set; actual labels are "
    "not used for feature/candidate generation."
)

FEATURE_ROOT = ROOT / "data/features/manual_rule_lab_v2_3_test_set"
INPUT_ROOT = ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/input_features"
LATEST_DIR = ROOT / "notebooks/rule_lab_v2_3_modular/lastest_csv_test_set_features"
REPORT_DIR = ROOT / "reports/evaluation"
LOG_PATH = ROOT / "logs/test_set_transnetv2_anchor_refresh_run_log.txt"

SPLIT_PATH = ROOT / "data/splits/video_split_v2_4.csv"
MANIFEST_PATH = ROOT / "data/video_metadata/video_manifest_v2_2.csv"
OPENCV_PATH = ROOT / "data/scene/scene_candidates_v2_4_merged_ffmpeg_fallback_labelrefreshed.csv"
RESNET_PATH = ROOT / "data/review/resnet_scene_candidate_human_review_v2_4_usefulness_analysis.csv"
CANONICAL_PATH = ROOT / "data/features/visual_scene_boundary_anchors_v2_4_with_split.csv"
TRANSNET_BEST_PATH = ROOT / "data/scene/transnetv2_conservative_best_candidate_v2_4_train.csv"
TRANSNET_REPORT_PATH = ROOT / "reports/scene/transnetv2_conservative_sweep_v2_4_report.json"
TRANSNET_PYTHONPATH = ROOT / "external/transnetv2/python"
TRANSNET_WEIGHT_PATH = TRANSNET_PYTHONPATH / "transnetv2_pytorch/transnetv2-pytorch-weights.pth"

TRANSNET_FAMILY = "transnetv2_threshold_0_7_dedup_5"
TRANSNET_THRESHOLD = 0.7
TRANSNET_DEDUP_SEC = 5.0
CROSS_SOURCE_DEDUP_SEC = 2.0

FINAL_ANCHOR_PATH = FEATURE_ROOT / "final_scene_boundary_anchor_v2_5_test_set.csv"
TRANSNET_ANCHOR_PATH = FEATURE_ROOT / "transnetv2_conservative_scene_anchor_v2_5_test_set.csv"
OCR_SCHEDULE_PATH = FEATURE_ROOT / "final_scene_anchor_ocr_schedule_v2_5_test_set.csv"
OCR_ANCHOR_FEATURE_PATH = FEATURE_ROOT / "final_scene_anchor_ocr_anchor_window_features_v2_5_test_set.csv"
OCR_TIMELINE_FEATURE_PATH = FEATURE_ROOT / "final_scene_anchor_ocr_20s_timeline_features_v2_5_test_set.csv"
FUSION_PATH = FEATURE_ROOT / "final_scene_audio_ocr_rule_input_v2_0_test_set.csv"

PRED_PATH = INPUT_ROOT / "v2_1b_test_set_predictions.csv"
REVIEW_PATH = INPUT_ROOT / "v2_1b_test_set_review_candidates.csv"
PRUNED_PATH = INPUT_ROOT / "v2_1b_test_set_pruned_candidates.csv"
OPEN_PATH = INPUT_ROOT / "v2_1b_test_set_open_candidates.csv"
OCR_SOURCE_PATH = INPUT_ROOT / "v2_5_test_set_ocr_candidate_sources.csv"
BLACK_FEATURE_PATH = INPUT_ROOT / "v2_5_test_set_black_screen_features.csv"
MANIFEST_OUT_PATH = INPUT_ROOT / "test_set_input_feature_manifest.json"
READINESS_OUT_PATH = INPUT_ROOT / "test_set_input_feature_readiness_report.json"

SUMMARY_MD_PATH = REPORT_DIR / "test_set_transnetv2_anchor_refresh_summary.md"
REPORT_JSON_PATH = REPORT_DIR / "test_set_transnetv2_anchor_refresh_report.json"

OCR_MODULE_PATH = ROOT / "scripts/ocr/extract_final_scene_anchor_ocr_v2_5_development.py"
BLACK_MODULE_PATH = ROOT / "scripts/scene/extract_final_scene_anchor_black_screen_features_v2_5_development.py"
FUSION_MODULE_PATH = ROOT / "scripts/fusion/build_final_scene_audio_ocr_rule_input_v2_0_development.py"
V21A_MODULE_PATH = ROOT / "scripts/detectors/run_state_machine_interval_detector_v2_1a_experimental.py"
V21B_MODULE_PATH = ROOT / "scripts/experiments/run_state_machine_ocr_phase_transition_v2_1b_candidate_development.py"
V21A_CONFIG_PATH = ROOT / "data/experiments/state_machine_rule_weight_sweep_v2_1a_development/candidate_rule_configs_v2_1a.json"
V21B_CONFIG_PATH = ROOT / "configs/experiments/state_machine_ocr_phase_transition_v2_1b_candidate_config.json"

DEV_LATEST = ROOT / "notebooks/rule_lab_v2_3_modular/lastest_csv"

RUN_ID = f"test_set_transnetv2_anchor_refresh_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


LOGGER = Logger(LOG_PATH)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def safe_float(value: Any, default: float | None = math.nan) -> float | None:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except Exception:
        return default


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def fmt_float(value: Any, digits: int = 6) -> str:
    out = safe_float(value)
    if out is None or not math.isfinite(out):
        return ""
    return f"{out:.{digits}f}".rstrip("0").rstrip(".")


def mmss(seconds: Any) -> str:
    sec = safe_float(seconds, 0.0) or 0.0
    total = max(0, int(round(sec)))
    return f"{total // 60:02d}:{total % 60:02d}"


def read_csv_dict(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_dict(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return max(sum(1 for _ in fh) - 1, 0)
    return 0


def source_composition(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    cols = {
        "opencv_ffmpeg": "has_opencv_ffmpeg",
        "resnet": "has_resnet",
        "transnetv2_conservative": "has_transnetv2_conservative",
    }
    out: dict[str, int] = {}
    for name, col in cols.items():
        out[name] = int(df[col].fillna(False).map(bool_value).sum()) if col in df.columns else 0
    if "source_relation" in df.columns:
        out["source_relation_counts"] = {
            str(k): int(v) for k, v in df["source_relation"].fillna("").value_counts().to_dict().items()
        }
    return out


def candidate_frame(sec: float, fps: float | None) -> int | str:
    if fps is None or not math.isfinite(fps):
        return ""
    return int(round(sec * fps))


def load_split_and_mapping() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_df = pd.read_csv(SPLIT_PATH)
    manifest_df = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else pd.DataFrame()
    test_split = split_df[split_df["video_id"].astype(int).isin(TEST_IDS) & split_df["split"].isin(TEST_SPLITS)].copy()
    if sorted(test_split["video_id"].astype(int).tolist()) != TEST_IDS:
        raise RuntimeError("Test split videos do not match expected [3,4,7,16,17,18].")
    dev_overlap = sorted(set(test_split["video_id"].astype(int)).intersection(DEVELOPMENT_IDS))
    if dev_overlap:
        raise RuntimeError(f"Development videos leaked into test set: {dev_overlap}")

    manifest_by_id = {}
    if not manifest_df.empty:
        manifest_by_id = {int(row["video_id"]): row for _, row in manifest_df.iterrows()}
    mapping_rows = []
    missing = []
    for _, split_row in test_split.sort_values("video_id").iterrows():
        vid = int(split_row["video_id"])
        manifest_row = manifest_by_id.get(vid)
        video_path = ""
        if manifest_row is not None:
            video_path = str(manifest_row.get("video_path", "") or "")
        if not video_path:
            video_path = str(split_row.get("video_path", "") or "")
        duration = safe_float(manifest_row.get("duration_sec") if manifest_row is not None else None, safe_float(split_row.get("video_duration_sec")))
        fps = safe_float(manifest_row.get("fps") if manifest_row is not None else None)
        frame_count = safe_int(manifest_row.get("frame_count") if manifest_row is not None else None)
        if frame_count is None and duration is not None and fps is not None:
            frame_count = int(round(duration * fps))
        original_split = str(split_row["split"])
        eval_subset = "diagnostic_subset" if original_split == "validation" else "pure_test"
        exists = bool(video_path and Path(video_path).exists())
        if not exists:
            missing.append(vid)
        mapping_rows.append(
            {
                "video_id": vid,
                "original_split_v2_4": original_split,
                "split_role_v2_5": "extended_evaluation",
                "evaluation_subset_v2_5": eval_subset,
                "split_terminology_note": TEST_NOTE,
                "video_title": (manifest_row.get("video_title") if manifest_row is not None else split_row.get("video_name", "")) or "",
                "video_path": video_path,
                "file_exists": exists,
                "duration_sec": duration if duration is not None else math.nan,
                "fps": fps if fps is not None else math.nan,
                "frame_count": frame_count if frame_count is not None else "",
                "mapping_source": "video_manifest_v2_2" if manifest_row is not None else "video_split_v2_4",
                "notes": "test set feature preparation only; no labels used for generation.",
            }
        )
    return split_df, pd.DataFrame(mapping_rows), {"source_video_missing_ids": missing}


def load_transnet_parameters(warnings: list[str]) -> dict[str, Any]:
    params = {
        "selected_family": TRANSNET_FAMILY,
        "threshold": TRANSNET_THRESHOLD,
        "dedup_window_sec": TRANSNET_DEDUP_SEC,
        "cross_source_dedup_sec": CROSS_SOURCE_DEDUP_SEC,
        "model_weight_path": str(TRANSNET_WEIGHT_PATH),
        "model_weight_exists": TRANSNET_WEIGHT_PATH.exists(),
        "report_path": str(TRANSNET_REPORT_PATH),
    }
    if not TRANSNET_WEIGHT_PATH.exists():
        raise RuntimeError(f"TransNetV2 checkpoint missing: {TRANSNET_WEIGHT_PATH}")
    if TRANSNET_BEST_PATH.exists():
        best = pd.read_csv(TRANSNET_BEST_PATH)
        if len(best):
            row = best.iloc[0]
            params["selected_family"] = str(row.get("selected_family") or TRANSNET_FAMILY)
            params["threshold"] = float(row.get("threshold"))
            params["dedup_window_sec"] = float(row.get("dedup_window_sec"))
            if (
                params["selected_family"] != TRANSNET_FAMILY
                or abs(params["threshold"] - TRANSNET_THRESHOLD) > 1e-9
                or abs(params["dedup_window_sec"] - TRANSNET_DEDUP_SEC) > 1e-9
            ):
                warnings.append("Best TransNetV2 conservative metadata differs from expected patch5 source; expected constants were retained.")
                params["selected_family"] = TRANSNET_FAMILY
                params["threshold"] = TRANSNET_THRESHOLD
                params["dedup_window_sec"] = TRANSNET_DEDUP_SEC
    return params


def cluster_raw_transnet(raw_rows: list[dict[str, Any]], family: str, dedup_sec: float) -> list[dict[str, Any]]:
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_video[int(row["video_id"])].append(row)
    out: list[dict[str, Any]] = []
    for vid in TEST_IDS:
        rows = sorted(by_video.get(vid, []), key=lambda r: float(r["candidate_sec"]))
        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_max: float | None = None
        for row in rows:
            sec = float(row["candidate_sec"])
            if not current:
                current = [row]
                current_max = sec
            elif current_max is not None and sec - current_max <= dedup_sec:
                current.append(row)
                current_max = max(current_max, sec)
            else:
                clusters.append(current)
                current = [row]
                current_max = sec
        if current:
            clusters.append(current)
        for idx, members in enumerate(clusters, start=1):
            best = sorted(members, key=lambda r: (-float(r["transnetv2_score"]), float(r["candidate_sec"])))[0]
            out.append(
                {
                    "sweep_family": family,
                    "video_id": vid,
                    "split": best.get("split", ""),
                    "candidate_sec": float(best["candidate_sec"]),
                    "candidate_frame": int(best["candidate_frame"]),
                    "transnetv2_score": float(best["transnetv2_score"]),
                    "threshold": TRANSNET_THRESHOLD,
                    "dedup_window_sec": dedup_sec,
                    "cluster_id": f"{family}_v{vid:02d}_c{idx:05d}",
                    "cluster_member_count": len(members),
                    "cluster_min_sec": min(float(r["candidate_sec"]) for r in members),
                    "cluster_max_sec": max(float(r["candidate_sec"]) for r in members),
                    "device_used": best.get("device_used", ""),
                    "fps": best.get("fps", ""),
                    "duration_sec": best.get("duration_sec", ""),
                    "source": "test_set_transnetv2_rerun_frame_predictions",
                    "notes": "dedup representative is highest score timestamp; tie uses earliest timestamp; no average timestamp; no labels used",
                }
            )
    return out


def run_transnetv2(mapping_df: pd.DataFrame, params: dict[str, Any], warnings: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    LOGGER.log("[STEP] Running TransNetV2 conservative inference for test set")
    if str(TRANSNET_PYTHONPATH) not in sys.path:
        sys.path.insert(0, str(TRANSNET_PYTHONPATH))
    cv_bin = Path(".venv/bin")
    if cv_bin.exists():
        os.environ["PATH"] = str(cv_bin) + os.pathsep + os.environ.get("PATH", "")

    import numpy as np  # type: ignore
    import torch  # type: ignore
    from transnetv2_pytorch import TransNetV2  # type: ignore

    cuda_available = bool(torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    fallback_used = False

    def make_model(dev: str) -> Any:
        LOGGER.log(f"  - TransNetV2 model init: device={dev}")
        return TransNetV2(device=dev)

    try:
        model = make_model(device)
    except Exception as exc:
        warnings.append(f"TransNetV2 CUDA model init failed; CPU fallback used: {exc}")
        device = "cpu"
        fallback_used = True
        model = make_model(device)

    raw_rows: list[dict[str, Any]] = []
    processed: list[int] = []
    failed: list[int] = []
    no_anchor: list[int] = []
    runtimes: dict[str, float] = {}
    for _, row in mapping_df.sort_values("video_id").iterrows():
        vid = int(row["video_id"])
        if not bool(row["file_exists"]):
            failed.append(vid)
            continue
        video_path = str(row["video_path"])
        fps = safe_float(row.get("fps"))
        duration = safe_float(row.get("duration_sec"))
        active_device = device
        t0 = time.time()
        LOGGER.log(f"  - video_id={vid}: TransNetV2 inference start")
        try:
            _, single_frame_predictions, _ = model.predict_video(video_path, quiet=True)
        except Exception as exc:
            if device != "cpu":
                warnings.append(f"TransNetV2 CUDA inference failed for video_id={vid}; CPU fallback used: {exc}")
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
                    _, single_frame_predictions, _ = model.predict_video(video_path, quiet=True)
                except Exception as cpu_exc:
                    failed.append(vid)
                    warnings.append(f"TransNetV2 CPU fallback failed for video_id={vid}: {cpu_exc}")
                    continue
            else:
                failed.append(vid)
                warnings.append(f"TransNetV2 inference failed for video_id={vid}: {exc}")
                continue
        predictions = single_frame_predictions.cpu().detach().numpy().reshape(-1)
        frame_count = len(predictions)
        try:
            fps_model = safe_float(model.get_video_fps(video_path))
            if fps_model:
                fps = fps_model
        except Exception:
            pass
        if not fps:
            failed.append(vid)
            warnings.append(f"TransNetV2 fps unavailable for video_id={vid}")
            continue
        if duration is None or not math.isfinite(duration):
            duration = frame_count / fps
        scenes = model.predictions_to_scenes_with_data(predictions, fps=fps, threshold=params["threshold"])
        count_for_video = 0
        for scene_idx, scene in enumerate(scenes):
            if scene_idx == len(scenes) - 1:
                continue
            frame = int(scene.get("end_frame", 0))
            start = max(0, frame - 2)
            end = min(len(predictions), frame + 3)
            window = predictions[start:end]
            score = float(np.max(window)) if len(window) else float(scene.get("probability") or 0.0)
            sec = frame / fps
            if sec < 0:
                continue
            raw_rows.append(
                {
                    "video_id": vid,
                    "split": row["original_split_v2_4"],
                    "candidate_sec": float(sec),
                    "candidate_frame": frame,
                    "transnetv2_score": score,
                    "threshold": params["threshold"],
                    "device_used": active_device,
                    "fps": float(fps),
                    "duration_sec": float(duration),
                }
            )
            count_for_video += 1
        if count_for_video == 0:
            no_anchor.append(vid)
        processed.append(vid)
        runtimes[str(vid)] = round(time.time() - t0, 3)
        LOGGER.log(f"  - video_id={vid}: raw_candidates={count_for_video}, runtime={runtimes[str(vid)]}s")

    rows = cluster_raw_transnet(raw_rows, params["selected_family"], params["dedup_window_sec"])
    columns = [
        "sweep_family",
        "video_id",
        "split",
        "candidate_sec",
        "candidate_frame",
        "transnetv2_score",
        "threshold",
        "dedup_window_sec",
        "cluster_id",
        "cluster_member_count",
        "cluster_min_sec",
        "cluster_max_sec",
        "device_used",
        "fps",
        "duration_sec",
        "source",
        "notes",
    ]
    write_csv_dict(TRANSNET_ANCHOR_PATH, rows, columns)
    stats = {
        "cuda_available": cuda_available,
        "initial_device": "cuda" if cuda_available else "cpu",
        "final_device": device,
        "fallback_used": fallback_used,
        "processed_videos": processed,
        "failed_videos": failed,
        "videos_with_no_transnetv2_anchor_detected": no_anchor,
        "raw_candidate_count": len(raw_rows),
        "conservative_anchor_count": len(rows),
        "conservative_anchor_count_by_video": {str(k): int(v) for k, v in Counter(int(r["video_id"]) for r in rows).items()},
        "runtimes_sec": runtimes,
    }
    if failed:
        raise RuntimeError(f"TransNetV2 inference failed for videos: {failed}")
    return pd.DataFrame(rows, columns=columns), stats


def load_scene_candidates(mapping_df: pd.DataFrame, transnet_df: pd.DataFrame, warnings: list[str]) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    video_info = {int(r["video_id"]): r for _, r in mapping_df.iterrows()}
    all_candidates: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    if OPENCV_PATH.exists():
        opencv = pd.read_csv(OPENCV_PATH)
        count = 0
        for idx, row in opencv.iterrows():
            vid = safe_int(row.get("video_id"))
            sec = safe_float(row.get("candidate_time_sec"))
            if vid not in TEST_ID_SET or sec is None or not math.isfinite(sec) or sec < 0:
                continue
            fps = safe_float(video_info[vid].get("fps"))
            all_candidates.append(
                {
                    "source": "opencv_ffmpeg",
                    "video_id": vid,
                    "candidate_sec": sec,
                    "candidate_frame": candidate_frame(sec, fps),
                    "score": safe_float(row.get("scene_change_score")),
                    "source_model": row.get("candidate_source") or row.get("method_used") or "opencv_ffmpeg",
                    "raw_row_index": idx,
                    "source_path": str(OPENCV_PATH),
                    "source_id": row.get("merged_candidate_id", ""),
                }
            )
            count += 1
        stats["opencv_ffmpeg"] = {"path": str(OPENCV_PATH), "test_candidate_count": count}
    else:
        warnings.append(f"OpenCV/FFmpeg source missing: {OPENCV_PATH}")

    if RESNET_PATH.exists():
        resnet = pd.read_csv(RESNET_PATH)
        count = 0
        for idx, row in resnet.iterrows():
            vid = safe_int(row.get("video_id"))
            sec = safe_float(row.get("candidate_time_sec"))
            if vid not in TEST_ID_SET or sec is None or not math.isfinite(sec) or sec < 0:
                continue
            fps = safe_float(video_info[vid].get("fps"))
            all_candidates.append(
                {
                    "source": "resnet",
                    "video_id": vid,
                    "candidate_sec": sec,
                    "candidate_frame": candidate_frame(sec, fps),
                    "score": safe_float(row.get("scene_change_score"), safe_float(row.get("cosine_distance"))),
                    "source_model": row.get("model_name") or row.get("candidate_source") or "resnet_embedding",
                    "raw_row_index": idx,
                    "source_path": str(RESNET_PATH),
                    "source_id": row.get("score_rank_in_video", ""),
                }
            )
            count += 1
        stats["resnet"] = {"path": str(RESNET_PATH), "test_candidate_count": count}
    else:
        warnings.append(f"ResNet source missing: {RESNET_PATH}")

    trans_count = 0
    for idx, row in transnet_df.iterrows():
        vid = safe_int(row.get("video_id"))
        sec = safe_float(row.get("candidate_sec"))
        if vid not in TEST_ID_SET or sec is None or not math.isfinite(sec) or sec < 0:
            continue
        all_candidates.append(
            {
                "source": "transnetv2_conservative",
                "video_id": vid,
                "candidate_sec": sec,
                "candidate_frame": safe_int(row.get("candidate_frame"), candidate_frame(sec, safe_float(video_info[vid].get("fps")))),
                "score": safe_float(row.get("transnetv2_score")),
                "source_model": row.get("sweep_family", TRANSNET_FAMILY),
                "raw_row_index": idx,
                "source_path": str(TRANSNET_ANCHOR_PATH),
                "source_id": row.get("cluster_id", ""),
                "threshold": safe_float(row.get("threshold")),
                "dedup_window_sec": safe_float(row.get("dedup_window_sec")),
            }
        )
        trans_count += 1
    stats["transnetv2_conservative"] = {
        "path": str(TRANSNET_ANCHOR_PATH),
        "test_candidate_count": trans_count,
        "selected_family": TRANSNET_FAMILY,
        "threshold": TRANSNET_THRESHOLD,
        "dedup_window_sec": TRANSNET_DEDUP_SEC,
    }

    canonical: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if CANONICAL_PATH.exists():
        can_df = pd.read_csv(CANONICAL_PATH)
        for _, row in can_df.iterrows():
            vid = safe_int(row.get("video_id"))
            sec = safe_float(row.get("canonical_boundary_time_sec"))
            split = str(row.get("split", ""))
            if vid in TEST_ID_SET and split in TEST_SPLITS and sec is not None and math.isfinite(sec) and sec >= 0:
                canonical[vid].append(
                    {
                        "video_id": vid,
                        "canonical_sec": sec,
                        "scene_boundary_anchor_id": row.get("scene_boundary_anchor_id", ""),
                        "source_relation": row.get("source_relation", ""),
                        "canonical_time_source": row.get("canonical_time_source", ""),
                    }
                )
    else:
        warnings.append(f"Canonical anchor source missing: {CANONICAL_PATH}")
    for values in canonical.values():
        values.sort(key=lambda item: float(item["canonical_sec"]))
    stats["canonical"] = {"path": str(CANONICAL_PATH), "test_anchor_count": sum(len(v) for v in canonical.values())}
    return all_candidates, canonical, stats


def build_final_anchors(candidates: list[dict[str, Any]], canonical: dict[int, list[dict[str, Any]]], mapping_df: pd.DataFrame) -> pd.DataFrame:
    video_info = {int(r["video_id"]): r for _, r in mapping_df.iterrows()}
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_video[int(candidate["video_id"])].append(candidate)

    rows: list[dict[str, Any]] = []
    source_order = ["opencv_ffmpeg", "resnet", "transnetv2_conservative"]
    for vid in TEST_IDS:
        vcands = sorted(by_video.get(vid, []), key=lambda r: (float(r["candidate_sec"]), str(r["source"])))
        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_max: float | None = None
        for cand in vcands:
            sec = float(cand["candidate_sec"])
            if not current:
                current = [cand]
                current_max = sec
            elif current_max is not None and sec - current_max <= CROSS_SOURCE_DEDUP_SEC:
                current.append(cand)
                current_max = max(current_max, sec)
            else:
                clusters.append(current)
                current = [cand]
                current_max = sec
        if current:
            clusters.append(current)

        fps = safe_float(video_info[vid].get("fps"))
        original_split = str(video_info[vid].get("original_split_v2_4", ""))
        eval_subset = str(video_info[vid].get("evaluation_subset_v2_5", ""))
        for cluster_idx, cluster in enumerate(clusters, start=1):
            secs = [float(member["candidate_sec"]) for member in cluster]
            cluster_min = min(secs)
            cluster_max = max(secs)
            sources = {str(member["source"]) for member in cluster}
            source_counts = Counter(str(member["source"]) for member in cluster)

            canonical_match = None
            for can in canonical.get(vid, []):
                can_sec = float(can["canonical_sec"])
                distance = min(abs(can_sec - sec) for sec in secs)
                if distance <= CROSS_SOURCE_DEDUP_SEC:
                    if canonical_match is None or distance < canonical_match["nearest_distance"]:
                        canonical_match = {**can, "nearest_distance": distance}

            if canonical_match is not None:
                anchor_sec = float(canonical_match["canonical_sec"])
                representative_rule = "canonical_anchor_priority"
                numeric_scores = [safe_float(member.get("score")) for member in cluster]
                numeric_scores = [x for x in numeric_scores if x is not None and math.isfinite(x)]
                confidence = max(numeric_scores) if numeric_scores else math.nan
            else:
                scored = [member for member in cluster if safe_float(member.get("score")) is not None and math.isfinite(safe_float(member.get("score")) or math.nan)]
                if scored:
                    representative = sorted(scored, key=lambda item: (-(safe_float(item.get("score")) or -1.0), float(item["candidate_sec"])))[0]
                    anchor_sec = float(representative["candidate_sec"])
                    confidence = safe_float(representative.get("score"))
                    representative_rule = "max_source_score_then_earliest"
                else:
                    representative = sorted(cluster, key=lambda item: float(item["candidate_sec"]))[0]
                    anchor_sec = float(representative["candidate_sec"])
                    confidence = math.nan
                    representative_rule = "earliest_timestamp_no_score_available"

            relation_sources = [src for src in source_order if src in sources]
            relation = "only_" + relation_sources[0] if len(relation_sources) == 1 else "multi_source:" + "+".join(relation_sources)

            members_json: list[dict[str, Any]] = []
            for member in sorted(cluster, key=lambda item: (float(item["candidate_sec"]), str(item["source"]))):
                members_json.append(
                    {
                        "candidate_frame": member.get("candidate_frame", ""),
                        "candidate_sec": member.get("candidate_sec", ""),
                        "raw_row_index": member.get("raw_row_index", ""),
                        "score": member.get("score", ""),
                        "source": member.get("source", ""),
                        "source_id": member.get("source_id", ""),
                        "source_model": member.get("source_model", ""),
                        "source_path": member.get("source_path", ""),
                    }
                )
            if canonical_match is not None:
                members_json.append(
                    {
                        "candidate_sec": canonical_match.get("canonical_sec"),
                        "canonical_time_source": canonical_match.get("canonical_time_source"),
                        "scene_boundary_anchor_id": canonical_match.get("scene_boundary_anchor_id"),
                        "source": "canonical_opencv_resnet_anchor_for_representative_priority",
                        "source_relation": canonical_match.get("source_relation"),
                    }
                )

            rows.append(
                {
                    "final_anchor_id": f"FSA25TEST_v{vid:02d}_{cluster_idx:05d}",
                    "video_id": vid,
                    "original_split_v2_4": original_split,
                    "split_role_v2_5": "extended_evaluation",
                    "evaluation_subset_v2_5": eval_subset,
                    "split_terminology_note": TEST_NOTE,
                    "anchor_sec": anchor_sec,
                    "anchor_frame": candidate_frame(anchor_sec, fps),
                    "cluster_min_sec": cluster_min,
                    "cluster_max_sec": cluster_max,
                    "cluster_member_count": len(cluster),
                    "source_relation": relation,
                    "has_opencv_ffmpeg": "opencv_ffmpeg" in sources,
                    "has_resnet": "resnet" in sources,
                    "has_transnetv2_conservative": "transnetv2_conservative" in sources,
                    "opencv_member_count": source_counts.get("opencv_ffmpeg", 0),
                    "resnet_member_count": source_counts.get("resnet", 0),
                    "transnetv2_member_count": source_counts.get("transnetv2_conservative", 0),
                    "source_members_json": json.dumps(members_json, ensure_ascii=False, sort_keys=True),
                    "representative_rule": representative_rule,
                    "confidence_or_score": confidence if confidence is not None else "",
                    "notes": "Test set v2.5 anchor generated with Development TransNetV2 conservative clustering policy; no labels used.",
                }
            )
    columns = [
        "final_anchor_id",
        "video_id",
        "original_split_v2_4",
        "split_role_v2_5",
        "evaluation_subset_v2_5",
        "split_terminology_note",
        "anchor_sec",
        "anchor_frame",
        "cluster_min_sec",
        "cluster_max_sec",
        "cluster_member_count",
        "source_relation",
        "has_opencv_ffmpeg",
        "has_resnet",
        "has_transnetv2_conservative",
        "opencv_member_count",
        "resnet_member_count",
        "transnetv2_member_count",
        "source_members_json",
        "representative_rule",
        "confidence_or_score",
        "notes",
    ]
    df = pd.DataFrame(rows, columns=columns).sort_values(["video_id", "anchor_sec", "final_anchor_id"]).reset_index(drop=True)
    df.to_csv(FINAL_ANCHOR_PATH, index=False)
    return df


def time_range(start: float, end: float, step: float) -> list[float]:
    values = []
    cur = start
    while cur <= end + 1e-9:
        values.append(round(cur, 6))
        cur += step
    return values


def build_ocr_schedule(anchor_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    anchor_context = 10.0
    anchor_step = 1.0
    background_step = 1.5
    dedup_tolerance = 0.1
    anchors_by_video = {vid: group.sort_values("anchor_sec").copy() for vid, group in anchor_df.groupby("video_id")}
    mapping_by_vid = {int(row["video_id"]): row for _, row in mapping_df.iterrows()}
    by_dedup: dict[tuple[int, int], dict[str, Any]] = {}
    source_priority = {"anchor_dense": 0, "background_regular": 1}

    def nearest_anchor(vid: int, time_sec: float) -> tuple[dict[str, Any] | None, float | None]:
        anchors = anchors_by_video.get(vid)
        if anchors is None or anchors.empty:
            return None, None
        idx = (anchors["anchor_sec"].astype(float) - time_sec).abs().idxmin()
        row = anchors.loc[idx].to_dict()
        distance = abs(float(row["anchor_sec"]) - time_sec)
        return row, distance

    def add_time(vid: int, time_sec: float, source: str) -> None:
        mapping = mapping_by_vid[vid]
        duration = safe_float(mapping.get("duration_sec"))
        fps = safe_float(mapping.get("fps"))
        if duration is None or not math.isfinite(duration) or time_sec < 0 or time_sec > duration + 1e-6:
            return
        nearest, distance = nearest_anchor(vid, time_sec)
        within = bool(distance is not None and distance <= anchor_context + 1e-9)
        if source == "background_regular" and within:
            return
        dedup_key = int(round(time_sec / dedup_tolerance))
        key = (vid, dedup_key)
        row = {
            "schedule_id": "",
            "video_id": vid,
            "original_split_v2_4": mapping.get("original_split_v2_4", ""),
            "split_role_v2_5": "extended_evaluation",
            "evaluation_subset_v2_5": mapping.get("evaluation_subset_v2_5", ""),
            "split_terminology_note": TEST_NOTE,
            "ocr_time_sec": time_sec,
            "ocr_frame_index": candidate_frame(time_sec, fps),
            "schedule_source": source,
            "nearest_final_anchor_id": nearest.get("final_anchor_id", "") if nearest else "",
            "nearest_final_anchor_sec": nearest.get("anchor_sec", "") if nearest else "",
            "distance_to_nearest_anchor_sec": distance if distance is not None else "",
            "within_anchor_context": within,
            "anchor_context_sec": anchor_context,
            "anchor_dense_step_sec": anchor_step,
            "background_step_sec": background_step,
            "dedup_key": f"v{vid}_t{dedup_key}",
            "video_path": mapping.get("video_path", ""),
            "notes": "OCR schedule only; no labels used.",
        }
        existing = by_dedup.get(key)
        if existing is None:
            by_dedup[key] = row
            return
        existing_distance = safe_float(existing.get("distance_to_nearest_anchor_sec"), 1e9) or 1e9
        new_distance = distance if distance is not None else 1e9
        if source_priority[source] < source_priority[existing["schedule_source"]] or (
            source_priority[source] == source_priority[existing["schedule_source"]] and new_distance < existing_distance
        ):
            by_dedup[key] = row

    for _, mapping in mapping_df.sort_values("video_id").iterrows():
        vid = int(mapping["video_id"])
        if not bool(mapping["file_exists"]):
            continue
        duration = safe_float(mapping.get("duration_sec"))
        if duration is None or not math.isfinite(duration) or duration <= 0:
            continue
        anchors = anchors_by_video.get(vid, pd.DataFrame())
        for _, anchor in anchors.iterrows():
            anchor_sec = safe_float(anchor.get("anchor_sec"))
            if anchor_sec is None or not math.isfinite(anchor_sec):
                continue
            start = max(0.0, anchor_sec - anchor_context)
            end = min(duration, anchor_sec + anchor_context)
            for time_sec in time_range(start, end, anchor_step):
                add_time(vid, time_sec, "anchor_dense")
        for time_sec in time_range(0.0, duration, background_step):
            add_time(vid, time_sec, "background_regular")

    rows = sorted(by_dedup.values(), key=lambda r: (int(r["video_id"]), safe_float(r["ocr_time_sec"], 0.0) or 0.0, r["schedule_source"]))
    for idx, row in enumerate(rows, start=1):
        row["schedule_id"] = f"OCRS25TEST_{idx:08d}"
    out = pd.DataFrame(rows)
    out.to_csv(OCR_SCHEDULE_PATH, index=False)
    return out


def build_or_refresh_ocr_features(schedule_df: pd.DataFrame, mapping_df: pd.DataFrame, anchor_df: pd.DataFrame, warnings: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    LOGGER.log("[STEP] Refreshing OCR frame/source rows and derived OCR features")
    ocrmod = import_module("ocr_v25_dev", OCR_MODULE_PATH)
    ocrmod.VERSION = "v2_5_test_set"
    ocrmod.SPLIT_NOTE = TEST_NOTE

    run_id = f"final_scene_anchor_ocr_v2_5_test_set_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    enriched = ocrmod.build_enriched_schedule(schedule_df, mapping_df, anchor_df, run_id)

    old = pd.read_csv(OCR_SOURCE_PATH) if OCR_SOURCE_PATH.exists() else pd.DataFrame()
    old_by_key: dict[tuple[int, float], dict[str, Any]] = {}
    if not old.empty:
        for _, row in old.iterrows():
            old_by_key[(int(row["video_id"]), round(float(row["timestamp_sec"]), 3))] = row.to_dict()

    backend_info, reader = ocrmod.select_backend("easyocr")
    if backend_info.get("ocr_backend_status") != "ready":
        warnings.append(f"OCR backend unavailable; missing rows will be marked skipped: {backend_info.get('warning', '')}")
    result_rows: list[dict[str, Any]] = []
    reused = 0
    generated = 0
    failed_new = 0
    skipped_new = 0
    cap_by_video: dict[int, Any] = {}
    import cv2  # type: ignore

    try:
        for _, sched_row in enriched.iterrows():
            vid = int(sched_row["video_id"])
            ts = round(float(sched_row["timestamp_sec"]), 3)
            old_row = old_by_key.get((vid, ts))
            if old_row is not None:
                merged = dict(old_row)
                for col in [
                    "run_id",
                    "version",
                    "original_split_v2_4",
                    "split_role_v2_5",
                    "evaluation_subset_v2_5",
                    "split_terminology_note",
                    "video_id",
                    "video_title",
                    "video_path",
                    "video_duration_sec",
                    "sample_id",
                    "schedule_id",
                    "timestamp_sec",
                    "timestamp_mmss",
                    "sampling_role",
                    "is_anchor_dense",
                    "is_background_regular",
                    "nearest_anchor_id",
                    "nearest_anchor_time_sec",
                    "nearest_anchor_delta_sec",
                    "anchor_ids_joined",
                    "anchor_source_relation_joined",
                    "anchor_model_source_joined",
                    "anchor_source_count_max",
                    "anchor_strength_score_max",
                ]:
                    src_col = "schedule_source" if col == "sampling_role" else col
                    if col in sched_row.index or src_col in sched_row.index:
                        merged[col] = sched_row.get(src_col, sched_row.get(col, merged.get(col, "")))
                result_rows.append(merged)
                reused += 1
                continue
            if backend_info.get("ocr_backend_status") != "ready":
                result_rows.append(ocrmod.empty_result(sched_row, backend_info, ocrmod.SKIPPED, warning=backend_info.get("warning", "")))
                skipped_new += 1
                continue
            cap = cap_by_video.get(vid)
            if cap is None:
                cap = cv2.VideoCapture(str(sched_row["video_path"]))
                cap_by_video[vid] = cap
            frame, status = ocrmod.decode_frame(cap, Path(str(sched_row["video_path"])), float(sched_row["timestamp_sec"]), float(sched_row["video_duration_sec"]))
            if frame is None:
                result_rows.append(ocrmod.empty_result(sched_row, backend_info, ocrmod.FAILED, error=status))
                failed_new += 1
                continue
            detections = ocrmod.run_easyocr(reader, frame) if backend_info.get("ocr_backend") == "easyocr" else []
            result_rows.append(ocrmod.result_from_detections(sched_row, backend_info, detections, frame.shape))
            generated += 1
    finally:
        for cap in cap_by_video.values():
            try:
                cap.release()
            except Exception:
                pass

    frame_df = pd.DataFrame(result_rows)
    # 가능하면 Development 결과 column 순서를 그대로 유지한다.
    result_cols = [col for col in ocrmod.RESULT_COLUMNS if col in frame_df.columns]
    extra_cols = [col for col in frame_df.columns if col not in result_cols]
    frame_df = frame_df[result_cols + extra_cols]
    frame_df.to_csv(OCR_SOURCE_PATH, index=False)

    anchor_features = ocrmod.build_anchor_features(frame_df, enriched, anchor_df, run_id)
    timeline_features = ocrmod.build_timeline_features(frame_df, mapping_df, run_id)
    anchor_features.to_csv(OCR_ANCHOR_FEATURE_PATH, index=False)
    timeline_features.to_csv(OCR_TIMELINE_FEATURE_PATH, index=False)

    stats = {
        "ocr_backend_status": backend_info.get("ocr_backend_status"),
        "ocr_backend": backend_info.get("ocr_backend"),
        "ocr_reused_existing_rows": reused,
        "ocr_generated_new_rows": generated,
        "ocr_failed_new_rows": failed_new,
        "ocr_skipped_new_rows": skipped_new,
        "ocr_frame_rows": len(frame_df),
        "ocr_status_counts": {str(k): int(v) for k, v in frame_df["ocr_status"].fillna("").value_counts().to_dict().items()} if "ocr_status" in frame_df else {},
        "ocr_anchor_feature_rows": len(anchor_features),
        "ocr_timeline_feature_rows": len(timeline_features),
    }
    return frame_df, anchor_features, timeline_features, stats


def refresh_black_features(anchor_df: pd.DataFrame, mapping_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    LOGGER.log("[STEP] Refreshing black-screen anchor features for new final anchors")
    if BLACK_FEATURE_PATH.exists():
        existing = pd.read_csv(BLACK_FEATURE_PATH)
        existing_ids = set(existing.get("final_scene_anchor_id", pd.Series(dtype=str)).astype(str))
        anchor_ids = set(anchor_df.get("final_scene_anchor_id", pd.Series(dtype=str)).astype(str))
        existing_videos = sorted(existing.get("video_id", pd.Series(dtype=int)).astype(int).unique().tolist()) if "video_id" in existing else []
        if len(existing) == len(anchor_df) and existing_ids == anchor_ids and existing_videos == TEST_IDS:
            LOGGER.log("[STEP] Reusing existing refreshed black-screen anchor features")
            stats = {
                "black_feature_rows": len(existing),
                "black_event_count": int(existing.get("black_event_near_anchor", pd.Series(dtype=bool)).fillna(False).map(bool_value).sum()) if len(existing) else 0,
                "black_end_support_eligible_count": int(existing.get("black_end_support_eligible", pd.Series(dtype=bool)).fillna(False).map(bool_value).sum()) if len(existing) else 0,
                "frame_sample_rows": None,
                "extraction_index_rows": None,
                "video_ids": existing_videos,
                "reused_existing_black_features": True,
            }
            return existing, stats
    blackmod = import_module("black_v25_dev", BLACK_MODULE_PATH)
    video_mapping = {}
    for _, row in mapping_df.iterrows():
        vid = int(row["video_id"])
        video_mapping[vid] = blackmod.VideoInfo(
            video_id=vid,
            original_split_v2_4=str(row["original_split_v2_4"]),
            split_role_v2_5="extended_evaluation",
            evaluation_subset_v2_5=str(row["evaluation_subset_v2_5"]),
            split_terminology_note=TEST_NOTE,
            video_path=str(row["video_path"]),
            file_exists=bool(row["file_exists"]),
            video_duration_sec=float(row["duration_sec"]),
            fps=float(row["fps"]) if math.isfinite(float(row["fps"])) else 0.0,
            frame_count=int(row["frame_count"]) if str(row["frame_count"]) not in {"", "nan"} else 0,
        )
    sample_rows, unique_reads = blackmod.build_sample_plan(anchor_df, video_mapping)
    all_frame_stats: dict[int, dict[str, Any]] = {}
    extraction_index = []
    for vid in TEST_IDS:
        info = video_mapping.get(vid)
        reads = unique_reads.get(vid, {})
        if info is None or not reads:
            continue
        start = time.time()
        frame_results, index_row = blackmod.read_video_frames(vid, info, reads)
        all_frame_stats.update(frame_results)
        extraction_index.append(index_row)
    feature_rows, frame_sample_rows = blackmod.calculate_anchor_features(anchor_df, sample_rows, all_frame_stats, video_mapping)
    out = pd.DataFrame(feature_rows)
    cols = [col for col in blackmod.FEATURE_COLUMNS if col in out.columns]
    extra = [col for col in out.columns if col not in cols]
    out = out[cols + extra]
    out.to_csv(BLACK_FEATURE_PATH, index=False)
    stats = {
        "black_feature_rows": len(out),
        "black_event_count": int(out.get("black_event_near_anchor", pd.Series(dtype=bool)).fillna(False).map(bool_value).sum()) if len(out) else 0,
        "black_end_support_eligible_count": int(out.get("black_end_support_eligible", pd.Series(dtype=bool)).fillna(False).map(bool_value).sum()) if len(out) else 0,
        "frame_sample_rows": len(frame_sample_rows),
        "extraction_index_rows": len(extraction_index),
        "video_ids": sorted(out["video_id"].astype(int).unique().tolist()) if "video_id" in out else [],
    }
    return out, stats


def build_fusion_input(anchor_df: pd.DataFrame, mapping_df: pd.DataFrame, ocr_anchor_df: pd.DataFrame, timeline_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    LOGGER.log("[STEP] Rebuilding fusion input from refreshed anchors/OCR features")
    fusion = import_module("fusion_v20_dev", FUSION_MODULE_PATH)
    audio_parts = []
    for name in ["blind_per_video_audio_relative_levels_v2_4_validation.csv", "blind_per_video_audio_relative_levels_v2_4_test.csv"]:
        path = ROOT / "data/audio" / name
        if path.exists():
            audio_parts.append(pd.read_csv(path))
    audio_df = pd.concat(audio_parts, ignore_index=True, sort=False) if audio_parts else pd.DataFrame()
    if not audio_df.empty:
        audio_df = audio_df[audio_df["video_id"].astype(int).isin(TEST_IDS)].copy()
        audio_df["per_video_relative_active_audio_score"] = audio_df.get("blind_relative_active_audio_score", 0)
        audio_df["per_video_relative_quiet_audio_score"] = audio_df.get("blind_relative_quiet_audio_score", 0)
        audio_df["per_video_local_context_shift_score"] = audio_df.get("spectral_flux_mean__directional_score", 0)
        audio_df["per_video_sustained_context_score"] = audio_df.get("onset_strength_mean__directional_score", 0)

    duration_by_video = {int(row["video_id"]): float(row["duration_sec"]) for _, row in mapping_df.iterrows()}
    ocr_by_id = {str(row["scene_boundary_anchor_id"]): row.to_dict() for _, row in ocr_anchor_df.iterrows() if str(row.get("scene_boundary_anchor_id", ""))}
    ocr_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for _, row in ocr_anchor_df.iterrows():
        ocr_by_video[int(row["video_id"])].append(row.to_dict())
    timeline_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for _, row in timeline_df.iterrows():
        timeline_by_video[int(row["video_id"])].append(row.to_dict())
    audio_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for _, row in audio_df.iterrows():
        audio_by_video[int(row["video_id"])].append(row.to_dict())
    for values in audio_by_video.values():
        values.sort(key=lambda r: fusion.fnum(r.get("start_sec")))

    decision_cols = [
        "ocr_hard_disclosure_flag",
        "ocr_product_cta_continuity_flag",
        "ocr_timeline_recent_hard_evidence_flag",
        "ocr_start_signal_level_v2_0",
        "ocr_end_signal_level_v2_0",
        "ocr_context_level_v2_0",
        "ocr_context_reliability_level",
        "audio_pre_relative_active_score",
        "audio_post_relative_active_score",
        "audio_context_relative_active_score",
        "audio_pre_relative_quiet_score",
        "audio_post_relative_quiet_score",
        "audio_context_relative_quiet_score",
        "audio_relative_local_shift_score",
        "audio_relative_sustained_context_score",
        "audio_start_signal_level_v2_0",
        "audio_end_signal_level_v2_0",
        "audio_context_level_v2_0",
        "audio_not_informative_flag",
        "opening_disclosure_guard_applied",
        "opening_disclosure_confirmed_by_later_product_cta",
        "hard_evidence_flag_v2_0",
        "support_evidence_flag_v2_0",
        "weak_context_flag_v2_0",
    ]
    bad_decision = fusion.forbidden(decision_cols)
    rows = []
    ocr_exact = 0
    ocr_nearest = 0
    audio_nonempty = 0
    timeline_nonempty = 0
    for _, srow in anchor_df.sort_values(["video_id", "anchor_sec"]).iterrows():
        s = srow.to_dict()
        vid = int(s["video_id"])
        t = fusion.fnum(s.get("anchor_sec"))
        final_id = str(s.get("final_anchor_id", ""))
        ocr = ocr_by_id.get(final_id)
        join_method = "scene_boundary_anchor_id_exact"
        if ocr is None:
            candidates = ocr_by_video.get(vid, [])
            nearest = min(candidates, key=lambda r: abs(fusion.fnum(r.get("canonical_boundary_time_sec")) - t), default=None)
            if nearest is not None and abs(fusion.fnum(nearest.get("canonical_boundary_time_sec")) - t) <= 0.25:
                ocr = nearest
                join_method = "video_id_time_nearest_0_25s"
                ocr_nearest += 1
            else:
                ocr = {}
                join_method = "missing"
        else:
            ocr_exact += 1
        dur = duration_by_video.get(vid, fusion.fnum(s.get("video_duration_sec"), 0.0))
        pre_tl = [r for r in timeline_by_video.get(vid, []) if fusion.fnum(r.get("segment_end_sec")) > max(0, t - 20) and fusion.fnum(r.get("segment_start_sec")) < t]
        post_tl = [r for r in timeline_by_video.get(vid, []) if fusion.fnum(r.get("segment_start_sec")) < t + 20 and fusion.fnum(r.get("segment_end_sec")) > t]
        context_tl = [r for r in timeline_by_video.get(vid, []) if fusion.fnum(r.get("segment_end_sec")) > max(0, t - 30) and fusion.fnum(r.get("segment_start_sec")) < t + 30]
        if context_tl:
            timeline_nonempty += 1
        audio_rows = audio_by_video.get(vid, [])
        pre_a = [r for r in audio_rows if fusion.overlap(r, max(0, t - 10), t)]
        post_a = [r for r in audio_rows if fusion.overlap(r, t, t + 10)]
        context_a = [r for r in audio_rows if fusion.overlap(r, max(0, t - 10), t + 10)]
        if context_a:
            audio_nonempty += 1

        anchor_disc = fusion.fnum(ocr.get("corrected_ad_disclosure_hit_count_sum"))
        anchor_sponsor = fusion.fnum(ocr.get("corrected_sponsor_keyword_count_sum"))
        anchor_product = fusion.fnum(ocr.get("corrected_brand_product_keyword_count_sum"))
        anchor_promo = fusion.fnum(ocr.get("corrected_promotion_discount_keyword_count_sum"))
        anchor_cta = fusion.fnum(ocr.get("corrected_purchase_cta_keyword_count_sum"))
        anchor_link = fusion.fnum(ocr.get("corrected_link_more_info_keyword_count_sum"))
        anchor_neg = fusion.fnum(ocr.get("corrected_negative_guard_keyword_count_sum"))
        pre_disc = fusion.sum_col(pre_tl, "corrected_ad_disclosure_hit_count_sum")
        post_disc = max(fusion.sum_col(post_tl, "corrected_ad_disclosure_hit_count_sum"), anchor_disc)
        pre_sponsor = fusion.sum_col(pre_tl, "corrected_sponsor_keyword_count_sum")
        post_sponsor = max(fusion.sum_col(post_tl, "corrected_sponsor_keyword_count_sum"), anchor_sponsor)
        pre_product = fusion.sum_col(pre_tl, "corrected_brand_product_keyword_count_sum")
        post_product = max(fusion.sum_col(post_tl, "corrected_brand_product_keyword_count_sum"), anchor_product)
        pre_promo = fusion.sum_col(pre_tl, "corrected_promotion_discount_keyword_count_sum")
        post_promo = max(fusion.sum_col(post_tl, "corrected_promotion_discount_keyword_count_sum"), anchor_promo)
        pre_cta = fusion.sum_col(pre_tl, "corrected_purchase_cta_keyword_count_sum")
        post_cta = max(fusion.sum_col(post_tl, "corrected_purchase_cta_keyword_count_sum"), anchor_cta)
        pre_link = fusion.sum_col(pre_tl, "corrected_link_more_info_keyword_count_sum")
        post_link = max(fusion.sum_col(post_tl, "corrected_link_more_info_keyword_count_sum"), anchor_link)
        pre_neg = fusion.sum_col(pre_tl, "corrected_negative_guard_keyword_count_sum")
        post_neg = max(fusion.sum_col(post_tl, "corrected_negative_guard_keyword_count_sum"), anchor_neg)
        pre_score = fusion.avg_col(pre_tl, "corrected_frame_ad_text_score_mean")
        post_score = max(fusion.avg_col(post_tl, "corrected_frame_ad_text_score_mean"), fusion.fnum(ocr.get("corrected_frame_ad_text_score_mean")))
        context_score = max(fusion.max_col(context_tl, "corrected_frame_ad_text_score_max"), fusion.fnum(ocr.get("corrected_frame_ad_text_score_max")))
        recent_hard_segments = [
            r
            for r in context_tl
            if fusion.fnum(r.get("corrected_ad_disclosure_hit_count_sum")) > 0
            and fusion.fnum(r.get("corrected_negative_guard_keyword_count_sum")) <= 0
            and (
                fusion.fnum(r.get("corrected_brand_product_keyword_count_sum")) > 0
                or fusion.fnum(r.get("corrected_purchase_cta_keyword_count_sum")) > 0
                or fusion.fnum(r.get("corrected_link_more_info_keyword_count_sum")) > 0
                or fusion.fnum(r.get("corrected_frame_ad_text_score_max")) >= 0.55
            )
        ]
        recent_hard = bool(recent_hard_segments)
        recent_time = max((fusion.fnum(r.get("segment_start_sec")) for r in recent_hard_segments), default="")
        valid_ratio = 0.0
        frame_count = fusion.fnum(ocr.get("ocr_frame_count"))
        if frame_count > 0:
            valid_ratio = (fusion.fnum(ocr.get("ocr_success_text_count")) + fusion.fnum(ocr.get("ocr_success_empty_count"))) / frame_count
        text_ratio = fusion.fnum(ocr.get("ocr_text_frame_ratio"))
        context_valid = max(valid_ratio, fusion.avg_col(context_tl, "ocr_text_frame_ratio"))
        reliability = fusion.reliability_level(context_valid if context_valid else text_ratio)
        exact = post_disc
        typo = 0.0
        prox = 0.0
        fuzzy = 0.0
        neg_total = pre_neg + post_neg
        has_product = pre_product + post_product > 0
        has_cta_link = pre_cta + post_cta + pre_link + post_link > 0
        post_increase = post_score > pre_score + 0.05 or context_score >= 0.55
        hard_disc = exact + typo + prox > 0 and neg_total <= 0 and (has_product or has_cta_link or post_increase or recent_hard)
        fuzzy_review = fuzzy > 0 and exact + typo + prox == 0
        product_cta = has_product and has_cta_link
        support_ocr = (pre_sponsor + post_sponsor > 0) or has_product or has_cta_link or (reliability in {"medium", "high"} and context_score >= 0.25) or ((pre_promo + post_promo) > 0 and (has_product or has_cta_link))
        weak_ocr = fuzzy_review or (pre_promo + post_promo > 0 and not (has_product or has_cta_link)) or context_valid < 0.4 or frame_count == 0

        pre_active = fusion.p90([fusion.fnum(r.get("per_video_relative_active_audio_score")) for r in pre_a])
        post_active = fusion.p90([fusion.fnum(r.get("per_video_relative_active_audio_score")) for r in post_a])
        context_active = fusion.p90([fusion.fnum(r.get("per_video_relative_active_audio_score")) for r in context_a])
        pre_quiet = fusion.p90([fusion.fnum(r.get("per_video_relative_quiet_audio_score")) for r in pre_a])
        post_quiet = fusion.p90([fusion.fnum(r.get("per_video_relative_quiet_audio_score")) for r in post_a])
        context_quiet = fusion.p90([fusion.fnum(r.get("per_video_relative_quiet_audio_score")) for r in context_a])
        local_shift = max(fusion.avg_col(context_a, "per_video_local_context_shift_score"), post_active - pre_active, post_quiet - pre_quiet)
        sustained = fusion.p90([fusion.fnum(r.get("per_video_sustained_context_score")) for r in context_a])
        audio_not_info = not context_a or max(context_active, context_quiet, abs(local_shift), sustained) < 0.05
        act_label = "not_informative" if audio_not_info else ("relative_active" if context_active >= context_quiet else "relative_quiet")
        start_audio = fusion.level(max(post_active, context_active, max(0, local_shift)))
        end_audio = fusion.level(max(post_quiet, context_quiet, max(0, post_quiet - pre_quiet)))
        context_audio = fusion.level(max(context_active, sustained))
        before_audio = fusion.level(pre_active)
        after_audio = fusion.level(post_active)

        opening_win = min(45.0, max(30.0, 0.02 * dur)) if dur else 30.0
        in_opening = t <= opening_win
        disclosure_present = exact + typo + prox > 0
        later_confirm = (post_product + post_cta + post_link > 0) or recent_hard or (post_active >= 0.65 and post_score > pre_score)
        opening_notice = in_opening and disclosure_present and not later_confirm
        opening_guard = opening_notice

        hard_reasons = []
        support_reasons = []
        weak_reasons = []
        if hard_disc:
            hard_reasons.append("hard_ocr_disclosure_confirmed")
        if product_cta:
            hard_reasons.append("product_cta_continuity")
        if recent_hard:
            hard_reasons.append("ocr_timeline_recent_hard")
        if support_ocr:
            support_reasons.append("ocr_support_context")
        if start_audio in {"medium", "high"} and (hard_disc or support_ocr):
            support_reasons.append("same_video_relative_audio_active_support")
        if end_audio in {"medium", "high"} and reliability == "high" and context_score < 0.2:
            support_reasons.append("audio_quiet_with_ocr_drop_end_support")
        if weak_ocr:
            weak_reasons.append("ocr_weak_or_unknown")
        if audio_not_info:
            weak_reasons.append("audio_not_informative")
        if opening_notice:
            weak_reasons.append("opening_disclosure_notice_review_only")
        hard_flag = bool(hard_reasons) and not opening_notice
        support_flag = bool(support_reasons)
        weak_flag = bool(weak_reasons)
        src_count = int(bool_value(s.get("has_opencv_ffmpeg"))) + int(bool_value(s.get("has_resnet"))) + int(bool_value(s.get("has_transnetv2_conservative")))
        scene_score = src_count / 3.0
        rows.append(
            {
                "version": "v2.0_test_set",
                "detector_id": "state_machine_interval_detector_v2_0_test_set",
                "original_split_v2_4": str(s.get("original_split_v2_4", "")),
                "split_role_v2_5": "extended_evaluation",
                "evaluation_subset_v2_5": str(s.get("evaluation_subset_v2_5", "")),
                "video_id": vid,
                "video_duration_sec": f"{dur:.6f}",
                "final_scene_anchor_id": final_id,
                "transition_time_anchor": f"{t:.6f}",
                "candidate_time_sec": f"{t:.6f}",
                "candidate_time_mmss": mmss(t),
                "final_anchor_source_count": src_count,
                "has_opencv_ffmpeg": str(bool_value(s.get("has_opencv_ffmpeg"))).lower(),
                "has_resnet": str(bool_value(s.get("has_resnet"))).lower(),
                "has_transnetv2_conservative": str(bool_value(s.get("has_transnetv2_conservative"))).lower(),
                "scene_anchor_role": "transition_time_anchor",
                "scene_transition_reliability_score": f"{scene_score:.6f}",
                "scene_transition_reliability_level": fusion.reliability_level(scene_score),
                "scene_is_direct_ad_start_evidence": "false",
                "scene_is_direct_ad_end_evidence": "false",
                "scene_used_for_ad_likelihood_directly": "false",
                "ocr_context_reliability_level": reliability,
                "ocr_pre_valid_frame_ratio": f"{fusion.avg_col(pre_tl, 'ocr_text_frame_ratio'):.6f}",
                "ocr_post_valid_frame_ratio": f"{max(fusion.avg_col(post_tl, 'ocr_text_frame_ratio'), text_ratio):.6f}",
                "ocr_context_valid_frame_ratio": f"{context_valid:.6f}",
                "ocr_pre_text_frame_ratio": f"{fusion.avg_col(pre_tl, 'ocr_text_frame_ratio'):.6f}",
                "ocr_post_text_frame_ratio": f"{max(fusion.avg_col(post_tl, 'ocr_text_frame_ratio'), text_ratio):.6f}",
                "corrected_pre_ad_disclosure_exact_count": f"{pre_disc:.0f}",
                "corrected_post_ad_disclosure_exact_count": f"{post_disc:.0f}",
                "corrected_pre_ad_disclosure_typo_count": "0",
                "corrected_post_ad_disclosure_typo_count": "0",
                "corrected_pre_ad_disclosure_proximity_count": "0",
                "corrected_post_ad_disclosure_proximity_count": "0",
                "corrected_pre_ad_disclosure_fuzzy_review_count": "0",
                "corrected_post_ad_disclosure_fuzzy_review_count": "0",
                "corrected_pre_brand_product_count": f"{pre_product:.0f}",
                "corrected_post_brand_product_count": f"{post_product:.0f}",
                "corrected_pre_purchase_cta_count": f"{pre_cta:.0f}",
                "corrected_post_purchase_cta_count": f"{post_cta:.0f}",
                "corrected_pre_link_more_info_count": f"{pre_link:.0f}",
                "corrected_post_link_more_info_count": f"{post_link:.0f}",
                "corrected_pre_promotion_discount_count": f"{pre_promo:.0f}",
                "corrected_post_promotion_discount_count": f"{post_promo:.0f}",
                "corrected_pre_sponsor_count": f"{pre_sponsor:.0f}",
                "corrected_post_sponsor_count": f"{post_sponsor:.0f}",
                "corrected_pre_negative_guard_count": f"{pre_neg:.0f}",
                "corrected_post_negative_guard_count": f"{post_neg:.0f}",
                "ocr_hard_disclosure_flag": str(hard_disc).lower(),
                "ocr_fuzzy_only_review_flag": str(fuzzy_review).lower(),
                "ocr_product_cta_continuity_flag": str(product_cta).lower(),
                "ocr_timeline_recent_hard_evidence_flag": str(recent_hard).lower(),
                "ocr_timeline_recent_hard_evidence_time": recent_time,
                "ocr_start_signal_level_v2_0": "high" if hard_disc else ("medium" if support_ocr else "low"),
                "ocr_end_signal_level_v2_0": "medium" if reliability == "high" and context_score < 0.2 and post_quiet >= 0.35 else "low",
                "ocr_context_level_v2_0": "high" if context_score >= 0.55 else ("medium" if context_score >= 0.25 or support_ocr else "low"),
                "audio_pre_relative_active_score": f"{pre_active:.6f}",
                "audio_post_relative_active_score": f"{post_active:.6f}",
                "audio_context_relative_active_score": f"{context_active:.6f}",
                "audio_pre_relative_quiet_score": f"{pre_quiet:.6f}",
                "audio_post_relative_quiet_score": f"{post_quiet:.6f}",
                "audio_context_relative_quiet_score": f"{context_quiet:.6f}",
                "audio_relative_local_shift_score": f"{local_shift:.6f}",
                "audio_relative_sustained_context_score": f"{sustained:.6f}",
                "audio_relative_activity_label": act_label,
                "audio_start_signal_level_v2_0": start_audio,
                "audio_end_signal_level_v2_0": end_audio,
                "audio_context_level_v2_0": context_audio,
                "audio_before_context_level_v2_0": before_audio,
                "audio_after_context_level_v2_0": after_audio,
                "audio_not_informative_flag": str(audio_not_info).lower(),
                "opening_disclosure_window_sec": f"{opening_win:.6f}",
                "is_in_opening_disclosure_window": str(in_opening).lower(),
                "opening_disclosure_notice_flag": str(opening_notice).lower(),
                "opening_disclosure_guard_applied": str(opening_guard).lower(),
                "opening_disclosure_confirmed_by_later_product_cta": str(later_confirm).lower(),
                "hard_evidence_flag_v2_0": str(hard_flag).lower(),
                "support_evidence_flag_v2_0": str(support_flag).lower(),
                "weak_context_flag_v2_0": str(weak_flag).lower(),
                "hard_evidence_reasons_v2_0": ";".join(hard_reasons),
                "support_evidence_reasons_v2_0": ";".join(support_reasons),
                "weak_context_reasons_v2_0": ";".join(weak_reasons),
                "evidence_tier_summary_v2_0": f"hard={len(hard_reasons)};support={len(support_reasons)};weak={len(weak_reasons)}",
                "decision_feature_columns_json": json.dumps(decision_cols, ensure_ascii=False),
                "forbidden_decision_columns_found": json.dumps(bad_decision, ensure_ascii=False),
                "label_derived_audio_columns_excluded": "true",
                "audit_columns_used_for_decision": "false",
                "ocr_join_method": join_method,
                "audio_window_row_count": len(context_a),
                "timeline_window_segment_count": len(context_tl),
            }
        )
    columns = pd.read_csv(FUSION_PATH, nrows=0).columns.tolist() if FUSION_PATH.exists() else list(rows[0].keys() if rows else [])
    out = pd.DataFrame(rows)
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    out = out[columns]
    out.to_csv(FUSION_PATH, index=False)
    stats = {
        "fusion_rows": len(out),
        "ocr_join_coverage": round((ocr_exact + ocr_nearest) / len(anchor_df), 6) if len(anchor_df) else 0.0,
        "audio_join_coverage": round(audio_nonempty / len(anchor_df), 6) if len(anchor_df) else 0.0,
        "timeline_window_coverage": round(timeline_nonempty / len(anchor_df), 6) if len(anchor_df) else 0.0,
        "forbidden_decision_columns_found": bad_decision,
    }
    return out, stats


def build_v21b_candidates(frame_df: pd.DataFrame) -> dict[str, Any]:
    LOGGER.log("[STEP] Regenerating v2_1b test-set candidate inputs")
    detmod = import_module("det_v21a", V21A_MODULE_PATH)
    v21b = import_module("v21b_candidate", V21B_MODULE_PATH)
    detmod.DEV_IDS = set(TEST_IDS)
    v21b.DEV_IDS = list(TEST_IDS)

    fusion_rows = read_csv_dict(FUSION_PATH)
    frame_rows = frame_df.to_dict("records")
    restoration_map = detmod.build_ocr_restoration_map(frame_rows)
    enhanced_result = detmod.enhance_rows_with_v21a_flags(fusion_rows, restoration_map)
    enhanced = enhanced_result[0] if isinstance(enhanced_result, tuple) else enhanced_result

    configs = json.loads(V21A_CONFIG_PATH.read_text(encoding="utf-8"))
    base_config = None
    for item in configs:
        if item.get("config_id") == "v2_1a_short_ad_safe_01":
            base_config = item
            break
    if base_config is None:
        raise RuntimeError("v2_1a_short_ad_safe_01 config missing")
    base_outputs = detmod.run_detector_for_config(enhanced, base_config)

    base_candidates: list[dict[str, Any]] = []
    bucket_map = {
        "prediction": base_outputs.get("predictions", []),
        "review_only": base_outputs.get("review_only", []),
        "overprediction_pruned_review": base_outputs.get("overprediction_pruned_review", []),
        "open": base_outputs.get("open_candidates", []),
    }
    for bucket, rows in bucket_map.items():
        for idx, row in enumerate(rows, start=1):
            out = dict(row)
            out["base_bucket"] = bucket
            out["base_candidate_id"] = row.get("candidate_id") or row.get("prediction_id") or f"{base_config['config_id']}_{bucket}_{idx:05d}"
            out["candidate_id"] = out["base_candidate_id"]
            out["ad_start_sec"] = row.get("ad_start_sec") or row.get("start_sec") or row.get("candidate_time_sec") or "0"
            out["ad_end_sec"] = row.get("ad_end_sec") or row.get("end_sec") or row.get("last_anchor_sec") or row.get("ad_start_sec") or "0"
            out["ad_duration_sec"] = row.get("ad_duration_sec") or row.get("duration_proxy_sec") or str(max(0.0, v21b.fnum(out["ad_end_sec"]) - v21b.fnum(out["ad_start_sec"])))
            if int(v21b.fnum(out.get("video_id"))) in TEST_ID_SET:
                base_candidates.append(out)

    ocr_by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        vid = int(v21b.fnum(row.get("video_id")))
        if vid in TEST_ID_SET:
            row["_time"] = v21b.fnum(row.get("timestamp_sec"))
            ocr_by_video[vid].append(row)
    for values in ocr_by_video.values():
        values.sort(key=lambda row: row["_time"])

    phase_features, phase_events, phase_warnings = v21b.build_phase_features(base_candidates, ocr_by_video)
    feature_by_id = {row["base_candidate_id"]: row for row in phase_features}
    cfg = json.loads(V21B_CONFIG_PATH.read_text(encoding="utf-8"))
    variants = cfg.get("candidate_variants", [])
    variant = next((row for row in variants if row.get("candidate_id") == "v2_1b_ocr_phase_transition_light"), None)
    if variant is None:
        raise RuntimeError("v2_1b_ocr_phase_transition_light variant missing")
    outputs, rule_events, budget_events = v21b.apply_candidate_variant(variant, base_candidates, feature_by_id, base_config)

    write_csv_dict(PRED_PATH, outputs["predictions"])
    write_csv_dict(REVIEW_PATH, outputs["review_only_candidates"])
    write_csv_dict(PRUNED_PATH, outputs["overprediction_pruned_review_candidates"])
    write_csv_dict(OPEN_PATH, outputs["open_candidates"])

    stats = {
        "v2_1a_base_prediction_count": len(bucket_map["prediction"]),
        "v2_1a_base_review_count": len(bucket_map["review_only"]),
        "base_candidate_count": len(base_candidates),
        "v2_1b_prediction_count": len(outputs["predictions"]),
        "v2_1b_review_count": len(outputs["review_only_candidates"]),
        "v2_1b_pruned_count": len(outputs["overprediction_pruned_review_candidates"]),
        "v2_1b_open_count": len(outputs["open_candidates"]),
        "v2_1b_phase_feature_count": len(phase_features),
        "v2_1b_phase_event_count": len(phase_events),
        "v2_1b_rule_event_count": len(rule_events),
        "v2_1b_budget_event_count": len(budget_events),
        "v2_1b_phase_warnings": phase_warnings,
    }
    return stats


def schema_validation() -> tuple[dict[str, Any], bool]:
    pairs = {
        "predictions": (PRED_PATH, PRED_PATH),
        "review_candidates": (REVIEW_PATH, REVIEW_PATH),
        "pruned_candidates": (PRUNED_PATH, PRUNED_PATH),
        "open_candidates": (OPEN_PATH, OPEN_PATH),
        "ocr_candidate_sources": (OCR_SOURCE_PATH, OCR_SOURCE_PATH),
        "black_screen_features": (BLACK_FEATURE_PATH, BLACK_FEATURE_PATH),
    }
    validation: dict[str, Any] = {}
    all_ok = True
    for name, (dev_path, test_path) in pairs.items():
        if not dev_path.exists() or not test_path.exists():
            validation[name] = {"dev_exists": dev_path.exists(), "test_exists": test_path.exists(), "matches_development_required_columns": False}
            all_ok = False
            continue
        dev_cols = pd.read_csv(dev_path, nrows=0).columns.tolist()
        test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()
        # Review candidate에는 v2_1b 진단 column이 추가될 수 있으므로 test input column 누락만 확인한다.
        missing = [col for col in dev_cols if col not in test_cols]
        ok = not missing
        validation[name] = {
            "development_column_count": len(dev_cols),
            "test_column_count": len(test_cols),
            "missing_development_columns": missing,
            "extra_test_columns": [col for col in test_cols if col not in dev_cols],
            "matches_development_required_columns": ok,
        }
        all_ok = all_ok and ok
    return validation, all_ok


def coverage_validation() -> dict[str, Any]:
    artifacts = {
        "final_anchors": FINAL_ANCHOR_PATH,
        "ocr_schedule": OCR_SCHEDULE_PATH,
        "ocr_candidate_sources": OCR_SOURCE_PATH,
        "black_screen_features": BLACK_FEATURE_PATH,
        "fusion_input": FUSION_PATH,
        "predictions": PRED_PATH,
        "review_candidates": REVIEW_PATH,
        "pruned_candidates": PRUNED_PATH,
        "open_candidates": OPEN_PATH,
    }
    coverage = {}
    for name, path in artifacts.items():
        if not path.exists():
            coverage[name] = []
            continue
        df = pd.read_csv(path, usecols=lambda c: c == "video_id")
        coverage[name] = sorted(df["video_id"].dropna().astype(int).unique().tolist()) if "video_id" in df.columns else []
    combined_candidate_ids = sorted(set(coverage["predictions"]) | set(coverage["review_candidates"]) | set(coverage["pruned_candidates"]) | set(coverage["open_candidates"]))
    return {
        "coverage_by_artifact": coverage,
        "combined_candidate_video_ids": combined_candidate_ids,
        "test_video_coverage_complete": set(TEST_IDS).issubset(set(coverage["final_anchors"]))
        and set(TEST_IDS).issubset(set(coverage["ocr_candidate_sources"]))
        and set(TEST_IDS).issubset(set(coverage["black_screen_features"]))
        and set(TEST_IDS).issubset(set(combined_candidate_ids)),
        "missing_video_or_feature_list": [
            {"artifact": name, "missing_video_ids": sorted(set(TEST_IDS) - set(ids))}
            for name, ids in coverage.items()
            if name not in {"predictions", "pruned_candidates", "open_candidates"} and sorted(set(TEST_IDS) - set(ids))
        ],
    }


def refresh_latest_folder() -> list[str]:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for child in LATEST_DIR.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    files = [
        TRANSNET_ANCHOR_PATH,
        FINAL_ANCHOR_PATH,
        OCR_SCHEDULE_PATH,
        OCR_ANCHOR_FEATURE_PATH,
        OCR_TIMELINE_FEATURE_PATH,
        FUSION_PATH,
        PRED_PATH,
        REVIEW_PATH,
        PRUNED_PATH,
        OPEN_PATH,
        OCR_SOURCE_PATH,
        BLACK_FEATURE_PATH,
        MANIFEST_OUT_PATH,
        READINESS_OUT_PATH,
        REPORT_JSON_PATH,
        SUMMARY_MD_PATH,
    ]
    copied = []
    for path in files:
        if path.exists() and path.is_file():
            target = LATEST_DIR / path.name
            shutil.copy2(path, target)
            copied.append(target.name)
    return copied


def main() -> None:
    started = now_iso()
    warnings: list[str] = []
    errors: list[str] = []
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    before = {
        "final_anchor_count": row_count(FINAL_ANCHOR_PATH),
        "ocr_schedule_count": row_count(OCR_SCHEDULE_PATH),
        "ocr_source_count": row_count(OCR_SOURCE_PATH),
        "ocr_anchor_feature_count": row_count(OCR_ANCHOR_FEATURE_PATH),
        "fusion_count": row_count(FUSION_PATH),
        "candidate_counts": {
            "predictions": row_count(PRED_PATH),
            "review_candidates": row_count(REVIEW_PATH),
            "pruned_candidates": row_count(PRUNED_PATH),
            "open_candidates": row_count(OPEN_PATH),
        },
    }
    final_before_df = pd.read_csv(FINAL_ANCHOR_PATH) if FINAL_ANCHOR_PATH.exists() else pd.DataFrame()
    before["final_anchor_source_composition"] = source_composition(final_before_df)
    LOGGER.log(f"[STEP] Before counts: {before}")

    _, mapping_df, split_check = load_split_and_mapping()
    params = load_transnet_parameters(warnings)
    if TRANSNET_ANCHOR_PATH.exists():
        transnet_df = pd.read_csv(TRANSNET_ANCHOR_PATH)
        if sorted(transnet_df.get("video_id", pd.Series(dtype=int)).dropna().astype(int).unique().tolist()) != TEST_IDS:
            transnet_df, transnet_stats = run_transnetv2(mapping_df, params, warnings)
        else:
            transnet_stats = {
                "reused_existing_transnetv2_anchor_file_after_successful_generation": True,
                "processed_videos": TEST_IDS,
                "failed_videos": [],
                "videos_with_no_transnetv2_anchor_detected": [],
                "conservative_anchor_count": int(len(transnet_df)),
                "conservative_anchor_count_by_video": {str(k): int(v) for k, v in transnet_df["video_id"].astype(int).value_counts().sort_index().to_dict().items()},
            }
            LOGGER.log("[STEP] Reusing existing TransNetV2 conservative test-set anchor CSV")
    else:
        transnet_df, transnet_stats = run_transnetv2(mapping_df, params, warnings)
    candidates, canonical, source_stats = load_scene_candidates(mapping_df, transnet_df, warnings)
    final_df = build_final_anchors(candidates, canonical, mapping_df)
    schedule_df = build_ocr_schedule(final_df, mapping_df)
    if OCR_SOURCE_PATH.exists() and OCR_ANCHOR_FEATURE_PATH.exists() and OCR_TIMELINE_FEATURE_PATH.exists():
        existing_frame = pd.read_csv(OCR_SOURCE_PATH)
        existing_anchor = pd.read_csv(OCR_ANCHOR_FEATURE_PATH)
        existing_timeline = pd.read_csv(OCR_TIMELINE_FEATURE_PATH)
        if len(existing_frame) == len(schedule_df) and len(existing_anchor) == len(final_df):
            frame_df, ocr_anchor_df, timeline_df = existing_frame, existing_anchor, existing_timeline
            ocr_stats = {
                "reused_existing_ocr_after_successful_generation": True,
                "ocr_frame_rows": int(len(frame_df)),
                "ocr_status_counts": {str(k): int(v) for k, v in frame_df["ocr_status"].fillna("").value_counts().to_dict().items()} if "ocr_status" in frame_df else {},
                "ocr_anchor_feature_rows": int(len(ocr_anchor_df)),
                "ocr_timeline_feature_rows": int(len(timeline_df)),
            }
            LOGGER.log("[STEP] Reusing existing refreshed OCR source/features")
        else:
            frame_df, ocr_anchor_df, timeline_df, ocr_stats = build_or_refresh_ocr_features(schedule_df, mapping_df, final_df, warnings)
    else:
        frame_df, ocr_anchor_df, timeline_df, ocr_stats = build_or_refresh_ocr_features(schedule_df, mapping_df, final_df, warnings)
    black_df, black_stats = refresh_black_features(final_df, mapping_df)
    fusion_df, fusion_stats = build_fusion_input(final_df, mapping_df, ocr_anchor_df, timeline_df)
    candidate_stats = build_v21b_candidates(frame_df)
    schema, schema_ok = schema_validation()
    coverage = coverage_validation()

    after = {
        "final_anchor_count": len(final_df),
        "ocr_schedule_count": len(schedule_df),
        "ocr_source_count": len(frame_df),
        "ocr_anchor_feature_count": len(ocr_anchor_df),
        "fusion_count": len(fusion_df),
        "candidate_counts": {
            "predictions": row_count(PRED_PATH),
            "review_candidates": row_count(REVIEW_PATH),
            "pruned_candidates": row_count(PRUNED_PATH),
            "open_candidates": row_count(OPEN_PATH),
        },
        "final_anchor_source_composition": source_composition(final_df),
    }
    report = {
        "task": "refresh_test_set_features_with_transnetv2_conservative_anchor",
        "created_at": now_iso(),
        "started_at": started,
        "test_set_definition": "Test Set",
        "test_set_called_as": "test set",
        "test_set_video_ids": TEST_IDS,
        "split_check": {
            **split_check,
            "development_video_leakage_into_test_set": bool(set(TEST_IDS).intersection(DEVELOPMENT_IDS)),
            "development_overlap_video_ids": sorted(set(TEST_IDS).intersection(DEVELOPMENT_IDS)),
        },
        "transnetv2": {
            "params": params,
            **transnet_stats,
            "anchor_file": str(TRANSNET_ANCHOR_PATH),
        },
        "source_stats": source_stats,
        "before": before,
        "after": after,
        "ocr_stats": ocr_stats,
        "black_stats": black_stats,
        "fusion_stats": fusion_stats,
        "candidate_stats": candidate_stats,
        "validation": {
            "schema": schema,
            "schema_matches_development": schema_ok,
            **coverage,
            "old_v1_or_v2_4_fallback_used_as_final_input": False,
            "actual_label_used_for_transnetv2_anchor_generation": False,
            "actual_label_used_for_feature_generation": False,
            "actual_label_used_for_candidate_generation": False,
            "evaluation_run_executed": False,
            "Development_current_state_modified": False,
            "Development_latest_csv_modified": False,
        },
        "output_files": {
            "transnetv2_anchor_csv": str(TRANSNET_ANCHOR_PATH),
            "final_anchor_csv": str(FINAL_ANCHOR_PATH),
            "ocr_schedule_csv": str(OCR_SCHEDULE_PATH),
            "ocr_anchor_features_csv": str(OCR_ANCHOR_FEATURE_PATH),
            "ocr_timeline_features_csv": str(OCR_TIMELINE_FEATURE_PATH),
            "fusion_input_csv": str(FUSION_PATH),
            "predictions": str(PRED_PATH),
            "review_candidates": str(REVIEW_PATH),
            "pruned_candidates": str(PRUNED_PATH),
            "open_candidates": str(OPEN_PATH),
            "ocr_candidate_sources": str(OCR_SOURCE_PATH),
            "black_screen_features": str(BLACK_FEATURE_PATH),
            "feature_manifest": str(MANIFEST_OUT_PATH),
            "readiness_report_json": str(READINESS_OUT_PATH),
            "summary_md": str(SUMMARY_MD_PATH),
            "report_json": str(REPORT_JSON_PATH),
            "run_log": str(LOG_PATH),
        },
        "warnings": warnings,
        "errors": errors,
    }

    manifest = {
        "task": "refresh_test_set_features_with_transnetv2_conservative_anchor",
        "created_at": now_iso(),
        "test_set_definition": "Test Set",
        "test_set_video_ids": TEST_IDS,
        "input_feature_root": str(INPUT_ROOT),
        "feature_root": str(FEATURE_ROOT),
        "output_files": {
            "predictions": str(PRED_PATH),
            "review_candidates": str(REVIEW_PATH),
            "pruned_candidates": str(PRUNED_PATH),
            "open_candidates": str(OPEN_PATH),
            "ocr_candidate_sources": str(OCR_SOURCE_PATH),
            "black_screen_features": str(BLACK_FEATURE_PATH),
            "manifest": str(MANIFEST_OUT_PATH),
            "readiness_report": str(READINESS_OUT_PATH),
        },
        "supporting_generated_files": {
            "transnetv2_conservative_anchors": str(TRANSNET_ANCHOR_PATH),
            "final_anchors": str(FINAL_ANCHOR_PATH),
            "ocr_schedule": str(OCR_SCHEDULE_PATH),
            "ocr_anchor_features": str(OCR_ANCHOR_FEATURE_PATH),
            "ocr_timeline_features": str(OCR_TIMELINE_FEATURE_PATH),
            "fusion_input": str(FUSION_PATH),
        },
        "actual_label_used_for_feature_generation": False,
        "actual_label_used_for_candidate_generation": False,
        "old_v1_or_v2_4_fallback_used_as_final_input": False,
        "evaluation_run_executed": False,
        "notes": "v2.4 scene/audio files are upstream sources only; final test-set inputs are regenerated as v2.5/v2.1b files.",
    }
    readiness = {
        **report,
        "transnetv2_conservative_test_set_created": True,
        "transnetv2_conservative_anchor_file": str(TRANSNET_ANCHOR_PATH),
        "transnetv2_conservative_anchor_count": len(transnet_df),
        "transnetv2_video_coverage": sorted(transnet_df["video_id"].dropna().astype(int).unique().tolist()) if len(transnet_df) else [],
        "final_anchor_count_before": before["final_anchor_count"],
        "final_anchor_count_after": after["final_anchor_count"],
        "final_anchor_source_composition_before": before["final_anchor_source_composition"],
        "final_anchor_source_composition_after": after["final_anchor_source_composition"],
        "OCR_schedule_row_count_before": before["ocr_schedule_count"],
        "OCR_schedule_row_count_after": after["ocr_schedule_count"],
        "OCR_feature_row_count_before": before["ocr_source_count"],
        "OCR_feature_row_count_after": after["ocr_source_count"],
        "fusion_row_count_before": before["fusion_count"],
        "fusion_row_count_after": after["fusion_count"],
        "v2_1b_candidate_counts_before": before["candidate_counts"],
        "v2_1b_candidate_counts_after": after["candidate_counts"],
        "OCR_candidate_source_row_count_before": before["ocr_source_count"],
        "OCR_candidate_source_row_count_after": after["ocr_source_count"],
        "black_feature_row_count": len(black_df),
        "schema_matches_development": schema_ok,
        "test_video_coverage_complete": coverage["test_video_coverage_complete"],
        "old_v1_or_v2_4_fallback_used_as_final_input": False,
        "actual_label_used_for_transnetv2_anchor_generation": False,
        "actual_label_used_for_feature_generation": False,
        "actual_label_used_for_candidate_generation": False,
        "evaluation_run_executed": False,
        "Development_current_state_modified": False,
        "Development_latest_csv_modified": False,
    }
    write_json(MANIFEST_OUT_PATH, manifest)
    write_json(READINESS_OUT_PATH, readiness)
    write_json(REPORT_JSON_PATH, report)

    summary = f"""# Test Set TransNetV2 Anchor Refresh Summary

- task: refresh_test_set_features_with_transnetv2_conservative_anchor
- test set: Test Set
- test videos: {TEST_IDS}
- TransNetV2 conservative anchors: {len(transnet_df)}
- final anchors: {before['final_anchor_count']} -> {after['final_anchor_count']}
- OCR schedule rows: {before['ocr_schedule_count']} -> {after['ocr_schedule_count']}
- OCR candidate/source rows: {before['ocr_source_count']} -> {after['ocr_source_count']}
- fusion rows: {before['fusion_count']} -> {after['fusion_count']}
- v2_1b candidates before: {before['candidate_counts']}
- v2_1b candidates after: {after['candidate_counts']}
- schema matches Development required columns: {schema_ok}
- test video coverage complete: {coverage['test_video_coverage_complete']}
- actual labels used for feature/candidate generation: false
- frozen patch5 evaluation executed: false

This refresh adds test-set TransNetV2 conservative anchors using the same threshold/dedup policy used for Development and rebuilds downstream test-set feature inputs without modifying Development current_state or Development latest_csv.
"""
    SUMMARY_MD_PATH.write_text(summary, encoding="utf-8")
    copied = refresh_latest_folder()
    report["latest_test_feature_folder_refreshed"] = True
    report["latest_test_feature_folder"] = str(LATEST_DIR)
    report["latest_test_feature_files"] = copied
    readiness["latest_test_feature_folder_refreshed"] = True
    readiness["latest_test_feature_folder"] = str(LATEST_DIR)
    readiness["latest_test_feature_files"] = copied
    write_json(REPORT_JSON_PATH, report)
    write_json(READINESS_OUT_PATH, readiness)
    LOGGER.log(f"[DONE] latest files copied: {copied}")
    LOGGER.log(json.dumps({"status": "success", "report": str(REPORT_JSON_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
