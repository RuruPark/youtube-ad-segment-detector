#!/usr/bin/env python3
"""OpenCV 장면 전환 추출에 실패한 v2.3 영상 4개를 재시도한다."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")

WINDOW_LABELS_PATH = PROJECT_ROOT / "data/windows/window_labels_5s_v2_3.csv"
SEGMENTS_PATH = PROJECT_ROOT / "data/segments/label_interval_context_segments_v2_3.csv"
AD_SEGMENTS_PATH = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_3.csv"
MANIFEST_PATH = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
VIDEO_INPUT_DIR = PROJECT_ROOT / "data/raw/videos"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts/utils/check_cv_environment.py"

ORIGINAL_CANDIDATES_PATH = PROJECT_ROOT / "data/scene/opencv_scene_candidates_v2_3.csv"
ORIGINAL_CANDIDATES_MMSS_PATH = PROJECT_ROOT / "data/scene/opencv_scene_candidates_v2_3_mmss.csv"
ORIGINAL_AUDIT_PATH = PROJECT_ROOT / "data/scene/opencv_scene_candidate_boundary_audit_v2_3.csv"
ORIGINAL_REPORT_PATH = PROJECT_ROOT / "reports/opencv_scene_candidates_v2_3_report.json"
ORIGINAL_LOG_PATH = PROJECT_ROOT / "logs/opencv_scene_candidates_v2_3_run_log.txt"

SCENE_DIR = PROJECT_ROOT / "data/scene"
OUTPUT_DIAGNOSIS = SCENE_DIR / "opencv_failed4_decoding_diagnosis_v2_3.csv"
OUTPUT_RETRY_CANDIDATES = SCENE_DIR / "opencv_scene_candidates_v2_3_retry_failed4.csv"
OUTPUT_RETRY_CANDIDATES_MMSS = SCENE_DIR / "opencv_scene_candidates_v2_3_retry_failed4_mmss.csv"
OUTPUT_RETRY_SUMMARY = SCENE_DIR / "opencv_scene_retry_failed4_video_summary_v2_3.csv"
OUTPUT_MERGED_CANDIDATES = SCENE_DIR / "opencv_scene_candidates_v2_3_merged_retry.csv"
OUTPUT_MERGED_CANDIDATES_MMSS = SCENE_DIR / "opencv_scene_candidates_v2_3_merged_retry_mmss.csv"
OUTPUT_MERGED_AUDIT = SCENE_DIR / "opencv_scene_candidate_boundary_audit_v2_3_merged_retry.csv"

REPORT_PATH = PROJECT_ROOT / "reports/retry_opencv_scene_change_failed4_v2_3_report.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/retry_opencv_scene_change_failed4_v2_3_summary.md"
LOG_PATH = PROJECT_ROOT / "logs/retry_opencv_scene_change_failed4_v2_3_run_log.txt"
SCRIPT_PATH = PROJECT_ROOT / "scripts/scene/retry_opencv_scene_change_failed4_v2_3.py"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"
LATEST_README = LATEST_DIR / "README_latest_files.md"

ESTIMATED_RUNTIME = "약 8~15분"
RUNTIME_ESTIMATION_REASON = "실패 영상 4개에 대해 frame-index seek, POS_MSEC timestamp seek, sequential decoding fallback을 순차적으로 시도하고, 후보 병합 및 광고 boundary audit을 재계산해야 함."
HARDCODED_FAILED_FILENAMES = [
    "다시 6시에 일어나기로 마음먹은 브이로그️ _ 소비 소비 소비_ _ 한달만에 찾은 iPad.mp4",
    "드디어 최종합격_ 출근 전 일상 브이로그 _ 출근룩 쇼핑과 OOTD, 한강뷰 카페 _ 건강검진 (나말고 고양이).mp4",
    "떨어졌던 회사,, 재도전 합니다 _ 면접스터디 가입(더보기TIP), 만다라트 계획표 _ 왕큰 김치볶음밥 밀프랩 _ 서울 브이로그.mp4",
    "집구석 대청소 후 서울 혼놀하는 브이로그 _ 직장인 도시락 13인분 밀프랩 해놓기 _ 48시간 착붙 일상.mp4",
]

SAMPLE_FPS = 1.0
RESIZE_SIZE = (320, 180)
MIN_SCORE_FLOOR = 0.35
STD_MULTIPLIER = 2.5
PERCENTILE_THRESHOLD = 95.0
DEDUP_TOLERANCE_SEC = 2.0
BOUNDARY_TOLERANCE_3S = 3.0
BOUNDARY_TOLERANCE_5S = 5.0
SEQUENTIAL_DIAG_MAX_FRAMES = 100
RUN_LOG: list[str] = []


try:
    cv2.setLogLevel(0)
except Exception:
    pass


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_LOG.append(f"[{stamp}] {message}")
    print(message, flush=True)


def step(number: int, message: str) -> None:
    log(f"[STEP {number}/10] {message}")


def ensure_inside_project(path: Path) -> Path:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise RuntimeError(f"Refusing to write outside project root: {resolved}")
    return path


def ensure_dirs() -> None:
    for path in [SCENE_DIR, PROJECT_ROOT / "reports", PROJECT_ROOT / "logs", PROJECT_ROOT / "scripts/scene", LATEST_DIR]:
        ensure_inside_project(path).mkdir(parents=True, exist_ok=True)


def clean_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return ""
        rounded = round(float(value), 6)
        if abs(rounded - round(rounded)) < 1e-9:
            return int(round(rounded))
        return rounded
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def mmss_floor(seconds: Any) -> str:
    sec = to_float(seconds)
    if sec is None:
        return ""
    total = int(math.floor(sec))
    return f"{total // 60:02d}분 {total % 60:02d}초"


def mmss_round(seconds: Any) -> str:
    sec = to_float(seconds)
    if sec is None:
        return ""
    total = int(round(sec))
    return f"{total // 60:02d}분 {total % 60:02d}초"


def mmss_colon(seconds: Any) -> str:
    sec = to_float(seconds)
    if sec is None:
        return ""
    total = int(round(sec))
    return f"{total // 60}:{total % 60:02d}"


def readable_duration(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}분 {total % 60}초"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_inside_project(path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt_value(row.get(column, "")) for column in columns})


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def verify_cv_environment() -> tuple[bool, str, list[Any]]:
    warnings: list[Any] = []
    executable = sys.executable
    in_cv = "/envs/cv/" in executable or executable.endswith("/envs/cv/bin/python") or executable.endswith("/envs/cv/bin/python3.10")
    if CHECK_ENV_SCRIPT.exists():
        cmd = ["conda", "run", "-n", "cv", "python", str(CHECK_ENV_SCRIPT)]
        log("Command: " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        log("cv check stdout: " + result.stdout.strip().replace("\n", " | "))
        if result.stderr.strip():
            log("cv check stderr: " + result.stderr.strip().replace("\n", " | "))
        if result.returncode != 0:
            warnings.append("cv_environment_check_failed")
            return False, executable, warnings
    if not in_cv:
        warnings.append("current_python_executable_not_in_cv")
    return in_cv, executable, warnings


def open_capture(path: Path) -> cv2.VideoCapture:
    try:
        return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])
    except Exception:
        return cv2.VideoCapture(str(path))


def read_frame_by_index(path: Path, frame_index: int) -> bool:
    cap = open_capture(path)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, _frame = cap.read()
    cap.release()
    return bool(ok)


def read_frame_by_msec(path: Path, timestamp_sec: float) -> bool:
    cap = open_capture(path)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
    ok, _frame = cap.read()
    cap.release()
    return bool(ok)


def sequential_read_count(path: Path, max_frames: int = SEQUENTIAL_DIAG_MAX_FRAMES) -> int:
    cap = open_capture(path)
    if not cap.isOpened():
        return 0
    count = 0
    for _ in range(max_frames):
        ok, _frame = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def preprocess_frame(frame: np.ndarray) -> dict[str, Any]:
    resized = cv2.resize(frame, RESIZE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    edges = cv2.Canny(gray, 80, 160)
    return {
        "gray": gray.astype(np.float32),
        "hist": hist.astype(np.float32),
        "brightness": float(gray.mean()),
        "edges": edges.astype(np.float32) / 255.0,
    }


def global_ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    a = gray_a.astype(np.float32)
    b = gray_b.astype(np.float32)
    c1 = 6.5025
    c2 = 58.5225
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(((a - mu_a) ** 2).mean())
    var_b = float(((b - mu_b) ** 2).mean())
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if denominator == 0:
        return 1.0
    return max(-1.0, min(1.0, ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / denominator))


def compare_features(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    hist_score = float(cv2.compareHist(previous["hist"], current["hist"], cv2.HISTCMP_BHATTACHARYYA))
    brightness_score = abs(current["brightness"] - previous["brightness"]) / 255.0
    ssim_score = (1.0 - global_ssim(previous["gray"], current["gray"])) / 2.0
    edge_score = float(np.mean(np.abs(current["edges"] - previous["edges"])))
    score = 0.45 * hist_score + 0.20 * brightness_score + 0.20 * ssim_score + 0.15 * edge_score
    return {
        "scene_change_score": max(0.0, min(1.0, score)),
        "hsv_hist_score": max(0.0, min(1.0, hist_score)),
        "brightness_score": max(0.0, min(1.0, brightness_score)),
        "ssim_change_score": max(0.0, min(1.0, ssim_score)),
        "edge_change_score": max(0.0, min(1.0, edge_score)),
    }


def score_from_frames(sampled: list[tuple[float, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_features: dict[str, Any] | None = None
    previous_time: float | None = None
    for sample_index, (timestamp, frame) in enumerate(sampled):
        current_features = preprocess_frame(frame)
        if previous_features is not None and previous_time is not None:
            rows.append(
                {
                    **compare_features(previous_features, current_features),
                    "sample_index": sample_index,
                    "candidate_time_sec": float(timestamp),
                    "prev_sample_time_sec": float(previous_time),
                    "next_sample_time_sec": float(timestamp),
                    "sample_interval_sec": float(timestamp) - float(previous_time),
                }
            )
        previous_features = current_features
        previous_time = float(timestamp)
    return rows


def sample_frame_index(path: Path, duration: float, fps: float) -> tuple[list[tuple[float, np.ndarray]], int]:
    cap = open_capture(path)
    sampled: list[tuple[float, np.ndarray]] = []
    read_count = 0
    if not cap.isOpened():
        return sampled, read_count
    for timestamp in np.arange(0.0, duration, 1.0 / SAMPLE_FPS):
        frame_index = int(round(float(timestamp) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = cap.read()
        if ok:
            sampled.append((float(timestamp), frame))
            read_count += 1
    cap.release()
    return sampled, read_count


def sample_pos_msec(path: Path, duration: float) -> tuple[list[tuple[float, np.ndarray]], int]:
    cap = open_capture(path)
    sampled: list[tuple[float, np.ndarray]] = []
    read_count = 0
    if not cap.isOpened():
        return sampled, read_count
    for timestamp in np.arange(0.0, duration, 1.0 / SAMPLE_FPS):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = cap.read()
        if ok:
            sampled.append((float(timestamp), frame))
            read_count += 1
    cap.release()
    return sampled, read_count


def sample_sequential(path: Path, duration: float, fps: float) -> tuple[list[tuple[float, np.ndarray]], int]:
    cap = open_capture(path)
    sampled: list[tuple[float, np.ndarray]] = []
    read_count = 0
    if not cap.isOpened():
        return sampled, read_count
    next_sample_time = 0.0
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        read_count += 1
        timestamp = frame_index / fps if fps > 0 else 0.0
        if timestamp + (0.5 / max(fps, 1.0)) >= next_sample_time:
            sampled.append((float(next_sample_time), frame))
            next_sample_time += 1.0 / SAMPLE_FPS
            while next_sample_time <= timestamp:
                next_sample_time += 1.0 / SAMPLE_FPS
        frame_index += 1
        if timestamp > duration + 1.0:
            break
    cap.release()
    return sampled, read_count


def threshold_for_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {
            "threshold": "",
            "score_mean": "",
            "score_std": "",
            "score_p50": "",
            "score_p90": "",
            "score_p95": "",
            "score_p99": "",
        }
    values = np.array([row["scene_change_score"] for row in scores], dtype=np.float32)
    return {
        "threshold": max(MIN_SCORE_FLOOR, float(values.mean()) + STD_MULTIPLIER * float(values.std()), float(np.percentile(values, PERCENTILE_THRESHOLD))),
        "score_mean": float(values.mean()),
        "score_std": float(values.std()),
        "score_p50": float(np.percentile(values, 50)),
        "score_p90": float(np.percentile(values, 90)),
        "score_p95": float(np.percentile(values, 95)),
        "score_p99": float(np.percentile(values, 99)),
    }


def select_candidates(scores: list[dict[str, Any]], threshold: float | str) -> list[dict[str, Any]]:
    if threshold == "" or not scores:
        return []
    selected = [row | {"candidate_type": "threshold_candidate"} for row in scores if float(row["scene_change_score"]) >= float(threshold)]
    if not selected:
        top = sorted(scores, key=lambda row: float(row["scene_change_score"]), reverse=True)[:5]
        selected = [row | {"candidate_type": "topk_fallback"} for row in top]
    return dedup_candidates(selected)


def dedup_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[clean_value(row.get("video_id"))].append(row)
    kept_all: list[dict[str, Any]] = []
    for _video_id, items in by_video.items():
        items = sorted(items, key=lambda row: (-float(row.get("scene_change_score") or 0), float(row.get("candidate_time_sec") or 0)))
        kept: list[dict[str, Any]] = []
        for row in items:
            timestamp = float(row.get("candidate_time_sec") or 0)
            if any(abs(timestamp - float(existing.get("candidate_time_sec") or 0)) <= DEDUP_TOLERANCE_SEC for existing in kept):
                continue
            kept.append(row)
        kept_all.extend(sorted(kept, key=lambda row: float(row.get("candidate_time_sec") or 0)))
    return sorted(kept_all, key=lambda row: (clean_value(row.get("video_id")), float(row.get("candidate_time_sec") or 0)))


def find_failed_videos(original_report: dict[str, Any], manifest: pd.DataFrame, warnings: list[Any]) -> list[dict[str, Any]]:
    report_filenames = set(original_report.get("videos_without_candidates_list") or [])
    per_video_zero = {
        clean_value(row.get("video_filename"))
        for row in original_report.get("per_video_scene_stats", [])
        if int(row.get("score_count") or 0) == 0 or int(row.get("candidate_count") or 0) == 0
    }
    report_filenames |= per_video_zero
    hardcoded = set(HARDCODED_FAILED_FILENAMES)
    if report_filenames != hardcoded:
        warnings.append(
            {
                "failed_video_list_discrepancy": {
                    "report_only": sorted(report_filenames - hardcoded),
                    "hardcoded_only": sorted(hardcoded - report_filenames),
                }
            }
        )
    filenames = sorted(report_filenames | hardcoded)
    manifest_by_filename = {clean_value(row.get("video_filename")): row.to_dict() for _, row in manifest.iterrows()}
    video_paths = {path.name: path for pattern in ("*.mp4", "*.MP4") for path in VIDEO_INPUT_DIR.rglob(pattern)}
    rows: list[dict[str, Any]] = []
    for filename in filenames:
        manifest_row = manifest_by_filename.get(filename, {})
        path = video_paths.get(filename) or Path(clean_value(manifest_row.get("video_path")))
        rows.append(
            {
                "video_id": clean_value(manifest_row.get("label_mapping_video_id")) or clean_value(manifest_row.get("video_id")),
                "video_title": clean_value(manifest_row.get("label_mapping_video_title"))
                or clean_value(manifest_row.get("matched_label_video_title"))
                or clean_value(manifest_row.get("video_title")),
                "video_filename": filename,
                "video_path": str(path),
            }
        )
    return rows


def diagnose_video(video: dict[str, Any]) -> dict[str, Any]:
    path = Path(clean_value(video.get("video_path")))
    cap = open_capture(path)
    cap_is_opened = bool(cap.isOpened())
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) if cap_is_opened else 0.0
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if cap_is_opened else 0.0
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if cap_is_opened else 0.0
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if cap_is_opened else 0.0
    duration = frame_count / fps if fps > 0 and frame_count > 0 else None
    if cap:
        cap.release()
    mid_frame = int(frame_count * 0.5) if frame_count else 0
    late_frame = int(max(0, frame_count - max(fps, 1))) if frame_count else 0
    mid_sec = float(duration * 0.5) if duration else 0.0
    late_sec = max(0.0, float(duration or 0) - 1.0)
    first_ok = read_frame_by_index(path, 0)
    random_mid_ok = read_frame_by_index(path, mid_frame)
    random_late_ok = read_frame_by_index(path, late_frame)
    pos0_ok = read_frame_by_msec(path, 0.0)
    pos5_ok = read_frame_by_msec(path, 5.0)
    pos_mid_ok = read_frame_by_msec(path, mid_sec)
    pos_late_ok = read_frame_by_msec(path, late_sec)
    sequential_count = sequential_read_count(path)
    sequential_ok = sequential_count > 1
    if not cap_is_opened:
        diagnosis = "capture_open_failed"
        recommended = "ffmpeg_fallback_needed"
    elif not first_ok and not pos0_ok and not sequential_ok:
        diagnosis = "metadata_ok_but_frame_read_failed"
        recommended = "ffmpeg_fallback_needed"
    elif (pos0_ok or pos5_ok or pos_mid_ok) and not random_mid_ok:
        diagnosis = "frame_index_seek_failed_but_pos_msec_ok"
        recommended = "pos_msec_seek"
    elif sequential_ok and not (pos0_ok or pos5_ok or pos_mid_ok):
        diagnosis = "frame_index_seek_failed_but_sequential_ok"
        recommended = "sequential_decode"
    elif first_ok or random_mid_ok:
        diagnosis = "retry_not_needed_or_frame_index_available"
        recommended = "frame_index_seek_retry"
    else:
        diagnosis = "all_opencv_methods_failed"
        recommended = "ffmpeg_fallback_needed"
    return {
        "video_id": clean_value(video.get("video_id")),
        "video_filename": clean_value(video.get("video_filename")),
        "video_path": str(path),
        "duration_sec": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "cap_is_opened": bool_text(cap_is_opened),
        "first_frame_read_ok": bool_text(first_ok),
        "random_seek_mid_read_ok": bool_text(random_mid_ok),
        "random_seek_late_read_ok": bool_text(random_late_ok),
        "pos_msec_0_read_ok": bool_text(pos0_ok),
        "pos_msec_5_read_ok": bool_text(pos5_ok),
        "pos_msec_mid_read_ok": bool_text(pos_mid_ok),
        "pos_msec_late_read_ok": bool_text(pos_late_ok),
        "sequential_read_ok": bool_text(sequential_ok),
        "sequential_read_frame_count": sequential_count,
        "diagnosis": diagnosis,
        "recommended_retry_method": recommended,
    }


def score_retry_video(video: dict[str, Any], diagnosis: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(clean_value(video.get("video_path")))
    duration = to_float(diagnosis.get("duration_sec"))
    fps = to_float(diagnosis.get("fps"))
    attempts: list[str] = []
    if duration is None or duration <= 0 or fps is None or fps <= 0:
        return [], {
            "retry_method_used": "none",
            "sample_count": 0,
            "read_frame_count": 0,
            "score_count": 0,
            "retry_success": "false",
            "ffmpeg_fallback_needed": "true",
            "warning": "duration_or_fps_invalid",
        }
    method_results: list[tuple[str, list[tuple[float, np.ndarray]], int]] = []
    for method_name, sampler in [
        ("frame_index_seek_retry", lambda: sample_frame_index(path, duration, fps)),
        ("pos_msec_seek", lambda: sample_pos_msec(path, duration)),
        ("sequential_decode", lambda: sample_sequential(path, duration, fps)),
    ]:
        attempts.append(method_name)
        sampled, read_count = sampler()
        scores = score_from_frames(sampled)
        method_results.append((method_name, sampled, read_count))
        if scores:
            threshold_meta = threshold_for_scores(scores)
            candidates = select_candidates(scores, threshold_meta["threshold"])
            values = np.array([row["scene_change_score"] for row in scores], dtype=np.float32)
            for rank, candidate in enumerate(sorted(candidates, key=lambda row: float(row["scene_change_score"]), reverse=True), start=1):
                candidate["video_id"] = clean_value(video.get("video_id"))
                candidate["video_title"] = clean_value(video.get("video_title"))
                candidate["video_filename"] = clean_value(video.get("video_filename"))
                candidate["video_path"] = clean_value(video.get("video_path"))
                candidate["threshold"] = threshold_meta["threshold"]
                candidate["method_used"] = method_name
                candidate["retry_attempts"] = ";".join(attempts)
                candidate["score_rank_in_video"] = rank
                candidate["score_percentile_in_video"] = float((values <= float(candidate["scene_change_score"])).mean())
                candidate["fps"] = fps
                candidate["duration_sec"] = duration
            summary = {
                "retry_method_used": method_name,
                "sample_count": len(sampled),
                "read_frame_count": read_count,
                "score_count": len(scores),
                **threshold_meta,
                "candidate_count": len(candidates),
                "retry_success": "true",
                "ffmpeg_fallback_needed": "false",
                "warning": "" if len(candidates) else "score_count_recovered_but_no_threshold_candidates",
            }
            return candidates, summary
    last_method, sampled, read_count = method_results[-1] if method_results else ("none", [], 0)
    return [], {
        "retry_method_used": last_method,
        "sample_count": len(sampled),
        "read_frame_count": read_count,
        "score_count": 0,
        "score_mean": "",
        "score_std": "",
        "score_p50": "",
        "score_p90": "",
        "score_p95": "",
        "score_p99": "",
        "threshold": "",
        "candidate_count": 0,
        "retry_success": "false",
        "ffmpeg_fallback_needed": "true",
        "warning": "all_opencv_retry_methods_score_count_zero",
    }


def nearest_window(windows: pd.DataFrame, video_id: str, timestamp: float) -> str:
    video_windows = windows[windows["video_id"].map(clean_value).eq(video_id)]
    if video_windows.empty:
        return ""
    starts = video_windows["window_start_sec"].astype(float)
    ends = video_windows["window_end_sec"].astype(float)
    matched = video_windows[(starts <= timestamp) & (timestamp < ends)]
    if matched.empty:
        matched = video_windows[(starts <= timestamp) & np.isclose(ends, timestamp)]
    return clean_value(matched.iloc[0].get("window_id")) if not matched.empty else ""


def nearest_boundary(ad_segments: pd.DataFrame, video_id: str, timestamp: float) -> dict[str, Any]:
    video_ads = ad_segments[ad_segments["video_id"].map(clean_value).eq(video_id)]
    best: tuple[float, str, float, pd.Series] | None = None
    for _, row in video_ads.iterrows():
        start = to_float(row.get("segment_start_sec"))
        end = to_float(row.get("segment_end_sec"))
        if start is None or end is None:
            continue
        for boundary_type, boundary_sec in [("ad_start", start), ("ad_end", end)]:
            distance = abs(timestamp - boundary_sec)
            if best is None or distance < best[0]:
                best = (distance, boundary_type, boundary_sec, row)
    if best is None:
        return {}
    return {
        "nearest_ad_interval_id": clean_value(best[3].get("ad_interval_id")),
        "nearest_ad_boundary_type": best[1],
        "nearest_ad_boundary_sec": best[2],
        "nearest_ad_boundary_mmss": mmss_round(best[2]),
        "distance_to_nearest_ad_boundary_sec": best[0],
        "is_near_ad_start_5s": bool_text(best[1] == "ad_start" and best[0] <= BOUNDARY_TOLERANCE_5S),
        "is_near_ad_end_5s": bool_text(best[1] == "ad_end" and best[0] <= BOUNDARY_TOLERANCE_5S),
        "is_near_any_ad_boundary_5s": bool_text(best[0] <= BOUNDARY_TOLERANCE_5S),
    }


def enrich_retry_candidates(candidates: list[dict[str, Any]], windows: pd.DataFrame, ad_segments: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: (clean_value(row.get("video_id")), float(row.get("candidate_time_sec") or 0))):
        timestamp = float(candidate["candidate_time_sec"])
        video_id = clean_value(candidate.get("video_id"))
        boundary = nearest_boundary(ad_segments, video_id, timestamp)
        rows.append(
            {
                "video_id": video_id,
                "video_title": clean_value(candidate.get("video_title")),
                "video_filename": clean_value(candidate.get("video_filename")),
                "video_path": clean_value(candidate.get("video_path")),
                "candidate_time_sec": timestamp,
                "candidate_time_mmss": mmss_round(timestamp),
                "candidate_time_mmss_floor": mmss_floor(timestamp),
                "candidate_time_mmss_round": mmss_round(timestamp),
                "scene_change_score": candidate.get("scene_change_score"),
                "threshold": candidate.get("threshold"),
                "candidate_type": candidate.get("candidate_type", ""),
                "method_used": candidate.get("method_used", ""),
                "retry_attempts": candidate.get("retry_attempts", ""),
                "score_rank_in_video": candidate.get("score_rank_in_video", ""),
                "score_percentile_in_video": candidate.get("score_percentile_in_video", ""),
                "prev_sample_time_sec": candidate.get("prev_sample_time_sec", ""),
                "next_sample_time_sec": candidate.get("next_sample_time_sec", ""),
                "sample_interval_sec": candidate.get("sample_interval_sec", ""),
                "fps": candidate.get("fps", ""),
                "duration_sec": candidate.get("duration_sec", ""),
                "nearest_window_id": nearest_window(windows, video_id, timestamp),
                "nearest_ad_interval_id": boundary.get("nearest_ad_interval_id", ""),
                "nearest_ad_boundary_type": boundary.get("nearest_ad_boundary_type", ""),
                "nearest_ad_boundary_sec": boundary.get("nearest_ad_boundary_sec", ""),
                "nearest_ad_boundary_mmss": boundary.get("nearest_ad_boundary_mmss", ""),
                "distance_to_nearest_ad_boundary_sec": boundary.get("distance_to_nearest_ad_boundary_sec", ""),
                "is_near_ad_start_5s": boundary.get("is_near_ad_start_5s", "false"),
                "is_near_ad_end_5s": boundary.get("is_near_ad_end_5s", "false"),
                "is_near_any_ad_boundary_5s": boundary.get("is_near_any_ad_boundary_5s", "false"),
                "review_note": "",
            }
        )
    return rows


def retry_mmss_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "video_id": row.get("video_id", ""),
            "video_title": row.get("video_title", ""),
            "video_filename": row.get("video_filename", ""),
            "candidate_time_sec": row.get("candidate_time_sec", ""),
            "candidate_time_mmss_floor": row.get("candidate_time_mmss_floor", ""),
            "candidate_time_mmss_round": row.get("candidate_time_mmss_round", ""),
            "scene_change_score": row.get("scene_change_score", ""),
            "method_used": row.get("method_used", ""),
            "nearest_window_id": row.get("nearest_window_id", ""),
            "nearest_ad_interval_id": row.get("nearest_ad_interval_id", ""),
            "nearest_ad_boundary_type": row.get("nearest_ad_boundary_type", ""),
            "review_note": "",
        }
        for row in rows
    ]


def standardize_original_candidates(original: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in original.iterrows():
        timestamp = to_float(row.get("candidate_frame_sec"))
        if timestamp is None:
            continue
        rows.append(
            {
                "candidate_source": "original_v2_3",
                "video_id": clean_value(row.get("video_id")),
                "video_title": clean_value(row.get("video_title")),
                "video_filename": clean_value(row.get("video_filename")),
                "video_path": clean_value(row.get("video_path")),
                "candidate_time_sec": timestamp,
                "candidate_time_mmss_floor": mmss_floor(timestamp),
                "candidate_time_mmss_round": mmss_round(timestamp),
                "scene_change_score": to_float(row.get("scene_change_score")) or 0,
                "threshold": clean_value(row.get("candidate_threshold")),
                "candidate_type": "threshold_candidate",
                "method_used": clean_value(row.get("method_used")) or "frame_index_seek_v2_3",
                "nearest_window_id": clean_value(row.get("window_id")),
                "nearest_ad_interval_id": clean_value(row.get("nearest_ad_interval_id")),
                "nearest_ad_boundary_type": clean_value(row.get("nearest_ad_boundary")),
                "nearest_ad_boundary_sec": clean_value(row.get("nearest_ad_boundary_sec")),
                "nearest_ad_boundary_mmss": clean_value(row.get("nearest_ad_boundary_mmss")),
                "distance_to_nearest_ad_boundary_sec": clean_value(row.get("distance_to_nearest_ad_boundary_sec")),
            }
        )
    return rows


def merge_candidates(original_rows: list[dict[str, Any]], retry_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    retry_standard = [{**row, "candidate_source": "retry_failed4"} for row in retry_rows]
    all_rows = original_rows + retry_standard
    before = len(all_rows)
    merged = dedup_candidates(all_rows)
    for index, row in enumerate(merged, start=1):
        row["merged_candidate_id"] = f"{clean_value(row.get('video_id'))}_MRG{index:06d}"
    return merged, before - len(merged)


def count_in_segment(candidates: pd.DataFrame, video_id: str, start: float, end: float) -> int:
    if candidates.empty:
        return 0
    subset = candidates[candidates["video_id"].map(clean_value).eq(video_id)].copy()
    if subset.empty:
        return 0
    times = subset["candidate_time_sec"].astype(float)
    return int(((times >= start) & (times <= end)).sum())


def nearest_candidate(candidates: pd.DataFrame, video_id: str, boundary_sec: float) -> tuple[str, str, str]:
    subset = candidates[candidates["video_id"].map(clean_value).eq(video_id)].copy() if not candidates.empty else pd.DataFrame()
    if subset.empty:
        return "", "", ""
    subset["_distance"] = (subset["candidate_time_sec"].astype(float) - boundary_sec).abs()
    row = subset.sort_values(["_distance", "scene_change_score"], ascending=[True, False]).iloc[0]
    sec = float(row["candidate_time_sec"])
    return str(fmt_value(sec)), mmss_round(sec), str(fmt_value(float(row["_distance"])))


def boundary_audit(merged_rows: list[dict[str, Any]], ad_segments: pd.DataFrame, all_segments: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = pd.DataFrame(merged_rows)
    rows: list[dict[str, Any]] = []
    start_hit_3 = start_hit_5 = end_hit_3 = end_hit_5 = both_hit_5 = 0
    boundary_hit_5 = 0
    for _, ad in ad_segments.iterrows():
        video_id = clean_value(ad.get("video_id"))
        interval_id = clean_value(ad.get("ad_interval_id"))
        start = to_float(ad.get("segment_start_sec")) or 0.0
        end = to_float(ad.get("segment_end_sec")) or 0.0
        video_candidates = candidates[candidates["video_id"].map(clean_value).eq(video_id)].copy() if not candidates.empty else pd.DataFrame()
        if video_candidates.empty:
            start_hits_3 = start_hits_5 = end_hits_3 = end_hits_5 = pd.DataFrame()
        else:
            times = video_candidates["candidate_time_sec"].astype(float)
            start_hits_3 = video_candidates[(times - start).abs() <= BOUNDARY_TOLERANCE_3S]
            start_hits_5 = video_candidates[(times - start).abs() <= BOUNDARY_TOLERANCE_5S]
            end_hits_3 = video_candidates[(times - end).abs() <= BOUNDARY_TOLERANCE_3S]
            end_hits_5 = video_candidates[(times - end).abs() <= BOUNDARY_TOLERANCE_5S]
        if len(start_hits_3):
            start_hit_3 += 1
        if len(start_hits_5):
            start_hit_5 += 1
            boundary_hit_5 += 1
        if len(end_hits_3):
            end_hit_3 += 1
        if len(end_hits_5):
            end_hit_5 += 1
            boundary_hit_5 += 1
        if len(start_hits_5) and len(end_hits_5):
            both_hit_5 += 1
        start_nearest_sec, start_nearest_mmss, start_nearest_dist = nearest_candidate(candidates, video_id, start)
        end_nearest_sec, end_nearest_mmss, end_nearest_dist = nearest_candidate(candidates, video_id, end)
        context = all_segments[
            all_segments["video_id"].map(clean_value).eq(video_id)
            & all_segments["ad_interval_id"].map(clean_value).eq(interval_id)
        ]
        pre = context[context["segment_type"].map(clean_value).eq("pre_ad_start_10s")]
        post = context[context["segment_type"].map(clean_value).eq("post_ad_end_10s")]
        pre_count = count_in_segment(candidates, video_id, float(pre.iloc[0]["segment_start_sec"]), float(pre.iloc[0]["segment_end_sec"])) if not pre.empty else 0
        post_count = count_in_segment(candidates, video_id, float(post.iloc[0]["segment_start_sec"]), float(post.iloc[0]["segment_end_sec"])) if not post.empty else 0
        rows.append(
            {
                "ad_interval_id": interval_id,
                "video_id": video_id,
                "video_title": clean_value(ad.get("video_title")),
                "video_filename": clean_value(ad.get("video_filename")),
                "ad_start_sec": start,
                "ad_start_mmss": mmss_round(start),
                "ad_end_sec": end,
                "ad_end_mmss": mmss_round(end),
                "ad_start_hit_3s": bool_text(len(start_hits_3) > 0),
                "ad_start_hit_5s": bool_text(len(start_hits_5) > 0),
                "ad_end_hit_3s": bool_text(len(end_hits_3) > 0),
                "ad_end_hit_5s": bool_text(len(end_hits_5) > 0),
                "both_boundary_hit_5s": bool_text(len(start_hits_5) > 0 and len(end_hits_5) > 0),
                "nearest_candidate_to_start_sec": start_nearest_sec,
                "nearest_candidate_to_start_mmss": start_nearest_mmss,
                "distance_to_start_candidate_sec": start_nearest_dist,
                "nearest_candidate_to_end_sec": end_nearest_sec,
                "nearest_candidate_to_end_mmss": end_nearest_mmss,
                "distance_to_end_candidate_sec": end_nearest_dist,
                "candidate_count_in_ad_interval": count_in_segment(candidates, video_id, start, end),
                "candidate_count_in_pre10": pre_count,
                "candidate_count_in_post10": post_count,
                "audit_note": "audit_metric_only_not_performance_claim",
            }
        )
    total_intervals = len(ad_segments)
    total_boundaries = total_intervals * 2
    summary = {
        "total_ad_intervals": total_intervals,
        "total_boundaries": total_boundaries,
        "start_hit_3s_count": start_hit_3,
        "start_hit_5s_count": start_hit_5,
        "end_hit_3s_count": end_hit_3,
        "end_hit_5s_count": end_hit_5,
        "boundary_hit_5s_count": boundary_hit_5,
        "boundary_hit_5s_rate": boundary_hit_5 / total_boundaries if total_boundaries else 0.0,
        "both_boundary_hit_5s_count": both_hit_5,
    }
    return rows, summary


def clear_latest_and_copy(files: list[Path]) -> tuple[bool, list[str]]:
    expected = PROJECT_ROOT / "outputs/latest_for_chatgpt"
    if LATEST_DIR.resolve() != expected.resolve():
        return False, []
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for child in LATEST_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    copied: list[str] = []
    for src in files:
        if src.exists():
            dst = LATEST_DIR / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    descriptions = {
        "opencv_failed4_decoding_diagnosis_v2_3.csv": "실패 4개 영상 OpenCV decoding 진단",
        "opencv_scene_candidates_v2_3_retry_failed4.csv": "실패 4개 영상 retry scene-change 후보",
        "opencv_scene_candidates_v2_3_retry_failed4_mmss.csv": "retry 후보 mm:ss 변환",
        "opencv_scene_retry_failed4_video_summary_v2_3.csv": "영상별 retry score/candidate 요약",
        "opencv_scene_candidates_v2_3_merged_retry.csv": "기존 v2_3 후보와 retry 후보 병합",
        "opencv_scene_candidates_v2_3_merged_retry_mmss.csv": "병합 후보 mm:ss 변환",
        "opencv_scene_candidate_boundary_audit_v2_3_merged_retry.csv": "병합 후보 기준 광고 boundary audit",
        "retry_opencv_scene_change_failed4_v2_3_report.json": "retry 상세 report",
        "retry_opencv_scene_change_failed4_v2_3_summary.md": "retry 요약",
        "retry_opencv_scene_change_failed4_v2_3_run_log.txt": "retry 실행 log",
        "retry_opencv_scene_change_failed4_v2_3.py": "retry 재현 스크립트",
    }
    LATEST_README.write_text(
        "# latest_for_chatgpt files\n\n"
        "latest_for_chatgpt는 최신 작업 핵심 파일만 모아둔 복사본 경로이다. 원본 파일은 프로젝트 내부 원래 경로에 존재한다.\n\n"
        "이번 작업명: retry_opencv_scene_change_failed4_v2_3\n\n"
        "복사된 파일 목록과 목적:\n\n"
        + "\n".join(f"- `{Path(path).name}`: {descriptions.get(Path(path).name, 'retry output')}" for path in copied)
        + "\n\nmp4 영상 파일, 원본 xlsx, frame, model, checkpoint, cache 파일은 복사하지 않는다.\n",
        encoding="utf-8",
    )
    copied.append(str(LATEST_README))
    return True, copied


def make_summary(report: dict[str, Any]) -> str:
    warnings = report.get("warnings", [])
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- 주요 warning 없음"
    sub_agent_text = "\n".join(f"- {key}: {value}" for key, value in report.get("sub_agent_results", {}).items()) or "- sub agent 검증 대기"
    return f"""# retry_opencv_scene_change_failed4_v2_3 summary

## 작업 시간

- estimated runtime: {report.get('estimated_runtime')}
- start time: {report.get('start_time')}
- end time: {report.get('end_time')}
- actual runtime: {report.get('actual_runtime_readable')}

## Retry 결과

- retry 대상 영상: {report.get('failed_video_count_from_original')}
- retry 성공 영상: {report.get('retry_success_count')}
- 여전히 실패한 영상: {report.get('retry_failed_count')}
- method distribution: `{report.get('retry_methods_used_distribution')}`
- ffmpeg fallback needed: `{report.get('ffmpeg_fallback_needed_videos')}`

## Candidate 병합

- original candidate count: {report.get('original_candidate_count')}
- retry candidate count: {report.get('retry_candidate_count')}
- merged candidate count: {report.get('merged_candidate_count')}
- duplicate removed count: {report.get('duplicate_removed_count')}

## Boundary Audit

- before boundary hit ±5s: {report.get('boundary_hit_5s_before')} / rate {report.get('boundary_hit_5s_rate_before')}
- after boundary hit ±5s: {report.get('boundary_hit_5s_after')} / rate {report.get('boundary_hit_5s_rate_after')}
- hit rate change: {report.get('hit_rate_change')}

이 값은 scene-change 후보 audit용 수치이며 final performance claim이 아니다.

## Sub Agent Results

{sub_agent_text}

## Warning

{warning_text}

## 다음 작업

OCR feature 생성 또는 scene 후보와 visual/audio cue 결합 단계로 넘어갈 수 있다.
"""


def main() -> None:
    ensure_dirs()
    warnings: list[Any] = []
    errors: list[Any] = []
    start_dt = datetime.now().astimezone()
    start_time = start_dt.isoformat(timespec="seconds")
    started = time.time()
    log(f"작업 시작 전 예상 작업 시간: {ESTIMATED_RUNTIME}")
    log(f"예상 근거: {RUNTIME_ESTIMATION_REASON}")
    log(f"작업 시작 시각: {start_time}")

    step(1, "작업 시간 기록 및 cv 환경 확인")
    cv_ok, python_executable, cv_warnings = verify_cv_environment()
    warnings.extend(cv_warnings)
    if not cv_ok:
        errors.append("cv_environment_check_failed")

    windows = read_csv(WINDOW_LABELS_PATH)
    segments = read_csv(SEGMENTS_PATH)
    ad_segments = read_csv(AD_SEGMENTS_PATH)
    manifest = read_csv(MANIFEST_PATH)
    original_candidates = read_csv(ORIGINAL_CANDIDATES_PATH)
    original_report = load_json(ORIGINAL_REPORT_PATH)
    original_audit = read_csv(ORIGINAL_AUDIT_PATH)

    step(2, "기존 report/log에서 실패 영상 식별")
    failed_videos = find_failed_videos(original_report, manifest, warnings)
    log(f"failed_videos={len(failed_videos)}: {[video['video_filename'] for video in failed_videos]}")

    step(3, "실패 영상 decoding 진단")
    diagnosis_rows = [diagnose_video(video) for video in failed_videos]
    diagnosis_columns = [
        "video_id",
        "video_filename",
        "video_path",
        "duration_sec",
        "fps",
        "frame_count",
        "width",
        "height",
        "cap_is_opened",
        "first_frame_read_ok",
        "random_seek_mid_read_ok",
        "random_seek_late_read_ok",
        "pos_msec_0_read_ok",
        "pos_msec_5_read_ok",
        "pos_msec_mid_read_ok",
        "pos_msec_late_read_ok",
        "sequential_read_ok",
        "sequential_read_frame_count",
        "diagnosis",
        "recommended_retry_method",
    ]
    write_csv(OUTPUT_DIAGNOSIS, diagnosis_rows, diagnosis_columns)

    step(4, "frame-index seek 재확인")
    step(5, "POS_MSEC timestamp seek 재시도")
    step(6, "sequential decoding fallback 재시도")
    retry_raw_candidates: list[dict[str, Any]] = []
    video_summary_rows: list[dict[str, Any]] = []
    diagnosis_by_filename = {row["video_filename"]: row for row in diagnosis_rows}
    for video in failed_videos:
        filename = clean_value(video.get("video_filename"))
        log(f"Retrying {filename}")
        candidates, summary = score_retry_video(video, diagnosis_by_filename[filename])
        retry_raw_candidates.extend(candidates)
        enriched_summary = {
            "video_id": clean_value(video.get("video_id")),
            "video_filename": filename,
            "duration_sec": diagnosis_by_filename[filename].get("duration_sec", ""),
            "fps": diagnosis_by_filename[filename].get("fps", ""),
            **summary,
        }
        video_summary_rows.append(enriched_summary)
        log(f"Retry result {filename}: method={summary.get('retry_method_used')}, score_count={summary.get('score_count')}, candidate_count={summary.get('candidate_count')}")

    step(7, "retry candidate 및 mm:ss CSV 생성")
    retry_rows = enrich_retry_candidates(retry_raw_candidates, windows, ad_segments)
    retry_candidate_columns = [
        "video_id",
        "video_title",
        "video_filename",
        "video_path",
        "candidate_time_sec",
        "candidate_time_mmss",
        "candidate_time_mmss_floor",
        "candidate_time_mmss_round",
        "scene_change_score",
        "threshold",
        "candidate_type",
        "method_used",
        "retry_attempts",
        "score_rank_in_video",
        "score_percentile_in_video",
        "prev_sample_time_sec",
        "next_sample_time_sec",
        "sample_interval_sec",
        "fps",
        "duration_sec",
        "nearest_window_id",
        "nearest_ad_interval_id",
        "nearest_ad_boundary_type",
        "nearest_ad_boundary_sec",
        "nearest_ad_boundary_mmss",
        "distance_to_nearest_ad_boundary_sec",
        "is_near_ad_start_5s",
        "is_near_ad_end_5s",
        "is_near_any_ad_boundary_5s",
        "review_note",
    ]
    write_csv(OUTPUT_RETRY_CANDIDATES, retry_rows, retry_candidate_columns)
    write_csv(
        OUTPUT_RETRY_CANDIDATES_MMSS,
        retry_mmss_rows(retry_rows),
        [
            "video_id",
            "video_title",
            "video_filename",
            "candidate_time_sec",
            "candidate_time_mmss_floor",
            "candidate_time_mmss_round",
            "scene_change_score",
            "method_used",
            "nearest_window_id",
            "nearest_ad_interval_id",
            "nearest_ad_boundary_type",
            "review_note",
        ],
    )
    for summary in video_summary_rows:
        video_id = clean_value(summary.get("video_id"))
        video_candidates = [row for row in retry_rows if clean_value(row.get("video_id")) == video_id]
        summary["candidate_count_near_ad_start_5s"] = sum(1 for row in video_candidates if row.get("is_near_ad_start_5s") == "true")
        summary["candidate_count_near_ad_end_5s"] = sum(1 for row in video_candidates if row.get("is_near_ad_end_5s") == "true")
    write_csv(
        OUTPUT_RETRY_SUMMARY,
        video_summary_rows,
        [
            "video_id",
            "video_filename",
            "duration_sec",
            "fps",
            "retry_method_used",
            "sample_count",
            "read_frame_count",
            "score_count",
            "score_mean",
            "score_std",
            "score_p50",
            "score_p90",
            "score_p95",
            "score_p99",
            "threshold",
            "candidate_count",
            "candidate_count_near_ad_start_5s",
            "candidate_count_near_ad_end_5s",
            "retry_success",
            "ffmpeg_fallback_needed",
            "warning",
        ],
    )

    step(8, "기존 성공 candidate와 병합")
    original_rows = standardize_original_candidates(original_candidates)
    merged_rows, duplicate_removed_count = merge_candidates(original_rows, retry_rows)
    merged_columns = [
        "merged_candidate_id",
        "candidate_source",
        "video_id",
        "video_title",
        "video_filename",
        "video_path",
        "candidate_time_sec",
        "candidate_time_mmss_floor",
        "candidate_time_mmss_round",
        "scene_change_score",
        "threshold",
        "candidate_type",
        "method_used",
        "nearest_window_id",
        "nearest_ad_interval_id",
        "nearest_ad_boundary_type",
        "nearest_ad_boundary_sec",
        "nearest_ad_boundary_mmss",
        "distance_to_nearest_ad_boundary_sec",
    ]
    write_csv(OUTPUT_MERGED_CANDIDATES, merged_rows, merged_columns)
    write_csv(
        OUTPUT_MERGED_CANDIDATES_MMSS,
        [
            {
                "video_id": row.get("video_id"),
                "video_title": row.get("video_title"),
                "video_filename": row.get("video_filename"),
                "candidate_source": row.get("candidate_source"),
                "candidate_time_sec": row.get("candidate_time_sec"),
                "candidate_time_mmss_floor": row.get("candidate_time_mmss_floor"),
                "candidate_time_mmss_round": row.get("candidate_time_mmss_round"),
                "scene_change_score": row.get("scene_change_score"),
                "method_used": row.get("method_used"),
                "nearest_window_id": row.get("nearest_window_id"),
                "nearest_ad_interval_id": row.get("nearest_ad_interval_id"),
                "review_note": "",
            }
            for row in merged_rows
        ],
        [
            "video_id",
            "video_title",
            "video_filename",
            "candidate_source",
            "candidate_time_sec",
            "candidate_time_mmss_floor",
            "candidate_time_mmss_round",
            "scene_change_score",
            "method_used",
            "nearest_window_id",
            "nearest_ad_interval_id",
            "review_note",
        ],
    )

    step(9, "boundary audit 재계산 및 retry 전후 비교")
    merged_audit_rows, merged_audit_summary = boundary_audit(merged_rows, ad_segments, segments)
    write_csv(
        OUTPUT_MERGED_AUDIT,
        merged_audit_rows,
        [
            "ad_interval_id",
            "video_id",
            "video_title",
            "video_filename",
            "ad_start_sec",
            "ad_start_mmss",
            "ad_end_sec",
            "ad_end_mmss",
            "ad_start_hit_3s",
            "ad_start_hit_5s",
            "ad_end_hit_3s",
            "ad_end_hit_5s",
            "both_boundary_hit_5s",
            "nearest_candidate_to_start_sec",
            "nearest_candidate_to_start_mmss",
            "distance_to_start_candidate_sec",
            "nearest_candidate_to_end_sec",
            "nearest_candidate_to_end_mmss",
            "distance_to_end_candidate_sec",
            "candidate_count_in_ad_interval",
            "candidate_count_in_pre10",
            "candidate_count_in_post10",
            "audit_note",
        ],
    )

    retry_success_count = sum(1 for row in video_summary_rows if clean_value(row.get("retry_success")) == "true")
    retry_failed_count = len(video_summary_rows) - retry_success_count
    ffmpeg_fallback_needed_videos = [
        row["video_filename"] for row in video_summary_rows if clean_value(row.get("ffmpeg_fallback_needed")) == "true"
    ]
    retry_methods_distribution = dict(sorted(Counter(clean_value(row.get("retry_method_used")) for row in video_summary_rows).items()))
    retry_score_count_by_video = {row["video_filename"]: int(row.get("score_count") or 0) for row in video_summary_rows}
    retry_candidate_count_by_video = {row["video_filename"]: int(row.get("candidate_count") or 0) for row in video_summary_rows}
    original_boundary_hit_before = int(original_report.get("boundary_hit_count_within_5s") or original_audit["hit_within_5s"].eq("true").sum())
    original_boundary_rate_before = float(original_report.get("boundary_hit_rate_within_5s") or 0.0)
    boundary_hit_after = int(merged_audit_summary["boundary_hit_5s_count"])
    boundary_rate_after = float(merged_audit_summary["boundary_hit_5s_rate"])

    end_dt = datetime.now().astimezone()
    end_time = end_dt.isoformat(timespec="seconds")
    elapsed = time.time() - started
    log(f"작업 종료 시각: {end_time}")
    log(f"실제 작업 시간: {readable_duration(elapsed)}")

    step(10, "report/summary/log/latest 갱신")
    generated_files = [
        OUTPUT_DIAGNOSIS,
        OUTPUT_RETRY_CANDIDATES,
        OUTPUT_RETRY_CANDIDATES_MMSS,
        OUTPUT_RETRY_SUMMARY,
        OUTPUT_MERGED_CANDIDATES,
        OUTPUT_MERGED_CANDIDATES_MMSS,
        OUTPUT_MERGED_AUDIT,
        REPORT_PATH,
        SUMMARY_PATH,
        LOG_PATH,
        SCRIPT_PATH,
    ]
    report = {
        "project_root": str(PROJECT_ROOT),
        "estimated_runtime": ESTIMATED_RUNTIME,
        "runtime_estimation_reason": RUNTIME_ESTIMATION_REASON,
        "start_time": start_time,
        "end_time": end_time,
        "actual_runtime_seconds": elapsed,
        "actual_runtime_readable": readable_duration(elapsed),
        "cv_environment_checked": cv_ok,
        "python_executable": python_executable,
        "old_project_modified": False,
        "failed_video_count_from_original": len(failed_videos),
        "failed_video_ids": [video["video_id"] for video in failed_videos],
        "failed_video_filenames": [video["video_filename"] for video in failed_videos],
        "retry_success_count": retry_success_count,
        "retry_failed_count": retry_failed_count,
        "retry_methods_used_distribution": retry_methods_distribution,
        "retry_score_count_by_video": retry_score_count_by_video,
        "retry_candidate_count_by_video": retry_candidate_count_by_video,
        "ffmpeg_fallback_needed_videos": ffmpeg_fallback_needed_videos,
        "original_candidate_count": int(len(original_rows)),
        "retry_candidate_count": int(len(retry_rows)),
        "merged_candidate_count": int(len(merged_rows)),
        "duplicate_removed_count": int(duplicate_removed_count),
        "boundary_hit_5s_before": original_boundary_hit_before,
        "boundary_hit_5s_after": boundary_hit_after,
        "boundary_hit_5s_rate_before": original_boundary_rate_before,
        "boundary_hit_5s_rate_after": boundary_rate_after,
        "hit_rate_change": boundary_rate_after - original_boundary_rate_before,
        **merged_audit_summary,
        "mmss_conversion_rule": "candidate_time_mmss_floor and candidate_time_mmss_round both provided; report comparisons use round display when one value is needed",
        "generated_files": [str(path) for path in generated_files],
        "sub_agent_results": {
            "sub_agent_1_decoding": "PENDING_EXTERNAL_REVIEW",
            "sub_agent_2_candidate_score": "PENDING_EXTERNAL_REVIEW",
            "sub_agent_3_boundary_output": "PENDING_EXTERNAL_REVIEW",
        },
        "warnings": warnings,
        "errors": errors,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(make_summary(report), encoding="utf-8")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")

    latest_ok, latest_files = clear_latest_and_copy(generated_files)
    report["latest_for_chatgpt_updated"] = latest_ok
    report["latest_for_chatgpt_files"] = latest_files
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(make_summary(report), encoding="utf-8")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    if latest_ok:
        for src in [REPORT_PATH, SUMMARY_PATH, LOG_PATH]:
            shutil.copy2(src, LATEST_DIR / src.name)
    print(
        json.dumps(
            {
                "retry_success_count": retry_success_count,
                "retry_failed_count": retry_failed_count,
                "retry_candidate_count": len(retry_rows),
                "merged_candidate_count": len(merged_rows),
                "boundary_hit_5s_before_after": [original_boundary_hit_before, boundary_hit_after],
                "actual_runtime_readable": readable_duration(elapsed),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
