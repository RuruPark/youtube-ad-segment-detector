#!/usr/bin/env python3
"""v2.3 window/segment 기준에 맞춰 OpenCV 장면 전환 후보를 추출한다."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("YASD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
OLD_PROJECT_ROOT = Path(os.environ.get("YASD_OLD_PROJECT_ROOT", PROJECT_ROOT / "_old_project_not_included")).resolve()

VIDEO_INPUT_DIR = PROJECT_ROOT / "data/raw/videos"
WINDOW_LABELS_PATH = PROJECT_ROOT / "data/windows/window_labels_5s_v2_3.csv"
SEGMENTS_PATH = PROJECT_ROOT / "data/segments/label_interval_context_segments_v2_3.csv"
MANIFEST_PATH = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts/utils/check_cv_environment.py"

SCENE_DIR = PROJECT_ROOT / "data/scene"
OUTPUT_VIDEO_SCAN = SCENE_DIR / "opencv_scene_video_scan_v2_3.csv"
OUTPUT_CANDIDATES = SCENE_DIR / "opencv_scene_candidates_v2_3.csv"
OUTPUT_CANDIDATES_MMSS = SCENE_DIR / "opencv_scene_candidates_v2_3_mmss.csv"
OUTPUT_AUDIT = SCENE_DIR / "opencv_scene_candidate_boundary_audit_v2_3.csv"
REPORT_PATH = PROJECT_ROOT / "reports/opencv_scene_candidates_v2_3_report.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/opencv_scene_candidates_v2_3_summary.md"
LOG_PATH = PROJECT_ROOT / "logs/opencv_scene_candidates_v2_3_run_log.txt"
SCRIPT_PATH = PROJECT_ROOT / "scripts/features/extract_opencv_scene_change_candidates_v2_3.py"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"
LATEST_README = LATEST_DIR / "README_latest_files.md"

SAMPLE_FPS = 1.0
RESIZE_WIDTH = 320
RESIZE_HEIGHT = 180
MIN_CANDIDATE_GAP_SEC = 3.0
LOCAL_PEAK_RADIUS = 2
MIN_SCORE_FLOOR = 0.35
STD_MULTIPLIER = 2.5
PERCENTILE_THRESHOLD = 95.0
BOUNDARY_TOLERANCE_SEC = 5.0
HASH_WARN_SECONDS = 30.0
RUN_LOG: list[str] = []


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_LOG.append(f"[{stamp}] {message}")
    print(message)


def step(number: int, message: str) -> None:
    log(f"[STEP {number}/8] {message}")


def ensure_inside_project(path: Path) -> Path:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise RuntimeError(f"Refusing to write outside project root: {resolved}")
    return path


def ensure_dirs() -> None:
    for path in [SCENE_DIR, PROJECT_ROOT / "reports", PROJECT_ROOT / "logs", PROJECT_ROOT / "scripts/features", LATEST_DIR]:
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


def mmss(seconds: Any) -> str:
    sec = to_float(seconds)
    if sec is None:
        return ""
    total = int(round(sec))
    return f"{total // 60}:{total % 60:02d}"


def korean_min_sec(seconds: Any) -> str:
    sec = to_float(seconds)
    if sec is None:
        return ""
    total = int(round(sec))
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


def sha256_file(path: Path) -> tuple[str, float]:
    start = time.time()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), time.time() - start


def scan_mp4_files(manifest: pd.DataFrame, warnings: list[Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_by_path: dict[str, dict[str, Any]] = {}
    for _, row in manifest.iterrows():
        path_text = clean_value(row.get("video_path"))
        filename = clean_value(row.get("video_filename"))
        mapped_id = clean_value(row.get("label_mapping_video_id")) or clean_value(row.get("video_id"))
        mapped_title = clean_value(row.get("label_mapping_video_title")) or clean_value(row.get("matched_label_video_title")) or clean_value(row.get("video_title"))
        payload = {
            "video_id": mapped_id,
            "video_title": mapped_title,
            "video_filename": filename,
            "video_path": path_text,
            "file_stem": clean_value(row.get("file_stem")),
            "title_match_status": clean_value(row.get("title_match_status")),
            "label_mapping_status": clean_value(row.get("label_mapping_status")),
            "duration_sec_manifest": to_float(row.get("duration_sec")),
            "fps_manifest": to_float(row.get("fps")),
            "frame_count_manifest": to_float(row.get("frame_count")),
            "width_manifest": to_float(row.get("width")),
            "height_manifest": to_float(row.get("height")),
        }
        if path_text:
            manifest_by_path[str(Path(path_text).resolve())] = payload
        if filename:
            manifest_by_path[filename] = payload

    paths = sorted({path.resolve() for pattern in ("*.mp4", "*.MP4") for path in VIDEO_INPUT_DIR.rglob(pattern)})
    rows: list[dict[str, Any]] = []
    for path in paths:
        manifest_payload = manifest_by_path.get(str(path)) or manifest_by_path.get(path.name) or {}
        file_sha256 = ""
        hash_elapsed = 0.0
        try:
            file_sha256, hash_elapsed = sha256_file(path)
            if hash_elapsed > HASH_WARN_SECONDS:
                warnings.append({"sha256_slow": str(path), "elapsed_sec": round(hash_elapsed, 3)})
        except Exception as exc:  # noqa: BLE001
            warnings.append({"sha256_failed": str(path), "error": str(exc)})
        cap = cv2.VideoCapture(str(path))
        metadata_valid = cap.isOpened()
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) if metadata_valid else 0.0
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if metadata_valid else 0.0
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if metadata_valid else 0.0
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if metadata_valid else 0.0
        duration = frame_count / fps if fps > 0 and frame_count > 0 else manifest_payload.get("duration_sec_manifest")
        if cap:
            cap.release()
        metadata_warning = ""
        if not metadata_valid or not duration:
            metadata_warning = "opencv_metadata_invalid_or_duration_unknown"
            warnings.append({"metadata_invalid_or_duration_unknown": str(path)})
        rows.append(
            {
                "video_id": manifest_payload.get("video_id", ""),
                "video_title": manifest_payload.get("video_title", ""),
                "video_filename": path.name,
                "video_path": str(path),
                "file_sha256": file_sha256,
                "file_size_bytes": path.stat().st_size,
                "duration_sec": duration,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "metadata_valid": bool_text(bool(metadata_valid and duration)),
                "metadata_warning": metadata_warning,
                "sha256_elapsed_sec": hash_elapsed,
                "manifest_match_status": "matched" if manifest_payload else "unmatched_manifest",
                "title_match_status": manifest_payload.get("title_match_status", ""),
                "label_mapping_status": manifest_payload.get("label_mapping_status", ""),
            }
        )
    return rows, manifest_by_path


def preprocess_frame(frame: np.ndarray) -> dict[str, Any]:
    resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
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
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if denominator == 0:
        return 1.0
    return max(-1.0, min(1.0, numerator / denominator))


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


def read_frame_at(cap: cv2.VideoCapture, fps: float, timestamp_sec: float) -> np.ndarray | None:
    frame_index = max(0, int(round(timestamp_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None


def extract_video_scores(video_row: dict[str, Any], warnings: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(clean_value(video_row.get("video_path")))
    duration = to_float(video_row.get("duration_sec"))
    fps = to_float(video_row.get("fps"))
    if duration is None or duration <= 0 or fps is None or fps <= 0:
        warnings.append({"scene_extraction_skipped": str(path), "reason": "duration_or_fps_invalid"})
        return [], {"sample_count": 0, "candidate_threshold": "", "score_mean": "", "score_std": "", "score_p95": ""}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        warnings.append({"scene_extraction_skipped": str(path), "reason": "opencv_capture_open_failed"})
        return [], {"sample_count": 0, "candidate_threshold": "", "score_mean": "", "score_std": "", "score_p95": ""}

    timestamps = list(np.arange(0.0, duration, 1.0 / SAMPLE_FPS))
    previous_features: dict[str, Any] | None = None
    previous_timestamp: float | None = None
    scores: list[dict[str, Any]] = []
    for sample_index, timestamp in enumerate(timestamps):
        frame = read_frame_at(cap, fps, float(timestamp))
        if frame is None:
            continue
        features = preprocess_frame(frame)
        if previous_features is not None and previous_timestamp is not None:
            score_payload = compare_features(previous_features, features)
            scores.append(
                {
                    **score_payload,
                    "sample_index": sample_index,
                    "candidate_frame_sec": float(timestamp),
                    "previous_sample_sec": previous_timestamp,
                }
            )
        previous_features = features
        previous_timestamp = float(timestamp)
    cap.release()

    score_values = np.array([row["scene_change_score"] for row in scores], dtype=np.float32)
    if score_values.size == 0:
        return [], {"sample_count": len(timestamps), "candidate_threshold": "", "score_mean": "", "score_std": "", "score_p95": ""}
    score_mean = float(score_values.mean())
    score_std = float(score_values.std())
    score_p95 = float(np.percentile(score_values, PERCENTILE_THRESHOLD))
    threshold = max(MIN_SCORE_FLOOR, score_mean + STD_MULTIPLIER * score_std, score_p95)
    metadata = {
        "sample_count": len(timestamps),
        "score_count": len(scores),
        "candidate_threshold": threshold,
        "score_mean": score_mean,
        "score_std": score_std,
        "score_p95": score_p95,
        "score_max": float(score_values.max()),
    }
    candidates = select_candidates(scores, threshold)
    return candidates, metadata


def select_candidates(scores: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    candidate_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(scores):
        score = float(row["scene_change_score"])
        if score < threshold:
            continue
        left = max(0, idx - LOCAL_PEAK_RADIUS)
        right = min(len(scores), idx + LOCAL_PEAK_RADIUS + 1)
        local_max = max(float(scores[j]["scene_change_score"]) for j in range(left, right))
        if score < local_max:
            continue
        candidate_rows.append(row)
    candidate_rows.sort(key=lambda item: (-float(item["scene_change_score"]), float(item["candidate_frame_sec"])))
    kept: list[dict[str, Any]] = []
    for row in candidate_rows:
        timestamp = float(row["candidate_frame_sec"])
        if any(abs(timestamp - float(existing["candidate_frame_sec"])) < MIN_CANDIDATE_GAP_SEC for existing in kept):
            continue
        kept.append(row)
    kept.sort(key=lambda item: float(item["candidate_frame_sec"]))
    return kept


def find_window_for_candidate(windows: pd.DataFrame, video_id: str, timestamp: float) -> dict[str, str]:
    video_windows = windows[windows["video_id"].map(clean_value).eq(video_id)]
    if video_windows.empty:
        return {}
    starts = video_windows["window_start_sec"].astype(float)
    ends = video_windows["window_end_sec"].astype(float)
    matched = video_windows[(starts <= timestamp) & (timestamp < ends)]
    if matched.empty:
        matched = video_windows[(starts <= timestamp) & (np.isclose(ends, timestamp))]
    return matched.iloc[0].to_dict() if not matched.empty else {}


def nearest_segment(candidate_sec: float, video_id: str, segments: pd.DataFrame) -> dict[str, Any]:
    video_segments = segments[segments["video_id"].map(clean_value).eq(video_id)]
    if video_segments.empty:
        return {}
    best: tuple[float, pd.Series] | None = None
    for _, row in video_segments.iterrows():
        start = to_float(row.get("segment_start_sec"))
        end = to_float(row.get("segment_end_sec"))
        if start is None or end is None:
            continue
        if start <= candidate_sec <= end:
            distance = 0.0
        else:
            distance = min(abs(candidate_sec - start), abs(candidate_sec - end))
        if best is None or distance < best[0]:
            best = (distance, row)
    if best is None:
        return {}
    row = best[1]
    return {
        "nearest_ad_interval_id": clean_value(row.get("ad_interval_id")),
        "nearest_ad_segment_type": clean_value(row.get("segment_type")),
        "nearest_segment_distance_sec": best[0],
    }


def nearest_ad_boundary(candidate_sec: float, video_id: str, ad_segments: pd.DataFrame) -> dict[str, Any]:
    video_ads = ad_segments[ad_segments["video_id"].map(clean_value).eq(video_id)]
    best: tuple[float, str, float, pd.Series] | None = None
    for _, row in video_ads.iterrows():
        start = to_float(row.get("segment_start_sec"))
        end = to_float(row.get("segment_end_sec"))
        if start is None or end is None:
            continue
        for boundary_name, boundary_sec in [("ad_start", start), ("ad_end", end)]:
            distance = abs(candidate_sec - boundary_sec)
            if best is None or distance < best[0]:
                best = (distance, boundary_name, boundary_sec, row)
    if best is None:
        return {}
    return {
        "nearest_ad_boundary": best[1],
        "nearest_ad_boundary_sec": best[2],
        "distance_to_nearest_ad_boundary_sec": best[0],
        "nearest_boundary_ad_interval_id": clean_value(best[3].get("ad_interval_id")),
    }


def enrich_candidates(
    raw_candidates_by_video: dict[str, tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]],
    windows: pd.DataFrame,
    segments: pd.DataFrame,
) -> list[dict[str, Any]]:
    ad_segments = segments[segments["segment_type"].map(clean_value).eq("ad_interval")].copy()
    rows: list[dict[str, Any]] = []
    for video_key, payload in raw_candidates_by_video.items():
        video_row = payload[0]
        candidates = payload[1]
        threshold_meta = payload[2]
        video_id = clean_value(video_row.get("video_id"))
        for rank, candidate in enumerate(candidates, start=1):
            timestamp = float(candidate["candidate_frame_sec"])
            window = find_window_for_candidate(windows, video_id, timestamp)
            nearest = nearest_segment(timestamp, video_id, segments)
            boundary = nearest_ad_boundary(timestamp, video_id, ad_segments)
            in_ad_interval = nearest.get("nearest_ad_segment_type") == "ad_interval" and float(nearest.get("nearest_segment_distance_sec", 999999)) == 0
            rows.append(
                {
                    "candidate_id": f"{video_id}_SC{rank:06d}",
                    "video_id": video_id,
                    "video_title": clean_value(video_row.get("video_title")),
                    "video_filename": clean_value(video_row.get("video_filename")),
                    "video_path": clean_value(video_row.get("video_path")),
                    "file_sha256": clean_value(video_row.get("file_sha256")),
                    "candidate_frame_sec": timestamp,
                    "candidate_frame_mmss": mmss(timestamp),
                    "candidate_frame_min_sec_text": korean_min_sec(timestamp),
                    "scene_change_score": candidate["scene_change_score"],
                    "hsv_hist_score": candidate["hsv_hist_score"],
                    "brightness_score": candidate["brightness_score"],
                    "ssim_change_score": candidate["ssim_change_score"],
                    "edge_change_score": candidate["edge_change_score"],
                    "candidate_threshold": threshold_meta.get("candidate_threshold", ""),
                    "sampling_fps": SAMPLE_FPS,
                    "sample_index": candidate.get("sample_index", ""),
                    "previous_sample_sec": candidate.get("previous_sample_sec", ""),
                    "candidate_rank_in_video": rank,
                    "nearest_ad_interval_id": nearest.get("nearest_ad_interval_id", ""),
                    "nearest_ad_segment_type": nearest.get("nearest_ad_segment_type", ""),
                    "nearest_segment_distance_sec": nearest.get("nearest_segment_distance_sec", ""),
                    "nearest_ad_boundary": boundary.get("nearest_ad_boundary", ""),
                    "nearest_ad_boundary_sec": boundary.get("nearest_ad_boundary_sec", ""),
                    "nearest_ad_boundary_mmss": mmss(boundary.get("nearest_ad_boundary_sec", "")),
                    "distance_to_nearest_ad_boundary_sec": boundary.get("distance_to_nearest_ad_boundary_sec", ""),
                    "nearest_boundary_ad_interval_id": boundary.get("nearest_boundary_ad_interval_id", ""),
                    "window_id": clean_value(window.get("window_id")),
                    "window_label_scope": clean_value(window.get("window_label_scope")),
                    "window_start_sec": clean_value(window.get("window_start_sec")),
                    "window_end_sec": clean_value(window.get("window_end_sec")),
                    "window_matched_ad_interval_ids": clean_value(window.get("matched_ad_interval_ids")),
                    "in_ad_interval": bool_text(bool(in_ad_interval)),
                    "comment": "",
                }
            )
    return rows


def boundary_audit(candidates: pd.DataFrame, segments: pd.DataFrame) -> list[dict[str, Any]]:
    ad_segments = segments[segments["segment_type"].map(clean_value).eq("ad_interval")].copy()
    rows: list[dict[str, Any]] = []
    for _, segment in ad_segments.iterrows():
        video_id = clean_value(segment.get("video_id"))
        video_candidates = candidates[candidates["video_id"].map(clean_value).eq(video_id)] if not candidates.empty else pd.DataFrame()
        for boundary_type, boundary_sec in [
            ("ad_start", to_float(segment.get("segment_start_sec"))),
            ("ad_end", to_float(segment.get("segment_end_sec"))),
        ]:
            if boundary_sec is None:
                continue
            if video_candidates.empty:
                hits = pd.DataFrame()
            else:
                distances = (video_candidates["candidate_frame_sec"].astype(float) - boundary_sec).abs()
                hits = video_candidates[distances <= BOUNDARY_TOLERANCE_SEC].copy()
                if not hits.empty:
                    hits["_distance"] = (hits["candidate_frame_sec"].astype(float) - boundary_sec).abs()
            if hits.empty:
                candidate_count = 0
                max_score = ""
                nearest_sec = ""
                nearest_mmss = ""
                nearest_distance = ""
                nearest_segment_type = ""
                nearest_candidate_id = ""
            else:
                candidate_count = int(len(hits))
                max_score = float(hits["scene_change_score"].astype(float).max())
                nearest = hits.sort_values(["_distance", "scene_change_score"], ascending=[True, False]).iloc[0]
                nearest_sec = float(nearest["candidate_frame_sec"])
                nearest_mmss = clean_value(nearest["candidate_frame_mmss"])
                nearest_distance = float(abs(nearest_sec - boundary_sec))
                nearest_segment_type = clean_value(nearest.get("nearest_ad_segment_type"))
                nearest_candidate_id = clean_value(nearest.get("candidate_id"))
            rows.append(
                {
                    "audit_id": f"{clean_value(segment.get('ad_interval_id'))}_{boundary_type}",
                    "video_id": video_id,
                    "video_title": clean_value(segment.get("video_title")),
                    "video_filename": clean_value(segment.get("video_filename")),
                    "ad_interval_id": clean_value(segment.get("ad_interval_id")),
                    "boundary_type": boundary_type,
                    "boundary_sec": boundary_sec,
                    "boundary_mmss": mmss(boundary_sec),
                    "boundary_min_sec_text": korean_min_sec(boundary_sec),
                    "search_window_start_sec": max(0.0, boundary_sec - BOUNDARY_TOLERANCE_SEC),
                    "search_window_end_sec": boundary_sec + BOUNDARY_TOLERANCE_SEC,
                    "candidate_count_within_5s": candidate_count,
                    "hit_within_5s": bool_text(candidate_count > 0),
                    "max_scene_change_score_within_5s": max_score,
                    "nearest_candidate_id": nearest_candidate_id,
                    "nearest_candidate_sec": nearest_sec,
                    "nearest_candidate_mmss": nearest_mmss,
                    "nearest_candidate_distance_sec": nearest_distance,
                    "nearest_ad_segment_type": nearest_segment_type,
                    "audit_note": "",
                }
            )
    return rows


def make_mmss_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidates:
        rows.append(
            {
                "video_id": row.get("video_id", ""),
                "video_title": row.get("video_title", ""),
                "candidate_frame_sec": row.get("candidate_frame_sec", ""),
                "candidate_frame_mmss": row.get("candidate_frame_mmss", ""),
                "candidate_frame_min_sec_text": row.get("candidate_frame_min_sec_text", ""),
                "scene_change_score": row.get("scene_change_score", ""),
                "nearest_ad_interval_id": row.get("nearest_ad_interval_id", ""),
                "nearest_ad_segment_type": row.get("nearest_ad_segment_type", ""),
                "window_id": row.get("window_id", ""),
                "comment": "",
            }
        )
    return rows


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
        "opencv_scene_candidates_v2_3.csv": "OpenCV scene-change 후보 전체 CSV",
        "opencv_scene_candidates_v2_3_mmss.csv": "후보 timestamp를 mm:ss와 n분 n초로 변환한 확인용 CSV",
        "opencv_scene_candidate_boundary_audit_v2_3.csv": "광고 start/end ±5초 후보 hit audit",
        "opencv_scene_video_scan_v2_3.csv": "mp4 탐색, SHA256, duration 기록",
        "opencv_scene_candidates_v2_3_report.json": "scene-change 후보 생성 상세 report",
        "opencv_scene_candidates_v2_3_summary.md": "scene-change 후보 생성 요약",
        "opencv_scene_candidates_v2_3_run_log.txt": "scene-change 후보 생성 실행 log",
        "extract_opencv_scene_change_candidates_v2_3.py": "OpenCV 후보 추출 재현 스크립트",
    }
    LATEST_README.write_text(
        "# latest_for_chatgpt files\n\n"
        "latest_for_chatgpt는 최신 작업 핵심 파일만 모아둔 복사본 경로이다. 원본 파일은 프로젝트 내부 원래 경로에 존재한다.\n\n"
        "이번 작업명: extract_opencv_scene_change_candidates_v2_3\n\n"
        "이번 작업에서 추가된 핵심 변경:\n\n"
        "- v2_3 window/segment 기준 OpenCV scene-change 후보 추출\n"
        "- 후보 timestamp 초 단위를 mm:ss / n분 n초 형식으로 변환\n"
        "- 광고 interval start/end ±5초 candidate audit 생성\n\n"
        "복사된 파일 목록과 목적:\n\n"
        + "\n".join(f"- `{Path(path).name}`: {descriptions.get(Path(path).name, 'scene-change output')}" for path in copied)
        + "\n\nmp4 영상 파일, 원본 xlsx, frame, model, checkpoint, cache 파일은 복사하지 않는다.\n",
        encoding="utf-8",
    )
    copied.append(str(LATEST_README))
    return True, copied


def make_summary(report: dict[str, Any]) -> str:
    warnings = report.get("warnings", [])
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- 주요 warning 없음"
    return f"""# opencv_scene_candidates_v2_3 summary

## Input

- project root: `{report.get('project_root')}`
- video input dir: `{report.get('video_input_dir')}`
- processed mp4 count: {report.get('processed_mp4_count')} / {report.get('mp4_file_count')}
- sampling fps: {report.get('sampling_fps')}

## Candidates

- generated candidate count: {report.get('scene_candidate_count')}
- duplicate candidate count: {report.get('duplicate_candidate_count')}
- videos with candidates: {report.get('videos_with_candidates')}
- videos without candidates: {report.get('videos_without_candidates')}

## Boundary Audit

- ad interval boundary count: {report.get('ad_interval_boundary_count')}
- boundary hit count within ±5s: {report.get('boundary_hit_count_within_5s')}
- boundary hit rate within ±5s: {report.get('boundary_hit_rate_within_5s')}

## Score Distribution

- score distribution: `{report.get('scene_change_score_distribution')}`

## Generated Files

- candidates: `{report.get('candidate_csv_path')}`
- mmss: `{report.get('candidate_mmss_csv_path')}`
- audit: `{report.get('candidate_audit_csv_path')}`
- report: `{REPORT_PATH}`

## Warning

{warning_text}

## 다음 작업

OCR feature 생성 또는 후보 visual/audio cue 결합 단계로 넘어갈 수 있다.
"""


def main() -> None:
    ensure_dirs()
    warnings: list[Any] = []
    errors: list[Any] = []
    started = time.time()

    step(1, "cv 환경 확인")
    cv_ok, python_executable, cv_warnings = verify_cv_environment()
    warnings.extend(cv_warnings)
    if not cv_ok:
        errors.append("cv_environment_check_failed")
        report = {
            "project_root": str(PROJECT_ROOT),
            "cv_environment_checked": False,
            "python_executable": python_executable,
            "errors": errors,
            "warnings": warnings,
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
        raise SystemExit(1)

    step(2, "입력 CSV 로드")
    windows = read_csv(WINDOW_LABELS_PATH)
    segments = read_csv(SEGMENTS_PATH)
    manifest = read_csv(MANIFEST_PATH)
    log(f"Loaded windows={len(windows)}, segments={len(segments)}, manifest={len(manifest)}")

    step(3, "mp4 파일 탐색 및 SHA256/duration 기록")
    video_scan_rows, _manifest_lookup = scan_mp4_files(manifest, warnings)
    write_csv(
        OUTPUT_VIDEO_SCAN,
        video_scan_rows,
        [
            "video_id",
            "video_title",
            "video_filename",
            "video_path",
            "file_sha256",
            "file_size_bytes",
            "duration_sec",
            "fps",
            "frame_count",
            "width",
            "height",
            "metadata_valid",
            "metadata_warning",
            "sha256_elapsed_sec",
            "manifest_match_status",
            "title_match_status",
            "label_mapping_status",
        ],
    )
    log(f"Found mp4 files={len(video_scan_rows)}")

    step(4, "OpenCV scene-change 후보 추출")
    raw_candidates_by_video: dict[str, tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = {}
    per_video_stats: list[dict[str, Any]] = []
    for index, video_row in enumerate(video_scan_rows, start=1):
        filename = clean_value(video_row.get("video_filename"))
        log(f"Processing video {index}/{len(video_scan_rows)}: {filename}")
        video_start = time.time()
        candidates, score_metadata = extract_video_scores(video_row, warnings)
        elapsed = time.time() - video_start
        raw_candidates_by_video[filename] = (video_row, candidates, score_metadata)
        per_video_stats.append(
            {
                "video_id": clean_value(video_row.get("video_id")),
                "video_filename": filename,
                "sample_count": score_metadata.get("sample_count", 0),
                "score_count": score_metadata.get("score_count", 0),
                "candidate_count": len(candidates),
                "candidate_threshold": score_metadata.get("candidate_threshold", ""),
                "score_mean": score_metadata.get("score_mean", ""),
                "score_std": score_metadata.get("score_std", ""),
                "score_p95": score_metadata.get("score_p95", ""),
                "score_max": score_metadata.get("score_max", ""),
                "elapsed_sec": elapsed,
            }
        )
        log(f"Finished {filename}: candidates={len(candidates)}, elapsed_sec={elapsed:.2f}")

    step(5, "timestamp 분/초 변환 및 window/segment 연결")
    candidate_rows = enrich_candidates(raw_candidates_by_video, windows, segments)
    candidate_columns = [
        "candidate_id",
        "video_id",
        "video_title",
        "video_filename",
        "video_path",
        "file_sha256",
        "candidate_frame_sec",
        "candidate_frame_mmss",
        "candidate_frame_min_sec_text",
        "scene_change_score",
        "hsv_hist_score",
        "brightness_score",
        "ssim_change_score",
        "edge_change_score",
        "candidate_threshold",
        "sampling_fps",
        "sample_index",
        "previous_sample_sec",
        "candidate_rank_in_video",
        "nearest_ad_interval_id",
        "nearest_ad_segment_type",
        "nearest_segment_distance_sec",
        "nearest_ad_boundary",
        "nearest_ad_boundary_sec",
        "nearest_ad_boundary_mmss",
        "distance_to_nearest_ad_boundary_sec",
        "nearest_boundary_ad_interval_id",
        "window_id",
        "window_label_scope",
        "window_start_sec",
        "window_end_sec",
        "window_matched_ad_interval_ids",
        "in_ad_interval",
        "comment",
    ]
    write_csv(OUTPUT_CANDIDATES, candidate_rows, candidate_columns)
    mmss_rows = make_mmss_rows(candidate_rows)
    write_csv(
        OUTPUT_CANDIDATES_MMSS,
        mmss_rows,
        [
            "video_id",
            "video_title",
            "candidate_frame_sec",
            "candidate_frame_mmss",
            "candidate_frame_min_sec_text",
            "scene_change_score",
            "nearest_ad_interval_id",
            "nearest_ad_segment_type",
            "window_id",
            "comment",
        ],
    )

    step(6, "candidate boundary audit 생성")
    candidates_df = pd.DataFrame(candidate_rows)
    audit_rows = boundary_audit(candidates_df, segments)
    write_csv(
        OUTPUT_AUDIT,
        audit_rows,
        [
            "audit_id",
            "video_id",
            "video_title",
            "video_filename",
            "ad_interval_id",
            "boundary_type",
            "boundary_sec",
            "boundary_mmss",
            "boundary_min_sec_text",
            "search_window_start_sec",
            "search_window_end_sec",
            "candidate_count_within_5s",
            "hit_within_5s",
            "max_scene_change_score_within_5s",
            "nearest_candidate_id",
            "nearest_candidate_sec",
            "nearest_candidate_mmss",
            "nearest_candidate_distance_sec",
            "nearest_ad_segment_type",
            "audit_note",
        ],
    )

    step(7, "검증 및 report/summary/log 생성")
    candidate_df = pd.DataFrame(candidate_rows)
    duplicate_candidate_count = 0
    score_distribution: dict[str, Any] = {}
    if not candidate_df.empty:
        duplicate_candidate_count = int(candidate_df.duplicated(subset=["video_id", "candidate_frame_sec"]).sum())
        scores = candidate_df["scene_change_score"].astype(float)
        score_distribution = {
            "min": float(scores.min()),
            "p25": float(scores.quantile(0.25)),
            "median": float(scores.quantile(0.5)),
            "p75": float(scores.quantile(0.75)),
            "p95": float(scores.quantile(0.95)),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
        }
    videos_with_candidates = sorted({row["video_filename"] for row in candidate_rows})
    videos_without_candidates = sorted(
        clean_value(row.get("video_filename"))
        for row in video_scan_rows
        if clean_value(row.get("video_filename")) not in set(videos_with_candidates)
    )
    audit_df = pd.DataFrame(audit_rows)
    boundary_count = int(len(audit_df))
    boundary_hit_count = int(audit_df["hit_within_5s"].eq("true").sum()) if not audit_df.empty else 0
    boundary_hit_rate = boundary_hit_count / boundary_count if boundary_count else 0.0
    candidate_scope_distribution = (
        dict(sorted(Counter(candidate_df["window_label_scope"].map(clean_value)).items())) if not candidate_df.empty else {}
    )
    candidate_nearest_segment_distribution = (
        dict(sorted(Counter(candidate_df["nearest_ad_segment_type"].map(clean_value)).items())) if not candidate_df.empty else {}
    )
    if duplicate_candidate_count:
        warnings.append({"duplicate_candidate_count": duplicate_candidate_count})
    if videos_without_candidates:
        warnings.append({"videos_without_candidates": videos_without_candidates})

    generated_files = [
        OUTPUT_VIDEO_SCAN,
        OUTPUT_CANDIDATES,
        OUTPUT_CANDIDATES_MMSS,
        OUTPUT_AUDIT,
        REPORT_PATH,
        SUMMARY_PATH,
        LOG_PATH,
        SCRIPT_PATH,
    ]
    report = {
        "project_root": str(PROJECT_ROOT),
        "video_input_dir": str(VIDEO_INPUT_DIR),
        "window_labels_path": str(WINDOW_LABELS_PATH),
        "segments_path": str(SEGMENTS_PATH),
        "manifest_path": str(MANIFEST_PATH),
        "cv_environment_checked": cv_ok,
        "python_executable": python_executable,
        "old_project_modified": False,
        "mp4_file_count": int(len(video_scan_rows)),
        "processed_mp4_count": int(sum(1 for row in per_video_stats if int(row.get("score_count") or 0) > 0)),
        "metadata_valid_mp4_count": int(sum(1 for row in video_scan_rows if row.get("metadata_valid") == "true")),
        "sampling_fps": SAMPLE_FPS,
        "scene_score_method": {
            "hsv_hist_weight": 0.45,
            "brightness_weight": 0.20,
            "ssim_change_weight": 0.20,
            "edge_change_weight": 0.15,
            "threshold_rule": f"max({MIN_SCORE_FLOOR}, mean + {STD_MULTIPLIER}*std, p{PERCENTILE_THRESHOLD}) per video",
            "min_candidate_gap_sec": MIN_CANDIDATE_GAP_SEC,
            "local_peak_radius_samples": LOCAL_PEAK_RADIUS,
        },
        "scene_candidate_count": int(len(candidate_rows)),
        "candidate_csv_path": str(OUTPUT_CANDIDATES),
        "candidate_mmss_csv_path": str(OUTPUT_CANDIDATES_MMSS),
        "candidate_audit_csv_path": str(OUTPUT_AUDIT),
        "video_scan_csv_path": str(OUTPUT_VIDEO_SCAN),
        "timestamp_mmss_conversion_done": OUTPUT_CANDIDATES_MMSS.exists(),
        "candidate_audit_done": OUTPUT_AUDIT.exists(),
        "duplicate_candidate_count": duplicate_candidate_count,
        "scene_change_score_distribution": score_distribution,
        "candidate_window_label_scope_distribution": candidate_scope_distribution,
        "candidate_nearest_segment_type_distribution": candidate_nearest_segment_distribution,
        "ad_interval_count": int((segments["segment_type"].map(clean_value) == "ad_interval").sum()),
        "ad_interval_boundary_count": boundary_count,
        "boundary_hit_count_within_5s": boundary_hit_count,
        "boundary_hit_rate_within_5s": boundary_hit_rate,
        "videos_with_candidates": len(videos_with_candidates),
        "videos_without_candidates": len(videos_without_candidates),
        "videos_without_candidates_list": videos_without_candidates,
        "per_video_scene_stats": per_video_stats,
        "generated_files": [str(path) for path in generated_files],
        "warnings": warnings,
        "errors": errors,
        "run_elapsed_sec": time.time() - started,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(make_summary(report), encoding="utf-8")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")

    step(8, "latest_for_chatgpt 갱신")
    latest_ok, latest_files = clear_latest_and_copy(generated_files)
    report["latest_for_chatgpt_updated"] = latest_ok
    report["latest_for_chatgpt_files"] = latest_files
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(make_summary(report), encoding="utf-8")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    if latest_ok:
        for src in [REPORT_PATH, SUMMARY_PATH, LOG_PATH]:
            shutil.copy2(src, LATEST_DIR / src.name)
    print(json.dumps({
        "mp4_file_count": report["mp4_file_count"],
        "processed_mp4_count": report["processed_mp4_count"],
        "scene_candidate_count": report["scene_candidate_count"],
        "boundary_hit_rate_within_5s": report["boundary_hit_rate_within_5s"],
        "latest_for_chatgpt_updated": report["latest_for_chatgpt_updated"],
        "errors": errors,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
