#!/usr/bin/env python3
"""OpenCV 처리에 실패한 v2.3 영상의 장면 전환 후보를 FFmpeg pipe로 보완한다."""

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

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
VIDEO_INPUT_DIR = PROJECT_ROOT / "data/raw/videos"
WINDOWS_PATH = PROJECT_ROOT / "data/windows/window_labels_5s_v2_3.csv"
AD_SEGMENTS_PATH = PROJECT_ROOT / "data/segments/ad_interval_segments_v2_3.csv"
ALL_SEGMENTS_PATH = PROJECT_ROOT / "data/segments/label_interval_context_segments_v2_3.csv"
MANIFEST_PATH = PROJECT_ROOT / "data/video_metadata/video_manifest_v2_2.csv"
RETRY_REPORT_PATH = PROJECT_ROOT / "reports/retry_opencv_scene_change_failed4_v2_3_report.json"
BASE_CANDIDATES_PREFERRED = PROJECT_ROOT / "data/scene/opencv_scene_candidates_v2_3_merged_retry.csv"
BASE_CANDIDATES_FALLBACK = PROJECT_ROOT / "data/scene/opencv_scene_candidates_v2_3.csv"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts/utils/check_cv_environment.py"

SCENE_DIR = PROJECT_ROOT / "data/scene"
PROBE_PATH = SCENE_DIR / "ffmpeg_failed4_probe_metadata_v2_3.csv"
SCORES_PATH = SCENE_DIR / "ffmpeg_scene_scores_failed4_v2_3.csv"
CANDIDATES_PATH = SCENE_DIR / "ffmpeg_scene_candidates_failed4_v2_3.csv"
CANDIDATES_MMSS_PATH = SCENE_DIR / "ffmpeg_scene_candidates_failed4_v2_3_mmss.csv"
VIDEO_SUMMARY_PATH = SCENE_DIR / "ffmpeg_scene_failed4_video_summary_v2_3.csv"
MERGED_PATH = SCENE_DIR / "scene_candidates_v2_3_merged_ffmpeg_fallback.csv"
MERGED_MMSS_PATH = SCENE_DIR / "scene_candidates_v2_3_merged_ffmpeg_fallback_mmss.csv"
BOUNDARY_AUDIT_PATH = SCENE_DIR / "scene_candidate_boundary_audit_v2_3_merged_ffmpeg_fallback.csv"

REPORT_PATH = PROJECT_ROOT / "reports/ffmpeg_fallback_scene_change_failed4_v2_3_report.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/ffmpeg_fallback_scene_change_failed4_v2_3_summary.md"
LOG_PATH = PROJECT_ROOT / "logs/ffmpeg_fallback_scene_change_failed4_v2_3_run_log.txt"
SCRIPT_PATH = PROJECT_ROOT / "scripts/scene/ffmpeg_fallback_scene_change_failed4_v2_3.py"
README_PATH = PROJECT_ROOT / "README.md"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"
LATEST_README = LATEST_DIR / "README_latest_files.md"

ESTIMATED_RUNTIME = "약 6~12분"
RUNTIME_ESTIMATION_REASON = "실패 영상 4개에 대해 ffmpeg 1fps decoding, scene-change score 계산, candidate 병합, boundary audit 재계산을 수행해야 함."
SCALE_W = 320
SCALE_H = 180
FRAME_SIZE = SCALE_W * SCALE_H * 3
DEDUP_TOLERANCE_SEC = 2.0
RUN_LOG: list[str] = []


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_LOG.append(f"[{stamp}] {message}")
    print(message, flush=True)


def step(number: int, message: str) -> None:
    log(f"[STEP {number}/12] {message}")


def clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        rounded = round(float(value), 6)
        return int(round(rounded)) if abs(rounded - round(rounded)) < 1e-9 else rounded
    if isinstance(value, (int, np.integer)):
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


def readable(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}분 {total % 60}초"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: fmt(row.get(col, "")) for col in columns})


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def run_cmd(args: list[str]) -> tuple[bool, str, str]:
    try:
        result = subprocess.run(args, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        return result.returncode == 0, result.stdout, result.stderr
    except FileNotFoundError as exc:
        return False, "", str(exc)


def verify_cv() -> tuple[bool, str, list[Any]]:
    warnings: list[Any] = []
    if CHECK_ENV_SCRIPT.exists():
        ok, out, err = run_cmd(["conda", "run", "-n", "cv", "python", str(CHECK_ENV_SCRIPT)])
        log("cv check stdout: " + out.strip().replace("\n", " | "))
        if err.strip():
            log("cv check stderr: " + err.strip().replace("\n", " | "))
        return ok, sys.executable, warnings if ok else warnings + ["cv_check_failed"]
    return True, sys.executable, warnings


def version_line(tool: str) -> tuple[bool, str]:
    ok, out, err = run_cmd([tool, "-version"])
    text = (out or err).strip().splitlines()
    return ok, text[0] if text else ""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def failed_videos(retry_report: dict[str, Any], manifest: pd.DataFrame, warnings: list[Any]) -> list[dict[str, Any]]:
    filenames = retry_report.get("ffmpeg_fallback_needed_videos") or retry_report.get("failed_video_filenames") or []
    if len(filenames) != 4:
        warnings.append({"failed_video_count_not_4": len(filenames)})
    by_filename = {clean(row.get("video_filename")): row.to_dict() for _, row in manifest.iterrows()}
    paths = {p.name: p for pattern in ("*.mp4", "*.MP4") for p in VIDEO_INPUT_DIR.rglob(pattern)}
    rows: list[dict[str, Any]] = []
    for filename in filenames:
        m = by_filename.get(filename, {})
        rows.append(
            {
                "video_id": clean(m.get("label_mapping_video_id")) or clean(m.get("video_id")),
                "video_title": clean(m.get("label_mapping_video_title")) or clean(m.get("matched_label_video_title")) or clean(m.get("video_title")),
                "video_filename": filename,
                "video_path": str(paths.get(filename) or Path(clean(m.get("video_path")))),
            }
        )
    return rows


def parse_rate(rate: str) -> float | None:
    if not rate:
        return None
    if "/" in rate:
        a, b = rate.split("/", 1)
        try:
            return float(a) / float(b) if float(b) else None
        except Exception:
            return None
    return to_float(rate)


def ffprobe(video: dict[str, Any]) -> dict[str, Any]:
    path = clean(video.get("video_path"))
    ok, out, err = run_cmd(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path])
    row = {
        "video_id": clean(video.get("video_id")),
        "video_filename": clean(video.get("video_filename")),
        "video_path": path,
        "ffprobe_success": bool_text(ok),
        "duration_sec": "",
        "width": "",
        "height": "",
        "avg_frame_rate": "",
        "r_frame_rate": "",
        "codec_name": "",
        "pix_fmt": "",
        "bit_rate": "",
        "video_stream_found": "false",
        "ffprobe_warning": "" if ok else err.strip(),
    }
    if not ok:
        return row
    data = json.loads(out)
    streams = data.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    if not vstreams:
        row["ffprobe_warning"] = "no_video_stream"
        return row
    stream = vstreams[0]
    fmt_data = data.get("format", {})
    row.update(
        {
            "duration_sec": to_float(stream.get("duration")) or to_float(fmt_data.get("duration")) or "",
            "width": stream.get("width", ""),
            "height": stream.get("height", ""),
            "avg_frame_rate": stream.get("avg_frame_rate", ""),
            "r_frame_rate": stream.get("r_frame_rate", ""),
            "codec_name": stream.get("codec_name", ""),
            "pix_fmt": stream.get("pix_fmt", ""),
            "bit_rate": stream.get("bit_rate") or fmt_data.get("bit_rate", ""),
            "video_stream_found": "true",
        }
    )
    return row


def frame_features(rgb: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    edges = cv2.Canny(gray, 80, 160)
    return {"gray": gray.astype(np.float32), "hist": hist.astype(np.float32), "edges": edges.astype(np.float32) / 255.0, "gray_mean": float(gray.mean())}


def compare(prev: dict[str, Any], cur: dict[str, Any]) -> dict[str, float]:
    hsv_hist_diff = float(cv2.compareHist(prev["hist"], cur["hist"], cv2.HISTCMP_BHATTACHARYYA))
    pixel_absdiff_mean = float(np.mean(np.abs(cur["gray"] - prev["gray"])) / 255.0)
    edge_diff = float(np.mean(np.abs(cur["edges"] - prev["edges"])))
    gray_mean_diff = abs(cur["gray_mean"] - prev["gray_mean"]) / 255.0
    score = 0.4 * hsv_hist_diff + 0.3 * pixel_absdiff_mean + 0.2 * edge_diff + 0.1 * gray_mean_diff
    return {
        "hsv_hist_diff": max(0.0, min(1.0, hsv_hist_diff)),
        "pixel_absdiff_mean": max(0.0, min(1.0, pixel_absdiff_mean)),
        "edge_diff": max(0.0, min(1.0, edge_diff)),
        "gray_mean_diff": max(0.0, min(1.0, gray_mean_diff)),
        "scene_change_score": max(0.0, min(1.0, score)),
    }


def decode_scores(video: dict[str, Any], probe: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    duration = to_float(probe.get("duration_sec"))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", clean(video.get("video_path")),
        "-vf", f"fps=1,scale={SCALE_W}:{SCALE_H}", "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    scores: list[dict[str, Any]] = []
    prev_features: dict[str, Any] | None = None
    prev_time: float | None = None
    decoded = 0
    while True:
        chunk = proc.stdout.read(FRAME_SIZE) if proc.stdout else b""
        if not chunk:
            break
        if len(chunk) != FRAME_SIZE:
            break
        timestamp = float(decoded)
        if duration is not None and timestamp > duration + 1.0:
            break
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape((SCALE_H, SCALE_W, 3))
        cur_features = frame_features(frame)
        if prev_features is not None and prev_time is not None:
            scores.append(
                {
                    "video_id": clean(video.get("video_id")),
                    "video_filename": clean(video.get("video_filename")),
                    "prev_sample_time_sec": prev_time,
                    "next_sample_time_sec": timestamp,
                    "score_time_sec": timestamp,
                    **compare(prev_features, cur_features),
                }
            )
        prev_features = cur_features
        prev_time = timestamp
        decoded += 1
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    returncode = proc.wait()
    meta = {
        "decoded_frame_count": decoded,
        "score_count": len(scores),
        "ffmpeg_returncode": returncode,
        "ffmpeg_stderr": stderr.strip()[:500],
        "duration_sec": duration,
    }
    return scores, meta


def score_stats(scores: list[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {"threshold": "", "score_mean": "", "score_std": "", "score_p50": "", "score_p90": "", "score_p95": "", "score_p99": ""}
    values = np.array([s["scene_change_score"] for s in scores], dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    p95 = float(np.percentile(values, 95))
    return {
        "score_mean": float(values.mean()),
        "score_std": float(values.std()),
        "score_p50": median,
        "score_p90": float(np.percentile(values, 90)),
        "score_p95": p95,
        "score_p99": float(np.percentile(values, 99)),
        "threshold": max(median + 3.0 * mad, p95),
    }


def dedup(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[clean(row.get("video_id"))].append(row)
    kept_all: list[dict[str, Any]] = []
    removed = 0
    for video_id, items in by_video.items():
        ranked = sorted(items, key=lambda r: (-float(r.get("scene_change_score") or 0), float(r.get("candidate_time_sec") or 0)))
        kept: list[dict[str, Any]] = []
        for row in ranked:
            t = float(row.get("candidate_time_sec") or 0)
            if any(abs(t - float(k.get("candidate_time_sec") or 0)) <= DEDUP_TOLERANCE_SEC for k in kept):
                removed += 1
                continue
            kept.append(row)
        kept_all.extend(sorted(kept, key=lambda r: float(r.get("candidate_time_sec") or 0)))
    return sorted(kept_all, key=lambda r: (clean(r.get("video_id")), float(r.get("candidate_time_sec") or 0))), removed


def nearest_window(windows: pd.DataFrame, video_id: str, timestamp: float) -> str:
    subset = windows[windows["video_id"].map(clean).eq(video_id)]
    if subset.empty:
        return ""
    starts = subset["window_start_sec"].astype(float)
    ends = subset["window_end_sec"].astype(float)
    hit = subset[(starts <= timestamp) & (timestamp < ends)]
    return clean(hit.iloc[0].get("window_id")) if not hit.empty else ""


def nearest_boundary(ad_segments: pd.DataFrame, video_id: str, timestamp: float) -> dict[str, Any]:
    subset = ad_segments[ad_segments["video_id"].map(clean).eq(video_id)]
    best: tuple[float, str, float, pd.Series] | None = None
    for _, row in subset.iterrows():
        for btype, sec in [("ad_start", to_float(row.get("segment_start_sec"))), ("ad_end", to_float(row.get("segment_end_sec")))]:
            if sec is None:
                continue
            dist = abs(timestamp - sec)
            if best is None or dist < best[0]:
                best = (dist, btype, sec, row)
    if best is None:
        return {}
    return {
        "nearest_ad_interval_id": clean(best[3].get("ad_interval_id")),
        "nearest_ad_boundary_type": best[1],
        "nearest_ad_boundary_sec": best[2],
        "nearest_ad_boundary_mmss": mmss_floor(best[2]),
        "distance_to_nearest_ad_boundary_sec": best[0],
        "is_near_ad_start_3s": bool_text(best[1] == "ad_start" and best[0] <= 3),
        "is_near_ad_start_5s": bool_text(best[1] == "ad_start" and best[0] <= 5),
        "is_near_ad_end_3s": bool_text(best[1] == "ad_end" and best[0] <= 3),
        "is_near_ad_end_5s": bool_text(best[1] == "ad_end" and best[0] <= 5),
        "is_near_any_ad_boundary_5s": bool_text(best[0] <= 5),
    }


def make_candidates(video: dict[str, Any], scores: list[dict[str, Any]], stats: dict[str, Any], windows: pd.DataFrame, ad_segments: pd.DataFrame) -> list[dict[str, Any]]:
    if not scores or stats.get("threshold") == "":
        return []
    values = np.array([s["scene_change_score"] for s in scores], dtype=np.float32)
    selected = [s for s in scores if s["scene_change_score"] >= float(stats["threshold"])]
    candidates: list[dict[str, Any]] = []
    for score in selected:
        t = float(score["score_time_sec"])
        boundary = nearest_boundary(ad_segments, clean(video.get("video_id")), t)
        candidates.append(
            {
                "video_id": clean(video.get("video_id")),
                "video_title": clean(video.get("video_title")),
                "video_filename": clean(video.get("video_filename")),
                "video_path": clean(video.get("video_path")),
                "candidate_time_sec": t,
                "candidate_time_mmss": mmss_floor(t),
                "candidate_time_mmss_floor": mmss_floor(t),
                "candidate_time_mmss_round": mmss_round(t),
                "scene_change_score": score["scene_change_score"],
                "threshold": stats["threshold"],
                "candidate_source": "ffmpeg_fallback_failed4",
                "method_used": "ffmpeg_pipe_1fps",
                "score_rank_in_video": 0,
                "score_percentile_in_video": float((values <= score["scene_change_score"]).mean()),
                "prev_sample_time_sec": score["prev_sample_time_sec"],
                "next_sample_time_sec": score["next_sample_time_sec"],
                "sample_interval_sec": score["next_sample_time_sec"] - score["prev_sample_time_sec"],
                "fps_source": "ffmpeg_vf_fps_1",
                "duration_sec": stats.get("duration_sec", ""),
                "nearest_window_id": nearest_window(windows, clean(video.get("video_id")), t),
                **boundary,
                "review_note": "",
            }
        )
    candidates = sorted(candidates, key=lambda r: -float(r["scene_change_score"]))
    for i, row in enumerate(candidates, start=1):
        row["score_rank_in_video"] = i
    deduped, _ = dedup(candidates)
    return deduped


def standardize_base(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        t = to_float(row.get("candidate_time_sec")) or to_float(row.get("candidate_frame_sec"))
        if t is None:
            continue
        source = clean(row.get("candidate_source")) or "original_opencv_v2_3"
        if source == "original_v2_3":
            source = "original_opencv_v2_3"
        rows.append(
            {
                "candidate_source": source,
                "video_id": clean(row.get("video_id")),
                "video_title": clean(row.get("video_title")),
                "video_filename": clean(row.get("video_filename")),
                "video_path": clean(row.get("video_path")),
                "candidate_time_sec": t,
                "candidate_time_mmss_floor": mmss_floor(t),
                "candidate_time_mmss_round": mmss_round(t),
                "scene_change_score": to_float(row.get("scene_change_score")) or 0.0,
                "threshold": clean(row.get("threshold")) or clean(row.get("candidate_threshold")),
                "method_used": clean(row.get("method_used")) or "opencv_v2_3",
                "nearest_window_id": clean(row.get("nearest_window_id")) or clean(row.get("window_id")),
                "nearest_ad_interval_id": clean(row.get("nearest_ad_interval_id")),
                "nearest_ad_boundary_type": clean(row.get("nearest_ad_boundary_type")) or clean(row.get("nearest_ad_boundary")),
                "nearest_ad_boundary_sec": clean(row.get("nearest_ad_boundary_sec")),
                "nearest_ad_boundary_mmss": clean(row.get("nearest_ad_boundary_mmss")),
                "distance_to_nearest_ad_boundary_sec": clean(row.get("distance_to_nearest_ad_boundary_sec")),
            }
        )
    return rows


def count_between(cands: pd.DataFrame, video_id: str, start: float, end: float) -> int:
    subset = cands[cands["video_id"].map(clean).eq(video_id)]
    if subset.empty:
        return 0
    times = subset["candidate_time_sec"].astype(float)
    return int(((times >= start) & (times <= end)).sum())


def nearest_candidate(cands: pd.DataFrame, video_id: str, boundary: float) -> tuple[str, str, str]:
    subset = cands[cands["video_id"].map(clean).eq(video_id)].copy()
    if subset.empty:
        return "", "", ""
    subset["_dist"] = (subset["candidate_time_sec"].astype(float) - boundary).abs()
    row = subset.sort_values(["_dist", "scene_change_score"], ascending=[True, False]).iloc[0]
    sec = float(row["candidate_time_sec"])
    return str(fmt(sec)), mmss_floor(sec), str(fmt(float(row["_dist"])))


def boundary_audit(merged: list[dict[str, Any]], ad_segments: pd.DataFrame, all_segments: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cands = pd.DataFrame(merged)
    rows: list[dict[str, Any]] = []
    sh3 = sh5 = eh3 = eh5 = both5 = 0
    for _, ad in ad_segments.iterrows():
        vid = clean(ad.get("video_id"))
        aid = clean(ad.get("ad_interval_id"))
        start = float(ad["segment_start_sec"])
        end = float(ad["segment_end_sec"])
        subset = cands[cands["video_id"].map(clean).eq(vid)].copy() if not cands.empty else pd.DataFrame()
        times = subset["candidate_time_sec"].astype(float) if not subset.empty else pd.Series(dtype=float)
        start3 = int(((times - start).abs() <= 3).sum())
        start5 = int(((times - start).abs() <= 5).sum())
        end3 = int(((times - end).abs() <= 3).sum())
        end5 = int(((times - end).abs() <= 5).sum())
        sh3 += int(start3 > 0)
        sh5 += int(start5 > 0)
        eh3 += int(end3 > 0)
        eh5 += int(end5 > 0)
        both5 += int(start5 > 0 and end5 > 0)
        ns_sec, ns_mmss, ns_dist = nearest_candidate(cands, vid, start)
        ne_sec, ne_mmss, ne_dist = nearest_candidate(cands, vid, end)
        context = all_segments[(all_segments["video_id"].map(clean).eq(vid)) & (all_segments["ad_interval_id"].map(clean).eq(aid))]
        pre = context[context["segment_type"].map(clean).eq("pre_ad_start_10s")]
        post = context[context["segment_type"].map(clean).eq("post_ad_end_10s")]
        rows.append(
            {
                "ad_interval_id": aid,
                "video_id": vid,
                "video_title": clean(ad.get("video_title")),
                "video_filename": clean(ad.get("video_filename")),
                "ad_start_sec": start,
                "ad_end_sec": end,
                "ad_start_mmss": mmss_floor(start),
                "ad_end_mmss": mmss_floor(end),
                "start_hit_3s": bool_text(start3 > 0),
                "start_hit_5s": bool_text(start5 > 0),
                "end_hit_3s": bool_text(end3 > 0),
                "end_hit_5s": bool_text(end5 > 0),
                "both_boundary_hit_5s": bool_text(start5 > 0 and end5 > 0),
                "nearest_candidate_to_start_sec": ns_sec,
                "nearest_candidate_to_start_mmss": ns_mmss,
                "distance_to_start_candidate_sec": ns_dist,
                "nearest_candidate_to_end_sec": ne_sec,
                "nearest_candidate_to_end_mmss": ne_mmss,
                "distance_to_end_candidate_sec": ne_dist,
                "candidate_count_in_ad_interval": count_between(cands, vid, start, end),
                "candidate_count_in_pre10": count_between(cands, vid, float(pre.iloc[0]["segment_start_sec"]), float(pre.iloc[0]["segment_end_sec"])) if not pre.empty else 0,
                "candidate_count_in_post10": count_between(cands, vid, float(post.iloc[0]["segment_start_sec"]), float(post.iloc[0]["segment_end_sec"])) if not post.empty else 0,
                "candidate_count_near_start_5s": start5,
                "candidate_count_near_end_5s": end5,
            }
        )
    total = len(ad_segments) * 2
    summary = {
        "total_ad_intervals": len(ad_segments),
        "total_boundaries": total,
        "start_hit_3s_count": sh3,
        "start_hit_5s_count": sh5,
        "end_hit_3s_count": eh3,
        "end_hit_5s_count": eh5,
        "boundary_hit_5s_count": sh5 + eh5,
        "boundary_hit_5s_rate": (sh5 + eh5) / total if total else 0.0,
        "both_boundary_hit_5s_count": both5,
    }
    return rows, summary


def write_summary(report: dict[str, Any]) -> None:
    warning_text = "\n".join(f"- {w}" for w in report.get("warnings", [])) if report.get("warnings") else "- 주요 warning 없음"
    SUMMARY_PATH.write_text(
        f"""# ffmpeg_fallback_scene_change_failed4_v2_3 summary

## 작업 시간

- estimated runtime: {report.get('estimated_runtime')}
- start time: {report.get('start_time')}
- end time: {report.get('end_time')}
- actual runtime: {report.get('actual_runtime_readable')}

## FFmpeg Fallback

- ffmpeg available: {report.get('ffmpeg_available')}
- ffprobe available: {report.get('ffprobe_available')}
- failed video count: {report.get('failed_video_count')}
- ffprobe success count: {report.get('ffprobe_success_count')}
- ffmpeg decode success count: {report.get('ffmpeg_decode_success_count')}
- scene score success count: {report.get('scene_score_success_count')}
- decoded frame count by video: `{report.get('decoded_frame_count_by_video')}`
- score count by video: `{report.get('score_count_by_video')}`
- candidate count by video: `{report.get('ffmpeg_candidate_count_by_video')}`

## Candidate 병합

- original candidate count: {report.get('original_candidate_count')}
- ffmpeg fallback candidate count: {report.get('ffmpeg_fallback_candidate_count')}
- merged candidate count: {report.get('merged_candidate_count')}
- duplicate removed count: {report.get('duplicate_removed_count')}

## Boundary Audit

- before ±5s hit: {report.get('boundary_hit_5s_before')} / rate {report.get('boundary_hit_5s_rate_before')}
- after ±5s hit: {report.get('boundary_hit_5s_after')} / rate {report.get('boundary_hit_5s_rate_after')}
- hit rate change: {report.get('hit_rate_change')}

이 값은 scene-change 후보 audit용 수치이며 final performance claim이 아니다.

## Warning

{warning_text}
""",
        encoding="utf-8",
    )


def update_readme() -> None:
    section = """## FFmpeg Fallback Scene-change v2.3

OpenCV VideoCapture가 일부 mp4에서 metadata만 읽고 frame decode에 실패했기 때문에, 실패 4개 영상에 대해 ffmpeg 기반 1fps frame decode fallback을 수행했다. frame image 파일은 저장하지 않고 pipe 기반 raw frame stream으로 처리했다.

생성된 scene candidate는 기존 OpenCV candidate와 병합되었다. 이 결과는 scene-change evidence audit용이며 final 광고 탐지 성능 claim이 아니다. 이후 OCR/audio/rule detector는 `data/scene/scene_candidates_v2_3_merged_ffmpeg_fallback.csv`를 기준 scene evidence로 사용할 수 있다.
"""
    text = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "# youtube_ad_segment_detector\n"
    marker = "## FFmpeg Fallback Scene-change v2.3"
    if marker in text:
        before, _, after = text.partition(marker)
        next_idx = after.find("\n## ")
        text = before.rstrip() + "\n\n" + section + (after[next_idx:] if next_idx >= 0 else "")
    else:
        text = text.rstrip() + "\n\n" + section
    README_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def latest_copy(files: list[Path]) -> tuple[bool, list[str]]:
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
    LATEST_README.write_text(
        "# latest_for_chatgpt files\n\n이번 작업명: ffmpeg_fallback_scene_change_failed4_v2_3\n\n"
        "ffmpeg fallback 후보/score/metadata/report/log/script 핵심 파일만 복사했다. mp4, 원본 xlsx, frame image, model, checkpoint, cache 파일은 복사하지 않는다.\n",
        encoding="utf-8",
    )
    copied.append(str(LATEST_README))
    return True, copied


def main() -> None:
    for d in [SCENE_DIR, PROJECT_ROOT / "reports", PROJECT_ROOT / "logs", PROJECT_ROOT / "scripts/scene", LATEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    start_dt = datetime.now().astimezone()
    start_time = start_dt.isoformat(timespec="seconds")
    started = time.time()
    warnings: list[Any] = []
    errors: list[Any] = []
    log(f"작업 시작 전 예상 작업 시간: {ESTIMATED_RUNTIME}")
    log(f"예상 근거: {RUNTIME_ESTIMATION_REASON}")
    log(f"작업 시작 시각: {start_time}")

    step(1, "작업 시간 기록 및 cv 환경 확인")
    cv_ok, python_executable, cv_warnings = verify_cv()
    warnings.extend(cv_warnings)
    step(2, "ffmpeg/ffprobe 사용 가능 여부 확인")
    ffmpeg_ok, ffmpeg_version = version_line("ffmpeg")
    ffprobe_ok, ffprobe_version = version_line("ffprobe")
    if not (ffmpeg_ok and ffprobe_ok):
        errors.append("ffmpeg_or_ffprobe_unavailable")

    windows = read_csv(WINDOWS_PATH)
    ad_segments = read_csv(AD_SEGMENTS_PATH)
    all_segments = read_csv(ALL_SEGMENTS_PATH)
    manifest = read_csv(MANIFEST_PATH)
    retry_report = load_json(RETRY_REPORT_PATH)
    base_path = BASE_CANDIDATES_PREFERRED if BASE_CANDIDATES_PREFERRED.exists() else BASE_CANDIDATES_FALLBACK
    base_candidates = read_csv(base_path)

    step(3, "기존 report/log에서 실패 영상 4개 식별")
    videos = failed_videos(retry_report, manifest, warnings)
    step(4, "실패 영상별 ffprobe metadata 확인")
    probe_rows = [ffprobe(v) for v in videos]
    write_csv(PROBE_PATH, probe_rows, ["video_id", "video_filename", "video_path", "ffprobe_success", "duration_sec", "width", "height", "avg_frame_rate", "r_frame_rate", "codec_name", "pix_fmt", "bit_rate", "video_stream_found", "ffprobe_warning"])

    step(5, "ffmpeg 1fps pipe decoding 테스트")
    step(6, "ffmpeg frame stream 기반 scene score 계산")
    all_scores: list[dict[str, Any]] = []
    all_ffmpeg_candidates: list[dict[str, Any]] = []
    video_summaries: list[dict[str, Any]] = []
    for video, probe in zip(videos, probe_rows):
        log(f"ffmpeg decoding: {video['video_filename']}")
        scores, meta = decode_scores(video, probe)
        stats = score_stats(scores)
        stats["duration_sec"] = meta.get("duration_sec")
        candidates = make_candidates(video, scores, stats, windows, ad_segments)
        all_scores.extend(scores)
        all_ffmpeg_candidates.extend(candidates)
        near_start3 = sum(1 for c in candidates if c.get("is_near_ad_start_3s") == "true")
        near_start5 = sum(1 for c in candidates if c.get("is_near_ad_start_5s") == "true")
        near_end3 = sum(1 for c in candidates if c.get("is_near_ad_end_3s") == "true")
        near_end5 = sum(1 for c in candidates if c.get("is_near_ad_end_5s") == "true")
        warning = meta.get("ffmpeg_stderr", "")
        if scores and not candidates:
            warning = (warning + "; " if warning else "") + "score_count_recovered_but_candidate_count_zero"
        video_summaries.append(
            {
                "video_id": video["video_id"],
                "video_filename": video["video_filename"],
                "duration_sec": probe.get("duration_sec"),
                "codec_name": probe.get("codec_name"),
                "pix_fmt": probe.get("pix_fmt"),
                "decoded_frame_count": meta["decoded_frame_count"],
                "score_count": meta["score_count"],
                **stats,
                "candidate_count": len(candidates),
                "candidate_count_near_ad_start_3s": near_start3,
                "candidate_count_near_ad_start_5s": near_start5,
                "candidate_count_near_ad_end_3s": near_end3,
                "candidate_count_near_ad_end_5s": near_end5,
                "ffmpeg_decode_success": bool_text(meta["decoded_frame_count"] > 0),
                "scene_score_success": bool_text(meta["score_count"] > 0),
                "fallback_status": "success" if meta["score_count"] > 0 else "failed",
                "warning": warning,
            }
        )
        log(f"decoded={meta['decoded_frame_count']} score_count={meta['score_count']} candidates={len(candidates)}")

    step(7, "candidate threshold 적용 및 dedup")
    all_ffmpeg_candidates, ffmpeg_dup_removed = dedup(all_ffmpeg_candidates)
    step(8, "retry candidate 및 mm:ss CSV 생성")
    score_cols = ["video_id", "video_filename", "prev_sample_time_sec", "next_sample_time_sec", "score_time_sec", "hsv_hist_diff", "pixel_absdiff_mean", "edge_diff", "gray_mean_diff", "scene_change_score"]
    write_csv(SCORES_PATH, all_scores, score_cols)
    cand_cols = ["video_id", "video_title", "video_filename", "video_path", "candidate_time_sec", "candidate_time_mmss", "candidate_time_mmss_floor", "candidate_time_mmss_round", "scene_change_score", "threshold", "candidate_source", "method_used", "score_rank_in_video", "score_percentile_in_video", "prev_sample_time_sec", "next_sample_time_sec", "sample_interval_sec", "fps_source", "duration_sec", "nearest_window_id", "nearest_ad_interval_id", "nearest_ad_boundary_type", "nearest_ad_boundary_sec", "nearest_ad_boundary_mmss", "distance_to_nearest_ad_boundary_sec", "is_near_ad_start_3s", "is_near_ad_start_5s", "is_near_ad_end_3s", "is_near_ad_end_5s", "is_near_any_ad_boundary_5s", "review_note"]
    write_csv(CANDIDATES_PATH, all_ffmpeg_candidates, cand_cols)
    write_csv(CANDIDATES_MMSS_PATH, all_ffmpeg_candidates, ["video_id", "video_title", "video_filename", "candidate_time_sec", "candidate_time_mmss_floor", "candidate_time_mmss_round", "scene_change_score", "nearest_window_id", "nearest_ad_interval_id", "review_note"])
    summary_cols = ["video_id", "video_filename", "duration_sec", "codec_name", "pix_fmt", "decoded_frame_count", "score_count", "score_mean", "score_std", "score_p50", "score_p90", "score_p95", "score_p99", "threshold", "candidate_count", "candidate_count_near_ad_start_3s", "candidate_count_near_ad_start_5s", "candidate_count_near_ad_end_3s", "candidate_count_near_ad_end_5s", "ffmpeg_decode_success", "scene_score_success", "fallback_status", "warning"]
    write_csv(VIDEO_SUMMARY_PATH, video_summaries, summary_cols)

    step(9, "기존 v2_3 candidate와 병합")
    base_rows = standardize_base(base_candidates)
    merged_rows, total_dup_removed = dedup(base_rows + all_ffmpeg_candidates)
    for i, row in enumerate(merged_rows, start=1):
        row["merged_candidate_id"] = f"{clean(row.get('video_id'))}_FFM{str(i).zfill(6)}"
    merged_cols = ["merged_candidate_id", "candidate_source", "video_id", "video_title", "video_filename", "video_path", "candidate_time_sec", "candidate_time_mmss_floor", "candidate_time_mmss_round", "scene_change_score", "threshold", "method_used", "nearest_window_id", "nearest_ad_interval_id", "nearest_ad_boundary_type", "nearest_ad_boundary_sec", "nearest_ad_boundary_mmss", "distance_to_nearest_ad_boundary_sec"]
    write_csv(MERGED_PATH, merged_rows, merged_cols)
    write_csv(MERGED_MMSS_PATH, merged_rows, ["video_id", "video_title", "video_filename", "candidate_source", "candidate_time_sec", "candidate_time_mmss_floor", "candidate_time_mmss_round", "scene_change_score", "method_used", "nearest_window_id", "nearest_ad_interval_id"])

    step(10, "광고 boundary audit 재계산")
    audit_rows, audit_summary = boundary_audit(merged_rows, ad_segments, all_segments)
    audit_cols = ["ad_interval_id", "video_id", "video_title", "video_filename", "ad_start_sec", "ad_end_sec", "ad_start_mmss", "ad_end_mmss", "start_hit_3s", "start_hit_5s", "end_hit_3s", "end_hit_5s", "both_boundary_hit_5s", "nearest_candidate_to_start_sec", "nearest_candidate_to_start_mmss", "distance_to_start_candidate_sec", "nearest_candidate_to_end_sec", "nearest_candidate_to_end_mmss", "distance_to_end_candidate_sec", "candidate_count_in_ad_interval", "candidate_count_in_pre10", "candidate_count_in_post10", "candidate_count_near_start_5s", "candidate_count_near_end_5s"]
    write_csv(BOUNDARY_AUDIT_PATH, audit_rows, audit_cols)

    step(11, "QA report/summary/log 생성")
    retry_before = load_json(RETRY_REPORT_PATH)
    before_hit = retry_before.get("boundary_hit_5s_after") or retry_before.get("boundary_hit_5s_before")
    before_rate = retry_before.get("boundary_hit_5s_rate_after") or retry_before.get("boundary_hit_5s_rate_before")
    end_time = datetime.now().astimezone().isoformat(timespec="seconds")
    elapsed = time.time() - started
    generated_files = [PROBE_PATH, SCORES_PATH, CANDIDATES_PATH, CANDIDATES_MMSS_PATH, VIDEO_SUMMARY_PATH, MERGED_PATH, MERGED_MMSS_PATH, BOUNDARY_AUDIT_PATH, REPORT_PATH, SUMMARY_PATH, LOG_PATH, SCRIPT_PATH]
    report = {
        "project_root": str(PROJECT_ROOT),
        "estimated_runtime": ESTIMATED_RUNTIME,
        "runtime_estimation_reason": RUNTIME_ESTIMATION_REASON,
        "start_time": start_time,
        "end_time": end_time,
        "actual_runtime_seconds": elapsed,
        "actual_runtime_readable": readable(elapsed),
        "cv_environment_checked": cv_ok,
        "python_executable": python_executable,
        "ffmpeg_available": ffmpeg_ok,
        "ffprobe_available": ffprobe_ok,
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_version": ffprobe_version,
        "failed_video_count": len(videos),
        "failed_video_ids": [v["video_id"] for v in videos],
        "ffprobe_success_count": sum(1 for r in probe_rows if r.get("ffprobe_success") == "true"),
        "ffmpeg_decode_success_count": sum(1 for r in video_summaries if r.get("ffmpeg_decode_success") == "true"),
        "scene_score_success_count": sum(1 for r in video_summaries if r.get("scene_score_success") == "true"),
        "score_count_by_video": {r["video_filename"]: int(r["score_count"]) for r in video_summaries},
        "decoded_frame_count_by_video": {r["video_filename"]: int(r["decoded_frame_count"]) for r in video_summaries},
        "ffmpeg_candidate_count_by_video": {r["video_filename"]: int(r["candidate_count"]) for r in video_summaries},
        "original_candidate_count": len(base_rows),
        "ffmpeg_fallback_candidate_count": len(all_ffmpeg_candidates),
        "merged_candidate_count": len(merged_rows),
        "duplicate_removed_count": total_dup_removed,
        "ffmpeg_internal_duplicate_removed_count": ffmpeg_dup_removed,
        "boundary_hit_5s_before": before_hit,
        "boundary_hit_5s_after": audit_summary["boundary_hit_5s_count"],
        "boundary_hit_5s_rate_before": before_rate,
        "boundary_hit_5s_rate_after": audit_summary["boundary_hit_5s_rate"],
        "hit_rate_change": audit_summary["boundary_hit_5s_rate"] - float(before_rate or 0),
        **audit_summary,
        "generated_files": [str(p) for p in generated_files],
        "sub_agent_results": {"sub_agent_1_decode": "PENDING_EXTERNAL_REVIEW", "sub_agent_2_score_candidate": "PENDING_EXTERNAL_REVIEW", "sub_agent_3_boundary_output": "PENDING_EXTERNAL_REVIEW"},
        "warnings": warnings,
        "errors": errors,
        "old_project_modified": False,
    }
    update_readme()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(report)
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")

    step(12, "README/latest_for_chatgpt 갱신")
    latest_ok, latest_files = latest_copy(generated_files)
    report["latest_for_chatgpt_updated"] = latest_ok
    report["latest_for_chatgpt_files"] = latest_files
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(report)
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    if latest_ok:
        for src in [REPORT_PATH, SUMMARY_PATH, LOG_PATH]:
            shutil.copy2(src, LATEST_DIR / src.name)
    log(f"작업 종료 시각: {end_time}")
    log(f"실제 작업 시간: {readable(elapsed)}")
    print(json.dumps({"ffmpeg_fallback_candidate_count": len(all_ffmpeg_candidates), "merged_candidate_count": len(merged_rows), "boundary_hit_5s_before_after": [before_hit, audit_summary["boundary_hit_5s_count"]], "actual_runtime_readable": readable(elapsed), "errors": errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
