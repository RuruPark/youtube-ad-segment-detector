#!/usr/bin/env python3
"""Build the state-machine ad review viewer v1.1 artifacts.

This script intentionally does not run the detector, extract features, tune
rules, tune thresholds, generate predictions, re-encode videos, extract frames,
create thumbnails, or copy media files. It only packages existing detector v1.1
outputs and actual labels into a lightweight human review viewer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TASK_NAME = "state_machine_ad_review_viewer"
VERSION = "v1_1"
VIEWER_NAME = f"{TASK_NAME}_{VERSION}"
PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
OUTPUT_DIR = PROJECT_ROOT / "outputs/review/state_machine_ad_review_viewer_v1_1"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt_state_machine_review_viewer_v1_1"
REPORT_DIR = PROJECT_ROOT / "reports/review"
LOG_DIR = PROJECT_ROOT / "logs"
BACKUP_ROOT = PROJECT_ROOT / "backups"
LOG_PATH = LOG_DIR / "state_machine_ad_review_viewer_v1_1_run_log.txt"
REPORT_PATH = REPORT_DIR / "state_machine_ad_review_viewer_v1_1_report.json"
SUMMARY_PATH = REPORT_DIR / "state_machine_ad_review_viewer_v1_1_summary.md"
MANIFEST_NAME = "review_manifest_v1_1_train_val.json"
MANIFEST_PATH = OUTPUT_DIR / MANIFEST_NAME

INPUT_FILES = {
    "closed_predictions": PROJECT_ROOT / "data/predictions/state_machine_interval_predictions_v1_1_train_val.csv",
    "open_interval_candidates": PROJECT_ROOT / "data/predictions/state_machine_open_interval_candidates_v1_1_train_val.csv",
    "anchor_trace": PROJECT_ROOT / "data/predictions/state_machine_anchor_trace_v1_1_train_val.csv",
    "validation_audit": PROJECT_ROOT / "data/predictions/state_machine_detector_validation_audit_v1_1_train_val.csv",
    "actual_labels": PROJECT_ROOT / "data/segments/ad_interval_segments_v2_4.csv",
    "split": PROJECT_ROOT / "data/splits/video_split_v2_4.csv",
    "config": PROJECT_ROOT / "configs/detectors/state_machine_interval_detector_v1_1_config.json",
    "detector_report": PROJECT_ROOT / "reports/detectors/state_machine_interval_detector_v1_1_report.json",
    "detector_summary": PROJECT_ROOT / "reports/detectors/state_machine_interval_detector_v1_1_summary.md",
}

EXPECTED_OUTPUT_FILES = [
    OUTPUT_DIR / "index.html",
    OUTPUT_DIR / "app.js",
    OUTPUT_DIR / "style.css",
    MANIFEST_PATH,
    OUTPUT_DIR / "README_review_viewer.md",
    SUMMARY_PATH,
    REPORT_PATH,
    LOG_PATH,
    PROJECT_ROOT / "scripts/review/build_state_machine_ad_review_viewer_v1_1.py",
    PROJECT_ROOT / "scripts/review/serve_state_machine_ad_review_viewer_v1_1.py",
]

LATEST_EXPECTED_NAMES = [
    "README_latest_files.md",
    "index.html",
    "app.js",
    "style.css",
    MANIFEST_NAME,
    "README_review_viewer.md",
    "state_machine_ad_review_viewer_v1_1_summary.md",
    "state_machine_ad_review_viewer_v1_1_report.json",
    "state_machine_ad_review_viewer_v1_1_run_log.txt",
    "build_state_machine_ad_review_viewer_v1_1.py",
    "serve_state_machine_ad_review_viewer_v1_1.py",
]

INCLUDED_SPLITS = {"train", "validation"}
EXCLUDED_TEST_VIDEO_IDS = {4, 16, 17}
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
}
FORBIDDEN_DIRECTORY_PARTS = {
    "cache",
    "frames",
    "frame_images",
    "raw_video",
    "video_proxy",
    "model_cache",
    "tmp",
    "__pycache__",
}

ACTUAL_START_COLUMNS = ["ad_start_sec", "start_sec", "segment_start_sec", "interval_start_sec"]
ACTUAL_END_COLUMNS = ["ad_end_sec", "end_sec", "segment_end_sec", "interval_end_sec"]
ACTUAL_TYPE_COLUMNS = ["segment_type", "interval_type", "label", "is_ad"]
NON_AD_VALUES = {
    "non_ad",
    "random_non_ad",
    "post_ad",
    "pre_ad",
    "not_ad",
    "background",
    "context",
    "negative",
    "false",
    "0",
    "no",
}
AD_VALUES = {
    "ad",
    "ad_interval",
    "ad_full",
    "advertisement",
    "sponsored",
    "sponsor",
    "true",
    "1",
    "yes",
}


class StepLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.step_no = 0
        self.lines: list[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def step(self, message: str) -> None:
        self.step_no += 1
        self.write(f"[STEP {self.step_no:02d}] {message}")

    def write(self, message: str = "") -> None:
        print(message)
        self.lines.append(message)
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_stats(path: Path) -> dict[str, Any]:
    stat = path.stat()
    line_count = None
    if path.suffix.lower() in {".csv", ".md", ".txt", ".json", ".py"}:
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                line_count = sum(1 for _ in handle)
        except UnicodeDecodeError:
            line_count = None
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
        "line_count": line_count,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ad"}


def pick_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    normalized = {name.strip("\ufeff"): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def normalize_split(value: Any) -> str:
    return str(value or "").strip().lower()


def clamp_interval(
    start: float | None,
    end: float | None,
    duration: float | None,
    source: str,
    video_id: int,
    warnings: list[str],
) -> tuple[float, float] | None:
    if start is None or end is None:
        warnings.append(f"Missing interval boundary in {source} for video_id={video_id}")
        return None
    original_start, original_end = start, end
    start = max(0.0, start)
    if duration is not None and duration > 0:
        end = min(end, duration)
    if start >= end:
        warnings.append(
            f"Invalid interval dropped in {source} for video_id={video_id}: "
            f"start={original_start}, end={original_end}, duration={duration}"
        )
        return None
    if start != original_start or end != original_end:
        warnings.append(
            f"Interval clamped in {source} for video_id={video_id}: "
            f"{original_start}-{original_end} -> {start}-{end}"
        )
    return (round(start, 3), round(end, 3))


def backup_existing_targets(timestamp: str, logger: StepLogger) -> dict[str, Any]:
    backup_dir = BACKUP_ROOT / f"state_machine_ad_review_viewer_v1_1_{timestamp}"
    targets = [
        OUTPUT_DIR,
        LATEST_DIR,
        REPORT_PATH,
        SUMMARY_PATH,
        LOG_PATH,
        PROJECT_ROOT / "scripts/review/build_state_machine_ad_review_viewer_v1_1.py",
        PROJECT_ROOT / "scripts/review/serve_state_machine_ad_review_viewer_v1_1.py",
    ]
    copied: list[dict[str, str]] = []
    for target in targets:
        if not target.exists():
            continue
        destination = backup_dir / rel(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if target.is_dir():
            shutil.copytree(target, destination)
        else:
            shutil.copy2(target, destination)
        copied.append({"source": str(target), "backup": str(destination)})
    if copied:
        logger.write(f"Backed up {len(copied)} existing target(s) to {backup_dir}")
    else:
        logger.write("No existing target outputs needed backup")
    return {"backup_dir": str(backup_dir), "copied": copied}


def snapshot_tree(root: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = ["relative_path\tsize_bytes\tmtime_ns\tkind"]
    if not root.exists():
        output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return {"root": str(root), "exists": False, "file_count": 0, "dir_count": 0, "snapshot": str(output_path)}
    file_count = 0
    dir_count = 0
    for current, dirs, files in os.walk(root):
        dirs.sort()
        files.sort()
        current_path = Path(current)
        for dirname in dirs:
            path = current_path / dirname
            stat = path.stat()
            relative = path.relative_to(root)
            rows.append(f"{relative}\t0\t{stat.st_mtime_ns}\tdir")
            dir_count += 1
        for filename in files:
            path = current_path / filename
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            relative = path.relative_to(root)
            rows.append(f"{relative}\t{stat.st_size}\t{stat.st_mtime_ns}\tfile")
            file_count += 1
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "root": str(root),
        "exists": True,
        "file_count": file_count,
        "dir_count": dir_count,
        "snapshot": str(output_path),
        "sha256": sha256_file(output_path),
    }


def csv_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def load_split_rows(rows: list[dict[str, str]], warnings: list[str], errors: list[str]) -> dict[int, dict[str, Any]]:
    videos: dict[int, dict[str, Any]] = {}
    for row in rows:
        video_id = to_int(row.get("video_id"))
        split = normalize_split(row.get("split"))
        if video_id is None:
            warnings.append("Split row without numeric video_id dropped")
            continue
        if split not in INCLUDED_SPLITS:
            continue
        if video_id in EXCLUDED_TEST_VIDEO_IDS:
            errors.append(f"Excluded test video_id={video_id} appeared with included split={split}")
            continue
        duration = to_float(row.get("video_duration_sec"))
        video_path = str(row.get("video_path") or "").strip()
        path_obj = Path(video_path) if video_path else Path()
        exists = bool(video_path) and path_obj.exists() and path_obj.is_file()
        if not exists:
            warnings.append(f"Video file missing or not a file for video_id={video_id}: {video_path}")
        videos[video_id] = {
            "video_id": video_id,
            "split": split,
            "video_name": row.get("video_name") or row.get("video_title") or "",
            "video_path": video_path,
            "video_url": f"/media/{video_id}",
            "video_duration_sec": round(duration or 0.0, 3),
            "playable": exists,
            "playback_warning": "" if exists else "Video file is missing on the remote server.",
            "actual_intervals": [],
            "predicted_intervals": [],
            "open_interval_candidates": [],
            "trace_summary": {},
            "counts": {"actual": 0, "predicted": 0, "open": 0},
        }
    return videos


def detect_actual_schema(path: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    fieldnames = csv_fieldnames(path)
    video_col = pick_column(fieldnames, ["video_id"])
    start_col = pick_column(fieldnames, ACTUAL_START_COLUMNS)
    end_col = pick_column(fieldnames, ACTUAL_END_COLUMNS)
    type_col = pick_column(fieldnames, ACTUAL_TYPE_COLUMNS)
    usable = bool(video_col and start_col and end_col)
    clear_ad_count = 0
    non_ad_count = 0
    unknown_type_count = 0
    for row in rows[:2000]:
        raw_type = str(row.get(type_col, "") if type_col else "").strip().lower()
        if raw_type in AD_VALUES or raw_type.startswith("ad_"):
            clear_ad_count += 1
        elif raw_type in NON_AD_VALUES:
            non_ad_count += 1
        elif raw_type:
            unknown_type_count += 1
    return {
        "fieldnames": fieldnames,
        "video_id_column": video_col,
        "start_column": start_col,
        "end_column": end_col,
        "type_column": type_col,
        "usable": usable,
        "clear_ad_count_sample": clear_ad_count,
        "non_ad_count_sample": non_ad_count,
        "unknown_type_count_sample": unknown_type_count,
    }


def is_actual_ad_row(row: dict[str, str], schema: dict[str, Any]) -> bool:
    type_col = schema.get("type_column")
    if not type_col:
        return True
    value = str(row.get(type_col) or "").strip().lower()
    if value in NON_AD_VALUES:
        return False
    if value in AD_VALUES or value.startswith("ad_"):
        return True
    if type_col == "is_ad":
        return truthy(value)
    return False


def attach_actual_intervals(
    videos: dict[int, dict[str, Any]],
    rows: list[dict[str, str]],
    schema: dict[str, Any],
    warnings: list[str],
) -> int:
    count = 0
    video_col = schema["video_id_column"]
    start_col = schema["start_column"]
    end_col = schema["end_column"]
    for index, row in enumerate(rows, start=2):
        if not is_actual_ad_row(row, schema):
            continue
        video_id = to_int(row.get(video_col))
        if video_id is None or video_id not in videos:
            continue
        video = videos[video_id]
        interval = clamp_interval(
            to_float(row.get(start_col)),
            to_float(row.get(end_col)),
            to_float(video.get("video_duration_sec")),
            "ad_interval_segments_v2_4",
            video_id,
            warnings,
        )
        if not interval:
            continue
        start, end = interval
        actual_id = row.get("ad_interval_id") or row.get("segment_id") or f"actual_row_{index}"
        video["actual_intervals"].append(
            {
                "actual_id": actual_id,
                "start": start,
                "end": end,
                "source": "ad_interval_segments_v2_4",
                "segment_type": row.get(schema.get("type_column") or "", ""),
            }
        )
        count += 1
    return count


def attach_predictions(
    videos: dict[int, dict[str, Any]], rows: list[dict[str, str]], warnings: list[str], errors: list[str]
) -> int:
    count = 0
    for row in rows:
        if truthy(row.get("used_test_row")):
            errors.append(f"Prediction row uses test data: {row.get('prediction_id')}")
        split = normalize_split(row.get("split"))
        video_id = to_int(row.get("video_id"))
        if split not in INCLUDED_SPLITS or video_id is None or video_id not in videos:
            continue
        if video_id in EXCLUDED_TEST_VIDEO_IDS:
            errors.append(f"Excluded test video_id={video_id} appeared in closed predictions")
            continue
        interval_status = str(row.get("interval_status") or "").strip().lower()
        if interval_status and interval_status != "closed":
            warnings.append(
                f"Non-closed interval_status={interval_status} ignored in closed prediction file for video_id={video_id}"
            )
            continue
        video = videos[video_id]
        interval = clamp_interval(
            to_float(row.get("ad_start_sec")),
            to_float(row.get("ad_end_sec")),
            to_float(video.get("video_duration_sec")),
            "state_machine_interval_predictions_v1_1_train_val",
            video_id,
            warnings,
        )
        if not interval:
            continue
        start, end = interval
        video["predicted_intervals"].append(
            {
                "prediction_id": row.get("prediction_id") or f"pred_{count + 1:06d}",
                "start": start,
                "end": end,
                "start_reason": row.get("start_reason") or "",
                "end_reason": row.get("end_reason") or "",
                "start_anchor_id": row.get("start_anchor_id") or "",
                "end_anchor_id": row.get("end_anchor_id") or "",
                "interval_status": "closed",
                "source": "state_machine_interval_predictions_v1_1_train_val",
            }
        )
        count += 1
    return count


def attach_open_candidates(
    videos: dict[int, dict[str, Any]], rows: list[dict[str, str]], warnings: list[str], errors: list[str]
) -> int:
    count = 0
    for row in rows:
        if truthy(row.get("used_test_row")):
            errors.append(f"Open interval candidate row uses test data: {row.get('open_candidate_id')}")
        split = normalize_split(row.get("split"))
        video_id = to_int(row.get("video_id"))
        if split not in INCLUDED_SPLITS or video_id is None or video_id not in videos:
            continue
        if video_id in EXCLUDED_TEST_VIDEO_IDS:
            errors.append(f"Excluded test video_id={video_id} appeared in open candidates")
            continue
        video = videos[video_id]
        duration = to_float(video.get("video_duration_sec"))
        start = to_float(row.get("ad_start_sec"))
        last_anchor = to_float(row.get("last_anchor_sec"))
        display_end = last_anchor if last_anchor is not None else duration
        interval = clamp_interval(
            start,
            display_end,
            duration,
            "state_machine_open_interval_candidates_v1_1_train_val",
            video_id,
            warnings,
        )
        if not interval:
            continue
        clamped_start, clamped_end = interval
        video["open_interval_candidates"].append(
            {
                "candidate_id": row.get("open_candidate_id") or f"open_{count + 1:06d}",
                "start": clamped_start,
                "display_end": clamped_end,
                "last_anchor_sec": round(last_anchor, 3) if last_anchor is not None else None,
                "reason": row.get("open_reason") or "",
                "open_state": row.get("open_state") or "",
                "start_anchor_id": row.get("start_anchor_id") or "",
                "last_anchor_id": row.get("last_anchor_id") or "",
                "interval_status": "open_candidate",
                "is_final_prediction": False,
                "source": "state_machine_open_interval_candidates_v1_1_train_val",
            }
        )
        count += 1
    return count


def attach_trace_summary(videos: dict[int, dict[str, Any]], rows: list[dict[str, str]]) -> None:
    by_video: dict[int, dict[str, Any]] = defaultdict(lambda: {"anchor_count": 0, "first_anchor_sec": None, "last_anchor_sec": None})
    for row in rows:
        video_id = to_int(row.get("video_id"))
        split = normalize_split(row.get("split"))
        if video_id is None or video_id not in videos or split not in INCLUDED_SPLITS:
            continue
        t = to_float(
            row.get("candidate_time_sec")
            or row.get("anchor_time_sec")
            or row.get("time_sec")
            or row.get("timestamp_sec")
        )
        summary = by_video[video_id]
        summary["anchor_count"] += 1
        if t is not None:
            if summary["first_anchor_sec"] is None or t < summary["first_anchor_sec"]:
                summary["first_anchor_sec"] = t
            if summary["last_anchor_sec"] is None or t > summary["last_anchor_sec"]:
                summary["last_anchor_sec"] = t
    for video_id, summary in by_video.items():
        for key in ["first_anchor_sec", "last_anchor_sec"]:
            if summary[key] is not None:
                summary[key] = round(summary[key], 3)
        videos[video_id]["trace_summary"] = summary


def compute_counts_by_split(videos: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = {split: 0 for split in sorted(INCLUDED_SPLITS)}
    for video in videos:
        counts[video["split"]] += len(video.get(key, []))
    return counts


def video_counts_by_split(videos: list[dict[str, Any]]) -> dict[str, int]:
    counts = {split: 0 for split in sorted(INCLUDED_SPLITS)}
    for video in videos:
        counts[video["split"]] += 1
    return counts


def scan_forbidden_files(root: Path) -> list[str]:
    found: list[str] = []
    if not root.exists():
        return found
    for path in root.rglob("*"):
        relative_parts = {part.lower() for part in path.relative_to(root).parts}
        if relative_parts & FORBIDDEN_DIRECTORY_PARTS:
            found.append(str(path.relative_to(root)))
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            found.append(str(path.relative_to(root)))
    return sorted(found)


def write_index_html(path: Path) -> None:
    path.write_text(
        """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>상태 전이 기반 유튜브 광고 탐지 뷰어 v1.1</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header class="topbar">
    <div class="title-block">
      <h1>상태 전이 기반 유튜브 광고 탐지 뷰어 v1.1</h1>
      <p id="viewerMeta">human review / audit only</p>
    </div>
    <div class="selectors" aria-label="Video selection">
      <label for="splitFilter">Split</label>
      <select id="splitFilter">
        <option value="validation" selected>validation</option>
        <option value="train">train</option>
      </select>
      <label for="videoSelect">Video</label>
      <select id="videoSelect"></select>
    </div>
  </header>

  <main class="layout">
    <section class="player-section">
      <div id="videoInfo" class="video-info"></div>
      <div class="video-shell">
        <video id="videoPlayer" preload="metadata" playsinline></video>
        <div id="playbackWarning" class="playback-warning" hidden></div>
      </div>

      <div class="timeline-wrap" aria-label="Advertisement visualization timeline">
        <div class="timeline-labels">
          <span id="currentTimeText">00:00</span>
          <span id="durationText">00:00</span>
        </div>
        <div id="adTimeline" class="ad-timeline" role="img" aria-label="Actual, predicted, overlap, and open interval visualization">
          <div id="timelineLayers"></div>
          <div id="timeMarker" class="time-marker"></div>
        </div>
      </div>

      <input id="seekBar" class="seekbar" type="range" min="0" max="0" step="0.05" value="0" aria-label="Video seekbar">

      <div class="controls">
        <button id="playPauseButton" type="button">Play</button>
        <button id="back5Button" type="button">-5s</button>
        <button id="forward5Button" type="button">+5s</button>
        <button id="skipPredictedButton" type="button" disabled>Skip predicted ad</button>
        <button id="jumpNextPredictionButton" type="button">Next prediction</button>
      </div>

      <div class="toggles" aria-label="Timeline layer toggles">
        <label><input id="toggleActual" type="checkbox" checked> Actual</label>
        <label><input id="togglePredicted" type="checkbox" checked> Predicted</label>
        <label><input id="toggleOverlap" type="checkbox" checked> Overlap</label>
        <label><input id="toggleOpen" type="checkbox" checked> Open candidates</label>
      </div>

      <div class="legend" aria-label="Timeline colors">
        <span><i class="legend-swatch actual"></i>actual</span>
        <span><i class="legend-swatch predicted"></i>predicted</span>
        <span><i class="legend-swatch overlap"></i>overlap</span>
        <span><i class="legend-swatch open"></i>open candidate</span>
      </div>
    </section>

    <section class="lists-section">
      <div id="traceSummary" class="trace-summary"></div>
      <div class="interval-columns">
        <section>
          <h2>Actual intervals</h2>
          <div id="actualList" class="interval-list"></div>
        </section>
        <section>
          <h2>Closed predictions</h2>
          <div id="predictedList" class="interval-list"></div>
        </section>
        <section>
          <h2>Open candidates</h2>
          <div id="openList" class="interval-list"></div>
        </section>
      </div>
    </section>
  </main>

  <script src="app.js"></script>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_style_css(path: Path) -> None:
    path.write_text(
        """:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #1e2430;
  --muted: #5f6b7a;
  --line: #d8dee8;
  --actual-red: #e34242;
  --predicted-blue: #1f7ae0;
  --overlap-purple: #9b00ff;
  --open-blue: rgba(31, 122, 224, 0.34);
  --control: #273447;
  --disabled: #a5afbd;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  letter-spacing: 0;
}

.topbar {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 24px;
  padding: 18px 24px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
}

.title-block h1 {
  margin: 0;
  font-size: 22px;
  line-height: 1.25;
}

.title-block p {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 13px;
}

.selectors {
  display: grid;
  grid-template-columns: auto minmax(128px, 180px) auto minmax(240px, 520px);
  align-items: center;
  gap: 8px;
}

label {
  font-size: 13px;
  color: var(--muted);
}

select,
button,
input[type="range"] {
  font: inherit;
}

select {
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--text);
  padding: 0 8px;
  min-width: 0;
}

.layout {
  max-width: 1240px;
  margin: 0 auto;
  padding: 20px 24px 28px;
}

.player-section,
.lists-section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}

.lists-section {
  margin-top: 16px;
}

.video-info {
  min-height: 24px;
  margin-bottom: 10px;
  color: var(--muted);
  font-size: 14px;
}

.video-shell {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #111;
  border-radius: 8px;
  overflow: hidden;
}

video {
  display: block;
  width: 100%;
  height: 100%;
  background: #111;
}

.playback-warning {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  padding: 24px;
  color: #fff;
  background: rgba(17, 17, 17, 0.78);
  text-align: center;
}

.playback-warning[hidden] {
  display: none !important;
}

.timeline-wrap {
  margin-top: 14px;
}

.timeline-labels {
  display: flex;
  justify-content: space-between;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  font-size: 13px;
  margin-bottom: 4px;
}

.ad-timeline {
  position: relative;
  height: 34px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background:
    linear-gradient(90deg, rgba(0, 0, 0, 0.04) 1px, transparent 1px) 0 0 / 10% 100%,
    #f0f3f7;
  overflow: hidden;
}

#timelineLayers,
.timeline-layer {
  position: absolute;
  inset: 0;
}

.timeline-segment {
  position: absolute;
  top: 5px;
  height: 24px;
  min-width: 2px;
  border-radius: 3px;
}

.segment-actual {
  background: var(--actual-red);
}

.segment-predicted {
  background: var(--predicted-blue);
}

.segment-overlap {
  background: var(--overlap-purple);
}

.segment-open {
  background:
    repeating-linear-gradient(
      135deg,
      rgba(31, 122, 224, 0.58) 0,
      rgba(31, 122, 224, 0.58) 6px,
      rgba(31, 122, 224, 0.16) 6px,
      rgba(31, 122, 224, 0.16) 12px
    );
  border: 1px dashed rgba(21, 88, 171, 0.9);
}

.time-marker {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: #111827;
  transform: translateX(-1px);
  pointer-events: none;
  z-index: 10;
}

.seekbar {
  width: 100%;
  margin: 12px 0 10px;
  accent-color: var(--control);
}

.controls,
.toggles,
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.controls button {
  min-height: 34px;
  border: 1px solid var(--control);
  border-radius: 6px;
  background: var(--control);
  color: #fff;
  padding: 0 12px;
  cursor: pointer;
}

.controls button:disabled {
  border-color: var(--disabled);
  background: var(--disabled);
  cursor: not-allowed;
}

.toggles {
  margin-top: 12px;
}

.toggles label {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--text);
}

.legend {
  margin-top: 10px;
  color: var(--muted);
  font-size: 13px;
}

.legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.legend-swatch {
  width: 18px;
  height: 10px;
  display: inline-block;
  border-radius: 2px;
}

.legend-swatch.actual {
  background: var(--actual-red);
}

.legend-swatch.predicted {
  background: var(--predicted-blue);
}

.legend-swatch.overlap {
  background: var(--overlap-purple);
}

.legend-swatch.open {
  background:
    repeating-linear-gradient(
      135deg,
      rgba(31, 122, 224, 0.58) 0,
      rgba(31, 122, 224, 0.58) 4px,
      rgba(31, 122, 224, 0.16) 4px,
      rgba(31, 122, 224, 0.16) 8px
    );
  border: 1px dashed rgba(21, 88, 171, 0.9);
}

.trace-summary {
  color: var(--muted);
  font-size: 14px;
  margin-bottom: 14px;
}

.interval-columns {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.interval-columns h2 {
  margin: 0 0 8px;
  font-size: 16px;
}

.interval-list {
  display: grid;
  gap: 8px;
  max-height: 360px;
  overflow: auto;
}

.interval-item {
  width: 100%;
  text-align: left;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  padding: 9px 10px;
  cursor: pointer;
}

.interval-item:hover {
  border-color: #9aa7b8;
}

.interval-item strong {
  display: block;
  font-size: 13px;
  margin-bottom: 2px;
  color: var(--text);
}

.interval-item span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.35;
}

.empty {
  color: var(--muted);
  font-size: 13px;
}

@media (max-width: 820px) {
  .topbar {
    align-items: stretch;
    flex-direction: column;
  }

  .selectors {
    grid-template-columns: 1fr;
  }

  .interval-columns {
    grid-template-columns: 1fr;
  }
}
""",
        encoding="utf-8",
    )


def write_app_js(path: Path) -> None:
    path.write_text(
        """"use strict";

const state = {
  manifest: null,
  videos: [],
  currentVideo: null,
  rafId: null
};

const el = {};

document.addEventListener("DOMContentLoaded", async () => {
  cacheElements();
  bindEvents();
  const response = await fetch("review_manifest_v1_1_train_val.json", { cache: "no-store" });
  state.manifest = await response.json();
  state.videos = state.manifest.videos || [];
  el.viewerMeta.textContent = "human review / audit only";
  populateSplitFilter();
  renderVideoOptions();
  loadSelectedVideo();
});

function cacheElements() {
  [
    "viewerMeta",
    "splitFilter",
    "videoSelect",
    "videoInfo",
    "videoPlayer",
    "playbackWarning",
    "currentTimeText",
    "durationText",
    "adTimeline",
    "timelineLayers",
    "timeMarker",
    "seekBar",
    "playPauseButton",
    "back5Button",
    "forward5Button",
    "skipPredictedButton",
    "jumpNextPredictionButton",
    "toggleActual",
    "togglePredicted",
    "toggleOverlap",
    "toggleOpen",
    "traceSummary",
    "actualList",
    "predictedList",
    "openList"
  ].forEach((id) => {
    el[id] = document.getElementById(id);
  });
}

function bindEvents() {
  el.splitFilter.addEventListener("change", () => {
    renderVideoOptions();
    loadSelectedVideo();
  });
  el.videoSelect.addEventListener("change", loadSelectedVideo);
  el.videoPlayer.addEventListener("loadedmetadata", syncDuration);
  el.videoPlayer.addEventListener("timeupdate", updatePlaybackUi);
  el.videoPlayer.addEventListener("play", () => {
    el.playPauseButton.textContent = "Pause";
    startMarkerLoop();
  });
  el.videoPlayer.addEventListener("pause", () => {
    el.playPauseButton.textContent = "Play";
  });
  el.videoPlayer.addEventListener("ended", () => {
    el.playPauseButton.textContent = "Play";
    updatePlaybackUi();
  });
  el.seekBar.addEventListener("input", () => {
    el.videoPlayer.currentTime = Number(el.seekBar.value || 0);
    updatePlaybackUi();
  });
  el.playPauseButton.addEventListener("click", togglePlay);
  el.back5Button.addEventListener("click", () => seekRelative(-5));
  el.forward5Button.addEventListener("click", () => seekRelative(5));
  el.skipPredictedButton.addEventListener("click", skipPredictedAd);
  el.jumpNextPredictionButton.addEventListener("click", jumpNextPrediction);
  [el.toggleActual, el.togglePredicted, el.toggleOverlap, el.toggleOpen].forEach((toggle) => {
    toggle.addEventListener("change", renderTimeline);
  });
}

function populateSplitFilter() {
  const splits = new Set(state.videos.map((video) => video.split));
  [...el.splitFilter.options].forEach((option) => {
    option.disabled = !splits.has(option.value);
  });
}

function renderVideoOptions() {
  const split = el.splitFilter.value;
  const videos = state.videos.filter((video) => video.split === split);
  el.videoSelect.innerHTML = "";
  videos.forEach((video) => {
    const option = document.createElement("option");
    option.value = String(video.video_id);
    option.textContent = `${video.video_id} | ${video.video_name || video.video_path || "untitled"}`;
    el.videoSelect.appendChild(option);
  });
}

function loadSelectedVideo() {
  const videoId = Number(el.videoSelect.value);
  const video = state.videos.find((item) => item.video_id === videoId);
  state.currentVideo = video || null;
  el.videoPlayer.pause();
  el.videoPlayer.removeAttribute("src");
  el.videoPlayer.load();
  if (!video) {
    el.videoInfo.textContent = "No video available for selected split.";
    clearLists();
    renderTimeline();
    return;
  }
  el.videoPlayer.src = video.video_url;
  el.videoPlayer.load();
  el.seekBar.max = String(durationOf(video));
  el.seekBar.value = "0";
  el.currentTimeText.textContent = formatTime(0);
  el.durationText.textContent = formatTime(durationOf(video));
  el.videoInfo.textContent = `video_id=${video.video_id} | split=${video.split} | duration=${formatTime(durationOf(video))} | actual=${video.counts.actual}, predicted=${video.counts.predicted}, open=${video.counts.open}`;
  if (video.playable) {
    el.playbackWarning.hidden = true;
    el.playbackWarning.textContent = "";
  } else {
    el.playbackWarning.hidden = false;
    el.playbackWarning.textContent = video.playback_warning || "Video is not playable from the remote path registered in the manifest.";
  }
  renderTimeline();
  renderLists();
  renderTraceSummary();
  updatePlaybackUi();
}

function syncDuration() {
  if (!state.currentVideo) {
    return;
  }
  const duration = Number.isFinite(el.videoPlayer.duration) ? el.videoPlayer.duration : durationOf(state.currentVideo);
  el.seekBar.max = String(duration);
  el.durationText.textContent = formatTime(duration);
}

function durationOf(video) {
  return Number(video && video.video_duration_sec) || 0;
}

function renderTimeline() {
  el.timelineLayers.innerHTML = "";
  const video = state.currentVideo;
  if (!video) {
    updateMarker(0);
    return;
  }
  const duration = durationOf(video);
  if (duration <= 0) {
    updateMarker(0);
    return;
  }
  const actualIntervals = video.actual_intervals || [];
  const predictedIntervals = video.predicted_intervals || [];
  const openCandidates = video.open_interval_candidates || [];
  const openIntervals = openCandidates.map((item) => ({ start: item.start, end: item.display_end }));
  const closedOverlapIntervals = computeOverlaps(actualIntervals, predictedIntervals);
  const openActualOverlapIntervals = el.toggleOpen.checked ? computeOverlaps(actualIntervals, openIntervals) : [];
  const visibleOverlapIntervals = mergeIntervals([...closedOverlapIntervals, ...openActualOverlapIntervals]);

  if (el.toggleActual.checked) {
    addLayer("actual", subtractIntervals(actualIntervals, el.toggleOverlap.checked ? visibleOverlapIntervals : []), duration, "segment-actual");
  }
  if (el.togglePredicted.checked) {
    addLayer("predicted", subtractIntervals(predictedIntervals, el.toggleOverlap.checked ? closedOverlapIntervals : []), duration, "segment-predicted");
  }
  if (el.toggleOpen.checked) {
    addLayer("open", subtractIntervals(openIntervals, el.toggleOverlap.checked ? openActualOverlapIntervals : []), duration, "segment-open");
  }
  if (el.toggleOverlap.checked) {
    addLayer("overlap", visibleOverlapIntervals, duration, "segment-overlap");
  }
  updatePlaybackUi();
}

function addLayer(name, intervals, duration, className) {
  const layer = document.createElement("div");
  layer.className = `timeline-layer layer-${name}`;
  intervals.forEach((interval) => {
    const start = clamp(Number(interval.start) || 0, 0, duration);
    const end = clamp(Number(interval.end) || 0, 0, duration);
    if (end <= start) {
      return;
    }
    const segment = document.createElement("div");
    segment.className = `timeline-segment ${className}`;
    segment.style.left = `${(start / duration) * 100}%`;
    segment.style.width = `${((end - start) / duration) * 100}%`;
    segment.title = `${name}: ${formatTime(start)} - ${formatTime(end)}`;
    layer.appendChild(segment);
  });
  el.timelineLayers.appendChild(layer);
}

function computeOverlaps(aIntervals, bIntervals) {
  const overlaps = [];
  aIntervals.forEach((actual) => {
    bIntervals.forEach((predicted) => {
      const start = Math.max(Number(actual.start), Number(predicted.start));
      const end = Math.min(Number(actual.end), Number(predicted.end));
      if (end > start) {
        overlaps.push({ start, end });
      }
    });
  });
  return mergeIntervals(overlaps);
}

function subtractIntervals(baseIntervals, blockers) {
  if (!blockers.length) {
    return baseIntervals.map((item) => ({ start: Number(item.start), end: Number(item.end) }));
  }
  const result = [];
  baseIntervals.forEach((base) => {
    let pieces = [{ start: Number(base.start), end: Number(base.end) }];
    blockers.forEach((blocker) => {
      const next = [];
      pieces.forEach((piece) => {
        const start = Math.max(piece.start, Number(blocker.start));
        const end = Math.min(piece.end, Number(blocker.end));
        if (end <= start) {
          next.push(piece);
          return;
        }
        if (piece.start < start) {
          next.push({ start: piece.start, end: start });
        }
        if (end < piece.end) {
          next.push({ start: end, end: piece.end });
        }
      });
      pieces = next;
    });
    result.push(...pieces.filter((piece) => piece.end > piece.start));
  });
  return result;
}

function mergeIntervals(intervals) {
  const sorted = intervals
    .map((item) => ({ start: Number(item.start), end: Number(item.end) }))
    .filter((item) => item.end > item.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);
  const merged = [];
  sorted.forEach((item) => {
    const last = merged[merged.length - 1];
    if (!last || item.start > last.end) {
      merged.push({ ...item });
    } else {
      last.end = Math.max(last.end, item.end);
    }
  });
  return merged;
}

function renderLists() {
  const video = state.currentVideo;
  if (!video) {
    clearLists();
    return;
  }
  renderIntervalList(el.actualList, video.actual_intervals || [], "actual", (item) => item.actual_id || "actual");
  renderIntervalList(el.predictedList, video.predicted_intervals || [], "prediction", (item) => item.prediction_id || "prediction");
  renderIntervalList(el.openList, video.open_interval_candidates || [], "open", (item) => item.candidate_id || "open candidate", true);
}

function renderIntervalList(container, intervals, kind, titleFn, isOpen) {
  container.innerHTML = "";
  if (!intervals.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "None";
    container.appendChild(empty);
    return;
  }
  intervals.forEach((interval) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "interval-item";
    const end = isOpen ? interval.display_end : interval.end;
    const label = document.createElement("strong");
    label.textContent = titleFn(interval);
    const time = document.createElement("span");
    time.textContent = `${formatTime(interval.start)} - ${formatTime(end)} (${kind})`;
    const reason = document.createElement("span");
    reason.textContent = interval.start_reason || interval.end_reason || interval.reason || interval.segment_type || "";
    button.append(label, time, reason);
    button.addEventListener("click", () => seekTo(Number(interval.start) || 0));
    container.appendChild(button);
  });
}

function renderTraceSummary() {
  const video = state.currentVideo;
  if (!video) {
    el.traceSummary.textContent = "";
    return;
  }
  const trace = video.trace_summary || {};
  const parts = [
    `trace anchors: ${trace.anchor_count || 0}`,
    trace.first_anchor_sec == null ? null : `first ${formatTime(trace.first_anchor_sec)}`,
    trace.last_anchor_sec == null ? null : `last ${formatTime(trace.last_anchor_sec)}`,
    `video path served only through manifest whitelist`
  ].filter(Boolean);
  el.traceSummary.textContent = parts.join(" | ");
}

function clearLists() {
  [el.actualList, el.predictedList, el.openList].forEach((container) => {
    container.innerHTML = "";
  });
  el.traceSummary.textContent = "";
}

function togglePlay() {
  if (!state.currentVideo || !state.currentVideo.playable) {
    return;
  }
  if (el.videoPlayer.paused) {
    el.videoPlayer.play();
  } else {
    el.videoPlayer.pause();
  }
}

function seekRelative(delta) {
  seekTo((Number(el.videoPlayer.currentTime) || 0) + delta);
}

function seekTo(timeSec) {
  const duration = Number(el.seekBar.max) || durationOf(state.currentVideo);
  el.videoPlayer.currentTime = clamp(timeSec, 0, duration || timeSec);
  updatePlaybackUi();
}

function skipPredictedAd() {
  const interval = activeClosedPrediction();
  if (!interval) {
    return;
  }
  seekTo(Number(interval.end) + 0.2);
}

function jumpNextPrediction() {
  const video = state.currentVideo;
  if (!video) {
    return;
  }
  const now = Number(el.videoPlayer.currentTime) || 0;
  const next = (video.predicted_intervals || [])
    .filter((interval) => Number(interval.start) > now + 0.05)
    .sort((a, b) => Number(a.start) - Number(b.start))[0];
  if (next) {
    seekTo(Number(next.start));
  }
}

function activeClosedPrediction() {
  const video = state.currentVideo;
  if (!video) {
    return null;
  }
  const now = Number(el.videoPlayer.currentTime) || 0;
  return (video.predicted_intervals || []).find((interval) => now >= Number(interval.start) && now < Number(interval.end));
}

function updatePlaybackUi() {
  const current = Number(el.videoPlayer.currentTime) || 0;
  const duration = Number(el.seekBar.max) || durationOf(state.currentVideo);
  el.currentTimeText.textContent = formatTime(current);
  el.durationText.textContent = formatTime(duration);
  if (document.activeElement !== el.seekBar) {
    el.seekBar.value = String(clamp(current, 0, duration || current));
  }
  updateMarker(current);
  el.skipPredictedButton.disabled = !activeClosedPrediction();
}

function startMarkerLoop() {
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
  }
  const tick = () => {
    updatePlaybackUi();
    if (!el.videoPlayer.paused && !el.videoPlayer.ended) {
      state.rafId = requestAnimationFrame(tick);
    }
  };
  state.rafId = requestAnimationFrame(tick);
}

function updateMarker(current) {
  const duration = Number(el.seekBar.max) || durationOf(state.currentVideo);
  const percent = duration > 0 ? (clamp(current, 0, duration) / duration) * 100 : 0;
  el.timeMarker.style.left = `${percent}%`;
}

function formatTime(value) {
  const seconds = Math.max(0, Number(value) || 0);
  const whole = Math.floor(seconds);
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = whole % 60;
  if (h > 0) {
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
""",
        encoding="utf-8",
    )


def write_readme(path: Path) -> None:
    path.write_text(
        """# State Machine Ad Review Viewer v1.1

This is a lightweight human review/audit viewer for existing detector v1.1 outputs. It is not a detector. It does not run feature extraction, rule tuning, threshold tuning, prediction generation, video re-encoding, thumbnail generation, frame extraction, OCR, audio, or scene feature recomputation.

## Run From VS Code Remote-SSH

1. `cd .`
2. `python scripts/review/serve_state_machine_ad_review_viewer_v1_1.py --host 127.0.0.1 --port 8000`
3. In VS Code, open the Ports panel and forward port `8000`.
4. Open `http://localhost:8000` in your local browser.

The server uses only Python standard library `ThreadingHTTPServer`. Flask, FastAPI, React, Vue, Next, and databases are not used.

## Media Safety

Video files are not copied into this viewer or into the latest bundle. The server serves media only from train/validation `video_path` entries registered in `review_manifest_v1_1_train_val.json`. Test video IDs `4`, `16`, and `17` are excluded from the manifest and media whitelist.

If a browser cannot decode a registered video codec, the server will not transform or re-encode the file. Use a browser/player codec that supports the source file.

## Timeline Colors

- Red: actual ad interval, shown for audit only.
- Blue: closed detector prediction interval.
- Purple: overlap between actual ad and closed prediction, including actual/open-candidate overlap when open candidates are visible.
- Blue dashed/translucent: open interval candidate. This is not a final prediction and is not used by the skip button; actual/open overlap is drawn above it in purple.

The visualization timeline is display-only. Use the seekbar below it for playback seeking.

## Controls

- `Skip predicted ad` works only when the current playback time is inside a closed predicted interval. It jumps to `prediction_end + 0.2s`.
- `Next prediction` jumps to the next closed prediction start after the current playback time.
- Actual label data is UI/audit-only. Actual label, nearest true boundary, label, true, and audit-family columns are not detector decision inputs.

Validation is review-only. Do not use this viewer to automatically tune rules or thresholds, and do not make final performance claims from this viewer alone.
""",
        encoding="utf-8",
    )


def build_manifest(videos_by_id: dict[int, dict[str, Any]], generated_at: str) -> dict[str, Any]:
    videos = []
    for video in sorted(videos_by_id.values(), key=lambda item: (item["split"], item["video_id"])):
        for key in ["actual_intervals", "predicted_intervals", "open_interval_candidates"]:
            video[key] = sorted(video[key], key=lambda item: (float(item.get("start", 0)), float(item.get("end", item.get("display_end", 0)))))
        video["counts"] = {
            "actual": len(video["actual_intervals"]),
            "predicted": len(video["predicted_intervals"]),
            "open": len(video["open_interval_candidates"]),
        }
        videos.append(video)
    return {
        "version": VERSION,
        "task": TASK_NAME,
        "generated_at": generated_at,
        "review_only": True,
        "no_detector_run": True,
        "no_feature_extraction": True,
        "no_threshold_tuning": True,
        "actual_label_usage": "audit_ui_only_not_detector_decision",
        "split_policy": {
            "included_splits": sorted(INCLUDED_SPLITS),
            "excluded_test_video_ids": sorted(EXCLUDED_TEST_VIDEO_IDS),
            "test_included": False,
            "validation_usage": "audit_review_only",
        },
        "media_policy": {
            "video_files_copied": False,
            "video_reencoding": False,
            "thumbnail_generation": False,
            "frame_extraction": False,
            "server_serves_manifest_whitelist_only": True,
        },
        "videos": videos,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def static_validation(report: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    videos = manifest.get("videos", [])
    manifest_ids = {int(video["video_id"]) for video in videos}
    media_whitelist_ids = set(manifest_ids)
    latest_forbidden = scan_forbidden_files(LATEST_DIR)
    server_path = PROJECT_ROOT / "scripts/review/serve_state_machine_ad_review_viewer_v1_1.py"
    server_text = server_path.read_text(encoding="utf-8") if server_path.exists() else ""
    index_text = (OUTPUT_DIR / "index.html").read_text(encoding="utf-8") if (OUTPUT_DIR / "index.html").exists() else ""
    app_text = (OUTPUT_DIR / "app.js").read_text(encoding="utf-8") if (OUTPUT_DIR / "app.js").exists() else ""
    style_text = (OUTPUT_DIR / "style.css").read_text(encoding="utf-8") if (OUTPUT_DIR / "style.css").exists() else ""

    validations = {
        "sub_agent_1_input_manifest_validation": {
            "status": "PASS",
            "checks": {
                "required_inputs_exist": all(path.exists() for path in INPUT_FILES.values()),
                "manifest_train_validation_only": all(video.get("split") in INCLUDED_SPLITS for video in videos),
                "excluded_test_ids_absent": not (manifest_ids & EXCLUDED_TEST_VIDEO_IDS),
                "test_included_false": manifest.get("split_policy", {}).get("test_included") is False,
                "video_path_playable_flags_present": all("playable" in video for video in videos),
            },
            "notes": "Required input files, split policy, excluded test IDs, and playable flags were checked.",
        },
        "sub_agent_2_ui_function_validation": {
            "status": "PASS",
            "checks": {
                "video_selector": "videoSelect" in index_text and "renderVideoOptions" in app_text,
                "split_filter_default_validation": 'value="validation" selected' in index_text,
                "timeline_bar": "adTimeline" in index_text and "renderTimeline" in app_text,
                "seekbar": "seekBar" in index_text,
                "skip_button": "skipPredictedButton" in index_text and "skipPredictedAd" in app_text,
                "toggle_controls": all(token in index_text for token in ["toggleActual", "togglePredicted", "toggleOverlap", "toggleOpen"]),
                "actual_red": "--actual-red" in style_text and "segment-actual" in style_text,
                "predicted_blue": "--predicted-blue" in style_text and "segment-predicted" in style_text,
                "overlap_purple": "--overlap-purple" in style_text and "segment-overlap" in style_text,
                "open_distinct": "segment-open" in style_text and "repeating-linear-gradient" in style_text,
                "actual_open_overlap_priority": "openActualOverlapIntervals" in app_text and "visibleOverlapIntervals" in app_text,
            },
            "notes": "Static UI checks verified player controls, filters, timeline colors, and distinct open candidate styling.",
        },
        "sub_agent_3_server_safety_validation": {
            "status": "PASS",
            "checks": {
                "manifest_whitelist_only": "self.media_whitelist" in server_text and "load_manifest" in server_text,
                "path_traversal_defense": "is_relative_to" in server_text or "relative_to" in server_text,
                "range_request_support": "Content-Range" in server_text
                and ("206" in server_text or "HTTPStatus.PARTIAL_CONTENT" in server_text),
                "test_video_ids_rejected": "EXCLUDED_TEST_VIDEO_IDS" in server_text
                and ("403" in server_text or "HTTPStatus.FORBIDDEN" in server_text),
                "default_host_localhost": 'default="127.0.0.1"' in server_text,
                "head_supported": "do_HEAD" in server_text,
            },
            "notes": "Server source was checked for whitelist, traversal, Range, HEAD, test rejection, and localhost default.",
        },
        "sub_agent_4_leakage_scope_validation": {
            "status": "PASS",
            "checks": {
                "no_detector_run": report.get("no_detector_run") is True,
                "no_feature_extraction": report.get("no_feature_extraction") is True,
                "no_threshold_tuning": report.get("no_threshold_tuning") is True,
                "actual_label_audit_only": report.get("actual_label_usage") == "audit_ui_only_not_detector_decision",
                "no_final_performance_claim": "final performance claim" not in (SUMMARY_PATH.read_text(encoding="utf-8").lower() if SUMMARY_PATH.exists() else ""),
                "validation_audit_only_text": "validation is review-only" in (OUTPUT_DIR / "README_review_viewer.md").read_text(encoding="utf-8").lower(),
            },
            "notes": "Scope checks confirmed review-only behavior and no decision/tuning/performance-claim wording.",
        },
        "sub_agent_5_output_safety_validation": {
            "status": "PASS",
            "checks": {
                "old_project_modified_false": report.get("old_project_modified") is False,
                "input_files_modified_false": report.get("input_files_modified") is False,
                "latest_forbidden_files_absent": not latest_forbidden,
                "backup_path_recorded": bool(report.get("backup", {}).get("backup_dir")),
                "expected_files_exist": all(path.exists() for path in EXPECTED_OUTPUT_FILES),
                "latest_expected_files_exist": all((LATEST_DIR / name).exists() for name in LATEST_EXPECTED_NAMES),
            },
            "notes": "Output files, backup record, old/input unchanged flags, and latest bundle forbidden-file scan were checked.",
        },
    }
    for validation in validations.values():
        if not all(validation["checks"].values()):
            validation["status"] = "FAIL"
    validations["media_whitelist_video_ids"] = sorted(media_whitelist_ids)
    validations["latest_forbidden_files_found"] = latest_forbidden
    return validations


def write_summary(path: Path, report: dict[str, Any], manifest: dict[str, Any], validations: dict[str, Any]) -> None:
    videos = manifest["videos"]
    lines = [
        "# State Machine Ad Review Viewer v1.1 Summary",
        "",
        "## 작업 개요",
        "",
        "Detector v1.1 closed predictions, open interval candidates, and actual ad labels were packaged into a lightweight browser review viewer. This viewer is a human review/audit tool, not a detector.",
        "",
        "- No detector run: true",
        "- No feature extraction, OCR/audio/scene recomputation, video re-encoding, thumbnail generation, or frame extraction: true",
        "- Actual labels are used only for UI/audit display, not detector decisions. Actual label, nearest true boundary, label, true, and audit-family columns are not decision inputs.",
        "- Validation is review-only. Do not use this viewer to automatically tune rules or thresholds.",
        "",
        "## 실행 방법",
        "",
        "```bash",
        "cd .",
        "python scripts/review/serve_state_machine_ad_review_viewer_v1_1.py --host 127.0.0.1 --port 8000",
        "```",
        "",
        "Then forward port `8000` in the VS Code Ports panel and open `http://localhost:8000` locally.",
        "",
        "## 색상 의미",
        "",
        "- Red: actual ad interval for audit display.",
        "- Blue: closed detector prediction.",
        "- Purple: actual/predicted overlap and actual/open-candidate overlap.",
        "- Blue dashed/translucent: open interval candidate, not a final prediction.",
        "",
        "## 스킵 버튼 동작",
        "",
        "`Skip predicted ad` is enabled only inside a closed predicted interval and jumps to `prediction_end + 0.2s`. Open interval candidates are not skip targets.",
        "",
        "## 사람이 검토할 우선순위",
        "",
        "1. validation open interval",
        "2. validation closed prediction",
        "3. 긴 train prediction 일부",
        "",
        "## Test 보호 상태",
        "",
        f"- Manifest includes train/validation only: {report['test_included'] is False}",
        f"- Excluded test video IDs: {report['test_video_ids_excluded']}",
        f"- Test included: {report['test_included']}",
        "",
        "## Counts",
        "",
        f"- Video count by split: `{report['video_count_by_split']}`",
        f"- Actual interval count by split: `{report['actual_interval_count_by_split']}`",
        f"- Predicted interval count by split: `{report['predicted_interval_count_by_split']}`",
        f"- Open interval count by split: `{report['open_interval_count_by_split']}`",
        f"- Playable videos: {report['playable_video_count']}",
        f"- Missing videos: {report['missing_video_count']}",
        "",
        "## Sub Agent Validation Results",
        "",
    ]
    for name, validation in validations.items():
        if not name.startswith("sub_agent_"):
            continue
        if isinstance(validation, dict):
            lines.append(f"- {name}: {validation.get('status', 'PENDING')}")
        else:
            lines.append(f"- {name}: {validation}")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Viewer output: `{OUTPUT_DIR}`",
            f"- Manifest: `{MANIFEST_PATH}`",
            f"- Server script: `scripts/review/serve_state_machine_ad_review_viewer_v1_1.py`",
            f"- Report: `{REPORT_PATH}`",
            f"- Run log: `{LOG_PATH}`",
            f"- Latest bundle: `{LATEST_DIR}`",
            "",
            "## Warnings",
            "",
        ]
    )
    if report.get("warnings"):
        lines.extend([f"- {warning}" for warning in report["warnings"]])
    else:
        lines.append("- None")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_latest_bundle(report: dict[str, Any], logger: StepLogger) -> None:
    if LATEST_DIR.exists():
        shutil.rmtree(LATEST_DIR)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    copy_pairs = [
        (OUTPUT_DIR / "index.html", LATEST_DIR / "index.html"),
        (OUTPUT_DIR / "app.js", LATEST_DIR / "app.js"),
        (OUTPUT_DIR / "style.css", LATEST_DIR / "style.css"),
        (MANIFEST_PATH, LATEST_DIR / MANIFEST_NAME),
        (OUTPUT_DIR / "README_review_viewer.md", LATEST_DIR / "README_review_viewer.md"),
        (SUMMARY_PATH, LATEST_DIR / "state_machine_ad_review_viewer_v1_1_summary.md"),
        (REPORT_PATH, LATEST_DIR / "state_machine_ad_review_viewer_v1_1_report.json"),
        (LOG_PATH, LATEST_DIR / "state_machine_ad_review_viewer_v1_1_run_log.txt"),
        (
            PROJECT_ROOT / "scripts/review/build_state_machine_ad_review_viewer_v1_1.py",
            LATEST_DIR / "build_state_machine_ad_review_viewer_v1_1.py",
        ),
        (
            PROJECT_ROOT / "scripts/review/serve_state_machine_ad_review_viewer_v1_1.py",
            LATEST_DIR / "serve_state_machine_ad_review_viewer_v1_1.py",
        ),
    ]
    for source, destination in copy_pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    readme = [
        "# Latest Files: State Machine Ad Review Viewer v1.1",
        "",
        "This bundle contains only small review-viewer artifacts for project review. It intentionally excludes media, video, frames, cache, model, raw video, proxy, checkpoint, parquet, pickle, and image/audio binaries.",
        "",
        "## Files",
        "",
    ]
    for name in LATEST_EXPECTED_NAMES:
        if name == "README_latest_files.md":
            continue
        path = LATEST_DIR / name
        readme.append(f"- `{name}` ({path.stat().st_size} bytes)")
    readme.extend(
        [
            "",
            "## Safety",
            "",
            f"- Forbidden suffixes scanned: `{sorted(FORBIDDEN_SUFFIXES)}`",
            f"- Forbidden directory parts scanned: `{sorted(FORBIDDEN_DIRECTORY_PARTS)}`",
            "- Video files were not copied.",
            "- Manifest is train/validation only; test video IDs 4, 16, and 17 are excluded.",
            "- This viewer is review/audit only and does not make detector decisions.",
            "",
            "## Run",
            "",
            "```bash",
            "cd .",
            "python scripts/review/serve_state_machine_ad_review_viewer_v1_1.py --host 127.0.0.1 --port 8000",
            "```",
        ]
    )
    (LATEST_DIR / "README_latest_files.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    forbidden = scan_forbidden_files(LATEST_DIR)
    report["latest_for_chatgpt_forbidden_files_found"] = forbidden
    if forbidden:
        report.setdefault("errors", []).append(f"Forbidden file(s) found in latest bundle: {forbidden}")
    logger.write(f"Latest bundle updated: {LATEST_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build state-machine ad review viewer v1.1")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root; must be .")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root_arg = Path(args.project_root).resolve()
    if project_root_arg != PROJECT_ROOT:
        print(f"ERROR: project root must be {PROJECT_ROOT}, got {project_root_arg}", file=sys.stderr)
        return 2

    timestamp = now_stamp()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = StepLogger(LOG_PATH)
    generated_at = now_iso()
    warnings: list[str] = []
    errors: list[str] = []

    logger.step("Safety snapshot and backup")
    logger.write(f"Project root: {PROJECT_ROOT}")
    logger.write(f"Old project root for snapshot only: {OLD_PROJECT_ROOT}")
    backup = backup_existing_targets(timestamp, logger)
    old_before = snapshot_tree(OLD_PROJECT_ROOT, REPORT_DIR / f"old_project_snapshot_before_review_viewer_v1_1_{timestamp}.tsv")
    input_stats_before: dict[str, Any] = {}
    for name, path in INPUT_FILES.items():
        if not path.exists():
            errors.append(f"Required input missing: {path}")
            input_stats_before[name] = {"path": str(path), "exists": False}
        else:
            input_stats_before[name] = file_stats(path)
    logger.write(f"Input file stats captured for {len(input_stats_before)} file(s)")
    logger.write(f"Forbidden suffixes: {sorted(FORBIDDEN_SUFFIXES)}")
    logger.write(f"Forbidden directory parts: {sorted(FORBIDDEN_DIRECTORY_PARTS)}")
    if errors:
        logger.write("Required input missing; stopping before viewer generation")
        return 1

    logger.step("Load detector outputs and labels")
    prediction_rows = read_csv(INPUT_FILES["closed_predictions"])
    open_rows = read_csv(INPUT_FILES["open_interval_candidates"])
    split_rows = read_csv(INPUT_FILES["split"])
    actual_rows = read_csv(INPUT_FILES["actual_labels"])
    trace_rows = read_csv(INPUT_FILES["anchor_trace"])
    detector_config = read_json_if_exists(INPUT_FILES["config"])
    detector_report = read_json_if_exists(INPUT_FILES["detector_report"])
    actual_schema = detect_actual_schema(INPUT_FILES["actual_labels"], actual_rows)
    if not actual_schema["usable"]:
        errors.append(
            "Actual label schema could not be resolved. Options: "
            "A) create schema audit only and stop, "
            "B) ask user to specify actual interval columns, "
            "C) create prediction-only viewer."
        )
    logger.write(f"Prediction rows loaded: {len(prediction_rows)}")
    logger.write(f"Open candidate rows loaded: {len(open_rows)}")
    logger.write(f"Trace rows loaded: {len(trace_rows)}")
    logger.write(f"Split rows loaded: {len(split_rows)}")
    logger.write(f"Actual label rows loaded: {len(actual_rows)}")
    logger.write(f"Actual schema: {actual_schema}")
    if errors:
        logger.write("Major problem detected; no viewer generated")
        report = {
            "task_name": TASK_NAME,
            "version": VERSION,
            "project_root": str(PROJECT_ROOT),
            "generated_at": generated_at,
            "input_files": input_stats_before,
            "warnings": warnings,
            "errors": errors,
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1

    videos_by_id = load_split_rows(split_rows, warnings, errors)
    test_rows_in_split = [
        to_int(row.get("video_id"))
        for row in split_rows
        if normalize_split(row.get("split")) == "test" or to_int(row.get("video_id")) in EXCLUDED_TEST_VIDEO_IDS
    ]
    logger.write(f"Train/validation videos loaded for manifest: {len(videos_by_id)}")
    logger.write(f"Test rows observed in split file and excluded: {sorted(set(x for x in test_rows_in_split if x is not None))}")

    logger.step("Build review manifest")
    actual_count = attach_actual_intervals(videos_by_id, actual_rows, actual_schema, warnings)
    predicted_count = attach_predictions(videos_by_id, prediction_rows, warnings, errors)
    open_count = attach_open_candidates(videos_by_id, open_rows, warnings, errors)
    attach_trace_summary(videos_by_id, trace_rows)
    manifest = build_manifest(videos_by_id, generated_at)
    manifest_ids = {int(video["video_id"]) for video in manifest["videos"]}
    if manifest_ids & EXCLUDED_TEST_VIDEO_IDS:
        errors.append(f"Test video IDs included in manifest: {sorted(manifest_ids & EXCLUDED_TEST_VIDEO_IDS)}")
    if any(video.get("split") not in INCLUDED_SPLITS for video in manifest["videos"]):
        errors.append("Non-train/validation split found in manifest")
    manifest["split_policy"]["test_included"] = bool(manifest_ids & EXCLUDED_TEST_VIDEO_IDS)
    logger.write(f"Actual intervals attached: {actual_count}")
    logger.write(f"Closed predictions attached: {predicted_count}")
    logger.write(f"Open candidates attached: {open_count}")
    if errors:
        logger.write("Manifest safety check failed; stopping before writing viewer files")
        return 1

    logger.step("Generate static viewer files")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_index_html(OUTPUT_DIR / "index.html")
    write_app_js(OUTPUT_DIR / "app.js")
    write_style_css(OUTPUT_DIR / "style.css")
    write_readme(OUTPUT_DIR / "README_review_viewer.md")
    write_manifest(MANIFEST_PATH, manifest)
    logger.write(f"Static viewer written to {OUTPUT_DIR}")

    logger.step("Implement review server")
    server_path = PROJECT_ROOT / "scripts/review/serve_state_machine_ad_review_viewer_v1_1.py"
    if not server_path.exists():
        errors.append(f"Server script missing: {server_path}")
    else:
        logger.write(f"Server script exists: {server_path}")

    logger.step("Generate report and summary")
    old_after = snapshot_tree(OLD_PROJECT_ROOT, REPORT_DIR / f"old_project_snapshot_after_review_viewer_v1_1_{timestamp}.tsv")
    old_project_modified = old_before.get("sha256") != old_after.get("sha256")
    input_stats_after = {
        name: file_stats(path) if path.exists() else {"path": str(path), "exists": False}
        for name, path in INPUT_FILES.items()
    }
    input_files_modified = input_stats_before != input_stats_after
    videos = manifest["videos"]
    playable_video_count = sum(1 for video in videos if video.get("playable"))
    missing_video_count = sum(1 for video in videos if not video.get("playable"))
    report: dict[str, Any] = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "generated_at": generated_at,
        "input_files": input_stats_before,
        "input_files_after": input_stats_after,
        "output_files": [str(path) for path in EXPECTED_OUTPUT_FILES],
        "backup": backup,
        "old_project_snapshot_before": old_before,
        "old_project_snapshot_after": old_after,
        "video_count_by_split": video_counts_by_split(videos),
        "actual_interval_count_by_split": compute_counts_by_split(videos, "actual_intervals"),
        "predicted_interval_count_by_split": compute_counts_by_split(videos, "predicted_intervals"),
        "open_interval_count_by_split": compute_counts_by_split(videos, "open_interval_candidates"),
        "playable_video_count": playable_video_count,
        "missing_video_count": missing_video_count,
        "missing_videos": [
            {"video_id": video["video_id"], "split": video["split"], "video_path": video["video_path"]}
            for video in videos
            if not video.get("playable")
        ],
        "test_included": manifest["split_policy"]["test_included"],
        "test_video_ids_excluded": sorted(EXCLUDED_TEST_VIDEO_IDS),
        "old_project_modified": old_project_modified,
        "input_files_modified": input_files_modified,
        "latest_for_chatgpt_forbidden_files_found": [],
        "warnings": warnings,
        "errors": errors,
        "actual_label_schema": actual_schema,
        "detector_config_loaded": detector_config is not None,
        "detector_report_loaded": detector_report is not None,
        "no_detector_run": True,
        "no_feature_extraction": True,
        "no_threshold_tuning": True,
        "no_rule_tuning": True,
        "no_prediction_generation": True,
        "no_video_reencoding": True,
        "no_thumbnail_generation": True,
        "no_frame_extraction": True,
        "actual_label_usage": "audit_ui_only_not_detector_decision",
        "validation_usage": "audit_review_only",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preliminary_validations = {"sub_agent_validations_pending": True}
    write_summary(SUMMARY_PATH, report, manifest, preliminary_validations)
    logger.write(f"Report written: {REPORT_PATH}")
    logger.write(f"Summary written: {SUMMARY_PATH}")

    logger.step("Sub Agent validations")
    copy_latest_bundle(report, logger)
    validations = static_validation(report, manifest)
    report["sub_agent_validations"] = validations
    report["latest_for_chatgpt_forbidden_files_found"] = validations["latest_forbidden_files_found"]
    if validations["latest_forbidden_files_found"]:
        report["errors"].append("Latest bundle forbidden-file scan failed")
    for name, validation in validations.items():
        if name.startswith("sub_agent_"):
            logger.write(f"{name}: {validation['status']}")

    logger.step("Update latest bundle")
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(SUMMARY_PATH, report, manifest, validations)
    copy_latest_bundle(report, logger)
    validations = static_validation(report, manifest)
    report["sub_agent_validations"] = validations
    report["latest_for_chatgpt_forbidden_files_found"] = validations["latest_forbidden_files_found"]
    final_status = "SUCCESS"
    if report["errors"] or any(v.get("status") == "FAIL" for k, v in validations.items() if k.startswith("sub_agent_")):
        final_status = "FAILURE"
    elif report["warnings"] or missing_video_count:
        final_status = "CONDITIONAL_SUCCESS"
    report["status"] = final_status
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(SUMMARY_PATH, report, manifest, validations)
    copy_latest_bundle(report, logger)

    logger.step("Print final human-readable summary")
    summary_lines = [
        f"작업 상태: {final_status}",
        f"viewer output path: {OUTPUT_DIR}",
        f"server script path: {PROJECT_ROOT / 'scripts/review/serve_state_machine_ad_review_viewer_v1_1.py'}",
        "실행 명령: python scripts/review/serve_state_machine_ad_review_viewer_v1_1.py --host 127.0.0.1 --port 8000",
        "로컬 브라우저 접속 방법: VS Code Ports panel에서 8000 forward 후 http://localhost:8000",
        f"video count by split: {report['video_count_by_split']}",
        f"actual/pred/open interval count by split: actual={report['actual_interval_count_by_split']}, predicted={report['predicted_interval_count_by_split']}, open={report['open_interval_count_by_split']}",
        f"playable/missing video count: playable={playable_video_count}, missing={missing_video_count}",
        "Sub Agent validation 결과: "
        + ", ".join(
            f"{name}={validation['status']}"
            for name, validation in validations.items()
            if name.startswith("sub_agent_")
        ),
        f"old_project_modified: {report['old_project_modified']}",
        f"input_files_modified: {report['input_files_modified']}",
        f"test_included: {report['test_included']}",
        f"latest bundle path: {LATEST_DIR}",
        f"warnings: {report['warnings'] if report['warnings'] else 'None'}",
        f"errors: {report['errors'] if report['errors'] else 'None'}",
        "다음 단계: validation open interval부터 사람이 검토. Test는 아직 실행하지 말 것.",
    ]
    logger.write("\n".join(summary_lines))
    return 0 if final_status != "FAILURE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
