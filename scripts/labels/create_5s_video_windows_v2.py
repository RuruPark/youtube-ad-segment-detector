#!/usr/bin/env python3
"""Create 5-second video windows and label/context segment metadata v2.

Public repository note:
    raw videos and private label files are intentionally not included. Set
    ``YASD_PROJECT_ROOT`` when running against a private local dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("YASD_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
OLD_PROJECT_ROOT = Path(os.environ.get("YASD_OLD_PROJECT_ROOT", PROJECT_ROOT / "_old_project_not_included")).resolve()
VIDEO_INPUT_DIR = PROJECT_ROOT / "data/raw/videos"
CLEAN_LABEL_PATH = PROJECT_ROOT / "data/labels/clean_ad_labels_v0.csv"
WINDOW_DIR = PROJECT_ROOT / "data/windows"
VIDEO_METADATA_DIR = PROJECT_ROOT / "data/video_metadata"
SEGMENT_DIR = PROJECT_ROOT / "data/segments"
REPORT_DIR = PROJECT_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"

SCRIPT_PATH = PROJECT_ROOT / "scripts/labels/create_5s_video_windows_v2.py"
VIDEO_MANIFEST_PATH = VIDEO_METADATA_DIR / "video_manifest_v2.csv"
WINDOW_GRID_PATH = WINDOW_DIR / "window_grid_5s_v2.csv"
WINDOW_LABELS_PATH = WINDOW_DIR / "window_labels_5s_v2.csv"
AD_INTERVAL_SEGMENTS_PATH = SEGMENT_DIR / "ad_interval_segments_v2.csv"
AD_CONTEXT_SEGMENTS_PATH = SEGMENT_DIR / "ad_context_segments_v2.csv"
COMBINED_SEGMENTS_PATH = SEGMENT_DIR / "label_interval_context_segments_v2.csv"
REPORT_PATH = REPORT_DIR / "create_5s_video_windows_v2_report.json"
SUMMARY_PATH = REPORT_DIR / "create_5s_video_windows_v2_summary.md"
LOG_PATH = LOG_DIR / "create_5s_video_windows_v2_run_log.txt"
README_PATH = PROJECT_ROOT / "README.md"
LATEST_README_PATH = LATEST_DIR / "README_latest_files.md"
CHECK_ENV_SCRIPT = PROJECT_ROOT / "scripts/utils/check_cv_environment.py"

WINDOW_SIZE_SEC = 5.0
STRIDE_SEC = 5.0
CONTEXT_WINDOW_SEC = 10.0
INITIAL_ESTIMATE = "약 5~15분. mp4 파일 수, 파일 크기, SHA256 계산 여부에 따라 달라질 수 있음."

REQUIRED_LABEL_COLUMNS = [
    "video_id",
    "video_title",
    "ad_interval_id",
    "ad_start_sec",
    "ad_end_sec",
    "is_abrupt_transition_ad",
    "label_valid",
]

VIDEO_MANIFEST_COLUMNS = [
    "video_id",
    "video_title",
    "video_filename",
    "video_path",
    "file_stem",
    "normalized_file_stem",
    "matched_label_video_title",
    "normalized_matched_label_video_title",
    "title_match_status",
    "title_match_method",
    "referenced_v_pattern",
    "file_size_bytes",
    "file_sha256",
    "duration_sec",
    "fps",
    "frame_count",
    "width",
    "height",
    "metadata_source",
    "metadata_valid",
    "metadata_warning",
]

WINDOW_COLUMNS = [
    "video_id",
    "video_title",
    "video_filename",
    "video_path",
    "file_stem",
    "title_match_status",
    "window_id",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "window_duration_sec",
    "window_size_sec",
    "stride_sec",
    "is_partial_window",
    "video_duration_sec",
    "fps",
    "frame_count",
    "width",
    "height",
]

OVERLAP_COLUMNS = [
    "overlap_any_ad",
    "overlap_ad_sec",
    "overlap_ad_ratio",
    "matched_ad_interval_ids",
    "max_overlap_ad_interval_id",
    "max_overlap_ad_sec",
    "max_overlap_ad_ratio",
    "matched_abrupt_scope_values",
    "window_label_scope",
]

SEGMENT_COLUMNS = [
    "segment_id",
    "segment_type",
    "boundary_role",
    "video_id",
    "video_title",
    "video_filename",
    "video_path",
    "file_stem",
    "title_match_status",
    "ad_interval_id",
    "source_label_row_index",
    "segment_start_sec",
    "segment_end_sec",
    "segment_duration_sec",
    "context_window_sec",
    "ad_start_sec",
    "ad_end_sec",
    "is_abrupt_transition_ad",
    "label_valid",
    "video_duration_sec",
    "clipping_applied",
    "segment_valid",
    "segment_warning",
]

RUN_LOG: list[str] = []


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_LOG.append(f"[{timestamp}] {message}")
    print(f"[create_5s_video_windows_v2] {message}")


def project_path(relative_path: str | Path) -> Path:
    path = PROJECT_ROOT / relative_path
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise RuntimeError(f"Refusing to write outside project root: {resolved}")
    return path


def safe_output_path(path: Path) -> Path:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        raise RuntimeError(f"Refusing to write outside project root: {resolved}")
    return path


def ensure_dirs() -> None:
    for path in [
        WINDOW_DIR,
        VIDEO_METADATA_DIR,
        SEGMENT_DIR,
        REPORT_DIR,
        LOG_DIR,
        PROJECT_ROOT / "scripts/labels",
        LATEST_DIR,
    ]:
        safe_output_path(path).mkdir(parents=True, exist_ok=True)


def normalize_title(text: Any) -> str:
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.strip().casefold()
    value = re.sub(r"[_\\-]+", " ", value)
    value = re.sub(r"[^0-9a-z가-힣\s]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def safe_id_part(text: str) -> str:
    value = normalize_title(text)
    value = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", value)
    value = value.strip("_")
    return value[:80] or "video"


def clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def to_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None


def fmt_num(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (int, float)):
        rounded = round(float(value), 6)
        if abs(rounded - round(rounded)) < 1e-9:
            return int(round(rounded))
        return rounded
    return value


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_mp4_files() -> list[Path]:
    if not VIDEO_INPUT_DIR.exists():
        return []
    return sorted(
        [
            path
            for path in VIDEO_INPUT_DIR.rglob("*")
            if path.is_file() and path.suffix in {".mp4", ".MP4"}
        ],
        key=lambda item: str(item),
    )


def estimated_after_scan(mp4_count: int) -> str:
    if mp4_count == 0:
        return "약 1~3분. mp4 파일이 없어 빈 산출물과 report만 생성."
    if mp4_count <= 20:
        return "약 5~15분. mp4 metadata와 SHA256 계산을 포함."
    return "약 15분 이상. mp4 파일 수가 많아 SHA256 및 metadata 추출 시간이 늘어날 수 있음."


def verify_cv_environment() -> tuple[bool, str, list[str]]:
    warnings: list[str] = []
    executable = sys.executable
    in_cv = "/envs/cv/" in executable or executable.endswith("/envs/cv/bin/python") or executable.endswith("/envs/cv/bin/python3.10")
    if CHECK_ENV_SCRIPT.exists():
        cmd = ["conda", "run", "-n", "cv", "python", str(CHECK_ENV_SCRIPT)]
        log("Command: " + " ".join(cmd))
        try:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
            log("cv check stdout: " + result.stdout.strip().replace("\n", " | "))
            if result.stderr.strip():
                log("cv check stderr: " + result.stderr.strip().replace("\n", " | "))
            if result.returncode != 0:
                warnings.append("check_cv_environment.py returned non-zero")
                return False, executable, warnings
        except Exception as exc:
            warnings.append(f"check_cv_environment.py failed to run: {exc!r}")
            return False, executable, warnings
    if not in_cv:
        warnings.append("current python executable is not inside cv env")
    return in_cv, executable, warnings


def load_labels() -> tuple[pd.DataFrame, bool, list[str], list[str]]:
    if not CLEAN_LABEL_PATH.exists():
        return pd.DataFrame(), False, REQUIRED_LABEL_COLUMNS.copy(), ["clean label file missing"]
    df = pd.read_csv(CLEAN_LABEL_PATH)
    missing = [column for column in REQUIRED_LABEL_COLUMNS if column not in df.columns]
    warnings = []
    if missing:
        warnings.append(f"missing required label columns: {missing}")
    return df, True, missing, warnings


def label_valid_mask(df: pd.DataFrame) -> pd.Series:
    if "label_valid" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df["label_valid"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def build_title_indexes(label_df: pd.DataFrame, missing_cols: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str], list[dict[str, Any]], list[str]]:
    if missing_cols:
        return {}, {}, {}, {}, set(), [], []
    title_groups: dict[str, dict[str, Any]] = {}
    normalized_groups: dict[str, dict[str, Any]] = {}
    exact_ambiguous: dict[str, dict[str, Any]] = {}
    normalized_ambiguous: dict[str, dict[str, Any]] = {}
    ambiguous_titles: list[dict[str, Any]] = []
    valid_df = label_df[label_valid_mask(label_df)].copy()

    for title, group in valid_df.groupby("video_title", dropna=False):
        title_text = clean_value(title)
        video_ids = sorted({clean_value(value) for value in group["video_id"].tolist() if clean_value(value)})
        info = {
            "video_title": title_text,
            "normalized_title": normalize_title(title_text),
            "video_ids": video_ids,
            "row_indexes": [int(idx) for idx in group.index.tolist()],
            "rows": group,
        }
        title_groups[title_text] = info

    for info in title_groups.values():
        normalized_groups.setdefault(info["normalized_title"], {"infos": []})["infos"].append(info)

    normalized_index: dict[str, dict[str, Any]] = {}
    for normalized, payload in normalized_groups.items():
        infos = payload["infos"]
        video_ids = sorted({vid for info in infos for vid in info["video_ids"]})
        if len(video_ids) > 1:
            ambiguous_payload = {
                "normalized_title": normalized,
                "candidate_titles": [info["video_title"] for info in infos],
                "candidate_video_ids": video_ids,
            }
            normalized_ambiguous[normalized] = ambiguous_payload
            ambiguous_titles.append(ambiguous_payload)
        else:
            normalized_index[normalized] = {
                "video_title": infos[0]["video_title"],
                "normalized_title": normalized,
                "video_ids": video_ids,
                "row_indexes": [idx for info in infos for idx in info["row_indexes"]],
            }

    exact_index: dict[str, dict[str, Any]] = {}
    for title, info in title_groups.items():
        if len(info["video_ids"]) == 1:
            exact_index[title] = {
                "video_title": info["video_title"],
                "normalized_title": info["normalized_title"],
                "video_ids": info["video_ids"],
                "row_indexes": info["row_indexes"],
            }
        else:
            ambiguous_payload = {
                "video_title": info["video_title"],
                "candidate_video_ids": info["video_ids"],
                "row_indexes": info["row_indexes"],
            }
            exact_ambiguous[title] = ambiguous_payload
            ambiguous_titles.append(ambiguous_payload)

    label_titles = {clean_value(value) for value in valid_df["video_title"].tolist() if clean_value(value)}
    label_video_ids = sorted({clean_value(value) for value in valid_df["video_id"].tolist() if clean_value(value)})
    return exact_index, normalized_index, exact_ambiguous, normalized_ambiguous, label_titles, ambiguous_titles, label_video_ids


def match_video_title(file_stem: str, exact_index: dict[str, dict[str, Any]], normalized_index: dict[str, dict[str, Any]], exact_ambiguous: dict[str, dict[str, Any]], normalized_ambiguous: dict[str, dict[str, Any]], label_available: bool, missing_cols: list[str]) -> dict[str, Any]:
    normalized_stem = normalize_title(file_stem)
    v_pattern = ";".join(re.findall(r"\bV\d{3,}\b", file_stem, flags=re.IGNORECASE))
    if not label_available:
        return {
            "video_id": f"UNMATCHED_{safe_id_part(file_stem)}",
            "video_title": file_stem,
            "matched_label_video_title": "",
            "normalized_matched_label_video_title": "",
            "title_match_status": "label_file_unavailable",
            "title_match_method": "none",
            "referenced_v_pattern": v_pattern,
        }
    if missing_cols:
        return {
            "video_id": f"UNMATCHED_{safe_id_part(file_stem)}",
            "video_title": file_stem,
            "matched_label_video_title": "",
            "normalized_matched_label_video_title": "",
            "title_match_status": "required_label_column_missing",
            "title_match_method": "none",
            "referenced_v_pattern": v_pattern,
        }
    if file_stem in exact_index:
        info = exact_index[file_stem]
        return {
            "video_id": info["video_ids"][0],
            "video_title": info["video_title"],
            "matched_label_video_title": info["video_title"],
            "normalized_matched_label_video_title": info["normalized_title"],
            "title_match_status": "exact",
            "title_match_method": "file_stem_exact_video_title",
            "referenced_v_pattern": v_pattern,
        }
    if file_stem in exact_ambiguous:
        return {
            "video_id": f"UNMATCHED_{safe_id_part(file_stem)}",
            "video_title": file_stem,
            "matched_label_video_title": file_stem,
            "normalized_matched_label_video_title": normalized_stem,
            "title_match_status": "ambiguous",
            "title_match_method": "exact_multiple_video_ids",
            "referenced_v_pattern": v_pattern,
        }
    if normalized_stem in normalized_index:
        info = normalized_index[normalized_stem]
        return {
            "video_id": info["video_ids"][0],
            "video_title": info["video_title"],
            "matched_label_video_title": info["video_title"],
            "normalized_matched_label_video_title": info["normalized_title"],
            "title_match_status": "normalized",
            "title_match_method": "normalized_file_stem_video_title",
            "referenced_v_pattern": v_pattern,
        }
    if normalized_stem in normalized_ambiguous:
        return {
            "video_id": f"UNMATCHED_{safe_id_part(file_stem)}",
            "video_title": file_stem,
            "matched_label_video_title": "",
            "normalized_matched_label_video_title": normalized_stem,
            "title_match_status": "ambiguous",
            "title_match_method": "normalized_multiple_video_ids",
            "referenced_v_pattern": v_pattern,
        }
    return {
        "video_id": f"UNMATCHED_{safe_id_part(file_stem)}",
        "video_title": file_stem,
        "matched_label_video_title": "",
        "normalized_matched_label_video_title": "",
        "title_match_status": "unmatched",
        "title_match_method": "none",
        "referenced_v_pattern": v_pattern,
    }


def ffprobe_duration(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return to_float(result.stdout.strip())


def extract_video_metadata(path: Path) -> dict[str, Any]:
    warning: list[str] = []
    cap = cv2.VideoCapture(str(path))
    source = "opencv"
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    cap.release()
    duration = frame_count / fps if fps > 0 and frame_count > 0 else None
    valid = duration is not None and duration > 0
    if not valid:
        warning.append("opencv_duration_unavailable")
        fallback_duration = ffprobe_duration(path)
        if fallback_duration is not None and fallback_duration > 0:
            duration = fallback_duration
            source = "ffprobe_fallback"
            valid = True
        else:
            warning.append("ffprobe_duration_unavailable")
    if fps <= 0:
        warning.append("fps_zero_or_missing")
    if frame_count <= 0:
        warning.append("frame_count_zero_or_missing")
    return {
        "duration_sec": duration,
        "fps": fps if fps > 0 else None,
        "frame_count": frame_count if frame_count > 0 else None,
        "width": int(width) if width > 0 else None,
        "height": int(height) if height > 0 else None,
        "metadata_source": source,
        "metadata_valid": valid,
        "metadata_warning": "; ".join(sorted(set(warning))),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    safe_output_path(path)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {column: fmt_num(row.get(column, "")) for column in columns}
            writer.writerow(cleaned)


def create_video_manifest(mp4_files: list[Path], exact_index: dict[str, dict[str, Any]], normalized_index: dict[str, dict[str, Any]], exact_ambiguous: dict[str, dict[str, Any]], normalized_ambiguous: dict[str, dict[str, Any]], label_available: bool, missing_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, path in enumerate(mp4_files, start=1):
        log(f"Processing video metadata {idx}/{len(mp4_files)}: {path.name}")
        file_stem = path.stem
        match = match_video_title(file_stem, exact_index, normalized_index, exact_ambiguous, normalized_ambiguous, label_available, missing_cols)
        metadata = extract_video_metadata(path)
        sha = sha256_file(path)
        row = {
            **match,
            "video_filename": path.name,
            "video_path": str(path),
            "file_stem": file_stem,
            "normalized_file_stem": normalize_title(file_stem),
            "file_size_bytes": path.stat().st_size,
            "file_sha256": sha,
            **metadata,
        }
        rows.append(row)
    return rows


def create_windows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video in manifest_rows:
        if not video.get("metadata_valid") or not video.get("duration_sec"):
            continue
        duration = float(video["duration_sec"])
        start = 0.0
        window_index = 1
        while start < duration:
            end = min(start + WINDOW_SIZE_SEC, duration)
            if end <= start:
                break
            rows.append(
                {
                    "video_id": video["video_id"],
                    "video_title": video["video_title"],
                    "video_filename": video["video_filename"],
                    "video_path": video["video_path"],
                    "file_stem": video["file_stem"],
                    "title_match_status": video["title_match_status"],
                    "window_id": f"{video['video_id']}_W{window_index:06d}",
                    "window_index": window_index,
                    "window_start_sec": start,
                    "window_end_sec": end,
                    "window_duration_sec": end - start,
                    "window_size_sec": WINDOW_SIZE_SEC,
                    "stride_sec": STRIDE_SEC,
                    "is_partial_window": bool_text((end - start) < WINDOW_SIZE_SEC - 1e-6),
                    "video_duration_sec": duration,
                    "fps": video.get("fps"),
                    "frame_count": video.get("frame_count"),
                    "width": video.get("width"),
                    "height": video.get("height"),
                }
            )
            start += STRIDE_SEC
            window_index += 1
    return rows


def valid_label_rows(label_df: pd.DataFrame, missing_cols: list[str]) -> pd.DataFrame:
    if missing_cols or label_df.empty:
        return pd.DataFrame(columns=label_df.columns)
    df = label_df.copy()
    df["_source_label_row_index"] = df.index + 2
    df["_video_id_text"] = df["video_id"].map(clean_value)
    df["_ad_start_sec_num"] = df["ad_start_sec"].map(to_float)
    df["_ad_end_sec_num"] = df["ad_end_sec"].map(to_float)
    df["_label_valid_bool"] = label_valid_mask(df)
    mask = (
        df["_label_valid_bool"]
        & df["_video_id_text"].ne("")
        & df["_ad_start_sec_num"].notna()
        & df["_ad_end_sec_num"].notna()
        & (df["_ad_start_sec_num"] < df["_ad_end_sec_num"])
    )
    return df[mask].copy()


def matched_label_keys(manifest_rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (clean_value(row["video_id"]), clean_value(row["matched_label_video_title"]))
        for row in manifest_rows
        if row.get("title_match_status") in {"exact", "normalized"} and clean_value(row.get("matched_label_video_title"))
    }


def create_window_labels(window_rows: list[dict[str, Any]], label_df: pd.DataFrame, missing_cols: list[str], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_rows = valid_label_rows(label_df, missing_cols)
    allowed_keys = matched_label_keys(manifest_rows)
    labels_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for _, row in label_rows.iterrows():
        key = (clean_value(row["video_id"]), clean_value(row["video_title"]))
        if key not in allowed_keys:
            continue
        labels_by_key[key].append(
            {
                "ad_interval_id": clean_value(row["ad_interval_id"]),
                "ad_start_sec": float(row["_ad_start_sec_num"]),
                "ad_end_sec": float(row["_ad_end_sec_num"]),
                "is_abrupt_transition_ad": clean_value(row["is_abrupt_transition_ad"]) or "not_reviewed",
            }
        )

    output: list[dict[str, Any]] = []
    for window in window_rows:
        win_start = float(window["window_start_sec"])
        win_end = float(window["window_end_sec"])
        win_duration = max(float(window["window_duration_sec"]), 1e-9)
        overlaps: list[dict[str, Any]] = []
        label_key = (clean_value(window["video_id"]), clean_value(window["video_title"]))
        for label in labels_by_key.get(label_key, []):
            overlap = max(0.0, min(win_end, label["ad_end_sec"]) - max(win_start, label["ad_start_sec"]))
            if overlap > 0:
                overlaps.append({**label, "overlap_sec": overlap, "overlap_ratio": overlap / win_duration})

        overlap_sum = min(sum(item["overlap_sec"] for item in overlaps), win_duration)
        max_overlap = max(overlaps, key=lambda item: item["overlap_sec"], default=None)
        scopes = sorted({item["is_abrupt_transition_ad"] for item in overlaps})
        qualifying_scopes = sorted({item["is_abrupt_transition_ad"] for item in overlaps if item["overlap_ratio"] >= 0.5})
        if not overlaps:
            scope_label = "outside_ad"
        elif len(qualifying_scopes) == 1:
            value = qualifying_scopes[0]
            if value == "yes":
                scope_label = "abrupt_ad"
            elif value == "no":
                scope_label = "out_of_scope_ad"
            elif value == "unclear":
                scope_label = "uncertain_ad"
            elif value == "not_reviewed":
                scope_label = "not_reviewed_ad"
            else:
                scope_label = "mixed_or_boundary"
        else:
            scope_label = "mixed_or_boundary"

        output.append(
            {
                **window,
                "overlap_any_ad": bool_text(bool(overlaps)),
                "overlap_ad_sec": overlap_sum,
                "overlap_ad_ratio": overlap_sum / win_duration,
                "matched_ad_interval_ids": ";".join(item["ad_interval_id"] for item in overlaps),
                "max_overlap_ad_interval_id": max_overlap["ad_interval_id"] if max_overlap else "",
                "max_overlap_ad_sec": max_overlap["overlap_sec"] if max_overlap else 0,
                "max_overlap_ad_ratio": max_overlap["overlap_ratio"] if max_overlap else 0,
                "matched_abrupt_scope_values": ";".join(scopes),
                "window_label_scope": scope_label,
            }
        )
    return output


def manifest_by_label_key(manifest_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (clean_value(row["video_id"]), clean_value(row["matched_label_video_title"])): row
        for row in manifest_rows
        if row.get("title_match_status") in {"exact", "normalized"}
    }


def create_ad_interval_segments(label_df: pd.DataFrame, missing_cols: list[str], manifest_rows: list[dict[str, Any]], report_warnings: list[Any]) -> list[dict[str, Any]]:
    valid_labels = valid_label_rows(label_df, missing_cols)
    videos = manifest_by_label_key(manifest_rows)
    rows: list[dict[str, Any]] = []
    for _, label in valid_labels.iterrows():
        video_id = clean_value(label["video_id"])
        video = videos.get((video_id, clean_value(label["video_title"])))
        if video is None:
            continue
        start = float(label["_ad_start_sec_num"])
        end = float(label["_ad_end_sec_num"])
        original_start, original_end = start, end
        duration = video.get("duration_sec")
        clipping = False
        warnings: list[str] = []
        if duration is not None:
            start = max(0.0, start)
            end = min(float(duration), end)
            clipping = start != original_start or end != original_end
            if clipping:
                warnings.append("segment_clipped_to_video_duration")
        segment_valid = start < end
        if not segment_valid:
            warnings.append("segment_start_not_less_than_end")
        if segment_valid:
            rows.append(
                {
                    "segment_id": f"{video_id}_{clean_value(label['ad_interval_id'])}_AD",
                    "segment_type": "ad_interval",
                    "boundary_role": "",
                    "video_id": video_id,
                    "video_title": video["video_title"],
                    "video_filename": video["video_filename"],
                    "video_path": video["video_path"],
                    "file_stem": video["file_stem"],
                    "title_match_status": video["title_match_status"],
                    "ad_interval_id": clean_value(label["ad_interval_id"]),
                    "source_label_row_index": int(label["_source_label_row_index"]),
                    "segment_start_sec": start,
                    "segment_end_sec": end,
                    "segment_duration_sec": end - start,
                    "context_window_sec": "",
                    "ad_start_sec": original_start,
                    "ad_end_sec": original_end,
                    "is_abrupt_transition_ad": clean_value(label["is_abrupt_transition_ad"]) or "not_reviewed",
                    "label_valid": bool_text(True),
                    "video_duration_sec": duration,
                    "clipping_applied": bool_text(clipping),
                    "segment_valid": bool_text(segment_valid),
                    "segment_warning": "; ".join(warnings),
                }
            )
    return rows


def create_context_segments(label_df: pd.DataFrame, missing_cols: list[str], manifest_rows: list[dict[str, Any]], report_warnings: list[Any]) -> tuple[list[dict[str, Any]], int, int]:
    valid_labels = valid_label_rows(label_df, missing_cols)
    videos = manifest_by_label_key(manifest_rows)
    rows: list[dict[str, Any]] = []
    empty_count = 0
    duration_unknown_not_clipped = 0

    for _, label in valid_labels.iterrows():
        video_id = clean_value(label["video_id"])
        video = videos.get((video_id, clean_value(label["video_title"])))
        if video is None:
            continue
        ad_start = float(label["_ad_start_sec_num"])
        ad_end = float(label["_ad_end_sec_num"])
        duration = video.get("duration_sec")
        contexts = [
            ("PRE10", "pre_ad_start_10s", "before_ad_start", max(0.0, ad_start - CONTEXT_WINDOW_SEC), ad_start),
            ("POST10", "post_ad_end_10s", "after_ad_end", ad_end, ad_end + CONTEXT_WINDOW_SEC),
        ]
        for suffix, segment_type, boundary_role, start, end in contexts:
            clipping = False
            warnings: list[str] = []
            if segment_type == "post_ad_end_10s":
                if duration is not None:
                    original_end = end
                    end = min(float(duration), end)
                    clipping = end != original_end
                    if clipping:
                        warnings.append("context_clipped_to_video_duration")
                else:
                    duration_unknown_not_clipped += 1
                    warnings.append("duration_unknown_context_not_clipped")
            if start >= end:
                empty_count += 1
                report_warnings.append(
                    {
                        "ad_interval_id": clean_value(label["ad_interval_id"]),
                        "segment_type": segment_type,
                        "reason": "segment_start_sec_gte_segment_end_sec",
                    }
                )
                continue
            rows.append(
                {
                    "segment_id": f"{video_id}_{clean_value(label['ad_interval_id'])}_{suffix}",
                    "segment_type": segment_type,
                    "boundary_role": boundary_role,
                    "video_id": video_id,
                    "video_title": video["video_title"],
                    "video_filename": video["video_filename"],
                    "video_path": video["video_path"],
                    "file_stem": video["file_stem"],
                    "title_match_status": video["title_match_status"],
                    "ad_interval_id": clean_value(label["ad_interval_id"]),
                    "source_label_row_index": int(label["_source_label_row_index"]),
                    "segment_start_sec": start,
                    "segment_end_sec": end,
                    "segment_duration_sec": end - start,
                    "context_window_sec": CONTEXT_WINDOW_SEC,
                    "ad_start_sec": ad_start,
                    "ad_end_sec": ad_end,
                    "is_abrupt_transition_ad": clean_value(label["is_abrupt_transition_ad"]) or "not_reviewed",
                    "label_valid": bool_text(True),
                    "video_duration_sec": duration,
                    "clipping_applied": bool_text(clipping),
                    "segment_valid": bool_text(True),
                    "segment_warning": "; ".join(warnings),
                }
            )
    return rows, empty_count, duration_unknown_not_clipped


def update_readme() -> None:
    section = """## 5-second Window Grid and Label Interval Segments v2

mp4 파일 기준으로 5초 window grid를 생성했다. mp4 파일명은 `video_id`가 아니라 `clean_ad_labels_v0.csv`의 `video_title` 기준으로 매핑했고, 매칭 성공 시 라벨 파일의 `video_id`를 사용했다.

생성 파일:

- `data/video_metadata/video_manifest_v2.csv`
- `data/windows/window_grid_5s_v2.csv`
- `data/windows/window_labels_5s_v2.csv`
- `data/segments/ad_interval_segments_v2.csv`
- `data/segments/ad_context_segments_v2.csv`
- `data/segments/label_interval_context_segments_v2.csv`

5초 window는 최종 광고 판정 단위가 아니라 Scene/OCR/audio evidence 수집 단위이다. `ad_interval_segments_v2.csv`는 라벨링된 광고 구간 자체를 보존하는 interval-level metadata이고, `ad_context_segments_v2.csv`는 광고 시작 전 10초, 광고 종료 후 10초 구간을 따로 분석하기 위한 metadata이다.

최종 광고 판단은 이후 rule-based temporal aggregation에서 interval 단위로 수행할 수 있다. 이번 작업에서는 영상 클립, 프레임, OCR, Scene feature, Audio feature, ResNet, Gemma를 생성하거나 실행하지 않았다.
"""
    text = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else "# youtube_ad_segment_detector\n"
    marker = "## 5-second Window Grid and Label Interval Segments v2"
    if marker not in text:
        text = text.rstrip() + "\n\n" + section
    else:
        before, _, after = text.partition(marker)
        next_idx = after.find("\n## ")
        if next_idx == -1:
            text = before.rstrip() + "\n\n" + section
        else:
            text = before.rstrip() + "\n\n" + section + after[next_idx:]
    README_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")


def clear_latest_dir(report: dict[str, Any]) -> bool:
    expected = PROJECT_ROOT / "outputs/latest_for_chatgpt"
    if LATEST_DIR.resolve() != expected.resolve():
        report.setdefault("errors", []).append(f"latest_for_chatgpt path mismatch: {LATEST_DIR.resolve()}")
        return False
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for child in LATEST_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return True


def refresh_latest(report: dict[str, Any]) -> list[str]:
    latest_files = [
        (README_PATH, "README.md", "프로젝트 README와 v2 window/segment 설명"),
        (VIDEO_MANIFEST_PATH, "video_manifest_v2.csv", "mp4 파일별 title 매칭 및 duration metadata"),
        (WINDOW_GRID_PATH, "window_grid_5s_v2.csv", "5초 단위 전체 영상 timeline grid"),
        (WINDOW_LABELS_PATH, "window_labels_5s_v2.csv", "5초 window와 광고 label overlap 매핑"),
        (AD_INTERVAL_SEGMENTS_PATH, "ad_interval_segments_v2.csv", "라벨링된 광고 interval 자체 metadata"),
        (AD_CONTEXT_SEGMENTS_PATH, "ad_context_segments_v2.csv", "광고 시작 전 10초/종료 후 10초 context metadata"),
        (COMBINED_SEGMENTS_PATH, "label_interval_context_segments_v2.csv", "ad/pre/post segment 통합 metadata"),
        (REPORT_PATH, "create_5s_video_windows_v2_report.json", "v2 상세 report"),
        (SUMMARY_PATH, "create_5s_video_windows_v2_summary.md", "v2 요약"),
        (LOG_PATH, "create_5s_video_windows_v2_run_log.txt", "v2 실행 log"),
        (SCRIPT_PATH, "create_5s_video_windows_v2.py", "v2 재현 실행 스크립트"),
    ]
    copied: list[str] = []
    descriptions: list[str] = []
    for src, name, purpose in latest_files:
        if src.exists():
            dst = LATEST_DIR / name
            shutil.copy2(src, dst)
            copied.append(str(dst))
            descriptions.append(f"- `{name}`: {purpose}.")

    LATEST_README_PATH.write_text(
        "# latest_for_chatgpt files\n\n"
        "latest_for_chatgpt는 최신 작업 핵심 파일만 모아둔 복사본 경로이다. 원본 파일은 프로젝트 내부 원래 경로에 존재한다.\n\n"
        "이번 작업명: create_5s_video_windows_v2\n\n"
        "이번 작업에서 추가된 핵심 변경:\n\n"
        "- mp4 파일명과 `video_title` 기반 매핑\n"
        "- 5초 window v2 생성\n"
        "- 광고 interval segment 생성\n"
        "- 광고 시작 전 10초 / 종료 후 10초 context segment 생성\n\n"
        "복사된 파일 목록과 목적:\n\n"
        + "\n".join(descriptions)
        + "\n\nmp4 영상 파일, 원본 xlsx, 프레임, 모델, checkpoint, cache 파일은 복사하지 않는다.\n",
        encoding="utf-8",
    )
    copied.append(str(LATEST_README_PATH))
    return copied


def distribution(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(clean_value(value) for value in values).items()))


def make_summary(report: dict[str, Any]) -> str:
    warnings = report.get("warnings", [])
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- 주요 warning 없음"
    return f"""# create_5s_video_windows_v2 summary

## 작업 시간

- 시작 전 예상 작업 시간: {report['estimated_work_time_initial']}
- 스캔 후 예상 작업 시간: {report['estimated_work_time_after_scan']}

## 결과 요약

- mp4 파일 수: {report['mp4_file_count']}
- metadata 정상 추출 영상 수: {report['valid_video_metadata_count']}
- metadata 실패/unknown 영상 수: {report['invalid_video_metadata_count']}
- video_title exact match 수: {report['title_match_exact_count']}
- video_title normalized match 수: {report['title_match_normalized_count']}
- unmatched 영상 수: {report['title_match_unmatched_count']}
- ambiguous 영상 수: {report['title_match_ambiguous_count']}
- 전체 5초 window 수: {report['total_window_count']}
- partial window 수: {report['partial_window_count']}
- label overlap 매핑: {'가능' if report['label_file_found'] and report['required_label_columns_found'] else '불가'}

## 분포

- window_label_scope: `{report['window_label_scope_distribution']}`
- segment_type: `{report['segment_type_distribution']}`
- boundary_role: `{report['boundary_role_distribution']}`

## Segment 수

- 광고 interval segment: {report['ad_interval_segment_count']}
- 광고 시작 전 10초 segment: {report['pre_ad_start_10s_segment_count']}
- 광고 종료 후 10초 segment: {report['post_ad_end_10s_segment_count']}
- 통합 segment: {report['combined_segment_count']}

## 주의할 warning

{warning_text}

## latest_for_chatgpt

- 내부 정리 수행: {report['latest_for_chatgpt_cleared']}
- 최신 핵심 파일 복사 완료: {report['latest_for_chatgpt_updated']}

## 서브 에이전트 검토

{report.get('sub_agent_review_summary', '서브 에이전트 검토 예정')}

## 다음 작업

1. `window_labels_5s_v2.csv`와 segment CSV를 검토한다.
2. `is_abrupt_transition_ad` 검토 완료 후 window scope 분포를 다시 생성한다.
3. 이후 Scene/OCR/audio feature를 window/segment 기준으로 별도 생성한다.
"""


def main() -> None:
    print(f"예상 작업 시간: {INITIAL_ESTIMATE}")
    ensure_dirs()
    log("Started create_5s_video_windows_v2.")
    log("No video clipping, frame extraction, OCR, Scene, Audio, ResNet, Gemma, threshold tuning, or evaluation will be run.")

    errors: list[Any] = []
    warnings: list[Any] = []
    cv_ok, python_executable, cv_warnings = verify_cv_environment()
    warnings.extend(cv_warnings)
    if not cv_ok:
        errors.append("cv_environment_check_failed")

    label_df, label_file_found, missing_label_cols, label_warnings = load_labels()
    warnings.extend(label_warnings)
    required_label_columns_found = label_file_found and not missing_label_cols
    log(f"Loaded labels: found={label_file_found}, rows={len(label_df)}, missing_columns={missing_label_cols}")

    exact_index, normalized_index, exact_ambiguous, normalized_ambiguous, label_titles, ambiguous_title_matches, label_video_ids = build_title_indexes(label_df, missing_label_cols)
    mp4_files = find_mp4_files()
    after_scan_estimate = estimated_after_scan(len(mp4_files))
    log(f"Found {len(mp4_files)} mp4 files. Estimated after scan: {after_scan_estimate}")

    manifest_rows = create_video_manifest(mp4_files, exact_index, normalized_index, exact_ambiguous, normalized_ambiguous, label_file_found, missing_label_cols)
    window_rows = create_windows(manifest_rows)
    window_label_rows = create_window_labels(window_rows, label_df, missing_label_cols, manifest_rows)
    interval_segments = create_ad_interval_segments(label_df, missing_label_cols, manifest_rows, warnings)
    context_segments, empty_context_count, duration_unknown_context_not_clipped_count = create_context_segments(label_df, missing_label_cols, manifest_rows, warnings)
    combined_segments = []
    for row in interval_segments:
        combined = {**row}
        combined["boundary_role"] = ""
        combined_segments.append(combined)
    combined_segments.extend(context_segments)

    write_csv(VIDEO_MANIFEST_PATH, manifest_rows, VIDEO_MANIFEST_COLUMNS)
    write_csv(WINDOW_GRID_PATH, window_rows, WINDOW_COLUMNS)
    write_csv(WINDOW_LABELS_PATH, window_label_rows, WINDOW_COLUMNS + OVERLAP_COLUMNS)
    write_csv(AD_INTERVAL_SEGMENTS_PATH, interval_segments, SEGMENT_COLUMNS)
    write_csv(AD_CONTEXT_SEGMENTS_PATH, context_segments, SEGMENT_COLUMNS)
    write_csv(COMBINED_SEGMENTS_PATH, combined_segments, SEGMENT_COLUMNS)
    log("Wrote v2 CSV outputs.")

    update_readme()
    log("Updated README v2 section.")

    status_counts = Counter(row["title_match_status"] for row in manifest_rows)
    matched_titles = {row["matched_label_video_title"] for row in manifest_rows if row["title_match_status"] in {"exact", "normalized"}}
    videos_without_label = [row["video_filename"] for row in manifest_rows if row["title_match_status"] in {"unmatched", "ambiguous", "label_file_unavailable", "required_label_column_missing"}]
    label_titles_without_video = sorted(title for title in label_titles if title not in matched_titles)
    unmatched_or_ambiguous_video_id = [row["video_id"] for row in manifest_rows if row["title_match_status"] in {"unmatched", "ambiguous"}]
    labels_without_video = label_titles_without_video

    metadata_invalid = [row["video_filename"] for row in manifest_rows if not row.get("metadata_valid") or not row.get("duration_sec")]
    metadata_warnings = [
        {"video_filename": row["video_filename"], "metadata_warning": row["metadata_warning"]}
        for row in manifest_rows
        if row.get("metadata_warning")
    ]

    label_valid_count = int(label_valid_mask(label_df).sum()) if label_file_found and "label_valid" in label_df.columns else 0
    label_invalid_count = int(len(label_df) - label_valid_count) if label_file_found else 0
    duplicate_video_id_count = sum(1 for _, count in Counter(row["video_id"] for row in manifest_rows).items() if count > 1)
    duplicate_window_id_count = sum(1 for _, count in Counter(row["window_id"] for row in window_rows).items() if count > 1)
    duplicate_ad_interval_id_count = sum(1 for _, count in Counter(clean_value(value) for value in label_df["ad_interval_id"].tolist()).items() if count > 1) if label_file_found and "ad_interval_id" in label_df.columns else 0
    duplicate_segment_id_count = sum(1 for _, count in Counter(row["segment_id"] for row in combined_segments).items() if count > 1)

    scopes = [clean_value(value) or "not_reviewed" for value in label_df["is_abrupt_transition_ad"].tolist()] if label_file_found and "is_abrupt_transition_ad" in label_df.columns else []
    abrupt_scope_distribution = distribution(scopes)
    abrupt_scope_not_fully_reviewed = bool(scopes and any(scope == "not_reviewed" for scope in scopes))
    if abrupt_scope_not_fully_reviewed:
        warnings.append("abrupt_scope_not_fully_reviewed: is_abrupt_transition_ad includes not_reviewed")
    if videos_without_label:
        warnings.append({"videos_without_label_title_match": videos_without_label})
    if label_titles_without_video:
        warnings.append({"label_titles_without_video_file_count": len(label_titles_without_video)})
    if duplicate_window_id_count:
        errors.append("duplicate_window_id_detected")
    if duplicate_segment_id_count:
        errors.append("duplicate_segment_id_detected")

    clipped_segment_count = sum(1 for row in combined_segments if row.get("clipping_applied") == "true")
    report: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "video_input_dir": str(VIDEO_INPUT_DIR),
        "clean_label_path": str(CLEAN_LABEL_PATH),
        "estimated_work_time_initial": INITIAL_ESTIMATE,
        "estimated_work_time_after_scan": after_scan_estimate,
        "cv_environment_checked": cv_ok,
        "python_executable": python_executable,
        "old_project_modified": False,
        "mp4_file_count": len(mp4_files),
        "valid_video_metadata_count": sum(1 for row in manifest_rows if row.get("metadata_valid")),
        "invalid_video_metadata_count": sum(1 for row in manifest_rows if not row.get("metadata_valid")),
        "metadata_invalid_or_duration_unknown": metadata_invalid,
        "total_video_duration_sec": sum(float(row["duration_sec"]) for row in manifest_rows if row.get("duration_sec")),
        "total_window_count": len(window_rows),
        "partial_window_count": sum(1 for row in window_rows if row.get("is_partial_window") == "true"),
        "video_manifest_path": str(VIDEO_MANIFEST_PATH),
        "window_grid_path": str(WINDOW_GRID_PATH),
        "window_labels_path": str(WINDOW_LABELS_PATH),
        "ad_interval_segments_path": str(AD_INTERVAL_SEGMENTS_PATH),
        "ad_context_segments_path": str(AD_CONTEXT_SEGMENTS_PATH),
        "label_interval_context_segments_path": str(COMBINED_SEGMENTS_PATH),
        "label_file_found": label_file_found,
        "required_label_columns_found": required_label_columns_found,
        "missing_label_columns": missing_label_cols,
        "label_row_count": len(label_df) if label_file_found else 0,
        "label_valid_count": label_valid_count,
        "label_invalid_count": label_invalid_count,
        "label_video_id_count": len(set(clean_value(value) for value in label_df["video_id"].tolist() if clean_value(value))) if label_file_found and "video_id" in label_df.columns else 0,
        "label_video_title_count": len(label_titles),
        "title_match_exact_count": status_counts.get("exact", 0),
        "title_match_normalized_count": status_counts.get("normalized", 0),
        "title_match_unmatched_count": status_counts.get("unmatched", 0),
        "title_match_ambiguous_count": status_counts.get("ambiguous", 0),
        "videos_without_label_title_match": videos_without_label,
        "label_titles_without_video_file": label_titles_without_video,
        "ambiguous_video_title_matches": ambiguous_title_matches,
        "unmatched_or_ambiguous_video_id": unmatched_or_ambiguous_video_id,
        "videos_without_label": videos_without_label,
        "labels_without_video": labels_without_video,
        "duplicate_video_id_count": duplicate_video_id_count,
        "duplicate_window_id_count": duplicate_window_id_count,
        "duplicate_ad_interval_id_count": duplicate_ad_interval_id_count,
        "duplicate_segment_id_count": duplicate_segment_id_count,
        "window_label_scope_distribution": distribution([row["window_label_scope"] for row in window_label_rows]),
        "abrupt_scope_distribution": abrupt_scope_distribution,
        "abrupt_scope_not_fully_reviewed": abrupt_scope_not_fully_reviewed,
        "ad_interval_segment_count": len(interval_segments),
        "pre_ad_start_10s_segment_count": sum(1 for row in context_segments if row["segment_type"] == "pre_ad_start_10s"),
        "post_ad_end_10s_segment_count": sum(1 for row in context_segments if row["segment_type"] == "post_ad_end_10s"),
        "combined_segment_count": len(combined_segments),
        "segment_type_distribution": distribution([row["segment_type"] for row in combined_segments]),
        "boundary_role_distribution": distribution([row["boundary_role"] for row in combined_segments if row.get("boundary_role")]),
        "empty_context_segment_count": empty_context_count,
        "clipped_segment_count": clipped_segment_count,
        "duration_unknown_context_not_clipped_count": duration_unknown_context_not_clipped_count,
        "metadata_warnings": metadata_warnings,
        "generated_files": [
            str(VIDEO_MANIFEST_PATH),
            str(WINDOW_GRID_PATH),
            str(WINDOW_LABELS_PATH),
            str(AD_INTERVAL_SEGMENTS_PATH),
            str(AD_CONTEXT_SEGMENTS_PATH),
            str(COMBINED_SEGMENTS_PATH),
            str(REPORT_PATH),
            str(SUMMARY_PATH),
            str(LOG_PATH),
            str(SCRIPT_PATH),
        ],
        "latest_for_chatgpt_cleared": False,
        "latest_for_chatgpt_updated": False,
        "latest_for_chatgpt_files": [],
        "warnings": warnings,
        "errors": errors,
    }

    clear_ok = clear_latest_dir(report)
    report["latest_for_chatgpt_cleared"] = clear_ok
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(make_summary(report), encoding="utf-8")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    copied = refresh_latest(report) if clear_ok else []
    report["latest_for_chatgpt_updated"] = bool(copied)
    report["latest_for_chatgpt_files"] = copied
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(make_summary(report), encoding="utf-8")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    if clear_ok:
        shutil.copy2(REPORT_PATH, LATEST_DIR / "create_5s_video_windows_v2_report.json")
        shutil.copy2(SUMMARY_PATH, LATEST_DIR / "create_5s_video_windows_v2_summary.md")
        shutil.copy2(LOG_PATH, LATEST_DIR / "create_5s_video_windows_v2_run_log.txt")
    log("Finished create_5s_video_windows_v2.")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    if clear_ok:
        shutil.copy2(LOG_PATH, LATEST_DIR / "create_5s_video_windows_v2_run_log.txt")
    print(json.dumps({
        "mp4_file_count": report["mp4_file_count"],
        "valid_video_metadata_count": report["valid_video_metadata_count"],
        "total_window_count": report["total_window_count"],
        "partial_window_count": report["partial_window_count"],
        "title_match_exact_count": report["title_match_exact_count"],
        "title_match_normalized_count": report["title_match_normalized_count"],
        "title_match_unmatched_count": report["title_match_unmatched_count"],
        "title_match_ambiguous_count": report["title_match_ambiguous_count"],
        "ad_interval_segment_count": report["ad_interval_segment_count"],
        "combined_segment_count": report["combined_segment_count"],
        "errors": report["errors"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
