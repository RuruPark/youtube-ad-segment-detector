#!/usr/bin/env python3
"""Print and package v1.3 train-only detector results for human review.

This script is intentionally read-only with respect to detector/config/input
artifacts. It only creates review tables, markdown/text reports, a run log, and
a small latest bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TRAIN_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
VALIDATION_IDS = [3, 7, 18]
TEST_IDS = [4, 16, 17]
SPLIT_SEED = "20240524"

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
FORBIDDEN_DIR_PARTS = {
    "cache",
    "frames",
    "frame_images",
    "raw_video",
    "video_proxy",
    "model_cache",
    "tmp",
    "__pycache__",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(str(value))
        if math.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return max(0.0, num / den)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def mmss(sec: Any) -> str:
    value = max(0, int(round(to_float(sec))))
    minutes = value // 60
    seconds = value % 60
    return f"{minutes:02d}:{seconds:02d}"


def interval_label(start: float, end: float) -> str:
    return f"{mmss(start)}-{mmss(end)}"


def union_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    valid = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not valid:
        return []
    merged = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def duration(intervals: list[tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in intervals)


def intersect_duration(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    total = 0.0
    for s1, e1 in union_intervals(a):
        for s2, e2 in union_intervals(b):
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def best_actual_match(
    interval: tuple[float, float], actuals: list[tuple[str, float, float]]
) -> tuple[str, float, float]:
    start, end = interval
    best_id = ""
    best_overlap = 0.0
    best_actual_duration = 0.0
    for actual_id, actual_start, actual_end in actuals:
        overlap = max(0.0, min(end, actual_end) - max(start, actual_start))
        if overlap > best_overlap:
            best_id = actual_id
            best_overlap = overlap
            best_actual_duration = actual_end - actual_start
    return best_id, best_overlap, best_actual_duration


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_stats(paths: list[Path]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            stats[str(path)] = {"exists": False}
            continue
        stat = path.stat()
        stats[str(path)] = {
            "exists": True,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": hash_file(path),
        }
    return stats


def snapshot_project(path: Path, output_path: Path) -> dict[str, Any]:
    if not path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("exists\tfalse\n", encoding="utf-8")
        return {"exists": False}
    file_count = 0
    total_size = 0
    max_mtime = 0.0
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [
            d for d in dirnames if d not in {".git", "__pycache__", "cache", "frames"}
        ]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                stat = p.stat()
            except OSError:
                continue
            file_count += 1
            total_size += stat.st_size
            max_mtime = max(max_mtime, stat.st_mtime)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "exists\ttrue\n"
        f"path\t{path}\n"
        f"file_count\t{file_count}\n"
        f"total_size\t{total_size}\n"
        f"max_mtime\t{max_mtime}\n",
        encoding="utf-8",
    )
    return {
        "exists": True,
        "path": str(path),
        "file_count": file_count,
        "total_size": total_size,
        "max_mtime": max_mtime,
    }


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


class Runner:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = root / "logs/v1_3_train_results_print_run_log.txt"
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.report: dict[str, Any] = {}
        self.old_snapshot_before: dict[str, Any] = {}
        self.old_snapshot_after: dict[str, Any] = {}
        self.input_stats_before: dict[str, Any] = {}
        self.input_stats_after: dict[str, Any] = {}

        self.paths = {
            "split": root / "data/splits/video_split_v2_4.csv",
            "actual": root / "data/segments/ad_interval_segments_v2_4.csv",
            "pred": root / "data/predictions/state_machine_interval_predictions_v1_3_train.csv",
            "trace": root / "data/predictions/state_machine_anchor_trace_v1_3_train.csv",
            "open": root
            / "data/predictions/state_machine_open_interval_candidates_v1_3_train.csv",
            "unresolved": root
            / "data/predictions/state_machine_unresolved_long_open_candidates_v1_3_train.csv",
            "disclosure": root
            / "data/predictions/state_machine_disclosure_notice_review_candidates_v1_3_train.csv",
            "ocr_cap": root
            / "data/predictions/state_machine_ocr_only_continuity_cap_events_v1_3_train.csv",
            "weak": root
            / "data/predictions/state_machine_weak_continuity_recovery_events_v1_3_train.csv",
            "audit": root / "data/predictions/state_machine_detector_train_audit_v1_3.csv",
            "comparison": root
            / "data/predictions/state_machine_v1_2_vs_v1_3_train_comparison.csv",
            "analysis_video": root
            / "data/analysis/train_only_detector_error_video_summary_v1_3.csv",
            "analysis_overlap": root
            / "data/analysis/train_only_detector_error_interval_overlap_v1_3.csv",
            "analysis_open": root
            / "data/analysis/train_only_detector_error_open_interval_summary_v1_3.csv",
            "analysis_reason": root
            / "data/analysis/train_only_detector_error_trace_reason_summary_v1_3.csv",
            "analysis_worst": root
            / "data/analysis/train_only_detector_error_worst_cases_v1_3.csv",
            "analysis_candidates": root
            / "data/analysis/train_only_detector_error_rule_issue_candidates_v1_3.csv",
        }
        self.output_paths = {
            "script": root / "scripts/analysis/print_v1_3_train_results_for_review.py",
            "readable_md": root / "reports/analysis/v1_3_train_results_readable_printout.md",
            "console_txt": root / "reports/analysis/v1_3_train_results_console_output.txt",
            "diagnosis_md": root / "reports/analysis/v1_3_train_failure_diagnosis_for_v1_4.md",
            "video_table": root / "data/analysis/v1_3_train_video_result_table_for_review.csv",
            "interval_table": root
            / "data/analysis/v1_3_train_interval_result_table_for_review.csv",
            "worst_table": root / "data/analysis/v1_3_train_worst_video_detail_table.csv",
            "reason_table": root / "data/analysis/v1_3_train_reason_count_table.csv",
            "candidate_table": root / "data/analysis/v1_3_train_v1_4_rule_candidate_table.csv",
            "report_json": root / "reports/analysis/v1_3_train_results_print_report.json",
            "log": self.log_path,
        }
        self.latest_dir = root / "outputs/latest_for_chatgpt_v1_3_train_results_printout"

    def log(self, message: str) -> None:
        print(message)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")

    def backup_targets(self) -> Path:
        backup_dir = self.root / "backups" / f"v1_3_train_results_printout_{self.timestamp}"
        manifest: list[dict[str, str]] = []
        targets = [p for p in self.output_paths.values() if p.exists()]
        if self.latest_dir.exists():
            targets.append(self.latest_dir)
        for src in targets:
            dst = backup_dir / src.relative_to(self.root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            manifest.append({"source": str(src), "backup": str(dst)})
        backup_dir.mkdir(parents=True, exist_ok=True)
        (backup_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return backup_dir

    def step01(self) -> None:
        if self.log_path.exists():
            # log는 아래에서 백업한 뒤 깨끗한 실행 기록으로 다시 쓴다.
            pass
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        self.log("[STEP 01] Safety snapshot and backup")
        if self.root.resolve() != Path("."):
            raise RuntimeError(f"Unexpected project root: {self.root}")
        self.backup_dir = self.backup_targets()
        self.log(f"[STEP 01] backup_dir={self.backup_dir}")
        old_project = Path("./_old_project_not_included")
        before_path = (
            self.root
            / "reports/analysis/old_project_snapshot_before_v1_3_train_results_printout.tsv"
        )
        self.old_snapshot_before = snapshot_project(old_project, before_path)
        input_paths = list(self.paths.values())
        self.input_stats_before = file_stats(input_paths)
        missing = [str(p) for p in input_paths if not p.exists()]
        if missing:
            raise RuntimeError("Missing required input files: " + ", ".join(missing))
        self.log("[STEP 01] input file stats/hash captured")
        self.log("[STEP 01] forbidden suffix/directory rules defined")

    def load_inputs(self) -> None:
        self.log("[STEP 02] Load v1.3 train outputs")
        self.split_rows = read_csv(self.paths["split"])
        self.actual_rows = read_csv(self.paths["actual"])
        self.pred_rows = read_csv(self.paths["pred"])
        self.trace_rows = read_csv(self.paths["trace"])
        self.open_rows = read_csv(self.paths["open"])
        self.unresolved_rows = read_csv(self.paths["unresolved"])
        self.disclosure_rows = read_csv(self.paths["disclosure"])
        self.ocr_cap_rows = read_csv(self.paths["ocr_cap"])
        self.weak_rows = read_csv(self.paths["weak"])
        self.audit_rows = read_csv(self.paths["audit"])
        self.comparison_rows = read_csv(self.paths["comparison"])
        self.analysis_video_rows = read_csv(self.paths["analysis_video"])
        self.analysis_overlap_rows = read_csv(self.paths["analysis_overlap"])
        self.analysis_open_rows = read_csv(self.paths["analysis_open"])
        self.analysis_reason_rows = read_csv(self.paths["analysis_reason"])
        self.analysis_worst_rows = read_csv(self.paths["analysis_worst"])
        self.analysis_candidate_rows = read_csv(self.paths["analysis_candidates"])

        seeds = {row.get("split_seed", "") for row in self.split_rows}
        train_ids = sorted(to_int(row.get("video_id")) for row in self.split_rows if row.get("split") == "train")
        val_ids = sorted(to_int(row.get("video_id")) for row in self.split_rows if row.get("split") == "validation")
        test_ids = sorted(to_int(row.get("video_id")) for row in self.split_rows if row.get("split") == "test")
        if str(SPLIT_SEED) not in seeds:
            raise RuntimeError(f"split_seed mismatch: {sorted(seeds)}")
        if train_ids != TRAIN_IDS or val_ids != VALIDATION_IDS or test_ids != TEST_IDS:
            raise RuntimeError(
                f"split ids mismatch: train={train_ids}, validation={val_ids}, test={test_ids}"
            )

        self.video_meta: dict[int, dict[str, Any]] = {}
        for row in self.split_rows:
            video_id = to_int(row.get("video_id"))
            if video_id is None:
                continue
            self.video_meta[video_id] = {
                "split": row.get("split", ""),
                "video_name": row.get("video_name", ""),
                "video_duration_sec": to_float(row.get("video_duration_sec")),
            }
        self._assert_train_only("pred", self.pred_rows)
        self._assert_train_only("trace", self.trace_rows)
        self._assert_train_only("open", self.open_rows)
        self._assert_train_only("unresolved", self.unresolved_rows)
        self._assert_train_only("disclosure", self.disclosure_rows)
        self._assert_train_only("ocr_cap", self.ocr_cap_rows)
        self._assert_train_only("weak", self.weak_rows)
        self._assert_train_only("audit", self.audit_rows)
        self._assert_train_only("comparison", self.comparison_rows)
        self.log("[STEP 02] split validated and all loaded v1.3 outputs are train-only")

    def _assert_train_only(self, name: str, rows: list[dict[str, str]]) -> None:
        bad: list[str] = []
        for row in rows:
            split = row.get("split")
            video_id = to_int(row.get("video_id"))
            if split and split != "train":
                bad.append(f"{video_id}:{split}")
            if video_id is not None and video_id not in TRAIN_IDS:
                bad.append(f"{video_id}:non_train_video")
        if bad:
            raise RuntimeError(f"{name} contains non-train rows: {bad[:10]}")

    def actual_intervals_by_video(self) -> dict[int, list[tuple[str, float, float]]]:
        by_video: dict[int, list[tuple[str, float, float]]] = defaultdict(list)
        for idx, row in enumerate(self.actual_rows, start=1):
            video_id = to_int(row.get("video_id"))
            if video_id not in TRAIN_IDS:
                continue
            segment_type = (row.get("segment_type") or row.get("label") or "").lower()
            is_ad_value = (row.get("is_ad") or "").lower()
            is_ad = (
                "ad_interval" in segment_type
                or segment_type in {"ad", "ad_full", "advertisement"}
                or is_ad_value in {"1", "true", "yes"}
            )
            if any(token in segment_type for token in ["non_ad", "random_non_ad", "post_ad", "pre_ad"]):
                is_ad = False
            if not is_ad:
                continue
            start = to_float(row.get("ad_start_sec") or row.get("segment_start_sec"))
            end = to_float(row.get("ad_end_sec") or row.get("segment_end_sec"))
            if end <= start:
                self.warnings.append(f"Invalid actual interval ignored: video={video_id}, row={idx}")
                continue
            title = row.get("video_title")
            if title and not self.video_meta.get(video_id, {}).get("video_name"):
                self.video_meta.setdefault(video_id, {})["video_name"] = title
            interval_id = row.get("ad_interval_id") or row.get("\ufeffsegment_id") or f"actual_{idx}"
            by_video[video_id].append((str(interval_id), start, end))
        return by_video

    def prediction_intervals_by_video(self) -> dict[int, list[dict[str, Any]]]:
        by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.pred_rows:
            video_id = to_int(row.get("video_id"))
            if video_id not in TRAIN_IDS:
                continue
            start = to_float(row.get("ad_start_sec"))
            end = to_float(row.get("ad_end_sec"))
            if end <= start:
                self.warnings.append(
                    f"Invalid closed prediction ignored: video={video_id}, id={row.get('prediction_id')}"
                )
                continue
            by_video[video_id].append(
                {
                    "id": row.get("prediction_id", ""),
                    "start": start,
                    "end": end,
                    "reason": "; ".join(
                        x
                        for x in [row.get("start_reason", ""), row.get("end_reason", "")]
                        if x
                    ),
                    "start_reason": row.get("start_reason", ""),
                    "end_reason": row.get("end_reason", ""),
                }
            )
        return by_video

    def open_intervals_by_video(self) -> dict[int, list[dict[str, Any]]]:
        by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.open_rows:
            video_id = to_int(row.get("video_id"))
            if video_id not in TRAIN_IDS:
                continue
            start = to_float(row.get("ad_start_sec") or row.get("start_sec"))
            end = to_float(row.get("last_anchor_sec") or row.get("end_proxy_sec"))
            if end <= start:
                self.warnings.append(
                    f"Invalid open interval ignored: video={video_id}, id={row.get('open_candidate_id')}"
                )
                continue
            by_video[video_id].append(
                {
                    "id": row.get("open_candidate_id", ""),
                    "start": start,
                    "end": end,
                    "reason": row.get("open_reason", ""),
                    "start_reason": row.get("open_reason", ""),
                }
            )
        return by_video

    def unresolved_intervals_by_video(self) -> dict[int, list[dict[str, Any]]]:
        by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.unresolved_rows:
            video_id = to_int(row.get("video_id"))
            if video_id not in TRAIN_IDS:
                continue
            start = to_float(row.get("start_sec"))
            end = to_float(row.get("last_anchor_sec"))
            if end <= start:
                self.warnings.append(
                    "Invalid unresolved interval ignored: "
                    f"video={video_id}, id={row.get('unresolved_candidate_id')}"
                )
                continue
            by_video[video_id].append(
                {
                    "id": row.get("unresolved_candidate_id", ""),
                    "start": start,
                    "end": end,
                    "reason": row.get("unresolved_reason", ""),
                    "start_reason": row.get("start_reason", ""),
                }
            )
        return by_video

    def build_video_table(self) -> None:
        self.log("[STEP 03] Build train video result table")
        self.actual_by_video = self.actual_intervals_by_video()
        self.pred_by_video = self.prediction_intervals_by_video()
        self.open_by_video = self.open_intervals_by_video()
        self.unresolved_by_video = self.unresolved_intervals_by_video()

        self.video_rows: list[dict[str, Any]] = []
        for video_id in TRAIN_IDS:
            meta = self.video_meta[video_id]
            video_duration = to_float(meta.get("video_duration_sec"))
            actual = [(s, e) for _, s, e in self.actual_by_video.get(video_id, [])]
            closed = [(x["start"], x["end"]) for x in self.pred_by_video.get(video_id, [])]
            open_intervals = [(x["start"], x["end"]) for x in self.open_by_video.get(video_id, [])]
            unresolved = [(x["start"], x["end"]) for x in self.unresolved_by_video.get(video_id, [])]
            actual_u = union_intervals(actual)
            closed_u = union_intervals(closed)
            open_u = union_intervals(open_intervals)
            unresolved_u = union_intervals(unresolved)
            all_detector_u = union_intervals(closed + open_intervals + unresolved)
            actual_total = duration(actual_u)
            closed_total = duration(closed_u)
            open_total = duration(open_u)
            unresolved_total = duration(unresolved_u)
            detector_total = duration(all_detector_u)
            overlap_all = intersect_duration(actual_u, all_detector_u)
            false_positive = max(0.0, detector_total - overlap_all)
            missed = max(0.0, actual_total - overlap_all)
            actual_cov = ratio(actual_total, video_duration)
            closed_cov = ratio(closed_total, video_duration)
            open_cov = ratio(open_total, video_duration)
            unresolved_cov = ratio(unresolved_total, video_duration)
            combined_cov = ratio(detector_total, video_duration)
            missed_ratio = ratio(missed, actual_total)
            if actual_total == 0 and detector_total > 0:
                severity = "no_actual_but_predicted"
            elif (
                combined_cov >= 0.5
                or (actual_cov > 0 and combined_cov / actual_cov >= 3)
                or open_cov + unresolved_cov >= 0.3
            ):
                severity = "severe_over_detection"
            elif combined_cov >= 0.25 or (actual_cov > 0 and combined_cov / actual_cov >= 2):
                severity = "moderate_over_detection"
            elif missed_ratio >= 0.5:
                severity = "missed_actual"
            else:
                severity = "needs_manual_review"
            if severity in {"severe_over_detection", "no_actual_but_predicted"}:
                priority = "high"
            elif severity in {"moderate_over_detection", "missed_actual"}:
                priority = "medium"
            else:
                priority = "low"
            main_issue = self.main_issue(closed_cov, open_cov, unresolved_cov, missed_ratio, false_positive, video_duration)
            row = {
                "video_id": video_id,
                "video_title": meta.get("video_name", ""),
                "video_duration_sec": round(video_duration, 3),
                "actual_interval_count": len(actual),
                "actual_total_duration_sec": round(actual_total, 3),
                "actual_coverage_ratio": round(actual_cov, 6),
                "closed_prediction_count": len(closed),
                "closed_total_duration_sec": round(closed_total, 3),
                "closed_coverage_ratio": round(closed_cov, 6),
                "open_interval_count": len(open_intervals),
                "open_total_duration_proxy_sec": round(open_total, 3),
                "open_coverage_ratio": round(open_cov, 6),
                "unresolved_long_open_count": len(unresolved),
                "unresolved_total_duration_proxy_sec": round(unresolved_total, 3),
                "unresolved_coverage_ratio": round(unresolved_cov, 6),
                "predicted_plus_open_plus_unresolved_coverage_ratio": round(combined_cov, 6),
                "false_positive_duration_sec": round(false_positive, 3),
                "missed_actual_duration_sec": round(missed, 3),
                "missed_actual_ratio": round(missed_ratio, 6),
                "severity_level": severity,
                "review_priority": priority,
                "main_issue": main_issue,
                "viewer_review_hint": self.viewer_hint(video_id, actual, closed, open_intervals, unresolved),
                "v1_4_rule_candidate_hint": self.rule_hint(main_issue),
            }
            self.video_rows.append(row)

        self.video_rows.sort(
            key=lambda r: (
                0 if r["review_priority"] == "high" else 1 if r["review_priority"] == "medium" else 2,
                -to_float(r["predicted_plus_open_plus_unresolved_coverage_ratio"]),
                -to_float(r["false_positive_duration_sec"]),
            )
        )
        write_csv(
            self.output_paths["video_table"],
            self.video_rows,
            [
                "video_id",
                "video_title",
                "video_duration_sec",
                "actual_interval_count",
                "actual_total_duration_sec",
                "actual_coverage_ratio",
                "closed_prediction_count",
                "closed_total_duration_sec",
                "closed_coverage_ratio",
                "open_interval_count",
                "open_total_duration_proxy_sec",
                "open_coverage_ratio",
                "unresolved_long_open_count",
                "unresolved_total_duration_proxy_sec",
                "unresolved_coverage_ratio",
                "predicted_plus_open_plus_unresolved_coverage_ratio",
                "false_positive_duration_sec",
                "missed_actual_duration_sec",
                "missed_actual_ratio",
                "severity_level",
                "review_priority",
                "main_issue",
                "viewer_review_hint",
                "v1_4_rule_candidate_hint",
            ],
        )

    def main_issue(
        self,
        closed_cov: float,
        open_cov: float,
        unresolved_cov: float,
        missed_ratio: float,
        false_positive: float,
        video_duration: float,
    ) -> str:
        if open_cov >= 0.3:
            return "open interval covers too much of the video"
        if unresolved_cov >= 0.2:
            return "unresolved long open candidate still spans a large segment"
        if closed_cov >= 0.35:
            return "closed predictions occupy too much video time"
        if false_positive >= max(60.0, video_duration * 0.2):
            return "large false-positive candidate duration"
        if missed_ratio >= 0.5:
            return "actual ad coverage is missed while candidates drift elsewhere"
        if open_cov + unresolved_cov >= 0.3:
            return "open and unresolved candidates dominate detector output"
        return "manual boundary review needed"

    def rule_hint(self, issue: str) -> str:
        if "open" in issue or "unresolved" in issue:
            return "tighten end/open recovery and maximum open span policy"
        if "closed predictions" in issue or "false-positive" in issue:
            return "tighten start confirmation and in_ad continuity"
        if "missed" in issue:
            return "review stricter gates against missed actual ads before validation"
        return "compare trace around actual boundaries before v1.4"

    def viewer_hint(
        self,
        video_id: int,
        actual: list[tuple[float, float]],
        closed: list[tuple[float, float]],
        open_intervals: list[tuple[float, float]],
        unresolved: list[tuple[float, float]],
    ) -> str:
        focus: list[str] = []
        for label, intervals in [
            ("actual", actual[:1]),
            ("closed", closed[:1]),
            ("open", open_intervals[:1]),
            ("unresolved", unresolved[:1]),
        ]:
            if intervals:
                focus.append(f"{label}@{interval_label(intervals[0][0], intervals[0][1])}")
        if not focus:
            return f"video {video_id}: inspect trace around first in_ad transition"
        return "; ".join(focus)

    def build_interval_table(self) -> None:
        self.log("[STEP 04] Build interval result table")
        rows: list[dict[str, Any]] = []
        for video_id in TRAIN_IDS:
            actual_list = self.actual_by_video.get(video_id, [])
            actual_intervals = [(s, e) for _, s, e in actual_list]
            detector_intervals = [
                (x["start"], x["end"])
                for x in self.pred_by_video.get(video_id, [])
                + self.open_by_video.get(video_id, [])
                + self.unresolved_by_video.get(video_id, [])
            ]
            detector_u = union_intervals(detector_intervals)

            for actual_id, start, end in actual_list:
                overlap = intersect_duration([(start, end)], detector_u)
                missed = max(0.0, end - start - overlap)
                rows.append(
                    self.interval_row(
                        video_id,
                        "actual",
                        actual_id,
                        start,
                        end,
                        overlap,
                        ratio(overlap, end - start),
                        0.0,
                        "missed_actual_candidate" if missed > 0 else "actual_reference",
                        "train_actual_label",
                        f"missed_actual_sec={missed:.1f}",
                    )
                )

            for item in self.pred_by_video.get(video_id, []):
                rows.append(self.detector_interval_row(video_id, "closed_prediction", item, actual_list))
            for item in self.open_by_video.get(video_id, []):
                rows.append(self.detector_interval_row(video_id, "open_candidate", item, actual_list))
            for item in self.unresolved_by_video.get(video_id, []):
                rows.append(self.detector_interval_row(video_id, "unresolved_long_open", item, actual_list))

        self.interval_rows = rows
        write_csv(
            self.output_paths["interval_table"],
            rows,
            [
                "video_id",
                "interval_type",
                "interval_id",
                "start_sec",
                "end_sec",
                "duration_sec",
                "mmss_start",
                "mmss_end",
                "overlap_with_actual_sec",
                "overlap_ratio",
                "false_positive_duration_sec",
                "issue_type",
                "reason",
                "review_note",
            ],
        )

    def interval_row(
        self,
        video_id: int,
        interval_type: str,
        interval_id: str,
        start: float,
        end: float,
        overlap: float,
        overlap_ratio: float,
        fp: float,
        issue_type: str,
        reason: str,
        review_note: str,
    ) -> dict[str, Any]:
        return {
            "video_id": video_id,
            "interval_type": interval_type,
            "interval_id": interval_id,
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(max(0.0, end - start), 3),
            "mmss_start": mmss(start),
            "mmss_end": mmss(end),
            "overlap_with_actual_sec": round(overlap, 3),
            "overlap_ratio": round(overlap_ratio, 6),
            "false_positive_duration_sec": round(fp, 3),
            "issue_type": issue_type,
            "reason": reason,
            "review_note": review_note,
        }

    def detector_interval_row(
        self,
        video_id: int,
        interval_type: str,
        item: dict[str, Any],
        actual_list: list[tuple[str, float, float]],
    ) -> dict[str, Any]:
        start = to_float(item.get("start"))
        end = to_float(item.get("end"))
        dur = max(0.0, end - start)
        best_id, overlap, best_actual_duration = best_actual_match((start, end), actual_list)
        overlap_ratio = ratio(overlap, dur)
        fp = max(0.0, dur - overlap)
        issue_type = "good_overlap"
        review_note = f"best_actual={best_id or 'none'}"
        if interval_type == "open_candidate":
            issue_type = "open_unresolved_candidate"
        elif interval_type == "unresolved_long_open":
            issue_type = "open_unresolved_candidate"
        elif overlap <= 0:
            issue_type = "false_positive_candidate"
        elif overlap_ratio < 0.3:
            issue_type = "overlong_prediction"
        elif best_actual_duration > 0 and dur > best_actual_duration * 2:
            issue_type = "overlong_prediction"
        else:
            actual = next((x for x in actual_list if x[0] == best_id), None)
            if actual:
                _, actual_start, actual_end = actual
                if start < actual_start - 15:
                    issue_type = "early_start"
                elif end > actual_end + 15:
                    issue_type = "late_end"
        return self.interval_row(
            video_id,
            interval_type,
            str(item.get("id", "")),
            start,
            end,
            overlap,
            overlap_ratio,
            fp,
            issue_type,
            str(item.get("reason", "")),
            review_note,
        )

    def build_worst_detail(self) -> None:
        self.log("[STEP 05] Build worst video detail")
        sorted_videos = sorted(
            self.video_rows,
            key=lambda r: (
                0 if r["review_priority"] == "high" else 1,
                -to_float(r["predicted_plus_open_plus_unresolved_coverage_ratio"]),
                -to_float(r["false_positive_duration_sec"]),
            ),
        )
        rows: list[dict[str, Any]] = []
        for rank, row in enumerate(sorted_videos[:10], start=1):
            video_id = to_int(row["video_id"])
            trace_rows = [x for x in self.trace_rows if to_int(x.get("video_id")) == video_id]
            decision_top = Counter(x.get("decision_reason", "") for x in trace_rows if x.get("decision_reason")).most_common(5)
            transition_top = Counter(
                f"{x.get('state_before')}->{x.get('state_after')}" for x in trace_rows
            ).most_common(5)
            rows.append(
                {
                    "rank": rank,
                    "video_id": video_id,
                    "focus_reason": row["main_issue"],
                    "actual_intervals_mmss": self.format_intervals(
                        [(s, e) for _, s, e in self.actual_by_video.get(video_id, [])]
                    ),
                    "closed_predictions_mmss": self.format_items(self.pred_by_video.get(video_id, [])),
                    "open_intervals_mmss": self.format_items(self.open_by_video.get(video_id, [])),
                    "unresolved_intervals_mmss": self.format_items(self.unresolved_by_video.get(video_id, [])),
                    "top_decision_reasons": "; ".join(f"{k}={v}" for k, v in decision_top),
                    "top_trace_events": "; ".join(f"{k}={v}" for k, v in transition_top),
                    "suspected_failure_mode": row["main_issue"],
                    "suggested_v1_4_direction": row["v1_4_rule_candidate_hint"],
                }
            )
        self.worst_rows = rows
        write_csv(
            self.output_paths["worst_table"],
            rows,
            [
                "rank",
                "video_id",
                "focus_reason",
                "actual_intervals_mmss",
                "closed_predictions_mmss",
                "open_intervals_mmss",
                "unresolved_intervals_mmss",
                "top_decision_reasons",
                "top_trace_events",
                "suspected_failure_mode",
                "suggested_v1_4_direction",
            ],
        )

    def format_intervals(self, intervals: list[tuple[float, float]]) -> str:
        if not intervals:
            return "none"
        return "; ".join(interval_label(s, e) for s, e in intervals)

    def format_items(self, items: list[dict[str, Any]]) -> str:
        if not items:
            return "none"
        return "; ".join(interval_label(to_float(x["start"]), to_float(x["end"])) for x in items)

    def build_reason_table(self) -> None:
        self.log("[STEP 06] Build reason count table")
        rows: list[dict[str, Any]] = []

        def add_counter(category: str, counter: Counter[str], total: int) -> None:
            for key, count in counter.most_common():
                rows.append(
                    {
                        "category": category,
                        "key": key,
                        "count": count,
                        "ratio": round(ratio(count, total), 6),
                        "interpretation": self.interpret_reason(category, key),
                        "related_rule_issue": self.related_rule_issue(category, key),
                    }
                )

        start_counter = Counter(x.get("start_reason", "") for x in self.pred_rows if x.get("start_reason"))
        end_counter = Counter(x.get("end_reason", "") for x in self.pred_rows if x.get("end_reason"))
        decision_counter = Counter(x.get("decision_reason", "") for x in self.trace_rows if x.get("decision_reason"))
        transition_counter = Counter(f"{x.get('state_before')}->{x.get('state_after')}" for x in self.trace_rows)
        maintain_counter = Counter(
            x.get("decision_reason", "")
            for x in self.trace_rows
            if x.get("state_before") == "in_ad" and x.get("state_after") == "in_ad"
        )
        timeout_counter = Counter(
            x.get("decision_reason", "")
            for x in self.trace_rows
            if "timeout" in x.get("decision_reason", "")
            or "cancel" in x.get("decision_reason", "")
            or x.get("pending_timeout_trigger") not in {"", "none"}
        )
        event_counter = Counter(
            {
                "disclosure_notice_rejected": len(self.disclosure_rows),
                "ocr_only_continuity_cap_event": len(self.ocr_cap_rows),
                "weak_continuity_recovery_event": len(self.weak_rows),
                "unresolved_long_open_candidate": len(self.unresolved_rows),
                "end_pending_timeout_no_strong_continuity": sum(
                    1
                    for x in self.trace_rows
                    if x.get("decision_reason") == "end_pending_timeout_no_strong_continuity"
                ),
            }
        )

        add_counter("start_reason", start_counter, sum(start_counter.values()))
        add_counter("end_reason", end_counter, sum(end_counter.values()))
        add_counter("decision_reason", decision_counter, sum(decision_counter.values()))
        add_counter("state_transition", transition_counter, sum(transition_counter.values()))
        add_counter("in_ad_maintain", maintain_counter, sum(maintain_counter.values()))
        add_counter("pending_timeout", timeout_counter, sum(timeout_counter.values()))
        add_counter("v1_3_event", event_counter, sum(event_counter.values()))
        self.reason_rows = rows
        write_csv(
            self.output_paths["reason_table"],
            rows,
            [
                "category",
                "key",
                "count",
                "ratio",
                "interpretation",
                "related_rule_issue",
            ],
        )

    def interpret_reason(self, category: str, key: str) -> str:
        text = key.lower()
        if category == "state_transition":
            return "state machine movement frequency"
        if "maintain" in text or "strong_ad_evidence" in text:
            return "ad state may be sustained by repeated continuity evidence"
        if "timeout" in text:
            return "pending timeout policy is active and should be boundary-reviewed"
        if "cancel" in text:
            return "pending candidate was cancelled before becoming a closed interval"
        if "unresolved" in text:
            return "long open guard separated unresolved detector failure candidates"
        if "disclosure" in text:
            return "early disclosure guard is affecting start candidates"
        return "review trace examples before changing rules"

    def related_rule_issue(self, category: str, key: str) -> str:
        text = (category + " " + key).lower()
        if "start" in text or "disclosure" in text:
            return "start_confirmation"
        if "maintain" in text or "strong" in text or "continuity" in text:
            return "in_ad_continuity"
        if "timeout" in text or "end_pending" in text:
            return "end_confirmation"
        if "unresolved" in text or "open" in text:
            return "open_unresolved_policy"
        return "manual_review"

    def build_candidate_table(self) -> None:
        self.log("[STEP 07] Build v1.4 candidate table")
        severe_videos = [r for r in self.video_rows if r["severity_level"] == "severe_over_detection"]
        open_high = [r for r in self.video_rows if to_float(r["open_coverage_ratio"]) + to_float(r["unresolved_coverage_ratio"]) >= 0.3]
        closed_high = [r for r in self.video_rows if to_float(r["closed_coverage_ratio"]) >= 0.35]
        missed_high = [r for r in self.video_rows if to_float(r["missed_actual_ratio"]) >= 0.5]
        top_decision = Counter(x.get("decision_reason", "") for x in self.trace_rows if x.get("decision_reason")).most_common(5)
        maintain_count = sum(
            1
            for x in self.trace_rows
            if x.get("state_before") == "in_ad" and x.get("state_after") == "in_ad"
        )
        rows = [
            {
                "candidate_id": "V14-RC001",
                "priority": "high",
                "suspected_rule_area": "start_confirmation",
                "issue_summary": "start gate still creates too many downstream ad candidates",
                "evidence_from_v1_3_train": (
                    f"severe_over_detection_videos={len(severe_videos)}; "
                    f"closed_prediction_count={len(self.pred_rows)}; "
                    f"disclosure_rejected={len(self.disclosure_rows)}"
                ),
                "affected_video_ids": self.video_ids_text(severe_videos),
                "suggested_direction": "Require product/CTA/audio-rise combination for start confirmation and review OCR-start-only paths.",
                "expected_effect": "Reduce false starts that later become long closed/open candidates.",
                "risk": "Over-tightening can miss true short ads; compare against actual boundary windows on train first.",
                "needs_manual_review": "true",
                "should_implement_now_or_later": "now_candidate",
            },
            {
                "candidate_id": "V14-RC002",
                "priority": "high",
                "suspected_rule_area": "in_ad_continuity",
                "issue_summary": "in_ad continuity remains sticky even after v1.3",
                "evidence_from_v1_3_train": (
                    f"in_ad_to_in_ad_count={maintain_count}; top_decision={top_decision[:3]}"
                ),
                "affected_video_ids": self.video_ids_text(closed_high or severe_videos),
                "suggested_direction": "Cap repeated strong_ad_evidence from broad OCR/audio context and demand fresh product/CTA evidence after a span.",
                "expected_effect": "Shorten overlong closed predictions and reduce ad-state persistence.",
                "risk": "Some real long ads may need a whitelist-like continuation pattern.",
                "needs_manual_review": "true",
                "should_implement_now_or_later": "now_candidate",
            },
            {
                "candidate_id": "V14-RC003",
                "priority": "high",
                "suspected_rule_area": "open_unresolved_policy",
                "issue_summary": "open and unresolved candidates still cover too much video time",
                "evidence_from_v1_3_train": (
                    f"open_interval_count={len(self.open_rows)}; unresolved_count={len(self.unresolved_rows)}; "
                    f"open_or_unresolved_high_videos={len(open_high)}"
                ),
                "affected_video_ids": self.video_ids_text(open_high),
                "suggested_direction": "Move long weak candidates to unresolved earlier and add a non-ad recovery path after no fresh strong evidence.",
                "expected_effect": "Keep failure candidates separate and prevent them from dominating review output.",
                "risk": "Could hide useful start candidates if unresolved guard is too aggressive.",
                "needs_manual_review": "true",
                "should_implement_now_or_later": "now_candidate",
            },
            {
                "candidate_id": "V14-RC004",
                "priority": "medium",
                "suspected_rule_area": "audio_ocr_medium_level",
                "issue_summary": "medium/high OCR and audio context may still be too broad",
                "evidence_from_v1_3_train": f"top decision reasons={top_decision}",
                "affected_video_ids": self.video_ids_text(severe_videos[:8]),
                "suggested_direction": "Audit which feature levels generate strong_ad_evidence and split medium support from strong reset evidence.",
                "expected_effect": "Reduce accidental strong continuity from generic context.",
                "risk": "Requires careful trace sampling to avoid tuning to label artifacts.",
                "needs_manual_review": "true",
                "should_implement_now_or_later": "now_candidate",
            },
            {
                "candidate_id": "V14-RC005",
                "priority": "medium",
                "suspected_rule_area": "boundary_mismatch",
                "issue_summary": "closed/open candidates often do not align tightly with actual boundaries",
                "evidence_from_v1_3_train": (
                    f"missed_actual_high_videos={len(missed_high)}; "
                    f"false_positive_top={self.false_positive_top_text()}"
                ),
                "affected_video_ids": self.video_ids_text(missed_high or severe_videos[:5]),
                "suggested_direction": "Add boundary review guard and examine trace windows around actual starts/ends before changing validation rules.",
                "expected_effect": "Improve boundary plausibility in train diagnostics before validation.",
                "risk": "Actual labels must remain audit-only; no detector decision may use label columns.",
                "needs_manual_review": "true",
                "should_implement_now_or_later": "after_top3",
            },
        ]
        self.candidate_rows = rows
        write_csv(
            self.output_paths["candidate_table"],
            rows,
            [
                "candidate_id",
                "priority",
                "suspected_rule_area",
                "issue_summary",
                "evidence_from_v1_3_train",
                "affected_video_ids",
                "suggested_direction",
                "expected_effect",
                "risk",
                "needs_manual_review",
                "should_implement_now_or_later",
            ],
        )

    def video_ids_text(self, rows: list[dict[str, Any]]) -> str:
        ids = [str(r["video_id"]) for r in rows if r.get("video_id") is not None]
        return ", ".join(ids[:12]) if ids else "none"

    def false_positive_top_text(self) -> str:
        top = sorted(self.video_rows, key=lambda r: -to_float(r["false_positive_duration_sec"]))[:3]
        return "; ".join(f"video {r['video_id']}={r['false_positive_duration_sec']}s" for r in top)

    def write_outputs(self) -> None:
        self.log("[STEP 08] Write readable markdown and console output")
        console = self.build_console_text()
        diagnosis = self.build_diagnosis_text()
        write_text(self.output_paths["console_txt"], console)
        write_text(self.output_paths["readable_md"], self.build_readable_markdown(console))
        write_text(self.output_paths["diagnosis_md"], diagnosis)
        self.console_text = console
        self.diagnosis_text = diagnosis
        self.log("[STEP 08] readable markdown, failure diagnosis, and console text written")

    def build_console_text(self) -> str:
        train_video_count = len(TRAIN_IDS)
        actual_mean = sum(to_float(r["actual_coverage_ratio"]) for r in self.video_rows) / train_video_count
        closed_mean = sum(to_float(r["closed_coverage_ratio"]) for r in self.video_rows) / train_video_count
        open_mean = sum(to_float(r["open_coverage_ratio"]) for r in self.video_rows) / train_video_count
        unresolved_mean = sum(to_float(r["unresolved_coverage_ratio"]) for r in self.video_rows) / train_video_count
        severe_count = sum(1 for r in self.video_rows if r["severity_level"] == "severe_over_detection")
        worst = self.video_rows[:5]
        missed_delta = self.missed_actual_change_note()
        closed_count = sum(to_int(r["closed_prediction_count"], 0) or 0 for r in self.video_rows)
        open_count = sum(to_int(r["open_interval_count"], 0) or 0 for r in self.video_rows)
        unresolved_count = sum(to_int(r["unresolved_long_open_count"], 0) or 0 for r in self.video_rows)

        lines = [
            "# v1.3 Train Results Review Printout",
            "",
            "## 전체 요약",
            f"- train video count: {train_video_count}",
            f"- split_seed: {SPLIT_SEED}",
            f"- actual 평균 coverage: {pct(actual_mean)}",
            f"- closed predicted 평균 coverage: {pct(closed_mean)}",
            f"- open 평균 coverage: {pct(open_mean)}",
            f"- unresolved 평균 coverage: {pct(unresolved_mean)}",
            f"- severe over-detection count: {severe_count}",
            f"- worst videos: {', '.join(str(r['video_id']) for r in worst)}",
            f"- closed prediction count: {closed_count}",
            f"- open interval count: {open_count}",
            f"- unresolved long open count: {unresolved_count}",
            f"- missed actual 변화: {missed_delta}",
            "",
            "## 영상별 한 줄 요약",
        ]
        for row in sorted(self.video_rows, key=lambda r: to_int(r["video_id"], 0) or 0):
            lines.append(
                "video {video_id} | actual {actual} | closed {closed} | open {open_} | "
                "unresolved {unresolved} | severity {severity} | main issue: {issue}".format(
                    video_id=row["video_id"],
                    actual=pct(to_float(row["actual_coverage_ratio"])),
                    closed=pct(to_float(row["closed_coverage_ratio"])),
                    open_=pct(to_float(row["open_coverage_ratio"])),
                    unresolved=pct(to_float(row["unresolved_coverage_ratio"])),
                    severity=row["severity_level"],
                    issue=row["main_issue"],
                )
            )
        lines.extend(["", "## Worst 5 영상 상세"])
        for row in self.worst_rows[:5]:
            video_id = to_int(row["video_id"])
            false_positive_notes = [
                x
                for x in self.interval_rows
                if to_int(x["video_id"]) == video_id
                and x["interval_type"] != "actual"
                and to_float(x["false_positive_duration_sec"]) > 0
            ]
            missed_notes = [
                x
                for x in self.interval_rows
                if to_int(x["video_id"]) == video_id
                and x["interval_type"] == "actual"
                and "missed_actual_sec=0.0" not in str(x["review_note"])
            ]
            lines.extend(
                [
                    f"### video {video_id} - {row['focus_reason']}",
                    f"- 실제 광고 구간: {row['actual_intervals_mmss']}",
                    f"- closed prediction: {row['closed_predictions_mmss']}",
                    f"- open interval: {row['open_intervals_mmss']}",
                    f"- unresolved long open: {row['unresolved_intervals_mmss']}",
                    f"- false positive 의심 구간: {self.short_interval_notes(false_positive_notes[:5])}",
                    f"- missed actual 의심 구간: {self.short_interval_notes(missed_notes[:5])}",
                    f"- 주요 decision reason: {row['top_decision_reasons']}",
                    f"- viewer 확인 시각: {self.viewer_hint_for_worst(video_id)}",
                    "",
                ]
            )
        lines.extend(["## Reason 분포"])
        for category in [
            "start_reason",
            "end_reason",
            "decision_reason",
            "in_ad_maintain",
            "pending_timeout",
            "v1_3_event",
        ]:
            subset = [r for r in self.reason_rows if r["category"] == category][:8]
            lines.append(f"### {category}")
            for r in subset:
                lines.append(f"- {r['key']}: {r['count']} ({pct(to_float(r['ratio']))})")
        lines.extend(["", "## v1.4 후보"])
        for row in self.candidate_rows[:5]:
            lines.append(
                f"- {row['candidate_id']} [{row['priority']}] {row['issue_summary']} | "
                f"근거: {row['evidence_from_v1_3_train']} | 주의: {row['risk']}"
            )
        lines.extend(
            [
                "",
                "## 다음 단계",
                "- 이 출력 결과를 보고 v1.4 rule 수정 범위를 선택한다.",
                "- validation/test는 아직 실행하지 않는다.",
            ]
        )
        return "\n".join(lines)

    def missed_actual_change_note(self) -> str:
        v12_missed = []
        v13_missed = []
        by_video = {to_int(r.get("video_id")): r for r in self.comparison_rows}
        # v1.2 comparison에는 missed duration이 항상 없으므로 v1.3
        # train table을 사용하고 review-only 비교로 표시한다.
        total_missed = sum(to_float(r["missed_actual_duration_sec"]) for r in self.video_rows)
        if not by_video:
            return f"v1.2 missed 비교 입력 부족; v1.3 missed total={total_missed:.1f}s"
        return f"v1.2 대비 직접 missed 증감 컬럼은 없음; v1.3 missed total={total_missed:.1f}s, review-only"

    def short_interval_notes(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "none"
        return "; ".join(
            f"{r['interval_type']} {r['mmss_start']}-{r['mmss_end']} fp={r['false_positive_duration_sec']}s"
            for r in rows
        )

    def viewer_hint_for_worst(self, video_id: int) -> str:
        row = next((r for r in self.video_rows if to_int(r["video_id"]) == video_id), None)
        if not row:
            return "trace first in_ad and end_pending events"
        return row["viewer_review_hint"]

    def build_readable_markdown(self, console: str) -> str:
        return (
            console
            + "\n\n## 산출 파일\n"
            + "\n".join(f"- {path}" for key, path in self.output_paths.items() if key != "log")
            + "\n"
        )

    def build_diagnosis_text(self) -> str:
        candidates = "\n".join(
            [
                (
                    f"### {row['candidate_id']} {row['issue_summary']}\n"
                    f"- suspected_rule_area: {row['suspected_rule_area']}\n"
                    f"- evidence: {row['evidence_from_v1_3_train']}\n"
                    f"- suggested_direction: {row['suggested_direction']}\n"
                    f"- caution: {row['risk']}\n"
                    f"- requires_manual_review: {row['needs_manual_review']}\n"
                )
                for row in self.candidate_rows
            ]
        )
        return f"""# v1.3 Train Failure Diagnosis for v1.4

## 결론
- v1.3은 v1.2 대비 open coverage는 줄었지만 train over-detection은 여전히 심각한 검토 대상이다.
- severe over-detection이 남아 있으므로 validation으로 넘어가기 전에 train 실패 원인을 먼저 확인해야 한다.
- v1.4는 train over-detection 원인 수정 후보를 좁히는 것이 목적이어야 한다.

## 핵심 문제
1. start gate 문제: start confirmation이 여전히 긴 후보의 출발점이 되는지 확인해야 한다.
2. in_ad continuity 문제: strong_ad_evidence와 OCR/audio context가 광고 상태를 오래 유지시키는지 확인해야 한다.
3. end/open unresolved 문제: open/unresolved candidate가 별도로 줄었더라도 영상 단위 coverage가 여전히 큰지 확인해야 한다.
4. OCR/audio signal이 광고 상태 유지에 과도하게 작동하는 문제: medium/high level이 strong evidence처럼 작동하는 구간을 샘플링해야 한다.
5. 실제 광고와 detector prediction boundary mismatch 문제: actual label은 audit-only로 쓰고, boundary 주변 trace를 검토해야 한다.

## v1.4 후보
{candidates}
"""

    def generate_report(self) -> None:
        self.log("[STEP 09] Generate report JSON")
        old_after_path = (
            self.root
            / "reports/analysis/old_project_snapshot_after_v1_3_train_results_printout.tsv"
        )
        self.old_snapshot_after = snapshot_project(Path("./_old_project_not_included"), old_after_path)
        self.input_stats_after = file_stats(list(self.paths.values()))
        input_modified = [
            path
            for path, before in self.input_stats_before.items()
            if before != self.input_stats_after.get(path)
        ]
        old_modified = self.old_snapshot_before != self.old_snapshot_after
        self.report = {
            "task_name": "v1_3_train_results_printout_for_review",
            "project_root": str(self.root),
            "run_scope": "train_only",
            "split_seed": SPLIT_SEED,
            "detector_reexecuted": False,
            "rule_modified": False,
            "threshold_tuning": False,
            "feature_extraction": False,
            "input_files": {k: str(v) for k, v in self.paths.items()},
            "output_files": {k: str(v) for k, v in self.output_paths.items()},
            "latest_bundle": str(self.latest_dir),
            "train_video_ids": TRAIN_IDS,
            "excluded_validation_video_ids": VALIDATION_IDS,
            "excluded_test_video_ids": TEST_IDS,
            "row_counts": {
                "video_table": len(self.video_rows),
                "interval_table": len(self.interval_rows),
                "worst_table": len(self.worst_rows),
                "reason_table": len(self.reason_rows),
                "candidate_table": len(self.candidate_rows),
                "closed_prediction_input": len(self.pred_rows),
                "open_input": len(self.open_rows),
                "unresolved_input": len(self.unresolved_rows),
                "trace_input": len(self.trace_rows),
            },
            "metric_summary": self.metric_summary(),
            "worst_videos": [row["video_id"] for row in self.worst_rows[:10]],
            "rule_candidates": self.candidate_rows,
            "sub_agent_results": [],
            "warnings": self.warnings,
            "errors": self.errors,
            "old_project_modified": old_modified,
            "input_files_modified": input_modified,
            "validation_test_output_count": 0,
            "no_detector_run": True,
            "no_rule_change": True,
            "no_threshold_tuning": True,
            "no_feature_extraction": True,
            "train_only_result_printout": True,
        }
        write_text(
            self.output_paths["report_json"],
            json.dumps(self.report, ensure_ascii=False, indent=2),
        )
        self.log("[STEP 09] report JSON written")

    def metric_summary(self) -> dict[str, Any]:
        n = len(self.video_rows) or 1
        return {
            "train_video_count": len(self.video_rows),
            "mean_actual_coverage": round(sum(to_float(r["actual_coverage_ratio"]) for r in self.video_rows) / n, 6),
            "mean_closed_coverage": round(sum(to_float(r["closed_coverage_ratio"]) for r in self.video_rows) / n, 6),
            "mean_open_coverage": round(sum(to_float(r["open_coverage_ratio"]) for r in self.video_rows) / n, 6),
            "mean_unresolved_coverage": round(sum(to_float(r["unresolved_coverage_ratio"]) for r in self.video_rows) / n, 6),
            "severe_over_detection_count": sum(
                1 for r in self.video_rows if r["severity_level"] == "severe_over_detection"
            ),
            "closed_prediction_count": len(self.pred_rows),
            "open_interval_count": len(self.open_rows),
            "unresolved_long_open_count": len(self.unresolved_rows),
        }

    def internal_validation(self) -> None:
        self.log("[STEP 10] Sub Agent validations")
        if len(self.video_rows) != len(TRAIN_IDS):
            raise RuntimeError("video table is not train video count")
        for table_name in ["video_table", "interval_table", "worst_table", "reason_table", "candidate_table"]:
            path = self.output_paths[table_name]
            if not path.exists():
                raise RuntimeError(f"missing output: {path}")
        bad_video_ids = [
            row["video_id"]
            for row in self.video_rows
            if to_int(row["video_id"]) not in TRAIN_IDS
        ]
        if bad_video_ids:
            raise RuntimeError(f"non-train video ids in video table: {bad_video_ids}")
        for row in self.video_rows:
            for key in [
                "actual_coverage_ratio",
                "closed_coverage_ratio",
                "open_coverage_ratio",
                "unresolved_coverage_ratio",
                "predicted_plus_open_plus_unresolved_coverage_ratio",
            ]:
                value = to_float(row[key])
                if value < -1e-9 or value > 1.000001:
                    raise RuntimeError(f"coverage out of range: video={row['video_id']} key={key} value={value}")
        self.log("[STEP 10] internal validation passed; external Sub Agent results should be recorded by Main Agent")

    def update_latest_bundle(self) -> None:
        self.log("[STEP 11] Update latest bundle")
        if self.latest_dir.exists():
            shutil.rmtree(self.latest_dir)
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        bundle_sources = [
            self.output_paths["readable_md"],
            self.output_paths["console_txt"],
            self.output_paths["diagnosis_md"],
            self.output_paths["video_table"],
            self.output_paths["interval_table"],
            self.output_paths["worst_table"],
            self.output_paths["reason_table"],
            self.output_paths["candidate_table"],
            self.output_paths["report_json"],
            self.output_paths["log"],
            self.output_paths["script"],
        ]
        for src in bundle_sources:
            shutil.copy2(src, self.latest_dir / src.name)
        forbidden = self.scan_forbidden(self.latest_dir)
        readme = (
            "# v1.3 Train Results Printout Latest Files\n\n"
            "- Train-only v1.3 result printout for human review.\n"
            "- Not detector implementation, not rule tuning, and not a final performance report.\n"
            "- Validation/test rows are excluded.\n"
            "- No media/frame/cache/model/raw video/proxy/checkpoint files copied.\n\n"
            "## Files\n"
        )
        for p in sorted(self.latest_dir.iterdir()):
            if p.is_file():
                readme += f"- {p.name}\n"
        readme += "- README_latest_files.md\n"
        (self.latest_dir / "README_latest_files.md").write_text(readme, encoding="utf-8")
        if forbidden:
            raise RuntimeError(f"latest bundle forbidden files found: {forbidden}")
        self.report["latest_for_chatgpt_forbidden_files_found"] = forbidden
        write_text(
            self.output_paths["report_json"],
            json.dumps(self.report, ensure_ascii=False, indent=2),
        )
        shutil.copy2(self.output_paths["report_json"], self.latest_dir / self.output_paths["report_json"].name)
        shutil.copy2(self.output_paths["log"], self.latest_dir / self.output_paths["log"].name)
        self.log("[STEP 11] latest bundle refreshed and forbidden scan clean")

    def scan_forbidden(self, directory: Path) -> list[str]:
        found: list[str] = []
        for p in directory.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(directory)
            parts = {part.lower() for part in rel.parts}
            if p.suffix.lower() in FORBIDDEN_SUFFIXES or parts & FORBIDDEN_DIR_PARTS:
                found.append(str(rel))
        return found

    def final_stdout(self) -> None:
        self.log("[STEP 12] Final stdout summary")
        print(self.console_text)
        print("")
        print("## 작업 상태")
        print("작업 상태: SUCCESS")
        print(f"output paths: {self.output_paths['readable_md']}, {self.output_paths['video_table']}")
        print(f"latest bundle path: {self.latest_dir}")
        print("validation/test output count: 0")
        print(f"old_project_modified: {self.report.get('old_project_modified')}")
        print(f"input_files_modified: {self.report.get('input_files_modified')}")
        print(f"warnings/errors: {self.warnings} / {self.errors}")

    def run(self) -> None:
        self.step01()
        self.load_inputs()
        self.build_video_table()
        self.build_interval_table()
        self.build_worst_detail()
        self.build_reason_table()
        self.build_candidate_table()
        self.write_outputs()
        self.generate_report()
        self.internal_validation()
        self.update_latest_bundle()
        self.final_stdout()


def main() -> int:
    args = parse_args()
    runner = Runner(Path(args.project_root))
    try:
        runner.run()
        return 0
    except Exception as exc:  # noqa: BLE001 - top-level reporting
        runner.errors.append(str(exc))
        runner.log(f"[FAILURE] {exc}")
        if runner.report:
            runner.report["errors"] = runner.errors
            write_text(
                runner.output_paths["report_json"],
                json.dumps(runner.report, ensure_ascii=False, indent=2),
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
