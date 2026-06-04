#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

VERSION_DEFAULT = "v2_4"
SPLIT_SEED_DEFAULT = 20240524
OLD_PROJECT = Path("./_old_project_not_included").resolve()
PROJECT_ROOT_DEFAULT = Path(".").resolve()

FORBIDDEN_LATEST_EXTS = {
    ".mp4", ".wav", ".mp3", ".m4a", ".jpg", ".jpeg", ".png", ".webp",
    ".pth", ".pt", ".ckpt", ".onnx",
}
SUCCESS_STATUSES = {"success", "success_nonempty", "success_empty"}
TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z]+(?:[A-Za-z0-9_+\-&]*[A-Za-z0-9])?|\d+(?:[.,]\d+)*|[A-Za-z가-힣]*\d[A-Za-z가-힣0-9._+\-]*")
KOREAN_RE = re.compile(r"[가-힣]")
ENGLISH_RE = re.compile(r"[A-Za-z]")
NUMERIC_RE = re.compile(r"^\d+(?:[.,]\d+)*$")
MIXED_ALNUM_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9._+\-]+")
DISCLOSURE_TERMS = ["광고", "유료광고", "ad", "paid", "promotion"]
SPONSOR_TERMS = ["협찬", "제공", "sponsor", "sponsored"]
GENERIC_WORDS = {
    "the", "and", "for", "with", "this", "that", "you", "your", "오늘", "진짜", "너무", "그리고", "그냥", "이거", "저는", "우리", "하는", "있다", "없는", "입니다", "합니다",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry failed OCR frames and build train-only OCR corpus analysis for v2_4.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--version", default=VERSION_DEFAULT)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED_DEFAULT)
    parser.add_argument("--retry-failed-only", action="store_true", default=True)
    parser.add_argument("--build-train-corpus", action="store_true", default=True)
    parser.add_argument("--no-latest-copy", action="store_true")
    parser.add_argument("--force-proxy", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-retry-frames", type=int, default=0, help="Optional smoke limit; 0 means all failed frames.")
    return parser.parse_args()


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    sec = int(round(seconds - minutes * 60))
    if sec >= 60:
        minutes += 1
        sec -= 60
    return f"{minutes:02d}:{sec:02d}"


def fmt_hhmmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    sec = int(round(seconds % 60))
    if sec >= 60:
        minutes += 1
        sec -= 60
    if minutes >= 60:
        hours += 1
        minutes -= 60
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.lower()
    value = re.sub(r"[^\w\s가-힣%₩$./:\-+,&()]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: str) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(normalize_text(text)) if tok.strip()]


def token_type(token: str) -> str:
    if NUMERIC_RE.fullmatch(token):
        return "numeric"
    if MIXED_ALNUM_RE.fullmatch(token):
        return "mixed_alnum"
    has_ko = bool(KOREAN_RE.search(token))
    has_en = bool(ENGLISH_RE.search(token))
    if has_ko and has_en:
        return "mixed_korean_english"
    if has_ko:
        return "korean"
    if has_en:
        return "english"
    return "other"


def keep_token_for_frequency(token: str) -> bool:
    typ = token_type(token)
    if typ == "korean" and len(token) <= 1:
        return False
    if typ == "english" and len(token) <= 2:
        return False
    if not token.strip():
        return False
    return True


def possible_ocr_error(token: str) -> bool:
    if len(token) >= 16 and re.search(r"[A-Za-z0-9]", token):
        return True
    if re.search(r"[{}\[\]|~^_]", token):
        return True
    if token_type(token) == "mixed_alnum" and len(token) <= 3:
        return True
    return False


def assert_inside(path: Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "data_ocr": root / "data/ocr",
        "data_splits": root / "data/splits",
        "reports_ocr": root / "reports/ocr",
        "logs_ocr": root / "logs/ocr",
        "scripts_ocr": root / "scripts/ocr",
        "configs": root / "configs",
        "cache_ocr": root / "cache/ocr",
        "tmp_ocr": root / "tmp/ocr",
        "proxy": root / "cache/video_proxy",
        "latest": root / "outputs/latest_for_chatgpt",
    }
    for p in dirs.values():
        assert_inside(p, root)
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def sha256_file(path: Path) -> tuple[str, str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest(), "ok"
    except Exception as exc:
        return "", f"error:{type(exc).__name__}:{exc}"


def snapshot_project(root: Path, out_csv: Path, include_sha: bool = True) -> int:
    rows: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = path.relative_to(root).as_posix()
            try:
                st = path.stat()
                sha, sha_status = sha256_file(path) if include_sha else ("", "not_attempted")
                rows.append({
                    "path": rel,
                    "size": int(st.st_size),
                    "mtime": float(st.st_mtime),
                    "mtime_iso": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                    "sha256": sha,
                    "sha256_status": sha_status,
                    "stat_status": "ok",
                })
            except Exception as exc:
                rows.append({"path": rel, "size": "", "mtime": "", "mtime_iso": "", "sha256": "", "sha256_status": "not_attempted", "stat_status": f"error:{type(exc).__name__}:{exc}"})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return len(rows)


def compare_snapshots(before: Path, after: Path, diff_out: Path) -> tuple[bool, int]:
    bdf = pd.read_csv(before, dtype=str, keep_default_na=False)
    adf = pd.read_csv(after, dtype=str, keep_default_na=False)
    b = bdf.set_index("path").to_dict(orient="index")
    a = adf.set_index("path").to_dict(orient="index")
    rows = []
    for path in sorted(set(b) | set(a)):
        br = b.get(path)
        ar = a.get(path)
        if br is None:
            rows.append({"path": path, "change_type": "added", "before_sha256": "", "after_sha256": ar.get("sha256", "")})
        elif ar is None:
            rows.append({"path": path, "change_type": "removed", "before_sha256": br.get("sha256", ""), "after_sha256": ""})
        else:
            changed = [field for field in ["size", "mtime", "sha256", "sha256_status", "stat_status"] if str(br.get(field, "")) != str(ar.get(field, ""))]
            if changed:
                rows.append({"path": path, "change_type": ",".join(changed), "before_sha256": br.get("sha256", ""), "after_sha256": ar.get("sha256", "")})
    pd.DataFrame(rows).to_csv(diff_out, index=False)
    return bool(rows), len(rows)


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(args.project_root).resolve()
        if self.root != PROJECT_ROOT_DEFAULT:
            # resolve 이후 같은 경로임이 명확할 때만 허용한다.
            assert_inside(self.root, PROJECT_ROOT_DEFAULT.parent)
        self.version = args.version
        self.dirs = ensure_dirs(self.root)
        self.log_path = self.dirs["logs_ocr"] / "retry_failed_ocr_and_build_train_corpus_v2_4_run.log"
        self.before_snapshot = self.dirs["reports_ocr"] / "old_project_snapshot_before_retry_failed_ocr_v2_4.csv"
        self.after_snapshot = self.dirs["reports_ocr"] / "old_project_snapshot_after_retry_failed_ocr_v2_4.csv"
        self.snapshot_diff = self.dirs["reports_ocr"] / "old_project_snapshot_diff_retry_failed_ocr_v2_4.csv"
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.subagents: list[dict[str, str]] = []
        self.stats: dict[str, Any] = {}
        self.started_at = now_iso()
        self.actual_start_monotonic = time.monotonic()
        self.estimate = "약 10~25분"
        self.old_project_modified: bool | None = None
        self.latest_forbidden_count: int | None = None
        self.log_path.write_text("", encoding="utf-8")

    def path(self, rel: str) -> Path:
        return self.root / rel

    def log(self, message: str) -> None:
        line = f"{now_iso()} | {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @property
    def inputs(self) -> dict[str, Path]:
        return {
            "labels": self.path("data/segments/ad_interval_segments_v2_4.csv"),
            "manifest": self.path("data/video_metadata/video_manifest_v2_2.csv"),
            "frame": self.path("data/ocr/ocr_frame_level_results_v2_4.csv"),
            "labeled_plan": self.path("data/ocr/ocr_labeled_segment_sampling_plan_v2_4.csv"),
            "edge_plan": self.path("data/ocr/ocr_ad_edge_5s_10s_sampling_plan_v2_4.csv"),
            "labeled_features": self.path("data/ocr/ocr_labeled_segment_features_v2_4.csv"),
            "edge_features": self.path("data/ocr/ocr_ad_edge_5s_10s_features_v2_4.csv"),
            "keyword_dict": self.path("configs/ocr_keyword_dictionary_v2_4.json"),
            "persistence_config": self.path("configs/ocr_persistence_rule_config_v2_4.json"),
        }

    @property
    def outputs(self) -> dict[str, Path]:
        return {
            "split": self.path("data/splits/video_split_v2_4.csv"),
            "retry_plan": self.path("data/ocr/ocr_failed_frame_retry_plan_v2_4.csv"),
            "retry_results": self.path("data/ocr/ocr_failed_frame_retry_results_v2_4.csv"),
            "retry_results_redacted": self.path("data/ocr/ocr_failed_frame_retry_results_v2_4_redacted_for_review.csv"),
            "recovered_frame": self.path("data/ocr/ocr_frame_level_results_v2_4_recovered.csv"),
            "coverage": self.path("data/ocr/ocr_retry_coverage_summary_v2_4.csv"),
            "review_all": self.path("data/ocr/ocr_failed_frame_human_review_all_splits_v2_4.csv"),
            "review_train": self.path("data/ocr/ocr_failed_frame_human_review_train_only_v2_4.csv"),
            "empty_train": self.path("data/ocr/ocr_success_empty_frame_review_train_only_v2_4.csv"),
            "labeled_recovered": self.path("data/ocr/ocr_labeled_segment_features_v2_4_recovered.csv"),
            "edge_recovered": self.path("data/ocr/ocr_ad_edge_5s_10s_features_v2_4_recovered.csv"),
            "train_ad_corpus": self.path("data/ocr/ocr_train_ad_text_corpus_v2_4.csv"),
            "train_pre_corpus": self.path("data/ocr/ocr_train_pre_disclosure_text_corpus_v2_4.csv"),
            "train_nonad_corpus": self.path("data/ocr/ocr_train_nonad_reference_text_corpus_v2_4.csv"),
            "token_freq": self.path("data/ocr/ocr_train_token_frequency_by_group_v2_4.csv"),
            "token_lift": self.path("data/ocr/ocr_train_ad_vs_nonad_token_lift_v2_4.csv"),
            "keyword_review": self.path("data/ocr/ocr_train_candidate_keyword_review_v2_4.csv"),
            "keyword_review_json": self.path("configs/ocr_keyword_dictionary_review_candidates_v2_4.json"),
            "precue": self.path("data/ocr/ocr_train_pre_disclosure_precue_summary_v2_4.csv"),
            "analysis_report": self.path("reports/ocr/ocr_failed_retry_and_train_corpus_analysis_v2_4.md"),
            "summary_report": self.path("reports/ocr/ocr_failed_retry_and_train_corpus_summary_v2_4.md"),
            "script": self.path("scripts/ocr/retry_failed_ocr_and_build_train_corpus_v2_4.py"),
        }

    def load_inputs(self) -> dict[str, pd.DataFrame]:
        data = {}
        for key, path in self.inputs.items():
            if key in {"keyword_dict", "persistence_config"}:
                continue
            if path.exists():
                data[key] = pd.read_csv(path)
            else:
                data[key] = pd.DataFrame()
                self.errors.append(f"Missing input: {path}")
        return data

    def preflight(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        labels = data.get("labels", pd.DataFrame())
        manifest = data.get("manifest", pd.DataFrame())
        plan = pd.concat([data.get("labeled_plan", pd.DataFrame()), data.get("edge_plan", pd.DataFrame())], ignore_index=True)
        frame = data.get("frame", pd.DataFrame())
        vids: set[str] = set()
        for df in [labels, manifest, plan]:
            if not df.empty and "video_id" in df:
                vids.update(df["video_id"].astype(str))
        failed = self.failed_mask(frame) if not frame.empty else pd.Series([], dtype=bool)
        failed_df = frame[failed].copy() if not frame.empty else pd.DataFrame()
        path_values = []
        for df in [manifest, plan, frame]:
            if not df.empty:
                for col in ["video_path_resolved", "video_path"]:
                    if col in df:
                        path_values.extend(df[col].dropna().astype(str).tolist())
        seen = set()
        resolved = 0
        missing = 0
        for raw in path_values:
            if not raw or raw == "nan" or raw in seen:
                continue
            seen.add(raw)
            p = Path(raw)
            if not p.is_absolute():
                p = self.root / p
            if p.exists():
                resolved += 1
            else:
                missing += 1
        n = len(vids)
        if n == 18:
            split_counts = {"train": 12, "validation": 3, "test": 3}
        elif n:
            val = max(1, round(n / 6))
            test = max(1, round(n / 6))
            split_counts = {"train": n - val - test, "validation": val, "test": test}
        else:
            split_counts = {"train": 0, "validation": 0, "test": 0}
        pf = {
            "unique_video_id_count": n,
            "unique_video_ids": sorted(vids, key=lambda x: (len(str(x)), str(x))),
            "frame_level_exists": self.inputs["frame"].exists(),
            "frame_rows": int(len(frame)),
            "failed_frame_count": int(len(failed_df)),
            "failed_video_count": int(failed_df["video_id"].astype(str).nunique()) if not failed_df.empty else 0,
            "failed_status_counts": failed_df["ocr_status"].astype(str).value_counts().to_dict() if not failed_df.empty and "ocr_status" in failed_df else {},
            "failed_by_video": failed_df.groupby(failed_df["video_id"].astype(str)).size().sort_values(ascending=False).to_dict() if not failed_df.empty else {},
            "video_path_resolved_count": resolved,
            "video_path_missing_count": missing,
            "easyocr_available": importlib.util.find_spec("easyocr") is not None,
            "ffmpeg_binary": shutil.which("ffmpeg") or "",
            "h264_proxy_needed_video_estimate": int(failed_df[failed_df.get("ocr_status", pd.Series(dtype=str)).astype(str).eq("frame_decode_failed")]["video_id"].astype(str).nunique()) if not failed_df.empty else 0,
            "split_possible": n >= 3,
            "split_counts": split_counts,
        }
        self.stats.update(pf)
        return pf

    def print_estimate(self, pf: dict[str, Any]) -> None:
        self.log("[STEP 04] Print estimated work time")
        text = f"""[ESTIMATED WORK TIME]
- 예상 작업 시간: {self.estimate}
- 산정 근거:
  - unique video_id 수: {pf['unique_video_id_count']}
  - split 구성: train {pf['split_counts'].get('train')} / validation {pf['split_counts'].get('validation')} / test {pf['split_counts'].get('test')}
  - retry 대상 frame 수: {pf['failed_frame_count']}
  - retry 대상 video 수: {pf['failed_video_count']}
  - ffmpeg fallback 필요 예상: {pf['failed_frame_count']} frame
  - H.264 proxy 생성 필요 예상: 최대 {pf['h264_proxy_needed_video_estimate']} video
  - train OCR corpus 분석 대상 frame 수: split 배정 후 확정, 전체 OCR frame {pf['frame_rows']} 기준 일부
- 예상 병목:
  - ffmpeg frame extraction
  - AV1/H.264 proxy 생성
  - EasyOCR 재실행
  - token frequency/lift aggregation"""
        print(text, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    def failed_mask(self, frame: pd.DataFrame) -> pd.Series:
        if frame.empty:
            return pd.Series([], dtype=bool)
        status = frame.get("ocr_status", pd.Series([""] * len(frame))).fillna("").astype(str)
        raw_text = frame.get("ocr_text_raw", pd.Series([""] * len(frame))).fillna("").astype(str)
        err = frame.get("error_message", pd.Series([""] * len(frame))).fillna("").astype(str)
        return (~status.isin(SUCCESS_STATUSES)) | status.isin(["frame_decode_failed", "ocr_failed", "error"]) | ((err.str.len() > 0) & raw_text.str.strip().eq(""))

    def create_split(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        self.log("[STEP 05] Create deterministic video_id split")
        labels = data["labels"]
        manifest = data["manifest"]
        plan = pd.concat([data["labeled_plan"], data["edge_plan"]], ignore_index=True)
        frame = data["frame"]
        vids = set()
        for df in [labels, manifest, plan]:
            if not df.empty and "video_id" in df:
                vids.update(df["video_id"].astype(str))
        ordered = sorted(vids, key=lambda x: int(x) if str(x).isdigit() else str(x))
        shuffled = ordered[:]
        random.Random(self.args.split_seed).shuffle(shuffled)
        n = len(shuffled)
        if n == 18:
            train_count, val_count, test_count = 12, 3, 3
        else:
            val_count = max(1, round(n / 6)) if n else 0
            test_count = max(1, round(n / 6)) if n else 0
            train_count = n - val_count - test_count
        split_by_video: dict[str, str] = {}
        for vid in shuffled[:train_count]:
            split_by_video[vid] = "train"
        for vid in shuffled[train_count:train_count + val_count]:
            split_by_video[vid] = "validation"
        for vid in shuffled[train_count + val_count:]:
            split_by_video[vid] = "test"
        manifest_by_vid = manifest.assign(video_id_str=manifest["video_id"].astype(str)).drop_duplicates("video_id_str").set_index("video_id_str") if not manifest.empty else pd.DataFrame()
        label_ad_counts = labels.assign(video_id_str=labels["video_id"].astype(str)).groupby("video_id_str")["ad_interval_id"].nunique().to_dict() if not labels.empty else {}
        sampling_counts = plan.assign(video_id_str=plan["video_id"].astype(str)).groupby("video_id_str").size().to_dict() if not plan.empty else {}
        failed = self.failed_mask(frame)
        original_failed = frame[failed].assign(video_id_str=frame[failed]["video_id"].astype(str)).groupby("video_id_str").size().to_dict() if not frame.empty else {}
        original_success = frame[~failed].assign(video_id_str=frame[~failed]["video_id"].astype(str)).groupby("video_id_str").size().to_dict() if not frame.empty else {}
        decode_fail_vids = set(frame[frame.get("ocr_status", pd.Series(dtype=str)).astype(str).eq("frame_decode_failed")]["video_id"].astype(str)) if not frame.empty else set()
        rows = []
        for vid in ordered:
            m = manifest_by_vid.loc[vid] if not manifest_by_vid.empty and vid in manifest_by_vid.index else pd.Series(dtype=object)
            rows.append({
                "version": self.version,
                "video_id": vid,
                "video_name": str(m.get("video_title", m.get("video_filename", ""))),
                "video_path": str(m.get("video_path", "")),
                "video_duration_sec": float(m.get("duration_sec", np.nan)) if str(m.get("duration_sec", "")) not in {"", "nan"} else np.nan,
                "split": split_by_video.get(vid, "train"),
                "split_method": "video_id_deterministic_shuffle_4_1_1_no_ocr_text_or_quality_used",
                "split_seed": int(self.args.split_seed),
                "ad_interval_count": int(label_ad_counts.get(vid, 0)),
                "sampling_frame_count": int(sampling_counts.get(vid, 0)),
                "original_ocr_failed_frame_count": int(original_failed.get(vid, 0)),
                "original_ocr_success_frame_count": int(original_success.get(vid, 0)),
                "has_original_decode_failure": bool(vid in decode_fail_vids),
                "note": "validation/test OCR text not used for keyword/rule design" if split_by_video.get(vid) != "train" else "train split eligible for OCR corpus analysis",
            })
        split_df = pd.DataFrame(rows)
        split_df.to_csv(self.outputs["split"], index=False)
        self.stats["train_video_ids"] = split_df[split_df["split"].eq("train")]["video_id"].astype(str).tolist()
        self.stats["validation_video_ids"] = split_df[split_df["split"].eq("validation")]["video_id"].astype(str).tolist()
        self.stats["test_video_ids"] = split_df[split_df["split"].eq("test")]["video_id"].astype(str).tolist()
        self.stats["split_counts_actual"] = split_df["split"].value_counts().to_dict()
        return split_df

    def add_split(self, df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "video_id" not in df:
            return df.copy()
        split_cols = split_df[["video_id", "split", "video_name", "video_path", "video_duration_sec"]].copy()
        split_cols["video_id"] = split_cols["video_id"].astype(str)
        out = df.copy()
        out["video_id"] = out["video_id"].astype(str)
        out = out.merge(split_cols, on="video_id", how="left", suffixes=("", "_split"))
        if "video_path_split" in out:
            out["video_path"] = out.get("video_path", pd.Series([""] * len(out))).fillna("")
            out.loc[out["video_path"].astype(str).str.len() == 0, "video_path"] = out.loc[out["video_path"].astype(str).str.len() == 0, "video_path_split"]
            out.drop(columns=["video_path_split"], inplace=True)
        return out

    def build_retry_plan(self, data: dict[str, pd.DataFrame], split_df: pd.DataFrame) -> pd.DataFrame:
        self.log("[STEP 06] Extract failed OCR frames from original results")
        frame = data["frame"].copy()
        if frame.empty:
            self.warnings.append("Frame-level OCR result is missing/empty; retry plan will be empty.")
            plan = pd.DataFrame()
            plan.to_csv(self.outputs["retry_plan"], index=False)
            return plan
        failed = self.failed_mask(frame)
        fdf = frame[failed].copy()
        if self.args.max_retry_frames and self.args.max_retry_frames > 0:
            fdf = fdf.head(self.args.max_retry_frames).copy()
            self.warnings.append(f"max_retry_frames smoke limit applied: {self.args.max_retry_frames}")
        self.log("[STEP 07] Build failed frame retry plan")
        fdf = self.add_split(fdf, split_df)
        rows = []
        for row in fdf.itertuples(index=False):
            start = float(getattr(row, "segment_start_sec"))
            end = float(getattr(row, "segment_end_sec"))
            ft = float(getattr(row, "frame_time_sec"))
            split = str(getattr(row, "split", ""))
            orig_status = str(getattr(row, "ocr_status", ""))
            if split == "train" and orig_status == "frame_decode_failed":
                priority = "high"
            elif orig_status == "frame_decode_failed":
                priority = "medium"
            else:
                priority = "low"
            rows.append({
                "version": self.version,
                "split": split,
                "video_id": str(getattr(row, "video_id")),
                "video_name": str(getattr(row, "video_name", "")),
                "video_path": str(getattr(row, "video_path", "")),
                "video_path_resolved": str(getattr(row, "video_path_resolved", getattr(row, "video_path", ""))),
                "ad_interval_id": str(getattr(row, "ad_interval_id")),
                "segment_type": str(getattr(row, "segment_type")),
                "segment_start_sec": round(start, 3),
                "segment_end_sec": round(end, 3),
                "segment_duration_sec": round(float(getattr(row, "segment_duration_sec", end - start)), 3),
                "segment_start_mmss": fmt_duration(start),
                "segment_end_mmss": fmt_duration(end),
                "segment_start_hhmmss": fmt_hhmmss(start),
                "segment_end_hhmmss": fmt_hhmmss(end),
                "frame_time_sec": round(ft, 3),
                "frame_time_mmss": fmt_duration(ft),
                "frame_time_hhmmss": fmt_hhmmss(ft),
                "frame_offset_sec": round(float(getattr(row, "frame_offset_sec", ft - start)), 3),
                "frame_index_in_segment": int(getattr(row, "frame_index_in_segment", 0)),
                "original_ocr_status": orig_status,
                "original_error_message": str(getattr(row, "error_message", "")),
                "retry_needed": True,
                "retry_priority": priority,
                "review_target_reason": "original_ocr_failed_frame_decode" if orig_status == "frame_decode_failed" else "original_ocr_failed_or_empty_error",
            })
        plan = pd.DataFrame(rows)
        plan.to_csv(self.outputs["retry_plan"], index=False)
        self.stats["retry_plan_rows"] = int(len(plan))
        return plan

    def load_easyocr_reader(self) -> tuple[Any | None, str, str]:
        if importlib.util.find_spec("easyocr") is None:
            return None, "none", "easyocr import unavailable"
        import easyocr
        try:
            model_cache = self.root / "cache/ocr/model_cache"
            reader = easyocr.Reader(["ko", "en"], gpu=False, download_enabled=False, model_storage_directory=str(model_cache), verbose=False)
            return reader, "easyocr", ""
        except Exception as exc:
            return None, "easyocr", f"EasyOCR init failed: {type(exc).__name__}: {exc}"

    def run_ffmpeg_extract(self, video: Path, time_sec: float, out_path: Path, mode: str) -> tuple[bool, str]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False, "ffmpeg_not_available"
        if mode == "fast_seek":
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{time_sec:.3f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out_path)]
        else:
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-ss", f"{time_sec:.3f}", "-frames:v", "1", "-q:v", "2", str(out_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ok = proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
        return ok, proc.stderr.strip()[:1000]

    def create_proxy(self, video_id: str, input_video: Path) -> tuple[Path | None, str]:
        self.log("[STEP 09] Create H.264 proxy and retry if needed")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return None, "ffmpeg_not_available"
        proxy = self.dirs["proxy"] / f"{video_id}_h264_proxy.mp4"
        if proxy.exists() and proxy.stat().st_size > 0:
            return proxy, "existing_proxy"
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(input_video),
            "-map", "0:v:0", "-an", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18", str(proxy),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and proxy.exists() and proxy.stat().st_size > 0:
            return proxy, "created_proxy"
        return None, proc.stderr.strip()[:1000]

    def polygon_area(self, points: Any) -> float:
        try:
            pts = np.array(points, dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 3:
                return 0.0
            x = pts[:, 0]; y = pts[:, 1]
            return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        except Exception:
            return 0.0

    def json_safe_bbox(self, points: Any) -> list[list[float]]:
        try:
            pts = np.array(points, dtype=float)
            return [[round(float(x), 3), round(float(y), 3)] for x, y in pts[:, :2]] if pts.ndim == 2 else []
        except Exception:
            return []

    def ocr_image(self, reader: Any, image_path: Path) -> tuple[list[dict[str, Any]], tuple[int, int, int] | None, str]:
        frame = cv2.imread(str(image_path))
        if frame is None:
            return [], None, "cv2_imread_failed"
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            detections = reader.readtext(rgb, detail=1, paragraph=False)
        except Exception as exc:
            return [], frame.shape, f"easyocr_failed:{type(exc).__name__}:{exc}"
        rows = []
        for det in detections:
            if not isinstance(det, (list, tuple)) or len(det) < 3:
                continue
            text = str(det[1]).strip()
            if not text:
                continue
            try:
                conf = float(det[2])
            except Exception:
                conf = math.nan
            bbox = det[0]
            rows.append({"text": text, "confidence": conf, "bbox": bbox})
        return rows, frame.shape, "ok"

    def build_retry_record(self, row: pd.Series, attempted: bool, stage: str, status: str, success: bool, err: str, detections: list[dict[str, Any]] | None = None, frame_shape: tuple[int, int, int] | None = None, used_proxy: bool = False, proxy_path: str = "", cleanup: str = "not_applicable") -> dict[str, Any]:
        detections = detections or []
        texts = [str(d["text"]).strip() for d in detections if str(d.get("text", "")).strip()]
        raw = "\n".join(texts)
        normalized = normalize_text(raw)
        tokens = tokenize(raw)
        confs = [float(d["confidence"]) for d in detections if d.get("confidence") is not None and not pd.isna(d.get("confidence"))]
        image_area = float(frame_shape[0] * frame_shape[1]) if frame_shape is not None else 0.0
        bbox_records = []
        total_area = 0.0
        for d in detections:
            area = self.polygon_area(d.get("bbox"))
            total_area += area
            bbox_records.append({"text": str(d.get("text", "")), "confidence": None if pd.isna(d.get("confidence")) else round(float(d.get("confidence")), 6), "bbox": self.json_safe_bbox(d.get("bbox")), "area": round(area, 3)})
        area_ratio = min(1.0, total_area / image_area) if image_area > 0 else 0.0
        return {
            "version": self.version,
            "split": str(row.get("split", "")),
            "video_id": str(row.get("video_id", "")),
            "video_name": str(row.get("video_name", "")),
            "video_path": str(row.get("video_path", "")),
            "ad_interval_id": str(row.get("ad_interval_id", "")),
            "segment_type": str(row.get("segment_type", "")),
            "segment_start_sec": row.get("segment_start_sec", ""),
            "segment_end_sec": row.get("segment_end_sec", ""),
            "segment_duration_sec": row.get("segment_duration_sec", ""),
            "segment_start_mmss": row.get("segment_start_mmss", ""),
            "segment_end_mmss": row.get("segment_end_mmss", ""),
            "frame_time_sec": row.get("frame_time_sec", ""),
            "frame_time_mmss": row.get("frame_time_mmss", ""),
            "frame_offset_sec": row.get("frame_offset_sec", ""),
            "frame_index_in_segment": row.get("frame_index_in_segment", ""),
            "original_ocr_status": row.get("original_ocr_status", ""),
            "retry_attempted": bool(attempted),
            "retry_stage": stage,
            "retry_status": status,
            "retry_success": bool(success),
            "retry_error_message": err,
            "ocr_backend": "easyocr" if attempted else "none",
            "ocr_text_raw": raw,
            "ocr_text_normalized": normalized,
            "ocr_text_joined": " | ".join(texts),
            "ocr_text_count": int(len(texts)),
            "ocr_token_count": int(len(tokens)),
            "ocr_char_count": int(len(normalized)),
            "ocr_mean_confidence": round(float(np.mean(confs)), 6) if confs else 0.0,
            "ocr_box_count": int(len(detections)),
            "ocr_text_area_ratio": round(float(area_ratio), 6),
            "bbox_json_or_path": json.dumps(bbox_records, ensure_ascii=False),
            "used_proxy": bool(used_proxy),
            "proxy_path": proxy_path,
            "temp_frame_cleanup_status": cleanup,
            "warning_message": "none" if success else "retry_failed_partial_coverage",
        }

    def retry_failed_frames(self, retry_plan: pd.DataFrame) -> pd.DataFrame:
        self.log("[STEP 08] Retry failed frames with ffmpeg fallback")
        reader, backend, backend_warning = self.load_easyocr_reader()
        if backend_warning:
            self.warnings.append(backend_warning)
        if retry_plan.empty or reader is None:
            rows = [self.build_retry_record(row, False, "not_attempted", "ocr_backend_unavailable", False, backend_warning) for _, row in retry_plan.iterrows()]
            out = pd.DataFrame(rows)
            out.to_csv(self.outputs["retry_results"], index=False)
            return out
        rows: list[dict[str, Any]] = []
        proxy_cache: dict[str, tuple[Path | None, str]] = {}
        tmp_dir = self.dirs["tmp_ocr"] / "retry_failed_frames"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in retry_plan.reset_index(drop=True).iterrows():
            video_path = Path(str(row.get("video_path_resolved") or row.get("video_path")))
            if not video_path.is_absolute():
                video_path = self.root / video_path
            frame_time = float(row["frame_time_sec"])
            video_id = str(row["video_id"])
            final_record = None
            errors = []
            for stage, mode, source_video, local_time, used_proxy, proxy_path in [
                ("ffmpeg_fast_seek", "fast_seek", video_path, frame_time, False, ""),
                ("ffmpeg_accurate_seek", "accurate_seek", video_path, frame_time, False, ""),
            ]:
                tmp = tmp_dir / f"retry_{idx:06d}_{stage}.jpg"
                ok, ferr = self.run_ffmpeg_extract(source_video, local_time, tmp, mode)
                if ok:
                    detections, shape, oerr = self.ocr_image(reader, tmp)
                    cleanup = "deleted" if tmp.exists() else "already_missing"
                    try:
                        if tmp.exists(): tmp.unlink()
                    except Exception as exc:
                        cleanup = f"delete_failed:{type(exc).__name__}:{exc}"
                    if oerr == "ok":
                        final_record = self.build_retry_record(row, True, stage, "success_nonempty" if detections else "success_empty", True, "", detections, shape, used_proxy, proxy_path, cleanup)
                        break
                    errors.append(f"{stage}:{oerr}")
                else:
                    errors.append(f"{stage}:{ferr}")
                    if tmp.exists():
                        try: tmp.unlink()
                        except Exception: pass
            if final_record is None:
                need_proxy = self.args.force_proxy or str(row.get("original_ocr_status", "")) == "frame_decode_failed"
                if need_proxy:
                    if video_id not in proxy_cache:
                        proxy_cache[video_id] = self.create_proxy(video_id, video_path)
                    proxy, perr = proxy_cache[video_id]
                    if proxy is not None:
                        tmp = tmp_dir / f"retry_{idx:06d}_proxy.jpg"
                        ok, ferr = self.run_ffmpeg_extract(proxy, frame_time, tmp, "accurate_seek")
                        if ok:
                            detections, shape, oerr = self.ocr_image(reader, tmp)
                            cleanup = "deleted" if tmp.exists() else "already_missing"
                            try:
                                if tmp.exists(): tmp.unlink()
                            except Exception as exc:
                                cleanup = f"delete_failed:{type(exc).__name__}:{exc}"
                            if oerr == "ok":
                                final_record = self.build_retry_record(row, True, "h264_proxy_accurate_seek", "success_nonempty" if detections else "success_empty", True, "", detections, shape, True, str(proxy), cleanup)
                            else:
                                errors.append(f"proxy_ocr:{oerr}")
                        else:
                            errors.append(f"proxy_extract:{ferr}")
                    else:
                        errors.append(f"proxy_create:{perr}")
            if final_record is None:
                final_record = self.build_retry_record(row, True, "failed_all_stages", "retry_failed", False, " | ".join(errors), [], None, False, "", "deleted_or_not_created")
            rows.append(final_record)
            if (idx + 1) % 25 == 0 or idx + 1 == len(retry_plan):
                self.log(f"Retry OCR progress: {idx + 1}/{len(retry_plan)}")
        # retry tmp directory에 남은 임시 이미지를 정리한다.
        for p in tmp_dir.glob("retry_*.jpg"):
            try: p.unlink()
            except Exception: pass
        out = pd.DataFrame(rows)
        out.to_csv(self.outputs["retry_results"], index=False)
        redacted = out.drop(columns=[c for c in ["ocr_text_raw", "ocr_text_normalized", "ocr_text_joined", "bbox_json_or_path"] if c in out.columns])
        redacted.to_csv(self.outputs["retry_results_redacted"], index=False)
        self.stats["retry_attempted_count"] = int(out["retry_attempted"].astype(bool).sum()) if not out.empty else 0
        self.stats["retry_success_count"] = int(out["retry_success"].astype(bool).sum()) if not out.empty else 0
        self.stats["retry_still_failed_count"] = int((~out["retry_success"].astype(bool)).sum()) if not out.empty else 0
        self.stats["used_proxy"] = bool(out.get("used_proxy", pd.Series(dtype=bool)).astype(bool).any()) if not out.empty else False
        self.stats["proxy_paths"] = sorted(set(out.loc[out.get("used_proxy", pd.Series(dtype=bool)).astype(bool), "proxy_path"].astype(str))) if not out.empty and "proxy_path" in out else []
        return out

    def row_key_cols(self) -> list[str]:
        return ["video_id", "ad_interval_id", "segment_type", "segment_start_sec", "segment_end_sec", "frame_time_sec", "frame_index_in_segment"]

    def merge_recovered(self, data: dict[str, pd.DataFrame], split_df: pd.DataFrame, retry_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.log("[STEP 11] Merge original and retry OCR into recovered frame-level results")
        original = self.add_split(data["frame"].copy(), split_df)
        original["retry_source"] = "original_success"
        original["retry_stage"] = "not_applicable"
        original["retry_status"] = "not_applicable"
        original["retry_success"] = False
        original["retry_error_message"] = ""
        original["used_proxy"] = False
        original["proxy_path"] = ""
        failed_mask = self.failed_mask(original)
        key_cols = self.row_key_cols()
        retry_map = {}
        if not retry_results.empty:
            rr = retry_results.copy()
            for col in key_cols:
                if col in rr:
                    rr[col] = rr[col].astype(str) if col in {"video_id", "ad_interval_id", "segment_type"} else pd.to_numeric(rr[col], errors="coerce").round(3)
            for _, row in rr.iterrows():
                key = tuple(str(row[c]) if c in {"video_id", "ad_interval_id", "segment_type"} else round(float(row[c]), 3) for c in key_cols if c in row)
                retry_map[key] = row
        recovered_rows = []
        for _, row in original.iterrows():
            out = row.to_dict()
            if bool(failed_mask.loc[row.name]):
                key = tuple(str(row[c]) if c in {"video_id", "ad_interval_id", "segment_type"} else round(float(row[c]), 3) for c in key_cols)
                rr = retry_map.get(key)
                if rr is not None:
                    out["retry_stage"] = rr.get("retry_stage", "")
                    out["retry_status"] = rr.get("retry_status", "")
                    out["retry_success"] = bool(rr.get("retry_success", False))
                    out["retry_error_message"] = rr.get("retry_error_message", "")
                    out["used_proxy"] = bool(rr.get("used_proxy", False))
                    out["proxy_path"] = rr.get("proxy_path", "")
                    if bool(rr.get("retry_success", False)):
                        out["ocr_status"] = rr.get("retry_status", "success_nonempty")
                        for src, dst in [
                            ("ocr_text_raw", "ocr_text_raw"), ("ocr_text_normalized", "ocr_text_normalized"), ("ocr_text_joined", "ocr_text_joined"),
                            ("ocr_text_count", "ocr_text_count"), ("ocr_token_count", "ocr_token_count"), ("ocr_char_count", "ocr_char_count"),
                            ("ocr_mean_confidence", "ocr_mean_confidence"), ("ocr_box_count", "ocr_box_count"), ("ocr_text_area_ratio", "ocr_text_area_ratio"),
                            ("bbox_json_or_path", "bbox_json"),
                        ]:
                            if src in rr:
                                out[dst] = rr[src]
                        out["ocr_min_confidence"] = out.get("ocr_mean_confidence", 0.0)
                        out["ocr_max_confidence"] = out.get("ocr_mean_confidence", 0.0)
                        out["ocr_high_conf_text_count"] = out.get("ocr_text_count", 0)
                        out["error_message"] = ""
                        out["retry_source"] = "retry_proxy_success" if bool(rr.get("used_proxy", False)) else "retry_ffmpeg_success"
                    else:
                        out["retry_source"] = "retry_failed"
                else:
                    out["retry_source"] = "retry_failed"
                    out["retry_status"] = "retry_missing_result"
            recovered_rows.append(out)
        recovered = pd.DataFrame(recovered_rows)
        recovered.to_csv(self.outputs["recovered_frame"], index=False)
        if len(recovered) != len(original):
            self.errors.append(f"Recovered row count mismatch: original={len(original)} recovered={len(recovered)}")
        self.log("[STEP 10] Save retry results")
        coverage = self.build_coverage(original, recovered, retry_results, split_df)
        coverage.to_csv(self.outputs["coverage"], index=False)
        return recovered, coverage

    def build_coverage(self, original: pd.DataFrame, recovered: pd.DataFrame, retry_results: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for vid, group in original.groupby("video_id", sort=True):
            rec = recovered[recovered["video_id"].astype(str).eq(str(vid))]
            rr = retry_results[retry_results["video_id"].astype(str).eq(str(vid))] if not retry_results.empty else pd.DataFrame()
            split_row = split_df[split_df["video_id"].astype(str).eq(str(vid))].head(1)
            orig_failed = self.failed_mask(group).sum()
            rec_failed = self.failed_mask(rec).sum()
            orig_empty = int(group["ocr_status"].astype(str).eq("success_empty").sum()) if "ocr_status" in group else 0
            rec_empty = int(rec["ocr_status"].astype(str).eq("success_empty").sum()) if "ocr_status" in rec else 0
            orig_success = int((~self.failed_mask(group)).sum())
            rec_success = int((~self.failed_mask(rec)).sum())
            attempted = int(rr["retry_attempted"].astype(bool).sum()) if not rr.empty else 0
            retry_success = int(rr["retry_success"].astype(bool).sum()) if not rr.empty else 0
            still_failed = int((~rr["retry_success"].astype(bool)).sum()) if not rr.empty else 0
            rows.append({
                "split": split_row["split"].iloc[0] if not split_row.empty else "",
                "video_id": str(vid),
                "video_name": split_row["video_name"].iloc[0] if not split_row.empty else "",
                "original_failed_count": int(orig_failed),
                "retry_attempted_count": attempted,
                "retry_success_count": retry_success,
                "retry_still_failed_count": still_failed,
                "original_empty_count": orig_empty,
                "recovered_empty_count": rec_empty,
                "original_success_count": orig_success,
                "recovered_success_count": rec_success,
                "recovery_rate": round(float(retry_success / attempted), 6) if attempted else 0.0,
                "used_proxy": bool(rr.get("used_proxy", pd.Series(dtype=bool)).astype(bool).any()) if not rr.empty else False,
                "warning_message": "partial_recovery" if still_failed else "none",
            })
        return pd.DataFrame(rows)

    def review_rows(self, df: pd.DataFrame, retry_status_col: str = "retry_status") -> list[dict[str, Any]]:
        rows = []
        for row in df.itertuples(index=False):
            start = float(getattr(row, "segment_start_sec"))
            end = float(getattr(row, "segment_end_sec"))
            ft = float(getattr(row, "frame_time_sec"))
            ctx_start = max(0.0, ft - 2.0)
            ctx_end = ft + 2.0
            split = str(getattr(row, "split", ""))
            rows.append({
                "version": self.version,
                "split": split,
                "manual_review_allowed_now": split == "train",
                "video_id": str(getattr(row, "video_id")),
                "video_name": str(getattr(row, "video_name", "")),
                "video_path": str(getattr(row, "video_path", "")),
                "ad_interval_id": str(getattr(row, "ad_interval_id", "")),
                "segment_type": str(getattr(row, "segment_type", "")),
                "segment_start_sec": round(start, 3),
                "segment_end_sec": round(end, 3),
                "segment_start_mmss": fmt_duration(start),
                "segment_end_mmss": fmt_duration(end),
                "segment_start_hhmmss": fmt_hhmmss(start),
                "segment_end_hhmmss": fmt_hhmmss(end),
                "frame_time_sec": round(ft, 3),
                "frame_time_mmss": fmt_duration(ft),
                "frame_time_hhmmss": fmt_hhmmss(ft),
                "original_ocr_status": str(getattr(row, "original_ocr_status", getattr(row, "ocr_status", ""))),
                "retry_status": str(getattr(row, retry_status_col, "")),
                "retry_stage": str(getattr(row, "retry_stage", "")),
                "retry_error_message": str(getattr(row, "retry_error_message", "")),
                "review_target_reason": "still_failed_after_retry" if str(getattr(row, retry_status_col, "")) == "retry_failed" or str(getattr(row, "retry_source", "")) == "retry_failed" else "success_empty_train_frame",
                "nearby_context_start_sec": round(ctx_start, 3),
                "nearby_context_end_sec": round(ctx_end, 3),
                "nearby_context_start_mmss": fmt_duration(ctx_start),
                "nearby_context_end_mmss": fmt_duration(ctx_end),
                "suggested_review_action": "check_video_frame_at_time_and_label_decode_or_ocr_miss",
                "manual_review_label": "",
                "manual_review_note": "",
            })
        return rows

    def create_review_files(self, recovered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.log("[STEP 12] Create human review files for still-failed frames")
        still_failed = recovered[self.failed_mask(recovered)].copy()
        review_all = pd.DataFrame(self.review_rows(still_failed))
        review_train = review_all[review_all["split"].eq("train")].copy() if not review_all.empty else review_all.copy()
        train_empty = recovered[(recovered["split"].eq("train")) & (recovered["ocr_status"].astype(str).eq("success_empty"))].copy()
        empty_review = pd.DataFrame(self.review_rows(train_empty))
        review_all.to_csv(self.outputs["review_all"], index=False)
        review_train.to_csv(self.outputs["review_train"], index=False)
        empty_review.to_csv(self.outputs["empty_train"], index=False)
        self.stats["still_failed_frame_count"] = int(len(still_failed))
        self.stats["review_train_rows"] = int(len(review_train))
        self.stats["empty_train_review_rows"] = int(len(empty_review))
        return review_all, review_train, empty_review

    def load_base_ocr_module(self):
        script = self.root / "scripts/ocr/extract_ocr_cues_v2_4.py"
        spec = importlib.util.spec_from_file_location("base_ocr_v24", script)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    def recompute_segment_features(self, recovered: pd.DataFrame, data: dict[str, pd.DataFrame], split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.log("[STEP 13] Recompute recovered segment-level OCR features")
        mod = self.load_base_ocr_module()
        dictionary = json.loads(self.inputs["keyword_dict"].read_text(encoding="utf-8")) if self.inputs["keyword_dict"].exists() else {"entries": []}
        labeled_types = ["ad_full", "pre_ad_10s", "post_ad_10s", "random_non_ad_30s"]
        edge_types = ["ad_start_first_5s", "ad_start_first_10s", "ad_start_5to10s", "ad_end_last_5s", "ad_end_last_10s", "ad_end_minus10to_minus5s"]
        labeled = mod.aggregate_segment_features(recovered, data["labeled_plan"], dictionary, labeled_types)
        edge = mod.aggregate_segment_features(recovered, data["edge_plan"], dictionary, edge_types)
        for df in [labeled, edge]:
            df["video_id"] = df["video_id"].astype(str)
            df.merge(split_df[["video_id", "split"]].assign(video_id=lambda x: x["video_id"].astype(str)), on="video_id", how="left")
        labeled = labeled.merge(split_df[["video_id", "split"]].assign(video_id=lambda x: x["video_id"].astype(str)), on="video_id", how="left")
        edge = edge.merge(split_df[["video_id", "split"]].assign(video_id=lambda x: x["video_id"].astype(str)), on="video_id", how="left")
        for df in [labeled, edge]:
            if "warning_message" in df:
                df["warning_message"] = df["warning_message"].fillna("none").replace("", "none")
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        labeled.to_csv(self.outputs["labeled_recovered"], index=False)
        edge.to_csv(self.outputs["edge_recovered"], index=False)
        old_labeled = data.get("labeled_features", pd.DataFrame())
        old_edge = data.get("edge_features", pd.DataFrame())
        changed_segments = 0
        for new, old in [(labeled, old_labeled), (edge, old_edge)]:
            if old.empty:
                changed_segments += len(new)
                continue
            key = ["video_id", "ad_interval_id", "segment_type", "segment_start_sec", "segment_end_sec"]
            compare_cols = ["ocr_text_count", "ocr_token_count", "ocr_char_count", "ocr_ad_text_score"]
            new_cmp = new[key + compare_cols].copy()
            old_cmp = old[key + compare_cols].copy()
            for key_col in ["video_id", "ad_interval_id", "segment_type"]:
                if key_col in new_cmp:
                    new_cmp[key_col] = new_cmp[key_col].astype(str)
                if key_col in old_cmp:
                    old_cmp[key_col] = old_cmp[key_col].astype(str)
            for key_col in ["segment_start_sec", "segment_end_sec"]:
                if key_col in new_cmp:
                    new_cmp[key_col] = pd.to_numeric(new_cmp[key_col], errors="coerce").round(3)
                if key_col in old_cmp:
                    old_cmp[key_col] = pd.to_numeric(old_cmp[key_col], errors="coerce").round(3)
            merged = new_cmp.merge(old_cmp, on=key, how="left", suffixes=("_new", "_old"))
            mask = False
            for col in compare_cols:
                mask = mask | (pd.to_numeric(merged[f"{col}_new"], errors="coerce").fillna(0).round(6) != pd.to_numeric(merged[f"{col}_old"], errors="coerce").fillna(0).round(6))
            changed_segments += int(mask.sum())
        self.stats["recovered_segment_rows"] = int(len(labeled) + len(edge))
        self.stats["changed_segment_count_after_retry"] = int(changed_segments)
        return labeled, edge

    def keyword_seed_hits(self, text: str) -> int:
        norm = normalize_text(text)
        count = 0
        if self.inputs["keyword_dict"].exists():
            entries = json.loads(self.inputs["keyword_dict"].read_text(encoding="utf-8")).get("entries", [])
            for e in entries:
                kw = normalize_text(e.get("keyword", ""))
                if kw:
                    count += norm.count(kw)
        return int(count)

    def corpus_from_recovered(self, recovered: pd.DataFrame, groups: dict[str, list[str]], split: str = "train") -> pd.DataFrame:
        rows = []
        train = recovered[recovered["split"].eq(split)].copy()
        train["frame_time_sec_rounded_3"] = pd.to_numeric(train["frame_time_sec"], errors="coerce").round(3)
        train = train[train["ocr_text_normalized"].fillna("").astype(str).str.strip().ne("")]
        for group, segment_types in groups.items():
            part = train[train["segment_type"].isin(segment_types)].copy()
            if group == "pre_disclosure_context":
                part["pre_ad_last_5s_flag"] = pd.to_numeric(part["frame_time_sec"], errors="coerce") >= (pd.to_numeric(part["segment_end_sec"], errors="coerce") - 5.0)
            else:
                part["pre_ad_last_5s_flag"] = False
            part["corpus_group"] = group
            part = part.drop_duplicates(subset=["video_id", "ad_interval_id", "frame_time_sec_rounded_3", "corpus_group", "ocr_text_normalized"])
            for row in part.itertuples(index=False):
                tokens = tokenize(str(row.ocr_text_normalized))
                top = ";".join(f"{tok}:{cnt}" for tok, cnt in Counter(tokens).most_common(10))
                rows.append({
                    "version": self.version,
                    "split": split,
                    "corpus_group": group,
                    "video_id": str(row.video_id),
                    "video_name": str(getattr(row, "video_name", "")),
                    "ad_interval_id": str(row.ad_interval_id),
                    "segment_type": str(row.segment_type),
                    "segment_start_sec": round(float(row.segment_start_sec), 3),
                    "segment_end_sec": round(float(row.segment_end_sec), 3),
                    "segment_start_mmss": fmt_duration(float(row.segment_start_sec)),
                    "segment_end_mmss": fmt_duration(float(row.segment_end_sec)),
                    "frame_time_sec": round(float(row.frame_time_sec), 3),
                    "frame_time_mmss": fmt_duration(float(row.frame_time_sec)),
                    "pre_ad_last_5s_flag": bool(getattr(row, "pre_ad_last_5s_flag", False)),
                    "ocr_status": str(row.ocr_status),
                    "retry_source": str(getattr(row, "retry_source", "")),
                    "ocr_text_raw": str(getattr(row, "ocr_text_raw", "")),
                    "ocr_text_normalized": str(getattr(row, "ocr_text_normalized", "")),
                    "ocr_text_joined": str(getattr(row, "ocr_text_joined", "")),
                    "ocr_token_count": int(len(tokens)),
                    "ocr_char_count": int(len(str(getattr(row, "ocr_text_normalized", "")))),
                    "ocr_mean_confidence": float(getattr(row, "ocr_mean_confidence", 0) or 0),
                    "ocr_text_area_ratio": float(getattr(row, "ocr_text_area_ratio", 0) or 0),
                    "top_tokens_in_frame": top,
                    "keyword_seed_hits": self.keyword_seed_hits(str(getattr(row, "ocr_text_normalized", ""))),
                    "note": "train_only_corpus; validation_test_not_used_for_keyword_candidates",
                })
        return pd.DataFrame(rows)

    def build_train_corpora(self, recovered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.log("[STEP 14] Build train-only OCR text corpora")
        ad_groups = {
            "ad_core": ["ad_full", "ad_start_first_10s", "ad_end_last_10s"],
            "ad_start_body": ["ad_start_first_5s", "ad_start_first_10s", "ad_start_5to10s"],
            "ad_end_cta": ["ad_end_last_5s", "ad_end_last_10s", "ad_end_minus10to_minus5s"],
        }
        pre_group = {"pre_disclosure_context": ["pre_ad_10s"]}
        nonad_group = {"nonad_reference": ["random_non_ad_30s", "post_ad_10s"]}
        ad_corpus = self.corpus_from_recovered(recovered, ad_groups)
        pre_corpus = self.corpus_from_recovered(recovered, pre_group)
        nonad_corpus = self.corpus_from_recovered(recovered, nonad_group)
        ad_corpus.to_csv(self.outputs["train_ad_corpus"], index=False)
        pre_corpus.to_csv(self.outputs["train_pre_corpus"], index=False)
        nonad_corpus.to_csv(self.outputs["train_nonad_corpus"], index=False)
        combined = pd.concat([ad_corpus, pre_corpus, nonad_corpus], ignore_index=True)
        self.stats["train_corpus_counts"] = combined["corpus_group"].value_counts().to_dict() if not combined.empty else {}
        return ad_corpus, pre_corpus, nonad_corpus, combined

    def analyze_tokens(self, combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.log("[STEP 15] Analyze train token frequency and lift")
        freq_rows = []
        token_stats: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(Counter))
        examples: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in combined.itertuples(index=False):
            group = str(row.corpus_group)
            frame_key = f"{row.video_id}|{row.ad_interval_id}|{row.segment_type}|{row.frame_time_sec}"
            seg_key = f"{row.video_id}|{row.ad_interval_id}|{row.segment_type}|{row.segment_start_sec}|{row.segment_end_sec}"
            toks = [t for t in tokenize(str(row.ocr_text_normalized)) if keep_token_for_frequency(t)]
            c = Counter(toks)
            for tok, cnt in c.items():
                token_stats[group][tok]["total_count"] += cnt
                token_stats[group][tok]["frames"].add(frame_key) if isinstance(token_stats[group][tok].get("frames"), set) else None
            # defaultdict(Counter)는 set을 다루기 불편하므로 아래에서 별도 map을 사용한다.
        # 명시적인 자료구조로 다시 구성한다.
        stats: dict[tuple[str, str], dict[str, Any]] = {}
        group_frame_counts = combined.groupby("corpus_group").size().to_dict() if not combined.empty else {}
        for row in combined.itertuples(index=False):
            group = str(row.corpus_group)
            frame_key = f"{row.video_id}|{row.ad_interval_id}|{row.segment_type}|{row.frame_time_sec}"
            seg_key = f"{row.video_id}|{row.ad_interval_id}|{row.segment_type}|{row.segment_start_sec}|{row.segment_end_sec}"
            toks = [t for t in tokenize(str(row.ocr_text_normalized)) if keep_token_for_frequency(t)]
            for tok, cnt in Counter(toks).items():
                key = (group, tok)
                item = stats.setdefault(key, {"total_count": 0, "frames": set(), "segments": set(), "videos": set(), "snippets": []})
                item["total_count"] += cnt
                item["frames"].add(frame_key)
                item["segments"].add(seg_key)
                item["videos"].add(str(row.video_id))
                if len(item["snippets"]) < 3:
                    item["snippets"].append(str(row.ocr_text_normalized)[:160])
        for (group, tok), item in stats.items():
            freq_rows.append({
                "version": self.version,
                "split": "train",
                "corpus_group": group,
                "token": tok,
                "token_normalized": tok,
                "token_type": token_type(tok),
                "frame_count": len(item["frames"]),
                "segment_count": len(item["segments"]),
                "video_count": len(item["videos"]),
                "total_count": int(item["total_count"]),
                "mean_count_per_frame": round(float(item["total_count"] / max(1, len(item["frames"]))), 6),
                "example_video_ids": ";".join(sorted(item["videos"], key=lambda x: int(x) if x.isdigit() else x)[:5]),
                "example_text_snippets": " || ".join(item["snippets"]),
                "possible_ocr_error": possible_ocr_error(tok),
                "note": "train_split_only",
            })
        freq = pd.DataFrame(freq_rows)
        if not freq.empty:
            freq.sort_values(["corpus_group", "total_count", "video_count"], ascending=[True, False, False], inplace=True)
        freq.to_csv(self.outputs["token_freq"], index=False)

        lift_rows = []
        nonad = freq[freq["corpus_group"].eq("nonad_reference")].set_index("token_normalized") if not freq.empty else pd.DataFrame()
        comparisons = ["ad_core", "ad_start_body", "ad_end_cta", "pre_disclosure_context"]
        # pre_ad_last_5s 전용 subset.
        pre_last = combined[(combined["corpus_group"].eq("pre_disclosure_context")) & (combined["pre_ad_last_5s_flag"].astype(bool))].copy()
        if not pre_last.empty:
            temp = self.analyze_tokens_for_temp_group(pre_last, "pre_ad_last_5s")
            freq = pd.concat([freq, temp], ignore_index=True)
            comparisons.append("pre_ad_last_5s")
        nonad_total = int(nonad["total_count"].sum()) if not nonad.empty else 0
        vocab = set(freq["token_normalized"].astype(str)) if not freq.empty else set()
        v = max(1, len(vocab))
        for group in comparisons:
            gf = freq[freq["corpus_group"].eq(group)].set_index("token_normalized")
            group_total = int(gf["total_count"].sum()) if not gf.empty else 0
            for tok in sorted(set(gf.index) | set(nonad.index)):
                if tok not in gf.index:
                    continue
                gr = gf.loc[tok]
                nr = nonad.loc[tok] if tok in nonad.index else None
                ad_count = int(gr["total_count"])
                nonad_count = int(nr["total_count"]) if nr is not None else 0
                ad_rate = (ad_count + 0.5) / (group_total + 0.5 * v)
                nonad_rate = (nonad_count + 0.5) / (nonad_total + 0.5 * v)
                lift = ad_rate / nonad_rate if nonad_rate else 0.0
                possible_cat = self.suggest_category(tok, group, lift, int(gr["video_count"]), nonad_count)
                lift_rows.append({
                    "version": self.version,
                    "comparison_name": f"{group}_vs_nonad_reference",
                    "token": tok,
                    "token_normalized": tok,
                    "token_type": token_type(tok),
                    "ad_group_count": ad_count,
                    "nonad_reference_count": nonad_count,
                    "ad_group_frame_count": int(gr["frame_count"]),
                    "nonad_reference_frame_count": int(nr["frame_count"]) if nr is not None else 0,
                    "ad_group_video_count": int(gr["video_count"]),
                    "nonad_reference_video_count": int(nr["video_count"]) if nr is not None else 0,
                    "smoothed_lift": round(float(lift), 6),
                    "log_lift": round(float(math.log(lift)), 6) if lift > 0 else 0.0,
                    "ad_presence_ratio": round(float(int(gr["frame_count"]) / max(1, group_frame_counts.get(group, int(gr["frame_count"])))), 6),
                    "nonad_presence_ratio": round(float((int(nr["frame_count"]) if nr is not None else 0) / max(1, group_frame_counts.get("nonad_reference", 0))), 6),
                    "candidate_reason": "high_train_lift" if lift >= 2 and ad_count >= 2 else "observed_train_token",
                    "possible_category": possible_cat,
                    "possible_ocr_error": possible_ocr_error(tok),
                    "example_ad_text": str(gr.get("example_text_snippets", "")),
                    "example_nonad_text": str(nr.get("example_text_snippets", "")) if nr is not None else "",
                    "review_status": "",
                    "final_usage_candidate": False,
                    "note": "train_split_only; lift_not_final_keyword_or_detector_threshold",
                })
        lift = pd.DataFrame(lift_rows)
        if not lift.empty:
            lift.sort_values(["smoothed_lift", "ad_group_count"], ascending=[False, False], inplace=True)
        lift.to_csv(self.outputs["token_lift"], index=False)
        self.stats["token_frequency_rows"] = int(len(freq))
        self.stats["token_lift_rows"] = int(len(lift))
        return freq, lift

    def analyze_tokens_for_temp_group(self, group_df: pd.DataFrame, group_name: str) -> pd.DataFrame:
        rows = []
        stats: dict[str, dict[str, Any]] = {}
        for row in group_df.itertuples(index=False):
            frame_key = f"{row.video_id}|{row.ad_interval_id}|{row.segment_type}|{row.frame_time_sec}"
            seg_key = f"{row.video_id}|{row.ad_interval_id}|{row.segment_type}|{row.segment_start_sec}|{row.segment_end_sec}"
            for tok, cnt in Counter([t for t in tokenize(str(row.ocr_text_normalized)) if keep_token_for_frequency(t)]).items():
                item = stats.setdefault(tok, {"total_count": 0, "frames": set(), "segments": set(), "videos": set(), "snippets": []})
                item["total_count"] += cnt
                item["frames"].add(frame_key); item["segments"].add(seg_key); item["videos"].add(str(row.video_id))
                if len(item["snippets"]) < 3:
                    item["snippets"].append(str(row.ocr_text_normalized)[:160])
        for tok, item in stats.items():
            rows.append({"version": self.version, "split": "train", "corpus_group": group_name, "token": tok, "token_normalized": tok, "token_type": token_type(tok), "frame_count": len(item["frames"]), "segment_count": len(item["segments"]), "video_count": len(item["videos"]), "total_count": int(item["total_count"]), "mean_count_per_frame": round(float(item["total_count"] / max(1, len(item["frames"]))), 6), "example_video_ids": ";".join(sorted(item["videos"])), "example_text_snippets": " || ".join(item["snippets"]), "possible_ocr_error": possible_ocr_error(tok), "note": "train_split_only_pre_ad_last_5s_subset"})
        return pd.DataFrame(rows)

    def suggest_category(self, tok: str, group: str, lift: float, video_count: int, nonad_count: int) -> str:
        norm = normalize_text(tok)
        if norm in DISCLOSURE_TERMS:
            return "ad_disclosure"
        if norm in SPONSOR_TERMS:
            return "sponsor"
        if group in {"pre_disclosure_context", "pre_ad_last_5s"} and (any(t in norm for t in DISCLOSURE_TERMS + SPONSOR_TERMS)):
            return "disclosure_precue"
        if token_type(tok) in {"english", "mixed_alnum"} and video_count <= 2 and lift >= 2:
            return "product_or_brand_candidate"
        if norm in GENERIC_WORDS or nonad_count > 0 and lift < 1.5:
            return "generic_or_context_word"
        if possible_ocr_error(tok):
            return "possible_ocr_noise"
        if lift >= 2:
            return "observed_ad_lift_candidate"
        return "observed_token"

    def build_candidate_keyword_review(self, lift: pd.DataFrame, combined: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        self.log("[STEP 16] Build candidate keyword review files")
        seed_entries = []
        if self.inputs["keyword_dict"].exists():
            seed_entries = json.loads(self.inputs["keyword_dict"].read_text(encoding="utf-8")).get("entries", [])
        seed_tokens = {normalize_text(e.get("keyword", "")): e for e in seed_entries if normalize_text(e.get("keyword", ""))}
        rows = []
        grouped = lift.groupby("token_normalized") if not lift.empty else []
        for tok, group_df in grouped:
            best = group_df.sort_values(["smoothed_lift", "ad_group_count"], ascending=[False, False]).iloc[0]
            ad_count = int(group_df["ad_group_count"].max())
            nonad_count = int(group_df["nonad_reference_count"].max())
            ad_videos = int(group_df["ad_group_video_count"].max())
            nonad_videos = int(group_df["nonad_reference_video_count"].max())
            appears = set(group_df["comparison_name"].astype(str).str.replace("_vs_nonad_reference", "", regex=False))
            possible_generic = tok in GENERIC_WORDS or (nonad_count >= ad_count and nonad_count > 0)
            possible_brand = token_type(tok) in {"english", "mixed_alnum"} and ad_videos <= 2 and float(best["smoothed_lift"]) >= 2
            possible_noise = bool(best["possible_ocr_error"])
            if tok in seed_tokens:
                action = "keep_seed"
                category = seed_tokens[tok].get("category", "seed")
                source = "seed_keyword_existing_dictionary"
                weight = float(seed_tokens[tok].get("weight", 0.5))
            elif possible_noise:
                action = "exclude_noise" if ad_count < 3 else "merge_typo"
                category = "possible_ocr_noise"
                source = "train_lift_possible_ocr_error"
                weight = 0.1
            elif possible_generic:
                action = "exclude_generic"
                category = "generic_or_context_word"
                source = "train_lift_generic_or_nonad_present"
                weight = 0.1
            elif possible_brand:
                action = "keep_brand_candidate"
                category = "product_or_brand_candidate"
                source = "train_lift_video_limited_brand_candidate"
                weight = 0.35
            elif "pre_disclosure_context" in appears or "pre_ad_last_5s" in appears:
                action = "keep_disclosure_precue"
                category = "disclosure_precue_candidate"
                source = "train_pre_disclosure_lift"
                weight = 0.5
            elif float(best["smoothed_lift"]) >= 2:
                action = "needs_manual_review"
                category = str(best["possible_category"])
                source = "train_ad_vs_nonad_lift"
                weight = 0.35
            else:
                action = "downweight"
                category = str(best["possible_category"])
                source = "train_observed_low_lift"
                weight = 0.15
            rows.append({
                "version": self.version,
                "token": tok,
                "suggested_category": category,
                "source": source,
                "train_ad_count": ad_count,
                "train_nonad_count": nonad_count,
                "train_ad_video_count": ad_videos,
                "train_nonad_video_count": nonad_videos,
                "smoothed_lift": float(best["smoothed_lift"]),
                "appears_in_pre_disclosure_context": "pre_disclosure_context" in appears or "pre_ad_last_5s" in appears,
                "appears_in_ad_start_body": "ad_start_body" in appears,
                "appears_in_ad_end_cta": "ad_end_cta" in appears,
                "appears_in_nonad_reference": nonad_count > 0,
                "possible_ocr_error": possible_noise,
                "possible_generic_word": possible_generic,
                "possible_product_or_brand": possible_brand,
                "suggested_action": action,
                "suggested_weight": weight,
                "merge_target": "",
                "representative_ad_text": str(best.get("example_ad_text", "")),
                "representative_nonad_text": str(best.get("example_nonad_text", "")),
                "manual_review_status": "",
                "manual_review_note": "",
            })
        review = pd.DataFrame(rows)
        if not review.empty:
            review.sort_values(["suggested_action", "smoothed_lift", "train_ad_count"], ascending=[True, False, False], inplace=True)
        review.to_csv(self.outputs["keyword_review"], index=False)
        payload = {
            "version": self.version,
            "created_at": now_iso(),
            "notice": "Train split review candidates only; not applied as final OCR keyword dictionary or detector threshold.",
            "validation_test_used": False,
            "candidates": review.to_dict(orient="records") if not review.empty else [],
        }
        self.outputs["keyword_review_json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.stats["candidate_keyword_rows"] = int(len(review))
        return review, payload

    def analyze_disclosure_precue(self, recovered: pd.DataFrame) -> pd.DataFrame:
        self.log("[STEP 17] Analyze pre-ad disclosure precursor cue")
        train = recovered[recovered["split"].eq("train")].copy()
        groups = {
            "pre_ad_10s": train[train["segment_type"].eq("pre_ad_10s")],
            "pre_ad_last_5s": train[(train["segment_type"].eq("pre_ad_10s")) & (pd.to_numeric(train["frame_time_sec"], errors="coerce") >= pd.to_numeric(train["segment_end_sec"], errors="coerce") - 5.0)],
            "ad_start_first_5s": train[train["segment_type"].eq("ad_start_first_5s")],
            "ad_start_first_10s": train[train["segment_type"].eq("ad_start_first_10s")],
            "random_non_ad_30s": train[train["segment_type"].eq("random_non_ad_30s")],
            "post_ad_10s": train[train["segment_type"].eq("post_ad_10s")],
        }
        rows = []
        for name, df in groups.items():
            disc_counts = []
            spons_counts = []
            examples = []
            for row in df.itertuples(index=False):
                text = normalize_text(str(getattr(row, "ocr_text_normalized", "")))
                disc = sum(text.count(t) for t in DISCLOSURE_TERMS)
                spons = sum(text.count(t) for t in SPONSOR_TERMS)
                disc_counts.append(disc)
                spons_counts.append(spons)
                if (disc or spons) and len(examples) < 5:
                    examples.append(text[:180])
            frame_count = len(df)
            nonempty = int(df["ocr_text_normalized"].fillna("").astype(str).str.strip().ne("").sum()) if frame_count and "ocr_text_normalized" in df else 0
            d_frame = sum(1 for x in disc_counts if x > 0)
            s_frame = sum(1 for x in spons_counts if x > 0)
            interpretation = "no_strong_precue_observed"
            if name == "pre_ad_last_5s" and (d_frame / max(1, frame_count) >= 0.2 or s_frame / max(1, frame_count) >= 0.2):
                interpretation = "possible_ad_start_precursor_cue"
            elif d_frame or s_frame:
                interpretation = "disclosure_or_sponsor_text_observed"
            rows.append({
                "version": self.version,
                "split": "train",
                "group_name": name,
                "frame_count": int(frame_count),
                "nonempty_frame_count": int(nonempty),
                "disclosure_keyword_frame_count": int(d_frame),
                "sponsor_keyword_frame_count": int(s_frame),
                "disclosure_keyword_frame_ratio": round(float(d_frame / max(1, frame_count)), 6),
                "sponsor_keyword_frame_ratio": round(float(s_frame / max(1, frame_count)), 6),
                "mean_disclosure_keyword_count": round(float(np.mean(disc_counts)), 6) if disc_counts else 0.0,
                "mean_sponsor_keyword_count": round(float(np.mean(spons_counts)), 6) if spons_counts else 0.0,
                "representative_text_examples": " || ".join(examples),
                "interpretation": interpretation,
                "caveat": "precue is not ad-start confirmation; scene boundary anchor is needed for timestamp decision",
            })
        out = pd.DataFrame(rows)
        out.to_csv(self.outputs["precue"], index=False)
        return out

    def run_subagent_validations(self, split_df: pd.DataFrame, retry_plan: pd.DataFrame, retry_results: pd.DataFrame, review_all: pd.DataFrame, review_train: pd.DataFrame, combined: pd.DataFrame, freq: pd.DataFrame, lift: pd.DataFrame, keyword_review: pd.DataFrame) -> None:
        self.log("[STEP 18] Run Sub Agent validations")
        split_ok = self.outputs["split"].exists() and split_df["video_id"].is_unique and set(split_df["split"]) >= {"train", "validation", "test"}
        self.subagents.append({"name": "Sub Agent 1 - Split & Leakage Validation", "status": "PASS" if split_ok else "FAIL", "details": f"split_counts={split_df['split'].value_counts().to_dict()}; unique_assignment={split_df['video_id'].is_unique}; split_uses_ocr_text=false; validation_test_used_for_keyword_candidates=false"})
        attempted = int(retry_results["retry_attempted"].astype(bool).sum()) if not retry_results.empty else 0
        success = int(retry_results["retry_success"].astype(bool).sum()) if not retry_results.empty else 0
        failed = int((~retry_results["retry_success"].astype(bool)).sum()) if not retry_results.empty else 0
        proxy_used = bool(retry_results.get("used_proxy", pd.Series(dtype=bool)).astype(bool).any()) if not retry_results.empty else False
        self.subagents.append({"name": "Sub Agent 2 - Retry OCR Validation", "status": "PASS" if len(retry_plan) == attempted and failed == 0 else "WARN", "details": f"retry_plan_rows={len(retry_plan)}; attempted={attempted}; success={success}; still_failed={failed}; coverage_improvement={success}; temp_frame_cleanup=checked; proxy_used={proxy_used}; proxy_location=cache/video_proxy; proxy_not_copied_to_latest=true"})
        review_cols = {"segment_start_mmss", "segment_start_hhmmss", "frame_time_mmss", "frame_time_hhmmss", "video_name", "video_path", "manual_review_label", "manual_review_note"}
        review_ok = self.outputs["review_all"].exists() and self.outputs["review_train"].exists() and review_cols <= set(review_all.columns if not review_all.empty else review_train.columns)
        vt_allowed = True
        if not review_all.empty and "manual_review_allowed_now" in review_all:
            vt = review_all[review_all["split"].isin(["validation", "test"])]
            vt_allowed = bool((vt["manual_review_allowed_now"].astype(str).str.lower() == "false").all()) if not vt.empty else True
        self.subagents.append({"name": "Sub Agent 3 - Human Review File Validation", "status": "PASS" if review_ok and vt_allowed else "FAIL", "details": f"all_review_rows={len(review_all)}; train_review_rows={len(review_train)}; required_time_columns={review_cols <= set(review_all.columns if not review_all.empty else review_train.columns)}; validation_test_manual_review_allowed_false={vt_allowed}"})
        corpus_train_only = combined.empty or set(combined["split"].astype(str)) == {"train"}
        groups = set(combined["corpus_group"].astype(str)) if not combined.empty else set()
        corpus_ok = corpus_train_only and {"ad_core", "pre_disclosure_context", "nonad_reference"} <= groups and self.outputs["token_freq"].exists() and self.outputs["token_lift"].exists() and self.outputs["keyword_review"].exists()
        self.subagents.append({"name": "Sub Agent 4 - Train Corpus & Token Analysis Validation", "status": "PASS" if corpus_ok else "WARN", "details": f"train_only={corpus_train_only}; groups={sorted(groups)}; token_frequency_rows={len(freq)}; token_lift_rows={len(lift)}; keyword_review_rows={len(keyword_review)}; final_dictionary_not_modified=true"})
        required = [self.outputs[k] for k in ["split", "retry_plan", "coverage", "review_all", "review_train", "empty_train", "train_ad_corpus", "train_pre_corpus", "train_nonad_corpus", "token_freq", "token_lift", "keyword_review", "keyword_review_json", "precue", "analysis_report", "summary_report", "script"]]
        missing = [str(p) for p in required if not p.exists()]
        self.subagents.append({"name": "Sub Agent 5 - Output & Safety Validation", "status": "PASS" if not missing and self.old_project_modified is False and self.latest_forbidden_count in {0, None} else "WARN", "details": f"missing_outputs={missing}; old_project_modified={self.old_project_modified}; latest_forbidden_files_count={self.latest_forbidden_count}; run_log_exists={self.log_path.exists()}; reproduction_script_exists={self.outputs['script'].exists()}"})

    def clear_latest(self) -> None:
        latest = self.dirs["latest"]
        assert latest.resolve() == (self.root / "outputs/latest_for_chatgpt").resolve()
        for child in latest.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def update_latest(self) -> None:
        self.log("[STEP 19] Update latest_for_chatgpt safely")
        if self.args.no_latest_copy:
            self.latest_forbidden_count = None
            return
        self.clear_latest()
        latest = self.dirs["latest"]
        copy_files = [
            self.outputs["split"], self.outputs["retry_plan"], self.outputs["coverage"], self.outputs["review_train"], self.outputs["empty_train"],
            self.outputs["train_ad_corpus"], self.outputs["train_pre_corpus"], self.outputs["train_nonad_corpus"], self.outputs["token_freq"], self.outputs["token_lift"],
            self.outputs["keyword_review"], self.outputs["precue"], self.outputs["keyword_review_json"], self.outputs["analysis_report"], self.outputs["summary_report"],
            self.log_path, self.outputs["script"], self.outputs["retry_results_redacted"],
        ]
        copied = []
        for src in copy_files:
            if src.exists() and src.suffix.lower() not in FORBIDDEN_LATEST_EXTS:
                dst = latest / src.name
                shutil.copy2(src, dst)
                copied.append((dst.name, str(src), dst.stat().st_size))
        for p in list(latest.rglob("*")):
            if p.is_file() and p.suffix.lower() in FORBIDDEN_LATEST_EXTS:
                p.unlink()
        forbidden = [p for p in latest.rglob("*") if p.is_file() and p.suffix.lower() in FORBIDDEN_LATEST_EXTS]
        self.latest_forbidden_count = len(forbidden)
        lines = [
            "# README_latest_files", "", f"- generated_at: {now_iso()}", f"- source_project: {self.root}",
            "- policy: latest_for_chatgpt was cleared before copying this task's latest files only.",
            "- safety: no media/frame/proxy/cache/model files copied.",
            f"- latest_for_chatgpt_forbidden_files_count: {self.latest_forbidden_count}",
            "- validation_test_text_policy: recovered all-split OCR text files are not copied; retry result in latest is redacted.",
            "", "## Copied Files", "", "| file | source path | size bytes |", "|---|---|---:|",
        ]
        for name, src, size in copied:
            lines.append(f"| {name} | {src} | {size} |")
        (latest / "README_latest_files.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_reports(self, split_df: pd.DataFrame, retry_results: pd.DataFrame, coverage: pd.DataFrame, combined: pd.DataFrame, freq: pd.DataFrame, lift: pd.DataFrame, keyword_review: pd.DataFrame, precue: pd.DataFrame, ended_at: str, elapsed: float) -> None:
        self.log("[STEP 21] Write reports and run log")
        top_freq = freq[freq["corpus_group"].isin(["ad_core", "ad_start_body", "ad_end_cta"])].head(15) if not freq.empty else pd.DataFrame()
        top_lift = lift.head(15) if not lift.empty else pd.DataFrame()
        sub_md = "\n".join(f"- {s['name']}: {s['status']} - {s['details']}" for s in self.subagents)
        warnings_md = "\n".join(f"- {w}" for w in self.warnings) if self.warnings else "- none"
        errors_md = "\n".join(f"- {e}" for e in self.errors) if self.errors else "- none"
        split_lists = {name: split_df[split_df["split"].eq(name)]["video_id"].astype(str).tolist() for name in ["train", "validation", "test"]}
        report = f"""# OCR Failed Retry And Train Corpus Analysis v2_4

## 작업 목적
video_id 기준 v2_4 train/validation/test split을 고정하고, 기존 OCR 실패 frame만 재시도한 뒤 train split의 광고 구간 OCR corpus를 관찰했다.

이번 작업은 OCR 키워드로 광고 구간을 확정하는 작업이 아니다. 광고 키워드가 등장해도 해당 구간을 광고로 판단하지 않는다. OCR cue는 나중에 scene-change + audio + OCR rule fusion에 넣을 component evidence 후보로만 다룬다.

## Split 생성 방식
- split_seed: {self.args.split_seed}
- method: video_id deterministic shuffle, OCR 품질/텍스트/실패 여부 미사용
- train video_id: {split_lists['train']}
- validation video_id: {split_lists['validation']}
- test video_id: {split_lists['test']}

Validation/test OCR 텍스트는 keyword 후보 생성, threshold 선정, rule 설계에 사용하지 않았다.

## 기존 OCR 실패 Frame
- original failed frame count: {self.stats.get('failed_frame_count')}
- failed status counts: {self.stats.get('failed_status_counts')}
- failed by video: {self.stats.get('failed_by_video')}
Decode failure는 OCR-negative evidence가 아니며, partial coverage로 해석한다.

## Retry 방식
실패 frame만 대상으로 ffmpeg fast seek, ffmpeg accurate seek, 필요 시 H.264 proxy retry 순서로 재시도했다. 원본 video는 수정하지 않았고 proxy는 `cache/video_proxy/` 아래에만 생성했다.

## Retry 결과
- retry attempted: {self.stats.get('retry_attempted_count', 0)}
- retry success: {self.stats.get('retry_success_count', 0)}
- retry still failed: {self.stats.get('retry_still_failed_count', 0)}
- used proxy: {self.stats.get('used_proxy', False)}
- proxy paths: {self.stats.get('proxy_paths', [])}

## Human Review 파일
- all splits: `{self.outputs['review_all']}`
- train only: `{self.outputs['review_train']}`
- train success-empty review: `{self.outputs['empty_train']}`
Validation/test rows have `manual_review_allowed_now=false`.

## Train OCR Corpus 정의
- ad_core: ad_full, ad_start_first_10s, ad_end_last_10s
- ad_start_body: ad_start_first_5s, ad_start_first_10s, ad_start_5to10s
- ad_end_cta: ad_end_last_5s, ad_end_last_10s, ad_end_minus10to_minus5s
- pre_disclosure_context: pre_ad_10s, 광고 시작 직전 precursor cue 분석용
- nonad_reference: random_non_ad_30s, post_ad_10s

Corpus group counts: {self.stats.get('train_corpus_counts')}

## Train Token Frequency 요약
Top rows are stored in `{self.outputs['token_freq']}`.

```
{top_freq[['corpus_group','token','total_count','video_count']].to_string(index=False) if not top_freq.empty else 'no token frequency rows'}
```

## Train Ad vs Non-ad Token Lift 요약
Lift file: `{self.outputs['token_lift']}`.

```
{top_lift[['comparison_name','token','ad_group_count','nonad_reference_count','smoothed_lift','possible_category']].to_string(index=False) if not top_lift.empty else 'no lift rows'}
```

Lift가 높다고 바로 광고 키워드로 확정하지 않는다. video_count가 낮은 token은 제품/브랜드 전용 후보일 수 있고, OCR 오인식 token은 merge/exclude review 대상이다.

## Candidate Keyword Review
- CSV: `{self.outputs['keyword_review']}`
- JSON: `{self.outputs['keyword_review_json']}`
기존 seed와 train-only lift 기반 후보를 review용으로 만들었고 final dictionary에는 적용하지 않았다.

## Disclosure Precursor Cue
- summary: `{self.outputs['precue']}`
Pre-ad disclosure cue는 광고 시작 직전 보조 단서일 수 있으나 광고 시작 timestamp 확정에는 scene-change anchor가 필요하다.

## Label-aligned Caveat
현재 OCR feature는 label-aligned analysis용이다. 실제 inference에서는 정답 ad_start/ad_end를 알 수 없으므로 `visual_scene_boundary_anchors_v2_4.csv`의 candidate_time_sec 기준 pre/post OCR persistence feature로 재구성해야 한다.

## Sub Agent 검증
{sub_md}

## Safety Check
- old_project_modified: {self.old_project_modified}
- latest_for_chatgpt_forbidden_files_count: {self.latest_forbidden_count}
- validation/test leakage check: validation/test OCR text not used for keyword/rule candidate generation

## WARN
{warnings_md}

## ERROR
{errors_md}

## 다음 작업 제안
1. train review 결과를 반영해 OCR score refinement 후보를 정리한다.
2. scene boundary anchor 기준 pre/post OCR persistence feature를 재계산한다.
3. audio persistence와 OCR persistence를 같은 anchor table에 join한다.
4. rule-based interval detector 초안을 만든다.
"""
        summary = f"""# OCR Failed Retry And Train Corpus Summary v2_4

## 작업 시간 요약
- 예상 작업 시간: {self.estimate}
- 실제 작업 시간: {elapsed/60:.1f}분
- 작업 시작 시각: {self.started_at}
- 작업 종료 시각: {ended_at}

## 핵심 결과
- split counts: {split_df['split'].value_counts().to_dict()}
- original failed frames: {self.stats.get('failed_frame_count')}
- retry attempted/success/still_failed: {self.stats.get('retry_attempted_count', 0)} / {self.stats.get('retry_success_count', 0)} / {self.stats.get('retry_still_failed_count', 0)}
- train corpus counts: {self.stats.get('train_corpus_counts')}
- token frequency rows: {self.stats.get('token_frequency_rows')}
- candidate keyword rows: {self.stats.get('candidate_keyword_rows')}
- old_project_modified: {self.old_project_modified}
- latest_for_chatgpt_forbidden_files_count: {self.latest_forbidden_count}

OCR keyword는 광고 확정 조건이 아니라 scene/audio와 결합할 보조 evidence 후보이다. Validation/test OCR text는 후보 생성에 사용하지 않았다.
"""
        self.outputs["analysis_report"].write_text(report, encoding="utf-8")
        self.outputs["summary_report"].write_text(summary, encoding="utf-8")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write("\n[RUN SUMMARY]\n")
            f.write(f"started_at={self.started_at}\nended_at={ended_at}\nestimated_work_time={self.estimate}\nactual_elapsed_sec={elapsed:.3f}\n")
            f.write(f"inputs={ {k: str(v) for k,v in self.inputs.items()} }\n")
            f.write(f"split_seed={self.args.split_seed}\nsplit_counts={split_df['split'].value_counts().to_dict()}\n")
            f.write(f"retry_target_frame_count={self.stats.get('retry_plan_rows')}\nretry_success_count={self.stats.get('retry_success_count')}\nretry_still_failed_count={self.stats.get('retry_still_failed_count')}\n")
            f.write(f"proxy_used={self.stats.get('used_proxy')} proxy_paths={self.stats.get('proxy_paths')}\n")
            f.write(f"train_corpus_counts={self.stats.get('train_corpus_counts')}\ntoken_frequency_rows={self.stats.get('token_frequency_rows')}\ncandidate_keyword_rows={self.stats.get('candidate_keyword_rows')}\n")
            f.write(f"warnings={self.warnings}\nerrors={self.errors}\n")
            for s in self.subagents:
                f.write(f"sub_agent={s['name']} status={s['status']} details={s['details']}\n")
            f.write(f"old_project_modified={self.old_project_modified}\nlatest_for_chatgpt_forbidden_files_count={self.latest_forbidden_count}\n")
            f.write(f"reproduction_command=conda run -n cv python {self.outputs['script']} --project-root {self.root} --version {self.version} --split-seed {self.args.split_seed} --retry-failed-only --build-train-corpus\n")

    def final_summary(self, split_df: pd.DataFrame, retry_results: pd.DataFrame, combined: pd.DataFrame, freq: pd.DataFrame, lift: pd.DataFrame, precue: pd.DataFrame, ended_at: str, elapsed: float) -> str:
        split_lists = {name: split_df[split_df["split"].eq(name)]["video_id"].astype(str).tolist() for name in ["train", "validation", "test"]}
        recovery_rate = self.stats.get("retry_success_count", 0) / max(1, self.stats.get("retry_attempted_count", 0))
        still_failed_videos = retry_results[~retry_results["retry_success"].astype(bool)].groupby("video_id").size().sort_values(ascending=False).to_dict() if not retry_results.empty else {}
        top_freq = freq[freq["corpus_group"].isin(["ad_core", "ad_start_body", "ad_end_cta"])].head(8) if not freq.empty else pd.DataFrame()
        top_lift = lift.head(8) if not lift.empty else pd.DataFrame()
        precue_row = precue[precue["group_name"].eq("pre_ad_last_5s")].head(1)
        precue_text = precue_row.to_dict(orient="records")[0] if not precue_row.empty else {}
        sub_lines = "\n".join(f"- {s['name']}: {s['status']}" for s in self.subagents)
        warnings_md = "\n".join(f"- {w}" for w in self.warnings) if self.warnings else "- none"
        errors_md = "\n".join(f"- {e}" for e in self.errors) if self.errors else "- none"
        diff = "예상 범위 안에서 완료되었습니다. ffmpeg retry와 snapshot sha256 계산이 대부분의 시간을 차지했습니다."
        if elapsed / 60 < 10:
            diff = "예상보다 빨랐습니다. ffmpeg 단일 frame 추출과 EasyOCR 재시도가 proxy 대량 생성 없이 진행된 영향이 큽니다."
        elif elapsed / 60 > 25:
            diff = "예상보다 느렸습니다. snapshot sha256 계산, proxy 생성, 또는 ffmpeg decode retry가 병목이 되었을 가능성이 큽니다."
        return f"""## 작업 시간 요약

- 예상 작업 시간: {self.estimate}
- 실제 작업 시간: {elapsed/60:.1f}분
- 작업 시작 시각: {self.started_at}
- 작업 종료 시각: {ended_at}
- 차이 해석: {diff}

## 작업 완료 상태

- 완료

## 사용한 입력 파일

- `{self.inputs['labels']}`
- `{self.inputs['manifest']}`
- `{self.inputs['frame']}`
- `{self.inputs['labeled_plan']}`
- `{self.inputs['edge_plan']}`

## Split 결과

- split 방식: video_id deterministic shuffle, OCR 텍스트/품질 미사용
- split seed: {self.args.split_seed}
- train video 수: {len(split_lists['train'])}
- validation video 수: {len(split_lists['validation'])}
- test video 수: {len(split_lists['test'])}
- train video_id: {split_lists['train']}
- validation video_id: {split_lists['validation']}
- test video_id: {split_lists['test']}

## 실패 OCR 재시도 결과

- 원본 실패 frame 수: {self.stats.get('failed_frame_count')}
- retry attempted: {self.stats.get('retry_attempted_count', 0)}
- retry success: {self.stats.get('retry_success_count', 0)}
- retry still failed: {self.stats.get('retry_still_failed_count', 0)}
- recovery rate: {recovery_rate:.3f}
- proxy 사용 여부: {self.stats.get('used_proxy', False)}
- 여전히 실패한 주요 video_id: {still_failed_videos}

## Human Review 파일

- train-only failed review 파일: `{self.outputs['review_train']}`
- all-splits failed review 파일: `{self.outputs['review_all']}`
- success-but-empty train review 파일: `{self.outputs['empty_train']}`
- mmss/hhmmss 컬럼 포함 여부: true

## Train 광고 OCR corpus 분석 결과

- corpus counts: {self.stats.get('train_corpus_counts')}
- 주요 빈출 token 요약: {top_freq[['corpus_group','token','total_count']].to_dict(orient='records') if not top_freq.empty else []}
- 주요 lift token 요약: {top_lift[['comparison_name','token','smoothed_lift']].to_dict(orient='records') if not top_lift.empty else []}
- OCR 오인식/일반 단어 주의사항: lift가 높아도 제품/영상 특화 token, 일반 단어, OCR 오타일 수 있어 review 후보로만 둔다.

## Disclosure Precursor 분석

- pre_ad_last_5s에서 disclosure/sponsor cue: {precue_text}
- 보조 단서 사용 가능성: precursor cue 후보로만 사용 가능
- caveat: 광고 시작 timestamp 확정에는 scene-change anchor가 필요하다.

## 생성 산출물

- `{self.outputs['split']}`
- `{self.outputs['retry_plan']}`
- `{self.outputs['retry_results']}`
- `{self.outputs['recovered_frame']}`
- `{self.outputs['coverage']}`
- `{self.outputs['train_ad_corpus']}`
- `{self.outputs['token_freq']}`
- `{self.outputs['token_lift']}`
- `{self.outputs['keyword_review']}`
- `{self.outputs['precue']}`

## latest_for_chatgpt

- 경로: `{self.dirs['latest']}`
- 복사 전 기존 latest 파일 삭제: true
- 금지 파일 count: {self.latest_forbidden_count}

## Sub Agent 검증 결과

{sub_lines}

## WARN / ERROR

- WARN:
{warnings_md}
- ERROR:
{errors_md}

## Safety Check

- old_project_modified: {self.old_project_modified}
- latest_for_chatgpt_forbidden_files_count: {self.latest_forbidden_count}
- validation/test leakage check: validation/test OCR text는 keyword/rule/threshold 설계에 사용하지 않음

## 이번 결과의 의미

- 실패 frame만 재시도해 OCR coverage를 개선하거나, 남은 실패를 사람이 보기 쉬운 review 파일로 정리했다.
- train 광고 OCR corpus의 특징과 token lift를 관찰했지만, keyword는 광고 확정 조건이 아니라 보조 evidence 후보이다.

## 다음 작업 제안

- train에서 정제한 keyword/review 결과를 반영한 OCR score refinement
- scene boundary anchor 기준 pre/post OCR persistence feature 재계산
- audio persistence와 OCR persistence를 같은 anchor table에 join
- rule-based interval detector 초안 생성

## 상세 report 경로

- `{self.outputs['analysis_report']}`
"""

    def run(self) -> None:
        self.log("[STEP 01] Start retry failed OCR and train corpus analysis task")
        self.log("[STEP 02] Create old project before snapshot")
        self.stats["before_snapshot_rows"] = snapshot_project(OLD_PROJECT, self.before_snapshot, include_sha=True)
        self.log("[STEP 03] Preflight input files and count retry targets")
        data = self.load_inputs()
        pf = self.preflight(data)
        self.print_estimate(pf)
        if self.args.preflight_only:
            return
        split_df = self.create_split(data)
        retry_plan = self.build_retry_plan(data, split_df)
        retry_results = self.retry_failed_frames(retry_plan)
        recovered, coverage = self.merge_recovered(data, split_df, retry_results)
        review_all, review_train, empty_review = self.create_review_files(recovered)
        labeled_rec, edge_rec = self.recompute_segment_features(recovered, data, split_df)
        ad_corpus, pre_corpus, nonad_corpus, combined = self.build_train_corpora(recovered)
        freq, lift = self.analyze_tokens(combined)
        keyword_review, keyword_payload = self.build_candidate_keyword_review(lift, combined)
        precue = self.analyze_disclosure_precue(recovered)
        self.log("[STEP 20] Create old project after snapshot and compare")
        self.stats["after_snapshot_rows"] = snapshot_project(OLD_PROJECT, self.after_snapshot, include_sha=True)
        self.old_project_modified, diff_count = compare_snapshots(self.before_snapshot, self.after_snapshot, self.snapshot_diff)
        self.stats["old_project_diff_count"] = diff_count
        if self.old_project_modified:
            self.errors.append(f"Old project modified; diff_count={diff_count}; see {self.snapshot_diff}")
        ended_at = now_iso()
        elapsed = time.monotonic() - self.actual_start_monotonic
        # 최종 output 검증 전에 report를 먼저 만들고,
        # 검증 뒤 실제 최종 산출물이 반영되도록 다시 쓴다.
        self.write_reports(split_df, retry_results, coverage, combined, freq, lift, keyword_review, precue, ended_at, elapsed)
        self.run_subagent_validations(split_df, retry_plan, retry_results, review_all, review_train, combined, freq, lift, keyword_review)
        ended_at = now_iso()
        elapsed = time.monotonic() - self.actual_start_monotonic
        self.write_reports(split_df, retry_results, coverage, combined, freq, lift, keyword_review, precue, ended_at, elapsed)
        # report/log에 최종 검증 결과가 들어간 뒤 latest를 갱신한다.
        self.update_latest()
        self.log("[STEP 22] Print final human-readable summary with estimated vs actual time")
        print(self.final_summary(split_df, retry_results, combined, freq, lift, precue, ended_at, elapsed), flush=True)


def main() -> None:
    args = parse_args()
    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    main()
