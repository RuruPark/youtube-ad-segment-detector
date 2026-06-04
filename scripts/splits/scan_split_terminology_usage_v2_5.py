#!/usr/bin/env python3
"""Scan project usage of legacy split terminology.

The scan is intentionally conservative: it records occurrences and classifies
where they live, but it does not patch historical reports, outputs, backups, or
binary/media/model/cache files.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(".")
USAGE_INVENTORY_CSV = PROJECT_ROOT / "reports/splits/split_terminology_v2_5_usage_inventory.csv"
TERMS = [
    "validation/test",
    "test row-level",
    "train_val",
    "train_only",
    "video_split_v2_4",
    "split_role",
    "holdout",
    "validation",
    "train",
    "test",
    "val",
]
COLUMNS = [
    "file_path",
    "file_type",
    "matched_term",
    "line_number",
    "line_excerpt",
    "classification",
    "recommended_action",
    "patch_status",
    "note",
]
EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "cache",
    "model_cache",
    "models",
    "external",
    "data/frames",
    "data/raw",
    "data/videos",
    "cache/ocr",
    "cache/video_proxy",
}
BINARY_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".m4a",
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
    ".pt", ".pth", ".ckpt", ".onnx", ".pkl", ".pickle",
    ".xlsx", ".xls", ".parquet", ".sqlite", ".db", ".zip", ".tar", ".gz",
}
MAX_SCAN_BYTES = 2_000_000
MAX_ACTIVE_MATCHES_PER_FILE = 40
MAX_HISTORICAL_MATCHES_PER_FILE = 5
TERM_PATTERNS = [(term, re.compile(re.escape(term), re.IGNORECASE)) for term in TERMS]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def is_excluded(path: Path) -> bool:
    relative = rel(path)
    parts = set(Path(relative).parts)
    if path.suffix.lower() in BINARY_SUFFIXES:
        return True
    for excluded in EXCLUDED_DIRS:
        excluded_parts = tuple(Path(excluded).parts)
        rel_parts = Path(relative).parts
        if rel_parts[: len(excluded_parts)] == excluded_parts:
            return True
    return bool(parts & {"frame_images", "raw_video", "proxy", "checkpoint", "checkpoints"})


def classify(path: Path) -> tuple[str, str, str]:
    relative = rel(path)
    parts = Path(relative).parts
    if is_excluded(path):
        return "unsafe_or_binary", "skip", "excluded media/cache/model/binary path"
    if parts and parts[0] == "backups":
        return "backup", "skip", "backup path is never patched"
    if parts and parts[0] == "outputs":
        if len(parts) > 1 and parts[1].startswith("latest_"):
            return "latest_bundle", "skip", "latest bundle is historical copy only"
        return "historical_output", "leave_historical", "outputs are historical artifacts"
    if parts and parts[0] == "reports":
        if len(parts) > 1 and parts[1] == "splits":
            return "active_doc", "add_reference_note_only", "split reports are maintained by this adoption task"
        return "historical_report", "leave_historical", "past reports preserve reproducibility"
    if parts and parts[0] == "scripts":
        return "active_source", "add_reference_note_only", "source logic keeps legacy split compatibility"
    if parts and parts[0] == "src":
        return "active_source", "add_reference_note_only", "source logic keeps legacy split compatibility"
    if parts and parts[0] == "configs":
        return "active_config", "add_reference_note_only", "config comments may reference v2.5 terminology"
    if parts and parts[0] == "docs":
        return "active_doc", "add_reference_note_only", "docs can reference v2.5 terminology"
    if relative == "README.md" or relative.startswith("README"):
        return "active_doc", "patch_now", "root README can safely link to v2.5 terminology docs"
    if parts and parts[0] == "data":
        return "historical_output", "leave_historical", "data artifacts keep original columns and names"
    return "active_doc", "add_reference_note_only", "unclassified text file; note only"


def file_type(path: Path) -> str:
    return path.suffix.lower().lstrip(".") or "no_ext"


def scan_project() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        classification, recommended, note = classify(path)
        if classification == "unsafe_or_binary":
            rows.append({
                "file_path": rel(path),
                "file_type": file_type(path),
                "matched_term": "",
                "line_number": "",
                "line_excerpt": "",
                "classification": classification,
                "recommended_action": recommended,
                "patch_status": "skipped",
                "note": note,
            })
            continue
        try:
            if path.stat().st_size > MAX_SCAN_BYTES:
                rows.append({
                    "file_path": rel(path),
                    "file_type": file_type(path),
                    "matched_term": "",
                    "line_number": "",
                    "line_excerpt": "",
                    "classification": classification,
                    "recommended_action": "skip",
                    "patch_status": "skipped",
                    "note": "text-like file skipped because it exceeds scan size limit",
                })
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            rows.append({
                "file_path": rel(path),
                "file_type": file_type(path),
                "matched_term": "",
                "line_number": "",
                "line_excerpt": "",
                "classification": classification,
                "recommended_action": "skip",
                "patch_status": "skipped",
                "note": f"read failed: {exc}",
            })
            continue
        match_count = 0
        max_matches = MAX_ACTIVE_MATCHES_PER_FILE
        if classification in {"historical_report", "historical_output", "latest_bundle", "backup"}:
            max_matches = MAX_HISTORICAL_MATCHES_PER_FILE
        for line_number, line in enumerate(text.splitlines(), start=1):
            if match_count >= max_matches:
                rows.append({
                    "file_path": rel(path),
                    "file_type": file_type(path),
                    "matched_term": "__match_cap__",
                    "line_number": line_number,
                    "line_excerpt": "",
                    "classification": classification,
                    "recommended_action": recommended,
                    "patch_status": "skipped",
                    "note": f"match cap reached for this file: {max_matches}",
                })
                break
            for term, pattern in TERM_PATTERNS:
                if pattern.search(line):
                    rows.append({
                        "file_path": rel(path),
                        "file_type": file_type(path),
                        "matched_term": term,
                        "line_number": line_number,
                        "line_excerpt": line.strip()[:240],
                        "classification": classification,
                        "recommended_action": recommended,
                        "patch_status": "not_patched",
                        "note": note,
                    })
                    match_count += 1
                    break
    return rows


def write_inventory(rows: list[dict[str, Any]], output_path: Path = USAGE_INVENTORY_CSV) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", quoting=csv.QUOTE_ALL, escapechar="\\")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def main() -> int:
    rows = scan_project()
    write_inventory(rows)
    print(f"wrote {USAGE_INVENTORY_CSV} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
