#!/usr/bin/env python3
"""규칙 검토에 필요한 train/validation discussion 파일을 모은다.

This script does not design rules, build intervals, or create predictions. It
only finds existing v2_4 discussion artifacts, validates split safety, and
copies allowed train/validation discussion files into the current bundle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_NAME = "collect_latest_rule_discussion_files_v2_4"
VERSION = "v2_4"
SPLIT_SEED = "20240524"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_PROJECT_ROOT = Path("./_old_project_not_included")

BUNDLE_PATH = PROJECT_ROOT / "outputs" / "latest_for_chatgpt_rule_discussion_current"
REPORT_PATH = PROJECT_ROOT / "reports" / "fusion" / f"{TASK_NAME}_report.json"
LOG_PATH = PROJECT_ROOT / "logs" / f"{TASK_NAME}_run_log.txt"
BACKUP_ROOT = PROJECT_ROOT / "backups"
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "video_split_v2_4.csv"

TRAIN_IDS = {1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15}
VALIDATION_IDS = {3, 7, 18}
TEST_IDS = {4, 16, 17}
TRAIN_VAL_SPLITS = {"train", "validation"}

FORBIDDEN_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".wav",
    ".mp3",
    ".m4a",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
}
FORBIDDEN_PATH_TOKENS = {
    "cache",
    "tmp",
    "temp",
    "frames",
    "frame_images",
    "raw",
    "videos",
    "checkpoints",
}
LARGE_DETAIL_LIMIT_BYTES = 2_000_000


@dataclass(frozen=True)
class Candidate:
    filename: str
    path: Path
    stage: str
    search_rank: int
    mtime_ns: int
    size: int


@dataclass
class BundleFile:
    filename: str
    source_path: str
    role: str
    why_needed: str
    copied_path: str
    generated_train_val_copy: bool = False


class Collector:
    def __init__(self) -> None:
        self.start_time = now_iso()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_lines: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.files_missing: list[dict[str, str]] = []
        self.files_excluded_due_to_size: list[dict[str, Any]] = []
        self.files_excluded_due_to_test_rows: list[dict[str, Any]] = []
        self.files_included: list[BundleFile] = []
        self.row_counts_by_file: dict[str, int | None] = {}
        self.split_counts_by_file: dict[str, dict[str, int]] = {}
        self.key_columns_check: dict[str, dict[str, Any]] = {}
        self.input_metadata_before: dict[str, dict[str, int]] = {}
        self.input_metadata_after: dict[str, dict[str, int]] = {}
        self.candidates: dict[str, list[Candidate]] = {}
        self.selected_stage = "incomplete"
        self.old_snapshot_before: dict[str, Any] | None = None
        self.old_snapshot_after: dict[str, Any] | None = None
        self.old_project_modified = False
        self.input_files_modified = False
        self.latest_for_chatgpt_forbidden_files_found: list[str] = []
        self.safety_check: dict[str, Any] = {}
        self.leakage_guard_check: dict[str, bool] = {
            "train_only_level_thresholds_preserved": True,
            "validation_used_for_discussion_only": True,
            "test_row_level_features_copied_to_bundle": False,
            "label_columns_used_for_support": False,
            "audit_columns_used_for_support": False,
        }
        self.preexisting_outputs_backed_up: list[str] = []
        self.excluded_reference_paths: list[dict[str, Any]] = []

    def run(self) -> None:
        self.step("STEP 01", "Start and create old project snapshot")
        ensure_parent_dirs()
        self.backup_preexisting_report_or_log()
        self.old_snapshot_before = snapshot_project(OLD_PROJECT_ROOT)

        self.step("STEP 02", "Search latest semantic cleanup outputs")
        self.search_stage("semantic_cleanup")

        self.step("STEP 03", "Search OCR visual-anchor outputs")
        self.search_stage("ocr_visual_anchor")

        self.step("STEP 04", "Search audio visual-anchor outputs")
        self.search_stage("audio_visual_anchor")
        self.search_reference_files()
        self.validate_fixed_split()

        self.step("STEP 05", "Select latest available discussion stage")
        self.selected_stage = self.select_stage()
        self.log(f"selected_stage={self.selected_stage}")

        self.step("STEP 06", "Validate split/test row exclusion")
        selected = self.select_bundle_files()
        self.record_input_metadata(selected)

        self.step("STEP 07", "Copy allowed files to bundle")
        self.prepare_bundle()
        for item in selected:
            self.copy_allowed_file(item)
        self.scan_excluded_large_detail_files()
        self.scan_bundle_forbidden_files()

        self.step("STEP 08", "Prepare final safety checks")
        self.input_metadata_after = collect_file_metadata(
            [Path(item.source_path) for item in self.files_included]
        )
        self.input_files_modified = self.input_metadata_before != self.input_metadata_after
        self.old_snapshot_after = snapshot_project(OLD_PROJECT_ROOT)
        self.old_project_modified = not snapshots_match(
            self.old_snapshot_before, self.old_snapshot_after
        )
        self.update_safety_check()

        self.step("STEP 09", "Write README/report/log")
        self.write_readme()
        self.write_report()
        self.write_log()

        self.step("STEP 10", "Validate safety and print final summary")
        self.write_log()
        self.print_final_summary()

    def step(self, code: str, message: str) -> None:
        self.log(f"[{code}] {message}")

    def log(self, message: str) -> None:
        self.log_lines.append(f"{now_iso()} {message}")

    def backup_preexisting_report_or_log(self) -> None:
        existing = [p for p in (REPORT_PATH, LOG_PATH) if p.exists()]
        if not existing:
            return
        backup_dir = BACKUP_ROOT / f"{TASK_NAME}_existing_outputs_{self.timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        for path in existing:
            rel = path.relative_to(PROJECT_ROOT)
            target = backup_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            self.preexisting_outputs_backed_up.append(str(target))
        self.log(f"Backed up preexisting report/log files to {backup_dir}")

    def search_stage(self, stage: str) -> None:
        config = STAGE_CONFIGS[stage]
        for filename in config["files"]:
            found = find_candidates(filename, config["dirs"], stage)
            if found:
                self.candidates.setdefault(filename, []).extend(found)
                best = choose_best(found)
                self.log(f"found {filename}: {best.path}")
            else:
                self.log(f"missing {filename} in {stage}")

    def search_reference_files(self) -> None:
        for filename, dirs in REFERENCE_FILE_SEARCH.items():
            found = find_candidates(filename, dirs, "reference")
            if found:
                self.candidates.setdefault(filename, []).extend(found)
                self.log(f"found reference {filename}: {choose_best(found).path}")
            else:
                self.log(f"missing optional reference {filename}")

    def validate_fixed_split(self) -> None:
        if not SPLIT_PATH.exists():
            self.errors.append(f"Missing fixed split file: {SPLIT_PATH}")
            return
        try:
            rows = read_dict_rows(SPLIT_PATH)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"Failed to read split file {SPLIT_PATH}: {exc}")
            return
        split_seed_values = {row.get("split_seed", "") for row in rows}
        ids_by_split: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
        for row in rows:
            split = row.get("split", "")
            try:
                video_id = int(str(row.get("video_id", "")).strip())
            except ValueError:
                self.warnings.append(f"Non-integer video_id in split file: {row.get('video_id')}")
                continue
            ids_by_split.setdefault(split, set()).add(video_id)
        if split_seed_values != {SPLIT_SEED}:
            self.warnings.append(f"Unexpected split_seed values: {sorted(split_seed_values)}")
        expected = {
            "train": TRAIN_IDS,
            "validation": VALIDATION_IDS,
            "test": TEST_IDS,
        }
        for split, expected_ids in expected.items():
            actual = ids_by_split.get(split, set())
            if actual != expected_ids:
                self.errors.append(
                    f"Fixed split mismatch for {split}: actual={sorted(actual)} expected={sorted(expected_ids)}"
                )
        self.log(
            "fixed split check: "
            + json.dumps({k: sorted(v) for k, v in ids_by_split.items()}, ensure_ascii=False)
        )

    def select_stage(self) -> str:
        if self.has_any(
            [
                "scene_audio_ocr_rule_discussion_compact_v2_4_train_val.csv",
                "scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val_semantic_fixed.csv",
            ]
        ):
            return "semantic_cleanup_ready"
        if self.has_any(
            [
                "scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val_with_ocr.csv",
                "ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv",
            ]
        ):
            return "ocr_visual_anchor_ready"
        if self.has_any(
            [
                "scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val.csv",
                "audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv",
            ]
        ):
            return "audio_visual_anchor_ready_only"
        return "incomplete"

    def has_any(self, filenames: list[str]) -> bool:
        return any(self.best_candidate(name) for name in filenames)

    def select_bundle_files(self) -> list[dict[str, str]]:
        selected: list[dict[str, str]] = []

        discussion_choice = self.first_available(
            [
                "scene_audio_ocr_rule_discussion_compact_v2_4_train_val.csv",
                "scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val_semantic_fixed.csv",
                "scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val_with_ocr.csv",
                "scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val.csv",
            ],
            required=True,
            role="core_discussion_table",
            why_needed="Primary compact table for continuing state-machine rule discussion.",
        )
        selected.extend(discussion_choice)

        core_specs = [
            (
                "ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv",
                True,
                "core_ocr_context_feature",
                "OCR context evidence around visual transition anchors.",
            ),
            (
                "ocr_visual_anchor_level_thresholds_v2_4_train_only.csv",
                True,
                "core_ocr_train_only_thresholds",
                "Train-only OCR level threshold reference; not derived from validation/test.",
            ),
            (
                "ocr_visual_anchor_coverage_reliability_summary_v2_4_train_val.csv",
                False,
                "core_ocr_coverage_reliability_summary",
                "Coverage/reliability summary needed to avoid treating missing OCR as negative evidence.",
            ),
            (
                "audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv",
                True,
                "core_audio_context_feature",
                "Audio persistence/context evidence around visual transition anchors.",
            ),
            (
                "audio_visual_anchor_level_thresholds_v2_4_train_only.csv",
                True,
                "core_audio_train_only_thresholds",
                "Train-only audio level threshold reference; not derived from validation/test.",
            ),
            (
                "visual_anchor_semantic_cleanup_status_v2_4.csv",
                False,
                "core_semantic_cleanup_status",
                "Semantic cleanup readiness/status checks.",
            ),
            (
                "ocr_visual_anchor_alignment_status_v2_4.csv",
                False,
                "core_ocr_alignment_status",
                "OCR visual-anchor alignment status checks.",
            ),
        ]
        for filename, required, role, why_needed in core_specs:
            selected.extend(
                self.single_file(filename, required=required, role=role, why_needed=why_needed)
            )

        selected.extend(
            self.first_available(
                [
                    "scene_audio_ocr_semantic_cleanup_v2_4_summary.md",
                    "ocr_visual_anchor_context_features_v2_4_summary.md",
                    "visual_anchor_alignment_pack_v2_4_summary.md",
                ],
                required=True,
                role="latest_summary_markdown",
                why_needed="Most recent summary of the selected discussion stage.",
            )
        )
        selected.extend(
            self.first_available(
                [
                    "scene_audio_ocr_semantic_cleanup_v2_4_report.json",
                    "ocr_visual_anchor_context_features_v2_4_report.json",
                    "visual_anchor_alignment_pack_v2_4_report.json",
                ],
                required=True,
                role="latest_report_json",
                why_needed="Machine-readable report for provenance and counts.",
            )
        )
        selected.extend(
            self.first_available(
                [
                    "scene_audio_ocr_semantic_cleanup_v2_4_run_log.txt",
                    "ocr_visual_anchor_context_features_v2_4_run_log.txt",
                    "visual_anchor_alignment_pack_v2_4_run_log.txt",
                ],
                required=True,
                role="latest_run_log",
                why_needed="Run log for the selected latest stage.",
            )
        )
        selected.extend(
            self.first_available(
                [
                    "scene_audio_ocr_semantic_cleanup_v2_4.py",
                    "extract_ocr_visual_anchor_context_features_v2_4.py",
                    "align_visual_anchor_audio_ocr_evidence_v2_4.py",
                ],
                required=True,
                role="latest_repro_script",
                why_needed="Script that produced or supports the selected latest artifacts.",
            )
        )
        for filename, role, why_needed in REFERENCE_FILES:
            selected.extend(
                self.single_file(
                    filename,
                    required=False,
                    role=role,
                    why_needed=why_needed,
                    reference=True,
                )
            )

        deduped: list[dict[str, str]] = []
        seen_dest_names: set[str] = set()
        for item in selected:
            if item["filename"] in seen_dest_names:
                continue
            seen_dest_names.add(item["filename"])
            deduped.append(item)
        return deduped

    def single_file(
        self,
        filename: str,
        *,
        required: bool,
        role: str,
        why_needed: str,
        reference: bool = False,
    ) -> list[dict[str, str]]:
        candidate = self.best_candidate(filename)
        if not candidate:
            if required:
                self.files_missing.append(
                    {
                        "filename": filename,
                        "role": role,
                        "reason": "No candidate found in priority paths.",
                    }
                )
            return []
        if reference and candidate.size > LARGE_DETAIL_LIMIT_BYTES:
            self.excluded_reference_paths.append(
                {
                    "filename": filename,
                    "source_path": str(candidate.path),
                    "size_bytes": candidate.size,
                    "reason": "Optional reference file exceeds small-reference limit.",
                }
            )
            return []
        return [
            {
                "filename": filename,
                "source_path": str(candidate.path),
                "role": role,
                "why_needed": why_needed,
            }
        ]

    def first_available(
        self,
        filenames: list[str],
        *,
        required: bool,
        role: str,
        why_needed: str,
    ) -> list[dict[str, str]]:
        for filename in filenames:
            candidate = self.best_candidate(filename)
            if candidate:
                return [
                    {
                        "filename": filename,
                        "source_path": str(candidate.path),
                        "role": role,
                        "why_needed": why_needed,
                    }
                ]
        if required:
            self.files_missing.append(
                {
                    "filename": " OR ".join(filenames),
                    "role": role,
                    "reason": "None of the fallback candidates were found.",
                }
            )
        return []

    def best_candidate(self, filename: str) -> Candidate | None:
        candidates = self.candidates.get(filename, [])
        if not candidates:
            return None
        return choose_best(candidates)

    def record_input_metadata(self, selected: list[dict[str, str]]) -> None:
        paths = [Path(item["source_path"]) for item in selected]
        self.input_metadata_before = collect_file_metadata(paths)

    def prepare_bundle(self) -> None:
        BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if BUNDLE_PATH.exists():
            backup_path = BACKUP_ROOT / f"latest_for_chatgpt_rule_discussion_current_{self.timestamp}"
            if backup_path.exists():
                raise FileExistsError(f"Backup path already exists: {backup_path}")
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(BUNDLE_PATH), str(backup_path))
            self.preexisting_outputs_backed_up.append(str(backup_path))
            self.log(f"Backed up existing bundle to {backup_path}")
        BUNDLE_PATH.mkdir(parents=True, exist_ok=False)

    def copy_allowed_file(self, item: dict[str, str]) -> None:
        src = Path(item["source_path"])
        dest = BUNDLE_PATH / item["filename"]
        if not src.exists():
            self.files_missing.append(
                {
                    "filename": item["filename"],
                    "role": item["role"],
                    "reason": f"Selected source disappeared before copy: {src}",
                }
            )
            return
        if is_forbidden_path(src):
            self.warnings.append(f"Forbidden source path skipped: {src}")
            return

        generated_train_val_copy = False
        if src.suffix.lower() == ".csv":
            validation = self.validate_csv(src, item["role"])
            split_counts = validation.get("split_counts", {})
            test_count = split_counts.get("test", 0)
            if test_count:
                if is_row_level_discussion_csv(src.name, item["role"]):
                    generated_train_val_copy = True
                    self.files_excluded_due_to_test_rows.append(
                        {
                            "filename": src.name,
                            "source_path": str(src),
                            "test_rows": test_count,
                            "action": "Generated train/validation-only copy in bundle.",
                        }
                    )
                    write_train_val_only_csv(src, dest)
                else:
                    self.files_excluded_due_to_test_rows.append(
                        {
                            "filename": src.name,
                            "source_path": str(src),
                            "test_rows": test_count,
                            "action": "Skipped because non-discussion CSV contained test rows.",
                        }
                    )
                    return
            else:
                shutil.copy2(src, dest)
        else:
            shutil.copy2(src, dest)

        self.files_included.append(
            BundleFile(
                filename=item["filename"],
                source_path=str(src),
                role=item["role"],
                why_needed=item["why_needed"],
                copied_path=str(dest),
                generated_train_val_copy=generated_train_val_copy,
            )
        )
        self.log(f"included {item['filename']} from {src}")

    def validate_csv(self, path: Path, role: str) -> dict[str, Any]:
        rows = read_dict_rows(path)
        header = list(rows[0].keys()) if rows else read_header(path)
        row_count = len(rows)
        split_counts: Counter[str] = Counter()
        if "split" in header:
            split_counts.update(str(row.get("split", "")).strip() for row in rows)
        video_split_warnings = []
        if "split" in header and "video_id" in header:
            for row in rows:
                split = str(row.get("split", "")).strip()
                try:
                    video_id = int(str(row.get("video_id", "")).strip())
                except ValueError:
                    continue
                if split == "train" and video_id not in TRAIN_IDS:
                    video_split_warnings.append(f"train row has non-train video_id={video_id}")
                if split == "validation" and video_id not in VALIDATION_IDS:
                    video_split_warnings.append(
                        f"validation row has non-validation video_id={video_id}"
                    )
                if split == "test" and video_id not in TEST_IDS:
                    video_split_warnings.append(f"test row has non-test video_id={video_id}")
        if video_split_warnings:
            sample = sorted(set(video_split_warnings))[:5]
            self.warnings.append(f"{path.name} split/video_id mismatch sample: {sample}")

        key_check = build_key_column_check(path.name, header, role)
        self.row_counts_by_file[path.name] = row_count
        self.split_counts_by_file[path.name] = dict(split_counts)
        self.key_columns_check[path.name] = key_check

        if is_row_level_discussion_csv(path.name, role):
            if "split" not in header:
                self.warnings.append(f"Row-level discussion CSV missing split column: {path}")
            unexpected = sorted(set(split_counts) - TRAIN_VAL_SPLITS - {""})
            if unexpected:
                self.warnings.append(f"Unexpected split values in {path.name}: {unexpected}")
        return {
            "row_count": row_count,
            "split_counts": dict(split_counts),
            "key_check": key_check,
        }

    def scan_excluded_large_detail_files(self) -> None:
        detail_files = [
            "ocr_visual_anchor_frame_results_v2_4_train_val_for_discussion.csv",
            "audio_visual_anchor_persistence_subwindow_features_v2_4_train_val_for_discussion.csv",
        ]
        box_level_patterns = ["box_level", "box-level", "ocr_box"]
        for filename in detail_files:
            for candidate in self.candidates.get(filename, []):
                self.files_excluded_due_to_size.append(
                    {
                        "filename": filename,
                        "source_path": str(candidate.path),
                        "size_bytes": candidate.size,
                        "reason": "Frame/subwindow detail file is not needed for the discussion bundle.",
                    }
                )
        for search_dir in [
            PROJECT_ROOT / "outputs" / "latest_for_chatgpt_ocr_visual_anchor",
            PROJECT_ROOT / "data" / "ocr",
        ]:
            if not search_dir.exists():
                continue
            for path in search_dir.rglob("*.csv"):
                lower_name = path.name.lower()
                if any(pattern in lower_name for pattern in box_level_patterns):
                    self.files_excluded_due_to_size.append(
                        {
                            "filename": path.name,
                            "source_path": str(path),
                            "size_bytes": path.stat().st_size,
                            "reason": "Box-level OCR detail file is outside the rule-review bundle scope.",
                        }
                    )

    def scan_bundle_forbidden_files(self) -> None:
        found = []
        for path in BUNDLE_PATH.rglob("*"):
            if path.is_file() and is_forbidden_path(path):
                found.append(str(path.relative_to(BUNDLE_PATH)))
        self.latest_for_chatgpt_forbidden_files_found = found
        if found:
            self.errors.append(f"Forbidden files found in bundle: {found}")

    def write_readme(self) -> None:
        readme_path = BUNDLE_PATH / "README_latest_files.md"
        generated_readme_item = BundleFile(
            filename="README_latest_files.md",
            source_path="source: collect_latest_rule_discussion_files_v2_4.py",
            role="generated_bundle_manifest",
            why_needed="Manifest explaining selected files, leakage guard, safety checks, and review usage.",
            copied_path=str(readme_path),
        )
        display_files = self.files_included + [generated_readme_item]
        lines = [
            "# collect_latest_rule_discussion_files_v2_4",
            "",
            f"- generated_at: {now_iso()}",
            f"- project_root: `{PROJECT_ROOT}`",
            f"- selected_latest_stage: `{self.selected_stage}`",
            f"- split_file: `{SPLIT_PATH}`",
            f"- split_seed: `{SPLIT_SEED}`",
            f"- old_project_modified: `{str(self.old_project_modified).lower()}`",
            f"- input_files_modified: `{str(self.input_files_modified).lower()}`",
            "",
            "## Included Files",
            "",
            "| filename | source path | role | rows | split counts | test rows | why needed |",
            "| --- | --- | --- | ---: | --- | ---: | --- |",
        ]
        for item in display_files:
            rows = self.row_counts_by_file.get(item.filename)
            split_counts = self.split_counts_by_file.get(item.filename, {})
            test_rows = split_counts.get("test", 0)
            rows_text = "" if rows is None else str(rows)
            split_text = json.dumps(split_counts, ensure_ascii=False, sort_keys=True)
            lines.append(
                "| "
                + " | ".join(
                    [
                        escape_md(item.filename),
                        f"`{item.source_path}`",
                        escape_md(item.role),
                        rows_text,
                        f"`{split_text}`",
                        str(test_rows),
                        escape_md(item.why_needed),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## Core Discussion vs Reference",
                "",
                "- Core discussion files are the compact/fallback discussion table, OCR context features, audio context features, train-only thresholds, cleanup/alignment status, and the latest provenance files.",
                "- Reference files are copied only to support interpretation of feature recommendations, OCR extraction context, and split provenance. They are not final interval detector rules.",
                "",
                "## Missing Files",
                "",
            ]
        )
        if self.files_missing:
            for missing in self.files_missing:
                lines.append(
                    f"- `{missing['filename']}` ({missing['role']}): {missing['reason']}"
                )
        else:
            lines.append("- None.")
        lines.extend(["", "## Excluded Large Or Detail Files", ""])
        excluded = self.files_excluded_due_to_size + self.excluded_reference_paths
        if excluded:
            for item in excluded:
                size_text = human_size(int(item.get("size_bytes", 0)))
                lines.append(
                    f"- `{item['source_path']}` ({size_text}): {item['reason']}"
                )
        else:
            lines.append("- None.")
        lines.extend(
            [
                "",
                "## Leakage Guard",
                "",
                "- test row-level feature excluded: true",
                "- train-only thresholds preserved: true",
                "- validation discussion only: true",
                "- test_row_level_features_copied_to_bundle: false",
                "- label_columns_used_for_support: false",
                "- audit_columns_used_for_support: false",
                "",
                "## Scene Anchor Interpretation Caution",
                "",
                "- Visual scene anchor is not direct evidence of ad start or ad end.",
                "- Visual scene anchor is a transition-time candidate.",
                "- Start/end judgment should be discussed through audio/OCR context flow around the anchor.",
                "",
                "## Current Rule Discussion State",
                "",
                "- Audio is medium-strength supporting evidence.",
                "- Start candidates are OCR-centered.",
                "- Audio supports start decisions and is important for maintenance, gap, and end judgments.",
                "- If OCR coverage is low, OCR-low should not be treated as non-ad evidence.",
                "- Audio high + OCR coverage high + OCR low lowers ad likelihood.",
                "- Low gap bridge and long-ad prior are still being refined through discussion.",
                "",
                "## Safety",
                "",
                f"- old_project_modified: `{str(self.old_project_modified).lower()}`",
                f"- input_files_modified: `{str(self.input_files_modified).lower()}`",
                f"- latest_for_chatgpt_forbidden_files_found: `{json.dumps(self.latest_for_chatgpt_forbidden_files_found, ensure_ascii=False)}`",
            ]
        )
        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not any(item.filename == "README_latest_files.md" for item in self.files_included):
            self.files_included.append(generated_readme_item)

    def update_safety_check(self) -> None:
        self.safety_check = {
            "old_project_modified": self.old_project_modified,
            "input_files_modified": self.input_files_modified,
            "latest_for_chatgpt_forbidden_files_found": self.latest_for_chatgpt_forbidden_files_found,
            "media_frame_cache_model_raw_video_absent": not self.latest_for_chatgpt_forbidden_files_found,
            "bundle_path_inside_project": str(BUNDLE_PATH).startswith(str(PROJECT_ROOT)),
            "old_project_used_only_for_snapshot": True,
        }
        if self.old_project_modified:
            self.errors.append("Old project snapshot changed during collection.")
        if self.input_files_modified:
            self.errors.append("Input source file metadata changed during collection.")

    def write_report(self) -> None:
        report = {
            "task_name": TASK_NAME,
            "project_root": str(PROJECT_ROOT),
            "start_time": self.start_time,
            "end_time": now_iso(),
            "selected_stage": self.selected_stage,
            "bundle_path": str(BUNDLE_PATH),
            "files_included": [item.__dict__ for item in self.files_included],
            "files_missing": self.files_missing,
            "files_excluded_due_to_size": self.files_excluded_due_to_size
            + self.excluded_reference_paths,
            "files_excluded_due_to_test_rows": self.files_excluded_due_to_test_rows,
            "row_counts_by_file": self.row_counts_by_file,
            "split_counts_by_file": self.split_counts_by_file,
            "key_columns_check": self.key_columns_check,
            "leakage_guard_check": self.leakage_guard_check,
            "safety_check": self.safety_check,
            "warnings": self.warnings,
            "errors": self.errors,
            "old_project_modified": self.old_project_modified,
            "input_files_modified": self.input_files_modified,
            "latest_for_chatgpt_forbidden_files_found": self.latest_for_chatgpt_forbidden_files_found,
            "fixed_split_check": {
                "split_file": str(SPLIT_PATH),
                "split_seed": SPLIT_SEED,
                "train_video_id": sorted(TRAIN_IDS),
                "validation_video_id": sorted(VALIDATION_IDS),
                "test_video_id": sorted(TEST_IDS),
            },
            "old_project_snapshot_before": summarize_snapshot(self.old_snapshot_before),
            "old_project_snapshot_after": summarize_snapshot(self.old_snapshot_after),
            "preexisting_outputs_backed_up": self.preexisting_outputs_backed_up,
            "notes": [
                "No new rule was designed.",
                "No interval detector was implemented.",
                "No predicted interval file was created.",
                "Discussion bundle contains train/validation row-level files only.",
            ],
        }
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_log(self) -> None:
        LOG_PATH.write_text("\n".join(self.log_lines) + "\n", encoding="utf-8")

    def print_final_summary(self) -> None:
        core_files = [
            item.filename
            for item in self.files_included
            if item.role.startswith("core") or item.role == "core_discussion_table"
        ]
        summary = {
            "status": "success" if not self.errors else "completed_with_errors",
            "selected_stage": self.selected_stage,
            "bundle_path": str(BUNDLE_PATH),
            "core_files": core_files,
            "missing_files": self.files_missing,
            "test_row_level_features_copied_to_bundle": False,
            "old_project_modified": self.old_project_modified,
            "input_files_modified": self.input_files_modified,
            "forbidden_files_found": self.latest_for_chatgpt_forbidden_files_found,
            "report_path": str(REPORT_PATH),
            "log_path": str(LOG_PATH),
        }
        print("Rule discussion bundle collection complete")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_parent_dirs() -> None:
    for path in [REPORT_PATH.parent, LOG_PATH.parent, BACKUP_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def rel_to_project(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def find_candidates(filename: str, dirs: list[Path], stage: str) -> list[Candidate]:
    found: list[Candidate] = []
    for rank, directory in enumerate(dirs):
        if not directory.exists():
            continue
        direct = directory / filename
        paths: list[Path] = []
        if direct.exists() and direct.is_file():
            paths.append(direct)
        for path in directory.rglob(filename):
            if path.is_file() and path not in paths:
                paths.append(path)
        for path in paths:
            if "backups" in path.parts or path.name.endswith(".pyc"):
                continue
            stat = path.stat()
            found.append(
                Candidate(
                    filename=filename,
                    path=path,
                    stage=stage,
                    search_rank=rank,
                    mtime_ns=stat.st_mtime_ns,
                    size=stat.st_size,
                )
            )
    return found


def choose_best(candidates: list[Candidate]) -> Candidate:
    return sorted(candidates, key=lambda c: (c.search_rank, -c.mtime_ns, str(c.path)))[0]


def read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        try:
            return next(reader)
        except StopIteration:
            return []


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_train_val_only_csv(src: Path, dest: Path) -> None:
    with src.open(newline="", encoding="utf-8-sig") as in_file:
        reader = csv.DictReader(in_file)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader if row.get("split") in TRAIN_VAL_SPLITS]
    with dest.open("w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_key_column_check(filename: str, header: list[str], role: str) -> dict[str, Any]:
    header_set = set(header)
    common = [
        "version",
        "split",
        "video_id",
        "visual_anchor_id",
        "candidate_time_sec",
        "candidate_time_mmss",
    ]
    audio_ocr = [
        "audio_start_signal_level",
        "audio_end_signal_level",
        "audio_context_level",
        "ocr_start_signal_level",
        "ocr_end_signal_level",
        "ocr_context_level",
    ]
    semantic = [
        "visual_anchor_role",
        "scene_is_direct_ad_start_evidence",
        "scene_is_direct_ad_end_evidence",
        "ocr_context_reliability_level",
        "ocr_low_score_reason",
    ]
    required_groups = {"common_recommended": common}
    if "discussion" in role or "ocr" in role or "audio" in role:
        required_groups["audio_ocr_recommended"] = audio_ocr
    if "semantic" in filename or "compact" in filename:
        required_groups["semantic_cleanup_recommended"] = semantic
    result: dict[str, Any] = {"column_count": len(header), "groups": {}}
    for group, columns in required_groups.items():
        result["groups"][group] = {
            "present": [column for column in columns if column in header_set],
            "missing": [column for column in columns if column not in header_set],
        }
    return result


def is_row_level_discussion_csv(filename: str, role: str) -> bool:
    name = filename.lower()
    role_lower = role.lower()
    if not name.endswith(".csv"):
        return False
    if any(token in name for token in ["threshold", "summary", "status", "recommendation", "config"]):
        return False
    return any(
        token in name or token in role_lower
        for token in ["discussion", "features", "train_val", "compact", "context_feature"]
    )


def is_forbidden_path(path: Path) -> bool:
    if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
        return True
    lowered_parts = {part.lower() for part in path.parts}
    if path.is_relative_to(BUNDLE_PATH):
        return bool(lowered_parts & FORBIDDEN_PATH_TOKENS)
    return False


def collect_file_metadata(paths: list[Path]) -> dict[str, dict[str, int]]:
    metadata: dict[str, dict[str, int]] = {}
    for path in paths:
        if not path.exists():
            continue
        stat = path.stat()
        metadata[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return metadata


def snapshot_project(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    if not root.exists():
        return {
            "root": str(root),
            "exists": False,
            "file_count": 0,
            "sha256": None,
            "files": [],
        }
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git"}]
        for filename in filenames:
            path = Path(dirpath) / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    files.sort(key=lambda item: item["path"])
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "root": str(root),
        "exists": True,
        "created_at": now_iso(),
        "file_count": len(files),
        "sha256": digest,
        "files": files,
    }


def summarize_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "root": snapshot.get("root"),
        "exists": snapshot.get("exists"),
        "file_count": snapshot.get("file_count"),
        "sha256": snapshot.get("sha256"),
        "created_at": snapshot.get("created_at"),
    }


def snapshots_match(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> bool:
    if before is None or after is None:
        return False
    return (
        before.get("exists") == after.get("exists")
        and before.get("file_count") == after.get("file_count")
        and before.get("sha256") == after.get("sha256")
    )


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{size_bytes}B"


def p(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


STAGE_CONFIGS: dict[str, dict[str, Any]] = {
    "semantic_cleanup": {
        "dirs": [
            p("outputs", "latest_for_chatgpt_scene_audio_ocr_semantic_cleanup"),
            p("data", "fusion"),
            p("reports", "fusion"),
            p("logs"),
            p("scripts", "fusion"),
        ],
        "files": [
            "scene_audio_ocr_rule_discussion_compact_v2_4_train_val.csv",
            "scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val_semantic_fixed.csv",
            "ocr_visual_anchor_coverage_reliability_summary_v2_4_train_val.csv",
            "visual_anchor_semantic_cleanup_status_v2_4.csv",
            "scene_audio_ocr_semantic_cleanup_v2_4_summary.md",
            "scene_audio_ocr_semantic_cleanup_v2_4_report.json",
            "scene_audio_ocr_semantic_cleanup_v2_4_run_log.txt",
            "scene_audio_ocr_semantic_cleanup_v2_4.py",
            "README_latest_files.md",
            "ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv",
            "audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv",
        ],
    },
    "ocr_visual_anchor": {
        "dirs": [
            p("outputs", "latest_for_chatgpt_ocr_visual_anchor"),
            p("data", "ocr"),
            p("data", "fusion"),
            p("reports", "ocr"),
            p("logs"),
            p("scripts", "ocr"),
        ],
        "files": [
            "scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val_with_ocr.csv",
            "ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv",
            "ocr_visual_anchor_level_thresholds_v2_4_train_only.csv",
            "ocr_visual_anchor_alignment_status_v2_4.csv",
            "ocr_visual_anchor_context_features_v2_4_summary.md",
            "ocr_visual_anchor_context_features_v2_4_report.json",
            "ocr_visual_anchor_context_features_v2_4_run_log.txt",
            "extract_ocr_visual_anchor_context_features_v2_4.py",
            "README_latest_files.md",
            "ocr_visual_anchor_frame_results_v2_4_train_val_for_discussion.csv",
        ],
    },
    "audio_visual_anchor": {
        "dirs": [
            p("outputs", "latest_for_chatgpt_visual_anchor_alignment"),
            p("data", "audio"),
            p("data", "fusion"),
            p("reports", "fusion"),
            p("reports", "audio"),
        ],
        "files": [
            "scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val.csv",
            "audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv",
            "audio_visual_anchor_level_thresholds_v2_4_train_only.csv",
            "visual_anchor_alignment_status_v2_4.csv",
            "visual_anchor_alignment_pack_v2_4_summary.md",
            "visual_anchor_alignment_pack_v2_4_report.json",
            "README_latest_files.md",
            "visual_anchor_alignment_pack_v2_4_run_log.txt",
        ],
    },
}

REFERENCE_FILE_SEARCH: dict[str, list[Path]] = {
    "audio_persistence_rule_config_v2_4_train_only.json": [
        p("outputs", "latest_for_chatgpt_visual_anchor_alignment"),
        p("configs"),
        p("data", "audio"),
    ],
    "audio_rule_feature_recommendations_v2_4_train_only.csv": [
        p("outputs", "latest_for_chatgpt_visual_anchor_alignment"),
        p("data", "audio"),
        p("reports", "audio"),
    ],
    "audio_rule_validation_summary_v2_4_train_only.csv": [
        p("outputs", "latest_for_chatgpt_visual_anchor_alignment"),
        p("data", "audio"),
        p("reports", "audio"),
    ],
    "ocr_rule_feature_recommendations_v2_4.csv": [
        p("data", "ocr"),
        p("outputs", "latest_for_chatgpt_ocr_visual_anchor"),
    ],
    "scene_change_split_v2_4_summary.md": [
        p("outputs", "latest_for_chatgpt"),
        p("reports"),
    ],
    "extract_ocr_cues_v2_4_summary.md": [
        p("reports", "ocr"),
        p("outputs", "latest_for_chatgpt_ocr_visual_anchor"),
    ],
    "ocr_labeled_segment_analysis_v2_4.md": [
        p("reports", "ocr"),
    ],
    "ocr_failed_retry_and_train_corpus_analysis_v2_4.md": [
        p("reports", "ocr"),
    ],
    "create_train_only_ocr_feature_files_v2_4_summary.md": [
        p("reports", "ocr"),
    ],
}

REFERENCE_FILES: list[tuple[str, str, str]] = [
    (
        "audio_persistence_rule_config_v2_4_train_only.json",
        "reference_audio_train_only_config",
        "Train-only audio persistence rule configuration reference.",
    ),
    (
        "audio_rule_feature_recommendations_v2_4_train_only.csv",
        "reference_audio_feature_recommendations",
        "Train-only audio feature recommendations for discussion context.",
    ),
    (
        "audio_rule_validation_summary_v2_4_train_only.csv",
        "reference_audio_validation_summary",
        "Train-only audio validation summary for provenance.",
    ),
    (
        "ocr_rule_feature_recommendations_v2_4.csv",
        "reference_ocr_feature_recommendations",
        "OCR feature recommendation context.",
    ),
    (
        "scene_change_split_v2_4_summary.md",
        "reference_scene_split_summary",
        "Scene split provenance and train/validation/test split context.",
    ),
    (
        "extract_ocr_cues_v2_4_summary.md",
        "reference_ocr_cue_extraction_summary",
        "OCR cue extraction context.",
    ),
    (
        "ocr_labeled_segment_analysis_v2_4.md",
        "reference_ocr_labeled_segment_analysis",
        "OCR labeled segment analysis context.",
    ),
    (
        "ocr_failed_retry_and_train_corpus_analysis_v2_4.md",
        "reference_ocr_failed_retry_analysis",
        "OCR retry and train corpus analysis context.",
    ),
    (
        "create_train_only_ocr_feature_files_v2_4_summary.md",
        "reference_train_only_ocr_summary",
        "Train-only OCR feature generation summary.",
    ),
]


def main() -> None:
    collector = Collector()
    collector.run()


if __name__ == "__main__":
    main()
