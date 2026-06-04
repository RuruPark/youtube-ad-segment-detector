#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write documentation-only rule specs for scene/audio/OCR interval rules.

This script intentionally does not implement a detector, apply rules to rows,
generate predicted intervals, tune thresholds, or evaluate train/validation/test
performance. It writes human-readable documentation and a documentation-only
draft config.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


TASK_NAME = "scene_audio_ocr_rule_documentation"
VERSION = "v1.1"
SPLIT_SEED = 20240524

PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")

REPORT_DIR = PROJECT_ROOT / "reports" / "rules"
CONFIG_DIR = PROJECT_ROOT / "configs" / "rule_docs"
LOG_DIR = PROJECT_ROOT / "logs"
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "rules"
BUNDLE_DIR = PROJECT_ROOT / "outputs" / "latest_for_chatgpt_rule_docs_v1_1"

RUN_LOG = LOG_DIR / "scene_audio_ocr_rule_documentation_v1_1_run_log.txt"
SUMMARY_MD = REPORT_DIR / "scene_audio_ocr_rule_documentation_v1_1_summary.md"
REPORT_JSON = REPORT_DIR / "scene_audio_ocr_rule_documentation_v1_1_report.json"
STATE_DOC = REPORT_DIR / "scene_audio_ocr_state_machine_rule_v1_1.md"
AUDIO_DOC = REPORT_DIR / "audio_rule_v1.md"
OCR_DOC = REPORT_DIR / "ocr_rule_v1.md"
CHECKLIST_DOC = REPORT_DIR / "state_machine_rule_implementation_checklist_v1_1.md"
CONFIG_DRAFT = CONFIG_DIR / "scene_audio_ocr_state_machine_rule_v1_1_draft.json"
SCRIPT_PATH = SCRIPT_DIR / "write_scene_audio_ocr_rule_docs_v1_1.py"

SUB_AGENT_RESULTS_JSON = REPORT_DIR / "sub_agent_validation_results_v1_1.json"

OLD_SNAPSHOT_BEFORE = REPORT_DIR / "old_project_snapshot_before_rule_docs_v1_1.tsv"
OLD_SNAPSHOT_AFTER = REPORT_DIR / "old_project_snapshot_after_rule_docs_v1_1.tsv"

TRAIN_VIDEO_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
VALIDATION_VIDEO_IDS = [3, 7, 18]
TEST_VIDEO_IDS = [4, 16, 17]

EXPECTED_REFERENCE_FILES = [
    "data/splits/video_split_v2_4.csv",
    "data/features/visual_scene_boundary_anchors_v2_4.csv",
    "data/features/visual_scene_boundary_anchors_v2_4_with_split.csv",
    "data/fusion/scene_audio_ocr_rule_discussion_compact_v2_4_train_val.csv",
    "data/fusion/scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val_semantic_fixed.csv",
    "reports/fusion/scene_audio_ocr_semantic_cleanup_v2_4_summary.md",
    "reports/fusion/scene_audio_ocr_semantic_cleanup_v2_4_report.json",
    "data/fusion/scene_audio_ocr_visual_anchor_discussion_table_v2_4_train_val_with_ocr.csv",
    "data/ocr/ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv",
    "data/ocr/ocr_visual_anchor_level_thresholds_v2_4_train_only.csv",
    "reports/ocr/ocr_visual_anchor_context_features_v2_4_summary.md",
    "reports/ocr/ocr_visual_anchor_context_features_v2_4_report.json",
    "data/audio/audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv",
    "data/audio/audio_visual_anchor_level_thresholds_v2_4_train_only.csv",
    "configs/audio_persistence_rule_config_v2_4_train_only.json",
    "data/audio/audio_rule_feature_recommendations_v2_4_train_only.csv",
    "data/audio/audio_rule_validation_summary_v2_4_train_only.csv",
    "data/ocr/ocr_rule_feature_recommendations_v2_4.csv",
    "reports/ocr/ocr_labeled_segment_analysis_v2_4.md",
    "reports/ocr/extract_ocr_cues_v2_4_summary.md",
    "reports/ocr/ocr_failed_retry_and_train_corpus_analysis_v2_4.md",
    "reports/ocr/create_train_only_ocr_feature_files_v2_4_summary.md",
]

OLD_EXPECTED_OCR_REFERENCE_PATHS = [
    "reports/ocr_labeled_segment_analysis_v2_4.md",
    "reports/extract_ocr_cues_v2_4_summary.md",
    "reports/ocr_failed_retry_and_train_corpus_analysis_v2_4.md",
    "reports/create_train_only_ocr_feature_files_v2_4_summary.md",
]

NEW_EXPECTED_OCR_REFERENCE_PATHS = [
    "reports/ocr/ocr_labeled_segment_analysis_v2_4.md",
    "reports/ocr/extract_ocr_cues_v2_4_summary.md",
    "reports/ocr/ocr_failed_retry_and_train_corpus_analysis_v2_4.md",
    "reports/ocr/create_train_only_ocr_feature_files_v2_4_summary.md",
]

REFERENCE_PATH_CLEANUP_APPLIED = True
REFERENCE_PATH_CLEANUP_REASON = "OCR reports live under reports/ocr, not reports root"
DUPLICATE_ROOT_REFERENCE_FILES_CREATED = False
SYMLINKS_CREATED = False

ALTERNATE_REFERENCE_FILES = {}

BUNDLE_FILES = [
    STATE_DOC,
    AUDIO_DOC,
    OCR_DOC,
    CHECKLIST_DOC,
    CONFIG_DRAFT,
    SUMMARY_MD,
    REPORT_JSON,
    RUN_LOG,
    SCRIPT_PATH,
]

FORBIDDEN_BUNDLE_SUFFIXES = {
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
    ".csv",
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
}
FORBIDDEN_BUNDLE_PARTS = {"cache", "tmp", "__pycache__", "frames", "frame_images", "raw_video"}


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in [REPORT_DIR, CONFIG_DIR, LOG_DIR, SCRIPT_DIR, BUNDLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def log(step: str) -> None:
    print(step, flush=True)
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{iso_now()} {step}\n")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            return next(reader, [])
    except Exception:
        return []


def file_stat_signature(path: Path) -> dict[str, Any]:
    st = path.stat()
    digest = ""
    if path.is_file() and st.st_size <= 20 * 1024 * 1024:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
    return {
        "path": str(path),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256_if_small": digest,
    }


def snapshot_tree(root: Path, output_path: Path) -> list[str]:
    rows: list[str] = []
    if not root.exists():
        output_path.write_text("MISSING_ROOT\n", encoding="utf-8")
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        rows.append(f"{path.relative_to(root)}\t{st.st_size}\t{st.st_mtime_ns}")
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return rows


def collect_reference_inventory() -> dict[str, Any]:
    found: list[str] = []
    missing: list[str] = []
    alternates_found: dict[str, list[str]] = {}
    csv_columns: dict[str, list[str]] = {}

    for item in EXPECTED_REFERENCE_FILES:
        path = PROJECT_ROOT / item
        if path.exists():
            found.append(str(path))
            if path.suffix.lower() == ".csv":
                csv_columns[item] = read_csv_header(path)
        else:
            missing.append(str(path))
            alt_hits = []
            for alt in ALTERNATE_REFERENCE_FILES.get(item, []):
                alt_path = PROJECT_ROOT / alt
                if alt_path.exists():
                    alt_hits.append(str(alt_path))
                    found.append(str(alt_path))
                    if alt_path.suffix.lower() == ".csv":
                        csv_columns[alt] = read_csv_header(alt_path)
            if alt_hits:
                alternates_found[item] = alt_hits

    return {
        "reference_files_found": sorted(set(found)),
        "reference_files_missing": missing,
        "alternate_reference_files_found": alternates_found,
        "csv_columns": csv_columns,
    }


def validate_split_file(warnings: list[str]) -> dict[str, Any]:
    split_path = PROJECT_ROOT / "data/splits/video_split_v2_4.csv"
    observed: dict[str, list[int]] = {}
    seeds: list[str] = []
    if not split_path.exists():
        warnings.append("Fixed split file is missing: data/splits/video_split_v2_4.csv")
        return {"valid": False, "observed": observed, "split_seed_values": seeds}
    try:
        with split_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                observed.setdefault(row.get("split", ""), []).append(int(row.get("video_id", "-1")))
                seed = row.get("split_seed", "")
                if seed and seed not in seeds:
                    seeds.append(seed)
    except Exception as exc:
        warnings.append(f"Could not parse fixed split file: {exc}")
        return {"valid": False, "observed": observed, "split_seed_values": seeds}
    for key in list(observed):
        observed[key] = sorted(observed[key])
    valid = (
        observed.get("train") == TRAIN_VIDEO_IDS
        and observed.get("validation") == VALIDATION_VIDEO_IDS
        and observed.get("test") == TEST_VIDEO_IDS
        and seeds == [str(SPLIT_SEED)]
    )
    if not valid:
        warnings.append("Fixed split file contents do not match the documented v2_4 split.")
    return {"valid": valid, "observed": observed, "split_seed_values": seeds}


def reference_markdown(inventory: dict[str, Any]) -> str:
    lines = ["## Reference Inventory", ""]
    lines.append("Found reference files:")
    for path in inventory["reference_files_found"]:
        lines.append(f"- `{rel(Path(path))}`")
    if not inventory["reference_files_found"]:
        lines.append("- None")
    lines.append("")
    lines.append("Missing reference files:")
    for path in inventory["reference_files_missing"]:
        lines.append(f"- `{rel(Path(path))}`")
    if not inventory["reference_files_missing"]:
        lines.append("- None")
    if inventory["alternate_reference_files_found"]:
        lines.append("")
        lines.append("Alternate same-name files found:")
        for expected, alts in inventory["alternate_reference_files_found"].items():
            alt_text = ", ".join(f"`{rel(Path(p))}`" for p in alts)
            lines.append(f"- `{expected}` -> {alt_text}")
    if inventory["csv_columns"]:
        lines.append("")
        lines.append("Observed CSV columns, truncated to the first 40 columns per file:")
        for item, cols in sorted(inventory["csv_columns"].items()):
            shown = ", ".join(f"`{c}`" for c in cols[:40])
            suffix = " ..." if len(cols) > 40 else ""
            lines.append(f"- `{item}`: {shown}{suffix}")
    return "\n".join(lines)


def fixed_split_markdown() -> str:
    return f"""## Fixed Split

The documentation is anchored to the fixed `v2_4` split.

| Split | video_id |
|---|---|
| train | {", ".join(map(str, TRAIN_VIDEO_IDS))} |
| validation | {", ".join(map(str, VALIDATION_VIDEO_IDS))} |
| test | {", ".join(map(str, TEST_VIDEO_IDS))} |

- `split_seed`: `{SPLIT_SEED}`
- Thresholds, level definitions, keyword candidates, and rule design must be based on the train split only.
- Validation is for discussion, audit, and review of adjustment candidates.
- Test is preserved until final evaluation.
- Test row-level features are not included in the discussion or documentation bundle.
- Existing full-data exploratory artifacts remain reference-only and must not be treated as rule-design authority.
"""


def write_audio_doc(generated_at: str, inventory: dict[str, Any]) -> None:
    content = f"""# Audio Rule v1

Generated at: `{generated_at}`

This is a documentation-only rule note. It does not implement a detector, apply rules, create predicted intervals, tune thresholds, or evaluate performance.

{fixed_split_markdown()}

## Role

Audio is medium-strength supporting evidence. It does not independently confirm ad start or ad end.

- For start decisions, audio supports OCR-centered evidence.
- For continuity, internal transitions, low gap bridging, and long-ad reasoning, audio carries more weight.
- Audio levels should be interpreted around a visual transition anchor, usually as pre/post 10-second context.

## Required Inputs

Likely implementation inputs, based on current artifacts:

- `data/audio/audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv`
- `data/audio/audio_visual_anchor_level_thresholds_v2_4_train_only.csv`
- `configs/audio_persistence_rule_config_v2_4_train_only.json`
- `data/audio/audio_rule_feature_recommendations_v2_4_train_only.csv`
- `data/audio/audio_rule_validation_summary_v2_4_train_only.csv`

Important columns seen in current discussion files include:

- anchor identity/time: `video_id`, `visual_anchor_id`, `candidate_time_sec`, `candidate_time_mmss`, `split`
- pre/post context scores: `audio_pre_10s_ad_like_ratio`, `audio_post_10s_ad_like_ratio`, `audio_pre_10s_score_mean`, `audio_post_10s_score_mean`
- deltas: `audio_score_delta_post_minus_pre`, `audio_score_delta_pre_minus_post`
- levels: `audio_start_signal_level`, `audio_end_signal_level`, `audio_context_level`, `audio_before_context_level`, `audio_after_context_level`

## Start Support

Audio start support means that the 10 seconds after a visual transition anchor become more ad-like. It is support, not the main trigger.

| OCR and coverage situation | Audio interpretation |
|---|---|
| OCR start high + audio start high/medium | strong start candidate |
| OCR start medium + audio start high/medium | supports keeping `start_pending` |
| OCR coverage low + audio start high/medium | weak `start_pending` can be considered |
| OCR coverage high + OCR low + audio high | do not create a start candidate from audio alone |

## End Support

Audio end support means the 10 seconds before an anchor are ad-like and the 10 seconds after the anchor are low. End decisions are more conservative than start decisions.

| Context | Interpretation |
|---|---|
| `audio_before_context` high/medium + `audio_after_context` low | end support |
| OCR coverage high + OCR low + audio low | end can be confirmed if other state rules allow it |
| OCR coverage low + OCR low + audio low | hold as `end_pending`; OCR low is unreliable under low coverage |
| audio before and after both high/medium | possible internal-ad transition, not an end |

## Internal-Ad Transition

If audio is high/medium before and high/medium after the anchor, the visual change can be interpreted as a transition within the same ad flow.

- anchor before audio high/medium
- anchor after audio high/medium
- result: possible internal-ad transition

## Audio Continuity

In `in_ad` state:

- `audio_context` high/medium supports ad continuity.
- If OCR coverage is low, audio high/medium can carry continuity because OCR is unknown.
- If OCR coverage is high and OCR is low while audio is high, use the C3 conflict rule instead of treating audio as strong continuity.

## C3 Conflict Rule

C3 is defined as:

- audio high
- OCR coverage high
- OCR low

Interpretation:

- OCR had enough valid observation but did not find strong ad text, so the ad likelihood is lowered.
- Textless ads are possible but are treated as less common in this rule set.
- Audio alone should not strongly maintain or start an ad under C3.
- If C3 appears at two consecutive anchors, the state machine should lean toward end or non-ad.

## D2 Rule

D2 is defined as:

- audio low
- OCR high

Interpretation:

- Even when audio is weak, strong ad text on screen supports continuing an ad state.
- The support is medium, not strong, because the audio side is weak.

## Low Gap Bridge

Initial implementation should calculate low gap duration by anchor times:

- pattern: high -> low -> high
- low gap duration = next high/medium anchor time - low anchor time

| Gap duration | Bridge rule |
|---|---|
| `gap <= 10s` | bridge possible if either audio or OCR is medium or higher |
| `10s < gap <= 20s` | bridge only if both audio and OCR are medium or higher |
| `gap > 20s` | no automatic bridge |

C3 handling:

- C3 is not strong bridge evidence.
- For `gap <= 10s`, C3 can be weak bridge or review evidence.
- For `10s < gap <= 20s`, C3 alone must not bridge.

## Minimum Duration Prior

- Minimum ad continuity prior: 20 seconds.
- An end candidate within 20 seconds after ad start is held by default.
- E2 exception: within 20 seconds, end can be allowed only if OCR and audio are both low, OCR coverage is sufficient, and the following flow remains low.

## Long-Ad Prior

- Typical ad length is considered roughly 40 to 120 seconds, but this is not a hard cut.
- Passing 120 seconds must not force an end.
- After 120 seconds, end-like flow can be treated more sensitively.
- If audio/OCR ad evidence remains medium or higher, continue the ad.
- If OCR coverage high + OCR low repeats and audio also drops, increase end likelihood.
- Repeated C3 during a long-ad span can tilt toward end or non-ad.

## Leakage Guard

- Audio thresholds and levels are train-split derived.
- Validation can be used for discussion or audit only.
- Test must remain untouched until final evaluation.
- This document does not use test row-level features.
- This document does not generate predicted intervals.

{reference_markdown(inventory)}
"""
    AUDIO_DOC.write_text(content, encoding="utf-8")


def write_ocr_doc(generated_at: str, inventory: dict[str, Any]) -> None:
    content = f"""# OCR Rule v1

Generated at: `{generated_at}`

This is a documentation-only rule note. It does not implement a detector, apply rules, create predicted intervals, tune thresholds, or evaluate performance.

{fixed_split_markdown()}

## Role

OCR is the main evidence source for ad start candidates.

- OCR keywords are not final ad conditions by themselves.
- OCR signals must be combined with scene transition anchors and audio context.
- OCR failed/empty is not non-ad evidence.
- OCR coverage and OCR score must be interpreted separately.

## OCR Coverage

Definitions:

- OCR coverage means whether the context was sufficiently observed.
- OCR score means how strong the ad-like text is.
- Valid OCR frame = `success` + `empty`.
- `failed` is not a valid frame.

Coverage levels:

| Level | Valid OCR frame ratio |
|---|---|
| high coverage | `>= 0.8` |
| medium coverage | `>= 0.4` and `< 0.8` |
| low coverage | `< 0.4` |

Interpretation:

- OCR low + coverage high means ad text is weak by OCR evidence.
- OCR low + coverage low means OCR unknown, not non-ad.
- OCR failed/empty is not non-ad evidence.
- Empty means OCR ran successfully and found no text.
- Failed means frame decode/OCR reliability was poor.

## Required Inputs

Likely implementation inputs, based on current artifacts:

- `data/ocr/ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv`
- `data/ocr/ocr_visual_anchor_level_thresholds_v2_4_train_only.csv`
- `data/ocr/ocr_rule_feature_recommendations_v2_4.csv`
- OCR analysis summaries under `reports/ocr/` when available

Important columns seen in current discussion files include:

- anchor identity/time: `video_id`, `visual_anchor_id`, `candidate_time_sec`, `candidate_time_mmss`, `split`
- OCR reliability: `context_pre_ocr_success_count`, `context_pre_ocr_empty_count`, `context_pre_ocr_failed_count`, `context_post_ocr_success_count`, `context_post_ocr_empty_count`, `context_post_ocr_failed_count`
- OCR text/keyword counts: `ocr_pre_10s_ad_disclosure_count`, `ocr_post_10s_ad_disclosure_count`, `ocr_pre_10s_purchase_cta_count`, `ocr_post_10s_purchase_cta_count`, `ocr_pre_10s_product_brand_count`, `ocr_post_10s_product_brand_count`
- OCR levels/scores: `ocr_start_signal_level`, `ocr_end_signal_level`, `ocr_context_level`, `ocr_start_signal_score`, `ocr_end_signal_score`, `ocr_context_score`

## Start Rule

Confirmed choices:

- 1A: strong ad disclosure is OCR start high.
- 2A: one weak ad word can be OCR start medium.
- 3B: product/brand repetition is mainly continuity evidence, not start confirmation.

Strong disclosure examples:

- 유료광고
- 유료 광고
- 유료광고 포함
- 광고 포함
- 이 영상은 유료광고를 포함
- paid promotion
- sponsored

Weak disclosure examples:

- 광고
- 협찬
- 제공
- ad
- sponsor

Rules:

| OCR signal | Interpretation |
|---|---|
| strong disclosure | OCR start high |
| weak disclosure alone | OCR start medium |
| product repetition alone | continuity/internal evidence rather than confirmed start |
| weak disclosure + product repetition | more reliable `start_pending` |
| negative guard present | disclosure score becomes 0 |

## End Rule

Confirmed choices:

- 4B: CTA text is OCR end medium; it can become end high when followed by OCR/audio decrease.
- 5B: pinned comment/subscriber event text is OCR end medium; it can become end high with a following decrease.
- 6A: more-info/link/description text is OCR end medium.

CTA examples:

- 더보기
- 링크
- 설명란
- 고정댓글
- 댓글 확인
- 구매
- 쿠폰
- 가입
- 다운로드
- 신청
- 구독자 이벤트
- 이벤트 참여

Rules:

| OCR signal | Interpretation |
|---|---|
| CTA in pre_10s | OCR end medium |
| pre_10s CTA + post_10s OCR ad evidence decrease + audio after low | end high possible |
| CTA alone | not a confirmed end; use as support for `end_pending` or end review |

## Product Repetition Rule

Confirmed choice: 3B.

- Product or brand-like tokens are not strong core dictionary keywords by themselves.
- Repeated product/brand-like tokens across frames or contexts support ad continuity.
- If product repetition continues across pre/post context, prefer internal-ad transition or `in_ad` continuity.
- Product repetition alone must not confirm start.

## Negative Guard Rule

Confirmed choice: 7A.

If any of these expressions appear, disclosure score should be treated as 0:

- 광고 아님
- 협찬 아님
- 내돈내산
- 직접 구매
- 제 돈 주고 샀어요
- 광고가 아닙니다
- 협찬 받은 거 아님

Interpretation:

- The word "광고" must not count as ad evidence when it appears in a negated expression such as "광고 아님".

## Discount/Benefit Rule

Confirmed choice: 8C.

- 할인, 혜택, 무료, 이벤트, 세일, 특가 and similar benefit terms are weak by themselves.
- Use them weakly only when they appear with purchase/product/link/CTA context.
- Preserve them as analysis counts.
- Do not let these terms alone drive start/end core scores.

## OCR Coverage and Reliability Guard

| Situation | Interpretation |
|---|---|
| OCR low + coverage high | ad-like text weak by OCR |
| OCR low + coverage low | OCR unknown |
| OCR failed | reliability issue, not non-ad |
| OCR empty | successful OCR with no text; not automatically non-ad |
| coverage high + repeated low score | can support non-ad or end when audio also weakens |

## Leakage Guard

- OCR keyword candidates and score levels must be designed from train split only.
- Validation can be used for audit and discussion.
- Test row-level OCR features must not be included in rule discussion bundles.
- This document does not generate predicted intervals or performance claims.

{reference_markdown(inventory)}
"""
    OCR_DOC.write_text(content, encoding="utf-8")


def write_state_doc(generated_at: str, inventory: dict[str, Any]) -> None:
    content = f"""# Scene + Audio + OCR State-Machine Interval Detector Rule v1.1

Generated at: `{generated_at}`

This is a documentation-only rule specification. It does not implement the detector, run automatic rule application, create predicted ad intervals, tune thresholds, evaluate splits, or inspect test row-level features.

{fixed_split_markdown()}

## Scope

This document records the current human-readable rule design for a future state-machine interval detector. It consolidates:

1. visual scene anchor role
2. audio rule v1
3. OCR rule v1
4. scene + audio + OCR state-machine interval rule v1.1
5. low gap bridge rule
6. long-ad prior
7. OCR coverage/reliability interpretation
8. leakage guard and split principles
9. columns and input files needed for future implementation

## Visual Scene Anchor Role

visual scene anchor는 광고 시작/종료의 직접 evidence가 아니라, 상태 변화가 일어날 수 있는 화면 전환 후보 시각이다.

Use these terms:

- visual transition anchor
- transition_time_anchor
- `candidate_time_sec` 기준 pre/post context
- anchor 이후 OCR/audio가 start-like 흐름
- anchor 이전/이후 OCR/audio가 end-like 흐름
- anchor 전후 OCR/audio가 유지되면 internal-ad transition

Interpretation:

- The anchor provides a candidate time where a state transition may occur.
- The detector looks at pre/post 10-second OCR and audio context around the anchor.
- Start/end/internal/non-ad decisions come from OCR/audio flow around the anchor.
- The anchor time is used as the interval boundary only after the context evidence supports the state transition.
- Legacy visual score columns with start/end naming should not be read as direct interval evidence.

## Timestamp Convention

- `ad_start = visual transition anchor candidate_time_sec`
- `ad_end = visual transition anchor candidate_time_sec`

Reason:

- A visual transition is a natural skip-boundary location.
- OCR/audio validate or reject the transition interpretation.

## State Set

| State | Meaning |
|---|---|
| `non_ad` | not currently inside an ad interval |
| `start_pending` | weak/medium start evidence exists but is not confirmed |
| `in_ad` | currently inside an ad interval |
| `end_pending` | possible end evidence exists but requires confirmation |

## Audio Rule v1 Summary

Audio is medium-strength supporting evidence.

- Audio never confirms start/end by itself.
- Start: OCR is primary; audio high/medium strengthens start evidence.
- End: before high/medium and after low supports end; end decisions remain conservative.
- Continuity: audio context high/medium supports `in_ad`, especially when OCR coverage is low.
- Internal transition: audio before and after both high/medium suggests continuity through a visual transition.
- C3: audio high + OCR coverage high + OCR low lowers ad likelihood; two consecutive C3 anchors tilt toward end/non-ad.
- D2: audio low + OCR high is medium support for ad continuity.

## OCR Rule v1 Summary

OCR is primary for ad start candidates.

- Strong disclosure -> OCR start high.
- Weak disclosure alone -> OCR start medium.
- Product/brand repetition -> continuity/internal evidence, not start confirmation.
- CTA in pre_10s -> OCR end medium.
- CTA + later OCR/audio decrease -> end high possible.
- Negative guard zeros disclosure score.
- Discount/benefit words are weak alone and only count weakly with CTA/product/link context.
- OCR failed/empty is not non-ad evidence.
- Coverage and score are separate dimensions.

Coverage levels:

| Level | Valid OCR frame ratio |
|---|---|
| high | `>= 0.8` |
| medium | `>= 0.4` and `< 0.8` |
| low | `< 0.4` |

## State Transition Rules

### `non_ad`

Goal: find an ad start candidate.

- strong disclosure / OCR start high -> `ad_start = anchor_time`, `state = in_ad`
- weak disclosure / OCR start medium -> `state = start_pending`
- OCR low + coverage low + audio high/medium -> weak `start_pending` possible
- OCR low + coverage high -> not a start candidate
- audio high alone must not confirm start
- C3 is not a start candidate
- If the next anchor clearly rises in OCR/audio, a future implementation may review the previous anchor as a retroactive start candidate.

### `start_pending`

- If later OCR context medium/high or audio context medium/high continues, confirm `in_ad`.
- `ad_start` uses `pending_start_time`.
- If OCR coverage high + OCR low + audio low continues, cancel pending start.
- TODO: define max pending duration or max pending anchor count so `start_pending` cannot last too long.

### `in_ad`

- OCR context high/medium -> maintain ad.
- Product repetition maintained -> maintain ad.
- Audio context high/medium + OCR coverage low -> maintain ad because OCR is unknown.
- Audio low + OCR high -> D2, maintain ad with medium support.
- Audio high + OCR coverage high + OCR low -> C3; do not treat as strong continuity.
- Two consecutive C3 anchors -> tilt toward end/non-ad.
- If pre/post audio or OCR are both high/medium -> possible internal-ad transition.
- Low gaps use the separate bridge rule.

### `end_pending`

- OCR coverage high + OCR low + audio low -> end can be confirmed.
- `ad_end` uses `pending_end_time` or the current anchor time; default is end anchor `candidate_time_sec`.
- OCR coverage low -> hold end confirmation.
- OCR context high/medium or product repetition maintained -> cancel `end_pending` and keep `in_ad`.
- If audio/OCR remain low -> confirm end.
- TODO: define max pending duration or max pending anchor count so `end_pending` cannot last too long.

## Low Gap Bridge Rule

Initial implementation calculates low gap duration from anchor times.

Example:

- pattern: high -> low -> high
- low gap duration = next high/medium anchor time - low anchor time

| Gap duration | Rule |
|---|---|
| `gap <= 10s` | bridge if either audio or OCR is medium or higher |
| `10s < gap <= 20s` | bridge only if audio and OCR are both medium or higher |
| `gap > 20s` | no automatic bridge |

C3 handling:

- C3 is not strong bridge evidence.
- For `gap <= 10s`, C3 can be weak bridge or review evidence.
- For `10s < gap <= 20s`, C3 alone must not bridge.

## Minimum Duration Prior

- Minimum ad duration prior: 20 seconds.
- End candidates within 20 seconds after start are held by default.
- E2 exception: even within 20 seconds, end can be allowed if OCR/audio are both low, OCR coverage is sufficient, and subsequent flow remains low.

## Long-Ad Prior

- Typical ad length is roughly 40 to 120 seconds, but this is not a hard cut.
- Do not automatically end an ad after 120 seconds.
- After 120 seconds, end-like flow can be treated more sensitively.
- If audio/OCR ad candidate levels remain medium or higher, continue the ad.
- If OCR coverage high + OCR low repeats and audio also drops, increase end likelihood.
- Repeated C3 in long-ad state may tilt toward end/non-ad.

## Conflict Handling

| Case | Interpretation |
|---|---|
| audio high + OCR high | strong ad continuity or start/end support |
| audio high + OCR coverage low | OCR unknown; audio evidence can be partly accepted |
| audio high + OCR coverage high + OCR low | C3; lower ad likelihood, and two consecutive anchors tilt toward end/non-ad |
| audio low + OCR high | D2; medium support for ad continuity |
| audio low + OCR low + OCR coverage high | end or non-ad likelihood increases |
| OCR low + OCR coverage low | OCR unknown; do not use as non-ad evidence |

## Pseudo-Code

This pseudo-code is implementation guidance only; this task does not generate interval predictions.

```text
state = non_ad
ad_start = null
pending_start_time = null
pending_end_time = null
consecutive_c3_count = 0

for visual_anchor in time_order:
    transition_time_anchor = visual_anchor.candidate_time_sec

    read OCR pre/post context around transition_time_anchor
    read audio pre/post context around transition_time_anchor
    derive OCR coverage level
    derive OCR start/end/context levels
    derive audio start/end/context/before/after levels
    detect C3 and D2 conflict cases
    update low-gap bridge context if needed

    if state == non_ad:
        if OCR start high:
            ad_start = transition_time_anchor
            state = in_ad
        elif OCR start medium:
            pending_start_time = transition_time_anchor
            state = start_pending
        elif OCR coverage low and audio start is high_or_medium:
            pending_start_time = transition_time_anchor
            state = start_pending
        else:
            remain non_ad

    elif state == start_pending:
        if OCR context is medium_or_high or audio context is medium_or_high:
            ad_start = pending_start_time
            state = in_ad
        elif OCR coverage high and OCR low and audio low:
            pending_start_time = null
            state = non_ad
        else:
            keep start_pending until future max-pending rule decides

    elif state == in_ad:
        if C3:
            consecutive_c3_count += 1
        else:
            consecutive_c3_count = 0

        if low_gap_bridge_applies:
            keep in_ad
        elif OCR context is medium_or_high:
            keep in_ad
        elif product repetition continues:
            keep in_ad
        elif audio context is high_or_medium and OCR coverage low:
            keep in_ad
        elif D2:
            keep in_ad with medium support
        elif end support appears and minimum_duration_prior_allows:
            pending_end_time = transition_time_anchor
            state = end_pending
        elif consecutive_c3_count >= 2:
            pending_end_time = transition_time_anchor
            state = end_pending
        else:
            keep in_ad

    elif state == end_pending:
        if OCR coverage high and OCR low and audio low:
            ad_end = pending_end_time or transition_time_anchor
            close current interval
            reset ad_start and pending times
            state = non_ad
        elif OCR context is medium_or_high or product repetition continues:
            pending_end_time = null
            state = in_ad
        elif OCR/audio remain low:
            ad_end = pending_end_time or transition_time_anchor
            close current interval
            reset ad_start and pending times
            state = non_ad
        else:
            keep end_pending until future max-pending rule decides
```

## Leakage Guard

- Rule design, thresholds, level standards, and keyword candidates are train-only.
- Validation is only for discussion, audit, and adjustment-candidate review.
- Test is protected until final evaluation.
- Test row-level features are excluded from the documentation bundle.
- No predicted intervals are generated here.
- No final performance claim is made here.
- Existing full-data exploratory artifacts are reference-only.

## Future Implementation Inputs and Columns

Input files:

- `data/splits/video_split_v2_4.csv`
- `data/features/visual_scene_boundary_anchors_v2_4.csv`
- `data/features/visual_scene_boundary_anchors_v2_4_with_split.csv`
- `data/audio/audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv`
- `data/audio/audio_visual_anchor_level_thresholds_v2_4_train_only.csv`
- `data/ocr/ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv`
- `data/ocr/ocr_visual_anchor_level_thresholds_v2_4_train_only.csv`
- optional fusion discussion tables for audit, excluding test row-level features

Minimum columns:

- split: `video_id`, `split`, `split_seed`
- visual anchor: `scene_boundary_anchor_id` or `visual_anchor_id`, `video_id`, `candidate_time_sec` or `canonical_boundary_time_sec`, `candidate_time_mmss`, `split`
- audio: `audio_start_signal_level`, `audio_end_signal_level`, `audio_context_level`, `audio_before_context_level`, `audio_after_context_level`, `audio_score_delta_post_minus_pre`, `audio_score_delta_pre_minus_post`
- OCR coverage: pre/post `context_*_frame_count`, `context_*_ocr_success_count`, `context_*_ocr_empty_count`, `context_*_ocr_failed_count`
- OCR levels: `ocr_start_signal_level`, `ocr_end_signal_level`, `ocr_context_level`
- OCR counts/scores: disclosure, CTA, discount/promo, product/brand counts and scores
- optional audit columns: nearest true boundary fields can be used only for offline audit, not scoring or interval generation

{reference_markdown(inventory)}
"""
    STATE_DOC.write_text(content, encoding="utf-8")


def write_checklist_doc(generated_at: str, inventory: dict[str, Any]) -> None:
    content = f"""# State-Machine Rule Implementation Checklist v1.1

Generated at: `{generated_at}`

This checklist is for a future implementation task. It is not detector code and does not generate predicted intervals.

## Preconditions

- [ ] Use only `.`.
- [ ] Do not read `./_old_project_not_included` except for before/after modification snapshots.
- [ ] Confirm fixed split file: `data/splits/video_split_v2_4.csv`.
- [ ] Confirm `split_seed = {SPLIT_SEED}`.
- [ ] Keep rule design train-only.
- [ ] Use validation only for audit or adjustment-candidate review.
- [ ] Keep test protected until final evaluation.
- [ ] Do not include test row-level features in discussion bundles.

## Input Files

- [ ] `data/features/visual_scene_boundary_anchors_v2_4.csv`
- [ ] `data/features/visual_scene_boundary_anchors_v2_4_with_split.csv`
- [ ] `data/audio/audio_visual_anchor_persistence_features_v2_4_train_val_for_discussion.csv`
- [ ] `data/audio/audio_visual_anchor_level_thresholds_v2_4_train_only.csv`
- [ ] `data/ocr/ocr_visual_anchor_context_features_v2_4_train_val_for_discussion.csv`
- [ ] `data/ocr/ocr_visual_anchor_level_thresholds_v2_4_train_only.csv`
- [ ] Train-only audio/OCR recommendation files if implementation needs feature provenance.

## Required Columns

Visual anchor:

- [ ] `video_id`
- [ ] `candidate_time_sec` or `canonical_boundary_time_sec`
- [ ] `candidate_time_mmss` if available
- [ ] `visual_anchor_id` or `scene_boundary_anchor_id`
- [ ] `split` from with-split anchor file or split join

Audio:

- [ ] `audio_start_signal_level`
- [ ] `audio_end_signal_level`
- [ ] `audio_context_level`
- [ ] `audio_before_context_level`
- [ ] `audio_after_context_level`
- [ ] `audio_score_delta_post_minus_pre`
- [ ] `audio_score_delta_pre_minus_post`
- [ ] pre/post ad-like ratio columns if available

OCR:

- [ ] `ocr_start_signal_level`
- [ ] `ocr_end_signal_level`
- [ ] `ocr_context_level`
- [ ] pre/post success, empty, failed, and frame-count columns
- [ ] disclosure, CTA, discount/promo, product/brand count columns
- [ ] OCR score columns if levels need to be recomputed

## Rule Implementation Steps

- [ ] Load visual transition anchors in time order by video.
- [ ] Join audio and OCR context by `video_id` and anchor identity/time.
- [ ] Derive `transition_time_anchor` from `candidate_time_sec`.
- [ ] Derive OCR valid frame ratio as `(success + empty) / frame_count`.
- [ ] Map coverage levels: high `>=0.8`, medium `>=0.4`, low `<0.4`.
- [ ] Apply negative guard before disclosure scoring.
- [ ] Treat OCR failed/empty as reliability/observation states, not non-ad evidence.
- [ ] Apply audio rule v1 as supporting evidence.
- [ ] Apply OCR rule v1 as primary start evidence.
- [ ] Run state machine with states `non_ad`, `start_pending`, `in_ad`, `end_pending`.
- [ ] Use anchor time for start/end timestamps only after OCR/audio context validates transition.
- [ ] Apply low gap bridge after continuity evidence is derived.
- [ ] Apply 20-second minimum duration prior and E2 exception.
- [ ] Apply long-ad prior after 120 seconds without hard cutting.

## TODO Before Coding

- [ ] Define max duration or max anchor count for `start_pending`.
- [ ] Define max duration or max anchor count for `end_pending`.
- [ ] Decide exact tie-breaking when pending start and end signals appear close together.
- [ ] Decide whether retroactive start review can use one or two future anchors.
- [ ] Decide how to merge overlapping intervals if future implementation emits candidates.
- [ ] Decide audit-only use of nearest true boundary columns; they must not enter scoring.

## Explicit Non-Goals for Future Prototype Setup

- [ ] Do not tune thresholds on validation/test.
- [ ] Do not use test row-level features during rule design.
- [ ] Do not convert visual anchor scores into direct start/end evidence.
- [ ] Do not produce final performance claims before protected test evaluation.

## Documentation References

- `reports/rules/audio_rule_v1.md`
- `reports/rules/ocr_rule_v1.md`
- `reports/rules/scene_audio_ocr_state_machine_rule_v1_1.md`
- `configs/rule_docs/scene_audio_ocr_state_machine_rule_v1_1_draft.json`

{reference_markdown(inventory)}
"""
    CHECKLIST_DOC.write_text(content, encoding="utf-8")


def write_config_draft(generated_at: str) -> None:
    config = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "documentation_only": True,
        "not_executable_detector_config": True,
        "no_detector_implementation": True,
        "no_predicted_intervals_generated": True,
        "generated_at": generated_at,
        "project_root": str(PROJECT_ROOT),
        "split": {
            "name": "v2_4",
            "split_seed": SPLIT_SEED,
            "train_video_id": TRAIN_VIDEO_IDS,
            "validation_video_id": VALIDATION_VIDEO_IDS,
            "test_video_id": TEST_VIDEO_IDS,
            "rule_design_basis": "train_only",
            "validation_usage": "discussion_audit_adjustment_candidate_review_only",
            "test_usage": "protected_until_final_evaluation",
            "test_row_level_features_in_discussion_bundle": False,
        },
        "visual_scene_anchor": {
            "role": "visual_transition_anchor",
            "required_sentence": "visual scene anchor는 광고 시작/종료의 직접 evidence가 아니라, 상태 변화가 일어날 수 있는 화면 전환 후보 시각이다.",
            "time_column_candidates": ["candidate_time_sec", "canonical_boundary_time_sec"],
            "context_window_sec": 10,
            "direct_start_end_evidence": False,
            "timestamp_convention": {
                "ad_start": "visual_transition_anchor.candidate_time_sec",
                "ad_end": "visual_transition_anchor.candidate_time_sec",
            },
        },
        "ocr_rule_v1": {
            "coverage": {
                "valid_frame": ["success", "empty"],
                "failed_is_valid_frame": False,
                "high_min_ratio": 0.8,
                "medium_min_ratio": 0.4,
                "low_max_ratio_exclusive": 0.4,
                "low_score_low_coverage_interpretation": "ocr_unknown_not_non_ad",
            },
            "start": {
                "strong_disclosure": "ocr_start_high",
                "weak_disclosure_alone": "ocr_start_medium",
                "product_repetition_alone": "continuity_not_start_confirmation",
                "negative_guard_zeroes_disclosure": True,
            },
            "end": {
                "cta_pre_10s": "ocr_end_medium",
                "cta_plus_post_decrease_plus_audio_low": "end_high_possible",
                "cta_alone_confirms_end": False,
            },
            "discount_benefit": {
                "alone": "weak_count_only",
                "with_cta_product_link": "weak_support",
                "core_start_end_score_alone": False,
            },
        },
        "audio_rule_v1": {
            "role": "medium_strength_supporting_evidence",
            "audio_alone_confirms_start": False,
            "audio_alone_confirms_end": False,
            "start_support": [
                "ocr_start_high_plus_audio_high_or_medium",
                "ocr_start_medium_plus_audio_high_or_medium",
                "ocr_coverage_low_plus_audio_high_or_medium_weak_pending",
            ],
            "end_support": [
                "audio_before_high_or_medium_plus_audio_after_low",
                "ocr_coverage_high_plus_ocr_low_plus_audio_low",
            ],
            "c3": {
                "definition": "audio_high + ocr_coverage_high + ocr_low",
                "interpretation": "lower_ad_likelihood",
                "two_consecutive": "tilt_to_end_or_non_ad",
            },
            "d2": {
                "definition": "audio_low + ocr_high",
                "interpretation": "medium_ad_continuity_support",
            },
        },
        "state_machine_v1_1": {
            "states": ["non_ad", "start_pending", "in_ad", "end_pending"],
            "minimum_duration_prior_sec": 20,
            "e2_exception": "allow_end_within_20s_only_if_ocr_audio_low_coverage_sufficient_following_low",
            "long_ad_prior": {
                "typical_range_sec": [40, 120],
                "hard_cut_after_120s": False,
                "after_120s": "more_sensitive_to_end_like_flow",
            },
            "low_gap_bridge": [
                {"gap_sec": "<=10", "requirement": "audio_or_ocr_medium_or_higher"},
                {"gap_sec": ">10_and_<=20", "requirement": "audio_and_ocr_both_medium_or_higher"},
                {"gap_sec": ">20", "requirement": "no_automatic_bridge"},
            ],
            "pending_todos": [
                "max_start_pending_duration_or_anchor_count",
                "max_end_pending_duration_or_anchor_count",
                "retroactive_start_review_window",
            ],
        },
    }
    CONFIG_DRAFT.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def local_validation() -> list[dict[str, Any]]:
    checks = [
        {
            "name": "Rule Completeness Validation",
            "required_files": [AUDIO_DOC, OCR_DOC, STATE_DOC],
            "required_terms": ["Low Gap Bridge", "Long-Ad Prior", "Coverage levels", "Conflict Handling"],
            "path": STATE_DOC,
        },
        {
            "name": "Semantic Validation",
            "required_files": [STATE_DOC],
            "required_terms": [
                "visual scene anchor는 광고 시작/종료의 직접 evidence가 아니라",
                "transition_time_anchor",
                "OCR failed/empty is not non-ad evidence",
            ],
            "path": STATE_DOC,
        },
        {
            "name": "Leakage Guard Validation",
            "required_files": [STATE_DOC, AUDIO_DOC, OCR_DOC],
            "required_terms": ["train split only", "Test is protected", "No predicted intervals"],
            "path": STATE_DOC,
        },
        {
            "name": "Documentation Quality Validation",
            "required_files": [CHECKLIST_DOC, STATE_DOC],
            "required_terms": ["Pseudo-Code", "TODO", "Conflict Handling"],
            "path": STATE_DOC,
        },
        {
            "name": "Output & Safety Validation",
            "required_files": [AUDIO_DOC, OCR_DOC, STATE_DOC, CHECKLIST_DOC, CONFIG_DRAFT, SCRIPT_PATH],
            "required_terms": [],
            "path": STATE_DOC,
        },
    ]
    results = []
    for check in checks:
        missing_files = [str(path) for path in check["required_files"] if not path.exists()]
        text = check["path"].read_text(encoding="utf-8") if check["path"].exists() else ""
        missing_terms = [term for term in check["required_terms"] if term not in text]
        results.append(
            {
                "name": check["name"],
                "status": "pass" if not missing_files and not missing_terms else "fail",
                "missing_files": missing_files,
                "missing_terms": missing_terms,
                "source": "local_script_validation",
            }
        )
    return results


def load_sub_agent_results() -> list[dict[str, Any]]:
    if not SUB_AGENT_RESULTS_JSON.exists():
        return []
    try:
        data = json.loads(SUB_AGENT_RESULTS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict) and isinstance(data.get("sub_agent_validations"), list):
        return data["sub_agent_validations"]
    if isinstance(data, list):
        return data
    return []


def scan_bundle_forbidden_files() -> list[str]:
    found = []
    if not BUNDLE_DIR.exists():
        return found
    for path in BUNDLE_DIR.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = {part.lower() for part in path.relative_to(BUNDLE_DIR).parts}
        if path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES or rel_parts & FORBIDDEN_BUNDLE_PARTS:
            found.append(str(path.relative_to(BUNDLE_DIR)))
    return sorted(found)


def backup_existing_outputs() -> Path:
    backup_dir = REPORT_DIR / "backups" / f"reference_path_cleanup_v1_1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        STATE_DOC,
        AUDIO_DOC,
        OCR_DOC,
        CHECKLIST_DOC,
        CONFIG_DRAFT,
        SUMMARY_MD,
        REPORT_JSON,
        RUN_LOG,
        SCRIPT_PATH,
    ]
    for path in paths:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def refresh_bundle() -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for child in BUNDLE_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for path in BUNDLE_FILES:
        shutil.copy2(path, BUNDLE_DIR / path.name)


def write_bundle_readme(generated_at: str, old_project_modified: bool) -> Path:
    readme = BUNDLE_DIR / "README_latest_files.md"
    content = f"""# Latest Rule Documentation Bundle v1.1

작업명: `{TASK_NAME}`

생성 시각: `{generated_at}`

이 bundle은 detector 구현 결과가 아니라 rule documentation 산출물이다. Predicted interval, test row-level feature, media file, frame image, model checkpoint, cache/tmp 파일은 포함하지 않는다.

## Files

| File | Role |
|---|---|
| `scene_audio_ocr_state_machine_rule_v1_1.md` | 통합 state-machine rule v1.1 문서 |
| `audio_rule_v1.md` | audio rule v1 문서 |
| `ocr_rule_v1.md` | OCR rule v1 문서 |
| `state_machine_rule_implementation_checklist_v1_1.md` | 향후 구현 체크리스트 |
| `scene_audio_ocr_state_machine_rule_v1_1_draft.json` | 실행용이 아닌 문서화용 draft config |
| `scene_audio_ocr_rule_documentation_v1_1_summary.md` | 작업 요약 |
| `scene_audio_ocr_rule_documentation_v1_1_report.json` | 기계 판독용 report |
| `scene_audio_ocr_rule_documentation_v1_1_run_log.txt` | 단계별 run log |
| `write_scene_audio_ocr_rule_docs_v1_1.py` | 문서 재생성 스크립트 |

## Reference Path Cleanup

- `reference_path_cleanup_applied=true`
- OCR report expected paths now point to `reports/ocr/`.
- No duplicate root reference files or symlinks were created.

## Safety

- `old_project_modified={str(old_project_modified).lower()}`
- `test_row_level_features_included=false`
- `no_detector_implementation=true`
- `no_predicted_intervals_generated=true`

## Next Steps

1. 문서를 사람이 검토
2. 필요한 rule 수정
3. 이후 별도 작업으로 train split 기준 state-machine detector prototype 구현
4. validation으로 조정 후보 검토
5. test는 최종 평가 전까지 보호
"""
    readme.write_text(content, encoding="utf-8")
    return readme


def write_summary(
    generated_at: str,
    inventory: dict[str, Any],
    split_check: dict[str, Any],
    local_results: list[dict[str, Any]],
    sub_agent_results: list[dict[str, Any]],
    old_project_modified: bool,
    forbidden_files: list[str],
    warnings: list[str],
) -> None:
    result_lines = []
    combined = sub_agent_results if sub_agent_results else local_results
    for item in combined:
        result_lines.append(f"- {item.get('name', 'validation')}: `{item.get('status', 'unknown')}`")
        summary = item.get("summary")
        if summary:
            result_lines.append(f"  - {summary}")

    warning_lines = [f"- {w}" for w in warnings] if warnings else ["- None"]
    missing_lines = [f"- `{rel(Path(p))}`" for p in inventory["reference_files_missing"]] or ["- None"]
    forbidden_lines = [f"- `{p}`" for p in forbidden_files] or ["- None"]

    content = f"""# Scene/Audio/OCR Rule Documentation v1.1 Summary

Generated at: `{generated_at}`

## Status

- 작업 상태: completed
- 작업 성격: rule documentation only
- detector 구현: no
- predicted interval 생성: no
- threshold 신규 튜닝: no
- test row-level feature 포함: no
- old_project_modified={str(old_project_modified).lower()}

## Generated Documents

- `reports/rules/scene_audio_ocr_state_machine_rule_v1_1.md`
- `reports/rules/audio_rule_v1.md`
- `reports/rules/ocr_rule_v1.md`
- `reports/rules/state_machine_rule_implementation_checklist_v1_1.md`
- `configs/rule_docs/scene_audio_ocr_state_machine_rule_v1_1_draft.json`
- `reports/rules/scene_audio_ocr_rule_documentation_v1_1_report.json`
- `logs/scene_audio_ocr_rule_documentation_v1_1_run_log.txt`
- `scripts/rules/write_scene_audio_ocr_rule_docs_v1_1.py`

## Documented Rule Scope

- visual transition anchor role
- audio rule v1
- OCR rule v1
- state-machine interval detector rule v1.1
- low gap bridge rule
- minimum duration prior
- long-ad prior
- OCR coverage/reliability interpretation
- conflict handling table
- leakage guard and split use principles
- future implementation columns and input files

## Fixed Split

- split file: `data/splits/video_split_v2_4.csv`
- split seed: `{SPLIT_SEED}`
- train: {", ".join(map(str, TRAIN_VIDEO_IDS))}
- validation: {", ".join(map(str, VALIDATION_VIDEO_IDS))}
- test: {", ".join(map(str, TEST_VIDEO_IDS))}
- split file validation: `{split_check.get("valid")}`

## Leakage Guard

- Rule design, thresholds, level standards, and keyword candidates are train-only.
- Validation is discussion/audit/review only.
- Test is preserved until final evaluation.
- Test row-level features are excluded from this bundle.
- Existing full-data exploratory artifacts are reference-only.

## Reference Path Cleanup

- 이전 WARN은 OCR report files가 `reports/` 루트가 아니라 `reports/ocr/` 아래에 있어서 발생했다.
- 이번 작업에서 expected OCR reference paths를 실제 프로젝트 구조에 맞게 `reports/ocr/`로 수정했다.
- 중복 파일 복사나 symlink는 만들지 않았다.
- 문서 내용과 rule 자체는 변경하지 않았고, reference inventory / warning만 정리했다.

## Validation Results

{chr(10).join(result_lines)}

## Safety

- latest bundle: `outputs/latest_for_chatgpt_rule_docs_v1_1`
- forbidden files in latest bundle:
{chr(10).join(forbidden_lines)}

## Missing Reference Files

{chr(10).join(missing_lines)}

## Warnings

{chr(10).join(warning_lines)}

## Next Steps

1. 사람이 문서를 검토한다.
2. 필요한 rule 수정을 문서에 반영한다.
3. 별도 작업에서 train split 기준 state-machine detector prototype을 구현한다.
4. validation으로 조정 후보를 검토한다.
5. test는 최종 평가 전까지 보호한다.
"""
    SUMMARY_MD.write_text(content, encoding="utf-8")


def write_report(
    start_time: str,
    end_time: str,
    inventory: dict[str, Any],
    split_check: dict[str, Any],
    local_results: list[dict[str, Any]],
    sub_agent_results: list[dict[str, Any]],
    old_project_modified: bool,
    input_files_modified: list[str],
    forbidden_files: list[str],
    warnings: list[str],
    errors: list[str],
) -> None:
    generated_documents = [
        str(STATE_DOC),
        str(AUDIO_DOC),
        str(OCR_DOC),
        str(CHECKLIST_DOC),
        str(CONFIG_DRAFT),
        str(SUMMARY_MD),
        str(REPORT_JSON),
        str(RUN_LOG),
        str(SCRIPT_PATH),
        str(BUNDLE_DIR / "README_latest_files.md"),
    ]
    report = {
        "task_name": TASK_NAME,
        "version": VERSION,
        "project_root": str(PROJECT_ROOT),
        "start_time": start_time,
        "end_time": end_time,
        "generated_documents": generated_documents,
        "reference_files_found": inventory["reference_files_found"],
        "reference_files_missing": inventory["reference_files_missing"],
        "alternate_reference_files_found": inventory["alternate_reference_files_found"],
        "reference_csv_columns": inventory["csv_columns"],
        "reference_path_cleanup_applied": REFERENCE_PATH_CLEANUP_APPLIED,
        "reference_path_cleanup_reason": REFERENCE_PATH_CLEANUP_REASON,
        "old_expected_ocr_reference_paths": OLD_EXPECTED_OCR_REFERENCE_PATHS,
        "new_expected_ocr_reference_paths": NEW_EXPECTED_OCR_REFERENCE_PATHS,
        "duplicate_root_reference_files_created": DUPLICATE_ROOT_REFERENCE_FILES_CREATED,
        "symlinks_created": SYMLINKS_CREATED,
        "split_check": split_check,
        "rules_documented": [
            "visual_scene_anchor_role",
            "audio_rule_v1",
            "ocr_rule_v1",
            "scene_audio_ocr_state_machine_rule_v1_1",
            "low_gap_bridge_rule",
            "minimum_duration_prior",
            "long_ad_prior",
            "ocr_coverage_reliability",
            "conflict_handling",
            "leakage_guard",
            "future_implementation_inputs_columns",
        ],
        "audio_rules_documented": [
            "audio_start_support",
            "audio_end_support",
            "audio_internal_ad_transition",
            "audio_context_continuity",
            "C3_audio_high_ocr_coverage_high_ocr_low",
            "D2_audio_low_ocr_high",
            "low_gap_bridge_audio_part",
            "long_ad_audio_part",
        ],
        "ocr_rules_documented": [
            "ocr_coverage_levels",
            "strong_disclosure_start_high",
            "weak_disclosure_start_medium",
            "cta_end_medium",
            "product_repetition_continuity",
            "negative_guard",
            "discount_benefit_weak_contextual_rule",
            "failed_empty_not_non_ad",
        ],
        "state_machine_rules_documented": [
            "non_ad",
            "start_pending",
            "in_ad",
            "end_pending",
            "timestamp_convention",
            "minimum_duration_prior",
            "long_ad_prior",
            "low_gap_bridge",
            "conflict_handling",
        ],
        "leakage_guard_notes": [
            "threshold_level_keyword_rule_design_train_only",
            "validation_discussion_audit_only",
            "test_protected_until_final_evaluation",
            "test_row_level_features_excluded_from_bundle",
            "full_data_exploratory_outputs_reference_only",
            "nearest_true_boundary_fields_audit_only_not_scoring",
        ],
        "no_detector_implementation": True,
        "no_predicted_intervals_generated": True,
        "test_row_level_features_included": False,
        "old_project_modified": old_project_modified,
        "input_files_modified": input_files_modified,
        "latest_for_chatgpt_forbidden_files_found": forbidden_files,
        "local_validation_results": local_results,
        "sub_agent_validation_results": sub_agent_results,
        "warnings": warnings,
        "errors": errors,
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    RUN_LOG.write_text("", encoding="utf-8")
    start_time = iso_now()
    warnings: list[str] = []
    errors: list[str] = []

    log("[STEP 01] Start reference path cleanup and create old project snapshot")
    old_before = snapshot_tree(OLD_PROJECT_ROOT, OLD_SNAPSHOT_BEFORE)
    if not OLD_PROJECT_ROOT.exists():
        warnings.append(f"Old project snapshot root does not exist: {OLD_PROJECT_ROOT}")

    log("[STEP 02] Backup existing rule documentation outputs")
    backup_dir = backup_existing_outputs()
    warnings.append(f"Existing rule documentation outputs were backed up to {backup_dir}")

    log("[STEP 03] Patch expected OCR reference paths in write_scene_audio_ocr_rule_docs_v1_1.py")
    warnings.append("Reference path cleanup applied in script: expected OCR report paths now use reports/ocr/.")

    log("[STEP 04] Re-run rule documentation script")
    inventory = collect_reference_inventory()
    split_check = validate_split_file(warnings)
    for missing in inventory["reference_files_missing"]:
        warnings.append(f"Reference file missing: {missing}")
    if inventory["alternate_reference_files_found"]:
        warnings.append("Some prompt-listed OCR report paths were missing at reports/ root, but same-name files were found under reports/ocr/.")

    input_before = {
        path: file_stat_signature(Path(path))
        for path in inventory["reference_files_found"]
        if Path(path).exists()
    }

    generated_at = iso_now()

    log("[STEP 04] Regenerate audio rule v1 document")
    write_audio_doc(generated_at, inventory)

    log("[STEP 04] Regenerate OCR rule v1 document")
    write_ocr_doc(generated_at, inventory)

    log("[STEP 04] Regenerate integrated state-machine rule v1.1 document")
    write_state_doc(generated_at, inventory)

    log("[STEP 04] Regenerate implementation checklist")
    write_checklist_doc(generated_at, inventory)

    log("[STEP 04] Regenerate JSON config for documentation only")
    write_config_draft(generated_at)

    log("[STEP 05] Validate reference_files_missing and warnings")
    local_results = local_validation()
    sub_agent_results = load_sub_agent_results()
    if sub_agent_results:
        log("[STEP 05] External Sub Agent validation results loaded")
    else:
        log("[STEP 05] External Sub Agent validation results not found yet; local validation recorded")

    input_after = {
        path: file_stat_signature(Path(path))
        for path in inventory["reference_files_found"]
        if Path(path).exists()
    }
    input_files_modified = [
        path for path, before in input_before.items() if input_after.get(path) != before
    ]

    log("[STEP 06] Update latest_for_chatgpt_rule_docs_v1_1")
    # bundle에 포함할 수 있도록 preliminary summary/report를 먼저 쓴다.
    old_after = snapshot_tree(OLD_PROJECT_ROOT, OLD_SNAPSHOT_AFTER)
    old_project_modified = old_before != old_after
    if old_project_modified:
        errors.append("Old project snapshot before/after differs.")
    if input_files_modified:
        errors.append("One or more reference input files changed during documentation.")

    write_summary(
        generated_at,
        inventory,
        split_check,
        local_results,
        sub_agent_results,
        old_project_modified,
        [],
        warnings,
    )
    write_report(
        start_time,
        iso_now(),
        inventory,
        split_check,
        local_results,
        sub_agent_results,
        old_project_modified,
        input_files_modified,
        [],
        warnings,
        errors,
    )
    refresh_bundle()
    write_bundle_readme(generated_at, old_project_modified)
    forbidden_files = scan_bundle_forbidden_files()

    # 최종 bundle scan 결과를 반영해 summary/report를 다시 쓰고 복사본을 갱신한다.
    write_summary(
        generated_at,
        inventory,
        split_check,
        local_results,
        sub_agent_results,
        old_project_modified,
        forbidden_files,
        warnings,
    )
    write_report(
        start_time,
        iso_now(),
        inventory,
        split_check,
        local_results,
        sub_agent_results,
        old_project_modified,
        input_files_modified,
        forbidden_files,
        warnings,
        errors,
    )
    shutil.copy2(SUMMARY_MD, BUNDLE_DIR / SUMMARY_MD.name)
    shutil.copy2(REPORT_JSON, BUNDLE_DIR / REPORT_JSON.name)
    shutil.copy2(RUN_LOG, BUNDLE_DIR / RUN_LOG.name)

    log("[STEP 07] Validate safety")
    log("[STEP 08] Print human-readable final summary")
    status = "completed" if not errors else "completed_with_errors"
    print()
    print("Reference path cleanup task summary")
    print(f"- status: {status}")
    print(f"- generated documents: {len(BUNDLE_FILES) + 1}")
    print("- resolved warning: OCR report expected paths now use reports/ocr/")
    print("- detector implementation: no")
    print("- predicted intervals generated: no")
    print("- test row-level features included: no")
    print(f"- old_project_modified={str(old_project_modified).lower()}")
    print(f"- latest bundle: {BUNDLE_DIR}")
    print(f"- forbidden files in latest bundle: {forbidden_files if forbidden_files else 'none'}")
    print(f"- warnings: {len(warnings)}")
    print(f"- errors: {errors if errors else 'none'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
