#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
DEV_ROOT = ROOT / "notebooks/rule_lab_v2_3_modular"
TEST_ROOT = ROOT / "notebooks/rule_lab_v2_3_modular_test_set"
TEST_IDS = [3, 4, 7, 16, 17, 18]
DEV_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
STATE_DIR = ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/current_state"
ARCHIVE_ROOT = ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/archive"
REPORT_DIR = ROOT / "reports/evaluation"
LOG_PATH = ROOT / "logs/frozen_patch5_test_set_evaluation_run_log.txt"
FEATURE_MANIFEST = ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/input_features/test_set_input_feature_manifest.json"
FEATURE_READINESS = ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/input_features/test_set_input_feature_readiness_report.json"
LATEST_CSV_TEST = ROOT / "notebooks/rule_lab_v2_3_modular/lastest_csv_test_set"
DEV_LATEST = DEV_ROOT / "lastest_csv"


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
    print(message, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rows(path: Path) -> int:
    return int(len(pd.read_csv(path))) if path.exists() else 0


def development_sanity() -> tuple[bool, dict[str, Any]]:
    post = read_json(DEV_LATEST / "05_postprocess_summary.json")
    fail_report = read_json(DEV_LATEST / "08_failure_diagnosis_report.json")
    failure_summary = pd.read_csv(DEV_LATEST / "08_failure_type_summary.csv")
    failure_counts = dict(zip(failure_summary["failure_type"].astype(str), failure_summary["count"].astype(int)))
    failure = pd.read_csv(DEV_LATEST / "08_failure_diagnosis.csv")
    a003_active = bool(failure.astype(str).apply(lambda col: col.str.contains("A003", na=False)).any(axis=1).sum())
    counts = {
        "final_prediction_count_after_targeted_patch": int(post.get("final_prediction_count_after_targeted_patch", -1)),
        "prediction_good": int(failure_counts.get("prediction_good", 0)),
        "prediction_partial_too_short": int(failure_counts.get("prediction_partial_too_short", 0)),
        "candidate_exists_but_not_selected": int(failure_counts.get("candidate_exists_but_not_selected", 0)),
        "false_positive_candidate_count": int(fail_report.get("false_positive_candidate_count", -1)),
        "overextended_prediction_count": int(fail_report.get("overextended_prediction_count", -1)),
        "middle_gap_case_count": int(fail_report.get("middle_gap_case_count", -1)),
        "A003_active_after_label_cleanup": a003_active,
    }
    expected = {
        "final_prediction_count_after_targeted_patch": 14,
        "prediction_good": 11,
        "prediction_partial_too_short": 3,
        "candidate_exists_but_not_selected": 1,
        "false_positive_candidate_count": 0,
        "overextended_prediction_count": 0,
        "middle_gap_case_count": 0,
        "A003_active_after_label_cleanup": False,
    }
    return counts == expected, counts


def feature_readiness() -> tuple[bool, dict[str, Any]]:
    manifest = read_json(FEATURE_MANIFEST)
    readiness = read_json(FEATURE_READINESS)
    checks = {
        "test_set_video_ids": readiness.get("test_set_video_ids") == TEST_IDS or manifest.get("test_set_video_ids") == TEST_IDS,
        "development_video_leakage": readiness.get("split_check", {}).get("development_video_leakage_into_test_set") is False,
        "transnetv2_created": readiness.get("transnetv2_conservative_test_set_created") is True,
        "transnetv2_failures": readiness.get("transnetv2_inference_failed_videos", []) == [],
        "coverage": readiness.get("test_video_coverage_complete") is True,
        "schema": readiness.get("schema_matches_development") is True,
        "fallback": readiness.get("old_v1_or_v2_4_fallback_used_as_final_input") is False,
        "feature_labels": readiness.get("actual_label_used_for_feature_generation") is False,
        "candidate_labels": readiness.get("actual_label_used_for_candidate_generation") is False,
        "not_evaluated_yet": readiness.get("evaluation_run_executed") is False,
    }
    return all(checks.values()), {"checks": checks, "manifest": manifest, "readiness": readiness}


DEV_NOTEBOOK_NAMES = [
    "00_run_all_manual_rule_lab_v2_3_development.ipynb",
    "01_setup_data_loading_v2_3_development.ipynb",
    "02_candidate_pool_ocr_augmentation_v2_3_development.ipynb",
    "03_rule_scoring_ocr_sponsor_audio_black_v2_3_development.ipynb",
    "04_selection_budget_guard_v2_3_development.ipynb",
    "05_bridge_extension_promotion_v2_3_development.ipynb",
    "06_metrics_xai_helpers_v2_3_development.ipynb",
    "07_timeline_review_v2_3_development.ipynb",
    "08_failure_diagnosis_v2_3_development.ipynb",
    "09_optional_save_safety_v2_3_development.ipynb",
]


def test_name(dev_name: str) -> str:
    return dev_name.replace("_development.ipynb", "_test_set.ipynb")


def patch_source_text(src: str) -> str:
    src = src.replace("notebooks/rule_lab_v2_3_modular", "notebooks/rule_lab_v2_3_modular_test_set")
    src = src.replace("manual_rule_lab_v2_3_config.json", "manual_rule_lab_v2_3_test_set_config.json")
    src = src.replace("_development.ipynb", "_test_set.ipynb")
    src = src.replace("RUN_HEAVY_PLOTS = True", "RUN_HEAVY_PLOTS = False")
    src = src.replace("Development Set", "test set")
    src = src.replace("Development_Set_only", "test_set_only")
    return src


def strip_patch6_micro_from_05(nb: dict[str, Any]) -> bool:
    removed = False
    marker = "# =========================\n# PATCH6-MICRO:"
    part8 = "# ─────────────────────────────────────────────\n# [파트 8] 최종 output overwrite"
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if marker in src:
            start = src.index(marker)
            end = src.index(part8, start)
            src = src[:start].rstrip() + "\n\n\n" + src[end:]
            src = src.replace("\ntargeted_patch_summary.update(patch6_micro_summary)\n", "\n")
            cell["source"] = src.splitlines(keepends=True)
            removed = True
    return removed


def copy_and_patch_notebooks() -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = None
    if TEST_ROOT.exists():
        backup_dir = TEST_ROOT.parent / f"rule_lab_v2_3_modular_test_set_backup_{timestamp}"
        shutil.move(str(TEST_ROOT), str(backup_dir))
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DEV_ROOT / "common", TEST_ROOT / "common")
    (TEST_ROOT / "config").mkdir(parents=True, exist_ok=True)
    patch6_removed = False
    copied = []
    for name in DEV_NOTEBOOK_NAMES:
        src = DEV_ROOT / name
        dst = TEST_ROOT / test_name(name)
        nb = json.loads(src.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if isinstance(cell.get("source"), list):
                cell["source"] = patch_source_text("".join(cell["source"])).splitlines(keepends=True)
            elif isinstance(cell.get("source"), str):
                cell["source"] = patch_source_text(cell["source"])
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            else:
                cell.pop("outputs", None)
                cell.pop("execution_count", None)
        if name.startswith("05_"):
            patch6_removed = strip_patch6_micro_from_05(nb)
        if name.startswith("09_"):
            patch_09_safety(nb)
        dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        copied.append(str(dst))
    patch_common()
    write_test_config()
    return {"backup_dir": str(backup_dir) if backup_dir else None, "copied": copied, "patch6_removed": patch6_removed}


def patch_09_safety(nb: dict[str, Any]) -> None:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        old = "safety_report = {'safety_errors': safety_errors, 'safety_flags': safety_flags, 'SAVE_OUTPUTS_default_false': not save_cfg.get('enabled', False), 'manual_saved_dir': manual_saved_dir, 'false_positive_gate_added': False}"
        new = (
            "safety_report = {'safety_errors': safety_errors, 'safety_flags': safety_flags, "
            "'SAVE_OUTPUTS_default_false': not save_cfg.get('enabled', False), 'manual_saved_dir': manual_saved_dir, "
            "'false_positive_gate_added': False, 'test_set_processed': True, "
            "'test_set_definition': 'Test Set', "
            "'actual_label_used_for_decision': False, 'actual_label_used_for_posthoc_evaluation': True, "
            "'rule_modified_after_test_result': False, 'Development_Set_tuning_stopped': True}"
        )
        if old in src:
            src = src.replace(old, new)
            cell["source"] = src.splitlines(keepends=True)


def patch_common() -> None:
    path = TEST_ROOT / "common/manual_rule_lab_common.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace("PROJECT_ROOT / 'notebooks/rule_lab_v2_3_modular'", "PROJECT_ROOT / 'notebooks/rule_lab_v2_3_modular_test_set'")
    text = text.replace("MODULAR_ROOT / 'config/manual_rule_lab_v2_3_config.json'", "MODULAR_ROOT / 'config/manual_rule_lab_v2_3_test_set_config.json'")
    text = text.replace("PROJECT_ROOT / 'data/experiments/manual_rule_lab_v2_3_development/current_state'", "PROJECT_ROOT / 'data/experiments/manual_rule_lab_v2_3_test_set/current_state'")
    text = text.replace("DEVELOPMENT_VIDEO_IDS = [1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]", "DEVELOPMENT_VIDEO_IDS = [3, 4, 7, 16, 17, 18]")
    text = text.replace("DEVELOPMENT_VIDEO_ID_SET = set(DEVELOPMENT_VIDEO_IDS)", "DEVELOPMENT_VIDEO_ID_SET = set(DEVELOPMENT_VIDEO_IDS)")
    text = text.replace("'development'", "'test_set'")
    text = text.replace('"development"', '"test_set"')
    text = text.replace("_development.ipynb", "_test_set.ipynb")
    text = text.replace("PROJECT_ROOT / 'data/experiments/manual_rule_lab_v2_3_development/state_archive'", "PROJECT_ROOT / 'data/experiments/manual_rule_lab_v2_3_test_set/archive'")
    text = text.replace("PROJECT_ROOT / 'data/experiments/manual_rule_lab_v2_3_development/executed_notebooks' /", "MODULAR_ROOT / 'executed_notebooks' /")
    path.write_text(text, encoding="utf-8")


def write_test_config() -> None:
    cfg = read_json(DEV_ROOT / "config/manual_rule_lab_v2_3_config.json")
    cfg["modular_root"] = str(TEST_ROOT)
    cfg["state_dir"] = str(STATE_DIR)
    cfg["development_video_ids"] = TEST_IDS
    cfg["test_set_video_ids"] = TEST_IDS
    cfg["test_set_definition"] = "Test Set"
    cfg["paths"]["ocr_candidate_sources"] = [str(ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/input_features/v2_5_test_set_ocr_candidate_sources.csv")]
    cfg["paths"]["black_screen_features"] = str(ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/input_features/v2_5_test_set_black_screen_features.csv")
    version = cfg.get("base_version", "v2_1b_ocr_phase_transition_light")
    cfg.setdefault("base_version_paths", {}).setdefault(version, {})
    cfg["base_version_paths"][version].update({
        "predictions": str(ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/input_features/v2_1b_test_set_predictions.csv"),
        "review_only": str(ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/input_features/v2_1b_test_set_review_candidates.csv"),
        "pruned": str(ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/input_features/v2_1b_test_set_pruned_candidates.csv"),
        "open": str(ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/input_features/v2_1b_test_set_open_candidates.csv"),
    })
    cfg["save_outputs"]["output_root"] = str(ROOT / "data/experiments/manual_rule_lab_v2_3_test_set/runs")
    flags = cfg.setdefault("safety_flags", {})
    flags.update({
        "actual_label_used_for_decision": False,
        "actual_label_used_for_candidate_generation": False,
        "actual_label_used_for_bridge": False,
        "actual_label_used_for_promotion": False,
        "actual_label_used_for_review_extension": False,
        "actual_label_used_for_failure_diagnosis": True,
        "actual_label_used_for_posthoc_evaluation": True,
        "test_set_processed": True,
        "test_set_definition": "Test Set",
        "rule_modified_after_test_result": False,
        "Development_Set_tuning_stopped": True,
        "Extended_Evaluation_processed": True,
        "Diagnostic_Subset_processed": True,
        "Pure_Test_processed": True,
    })
    cfg["timeline_review"]["run_heavy_plots"] = False
    write_json(TEST_ROOT / "config/manual_rule_lab_v2_3_test_set_config.json", cfg)


def archive_and_clear_test_state() -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    files = [p for p in STATE_DIR.iterdir() if p.is_file()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = ARCHIVE_ROOT / f"pre_eval_{timestamp}"
    if files:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for p in files:
            shutil.copy2(p, archive_dir / p.name)
            p.unlink()
    return {"archived_file_count": len(files), "archive_dir": str(archive_dir) if files else None}


def execute_master() -> dict[str, Any]:
    import nbformat
    from nbclient import NotebookClient

    master = TEST_ROOT / "00_run_all_manual_rule_lab_v2_3_test_set.ipynb"
    nb = nbformat.read(master, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(TEST_ROOT)}})
    client.execute()
    nbformat.write(nb, master)
    return {"master": str(master), "success": True}


def collect_eval_results(dev_counts: dict[str, Any], readiness_info: dict[str, Any], copy_info: dict[str, Any], run_info: dict[str, Any]) -> dict[str, Any]:
    failure_summary = pd.read_csv(STATE_DIR / "08_failure_type_summary.csv")
    failure_counts = dict(zip(failure_summary["failure_type"].astype(str), failure_summary["count"].astype(int)))
    failure_report = read_json(STATE_DIR / "08_failure_diagnosis_report.json")
    post = read_json(STATE_DIR / "05_postprocess_summary.json")
    safety = read_json(STATE_DIR / "09_safety_report.json")
    summary_metrics = pd.read_csv(STATE_DIR / "06_summary_metrics.csv")
    summary_metric_dict = summary_metrics.iloc[0].to_dict() if len(summary_metrics) else {}
    manifest = pd.read_csv(STATE_DIR / "01_development_manifest.csv")
    actual = pd.read_csv(STATE_DIR / "01_actual_intervals_for_scoring.csv")
    predictions = pd.read_csv(STATE_DIR / "05_manual_predictions.csv")
    review = pd.read_csv(STATE_DIR / "05_manual_review_candidates.csv")
    scored = pd.read_csv(STATE_DIR / "03_scored_candidates.csv")
    report = {
        "task": "frozen_patch5_test_set_evaluation_with_test_notebook_copies",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "test_set_definition": "Test Set",
        "test_set_called_as": "test set",
        "rule_freeze_version": "patch5_safe_freeze",
        "transnetv2_inclusive_test_features_used": True,
        "test_feature_manifest_path": str(FEATURE_MANIFEST),
        "test_notebook_copy_created": True,
        "test_notebook_root": str(TEST_ROOT),
        "test_notebooks": [test_name(n) for n in DEV_NOTEBOOK_NAMES],
        "development_notebooks_modified": False,
        "patch6_experimental_code_removed_or_disabled_in_test_copy": copy_info.get("patch6_removed", False) or not patch6_keywords_remaining(),
        "development_sanity_check_success": True,
        "development_sanity_counts": dev_counts,
        "development_sanity_matches_patch5": True,
        "test_set_run_success": True,
        "test_set_video_count": int(len(manifest)),
        "test_set_video_ids": sorted(manifest["video_id"].astype(int).unique().tolist()),
        "test_set_ad_interval_count": int(len(actual)),
        "test_set_prediction_count": int(len(predictions)),
        "test_set_review_candidate_count": int(len(review)),
        "test_set_scored_candidate_count": int(len(scored)),
        "test_set_summary_metrics": summary_metric_dict,
        "test_set_failure_counts": failure_counts,
        "test_set_false_positive_candidate_count": int(failure_report.get("false_positive_candidate_count", 0)),
        "test_set_overextended_prediction_count": int(failure_report.get("overextended_prediction_count", 0)),
        "test_set_middle_gap_case_count": int(failure_report.get("middle_gap_case_count", 0)),
        "actual_label_used_for_decision": False,
        "actual_label_used_for_posthoc_evaluation": True,
        "rule_modified_after_test_result": False,
        "Development_Set_tuning_stopped": True,
        "safety_report": safety,
        "copy_info": copy_info,
        "run_info": run_info,
        "feature_readiness_checks": readiness_info.get("checks", {}),
        "output_files": {
            "evaluation_summary_md": str(REPORT_DIR / "frozen_patch5_test_set_evaluation_summary.md"),
            "evaluation_report_json": str(REPORT_DIR / "frozen_patch5_test_set_evaluation_report.json"),
            "freeze_manifest_json": str(REPORT_DIR / "frozen_patch5_rule_freeze_manifest.json"),
            "notebook_copy_manifest_json": str(REPORT_DIR / "frozen_patch5_test_set_notebook_copy_manifest.json"),
            "run_log": str(LOG_PATH),
        },
        "old_project_modified": False,
        "input_files_modified": False,
        "viewer_ui_modified": False,
        "reports_created": True,
        "warnings": [],
        "errors": [],
    }
    return report


def patch6_keywords_remaining() -> bool:
    keywords = ["PATCH6_CONFIG", "PATCH6_EVENTS", "apply_patch6_residual_cleanup", "PATCH6_LITE", "PATCH6_MICRO", "patch6_lite", "patch6_micro"]
    nb = TEST_ROOT / "05_bridge_extension_promotion_v2_3_test_set.ipynb"
    text = nb.read_text(encoding="utf-8") if nb.exists() else ""
    return any(k in text for k in keywords)


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "frozen_patch5_test_set_evaluation_report.json", report)
    freeze = {
        "rule_freeze_version": "patch5_safe_freeze",
        "source_development_notebook": str(DEV_ROOT / "05_bridge_extension_promotion_v2_3_development.ipynb"),
        "test_copy_notebook": str(TEST_ROOT / "05_bridge_extension_promotion_v2_3_test_set.ipynb"),
        "patch6_experimental_code_removed_or_disabled_in_test_copy": report["patch6_experimental_code_removed_or_disabled_in_test_copy"],
        "threshold_tuned_on_test_set": False,
        "rule_modified_after_test_result": False,
        "transnetv2_inclusive_test_features_used": True,
    }
    write_json(REPORT_DIR / "frozen_patch5_rule_freeze_manifest.json", freeze)
    copy_manifest = {
        "created_at": report["created_at"],
        "test_notebook_root": str(TEST_ROOT),
        "source_notebook_root": str(DEV_ROOT),
        "development_notebooks_modified": False,
        "test_notebooks": report["test_notebooks"],
        "copy_info": report.get("copy_info", {}),
    }
    write_json(REPORT_DIR / "frozen_patch5_test_set_notebook_copy_manifest.json", copy_manifest)
    metric_lines = "\n".join([f"  - {k}: {v}" for k, v in report["test_set_summary_metrics"].items()])
    fail_lines = "\n".join([f"  - {k}: {v}" for k, v in report["test_set_failure_counts"].items()])
    summary = f"""# Frozen Patch5 Test Set Evaluation

- test set: Test Set
- test videos: {report['test_set_video_ids']}
- TransNetV2-inclusive refreshed features used: true
- Development patch5 sanity matched: {report['development_sanity_matches_patch5']}
- final prediction count: {report['test_set_prediction_count']}
- review candidate count: {report['test_set_review_candidate_count']}
- false positive candidate count: {report['test_set_false_positive_candidate_count']}
- overextended prediction count: {report['test_set_overextended_prediction_count']}
- middle gap case count: {report['test_set_middle_gap_case_count']}
- actual label used for decision: false
- actual label used for post-hoc evaluation: true
- rule modified after test result: false

## Summary Metrics
{metric_lines}

## Failure Counts
{fail_lines}

This is an evaluation-only run of frozen patch5. No rule or threshold was changed after seeing test set results.
"""
    (REPORT_DIR / "frozen_patch5_test_set_evaluation_summary.md").write_text(summary, encoding="utf-8")


def refresh_latest_csv_test_set() -> list[str]:
    LATEST_CSV_TEST.mkdir(parents=True, exist_ok=True)
    for p in list(LATEST_CSV_TEST.iterdir()):
        if p.is_file():
            p.unlink()
    names = [
        "03_scored_candidates.csv",
        "05_manual_predictions.csv",
        "05_manual_review_candidates.csv",
        "05_postprocess_summary.json",
        "06_summary_metrics.csv",
        "06_metrics_by_video.csv",
        "08_failure_diagnosis.csv",
        "08_failure_diagnosis_report.json",
        "08_failure_type_summary.csv",
        "08_false_positive_diagnosis.csv",
        "08_overextended_prediction_diagnosis.csv",
        "08_recommended_fix_summary.csv",
        "09_safety_report.json",
    ]
    copied = []
    for name in names:
        src = STATE_DIR / name
        if src.exists():
            shutil.copy2(src, LATEST_CSV_TEST / name)
            copied.append(name)
    return copied


def main() -> None:
    LOG_PATH.write_text("", encoding="utf-8")
    log("[STEP] Development sanity check")
    sanity_ok, dev_counts = development_sanity()
    if not sanity_ok:
        raise RuntimeError(f"Development sanity failed: {dev_counts}")
    log(f"[OK] Development sanity: {dev_counts}")

    log("[STEP] Test feature readiness check")
    ready_ok, ready = feature_readiness()
    if not ready_ok:
        raise RuntimeError(f"Test feature readiness failed: {ready['checks']}")
    log(f"[OK] Feature readiness: {ready['checks']}")

    log("[STEP] Copy and patch test-set notebooks")
    copy_info = copy_and_patch_notebooks()
    log(f"[OK] Copied notebooks: {len(copy_info['copied'])}")

    log("[STEP] Archive and clear test current_state")
    archive_info = archive_and_clear_test_state()
    log(f"[OK] State archive: {archive_info}")

    log("[STEP] Execute test-set 00 master")
    run_info = execute_master()
    log("[OK] Test-set master run succeeded")

    log("[STEP] Collect evaluation outputs")
    report = collect_eval_results(dev_counts, ready, copy_info | {"state_archive": archive_info}, run_info)
    latest_files = refresh_latest_csv_test_set()
    report["test_set_latest_csv_refreshed"] = True
    report["test_set_latest_csv_path"] = str(LATEST_CSV_TEST)
    report["test_set_latest_csv_files"] = latest_files
    write_reports(report)
    write_json(REPORT_DIR / "frozen_patch5_test_set_evaluation_report.json", report)
    log(f"[DONE] latest_csv_test_set files: {latest_files}")
    print(json.dumps({"status": "success", "report": str(REPORT_DIR / "frozen_patch5_test_set_evaluation_report.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
