#!/usr/bin/env python3
"""Initialize the isolated youtube_ad_segment_detector project.

This script intentionally uses only Python's standard library so the project can
be initialized without installing dependencies outside the project root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import posixpath
import re
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(".")
OLD_PROJECT_ROOT = Path("./_old_project_not_included")
INPUT_LABEL_FILE = PROJECT_ROOT / "new_ad_labeling.xlsx"
RAW_LABEL_COPY = PROJECT_ROOT / "data/raw_labels/new_ad_labeling.xlsx"
CLEAN_LABEL_CSV = PROJECT_ROOT / "data/labels/clean_ad_labels_v0.csv"
REVIEW_XLSX = PROJECT_ROOT / "data/labels/clean_ad_labels_v0_review.xlsx"
README_PATH = PROJECT_ROOT / "README.md"
MANIFEST_PATH = PROJECT_ROOT / "project_manifest.json"
REPORT_PATH = PROJECT_ROOT / "reports/initialize_project_v1_report.json"
SUMMARY_PATH = PROJECT_ROOT / "reports/initialize_project_v1_summary.md"
LOG_PATH = PROJECT_ROOT / "logs/initialize_project_v1_run_log.txt"
LATEST_DIR = PROJECT_ROOT / "outputs/latest_for_chatgpt"
LATEST_README = LATEST_DIR / "README_latest_files.md"

FOLDERS = [
    "data",
    "data/raw_labels",
    "data/labels",
    "data/videos",
    "data/frames",
    "data/features",
    "data/features/scene",
    "data/features/ocr",
    "data/features/audio",
    "data/features/resnet",
    "data/splits",
    "scripts",
    "scripts/labels",
    "scripts/features",
    "scripts/rules",
    "scripts/evaluation",
    "scripts/utils",
    "configs",
    "configs/rules",
    "configs/features",
    "workspaces",
    "workspaces/label_pipeline_v1",
    "workspaces/scene_pipeline_v1",
    "workspaces/ocr_pipeline_v1",
    "workspaces/rule_detector_v1",
    "workspaces/evaluation_v1",
    "runs",
    "outputs",
    "outputs/latest_for_chatgpt",
    "reports",
    "logs",
    "docs",
    "notebooks",
]

CLEAN_LABEL_COLUMNS = [
    "ad_interval_id",
    "video_id",
    "video_title",
    "ad_start_min",
    "ad_start_sec_part",
    "ad_end_min",
    "ad_end_sec_part",
    "ad_start_sec",
    "ad_end_sec",
    "ad_duration_sec",
    "is_abrupt_transition_ad",
    "label_valid",
    "invalid_reason",
    "source_file",
    "source_sheet",
    "source_row_number",
]

XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

RUN_LOG: list[str] = []


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    RUN_LOG.append(f"[{timestamp}] {message}")


def project_path(relative_path: str | Path) -> Path:
    path = PROJECT_ROOT / relative_path
    resolved = path.resolve()
    root_resolved = PROJECT_ROOT.resolve()
    if not str(resolved).startswith(str(root_resolved) + os.sep) and resolved != root_resolved:
        raise ValueError(f"Refusing to write outside project root: {resolved}")
    return path


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_header(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"[\s_\-\(\)\[\]\{\}:：/\\]+", "", text)
    return text


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) if isinstance(value, float) else False:
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def as_output_number(value: float | None) -> int | float | str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return round(value, 3)


def column_letter_to_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref.upper())
    if not letters:
        return 1
    result = 0
    for char in letters.group(0):
        result = result * 26 + ord(char) - ord("A") + 1
    return result


def column_index_to_letter(index: int) -> str:
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters or "A"


def parse_float_text(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall(f"{{{XLSX_MAIN_NS}}}si"):
        parts = [node.text or "" for node in si.findall(f".//{{{XLSX_MAIN_NS}}}t")]
        strings.append("".join(parts))
    return strings


def read_workbook_sheets(zf: zipfile.ZipFile) -> list[dict[str, str]]:
    workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
    rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    relationships = {}
    for rel in rel_root.findall(f"{{{PKG_REL_NS}}}Relationship"):
        relationships[rel.attrib["Id"]] = rel.attrib["Target"]

    sheets = []
    for sheet in workbook_root.findall(f".//{{{XLSX_MAIN_NS}}}sheet"):
        rel_id = sheet.attrib.get(f"{{{XLSX_REL_NS}}}id")
        target = relationships.get(rel_id or "", "")
        if target.startswith("/"):
            path = target.lstrip("/")
        else:
            path = posixpath.normpath(posixpath.join("xl", target))
        sheets.append(
            {
                "name": sheet.attrib.get("name", ""),
                "sheet_id": sheet.attrib.get("sheetId", ""),
                "path": path,
            }
        )
    return sheets


def get_cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        parts = [node.text or "" for node in cell.findall(f".//{{{XLSX_MAIN_NS}}}t")]
        return "".join(parts)

    value_node = cell.find(f"{{{XLSX_MAIN_NS}}}v")
    value_text = "" if value_node is None or value_node.text is None else value_node.text

    if cell_type == "s":
        index = int(value_text) if value_text.strip().isdigit() else -1
        return shared_strings[index] if 0 <= index < len(shared_strings) else ""
    if cell_type == "str":
        return value_text
    if cell_type == "b":
        return value_text == "1"
    if value_text == "":
        return ""
    number = parse_float_text(value_text)
    if number is None:
        return value_text
    if number.is_integer():
        return int(number)
    return number


def read_worksheet_rows(zf: zipfile.ZipFile, sheet_path: str, shared_strings: list[str]) -> list[dict[str, Any]]:
    root = ET.fromstring(zf.read(sheet_path))
    rows: list[dict[str, Any]] = []
    for row in root.findall(f".//{{{XLSX_MAIN_NS}}}row"):
        row_number = int(float(row.attrib.get("r", str(len(rows) + 1))))
        values: dict[int, Any] = {}
        for cell in row.findall(f"{{{XLSX_MAIN_NS}}}c"):
            cell_ref = cell.attrib.get("r", "")
            column_index = column_letter_to_index(cell_ref) if cell_ref else len(values) + 1
            values[column_index] = get_cell_value(cell, shared_strings)
        rows.append({"row_number": row_number, "values": values})
    return rows


def header_candidates(header: Any) -> set[str]:
    h = normalize_header(header)
    candidates: set[str] = set()
    if not h:
        return candidates

    has_video = "영상" in h or "video" in h
    has_ad = "광고" in h or h.startswith("ad") or "ad" in h
    has_start = "시작" in h or "start" in h
    has_end = "종료" in h or "끝" in h or "end" in h
    has_min = "분" in h or "min" in h or "minute" in h
    has_sec = "초" in h or "sec" in h or "second" in h
    has_time = "시간" in h or "time" in h

    if ("영상번호" in h) or ("videono" in h) or ("videonumber" in h) or ("videoid" in h):
        candidates.add("video_id")
    if (has_video and ("제목" in h or "title" in h)) or h in {"제목", "title", "videotitle"}:
        candidates.add("video_title")
    if has_ad and has_start and has_min:
        candidates.add("ad_start_min")
    if has_ad and has_start and has_sec:
        candidates.add("ad_start_sec_part")
    if has_ad and has_end and has_min:
        candidates.add("ad_end_min")
    if has_ad and has_end and has_sec:
        candidates.add("ad_end_sec_part")
    if has_ad and has_start and not has_min and not has_sec:
        candidates.add("ad_start_combined")
    if has_ad and has_end and not has_min and not has_sec:
        candidates.add("ad_end_combined")
    if h in {"adstart", "광고시작", "광고시작시간"} or (has_ad and has_start and has_time and not has_min and not has_sec):
        candidates.add("ad_start_combined")
    if h in {"adend", "광고종료", "광고종료시간"} or (has_ad and has_end and has_time and not has_min and not has_sec):
        candidates.add("ad_end_combined")
    if "abrupt" in h or "전환형" in h or "갑작" in h:
        candidates.add("is_abrupt_transition_ad")
    return candidates


def infer_mapping_from_header(values: dict[int, Any]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    mapping: dict[str, int] = {}
    ambiguous: list[dict[str, Any]] = []
    claims: dict[str, list[tuple[int, str]]] = {}

    for column_index, value in values.items():
        candidates = header_candidates(value)
        if not candidates:
            continue
        if len(candidates) > 1:
            ambiguous.append(
                {
                    "column_index": column_index,
                    "column_letter": column_index_to_letter(column_index),
                    "header": clean_text(value),
                    "candidate_meanings": sorted(candidates),
                }
            )
            continue
        candidate = next(iter(candidates))
        claims.setdefault(candidate, []).append((column_index, clean_text(value)))

    for canonical, matches in claims.items():
        if len(matches) == 1:
            mapping[canonical] = matches[0][0]
        else:
            ambiguous.append(
                {
                    "canonical_column": canonical,
                    "candidate_columns": [
                        {
                            "column_index": column_index,
                            "column_letter": column_index_to_letter(column_index),
                            "header": header,
                        }
                        for column_index, header in matches
                    ],
                }
            )

    return mapping, ambiguous


def mapping_score(mapping: dict[str, int]) -> int:
    score = 0
    if "video_id" in mapping:
        score += 3
    if "video_title" in mapping:
        score += 3
    if {"ad_start_min", "ad_start_sec_part"}.issubset(mapping) or "ad_start_combined" in mapping:
        score += 3
    if {"ad_end_min", "ad_end_sec_part"}.issubset(mapping) or "ad_end_combined" in mapping:
        score += 3
    if "is_abrupt_transition_ad" in mapping:
        score += 1
    return score


def select_main_sheet(zf: zipfile.ZipFile) -> dict[str, Any]:
    shared_strings = read_shared_strings(zf)
    sheets = read_workbook_sheets(zf)
    analyses = []
    for sheet in sheets:
        rows = read_worksheet_rows(zf, sheet["path"], shared_strings)
        best: dict[str, Any] | None = None
        for row in rows[:20]:
            mapping, ambiguous = infer_mapping_from_header(row["values"])
            score = mapping_score(mapping)
            candidate = {
                "sheet": sheet,
                "rows": rows,
                "header_row_number": row["row_number"],
                "mapping": mapping,
                "ambiguous": ambiguous,
                "score": score,
            }
            if best is None or score > best["score"]:
                best = candidate
        if best is not None:
            analyses.append(best)

    if not analyses:
        return {
            "sheet": {"name": "", "path": ""},
            "rows": [],
            "header_row_number": None,
            "mapping": {},
            "ambiguous": [],
            "score": 0,
            "all_sheets": sheets,
            "sheet_analyses": [],
        }

    selected = max(analyses, key=lambda item: item["score"])
    selected["all_sheets"] = sheets
    selected["sheet_analyses"] = [
        {
            "sheet_name": item["sheet"]["name"],
            "header_row_number": item["header_row_number"],
            "score": item["score"],
            "detected_mapping": mapping_to_headers(item["mapping"], item["rows"], item["header_row_number"]),
            "ambiguous_columns": item["ambiguous"],
        }
        for item in analyses
    ]
    return selected


def mapping_to_headers(mapping: dict[str, int], rows: list[dict[str, Any]], header_row_number: int | None) -> dict[str, str]:
    header_values: dict[int, Any] = {}
    for row in rows:
        if row["row_number"] == header_row_number:
            header_values = row["values"]
            break
    return {
        canonical: clean_text(header_values.get(column_index, ""))
        for canonical, column_index in sorted(mapping.items(), key=lambda item: item[0])
    }


def parse_combined_time(value: Any) -> tuple[float | None, float | None, float | None, str | None]:
    if is_blank(value):
        return None, None, None, "blank_time"
    text = clean_text(value)

    colon_match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)(?::(\d+(?:\.\d+)?))(?:[:：](\d+(?:\.\d+)?))?\s*", text)
    if colon_match:
        first = float(colon_match.group(1))
        second = float(colon_match.group(2))
        third = colon_match.group(3)
        if third is None:
            total = first * 60 + second
        else:
            total = first * 3600 + second * 60 + float(third)
        minute = math.floor(total / 60)
        second_part = total - minute * 60
        return float(minute), second_part, total, None

    minute_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:분|min|minute)", text, re.IGNORECASE)
    second_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:초|sec|second)", text, re.IGNORECASE)
    if minute_match or second_match:
        minute = float(minute_match.group(1)) if minute_match else 0.0
        second_part = float(second_match.group(1)) if second_match else 0.0
        total = minute * 60 + second_part
        return minute, second_part, total, None

    numeric = to_number(value)
    if numeric is not None:
        minute = math.floor(numeric / 60)
        second_part = numeric - minute * 60
        return float(minute), second_part, numeric, "combined_numeric_assumed_seconds"

    return None, None, None, "unparseable_time"


def parse_separate_time(min_value: Any, sec_value: Any) -> tuple[float | None, float | None, float | None, list[str]]:
    warnings: list[str] = []
    minute = to_number(min_value)
    second_part = to_number(sec_value)
    if minute is None or second_part is None:
        return minute, second_part, None, warnings
    if second_part >= 60:
        warnings.append("second_part_gte_60")
    total = minute * 60 + second_part
    return minute, second_part, total, warnings


def row_value(row: dict[str, Any], mapping: dict[str, int], column_name: str) -> Any:
    column_index = mapping.get(column_name)
    if column_index is None:
        return ""
    return row["values"].get(column_index, "")


def build_clean_rows(selected: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = selected["mapping"]
    rows = selected["rows"]
    header_row_number = selected["header_row_number"]
    source_sheet = selected["sheet"]["name"]
    clean_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    input_row_count = 0

    for row in rows:
        row_number = row["row_number"]
        if header_row_number is not None and row_number <= header_row_number:
            continue

        mapped_values = [row_value(row, mapping, key) for key in mapping]
        if all(is_blank(value) for value in mapped_values):
            continue

        input_row_count += 1
        video_id = clean_text(row_value(row, mapping, "video_id"))
        video_title = clean_text(row_value(row, mapping, "video_title"))
        abrupt_value = clean_text(row_value(row, mapping, "is_abrupt_transition_ad"))
        if abrupt_value not in {"yes", "no", "unclear", "not_reviewed"}:
            abrupt_value = "not_reviewed"

        time_warnings: list[str] = []
        if "ad_start_combined" in mapping:
            start_min, start_sec_part, start_sec, warning = parse_combined_time(row_value(row, mapping, "ad_start_combined"))
            if warning:
                time_warnings.append(f"start:{warning}")
        else:
            start_min, start_sec_part, start_sec, sep_warnings = parse_separate_time(
                row_value(row, mapping, "ad_start_min"),
                row_value(row, mapping, "ad_start_sec_part"),
            )
            time_warnings.extend(f"start:{warning}" for warning in sep_warnings)

        if "ad_end_combined" in mapping:
            end_min, end_sec_part, end_sec, warning = parse_combined_time(row_value(row, mapping, "ad_end_combined"))
            if warning:
                time_warnings.append(f"end:{warning}")
        else:
            end_min, end_sec_part, end_sec, sep_warnings = parse_separate_time(
                row_value(row, mapping, "ad_end_min"),
                row_value(row, mapping, "ad_end_sec_part"),
            )
            time_warnings.extend(f"end:{warning}" for warning in sep_warnings)

        invalid_reasons = []
        if not video_id:
            invalid_reasons.append("missing_video_id")
        if not video_title:
            invalid_reasons.append("missing_video_title")
        if start_sec is None:
            invalid_reasons.append("ad_start_sec_not_numeric")
        if end_sec is None:
            invalid_reasons.append("ad_end_sec_not_numeric")

        duration = None
        if start_sec is not None and end_sec is not None:
            duration = end_sec - start_sec
            if end_sec <= start_sec:
                invalid_reasons.append("ad_end_sec_not_greater_than_ad_start_sec")
            if duration <= 0:
                invalid_reasons.append("ad_duration_sec_not_positive")
            if 0 < duration < 3:
                time_warnings.append("duration_very_short_lt_3_sec")
            if duration > 600:
                time_warnings.append("duration_very_long_gt_600_sec")

        if time_warnings:
            warnings.append(
                {
                    "source_row_number": row_number,
                    "video_id": video_id,
                    "warnings": sorted(set(time_warnings)),
                }
            )

        clean_rows.append(
            {
                "ad_interval_id": f"A{len(clean_rows) + 1:03d}",
                "video_id": video_id,
                "video_title": video_title,
                "ad_start_min": as_output_number(start_min),
                "ad_start_sec_part": as_output_number(start_sec_part),
                "ad_end_min": as_output_number(end_min),
                "ad_end_sec_part": as_output_number(end_sec_part),
                "ad_start_sec": as_output_number(start_sec),
                "ad_end_sec": as_output_number(end_sec),
                "ad_duration_sec": as_output_number(duration),
                "is_abrupt_transition_ad": abrupt_value,
                "label_valid": "false" if invalid_reasons else "true",
                "invalid_reason": "; ".join(sorted(set(invalid_reasons))),
                "source_file": str(RAW_LABEL_COPY.relative_to(PROJECT_ROOT)),
                "source_sheet": source_sheet,
                "source_row_number": row_number,
            }
        )

    stats = {
        "input_row_count": input_row_count,
        "clean_label_row_count": len(clean_rows),
        "valid_row_count": sum(1 for row in clean_rows if row["label_valid"] == "true"),
        "invalid_row_count": sum(1 for row in clean_rows if row["label_valid"] == "false"),
        "row_warnings": warnings,
    }
    return clean_rows, stats


def required_column_status(mapping: dict[str, int]) -> list[str]:
    missing = []
    if "video_id" not in mapping:
        missing.append("video_id")
    if "video_title" not in mapping:
        missing.append("video_title")
    if "ad_start_combined" not in mapping and not {"ad_start_min", "ad_start_sec_part"}.issubset(mapping):
        missing.append("ad_start_time")
    if "ad_end_combined" not in mapping and not {"ad_end_min", "ad_end_sec_part"}.issubset(mapping):
        missing.append("ad_end_time")
    return missing


def write_clean_csv(rows: list[dict[str, Any]]) -> None:
    with CLEAN_LABEL_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLEAN_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def xml_text(parent: ET.Element, tag: str, text: str | None = None, attrib: dict[str, str] | None = None) -> ET.Element:
    element = ET.SubElement(parent, tag, attrib or {})
    if text is not None:
        element.text = text
    return element


def add_inline_cell(row_el: ET.Element, row_idx: int, col_idx: int, value: Any, style_idx: int = 2) -> None:
    ref = f"{column_index_to_letter(col_idx)}{row_idx}"
    attrib = {"r": ref, "t": "inlineStr", "s": str(style_idx)}
    cell = ET.SubElement(row_el, f"{{{XLSX_MAIN_NS}}}c", attrib)
    is_el = ET.SubElement(cell, f"{{{XLSX_MAIN_NS}}}is")
    text_el = ET.SubElement(is_el, f"{{{XLSX_MAIN_NS}}}t")
    text = clean_text(value)
    if text.startswith(" ") or text.endswith(" "):
        text_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_el.text = text


def build_sheet_xml(
    rows: list[list[Any]],
    widths: list[float],
    freeze: bool = False,
    autofilter: bool = False,
    validation_col_idx: int | None = None,
) -> bytes:
    worksheet = ET.Element(f"{{{XLSX_MAIN_NS}}}worksheet")
    sheet_views = ET.SubElement(worksheet, f"{{{XLSX_MAIN_NS}}}sheetViews")
    sheet_view = ET.SubElement(sheet_views, f"{{{XLSX_MAIN_NS}}}sheetView", {"workbookViewId": "0"})
    if freeze:
        ET.SubElement(sheet_view, f"{{{XLSX_MAIN_NS}}}pane", {"ySplit": "1", "topLeftCell": "A2", "activePane": "bottomLeft", "state": "frozen"})
        ET.SubElement(sheet_view, f"{{{XLSX_MAIN_NS}}}selection", {"pane": "bottomLeft"})

    cols = ET.SubElement(worksheet, f"{{{XLSX_MAIN_NS}}}cols")
    for index, width in enumerate(widths, start=1):
        ET.SubElement(cols, f"{{{XLSX_MAIN_NS}}}col", {"min": str(index), "max": str(index), "width": str(width), "customWidth": "1"})

    sheet_data = ET.SubElement(worksheet, f"{{{XLSX_MAIN_NS}}}sheetData")
    for row_idx, row_values in enumerate(rows, start=1):
        row_el = ET.SubElement(sheet_data, f"{{{XLSX_MAIN_NS}}}row", {"r": str(row_idx)})
        for col_idx, value in enumerate(row_values, start=1):
            style_idx = 1 if row_idx == 1 else 2
            add_inline_cell(row_el, row_idx, col_idx, value, style_idx)

    if autofilter and rows and rows[0]:
        last_col = column_index_to_letter(len(rows[0]))
        last_row = max(1, len(rows))
        ET.SubElement(worksheet, f"{{{XLSX_MAIN_NS}}}autoFilter", {"ref": f"A1:{last_col}{last_row}"})

    if validation_col_idx is not None:
        col = column_index_to_letter(validation_col_idx)
        data_validations = ET.SubElement(worksheet, f"{{{XLSX_MAIN_NS}}}dataValidations", {"count": "1"})
        validation = ET.SubElement(
            data_validations,
            f"{{{XLSX_MAIN_NS}}}dataValidation",
            {
                "type": "list",
                "allowBlank": "1",
                "showErrorMessage": "1",
                "sqref": f"{col}2:{col}1048576",
            },
        )
        xml_text(validation, f"{{{XLSX_MAIN_NS}}}formula1", '"not_reviewed,yes,no,unclear"')

    ET.SubElement(worksheet, f"{{{XLSX_MAIN_NS}}}pageMargins", {"left": "0.7", "right": "0.7", "top": "0.75", "bottom": "0.75", "header": "0.3", "footer": "0.3"})
    return ET.tostring(worksheet, encoding="utf-8", xml_declaration=True)


def write_review_xlsx(rows: list[dict[str, Any]]) -> None:
    ET.register_namespace("", XLSX_MAIN_NS)
    review_rows = [CLEAN_LABEL_COLUMNS] + [[row.get(column, "") for column in CLEAN_LABEL_COLUMNS] for row in rows]
    guide_rows = [
        ["section", "guide"],
        ["이번 프로젝트의 새 목표", "모든 광고/홍보 탐지가 아니라, 일반 콘텐츠 흐름에서 갑자기 분리되어 삽입되는 전환형 광고 구간 탐지."],
        ["yes 기준", "광고 시작 전후에 뚜렷한 scene/audio/visual transition이 있음. 광고 파트가 별도 블록처럼 분리됨. 광고 종료 후 원래 콘텐츠 흐름으로 복귀함. 해당 구간을 스킵해도 영상 흐름이 자연스러움."],
        ["no 기준", "일반 대화 흐름 안에서 자연스럽게 제품을 언급함. 장면 전환 없이 협찬 멘트가 이어짐. 제품 사용이 본편 흐름에 자연스럽게 포함됨. 스킵하면 본편 흐름이 어색해질 수 있음."],
        ["unclear 기준", "전환형 광고인지 판단이 애매함. 시작/종료 경계가 불명확함. scene transition은 있지만 일반 편집 전환인지 광고 전환인지 애매함."],
        ["not_reviewed 기준", "아직 사람이 검토하지 않은 기본 상태."],
        ["검토 컬럼", "label_review sheet의 is_abrupt_transition_ad 컬럼에서 not_reviewed, yes, no, unclear 중 하나를 선택."],
    ]
    widths = [16, 18, 36, 14, 18, 14, 18, 14, 14, 16, 24, 12, 36, 32, 18, 18]
    guide_widths = [28, 120]

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
""".strip()
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
""".strip()
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="label_review" sheetId="1" r:id="rId1"/>
    <sheet name="label_guide" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>
""".strip()
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
""".strip()
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFECEFF3"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="3">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
  <dxfs count="0"/>
  <tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>
</styleSheet>
""".strip()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>initialize_project_v1</dc:creator>
  <cp:lastModifiedBy>initialize_project_v1</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
""".strip()
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>initialize_project_v1</Application>
</Properties>
""".strip()

    with zipfile.ZipFile(REVIEW_XLSX, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", styles)
        zf.writestr("xl/worksheets/sheet1.xml", build_sheet_xml(review_rows, widths, freeze=True, autofilter=True, validation_col_idx=CLEAN_LABEL_COLUMNS.index("is_abrupt_transition_ad") + 1))
        zf.writestr("xl/worksheets/sheet2.xml", build_sheet_xml(guide_rows, guide_widths, freeze=True, autofilter=False))
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)


def write_readme() -> None:
    README_PATH.write_text(
        """# youtube_ad_segment_detector

## 프로젝트 목표

유튜브 영상에서 일반 콘텐츠 흐름과 분리되어 갑자기 삽입되는 전환형 광고 구간을 탐지한다.
최종 출력은 `video_id, ad_start_sec, ad_end_sec` 형식이다.

## 기존 목표와의 차이

기존 목표는 유튜브 영상 내 광고/홍보 구간 전체 탐지에 가까웠다.
새 목표는 라벨 기준을 정리하여, 자연스럽게 이어지는 홍보는 제외하고 갑작스러운 장면 전환형 광고만 탐지한다.

## 현재 라벨 파일

- `data/raw_labels/new_ad_labeling.xlsx`
- `data/labels/clean_ad_labels_v0.csv`
- `data/labels/clean_ad_labels_v0_review.xlsx`

## 향후 작업 순서

1. clean label 검토
2. `is_abrupt_transition_ad` 라벨 확인
3. 영상 파일 수집 및 `data/videos/` 정리
4. 5초 window 생성
5. Scene-change feature 생성
6. OCR feature 생성
7. rule-based abrupt-transition ad detector 작성
8. interval 단위 평가

## 주의사항

- 기존 `./_old_project_not_included`와 분리된 프로젝트이다.
- 모든 작업은 `.` 안에서만 수행한다.
- LLM/Gemma는 이번 최종 구조의 필수 구성요소가 아니다.
- Scene/OCR/audio/visual cue 기반 rule-based detector를 우선 구현한다.
""",
        encoding="utf-8",
    )


def write_manifest(report: dict[str, Any], generated_files: list[str]) -> None:
    manifest = {
        "project_root": str(PROJECT_ROOT),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_label_file": str(INPUT_LABEL_FILE),
        "raw_label_copy_path": str(RAW_LABEL_COPY),
        "clean_label_csv_path": str(CLEAN_LABEL_CSV),
        "review_xlsx_path": str(REVIEW_XLSX),
        "project_goal": "유튜브 영상에서 일반 콘텐츠 흐름과 분리되어 갑자기 삽입되는 전환형 광고 구간 탐지",
        "detector_scope": "자연스럽게 이어지는 홍보 전체가 아니라 갑작스러운 scene/audio/visual transition을 동반하는 별도 광고 블록",
        "folder_structure_created": FOLDERS,
        "warnings": report.get("warnings", []),
        "next_steps": [
            "clean label 검토",
            "is_abrupt_transition_ad 값을 yes/no/unclear로 사람 검토",
            "영상 파일 수집 및 data/videos/ 정리",
            "5초 window 생성",
            "Scene-change feature 생성",
            "OCR feature 생성",
            "rule-based abrupt-transition ad detector 작성",
            "interval 단위 평가",
        ],
        "generated_files": generated_files,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary(report: dict[str, Any]) -> None:
    warnings = report.get("warnings", [])
    warning_text = "\n".join(f"- {warning}" for warning in warnings) if warnings else "- 주요 warning 없음"
    SUMMARY_PATH.write_text(
        f"""# initialize_project_v1 summary

## 완료 여부

- 새 프로젝트 폴더 구조 생성: 완료
- 새 라벨 파일 인식: {'완료' if report.get('input_file_exists') else '실패'}
- clean label 생성: 완료
- review Excel 생성: 완료

## 라벨 요약

- detected sheet: `{report.get('detected_sheet_name', '')}`
- input rows: {report.get('input_row_count', 0)}
- clean label rows: {report.get('clean_label_row_count', 0)}
- valid rows: {report.get('valid_row_count', 0)}
- invalid rows: {report.get('invalid_row_count', 0)}
- missing required columns: {report.get('missing_required_columns', [])}
- ambiguous columns: {report.get('ambiguous_columns', [])}

## 주요 warning

{warning_text}

## 다음 작업

1. `data/labels/clean_ad_labels_v0_review.xlsx`에서 `is_abrupt_transition_ad` 컬럼을 검토한다.
2. `yes/no/unclear` 검토 완료본을 별도 파일로 저장한다.
3. 영상 파일을 `data/videos/`에 정리한 뒤 window/feature/rule detector 작업을 시작한다.
""",
        encoding="utf-8",
    )


def copy_latest_files() -> None:
    latest_files = [
        (README_PATH, LATEST_DIR / "README.md"),
        (MANIFEST_PATH, LATEST_DIR / "project_manifest.json"),
        (REPORT_PATH, LATEST_DIR / "initialize_project_v1_report.json"),
        (SUMMARY_PATH, LATEST_DIR / "initialize_project_v1_summary.md"),
        (LOG_PATH, LATEST_DIR / "initialize_project_v1_run_log.txt"),
        (CLEAN_LABEL_CSV, LATEST_DIR / "clean_ad_labels_v0.csv"),
    ]
    for src, dst in latest_files:
        shutil.copy2(src, dst)
    LATEST_README.write_text(
        """# latest_for_chatgpt files

이 폴더는 다음 대화나 검토에서 바로 확인할 수 있는 최신 요약 파일만 모아 둔다.

- `README.md`: 새 프로젝트 목표와 작업 순서 요약.
- `project_manifest.json`: 프로젝트 경로, 주요 산출물, 다음 단계의 기계 판독용 manifest.
- `initialize_project_v1_report.json`: 입력 라벨 파일, 컬럼 매핑, row count, warning을 포함한 상세 report.
- `initialize_project_v1_summary.md`: 사람이 빠르게 읽기 위한 초기화 요약.
- `initialize_project_v1_run_log.txt`: 초기화 처리 과정 log.
- `clean_ad_labels_v0.csv`: 이후 라벨 검토와 feature 작업의 기준 clean label.

원본 xlsx, 영상 파일, 프레임 파일, 모델 파일, 캐시 파일은 이 폴더에 복사하지 않는다.
""",
        encoding="utf-8",
    )


def create_gitkeep_files() -> None:
    for folder in FOLDERS:
        gitkeep = project_path(folder) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")


def main() -> None:
    if PROJECT_ROOT.resolve() == OLD_PROJECT_ROOT.resolve():
        raise RuntimeError("New project root unexpectedly matches old project root.")
    log("Command: mkdir -p <requested youtube_ad_segment_detector folder structure>")
    for folder in FOLDERS:
        project_path(folder).mkdir(parents=True, exist_ok=True)
    log("Created requested folder structure inside ..")

    create_gitkeep_files()
    log("Created .gitkeep files for empty folder preservation.")

    input_exists = INPUT_LABEL_FILE.exists()
    input_sha = sha256_file(INPUT_LABEL_FILE)
    if not input_exists:
        raise FileNotFoundError(f"Missing input label file: {INPUT_LABEL_FILE}")

    if RAW_LABEL_COPY.exists():
        raw_sha_before = sha256_file(RAW_LABEL_COPY)
        if raw_sha_before != input_sha:
            raise RuntimeError(f"Raw label copy already exists with different SHA256: {RAW_LABEL_COPY}")
        log("Raw label copy already existed with matching SHA256; left it unchanged.")
    else:
        shutil.copy2(INPUT_LABEL_FILE, RAW_LABEL_COPY)
        log(f"Copied raw label file to {RAW_LABEL_COPY}.")
    raw_sha = sha256_file(RAW_LABEL_COPY)

    with zipfile.ZipFile(RAW_LABEL_COPY) as zf:
        selected = select_main_sheet(zf)

    clean_rows, stats = build_clean_rows(selected)
    write_clean_csv(clean_rows)
    log(f"Wrote clean label CSV to {CLEAN_LABEL_CSV}.")

    write_review_xlsx(clean_rows)
    log(f"Wrote review Excel to {REVIEW_XLSX}.")

    write_readme()
    log(f"Wrote README to {README_PATH}.")

    duplicate_video_ids = sorted([video_id for video_id, count in Counter(row["video_id"] for row in clean_rows if row["video_id"]).items() if count > 1])
    duplicate_interval_ids = sorted([interval_id for interval_id, count in Counter(row["ad_interval_id"] for row in clean_rows).items() if count > 1])
    missing_required_columns = required_column_status(selected["mapping"])

    warnings: list[Any] = []
    if selected.get("score", 0) < 12:
        warnings.append("main_label_sheet_detected_with_partial_confidence")
    if missing_required_columns:
        warnings.append(f"missing_required_columns: {missing_required_columns}")
    if selected.get("ambiguous"):
        warnings.append("ambiguous_columns_detected")
    warnings.extend(stats["row_warnings"])

    generated_files = [
        str(RAW_LABEL_COPY),
        str(CLEAN_LABEL_CSV),
        str(REVIEW_XLSX),
        str(README_PATH),
        str(MANIFEST_PATH),
        str(REPORT_PATH),
        str(SUMMARY_PATH),
        str(LOG_PATH),
        str(LATEST_DIR / "README.md"),
        str(LATEST_DIR / "project_manifest.json"),
        str(LATEST_DIR / "initialize_project_v1_report.json"),
        str(LATEST_DIR / "initialize_project_v1_summary.md"),
        str(LATEST_DIR / "initialize_project_v1_run_log.txt"),
        str(LATEST_DIR / "clean_ad_labels_v0.csv"),
        str(LATEST_README),
        str(PROJECT_ROOT / "scripts/labels/initialize_project_v1.py"),
    ]

    report = {
        "project_root": str(PROJECT_ROOT),
        "input_file_exists": input_exists,
        "input_file_path": str(INPUT_LABEL_FILE),
        "input_file_sha256": input_sha,
        "copied_raw_label_path": str(RAW_LABEL_COPY),
        "copied_raw_label_sha256": raw_sha,
        "available_sheets": [sheet["name"] for sheet in selected.get("all_sheets", [])],
        "sheet_analyses": selected.get("sheet_analyses", []),
        "detected_sheet_name": selected["sheet"].get("name", ""),
        "detected_column_mapping": mapping_to_headers(selected["mapping"], selected["rows"], selected["header_row_number"]),
        "detected_column_indexes": {key: column_index_to_letter(value) for key, value in sorted(selected["mapping"].items())},
        "ambiguous_columns": selected.get("ambiguous", []),
        "missing_required_columns": missing_required_columns,
        "input_row_count": stats["input_row_count"],
        "clean_label_row_count": stats["clean_label_row_count"],
        "valid_row_count": stats["valid_row_count"],
        "invalid_row_count": stats["invalid_row_count"],
        "has_duplicate_video_id": bool(duplicate_video_ids),
        "duplicate_video_ids": duplicate_video_ids,
        "has_duplicate_ad_interval_id": bool(duplicate_interval_ids),
        "duplicate_ad_interval_ids": duplicate_interval_ids,
        "generated_files_list": generated_files,
        "old_project_path_modified": False,
        "old_project_path": str(OLD_PROJECT_ROOT),
        "warnings": warnings,
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote report to {REPORT_PATH}.")

    write_summary(report)
    log(f"Wrote summary to {SUMMARY_PATH}.")

    write_manifest(report, generated_files)
    log(f"Wrote manifest to {MANIFEST_PATH}.")

    log("Command: python scripts/labels/initialize_project_v1.py")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    copy_latest_files()
    log(f"Copied latest summary files into {LATEST_DIR}.")
    LOG_PATH.write_text("\n".join(RUN_LOG) + "\n", encoding="utf-8")
    shutil.copy2(LOG_PATH, LATEST_DIR / "initialize_project_v1_run_log.txt")

    print(json.dumps({
        "project_root": str(PROJECT_ROOT),
        "detected_sheet_name": report["detected_sheet_name"],
        "row_count": report["clean_label_row_count"],
        "valid_row_count": report["valid_row_count"],
        "invalid_row_count": report["invalid_row_count"],
        "missing_required_columns": report["missing_required_columns"],
        "ambiguous_columns_count": len(report["ambiguous_columns"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
