from __future__ import annotations

import re

from pathlib import Path
from typing import BinaryIO

from pypeh.adapters.persistence.filesystem import (
    ensure_filesystem_parent_directory,
    join_filesystem_path,
)
from pypeh.core.models.internal_data_layout import DatasetSeries


_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")
_MAX_SHEET_NAME_LENGTH = 31
_FilesystemExcelDestination = str


def _require_dependencies():
    try:
        import polars as pl
        import xlsxwriter
    except ImportError as exc:
        raise ImportError(
            "DatasetSeries Excel export requires the dataframe export "
            "dependencies ('polars' and 'xlsxwriter')."
        ) from exc
    return pl, xlsxwriter


def _label_tail(label: str) -> str:
    stripped = label.strip().rstrip("/#")
    for separator in ("/", "#"):
        if separator in stripped:
            stripped = stripped.rsplit(separator, 1)[-1]
    return stripped


def _safe_sheet_name(label: str, used_names: set[str]) -> str:
    sheet_name = _INVALID_SHEET_CHARS.sub("_", _label_tail(label)).strip()
    sheet_name = sheet_name.strip("'")
    if not sheet_name:
        sheet_name = "Sheet"

    base_name = sheet_name[:_MAX_SHEET_NAME_LENGTH]
    candidate = base_name
    suffix = 1
    while candidate.lower() in used_names:
        suffix_text = f"_{suffix}"
        candidate = (
            base_name[: _MAX_SHEET_NAME_LENGTH - len(suffix_text)]
            + suffix_text
        )
        suffix += 1

    used_names.add(candidate.lower())
    return candidate


def _workbook_stem(label: str) -> str:
    stem = _INVALID_SHEET_CHARS.sub("_", _label_tail(label)).strip()
    stem = stem.strip(".'")
    return stem or "dataset_series"


def _is_excel_workbook_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".xlsx"


def _dump_dataset_series_to_excel_workbook(
    dataset_series: DatasetSeries,
    destination: str | Path | BinaryIO,
    **write_options,
):
    pl, xlsxwriter = _require_dependencies()
    if len(dataset_series) == 0:
        raise ValueError("Cannot export an empty DatasetSeries to Excel.")

    workbook_destination = (
        str(destination) if isinstance(destination, Path) else destination
    )
    workbook = xlsxwriter.Workbook(
        workbook_destination,
        {"in_memory": hasattr(destination, "write")},
    )
    used_names: set[str] = set()

    try:
        for dataset_label in dataset_series:
            dataset = dataset_series[dataset_label]
            assert dataset is not None
            data = dataset.data
            if data is None:
                continue
            if not isinstance(data, pl.DataFrame):
                raise TypeError(
                    "DatasetSeries Excel export expects each dataset.data to "
                    "be a polars.DataFrame. "
                    f"Dataset {dataset.label!r} has {type(data).__name__}."
                )

            sheet_name = _safe_sheet_name(dataset.label, used_names)
            data.write_excel(
                workbook,
                worksheet=sheet_name,
                **write_options,
            )
    finally:
        workbook.close()


def dump_dataset_series_to_excel(
    dataset_series: DatasetSeries,
    destination: str | Path | BinaryIO,
    **write_options,
) -> list[Path] | list[str | Path | BinaryIO]:
    """
    Export a DatasetSeries to a single Excel workbook.

    Each Dataset becomes one worksheet, using the Dataset label as the source
    for the sheet name. This is an export format only; DatasetSeries metadata
    is not serialized into the workbook.
    """
    if hasattr(destination, "write"):
        _dump_dataset_series_to_excel_workbook(
            dataset_series, destination, **write_options
        )
        return [destination]

    destination = Path(destination)
    if not _is_excel_workbook_path(destination):
        destination.mkdir(parents=True, exist_ok=True)
        destination = (
            destination / f"{_workbook_stem(dataset_series.label)}.xlsx"
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    _dump_dataset_series_to_excel_workbook(
        dataset_series, destination, **write_options
    )
    return [destination]


def _dataset_series_workbook_name(dataset_series: DatasetSeries) -> str:
    return f"{_workbook_stem(dataset_series.label)}.xlsx"


def dump_dataset_series_to_excel_filesystem(
    dataset_series: DatasetSeries,
    file_system,
    destination: _FilesystemExcelDestination,
    **write_options,
) -> list[str]:
    """
    Export a DatasetSeries to one Excel workbook through an fsspec filesystem.
    """
    if not _is_excel_workbook_path(destination):
        destination = join_filesystem_path(
            file_system,
            destination,
            _dataset_series_workbook_name(dataset_series),
        )

    ensure_filesystem_parent_directory(file_system, destination)
    with file_system.open(destination, "wb") as output_file:
        _dump_dataset_series_to_excel_workbook(
            dataset_series,
            output_file,
            **write_options,
        )
    return [destination]
