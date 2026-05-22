import io
from zipfile import ZipFile

import pytest

from pypeh.adapters.persistence.dataset_excel import (
    dump_dataset_series_to_excel,
)
from pypeh.adapters.persistence.serializations import ExcelIO
from pypeh.core.models.internal_data_layout import DatasetSeries


@pytest.fixture
def excel_dataset_series():
    pl = pytest.importorskip("polars")

    series = DatasetSeries(label="https://example.org/export/session")
    sample = series.add_empty_dataset("https://example.org/datasets/SAMPLE")
    sample.data = pl.DataFrame({"id_sample": ["sample-a", "sample-b"]})
    lab = series.add_empty_dataset("https://example.org/datasets/LAB")
    lab.data = pl.DataFrame(
        {"id_sample": ["sample-a", "sample-b"], "chol": [1.2, 3.4]}
    )
    return series


def _workbook_sheet_names(workbook_bytes: bytes) -> list[str]:
    with ZipFile(io.BytesIO(workbook_bytes)) as workbook:
        workbook_xml = workbook.read("xl/workbook.xml").decode("utf-8")
    names = []
    for token in workbook_xml.split("<sheet "):
        marker = 'name="'
        if marker not in token:
            continue
        names.append(token.split(marker, 1)[1].split('"', 1)[0])
    return names


@pytest.mark.dataframe
class TestDatasetSeriesExcelExport:
    def test_dump_dataset_series_to_excel_writes_one_sheet_per_dataset(
        self, excel_dataset_series
    ):
        pytest.importorskip("xlsxwriter")

        buffer = io.BytesIO()
        outputs = dump_dataset_series_to_excel(excel_dataset_series, buffer)

        assert outputs == [buffer]
        assert _workbook_sheet_names(buffer.getvalue()) == ["SAMPLE", "LAB"]

        workbook = ExcelIO().load(buffer.getvalue())
        assert set(workbook) == {"SAMPLE", "LAB"}
        assert workbook["SAMPLE"].shape == (2, 1)
        assert workbook["LAB"].shape == (2, 2)

    def test_dump_dataset_series_to_excel_rejects_non_dataframe_data(self):
        pytest.importorskip("xlsxwriter")

        series = DatasetSeries(label="series")
        dataset = series.add_empty_dataset("INVALID")
        dataset.data = {"not": "a dataframe"}

        with pytest.raises(TypeError, match="polars.DataFrame"):
            dump_dataset_series_to_excel(series, io.BytesIO())

    def test_dump_dataset_series_to_excel_deduplicates_sheet_names(self):
        pytest.importorskip("xlsxwriter")
        pl = pytest.importorskip("polars")

        series = DatasetSeries(label="series")
        first = series.add_empty_dataset("https://example.org/datasets/SAMPLE")
        first.data = pl.DataFrame({"value": [1]})
        second = series.add_empty_dataset("https://other.example/SAMPLE")
        second.data = pl.DataFrame({"value": [2]})

        buffer = io.BytesIO()
        dump_dataset_series_to_excel(series, buffer)

        assert _workbook_sheet_names(buffer.getvalue()) == [
            "SAMPLE",
            "SAMPLE_1",
        ]
