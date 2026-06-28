import io
from datetime import datetime

import fsspec
import linkml_runtime.loaders
import pytest
import rdflib
from peh_model.peh import EntityList
from pydantic import BaseModel

from pypeh.core.cache.containers import CacheContainer, CacheContainerFactory
from pypeh.core.models.constants import ValidationErrorLevel
from pypeh.core.models.validation_errors import (
    TypeCastError,
    ValidationErrorReport,
)
from pypeh.adapters.persistence.serializations import (
    IOAdapterFactory,
    IOAdapter,
    JsonIO,
    YamlIO,
    ExcelIO,
    CsvIO,
)
from tests.test_utils.dirutils import get_absolute_path
from tests.test_utils.xlsx import write_minimal_xlsx


class MockAdapter(IOAdapter):
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class MockModel(BaseModel):
    empty: str


@pytest.mark.core
class TestIOAdapterFactory:
    @pytest.mark.parametrize(
        "format_name, expected_adapter",
        [
            ("json", "JsonIO"),
            ("yaml", "YamlIO"),
            ("csv", "CsvIO"),
            ("parquet", "ParquetIO"),
            ("pq", "ParquetIO"),
            ("dataset-parquet", "ParquetIO"),
            ("pypeh-dataset-parquet", "ParquetIO"),
            ("xlsx", "ExcelIO"),
            ("xls", "ExcelIO"),
        ],
    )
    def test_create_known_adapters(self, format_name, expected_adapter):
        adapter = IOAdapterFactory.create(format_name)
        assert adapter.__class__.__name__ == expected_adapter

    def test_create_unknown_adapter_raises_error(self):
        with pytest.raises(
            ValueError, match="No adapter registered for dataformat: unknown"
        ):
            IOAdapterFactory.create("unknown")

    def test_register_adapter(self):
        IOAdapterFactory.register_adapter("mock", MockAdapter)
        adapter = IOAdapterFactory.create("mock", test_param=True)

        assert isinstance(adapter, MockAdapter)
        assert adapter.kwargs["test_param"]


@pytest.mark.core
class TestYamlIO:
    def test_basic(self):
        source = get_absolute_path(
            "./input/config_basic/_Reference_YAML/observable_entities.yaml"
        )
        yaml_io = YamlIO()
        yaml_io.load(source)

    def test_wrong_schema(self, caplog):
        source = get_absolute_path(
            "./input/config_basic/_Reference_YAML/observable_entities.yaml"
        )
        yaml_io = YamlIO()
        with pytest.raises(ValueError):
            _ = yaml_io.load(source, target_class=MockModel)

    def test_wrong_input(self):
        source = get_absolute_path("./input/wrong_input/random.yaml")
        yaml_io = YamlIO()
        with pytest.raises(TypeError):
            yaml_io.load(source)

    def test_textio(self):
        source = get_absolute_path(
            "./input/config_basic/_Reference_YAML/observable_entities.yaml"
        )
        yaml_io = YamlIO()
        with open(source, "r") as f:
            data = yaml_io.load(f)
        assert isinstance(data, EntityList)


@pytest.mark.core
class TestJsonIO:
    def test_basic(self):
        source = get_absolute_path("./input/observation_results.json")
        json_io = JsonIO()
        with open(source, "r") as f:
            data = json_io.load(f)
        assert isinstance(data, EntityList)


@pytest.mark.dataframe
class TestCsvIO:
    def test_basic_import(self):
        source = get_absolute_path(
            "./input/config_basic/_Tabular_Data/sampling_data_to_import.csv"
        )
        csv_io = CsvIO()
        with fsspec.open(source, "r") as f:
            data = csv_io.load(f, raise_if_empty=False, infer_schema_length=5)  # type: ignore
        from polars import DataFrame

        assert isinstance(data, DataFrame)

        with fsspec.open(source, "rb") as f:
            data = csv_io.load(f, raise_if_empty=False, infer_schema_length=5)  # type: ignore
        assert isinstance(data, DataFrame)


@pytest.mark.dataframe
class TestXlsIO:
    def test_basic_import(self):
        source = get_absolute_path(
            "./input/config_basic/_Tabular_Data/sampling_data_to_import.xlsx"
        )
        excel_io = ExcelIO()
        with fsspec.open(source, "rb") as f:
            data = excel_io.load(f)  # type: ignore
        assert isinstance(data, dict)

    def test_invalid_excel(self):
        source = get_absolute_path(
            "./input/config_invalid/_Tabular_Data/invalid_excel.xlsx"
        )
        excel_io = ExcelIO()
        with fsspec.open(source, "rb") as f:
            with pytest.raises(Exception) as excinfo:
                _ = excel_io.load(f)  # type: ignore
        assert isinstance(excinfo.value, Exception)

    def test_typed_sheet(self):
        typed_dict = {
            "id_sample": "string",
            "samplingyear": "float",
            "samplingmonth": "string",
            "samplingday": "float",
            "samplinghour": "float",
            "samplingminutes": "float",
        }
        source = get_absolute_path("./input/validation_test_03_data.xlsx")
        excel_io = ExcelIO()
        with fsspec.open(source, "rb") as f:
            result = excel_io.load_section(
                f,  # type: ignore
                section_name="SAMPLE",
                data_schema=typed_dict,
            )
        import polars as pl

        assert isinstance(result, pl.DataFrame)

    def test_typed_excel(self):
        typed_dict = {
            "SAMPLE": {
                "id_sample": "string",
                "samplingyear": "float",
                "samplingmonth": "string",
                "samplingday": "float",
                "samplinghour": "float",
                "samplingminutes": "float",
            },
            "SAMPLETIMEPOINT_BSS": {
                "id_sample": "integer",
                "chol": "float",
                "chol_loq": "float",
                "chol_lod": "float",
            },
        }
        source = get_absolute_path("./input/validation_test_03_data.xlsx")
        excel_io = ExcelIO()
        with fsspec.open(source, "rb") as f:
            result = excel_io.load(
                f,  # type: ignore
                data_schema=typed_dict,
            )
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "cast_error_policy, expected_chol_read_type",
        [
            ("null", "float"),
            ("raise", "string"),
            ("report", "string"),
        ],
    )
    def test_typed_sheet_passes_dtypes_to_reader(
        self, monkeypatch, cast_error_policy, expected_chol_read_type
    ):
        import polars as pl

        from pypeh.adapters.persistence.dataframe import ExcelIOImpl

        captured_options = {}

        def fake_load(self, source, **options):
            captured_options.update(options)
            return pl.DataFrame(
                {
                    "id_sample": ["sample_a"],
                    "chol": ["1.2"],
                    "sample_date": ["2025-10-13 00:00:00"],
                    "sample_datetime": ["2025-10-13 13:45:12"],
                }
            )

        monkeypatch.setattr(ExcelIOImpl, "_load", fake_load)

        result = ExcelIOImpl().load_section(
            "unused.xlsx",
            section_name="SAMPLE",
            data_schema={
                "id_sample": "string",
                "chol": "float",
                "sample_date": "date",
                "sample_datetime": "datetime",
            },
            cast_error_policy=cast_error_policy,
        )

        assert captured_options["read_options"] == {
            "dtypes": {
                "id_sample": "string",
                "chol": expected_chol_read_type,
            }
        }
        assert result.schema["chol"] == pl.Float64
        assert result.schema["sample_date"] == pl.Date
        assert result.schema["sample_datetime"] == pl.Datetime

    def test_typed_sheet_type_mismatch_is_loaded_as_null(self, tmp_path):
        source = tmp_path / "typed_mismatch.xlsx"
        write_minimal_xlsx(
            source,
            sheet_name="SAMPLE",
            headers=["id_sample", "chol"],
            rows=[
                ["sample_a", 1.2],
                ["sample_b", "oops"],
                ["sample_c", 3.4],
            ],
        )

        excel_io = ExcelIO()
        result = excel_io.load_section(
            source,
            section_name="SAMPLE",
            data_schema={"id_sample": "string", "chol": "float"},
            cast_error_policy="null",
        )

        assert result.shape == (3, 2)
        assert result["id_sample"].to_list() == [
            "sample_a",
            "sample_b",
            "sample_c",
        ]
        assert result["chol"].to_list() == [1.2, None, 3.4]

    def test_typed_sheet_date_accepts_midnight_datetime_strings(
        self, tmp_path
    ):
        import polars as pl

        source = tmp_path / "typed_dates.xlsx"
        write_minimal_xlsx(
            source,
            sheet_name="SAMPLE",
            headers=["id_sample", "onderzoeksdatum"],
            rows=[
                ["sample_a", "2025-10-13 00:00:00"],
                ["sample_b", "2025-10-14 00:00:00"],
                ["sample_c", "2025-10-15"],
            ],
        )

        excel_io = ExcelIO()
        result = excel_io.load_section(
            source,
            section_name="SAMPLE",
            data_schema={"id_sample": "string", "onderzoeksdatum": "date"},
            cast_error_policy="raise",
        )

        assert result.schema["onderzoeksdatum"] == pl.Date
        assert [value.isoformat() for value in result["onderzoeksdatum"]] == [
            "2025-10-13",
            "2025-10-14",
            "2025-10-15",
        ]

    def test_typed_sheet_datetime_accepts_datetime_strings(self, tmp_path):
        import polars as pl

        source = tmp_path / "typed_datetimes.xlsx"
        write_minimal_xlsx(
            source,
            sheet_name="SAMPLE",
            headers=["id_sample", "onderzoeksdatum"],
            rows=[
                ["sample_a", "2025-10-13 00:00:00"],
                ["sample_b", "2025-10-14 13:45:12"],
            ],
        )

        excel_io = ExcelIO()
        result = excel_io.load_section(
            source,
            section_name="SAMPLE",
            data_schema={
                "id_sample": "string",
                "onderzoeksdatum": "datetime",
            },
            cast_error_policy="raise",
        )

        assert result.schema["onderzoeksdatum"] == pl.Datetime
        assert [
            value.isoformat(sep=" ") for value in result["onderzoeksdatum"]
        ] == [
            "2025-10-13 00:00:00",
            "2025-10-14 13:45:12",
        ]

    def test_typed_sheet_date_accepts_midnight_native_datetimes(
        self, monkeypatch
    ):
        import polars as pl

        from pypeh.adapters.persistence.dataframe import ExcelIOImpl

        def fake_load(self, source, **options):
            return pl.DataFrame(
                {
                    "id_sample": ["sample_a", "sample_b"],
                    "onderzoeksdatum": [
                        datetime(2025, 10, 13),
                        datetime(2025, 10, 14),
                    ],
                }
            )

        monkeypatch.setattr(ExcelIOImpl, "_load", fake_load)

        result = ExcelIOImpl().load_section(
            "unused.xlsx",
            section_name="SAMPLE",
            data_schema={"id_sample": "string", "onderzoeksdatum": "date"},
            cast_error_policy="raise",
        )

        assert result.schema["onderzoeksdatum"] == pl.Date
        assert [value.isoformat() for value in result["onderzoeksdatum"]] == [
            "2025-10-13",
            "2025-10-14",
        ]

    def test_typed_sheet_date_rejects_non_midnight_datetime_strings(
        self, tmp_path
    ):
        source = tmp_path / "typed_date_with_time.xlsx"
        write_minimal_xlsx(
            source,
            sheet_name="SAMPLE",
            headers=["id_sample", "onderzoeksdatum"],
            rows=[
                ["sample_a", "2025-10-13 00:00:00"],
                ["sample_b", "2025-10-14 13:45:12"],
            ],
        )

        excel_io = ExcelIO()
        with pytest.raises(
            TypeCastError,
            match="would discard non-midnight time values",
        ):
            excel_io.load_section(
                source,
                section_name="SAMPLE",
                data_schema={
                    "id_sample": "string",
                    "onderzoeksdatum": "date",
                },
                cast_error_policy="raise",
            )

    def test_typed_sheet_date_rejects_non_midnight_native_datetimes(
        self, monkeypatch
    ):
        import polars as pl

        from pypeh.adapters.persistence.dataframe import ExcelIOImpl

        def fake_load(self, source, **options):
            return pl.DataFrame(
                {
                    "id_sample": ["sample_a", "sample_b"],
                    "onderzoeksdatum": [
                        datetime(2025, 10, 13),
                        datetime(2025, 10, 14, 13, 45, 12),
                    ],
                }
            )

        monkeypatch.setattr(ExcelIOImpl, "_load", fake_load)

        with pytest.raises(
            TypeCastError,
            match="would discard non-midnight time values",
        ):
            ExcelIOImpl().load_section(
                "unused.xlsx",
                section_name="SAMPLE",
                data_schema={
                    "id_sample": "string",
                    "onderzoeksdatum": "date",
                },
                cast_error_policy="raise",
            )

    def test_typed_sheet_date_rejects_numeric_values_when_requested(
        self, monkeypatch
    ):
        import polars as pl

        from pypeh.adapters.persistence.dataframe import ExcelIOImpl

        def fake_load(self, source, **options):
            return pl.DataFrame(
                {
                    "id_sample": ["sample_a"],
                    "onderzoeksdatum": [45943],
                }
            )

        monkeypatch.setattr(ExcelIOImpl, "_load", fake_load)

        with pytest.raises(
            TypeCastError,
            match="conversion from `Int64` to `date` is not supported",
        ):
            ExcelIOImpl().load_section(
                "unused.xlsx",
                section_name="SAMPLE",
                data_schema={
                    "id_sample": "string",
                    "onderzoeksdatum": "date",
                },
                cast_error_policy="raise",
            )

    def test_typed_sheet_date_loads_numeric_values_as_null(self, monkeypatch):
        import polars as pl

        from pypeh.adapters.persistence.dataframe import ExcelIOImpl

        def fake_load(self, source, **options):
            return pl.DataFrame(
                {
                    "id_sample": ["sample_a"],
                    "onderzoeksdatum": [45943],
                }
            )

        monkeypatch.setattr(ExcelIOImpl, "_load", fake_load)

        result = ExcelIOImpl().load_section(
            "unused.xlsx",
            section_name="SAMPLE",
            data_schema={"id_sample": "string", "onderzoeksdatum": "date"},
            cast_error_policy="null",
        )

        assert result.schema["onderzoeksdatum"] == pl.Date
        assert result["onderzoeksdatum"].to_list() == [None]

    def test_typed_sheet_invalid_date_raises_when_requested(self, tmp_path):
        source = tmp_path / "typed_invalid_date.xlsx"
        write_minimal_xlsx(
            source,
            sheet_name="SAMPLE",
            headers=["id_sample", "onderzoeksdatum"],
            rows=[
                ["sample_a", "2025-10-13 00:00:00"],
                ["sample_b", "not-a-date"],
            ],
        )

        excel_io = ExcelIO()
        with pytest.raises(
            TypeCastError,
            match="Failed to cast Excel sheet 'SAMPLE'",
        ):
            excel_io.load_section(
                source,
                section_name="SAMPLE",
                data_schema={
                    "id_sample": "string",
                    "onderzoeksdatum": "date",
                },
                cast_error_policy="raise",
            )

    def test_typed_sheet_type_mismatch_raises_when_requested(self, tmp_path):
        source = tmp_path / "typed_mismatch.xlsx"
        write_minimal_xlsx(
            source,
            sheet_name="SAMPLE",
            headers=["id_sample", "chol"],
            rows=[
                ["sample_a", 1.2],
                ["sample_b", "oops"],
                ["sample_c", 3.4],
            ],
        )

        excel_io = ExcelIO()
        with pytest.raises(
            TypeCastError,
            match="Failed to cast Excel sheet 'SAMPLE'",
        ):
            excel_io.load_section(
                source,
                section_name="SAMPLE",
                data_schema={"id_sample": "string", "chol": "float"},
                cast_error_policy="raise",
            )

    def test_typed_sheet_type_mismatch_returns_report_when_requested(
        self, tmp_path
    ):
        source = tmp_path / "typed_mismatch.xlsx"
        write_minimal_xlsx(
            source,
            sheet_name="SAMPLE",
            headers=["id_sample", "chol"],
            rows=[
                ["sample_a", 1.2],
                ["sample_b", "oops"],
                ["sample_c", 3.4],
            ],
        )

        excel_io = ExcelIO()
        result = excel_io.load_section(
            source,
            section_name="SAMPLE",
            data_schema={"id_sample": "string", "chol": "float"},
            cast_error_policy="report",
        )

        assert isinstance(result, ValidationErrorReport)
        assert result.total_errors == 1
        assert result.error_counts[ValidationErrorLevel.FATAL] == 1
        assert len(result.groups) == 1
        assert len(result.groups[0].errors) == 1
        error = result.groups[0].errors[0]
        assert error.level == ValidationErrorLevel.FATAL
        assert error.type == "TypeCastError"
        assert "Failed to cast Excel sheet 'SAMPLE'" in error.message


@pytest.mark.core
class TestDump:
    @pytest.fixture(scope="class")
    def container(self) -> CacheContainer:
        source = get_absolute_path(
            "./input/config_basic/_Reference_YAML/observable_properties.yaml"
        )
        yaml_io = YamlIO()
        entity_list = yaml_io.load(source)
        assert isinstance(entity_list, EntityList)
        cache = CacheContainerFactory.new()
        cache.unpack_entity_list(entity_list=entity_list)
        return cache

    def test_dump_cache_yaml(self, container):
        entity_list = container.pack_entity_list()
        adapter = IOAdapterFactory.create(format="yaml")
        buffer = io.StringIO()
        loader = linkml_runtime.loaders.YAMLLoader()

        adapter.dump(entity_list, buffer)
        data = buffer.getvalue()
        assert len(data) > 0
        new_entity_list = loader.load_any(source=data, target_class=EntityList)
        assert isinstance(entity_list, EntityList)
        assert entity_list == new_entity_list

    @pytest.mark.parametrize("format", ["trig", "turtle"])
    def test_dump_cache_rdf(self, container, format):
        entity_list = container.pack_entity_list()
        adapter = IOAdapterFactory.create(format=format)
        buffer = io.BytesIO()

        adapter.dump(entity_list, buffer)
        data = buffer.getvalue()
        assert data, "RDF serialization is empty"
        if format == "trig":
            g = rdflib.Dataset()
            g.parse(data=data, format="trig")
            assert len(g) > 0
        else:
            g = rdflib.Graph()
            g.parse(data=data, format=format)
            ns = dict(g.namespaces())
            assert "peh" in ns
            assert "pehterms" in ns
            OP = rdflib.URIRef(ns["pehterms"] + "ObservableProperty")
            observable_properties = set(g.subjects(rdflib.RDF.type, OP))
            assert (
                observable_properties
            ), "No ObservableProperty instances found"
            EL = rdflib.URIRef(ns["pehterms"] + "EntityList")
            assert (None, rdflib.RDF.type, EL) in g, "No EntityList found"
            entity_lists = list(g.subjects(rdflib.RDF.type, EL))
            assert entity_lists, "No EntityList subjects found"
            linked_observable_properties = set()
            for el in entity_lists:
                props = {
                    obj
                    for obj in g.objects(el)
                    if obj in observable_properties
                }
                assert props, "EntityList has no ObservableProperty links"
                linked_observable_properties.update(props)
            assert observable_properties == linked_observable_properties
            for p in observable_properties:
                assert (
                    p,
                    rdflib.RDFS.label,
                    None,
                ) in g, f"Observable property {p} has no rdfs label"
