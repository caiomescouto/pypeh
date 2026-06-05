import importlib

from pathlib import Path

import peh_model.peh as peh
import pytest

from pypeh import LocalFileConfig, Session
from pypeh.core.models.constants import ValidationErrorLevel
from pypeh.core.models.internal_data_layout import DatasetSeries
from pypeh.core.models.validation_errors import ValidationErrorReport


pytestmark = [
    pytest.mark.integration,
]


INPUT_DIR = Path(__file__).with_name("input")
CONNECTION_LABEL = "local_file"

DATA_IMPORT_CONFIG_ID = (
    "https://w3id.org/peh/id/data-import-config/" "01KMYXDM0D3BMZTKC93YR0MEJQ"
)
ENRICHMENT_OBSERVATION_GROUP_ID = (
    "https://w3id.org/peh/id/observation-group/" "01KMYXDM0D4BMZTKC93YR0MPJV"
)
AGGREGATION_OBSERVATION_GROUP_ID = (
    "https://w3id.org/peh/id/observation-group/" "basic-aggregation-test"
)
EXCEL_SOURCE = "DataExample_PARC_ALIGNED_STUDIES_ADULTS.xlsx"

CACHE_SOURCES = (
    "PAS_adults_derived_variables.yaml",
    "TEST_STUDY-config.yaml",
    "PAS_adults_observable_properties.yaml",
)


@pytest.fixture
def dataframe_dependencies():
    importlib.import_module("polars")
    importlib.import_module("fastexcel")
    importlib.import_module("pandera")


@pytest.fixture
def session(dataframe_dependencies):
    session = Session(
        connection_config=[
            LocalFileConfig(
                label=CONNECTION_LABEL,
                config_dict={"root_folder": str(INPUT_DIR)},
            )
        ],
        default_connection=CONNECTION_LABEL,
    )
    for source in CACHE_SOURCES:
        session.load_persisted_cache(
            source,
            connection_label=CONNECTION_LABEL,
        )
    return session


@pytest.fixture
def data_import_config(session):
    data_import_config = session.load_resource(
        DATA_IMPORT_CONFIG_ID,
        resource_type="DataImportConfig",
        connection_label=CONNECTION_LABEL,
    )
    assert isinstance(data_import_config, peh.DataImportConfig)
    return data_import_config


def _assert_dataset_series_has_data(dataset_series):
    assert isinstance(dataset_series, DatasetSeries)
    assert len(dataset_series) > 0

    datasets_with_data = [
        dataset_label
        for dataset_label in dataset_series
        if dataset_series[dataset_label] is not None
        and dataset_series[dataset_label].data is not None
    ]
    assert datasets_with_data


def _assert_validation_reports_are_well_formed(validation_reports):
    assert isinstance(validation_reports, dict)
    assert validation_reports

    for report in validation_reports.values():
        assert isinstance(report, ValidationErrorReport)
        assert report.error_counts[ValidationErrorLevel.FATAL] == 0
        assert len(report.unexpected_errors) == 0
        assert sum(report.error_counts.values()) == report.total_errors


def test_parc_aligned_study_dataops_roundtrip(
    session,
    data_import_config,
    tmp_path,
):
    dataset_series = session.import_tabular_dataset_series(
        source=EXCEL_SOURCE,
        data_import_config=data_import_config,
        connection_label=CONNECTION_LABEL,
    )
    _assert_dataset_series_has_data(dataset_series)

    validation_reports = session.validate_tabular_dataset_series(
        dataset_series=dataset_series,
        allow_incomplete=True,
    )
    _assert_validation_reports_are_well_formed(validation_reports)

    observation_dataset_series = session.split_dataset_series_by_observation(
        dataset_series
    )
    _assert_dataset_series_has_data(observation_dataset_series)

    parquet_paths = session.dump_tabular_dataset_series(
        observation_dataset_series,
        output_path=str(tmp_path / "roundtrip"),
        connection_label=CONNECTION_LABEL,
    )
    assert parquet_paths
    assert all(Path(path).is_file() for path in parquet_paths)

    loaded_dataset_series = session.read_tabular_dataset_series(
        parquet_paths,
        file_format="parquet",
        connection_label=CONNECTION_LABEL,
    )

    _assert_dataset_series_has_data(loaded_dataset_series)
    assert set(loaded_dataset_series.parts) == set(
        observation_dataset_series.parts
    )

    target_observations, source_observations = zip(
        *session.unpack_derived_observation_group(
            ENRICHMENT_OBSERVATION_GROUP_ID
        )
    )
    enriched_dataset_series = session.enrich(
        loaded_dataset_series,
        target_observations=list(target_observations),
        target_derived_from=list(source_observations),
    )

    _assert_dataset_series_has_data(enriched_dataset_series)

    dest = session.dump_tabular_dataset_series(
        enriched_dataset_series,
        output_path=str(tmp_path / "roundtrip"),
        connection_label=CONNECTION_LABEL,
        file_format="xlsx",
    )
    assert len(dest) == 1
    assert dest[0].split("/")[-1] == enriched_dataset_series.label + ".xlsx"

    target_observations, source_observations = zip(
        *session.unpack_derived_observation_group(
            AGGREGATION_OBSERVATION_GROUP_ID
        )
    )
    print(target_observations)
    print(source_observations)
    aggregated_dataset_series = session.aggregate(
        source_dataset_series=enriched_dataset_series,
        target_observations=list(target_observations),
        target_derived_from=list(source_observations),
    )
    _assert_dataset_series_has_data(aggregated_dataset_series)
