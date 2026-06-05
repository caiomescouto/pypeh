import importlib

import pytest

from pypeh.adapters.persistence.dataset_parquet import (
    dump_dataset_series_to_parquet,
    dump_dataset_series_to_parquet_filesystem,
    load_dataset_series_from_parquet,
    load_dataset_series_from_parquet_filesystem,
)
from pypeh.core.models.constants import ObservablePropertyValueType
from pypeh.core.models.internal_data_layout import Dataset, DatasetSeries


pytestmark = pytest.mark.dataframe


@pytest.fixture
def dataset_series():
    pl = importlib.import_module("polars")

    series = DatasetSeries(label="example_series")

    sample = series.add_empty_dataset(
        "SAMPLE", metadata={"described_by": "peh:layout_section_sample"}
    )
    series.add_observable_property(
        observation_id="peh:obs_sample",
        observable_property_id="peh:prop_id_sample",
        data_type=ObservablePropertyValueType.STRING,
        dataset_label="SAMPLE",
        element_label="id_sample",
        is_primary_key=True,
    )
    sample.add_observation_to_index("peh:obs_sample")
    sample.data = pl.DataFrame({"id_sample": ["sample-a", "sample-b"]})

    lab = series.add_empty_dataset(
        "LAB", metadata={"described_by": "peh:layout_section_lab"}
    )
    series.add_observable_property(
        observation_id="peh:obs_lab",
        observable_property_id="peh:prop_id_sample",
        data_type=ObservablePropertyValueType.STRING,
        dataset_label="LAB",
        element_label="id_sample",
    )
    series.add_observable_property(
        observation_id="peh:obs_lab",
        observable_property_id="peh:prop_chol",
        data_type=ObservablePropertyValueType.FLOAT,
        dataset_label="LAB",
        element_label="chol",
    )
    lab.add_observation_to_index("peh:obs_lab")
    lab.schema.add_foreign_key_link(
        element_label="id_sample",
        foreign_key_dataset_label="SAMPLE",
        foreign_key_element_label="id_sample",
    )
    lab.data = pl.DataFrame(
        {
            "id_sample": ["sample-a", "sample-b"],
            "chol": [1.2, 3.4],
        }
    )

    return series


def test_dataset_series_roundtrip_preserves_join_and_context(
    tmp_path, dataset_series
):
    dump_dataset_series_to_parquet(dataset_series, tmp_path)

    loaded = load_dataset_series_from_parquet(tmp_path)

    assert set(loaded.parts) == {"SAMPLE", "LAB"}
    join = loaded.resolve_join("LAB", "SAMPLE")
    assert join is not None
    assert join.left_elements == ("id_sample",)
    assert join.right_elements == ("id_sample",)
    assert join.left_dataset == "LAB"
    assert join.right_dataset == "SAMPLE"
    assert loaded.context_lookup("peh:obs_lab", "peh:prop_chol") == (
        "LAB",
        "chol",
    )
    assert loaded.context_lookup("peh:obs_sample", "peh:prop_id_sample") == (
        "SAMPLE",
        "id_sample",
    )
    lab = loaded["LAB"]
    assert isinstance(lab, Dataset)
    assert lab.observation_ids == {"peh:obs_lab"}
    assert lab.data is not None
    assert lab.data.to_dict(as_series=False) == {
        "id_sample": ["sample-a", "sample-b"],
        "chol": [1.2, 3.4],
    }
    assert lab.schema.primary_keys == set()
    assert (
        lab.schema.elements["chol"].observable_property_id == "peh:prop_chol"
    )
    assert (
        lab.schema.elements["chol"].data_type
        == ObservablePropertyValueType.FLOAT
    )
    assert (
        lab.schema.foreign_keys["id_sample"].reference.dataset_label
        == "SAMPLE"
    )
    assert (
        lab.schema.foreign_keys["id_sample"].reference.element_label
        == "id_sample"
    )


def test_dataset_series_load_raises_for_missing_foreign_dataset(
    tmp_path, dataset_series
):
    lab = dataset_series["LAB"]
    assert isinstance(lab, Dataset)
    partial_series = DatasetSeries(label="partial_series")
    partial_series.register_dataset(lab)
    outputs = dump_dataset_series_to_parquet(partial_series, tmp_path)

    with pytest.raises(
        ValueError, match="Foreign key references missing dataset 'SAMPLE'"
    ):
        load_dataset_series_from_parquet(outputs)

    loaded = load_dataset_series_from_parquet(
        outputs, validate_foreign_keys=False
    )
    assert set(loaded.parts) == {"LAB"}
    assert (
        loaded["LAB"].schema.foreign_keys["id_sample"].reference.dataset_label
        == "SAMPLE"
    )


def test_dataset_series_roundtrip_with_fsspec_filesystem(dataset_series):
    fsspec = importlib.import_module("fsspec")
    file_system = fsspec.filesystem("memory")

    outputs = dump_dataset_series_to_parquet_filesystem(
        dataset_series, file_system, "memory-series"
    )
    loaded = load_dataset_series_from_parquet_filesystem(
        file_system, "memory-series"
    )

    assert len(outputs) == 2
    assert all(file_system.exists(path) for path in outputs)
    assert set(loaded.parts) == {"SAMPLE", "LAB"}
    assert loaded.context_lookup("peh:obs_lab", "peh:prop_chol") == (
        "LAB",
        "chol",
    )
    assert loaded.resolve_join("LAB", "SAMPLE") is not None
