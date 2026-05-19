import pytest

from contextlib import contextmanager

from pypeh import LocalFileConfig, Session
from pypeh.adapters.persistence.hosts import DirectoryIO
from pypeh.core.models.constants import ObservablePropertyValueType
from pypeh.core.models.internal_data_layout import Dataset, DatasetSeries


@pytest.fixture
def parquet_session(tmp_path):
    return Session(
        connection_config=[
            LocalFileConfig(
                label="local_file",
                config_dict={"root_folder": str(tmp_path)},
            )
        ],
        default_connection=None,
    )


class _SingleConnectionManager:
    def __init__(self, connection):
        self.connection = connection

    @contextmanager
    def get_connection(self, connection_label=None):
        yield self.connection


@pytest.fixture
def s3_like_parquet_session():
    session = Session(default_connection=None)
    connection = DirectoryIO(
        root="example-bucket/base-prefix",
        protocol="memory",
    )
    session.connection_manager = _SingleConnectionManager(connection)
    return session, connection


@pytest.fixture
def dataset_series():
    pl = pytest.importorskip("polars")

    series = DatasetSeries(label="session_series")
    sample = series.add_empty_dataset("SAMPLE")
    sample.add_observation_to_index("peh:obs_sample")
    series.add_observable_property(
        observation_id="peh:obs_sample",
        observable_property_id="peh:prop_id_sample",
        data_type=ObservablePropertyValueType.STRING,
        dataset_label="SAMPLE",
        element_label="id_sample",
        is_primary_key=True,
    )
    sample.data = pl.DataFrame({"id_sample": ["sample-a", "sample-b"]})

    lab = series.add_empty_dataset("LAB")
    lab.add_observation_to_index("peh:obs_lab")
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
    lab.schema.add_foreign_key_link(
        element_label="id_sample",
        foreign_key_dataset_label="SAMPLE",
        foreign_key_element_label="id_sample",
    )
    lab.data = pl.DataFrame(
        {"id_sample": ["sample-a", "sample-b"], "chol": [1.2, 3.4]}
    )

    return series


@pytest.mark.dataframe
class TestSessionParquet:
    def test_session_dataset_series_parquet_roundtrip(
        self, parquet_session, dataset_series
    ):
        source_paths = parquet_session.dump_tabular_dataset_series(
            dataset_series, "series", connection_label="local_file"
        )

        loaded = parquet_session.read_tabular_dataset_series(
            source_paths, file_format="parquet", connection_label="local_file"
        )

        assert set(loaded.parts) == {"SAMPLE", "LAB"}
        assert loaded.context_lookup("peh:obs_lab", "peh:prop_chol") == (
            "LAB",
            "chol",
        )
        join = loaded.resolve_join("LAB", "SAMPLE")
        assert join is not None
        assert join.left_elements == ("id_sample",)
        assert join.right_elements == ("id_sample",)
        lab_dataset = loaded["LAB"]
        assert isinstance(lab_dataset, Dataset)
        lab_data = lab_dataset.data
        assert lab_data is not None
        assert lab_data.shape == (2, 2)

    def test_session_dump_s3_like_paths_are_filesystem_relative(
        self, s3_like_parquet_session, dataset_series
    ):
        session, connection = s3_like_parquet_session

        source_paths = session.dump_tabular_dataset_series(
            dataset_series,
            connection_label="data",
        )

        assert source_paths == [
            "example-bucket/base-prefix/SAMPLE.parquet",
            "example-bucket/base-prefix/LAB.parquet",
        ]
        assert all(not path.startswith("s3://") for path in source_paths)
        assert all(
            connection.file_system.exists(path) for path in source_paths
        )

    def test_session_s3_like_dump_paths_roundtrip_through_session_read(
        self, s3_like_parquet_session, dataset_series
    ):
        session, _ = s3_like_parquet_session

        source_paths = session.dump_tabular_dataset_series(
            dataset_series,
            connection_label="data",
        )
        loaded = session.read_tabular_dataset_series(
            source_paths,
            file_format="parquet",
            connection_label="data",
        )

        assert set(loaded.parts) == {"SAMPLE", "LAB"}

    def test_session_read_dataset_series_requires_explicit_files(
        self, parquet_session
    ):
        with pytest.raises(
            TypeError,
            match="expects source_paths to be a sequence of parquet file paths",
        ):
            parquet_session.read_tabular_dataset_series(
                "series", connection_label="local_file"
            )
