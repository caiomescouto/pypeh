import fsspec
import pytest

from pathlib import Path
from peh_model.peh import EntityList

from pypeh.adapters.persistence.hosts import (
    DirectoryIO,
    FileIO,
    LocalStorageProvider,
    S3StorageProvider,
)
from pypeh.core.models.settings import (
    LocalFileConfig,
    LocalFileSettings,
    S3Config,
    S3Settings,
)
from pypeh.core.session.connections import ConnectionManager

from tests.test_utils.dirutils import get_absolute_path


class TestFileIO:
    @pytest.mark.core
    @pytest.mark.parametrize(
        "file_path, expected_format",
        [
            ("example.yaml", "yaml"),
            ("config.json", "json"),
            ("data.TXT", "txt"),
            ("no_extension", ""),
        ],
    )
    def test_get_format(self, file_path, expected_format):
        assert FileIO.get_format(file_path) == expected_format

    @pytest.mark.core
    def test_basic(self):
        source = get_absolute_path(
            "./input/config_basic/_Reference_YAML/observable_entities.yaml"
        )
        fio = FileIO(fsspec.filesystem("file"))
        data = fio.load(source)
        assert isinstance(data, EntityList)

    @pytest.mark.core
    def test_unknown_format(self):
        pass

    @pytest.mark.core
    def test_incompatible_file(self):
        pass


@pytest.mark.dataframe
class TestFileIOCsv:
    def test_basic(self):
        source = get_absolute_path(
            "./input/config_basic/_Tabular_Data/sampling_data_to_import.csv"
        )
        fio = FileIO(fsspec.filesystem("file"))
        data = fio.load(source, raise_if_empty=False)
        from polars import DataFrame

        assert isinstance(data, DataFrame)


@pytest.mark.dataframe
class TestFileIOExcel:
    def test_basic(self):
        source = get_absolute_path(
            "./input/config_basic/_Tabular_Data/sampling_data_to_import.xlsx"
        )
        fio = FileIO(fsspec.filesystem("file"))
        data = fio.load(source)
        assert isinstance(data, dict)


class TestDirectoryIO:
    @pytest.fixture(scope="class")
    def provider(self):
        abs_root_folder = get_absolute_path("./input")
        config = LocalFileConfig(config_dict={"root_folder": abs_root_folder})
        settings = config.make_settings(_env_file=None)
        return ConnectionManager._create_adapter(settings)

    @pytest.mark.dataframe
    def test_basic(self, provider):
        source = "config_basic"
        all_data = list(provider.load(source, maxdepth=10))
        assert len(all_data) > 1

    @pytest.mark.core
    def test_no_root(self, provider):
        source = "config_basic/_Reference_YAML"
        all_data = provider.load(source)
        assert len(all_data) > 0

    @pytest.mark.core
    def test_unknown_format(self, provider):
        source = Path("unknown_format")
        all_data = list(provider.load(str(source)))
        assert len(all_data) == 0

    @pytest.mark.core
    def test_incompatible_file(self, provider):
        source = Path("wrong_input")
        with pytest.raises(TypeError):
            _ = list(provider.load(str(source)))

    @pytest.mark.core
    def test_walk(self, provider):
        source = Path("config_basic")
        i = 0
        for _ in provider.walk(str(source), format="yaml"):
            i += 1
        assert i > 1


@pytest.mark.core
class TestLocalStorageProvider:
    def test_with_settings(self):
        rel_root_folder = "./input/config_basic/_Reference_YAML"
        abs_root_folder = get_absolute_path(rel_root_folder)
        source_file = "observable_entities.yaml"
        config = LocalFileConfig(config_dict={"root_folder": abs_root_folder})
        settings = config.make_settings(_env_file=None)
        provider = ConnectionManager._create_adapter(settings)

        data = provider.load(source_file, format="yaml")
        assert data is not None
        assert isinstance(data, EntityList)


@pytest.mark.s3
class TestS3StorageProvider:
    def test_basic(self, monkeypatch):
        monkeypatch.setenv("MYBUCKET_BUCKET_NAME", "my-test-bucket")
        monkeypatch.setenv(
            "MYBUCKET_ENDPOINT_URL", "http://endpoint-example.local"
        )
        override = {"aws_region": "eu-central-1"}

        config_base = S3Config(env_prefix="MYBUCKET_", config_dict=override)
        settings = config_base.make_settings()
        s3io = S3StorageProvider(settings)
        assert s3io is not None

    def test_path(self, monkeypatch):
        monkeypatch.setenv("MYBUCKET_BUCKET_NAME", "my-test-bucket")
        monkeypatch.setenv(
            "MYBUCKET_ENDPOINT_URL", "http://endpoint-example.local"
        )
        override = {"aws_region": "eu-central-1"}

        config_base = S3Config(env_prefix="MYBUCKET_", config_dict=override)
        settings = config_base.make_settings()
        s3io = S3StorageProvider(settings)
        assert s3io is not None
        test = s3io._normalize_root("s3://my-test-bucket")
        assert test == "my-test-bucket"

    def test_remote_path_normalization_is_idempotent_under_root(self):
        remote_io = DirectoryIO(
            root="my-test-bucket/base-prefix",
            protocol="memory",
        )

        assert (
            remote_io._normalize_path("output")
            == "my-test-bucket/base-prefix/output"
        )
        assert (
            remote_io._normalize_path("my-test-bucket/base-prefix/output")
            == "my-test-bucket/base-prefix/output"
        )


@pytest.mark.s3
class TestDirectoryIOUriNormalization:
    def test_scheme_uri_same_bucket_is_honored(self):
        remote_io = DirectoryIO(root="my-bucket", protocol="s3")

        assert (
            remote_io._normalize_path("s3://my-bucket/out/a.parquet")
            == "my-bucket/out/a.parquet"
        )

    def test_scheme_uri_foreign_bucket_raises(self):
        remote_io = DirectoryIO(root="my-bucket", protocol="s3")

        with pytest.raises(ValueError, match="other-bucket"):
            remote_io._normalize_path("s3://other-bucket/out/a.parquet")

    def test_scheme_uri_same_bucket_is_absolute_under_prefixed_root(self):
        remote_io = DirectoryIO(root="my-bucket/base-prefix", protocol="s3")

        assert (
            remote_io._normalize_path("s3://my-bucket/other/out")
            == "my-bucket/other/out"
        )

    def test_scheme_uri_foreign_bucket_raises_under_prefixed_root(self):
        remote_io = DirectoryIO(root="my-bucket/base-prefix", protocol="s3")

        with pytest.raises(ValueError, match="other-bucket"):
            remote_io._normalize_path("s3://other-bucket/out")

    def test_relative_path_still_roots_under_root(self):
        remote_io = DirectoryIO(root="my-bucket/base-prefix", protocol="s3")

        assert (
            remote_io._normalize_path("out/a.parquet")
            == "my-bucket/base-prefix/out/a.parquet"
        )

    def test_memory_scheme_uri_same_bucket_is_honored(self):
        remote_io = DirectoryIO(
            root="example-bucket/base-prefix", protocol="memory"
        )

        assert (
            remote_io._normalize_path(
                "memory://example-bucket/base-prefix/out"
            )
            == "example-bucket/base-prefix/out"
        )

    def test_memory_scheme_uri_foreign_bucket_raises(self):
        remote_io = DirectoryIO(
            root="example-bucket/base-prefix", protocol="memory"
        )

        with pytest.raises(ValueError, match="other"):
            remote_io._normalize_path("memory://other/out")


@pytest.mark.core
class TestDirectoryIOSchemeMismatch:
    def test_s3_uri_on_file_connection_raises(self, tmp_path):
        local_io = DirectoryIO(root=str(tmp_path), protocol="file")

        with pytest.raises(ValueError, match="scheme 's3'"):
            local_io._normalize_path("s3://other-bucket/out")

    def test_file_uri_is_treated_as_absolute(self, tmp_path):
        local_io = DirectoryIO(root=str(tmp_path), protocol="file")
        target = tmp_path / "abs" / "a.parquet"
        # Build a portable file URI: ``file:///C:/...`` on Windows and
        # ``file:///tmp/...`` on POSIX.
        uri = "file:///" + target.as_posix().lstrip("/")

        assert local_io._normalize_path(uri) == str(target)

    def test_relative_and_absolute_paths_are_unchanged(self, tmp_path):
        local_io = DirectoryIO(root=str(tmp_path), protocol="file")

        assert local_io._normalize_path("out/a.parquet") == str(
            tmp_path / "out" / "a.parquet"
        )

        absolute = str(tmp_path / "abs" / "out" / "a.parquet")
        assert local_io._normalize_path(absolute) == absolute


@pytest.mark.core
class TestDatabaseProvider:
    def test_basic(self):
        # check ability to connect to pre-existing sql-db
        assert True


@pytest.mark.core
class TestWebserviceProvider:
    def test_basic(self):
        # use mock here
        assert True

    def test_yaml(self):
        pass

    def test_json(self):
        pass


class TestHostFactory:
    @pytest.mark.core
    def test_factory_local(self):
        file_settings = LocalFileSettings()
        file_service = ConnectionManager._create_adapter(file_settings)
        assert isinstance(file_service, LocalStorageProvider)

    @pytest.mark.s3
    def test_factory_s3(self):
        file_settings = S3Settings(
            aws_access_key_id="test",
            aws_secret_access_key="test",
            aws_session_token="test",
            bucket_name="TEST",
        )
        file_service = ConnectionManager._create_adapter(file_settings)
        assert isinstance(file_service, S3StorageProvider)
