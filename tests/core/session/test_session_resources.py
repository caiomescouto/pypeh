import pytest
import re
import yaml
import inspect

from peh_model.peh import Observation, ObservableProperty, DerivedObservation

from pypeh import Session
from pypeh.core.cache.containers import CacheContainerView
from pypeh.core.interfaces.dataops import (
    AggregationInterface,
    DataEnrichmentInterface,
)
from pypeh.core.models.internal_data_layout import DatasetSeries
from pypeh.core.models.settings import LocalFileConfig

from pypeh.core.utils.namespaces import NamespaceManager
from tests.test_utils.dirutils import get_absolute_path


def get_session(
    root_folder: str = "./input/default_localfile_data",
) -> Session:
    session = Session(
        connection_config=[
            LocalFileConfig(
                label="local_file",
                config_dict={
                    "root_folder": get_absolute_path(root_folder),
                },
            ),
        ],
        default_connection=None,
    )
    return session


@pytest.mark.core
class TestSessionResource:
    def test_load_resource(self):
        session = get_session()
        resource_identifier = "peh:OBSERVATION_ADULTS_URINE_LAB"
        resource_type = "Observation"
        resource_path = "observations.yaml"
        connection_label = "local_file"
        ret = session.load_resource(
            resource_identifier, resource_type, resource_path, connection_label
        )
        assert isinstance(ret, Observation)


@pytest.mark.core
class TestSessionDump:
    @staticmethod
    def _get_tmp_session(tmp_path) -> Session:
        return Session(
            connection_config=[
                LocalFileConfig(
                    label="local_file",
                    config_dict={"root_folder": str(tmp_path)},
                ),
            ],
            default_connection=None,
        )

    def test_dump_entity_list(self, tmp_path):
        session = get_session()
        assert isinstance(session, Session)
        session.load_persisted_cache(
            source="observations.yaml", connection_label="local_file"
        )
        dest = tmp_path / "out.yaml"
        session.dump_cache(
            output_path=dest,
            connection_label="local_file",
        )
        data = dest.read_bytes()
        assert data, "Dumped file is empty"
        test_data = yaml.safe_load(data.decode("utf-8"))
        assert isinstance(test_data, dict)
        assert "observations" in test_data

    def test_dump_entity_list_includes_derived_observations(self, tmp_path):
        session = get_session("./input/unpack_resource")
        assert isinstance(session, Session)
        session.load_persisted_cache(
            source="unpack_derived_observation_group.yaml",
            connection_label="local_file",
        )
        dest = tmp_path / "out.yaml"
        session.dump_cache(
            output_path=dest,
            connection_label="local_file",
        )
        data = dest.read_bytes()
        assert data, "Dumped file is empty"
        test_data = yaml.safe_load(data.decode("utf-8"))
        assert isinstance(test_data, dict)
        assert "derived_observations" in test_data
        assert len(test_data["derived_observations"]) == 3

    def test_dumped_yaml_with_non_ascii_reads_back_as_utf8(self, tmp_path):
        source_session = self._get_tmp_session(tmp_path)
        source_session.cache.add(
            ObservableProperty(
                id="peh:unicode_property",
                ui_label="caf\u00e9",
                value_type="string",
            )
        )

        destination = tmp_path / "unicode_cache.yaml"
        source_session.dump_cache(
            output_path=str(destination),
            connection_label="local_file",
        )

        data = destination.read_bytes()
        assert data.decode("utf-8")

        loaded_session = self._get_tmp_session(tmp_path)
        loaded_session.load_persisted_cache(
            source="unicode_cache.yaml", connection_label="local_file"
        )

        loaded = loaded_session.cache.require(
            "peh:unicode_property", "ObservableProperty"
        )
        assert isinstance(loaded, ObservableProperty)
        assert loaded.ui_label == "caf\u00e9"

    def test_loading_cp1252_yaml_raises_unicode_decode_error(self, tmp_path):
        cp1252_yaml = (
            b"observable_properties:\n"
            b"  - id: peh:cp1252_property\n"
            b"    ui_label: caf\xe9\n"
            b"    value_type: string\n"
        )
        (tmp_path / "cp1252_cache.yaml").write_bytes(cp1252_yaml)

        session = self._get_tmp_session(tmp_path)
        with pytest.raises(UnicodeDecodeError, match="UTF-8 encoded"):
            session.load_persisted_cache(
                source="cp1252_cache.yaml", connection_label="local_file"
            )


@pytest.mark.core
class TestSessionMint:
    def test_mint_and_cache(self):
        session = get_session()
        assert isinstance(session, Session)
        namespace_manager = NamespaceManager()
        namespace_manager.bind("test", "www.example.com")
        session.bind_namespace_manager(namespace_manager=namespace_manager)
        ret = session.mint_and_cache(
            ObservableProperty, namespace_key="test", ui_label="test"
        )
        next_instance = next(session.cache.get_all("ObservableProperty"))
        assert isinstance(next_instance, ObservableProperty)
        assert next_instance.id == ret.id

    def test_mint_and_cache_resource(self):
        session = get_session()
        assert isinstance(session, Session)
        namespace_manager = NamespaceManager(
            default_base_uri="https://w3id.org/example/id/"
        )
        session.bind_namespace_manager(namespace_manager=namespace_manager)
        ret = session.mint_and_cache(ObservableProperty, ui_label="test")
        next_instance = next(session.cache.get_all("ObservableProperty"))
        assert isinstance(next_instance, ObservableProperty)
        assert next_instance.id == ret.id
        pattern = r"^https://w3id\.org/example/id/observable-property/[0-9A-HJKMNP-TV-Z]{26}$"
        assert re.match(
            pattern, ret.id
        ), f"IRI did not match expected pattern: {ret.id}"


@pytest.mark.core
class TestSessionUnpack:
    def test_unpack_derived_observation_group(self):
        session = get_session("./input/unpack_resource")
        assert isinstance(session, Session)
        session.load_persisted_cache(
            source="unpack_derived_observation_group.yaml",
            connection_label="local_file",
        )
        count = 0
        for target, source in session.unpack_derived_observation_group(
            observation_group_id="example:this_group"
        ):
            assert isinstance(target, DerivedObservation)
            assert isinstance(source, Observation)
            count += 1
        assert count == 3


class RecordingAggregationAdapter(AggregationInterface):
    def __init__(self, result: DatasetSeries):
        self.calls: list[dict] = []
        self._result = result

    def select_field(self, dataset, field_label: str):
        return None

    def get_element_labels(self, data):
        return []

    def get_element_values(self, data, element_label: str, as_list=True):
        return []

    def check_element_has_empty_values(self, data, element_label: str) -> bool:
        return False

    def check_element_has_only_empty_values(
        self, data, element_label: str
    ) -> bool:
        return False

    def subset(
        self,
        data,
        element_group: list[str],
        id_group=None,
        identifying_elements=None,
    ):
        return data

    def collect(self, datasets: dict):
        return datasets

    def type_mapper(self, peh_value_type):
        return peh_value_type

    def _calculate_for_stratum(
        self, df, group_cols, value_col: str, stat_builders: list, **kwargs
    ):
        return df

    def calculate_for_strata(
        self,
        df,
        stratifications,
        value_col: str,
        stat_builders: list[str],
        **kwargs,
    ):
        return df

    def group_results(self, results_to_collect: list, strata=None):
        return results_to_collect[0]

    def summarize(
        self,
        source_dataset_series: DatasetSeries,
        target_observations: list[Observation],
        target_derived_from: list[Observation],
        cache_view: CacheContainerView,
        target_label_collision_strategy: str = "error",
    ) -> DatasetSeries:
        self.calls.append(
            {
                "source_dataset_series": source_dataset_series,
                "target_observations": target_observations,
                "target_derived_from": target_derived_from,
                "cache_view": cache_view,
                "target_label_collision_strategy": (
                    target_label_collision_strategy
                ),
            }
        )
        return self._result


class RecordingEnrichmentAdapter(DataEnrichmentInterface):
    def __init__(self, result: DatasetSeries):
        self.calls: list[dict] = []
        self._result = result

    def select_field(self, dataset, field_label: str):
        return None

    def get_element_labels(self, data):
        return []

    def get_element_values(self, data, element_label: str, as_list=True):
        return []

    def check_element_has_empty_values(self, data, element_label: str) -> bool:
        return False

    def check_element_has_only_empty_values(
        self, data, element_label: str
    ) -> bool:
        return False

    def subset(
        self,
        data,
        element_group: list[str],
        id_group=None,
        identifying_elements=None,
    ):
        return data

    def collect(self, datasets: dict):
        return datasets

    def type_mapper(self, peh_value_type):
        return peh_value_type

    def apply_map(
        self, dataset, map_fn, field_label, output_dtype, base_fields, **kwargs
    ):
        return dataset

    def map_type(self, peh_value_type: str):
        return peh_value_type

    def enrich(
        self,
        source_dataset_series: DatasetSeries,
        target_observations: list[Observation],
        target_derived_from: list[Observation],
        cache_view: CacheContainerView,
        target_label_collision_strategy: str = "error",
    ) -> DatasetSeries:
        self.calls.append(
            {
                "source_dataset_series": source_dataset_series,
                "target_observations": target_observations,
                "target_derived_from": target_derived_from,
                "cache_view": cache_view,
                "target_label_collision_strategy": (
                    target_label_collision_strategy
                ),
            }
        )
        return self._result

    def split_by_observation(
        self,
        dataset_series: DatasetSeries,
        *,
        new_label: str | None = None,
        cache_view: CacheContainerView | None = None,
        label_collision_strategy: str = "prefix_source_dataset",
    ) -> DatasetSeries:
        self.calls.append(
            {
                "dataset_series": dataset_series,
                "new_label": new_label,
                "cache_view": cache_view,
                "label_collision_strategy": label_collision_strategy,
            }
        )
        return self._result


@pytest.mark.core
class TestAdapterSignatureContracts:
    @staticmethod
    def _signature_without_annotations(func):
        sig = inspect.signature(func)
        params = [
            param.replace(annotation=inspect.Signature.empty)
            for param in sig.parameters.values()
        ]
        return sig.replace(
            parameters=params,
            return_annotation=inspect.Signature.empty,
        )

    def test_recording_enrichment_adapter_enrich_signature_matches_interface(
        self,
    ):
        assert self._signature_without_annotations(
            RecordingEnrichmentAdapter.enrich
        ) == self._signature_without_annotations(
            DataEnrichmentInterface.enrich
        )

    def test_recording_enrichment_adapter_split_signature_matches_interface(
        self,
    ):
        assert self._signature_without_annotations(
            RecordingEnrichmentAdapter.split_by_observation
        ) == self._signature_without_annotations(
            DataEnrichmentInterface.split_by_observation
        )

    def test_recording_aggregation_adapter_summarize_signature_matches_interface(
        self,
    ):
        assert self._signature_without_annotations(
            RecordingAggregationAdapter.summarize
        ) == self._signature_without_annotations(
            AggregationInterface.summarize
        )


@pytest.mark.core
class TestSessionAggregate:
    @staticmethod
    def _make_observation(label: str) -> Observation:
        return Observation(
            id=f"peh:{label}",
            ui_label=label,
            observation_design="peh:test_observation_design",
        )

    def test_aggregate_delegates_to_adapter(self):
        session = get_session()
        source_dataset_series = DatasetSeries(label="source")
        expected = DatasetSeries(label="summary")
        adapter = RecordingAggregationAdapter(result=expected)
        session.register_adapter("aggregation", adapter)

        target_observations = [
            self._make_observation("target_a"),
            self._make_observation("target_b"),
        ]
        target_derived_from = [
            self._make_observation("source_a"),
            self._make_observation("source_b"),
        ]
        target_dataset_labels = ["TARGET_A", "TARGET_B"]

        result = session.aggregate(
            source_dataset_series=source_dataset_series,
            target_observations=target_observations,
            target_derived_from=target_derived_from,
            target_dataset_labels=target_dataset_labels,
            target_label_collision_strategy="prefix_source_dataset",
        )

        assert result is expected
        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        assert call["source_dataset_series"] is source_dataset_series
        assert call["target_observations"] is target_observations
        assert call["target_derived_from"] is target_derived_from
        assert (
            call["target_label_collision_strategy"] == "prefix_source_dataset"
        )
        assert isinstance(call["cache_view"], CacheContainerView)
        assert call["cache_view"]._container is session.cache

    def test_aggregate_requires_matching_target_lengths(self):
        session = get_session()
        source_dataset_series = DatasetSeries(label="source")
        with pytest.raises(AssertionError):
            session.aggregate(
                source_dataset_series=source_dataset_series,
                target_observations=[self._make_observation("target_a")],
                target_derived_from=[
                    self._make_observation("source_a"),
                    self._make_observation("source_b"),
                ],
            )


@pytest.mark.core
class TestSessionEnrich:
    @staticmethod
    def _make_observation(label: str) -> Observation:
        return Observation(
            id=f"peh:{label}",
            ui_label=label,
            observation_design="peh:test_observation_design",
        )

    def test_enrich_delegates_to_adapter(self):
        session = get_session()
        source_dataset_series = DatasetSeries(label="source")
        expected = DatasetSeries(label="enriched")
        adapter = RecordingEnrichmentAdapter(result=expected)
        session.register_adapter("enrichment", adapter)

        target_observations = [
            self._make_observation("target_a"),
            self._make_observation("target_b"),
        ]
        target_derived_from = [
            self._make_observation("source_a"),
            self._make_observation("source_b"),
        ]
        target_dataset_labels = ["TARGET_A", "TARGET_B"]

        result = session.enrich(
            source_dataset_series=source_dataset_series,
            target_observations=target_observations,
            target_derived_from=target_derived_from,
            target_dataset_labels=target_dataset_labels,
            target_label_collision_strategy="prefix_source_dataset",
        )

        assert result is expected
        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        assert call["source_dataset_series"] is source_dataset_series
        assert call["target_observations"] is target_observations
        assert call["target_derived_from"] is target_derived_from
        assert (
            call["target_label_collision_strategy"] == "prefix_source_dataset"
        )
        assert isinstance(call["cache_view"], CacheContainerView)
        assert call["cache_view"]._container is session.cache

    def test_enrich_requires_matching_target_lengths(self):
        session = get_session()
        source_dataset_series = DatasetSeries(label="source")
        with pytest.raises(AssertionError):
            session.enrich(
                source_dataset_series=source_dataset_series,
                target_observations=[self._make_observation("target_a")],
                target_derived_from=[
                    self._make_observation("source_a"),
                    self._make_observation("source_b"),
                ],
            )


@pytest.mark.core
class TestSessionSplitDatasetSeriesByObservation:
    def test_split_dataset_series_by_observation_delegates_to_adapter(self):
        session = get_session()
        source_dataset_series = DatasetSeries(label="source")
        expected = DatasetSeries(label="split")
        adapter = RecordingEnrichmentAdapter(result=expected)
        session.register_adapter("dataops", adapter)

        result = session.split_dataset_series_by_observation(
            source_dataset_series=source_dataset_series,
            new_dataset_series_label="custom_split",
            label_collision_strategy="prefix_source_dataset",
        )

        assert result is expected
        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        assert call["dataset_series"] is source_dataset_series
        assert call["new_label"] == "custom_split"
        assert call["label_collision_strategy"] == "prefix_source_dataset"
        assert isinstance(call["cache_view"], CacheContainerView)
        assert call["cache_view"]._container is session.cache

    def test_split_dataset_series_by_observation_uses_default_label(self):
        session = get_session()
        source_dataset_series = DatasetSeries(label="source")
        expected = DatasetSeries(label="split")
        adapter = RecordingEnrichmentAdapter(result=expected)
        session.register_adapter("dataops", adapter)

        result = session.split_dataset_series_by_observation(
            source_dataset_series=source_dataset_series,
        )

        assert result is expected
        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        assert call["dataset_series"] is source_dataset_series
        assert call["new_label"] is None
        assert call["label_collision_strategy"] == "prefix_source_dataset"
        assert isinstance(call["cache_view"], CacheContainerView)
        assert call["cache_view"]._container is session.cache
