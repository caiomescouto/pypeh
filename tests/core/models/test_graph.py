import pytest
import itertools

from pypeh.adapters.persistence.hosts import DirectoryIO
from pypeh.core.cache.containers import (
    CacheContainerFactory,
    CacheContainerView,
)
from pypeh.core.cache.utils import load_entities_from_tree
from pypeh.core.models.graph import Delayed, Graph, Node
from pypeh.core.interfaces.dataops import DataEnrichmentInterface
from pypeh.core.models.internal_data_layout import (
    ContextIndexProtocol,
    JoinSpec,
)
from tests.test_utils.dirutils import get_absolute_path


@pytest.mark.core
class TestGraph:
    def test_add_node(self):
        g = Graph()
        g._add_node(Node("A", "A"))
        assert Node("A", "A") in g.nodes
        assert Node("A", "A") in g.graph
        assert g.graph[Node("A", "A")] == set()

    def test_add_edge(self):
        g = Graph()
        g.add_edge(Node("A", "A"), Node("B", "B"))
        assert Node("A", "A") in g.nodes
        assert Node("B", "B") in g.nodes
        assert Node("B", "B") in g.graph[Node("A", "A")]
        assert g.graph[Node("B", "B")] == set()

    def test_get_edges(self):
        g = Graph()
        g.add_edge(Node("A", "A"), Node("B", "B"))
        g.add_edge(Node("A", "A"), Node("C", "C"))
        g.add_edge(Node("B", "B"), Node("C", "C"))
        edges = g.edges
        expected_edges = [
            (Node("A", "A"), Node("B", "B")),
            (Node("A", "A"), Node("C", "C")),
            (Node("B", "B"), Node("C", "C")),
        ]
        assert set(edges) == set(expected_edges)

    def test_get_children(self):
        g = Graph()
        g.add_edge(Node("A", "A"), Node("B", "B"))
        g.add_edge(Node("A", "A"), Node("C", "C"))
        children = g.get_children(Node("A", "A"))
        assert children == {Node("B", "B"), Node("C", "C")}
        assert g.get_children(Node("B", "B")) == set()

    def test_get_parents(self):
        g = Graph()
        g.add_edge(Node("A", "A"), Node("B", "B"))
        g.add_edge(Node("C", "C"), Node("B", "B"))
        parents = g.get_parents(Node("B", "B"))
        assert parents == {Node("A", "A"), Node("C", "C")}
        assert g.get_parents(Node("A", "A")) == set()

    def test_topological_sort(self):
        g = Graph()
        g.add_edge(Node("A", "A"), Node("B", "B"))
        g.add_edge(Node("B", "B"), Node("C", "C"))
        g.add_edge(Node("A", "A"), Node("C", "C"))
        g.add_edge(Node("C", "C"), Node("D", "D"))
        g.add_edge(Node("B", "B"), Node("D", "D"))
        g.add_edge(Node("E", "E"), Node("D", "D"))
        sorted_nodes = g.topological_sort()
        assert (
            sorted_nodes.index(Node("A", "A"))
            < sorted_nodes.index(Node("B", "B"))
            < sorted_nodes.index(Node("C", "C"))
            < sorted_nodes.index(Node("D", "D"))
        )

    def test_build_callable_uses_type_mapper(self):
        class RecordingEnrichmentInterface(DataEnrichmentInterface):
            def __init__(self):
                self.output_dtype = None

            def select_field(self, dataset, field_label: str):
                return field_label

            def get_element_labels(self, data):
                return []

            def get_element_values(
                self, data, element_label: str, as_list=True
            ):
                return []

            def check_element_has_empty_values(
                self, data, element_label: str
            ) -> bool:
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
                return f"mapped:{peh_value_type}"

            def map_type(self, peh_value_type: str):
                raise AssertionError("build_callable should use type_mapper")

            def apply_map(
                self,
                dataset,
                map_fn,
                field_label,
                output_dtype,
                base_fields,
                **kwargs,
            ):
                self.output_dtype = output_dtype
                return dataset

        adapter = RecordingEnrichmentInterface()
        delayed = Delayed(map_fn=lambda source: source, output_dtype="decimal")
        delayed.add_parent(Node("dataset", "source"), "source")
        compute = adapter.build_callable(delayed)

        compute(
            {"dataset": object()},
            node=Node("dataset", "derived"),
            base_fields={"dataset": []},
        )

        assert adapter.output_dtype == "mapped:decimal"

    @pytest.mark.parametrize(
        ("raw_value", "value_type", "expected"),
        [
            ("1024", "float", 1024.0),
            ("1024", "integer", 1024),
            ("false", "boolean", False),
            ("1024", "string", "1024"),
        ],
    )
    def test_scalar_value_resolver_uses_type_mapper(
        self, raw_value, value_type, expected
    ):
        class ScalarResolvingEnrichmentInterface(DataEnrichmentInterface):
            def __init__(self):
                self.mapped_value_types = []

            def type_mapper(self, peh_value_type):
                self.mapped_value_types.append(peh_value_type)
                return {
                    "boolean": "Boolean",
                    "float": "Float64",
                    "integer": "Int64",
                    "string": "String",
                }[peh_value_type]

        adapter = ScalarResolvingEnrichmentInterface()

        assert adapter._resolve_scalar_value(raw_value, value_type) == expected
        assert adapter.mapped_value_types == [value_type]

    def test_scalar_value_resolver_requires_value_type(self):
        class ScalarResolvingEnrichmentInterface(DataEnrichmentInterface):
            def type_mapper(self, peh_value_type):
                raise AssertionError("value_type should be checked first")

        adapter = ScalarResolvingEnrichmentInterface()

        with pytest.raises(
            ValueError,
            match="Scalar CalculationKeywordArguments require value_type.",
        ):
            adapter._resolve_scalar_value("1024", None)

    def test_topological_sort_with_single_node(self):
        g = Graph()
        g._add_node(Node("A", "A"))
        sorted_nodes = g.topological_sort()
        assert sorted_nodes == [Node("A", "A")]

    def test_topological_sort_with_independent_nodes(self):
        g = Graph()
        g.add_edge(Node("A", "A"), Node("B", "B"))
        g.add_edge(Node("C", "C"), Node("D", "D"))
        g.add_edge(Node("E", "E"), Node("F", "F"))
        sorted_nodes = g.topological_sort()
        assert sorted_nodes.index(Node("A", "A")) < sorted_nodes.index(
            Node("B", "B")
        )
        assert sorted_nodes.index(Node("C", "C")) < sorted_nodes.index(
            Node("D", "D")
        )
        assert sorted_nodes.index(Node("E", "E")) < sorted_nodes.index(
            Node("F", "F")
        )

    def test_topological_sort_with_cycle(self):
        g = Graph()
        g.add_edge(Node("A", "A"), Node("B", "B"))
        g.add_edge(Node("B", "B"), Node("C", "C"))
        g.add_edge(Node("C", "C"), Node("A", "A"))  # Creates a cycle
        g.add_edge(Node("D", "D"), Node("E", "E"))

        with pytest.raises(ValueError) as excinfo:
            g.topological_sort()

        assert "Circular dependency detected" in str(excinfo.value)

    def test_add_calculation_scalar_argument(self):
        g = Graph()
        target = Node("A", "result")
        g.add_calculation_target(
            target=target,
            function_name="builtins.abs",
            result_dtype="integer",
        )
        g.add_calculation_scalar_argument(
            target=target,
            source_mapping_name="offset",
            value=2,
        )
        delayed = g.delayed_fns[target]
        assert delayed.arg_values == {"offset": 2}


class MockIndex(ContextIndexProtocol):
    def context_lookup(
        self, observation_id: str, observable_property_id: str
    ) -> tuple[str, str]:
        # Keep node labels aligned with observation ids used in assertions.
        return (observation_id, observable_property_id)


@pytest.mark.core
class TestEnrichmentInterfaceCore:
    def mock_join_spec_mapping(
        self, observations: list
    ) -> dict[frozenset, JoinSpec | None]:
        """
        Graph-focused tests don't exercise real join execution; provide a permissive
        dummy join mapping so dependency graph construction can proceed without
        requiring internal data layout resolution.
        """
        labels = [str(obs.id) for obs in observations if obs.id is not None]
        mapping: dict[frozenset, JoinSpec | None] = {}
        for left_label, right_label in itertools.combinations(labels, 2):
            mapping[frozenset((left_label, right_label))] = JoinSpec(
                left_elements=("__mock_join_key__",),
                left_dataset=left_label,
                right_elements=("__mock_join_key__",),
                right_dataset=right_label,
            )
        return mapping

    def container(self, path: str) -> CacheContainerView:
        source = get_absolute_path(path)
        container = CacheContainerFactory.new()
        host = DirectoryIO()
        roots = host.load(source, format="yaml", maxdepth=3)
        for root in roots:
            for entity in load_entities_from_tree(root):
                container.add(entity)

        return CacheContainerView(container)

    def test_building_dependency_graph(self):
        interface = DataEnrichmentInterface()
        container = self.container(
            "./input/dependency_graph/Enrichment_01_SINGLE_SOURCE"
        )
        observations = list(container.get_all("Observation"))
        g = interface.build_dependency_graph(
            observations,  # type: ignore
            context_index=MockIndex(),
            cache_view=container,
            join_spec_mapping=self.mock_join_spec_mapping(observations),
        )

        # Simple check to see if the dependency graph is built
        assert isinstance(g, Graph)

    def test_topological_sort_single_source(self):
        interface = DataEnrichmentInterface()
        src_path = "./input/dependency_graph/Enrichment_01_SINGLE_SOURCE"
        container = self.container(src_path)
        observations = list(container.get_all("Observation"))
        g = interface.build_dependency_graph(
            observations,  # type: ignore
            context_index=MockIndex(),
            cache_view=container,
            join_spec_mapping=self.mock_join_spec_mapping(observations),
        )
        sorted_nodes = g.topological_sort()
        # Simple check to see if the sorted variables list is correct
        assert isinstance(g, Graph)
        assert all(isinstance(var, Node) for var in sorted_nodes)
        assert len(sorted_nodes) == len(g.nodes)
        assert sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECT_ENRICHED",
                "peh:agemonths",
            )
        ) > sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECTUNIQUE_INGESTED",
                "N1Birthdate",
            )
        )
        assert sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECT_ENRICHED",
                "peh:agemonths",
            )
        ) > sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECTUNIQUE_INGESTED",
                "Todaysdate",
            )
        )

    def test_topological_sort_linked_source(self):
        interface = DataEnrichmentInterface()
        src_path = "./input/dependency_graph/Enrichment_02_LINKED_SOURCE"
        container = self.container(src_path)
        observations = list(container.get_all("Observation"))
        g = interface.build_dependency_graph(
            observations,  # type: ignore
            context_index=MockIndex(),
            cache_view=container,
            join_spec_mapping=self.mock_join_spec_mapping(observations),
        )
        sorted_nodes = g.topological_sort()
        # Simple check to see if the sorted variables list is correct
        assert isinstance(g, Graph)
        assert all(isinstance(var, Node) for var in sorted_nodes)
        assert len(sorted_nodes) == len(g.nodes)
        assert sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECT_ENRICHED",
                "peh:agemonths",
            )
        ) > sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECTUNIQUE_INGESTED",
                "N1Birthdate",
            )
        )
        assert sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECT_ENRICHED",
                "peh:agemonths",
            )
        ) > sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_HOUSEHOLD_INGESTED",
                "Todaysdate",
            )
        )

    def test_topological_sort_multi_steps(self):
        interface = DataEnrichmentInterface()
        src_path = "./input/dependency_graph/Enrichment_03_MULTI_STEP"
        container = self.container(src_path)
        observations = list(container.get_all("Observation"))
        g = interface.build_dependency_graph(
            observations,  # type: ignore
            context_index=MockIndex(),
            cache_view=container,
            join_spec_mapping=self.mock_join_spec_mapping(observations),
        )
        sorted_nodes = g.topological_sort()
        # Simple check to see if the sorted variables list is correct
        assert isinstance(g, Graph)
        assert all(isinstance(var, Node) for var in sorted_nodes)
        assert len(sorted_nodes) == len(g.nodes)
        assert sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECT_ENRICHED",
                "peh:agemonths",
            )
        ) > sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECTUNIQUE_INGESTED",
                "peh:N1Birthdate",
            )
        )
        assert sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECT_ENRICHED",
                "peh:agemonths",
            )
        ) > sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECT_ENRICHED",
                "peh:Todaysdate",
            )
        )
        assert sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECT_ENRICHED",
                "peh:Todaysdate",
            )
        ) > sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECTUNIQUE_INGESTED",
                "peh:current_day",
            )
        )
        assert sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECT_ENRICHED",
                "peh:Todaysdate",
            )
        ) > sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECTUNIQUE_INGESTED",
                "peh:current_month",
            )
        )
        assert sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECT_ENRICHED",
                "peh:Todaysdate",
            )
        ) > sorted_nodes.index(
            Node(
                "peh:ENRICHMENT_TEST_OBSERVATION_SUBJECTUNIQUE_INGESTED",
                "peh:current_year",
            )
        )
