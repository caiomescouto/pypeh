from __future__ import annotations

import itertools
import logging
import warnings

from collections import defaultdict
from dataclasses import dataclass, field
from peh_model import peh
from typing import (
    TYPE_CHECKING,
    Generator,
    Generic,
    Protocol,
)

from pypeh.core.cache.containers import CacheContainer, CacheContainerView
from pypeh.core.models.identifiers import (
    IdentifierContext,
    IdentifierProvider,
    generate_ulid,
)
from pypeh.core.models.typing import T_DataType
from pypeh.core.models.constants import ObservablePropertyValueType
from pypeh.core.models.validation_errors import DatasetSchemaError

if TYPE_CHECKING:
    from typing import Any


logger = logging.getLogger(__name__)


def _mint_identifier(
    resource_class: type,
    *,
    identifier_provider: IdentifierProvider | None = None,
    label: str | None = None,
    described_by: str | None = None,
) -> str:
    if identifier_provider is not None:
        return identifier_provider.mint(
            IdentifierContext(
                resource_class=resource_class,
                label=label,
                described_by=described_by,
            )
        )
    return generate_ulid()


class ContextIndexProtocol(Protocol):
    def context_lookup(
        self, observation_id: str, observable_property_id: str
    ) -> tuple[str, str]: ...


@dataclass
class JoinSpec:
    left_elements: tuple[str, ...]
    left_dataset: str
    right_elements: tuple[str, ...]
    right_dataset: str


@dataclass
class DatasetSchemaElement:
    label: str
    observable_property_id: str
    data_type: ObservablePropertyValueType
    identifier: str | None = None
    identifier_provider: IdentifierProvider | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self):
        if self.identifier is None:
            self.identifier = _mint_identifier(
                DatasetSchemaElement,
                identifier_provider=self.identifier_provider,
                label=self.label,
            )


@dataclass
class ElementReference:
    dataset_label: str
    element_label: str


@dataclass
class ForeignKey:
    element_label: str
    reference: ElementReference
    identifier: str | None = None
    identifier_provider: IdentifierProvider | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self):
        if self.identifier is None:
            self.identifier = _mint_identifier(
                ForeignKey,
                identifier_provider=self.identifier_provider,
                label=self.element_label,
            )


@dataclass
class DatasetSchema:
    elements: dict[str, DatasetSchemaElement] = field(default_factory=dict)
    primary_keys: set[str] = field(default_factory=set)
    foreign_keys: dict[str, ForeignKey] = field(default_factory=dict)
    identifier: str | None = None
    identifier_provider: IdentifierProvider | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self):
        if self.identifier is None:
            self.identifier = _mint_identifier(
                DatasetSchema,
                identifier_provider=self.identifier_provider,
            )
        self._type = self.get_type_annotations()
        self._elements_by_observable_property: dict[str, set[str]] = (
            self.build_observable_property_index()
        )

    def get_type_annotations(self) -> dict[str, ObservablePropertyValueType]:
        ret: dict[str, ObservablePropertyValueType] = dict()
        for element in self.elements.values():
            data_type = element.data_type
            if data_type is not None:
                ret[element.label] = data_type
            else:
                ret[element.label] = ObservablePropertyValueType.STRING

        return ret

    def build_observable_property_index(self) -> dict[str, set[str]]:
        # TODO: fix this, a single dataset can contain multiple occurrences of the same observable property
        elements_by_observable_property = defaultdict(set)
        for element_label, element in self.elements.items():
            elements_by_observable_property[
                element.observable_property_id
            ].add(element_label)
        return elements_by_observable_property

    def get_type(self, element_label: str) -> ObservablePropertyValueType:
        return self._type[element_label]

    def get_element_by_label(
        self, element_label: str
    ) -> DatasetSchemaElement | None:
        return self.elements.get(element_label, None)

    def yield_elements_by_observable_property_id(
        self, observable_property_id: str
    ) -> Generator[DatasetSchemaElement, None, None]:
        element_labels = self._elements_by_observable_property[
            observable_property_id
        ]
        for element in element_labels:
            ret = self.get_element_by_label(element_label=element)
            assert ret is not None
            yield ret

    def get_observable_property_ids(self) -> list[str]:
        return [
            element.observable_property_id
            for element in self.elements.values()
        ]

    def get_element_labels(self) -> list[str]:
        return list(self.elements.keys())

    def add_observable_property(
        self,
        observable_property_id: str,
        data_type: ObservablePropertyValueType,
        element_label: str | None = None,
        is_primary_key: bool = False,
    ) -> DatasetSchemaElement:
        if element_label is None:
            element_label = observable_property_id
        if element_label in self.elements:
            raise ValueError(
                f"DatasetSchema already contains an Element with label {element_label}"
            )
        assert isinstance(data_type, ObservablePropertyValueType)
        new_element = DatasetSchemaElement(
            label=element_label,
            observable_property_id=observable_property_id,
            data_type=data_type,
            identifier_provider=self.identifier_provider,
        )
        self.elements[element_label] = new_element
        self._type[element_label] = data_type
        if observable_property_id in self._elements_by_observable_property:
            raise ValueError(
                f"ObservableProperty with id {observable_property_id} already in schema"
            )
        self._elements_by_observable_property[observable_property_id].add(
            element_label
        )
        if is_primary_key:
            self.primary_keys.add(element_label)

        return new_element

    def add_foreign_key_link(
        self,
        element_label: str,
        foreign_key_dataset_label: str,
        foreign_key_element_label: str,
    ):
        foreign_key_object = ForeignKey(
            element_label=element_label,
            reference=ElementReference(
                dataset_label=foreign_key_dataset_label,
                element_label=foreign_key_element_label,
            ),
            identifier_provider=self.identifier_provider,
        )
        self.foreign_keys[element_label] = foreign_key_object

    # TODO: move method, this is probably not the right location
    def apply_context_to_expression(
        self,
        expression: peh.ValidationExpression,
        context: dict[str, dict[str, str]],
        this_dataset: str,  # temporary fix
    ):
        expression_stack = [expression]
        while expression_stack:
            expression = expression_stack.pop()
            assert isinstance(expression, peh.ValidationExpression)
            conditional_expression = expression.validation_condition_expression
            if conditional_expression is not None:
                assert isinstance(
                    conditional_expression, peh.ValidationExpression
                )
                expression_stack.append(conditional_expression)
            arg_expressions = expression.validation_arg_expressions
            if arg_expressions is not None:
                for arg_expression in arg_expressions:
                    assert isinstance(arg_expression, peh.ValidationExpression)
                    expression_stack.append(arg_expression)

            # apply context
            arg_field_references = (
                expression.validation_arg_contextual_field_references
            )
            if arg_field_references is not None:
                for arg_ref in arg_field_references:
                    assert isinstance(arg_ref, peh.ContextualFieldReference)
                    field_label = arg_ref.field_label
                    if field_label in context:
                        retrieved_context = context[field_label]
                        if len(retrieved_context) == 1:
                            new_dataset_label = next(iter(retrieved_context))
                        else:
                            new_dataset_label = this_dataset
                        new_field_label = retrieved_context[new_dataset_label]
                        arg_ref.dataset_label = new_dataset_label
                        arg_ref.field_label = new_field_label

            subject_field_references = (
                expression.validation_subject_contextual_field_references
            )
            if subject_field_references is not None:
                for subject_field_ref in subject_field_references:
                    assert isinstance(
                        subject_field_ref, peh.ContextualFieldReference
                    )
                    field_label = subject_field_ref.field_label
                    if field_label in context:
                        retrieved_context = context[field_label]
                        if len(retrieved_context) == 1:
                            new_dataset_label = next(iter(retrieved_context))
                        else:
                            new_dataset_label = this_dataset
                        new_field_label = retrieved_context[new_dataset_label]
                        subject_field_ref.dataset_label = new_dataset_label
                        subject_field_ref.field_label = new_field_label

    def apply_context(
        self,
        context: dict[str, dict[str, str]],
        cache: CacheContainer,
        this_dataset: str,  # temporary fix
    ):
        """
        ObservableProperties require context for:
        - CalculationDesigns
        - ValidationDesigns
        """
        for schema_element in self.elements.values():
            observable_property_id = schema_element.observable_property_id
            observable_property = cache.require(
                observable_property_id, "ObservableProperty"
            )
            assert isinstance(observable_property, peh.ObservableProperty)
            validation_designs = observable_property.validation_designs
            if validation_designs is not None:
                for validation_design in validation_designs:
                    assert isinstance(validation_design, peh.ValidationDesign)
                    expression = validation_design.validation_expression
                    assert isinstance(expression, peh.ValidationExpression)
                    self.apply_context_to_expression(
                        expression, context, this_dataset=this_dataset
                    )

            # CALCULATION DESIGN EXCLUDED FOR NOW
            # if observable_property.calculation_design is not None:
            #    pass

    def __len__(self):
        return len(self.elements)

    #### EXTRACT INFO FROM SCHEMA ####

    @staticmethod
    def _pack_join_elements(elements: list[str]) -> tuple[str, ...]:
        return tuple(elements)

    @staticmethod
    def _collect_fk_pairs(
        schema: DatasetSchema, referenced_dataset_label: str
    ) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for fk in schema.foreign_keys.values():
            if fk.reference.dataset_label == referenced_dataset_label:
                pairs.append((fk.element_label, fk.reference.element_label))
        pairs.sort(key=lambda pair: (pair[0], pair[1]))
        return pairs

    @staticmethod
    def _collect_fk_pairs_by_dataset(
        schema: DatasetSchema,
    ) -> dict[str, list[tuple[str, str]]]:
        by_dataset: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for fk in schema.foreign_keys.values():
            by_dataset[fk.reference.dataset_label].append(
                (fk.element_label, fk.reference.element_label)
            )
        for pairs in by_dataset.values():
            pairs.sort(key=lambda pair: (pair[0], pair[1]))
        return by_dataset

    def detect_join(
        self,
        dataset_label: str,
        other_schema: DatasetSchema,
        other_dataset_label: str,
    ) -> JoinSpec | None:
        # Case 1: A -> B directly (supports multi-column keys)
        left_pairs = self._collect_fk_pairs(self, other_dataset_label)
        if len(left_pairs) > 0:
            left_elements = [pair[0] for pair in left_pairs]
            right_elements = [pair[1] for pair in left_pairs]
            return JoinSpec(
                left_elements=self._pack_join_elements(left_elements),
                left_dataset=dataset_label,
                right_elements=self._pack_join_elements(right_elements),
                right_dataset=other_dataset_label,
            )

        # Case 2: B -> A directly (supports multi-column keys)
        right_pairs = self._collect_fk_pairs(other_schema, dataset_label)
        if len(right_pairs) > 0:
            left_elements = [pair[1] for pair in right_pairs]
            right_elements = [pair[0] for pair in right_pairs]
            return JoinSpec(
                left_elements=self._pack_join_elements(left_elements),
                left_dataset=dataset_label,
                right_elements=self._pack_join_elements(right_elements),
                right_dataset=other_dataset_label,
            )

        # Case 3: shared third dataset (hub), resolved as direct inferred join
        refs_a = self._collect_fk_pairs_by_dataset(self)
        refs_b = self._collect_fk_pairs_by_dataset(other_schema)
        shared = sorted(set(refs_a.keys()).intersection(set(refs_b.keys())))
        if len(shared) > 0:
            for shared_label in shared:
                a_by_ref = {
                    ref_element: local_element
                    for local_element, ref_element in refs_a[shared_label]
                }
                b_by_ref = {
                    ref_element: local_element
                    for local_element, ref_element in refs_b[shared_label]
                }
                shared_ref_elements = sorted(
                    set(a_by_ref.keys()).intersection(set(b_by_ref.keys()))
                )
                if len(shared_ref_elements) == 0:
                    continue
                left_elements = [
                    a_by_ref[ref_element]
                    for ref_element in shared_ref_elements
                ]
                right_elements = [
                    b_by_ref[ref_element]
                    for ref_element in shared_ref_elements
                ]
                return JoinSpec(
                    left_elements=self._pack_join_elements(left_elements),
                    left_dataset=dataset_label,
                    right_elements=self._pack_join_elements(right_elements),
                    right_dataset=other_dataset_label,
                )

        return None


@dataclass(kw_only=True)
class Resource:
    """
    Parent class for Dataset and DatasetSeries
    """

    label: str
    identifier: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    identifier_provider: IdentifierProvider | None = field(default=None)

    def __post_init__(self):
        if self.identifier is None:
            self.identifier = _mint_identifier(
                self.__class__,
                identifier_provider=self.identifier_provider,
                label=self.label,
                described_by=self.described_by,
            )

    def add_metadata(self, metadata_key: str, metadata_value: Any) -> bool:
        if metadata_key in self.metadata:
            raise KeyError(
                f"{metadata_key} key already used in metadata mapping"
            )
        self.metadata[metadata_key] = metadata_value

        return True

    @property
    def described_by(self) -> str | peh.NamedThingId | None:
        return self.metadata.get("described_by", None)


@dataclass(kw_only=True)
class Dataset(Resource, Generic[T_DataType]):
    schema: DatasetSchema | None = field(default=None)
    data: T_DataType | None = field(default=None)
    part_of: DatasetSeries | None = field(default=None)
    observation_ids: set[str] = field(default_factory=set)

    def __post_init__(self):
        super().__post_init__()
        if self.schema is None:
            self.schema = DatasetSchema(
                identifier_provider=self.identifier_provider,
            )
        elif self.schema.identifier is None:
            self.schema.identifier_provider = self.identifier_provider
            self.schema.identifier = _mint_identifier(
                DatasetSchema,
                identifier_provider=self.identifier_provider,
                label=f"{self.label}_schema",
            )

    def get_type_annotations(self) -> dict[str, ObservablePropertyValueType]:
        assert self.schema is not None
        return self.schema.get_type_annotations()

    #### EXTRACT INFO FROM DATASET ####

    def get_element_labels(self) -> list[str]:
        assert self.schema is not None
        return self.schema.get_element_labels()

    def get_schema_element_by_label(
        self, element_label: str
    ) -> DatasetSchemaElement | None:
        assert self.schema is not None
        return self.schema.get_element_by_label(element_label)

    def get_schema_element_by_observable_property_id(
        self, observable_property_id: str
    ) -> Generator[DatasetSchemaElement, None, None]:
        assert self.schema is not None
        yield from self.schema.yield_elements_by_observable_property_id(
            observable_property_id
        )

    def get_observable_property_ids(self) -> list[str]:
        assert self.schema is not None
        return self.schema.get_observable_property_ids()

    def get_primary_keys(self) -> set[str] | None:
        assert self.schema is not None
        return self.schema.primary_keys

    def resolve_join(self, other: Dataset) -> JoinSpec | None:
        schema = self.schema
        assert schema is not None
        assert other.schema is not None
        return schema.detect_join(
            dataset_label=self.label,
            other_schema=other.schema,
            other_dataset_label=other.label,
        )

    def add_observable_property(
        self,
        observable_property_id: str,
        data_type: ObservablePropertyValueType,
        element_label: str | None = None,
        is_primary_key: bool = False,
    ) -> DatasetSchemaElement:
        assert self.schema is not None
        return self.schema.add_observable_property(
            observable_property_id,
            data_type,
            element_label,
            is_primary_key,
        )

    def add_observation_to_index(self, observation_id: str):
        if self.part_of:
            self.part_of._register_observation(observation_id, self.label)
        self.observation_ids.add(observation_id)

    def remove_observation_from_index(self, observation_id: str):
        self.observation_ids.remove(observation_id)
        if self.part_of:
            self.part_of._unregister_observation(observation_id)

    def add_data(
        self,
        data: T_DataType,
        data_labels: list[str],
        overwrite: bool = True,
        allow_incomplete: bool = False,
    ):
        """
        data_labels are only used for verification
        """
        if not overwrite:
            if self.data is not None:
                raise NotImplementedError(
                    "Dataset.add_data does not support overwrite=False when "
                    f"data already exists. dataset_label={self.label!r}."
                )

        assert self.schema is not None
        if len(self.schema) > 0:
            if allow_incomplete:
                assert self.contained_in_schema(data_labels)
            else:
                assert self.matches_schema(data_labels)

            self.data = data

    #### VERIFY DATASET ####

    def contained_in_schema(self, element_labels: list[str]) -> bool:
        """
        Check if the columns from a set of data can be found amongst the labels defined in the dataset schema
        """
        raw_data_labels = set(element_labels)
        schema_labels = set(self.get_element_labels())
        label_diff = raw_data_labels.difference(schema_labels)
        if len(label_diff) != 0:
            raise DatasetSchemaError(
                f"Data Schema Error: label(s) {', '.join(label_diff)} are undefined",
                dataset_label=self.label,
                data_labels=element_labels,
                schema_labels=self.get_element_labels(),
                undefined_labels=list(label_diff),
            )
        return len(label_diff) == 0

    def matches_schema(self, element_labels: list[str]) -> bool:
        """
        Check if the columns from a set of data match the labels defined in the dataset schema
        """
        raw_data_labels = set(element_labels)
        schema_labels = set(self.get_element_labels())
        error_str = []
        undefined_diff = raw_data_labels.difference(schema_labels)
        if len(undefined_diff) > 0:
            error_str.append(
                f"label(s) {', '.join(undefined_diff)} are undefined"
            )
        missing_diff = schema_labels.difference(raw_data_labels)
        if len(missing_diff) > 0:
            error_str.append(f"label(s) {', '.join(missing_diff)} are missing")
        if raw_data_labels != schema_labels:
            raise DatasetSchemaError(
                f"Data Schema Error: {', '.join(error_str)}",
                dataset_label=self.label,
                data_labels=element_labels,
                schema_labels=self.get_element_labels(),
                missing_labels=list(missing_diff),
                undefined_labels=list(undefined_diff),
            )
        return True


@dataclass(kw_only=True)
class DatasetSeries(Resource, Generic[T_DataType]):
    parts: dict[str, Dataset[T_DataType]] = field(default_factory=dict)
    _obs_index: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    _context_index: dict[tuple[str, str], tuple[str, str]] = field(
        default_factory=dict
    )  # {(observation_id, observable_property_id): (dataset_label, dataset_element_label}}

    def _register_observation(
        self,
        observation_id: str,
        dataset_label: str,
    ):
        if observation_id in self._obs_index:
            found_set = self._obs_index[observation_id]
            if dataset_label in found_set:
                raise ValueError(
                    f"Observation with id {observation_id} is already registered for dataset with label {dataset_label}."
                )
        assert (
            dataset_label in self.parts
        ), f"Failed to register observation with id {observation_id}. Could not find dataset with label {dataset_label}"
        self._obs_index[observation_id].add(dataset_label)

    def _unregister_observation(
        self,
        observation_id: str,
        observable_property_ids: list[str] | None = None,
    ):
        self._obs_index.pop(observation_id, None)
        if observable_property_ids is not None:
            for obs_prop_id in observable_property_ids:
                self._context_index.pop((observation_id, obs_prop_id))

    def _register_observable_property(
        self,
        observable_property_id: str,
        observation_id: str,
        dataset_label: str,
        element_label: str,
    ):
        assert isinstance(
            element_label, str
        ), "Context index element labels must be strings"
        self._context_index[(observation_id, observable_property_id)] = (
            dataset_label,
            element_label,
        )

    def _unregister_observable_property(
        self,
        observable_property_id: str,
        observation_id: str,
    ):
        self._context_index.pop((observation_id, observable_property_id))

    def build_observation_index(self):
        for dataset_label in self.parts:
            dataset = self[dataset_label]
            assert dataset is not None
            for obs in dataset.observation_ids:
                self._register_observation(obs, dataset_label)

    def build_context_index(self):
        """
        Only works if each dataset only contains one observation
        """
        for dataset_label in self.parts:
            dataset = self[dataset_label]
            assert dataset is not None
            assert (
                len(dataset.observation_ids) == 1
            ), "Cannot build context index if any dataset contains more than one observation"
            observation_id = next(iter(dataset.observation_ids))
            assert dataset.schema is not None
            for element_label, element in dataset.schema.elements.items():
                self._register_observable_property(
                    element.observable_property_id,
                    observation_id,
                    dataset_label,
                    element_label,
                )

    def build_indices(self):
        """
        Building these indices only works if each dataset only contains one observation
        """
        self.build_observation_index()
        self.build_context_index()

    #### CONSTRUCT DATASETSERIES ####
    def apply_context(self, cache: CacheContainer):
        # THIS METHOD MODIFIES THE CACHE !!!
        context = self._get_validation_index()
        for dataset_label in self:
            dataset = self.get(dataset_label)
            assert dataset is not None
            assert dataset.schema is not None
            dataset.schema.apply_context(
                context, cache, this_dataset=self.label
            )

    @classmethod
    def from_peh_datalayout(
        cls,
        data_layout: peh.DataLayout,
        cache_view: CacheContainerView,
        apply_context: bool = False,
        identifier_provider: IdentifierProvider | None = None,
    ) -> DatasetSeries:
        if apply_context:
            warnings.warn(
                "apply_context flag for `DatasetSeries.from_peh_datalayout` is deprecated and will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        dataset_series_label = data_layout.ui_label
        assert dataset_series_label is not None
        ret = DatasetSeries(
            label=dataset_series_label,
            identifier_provider=identifier_provider,
            metadata={"described_by": data_layout.id},
        )
        sections = getattr(data_layout, "sections")
        if sections is None:
            raise ValueError("No sections found in DataLayout")
        for section in sections:
            dataset_label = getattr(section, "ui_label")
            dataset = ret.add_empty_dataset(
                dataset_label=dataset_label,
                identifier_provider=identifier_provider,
                metadata={"described_by": section.id},
            )
            assert isinstance(section, peh.DataLayoutSection)
            section_elements = section.elements
            assert section_elements is not None
            for element in section_elements:
                assert isinstance(element, peh.DataLayoutElement)
                element_label = element.label
                assert element_label is not None
                observable_property_id = element.observable_property
                assert observable_property_id is not None
                identifying = getattr(
                    element, "is_observable_entity_key", False
                )
                observation_id = _mint_identifier(
                    peh.Observation,
                    identifier_provider=identifier_provider,
                )
                observable_property = cache_view.require(
                    observable_property_id, "ObservableProperty"
                )
                assert isinstance(observable_property, peh.ObservableProperty)
                data_type = ObservablePropertyValueType(
                    getattr(observable_property, "value_type", "string")
                )
                ret.add_observable_property(
                    observation_id=observation_id,
                    observable_property_id=observable_property_id,
                    data_type=data_type,
                    dataset_label=dataset_label,
                    element_label=element_label,
                    is_primary_key=identifying,
                )
                # ADD FOREIGN KEYS
                foreign_key = element.foreign_key_link
                if foreign_key is not None:
                    assert isinstance(foreign_key, peh.DataLayoutElementLink)
                    section_id = foreign_key.section
                    assert isinstance(section_id, str)
                    foreign_key_element_label = foreign_key.label
                    assert foreign_key_element_label is not None
                    section = cache_view.require(
                        section_id, "DataLayoutSection"
                    )
                    foreign_key_dataset_label = section.ui_label
                    assert (
                        foreign_key_dataset_label is not None
                    ), f"Cannot create foreign_key_link, ui_label is None for section {section.id}"
                    assert dataset.schema is not None
                    dataset.schema.add_foreign_key_link(
                        element_label=element_label,
                        foreign_key_dataset_label=foreign_key_dataset_label,
                        foreign_key_element_label=foreign_key_element_label,
                    )

        if apply_context:
            ret.apply_context(cache=cache_view._container)

        return ret

    @classmethod
    def from_peh_data_import_config(
        cls,
        data_import_config: peh.DataImportConfig,
        cache_view: CacheContainerView,
        identifier_provider: IdentifierProvider | None = None,
    ) -> DatasetSeries:
        data_layout_id = data_import_config.layout
        assert data_layout_id is not None
        data_layout = cache_view.require(data_layout_id, "DataLayout")
        assert isinstance(data_layout, peh.DataLayout)
        layout_label = data_layout.ui_label
        if layout_label is None:
            layout_label = generate_ulid()
        ret = DatasetSeries(
            label=layout_label,
            identifier_provider=identifier_provider,
            metadata={"described_by": data_layout_id},
        )

        # add Observation links
        section_mapping = data_import_config.section_mapping
        assert isinstance(section_mapping, peh.DataImportSectionMapping)
        section_mapping_links = section_mapping.section_mapping_links
        assert isinstance(section_mapping_links, list)
        for link in section_mapping_links:
            assert isinstance(link, peh.DataImportSectionMappingLink)
            section_id = link.section
            assert isinstance(section_id, str)
            layout_section = cache_view.require(
                section_id, "DataLayoutSection"
            )
            assert isinstance(layout_section, peh.DataLayoutSection)
            dataset_label = layout_section.ui_label
            assert dataset_label is not None
            dataset = ret.add_empty_dataset(
                dataset_label=dataset_label,
                identifier_provider=identifier_provider,
                metadata={"described_by": section_id},
            )
            observation_ids = link.observation_id_list
            assert observation_ids is not None
            assert isinstance(observation_ids, list)
            for observation_id in observation_ids:
                # observation_observable_properties
                observation = cache_view.require(observation_id, "Observation")
                assert isinstance(observation, peh.Observation)
                observation_design_id = observation.observation_design
                assert isinstance(
                    observation_design_id, peh.ObservationDesignId
                )
                observation_design = cache_view.require(
                    observation_design_id, "ObservationDesign"
                )
                assert isinstance(observation_design, peh.ObservationDesign)
                obs_prop_spec_dict = {}
                obs_prop_specs = (
                    observation_design.observable_property_specifications
                )
                assert obs_prop_specs is not None
                for obs_prop_spec in obs_prop_specs:
                    assert isinstance(
                        obs_prop_spec, peh.ObservablePropertySpecification
                    )
                    obs_prop_id = str(obs_prop_spec.observable_property)
                    obs_prop_spec_dict[obs_prop_id] = obs_prop_spec
                # data_layout_section_observable_properties
                elements = layout_section.elements
                assert elements is not None
                # labeled_observable_property_specifications = {element_label: ObservablePropertySpecification}
                labeled_observable_property_specifications = {}
                for element in elements:
                    assert isinstance(element, peh.DataLayoutElement)
                    if (
                        element.observable_property
                        in obs_prop_spec_dict.keys()
                    ):
                        element_label = element.label
                        assert element_label is not None
                        obs_prop_id = element.observable_property
                        assert isinstance(obs_prop_id, str)
                        obs_prop = cache_view.require(
                            obs_prop_id, "ObservableProperty"
                        )
                        assert isinstance(obs_prop, peh.ObservableProperty)
                        obs_prop_spec = obs_prop_spec_dict[obs_prop_id]
                        assert isinstance(
                            obs_prop_spec, peh.ObservablePropertySpecification
                        )
                        labeled_observable_property_specifications[
                            element_label
                        ] = obs_prop_spec
                        # process foreign keys
                        foreign_key = element.foreign_key_link
                        if foreign_key is not None:
                            assert isinstance(
                                foreign_key, peh.DataLayoutElementLink
                            )
                            section_id = foreign_key.section
                            assert isinstance(section_id, str)
                            foreign_key_element_label = foreign_key.label
                            assert foreign_key_element_label is not None
                            section = cache_view.require(
                                section_id, "DataLayoutSection"
                            )
                            foreign_key_dataset_label = section.ui_label
                            assert (
                                foreign_key_dataset_label is not None
                            ), f"Cannot create foreign_key_link, ui_label is None for section {section.id}"
                            assert dataset.schema is not None
                            dataset.schema.add_foreign_key_link(
                                element_label=element_label,
                                foreign_key_dataset_label=foreign_key_dataset_label,
                                foreign_key_element_label=foreign_key_element_label,
                            )

                observation = cache_view.require(observation_id)
                assert isinstance(observation, peh.Observation)
                ret.add_observation(
                    dataset_label=dataset_label,
                    observation=observation,
                    labeled_observable_property_specifications=labeled_observable_property_specifications,
                    cache_view=cache_view,
                )

        return ret

    def register_dataset(self, dataset: Dataset):
        dataset.part_of = self
        self.parts[dataset.label] = dataset

    def add_data(
        self,
        dataset_label: str,
        data: T_DataType,
        data_labels: list[str],
        overwrite: bool = True,
        allow_incomplete: bool = False,
    ):
        dataset = self.parts.get(dataset_label, None)
        assert dataset is not None
        dataset.add_data(
            data=data,
            data_labels=data_labels,
            overwrite=overwrite,
            allow_incomplete=allow_incomplete,
        )

    def add_observable_property(
        self,
        observation_id: str,
        observable_property_id: str,
        data_type: ObservablePropertyValueType,
        dataset_label: str,
        element_label: str | None = None,
        is_primary_key: bool = False,
    ):
        assert dataset_label in self.parts
        dataset = self[dataset_label]
        assert dataset is not None
        schema_element = dataset.add_observable_property(
            observable_property_id,
            data_type,
            element_label,
            is_primary_key,
        )
        self._register_observable_property(
            observable_property_id=observable_property_id,
            observation_id=observation_id,
            dataset_label=dataset_label,
            element_label=schema_element.label,
        )
        return schema_element

    def add_empty_dataset(
        self,
        dataset_label: str,
        identifier_provider: IdentifierProvider | None = None,
        metadata: dict | None = None,
    ) -> Dataset:
        dataset = Dataset(
            label=dataset_label,
            identifier_provider=identifier_provider
            or self.identifier_provider,
            metadata=metadata or {},
        )
        self.register_dataset(dataset)

        return dataset

    def add_observation(
        self,
        dataset_label: str,
        observation: peh.Observation,
        labeled_observable_property_specifications: dict[
            str, peh.ObservablePropertySpecification
        ],
        cache_view: CacheContainerView,
        data: T_DataType | None = None,
    ) -> Dataset:
        """
        labeled_observable_property_specifications: dict[element_label: ObservablePropertySpecification]
        """
        if dataset_label not in self.parts:
            dataset = self.add_empty_dataset(dataset_label)
        else:
            dataset = self[dataset_label]
        assert dataset is not None
        dataset.add_observation_to_index(observation.id)
        for (
            element_label,
            observable_property_specification,
        ) in labeled_observable_property_specifications.items():
            identifying = (
                str(observable_property_specification.specification_category)
                == peh.ObservablePropertySpecificationCategory.identifying.text
            )
            observable_property_id = (
                observable_property_specification.observable_property
            )
            assert isinstance(observable_property_id, str)
            observable_property = cache_view.require(
                observable_property_id, "ObservableProperty"
            )
            assert isinstance(observable_property, peh.ObservableProperty)
            if identifying:
                assert dataset.schema is not None
                if (
                    observable_property.id
                    in dataset.schema._elements_by_observable_property
                ):
                    if element_label is None:
                        element_labels = (
                            dataset.schema._elements_by_observable_property[
                                observable_property.id
                            ]
                        )
                        assert (
                            len(element_labels) == 1
                        ), "Expected identifying ObservableProperty to map to one schema element"
                        element_label = next(iter(element_labels))
                    self._register_observable_property(
                        observable_property_id=observable_property.id,
                        observation_id=observation.id,
                        dataset_label=dataset_label,
                        element_label=element_label,
                    )
                    continue
            _ = self.add_observable_property(
                observation_id=observation.id,
                observable_property_id=observable_property.id,
                data_type=ObservablePropertyValueType(
                    getattr(observable_property, "value_type", "string")
                ),
                dataset_label=dataset_label,
                element_label=element_label,
                is_primary_key=identifying,
            )

        if data is not None:
            dataset.add_data(
                data=data,
                data_labels=list(
                    labeled_observable_property_specifications.keys()
                ),
            )

        return dataset

    #### EXTRACT INFO FROM DATASETSERIES ####

    def get_type_annotations(
        self,
    ) -> dict[str, dict[str, ObservablePropertyValueType]]:
        ret: dict[str, dict[str, ObservablePropertyValueType]] = dict()
        for dataset in self.parts.values():
            label = dataset.label
            ret[label] = dataset.get_type_annotations()

        return ret

    def resolve_join(
        self, left_dataset_label: str, right_dataset_label: str
    ) -> JoinSpec | None:
        left = self.get(left_dataset_label)
        assert left is not None
        right = self.get(right_dataset_label)
        assert right is not None
        return left.resolve_join(right)

    def resolve_all_joins(self) -> dict[frozenset, JoinSpec | None]:
        ret = {}
        for combo in itertools.combinations(self.parts, 2):
            key = frozenset(combo)
            ret[key] = self.resolve_join(*combo)
        return ret

    def _get_validation_index(self) -> dict[str, dict[str, str]]:
        """
        # TEMPORARY SOLUTION !!!!
        ObservablePropertyId -> dataset_label, field_label
        We currently assume that each observable_property only occurs once in the DatasetSeries
        """
        field_ref_dict = defaultdict(dict)
        for dataset_label in self:
            dataset = self.get(dataset_label)
            assert dataset is not None
            assert dataset.schema is not None
            for (
                observable_property_id,
                element_labels,
            ) in dataset.schema._elements_by_observable_property.items():
                assert (
                    len(element_labels) == 1
                ), "TEMPORARY RESTRICTION observable_properties should be unique within a dataset"
                element_label = next(iter(element_labels))
                field_ref_dict[observable_property_id][dataset_label] = (
                    element_label
                )

        return field_ref_dict

    def get_contextual_field_reference_index(
        self,
    ) -> dict[str, tuple[str, str] | None]:
        field_ref_dict = {}
        for dataset_label in self:
            dataset = self.get(dataset_label)
            assert dataset is not None
            assert dataset.schema is not None
            for (
                observable_property_id,
                element_label,
            ) in dataset.schema._elements_by_observable_property.items():
                if observable_property_id in field_ref_dict:
                    field_ref_dict[observable_property_id] = None
                else:
                    field_ref_dict[observable_property_id] = (
                        dataset_label,
                        element_label,
                    )

        return field_ref_dict

    def get_datasets_by_observation(
        self, observation_id: str
    ) -> Generator[Dataset, None, None]:
        dataset_labels = self._obs_index.get(observation_id, None)
        if dataset_labels is None:
            return
        for dataset_label in dataset_labels:
            ret = self[dataset_label]
            assert ret is not None
            yield ret

    def context_lookup(
        self, observation_id: str, observable_property_id: str
    ) -> tuple[str, str]:
        assert (
            self._obs_index is not None
        ), "Cannot perform lookup without observation index"
        assert (
            self._context_index is not None
        ), "Cannot perform lookup without context index"
        ret = self._context_index.get(
            (observation_id, observable_property_id), None
        )
        if ret is None:
            raise ValueError(
                f"Could not find DatasetElement linked to observation with id {observation_id} and observable property with id {observable_property_id}"
            )

        return ret

    #### CORE FUNCTIONALITY ####

    @property
    def data_import_config(self) -> str | None:
        return self.metadata.get("data_import_config", None)

    @property
    def observations(self) -> dict[str, set[str]]:
        ret = {}
        for dataset_label in self:
            dataset = self[dataset_label]
            assert dataset is not None
            ret[dataset_label] = dataset.observation_ids

        return ret

    def __len__(self):
        return len(self.parts)

    def get(self, key, default=None) -> Dataset | None:
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key: str) -> Dataset | None:
        return self.parts.get(key)

    def __setitem__(self, key: str, value: Dataset) -> None:
        self.parts[key] = value

    def update(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise TypeError(
                    "update expected at most 1 arguments, "
                    "got %d" % len(args)
                )
            assert len(args) == 1
            other = dict(args[0])
            for key in other:
                self.parts[key] = other[key]
        for key in kwargs:
            self.parts[key] = kwargs[key]

    def __iter__(self):
        return iter(self.parts)
