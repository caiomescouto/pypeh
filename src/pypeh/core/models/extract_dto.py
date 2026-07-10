from __future__ import annotations

import logging

from collections import defaultdict
from pydantic import BaseModel, field_validator
from typing import Generic, Any, Sequence, TYPE_CHECKING

from pypeh.core.models.typing import T_DataType
from pypeh.core.models.constants import ObservablePropertyValueType
from pypeh.core.models.validation_dto import (
    cast_to_peh_value_type,
    merge_dependencies,
)
from peh_model import pydanticmodel_v2 as pehs
from peh_model import peh


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FilterExpression(BaseModel):
    conditional_expression: FilterExpression | None = None
    arg_expressions: list[FilterExpression] | None = None
    command: str
    arg_values: list[Any] | None = None
    arg_columns: list[str] | None = None
    subject: list[str] | None = None
    dependent_contextual_field_references: dict[str, set[str]] | None = None

    @field_validator("command", mode="before")
    @classmethod
    def command_to_str(cls, v):
        if v is None:
            return "conjunction"
        elif isinstance(v, peh.PermissibleValue):
            return v.text
        elif isinstance(v, str):
            return v
        elif isinstance(v, peh.ObservationFilterCommand):
            return str(v)
        else:
            logger.error(
                f"No conversion defined for {v} of type {v.__class__}"
            )
            raise NotImplementedError(
                "FilterExpression.command_to_str encountered an "
                "unsupported command value. "
                f"value={v!r}, value_type={v.__class__.__name__}."
            )

    @classmethod
    def from_peh(
        cls,
        expression: peh.ObservationFilterExpression
        | pehs.ObservationFilterExpression,
        type_annotations: dict[str, dict[str, ObservablePropertyValueType]]
        | None = None,
        dataset_label: str | None = None,
    ) -> "FilterExpression":
        dependent_contextual_field_references = defaultdict(set)
        conditional_expression = getattr(
            expression, "filter_condition_expression", None
        )
        conditional_expression_instance = None
        if conditional_expression is not None:
            conditional_expression_instance = FilterExpression.from_peh(
                conditional_expression,
                type_annotations,
                dataset_label=dataset_label,
            )
            dependent_contextual_field_references = merge_dependencies(
                dependent_contextual_field_references,
                conditional_expression_instance.dependent_contextual_field_references,
            )

        arg_expressions = getattr(expression, "filter_arg_expressions", None)
        arg_expression_instances = None
        if arg_expressions is not None:
            arg_expression_instances = []
            for nested_expr in arg_expressions:
                new_arg_expression = FilterExpression.from_peh(
                    nested_expr, type_annotations, dataset_label=dataset_label
                )
                arg_expression_instances.append(new_arg_expression)
                dependent_contextual_field_references = merge_dependencies(
                    dependent_contextual_field_references,
                    new_arg_expression.dependent_contextual_field_references,
                )
        filter_command = getattr(expression, "filter_command", "conjunction")

        subject_contextual_field_references = getattr(
            expression, "filter_subject_contextual_field_references", None
        )
        subject_columns = None
        data_type = None
        if subject_contextual_field_references is not None:
            subject_columns = []
            data_types = set()
            assert type_annotations is not None
            for field_reference in subject_contextual_field_references:
                ref_dataset_label = getattr(
                    field_reference, "dataset_label", None
                )
                assert ref_dataset_label is not None
                if ref_dataset_label != dataset_label:
                    dependent_contextual_field_references[
                        ref_dataset_label
                    ].add(field_reference.field_label)
                subject_columns.append(field_reference.field_label)
                data_type = type_annotations.get(ref_dataset_label, {}).get(
                    field_reference.field_label, None
                )
                assert (
                    data_type is not None
                ), f"Did not find type_annotation for dataset with label {ref_dataset_label} and field_label {field_reference.field_label}"
                data_types.add(data_type)
            assert (
                len(data_types) <= 1
            ), f'Found the following datatypes for the subject_contextual_field_references: {", ".join(dt for dt in data_types)}'
            data_type = next(iter(data_types), None)
        if data_type is None:
            data_type = ObservablePropertyValueType.STRING

        arg_values = getattr(expression, "filter_arg_values", None)
        if arg_values is not None:
            assert isinstance(arg_values, Sequence)
            try:
                arg_values = [
                    cast_to_peh_value_type(arg_value, data_type)
                    for arg_value in arg_values
                ]
            except Exception as e:
                logger.error(
                    f"Could not cast values in {arg_values} to {data_type}: {e}"
                )
                raise

        arg_contextual_field_references = getattr(
            expression, "filter_arg_contextual_field_references", None
        )
        arg_columns = None
        if arg_contextual_field_references is not None:
            arg_columns = []
            for field_reference in arg_contextual_field_references:
                ref_dataset_label = getattr(
                    field_reference, "dataset_label", None
                )
                assert ref_dataset_label is not None
                if ref_dataset_label != dataset_label:
                    dependent_contextual_field_references[
                        ref_dataset_label
                    ].add(field_reference.field_label)
                arg_columns.append(field_reference.field_label)

        return cls(
            conditional_expression=conditional_expression_instance,
            arg_expressions=arg_expression_instances,
            command=filter_command,
            arg_values=arg_values,
            arg_columns=arg_columns,
            subject=subject_columns,
            dependent_contextual_field_references=dependent_contextual_field_references,
        )


class FilterConfig(BaseModel, Generic[T_DataType]):
    name: str
    filter_expression: FilterExpression
    select: list[str] | None = None
    dependent_contextual_field_references: dict[str, set[str]] = defaultdict(
        set
    )
