import pytest
from peh_model import peh

from pypeh.core.models.constants import ObservablePropertyValueType
from pypeh.core.models.extract_dto import (
    FilterExpression,
    FilterConfig,
)


@pytest.mark.core
class TestFilterExpressionFromPeh:
    def test_from_peh_maps_simple_command(self):
        obs_filter = peh.ObservationFilterExpression(
            filter_command="is_in",
            filter_subject_contextual_field_references=[
                peh.ContextualFieldReference(
                    field_label="country", dataset_label="D1"
                )
            ],
            filter_arg_values=["BE", "BR"],
        )

        result = FilterExpression.from_peh(
            obs_filter,
            type_annotations={
                "D1": {"country": ObservablePropertyValueType.STRING}
            },
            dataset_label="D1",
        )

        assert result.command == "is_in"
        assert result.subject == ["country"]
        assert result.arg_values == ["BE", "BR"]

    def test_from_peh_nested_conjunction(self):
        nested_filter1 = peh.ObservationFilterExpression(
            filter_command="is_equal_to",
            filter_subject_contextual_field_references=[
                peh.ContextualFieldReference(
                    field_label="status", dataset_label="D1"
                )
            ],
            filter_arg_values=["active"],
        )
        nested_filter2 = peh.ObservationFilterExpression(
            filter_command="is_greater_than",
            filter_subject_contextual_field_references=[
                peh.ContextualFieldReference(
                    field_label="age", dataset_label="D1"
                )
            ],
            filter_arg_values=["18"],
        )
        obs_filter = peh.ObservationFilterExpression(
            filter_command="conjunction",
            filter_arg_expressions=[nested_filter1, nested_filter2],
        )

        result = FilterExpression.from_peh(
            obs_filter,
            type_annotations={
                "D1": {
                    "status": ObservablePropertyValueType.STRING,
                    "age": ObservablePropertyValueType.INTEGER,
                }
            },
            dataset_label="D1",
        )

        assert result.command == "conjunction"
        assert result.arg_expressions is not None
        assert len(result.arg_expressions) == 2
        assert result.arg_expressions[0].command == "is_equal_to"
        assert result.arg_expressions[1].command == "is_greater_than"

    def test_command_to_str_accepts_enum(self):
        """Test that peh.ObservationFilterCommand enum normalizes to string."""
        obs_filter = peh.ObservationFilterExpression(
            filter_command=peh.ObservationFilterCommand.is_equal_to,
            filter_subject_contextual_field_references=[
                peh.ContextualFieldReference(
                    field_label="name", dataset_label="D1"
                )
            ],
            filter_arg_values=["John"],
        )

        result = FilterExpression.from_peh(
            obs_filter,
            type_annotations={
                "D1": {"name": ObservablePropertyValueType.STRING}
            },
            dataset_label="D1",
        )

        assert result.command == "is_equal_to"
        assert isinstance(result.command, str)

    def test_from_peh_tracks_cross_dataset_dependencies(self):
        obs_filter = peh.ObservationFilterExpression(
            filter_command="is_in",
            filter_subject_contextual_field_references=[
                peh.ContextualFieldReference(
                    field_label="country", dataset_label="D2"
                )
            ],
            filter_arg_values=["BE", "BR"],
        )

        result = FilterExpression.from_peh(
            obs_filter,
            type_annotations={
                "D1": {"x": ObservablePropertyValueType.STRING},
                "D2": {"country": ObservablePropertyValueType.STRING},
            },
            dataset_label="D1",
        )

        assert result.dependent_contextual_field_references is not None
        assert "D2" in result.dependent_contextual_field_references
        assert "country" in result.dependent_contextual_field_references["D2"]

    def test_from_peh_maps_arg_columns(self):
        obs_filter = peh.ObservationFilterExpression(
            filter_command="is_in",
            filter_subject_contextual_field_references=[
                peh.ContextualFieldReference(
                    field_label="status", dataset_label="D1"
                )
            ],
            filter_arg_contextual_field_references=[
                peh.ContextualFieldReference(
                    field_label="allowed_statuses", dataset_label="D1"
                )
            ],
        )

        result = FilterExpression.from_peh(
            obs_filter,
            type_annotations={
                "D1": {
                    "status": ObservablePropertyValueType.STRING,
                    "allowed_statuses": ObservablePropertyValueType.STRING,
                }
            },
            dataset_label="D1",
        )

        assert result.arg_columns == ["allowed_statuses"]
        assert result.subject == ["status"]

    def test_from_peh_nests_conditional_expression(self):
        condition = peh.ObservationFilterExpression(
            filter_command="is_greater_than",
            filter_subject_contextual_field_references=[
                peh.ContextualFieldReference(
                    field_label="age", dataset_label="D1"
                )
            ],
            filter_arg_values=["18"],
        )
        obs_filter = peh.ObservationFilterExpression(
            filter_command="is_in",
            filter_subject_contextual_field_references=[
                peh.ContextualFieldReference(
                    field_label="country", dataset_label="D1"
                )
            ],
            filter_arg_values=["BE", "BR"],
            filter_condition_expression=condition,
        )

        result = FilterExpression.from_peh(
            obs_filter,
            type_annotations={
                "D1": {
                    "country": ObservablePropertyValueType.STRING,
                    "age": ObservablePropertyValueType.INTEGER,
                }
            },
            dataset_label="D1",
        )

        assert result.command == "is_in"
        assert result.conditional_expression is not None
        assert result.conditional_expression.command == "is_greater_than"
        assert result.conditional_expression.subject == ["age"]

    def test_from_peh_casts_numeric_values(self):
        obs_filter = peh.ObservationFilterExpression(
            filter_command="is_greater_than_or_equal_to",
            filter_subject_contextual_field_references=[
                peh.ContextualFieldReference(
                    field_label="age", dataset_label="D1"
                )
            ],
            filter_arg_values=["18"],
        )

        result = FilterExpression.from_peh(
            obs_filter,
            type_annotations={
                "D1": {"age": ObservablePropertyValueType.INTEGER}
            },
            dataset_label="D1",
        )

        assert result.arg_values == [18]
        assert isinstance(result.arg_values[0], int)


@pytest.mark.core
class TestFilterConfig:
    def test_filter_config_fields(self):
        filter_expr = FilterExpression(
            command="is_equal_to",
            subject=["age"],
            arg_values=[18],
        )
        filter_config = FilterConfig(
            name="age_check",
            filter_expression=filter_expr,
            select=["name", "age"],
        )

        assert filter_config.name == "age_check"
        assert filter_config.filter_expression == filter_expr
        assert filter_config.select == ["name", "age"]
        assert filter_config.dependent_contextual_field_references == {}
