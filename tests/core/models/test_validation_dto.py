from dataclasses import dataclass

import pytest
from peh_model import peh

from pypeh.core.models.constants import (
    ObservablePropertyValueType,
    ValidationErrorLevel,
)
from pypeh.core.models.validation_dto import (
    ValidationDesign,
    convert_peh_validation_error_level_to_validation_dto_error_level,
)


@dataclass
class DummyMetadata:
    field: str
    value: str


@pytest.mark.core
class TestValidationDesignFromBounds:
    @pytest.mark.parametrize(
        "peh_level, expected_level",
        [
            ("info", ValidationErrorLevel.INFO),
            ("warning", ValidationErrorLevel.WARNING),
            ("error", ValidationErrorLevel.ERROR),
            ("fatal", ValidationErrorLevel.FATAL),
        ],
    )
    def test_convert_peh_validation_error_level_accepts_linkml_values(
        self, peh_level, expected_level
    ):
        validation_design = peh.ValidationDesign(
            validation_error_level=peh_level
        )

        result = (
            convert_peh_validation_error_level_to_validation_dto_error_level(
                validation_design.validation_error_level
            )
        )

        assert result == expected_level

    def test_list_from_bounds_creates_min_and_max_validations(self):
        validations = ValidationDesign.list_from_bounds(
            min_value=1,
            max_value=9,
            type_annotations={"D": {"x": ObservablePropertyValueType.FLOAT}},
            dataset_label="D",
        )

        assert [vd.name for vd in validations] == ["min", "max"]
        assert (
            validations[0].expression.command == "is_greater_than_or_equal_to"
        )
        assert validations[0].expression.arg_values == [1]
        assert validations[1].expression.command == "is_less_than_or_equal_to"
        assert validations[1].expression.arg_values == [9]

    def test_list_from_metadata_respects_skip_fields(self):
        metadata = [
            DummyMetadata(field="min", value="1"),
            DummyMetadata(field="max", value="9"),
            DummyMetadata(field="is_equal_to", value="5"),
        ]

        validations = ValidationDesign.list_from_metadata(
            metadata=metadata,
            type_annotations={"D": {"x": ObservablePropertyValueType.FLOAT}},
            dataset_label="D",
            skip_fields={"min", "max"},
        )

        assert len(validations) == 1
        assert validations[0].name == "is_equal_to"
        assert validations[0].expression.command == "is_equal_to"
        assert validations[0].expression.arg_values == [5.0]
