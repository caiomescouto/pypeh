import pytest

from pypeh.core.models.constants import ObservablePropertyValueType
from pypeh.core.models.internal_data_layout import DatasetSchemaElement

pytestmark = [pytest.mark.dataframe, pytest.mark.other]


@pytest.fixture
def adapter():
    from pypeh.adapters.metadata.dataframe_adapter import (
        DataFrameMetadataAdapter,
    )

    return DataFrameMetadataAdapter()


def _element(data_type: ObservablePropertyValueType) -> DatasetSchemaElement:
    return DatasetSchemaElement(
        label="chol",
        observable_property_id="peh:prop_chol",
        data_type=data_type,
    )


class TestMapDataType:
    @pytest.mark.parametrize(
        ("value_type", "expected"),
        [
            (ObservablePropertyValueType.STRING, "STRING"),
            (ObservablePropertyValueType.INTEGER, "INTEGER"),
            (ObservablePropertyValueType.FLOAT, "FLOAT"),
            (ObservablePropertyValueType.BOOLEAN, "BOOLEAN"),
            (ObservablePropertyValueType.DATE, "DATE"),
            (ObservablePropertyValueType.DATETIME, "DATETIME"),
        ],
    )
    def test_declared_value_types(self, adapter, value_type, expected):
        assert adapter.map_data_type(_element(value_type)) == expected
