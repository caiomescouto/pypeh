from __future__ import annotations

import logging

from polars import DataFrame
from typing import TYPE_CHECKING

from pypeh.adapters.dataops.dataframe_adapter import DataFrameAdapter
from pypeh.core.interfaces.metadata import MetadataInterface
from pypeh.core.models.constants import ObservablePropertyValueType

if TYPE_CHECKING:
    from pypeh.core.models.internal_data_layout import DatasetSchemaElement

logger = logging.getLogger(__name__)


class DataFrameMetadataAdapter(DataFrameAdapter, MetadataInterface[DataFrame]):
    data_format = DataFrame

    def map_data_type(self, schema_element: DatasetSchemaElement) -> str:
        """
        Map the schema element's declared value type to the
        wire-format vocabulary (e.g. "STRING", "INTEGER", "FLOAT").
        """
        data_type = schema_element.data_type
        if data_type is None:
            return "UNKNOWN"
        return ObservablePropertyValueType(data_type).name
