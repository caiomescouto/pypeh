from __future__ import annotations

import logging
import polars as pl
from typing import TYPE_CHECKING

from polars.datatypes import DataType
from enum import Enum

from pypeh.core.interfaces.dataops import (
    JoinPlan,
    DataOpsInterface,
)
from pypeh.core.models.constants import ObservablePropertyValueType

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class DataFrameAdapter(DataOpsInterface[pl.DataFrame]):
    data_format = pl.DataFrame

    def get_element_labels(self, data: pl.DataFrame) -> list[str]:
        return data.columns

    def get_element_values(
        self, data: pl.DataFrame, element_label: str, as_list=False
    ) -> list[str] | set[str]:
        if as_list:
            return data.get_column(element_label).to_list()
        return set(data.get_column(element_label))

    def check_element_has_empty_values(
        self, data: pl.DataFrame, element_label: str
    ) -> bool:
        return data.select(pl.col(element_label).is_null().any()).item()

    def check_element_has_only_empty_values(
        self, data: pl.DataFrame, element_label: str
    ) -> bool:
        return data.select(pl.col(element_label).is_null().all()).item()

    def subset(
        self,
        data: pl.DataFrame,
        element_group: list[str],
        id_group: list[tuple[Any]] | None = None,
        identifying_elements: list[str] | None = None,
    ) -> pl.DataFrame:
        if id_group is None:
            ret = data.select(element_group)
        else:
            assert identifying_elements is not None
            ret = data.filter(
                pl.struct(identifying_elements).is_in(id_group)
            ).select(element_group)

        return ret

    def relabel(
        self, data: pl.DataFrame, element_mapping: dict[str, str]
    ) -> pl.DataFrame:
        return data.rename(element_mapping)

    def type_mapper(
        self, peh_value_type: str | ObservablePropertyValueType
    ) -> type[DataType]:
        if isinstance(peh_value_type, Enum):
            peh_value_type = peh_value_type.value

        match peh_value_type:
            case "string":
                return pl.String
            case "boolean":
                return pl.Boolean
            case "date":
                return pl.Date
            case "datetime":
                return pl.Datetime
            case "decimal":
                return pl.Float64
            case "integer":
                return pl.Int64
            case "float":
                return pl.Float64
            case _:
                return pl.String

    def normalize_input(self, data: pl.DataFrame) -> pl.LazyFrame:
        return data.lazy()

    def normalize_output(
        self, data: pl.DataFrame | pl.LazyFrame
    ) -> pl.DataFrame:
        if isinstance(data, pl.LazyFrame):
            return data.collect()
        return data

    def execute_join_plan(
        self,
        base_data: pl.DataFrame | pl.LazyFrame,
        datasets: dict[str, pl.DataFrame | pl.LazyFrame],
        join_plan: JoinPlan,
    ) -> pl.DataFrame | pl.LazyFrame:
        joined = base_data
        seen_edges = set()

        for edge in join_plan.edges:
            # Invariant: JoinPlan.from_join_specs orients every edge so the
            # base dataset is always the left side.
            if edge.left_dataset != join_plan.base_dataset_label:
                raise ValueError(
                    "JoinEdge is not oriented to base dataset "
                    f"'{join_plan.base_dataset_label}': {edge}"
                )
            left_on = edge.left_elements
            right_on = edge.right_elements
            other_dataset_label = edge.right_dataset

            edge_signature = (
                left_on,
                join_plan.base_dataset_label,
                right_on,
                other_dataset_label,
            )
            if edge_signature in seen_edges:
                continue
            seen_edges.add(edge_signature)

            other_dataset = datasets.get(other_dataset_label, None)
            assert other_dataset is not None

            required_fields = list(right_on)
            extra_fields = join_plan.required_fields_by_dataset.get(
                other_dataset_label, set()
            )
            for field in sorted(extra_fields):
                if field not in required_fields:
                    required_fields.append(field)

            if isinstance(joined, pl.LazyFrame):
                if isinstance(other_dataset, pl.LazyFrame):
                    selected_lf = other_dataset.select(required_fields)
                else:
                    selected_lf = other_dataset.select(required_fields).lazy()
                joined = joined.join(
                    selected_lf,
                    left_on=list(left_on),
                    right_on=list(right_on),
                    how=join_plan.how,
                    coalesce=join_plan.how == "full",
                )
            else:
                if isinstance(other_dataset, pl.LazyFrame):
                    selected_df = other_dataset.select(
                        required_fields
                    ).collect()
                else:
                    selected_df = other_dataset.select(required_fields)
                joined = joined.join(
                    selected_df,
                    left_on=list(left_on),
                    right_on=list(right_on),
                    how=join_plan.how,
                    coalesce=join_plan.how == "full",
                )

        return joined
