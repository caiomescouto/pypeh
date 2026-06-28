from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from polars.datatypes import DataType, DataTypeClass

from pypeh.core.models.validation_errors import TypeCastError


TEMPORAL_BASE_TYPES = {pl.Date, pl.Datetime}


@dataclass(frozen=True)
class TemporalCastCheck:
    column_name: str
    source_type: str
    target_type: str
    reason: str

    def validate(
        self,
        source_data: pl.DataFrame,
        cast_data: pl.DataFrame,
        *,
        section_name: str,
    ) -> None:
        failed_mask = (
            source_data.get_column(self.column_name).is_not_null()
            & cast_data.get_column(self.column_name).is_null()
        )
        failed_count = failed_mask.sum()
        if not failed_count:
            return

        failed_values = (
            source_data.filter(failed_mask)
            .get_column(self.column_name)
            .head(5)
            .to_list()
        )
        raise TypeCastError(
            "Failed to cast Excel sheet "
            f"{section_name!r} using cast_error_policy='raise': "
            f"conversion from `{self.source_type}` to `{self.target_type}` "
            f"{self.reason} in column {self.column_name!r} for "
            f"{failed_count} value(s): {failed_values!r}"
        )


@dataclass(frozen=True)
class TemporalCast:
    expression: pl.Expr
    strict_check: TemporalCastCheck | None = None


def is_temporal_type(polars_type: DataType | DataTypeClass) -> bool:
    return polars_type.base_type() in TEMPORAL_BASE_TYPES


def build_temporal_cast(
    column_name: str,
    source_type: DataType | DataTypeClass,
    target_type: DataType | DataTypeClass,
    *,
    strict: bool,
) -> TemporalCast | None:
    source_base = source_type.base_type()
    target_base = target_type.base_type()
    if target_base not in TEMPORAL_BASE_TYPES:
        return None

    column = pl.col(column_name)
    if target_base == pl.Datetime:
        if source_base == pl.String:
            return TemporalCast(
                column.str.to_datetime(strict=strict).alias(column_name)
            )
        if source_base in {pl.Date, pl.Datetime}:
            return TemporalCast(column.cast(target_type, strict=strict))
        return _unsupported_temporal_cast(
            column_name,
            source_type=source_base.__name__,
            target_type="datetime",
            strict=strict,
        )

    if source_base == pl.String:
        parsed = column.str.to_datetime(strict=strict)
        return TemporalCast(
            _midnight_datetime_to_date_expr(parsed, column_name),
            strict_check=(
                _lossless_date_check(
                    column_name,
                    source_type="str",
                )
                if strict
                else None
            ),
        )
    if source_base == pl.Datetime:
        return TemporalCast(
            _midnight_datetime_to_date_expr(column, column_name),
            strict_check=(
                _lossless_date_check(
                    column_name,
                    source_type="datetime",
                )
                if strict
                else None
            ),
        )
    if source_base == pl.Date:
        return TemporalCast(column.cast(target_type, strict=strict))

    return _unsupported_temporal_cast(
        column_name,
        source_type=source_base.__name__,
        target_type="date",
        strict=strict,
    )


def _midnight_datetime_to_date_expr(
    datetime_expr: pl.Expr, column_name: str
) -> pl.Expr:
    return (
        pl.when(
            (datetime_expr.dt.hour() == 0)
            & (datetime_expr.dt.minute() == 0)
            & (datetime_expr.dt.second() == 0)
            & (datetime_expr.dt.nanosecond() == 0)
        )
        .then(datetime_expr.dt.date())
        .otherwise(pl.lit(None, dtype=pl.Date))
        .alias(column_name)
    )


def _lossless_date_check(
    column_name: str,
    *,
    source_type: str,
) -> TemporalCastCheck:
    return TemporalCastCheck(
        column_name=column_name,
        source_type=source_type,
        target_type="date",
        reason="would discard non-midnight time values",
    )


def _unsupported_temporal_cast(
    column_name: str,
    *,
    source_type: str,
    target_type: str,
    strict: bool,
) -> TemporalCast:
    return TemporalCast(
        pl.lit(
            None, dtype=pl.Datetime if target_type == "datetime" else pl.Date
        ).alias(column_name),
        strict_check=(
            TemporalCastCheck(
                column_name=column_name,
                source_type=source_type,
                target_type=target_type,
                reason="is not supported",
            )
            if strict
            else None
        ),
    )
