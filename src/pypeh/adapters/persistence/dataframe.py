from __future__ import annotations

import logging
import polars as pl
import io

from pathlib import Path
from typing import Union, IO, Literal, Mapping
from polars.datatypes import DataType, DataTypeClass

from pypeh.core.models.constants import ObservablePropertyValueType
from pypeh.core.models.validation_errors import (
    TypeCastError,
    ValidationErrorReport,
    build_type_cast_error_report,
)
from pypeh.adapters.persistence.temporal import (
    TemporalCastCheck,
    build_temporal_cast,
    is_temporal_type,
)
from pypeh.adapters.persistence.serializations import IOAdapter

logger = logging.getLogger(__name__)


DATAFRAME_TYPE_MAPPING: dict[
    ObservablePropertyValueType, DataType | DataTypeClass
] = {
    ObservablePropertyValueType.DATE: pl.Date,
    ObservablePropertyValueType.DATETIME: pl.Datetime,
    ObservablePropertyValueType.BOOLEAN: pl.Boolean,
    ObservablePropertyValueType.FLOAT: pl.Float64,
    ObservablePropertyValueType.INTEGER: pl.Int64,
    ObservablePropertyValueType.STRING: pl.Utf8,
    ObservablePropertyValueType.CATEGORICAL: pl.Utf8,
    ObservablePropertyValueType.DECIMAL: pl.Float64,
}


CastErrorPolicy = Literal["null", "raise", "report"]


class CsvIOImpl(IOAdapter):
    def load(
        self, source: Union[str, Path, IO[str], IO[bytes]], **kwargs
    ) -> pl.DataFrame:
        try:
            if hasattr(source, "read") and not isinstance(source, (str, Path)):
                encoding = kwargs.get("encoding", None)
                if encoding is None:
                    encoding = "utf-8"
                data = source.read()
                if isinstance(data, memoryview):
                    data = data.tobytes().decode(encoding)
                elif isinstance(data, (bytes, bytearray)):
                    data = data.decode(encoding)
                buffer = io.StringIO(data)
                return pl.read_csv(buffer, **kwargs)
            else:
                return pl.read_csv(source=str(source), **kwargs)

        except Exception as e:
            logger.error(f"Error in CSVIOImpl: {e}")
            raise

    def dump(self, destination: str, **kwargs):
        raise NotImplementedError(
            "CsvIOImpl.dump is not implemented yet. "
            f"destination={destination!r}, kwargs={kwargs!r}."
        )


class ExcelIOImpl(IOAdapter):
    @staticmethod
    def _build_typed_schema(
        data_schema: dict[str, str] | None,
    ) -> Mapping[str, DataType | DataTypeClass] | None:
        if data_schema is None:
            return None

        typed_schema: dict[str, DataType | DataTypeClass] = {}
        for key, value in data_schema.items():
            polars_type = DATAFRAME_TYPE_MAPPING.get(value, None)
            if polars_type is None:
                logger.debug(
                    f"Cound not find {value} in DATAFRAME_TYPE_MAPPING"
                )
                raise KeyError(
                    f"Could not find {value} in DATAFRAME_TYPE_MAPPING"
                )
            typed_schema[key] = polars_type
        return typed_schema

    @staticmethod
    def _validate_cast_error_policy(
        cast_error_policy: str,
    ) -> CastErrorPolicy:
        if cast_error_policy not in {"null", "raise", "report"}:
            raise ValueError(
                "cast_error_policy must be one of 'null', 'raise', or "
                "'report'"
            )
        return cast_error_policy

    @staticmethod
    def _build_read_dtypes(
        typed_schema: Mapping[str, DataType | DataTypeClass] | None,
        *,
        cast_error_policy: CastErrorPolicy,
    ) -> dict[str, str] | None:
        if typed_schema is None:
            return None

        def to_reader_dtype(
            polars_type: DataType | DataTypeClass,
        ) -> str | None:
            if is_temporal_type(polars_type):
                return None

            base_type = polars_type.base_type()
            if base_type in {
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.Int128,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            }:
                return "int"
            if base_type in {pl.Float32, pl.Float64}:
                return "float"
            if base_type == pl.Boolean:
                return "boolean"
            return "string"

        if cast_error_policy == "null":
            read_dtypes = {}
            for column_name, polars_type in typed_schema.items():
                reader_dtype = to_reader_dtype(polars_type)
                if reader_dtype is not None:
                    read_dtypes[column_name] = reader_dtype
            return read_dtypes or None

        read_dtypes = {
            column_name: "string"
            for column_name, polars_type in typed_schema.items()
            if not is_temporal_type(polars_type)
        }
        return read_dtypes or None

    def _cast_frame_to_schema(
        self,
        data: pl.DataFrame,
        typed_schema: Mapping[str, DataType | DataTypeClass] | None,
        *,
        section_name: str,
        cast_error_policy: CastErrorPolicy,
    ) -> pl.DataFrame:
        if typed_schema is None:
            return data

        strict = cast_error_policy == "raise"
        cast_expressions = []
        strict_temporal_checks: list[TemporalCastCheck] = []
        for column_name, polars_type in typed_schema.items():
            if column_name not in data.columns:
                continue

            source_type = data.schema[column_name].base_type()
            temporal_cast = build_temporal_cast(
                column_name,
                source_type,
                polars_type,
                strict=strict,
            )
            if temporal_cast is not None:
                cast_expressions.append(temporal_cast.expression)
                if temporal_cast.strict_check is not None:
                    strict_temporal_checks.append(temporal_cast.strict_check)
                continue

            cast_expressions.append(
                pl.col(column_name).cast(
                    polars_type,
                    strict=strict,
                )
            )
        if not cast_expressions:
            return data

        try:
            cast_data = data.with_columns(cast_expressions)
        except pl.exceptions.PolarsError as exc:
            raise TypeCastError(
                "Failed to cast Excel sheet "
                f"{section_name!r} using cast_error_policy='raise': {exc}"
            ) from exc

        for check in strict_temporal_checks:
            check.validate(data, cast_data, section_name=section_name)

        return cast_data

    def _load(
        self, source: Union[str, Path, IO[str], IO[bytes], bytes], **options
    ) -> pl.DataFrame | dict[str, pl.DataFrame]:
        if isinstance(source, bytes):
            # Handle raw bytes data
            result = pl.read_excel(
                source=io.BytesIO(source),
                **options,
            )
        elif hasattr(source, "read") and not isinstance(source, (str, Path)):
            # Handle file-like objects
            if isinstance(source, IO) and "b" not in getattr(
                source, "mode", "b"
            ):
                raise ValueError("Excel source must be opened in binary mode")
            data = source.read()  # type: ignore
            result = pl.read_excel(  # type: ignore
                source=io.BytesIO(data),  # type: ignore
                **options,
            )
        else:
            # Handle file paths
            result = pl.read_excel(  # type: ignore
                source=str(source),  # type: ignore
                **options,
            )
        return result

    def _read_source_data(
        self, source: Union[str, Path, IO[str], IO[bytes]]
    ) -> bytes | None:
        """Read data from source once and cache it"""
        if hasattr(source, "read") and not isinstance(source, (str, Path)):
            if isinstance(source, IO) and "b" not in getattr(
                source, "mode", "b"
            ):
                raise ValueError("Excel source must be opened in binary mode")
            ret = source.read()
            assert isinstance(ret, bytes)
            return ret
        return None

    def load_section(
        self,
        source: Union[str, Path, IO[str], IO[bytes], bytes],
        section_name: str,
        data_schema: dict[str, str] | None = None,
        cast_error_policy: CastErrorPolicy = "null",
        cached_data: bytes | None = None,
    ) -> pl.DataFrame | ValidationErrorReport:
        typed_schema = self._build_typed_schema(data_schema)
        cast_error_policy = self._validate_cast_error_policy(cast_error_policy)

        default = {
            "engine": "calamine",
            "has_header": True,
        }
        options = {
            **default,
            "sheet_name": section_name,
        }
        read_dtypes = self._build_read_dtypes(
            typed_schema, cast_error_policy=cast_error_policy
        )
        if read_dtypes is not None:
            options["read_options"] = {"dtypes": read_dtypes}

        ret = self._load(source, **options)
        assert isinstance(ret, pl.DataFrame)
        try:
            return self._cast_frame_to_schema(
                ret,
                typed_schema,
                section_name=section_name,
                cast_error_policy=(
                    "raise"
                    if cast_error_policy == "report"
                    else cast_error_policy
                ),
            )
        except TypeCastError as exc:
            if cast_error_policy != "report":
                raise
            return build_type_cast_error_report(
                exc,
                group_id=section_name,
                group_type="excel_cast_error",
                name=f"Excel cast error in sheet {section_name!r}",
                metadata={"section_name": section_name},
                source="ExcelIOImpl.load_section",
            )

    def load(
        self,
        source: Union[str, Path, IO[str], IO[bytes]],
        data_schema: dict[str, dict[str, str]] | None = None,
        cast_error_policy: CastErrorPolicy = "null",
        **kwargs,
    ) -> dict[str, pl.DataFrame | ValidationErrorReport]:
        try:
            cast_error_policy = self._validate_cast_error_policy(
                cast_error_policy
            )
            # if data_schema is provided we need to load each sheet individually
            if data_schema is not None:
                cached_data = self._read_source_data(source)
                assert cached_data is not None
                result = {}
                for section_name, typing_dict in data_schema.items():
                    result[section_name] = self.load_section(
                        cached_data,
                        section_name,
                        typing_dict,
                        cast_error_policy=cast_error_policy,
                    )

            else:
                default = {
                    "sheet_id": 0,
                    "engine": "calamine",
                    "has_header": True,
                }
                options = {**default, **kwargs}
                result = self._load(source, **options)
            assert isinstance(result, dict)

            return result

        except Exception as e:
            logger.error(f"Error in ExcelIOImpl: {e}")
            raise

    def dump(self, destination: str, **kwargs):
        raise NotImplementedError(
            "ExcelIOImpl.dump is not implemented yet. "
            f"destination={destination!r}, kwargs={kwargs!r}."
        )
