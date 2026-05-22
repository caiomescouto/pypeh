from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping
from urllib.parse import quote

from pypeh.core.models.constants import ObservablePropertyValueType
from pypeh.core.models.internal_data_layout import (
    Dataset,
    DatasetSchema,
    DatasetSchemaElement,
    DatasetSeries,
    ElementReference,
    ForeignKey,
)
from pypeh.adapters.persistence.filesystem import (
    ensure_filesystem_directory,
    join_filesystem_path,
)


PYPEH_DATASET_METADATA_KEY = b"pypeh.dataset.v1"
_FilesystemParquetSource = str | list[str] | tuple[str, ...]


@dataclass(frozen=True)
class _DatasetSeriesMetadata:
    label: str
    identifier: str
    metadata: dict[str, Any]

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[str, Any] | None
    ) -> "_DatasetSeriesMetadata | None":
        if metadata is None:
            return None
        return cls(
            label=metadata["label"],
            identifier=metadata["identifier"],
            metadata=dict(metadata.get("metadata", {})),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "identifier": self.identifier,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class _DatasetContextLink:
    observation_id: str
    observable_property_id: str
    element_label: str

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[str, str]
    ) -> "_DatasetContextLink":
        return cls(
            observation_id=metadata["observation_id"],
            observable_property_id=metadata["observable_property_id"],
            element_label=metadata["element_label"],
        )

    def to_metadata(self) -> dict[str, str]:
        return {
            "observation_id": self.observation_id,
            "observable_property_id": self.observable_property_id,
            "element_label": self.element_label,
        }


@dataclass(frozen=True)
class _DatasetParquetRecord:
    dataset: Dataset
    series_metadata: _DatasetSeriesMetadata | None
    context_links: list[_DatasetContextLink]
    source: Path | BinaryIO


def _require_dependencies():
    try:
        import polars as pl
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Dataset parquet persistence requires the dataframe dependencies "
            "('polars' and 'pyarrow')."
        ) from exc
    return pl, pq


def _schema_element_to_metadata(
    element: DatasetSchemaElement, *, is_primary_key: bool
) -> dict[str, Any]:
    return {
        "observable_property_id": element.observable_property_id,
        "data_type": element.data_type.value,
        "identifier": element.identifier,
        "is_primary_key": is_primary_key,
    }


def _dataset_to_metadata(dataset: Dataset) -> dict[str, Any]:
    return {
        "label": dataset.label,
        "identifier": dataset.identifier,
        "metadata": dataset.metadata,
        "observation_ids": sorted(dataset.observation_ids),
        "schema": {
            "identifier": dataset.schema.identifier,
            "elements": {
                element_label: _schema_element_to_metadata(
                    element,
                    is_primary_key=element_label
                    in dataset.schema.primary_keys,
                )
                for element_label, element in dataset.schema.elements.items()
            },
            "primary_keys": sorted(dataset.schema.primary_keys),
            "foreign_keys": {
                element_label: {
                    "identifier": foreign_key.identifier,
                    "references_dataset": (
                        foreign_key.reference.dataset_label
                    ),
                    "references_element": (
                        foreign_key.reference.element_label
                    ),
                }
                for element_label, foreign_key in (
                    dataset.schema.foreign_keys.items()
                )
            },
        },
    }


def _series_to_metadata(dataset: Dataset) -> _DatasetSeriesMetadata | None:
    series = dataset.part_of
    if series is None:
        return None
    return _DatasetSeriesMetadata(
        label=series.label,
        identifier=series.identifier,
        metadata=series.metadata,
    )


def _context_links_to_metadata(dataset: Dataset) -> list[_DatasetContextLink]:
    series = dataset.part_of
    if series is None:
        return []
    links: list[_DatasetContextLink] = []
    for (
        observation_id,
        observable_property_id,
    ), (dataset_label, element_label) in series._context_index.items():
        if dataset_label != dataset.label:
            continue
        links.append(
            _DatasetContextLink(
                observation_id=observation_id,
                observable_property_id=observable_property_id,
                element_label=element_label,
            )
        )
    links.sort(
        key=lambda link: (
            link.observation_id,
            link.observable_property_id,
            link.element_label,
        )
    )
    return links


def _build_metadata_payload(dataset: Dataset) -> dict[str, Any]:
    series_metadata = _series_to_metadata(dataset)
    return {
        "format": "pypeh.dataset.parquet",
        "version": 1,
        "series": (
            series_metadata.to_metadata()
            if series_metadata is not None
            else None
        ),
        "dataset": _dataset_to_metadata(dataset),
        "context_links": [
            link.to_metadata() for link in _context_links_to_metadata(dataset)
        ],
    }


def _decode_metadata(payload: bytes | None) -> dict[str, Any]:
    if payload is None:
        raise ValueError(
            "Parquet file does not contain pypeh dataset metadata."
        )
    data = json.loads(payload.decode("utf-8"))
    if data.get("format") != "pypeh.dataset.parquet":
        raise ValueError("Parquet file contains unsupported pypeh metadata.")
    if data.get("version") != 1:
        raise ValueError(
            "Parquet file contains unsupported pypeh metadata version."
        )
    return data


def _metadata_to_dataset(metadata: dict[str, Any], data: Any) -> Dataset:
    dataset_metadata = metadata["dataset"]
    schema_metadata = dataset_metadata["schema"]

    elements = {
        element_label: DatasetSchemaElement(
            label=element_label,
            observable_property_id=element_metadata["observable_property_id"],
            data_type=ObservablePropertyValueType(
                element_metadata["data_type"]
            ),
            identifier=element_metadata["identifier"],
        )
        for element_label, element_metadata in (
            schema_metadata["elements"].items()
        )
    }
    foreign_keys = {
        element_label: ForeignKey(
            element_label=element_label,
            reference=ElementReference(
                dataset_label=foreign_key_metadata["references_dataset"],
                element_label=foreign_key_metadata["references_element"],
            ),
            identifier=foreign_key_metadata["identifier"],
        )
        for element_label, foreign_key_metadata in (
            schema_metadata.get("foreign_keys", {}).items()
        )
    }
    schema = DatasetSchema(
        elements=elements,
        primary_keys=set(schema_metadata.get("primary_keys", [])),
        foreign_keys=foreign_keys,
        identifier=schema_metadata["identifier"],
    )
    return Dataset(
        label=dataset_metadata["label"],
        identifier=dataset_metadata["identifier"],
        metadata=dataset_metadata.get("metadata", {}),
        schema=schema,
        data=data,
        observation_ids=set(dataset_metadata.get("observation_ids", [])),
    )


def _dump_dataset_to_parquet(
    dataset: Dataset, destination: str | Path | BinaryIO
) -> str | Path | BinaryIO:
    """
    Dump one Dataset to one parquet file with pypeh schema metadata.

    DatasetSeries parquet persistence stores one file per Dataset. Keep this
    helper private so callers stay oriented around DatasetSeries.
    """
    pl, pq = _require_dependencies()
    if not isinstance(dataset.data, pl.DataFrame):
        raise TypeError(
            "DatasetSeries parquet persistence expects each dataset.data to be a "
            "polars.DataFrame."
        )
    assert dataset.data is not None
    table = dataset.data.to_arrow()
    metadata = dict(table.schema.metadata or {})
    metadata[PYPEH_DATASET_METADATA_KEY] = json.dumps(
        _build_metadata_payload(dataset),
        sort_keys=True,
    ).encode("utf-8")
    table = table.replace_schema_metadata(metadata)

    if isinstance(destination, (str, Path)):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination)
    return destination


def _load_dataset_record(
    source: str | Path | BinaryIO,
) -> _DatasetParquetRecord:
    pl, pq = _require_dependencies()
    normalized_source = (
        Path(source) if isinstance(source, (str, Path)) else source
    )
    table = pq.read_table(normalized_source)
    metadata = _decode_metadata(
        (table.schema.metadata or {}).get(PYPEH_DATASET_METADATA_KEY)
    )
    dataset = _metadata_to_dataset(metadata, pl.from_arrow(table))
    return _DatasetParquetRecord(
        dataset=dataset,
        series_metadata=_DatasetSeriesMetadata.from_metadata(
            metadata.get("series")
        ),
        context_links=[
            _DatasetContextLink.from_metadata(link)
            for link in metadata.get("context_links", [])
        ],
        source=normalized_source,
    )


def _dataset_filename(dataset_label: str) -> str:
    return f"{quote(dataset_label, safe='')}.parquet"


def dump_dataset_series_to_parquet(
    dataset_series: DatasetSeries, destination: str | Path
) -> list[Path]:
    """
    Dump every Dataset in a DatasetSeries to a parquet file in destination.
    """
    if hasattr(destination, "write"):
        raise TypeError(
            "DatasetSeries parquet persistence writes one parquet file per "
            "Dataset, so destination must be a directory path."
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for dataset_label in dataset_series:
        dataset = dataset_series[dataset_label]
        assert dataset is not None
        outputs.append(
            _dump_dataset_to_parquet(
                dataset,
                destination / _dataset_filename(dataset.label),
            )
        )
    return outputs


def dump_dataset_series_to_parquet_filesystem(
    dataset_series: DatasetSeries,
    file_system,
    destination: str,
) -> list[str]:
    """
    Dump every Dataset in a DatasetSeries through an fsspec filesystem.
    """
    ensure_filesystem_directory(file_system, destination)
    outputs = []
    for dataset_label in dataset_series:
        dataset = dataset_series[dataset_label]
        assert dataset is not None
        output_path = join_filesystem_path(
            file_system, destination, _dataset_filename(dataset.label)
        )
        with file_system.open(output_path, "wb") as output_file:
            _dump_dataset_to_parquet(dataset, output_file)
        outputs.append(output_path)
    return outputs


def _normalize_parquet_sources(
    source: str | Path | BinaryIO | Iterable[str | Path],
):
    if hasattr(source, "read") and not isinstance(source, (str, Path)):
        return [source]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            return sorted(path.glob("*.parquet"))
        return [path]
    return [Path(item) for item in source]


def _parquet_files_from_filesystem(
    file_system, source: _FilesystemParquetSource
) -> list[str]:
    if not isinstance(source, str):
        return list(source)

    if file_system.isfile(source):
        return [source]
    if file_system.isdir(source):
        pattern = join_filesystem_path(file_system, source, "*.parquet")
        return sorted(file_system.glob(pattern))
    raise ValueError(f"Path does not exist: {source}")


def _validate_foreign_keys(dataset_series: DatasetSeries) -> None:
    for dataset in dataset_series.parts.values():
        for foreign_key in dataset.schema.foreign_keys.values():
            referenced_dataset = dataset_series.get(
                foreign_key.reference.dataset_label
            )
            if referenced_dataset is None:
                raise ValueError(
                    "Foreign key references missing dataset "
                    f"{foreign_key.reference.dataset_label!r} from dataset "
                    f"{dataset.label!r}."
                )
            if (
                referenced_dataset.schema.get_element_by_label(
                    foreign_key.reference.element_label
                )
                is None
            ):
                raise ValueError(
                    "Foreign key references missing element "
                    f"{foreign_key.reference.element_label!r} on dataset "
                    f"{foreign_key.reference.dataset_label!r}."
                )


def load_dataset_series_from_parquet(
    source: str | Path | BinaryIO | Iterable[str | Path],
    *,
    validate_foreign_keys: bool = True,
) -> DatasetSeries:
    """
    Load one or more pypeh dataset parquet files into a DatasetSeries.
    """
    sources = _normalize_parquet_sources(source)
    if len(sources) == 0:
        raise ValueError("No parquet files found to load.")

    records = [_load_dataset_record(path) for path in sources]
    return _build_dataset_series_from_records(records, validate_foreign_keys)


def load_dataset_series_from_parquet_filesystem(
    file_system,
    source: _FilesystemParquetSource,
    *,
    validate_foreign_keys: bool = True,
) -> DatasetSeries:
    """
    Load a DatasetSeries from pypeh dataset parquet files via fsspec.
    """
    sources = _parquet_files_from_filesystem(file_system, source)
    if len(sources) == 0:
        raise ValueError("No parquet files found to load.")

    records = []
    for path in sources:
        with file_system.open(path, "rb") as source_file:
            records.append(_load_dataset_record(source_file))
    return _build_dataset_series_from_records(records, validate_foreign_keys)


def _build_dataset_series_from_records(
    records: list[_DatasetParquetRecord],
    validate_foreign_keys: bool,
) -> DatasetSeries:
    series_metadata = records[0].series_metadata
    if series_metadata is None:
        series_metadata = _DatasetSeriesMetadata(
            label="dataset_series",
            identifier=records[0].dataset.identifier,
            metadata={},
        )

    for record in records[1:]:
        if record.series_metadata is None:
            continue
        if (
            record.series_metadata.identifier != series_metadata.identifier
            or record.series_metadata.label != series_metadata.label
        ):
            raise ValueError(
                "Parquet files do not belong to the same DatasetSeries."
            )

    dataset_series = DatasetSeries(
        label=series_metadata.label,
        identifier=series_metadata.identifier,
        metadata=series_metadata.metadata,
    )
    for record in records:
        dataset_series.register_dataset(record.dataset)

    dataset_series.build_observation_index()
    for record in records:
        for context_link in record.context_links:
            dataset_series._register_observable_property(
                observable_property_id=context_link.observable_property_id,
                observation_id=context_link.observation_id,
                dataset_label=record.dataset.label,
                element_label=context_link.element_label,
            )

    if validate_foreign_keys:
        _validate_foreign_keys(dataset_series)

    return dataset_series
