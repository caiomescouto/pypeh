# Session API

`Session` is the main orchestration object in `pypeh`. It holds configured
connections, an in-memory cache of PEH model resources, optional namespace
management, and the adapters used for tabular data operations.

Import it from the package root:

```python
from pypeh import Session
```

## Create a Session

For simple local workflows, create an empty session and load resources later:

```python
session = Session()
session.load_persisted_cache(source="config")
```

For explicit local file storage, pass a `LocalFileConfig` and make it the
default connection:

```python
from pypeh import LocalFileConfig, Session

session = Session(
    connection_config=[
        LocalFileConfig(
            label="local_file",
            config_dict={"root_folder": "path/to/project"},
        ),
    ],
    default_connection="local_file",
)
```

You can also load the default persisted cache during initialization:

```python
session = Session(
    connection_config=[
        LocalFileConfig(
            label="local_file",
            config_dict={"root_folder": "path/to/project"},
        ),
    ],
    default_connection="local_file",
    load_from_default_connection="",
)
```

## Environment-Configured Default Cache

If `DEFAULT_PERSISTED_CACHE_TYPE=LocalFile` is set, `Session()` creates a
default local-file cache connection from environment variables with the
`DEFAULT_PERSISTED_CACHE_` prefix.

For example:

```bash
export DEFAULT_PERSISTED_CACHE_TYPE=LocalFile
export DEFAULT_PERSISTED_CACHE_ROOT_FOLDER=/path/to/project
```

Then:

```python
session = Session()
session.load_persisted_cache()
```

## Load PEH Resources

Use `load_persisted_cache` to load YAML resources into the session cache:

```python
session.load_persisted_cache(
    source="observations.yaml",
    connection_label="local_file",
)
```

Use `load_resource` when you need one resource by identifier and type:

```python
observation = session.load_resource(
    resource_identifier="peh:OBSERVATION_ADULTS_URINE_LAB",
    resource_type="Observation",
    resource_path="observations.yaml",
    connection_label="local_file",
)
```

You can retrieve already-cached resources with `get_resource`:

```python
observation = session.get_resource(
    "peh:OBSERVATION_ADULTS_URINE_LAB",
    "Observation",
)
```

## Import Tabular Data

`import_tabular_dataset_series` imports external tabular data into a
`DatasetSeries` using a PEH `DataImportConfig`. The `import_` name is used
because this workflow needs import mapping metadata from the
`DataImportConfig`, not only a file path.

```python
from peh_model.peh import (
    DataImportConfig,
    DataImportSectionMapping,
    DataImportSectionMappingLink,
)

data_import_config = DataImportConfig(
    id="peh:IMPORT_CONFIG_SAMPLE_METADATA",
    layout="peh:CODEBOOK_v2.4_LAYOUT_SAMPLE_METADATA",
    section_mapping=DataImportSectionMapping(
        section_mapping_links=[
            DataImportSectionMappingLink(
                section="SAMPLE_METADATA_SECTION_SAMPLE",
                observation_id_list=["peh:VALIDATION_TEST_SAMPLE_METADATA"],
            ),
        ]
    ),
)

dataset_series = session.import_tabular_dataset_series(
    source="sample_metadata.xlsx",
    data_import_config=data_import_config,
    connection_label="local_file",
)
```

`load_tabular_dataset_series` is kept as a deprecated compatibility alias. It
accepts the same arguments, logs a warning, and forwards to
`import_tabular_dataset_series`.

The method checks loaded labels against the expected schema. By default, type
cast and schema errors are raised. Use `cast_error_policy="report"` or
`schema_error_policy="report"` to receive a `ValidationErrorReportCollection`
instead.

```python
result = session.import_tabular_dataset_series(
    source="sample_metadata.xlsx",
    data_import_config=data_import_config,
    connection_label="local_file",
    cast_error_policy="report",
    schema_error_policy="report",
)
```

Set `allow_incomplete=True` to allow missing labels while still reporting
undefined labels.

## Persist or Export Tabular Data

Use `dump_tabular_dataset_series` to persist a tabular `DatasetSeries` after it
has been imported, validated, enriched, or otherwise prepared.

Use `file_format="parquet"` when you need pypeh semantic persistence. The
parquet format writes one file per `Dataset` in the series and returns the
paths that were written.

```python
parquet_paths = session.dump_tabular_dataset_series(
    dataset_series=dataset_series,
    output_path="exports/sample_metadata",
    file_format="parquet",
    connection_label="local_file",
)
```

Read the files back with `read_tabular_dataset_series`. Pass the list returned
by `dump_tabular_dataset_series`, or another sequence of pypeh dataset parquet
paths.

```python
restored_dataset_series = session.read_tabular_dataset_series(
    source_paths=parquet_paths,
    connection_label="local_file",
)
```

Reading currently supports `file_format="parquet"` only. It validates
foreign-key references by default; set `validate_foreign_keys=False` when
loading a partial subset intentionally.

Use `file_format="xlsx"` when you need a human-facing Excel export. The XLSX
format writes a single workbook, with one worksheet per `Dataset`. Each
worksheet contains the Polars dataframe stored in `dataset.data`. Dataset labels
are used as worksheet names.

```python
xlsx_paths = session.dump_tabular_dataset_series(
    dataset_series=dataset_series,
    output_path="exports/sample_metadata.xlsx",
    file_format="xlsx",
    connection_label="local_file",
)
```

The returned list contains one workbook path. XLSX export is not a semantic
persistence format: it does not preserve `DatasetSeries` metadata, schemas,
foreign-key links, or context indexes for round-tripping through
`read_tabular_dataset_series`. Excel export requires the `xlsxwriter`
dependency used by Polars' native `DataFrame.write_excel` support.

## Split Data by Observation

Use `split_dataset_series_by_observation` to normalize an imported
`DatasetSeries` into observation-specific datasets. This is useful when source
files were organized for collection or import, but downstream validation,
enrichment, or export workflows should operate on one observation at a time.

```python
observation_dataset_series = session.split_dataset_series_by_observation(
    source_dataset_series=dataset_series,
)
```

The method delegates to the registered data-operations adapter. The adapter
uses the `DatasetSeries` schema, observation membership, contextual field
references, and foreign-key metadata to construct a new `DatasetSeries`.

Datasets that contain fields for multiple observations may be split apart. If
fields for one observation are spread across multiple datasets, the adapter can
join those fields into one output dataset when the `DatasetSeries` declares the
required foreign-key links.

When multiple source datasets contribute fields with the same column label to
one observation dataset, `label_collision_strategy` controls how split output
labels are handled:

- `"prefix_source_dataset"` is the default and preserves the historical
  behavior. The first occurrence keeps its label; later collisions are prefixed
  with the source dataset label, for example `UAPFAS_egg_lab__pftrds`.
- `"prefix_observable_property_id"` prefixes collisions with the unique tail of
  the observable property identifier, for example `01KT68...__pftrds`. This is
  useful when provenance should be tied to the semantic variable rather than the
  source table.
- `"error"` raises a `ValueError` instead of renaming colliding fields.

Split stores the semantic mapping from `(observation, observable property)` to
the actual output `DatasetSchemaElement.label`. Downstream dataframe operations
use that schema label, not the observable property's `short_name` or
`ui_label`, so enrichment can still find columns that were renamed during
split. Split also records the original source dataset and field in dataset
metadata so later label strategies can preserve source provenance.

Pass `new_dataset_series_label` when you want to control the returned series
label:

```python
observation_dataset_series = session.split_dataset_series_by_observation(
    source_dataset_series=dataset_series,
    new_dataset_series_label="study_by_observation",
    label_collision_strategy="prefix_source_dataset",
)
```

The returned `DatasetSeries` contains datasets organized by observation. Each
output dataset is associated with one observation and contains the fields
relevant to that observation.

## Validate Tabular Data

Validate one dataset:

```python
report = session.validate_tabular_dataset(
    data=dataset_series["SAMPLE"],
    dependent_data=dataset_series,
)
```

Validate every dataset in a series:

```python
reports = session.validate_tabular_dataset_series(dataset_series)
```

Build validation configuration from a cached `DataLayout`:

```python
data_layout = session.load_resource(
    "peh:CODEBOOK_v2.4_LAYOUT_SAMPLE_METADATA",
    "DataLayout",
)
validation_configs = session.build_validation_config(data_layout)
```

## Register Adapters

The session can use default adapters where available. To override a workflow,
register an adapter for its functionality key:

```python
session.register_adapter("validation", validation_adapter)
session.register_adapter("dataops", dataops_adapter)
session.register_adapter("enrichment", enrichment_adapter)
session.register_adapter("aggregate", aggregation_adapter)
```

You can also register by import path:

```python
session.register_adapter_by_name(
    "validation",
    "my_package.validation",
    "MyValidationAdapter",
)
```

## Enrichment and Aggregation

`enrich` and `aggregate` delegate to the registered adapter while passing a
cache view and the source and target observations.

When the target observations are stored as an `ObservationGroup` of
`DerivedObservation` resources, use `unpack_derived_observation_group` to
resolve each target observation and the source observation referenced by its
`was_derived_from` field.

```python
observation_pairs = list(
    session.unpack_derived_observation_group(
        observation_group_id="peh:TARGET_DERIVED_OBSERVATIONS"
    )
)
target_observations = [target for target, source in observation_pairs]
source_observations = [source for target, source in observation_pairs]
```

```python
enriched = session.enrich(
    source_dataset_series=dataset_series,
    target_observations=target_observations,
    target_derived_from=source_observations,
)

summary = session.aggregate(
    source_dataset_series=dataset_series,
    target_observations=target_observations,
    target_derived_from=source_observations,
)
```

The target observation list and source observation list must have the same
length.

Derived target observations can also contain multiple observable properties
whose preferred output labels are identical. By default, `enrich` and
`aggregate` reject this because dataframe adapters need concrete column names
and silently overwriting one target would make the dependency graph
inconsistent.

Use `target_label_collision_strategy` when duplicate target labels are expected:

- `"error"` is the default. It raises when two target observable properties
  resolve to the same output label.
- `"prefix_observable_property_id"` prefixes the output label with the unique
  tail of the observable property identifier, which is robust but less
  human-readable.
- `"prefix_source_dataset"` prefixes the output label with the source dataset
  that feeds the derived
  calculation. This is useful when two derived variables have the same
  `short_name`/`ui_label` but come from different source tabs. If enrichment is
  performed after `split_dataset_series_by_observation`, this strategy uses the
  source provenance recorded by split rather than the post-split dataset label.

```python
enriched = session.enrich(
    source_dataset_series=observation_dataset_series,
    target_observations=target_observations,
    target_derived_from=source_observations,
    target_label_collision_strategy="prefix_source_dataset",
)

summary = session.aggregate(
    source_dataset_series=observation_dataset_series,
    target_observations=target_observations,
    target_derived_from=source_observations,
    target_label_collision_strategy="prefix_source_dataset",
)
```

## Namespaces and Minting

Bind a `NamespaceManager` before minting new PEH resources:

```python
from peh_model.peh import ObservableProperty
from pypeh import NamespaceManager

namespace_manager = NamespaceManager(
    default_base_uri="https://w3id.org/example/id/"
)
session.bind_namespace_manager(namespace_manager)

observable_property = session.mint_and_cache(
    ObservableProperty,
    ui_label="cholesterol",
)
```

The minted resource is added to the session cache.
