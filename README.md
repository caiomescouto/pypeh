# pypeh

`pypeh` is a lightweight ETL and data-ops toolkit for **Personal Exposure and Health (PEH)** data.

It helps you:
- work with PEH-model resources in Python
- load/transform/validate PEH study data
- support **FAIR** data workflows (findable, accessible, interoperable, reusable)

The toolkit is built to interact with the PEH model from PARC:
- https://github.com/eu-parc/parco-hbm

## Install

Core package:
```bash
uv pip install pypeh
```

With dataframe adapter extras (Polars-based workflows):
```bash
uv pip install "pypeh[dataframe-adapter]"
```

## Basic Usage

```python
from pypeh import Session

# Start a session
session = Session()
# Load PEH model resources (e.g. YAML configs) into cache
session.load_persisted_cache(source="config")
# Load tabular data as a DatasetSeries using a DataImportConfig from cache
data_import_config = session.cache.get(
    "<data_import_config_id>", "DataImportConfig"
)
dataset_series = session.load_tabular_dataset_series(
    source="my_data.xlsx",
    data_import_config=data_import_config,
)
```

From there you can use adapters for:
- validation
- enrichment (derived variables)
- aggregation
- export/persistence

## Run Tests

```bash
make test-core
make test-dataframe
make test-rocrate
```

## Documentation

Build the static documentation site with MkDocs:

```bash
make docs
```

Preview it locally:

```bash
make docs-serve
```
