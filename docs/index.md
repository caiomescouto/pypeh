# pypeh

`pypeh` is a lightweight ETL and data-operations toolkit for Personal Exposure
and Health (PEH) data.

It helps you load PEH model resources, keep them in a session cache, read
tabular study data into structured datasets, and run validation, enrichment,
aggregation, export, or persistence workflows through adapters.

## Install

Install the core package:

```bash
uv pip install pypeh
```

Install dataframe support for Polars-based tabular workflows:

```bash
uv pip install "pypeh[dataframe-adapter]"
```

## First Steps

Most workflows start with a `Session`:

```python
from pypeh import Session

session = Session()
session.load_persisted_cache(source="config")
```

See the Session API guide for the first documented workflow surface.

For YAML configuration of enrichment and aggregation calculations, see the
Calculation Configuration guide.

## Build These Docs

The documentation site is built with MkDocs:

```bash
make docs
```

To preview locally:

```bash
make docs-serve
```
