# Calculation Configuration

PEH calculation configuration describes derived values in model resources. In
`pypeh`, the same PEH shape is used for enrichment and aggregation:

```yaml
calculation_design:
  calculation_name: mean
  calculation_implementation:
    function_name: pypeh.adapters.aggregation.polars_adapter.statistics.statistics_mean
    function_kwargs:
    - contextual_field_reference:
        dataset_label: peh:SOURCE_OBSERVATION
        field_label: peh:MEASURED_PROPERTY
    - mapping_name: cutoff
      value: '0.75'
      value_type: float
```

This page documents the parts of that configuration that users are expected to
write. It intentionally does not document every adapter method.

## Interface Level

The interface-level configuration is adapter-independent. These fields describe
what should be calculated and how calculation arguments map to PEH data:

- `calculation_design`: attaches a calculation to an observable property.
- `calculation_implementation.function_name`: import path of the calculation
  function to use.
- `calculation_implementation.function_kwargs`: arguments passed to that
  function.
- `CalculationKeywordArgument.contextual_field_reference`: points to another
  PEH observation/property and is resolved to the relevant source field.
- `CalculationKeywordArgument.mapping_name`: names the function argument.
- `CalculationKeywordArgument.value`: scalar argument value.
- `CalculationKeywordArgument.value_type`: PEH value type used to coerce scalar
  values through the active adapter's `type_mapper`.

Scalar calculation arguments must include both `mapping_name` and `value_type`.
For example:

```yaml
- mapping_name: cutoff
  value: '0.75'
  value_type: float
```

Contextual field arguments may be used in two ways:

```yaml
# Primary source value for aggregation, or unnamed source argument where the
# workflow expects one.
- contextual_field_reference:
    dataset_label: peh:SOURCE_OBSERVATION
    field_label: peh:MEASURED_PROPERTY

# Named contextual argument passed to the function.
- mapping_name: below_col
  contextual_field_reference:
    dataset_label: peh:SOURCE_OBSERVATION
    field_label: peh:LIMIT_PROPERTY
```

## Enrichment

Enrichment creates new fields from existing fields, usually row by row.

For enrichment, every named contextual field reference becomes a named function
argument. Scalar arguments are also named function arguments.

```yaml
calculation_design:
  calculation_name: corrected_measurement
  calculation_implementation:
    function_name: my_project.enrichment.correct_measurement
    function_kwargs:
    - mapping_name: measured
      contextual_field_reference:
        dataset_label: peh:LAB_OBSERVATION
        field_label: peh:MEASURED_VALUE
    - mapping_name: correction_factor
      value: '1.25'
      value_type: float
```

The configured function should therefore accept arguments with the same names:

```python
def correct_measurement(measured, correction_factor):
    return measured * correction_factor
```

## Aggregation

Aggregation summarizes one source observation into a target summary observation.
Identifying observable properties in the target observation design become
stratification columns.

For aggregation, the primary source value is the contextual field reference
without `mapping_name`:

```yaml
- contextual_field_reference:
    dataset_label: peh:SOURCE_OBSERVATION
    field_label: peh:BIRTH_WEIGHT
```

Additional contextual arguments must be named. They are passed as source column
labels to the statistic function:

```yaml
- mapping_name: below_col
  contextual_field_reference:
    dataset_label: peh:SOURCE_OBSERVATION
    field_label: peh:LIMIT_OF_QUANTIFICATION
```

Scalar arguments are named and typed:

```yaml
- mapping_name: cutoff
  value: '0.75'
  value_type: float
```

A complete aggregation statistic can look like this:

```yaml
calculation_design:
  calculation_name: mean_with_cutoff
  calculation_implementation:
    function_name: pypeh.adapters.aggregation.polars_adapter.statistics.statistics_mean
    function_kwargs:
    - contextual_field_reference:
        dataset_label: peh:SOURCE_OBSERVATION
        field_label: peh:BIRTH_WEIGHT
    - mapping_name: cutoff
      value: '0.75'
      value_type: float
```

## Adapter Level

The adapter level determines how interface-level configuration is executed.
Users normally see this through `function_name`.

With the Polars dataframe adapter:

- source fields resolve to Polars column labels
- scalar values are coerced using the Polars adapter's `type_mapper`
- unsupported keyword arguments are ignored per statistic function
- aggregation statistic functions live under
  `pypeh.adapters.aggregation.polars_adapter.statistics`

For example, the Polars aggregation statistics currently include functions such
as:

```yaml
function_name: pypeh.adapters.aggregation.polars_adapter.statistics.statistics_mean
function_name: pypeh.adapters.aggregation.polars_adapter.statistics.statistics_sem
function_name: pypeh.adapters.aggregation.polars_adapter.statistics.stat_count_below
```

Adapter-level names are implementation details in the sense that another adapter
may provide different function paths or support different keyword arguments. The
PEH configuration pattern remains the same: contextual field references for
source fields, named scalar arguments with explicit `value_type`, and
adapter-resolved execution.
