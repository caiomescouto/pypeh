import pytest

import math

from pypeh.core.interfaces.dataops import AggregationInterface


@pytest.fixture(scope="module")
def setup_adapter():
    return AggregationInterface.get_default_adapter_class()


@pytest.fixture()
def pl():
    import polars as pl

    return pl


@pytest.fixture
def sample_dataframe(pl):
    """Create a sample dataframe for testing."""

    return pl.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        }
    )


@pytest.fixture
def dataframe_with_nulls(pl):
    """Create a dataframe with null values for testing."""

    return pl.DataFrame(
        {
            "value": [1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0, 9.0, 10.0],
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        }
    )


@pytest.fixture
def small_dataframe(pl):
    """Create a small dataframe for testing edge cases."""

    return pl.DataFrame(
        {
            "value": [1.0, 2.0, 3.0],
        }
    )


@pytest.mark.dataframe
class TestStatCount:
    """Test suite for stat_count function."""

    def test_stat_count_basic(self, sample_dataframe, setup_adapter, pl):
        """Test basic count statistics without nulls."""
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("stat_count")("value")
        result = sample_dataframe.select(exprs)

        assert result.shape == (1, 3)
        assert result["n"][0] == 10
        assert result["missing_n"][0] == 0
        assert result["missing_pct"][0] == 0.0

    def test_stat_count_with_nulls(
        self, dataframe_with_nulls, setup_adapter, pl
    ):
        """Test count statistics with null values."""
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("stat_count")("value")
        result = dataframe_with_nulls.select(exprs)

        assert result["n"][0] == 10
        assert result["missing_n"][0] == 2
        assert result["missing_pct"][0] == 0.2

    def test_stat_count_all_nulls(self, setup_adapter, pl):
        """Test count statistics when all values are null."""

        df = pl.DataFrame({"value": [None, None, None]})
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_count")("value")
        result = df.select(exprs)

        assert result["n"][0] == 3
        assert result["missing_n"][0] == 3
        assert result["missing_pct"][0] == 1.0

    def test_stat_count_empty_dataframe(self, setup_adapter, pl):
        """Test count statistics on empty dataframe."""

        df = pl.DataFrame({"value": []}, schema={"value": pl.Float64})

        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_count")("value")
        result = df.select(exprs)

        assert result["n"][0] == 0
        assert result["missing_n"][0] == 0

    @pytest.mark.parametrize("name", ["n", "missing_n", "missing_pct"])
    def test_statistics_count_unit_fn(
        self, name, sample_dataframe, setup_adapter, pl
    ):
        """Test that count statistics can be called with unit function."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name(
            f"statistics_count_{name}"
        )("value")
        result = sample_dataframe.select(exprs)

        assert name in result.columns


@pytest.mark.dataframe
class TestStatCountBelow:
    """Test suite for stat_count_below function."""

    def test_stat_count_below_basic(self, setup_adapter, pl):
        """Test counting rows where value is below another float column."""
        df = pl.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 4.0],
                "limit": [2.0, 2.0, 5.0, 4.0],
            }
        )
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("stat_count_below")(
            "value", below_col="limit"
        )
        result = df.select(exprs)

        assert result.shape == (1, 2)
        assert result["below_n"][0] == 2
        assert result["below_pct"][0] == pytest.approx(0.5)

    def test_stat_count_below_does_not_count_equal_values(
        self, setup_adapter, pl
    ):
        """Test that only strictly lower values are counted."""
        df = pl.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 4.0],
                "limit": [1.0, 2.0, 4.0, 4.0],
            }
        )
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("stat_count_below")(
            "value", below_col="limit"
        )
        result = df.select(exprs)

        assert result["below_n"][0] == 1
        assert result["below_pct"][0] == pytest.approx(0.25)

    def test_stat_count_below_custom_aliases(self, setup_adapter, pl):
        """Test that result aliases can be customized."""
        df = pl.DataFrame(
            {
                "value": [1.0, 2.0, 3.0],
                "limit": [2.0, 2.0, 2.0],
            }
        )
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("stat_count_below")(
            "value",
            below_col="limit",
            result_aliases=["under_n", "under_pct"],
        )
        result = df.select(exprs)

        assert "under_n" in result.columns
        assert "under_pct" in result.columns
        assert result["under_n"][0] == 1
        assert result["under_pct"][0] == pytest.approx(1 / 3)

    @pytest.mark.parametrize("name", ["n", "pct"])
    def test_statistics_count_below_unit_fn(self, name, setup_adapter, pl):
        """Test that count-below statistics can be called with unit function."""
        df = pl.DataFrame(
            {
                "value": [1.0, 2.0, 3.0],
                "limit": [2.0, 2.0, 2.0],
            }
        )
        adapter = setup_adapter()
        stat_name = f"statistics_count_below_{name}"
        result_name = f"below_{name}"

        exprs = adapter._get_stat_function_from_name(stat_name)(
            "value", below_col="limit"
        )
        result = df.select(exprs)

        assert result_name in result.columns

    def test_stat_count_below_for_grouped_aggregation(self, setup_adapter, pl):
        """Test count-below statistics inside grouped lazy aggregation."""
        df = pl.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
                "limit": [2.0, 3.0, 3.0, 6.0, 6.0, 6.0],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=df.lazy(),
            stratifications=[["group"]],
            value_col="value",
            stat_builders=["stat_count_below"],
            below_col="limit",
        ).sort("group")

        assert result["below_n"].to_list() == [2, 1]
        assert result["below_pct"].to_list() == pytest.approx([2 / 3, 1 / 3])


@pytest.mark.dataframe
class TestStatArithmetic:
    """Test suite for stat_arithmetic function."""

    def test_stat_arithmetic_basic(
        self, dataframe_with_nulls, setup_adapter, pl
    ):
        """Test basic arithmetic statistics."""
        import scipy.stats as stats

        adapter = setup_adapter()

        df = dataframe_with_nulls

        exprs = adapter._get_stat_function_from_name("stat_arithmetic")(
            "value"
        )
        result = df.select(exprs)

        assert result.shape == (1, 5)
        assert result["mean"][0] == pytest.approx(
            stats.tmean(df["value"].drop_nulls().to_list())
        )
        # Standard deviation of 1-10 is approximately 3.028
        assert result["st"][0] == pytest.approx(
            stats.tstd(df["value"].drop_nulls().to_list())
        )
        # SEM = std / sqrt(n)
        assert result["sem"][0] == pytest.approx(
            stats.sem(df["value"].drop_nulls().to_list())
        )

    def test_stat_arithmetic_confidence_intervals(
        self, sample_dataframe, setup_adapter, pl
    ):
        """Test that confidence intervals are correctly calculated."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_arithmetic")(
            "value"
        )
        result = sample_dataframe.select(exprs)

        mean = result["mean"][0]
        sem = result["sem"][0]

        # 95% CI: mean ± 1.96 * SEM
        expected_ci_lower = mean - 1.96 * sem
        expected_ci_upper = mean + 1.96 * sem

        assert result["mean_95_ci_lower"][0] == pytest.approx(
            expected_ci_lower
        )
        assert result["mean_95_ci_upper"][0] == pytest.approx(
            expected_ci_upper
        )

    def test_stat_arithmetic_single_value(self, setup_adapter, pl):
        """Test arithmetic statistics with a single value."""

        df = pl.DataFrame({"value": [5.0]})
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_arithmetic")(
            "value"
        )
        result = df.select(exprs)

        assert result["mean"][0] == 5.0
        assert result["st"][0] is None or math.isnan(result["st"][0])
        assert result["sem"][0] is None or math.isnan(result["sem"][0])

    def test_stat_arithmetic_with_nulls(
        self, dataframe_with_nulls, setup_adapter, pl
    ):
        """Test arithmetic statistics ignore null values."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_arithmetic")(
            "value"
        )
        result = dataframe_with_nulls.select(exprs)

        # Mean of [1, 2, 4, 5, 7, 8, 9, 10] = 46/8 = 5.75
        assert result["mean"][0] == pytest.approx(5.75)
        assert result["st"][0] > 0

    def test_stat_arithmetic_cutoff_sufficient_values(self, setup_adapter, pl):
        """Test arithmetic statistics are returned when cutoff is met."""
        df = pl.DataFrame({"value": [1.0, 2.0, None, None]})
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("stat_arithmetic")(
            "value", cutoff=0.5
        )
        result = df.select(exprs)

        assert result["mean"][0] == pytest.approx(1.5)
        assert result["st"][0] is not None
        assert result["sem"][0] is not None
        assert result["mean_95_ci_lower"][0] is not None
        assert result["mean_95_ci_upper"][0] is not None

    def test_stat_arithmetic_cutoff_insufficient_values(
        self, setup_adapter, pl
    ):
        """Test arithmetic statistics return null when cutoff is not met."""
        df = pl.DataFrame({"value": [1.0, None, None, None]})
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("stat_arithmetic")(
            "value", cutoff=0.5
        )
        result = df.select(exprs)

        assert result.row(0) == (None, None, None, None, None)

    def test_stat_arithmetic_without_cutoff_allows_sparse_values(
        self, setup_adapter, pl
    ):
        """Test current behavior is preserved when cutoff is not provided."""
        df = pl.DataFrame({"value": [1.0, None, None, None]})
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("stat_arithmetic")(
            "value"
        )
        result = df.select(exprs)

        assert result["mean"][0] == pytest.approx(1.0)

    def test_stat_arithmetic_cutoff_for_grouped_aggregation(
        self, setup_adapter, pl
    ):
        """Test cutoff is evaluated per group."""
        df = pl.DataFrame(
            {
                "value": [1.0, 2.0, None, 4.0, None, None],
                "group": ["A", "A", "A", "B", "B", "B"],
            }
        )
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=df.lazy(),
            stratifications=[["group"]],
            value_col="value",
            stat_builders=["stat_arithmetic"],
            cutoff=0.5,
        ).sort("group")

        assert result["mean"].to_list() == pytest.approx([1.5, None])

    def test_stat_arithmetic_invalid_cutoff(self, setup_adapter):
        """Test cutoff must be between zero and one."""
        adapter = setup_adapter()

        with pytest.raises(ValueError, match="cutoff must be between 0 and 1"):
            adapter._get_stat_function_from_name("stat_arithmetic")(
                "value", cutoff=1.1
            )

    @pytest.mark.parametrize(
        "values,expected_mean",
        [
            ([1.0, 2.0, 3.0], 2.0),
            ([0.0, 0.0, 0.0], 0.0),
            ([10.0, 20.0, 30.0], 20.0),
            ([-5.0, 0.0, 5.0], 0.0),
        ],
    )
    def test_stat_arithmetic_parametrized(
        self, values, expected_mean, setup_adapter, pl
    ):
        """Test arithmetic statistics with various datasets."""

        df = pl.DataFrame({"value": values})
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_arithmetic")(
            "value"
        )
        result = df.select(exprs)

        assert result["mean"][0] == pytest.approx(expected_mean)

    @pytest.mark.parametrize(
        "name", ["mean", "st", "sem", "mean_95_ci_lower", "mean_95_ci_upper"]
    )
    def test_statistics_arithmetic_unit_fn(
        self, name, sample_dataframe, setup_adapter, pl
    ):
        """Test that arithmetic statistics can be called with unit function."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name(f"statistics_{name}")(
            "value"
        )
        result = sample_dataframe.select(exprs)

        assert name in result.columns

    def test_statistics_arithmetic_unit_fn_with_cutoff(
        self, setup_adapter, pl
    ):
        """Test arithmetic unit functions accept cutoff."""
        df = pl.DataFrame({"value": [1.0, None, None, None]})
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("statistics_mean")(
            "value", cutoff=0.5
        )
        result = df.select(exprs)

        assert result["mean"][0] is None


@pytest.mark.dataframe
class TestStatGeometric:
    """Test suite for stat_geometric function."""

    def test_stat_geometric_basic(self, sample_dataframe, setup_adapter, pl):
        """Test basic geometric mean calculation."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_geometric")("value")
        result = sample_dataframe.select(exprs)

        assert result.shape == (1, 3)
        # Geometric mean of 1-10 is approximately 4.529
        assert result["geom_mean"][0] == pytest.approx(4.528728688116765)
        assert result["geom_mean_95_ci_lower"][0] > 0
        assert result["geom_mean_95_ci_upper"][0] > result["geom_mean"][0]

    def test_stat_geometric_confidence_intervals(self, setup_adapter, pl):
        """Test that geometric mean confidence intervals are properly calculated."""

        df = pl.DataFrame({"value": [2.0, 4.0, 8.0, 16.0]})
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_geometric")("value")
        result = df.select(exprs)

        # Geometric mean of [2, 4, 8, 16] = (2*4*8*16)^(1/4) = 5.657
        geom_mean = result["geom_mean"][0]
        ci_lower = result["geom_mean_95_ci_lower"][0]
        ci_upper = result["geom_mean_95_ci_upper"][0]

        assert ci_lower < geom_mean
        assert ci_upper > geom_mean
        assert ci_lower > 0

    def test_stat_geometric_with_nulls(
        self, dataframe_with_nulls, setup_adapter, pl
    ):
        """Test geometric mean with null values."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_geometric")("value")
        result = dataframe_with_nulls.select(exprs)

        assert result["geom_mean"][0] > 0
        assert result["geom_mean_95_ci_lower"][0] > 0

    def test_stat_geometric_cutoff_sufficient_values(self, setup_adapter, pl):
        """Test geometric statistics are returned when cutoff is met."""
        df = pl.DataFrame({"value": [2.0, 8.0, None, None]})
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("stat_geometric")(
            "value", cutoff=0.5
        )
        result = df.select(exprs)

        assert result["geom_mean"][0] == pytest.approx(4.0)
        assert result["geom_mean_95_ci_lower"][0] is not None
        assert result["geom_mean_95_ci_upper"][0] is not None

    def test_stat_geometric_cutoff_insufficient_values(
        self, setup_adapter, pl
    ):
        """Test geometric statistics return null when cutoff is not met."""
        df = pl.DataFrame({"value": [2.0, 8.0, None, None]})
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("stat_geometric")(
            "value", cutoff=0.75
        )
        result = df.select(exprs)

        assert result.row(0) == (None, None, None)

    def test_statistics_geometric_unit_fn_with_cutoff(self, setup_adapter, pl):
        """Test geometric unit functions accept cutoff."""
        df = pl.DataFrame({"value": [2.0, 8.0, None, None]})
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("statistics_geom_mean")(
            "value", cutoff=0.75
        )
        result = df.select(exprs)

        assert result["geom_mean"][0] is None

    @pytest.mark.parametrize(
        "values,expected_geom_mean",
        [
            ([1.0, 1.0, 1.0], 1.0),
            ([2.0, 8.0], 4.0),
            ([4.0, 9.0], 6.0),
        ],
    )
    def test_stat_geometric_parametrized(
        self, values, expected_geom_mean, setup_adapter, pl
    ):
        """Test geometric mean with various datasets."""

        df = pl.DataFrame({"value": values})
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_geometric")("value")
        result = df.select(exprs)

        assert result["geom_mean"][0] == pytest.approx(expected_geom_mean)

    @pytest.mark.parametrize(
        "name", ["geom_mean", "geom_mean_95_ci_lower", "geom_mean_95_ci_upper"]
    )
    def test_statistics_geometric_unit_fn(
        self, name, sample_dataframe, setup_adapter, pl
    ):
        """Test that geometric mean can be called with unit function."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name(f"statistics_{name}")(
            "value"
        )
        result = sample_dataframe.select(exprs)

        assert name in result.columns


@pytest.mark.dataframe
class TestStatPercentiles:
    """Test suite for stat_percentiles function."""

    def test_stat_percentiles_default_quantiles(
        self, sample_dataframe, setup_adapter, pl
    ):
        """Test percentile calculation with default quantiles."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_percentiles")(
            "value"
        )
        result = sample_dataframe.select(exprs)
        # Should return 7 percentiles + 7 CI lower + 7 CI upper = 21 columns
        assert result.shape == (1, 21)

    def test_stat_percentiles_custom_quantiles(
        self, sample_dataframe, setup_adapter, pl
    ):
        """Test percentile calculation with custom quantiles."""
        custom_quants = [0.25, 0.5, 0.75]
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_percentiles")(
            "value", quants=custom_quants
        )
        result = sample_dataframe.select(exprs)

        # Should return 3 percentiles + 3 CI lower + 3 CI upper = 9 columns
        assert result.shape == (1, 9)

        # Check median (50th percentile) - polars might use different interpolation
        # Median should be between 5th and 6th value (5 and 6)
        assert 5.0 <= result["p50"][0] <= 6.0

    def test_stat_percentiles_values(
        self, sample_dataframe, setup_adapter, pl
    ):
        """Test that percentile values are correctly calculated."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_percentiles")(
            "value", quants=[0.25, 0.5, 0.75]
        )
        result = sample_dataframe.select(exprs)

        # For data [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert result["p25"][0] == pytest.approx(3)
        assert result["p50"][0] == pytest.approx(6)
        assert result["p75"][0] == pytest.approx(8)

    def test_stat_percentiles_confidence_intervals_exist(
        self, sample_dataframe, setup_adapter, pl
    ):
        """Test that confidence interval columns are created."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_percentiles")(
            "value", quants=[0.5]
        )
        result = sample_dataframe.select(exprs)

        assert "p50" in result.columns
        assert "p50_ci_lower" in result.columns
        assert "p50_ci_upper" in result.columns

    def test_stat_percentiles_small_sample(
        self, small_dataframe, setup_adapter, pl
    ):
        """Test percentiles with small sample size."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_percentiles")(
            "value", quants=[0.5]
        )
        result = small_dataframe.select(exprs)

        # With only 3 values, median should be middle value
        assert result["p50"][0] == pytest.approx(2.0)

    @pytest.mark.parametrize("quantile", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_stat_percentiles_boundary_quantiles(
        self, sample_dataframe, quantile, setup_adapter, pl
    ):
        """Test percentiles at various boundary quantiles."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name("stat_percentiles")(
            "value", quants=[quantile]
        )
        result = sample_dataframe.select(exprs)

        col_name = f"p{int(quantile * 100)}"
        assert col_name in result.columns
        assert result[col_name][0] is not None

    @pytest.mark.parametrize(
        "name",
        [
            "p5",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
            "p5_ci_lower",
            "p5_ci_upper",
            "p10_ci_lower",
            "p10_ci_upper",
            "p25_ci_lower",
            "p25_ci_upper",
            "p50_ci_lower",
            "p50_ci_upper",
            "p75_ci_lower",
            "p75_ci_upper",
            "p90_ci_lower",
            "p90_ci_upper",
            "p95_ci_lower",
            "p95_ci_upper",
        ],
    )
    def test_stat_percentiles_unit_fn(
        self, sample_dataframe, setup_adapter, pl, name
    ):
        """Test that percentile function can be called with unit function."""
        adapter = setup_adapter()
        exprs = adapter._get_stat_function_from_name(
            f"statistics_percentiles_{name}"
        )("value")
        result = sample_dataframe.select(exprs)

        assert name in result.columns


@pytest.mark.dataframe
class TestPercentileCIHelpers:
    """Test suite for _percentile_ci_lower and _percentile_ci_upper helper functions."""

    def setup_formulas(self):
        from pypeh.adapters.aggregation.polars_adapter.statistics import (
            _percentile_ci_lower,
            _percentile_ci_upper,
        )

        return _percentile_ci_lower, _percentile_ci_upper

    def test_percentile_ci_lower_basic(self, sample_dataframe):
        """Test lower confidence interval calculation."""
        import polars as pl

        expr = self.setup_formulas()[0]("value", 0.5)

        result = sample_dataframe.select(expr)

        n = pl.col("value").count()
        se = (n * 0.5 * (1 - 0.5)).sqrt()
        ci_lower = (n * 0.5 - 1.96 * se).ceil()
        expected = sample_dataframe.select(
            pl.col("value").quantile((ci_lower / n)).alias("p50_ci_lower")
        )

        # Should return a value or null
        assert result.shape == (1, 1)
        assert result.equals(expected)

    def test_percentile_ci_upper_basic(self, sample_dataframe):
        """Test upper confidence interval calculation."""
        import polars as pl

        expr = self.setup_formulas()[1]("value", 0.5)
        result = sample_dataframe.select(expr)

        n = pl.col("value").count()
        se = (n * 0.5 * (1 - 0.5)).sqrt()
        ci_lower = (n * 0.5 + 1.96 * se).ceil()
        expected = sample_dataframe.select(
            pl.col("value").quantile((ci_lower / n)).alias("p50_ci_upper")
        )

        # Should return a value or null
        assert result.shape == (1, 1)
        assert result.equals(expected)

    def test_percentile_ci_lower_min_handling(self, small_dataframe):
        """Test that CI lower returns min when calculated index is negative."""
        # With small sample, some quantiles might have ci_lower < 0
        expr = self.setup_formulas()[0]("value", 0.05)
        result = small_dataframe.select(expr)

        assert result.shape == (1, 1)
        assert result["p5_ci_lower"][0] == 1

    def test_percentile_ci_upper_null_handling(self):
        """Test that CI upper returns null when calculated index exceeds n."""

        import polars as pl

        df = pl.DataFrame({"value": [1.0, 2.0]})
        expr = self.setup_formulas()[1]("value", 0.95)
        result = df.select(expr)

        # Should handle null appropriately
        assert result.shape == (1, 1)

    @pytest.mark.parametrize("quantile", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_percentile_ci_naming_convention(self, sample_dataframe, quantile):
        """Test that CI columns follow correct naming convention."""
        _percentile_ci_lower, _percentile_ci_upper = self.setup_formulas()
        expr_lower = _percentile_ci_lower("value", quantile)
        expr_upper = _percentile_ci_upper("value", quantile)

        result = sample_dataframe.select([expr_lower, expr_upper])

        expected_name = f"p{int(quantile * 100)}"
        assert f"{expected_name}_ci_lower" in result.columns
        assert f"{expected_name}_ci_upper" in result.columns


@pytest.mark.dataframe
class TestIntegration:
    """Integration tests combining multiple statistics functions."""

    def test_all_stats_together(self, sample_dataframe, setup_adapter, pl):
        """Test that all statistics can be computed together."""
        adapter = setup_adapter()

        count_exprs = adapter._get_stat_function_from_name("stat_count")(
            "value"
        )
        arith_exprs = adapter._get_stat_function_from_name("stat_arithmetic")(
            "value"
        )
        geom_exprs = adapter._get_stat_function_from_name("stat_geometric")(
            "value"
        )
        perc_exprs = adapter._get_stat_function_from_name("stat_percentiles")(
            "value", quants=[0.25, 0.5, 0.75]
        )

        all_exprs = count_exprs + arith_exprs + geom_exprs + perc_exprs
        result = sample_dataframe.select(all_exprs)

        # Should have all columns
        assert "n" in result.columns
        assert "mean" in result.columns
        assert "geom_mean" in result.columns
        assert "p50" in result.columns


@pytest.mark.dataframe
class TestFrequencyTable:
    """Test suite for frequency table generation."""

    def test_frequency_table_basic(
        self, dataframe_with_nulls, setup_adapter, pl
    ):
        """Test basic frequency table generation."""
        adapter = setup_adapter()

        exprs = adapter._get_stat_function_from_name("frequency_table")(
            ["value"]
        )

        result = exprs(dataframe_with_nulls)

        assert result.shape == (9, 2)
        assert "value" in result.columns
        assert "frequency" in result.columns
        assert (
            result.filter(pl.col("value").is_null())
            .select(pl.col("frequency"))
            .item()
            == 2
        )
