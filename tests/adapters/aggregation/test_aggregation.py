import pytest

from pypeh.core.interfaces.dataops import AggregationInterface


@pytest.fixture(scope="module")
def setup_adapter():
    """Get the default aggregation adapter class."""
    return AggregationInterface.get_default_adapter_class()


@pytest.fixture
def pl():
    """Import and return polars module."""
    import polars as pl

    return pl


@pytest.fixture
def sample_dataframe(pl):
    """Create a sample dataframe for testing aggregation."""
    return pl.DataFrame(
        {
            "measurement": [
                10.0,
                12.0,
                11.0,
                15.0,
                14.0,
                20.0,
                22.0,
                21.0,
                25.0,
                24.0,
            ],
            "group_a": ["X", "X", "X", "X", "X", "Y", "Y", "Y", "Y", "Y"],
            "group_b": ["A", "A", "B", "B", "B", "A", "A", "B", "B", "B"],
            "category": [
                "cat1",
                "cat1",
                "cat2",
                "cat2",
                "cat1",
                "cat1",
                "cat2",
                "cat2",
                "cat1",
                "cat2",
            ],
        }
    )


@pytest.fixture
def dataframe_with_nulls(pl):
    """Create a dataframe with null values for testing."""
    return pl.DataFrame(
        {
            "measurement": [
                10.0,
                None,
                11.0,
                15.0,
                None,
                20.0,
                22.0,
                21.0,
                None,
                24.0,
            ],
            "group_a": ["X", "X", "X", "X", "X", "Y", "Y", "Y", "Y", "Y"],
            "group_b": ["A", "A", "B", "B", "B", "A", "A", "B", "B", "B"],
        }
    )


@pytest.mark.dataframe
class TestDataFrameAggregationAdapter:
    """Test suite for DataFrameAggregationAdapter class."""

    def test_adapter_instantiation(self, setup_adapter):
        """Test that the adapter can be instantiated."""
        adapter = setup_adapter()
        assert adapter is not None
        assert hasattr(adapter, "calculate_for_strata")
        assert hasattr(adapter, "_calculate_for_stratum")
        assert hasattr(adapter, "_get_stat_function_from_name")

    def test_get_stat_function_from_name(self, setup_adapter):
        """Test retrieving stat functions by name."""
        adapter = setup_adapter()

        # Test that known functions can be retrieved
        stat_count = adapter._get_stat_function_from_name("stat_count")
        assert callable(stat_count)

        stat_arithmetic = adapter._get_stat_function_from_name(
            "stat_arithmetic"
        )
        assert callable(stat_arithmetic)

        stat_geometric = adapter._get_stat_function_from_name("stat_geometric")
        assert callable(stat_geometric)

        stat_percentiles = adapter._get_stat_function_from_name(
            "stat_percentiles"
        )
        assert callable(stat_percentiles)

    def test_get_stat_function_invalid_name(self, setup_adapter):
        """Test that invalid function names raise AttributeError."""
        adapter = setup_adapter()

        with pytest.raises(AttributeError):
            adapter._get_stat_function_from_name("invalid_function_name")


@pytest.mark.dataframe
class TestSummarizeMethod:
    """Test suite for the calculate_for_strata method."""

    def test_summarize_single_stratification(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test calculate_for_strata with a single stratification."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[["group_a"]],
            value_col="measurement",
            stat_builders=["stat_count", "stat_arithmetic"],
        )

        # Should have 2 groups (X and Y)
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 2
        assert "group_a" in result.columns
        assert "n" in result.columns
        assert "mean" in result.columns

    def test_summarize_multiple_stratifications(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test calculate_for_strata with multiple stratifications."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[["group_a"], ["group_b"], ["group_a", "group_b"]],
            value_col="measurement",
            stat_builders=["stat_count"],
        )

        # Should have results from all three stratifications
        # group_a: 2 groups, group_b: 2 groups, group_a+group_b: 4 groups = 8 total
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 8
        assert "n" in result.columns

        # Check that all group columns exist (some will be null for certain stratifications)
        assert "group_a" in result.columns
        assert "group_b" in result.columns

    def test_summarize_with_nulls(
        self, setup_adapter, dataframe_with_nulls, pl
    ):
        """Test calculate_for_strata handles null values correctly."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=dataframe_with_nulls.lazy(),
            stratifications=[["group_a"]],
            value_col="measurement",
            stat_builders=["stat_count"],
        )

        assert result.shape[0] == 2
        # Check that missing values are counted
        assert "missing_n" in result.columns
        total_missing = result["missing_n"].sum()
        assert total_missing == 3  # We have 3 null values

    def test_summarize_multiple_stat_builders(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test calculate_for_strata with multiple stat builders."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[["group_a"]],
            value_col="measurement",
            stat_builders=["stat_count", "stat_arithmetic", "stat_geometric"],
        )

        # Should have columns from all stat builders
        assert "n" in result.columns  # from stat_count
        assert "mean" in result.columns  # from stat_arithmetic
        assert "geom_mean" in result.columns  # from stat_geometric
        assert result.shape[0] == 2

    def test_summarize_ignores_unsupported_kwargs(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test kwargs are only passed to stat builders that accept them."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[["group_a"]],
            value_col="measurement",
            stat_builders=["stat_count", "stat_arithmetic"],
            cutoff=0.75,
        )

        assert "n" in result.columns
        assert "mean" in result.columns
        assert result.shape[0] == 2

    def test_summarize_with_percentiles(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test calculate_for_strata with percentile statistics."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[["group_a"]],
            value_col="measurement",
            stat_builders=["stat_percentiles"],
            quants=[0.25, 0.5, 0.75],
        )

        # Should have percentile columns
        assert "p25" in result.columns
        assert "p50" in result.columns
        assert "p75" in result.columns
        assert result.shape[0] == 2

    def test_summarize_empty_stratifications(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test calculate_for_strata with empty stratifications list (behaves like None)."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[],
            value_col="measurement",
            stat_builders=["stat_count"],
        )

        # Empty list behaves like None - returns overall statistics
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 1
        assert "n" in result.columns
        assert result["n"][0] == 10

    def test_summarize_none_stratifications(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test calculate_for_strata with stratifications=None (no grouping)."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=None,
            value_col="measurement",
            stat_builders=["stat_count", "stat_arithmetic"],
        )

        # Should return a single row with overall statistics
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 1
        assert "n" in result.columns
        assert "mean" in result.columns
        # Should have overall stats for all 10 rows
        assert result["n"][0] == 10
        assert result["mean"][0] == pytest.approx(
            17.4
        )  # mean of all measurements

    def test_summarize_nested_stratification(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test calculate_for_strata with nested/combined stratification."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[["group_a", "group_b"]],
            value_col="measurement",
            stat_builders=["stat_count", "stat_arithmetic"],
        )

        # Should have 4 combinations (X-A, X-B, Y-A, Y-B)
        assert result.shape[0] == 4
        assert "group_a" in result.columns
        assert "group_b" in result.columns

        # Verify all combinations exist
        combinations = result.select(["group_a", "group_b"]).unique()
        assert combinations.shape[0] == 4


@pytest.mark.dataframe
class TestInternalSummarizeMethod:
    """Test suite for the _calculate_for_stratum internal method."""

    def test_internal_summarize_basic(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test _calculate_for_stratum with basic parameters."""
        adapter = setup_adapter()

        result = adapter._calculate_for_stratum(
            df=sample_dataframe.lazy(),
            group_cols=["group_a"],
            value_col="measurement",
            stat_builders=["stat_count"],
        )

        # Result should be LazyFrame
        assert hasattr(result, "collect")
        collected = result.collect()

        assert collected.shape[0] == 2
        assert "group_a" in collected.columns
        assert "n" in collected.columns

    def test_internal_summarize_multiple_groups(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test _calculate_for_stratum with multiple grouping columns."""
        adapter = setup_adapter()

        result = adapter._calculate_for_stratum(
            df=sample_dataframe.lazy(),
            group_cols=["group_a", "group_b"],
            value_col="measurement",
            stat_builders=["stat_arithmetic"],
        ).collect()

        assert result.shape[0] == 4  # 2x2 combinations
        assert "group_a" in result.columns
        assert "group_b" in result.columns
        assert "mean" in result.columns
        assert "st" in result.columns

    def test_internal_summarize_stratification_column(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test that _calculate_for_stratum adds stratification column correctly."""
        adapter = setup_adapter()

        result = adapter._calculate_for_stratum(
            df=sample_dataframe.lazy(),
            group_cols=["group_a", "category"],
            value_col="measurement",
            stat_builders=["stat_count"],
        ).collect()

        # All rows should have the same stratification value
        # The stratification should contain the group column names
        assert {"group_a", "category"}.issubset(set(result.columns))

    def test_internal_summarize_with_kwargs(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test _calculate_for_stratum passes kwargs to stat functions."""
        adapter = setup_adapter()

        result = adapter._calculate_for_stratum(
            df=sample_dataframe.lazy(),
            group_cols=["group_a"],
            value_col="measurement",
            stat_builders=["stat_percentiles"],
            quants=[0.5, 0.9],
        ).collect()

        # Should only have p50 and p90 (and their CIs)
        assert "p50" in result.columns
        assert "p90" in result.columns
        # But not p25 (which is in default quantiles)
        assert "p25" not in result.columns


@pytest.mark.dataframe
class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_full_workflow_single_stratification(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test complete workflow with single stratification."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[["group_a"]],
            value_col="measurement",
            stat_builders=["stat_count", "stat_arithmetic"],
        )

        # Verify structure
        assert result.shape[0] == 2
        assert "group_a" in result.columns

        # Verify statistics are calculated
        group_x = result.filter(pl.col("group_a") == "X")
        assert group_x["n"][0] == 5
        assert group_x["mean"][0] > 0

    def test_full_workflow_complex_stratifications(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test complete workflow with complex stratifications."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[
                ["group_a"],
                ["group_b"],
                ["category"],
                ["group_a", "group_b"],
            ],
            value_col="measurement",
            stat_builders=["stat_count", "stat_arithmetic"],
        )

        # Should have: 2 (group_a) + 2 (group_b) + 2 (category) + 4 (group_a x group_b) = 10
        assert result.shape[0] == 10

    def test_comparison_across_groups(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test that results make sense when comparing across groups."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[["group_a"]],
            value_col="measurement",
            stat_builders=["stat_count", "stat_arithmetic"],
        )

        group_x = result.filter(pl.col("group_a") == "X")
        group_y = result.filter(pl.col("group_a") == "Y")

        # Both groups should have equal counts
        assert group_x["n"][0] == 5
        assert group_y["n"][0] == 5

        # Group Y has higher values (20-25) than group X (10-15)
        assert group_y["mean"][0] > group_x["mean"][0]

    def test_all_statistics_together(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test using all available statistics functions together."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[["group_a"]],
            value_col="measurement",
            stat_builders=["stat_count", "stat_arithmetic", "stat_geometric"],
        )

        # Check all statistic types are present
        expected_columns = [
            "n",
            "missing_n",
            "missing_pct",  # stat_count
            "mean",
            "st",
            "sem",
            "mean_95_ci_lower",
            "mean_95_ci_upper",  # stat_arithmetic
            "geom_mean",
            "geom_mean_95_ci_lower",
            "geom_mean_95_ci_upper",  # stat_geometric
        ]

        for col in expected_columns:
            assert col in result.columns

    def test_edge_case_single_value_per_group(self, setup_adapter, pl):
        """Test with only one value per group."""
        df = pl.DataFrame(
            {
                "value": [10.0, 20.0],
                "group": ["A", "B"],
            }
        )

        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=df.lazy(),
            stratifications=[["group"]],
            value_col="value",
            stat_builders=["stat_count", "stat_arithmetic"],
        )

        # Should still work, even though std will be NaN
        assert result.shape[0] == 2
        assert result["n"][0] == 1
        assert result["n"][1] == 1

    def test_three_way_stratification(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test three-way stratification."""
        adapter = setup_adapter()

        result = adapter.calculate_for_strata(
            df=sample_dataframe.lazy(),
            stratifications=[["group_a", "group_b", "category"]],
            value_col="measurement",
            stat_builders=["stat_count", "stat_arithmetic", "stat_geometric"],
        )

        # Should have multiple three-way combinations
        assert result.shape[0] > 0
        assert "group_a" in result.columns
        assert "group_b" in result.columns
        assert "category" in result.columns

        # Each row should have a combination of all three grouping variables
        assert all(result["n"] > 0)


@pytest.mark.dataframe
class TestCalculateFrequencies:
    """Test suite for the calculate_frequencies method."""

    def test_calculate_frequencies_basic(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test calculate_frequencies with basic parameters."""
        adapter = setup_adapter()

        result = adapter._calculate_frequency(
            df=sample_dataframe.lazy(),
            group_cols=None,
            value_col="group_a",
        )

        assert result.shape[0] == 2
        assert "value" in result.columns
        assert "frequency" in result.columns
        assert (
            result.filter(pl.col("value").eq("X"))
            .select(pl.col("frequency"))
            .item()
            == 5
        )
        assert (
            result.filter(pl.col("value").eq("Y"))
            .select(pl.col("frequency"))
            .item()
            == 5
        )

    def test_calculate_frequencies_group_col(
        self, setup_adapter, sample_dataframe, pl
    ):
        """Test calculate_frequencies with basic parameters."""
        adapter = setup_adapter()

        result = adapter._calculate_frequency(
            df=sample_dataframe.lazy(),
            group_cols=["group_a", "group_b"],
            value_col="category",
            result_aliases=["value", "frequency"],
        )

        assert (
            result.shape[0] == 7
        )  # There are 7 unique combinations of group_a, group_b, and category in the sample dataframe
        assert result.shape[1] == 4  # group_a, group_b, value, frequency
        assert "value" in result.columns
        assert "frequency" in result.columns
        assert (
            result.filter(
                pl.col("group_a").eq("X")
                & pl.col("group_b").eq("A")
                & pl.col("value").eq("cat1")
            )
            .select(pl.col("frequency"))
            .item()
            == 2
        )
        assert (
            result.filter(
                pl.col("group_a").eq("X")
                & pl.col("group_b").eq("B")
                & pl.col("value").eq("cat1")
            )
            .select(pl.col("frequency"))
            .item()
            == 1
        )
