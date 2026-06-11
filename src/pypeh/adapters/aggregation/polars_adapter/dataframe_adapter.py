import inspect
from functools import reduce
from itertools import chain
from typing import Callable

import polars as pl

from pypeh.core.interfaces.dataops import AggregationInterface
from pypeh.adapters.dataops.dataframe_adapter import DataFrameAdapter
import pypeh.adapters.aggregation.polars_adapter.statistics as stats


class DataFrameAggregationAdapter(
    DataFrameAdapter, AggregationInterface[pl.DataFrame]
):
    data_format = pl.DataFrame

    def calculate_for_strata(
        self,
        df: pl.LazyFrame,
        stratifications: list[list[str]] | None,
        value_col: str,
        stat_builders: list[str],
        **kwargs,
    ) -> pl.DataFrame:
        if not stratifications:
            return self._calculate_for_stratum(
                df=df,
                group_cols=None,
                value_col=value_col,
                stat_builders=stat_builders,
                **kwargs,
            ).collect()

        summary_dfs = []
        for strat in stratifications:
            summary_df = self._calculate_for_stratum(
                df=df,
                group_cols=strat,
                value_col=value_col,
                stat_builders=stat_builders,
                **kwargs,
            )
            summary_dfs.append(summary_df)

        combined_summary = pl.concat(summary_dfs, how="diagonal").collect()
        return combined_summary

    def _calculate_for_stratum(
        self,
        df: pl.LazyFrame,
        group_cols: list[str] | None,
        value_col: str,
        stat_builders: list[str],
        result_aliases: list[str] | None = None,
        stat_kwargs: list[dict] | None = None,
        **kwargs,
    ) -> pl.LazyFrame:
        if stat_kwargs is None:
            stat_kwargs = [{} for _ in stat_builders]
        if len(stat_kwargs) != len(stat_builders):
            raise ValueError(
                "stat_kwargs must contain one kwargs mapping per stat builder."
            )

        result_alias_iter = (
            result_aliases
            if result_aliases is not None
            else [None for _ in stat_builders]
        )
        if len(result_alias_iter) != len(stat_builders):
            raise ValueError(
                "result_aliases must contain one alias per stat builder."
            )

        exprs = list(
            chain.from_iterable(
                self._call_stat_function(
                    stat_builder,
                    value_col,
                    result_alias=result_alias,
                    stat_kwargs={**kwargs, **per_stat_kwargs},
                )
                for stat_builder, result_alias, per_stat_kwargs in zip(
                    stat_builders,
                    result_alias_iter,
                    stat_kwargs,
                )
            )
        )

        if not group_cols:
            return df.select(exprs)

        return df.group_by(group_cols).agg(exprs)

    def _get_stat_function_from_name(self, function_name: str):
        return getattr(stats, function_name)

    def _get_stat_function(self, fn: str | Callable):
        if isinstance(fn, str):
            return self._get_stat_function_from_name(fn)
        elif callable(fn):
            return fn
        else:
            raise ValueError(f"Invalid function specification: {fn}")

    def _filter_stat_kwargs(self, fn: Callable, kwargs: dict) -> dict:
        signature = inspect.signature(fn)
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ):
            return kwargs

        accepted_kwargs = {
            name
            for name, param in signature.parameters.items()
            if param.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        return {
            name: value
            for name, value in kwargs.items()
            if name in accepted_kwargs
        }

    def _call_stat_function(
        self,
        stat_builder: str | Callable,
        value_col: str,
        result_alias: str | None,
        stat_kwargs: dict,
    ) -> list[pl.Expr]:
        fn = self._get_stat_function(stat_builder)
        if result_alias is not None:
            stat_kwargs = {
                **stat_kwargs,
                "result_alias": result_alias,
            }

        return fn(value_col, **self._filter_stat_kwargs(fn, stat_kwargs))

    def group_results(
        self,
        results_to_collect: list[pl.LazyFrame],
        strata: list[str] | None = None,
    ) -> pl.DataFrame:
        if strata is None:
            ret = pl.concat(results_to_collect, how="horizontal")
        else:
            ret = reduce(
                lambda left, right: left.join(right, on=strata, how="inner"),
                results_to_collect,
            )
        return ret.collect()

    def _calculate_frequency(
        self,
        df: pl.LazyFrame,
        group_cols: list[str] | None,
        value_col: str,
        result_aliases: list[str] = ["value", "frequency"],
    ) -> pl.LazyFrame:
        if group_cols:
            cols = group_cols + [value_col]
        else:
            cols = [value_col]

        fn = self._get_stat_function_from_name("frequency_table")(
            cols, result_aliases=result_aliases
        )
        return fn(df).collect()
