from typing import Callable

import polars as pl


def _validate_cutoff(cutoff: float | None) -> None:
    if cutoff is None:
        return
    if not 0 <= cutoff <= 1:
        raise ValueError("cutoff must be between 0 and 1")


def _sufficient_values(value_col: str, cutoff: float) -> pl.Expr:
    non_null_n = pl.col(value_col).is_not_null().sum()
    return (pl.len() > 0) & ((non_null_n / pl.len()) >= cutoff)


def _alias_with_cutoff(
    value_col: str,
    expr: pl.Expr,
    result_alias: str,
    cutoff: float | None,
) -> pl.Expr:
    _validate_cutoff(cutoff)
    if cutoff is None:
        return expr.alias(result_alias)

    return (
        pl.when(_sufficient_values(value_col, cutoff))
        .then(expr)
        .otherwise(None)
        .alias(result_alias)
    )


def stat_count(
    value_col: str,
    *,
    result_aliases: list[str] = ["n", "missing_n", "missing_pct"],
) -> list[pl.Expr]:
    null_count = pl.col(value_col).null_count()
    return [
        pl.len().alias(result_aliases[0]),
        null_count.alias(result_aliases[1]),
        (null_count / pl.len()).alias(result_aliases[2]),
    ]


def statistics_count_n(
    value_cont: str, result_alias: str = "n"
) -> list[pl.Expr]:
    return [
        stat_count(
            value_cont,
            result_aliases=[result_alias, "missing_n", "missing_pct"],
        )[0].alias(result_alias)
    ]


def statistics_count_missing_n(
    value_cont: str, result_alias: str = "missing_n"
) -> list[pl.Expr]:
    return [
        stat_count(
            value_cont, result_aliases=["n", result_alias, "missing_pct"]
        )[1]
    ]


def statistics_count_missing_pct(
    value_cont: str, result_alias: str = "missing_pct"
) -> list[pl.Expr]:
    return [
        stat_count(
            value_cont, result_aliases=["n", "missing_n", result_alias]
        )[2].alias(result_alias)
    ]


def stat_count_below(
    value_col: str,
    below_col: str,
    *,
    result_aliases: list[str] = ["below_n", "below_pct"],
) -> list[pl.Expr]:
    below_n = (pl.col(value_col) < pl.col(below_col)).sum()
    return [
        below_n.alias(result_aliases[0]),
        (below_n / pl.len()).alias(result_aliases[1]),
    ]


def statistics_count_below_n(
    value_cont: str,
    below_col: str,
    result_alias: str = "below_n",
) -> list[pl.Expr]:
    return [
        stat_count_below(
            value_cont,
            below_col,
            result_aliases=[result_alias, "below_pct"],
        )[0].alias(result_alias)
    ]


def statistics_count_below_pct(
    value_cont: str,
    below_col: str,
    result_alias: str = "below_pct",
) -> list[pl.Expr]:
    return [
        stat_count_below(
            value_cont,
            below_col,
            result_aliases=["below_n", result_alias],
        )[1].alias(result_alias)
    ]


def stat_arithmetic(
    value_col: str,
    *,
    result_aliases: list[str] = [
        "mean",
        "st",
        "sem",
        "mean_95_ci_lower",
        "mean_95_ci_upper",
    ],
    cutoff: float | None = None,
) -> list[pl.Expr]:
    n = pl.col(value_col).is_finite().sum()
    mean = pl.col(value_col).mean()
    sem = pl.col(value_col).std() / n.sqrt()
    return [
        _alias_with_cutoff(value_col, mean, result_aliases[0], cutoff),
        _alias_with_cutoff(
            value_col, pl.col(value_col).std(), result_aliases[1], cutoff
        ),
        _alias_with_cutoff(value_col, sem, result_aliases[2], cutoff),
        _alias_with_cutoff(
            value_col, mean - 1.96 * sem, result_aliases[3], cutoff
        ),
        _alias_with_cutoff(
            value_col, mean + 1.96 * sem, result_aliases[4], cutoff
        ),
    ]


def statistics_mean(
    value_cont: str,
    result_alias: str = "mean",
    cutoff: float | None = None,
) -> list[pl.Expr]:
    return [
        stat_arithmetic(
            value_cont,
            result_aliases=[
                result_alias,
                "st",
                "sem",
                "mean_95_ci_lower",
                "mean_95_ci_upper",
            ],
            cutoff=cutoff,
        )[0].alias(result_alias)
    ]


def statistics_st(
    value_cont: str,
    result_alias: str = "st",
    cutoff: float | None = None,
) -> list[pl.Expr]:
    return [
        stat_arithmetic(
            value_cont,
            result_aliases=[
                "mean",
                result_alias,
                "sem",
                "mean_95_ci_lower",
                "mean_95_ci_upper",
            ],
            cutoff=cutoff,
        )[1].alias(result_alias)
    ]


def statistics_sem(
    value_cont: str,
    result_alias: str = "sem",
    cutoff: float | None = None,
) -> list[pl.Expr]:
    return [
        stat_arithmetic(
            value_cont,
            result_aliases=[
                "mean",
                "st",
                result_alias,
                "mean_95_ci_lower",
                "mean_95_ci_upper",
            ],
            cutoff=cutoff,
        )[2].alias(result_alias)
    ]


def statistics_mean_95_ci_lower(
    value_cont: str,
    result_alias: str = "mean_95_ci_lower",
    cutoff: float | None = None,
) -> list[pl.Expr]:
    return [
        stat_arithmetic(
            value_cont,
            result_aliases=[
                "mean",
                "st",
                "sem",
                result_alias,
                "mean_95_ci_upper",
            ],
            cutoff=cutoff,
        )[3].alias(result_alias)
    ]


def statistics_mean_95_ci_upper(
    value_cont: str,
    result_alias: str = "mean_95_ci_upper",
    cutoff: float | None = None,
) -> list[pl.Expr]:
    return [
        stat_arithmetic(
            value_cont,
            result_aliases=[
                "mean",
                "st",
                "sem",
                "mean_95_ci_lower",
                result_alias,
            ],
            cutoff=cutoff,
        )[4].alias(result_alias)
    ]


def stat_geometric(
    value_col: str,
    *,
    result_aliases: list[str] = [
        "geom_mean",
        "geom_mean_95_ci_lower",
        "geom_mean_95_ci_upper",
    ],
    cutoff: float | None = None,
) -> list[pl.Expr]:
    n = pl.col(value_col).is_finite().sum()
    log_mean = pl.col(value_col).log().mean()
    se = pl.col(value_col).log().std() / n.sqrt()
    return [
        _alias_with_cutoff(
            value_col, log_mean.exp(), result_aliases[0], cutoff
        ),
        _alias_with_cutoff(
            value_col,
            (log_mean - 1.96 * se).exp(),
            result_aliases[1],
            cutoff,
        ),
        _alias_with_cutoff(
            value_col,
            (log_mean + 1.96 * se).exp(),
            result_aliases[2],
            cutoff,
        ),
    ]


def statistics_geom_mean(
    value_cont: str,
    result_alias: str = "geom_mean",
    cutoff: float | None = None,
) -> list[pl.Expr]:
    return [
        stat_geometric(
            value_cont,
            result_aliases=[
                result_alias,
                "geom_mean_95_ci_lower",
                "geom_mean_95_ci_upper",
            ],
            cutoff=cutoff,
        )[0].alias(result_alias)
    ]


def statistics_geom_mean_95_ci_lower(
    value_cont: str,
    result_alias: str = "geom_mean_95_ci_lower",
    cutoff: float | None = None,
) -> list[pl.Expr]:
    return [
        stat_geometric(
            value_cont,
            result_aliases=[
                "geom_mean",
                result_alias,
                "geom_mean_95_ci_upper",
            ],
            cutoff=cutoff,
        )[1].alias(result_alias)
    ]


def statistics_geom_mean_95_ci_upper(
    value_cont: str,
    result_alias: str = "geom_mean_95_ci_upper",
    cutoff: float | None = None,
) -> list[pl.Expr]:
    return [
        stat_geometric(
            value_cont,
            result_aliases=[
                "geom_mean",
                "geom_mean_95_ci_lower",
                result_alias,
            ],
            cutoff=cutoff,
        )[2].alias(result_alias)
    ]


def _percentile_ci_lower(
    value_col: str,
    q: float,
    *,
    result_aliases: list[str] = ["p", "ci_lower"],
) -> pl.Expr:
    """Calculate the lower confidence interval for a given percentile.

    Formula (Conover, 1999):
        j = nq - 1.96 root(nq(1-q))

    for ci_lower / n in [0,1]
    """
    n = pl.col(value_col).is_finite().sum()
    se = (n * q * (1 - q)).sqrt()
    ci_lower = (n * q - 1.96 * se).ceil()

    return (
        pl.col(value_col)
        .quantile((ci_lower / n).clip(0, 1), interpolation="nearest")
        .alias(f"{result_aliases[0]}{int(q * 100)}_{result_aliases[1]}")
    )


def _percentile_ci_upper(
    value_col: str,
    q: float,
    *,
    result_aliases: list[str] = ["p", "ci_upper"],
) -> pl.Expr:
    """Calculate the upper confidence interval for a given percentile.

    Formula (Conover, 1999):
        k = nq + 1.96 root(nq(1-q))
    """
    n = pl.col(value_col).is_finite().sum()
    se = (n * q * (1 - q)).sqrt()
    ci_upper = (n * q + 1.96 * se).ceil()

    return (
        pl.col(value_col)
        .quantile((ci_upper / n).clip(0, 1), interpolation="nearest")
        .alias(f"{result_aliases[0]}{int(q * 100)}_{result_aliases[1]}")
    )


def stat_percentiles(
    value_col: str,
    quants: list[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    *,
    result_aliases: list[str] = ["p", "ci_lower", "ci_upper"],
) -> list[pl.Expr]:
    quantile_exprs = [
        pl.col(value_col)
        .quantile(q)
        .alias(f"{result_aliases[0]}{int(q * 100)}")
        for q in quants
    ]
    quantile_ci_lower_exprs = [
        _percentile_ci_lower(
            value_col, q, result_aliases=[result_aliases[0], result_aliases[1]]
        )
        for q in quants
    ]
    quantile_ci_upper_exprs = [
        _percentile_ci_upper(
            value_col, q, result_aliases=[result_aliases[0], result_aliases[2]]
        )
        for q in quants
    ]
    return quantile_exprs + quantile_ci_lower_exprs + quantile_ci_upper_exprs


def statistics_percentiles_p5(
    value_cont: str, quants: list[float] = [0.05], result_aliases: str = "p5"
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[0].alias(result_aliases)
    ]


def statistics_percentiles_p10(
    value_cont: str, quants: list[float] = [0.1], result_aliases: str = "p10"
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[0].alias(result_aliases)
    ]


def statistics_percentiles_p25(
    value_cont: str, quants: list[float] = [0.25], result_aliases: str = "p25"
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[0].alias(result_aliases)
    ]


def statistics_percentiles_p50(
    value_cont: str, quants: list[float] = [0.5], result_aliases: str = "p50"
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[0].alias(result_aliases)
    ]


def statistics_percentiles_p75(
    value_cont: str, quants: list[float] = [0.75], result_aliases: str = "p75"
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[0].alias(result_aliases)
    ]


def statistics_percentiles_p90(
    value_cont: str, quants: list[float] = [0.9], result_aliases: str = "p90"
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[0].alias(result_aliases)
    ]


def statistics_percentiles_p95(
    value_cont: str, quants: list[float] = [0.95], result_aliases: str = "p95"
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[0].alias(result_aliases)
    ]


def statistics_percentiles_p5_ci_lower(
    value_cont: str,
    quants: list[float] = [0.05],
    result_aliases: str = "p5_ci_lower",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[1].alias(result_aliases)
    ]


def statistics_percentiles_p10_ci_lower(
    value_cont: str,
    quants: list[float] = [0.1],
    result_aliases: str = "p10_ci_lower",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[1].alias(result_aliases)
    ]


def statistics_percentiles_p25_ci_lower(
    value_cont: str,
    quants: list[float] = [0.25],
    result_aliases: str = "p25_ci_lower",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[1].alias(result_aliases)
    ]


def statistics_percentiles_p50_ci_lower(
    value_cont: str,
    quants: list[float] = [0.5],
    result_aliases: str = "p50_ci_lower",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[1].alias(result_aliases)
    ]


def statistics_percentiles_p75_ci_lower(
    value_cont: str,
    quants: list[float] = [0.75],
    result_aliases: str = "p75_ci_lower",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[1].alias(result_aliases)
    ]


def statistics_percentiles_p90_ci_lower(
    value_cont: str,
    quants: list[float] = [0.9],
    result_aliases: str = "p90_ci_lower",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[1].alias(result_aliases)
    ]


def statistics_percentiles_p95_ci_lower(
    value_cont: str,
    quants: list[float] = [0.95],
    result_aliases: str = "p95_ci_lower",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[1].alias(result_aliases)
    ]


def statistics_percentiles_p5_ci_upper(
    value_cont: str,
    quants: list[float] = [0.05],
    result_aliases: str = "p5_ci_upper",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[2].alias(result_aliases)
    ]


def statistics_percentiles_p10_ci_upper(
    value_cont: str,
    quants: list[float] = [0.1],
    result_aliases: str = "p10_ci_upper",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[2].alias(result_aliases)
    ]


def statistics_percentiles_p25_ci_upper(
    value_cont: str,
    quants: list[float] = [0.25],
    result_aliases: str = "p25_ci_upper",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[2].alias(result_aliases)
    ]


def statistics_percentiles_p50_ci_upper(
    value_cont: str,
    quants: list[float] = [0.5],
    result_aliases: str = "p50_ci_upper",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[2].alias(result_aliases)
    ]


def statistics_percentiles_p75_ci_upper(
    value_cont: str,
    quants: list[float] = [0.75],
    result_aliases: str = "p75_ci_upper",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[2].alias(result_aliases)
    ]


def statistics_percentiles_p90_ci_upper(
    value_cont: str,
    quants: list[float] = [0.9],
    result_aliases: str = "p90_ci_upper",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[2].alias(result_aliases)
    ]


def statistics_percentiles_p95_ci_upper(
    value_cont: str,
    quants: list[float] = [0.95],
    result_aliases: str = "p95_ci_upper",
) -> list[pl.Expr]:
    return [
        stat_percentiles(
            value_cont,
            quants=quants,
            result_aliases=["p", "ci_lower", "ci_upper"],
        )[2].alias(result_aliases)
    ]


def frequency_table(
    value_cols: list[str],
    *,
    result_aliases: list[str] = ["value", "frequency"],
) -> Callable[[pl.DataFrame | pl.LazyFrame], pl.DataFrame | pl.LazyFrame]:
    def get_result(
        data: pl.DataFrame | pl.LazyFrame,
    ) -> pl.DataFrame | pl.LazyFrame:
        return (
            data.group_by([pl.col(col) for col in value_cols])
            .agg(pl.len())
            .rename(
                {
                    value_cols[-1]: result_aliases[0],
                    "len": result_aliases[1],
                }
            )
            .sort(result_aliases[0])
        )

    return get_result
