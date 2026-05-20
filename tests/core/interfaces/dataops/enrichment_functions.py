import datetime
import polars as pl


def age_in_months_from_birth_and_interview_dates(birth_date, interview_date):
    return pl.Series(
        (
            (interview.year - birth.year) * 12
            + (interview.month - birth.month)
            for birth, interview in zip(birth_date, interview_date)
        ),
        dtype=pl.Int64,
    )


def datetime_from_year_month_day(
    year: pl.Series, month: pl.Series, day: pl.Series
) -> pl.Series:
    return pl.Series(
        (
            datetime.datetime(int(y), int(m), int(d))
            for y, m, d in zip(year, month, day)
        ),
        dtype=pl.Datetime,
    )


def date_from_year_month_day(
    year: pl.Series, month: pl.Series, day: pl.Series
) -> pl.Series:
    return pl.Series(
        (
            datetime.date(int(y), int(m), int(d))
            for y, m, d in zip(year, month, day)
        ),
        dtype=pl.Date,
    )


def transform_birthweight(
    birthweight: pl.Series, constant: float
) -> pl.Series:
    return birthweight + constant
