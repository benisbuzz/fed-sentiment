import json
import pandas as pd
import os
from pathlib import Path
import regex as re
import structlog
import numpy as np
from typing import Union, Optional, Tuple
import statsmodels.api as sm
from statsmodels.iolib.summary import Summary
from talib import MA as ma

logger = structlog.get_logger()


def _get_sentiment(results_dir: Union[str, Path]):
    filenames = os.listdir(results_dir)
    result = []
    for filename in filenames:
        logger.info(f"retrieving {filename}")
        with open(results_dir / filename, "r") as file:
            json_file = json.load(file)
        sentiments = [sentiment["label"] for sentiment in json_file]
        if sentiments.count(1) < 5 or sentiments.count(0) < 5:
            continue
        date = pd.Timestamp(re.search(r"\d{8}", filename).group(0))
        labels = (
            [0.5] * (sentiments.count(2) + sentiments.count(3))
            + [0] * sentiments.count(0)
            + [1] * sentiments.count(1)
        )
        # sentiments.count(1) / (sentiments.count(0) + sentiments.count(1))
        result.append((date, np.mean(labels)))
    return result


def create_index(prompt_name: str, model: str) -> pd.Series:
    results_dir = Path(f"./data/results/{prompt_name}/{model}")
    press_conferences = os.listdir(results_dir / "press_conferences")
    speeches = os.listdir(results_dir / "speeches_2")
    press_conferences = _get_sentiment(
        Path(f"./data/results/{prompt_name}/{model}/press_conferences")
    )
    speeches = _get_sentiment(Path(f"./data/results/{prompt_name}/{model}/speeches_2"))
    result = press_conferences + speeches
    idx, values = zip(*result)
    return pd.Series(values, idx).sort_index()


def get_equivalent_series(
    hawk_series: pd.Series,
    data_series: pd.Series,
    log: bool,
    ma_lag: Optional[int] = None,
    shift_lag: Optional[int] = None,
) -> pd.Series:
    hawk_series = hawk_series[~hawk_series.index.duplicated(keep="first")]
    if ma_lag:
        hawk_series = ma(hawk_series, ma_lag)
    if shift_lag:
        hawk_series.index += pd.DateOffset(days=shift_lag)
    data_series = data_series.resample("D").ffill().reindex(hawk_series.index)
    if log:
        hawk_series = np.log(hawk_series / hawk_series.shift(1))
        data_series = np.log(data_series / data_series.shift(1))

    common_index = data_series.dropna().index.intersection(hawk_series.dropna().index)
    return (hawk_series[common_index], data_series[common_index])


def get_correlation_df(
    hawk_series: pd.Series,
    data_series: pd.Series,
    ma_lags: range,
    shift_lags: range,
    log: bool,
) -> pd.DataFrame:
    df = pd.DataFrame(
        data={f"ma_{i}": [None] * len(shift_lags) for i in ma_lags},
        index=[f"shift_{i}" for i in shift_lags],
    )
    for ma_lag in ma_lags:
        for shift_lag in shift_lags:
            equalised_series = get_equivalent_series(
                hawk_series, data_series, log, ma_lag, shift_lag
            )
            df.loc[f"shift_{shift_lag}", f"ma_{ma_lag}"] = np.corrcoef(
                np.array(equalised_series[0]), np.array(equalised_series[1])
            )[0][1]
    return df


def get_regression_summary(dependant: np.array, *args: Tuple[np.array, ...]) -> Summary:
    regressors = []
    for arg in args:
        regressors.append(arg)
    x = sm.add_constant(np.column_stack(regressors))
    model = sm.OLS(dependant, x).fit()
    return model.summary()

def get_regression_df(hawk_series: pd.Series, data_series: pd.Series, log: bool, ma_lags: range, shift_lags: range) -> pd.DataFrame:
    df = pd.DataFrame(
        data={f"ma_{i}": [None] * len(shift_lags) for i in ma_lags},
        index=[f"shift_{i}" for i in shift_lags],
    )
    for ma_lag in ma_lags:
        for shift_lag in shift_lags:
            series = get_equivalent_series(hawk_series, data_series, log, ma_lag, shift_lag)
            ols = get_regression_summary(np.array(series[0]), np.array(series[1]))
            df.loc[f"shift_{shift_lag}", f"ma_{ma_lag}"] = ols.tables[0].data[5][-1]
    return df


def get_loc_max(df: pd.DataFrame) -> tuple[str, str]:
    max_value = df.values.max()
    return df[df == max_value].stack().index[0]

def get_loc_min(df: pd.DataFrame) -> tuple[str, str]:
    min_value = df.values.min()
    return df[df == min_value].stack().index[0]