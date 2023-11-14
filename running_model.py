import os
from api_calls import get_multiple_api_calls
from pathlib import Path
from typing import Union
import regex as re
import pandas as pd
import numpy as np
import structlog

RESULTS = Path("data/results")
logger = structlog.get_logger()


def _fetch_df(input_path: Union[str, Path], output_path: Union[str, Path]):
    if os.path.exists(output_path):
        file = pd.read_parquet(output_path)
        if all(col_name in file for col_name in ["label", "reason"]):
            logger.info(f"{output_path} already created")
            return file
    file = pd.read_parquet(input_path)
    for col_name in ["label", "reason"]:
        file[col_name] = [[] for _ in range(file.shape[0])]
    file.to_parquet(output_path)
    return file


def _find_label(string: Union[str, float]) -> int | None:
    if not string:
        return np.nan
    match = re.search(r"\b(hawkish|dovish|neutral)\b", string, re.IGNORECASE)
    if match:
        sentiment = match.group(0).lower()
        if sentiment == "dovish":
            return 0
        elif sentiment == "hawkish":
            return 1
        else:
            return 2
    else:
        return np.nan


async def call_from_df(
    df: pd.DataFrame,
    important_words: list[str],
    output_dir: Union[str, Path],
    model: str,
    instructions: list[str],
):

    for i in range(df.shape[0]):
        # If we already have labels, continue
        if any(df.iloc[i, -2]):
            logger.info(f"got {df.index[i]}, moving on")
            continue
        # Filter out useless sentences
        sentences = [
            sentence if any(word in sentence for word in important_words) else None
            for sentence in df.iloc[i, -3]
        ]
        if len([sentence for sentence in sentences if sentence]) < 20:
            continue
        # Make all the API calls
        sentiments = await get_multiple_api_calls(model, instructions, sentences)
        # Check if our prompt has multiple outputs
        multiple_instructions = all(len(sentiment) == 2 for sentiment in sentiments)
        # Add labels to df
        df.iloc[i, -2] = np.append(
            df.iloc[i, -2],
            [
                _find_label(sentiment[1])
                if multiple_instructions
                else _find_label(sentiment[0])
                for sentiment in sentiments
            ],
        )
        # Add reasoning to df
        df.iloc[i, -1] = np.append(
            df.iloc[i, -1], [sentiment[0] for sentiment in sentiments]
        )
        # If we have added 5 rows, do some logging and save the df
        if i % 5 == 0:
            logger.info(
                "saving df", index=i, progress=f"{round(i / df.shape[0] * 100, 2)}%"
            )
            df.to_parquet(output_dir)
    df.to_parquet(output_dir)
    return df


def get_hawk_score(results_df: pd.DataFrame):
    assert "label" in results_df.columns, "No Label Column"
    hawk_scores = {
        label: [] for label in ["hawk_score_1", "hawk_score_2", "hawk_score_3"]
    }
    for sentiments in [list(ele) for ele in results_df["label"]]:
        if all(sentiments.count(num) > 5 for num in [0.0, 1.0, 2.0]):
            hawk_scores["hawk_score_1"].append(
                np.mean(
                    [0] * sentiments.count(0.0)
                    + [0.5] * sentiments.count(2.0)
                    + [1] * sentiments.count(1.0)
                )
            )
            hawk_scores["hawk_score_2"].append(
                sentiments.count(1) / (sentiments.count(0) + sentiments.count(1))
            )
            hawk_scores["hawk_score_3"].append(
                sentiments.count(1)
                / (sentiments.count(0) + sentiments.count(1) + sentiments.count(2))
            )
        else:
            for i in range(1, 4):
                hawk_scores[f"hawk_score_{i}"].append(np.nan)
    return pd.concat([results_df, pd.DataFrame(hawk_scores, index=results_df.index)], axis=1)

def get_title_significance(titles: list[str], instructions: list[str]):
    pass



async def run_model(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    important_words: list[str],
    model: str,
    instructions: list[str],
) -> pd.DataFrame:
    base_df = _fetch_df(input_path, output_path)
    for i in range(1,4):
        if f"hawk_score_{i}" in base_df.columns:
            del base_df[f"hawk_score_{i}"]
    results_df = await call_from_df(
        base_df, important_words, output_path, model, instructions
    )
    hawk_score_df = get_hawk_score(results_df)
    hawk_score_df.to_parquet(output_path)
    return hawk_score_df
