import json
import pandas as pd
import os
from pathlib import Path
import regex as re
import structlog

logger = structlog.get_logger()

def create_index(prompt_name: str, model: str) -> pd.Series:
    results_dir = Path(f"./data/results/{prompt_name}/{model}")
    results_files = os.listdir(results_dir)
    result = []
    for file_name in results_files:
        logger.info(f"retreiving {file_name}")
        with open(results_dir/file_name, "r") as file:
            json_file = json.load(file)
        sentiments = [sentiment["label"] for sentiment in json_file]
        date = pd.Timestamp(re.search(r'\d{8}', file_name).group(0))
        result.append((date, sentiments.count(1) / (sentiments.count(0) + sentiments.count(1))))
    idx, values = zip(*result)
    return pd.Series(values, idx).sort_index()
