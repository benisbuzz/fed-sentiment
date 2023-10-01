import asyncio
from time import sleep
import pandas as pd
import openai
import structlog
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
import api_calls as api

load_dotenv()

logger = structlog.get_logger(__name__)



def get_justify_prompt(sentiment: str) -> str:
    return (
        f"The following sentence indicates a {sentiment} stance on US monetary policy. Explain why the sentence is {sentiment} "
        f"in less than 50 words. Start your answer with 'This sentence is {sentiment} because'. The sentence: "
    )


async def get_multiple_answers(
    sentiments: list[int], base_inputs: list[str]
) -> list[str]:
    key = {0: "dovish", 1: "hawkish", 2: "neutral", "-": "neutral"}
    futures = [
        api.get_api_call(
            [
                {
                    "role": "user",
                    "content": get_justify_prompt(key[sentiment]) + base_input,
                }
            ]
        )
        for sentiment, base_input in zip(sentiments, base_inputs)
    ]
    return await asyncio.gather(*futures)


async def get_reasoning(annotated_data: pd.DataFrame, file_name: str) -> pd.DataFrame:
    raw_sentiment = await get_multiple_answers(
        list(annotated_data.loc[:, "label"]), list(annotated_data.loc[:, "sentence"])
    )
    for i, justification in enumerate(raw_sentiment):
        annotated_data.iloc[i, 2] = justification
        annotated_data.to_csv(f"{file_name}.csv")
    return annotated_data
