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


async def get_sentence_reasoning(
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
            ],
            model = "gpt-4"
        )
        for sentiment, base_input in zip(sentiments, base_inputs)
    ]
    return await asyncio.gather(*futures)


async def get_reasoning_df(annotated_data: pd.DataFrame, file_name: str) -> pd.DataFrame:
    raw_sentiment = await get_sentence_reasoning(
        list(annotated_data["label"]), list(annotated_data["sentence"])
    )
    annotated_data = annotated_data.assign(reason = raw_sentiment)
    annotated_data.to_csv(file_name)
    return annotated_data
