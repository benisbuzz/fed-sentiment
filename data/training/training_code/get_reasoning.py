import asyncio
from time import sleep
import pandas as pd
import openai
import structlog
import os
from aiolimiter import AsyncLimiter

openai.api_key = os.getenv("OPENAI_API_KEY")

logger = structlog.get_logger(__name__)

limiter = AsyncLimiter(1000)

def get_justify_prompt(sentiment):
    return (f"The following sentence indicates a {sentiment} stance on US monetary policy. Explain why the sentence is {sentiment} "
            f"in less than 50 words. Start your answer with 'This sentence is {sentiment} because'. The sentence: ")

limiter = AsyncLimiter(1000)
async def get_api_call(messages: list[dict[str, str]]) -> str:
    async with limiter:
        completion = None
        error_count = 0
        while completion is None:
            try:
                completion = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.0,
                )
                logger.info("got api call")
            except openai.error.RateLimitError as rate_limit_error:
                print(f"{rate_limit_error}, sleeping for 1 second")
                sleep(1)
            except Exception as e:
                error_count += 1
                print(f"{e}: Trying again - (50-error count) / 50 attempts remaining")
                if error_count == 50:
                    print(type(e))
                    raise RuntimeError("Failed 50 times")
    return completion.choices[0].message.content

async def get_multiple_answers(sentiments: list[int], base_inputs: list[str]) -> list[str]:
    key = {0: "dovish", 1: "hawkish", 2: "neutral", "-": "neutral"}
    futures = [get_api_call(
        [{"role": "user", "content": get_justify_prompt(key[sentiment]) + base_input}]
        ) for sentiment, base_input in zip(sentiments, base_inputs)]
    return await asyncio.gather(*futures)

async def get_reasoning(annotated_data: pd.DataFrame, file_name: str) -> pd.DataFrame:
    raw_sentiment = await get_multiple_answers(
        list(annotated_data.iloc[:, 1]), list(annotated_data.iloc[:, 0])
    )
    for i, justification in enumerate(raw_sentiment):
        annotated_data.iloc[i, 2] = justification
        annotated_data.to_csv(f"{file_name}.csv")
    return annotated_data
