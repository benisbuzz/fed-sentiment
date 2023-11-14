from dotenv import load_dotenv
import aiohttp
import ssl
import certifi
import os
import structlog
import asyncio
import time
from typing import Optional, Union

load_dotenv()

logger = structlog.get_logger()

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
SEMAPHORE = asyncio.Semaphore(12)


def open_session() -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(total=10.0)
    return aiohttp.ClientSession(timeout=timeout)


async def close_session(session: aiohttp.ClientSession) -> None:
    await session.close()


async def _get_api_call(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict[str, str]],
    index: Optional[int] = 1,
    total: Optional[int] = 1,
) -> str | ConnectionAbortedError:
    base_wait_time = 1
    max_retries = 20
    async with SEMAPHORE:
        for i in range(max_retries):
            wait_time = base_wait_time * (1.9**i)
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.0,
                },
                ssl=SSL_CONTEXT,
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            ) as response:
                logger.info(response.status)
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"completed {index} / {total}")
                    return result["choices"][0]["message"]["content"]
                elif response.status in [429, 502]:
                    logger.error(
                        f"Got {response.status}",
                        response=response.status,
                        wait_time=wait_time,
                        index=index,
                        total=total,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Unexpected Error",
                        response_status=response.status,
                        wait_time=wait_time,
                        index=index,
                        total=total,
                    )
                    await asyncio.sleep(wait_time)
    raise ConnectionAbortedError("Too many errors. Aborted API Call")


async def _get_api_call_error_handling(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict[str, str]],
    index: Optional[int] = 1,
    total: Optional[int] = 1,
) -> str:
    counter = 0
    base_wait_time = 1
    while counter <= 10:
        try:
            return await _get_api_call(session, model, messages, index, total)
        except asyncio.TimeoutError:
            wait_time = base_wait_time * (1.9**counter)
            logger.error(
                "Got timeout error. Initiating exponential backoff.",
                wait_time=wait_time,
                counter=counter,
            )
            time.sleep(wait_time)
            counter += 1
            try:
                return await _get_api_call(session, model, messages, index, total)
            except asyncio.TimeoutError:
                wait_time = base_wait_time * (1.9**counter)
                logger.error(
                    "got timeout error. Initiating exponential backoff",
                    wait_time=5,
                    counter=counter,
                )
                time.sleep(wait_time)
                counter += 1
        except aiohttp.ClientOSError:
            wait_time = base_wait_time * (1.9**counter)
            logger.error(
                "Client OS Error. Initiating exponential backoff",
                wait_time=wait_time,
                counter=counter,
            )
            time.sleep(wait_time)
            counter += 1
            try:
                return await _get_api_call(session, model, messages, index, total)
            except aiohttp.ClientOSError:
                wait_time = base_wait_time * (1.9**counter)
                logger.error("got OS Error, waiting", wait_time=5, counter=counter)
                time.sleep(wait_time)
                counter += 1
    raise TimeoutError("Exhausted Error Handling")


async def get_api_call(
    session: aiohttp.ClientSession,
    model: str,
    instructions: list[str],
    base_input: Union[str, None],
    index: Optional[int] = 1,
    total: Optional[int] = 1,
) -> list[str | None] | None:
    if not base_input:
        return [None] * len(instructions)
    answers = []
    messages = [
        {"role": "user", "content": instructions[0] + base_input},
    ]
    try:
        answer = await _get_api_call_error_handling(
            session, model, messages, index, total
        )
    except TimeoutError:
        answers.append(None)
        logger.info(f"Couldn't get API call for {index}, defaulting to None")
        return None
    answers.append(answer)
    for prompt in instructions[1:]:
        messages.extend(
            (
                {"role": "assistant", "content": answer},
                {"role": "user", "content": prompt},
            )
        )
        answer = await _get_api_call_error_handling(
            session, model, messages, index, total
        )
        answers.append(answer)
    return answers


async def get_multiple_api_calls(
    model: str, instructions: list[str], base_inputs: list[str]
) -> list[list[str]]:
    session = open_session()
    futures = [
        get_api_call(session, model, instructions, base_input, index, len(base_inputs))
        for index, base_input in enumerate(base_inputs, start=1)
    ]
    result = await asyncio.gather(*futures)
    await close_session(session)
    return result


def get_justify_prompt(sentiment):
    return (
        f"The following sentence indicates a {sentiment} stance on US monetary policy. Explain why the sentence is {sentiment} "
        f"in less than 50 words. Start your answer with 'This sentence is {sentiment} because'. The sentence: "
    )


async def get_multiple_api_calls_given_sentiment(
    sentiments: list[int], base_inputs: list[str]
) -> list[str]:
    key = {0: "dovish", 1: "hawkish", 2: "neutral", "-": "neutral"}
    session = open_session()
    futures = [
        _get_api_call_error_handling(
            session,
            "gpt-4",
            [
                {
                    "role": "user",
                    "content": get_justify_prompt(key[sentiment]) + base_input,
                }
            ],
        )
        for sentiment, base_input in zip(sentiments, base_inputs)
    ]
    result = await asyncio.gather(*futures)
    await close_session(session)
    return result
