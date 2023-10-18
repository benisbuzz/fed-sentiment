from dotenv import load_dotenv
import aiohttp
import ssl
import certifi
import os
import structlog
import asyncio
import time
from typing import Optional

load_dotenv()

logger = structlog.get_logger()

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
SEMAPHORE = asyncio.Semaphore(25)


def open_session() -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(total=30.0)
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
                if response.status == 200:
                    result = await response.json()
                    logger.info(
                        f"completed {index} / {total}",
                        response_status=response.status,
                        index=index,
                        total=total,
                    )
                    return result["choices"][0]["message"]["content"]
                elif response.status == 429:
                    wait_time = base_wait_time * (2**i)
                    logger.error(
                        "Rate limit error. Initiaiting exponential backoff",
                        response_status=response.status,
                        wait_time=wait_time,
                        index=index,
                        total=total,
                    )
                    await asyncio.sleep(wait_time)
                elif response.status == 502:
                    logger.error(
                        "Bad gateway, waiting..",
                        response_status=response.status,
                        wait_time=5,
                        index=index,
                        total=total,
                    )
                    await asyncio.sleep(5)
                else:
                    logger.error(
                        f"Unexpected Error. Waiting...",
                        response_status=response.status,
                        wait_time=2,
                        index=index,
                        total=total,
                    )
                    await asyncio.sleep(2)
    raise ConnectionAbortedError("Too many errors. Aborted API Call")


async def _get_api_call_handle_timeout(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict[str, str]],
    index: Optional[int] = 1,
    total: Optional[int] = 1,
) -> str:
    counter = 0
    while counter < 5:
        try:
            return await _get_api_call(session, model, messages, index)
        except asyncio.TimeoutError:
            logger.error("got timeout error, waiting", wait_time=5, counter=counter)
            time.sleep(5)
            try:
                return await _get_api_call(session, model, messages, index)
            except asyncio.TimeoutError:
                logger.error("got timeout error, waiting", wait_time=5, counter=counter)
                time.sleep(5)
                counter += 1
                pass
        except aiohttp.ClientOSError:
            logger.error("Client OS Error, waiting", wait_time=5, counter=counter)
            time.sleep(5)
            try:
                return await _get_api_call(session, model, messages, index)
            except aiohttp.ClientOSError:
                logger.error("got OS Error, waiting", wait_time=5, counter=counter)
                time.sleep(5)
                counter += 1
                pass


async def get_api_call(
    session: aiohttp.ClientSession,
    model: str,
    instructions: list[str],
    base_input: str,
    index: Optional[int] = 1,
    total: Optional[int] = 1,
) -> list[str]:
    answers = []
    messages = [
        {"role": "user", "content": instructions[0] + base_input},
    ]
    answer = await _get_api_call_handle_timeout(session, model, messages, index, total)
    answers.append(answer)
    for prompt in instructions[1:]:
        messages.extend(
            (
                {"role": "assistant", "content": answer},
                {"role": "user", "content": prompt},
            )
        )
        answer = await _get_api_call_handle_timeout(
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
