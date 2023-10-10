from dotenv import load_dotenv
import aiohttp
import ssl
import certifi
import os
import structlog
import asyncio

load_dotenv()

logger = structlog.get_logger()

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
SEMAPHORE = asyncio.Semaphore(100)

def open_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession()

async def close_session(session: aiohttp.ClientSession) -> None:
    await session.close()


async def _get_api_call(
    session: aiohttp.ClientSession, messages: list[dict[str, str]], model: str
) -> str | ConnectionAbortedError:
    base_wait_time = 1
    max_retries = 10
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
                    logger.info("got response", response_status=response.status)
                    return result["choices"][0]["message"]["content"]
                elif response.status == 429:
                    wait_time = base_wait_time * (2**i)
                    logger.error(
                        "Rate limit error. Initiaiting exponential backoff",
                        response_status=response.status,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                elif response.status == 502:
                    logger.error(
                        "Bad gateway, waiting..",
                        response_status = response.status,
                        wait_time = 1,
                    )
                    await asyncio.sleep(1)
                else:
                    logger.error(
                        f"Unexpected Error. Waiting...",
                        response_status=response.status,
                        wait_time=2,
                    )
                    await asyncio.sleep(2)
    raise ConnectionAbortedError("Too many errors. Aborted API Call")


async def get_api_call(session: aiohttp.ClientSession, messages: list[dict[str, str]], model: str) -> str:
    counter = 0
    while counter <= 50:
        try:
            return await _get_api_call(session, messages, model)
        except Exception as e:
            logger.error(f"got {e}. waiting...", error_count=counter, wait_time=4)
            await asyncio.sleep(4)
            counter += 1
    return "Unable to make api call"


async def get_multiple_api_calls(
    system: str, prompts: list[str], model: str
) -> list[str]:
    session = open_session()
    futures = [
        get_api_call(
            session,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            model,
        )
        for prompt in prompts
    ]
    result = await asyncio.gather(*futures)
    await close_session(session)
    return result

