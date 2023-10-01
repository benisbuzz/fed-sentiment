from dotenv import load_dotenv
import aiohttp
import ssl
import certifi
import os
import structlog
import asyncio
from aiolimiter import AsyncLimiter

load_dotenv()

logger = structlog.get_logger()

SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

limiter = AsyncLimiter(1000)


async def get_api_call(messages: list[dict[str, str]]) -> str:
    async with limiter:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": messages,
                    "temperature": 0.0,
                },
                ssl=SSL_CONTEXT,
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    logger.info(
                        f"Failed to get a valid response. Status code: {response.status}"
                    )
                    return await response.text()


async def get_answer(
    base_prompts: list[str], base_input: str, with_input: bool
) -> dict[str, str]:
    answers = {}
    messages = [
        {"role": "user", "content": base_prompts[0] + base_input},
    ]
    answer = await get_api_call(messages)
    if with_input:
        answers["input"] = base_input
    answers["output_1"] = answer
    for i, prompt in enumerate(base_prompts[1:], start=2):
        messages.extend(
            (
                {"role": "assistant", "content": answer},
                {"role": "user", "content": prompt},
            )
        )
        answer = await get_api_call(messages)
        answers[f"output_{i}"] = answer
    return answers


async def get_multiple_answers(
    base_prompts: list[str], base_inputs: list[str], with_input: bool
) -> list[dict[str, str]]:
    futures = [
        get_answer(base_prompts, base_input, with_input) for base_input in base_inputs
    ]
    logger.info("fetched futures")
    return await asyncio.gather(*futures)
