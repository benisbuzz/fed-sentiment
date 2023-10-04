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

import asyncio
import aiohttp
import os

async def get_api_call(messages: list[dict[str, str]]) -> str:
    base_wait_time = 1 
    max_retries = 10
    async with aiohttp.ClientSession() as session:
        for i in range(max_retries):
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "ft:gpt-3.5-turbo-0613:personal::862ZOMQr",
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
                    wait_time = base_wait_time * (2 ** i)
                    logger.info(f"Rate limit exceeded. Initiating Exponential backoff", wait_time=wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Unexpected error. Initiaiting exponential backoff", response_status=response.status, wait_time=wait_time
                    )
                    wait_time = base_wait_time * (2 ** i)
                    await asyncio.sleep(wait_time)
        logger.error("Exhausted all retries without a successful request")
        return "Error: Exhausted all retries without a successful request"


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
    return await asyncio.gather(*futures)
