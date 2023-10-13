import os
import api_calls as api
import tokenisation as tkn
from pathlib import Path
from typing import Union, Optional
import regex as re
import json
import structlog

RESULTS = Path("./data/results")
logger = structlog.get_logger()


def get_txt_list(path: Union[str, Path]) -> list[str]:
    with open(path, "r") as file:
        return file.read().splitlines()


def _make_dir(dir: Union[Path, str]) -> None:
    if not os.path.exists(dir):
        logger.info("creating dir", path=dir)
        os.makedirs(dir)


def _find_label(string: str) -> int | None:
    match = re.search(r"\b(hawkish|dovish|neutral|irrelevant)\b", string, re.IGNORECASE)
    if match:
        sentiment = match.group(0).lower()
        if sentiment == "dovish":
            return 0
        elif sentiment == "hawkish":
            return 1
        elif sentiment == "neutral":
            return 2
        else:
            return 3
    else:
        return None


def _get_file_name(model: str, prompt_name: str, date: str) -> str:
    return f"{model}_{prompt_name}_{date}.json"


async def _scrape_and_call(
    dates: list[str],
    base_url: str,
    is_pc: bool,
    model: str,
    instructions: list[str],
    prompt_name: str,
):
    output_dir = RESULTS / prompt_name / model
    # First, lets create output dir if it does not already exist
    _make_dir(output_dir)
    for date in dates:
        logger.info(f"working on {date}")
        result = []
        # Now check if we have already got the file we are about to make, if we do, move on
        file_name = _get_file_name(model, prompt_name, date)
        if os.path.isfile(output_dir / file_name):
            logger.info(f"already got {file_name}. Continuing")
            continue
        # Now we start to tokenise, beginning with getting raw text
        raw_release = tkn.get_pdf_text(base_url + date + ".pdf")
        logger.info("fetched release text", text_len=len(raw_release))
        if is_pc:
            # If we are working with PC, just fetch chairman lines and filter out irrelevant sentence
            speaker_split = tkn.get_speaker_text(
                raw_release, get_txt_list("./data/scraping/fed_chairs.txt")
            )
            logger.info("split pc into chair lines", num_lines=len(speaker_split))
            final_split = tkn.get_important_text(speaker_split)
            logger.info("finished tokenising", num_sentences=len(final_split))
        else:
            # If its not a PC, just filter out irrelevant sentences
            final_split = tkn.get_important_text(tkn.basic_tokeniser(raw_release))
            logger.info("finished tokenising", num_sentences=len(final_split))
        # Now we can run these tokenised sentences through chatgpt with the passed instructions
        sentiments = await api.get_multiple_api_calls(model, instructions, final_split)
        logger.info("finished getting sentiments")
        # Now lets get individual lists for setniment reasoning and sentiment labels.
        # This will depend on whether we have multiple instructions for each sentence
        multiple_instructions = all(len(sentiment) == 2 for sentiment in sentiments)
        labels = [
            _find_label(sentiment[1])
            if multiple_instructions is True
            else _find_label(sentiment)
            for sentiment in sentiments
        ]
        reasons = [
            sentiment[0] if multiple_instructions is True else sentiment
            for sentiment in sentiments
        ]
        # Now lets build our result list for this date
        for sentence, label, reason in zip(final_split, labels, reasons):
            result.append({"sentence": sentence, "label": label, "reason": reason})
        # Finally we can export to json
        with open(output_dir / file_name, "w") as file:
            json.dump(result, file)
        logger.info("wrote file", date=date)


async def test_model(
    model: str,
    prompt_name: str,
    pc_dates: Optional[list[str]] = None,
    sp_dates: Optional[list[str]] = None,
) -> None:
    instructions = get_txt_list(f"./data/prompts/{prompt_name}.txt")
    if pc_dates:
        base_url = "https://www.federalreserve.gov/mediacenter/files/"
        await _scrape_and_call(
            pc_dates, base_url, True, model, instructions, prompt_name
        )
    if sp_dates:
        base_url = "https://www.federalreserve.gov/newsevents/speech/files/"
        await _scrape_and_call(
            sp_dates, base_url, False, model, instructions, prompt_name
        )
