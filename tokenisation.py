import io
import requests
from pydantic import HttpUrl
from PyPDF2 import PdfReader
import regex as re
import nltk.data

important_words = open("./data/scraping/important_words.txt").read().splitlines()

TOKENISER = nltk.data.load("tokenizers/punkt/english.pickle")


def get_pdf_text(url: HttpUrl | str) -> str:
    response = requests.get(str(url))
    mem_obj = io.BytesIO(response.content)
    pdf = PdfReader(mem_obj)
    press_conference = "".join(
        pdf.pages[i].extract_text().replace("\n", "") for i in range(len(pdf.pages))
    )
    return press_conference


def get_speaker_text(press_conference: str, fed_chairs: str | list) -> list[str]:
    if not isinstance(fed_chairs, list):
        fed_chairs = [fed_chairs]
    all_matches = []
    for fed_chair in fed_chairs:
        spaced_chair = "[\\s\\n]*".join(fed_chair)
        pattern = (
            f"{spaced_chair}[\\s]*\\.[\\s]*(.*?)(?=(?:[A-Z][\\s]*[A-Z][\\s]*\\.)|$)"
        )
        matches = re.findall(pattern, press_conference, re.DOTALL)
        for match in matches:
            all_matches.extend(TOKENISER.tokenize(match.strip().replace("\n", " ")))
    return all_matches


def basic_tokeniser(text: str) -> list[str]:
    return TOKENISER.tokenize(text)


def get_important_text(all_lines: list[str]) -> list[str]:
    return [
        line
        for line in all_lines
        if any(word in line.lower() for word in important_words)
    ]
