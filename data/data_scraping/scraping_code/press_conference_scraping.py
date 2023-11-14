import os
import re
import urllib.request
from io import StringIO
from time import sleep

import pandas as pd
from nltk.tokenize import sent_tokenize
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

input_file_path = "../press_conferences/pdf/"
output_file_path = "../scraped_data/press_conferences.parquet"


def download_meeting_press_conference():
    with open("../press_conferences/press_conference_dates.txt", "r") as file:
        dates = file.read().splitlines()
    opener = urllib.request.URLopener()
    opener.addheader("User-Agent", "whatever")
    for index, date in enumerate(dates):
        if not index % 5:
            print(index)
        try:
            file_name = "../press_conferences/pdf/" + date + ".pdf"
            print(file_name)
            if not os.path.isfile(file_name):
                post_fix = f"/mediacenter/files/{date}.pdf"
                curr_url = "https://www.federalreserve.gov" + post_fix
                opener.retrieve(curr_url, file_name)
                sleep(3)

        except Exception as e:
            print(e)


def convert_pdf_to_string(file_path):
    output_string = StringIO()
    with open(file_path, "rb") as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return output_string.getvalue()


def convert_title_to_filename(title):
    filename = title.lower()
    filename = filename.replace(" ", "_")
    return filename


def split_to_title_and_page_num(table_of_contents_entry):
    title_and_page_num = table_of_contents_entry.strip()

    title = None
    page_num = None

    if len(title_and_page_num) > 0:
        if title_and_page_num[-1].isdigit():
            i = -2
            while title_and_page_num[i].isdigit():
                i -= 1

            title = title_and_page_num[:i].strip()
            page_num = int(title_and_page_num[i:].strip())

    return title, page_num


def sentence_tokenize(text):
    return sent_tokenize(text.replace("\n", ""))


def save_csv(df, name, location):
    new_name = name.replace(".pdf", "")
    df.to_csv(location + new_name + ".csv")


def get_all_files(input_path, output_path):
    lst = os.listdir(input_path)
    lst.sort()
    df = pd.DataFrame({"date": [], "speaker": [], "sentence": []})
    for f_name in lst:
        file_path = os.path.join(input_path, f_name)
        if os.path.isfile(file_path) and not f_name.startswith("."):
            print(file_path)
            date = pd.Timestamp(re.search(r"\d{8}", f_name).group(0))
            print(date)
            text = convert_pdf_to_string(file_path)
            max_pages = split_to_title_and_page_num(text)[1]  # max number of pages

            # ------------------------- initial cleaning and tokenization process
            t = re.sub(r"(\r\n|\r|\n)", " ", text)
            t = re.sub("  ", " ", t)
            t = re.sub(r"(?<=[.,;])(?=[^\s])", r" ", t)
            sent_tokens = sentence_tokenize(
                text
            )  # tokenize the text in terms of sentences
            sent_tokens = [re.sub(r"\s+", " ", sent) for sent in sent_tokens]

            # ------------------------ cleaning process by removing title and page text in sentences
            temp = sent_tokens[0].split(" ")

            # Ex.
            # May 4, 2022 Chair Powell’s Press Conference PRELIMINARY
            # Transcript of Chair Powell’s Press Conference May 4, 2022 CHAIR POWELL.

            t = temp.index("Transcript")
            press_title = " ".join(
                temp[:t]
            )  # May 4, 2022 Chair Powell’s Press Conference PRELIMINARY
            print(press_title)

            speaker = temp[-2] + " " + (temp[-1])[:-1]  # CHAIR POWELL

            speakers = []
            sentences = []
            print(sent_tokens)
            for i in range(1, len(sent_tokens)):
                s = sent_tokens[i]
                if press_title in s:
                    s = re.sub(press_title, "", s)
                sub = r"Page.+\b{}\b".format(str(max_pages))
                s = re.sub(sub, "", s)
                for j in range(1, max_pages + 1):
                    sub = r"\b{}\b.+\b{}\b".format(str(j), str(max_pages))
                    s = re.sub(sub, "", s)
                    s = s.strip().replace("  ", " ").replace("\n", "")
                if (
                    s.strip().replace(" ", "")[:-1].isupper()
                ):  # CHAIR YELLEN. -> CHAIRYELLEN check if uppercase
                    speaker = s.strip().replace("\n", "")[:-1]  # set the speaker
                    continue
                # -----------------------------
                speakers.append(speaker)
                sentences.append(s)
            select_speakers, select_sentences = zip(
                *[
                    (speaker, sentence)
                    for speaker, sentence in zip(speakers, sentences)
                    if "CHAIR" in speaker and "?" not in sentence
                ]
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "date": date,
                            "speaker": select_speakers[0],
                            "sentence": [select_sentences],
                        }
                    ),
                ]
            )
    df.set_index("date", drop=True).to_parquet(output_path)


if __name__ == "__main__":
    download_meeting_press_conference()
    get_all_files(input_path=input_file_path, output_path=output_file_path)
