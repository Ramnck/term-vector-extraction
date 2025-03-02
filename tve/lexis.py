"""
This module provides stopwords and functions for processing Russian and English text.
#### Lexic functions:
- replace_words_with_custom_function (apply custom function to words in text)
- lemmatize_doc (lemmatize document and exclude stopwords)
- lemmatize_ru_word (lemmatize one Russian word)
- pos_tags_ru/pos_tags_en (get POS tags for Russian/English word)
#### Patent ID functions:
- extract_number (extract number of patent from ID)
- extract_citaions (uses RegEx to extract citaions from text)
#### Utils:
- make_extended_term_vec (accumulate several term vectors into one)
- escape_elasticsearch_query (make query suitable for Elasticsearch)
"""

import re
import time
from itertools import chain, compress, cycle
from pathlib import Path
from typing import Callable

import nltk
import nltk.corpus
from pymorphy3 import MorphAnalyzer

morph = MorphAnalyzer()


def pos_tags_ru(word: str, simplify: bool = False) -> list[str]:
    tags = [i.tag.POS for i in morph.parse(word)]
    if simplify:
        simplify_dict = {
            "NOUN": "NOUN",
            "ADJF": "ADJ",
            "ADJS": "ADJ",
            "COMP": "ADJ",
            "PRTF": "ADJ",
            "PRTS": "ADJ",
            "VERB": "VERB",
            "INFN": "VERB",
            "GRND": "VERB",
            "ADVB": "ADV",
        }
        for tag in tags:
            tag = simplify_dict.get(tag, "")
        tags = list(compress(tags, tags))
    tags = list(dict.fromkeys(tags))
    return tags


def pos_tag_en(word: str, simplify: bool = False) -> str:
    tag = nltk.pos_tag([word], tagset="universal")[0][1]
    if tag == "X":
        tag = "NOUN"
    if simplify:
        simplify_list = ["ADJ", "ADV", "NOUN", "VERB"]
        if tag not in simplify_list:
            tag = ""
    return tag


def escape_elasticsearch_query(query):
    return re.sub(
        '(\+|\=|&&|\|\||\>|\<|\!|\(|\)|\{|\}|\[|\]|\^|"|~|\*|\?|\:|\\\|\/)',
        "\\\\\\1",
        query,
    )


def extract_number(text: str, stripped: bool = True) -> str:
    m = re.findall(r"(\d{2,4}/)?(\d+)(?:/\d+)?", text)
    res = "".join(m[0]) if len(m) > 0 else ""
    res = res.replace("/", "")
    return res.lstrip("0") if stripped else res


def clean_ru_text(text) -> str:
    pattern = re.compile(r"(\w[-\w]*)")
    russian_letters_and_spaces = pattern.findall(text)
    cleaned_string = " ".join(russian_letters_and_spaces)
    return cleaned_string


def lemmatize_ru_word(word: str) -> str:
    return morph.normal_forms(word)[0]


try:
    with open(Path("data") / "SimilarStopWords.txt", encoding="utf-8") as file:
        stopwords_iteco = [i.strip() for i in file]
except FileNotFoundError:
    stopwords_iteco = []

try:
    with open(Path("data") / "MyStopWords.txt", encoding="utf-8") as file:
        stopwords_my = [i.strip() for i in file]
except FileNotFoundError:
    stopwords_my = []
stopwords_nltk_ru = list(nltk.corpus.stopwords.words("russian"))
stopwords_ru = list(set(chain(stopwords_my, stopwords_nltk_ru, stopwords_iteco)))


stopwords_nltk_en = list(nltk.corpus.stopwords.words("english"))
stopwords_en = list(set(stopwords_nltk_en))

stopwords_ru_en = set()
stopwords_ru_en |= set(stopwords_ru)
stopwords_ru_en |= set(stopwords_en)

stopwords_ru_en = list(stopwords_ru_en)


def lemmatize_doc(
    doc: str | list[str], stopwords: list[str] = stopwords_ru_en
) -> list[str]:
    tokens = []

    if isinstance(doc, str):
        doc = doc.split()

    for token in doc:
        match = re.findall(r"([А-Яа-яA-Za-zёЁ]+)(-[А-Яа-яA-Za-zёЁ]+)?", token)
        token = "".join(match[0]) if len(match) > 0 else None
        if token:
            token = lemmatize_ru_word(token)
            if token not in stopwords:
                tokens.append(token)
    return tokens


def replace_words_with_custom_function(
    text: str, custom_function: Callable[[str], str]
):
    # Function to replace a Russian word using the custom function
    def apply_custom_function(match):
        return custom_function(match.group(0))

    # Regex pattern that matches Russian words (sequences of Cyrillic characters)
    cyrillic_pattern = r"\b[а-яА-ЯёЁ]+\b"

    # Use re.sub with the regex pattern
    result = re.sub(cyrillic_pattern, apply_custom_function, text)

    return result


def make_extended_term_vec(
    list_of_vecs: list[list[str]], base_vec: list[str] = [], length: int = 100
) -> list[str]:
    vec = set(base_vec)

    iterator = cycle([cycle(i) for i in list_of_vecs if i])

    timeout = 0
    timeout_val = sum(map(len, list_of_vecs)) * 3

    while len(vec) < length and timeout < timeout_val:
        vec.add(lemmatize_ru_word(next(next(iterator))))
        timeout += 1

    return list(vec)


def extract_citaions(text: str) -> list[str]:
    # Регулярное выражение для патентной заявки
    pattern = r"\b([A-Z]{2}) ?(?:\d{2,4}[/\\])?(\d+)(?:[/\\]\d{1,4})?,? ?\n?([A-ZА-Я]\d?)?,? ?(\d{2,4})[ .,/\-](\d{2})[ .,/\-](\d{2,4})\b"

    delete_pattern = r"(кл. |\n)"
    text = re.sub(delete_pattern, "", text)
    text = text.replace("Авторское свидетельство СССР N", "SU")

    # Найти все патентные документы в строке
    matches = re.findall(pattern, text)

    out = []
    for match_ in matches:
        match = list(match_)
        date = match[-3:]
        if len(match[-1]) == 4:
            date = date[::-1]
        elif len(match[0]) == 4:
            pass
        else:
            if int(date[0]) > time.gmtime(time.time()).tm_year % 100:
                date[0] = "19" + date[0]
            else:
                date[0] = "20" + date[0]

        date = "".join(date)

        id = "".join(match[:3])

        out.append(id + "_" + date)
    return out
