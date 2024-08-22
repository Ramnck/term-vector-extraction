import re
from itertools import chain, cycle
from pathlib import Path
from typing import Callable

import nltk.corpus
from pymorphy3 import MorphAnalyzer

from api import DocumentBase

morph = MorphAnalyzer()


def extract_number(text: str) -> str:
    return re.findall(r"\d+", text)[0]


def clean_ru_text(text) -> str:
    pattern = re.compile(r"(\w[-\w]*)")
    russian_letters_and_spaces = pattern.findall(text)
    cleaned_string = " ".join(russian_letters_and_spaces)
    return cleaned_string


def lemmatize_ru_word(word: str) -> str:
    return morph.normal_forms(word)[0]


with open(Path("data") / "SimilarStopWords.txt", encoding="utf-8") as file:
    stopwords_iteco = [i.strip() for i in file]
with open(Path("data") / "MyStopWords.txt", encoding="utf-8") as file:
    stopwords_my = [i.strip() for i in file]
stopwords_nltk_ru = list(nltk.corpus.stopwords.words("russian"))
stopwords_nltk_en = list(nltk.corpus.stopwords.words("english"))

stopwords_ru = set()
stopwords_ru |= set(stopwords_nltk_en)
for word in chain(stopwords_my, stopwords_nltk_ru, stopwords_iteco):
    stopwords_ru.add(lemmatize_ru_word(word))
stopwords_ru = list(stopwords_ru)


def lemmatize_doc(
    doc: str | list[str], stopwords: list[str] = stopwords_ru
) -> list[str]:
    tokens = []

    if isinstance(doc, str):
        doc = doc.split()

    for token in doc:
        token = "".join(re.findall(r"[А-Яа-яA-Za-z0-9]+", token))
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

    while len(vec) < length:
        vec.add(lemmatize_ru_word(next(next(iterator))))
        timeout += 1
        if timeout > timeout_val:
            break

    return list(vec)
