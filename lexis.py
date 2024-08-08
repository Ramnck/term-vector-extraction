from pymorphy3 import MorphAnalyzer
import nltk.corpus
import re

from typing import Callable
from api import DocumentBase

morph = MorphAnalyzer()


# def clean_ru_text(text) -> str:
# patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-«»–]+"
# return re.sub(patterns, " ", text)


def clean_ru_text(text) -> str:
    pattern = re.compile(r"(\w[-\w]*)")
    russian_letters_and_spaces = pattern.findall(text)
    cleaned_string = " ".join(russian_letters_and_spaces)
    return cleaned_string


def lemmatize_ru_word(word: str) -> str:
    return morph.normal_forms(word)[0]


with open("SimilarStopWords.txt", encoding="utf-8") as file:
    stopwords_iteco = [i.strip() for i in file]
with open("MyStopWords.txt", encoding="utf-8") as file:
    stopwords_my = [i.strip() for i in file]
stopwords_nltk_ru = list(nltk.corpus.stopwords.words("russian"))
stopwords_nltk_en = list(nltk.corpus.stopwords.words("english"))

stopwords_ru = set()
stopwords_ru |= set(stopwords_nltk_en)
for word in stopwords_my:
    stopwords_ru.add(lemmatize_ru_word(word))
for word in stopwords_nltk_ru:
    stopwords_ru.add(lemmatize_ru_word(word))
for word in stopwords_iteco:
    stopwords_ru.add(lemmatize_ru_word(word))
stopwords_ru = list(stopwords_ru)


def lemmatize_doc(doc: str | list[str], stopwords: list[str]) -> str:
    tokens = []

    if isinstance(doc, str):
        doc = doc.split()

    for token in doc:
        token = token.strip()
        if token:
            token = lemmatize_ru_word(token)
            if token not in stopwords:
                tokens.append(token)
    if len(tokens) > 2:
        return " ".join(tokens)
    return None


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
