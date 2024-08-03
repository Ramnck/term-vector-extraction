from pymorphy3 import MorphAnalyzer
import nltk.corpus
import re


morph = MorphAnalyzer()


# def clean_ru_text(text) -> str:
# patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-«»–]+"
# return re.sub(patterns, " ", text)


def clean_ru_text(text) -> str:
    pattern = re.compile(r"[а-яА-ЯёЁ]+(?:-[а-яА-ЯёЁ]+)*|\s+")
    russian_letters_and_spaces = pattern.findall(text)
    cleaned_string = "".join(russian_letters_and_spaces)
    return cleaned_string


def lemmatize_ru_word(word: str) -> str:
    return morph.normal_forms(word)[0]


with open("SimilarStopWords.txt", encoding="utf-8") as file:
    stopwords_iteco = [i.strip() for i in file]
with open("MyStopWords.txt", encoding="utf-8") as file:
    stopwords_my = [i.strip() for i in file]
stopwords_nltk_ru = list(nltk.corpus.stopwords.words("russian"))

stopwords_ru = set()
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
