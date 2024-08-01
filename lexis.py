from pymorphy3 import MorphAnalyzer
import nltk.corpus
import re

with open("SimilarStopWords.txt") as file:
    stopwords_iteco = [i for i in file]
stopwords_nltk_ru = list(nltk.corpus.stopwords.words("russian"))
stopwords_ru = list(set(stopwords_nltk_ru) | set(stopwords_iteco))

morph = MorphAnalyzer()


def clean_text_ru(text):
    patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
    return re.sub(patterns, " ", text)


def lemmatize_ru_word(word: str):
    return morph.normal_forms(word)[0]


def lemmatize_doc(doc: str, stopwords: list[str] = []):
    tokens = []
    for token in doc.split():
        token = token.strip()
        if token:
            token = lemmatize_ru_word(token)
            if token not in stopwords:
                tokens.append(token)
    if len(tokens) > 2:
        return tokens
    return None
