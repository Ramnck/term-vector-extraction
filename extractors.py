from pymorphy3 import MorphAnalyzer
import nltk.corpus
from sklearn.feature_extraction.text import CountVectorizer

from api import KeyWordExtractorBase, DocumentBase

from rake_nltk import Rake
import yake
from summa import keywords
from keybert import KeyBERT
import re
import pke


with open("SimilarStopWords.txt") as file:
    stopwords_iteco = [i for i in file]
stopwords_nltk = list(nltk.corpus.stopwords.words("russian"))
stopwords = list(set(stopwords_nltk) | set(stopwords_iteco))


morph = MorphAnalyzer()


def clean_text(text):
    patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
    return re.sub(patterns, " ", text)


def lemmatize(doc):
    tokens = []
    for token in doc.split():
        if token:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            if token not in stopwords:
                tokens.append(token)
    if len(tokens) > 2:
        return tokens
    return None


class RAKExtractor(KeyWordExtractorBase):
    def __init__(self) -> None:
        self.rake = Rake(
            stopwords=stopwords,
            max_length=1,
            include_repeated_phrases=False,
        )

    def get_keywords(self, doc: DocumentBase, num=50) -> list:
        self.rake.extract_keywords_from_text(doc.text)
        out = self.rake.get_ranked_phrases()
        return out[:num]

    @classmethod
    def get_name(self) -> str:
        return "RAKE"


class YAKExtractor(KeyWordExtractorBase):
    def __init__(self, dedupLim=0.9) -> None:
        self.y = yake.KeywordExtractor(
            lan="ru", n=1, dedupLim=dedupLim, top=50, stopwords=stopwords
        )

    def get_keywords(self, doc: DocumentBase, num=50) -> list:
        out = self.y.extract_keywords(clean_text(doc.text))
        return [i[0] for i in out[:num]]

    @classmethod
    def get_name(self) -> str:
        return "YAKE"


class TextRankExtractor(KeyWordExtractorBase):
    def __init__(self) -> None:
        pass

    def get_keywords(self, doc: DocumentBase, num=50) -> list:
        text_clean = " ".join(lemmatize(doc.text))
        out = keywords.keywords(text_clean, language="russian").split("\n")
        return out[:num]

    @classmethod
    def get_name(self) -> str:
        return "TXTR"


class KeyBERTExtractor(KeyWordExtractorBase):
    def __init__(self) -> None:
        self.model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")

    def get_keywords(self, doc: DocumentBase, num=50) -> list:
        vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            stop_words=[i.encode("utf-8") for i in stopwords],
            encoding="utf-8",
        )

        out = self.model.extract_keywords(
            clean_text(doc.text),
            vectorizer=vectorizer,
            top_n=num,
            use_mmr=True,
        )
        return [i[0] for i in out]

    @classmethod
    def get_name(self) -> str:
        return "KBRT"


class MultipartiteRankExtractor(KeyWordExtractorBase):
    def __init__(self) -> None:
        self.extractor = pke.unsupervised.MultipartiteRank()

    def get_keywords(self, doc: DocumentBase, num=50) -> list:
        self.extractor.load_document(
            clean_text(doc.text), language="ru", stoplist=stopwords
        )
        self.extractor.candidate_selection()
        self.extractor.candidate_weighting(alpha=1.1, threshold=0.74, method="average")
        return [i[0] for i in self.extractor.get_n_best(n=num)]

    @classmethod
    def get_name(self) -> str:
        return "MPTE"
