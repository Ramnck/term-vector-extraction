from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords

from api import KeyWordExtractorBase, DocumentBase, clean_text

from rake_nltk import Rake
import yake
from summa import keywords
from keybert import KeyBERT


with open("SimilarStopWords.txt") as file:
    stopwords_iteco = [i for i in file]
stopwords_ru = list(stopwords.words("russian"))
stopwords = list(set(stopwords_ru) | set(stopwords_iteco))

morph = MorphAnalyzer()


def lemmatize(doc):
    tokens = []
    for token in doc.split():
        if token and token not in stopwords:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens.append(token)
    if len(tokens) > 2:
        return tokens
    return None


class RAKExtractor(KeyWordExtractorBase):
    def __init__(self) -> None:
        self.rake = Rake(stopwords=stopwords, max_length=1)

    def get_keywords(self, doc: DocumentBase, num=30) -> list:
        self.rake.extract_keywords_from_text(doc.text)
        out = self.rake.get_ranked_phrases()
        return out[:num]

    @classmethod
    def get_name(self) -> str:
        return "RAKE"


class YAKExtractor(KeyWordExtractorBase):
    def __init__(self, dedupLim=0.3) -> None:
        self.y = yake.KeywordExtractor(lan="ru", n=1, dedupLim=dedupLim, top=50)

    def get_keywords(self, doc: DocumentBase, num=30) -> list:
        out = self.y.extract_keywords(clean_text(doc.text))
        return [i[0] for i in out[:num]]

    @classmethod
    def get_name(self) -> str:
        return "YAKE"


class TextRankExtractor(KeyWordExtractorBase):
    def __init__(self) -> None:
        pass

    def get_keywords(self, doc: DocumentBase, num=30) -> list:
        text_clean = ""
        for i in doc.text.split():
            if i not in stopwords:
                text_clean += i + " "
        out = keywords.keywords(text_clean, language="russian").split("\n")
        return out[:num]

    @classmethod
    def get_name(self) -> str:
        return "TXTR"


class KeyBERTExtractor(KeyWordExtractorBase):
    def __init__(self) -> None:
        self.model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")

    def get_keywords(self, doc: DocumentBase, num=30) -> list:
        out = self.model.extract_keywords(doc.text, keyphrase_ngram_range=(1, 1))
        return [i[0] for i in out[:num]]

    @classmethod
    def get_name(self) -> str:
        return "KBRT"
