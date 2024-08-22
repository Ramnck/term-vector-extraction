import yake

from api import DocumentBase, KeyWordExtractorBase
from lexis import (
    clean_ru_text,
    lemmatize_doc,
    lemmatize_ru_word,
    replace_words_with_custom_function,
    stopwords_ru,
)


class YAKExtractor(KeyWordExtractorBase):
    def __init__(self, dedupLim=0.9) -> None:
        self.model = None
        self.method_name = "YAKE"
        self.y = yake.KeywordExtractor(
            lan="ru", n=1, dedupLim=dedupLim, top=50, stopwords=stopwords_ru
        )

    def get_keywords(self, doc: DocumentBase, num=50, **kwargs) -> list[str]:
        cleaned_text = replace_words_with_custom_function(doc.text, lemmatize_ru_word)
        out = self.y.extract_keywords(cleaned_text)
        return [i[0] for i in out[:num]]

    def get_name(self) -> str:
        return self.method_name
