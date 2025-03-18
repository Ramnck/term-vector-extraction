import yake

from ..base import DocumentBase, KeyWordExtractorBase
from ..lexis import lemmatize_ru_word, replace_words_with_custom_function, stopwords_ru


class YAKExtractor(KeyWordExtractorBase):
    def __init__(self, dedupLim=0.9, max_ngram_size: int = 1) -> None:
        self.model = None
        self.name = "YAKE"
        self.y = yake.KeywordExtractor(
            lan="ru",
            n=max_ngram_size,
            dedupLim=dedupLim,
            top=200,
            stopwords=stopwords_ru,
        )

    def get_keywords(
        self, doc: DocumentBase, num: int = 50, with_scores: bool = False, **kwargs
    ) -> list[str]:
        cleaned_text = replace_words_with_custom_function(doc.text, lemmatize_ru_word)
        out = self.y.extract_keywords(cleaned_text)[:num]
        return out if with_scores else [x[0] for x in out]
