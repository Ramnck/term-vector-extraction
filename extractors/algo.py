from api import KeyWordExtractorBase, DocumentBase
from lexis import (
    stopwords_ru,
    clean_ru_text,
    lemmatize_doc,
    lemmatize_ru_word,
    replace_words_with_custom_function,
)
# from rake_nltk import Rake
import yake
# from summa import keywords
# import pke


# class RAKExtractor(KeyWordExtractorBase):
#     def __init__(self) -> None:
#         self.model = None
#         self.method_name = "RAKE"
#         self.rake = Rake(
#             stopwords=stopwords_ru,
#             max_length=1,
#             include_repeated_phrases=False,
#         )

#     def get_keywords(self, doc: DocumentBase, num=50, **kwargs) -> list:
#         self.rake.extract_keywords_from_text(doc.text)
#         out = self.rake.get_ranked_phrases()
#         return out[:num]

#     def get_name(self) -> str:
#         return self.method_name


class YAKExtractor(KeyWordExtractorBase):
    def __init__(self, dedupLim=0.9) -> None:
        self.model = None
        self.method_name = "YAKE"
        self.y = yake.KeywordExtractor(
            lan="ru", n=1, dedupLim=dedupLim, top=50, stopwords=stopwords_ru
        )

    def get_keywords(self, doc: DocumentBase, num=50, **kwargs) -> list:
        cleaned_text = replace_words_with_custom_function(doc.text, lemmatize_ru_word)
        out = self.y.extract_keywords(cleaned_text)
        return [i[0] for i in out[:num]]

    def get_name(self) -> str:
        return self.method_name


# class TextRankExtractor(KeyWordExtractorBase):
#     def __init__(self) -> None:
#         self.method_name = "TXTR"
#         self.model = None
#         pass

#     def get_keywords(self, doc: DocumentBase, num=50, **kwargs) -> list:
#         text_clean = lemmatize_doc(doc.text, stopwords_ru)
#         out = keywords.keywords(text_clean, language="russian").split("\n")
#         return out[:num]

#     def get_name(self) -> str:
#         return self.method_name


# class MultipartiteRankExtractor(KeyWordExtractorBase):
#     def __init__(self) -> None:
#         self.method_name = "MPTE"
#         self.model = None
#         self.extractor = pke.unsupervised.MultipartiteRank()

#     def get_keywords(self, doc: DocumentBase, num=50, **kwargs) -> list:
#         self.extractor.load_document(
#             clean_ru_text(doc.text), language="ru", stoplist=stopwords_ru
#         )
#         self.extractor.candidate_selection()
#         self.extractor.candidate_weighting(alpha=1.1, threshold=0.74, method="average")
#         return [i[0] for i in self.extractor.get_n_best(n=num)]

#     def get_name(self) -> str:
#         return self.method_name
