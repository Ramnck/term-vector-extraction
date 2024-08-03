from api import KeyWordExtractorBase, DocumentBase, EmbedderBase
from lexis import clean_ru_text, lemmatize_doc, stopwords_ru

from keybert import KeyBERT

import torch
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from transformers import LongformerModel, LongformerTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from keybert._mmr import mmr
from keybert._maxsum import max_sum_distance
from typing import Any


class RuLongrormerEmbedder(EmbedderBase):
    def __init__(
        self,
        model: str = "kazzand/ru-longformer-tiny-16384",
    ):
        self.model = LongformerModel.from_pretrained(model)
        self.tokenizer = LongformerTokenizerFast.from_pretrained(model)

    def embed(self, documents: list[str], **kwargs) -> np.ndarray:
        device = "cuda"
        self.model.to(device)

        batches = [
            self.tokenizer(document, return_tensors="pt") for document in documents
        ]

        # set global attention for cls token
        outputs = []
        for batch in batches:
            global_attention_mask = [
                [
                    1 if token_id == self.tokenizer.cls_token_id else 0
                    for token_id in input_ids
                ]
                for input_ids in batch["input_ids"]
            ]

            # add global attention mask to batch
            batch["global_attention_mask"] = torch.tensor(global_attention_mask)

            with torch.no_grad():
                output = self.model(**batch.to(device))
            tensor = output.last_hidden_state[:, 0, :].cpu()
            outputs.append(tensor)

        outputs = torch.cat(outputs)

        return np.array(outputs)


class KeyBERTModel:
    def __init__(
        self, doc_embedder: EmbedderBase, word_embedder: EmbedderBase | None = None
    ) -> None:
        self.word_embedder = (
            word_embedder if word_embedder is not None else doc_embedder
        )
        self.doc_embedder = doc_embedder

    def extract_doc_embedding(self, doc: str) -> list[list[float]]:
        return self.doc_embedder.embed([doc])

    def extract_word_embeddings(self, words: list[str]) -> list[list[float]]:
        return self.word_embedder.embed(words)

    def extract_keywords(
        self,
        document: str,
        top_n: int = 50,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        **kwargs
    ) -> list[tuple[str, float]]:
        """
        Arguments:
            document: str
            top_n: Return the top n keywords/keyphrases
            use_maxsum: Whether to use Max Sum Distance for the selection
                        of keywords/keyphrases.
            use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the
                     selection of keywords/keyphrases.
            diversity: The diversity of the results between 0 and 1 if `use_mmr`
                       is set to True.
            nr_candidates: Whether to use Maximal Marginal Relevance (MMR) for the
                     selection of keywords/keyphrases.
        """

        lemmatized_text = lemmatize_doc(document, stopwords_ru)

        words = [word for word in set(lemmatized_text.split()) if word]

        word_embeddings = self.extract_word_embeddings(words)
        doc_embedding = self.extract_doc_embedding(document)

        # Maximal Marginal Relevance (MMR)
        if use_mmr:
            keywords = mmr(
                doc_embedding,
                word_embeddings,
                words,
                top_n,
                diversity,
            )
        # Cosine-based keyword extraction
        else:
            distances = cosine_similarity(doc_embedding, word_embeddings)
            keywords = [
                (words[index], round(float(distances[0][index]), 4))
                for index in distances.argsort()[0][-top_n:]
            ][::-1]

        return keywords


class KeyBERTExtractor(KeyWordExtractorBase):
    def __init__(
        self,
        model: str | KeyBERTModel | Any = "paraphrase-multilingual-MiniLM-L12-v2",
        method_name: str = "KBRT",
    ) -> None:
        self.method_name = method_name
        if isinstance(model, KeyBERTModel):
            self.model = model
        else:
            self.model = KeyBERT(model)

    def get_keywords(self, doc: DocumentBase, num=50, **kwargs) -> list:
        text_extraction_func = kwargs.get(
            "text_extraction_func", lambda doc: clean_ru_text(doc.text)
        )

        text = text_extraction_func(doc)

        out = self.model.extract_keywords(
            text,
            top_n=num,
            use_mmr=kwargs.get("use_mmr", False),
            use_maxsum=kwargs.get("use_maxsum", False),
        )

        return [i[0] for i in out]

    def get_name(self) -> str:
        return self.method_name
