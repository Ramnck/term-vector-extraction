import logging
from operator import itemgetter
from typing import Any, Callable, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModel,
    AutoTokenizer,
    LongformerModel,
    LongformerTokenizerFast,
)

from ..base import DocumentBase, EmbedderBase, KeyWordExtractorBase
from ..lexis import clean_ru_text, lemmatize_doc, stopwords_ru_en

logging.getLogger(
    "transformers_modules.jinaai.xlm-roberta-flash-implementation"
).setLevel(logging.ERROR)


def mmr(
    doc_embedding: np.ndarray,
    word_embeddings: np.ndarray,
    words: List[str],
    top_n: int = 5,
    diversity: float = 0.8,
) -> List[Tuple[str, float]]:
    """Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.


    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.

    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances

    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(min(top_n - 1, len(words) - 1)):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(
            word_similarity[candidates_idx][:, keywords_idx], axis=1
        )

        # Calculate MMR
        mmr = (
            1 - diversity
        ) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    # Extract and sort keywords in descending similarity
    keywords = [
        (words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4))
        for idx in keywords_idx
    ]
    keywords = sorted(keywords, key=itemgetter(1), reverse=True)
    return keywords


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class RuLongrormerEmbedder(EmbedderBase):
    def __init__(
        self,
        model: str = "kazzand/ru-longformer-tiny-16384",
    ):
        self.model = LongformerModel.from_pretrained(model)
        self.tokenizer = LongformerTokenizerFast.from_pretrained(model)

    def encode(self, documents: list[str], **kwargs) -> np.ndarray:
        device = "cuda"
        model = self.model.to(device)

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
                output = model(**batch.to(device))
            tensor = output.last_hidden_state[:, 0, :].cpu()
            outputs.append(tensor)

        outputs = torch.cat(outputs)

        return np.array(outputs)


class TransformerEmbedder(EmbedderBase):
    def __init__(self, model, pooling_func: Callable = mean_pooling, device="cuda"):
        self.model = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pooling_func = pooling_func
        self.device = device

    def encode(self, documents: list[str], **kwargs) -> np.ndarray:
        encoded_input = self.tokenizer(
            documents, padding=True, truncation=True, max_length=24, return_tensors="pt"
        )

        if self.model.device.type != self.device:
            self.model = self.model.to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input.to(self.device))

        # Perform pooling. In this case, mean pooling
        sentence_embeddings = self.pooling_func(
            model_output, encoded_input["attention_mask"]
        )
        return sentence_embeddings.cpu().numpy()


class KeyBERTModel:
    def __init__(
        self,
        doc_embedder: EmbedderBase | SentenceTransformer,
        word_embedder: EmbedderBase | SentenceTransformer | None = None,
        max_ngram_size: int = 1,
        doc_prefix: str = "",
        word_prefix: str = "",
        **kwargs,
    ) -> None:

        self.doc_prefix = doc_prefix
        self.word_prefix = word_prefix

        self.word_embedder = (
            word_embedder if word_embedder is not None else doc_embedder
        )
        self.doc_embedder = doc_embedder
        self.encode_kwargs = {"show_progress_bar": False}
        self._max_ngram_size = max_ngram_size

    def extract_doc_embedding(self, doc: str, **kwargs) -> list[list[float]]:
        kwargs.update(self.encode_kwargs)
        return self.doc_embedder.encode([self.doc_prefix + doc], **kwargs)

    def extract_word_embedding(self, words: list[str], **kwargs) -> list[float]:
        kwargs.update(self.encode_kwargs)
        return self.word_embedder.encode(
            [self.word_prefix + word for word in words], **kwargs
        )

    def extract_keywords(
        self,
        document: str,
        top_n: int = 50,
        use_mmr: bool = True,
        diversity: float = 0.7,
        nr_candidates: int = 20,
        **kwargs,
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

        # lemmatized_text = lemmatize_doc(document, stopwords_ru_en)

        # words = [word for word in set(lemmatized_text) if word]

        count = CountVectorizer(
            ngram_range=(1, self._max_ngram_size),
            stop_words=stopwords_ru_en,
            min_df=1,
            token_pattern=r"[А-Яа-яA-Za-zёЁ]+-?[А-Яа-яA-Za-zёЁ]*",
        ).fit([document])

        words = count.get_feature_names_out()

        word_embeddings = self.extract_word_embedding(
            words, **kwargs.get("word_embed_kwargs", {})
        )
        doc_embedding = self.extract_doc_embedding(
            document, **kwargs.get("doc_embed_kwargs", {})
        )

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
    """Keyword extractor which can use SentenceTransformer or your own KeyBERTModel implementation as underlying model."""

    def __init__(
        self,
        model: str | KeyBERTModel | Any | SentenceTransformer,
        method_name: str = "KBRT",
        max_ngram_size: int = 1,
        text_extraction_func: Callable[[DocumentBase], str] = lambda doc: doc.text,
        doc_prefix: str = "",
        word_prefix: str = "",
        **kwargs,
    ) -> None:
        self.name = method_name
        if isinstance(model, KeyBERTModel):
            self.model = model
        elif isinstance(model, (SentenceTransformer, EmbedderBase)):
            self.model = KeyBERTModel(
                model,
                doc_prefix=doc_prefix,
                word_prefix=word_prefix,
                max_ngram_size=max_ngram_size,
            )
        else:
            raise RuntimeError("Error in parsing model")

        self.doc_prefix = doc_prefix
        self.word_prefix = word_prefix

        if "text_extraction_func" in kwargs.keys():
            self.text_extraction_func = kwargs["text_extraction_func"]
            del kwargs["text_extraction_func"]
        else:
            self.text_extraction_func = lambda doc: doc.text
        self.kwargs = kwargs

    def get_keywords(
        self,
        doc: DocumentBase,
        num: int = 50,
        use_mmr: bool = True,
        diversity: float = 0.7,
        **kwargs,
    ) -> list[str]:

        text = self.text_extraction_func(doc)

        kwargs.update(self.kwargs)

        out = self.model.extract_keywords(
            text, top_n=num, use_mmr=use_mmr, diversity=diversity, **kwargs
        )

        return [i[0] for i in out]
