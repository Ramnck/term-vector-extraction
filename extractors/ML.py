from api import KeyWordExtractorBase, DocumentBase
from keybert import KeyBERT
from keybert.backend._base import BaseEmbedder
import torch
import numpy as np
from transformers import LongformerModel, LongformerTokenizerFast

from lexis import clean_ru_text, lemmatize_doc, stopwords_ru
from sklearn.feature_extraction.text import CountVectorizer


class RuLongrormerModel(BaseEmbedder):
    def __init__(
        self,
        embedding_model: str = "kazzand/ru-longformer-tiny-16384",
        word_embedding_model=None,
    ):
        self.model = LongformerModel.from_pretrained(embedding_model)
        self.tokenizer = LongformerTokenizerFast.from_pretrained(embedding_model)

    def embed(self, documents: list[str] | str, verbose: bool = False) -> np.ndarray:
        device = "cuda"
        self.model.to(device)

        if isinstance(documents, str):
            documents = [documents]

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


class KeyBERTExtractor(KeyWordExtractorBase):
    def __init__(
        self,
        model: str | BaseEmbedder = "paraphrase-multilingual-MiniLM-L12-v2",
        method_name: str = "KBRT",
    ) -> None:
        self.method_name = method_name
        self.model = KeyBERT(model=model)

    def get_keywords(self, doc: DocumentBase, num=50, **kwargs) -> list:
        vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            stop_words=[i.encode("utf-8") for i in stopwords_ru],
            encoding="utf-8",
        )

        text_extraction_func = kwargs.get(
            "text_extraction_func", lambda doc: clean_ru_text(doc.text)
        )

        text = text_extraction_func(doc)

        out = self.model.extract_keywords(
            text,
            vectorizer=vectorizer,
            top_n=num,
            use_mmr=kwargs.get("use_mmr", False),
            use_maxsum=kwargs.get("use_maxsum", False),
        )
        return [i[0] for i in out]

    def get_name(self) -> str:
        return self.method_name
