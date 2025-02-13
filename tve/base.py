"""
This is a module which contains all base interfaces for the TVE project.
#### Interfaces:
- DocumentBase: handling patent documents
- KeyWordExtractorBase: extracting keywords
- LoaderBase: loading and searching patent documents (by ID or keywords)
- EmbedderBase: extracting embeddings from document
- TranslatorBase: translating term-vectors or expanding them (using LLMs) 
"""

from abc import ABC

import numpy as np


class DocumentBase(ABC):
    def __init__(self) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        raise self.text

    @property
    def citations(self) -> list[str]:
        raise NotImplementedError

    @property
    def cluster(self) -> list[str]:
        raise NotImplementedError

    @property
    def text(self) -> str:
        raise NotImplementedError

    @property
    def id(self) -> str:
        raise NotImplementedError

    @property
    def date(self) -> str:
        raise NotImplementedError

    @property
    def id_date(self) -> str:
        if self.id is None or self.date is None:
            return None
        return self.id + "_" + self.date

    def __hash__(self) -> int:
        return hash(self.id)


class KeyWordExtractorBase(ABC):
    _name = "NOT_IMPL"

    def __init__(self, max_ngram_size: int = 1) -> None:
        raise NotImplementedError

    def get_keywords(self, doc: DocumentBase, **kwargs) -> list[str]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, n: str):
        self._name = n


class LoaderBase(ABC):
    def __init__(self) -> None:
        raise NotImplementedError

    # Get document by ID
    async def get_doc(self, id: str) -> DocumentBase | None:
        raise NotImplementedError

    async def get_random_doc(self) -> DocumentBase:
        raise NotImplementedError

    async def find_relevant_by_keywords(
        self, kws: list[str], num_of_docs: int, offset: int, timeout: int
    ) -> list[str]:
        raise NotImplementedError


class EmbedderBase(ABC):
    def __init__(self, model):
        raise NotImplementedError

    def embed(self, documents: list[str], **kwargs) -> np.ndarray:
        raise NotImplementedError


class TranslatorBase(ABC):
    _name = "NOT_IMPL"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def translate_list(
        self,
        words: list[str],
        from_lang: str = "ru",
        to_lang: str = "en",
        num_of_suggestions: int = 2,
        **kwargs,
    ) -> list[str]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, n: str):
        self._name = n
