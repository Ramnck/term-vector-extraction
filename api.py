from abc import ABC


class DocumentBase(ABC):
    def __init__(self) -> None:
        raise NotImplementedError

    @property
    def citations(self) -> list:
        raise NotImplementedError

    @property
    def text(self) -> str:
        raise NotImplementedError

    @property
    def id(self) -> str:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError


class KeyWordExtractorBase(ABC):
    def __init__(self) -> None:
        raise NotImplementedError

    def get_keywords(self, doc: DocumentBase) -> list:
        raise NotImplementedError

    def get_name(self) -> str:
        return "NOT_IMPL"


class LoaderBase(ABC):
    def __init__(self) -> None:
        raise NotImplementedError

    # Get document by ID
    async def get_doc(self, id: str) -> DocumentBase:
        raise NotImplementedError

    async def get_random_doc(self) -> DocumentBase:
        raise NotImplementedError

    async def find_relevant_by_keywords(self, kws) -> list:
        raise NotImplementedError

    # Iterate over documents
    def __aiter__(self):
        raise NotImplementedError

    async def __anext__(self) -> DocumentBase:
        raise NotImplementedError
