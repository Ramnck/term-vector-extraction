from ..base import DocumentBase


class BlankDoc(DocumentBase):
    def __init__(self, base_doc: DocumentBase | None = None) -> None:
        if base_doc is None:
            self._text = None
            self._citations = None
            self._cluster = None
            self._id = None
            self._date = None
        else:
            self._text = base_doc.text
            self._citations = base_doc.citations
            self._cluster = base_doc.cluster
            self._id = base_doc.id
            self._date = base_doc.date

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    @property
    def citations(self) -> list[str]:
        return self._citations

    @citations.setter
    def citations(self, value):
        self._citations = value

    @property
    def cluster(self) -> list[str]:
        return self._cluster

    @cluster.setter
    def cluster(self, value):
        self._cluster = value

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def date(self) -> str:
        return self._date

    @date.setter
    def date(self, value):
        self._date = value
