"""
This module provides a class for handling Blank patent documents. (BlankDoc)
They are easy-modifiable, can be inherited from other docs, but cannot be saved.
"""

from ..base import DocumentBase


class BlankDoc(DocumentBase):
    """A class representing a blank document that can inherit attributes from another document."""

    def __init__(self, base_doc: DocumentBase | None = None) -> None:
        """
        Initialize the BlankDoc.

        :param base_doc: An instance of DocumentBase to inherit attributes from, or None if starting fresh.
        """
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
        """
        Get the text of the document.

        :return: The text content of the document.
        """
        return self._text

    @text.setter
    def text(self, value):
        """
        Set the text of the document.

        :param value: The new text content for the document.
        """
        self._text = value

    @property
    def citations(self) -> list[str]:
        """
        Get the citations associated with the document.

        :return: A list of citation strings.
        """
        return self._citations

    @citations.setter
    def citations(self, value):
        """
        Set the citations for the document.

        :param value: A new list of citation strings.
        """
        self._citations = value

    @property
    def cluster(self) -> list[str]:
        """
        Get the cluster associated with the document.

        :return: A list of cluster identifiers.
        """
        return self._cluster

    @cluster.setter
    def cluster(self, value):
        """
        Set the cluster for the document.

        :param value: A new list of cluster identifiers.
        """
        self._cluster = value

    @property
    def id(self) -> str:
        """
        Get the unique identifier of the document.

        :return: The document's ID.
        """
        return self._id

    @id.setter
    def id(self, value):
        """
        Set the unique identifier for the document.

        :param value: The new ID for the document.
        """
        self._id = value

    @property
    def date(self) -> str:
        """
        Get the date associated with the document.

        :return: The date string of the document.
        """
        return self._date

    @date.setter
    def date(self, value):
        """
        Set the date for the document.

        :param value: The new date string for the document.
        """
        self._date = value
