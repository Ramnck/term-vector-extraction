"""
This module provides a read-only class for handling XML patent documents. (XMLDocument)
document can be modified via document.raw_xml
"""

import re
import xml.etree.ElementTree as ET
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from ..base import DocumentBase
from ..lexis import extract_citaions


class XMLDocument(DocumentBase):

    class Namespace(StrEnum):
        xmlschema_address = "http://www.wipo.int/standards/XMLSchema/ST96"
        com = f"{{{xmlschema_address}/Common}}"
        pat = f"{{{xmlschema_address}/Patent}}"

    def __init__(self, raw: ET.Element | str) -> None:
        if isinstance(raw, ET.Element):
            self.raw_xml = raw
        else:
            self.raw_xml = ET.fromstring(raw)

    @cached_property
    def _freeformat_citations_field(self) -> str:
        tag = self.raw_xml.find(XMLDocument.Namespace.pat + "BibliographicData")
        tag = tag.find(XMLDocument.Namespace.pat + "ReferenceCitationBag")
        refs = []
        if tag is None:
            return ""
        for i in tag.findall(XMLDocument.Namespace.pat + "ReferenceCitationFreeFormat"):
            if i.text:
                refs.append(i.text)
            for j in i:
                if j.text:
                    refs.append(j.text)
        citations = " ".join(refs)

        return citations

    @cached_property
    def citations(self) -> list[str]:
        citations_free = self._freeformat_citations_field

        citations = extract_citaions(citations_free)

        if len(citations) == 0:
            tag = self.raw_xml.find(XMLDocument.Namespace.pat + "BibliographicData")
            tag = tag.find(XMLDocument.Namespace.pat + "ReferenceCitationBag")
            if tag is not None:
                for ref_cite in tag.findall(
                    XMLDocument.Namespace.pat + "ReferenceCitation"
                ):
                    pat_cite = ref_cite.find(
                        XMLDocument.Namespace.com + "PatentCitation"
                    )
                    if pat_cite is not None:
                        cited_doc_id = pat_cite.find(
                            XMLDocument.Namespace.com
                            + "CitedPatentDocumentIdentification"
                        )
                        if cited_doc_id is not None:
                            pat_off = cited_doc_id.find(
                                XMLDocument.Namespace.com + "IPOfficeCode"
                            )
                            doc_num = cited_doc_id.find(
                                XMLDocument.Namespace.com + "DocumentNumber"
                            )
                            kind = cited_doc_id.find(
                                XMLDocument.Namespace.com + "PatentDocumentKindCode"
                            )
                            date = cited_doc_id.find(
                                XMLDocument.Namespace.com + "PatentDocumentDate"
                            )

                            if (
                                pat_off is not None
                                and doc_num is not None
                                and kind is not None
                                and date is not None
                            ):
                                cite_id = (
                                    pat_off.text
                                    + doc_num.text
                                    + kind.text
                                    + "_"
                                    + date.text
                                )
                                citations.append(cite_id)

        return citations

    @cached_property
    def description(self) -> str:
        description = self.raw_xml.find(XMLDocument.Namespace.pat + "Description")
        if description is None:
            return ""
        description = " ".join([i.text for i in description if i.text is not None])
        return description

    @cached_property
    def claims(self) -> str:
        claims_root = self.raw_xml.find(XMLDocument.Namespace.pat + "Claims")

        if claims_root is None:
            return ""

        claims = []

        for claim in claims_root.findall(XMLDocument.Namespace.pat + "Claim"):
            for claim_text in claim.findall(XMLDocument.Namespace.pat + "ClaimText"):
                if claim_text.text is not None:
                    claims.append(claim_text.text)

        return " ".join(claims)

    @cached_property
    def abstract(self) -> str:
        abstracts = []
        for abstract in self.raw_xml.findall(XMLDocument.Namespace.pat + "Abstract"):
            for p in abstract:
                if p.text is not None:
                    abstracts.append(p.text)

        return " ".join(abstracts)

    @cached_property
    def text(self) -> str:
        return " ".join([self.description, self.claims, self.abstract])

    @cached_property
    def id(self) -> str:
        def extract_id(tag: ET.Element) -> str:
            c = tag.find(XMLDocument.Namespace.com + "IPOfficeCode")
            n = tag.find(XMLDocument.Namespace.pat + "PublicationNumber")
            k = tag.find(XMLDocument.Namespace.com + "PatentDocumentKindCode")

            # check if there is a None
            if c is None or n is None or k is None:
                return None
            else:
                return c.text + n.text.lstrip("0") + k.text

        id_temp = extract_id(self.raw_xml)

        if id_temp is None:
            tag = self.raw_xml.find(XMLDocument.Namespace.pat + "BibliographicData")
            tags = tag.findall(
                XMLDocument.Namespace.pat + "PatentPublicationIdentification"
            )
            for i in tags:
                if i.attrib[XMLDocument.Namespace.pat + "dataFormat"] == "original":
                    tag = i
                    break
            id_temp = extract_id(tag)

        return id_temp

    @cached_property
    def date(self) -> str:
        tag = self.raw_xml.find(XMLDocument.Namespace.com + "PublicationDate")
        if tag is None:
            tag = self.raw_xml.find(XMLDocument.Namespace.pat + "BibliographicData")
            tags = tag.findall(
                XMLDocument.Namespace.pat + "PatentPublicationIdentification"
            )
            for i in tags:
                if i.attrib[XMLDocument.Namespace.pat + "dataFormat"] == "original":
                    tag = i
                    break
            tag = tag.find(XMLDocument.Namespace.com + "PublicationDate")
        return tag.text

    @cached_property
    def cluster(self) -> list[str]:
        xml = self.raw_xml
        out = list()

        raw_str_xmls = list()

        raw_str_xmls.append(xml)

        citation_docs = xml.find(XMLDocument.Namespace.pat + "DocumentCitationBag")
        if citation_docs is not None:
            raw_str_xmls += citation_docs.findall(
                XMLDocument.Namespace.pat + "DocumentCitation"
            )

        analog_cited_docs = xml.find(XMLDocument.Namespace.pat + "AnalogOfCitationBag")
        if analog_cited_docs is not None:
            raw_str_xmls += analog_cited_docs.findall(
                XMLDocument.Namespace.pat + "AnalogOfCitation"
            )

        analog_docs = xml.find(XMLDocument.Namespace.pat + "DocumentAnalogBag")
        if analog_docs is not None:
            raw_str_xmls += analog_docs.findall(
                XMLDocument.Namespace.pat + "DocumentAnalog"
            )

        raw_str_xmls = list(dict.fromkeys(raw_str_xmls))

        for raw_str_xml in raw_str_xmls:
            temp_doc = XMLDocument(raw_str_xml)
            if temp_doc.id_date is not None:
                out.append(temp_doc.id_date)
            out += temp_doc.citations

        return list(dict.fromkeys(out))

    def save_file(self, path: Path | str):
        path = Path(path)
        if path.is_dir():
            path /= self.id_date + ".xml"

        with open(path, "w+", encoding="utf-8") as file:
            string = ET.tostring(self.raw_xml, encoding="unicode")
            file.write(string)

    @classmethod
    def load_file(cls, path: Path | str):
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()
            return cls(text)
