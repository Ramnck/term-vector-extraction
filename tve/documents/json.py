"""
This module provides a read-only class for handling JSON patent documents. (JSONDocument)
document can be modified via document.raw_json
"""

import json
import re
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path

from ..base import DocumentBase
from ..lexis import extract_citaions


class JSONDocument(DocumentBase):
    def __init__(self, raw: str | dict) -> None:
        if isinstance(raw, str):
            raw = json.loads(raw)
        self.raw_json = raw

    @cached_property
    def cluster(self) -> list[str]:
        cluster = [self.id_date] + self.raw_json.get("cluster", []) + self.citations
        return list(dict.fromkeys(cluster))

    @cached_property
    def _raw_citation_field(self) -> str:
        biblio = self.raw_json.get("biblio", {})
        bib = biblio.get("ru", None)
        if not bib:
            bib = biblio.get("en", None)
            if not bib:
                return ""
        return bib.get("citations", "")

    @cached_property
    def citations(self) -> list[str]:
        cites = [i["id"] for i in self.raw_json["common"].get("citated_docs", [])]
        cites += extract_citaions(self._raw_citation_field)
        return list(dict.fromkeys(cites))

    @cached_property
    def description(self) -> str:
        json_block = self.raw_json.get("description_cleaned", None)
        if json_block:
            json_text = json_block.get("ru", None)
            if json_text is None:
                json_text = json_block.get("en", None)
            return json_text

        json_block = self.raw_json.get("description", {})
        json_text = json_block.get("ru", None)
        if json_text is None:
            json_text = json_block.get("en", None)

        if json_text:
            xml = ET.fromstring(json_text)
            return " ".join(i.text for i in xml if i.text is not None)
        else:
            return ""

    @cached_property
    def abstract(self) -> str:
        json_block = self.raw_json.get("abstract_cleaned", None)
        if json_block:
            json_text = json_block.get("ru", None)
            if json_text is None:
                json_text = json_block.get("en", None)
            return json_text

        if not json_block:
            json_block = self.raw_json.get("abstract", {})
        json_text = json_block.get("ru", None)
        if json_text is None:
            json_text = json_block.get("en", None)

        if json_text:
            output = ""
            xml = ET.fromstring(json_text)
            for i in xml:
                if i.text:
                    output += i.text + " "
                for j in i:
                    if j.text:
                        output += j.text + " "
            return output
        else:
            return ""

    @cached_property
    def claims(self) -> str:
        json_block = self.raw_json.get("claims_cleaned", None)
        if json_block:
            json_text = json_block.get("ru", None)
            if json_text is None:
                json_text = json_block.get("en", None)
            return json_text

        json_block = self.raw_json.get("claims", {})
        json_text = json_block.get("ru", None)
        if json_text is None:
            json_text = json_block.get("en", None)

        if json_text:
            xml = ET.fromstring(json_text)
            out = []
            for i in xml:
                if i.text is not None:
                    out.append(i.text)
                for j in i:
                    if j.text is not None:
                        out.append(j.text)
            return " ".join(out)
        else:
            return ""

    @cached_property
    def text(self) -> str:
        return " ".join([self.abstract, self.claims, self.description])

    @cached_property
    def id(self) -> str:
        out = self.raw_json.get("id", "")
        out = out.split("_")[0]
        if not out:
            pub_off = self.raw_json["common"]["publishing_office"]
            doc_num = self.raw_json["common"]["document_number"]
            kind = self.raw_json["common"]["kind"]
            out = pub_off + doc_num.lstrip("0") + kind
        return out

    @cached_property
    def date(self) -> str:
        date = self.raw_json["common"]["publication_date"]
        split_date = re.split(r"[ .,/\\-]", date)
        return "".join(split_date)

    def save_file(self, path: Path | str):
        path = Path(path)
        if path.is_dir():
            path /= self.id_date + ".json"

        with open(path, "w+", encoding="utf-8") as file:
            json.dump(self.raw_json, file)

    @classmethod
    def load_file(cls, path: Path | str):
        with open(path, "r", encoding="utf-8") as file:
            text = json.load(file)
            return cls(text)
