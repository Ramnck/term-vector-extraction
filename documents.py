from api import DocumentBase, LoaderBase
import random
import aiohttp
import aiofiles
import xml.etree.ElementTree as ET
import re
import json
import logging
from enum import StrEnum

from pathlib import Path

logger = logging.getLogger(__name__)
rand_terms = "ракета машина смазка установка самолет вертолёт автомат мотоцикл насос инструмент лист дерево обработка рост эволюция".split()


class FipsDoc(DocumentBase):
    def __init__(self, raw) -> None:
        self.raw_json = raw

    @property
    def citations(self) -> list[str]:
        return [i["identity"] for i in self.raw_json["common"]["citated_docs"]]

    @property
    def xml(self):
        tree = ET.fromstring(self.raw_json["description"]["ru"])
        return tree

    @property
    def description(self) -> str:
        json_description = self.raw_json.get("description", {}).get("ru", "")
        if json_description:
            xml = ET.fromstring(json_description)
            return " ".join(i.text for i in xml if i.text is not None)
        else:
            return ""

    @property
    def abstract(self) -> str:
        json_abstract = self.raw_json.get("abstract", {}).get("ru", "")
        if json_abstract:
            xml = ET.fromstring(json_abstract)
            return " ".join(i.text for i in xml if i.text is not None)
        else:
            return ""

    @property
    def claims(self) -> str:
        json_claims = self.raw_json.get("claims", {}).get("ru", "")
        if json_claims:
            xml = ET.fromstring(self.raw_json["claims"]["ru"])
            out = ""
            for i in xml:
                if i.text is not None:
                    out += i.text + " "
                for j in i:
                    if j.text is not None:
                        out += j.text + " "
            return out
        else:
            return ""

    @property
    def text(self) -> str:
        return " ".join([self.abstract, self.claims, self.description])

    @property
    def id(self) -> str:
        temp = self.raw_json["id"]
        return temp[: temp.index("_")]

    @property
    def date(self) -> str:
        return self.raw_json["common"]["publication_date"].replace(".", "")


class FipsAPI(LoaderBase):
    api_url = "https://searchplatform.rospatent.gov.ru/patsearch/v0.2/"

    def __init__(self, api_key) -> None:
        self.api_key = api_key

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _search_query(self, **kwargs) -> dict:
        async with aiohttp.ClientSession() as session:
            data = json.dumps(kwargs)

            async with session.post(
                self.api_url + "search",
                data=data.encode("utf-8"),
                headers=self._headers,
            ) as res:
                return await res.json()

    async def get_doc(self, id: str) -> FipsDoc | None:
        num_of_doc = re.findall(r"\d+", id)[0]

        res = await self._search_query(q=f"PN={num_of_doc}")
        res = res["hits"]

        langs = {hit["snippet"]["lang"]: hit for hit in res}
        res = langs.get("ru", None)
        if res is None:
            res = langs.get("en", None)

        if res is not None:
            res = await self.get_doc_by_id_date(res["id"])
        return res

    async def get_doc_by_id_date(self, id_date: str) -> FipsDoc | None:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.api_url + "docs/" + id_date, headers=self._headers
            ) as res:
                try:
                    return FipsDoc(await res.json())
                except aiohttp.ContentTypeError as ex:
                    logger.error(id_date + " document not found")
                    return None

    async def get_random_doc(self) -> FipsDoc:
        res = await self._search_query(q=random.choice(rand_terms), limit=20)
        doc = random.choice(res["hits"])
        return await self.get_doc_by_id_date(doc["id"])

    async def find_relevant_by_keywords(
        self, kws: list, num_of_docs=20, offset=0
    ) -> list[str]:
        res = await self._search_query(
            q=" OR ".join(kws), offset=offset, limit=num_of_docs
        )
        docs = [
            re.sub(r"[^0-9a-zA-Zа-яА-Я_]", "", i["id"]) for i in res.get("hits", [])
        ]

        return docs


class XMLDoc(DocumentBase):
    def extract_citaions(text: str) -> list[str]:
        # Регулярное выражение для патентной заявки
        pattern = (
            r"\b[A-Z]{2} ?\d+ ?[A-ZА-Я]\d?, ?\d{2,4}[ .,/\\-]\d{2}[ .,/\\-]\d{2,4}\b"
        )

        # Найти все патентные документы в строке
        matches = re.findall(pattern, text)
        out = []
        for match in matches:
            id_date = match.split(", ")
            if len(id_date) == 1:
                id_date = match.split(",")

            if len(id_date) < 2:
                continue

            id = re.sub(r"[^0-9a-zA-Zа-яА-Я]", "", id_date[0])
            date = " ".join(id_date[1:])

            dates = re.findall(r"\d+", date)

            if len(dates[2]) == 4:
                dates = dates[::-1]

            date = "".join(dates)

            out.append(id + "_" + date)
        return out

    class Namespace(StrEnum):
        xmlschema_address = "http://www.wipo.int/standards/XMLSchema/ST96"
        com = f"{{{xmlschema_address}/Common}}"
        pat = f"{{{xmlschema_address}/Patent}}"

    def __init__(self, raw: ET.Element | str) -> None:
        if isinstance(raw, ET.Element):
            self.xml_obj = raw
        else:
            self.xml_obj = ET.fromstring(raw)

    @property
    def _raw_citations_field(self) -> str:
        tag = self.xml_obj.find(XMLDoc.Namespace.pat + "BibliographicData")
        tag = tag.find(XMLDoc.Namespace.pat + "ReferenceCitationBag")
        refs = []
        if tag is None:
            return ""
        for i in tag.findall(XMLDoc.Namespace.pat + "ReferenceCitationFreeFormat"):
            if i.text:
                refs.append(i.text)
            for j in i:
                if j.text:
                    refs.append(j.text)
        citations = " ".join(refs)

        return citations

    @property
    def citations(self) -> list[str]:
        citations = self._raw_citations_field

        return XMLDoc.extract_citaions(citations)

    @property
    def description(self) -> str:
        description = self.xml_obj.find(XMLDoc.Namespace.pat + "Description")
        description = " ".join([i.text for i in description if i.text is not None])
        return description

    @property
    def claims(self) -> str:
        claims_root = self.xml_obj.find(XMLDoc.Namespace.pat + "Claims")

        claims = []

        for claim in claims_root.findall(XMLDoc.Namespace.pat + "Claim"):
            for claim_text in claim.findall(XMLDoc.Namespace.pat + "ClaimText"):
                if claim_text.text is not None:
                    claims.append(claim_text.text)

        return " ".join(claims)

    @property
    def abstract(self) -> str:
        abstracts = []
        for abstract in self.xml_obj.findall(XMLDoc.Namespace.pat + "Abstract"):
            for p in abstract:
                if p.text is not None:
                    abstracts.append(p.text)

        return " ".join(abstracts)

    @property
    def text(self) -> str:
        return " ".join([self.description, self.claims, self.abstract])

    @property
    def id(self) -> str:
        def extract_id(tag: ET.Element) -> str:
            c = tag.find(XMLDoc.Namespace.com + "IPOfficeCode")
            n = tag.find(XMLDoc.Namespace.pat + "PublicationNumber")
            k = tag.find(XMLDoc.Namespace.com + "PatentDocumentKindCode")

            # check if there is a None
            if c is None or n is None or k is None:
                return None
            else:
                return c.text + n.text.lstrip("0") + k.text

        id_temp = extract_id(self.xml_obj)

        if id_temp is None:
            tag = self.xml_obj.find(XMLDoc.Namespace.pat + "BibliographicData")
            tags = tag.findall(XMLDoc.Namespace.pat + "PatentPublicationIdentification")
            for i in tags:
                if i.attrib[XMLDoc.Namespace.pat + "dataFormat"] == "original":
                    tag = i
                    break
            id_temp = extract_id(tag)

        return id_temp

    @property
    def date(self) -> set:
        tag = self.xml_obj.find(XMLDoc.Namespace.com + "PublicationDate")
        if tag is None:
            tag = self.xml_obj.find(XMLDoc.Namespace.pat + "BibliographicData")
            tags = tag.findall(XMLDoc.Namespace.pat + "PatentPublicationIdentification")
            for i in tags:
                if i.attrib[XMLDoc.Namespace.pat + "dataFormat"] == "original":
                    tag = i
                    break
            tag = tag.find(XMLDoc.Namespace.com + "PublicationDate")
        return tag.text

    @property
    def cluster(self) -> set[str]:
        xml = self.xml_obj
        out = set()

        raw_str_xmls = set()

        raw_str_xmls.add(xml)

        citation_docs = xml.find(XMLDoc.Namespace.pat + "DocumentCitationBag")
        if citation_docs is not None:
            raw_str_xmls |= set(
                citation_docs.findall(XMLDoc.Namespace.pat + "DocumentCitation")
            )

        analog_cited_docs = xml.find(XMLDoc.Namespace.pat + "AnalogOfCitationBag")
        if analog_cited_docs is not None:
            raw_str_xmls |= set(
                analog_cited_docs.findall(XMLDoc.Namespace.pat + "AnalogOfCitation")
            )

        analog_docs = xml.find(XMLDoc.Namespace.pat + "DocumentAnalogBag")
        if analog_docs is not None:
            raw_str_xmls |= set(
                analog_docs.findall(XMLDoc.Namespace.pat + "DocumentAnalog")
            )

        for raw_str_xml in raw_str_xmls:
            temp_doc = XMLDoc(raw_str_xml)
            if temp_doc.id_date is not None:
                out.add(temp_doc.id_date)
            out |= set(temp_doc.citations)

        return out


class FileSystem(LoaderBase):
    def __init__(self, path: str | Path) -> None:
        self.init_path = Path(path).absolute()

    async def _open_file(self, path: Path) -> XMLDoc:
        async with aiofiles.open(path) as file:
            doc = XMLDoc(await file.read())
        return doc

    async def get_doc(self, id_date: str) -> XMLDoc | None:
        num_of_doc = re.findall(r"\d+", id_date)[0]
        for file_path in self.init_path.iterdir():
            if file_path.is_dir():
                file_path = next(iter(file_path.iterdir()))
            if num_of_doc in str(file_path):
                return await self._open_file(file_path)
        logger.error(id_date + " document not found")
        return None

    async def get_random_doc(self) -> XMLDoc:
        list_of_files = list(self.init_path.iterdir())
        doc_path = random.choice(list_of_files)
        return self._open_file(doc_path)

    async def find_relevant_by_keywords(self, kws) -> list:
        raise NotImplementedError

    def __aiter__(self):
        self.diriter = self.init_path.iterdir()
        return self

    async def __anext__(self) -> XMLDoc:
        try:
            doc_path = next(self.diriter)
        except StopIteration:
            raise StopAsyncIteration

        if doc_path.is_dir():
            for file in doc_path.iterdir():
                doc_path = file

        return await self._open_file(doc_path)
