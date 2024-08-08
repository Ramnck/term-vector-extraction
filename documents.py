from api import DocumentBase, LoaderBase
import random
import aiohttp
import aiofiles
import xml.etree.ElementTree as ET
import re
from enum import StrEnum

from pathlib import Path


rand_terms = "ракета машина смазка установка самолет вертолёт автомат мотоцикл насос инструмент лист дерево обработка рост эволюция".split()


class FipsDoc(DocumentBase):
    def __init__(self, raw) -> None:
        self.raw_json = raw

    @property
    def citations(self) -> list[str]:
        return [i["identity"] for i in self.raw_json["common"]["citated_docs"]]

    def get_xml(self):
        tree = ET.fromstring(self.raw_json["description"]["ru"])
        return tree

    @property
    def text(self) -> str:
        return "\n".join(i.text for i in self.get_xml() if i.text is not None)

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

    async def get_doc(self, id_date: str) -> FipsDoc:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.api_url + "docs/" + id_date, headers=self._headers
            ) as res:
                try:
                    return FipsDoc(await res.json())
                except aiohttp.ContentTypeError as ex:
                    # pprint(res.text)
                    raise Exception("Invalid JSON response")

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _search_query(self, **kwargs) -> dict:
        async with aiohttp.ClientSession() as session:
            data = ""
            for key, value in kwargs.items():
                data += '"{0}": {1}, '.format(
                    key, '"' + value + '"' if type(value) == str else value
                )
            data = "{" + data[:-2] + "}"

            async with session.post(
                self.api_url + "search",
                data=data.encode("utf-8"),
                headers=self._headers,
            ) as res:
                return await res.json()

    async def get_random_doc(self) -> FipsDoc:
        res = await self._search_query(q=random.choice(rand_terms), limit=20)
        doc = random.choice(res["hits"])
        return await self.get_doc(doc["id"])

    async def find_relevant_by_keywords(self, kws: list, num_of_docs=20) -> list:
        res = await self._search_query(q=" OR ".join(kws), offset=1, limit=num_of_docs)
        docs = [i["id"][: i["id"].index("_")] for i in res.get("hits", [])]

        return docs


class XMLDoc(DocumentBase):
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
    def citations(self) -> list[str]:
        tag = self.xml_obj.find(XMLDoc.Namespace.pat + "BibliographicData")
        tag = tag.find(XMLDoc.Namespace.pat + "ReferenceCitationBag")
        refs = []
        if tag is None:
            return []
        for i in tag.findall(XMLDoc.Namespace.pat + "ReferenceCitationFreeFormat"):
            if i.text:
                refs.append(i.text)
            for j in i:
                if j.text:
                    refs.append(j.text)
        citations = " ".join(refs)

        pattern = r"\b[A-Z]{2} ?\d+ ?[A-ZА-Я]\d?\b"
        matches = re.findall(pattern, citations)

        return [x.replace(" ", "") for x in matches]

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
        return "\n".join([self.description, self.claims, self.abstract])

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
            id_temp = extract_id(
                self.xml_obj.find(XMLDoc.Namespace.pat + "BibliographicData")
            )

        return id_temp

    @property
    def date(self) -> set:
        tag = self.xml_obj.find(XMLDoc.Namespace.com + "PublicationDate")
        return tag.text

    @property
    def cluster(self) -> set:
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
            out.add(temp_doc.id)
            out |= set(temp_doc.citations)

        return out


class FileSystem(LoaderBase):
    def __init__(self, path: str | Path) -> None:
        self.init_path = Path(path).absolute()

    async def _open_file(self, path: Path) -> XMLDoc:
        async with aiofiles.open(path) as file:
            doc = XMLDoc(await file.read())
        return doc

    async def get_doc(self, id: str) -> XMLDoc:
        for file_path in self.init_path.iterdir():
            if id in str(file_path):
                return await self._open_file(file_path)
        raise FileNotFoundError

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
