import asyncio
import json
import logging
import random
import re
import xml.etree.ElementTree as ET
from enum import StrEnum
from itertools import compress, product
from pathlib import Path

import aiofiles
import aiohttp
from elasticsearch7 import AsyncElasticsearch, Elasticsearch, NotFoundError

from api import DocumentBase, LoaderBase
from lexis import extract_number
from utils import ForgivingTaskGroup

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

es_logger = logging.getLogger("elasticsearch")
es_logger.setLevel(logging.WARN)
rand_terms = "ракета машина смазка установка самолет вертолёт автомат мотоцикл насос инструмент лист дерево обработка рост эволюция".split()


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


class FipsDoc(DocumentBase):
    def __init__(self, raw) -> None:
        self.raw_json = raw

    @property
    def cluster(self) -> list[str]:
        return [self.id_date]

    @property
    def citations(self) -> list[str]:
        return [i["id"] for i in self.raw_json["common"]["citated_docs"]]

    @property
    def description(self) -> str:
        json_block = self.raw_json.get("description", {})
        json_text = json_block.get("ru", None)
        if json_text is None:
            json_text = json_block.get("en", None)

        if json_text:
            xml = ET.fromstring(json_text)
            return " ".join(i.text for i in xml if i.text is not None)
        else:
            return ""

    @property
    def abstract(self) -> str:
        json_block = self.raw_json.get("abstract", {})
        json_text = json_block.get("ru", None)
        if json_text is None:
            json_text = json_block.get("en", None)

        if json_text:
            xml = ET.fromstring(json_text)
            return " ".join(i.text for i in xml if i.text is not None)
        else:
            return ""

    @property
    def claims(self) -> str:
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

    @property
    def text(self) -> str:
        return " ".join([self.abstract, self.claims, self.description])

    @property
    def id(self) -> str:
        out = self.raw_json.get("id", "")
        out = out.split("_")[0]
        if not out:
            pub_off = self.raw_json["common"]["publishing_office"]
            doc_num = self.raw_json["common"]["document_number"]
            kind = self.raw_json["common"]["kind"]
            out = pub_off + doc_num.lstrip("0") + kind
        return out

    @property
    def date(self) -> str:
        date = self.raw_json["common"]["publication_date"]
        split_date = re.split(r"[ .,/\\-]", date)
        return "".join(split_date)


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
            timeout = kwargs.get("timeout", 30)
            if "timeout" in kwargs.keys():
                del kwargs["timeout"]

            data = json.dumps(kwargs, ensure_ascii=False)

            async with session.post(
                self.api_url + "search",
                data=data.encode("utf-8"),
                headers=self._headers,
                timeout=timeout,
            ) as res:
                try:
                    return await res.json()
                except aiohttp.ContentTypeError as ex:
                    logger.debug(await res.text())
                    return {}

    async def get_doc(self, id: str, timeout: int = 10) -> FipsDoc | None:
        pub_office = re.search(r"[A-Z]{2}", id)
        num_of_doc = extract_number(id).lstrip("0")

        request = {"q": f"PN={num_of_doc}", "timeout": timeout, "limit": 3}

        if pub_office is not None:
            request["filter"] = {"country": {"values": [pub_office.group()]}}

        res = await self._search_query(**request)

        if error := res.get("error"):
            logger.error(error)
            return None

        res = res.get("hits", [])

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
                self.api_url + "docs/" + id_date, headers=self._headers, timeout=5
            ) as res:
                try:
                    return FipsDoc(await res.json())
                except aiohttp.ContentTypeError as ex:
                    logger.error(id_date + " document not found")
                    logger.debug(await res.text())
                    return None

    async def get_random_doc(self) -> FipsDoc | None:
        res = await self._search_query(q=random.choice(rand_terms), limit=20)
        if error := res.get("error"):
            logger.error(error)
            return None
        doc = random.choice(res.get("hits", []))
        return await self.get_doc_by_id_date(doc.get("id", "no_id"))

    async def find_relevant_by_keywords(
        self, kws: list[str], num_of_docs=20, offset=0, timeout: int = 30
    ) -> list[str]:
        res = await self._search_query(
            q=" OR ".join(compress(kws, kws)),
            offset=offset,
            limit=num_of_docs,
            timeout=timeout,
        )

        if error := res.get("error"):
            logger.error(error)
            return []

        docs = [
            re.sub(r"[^0-9a-zA-Zа-яА-Я_]", "", i["id"]) for i in res.get("hits", [])
        ]

        return docs


class XMLDoc(DocumentBase):
    def extract_citaions(text: str) -> list[str]:
        # Регулярное выражение для патентной заявки
        pattern = r"\b([A-Z]{2}) ?(?:\d{2,4}[/\\])?(\d+),? ?\n?([A-ZА-Я]\d?)?,? ?(\d{2,4})[ .,/\-](\d{2})[ .,/\-](\d{2,4})\b"

        delete_pattern = r"(кл. |\n)"
        text = re.sub(delete_pattern, "", text)
        text = text.replace("Авторское свидетельство СССР N", "SU")

        # Найти все патентные документы в строке
        matches = re.findall(pattern, text)

        out = []
        for match in matches:
            date = match[-3:]
            if len(match[-1]) == 4:
                date = date[::-1]
            date = "".join(date)

            id = "".join(match[:3])

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
    def _freeformat_citations_field(self) -> str:
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
        citations_free = self._freeformat_citations_field

        citations = XMLDoc.extract_citaions(citations_free)

        if len(citations) == 0:
            tag = self.xml_obj.find(XMLDoc.Namespace.pat + "BibliographicData")
            tag = tag.find(XMLDoc.Namespace.pat + "ReferenceCitationBag")
            if tag is not None:
                for ref_cite in tag.findall(XMLDoc.Namespace.pat + "ReferenceCitation"):
                    pat_cite = ref_cite.find(XMLDoc.Namespace.com + "PatentCitation")
                    if pat_cite is not None:
                        cited_doc_id = pat_cite.find(
                            XMLDoc.Namespace.com + "CitedPatentDocumentIdentification"
                        )
                        if cited_doc_id is not None:
                            pat_off = cited_doc_id.find(
                                XMLDoc.Namespace.com + "IPOfficeCode"
                            )
                            doc_num = cited_doc_id.find(
                                XMLDoc.Namespace.com + "DocumentNumber"
                            )
                            kind = cited_doc_id.find(
                                XMLDoc.Namespace.com + "PatentDocumentKindCode"
                            )
                            date = cited_doc_id.find(
                                XMLDoc.Namespace.com + "PatentDocumentDate"
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

    @property
    def description(self) -> str:
        description = self.xml_obj.find(XMLDoc.Namespace.pat + "Description")
        if description is None:
            return ""
        description = " ".join([i.text for i in description if i.text is not None])
        return description

    @property
    def claims(self) -> str:
        claims_root = self.xml_obj.find(XMLDoc.Namespace.pat + "Claims")

        if claims_root is None:
            return ""

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
    def date(self) -> str:
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
    def cluster(self) -> list[str]:
        xml = self.xml_obj
        out = list()

        raw_str_xmls = list()

        raw_str_xmls.append(xml)

        citation_docs = xml.find(XMLDoc.Namespace.pat + "DocumentCitationBag")
        if citation_docs is not None:
            raw_str_xmls += citation_docs.findall(
                XMLDoc.Namespace.pat + "DocumentCitation"
            )

        analog_cited_docs = xml.find(XMLDoc.Namespace.pat + "AnalogOfCitationBag")
        if analog_cited_docs is not None:
            raw_str_xmls += analog_cited_docs.findall(
                XMLDoc.Namespace.pat + "AnalogOfCitation"
            )

        analog_docs = xml.find(XMLDoc.Namespace.pat + "DocumentAnalogBag")
        if analog_docs is not None:
            raw_str_xmls += analog_docs.findall(XMLDoc.Namespace.pat + "DocumentAnalog")

        raw_str_xmls = list(dict.fromkeys(raw_str_xmls))

        for raw_str_xml in raw_str_xmls:
            temp_doc = XMLDoc(raw_str_xml)
            if temp_doc.id_date is not None:
                out.append(temp_doc.id_date)
            out += temp_doc.citations

        return list(dict.fromkeys(out))


class FileSystem(LoaderBase):
    def __init__(self, path: str | Path) -> None:
        self.init_path = Path(path).absolute()

    async def _open_file(self, path: Path) -> XMLDoc:
        async with aiofiles.open(path) as file:
            doc = XMLDoc(await file.read())
        return doc

    async def get_doc(self, id_date: str) -> XMLDoc | None:
        # num_of_doc = extract_number(id_date).lstrip("0")
        for path in self.init_path.iterdir():
            if path.is_dir():
                file_paths = list(path.iterdir())
            elif path.is_file():
                file_paths = [path]
            for file_path in file_paths:
                if id_date in file_path.stem:
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


class InternalESAPI(LoaderBase):
    def __init__(
        self,
        elastic_ip: str | None,
        doc_api_url: str | None = None,
        doc_api_key: str | None = None,
    ) -> None:
        if elastic_ip is None:
            raise ValueError("ElasticSearch Url not specified")
        Elasticsearch(hosts=[elastic_ip]).close()

        self.doc_api_url = doc_api_url
        self.doc_api_key = doc_api_key

        self.es = AsyncElasticsearch(hosts=[elastic_ip])
        self.index = ["july24_ru", "may22_us", "may22_de", "may22_kr", "may22_ep"]
        self.languages = ["ja", "fr", "ru", "it", "de", "en", "ko", "es"]
        self.fields_without_languages = [
            "biblio.{}.title",
            "abstract_cleaned.{}",
            "claims_cleaned.{}",
            "description_cleaned.{}",
            "layers.lexis.biblio.{}.title",
            "layers.lexis.abstract_cleaned.{}",
            "layers.lexis.claims_cleaned.{}",
            "layers.lexis.description_cleaned.{}",
        ]

    @property
    def _fields(self) -> list[str]:
        return [
            f.format(l)
            for f, l in product(self.fields_without_languages, self.languages)
        ]

    def __del__(self):
        hosts = self.es.transport.hosts
        del self.es
        Elasticsearch(hosts=hosts).close()

    async def get_document_list_from_range(
        self,
        from_date: str,
        to_date: str,
        take_every: int = 100,
        kind: str = "(A1 or B2)",
        timeout: int = 60 * 2,
    ) -> list[str]:

        url = self.document_api_url + "/API/query/"

        data = {
            "DP1": from_date,
            "DP2": to_date,
            "KI": kind,
            "TakeXpart": take_every,
            "freetext": "",
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.doc_api_key,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url=url, data=data, timeout=timeout, headers=headers
            ) as res:
                out = await res.json()
                return out.get("IDs", [])

        # async def get_cluster_from_document(self, doc: str | DocumentBase, timeout: int = 60 * 2) -> list[XMLDoc]:
        #     if isinstance(doc, str):
        #         doc_id = doc
        #         doc = None
        #     elif isinstance(doc, DocumentBase):
        #         doc_id = doc.id_date

        #     url = self.document_api_url + "/API/" + doc_id

        #     headers = {"Authorization": self.doc_api_key}

        #     async with aiohttp.ClientSession() as session:
        #         async with session.get(url=url, timeout=timeout, headers=headers) as res:
        #             out = await res.json()
        #             cluster = out.get("luster", [])

        #     async with
        pass

    async def find_relevant_by_keywords(
        self, kws: list[str], num_of_docs=20, offset=0, timeout: int = 30
    ) -> list[str]:
        _source_includes = [
            "common.document_number",
            "common.kind",
            "common.publishing_office",
            "id",
        ]
        query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "query_string": {
                                "query": " ".join(kws),
                                "fields": self._fields,
                                "type": "most_fields",
                                "default_operator": "OR",
                                "quote_field_suffix": ".no_stemmer",
                            }
                        }
                    ],
                    "minimum_should_match": 1,
                }
            }
        }
        res = await self.es.search(
            index=self.index,
            body=query,
            _source_includes=_source_includes,
            request_timeout=timeout,
            size=num_of_docs,
            from_=offset,
        )

        docs = [
            re.sub(r"[^0-9a-zA-Zа-яА-Я_]", "", i["_source"]["id"])
            for i in res["hits"]["hits"]
        ]

        return docs

    async def get_doc(self, id: str) -> FipsDoc | None:
        num_of_doc = re.findall(r"\d+", id)[0]

        id_split = id.split(num_of_doc)
        id_split = list(compress(id_split, id_split))

        pub_office = ""
        num_of_doc = num_of_doc.lstrip("0")
        kind = ""

        if len(id_split) > 0:
            pub_office = id_split[0]
        if len(id_split) > 1:
            kind = id_split[1].split("_")[0]

        _source_includes = [
            "common.document_number",
            "common.kind",
            "common.publishing_office",
            "common.publication_date",
            "id",
            "snippet",
            # "common.citated_docs",
            # "claims",
            # "abstract",
            # "description",
        ]

        query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "query_string": {
                                "query": num_of_doc,
                                "fields": ["id", "common.document_number"],
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }
        }

        if pub_office:
            query["query"]["bool"]["should"].append(
                {
                    "query_string": {
                        "query": pub_office,
                        "fields": ["common.publishing_office"],
                    }
                }
            )

        if kind:
            query["query"]["bool"]["should"].append(
                {
                    "query_string": {
                        "query": kind,
                        "fields": ["common.kind"],
                    }
                }
            )

        res = await self.es.search(
            index=self.index,
            body=query,
            _source_includes=_source_includes,
            size=3,
            request_timeout=5,
        )

        res = res["hits"]["hits"]

        if res:
            return await self.get_doc_by_id_date(res[0]["_id"])
        else:
            return None

    async def get_doc_by_id_date(self, id_date: str) -> FipsDoc | None:
        _source_includes = [
            "common.document_number",
            "common.kind",
            "common.publishing_office",
            "common.publication_date",
            "id",
            "common.citated_docs",
            "claims",
            "abstract",
            "description",
        ]

        for index in self.index:
            try:
                res = await self.es.get(
                    id=id_date,
                    index=index,
                    _source_includes=_source_includes,
                    request_timeout=5,
                )
                return FipsDoc(res["_source"])
            except NotFoundError as ex:
                continue

        return None
