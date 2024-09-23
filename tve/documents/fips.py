import json
import logging
import random
import re
import xml.etree.ElementTree as ET
from itertools import compress
from pathlib import Path

import aiohttp

from ..base import DocumentBase, LoaderBase
from ..lexis import extract_citaions, extract_number

logger = logging.getLogger(__name__)

rand_terms = "ракета машина смазка установка самолет вертолёт автомат мотоцикл насос инструмент лист дерево обработка рост эволюция".split()


class FIPSDocument(DocumentBase):
    def __init__(self, raw: str | dict) -> None:
        if isinstance(raw, str):
            raw = json.loads(raw)
        self.raw_json = raw

    @property
    def cluster(self) -> list[str]:
        cluster = [self.id_date] + self.citations
        cluster += self.raw_json.get("cluster", [])

        return list(dict.fromkeys(cluster))

    @property
    def _raw_citation_field(self) -> str:
        biblio = self.raw_json.get("biblio", {})
        bib = biblio.get("ru", None)
        if not bib:
            bib = biblio.get("en", None)
            if not bib:
                return ""
        return bib.get("citations", "")

    @property
    def citations(self) -> list[str]:
        cites = [i["id"] for i in self.raw_json["common"].get("citated_docs", [])]
        # "biblio": {"ru": {"citations":
        cites += extract_citaions(self._raw_citation_field)
        return list(dict.fromkeys(cites))

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


class FIPSAPILoader(LoaderBase):
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

    async def get_doc(self, id: str, timeout: int = 10) -> FIPSDocument | None:
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
            res = await self.get_doc_by_id_date(res["id"], timeout=timeout // 2)
        return res

    async def get_doc_by_id_date(
        self, id_date: str, timeout: int = 5
    ) -> FIPSDocument | None:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.api_url + "docs/" + id_date, headers=self._headers, timeout=timeout
            ) as res:
                try:
                    return FIPSDocument(await res.json())
                except aiohttp.ContentTypeError as ex:
                    # logger.error(id_date + " document not found")
                    logger.debug(await res.text())
                    return None

    async def get_random_doc(self) -> FIPSDocument | None:
        res = await self._search_query(q=random.choice(rand_terms), limit=20)
        if error := res.get("error"):
            logger.error(error)
            return None
        doc = random.choice(res.get("hits", []))
        return await self.get_doc_by_id_date(doc.get("id", "no_id"))

    async def find_relevant_by_keywords(
        self, kws: list[str], num_of_docs: int = 35, offset: int = 0, timeout: int = 30
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
