import json
import logging
import random
import re
from functools import cached_property
from itertools import compress

import aiohttp

from ..base import LoaderBase
from ..lexis import extract_number
from .json import JSONDocument

logger = logging.getLogger(__name__)

rand_terms = "ракета машина смазка установка самолет вертолёт автомат мотоцикл насос инструмент лист дерево обработка рост эволюция".split()


class FIPSAPILoader(LoaderBase):
    api_url = "https://searchplatform.rospatent.gov.ru/patsearch/v0.2/"

    def __init__(self, api_key) -> None:
        self.api_key = api_key

    @cached_property
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

    async def get_doc(self, id: str, timeout: int = 10) -> JSONDocument | None:
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
    ) -> JSONDocument | None:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.api_url + "docs/" + id_date, headers=self._headers, timeout=timeout
            ) as res:
                try:
                    return JSONDocument(await res.json())
                except aiohttp.ContentTypeError as ex:
                    # logger.error(id_date + " document not found")
                    logger.debug(await res.text())
                    return None

    async def get_random_doc(self) -> JSONDocument | None:
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
