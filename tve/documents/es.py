import asyncio
import logging
import re
from functools import cached_property
from itertools import compress, product

import aiohttp
from elasticsearch7 import (
    AsyncElasticsearch,
    Elasticsearch,
    NotFoundError,
    TransportError,
)

from ..base import LoaderBase
from .fips import FIPSDocument

logger = logging.getLogger(__name__)

logging.getLogger("elasticsearch").setLevel(logging.ERROR)


class ESAPILoader(LoaderBase):
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
        self.index = [
            "july24_ru",
            "may22_us",
            "may22_de",
            "may22_kr",
            "may22_ep",
            "may22_gb",
        ]
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

    def close(self):
        old_loop = asyncio.get_event_loop()
        if old_loop.is_closed():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(self.es.close())
            new_loop.close()
            asyncio.set_event_loop(old_loop)
        else:
            old_loop.run_until_complete(self.es.close())

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

        # async def get_cluster_from_document(self, doc: str | DocumentBase, timeout: int = 60 * 2) -> list[XMLDocument]:
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
        self,
        kws: list[str],
        num_of_docs: int = 35,
        offset: int = 0,
        timeout: int = 30,
    ) -> list[str]:
        _source_includes = [
            "common.document_number",
            "common.kind",
            "common.publishing_office",
            "id",
        ]

        query_string = " OR ".join(map(lambda x: f"({x})", kws))

        query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "query_string": {
                                "query": query_string,
                                "fields": self._fields,
                                "type": "most_fields",
                                "default_operator": "OR",
                                "quote_field_suffix": ".no_stemmer",
                                "enable_position_increments": False,
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

    async def get_doc(self, id: str, timeout: int = 5) -> FIPSDocument | None:

        match = re.search(r"([A-Z]{2})?(\d+)([A-Z]\d?)?(_\d+)?", id)

        pub_office = match.group(1)
        num_of_doc = match.group(2)
        kind = match.group(3)
        # date = match.group(4)

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
                        "fields": ["id", "common.publishing_office"],
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
            request_timeout=timeout,
        )

        res = res["hits"]["hits"]

        if res:
            return await self.get_doc_by_id_date(res[0]["_id"], timeout)
        else:
            return None

    async def get_doc_by_id_date(
        self, id_date: str, timeout: int = 5
    ) -> FIPSDocument | None:
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
                    request_timeout=timeout,
                )
                return FIPSDocument(res["_source"])
            except NotFoundError as ex:
                continue
            except TransportError as ex:
                continue

        return None
