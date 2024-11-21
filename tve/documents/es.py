import asyncio
import logging
import re
from collections import defaultdict
from functools import cached_property
from itertools import compress, product
from operator import itemgetter

import aiohttp
from elasticsearch7 import (
    AsyncElasticsearch,
    Elasticsearch,
    NotFoundError,
    TransportError,
)

from ..base import LoaderBase
from ..lexis import escape_elasticsearch_query
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
            "may22_ru_apps",
            # "may22f_ru_apps",
        ]
        # self.languages = ["ja", "fr", "ru", "it", "de", "en", "ko", "es"]
        self.languages = ["ru", "en"]
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

    @cached_property
    def _default_source_includes(self) -> list[str]:
        _source_includes = [
            "common.document_number",
            "common.kind",
            "common.publishing_office",
            "common.publication_date",
            "common.application",
            "common.priority",
            "common.classification",
            "id",
        ]

        return _source_includes

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
        raw_kws: list[str] | list[tuple[str, float]],
        num_of_docs: int = 35,
        offset: int = 0,
        timeout: int = 30,
        with_scores: bool = False,
    ) -> list[str]:
        _source_includes = self._default_source_includes

        kws = set()
        kws_dict = defaultdict(lambda: 1)
        for raw_kw in raw_kws:
            if isinstance(raw_kw, (tuple, list)):
                raw_kw, score = raw_kw
            elif isinstance(raw_kw, str):
                score = 1
            else:
                raise ValueError(
                    "Keywords must be strings or tuples of strings and floats"
                )

            kw = "".join(re.findall(r"[%\*,\[\]\+/0-9a-zA-Zа-яА-ЯёЁ -]", raw_kw))
            kw = re.sub(r"or|and|not", "", kw.lower())
            kw = re.sub(
                r"\s+|-+",
                lambda x: " " if " " in x.group() else "-",
                kw,
            ).strip(" -\n\r")
            if kw:
                kws.add(kw)
                kws_dict[kw] = score

        # kws is iterable of keywords, kws_dict is mapping of kw to score

        query_string = " OR ".join(
            map(
                lambda x: f"({escape_elasticsearch_query(x)})"
                + (f"^{kws_dict[x]}" if with_scores else ""),
                kws,
            )
        )

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

        docs = [re.sub(r"с", "", i["_source"]["id"]) for i in res["hits"]["hits"]]

        return docs

    async def get_doc(self, id: str, timeout: int = 5) -> FIPSDocument | None:

        match = re.search(r"([A-Z]{2})?(\d+)([A-Z]\d?)?(_\d+)?", id)

        pub_office = match.group(1)
        num_of_doc = match.group(2)
        kind = match.group(3)
        date = match.group(4)

        docs = await self._search_query(
            num_of_doc, kind, pub_office, size=3, timeout=timeout
        )

        if docs:
            return await self.get_doc_by_id_date(docs[0], timeout)
        else:
            return None

    async def get_doc_by_id_date(
        self, id_date: str, timeout: int = 5
    ) -> FIPSDocument | None:

        _source_includes = self._default_source_includes + [
            "common.citated_docs",
            "claims",
            "claims_cleaned",
            "abstract",
            "abstract_cleaned",
            "description",
            "description_cleaned",
        ]

        for index in self.index:
            try:
                res = await self.es.get(
                    id=id_date,
                    index=index,
                    _source_includes=_source_includes,
                    request_timeout=timeout,
                )
                data = res["_source"]
                for k in ("claims", "abstract", "description"):
                    if (k + "_cleaned") in data.keys():
                        del data[k]
                return FIPSDocument(data)
            except NotFoundError as ex:
                continue
            except TransportError as ex:
                continue

        return None

    async def _search_query(
        self,
        number: str = "",
        kind: str = "",
        pub_office: str = "",
        app_number: str = "",
        size: int = 5,
        timeout: int = 5,
    ) -> list:

        _source_includes = self._default_source_includes + ["snippet"]

        query = {
            "query": {
                "bool": {
                    "should": [],
                    "minimum_should_match": 1,
                }
            }
        }

        if app_number:
            query["query"]["bool"]["should"].append(
                {
                    "query_string": {
                        "query": app_number,
                        "fields": ["common.application.number"],
                    }
                }
            )

        if number:
            query["query"]["bool"]["should"].append(
                {
                    "query_string": {
                        "query": number,
                        "fields": ["id", "common.document_number"],
                    }
                }
            )

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
            size=size,
            request_timeout=timeout,
        )

        res = res["hits"]["hits"]

        return res
