import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import aiofiles
import numpy as np
from dotenv import load_dotenv

from api import DocumentBase, KeyWordExtractorBase, LoaderBase
from lexis import (
    clean_ru_text,
    extract_number,
    lemmatize_ru_word,
    make_extended_term_vec,
)

name = [Path(sys.argv[0]).name] + sys.argv[1:]
filename = " ".join(name)
filename = re.sub(r'[\\/:*?"<>|]', "", filename)

filepath = Path("data") / "logs" / filename
filepath = filepath.parent / (filepath.name + ".log.txt")


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    filename=filepath,
    filemode="w+",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

BASE_DATA_PATH = Path("data")
FIPS_API_KEY = os.getenv("FIPS_API_KEY")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")


if FIPS_API_KEY is None:
    logger.error("Не указан API ключ")
    exit(1)


async def get_cluster_from_document(
    doc: DocumentBase, api: LoaderBase
) -> list[DocumentBase]:
    sub_docs = []
    async with asyncio.TaskGroup() as tg:
        for sub_doc_id_date in doc.cluster[1:]:
            sub_docs.append(tg.create_task(api.get_doc(sub_doc_id_date)))
    sub_docs = [i.result() for i in sub_docs if i.result() is not None]
    sub_docs = [doc] + sub_docs

    logger.debug(f"Total {len(sub_docs)} documents")

    return sub_docs


async def extract_keywords_from_docs(
    docs: DocumentBase | list[DocumentBase] | None,
    extractors: list[KeyWordExtractorBase],
    performance: dict | None = None,
) -> dict[str, list[list[str]]]:
    kws = {}

    if docs is None:
        logger.error("No docs given to extract_keywords_from_docs")
        return {i.get_name(): [[""]] for i in extractors}

    if isinstance(docs, DocumentBase):
        docs = [docs]

    for ex in extractors:
        name = ex.get_name()
        try:
            tmp_kws = []

            for doc in docs:
                if doc is not None:
                    t = time.time()
                    kw = ex.get_keywords(doc)
                    if len(kw) > 2:
                        tmp_kws.append(kw)
                        if performance is not None:
                            performance[name]["time"].append(time.time() - t)
                            performance[name]["document_len"].append(len(doc.text))

            kws[name] = tmp_kws

        except Exception as eee:
            logger.error(f"Exception in extractor for: {eee}")

    return kws


async def get_relevant(
    keywords: dict[str, list[str]], api: LoaderBase
) -> dict[str, list[str]]:
    relevant = {}

    async with asyncio.TaskGroup() as tg:
        for extractor_name, kw in keywords.items():
            relevant[extractor_name] = tg.create_task(
                api.find_relevant_by_keywords(kw, num_of_docs=30)
            )

    relevant = {k: v.result() for k, v in relevant.items()}

    return relevant


async def save_data_to_json(
    obj: dict | list, path_of_file: Path
) -> bool:  # False on success
    try:
        async with aiofiles.open(
            path_of_file,
            "w+",
            encoding="utf-8",
        ) as file:
            await file.write(json.dumps(obj, ensure_ascii=False, indent=4))
        return False
    except FileNotFoundError:
        return True


async def load_data_from_json(path_of_file: Path) -> dict | None:
    try:
        async with aiofiles.open(
            path_of_file,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.loads(await file.read())
        return data
    except FileNotFoundError:
        return None


async def test_different_vectors(
    data_keywords: dict[str, list[list[str]]],
    methods: list[str],
    lens_of_vec: list[int],
    api: LoaderBase,
) -> dict[str, list[str]]:

    relevant = {}
    try:
        for len_of_vec in lens_of_vec:
            for extractor_name, term_vec_vec in data_keywords.items():
                for sub_methods in [methods[:2], methods[2:]]:
                    async with asyncio.TaskGroup() as tg:
                        for method in sub_methods:
                            name = extractor_name + "_" + method + "_" + str(len_of_vec)
                            if method == "expand":
                                term_vec = make_extended_term_vec(
                                    term_vec_vec[1:],
                                    base_vec=term_vec_vec[0],
                                    length=len_of_vec,
                                )
                            elif method == "mix":
                                term_vec = make_extended_term_vec(
                                    term_vec_vec, length=len_of_vec
                                )
                            elif method == "shuffle":
                                for i in term_vec_vec:
                                    random.shuffle(i)
                                term_vec = make_extended_term_vec(
                                    term_vec_vec, length=len_of_vec
                                )
                            elif method == "raw":
                                term_vec = term_vec_vec[0]

                            relevant[name] = tg.create_task(
                                api.find_relevant_by_keywords(term_vec, 30)
                            )
    except* Exception as exs:
        for ex in exs.exceptions:
            logger.error("Exception in test_different, vectors %s" % str(ex))

    relevant = {k: v.result() for k, v in relevant.items()}

    return relevant
