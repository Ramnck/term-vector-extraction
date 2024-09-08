import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from itertools import product
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

logging_format = "%(name)s - %(asctime)s - %(levelname)s - %(message)s"

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
    datefmt="%H:%M:%S",
    filename=filepath,
    filemode="w+",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(logging_format))
logging.getLogger().addHandler(handler)

BASE_DATA_PATH = Path("data")
FIPS_API_KEY = os.getenv("FIPS_API_KEY")
ES_URL = os.getenv("ES_URL")


if FIPS_API_KEY is None:
    logger.error("Не указан API ключ")
    exit(1)


class ForgivingTaskGroup(asyncio.TaskGroup):
    _abort = lambda self: None


async def get_cluster_from_document(
    doc: DocumentBase, api: LoaderBase
) -> list[DocumentBase]:
    sub_docs = []
    async with ForgivingTaskGroup() as tg:
        for sub_doc_id_date in doc.cluster[1:]:
            sub_docs.append(tg.create_task(api.get_doc(sub_doc_id_date)))
    sub_docs = [i.result() for i in sub_docs if i.result() is not None]
    sub_docs = [doc] + sub_docs

    logger.debug(f"Total {len(sub_docs)} documents")

    return sub_docs


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


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

    async with ForgivingTaskGroup() as tg:
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
    num_of_workers: int | None = None,
) -> dict[str, list[str]]:

    relevant = {}
    method_array = list(product(lens_of_vec, data_keywords.items(), methods))

    for batch in batched(
        method_array,
        n=num_of_workers if num_of_workers is not None else len(method_array),
    ):
        try:
            async with ForgivingTaskGroup() as tg:
                for len_of_vec, (extractor_name, term_vec_vec), method in batch:
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
                        term_vec_vec_copy = [i.copy() for i in term_vec_vec]
                        for i in term_vec_vec_copy:
                            random.shuffle(i)
                        term_vec = make_extended_term_vec(
                            term_vec_vec_copy, length=len_of_vec
                        )
                    elif method == "raw":
                        term_vec = term_vec_vec[0]

                    relevant[name] = tg.create_task(
                        api.find_relevant_by_keywords(term_vec, 35)
                    )

        except* Exception as exs:
            for ex in exs.exceptions:
                logger.error("Exception in test_different - %s" % str(ex))

    relevant_results = {}
    for k, v in relevant.items():
        try:
            relevant_results[k] = v.result()
        except:
            relevant_results[k] = []

    return relevant_results
