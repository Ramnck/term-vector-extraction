import logging
import os
import random
import re
import sys
import time
from itertools import compress, product
from pathlib import Path
from typing import Union

from dotenv import load_dotenv

from .base import DocumentBase, KeyWordExtractorBase, LoaderBase, TranslatorBase
from .lexis import (
    clean_ru_text,
    extract_number,
    lemmatize_ru_word,
    make_extended_term_vec,
)
from .utils import ForgivingTaskGroup, batched, flatten_kws

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
CACHE_DIR = Path("E:") / "FIPS" / "cache"
PROMT_IP = os.getenv("PROMT_IP")


async def get_cluster_from_document(
    doc: DocumentBase, api: LoaderBase, timeout: int = 60
) -> list[DocumentBase]:

    sub_docs_futures = []

    try:
        async with ForgivingTaskGroup() as tg:
            for sub_doc_id_date in doc.cluster[1:]:
                sub_docs_futures.append(
                    tg.create_task(api.get_doc(sub_doc_id_date, timeout=timeout))
                )
    except* Exception as exs:
        for ex in exs.exceptions:
            logger.error("Exception in get_cluster_from_document - %s" % str(type(ex)))

    sub_docs = []

    for future in sub_docs_futures:
        try:
            if future.result() is not None:
                sub_docs.append(future.result())
        except:
            pass
    sub_docs = [doc] + sub_docs

    logger.debug(f"Total {len(sub_docs)} documents")

    return sub_docs


async def extract_keywords_from_docs(
    docs: DocumentBase | list[DocumentBase] | None,
    extractors: list[KeyWordExtractorBase],
    performance: dict | None = None,
    num_of_keywords: int = 50,
) -> dict[str, list[list[str]]]:
    kws = {}

    if docs is None:
        logger.error("No docs given to extract_keywords_from_docs")
        return {i.name: [[""]] for i in extractors}

    if isinstance(docs, DocumentBase):
        docs = [docs]

    for ex in extractors:
        name = ex.name

        # if 1:
        try:
            tmp_kws = []

            for doc in docs:
                if doc is not None:
                    t = time.time()
                    kw = ex.get_keywords(doc, num=num_of_keywords)
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
    keywords: dict[str, Union[str, list, dict]],
    api: LoaderBase,
) -> dict[str, list[str]]:
    relevant = {}

    try:
        async with ForgivingTaskGroup() as tg:
            for extractor_name, kw in keywords.items():
                relevant[extractor_name] = tg.create_task(
                    api.find_relevant_by_keywords(
                        flatten_kws(kw),
                        num_of_docs=50,
                        timeout=90,
                    )
                )
    except* Exception as exs:
        for ex in exs.exceptions:
            logger.error("Ex in get_relevant - %s" % str(ex))

    relevant_results = {}
    for k, v in relevant.items():
        try:
            relevant_results[k] = v.result()
        except Exception as ex:
            logger.error("ex in future.result() - %s - %s" % (str(type(ex)), str(ex)))
            relevant_results[k] = []

    return relevant_results


async def test_different_vectors(
    data_keywords: dict[str, list[list[str]]],
    methods: list[str],
    lens_of_vec: list[int],
    api: LoaderBase,
    num_of_workers: int | None = None,
    timeout: int = 30,
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
                        term_vec = term_vec_vec[0][:len_of_vec]

                    relevant[name] = tg.create_task(
                        api.find_relevant_by_keywords(term_vec, 35, timeout=timeout)
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


async def test_translation(
    data_keywords: dict[str, list[list[str]]],
    api: LoaderBase,
    translators: list[TranslatorBase],
    nums_of_translations: list[int] = [3],
    num_of_relevant: int = 35,
    timeout: int = 30,
) -> dict[str, list[str]]:

    relevant = {}

    methods = product(data_keywords.items(), nums_of_translations)

    for (extractor_name, term_vec_vec), num in methods:
        # name = "_".join([extractor_name, str(num)])
        name = extractor_name
        for translator in translators:
            try:

                trans = await translator.translate_list(
                    term_vec_vec[0], num_of_suggestions=num
                )
                tv = list(
                    map(lambda x: "".join(re.findall(r"[A-Za-zа-яА-ЯёЁ -]", x)), trans)
                )
                tv = list(compress(tv, tv))
                if len(tv) < 2:
                    logger.error("Empty translations list in %s" % name)
                    continue
                relevant[name + "_" + translator.name] = (
                    await api.find_relevant_by_keywords(
                        tv,
                        num_of_docs=num_of_relevant,
                        timeout=timeout,
                    )
                )

            except Exception as ex:
                logger.error("Exception in test_translation - %s" % str(ex))

    return relevant


async def get_translations(
    data_keywords: dict[str, list[list[str]]],
    translators: list[TranslatorBase],
    nums_of_translations: list[int] = [3],
) -> dict[str, list[str]]:

    methods = product(data_keywords.items(), nums_of_translations, translators)

    out = {}

    for (extractor_name, term_vec_vec), num, translator in methods:
        name = "_".join([extractor_name, translator.name, str(num)])
        # name = extractor_name
        try:

            trans = await translator.translate_list(
                term_vec_vec[0], num_of_suggestions=num
            )
            tv = list(
                map(lambda x: "".join(re.findall(r"[A-Za-zа-яА-ЯёЁ -]", x)), trans)
            )
            tv = list(compress(tv, tv))
            if len(tv) < 2:
                logger.error("Empty translations list in %s" % name)
                continue
            out[name] = [tv]

        except Exception as ex:
            logger.error("Exception in test_translation - %s" % str(ex))

    return out
