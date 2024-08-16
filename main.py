import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from documents import FipsAPI, FileSystem

from sentence_transformers import SentenceTransformer
from extractors import (
    YAKExtractor,
    KeyBERTExtractor,
    KeyBERTModel,
    RuLongrormerEmbedder,
)
from lexis import (
    clean_ru_text,
    lemmatize_ru_word,
    make_extended_term_vec,
    extract_number,
)
from api import DocumentBase, LoaderBase, KeyWordExtractorBase

import logging
import asyncio
import time
from pathlib import Path
import json
import numpy as np
import aiofiles
import sys

from itertools import cycle, chain
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    filename="log.txt",
    filemode="w+",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


BASE_DATA_PATH = Path("data")
FIPS_API_KEY = os.getenv("FIPS_API_KEY")

if FIPS_API_KEY is None:
    logger.error("Не указан API ключ")
    exit(1)

api = FipsAPI(FIPS_API_KEY)
loader = FileSystem(Path("data") / "raw" / "clusters")

extractors = [
    YAKExtractor(),
    KeyBERTExtractor(
        SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ),
        "multilingual-mpnet",
    ),
    KeyBERTExtractor(
        SentenceTransformer("intfloat/multilingual-e5-large"),
        "e5-large",
        text_extraction_func=lambda doc: "query: " + clean_ru_text(doc.text),
    ),
]


def find_last_file(base_name_of_file: Path):
    num = 1
    while os.path.exists(name_of_file := base_name_of_file + "_" + str(num)):
        num += 1
    return name_of_file


async def get_cluster_from_document(
    doc: DocumentBase, api: LoaderBase
) -> list[DocumentBase]:
    sub_docs = []
    async with asyncio.TaskGroup() as tg:
        for sub_doc_id_date in doc.cluster:
            sub_docs.append(tg.create_task(api.get_doc(sub_doc_id_date)))
    sub_docs = [i.result() for i in sub_docs if i.result() is not None]
    sub_docs = set(sub_docs) - set([doc])
    sub_docs = [doc] + list(sub_docs)

    logger.debug(f"Total {len(sub_docs)} documents")

    return sub_docs


async def extract_keywords_from_docs(
    docs: DocumentBase | list[DocumentBase],
    extractors: list[KeyWordExtractorBase],
    performance: dict | None = None,
) -> dict[str, list[list[str]]]:
    kws = {}

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
                    if kw:
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
    async with asyncio.TaskGroup() as tg:
        for method in methods:
            for len_of_vec in lens_of_vec:
                for extractor_name, term_vec_vec in data_keywords.items():
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

                    relevant[name] = tg.create_task(
                        api.find_relevant_by_keywords(term_vec, 30)
                    )
    relevant = {k: v.result() for k, v in relevant.items()}

    return relevant


async def main(num_of_docs: int | None = None, name_of_experiment: str = "KWE"):
    logger.info("Начало обработки")

    performance = {i.get_name(): {"document_len": [], "time": []} for i in extractors}

    num_of_doc = 0
    async for doc in tqdm_asyncio(aiter(loader), total=num_of_docs, desc="Progress"):

        data = {"56": doc.citations, "cluster": list(doc.cluster)}

        cluster = await get_cluster_from_document(doc, api)

        keywords = await extract_keywords_from_docs(
            cluster, extractors, performance=performance
        )
        # for extractor_name in keywords.keys():
        # keywords[extractor_name] = make_extended_term_vec(keywords[extractor_name])
        data["keywords"] = keywords

        # relevant = await get_relevant(keywords, api)

        # path_of_file = (
        #     BASE_DATA_PATH / "eval" / name_of_experiment / (doc.id_date + ".json")
        # )
        # data = await load_data_from_json(path_of_file)
        # if not data:
        #     logger.error("File %s not found" % path_of_file.name)
        #     continue

        # relevant = await test_different_vectors(
        #     data["keywords"], ["expand", "mix"], [100, 125, 150, 175, 200], api
        # )

        # data["relevant"] = relevant

        path_of_file = (
            BASE_DATA_PATH / "eval" / name_of_experiment / (doc.id_date + ".json")
        )
        if await save_data_to_json(data, path_of_file):
            logger.error("Error occured while saving %s file" % path_of_file.name)

        num_of_doc += 1
        if num_of_doc >= num_of_docs:
            break

    logger.info("Средняя скорость работы алгоритмов:")
    for extractor_name, value in performance.items():
        mean_time = np.mean(value["time"])
        out = f"{extractor_name} : {round(mean_time, 2)} s"
        logger.info(out)

    path_of_file = BASE_DATA_PATH / "eval" / (name_of_experiment + "_performance.json")
    if await save_data_to_json(performance, path_of_file):
        logger.error("Error occured while saving %s file" % path_of_file.name)


if __name__ == "__main__":
    # import profile
    # profile.run('main(1)')
    duplicate_log_to_stdout = True

    if duplicate_log_to_stdout:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    coro = main(80, "term_vectors")
    asyncio.run(coro)
