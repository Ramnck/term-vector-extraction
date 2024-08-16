import os

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

from chain import (
    get_cluster_from_document,
    extract_keywords_from_docs,
    get_relevant,
    save_data_to_json,
    load_data_from_json,
    test_different_vectors,
    FIPS_API_KEY,
    BASE_DATA_PATH,
)
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

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
        for extractor_name in keywords.keys():
            keywords[extractor_name] = make_extended_term_vec(keywords[extractor_name])
        data["keywords"] = keywords

        relevant = await get_relevant(keywords, api)

        data["relevant"] = relevant

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
    # duplicate_log_to_stdout = True

    # if duplicate_log_to_stdout:
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    coro = main(80, "term_vectors")
    asyncio.run(coro)
