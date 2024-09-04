import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import asyncio
import json
import logging
import sys
import time
from itertools import chain, cycle
from pathlib import Path

import aiofiles
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm_asyncio

from api import DocumentBase, KeyWordExtractorBase, LoaderBase
from chain import (
    BASE_DATA_PATH,
    FIPS_API_KEY,
    extract_keywords_from_docs,
    get_cluster_from_document,
    get_relevant,
    load_data_from_json,
    save_data_to_json,
    test_different_vectors,
)
from documents import FileSystem, FipsAPI
from extractors import (
    KeyBERTExtractor,
    KeyBERTModel,
    RuLongrormerEmbedder,
    YAKExtractor,
)
from lexis import (
    clean_ru_text,
    extract_number,
    lemmatize_ru_word,
    make_extended_term_vec,
)

load_dotenv()

logger = logging.getLogger(__name__)

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
        text_extraction_func=lambda doc: "query: " + doc.text,
    ),
]


async def main(
    loader: LoaderBase,
    api: LoaderBase,
    num_of_docs: int | None = None,
    name_of_experiment: str = "KWE",
):
    logger.info("Начало обработки")

    performance = {i.get_name(): {"document_len": [], "time": []} for i in extractors}

    num_of_doc = 0
    async for doc in tqdm_asyncio(aiter(loader), total=num_of_docs, desc="Progress"):
        num_of_doc += 1

        data = {"doc_id": doc.id, "56": doc.citations, "cluster": doc.cluster}

        cluster = await get_cluster_from_document(doc, api)

        keywords = await extract_keywords_from_docs(
            cluster, extractors, performance=performance
        )

        if not keywords["YAKE"][0][0]:
            logger.info("Doc %d has empty kws" % doc.id)
            logger.info(" ".join(map(lambda x: x.id_date, cluster)))

        for extractor_name in keywords.keys():
            keywords[extractor_name] = keywords[extractor_name][0]
        data["keywords"] = keywords

        relevant = await get_relevant(keywords, api)

        data["relevant"] = relevant

        path_of_file = (
            BASE_DATA_PATH / "eval" / name_of_experiment / (doc.id_date + ".json")
        )
        if await save_data_to_json(data, path_of_file):
            logger.error("Error occured while saving %s file" % path_of_file.name)

        if num_of_docs is not None:
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

    parser = argparse.ArgumentParser(description="Extract term vectors from documents")
    parser.add_argument("-i", "--input", default="clusters")
    parser.add_argument("-o", "--output", default="80")
    parser.add_argument("-n", "--number", default=None, type=int)

    args = parser.parse_args()

    api = FipsAPI(FIPS_API_KEY)
    loader = FileSystem(Path("data") / "raw" / args.input)

    coro = main(loader, api, args.number, args.output)
    asyncio.run(coro)
