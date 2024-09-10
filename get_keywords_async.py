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
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from api import DocumentBase, KeyWordExtractorBase, LoaderBase
from chain import (
    BASE_DATA_PATH,
    ES_URL,
    FIPS_API_KEY,
    ForgivingTaskGroup,
    batched,
    extract_keywords_from_docs,
    get_cluster_from_document,
    get_relevant,
    load_data_from_json,
    save_data_to_json,
    test_different_vectors,
)
from documents import BlankDoc, FileSystem, FipsAPI
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


async def process_document(
    doc: DocumentBase,
    api: LoaderBase,
    name_of_experiment: Path | str,
    performance: dict | None = None,
):
    data = {"doc_id": doc.id, "56": doc.citations, "cluster": doc.cluster}

    cluster = await get_cluster_from_document(doc, api)

    # keywords = await extract_keywords_from_docs(
    #     cluster, extractors, performance=performance
    # )

    keywords = {}

    for ex in extractors:
        temp_doc = BlankDoc()
        temp_doc.text = " ".join(map(lambda x: x.text, cluster))
        keywords[ex.get_name()] = [ex.get_keywords(temp_doc, num=200)]

    if not keywords["YAKE"][0][0]:
        logger.error("Doc  %d has empty kws" % doc.id)
        logger.error("  ".join(map(lambda x: x.id_date, cluster)))

    data["keywords"] = keywords

    path_of_file = (
        BASE_DATA_PATH / "eval" / name_of_experiment / (doc.id_date + ".json")
    )
    if await save_data_to_json(data, path_of_file):
        logger.error("Error occured while saving %s file" % path_of_file.name)


async def main(
    loader: LoaderBase,
    api: LoaderBase,
    num_of_docs: int | None = None,
    name_of_experiment: str = "KWE",
    num_of_workers: int = 1,
):
    logger.info("Начало обработки")

    # performance = {i.get_name(): {"document_len": [], "time": []} for i in extractors}
    performance = None

    progress_bar = tqdm(desc="Progress", total=num_of_docs)

    num_of_doc = 0
    # async for doc in tqdm_asyncio(aiter(loader), total=num_of_docs, desc="Progress"):

    docs = [doc async for doc in loader][:num_of_docs]

    for doc_batch in batched(docs, n=num_of_workers):
        async with ForgivingTaskGroup() as tg:

            def new_on_task_done(task):
                asyncio.TaskGroup._on_task_done(tg, task)
                progress_bar.update(1)

            tg._on_task_done = new_on_task_done

            for doc in doc_batch:
                tg.create_task(
                    process_document(
                        doc,
                        api,
                        name_of_experiment=name_of_experiment,
                        performance=performance,
                    )
                )
    progress_bar.close()

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
    parser.add_argument("-w", "--num-of-workers", default=1, type=int)

    args = parser.parse_args()

    api = FipsAPI(FIPS_API_KEY)
    loader = FileSystem(Path("data") / "raw" / args.input)

    coro = main(loader, api, args.number, args.output, args.num_of_workers)
    asyncio.run(coro)
