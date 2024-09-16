import argparse
import asyncio
import json
import logging
import os
import sys
import time
from itertools import product
from pathlib import Path

import aiofiles
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from api import LoaderBase
from chain import (
    BASE_DATA_PATH,
    ES_URL,
    FIPS_API_KEY,
    extract_keywords_from_docs,
    get_cluster_from_document,
    get_relevant,
    test_different_vectors,
)
from documents import FileSystem, FipsAPI, InternalESAPI
from extractors.translators import LLMTranslator, PROMTTranslator
from utils import ForgivingTaskGroup, batched, load_data_from_json, save_data_to_json

translators = [
    LLMTranslator(),
    # PROMTTranslator()
]

logger = logging.getLogger(__name__)


async def process_document(data_dict: dict, dir_path: Path):

    raw_keywords = data_dict["keywords"].copy()
    del data_dict["keywords"]

    keywords = {}
    for k, v in raw_keywords.items():
        keywords[k] = v[0][:50]

    new_keywords = {}
    for k, v in keywords.items():
        for translator in translators:
            new_keywords[k + "_" + translator.get_name()] = (
                v + await translator.translate(v)
            )

    data_dict["keywords"] = new_keywords

    file_path = dir_path / (data_dict["doc_id"] + ".json")
    if await save_data_to_json(data_dict, file_path):
        raise RuntimeError("Error saving file %s" % file_path)


async def main(
    # api: LoaderBase,
    num_of_docs: int,
    input_path: str,
    output_path: str | None,
    num_of_workers: int,
):

    dir_path = BASE_DATA_PATH / "eval" / input_path

    doc_paths = list(dir_path.iterdir())[:num_of_docs]

    # methods = ["raw"]
    # lens_of_vec = [125, 150, 175, 200]

    progress_bar = tqdm(desc="Progress", total=num_of_docs)

    for doc_path_batch in batched(doc_paths, n=num_of_workers):
        # try:
        async with ForgivingTaskGroup() as tg:

            def new_on_task_done(task):
                asyncio.TaskGroup._on_task_done(tg, task)
                progress_bar.update(1)

            tg._on_task_done = new_on_task_done

            for doc_path in doc_path_batch:

                data = await load_data_from_json(doc_path)

                # rel_coro = test_different_vectors(
                #     data["keywords"], methods, lens_of_vec, api, timeout=180
                # )

                tg.create_task(
                    process_document(data, BASE_DATA_PATH / "eval" / output_path)
                )

    # except* Exception as exs:
    #     for ex in exs.exceptions:
    #         logger.error("Exception in async for in main() - %s" % str(ex))

    progress_bar.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get relevant using term vectors vectors"
    )
    parser.add_argument("-i", "--input", default="500")
    parser.add_argument("-o", "--output", default="500_trans")
    parser.add_argument("-n", "--number", default=500, type=int)
    parser.add_argument("-w", "--num-of-workers", default=2, type=int)

    args = parser.parse_args()

    # api = FipsAPI(FIPS_API_KEY)
    # api = InternalESAPI(ES_URL)
    # loader = FileSystem(Path("data") / "raw" / args.docs)

    coro = main(args.number, args.input, args.output, args.num_of_workers)
    asyncio.run(coro)
