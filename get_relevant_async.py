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
    ForgivingTaskGroup,
    batched,
    extract_keywords_from_docs,
    get_cluster_from_document,
    get_relevant,
    load_data_from_json,
    save_data_to_json,
    test_different_vectors,
)
from documents import FileSystem, FipsAPI, InternalESAPI

logger = logging.getLogger(__name__)


async def main(
    loader: LoaderBase,
    api: LoaderBase,
    num_of_docs: int | None,
    input_path: str,
    output_path: str | None,
    num_of_workers: int,
):

    docs = [doc async for doc in loader][:num_of_docs]

    methods = ["raw"]
    lens_of_vec = [125, 150, 175, 200]

    relevant = dict()

    progress_bar = tqdm(desc="Progress", total=num_of_docs)

    for doc_batch in batched(docs, n=num_of_workers):
        try:
            async with ForgivingTaskGroup() as tg:

                def new_on_task_done(task):
                    asyncio.TaskGroup._on_task_done(tg, task)
                    progress_bar.update(1)

                tg._on_task_done = new_on_task_done

                for doc in doc_batch:

                    doc_id = doc.id_date
                    path_of_file = (
                        BASE_DATA_PATH / "eval" / input_path / (doc_id + ".json")
                    )
                    data = await load_data_from_json(path_of_file)
                    relevant[doc_id] = tg.create_task(
                        test_different_vectors(
                            data["keywords"], methods, lens_of_vec, api, timeout=180
                        )
                    )

        except* Exception as exs:
            for ex in exs.exceptions:
                logger.error("Exception in test_different - %s" % str(ex))

    progress_bar.close()

    relevant_results = {}
    for k, v in relevant.items():
        try:
            relevant_results[k] = v.result()
        except:
            pass

    for doc_id, data in relevant_results.items():
        path_of_file = BASE_DATA_PATH / "eval" / output_path / (doc_id + ".json")

        if await save_data_to_json(data, path_of_file):
            logger.error("Error occured while saving %s file" % path_of_file.name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get relevant using term vectors vectors"
    )
    parser.add_argument("-i", "--input", default="80")
    parser.add_argument("-o", "--output", default="80_rel")
    parser.add_argument(
        "-d",
        "--docs",
        default="clusters",
        help="Where to iterate over document id's (path to xmls)",
    )
    parser.add_argument("-n", "--number", default=None, type=int)
    parser.add_argument("-w", "--num-of-workers", default=50, type=int)

    args = parser.parse_args()

    # api = FipsAPI(FIPS_API_KEY)
    api = InternalESAPI(ES_URL)
    loader = FileSystem(Path("data") / "raw" / args.docs)

    coro = main(loader, api, args.number, args.input, args.output, args.num_of_workers)
    asyncio.run(coro)
