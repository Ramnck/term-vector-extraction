import argparse
import asyncio
import json
import logging
import os
import sys
import time
from itertools import product
from operator import itemgetter
from pathlib import Path

import aiofiles
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from tve.base import LoaderBase
from tve.documents import ESAPILoader, FIPSAPILoader, FSLoader
from tve.pipeline import (
    BASE_DATA_PATH,
    ES_URL,
    FIPS_API_KEY,
    PROMT_IP,
    extract_keywords_from_docs,
    get_cluster_from_document,
    get_relevant,
    test_different_vectors,
    test_translation,
)
from tve.translators.promt import PROMTTranslator
from tve.utils import (
    ForgivingTaskGroup,
    batched,
    load_data_from_json,
    save_data_to_json,
)

logger = logging.getLogger(__name__)

translator = PROMTTranslator(PROMT_IP, enable_cache=True)


async def process_document(relevant_coroutine, data_dict, dir_path):
    relevant = await relevant_coroutine
    data_dict["relevant"] = relevant

    file_path = dir_path / (data_dict["doc_id"] + ".json")
    if await save_data_to_json(data_dict, file_path):
        raise RuntimeError("Error saving file %s" % file_path)


async def main(
    api: LoaderBase,
    num_of_docs: int | None,
    input_path: str,
    output_path: str | None,
    num_of_workers: int,
):

    dir_path = BASE_DATA_PATH / "eval" / input_path

    doc_paths = list(dir_path.iterdir())[:num_of_docs]

    methods = ["raw"]
    lens_of_vec = [125, 150, 175, 200]

    progress_bar = tqdm(desc="Progress", total=num_of_docs)

    for doc_path_batch in batched(doc_paths, n=num_of_workers):
        try:
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

                    kws = {k: data["keywords"][k] for k in ["YAKE", "jina", "e5-large"]}

                    rel_coro = test_translation(
                        kws, api, translator, range(1, 4), 1, num_of_relevant=50
                    )

                    tg.create_task(
                        process_document(
                            rel_coro, data, BASE_DATA_PATH / "eval" / output_path
                        )
                    )

        except* Exception as exs:
            for ex in exs.exceptions:
                logger.error("Exception in test_different - %s" % str(ex))

    progress_bar.close()
    n_tr = sum(map(lambda x: len(x["tr"]), translator.cache.values()))
    n_w = len(translator.cache)
    logger.info(
        "В среднем %3.2f переводов на слово (всего %d слов)" % (n_tr / n_w, n_w)
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get relevant using term vectors vectors"
    )
    parser.add_argument("-i", "--input", default="80")
    parser.add_argument("-o", "--output", default="80_rel")
    parser.add_argument("-n", "--number", default=None, type=int)
    parser.add_argument("-w", "--num-of-workers", default=50, type=int)

    args = parser.parse_args()

    # api = FIPSAPILoader(FIPS_API_KEY)
    api = ESAPILoader(ES_URL)
    # loader = FSLoader(Path("data") / "raw" / args.docs)

    coro = main(api, args.number, args.input, args.output, args.num_of_workers)
    asyncio.run(coro)
