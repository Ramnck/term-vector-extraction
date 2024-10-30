import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from itertools import compress, product
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
    CircularTaskGroup,
    ForgivingTaskGroup,
    batched,
    flatten_kws,
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

    doc_paths = [i for i in dir_path.iterdir() if i.is_file()][:num_of_docs]
    os.makedirs(BASE_DATA_PATH / "eval" / output_path, exist_ok=True)

    methods = ["raw"]
    lens_of_vec = [125, 150, 175, 200]

    progress_bar = tqdm(desc="Progress", total=len(doc_paths))

    def exception_handler(loop, ctx):
        ex = ctx["exception"]
        logger.error(ex)

    async with CircularTaskGroup(
        num_of_workers, lambda x: progress_bar.update(1), exception_handler
    ) as tg:
        for doc_path in doc_paths:
            data = await load_data_from_json(doc_path)

            kws = {}

            for k, v in data["keywords"].items():
                if "YAKE" in k or "PatS" in k:
                    if isinstance(v, dict):
                        kw = sum(v.values(), [])
                    elif isinstance(v, list):
                        # v_str = "|".join(v)
                        kw = list(
                            compress(
                                v,
                                map(
                                    lambda x: len(re.findall(r"[а-яА-ЯёЁ]+", x)) == 0, v
                                ),
                            )
                        )
                else:
                    kw = flatten_kws(v, "слово/фраза")

                kws[k] = kw[:175]

            rel_coro = get_relevant(kws, api)

            coro = process_document(
                rel_coro, data, BASE_DATA_PATH / "eval" / output_path
            )

            await tg.create_task(coro)

    progress_bar.close()
    n_tr = sum(map(lambda x: len(x["tr"]), translator.cache.values()))
    n_w = len(translator.cache)
    if n_w > 0:
        logger.info(
            "В среднем %3.2f переводов на слово (всего %d слов)" % (n_tr / n_w, n_w)
        )
        await save_data_to_json(translator.cache, BASE_DATA_PATH / "cache.json")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get relevant using term vectors vectors"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-n", "--number", default=None, type=int)
    parser.add_argument("-w", "--num-of-workers", default=5, type=int)

    args = parser.parse_args()

    # api = FIPSAPILoader(FIPS_API_KEY)
    api = ESAPILoader(ES_URL)

    api.index = ["may22_us"]

    if args.output is None:
        args.output = args.input + "_rel"

    coro = main(api, args.number, args.input, args.output, args.num_of_workers)
    asyncio.run(coro)
    api.close()
