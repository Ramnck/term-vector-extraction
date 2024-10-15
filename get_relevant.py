import argparse
import asyncio
import json
import logging
import os
import re
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

    doc_paths = list(dir_path.iterdir())[:num_of_docs]
    os.makedirs(BASE_DATA_PATH / "eval" / output_path, exist_ok=True)

    methods = ["raw"]
    lens_of_vec = [125, 150, 175, 200]

    progress_bar = tqdm(desc="Progress", total=len(doc_paths))

    for doc_path_batch in batched(doc_paths, n=num_of_workers):
        try:
            async with ForgivingTaskGroup(progress_bar) as tg:

                for doc_path in doc_path_batch:

                    data = await load_data_from_json(doc_path)

                    # rel_coro = test_different_vectors(
                    #     data["keywords"], methods, lens_of_vec, api, timeout=180
                    # )

                    # kws = {
                    # k: data["keywords"][k]
                    # for k in ["YAKE", "jina", "e5-large", "iteco"]
                    # }
                    # kws = data["keywords"]
                    # rel_coro = test_translation(
                    # kws, api, translator, num_of_relevant=50
                    # )

                    def clean_func(word):
                        match = re.findall(
                            r"([А-Яа-яA-Za-zёЁ]+)(-[А-Яа-яA-Za-zёЁ]+)?", word
                        )
                        return " ".join(("".join(i) for i in match))

                    kws = map(clean_func, flatten_kws(data["keywords"]))
                    kws = list(kws)

                    rel_coro = get_relevant(kws, api)

                    tg.create_task(
                        process_document(
                            rel_coro, data, BASE_DATA_PATH / "eval" / output_path
                        )
                    )

        except* Exception as exs:
            for ex in exs.exceptions:
                logger.error("Exception in main - %s" % str(ex))

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

    try:
        asyncio.run(api.close())
    except:
        pass
