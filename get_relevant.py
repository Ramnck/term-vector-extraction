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

# translator = PROMTTranslator(PROMT_IP, enable_cache=True)


async def main(
    api: LoaderBase,
    num_of_docs: int | None,
    input_path: str,
    output_path: str | None,
    num_of_workers: int,
    skip_done: bool = False,
    rewrite: bool = True,
    timeout: int = 90,
):

    dir_path = BASE_DATA_PATH / "eval" / input_path

    doc_paths = [i for i in dir_path.iterdir() if i.is_file()][:num_of_docs]
    os.makedirs(BASE_DATA_PATH / "eval" / output_path, exist_ok=True)

    methods = ["raw"]
    lens_of_vec = [125, 150, 175, 200]

    progress_bar = tqdm(desc="Progress", total=len(doc_paths))

    def exception_handler(loop, ctx):
        ex = ctx["exception"]
        logger.error(f"error in main - {ex}")

    async def process_document(input_path, output_path, tg: CircularTaskGroup):

        old_data = await load_data_from_json(output_path) if rewrite else None

        relevant = old_data.get("relevant", {}) if old_data else {}

        try:
            input_data = await load_data_from_json(input_path)
        except json.JSONDecodeError:
            logger.error(f"{input_path.parent / input_path.name} have broken structure")
            return

        keywords = input_data.get("keywords", {})

        futures = {}

        for name, raw_kws in keywords.items():

            if skip_done and len(relevant.get(name, [])) > 3:
                continue

            if "YAKE" in name or "PatS" in name:
                if isinstance(raw_kws, dict):
                    kws = sum(raw_kws.values(), [])
                elif isinstance(raw_kws, list):
                    kws = list(
                        compress(
                            raw_kws,
                            map(
                                lambda x: len(re.findall(r"[а-яА-ЯёЁ]+", x)) == 0,
                                raw_kws,
                            ),
                        )
                    )
                else:
                    logger.error(
                        f"{input_path.stem} - {name} - WRONG TYPE({type(raw_kws)})"
                    )
            else:
                if isinstance(raw_kws, (list, dict)):
                    kws = flatten_kws(raw_kws, "слово/фраза")
                else:
                    logger.error(
                        f"{input_path.stem} - {name} - WRONG TYPE({type(raw_kws)})"
                    )
            kws = [k for k in kws if isinstance(k, str) and len(k) > 2]
            kws = kws[:175]

            if any(map(lambda x: len(x) == 1, kws)):
                logger.warning(
                    f"{input_path.stem} - {name} - kws have letter instead of word"
                )

            futures[name] = await tg.create_task(
                api.find_relevant_by_keywords(kws, num_of_docs=50, timeout=timeout)
            )

        while not all(map(lambda x: x.done(), futures.values())):
            await asyncio.sleep(0.1)

        for name, task in futures.items():
            try:
                relevant[name] = task.result()
            except Exception as ex:
                logger.error(f"{input_path.stem} - {name} - error: {ex}")
                relevant[name] = []
                pass

        input_data["relevant"] = relevant

        if await save_data_to_json(input_data, output_path):
            raise RuntimeError("Error saving file %s" % output_path)

    for doc_path_batch in batched(doc_paths, 15):
        async with CircularTaskGroup(
            num_of_workers, exception_handler=exception_handler
        ) as task_pool:
            async with ForgivingTaskGroup(progress_bar) as main_tg:

                for doc_path in doc_path_batch:
                    main_tg.create_task(
                        process_document(
                            doc_path,
                            BASE_DATA_PATH / "eval" / output_path / doc_path.name,
                            task_pool,
                        )
                    )

    progress_bar.close()
    # n_tr = sum(map(lambda x: len(x["tr"]), translator.cache.values()))
    # n_w = len(translator.cache)
    # if n_w > 0:
    #     logger.info(
    #         "В среднем %3.2f переводов на слово (всего %d слов)" % (n_tr / n_w, n_w)
    #     )
    #     await save_data_to_json(translator.cache, BASE_DATA_PATH / "cache.json")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get relevant using term vectors vectors"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-n", "--number", default=None, type=int)
    parser.add_argument("-w", "--num-of-workers", default=5, type=int)
    parser.add_argument("--no-rewrite", action="store_true", default=False)
    parser.add_argument("--skip", "--skip-done", action="store_true", default=False)
    parser.add_argument("-t", "--timeout", default=90, type=int)

    args = parser.parse_args()

    # api = FIPSAPILoader(FIPS_API_KEY)
    api = ESAPILoader(ES_URL)

    api.index = ["may22_us"]

    if args.output is None:
        args.output = args.input + "_rel"

    coro = main(
        api,
        args.number,
        args.input,
        args.output,
        args.num_of_workers,
        skip_done=args.skip,
        rewrite=not args.no_rewrite,
        timeout=args.timeout,
    )
    asyncio.run(coro)
    api.close()
