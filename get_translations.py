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

# from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_models import GigaChat
from langchain_community.llms.yandex import YandexGPT
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from tve.base import LoaderBase

# from tve.documents import FSLoader, FIPSAPILoader, ESAPILoader
from tve.pipeline import (
    BASE_DATA_PATH,
    ES_URL,
    FIPS_API_KEY,
    extract_keywords_from_docs,
    get_cluster_from_document,
    get_relevant,
    test_different_vectors,
)

# from tve.translators.promt import PROMTTranslator
from tve.translators.langchain import LangChainTranslator
from tve.translators.prompts import PromptTemplate, en_expand_prompt, ru_expand_prompt
from tve.utils import (
    ForgivingTaskGroup,
    batched,
    load_data_from_json,
    save_data_to_json,
)

logging.getLogger("openai._base_client").setLevel(logging.WARN)
logging.getLogger("httpx").setLevel(logging.WARN)

# chatgpt = ChatOpenAI(
#     model="gpt-4o-mini-2024-07-18",
#     temperature=0.5,
#     max_tokens=None,
#     timeout=None,
# )

giga = GigaChat(
    streaming=True, scope="GIGACHAT_API_PERS", model="GigaChat", verify_ssl_certs=False
)

# yc iam create-token
yandex = YandexGPT(
    model_uri=f"gpt://{os.getenv('YANDEX_FOLDER_ID')}/yandexgpt-lite/latest"
)


translators = [
    # LangChainTranslator(chatgpt, name="gpt-4o-mini", default_prompt=en_expand_prompt),
    LangChainTranslator(giga, "giga", default_prompt=ru_expand_prompt),
    LangChainTranslator(yandex, "yandex", "ru", default_prompt=ru_expand_prompt),
    # LLMTranslator(),
    # PROMTTranslator(),
]

logger = logging.getLogger(__name__)


async def process_document(
    data_dict: dict,
    dir_path: Path,
    skip_done: bool = False,
    rewrite: bool = True,
    sleep_time: float = 0.0,
):

    raw_keywords = data_dict["keywords"].copy()
    del data_dict["keywords"]

    outfile_path = dir_path / (data_dict["doc_id"] + ".json")

    # nums = [2, 3]
    nums = [3]

    keywords = {}
    for k, v in raw_keywords.items():
        if k in ["YAKE"]:
            keywords[k] = v[0][:50]

    old_data = (await load_data_from_json(outfile_path)) if rewrite else None

    new_keywords = old_data.get("keywords", {}) if old_data else {}

    for extractor_name, kws in keywords.items():
        for translator in translators:
            for num in nums:
                name = []
                # name.append(extractor_name)
                name.append(translator.name)
                # if len(nums) > 1:
                #     name.append(str(num))
                name = "_".join(name)

                if skip_done and name in new_keywords.keys():
                    continue

                tr = await translator.translate_list(kws, num_of_suggestions=num)
                # new_keywords[name] = [tr]
                new_keywords[name] = [kws + tr]

    data_dict["keywords"] = new_keywords

    if await save_data_to_json(data_dict, outfile_path):
        raise RuntimeError("Error saving file %s" % outfile_path)
    time.sleep(sleep_time)


async def main(
    # api: LoaderBase,
    num_of_docs: int,
    input_path: str,
    output_path: str | None,
    num_of_workers: int,
    skip_done: bool = False,
    rewrite: bool = True,
    sleep_time: float = 0.0,
):

    dir_path = BASE_DATA_PATH / "eval" / input_path

    doc_paths = list(dir_path.iterdir())[:num_of_docs]

    # methods = ["raw"]
    # lens_of_vec = [125, 150, 175, 200]

    progress_bar = tqdm(desc="Progress", total=len(doc_paths))
    os.makedirs(BASE_DATA_PATH / "eval" / output_path, exist_ok=True)

    for doc_path_batch in batched(doc_paths, n=num_of_workers):
        # try:
        async with ForgivingTaskGroup(progress_bar=progress_bar) as tg:

            for doc_path in doc_path_batch:

                data = await load_data_from_json(doc_path)

                # rel_coro = test_different_vectors(
                #     data["keywords"], methods, lens_of_vec, api, timeout=180
                # )

                tg.create_task(
                    process_document(
                        data,
                        BASE_DATA_PATH / "eval" / output_path,
                        skip_done=skip_done,
                        rewrite=rewrite,
                        sleep_time=sleep_time,
                    )
                )

    # except* Exception as exs:
    #     for ex in exs.exceptions:
    #         logger.error("Exception in async for in main() - %s" % str(ex))

    progress_bar.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get translations using term vectors vectors"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-n", "--number", default=500, type=int)
    parser.add_argument("-w", "--num-of-workers", default=5, type=int)
    parser.add_argument("--no-rewrite", action="store_true", default=False)
    parser.add_argument("--skip", "--skip-done", action="store_true", default=False)
    parser.add_argument("--sleep", type=float, default=0.0)

    args = parser.parse_args()

    # api = FipsAPI(FIPS_API_KEY)
    # api = InternalESAPI(ES_URL)
    # loader = FileSystem(Path("data") / "raw" / args.docs)

    if args.output is None:
        args.output = args.input + "_trans"

    coro = main(
        args.number,
        args.input,
        args.output,
        args.num_of_workers,
        skip_done=args.skip,
        rewrite=not args.no_rewrite,
        sleep_time=args.sleep,
    )
    asyncio.run(coro)
