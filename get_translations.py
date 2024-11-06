import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from itertools import product
from pathlib import Path

import aiofiles
import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from tve.base import LoaderBase

# from tve.documents import FSLoader, FIPSAPILoader, ESAPILoader
from tve.pipeline import (
    BASE_DATA_PATH,
    ES_URL,
    FIPS_API_KEY,
    PROMT_IP,
    extract_keywords_from_docs,
    get_cluster_from_document,
    get_relevant,
    test_different_vectors,
)
from tve.prompts import PromptTemplate, en_expand_prompt, ru_expand_prompt
from tve.translators.promt import PROMTTranslator
from tve.utils import (
    CircularTaskGroup,
    ForgivingTaskGroup,
    batched,
    flatten_kws,
    load_data_from_json,
    save_data_to_json,
)

# from langchain_community.chat_models import GigaChat
# from langchain_community.llms.yandex import YandexGPT
# from langchain_openai.chat_models import ChatOpenAI


# from tve.translators.langchain import LangChainTranslator


logging.getLogger("openai._base_client").setLevel(logging.WARN)
logging.getLogger("httpx").setLevel(logging.WARN)

# chatgpt = ChatOpenAI(
#     model="gpt-4o-mini-2024-07-18",
#     temperature=0.5,
#     max_tokens=None,
#     timeout=None,
# )

# giga = GigaChat(
#     streaming=True, scope="GIGACHAT_API_PERS", model="GigaChat", verify_ssl_certs=False
# )

# yc iam create-token
# yandex = YandexGPT(model_uri=f"gpt://{os.getenv('YANDEX_FOLDER_ID')}/yandexgpt/rc")


translators = [
    # LangChainTranslator(chatgpt, name="gpt-4o-mini", default_prompt=ru_expand_prompt),
    # LangChainTranslator(giga, "giga", default_prompt=ru_expand_prompt),
    # LangChainTranslator(yandex, "yandex", "ru", default_prompt=ru_expand_prompt),
    # LLMTranslator(),
    PROMTTranslator(PROMT_IP),
]

logger = logging.getLogger(__name__)


async def process_document(
    data_dict: dict,
    dir_path: Path,
    skip_done: bool = False,
    rewrite: bool = True,
    sleep_time: float = 0.0,
    timeout: float = 50,
):

    raw_keywords = data_dict["keywords"].copy()
    del data_dict["keywords"]

    outfile_path = dir_path / (data_dict["doc_id"] + ".json")

    # nums = [2, 3]
    nums = [3]

    keywords = {}
    for k, v in raw_keywords.items():
        # if k in ["YAKE", "PatS"]:
        if isinstance(v[0], list):
            v = v[0]
        keywords[k] = v[:50]

    old_data = (await load_data_from_json(outfile_path)) if rewrite else None

    new_keywords = old_data.get("keywords", {}) if old_data else {}

    futures = {}
    try:
        async with ForgivingTaskGroup() as tg:
            for extractor_name, kws in keywords.items():
                for translator in translators:
                    for num in nums:
                        name = []
                        name.append(extractor_name)
                        name.append(translator.name)
                        # if len(nums) > 1:
                        #     name.append(str(num))
                        name = "_".join(name)

                        if (
                            skip_done
                            and hasattr(new_keywords.get(name, []), "__len__")
                            and len(new_keywords.get(name, [])) > 3
                        ):
                            continue

                        flat_kws = flatten_kws(kws)
                        tr = tg.create_task(
                            translator.translate_list(
                                flat_kws, num_of_suggestions=num, noexcept=False
                            )
                        )
                        # new_keywords[name] = [tr]
                        futures[name] = tr
    except* Exception as exs:
        for ex in exs.exceptions:
            logger.error(f"Exception in process document - {ex}")

    for name, future in futures.items():
        try:
            if future.result() is not None:
                new_keywords[name] = future.result()
        except RuntimeError as ex:
            _, text = str(ex).split(";")
            if isinstance(text, list):
                text = "".join(text)
            os.makedirs(dir_path / "errors", exist_ok=True)
            with open(
                dir_path / "errors" / (data_dict["doc_id"] + ".txt"),
                encoding="utf-8",
                mode="w",
            ) as f:
                f.write(f'"{name}": ' + text)
            all_kws = re.findall(r"(?:\")([\w -]+)(?:\"[:,\]])", text)
            en_kws = [i for i in all_kws if len(re.findall(r"[А-Яа-яЁё]", i)) == 0]
            new_keywords[name] = en_kws
        except Exception as ex:
            logger.error(f"Exception in process_document.future.result - {ex}")
            # raise ex

    data_dict["keywords"] = new_keywords

    if await save_data_to_json(data_dict, outfile_path):
        raise RuntimeError("Error saving file %s" % outfile_path)
    time.sleep(sleep_time)


async def main(
    num_of_docs: int,
    input_path: str,
    output_path: str | None,
    num_of_workers: int,
    skip_done: bool = False,
    rewrite: bool = True,
    sleep_time: float = 0.0,
):

    dir_path = BASE_DATA_PATH / "eval" / input_path

    doc_paths = [i for i in dir_path.iterdir() if i.is_file()][:num_of_docs]

    progress_bar = tqdm(desc="Progress", total=len(doc_paths))
    os.makedirs(BASE_DATA_PATH / "eval" / output_path, exist_ok=True)

    def exception_handler(loop, ctx):
        ex = ctx["exception"]
        logger.error(f"Exception in exception_handler: {ex}")

    async with CircularTaskGroup(
        num_of_workers, lambda x: progress_bar.update(1), exception_handler
    ) as tg:
        for doc_path in doc_paths:
            data = await load_data_from_json(doc_path)
            await tg.create_task(
                process_document(
                    data,
                    BASE_DATA_PATH / "eval" / output_path,
                    skip_done=skip_done,
                    rewrite=rewrite,
                    sleep_time=sleep_time,
                )
            )

    progress_bar.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get translations using term vectors vectors"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-n", "--number", default=None, type=int)
    parser.add_argument("-w", "--num-of-workers", default=5, type=int)
    parser.add_argument("--no-rewrite", action="store_true", default=False)
    parser.add_argument("--skip", "--skip-done", action="store_true", default=False)
    parser.add_argument("--sleep", type=float, default=0.0)

    args = parser.parse_args()

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
