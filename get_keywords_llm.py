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
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from tve.base import LoaderBase
from tve.documents.fs import FSLoader
from tve.pipeline import (
    BASE_DATA_PATH,
    ES_URL,
    FIPS_API_KEY,
    extract_keywords_from_docs,
    get_cluster_from_document,
    get_relevant,
    test_different_vectors,
)
from tve.prompts import PromptTemplate, en_expand_prompt, ru_expand_prompt

# from tve.translators.promt import PROMTTranslator
from tve.translators.langchain import LangChainTranslator
from tve.utils import (
    CircularTaskGroup,
    ForgivingTaskGroup,
    batched,
    flatten_kws,
    load_data_from_json,
    save_data_to_json,
)

logging.getLogger("openai._base_client").setLevel(logging.WARN)
logging.getLogger("httpx").setLevel(logging.WARN)

logger = logging.getLogger(__name__)


async def main(
    # api: LoaderBase,
    model: str,
    prompt: str,
    num_of_docs: int,
    input_path: str,
    output_path: str | None,
    num_of_workers: int,
    skip_done: bool = False,
    rewrite: bool = True,
    return_json: bool = False,
    sleep_time: float = 0.0,
):

    if not prompt.endswith(".json"):
        prompt += ".json"

    prompt = await load_data_from_json(Path(BASE_DATA_PATH) / "prompts" / prompt)

    try:
        prompt = PromptTemplate(**prompt)
    except:
        logger.error("Error parsing prompt")
        exit(1)

    if model == "y":
        from langchain_community.llms.yandex import YandexGPT

        yandex = YandexGPT(
            model_uri=f"gpt://{os.getenv('YANDEX_FOLDER_ID')}/yandexgpt/rc"
        )
        model = LangChainTranslator(
            yandex, "yandex", default_prompt=prompt, return_json=return_json
        )
        num_of_workers = min(num_of_workers, 10)
    elif model == "s":
        from langchain_community.chat_models import GigaChat

        giga = GigaChat(
            streaming=True,
            scope="GIGACHAT_API_PERS",
            model="GigaChat",
            verify_ssl_certs=False,
        )
        model = LangChainTranslator(
            giga, "giga", default_prompt=prompt, return_json=return_json
        )
        num_of_workers = 1
    elif model == "o":
        from langchain_openai.chat_models import ChatOpenAI

        chatgpt = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            temperature=0.5,
            max_tokens=None,
            timeout=None,
        )
        model = LangChainTranslator(
            chatgpt, name="gpt-4o-mini", default_prompt=prompt, return_json=return_json
        )

    else:
        logger.error("Error - model not supported")
        exit(1)

    input_path = BASE_DATA_PATH / "raw" / input_path
    loader = FSLoader(input_path)
    output_path = BASE_DATA_PATH / "eval" / output_path
    os.makedirs(output_path, exist_ok=True)

    paths = [i.stem for i in input_path.iterdir() if i.is_file()][:num_of_docs]

    async def task(doc_id):
        doc = await loader.get_doc(doc_id)

        old_data = await load_data_from_json(output_path / f"{doc_id}.json")
        data = (
            {"doc_id": doc.id_date, "keywords": {}, "cluster": doc.cluster}
            if not rewrite or not old_data
            else old_data
        )

        if skip_done and len(data["keywords"].get(model.name, [])) > 0:
            return

        if return_json:
            try:
                kws = await model.translate_list(
                    [doc.text], num_of_suggestions=50, format_output=False
                )
            except RuntimeError as ex:
                name, text = str(ex).split(";")
                if isinstance(text, list):
                    text = "".join(text)
                os.makedirs(output_path / "errors", exist_ok=True)
                path = output_path / "errors" / (doc_id + ".json")
                err_data = await load_data_from_json(path)
                err_data = {} if not err_data else err_data
                err_data[name] = text
                await save_data_to_json(err_data, path)

                kws = re.findall(r"(?:\")([\w -]+)(?:\"[:,\]])", text)

        else:
            messages = [
                SystemMessage(model.system_prompt(50)),
                HumanMessage(model.human_promt(doc.text)),
            ]

            try:
                kws = await model.make_chat_completion(messages)
                kws = kws.split(",")
            except Exception as ex:
                kws = []
                logger.error(f"Exception in make_chat_completion {ex}")

        kws = list(set((i.strip() for i in kws if i.strip())))

        data["keywords"][model.name] = kws

        path = output_path / f"{doc_id}.json"
        try:
            await save_data_to_json(data, path)
        except:
            logger.error(f"cannot save {path}")

    progress_bar = tqdm(total=len(paths), desc="Docs")

    async with CircularTaskGroup(
        num_of_workers, lambda x: progress_bar.update(1)
    ) as tg:
        for doc_id in paths:
            await tg.create_task(task(doc_id))

    progress_bar.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get keywords using proprietary llm")

    parser.add_argument("-m", "--model", default="y")
    parser.add_argument("-p", "--prompt", default="ru_3")
    parser.add_argument("-i", "--input", default="l")
    parser.add_argument("-o", "--output", default="llm_ex")
    parser.add_argument("--skip", "--skip-done", action="store_true", default=True)

    # parser.add_argument("-m", "--model", required=True)
    # parser.add_argument("-p", "--prompt", required=True)
    # parser.add_argument("-i", "--input", required=True)
    # parser.add_argument("-o", "--output", default=None)
    # parser.add_argument("--skip", "--skip-done", action="store_true", default=False)
    parser.add_argument("-n", "--number", default=None, type=int)
    parser.add_argument("-w", "--num-of-workers", default=10, type=int)
    parser.add_argument("--no-rewrite", action="store_true", default=False)
    parser.add_argument("-j", "--json", action="store_true", default=False)
    parser.add_argument("--sleep", type=float, default=0.0)

    args = parser.parse_args()

    # api = FipsAPI(FIPS_API_KEY)
    # api = InternalESAPI(ES_URL)
    # loader = FileSystem(Path("data") / "raw" / args.docs)

    if args.output is None:
        args.output = args.input + "_trans"

    coro = main(
        args.model,
        args.prompt,
        args.number,
        args.input,
        args.output,
        args.num_of_workers,
        skip_done=args.skip,
        rewrite=not args.no_rewrite,
        return_json=args.json,
        sleep_time=args.sleep,
    )
    asyncio.run(coro)
