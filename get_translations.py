import argparse
import asyncio
import logging
import os
import re
import time
from pathlib import Path

from tqdm import tqdm

from tve.base import LoaderBase
from tve.pipeline import (
    DATA_PATH,
    ES_URL,
    FIPS_API_KEY,
    PROMT_IP,
    extract_keywords_from_docs,
    get_cluster_from_document,
    get_relevant,
    test_different_vectors,
)
from tve.prompts import en_expand_prompt, ru_expand_prompt
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


translators = []

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
    nums = [2]

    keywords = {}
    for k, v in raw_keywords.items():
        # if k in ["YAKE", "PatS"]:
        if len(v) == 0:
            logger.warning(f"No keywords for {k} in {data_dict['doc_id']}")
            continue
        if isinstance(v[0], list):
            v = [i[0] for i in v]
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
                        # name.append(str(num))
                        name = "_".join(name)

                        if (
                            skip_done
                            and hasattr(new_keywords.get(name, []), "__len__")
                            and len(new_keywords.get(name, [])) > 3
                        ):
                            continue
                        if len(kws) == 0:
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
        pass
        # for ex in exs.exceptions:
        #     logger.error(f"Exception in process document - {ex}")

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
            logger.error(
                f"Exception in process_document.future.result ({data_dict['doc_id']}  {name}) - {ex}"
            )
            # raise ex

    data_dict["keywords"] = new_keywords

    if rewrite:
        data_dict_to_save = await load_data_from_json(outfile_path)
        if data_dict_to_save:
            data_dict["keywords"] = data_dict_to_save["keywords"]
            data_dict["keywords"].update(new_keywords)

    if await save_data_to_json(data_dict, outfile_path):
        raise RuntimeError("Error saving file %s" % outfile_path)
    time.sleep(sleep_time)


async def main(
    num_of_docs: int,
    input_path: str,
    output_path: str | None,
    num_of_workers: int,
    models: str = "",
    skip_done: bool = False,
    rewrite: bool = True,
    sleep_time: float = 0.0,
):
    for model in models:
        if model == "y":
            from langchain_community.llms.yandex import YandexGPT

            from tve.translators.langchain import LangChainTranslator

            yandex = YandexGPT(
                model_uri=f"gpt://{os.getenv('YANDEX_FOLDER_ID')}/yandexgpt/rc"
            )
            model = LangChainTranslator(
                yandex,
                "yandex",
                default_prompt=ru_expand_prompt,
            )
            num_of_workers = min(num_of_workers, 10)
        elif model == "g":
            from langchain_community.chat_models import GigaChat

            from tve.translators.langchain import LangChainTranslator

            giga = GigaChat(
                streaming=True,
                scope="GIGACHAT_API_PERS",
                model="GigaChat",
                verify_ssl_certs=False,
            )
            model = LangChainTranslator(
                giga,
                "giga",
                default_prompt=ru_expand_prompt,
            )
            num_of_workers = 1
        elif model == "c":
            from langchain_openai.chat_models import ChatOpenAI

            from tve.translators.langchain import LangChainTranslator

            chatgpt = ChatOpenAI(
                model="gpt-4o-mini-2024-07-18",
                temperature=0.5,
                max_tokens=None,
                timeout=None,
            )
            model = LangChainTranslator(
                chatgpt,
                name="gpt-4o-mini",
                default_prompt=en_expand_prompt,
            )
        elif model == "p":
            from tve.translators.promt import PROMTTranslator

            model = PROMTTranslator(PROMT_IP)
            model.name = "promt"

        else:
            logger.error("Error - model not supported")
            exit(1)
        translators.append(model)

    dir_path = DATA_PATH / input_path

    doc_paths = [i for i in dir_path.iterdir() if i.is_file()][:num_of_docs]

    progress_bar = tqdm(desc="Progress", total=len(doc_paths))
    os.makedirs(DATA_PATH / output_path, exist_ok=True)

    def exception_handler(loop, ctx):
        ex = ctx["exception"]
        logger.error(f"Exception in exception_handler: {ex}")

    async with CircularTaskGroup(
        num_of_workers, lambda x: progress_bar.update(1), exception_handler
    ) as tg:
        for doc_path in doc_paths:
            # if doc_path.stem != "RU58886U1_20061210":
            #     progress_bar.update(1)
            #     continue
            data = await load_data_from_json(doc_path)
            await tg.create_task(
                process_document(
                    data,
                    DATA_PATH / output_path,
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
    parser.add_argument("--no-skip", action="store_true", default=False)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("-m", "--models", required=True)

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input + "_trans"

    coro = main(
        args.number,
        args.input,
        args.output,
        args.num_of_workers,
        skip_done=not args.no_skip,
        rewrite=not args.no_rewrite,
        sleep_time=args.sleep,
        models=args.models,
    )
    asyncio.run(coro)
