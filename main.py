import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from documents import FipsAPI, FileSystem
from extractors import (
    RAKExtractor,
    YAKExtractor,
    KeyBERTExtractor,
    KeyBERTModel,
    RuLongrormerEmbedder,
)
from lexis import clean_ru_text

import logging
import asyncio
import time
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DATA_PATH = Path("\\prog\\py\\fips\\kwe\\data")
FIPS_API_KEY = os.getenv("FIPS_API_KEY")


def find_last_file(base_name_of_file: Path):
    num = 1
    while os.path.exists(name_of_file := base_name_of_file + "_" + str(num)):
        num += 1
    return name_of_file


async def main(num_of_docs=None, name_of_experiment="KWE"):
    api = FipsAPI(FIPS_API_KEY)
    loader = FileSystem("data\\raw\\skolkovo")

    extractors = [
        RAKExtractor(),
        YAKExtractor(),
        KeyBERTExtractor(method_name="paraphrase-multilingual-MiniLM-L12-v2"),
        KeyBERTExtractor(
            KeyBERTModel(RuLongrormerEmbedder()),
            method_name="ru-longformer-tiny-16384",
        ),
        KeyBERTExtractor(
            SentenceTransformer("DiTy/bi-encoder-russian-msmarco"),
            "bi-encoder-russian-msmarco",
        ),
        KeyBERTExtractor(
            SentenceTransformer("cointegrated/rubert-tiny2"), "rubert-tiny2"
        ),
    ]

    with open(
        BASE_DATA_PATH / "eval" / "names_of_methods.json", "w+", encoding="utf-8"
    ) as file:
        names = [i.get_name() for i in extractors]
        json.dump(names, file, indent=4, ensure_ascii=False)

    exit()

    logger.info("Начало обработки")

    performance = {i.get_name(): [0, 0] for i in extractors}
    last_indicated_percent = -99999
    num_of_doc = 0

    async for doc in loader:
        if num_of_doc >= num_of_docs:
            break

        kws = {}
        relevant = {}

        for ex in extractors:
            name = ex.get_name()
            try:
                t = time.time()

                text_extraction_func = (
                    lambda x: f"[CLS] {clean_ru_text(x.claims)}  {clean_ru_text(x.abstract)} [CLS] {clean_ru_text(x.description)}"
                )

                kw = ex.get_keywords(
                    doc,
                    text_extraction_func=text_extraction_func,
                )

                kws[name] = kw

                performance[name][0] += time.time() - t
                performance[name][1] += 1

            except Exception as eee:
                logger.error(f"Exception in extractor for: {eee}")
                break

        async with asyncio.TaskGroup() as tg:
            for extractor_name, kw in kws.items():
                relevant[extractor_name] = tg.create_task(
                    api.find_relevant_by_keywords(kw)
                )
        relevant = {k: v.result() for k, v in relevant.items()}

        data = {"56": doc.citations}

        data["keywords"] = kws
        data["relevant"] = relevant

        with open(
            BASE_DATA_PATH / "eval" / "kwe" / (doc.id + ".json"),
            "w+",
            encoding="utf-8",
        ) as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        num_of_doc += 1
        if num_of_doc * 100 // num_of_docs >= last_indicated_percent:
            last_indicated_percent = num_of_doc * 100 // num_of_docs
            logger.info(f"{last_indicated_percent} % done")

    logger.info("Средняя скорость работы алгоритмов:")
    for k, v in performance.items():
        if v[1]:
            logger.info(f"{k} : {round(v[0]/v[1], 2)} s")

    with open(
        BASE_DATA_PATH / "eval" / "names_of_methods.json", "w+", encoding="utf-8"
    ) as file:
        names = [i.get_name() for i in extractors]
        json.dump(names, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # import profile
    # profile.run('main(1)')
    asyncio.run(main(79))
