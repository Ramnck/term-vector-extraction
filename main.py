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
import numpy as np

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
        YAKExtractor(),
        KeyBERTExtractor(
            "paraphrase-multilingual-MiniLM-L12-v2", method_name="standart"
        ),
        KeyBERTExtractor(
            KeyBERTModel(RuLongrormerEmbedder()),
            method_name="ru-longformer",
            text_extraction_func=lambda doc: f"[CLS] {clean_ru_text(doc.claims)}  {clean_ru_text(doc.abstract)} [CLS] {clean_ru_text(doc.description)}",
        ),
        KeyBERTExtractor(
            SentenceTransformer("DiTy/bi-encoder-russian-msmarco"),
            "bi-encoder-ru",
        ),
        KeyBERTExtractor(
            SentenceTransformer("cointegrated/rubert-tiny2"), "rubert-tiny"
        ),
        KeyBERTExtractor(SentenceTransformer("all-mpnet-base-v2"), "all-mpnet"),
        KeyBERTExtractor(
            SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1"),
            "multi-qa-mpnet",
        ),
        KeyBERTExtractor(
            SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            ),
            "multilingual-mpnet",
        ),
        KeyBERTExtractor(
            SentenceTransformer("sentence-transformers/allenai-specter"),
            "allenai-specter",
        ),
        KeyBERTExtractor(
            SentenceTransformer(
                "sentence-transformers/distiluse-base-multilingual-cased-v2"
            ),
            "distiluse-base",
        ),
        KeyBERTExtractor(
            SentenceTransformer("intfloat/multilingual-e5-large"),
            "e5-large",
            text_extraction_func=lambda doc: "query: " + clean_ru_text(doc.text),
        ),
        KeyBERTExtractor(
            SentenceTransformer("deepvk/USER-bge-m3"),
            "USER-bge",
            # text_extraction_func=lambda doc: "query: " + clean_ru_text(doc.text),
        ),
    ]

    logger.info("Начало обработки")

    performance = {i.get_name(): {"document": [], "time": []} for i in extractors}
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

                kw = ex.get_keywords(doc, use_mmr=True)

                kws[name] = kw

                performance[name]["time"].append(time.time() - t)
                performance[name]["document"].append(doc.id)

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
            logger.info(f"{last_indicated_percent} % done ({num_of_doc} docs)")

    logger.info("Средняя скорость работы алгоритмов:")
    for extractor_name, value in performance.items():
        mean_time = np.mean(value["time"])
        out = f"{extractor_name} : {round(mean_time, 2)} s"
        logger.info(out)

    with open(
        BASE_DATA_PATH / "eval" / "performance.json", "w+", encoding="utf-8"
    ) as file:
        json.dump(performance, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # import profile
    # profile.run('main(1)')
    asyncio.run(main(79))
