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
from lexis import clean_ru_text, lemmatize_ru_word

import logging
import asyncio
import time
from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from itertools import cycle


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DATA_PATH = Path("\\prog\\py\\fips\\kwe\\data")
FIPS_API_KEY = os.getenv("FIPS_API_KEY")


def find_last_file(base_name_of_file: Path):
    num = 1
    while os.path.exists(name_of_file := base_name_of_file + "_" + str(num)):
        num += 1
    return name_of_file


async def main(num_of_docs: int | None = None, name_of_experiment: str = "KWE"):
    api = FipsAPI(FIPS_API_KEY)
    loader = FileSystem("data\\raw\\clusters")

    extractors = [
        YAKExtractor(),
        KeyBERTExtractor(
            SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            ),
            "multilingual-mpnet",
        ),
        KeyBERTExtractor(
            SentenceTransformer("intfloat/multilingual-e5-large"),
            "e5-large",
            text_extraction_func=lambda doc: "query: " + clean_ru_text(doc.text),
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

        sub_docs = []
        async with asyncio.TaskGroup() as tg:
            for sub_doc_id_date in doc.cluster:
                # sub_doc_id, sub_doc_date = sub_doc_id_date.split("_")
                sub_docs.append(tg.create_task(api.get_doc(sub_doc_id_date)))
        sub_docs = [i.result() for i in sub_docs if i.result() is not None]
        logger.info(f"total {len(sub_docs)} sub_docs")

        for ex in extractors:
            name = ex.get_name()
            try:
                t = time.time()

                tmp_kws = []

                for sub_doc in sub_docs:
                    if sub_doc is not None:
                        tmp_kws.append(ex.get_keywords(sub_doc, use_mmr=True))
                tmp_kws.append(ex.get_keywords(doc, use_mmr=True))

                kw = set()
                iterator = cycle([cycle(i) for i in tmp_kws if i])

                try:
                    async with asyncio.timeout(
                        0.2
                    ):  # таймаут чтоб не уйти в бесконечный while
                        while len(kw) < 100:
                            kw.add(lemmatize_ru_word(next(next(iterator))))
                            await asyncio.sleep(0)
                except TimeoutError as ex:
                    pass

                kw = list(kw)

                kws[name] = kw

                performance[name]["time"].append(time.time() - t)
                performance[name]["document"].append(doc.id_date)

            except Exception as eee:
                logger.error(f"Exception in extractor for: {eee}")

        async with asyncio.TaskGroup() as tg:
            for extractor_name, kw in kws.items():
                relevant[extractor_name] = tg.create_task(
                    api.find_relevant_by_keywords(kw, num_of_docs=30)
                )
        relevant = {k: v.result() for k, v in relevant.items()}

        data = {"56": doc.citations, "cluster": list(doc.cluster)}

        data["keywords"] = kws
        data["relevant"] = relevant

        with open(
            BASE_DATA_PATH / "eval" / name_of_experiment / (doc.id_date + ".json"),
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
    coro = main(80, "ex_from_cluster")
    asyncio.run(coro)
