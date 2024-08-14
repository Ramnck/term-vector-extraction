import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from documents import FipsAPI, FileSystem
from extractors import (
    YAKExtractor,
    KeyBERTExtractor,
    KeyBERTModel,
    RuLongrormerEmbedder,
)
from lexis import (
    clean_ru_text,
    lemmatize_ru_word,
    make_extended_term_vec,
    extract_number,
)
from api import DocumentBase

import logging
import asyncio
import time
from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from itertools import cycle
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


BASE_DATA_PATH = Path("\\prog\\py\\fips\\kwe\\data")
FIPS_API_KEY = os.getenv("FIPS_API_KEY")

if FIPS_API_KEY is None:
    logger.error("Не указан API ключ")
    exit(1)

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


def find_last_file(base_name_of_file: Path):
    num = 1
    while os.path.exists(name_of_file := base_name_of_file + "_" + str(num)):
        num += 1
    return name_of_file


async def extract_keywords(
    doc: DocumentBase, performance: dict | None = None
) -> dict[str, list[list[str]]]:
    kws = {}

    sub_docs = []
    async with asyncio.TaskGroup() as tg:
        for sub_doc_id_date in doc.cluster:
            sub_docs.append(tg.create_task(api.get_doc(sub_doc_id_date)))
    sub_docs = [i.result() for i in sub_docs if i.result() is not None]
    logger.debug(f"total {len(sub_docs)} sub_docs")

    for ex in extractors:
        name = ex.get_name()
        try:
            # if 1:
            t = time.time()

            tmp_kws = []

            for sub_doc in sub_docs:
                if sub_doc is not None:
                    tmp_kws.append(ex.get_keywords(sub_doc))
            base_kws = ex.get_keywords(doc)
            tmp_kws.insert(0, base_kws)

            kws[name] = tmp_kws

            if performance is not None:
                performance[name]["time"].append(time.time() - t)
                performance[name]["document_len"].append(len(doc.text))

        except Exception as eee:
            logger.error(f"Exception in extractor for: {eee}")

    return kws


async def get_relevant(keywords: dict[str, list[str]]) -> dict[str, list[str]]:
    relevant = {}

    async with asyncio.TaskGroup() as tg:
        for extractor_name, kw in keywords.items():
            relevant[extractor_name] = tg.create_task(
                api.find_relevant_by_keywords(kw, num_of_docs=30)
            )

    relevant = {k: v.result() for k, v in relevant.items()}

    return relevant


async def main(num_of_docs: int | None = None, name_of_experiment: str = "KWE"):
    logger.info("Начало обработки")

    performance = {i.get_name(): {"document_len": [], "time": []} for i in extractors}

    async for num_of_doc, doc in enumerate(
        tqdm_asyncio(aiter(loader), total=num_of_docs, desc="Progress")
    ):
        if num_of_doc >= num_of_docs:
            break

        data = {"56": doc.citations, "cluster": list(doc.cluster)}

        keywords = await extract_keywords(doc, performance=performance)

        for extractor_name in keywords.keys():
            keywords[extractor_name] = make_extended_term_vec(keywords[extractor_name])
        data["keywords"] = keywords

        relevant = await get_relevant(keywords)

        data["relevant"] = relevant

        with open(
            BASE_DATA_PATH / "eval" / name_of_experiment / (doc.id_date + ".json"),
            "w+",
            encoding="utf-8",
        ) as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

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
    coro = main(2, "term_vectors")
    asyncio.run(coro)
