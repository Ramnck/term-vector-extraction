import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from documents import FipsAPI, FileSystem
from extractors import RAKExtractor, YAKExtractor, TextRankExtractor, KeyBERTExtractor


import logging
import asyncio
import time
from pathlib import Path
import json


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
        TextRankExtractor(),
        KeyBERTExtractor(),
    ]

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
                kws[name] = ex.get_keywords(doc)

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

        data = {"56": doc.citations}

        for i, j in relevant.items():
            data[i + "_relevant"] = j.result()

        for i, j in kws.items():
            data[i + "_keywords"] = j

        with open(
            BASE_DATA_PATH / "kwe" / (doc.id + ".json"), "w+", encoding="utf-8"
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


if __name__ == "__main__":
    # import profile
    # profile.run('main(1)')
    asyncio.run(main(79))
