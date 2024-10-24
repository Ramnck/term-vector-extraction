import argparse
import asyncio
import logging
import os
from collections import Counter
from datetime import datetime
from operator import itemgetter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tve.base import DocumentBase, LoaderBase
from tve.documents import ESAPILoader, FSLoader
from tve.pipeline import BASE_DATA_PATH, ES_URL
from tve.utils import CircularTaskGroup, ForgivingTaskGroup, batched

logger = logging.getLogger(__name__)


async def process_document(loader: LoaderBase, dataset_path: Path, doc_id: str):
    doc = await loader.get_doc(doc_id)
    if not doc:
        raise RuntimeError(f"{doc_id} from {dataset_path.name} not found")
    else:
        doc.save_file(dataset_path)


async def preprocess_document(api: LoaderBase, priorities_dict: dict, doc_id: str):
    us_doc = await api.get_doc(doc_id)
    if not us_doc:
        raise RuntimeError(f"{doc_id} not found")


async def main(
    loader: LoaderBase,
    api: LoaderBase,
    input_path: str | Path,
    path: Path,
    num_of_docs: int | None = None,
    num_of_workers: int = 1,
):
    logger.info("Начало обработки")
    pd_data = pd.read_csv(input_path, delimiter="; ")
    doc_ids = []
    c = Counter()

    for i in pd_data.iterrows():
        d = i[1]

        doc_id = f'{d["PO"]}{d["PN"]}{d["KI"]}_{d["DP"].replace(".", "")}'
        date = datetime.strptime(d["DP"], "%Y.%m.%d")
        doc_ids.append((doc_id, date.month))
        c[date.month] += 1

    doc_ids = doc_ids[:num_of_docs]

    small_dataset = ([i for i in doc_ids if i[1] in [2, 4]], "150")
    medium_dataset = ([i for i in doc_ids if i[1] in [10, 11, 12]], "300")

    logger.info(f"small dataset len: {len(small_dataset)}")
    logger.info(f"medium dataset len: {len(medium_dataset)}")
    # import calendar
    # for k, v in sorted(c.items(), key=itemgetter(0)):
    #     print(f"{calendar.month_name[k]}:  {v}")
    # print(f"{k}:  {v}")

    # return

    def exception_handler(loop, ctx):
        ex = ctx["exception"]
        logger.error(f"{ex}")

    priorities_dict = {}

    for dataset, name in (small_dataset, medium_dataset):
        progress_bar = tqdm(desc=f"Progress ({name})", total=len(dataset))
        dataset_path = path / name
        os.makedirs(dataset_path, exist_ok=True)

        async with CircularTaskGroup(
            num_of_workers, lambda x: progress_bar.update(1), exception_handler
        ) as tg:
            for doc_id, month in dataset:
                await tg.create_task(process_document(loader, dataset_path, doc_id))

        progress_bar.close()

    async def coroutine(doc_id):
        doc = await loader.get_doc(doc_id)
        if not doc:
            raise RuntimeError(f"{doc_id} not found")

    progress_bar = tqdm(desc="Progress", total=len(doc_ids))
    async with CircularTaskGroup(
        num_of_workers, lambda x: progress_bar.update(1), exception_handler
    ) as tg:
        for doc_id, month in doc_ids:
            task = await tg.create_task(coroutine(doc_id))
    progress_bar.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get dataset from mapping file")
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-n", "--number", default=None, type=int)
    parser.add_argument("-w", "--num-of-workers", default=1, type=int)

    args = parser.parse_args()

    api = ESAPILoader(ES_URL)
    data_path = Path("data") / "eval" / args.path
    loader = FSLoader(data_path, data_path, api)

    coro = main(
        loader,
        api,
        args.input,
        BASE_DATA_PATH / "eval" / args.path,
        args.number,
        args.num_of_workers,
    )
    asyncio.run(coro)
