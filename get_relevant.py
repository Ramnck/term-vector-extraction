import os
import logging
import asyncio
import time
from pathlib import Path
import json
import numpy as np
import aiofiles
import sys

from chain import (
    get_cluster_from_document,
    extract_keywords_from_docs,
    get_relevant,
    save_data_to_json,
    load_data_from_json,
    test_different_vectors,
    FIPS_API_KEY,
    BASE_DATA_PATH,
)
from tqdm.asyncio import tqdm_asyncio
from documents import FipsAPI, FileSystem


logger = logging.getLogger(__name__)


async def main(name_of_experiment: str, num_of_docs: int):

    loader = FileSystem(BASE_DATA_PATH / "raw" / "clusters")
    api = FipsAPI(FIPS_API_KEY)

    num_of_doc = 0
    async for doc in tqdm_asyncio(aiter(loader), total=num_of_docs, desc="Progress"):
        path_of_file = (
            BASE_DATA_PATH / "eval" / name_of_experiment / (doc.id_date + ".json")
        )
        data = await load_data_from_json(path_of_file)
        if not data:
            logger.error("File %s not found" % path_of_file.name)
            num_of_doc += 1
            continue

        relevant = await test_different_vectors(
            data["keywords"], ["expand", "mix"], [100, 125, 150, 175, 200], api
        )

        data["relevant"] = relevant

        path_of_file = (
            BASE_DATA_PATH
            / "eval"
            / (name_of_experiment + "_rel")
            / (doc.id_date + ".json")
        )
        if await save_data_to_json(data, path_of_file):
            logger.error("Error occured while saving %s file" % path_of_file.name)

        num_of_doc += 1
        if num_of_doc >= num_of_docs:
            break


if __name__ == "__main__":
    asyncio.run(main("term_vectors", 2))
