import argparse
import json
import logging
import os
import sys
import time

import aiofiles
import asyncio
import numpy as np
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

from api import LoaderBase
from chain import (
    BASE_DATA_PATH,
    FIPS_API_KEY,
    extract_keywords_from_docs,
    get_cluster_from_document,
    get_relevant,
    load_data_from_json,
    save_data_to_json,
    test_different_vectors,
)
from documents import FileSystem, FipsAPI

logger = logging.getLogger(__name__)


async def main(
    loader: LoaderBase,
    api: LoaderBase,
    num_of_docs: int | None,
    name_of_experiment: str,
):

    num_of_doc = 0
    async for doc in tqdm_asyncio(
        aiter(loader), total=num_of_docs, desc="Progress"
    ):
        num_of_doc += 1

        path_of_file = (
            BASE_DATA_PATH
            / "eval"
            / name_of_experiment
            / (doc.id_date + ".json")
        )
        data = await load_data_from_json(path_of_file)
        if not data:
            logger.error("File %s not found" % path_of_file.name)
            num_of_doc += 1
            continue

        relevant = await test_different_vectors(
            data["keywords"],
            ["expand", "mix", "shuffle", "raw"],
            [75, 100, 125, 150, 175, 200],
            api,
        )

        data["relevant"] = relevant

        path_of_file = (
            BASE_DATA_PATH
            / "eval"
            / (name_of_experiment + "_rel")
            / (doc.id_date + ".json")
        )
        if await save_data_to_json(data, path_of_file):
            logger.error(
                "Error occured while saving %s file" % path_of_file.name
            )

        if num_of_docs is not None:
            if num_of_doc >= num_of_docs:
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get relevant using term vectors vectors"
    )
    parser.add_argument("-i", "--input", default="clusters")
    parser.add_argument("-o", "--output", default="80")
    parser.add_argument("-n", "--number", default=None, type=int)

    args = parser.parse_args()

    api = FipsAPI(FIPS_API_KEY)
    loader = FileSystem(Path("data") / "raw" / args.input)

    coro = main(loader, api, args.number, args.output)
    asyncio.run(coro)
