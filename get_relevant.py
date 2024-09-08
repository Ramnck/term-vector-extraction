import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import aiofiles
import numpy as np
from tqdm.asyncio import tqdm_asyncio

from api import LoaderBase
from chain import (
    BASE_DATA_PATH,
    ES_URL,
    FIPS_API_KEY,
    extract_keywords_from_docs,
    get_cluster_from_document,
    get_relevant,
    load_data_from_json,
    save_data_to_json,
    test_different_vectors,
)
from documents import FileSystem, FipsAPI, InternalESAPI

logger = logging.getLogger(__name__)


async def main(
    loader: LoaderBase,
    api: LoaderBase,
    num_of_docs: int | None,
    input_path: str,
    output_path: str | None,
):

    num_of_doc = 0
    async for doc in tqdm_asyncio(aiter(loader), total=num_of_docs, desc="Progress"):
        num_of_doc += 1

        path_of_file = BASE_DATA_PATH / "eval" / input_path / (doc.id_date + ".json")
        data = await load_data_from_json(path_of_file)
        if not data:
            logger.error("File %s not found" % path_of_file.name)
            continue

        relevant = await test_different_vectors(
            data["keywords"],
            ["expand", "mix", "shuffle", "raw"],
            list(range(75, 401, 25)),
            api,
        )

        # keywords = {k: v[0] for k, v in data["keywords"].items()}

        # relevant = await get_relevant(keywords, api)

        data["relevant"] = relevant

        path_of_file = BASE_DATA_PATH / "eval" / output_path / (doc.id_date + ".json")
        if await save_data_to_json(data, path_of_file):
            logger.error("Error occured while saving %s file" % path_of_file.name)

        if num_of_docs is not None:
            if num_of_doc >= num_of_docs:
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Get relevant using term vectors vectors"
    )
    parser.add_argument("-i", "--input", default="80")
    parser.add_argument("-o", "--output", default="80_rel")
    parser.add_argument(
        "-d",
        "--docs",
        default="clusters",
        help="Where to iterate over document id's (path to xmls)",
    )
    parser.add_argument("-n", "--number", default=None, type=int)

    args = parser.parse_args()

    # api = FipsAPI(FIPS_API_KEY)
    api = InternalESAPI(ES_URL)
    loader = FileSystem(Path("data") / "raw" / args.docs)

    coro = main(loader, api, args.number, args.input, args.output)
    asyncio.run(coro)
