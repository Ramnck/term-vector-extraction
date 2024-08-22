import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import aiofiles
import numpy as np
from sentence_transformers import SentenceTransformer
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
from extractors import (
    KeyBERTExtractor,
    KeyBERTModel,
    RuLongrormerEmbedder,
    YAKExtractor,
)

logger = logging.getLogger(__name__)

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
        text_extraction_func=lambda doc: "query: " + doc.text,
    ),
]


async def main(
    loader: LoaderBase,
    api: LoaderBase,
    num_of_docs: int | None,
    input_path: str,
    output_path: str | None,
):

    if output_path is None:
        output_path = input_path

    num_of_doc = 0
    async for doc in tqdm_asyncio(aiter(loader), total=num_of_docs, desc="Progress"):
        num_of_doc += 1
        path_of_file = BASE_DATA_PATH / "eval" / input_path / (doc.id_date + ".json")
        data = await load_data_from_json(path_of_file)
        if not data:
            logger.error("File %s not found" % path_of_file.name)
            continue

        keywords = data["keywords"]

        if len(keywords.get("YAKE", [])) >= len(doc.cluster):
            continue

        cluster = await get_cluster_from_document(doc, api)

        keywords = await extract_keywords_from_docs(cluster, extractors)

        data["keywords"] = keywords

        path_of_file = BASE_DATA_PATH / "eval" / output_path / (doc.id_date + ".json")
        if await save_data_to_json(data, path_of_file):
            logger.error("Error occured while saving %s file" % path_of_file.name)

        if num_of_docs is not None:
            if num_of_doc >= num_of_docs:
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Update keywords(repeat querries) after get_keywords.py"
    )

    parser.add_argument("-i", "--input", default="500")
    parser.add_argument("-o", "--output", default="500")
    parser.add_argument(
        "-d",
        "--docs",
        default="500RU",
        help="Where to iterate over document id's (path to xmls)",
    )
    parser.add_argument("-n", "--number", default=None, type=int)

    args = parser.parse_args()

    api = FipsAPI(FIPS_API_KEY)
    loader = FileSystem(Path("data") / "raw" / args.docs)

    coro = main(loader, api, args.number, args.input, args.output)
    asyncio.run(coro)
