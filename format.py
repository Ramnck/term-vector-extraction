import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

from tve.pipeline import BASE_DATA_PATH
from tve.utils import load_data_from_json

logger = logging.getLogger(__name__)


def format_id(pat_id: str) -> str:
    pattern = r"([A-Z]{2})(\d+)([A-ZА-Я]\d?)_(\d+)"
    match = re.findall(pattern, pat_id)[0]
    off = match[0]
    num = match[1].lstrip("0")
    kind = match[2]
    date = match[3]
    date = ".".join((date[:4], date[4:6], date[-2:]))
    out = "; ".join((off, num, kind, date))
    return out


async def main(input_path: str, output_path: str | None, priority: bool = False):
    errors = Counter()
    input_dir_path = BASE_DATA_PATH / "eval" / input_path
    dir_path = BASE_DATA_PATH / "eval" / output_path

    for file_path in input_dir_path.iterdir():
        data = await load_data_from_json(file_path)
        doc_id = data["doc_id"] if not priority else data.get("cluster", ["", ""])[1]
        all_relevant = data["relevant"]

        for method_name, relevant in all_relevant.items():
            # extractor_name, num = method_name.split("_")
            if len(relevant) == 0:
                errors[method_name + "_no_rel"] += 1
                continue
            if isinstance(relevant[0], list):
                relevant = relevant[0]
            extractor_name = method_name
            path = dir_path / extractor_name
            os.makedirs(path, exist_ok=True)
            with open(path / "mapping.txt", "a+", encoding="utf-8") as file:
                result_path = path
                os.makedirs(result_path, exist_ok=True)
                number = len(list(result_path.iterdir()))
                name_of_result = f"result_list{number}.txt"
                file.write(f"{doc_id} {name_of_result}\n")
            with open(result_path / name_of_result, "w+", encoding="utf-8") as file:
                for res in relevant:
                    try:
                        file.write(format_id(res) + "\n")
                    except Exception as ex:
                        logger.error(f"Error in main: {res}")
                        # raise ex

    print("Errors:", errors)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Форматировать вывод моей программы в вывод читаемый для утилиты оценки качества"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--priority", action="store_true", default=False)

    args = parser.parse_args()

    coro = main(
        args.input,
        args.output,
        args.priority,
    )
    asyncio.run(coro)
