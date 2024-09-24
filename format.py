import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from itertools import product
from pathlib import Path

from tve.pipeline import BASE_DATA_PATH
from tve.utils import load_data_from_json


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


async def main(
    input_path: str,
    output_path: str | None,
):
    input_dir_path = BASE_DATA_PATH / "eval" / input_path
    dir_path = BASE_DATA_PATH / "eval" / output_path

    for file_path in input_dir_path.iterdir():
        data = await load_data_from_json(file_path)
        doc_id = data["doc_id"]
        doc_analog = data["cluster"][1]
        all_relevant = data["relevant"]

        for method_name, relevant in all_relevant.items():
            extractor_name, num = method_name.split("_")
            path = dir_path / extractor_name / num
            os.makedirs(path, exist_ok=True)
            with open(path / "mapping.txt", "a+", encoding="utf-8") as file:
                result_path = path / "result_lists"
                os.makedirs(result_path, exist_ok=True)
                number = len(list(result_path.iterdir()))
                name_of_result = f"result_list{number}.txt"
                file.write(f"{doc_analog} {name_of_result}\n")
            with open(result_path / name_of_result, "w+", encoding="utf-8") as file:
                for res in relevant:
                    file.write(format_id(res) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Форматировать вывод моей программы в вывод читаемый для утилиты оценки качества"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)

    args = parser.parse_args()

    coro = main(
        args.input,
        args.output,
    )
    asyncio.run(coro)
