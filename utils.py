import asyncio
import json
from pathlib import Path

import aiofiles


class ForgivingTaskGroup(asyncio.TaskGroup):
    _abort = lambda self: None


def batched(iterable, n=1):
    l = len(iterable)

    iterator = iter(iterable)
    batch = []
    try:
        while True:
            batch = []
            for _ in range(n):
                batch.append(next(iterator))
            yield batch
    except StopIteration:
        if batch:
            yield batch


async def save_data_to_json(
    obj: dict | list, path_of_file: Path
) -> bool:  # False on success
    try:
        async with aiofiles.open(
            path_of_file,
            "w+",
            encoding="utf-8",
        ) as file:
            await file.write(json.dumps(obj, ensure_ascii=False, indent=4))
        return False
    except FileNotFoundError:
        return True


async def load_data_from_json(path_of_file: Path) -> dict | None:
    try:
        async with aiofiles.open(
            path_of_file,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.loads(await file.read())
        return data
    except FileNotFoundError:
        return None
