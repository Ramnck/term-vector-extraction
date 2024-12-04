import asyncio
import json
from pathlib import Path
from types import TracebackType
from typing import Callable, Iterable

import aiofiles
from tqdm import tqdm


class ForgivingTaskGroup(asyncio.TaskGroup):
    _abort = lambda self: None

    def __init__(self, progress_bar: tqdm | None = None) -> None:
        self._progress_bar = progress_bar
        super().__init__()

    def _on_task_done(self, task: asyncio.Task[object]) -> None:
        super(ForgivingTaskGroup, self)._on_task_done(task)
        if self._progress_bar:
            self._progress_bar.update(1)


class CircularTaskGroup:
    def __init__(
        self,
        num_of_workers: int,
        default_callback: Callable | None = None,
        exception_handler: Callable = lambda loop, ctx: print(
            "EXCEPTION", ctx["exception"]
        ),
    ) -> None:
        self._default_callback = default_callback
        self._num_of_workers = num_of_workers
        self._exception_handler = exception_handler

    async def __aenter__(self):
        self._tasks = []
        asyncio.get_event_loop().set_exception_handler(self._exception_handler)
        return self

    async def create_task(self, coroutine):
        while len(self._tasks) >= self._num_of_workers:
            self._tasks = [i for i in self._tasks if not i.done()]
            await asyncio.sleep(0.01)
        task = asyncio.create_task(coroutine)

        if self._default_callback:
            task.add_done_callback(self._default_callback)
        self._tasks.append(task)
        return task

    async def __aexit__(self, exc_type, exc, tb):
        if len(self._tasks) > 0:
            await asyncio.wait(self._tasks)
        asyncio.get_event_loop().set_exception_handler(None)


def batched(iterable, n=1):
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
    obj: dict | list, path_of_file: Path, noexcept: bool = False
) -> bool:  # False on success
    """Return true if error occurs and noexcept==True"""
    try:
        async with aiofiles.open(
            path_of_file,
            "w+",
            encoding="utf-8",
        ) as file:
            await file.write(json.dumps(obj, ensure_ascii=False, indent=4))
        return False
    except FileNotFoundError as ex:
        if noexcept:
            return True
        else:
            raise ex


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


def flatten_kws(input_data, stopword: str = "") -> list[str]:
    def recursive_unwrap(data):
        if isinstance(data, dict):
            return list(data.keys()) + list(
                sum(map(recursive_unwrap, data.values()), [])
            )
        elif isinstance(data, list):
            return sum(map(recursive_unwrap, data), [])
        elif isinstance(data, str):
            return [data]
        else:
            return []

    out = []

    for i in recursive_unwrap(input_data):
        i = i.strip()
        if stopword:
            if stopword in i:
                continue
        out.append(i)

    return out
