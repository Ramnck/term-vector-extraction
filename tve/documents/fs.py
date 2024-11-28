import logging
import os
import random
import re
from itertools import compress
from pathlib import Path

import aiofiles

from ..base import DocumentBase, LoaderBase
from ..lexis import extract_number
from .fips import JSONDocument
from .xml import XMLDocument

logger = logging.getLogger(__name__)


class FSLoader(LoaderBase):
    def __init__(
        self,
        path: str | Path,
        cache_path: str | Path | None = None,
        api_to_load_files: LoaderBase | None = None,
    ) -> None:
        self.init_path = Path(path).absolute()
        self.api = api_to_load_files

        if cache_path is not None:
            self.cache_path = Path(cache_path) / self.init_path.name
            if not self.cache_path.exists():
                os.makedirs(self.cache_path)
        else:
            self.cache_path = None

    async def _open_file(self, path: Path | str) -> DocumentBase:
        path = Path(path)
        docs = {".xml": XMLDocument, ".json": JSONDocument}
        doc = docs[path.suffix].load_file(path)
        return doc

    async def get_doc(
        self, id_date: str, check_cache: bool = True, timeout: int = 10
    ) -> DocumentBase | None:
        # num_of_doc = extract_number(id_date).lstrip("0")
        start_paths = [self.init_path]
        if self.cache_path and check_cache:
            start_paths.append(self.cache_path)
            paths = [i for i in self.cache_path.parent.iterdir() if i.is_dir()]
            paths.remove(self.cache_path)
            start_paths += paths

        for start_path in start_paths:
            for path in start_path.iterdir():
                if path.is_dir():
                    file_paths = list(path.iterdir())
                elif path.is_file():
                    file_paths = [path]

                for file_path in file_paths:
                    if extract_number(file_path.stem) == "":
                        name = file_path.parent.stem
                    else:
                        name = file_path.stem
                    match = re.search(r"([A-Z]{2})?(\d+)", id_date.replace("/", ""))

                    if not match:
                        logger.error("invalid id specified: %s" % id_date)
                        return None

                    groups = match.groups()

                    if all(map(lambda x: x in name, compress(groups, groups))):
                        return await self._open_file(file_path)

        if self.api and self.cache_path:
            doc = await self.api.get_doc(id_date, timeout)
            if doc:
                doc.save_file(self.cache_path)
                return doc

        # logger.error(id_date + " document not found")
        return None

    async def get_random_doc(self) -> DocumentBase:
        list_of_files = list(self.init_path.iterdir())
        doc_path = random.choice(list_of_files)
        if doc_path.is_dir():
            doc_path = random.choice(list(doc_path.iterdir()))
        return await self._open_file(doc_path)

    async def find_relevant_by_keywords(self, kws) -> list:
        raise NotImplementedError

    def __aiter__(self):
        self.diriter = self.init_path.iterdir()
        return self

    async def __anext__(self) -> DocumentBase:
        try:
            doc_path = next(self.diriter)
        except StopIteration:
            raise StopAsyncIteration

        if doc_path.is_dir():
            for file in doc_path.iterdir():
                doc_path = file

        return await self._open_file(doc_path)
