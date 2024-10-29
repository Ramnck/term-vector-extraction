import asyncio
import json
import logging
import os
import re

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import aiohttp
from llama_cpp import Llama

from ..base import TranslatorBase
from ..prompts import en_translate, ru_translate
from ..utils import ForgivingTaskGroup

logger = logging.getLogger(__name__)


class LlamaTranslator(TranslatorBase):
    def __init__(self, model_path, name: str = "llama") -> None:
        self.llm_llama = Llama(
            model_path=model_path,
            # model_path="E:\\models\\ggufs\\gemma-2-2b-it-Q8_0.gguf",
            verbose=False,
            n_ctx=0,
            n_gpu_layers=-1,
            n_threads=4,
            n_threads_batch=2,  # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            # n_ctx=2048, # Uncomment to increase the context window
        )
        self.name = name
        # self.llm = Llama(model_path=model_path, n_ctx=0, verbose=False, n_gpu_layers=-1)
        pass

    async def translate_list(
        self, words: list[str], num_of_suggestions: int = 2, **kwargs
    ) -> list[str]:

        default_prompt = ru_translate
        # default_prompt = en_translate

        role = default_prompt.role
        task = default_prompt.task.format(num_of_suggestions)
        ans_format = default_prompt.answer_format
        tail = default_prompt.tail

        sys_prompt = " ".join([role, task, ans_format])
        usr_prompt = ", ".join(words)
        prompt = " ".join([sys_prompt, tail, usr_prompt]) + "."

        translations = {}

        output = self.llm_llama.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt},
            ],
            stop=None,
            response_format={"type": "json_object"},
            max_tokens=1600,
        )

        str_output = output["choices"][0]["message"]["content"]

        words = [i.strip() for i in re.split(r"\s*,\s*", str_output.replace("\n", ""))]

        extension = set(words)

        return list(extension)
