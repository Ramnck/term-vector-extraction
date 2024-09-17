import asyncio
import json
import logging
import os
import re

os.environ["TF_ENABLE_ONEDNN_OPTS"] = 0
import aiohttp
from llama_cpp import Llama

from api import TranslatorBase
from utils import ForgivingTaskGroup

llm_llama = Llama(
    model_path="E:\\models\\ggufs\\Vikhr-Gemma-2B-instruct-Q8_0.gguf",
    # model_path="E:\\models\\ggufs\\gemma-2-2b-it-Q8_0.gguf",
    verbose=False,
    n_ctx=0,
    n_gpu_layers=-1,
    n_threads=4,
    n_threads_batch=2,  # Uncomment to use GPU acceleration
    # seed=1337, # Uncomment to set a specific seed
    # n_ctx=2048, # Uncomment to increase the context window
)
logger = logging.getLogger(__name__)


class LLMTranslator(TranslatorBase):
    def __init__(self) -> None:
        # self.llm = Llama(model_path=model_path, n_ctx=0, verbose=False, n_gpu_layers=-1)
        pass

    async def translate(
        self, words: list[str], num_of_suggestions: int = 2, **kwargs
    ) -> list[str]:
        # role = 'Ты - помощник копирайтера.'
        role = "You're an assistant copywriter."

        task = f"Тебе подаётся набор слов, твоя задача - каждое слово перевести на английский язык и предложить {num_of_suggestions} наиболее близких по смыслу перевода. Учитывай все слова что-бы понять общую тематику набора слов."
        # task = f"You are given a set of words in Russian, your task is to translate each word into English and offer {num_of_suggestions} closest translations. Take into account all the words to understand the general theme of the set of words."

        ans_format = "Отвечай кратко, без пояснений, в ответе укажи все предложенные слова через запятую."
        # ans_format = 'Отвечай кратко, без пояснений в формате json - {"слово на русском": ["перевод 1", "перевод 2"]}.'
        # ans_format = 'Give me the answer in json format - [ {"word_ru": "", "translation_en": ["", ""]}, ...].'

        # ans_format = "Answer briefly and without explanation, give me only set of all offered words separated by comma."

        # connector = "Set of words:"
        connector = "Набор слов:"

        sys_prompt = " ".join([role, task, ans_format])
        usr_prompt = ", ".join(words)
        prompt = " ".join([sys_prompt, connector, usr_prompt]) + "."

        translations = {}

        output = llm_llama.create_chat_completion(
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

    def get_name(self) -> str:
        return "llm"
