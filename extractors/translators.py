import asyncio
import json
import logging
import re

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

        # while len(translations) < len(words):
        output = llm_llama.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt},
            ],
            stop=None,
            response_format={"type": "json_object"},
            max_tokens=1600,
        )

        str_output = output["choices"][0]["message"]["content"]

        # return list(set(str_output.split(", ")))

        # data_list = re.findall(r'{.+?}', str_output.replace("\n", ""))

        # data_dict = {}

        # for i in data_list:
        #     if len(data_dict) < len(words):
        #         j = json.loads(i)
        #         data_dict["word_ru"] = j["translation_en"]
        #     else:
        #         break

        # print(str_output)

        words = [i.strip() for i in re.split(r"\s*,\s*", str_output.replace("\n", ""))]

        extension = set(words)

        # extension = set()

        # if "}" not in str_output:
        #     str_output = str_output[:str_output.rindex("]") + 1] + "}"

        # try:
        #     data_dict = json.loads(str_output)
        # except json.JSONDecodeError as ex:
        #     logger.error(str_output)
        #     raise ex

        # try:
        #     for word, translations in data_dict.items():
        #         extension.update(translations)
        # except TypeError as ex:
        #     logger.error(str_output)
        #     logger.error(translations)
        #     raise ex

        return list(extension)

    def get_name(self) -> str:
        return "llm"


class PROMTTranslator(TranslatorBase):
    def __init__(self) -> None:
        self.api_url = "http://rospsearch01/twsas/Services/v1/rest.svc"

    async def _translate_word(
        self,
        word: str,
        from_lang: str,
        to_lang: str,
        profile: str = "Универсальный",
        remove_linebreaks: bool = True,
    ) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url + "/TranslateTextWithED",
                data={
                    "from": from_lang,
                    "to": to_lang,
                    "profile": profile,
                    "removeLinebreaks": remove_linebreaks,
                    "text": word,
                },
            ) as res:
                out = await res.text(encoding="utf-8")

        return {"text": word, "translations": out.split("; ")}

    async def _choose_words(
        self, word: str, translations: list[str], num_words: int = 2
    ) -> list[str]:

        role = "You're an assistant copywriter."
        task = f"I give you a set of words and phrases in English. Your task is to choose {num_words} words or phrases that mean different parts of speech. For example: 1 verb, 1 noun or 1 noun, 1 adjective or 1 verb, 1 adjective or etc. Or return me {num_words} words or phrases if they all belong to the same part of speech."
        ans_format = 'Answer me succinctly and without explanation in json array of your suggestions: ["", ""].'

        sys_prompt = " ".join([role, task, ans_format])

        prompt = ", ".join(translations)

        output = llm_llama.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": " ".join([sys_prompt, "Here is set of words:", prompt])
                    + ".",
                },
            ],
            stop=None,
        )
        str_output = output["choices"][0]["message"]["content"]

        data = json.loads(
            str_output[str_output.index("[") : str_output.rindex("]") + 1]
        )

        return data

    async def translate(
        self,
        words: list[str],
        from_lang: str = "ru",
        to_lang: str = "en",
        num_of_suggestions: int = 2,
        **kwargs,
    ) -> list[str]:

        res = []

        if 1:
            # try:
            async with ForgivingTaskGroup() as tg:
                for word in words:
                    res.append(
                        tg.create_task(self._translate_word(word, from_lang, to_lang))
                    )
        # except* Exception as exs:
        # for ex in exs.exceptions:
        #     logger.error("Exception in get_cluster_from_document - %s" % str(type(ex)))

        out = []

        for future in res:
            try:
                if future.result() is not None:
                    out.append(future.result())
            except:
                pass

        extension = []

        for data_dict in out:
            words = await self._choose_words(
                data_dict["translations"], num_words=num_of_suggestions
            )
            extension += words

        return extension

    def get_name(self) -> str:
        return "promt"
