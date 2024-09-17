import logging
from operator import itemgetter

import aiohttp
import nltk

from api import TranslatorBase
from lexis import pos_tag_en, pos_tags_ru
from utils import ForgivingTaskGroup

logger = logging.getLogger(__name__)


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

        return {"word": word, "translations": out.split("; ")}

    async def _choose_words_en(
        self, word: str, translations: list[str], num_words: int = 2
    ) -> dict[str, list[str]]:

        # role = "You're an assistant copywriter."
        # task = f"I give you a set of words and phrases in English. Your task is to choose {num_words} words or phrases that mean different parts of speech. For example: 1 verb, 1 noun or 1 noun, 1 adjective or 1 verb, 1 adjective or etc. Or return me {num_words} words or phrases if they all belong to the same part of speech."
        # ans_format = 'Answer me succinctly and without explanation in json array of your suggestions: ["", ""].'

        # sys_prompt = " ".join([role, task, ans_format])

        # prompt = ", ".join(translations)

        # output = llm_llama.create_chat_completion(
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": " ".join([sys_prompt, "Here is set of words:", prompt])
        #             + ".",
        #         },
        #     ],
        #     stop=None,
        # )
        # str_output = output["choices"][0]["message"]["content"]

        # data = json.loads(
        #     str_output[str_output.index("[") : str_output.rindex("]") + 1]
        # )

        data = {"same_pos": [], "diff_pos": []}

        w_tags = pos_tags_ru(word, simplify=True)

        global_pos_priority = ["NOUN", "VERB", "ADJ", "ADV"]
        pos_priority = list(dict.fromkeys(w_tags + global_pos_priority))

        pos_word_dict = {}

        for tr in translations:
            try:
                pos_word_dict[pos_tags_ru(tr)].append(tr)
            except:
                pos_word_dict[pos_tag_en(tr)] = [tr]

        tmp = {k: v.copy() for k, v in pos_word_dict.items()}
        for pos in pos_priority:
            while len(data["same_pos"]) < num_words and len(tmp.get(pos, [])) > 0:
                data["same_pos"].append(tmp[pos].pop(0))

        tmp = {k: v.copy() for k, v in pos_word_dict.items()}
        for pos in pos_priority:
            try:
                data["diff_pos"].append(tmp[pos].pop(0))
            except:
                pass
            if len(data["diff_pos"]) >= num_words:
                break

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

        try:
            # if 1:
            async with ForgivingTaskGroup() as tg:
                for word in words:
                    res.append(
                        tg.create_task(self._translate_word(word, from_lang, to_lang))
                    )
        except* Exception as exs:
            for ex in exs.exceptions:
                logger.error(
                    "Exception in PROMTTranslator.translate - %s" % str(type(ex))
                )
                print(ex)

        out = []

        for future in res:
            try:
                if future.result() is not None:
                    out.append(future.result())
            except:
                pass

        data_dict_list = [
            await self._choose_words_en(
                d["word"], d["translations"], num_words=num_of_suggestions
            )
            for d in out
        ]

        data_output = {
            "same_pos": sum(map(itemgetter("same_pos"), data_dict_list), start=[]),
            "diff_pos": sum(map(itemgetter("same_pos"), data_dict_list), start=[]),
        }

        # for data_dict in out:
        #     words = await self._choose_words_en(
        #         data_dict["translations"], num_words=num_of_suggestions
        #     )
        #     extension += words

        return data_output

    def get_name(self) -> str:
        return "promt"
