import logging
import urllib.request
from operator import itemgetter
from urllib.parse import quote

import aiohttp
import nltk
from base import TranslatorBase
from lexis import pos_tag_en, pos_tags_ru
from utils import ForgivingTaskGroup

logger = logging.getLogger(__name__)


class PROMTTranslator(TranslatorBase):
    def __init__(self, ip: str) -> None:
        self.ip = ip
        self.api_url = f"http://{ip}/twsas/Services/v1/rest.svc"

    async def _translate_word(
        self,
        word: str,
        from_lang: str,
        to_lang: str,
        profile: str = "Универсальный",
        remove_linebreaks: bool = True,
    ) -> dict:

        word = quote(word)
        profile = quote(profile)

        req_data = f"text={word}&from={from_lang}&to={to_lang}&profile={profile}"

        req_url = self.api_url + "/TranslateTextWithED" + "?" + req_data

        with urllib.request.urlopen(req_url) as response:
            out = response.read().decode(response.headers.get_content_charset())

        return {"word": word, "translations": out.split("; ")}

    async def _choose_words_en(
        self, word: str, translations: list[str], num_words: int = 2
    ) -> dict[str, list[str]]:

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

    async def translate_list(
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
            "diff_pos": sum(map(itemgetter("diff_pos"), data_dict_list), start=[]),
        }

        # for data_dict in out:
        #     words = await self._choose_words_en(
        #         data_dict["translations"], num_words=num_of_suggestions
        #     )
        #     extension += words

        return data_output

    def get_name(self) -> str:
        return "promt"
