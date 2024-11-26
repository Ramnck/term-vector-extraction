import logging
import urllib.request
from operator import itemgetter
from urllib.parse import quote

import aiohttp
import nltk

from ..base import TranslatorBase
from ..lexis import pos_tag_en, pos_tags_ru
from ..utils import ForgivingTaskGroup

logger = logging.getLogger(__name__)


class PROMTTranslator(TranslatorBase):
    def __init__(self, ip: str, enable_cache: bool = False) -> None:
        self.name = "promt"
        self.ip = ip
        self.api_url = f"http://{ip}/twsas/Services/v1/rest.svc"

        self.cache = {} if enable_cache else None

    async def _translate_word(
        self,
        word: str,
        from_lang: str,
        to_lang: str,
        profile: str = "Универсальный",
        remove_linebreaks: bool = True,
    ) -> dict:

        if self.cache is None or word not in self.cache.keys():
            q_word = quote(word)
            profile = quote(profile)

            req_data = f"text={q_word}&from={from_lang}&to={to_lang}&profile={profile}"

            req_url = self.api_url + "/TranslateText" + "?" + req_data
            # req_url = self.api_url + "/TranslateTextWithED" + "?" + req_data

            with urllib.request.urlopen(req_url) as response:
                out = response.read().decode(response.headers.get_content_charset())

            out = out.replace('"', "").split("; ")
            if self.cache is not None:
                self.cache[word] = {"n": 1, "tr": out}
        else:
            self.cache[word]["n"] += 1
            out = self.cache[word]["tr"]

        return {"word": word, "translations": out}

    # async def _choose_words_en(
    #     self, word: str, translations: list[str], num_words: int = 2
    # ) -> dict[str, list[str]]:

    #     data = []

    #     w_tags = pos_tags_ru(word, simplify=True)

    #     global_pos_priority = ["NOUN", "VERB", "ADJ", "ADV"]
    #     pos_priority = list(dict.fromkeys(w_tags + global_pos_priority))

    #     pos_word_dict = {}

    #     for tr in translations:
    #         try:
    #             pos_word_dict[pos_tags_ru(tr)].append(tr)
    #         except:
    #             pos_word_dict[pos_tag_en(tr)] = [tr]

    #     tmp = {k: v.copy() for k, v in pos_word_dict.items()}
    #     for pos in pos_priority:
    #         while len(data) < num_words and len(tmp.get(pos, [])) > 0:
    #             data.append(tmp[pos].pop(0))

    #     return data

    async def translate_list(
        self,
        words: list[str],
        from_lang: str = "ru",
        to_lang: str = "en",
        num_of_suggestions: int = 2,
        noexcept: bool = True,
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
            pass
            # for ex in exs.exceptions:
            #     logger.error(
            #         "Exception in PROMTTranslator.translate - %s" % str(type(ex))
            #     )

        out = []

        for future in res:
            try:
                if future.result() is not None:
                    out.append(future.result())
            except Exception as ex:

                if noexcept:
                    logger.error(
                        "Exception in PROMTTranslator.translate - %s" % str(type(ex))
                    )
                else:
                    raise ex

        data_dict = {d["word"]: d["translations"] for d in out}

        # data_output = {
        #     "same_pos": sum(map(itemgetter("same_pos"), data_dict_list), start=[]),
        #     "diff_pos": sum(map(itemgetter("diff_pos"), data_dict_list), start=[]),
        # }

        # for data_dict in out:
        #     words = await self._choose_words_en(
        #         data_dict["translations"], num_words=num_of_suggestions
        #     )
        #     extension += words

        return data_dict
