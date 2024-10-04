import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..base import TranslatorBase
from .prompts import en_promt, ru_promt

logger = logging.getLogger(__name__)

# from.llama import Llama

# model = Llama(
#             model_path="E:\\models\\ggufs\\gemma-2-2b-it-Q8_0.gguf",
#             verbose=False,
#             n_ctx=0,
#             n_gpu_layers=-1,
#             n_threads=6,
#             n_threads_batch=2,  # Uncomment to use GPU acceleration
#             # seed=1337, # Uncomment to set a specific seed
#             # n_ctx=2048, # Uncomment to increase the context window
#         )


class LangChainTranslator(TranslatorBase):
    def __init__(self, agent: BaseChatModel, name: str = "langchain", lang: str = "en"):
        self.llm = agent.bind(response_format={"type": "json_object"})
        self.name = name
        self.lang = lang

    @property
    def default_prompt(self):
        if self.lang == "en":
            return en_promt
        elif self.lang == "ru":
            return ru_promt
        else:
            raise ValueError(f"Language {self.lang} is not supported")

    def system_prompt(self, num_of_words: int | str = 2) -> str:
        prompt = " ".join(
            (
                self.default_prompt.role,
                self.default_prompt.task.format(num_of_words),
                self.default_prompt.answer_format,
            )
        )
        return prompt

    def human_promt(self, text: str) -> str:
        prompt = " ".join((self.default_prompt.tail, text))
        return prompt

    async def translate_list(
        self, words: list[str], num_of_suggestions: int = 2, **kwargs
    ) -> list[str]:
        messages = [
            SystemMessage(self.system_prompt(num_of_suggestions)),
            HumanMessage(self.human_promt(", ".join(words))),
        ]

        chunks = [
            chunk.content if isinstance(chunk, BaseMessage) else chunk
            async for chunk in self.llm.astream(messages)
        ]
        text = "".join(chunks)
        try:
            json_answer = json.loads(text)
        except json.JSONDecodeError:
            try:
                json_answer = json.loads(text[text.index("{") : text.rindex("}") + 1])
            except json.JSONDecodeError:
                json_answer = {}
                logger.error("Error in translate_list: %s" % text)
        try:
            out = sum(json_answer.values(), [])
        except TypeError:
            logger.error(f"Error in translate_list sum: {json_answer}")
            out = []
        return out
