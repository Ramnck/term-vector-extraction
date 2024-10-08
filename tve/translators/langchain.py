import json
import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..base import TranslatorBase
from .prompts import PromptTemplate, en_translate, ru_translate

logger = logging.getLogger(__name__)

# from.llama import Llama

# model = Llama(
#     model_path="E:\\models\\ggufs\\gemma-2-2b-it-Q8_0.gguf",
#     verbose=False,
#     n_ctx=0,
#     n_gpu_layers=-1,
#     n_threads=6,
#     n_threads_batch=2,  # Uncomment to use GPU acceleration
#     # seed=1337, # Uncomment to set a specific seed
#     # n_ctx=2048, # Uncomment to increase the context window
# )
# model = None

# def fix_string(str_to_fix: str) -> str:
#     prompt = 'You are a programmer`s assistant. You are given a broken JSON object. Your task is to fix this JSON object and bring it to this format - {"<string>": [<list of strings>]}. If key doesn`t have quotation marks, so add it. If value is represented by one string, so wrap it into square brackets. Answer me in JSON format, where answer is fixed JSON. Here is broken JSON object: ' + str_to_fix

#     output = model.create_chat_completion(
#         messages=[
#                 {"role": "user", "content": prompt},
#             ],
#             stop=None,
#             response_format={"type": "json_object"},
#             max_tokens=1600,
#     )

#     text = output["choices"][0]["message"]["content"]

#     text_under_braces = text[text.index("{") : text.rindex("}") + 1]

#     return text_under_braces


class LangChainTranslator(TranslatorBase):
    def __init__(
        self,
        agent: BaseChatModel,
        name: str = "langchain",
        lang: str = "en",
        default_prompt: PromptTemplate | None = None,
    ):
        self.llm = agent.bind(response_format={"type": "json_object"})
        self.name = name
        self.lang = lang
        self._default_prompt = default_prompt

    async def make_chat_completion(self, messages: list[BaseMessage]) -> str:
        chunks = [
            chunk.content if isinstance(chunk, BaseMessage) else chunk
            async for chunk in self.llm.astream(messages)
        ]
        text = "".join(chunks)
        return text

    async def fix_json_string(self, str_to_fix: str) -> str:

        prompt = 'You are a programmer`s assistant. You are given a broken JSON object. Your task is to fix this JSON object and bring it to this format - {"<string>": [<list of strings>]}. If key doesn`t have quotation marks, so add it. If value is represented by one string, so wrap it into square brackets. Answer me in JSON format, where answer is fixed JSON. Here is broken JSON object: '

        messages = [
            SystemMessage(prompt),
            HumanMessage(str_to_fix),
        ]

        text = await self.make_chat_completion(messages)

        text_under_braces = text[text.index("{") : text.rindex("}") + 1]

        return text_under_braces

    @property
    def default_prompt(self):
        if self._default_prompt:
            return self._default_prompt

        if self.lang == "en":
            return en_translate
        elif self.lang == "ru":
            return ru_translate
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

        text = await self.make_chat_completion(messages)

        try:
            json_answer = json.loads(text)
        except json.JSONDecodeError:
            try:
                text_under_braces = text[text.index("{") : text.rindex("}") + 1]
                json_answer = json.loads(text_under_braces)
            except json.JSONDecodeError:
                try:
                    model_fixed = "model_fixed не присвоено значение"
                    model_fixed = await self.fix_json_string(text_under_braces)
                    json_answer = json.loads(model_fixed)
                except:
                    json_answer = {}
                    logger.error(
                        "Error in translate_list json serialize: %s\n\nMODEL OUTPUT:%s"
                        % (text, model_fixed)
                    )
            except ValueError:
                json_answer = {}
                logger.error("Answer does not obtain json: %s" % text)
        try:
            out = sum(json_answer.values(), [])
        except TypeError:
            logger.error(f"Error in translate_list sum: {json_answer}")
            out = []
        return out
