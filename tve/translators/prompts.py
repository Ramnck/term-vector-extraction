from dataclasses import dataclass


@dataclass
class DefaultPrompt:
    role: str
    task: str
    answer_format: str
    tail: str


role_ru = "Ты - помощник переводчика."
role_en = "You are a translator's assistant."

task_ru = "Тебе подаётся набор слов или фраз, твоя задача - каждое слово перевести на английский язык и предложить {} наиболее близких по смыслу перевода или слова/фразы с одинаковым смыслом. Учитывай все слова что-бы понять общую тематику набора слов."
task_en = "You are given a set of words/phrases in Russian, your task is to translate each word into English and offer {} closest translations or words/phrases with a similar meaning. Take into account all the words to understand the general theme of the set of words."

answer_format_ru = 'Отвечай кратко, без пояснений в формате JSON, где ключ - слово для перевода, значение - переводы. Например: {"<слово на русском>": ["<список переводов>"]}.'
answer_format_en = 'Give me the answer in JSON format where key is word to translate, value is list of translations. Example: {"<ru word>": [<list of translations>]}.'


tail_ru = "Набор слов:"
tail_en = "Set of words:"

en_promt = DefaultPrompt(
    role=role_en, task=task_en, answer_format=answer_format_en, tail=tail_en
)
ru_promt = DefaultPrompt(
    role=role_ru, task=task_ru, answer_format=answer_format_ru, tail=tail_ru
)
