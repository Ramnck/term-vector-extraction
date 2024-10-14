from dataclasses import dataclass


@dataclass
class PromptTemplate:
    role: str
    task: str
    answer_format: str
    tail: str

    def __str__(self) -> str:
        return " ".join([self.role, self.task, self.answer_format, self.tail, ""])

    def __repr__(self) -> str:
        return self.__str__()


role_ru = "Ты - помощник переводчика."
role_en = "You are a translator's assistant."

task_ru = "Тебе подаётся набор слов или фраз, твоя задача - каждое слово перевести на английский язык и предложить {} наиболее близких по смыслу перевода или слова/фразы с одинаковым смыслом, постарайся предложить наиболее разнообразные варианты. Учитывай все слова что-бы понять общую тематику набора слов."
task_en = "You are given a set of words/phrases in Russian, your task is to translate each word into English and offer {} closest translations or words/phrases with a similar meaning, try to offer the most diverse candidates. Take into account all the words to understand the general theme of the set of words."

answer_format_ru = 'Отвечай кратко, без пояснений в формате JSON, где ключ - слово для перевода, значение - переводы. Например: {"<слово на русском>": ["<список переводов>"]}.'
answer_format_en = 'Give me the answer in JSON format where key is word to translate, value is list of translations. Example: {"<ru word>": [<list of translations>]}.'


tail_ru = "Набор слов:"
tail_en = "Set of words:"

en_translate = PromptTemplate(
    role=role_en, task=task_en, answer_format=answer_format_en, tail=tail_en
)
ru_translate = PromptTemplate(
    role=role_ru, task=task_ru, answer_format=answer_format_ru, tail=tail_ru
)

r = "You are an assistant patent attorney."
t = (
    "You are given a set of words/phrases from patent application, your task is to suggest {} synonyms or words/phrases with similar or close meaning, try to offer the most diverse options from those used in technical texts and/or patent documents. "
    "Take into account all the words to understand the general theme of the set of words."
)
ans = 'Give me the answer in JSON format where key is input word/phrase, value is list of suggestions. Example: {"<word/phrase>": [<list of suggestions>]}.'
e = "Here is set of words/phrases:"

en_expand_prompt = PromptTemplate(role=r, task=t, answer_format=ans, tail=e)

r = "Вы – помощник патентного поверенного."
t = (
    "Вам дан набор слов/фраз из патентной заявки, ваша задача - предложить {} синонимa или слова/фразы с похожим или близким по смыслу значением, постарайтесь предложить наиболее разнообразные варианты из тех, которые используются в технических текстах и/или патентных документах. "
    "Учитывайте все слова, чтобы понять общую тему набора слов. "
    "Предлагайте слова/фразы на английском языке."
)
ans = 'Дайте мне ответ в формате JSON, где ключ - входной термин/фраза, значение - список предложений. Формат ответа: {"<термин/фраза>": ["<предложение 1>", "<предложение 2>", "<предложение 3>"]}.'
# "Пример входного списка терминов/фраз: code, record. Пример ответа: {"code": ["program", "instructions", "algorithm"], "record": ["log", "note", "file"]}"
e = "Вот список терминов/фраз:"

ru_expand_prompt = PromptTemplate(role=r, task=t, answer_format=ans, tail=e)
