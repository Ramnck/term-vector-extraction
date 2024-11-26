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

r = "You are a patent attorney."
t = (
    # "You have been given a set of terms/phrases from a patent application, your task is to suggest {} synonyms or words/phrases most commonly found in technical/patent documents together with each term from the set and contextually close to it, try to suggest the most diverse options. "
    # "Take into account all the words to understand the general theme of the set of terms/phrases."
    "You are given a set of terms/collocations in Russian from a patent application, your task is to propose {} words or collocations in English most frequently occurring in technical/patent documents together with each term from the set and contextually close to it, try to propose the most diverse variants. "
    "Each suggested collocation is no longer than two words. "
    "Consider all words to understand the overall theme of the set of words. "
    # "Suggest words or phrases in English only, do not use Russian language. "
)
# ans = 'Give me the answer in JSON format where key is input terms/phrases, value is list of suggestions. Example: {"<term/phrase>": [<list of suggestions>]}.'
ans = 'Give me a short and unexplained answer in JSON format, where key is the input term/collocation, value is a list of sentences. Response format: {"<term/collocation>": ["<suggestion 1>", "<suggestion 2>"]}'
e = "Here is set of terms/collocation:"

en_expand_prompt = PromptTemplate(role=r, task=t, answer_format=ans, tail=e)

# r = "Вы – помощник патентного поверенного."
# t = (
#     "Вам дан набор терминов/фраз из патентной заявки, ваша задача - предложить {} синонимa или слова/фразы на английском языке, наиболее часто встречающиеся в технических/патентных документах совместно с каждым термином из набора и контекстно близких ему, постарайтесь предложить наиболее разнообразные варианты. "
#     "Каждая предложенная фраза не должна быть длиннее 3х слов. "
#     "Учитывайте все слова, чтобы понять общую тему набора слов. "
#     # "Предлагайте термины/фразы на английском языке."
# )
# ans = 'Дайте мне ответ кратко и без пояснений в формате JSON, где ключ - входной термин/фраза, значение - список предложений. Формат ответа: {"<термин/фраза>": ["<предложение 1>", "<предложение 2>", "<предложение 3>"]}.'
# # "Пример входного списка терминов/фраз: code, record. Пример ответа: {"code": ["program", "instructions", "algorithm"], "record": ["log", "note", "file"]}"
# e = "Вот список терминов/фраз:"


r = "Вы – патентный поверенный."
t = (
    "Вам дан набор терминов/словосочетаний из патентной заявки, ваша задача - предложить {} слова или словосочетания на английском языке наиболее часто встречающиеся в технических/патентных документах совместно с каждым термином из набора и контекстно близких ему, постарайтесь предложить наиболее разнообразные варианты. "
    "Каждое предложенное словосочетание не длиннее двух слов. "
    "Учитывайте все слова, чтобы понять общую тему набора слов. "
    # "Предлагайте слова или фразы только на английском языке."
)
ans = 'Дайте мне ответ кратко и без пояснений в формате JSON, где ключ - входной термин/словосочетание, значение - список предложений. Формат ответа: {"<термин/словосочетание>": ["<предложение 1>", "<предложение 2>"]}.'
# "Пример входного списка терминов/фраз: code, record. Пример ответа: {"code": ["program", "instructions", "algorithm"], "record": ["log", "note", "file"]}"
e = "Вот список терминов/словосочетаний:"

ru_expand_prompt = PromptTemplate(role=r, task=t, answer_format=ans, tail=e)
