import os

from duckpy import Client
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.agents import Tool
from langchain.base_language import BaseLanguageModel


MAX_SEARCH_RESULTS = 20  # Number of search results to observe at a time

search_description = """Useful for when you need to ask with search."""

notepad_description = """ Useful for when you need to note-down specific information for later reference."""


async def ddg(query: str):
    if query is None or query.lower().strip().strip('"') == "none" or query.lower().strip().strip('"') == "null":
        x = "The action_input field is empty. Please provide a search query."
        return [x]
    else:
        client = Client()
        return client.search(query)[:MAX_SEARCH_RESULTS]


async def notepad(x: str) -> str:
    return f"{[x]}"


search_tool = Tool(name="Tool_Search",
                   func=lambda x: x,
                   coroutine=ddg,
                   description=search_description)

note_tool = Tool(name="Tool_Notepad",
                   func=lambda x: x,
                   coroutine=notepad,
                   description=notepad_description)

async def final(x: str):
    pass

finish_description = """Useful when you have enough information to produce a
final answer that achieves the original Goal."""

finish_tool = Tool(name="Tool_Finish",
                   func=lambda x: x,
                   coroutine=final,
                   description=finish_description)

def rewrite_search_query(q: str, search_history, llm: BaseLanguageModel) -> str:
    history_string = '\n'.join(search_history)
    template ="""We are using the Search tool.
                 # Previous queries:
                 {history_string}. \n\n Rewrite query {action_input} to be
                 different from the previous queries."""
    prompt = PromptTemplate(template=template,
                            input_variables=["action_input", "history_string"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    result = llm_chain.predict(action_input=q, history_string=history_string)
    return result
