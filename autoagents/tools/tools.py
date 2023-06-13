import os

from duckpy import Client
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI

from autoagents.utils.utils import OpenAICred


MAX_SEARCH_RESULTS = 20  # Number of search results to observe at a time

search_description = """ Useful for when you need to ask with search. Use direct language and be
EXPLICIT in what you want to search.

## Examples of incorrect use
1.  Action: Search
    Action Input: "[name of bagel shop] menu"

The Action Input cannot be None or empty.
"""

notepad_description = """ Useful for when you need to note-down specific
information for later reference. Please provide full information you want to
note-down in the Action Input and all future prompts will remember it.
This is the mandatory tool after using the search tool.
Using Notepad does not always lead to a final answer.

## Exampels of using notepad tool
Action: Notepad
Action Input: the information you want to note-down
"""

async def ddg(query: str):
    if query is None or query.lower().strip().strip('"') == "none" or query.lower().strip().strip('"') == "null":
        x = "The action input field is empty. Please provide a search query."
        return [x]
    else:
        client = Client()
        return client.search(query)[:MAX_SEARCH_RESULTS]


async def notepad(x: str) -> str:
    return f"{[x]}"


search_tool = Tool(name="Search",
                   func=lambda x: x,
                   coroutine=ddg,
                   description=search_description)

note_tool = Tool(name="Notepad",
                   func=lambda x: x,
                   coroutine=notepad,
                   description=notepad_description)


def rewrite_search_query(q: str, search_history, cred: OpenAICred) -> str:
    history_string = '\n'.join(search_history)
    template ="""We are using the Search tool.
                 # Previous queries:
                 {history_string}. \n\n Rewrite query {action_input} to be
                 different from the previous ones."""
    llm = ChatOpenAI(temperature=0,
                     openai_api_key=cred.key,
                     openai_organization=cred.org)
    prompt = PromptTemplate(template=template,
                            input_variables=["action_input", "history_string"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.predict(action_input=q, history_string=history_string)
