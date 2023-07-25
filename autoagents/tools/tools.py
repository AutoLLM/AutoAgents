import wikipedia
import requests
from elasticsearch import Elasticsearch

from duckpy import Client
from langchain import PromptTemplate, OpenAI, LLMChain, Wikipedia
from langchain.agents import Tool
from langchain.agents.react.base import DocstoreExplorer

from langchain.base_language import BaseLanguageModel


MAX_SEARCH_RESULTS = 5  # Number of search results to observe at a time

INDEX_NAME = "wiki-dump-2017"

# Create the client instance
es_client = None

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

wiki_notepad_description = """ Useful for when you need to note-down specific
information for later reference. Please provide full information you want to
note-down in the Action Input and all future prompts will remember it.
This is the mandatory tool after using the wikipedia tool.
Using Notepad does not always lead to a final answer.

## Exampels of using notepad tool
Action: Notepad
Action Input: the information you want to note-down
"""

wiki_search_description = """ Useful for when you need to get some information about a certain entity. Use direct language and be
concise about what you want to retrieve. Note: the action input MUST be a wikipedia entity instead of a long sentence.
                        
## Examples of correct use
1.  Action: Wikipedia
    Action Input: Colorado orogeny

The Action Input cannot be None or empty.
"""

wiki_lookup_description = """ This tool is helpful when you want to retrieve sentences containing a specific text snippet after checking a Wikipedia entity. 
It should be utilized when a successful Wikipedia search does not provide sufficient information. 
Keep your lookup concise, using no more than three words.

## Examples of correct use
1.  Action: Lookup
    Action Input: eastern sector

The Action Input cannot be None or empty.
"""

async def ddg(query: str):
    if query is None or query.lower().strip().strip('"') == "none" or query.lower().strip().strip('"') == "null":
        x = "The action input field is empty. Please provide a search query."
        return [x]
    else:
        client = Client()
        return client.search(query)[:MAX_SEARCH_RESULTS]

docstore=DocstoreExplorer(Wikipedia())

async def notepad(x: str) -> str:
    return f"{[x]}"

async def wikisearch(x: str) -> str:
    title_list = wikipedia.search(x)
    if not title_list:
        return docstore.search(x)
    title = title_list[0]
    return f"Wikipedia Page Title: {title}\nWikipedia Page Content: {docstore.search(title)}"

async def wikilookup(x: str) -> str:
    return docstore.lookup(x)

async def wikidumpsearch_es(x: str) -> str:
    global es_client
    if es_client is None:
        es_client = Elasticsearch("http://localhost:9200")
    resp = es_client.search(
        index=INDEX_NAME, query={"match": {"text": x}}, size=5
    )
    res = []
    for hit in resp['hits']['hits']:
        doc = hit["_source"]
        res.append({
            "title": doc["title"],
            "text": ''.join(sent for sent in doc["text"][1]),
            "url": doc["url"]
        })
        if doc["title"] == x:
            return [{
                "title": doc["title"],
                "text": '\n'.join(''.join(paras) for paras in doc["text"][1:3])
                    if len(doc["text"]) > 2
                    else '\n'.join(''.join(paras) for paras in doc["text"]),
                "url": doc["url"]
            }]
    return res
        
async def wikidumpsearch_embed(x: str) -> str:
    response = requests.post(
        url="http://0.0.0.0:8080/query",
        json={"query_list": [x]}
    ).json()["result"][0]["top_answers"]
    res = []
    for obj in response[:min(5, len(response))]:
        paras = obj["text"].split('\n')
        cur = {
            "title": obj["sources"][0]["title"],
            "text": paras[min(1, len(paras) - 1)],
            "url": obj["sources"][0]["url"]
        }
        res.append(cur)
        if cur["title"] == x:
            return [{
                "title": obj["sources"][0]["title"],
                "text": '\n'.join(paras[1:3] if len(paras) > 2 else paras),
                "url": obj["sources"][0]["url"]
            }]
    return res


search_tool = Tool(name="Search",
                   func=lambda x: x,
                   coroutine=ddg,
                   description=search_description)

note_tool = Tool(name="Notepad",
                   func=lambda x: x,
                   coroutine=notepad,
                   description=notepad_description)

wiki_note_tool = Tool(name="Notepad",
                   func=lambda x: x,
                   coroutine=notepad,
                   description=wiki_notepad_description)

wiki_search_tool = Tool(
        name="Wikipedia",
        func=lambda x: x,
        coroutine=wikisearch,
        description=wiki_search_description
    )

wiki_lookup_tool = Tool(
        name="Lookup",
        func=lambda x: x,
        coroutine=wikilookup,
        description=wiki_lookup_description
    )

wiki_dump_search_tool = Tool(
        name="Wikipedia",
        func=lambda x: x,
        coroutine=wikidumpsearch_embed,
        description=wiki_search_description
    )

def rewrite_search_query(q: str, search_history, llm: BaseLanguageModel) -> str:
    history_string = '\n'.join(search_history)
    template ="""We are using the Search tool.
                 # Previous queries:
                 {history_string}. \n\n Rewrite query {action_input} to be
                 different from the previous ones."""
    prompt = PromptTemplate(template=template,
                            input_variables=["action_input", "history_string"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain.predict(action_input=q, history_string=history_string)
