import wikipedia
import requests
from elasticsearch import Elasticsearch

from duckpy import Client
from langchain import PromptTemplate, LLMChain, Wikipedia
from langchain.agents import Tool
from langchain.agents.react.base import DocstoreExplorer

from langchain.base_language import BaseLanguageModel


MAX_SEARCH_RESULTS = 5  # Number of search results to observe at a time

INDEX_NAME = "wiki-dump-2017"

# Create the client instance
es_client = None

search_description = """ Useful for when you need to ask with search. Use direct language and be
EXPLICIT in what you want to search. Do NOT use filler words.

## Examples of incorrect use
{
     "action": "Tool_Search",
     "action_input": "[name of bagel shop] menu"
}

The action_input cannot be None or empty.
"""

notepad_description = """ Useful for when you need to note-down specific
information for later reference. Please provide the website and full
information you want to note-down in the action_input and all future prompts
will remember it. This is the mandatory tool after using the Tool_Search.
Using Tool_Notepad does not always lead to a final answer.

## Examples of using Notepad tool
{
    "action": "Tool_Notepad",
    "action_input": "(www.website.com) the information you want to note-down"
}
"""

wiki_notepad_description = """ Useful for when you need to note-down specific
information for later reference. Please provide the website and full
information you want to note-down in the action_input and all future prompts
will remember it. This is the mandatory tool after using the Tool_Wikipedia.
Using Tool_Notepad does not always lead to a final answer.

## Examples of using Notepad tool
{
    "action": "Tool_Notepad",
    "action_input": "(www.website.com) the information you want to note-down"
}
"""

wiki_search_description = """ Useful for when you need to get some information about a certain entity. Use direct language and be
concise about what you want to retrieve. Note: the action input MUST be a wikipedia entity instead of a long sentence.
                        
## Examples of correct use
1.  Action: Tool_Wikipedia
    Action Input: Colorado orogeny

The Action Input cannot be None or empty.
"""

wiki_lookup_description = """ This tool is helpful when you want to retrieve sentences containing a specific text snippet after checking a Wikipedia entity. 
It should be utilized when a successful Wikipedia search does not provide sufficient information. 
Keep your lookup concise, using no more than three words.

## Examples of correct use
1.  Action: Tool_Lookup
    Action Input: eastern sector

The Action Input cannot be None or empty.
"""


async def ddg(query: str):
    if query is None or query.lower().strip().strip('"') == "none" or query.lower().strip().strip('"') == "null":
        x = "The action_input field is empty. Please provide a search query."
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
        index=INDEX_NAME, query={"match": {"text": x}}, size=MAX_SEARCH_RESULTS
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
    res = []
    for obj in vector_search(x):
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

def vector_search(
        query: str,
        url: str = "http://0.0.0.0:8080/query",
        max_candidates: int = MAX_SEARCH_RESULTS
    ):
    response = requests.post(
        url=url, json={"query_list": [query]}
    ).json()["result"][0]["top_answers"]
    return response[:min(max_candidates, len(response))]


search_tool = Tool(name="Tool_Search",
                   func=lambda x: x,
                   coroutine=ddg,
                   description=search_description)

note_tool = Tool(name="Tool_Notepad",
                   func=lambda x: x,
                   coroutine=notepad,
                   description=notepad_description)

wiki_note_tool = Tool(name="Tool_Notepad",
                   func=lambda x: x,
                   coroutine=notepad,
                   description=wiki_notepad_description)

wiki_search_tool = Tool(
        name="Tool_Wikipedia",
        func=lambda x: x,
        coroutine=wikisearch,
        description=wiki_search_description
    )

wiki_lookup_tool = Tool(
        name="Tool_Lookup",
        func=lambda x: x,
        coroutine=wikilookup,
        description=wiki_lookup_description
    )

wiki_dump_search_tool = Tool(
        name="Tool_Wikipedia",
        func=lambda x: x,
        coroutine=wikidumpsearch_embed,
        description=wiki_search_description
    )

async def final(x: str):
    pass

finish_description = """ Useful when you have enough information to produce a
final answer that achieves the original Goal.

You must also include this key in the output for the Tool_Finish action
"citations": ["www.example.com/a/list/of/websites: what facts you got from the website",
"www.example.com/used/to/produce/the/action/and/action/input: "what facts you got from the website",
"www.webiste.com/include/the/citations/from/the/previous/steps/as/well: "what facts you got from the website",
"www.website.com": "this section is only needed for the final answer"]

## Examples of using Finish tool
{
    "action": "Tool_Finish",
    "action_input": "final answer",
    "citations": ["www.example.com: what facts you got from the website"]
}
"""

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


### Prompt V3 tools

search_description_v3 = """Useful for when you need to ask with search."""
notepad_description_v3 = """ Useful for when you need to note-down specific information for later reference."""
finish_description_v3 = """Useful when you have enough information to produce a final answer that achieves the original Goal."""

search_tool_v3 = Tool(name="Tool_Search",
                      func=lambda x: x,
                      coroutine=ddg,
                      description=search_description_v3)

note_tool_v3 = Tool(name="Tool_Notepad",
                    func=lambda x: x,
                    coroutine=notepad,
                    description=notepad_description_v3)

finish_tool_v3 = Tool(name="Tool_Finish",
                      func=lambda x: x,
                      coroutine=final,
                      description=finish_description_v3)
