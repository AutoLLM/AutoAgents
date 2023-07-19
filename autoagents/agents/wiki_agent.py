from langchain.base_language import BaseLanguageModel
from autoagents.agents.search import ActionRunner
from autoagents.tools.tools import wiki_dump_search_tool, note_tool, finish_tool


class WikiActionRunner(ActionRunner):

    def __init__(self, outputq, llm: BaseLanguageModel, persist_logs: bool = False):

        super().__init__(
            outputq, llm, persist_logs,
            tools=[wiki_dump_search_tool, note_tool, finish_tool],
            search_tool_name="Tool_Wikipedia"
        )
