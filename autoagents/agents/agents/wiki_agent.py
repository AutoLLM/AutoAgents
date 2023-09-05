from langchain.base_language import BaseLanguageModel
from autoagents.agents.agents.search import ActionRunner
from autoagents.agents.tools.tools import wiki_dump_search_tool, wiki_note_tool, finish_tool


# Set up the base template
template = """We are working together to satisfy the user's original goal
step-by-step. Today is {today}.

## Goal
{input}

## Available Tools
If you require assistance or additional information, you should use *only* one
of the tools: {tools}

## Observation
{observation}

## Notepad
{notepad}

## Action History
{agent_scratchpad}

You MUST produce JSON output with below keys:
"thought": "current train of thought",
"reasoning": "reasoning",
"plan": [
"short bulleted",
"list that conveys",
"next-step plan",
],
"action": "the action to take",
"action_input": "the input to the Action"
"""


class WikiActionRunner(ActionRunner):

    def __init__(self, outputq, llm: BaseLanguageModel, persist_logs: bool = False):

        super().__init__(
            outputq, llm, persist_logs,
            prompt_template=template,
            tools=[wiki_dump_search_tool, wiki_note_tool, finish_tool]
        )
