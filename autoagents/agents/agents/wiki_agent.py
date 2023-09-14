from langchain.base_language import BaseLanguageModel
from autoagents.agents.agents.search import ActionRunner
from autoagents.agents.agents.search_v3 import ActionRunnerV3
from autoagents.agents.tools.tools import wiki_dump_search_tool, wiki_note_tool, finish_tool


# Set up the base template
template = """We are working together to satisfy the user's original goal
step-by-step. Play to your strengths as an LLM. Make sure the plan is
achievable using the available tools. The final answer should be descriptive,
and should include all relevant details.

Today is {today}.

## Goal:
{input}

If you require assistance or additional information, you should use *only* one
of the following tools: {tools}.

## History
{agent_scratchpad}

Do not repeat any past actions in History, because you will not get additional
information. If the last action is Tool_Wikipedia, then you should use Tool_Notepad to keep
critical information. If you have gathered all information in your plannings
to satisfy the user's original goal, then respond immediately with the Finish
Action.

## Output format
You MUST produce JSON output with below keys:
"thought": "current train of thought",
"reasoning": "reasoning",
"plan": [
"short bulleted",
"list that conveys",
"next-step plan",
],
"action": "the action to take",
"action_input": "the input to the Action",
"""


class WikiActionRunner(ActionRunner):

    def __init__(self, outputq, llm: BaseLanguageModel, persist_logs: bool = False):

        super().__init__(
            outputq, llm, persist_logs,
            prompt_template=template,
            tools=[wiki_dump_search_tool, wiki_note_tool, finish_tool]
        )

class WikiActionRunnerV3(ActionRunnerV3):
    def __init__(self, outputq, llm: BaseLanguageModel, persist_logs: bool = False):

        super().__init__(
            outputq, llm, persist_logs,
            tools=[wiki_dump_search_tool, wiki_note_tool, finish_tool]
        )
