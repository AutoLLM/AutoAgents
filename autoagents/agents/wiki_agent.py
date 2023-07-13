from langchain.base_language import BaseLanguageModel
from autoagents.agents.search import ActionRunner
from autoagents.tools.tools import wiki_dump_search_tool, note_tool


# Set up the base template
template = """
We are working together to satisfy the user's original goal step-by-step. Play to your strengths as an LLM.
Make sure the plan is achievable using the
available tools. You SHOULD directly produce a `Final Answer:` when you
think you have good-enough information to achieve the Goal. The final answer should be concise and quote the original document.
Today is {today}.

## Goal:
{input}

If you require assistance or additional information, you should use *only* one of the following tools:
{tools}.

## Output format
You MUST produce Output in the following format:

Thought: you should always think about what to do when you think you have not achieved the Goal.
Reasoning: reasoning
Plan:
- short bulleted
- list that conveys
- next-step plan
Action: the action to take, should be ONE OF {tool_names}
Action Input: the input to the Action
Observation: the result of the Action
... (this Thought/Reasoning/Plan/Action/Action Input/Observation can repeat N
times until there is a Final Answer)
Final Answer: the final answer to achieve the original Goal which can be the
only output or when you have no Action to do next.

## History
{agent_scratchpad}

Do not repeat any past actions in History, because you will not get additional information.
If the last action is search, then you should use notepad to keep critical information.
If you have gathered all information in your plannings to satisfy the user's original goal, then respond immediately as the Final Answer.
"""


class WikiActionRunner(ActionRunner):

    def __init__(self, outputq, llm: BaseLanguageModel, persist_logs: bool = False):

        super().__init__(
            outputq, llm, persist_logs,
            prompt_template=template,
            tools=[wiki_dump_search_tool, note_tool],
            search_tool_name="Wikipedia"
        )
