import os
import json
import uuid
import re
from datetime import date
import asyncio
from collections import defaultdict
from pprint import pprint
from typing import List, Union, Any, Optional, Dict

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.base_language import BaseLanguageModel

from autoagents.agents.tools.tools import search_tool, note_tool, rewrite_search_query, finish_tool
from autoagents.agents.utils.logger import InteractionsLogger
from autoagents.agents.utils.constants import LOG_SAVE_DIR

from pydantic import BaseModel, ValidationError, Extra  # pydantic==1.10.11


class InterOutputSchema(BaseModel):
    thought: str
    reasoning: str
    plan: List[str]
    action: str
    action_input: str
    class Config:
        extra = Extra.forbid


class FinalOutputSchema(BaseModel):
    thought: str
    reasoning: str
    plan: List[str]
    action: str
    action_input: str
    citations: List[str]
    class Config:
        extra = Extra.forbid


def check_valid(o):
    try:
        if o.get("action") == "Tool_Finish":
            FinalOutputSchema(**o)
        else:
            InterOutputSchema(**o)
    except ValidationError:
        return False
    return True


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
information. If the last action is Tool_Search, then you should use Tool_Notepad to keep
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


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    ialogger: InteractionsLogger

    def format(self, **kwargs) -> str:
        # Get the intermediate steps [(AgentAction, Observation)]
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        history = []
        # Set the agent_scratchpad variable to that value
        for i, (action, observation) in enumerate(intermediate_steps):
            if action.tool not in [tool.name for tool in self.tools]:
                raise Exception("Invalid tool requested by the model.")
            parsed = json.loads(action.log)
            if i == len(intermediate_steps) - 1:
                # Add observation only for the last action
                parsed["observation"] = observation
            history.append(parsed)
        self.ialogger.add_history(history)
        kwargs["agent_scratchpad"] = json.dumps(history)
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["today"] = date.today()
        final_prompt = self.template.format(**kwargs)
        self.ialogger.add_system(final_prompt)
        return final_prompt


class CustomOutputParser(AgentOutputParser):
    class Config:
        arbitrary_types_allowed = True
    ialogger: InteractionsLogger
    llm: BaseLanguageModel
    new_action_input: Optional[str]
    action_history = defaultdict(set)

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        self.ialogger.add_ai(llm_output)
        parsed = json.loads(llm_output)
        if not check_valid(parsed):
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        # Parse out the action and action input
        action = parsed["action"]
        action_input = parsed["action_input"]

        if action == "Tool_Finish":
            return AgentFinish(return_values={"output": action_input}, log=llm_output)

        if action_input in self.action_history[action]:
            new_action_input = rewrite_search_query(action_input,
                                                    self.action_history[action],
                                                    self.llm)
            self.ialogger.add_message({"query_rewrite": True})
            self.new_action_input = new_action_input
            self.action_history[action].add(new_action_input)
            return AgentAction(tool=action, tool_input=new_action_input, log=llm_output)
        else:
            # Return the action and action input
            self.action_history[action].add(action_input)
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)


class ActionRunner:
    def __init__(self,
                 outputq,
                 llm: BaseLanguageModel,
                 persist_logs: bool = False,
                 prompt_template: str = template,
                 tools: List[Tool] = [search_tool, note_tool, finish_tool]):
        self.ialogger = InteractionsLogger(name=f"{uuid.uuid4().hex[:6]}", persist=persist_logs)
        prompt = CustomPromptTemplate(template=prompt_template,
                                      tools=tools,
                                      input_variables=["input", "intermediate_steps"],
                                      ialogger=self.ialogger)

        output_parser = CustomOutputParser(ialogger=self.ialogger, llm=llm)
        self.model_name = llm.model_name

        class MyCustomHandler(AsyncCallbackHandler):
            def __init__(self):
                pass

            async def on_chain_end(self, outputs, **kwargs) -> None:
                if "text" in outputs:
                    await outputq.put(outputs["text"])

            async def on_agent_action(
                    self,
                    action: AgentAction,
                    *,
                    run_id: uuid.UUID,
                    parent_run_id: Optional[uuid.UUID] = None,
                    **kwargs: Any,
                    ) -> None:
                if (new_action_input := output_parser.new_action_input):
                    await outputq.put(RuntimeWarning(f"Action Input Rewritten: {new_action_input}"))
                    # Notify users
                    output_parser.new_action_input = None

            async def on_tool_start(
                    self,
                    serialized: Dict[str, Any],
                    input_str: str,
                    *,
                    run_id: uuid.UUID,
                    parent_run_id: Optional[uuid.UUID] = None,
                    **kwargs: Any,
                    ) -> None:
                pass

            async def on_tool_end(
                    self,
                    output: str,
                    *,
                    run_id: uuid.UUID,
                    parent_run_id: Optional[uuid.UUID] = None,
                    **kwargs: Any,
                    ) -> None:
                await outputq.put(output)

        handler = MyCustomHandler()

        llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
        tool_names = [tool.name for tool in tools]
        for tool in tools:
            tool.callbacks = [handler]

        agent = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["0xdeadbeef"],  # required
                allowed_tools=tool_names
                )
        callback_manager = AsyncCallbackManager([handler])

        # Finally create the Executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                                 tools=tools,
                                                                 verbose=False,
                                                                 callback_manager=callback_manager)

    async def run(self, goal: str, outputq, save_dir=LOG_SAVE_DIR):
        self.ialogger.set_goal(goal)
        try:
            with get_openai_callback() as cb:
                output = await self.agent_executor.arun(goal)
                self.ialogger.add_cost({"total_tokens": cb.total_tokens,
                                        "prompt_tokens": cb.prompt_tokens,
                                        "completion_tokens": cb.completion_tokens,
                                        "total_cost": cb.total_cost,
                                        "successful_requests": cb.successful_requests})
            self.ialogger.save(save_dir)
        except Exception as e:
            self.ialogger.add_message({"error": str(e)})
            self.ialogger.save(save_dir)
            await outputq.put(e)
            return
        return output
