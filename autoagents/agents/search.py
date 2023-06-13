from typing import List, Union, Any, Optional, Dict
import uuid
import re
from datetime import date
import asyncio
from collections import defaultdict
import os

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager


from autoagents.tools.tools import search_tool, note_tool, rewrite_search_query
from autoagents.utils.logger import InteractionsLogger
from autoagents.utils.utils import OpenAICred


# Set up the base template
template = """
We are working together to satisfy the user's original goal step-by-step. Play to your strengths as an LLM.
Make sure the plan is achievable using the
available tools. You SHOULD directly produce a `Final Answer:` when you
think you have good-enough information to achieve the Goal. The final answer should be descriptive should be descriptive, encompassing all relevant details..
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


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    ialogger: InteractionsLogger

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        outputs = ""
        # Set the agent_scratchpad variable to that value
        for action, observation in intermediate_steps[:-1]:
            outputs += f"{action.log}\n"
        if len(intermediate_steps) > 0:
            action, observation = intermediate_steps[-1]
            # self.ialogger.add_system({"action": action, "observation": observation})
            if action.tool not in ("Search", "Notepad"):
                raise Exception("Invalid tool requested by the model.")
            if action.tool == "Notepad":
                outputs += f"{action.log}\n"
                outputs += f"Observation: {observation}\n"
            elif action.tool == "Search":
                current = "".join([f"{d}" for d in observation])
                outputs += f"{action.log}\n"
                outputs += f"Observation: {current}\n"

            # Parse the output ofr the last step for the reasoning and plan
            regex = r"Thought\s*\d*\s*:(.*?)\n(.*)"
            match = re.search(regex, action.log, re.DOTALL)
            thoughts = match.group(1).strip() if match else ""

            regex = r"Reasoning\s*\d*\s*:(.*?)\n(.*)"
            match = re.search(regex, action.log, re.DOTALL)
            reasoning = match.group(1).strip() if match else ""

            regex = r"Plan\s*\d*\s*:(.*?)\nAction(.*)"
            match = re.search(regex, action.log, re.DOTALL)
            plans = match.group(1).strip() if match else ""
            self.ialogger.add_structured_data({"output":{"thoughts": thoughts,
                                                         "reasoning": reasoning,
                                                         "plans": plans,
                                                         "action": action.tool,
                                                         "action_input": action.tool_input,
                                                         "raw_output":action.log},
                                                         "observation": observation})
        kwargs["agent_scratchpad"] = outputs
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["today"] = date.today()
        final_prompt = self.template.format(**kwargs)
        self.ialogger.add_system({"value": final_prompt})
        return final_prompt


class CustomOutputParser(AgentOutputParser):
    class Config:
        arbitrary_types_allowed = True
    ialogger: InteractionsLogger
    cred: OpenAICred
    new_action_input: Optional[str]

    action_history = defaultdict(set)

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        self.ialogger.add_ai(llm_output)
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            self.ialogger.add_structured_data({"output": {"action": "Final Answer",
                                                          "action_input": final_answer,
                                                          "raw_output": llm_output}})
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": final_answer},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2).strip().strip('"')

        if action_input in self.action_history[action]:
            new_action_input = rewrite_search_query(action_input,
                                                    self.action_history[action],
                                                    cred)
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
                 cred: OpenAICred,
                 model_name: str,
                 persist_logs: bool = False):
        self.ialogger = InteractionsLogger(name=f"{uuid.uuid4().hex[:6]}", persist=persist_logs)
        tools = [search_tool, note_tool]
        prompt = CustomPromptTemplate(
                template=template,
                tools=tools,
                input_variables=["input", "intermediate_steps"],
                ialogger=self.ialogger)

        output_parser = CustomOutputParser(ialogger=self.ialogger, cred=cred)

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
                    # Notify users
                    await outputq.put(RuntimeWarning(f"Action Input Rewritten: {new_action_input}"))
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

        llm = ChatOpenAI(openai_api_key=cred.key,
                         openai_organization=cred.org,
                         temperature=0,
                         model_name=model_name)
        llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
        tool_names = [tool.name for tool in tools]
        for tool in tools:
            tool.callbacks = [handler]

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        callback_manager = AsyncCallbackManager([handler])

        # Finally create the Executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                                 tools=tools,
                                                                 verbose=False,
                                                                 callback_manager=callback_manager)

    async def run(self, goal: str, outputq):
        self.ialogger.set_goal(goal)
        try:
            with get_openai_callback() as cb:
                output = await self.agent_executor.arun(goal)
                self.ialogger.add_cost({"total_tokens": cb.total_tokens,
                                        "prompt_tokens": cb.prompt_tokens,
                                        "completion_tokens": cb.completion_tokens,
                                        "total_cost": cb.total_cost,
                                        "successful_requests": cb.successful_requests})
            self.ialogger.save()
        except Exception as e:
            self.ialogger.add_message({"error": str(e)})
            self.ialogger.save()
            await outputq.put(e)
            return
        return output
