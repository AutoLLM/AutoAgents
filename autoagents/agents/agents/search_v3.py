from datetime import date
import json
import uuid
from collections import defaultdict
from typing import List, Union, Any, Optional, Dict

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.base_language import BaseLanguageModel

from autoagents.agents.tools.tools import search_tool_v3, note_tool_v3, finish_tool_v3
from autoagents.agents.utils.logger import InteractionsLogger
from autoagents.agents.utils.constants import LOG_SAVE_DIR
from autoagents.agents.agents.search import check_valid


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
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
        goal = kwargs["input"]
        goal = f"Today is {date.today()}. {goal}"
        list_prompt =[]
        list_prompt.append({"role": "goal", "content": goal})
        list_prompt.append({"role": "tools", "content": [{tool.name: tool.description} for tool in self.tools]})
        list_prompt.append({"role": "history", "content": history})
        return json.dumps(list_prompt)


class CustomOutputParser(AgentOutputParser):
    class Config:
        arbitrary_types_allowed = True
    ialogger: InteractionsLogger
    llm: BaseLanguageModel
    new_action_input: Optional[str]
    action_history = defaultdict(set)

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        try:
            parsed = json.loads(llm_output)
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        if not check_valid(parsed):
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        self.ialogger.add_ai(llm_output)
        # Parse out the action and action input
        action = parsed["action"]
        action_input = parsed["action_input"]

        if action == "Tool_Finish":
            return AgentFinish(return_values={"output": action_input}, log=llm_output)
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


class ActionRunnerV3:
    def __init__(self,
                 outputq,
                 llm: BaseLanguageModel,
                 persist_logs: bool = False,
                 tools = [search_tool_v3, note_tool_v3, finish_tool_v3]):
        self.ialogger = InteractionsLogger(name=f"{uuid.uuid4().hex[:6]}", persist=persist_logs)
        self.ialogger.set_tools([{tool.name: tool.description} for tool in tools])
        prompt = CustomPromptTemplate(tools=tools,
                                      input_variables=["input", "intermediate_steps"],
                                      ialogger=self.ialogger)
        output_parser = CustomOutputParser(ialogger=self.ialogger, llm=llm)

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
                pass

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
        goal = f"Today is {date.today()}. {goal}"
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
