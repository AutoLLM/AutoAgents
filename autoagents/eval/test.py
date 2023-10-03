import argparse
import asyncio
import json
import os
from tqdm.asyncio import tqdm_asyncio

from autoagents.agents.agents.search import ActionRunner
from autoagents.agents.agents.wiki_agent import WikiActionRunner, WikiActionRunnerV3
from autoagents.agents.agents.search_v3 import ActionRunnerV3
from autoagents.agents.models.custom import CustomLLM, CustomLLMV3
from autoagents.agents.utils.constants import LOG_SAVE_DIR
from autoagents.data.dataset import BAMBOOGLE, DEFAULT_Q, FT, HF
from autoagents.eval.bamboogle import eval as eval_bamboogle
from autoagents.eval.hotpotqa.eval_async import HotpotqaAsyncEval, NUM_SAMPLES_TOTAL
from langchain.chat_models import ChatOpenAI


OPENAI_MODEL_NAMES = {"gpt-3.5-turbo", "gpt-4"}
AWAIT_TIMEOUT: int = 120
MAX_RETRIES: int = 5


async def work(user_input: str, model: str, temperature: int, agent: str, prompt_version: str, persist_logs: bool, log_save_dir: str):
    if model not in OPENAI_MODEL_NAMES:
        if prompt_version == "v2":
            llm = CustomLLM(
                model_name=model,
                temperature=temperature,
                request_timeout=AWAIT_TIMEOUT
            )
        elif prompt_version == "v3":
            llm = CustomLLMV3(
                model_name=model,
                temperature=temperature,
                request_timeout=AWAIT_TIMEOUT
            )
    else:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_organization=os.getenv("OPENAI_API_ORG"),
            temperature=temperature,
            model_name=model,
            request_timeout=AWAIT_TIMEOUT
        )

    retry_count = 0
    while retry_count < MAX_RETRIES:
        outputq = asyncio.Queue()
        if agent == "ddg":
            if prompt_version == "v2":
                runner = ActionRunner(outputq, llm=llm, persist_logs=persist_logs)
            elif prompt_version == "v3":
                runner = ActionRunnerV3(outputq, llm=llm, persist_logs=persist_logs)
        elif agent == "wiki":
            if prompt_version == "v2":
                runner = WikiActionRunner(outputq, llm=llm, persist_logs=persist_logs)
            elif prompt_version == "v3":
                runner = WikiActionRunnerV3(outputq, llm=llm, persist_logs=persist_logs)
        task = asyncio.create_task(runner.run(user_input, outputq, log_save_dir))
        while True:
            try:
                output = await asyncio.wait_for(outputq.get(), AWAIT_TIMEOUT)
            except asyncio.TimeoutError:
                task.cancel()
                retry_count += 1
                break
            if isinstance(output, RuntimeWarning):
                print(f"Question: {user_input}")
                print(output)
                continue
            elif isinstance(output, Exception):
                task.cancel()
                print(f"Question: {user_input}")
                print(output)
                retry_count += 1
                break
            try:
                parsed = json.loads(output)
                print(json.dumps(parsed, indent=2))
                print("-----------------------------------------------------------")
                if parsed["action"] == "Tool_Finish":
                    return await task
            except:
                print(f"Question: {user_input}")
                print(output)
                print("-----------------------------------------------------------")
    

async def main(questions, args):
    sem = asyncio.Semaphore(10)
    
    async def safe_work(user_input: str, model: str, temperature: int, agent: str, prompt_version: str, persist_logs: bool, log_save_dir: str):
        async with sem:
            return await work(user_input, model, temperature, agent, prompt_version, persist_logs, log_save_dir)
    
    persist_logs = True if args.persist_logs else False
    await tqdm_asyncio.gather(*[safe_work(q, args.model, args.temperature, args.agent, args.prompt_version, persist_logs, args.log_save_dir) for q in questions])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="model to be tested")
    parser.add_argument("--temperature", type=float, default=0, help="model temperature")
    parser.add_argument("--agent",
        default="ddg",
        const="ddg",
        nargs="?",
        choices=("ddg", "wiki"),
        help='which action agent we want to interact with(default: ddg)'
    )
    parser.add_argument("--persist-logs", action="store_true", help="persist logs on disk, enable this feature for later eval purpose")
    parser.add_argument("--log-save-dir", type=str, default=LOG_SAVE_DIR, help="dir to save logs")
    parser.add_argument("--dataset",
        default="default",
        const="default",
        nargs="?",
        choices=("default", "hotpotqa", "ft", "hf", "bamboogle"),
        help='which dataset we want to interact with(default: default)'
    )
    parser.add_argument("--eval", action="store_true", help="enable automatic eval")
    parser.add_argument("--prompt-version",
        default="v2",
        const="v3",
        nargs="?",
        choices=("v2", "v3"),
        help='which version of prompt to use(default: v2)'
    )
    parser.add_argument("--slice", type=int, help="slice the dataset from left, question list will start from index 0 to slice - 1")
    args = parser.parse_args()
    print(args)
    if args.prompt_version == "v3" and args.model in OPENAI_MODEL_NAMES:
        raise ValueError("Prompt v3 is not compatiable with OPENAI models, please adjust your settings!")
    if not args.persist_logs and args.eval:
        raise ValueError("Please enable persist_logs feature to allow eval code to run!")
    if not args.log_save_dir and args.persist_logs:
        raise ValueError("Please endbale persist_logs feature to configure log dir location!")
    questions = []
    if args.dataset == "ft":
        questions = [q for _, q in FT]
    elif args.dataset == "hf":
        questions = [q for _, q in HF]
    elif args.dataset == "hotpotqa":
        hotpotqa_eval = HotpotqaAsyncEval(model=args.model)
        questions = hotpotqa_eval.get_questions(args.slice or NUM_SAMPLES_TOTAL)
    elif args.dataset == "bamboogle":
        questions = BAMBOOGLE["questions"]
    else:
        questions = [q for _, q in DEFAULT_Q]
    if args.slice and args.dataset != "hotpotqa":
        questions = questions[:args.slice]
    asyncio.run(main(questions, args))
    if args.eval:
        if args.dataset == "bamboogle":
            if args.log_save_dir:
                asyncio.run(eval_bamboogle(args.log_save_dir))
            else:
                asyncio.run(eval_bamboogle())
        elif args.dataset == "hotpotqa":
            if args.log_save_dir:
                hotpotqa_eval.run(args.log_save_dir)
            else:
                hotpotqa_eval.run()
