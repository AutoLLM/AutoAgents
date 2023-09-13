# Script generates action data from goals calling GPT-4
import os
import asyncio
import argparse

from multiprocessing import Pool

from autoagents.agents.agents.search import ActionRunner
from langchain.chat_models import ChatOpenAI
import json


async def work(user_input):
    outputq = asyncio.Queue()
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                     openai_organization=os.getenv("OPENAI_API_ORG"),
                     temperature=0.,
                     model_name="gpt-4")
    runner = ActionRunner(outputq, llm=llm, persist_logs=True)
    task = asyncio.create_task(runner.run(user_input, outputq))

    while True:
        output = await outputq.get()
        if isinstance(output, RuntimeWarning):
            print(output)
            continue
        elif isinstance(output, Exception):
            print(output)
            return
        try:
            parsed = json.loads(output)
            if parsed["action"] in ("Tool_Finish", "Tool_Abort"):
                break
        except:
            pass
    await task
    print(f"{user_input}")

def main(q):
    asyncio.run(work(q))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--goals', type=str, help="file containing JSON array of goals", required=True)
    args = parser.parse_args()
    with open(args.goals, "r") as file:
        data = json.load(file)
    with Pool(processes=4) as pool:
        pool.map(main, data)
