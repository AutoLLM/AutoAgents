# Script generates action data from goals calling GPT-4
import os
import asyncio

from multiprocessing import Pool

from autoagents.agents.search import ActionRunner
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
    with open("goals_train.json", "r") as file:
        data = json.load(file)
    with Pool(processes=4) as pool:
        pool.map(main, data)
