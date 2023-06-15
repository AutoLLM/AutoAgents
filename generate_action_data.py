import os
import asyncio
from autoagents.agents.search import ActionRunner
from pprint import pprint
import json
from ast import literal_eval
from multiprocessing import Pool, TimeoutError
import pdb


async def work(user_input):
    outputq = asyncio.Queue()

    API_O = os.getenv("OPENAI_API_KEY")
    runner = ActionRunner(outputq, api_key=API_O, model_name="gpt-4")
    task = asyncio.create_task(runner.run(user_input, outputq))

    while True:
        output = await outputq.get()
        if isinstance(output, Exception):
            print(output)
            return
        try:
            pprint(literal_eval(output))
        except:
            print(output)
        print("-----------------------------------------------------------")
        if "Final Answer:" in output:
            break
    await task

def main(q):
    asyncio.run(work(q))

if __name__ == "__main__":

    Q = []
    # load a json file, which is a list of dictionaries
    with open('./action_data/self-gen.json', 'r') as f:
        data = json.load(f)
    for item in data:
        Q.append(item['goal'])

    with Pool(processes=5) as pool:
        print(pool.map(main, Q))