import glob
import json
import os
import asyncio
import sys

from dataset import BAMBOOGLE
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


async def eval():
    files = glob.glob(f"{sys.argv[1]}/*.json")
    evalllm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_organization=os.getenv("OPENAI_API_ORG"),
        temperature=0,
        model="gpt-3.5-turbo",
        request_timeout=120
    )
    print(files)
    accuracy, total = 0, 0
    for file in files:
        with open(file, "r") as f:
            log_data = json.load(f)
            question = log_data[0]["goal"]
            try:
                prediction = json.loads(
                    log_data[-3]["conversations"][-1]["value"])["action_input"]
                for i in range(len(BAMBOOGLE["questions"])):
                    if question == BAMBOOGLE["questions"][i]:
                        answer = BAMBOOGLE["answers"][i]
                        print(question)
                        print(answer)
                        print(prediction)
                        resp = await evalllm.agenerate([[HumanMessage(
                            content=f"Given a question and a pair of answers. Determine if Answer1 can be strictly infered from Answer2. Return False if given the information in Answer2, we cannot determine whether Answer1 is right. Add detailed explaination and reasioning. Format your answer in JSON with a boolean field called 'is_inferable' and a string field 'reasoning' that can be loaded in python.\n\nQuestion: '{question}'\n\nAnswer1: '{answer}'\n\nAnswer2: '{prediction}'"
                        )]])
                        resp_obj = json.loads(resp.generations[0][0].text.strip())
                        is_correct = int(resp_obj.get("is_inferable", 0))
                        print(is_correct)
                        accuracy += is_correct
                        total += 1
            except:
                pass
    print(f"accuracy overall is {accuracy}/{total}={accuracy/total}")

asyncio.run(eval())

