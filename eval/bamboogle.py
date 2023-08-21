import glob
import json
import os
import asyncio
import sys
import shutil

from dataset import BAMBOOGLE
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from eval.metrics import get_common_stats


async def eval():
    eval_results_path = sys.argv[1]
    files = glob.glob(f"{eval_results_path}/*.json")
    evalllm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_organization=os.getenv("OPENAI_API_ORG"),
        temperature=0,
        model="gpt-3.5-turbo",
        request_timeout=120
    )
    print(f"Found {len(files)} log files! Starts to analyze......")
    common_stats = get_common_stats(files)
    print(common_stats)
    accuracy, success_total = 0, 0
    correct_res_dir, wrong_res_dir, err_res_dir = f"{eval_results_path}-eval/correct", f"{eval_results_path}-eval/wrong", f"{eval_results_path}-eval/error"
    os.makedirs(correct_res_dir, exist_ok=True)
    os.makedirs(wrong_res_dir, exist_ok=True)
    os.makedirs(err_res_dir, exist_ok=True)
    for file in tqdm(files):
        finish = False
        with open(file, "r") as f:
            log_data = json.load(f)
            question = log_data[0]["goal"]
            for entry in log_data:
                if "error" in entry:
                    shutil.copy2(file, err_res_dir)
                elif "conversations" in entry:
                    output = json.loads(entry["conversations"][-1]["value"])
                    if output["action"] == "Tool_Finish":
                        finish = True
                        action_input = output["action_input"]
                        success_total += 1
                        for i in range(len(BAMBOOGLE["questions"])):
                            if question == BAMBOOGLE["questions"][i]:
                                answer = BAMBOOGLE["answers"][i]
                                resp = await evalllm.agenerate([[HumanMessage(
                                    content=f"Given a question and a pair of answers. Determine if Answer1 can be strictly infered from Answer2. Return False if given the information in Answer2, we cannot determine whether Answer1 is right. Add detailed explaination and reasioning. Format your answer in JSON with a boolean field called 'is_inferable' and a string field 'reasoning' that can be loaded in python.\n\nQuestion: '{question}'\n\nAnswer1: '{answer}'\n\nAnswer2: '{action_input}'"
                                )]])
                                resp_obj = json.loads(resp.generations[0][0].text.strip())
                                is_correct = int(resp_obj.get("is_inferable", 0))
                                if is_correct:
                                    shutil.copy2(file, correct_res_dir)
                                else:
                                    shutil.copy2(file, wrong_res_dir)
                                accuracy += is_correct
            if not finish:
                common_stats["average_answer_missing"] += 1
                shutil.copy2(file, wrong_res_dir)
    print(f'accuracy overall is {accuracy}/{common_stats["total_samples"]}={accuracy/common_stats["total_samples"]}')
    print(f"accuracy on successful runs is {accuracy}/{success_total}={accuracy/success_total}")
    common_stats["accuracy on successful runs"] = accuracy/success_total
    common_stats["accuracy"] = accuracy/common_stats["total_samples"]
    common_stats["average_answer_missing"] = common_stats["average_answer_missing"]/common_stats["total_samples"]
    with open(f"{eval_results_path}-eval/stats.json", "w") as f:
        json.dump(common_stats, f)

asyncio.run(eval())

