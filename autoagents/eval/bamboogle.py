import glob
import json
import os
import pprint
import shutil

from autoagents.data.dataset import BAMBOOGLE
from autoagents.eval.metrics import get_common_stats
from autoagents.eval.hotpotqa.eval_async import check_answer_equivalency
from autoagents.agents.utils.constants import LOG_SAVE_DIR
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI


async def eval(eval_results_path: str=LOG_SAVE_DIR):
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
    accuracy = 0
    correct_res_dir, wrong_res_dir, err_res_dir = f"{eval_results_path}-eval/correct", f"{eval_results_path}-eval/wrong", f"{eval_results_path}-eval/error"
    os.makedirs(correct_res_dir, exist_ok=True)
    os.makedirs(wrong_res_dir, exist_ok=True)
    os.makedirs(err_res_dir, exist_ok=True)
    for file in tqdm(files):
        finish = False
        with open(file, "r") as f:
            log_data = json.load(f)
            has_error = any([True if "error" in entry else False for entry in log_data])
            for entry in log_data:
                if not has_error:
                    if "goal" in entry:
                        question = entry["goal"]
                    if "conversations" in entry:
                        output = json.loads(entry["conversations"][-1]["value"])
                        if output["action"] == "Tool_Finish":
                            finish = True
                            action_input = output["action_input"]
                            for i in range(len(BAMBOOGLE["questions"])):
                                if question == BAMBOOGLE["questions"][i]:
                                    answer = BAMBOOGLE["answers"][i]
                                    resp_obj = await check_answer_equivalency(question, answer, action_input, evalllm)
                                    is_correct = int(resp_obj.get("is_inferable", 0))
                                    if is_correct:
                                        shutil.copy2(file, correct_res_dir)
                                    else:
                                        shutil.copy2(file, wrong_res_dir)
                                    accuracy += is_correct
                else:
                    shutil.copy2(file, err_res_dir)
            if not finish:
                shutil.copy2(file, wrong_res_dir)
    counts = common_stats["counts"]
    total_samples = counts["total_samples"]
    finished_samples = counts["finished_samples"]
    print(f'accuracy overall is {accuracy}/{total_samples}={accuracy/total_samples}')
    print(f'accuracy on finished samples is {accuracy}/{finished_samples}={accuracy/finished_samples}')
    counts["accuracy on finished samples"] = accuracy/finished_samples
    counts["accuracy"] = accuracy/total_samples
    counts["average_answer_missing"] = (total_samples - finished_samples) / total_samples
    pprint.pprint(common_stats)
    with open(f"{eval_results_path}-eval/stats.json", "w") as f:
        json.dump(common_stats, f)
