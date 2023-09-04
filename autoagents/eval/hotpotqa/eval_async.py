import os
import json
import asyncio
import requests
from typing import Optional, Union
from tqdm.asyncio import tqdm_asyncio
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

from autoagents.eval.hotpotqa.constants import *
from autoagents.eval.hotpotqa.hotpotqa_eval import eval


class HotpotqaAsyncEval:

    def __init__(
        self,
        model: str,
        ckpt_dir: Optional[str] = None,
        pred_file: Optional[str] = None
    ):

        if ckpt_dir is None:
            ckpt_dir = os.path.join(PARENT_DIRECTORY, f"results_{model}")

        if not os.path.isdir(ckpt_dir):
            os.mkdir(ckpt_dir)

        self.pred_file = pred_file or os.path.join(ckpt_dir, "prediction.json")
        self.new_log_dir = os.path.join(ckpt_dir, "data")
        self.wrong_ans_file = os.path.join(ckpt_dir, "wrong_ans.json")

        if not os.path.isdir(LOG_DATA_DIR):
            os.mkdir(LOG_DATA_DIR)

    def get_questions(self, total: Optional[int] = None):
        dataset = prepare_dataset(total=total, pred_ckpt=self.pred_file)
        return [data["question"] for data in dataset]

    def run(self, log_dir: Optional[str] = None):

        if log_dir is None:
            if not os.path.isdir(self.new_log_dir):
                os.mkdir(self.new_log_dir)
            if os.path.isdir(LOG_DATA_DIR):
                for log_file in os.listdir(LOG_DATA_DIR):
                    os.rename(
                        src=os.path.join(LOG_DATA_DIR, log_file),
                        dst=os.path.join(self.new_log_dir, log_file)
                    )
                os.rmdir(LOG_DATA_DIR)
            log_dir = self.new_log_dir

        pred_dict = predict_log_dir(log_dir=log_dir, pred_ckpt=self.pred_file)

        self.save_output(pred_dict=pred_dict)

        eval(self.pred_file, GT_FILE)

    def save_output(self, pred_dict: dict):

        with open(self.pred_file, 'w') as f:
            json.dump(pred_dict, f, indent=2)

        wrong_ans = []
        for qid, stat in pred_dict["statistics"].items():
            if stat["equivalency"] == 0:
                wrong_ans.append({
                    "question": stat["question"],
                    "gt_answer": stat["gt_answer"],
                    "prediction": pred_dict["answer"].get(qid, ''),
                    "reasoning": stat["reasoning"]
                })
        with open(self.wrong_ans_file, 'w') as f:
            json.dump(wrong_ans, f, indent=2)


def get_pred_dict(pred_ckpt: Optional[str] = None):

    if pred_ckpt is not None and os.path.isfile(pred_ckpt):
        with open(pred_ckpt, 'r') as f:
            return json.load(f)
    
    return {"answer": {}, "statistics": {}, "sp": {}, "error": {}}


def prepare_dataset(
    total: Optional[int] = None,
    pred_ckpt: Optional[Union[str, dict]] = None,
    log_dir: Optional[str] = None
):

    full_dataset = get_hotpotqa_fullwiki_devset()

    if total is None:
        total = len(full_dataset)

    if log_dir is not None and os.path.isdir(log_dir):
        goal_set: set = set()
        for log_file in os.listdir(log_dir):
            with open(os.path.join(log_dir, log_file), 'r') as f:
                log_data = json.load(f)
                if log_data and isinstance(log_data, list):
                    goal = log_data[0].get("goal")
                    if goal:
                        goal_set.add(goal)
        return [data for data in full_dataset if data["question"] in goal_set]

    if isinstance(pred_ckpt, dict):
        pred_dict = pred_ckpt
    else:
        pred_dict = get_pred_dict(pred_ckpt=pred_ckpt)

    dataset = []
    num_new_ids = 0
    for data in full_dataset:
        if data["_id"] not in pred_dict["statistics"]:
            if len(pred_dict["statistics"]) + num_new_ids >= total:
                break
            dataset.append(data)
            num_new_ids += 1
        elif data["_id"] in pred_dict.get("error", []):
            dataset.append(data)

    return dataset


def get_hotpotqa_fullwiki_devset(file: str = GT_FILE, url: str = GT_URL):

    if not os.path.isfile(file):
        response = requests.get(url)
        with open(file, 'wb') as f:
            f.write(response.content)

    with open(file, 'r') as f:
        return json.load(f)


def evaluate_log_dir(
    log_dir: str = LOG_DATA_DIR,
    pred_ckpt: Optional[str] = None
):
    pred_ckpt = pred_ckpt or os.path.join(PARENT_DIRECTORY, "prediction.json")
    pred_dict = predict_log_dir(log_dir=log_dir, pred_ckpt=pred_ckpt)
    with open(pred_ckpt, 'w') as f:
        json.dump(pred_dict, f, indent=2)
    eval(pred_ckpt, GT_FILE)


def predict_log_dir(
    log_dir: str = LOG_DATA_DIR,
    pred_ckpt: Optional[str] = None
):
    dataset = {
        data["question"]: data for data in prepare_dataset(log_dir=log_dir)
    }

    pred_dict = get_pred_dict(pred_ckpt=pred_ckpt)

    asyncio.run(collect_metrics(
        pred_dict=pred_dict, dataset=dataset, log_files=[
            os.path.join(log_dir, file) for file in os.listdir(log_dir)
        ]
    ))

    return pred_dict


async def collect_metrics(pred_dict: dict, dataset: dict, log_files: list):

    semaphore = asyncio.Semaphore(10)

    async def process_log_file(log_file: str):
        async with semaphore:
            with open(log_file, "r") as f:
                log_data = json.load(f)
                await evaluate_log_data(log_data, pred_dict, dataset)

    await tqdm_asyncio.gather(*[
        process_log_file(log_file) for log_file in log_files
    ])


async def evaluate_log_data(
    log_data: dict, pred_dict: dict, dataset: dict
):
    
    if not log_data or not isinstance(log_data, list):
        return
    question: str = log_data[0].get("goal")
    gt = dataset.get(question)
    if gt is None:
        return

    qid = gt["_id"]
    if qid in pred_dict["answer"]:
        return
    for key in list(pred_dict.keys()):
        if qid in pred_dict[key]:
            del pred_dict[key][qid]

    titles = []
    statistics = {
        "steps": 0, "equivalency": 0, "reasoning": '', "question": question, "gt_answer": gt["answer"], "gt_citations": [fact[0] for fact in gt["supporting_facts"]], "raw_citation_urls": [], "citations": {}, "rewritten": 0, "search_invoked": 0, "notepad_invoked": 0, "parse_error": 0, "invalid_tool": 0, "context_len_err": 0
    }
    
    for entry in log_data:

        if "query_rewrite" in entry:
            statistics["rewritten"] += 1

        if "error" in entry:
            if "Could not parse LLM output:" in entry["error"]:
                statistics["parse_error"] += 1
            elif "Invalid tool requested by the model." in entry["error"]:
                statistics["invalid_tool"] += 1
            elif "This model's maximum context length is" in entry["error"]:
                statistics["context_len_err"] += 1
            pred_dict["error"][qid] = entry["error"]

        if "conversations" in entry:
            await process_conversation_log(
                entry["conversations"], pred_dict, statistics, titles, gt
            )

    if titles:
        pred_dict["sp"][qid] = titles
    if isinstance(statistics["citations"], dict):
        statistics["citations"] = []
    pred_dict["statistics"][qid] = statistics
    if qid not in pred_dict["answer"] and qid not in pred_dict["error"]:
        pred_dict["error"][qid] = json.dumps(statistics, indent=2)


async def process_conversation_log(
    conversations: list, pred_dict: dict, statistics: dict, titles: list, gt: dict
):

    statistics["steps"] += 1

    history = conversations[0]["value"]
    if history and "observation" in history[-1]:
        observation = history[-1].get("observation")
        if isinstance(observation, list) and isinstance(observation[0], dict) and "title" in observation[0]:
            titles.append([doc["title"] for doc in observation])
            for doc in observation:
                statistics["citations"][doc["url"]] = doc["title"]

    try:
        prediction = json.loads(conversations[-1]["value"])
    except:
        statistics["parse_error"] += 1
        return
    action = prediction["action"]
    if action == "Tool_Wikipedia":
        statistics["search_invoked"] += 1
    elif action == "Tool_Notepad":
        statistics["notepad_invoked"] += 1
    elif action == "Tool_Finish":
        final_answer: str = prediction["action_input"]

        # Get list of citations
        citations = []
        for citation in prediction.get("citations", []):
            if ": " not in citation:
                continue
            url = citation.split(": ")[0]
            statistics["raw_citation_urls"].append(url)
            if url in statistics["citations"]:
                citations.append(statistics["citations"].get(url))
        statistics["citations"] = citations

        evalllm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_organization=os.getenv("OPENAI_API_ORG"),
            temperature=0,
            model=EVAL_MODEL_NAME,
            request_timeout=AWAIT_TIMEOUT
        )

        await evaluate_final_answer(final_answer, gt, evalllm, pred_dict, statistics)


async def evaluate_final_answer(
    final_answer: str, data: dict, evalllm, pred_dict, statistics
):

    question: str = data["question"]
    gt_answer: str = data["answer"]

    try:
        # Use GPT use determine if the final output is equivalent with the ground truth
        resp = await evalllm.agenerate([[HumanMessage(
            content=f"Given a question and a pair of answers. Determine if Answer1 can be strictly infered from Answer2. Return False if given the information in Answer2, we cannot determine whether Answer1 is right. Add detailed explaination and reasioning. Format your answer in JSON with a boolean field called 'is_inferable' and a string field 'reasoning' that can be loaded in python.\n\nQuestion: '{question}'\n\nAnswer1: '{gt_answer}'\n\nAnswer2: '{final_answer}'"
        )]])
        resp_obj = json.loads(resp.generations[0][0].text.strip())
        statistics["equivalency"] = int(resp_obj.get("is_inferable", 0))
        statistics["reasoning"] = resp_obj.get("reasoning", '')

        pred_dict["answer"][data["_id"]] = final_answer
        if data["_id"] in pred_dict["error"]:
            del pred_dict["error"][data["_id"]]

    except Exception as e:
        pred_dict["error"][data["_id"]] = f"Error during evalutaion: {e}"
