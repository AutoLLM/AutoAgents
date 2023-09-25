import os
import json
import asyncio
import requests
from argparse import ArgumentParser
from typing import Optional, Union
from tqdm.asyncio import tqdm_asyncio
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

from autoagents.eval.hotpotqa.constants import *
from autoagents.agents.utils.constants import LOG_SAVE_DIR
from autoagents.eval.metrics import get_summary_from_log_data
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

    def get_questions(self, total: Optional[int] = None):
        dataset = prepare_dataset(total=total, pred_ckpt=self.pred_file)
        return [data["question"] for data in dataset]

    def run(self, log_dir: Optional[str] = None):

        if log_dir is None:
            if not os.path.isdir(self.new_log_dir):
                os.mkdir(self.new_log_dir)
            if os.path.isdir(LOG_SAVE_DIR):
                for log_file in os.listdir(LOG_SAVE_DIR):
                    os.rename(
                        src=os.path.join(LOG_SAVE_DIR, log_file),
                        dst=os.path.join(self.new_log_dir, log_file)
                    )
                os.rmdir(LOG_SAVE_DIR)
            log_dir = self.new_log_dir

        pred_dict = predict_log_dir(log_dir=log_dir, pred_ckpt=self.pred_file)

        self.save_output(pred_dict=pred_dict)

        eval(self.pred_file, GT_FILE)

    def save_output(self, pred_dict: dict):

        with open(self.pred_file, 'w') as f:
            json.dump(pred_dict, f, indent=2)

        wrong_ans = []
        for qid, stat in pred_dict["statistics"].items():
            if stat["summary"]["counts"].get("equivalency", 0) == 0:
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
    filtered_dataset = []

    if total is None:
        total = len(full_dataset)

    if log_dir is not None and os.path.isdir(log_dir):
        goal_set: set = set()
        for log_file in os.listdir(log_dir):
            with open(os.path.join(log_dir, log_file), 'r') as f:
                try:
                    log_data = json.load(f)
                except json.decoder.JSONDecodeError:
                    continue
                if log_data and isinstance(log_data, list):
                    goal = None
                    for entry in log_data:
                        if "goal" in entry:
                            goal = entry["goal"]
                            break
                    if goal:
                        goal_set.add(goal)
        for data in full_dataset:
            for goal in goal_set:
                if data["question"] in goal:
                    filtered_dataset.append(data)
        return filtered_dataset

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
    log_dir: str = LOG_SAVE_DIR,
    pred_ckpt: Optional[str] = None
):
    pred_ckpt = pred_ckpt or os.path.join(PARENT_DIRECTORY, "prediction.json")
    pred_dict = predict_log_dir(log_dir=log_dir, pred_ckpt=pred_ckpt)
    with open(pred_ckpt, 'w') as f:
        json.dump(pred_dict, f, indent=2)
    eval(pred_ckpt, GT_FILE)


def predict_log_dir(
    log_dir: str = LOG_SAVE_DIR,
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
                try:
                    log_data = json.load(f)
                except json.decoder.JSONDecodeError:
                    return
                await evaluate_log_data(log_data, pred_dict, dataset)

    await tqdm_asyncio.gather(*[
        process_log_file(log_file) for log_file in log_files
    ])


async def evaluate_log_data(
    log_data: dict, pred_dict: dict, dataset: dict
):
    
    if not log_data or not isinstance(log_data, list):
        return
    summary = get_summary_from_log_data(log_data=log_data)
    question = summary["question"]
    if question is None:
        return
    gt = None
    for q in dataset:
        if q in question:
            gt = dataset[q]
            break
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
        "reasoning": '',
        "question": question,
        "gt_answer": gt["answer"],
        "gt_citations": [fact[0] for fact in gt["supporting_facts"]],
        "raw_citation_urls": [],
        "citations": {},
        "summary": summary
    }

    if summary["answer"] is not None:
        pred_dict["answer"][gt["_id"]] = summary["answer"]
        if gt["_id"] in pred_dict["error"]:
            del pred_dict["error"][gt["_id"]]
        await evaluate_final_answer(summary["answer"], gt, pred_dict, statistics)
    
    for entry in log_data:

        if "error" in entry:
            pred_dict["error"][qid] = entry["error"]

        if "conversations" in entry:
            await process_conversation_log(
                entry["conversations"], statistics, titles
            )

    if titles:
        pred_dict["sp"][qid] = titles
    if isinstance(statistics["citations"], dict):
        statistics["citations"] = []
    pred_dict["statistics"][qid] = statistics
    if qid not in pred_dict["answer"] and qid not in pred_dict["error"]:
        pred_dict["error"][qid] = json.dumps(statistics, indent=2)


async def process_conversation_log(
    conversations: list, statistics: dict, titles: list
):  
    try:
        observation = conversations[0]["value"][-1]["observation"]
        titles.append([doc["title"] for doc in observation])
        for doc in observation:
            statistics["citations"][doc["url"]] = doc["title"]
    except:
        pass

    try:
        prediction = json.loads(conversations[-1]["value"])
    except json.decoder.JSONDecodeError:
        statistics["summary"]["error_counts"]["parse_error"] += 1
        return
    if prediction["action"] == "Tool_Finish":
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


async def evaluate_final_answer(
    final_answer: str, data: dict, pred_dict, statistics, llm=None
):

    question: str = data["question"]
    gt_answer: str = data["answer"]

    try:
        # Use GPT to determine if the final output is equivalent with the ground truth
        resp_obj = await check_answer_equivalency(question, gt_answer, final_answer, llm)
        statistics["summary"]["counts"]["equivalency"] = int(resp_obj.get("is_inferable", 0))
        statistics["reasoning"] = resp_obj.get("reasoning", '')

    except Exception as e:
        pred_dict["error"][data["_id"]] = f"Error during evalutaion: {e}"


async def check_answer_equivalency(question: str, answer1: str, answer2: str, llm=None):

    if llm is None:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_organization=os.getenv("OPENAI_API_ORG"),
            temperature=0,
            model=EVAL_MODEL_NAME,
            request_timeout=AWAIT_TIMEOUT
        )

    # Use GPT to determine if the answer1 is equivalent with answer2
    resp = await llm.agenerate([[HumanMessage(
        content=f"Given a question and a pair of answers. Determine if Answer1 can be strictly infered from Answer2. Return False if given the information in Answer2, we cannot determine whether Answer1 is right. Add detailed explaination and reasioning. Format your answer in JSON with a boolean field called 'is_inferable' and a string field 'reasoning' that can be loaded in python.\n\nQuestion: '{question}'\n\nAnswer1: '{answer1}'\n\nAnswer2: '{answer2}'"
    )]])
    return json.loads(resp.generations[0][0].text.strip())


def main():

    parser = ArgumentParser()
    parser.add_argument("log_dir", type=str, help="path of the log directory")
    parser.add_argument("--pred_ckpt", type=str, help="path of the log directory")
    args = parser.parse_args()
    evaluate_log_dir(args.log_dir, args.pred_ckpt)

if __name__ == "__main__":
    main()
