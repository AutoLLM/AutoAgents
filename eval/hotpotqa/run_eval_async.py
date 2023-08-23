import sys
import os
import json
import time
import asyncio
import requests
from dataclasses import dataclass
from langchain.chat_models import ChatOpenAI

from constants import *
from hotpotqa_eval import eval
from run_eval import evaluate_final_answer

sys.path.insert(0, os.path.dirname(os.path.dirname(PARENT_DIRECTORY)))
from test import main


@dataclass
class WorkerArgs:
    agent: str = "wiki"
    model: str = MODEL_NAME
    temperature: float = TEMPERATURE
    persist_logs: bool = PERSIST_LOGS


def prepare_dataset(pred_dict: dict):

    if not os.path.isfile(GT_FILE):
        response = requests.get(GT_URL)
        with open(GT_FILE, 'wb') as f:
            f.write(response.content)

    with open(GT_FILE, 'r') as f:
        full_dataset = json.load(f)
        dataset = {}
        num_new_ids = 0
        for data in full_dataset:
            if data["_id"] not in pred_dict["statistics"]:
                if len(pred_dict["statistics"]) + num_new_ids >= NUM_SAMPLES_TOTAL:
                    break
                dataset[data["question"]] = data
                num_new_ids += 1
            elif data["_id"] in pred_dict.get("error", []):
                dataset[data["question"]] = data

    return dataset


async def collect_metrics(pred_dict: dict, dataset: dict, log_files: list):

    semaphore = asyncio.Semaphore(10)

    async def process_log_file(log_file: str):
        async with semaphore:
            with open(log_file, "r") as f:
                log_data = json.load(f)
                await evaluate_log_data(log_data, pred_dict, dataset)

    await asyncio.gather(*[
        process_log_file(log_file) for log_file in log_files
    ])

    save_output(pred_dict=pred_dict)


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
                entry["conversations"], statistics, titles, gt
            )

    if titles:
        pred_dict["sp"][qid] = titles
    pred_dict["statistics"][qid] = statistics
    if qid not in pred_dict["answer"] and qid not in pred_dict["error"]:
        pred_dict["error"][qid] = json.dumps(statistics, indent=2)


async def process_conversation_log(
    conversations: list, statistics: dict, titles: list, gt: dict
):

    statistics["steps"] += 1

    history = conversations[0]["value"]
    if history and "observation" in history[0]:
        observation = history[0].get("observation")
        if isinstance(observation, list) and isinstance(observation[0], dict) and "title" in observation[0]:
            titles.append([doc["title"] for doc in observation])
            for doc in observation:
                statistics["citations"][doc["url"]] = doc["title"]

    prediction = json.loads(conversations[-1]["value"])
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

        await evaluate_final_answer(final_answer, gt, evalllm, statistics)


def save_output(pred_dict: dict):

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(pred_dict, f)

    wrong_ans = []
    for qid, stat in pred_dict["statistics"].items():
        if stat["equivalency"] == 0:
            wrong_ans.append({
                "question": stat["question"],
                "gt_answer": stat["gt_answer"],
                "prediction": pred_dict["answer"].get(qid, ''),
                "reasoning": stat["reasoning"]
            })
    with open(WRONG_ANS_OUTPUT_FILE, 'w') as f:
        json.dump(wrong_ans, f)


def run():

    pred_dict = {"answer": {}, "statistics": {}, "sp": {}, "error": {}}

    if os.path.isfile(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            pred_dict = json.load(f)

    if not os.path.isdir(LOG_DATA_DIR):
        os.mkdir(LOG_DATA_DIR)
    new_log_dir: str = os.path.join(PARENT_DIRECTORY, f"data-{MODEL_NAME}")
    if not os.path.isdir(new_log_dir):
        os.mkdir(new_log_dir)
    
    dataset = prepare_dataset(pred_dict)

    # Retry until we get all the final answers or until the limit of rounds
    for round in range(MAX_RETRY_ROUND + 1):

        asyncio.run(main(dataset.keys(), WorkerArgs()))

        asyncio.run(collect_metrics(
            pred_dict=pred_dict, dataset=dataset, log_files=[
                os.path.join(LOG_DATA_DIR, file)
                for file in os.listdir(LOG_DATA_DIR)
            ]
        ))

        for file in os.listdir(LOG_DATA_DIR):
            os.rename(
                src=os.path.join(LOG_DATA_DIR, file),
                dst=os.path.join(new_log_dir, file)
            )

        if round == MAX_RETRY_ROUND or not pred_dict["error"]:
            break

        retry_data = {}
        for data in dataset.values():
            if data["_id"] in pred_dict["error"]:
                retry_data[data["question"]] = data
        dataset = retry_data

        time.sleep(ROUND_WAITTIME)

    os.rmdir(LOG_DATA_DIR)

    eval(OUTPUT_FILE, GT_FILE)


if __name__ == "__main__":
    run()
