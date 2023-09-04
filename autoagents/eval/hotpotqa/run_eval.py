import os
import time
import asyncio
import json
import logging
from langchain.chat_models import ChatOpenAI
from pprint import pformat
from ast import literal_eval
from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm

from autoagents.agents.agents.wiki_agent import WikiActionRunner
from autoagents.agents.models.custom import CustomLLM
from autoagents.eval.hotpotqa.eval_async import (
    evaluate_final_answer, prepare_dataset
)
from autoagents.eval.hotpotqa.hotpotqa_eval import eval
from autoagents.eval.hotpotqa.constants import *


if not os.path.isdir(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
log_filehandler = logging.FileHandler(RUN_EVAL_LOG_FILE)
log_filehandler.setLevel(logging.DEBUG)
log_filehandler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n%(message)s')
)
logger.addHandler(log_filehandler)


def get_llms():
    if MODEL_NAME not in OPENAI_MODEL_NAMES:
        llm = CustomLLM(
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            request_timeout=AWAIT_TIMEOUT
        )
    else:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_organization=os.getenv("OPENAI_API_ORG"),
            temperature=TEMPERATURE,
            model_name=MODEL_NAME,
            request_timeout=AWAIT_TIMEOUT
        )

    evalllm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_organization=os.getenv("OPENAI_API_ORG"),
        temperature=0,
        model=EVAL_MODEL_NAME,
        request_timeout=AWAIT_TIMEOUT
    )
    return llm, evalllm


async def work(data, pred_dict):
    outputq = asyncio.Queue()
    user_input = data["question"]

    llm, evalllm = get_llms()
    runner = WikiActionRunner(outputq, llm=llm, persist_logs=PERSIST_LOGS)
    task = asyncio.create_task(runner.run(user_input, outputq))

    titles = []
    statistics = {
        "steps": 0, "equivalency": 0, "reasoning": '', "question": user_input, "gt_answer": data["answer"], "raw_citation_urls": [], "citations": {}, "rewritten": 0, "search_invoked": 0, "notepad_invoked": 0, "multi_tools": 0, "parse_error": 0, "invalid_tool": 0, "context_len_err": 0
    }
    for _ in range(runner.agent_executor.max_iterations or MAX_ROUND_STEPS):

        try:
            output = await asyncio.wait_for(outputq.get(), AWAIT_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"Question: {user_input}\nError: Timed out waiting for output from queue\n")
            pred_dict["error"][data["_id"]] = "Timed out waiting for output from queue."
            break
        statistics["steps"] += 1

        if isinstance(output, Exception):
            logger.error(f"Question: {user_input}\nError: {output}\n")
            if isinstance(output, RuntimeWarning) and "Action Input Rewritten: " in str(output):
                statistics["rewritten"] += 1
                continue
            else:
                if "Could not parse LLM output: " in str(output):
                    statistics["parse_error"] += 1
                elif "Invalid tool requested by the model." in str(output):
                    statistics["invalid_tool"] += 1
                elif "This model's maximum context length is" in str(output):
                    statistics["context_len_err"] += 1
                pred_dict["error"][data["_id"]] = str(output)
                break
        
        parsed = get_parsed_output(user_input, output, statistics, titles)

        if isinstance(parsed, dict) and parsed.get("action") == "Tool_Finish":
            final_answer: str = parsed["action_input"]
            logger.info(f"Question: {user_input}\nFinal Output: {final_answer}\n")

            # Get list of citations
            citations = []
            for citation in parsed.get("citations", []):
                if ": " not in citation:
                    continue
                url = citation.split(": ")[0]
                statistics["raw_citation_urls"].append(url)
                if url in statistics["citations"]:
                    citations.append(statistics["citations"].get(url))
            statistics["citations"] = citations

            await evaluate_final_answer(final_answer, data, evalllm, pred_dict, statistics)

            break
    if titles:
        pred_dict["sp"][data["_id"]] = json.dumps(titles)
    if isinstance(statistics["citations"], dict):
        statistics["citations"] = []
    pred_dict["statistics"][data["_id"]] = json.dumps(statistics)
    if data["_id"] not in pred_dict["answer"] and data["_id"] not in pred_dict["error"]:
        pred_dict["error"][data["_id"]] = json.dumps(statistics, indent=2)

    # await task
    try:
        return await asyncio.wait_for(task, AWAIT_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error(f"Question: {user_input}\nError: Timed out waiting for task to complete\n")
        pred_dict["error"][data["_id"]] = "Timed out waiting for task to complete."


def get_parsed_output(user_input, output, statistics, titles):
    parsed = None
    try:
        parsed = json.loads(output)
        logger.debug(f"Question: {user_input}\n{json.dumps(parsed, indent=2)}")
        if parsed["action"] == "Tool_Wikipedia":
            statistics["search_invoked"] += 1
        elif parsed["action"] == "Tool_Notepad":
            statistics["notepad_invoked"] += 1
    except:
        try:
            parsed = literal_eval(output)
            logger.debug(f"Question: {user_input}\n{json.dumps(parsed, indent=2)}")
            if isinstance(parsed, list) and isinstance(parsed[0], dict) and "title" in parsed[0]:
                titles.append([doc["title"] for doc in parsed])
                for doc in parsed:
                    statistics["citations"][doc["url"]] = doc["title"]
        except:
            logger.debug(f"Question: {user_input}\n{output}")
    return parsed


def save_output():

    if PERSIST_LOGS:
        for log_file in os.listdir(LOG_DATA_DIR):
            os.rename(
                src=os.path.join(LOG_DATA_DIR, log_file),
                dst=os.path.join(NEW_LOG_DIR, log_file)
            )
        os.rmdir(LOG_DATA_DIR)

    output_dict = dict(pred_dict)
    for k in list(output_dict.keys()):
        output_dict[k] = dict(output_dict[k])
        if k in ("sp", "statistics"):
            for qid in output_dict[k]:
                output_dict[k][qid] = json.loads(output_dict[k][qid])
                if isinstance(output_dict[k][qid], str):
                    output_dict[k][qid] = json.loads(output_dict[k][qid])

    logger.info(pformat(output_dict, indent=2))
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_dict, f, indent=2)

    wrong_ans = []
    for qid, stat in output_dict["statistics"].items():
        if stat["equivalency"] == 0:
            wrong_ans.append({
                "question": stat["question"],
                "gt_answer": stat["gt_answer"],
                "prediction": output_dict["answer"].get(qid, ''),
                "reasoning": stat["reasoning"]
            })
    with open(WRONG_ANS_OUTPUT_FILE, 'w') as f:
        json.dump(wrong_ans, f, indent=2)


def initialize_pred_dict():

    pred_dict["answer"] = manager.dict()
    pred_dict["statistics"] = manager.dict()
    pred_dict["sp"] = manager.dict()
    pred_dict["error"] = manager.dict()

    cur_dict = {}
    if os.path.isfile(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            cur_dict = json.load(f)
            pred_dict["answer"].update(cur_dict["answer"])
            for _id, sp in cur_dict["sp"].items():
                pred_dict["sp"][_id] = json.dumps(sp)
            for _id, stat in cur_dict["statistics"].items():
                pred_dict["statistics"][_id] = json.dumps(stat)


def retry(dataset):

    # Retry until we get all the final answers
    round = 0
    while pred_dict["error"] and round < MAX_RETRY_ROUND:

        logger.info(
            f"Round {round}. Start retrying failed samples: "
            f"{json.dumps(dict(pred_dict['error']), indent=2)}"
        )

        retry_data = []
        for i in range(len(dataset)):
            if dataset[i]["_id"] in pred_dict["error"]:
                retry_data.append(dataset[i])
                del pred_dict["error"][dataset[i]["_id"]]

        time.sleep(ROUND_WAITTIME)

        with Pool(processes=10) as pool:
            for _ in tqdm(pool.imap_unordered(
                partial(main, pred_dict=pred_dict), retry_data
            ), total=len(retry_data)):
                pass

        round += 1


def main(data, pred_dict):
    asyncio.run(work(data, pred_dict))


if __name__ == "__main__":

    manager = Manager()

    pred_dict = manager.dict()

    initialize_pred_dict()

    dataset = prepare_dataset(total=NUM_SAMPLES_TOTAL, pred_ckpt=pred_dict)

    if PERSIST_LOGS:
        if not os.path.isdir(LOG_DATA_DIR):
            os.mkdir(LOG_DATA_DIR)
        if not os.path.isdir(NEW_LOG_DIR):
            os.mkdir(NEW_LOG_DIR)

    with Pool(processes=10) as pool:
        for _ in tqdm(pool.imap_unordered(
            partial(main, pred_dict=pred_dict), dataset
        ), total=len(dataset)):
            pass

    retry(dataset=dataset)

    save_output()

    eval(OUTPUT_FILE, GT_FILE)
