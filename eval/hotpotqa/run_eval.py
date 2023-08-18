import os
import time
import requests
import asyncio
import json
import logging
from autoagents.agents.wiki_agent import WikiActionRunner
from autoagents.models.custom import CustomLLM
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from pprint import pformat
from ast import literal_eval
from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm

from hotpotqa_eval import eval


GT_FILE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hotpot_dev_fullwiki_v1.json"
)
GT_URL: str = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"

MODEL_NAME: str = "gpt-4"
EVAL_MODEL_NAME: str = "gpt-3.5-turbo"
TEMPERATURE: int = 0
NUM_SAMPLES_TOTAL: int = 200
AWAIT_TIMEOUT: int = 120
ROUND_WAITTIME: int = 60
MAX_RETRY_ROUND: int = 2

OPENAI_MODEL_NAMES = {"gpt-3.5-turbo", "gpt-4"}

OUTPUT_FILE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"prediction_{MODEL_NAME}.json"
)
WRONG_ANS_OUTPUT_FILE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"wrong_answers_{MODEL_NAME}.json"
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
log_filehandler = logging.FileHandler("run_eval.log")
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
    runner = WikiActionRunner(outputq, llm=llm)
    task = asyncio.create_task(runner.run(user_input, outputq))

    titles = []
    statistics = {
        "steps": 0, "equivalency": 0, "reasoning": '', "question": user_input, "gt_answer": data["answer"], "citations": {}, "rewritten": 0, "search_invoked": 0, "notepad_invoked": 0, "multi_tools": 0, "parse_error": 0, "invalid_tool": 0, "context_len_err": 0
    }
    while True:

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
                url = citation.split(": ")[0]
                if url in statistics["citations"]:
                    citations.append(statistics["citations"].get(url))
            statistics["citations"] = citations

            await evaluate_final_answer(final_answer, data, evalllm, statistics)

            break
    if titles:
        pred_dict["sp"][data["_id"]] = json.dumps(titles)
    pred_dict["statistics"][data["_id"]] = json.dumps(statistics)

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


async def evaluate_final_answer(final_answer, data, evalllm, statistics):

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
        logger.error(f"Question: {question}\nError: {e}\n")
        pred_dict["error"][data["_id"]] = "Error during evalutaion."


def save_output():

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
        json.dump(output_dict, f)

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
        json.dump(wrong_ans, f)


def prepare_dataset():

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

    if not os.path.isfile(GT_FILE):
        logger.info(f"Download ground truth file from {GT_URL} ...")
        response = requests.get(GT_URL)
        with open(GT_FILE, 'wb') as f:
            f.write(response.content)

    with open(GT_FILE, 'r') as f:
        full_dataset = json.load(f)
        dataset = []
        num_new_ids = 0
        for data in full_dataset:
            if data["_id"] not in pred_dict["statistics"]:
                if len(pred_dict["statistics"]) + num_new_ids >= NUM_SAMPLES_TOTAL:
                    break
                dataset.append(data)
                num_new_ids += 1
            elif data["_id"] in cur_dict.get("error", []):
                dataset.append(data)

    return dataset


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
    
    dataset = prepare_dataset()

    with Pool(processes=10) as pool:
        for _ in tqdm(pool.imap_unordered(
            partial(main, pred_dict=pred_dict), dataset
        ), total=len(dataset)):
            pass

    retry(dataset=dataset)

    save_output()

    eval(OUTPUT_FILE, GT_FILE)
