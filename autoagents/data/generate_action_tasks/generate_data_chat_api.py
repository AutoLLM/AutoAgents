"""
run:
python -m generate_data_chat_api generate_agents_data \
  --output_dir ./new_data \
  --seed_tasks_path ./seed_tasks.jsonl \
  --num_agents_to_generate 1000 \
  --model_name="gpt-4" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils

import fire
import pdb

'''
The tools used by Auto-GPT are available at:
https://github.com/Significant-Gravitas/Auto-GPT/tree/master/autogpt/commands
'''

def encode_prompt(prompt_agents):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_agents):
        (name, goal) = task_dict["name"], task_dict["goal"]
        if not goal:
            raise
        prompt += f"###\n"
        prompt += f"{idx + 1}. Name: {name}\n"
        prompt += f"{idx + 1}. Goal:\n{goal}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Name:"
    return prompt


def post_process_chat_gpt_response(num_prompt_agents, response):
    print ("post_process_gpt_response")
    if response is None:
        return []
    raw_instructions = f"{num_prompt_agents+1}. Name:" + response['message']['content']
    raw_instructions = re.split("###", raw_instructions)
    agents = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_agents + 1
        splitted_data = re.split(f"{idx}\.\s+(Name|Goal):", inst)
        if len(splitted_data) != 5:
            continue
        else:
            name = splitted_data[2].strip()
            goal = splitted_data[4].strip()
        # filter out too short or too long role
        if len(goal.split()) <= 3 or len(goal.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "kill",
            "harm",
            "discriminate",
            "racist",
            "figure",
            "plot",
            "chart",
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, goal) for word in blacklist):
            continue
        # filter those starting with punctuation
        if goal[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not goal[0].isascii():
            continue
        agents.append({"name": name, "goal": goal})
    return agents


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_agents_data(
    output_dir="./",
    seed_tasks_path="./new_seed_tasks.jsonl",
    num_agents_to_generate=50,
    model_name="gpt-3.5-turbo",
    num_prompt_agents=8,
    temperature=1.0,
    top_p=1.0,
    num_cpus=8,
):
    print ("generate_instruction_following_data")
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_agent_data = [
        {"name": t["name"], "goal": t["goal"], "task_id": t["task_id"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_agent_data)} human-written seed agents")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_agent_data = []
    machine_data_path = os.path.join(output_dir, "self-gen-batch1.json")
    if os.path.exists(machine_data_path):
        # machine_agent_data = utils.jload(machine_data_path)
        machine_agent_data = [json.loads(l) for l in open(machine_data_path, "r")]
        print(f"Loaded {len(machine_agent_data)} machine-generated agents")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_agents_to_generate)
    if machine_agent_data:
        progress_bar.update(len(machine_agent_data))

    previous_goals = []
    if os.path.isfile("previous_goals.json"):
        with open("previous_goals.json", 'r') as f:
            previous_goals = json.load(f)

    # first we tokenize all the seed instructions and generated machine instructions
    all_goals = [d["goal"] for d in seed_agent_data] + [
        d["goal"] for d in machine_agent_data
    ] + previous_goals
    all_goals = list(set(all_goals))
    all_instruction_tokens = [scorer._tokenizer.tokenize(role) for role in all_goals]

    while len(machine_agent_data) < num_agents_to_generate:
        request_idx += 1

        # only sampling from the seed tasks
        prompt_agents = random.sample(seed_agent_data, num_prompt_agents)
        prompt = encode_prompt(prompt_agents)

        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n60", "60."],
        )
        request_start = time.time()
        results = utils.openai_completion(
            prompts=prompt,
            model_name=model_name,
            batch_size=1,
            decoding_args=decoding_args,
            logit_bias={"100257": -100},  # prevent the <|endoftext|> from being generated
            # "100265":-100, "100276":-100 for <|im_end|> and <endofprompt> token 
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        agent_data = post_process_chat_gpt_response(num_prompt_agents, results)

        total = len(agent_data)
        keep = 0
        for agent_data_entry in agent_data:
            # computing similarity with the pre-tokenzied instructions
            new_agent_tokens = scorer._tokenizer.tokenize(agent_data_entry["goal"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_agent_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            # most_similar_instructions = {
            #     all_goals[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            # }
            max_score = max(rouge_scores)
            if max_score > 0.40:
                continue
            else:
                keep += 1
            # agent_data_entry["most_similar_instructions"] = most_similar_instructions
            # agent_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            agent_data_entry["max_similarity_score"] = max_score
            agent_data_entry["seed_tasks"] = [task["task_id"] for task in prompt_agents]
            machine_agent_data.append(agent_data_entry)
            all_goals.append(agent_data_entry["goal"])
            all_instruction_tokens.append(new_agent_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} agents, kept {keep} agents")
        utils.jdump(machine_agent_data, os.path.join(output_dir, "self-gen.json"))


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)