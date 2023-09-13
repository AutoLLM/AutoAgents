"""
run:
python -m generate_data generate_agents_data \
  --output_dir ./ \
  --num_agents_to_generate 10 \
  --model_name="text-davinci-003" \
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


def post_process_gpt_response(num_prompt_agents, response):
    print ("post_process_gpt_response")
    if response is None:
        return []
    raw_instructions = f"{num_prompt_agents+1}. Name:" + response["text"]
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
            role = splitted_data[4].strip()
            # goals = splitted_data[6].strip()
            # goals = "" if goals.lower() == "<nogoal>" else goals
        # filter out too short or too long role
        if len(role.split()) <= 3 or len(role.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "kill",
            "harm",
            "discriminate",
        ]
        blacklist += []
        if any(find_word_in_string(word, role) for word in blacklist):
            continue
        # filter those starting with punctuation
        if role[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not role[0].isascii():
            continue
        agents.append({"name": name, "goal": role})
    return agents


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_agents_data(
    output_dir="./",
    seed_tasks_path="./seed_tasks.jsonl",
    num_agents_to_generate=20,
    model_name="text-davinci-003",
    num_prompt_agents=5,
    request_batch_size=1,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
):
    print ("generate_instruction_following_data")
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_agent_data = [
        {"name": t["name"], "goal": t["goal"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_agent_data)} human-written seed agents")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_agent_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_agent_data = utils.jload(os.path.join(output_dir, "regen.json"))
        print(f"Loaded {len(machine_agent_data)} machine-generated agents")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_agents_to_generate)
    if machine_agent_data:
        progress_bar.update(len(machine_agent_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_roles = [d["goal"] for d in seed_agent_data] + [
        d["goal"] for d in machine_agent_data
    ]
    all_instruction_tokens = [scorer._tokenizer.tokenize(role) for role in all_roles]

    while len(machine_agent_data) < num_agents_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_agents = random.sample(seed_agent_data, num_prompt_agents)
            prompt = encode_prompt(prompt_agents)
            batch_inputs.append(prompt)
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=["\n20", "20.", "20."],
        )
        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        agent_data = []
        for result in results:
            new_agents = post_process_gpt_response(num_prompt_agents, result)
            agent_data += new_agents

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
            most_similar_instructions = {
                all_roles[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            agent_data_entry["most_similar_instructions"] = most_similar_instructions
            agent_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_agent_data.append(agent_data_entry)
            all_roles.append(agent_data_entry["goal"])
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