import json
import glob
from collections import Counter
import argparse

counts = Counter()

Goals = set()

def process_file(data, name):
    if name.startswith("train_data") or name.startswith("final") or name.startswith("ab_"):
        return
    for d in data:
        if "error" in d:
            counts["error"] += 1
            return
        elif "query_rewrite" in d:
            counts["query_rewrite"] += 1
            return
    output = json.loads(data[-3]["conversations"][2]["value"])
    if output["action"] != "Tool_Finish":
        counts["no_final_answer"] += 1
        return
    # remove dups in case
    goal = data[0]["goal"]
    if goal not in Goals:
        Goals.add(goal)
    else:
        return
    costs = data[-2]
    data = data[1:-2]

    counts["conv_len"] += len(data)
    counts["total"] += 1
    
    counts["totals_cost"] += costs["metrics"]["total_cost"]
    data_new = []
    for d in data:
        convs = []
        for conv in d["conversations"]:
            k, v = conv.values()
            if k == "system":
                convs.append({"from": "human", "value": v})
            elif k == "ai":
                convs.append({"from": "gpt", "value": v})
        assert len(convs) == 2
        data_new.append({"id": d["id"], "conversations": convs})
    return data_new

def main(dir_path, save=False):
    assert dir_path is not None
    train_data = []
    for name in glob.glob(f"{dir_path}/*.json"):
        dname = name.split("/")[-1].split(".")[0]
        with open(f"{dir_path}/{dname}.json", "r") as file:
            data = json.load(file)
        if (filtered_data := process_file(data, dname)):
            train_data += filtered_data
    if save:
        with open(f"{dir_path}/sft_data.json", "w") as f:
            json.dump(train_data, f, indent=2)
    print(counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir_path', type=str)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    main(args.data_dir_path, args.save)

