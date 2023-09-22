import json
import glob
from argparse import ArgumentParser
from collections import Counter, defaultdict
import numpy as np
from pprint import pprint


def get_common_stats(log_files):
    stats = {
        "counts": Counter(),  # general counters
        "error_counts": Counter(),  # error counters
        "plan_counts": Counter(),  # plan patterns
        "len_history_trace": [],
        "len_initial_plan": []
    }
    samples = set()
    finished_samples = set()
    for file in log_files:
        with open(file, "r") as f:
            log_data = json.load(f)
            summary = get_summary_from_log_data(log_data=log_data)
            stats["counts"] += summary["counts"]
            stats["error_counts"] += summary["error_counts"]
            stats["plan_counts"] += summary["plan_counts"]
            stats["len_history_trace"].extend(summary["len_history_trace"])
            stats["len_initial_plan"].extend(summary["len_initial_plan"])
            if summary["question"] is not None:
                samples.add(summary["question"])
                if summary["answer"] is not None:
                    finished_samples.add(summary["question"])
    stats["counts"]["total_samples"] = len(samples)
    stats["counts"]["finished_samples"] = len(finished_samples)

    hist, rng = np.histogram(stats["len_history_trace"], bins=range(0, 16))
    stats["len_history_trace"] = hist.tolist()
    
    hist, rng = np.histogram(stats["len_initial_plan"], bins=range(0, 16))
    stats["len_initial_plan"] = hist.tolist()

    return stats


def get_summary_from_log_data(log_data: list):

    counts = Counter()  # general counters
    error_counts = Counter()  # error counters
    plan_counts = Counter()  # plan patterns
    len_initial_plan = []
    len_history_trace = []

    summary = dict(
        counts=counts,
        error_counts=error_counts,
        plan_counts=plan_counts,
        len_history_trace=len_history_trace,
        len_initial_plan=len_initial_plan,
        question=None,
        answer=None
    )
    
    # Handle errors and rewrites
    is_valid: bool = True
    counts["total_logs"] += 1
    for entry in log_data:
        if "id" in entry:
            counts["total_steps"] += 1
        if "error" in entry:
            if "Expecting value" in entry["error"]:
                # This is the old rewrite error
                pass
            elif "Invalid tool requested by the model." in entry["error"]:
                error_counts["invalid_tools_error"] += 1
                is_valid = False
            elif "This model's maximum context length" in entry["error"]:
                error_counts["context_len_error"] += 1
                if len(log_data) < 4:
                    is_valid = False
            elif "[Errno -3] Temporary failure in name resolution" in entry["error"]:
                error_counts["dns_error"] += 1
            elif "Could not parse LLM output:" in entry["error"]:
                error_counts["parse_error"] += 1
                is_valid = False
            elif "Rate limit reached for " in entry["error"]:
                error_counts["rate_limit_error"] += 1
                is_valid = False
            else:
                error_counts["other_error"] += 1
                is_valid = False
        elif "query_rewrite" in entry:
            counts["total_rewrites"] += 1

    if not is_valid:
        return summary
    counts["total_valid"] += 1

    for entry in log_data:
        if "goal" in entry:
            summary["question"] = entry["goal"]
        if "conversations" in entry:
            counts["valid_steps"] += 1
            prediction = json.loads(entry["conversations"][-1]["value"])
            action = prediction["action"]
            if action == "Tool_Search" or action == "Tool_Wikipedia":
                counts["search_invoked"] += 1
            elif action == "Tool_Notepad":
                counts["notepad_invoked"] += 1
            elif action == "Tool_Finish":
                summary["answer"] = prediction["action_input"]

    # do last-step history analysis, log_data[-3]
    try:
        last_convo = log_data[-3]["conversations"]
        if last_convo[1]["from"] in ("gpt", "ai"):
            # we don't have the system key in prompt_v3
            output = json.loads(last_convo[1]["value"])
        else:
            output = json.loads(last_convo[2]["value"])
    except:
        return summary

    counts[f"EndWith_{output['action']}"] += 1

    if last_convo[0]["from"] == "history":
        hist = last_convo[0]["value"]
        actions = [h["action"] for h in hist]
        if len(actions) < 5 and len(actions) > 0:
            actions_str = "->".join(actions) 
            plan_counts[actions_str] += 1
            if actions_str == "Tool_Notepad":
                pass
            if actions_str == "Tool_Search->Tool_Search->Tool_Notepad":
                pass
            if actions_str == "Tool_Search->Tool_Search->Tool_Search->Tool_Notepad":
                pass
        plans = []
        for plan in [h["plan"] for h in hist]:
            plans.extend(plan)
        for plan in plans:
            if plan.startswith("Visit"):
                counts["visit_in_plan"] += 1
                break

        len_hist = len(hist)
        if len_hist > 0:
            len_plan0 = len(hist[0]["plan"])
            len_initial_plan.append(len_plan0)
            if len_plan0 == 1:
                pass
        counts["len_hist"] += len_hist
        len_history_trace.append(len_hist)

        # find out if there are duplicate action+action_inputs
        inputs = defaultdict(set)
        plans = set()
        for h in hist + [output]:
            if h["action"] in inputs:
                if h["action_input"] in inputs[h["action"]]:
                    if output["action"] == "Tool_Finish":
                        counts["Finish_with_dups"] += 1
                        break
                    else: # only count duplicates that didn't finish
                        counts["duplicate_actions"] += 1
                        break
            inputs[h["action"]].add(h["action_input"])

    return summary


def main():

    parser = ArgumentParser()
    parser.add_argument("log_dir", type=str, help="path of the log directory")
    args = parser.parse_args()

    stats = get_common_stats(log_files=glob.glob(f"{args.log_dir}/*.json"))
    pprint(stats)


if __name__ == "__main__":
    main()
