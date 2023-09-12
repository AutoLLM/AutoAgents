import json


def get_common_stats(log_files):
    stats = {
            "average_search_invoked": 0,
            "average_notepad_invoked": 0,
            "average_rewritten": 0,
            "invalid_tools": 0,
            "average_steps": 0,
            "parse_error": 0,
            "context_len_err": 0,
            "total_samples": 0,
            "average_answer_missing": 0,
            "finished_samples": 0
    }
    samples = set()
    finished_samples = set()
    for file in log_files:
        with open(file, "r") as f:
            log_data = json.load(f)
            for entry in log_data:
                if "goal" in entry:
                    question = entry["goal"]
                    samples.add(question)
                if "query_rewrite" in entry:
                    stats["average_rewritten"] += 1
                if "error" in entry:
                    if "Could not parse LLM output:" in entry["error"]:
                        stats["parse_error"] += 1
                    elif "Invalid tool requested by the model." in entry["error"]:
                        stats["invalid_tools"] += 1
                    elif "This model's maximum context length is" in entry["error"]:
                        stats["context_len_err"] += 1
                if "conversations" in entry:
                    stats["average_steps"] += 1
                    prediction = json.loads(entry["conversations"][-1]["value"])
                    action = prediction["action"]
                    if action == "Tool_Search" or action == "Tool_Wikipedia":
                        stats["average_search_invoked"] += 1
                    elif action == "Tool_Notepad":
                        stats["average_notepad_invoked"] += 1
                    elif action == "Tool_Finish":
                        finished_samples.add(question)
    stats["total_samples"] = len(samples)
    stats["finished_samples"] = len(finished_samples)
    stats["average_steps"] = stats["average_steps"] / len(log_files)
    stats["average_rewritten"] = stats["average_rewritten"] / len(log_files)
    stats["average_search_invoked"] = stats["average_search_invoked"] / len(log_files)
    stats["average_notepad_invoked"] = stats["average_notepad_invoked"] / len(log_files)
    return stats
