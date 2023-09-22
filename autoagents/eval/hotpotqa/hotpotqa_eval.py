import sys
import numpy as np
import ujson as json
import re
import string
from collections import Counter


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    #     return ZERO_METRIC
    # if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    #     return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold, statistics):

    # Only match titles
    cur_sp_pred = set(title for rank in prediction for title in rank)
    gold_sp_pred = set(x[0] for x in gold)

    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall

    if all(e in cur_sp_pred for e in gold_sp_pred) and \
    statistics["summary"]["counts"].get("equivalency", 0) == 0:
        metrics["wrong_infer"] += 1

    title_to_ranks = {}
    for title_list in prediction:
        for i, title in enumerate(title_list):
            if title not in gold_sp_pred:
                continue
            if title not in title_to_ranks:
                title_to_ranks[title] = [i + 1] * 3
            title_to_ranks[title][0] = min(i + 1, title_to_ranks[title][0])
            title_to_ranks[title][2] = i + 1
    n_gt_titles = len(gold_sp_pred)
    cur_ranks = title_to_ranks.values()
    metrics["max_mrr"] += sum(1 / ranks[0] for ranks in cur_ranks) / n_gt_titles
    metrics["first_mrr"] += sum(1 / ranks[1] for ranks in cur_ranks) / n_gt_titles
    metrics["last_mrr"] += sum(1 / ranks[2] for ranks in cur_ranks) / n_gt_titles

    return em, prec, recall

def update_last_sp(metrics, statistics, gold):

    # Only match titles
    cur_sp_pred = set(title for title in statistics.get("citations", []))
    gold_sp_pred = set(x[0] for x in gold)

    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['last_sp_em'] += em
    metrics['last_sp_f1'] += f1
    metrics['last_sp_prec'] += prec
    metrics['last_sp_recall'] += recall

    return em, prec, recall

def eval(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
        print(f"len answer = {len(prediction['answer'])}, len error = {len(prediction['error'])}, len sp = {len(prediction['sp'])}, len statistics = {len(prediction['statistics'])}")

    with open(gold_file) as f:
        gold = []
        for data in json.load(f):
            if data["_id"] in prediction["statistics"]:
                gold.append(data)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'last_sp_em': 0, 'last_sp_f1': 0, 'last_sp_prec': 0, 'last_sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0,
        "max_mrr": 0, "first_mrr": 0, "last_mrr": 0, "wrong_infer": 0}
    stats = {
        "counts": Counter(),  # general counters
        "error_counts": Counter(),  # error counters
        "plan_counts": Counter(),  # plan patterns
        "len_history_trace": [],
        "len_initial_plan": []
    }
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        summary = prediction['statistics'][cur_id]["summary"]
        stats["counts"] += summary["counts"]
        stats["error_counts"] += summary["error_counts"]
        stats["plan_counts"] += summary["plan_counts"]
        stats["len_history_trace"].extend(summary["len_history_trace"])
        stats["len_initial_plan"].extend(summary["len_initial_plan"])

        if cur_id not in prediction['sp']:
            print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'], prediction['statistics'][cur_id])
            update_last_sp(
                metrics, prediction['statistics'].get(cur_id, {}), dp['supporting_facts']
            )

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    hist, rng = np.histogram(stats["len_history_trace"], bins=range(0, 16))
    stats["len_history_trace"] = hist.tolist()
    hist, rng = np.histogram(stats["len_initial_plan"], bins=range(0, 16))
    stats["len_initial_plan"] = hist.tolist()
    
    stats["error_rate"] = {
        error: cnt / N
        for error, cnt in stats["error_counts"].items()
    }
    stats["avg_metrics"] = {
        metric: cnt / N
        for metric, cnt in stats["counts"].items()
    }
    metrics.update(stats)

    metrics["ans_missing_rate"] = 1 - len(prediction["answer"]) / N
    metrics["sp_missing_rate"] = 1 - len(prediction["sp"]) / N
    metrics["num_evaluated"] = N
    metrics["llm_accuracy_on_finished_samples"] = stats["avg_metrics"]["equivalency"] / (1 - metrics["ans_missing_rate"])

    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    eval(sys.argv[1], sys.argv[2])
