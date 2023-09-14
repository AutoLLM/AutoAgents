import os
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", type=str,
        help="file containing the evaluation results"
    )
    args = parser.parse_args()

    if os.path.isfile(args.input_file):
        with open(args.input_file, 'r') as f:
            results = json.load(f)
            num = len(results)

            stats = {"avg_overall": 0, "avg_clarity": 0, "avg_effectiveness": 0}
            for obj in results:
                stats["avg_overall"] += obj.get("overall_score", 0)
                stats["avg_clarity"] += obj.get("clarity_score", 0)
                stats["avg_effectiveness"] += obj.get("effectiveness_score", 0)

            for key in stats:
                stats[key] /= num
            stats["num"] = num
            print(stats)
