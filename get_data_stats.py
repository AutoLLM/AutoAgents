import os
import json
import pdb
from collections import Counter

def report_data_statistics(folder_path):
    rewrite_num, error_num, final_answer_num, parse_error = 0, 0, 0, 0
    timeout_num = 0
    total = 0
    round_counter = Counter()
    queries_with_errors = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".clean.json"):
            rm_flag = False
            file_path = os.path.join(folder_path, filename)

            # Read the JSON file and save to a dictionary
            with open(file_path, "r") as file:
                data = json.load(file)
            
            goal = data['goal']
            
            total += 1
            
            # Check if there is a key 'rewrite_query'
            if 'query_rewrite' in data:
                rewrite_query = data['query_rewrite']
                rewrite_num += 1
                print(f"query_rewrite found: {rewrite_query}")
            
            # Check if there is a key 'error'
            if 'error' in data:
                error = data['error']
                error_num += 1
                print(f"error found: {error}")
                print (filename)
                queries_with_errors.append(goal)
                if "Rate limit reached" in error:
                    timeout_num += 1
                # delte file_path locally
                rm_flag = True
            else:
                n_round = len(data)
                round_counter[n_round-2] += 1   # there is a field for goal and datetime which needs to be substracted.
                # if n_round >= 6:
                #     print (filename, goal, n_round)
            # Get the second last key and value
            try:
                if len(data) >= 2:
                    keys = list(data.keys())
                    second_last_key = keys[-2]
                    second_last_value = data[second_last_key]
                    action, action_input = second_last_value['output']['action'], second_last_value['output']['action_input']
                    # print(f"Final action: {action}, text: {action_input}")
                    if action == "Final Answer" and action_input:
                        final_answer_num += 1
            except:
                # print (filename)
                parse_error += 1
                # check if file_path exist, if so, delete it
                rm_flag = True

    print (f"Total number of files: {total}")
    print (f"Number of files with rewrite_query: {rewrite_num}")
    print (f"Number of files with error: {error_num}")
    print (f"Number of files with tle error: {timeout_num}")
    print (f"Number of files with final answer: {final_answer_num}")
    print (f"Round Counter: {round_counter}")

    # save queries_with_errors to a file
    # with open("./generate_data/queries_with_errors.txt", "w") as f:
    #     for query in queries_with_errors:
    #         f.write(query + "\n")


if __name__ == "__main__":
    folder_path = "./data"
    report_data_statistics(folder_path)