import glob
import json
from colorama import Fore, Back, Style

files1 = glob.glob(f"longchat-13b-16k-v3-3-ddg-bamboogle/*.json")
files2 = glob.glob(f"longchat-13b-16k-v3-3-wiki-bamboogle/*.json")
ddg_steps, wiki_steps, total1, total2, ddg_use_goal_in_question, wiki_use_goal_in_question, same_first_query = 0, 0, 0, 0, 0, 0, 0

for file1 in files1:
    with open(file1, "r") as f:
        log_data1 = json.load(f)
        question1 = log_data1[0]["goal"]
        for file2 in files2:
            with open(file2, "r") as f:
                log_data2 = json.load(f)
                question2 = log_data2[0]["goal"]
                if question1 == question2:
                    print(Fore.RED + f"Question: {question1}")
                    ddg_steps += len(log_data1) - 3
                    wiki_steps += len(log_data2) - 3
                    try:
                        ddg_search_term = json.loads(
                        log_data1[1]["conversations"][-1]["value"])["action_input"]
                        ddg_plans = json.loads(
                        log_data1[1]["conversations"][-1]["value"])["plan"]
                        wiki_search_term = json.loads(
                        log_data2[1]["conversations"][-1]["value"])["action_input"]
                        wiki_plans = json.loads(
                        log_data1[1]["conversations"][-1]["value"])["plan"]
                        total1 += 1
                        total2 += 1
                        if ddg_search_term.lower() == wiki_search_term.lower():
                            print(Fore.GREEN + f"Same First Search Query: {ddg_search_term}")
                            same_first_query += 1
                        else:
                            print(Fore.GREEN + f"Ddg First Search Query: {ddg_search_term}")
                            print(Fore.GREEN + f"WikiFirst Search Query: {wiki_search_term}")
                        print(f"Ddg plan steps: {len(ddg_plans)} wiki plan steps: {len(wiki_plans)}")
                        if question1 == ddg_search_term or question1[:-1] == ddg_search_term:
                            ddg_use_goal_in_question += 1
                        if question2 == wiki_search_term or question1[:-1] == wiki_search_term:
                            wiki_use_goal_in_question += 1
                    except:
                        pass
                    break

print(Style.RESET_ALL)
print(f"Average number of steps: ddg {ddg_steps/total1} wiki {wiki_steps/total2}")
print(f"Ddg directly use goal as first question rate: {ddg_use_goal_in_question}/{total1}={ddg_use_goal_in_question/total1}")
print(f"Wiki directly use goal as first question rate: {wiki_use_goal_in_question}/{total2}={wiki_use_goal_in_question/total2}")
print(f"Same first query rate: {same_first_query}/{total1}={same_first_query/total1}")