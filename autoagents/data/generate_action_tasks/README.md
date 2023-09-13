# Generate_action_tasks
Generate action tasks for AutoGPT following self-instruct. 


## What does this repo do

This repo only generates the tasks that needs the agent to complete, not the full action data that includes reasoning, planning and execution of tools. That part of codes is implemented together with the AutoAgent repo. 


## Repo Structure

* `REACT.ipynb` is the exploration of creating action data.
* `generate_data_chat_api.py` is the main file that adopts openAI's API to generate more tasks based on in-context learning. 
* `prompt.txt` is the prompt used by `generate_data_chat_api.py`
* `seed_tasks.jsonl`: the manually labeled tasks for in-context learning
* `generate_data_chat_api.py`: uses the completion API, not the chat API. 


## Usage
Simply run 
`sh run_genenerate_data.sh`. Parameters could be modified. 

