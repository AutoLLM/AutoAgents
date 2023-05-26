import os
import json
from typing import Dict, Any
import uuid

import huggingface_hub
from huggingface_hub import Repository


class InteractionsLogger:
    def __init__(self, name: str, persist=False):
        self.persist = persist
        self.counter = 0
        self.name = name  # unique id
        HF_TOKEN = os.environ.get("HF_TOKEN")
        HF_DATASET_REPO_URL = os.environ.get("HF_DATASET_REPO_URL")
        if (HF_TOKEN is not None) and (HF_DATASET_REPO_URL is not None):
            self.repo = Repository(
                local_dir="data", clone_from=HF_DATASET_REPO_URL, use_auth_token=HF_TOKEN
            )
        else:
            self.persist = False

    def set_goal(self, goal: str):
        # Initialize two variables for saving two files (self.messages for
        # training and self.structure_data for later use)
        self.messages = [{"goal": goal}]
        self.structured_data = {"goal": goal}

    def add_system(self, more: Dict):
        self.convos = [{"from": "system"} | more]

    def add_ai(self, msg: str):
        self.convos.append({"from": "ai", "value": msg})
        self.messages.append({"id": f"{self.name}_{self.counter}", "conversations": self.convos})
        self.counter += 1

    def add_structured_data(self, data: Dict[str, Any]):
        self.structured_data.update({f"turn_{self.counter}": data})

    def add_message(self, data: Dict[str, Any]):
        self.structured_data.update(data)

    def save(self):
        if self.persist:
            # TODO: want to add retry in a loop?
            self.repo.git_pull()
            fname = uuid.uuid4().hex[:16]
            with open(f"./data/{fname}.json", "w") as f:
                json.dump(self.messages, f, indent=2)
            with open(f"./data/{fname}.clean.json", "w") as f:
                json.dump(self.structured_data, f, indent=2)
            commit_url = self.repo.push_to_hub()

    def add_cost(self, cost):
        self.messages.append({"metrics": cost})
