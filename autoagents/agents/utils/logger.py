import os
import json
from typing import Dict, Any, List
import uuid
from datetime import datetime
import pytz

import huggingface_hub
from huggingface_hub import Repository


class InteractionsLogger:
    def __init__(self, name: str, persist=False):
        self.persist = persist
        self.messages = []
        self.counter = 0
        self.name = name  # unique id
        HF_TOKEN = os.environ.get("HF_TOKEN")
        HF_DATASET_REPO_URL = os.environ.get("HF_DATASET_REPO_URL")
        if (HF_TOKEN is not None) and (HF_DATASET_REPO_URL is not None):
            self.repo = Repository(
                local_dir="data", clone_from=HF_DATASET_REPO_URL, use_auth_token=HF_TOKEN
            )
        else:
            self.repo = None

    def set_goal(self, goal: str):
        self.messages.append({"goal": goal})

    def set_tools(self, tools: List):
        self.messages.append({"tools": tools})

    def add_history(self, hist: Dict):
        self.convos = [{"from": "history", "value": hist}]

    def add_ai(self, msg: Dict):
        self.convos.append({"from": "ai", "value": msg})
        self.messages.append({"id": f"{self.name}_{self.counter}", "conversations": self.convos})
        self.counter += 1

    def add_system(self, more: Dict):
        self.convos.append({"from": "system", "value": more})

    def add_message(self, data: Dict[str, Any]):
        self.messages.append(data)

    def save(self):
        self.add_message({"datetime": datetime.now(pytz.utc).strftime("%m/%d/%Y %H:%M:%S %Z%z")})
        if self.persist:
            # TODO: want to add retry in a loop?
            if self.repo is not None:
                self.repo.git_pull()
            fname = uuid.uuid4().hex[:16]
            with open(f"./data/{fname}.json", "w") as f:
                json.dump(self.messages, f, indent=2)
            if self.repo is not None:
                commit_url = self.repo.push_to_hub()

    def add_cost(self, cost):
        self.messages.append({"metrics": cost})
