import os


MODEL_NAME: str = "gpt-3.5-turbo"

PERSIST_LOGS: bool = True

EVAL_MODEL_NAME: str = "gpt-3.5-turbo"

TEMPERATURE: float = 0

NUM_SAMPLES_TOTAL: int = 200

AWAIT_TIMEOUT: int = 360

ROUND_WAITTIME: int = 10

MAX_RETRY_ROUND: int = 1

MAX_ROUND_STEPS: int = 30

OPENAI_MODEL_NAMES = {"gpt-3.5-turbo", "gpt-4"}

PARENT_DIRECTORY: str = os.path.dirname(os.path.abspath(__file__))

GT_FILE: str = os.path.join(PARENT_DIRECTORY, "hotpot_dev_fullwiki_v1.json")

GT_URL: str = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"

RESULTS_DIR: str = os.path.join(PARENT_DIRECTORY, f"results_{MODEL_NAME}")

OUTPUT_FILE: str = os.path.join(RESULTS_DIR, f"prediction.json")

RUN_EVAL_LOG_FILE: str = os.path.join(RESULTS_DIR, "run_eval.log")

WRONG_ANS_OUTPUT_FILE: str = os.path.join(RESULTS_DIR, f"wrong_answers.json")

LOG_DATA_DIR: str = os.path.join(os.getcwd(), "data")

NEW_LOG_DIR: str = os.path.join(RESULTS_DIR, "data")
