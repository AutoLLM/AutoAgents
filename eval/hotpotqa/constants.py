import os


GT_FILE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hotpot_dev_fullwiki_v1.json"
)

GT_URL: str = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"

MODEL_NAME: str = "gpt-3.5-turbo"

PERSIST_LOGS: bool = True

EVAL_MODEL_NAME: str = "gpt-3.5-turbo"

TEMPERATURE: int = 0

NUM_SAMPLES_TOTAL: int = 10

AWAIT_TIMEOUT: int = 360

ROUND_WAITTIME: int = 10

MAX_RETRY_ROUND: int = 1

OPENAI_MODEL_NAMES = {"gpt-3.5-turbo", "gpt-4"}

PARENT_DIRECTORY: str = os.path.dirname(os.path.abspath(__file__))

OUTPUT_FILE: str = os.path.join(
    PARENT_DIRECTORY, f"prediction_{MODEL_NAME}.json"
)

WRONG_ANS_OUTPUT_FILE: str = os.path.join(
    PARENT_DIRECTORY, f"wrong_answers_{MODEL_NAME}.json"
)

LOG_DATA_DIR: str = os.path.join(os.getcwd(), "data")
