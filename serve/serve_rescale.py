import uvicorn

from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

import os

# TODO: change from env var to proper args (passing the rest of the args to create_model_worker)
rescale = int(os.environ.get("CONDENSE_RESCALE", 1))
if rescale > 1:
    longchat.train.monkey_patch.llama_condense_monkey_patch import replace_llama_with_condense
    replace_llama_with_condense(rescale)



from fastchat.serve.model_worker import create_model_worker

if __name__ == "__main__":
    args, worker = create_model_worker()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
