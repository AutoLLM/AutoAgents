import os

rescale = int(os.environ.get("CONDENSE_RESCALE", 1))
if rescale > 1:
    from longchat.train.monkey_patch.llama_condense_monkey_patch import replace_llama_with_condense
    replace_llama_with_condense(rescale)

from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from fastchat.train.train import train

if __name__ == "__main__":
    train()
