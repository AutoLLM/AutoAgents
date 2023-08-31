# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
# Need to call this before importing transformers.
try:
    from fastchat.train.llama_flash_attn_monkey_patch import (
        replace_llama_attn_with_flash_attn,
    )

    replace_llama_attn_with_flash_attn()
except ImportError:
    pass # ignore if flash-attn not installed

from typing import Dict
import torch
from fastchat.conversation import (
        SeparatorStyle,
        Conversation,
        register_conv_template,
        get_conv_template
)
from fastchat.model.model_adapter import (
        get_conversation_template,
        BaseModelAdapter,
        register_model_adapter
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer
)

import fastchat.train.train as train
from fastchat.train.train import rank0_print
from fastchat.train.train import IGNORE_TOKEN_ID

class ActionAdapter(BaseModelAdapter):
    """The model adapter for Action Vicuna"""

    def match(self, model_path: str):
        return "action" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, config=config, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("action")


def preprocess(
    sources,
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("action")
    roles = {"goal": conv.roles[0],
             "tools": conv.roles[1],
             "history": conv.roles[2],
             "next_action": conv.roles[3]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE

    # Mask targets
    sep = conv.sep + conv.roles[-1] + ": "
    for conversation, target in zip(conversations, targets):
        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        inputs, outputs = conversation.split(sep)  # [.., goal, history | next_action]
        cur_len = 0
        instruction_len = len(tokenizer(inputs + sep).input_ids) - 1
        target[cur_len:cur_len+instruction_len] = IGNORE_TOKEN_ID
        cur_len += instruction_len - 1
        outputs_len = len(tokenizer(outputs).input_ids)
        target[cur_len+outputs_len:] = IGNORE_TOKEN_ID
        cur_len += outputs_len

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )



if __name__ == "__main__":
    # Action LLM default template
    register_conv_template(
        Conversation(
            name="action",
            system_message="Below is a goal you need to achieve. Given the available tools and history of past actions provide the next action to perform.",
            roles=("### Goal", "### Tools", "### History", "### Next action"),
            messages=(),
            offset=0,
            sep_style=SeparatorStyle.ADD_COLON_SINGLE,
            sep="\n\n",  # separator between roles
            )
    )
    register_model_adapter(ActionAdapter)

    # Monkeypatch preprocessing which handles the action roles
    train.preprocess = preprocess
    train.train()

