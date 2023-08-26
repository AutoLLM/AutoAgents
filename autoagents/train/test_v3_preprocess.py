import numpy as np
import json
import torch
import click
import tempfile
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data import Dataset
from typing import Dict

from fastchat.conversation import (
        SeparatorStyle,
        Conversation,
        register_conv_template
)
from fastchat.model.model_adapter import (
        get_conversation_template,
        register_model_adapter
)

from train_v3 import ActionAdapter

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def rank0_print(*x):
    print(*x)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
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

    show_index = 3
    # Mask targets
    sep = conv.sep + conv.roles[-1] + ": "
    for i, (conversation, target) in enumerate(zip(conversations, targets)):
        if i == show_index:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        total_len = int(target.ne(tokenizer.pad_token_id).sum()) # = len(tokenizer(conversation).input_ids)
        inputs, outputs = conversation.split(sep)  # [.., goal, history | next_action]
        cur_len = 0
        instruction_len = len(tokenizer(inputs + sep).input_ids) - 1
        target[cur_len:cur_len+instruction_len] = IGNORE_TOKEN_ID
        cur_len += instruction_len - 1
        outputs_len = len(tokenizer(outputs).input_ids)
        target[cur_len+outputs_len:] = IGNORE_TOKEN_ID
        cur_len += outputs_len

        if i == show_index:
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

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_path
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        SupervisedDataset
    )
    rank0_print("Loading data...")
    raw_data = json.load(open(data_path, "r"))

    # Split train/test
    np.random.seed(0)
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


@click.command()
@click.option('--model_name_or_path', required=True)
@click.option('--data_path', required=True)
@click.option('--model_max_length', default=4096)
def main(model_name_or_path, data_path, model_max_length):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=tempfile.TemporaryDirectory().name,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    make_supervised_data_module(tokenizer=tokenizer, data_path=data_path)

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
    main()
