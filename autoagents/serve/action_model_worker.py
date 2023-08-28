import uvicorn
import fastchat.serve.model_worker as model_worker
from fastchat.conversation import (
        SeparatorStyle,
        Conversation,
        register_conv_template,
        get_conv_template
)
from fastchat.model.model_adapter import (
        register_model_adapter,
        BaseModelAdapter
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)

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
    args, model_worker.worker = model_worker.create_model_worker()
    # hardcode the conv template
    args.conv_template = "action"
    uvicorn.run(model_worker.app, host=args.host, port=args.port, log_level="info")
