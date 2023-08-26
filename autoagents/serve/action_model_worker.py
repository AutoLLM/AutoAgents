import uvicorn
from fastchat.conversation import (
        SeparatorStyle,
        Conversation,
        register_conv_template
)

import fastchat.serve.model_worker as model_worker

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
    args, model_worker.worker = model_worker.create_model_worker()
    # hardcode the conv template
    args.conv_template = "action"
    uvicorn.run(model_worker.app, host=args.host, port=args.port, log_level="info")
