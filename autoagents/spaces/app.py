import os
import asyncio
import random
from datetime import date, datetime, timezone, timedelta
from ast import literal_eval

import streamlit as st
import openai

from autoagents.utils.constants import MAIN_HEADER, MAIN_CAPTION, SAMPLE_QUESTIONS
from autoagents.agents.search import ActionRunner

from langchain.chat_models import ChatOpenAI


async def run():
    output_acc = ""
    st.session_state["random"] = random.randint(0, 99)
    if "task" not in st.session_state:
        st.session_state.task = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = "gpt-3.5-turbo"

    st.set_page_config(
        page_title="Search Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(MAIN_HEADER)
    st.caption(MAIN_CAPTION)

    with st.form("my_form", clear_on_submit=False):
        st.markdown("<style> .inter { white-space: pre-line; } </style>", unsafe_allow_html=True)
        user_input = st.text_input(
            "You: ",
            key="input",
            placeholder="Ask me anything ...",
            label_visibility="hidden",
        )

        submitted = st.form_submit_button(
            "Search", help="Hit to submit the search query."
        )

        # Ask the user to enter their OpenAI API key
        if (api_key := st.sidebar.text_input("OpenAI api-key", type="password")):
            api_org = None
        else:
            api_key, api_org = os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_API_ORG")
        with st.sidebar:
            model_dict = {
                "gpt-3.5-turbo": "GPT-3.5-turbo",
                "gpt-4": "GPT-4 (Better but slower)",
            }
            st.radio(
                "OpenAI model",
                model_dict.keys(),
                key="model_name",
                format_func=lambda x: model_dict[x],
            )

            time_zone = str(datetime.now(timezone(timedelta(0))).astimezone().tzinfo)
            st.markdown(f"**The system time zone is {time_zone} and the date is {date.today()}**")

            st.markdown("**Example Queries:**")
            for q in SAMPLE_QUESTIONS:
                st.markdown(f"*{q}*")

        if not api_key:
            st.warning(
                "API key required to try this app. The API key is not stored in any form. [This](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key) might help."
            )
        elif api_org and st.session_state.model_name == "gpt-4":
            st.warning(
                "The free API key does not support GPT-4. Please switch to GPT-3.5-turbo or input your own API key."
            )
        else:
            outputq = asyncio.Queue()
            runner = ActionRunner(outputq,
                                  ChatOpenAI(openai_api_key=api_key,
                                             openai_organization=api_org,
                                             temperature=0,
                                             model_name=st.session_state.model_name),
                                  persist_logs=True)  # log to HF-dataset

            async def cleanup(e):
                st.error(e)
                await st.session_state.task
                st.session_state.task = None
                st.stop()

            placeholder = st.empty()

            if user_input and submitted:
                if st.session_state.task is not None:
                    with placeholder.container():
                        st.session_state.task.cancel()
                        st.warning("Previous search aborted", icon="⚠️")

                st.session_state.task = asyncio.create_task(
                    runner.run(user_input, outputq)
                )
                iterations = 0
                with st.expander("Search Results", expanded=True):
                    while True:
                        with st.spinner("Wait for it..."):
                            output = await outputq.get()
                            placeholder.empty()
                            if isinstance(output, Exception):
                                if isinstance(output, openai.error.AuthenticationError):
                                    await cleanup(f"AuthenticationError: Invalid OpenAI API key.")
                                elif isinstance(output, openai.error.InvalidRequestError) \
                                      and output._message == "The model: `gpt-4` does not exist":
                                    await cleanup(f"The free API key does not support GPT-4. Please switch to GPT-3.5-turbo or input your own API key.")
                                elif isinstance(output, openai.error.OpenAIError):
                                    await cleanup(output)
                                elif isinstance(output, RuntimeWarning):
                                    st.warning(output)
                                    continue
                                else:
                                    await cleanup("Something went wrong. Please try searching again.")
                                return
                            try:
                                output_fmt = literal_eval(output)
                                st.json(output_fmt, expanded=False)
                                st.write("---")
                                iterations += 1
                            except:
                                output_acc += "\n" + output
                                st.markdown(f"<div class=\"inter\"> {output} </div>",
                                            unsafe_allow_html=True)
                            if iterations >= runner.agent_executor.max_iterations:
                                await cleanup(
                                    f"Maximum iterations ({iterations}) exceeded. You can try running the search again or try a variation of the query."
                                )
                                return
                            if "Final Answer:" in output:
                                break
                # Found the answer
                final_answer = await st.session_state.task
                final_answer = final_answer.replace("$", "\$")
                # st.success accepts md
                st.success(final_answer, icon="✅")
                st.balloons()
                st.session_state.task = None
                st.stop()

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.set_debug(enabled=False)
    loop.run_until_complete(run())
