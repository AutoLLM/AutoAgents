## How to use this repo?

This repo contains the entire code to run the search agent from your local browser. All you need is an OpenAI API key to begin. The agent is built using LangChain and Streamlit. The repo contains `autoagents/agents`, which currently only includes a search agent and will be extended to more tool-using agents in the future. The search agent uses an Auto-GPT-style prompt to plan and execute multiple searches to reach a final answer which combines information from various search results. `autoagents/tools` contain two tools at the moment, a search engine API and a notepad used to note-down intermediate information. `autoagents/utils` has some utilities like the dataset logger. `autoagents/spaces` contains code for the Streamlit app hosted as a HuggingFace space.

To run the search agent locally:

1. Clone the repo
`git clone https://github.com/AutoLLM/AutoAgents.git` and `cd AutoAgents`

2. Install the dependencies
`pip install -r requirements.txt`

3. Make sure you have your OpenAI API key set as an environment variable. Alternatively, you can also feed it through the input text-box on the sidebar.
`export OPENAI_API_KEY=sk-xxxxxx`

4. Run the Streamlit app
`streamlit run autoagents/spaces/app.py`


This should open a browser window where you can type your search query.
