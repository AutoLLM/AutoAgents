# AutoAgents

<p align="center"><img src="https://raw.githubusercontent.com/AutoLLM/AutoAgents/assets/images/logo.png?raw=true" width=400/></p>

Unlock complex question answering in LLMs with enhanced chain-of-thought reasoning and information-seeking capabilities.

## 👉	Overview

The purpose of this project is to extend LLMs ability to answer more complex questions through chain-of-thought reasoning and information-seeking actions.

We are excited to release the initial version of AutoAgents, a proof-of-concept on what can be achieved with only well-written prompts. This is the initial step towards our first big milestone, releasing and open-sourcing the AutoAgents 7B model!
 
Come try out our [Huggingface Space](https://huggingface.co/spaces/AutoLLM/AutoAgents)!



## 🤖 The AutoAgents Project

This project demonstrates LLMs capability to execute a complex user goal: understand a user's goal, generate a plan, use proper tools, and deliver a final result.

For simplicity, our first attempt starts with a Web Search Agent.



## 💫 How it works:

<p align="left"><img src="https://raw.githubusercontent.com/AutoLLM/AutoAgents/assets/images/agent.png" width=830/></p>



## 📔 Examples

Ask your AutoAgent to do what a real person would do using the internet:

For example:

*1. Recommend a kid friendly movie that is playing at a theater near Sunnyvale. Give me the showtimes and a link to purchase the tickets*

*2. What is the average age of the past three president when they took office*

*3. What is the mortgage rate right now and how does that compare to the past two years*



## 💁 Roadmap

* ~~HuggingFace Space demo using OpenAI models~~ [LINK](https://huggingface.co/spaces/AutoLLM/AutoAgents)
* AutoAgents [7B] Model
  * Initial Release:
    * Finetune and release a 7B parameter fine-tuned search model
* AutoAgents Dataset
  * A high-quality dataset for a diverse set of search scenarios (why quality and diversity?<sup>[1](https://arxiv.org/abs/2305.11206)</sup>)
* Reduce Model Inference Overhead
* Affordance Modeling <sup>[2](https://en.wikipedia.org/wiki/Affordance)</sup>
* Extend Support to Additional Tools
* Customizable Document Search set (e.g. personal documents)
* Support Multi-turn Dialogue
* Advanced Flow Control in Plan Execution

We are actively developing a few interesting things, check back here or follow us on [Twitter](https://twitter.com/AutoLLM) for any new development.
 
If you are interested in any other problems, feel free to shoot us an issue.



## 🧭 How to use this repo?

This repo contains the entire code to run the search agent from your local browser. All you need is an OpenAI API key to begin.

To run the search agent locally:

1. Clone the repo and change the directory

  ```bash
  git clone https://github.com/AutoLLM/AutoAgents.git
  cd AutoAgents
  ```

2. Install the dependencies

  ```bash
  pip install -r requirements.txt
  ```

3. Install the `autoagents` package

  ```bash
  pip install -e .
  ```

4. Make sure you have your OpenAI API key set as an environment variable. Alternatively, you can also feed it through the input text-box on the sidebar.

  ```bash
  export OPENAI_API_KEY=sk-xxxxxx
  ```

5. Run the Streamlit app

  ```bash
  streamlit run autoagents/agents/spaces/app.py
  ```

This should open a browser window where you can type your search query.
