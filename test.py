import os
import asyncio
import json
import argparse

from autoagents.agents.search import ActionRunner
from autoagents.agents.wiki_agent import WikiActionRunner
from langchain.chat_models import ChatOpenAI
from autoagents.models.custom import CustomLLM
from pprint import pprint


OPENAI_MODEL_NAMES = {"gpt-3.5-turbo", "gpt-4"}
AWAIT_TIMEOUT: int = 120


async def work(user_input, model: str, temperature: int, use_wikiagent: bool, persist_logs: bool):
    outputq = asyncio.Queue()
    if model not in OPENAI_MODEL_NAMES:
        llm = CustomLLM(
            model_name=model,
            temperature=temperature,
            request_timeout=AWAIT_TIMEOUT
        )
    else:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_organization=os.getenv("OPENAI_API_ORG"),
            temperature=temperature,
            model_name=model,
            request_timeout=AWAIT_TIMEOUT
        )
    runner = ActionRunner(outputq, llm=llm, persist_logs=persist_logs) if not use_wikiagent \
        else WikiActionRunner(outputq, llm=llm, persist_logs=persist_logs)
    task = asyncio.create_task(runner.run(user_input, outputq))

    while True:
        try:
            output = await asyncio.wait_for(outputq.get(), AWAIT_TIMEOUT)
        except asyncio.TimeoutError:
            break
        if isinstance(output, RuntimeWarning):
            print(f"Question: {user_input}")
            print(output)
            continue
        elif isinstance(output, Exception):
            print(f"Question: {user_input}")
            print(output)
            return
        try:
            parsed = json.loads(output)
            print(json.dumps(parsed, indent=2))
            print("-----------------------------------------------------------")
            if parsed["action"] == "Tool_Finish":
                break
        except:
            print(f"Question: {user_input}")
            print(output)
            print("-----------------------------------------------------------")
    return await task

Q = [
     (0, "list 3 cities and their current populations where Paramore is playing this year."),
     (1, "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"),
     (2, "How many watermelons can fit in a Tesla Model S?"),
     (3, "Recommend me some laptops suitable for UI designers under $2000. Please include brand and price."),
     (4, "Build me a vacation plan for Rome and Milan this summer for seven days. Include place to visit and hotels to stay. "),
     (5, "What is the sum of ages of the wives of Barack Obama and Donald Trump?"),
     (6, "Who is the most recent NBA MVP? Which team does he play for? What is his season stats?"),
     (7, "What were the scores for the last three games for the Los Angeles Lakers? Provide the dates and opposing teams."),
     (8, "Which team won in women's volleyball in the Summer Olympics that was held in London?"),
     (9, "Provide a summary of the latest COVID-19 research paper published. Include the title, authors and abstract."),
     (10, "What is the top grossing movie in theatres this week? Provide the movie title, director, and a brief synopsis of the movie's plot. Attach a review for this movie."),
     (11, "Recommend a bagel shop near the Strip district in Pittsburgh that offer vegan food"),
     (12, "Who are some top researchers in the field of machine learning systems nowadays?"),
     ]

Q_HOTPOTQA = [
    (0, "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?"),
    (1, "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"),
    (2, "Were Scott Derrickson and Ed Wood of the same nationality?"),
    (3, "Who won the last NBA championship and what's the series score?"),
    (4, "Who is the current CEO of Apple Inc and what has been done by him?"),
    (5, "Who is indicted after a special counsel investigation charges him with mishandling classified documents and how old is him?")
]

FT = [(0, "Briefly explain the current global climate change adaptation strategy and its effectiveness."),
      (1, "What steps should be taken to prepare a backyard garden for spring planting?"),
      (2, "Report the critical reception of the latest superhero movie."),
      (3, "When is the next NBA or NFL finals game scheduled?"),
      (4, "Which national parks or nature reserves are currently open for visitors near Denver, Colorado?"),
      (5, "Who are the most recent Nobel Prize winners in physics, chemistry, and medicine, and what are their respective contributions?"),
      ]

HF = [(0, "Recommend me a movie in theater now to watch with kids."),
      (1, "Who is the most recent NBA MVP? Which team does he play for? What are his career stats?"),
      (2, "Who is the head coach of AC Milan now? How long has he been coaching the team?"),
      (3, "What is the mortgage rate right now and how does that compare to the past two years?"),
      (4, "What is the weather like in San Francisco today? What about tomorrow?"),
      (5, "When and where is the upcoming concert for Taylor Swift? Share a link to purchase tickets."),
      (6, "Find me recent studies focusing on hallucination in large language models. Provide the link to each study found."),
      ]


async def main(questions, model: str, temperature: int, use_wikiagent: bool, persist_logs: bool):
    await asyncio.gather(*[work(q, model, temperature, use_wikiagent, persist_logs) for q in questions])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--agent",
        default="ddg",
        const="ddg",
        nargs="?",
        choices=("ddg", "wiki"),
        help='which action agent we want to interact with(default: ddg)'
    )
    parser.add_argument("--data-json", type=str)
    parser.add_argument("--persist-logs", action='store_true')
    args = parser.parse_args()
    print(args)
    use_wikiagent = False if args.agent == "ddg" else True
    questions = []
    if use_wikiagent:
        questions = [q for _, q in Q_HOTPOTQA]
    else:
        questions = [q for _, q in HF]
    if args.data_json:
        # TODO: prepare dataset
        pass
    persist_logs = True if args.persist_logs else False
    asyncio.run(main(questions, args.model, args.temperature, use_wikiagent, persist_logs))
