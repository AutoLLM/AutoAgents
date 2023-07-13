import os
import asyncio
import json

from autoagents.agents.search import ActionRunner
from autoagents.agents.wiki_agent import WikiActionRunner
from langchain.chat_models import ChatOpenAI


USE_WIKIAGENT: bool = False


async def work(user_input):
    outputq = asyncio.Queue()
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                     openai_organization=os.getenv("OPENAI_API_ORG"),
                     temperature=0,
                     model_name="gpt-4")
    runner = ActionRunner(outputq, llm=llm) if not USE_WIKIAGENT \
        else WikiActionRunner(outputq, llm=llm)
    task = asyncio.create_task(runner.run(user_input, outputq))

    while True:
        output = await outputq.get()
        if isinstance(output, Exception):
            print(f"Question: {user_input}")
            print(output)
            return
        try:
            parsed = json.loads(output)
            print(json.dumps(parsed, indent=2))
            print("-----------------------------------------------------------")
            if parsed["action"] == "Finish":
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

def main(q):
    return asyncio.run(work(q))

if __name__ == "__main__":
    for i, q in (Q if not USE_WIKIAGENT else Q_HOTPOTQA):
        if i == 2:
            main(q)
