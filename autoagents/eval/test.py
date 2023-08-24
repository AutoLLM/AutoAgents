import os
import asyncio

from ast import literal_eval
from multiprocessing import Pool, TimeoutError

from autoagents.agents.search import ActionRunner
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from autoagents.agents.models.custom import CustomLLM
import json
from pprint import pprint


async def work(user_input):
    outputq = asyncio.Queue()
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                     openai_organization=os.getenv("OPENAI_API_ORG"),
                     temperature=0.,
                     model_name="gpt-4")
    runner = ActionRunner(outputq, llm=llm, persist_logs=False)
    task = asyncio.create_task(runner.run(user_input, outputq))

    while True:
        output = await outputq.get()
        if isinstance(output, RuntimeWarning):
            print(output)
            continue
        elif isinstance(output, Exception):
            print(output)
            return
        try:
            parsed = json.loads(output)
            print(json.dumps(parsed, indent=2))
            print("-----------------------------------------------------------")
            if parsed["action"] == "Tool_Finish":
                break
        except:
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


def main(q):
    return asyncio.run(work(q))

if __name__ == "__main__":
    for i, q in HF:
        if i == 5:
            main(q)
