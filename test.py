import os
import asyncio

from ast import literal_eval
from multiprocessing import Pool, TimeoutError

from autoagents.agents.search import ActionRunner
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from autoagents.models.custom import CustomLLM
import json


async def work(user_input):
    outputq = asyncio.Queue()
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                     openai_organization=os.getenv("OPENAI_API_ORG"),
                     temperature=0,
                     model_name="gpt-4")
    runner = ActionRunner(outputq, llm=llm)
    task = asyncio.create_task(runner.run(user_input, outputq))

    while True:
        output = await outputq.get()
        if isinstance(output, Exception):
            print(output)
            return
        try:
            parsed = json.loads(output)
            print(json.dumps(parsed, indent=2))
            print("-----------------------------------------------------------")
            if parsed["action"] == "Finish":
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

def main(q):
    return asyncio.run(work(q))

if __name__ == "__main__":
    for i, q in Q:
        if i == 2:
            main(q)
