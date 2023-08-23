import os
import asyncio

from pprint import pprint
from ast import literal_eval
from multiprocessing import Pool, TimeoutError

from autoagents.agents.search import ActionRunner
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI


async def work(user_input):
    outputq = asyncio.Queue()
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),
                     openai_organization=os.getenv("OPENAI_API_ORG"),
                     temperature=0,
                     model_name="gpt-3.5-turbo")
    runner = ActionRunner(outputq, llm=llm)
    task = asyncio.create_task(runner.run(user_input, outputq))

    while True:
        output = await outputq.get()
        if isinstance(output, Exception):
            print(output)
            return
        try:
            pprint(literal_eval(output))
        except:
            print(output)
        print("-----------------------------------------------------------")
        if "Final Answer:" in output:
            break
    await task

Q = [
     "list 5 cities and their current populations where Paramore is playing this year.",
     "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?",
     "How many watermelons can fit in a Tesla Model S?",
     "Recommend me some laptops suitable for UI designers under $2000. Please include brand and price."
     "Build me a vacation plan for Rome and Milan this summer for seven days. Include place to visit and hotels to stay. ",
     "What is the sum of ages of the wives of Barack Obama and Donald Trump?",
     "Who is the most recent NBA MVP? Which team does he play for? What is his season stats?",
     "What were the scores for the last three games for the Los Angeles Lakers? Provide the dates and opposing teams.",
     "Which team won in women's volleyball in the Summer Olympics that was held in London?",
     "Provide a summary of the latest COVID-19 research paper published. Include the title, authors and abstract.",
     "What is the top grossing movie in theatres this week? Provide the movie title, director, and a brief synopsis of the movie's plot. Attach a review for this movie.",
     "Recommend a bagel shop near the Strip district in Pittsburgh that offer vegan food",
     "Who are some top researchers in the field of machine learning systems nowadays?"
     ]

def main(q):
    asyncio.run(work(q))

if __name__ == "__main__":
    with Pool(processes=10) as pool:
        print(pool.map(main, Q))
