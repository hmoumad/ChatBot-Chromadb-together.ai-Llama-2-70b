from langchain_community.llms import Replicate
from decouple import config
import os

# Load environment variables
REPLICATE_API_TOKEN = config("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

llmGemma = Replicate(
    model = "google-deepmind/gemma-7b-it:2790a695e5dcae15506138cc4718d1106d0d475e6dca4b1d43f42414647993d5",
    input={ 
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.7,
        "max_new_tokens": 512,
        "min_new_tokens": -1,
        "repetition_penalty": 1
    }
)

prompt: str = """
Human: Answer the following question and show your reasoning to the final answer: who is the king of Morocco ??
AI
"""

response: str = llmGemma(prompt=prompt)

print(response)