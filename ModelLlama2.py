from langchain_community.llms import Replicate
from decouple import config
import os

# Load environment variables
REPLICATE_API_TOKEN = config("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

llmLlama2 = Replicate(
    model = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.7,
        "max_new_tokens": 512,
        "min_new_tokens": -1,
        "repetition_penalty": 1
    }
)

# prompt: str = """
# Human: Answer the following question and show your reasoning to the final answer: who is the king of Morocco ??
# AI
# """

# response: str = llmLlama2(prompt=prompt)
# print(response)