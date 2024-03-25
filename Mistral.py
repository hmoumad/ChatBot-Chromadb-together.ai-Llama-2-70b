from langchain.llms import Together
from decouple import config
import os

# Load environment variables
together_api_key = config("together_api_key")
os.environ["together_api_key"] = together_api_key

llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=128,
    top_k=1,
    together_api_key=together_api_key
)

Prompt = """You are a teacher with a deep knowledge of machine learning and AI. \
You provide succinct and accurate answers. Answer the following question: 

What is a large language model?"""

response = llm(Prompt)
print(response)

