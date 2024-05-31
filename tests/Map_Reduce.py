from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.llms import Together
from decouple import config
import os

# Load environment variables
together_api_key = config("together_api_key")
os.environ["together_api_key"] = together_api_key

llm = Together(
    model="togethercomputer/llama-2-70b-chat",
    temperature=0.7,
    max_tokens=128,
    top_k=1,
    together_api_key=together_api_key
)


question = "Who are you ?"
answer = "i'm a human and i i'm 23 years old."


embedding_func = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma.from_texts(
    question=question,
    embedding=embedding_func,
    answer=answer
)

QA_Prompt = PromptTemplate(
    template="""You are an expert in generating incorrect answers. Please provide two other incorrect responses generate just 2 that are plausible but not correct to the following question:\
        
    Question: '{question}' 
    \nCorrect Answer: '{answer}' 

    Generate two incorrect responses to the question '{question}' 
    based on the correct answer '{answer}' 
    In order to make a MCQ generate big differents answers that are not correct

    Answer:""",
    input_variables=["question", "answer"]
)

QA_Chain = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = vector_db.as_retriever(),
    return_source_documents = True,
    chain_type = "map_reduce"
)

question = "what areas is Python Mostly used ??"
response = QA_Chain({"query": question})

print("===================Response==========================")
print(response["result"])

print("===================Source==========================")
print(response["source_documents"])