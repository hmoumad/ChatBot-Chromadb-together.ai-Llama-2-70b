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


TEXT = ["Python is a versatile and widely used programming language known for its clean and readable syntax, which relies on indentation for code structure",
        "It is a general-purpose language suitable for web development, data analysis, AI, machine learning, and automation. Python offers an extensive standard library with modules covering a broad range of tasks, making it efficient for developers.",
        "It is cross-platform, running on Windows, macOS, Linux, and more, allowing for broad application compatibility."
        "Python has a large and active community that develops libraries, provides documentation, and offers support to newcomers.",
        "It has particularly gained popularity in data science and machine learning due to its ease of use and the availability of powerful libraries and frameworks."]

meta_data = [{"source": "document 1", "page": 1},
             {"source": "document 2", "page": 2},
             {"source": "document 3", "page": 3},
             {"source": "document 4", "page": 4}]


embedding_func = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma.from_texts(
    texts=TEXT,
    embedding=embedding_func,
    metadatas=meta_data
)

QA_Prompt = PromptTemplate(
    template="""Use the following pieces of context to answer to the user question
    context = {text}
    question = {question}
    Answer:""",
    input_variables=["text", "question"]
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