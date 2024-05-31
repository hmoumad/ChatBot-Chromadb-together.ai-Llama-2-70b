# Requirements:
# Before we start, make sure you have the following libraries installed:
    # pip install streamlit
    # get your token in http://hf.co/settings/tokens

import streamlit as st
import ExtractDataOCR
import ExtractDataPyPDF2_2

from langchain_nomic import NomicEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_together import Together
from decouple import config

# Load environment variables
together_api_key = config("together_api_key")
os.environ["together_api_key"] = together_api_key

# Load environment variables
nomic_api_key = config("nomic_api_key")
os.environ["nomic_api_key"] = nomic_api_key

llmLlama_3 = Together(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.7,
    max_tokens=128,
    top_k=1,
    together_api_key=together_api_key
)


def main():
    st.set_page_config("ChatPDF")
    st.title("Chat with your PDF")

    # Chat history global variable
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # vector_db glbal variable
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    # Define File_Text as a global variable outside of any function scope
    File_Text = None

    with st.sidebar:

        st.header("Start your communication")
        st.subheader("Upload your PDF 📖")
        
        pdf_file = st.file_uploader("upload your pdf file and start process")
        if st.button("Start Asking"):
            st.spinner("Processing")

            if pdf_file:

                # Extract file path from uploaded file object
                FILE_PATH = os.path.join(pdf_file.name)
                with open(FILE_PATH, "wb") as f:
                    f.write(pdf_file.read())
                
                # Application of OCR to transfrome PDF to images and extract texts from this images
                # File_Text = ExtractDataOCR.extract_text_from_pdf(FILE_PATH) # you can incomment this line if you want to use OCR

                # Application of OCR to transfrome PDF to images and extract texts from this images
                File_Text = ExtractDataPyPDF2_2.Extract_text_pypdf2(FILE_PATH) # you can comment this line if you uncomment OCR

    if File_Text is not None:
        if isinstance(File_Text, list):
            File_Text = ''.join(File_Text)  # Join the list into a single string
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )

        chunks = text_splitter.split_text(File_Text)
        # st.write(chunks)

        # Store embeddings in ChromaDB
        CHROMA_DATA_PATH = "chroma_data/"
        COLLECTION_NAME = pdf_file.name[:-4]

        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # create the open-source embedding function
        # embedding_function = NomicEmbeddings(model="nomic-embed-text-v1")

        # Create path of destination files
        persist_directory = os.path.join(CHROMA_DATA_PATH, COLLECTION_NAME)

        # save documents to disk thanks to Chromadb
        st.session_state.vector_db = Chroma.from_texts(
            chunks, 
            embedding_function, 
            persist_directory=persist_directory
        )

        # show alert to show termination of the process
        st.success("All id Done Good ~~!!")

    # User input "this varaiable stock the question asked by user"
    if user_input := st.chat_input("Ask me Question about your PDF File 📖"):

        # # pass the user input to vector database and applique semantic search
        # semantic_search = st.session_state.vector_db.max_marginal_relevance_search(user_input, k=2)

        QA_Chain = RetrievalQA.from_chain_type(
            llm = llmLlama_3,
            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents = True,
            chain_type = "map_reduce"
        )

        # Create Prompte to passe the question withe more context to the model
        QA_Prompt = PromptTemplate(
            template=""" 
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            As an expert of Question answering for all resharches articles,\
            <|eot_id|>

            <|start_header_id|>user<|end_header_id|>
            Use the following pieces of context and your backgruond Knowledge to answer to the user question
            question = {question}

            Please use at maxe 50 words in your answer 
            <|eot_id|>

            <|start_header_id|>assistant<|end_header_id|>
            Answer: """,
            
            input_variables=["question"]
        )

        question_prompt = QA_Prompt.format(question=user_input)

        response = QA_Chain({"query": question_prompt + "\n"})

        response_value = response["result"]

         # Function to extract text until ".assistant"
        def extract_text_until_assistant(text):
            return text.split('.assistant')[0]
        
        # Loop through the list and apply the function
        final_answers = extract_text_until_assistant(response_value)
        response_source = response["source_documents"]
        print("==============================la reponse sans la formater ======================================")
        print(response_value)
        print("==============================la source de la reponse ======================================")
        print(response_source)

        # Add user mess age to chat history
        st.session_state.chat_history.append({"role": "user", "message": user_input})

        # Generate response from the chatbot
        st.session_state.chat_history.append({"role": "bot", "message": final_answers})

    # Display chat history
    for item in st.session_state.chat_history:

        if item["role"] == "user":
            # st.write("You: ", item["message"])
            with st.chat_message("user"):
                st.markdown(item["message"])

        else:
            # st.write("Bot: ", item["message"])
            with st.chat_message("assistant"):
                st.write(item["message"])

if __name__ == "__main__":
    main()