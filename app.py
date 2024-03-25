# Requirements:
# Before we start, make sure you have the following libraries installed:
    # pip install streamlit
    # get your token in http://hf.co/settings/tokens

import streamlit as st
import ExtractDataOCR
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitt
er
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.llms import Together
from decouple import config

# Load environment variables
together_api_key = config("together_api_key")
os.environ["together_api_key"] = together_api_key

llmMistral = Together(
    model="togethercomputer/llama-2-70b-chat",
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
                File_Text = ExtractDataOCR.extract_text_from_pdf(FILE_PATH)
                # st.success("Uploaded successfully")

    # # Display the value of File_Text
    # if File_Text is not None:
    #     st.write("Extracted text from PDF:")
    #     st.write(File_Text)

    if File_Text is not None:
        if isinstance(File_Text, list):
            File_Text = ''.join(File_Text)  # Join the list into a single string
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(File_Text)
        # st.write(chunks)

        # Store embeddings in ChromaDB
        CHROMA_DATA_PATH = "chroma_data/"
        COLLECTION_NAME = pdf_file.name[:-4]

        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        persist_directory = os.path.join(CHROMA_DATA_PATH, COLLECTION_NAME)
        # save to disk
        st.session_state.vector_db = Chroma.from_texts(
            chunks, 
            embedding_function, 
            persist_directory=persist_directory
        )

        st.success("All id Done Good ~~!!")

    # User input
    if user_input := st.chat_input("Ask me Question about your PDF File 📖"):

        # pass the user input to vector database and applique semantic search
        semantic_search = st.session_state.vector_db.max_marginal_relevance_search(user_input, k=2)

        QA_Prompt = PromptTemplate(
            template="""Use the following pieces of context to answer to the user question
            context = {text}
            question = {question}
            Answer:""",
            input_variables=["text", "question"]
        )

        QA_Chain = RetrievalQA.from_chain_type(
            llm = llmMistral,
            retriever = st.session_state.vector_db.as_retriever(),
            return_source_documents = True,
            chain_type = "map_reduce"
        )

        question = user_input
        response = QA_Chain({"query": question})
        response_value = response["result"]
        # response_source = response["source_documents"]

        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "message": user_input})

        # Generate response from the chatbot
        st.session_state.chat_history.append({"role": "bot", "message": response_value})

        
        # # Generate source from the chatbot
        # st.session_state.chat_history.append({"role": "bot", "message": response_source})

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
