# RAG (Retrieval-Augmented Generation) application using Streamlit and LangChain.
# This app loads content from a website, stores it as embeddings in a vector database,
# retrieves the most relevant text chunks for a user question, and then asks an LLM
# to answer based on that retrieved context.

import streamlit as st
import time

# LangChain/OpenAI imports
from langchain_openai import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from a .env file, such as OPENAI_API_KEY
from dotenv import load_dotenv
load_dotenv()


# --------------------
# 1. App UI setup
# --------------------
# Display the title shown at the top of the Streamlit web app.
st.title("RAG Application")

# --------------------
# 2. Define the source webpage(s)
# --------------------
# This is the URL that will be scraped and used as the knowledge base.
# Replace it with your actual website or documentation page.
urls = ['https://example.com']  # Replace with your actual URLs

# --------------------
# 3. Load data from the URL
# --------------------
# UnstructuredURLLoader fetches content from the URL and returns one or more documents.
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

# --------------------
# 4. Split content into chunks
# --------------------
# Large text is broken into smaller pieces so the retriever can find the most relevant text.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

# Store the split chunks in a variable.
all_splits = docs

# --------------------
# 5. Create embeddings and vector store
# --------------------
# OpenAIEmbeddings converts each chunk into a numerical vector.
# Chroma stores those vectors so we can retrieve semantically similar text later.
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# --------------------
# 6. Create the retriever
# --------------------
# This searches the vector database for the most relevant text chunks.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# --------------------
# 7. Create the LLM
# --------------------
# OpenAI is the language model that will answer the user's question.
llm = OpenAI(temperature=0.4, max_tokens=500)

# --------------------
# 8. User input
# --------------------
# Chat input box for the user to ask a question.
query = st.chat_input("Say something: ")

# --------------------
# 9. Build the prompt
# --------------------
# The system prompt gives the model instructions about how to answer.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# --------------------
# 10. Run the RAG pipeline
# --------------------
# If the user has entered a question, retrieve the most relevant context and generate an answer.
if query:
    # Combine the prompt with the retrieved documents.
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Build the full retrieval + generation chain.
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Run the chain with the user's input.
    response = rag_chain.invoke({"input": query})

    # Display the answer in the app.
    st.write(response["answer"])