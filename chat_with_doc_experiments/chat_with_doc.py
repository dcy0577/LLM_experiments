from langchain import PromptTemplate
import openai
from langchain.llms import OpenAI
import dotenv
import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chat_models import ChatOpenAI
import textwrap
from langchain.chains.api import open_meteo_docs

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Document Loader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

# loader = TextLoader('./example.txt')
loader = PyPDFLoader('./data/VectorScriptGuide.pdf')
documents = loader.load()

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
#Create the vectorized db
# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS
db = FAISS.from_documents(docs, embeddings)

# tech_template = """use the following pieces of context to answer the question at the end. If you don't know the answer, 
# just say that you don't know, don't try to make up an answer. If you are asked to provide example, please provide at least one code snippet according to the given context.

# {context}

# Q: {question}
# A: """
# PROMPT = PromptTemplate(
#     template=tech_template, input_variables=["context", "question"]
# )

tech_template = """
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
If you are asked to provide example, please provide at least one code snippet according to the given context.

Question: {question}
=========
{summaries}
=========
Answer: """
PROMPT = PromptTemplate(
    template=tech_template, input_variables=["summaries", "question"]
)

def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)

chat_history = []
def answer_query(query, chat_history):

    chat = ChatOpenAI(temperature=0)
    # chain = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     chain_type="stuff",
    #     retriever=db.as_retriever(),
    #     chain_type_kwargs={"verbose": True, "prompt": PROMPT},
    #     return_source_documents=True
    # )
    # result = chain({"query": query})
    # return print(result["result"])

    question_generator = LLMChain(llm=chat, verbose=True, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_with_sources_chain(chat, chain_type="stuff", verbose=True, prompt=PROMPT)
    chain = ConversationalRetrievalChain(retriever=db.as_retriever(search_type="mmr",search_kwargs={"k": 3}),question_generator=question_generator, combine_docs_chain=doc_chain, verbose =True)
    result = chain({"question": query, "chat_history": chat_history})
    return result
while True:
    
    user_input = input("Ask a question: ")
    result = answer_query(user_input, chat_history)
    print(result["answer"])
    chat_history.append((user_input, result["answer"]))
    print(chat_history)



