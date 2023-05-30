from langchain import PromptTemplate
import openai
import dotenv
import os
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chat_models import ChatOpenAI
import textwrap
from langchain.chains.api import open_meteo_docs

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Document Loader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

# loader = TextLoader('./example.txt')
loader = PyPDFDirectoryLoader('./data')
documents = loader.load()

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
#Create the vectorized db
# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS
db = FAISS.from_documents(docs, embeddings)

tech_template = """use the following pieces of context to answer the question at the end. If you don't know the answer, 
just say that you don't know, don't try to make up an answer. If you are asked to provide example, please provide at least one code snippet according to the given context.

{context}

Question: {question}
Answer: """
PROMPT = PromptTemplate(
    template=tech_template, input_variables=["context", "question"]
)


def answer_query(query):

    chat = ChatOpenAI(temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"verbose": True, "prompt": PROMPT},
        return_source_documents=True
    )
    result = chain({"query": query})
    return print(result["result"])

while True:
    # generate 10 question and answer sets about dynamic arrays in vectorscript
    user_input = input("Ask a question: ")
    answer_query(user_input)




