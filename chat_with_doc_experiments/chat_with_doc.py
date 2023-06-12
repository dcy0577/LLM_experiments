from langchain import PromptTemplate
import openai
from langchain.llms import OpenAI
import dotenv
import os
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chat_models import ChatOpenAI
import textwrap
from langchain.chains.api import open_meteo_docs
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, MathpixPDFLoader, PyPDFDirectoryLoader, UnstructuredMarkdownLoader, UnstructuredHTMLLoader
from custom_text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language, SpacyTextSplitter, NLTKTextSplitter, MarkdownTextSplitter

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

persist_directory = './db'
with os.scandir(persist_directory) as it:
    # load vector db if it exists
    if any(it):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        documents = []
        for file in os.listdir('./data_markdown'):
            if file.endswith('.md'):
                md_path = os.path.join('./data_markdown', file)
                loader = UnstructuredMarkdownLoader(md_path)
                documents.extend(loader.load())
            # elif file.endswith('.pdf'):
            #     pdf_path = os.path.join('./data', file)
            #     loader = PyPDFLoader(pdf_path)
            #     documents.extend(loader.load())
        # Document Loader
        # loader = UnstructuredPDFLoader('./data/sub_data/VWKeyboardShortcuts.pdf')
        # loader = UnstructuredPDFLoader('./data/VectorworksUsersGuide_2023_Sp4.pdf')


        # Text Splitter
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=2000, chunk_overlap=0)
        # # text_splitter = NLTKTextSplitter(chunk_size=1000)
        # # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=20)
        # docs = text_splitter.split_documents(documents)

        # no splitting
        docs = [doc for doc in documents]
        
        # Create the vectorized db
        # Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
        db = Chroma.from_documents(documents=docs, 
                                   embedding=embeddings, 
                                   persist_directory=persist_directory)
        db.persist()

# tech_template = """use the following pieces of context to answer the question at the end. If you don't know the answer, 
# just say that you don't know, don't try to make up an answer. If you are asked to provide example, please provide at least one code snippet according to the given context.

# {context}

# Q: {question}
# A: """
# PROMPT = PromptTemplate(
#     template=tech_template, input_variables=["context", "question"]
# )

tech_template = """
Given the following extracted parts of a markdown format documentation and a question, create a final answer with references ("SOURCES"). 
ALWAYS return a "SOURCES" part in your answer.
If you don't find the answer to the user's question with the contents provided to you, answer that you didn't find the answer in the contents and propose him to rephrase his query with more details.
If you are asked to provide code example, please provide at least one code snippet according to the given document.
If needed, provide your answer in bullet points.
If asked an irrelevant question, you will gently guide the conversation back to the topic of the documentation of vectorworks.
The content are given in markdown format. You should use markdown syntax to understand the content.

Question: {question}
=========
{context}
=========
Answer: """
NEW_PROMPT = PromptTemplate(
    template=tech_template, input_variables=["context", "question"]
)

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
    doc_chain = load_qa_with_sources_chain(chat, chain_type="stuff", 
                                           verbose=True, 
                                           prompt=NEW_PROMPT,
                                           document_variable_name = "context")
    chain = ConversationalRetrievalChain(retriever=db.as_retriever(search_kwargs={"k": 2}), 
                                         question_generator=question_generator, 
                                         combine_docs_chain=doc_chain, 
                                         verbose =True)
    result = chain({"question": query, "chat_history": chat_history})
    return result

while True:
    
    user_input = input("Ask a question: ")
    result = answer_query(user_input, chat_history)
    print(result["answer"])
    chat_history.append((user_input, result["answer"]))
    # only keep the last 5 history
    if len(chat_history) > 5:
        chat_history.pop(0)
    # print(chat_history)



