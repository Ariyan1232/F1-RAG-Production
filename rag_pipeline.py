from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

load_dotenv()
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def create_kb():
    '''Create a vector store from knowledge base documents.'''

    #split the knowledge base documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE, 
                   metadata={"source": "langchain_knowledge_base.md"})

    chunks = splitter.split_documents([doc])

    #create a vector store from the chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store

def demo_basic_rag():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = init_chat_model(model="gpt-4o", temperature=0.2)

    #RAG Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the context below. 
If the answer is not contained within the context, say "I don't know".

{context}

Question: {question}


Answer:

Make sure to answer in a concise manner, and if you don't know just say "I don't know"."""
    )

    #Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    #Rag chain 
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Basic RAG Demo\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

if __name__ == "__main__":
    demo_basic_rag()