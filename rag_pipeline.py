import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


from fastf1_ingest import get_race_documents
from config import GOOGLE_API_KEY

embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

PERSIST_DIR = "./chroma_db"


def create_kb(documents):
    '''Create or load a persistent vector store from F1 documents.'''

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("Loading existing vector store and adding new documents...")
        vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings_model,
        )
        vector_store.add_documents(chunks)
    else:
        print("Creating new vector store...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIR,
        )

    return vector_store


def demo_basic_rag():

    # Real F1 data instead of the old hardcoded KNOWLEDGE_BASE
    docs = get_race_documents(2023, "Monaco")
    vector_store = create_kb(docs)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the context below.
If the answer is not contained within the context, say "I don't know".

{context}

Question: {question}

Answer:

Make sure to answer in a concise manner, and if you don't know just say "I don't know"."""
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "Who won the 2023 Monaco Grand Prix?",
        "Who finished on the podium at the 2023 Monaco GP?",
        "What team does Max Verstappen drive for?",
        "Who won the 2022 championship?",  # not in this dataset — tests "I don't know"
    ]

    print("Basic RAG Demo\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    demo_basic_rag()