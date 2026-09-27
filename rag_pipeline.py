import time
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
EMBED_BATCH_SIZE = 90  # stays under the free tier's 100 requests/minute
EMBED_BATCH_WAIT = 61  # seconds between batches


def create_kb(documents):
    '''Create or load a persistent vector store from F1 documents.'''

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Deterministic IDs (doc_id + chunk index) make re-ingesting a race an
    # upsert instead of adding duplicate chunks.
    chunk_counts = {}
    ids = []
    for chunk in chunks:
        doc_id = chunk.metadata["doc_id"]
        n = chunk_counts.get(doc_id, 0)
        chunk_counts[doc_id] = n + 1
        ids.append(f"{doc_id}#{n}")

    # Creates the store on first run, loads it otherwise.
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings_model,
    )

    # Only embed chunks that aren't stored yet, so re-runs cost no API quota.
    existing = set(vector_store.get(ids=ids, include=[])["ids"])
    new = [(chunk, chunk_id) for chunk, chunk_id in zip(chunks, ids) if chunk_id not in existing]
    print(f"{len(existing)} chunks already stored, embedding {len(new)} new chunks...")

    # Gemini's free tier allows 100 embedding requests per minute.
    for start in range(0, len(new), EMBED_BATCH_SIZE):
        if start > 0:
            print(f"  waiting {EMBED_BATCH_WAIT}s for the embedding rate limit...")
            time.sleep(EMBED_BATCH_WAIT)
        batch = new[start:start + EMBED_BATCH_SIZE]
        vector_store.add_documents(
            [chunk for chunk, _ in batch],
            ids=[chunk_id for _, chunk_id in batch],
        )
        print(f"  embedded {start + len(batch)}/{len(new)}")

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