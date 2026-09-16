from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

import os
print("Key found:", os.getenv("GOOGLE_API_KEY") is not None)
print("Key starts with:", os.getenv("GOOGLE_API_KEY")[:5] if os.getenv("GOOGLE_API_KEY") else "N/A")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def basic_embeddings():

    #single text
    text = "What is F1"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"Vector values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")

def batch_embeddings():
    text = [
        "What is F1",
        "Who won the 2023 Monaco Grand Prix?",
        "What is the Drag Reduction System (DRS)?"
    ]

    batch_embeddings = embeddings.embed_documents(text)
    for i, emb in enumerate(batch_embeddings):
        print(f"Vector dimensions: {len(emb)}")
        print(f"Vector values: {emb[:5]}")
        print(f"Vector norm: {np.linalg.norm(emb):.4f}")

def similarity_search():

    # Example document embeddings
    docs = [
        "Max Verstappen won the 2023 Monaco Grand Prix.",
        "Fernando Alonso finished second in the 2023 Monaco GP.",
        "Esteban Ocon completed the podium in third place."
    ]

    query = "What was the result of the 2023 Monaco Grand Prix?"

    # embed documents and query
    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)

    #compute cosine similarity
    def cosine_similarity(vec_a, vec_b):
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked documents by similarity:")
    for doc, score in ranked_docs:
        print(f" {score:.4f}; {doc}")

if __name__ == "__main__":
    similarity_search()
