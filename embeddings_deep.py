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

if __name__ == "__main__":
    basic_embeddings()
