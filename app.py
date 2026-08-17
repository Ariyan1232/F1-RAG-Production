import chromadb
chroma_client = chromadb.Client()

collection_name = "test_collection"

collection = chroma_client.create_collection(name="my_collection")


documents=[
    {"id": "doc1", "text": "This is the first document."},
    {"id": "doc2", "text": "This is the second document."},
    {"id": "doc3", "text": "This is the third document."}
]

for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=doc["text"])

query_text = "first document"

results = collection.query(
    query_texts=[query_text],
    n_results=3
)

print(results)