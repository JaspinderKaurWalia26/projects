import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# Load same embedding model
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def search_similar_documents(
    query: str,
    collection_name: str,
    top_k: int = 5,
    persist_directory: str = "data/vector_store"
):
    # Create Chroma client
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name)

    # Convert query to embedding
    query_embedding = _embedder.encode([query])[0].tolist()

    # Search in vector DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity_score": 1 - results["distances"][0][i]
        })

    return docs




if __name__ == "__main__":
    results = search_similar_documents(
        query="What is the admission process?",
        collection_name="faq_docs",
        top_k=3
    )

    print("\nSearch Results:\n" + "-"*40)

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Similarity:", round(doc["similarity_score"], 3))
        print("Source:", doc["metadata"].get("source"))
        print("Content:", doc["content"][:200], "...")