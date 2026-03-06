import chromadb
from typing import List, Dict, Optional
from src.embedding import EmbeddingManager
from src.config import settings


# Only one instance of EmbeddingManager is created and reused
_embedding_manager: Optional[EmbeddingManager] = None

def _get_embedder() -> EmbeddingManager:
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(settings.embedding_model)
    return _embedding_manager

# ChromaDB search
def search_similar_documents(
    query: str,
    collection_name: str = "faq_docs",
    top_k: int = settings.TOP_K_RESULTS,
    persist_directory: str = "data/vector_store",
) -> List[Dict]:
    """
    Retrieve top-k similar documents from ChromaDB using embeddings.
    Returns a list of dictionaries with content, metadata, and similarity score.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # Connect to persistent ChromaDB client
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(name=collection_name)

    # Generate embedding for query
    embedder = _get_embedder()
    query_embedding = embedder.generate_embeddings([query])[0].tolist()

    # Perform vector search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Format results with similarity scores
    docs: List[Dict] = []
    for content, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        docs.append(
            {
                "content": content,
                "metadata": metadata,
                "similarity_score": round(1 - distance / 2, 4),
            }
        )

    return docs



# Retriever
class ChromaRetriever:

    def __init__(self, collection_name: str = "faq_docs", persist_directory: str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k relevant documents for a query.
        Uses search_similar_documents internally.
        """
        return search_similar_documents(
            query, self.collection_name, top_k, self.persist_directory
        )















    
    
    
    














