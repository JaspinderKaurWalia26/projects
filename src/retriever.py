import chromadb
from typing import List, Dict, Optional
from src.embedding import EmbeddingManager
from src.logger import setup_logger
from src.config import settings

logger = setup_logger(__name__)

# Only one instance of EmbeddingManager is created and reused
_embedding_manager: Optional[EmbeddingManager] = None


def _get_embedder() -> EmbeddingManager:
    """
    Get or create a singleton EmbeddingManager instance.
    Returns:
        EmbeddingManager instance
    """
    global _embedding_manager
    if _embedding_manager is None:
        logger.info("Initializing EmbeddingManager...")
        _embedding_manager = EmbeddingManager(settings.embedding_model)
    return _embedding_manager


def search_similar_documents(
    query: str,
    collection_name: str = settings.collection_name,
    top_k: int =settings.top_k_results,
    persist_directory: str = settings.vector_store_path,
) -> List[Dict]:
    """
    Retrieve top-k similar documents from ChromaDB using embeddings.
    Args:
        query: Search query string
        collection_name: ChromaDB collection name
        top_k: Number of results to return
        persist_directory: Path to ChromaDB persistence directory
    Returns:
        List of dicts with content, metadata, and similarity score
    Raises:
        ValueError: If query is empty
        RuntimeError: If collection not found
    """

    # Check if query is empty
    if not query.strip():
        logger.warning("Empty query provided")
        raise ValueError("Query cannot be empty")

    try:
        # Connect to persistent ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)

        # Get collection - raises error if not found
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            logger.error(f"Collection not found: {collection_name}")
            raise RuntimeError(f"Collection not found: {collection_name}")

        # Generate embedding for query
        embedder = _get_embedder()
        logger.info(f"Generating embedding for query: {query[:50]}...")
        query_embedding = embedder.generate_embeddings([query])[0].tolist()

        # Perform vector search
        logger.info(f"Searching top {top_k} similar documents...")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results with similarity scores
        docs: List[Dict] = []
        for content, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            docs.append({
                "content": content,
                "metadata": metadata,
                "similarity_score": round(1 - distance / 2, 4),
            })

        logger.info(f"Found {len(docs)} similar documents")
        return docs

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise


class ChromaRetriever:
    """ChromaDB based document retriever."""

    def __init__(
        self,
        collection_name: str = "faq_docs",
        persist_directory: str = "data/vector_store"
    ):
        """
        Initialize ChromaRetriever.
        Args:
            collection_name: ChromaDB collection name
            persist_directory: Path to ChromaDB persistence directory
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        logger.info(f"ChromaRetriever initialized. Collection: {collection_name}")

    def retrieve(self, query: str, top_k: int = settings.top_k_results) -> List[Dict]:
        """
        Retrieve top-k relevant documents for a query.
        Args:
            query: Search query string
            top_k: Number of results to return
        Returns:
            List of dicts with content, metadata, and similarity score
        """
        logger.info(f"Retrieving top {top_k} documents for query: {query[:50]}...")
        return search_similar_documents(
            query,
            self.collection_name,
            top_k,
            self.persist_directory
        )