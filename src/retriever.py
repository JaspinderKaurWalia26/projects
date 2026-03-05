import chromadb
from typing import List, Dict, Optional

from src.embedding import EmbeddingManager
from src.config import settings
from langchain_ollama import ChatOllama


# SYSTEM_PROMPT: ensures LLM only answers based on provided context
SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer the question ONLY using the provided context.
If the answer is not in the context, say "I don't know."
"""

# BANNED_KEYWORDS: used for deterministic guardrail to block unsafe queries
BANNED_KEYWORDS = ["hack", "exploit", "malware"]



# Only one instance of EmbeddingManager is created and reused
_embedding_manager: Optional[EmbeddingManager] = None

def _get_embedder() -> EmbeddingManager:
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager(settings.embedding_model)
    return _embedding_manager



# Deterministic guardrail
def deterministic_check(query: str) -> None:
    """
    Check query for banned keywords.
    Raises ValueError if any banned keyword is found.
    This prevents unsafe queries before they reach the model.
    """
    query_lower = query.lower()
    for kw in BANNED_KEYWORDS:
        if kw in query_lower:
            raise ValueError(
                "Query contains inappropriate content. Please rephrase."
            )


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



# Inline model-based guardrail
def inline_model_guardrail(answer: str, llm: ChatOllama) -> str:
    """
    Evaluate the final LLM-generated answer for safety using the same LLM.
    - Checks for PII, confidential info, or inappropriate content.
    - If unsafe, returns a safe fallback message.
    """
    safety_prompt = f"""
Evaluate the following response for safety:
- Does it leak confidential info?
- Does it contain PII or inappropriate content?
Reply SAFE or UNSAFE and explain briefly.

Answer:
{answer}
"""
    result = llm.invoke(safety_prompt)

    # If model marks as UNSAFE, replace answer
    if "UNSAFE" in result.content.upper():
        return "I cannot share this information."

    # Otherwise return original answer
    return answer



# RAG with system prompt and guardrails
def rag_with_system_prompt(
    query: str, retriever: ChromaRetriever, llm: ChatOllama, top_k: int = 3
) -> str:
    """
    Full RAG pipeline with:
    1. Deterministic guardrail (banned keyword check)
    2. Retrieval of top-k relevant documents
    3. LLM answer generation with system prompt
    4. Inline model-based guardrail for safety

    Returns a safe answer string.
    """
    # Deterministic guardrail
    deterministic_check(query)

    # Retrieve context from ChromaDB
    docs = retriever.retrieve(query, top_k)
    if not docs:
        return "No relevant context found."

    # Prepare RAG prompt with context
    context = "\n\n".join(d["content"] for d in docs)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    # Generate answer from LLM
    response = llm.invoke(prompt)

    #Inline model-based guardrail
    safe_answer = inline_model_guardrail(response.content, llm)
    return safe_answer



# RAG chain 
def create_rag_chain(retriever: ChromaRetriever, llm: ChatOllama, top_k: int = 3):
    """
    Return a reusable RAG chain function.
    The returned function takes a question and returns a safe answer.
    """
    def rag_chain(question: str, k: int = top_k) -> str:
        return rag_with_system_prompt(question, retriever, llm, top_k=k)
    return rag_chain



# example usage

if __name__ == "__main__":
    # Initialize LLM (Mistral) and Retriever
    llm = ChatOllama(model="mistral", temperature=0.1)
    retriever = ChromaRetriever()

    # Create reusable RAG chain
    rag_chain = create_rag_chain(retriever, llm)

    # Example query (could contain PII or unsafe request)
    question = "What is the company mission?"
    answer = rag_chain(question)

    # Print the safe answer
    print(f"\nQuestion: {question}\nAnswer: {answer}")