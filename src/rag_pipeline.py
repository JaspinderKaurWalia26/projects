from langchain_ollama import ChatOllama
from src.retriever import ChromaRetriever
from src.logger import setup_logger

logger = setup_logger(__name__)

# SYSTEM_PROMPT: ensures LLM only answers based on provided context
SYSTEM_PROMPT = """
You are a helpful AI assistant that answers questions ONLY based on the provided context documents.

STRICT RULES:
1. Answer ONLY using information from the provided context below.
2. Give answers directly without adding phrases like "according to the documents".
3. If the answer is not in the context, say exactly: "I don't know. This information is not available in the provided documents."
4. Do NOT make up information or use your general knowledge.
5. Do NOT hallucinate facts, numbers, or details not present in context.
6. Keep answers concise and accurate.
7. If partial information is available, provide it and mention what's missing.
8. Always be helpful but honest about limitations.
9. If the question asks about goals, purpose, rules, policies, or procedures, respond ONLY if it is explicitly mentioned in the context.
10. For questions about the AI itself (like its mission or purpose), respond:
    "I am an AI model and do not have personal goals. "

Better to say "I don't know" than provide incorrect information.
"""


def rag_with_system_prompt(
    query: str,
    retriever: ChromaRetriever,
    llm: ChatOllama,
    top_k: int = 3
) -> str:
    """
    Core RAG pipeline - retrieval + LLM answer generation.
    Args:
        query: User question
        retriever: ChromaRetriever instance
        llm: ChatOllama instance
        top_k: Number of documents to retrieve
    Returns:
        Answer string from LLM
    Raises:
        RuntimeError: If LLM call fails
    """

    # Retrieve context from ChromaDB
    docs = retriever.retrieve(query, top_k)

    if not docs:
        logger.warning("No relevant context found")
        return "No relevant context found."

    # Prepare RAG prompt with context
    context = "\n\n".join(d["content"] for d in docs)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    try:
        # Generate answer from LLM
        logger.info("Generating answer from LLM...")
        response = llm.invoke(prompt)
        logger.info("Answer generated successfully")
        return response.content

    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise RuntimeError(f"LLM call failed: {e}")


def create_rag_chain(retriever: ChromaRetriever, llm: ChatOllama, top_k: int = 3):
    """
    Return a reusable RAG chain function.
    Args:
        retriever: ChromaRetriever instance
        llm: ChatOllama instance
        top_k: Number of documents to retrieve
    Returns:
        RAG chain function
    """
    logger.info("Creating RAG chain...")

    def rag_chain(question: str, k: int = top_k) -> str:
        return rag_with_system_prompt(question, retriever, llm, top_k=k)

    return rag_chain
