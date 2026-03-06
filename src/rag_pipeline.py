from langchain_ollama import ChatOllama
from src.guardrails import deterministic_check, inline_model_guardrail
from src.retriever import ChromaRetriever

# SYSTEM_PROMPT: ensures LLM only answers based on provided context
SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer the question ONLY using the provided context.
If the answer is not in the context, say "I don't know."
"""

# RAG with system prompt 
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
