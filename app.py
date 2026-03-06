from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from src.retriever import ChromaRetriever
from src.rag_pipeline import create_rag_chain
from src.guardrails import deterministic_check, inline_model_guardrail
from src.logger import setup_logger
from src.config import settings

logger = setup_logger("app")

app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="RAG based question answering API"
)


# Request body structure
class Query(BaseModel):
    question: str


# Initialize once
llm = ChatOllama(model=settings.llm_model, temperature=settings.llm_temperature)

retriever = ChromaRetriever(
    collection_name=settings.collection_name,
    persist_directory=settings.vector_store_path
)
rag_chain = create_rag_chain(retriever, llm)


@app.get("/health")
def health_check():
    """Check if API is running."""
    return {"status": "ok"}


@app.post("/ask")
def ask(query: Query):
    """
    Answer a question using RAG pipeline.
    Args:
        query: Question from user
    Returns:
        Answer from RAG pipeline
    """
    try:
        # Deterministic guardrail check
        logger.info(f"Received question: {query.question[:50]}...")
        deterministic_check(query.question)

        # Get answer from RAG pipeline
        answer = rag_chain(query.question)

        # Inline model guardrail check
        safe_answer = inline_model_guardrail(answer, llm)

        logger.info("Answer generated successfully")
        return {"answer": safe_answer}

    except ValueError as e:
        # Guardrail or validation error
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        # Guardrail LLM error
        logger.error(f"Guardrail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")