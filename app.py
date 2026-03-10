import bleach
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from src.retriever import ChromaRetriever
from src.rag_pipeline import create_rag_chain
from src.guardrails import deterministic_check, inline_model_guardrail
from src.logger import setup_logger
from src.config import settings
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
from fastapi_cache.backends.redis import RedisBackend
from redis.asyncio import Redis
from hashlib import sha256
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from fastapi import Request

logger = setup_logger("app")


def question_key_builder(func, namespace, request, *args, **kwargs):
    query_obj = kwargs.get("query")  
    if not query_obj:
        return f"{namespace}:no-question"
    question = query_obj.question.strip().lower()  
    return f"{namespace}:{sha256(question.encode()).hexdigest()}"



@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = Redis(host="localhost", port=6379, decode_responses=True)
    pong = await redis.ping()
    print("Redis PING:", pong)  
    FastAPICache.init(RedisBackend(redis), prefix="rag-cache")
    yield
    await redis.close()


# FastAPI App
app = FastAPI(
    title="RAG API",
    version="1.0.0",
    description="RAG based question answering API",
    lifespan=lifespan
)


limiter = Limiter(key_func=get_remote_address)

app.state.limiter = limiter

app.add_middleware(SlowAPIMiddleware)

limiter = Limiter(key_func=get_remote_address)

# CORS Security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



# Request Schema
class Query(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
        description="User question"
    )

# Initialize LLM and Retriever
llm = ChatOllama(
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    num_predict=200
)

retriever = ChromaRetriever(
    collection_name=settings.collection_name,
    persist_directory=settings.vector_store_path
)

rag_chain = create_rag_chain(retriever, llm)


# Health Check
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
@cache(expire=300, key_builder=question_key_builder)  # cache for 5 minutes
@limiter.limit("10/minute")
async def ask(request: Request, query: Query):
    try:
        logger.info(f"Received question: {query.question[:50]}")

        # XSS protection
        clean_question = bleach.clean(query.question)

        # deterministic guardrail
        deterministic_check(clean_question)

        # run RAG pipeline
        answer = rag_chain(clean_question)

        # LLM guardrail
        #safe_answer = inline_model_guardrail(answer, llm)

        #logger.info("Answer generated successfully")

        return {"answer": answer}

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        logger.error(f"Guardrail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )