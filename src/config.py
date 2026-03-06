from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with type validation.
    All values can be overridden via environment variables or .env file.
    """

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval settings
    top_k_results: int = 3

    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.1

    # Path settings
    vector_store_path: str = "data/vector_store"
    collection_name: str = "faq_docs"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()