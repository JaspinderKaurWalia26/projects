from .loaders import load_faq_documents
from .chunker import chunk_documents
from .embedding import EmbeddingManager
from .vectorstore import VectorStore
from langchain_core.documents import Document
  
# Load documents
faq_docs = load_faq_documents("data/")
# Prepare for chunking
docs_for_chunking = [{"content": doc.page_content, "metadata": doc.metadata} for doc in faq_docs]
# Chunk documents
chunks = chunk_documents(docs_for_chunking)
# Convert chunks to Document objects
chunks_docs = [Document(page_content=chunk["content"], metadata=chunk["metadata"]) for chunk in chunks]
# Generate embeddings
embedding_manager = EmbeddingManager("all-MiniLM-L6-v2")
texts = [doc.page_content for doc in chunks_docs]
embeddings = embedding_manager.generate_embeddings(texts)
# Add to VectorStore
vectorstore = VectorStore(collection_name="faq_docs", persist_directory="data/vector_store")
vectorstore.add_documents(chunks_docs, embeddings)
