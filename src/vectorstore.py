import os
import uuid
import numpy as np
from typing import List, Any
import chromadb

# Handles storing document embeddings in ChromaDB
class VectorStore:
    
    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "data/vector_store"
    ):
        # Store collection details
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # ChromaDB client and collection
        self.client = None
        self.collection = None
        
        # Initialize vector store
        self._initialize_store()

    def _initialize_store(self):
        try:
            # Ensure persistence directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Create persistent ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for RAG"}
            )
            
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        # Validate input sizes
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        # Prepare data for ChromaDB
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique document ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Store document content and embedding
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
        
        # Add data to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text
        )
        
        print(f"Added {len(documents)} documents. Total now: {self.collection.count()}")