import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Store model name
        self.model_name = model_name
        
        # Placeholder for the loaded model
        self.model = None
        
        # Load the embedding model
        self._load_model()
    
    def _load_model(self):
        # Load SentenceTransformer model
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Print embedding dimension
        print(f"Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        # Ensure model is loaded
        if not self.model:
            raise ValueError("Model not loaded")
        
        # Generate embeddings for input texts
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        return embeddings