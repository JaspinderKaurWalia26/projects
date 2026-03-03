from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from src.config import settings

def chunk_documents(documents: List[Dict]) -> List[Dict]:
    # Print total number of documents
    print(f"Starting to chunk {len(documents)} documents...")
    
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    all_chunks = []
    
    # Process each document
    for doc_idx, doc in enumerate(documents):
        content = doc["content"]
        metadata = doc["metadata"]
        
        # Split document into chunks
        chunks = splitter.split_text(content)
        print(f"Document {doc_idx + 1}/{len(documents)}: {len(chunks)} chunks created")
        
        # Store valid chunks with metadata
        for idx, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 15:
                continue
            
            all_chunks.append({
                "content": chunk_text.strip(),
                "metadata": {
                    **metadata,
                    "chunk_number": idx,
                    "chunk_total": len(chunks)
                }
            })
    
    # Print final chunk count
    print(f"Chunking completed. Total chunks created: {len(all_chunks)}")
    
    return all_chunks