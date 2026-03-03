from langchain_community.document_loaders import DirectoryLoader, PDFMinerLoader, TextLoader, BSHTMLLoader
from typing import List

def load_faq_documents(data_folder: str = "data/") -> List:
    # TXT files
    txt_loader = DirectoryLoader(
        data_folder, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}
    )
    txt_documents = txt_loader.load()
    
    # PDF files
    pdf_loader = DirectoryLoader(data_folder, glob="**/*.pdf", loader_cls=PDFMinerLoader)
    pdf_documents = pdf_loader.load()
    
    # HTML files
    html_loader = DirectoryLoader(data_folder, glob="**/*.html", loader_cls=BSHTMLLoader)
    html_documents = html_loader.load()
    
    all_docs = txt_documents + pdf_documents + html_documents
    print(f"Total FAQ documents loaded: {len(all_docs)}")
    return all_docs