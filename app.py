import os
from langchain_community.document_loaders import PyPDFLoader
# --- FIX IS HERE: Updated import path ---
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_vector_db(pdf_path):
    """
    Processes a PDF and creates a searchable vector database.
    """
    # 1. Extract text from the PDF
    print(f"...Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 2. Split content into semantic chunks
    print("...Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # 3. Create Vector Store
    print("...Generating embeddings and vector store")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore

if __name__ == "__main__":
    print("Backend logic ready.")