import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_faiss_index():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_path = os.path.join(base_dir, "data", "policies.txt")
    index_path = os.path.join(base_dir, "data", "faiss_index_2")
    
    # Load documents
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print(f"Loading documents from {data_path}...")
    loader = TextLoader(data_path)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")
    
    # Create embeddings and vector store
    print("Creating embeddings using Gemini...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    print("Building FAISS index...")
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    vectorstore = FAISS.from_texts([texts[0]], embeddings, metadatas=[metadatas[0]])
    for i in range(1, len(texts)):
        vectorstore.add_texts([texts[i]], metadatas=[metadatas[i]])
    
    # Save the vector store locally
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    vectorstore.save_local(index_path)
    print(f"FAISS index saved to {index_path}.")

if __name__ == "__main__":
    build_faiss_index()
