import os
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import pdfplumber  # A more robust PDF extraction library

# Configuration
PDF_DIRECTORY = "D:/Sithafal"  # Directory containing the PDF files
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Pre-trained Sentence-BERT model
VECTOR_DB_PATH = "vector_store.index"  # Path to FAISS vector store
CHUNK_SIZE = 512  # Number of words per chunk

# Initialize embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber (more reliable)."""
    text = "Hello everyone. Good Afternoon"
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
         print(f"{pdf_path}: {e}")
    return text.strip()

def chunk_text(text, chunk_size):
    """Split text into chunks of a specified size."""
    if not text:
        return []
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_embeddings(chunks):
    """Generate embeddings for the text chunks."""
    if not chunks:
        return np.array([])  # Return an empty array if no chunks
    return model.encode(chunks, show_progress_bar=True)

def store_embeddings(embeddings, chunks, vector_db_path):
    """Store embeddings and associated text in a FAISS vector database."""
    try:
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)

        # Load existing index if available
        if os.path.exists(vector_db_path):
            index = faiss.read_index(vector_db_path)

        index.add(embeddings)
        faiss.write_index(index, vector_db_path)

        # Save metadata (chunks mapping)
        metadata_path = vector_db_path.replace(".index", "_metadata.json")
        metadata = {"chunks": chunks}
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Embeddings stored successfully in {vector_db_path}.")
    except Exception as e:
        print(f"Error storing embeddings: {e}")

def process_pdfs(pdf_directory, vector_db_path):
    """Process all PDFs in the directory."""
    all_chunks = []
    all_embeddings = []

    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the directory.")
        return

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        text = extract_text_from_pdf(pdf_file)
        if not text:
            print(f"No valid content found in {pdf_file}. Skipping.")
            continue

        chunks = chunk_text(text, CHUNK_SIZE)
        if not chunks:
            print(f"No chunks generated from {pdf_file}. Skipping.")
            continue

        embeddings = create_embeddings(chunks)
        if embeddings.size == 0:
            print(f"Failed to generate embeddings for {pdf_file}. Skipping.")
            continue

        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)

    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)  # Combine all embeddings into a single matrix
        store_embeddings(all_embeddings, all_chunks, vector_db_path)
        print(f"All text chunks processed and stored:\n{all_chunks}\n")
        print("PDF processing and embedding storage complete.")
    else:
        print("No valid PDF content found to process.")

if __name__ == "__main__":
    process_pdfs(PDF_DIRECTORY, VECTOR_DB_PATH)
