import argparse
import os
import shutil
import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        logging.info("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    chunks_with_ids = calculate_chunk_ids(chunks)
    add_to_chroma(chunks_with_ids)

def load_documents():
    logging.info("Loading documents...")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    logging.info(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents: list[Document]):
    logging.info("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(chunks)} chunks")
    return chunks

def add_to_chroma(chunks: list[Document]):
    logging.info("Adding documents to Chroma...")
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    logging.info(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    logging.info(f"New chunks to add: {len(new_chunks)}")

    if len(new_chunks):
        logging.info(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        
        try:
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
            logging.info("✅ Successfully added new documents")
        except Exception as e:
            logging.error(f"Error adding documents to Chroma: {e}")
    else:
        logging.info("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    logging.info("Calculating chunk IDs...")
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    logging.info(f"Calculated chunk IDs for {len(chunks)} chunks")
    return chunks

def clear_database():
    logging.info("Clearing database...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    logging.info("Database cleared")

if __name__ == "__main__":
    main()
