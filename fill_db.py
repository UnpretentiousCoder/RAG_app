import os
from pathlib import Path
import hashlib
import pickle
import time
from uuid import uuid4
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import chromadb
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader

def generate_document_id(chunk, chunk_index):
    """
    Generates a unique and consistent ID for a document chunk.
    
    The ID is composed of:
    - Base filename (e.g., "iFASTCorp-AR2023.pdf")
    - Zero-padded chunk index (e.g., "0001")
    - A truncated hash of the chunk's content for content-based consistency.
    
    Args:
        document (Document): The LangChain Document object for the chunk. #GOING TO PREVENT DUPES BY CHECKKING THE FIRST 10 ITEMS AND COMNPARE WITH THE SOURCE NAME
        chunk_index (int): The numerical index of the chunk within its parent document/file.
                             This is crucial for uniqueness when content might be repeated.
        
    Returns:
        str: A unique identifier for the chunk.
    """
    if chunk_index is None:
        raise ValueError("chunk_index must be provided for generating chunk IDs.")

    # Get a stable filename from the source metadata
    source = chunk.metadata.get('source', 'unknown')
    source_name = os.path.basename(source) if source != 'unknown' else 'unknown'
    
    # Generate a hash of the chunk's content
    # Using MD5 for brevity, SHA256 (from your previous code) is more secure if collision resistance is paramount,
    # but for ID generation, MD5 is typically sufficient and faster.
    normalized_content = " ".join(chunk.page_content.split())
    content_hash = hashlib.md5(normalized_content.encode('utf-8')).hexdigest()
    
    # Combine elements to form a unique ID
    # Including filename, chunk index, and a portion of content hash
    chunk_id=  f"{source_name}_{chunk_index:05d}_{content_hash[:10]}" # Increased chunk_index padding and hash length
    chunk.metadata["id"] = chunk_id

    return chunk

def main():

    # Define the path to your data folder
    DATA_PATH = r"data"
    CHROMA_PATH = r"chroma_db"
    COLLECTION_NAME = "IFast_Annual_Report2" # Define collection name for clarity

    # List of your PDF file names
    def get_pdf_file_paths(data_path):
        """
        Most efficient method to get PDF file paths
        Uses os.scandir() - fastest and most memory efficient for file discovery
        """
        try:
            with os.scandir(data_path) as entries:
                return [os.path.join(data_path, entry.name) #returns a list of full paths that meet the criteria
                    for entry in entries 
                    if entry.is_file() and entry.name.lower().endswith('.pdf')]
        except FileNotFoundError:
            print(f"Directory not found: {data_path}")
            return []

    file_paths = get_pdf_file_paths(DATA_PATH)

    if not file_paths:
        print("No PDF files found to process. Exiting.")
        # return or exit as needed
    else:
        print(f"Processing {len(file_paths)} PDF files...")
        # file_paths is ready to use
        
    try:
        # Initialize the UnstructuredLoader with chunking strategy???
        total_folder_size = len(file_paths)
        for i in range(total_folder_size):
            loader = UnstructuredPDFLoader(
                file_paths[i],
                mode = "single"
            )
            
            # Load documents
            pages = []
            print("Loading documents lazily...")
            for doc in loader.lazy_load():
                pages.append(doc)
            print(f"Successfully loaded {len(pages)} documents.")

            # Display sample content
            print("\n--- Content of the first loaded document (first 500 chars) ---")
            print(f"{pages[0].page_content[:500]}" )
            print("----------------------------------------------------------\n")
            print(f"Length of text in the first document: {len(pages[0].page_content)}")

            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )

            # Split documents into chunks
            chunks = text_splitter.split_documents(pages)
            print(f"Original documents split into {len(chunks)} chunks.")

            # Filter complex metadata to avoid Chroma issues
            filtered_chunks = filter_complex_metadata(chunks)
            print(f"Filtered chunks to {len(filtered_chunks)} valid documents.")

            #filtered_chunks = calculate_chunk_ids(filtered_chunks)
            # Display sample chunks
            print("\n--- Example of the first 3 final chunks ---")
            for i, chunk in enumerate(filtered_chunks[:3]):
                print(f"Chunk {i+1} (Length: {len(chunk.page_content)}):")
                print(chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content)
                print(f"Metadata: {chunk.metadata}")
                print("-" * 50)

            # Initialize embeddings
            print("\nInitializing embeddings...")
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
            # Set up Chroma vector store with duplicate prevention
            print("\nSetting up Chroma vector store...")
            
            #persistent_client = chromadb.PersistentClient()

            vector_store = Chroma(
                #client=persistent_client,##
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings, # Needed even for reading
                persist_directory=CHROMA_PATH,
            )
            #ALWAYS reset vector store
            #vector_store.reset_collection()
            #print("Chroma vector store has been reset and reinitialized.")

            # Check existing documents
            try:
                # Use _collection.count() for an accurate count if the collection exists
                existing_count = vector_store._collection.count() 
                print(f"Number of existing documents in DB: {existing_count}")

            except Exception as e: # Catch specific ChromaDB exceptions if possible
                existing_count = 0
                print(f"Could not retrieve existing document count (likely collection not yet created or error): {e}")
                print("Creating new collection.")

            # Generate consistent IDs for documents
            chunks_with_ids = []
            new_chunks = [] #contains new chunks to add
            for i, chunk in enumerate(filtered_chunks):
                chunk_with_ids= generate_document_id(chunk, i)  # This adds the ID directly to chunk.metadata
                chunks_with_ids.append(chunk_with_ids)

            # Check for duplicates if collection exists
            if existing_count > 0:
                # Get existing IDs
                existing_data = vector_store.get(include=[]) # Only fetch IDs for the documents we plan to add
                existing_ids = set(existing_data['ids']) if existing_data and existing_data['ids'] else set()
                print(f"Number of existing documents in DB: {len(existing_ids)}")
                
                # Filter out documents that already exist
                for chunk in chunks_with_ids:
                    if chunk.metadata["id"] not in existing_ids:
                        new_chunks.append(chunk)

                if len(new_chunks) and len(new_chunks)/len(chunks_with_ids) > 0.995:
                    print(f"👉 Adding new documents: {len(new_chunks)}")
                    new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
                    vector_store.add_documents(new_chunks, ids=new_chunk_ids)

                else:
                    print("✅ No new documents to add")
            else:
                # Empty or new collection, add all documents
                print(f"Adding all {len(chunks_with_ids)} documents to collection.")
                for chunk in chunks_with_ids:
                    new_chunks.append(chunk)

                new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
                vector_store.add_documents(new_chunks, ids=new_chunk_ids)


            # Get final count
            final_count = vector_store._collection.count()
            print(f"Final document count in the vector store: {final_count}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()