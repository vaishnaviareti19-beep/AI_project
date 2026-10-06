from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "medical_research_documents"
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# OLLAMA EMBEDDINGS
# ============================================================

def get_embeddings():
    """
    Create the local Ollama embedding model.
    """

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(documents):
    """
    Create a Chroma vector database
    containing the supplied document chunks.
    """

    Path(CHROMA_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Get existing document IDs
    try:
        existing_data = vector_store.get()

        existing_ids = existing_data.get("ids", [])

        if existing_ids:
            vector_store.delete(
                ids=existing_ids
            )

    except Exception:
        pass

    # Add new document chunks
    if documents:
        vector_store.add_documents(
            documents
        )

    return vector_store


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():
    """
    Load the existing Chroma vector database.
    """

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    return vector_store


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(
    query: str,
    k: int = 4
):
    """
    Search the vector database using
    semantic similarity.
    """

    vector_store = load_vector_store()

    try:
        results = vector_store.similarity_search(
            query,
            k=k
        )

        return results

    except Exception:
        return []


# ============================================================
# CLEAR VECTOR STORE
# ============================================================

def clear_vector_store():
    """
    Clear all documents from the Chroma collection.

    This avoids deleting the entire chroma_db folder,
    which can cause Windows file-locking errors.
    """

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    try:
        data = vector_store.get()

        ids = data.get("ids", [])

        if ids:
            vector_store.delete(
                ids=ids
            )

        return True

    except Exception as error:
        raise RuntimeError(
            f"Unable to clear vector database: {error}"
        )