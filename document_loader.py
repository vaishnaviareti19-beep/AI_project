from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
)


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def load_document(file_path: str):
    """
    Load a PDF or DOCX file using LangChain.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    else:
        raise ValueError(
            "Unsupported file type. "
            "Only PDF and DOCX files are supported."
        )

    documents = loader.load()

    return documents


def get_document_name(file_path: str) -> str:
    """
    Return only the document filename.
    """

    return Path(file_path).name