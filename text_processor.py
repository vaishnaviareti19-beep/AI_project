from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


def clean_text(text: str) -> str:
    """
    Clean unnecessary whitespace and null characters.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = " ".join(text.split())

    return text.strip()


def clean_documents(documents):
    """
    Clean the text of every loaded document
    while preserving metadata.
    """

    cleaned_documents = []

    for document in documents:

        document.page_content = clean_text(
            document.page_content
        )

        if document.page_content:
            cleaned_documents.append(document)

    return cleaned_documents


def split_documents(documents):
    """
    Split documents into smaller chunks
    for semantic retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_documents(documents)

    return chunks