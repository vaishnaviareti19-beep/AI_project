from langchain_ollama import OllamaLLM

from modules.vector_store import load_vector_store


LLM_MODEL = "llama3.2"


def get_llm():
    """
    Create the local Ollama language model.
    """

    return OllamaLLM(
        model=LLM_MODEL,
        temperature=0.1,
    )


def build_context(documents):
    """
    Convert retrieved documents into
    a readable context for the LLM.
    """

    context_parts = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            None
        )

        if page is not None:
            source_info = (
                f"{source}, Page {page + 1}"
            )
        else:
            source_info = source

        context_parts.append(
            f"SOURCE: {source_info}\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(
        context_parts
    )


def answer_question(
    question: str,
    k: int = 4
):
    """
    Retrieve relevant research content
    and generate an answer.
    """

    vector_store = load_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=k
    )

    if not documents:
        return (
            "No relevant information was found "
            "in the uploaded documents.",
            []
        )

    context = build_context(documents)

    prompt = f"""
You are a Medical Research Documentation Assistant.

Answer the user's question using ONLY the
research-document context provided below.

IMPORTANT RULES:

1. Use only information present in the context.
2. Do not invent facts.
3. Do not make a medical diagnosis.
4. Do not prescribe medicines.
5. Do not recommend treatments.
6. If the answer is not present in the documents,
   clearly say that it was not found.
7. Keep the answer clear and concise.
8. When useful, organize the answer using
   bullet points.
9. Mention limitations when relevant.

RESEARCH DOCUMENT CONTEXT
==========================

{context}

USER QUESTION
=============

{question}

ANSWER
======
"""

    llm = get_llm()

    answer = llm.invoke(prompt)

    return answer, documents