from pathlib import Path

import streamlit as st

from modules.document_loader import (
    load_document,
    get_document_name,
)

from modules.text_processor import (
    clean_documents,
    split_documents,
)

from modules.vector_store import (
    create_vector_store,
    clear_vector_store,
)

from modules.qa_engine import (
    answer_question,
)


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "Medical Research Documentation Assistant"

DOCUMENT_FOLDER = Path("data/documents")

DOCUMENT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .info-card {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    .source-box {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #dddddd;
        margin-bottom: 8px;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #cccccc;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div class="main-title">
        🧬 {APP_TITLE}
    </div>

    <div class="subtitle">
        Upload medical research documents, search their
        content, and ask questions using a local AI
        research assistant.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Document Management")

    uploaded_files = st.file_uploader(
        "Upload research documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload PDF or DOCX research documents.",
    )

    process_button = st.button(
        "⚡ Process Documents",
        use_container_width=True,
        type="primary",
    )

    st.divider()

    clear_button = st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True,
    )

    st.divider()

    st.caption(
        "📄 Supported formats: PDF and DOCX"
    )

    st.caption(
        "🤖 AI model: Ollama"
    )

    st.caption(
        "🔎 Vector database: Chroma"
    )

    st.caption(
        "⛓️ Framework: LangChain"
    )


# ============================================================
# CLEAR KNOWLEDGE BASE
# ============================================================

if clear_button:

    try:

        # Clear Chroma vector collection
        clear_vector_store()

        # Delete uploaded documents
        deleted_files = 0

        for file in DOCUMENT_FOLDER.iterdir():

            if file.is_file():

                if file.suffix.lower() in [".pdf", ".docx"]:

                    file.unlink()

                    deleted_files += 1

        st.success(
            "Knowledge base cleared successfully."
        )

        st.info(
            f"Removed {deleted_files} document(s)."
        )

        st.rerun()

    except Exception as error:

        st.error(
            f"Unable to clear knowledge base: {error}"
        )


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if process_button:

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF or DOCX file."
        )

    else:

        progress = st.progress(0)

        status_text = st.empty()

        all_documents = []

        successful_files = 0

        failed_files = 0

        total_files = len(uploaded_files)


        # ----------------------------------------------------
        # PROCESS EACH FILE
        # ----------------------------------------------------

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            file_path = (
                DOCUMENT_FOLDER /
                uploaded_file.name
            )

            try:

                status_text.write(
                    f"Processing: **{uploaded_file.name}**"
                )


                # --------------------------------------------
                # SAVE FILE
                # --------------------------------------------

                with open(
                    file_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )


                # --------------------------------------------
                # LOAD DOCUMENT
                # --------------------------------------------

                documents = load_document(
                    str(file_path)
                )


                # --------------------------------------------
                # CLEAN DOCUMENT
                # --------------------------------------------

                documents = clean_documents(
                    documents
                )


                # --------------------------------------------
                # ADD SOURCE METADATA
                # --------------------------------------------

                for document in documents:

                    document.metadata["source"] = (
                        get_document_name(
                            str(file_path)
                        )
                    )


                # --------------------------------------------
                # ADD TO ALL DOCUMENTS
                # --------------------------------------------

                all_documents.extend(
                    documents
                )

                successful_files += 1


            except Exception as error:

                failed_files += 1

                st.error(
                    f"Error processing "
                    f"{uploaded_file.name}: {error}"
                )


            # --------------------------------------------
            # UPDATE PROGRESS
            # --------------------------------------------

            progress.progress(
                (index + 1) / total_files
            )


        status_text.empty()


        # ====================================================
        # CREATE VECTOR DATABASE
        # ====================================================

        if all_documents:

            try:

                # Split documents into chunks
                chunks = split_documents(
                    all_documents
                )


                # Create/update Chroma database
                create_vector_store(
                    chunks
                )


                st.success(
                    f"Successfully processed "
                    f"{successful_files} document(s)."
                )


                st.info(
                    f"Created {len(chunks)} "
                    f"searchable text chunks."
                )


                if failed_files > 0:

                    st.warning(
                        f"{failed_files} document(s) "
                        f"could not be processed."
                    )


            except Exception as error:

                st.error(
                    "Vector database error: "
                    f"{error}"
                )

        else:

            st.error(
                "No valid document content was found."
            )


# ============================================================
# DOCUMENT LIST
# ============================================================

st.subheader(
    "📁 Uploaded Research Documents"
)


existing_files = [
    file
    for file in DOCUMENT_FOLDER.iterdir()
    if file.is_file()
    and file.suffix.lower() in [".pdf", ".docx"]
]


if existing_files:

    st.write(
        f"**{len(existing_files)} document(s) available**"
    )


    for file in sorted(
        existing_files,
        key=lambda item: item.name.lower()
    ):

        file_size = file.stat().st_size / 1024

        st.markdown(
            f"""
            <div class="info-card">
                📄 <strong>{file.name}</strong><br>
                <small>Size: {file_size:.1f} KB</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:

    st.info(
        "No research documents uploaded yet."
    )


# ============================================================
# QUESTION SECTION
# ============================================================

st.divider()

st.subheader(
    "🔎 Ask Your Research Question"
)

st.write(
    "Ask questions about the information contained "
    "in your uploaded research documents."
)


question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What are the major findings "
        "reported in this research?"
    ),
    height=120,
)


ask_button = st.button(
    "🤖 Ask Research Assistant",
    type="primary",
    use_container_width=True,
)


# ============================================================
# QUESTION ANSWERING
# ============================================================

if ask_button:

    # --------------------------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "Please enter a question."
        )


    # --------------------------------------------------------
    # CHECK DOCUMENTS
    # --------------------------------------------------------

    elif not existing_files:

        st.warning(
            "Please upload and process research "
            "documents first."
        )


    # --------------------------------------------------------
    # ANSWER QUESTION
    # --------------------------------------------------------

    else:

        with st.spinner(
            "🔎 Searching research documents and "
            "generating answer..."
        ):

            try:

                answer, sources = answer_question(
                    question.strip()
                )


                # ============================================
                # ANSWER
                # ============================================

                st.subheader(
                    "💡 Research Answer"
                )

                st.write(
                    answer
                )


                # ============================================
                # SUPPORTING SOURCES
                # ============================================

                if sources:

                    st.subheader(
                        "📚 Supporting Sources"
                    )

                    displayed_sources = set()


                    for source in sources:

                        source_name = (
                            source.metadata.get(
                                "source",
                                "Unknown source"
                            )
                        )


                        page = (
                            source.metadata.get(
                                "page",
                                None
                            )
                        )


                        if page is not None:

                            source_text = (
                                f"{source_name} "
                                f"— Page {page + 1}"
                            )

                        else:

                            source_text = (
                                source_name
                            )


                        if (
                            source_text
                            not in displayed_sources
                        ):

                            st.markdown(
                                f"""
                                <div class="source-box">
                                    📄 {source_text}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )


                            displayed_sources.add(
                                source_text
                            )


                else:

                    st.info(
                        "No supporting sources were returned."
                    )


            except Exception as error:

                st.error(
                    "Unable to answer the question: "
                    f"{error}"
                )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.divider()

st.subheader(
    "💭 Example Questions"
)

example_questions = [
    "What is the main topic of this research?",
    "What are the major findings?",
    "What is the objective of the study?",
    "What are the main conclusions?",
    "What research gaps are identified?",
]


for example in example_questions:

    st.markdown(
        f"• {example}"
    )


# ============================================================
# APPLICATION INFORMATION
# ============================================================

st.divider()

with st.expander(
    "ℹ️ About this Application"
):

    st.write(
        """
        The Medical Research Documentation Assistant is
        a local Retrieval-Augmented Generation (RAG)
        application.

        Documents are processed locally and converted
        into searchable text chunks.

        The application uses:

        • Streamlit for the user interface
        • LangChain for document processing and RAG
        • Ollama for local AI models
        • Chroma for vector storage
        • nomic-embed-text for document embeddings
        • Llama 3.2 for question answering

        The system retrieves relevant information from
        uploaded documents before generating an answer.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Medical Research Documentation Assistant "
    "| Streamlit | LangChain | Ollama | Chroma"
)

st.caption(
    "For research-document assistance only. "
    "This application is not a substitute for "
    "professional medical advice."
)