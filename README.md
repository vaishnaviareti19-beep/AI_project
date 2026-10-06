# Medical Research Documentation Assistant

## Overview

The Medical Research Documentation Assistant is a local AI-powered
document analysis system designed to help users understand and
search medical research documents.

Users can upload PDF or DOCX research documents and ask questions
about their content.

The system retrieves relevant sections from the documents and
uses a local Large Language Model to generate an answer.

## Main Features

- PDF document upload
- DOCX document upload
- Automatic text extraction
- Text cleaning
- Intelligent text chunking
- Semantic document search
- Local vector database
- AI-powered question answering
- Supporting source identification
- Page number references
- Multiple document support
- Knowledge-base clearing
- Local AI processing

## Technologies

- Python
- Streamlit
- LangChain
- Ollama
- Chroma
- PyPDF
- Python-docx

## Architecture

User
  |
  v
Streamlit Interface
  |
  v
Document Upload
  |
  v
PDF / DOCX Loader
  |
  v
Text Cleaning
  |
  v
Text Chunking
  |
  v
Ollama Embeddings
  |
  v
Chroma Vector Database
  |
  v
Semantic Retrieval
  |
  v
Ollama LLM
  |
  v
Research Answer + Sources

## Ollama Models

Install Ollama separately.

Pull the language model:

ollama pull llama3.2

Pull the embedding model:

ollama pull nomic-embed-text

Check installed models:

ollama list

## Installation

Create and activate a virtual environment:

python -m venv venv

Windows PowerShell:

.\venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

## Run

Start the Streamlit application:

streamlit run app.py

The application will open in the browser.

## Usage

1. Upload one or more research documents.
2. Click Process Documents.
3. Wait for the vector database to be created.
4. Enter a research question.
5. Click Ask Research Assistant.
6. Read the generated answer.
7. Review the supporting document sources.

## Safety

This application is designed for research-document assistance.

It should not be used for:

- Medical diagnosis
- Treatment decisions
- Medication prescriptions
- Emergency medical decisions

Always consult a qualified healthcare professional for medical advice.

## Project Type

AI / Machine Learning / Natural Language Processing /
Retrieval-Augmented Generation (RAG)