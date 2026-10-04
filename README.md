# Intelligent Research Paper Assistant Using LLM, RAG and Semantic Search

An intelligent research paper assistant that uses Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), Sentence Transformers, and FAISS semantic search** to analyze research papers and generate grounded answers.

The system retrieves relevant information from an uploaded research paper and uses a locally hosted LLM to answer questions only using the provided paper context.

## 🚀 Features

- 📄 Upload and analyze research papers in PDF format
- 🔎 Semantic search using Sentence Transformer embeddings
- ⚡ FAISS-based efficient similarity search
- 🤖 Local LLM integration using Ollama
- 💬 Context-aware conversational question answering
- 📝 Automatic research paper summary
- 📚 Source and evidence display
- 📊 Semantic similarity scores for retrieved evidence
- 🛡️ Grounded responses to reduce hallucination
- 🔄 Conversation memory for follow-up questions

## 🧠 Technologies Used

- Python
- Streamlit
- Ollama
- Llama 3.2 (3B)
- Sentence Transformers
- FAISS
- PyPDF
- NumPy
- Requests

## 🏗️ System Architecture

Research Paper PDF
        ↓
   PDF Extraction
        ↓
   Text Processing
        ↓
Sentence Transformer
     Embeddings
        ↓
   FAISS Semantic
       Search
        ↓
Relevant Evidence
        ↓
    RAG Context
        ↓
 Local LLM (Ollama)
        ↓
  Grounded Answer