import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Intelligent Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Intelligent Research Paper Assistant")

st.write(
    "Upload research papers and ask questions using "
    "LLM + RAG + semantic search."
)


# ==================================================
# SESSION STATE - CHAT HISTORY
# ==================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_embedding_model()


# ==================================================
# LLM QUESTION ANSWERING
# ==================================================

def ask_llm(
    context,
    question,
    chat_history
):

    previous_conversation = ""

    if chat_history:

        for item in chat_history[-5:]:

            previous_conversation += (
                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}\n\n"
            )

    else:

        previous_conversation = (
            "No previous conversation."
        )


    prompt = f"""
You are an intelligent conversational research paper
question-answering assistant.

Answer the user's question using ONLY the provided
research paper evidence.

Previous conversation is provided only to understand
references such as "this", "it", "these", or "the above".
It is NOT an additional source of factual information.

STRICT GROUNDING RULES:

1. Use ONLY information explicitly present in the
   research paper evidence.

2. Do NOT use outside knowledge.

3. Do NOT invent facts, numbers, datasets, authors,
   affiliations, results, algorithms, or performance.

4. If the requested information is not clearly available,
   respond:

"The information is not available in the provided
research paper."

5. Do not mix proposed methodology with future work.

6. Clearly distinguish between:
   - proposed methods
   - implemented methods
   - reported results
   - future work

7. Never create accuracy, dataset size, or performance
   values that are not reported.

8. Answer the exact question asked.

9. Do not add unrelated information from the evidence.

10. Keep the answer concise and academic.

11. If the question asks for techniques, mention only
    techniques actually relevant to the question.

PREVIOUS CONVERSATION:
----------------------
{previous_conversation}
----------------------

RESEARCH PAPER EVIDENCE:
------------------------
{context}
------------------------

CURRENT QUESTION:
{question}

ANSWER:
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",

            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False
            },

            timeout=120
        )

        response.raise_for_status()

        return response.json()["response"]

    except requests.exceptions.RequestException as e:

        return f"Error connecting to Ollama: {e}"


# ==================================================
# AUTOMATIC RESEARCH PAPER SUMMARY
# ==================================================

def generate_summary(context):

    prompt = f"""
You are an academic research paper summarization assistant.

Summarize the research paper using ONLY the provided context.

STRICT RULES:

1. Use ONLY information explicitly present in the context.
2. Do NOT use outside knowledge.
3. Do NOT invent facts, numbers, datasets, authors,
   results, algorithms, or performance values.
4. Clearly distinguish proposed ideas from implemented work.
5. Do not claim experimental results unless explicitly reported.
6. If information is missing, write:
   "Not clearly reported in the provided research paper."
7. Keep the summary concise and academic.

Return these sections:

### 🎯 Objective

### 🔬 Methodology

### 🧠 Key Techniques

### 📂 Dataset / Data

### 📊 Results

### ⚠️ Limitations

### 🚀 Future Work

RESEARCH PAPER CONTEXT:
-----------------------
{context}
-----------------------

SUMMARY:
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",

            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False
            },

            timeout=120
        )

        response.raise_for_status()

        return response.json()["response"]

    except requests.exceptions.RequestException as e:

        return f"Error connecting to Ollama: {e}"


# ==================================================
# PDF UPLOAD
# ==================================================

uploaded_files = st.file_uploader(
    "📄 Upload Research Paper(s)",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    all_chunks = []
    source_names = []


    # ==================================================
    # EXTRACT PDF TEXT
    # ==================================================

    for uploaded_file in uploaded_files:

        reader = PdfReader(
            uploaded_file
        )

        full_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:

                full_text += (
                    text + "\n"
                )


        # ==================================================
        # CREATE CHUNKS
        # ==================================================

        chunk_size = 1000

        chunks = [
            full_text[i:i + chunk_size]
            for i in range(
                0,
                len(full_text),
                chunk_size
            )
        ]


        for chunk in chunks:

            if chunk.strip():

                all_chunks.append(
                    chunk
                )

                source_names.append(
                    uploaded_file.name
                )


    st.success(
        f"✅ {len(uploaded_files)} paper(s) "
        f"processed successfully!"
    )

    st.write(
        f"Total text chunks: **{len(all_chunks)}**"
    )


    # ==================================================
    # CREATE EMBEDDINGS
    # ==================================================

    with st.spinner(
        "Creating semantic embeddings..."
    ):

        embeddings = model.encode(
            all_chunks,
            convert_to_numpy=True
        )


    # ==================================================
    # NORMALIZE EMBEDDINGS
    # ==================================================

    embeddings = embeddings.astype(
        "float32"
    )

    faiss.normalize_L2(
        embeddings
    )


    # ==================================================
    # CREATE FAISS COSINE INDEX
    # ==================================================

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )


    st.success(
        "✅ Research papers indexed successfully!"
    )


    # ==================================================
    # AUTOMATIC SUMMARY
    # ==================================================

    st.subheader(
        "📄 Research Paper Summary"
    )

    st.write(
        "Generate an automatic academic summary "
        "of the uploaded research paper."
    )


    if st.button(
        "📄 Generate Research Summary"
    ):

        max_summary_chunks = 15

        if len(all_chunks) <= max_summary_chunks:

            summary_chunks = all_chunks

        else:

            selected_indices = np.linspace(
                0,
                len(all_chunks) - 1,
                max_summary_chunks,
                dtype=int
            )

            summary_chunks = [
                all_chunks[i]
                for i in selected_indices
            ]


        summary_context = "\n\n".join(
            summary_chunks
        )


        with st.spinner(
            "🤖 Generating research paper summary..."
        ):

            summary = generate_summary(
                summary_context
            )


        st.markdown(
            summary
        )


    # ==================================================
    # CONVERSATION HISTORY
    # ==================================================

    if st.session_state.chat_history:

        st.subheader(
            "💬 Conversation History"
        )

        for i, item in enumerate(
            st.session_state.chat_history
        ):

            st.markdown(
                f"**👤 Question {i + 1}:** "
                f"{item['question']}"
            )

            st.markdown(
                f"**🤖 Answer:** "
                f"{item['answer']}"
            )

            st.divider()


    # ==================================================
    # QUESTION INPUT
    # ==================================================

    st.subheader(
        "🔎 Ask a Question"
    )

    question = st.text_input(
        "Enter your question about the research paper:",
        key="question_input"
    )


    if question:

        # ==================================================
        # QUESTION EMBEDDING
        # ==================================================

        question_embedding = model.encode(
            [question],
            convert_to_numpy=True
        )

        question_embedding = question_embedding.astype(
            "float32"
        )

        faiss.normalize_L2(
            question_embedding
        )


        # ==================================================
        # RETRIEVE TOP 5 CHUNKS
        # ==================================================

        distances, indices = index.search(
            question_embedding,
            5
        )


        retrieved_chunks = []
        retrieved_sources = []
        retrieved_scores = []


        for i, idx in enumerate(
            indices[0]
        ):

            similarity = float(
                distances[0][i]
            )

            retrieved_chunks.append(
                all_chunks[idx]
            )

            retrieved_sources.append(
                source_names[idx]
            )

            retrieved_scores.append(
                similarity
            )


        # ==================================================
        # REMOVE VERY WEAK RESULTS
        # ==================================================

        filtered_chunks = []
        filtered_sources = []
        filtered_scores = []


        for i in range(
            len(retrieved_chunks)
        ):

            similarity = retrieved_scores[i]

            # Keep reasonably related evidence
            if similarity >= 0.20:

                filtered_chunks.append(
                    retrieved_chunks[i]
                )

                filtered_sources.append(
                    retrieved_sources[i]
                )

                filtered_scores.append(
                    similarity
                )


        # If filtering removes everything,
        # use the strongest retrieved chunk.

        if not filtered_chunks:

            filtered_chunks = [
                retrieved_chunks[0]
            ]

            filtered_sources = [
                retrieved_sources[0]
            ]

            filtered_scores = [
                retrieved_scores[0]
            ]


        # ==================================================
        # LIMIT CONTEXT
        # ==================================================

        max_context_chunks = 3

        final_chunks = filtered_chunks[
            :max_context_chunks
        ]

        final_sources = filtered_sources[
            :max_context_chunks
        ]

        final_scores = filtered_scores[
            :max_context_chunks
        ]


        # ==================================================
        # CREATE FINAL CONTEXT
        # ==================================================

        context_parts = []

        for i, chunk in enumerate(
            final_chunks
        ):

            context_parts.append(
                f"""
EVIDENCE {i + 1}
SOURCE: {final_sources[i]}

{chunk}
"""
            )


        context = "\n\n".join(
            context_parts
        )


        # ==================================================
        # GENERATE ANSWER
        # ==================================================

        with st.spinner(
            "🤖 Generating answer using LLM..."
        ):

            answer = ask_llm(
                context,
                question,
                st.session_state.chat_history
            )


        # ==================================================
        # SAVE CHAT HISTORY
        # ==================================================

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": answer
            }
        )


        # ==================================================
        # DISPLAY ANSWER
        # ==================================================

        st.subheader(
            "🤖 AI Answer"
        )

        st.write(
            answer
        )


        # ==================================================
        # SOURCES
        # ==================================================

        st.subheader(
            "📚 Sources & Semantic Similarity"
        )

        unique_sources = list(
            dict.fromkeys(
                final_sources
            )
        )

        for source in unique_sources:

            st.write(
                f"📄 **{source}**"
            )


        # ==================================================
        # RETRIEVED EVIDENCE
        # ==================================================

        st.markdown(
            "### 📊 Retrieved Evidence"
        )


        for i in range(
            len(final_chunks)
        ):

            similarity = final_scores[i]

            st.write(
                f"**Retrieved Section {i + 1}** "
                f"— Semantic Similarity: "
                f"**{similarity:.4f}**"
            )

            st.caption(
                f"Source: {final_sources[i]}"
            )


        # ==================================================
        # VIEW EVIDENCE
        # ==================================================

        with st.expander(
            "🔍 View Retrieved Research Evidence"
        ):

            for i, chunk in enumerate(
                final_chunks
            ):

                similarity = final_scores[i]

                st.markdown(
                    f"### 📄 Retrieved Section {i + 1}"
                )

                st.write(
                    f"**Source:** "
                    f"{final_sources[i]}"
                )

                st.write(
                    f"**Semantic Similarity:** "
                    f"{similarity:.4f}"
                )

                st.markdown(
                    "#### Evidence Used"
                )

                st.write(
                    chunk
                )

                st.divider()


    # ==================================================
    # CLEAR CHAT
    # ==================================================

    st.divider()

    if st.button(
        "🗑️ Clear Conversation"
    ):

        st.session_state.chat_history = []

        st.rerun()