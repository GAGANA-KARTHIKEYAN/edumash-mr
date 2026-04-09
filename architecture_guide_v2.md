# 🧠 eduMASH-MR: Comprehensive Architecture & Replication Guide

This document serves as the master blueprint for the *eduMASH-MR* (Multimodal, Multilingual, Misconception-Aware RAG Tutor) project. It is written specifically so that another AI agent (or human engineer) can perfectly recreate, understand, and run this project on any machine.

---

## PART 1: PROJECT OVERVIEW

### End-to-End Flow
**eduMASH-MR** is a personalized, AI-driven educational tutoring dashboard. Unlike standard chatbots, it acts as a proactive tutor that quizzes the student based strictly on their own uploaded curriculum, dynamically detecting cognitive misconceptions.

1. **Upload & Ingestion:** The student uploads study materials (PDF, PPTX, TXT).
2. **Knowledge Extraction (Graph RAG):** The system chunks the text, extracts semantic triplets (Subject-Verb-Object), and builds a localized Knowledge Graph representing the curriculum.
3. **Curriculum Summary:** An LLM analyzes the graph and generates a multilingual, encouraging welcome message outlining the core topics detected.
4. **Adaptive Quiz Loop:**
   - **Question Generation:** The LLM generates a targeted question based on a curriculum concept (focusing on the student's historical weak areas).
   - **Multimodal Input:** The student answers the question using text, recording their voice (processed via OpenAI Whisper), or uploading an image of handwritten notes (processed via Tesseract OCR).
5. **Misconception Analysis & Feedback:**
   - The student's answer and the curriculum ground truth are retrieved.
   - A fine-tuned T5 model extracts a knowledge graph from the student's answer.
   - The graphs are compared to find missing, hallucinated, or misunderstood concepts.
   - The primary LLM generates a highly detailed, 1-page feedback report in the student's native language (e.g., English, Telugu, Spanish), explaining exactly what they got right, their conceptual misconception, and the correct theory.
6. **Reporting:** Once the quiz finishes, a personalized learning plan and mastery report are generated.

---

## PART 2: CODE STRUCTURE

The repository follows a clean, modular microservice-style design:

- **`app.py`** (Main UI)
  - The Streamlit frontend. Handles session state, file uploading, multimodal rendering, progress bars, and the interactive chat UI.
- **`main.py`** (System Entry Point)
  - Invoked during the "indexing" phase. Wires together document loading, graph building, and initializes the `RAGRetriever` pipeline.
- **`input_module/`**
  - `input_handler.py`: Unifies input modalities. Uses `openai-whisper` for audio and `pytesseract` for image OCR. Will fallback to text.
- **`retrieval_module/`**
  - `rag_retriever.py`: Houses the core vector database (FAISS/exact-match) and NetworkX graph state. Maps query concepts to node neighborhoods to fetch context.
- **`misconception_module/`**
  - `detector.py`: The semantic "critic". Loads fine-tuned Hugging Face T5 weights and PyTorch fusion models. Extracts triplets from student answers and compares them against ground-truth seed nodes.
- **`explanation_module/`**
  - `engine.py`: The LLM connective tissue. Houses Groq SDK and Google Generative AI SDK logic. Generates the curriculum summaries, questions, and the multi-section prompt for the final pedagogical evaluation.
- **`utils/`**
  - `pdf_loader.py`: Uses `pdfplumber` for robust spatial text extraction, bypassing garbled characters found in normal PyPDF loaders.
  - `graph_builder.py`: Houses regex/NLP pipelines to parse curriculum text into semantic relationships (nodes & edges).
  - `graph_renderer.py`: Generates the interactive PyVis HTML objects injected into Streamlit.
- **`training/`**
  - Houses scripts (`train_graph_gen.py`, `train_fusion.py`) and dataset preparation tools used previously to create the customized local weights.

---

## PART 3: DEPENDENCIES & ENVIRONMENT

**Python Version Requirement:** `Python 3.9+` (Recommended 3.10)

Dependencies (`requirements.txt`):
```text
streamlit>=1.30.0
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
groq>=0.4.0
google-generativeai>=0.3.0
pdfplumber>=0.10.0
pytesseract>=0.3.10
openai-whisper>=20231117
networkx>=3.0
pyvis>=0.3.2
Pillow>=10.0.0
```

*Note: For audio and image support, systemic dependencies are required:*
- **Tesseract OCR:** Must be installed on the OS level (e.g., `apt-get install tesseract-ocr` or Windows Tesseract installer).
- **FFmpeg:** Required by Whisper to process audio files (`apt-get install ffmpeg` or Chocolatey/brew).

---

## PART 4: HOW TO RUN THE PROJECT

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GAGANA-KARTHIKEYAN/edumash-mr.git
   cd edumash-mr
   ```

2. **Set up the virtual environment (highly recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the Application:**
   ```bash
   python -m streamlit run app.py --server.port 8501
   ```
   *The UI will launch on `http://localhost:8501`.*

---

## PART 5: MODELS & DATA

### Core Models:
1. **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`. Downloaded automatically from Hugging Face on the first run by the `sentence-transformers` library in `retrieval_module`.
2. **Generative Engine (API):**
   - **Primary:** `llama-3.3-70b-versatile` via Groq. (Injected seamlessly via UI).
   - **Fallback:** Gemini 2.0 Flash (`gemini-1.5-flash` or similar) via Google GenAI.
3. **Misconception Detection (Local Weights):**
   - A fine-tuned T5 sequence-to-sequence model used for knowledge graph generation.
   - Kept in `weights/t5_graph_gen_final` (local storage).
   - A fusion module model kept in `weights/fusion_module_final.pt`.
   - *Since weights are in `.gitignore`, another laptop must either run the training scripts in `training/` OR download the zipped weights manually from an external bucket (like Google Drive) into the `weights/` directory.*

### Data Storage:
- Uploaded PDFs/audio files are temporarily saved in the root `data/` directory.
- This directory is dynamically wiped and recreated when a student uploads new files.

---

## PART 6: CONFIGURATION DETAILS

- **API Keys:** There are no `.env` files required for core operation. Both Groq and Gemini API keys are entered directly, securely into the Streamlit sidebar (stored in session state).
- **NetworkX Tuning:** In `utils/graph_builder.py`, `MAX_NODES` may be tuned (currently ~150-300 depending on implementation) to prevent PyVis from rendering spaghetti graphs.
- **Concurrency / CPU:** In `retrieval_module`, `TOKENIZERS_PARALLELISM=false` is enforced to prevent deadlock warnings when deploying to Unix systems.

---

## PART 7: REPLICATION GUIDE FOR ANTIGRAVITY

If you (Antigravity) need to rewrite this project from scratch locally, follow this exact build order:

1. **Step 1: Scaffolding**
   - Create the directory structure: `utils/`, `input_module/`, `explanation_module/`, `retrieval_module/`, `misconception_module/`.
   - Create a blank `app.py` and `requirements.txt`.

2. **Step 2: Utility & Extraction (`utils/pdf_loader.py` & `utils/graph_builder.py`)**
   - Implement `pdfplumber` to pull text cleanly.
   - Implement a regex-based sentence matcher to extract Subject-Verb-Object triplets.

3. **Step 3: Core Retrieval (`retrieval_module/rag_retriever.py` & `main.py`)**
   - Write `RAGRetriever` encapsulating `SentenceTransformer` and `NetworkX`. Provide a `retrieve(concept)` method that returns semantic text chunks and the 1-hop subgraph relationships.
   - Write `build_system()` in `main.py` which ties `pdf_loader` -> `graph_builder` -> `RAGRetriever`.

4. **Step 4: AI Brain (`explanation_module/engine.py`)**
   - Implement the `configure_groq()` global state.
   - Write `generate_next_question()` utilizing Socratic prompting.
   - Write `evaluate_student_answer_full()` utilizing **Plain-Text Delimiters** (e.g., `[WHAT YOU GOT RIGHT]`) instead of strict JSON bounds to prevent long-generation ParseErrors on 4096-token outputs.

5. **Step 5: Misconception Evaluator (`misconception_module/detector.py`)**
   - Utilize standard rule-based semantic-similarity fallbacks (Sentence-BERT cosine similarity thresholds) if local T5 custom PyTorch weights are missing. Compare student triplets against reference chunks.

6. **Step 6: Frontend (`app.py`)**
   - Build Streamlit UI focusing intensely on `st.session_state` persistence.
   - Implement strict UI phases: `"upload"` -> `"indexing"` -> `"quiz"` -> `"report"`.
   - Bind Native Language multi-select (e.g. English, Telugu, Kannada) directly into all LLM prompt templates dynamically.

---

## PART 8: COMMON ISSUES & FIXES

1. **`ModuleNotFoundError: No module named 'tesseract'` or OCR Failing**
   - *Fix:* Ensure the system OS actually has Google Tesseract installed, not just the python wrapper (`sudo apt install tesseract-ocr`). Link the path properly if on Windows.

2. **Garbled PDF Extraction (Words merged together like `thetraffic`)**
   - *Fix:* This happens when using `PyPDF2` on columned academic PDFs. The project currently uses `pdfplumber(x_tolerance=2, y_tolerance=3)` to correct this. Do not revert to standard PyPDF.

3. **"Evaluation failed, falling back to Offline Mode" immediately after answering**
   - *Fix:* Check `engine_debug.log`. This usually implies Groq hit its strict Rate Limit or Max Tokens cutoff. Ensure `engine.py` is utilizing text-delimited parsing (`[SCORE]`, `[CORRECT EXPLANATION]`), as enforcing pure `{JSON}` output on massive pedagogical feedback causes JSON format truncation.

4. **Missing Weights / Model Exceptions (`FileNotFoundError`)**
   - *Fix:* Since `.pt` files aren't in GitHub, the `detector.py` falls back gracefully via a Try/Except block, or throws an error. Another system replicating this must re-train the models using the `./training` directory locally before `misconception_module` works with full fusion logic.

---
*Generated by Antigravity for seamless system replication.*
