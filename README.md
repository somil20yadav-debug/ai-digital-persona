# 🎭 YOU — AI Digital Persona & Writing Style Clone

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **College Capstone / Google AI-ML Program Project**  
> An end-to-end, privacy-first digital persona engine that learns, quantifies, and clones an individual's unique writing voice. Built using local few-shot retrieval-augmented generation (RAG), vector similarity search, and computational stylometry.

---

## 🚀 Key Highlights

* **100% Privacy & Zero Data Egress:** All embeddings and LLM inferences run completely locally using **Ollama (Llama 3 / 3.2)** and local transformer models. Private emails, notes, and messages never touch third-party servers.
* **Sub-Millisecond Semantic Retrieval:** Employs an in-process **C++ HNSW (Hierarchical Navigable Small World)** vector index (`hnswlib`) over dense 384-dimensional sentence embeddings (`all-MiniLM-L6-v2`).
* **Computational Stylometry:** Mathematical style profiling calculating vocabulary richness (Type-Token Ratio), average sentence cadence, readability scores, and n-gram frequencies to measure stylistic congruence before and after generation.
* **Dual Interface:** Comes with both a production-ready **FastAPI** REST API and an interactive **Streamlit** dashboard featuring Plotly radar charts for style visualization.

---

## 🏛️ System Architecture

```
[ User Input / Samples ]
           │
           ▼
┌───────────────────────────────────────────────┐
│              1. Stylometry Analyzer           │
│  - Vocab Richness   - Sentence Cadence        │
│  - Formality Score  - Readability Index       │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│           2. Dense Vector Indexing            │
│  - SentenceTransformers (all-MiniLM-L6-v2)    │
│  - HNSWLib Cosine Graph Index (Top-K)         │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│        3. Dynamic Few-Shot Synthesis          │
│  - Constructs context-aligned persona prompt  │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│          4. Local LLM Inference Engine        │
│  - Ollama Daemon (Llama 3 / 3.2 3B Quantized) │
│  - Non-streaming HTTP IPC                     │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│        5. Feedback & Similarity Scoring       │
│  - Computes Style Match %                     │
│  - Persists exchange in SQLite via SQLAlchemy │
└───────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
├── app.py                 # Streamlit interactive UI & analytics dashboard
├── api.py                 # FastAPI REST server (/generate endpoint)
├── config.py              # Global configuration & threshold settings
├── test_persona.py        # End-to-end pipeline verification script
├── requirements.txt       # Pinned dependencies
├── data/
│   ├── samples/           # Input text samples for persona training
│   └── you.db             # SQLite database storing samples and conversations
├── models/
│   └── embeddings/        # HNSW index, embeddings (.npy), and metadata
├── src/
│   ├── database.py        # SQLAlchemy ORM models (Samples, Conversations, Metrics)
│   ├── generator.py       # Vector retrieval, prompt synthesis, & Ollama client
│   ├── persona_trainer.py # Vector index builder & sample manager
│   └── style_analyzer.py  # Stylometric analysis & similarity math
└── tests/
    └── test_persona.py    # Python unittest test suite
```

---

## ⚡ Quickstart

### 1. Prerequisites & Virtual Environment
Ensure you have Python 3.10+ installed:
```bash
# Clone the repository
git clone https://github.com/your-username/you-persona-clone.git
cd you-persona-clone

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Local Ollama (Optional for Inference)
```bash
# Start Ollama daemon
ollama serve

# In another terminal, pull the model
ollama pull llama3
```
*(Note: If Ollama is offline, the pipeline still performs vector retrieval, stylometry analysis, and prompt synthesis gracefully).*

### 3. Run Pipeline Verification Test
```bash
python test_persona.py
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Run the FastAPI REST Server
```bash
uvicorn api:app --reload --port 8000
```
API Documentation will be live at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🧪 Running Unit Tests

Run the built-in test suite:
```bash
python -m unittest tests/test_persona.py
```

---

## 📊 Technical Evaluation & Metrics

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Vector Index** | `hnswlib` | C++ graph-based indexing with $O(\log N)$ search latency, far lighter than external DB clusters. |
| **Embeddings** | `all-MiniLM-L6-v2` | 384-dimensional dense vectors balancing semantic resolution with CPU inference speed. |
| **Inference** | Ollama (`llama3`) | On-device quantized execution preserving absolute data confidentiality. |
| **Persistence** | SQLAlchemy / SQLite | Lightweight ACID-compliant local storage with non-expiring session commits. |

---

## 🎓 Academic / Project Context
Developed as an applied project exploring:
1. Applied NLP and computational stylometry.
2. Few-shot in-context learning with quantized open-weights models.
3. Edge deployment and low-resource semantic search architectures.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
