# PDF Analysis (RAG + Flask)

A local PDF question-answering web app using:
- Flask for the web UI/API
- LangChain for retrieval pipeline
- ChromaDB for vector storage
- Ollama for embeddings + LLM inference

Users can:
- Browse PDFs from the `data/` folder
- Open a PDF in the built-in viewer
- Ask natural-language questions grounded in PDF content

## Project Structure

- `app.py`: Flask app and API routes
- `rag.py`: PDF loading, chunking, embeddings, vector DB, retrieval, and answer generation
- `templates/index.html`: Frontend UI
- `static/style.css`: Frontend styles
- `data/`: Place your PDF files here
- `db/`: Persisted Chroma vector database

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- Models available in Ollama:
  - `llama3.1`
  - `nomic-embed-text`

Optional (for reranking quality boost):
- `sentence-transformers`

## Installation

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
pip install flask markdown pymupdf
```

3. Pull Ollama models.

```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Run

1. Put one or more `.pdf` files inside `data/`.
2. Start Ollama (if not already running).
3. Run the app:

```powershell
python app.py
```

4. Open your browser at:
- `http://127.0.0.1:5000`

## API Endpoints

- `GET /` -> Main web UI
- `GET /pdfs` -> List all PDFs in `data/`
- `GET /pdf/<filename>` -> Serve selected PDF
- `POST /ask` -> Ask a question

Example request body for `/ask`:

```json
{ "question": "What is the main finding of this paper?" }
```

## Notes

- On first run, vector DB creation may take time; later runs reuse `db/`.
- `rag.py` loads/initializes the pipeline at import time, so startup cost is expected.
- If `sentence-transformers` is not installed, the app continues without reranking.

## Troubleshooting

- `Model not found`:
  - Run `ollama pull llama3.1` and `ollama pull nomic-embed-text`.
- Slow first startup:
  - Normal while building the Chroma database.
- No PDFs shown:
  - Ensure files are in `data/` and have `.pdf` extension.
- Import errors for Flask/Markdown/PyMuPDF:
  - Install missing packages with `pip install flask markdown pymupdf`.
