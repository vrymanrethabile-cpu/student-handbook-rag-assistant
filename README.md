# Student Handbook RAG Assistant

This project builds a retrieval-augmented generation (RAG) assistant for a student handbook. It extracts text from a PDF or text file, splits the content into chunks, generates TF-IDF embeddings, stores them in an in-memory vector database, and answers questions using the retrieved handbook context.

## Features

- Loads the handbook from a PDF or text file
- Splits text into searchable chunks with page tracking
- Generates TF-IDF embeddings for semantic retrieval
- Stores embeddings in an in-memory vector database
- Retrieves relevant chunks using cosine similarity
- Uses rule-based answer generation with context extraction
- Exposes a JSON API at `POST /ask`
- Returns a clear `"I could not find this information in the handbook."` response when the answer is not available
- Handles invalid requests gracefully

## Quick Start

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your handbook PDF to `data/` or use the default text version generated automatically by the app.

4. Start the API:

   ```bash
   uvicorn rag_app.api:app --host 0.0.0.0 --port 8000
   ```

5. Send a question:

   ```bash
   curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What is the attendance requirement?"}'
   ```

   Example response:

   ```json
   {
     "answer": "Students must attend at least 80% of all classes.",
     "source": "Page 1"
   }
   ```

## Project Structure

```text
.
├── rag_app/
│   ├── __init__.py
│   ├── api.py
│   └── handbook.py
├── tests/
│   └── test_rag.py
├── data/
│   └── student_handbook.txt
├── README.md
└── requirements.txt
```

## Testing

Run the test suite with:

```bash
pytest -q
```

## Notes

- The app gracefully handles invalid requests and missing information.
- The API is ready for future n8n integration because it accepts JSON input and returns JSON output.
- If no handbook file is supplied, the application creates a default in-memory handbook sample for testing and demonstration.
- The system uses TF-IDF (Term Frequency-Inverse Document Frequency) embeddings for semantic retrieval, stored in an in-memory vector database.
- TF-IDF embeddings are generated using pure Python and NumPy, avoiding external ML library dependencies that may be blocked by system policies.
- Answer generation uses rule-based pattern matching with context extraction, ensuring answers are based only on handbook content.
