# Agentic-RAG-with-LangChain (Groq)

This project has been migrated from OpenAI to Groq for chat inference and to free local HuggingFace embeddings for vector search.

## What changed
- Chat model: `ChatOpenAI` ➜ `ChatGroq` using Groq's Llama 3 model.
- Embeddings: `OpenAIEmbeddings` ➜ `HuggingFaceEmbeddings` (all-MiniLM-L6-v2).
- Prompt: OpenAI-specific agent prompt ➜ provider-agnostic `structured-chat-agent`.

## Prerequisites
- Python 3.11+ (3.13 supported)
- A Groq API key: https://console.groq.com
- Supabase project with a `documents` table and `match_documents` RPC (as already used here)

## Setup
1. Create a `.env` file (or edit the existing one) and set:
   - `GROQ_API_KEY="<your_groq_api_key>"`
   - `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The first run will download the HuggingFace embedding model automatically.

## Ingest your documents
Place PDFs under the `documents/` folder, then run:

```powershell
python ingest_in_db.py
```

## Run a console example
```powershell
python agentic_rag.py
```

## Run the Streamlit app
```powershell
streamlit run agentic_rag_streamlit.py
```

## Notes
- If you previously used OpenAI, the old `OPENAI_API_KEY` is no longer required.
- You can switch the Groq model by changing the `model` parameter in `ChatGroq`.
- For higher-quality retrieval, consider a stronger embedding model like `BAAI/bge-small-en-v1.5` and set `normalize_embeddings=True`.

## Fix Supabase vector dimensions (384 vs 1536)
If you see an error like `expected 1536 dimensions, not 384` during ingestion, your Supabase table was created for OpenAI embeddings (1536-dim), while the new HuggingFace model outputs 384-dim vectors.

You have two options:

1) Recommended: Change the Supabase column to 384-dim and re-ingest
    - In Supabase SQL editor:

```sql
-- Adjust to your schema/table/column names if different
BEGIN;
-- If you can afford to drop the existing embeddings column:
ALTER TABLE documents DROP COLUMN embedding;
ALTER TABLE documents ADD COLUMN embedding vector(384);
COMMIT;

-- If you have an RPC for similarity search, ensure it accepts vector(384)
-- Example:
CREATE OR REPLACE FUNCTION match_documents(
   query_embedding vector(384),
   match_count int DEFAULT 5
)
RETURNS TABLE(id uuid, content text, metadata jsonb, similarity float)
LANGUAGE SQL STABLE AS $$
   SELECT
      d.id,
      d.content,
      d.metadata,
      1 - (d.embedding <=> query_embedding) AS similarity
   FROM documents d
   ORDER BY d.embedding <=> query_embedding
   LIMIT match_count;
$$;
```

Then re-run:

```powershell
python ingest_in_db.py
```

2) Keep the 1536-dim column and use a 1536-dim embedding model
    - Most free local models are 384/768/1024 dims. 1536-dim is specific to OpenAI’s `text-embedding-3-small`.
    - If you want to keep 1536 dims without OpenAI, you’ll likely need to change the table to 384 (option 1).
