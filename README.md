# VannaAI RAG

## Overview

This project creates a system to answer questions based on the content of
the [Vanna AI GitHub repository](https://github.com/vanna-ai/vanna/tree/main).

## Performance

- **Indexing**: Takes about 50 seconds.
- **Response Time**: Averages 6 seconds per query.
- **Accuracy**: Achieves high accuracy through advanced search and ranking techniques.

## Setup

### Prerequisites

- **Docker**: Installed and running.
- **Git**: For cloning the repository.
- **Environment Variables**: Create a `.env` file with:
  ```plaintext
  AZURE_OPENAI_API_KEY=<your_key>
  AZURE_OPENAI_ENDPOINT=<your_endpoint>
  AZURE_API_VERSION=2024-12-01-preview
  MODEL_NAME=gpt-4.1
  EMBEDDING_MODEL_NAME=text-embedding-ada-002
  REPO_URL=https://github.com/vanna-ai/vanna.git
  LOCAL_PATH=.cache/vanna
  LAST_SHA_PATH=.cache/last_indexed.sha
  INDEX_STORAGE=.cache/index_storage
  PORT=8000
  ```

### Docker Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/elaiviaien/vanna-docs-agent.git
   cd vanna-docs-agent
   ```
2. Build and run the Docker container:
   ```bash
   docker compose up -d
   ```

### Running Locally

1. Clone and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Set up `.env`.
3. Run the application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### Testing the API

1. Test in-scope query:
   ```bash
   curl -X GET "http://localhost:8000/query?query=What is the purpose of the Vanna AI repository?"
   ```
2. Test out-of-scope query:
   ```bash
   curl -X GET "http://localhost:8000/query?query=What is the capital of France?"
   ```
3. Run evaluations:
   ```bash
   python rag_evaluator.py
   ```

---

### Indexing

The system indexes text-based files from the Vanna AI repository, excluding binary files and
irrelevant directories like `.git` or `node_modules`. The process involves:

- **File Filtering**: Only files with text-based MIME types (e.g., `.py`, `.md`, `.json`) are selected, ensuring
  efficient processing.
- **Chunking**: Files are split into smaller segments using type-specific strategies:
    - **Python Files**: A chunking strategy leveraging Abstract Syntax Trees (AST) splits code into semantically
      meaningful units, targeting approximately 20 lines per chunk with 5 lines of overlap to preserve
      context ([LlamaIndex CodeSplitter](https://docs.llamaindex.ai/en/v0.10.19/api/llama_index.core.node_parser.CodeSplitter.html)).
    - **JSON Files**: A parser tailored to JSON structure extracts hierarchical segments, maintaining data integrity.
    - **Other Text Files**: A sentence-based splitter divides content like Markdown into chunks, preserving narrative
      flow.
- **Token Limits**: Chunks are capped at 8,192 tokens using OpenAI's tokenizer to ensure compatibility with the
  embedding model.
- **Embedding Generation**: Embeddings are created with Azure OpenAI's `text-embedding-ada-002` model and stored in a
  vector store for retrieval.

**Note***: Implemented incremental indexing that updates only changed files, reducing processing time for subsequent
runs.

### Question Answering

The system uses a hybrid retrieval approach combining BM25 (keyword-based search) and vector store retriever (semantic
search) to fetch relevant content from the Vanna AI GitHub repository. This is ideal for repositories with diverse
content like code, documentation, and JSON files.

- **BM25 Retriever**: Matches exact terms (e.g., function names like sync_repo), excelling for precise, technical
  queries.
- **Vector Store Retriever**: Uses text-embedding-ada-002 embeddings for semantic understanding, handling broader
  queries (e.g., "How does indexing work?").

BM25 captures code-specific terms; semantic search handles documentation’s natural language.
Supports both specific (e.g., "Parameters of load_and_chunk") and general (e.g., "Purpose of Vanna AI") queries.
It combines precise matches with contextual relevance, ensuring comprehensive results.

**Post-processing:**

Results are merged, deduplicated, and re-ranked using a cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) to select
the top 5 nodes.
Limiting reranking to 5 nodes balances accuracy and speed.

### Out-of-Scope Handling

Out-of-scope questions are detected by analyzing retrieval scores. If the top 3 nodes' scores fall below a threshold (
-5), the query is classified as out-of-scope, triggering the response: "Sorry, the question is out of scope."

### Evaluation

The system is evaluated on:

- **Faithfulness**: Ensures answers are grounded in retrieved context.
- **Relevancy**: Confirms answers address the query.
- **Out-of-Scope Accuracy**: Verifies correct identification of irrelevant questions.
- **Response Time**: Measures query processing efficiency.

Evaluation involves 20 in-scope and 20 out-of-scope questions, with results saved in JSON files.

The use of automated evaluators with a out-of-scope assessor enhances evaluation precision, providing a scalable way to
test the system’s behavior across varied inputs.

### Technology Stack

- **Python 3.11**: Core programming language.
- **FastAPI**: High-performance API framework.
- **llama-index**: Indexing and retrieval framework.
- **Azure OpenAI**: GPT-4.1 for answers, `text-embedding-ada-002` for embeddings.
- **sentence-transformers**: Cross-encoder for reranking.
- **Docker**: Containerization for portability.

## Reranker note
Cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) was used to rerank the top 5 nodes retrieved by the BM25 and vector store retrievers. This model is designed for reranking tasks, where it takes pairs of query and document embeddings and predicts a relevance score.
But with it's dependency on the `sentence-transformers` library, it led to huge docker image size, so for now was taken desicion to use LLMReranker.


## Possible improvements

**Custom Chunkers with Line Tracking**

Custom chunkers that parse code files (e.g., Python) using Abstract Syntax Trees (AST) to split content into semantically meaningful units like functions or classes, while recording start and end line numbers. This enables precise GitHub links in responses, pointing to specific line ranges (e.g., `[https://github.com/vanna-ai/vanna/blob/main/indexing.py#L73-L108](https://github.com/vanna-ai/vanna/blob/main/indexing.py#L73-L108)`).

**Nemo Guardrails for Out-of-Scope Handling**

Integrating [Nemo Guardrails](https://docs.nvidia.com/nemo/guardrails/index.html), a framework for adding safety and alignment to LLMs, to enhance out-of-scope question detection. Defining rules to identify queries unrelated to the repository and trigger a standardized response (e.g., “Sorry, the question is out of scope”).

**Multi-Turn Conversations**

Enabling the system to maintain context across multiple queries, supporting follow-up questions. I would do it with llama-index `ChatEngine`

**Vector DB**

For storing embeddings, if I had more time, I would use a vector database like Weaviate instead of a local vector store. This would improve scalability and retrieval speed, especially for larger datasets. It would also allow for more advanced features like real-time updates and distributed storage.

Weaviate supports both vector and keyword search, which fits well with your hybrid retrieval approach. It also has intefgrations with llama-index and Azure OpenAI, making it easier to implement.

