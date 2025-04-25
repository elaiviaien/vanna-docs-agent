import json
import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

load_dotenv()
OUT_OF_SCOPE_THRESHOLD = -5

app = FastAPI()

INDEX_STORAGE = os.environ.get("INDEX_STORAGE", ".cache/index_storage")
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4.1')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'text-embedding-ada-002')
LOCAL_PATH = os.environ.get("LOCAL_PATH", ".cache/vanna")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

Settings.llm = AzureOpenAI(api_version=AZURE_API_VERSION,
                           azure_endpoint=AZURE_ENDPOINT,
                           api_key=AZURE_OPENAI_API_KEY,
                           model=MODEL_NAME,
                           deployment_name=MODEL_NAME)
Settings.embed_model = AzureOpenAIEmbedding(api_version=AZURE_API_VERSION,
                                            azure_endpoint=AZURE_ENDPOINT,
                                            api_key=AZURE_OPENAI_API_KEY,
                                            model=EMBEDDING_MODEL_NAME,
                                            deployment_name=EMBEDDING_MODEL_NAME)


def load_index():
    try:
        storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_STORAGE)
        index = load_index_from_storage(storage_context=storage_ctx)
        return index
    except (FileNotFoundError, json.JSONDecodeError):
        logging.error("Index not found or invalid. Please build it first.")
        return None


@app.get("/query")
async def query_index(query: str):
    index = load_index()
    if index:

        # Set up retrievers
        query_engine = index.as_query_engine()

        # Measure query time
        start_time = time.time()
        response = query_engine.query(query)
        end_time = time.time()
        query_time = end_time - start_time
        logging.info(f"Query time: {query_time:.2f} seconds")

        return {
            "query": query,
            "response": response.response,
        }
    else:
        return {"error": "Index not loaded or invalid"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
