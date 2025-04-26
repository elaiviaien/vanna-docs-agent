import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import Response
from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from rag_evaluator import run_rag_evaluator
from rag_service import RAGService
from utils import setup_logging

load_dotenv()
setup_logging(log_file="logs/app.log")
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service

    MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4.1')
    EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'text-embedding-ada-002')
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

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

    INDEX_STORAGE = os.environ.get("INDEX_STORAGE", ".cache/index_storage")
    LOCAL_PATH = os.environ.get("LOCAL_PATH", ".cache/vanna")

    logging.info(f"Initializing RAG service")
    rag_service = RAGService(index_storage=INDEX_STORAGE, local_path=LOCAL_PATH)

    if rag_service.load_index():
        rag_service.setup_query_engine()
        logging.info(f"RAG service initialized successfully")
    else:
        logging.error(f"Failed to initialize RAG service")

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/query")
async def query_index(query: str):
    global rag_service

    if not rag_service or not rag_service.query_engine:
        return {"error": "RAG service not initialized properly"}

    return rag_service.process_query(query)


@app.get("/evaluate")
async def evaluate_index(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_rag_evaluator)

    return Response(status_code=202,
                    content="Evaluation started in background.Check logs for progress and `evaluation_data` dir for results.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
