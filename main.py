import json
import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from llama_index.core import PromptTemplate, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.retrievers.bm25 import BM25Retriever

load_dotenv(override=True)
OUT_OF_SCOPE_THRESHOLD = -5

app = FastAPI()

INDEX_STORAGE = os.environ.get("INDEX_STORAGE", ".cache/index_storage")
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4.1')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'text-embedding-ada-002')
LOCAL_PATH = os.environ.get("LOCAL_PATH", ".cache/vanna")
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


def load_index():
    try:
        storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_STORAGE)
        index = load_index_from_storage(storage_context=storage_ctx)
        return index
    except (FileNotFoundError, json.JSONDecodeError):
        logging.error("Index not found or invalid. Please build it first.")
        return None


class HybridRetriever(BaseRetriever):
    def __init__(self, bm25_retriever, vector_retriever):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle):
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        all_nodes = bm25_nodes + vector_nodes
        unique_nodes = {node.node_id: node for node in all_nodes}
        return list(unique_nodes.values())


def get_github_url(node):
    file_path = node.metadata.get("file_path")
    if file_path:
        relative_path = os.path.relpath(file_path, LOCAL_PATH)
        url = f"https://github.com/vanna-ai/vanna/blob/main/{relative_path}"
        return url
    return None


@app.get("/query")
async def query_index(query: str):
    index = load_index()
    if index:
        # Define custom prompt for out-of-scope handling
        qa_prompt_tmpl = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

        # Set up retrievers
        nodes = list(index.docstore.docs.values())
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)
        vector_retriever = index.as_retriever(similarity_top_k=20)
        hybrid_retriever = HybridRetriever(bm25_retriever, vector_retriever)

        # Set up re-ranking
        rerank = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=5
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=[rerank],
            text_qa_template=qa_prompt_tmpl,
        )

        # Measure query time
        start_time = time.time()
        response = query_engine.query(query)
        end_time = time.time()
        query_time = end_time - start_time
        logging.info(f"Query time: {query_time:.2f} seconds")

        # Extract sources
        detailed_sources = []
        for node_with_score in response.source_nodes:
            node = node_with_score.node
            score = node_with_score.score
            github_url = get_github_url(node)
            detailed_sources.append({
                "node_id": node.node_id,
                "score": float(score),
                "text": node.text,
                "metadata": node.metadata,
                "github_url": github_url
            })
        top_scores = [node.score for node in response.source_nodes[:3]]  # top 3 results

        if not any(score > OUT_OF_SCOPE_THRESHOLD for score in top_scores):
            return {
                "query": query,
                "gpt_response": "Sorry, the question is out of scope.",
                "sources": [],
                "query_time": query_time,
            }
        return {
            "query": query,
            "gpt_response": response.response,
            "sources": detailed_sources,
            "query_time": query_time
        }
    else:
        return {"error": "Index not loaded or invalid"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
