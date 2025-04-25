import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import nest_asyncio
import openai
from dotenv import load_dotenv
from index.utils import sync_repo
from llama_index.core import (
    Settings, PromptTemplate,
    StorageContext, load_index_from_storage
)
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, BatchEvalRunner
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.retrievers.bm25 import BM25Retriever

from main import load_index  # fallback loader for first-run logic

# Setup
nest_asyncio.apply()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "OPENAI-API-KEY")

# Constants
INDEX_STORAGE    = Path(os.getenv("INDEX_STORAGE", "../.cache/index_storage"))
QUESTIONS_CACHE  = Path(os.getenv("QUESTIONS_CACHE", "eval_questions.json"))
OUT_OF_SCOPE_TH  = -5
ASSISTANT_MODEL  = os.getenv("ASSISTANT_MODEL", "gpt-4.1-nano")
REPO_URL         = os.getenv("REPO_URL", "https://github.com/vanna-ai/vanna.git")
LOCAL_PATH       = os.getenv("LOCAL_PATH", "../.cache/vanna")
LAST_SHA_PATH    = os.getenv("LAST_SHA_PATH", "../.cache/last_indexed.sha")

# Configure logging
logging.basicConfig(level=logging.INFO)
# Configure LlamaIndex
Settings.llm = AzureOpenAI(api_version="2024-12-01-preview",
    azure_endpoint="https://skryp-m9y44j2k-eastus2.cognitiveservices.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                           model=ASSISTANT_MODEL,
                           deployment_name="gpt-4.1-nano")
Settings.embed_model = AzureOpenAIEmbedding(api_version="2024-12-01-preview",
    azure_endpoint="https://skryp-m9y44j2k-eastus2.cognitiveservices.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                           model="text-embedding-ada-002",
                           deployment_name="text-embedding-ada-002")


class HybridRetriever(BaseRetriever):
    """Combines BM25 and Vector retrieval results."""

    def __init__(self, bm25_retriever, vector_retriever):
        super().__init__()
        self.bm25 = bm25_retriever
        self.vec = vector_retriever

    def _retrieve(self, query_bundle):
        nodes = self.bm25.retrieve(query_bundle) + self.vec.retrieve(query_bundle)
        # Deduplicate by node ID
        return list({node.node_id: node for node in nodes}.values())


def load_or_build_index():
    """Load persisted index or rebuild if repo changed."""
    changed = sync_repo(REPO_URL, LOCAL_PATH, LAST_SHA_PATH)

    if not changed and INDEX_STORAGE.exists():
        logging.info("No changes detected — loading persisted index.")
        storage = StorageContext.from_defaults(persist_dir=str(INDEX_STORAGE))
        return load_index_from_storage(storage)

    logging.info("Changes detected or first run — building index.")
    index = load_index()
    index.storage_context.persist(persist_dir=str(INDEX_STORAGE))
    return index


def load_or_generate_questions(documents) -> List[str]:
    """Load cached evaluation questions or generate and cache them."""
    if QUESTIONS_CACHE.exists():
        logging.info("Loading cached evaluation questions.")
        return json.loads(QUESTIONS_CACHE.read_text())

    logging.info("Generating evaluation questions and caching.")
    sampled_docs = random.sample(documents, k=10)
    generator = RagDatasetGenerator.from_documents(sampled_docs)
    questions = generator.generate_questions_from_nodes().to_pandas()["query"].tolist()

    QUESTIONS_CACHE.write_text(json.dumps(questions, indent=2))
    return questions


async def run_batch_eval(qe, eval_qs) -> Tuple[List, dict]:
    """Run batch evaluation of queries asynchronously."""
    responses = [await qe.aquery(query) for query in eval_qs]

    runner = BatchEvalRunner(
        {"faithfulness": FaithfulnessEvaluator(), "relevancy": RelevancyEvaluator()},
        workers=20
    )
    eval_results = await runner.aevaluate_queries(
        query_engine=qe,
        queries=eval_qs
    )
    return responses, eval_results


def evaluate_response_time_and_accuracy():
    """Main evaluation routine."""
    # Load index and documents
    index = load_or_build_index()
    docs = list(index.docstore.docs.values())
    eval_questions = load_or_generate_questions(docs)[:1]

    # Setup retriever and query engine
    bm25_retriever = BM25Retriever.from_defaults(nodes=docs, similarity_top_k=20)
    vector_retriever = index.as_retriever(similarity_top_k=20)
    hybrid_retriever = HybridRetriever(bm25_retriever, vector_retriever)

    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
        text_qa_template=PromptTemplate(
            "Context:\n{context_str}\n---\nAnswer based only on this context.\n"
            "Query: {query_str}\nAnswer:"
        ),
    )

    # Perform evaluation
    start_time = time.time()
    responses, results = asyncio.run(run_batch_eval(query_engine, eval_questions))
    total_time = time.time() - start_time

    # Calculate averages
    n_queries = len(eval_questions)
    avg_time = total_time / n_queries
    avg_faithfulness = sum(r.passing for r in results["faithfulness"]) / n_queries
    avg_relevancy = sum(r.passing for r in results["relevancy"]) / n_queries

    print(f"Average Response Time: {avg_time:.2f}s")
    print(f"Average Faithfulness:  {avg_faithfulness:.2f}")
    print(f"Average Relevancy:     {avg_relevancy:.2f}")

    # Save detailed results
    detailed_results = []

    for query, response, faith, relev in zip(eval_questions, responses, results["faithfulness"], results["relevancy"]):
        top_scores = [float(node.score) for node in response.source_nodes[:3]]
        response_text = response.response

        if not any(score > OUT_OF_SCOPE_TH for score in top_scores):
            response_text = "Sorry, the question is out of scope."
            faith.passing = False
            relev.passing = False

        detailed_results.append({
            "question": query,
            "response": response_text,
            "response_time": avg_time,
            "faithfulness": faith.passing,
            "relevancy": relev.passing,
            "top_scores": top_scores,
        })

    Path("evaluation_results_batch.json").write_text(json.dumps(detailed_results, indent=2))
    logging.info("Saved detailed batch evaluation results.")


if __name__ == "__main__":
    evaluate_response_time_and_accuracy()
