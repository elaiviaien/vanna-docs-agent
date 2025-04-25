import nest_asyncio
import os
import time
import logging
import json
import tiktoken
from uuid import uuid4
from dotenv import load_dotenv
from fastapi import FastAPI
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings, PromptTemplate,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.node_parser import CodeSplitter, SemanticSplitterNodeParser, JSONNodeParser, TokenTextSplitter
from llama_index.core.evaluation import DatasetGenerator, FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
import openai
from llama_index.core.evaluation import BatchEvalRunner

from index.utils import sync_repo
from main import load_index

# Apply nest_asyncio for running in Jupyter or similar environments
nest_asyncio.apply()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "OPENAI-API-KEY")

# Constants from main.py
OUT_OF_SCOPE_THRESHOLD = -5
INDEX_STORAGE = os.environ.get("INDEX_STORAGE", "../.cache/index_storage")
ASSISTANT_MODEL = os.getenv('ASSISTANT_MODEL', 'gpt-4.1-nano')

# Set global settings
Settings.llm = OpenAI(temperature=0, model="gpt-4.1-nano")
Settings.embed_model = OpenAIEmbedding()


REPO_URL = os.environ.get("REPO_URL", "https://github.com/vanna-ai/vanna.git")
LOCAL_PATH = os.environ.get("LOCAL_PATH", "../.cache/vanna")
LAST_SHA_PATH = os.environ.get("LAST_SHA_PATH", "../.cache/last_indexed.sha")


def load_docs():
    changed = sync_repo(REPO_URL, LOCAL_PATH, LAST_SHA_PATH)

    # Choose which files to index
    if not changed:
        logging.info("Full index: scanning all files.")
        files_to_idx = [os.path.join(dp, f)
                        for dp, _, fs in os.walk(LOCAL_PATH)
                        for f in fs]
    else:
        files_to_idx = changed
    documents = SimpleDirectoryReader(
        input_files=files_to_idx,
        recursive=True
    ).load_data()

    # Add metadata and compute line numbers
    for doc in documents:
        full_text = doc.text
        start_idx = 0
        doc.metadata = {
            "file_path": doc.metadata.get("file_name", "unknown")
        }

    py_docs = [file for file in documents if file.metadata.get("file_path").endswith(".py")]
    json_docs = [file for file in documents if file.metadata.get("file_path").endswith(".json")]
    other_docs = [file for file in documents if
                  not file.metadata.get("file_path").endswith(".py") and not file.metadata.get("file_path").endswith(
                      ".json")]
    return py_docs + json_docs + other_docs

# Generate evaluation questions

data_generator = RagDatasetGenerator.from_documents(load_docs()[:10])
eval_questions = data_generator.generate_questions_from_nodes()
eval_questions = eval_questions.to_pandas()["query"].tolist()

# Define evaluators
faithfulness_gpt4 = FaithfulnessEvaluator()
relevancy_gpt4 = RelevancyEvaluator()


# Adapted from main.py: HybridRetriever
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


# Evaluation function
def evaluate_response_time_and_accuracy():
    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0
    num_questions = len(eval_questions)
    results = []

    # Create index with given chunk size
    index = load_index()

    # Setup query engine (from main.py)
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

    # Set up reranker
    rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5
    )

    # Create query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[rerank],
        text_qa_template=qa_prompt_tmpl,
    )

    # Evaluate each question
    for question in eval_questions:
        start_time = time.time()
        response = query_engine.query(question)
        elapsed_time = time.time() - start_time

        # Check if response is out of scope
        top_scores = [float(node.score) for node in response.source_nodes[:3]]
        if not any(score > OUT_OF_SCOPE_THRESHOLD for score in top_scores):
            faithfulness_result = False
            relevancy_result = False
            response_text = "Sorry, the question is out of scope."
        else:
            faithfulness_result = faithfulness_gpt4.evaluate_response(
                response=response
            ).passing
            relevancy_result = relevancy_gpt4.evaluate_response(
                query=question, response=response
            ).passing
            response_text = response.response

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result

        # Store result for this question
        result = {
            "question": question,
            "response": response_text,
            "response_time": elapsed_time,
            "faithfulness": faithfulness_result,
            "relevancy": relevancy_result,
            "top_scores": top_scores
        }
        results.append(result)

    # Save results to a JSON file
    output_file = f"evaluation_results_chunk.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved evaluation results to {output_file}")

    average_response_time = float(total_response_time) / num_questions
    average_faithfulness = float(total_faithfulness) / num_questions
    average_relevancy = float(total_relevancy) / num_questions

    return average_response_time, average_faithfulness, average_relevancy


avg_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy()
print(
        f"Average Response time: {avg_time:.2f}s, Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")