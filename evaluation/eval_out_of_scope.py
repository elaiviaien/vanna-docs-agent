import asyncio
import json
import logging
import os
import random
import time
from typing import List, Dict, Any

import nest_asyncio
import numpy as np
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from main import HybridRetriever, get_github_url

# Apply nested asyncio for scripts
nest_asyncio.apply()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Constants
INDEX_STORAGE = os.environ.get("INDEX_STORAGE", "../.cache/index_storage")
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", "gpt-4.1")
OUT_OF_SCOPE_THRESHOLD = -5
EVAL_RESULTS_FILE = "out_of_scope_evaluation_results.json"
LOCAL_PATH = os.environ.get("LOCAL_PATH", "../.cache/vanna")
IN_SCOPE_QUESTIONS_CACHE = "in_scope_questions.json"
OUT_OF_SCOPE_QUESTIONS_CACHE = "out_of_scope_questions.json"
NUM_IN_SCOPE_QUESTIONS = 20  # Number of in-scope questions to generate
NUM_OUT_OF_SCOPE_QUESTIONS = 20  # Number of out-of-scope questions to generate

# LlamaIndex settings
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

def load_index():
    """Load the persisted index."""
    try:
        storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_STORAGE)
        index = load_index_from_storage(storage_context=storage_ctx)
        logging.info("Index loaded successfully.")
        return index
    except (FileNotFoundError, json.JSONDecodeError):
        logging.error("Index not found or invalid. Please build it first using indexing.py.")
        return None

def load_or_generate_in_scope_questions(documents: List[Any]) -> List[str]:
    """Load or generate in-scope questions using RagDatasetGenerator."""
    if os.path.exists(IN_SCOPE_QUESTIONS_CACHE):
        logging.info("Loading cached in-scope questions.")
        with open(IN_SCOPE_QUESTIONS_CACHE, 'r') as f:
            return json.load(f)
    else:
        logging.info("Generating in-scope questions and caching.")
        # Sample documents to avoid excessive processing
        sampled_docs = random.sample(documents, k=min(20, len(documents)))
        generator = RagDatasetGenerator.from_documents(sampled_docs)
        questions = generator.generate_questions_from_nodes().to_pandas()["query"].tolist()
        # Limit to desired number of questions
        questions = questions[:NUM_IN_SCOPE_QUESTIONS]
        with open(IN_SCOPE_QUESTIONS_CACHE, 'w') as f:
            json.dump(questions, f, indent=2)
        logging.info(f"Generated and cached {len(questions)} in-scope questions.")
        return questions

import os
import json
import logging
from typing import List
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# Define Pydantic model for structured output
class OutOfScopeQuestions(BaseModel):
    questions: List[str]

async def generate_out_of_scope_questions() -> List[str]:
    """Generate out-of-scope questions using an LLM with structured output."""
    if os.path.exists(OUT_OF_SCOPE_QUESTIONS_CACHE):
        logging.info("Loading cached out-of-scope questions.")
        with open(OUT_OF_SCOPE_QUESTIONS_CACHE, 'r') as f:
            return json.load(f)
    else:
        logging.info("Generating out-of-scope questions and caching.")
        # Configure LLM
        llm = OpenAI(temperature=0.7, model="gpt-4-turbo")
        Settings.llm = llm

        # Define prompt with structured output instructions
        prompt = PromptTemplate(
            f"Generate {NUM_OUT_OF_SCOPE_QUESTIONS} questions that are completely unrelated to software development, "
            "GitHub repositories, or AI codebases like Vanna AI. Focus on topics like geography, cooking, history, or health. "
            "Return the questions as a JSON object with a single key 'questions' containing a list of strings."
        )

        # Set up structured output parser
        parser = PydanticOutputParser(output_cls=OutOfScopeQuestions)

        try:
            # Generate structured response
            response = await llm.astructured_predict(
                output_cls=OutOfScopeQuestions,
                prompt=prompt,
                parser=parser
            )
            questions = response.questions
        except Exception as e:
            logging.error(f"Failed to generate structured output: {e}. Using fallback questions.")
            questions = [
                "What is the capital of Brazil?",
                "How do you bake a vanilla cake?",
                "Who was the first president of the United States?",
                "What are the health benefits of yoga?",
                "What is the largest ocean on Earth?"
            ][:NUM_OUT_OF_SCOPE_QUESTIONS]

        # Cache the questions
        with open(OUT_OF_SCOPE_QUESTIONS_CACHE, 'w') as f:
            json.dump(questions, f, indent=2)
        logging.info(f"Generated and cached {len(questions)} out-of-scope questions.")
        return questions

# Custom Out-of-Scope Evaluator
class OutOfScopeEvaluator(BaseEvaluator):
    def __init__(self, threshold: float = OUT_OF_SCOPE_THRESHOLD):
        super().__init__()
        self.threshold = threshold

    def _get_prompts(self) -> Dict[str, Any]:
        """Return an empty dict since no prompts are used."""
        return {}

    def _update_prompts(self, prompts: Dict[str, Any]) -> None:
        """Do nothing since no prompts are used."""
        pass

    async def _aevaluate(self, query: str, response: Any, contexts: List[str], **kwargs) -> Dict[str, Any]:
        """Evaluate if the response correctly handles out-of-scope questions."""
        # Extract top scores from response source nodes
        top_scores = [node.score for node in response.source_nodes[:3]] if response.source_nodes else []
        response_text = response.response.lower()
        expected_response = "out of scope"

        # Check if the question is out-of-scope based on scores
        is_out_of_scope = not any(score > self.threshold for score in top_scores)
        is_correct_response = expected_response in response_text

        # Scoring logic
        if is_out_of_scope:
            score = 1.0 if is_correct_response else 0.0
            reason = (
                "Correctly identified as out-of-scope and responded appropriately."
                if is_correct_response
                else "Identified as out-of-scope but failed to respond with 'out of scope'."
            )
        else:
            score = 0.0 if is_correct_response else 1.0
            reason = (
                "Incorrectly identified as out-of-scope when it was in-scope."
                if is_correct_response
                else "Correctly identified as in-scope."
            )

        return {
            "passing": score >= 0.7,
            "score": score,
            "reason": reason
        }

    async def aevaluate(self, query: str, response: Any, contexts: List[str], **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for _aevaluate."""
        return await self._aevaluate(query, response, contexts, **kwargs)

async def evaluate_out_of_scope():
    """Evaluate the agent's handling of dynamically generated in-scope and out-of-scope questions."""
    index = load_index()
    if not index:
        return {"error": "Index not loaded or invalid"}

    # Load documents for in-scope question generation
    documents = list(index.docstore.docs.values())
    in_scope_questions = load_or_generate_in_scope_questions(documents)
    out_of_scope_questions = await generate_out_of_scope_questions()

    # Setup retrievers
    nodes = documents
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)
    vector_retriever = index.as_retriever(similarity_top_k=20)
    hybrid_retriever = HybridRetriever(bm25_retriever, vector_retriever)

    # Setup re-ranking
    rerank = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=5
    )

    # Setup query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[rerank],
    )

    # Combine in-scope and out-of-scope questions
    all_questions = in_scope_questions + out_of_scope_questions
    is_out_of_scope = [False] * len(in_scope_questions) + [True] * len(out_of_scope_questions)

    # Initialize evaluator
    oos_evaluator = OutOfScopeEvaluator(threshold=OUT_OF_SCOPE_THRESHOLD)

    # Run queries and evaluate
    results = []
    response_times = []
    for query, expected_oos in zip(all_questions, is_out_of_scope):
        t0 = time.time()
        response = await query_engine.aquery(query)
        response_time = time.time() - t0
        response_times.append(response_time)

        # Collect source information
        sources = []
        for node_with_score in response.source_nodes:
            node = node_with_score.node
            github_url = get_github_url(node)
            sources.append({
                "node_id": node.node_id,
                "score": float(node_with_score.score),
                "text": node.text,
                "metadata": node.metadata,
                "github_url": github_url
            })
        top_scores = [node_with_score.score for node_with_score in response.source_nodes[:3]]  # top 3 results
        if not any(score > OUT_OF_SCOPE_THRESHOLD for score in top_scores):
            response.response = "Sorry, the question is out of scope."
        # Evaluate out-of-scope handling
        eval_result = await oos_evaluator._aevaluate(
            query=query,
            response=response,
            contexts=[node.text for node in response.source_nodes]
        )
        results.append({
            "question": query,
            "response": response.response,
            "response_time": response_time,
            "is_out_of_scope_expected": expected_oos,
            "out_of_scope_score": eval_result["score"],
            "out_of_scope_passing": eval_result["passing"],
            "out_of_scope_reason": eval_result["reason"],
            "sources": sources
        })

    # Compute metrics
    avg_response_time = np.mean(response_times)
    median_response_time = np.median(response_times)
    p95_response_time = np.percentile(response_times, 95)
    oos_accuracy = np.mean([r["out_of_scope_score"] for r in results])
    oos_passing_rate = np.mean([r["out_of_scope_passing"] for r in results])

    # Log summary
    logging.info(f"Average Response Time: {avg_response_time:.2f}s")
    logging.info(f"Median Response Time: {median_response_time:.2f}s")
    logging.info(f"95th Percentile Response Time: {p95_response_time:.2f}s")
    logging.info(f"Out-of-Scope Accuracy: {oos_accuracy:.2f}")
    logging.info(f"Out-of-Scope Passing Rate: {oos_passing_rate:.2f}")

    # Save detailed results
    output = {
        "summary": {
            "average_response_time": avg_response_time,
            "median_response_time": median_response_time,
            "p95_response_time": p95_response_time,
            "out_of_scope_accuracy": oos_accuracy,
            "out_of_scope_passing_rate": oos_passing_rate,
            "total_questions": len(all_questions),
            "in_scope_questions": len(in_scope_questions),
            "out_of_scope_questions": len(out_of_scope_questions)
        },
        "detailed_results": results
    }

    with open(EVAL_RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    logging.info(f"Saved evaluation results to {EVAL_RESULTS_FILE}")

    return output

if __name__ == "__main__":
    asyncio.run(evaluate_out_of_scope())