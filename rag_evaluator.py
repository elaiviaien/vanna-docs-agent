"""
RAG System Evaluation Module

This module provides comprehensive evaluation for Retrieval-Augmented Generation (RAG) systems,
including metrics for faithfulness, relevancy, response time, and scope boundary detection.
"""

import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import nest_asyncio
import numpy as np
from dotenv import load_dotenv
from llama_index.core import (
    Settings, PromptTemplate, StorageContext, load_index_from_storage, VectorStoreIndex
)
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, BatchEvalRunner, BaseEvaluator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from pydantic import BaseModel

from utils import sync_repo

# Initialize environment and async
nest_asyncio.apply()
load_dotenv()

# Common constants
INDEX_STORAGE = Path(os.getenv("INDEX_STORAGE", "../.cache/index_storage"))
GENERAL_RESULTS_PATH = "evaluation_results.json"
SCOPE_RESULTS_PATH = "out_of_scope_evaluation_results.json"
QUESTIONS_CACHE = "eval_questions.json"
IN_SCOPE_QUESTIONS_CACHE = "in_scope_questions.json"
OUT_OF_SCOPE_QUESTIONS_CACHE = "out_of_scope_questions.json"

# Configuration settings
OUT_OF_SCOPE_THRESHOLD = float(os.getenv("OUT_OF_SCOPE_THRESHOLD", "-5"))
NUM_SAMPLE_DOCS = int(os.getenv("NUM_SAMPLE_DOCS", "10"))
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "20"))
NUM_IN_SCOPE_QUESTIONS = int(os.getenv("NUM_IN_SCOPE_QUESTIONS", "20"))
NUM_OUT_OF_SCOPE_QUESTIONS = int(os.getenv("NUM_OUT_OF_SCOPE_QUESTIONS", "20"))
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "20"))
TOP_N_RERANK = int(os.getenv("TOP_N_RERANK", "5"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Repository settings
REPO_URL = os.getenv("REPO_URL", "https://github.com/vanna-ai/vanna.git")
LOCAL_PATH = os.getenv("LOCAL_PATH", "../.cache/vanna")
LAST_SHA_PATH = os.getenv("LAST_SHA_PATH", ".cache/last_indexed.sha")

# Azure OpenAI settings
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://skryp-m9y44j2k-eastus2.cognitiveservices.azure.com/")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-nano")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_index(self):
    print(f"[{time.time()}] Loading index...")
    index_start = time.time()
    try:
        storage_ctx = StorageContext.from_defaults(persist_dir=self.index_storage)
        self.index = load_index_from_storage(storage_context=storage_ctx)
        index_time = time.time() - index_start
        print(f"[{time.time()}] Index loaded in {index_time:.2f} seconds")
        return True
    except (FileNotFoundError, json.JSONDecodeError):
        logging.error("Index not found or invalid. Please build it first.")
        return False
class OutOfScopeQuestions(BaseModel):
    """Pydantic model for structured output of out-of-scope questions."""
    questions: List[str]


class OutOfScopeEvaluator(BaseEvaluator):
    """Evaluator for assessing out-of-scope question handling."""

    def __init__(self, threshold: float = OUT_OF_SCOPE_THRESHOLD):
        super().__init__()
        self.threshold = threshold

    def _get_prompts(self) -> Dict[str, Any]:
        return {}

    def _update_prompts(self, prompts: Dict[str, Any]) -> None:
        pass

    async def _aevaluate(self, query: str, response: Any, contexts: List[str], **kwargs) -> Dict[str, Any]:
        top_scores = [node.score for node in response.source_nodes[:3]] if response.source_nodes else []
        response_text = response.response.lower()
        expected_response = "out of scope"

        is_out_of_scope = not any(score > self.threshold for score in top_scores)
        is_correct_response = expected_response in response_text

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
        return await self._aevaluate(query, response, contexts, **kwargs)


class HybridRetriever(BaseRetriever):
    """Combines BM25 and Vector retrieval results for improved retrieval."""

    def __init__(self, bm25_retriever: BM25Retriever, vector_retriever: BaseRetriever):
        super().__init__()
        self.bm25 = bm25_retriever
        self.vec = vector_retriever

    def _retrieve(self, query_bundle):
        nodes = self.bm25.retrieve(query_bundle) + self.vec.retrieve(query_bundle)
        # Deduplicate by node ID
        return list({node.node_id: node for node in nodes}.values())


class RagEvaluator:
    """Main class for evaluating RAG systems with comprehensive metrics."""

    def __init__(self):
        self._setup_llm()
        self.index = None

    def _setup_llm(self):
        """Configure LLM and embedding models for evaluation."""
        if not AZURE_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")

        Settings.llm = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            model=MODEL_NAME,
            deployment_name=MODEL_NAME
        )

        Settings.embed_model = AzureOpenAIEmbedding(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            model=EMBEDDING_MODEL,
            deployment_name=EMBEDDING_MODEL
        )

    def load_index(self) -> Optional[VectorStoreIndex]:
        """Load persisted index or rebuild if repository changed."""
        try:
            changed = sync_repo(REPO_URL, LOCAL_PATH, LAST_SHA_PATH)

            if not changed and INDEX_STORAGE.exists():
                logger.info("No changes detected — loading persisted index.")
                storage = StorageContext.from_defaults(persist_dir=str(INDEX_STORAGE))
                self.index = load_index_from_storage(storage)
            else:
                logger.info("Changes detected or first run — building index.")
                self.index = fallback_load_index()

                # Ensure directory exists
                INDEX_STORAGE.parent.mkdir(parents=True, exist_ok=True)
                self.index.storage_context.persist(persist_dir=str(INDEX_STORAGE))

            return self.index
        except Exception as e:
            logger.error(f"Error loading or building index: {e}")
            raise

    def create_query_engine(self, documents: List) -> RetrieverQueryEngine:
        """Create a hybrid retrieval query engine with reranking."""
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=documents,
            similarity_top_k=TOP_K_RETRIEVE
        )
        vector_retriever = self.index.as_retriever(similarity_top_k=TOP_K_RETRIEVE)
        hybrid_retriever = HybridRetriever(bm25_retriever, vector_retriever)

        reranker = SentenceTransformerRerank(
            model=RERANKER_MODEL,
            top_n=TOP_N_RERANK
        )

        return RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=[reranker],
            text_qa_template=PromptTemplate(
                "Context:\n{context_str}\n---\nAnswer based only on this context.\n"
                "Query: {query_str}\nAnswer:"
            ),
        )

    @staticmethod
    def load_or_generate_questions(documents: List, cache_path: Path,
                                   num_samples: int = NUM_SAMPLE_DOCS,
                                   max_questions: int = MAX_QUESTIONS) -> List[str]:
        """Load cached evaluation questions or generate and cache them."""
        try:
            if cache_path.exists():
                logger.info(f"Loading cached questions from {cache_path}")
                return json.loads(cache_path.read_text())[:max_questions]

            logger.info(f"Generating questions from {num_samples} documents")
            if len(documents) < num_samples:
                num_samples = len(documents)
                logger.warning(f"Only {num_samples} documents available for sampling")

            sampled_docs = random.sample(documents, k=num_samples)
            generator = RagDatasetGenerator.from_documents(sampled_docs)
            questions = generator.generate_questions_from_nodes().to_pandas()["query"].tolist()

            # Ensure directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(questions, indent=2))
            return questions[:max_questions]
        except Exception as e:
            logger.error(f"Error loading or generating questions: {e}")
            raise

    @staticmethod
    async def generate_out_of_scope_questions(
            cache_path: Path = OUT_OF_SCOPE_QUESTIONS_CACHE,
            num_questions: int = NUM_OUT_OF_SCOPE_QUESTIONS
    ) -> List[str]:
        """Generate out-of-scope questions using an LLM with structured output."""
        try:
            if cache_path.exists():
                logger.info(f"Loading cached out-of-scope questions from {cache_path}")
                return json.loads(cache_path.read_text())[:num_questions]

            logger.info(f"Generating {num_questions} out-of-scope questions")

            prompt = PromptTemplate(
                f"Generate {num_questions} questions that are completely unrelated to software "
                f"development, GitHub repositories, or AI codebases like Vanna AI. Focus on topics like geography, "
                f"cooking, history, or health. Return the questions as a JSON object with a single key 'questions' "
                f"containing a list of strings."
            )

            parser = PydanticOutputParser(output_cls=OutOfScopeQuestions)
            llm = AzureOpenAI(
                api_version=AZURE_API_VERSION,
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                model=MODEL_NAME,
                deployment_name=MODEL_NAME)
            try:
                response = await llm.astructured_predict(
                    output_cls=OutOfScopeQuestions,
                    prompt=prompt,
                    parser=parser
                )
                questions = response.questions
            except Exception as e:
                logger.error(f"Failed to generate structured output: {e}. Using fallback questions.")
                questions = [
                                "What is the capital of Brazil?",
                                "How do you bake a vanilla cake?",
                                "Who was the first president of the United States?",
                                "What are the health benefits of yoga?",
                                "What is the largest ocean on Earth?"
                            ][:num_questions]

            # Ensure directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(questions, indent=2))
            return questions[:num_questions]
        except Exception as e:
            logger.error(f"Error generating out-of-scope questions: {e}")
            raise

    @staticmethod
    def process_source_nodes(response) -> List[Dict[str, Any]]:
        """Extract and process source nodes from response."""
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
        return sources

    async def evaluate_general_rag(self) -> Dict[str, Any]:
        """Evaluate general RAG metrics including faithfulness and relevancy."""
        logger.info("Starting general RAG evaluation")

        # Load index and documents
        if not self.index:
            self.index = self.load_index()

        docs = list(self.index.docstore.docs.values())
        eval_questions = self.load_or_generate_questions(docs, QUESTIONS_CACHE)

        # Create query engine
        query_engine = self.create_query_engine(docs)

        # Perform evaluation
        start_time = time.time()
        responses, results = await self.run_batch_eval(query_engine, eval_questions)
        total_time = time.time() - start_time

        # Calculate metrics
        n_queries = len(eval_questions)
        avg_time = total_time / n_queries
        avg_faithfulness = sum(r.passing for r in results["faithfulness"]) / n_queries
        avg_relevancy = sum(r.passing for r in results["relevancy"]) / n_queries

        # Log summary results
        logger.info(f"Average Response Time: {avg_time:.2f}s")
        logger.info(f"Average Faithfulness:  {avg_faithfulness:.2f}")
        logger.info(f"Average Relevancy:     {avg_relevancy:.2f}")

        # Save detailed results
        detailed_results = []
        for query, response, faith, relev in zip(
                eval_questions, responses,
                results["faithfulness"], results["relevancy"]):

            top_scores = [float(node.score) for node in response.source_nodes[:3]] if response.source_nodes else []
            response_text = response.response

            # Check if question is out of scope
            if not top_scores or not any(score > OUT_OF_SCOPE_THRESHOLD for score in top_scores):
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
                "sources": self.process_source_nodes(response)
            })

        # Ensure directory exists
        GENERAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        GENERAL_RESULTS_PATH.write_text(json.dumps(detailed_results, indent=2))
        logger.info(f"Saved detailed evaluation results to {GENERAL_RESULTS_PATH}")

        return {
            "avg_response_time": avg_time,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevancy": avg_relevancy,
            "detailed_results": detailed_results
        }

    async def evaluate_scope_boundaries(self) -> Dict[str, Any]:
        """Evaluate the system's ability to handle in-scope and out-of-scope questions."""
        logger.info("Starting scope boundary evaluation")

        # Load index and documents
        if not self.index:
            self.index = self.load_index()

        documents = list(self.index.docstore.docs.values())

        # Get in-scope and out-of-scope questions
        in_scope_questions = self.load_or_generate_questions(
            documents, IN_SCOPE_QUESTIONS_CACHE,
            max_questions=NUM_IN_SCOPE_QUESTIONS
        )
        out_of_scope_questions = await self.generate_out_of_scope_questions()

        # Setup query engine
        query_engine = self.create_query_engine(documents)

        # Combine questions and expected results
        all_questions = in_scope_questions + out_of_scope_questions
        is_out_of_scope = [False] * len(in_scope_questions) + [True] * len(out_of_scope_questions)

        # Initialize evaluator
        oos_evaluator = OutOfScopeEvaluator(threshold=OUT_OF_SCOPE_THRESHOLD)

        # Run evaluation
        results = []
        response_times = []

        for query, expected_oos in zip(all_questions, is_out_of_scope):
            t0 = time.time()
            response = await query_engine.aquery(query)
            response_time = time.time() - t0
            response_times.append(response_time)

            # Process results
            sources = self.process_source_nodes(response)
            top_scores = [node_with_score.score for node_with_score in response.source_nodes[:3]]

            # Handle out-of-scope detection
            is_oos_detected = not any(score > OUT_OF_SCOPE_THRESHOLD for score in top_scores) if top_scores else True
            if is_oos_detected:
                response.response = "Sorry, the question is out of scope."

            # Evaluate
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
        out_of_scope_accuracy = np.mean([r["out_of_scope_score"] for r in results])
        out_of_scope_passing_rate = np.mean([r["out_of_scope_passing"] for r in results])

        # Log results
        logger.info(f"Average Response Time: {avg_response_time:.2f}s")
        logger.info(f"Out-of-Scope Accuracy: {out_of_scope_accuracy:.2f}")
        logger.info(f"Out-of-Scope Passing Rate: {out_of_scope_passing_rate:.2f}")

        # Prepare output
        output = {
            "summary": {
                "average_response_time": avg_response_time,
                "out_of_scope_accuracy": out_of_scope_accuracy,
                "out_of_scope_passing_rate": out_of_scope_passing_rate,
                "total_questions": len(all_questions),
                "in_scope_questions": len(in_scope_questions),
                "out_of_scope_questions": len(out_of_scope_questions)
            },
            "detailed_results": results
        }

        # Save results
        SCOPE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCOPE_RESULTS_PATH.write_text(json.dumps(output, indent=2))
        logger.info(f"Saved scope evaluation results to {SCOPE_RESULTS_PATH}")

        return output

    @staticmethod
    async def run_batch_eval(query_engine: RetrieverQueryEngine,
                             eval_questions: List[str]) -> Tuple[List, Dict]:
        """Run batch evaluation of queries asynchronously."""
        try:
            logger.info(f"Running batch evaluation on {len(eval_questions)} questions")
            responses = [await query_engine.aquery(query) for query in eval_questions]

            runner = BatchEvalRunner(
                {"faithfulness": FaithfulnessEvaluator(), "relevancy": RelevancyEvaluator()},
                workers=min(20, len(eval_questions))
            )

            eval_results = await runner.aevaluate_queries(
                query_engine=query_engine,
                queries=eval_questions
            )
            return responses, eval_results
        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}")
            raise

    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run all evaluation types and return combined results."""
        try:
            logger.info("Starting comprehensive RAG evaluation")

            # Ensure we have an index
            if not self.index:
                self.index = self.load_index()

            # Run both evaluation types
            general_results = await self.evaluate_general_rag()
            scope_results = await self.evaluate_scope_boundaries()

            # Combine results
            return {
                "general_evaluation": general_results,
                "scope_boundary_evaluation": scope_results,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error during comprehensive evaluation: {e}")
            raise


async def main():
    """Main entry point."""
    evaluator = RagEvaluator()
    await evaluator.run_comprehensive_evaluation()


if __name__ == "__main__":
    asyncio.run(main())
