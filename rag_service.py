import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple

from llama_index.core import PromptTemplate, Settings, StorageContext, load_index_from_storage
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

OUT_OF_SCOPE_THRESHOLD = -5

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

class RAGService:
    def __init__(self, index_storage: str, local_path: str):
        self.index_storage = index_storage
        self.local_path = local_path
        self.index = None
        self.query_engine = None

    def load_index(self):
        logging.info(f"Loading index...")
        index_start = time.time()
        try:
            storage_ctx = StorageContext.from_defaults(persist_dir=self.index_storage)
            self.index = load_index_from_storage(storage_context=storage_ctx)
            index_time = time.time() - index_start
            logging.info(f"Index loaded in {index_time:.2f} seconds")
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            logging.error("Index not found or invalid. Please build it first.")
            return False

    def setup_query_engine(self):
        if not self.index:
            return False

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
        nodes = list(self.index.docstore.docs.values())

        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)

        vector_retriever = self.index.as_retriever(similarity_top_k=20)

        hybrid_retriever = HybridRetriever(bm25_retriever, vector_retriever)

        rerank = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=5
        )

        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            node_postprocessors=[rerank],
            text_qa_template=qa_prompt_tmpl,
        )
        return True

    def get_github_url(self, node):
        file_path = node.metadata.get("file_path")
        if file_path:
            url = f"https://github.com/vanna-ai/vanna/blob/main/{file_path}"
            return url
        return None

    def process_query(self, query: str) -> Dict[str, Any]:

        if not self.query_engine:
            return {"error": "Query engine not initialized"}

        start_time = time.time()

        response = self.query_engine.query(query)

        end_time = time.time()
        query_time = end_time - start_time

        url_to_score = {}
        for node_with_score in response.source_nodes:
            node = node_with_score.node
            score = node_with_score.score
            github_url = self.get_github_url(node)

            if github_url:
                # Keep highest score for each URL
                if github_url not in url_to_score or score > url_to_score[github_url]:
                    url_to_score[github_url] = score

        # Sort unique URLs by score
        source_items = sorted(url_to_score.items(), key=lambda x: x[1], reverse=True)
        sources = [url for url, _ in source_items]

        top_scores = [node.score for node in response.source_nodes[:3]]

        if not any(score > OUT_OF_SCOPE_THRESHOLD for score in top_scores):
            return {
                "response": "Sorry, the question is out of scope.",
                "sources": [],
                "query_time": query_time,
            }

        return {
            "response": response.response,
            "sources": sources,
            "query_time": query_time
        }