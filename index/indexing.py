import json
import logging
import os
import time

import tiktoken
from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    SimpleDirectoryReader, Document,
)
from llama_index.core.node_parser import CodeSplitter, SemanticSplitterNodeParser, \
    JSONNodeParser, TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from utils import filter_files, sync_repo,setup_logging

setup_logging()
load_dotenv()

REPO_URL = os.environ.get("REPO_URL", "https://github.com/vanna-ai/vanna.git")
LOCAL_PATH = os.environ.get("LOCAL_PATH", "./.cache/vanna")
LAST_SHA_PATH = os.environ.get("LAST_SHA_PATH", "./.cache/last_indexed.sha")
INDEX_STORAGE = os.environ.get("INDEX_STORAGE", "./.cache/index_storage")


def enforce_max_tokens(nodes, max_tokens=8192, overlap_tokens=100, model_name="gpt-4"):
    """
    Splits any node exceeding `max_tokens` into smaller chunks using OpenAI's tokenizer.
    """
    final_nodes = []

    # Get tokenizer for the given OpenAI model
    encoding = tiktoken.encoding_for_model(model_name)

    # Token-aware text splitter
    splitter = TokenTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        tokenizer=encoding.encode  # <- tokenizer function
    )

    for node in nodes:
        text = node.text
        if len(encoding.encode(text)) <= max_tokens:
            final_nodes.append(node)
        else:
            doc = Document(text=text, metadata=node.metadata)
            sub_nodes = splitter.get_nodes_from_documents([doc])
            final_nodes.extend(sub_nodes)

    return final_nodes
def load_and_chunk(files: list[str]) -> list:
    # Filter out non-text files explicitly before loading
    text_files = filter_files(files)
    if not text_files:
        logging.warning("No text files found in the provided files list.")

    documents = SimpleDirectoryReader(
        input_files=text_files,
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

    json_nodes = JSONNodeParser().get_nodes_from_documents(json_docs)
    py_nodes = CodeSplitter(language="python").get_nodes_from_documents(py_docs)
    splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95,
                                          embed_model=OpenAIEmbedding())
    nodes = splitter.get_nodes_from_documents(other_docs)
    logging.info(
        f"py docs {len(py_docs)}, py nodes {len(py_nodes)}, json docs {len(json_docs)}, json nodes {len(json_nodes)}, other docs {len(other_docs)}, other nodes {len(nodes)}")
    all_nodes = py_nodes + nodes + json_nodes
    safe_nodes = enforce_max_tokens(all_nodes, max_tokens=8192)

    return safe_nodes


def create_index(nodes: list) -> VectorStoreIndex:
    index = VectorStoreIndex(nodes)
    logging.info("Created new VectorStoreIndex from scratch.")
    return index


def main():
    # Sync repository and find changes
    changed = sync_repo(REPO_URL, LOCAL_PATH, LAST_SHA_PATH)

    # Choose which files to index
    if not changed:
        logging.info("Full index: scanning all files.")
        files_to_idx = [os.path.join(dp, f)
                        for dp, _, fs in os.walk(LOCAL_PATH)
                        for f in fs]
    else:
        files_to_idx = changed
    logging.info(f"Indexing {len(files_to_idx)} file(s)")

    # Load and chunk files
    t0 = time.time()
    nodes = load_and_chunk(files_to_idx)
    logging.info(f"Chunking done in {time.time() - t0:.2f}s")

    # Configure LLM & embedding
    Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding()

    # Create or update index
    t1 = time.time()
    try:
        storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_STORAGE)
        index = load_index_from_storage(storage_ctx)
        index.insert_nodes(nodes)
        logging.info("Index loaded from storage and updated incrementally.")
    except (FileNotFoundError, json.JSONDecodeError):
        logging.info("Invalid or missing index storage; building fresh storage in memory.")
        index = VectorStoreIndex(nodes)
        logging.info("Created new VectorStoreIndex from scratch.")

    index.storage_context.persist(persist_dir=INDEX_STORAGE)
    logging.info(f"Indexing complete in {time.time() - t1:.2f}s")


if __name__ == "__main__":
    main()
