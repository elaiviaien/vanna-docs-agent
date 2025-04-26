import json
import logging
import os
import time
from typing import List

import tiktoken
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
)
from llama_index.core.node_parser import (
    CodeSplitter,
    JSONNodeParser,
    TokenTextSplitter, SentenceSplitter
)
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from utils import filter_files, sync_repo, setup_logging

load_dotenv()

REPO_URL = os.environ.get("REPO_URL", "https://github.com/vanna-ai/vanna.git")
LOCAL_PATH = os.environ.get("LOCAL_PATH", "./.cache/vanna")
LAST_SHA_PATH = os.environ.get("LAST_SHA_PATH", "./.cache/last_indexed.sha")
INDEX_STORAGE = os.environ.get("INDEX_STORAGE", "./.cache/index_storage")
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'text-embedding-ada-002')
MAX_TOKENS = 8192
OVERLAP_TOKENS = 100
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
def get_embed_model():
    """Create and return the embedding model instance."""
    return AzureOpenAIEmbedding(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        model=EMBEDDING_MODEL_NAME,
        deployment_name=EMBEDDING_MODEL_NAME
    )


def enforce_max_tokens(nodes, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS, model_name=EMBEDDING_MODEL_NAME):
    """
    Splits any node exceeding `max_tokens` into smaller chunks using OpenAI's tokenizer.
    """
    final_nodes = []
    encoding = tiktoken.encoding_for_model(model_name)

    splitter = TokenTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        tokenizer=encoding.encode
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


def load_and_chunk(files: List[str]) -> List:
    """
    Load and chunk documents from the provided file paths.
    """
    # Filter out non-text files

    text_files = filter_files(files)
    if not text_files:
        logging.warning("No text files found in the provided files list.")
        return []

    # Load documents
    documents = SimpleDirectoryReader(
        input_files=text_files,
        recursive=True
    ).load_data()

    # Clean up file paths in metadata
    for doc in documents:
        doc.metadata = {
            "file_path": doc.metadata.get("file_path", "unknown").replace(".cache/vanna/", ""),
        }

    # Group documents by type for specialized processing
    py_docs = [file for file in documents if file.metadata.get("file_path", "").endswith(".py")]
    json_docs = [file for file in documents if file.metadata.get("file_path", "").endswith(".json")]
    other_docs = [file for file in documents if
                  not file.metadata.get("file_path", "").endswith(".py") and
                  not file.metadata.get("file_path", "").endswith(".json")]

    # Process documents with specialized parsers
    json_nodes = JSONNodeParser().get_nodes_from_documents(json_docs)
    py_nodes = CodeSplitter(language="python", chunk_lines=20, chunk_lines_overlap=5).get_nodes_from_documents(py_docs)

    splitter = SentenceSplitter()
    other_nodes = splitter.get_nodes_from_documents(other_docs)

    # Combine all nodes and enforce token limits
    all_nodes = py_nodes + other_nodes + json_nodes
    safe_nodes = enforce_max_tokens(all_nodes)
    return safe_nodes


def sync_repo_and_update_index():
    """Main function to sync repository and update index."""
    # Configure embedding
    Settings.embed_model = get_embed_model()


    # Get files changed since last sync
    changed = sync_repo(REPO_URL, LOCAL_PATH, LAST_SHA_PATH)
    nodes = []
    # Choose which files to index
    all_files_to_indx = [os.path.join(dp, f)
                        for dp, _, fs in os.walk(LOCAL_PATH)
                        for f in fs]
    if changed:
        # Load and chunk files
        t0 = time.time()
        nodes = load_and_chunk(changed)
        logging.info(f"Chunking done in {time.time() - t0:.2f}s")

    # Create or update index
    t1 = time.time()
    try:
        # load existing index
        storage_ctx = StorageContext.from_defaults(persist_dir=INDEX_STORAGE)
        index = load_index_from_storage(storage_ctx)
        if nodes:
            # Update index with new nodes
            index.insert_nodes(nodes)
        logging.info("Index loaded from storage and updated incrementally.")
    except (FileNotFoundError, json.JSONDecodeError):
        # Create new index if loading fails
        logging.info("Invalid or missing index storage; building fresh storage in memory.")
        # Load and chunk files
        t0 = time.time()
        nodes = load_and_chunk(all_files_to_indx)
        logging.info(f"Chunking done in {time.time() - t0:.2f}s")
        index = VectorStoreIndex(nodes)
        logging.info("Created new VectorStoreIndex from scratch.")

    # Save the index
    os.makedirs(os.path.dirname(INDEX_STORAGE), exist_ok=True)
    index.storage_context.persist(persist_dir=INDEX_STORAGE)
    logging.info(f"Indexing complete in {time.time() - t1:.2f}s")


if __name__ == "__main__":
    setup_logging()
    sync_repo_and_update_index()