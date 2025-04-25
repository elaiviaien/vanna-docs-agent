import json
import logging
import os
import time

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from utils import sync_repo

load_dotenv()

REPO_URL     = os.environ.get("REPO_URL", "https://github.com/vanna-ai/vanna.git")
LOCAL_PATH   = os.environ.get("LOCAL_PATH", "./.cache/vanna")
LAST_SHA_PATH = os.environ.get("LAST_SHA_PATH", "./.cache/last_indexed.sha")
INDEX_STORAGE = os.environ.get("INDEX_STORAGE", "./.cache/index_storage")

def load_and_chunk(files: list[str]) -> list:
    """
    Read each file into a Document, then split into chunk-nodes.
    """
    docs = []
    for path in files:
        if not path.lower().endswith((".py", ".md", ".txt")):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logging.warning(f"Could not read {path}: {e}")
            continue
        docs.append(Document(text=text, metadata={"source": path}))

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(docs)
    logging.info(f"Split {len(docs)} docs into {len(nodes)} nodes.")
    return nodes

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
