import os
from unittest.mock import patch, MagicMock

import pytest
import tiktoken
from llama_index.core import Document
from llama_index.core.schema import Node

from indexing import (
    enforce_max_tokens, load_and_chunk,
    MAX_TOKENS, OVERLAP_TOKENS
)
from indexing import filter_files

ENC = tiktoken.get_encoding("cl100k_base")


@pytest.fixture
def mock_embedding_model():
    """Create a mock AzureOpenAIEmbedding model."""
    mock_model = MagicMock()
    return mock_model


@pytest.fixture
def mock_nodes():
    """Create sample nodes for testing."""
    small_node = MagicMock(spec=Node)
    small_node.text = "This is a small text node."
    small_node.metadata = {"file_path": "file1.py", "start_line": 1, "end_line": 5}

    large_node = MagicMock(spec=Node)
    # Create a text that would exceed token limit
    large_node.text = "alphabet " * 10000  # String that more than MAX_TOKENS
    large_node.metadata = {"file_path": "file2.py", "start_line": 1, "end_line": 500}

    return [small_node, large_node]


@pytest.fixture
def mock_documents():
    """Create sample documents for testing."""
    py_doc = Document(
        text="def example():\n    return 'Hello World'",
        metadata={"file_path": "example.py"}
    )

    json_doc = Document(
        text='{"key": "value", "array": [1, 2, 3]}',
        metadata={"file_path": "config.json"}
    )

    md_doc = Document(
        text="# Heading\nThis is markdown content.",
        metadata={"file_path": "README.md"}
    )

    return [py_doc, json_doc, md_doc]


def test_enforce_max_tokens_within_limit(mock_nodes):
    """Test enforce_max_tokens doesn't split nodes that are within token limits."""
    with patch('tiktoken.encoding_for_model') as mock_encoding:
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text: [0] * min(len(text.split()),
                                                                   100)  # Small text returns few tokens
        mock_encoding.return_value = mock_tokenizer

        result = enforce_max_tokens([mock_nodes[0]], max_tokens=MAX_TOKENS)

        # Should have same number of nodes as input
        assert len(result) == 1
        assert result[0] == mock_nodes[0]


def test_enforce_max_tokens_exceeds_limit(mock_nodes):
    """Test enforce_max_tokens splits nodes that exceed token limits."""
    with patch('tiktoken.encoding_for_model') as mock_encoding:
        mock_tokenizer = MagicMock()
        # Make large text return more than MAX_TOKENS
        mock_tokenizer.encode.side_effect = lambda text: [0] * (MAX_TOKENS + 1000 if "alphabet " * 10000 in text else 50)
        mock_encoding.return_value = mock_tokenizer

        with patch('indexing.TokenTextSplitter') as MockSplitter:
            mock_splitter = MagicMock()
            MockSplitter.return_value = mock_splitter

            # Sub-nodes to be returned by the splitter
            sub_node1 = MagicMock(spec=Node)
            sub_node2 = MagicMock(spec=Node)
            mock_splitter.get_nodes_from_documents.return_value = [sub_node1, sub_node2]

            result = enforce_max_tokens([mock_nodes[1]], max_tokens=MAX_TOKENS)

            # Check split the large node into multiple nodes
            assert len(result) == 2
            assert result[0] == sub_node1
            assert result[1] == sub_node2

            # Verify splitter was called with correct parameters
            MockSplitter.assert_called_once_with(
                chunk_size=MAX_TOKENS,
                chunk_overlap=OVERLAP_TOKENS,
                tokenizer=mock_tokenizer.encode
            )


def test_load_and_chunk_empty_after_filter():
    """Test load_and_chunk when no files remain after filtering."""
    with patch('indexing.filter_files', return_value=[]):
        result = load_and_chunk(['file1.txt', 'file2.txt'])
        assert result == []


def test_load_and_chunk(mock_documents, mock_embedding_model):
    """Test load_and_chunk processes files correctly."""
    test_files = ["example.py", "config.json", "README.md"]

    with patch('indexing.filter_files', return_value=test_files):
        with patch('indexing.SimpleDirectoryReader') as MockReader:
            mock_reader_instance = MagicMock()
            MockReader.return_value = mock_reader_instance
            mock_reader_instance.load_data.return_value = mock_documents

            with patch('indexing.get_embed_model', return_value=mock_embedding_model):
                with patch('indexing.JSONNodeParser') as MockJSONParser:
                    with patch('indexing.CodeSplitter') as MockCodeSplitter:
                        with patch('indexing.SentenceSplitter') as MockSemanticSplitter:
                            with patch('indexing.enforce_max_tokens') as mock_enforce:
                                # Setup node parser mocks
                                json_node = MagicMock(spec=Node)
                                py_node = MagicMock(spec=Node)
                                md_node = MagicMock(spec=Node)

                                mock_json_parser = MagicMock()
                                MockJSONParser.return_value = mock_json_parser
                                mock_json_parser.get_nodes_from_documents.return_value = [json_node]

                                mock_code_splitter = MagicMock()
                                MockCodeSplitter.return_value = mock_code_splitter
                                mock_code_splitter.get_nodes_from_documents.return_value = [py_node]

                                mock_semantic_splitter = MagicMock()
                                MockSemanticSplitter.return_value = mock_semantic_splitter
                                mock_semantic_splitter.get_nodes_from_documents.return_value = [md_node]

                                # enforce_max_tokens mock
                                final_nodes = [MagicMock(spec=Node) for _ in range(3)]
                                mock_enforce.return_value = final_nodes

                                result = load_and_chunk(test_files)

                                # Verify correct calls were made
                                MockReader.assert_called_once_with(
                                    input_files=test_files,
                                    recursive=True
                                )

                                # Verify document metadata was processed
                                assert any(doc.metadata.get("file_path") for doc in mock_documents)

                                # Verify parsers were called with appropriate documents
                                mock_json_parser.get_nodes_from_documents.assert_called_once()
                                mock_code_splitter.get_nodes_from_documents.assert_called_once()
                                mock_semantic_splitter.get_nodes_from_documents.assert_called_once()

                                # Verify token enforcement was called with combined nodes
                                mock_enforce.assert_called_once()

                                assert result == final_nodes


def test_no_node_exceeds_max_token_limit():
    repo_path = os.environ.get("LOCAL_PATH", "./.cache/vanna")
    files = [
        os.path.join(dp, f)
        for dp, _, fs in os.walk(repo_path)
        for f in fs
        if filter_files([os.path.join(dp, f)])
    ]
    nodes = load_and_chunk(files)

    # find any nodes that are too large
    exceeding = []
    for idx, node in enumerate(nodes):
        token_count = len(ENC.encode(node.text))
        if token_count > MAX_TOKENS:
            exceeding.append({
                "index": idx,
                "file_path": node.metadata.get("file_path"),
                "start_line": node.metadata.get("start_line"),
                "end_line": node.metadata.get("end_line"),
                "tokens": token_count,
            })

    if exceeding:
        for e in exceeding:
            print(
                f"Node #{e['index']} in {e['file_path']} "
                f"(lines {e['start_line']}–{e['end_line']}) "
                f"has {e['tokens']} tokens!"
            )
    assert not exceeding, f"{len(exceeding)} nodes exceed {MAX_TOKENS} tokens"
