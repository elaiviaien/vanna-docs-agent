import os
import tiktoken

from indexing import load_and_chunk, filter_files

# pick the same encoding your embedding model uses:
ENC = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 8192

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
