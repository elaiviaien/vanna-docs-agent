import logging
import os

import git
from git import Repo


def init_or_update_repo(repo_url: str, path: str) -> Repo:
    """
    Clone the repo if not present, otherwise fetch the latest.
    """
    if not os.path.exists(path):
        logging.info(f"Cloning repository {repo_url} → {path}")
        return Repo.clone_from(repo_url, path, depth=1)
    repo = Repo(path)
    logging.info(f"Fetching updates for repository at {path}")
    repo.remotes.origin.fetch()
    return repo


def get_last_sha(sha_path: str) -> str | None:
    try:
        with open(sha_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def set_last_sha(sha_path: str, sha: str) -> None:
    os.makedirs(os.path.dirname(sha_path), exist_ok=True)
    with open(sha_path, "w") as f:
        f.write(sha)


def get_changed_files(repo: Repo, since_sha: str | None, base_path: str) -> list[str]:
    """
    Return a list of files changed on origin/main since `since_sha`.
    Empty if `since_sha` is None or on error.
    """
    if since_sha is None:
        logging.info("No previous SHA; will perform full scan.")
        return []
    try:
        diffs = repo.git.diff("--name-only", since_sha, "origin/main")
        files = [os.path.join(base_path, p) for p in diffs.splitlines()]
        logging.info(f"Detected {len(files)} changed file(s) since {since_sha}")
        return files
    except git.exc.GitCommandError as e:
        logging.warning(f"Git diff error: {e}; falling back to full scan.")
        return []

def sync_repo(repo_url: str, local_path: str, last_sha_path: str) -> list[str]:
    """
    Sync the repository and return a list of changed files.
    """
    repo = init_or_update_repo(repo_url, local_path)
    last_sha = get_last_sha(last_sha_path)
    changed_files = get_changed_files(repo, last_sha, local_path)
    new_sha = repo.commit("origin/main").hexsha
    set_last_sha(last_sha_path, new_sha)
    return changed_files


