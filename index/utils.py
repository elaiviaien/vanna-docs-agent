import logging
import mimetypes
import os
from types import MappingProxyType
from typing import Union, Optional, Iterable, Mapping
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


EXCLUDED_MIME_TYPES = {
    'image', 'audio', 'video', 'application/x-msdownload', 'application/zip', 'application/x-rar-compressed'
}


def is_text_file(file_path: str) -> bool:
    """
    Check if a file is text-based (not an image, audio, video, or binary file)
    by checking its MIME type.
    """
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and any(mime_type.startswith(exclude) for exclude in EXCLUDED_MIME_TYPES):
        return False
    return True


def filter_files(files: list[str]) -> list[str]:
    """
    Filter out files that can't be indexed (non-text, binary, etc.) from a list of files and exclude .git directories.
    """
    return [file for file in files if is_text_file(file) and ".git" not in file]

def setup_logging(level: Union[int, str] = logging.INFO,
                  log_format: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                  handlers: Optional[Iterable[logging.Handler]] = None,
                  library_log_levels: Mapping[str, int] = MappingProxyType({'httpx': logging.WARNING})) -> None:
    """
    Setup logging configuration. By default, logs are output to the console. Additional handlers can be provided. Log
    levels for specific libraries can be set.
    Args:
        level: The logging level. Can be an integer (e.g., logging.INFO) or a string (e.g., 'INFO').
            Strings must be uppercase. Defaults to logging.INFO.
        log_format: The log format.
            Defaults to '%(asctime)s - %(levelname)s - %(name)s - %(message)s'.
        handlers: Additional logging handlers. Defaults to None. If None, a StreamHandler is created for this call and
            logs are output to the console.
        library_log_levels: A mapping of library names and their respective log levels.
            Defaults to {'httpx': logging.WARNING}.
    """
    # Convert string log levels to uppercase
    if isinstance(level, str):
        level = level.upper()

    if handlers is None:
        handlers = [logging.StreamHandler()]

    logging.basicConfig(level=level, format=log_format, handlers=handlers)

    # Set log levels for specific libraries
    for library, lib_level in library_log_levels.items():
        logging.getLogger(library).setLevel(lib_level)