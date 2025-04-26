import logging
import mimetypes
import os
from typing import List, Union, Optional, Set

import git
from git import Repo, InvalidGitRepositoryError

EXCLUDED_MIME_PREFIXES: Set[str] = {
    'image', 'audio', 'video', 'application/x-msdownload',
    'application/zip', 'application/x-rar-compressed'
}

EXCLUDED_EXTENSIONS: Set[str] = {
    '.pyc', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg',
    '.woff', '.woff2', '.ttf', '.eot', '.otf', '.zip', '.tar',
    '.gz', '.rar', '.7z', '.exe', '.dll', '.so', '.dylib',
    '.class', '.jar', '.war', '.ear', '.db', '.sqlite', '.sqlite3'
}

EXCLUDED_DIRS: Set[str] = {
    '__pycache__', '.git', '.github', '.idea', '.vscode',
    'node_modules', 'venv', 'env', '.venv'
}


def setup_logging(
        level: Union[int, str] = logging.INFO,
        log_format: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        log_file: Optional[str] = None
) -> None:
    """Configure logging to output to console and file"""
    # Convert string log levels to uppercase
    if isinstance(level, str):
        level = level.upper()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set library-specific log levels
    logging.getLogger('httpx').setLevel(logging.WARNING)


def init_or_update_repo(repo_url: str, path: str) -> Repo:
    """
    Clone the repository if not present, otherwise fetch the latest changes.
    """
    try:
        repo = Repo(path)
        logging.info(f"Fetching updates for repository at {path}")
        repo.remotes.origin.fetch()
        return repo
    except InvalidGitRepositoryError:
        logging.info(f"Cloning repository {repo_url} → {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return Repo.clone_from(repo_url, path, depth=1)




def get_last_sha(sha_path: str) -> Optional[str]:
    """
    Retrieve the SHA hash of the last sync from the given file.
    """
    try:
        with open(sha_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def set_last_sha(sha_path: str, sha: str) -> None:
    """
    Save the given SHA hash to the specified file.
    """
    os.makedirs(os.path.dirname(sha_path), exist_ok=True)
    with open(sha_path, "w") as f:
        f.write(sha)


def get_changed_files(repo: Repo, since_sha: Optional[str], base_path: str) -> List[str]:
    """
    Get files changed in the repository since the specified SHA.
    """
    if since_sha is None:
        logging.info("No previous SHA; will perform full scan.")
        return []

    try:
        diffs = repo.git.diff("--name-only", since_sha, "origin/main")
        files = [os.path.join(base_path, p) for p in diffs.splitlines() if p]
        logging.info(f"Detected {len(files)} changed file(s) since {since_sha}")
        return files
    except git.exc.GitCommandError as e:
        logging.warning(f"Git diff error: {e}; falling back to full scan.")
        return []


def sync_repo(repo_url: str, local_path: str, last_sha_path: str) -> List[str]:
    """
    Sync a Git repository and return a list of changed files since last sync.
    """
    repo = init_or_update_repo(repo_url, local_path)
    last_sha = get_last_sha(last_sha_path)
    changed_files = get_changed_files(repo, last_sha, local_path)
    new_sha = repo.commit("origin/main").hexsha
    set_last_sha(last_sha_path, new_sha)
    return changed_files


def is_text_file(file_path: str) -> bool:
    """
    Check if a file is text-based (not binary) by examining its MIME type.
    """
    # Use absolute path to ensure existence check works
    abs_path = os.path.abspath(file_path)

    # Check file existence
    if not os.path.exists(abs_path) or os.path.isdir(abs_path):
        logging.debug(f"File does not exist or is directory: {file_path}")
        return False

    # Check file size (skip files > 10MB)
    try:
        if os.path.getsize(abs_path) > 10 * 1024 * 1024:
            logging.debug(f"File too large: {file_path}")
            return False
    except OSError:
        logging.debug(f"Cannot determine size: {file_path}")
        return False

    # Check file extension
    _, ext = os.path.splitext(file_path.lower())
    if ext in EXCLUDED_EXTENSIONS:
        logging.debug(f"Excluded extension {ext}: {file_path}")
        return False

    # Check MIME type
    mime_type, _ = mimetypes.guess_type(file_path)

    # If mime_type is None, assume it's a text file if not in excluded extensions
    if mime_type is None:
        # For files without a clear MIME type, try to detect if they're text
        try:
            with open(abs_path, 'rb') as f:
                sample = f.read(1024)
                # Check if the content appears to be binary
                if b'\x00' in sample:
                    logging.debug(f"Appears to be binary: {file_path}")
                    return False
            return True
        except Exception as e:
            logging.debug(f"Error reading file {file_path}: {e}")
            return False

    if any(mime_type.startswith(prefix) for prefix in EXCLUDED_MIME_PREFIXES):
        logging.debug(f"Excluded MIME type {mime_type}: {file_path}")
        return False

    return True


def filter_files(files: List[str]) -> List[str]:
    """
    Filter out non-text, binary, or otherwise undesirable files from a list.
    """
    filtered = []

    for file_path in files:
        # Skip files that aren't text
        if not is_text_file(file_path):
            continue

        # Check if file is in excluded directory
        parts = os.path.normpath(file_path).split(os.sep)
        if any(part in EXCLUDED_DIRS for part in parts):
            continue

        filtered.append(file_path)

    return filtered
