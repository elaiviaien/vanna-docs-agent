import logging
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open

import git
import pytest
from git.exc import GitCommandError

from utils import (
    setup_logging, init_or_update_repo, get_last_sha, set_last_sha,
    get_changed_files, sync_repo, is_text_file, filter_files
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def mock_repo():
    """Create a mock git.Repo object."""
    mock = MagicMock(spec=git.Repo)
    mock.remotes.origin = MagicMock()
    mock.git = MagicMock()
    mock.commit.return_value.hexsha = "0123456789abcdef0123456789abcdef01234567"
    return mock



def test_setup_logging_default():
    """Test setup_logging with default parameters."""
    with patch('logging.basicConfig') as mock_basic_config:
        with patch('logging.getLogger') as mock_get_logger:
            setup_logging()
            mock_basic_config.assert_called_once()
            mock_get_logger.assert_called_with('httpx')


def test_setup_logging_custom():
    """Test setup_logging with custom parameters."""
    custom_format = '%(levelname)s - %(message)s'
    custom_level = 'DEBUG'
    custom_handler = logging.NullHandler()
    custom_log_levels = {'requests': logging.ERROR}

    with patch('logging.basicConfig') as mock_basic_config:
        with patch('logging.getLogger') as mock_get_logger:
            setup_logging(
                level=custom_level,
                log_format=custom_format,
                handlers=[custom_handler],
                library_log_levels=custom_log_levels
            )

            mock_basic_config.assert_called_once_with(
                level='DEBUG',
                format=custom_format,
                handlers=[custom_handler]
            )
            mock_get_logger.assert_called_once_with('requests')



def test_init_or_update_repo_new_repo(temp_dir):
    """Test init_or_update_repo on a new repository."""
    repo_url = "https://github.com/example/repo.git"
    repo_path = os.path.join(temp_dir, "repo")

    mock_repo = MagicMock()

    with patch('os.path.exists', return_value=False):
        with patch('os.makedirs') as mock_makedirs:
            with patch('utils.Repo.clone_from', return_value=mock_repo) as mock_clone:
                result = init_or_update_repo(repo_url, repo_path)

                mock_makedirs.assert_called_once()
                mock_clone.assert_called_once_with(repo_url, repo_path, depth=1)
                assert result == mock_repo


def test_init_or_update_repo_existing_repo(temp_dir):
    """Test init_or_update_repo on an existing repository."""
    repo_url = "https://github.com/example/repo.git"
    repo_path = os.path.join(temp_dir, "repo")

    mock_repo = MagicMock()
    mock_repo.remotes.origin = MagicMock()

    with patch('os.path.exists', return_value=True):
        with patch('utils.Repo', return_value=mock_repo) as mock_repo_class:
            result = init_or_update_repo(repo_url, repo_path)

            mock_repo_class.assert_called_once_with(repo_path)
            mock_repo.remotes.origin.fetch.assert_called_once()
            assert result == mock_repo


def test_get_last_sha_existing_file(temp_dir):
    """Test get_last_sha when the file exists."""
    sha_path = os.path.join(temp_dir, "last_sha")
    expected_sha = "abcdef1234567890"

    with patch("builtins.open", mock_open(read_data=expected_sha)) as mock_file:
        result = get_last_sha(sha_path)
        mock_file.assert_called_once_with(sha_path, "r")
        assert result == expected_sha


def test_get_last_sha_missing_file(temp_dir):
    """Test get_last_sha when the file doesn't exist."""
    sha_path = os.path.join(temp_dir, "nonexistent")

    with patch("builtins.open", side_effect=FileNotFoundError):
        result = get_last_sha(sha_path)
        assert result is None


def test_set_last_sha(temp_dir):
    """Test set_last_sha writes the SHA correctly."""
    sha_path = os.path.join(temp_dir, "sub", "last_sha")
    sha = "0123456789abcdef"

    with patch("os.makedirs") as mock_makedirs:
        with patch("builtins.open", mock_open()) as mock_file:
            set_last_sha(sha_path, sha)
            mock_makedirs.assert_called_once()
            mock_file.assert_called_once_with(sha_path, "w")
            mock_file().write.assert_called_once_with(sha)


def test_get_changed_files_with_sha(mock_repo):
    """Test get_changed_files with a valid SHA."""
    base_path = "/repo"
    since_sha = "abcdef1234"
    mock_repo.git.diff.return_value = "file1.py\nfile2.py"

    result = get_changed_files(mock_repo, since_sha, base_path)

    mock_repo.git.diff.assert_called_once_with("--name-only", since_sha, "origin/main")
    assert result == ["/repo/file1.py", "/repo/file2.py"]


def test_get_changed_files_no_sha(mock_repo):
    """Test get_changed_files with no SHA (should return empty list)."""
    result = get_changed_files(mock_repo, None, "/repo")
    assert result == []
    mock_repo.git.diff.assert_not_called()


def test_get_changed_files_git_error(mock_repo):
    """Test get_changed_files when git diff fails."""
    mock_repo.git.diff.side_effect = GitCommandError("diff", 128)

    result = get_changed_files(mock_repo, "abcdef", "/repo")

    assert result == []


def test_sync_repo():
    """Test sync_repo integrates all repository functions correctly."""
    repo_url = "https://github.com/example/repo.git"
    local_path = "/repo"
    last_sha_path = "/last_sha"

    mock_repo = MagicMock()
    changed_files = ["/repo/file.py"]
    new_sha = "0123456789abcdef"

    # Setup commit mock to return an object with hexsha property
    commit_mock = MagicMock()
    commit_mock.hexsha = new_sha
    mock_repo.commit.return_value = commit_mock

    with patch('utils.init_or_update_repo', return_value=mock_repo) as mock_init:
        with patch('utils.get_last_sha', return_value="oldsha") as mock_get_sha:
            with patch('utils.get_changed_files', return_value=changed_files) as mock_get_changes:
                with patch('utils.set_last_sha') as mock_set_sha:
                    result = sync_repo(repo_url, local_path, last_sha_path)

                    mock_init.assert_called_once_with(repo_url, local_path)
                    mock_get_sha.assert_called_once_with(last_sha_path)
                    mock_get_changes.assert_called_once_with(mock_repo, "oldsha", local_path)
                    mock_repo.commit.assert_called_once_with("origin/main")
                    mock_set_sha.assert_called_once_with(last_sha_path, new_sha)
                    assert result == changed_files



def test_is_text_file_nonexistent():
    """Test is_text_file with a nonexistent file."""
    with patch('os.path.exists', return_value=False):
        assert not is_text_file("nonexistent.txt")


def test_is_text_file_directory():
    """Test is_text_file with a directory."""
    with patch('os.path.exists', return_value=True):
        with patch('os.path.isdir', return_value=True):
            assert not is_text_file("directory")


def test_is_text_file_too_large():
    """Test is_text_file with a file larger than 10MB."""
    with patch('os.path.exists', return_value=True):
        with patch('os.path.isdir', return_value=False):
            with patch('os.path.getsize', return_value=15 * 1024 * 1024):
                assert not is_text_file("large.txt")


def test_is_text_file_excluded_extension():
    """Test is_text_file with a file having excluded extension."""
    with patch('os.path.exists', return_value=True):
        with patch('os.path.isdir', return_value=False):
            with patch('os.path.getsize', return_value=1024):
                assert not is_text_file("image.png")
                assert not is_text_file("archive.zip")
                assert not is_text_file("binary.exe")


def test_is_text_file_excluded_mime():
    """Test is_text_file with a file having excluded MIME type."""
    with patch('os.path.exists', return_value=True):
        with patch('os.path.isdir', return_value=False):
            with patch('os.path.getsize', return_value=1024):
                with patch('mimetypes.guess_type', return_value=("image/jpeg", None)):
                    assert not is_text_file("some_image")

                with patch('mimetypes.guess_type', return_value=("application/zip", None)):
                    assert not is_text_file("compressed")


def test_is_text_file_valid():
    """Test is_text_file with a valid text file."""
    with patch('os.path.exists', return_value=True):
        with patch('os.path.isdir', return_value=False):
            with patch('os.path.getsize', return_value=1024):
                with patch('mimetypes.guess_type', return_value=("text/plain", None)):
                    assert is_text_file("valid.txt")


def test_filter_files():
    """Test filter_files excludes appropriate files."""
    test_files = [
        "/repo/file.py",  # Valid
        "/repo/image.png",  # Excluded extension
        "/repo/too_large.txt",  # Too large
        "/repo/.git/config",  # Excluded directory
        "/repo/node_modules/file",  # Excluded directory
        "/repo/valid.md",  # Valid
        "/repo/nonexistent"  # Doesn't exist
    ]

    # Setup mocks for each file type
    def mock_is_text_file(path):
        if path in ["/repo/file.py", "/repo/valid.md"]:
            return True
        return False

    with patch('utils.is_text_file', side_effect=mock_is_text_file):
        result = filter_files(test_files)
        assert "/repo/file.py" in result
        assert "/repo/valid.md" in result
        assert "/repo/image.png" not in result
        assert "/repo/too_large.txt" not in result
        assert "/repo/.git/config" not in result
        assert "/repo/nonexistent" not in result
