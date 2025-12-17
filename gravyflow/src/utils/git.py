import os
import logging

logger = logging.getLogger(__name__)

from git import Repo, InvalidGitRepositoryError

def get_current_repo():
    """
    Attempts to find and return a Git repository object based on the current working directory.
    Returns None if no repository is found.
    """
    cwd = os.getcwd()

    try:
        # Initialize the repository object based on the current working directory
        repo = Repo(cwd, search_parent_directories=True)
        logger.info("Found working directory repository.")
        return repo
    except InvalidGitRepositoryError:
        logger.warning("Current working directory is not a git repository.")
        return None