# File: rag_system/corpus/datasources/file_datasource.py
# Instruction: Replace the entire content of this file.

import logging
from pathlib import Path
from typing import List

# Use relative import for base class and error
from .base_datasource import DataSource, DataSourceError

logger = logging.getLogger(__name__)

class URLFileSource(DataSource):
    """
    A data source that reads URLs from a text file, one URL per line.
    Skips empty lines and lines starting with '#'.
    """

    def __init__(self, file_path: str):
        """
        Initializes the URLFileSource.

        Args:
            file_path: The path to the text file containing URLs.

        Raises:
            FileNotFoundError: If the file_path does not exist or is not a file.
            ValueError: If file_path is empty or invalid.
        """
        if not file_path:
            raise ValueError("File path cannot be empty for URLFileSource.")

        # Resolve the path to handle relative paths correctly
        self.file_path = Path(file_path).resolve()
        logger.info(f"Initializing URLFileSource with resolved path: {self.file_path}")

        if not self.file_path.is_file():
            logger.error(f"URL source file not found or is not a file: {self.file_path}")
            raise FileNotFoundError(f"URL source file not found: {self.file_path}")

    def get_urls(self) -> List[str]:
        """
        Reads URLs from the file, one per line, skipping empty lines and comments.

        Returns:
            A list of URL strings found in the file.

        Raises:
            DataSourceError: If reading the file fails.
        """
        logger.info(f"Reading URLs from file: {self.file_path}")
        urls = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    stripped_line = line.strip()
                    # Skip empty lines and lines starting with # (comments)
                    if stripped_line and not stripped_line.startswith('#'):
                        # Basic check if it looks like a URL (can be improved)
                        if stripped_line.startswith(('http://', 'https://')):
                            urls.append(stripped_line)
                        else:
                            logger.warning(f"Skipping invalid line {line_num} in {self.file_path}: Does not start with http/https.")
            logger.info(f"Found {len(urls)} valid URLs in {self.file_path}.")
            return urls
        except IOError as e:
            logger.error(f"Failed to read URL file '{self.file_path}': {e}", exc_info=True)
            raise DataSourceError(f"Could not read URL file: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error reading URL file '{self.file_path}': {e}", exc_info=True)
            raise DataSourceError(f"Unexpected error reading URL file: {e}") from e

    def get_name(self) -> str:
        """Returns the name including the file."""
        # Use self.file_path which is now a Path object
        return f"URLFileSource({self.file_path.name})"