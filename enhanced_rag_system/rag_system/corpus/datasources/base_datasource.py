# Defines DataSource (Interface)
# File: rag_system/corpus/datasources/base_datasource.py
# Instruction: Create this file or replace its entire content.

import abc
import logging
from typing import List

logger = logging.getLogger(__name__)

class DataSourceError(Exception):
    """Custom exception for DataSource errors."""
    pass

class DataSource(abc.ABC):
    """
    Abstract Base Class (Interface) for defining data sources,
    primarily responsible for providing a list of source identifiers (e.g., URLs).
    """

    @abc.abstractmethod
    def get_urls(self) -> List[str]:
        """
        Retrieves the list of source URLs provided by this data source.

        Returns:
            A list of URL strings.

        Raises:
            DataSourceError: If retrieving URLs fails (e.g., file not found).
            FileNotFoundError: If a required file source does not exist.
        """
        pass

    def get_name(self) -> str:
        """Return the name of the data source implementation."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Default representation."""
        return f"<{self.get_name()}>"