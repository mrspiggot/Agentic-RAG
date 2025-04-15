# Defines UserInterface (Interface)
# File: rag_system/ui/base_ui.py
# Instruction: Replace the entire content of this file.

import abc
from typing import Dict, Any

# Relative import for Result data model
from ..data_models.result import Result

class UserInterface(abc.ABC):
    """
    Abstract Base Class (Interface) for user interaction layers.

    Defines the contract for how the application core interacts with the
    user, whether through CLI, GUI, or other means.
    """

    @abc.abstractmethod
    def get_configuration_input(self) -> Dict[str, Any]:
        """
        Gets configuration settings from the user or environment specific
        to this UI mode (e.g., command-line arguments).

        Returns:
            A dictionary of configuration overrides or inputs specific to the UI mode.
            May be empty if all config comes from files/env vars.
        """
        pass

    @abc.abstractmethod
    def get_question(self) -> str:
        """
        Prompts the user and retrieves the question they want to ask.

        Returns:
            The user's question string, or an empty string/None to indicate exit.
        """
        pass

    @abc.abstractmethod
    def display_result(self, result: Result):
        """
        Displays the final Result object to the user in the appropriate format
        for this UI mode (e.g., console output, GUI tabs).

        Args:
            result: The Result object containing the answer and execution details.
        """
        pass

    @abc.abstractmethod
    def display_error(self, message: str):
        """
        Displays an error message to the user.

        Args:
            message: The error message string.
        """
        pass

    @abc.abstractmethod
    def display_progress(self, message: str):
        """
        Displays a progress or status message to the user.

        Args:
            message: The progress message string.
        """
        pass