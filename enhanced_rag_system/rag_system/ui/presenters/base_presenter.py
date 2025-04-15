# Defines OutputPresenter (Interface)
# File: rag_system/ui/presenters/base_presenter.py
# Instruction: Replace the entire content of this file.

import abc

# Relative import for Result data model
from ...data_models.result import Result

class OutputPresenter(abc.ABC):
    """
    Abstract Base Class (Interface) for formatting and presenting results.

    Decouples the result presentation logic from the main UI flow.
    """

    @abc.abstractmethod
    def present(self, result: Result):
        """
        Formats and presents the Result object to the user via the associated UI.

        Args:
            result: The Result object to present.
        """
        pass