# File: rag_system/ui/factories.py
# Instruction: Replace the entire content of this file.
#              (Updated to create StreamlitGUIInterface)

import logging

# Relative imports
from .base_ui import UserInterface
from .cli.interface import CLIInterface
from .gui.interface import StreamlitGUIInterface # <-- Import GUI Interface

logger = logging.getLogger(__name__)

class UserInterfaceFactory:
    """
    Factory for creating different UserInterface instances (CLI, GUI).
    """
    @staticmethod
    def create_ui(mode: str) -> UserInterface:
        """
        Creates a UserInterface instance based on the specified mode.

        Args:
            mode: The desired mode ("cli" or "gui").

        Returns:
            An instance conforming to the UserInterface interface.

        Raises:
            ValueError: If the mode is unknown or unsupported.
        """
        mode_lower = mode.lower()
        logger.info(f"Creating user interface for mode: '{mode_lower}'")

        if mode_lower == "cli":
            return CLIInterface()
        elif mode_lower == "gui":
            # Now creates the Streamlit interface
            return StreamlitGUIInterface()
        else:
            logger.error(f"Unknown UI mode requested: '{mode}'")
            raise ValueError(f"Unsupported UI mode: {mode}. Choose 'cli' or 'gui'.")