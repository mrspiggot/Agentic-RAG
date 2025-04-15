# Contains CLIInterface class
# File: rag_system/ui/cli/interface.py
# Instruction: Replace the entire content of this file.

import argparse
import logging
from typing import Dict, Any, Optional

# Relative imports
from ..base_ui import UserInterface
from ...data_models.result import Result
from ..presenters.cli_presenter import CLIPresenter # Import concrete presenter

logger = logging.getLogger(__name__)

class CLIInterface(UserInterface):
    """Command-Line Interface implementation."""

    def __init__(self):
        """Initializes the CLI interface with a CLIPresenter."""
        self.presenter = CLIPresenter()
        self.parser = self._setup_arg_parser()
        logger.debug("CLIInterface initialized.")

    def _setup_arg_parser(self) -> argparse.ArgumentParser:
        """Sets up the argument parser for CLI options."""
        parser = argparse.ArgumentParser(
            description="Enhanced RAG System - Command Line Interface."
        )
        parser.add_argument(
            "-f", "--url_file",
            type=str,
            help="Path to a text file containing URLs (one per line) for indexing.",
            default=None # Default is None, app core will use config defaults if not provided
        )
        # Add other arguments as needed (e.g., --provider, --model)
        # For now, we get the question interactively via get_question()
        # parser.add_argument(
        #     "question",
        #     type=str,
        #     nargs='?', # Make question optional here if using interactive input
        #     help="The question to ask the RAG system."
        # )
        return parser

    def get_configuration_input(self) -> Dict[str, Any]:
        """
        Parses command-line arguments for configuration.

        Returns:
            A dictionary containing parsed arguments (e.g., {'url_file': 'path/to/file.txt'}).
        """
        # In a more complex app, could merge args with config file settings here
        args = self.parser.parse_args()
        config_input = {}
        if args.url_file:
             config_input['url_file'] = args.url_file
             logger.info(f"URL file specified via CLI: {args.url_file}")
        # If question was an arg: config_input['question'] = args.question
        return config_input

    def get_question(self) -> str:
        """
        Prompts the user for a question via standard input.

        Returns:
            The question string, or an empty string if the user just hits Enter.
        """
        try:
            question = input("\nEnter your question (or press Enter to quit): ")
            return question.strip()
        except EOFError:
            print("\nExiting.") # Handle Ctrl+D
            return ""

    def display_result(self, result: Result):
        """Displays the result using the CLIPresenter."""
        self.presenter.present(result)

    def display_error(self, message: str):
        """Prints an error message to stderr."""
        print(f"\nERROR: {message}", file=sys.stderr)

    def display_progress(self, message: str):
        """Prints a progress message to stdout."""
        # Add newline for better separation from input prompt potentially
        print(f"... {message}")

# Required for display_error
import sys