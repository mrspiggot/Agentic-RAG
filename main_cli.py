# File: main_cli.py
# Instruction: Replace the entire content of this file.
#              (Removed try/except around rag_system imports)

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import List, Optional
import tempfile
import subprocess

# --- Path Setup ---
# Add the source directory to the Python path to allow importing rag_system
# Assumes main_cli.py is in the project root, and source code is in ./enhanced_rag_system
try:
    project_root_dir = Path(__file__).resolve().parent
    source_root = project_root_dir / "enhanced_rag_system"
    if not source_root.is_dir():
         # If running from within enhanced_rag_system, adjust
         alt_source_root = project_root_dir
         if (alt_source_root / "rag_system").is_dir():
             source_root = alt_source_root
             project_root_dir = source_root.parent
         else:
              raise FileNotFoundError("Could not reliably determine source root directory.")
    sys.path.insert(0, str(source_root))
    # print(f"DEBUG: Added to sys.path: {str(source_root)}") # Keep for temp debug if needed
except Exception as e:
     print(f"ERROR: Failed to setup Python path. Ensure script is run correctly relative to project structure.")
     print(f"Error details: {e}")
     sys.exit(1)

# --- Core Application Imports (Direct - Fail Fast if error) ---
from rag_system.core.application import RAGApplication
from rag_system.ui.factories import UserInterfaceFactory
from rag_system.config.settings import Configuration, ConfigurationError
from rag_system.utils.logger import setup_logger
from rag_system.corpus.corpus_manager import CorpusManagerError
from rag_system.workflow.engine import WorkflowError

# --- Helper Function for Graph Display ---
def display_graph_image(png_data: bytes):
    """Saves PNG bytes to a temp file and tries to open it."""
    if not png_data:
        logging.warning("No PNG data provided to display_graph_image.") # Use logging
        return

    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_file_path_str = temp_file.name
            temp_file.write(png_data)
            logging.info(f"Workflow graph saved temporarily to: {temp_file_path_str}")
            print(f"\nWorkflow graph image saved to: {temp_file_path_str}")

        # Attempt to open the file
        if sys.platform == "win32":
            os.startfile(temp_file_path_str)
        elif sys.platform == "darwin": # macOS
            subprocess.run(['open', temp_file_path_str], check=True)
        else: # Linux and other POSIX
            subprocess.run(['xdg-open', temp_file_path_str], check=True)
        print(f"Attempted to open graph image with default viewer...")

    except FileNotFoundError:
         logging.warning(f"Could not find command ('open' or 'xdg-open') to display image.")
         print(f"Could not find command to automatically open the image.")
    except Exception as e:
         logging.error(f"Failed to save or open graph image: {e}", exc_info=False)
         print(f"Failed to open graph image automatically: {e}")


# --- Main Execution ---

def main():
    # Setup logger first
    setup_logger(level=logging.INFO)
    logger = logging.getLogger(__name__) # Get logger for this script context
    logger.info("--- Starting Enhanced RAG System (CLI Mode) ---")

    parser = argparse.ArgumentParser(
        description="Enhanced RAG System - Command Line Interface."
    )
    parser.add_argument(
        "-f", "--url_file", type=str, default=None,
        help="Path to a text file containing URLs (one per line) for indexing."
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug level logging."
    )
    parser.add_argument(
        "--show-graph", action="store_true", help="Attempt to generate and display workflow graph after each question."
    )

    args = parser.parse_args()

    # Update log level AFTER parsing args
    if args.debug:
        setup_logger(level=logging.DEBUG) # Reconfigure root logger level (and others potentially)
        logger.info("Debug logging enabled.")

    app: Optional[RAGApplication] = None
    try:
        # Use project_root_dir determined earlier for .env path
        config = Configuration(dotenv_path=project_root_dir / '.env')
        app = RAGApplication(config=config)

        logger.info("Initializing document corpus...")
        app.initialize_corpus(url_file_path=args.url_file)
        logger.info("Document corpus initialization complete.")

        ui_factory = UserInterfaceFactory()
        ui = ui_factory.create_ui("cli")

        while True:
            question = ui.get_question()
            if not question:
                break

            ui.display_progress(f"Processing question: '{question[:50]}...'")
            try:
                 result = app.process_question(question)
                 ui.display_result(result)

                 if args.show_graph and app and app.workflow_manager:
                     print("\nAttempting to generate graph visualization...")
                     png_data = app.workflow_manager.get_graph_visualization_png()
                     if png_data:
                         display_graph_image(png_data)
                     else:
                         print("Graph generation skipped or failed (check logs/dependencies like playwright).")

            except (WorkflowError, CorpusManagerError, ValueError) as e:
                 logger.error(f"Error processing question: {e}", exc_info=False)
                 ui.display_error(f"Failed to process question: {e}")
            except Exception as e:
                 logger.error(f"An unexpected error occurred processing question: {e}", exc_info=True)
                 ui.display_error(f"An unexpected error occurred: {e}")

        logger.info("--- RAG System CLI Finished ---")

    # Keep broader error handling for application setup issues
    except ConfigurationError as e:
         print(f"\nCRITICAL CONFIG ERROR: {e}", file=sys.stderr)
         logger.critical(f"Configuration error: {e}", exc_info=True)
         sys.exit(1)
    except CorpusManagerError as e:
         print(f"\nCRITICAL CORPUS ERROR: {e}", file=sys.stderr)
         logger.critical(f"Corpus initialization error: {e}", exc_info=True)
         sys.exit(1)
    except WorkflowError as e:
         print(f"\nCRITICAL WORKFLOW ERROR: {e}", file=sys.stderr)
         logger.critical(f"Workflow initialization error: {e}", exc_info=True)
         sys.exit(1)
    except FileNotFoundError as e:
         print(f"\nERROR: File not found - {e}", file=sys.stderr)
         logger.error(f"File not found error: {e}", exc_info=True)
         sys.exit(1)
    except ImportError as e: # Catch import errors during app init now
        print(f"\nCRITICAL IMPORT ERROR: {e}. Please ensure all dependencies are installed.", file=sys.stderr)
        logger.critical(f"An critical import error occurred in main: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
         print(f"\nUNEXPECTED ERROR in main: {e}", file=sys.stderr)
         logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
         sys.exit(1)


if __name__ == "__main__":
    main()