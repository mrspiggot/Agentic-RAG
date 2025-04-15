# Contains CLIPresenter class
# File: rag_system/ui/presenters/cli_presenter.py
# Instruction: Replace the entire content of this file.

import logging
import textwrap

# Relative imports
from .base_presenter import OutputPresenter
from ...data_models.result import Result

logger = logging.getLogger(__name__)

class CLIPresenter(OutputPresenter):
    """Formats and prints results to the command line console."""

    def present(self, result: Result):
        """
        Formats and prints the Result object to standard output.

        Args:
            result: The Result object to present.
        """
        print("\n" + "="*80)
        print("--- Enhanced RAG System Result ---")
        print("="*80)

        # --- Answer ---
        print("\n>>> Answer:")
        if result.answer_text:
             # Wrap text for better readability in console
             wrapped_answer = textwrap.fill(result.answer_text, width=80)
             print(wrapped_answer)
        else:
             print("No answer was generated.")

        # --- Sources ---
        print("\n>>> Sources Used:")
        if result.final_source_summary:
             print(f"{'URL':<50} {'Type':<12} {'Relevance':<10} {'Usage':<5}")
             print("-"*80)
             for source in sorted(result.final_source_summary, key=lambda x: x.url): # Sort for consistency
                  relevance = f"{source.final_relevance_metric:.1f}" if source.final_relevance_metric is not None else "N/A"
                  print(f"{source.url:<50} {source.source_type:<12} {relevance:<10} {source.usage_count:<5}")
        else:
             print("No specific sources were tracked or used.")

        # --- Logs (Optional - potentially verbose) ---
        # Consider adding a flag or config setting to control log output verbosity
        # print("\n>>> Execution Logs (Summary):")
        # if result.log_entries:
        #     # Print first few and last few, or filter by level?
        #     log_summary = [f"{log.timestamp_utc} [{log.level}] {log.message}" for log in result.log_entries[:5]] # Example: first 5
        #     for log_line in log_summary:
        #          print(log_line)
        #     if len(result.log_entries) > 5: print("...")
        # else:
        #      print("No execution logs captured in result.")

        # --- Metadata ---
        print("\n>>> Execution Metadata:")
        if result.execution_metadata:
            duration = result.execution_metadata.get('total_duration_sec')
            if duration: print(f"- Total Duration: {duration:.3f} seconds")
            gen_meta = result.execution_metadata.get('generation_metadata', {})
            if gen_meta:
                 print(f"- Generation Time: {gen_meta.get('generation_time_sec', 'N/A'):.3f} seconds")
                 print(f"- Fallback Used: {gen_meta.get('is_fallback', False)}")
                 print(f"- Docs Used Count: {gen_meta.get('documents_used_count', 'N/A')}")
            eval_meta = result.execution_metadata.get('evaluation_results', {})
            if eval_meta:
                 print(f"- Evaluation Results: {eval_meta}") # Basic print for now
            error = result.execution_metadata.get('error')
            if error: print(f"- Error: {error}")
        else:
            print("No execution metadata available.")

        print("\n" + "="*80)