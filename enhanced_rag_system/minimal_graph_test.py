import streamlit as st
import logging
from typing import List, Optional, Any

# --- LangGraph Imports ---
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph # Direct import

# --- Setup Logging ---
# Configure basic logging for info level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# Get a logger for this specific script
logger = logging.getLogger(__name__)

logger.info("--- Script Start: Configured for Graphviz/pygraphviz rendering ---")

# --- Minimal Graph Definition ---
# Define the structure of the state
class State(TypedDict):
    messages: list[AnyMessage]

# Define a simple node function
def node(state: State):
    # Safely get messages, default to empty list if not present
    messages = state.get("messages", [])
    # Add a new message to the list
    new_message = AIMessage(content="Hello from node!") # Ensure content keyword arg
    logger.info("Executing simple graph node.")
    # Return the updated state component
    return {"messages": messages + [new_message]}

# --- Build Graph ---
# Initialize graph and error variables
graph: Optional[CompiledGraph] = None
compilation_error: Optional[str] = None

try:
    logger.info("Building minimal StateGraph instance...")
    # Create a StateGraph instance with the defined State
    graph_builder = StateGraph(State)
    # Add the 'node' function as a node in the graph
    graph_builder.add_node("node", node)
    # Define the entry point of the graph
    graph_builder.add_edge(START, "node")
    # Define the exit point of the graph
    graph_builder.add_edge("node", END)

    # Compile the graph definition into a runnable graph object
    graph = graph_builder.compile()
    logger.info("Minimal graph compiled successfully.")

except Exception as e:
    # Log and store any errors during graph compilation
    logger.error(f"Error compiling graph: {e}", exc_info=True)
    compilation_error = f"Failed to compile graph: {e}"


# --- Streamlit App ---
st.title("Minimal LangGraph Visualization Test (Graphviz)")
st.write("Testing `graph.get_graph().draw_png()` using Graphviz/pygraphviz.")
st.info("Ensure 'pygraphviz' is installed in the '.venv' and system 'graphviz' is installed (e.g., via Homebrew).")

# Display compilation status
if compilation_error:
    st.error(f"Graph Compilation Failed: {compilation_error}")
elif graph:
    st.success("Graph compiled successfully. Ready to render with Graphviz.")

    # Initialize variables for image data and potential errors
    graph_png_bytes: Optional[bytes] = None
    error_message: Optional[str] = None
    graph_render_attempted = False # Flag to track if button was clicked

    # Create button to trigger rendering
    if st.button("Generate & Display Graph PNG (using draw_png / Graphviz)"):
        graph_render_attempted = True
        logger.info("--- Button Clicked: Starting Graph Generation using Graphviz/pygraphviz ---")
        st.info("Attempting to generate graph PNG using local Graphviz/pygraphviz rendering...")

        try:
            # --- Call the Graphviz rendering function ---
            # This function requires the 'pygraphviz' Python package and
            # the underlying system 'graphviz' library (e.g., 'dot' executable).
            logger.info("Calling graph.get_graph(xray=True).draw_png()...")
            graph_png_bytes = graph.get_graph(xray=True).draw_png()
            # ---------------------------------------------

            if graph_png_bytes:
                logger.info(f"draw_png() returned {len(graph_png_bytes)} bytes successfully.")
            else:
                # This is less likely for draw_png if it doesn't raise an exception,
                # but handle the case where it might return None.
                logger.warning("draw_png() returned None or empty bytes without raising an exception.")
                error_message = "`draw_png()` returned no data. Check logs for internal errors."

        except ImportError as ie:
             # Catch error if pygraphviz isn't importable (shouldn't happen now)
             logger.error(f"ImportError during Graphviz rendering: {ie}. Is pygraphviz correctly installed in the venv?", exc_info=True)
             error_message = f"ImportError: {ie}. Check pygraphviz installation. See console."
        except FileNotFoundError as fnf_e:
             # Catch error if pygraphviz is installed but the system 'dot' executable isn't found
             logger.error(f"Graphviz FileNotFoundError: {fnf_e}. Can't find Graphviz executable (e.g., 'dot').", exc_info=True)
             logger.error("Ensure system Graphviz (e.g., from `brew install graphviz`) is in your system PATH.")
             error_message = f"Graphviz FileNotFoundError: {fnf_e}. Check system Graphviz PATH. See console."
        except Exception as e:
            # Catch any other unexpected errors during pygraphviz/Graphviz processing
            logger.error(f"Failed to draw workflow graph with Graphviz ({type(e).__name__}): {e}", exc_info=True)
            error_message = f"Error generating PNG via Graphviz: {type(e).__name__}: {e}. Check console."

        # --- Display Logic ---
        if graph_png_bytes:
            # If PNG bytes were generated, display the image
            try:
                st.image(graph_png_bytes, caption="Rendered Workflow Graph (Graphviz/pygraphviz)")
                st.success("Graph image displayed successfully!")
            except Exception as img_e:
                # Catch potential errors during Streamlit's image display
                logger.error(f"Streamlit failed to display the generated PNG bytes: {img_e}", exc_info=True)
                st.error(f"Streamlit error displaying image: {img_e}")
        elif error_message:
            # If an error occurred during generation, display the message
            st.error(error_message)
        elif graph_render_attempted:
            # If button was clicked but no image and no error (e.g., returned None)
            st.info("Graph visualization PNG data was not generated (function returned None or empty).")

else:
    # This handles the unlikely case where the graph object is None
    # even if no compilation_error was explicitly caught.
    st.error("Graph object is None, but no compilation error was recorded. Cannot proceed.")