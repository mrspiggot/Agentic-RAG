# File: folder_setup.py
# Instruction: Save this script in the parent directory where you want
#              the 'enhanced_rag_system' project to be created.
#              Run it using 'python folder_setup.py'.

import pathlib
import os

# --- Configuration ---
PROJECT_NAME = "enhanced_rag_system"

# Define the directory structure relative to PROJECT_NAME
DIRECTORIES = [
    "rag_system",
    "rag_system/core",
    "rag_system/config",
    "rag_system/ui",
    "rag_system/ui/cli",
    "rag_system/ui/gui",
    "rag_system/ui/presenters",
    "rag_system/data_models",
    "rag_system/corpus",
    "rag_system/corpus/datasources",
    "rag_system/corpus/loaders",
    "rag_system/corpus/splitters",
    "rag_system/corpus/embedding",
    "rag_system/corpus/vector_stores",
    "rag_system/llm",
    "rag_system/llm/providers",
    "rag_system/workflow",
    "rag_system/workflow/components",
    "rag_system/workflow/components/retrieval",
    "rag_system/workflow/components/evaluation",
    "rag_system/workflow/components/generation",
    "rag_system/workflow/nodes", # Optional node functions dir
    "rag_system/utils",
]

# Define the files to create (including __init__.py files) relative to PROJECT_NAME
# Using None for content means create an empty file.
# Using a string means create the file with that content.
FILES = {
    ".gitignore": """\
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
# Usually these files are written by a python script from a template
# before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal
media/

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
# According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
# Pipfile

# PEP 582; used by PDM, PEP 582 compatible installers
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static analysis results
.pytype/

# Cython debug symbols
cython_debug/

# VSCode settings
.vscode/
""",
    "requirements.txt": """\
# Core Dependencies
python-dotenv
langchain
langgraph
langchain-openai
# langchain-anthropic
# langchain-google-genai
chromadb
streamlit

# Optional for graph visualization
# playwright
# pyppeteer

# Optional for development
# black
# ruff
# pytest
""",
    ".env": """\
# LLM Provider API Keys (replace with your actual keys in a .env file)
OPENAI_API_KEY="sk-..."
# ANTHROPIC_API_KEY="sk-ant-..."
# GOOGLE_API_KEY="..."

# Web Search API Keys
TAVILY_API_KEY="tvly-..."

# --- Optional Configuration Overrides ---
# LLM_MODEL="gpt-4o-mini" # Default if not set
# LLM_TEMPERATURE="0.0"

# RETRIEVAL_K="7"
# RELEVANCE_THRESHOLD="4"
# MIN_RELEVANCE_THRESHOLD="2"

# CHUNK_SIZE="500"
# CHUNK_OVERLAP="50"

# MAX_TRANSFORM_ATTEMPTS="5"
# WEB_SEARCH_RESULTS="5"

# VECTOR_DB_PATH="" # Default: rag_system/chroma_db
# COLLECTION_NAME="rag-chroma-enhanced"
# DOCUMENT_URLS="https://url1.com,https://url2.com" # Comma-separated list, overrides defaults in config
""",
    "README.md": f"# {PROJECT_NAME}\n\nEnhanced RAG System Project.\n",
    "main_cli.py": """\
# Entry point for the Command-Line Interface
# (Implementation TBD)

# import argparse
# from rag_system.core.application import RAGApplication
# from rag_system.ui.factories import UserInterfaceFactory

def main():
    print(f"Starting Enhanced RAG System (CLI Mode)...")
    # Example Usage (replace with actual arg parsing and execution)
    # ui_factory = UserInterfaceFactory()
    # ui = ui_factory.create_ui("cli")
    # config_input = ui.get_configuration_input() # Simplified for now
    # app = RAGApplication(config_override=config_input)
    # question = ui.get_question()
    # if question:
    #     result = app.process_question(question)
    #     ui.display_result(result)
    print("CLI execution placeholder complete.")

if __name__ == "__main__":
    main()
""",
    "main_gui.py": """\
# Entry point for the Streamlit Graphical User Interface
# (Implementation TBD)

import streamlit as st
# from rag_system.core.application import RAGApplication
# from rag_system.ui.factories import UserInterfaceFactory

def main():
    st.set_page_config(layout="wide", page_title="Enhanced RAG System")
    st.title("Enhanced RAG System Interface")

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("Configuration")
        # Placeholder for API key input (use secrets management in real app)
        # openai_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key_gui")

        # Placeholder for URL file upload
        uploaded_file = st.file_uploader("Upload URL File (.txt)", type=['txt'])
        if uploaded_file is not None:
            # To read file as bytes:
            # bytes_data = uploaded_file.getvalue()
            # st.write(bytes_data)
            # To read file as string:
            stringio = uploaded_file # Use directly if text mode expected
            # string_data = stringio.read() # Read if needed
            st.success("File Uploaded!")
            # TODO: Process the uploaded file path/content

        # Placeholder for model selection (needs dynamic loading)
        # provider = st.selectbox("Select LLM Provider", ["OpenAI", "Anthropic", "Gemini"])
        # model = st.selectbox(f"Select {provider} Model", ["Model A", "Model B"]) # Populate dynamically

        st.divider()
        st.info("Configure settings and upload URLs here.")


    # --- Main Area ---
    st.header("Ask a Question")
    question = st.text_area("Enter your question:", height=100)

    if st.button("Answer"):
        if not question:
            st.warning("Please enter a question.")
        # elif not uploaded_file: # Add checks for config
        #     st.warning("Please upload a URL file.")
        else:
            st.info("Processing your question...")
            # Example interaction (replace with actual logic)
            # try:
            #     # Initialize application and process
            #     # This needs careful state management in Streamlit
            #     # app = RAGApplication(...) # Get or init app state
            #     # result = app.process_question(question)
            #
            #     # Display results in tabs (Placeholder)
            #     tab1, tab2, tab3, tab4 = st.tabs(["Answer", "Workflow Graph", "Logs", "Sources"])
            #     with tab1:
            #         st.subheader("Generated Answer")
            #         # st.markdown(result.answer_text)
            #         st.markdown("Placeholder for answer...")
            #     with tab2:
            #         st.subheader("Workflow Visualization")
            #         st.write("Placeholder for Mermaid graph.")
            #         # if result.graph_diagram_data:
            #         #    display_mermaid(result.graph_diagram_data) # Need mermaid display func
            #     with tab3:
            #         st.subheader("Execution Logs")
            #         st.json({"logs": ["Log entry 1...", "Log entry 2..."]}) # Placeholder
            #         # st.json({"logs": result.log_entries})
            #     with tab4:
            #         st.subheader("Sources Used")
            #         st.dataframe([{"url": "http://example.com", "score": 4}]) # Placeholder
            #         # st.dataframe(result.final_source_summary)
            #
            # except Exception as e:
            #     st.error(f"An error occurred: {e}")
            st.success("Placeholder: Processing complete!") # Remove later


if __name__ == "__main__":
    main()
""",
    # Add __init__.py and placeholder .py files for all modules
    "rag_system/__init__.py": None,
    "rag_system/core/__init__.py": None,
    "rag_system/core/application.py": "# Contains RAGApplication class\n",
    "rag_system/config/__init__.py": None,
    "rag_system/config/settings.py": "# Contains Configuration class\n",
    "rag_system/ui/__init__.py": None,
    "rag_system/ui/base_ui.py": "# Defines UserInterface (Interface)\n",
    "rag_system/ui/cli/__init__.py": None,
    "rag_system/ui/cli/interface.py": "# Contains CLIInterface class\n",
    "rag_system/ui/gui/__init__.py": None,
    "rag_system/ui/gui/interface.py": "# Contains StreamlitGUIInterface class\n",
    "rag_system/ui/presenters/__init__.py": None,
    "rag_system/ui/presenters/base_presenter.py": "# Defines OutputPresenter (Interface)\n",
    "rag_system/ui/presenters/cli_presenter.py": "# Contains CLIPresenter class\n",
    "rag_system/ui/presenters/gui_presenter.py": "# Contains StreamlitGUIPresenter class\n",
    "rag_system/ui/factories.py": "# Contains UserInterfaceFactory class\n",
    "rag_system/data_models/__init__.py": None,
    "rag_system/data_models/document.py": "# Defines Document class\n",
    "rag_system/data_models/result.py": "# Defines Result, SourceInfo, LogEntry classes\n",
    "rag_system/data_models/workflow_state.py": "# Defines WorkflowState, UrlUsageInfo classes\n",
    "rag_system/corpus/__init__.py": None,
    "rag_system/corpus/corpus_manager.py": "# Contains DocumentCorpus class\n",
    "rag_system/corpus/datasources/__init__.py": None,
    "rag_system/corpus/datasources/base_datasource.py": "# Defines DataSource (Interface)\n",
    "rag_system/corpus/datasources/file_datasource.py": "# Contains URLFileSource class\n",
    "rag_system/corpus/loaders/__init__.py": None,
    "rag_system/corpus/loaders/base_loader.py": "# Defines DocumentLoader (Interface)\n",
    "rag_system/corpus/loaders/web_loader.py": "# Contains WebDocumentLoader class\n",
    "rag_system/corpus/splitters/__init__.py": None,
    "rag_system/corpus/splitters/text_splitter.py": "# Contains TextSplitter class/interface\n",
    "rag_system/corpus/embedding/__init__.py": None,
    "rag_system/corpus/embedding/base_embedding.py": "# Defines EmbeddingModel (Interface)\n",
    "rag_system/corpus/embedding/openai_embedding.py": "# Contains OpenAIEmbeddingModel class\n",
    "rag_system/corpus/vector_stores/__init__.py": None,
    "rag_system/corpus/vector_stores/base_vector_store.py": "# Defines VectorStore (Interface)\n",
    "rag_system/corpus/vector_stores/chroma_vector_store.py": "# Contains ChromaVectorStore class\n",
    "rag_system/llm/__init__.py": None,
    "rag_system/llm/interaction.py": "# Defines ILLMInteraction (Interface), LangchainLLMInteraction\n",
    "rag_system/llm/providers/__init__.py": None,
    "rag_system/llm/providers/base_provider.py": "# Defines ILLMProvider (Interface)\n",
    "rag_system/llm/providers/openai_provider.py": "# Contains OpenAIProvider class\n",
    "rag_system/llm/providers/anthropic_provider.py": "# Placeholder for AnthropicProvider\n",
    "rag_system/llm/providers/gemini_provider.py": "# Placeholder for GeminiProvider\n",
    "rag_system/llm/factories.py": "# Contains LLMProviderFactory class\n",
    "rag_system/workflow/__init__.py": None,
    "rag_system/workflow/engine.py": "# Contains RAGWorkflowManager class\n",
    "rag_system/workflow/components/__init__.py": None,
    "rag_system/workflow/components/retrieval/__init__.py": None,
    "rag_system/workflow/components/retrieval/base_retriever.py": "# Defines Retriever (Interface)\n",
    "rag_system/workflow/components/retrieval/semantic.py": "# Contains SemanticRetriever class\n",
    "rag_system/workflow/components/evaluation/__init__.py": None,
    "rag_system/workflow/components/evaluation/base_evaluator.py": "# Defines Evaluator (Interface)\n",
    "rag_system/workflow/components/evaluation/relevance.py": "# Contains RelevanceEvaluator class\n",
    "rag_system/workflow/components/generation/__init__.py": None,
    "rag_system/workflow/components/generation/base_generator.py": "# Defines Generator (Interface)\n",
    "rag_system/workflow/components/generation/rag.py": "# Contains RAGGenerator class\n",
    "rag_system/workflow/components/generation/fallback.py": "# Contains FallbackGenerator class\n",
    "rag_system/workflow/components/factories.py": "# Contains RetrieverFactory, EvaluatorFactory, GeneratorFactory\n",
    "rag_system/workflow/nodes/__init__.py": None,
    "rag_system/utils/__init__.py": None,
    "rag_system/utils/logger.py": "# Logging setup functions\n",
}

# --- Script Logic ---
base_path = pathlib.Path(__file__).parent
project_root = base_path / PROJECT_NAME

print(f"Creating project structure for '{PROJECT_NAME}' in '{base_path}'...")

# Create root project directory
project_root.mkdir(exist_ok=True)
print(f"Ensured project root exists: {project_root}")

# Create subdirectories
for rel_dir in DIRECTORIES:
    dir_path = project_root / pathlib.Path(rel_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"  Ensured directory exists: {dir_path}")

# Create files
for rel_file, content in FILES.items():
    file_path = project_root / pathlib.Path(rel_file)
    # Ensure parent directory exists just in case (belt and suspenders)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Create file and write content if provided
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            if content:
                f.write(content)
            else:
                # Write a simple comment for empty __init__.py or placeholders
                if file_path.name == "__init__.py":
                    pass # Keep __init__.py empty
                elif file_path.suffix == ".py":
                     # Check if specific placeholder content was provided above
                     placeholder_content = FILES.get(rel_file, "# Placeholder\n") # Get content again
                     if placeholder_content is None or placeholder_content == "# Placeholder\n":
                        # Add a basic placeholder if no specific comment was provided
                        f.write("# Placeholder for module content\n")
                     # else content was already written above

        print(f"  Created/Updated file:   {file_path}")
    except Exception as e:
        print(f"  ERROR creating file {file_path}: {e}")


print("\nProject structure created successfully!")
print(f"Remember to create a '.env' file based on '.env' and add your API keys.")
print(f"Install dependencies using: pip install -r {PROJECT_NAME}/requirements.txt")