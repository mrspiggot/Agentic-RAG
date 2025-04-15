# File: collect_source_files.py
# Instruction: Save this script in the parent directory of the
#              'enhanced_rag_system' project folder.
#              Run it using 'python collect_source_files.py'.
#              It will create 'project_source_snapshot.txt' containing
#              the content of relevant project files.

import pathlib
import os

# --- Configuration ---
PROJECT_DIR_NAME = "enhanced_rag_system" # The name of the project folder
OUTPUT_FILENAME = "project_source_snapshot.txt"

# Directories to exclude (names, not full paths)
EXCLUDED_DIRS = {
    '.git', '__pycache__', 'venv', '.venv', 'env', 'ENV',
    'build', 'dist', 'site', '*.egg-info', '.mypy_cache',
    '.pytest_cache', 'htmlcov', '.tox', '.nox', 'node_modules',
    '.vscode', '.idea', # Add IDE-specific folders if needed
    'cython_debug',
}

# Specific files or patterns to exclude
EXCLUDED_FILES = {
    'folder_setup.py',          # Exclude setup script
    'collect_source_files.py',  # Exclude this script itself
    '.DS_Store',                # macOS specific
    '*.pyc', '*.pyo', '*.pyd',  # Compiled Python
    '*.so',                     # Compiled extensions
    '.env',                     # Exclude actual secrets file
    OUTPUT_FILENAME,            # Exclude the output file itself
}

# File extensions to include
INCLUDED_EXTENSIONS = {
    '.py',
    '.txt',
    '.md',
    '.gitignore',
    '.example', # To include .env
    # Add other relevant text-based extensions if needed (e.g., .yaml, .toml)
}
# --- End Configuration ---

def should_exclude(path: pathlib.Path, project_root: pathlib.Path) -> bool:
    """Check if a path should be excluded based on configuration."""
    relative_path = path.relative_to(project_root)

    # Check against excluded directory names in any part of the path
    for part in relative_path.parts:
        if part in EXCLUDED_DIRS:
            return True

    # Check against excluded file names/patterns
    if path.name in EXCLUDED_FILES:
        return True
    for pattern in EXCLUDED_FILES:
        if "*" in pattern and path.match(pattern):
             return True

    # Check if it's a directory (we only want files)
    if not path.is_file():
        return True

    # Check against included extensions (if not directory)
    if path.suffix.lower() not in INCLUDED_EXTENSIONS:
        return True

    return False

def main():
    """Main function to collect source files."""
    base_path = pathlib.Path(__file__).parent
    project_root = base_path / PROJECT_DIR_NAME
    output_file_path = base_path / OUTPUT_FILENAME

    if not project_root.is_dir():
        print(f"ERROR: Project directory '{project_root}' not found.")
        print(f"Please run this script from the parent directory containing '{PROJECT_DIR_NAME}'.")
        return

    print(f"Starting source file collection for project: {project_root}")
    print(f"Output will be saved to: {output_file_path}")

    collected_files_count = 0
    try:
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            # Write a header for the snapshot file
            outfile.write(f"# Source Code Snapshot for: {PROJECT_DIR_NAME}\n")
            outfile.write(f"# Generated on: {datetime.datetime.now().isoformat()}\n")
            outfile.write("# Encoding: utf-8\n")
            outfile.write("\n" + "="*80 + "\n\n")

            # Walk through the project directory
            for item in sorted(project_root.rglob('*')): # Sort for consistent order
                if not should_exclude(item, project_root):
                    relative_item_path = item.relative_to(project_root)
                    header = f"# File: {relative_item_path.as_posix()}" # Use posix paths for consistency
                    print(f"  Adding: {header}")

                    outfile.write(header + "\n")
                    outfile.write("# " + "-"*len(header) + "\n") # Underline for clarity

                    try:
                        content = item.read_text(encoding="utf-8")
                        outfile.write(content)
                        outfile.write("\n\n" + "="*80 + "\n\n") # Separator between files
                        collected_files_count += 1
                    except Exception as read_err:
                        outfile.write(f"\n# ERROR: Could not read file content: {read_err}\n")
                        outfile.write("\n\n" + "="*80 + "\n\n") # Separator
                        print(f"  WARNING: Could not read {relative_item_path}: {read_err}")

    except IOError as write_err:
        print(f"\nERROR: Could not write to output file '{output_file_path}': {write_err}")
        return
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return

    print(f"\nSuccessfully collected content from {collected_files_count} files.")
    print(f"Snapshot saved to '{output_file_path}'.")

# Import datetime for the header timestamp
import datetime

if __name__ == "__main__":
    main()