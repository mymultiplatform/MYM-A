import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

def find_python_files(root_path):
    """Recursively find all Python files in the directory and its subdirectories."""
    python_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                # Store tuple of (relative path for display, full path for reading)
                rel_path = os.path.relpath(full_path, root_path)
                python_files.append((rel_path, full_path))
    return sorted(python_files)

# Get folder path from user
folder_path = input("Paste your folder path here: ").strip()
# Remove quotes if user copied path with quotes 
folder_path = folder_path.strip('"\'')

# Create a new notebook
nb = new_notebook()

# Find all Python files recursively
py_files = find_python_files(folder_path)

if not py_files:
    print("No Python files found!")
else:
    print(f"\nFound {len(py_files)} Python files:")
    for rel_path, _ in py_files:
        print(f"- {rel_path}")

    # Process each Python file
    for rel_path, full_path in py_files:
        # Add markdown cell with filename and relative path
        header = f"# File: {rel_path}\n---"
        nb.cells.append(nbformat.v4.new_markdown_cell(header))
        
        # Read and add the Python file content
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
                content = file.read()
                nb.cells.append(nbformat.v4.new_code_cell(content))
        except Exception as e:
            error_msg = f"Error reading {rel_path}: {str(e)}"
            print(error_msg)
            nb.cells.append(nbformat.v4.new_markdown_cell(f"**{error_msg}**"))

    # Save the notebook in the root folder
    output_file = os.path.join(folder_path, 'combined_notebook.ipynb')
    with open(output_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"\nCreated notebook: {output_file}")