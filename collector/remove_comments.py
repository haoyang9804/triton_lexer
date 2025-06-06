import os
import argparse
from pathlib import Path
import pyparsing as pp
import subprocess

def remove_comments_and_docstrings(source):
    """
    Returns 'source' minus comments and docstrings.
    """
    # Define the grammar for Python comments and docstrings
    comment = pp.pythonStyleComment.suppress()
    docstring = pp.QuotedString('"""', multiline=True, unquoteResults=False).suppress() | \
                pp.QuotedString("'''", multiline=True, unquoteResults=False).suppress()
    
    # Combine the patterns
    filter_pattern = (comment | docstring)
    
    # Transform the source code
    return filter_pattern.transformString(source)

def process_file(file_path):
    """Process a single Python file"""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove comments and docstrings
        clean_code = remove_comments_and_docstrings(content)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(clean_code)
        
        # # Format the file using black
        subprocess.run(['black', file_path], check=True)
        
        print(f"Successfully processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def process_directory(directory):
    """Process all Python files in a directory"""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                process_file(file_path)

def main():
    parser = argparse.ArgumentParser(description='Remove comments and docstrings from Python files')
    parser.add_argument('--dir', required=True, help='Directory containing Python files')
    
    args = parser.parse_args()
    
    # Check if black is installed
    try:
        subprocess.run(['black', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: black is not installed. Please install it using 'pip install black'")
        return
    
    process_directory(args.dir)
    print("Processing completed")

if __name__ == "__main__":
    main()
