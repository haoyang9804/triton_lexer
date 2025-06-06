import os
import shutil
from pathlib import Path
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rename_files(source_dir: str, start_index: int = 1) -> int:
    """
    Rename files in the source directory
    
    Args:
        source_dir: Source directory containing files
        start_index: Starting index for file numbering
        
    Returns:
        int: Next available index
    """
    if not os.path.exists(source_dir):
        logger.error(f"Directory {source_dir} does not exist")
        return start_index
    
    # Get all Python files recursively
    python_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Sort files to ensure consistent ordering
    python_files.sort()
    
    # Rename files
    current_index = start_index
    for file_path in python_files:
        try:
            # Create new filename
            new_filename = f"{current_index}.py"
            new_path = os.path.join(os.path.dirname(file_path), new_filename)
            
            # Rename file
            os.rename(file_path, new_path)
            logger.info(f"Renamed: {file_path} -> {new_path}")
            
            current_index += 1
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    return current_index

def main():
    parser = argparse.ArgumentParser(description='Rename Python files in directories')
    parser.add_argument('--dirs', nargs='+', required=True, help='Directories to process')
    parser.add_argument('--start-index', type=int, default=1, help='Starting index for file numbering')
    
    args = parser.parse_args()
    
    current_index = args.start_index
    for directory in args.dirs:
        logger.info(f"Processing directory: {directory}")
        current_index = rename_files(directory, current_index)
    
    logger.info("File renaming completed")

if __name__ == "__main__":
    main() 