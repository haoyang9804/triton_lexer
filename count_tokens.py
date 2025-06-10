import json
import tokenize
import io
from pathlib import Path
from typing import List, Tuple

def count_tokens_in_code(code: str) -> Tuple[int, List[str]]:
    """Count the number of tokens in a Python code string using Python's tokenize module."""
    try:
        # Convert string to bytes for tokenize
        code_bytes = code.encode('utf-8')
        tokens = []
        
        # Tokenize the code
        for token in tokenize.tokenize(io.BytesIO(code_bytes).readline):
            if token.type != tokenize.ENCODING:  # Skip encoding token
                tokens.append(token.string)
        
        return len(tokens), tokens
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        return 0, []

def main():
    # Read the JSON file
    json_path = Path("triton_kernels.json")
    if not json_path.exists():
        print("Error: triton_kernels.json not found")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        return

    total_tokens = 0
    kernel_count = 0
    all_tokens = []

    # Process each kernel
    for kernel in data:
        if 'source' in kernel:
            code = '.\n'.join(kernel['source'])
            print(code)
            token_count, tokens = count_tokens_in_code(code)
            total_tokens += token_count
            all_tokens.extend(tokens)
            kernel_count += 1

    print(f"Total number of kernels processed: {kernel_count}")
    print(f"Total number of tokens: {total_tokens}")
    print(f"Average tokens per kernel: {total_tokens/kernel_count if kernel_count > 0 else 0:.2f}")
    
    # Print token statistics
    unique_tokens = set(all_tokens)
    print(f"Number of unique tokens: {len(unique_tokens)}")
    
    # Print most common tokens (top 10)
    from collections import Counter
    token_freq = Counter(all_tokens)
    print("\nMost common tokens:")
    for token, count in token_freq.most_common(10):
        print(f"'{token}': {count}")

if __name__ == "__main__":
    main() 