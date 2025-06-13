import argparse
import logging
import os
import json
import tqdm

from crawl import GitHubSearcher
from rename_files import rename_files
from remove_comments import process_directory
from extract_kernels import TritonKernelExtractor
from encoder import TritonTokenEncoder, process_json_file, verify, add_tl_torch_funcs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, help='GitHub API token')

    args = parser.parse_args()
    
    # Use environment variable if token not provided
    token = args.token or os.getenv('GITHUB_TOKEN')
    if not token:
        logger.warning("No GitHub token provided. Rate limits will be restricted.")
    
    searcher = GitHubSearcher(token, 'triton_repos')
    searcher.crawl('triton language:python', 1000)
    logger.info("Crawling completed")
    
    rename_files('triton_repos', 1)
    logger.info("File renaming completed")
    
    process_directory('triton_repos')
    logger.info("Comments removal completed")
    
    extractor = TritonKernelExtractor()
    kernels = extractor.extract_from_directory('triton_repos')
    extractor.save_kernels(kernels, 'triton_kernels.json')
    logger.info("Kernel extraction completed")
    
    codes = process_json_file('triton_kernels.json')
    results = []
    start_id = -1
    if os.path.exists('endecoder.log') and os.path.getsize('endecoder.log') > 0:
        with open('endecoder.log', 'r') as f:
            start_id = int(f.read())
    start_id += 1
    if os.path.exists('encoded_kernels.json'):
        with open('encoded_kernels.json', 'r') as f:
            original_results = json.load(f)
            results = original_results['kernels']
    for i, code in enumerate(tqdm.tqdm(codes[start_id:])):
        add_tl_torch_funcs(code)
        encoder = TritonTokenEncoderDecoder()
        if os.path.exists('encoded_kernels.json'):
            with open('encoded_kernels.json', 'r') as f:
                original_results = json.load(f)
                encoder.mapping.update(original_results['mapping'])
        with open('endecoder.log', 'w') as f:
            f.write(f'{start_id + i}')
        encoder.tokenize(code)
        encoded = encoder.encode()
        verify(encoder.decode(encoded), code, results, encoder.mapping, 'encoded_kernels.json')
        results.append({
            'code': code,
            'encoded': encoded
        })
    if encoder:
        result = {'kernels': results, 'mapping': encoder.mapping}
        
        with open('encoded_kernels.json', 'w') as f:
            json.dump(result, f)

    
        
    