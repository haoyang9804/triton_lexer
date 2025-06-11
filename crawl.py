"""
This script is used to crawl the GitHub repositories that are related to Triton.
It will search the repositories that are related to Triton, and then download the Python files that are related to Triton.
It will save the downloaded files to the output directory.

Usage:
python crawl_github.py --token <your_github_token> --output <output_directory> --max-repos <max_number_of_repositories> --query <search_query>

Example:
python crawl_github.py --token <your_github_token> --output triton_repos --max-repos 100 --query "triton language:python"
"""
import os
import requests
import base64
from typing import List, Dict
import json
from pathlib import Path
import logging
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubSearcher:
    def __init__(self, api_key: str = None, output_dir: str = "triton_repos"):
        """
        Initialize the GitHub searcher
        
        Args:
            api_key: GitHub API key
            output_dir: Directory to save downloaded files
        """
        self.api_key = api_key
        self.headers = {
            'Accept': 'application/vnd.github.v3+json'
        }
        if api_key:
            self.headers['Authorization'] = f'token {api_key}'
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Rate limiting
        self.rate_limit_remaining = 30
        self.rate_limit_reset = 0
    
    def check_rate_limit(self):
        """Check and handle GitHub API rate limiting"""
        if self.rate_limit_remaining <= 1:
            reset_time = self.rate_limit_reset - int(time.time())
            if reset_time > 0:
                logger.info(f"Rate limit reached. Waiting {reset_time} seconds...")
                time.sleep(reset_time)
            self.rate_limit_remaining = 30
    
    def search_repositories(self, query: str, max_repos: int = 100) -> List[Dict]:
        """
        Search GitHub repositories
        
        Args:
            query: Search query
            max_repos: Maximum number of repositories to return
            
        Returns:
            List of repository information
        """
        repos = []
        page = 1
        per_page = 100
        
        while len(repos) < max_repos:
            self.check_rate_limit()
            
            url = f'https://api.github.com/search/repositories'
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'page': page,
                'per_page': per_page
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 0))
                self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
                
                if not data['items']:
                    break
                    
                repos.extend(data['items'])
                page += 1
            else:
                logger.error(f"Failed to search repositories: {response.status_code}")
                break
        
        return repos[:max_repos]
    
    def get_python_files(self, owner: str, repo: str) -> List[Dict]:
        """Get all Python files in a repository"""
        url = f'https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1'
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            tree = response.json()['tree']
            return [item for item in tree if item['path'].endswith('.py')]
        return []
    
    def get_file_content(self, owner: str, repo: str, path: str) -> str:
        """Get file content from GitHub"""
        url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            content = response.json()['content']
            return base64.b64decode(content).decode('utf-8')
        return None
    
    def is_triton_file(self, content: str) -> bool:
        """Check if file contains Triton-related code"""
        return content.find("@triton.jit") != -1 and (
            content.find("triton_dist") == -1 and
            content.find("triton.language.extra.cuda") == -1
        )
    
    def process_repository(self, repo_info: Dict) -> None:
        """Process a single repository"""
        try:
            owner = repo_info['owner']['login']
            repo = repo_info['name']
            logger.info(f"Processing repository: {owner}/{repo}")
            
            # Get Python files
            python_files = self.get_python_files(owner, repo)
            logger.info(f"Found {len(python_files)} Python files in {owner}/{repo}")
            
            # Process each file
            for file_info in python_files:
                content = self.get_file_content(owner, repo, file_info['path'])
                if content and self.is_triton_file(content):
                    # Create directory structure
                    repo_dir = os.path.join(self.output_dir, f"{owner}_{repo}")
                    file_path = os.path.join(repo_dir, file_info['path'])
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    # Save file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    logger.info(f"Saved Triton file: {owner}/{repo}/{file_info['path']}")
                
        except Exception as e:
            logger.error(f"Error processing repository {owner}/{repo}: {str(e)}")
    
    def crawl(self, query: str = "triton language:python", max_repos: int = 100):
        """Main crawling function"""
        logger.info(f"Starting to search repositories with query: {query}")
        
        # Search repositories
        repos = self.search_repositories(query, max_repos)
        logger.info(f"Found {len(repos)} repositories")
        
        # Process repositories in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(tqdm(
                executor.map(self.process_repository, repos),
                total=len(repos),
                desc="Processing repositories"
            ))
        
        logger.info("Crawling completed")

def main():
    parser = argparse.ArgumentParser(description='GitHub Triton Code Crawler')
    parser.add_argument('--token', type=str, help='GitHub API token')
    parser.add_argument('--output', type=str, default='triton_repos', help='Output directory')
    parser.add_argument('--max-repos', type=int, default=1000, help='Maximum number of repositories to process')
    parser.add_argument('--query', type=str, default='triton language:python', help='Search query')
    
    args = parser.parse_args()
    
    # Use environment variable if token not provided
    token = args.token or os.getenv('GITHUB_TOKEN')
    if not token:
        logger.warning("No GitHub token provided. Rate limits will be restricted.")
    
    searcher = GitHubSearcher(token, args.output)
    searcher.crawl(args.query, args.max_repos)

if __name__ == "__main__":
    main()
