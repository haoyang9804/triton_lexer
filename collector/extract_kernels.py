import os
import ast
import logging
from typing import List, Dict, Tuple
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonKernelExtractor:
    def __init__(self):
        self.kernels = []
    
    def is_triton_decorator(self, decorator: ast.Call) -> bool:
        """Check if a decorator is a Triton decorator"""
        if isinstance(decorator.func, ast.Attribute):
            return decorator.func.value.id == 'triton'
        return False
    
    def extract_kernel_info(self, node: ast.FunctionDef) -> Dict:
        """Extract information from a Triton kernel function"""
        kernel_info = {
            'name': node.name,
            'decorators': [],
            'decorator_source': [],
            'args': [],
            'docstring': ast.get_docstring(node),
            'source': None
        }
        
        # Extract decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and self.is_triton_decorator(decorator):
                decorator_info = {
                    'name': decorator.func.attr,
                    'args': [],
                    'keywords': {}
                }
                
                # Extract positional arguments
                for arg in decorator.args:
                    if isinstance(arg, ast.Constant):
                        decorator_info['args'].append(arg.value)
                
                # Extract keyword arguments
                for keyword in decorator.keywords:
                    if isinstance(keyword.value, ast.Constant):
                        decorator_info['keywords'][keyword.arg] = keyword.value.value
                
                kernel_info['decorators'].append(decorator_info)
        
        # Extract function arguments
        for arg in node.args.args:
            arg_info = {
                'name': arg.arg,
                'annotation': ast.unparse(arg.annotation) if arg.annotation else None
            }
            kernel_info['args'].append(arg_info)
        
        return kernel_info
    
    def get_decorator_source(self, decorator: ast.Call) -> str:
        """Get the source code of a decorator"""
        if isinstance(decorator.func, ast.Attribute):
            decorator_str = f"@{decorator.func.value.id}.{decorator.func.attr}"
        else:
            decorator_str = f"@{ast.unparse(decorator.func)}"
        
        # Add arguments
        args = []
        for arg in decorator.args:
            args.append(ast.unparse(arg))
        
        # Add keyword arguments
        for keyword in decorator.keywords:
            args.append(f"{keyword.arg}={ast.unparse(keyword.value)}")
        
        if args:
            decorator_str += f"({', '.join(args)})"
        
        return decorator_str
    
    def reformat_source(self, source_lines: List[str]) -> List[str]:
        """
        重新格式化 kernel 源码，移除额外的缩进。
        
        Args:
            source_lines: kernel 源码的每一行
            
        Returns:
            List[str]: 重新格式化后的源码行
        """
        if not source_lines:
            return []
            
        # 找到第一行的缩进
        first_line = source_lines[0]
        base_indent = len(first_line) - len(first_line.lstrip())
        
        # 移除所有行的基础缩进
        reformatted_lines = []
        for line in source_lines:
            if line.strip():  # 非空行
                reformatted_lines.append(line[base_indent:])
            else:  # 空行
                reformatted_lines.append("")
                
        return reformatted_lines

    def process_file(self, file_path: str) -> List[Dict]:
        """Process a single Python file and extract Triton kernels"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            file_kernels = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get only the direct Triton decorators
                    triton_decorators = [
                        d for d in node.decorator_list
                        if isinstance(d, ast.Call) and self.is_triton_decorator(d)
                    ]
                    
                    if triton_decorators:
                        kernel_info = self.extract_kernel_info(node)
                        kernel_info['file'] = file_path
                        
                        # Get only the direct decorator source code
                        decorator_sources = []
                        for decorator in triton_decorators:
                            decorator_sources.append(self.get_decorator_source(decorator))
                        kernel_info['decorator_source'] = decorator_sources
                        
                        # Get function body source code and reformat it
                        source_lines = content.split('\n')[node.lineno-1:node.end_lineno]
                        kernel_info['source'] = self.reformat_source(source_lines)
                        
                        file_kernels.append(kernel_info)
            
            return file_kernels
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return []
    
    def extract_from_directory(self, directory: str) -> List[Dict]:
        """Extract Triton kernels from all Python files in a directory"""
        all_kernels = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    kernels = self.process_file(file_path)
                    all_kernels.extend(kernels)
                    logger.info(f"Found {len(kernels)} kernels in {file_path}")
        
        return all_kernels
    
    def save_kernels(self, kernels: List[Dict], output_file: str):
        """Save extracted kernels to a JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(kernels, f, indent=2)
            logger.info(f"Saved {len(kernels)} kernels to {output_file}")
        except Exception as e:
            logger.error(f"Error saving kernels: {str(e)}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract Triton kernels from Python files')
    parser.add_argument('--dir', required=True, help='Directory containing Python files')
    parser.add_argument('--output', default='triton_kernels.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    extractor = TritonKernelExtractor()
    kernels = extractor.extract_from_directory(args.dir)
    extractor.save_kernels(kernels, args.output)

if __name__ == "__main__":
    main() 