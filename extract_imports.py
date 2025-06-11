import os
import ast
from typing import Set, Dict, List
from collections import defaultdict

def extract_imports_from_code(code: str) -> Dict[str, Set[str]]:
    tree = ast.parse(code)
    imports = {
        'torch': set(),
        'triton': set(),
        'other': set()
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name.startswith('torch'):
                    imports['torch'].add(f"import {name.name}")
                elif name.name == 'triton' or name.name.startswith('triton.'):
                    imports['triton'].add(f"import {name.name}")
                else:
                    imports['other'].add(f"import {name.name}")
                    
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith('torch'):
                for name in node.names:
                    imports['torch'].add(f"from {node.module} import {name.name}")
            elif node.module and (node.module == 'triton' or node.module.startswith('triton.')):
                for name in node.names:
                    imports['triton'].add(f"from {node.module} import {name.name}")
            else:
                for name in node.names:
                    imports['other'].add(f"from {node.module} import {name.name}")
                    
    return imports

def extract_imports(file_path: str) -> Dict[str, Set[str]]:
    """
    从Python文件中提取所有导入语句
    
    Args:
        file_path: Python文件路径
        
    Returns:
        Dict[str, Set[str]]: 导入语句的分类字典
            - 'torch': torch相关的导入
            - 'triton': triton相关的导入
            - 'other': 其他导入
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return extract_imports_from_code(content)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {'torch': set(), 'triton': set(), 'other': set()}

def process_directory(directory: str) -> Dict[str, Set[str]]:
    """
    处理目录中的所有Python文件
    
    Args:
        directory: 目录路径
        
    Returns:
        Dict[str, Set[str]]: 所有导入语句的汇总
    """
    all_imports = {
        'torch': set(),
        'triton': set(),
        'other': set()
    }
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_imports = extract_imports(file_path)
                
                # 合并导入语句
                for category in all_imports:
                    all_imports[category].update(file_imports[category])
                    
    return all_imports

def save_imports(imports: Dict[str, Set[str]], output_file: str):
    """
    将导入语句保存到文件
    
    Args:
        imports: 导入语句字典
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入torch相关导入
        f.write("# Torch related imports\n")
        for imp in sorted(imports['torch']):
            f.write(f"{imp}\n")
        f.write("\n")
        
        # 写入triton相关导入
        f.write("# Triton related imports\n")
        for imp in sorted(imports['triton']):
            f.write(f"{imp}\n")
        f.write("\n")
        
        # 写入其他导入
        f.write("# Other imports\n")
        for imp in sorted(imports['other']):
            f.write(f"{imp}\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract imports from Python files')
    parser.add_argument('--dir', required=True, help='Directory containing Python files')
    parser.add_argument('--output', default='imports.txt', help='Output file path')
    
    args = parser.parse_args()
    
    print(f"Processing directory: {args.dir}")
    imports = process_directory(args.dir)
    
    print("\nFound imports:")
    print(f"Torch related: {len(imports['torch'])}")
    print(f"Triton related: {len(imports['triton'])}")
    print(f"Other: {len(imports['other'])}")
    
    save_imports(imports, args.output)
    print(f"\nImports saved to: {args.output}")

if __name__ == '__main__':
    main() 