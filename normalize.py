import ast
from typing import Dict, List, Tuple
import re
import builtins

class VariableNormalizer(ast.NodeTransformer):
    def __init__(self):
        self.variable_counter = 1
        self.function_counter = 0
        self.variable_map: Dict[str, str] = {}
        self.function_map: Dict[str, str] = {}
        # 创建内置函数和关键字的集合
        self.builtin_names = set(dir(builtins))
        self.keywords = {'True', 'False', 'None', 'and', 'or', 'not', 'is', 'in', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'def', 'class', 'return', 'break', 'continue', 'pass', 'raise', 'import', 'from', 'as', 'global', 'nonlocal', 'lambda', 'yield', 'async', 'await', 'tl'}
        
    def visit_Name(self, node: ast.Name) -> ast.Name:
        print(node.id)
        # 跳过内置函数和关键字
        if node.id in self.builtin_names or node.id in self.keywords:
            return node
            
        # 如果变量名已经在映射中，使用映射后的名称
        if node.id in self.variable_map:
            node.id = self.variable_map[node.id]
        else:
            # 创建新的变量名映射
            new_name = f'v{self.variable_counter}'
            self.variable_map[node.id] = new_name
            self.variable_counter += 1
            node.id = new_name
        return node
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # 处理函数名
        if node.name in self.function_map:
            node.name = self.function_map[node.name]
        else:
            new_name = f'f{self.function_counter}'
            self.function_map[node.name] = new_name
            self.function_counter += 1
            node.name = new_name
            
        # 处理函数体
        self.generic_visit(node)
        return node
        
    def visit_arg(self, node: ast.arg) -> ast.arg:
        # 处理函数参数
        if node.arg in self.variable_map:
            node.arg = self.variable_map[node.arg]
        else:
            new_name = f'v{self.variable_counter}'
            self.variable_map[node.arg] = new_name
            self.variable_counter += 1
            node.arg = new_name
        return node

def normalize_code(code: str) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    """
    规范化代码中的变量名和函数名
    
    Args:
        code: 输入的Python代码字符串
        
    Returns:
        Tuple[str, Dict[str, str], Dict[str, str]]: 
            - 规范化后的代码
            - 变量名映射字典
            - 函数名映射字典
    """
    try:
        # 解析代码
        tree = ast.parse(code)
        
        # 创建规范化器并转换AST
        normalizer = VariableNormalizer()
        normalized_tree = normalizer.visit(tree)
        
        # 将AST转换回代码字符串
        normalized_code = ast.unparse(normalized_tree)
        
        return normalized_code, normalizer.variable_map, normalizer.function_map
        
    except Exception as e:
        print(f"Error normalizing code: {e}")
        return code, {}, {}

def main():
    # 测试代码
    test_code = """
@triton.jit
def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
    """
    
    normalized_code, var_map, func_map = normalize_code(test_code)
    
    print("Normalized Code:")
    print(normalized_code)
    print("\nVariable Mapping:")
    for old, new in var_map.items():
        print(f"{old} -> {new}")
    print("\nFunction Mapping:")
    for old, new in func_map.items():
        print(f"{old} -> {new}")

if __name__ == "__main__":
    main()
