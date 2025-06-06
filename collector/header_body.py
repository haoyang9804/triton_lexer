import ast
import json
from typing import Tuple, List
from pathlib import Path

def split_header_body(source: str) -> Tuple[str, str]:
    """
    将 Triton kernel 源码分离为函数头和函数体。
    
    Args:
        source: Triton kernel 源码字符串
        
    Returns:
        Tuple[str, str]: (函数头, 函数体)
    """
    try:
        # 解析源码
        tree = ast.parse(source)
        
        # 获取函数定义节点
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break
                
        if not func_def:
            raise ValueError("No function definition found")
            
        # 获取参数列表
        args = []
        defaults = func_def.args.defaults
        # 补齐默认值列表
        num_no_default = len(func_def.args.args) - len(defaults)
        all_defaults = [None] * num_no_default + list(defaults)
        
        # 构建参数列表
        for arg, default in zip(func_def.args.args, all_defaults):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += ": " + ast.unparse(arg.annotation)
            if default is not None:
                arg_str += " = " + ast.unparse(default)
            args.append(arg_str)
            
        # 构建函数头
        header = f"def {func_def.name}({', '.join(args)}):"
        if func_def.returns:
            header = f"def {func_def.name}({', '.join(args)}) -> {ast.unparse(func_def.returns)}:"
            
        # 获取函数体
        body = ast.unparse(func_def.body)
        
        return header, body
        
    except Exception as e:
        raise ValueError(f"Failed to parse source code: {str(e)}")

def process_kernels(kernels: List[dict]) -> List[dict]:
    """
    处理多个 kernel，分离它们的函数头和函数体。
    
    Args:
        kernels: 包含 kernel 源码的字典列表
        
    Returns:
        List[dict]: 处理后的 kernel 列表，每个 kernel 包含 header 和 body
    """
    processed_kernels = []
    
    for kernel in kernels:
        try:
            source = kernel['code']
            header, body = split_header_body(source)
            
            processed_kernel = {
                'code': source,
                'header': header,
                'body': body
            }
            processed_kernels.append(processed_kernel)
            
        except Exception as e:
            print(f"Error processing kernel: {str(e)}")
            continue
            
    return processed_kernels

def process_json_file(input_file: str, output_file: str = None):
    """
    处理 JSON 文件中的 kernel 源码，添加 header 和 body 字段。
    
    Args:
        input_file: 输入 JSON 文件路径
        output_file: 输出 JSON 文件路径，如果为 None 则覆盖输入文件
    """
    try:
        # 读取 JSON 文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 处理每个 kernel
        for item in data:
            if 'source' in item:
                try:
                    header, body = split_header_body('\n'.join(item['source']))
                    item['header'] = header
                    item['body'] = body
                except Exception as e:
                    print(f"Error processing kernel ({item['file']}): {str(e)}")
                    continue
                    
        # 保存结果
        output_file = output_file or input_file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully processed {len(data)} kernels")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":

  #   a = [
  #     "    def forward(",
  #     "        output_ptr: tl.tensor,",
  #     "        input_ptr: tl.tensor,",
  #     "        y_size: tl.int32,",
  #     "        x_size: tl.int32,",
  #     "        y_stride: tl.int32,",
  #     "        x_stride: tl.int32,",
  #     "        dtype: tl.constexpr,",
  #     "        x_block_size: tl.constexpr,",
  #     "        require_x_boundary_check: tl.constexpr,",
  #     "    ):",
  #     "        y_offset = tl.program_id(0)",
  #     "",
  #     "        output_block_ptr = tl.make_block_ptr(",
  #     "            output_ptr,",
  #     "            shape=(y_size,),",
  #     "            strides=(1,),",
  #     "            offsets=(y_offset,),",
  #     "            block_shape=(1,),",
  #     "            order=(0,),",
  #     "        )",
  #     "        input_block_ptr = tl.make_block_ptr(",
  #     "            input_ptr,",
  #     "            shape=(y_size, x_size),",
  #     "            strides=(y_stride, x_stride),",
  #     "            offsets=(y_offset, 0),",
  #     "            block_shape=(1, x_block_size),",
  #     "            order=(1, 0),",
  #     "        )",
  #     "",
  #     "        if require_x_boundary_check:",
  #     "            input = tl.load(input_block_ptr, boundary_check=(1,))",
  #     "            condition = tl.arange(0, x_block_size) < x_size",
  #     "            input = tl.where(condition, input, float(\"-inf\"))",
  #     "        else:",
  #     "            input = tl.load(input_block_ptr)",
  #     "",
  #     "        output = tl.argmax(input, 1)",
  #     "        tl.store(output_block_ptr, output.to(dtype))"
  # ]

  #   test_code = '\n'.join(a)
  #   print(test_code)

  #   header, body = split_header_body(test_code)
  #   print("Header:")
  #   print(header)
  #   print("\nBody:")
  #   print(body)
    
    # 处理 JSON 文件
    json_file = "triton_kernels.json"
    if Path(json_file).exists():
        process_json_file(json_file)
    else:
        print(f"File {json_file} not found")
