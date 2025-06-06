import json

def simplify_kernels():
    # 读取原始文件
    with open('triton_kernels.json', 'r') as f:
        kernels = json.load(f)
    
    # 简化的内核列表
    simplified_kernels = []
    
    for kernel in kernels:
        simplified_kernel = {
            'source': kernel['source']
        }
        simplified_kernels.append(simplified_kernel)
    
    # 写入新文件
    with open('simplified_triton_kernels.json', 'w') as f:
        json.dump(simplified_kernels, f, indent=2)

if __name__ == '__main__':
    simplify_kernels()
