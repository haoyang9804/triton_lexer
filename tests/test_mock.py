def test():
    import json
    import tqdm
    import subprocess
    with open("../encoded_kernels.json", "r") as f:
        codes = [k['code'] for k in json.load(f)['kernels']]
    for code in tqdm.tqdm(codes):
        with open('../test.py', 'w') as f:
            f.write(
"""
import mtriton.tl as tl
from mtriton import *
from mtriton.tl import *
import numpy as np
""")
            f.write(code)
        result = subprocess.run(['python', '../test.py'], capture_output=True, text=True)
        if result.returncode != 0 or result.stderr != '':
            print(result.stderr)
            print(f'return code: {result.returncode}')
            print(code)
            import sys 
            sys.exit(1)

if __name__ == "__main__":
    test()