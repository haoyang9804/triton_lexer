import json

def triton_keyword_to_str(keyword: str) -> str:
    if keyword.startswith('TL__'):
        return 'tl.' + keyword[4:].replace('_', '.').lower()
    return keyword.replace('_', '.').lower()

with open('encoded_kernels.json', 'r') as f:
    encoded = json.load(f)
mapping = encoded['mapping']
reverse_mapping = {v: k for k, v in mapping.items()}
for k, v in mapping.items():
    if k.startswith('TL_'):
        print(triton_keyword_to_str(k))