# README

## Steps

1. Run `crawl.py` to crawl triton files from GitHub
2. Run `rename_files.py` to normalize the file names to avoid name collision
3. Run `remove_comments.py` to remove all comments from the renamed python files
4. Run `extract_kernels.py` to extract all kernel codes
5. Run `count_tokens.py` to count the overall tokens in the `triton_kernels.json`
6. (Optional) Run `header_body.py` to cut each kernel codes into kernel header and kernel body, both of which will be added into `triton_kernels.json`
7. Run `encoder.py` to encode the kernels.

## Token Count Result

```
Total number of kernels processed: 581
Total number of tokens: 397368
Average tokens per kernel: 683.94
Number of unique tokens: 3587

Most common tokens:
'.': 60315
'
': 41479
',': 39584
'(': 24826
')': 24826
'=': 17694
'tl': 15250
'*': 10389
':': 8449
'+': 7350
```

## Results

`codes` contains all renamed python files.
`triton_kernels.json` contains all kernel information extracted from the python files.