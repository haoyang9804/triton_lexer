# README

## Steps

1. Run `crawl.py` to crawl triton files from GitHub
2. Run `rename_files.py` to normalize the file names to avoid name collision
3. Run `remove_comments.py` to remove all comments from the renamed python files
4. Run `extract_kernels.py` to extract all kernel codes
5. Run `count_tokens.py` to count the overall tokens in the `triton_kernels.json`
6. Run `header_body.py` to cut each kernel codes into kernel header and kernel body, both of which will be added into `triton_kernels.json`

## Token Count Result

```
Total number of kernels processed: 480
Total number of tokens: 331791
Average tokens per kernel: 691.23
Number of unique tokens: 3338

Most common tokens:
'.': 50421
'
': 35044
',': 33144
'(': 20538
')': 20538
'=': 14217
'tl': 12245
'*': 9087
':': 6845
'+': 6430
```

## Results

`codes` contains all renamed python files.
`triton_kernels.json` contains all kernel information extracted from the python files.