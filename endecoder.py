import ply.lex as lex
from typing import List, Dict, Any, Tuple
from ply.lex import LexToken
import argparse
import json
import sys
import astwrite
import tqdm
import os
import re

def to_token(name: str) -> str:
    tok = ''
    for c in name:
        if c.isalpha():
            tok += c.upper()
        elif c == '.':
            tok += '_'
        else:
            tok += c
    return tok

# List of token names
tokens = (
    # Basic tokens
    'ID', 'INT', 'FLOAT', 'STRING',
    
    # Operators
    'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'DDVIDE',
    'GT', 'LT', 'EQ', 'ASSIGN', 'MOD', 'POW', 'BAND',
    'BOR', 'BXOR', 'LSHIFT', 'RSHIFT', 'EXCLAMATION',
    'TILDE',
    'PLUSEQ', 'MINUSEQ', 'TIMESEQ', 'DIVEQ', 'DDVIDEEQ', 
    'GTEQ', 'LTEQ', 'EQEQ', 'NEQ',
    'MODEQ', 'POWEQ', 'BANDEQ', 'BOREQ', 'BXOREQ', 'LSHIFTEQ',
    'RSHIFTEQ', 'TILDEEQ',
    
    # Delimiters
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
    'COLON', 'COMMA', 'DOT', 'ENTER',
    'LBRACKET', 'RBRACKET',
    
    # Blocks
    'DEF', 'IF', 'ELSE', 'ELIF', 'FOR', 'WHILE', 'WITH', 'TRY', 'EXCEPT', 'FINALLY',
    'ENDDEF', 'ENDIF', 'ENDELIF', 'ENDELSE', 'ENDFOR', 'ENDWHILE', 'ENDWITH', 'ENDTRY', 'ENDEXCEPT', 'ENDFINALLY',
    
    
    # Other Python keywords
    'IN', 'RANGE', 'RETURN', 'NOT', 'AND', 'OR', 'IS', 'NONE',
    'TRUE', 'FALSE', 'BREAK', 'CONTINUE', 'PASS', 'YIELD',
    
    # Built-in functions
    'MIN', 'PRINT', 'SUM', 'ANY', 'ALL', 'ABS', 'MAX',
    'ROUND', 'DIVMOD', 'LEN', 'ZIP', 'SLICE',
    
    # Type related
    'STR', 'TYPE', 'ISINSTANCE', 'TUPLE', 'LIST',
    'DICT', 'SET',
    
    # Triton specific
    'DL', 'LIBDEVICE', 'LIBSHMEM_DEVICE',
    'TRITON_HELPERS', 'LANGUAGE', 'TRITON', 'CEIL_DIV',
    'EXP', 'LOG', 'ATOMIC_ADD', 'ATOMIC_CAS', 'ATOMIC_MIN',
    'ATOMIC_XCHG', 'LD', 'ST', 'TID',
    'SHFL_DOWN_SYNC_I32', 'SHFL_UP_SYNC_I32', 'SHFL_SYNC_I32',
    'BALLOT_SYNC', 'FFS', 'SYNCTHREADS', 'FP_DOWNCAST_ROUNDING',
    'FP_UPCAST_ROUNDING', 'ELEMENT_TY', 'ELEMENT_TX',
    
    # Special string literals
    'CUDA_STR', 'CPU_STR', 'GPU_STR', 'MPS_STR', 'XPU_STR', 'IPU_STR', 'HPU_STR', 'MTIA_STR',
    'FLOAT32_STR', 'FLOAT64_STR', 'FLOAT16_STR', 'BFLOAT16_STR', 'INT8_STR', 'INT16_STR', 'INT32_STR', 'INT64_STR',
    'UINT8_STR', 'BOOL_STR', 'COMPLEX64_STR', 'COMPLEX128_STR',
    
    # Special argument keywords
    'DEVICE', 'DTYPE', 'REQUIRES_GRAD', 'PIN_MEMORY', 'NON_BLOCKING',
    'OUTPUT', 'DIM', 'KEEP_DIM', 'UNSQUEEZE', 'SQUEEZE',
    'STRIDE', 'SIZE', 'SHAPE', 'NUMEL', 'NDIM',
    'MEMORY_FORMAT', 'LAYOUT', 'GRAD_FN', 'GRAD',
)

def is_block_start(token: str) -> bool:
    return token == 'DEF' or token == 'IF' or token == 'ELSE' or token == 'ELIF' or token == 'FOR' or token == 'WHILE' or token == 'WITH' or token == 'TRY' or token == 'EXCEPT' or token == 'FINALLY'

def is_block_end(token: str) -> bool:
    return str(token).startswith('END') and len(str(token)) > 3

def token_to_str(token: str) -> str:
    dic = {
        'PLUS': '+',
        'MINUS': '-',
        'TIMES': '*',
        'DIVIDE': '/',
        'DDVIDE': '//',
        'GT': '>',
        'LT': '<',
        'EQ': '==',
        'ASSIGN': '=',
        'MOD': '%',
        'POW': '**',
        'BAND': '&',
        'BOR': '|',
        'BXOR': '^',
        'LSHIFT': '<<',
        'RSHIFT': '>>',
        'LPAREN': '(',
        'RPAREN': ')',
        'LBRACE': '{',
        'RBRACE': '}',
        'LBRACKET': '[',
        'RBRACKET': ']',
        'COLON': ':',
        'COMMA': ',',
        'DOT': '.',
        'LBRACKET': '[',
        'RBRACKET': ']',
        'EXCLAMATION': '!',
        'TILDE': '~',
        'GTEQ': '>=',
        'LTEQ': '<=',
        'EQEQ': '==',
        'NEQ': '!=',
        'PLUSEQ': '+=',
        'MINUSEQ': '-=',
        'TIMESEQ': '*=',
        'DIVEQ': '/=',
        'DDVIDEEQ': '//=',
        'MODEQ': '%=',
        'POWEQ': '**=',
        'BANDEQ': '&=',
        'BOREQ': '|=',
        'BXOREQ': '^=',
    }
    return dic.get(token, token)

class TLFuncLexer:
    def __init__(self):
        self.tl_pattern = re.compile(r'tl(?:\.[a-zA-Z_][a-zA-Z_0-9]*)*')
        self.tl_funcs = set()
        self.torch_pattern = re.compile(r'torch(?:\.[a-zA-Z_][a-zA-Z_0-9]*)*')
        self.torch_funcs = set()
    
    def tokenize(self, code: str) -> None:
        matches = self.tl_pattern.findall(code)
        for match in matches:
            self.tl_funcs.add(match)
        matches = self.torch_pattern.findall(code)
        for match in matches:
            self.torch_funcs.add(match)
    
class TritonLexer:
    # List of token names
    tokens = tokens
    
    # Regular expression rules for simple tokens
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_DDVIDE = r'//'
    t_GT = r'>'
    t_LT = r'<'
    t_EQ = r'=='
    t_ASSIGN = r'='
    t_MOD = r'%'
    t_POW = r'\*\*'
    t_BAND = r'&'
    t_BOR = r'\|'
    t_BXOR = r'\^'
    t_LSHIFT = r'<<'
    t_RSHIFT = r'>>'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_COLON = r':'
    t_COMMA = r','
    t_DOT = r'\.'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_EXCLAMATION = r'!'
    t_TILDE = r'~'
    t_GTEQ = r'>='
    t_LTEQ = r'<='
    t_EQEQ = r'=='
    t_NEQ = r'!='
    t_PLUSEQ = r'\+='
    t_MINUSEQ = r'-='
    t_TIMESEQ = r'\*='
    t_DIVEQ = r'/='
    t_DDVIDEEQ = r'//='
    t_MODEQ = r'%='
    t_POWEQ = r'\*\*='
    t_BANDEQ = r'&='
    t_BOREQ = r'\|='
    t_BXOREQ = r'\^='
    t_LSHIFTEQ = r'<<='
    t_RSHIFTEQ = r'>>='
    t_TILDEEQ = r'~='
    
    # Keywords
    keywords = {
        # Python keywords
        'def': 'DEF',
        'if': 'IF',
        'while': 'WHILE',
        'else': 'ELSE',
        'elif': 'ELIF',
        'for': 'FOR',
        'in': 'IN',
        'range': 'RANGE',
        'return': 'RETURN',
        'not': 'NOT',
        'and': 'AND',
        'or': 'OR',
        'is': 'IS',
        'None': 'NONE',
        'True': 'TRUE',
        'False': 'FALSE',
        'break': 'BREAK',
        'continue': 'CONTINUE',
        'pass': 'PASS',
        'yield': 'YIELD',
        
        # End
        'enddef': 'ENDDEF',
        'endif': 'ENDIF',
        'endelse': 'ENDELSE',
        'endelif': 'ENDELIF',
        'endfor': 'ENDFOR',
        'endwhile': 'ENDWHILE',
        'endwith': 'ENDWITH',
        'endtry': 'ENDTRY',
        'endexcept': 'ENDEXCEPT',
        'endfinally': 'ENDFINALLY',
       
        # Built-in functions
        'min': 'MIN',
        'print': 'PRINT',
        'sum': 'SUM',
        'any': 'ANY',
        'all': 'ALL',
        'abs': 'ABS',
        'max': 'MAX',
        'round': 'ROUND',
        'divmod': 'DIVMOD',
        'len': 'LEN',
        'zip': 'ZIP',
        'slice': 'SLICE',
        
        # Type related
        'str': 'STR',
        'type': 'TYPE',
        'isinstance': 'ISINSTANCE',
        'tuple': 'TUPLE',
        'list': 'LIST',
        'dict': 'DICT',
        'set': 'SET',
        
        # Torch specific
        'torch': 'TORCH',
        'torch.tensor': 'TORCH_TENSOR',
        'torch.device': 'TORCH_DEVICE',
        'torch.dtype': 'TORCH_DTYPE',
        'torch.layout': 'TORCH_LAYOUT',
        'torch.memory_format': 'TORCH_MEMORY_FORMAT',
        'torch.grad': 'TORCH_GRAD',
        'torch.requires_grad': 'TORCH_REQUIRES_GRAD',
        'torch.backward': 'TORCH_BACKWARD',
        'torch.detach': 'TORCH_DETACH',
        'torch.clone': 'TORCH_CLONE',
        'torch.contiguous': 'TORCH_CONTIGUOUS',
        'torch.cpu': 'TORCH_CPU',
        'torch.cuda': 'TORCH_CUDA',
        'torch.gpu': 'TORCH_GPU',
        'torch.cpu_device': 'TORCH_CPU_DEVICE',
        'torch.cuda_device': 'TORCH_CUDA_DEVICE',
        'torch.float32': 'TORCH_FLOAT32',
        'torch.float64': 'TORCH_FLOAT64',
        'torch.float16': 'TORCH_FLOAT16',
        'torch.bfloat16': 'TORCH_BFLOAT16',
        'torch.int8': 'TORCH_INT8',
        'torch.int16': 'TORCH_INT16',
        'torch.int32': 'TORCH_INT32',
        'torch.int64': 'TORCH_INT64',
        'torch.uint8': 'TORCH_UINT8',
        'torch.bool': 'TORCH_BOOL',
        'torch.complex64': 'TORCH_COMPLEX64',
        'torch.complex128': 'TORCH_COMPLEX128',
        'torch.quint8': 'TORCH_QUINT8',
        'torch.qint8': 'TORCH_QINT8',
        'torch.qint32': 'TORCH_QINT32',
        
        # Triton specific
        'dl': 'DL',
        'libdevice': 'LIBDEVICE',
        'libshmem_device': 'LIBSHMEM_DEVICE',
        'triton_helpers': 'TRITON_HELPERS',
        'language': 'LANGUAGE',
        'triton': 'TRITON',
        'ceil_div': 'CEIL_DIV',
        'exp': 'EXP',
        'log': 'LOG',
        'atomic_add': 'ATOMIC_ADD',
        'atomic_cas': 'ATOMIC_CAS',
        'atomic_min': 'ATOMIC_MIN',
        'atomic_xchg': 'ATOMIC_XCHG',
        'ld': 'LD',
        'st': 'ST',
        'tid': 'TID',
        'shfl_down_sync_i32': 'SHFL_DOWN_SYNC_I32',
        'shfl_up_sync_i32': 'SHFL_UP_SYNC_I32',
        'shfl_sync_i32': 'SHFL_SYNC_I32',
        'ballot_sync': 'BALLOT_SYNC',
        'ffs': 'FFS',
        '.to': 'DOTTO',
        '__syncthreads': 'SYNCTHREADS',
        'fp_downcast_rounding': 'FP_DOWNCAST_ROUNDING',
        'fp_upcast_rounding': 'FP_UPCAST_ROUNDING',
        'element_ty': 'ELEMENT_TY',
        'element_tx': 'ELEMENT_TX',
        
        # Special argument keywords
        'device': 'DEVICE',
        'dtype': 'DTYPE',
        'requires_grad': 'REQUIRES_GRAD',
        'pin_memory': 'PIN_MEMORY',
        'non_blocking': 'NON_BLOCKING',
        'out': 'OUTPUT',
        'dim': 'DIM',
        'keepdim': 'KEEP_DIM',
        'unsqueeze': 'UNSQUEEZE',
        'squeeze': 'SQUEEZE',
        'stride': 'STRIDE',
        'size': 'SIZE',
        'shape': 'SHAPE',
        'numel': 'NUMEL',
        'ndim': 'NDIM',
        'memory_format': 'MEMORY_FORMAT',
        'layout': 'LAYOUT',
        'grad_fn': 'GRAD_FN',
        'grad': 'GRAD',
    }
    
    # keywords = {k: v for k, v in zip(triton_keywords, tokens)}
    
    # Special string literals that should be treated as tokens
    special_strings = {
        'cuda': 'CUDA_STR',
        'cpu': 'CPU_STR',
        'gpu': 'GPU_STR',
        'mps': 'MPS_STR',
        'xpu': 'XPU_STR',
        'ipu': 'IPU_STR',
        'hpu': 'HPU_STR',
        'mtia': 'MTIA_STR',
        'float32': 'FLOAT32_STR',
        'float64': 'FLOAT64_STR',
        'float16': 'FLOAT16_STR',
        'bfloat16': 'BFLOAT16_STR',
        'int8': 'INT8_STR',
        'int16': 'INT16_STR',
        'int32': 'INT32_STR',
        'int64': 'INT64_STR',
        'uint8': 'UINT8_STR',
        'bool': 'BOOL_STR',
        'complex64': 'COMPLEX64_STR',
        'complex128': 'COMPLEX128_STR',
    }
    
    keywords.update(special_strings)
    
    # Regular expression rules with some action code
    def t_STRING(self, t):
        r'"[^"]*"|\'[^\']*\''
        # Check if the string content (without quotes) is a special string
        content = t.value[1:-1]  # Remove quotes
        if content in self.special_strings:
            t.type = self.special_strings[content]
            t.value = t.value  # 保留原始字符串（包括引号）
        else:
            t.value = t.value  # 保留原始字符串（包括引号）
        # 只有当内容不为空时才返回 token
        if content:
            return t
        return None  # 忽略空字符串
    
    def t_TL_FUNC(self, t):
        r'tl(\.[a-zA-Z_0-9]*)*'
        t.type = self.keywords.get(t.value, 'ERROR')
        return t
    
    def t_TORCH_FUNC(self, t):
        r'torch(\.[a-zA-Z_0-9]*)*'
        t.type = self.keywords.get(t.value, 'ERROR')
        return t
    
    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = self.keywords.get(t.value, 'ID')
        return t
    
    def t_FLOAT(self, t):
        r'(\d*\.\d+([eE][-+]?\d+)?|\d+[eE][-+]?\d+)'
        t.value = float(t.value)
        return t
    
    def t_INT(self, t):
        r'\d'
        t.value = int(t.value)
        return t
    
    # Define a rule so we can track line numbers
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
    
    # A string containing ignored characters (spaces and tabs)
    t_ignore = ' \t'
    
    # Error handling rule
    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}'")
        t.lexer.skip(1)
    
    def __init__(self):
        self.lexer = lex.lex(module=self)

class TritonTokenEncoderDecoder:
    def __init__(self):
        self.mapping = {}
        for i, v in enumerate(tokens):
            self.mapping[v] = i + 1
        self.next_value = len(self.mapping) + 1
        # add 50 variables
        for i in range(100):
            self.mapping[f'v{i}'] = self.next_value
            self.next_value += 1
        # add 5 functions
        for i in range(1):
            self.mapping[f'f{i}'] = self.next_value
            self.next_value += 1
        self.lexer = TritonLexer().lexer
        self.oldName2NewName = {} # nomalize identifiers' names: original name -> normalized name
        self.varid = 0
        self.fid = 0
        self.tokens = []
    
    def print_mapping(self):
        for k, v in self.mapping.items():
            print(f'{k}: {v}')
    
    def _new_token(self, type: str, value: str, lineno: int, lexpos: int):
        tok = LexToken()
        tok.type = type
        tok.value = value
        tok.lineno = lineno
        tok.lexpos = lexpos
        return tok

    
    def tokenize(self, string: str) -> int:
        self.tokens = []
        # For each block, add the end mark.
        # e.g., add endwhile for a while loop
        string = astwrite.add_end_marks(string)
        strs = string.split('\n')
        for linenum, s in enumerate(strs):
            last_token = None
            self.lexer.input(s)
            while True:
                tok = self.lexer.token()
                if not tok:
                    break
                tok.lineno = linenum + 1
                if tok.type == 'ID':
                    if self.oldName2NewName.get(tok.value) is not None:
                        tok.value = self.oldName2NewName[tok.value]
                    else:
                        if last_token and last_token.value == 'def':
                            fname = f'f{self.fid}'
                            self.oldName2NewName[tok.value] = fname
                            tok.value = fname
                            self.fid += 1
                        else:
                            vname = f'v{self.varid}'
                            self.oldName2NewName[tok.value] = vname
                            tok.value = vname
                            self.varid += 1
                self.tokens.append(tok)
                last_token = tok
            end_token = self._new_token('ENTER', '<enter>', -1, -1)
            self.tokens.append(end_token)
        return self.tokens

    def encode(self) -> List[int]:
        encoded = []
        for token in self.tokens:
            assert token.type in self.mapping, f'{token.type} not in mapping'
            if  token.type == 'ID' or token.type == 'STRING' or token.type == 'FLOAT' or token.type == 'INT':
                if self.mapping.get(token.value) is not None:
                    encoded.append(self.mapping[token.value])
                else:
                    self.mapping[token.value] = self.next_value
                    encoded.append(self.next_value)
                    self.next_value += 1
            else:
                encoded.append(self.mapping[token.type])
        """
        Now we need to add a delimiter between the kernel header and the kernel body.
        The encoded value of the delimiter is -1.
        """
        final_encoded = []
        find_def = False
        for i, v in enumerate(encoded):
            if v == self.mapping['DEF']:
                find_def = True
            final_encoded.append(v)
            # if this is the end of the kernel header
            if find_def and v == self.mapping['ENTER'] and i >= 2 and \
                encoded[i - 2] == self.mapping['RPAREN'] and encoded[i - 1] == self.mapping['COLON']:
                final_encoded.append(-1)
                find_def = False
        return final_encoded
    
    def decode(self, encoded: List[int]) -> str:
        reverse_mapping = {v: k for k, v in self.mapping.items()}
        token2value = {v: k for k, v in TritonLexer.keywords.items()}
        program = ''
        indent = 0
        first_token_of_the_line = True
        i = 0
        while i < len(encoded):
            v = encoded[i]
            if v == -1:
                i += 1
                continue
            if is_block_end(reverse_mapping[v]):
                indent -= 1
                i += 1
                continue
            if first_token_of_the_line:
                program += ' ' * (indent * 4)
                if is_block_start(reverse_mapping[v]):
                    indent += 1
                first_token_of_the_line = False
            
            # 检查下一个token是否是DOT
            next_is_dot = i + 1 < len(encoded) and encoded[i+1] in reverse_mapping and reverse_mapping.get(encoded[i+1]) == 'DOT'
            # 检查当前token是否是DOT
            current_is_dot = reverse_mapping[v] == 'DOT'
            # 检查前一个token是否是DOT
            prev_is_dot = i > 0 and encoded[i-1] in reverse_mapping and reverse_mapping[encoded[i-1]] == 'DOT'
            # 检查当前token是否是INT
            current_is_int = isinstance(reverse_mapping[v], int)
            # 检查下一个token是否是INT
            next_is_int = i + 1 < len(encoded) and encoded[i+1] in reverse_mapping and isinstance(reverse_mapping.get(encoded[i+1]), int)
            if reverse_mapping[v] == 'ENTER':
                program += '\n'
                first_token_of_the_line = True
            else:
                if token2value.get(reverse_mapping[v]) is not None:
                    program += str(token_to_str(token2value[reverse_mapping[v]]))
                else:
                    program += str(token_to_str(reverse_mapping[v]))
                
                # 只在以下情况添加空格：
                # 1. 当前token不是DOT
                # 2. 下一个token不是DOT
                # 3. 当前token不是ID
                # 4. 下一个token不是ID
                # 5. 当前token和下一个token不都是INT
                if not current_is_dot and not next_is_dot and \
                   not (reverse_mapping[v] == 'ID' and next_is_dot) and \
                   not (prev_is_dot and reverse_mapping[v] == 'ID') and \
                   not (current_is_int and next_is_int):
                    program += ' '
            i += 1
        return program

def add_tl_torch_funcs(string: str):
    tl_lexer = TLFuncLexer()
    tl_lexer.tokenize(string)
    global tokens
    tokens = tuple(list(TritonLexer.tokens) + [to_token(func) for func in tl_lexer.tl_funcs] + [to_token(func) for func in tl_lexer.torch_funcs])
    tokens = tuple(list(set(tokens)))
    TritonLexer.tokens = tokens
    for tlf in tl_lexer.tl_funcs:
        TritonLexer.keywords[tlf] = to_token(tlf)
    for tlf in tl_lexer.torch_funcs:
        TritonLexer.keywords[tlf] = to_token(tlf)

# Test function
def test1():
    data = '-1000'
    encoder = TritonTokenEncoderDecoder()
    encoder.print_mapping()
    print(encoder.tokenize(data))
    print(encoder.encode())
    print(encoder.decode(encoder.encode()))

def test2():
    data = '''def causal_conv1d_fwd_kernel(
    x: torch.GUGUGU,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_b * T, i_b * T + T

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, W)
    m_d = o_d < D

    if HAS_WEIGHT:

        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None], other=0).to(
            tl.float32
        )

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    for i_w in tl.static_range(-W + 1, 1):
        p_yi = tl.make_block_ptr(
            x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
        )

        b_yi = tl.load(p_yi, boundary_check=(0, 1))
        if HAS_WEIGHT:
            b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
        b_y += b_yi
    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

    if ACTIVATION == "swish" or ACTIVATION == "silu":
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        p_residual = tl.make_block_ptr(
            residual + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0)
        )
        b_residual = tl.load(p_residual, boundary_check=(0, 1))
        b_y += b_residual

    p_y = tl.make_block_ptr(
        y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0)
    )
    tl.store(
        p_y,
        tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
'''

    add_tl_torch_funcs(data)
    print(TritonLexer.tokens)
    encoder = TritonTokenEncoderDecoder()
    print(encoder.tokenize(data))
    print(encoder.encode())
    print(encoder.decode(encoder.encode()))

def read_file(filename: str) -> str:
    """读取文件内容"""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def process_json_file(filename: str) -> List[Dict[str, Any]]:
    """处理JSON文件"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            results = []
            for datum in data:
                source = '\n'.join(datum['source'])
                results.append(source)
            return results
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing JSON file: {e}")
        sys.exit(1)
        
def verify(program: str, original_program: str, results: List[Dict[str, Any]], mapping: Dict[str, str], output: str) -> bool:
    with open('test.py', 'w') as f:
        # triton_imports = '\n'.join(extract_imports_from_code(original_program)['triton'])
        triton_imports = '''
import triton
import triton.language as tl
from triton import *
from triton.language import *
import numpy as np
import torch
        '''
        f.write(f'\'\'\'{original_program}\n\'\'\'\n{triton_imports}\n' + program)
    import subprocess
    result = subprocess.run(['python', 'test.py'], capture_output=True, text=True)
    if result.returncode != 0 or result.stderr != '':
        if original_program != '':
            print(original_program)
        print(result.stderr)
        print(f'return code: {result.returncode}')
        print(program)
        with open(output, 'w') as f:
            json.dump({'kernels': results, 'mapping': mapping}, f)
        import sys 
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Triton Token Encoder/Decoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode a code string
  python endecoder.py -s "def kernel(x): return x + 1"
  
  # Encode a code file
  python endecoder.py -f kernel.py
  
  # Process a JSON file containing kernels
  python endecoder.py -j kernels.json
        """
    )
    
    # 创建互斥组，确保只能使用一种输入方式
    group = parser.add_mutually_exclusive_group(required=True)
    
    # 添加命令行参数
    group.add_argument('-s', '--string',
                      help='Input code string to encode')
    group.add_argument('-f', '--file',
                      help='Input code file to encode')
    group.add_argument('-j', '--json',
                      help='Input JSON file containing kernels to encode')
    
    # 添加输出选项
    parser.add_argument('-o', '--output',
                        required=True,
                       help='Output file path (default: print to stdout)')
    
    args = parser.parse_args()
    
    
    if args.string:
        add_tl_torch_funcs(args.string)
        encoder = TritonTokenEncoderDecoder()
        encoder.tokenize(args.string)
        encoded = encoder.encode()
        result = {
            'code': args.string,
            'encoded': encoded
        }
    elif args.file:
        code = read_file(args.file)
        add_tl_torch_funcs(code)
        encoder = TritonTokenEncoderDecoder()
        encoder.tokenize(code)
        encoded = encoder.encode()
        result = {
            'code': code,
            'encoded': encoded
        }
    elif args.json:
        # 处理JSON文件
        codes = process_json_file(args.json)
        results = []
        start_id = -1
        if os.path.exists('endecoder.log') and os.path.getsize('endecoder.log') > 0:
            with open('endecoder.log', 'r') as f:
                start_id = int(f.read())
        start_id += 1
        if os.path.exists('encoded_kernels.json'):
                with open('encoded_kernels.json', 'r') as f:
                    original_results = json.load(f)
                    results = original_results['kernels']
        for i, code in enumerate(tqdm.tqdm(codes[start_id:])):
            add_tl_torch_funcs(code)
            encoder = TritonTokenEncoderDecoder()
            if os.path.exists('encoded_kernels.json'):
                with open('encoded_kernels.json', 'r') as f:
                    original_results = json.load(f)
                    encoder.mapping.update(original_results['mapping'])
            with open('endecoder.log', 'w') as f:
                f.write(f'{start_id + i}')
            encoder.tokenize(code)
            encoded = encoder.encode()
            verify(encoder.decode(encoded), code, results, encoder.mapping, args.output)
            results.append({
                'code': code,
                'encoded': encoded
            })
        result = {'kernels': results, 'mapping': encoder.mapping}
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f)
    

if __name__ == '__main__':
    main()
    # test1()