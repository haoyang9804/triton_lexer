import ply.lex as lex
from typing import List, Dict, Any
from ply.lex import LexToken
import argparse
import json
import sys

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
    
    # Torch specific
    'TORCH_TENSOR', 'TORCH', 'TORCH_DEVICE', 'TORCH_DTYPE', 'TORCH_LAYOUT',
    'TORCH_MEMORY_FORMAT', 'TORCH_GRAD', 'TORCH_REQUIRES_GRAD', 'TORCH_BACKWARD',
    'TORCH_DETACH', 'TORCH_CLONE', 'TORCH_CONTIGUOUS', 'TORCH_CPU', 'TORCH_CUDA',
    'TORCH_GPU', 'TORCH_CPU_DEVICE', 'TORCH_CUDA_DEVICE', 'TORCH_FLOAT32',
    'TORCH_FLOAT64', 'TORCH_FLOAT16', 'TORCH_BFLOAT16', 'TORCH_INT8', 'TORCH_INT16',
    'TORCH_INT32', 'TORCH_INT64', 'TORCH_UINT8', 'TORCH_BOOL', 'TORCH_COMPLEX64',
    'TORCH_COMPLEX128', 'TORCH_QUINT8', 'TORCH_QINT8', 'TORCH_QINT32',
    
    # Triton specific
    'DL', 'LIBDEVICE', 'LIBSHMEM_DEVICE',
    'TRITON_HELPERS', 'LANGUAGE', 'TRITON', 'CEIL_DIV',
    'EXP', 'LOG', 'ATOMIC_ADD', 'ATOMIC_CAS', 'ATOMIC_MIN',
    'ATOMIC_XCHG', 'LD', 'ST', 'TID', 'SYNC_THREADS',
    'SHFL_DOWN_SYNC_I32', 'SHFL_UP_SYNC_I32', 'SHFL_SYNC_I32',
    'BALLOT_SYNC', 'FFS',
    
    # Triton tl module
    'TL_ARANGE', 'TL_ATOMIC_ADD', 'TL_ATOMIC_CAS', 'TL_ATOMIC_MIN',
    'TL_ATOMIC_XCHG', 'TL_ASSUME', 'TL_BFLOAT16', 'TL_BROADCAST_TO',
    'TL_CAST', 'TL_CDIV', 'TL_CONSTEXPR', 'TL_CUMSUM', 'TL_DEBUG_BARRIER',
    'TL_DEVICE_ASSERT', 'TL_DOT', 'TL_DOT_SCALED', 'TL_DTYPE', 'TL_EXP',
    'TL_EXPERIMENTAL_DESCRIPTOR_LOAD', 'TL_EXPERIMENTAL_DESCRIPTOR_STORE',
    'TL_EXTRA_CUDA_LIBDEVICE_ROUND', 'TL_FDIV', 'TL_FLOAT16', 'TL_FLOAT32',
    'TL_FLOAT8E4NV', 'TL_FLOAT8E5', 'TL_FLIP', 'TL_FLOOR', 'TL_FMA',
    'TL_FULL', 'TL_INT16', 'TL_INT2', 'TL_INT32', 'TL_INT64', 'TL_INT8',
    'TL_INTERLEAVE', 'TL_JOIN', 'TL_LOAD', 'TL_LOG', 'TL_LOG2',
    'TL_MAKE_BLOCK_PTR', 'TL_MAKE_TENSOR_DESCRIPTOR', 'TL_MATH_EXP2',
    'TL_MATH_FAST_EXPF', 'TL_MATH_LOG2', 'TL_MATH_MAX', 'TL_MATH_RSQRT',
    'TL_MAX', 'TL_MAX_CONTIGUOUS', 'TL_MAXIMUM', 'TL_MIN', 'TL_MINIMUM',
    'TL_MULTIPLE_OF', 'TL_NUM_PROGRAMS', 'TL_PABS', 'TL_PERMUTE',
    'TL_PI32_T', 'TL_POINTER_TYPE', 'TL_PROGRAM_ID', 'TL_RAND', 'TL_RANGE',
    'TL_RESHAPE', 'TL_RSQRT', 'TL_SIGMOID', 'TL_SPLIT', 'TL_SQRT',
    'TL_STANDARD_LOG2', 'TL_STATIC_ASSERT', 'TL_STATIC_PRINT',
    'TL_STATIC_RANGE', 'TL_STORE', 'TL_SUM', 'TL_SWIZZLE2D', 'TL_TENSOR',
    'TL_TO', 'TL_TRANS', 'TL_UINT32', 'TL_VIEW', 'TL_WHERE', 'TL_ZEROS',
    
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
        'sync_threads': 'SYNC_THREADS',
        'shfl_down_sync_i32': 'SHFL_DOWN_SYNC_I32',
        'shfl_up_sync_i32': 'SHFL_UP_SYNC_I32',
        'shfl_sync_i32': 'SHFL_SYNC_I32',
        'ballot_sync': 'BALLOT_SYNC',
        'ffs': 'FFS',
        
        # Triton tl module
        'tl.arange': 'TL_ARANGE',
        'tl.atomic_add': 'TL_ATOMIC_ADD',
        'tl.atomic_cas': 'TL_ATOMIC_CAS',
        'tl.atomic_min': 'TL_ATOMIC_MIN',
        'tl.atomic_xchg': 'TL_ATOMIC_XCHG',
        'tl.assume': 'TL_ASSUME',
        'tl.bfloat16': 'TL_BFLOAT16',
        'tl.broadcast_to': 'TL_BROADCAST_TO',
        'tl.cast': 'TL_CAST',
        'tl.cdiv': 'TL_CDIV',
        'tl.constexpr': 'TL_CONSTEXPR',
        'tl.cumsum': 'TL_CUMSUM',
        'tl.debug_barrier': 'TL_DEBUG_BARRIER',
        'tl.device_assert': 'TL_DEVICE_ASSERT',
        'tl.dot': 'TL_DOT',
        'tl.dot_scaled': 'TL_DOT_SCALED',
        'tl.dtype': 'TL_DTYPE',
        'tl.exp': 'TL_EXP',
        'tl.experimental_descriptor_load': 'TL_EXPERIMENTAL_DESCRIPTOR_LOAD',
        'tl.experimental_descriptor_store': 'TL_EXPERIMENTAL_DESCRIPTOR_STORE',
        'tl.extra_cuda_libdevice_round': 'TL_EXTRA_CUDA_LIBDEVICE_ROUND',
        'tl.fdiv': 'TL_FDIV',
        'tl.float16': 'TL_FLOAT16',
        'tl.float32': 'TL_FLOAT32',
        'tl.float8e4nv': 'TL_FLOAT8E4NV',
        'tl.float8e5': 'TL_FLOAT8E5',
        'tl.flip': 'TL_FLIP',
        'tl.floor': 'TL_FLOOR',
        'tl.fma': 'TL_FMA',
        'tl.full': 'TL_FULL',
        'tl.int16': 'TL_INT16',
        'tl.int2': 'TL_INT2',
        'tl.int32': 'TL_INT32',
        'tl.int64': 'TL_INT64',
        'tl.int8': 'TL_INT8',
        'tl.interleave': 'TL_INTERLEAVE',
        'tl.join': 'TL_JOIN',
        'tl.load': 'TL_LOAD',
        'tl.log': 'TL_LOG',
        'tl.log2': 'TL_LOG2',
        'tl.make_block_ptr': 'TL_MAKE_BLOCK_PTR',
        'tl.make_tensor_descriptor': 'TL_MAKE_TENSOR_DESCRIPTOR',
        'tl.math.exp2': 'TL_MATH_EXP2',
        'tl.math.fast_expf': 'TL_MATH_FAST_EXPF',
        'tl.math.log2': 'TL_MATH_LOG2',
        'tl.math.max': 'TL_MATH_MAX',
        'tl.math.rsqrt': 'TL_MATH_RSQRT',
        'tl.max': 'TL_MAX',
        'tl.max_contiguous': 'TL_MAX_CONTIGUOUS',
        'tl.maximum': 'TL_MAXIMUM',
        'tl.min': 'TL_MIN',
        'tl.minimum': 'TL_MINIMUM',
        'tl.multiple_of': 'TL_MULTIPLE_OF',
        'tl.num_programs': 'TL_NUM_PROGRAMS',
        'tl.pabs': 'TL_PABS',
        'tl.permute': 'TL_PERMUTE',
        'tl.pi32_t': 'TL_PI32_T',
        'tl.pointer_type': 'TL_POINTER_TYPE',
        'tl.program_id': 'TL_PROGRAM_ID',
        'tl.rand': 'TL_RAND',
        'tl.range': 'TL_RANGE',
        'tl.reshape': 'TL_RESHAPE',
        'tl.rsqrt': 'TL_RSQRT',
        'tl.sigmoid': 'TL_SIGMOID',
        'tl.split': 'TL_SPLIT',
        'tl.sqrt': 'TL_SQRT',
        'tl.standard_log2': 'TL_STANDARD_LOG2',
        'tl.static_assert': 'TL_STATIC_ASSERT',
        'tl.static_print': 'TL_STATIC_PRINT',
        'tl.static_range': 'TL_STATIC_RANGE',
        'tl.store': 'TL_STORE',
        'tl.sum': 'TL_SUM',
        'tl.swizzle2d': 'TL_SWIZZLE2D',
        'tl.tensor': 'TL_TENSOR',
        'tl.to': 'TL_TO',
        'tl.trans': 'TL_TRANS',
        'tl.uint32': 'TL_UINT32',
        'tl.view': 'TL_VIEW',
        'tl.where': 'TL_WHERE',
        'tl.zeros': 'TL_ZEROS',
        
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
            t.value = content
        else:
            t.value = content  # Remove quotes
        return t
    
    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*(?:\.[a-zA-Z_][a-zA-Z_0-9]*)*'
        t.type = self.keywords.get(t.value, 'ID')
        return t
    
    def t_FLOAT(self, t):
        r'\d*\.\d+'
        t.value = float(t.value)
        return t
    
    def t_INT(self, t):
        r'\d+'
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
        self.val2tokenid = {}
    
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
        class Block:
            def __init__(self, name: str, lexpos: int):
                self.name = name
                self.lexpos = lexpos
                self.next_block = None
                self.prev_block = None
            def set_new_block(self, name: str, lexpos: int):
                new_block = Block(name, lexpos)
                new_block.prev_block = self
                self.next_block = new_block
                return new_block
            def __str__(self):
                return f'{self.name} {self.lexpos}'

        strs = string.split('\n')
        block = Block('header', 0)
        for linenum, s in enumerate(strs):
            lexpos = -1
            last_token = None
            self.lexer.input(s)
            lexpos_decrease = 0
            first_token = True
            while True:
                tok = self.lexer.token()
                if not tok:
                    break
                tok.lexpos -= lexpos_decrease
                tok.lineno = linenum + 1
                if first_token:
                    first_token = False
                    if block.name != 'header' and tok.lexpos == block.lexpos:
                        self.tokens.append(self._new_token(f'END{block.name}', '', -1, -1))
                        block = block.prev_block
                    if is_block_start(tok.type):
                        block = block.set_new_block(tok.type, tok.lexpos)
                if tok.type == 'ID':
                    if self.oldName2NewName.get(tok.value) is not None:
                        lexpos_decrease += len(tok.value) - len(self.oldName2NewName[tok.value])
                        tok.value = self.oldName2NewName[tok.value]
                    else:
                        if last_token and last_token.value == 'def':
                            fname = f'f{self.fid}'
                            self.oldName2NewName[tok.value] = fname
                            lexpos_decrease += len(tok.value) - len(fname)
                            tok.value = fname
                            self.fid += 1
                        else:
                            vname = f'v{self.varid}'
                            lexpos_decrease += len(tok.value) - len(vname)
                            self.oldName2NewName[tok.value] = vname
                            tok.value = vname
                            self.varid += 1
                self.tokens.append(tok)
                lexpos = tok.lexpos
                last_token = tok
            if lexpos == -1: lexpos = 0
            end_token = self._new_token('ENTER', '<enter>', linenum + 1, lexpos + len(str(last_token.value)) if last_token is not None else 0)
            self.tokens.append(end_token)
        while block.name != 'header':
            self.tokens.append(self._new_token(f'END{block.name}', '', -1, -1))
            block = block.prev_block
        for i, token in enumerate(self.tokens):
            self.val2tokenid[token.value] = i
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
        for v in encoded:
            if v == -1:
                continue
            if is_block_end(reverse_mapping[v]):
                indent -= 1
                continue
            if first_token_of_the_line:
                program += ' ' * indent * 4
                first_token_of_the_line = False
            if reverse_mapping[v] == 'ENTER':
                program += '\n'
                first_token_of_the_line = True
            else:
                if token2value.get(reverse_mapping[v]) is not None:
                    program += str(token_to_str(token2value[reverse_mapping[v]]))
                else:
                    program += str(token_to_str(reverse_mapping[v]))
            program += ' '
            if is_block_start(reverse_mapping[v]):
                indent += 1
        return program
# class TritonTokenDecoder:
#     def __init__(self, encoder: TritonTokenEncoder):
#         self.mapping = encoder.mapping
#         self.tokens = encoder.tokens
    
#     def decode(self) -> str:
#         program = ''
#         prev_lineno = -1
#         prev_lexpos = -1
#         prev_token = None
#         for token in self.tokens:
#             if token.type == 'ENTER':
#                 program += '\n'
#             else:
#                 if token.lineno == prev_lineno:
#                     assert prev_token is not None, 'prev_token is None'
#                     if prev_token.type == 'STRING' or '_STR' in prev_token.type:
#                         self_length = len(str(prev_token.value)) + 2
#                     else:
#                         self_length = len(str(prev_token.value))
#                     program += ' ' * (token.lexpos - (prev_lexpos + self_length))
#                 else:
#                     program += ' ' * token.lexpos
#                 if token.type == 'STRING' or '_STR' in token.type: program += '"'
#                 program += str(token.value)
#                 if token.type == 'STRING' or '_STR' in token.type: program += '"'
#                 prev_token = token
#                 prev_lineno = token.lineno
#                 prev_lexpos = token.lexpos
        # return program

# Test function
def test1():
    data = '''
    def f():
        if x > 0.3:
            return x + y
        else:
            return 0
    '''
    # data = 'def batch_norm_forward_kernel(input_pointer, weight_pointer, bias_pointer, mean_pointer, inv_std_pointer, pre_act_add_pointer, pre_act_pointer, output_pointer, running_mean_pointer, running_var_pointer, batch_dim, spatial_dim, input_batch_stride, input_feat_stride, input_spatial_stride, pre_act_add_batch_stride, pre_act_add_feat_stride, pre_act_add_spatial_stride, pre_act_batch_stride, pre_act_feat_stride, pre_act_spatial_stride, output_batch_stride, output_feat_stride, output_spatial_stride, momentum, eps, param, affine: tl.constexpr, save_stats: tl.constexpr, track_running_stats: tl.constexpr, is_train: tl.constexpr, add_pre_act: tl.constexpr, act_func: tl.constexpr, save_pre_act: tl.constexpr, BLOCK_SIZE_BATCH: tl.constexpr, BLOCK_SIZE_SPATIAL: tl.constexpr):'
#     data = """
# def kernel(x, y):
#     tl.arange("x", "cuda")
#     """
    encoder = TritonTokenEncoderDecoder()
    encoder.print_mapping()
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
                break
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

def verify(program: str) -> bool:
    with open('test.py', 'w') as f:
        f.write('import triton\nimport triton.language as tl\nimport numpy as np\nimport torch\n' + program)
    import subprocess
    result = subprocess.run(['python', 'test.py'], capture_output=True, text=True)
    if result.returncode != 0 or result.stderr != '':
        print(result.stderr)
        print(f'return code: {result.returncode}')
        print(program)
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
                       help='Output file path (default: print to stdout)')
    parser.add_argument('--pretty',
                       action='store_true',
                       help='Pretty print the output')
    
    args = parser.parse_args()
    
    # 创建编码器
    encoder = TritonTokenEncoderDecoder()
    
    # 处理输入
    if args.string:
        # 处理代码字符串
        encoder.tokenize(args.string)
        encoded = encoder.encode()
        result = {
            'code': args.string,
            'encoded': encoded
        }
    elif args.file:
        # 处理代码文件
        code = read_file(args.file)
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
        for code in codes:
            encoder.tokenize(code)
            encoded = encoder.encode()
            print(code)
            verify(encoder.decode(encoded))
            print('====================')
            results.append({
                'code': code,
                'encoded': encoded
            })
        result = {'kernels': results}
    
    # 处理输出
    if args.output:
        # 写入文件
        with open(args.output, 'w') as f:
            if args.pretty:
                json.dump(result, f, indent=2)
            else:
                json.dump(result, f)
    else:
        # 打印到标准输出
        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))
    

if __name__ == '__main__':
    main()
    