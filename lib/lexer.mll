{
  open Parser
  exception Error of string
}

let digit = ['0'-'9']
let letter = ['a'-'z' 'A'-'Z']
let whitespace = [' ' '\t' '\r' '\n']
let identifier = letter (letter | digit | '_')*

rule token = parse
  | whitespace+ { token lexbuf }  (* 跳过空白字符 *)
  (* Python 关键字 *)
  | "def"       { DEF }
  | "if"        { IF }
  | "else"      { ELSE }
  | "elif"      { ELIF }
  | "for"       { FOR }
  | "in"        { IN }
  | "range"     { RANGE }
  | "return"    { RETURN }
  | "not"       { NOT }
  | "and"       { AND }
  | "or"        { OR }
  | "is"        { IS }
  | "None"      { NONE }
  | "True"      { TRUE }
  | "False"     { FALSE }
  | "min"       { MIN }
  | "print"     { PRINT }
  | "sum"       { SUM }
  | "any"       { ANY }
  | "all"       { ALL }
  | "abs"       { ABS }
  | "max"       { MAX }
  | "round"     { ROUND }
  | "divmod"    { DIVMOD }
  | "len"       { LEN }
  | "zip"       { ZIP }
  | "slice"     { SLICE }
  | "str"       { STR }
  | "type"      { TYPE }
  | "isinstance" { ISINSTANCE }
  | "tuple"     { TUPLE }
  | "list"      { LIST }
  | "dict"      { DICT }
  | "set"       { SET }
  | "torch"     { TORCH }
  | "dl"        { DL }
  | "libdevice" { LIBDEVICE }
  | "libshmem_device" { LIBSHMEM_DEVICE }
  | "triton_helpers" { TRITON_HELPERS }
  | "language"  { LANGUAGE }
  | "triton"    { TRITON }
  | "ceil_div"  { CEIL_DIV }
  | "exp"       { EXP }
  | "log"       { LOG }
  | "atomic_add" { ATOMIC_ADD }
  | "atomic_cas" { ATOMIC_CAS }
  | "atomic_min" { ATOMIC_MIN }
  | "atomic_xchg" { ATOMIC_XCHG }
  | "ld"        { LD }
  | "st"        { ST }
  | "tid"       { TID }
  | "__syncthreads" { SYNC_THREADS }
  | "__shfl_down_sync_i32" { SHFL_DOWN_SYNC_I32 }
  | "__shfl_up_sync_i32" { SHFL_UP_SYNC_I32 }
  | "__shfl_sync_i32" { SHFL_SYNC_I32 }
  | "__ballot_sync" { BALLOT_SYNC }
  | "ffs"       { FFS }
  (* Triton 关键字 *)
  | "tl.arange" { TL_ARANGE }
  | "tl.atomic_add" { TL_ATOMIC_ADD }
  | "tl.atomic_cas" { TL_ATOMIC_CAS }
  | "tl.atomic_min" { TL_ATOMIC_MIN }
  | "tl.atomic_xchg" { TL_ATOMIC_XCHG }
  | "tl.assume" { TL_ASSUME }
  | "tl.bfloat16" { TL_BFLOAT16 }
  | "tl.broadcast_to" { TL_BROADCAST_TO }
  | "tl.cast"   { TL_CAST }
  | "tl.cdiv"   { TL_CDIV }
  | "tl.constexpr" { TL_CONSTEXPR }
  | "tl.cumsum" { TL_CUMSUM }
  | "tl.debug_barrier" { TL_DEBUG_BARRIER }
  | "tl.device_assert" { TL_DEVICE_ASSERT }
  | "tl.dot"    { TL_DOT }
  | "tl.dot_scaled" { TL_DOT_SCALED }
  | "tl.dtype"  { TL_DTYPE }
  | "tl.exp"    { TL_EXP }
  | "tl._experimental_descriptor_load" { TL_EXPERIMENTAL_DESCRIPTOR_LOAD }
  | "tl._experimental_descriptor_store" { TL_EXPERIMENTAL_DESCRIPTOR_STORE }
  | "tl.extra.cuda.libdevice.round" { TL_EXTRA_CUDA_LIBDEVICE_ROUND }
  | "tl.fdiv"   { TL_FDIV }
  | "tl.float16" { TL_FLOAT16 }
  | "tl.float32" { TL_FLOAT32 }
  | "tl.float8e4nv" { TL_FLOAT8E4NV }
  | "tl.float8e5" { TL_FLOAT8E5 }
  | "tl.flip"   { TL_FLIP }
  | "tl.floor"  { TL_FLOOR }
  | "tl.fma"    { TL_FMA }
  | "tl.full"   { TL_FULL }
  | "tl.int16"  { TL_INT16 }
  | "tl.int2"   { TL_INT2 }
  | "tl.int32"  { TL_INT32 }
  | "tl.int64"  { TL_INT64 }
  | "tl.int8"   { TL_INT8 }
  | "tl.interleave" { TL_INTERLEAVE }
  | "tl.join"   { TL_JOIN }
  | "tl.load"   { TL_LOAD }
  | "tl.log"    { TL_LOG }
  | "tl.log2"   { TL_LOG2 }
  | "tl.make_block_ptr" { TL_MAKE_BLOCK_PTR }
  | "tl.make_tensor_descriptor" { TL_MAKE_TENSOR_DESCRIPTOR }
  | "tl.math.exp2" { TL_MATH_EXP2 }
  | "tl.math.fast_expf" { TL_MATH_FAST_EXPF }
  | "tl.math.log2" { TL_MATH_LOG2 }
  | "tl.math.max" { TL_MATH_MAX }
  | "tl.math.rsqrt" { TL_MATH_RSQRT }
  | "tl.max"    { TL_MAX }
  | "tl.max_contiguous" { TL_MAX_CONTIGUOUS }
  | "tl.maximum" { TL_MAXIMUM }
  | "tl.min"    { TL_MIN }
  | "tl.minimum" { TL_MINIMUM }
  | "tl.multiple_of" { TL_MULTIPLE_OF }
  | "tl.num_programs" { TL_NUM_PROGRAMS }
  | "tl.pabs"   { TL_PABS }
  | "tl.permute" { TL_PERMUTE }
  | "tl.pi32_t" { TL_PI32_T }
  | "tl.pointer_type" { TL_POINTER_TYPE }
  | "tl.program_id" { TL_PROGRAM_ID }
  | "tl.rand"   { TL_RAND }
  | "tl.range"  { TL_RANGE }
  | "tl.reshape" { TL_RESHAPE }
  | "tl.rsqrt"  { TL_RSQRT }
  | "tl.sigmoid" { TL_SIGMOID }
  | "tl.split"  { TL_SPLIT }
  | "tl.sqrt"   { TL_SQRT }
  | "tl.standard._log2" { TL_STANDARD_LOG2 }
  | "tl.static_assert" { TL_STATIC_ASSERT }
  | "tl.static_print" { TL_STATIC_PRINT }
  | "tl.static_range" { TL_STATIC_RANGE }
  | "tl.store"  { TL_STORE }
  | "tl.sum"    { TL_SUM }
  | "tl.swizzle2d" { TL_SWIZZLE2D }
  | "tl.tensor" { TL_TENSOR }
  | "tl.to"     { TL_TO }
  | "tl.trans"  { TL_TRANS }
  | "tl.uint32" { TL_UINT32 }
  | "tl.view"   { TL_VIEW }
  | "tl.where"  { TL_WHERE }
  | "tl.zeros"  { TL_ZEROS }
  (* 类型相关 *)
  (* | "int"       { INT_TYPE }
  | "float"     { FLOAT_TYPE }
  | "str"       { STR_TYPE }
  | "Boolean"      { BOOL_TYPE }
  | "Tuple"     { GEN_TUPLE_TYPE }
  | "List"      { LIST_TYPE }
  | "Dict"      { DICT_TYPE }
  | "Set"       { SET_TYPE }
  | "|"         { UNION }
  | "<"         { LT }
  | ">"         { GT } *)
  (* TODO: 补充常用数据类型 *)
  (* 常用数据类型 *)
  (* | "float16" { TL_FLOAT16 }
  | "float32" { TL_FLOAT32 }
  | "float64" { TL_FLOAT64 }
  | "int16" { TL_INT16 }
  | "int32" { TL_INT32 }
  | "int64" { TL_INT64 }
  | "int8" { TL_INT8 }
  | "uint32" { TL_UINT32 }
  | "bfloat16" { TL_BFLOAT16 }
  | "tensor" { TL_TENSOR }
  | "pointer_type" { TL_POINTER_TYPE }
  | "pi32_t" { TL_PI32_T }
  | "program_id" { TL_PROGRAM_ID } *)
  (* 运算符 *)
  | '+'         { PLUS }
  | '-'         { MINUS }
  | '*'         { TIMES }
  | '/'         { DIVIDE }
  | '>'         { GT }
  | "=="        { EQ }
  | '='         { ASSIGN }
  | '('         { LPAREN }
  | ')'         { RPAREN }
  | '{'         { LBRACE }
  | '}'         { RBRACE }
  | ':'         { COLON }
  | ','         { COMMA }
  (* 字符串 *)
  | '"' ([^'"']* as s) '"' { STRING s }
  | ''' ([^''']* as s) ''' { STRING s }
  (* 数字 *)
  | digit+ as n { INT (int_of_string n) }
  | digit+ '.' digit* as f { FLOAT (float_of_string f) }
  | '.' digit+ as f { FLOAT (float_of_string f) }
  (* 标识符 *)
  | identifier as id { ID id }
  | eof         { EOF }
  | _           { raise (Error ("Unexpected char: " ^ Lexing.lexeme lexbuf)) }

{
  let from_string s =
    let lexbuf = Lexing.from_string s in
    let rec tokens acc =
      match token lexbuf with
      | EOF -> List.rev (EOF :: acc)  (* 确保在末尾添加 EOF *)
      | t -> tokens (t :: acc)
    in
    tokens []
}