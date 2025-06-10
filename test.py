import triton
import triton.language as tl
import numpy as np
import torch
def f0 ( 
     v0 , 
     v1 , 
     v2 , 
     v3 , 
     v4 , 
     v5 : tl.constexpr , 
     v6 : tl.constexpr , 
     v7 : tl.constexpr , 
     v8 : tl.constexpr , 
 ) : 
 v9 , v10 = tl.program_id ( 0 ) , tl.program_id ( 1 ) 
 v11 , v12 = v10 // v5 , v10 % v5 
 if v8 : 
     v13 , v9 = tl.load ( v3 + v9 * 2 ) . v14 ( tl.int32 ) , tl.load ( 
     v3 + v9 * 2 + 1 
     ) . v14 ( tl.int32 ) 
     v15 , v16 = tl.load ( v2 + v13 ) . v14 ( tl.int32 ) , tl.load ( 
     v2 + v13 + 1 
     ) . v14 ( tl.int32 ) 
     v4 = v16 - v15 
 else : 
     v15 , v16 = v11 * v4 , v11 * v4 + v4 
 v17 = tl.make_block_ptr ( 
 v0 + ( v15 * v5 + v12 ) * v6 , 
 ( v4 , v6 ) , 
 ( v5 * v6 , 1 ) , 
 ( v9 * v6 , 0 ) , 
 ( v6 , v6 ) , 
 ( 1 , 0 ) , 
 ) 
 v18 = tl.make_block_ptr ( 
 v1 + ( v15 * v5 + v12 ) * v6 , 
 ( v4 , v6 ) , 
 ( v5 * v6 , 1 ) , 
 ( v9 * v6 , 0 ) , 
 ( v6 , v6 ) , 
 ( 1 , 0 ) , 
 ) 
 v19 = tl.load ( v17 , v20 = ( 0 , 1 ) ) 
 v19 = tl.where ( tl.arange ( 0 , v6 ) [ : , None ] > tl.arange ( 0 , v6 ) [ None , : ] , v19 , 0 ) 
 for v21 in range ( 1 , v6 ) : 
     v22 = tl.arange ( 0 , v6 ) == v21 
     v23 = tl.sum ( tl.where ( v22 [ : , None ] , v19 , 0 ) , 0 ) 
     v23 = v23 + tl.sum ( v23 [ : , None ] * v19 , 0 ) * ( tl.arange ( 0 , v6 ) < v21 ) 
     v19 = tl.where ( v22 [ : , None ] , v23 , v19 ) 
 v19 += tl.arange ( 0 , v6 ) [ : , None ] == tl.arange ( 0 , v6 ) [ None , : ] 
 tl.store ( v18 , v24 ( v25 ) , v20 = ( 0 , 1 ) ) 
 