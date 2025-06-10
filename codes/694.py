
import io
import logging
import os
import random
import re
import sys
import time


from tritondse.types import Architecture
from tritondse.seed import SeedStatus
import tritondse.logging

logger = tritondse.logging.get("routines")

NULL_PTR = 0


def rtn_ctype_b_loc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('__ctype_b_loc hooked')

    ctype  = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00"  
    ctype += b"\x02\x00\x03\x20\x02\x20\x02\x20\x02\x20\x02\x20\x02\x00\x02\x00"
    ctype += b"\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00"
    ctype += b"\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00\x02\x00"
    ctype += b"\x01\x60\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0"
    ctype += b"\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0"
    ctype += b"\x08\xd8\x08\xd8\x08\xd8\x08\xd8\x08\xd8\x08\xd8\x08\xd8\x08\xd8"
    ctype += b"\x08\xd8\x08\xd8\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0"
    ctype += b"\x04\xc0\x08\xd5\x08\xd5\x08\xd5\x08\xd5\x08\xd5\x08\xd5\x08\xc5"
    ctype += b"\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5"
    ctype += b"\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5\x08\xc5"
    ctype += b"\x08\xc5\x08\xc5\x08\xc5\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x04\xc0"
    ctype += b"\x04\xc0\x08\xd6\x08\xd6\x08\xd6\x08\xd6\x08\xd6\x08\xd6\x08\xc6"
    ctype += b"\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6"
    ctype += b"\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6\x08\xc6"
    ctype += b"\x08\xc6\x08\xc6\x08\xc6\x04\xc0\x04\xc0\x04\xc0\x04\xc0\x02\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    ctype += b"\x00\x00\x00\x00\x00\x00\x00\x00"

    
    alloc_size = 2*pstate.ptr_size + len(ctype)
    base_ctype = pstate.heap_allocator.alloc(alloc_size)

    ctype_table_offset = base_ctype + (pstate.ptr_size * 2)
    otable_offset = ctype_table_offset + 256

    pstate.memory.write_ptr(base_ctype, otable_offset)
    pstate.memory.write_ptr(base_ctype+pstate.ptr_size, 0)

    
    pstate.memory.write(ctype_table_offset, ctype)

    return base_ctype


def rtn_ctype_toupper_loc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    
    
    logger.debug('__ctype_toupper_loc hooked')

    ctype  = b"\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"
    ctype += b"\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"
    ctype += b"\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"
    ctype += b"\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"
    ctype += b"\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"
    ctype += b"\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf"
    ctype += b"\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef"
    ctype += b"\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff\xff\xff\xff"
    ctype += b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"
    ctype += b"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
    ctype += b"\x20\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2d\x2e\x2f"
    ctype += b"\x30\x31\x32\x33\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x3e\x3f"
    ctype += b"\x40\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f"
    ctype += b"\x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x5b\x5c\x5d\x5e\x5f"
    ctype += b"\x60\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f"
    ctype += b"\x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x7b\x7c\x7d\x7e\x7f"
    ctype += b"\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f"
    ctype += b"\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f"
    ctype += b"\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf"
    ctype += b"\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf"
    ctype += b"\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf"
    ctype += b"\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf"
    ctype += b"\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef"
    ctype += b"\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff"

    
    alloc_size = 2*pstate.ptr_size + len(ctype)
    base_ctype = pstate.heap_allocator.alloc(alloc_size)

    ctype_table_offset = base_ctype + (pstate.ptr_size * 2)
    otable_offset = ctype_table_offset + 256

    pstate.memory.write_ptr(base_ctype, otable_offset)
    pstate.memory.write_ptr(base_ctype+pstate.ptr_size, 0)

    
    pstate.memory.write(ctype_table_offset, ctype)

    return base_ctype


def rtn_errno_location(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('__errno_location hooked')

    
    
    
    segs = pstate.memory.find_map(pstate.EXTERN_SEG)
    if segs:
        mmap = segs[0]
        ERRNO = mmap.start + mmap.size - 4  
    else:
        assert False
    pstate.memory.write_dword(ERRNO, 0)

    return ERRNO


def rtn_libc_start_main(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('__libc_start_main hooked')

    
    main = pstate.get_argument_value(0)

    
    
    if pstate.architecture == Architecture.AARCH64:
        pass
    elif pstate.architecture == Architecture.ARM32:
        pass
    elif pstate.architecture in [Architecture.X86_64, Architecture.X86]:
        
        pstate.push_stack_value(main)
    else:
        assert False

    
    if se.config.is_format_raw():
        
        argc = len(se.config.program_argv)
    else:   
        argc = len(se.seed.content.argv) if se.seed.content.argv else len(se.config.program_argv)

    if pstate.architecture == Architecture.X86:
        
        
        pstate.write_argument_value(0 + 1, argc)
    else:
        pstate.write_argument_value(0, argc)
    logger.debug(f"argc = {argc}")

    
    addrs = list()

    if se.config.is_format_composite() and se.seed.content.argv:    
        argvs = se.seed.content.argv
        src = 'seed'
    else:  
        argvs = [x.encode("latin-1") for x in se.config.program_argv]  
        src = 'config'

    
    size = sum(len(x) for x in argvs)+len(argvs)+len(argvs)*pstate.ptr_size
    if size == 0:  
        size = pstate.ptr_size

    
    base = pstate.heap_allocator.alloc(size)

    for i, arg in enumerate(argvs):
        addrs.append(base)
        pstate.memory.write(base, arg + b'\x00')

        if se.config.is_format_composite() and se.seed.content.argv:    
            
            se.inject_symbolic_argv_memory(base, i, arg)
            

        logger.debug(f"({src}) argv[{i}] = {repr(pstate.memory.read(base, len(arg)))}")
        base += len(arg) + 1

    
    b_argv = base
    for addr in addrs:
        pstate.memory.write_ptr(base, addr)
        base += pstate.ptr_size

    
    if pstate.architecture == Architecture.X86:
        
        
        pstate.write_argument_value(1 + 1, b_argv)
    else:
        pstate.write_argument_value(1, b_argv)

    
    pstate.rtn_redirect_addr = main

    return None


def rtn_stack_chk_fail(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('__stack_chk_fail hooked')
    logger.critical('*** stack smashing detected ***: terminated')
    se.seed.status = SeedStatus.CRASH
    pstate.stop = True



def rtn_xstat(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('__xstat hooked')

    
    arg0 = pstate.get_argument_value(0)  
    arg1 = pstate.get_argument_value(1)  
    arg2 = pstate.get_argument_value(2)  

    if os.path.isfile(pstate.memory.read_string(arg1)):
        stat = os.stat(pstate.memory.read_string(arg1))
        pstate.memory.write_qword(arg2 + 0x00, stat.st_dev)
        pstate.memory.write_qword(arg2 + 0x08, stat.st_ino)
        pstate.memory.write_qword(arg2 + 0x10, stat.st_nlink)
        pstate.memory.write_dword(arg2 + 0x18, stat.st_mode)
        pstate.memory.write_dword(arg2 + 0x1c, stat.st_uid)
        pstate.memory.write_dword(arg2 + 0x20, stat.st_gid)
        pstate.memory.write_dword(arg2 + 0x24, 0)
        pstate.memory.write_qword(arg2 + 0x28, stat.st_rdev)
        pstate.memory.write_qword(arg2 + 0x30, stat.st_size)
        pstate.memory.write_qword(arg2 + 0x38, stat.st_blksize)
        pstate.memory.write_qword(arg2 + 0x40, stat.st_blocks)
        pstate.memory.write_qword(arg2 + 0x48, 0)
        pstate.memory.write_qword(arg2 + 0x50, 0)
        pstate.memory.write_qword(arg2 + 0x58, 0)
        pstate.memory.write_qword(arg2 + 0x60, 0)
        pstate.memory.write_qword(arg2 + 0x68, 0)
        pstate.memory.write_qword(arg2 + 0x70, 0)
        pstate.memory.write_qword(arg2 + 0x78, 0)
        pstate.memory.write_qword(arg2 + 0x80, 0)
        pstate.memory.write_qword(arg2 + 0x88, 0)
        return 0

    return pstate.minus_one


def rtn_abort(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('abort hooked')
    se.seed.status = SeedStatus.OK_DONE
    pstate.stop = True



def rtn_atoi(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('atoi hooked')

    ast = pstate.actx
    arg = pstate.get_argument_value(0)

    

    cells = {i: pstate.read_symbolic_memory_byte(arg+i).getAst() for i in range(10)}

    
    
    
    
    

    def multiply(ast, cells, index):
        n = ast.bv(0, 32)
        for i in range(index):
            n = n * 10 + (ast.zx(24, cells[i]) - 0x30)
        return n

    res = ast.ite(
              ast.lnot(ast.land([cells[0] >= 0x30, cells[0] <= 0x39])),
              multiply(ast, cells, 0),
              ast.ite(
                  ast.lnot(ast.land([cells[1] >= 0x30, cells[1] <= 0x39])),
                  multiply(ast, cells, 1),
                  ast.ite(
                      ast.lnot(ast.land([cells[2] >= 0x30, cells[2] <= 0x39])),
                      multiply(ast, cells, 2),
                      ast.ite(
                          ast.lnot(ast.land([cells[3] >= 0x30, cells[3] <= 0x39])),
                          multiply(ast, cells, 3),
                          ast.ite(
                              ast.lnot(ast.land([cells[4] >= 0x30, cells[4] <= 0x39])),
                              multiply(ast, cells, 4),
                              ast.ite(
                                  ast.lnot(ast.land([cells[5] >= 0x30, cells[5] <= 0x39])),
                                  multiply(ast, cells, 5),
                                  ast.ite(
                                      ast.lnot(ast.land([cells[6] >= 0x30, cells[6] <= 0x39])),
                                      multiply(ast, cells, 6),
                                      ast.ite(
                                          ast.lnot(ast.land([cells[7] >= 0x30, cells[7] <= 0x39])),
                                          multiply(ast, cells, 7),
                                          ast.ite(
                                              ast.lnot(ast.land([cells[8] >= 0x30, cells[8] <= 0x39])),
                                              multiply(ast, cells, 8),
                                              ast.ite(
                                                  ast.lnot(ast.land([cells[9] >= 0x30, cells[9] <= 0x39])),
                                                  multiply(ast, cells, 9),
                                                  multiply(ast, cells, 9)
                                              )
                                          )
                                      )
                                  )
                              )
                          )
                      )
                  )
              )
          )
    res = ast.sx(32, res)

    return res



def rtn_calloc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('calloc hooked')

    
    nmemb = pstate.get_argument_value(0)
    size  = pstate.get_argument_value(1)

    
    pstate.concretize_argument(0)  
    pstate.concretize_argument(1)  

    if nmemb == 0 or size == 0:
        ptr = NULL_PTR
    else:
        ptr = pstate.heap_allocator.alloc(nmemb * size)
        
        for index in range(nmemb * size):
            pstate.write_symbolic_memory_byte(ptr+index, pstate.actx.bv(0, 8))

    
    return ptr



def rtn_clock_gettime(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('clock_gettime hooked')

    
    clockid = pstate.get_argument_value(0)
    tp      = pstate.get_argument_value(1)

    
    pstate.concretize_argument(1)

    
    if tp == 0:
        return pstate.minus_one

    if pstate.time_inc_coefficient:
        t = pstate.time
    else:
        t = time.time()

    pstate.memory.write_ptr(tp, int(t))
    pstate.memory.write_ptr(tp+pstate.ptr_size, int(t * 1000000))

    
    return 0


def rtn_exit(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('exit hooked')
    arg = pstate.get_argument_value(0)
    pstate.stop = True
    return arg


def rtn_fclose(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('fclose hooked')

    
    arg0 = pstate.get_argument_value(0)     

    
    pstate.concretize_argument(0)

    if pstate.file_descriptor_exists(arg0):
        pstate.close_file_descriptor(arg0)
    else:
        return pstate.minus_one

    
    return 0


def rtn_fseek(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    

    
    
    
    logger.debug('fseek hooked')

    
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)
    arg2 = pstate.get_argument_value(2)

    if arg2 not in [0, 1, 2]:
        return pstate.minus_one
        

    if pstate.file_descriptor_exists(arg0):
        desc = pstate.get_file_descriptor(arg0)

        if desc.fd.seekable():
            r = desc.fd.seek(arg1, arg2)
            return r
        else:
            return -1
            
    else:
        return -1


def rtn_ftell(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    

    logger.debug('ftell hooked')

    
    arg0 = pstate.get_argument_value(0)

    if pstate.file_descriptor_exists(arg0):
        desc = pstate.get_file_descriptor(arg0)

        if desc.fd.seekable():
            return desc.fd.tell()
        else:
            return -1
            



def rtn_fgets(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('fgets hooked')

    
    buff, buff_ast = pstate.get_full_argument(0)
    size, size_ast = pstate.get_full_argument(1)
    fd = pstate.get_argument_value(2)
    
    pstate.concretize_argument(2)

    if pstate.file_descriptor_exists(fd):
        filedesc = pstate.get_file_descriptor(fd)
        offset = filedesc.offset
        data = filedesc.fgets(size)
        data_with_trail = data if data.endswith(b"\x00") else data+b"\x00" 

        if filedesc.is_input_fd():  
            
            if se.seed.is_raw() and se.seed.is_bootstrap_seed() and not data:
                data = b'\x00' * size

            if len(data) == size:  
                pstate.push_constraint(size_ast.getAst() == size)

            
            se.inject_symbolic_file_memory(buff, filedesc.name, data, offset)
            if data != data_with_trail: 
                 pstate.memory.write(buff+len(data), b"\x00")
            logger.debug(f"fgets() in {filedesc.name} = {repr(data_with_trail)}")
        else:
            pstate.concretize_argument(1)
            pstate.memory.write(buff, data_with_trail)

        return buff_ast if data else NULL_PTR
    else:
        logger.warning(f'File descriptor ({fd}) not found')
        return NULL_PTR



def rtn_fopen(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('fopen hooked')

    
    arg0 = pstate.get_argument_value(0)  
    arg1 = pstate.get_argument_value(1)  
    arg0s = pstate.memory.read_string(arg0)
    arg1s = pstate.memory.read_string(arg1)

    
    pstate.concretize_memory_bytes(arg0, len(arg0s)+1)  

    
    pstate.concretize_argument(1)

    if se.seed.is_file_defined(arg0s):
        logger.info(f"opening an input file: {arg0s}")
        
        data = se.seed.get_file_input(arg0s)
        filedesc = pstate.create_file_descriptor(arg0s, io.BytesIO(data))
        return filedesc.id
    else:
        
        try:
            fd = open(arg0s, arg1s)
            filedesc = pstate.create_file_descriptor(arg0s, fd)
            return filedesc.id
        except Exception as e:
            logger.debug(f"Failed to open {arg0s} {e}")
            return NULL_PTR


def rtn_fprintf(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('fprintf hooked')

    
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)

    
    

    arg1f = pstate.get_format_string(arg1)
    nbArgs = arg1f.count("{")
    args = pstate.get_format_arguments(arg1, [pstate.get_argument_value(x) for x in range(2, nbArgs+2)])
    try:
        s = arg1f.format(*args)
    except:
        
        logger.warning('Something wrong, probably UTF-8 string')
        s = ""

    if pstate.file_descriptor_exists(arg0):
        fdesc = pstate.get_file_descriptor(arg0)
        if arg0 not in [1, 2] or (arg0 == 1 and se.config.pipe_stdout) or (arg0 == 2 and se.config.pipe_stderr):
            fdesc.fd.write(s)
            fdesc.fd.flush()
    else:
        return 0

    
    return len(s)


def rtn___fprintf_chk(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('__fprintf_chk hooked')

    
    arg0 = pstate.get_argument_value(0)
    flag = pstate.get_argument_value(1)
    arg1 = pstate.get_argument_value(2)

    
    

    arg1f = pstate.get_format_string(arg1)
    nbArgs = arg1f.count("{")
    args = pstate.get_format_arguments(arg1, [pstate.get_argument_value(x) for x in range(3, nbArgs+2)])
    try:
        s = arg1f.format(*args)
    except:
        
        logger.warning('Something wrong, probably UTF-8 string')
        s = ""

    if pstate.file_descriptor_exists(arg0):
        fdesc = pstate.get_file_descriptor(arg0)
        if arg0 not in [1, 2] or (arg0 == 1 and se.config.pipe_stdout) or (arg0 == 2 and se.config.pipe_stderr):
            fdesc.fd.write(s)
            fdesc.fd.flush()
    else:
        return 0

    
    return len(s)



def rtn_fputc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('fputc hooked')

    
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)

    pstate.concretize_argument(0)
    pstate.concretize_argument(1)

    if pstate.file_descriptor_exists(arg1):
        fdesc = pstate.get_file_descriptor(arg1)
        if arg1 == 0:
            return 0
        elif (arg1 == 1 and se.config.pipe_stdout) or (arg1 == 2 and se.config.pipe_stderr):
            fdesc.fd.write(chr(arg0))
            fdesc.fd.flush()
        elif arg1 not in [0, 2]:
            fdesc.fd.write(chr(arg0))
        return 1
    else:
        return 0


def rtn_fputs(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('fputs hooked')

    
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)

    pstate.concretize_argument(0)
    pstate.concretize_argument(1)

    s = pstate.memory.read_string(arg0)

    

    if pstate.file_descriptor_exists(arg1):
        fdesc = pstate.get_file_descriptor(arg1)
        if arg1 == 0:
            return 0
        elif arg1 == 1:
            if se.config.pipe_stdout:
                fdesc.fd.write(s)
                fdesc.fd.flush()
        elif arg1 == 2:
            if se.config.pipe_stderr:
                fdesc.fd.write(s)
                fdesc.fd.flush()
        else:
            fdesc.fd.write(s)
    else:
        return 0

    
    return len(s)


def rtn_fread(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('fread hooked')

    
    ptr = pstate.get_argument_value(0)              
    size_t, size_ast = pstate.get_full_argument(1)  
    nmemb = pstate.get_argument_value(2)            
    fd = pstate.get_argument_value(3)               
    size = size_t * nmemb

    

    if pstate.file_descriptor_exists(fd):
        filedesc = pstate.get_file_descriptor(fd)
        offset = filedesc.offset
        data = filedesc.read(size)

        if filedesc.is_input_fd():  
            
            if se.seed.is_raw() and se.seed.is_bootstrap_seed() and not data:
                data = b'\x00' * size

            if len(data) == size:  
                pstate.push_constraint(size_ast.getAst() == size)

            se.inject_symbolic_file_memory(ptr, filedesc.name, data, offset)
            logger.debug(f"read in {filedesc.name} = {repr(data)}")
        else:
            pstate.concretize_argument(2)
            pstate.memory.write(ptr, data)

        return int(len(data)/size_t) if size_t else 0  
    else:
        logger.warning(f'File descriptor ({fd}) not found')
        return 0


def rtn_free(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('free hooked')

    
    ptr = pstate.get_argument_value(0)
    if ptr == 0:    
        return None
    pstate.heap_allocator.free(ptr)

    return None


def rtn_fwrite(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('fwrite hooked')

    
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)
    arg2 = pstate.get_argument_value(2)
    arg3 = pstate.get_argument_value(3)
    size = arg1 * arg2
    data = pstate.memory.read(arg0, size)

    if pstate.file_descriptor_exists(arg3):
        fdesc = pstate.get_file_descriptor(arg3)
        if arg3 == 0:
            return 0
        elif arg3 == 1:
            if se.config.pipe_stdout:
                fdesc.fd.buffer.write(data)
                fdesc.fd.flush()
        elif arg3 == 2:
            if se.config.pipe_stderr:
                fdesc.fd.buffer.write(data)
                fdesc.fd.flush()
        else:
            fdesc.fd.write(data)
    else:
        return 0

    
    return size


def rtn_write(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('write hooked')

    
    fd = pstate.get_argument_value(0)
    buf = pstate.get_argument_value(1)
    size = pstate.get_argument_value(2)
    data = pstate.memory.read(buf, size)

    if pstate.file_descriptor_exists(fd):
        fdesc = pstate.get_file_descriptor(fd)
        if fd == 0:
            return 0
        elif fd == 1:
            if se.config.pipe_stdout:
                fdesc.fd.buffer.write(data)
                fdesc.fd.flush()
        elif fd == 2:
            if se.config.pipe_stderr:
                fdesc.fd.buffer.write(data)
                fdesc.fd.flush()
        else:
            fdesc.fd.write(data)
    else:
        return 0

    
    return size


def rtn_gettimeofday(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('gettimeofday hooked')

    
    tv = pstate.get_argument_value(0)
    tz = pstate.get_argument_value(1)

    if tv == 0:
        return pstate.minus_one

    if pstate.time_inc_coefficient:
        t = pstate.time
    else:
        t = time.time()

    s = pstate.ptr_size
    pstate.memory.write_ptr(tv, int(t))
    pstate.memory.write_ptr(tv+pstate.ptr_size, int(t * 1000000))

    
    return 0


def rtn_malloc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('malloc hooked')

    
    size = pstate.get_argument_value(0)
    ptr = pstate.heap_allocator.alloc(size)

    
    return ptr


def rtn_open(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('open hooked')

    
    arg0 = pstate.get_argument_value(0)  
    flags = pstate.get_argument_value(1)  
    mode = pstate.get_argument_value(2)  
    arg0s = pstate.memory.read_string(arg0)

    
    pstate.concretize_memory_bytes(arg0, len(arg0s)+1)  

    
    pstate.concretize_argument(1)

    
    mode = ""
    if (flags & 0xFF) == 0x00:      
        mode = "r"
    elif (flags & 0xFF) == 0x01:    
        mode = "w"
    elif (flags & 0xFF) == 0x02:    
        mode = "r+"

    if flags & 0x0100:  
        mode += "x"
    if flags & 0x0200:  
        mode = "a"  

    
    mode += "b"

    if se.seed.is_file_defined(arg0s) and "r" in mode:  
        logger.info(f"opening an input file: {arg0s}")
        
        data = se.seed.get_file_input(arg0s)
        filedesc = pstate.create_file_descriptor(arg0s, io.BytesIO(data))
        return filedesc.id
    else:
        
        try:
            fd = open(arg0s, mode)  
            filedesc = pstate.create_file_descriptor(arg0s, fd)
            return filedesc.id
        except Exception as e:
            logger.debug(f"Failed to open {arg0s} {e}")
            return pstate.minus_one


def rtn_realloc(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('realloc hooked')

    
    oldptr = pstate.get_argument_value(0)
    size = pstate.get_argument_value(1)

    if oldptr == 0:
        
        ptr = pstate.heap_allocator.alloc(size)
        return ptr

    ptr = pstate.heap_allocator.alloc(size)
    if ptr == 0:
        return ptr

    if ptr not in pstate.heap_allocator.alloc_pool:
        logger.warning("Invalid ptr passed to realloc")
        pstate.heap_allocator.free(ptr)     

    old_memmap = pstate.heap_allocator.alloc_pool[oldptr]
    old_size = old_memmap.size
    size_to_copy = min(size, old_size)

    
    

    
    for index in range(size_to_copy):
        sym_c = pstate.read_symbolic_memory_byte(oldptr+index)
        pstate.write_symbolic_memory_byte(ptr+index, sym_c)

    pstate.heap_allocator.free(oldptr)

    return ptr


def rtn_memcmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('memcmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    size = pstate.get_argument_value(2)

    ptr_bit_size = pstate.ptr_bit_size

    ast = pstate.actx
    res = ast.bv(0, ptr_bit_size)

    
    pstate.concretize_argument(2)

    for index in range(size):
        cells1 = pstate.read_symbolic_memory_byte(s1+index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2+index).getAst()
        res = res + ast.ite(
                        cells1 == cells2,
                        ast.bv(0, ptr_bit_size),
                        ast.bv(1, ptr_bit_size)
                    )

    return res


def rtn_memcpy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('memcpy hooked')

    dst, dst_ast = pstate.get_full_argument(0)
    src = pstate.get_argument_value(1)
    cnt = pstate.get_argument_value(2)

    
    pstate.concretize_argument(2)

    for index in range(cnt):
        
        sym_src = pstate.read_symbolic_memory_byte(src+index)
        pstate.write_symbolic_memory_byte(dst+index, sym_src)

    return dst_ast


def rtn_mempcpy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('mempcpy hooked')

    dst, dst_ast = pstate.get_full_argument(0)
    src = pstate.get_argument_value(1)
    cnt = pstate.get_argument_value(2)

    
    pstate.concretize_argument(2)

    for index in range(cnt):
        
        sym_src = pstate.read_symbolic_memory_byte(src+index)
        pstate.write_symbolic_memory_byte(dst+index, sym_src)

    return dst + cnt


def rtn_memmem(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('memmem hooked')

    haystack    = pstate.get_argument_value(0)      
    haystacklen = pstate.get_argument_value(1)      
    needle      = pstate.get_argument_value(2)      
    needlelen   = pstate.get_argument_value(3)      

    s1 = pstate.memory.read(haystack, haystacklen)  
    s2 = pstate.memory.read(needle, needlelen)      

    offset = s1.find(s2)
    if offset == -1:
        
        return NULL_PTR

    for i, c in enumerate(s2):
        c1 = pstate.read_symbolic_memory_byte(haystack+offset+i)
        c2 = pstate.read_symbolic_memory_byte(needle+i)
        pstate.push_constraint(c1.getAst() == c2.getAst())

    

    
    return haystack + offset


def rtn_memmove(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('memmove hooked')

    dst, dst_ast = pstate.get_full_argument(0)
    src = pstate.get_argument_value(1)
    cnt = pstate.get_argument_value(2)

    
    pstate.concretize_argument(2)

    
    for index in range(cnt):
        
        sym_src = pstate.read_symbolic_memory_byte(src+index)
        pstate.write_symbolic_memory_byte(dst+index, sym_src)

    return dst_ast


def rtn_memset(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('memset hooked')

    dst, dst_ast = pstate.get_full_argument(0)
    src, src_ast = pstate.get_full_argument(1)
    size = pstate.get_argument_value(2)

    
    pstate.concretize_argument(2)

    sym_cell = pstate.actx.extract(7, 0, src_ast.getAst())

    
    for index in range(size):
        pstate.write_symbolic_memory_byte(dst+index, sym_cell)

    return dst_ast


def rtn_printf(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('printf hooked')

    
    fmt_addr = pstate.get_argument_value(0)
    fmt_str = pstate.get_format_string(fmt_addr)
    arg_count = fmt_str.count("{")
    arg_values = [pstate.get_argument_value(x) for x in range(1, arg_count + 1)]
    arg_formatted = pstate.get_format_arguments(fmt_addr, arg_values)
    try:
        s = fmt_str.format(*arg_formatted)
    except:
        logger.warning('Something wrong, probably UTF-8 string')
        s = ""

    if se.config.pipe_stdout:
        stdout = pstate.get_file_descriptor(1)
        stdout.fd.write(s)
        stdout.fd.flush()

    
    return len(s)


def rtn_pthread_create(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('pthread_create hooked')

    
    arg0 = pstate.get_argument_value(0)     
    arg1 = pstate.get_argument_value(1)     
    arg2 = pstate.get_argument_value(2)     
    arg3 = pstate.get_argument_value(3)     

    th = pstate.spawn_new_thread(arg2, arg3)

    
    pstate.memory.write_ptr(arg0, th.tid)

    
    return 0


def rtn_pthread_exit(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('pthread_exit hooked')

    
    arg0 = pstate.get_argument_value(0)

    
    pstate.current_thread.kill()

    
    

    
    return None


def rtn_pthread_join(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('pthread_join hooked')

    
    arg0 = pstate.get_argument_value(0)
    arg1 = pstate.get_argument_value(1)

    if arg0 in pstate._threads:
        
        
        
        pstate.current_thread.join_thread(arg0)
        logger.info(f"Thread id {pstate.current_thread.tid} is waiting thread id {arg0} to join")
    else:
        pstate.current_thread.cancel_join()
        logger.debug(f"Thread id {arg0} already destroyed")

    
    return 0


def rtn_pthread_mutex_destroy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('pthread_mutex_destroy hooked')

    
    arg0 = pstate.get_argument_value(0)  
    pstate.memory.write_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

    return 0


def rtn_pthread_mutex_init(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('pthread_mutex_init hooked')

    
    arg0 = pstate.get_argument_value(0)  
    arg1 = pstate.get_argument_value(1)  

    pstate.memory.write_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

    return 0


def rtn_pthread_mutex_lock(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('pthread_mutex_lock hooked')

    
    arg0 = pstate.get_argument_value(0)  
    mutex = pstate.memory.read_ptr(arg0)  

    
    if mutex == pstate.PTHREAD_MUTEX_INIT_MAGIC:
        logger.debug('mutex unlocked')
        pstate.memory.write_ptr(arg0, pstate.current_thread.tid)

    
    elif mutex != pstate.current_thread.tid:
        logger.debug('mutex locked')
        pstate.mutex_locked = True

    
    return 0


def rtn_pthread_mutex_unlock(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('pthread_mutex_unlock hooked')

    
    arg0 = pstate.get_argument_value(0)  

    pstate.memory.write_ptr(arg0, pstate.PTHREAD_MUTEX_INIT_MAGIC)

    
    return 0


def rtn_puts(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('puts hooked')

    arg0 = pstate.get_string_argument(0)

    
    if se.config.pipe_stdout:  
        sys.stdout.write(arg0 + '\n')
        sys.stdout.flush()

    
    return len(arg0) + 1


def rtn_rand(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('rand hooked')
    return random.randrange(0, 0xffffffff)


def rtn_read(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('read hooked')

    
    fd   = pstate.get_argument_value(0)
    buff = pstate.get_argument_value(1)
    size, size_ast = pstate.get_full_argument(2)

    if size_ast.isSymbolized():
        logger.warning(f'Reading from the file descriptor ({fd}) with a symbolic size')

    pstate.concretize_argument(0)

    if pstate.file_descriptor_exists(fd):
        filedesc = pstate.get_file_descriptor(fd)
        offset = filedesc.offset
        data = filedesc.read(size)

        if filedesc.is_input_fd():  
            
            if se.seed.is_raw() and se.seed.is_bootstrap_seed() and not data:
                data = b'\x00' * size

            if len(data) == size:  
                pstate.push_constraint(size_ast.getAst() == size)

            se.inject_symbolic_file_memory(buff, filedesc.name, data, offset)
            logger.debug(f"read in (input) {filedesc.name} = {repr(data)}")
        else:
            pstate.concretize_argument(2)
            pstate.memory.write(buff, data)

        return len(data)
    else:
        logger.warning(f'File descriptor ({fd}) not found')
        return pstate.minus_one  


def rtn_getchar(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('getchar hooked')

    
    filedesc = pstate.get_file_descriptor(0)    
    offset = filedesc.offset

    data = filedesc.read(1)
    if data:
        if filedesc.is_input_fd():  
            se.inject_symbolic_file_register(pstate.return_register, filedesc.name, ord(data), offset)
            data = pstate.read_symbolic_register(pstate.return_register).getAst()
            pstate.push_constraint(pstate.actx.land([0 <= data, data <= 255]))
            logger.debug(f"read in {filedesc.name} = {repr(data)}")
        return data
    else:
        return pstate.minus_one


def rtn_sem_destroy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('sem_destroy hooked')

    
    arg0 = pstate.get_argument_value(0)  

    
    pstate.memory.write_ptr(arg0, 0)

    
    return 0


def rtn_sem_getvalue(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('sem_getvalue hooked')

    
    arg0 = pstate.get_argument_value(0)  
    arg1 = pstate.get_argument_value(1)  

    value = pstate.memory.read_ptr(arg0)  

    
    pstate.memory.write_dword(arg1, value)  

    
    return 0


def rtn_sem_init(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('sem_init hooked')

    
    arg0 = pstate.get_argument_value(0)  
    arg1 = pstate.get_argument_value(1)  
    arg2 = pstate.get_argument_value(2)  

    
    pstate.memory.write_ptr(arg0, arg2)

    
    return 0


def rtn_sem_post(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('sem_post hooked')

    arg0 = pstate.get_argument_value(0)  

    
    value = pstate.memory.read_ptr(arg0)
    pstate.memory.write_ptr(arg0, value + 1)

    
    return 0


def rtn_sem_timedwait(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('sem_timedwait hooked')

    arg0 = pstate.get_argument_value(0)  
    arg1 = pstate.get_argument_value(1)  

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    value = pstate.memory.read_ptr(arg0)
    if value > 0:
        logger.debug('semaphore still not locked')
        pstate.memory.write_ptr(arg0, value - 1)
        pstate.semaphore_locked = False
    else:
        logger.debug('semaphore locked')
        pstate.semaphore_locked = True

    
    return 0


def rtn_sem_trywait(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('sem_trywait hooked')

    arg0 = pstate.get_argument_value(0)  

    
    
    
    value = pstate.memory.read_ptr(arg0)
    if value > 0:
        logger.debug('semaphore still not locked')
        pstate.memory.write_ptr(arg0, value - 1)
        pstate.semaphore_locked = False
    else:
        logger.debug('semaphore locked but continue')
        pstate.semaphore_locked = False

        
        segs = pstate.memory.find_map(pstate.EXTERN_SEG)
        if segs:
            mmap = segs[0]
            ERRNO = mmap.start + mmap.size - 4  
            pstate.memory.write_dword(ERRNO, 3406)
        else:
            assert False

        
        return pstate.minus_one

    
    return 0


def rtn_sem_wait(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('sem_wait hooked')

    arg0 = pstate.get_argument_value(0)  

    
    
    
    
    
    value = pstate.memory.read_ptr(arg0)
    if value > 0:
        logger.debug('semaphore still not locked')
        pstate.memory.write_ptr(arg0, value - 1)
        pstate.semaphore_locked = False
    else:
        logger.debug('semaphore locked')
        pstate.semaphore_locked = True

    
    return 0


def rtn_sleep(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('sleep hooked')

    
    if not se.config.skip_sleep_routine:
        t = pstate.get_argument_value(0)
        time.sleep(t)

    
    return 0


def rtn_sprintf(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('sprintf hooked')

    
    buff = pstate.get_argument_value(0)
    arg0 = pstate.get_argument_value(1)

    try:
        arg0f = pstate.get_format_string(arg0)
        nbArgs = arg0f.count("{")
        args = pstate.get_format_arguments(arg0, [pstate.get_argument_value(x) for x in range(2, nbArgs+2)])
        s = arg0f.format(*args)
    except:
        
        logger.warning('Something wrong, probably UTF-8 string')
        s = ""

    

    
    
    
    
    
    
    
    
    

    return len(s)


def rtn_strcasecmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strcasecmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    size = min(len(pstate.memory.read_string(s1)), len(pstate.memory.read_string(s2)) + 1)

    
    
    
    
    
    
    

    

    ptr_bit_size = pstate.ptr_bit_size
    ast = pstate.actx
    res = ast.bv(0, pstate.ptr_bit_size)
    for index in range(size):
        cells1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        cells1 = ast.ite(ast.land([cells1 >= ord('a'), cells1 <= ord('z')]), cells1 - 32, cells1)   
        cells2 = ast.ite(ast.land([cells2 >= ord('a'), cells2 <= ord('z')]), cells2 - 32, cells2)   
        res = res + ast.ite(cells1 == cells2, ast.bv(0, ptr_bit_size), ast.bv(1, ptr_bit_size))

    return res


def rtn_strchr(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strchr hooked')

    string = pstate.get_argument_value(0)
    char   = pstate.get_argument_value(1)
    ast    = pstate.actx
    ptr_bit_size = pstate.ptr_bit_size

    def rec(res, deep, maxdeep):
        if deep == maxdeep:
            return res
        cell = pstate.read_symbolic_memory_byte(string + deep).getAst()
        res  = ast.ite(cell == (char & 0xff), ast.bv(string + deep, ptr_bit_size), rec(res, deep + 1, maxdeep))
        return res

    sze = len(pstate.memory.read_string(string))
    res = rec(ast.bv(0, ptr_bit_size), 0, sze)

    for i, c in enumerate(pstate.memory.read_string(string)):
        pstate.push_constraint(pstate.read_symbolic_memory_byte(string+i).getAst() != 0x00)
    pstate.push_constraint(pstate.read_symbolic_memory_byte(string+sze).getAst() == 0x00)

    return res


def rtn_strcmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strcmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    size = min(len(pstate.memory.read_string(s1)), len(pstate.memory.read_string(s2))) + 1

    ptr_bit_size = pstate.ptr_bit_size
    ast = pstate.actx
    res = ast.bv(0, ptr_bit_size)
    for index in range(size, -1, -1):
        c1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        c2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        res = ast.ite(ast.lor([c1 == 0, c1 != c2]), ast.sx(ptr_bit_size - 8, c1 - c2), res)

    return res


def rtn_strcpy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strcpy hooked')

    dst  = pstate.get_argument_value(0)
    src  = pstate.get_argument_value(1)
    src_str = pstate.memory.read_string(src)
    size = len(src_str)

    
    for i, c in enumerate(src_str):
        pstate.push_constraint(pstate.read_symbolic_memory_byte(src + i).getAst() != 0x00)
    pstate.push_constraint(pstate.read_symbolic_memory_byte(src + size).getAst() == 0x00)

    
    for index in range(size+1):
        sym_c = pstate.read_symbolic_memory_byte(src+index)
        pstate.write_symbolic_memory_byte(dst+index, sym_c)

    return dst


def rtn_strdup(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strdup hooked')

    s  = pstate.get_argument_value(0)
    s_str = pstate.memory.read_string(s)
    size = len(s_str)

    

    
    for i, c in enumerate(s_str):
        pstate.push_constraint(pstate.read_symbolic_memory_byte(s + i).getAst() != 0x00)
    pstate.push_constraint(pstate.read_symbolic_memory_byte(s + size).getAst() == 0x00)

    
    ptr = pstate.heap_allocator.alloc(size + 1)

    
    for index in range(size+1):
        sym_c = pstate.read_symbolic_memory_byte(s+index)
        pstate.write_symbolic_memory_byte(ptr+index, sym_c)

    return ptr


def rtn_strerror(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strerror hooked')

    sys_errlist = [
        b"Success",
        b"Operation not permitted",
        b"No such file or directory",
        b"No such process",
        b"Interrupted system call",
        b"Input/output error",
        b"No such device or address",
        b"Argument list too long",
        b"Exec format error",
        b"Bad file descriptor",
        b"No child processes",
        b"Resource temporarily unavailable",
        b"Cannot allocate memory",
        b"Permission denied",
        b"Bad address",
        b"Block device required",
        b"Device or resource busy",
        b"File exists",
        b"Invalid cross-device link",
        b"No such device",
        b"Not a directory",
        b"Is a directory",
        b"Invalid argument",
        b"Too many open files in system",
        b"Too many open files",
        b"Inappropriate ioctl for device",
        b"Text file busy",
        b"File too large",
        b"No space left on device",
        b"Illegal seek",
        b"Read-only file system",
        b"Too many links",
        b"Broken pipe",
        b"Numerical argument out of domain",
        b"Numerical result out of range",
        b"Resource deadlock avoided",
        b"File name too long",
        b"No locks available",
        b"Function not implemented",
        b"Directory not empty",
        b"Too many levels of symbolic links",
        None,
        b"No message of desired type",
        b"Identifier removed",
        b"Channel number out of range",
        b"Level 2 not synchronized",
        b"Level 3 halted",
        b"Level 3 reset",
        b"Link number out of range",
        b"Protocol driver not attached",
        b"No CSI structure available",
        b"Level 2 halted",
        b"Invalid exchange",
        b"Invalid request descriptor",
        b"Exchange full",
        b"No anode",
        b"Invalid request code",
        b"Invalid slot",
        None,
        b"Bad font file format",
        b"Device not a stream",
        b"No data available",
        b"Timer expired",
        b"Out of streams resources",
        b"Machine is not on the network",
        b"Package not installed",
        b"Object is remote",
        b"Link has been severed",
        b"Advertise error",
        b"Srmount error",
        b"Communication error on send",
        b"Protocol error",
        b"Multihop attempted",
        b"RFS specific error",
        b"Bad message",
        b"Value too large for defined data type",
        b"Name not unique on network",
        b"File descriptor in bad state",
        b"Remote address changed",
        b"Can not access a needed shared library",
        b"Accessing a corrupted shared library",
        b".lib section in a.out corrupted",
        b"Attempting to link in too many shared libraries",
        b"Cannot exec a shared library directly",
        b"Invalid or incomplete multibyte or wide character",
        b"Interrupted system call should be restarted",
        b"Streams pipe error",
        b"Too many users",
        b"Socket operation on non-socket",
        b"Destination address required",
        b"Message too long",
        b"Protocol wrong type for socket",
        b"Protocol not available",
        b"Protocol not supported",
        b"Socket type not supported",
        b"Operation not supported",
        b"Protocol family not supported",
        b"Address family not supported by protocol",
        b"Address already in use",
        b"Cannot assign requested address",
        b"Network is down",
        b"Network is unreachable",
        b"Network dropped connection on reset",
        b"Software caused connection abort",
        b"Connection reset by peer",
        b"No buffer space available",
        b"Transport endpoint is already connected",
        b"Transport endpoint is not connected",
        b"Cannot send after transport endpoint shutdown",
        b"Too many references: cannot splice",
        b"Connection timed out",
        b"Connection refused",
        b"Host is down",
        b"No route to host",
        b"Operation already in progress",
        b"Operation now in progress",
        b"Stale NFS file handle",
        b"Structure needs cleaning",
        b"Not a XENIX named type file",
        b"No XENIX semaphores available",
        b"Is a named type file",
        b"Remote I/O error",
        b"Disk quota exceeded",
        b"No medium found",
        b"Wrong medium type",
        b"Operation canceled",
        b"Required key not available",
        b"Key has expired",
        b"Key has been revoked",
        b"Key was rejected by service",
        b"Owner died",
        b"State not recoverable"
    ]

    
    errnum = pstate.get_argument_value(0)
    try:
        string = sys_errlist[errnum]
    except:
        
        string = b'Error'

    
    
    
    ptr = pstate.heap_allocator.alloc(len(string) + 1)
    pstate.memory.write(ptr, string + b'\0')

    return ptr


def rtn_strlen(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strlen hooked')

    ptr_bit_size = pstate.ptr_bit_size

    
    s = pstate.get_argument_value(0)
    ast = pstate.actx

    
    def rec(res, s, deep, maxdeep):
        if deep == maxdeep:
            return res
        cell = pstate.read_symbolic_memory_byte(s+deep).getAst()
        res  = ast.ite(cell == 0x00, ast.bv(deep, ptr_bit_size), rec(res, s, deep + 1, maxdeep))
        return res

    sze = len(pstate.memory.read_string(s))
    res = ast.bv(sze, ptr_bit_size)
    res = rec(res, s, 0, sze)

    
    
    
    

    pstate.push_constraint(pstate.read_symbolic_memory_byte(s+sze).getAst() == 0x00)

    return res


def rtn_strncasecmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strncasecmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    sz = pstate.get_argument_value(2)
    maxlen = min(sz, min(len(pstate.memory.read_string(s1)), len(pstate.memory.read_string(s2))) + 1)

    ptr_bit_size = pstate.ptr_bit_size

    ast = pstate.actx
    res = ast.bv(0, ptr_bit_size)
    for index in range(maxlen):
        cells1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        cells1 = ast.ite(ast.land([cells1 >= ord('a'), cells1 <= ord('z')]), cells1 - 32, cells1)   
        cells2 = ast.ite(ast.land([cells2 >= ord('a'), cells2 <= ord('z')]), cells2 - 32, cells2)   
        res = res + ast.ite(cells1 == cells2, ast.bv(0, ptr_bit_size), ast.bv(1, ptr_bit_size))

    return res


def rtn_strncmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strncmp hooked')

    s1 = pstate.get_argument_value(0)
    s2 = pstate.get_argument_value(1)
    sz = pstate.get_argument_value(2)
    maxlen = min(sz, min(len(pstate.memory.read_string(s1)), len(pstate.memory.read_string(s2))) + 1)

    ptr_bit_size = pstate.ptr_bit_size

    ast = pstate.actx
    res = ast.bv(0, ptr_bit_size)
    for index in range(maxlen):
        cells1 = pstate.read_symbolic_memory_byte(s1 + index).getAst()
        cells2 = pstate.read_symbolic_memory_byte(s2 + index).getAst()
        res = res + ast.ite(cells1 == cells2, ast.bv(0, ptr_bit_size), ast.bv(1, ptr_bit_size))

    return res


def rtn_strncpy(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strncpy hooked')

    dst = pstate.get_argument_value(0)
    src = pstate.get_argument_value(1)
    cnt = pstate.get_argument_value(2)

    pstate.concretize_argument(2)

    for index in range(cnt):
        src_sym = pstate.read_symbolic_memory_byte(src+index)
        pstate.write_symbolic_memory_byte(dst+index, src_sym)

        if src_sym.getAst().evaluate() == 0:
            pstate.push_constraint(src_sym.getAst() == 0x00)
            break
        else:
            pstate.push_constraint(src_sym.getAst() != 0x00)

    return dst


def rtn_strtok_r(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strtok_r hooked')

    string  = pstate.get_argument_value(0)
    delim   = pstate.get_argument_value(1)
    saveptr = pstate.get_argument_value(2)
    saveMem = pstate.memory.read_ptr(saveptr)

    if string == 0:
        string = saveMem

    d = pstate.memory.read_string(delim)
    s = pstate.memory.read_string(string)

    tokens = re.split('[' + re.escape(d) + ']', s)

    
    for token in tokens:
        if token:
            offset = s.find(token)
            
            node = pstate.read_symbolic_memory_byte(string + offset + len(token)).getAst()
            try:
                pstate.push_constraint(pstate.actx.lor([node == ord(c) for c in d]))
            except:  
                pstate.push_constraint(node == ord(d))

            
            for index, char in enumerate(token):
                node = pstate.read_symbolic_memory_byte(string + offset + index).getAst()
                for delim in d:
                    pstate.push_constraint(node != ord(delim))

            pstate.memory.write_char(string + offset + len(token), 0)
            
            pstate.memory.write_ptr(saveptr, string + offset + len(token) + 1)
            
            return string + offset

    return NULL_PTR


def rtn_strtoul(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('strtoul hooked')

    nptr   = pstate.get_argument_value(0)
    nptrs  = pstate.get_string_argument(0)
    endptr = pstate.get_argument_value(1)
    base   = pstate.get_argument_value(2)

    for i, c in enumerate(nptrs):
        pstate.push_constraint(pstate.read_symbolic_memory_byte(nptr+i).getAst() == ord(c))

    pstate.concretize_argument(2)  

    try:
        return int(nptrs, base)
    except:
        return 0xffffffff


def rtn_getenv(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    
    name = pstate.get_argument_value(0)

    if name == 0:
        return NULL_PTR

    environ_name = pstate.memory.read_string(name)
    logger.warning(f"Target called getenv({environ_name})")
    host_env_val = os.getenv(environ_name)
    return host_env_val if host_env_val is not None else 0












def rtn_isspace(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    ptr_bit_size = pstate.ptr_bit_size
    ast = pstate.actx
    arg_sym = pstate.get_argument_symbolic(0)

    exp = arg_sym.getAst() == 0x20
    exp = ast.lor([exp, arg_sym.getAst() == 0xa])
    exp = ast.lor([exp, arg_sym.getAst() == 0x9])
    exp = ast.lor([exp, arg_sym.getAst() == 0xc])
    exp = ast.lor([exp, arg_sym.getAst() == 0xd])
    res = ast.ite(exp, ast.bv(0, ptr_bit_size), ast.bv(1, ptr_bit_size))
    return res


def rtn_assert_fail(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    msg = pstate.get_argument_value(0)

    msg = pstate.memory.read_string(msg)
    logger.warning(f"__assert_fail called : {msg}")

    
    pstate.write_register(pstate.return_register, 1)
    se.abort()


def rtn_setlocale(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    logger.debug('setlocale hooked')

    category = pstate.get_argument_value(0)
    locale   = pstate.get_argument_value(1)

    if locale != 0:
        logger.warning(f"Attempt to modify Locale. Currently not supported.")
        return 0

    
    segs = pstate.memory.find_map(pstate.EXTERN_SEG)
    if segs:
        mmap = segs[0]
        LC_ALL = mmap.start + mmap.size - 0x20    
    else:
        assert False
    print(f"selocale writing at {LC_ALL:

    if category == 0:
        pstate.memory.write(LC_ALL, b"en_US.UTF-8\x00")
    else:
        logger.warning(f"setlocale called with unsupported category={category}.")

    return LC_ALL


def rtn__setjmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    
    logger.warning("hooked _setjmp")
    return 0


def rtn_longjmp(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    
    
    
    
    logger.debug('longjmp hooked')
    pstate.stop = True


def rtn_atexit(se: 'SymbolicExecutor', pstate: 'ProcessState'):
    return 0


SUPPORTED_ROUTINES = {
    
    
    
    '__assert_fail':           rtn_assert_fail,
    '__ctype_b_loc':           rtn_ctype_b_loc,
    '__ctype_toupper_loc':     rtn_ctype_toupper_loc,
    '__errno_location':        rtn_errno_location,
    '__fprintf_chk':           rtn___fprintf_chk,
    '__libc_start_main':       rtn_libc_start_main,
    '__stack_chk_fail':        rtn_stack_chk_fail,
    '__xstat':                 rtn_xstat,
    'abort':                   rtn_abort,
    "atexit":                  rtn_atexit,
    "__cxa_atexit":            rtn_atexit,
    'atoi':                    rtn_atoi,
    'calloc':                  rtn_calloc,
    'clock_gettime':           rtn_clock_gettime,
    'exit':                    rtn_exit,
    'fclose':                  rtn_fclose,
    'fgets':                   rtn_fgets,
    'fopen':                   rtn_fopen,
    'fprintf':                 rtn_fprintf,
    'fputc':                   rtn_fputc,
    'fputs':                   rtn_fputs,
    'fread':                   rtn_fread,
    'free':                    rtn_free,
    'fwrite':                  rtn_fwrite,
    'gettimeofday':            rtn_gettimeofday,
    'malloc':                  rtn_malloc,
    'memcmp':                  rtn_memcmp,
    'memcpy':                  rtn_memcpy,
    'memmem':                  rtn_memmem,
    'memmove':                 rtn_memmove,
    'memset':                  rtn_memset,
    'open':                    rtn_open,
    'printf':                  rtn_printf,
    'pthread_create':          rtn_pthread_create,
    'pthread_exit':            rtn_pthread_exit,
    'pthread_join':            rtn_pthread_join,
    'pthread_mutex_destroy':   rtn_pthread_mutex_destroy,
    'pthread_mutex_init':      rtn_pthread_mutex_init,
    'pthread_mutex_lock':      rtn_pthread_mutex_lock,
    'pthread_mutex_unlock':    rtn_pthread_mutex_unlock,
    'puts':                    rtn_puts,
    'rand':                    rtn_rand,
    'read':                    rtn_read,
    'sem_destroy':             rtn_sem_destroy,
    'sem_getvalue':            rtn_sem_getvalue,
    'sem_init':                rtn_sem_init,
    'sem_post':                rtn_sem_post,
    'sem_timedwait':           rtn_sem_timedwait,
    'sem_trywait':             rtn_sem_trywait,
    'sem_wait':                rtn_sem_wait,
    'sleep':                   rtn_sleep,
    'sprintf':                 rtn_sprintf,
    'strcasecmp':              rtn_strcasecmp,
    'strchr':                  rtn_strchr,
    'strcmp':                  rtn_strcmp,
    'strcpy':                  rtn_strcpy,
    'strerror':                rtn_strerror,
    'strlen':                  rtn_strlen,
    'strncasecmp':             rtn_strncasecmp,
    'strncmp':                 rtn_strncmp,
    'strncpy':                 rtn_strncpy,
    'strtok_r':                rtn_strtok_r,
    'strtoul':                 rtn_strtoul,

    'write':                   rtn_write,
    'getenv':                  rtn_getenv,
    'fseek':                   rtn_fseek,
    'ftell':                   rtn_ftell,

    '_setjmp':                 rtn__setjmp,
    'longjmp':                 rtn_longjmp,
    'realloc':                 rtn_realloc,
    'setlocale':               rtn_setlocale,
    'strdup':                  rtn_strdup,
    'mempcpy':                 rtn_mempcpy,
    '__mempcpy':               rtn_mempcpy,
    'getchar':                 rtn_getchar,

    'isspace':                 rtn_isspace,
    
}


SUPORTED_GVARIABLES = {
    '__stack_chk_guard':    0xdead,
    'stderr':               0x0002,
    'stdin':                0x0000,
    'stdout':               0x0001,
}

