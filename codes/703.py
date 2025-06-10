
from __future__ import annotations
import io
import struct
import sys
import time
from typing import Union, Callable, Tuple, Optional, List, Dict



from triton import TritonContext, MemoryAccess, CALLBACK, CPUSIZE, Instruction, MODE, AST_NODE, SOLVER, EXCEPTION


from tritondse.thread_context import ThreadContext
from tritondse.heap_allocator import HeapAllocator
from tritondse.types import Architecture, Addr, ByteSize, BitSize, PathConstraint, Register, Expression, \
                            AstNode, Registers, SolverStatus, Model, SymbolicVariable, ArchMode, Perm, FileDesc, Endian
from tritondse.arch import ARCHS, CpuState
from tritondse.loaders.loader import Loader
from tritondse.memory import Memory, MemoryAccessViolation
import tritondse.logging

logger = tritondse.logging.get('processstate')


class ProcessState(object):
    

    STACK_SEG = "[stack]"
    EXTERN_SEG = "[extern]"

    def __init__(self, endianness: Endian = Endian.LITTLE, time_inc_coefficient: float = 0.0001):
        
        
        
        self.EXTERN_FUNC_BASE = 0x01000000  

        
        
        self.BASE_HEAP = 0x10000000
        self.END_HEAP = 0x6fffffff

        
        self.tt_ctx = TritonContext()
        
        self.actx: 'AstContext' = self.tt_ctx.getAstContext()
        

        
        self.cpu: Optional[CpuState] = None  
        self._archinfo = None

        
        self.memory: Memory = Memory(self.tt_ctx, endianness)
        

        
        self.stop = False

        
        

        
        self.dynamic_symbol_table: Dict[str, Tuple[Addr, bool]] = {}
        

        
        self._fd_table = {
            0: FileDesc(0, "stdin", sys.stdin),
            1: FileDesc(1, "stdout", sys.stdout),
            2: FileDesc(2, "stderr", sys.stderr),
        }
        
        self._fd_id = len(self._fd_table)

        
        self.heap_allocator: HeapAllocator = HeapAllocator(self.BASE_HEAP, self.END_HEAP, self.memory)
        

        
        self._utid = 0

        
        self._tid = self._utid

        
        self._threads = {
            self._tid: ThreadContext(self._tid)
        }

        
        self.PTHREAD_MUTEX_INIT_MAGIC = 0xdead

        
        self.mutex_locked = False
        self.semaphore_locked = False

        
        
        
        self.time = time.time()

        
        self.endianness = endianness  
        self.time_inc_coefficient = time_inc_coefficient

        
        self.__pcs_updated = False

        
        self.__current_inst = None

        
        self.__program_segments_mapping = {}

        
        self.rtn_redirect_addr = None

    @property
    def threads(self) -> List[ThreadContext]:
        
        return list(self._threads.values())

    @property
    def current_thread(self) -> ThreadContext:
        
        return self._threads[self._tid]

    def switch_thread(self, thread: ThreadContext) -> bool:
        
        assert (thread.tid in self._threads)

        try:
            if self.current_thread.is_dead():
                del self._threads[self._tid]
                
            else:  
                
                self.current_thread.save(self.tt_ctx)

            
            thread.count = 0  
            thread.restore(self.tt_ctx)
            self._tid = thread.tid
            return True

        except Exception as e:
            logger.error(f"Error while doing context switch: {e}")
            return False

    def spawn_new_thread(self, new_pc: Addr, args: Addr) -> ThreadContext:
        
        tid = self._get_unique_thread_id()
        thread = ThreadContext(tid)
        thread.save(self.tt_ctx)

        
        regs = [
            self.program_counter_register,
            self.stack_pointer_register,
            self.base_pointer_register,
            self._get_argument_register(0)
        ]
        for reg in regs:
            if reg.getId() in thread.sregs:
                del thread.sregs[reg.getId()]

        thread.cregs[self.program_counter_register.getId()] = new_pc  
        thread.cregs[self._get_argument_register(0).getId()] = args   
        stack = self.memory.map_from_name(self.STACK_SEG)
        stack_base_addr = ((stack.start + stack.size - self.ptr_size) - ((1 << 28) * tid))
        thread.cregs[self.base_pointer_register.getId()] = stack_base_addr
        thread.cregs[self.stack_pointer_register.getId()] = stack_base_addr

        if self.architecture == Architecture.AARCH64:
            thread.cregs[getattr(self.registers, 'x30').getId()] = 0xcafecafe
        elif self.architecture == Architecture.ARM32:
            thread.cregs[getattr(self.registers, 'r14').getId()] = 0xcafecafe
        elif self.architecture in [Architecture.X86, Architecture.X86_64]:
            self.memory.write_ptr(stack_base_addr, 0xcafecafe)

        
        self._threads[tid] = thread
        return thread

    def set_triton_mode(self, mode: MODE, value: int = True) -> None:
        
        self.tt_ctx.setMode(mode, value)

    def set_thumb(self, enable: bool) -> None:
        
        self.tt_ctx.setThumb(enable)

    def set_solver_timeout(self, timeout: int) -> None:
        
        self.tt_ctx.setSolverTimeout(timeout)

    def set_solver(self, solver: Union[str, SOLVER]) -> None:
        
        if isinstance(solver, str):
            solver = getattr(SOLVER, solver.upper(), SOLVER.Z3)
        self.tt_ctx.setSolver(solver)

    def _get_unique_thread_id(self) -> int:
        
        self._utid += 1
        return self._utid

    def create_file_descriptor(self, name: str, file: io.IOBase) -> FileDesc:
        
        new_fd_id = self._fd_id
        self._fd_id += 1
        filedesc = FileDesc(id=new_fd_id, name=name, fd=file)
        self._fd_table[new_fd_id] = filedesc
        return filedesc

    def close_file_descriptor(self, fd_id: int) -> None:
        
        filedesc = self._fd_table.pop(fd_id)
        if isinstance(filedesc.fd, io.IOBase):
            filedesc.fd.close()

    def get_file_descriptor(self, id_: int) -> FileDesc:
        
        return self._fd_table[id_]

    def file_descriptor_exists(self, id_: int) -> bool:
        
        return bool(id_ in self._fd_table)

    @property
    def architecture(self) -> Architecture:
        
        return Architecture(self.tt_ctx.getArchitecture())

    @architecture.setter
    def architecture(self, arch: Architecture) -> None:
        
        self.tt_ctx.setArchitecture(arch)

    @property
    def ptr_size(self) -> ByteSize:
        
        return self.tt_ctx.getGprSize()

    @property
    def ptr_bit_size(self) -> BitSize:
        
        return self.tt_ctx.getGprBitSize()

    @property
    def minus_one(self) -> int:
        
        return (1 << self.ptr_bit_size) - 1

    @property
    def registers(self) -> Registers:
        
        return self.tt_ctx.registers

    @property
    def return_register(self) -> Register:
        
        return getattr(self.registers, self._archinfo.ret_reg)

    @property
    def program_counter_register(self) -> Register:
        
        return getattr(self.registers, self._archinfo.pc_reg)

    @property
    def base_pointer_register(self) -> Register:
        
        return getattr(self.registers, self._archinfo.bp_reg)

    @property
    def stack_pointer_register(self) -> Register:
        
        return getattr(self.registers, self._archinfo.sp_reg)

    @property
    def _syscall_register(self) -> Register:
        
        return getattr(self.registers, self._archinfo.sys_reg)

    def _get_argument_register(self, i: int) -> Register:
        
        return getattr(self.registers, self._archinfo.reg_args[i])

    def initialize_context(self, arch: Architecture):
        
        self.architecture = arch
        self._archinfo = ARCHS[self.architecture]
        self.cpu = CpuState(self.tt_ctx, self._archinfo)

    def unpack_integer(self, data: bytes, size: int) -> int:
        
        s = "<" if self.endianness == Endian.LITTLE else ">"
        tab = {8: 'B', 16: 'H', 32: 'I', 64: 'Q'}
        s += tab[size]
        return struct.unpack(s, data)[0]

    def pack_integer(self, value: int, size: int) -> bytes:
        
        s = "<" if self.endianness == Endian.LITTLE else ">"
        tab = {8: 'B', 16: 'H', 32: 'I', 64: 'Q'}
        s += tab[size]
        return struct.pack(s, value)

    def read_register(self, register: Union[str, Register]) -> int:
        
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  
        return self.tt_ctx.getConcreteRegisterValue(reg)

    def write_register(self, register: Union[str, Register], value: int) -> None:
        
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  
        return self.tt_ctx.setConcreteRegisterValue(reg, value)

    def register_triton_callback(self, cb_type: CALLBACK, callback: Callable) -> None:
        
        self.tt_ctx.addCallback(cb_type, callback)

    def clear_triton_callbacks(self) -> None:
        
        self.tt_ctx.clearCallbacks()

    def is_heap_ptr(self, ptr: Addr) -> bool:
        
        if self.BASE_HEAP <= ptr < self.END_HEAP:
            return True
        return False

    def is_syscall(self) -> bool:
        
        return bool(self.current_instruction.getType() in self._archinfo.syscall_inst)

    def fetch_instruction(self, address: Addr = None, set_as_current: bool = True, disable_callbacks: bool = True) -> Instruction:
        
        if address is None:
            address = self.cpu.program_counter
        with self.memory.without_segmentation(disable_callbacks=disable_callbacks):
            data = self.memory.read(address, 16)
        i = Instruction(address, data)
        i.setThreadId(self.current_thread.tid)
        self.tt_ctx.disassembly(i)  
                                    

        if self.memory.segmentation_enabled:
            mmap = self.memory.get_map(address, i.getSize())
            if mmap is None:
                raise MemoryAccessViolation(address, Perm.X, memory_not_mapped=True)
            if Perm.X not in mmap.perm:  
                raise MemoryAccessViolation(address, Perm.X, map_perm=mmap.perm, perm_error=True)
        if set_as_current:
            self.__current_inst = i
        return i

    def process_instruction(self, instruction: Instruction) -> bool:
        
        self.__pcs_updated = False
        __len_pcs = self.tt_ctx.getPathPredicateSize()

        if not instruction.getDisassembly():  
            self.tt_ctx.disassembly(instruction)

        self.__current_inst = instruction
        ret = self.tt_ctx.buildSemantics(instruction)

        
        
        
        
        self.time += self.time_inc_coefficient

        if self.tt_ctx.getPathPredicateSize() > __len_pcs:
            self.__pcs_updated = True

        return ret == EXCEPTION.NO_FAULT

    @property
    def path_predicate_size(self) -> int:
        
        return self.tt_ctx.getPathPredicateSize()

    def is_path_predicate_updated(self) -> bool:
        
        return self.__pcs_updated

    @property
    def last_branch_constraint(self) -> PathConstraint:
        
        return self.tt_ctx.getPathConstraints()[-1]

    @property
    def current_instruction(self) -> Optional[Instruction]:
        
        return self.__current_inst

    def is_register_symbolic(self, register: Union[str, Register]) -> bool:
        
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register
        return self.tt_ctx.getRegisterAst(reg).isSymbolized()

    def read_symbolic_register(self, register: Union[str, Register]) -> Expression:
        
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  
        sym_reg = self.tt_ctx.getSymbolicRegister(reg)

        if sym_reg is None or reg.getBitSize() != sym_reg.getAst().getBitvectorSize():
            return self.tt_ctx.newSymbolicExpression(self.tt_ctx.getRegisterAst(reg))
        else:
            return sym_reg

    def write_symbolic_register(self, register: Union[str, Register], expr: Union[AstNode, Expression], comment: str = "") -> None:
        
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  
        exp = expr if hasattr(expr, "getAst") else self.tt_ctx.newSymbolicExpression(expr, f"assign {reg.getName()}: {comment}")
        self.write_register(reg, exp.getAst().evaluate())  
        self.tt_ctx.assignSymbolicExpressionToRegister(exp, reg)

    def read_symbolic_memory_int(self, addr: Addr, size: ByteSize) -> Expression:
        
        if size == 1:
            return self.read_symbolic_memory_byte(addr)
        elif size in [2, 4, 8, 16, 32, 64]:
            ast = self.tt_ctx.getMemoryAst(MemoryAccess(addr, size))
            return self.tt_ctx.newSymbolicExpression(ast)
        else:
            raise RuntimeError("size should be aligned [1, 2, 4, 8, 16, 32, 64] (bytes)")

    def read_symbolic_memory_byte(self, addr: Addr) -> Expression:
        
        res = self.tt_ctx.getSymbolicMemory(addr)
        if res is None:
            return self.tt_ctx.newSymbolicExpression(self.tt_ctx.getMemoryAst(MemoryAccess(addr, 1)))
        else:
            return res

    def read_symbolic_memory_bytes(self, addr: Addr, size: ByteSize) -> Expression:
        
        if size == 1:
            return self.read_symbolic_memory_byte(addr)
        else:  
            asts = [self.tt_ctx.getMemoryAst(MemoryAccess(addr+i, CPUSIZE.BYTE)) for i in range(size)]
            concat_expr = self.actx.concat(asts)
            return self.tt_ctx.newSymbolicExpression(concat_expr)

    def write_symbolic_memory_int(self, addr: Addr, size: ByteSize, expr: Union[AstNode, Expression]) -> None:
        
        expr = expr if hasattr(expr, "getAst") else self.tt_ctx.newSymbolicExpression(expr, f"assign memory")
        if size in [1, 2, 4, 8, 16, 32, 64]:
            self.tt_ctx.setConcreteMemoryValue(MemoryAccess(addr, size), expr.getAst().evaluate())  
            self.tt_ctx.assignSymbolicExpressionToMemory(expr, MemoryAccess(addr, size))
        else:
            raise RuntimeError("size should be aligned [1, 2, 4, 8, 16, 32, 64] (bytes)")

    def write_symbolic_memory_byte(self, addr: Addr, expr: Union[AstNode, Expression]) -> None:
        
        expr = expr if hasattr(expr, "getAst") else self.tt_ctx.newSymbolicExpression(expr, f"assign memory")
        ast = expr.getAst()
        assert ast.getBitvectorSize() == 8
        self.tt_ctx.setConcreteMemoryValue(MemoryAccess(addr, CPUSIZE.BYTE), ast.evaluate())  
        self.tt_ctx.assignSymbolicExpressionToMemory(expr, MemoryAccess(addr, CPUSIZE.BYTE))

    def is_memory_symbolic(self, addr: Addr, size: ByteSize) -> bool:
        
        for i in range(addr, addr+size):
            if self.tt_ctx.isMemorySymbolized(MemoryAccess(i, 1)):
                return True
        return False

    def push_constraint(self, constraint: AstNode, comment: str = "") -> None:
        
        self.tt_ctx.pushPathConstraint(constraint, comment)

    def get_path_constraints(self) -> List[PathConstraint]:
        
        return self.tt_ctx.getPathConstraints()

    def concretize_register(self, register: Union[str, Register]) -> None:
        
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register
        if self.tt_ctx.isRegisterSymbolized(reg):
            value = self.read_register(reg)
            self.push_constraint(self.read_symbolic_register(reg).getAst() == value)
        

    def concretize_memory_bytes(self, addr: Addr, size: ByteSize) -> None:
        
        data = self.memory.read(addr, size)
        if self.is_memory_symbolic(addr, size):
            if isinstance(data, bytes):
                data_ast = self.actx.concat([self.actx.bv(b, 8) for b in data])
                self.push_constraint(self.read_symbolic_memory_bytes(addr, size).getAst() == data_ast)
            else:
                self.push_constraint(self.read_symbolic_memory_bytes(addr, size).getAst() == data)
        

    def concretize_memory_int(self, addr: Addr, size: ByteSize) -> None:
        
        value = self.memory.read_uint(addr, size)
        if self.tt_ctx.isMemorySymbolized(MemoryAccess(addr, size)):
            self.push_constraint(self.read_symbolic_memory_int(addr, size).getAst() == value)
        

    def concretize_argument(self, index: int) -> None:
        
        try:
            self.concretize_register(self._get_argument_register(index))
        except IndexError:
            len_args = len(self._archinfo.reg_args)
            addr = self.cpu.stack_pointer + self.ptr_size + ((index-len_args) * self.ptr_size)  
            self.concretize_memory_int(addr, self.ptr_size)                     

    def write_argument_value(self, i: int, val: int) -> None:
        
        try:
            return self.write_register(self._get_argument_register(i), val)
        except IndexError:
            len_args = len(self._archinfo.reg_args)
            return self.write_stack_value(i-len_args, val, offset=1)

    def get_argument_value(self, i: int) -> int:
        
        try:
            return self.read_register(self._get_argument_register(i))
        except IndexError:
            len_args = len(self._archinfo.reg_args)
            return self.get_stack_value(i-len_args, offset=1)

    def get_argument_symbolic(self, i: int) -> Expression:
        
        try:
            return self.read_symbolic_register(self._get_argument_register(i))
        except IndexError:
            len_args = len(self._archinfo.reg_args)
            addr = self.cpu.stack_pointer + ((i-len_args) * self.ptr_size)
            return self.read_symbolic_memory_int(addr, self.ptr_size)

    def get_full_argument(self, i: int) -> Tuple[int, Expression]:
        
        return self.get_argument_value(i), self.get_argument_symbolic(i)

    def get_string_argument(self, idx: int) -> str:
        
        return self.memory.read_string(self.get_argument_value(idx))

    def get_format_string(self, addr: Addr) -> str:
        
        return self.memory.read_string(addr)                                             \
               .replace("%s", "{}").replace("%d", "{}").replace("%
               .replace("%
               .replace("%c", "{:c}").replace("%02x", "{:02x}").replace("%ld", "{}")    \
               .replace("%*s", "").replace("%lX", "{:X}").replace("%08x", "{:08x}")     \
               .replace("%u", "{}").replace("%lu", "{}").replace("%zu", "{}")           \
               .replace("%02u", "{:02d}").replace("%03u", "{:03d}")                     \
               .replace("%03d", "{:03d}").replace("%p", "{:

    def get_format_arguments(self, fmt_addr: Addr, args: List[int]) -> List[Union[int, str]]:
        
        
        s_str = self.memory.read_string(fmt_addr)
        post_string = [i for i, x in enumerate([i for i, c in enumerate(s_str) if c == '%']) if s_str[x+1] == "s"]
        for p in post_string:
            args[p] = self.memory.read_string(args[p])
            args[p] = args[p].encode("latin-1").decode(errors='replace')
        return args

    def get_stack_value(self, index: int, offset: int = 0) -> int:
        
        addr = self.cpu.stack_pointer + (offset * self.ptr_size) + (index * self.ptr_size)
        return self.memory.read_uint(addr, self.ptr_size)

    def write_stack_value(self, index: int, value: int, offset: int = 0) -> None:
        
        addr = self.cpu.stack_pointer + (offset * self.ptr_size) + (index * self.ptr_size)
        self.memory.write_int(addr, value, self.ptr_size)

    def pop_stack_value(self) -> int:
        
        val = self.memory.read_ptr(self.cpu.stack_pointer)
        self.cpu.stack_pointer += self.ptr_size
        return val

    def push_stack_value(self, value: int) -> None:
        
        self.memory.write_ptr(self.cpu.stack_pointer-self.ptr_size, value)
        self.cpu.stack_pointer -= self.ptr_size

    def is_halt_instruction(self) -> bool:
        
        halt_opc = self._archinfo.halt_inst
        return self.__current_inst.getType() == halt_opc

    def solve(self, constraint: Union[AstNode, List[AstNode]], with_pp: bool = True) -> Tuple[SolverStatus, Model]:
        
        if with_pp:
            cst = constraint if isinstance(constraint, list) else [constraint]
            final_cst = self.actx.land([self.tt_ctx.getPathPredicate()]+cst)
        else:
            final_cst = self.actx.land(constraint) if isinstance(constraint, list) else constraint

        model, status, _ = self.tt_ctx.getModel(final_cst, status=True)
        return SolverStatus(status), model

    def solve_no_pp(self, constraint: Union[AstNode, List[AstNode]]) -> Tuple[SolverStatus, Model]:
        
        return self.solve(constraint, with_pp=False)

    def symbolize_register(self, register: Union[str, Register], alias: str = None) -> SymbolicVariable:
        
        reg = getattr(self.tt_ctx.registers, register) if isinstance(register, str) else register  
        if alias:
            var = self.tt_ctx.symbolizeRegister(reg, alias)
        else:
            var = self.tt_ctx.symbolizeRegister(reg)
        return var

    def symbolize_memory_byte(self, addr: Addr, alias: str = None) -> SymbolicVariable:
        
        if alias:
            return self.tt_ctx.symbolizeMemory(MemoryAccess(addr, CPUSIZE.BYTE), alias)
        else:
            return self.tt_ctx.symbolizeMemory(MemoryAccess(addr, CPUSIZE.BYTE))

    def symbolize_memory_bytes(self, addr: Addr, size: ByteSize, alias_prefix: str = None, offset: int = 0) -> List[SymbolicVariable]:
        
        if alias_prefix:
            return [self.symbolize_memory_byte(addr+i, alias_prefix+f"[{i+offset}]") for i in range(size)]
        else:
            return [self.symbolize_memory_byte(addr+i) for i in range(size)]

    def get_expression_variable_values_model(self, exp: Union[AstNode, Expression], model: Model) -> Dict[SymbolicVariable: int]:
        
        ast = exp.getAst() if hasattr(exp, "getAst") else exp
        ast_vars = self.actx.search(ast, AST_NODE.VARIABLE)
        sym_vars = [x.getSymbolicVariable() for x in ast_vars]
        final_dict = {}
        for avar, svar in zip(ast_vars, sym_vars):
            if svar.getId() in model:
                final_dict[svar] = model[svar.getId()].getValue()
            else:
                final_dict[svar] = avar.evaluate()
        return final_dict

    def evaluate_expression_model(self, exp: Union[AstNode, Expression], model: Model) -> int:
        
        ast = exp.getAst() if hasattr(exp, "getAst") else exp

        variables = self.get_expression_variable_values_model(ast, model)

        backup = {}
        for var, value in variables.items():
            backup[var] = self.tt_ctx.getConcreteVariableValue(var)
            self.tt_ctx.setConcreteVariableValue(var, value)
        final_value = ast.evaluate()
        for var in variables.keys():
            self.tt_ctx.setConcreteVariableValue(var, backup[var])
        return final_value

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def solve_enumerate_expression(self, exp: Union[AstNode, Expression], constraints: List[AstNode], values_blacklist: List[int], limit: int) -> List[Tuple[Model, int]]:
        
        ast = exp.getAst() if hasattr(exp, "getAst") else exp

        constraint = self.actx.land(constraints + [ast != x for x in values_blacklist])

        result = []
        while limit:
            status, model = self.solve(constraint, with_pp=False)
            if status == SolverStatus.SAT:
                new_val = self.evaluate_expression_model(ast, model)
                result.append((model, new_val))
                constraint = self.actx.land([constraint, ast != new_val])
            else:
                return result
            limit -= 1
        return result

    @staticmethod
    def from_loader(loader: Loader) -> 'ProcessState':
        pstate = ProcessState(loader.endianness)

        
        pstate.initialize_context(loader.architecture)

        
        pstate.cpu.program_counter = loader.entry_point

        
        with pstate.memory.without_segmentation():
            
            for i, seg in enumerate(loader.memory_segments()):
                if not seg.size and not seg.content:
                    logger.warning(f"A segment have to provide either a size or a content {seg.name} (skipped)")
                    continue
                size = len(seg.content) if seg.content else seg.size
                logger.debug(f"Loading 0x{seg.address:
                pstate.memory.map(seg.address, size, seg.perms, seg.name)
                if seg.content:
                    pstate.memory.write(seg.address, seg.content)

        
        cur_linkage_address = pstate.EXTERN_FUNC_BASE

        
        with pstate.memory.without_segmentation():
            
            for fname, rel_addr in loader.imported_functions_relocations():
                logger.debug(f"Hooking {fname} at {rel_addr:

                
                
                if fname in pstate.dynamic_symbol_table:
                    (linkage_address, _) = pstate.dynamic_symbol_table[fname]
                    logger.debug(f"Already added. {fname} at {rel_addr:
                    pstate.memory.write_ptr(rel_addr, linkage_address)

                else:
                    
                    pstate.dynamic_symbol_table[fname] = (cur_linkage_address, True)

                    
                    pstate.memory.write_ptr(rel_addr, cur_linkage_address)
                    
                    cur_linkage_address += pstate.ptr_size

        
        
        try:
            stack = pstate.memory.map_from_name(pstate.STACK_SEG)
            alloc = 1 * pstate.ptr_size
            pstate.write_register(pstate.base_pointer_register, stack.start+stack.size-alloc)   
            pstate.write_register(pstate.stack_pointer_register, stack.start+stack.size-alloc)
        except AssertionError:
            logger.warning("no stack segment has been created by the loader")

        
        segs = pstate.memory.find_map(pstate.EXTERN_SEG)
        if segs:
            symb_base = segs[0].start

            
            for sname, rel_addr in loader.imported_variable_symbols_relocations():
                logger.debug(f"Hooking {sname} at {rel_addr:

                if pstate.architecture == Architecture.X86_64:  
                    
                    pstate.dynamic_symbol_table[sname] = (rel_addr, False)
                    
                else:
                    
                    pstate.dynamic_symbol_table[sname] = (symb_base, False)
                    pstate.memory.write_ptr(rel_addr, symb_base)

                symb_base += pstate.ptr_size

        for reg_name in pstate.cpu:
            if reg_name in loader.cpustate:
                setattr(pstate.cpu, reg_name, loader.cpustate[reg_name])

        if loader.arch_mode:    
            if loader.arch_mode == ArchMode.THUMB:
                pstate.set_thumb(True)
        return pstate
