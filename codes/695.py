
from __future__ import annotations


from triton import Instruction


from tritondse.callbacks import CbType, ProbeInterface
from tritondse.seed import Seed, SeedStatus
from tritondse.types import Architecture, Addr, Tuple, SolverStatus
from tritondse import SymbolicExecutor, ProcessState
from tritondse.exception import ProbeException
import tritondse.logging

logger = tritondse.logging.get("sanitizers")


def mk_new_crashing_seed(se, model) -> Seed:
    
    new_input = bytearray(se.seed.content)
    for k, v in model.items():
        new_input[k] = v.getValue()
    
    
    
    return Seed(bytes(new_input))


class UAFSanitizer(ProbeInterface):
    
    def __init__(self):
        super(UAFSanitizer, self).__init__()
        self._add_callback(CbType.MEMORY_READ, self._memory_read)
        self._add_callback(CbType.MEMORY_WRITE, self._memory_write)
        self._add_callback(CbType.PRE_RTN, self._free_routine, 'free')

    @staticmethod
    def check(se: SymbolicExecutor, pstate: ProcessState, ptr: Addr, description: str = None) -> bool:
        
        if pstate.is_heap_ptr(ptr) and pstate.heap_allocator.is_ptr_freed(ptr):
            if description:
                logger.critical(description)
            se.seed.status = SeedStatus.CRASH
            pstate.stop = True
            return True
        return False

    @staticmethod
    def _memory_read(se, pstate, mem):
        return UAFSanitizer.check(se, pstate, mem.getAddress(), f'UAF detected at {mem}')

    @staticmethod
    def _memory_write(se, pstate, mem, value):
        return UAFSanitizer.check(se, pstate, mem.getAddress(), f'UAF detected at {mem}')

    @staticmethod
    def _free_routine(se, pstate, name, addr):
        ptr = se.pstate.get_argument_value(0)
        return UAFSanitizer.check(se, pstate, ptr, f'Double free detected at {addr:


class NullDerefSanitizer(ProbeInterface):
    
    def __init__(self):
        super(NullDerefSanitizer, self).__init__()
        self._add_callback(CbType.MEMORY_READ, self._memory_read)
        self._add_callback(CbType.MEMORY_WRITE, self._memory_write)

    @staticmethod
    def check(se: SymbolicExecutor, pstate: ProcessState, ptr: Addr, description: str = None) -> bool:
        

        
        if pstate.current_instruction is None:
            return False

        
        
        
        
        
        
        
        
        
        

        
        
        
        

        
        if ptr == 0 or (pstate.memory.segmentation_enabled and not pstate.memory.is_mapped(ptr)):
            if description:
                logger.critical(description)
            se.seed.status = SeedStatus.CRASH

            
            
            
            raise ProbeException(description)

        return False

    @staticmethod
    def _memory_read(se, pstate, mem):
        return NullDerefSanitizer.check(se, pstate, mem.getAddress(), f'Invalid memory access when reading at {mem} from {pstate.current_instruction}')

    @staticmethod
    def _memory_write(se, pstate, mem, value):
        return NullDerefSanitizer.check(se, pstate, mem.getAddress(), f'Invalid memory access when writting at {mem} from {pstate.current_instruction}')


class FormatStringSanitizer(ProbeInterface):
    
    def __init__(self):
        super(FormatStringSanitizer, self).__init__()
        self._add_callback(CbType.PRE_RTN,  self._xprintf_arg0_routine, 'printf')
        self._add_callback(CbType.PRE_RTN, self._xprintf_arg1_routine, 'fprintf')
        self._add_callback(CbType.PRE_RTN, self._xprintf_arg1_routine, 'sprintf')
        self._add_callback(CbType.PRE_RTN, self._xprintf_arg1_routine, 'dprintf')
        self._add_callback(CbType.PRE_RTN, self._xprintf_arg1_routine, 'snprintf')

    @staticmethod
    def check(se, pstate, fmt_ptr, extra_data: Tuple[str, Addr] = None):
        
        symbolic_cells = []

        
        cur_ptr = fmt_ptr
        while se.pstate.memory.read_uchar(cur_ptr):  
            if se.pstate.is_memory_symbolic(cur_ptr, 1):
                symbolic_cells.append(cur_ptr)
            cur_ptr += 1

        if symbolic_cells:
            extra = f"(function {extra_data[0]}@{extra_data[1]:
            logger.warning(f'Potential format string of length {len(symbolic_cells)} on {fmt_ptr:x} {extra}')
            se.seed.status = SeedStatus.OK_DONE
            pp_seeds = []
            nopp_seeds = []

            for i in range(int(len(symbolic_cells) / 2)):
                
                cell1 = pstate.read_symbolic_memory_byte(symbolic_cells.pop(0)).getAst()
                cell2 = pstate.read_symbolic_memory_byte(symbolic_cells.pop(0)).getAst()

                
                st, model = pstate.solve([cell1 == ord('%'), cell2 == ord('s')], with_pp=True)
                if st == SolverStatus.SAT and model:
                    pp_seeds.append(mk_new_crashing_seed(se, model))

                
                st, model = pstate.solve_no_pp([cell1 == ord('%'), cell2 == ord('s')])
                if st == SolverStatus.SAT and model:
                    pp_seeds.append(mk_new_crashing_seed(se, model))

            
            if pp_seeds:
                s = pp_seeds[-1]
                se.enqueue_seed(s)  
                logger.warning(f'Found model that might lead to a crash: {s.hash} (with path predicate)')
            if nopp_seeds:
                s = nopp_seeds[-1]
                se.enqueue_seed(s)  
                logger.warning(f'Found model that might lead to a crash: {s.hash} (without path predicate)')

            
            pstate.stop = False
            return True
        return False

    @staticmethod
    def _xprintf_arg0_routine(se, pstate, name, addr):
        string_ptr = se.pstate.get_argument_value(0)
        FormatStringSanitizer.check(se, pstate, string_ptr, (name, addr))

    @staticmethod
    def _xprintf_arg1_routine(se, pstate, name, addr):
        string_ptr = se.pstate.get_argument_value(1)
        FormatStringSanitizer.check(se, pstate, string_ptr, (name, addr))


class IntegerOverflowSanitizer(ProbeInterface):
    
    def __init__(self):
        super(IntegerOverflowSanitizer, self).__init__()
        self._add_callback(CbType.POST_INST, self.check)

    @staticmethod
    def check(se: SymbolicExecutor, pstate: ProcessState, instruction: Instruction) -> bool:
        
        
        assert (pstate.architecture == Architecture.X86_64 or pstate.architecture == Architecture.AARCH64)

        rf = (pstate.registers.of if pstate.architecture == Architecture.X86_64 else pstate.registers.v)

        if pstate.read_register(rf):
            logger.warning(f'Integer overflow at {instruction}')
            
            se.seed.status = SeedStatus.CRASH
            return True

        else:  
            if pstate.is_register_symbolic(rf):
                sym_flag = pstate.read_symbolic_register(rf)
                _, model = pstate.solve_no_pp(sym_flag.getAst() == 1)
                if model:
                    logger.warning(f'Potential integer overflow at {instruction}')
                    crash_seed = mk_new_crashing_seed(se, model)
                    se.enqueue_seed(crash_seed)
                    return True

        return False
