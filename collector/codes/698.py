from __future__ import annotations
from enum import Enum, auto
from typing import Callable, Tuple, List, Optional, Union, Any
import enum_tools.documentation


from triton import CALLBACK, Instruction, MemoryAccess, OPCODE


from tritondse.process_state import ProcessState
from tritondse.types import Addr, Register, Expression, Edge, SymExType, AstNode
from tritondse.thread_context import ThreadContext
from tritondse.seed import Seed
from tritondse.memory import MemoryAccessViolation
import tritondse.logging

logger = tritondse.logging.get("callback")


@enum_tools.documentation.document_enum
class CbPos(Enum):

    BEFORE = auto()
    AFTER = auto()


@enum_tools.documentation.document_enum
class CbType(Enum):

    CTX_SWITCH = auto()
    MEMORY_READ = auto()
    MEMORY_WRITE = auto()
    POST_RTN = auto()
    POST_ADDR = auto()
    POST_EXEC = auto()
    POST_INST = auto()
    PRE_ADDR = auto()
    PRE_EXEC = auto()
    PRE_INST = auto()
    PRE_RTN = auto()
    REG_READ = auto()
    REG_WRITE = auto()
    NEW_INPUT = auto()
    EXPLORE_STEP = auto()
    PRE_MNEM = auto()
    POST_MNEM = auto()
    PRE_OPCODE = auto()
    POST_OPCODE = auto()
    BRANCH_COV = auto()
    SYMEX_SOLVING = auto()
    MEM_VIOLATION = auto()


AddrCallback = Callable[["SymbolicExecutor", ProcessState, Addr], None]
ExplorationStepCallback = Callable[["SymbolicExplorator"], None]
InstrCallback = Callable[["SymbolicExecutor", ProcessState, Instruction], None]
MemReadCallback = Callable[["SymbolicExecutor", ProcessState, MemoryAccess], None]
MemWriteCallback = Callable[["SymbolicExecutor", ProcessState, MemoryAccess, int], None]
MnemonicCallback = Callable[["SymbolicExecutor", ProcessState, OPCODE], None]
SymExSolvingCallback = Callable[
    ["SymbolicExecutor", ProcessState, Edge, SymExType, AstNode, List[AstNode]], bool
]
BranchCoveredCallback = Callable[["SymbolicExecutor", ProcessState, Edge], bool]
NewInputCallback = Callable[["SymbolicExecutor", ProcessState, Seed], Optional[Seed]]
OpcodeCallback = Callable[["SymbolicExecutor", ProcessState, bytes], None]
RegReadCallback = Callable[["SymbolicExecutor", ProcessState, Register], None]
RegWriteCallback = Callable[["SymbolicExecutor", ProcessState, Register, int], None]
RtnCallback = Callable[
    ["SymbolicExecutor", ProcessState, str, Addr], Optional[Union[int, Expression]]
]
SymExCallback = Callable[["SymbolicExecutor", ProcessState], None]
ThreadCallback = Callable[["SymbolicExecutor", ProcessState, ThreadContext], None]
MemoryViolationCallback = Callable[
    ["SymbolicExecutor", ProcessState, MemoryAccessViolation], None
]


class ProbeInterface(object):

    def __init__(self):
        self._cbs: List[Tuple[CbType, Callable, Optional[str]]] = []

    @property
    def callbacks(self) -> List[Tuple[CbType, Callable, Optional[Any]]]:
        return self._cbs

    def _add_callback(self, typ: CbType, callback: Callable, arg: str = None):

        self._cbs.append((typ, callback, arg))


class CallbackManager(object):

    def __init__(self):
        self._se = None

        self._step_cbs = []

        self._pc_addr_cbs = {}
        self._opcode_cbs = {}
        self._mnemonic_cbs = {}
        self._instr_cbs = {CbPos.BEFORE: [], CbPos.AFTER: []}
        self._pre_exec = []
        self._post_exec = []
        self._ctx_switch = []
        self._new_input_cbs = []
        self._branch_solving_cbs = []
        self._branch_covered_cbs = []
        self._pre_rtn_cbs = {}
        self._post_rtn_cbs = {}
        self._mem_violation_cbs = []

        self._mem_read_cbs = []
        self._mem_write_cbs = []
        self._reg_read_cbs = []
        self._reg_write_cbs = []
        self._empty = True

        self._func_to_register = {}

    def is_empty(self) -> bool:

        return self._empty

    def is_binded(self) -> bool:

        return bool(self._se)

    def _trampoline_mem_read_cb(self, ctx, mem):

        if self._se.pstate.memory.callbacks_enabled():
            for cb in self._mem_read_cbs:
                cb(self._se, self._se.pstate, mem)

    def _trampoline_mem_write_cb(self, ctx, mem, value):

        if self._se.pstate.memory.callbacks_enabled():
            for cb in self._mem_write_cbs:
                cb(self._se, self._se.pstate, mem, value)

    def _trampoline_reg_read_cb(self, ctx, reg):

        for cb in self._reg_read_cbs:
            cb(self._se, self._se.pstate, reg)

    def _trampoline_reg_write_cb(self, ctx, reg, value):

        for cb in self._reg_write_cbs:
            cb(self._se, self._se.pstate, reg, value)

    def unbind(self) -> None:

        if self.is_binded():
            self._se.pstate.clear_triton_callbacks()
            self._se = None

    def bind_to(self, se: "SymbolicExecutor") -> None:

        if self.is_binded() and self._se != se:
            logger.warning(
                "Callback_manager already bound (on a different executor instance)"
            )

        self._se = se

        if self._mem_read_cbs:
            se.pstate.register_triton_callback(
                CALLBACK.GET_CONCRETE_MEMORY_VALUE, self._trampoline_mem_read_cb
            )

        if self._mem_write_cbs:
            se.pstate.register_triton_callback(
                CALLBACK.SET_CONCRETE_MEMORY_VALUE, self._trampoline_mem_write_cb
            )

        if self._reg_read_cbs:
            se.pstate.register_triton_callback(
                CALLBACK.GET_CONCRETE_REGISTER_VALUE, self._trampoline_reg_read_cb
            )

        if self._reg_write_cbs:
            se.pstate.register_triton_callback(
                CALLBACK.SET_CONCRETE_REGISTER_VALUE, self._trampoline_reg_write_cb
            )

        if self._func_to_register:
            if se.loader:
                for fname in list(self._func_to_register):
                    cbs = self._func_to_register.pop(fname)
                    addr = se.loader.find_function_addr(fname)
                    if addr:
                        for cb in cbs:
                            self.register_pre_addr_callback(addr, cb)
                    else:
                        logger.warning(f"can't find function '{fname}' in {se.loader}")
            else:
                logger.warning(f"function callback to resolve but no program provided")

    def register_addr_callback(
        self, pos: CbPos, addr: Addr, callback: AddrCallback
    ) -> None:

        if addr not in self._pc_addr_cbs:
            self._pc_addr_cbs[addr] = {CbPos.BEFORE: [], CbPos.AFTER: []}

        self._pc_addr_cbs[addr][pos].append(callback)
        self._empty = False

    def register_pre_addr_callback(self, addr: Addr, callback: AddrCallback) -> None:

        self.register_addr_callback(CbPos.BEFORE, addr, callback)

    def register_post_addr_callback(self, addr: Addr, callback: AddrCallback) -> None:

        self.register_addr_callback(CbPos.AFTER, addr, callback)

    def get_address_callbacks(
        self, addr: Addr
    ) -> Tuple[List[AddrCallback], List[AddrCallback]]:

        cbs = self._pc_addr_cbs.get(addr, None)
        if cbs is not None:
            return cbs[CbPos.BEFORE], cbs[CbPos.AFTER]
        else:
            return [], []

    def register_opcode_callback(
        self, pos: CbPos, opcode: bytes, callback: OpcodeCallback
    ) -> None:

        if opcode not in self._opcode_cbs:
            self._opcode_cbs[opcode] = {CbPos.BEFORE: [], CbPos.AFTER: []}

        self._opcode_cbs[opcode][pos].append(callback)
        self._empty = False

    def register_pre_opcode_callback(
        self, opcode: bytes, callback: OpcodeCallback
    ) -> None:

        self.register_opcode_callback(CbPos.BEFORE, opcode, callback)

    def register_post_opcode_callback(
        self, opcode: bytes, callback: OpcodeCallback
    ) -> None:

        self.register_opcode_callback(CbPos.AFTER, opcode, callback)

    def get_opcode_callbacks(
        self, opcode: bytes
    ) -> Tuple[List[OpcodeCallback], List[OpcodeCallback]]:

        cbs = self._opcode_cbs.get(opcode, None)
        if cbs:
            return cbs[CbPos.BEFORE], cbs[CbPos.AFTER]
        else:
            return [], []

    def register_mnemonic_callback(
        self, pos: CbPos, mnemonic: OPCODE, callback: MnemonicCallback
    ) -> None:

        if mnemonic not in self._mnemonic_cbs:
            self._mnemonic_cbs[mnemonic] = {CbPos.BEFORE: [], CbPos.AFTER: []}

        self._mnemonic_cbs[mnemonic][pos].append(callback)
        self._empty = False

    def register_pre_mnemonic_callback(
        self, mnemonic: OPCODE, callback: MnemonicCallback
    ) -> None:

        self.register_mnemonic_callback(CbPos.BEFORE, mnemonic, callback)

    def register_post_mnemonic_callback(
        self, mnemonic: OPCODE, callback: MnemonicCallback
    ) -> None:

        self.register_mnemonic_callback(CbPos.AFTER, mnemonic, callback)

    def get_mnemonic_callbacks(
        self, mnemonic: OPCODE
    ) -> Tuple[List[MnemonicCallback], List[MnemonicCallback]]:

        cbs = self._mnemonic_cbs.get(mnemonic, None)
        if cbs:
            return cbs[CbPos.BEFORE], cbs[CbPos.AFTER]
        else:
            return [], []

    def register_function_callback(
        self, func_name: str, callback: AddrCallback
    ) -> None:

        if func_name in self._func_to_register:
            self._func_to_register[func_name].append(callback)
        else:
            self._func_to_register[func_name] = [callback]

    def register_instruction_callback(
        self, pos: CbPos, callback: InstrCallback
    ) -> None:

        self._instr_cbs[pos].append(callback)
        self._empty = False

    def register_pre_instruction_callback(self, callback: InstrCallback) -> None:

        self.register_instruction_callback(CbPos.BEFORE, callback)

    def register_post_instruction_callback(self, callback: InstrCallback) -> None:

        self.register_instruction_callback(CbPos.AFTER, callback)

    def get_instruction_callbacks(
        self,
    ) -> Tuple[List[InstrCallback], List[InstrCallback]]:

        return self._instr_cbs[CbPos.BEFORE], self._instr_cbs[CbPos.AFTER]

    def register_pre_execution_callback(self, callback: SymExCallback) -> None:

        self._pre_exec.append(callback)
        self._empty = False

    def register_post_execution_callback(self, callback: SymExCallback) -> None:

        self._post_exec.append(callback)
        self._empty = False

    def register_exploration_step_callback(
        self, callback: ExplorationStepCallback
    ) -> None:

        self._step_cbs.append(callback)

    def get_execution_callbacks(
        self,
    ) -> Tuple[List[SymExCallback], List[SymExCallback]]:

        return self._pre_exec, self._post_exec

    def register_memory_read_callback(self, callback: MemReadCallback) -> None:

        self._mem_read_cbs.append(callback)
        self._empty = False

    def register_memory_write_callback(self, callback: MemWriteCallback) -> None:

        self._mem_write_cbs.append(callback)
        self._empty = False

    def register_register_read_callback(self, callback: RegReadCallback) -> None:

        self._reg_read_cbs.append(callback)
        self._empty = False

    def register_register_write_callback(self, callback: RegWriteCallback) -> None:

        self._reg_write_cbs.append(callback)
        self._empty = False

    def register_thread_context_switch_callback(self, callback: ThreadCallback) -> None:

        self._ctx_switch.append(callback)
        self._empty = False

    def get_context_switch_callback(self) -> List[ThreadCallback]:

        return self._ctx_switch

    def register_new_input_callback(self, callback: NewInputCallback) -> None:

        self._new_input_cbs.append(callback)
        self._empty = False

    def get_new_input_callback(self) -> List[NewInputCallback]:

        return self._new_input_cbs

    def register_on_solving_callback(self, callback: SymExSolvingCallback) -> None:

        self._branch_solving_cbs.append(callback)
        self._empty = False

    def get_on_solving_callback(self) -> List[SymExSolvingCallback]:

        return self._branch_solving_cbs

    def register_on_branch_covered_callback(
        self, callback: BranchCoveredCallback
    ) -> None:

        self._branch_covered_cbs.append(callback)
        self._empty = False

    def get_on_branch_covered_callback(self) -> List[BranchCoveredCallback]:

        return self._branch_covered_cbs

    def register_memory_violation_callback(
        self, callback: MemoryViolationCallback
    ) -> None:

        self._mem_violation_cbs.append(callback)
        self._empty = False

    def get_memory_violation_callbacks(self) -> List[MemoryViolationCallback]:

        return self._mem_violation_cbs

    def get_exploration_step_callbacks(self) -> List[ExplorationStepCallback]:

        return self._step_cbs

    def register_pre_imported_routine_callback(
        self, routine_name: str, callback: RtnCallback
    ) -> None:

        if routine_name in self._pre_rtn_cbs:
            self._pre_rtn_cbs[routine_name].append(callback)
        else:
            self._pre_rtn_cbs[routine_name] = [callback]
        self._empty = False

    def register_post_imported_routine_callback(
        self, routine_name: str, callback: RtnCallback
    ) -> None:

        if routine_name in self._post_rtn_cbs:
            self._post_rtn_cbs[routine_name].append(callback)
        else:
            self._post_rtn_cbs[routine_name] = [callback]
        self._empty = False

    def get_imported_routine_callbacks(
        self, routine_name: str
    ) -> Tuple[List[RtnCallback], List[RtnCallback]]:

        pre_ret = (
            self._pre_rtn_cbs[routine_name] if routine_name in self._pre_rtn_cbs else []
        )
        post_ret = (
            self._post_rtn_cbs[routine_name]
            if routine_name in self._post_rtn_cbs
            else []
        )
        return pre_ret, post_ret

    def register_probe(self, probe: ProbeInterface) -> None:

        for kind, cb, arg in probe.callbacks:
            try:
                mapping_with_args = {
                    CbType.PRE_RTN: self.register_pre_imported_routine_callback,
                    CbType.POST_RTN: self.register_post_imported_routine_callback,
                    CbType.PRE_ADDR: self.register_pre_addr_callback,
                    CbType.POST_ADDR: self.register_post_addr_callback,
                    CbType.PRE_MNEM: self.register_pre_mnemonic_callback,
                    CbType.POST_MNEM: self.register_post_mnemonic_callback,
                    CbType.PRE_OPCODE: self.register_pre_opcode_callback,
                    CbType.POST_OPCODE: self.register_post_opcode_callback,
                }
                mapping_with_args[kind](arg, cb)
            except KeyError:
                mapping = {
                    CbType.CTX_SWITCH: self.register_thread_context_switch_callback,
                    CbType.MEMORY_READ: self.register_memory_read_callback,
                    CbType.MEMORY_WRITE: self.register_memory_write_callback,
                    CbType.POST_EXEC: self.register_post_execution_callback,
                    CbType.POST_INST: self.register_post_instruction_callback,
                    CbType.PRE_EXEC: self.register_pre_execution_callback,
                    CbType.PRE_INST: self.register_pre_instruction_callback,
                    CbType.REG_READ: self.register_register_read_callback,
                    CbType.REG_WRITE: self.register_register_write_callback,
                    CbType.NEW_INPUT: self.register_new_input_callback,
                    CbType.EXPLORE_STEP: self.register_exploration_step_callback,
                    CbType.BRANCH_COV: self.register_on_branch_covered_callback,
                    CbType.SYMEX_SOLVING: self.register_on_solving_callback,
                    CbType.MEM_VIOLATION: self.register_memory_violation_callback,
                }
                mapping[kind](cb)

    def fork(self) -> "CallbackManager":

        cbs = CallbackManager()

        cbs._pc_addr_cbs = self._pc_addr_cbs.copy()
        cbs._opcode_cbs = self._opcode_cbs.copy()
        cbs._mnemonic_cbs = self._mnemonic_cbs.copy()
        cbs._instr_cbs = self._instr_cbs.copy()
        cbs._pre_exec = self._pre_exec.copy()
        cbs._post_exec = self._post_exec.copy()
        cbs._ctx_switch = self._ctx_switch.copy()
        cbs._new_input_cbs = self._new_input_cbs.copy()
        cbs._branch_solving_cbs = self._branch_solving_cbs.copy()
        cbs._branch_covered_cbs = self._branch_covered_cbs.copy()
        cbs._pre_rtn_cbs = self._pre_rtn_cbs.copy()
        cbs._post_rtn_cbs = self._post_rtn_cbs.copy()
        cbs._mem_violation_cbs = self._mem_violation_cbs.copy()

        cbs._mem_read_cbs = self._mem_read_cbs.copy()
        cbs._mem_write_cbs = self._mem_write_cbs.copy()
        cbs._reg_read_cbs = self._reg_read_cbs.copy()
        cbs._reg_write_cbs = self._reg_write_cbs.copy()
        cbs._empty = self._empty

        cbs._func_to_register = self._func_to_register.copy()

        return cbs

    def unregister_callback(self, callback: Callable) -> None:

        for addr, itms in self._pc_addr_cbs.items():
            for loc in CbPos:
                if callback in itms[loc]:
                    itms[loc].remove(callback)

        for opcode, itms in self._opcode_cbs.items():
            for loc in CbPos:
                if callback in itms[loc]:
                    itms[loc].remove(callback)

        for mnemonic, itms in self._mnemonic_cbs.items():
            for loc in CbPos:
                if callback in itms[loc]:
                    itms[loc].remove(callback)

        for loc in CbPos:
            if callback in self._instr_cbs[loc]:
                self._instr_cbs[loc].remove(callback)

        for cb_list in [
            self._step_cbs,
            self._pre_exec,
            self._post_exec,
            self._ctx_switch,
            self._new_input_cbs,
            self._branch_solving_cbs,
            self._branch_covered_cbs,
            self._mem_read_cbs,
            self._mem_write_cbs,
            self._reg_read_cbs,
            self._reg_write_cbs,
            self._mem_violation_cbs,
        ]:
            if callback in cb_list:
                cb_list.remove(callback)

        for d in [self._pre_rtn_cbs, self._post_rtn_cbs]:
            for cb_list in d.values():
                if callback in cb_list:
                    cb_list.remove(callback)

    def reset(self) -> None:

        self._step_cbs = []

        self._pc_addr_cbs = {}
        self._opcode_cbs = {}
        self._mnemonic_cbs = {}
        self._instr_cbs = {CbPos.BEFORE: [], CbPos.AFTER: []}
        self._pre_exec = []
        self._post_exec = []
        self._ctx_switch = []
        self._new_input_cbs = []
        self._branch_solving_cbs = []
        self._branch_covered_cbs = []
        self._pre_rtn_cbs = {}
        self._post_rtn_cbs = {}
        self._mem_violation_cbs = []

        self._mem_read_cbs = []
        self._mem_write_cbs = []
        self._reg_read_cbs = []
        self._reg_write_cbs = []
        self._empty = True
