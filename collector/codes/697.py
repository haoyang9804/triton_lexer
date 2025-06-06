import platform
from collections import namedtuple


from triton import OPCODE, TritonContext


from tritondse.types import Architecture

Arch = namedtuple(
    "Arch", "ret_reg pc_reg bp_reg sp_reg sys_reg reg_args halt_inst syscall_inst"
)

ARCHS = {
    Architecture.X86: Arch(
        "eax",
        "eip",
        "ebp",
        "esp",
        "eax",
        [],
        OPCODE.X86.HLT,
        [OPCODE.X86.SYSCALL, OPCODE.X86.SYSENTER],
    ),
    Architecture.X86_64: Arch(
        "rax",
        "rip",
        "rbp",
        "rsp",
        "rax",
        ["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
        OPCODE.X86.HLT,
        [OPCODE.X86.SYSCALL, OPCODE.X86.SYSENTER],
    ),
    Architecture.AARCH64: Arch(
        "x0",
        "pc",
        "sp",
        "sp",
        "x8",
        ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"],
        OPCODE.AARCH64.HLT,
        [OPCODE.AARCH64.SVC],
    ),
    Architecture.ARM32: Arch(
        "r0",
        "pc",
        "r11",
        "sp",
        "r7",
        ["r0", "r1", "r2", "r3"],
        OPCODE.ARM32.HLT,
        [OPCODE.ARM32.SVC],
    ),
}


class CpuState(dict):

    def __init__(self, ctx: TritonContext, arch_info: Arch):
        super(CpuState, self).__init__()
        self.__ctx = ctx
        self.__archinfo = arch_info
        for r in ctx.getAllRegisters():
            self[r.getName()] = r

    def __getattr__(self, name: str):

        if name in self:
            return self.__ctx.getConcreteRegisterValue(self[name])
        else:
            super().__getattr__(name)

    def __setattr__(self, name: str, value: int):

        if name in self:
            self.__ctx.setConcreteRegisterValue(self[name], value)
        else:
            super().__setattr__(name, value)

    @property
    def program_counter(self) -> int:

        return getattr(self, self.__archinfo.pc_reg)

    @program_counter.setter
    def program_counter(self, value: int) -> None:

        setattr(self, self.__archinfo.pc_reg, value)

    @property
    def base_pointer(self) -> int:

        return getattr(self, self.__archinfo.bp_reg)

    @base_pointer.setter
    def base_pointer(self, value: int) -> None:

        setattr(self, self.__archinfo.bp_reg, value)

    @property
    def stack_pointer(self) -> int:

        return getattr(self, self.__archinfo.sp_reg)

    @stack_pointer.setter
    def stack_pointer(self, value: int) -> None:

        setattr(self, self.__archinfo.sp_reg, value)


def local_architecture() -> Architecture:

    arch_m = {
        "i386": Architecture.X86,
        "x86_64": Architecture.X86_64,
        "armv7l": Architecture.ARM32,
        "aarch64": Architecture.AARCH64,
    }
    return arch_m[platform.machine()]
