
from typing import Generator, Optional, Tuple
from pathlib import Path
import logging


import cle


from tritondse.loaders import Loader, LoadableSegment
from tritondse.types import Addr, Architecture, PathLike, Platform, Perm, Endian
from tritondse.routines import SUPPORTED_ROUTINES
import tritondse.logging

logger = tritondse.logging.get("loader")

_arch_mapper = {
    "ARMEL":   Architecture.ARM32,
    "AARCH64": Architecture.AARCH64,
    "AMD64":   Architecture.X86_64,
    "X86":   Architecture.X86,
}

_plfm_mapper = {
    "UNIX - Linux": Platform.LINUX,
    "UNIX - System V": Platform.LINUX,
    "windows": Platform.WINDOWS,
    "macos": Platform.MACOS
}


class CleLoader(Loader):
    EXTERN_SYM_BASE = 0x0f001000
    EXTERN_SYM_SIZE = 0x1000

    BASE_STACK = 0xf0000000
    END_STACK = 0x70000000  

    def __init__(self, path: PathLike, ld_path: Optional[PathLike] = None):
        super(CleLoader, self).__init__(path)
        self.path: Path = Path(path)  
        if not self.path.is_file():
            raise FileNotFoundError(f"file {path} not found (or not a file)")

        self._disable_vex_loggers()  

        self.ld_path = ld_path if ld_path is not None else ()
        self.ld = cle.Loader(str(path), ld_path=self.ld_path)

    def _disable_vex_loggers(self):
        for name, logger in logging.root.manager.loggerDict.items():
            if "pyvex" in name:
                logger.propagate = False

    @property
    def name(self) -> str:
        
        return f"CleLoader({self.path})"

    @property
    def architecture(self) -> Architecture:
        
        return _arch_mapper[self.ld.main_object.arch.name]

    @property
    def endianness(self) -> Endian:
        
        return Endian.LITTLE

    @property
    def entry_point(self) -> Addr:
        
        return self.ld.main_object.entry

    def memory_segments(self) -> Generator[LoadableSegment, None, None]:
        
        for obj in self.ld.all_objects:
            logger.debug(obj)
            for seg in obj.segments:
                segdata = self.ld.memory.load(seg.vaddr, seg.memsize)
                assert len(segdata) == seg.memsize
                perms = (Perm.R if seg.is_readable else 0) | (Perm.W if seg.is_writable else 0) | (Perm.X if seg.is_executable else 0)
                if seg.__class__.__name__ != "ExternSegment":
                    
                    logger.debug(f"Loading segment {seg} - perms:{perms}")
                yield LoadableSegment(seg.vaddr, perms, content=segdata, name=f"seg-{obj.binary_basename}")
        
        yield LoadableSegment(self.EXTERN_SYM_BASE, self.EXTERN_SYM_SIZE, Perm.R | Perm.W, name="[extern]")
        yield LoadableSegment(self.END_STACK, self.BASE_STACK-self.END_STACK+1, Perm.R | Perm.W, name="[stack]")

        
        yield LoadableSegment(0, 0x2000, Perm.R | Perm.W, name="[fs]")

    
    @property
    def cpustate(self):
        
        
        return {"fs": 0x1000}

    @property
    def platform(self) -> Optional[Platform]:
        
        return _plfm_mapper[self.ld.main_object.os]

    def imported_functions_relocations(self) -> Generator[Tuple[str, Addr], None, None]:
        
        
        
        for obj in self.ld.all_objects:
            for fun in obj.imports:
                if fun in SUPPORTED_ROUTINES:
                    reloc = obj.imports[fun]
                    got_entry_addr = reloc.relative_addr + obj.mapped_base
                    yield fun, got_entry_addr

        
        
        

        
        
        
        
        for obj in self.ld.all_objects:
            for (resolver_func, got_rva) in obj.irelatives:
                got_slot = got_rva + obj.mapped_base
                sym = self.ld.find_symbol(resolver_func)
                if sym is None:
                    continue
                fun = sym.name
                if fun in SUPPORTED_ROUTINES:
                    yield fun, got_slot

    def imported_variable_symbols_relocations(self) -> Generator[Tuple[str, Addr], None, None]:
        
        
        for s in self.ld.main_object.symbols:
            if s.resolved and s._type == cle.SymbolType.TYPE_OBJECT:
                logger.debug(f"CleLoader: hooking symbol {s.name} @ {s.relative_addr:
                s_addr = s.relative_addr + self.ld.main_object.mapped_base
                yield s.name, s_addr

    def find_function_addr(self, name: str) -> Optional[Addr]:
        
        res = [x for x in self.ld.find_all_symbols(name) if x.is_function]
        return res[0].rebased_addr if res else None  
