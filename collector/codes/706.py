from __future__ import annotations
from pathlib import Path
from typing import Optional, Generator, Tuple


import lief


from tritondse.types import (
    PathLike,
    Addr,
    Architecture,
    Platform,
    ArchMode,
    Perm,
    Endian,
    Format,
)
from tritondse.loaders.loader import Loader, LoadableSegment
import tritondse.logging

logger = tritondse.logging.get("loader")


class Program(Loader):

    EXTERN_SYM_BASE = 0x0F001000
    EXTERN_SYM_SIZE = 0x1000

    BASE_STACK = 0xF0000000
    END_STACK = 0x70000000

    def __init__(self, path: PathLike):

        super(Program, self).__init__(path)

        self._arch_mapper = {
            lief.Header.ARCHITECTURES.ARM: Architecture.ARM32,
            lief.Header.ARCHITECTURES.ARM64: Architecture.AARCH64,
            lief.Header.ARCHITECTURES.X86: Architecture.X86,
            lief.Header.ARCHITECTURES.X86_64: Architecture.X86_64,
        }

        self._plfm_mapper = {
            lief.Binary.FORMATS.ELF: Platform.LINUX,
            lief.Binary.FORMATS.PE: Platform.WINDOWS,
            lief.Binary.FORMATS.MACHO: Platform.MACOS,
        }

        self._format_mapper = {
            lief.Binary.FORMATS.ELF: Format.ELF,
            lief.Binary.FORMATS.PE: Format.PE,
            lief.Binary.FORMATS.MACHO: Format.MACHO,
        }

        self.path: Path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"file {path} not found (or not a file)")

        self._binary = lief.parse(str(self.path))
        if self._binary is None:
            raise FileNotFoundError(f"file {path} not recognised by lief")

        self._arch = self._load_arch()
        if self._arch is None:
            raise FileNotFoundError(
                f"binary {path} architecture unsupported {self._binary.abstract.header.architecture}"
            )

        try:
            self._plfm = self._plfm_mapper[self._binary.format]

        except KeyError:
            self._plfm = None

        self._funs = {f.name: f for f in self._binary.concrete.functions}

    @property
    def name(self) -> str:

        return f"Program({self.path})"

    @property
    def endianness(self) -> Endian:
        return {
            lief.Header.ENDIANNESS.LITTLE: Endian.LITTLE,
            lief.Header.ENDIANNESS.BIG: Endian.BIG,
        }[self._binary.abstract.header.endianness]

    @property
    def entry_point(self) -> Addr:

        return self._binary.entrypoint

    @property
    def architecture(self) -> Architecture:

        return self._arch

    @property
    def platform(self) -> Optional[Platform]:

        return self._plfm

    @property
    def format(self) -> Format:

        return self._format_mapper[self._binary.format]

    def _load_arch(self) -> Optional[Architecture]:

        arch = self._binary.abstract.header.architecture
        if arch in self._arch_mapper:
            arch = self._arch_mapper[arch]
            if arch == Architecture.X86:
                arch = (
                    Architecture.X86
                    if self._binary.abstract.header.is_32
                    else Architecture.X86_64
                )
            return arch
        else:
            return None

    @property
    def relocation_enum(self):

        arch_mapper = {
            lief._lief.ELF.ARCH.AARCH64: "AARCH64",
            lief._lief.ELF.ARCH.ARM: "ARM",
            lief._lief.ELF.ARCH.I386: "I386",
            lief._lief.ELF.ARCH.X86_64: "X86_64",
        }

        arch_str = arch_mapper[self._binary.concrete.header.machine_type]

        rel_enum = {}
        for attr_str in dir(lief.ELF.Relocation.TYPE):
            if attr_str.startswith(arch_str):
                attr_name = attr_str[len(arch_str) + 1 :]
                rel_enum[attr_name] = getattr(lief.ELF.Relocation.TYPE, attr_str)

        return rel_enum

    def _is_glob_dat(self, rel: lief.ELF.Relocation) -> bool:

        rel_enum = self.relocation_enum

        if "GLOB_DAT" in rel_enum:
            return rel.type == rel_enum["GLOB_DAT"]
        else:
            return False

    def memory_segments(self) -> Generator[LoadableSegment, None, None]:

        if self.format == Format.ELF:
            for i, seg in enumerate(self._binary.concrete.segments):
                if seg.type == lief.ELF.Segment.TYPE.LOAD:
                    content = bytearray(seg.content)
                    if seg.virtual_size != len(seg.content):
                        content += bytearray([0]) * (
                            seg.virtual_size - seg.physical_size
                        )
                    yield LoadableSegment(
                        seg.virtual_address,
                        perms=Perm(int(seg.flags)),
                        content=bytes(content),
                        name=f"seg{i}",
                    )
        else:
            raise NotImplementedError(
                f"memory segments not implemented for: {self.format.name}"
            )

        yield LoadableSegment(
            self.EXTERN_SYM_BASE, self.EXTERN_SYM_SIZE, Perm.R | Perm.W, name="[extern]"
        )
        yield LoadableSegment(
            self.END_STACK,
            self.BASE_STACK - self.END_STACK,
            Perm.R | Perm.W,
            name="[stack]",
        )

    def imported_functions_relocations(self) -> Generator[Tuple[str, Addr], None, None]:

        if self.format == Format.ELF:
            try:

                for rel in self._binary.concrete.pltgot_relocations:
                    yield rel.symbol.name, rel.address

                for rel in self._binary.dynamic_relocations:
                    if (
                        self._is_glob_dat(rel)
                        and rel.has_symbol
                        and not rel.symbol.is_variable
                    ):
                        yield rel.symbol.name, rel.address
            except Exception:
                logger.error("Something wrong with the pltgot relocations")

        else:
            raise NotImplementedError(
                f"Imported functions relocations not implemented for: {self.format.name}"
            )

    def imported_variable_symbols_relocations(
        self,
    ) -> Generator[Tuple[str, Addr], None, None]:

        if self.format == Format.ELF:
            rel_enum = self.relocation_enum

            for rel in self._binary.dynamic_relocations:
                if rel.has_symbol:

                    if rel.symbol.is_variable:
                        yield rel.symbol.name, rel.address
        else:
            raise NotImplementedError(
                f"Imported symbols relocations not implemented for: {self.format.name}"
            )

    def find_function_addr(self, name: str) -> Optional[Addr]:

        f = self._funs.get(name)
        return f.address if f else None

    @property
    def arch_mode(self) -> ArchMode:
        pass
