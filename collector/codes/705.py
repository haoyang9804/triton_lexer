from __future__ import annotations
from pathlib import Path
from typing import Optional, Generator, Tuple, Dict, List
from dataclasses import dataclass


from tritondse.types import Addr, Architecture, Platform, ArchMode, Perm, Endian
from tritondse.arch import ARCHS
import tritondse.logging

logger = tritondse.logging.get()


@dataclass
class LoadableSegment:

    address: int

    size: int = 0

    perms: Perm = Perm.R | Perm.W | Perm.X

    content: Optional[bytes] = None

    name: str = ""


class Loader(object):

    def __init__(self, path: str):
        self.bin_path = Path(path)

    @property
    def name(self) -> str:

        raise NotImplementedError()

    @property
    def entry_point(self) -> Addr:

        raise NotImplementedError()

    @property
    def architecture(self) -> Architecture:

        raise NotImplementedError()

    @property
    def arch_mode(self) -> Optional[ArchMode]:

        return None

    @property
    def platform(self) -> Optional[Platform]:

        return None

    @property
    def endianness(self) -> Endian:

        raise NotImplementedError()

    def memory_segments(self) -> Generator[LoadableSegment, None, None]:

        raise NotImplementedError()

    @property
    def cpustate(self) -> Dict[str, int]:

        return {}

    def imported_functions_relocations(self) -> Generator[Tuple[str, Addr], None, None]:

        yield from ()

    def imported_variable_symbols_relocations(
        self,
    ) -> Generator[Tuple[str, Addr], None, None]:

        yield from ()

    def find_function_addr(self, name: str) -> Optional[Addr]:

        return None


class RawBinaryLoader(Loader):

    def __init__(
        self,
        architecture: Architecture,
        cpustate: Dict[str, int] = None,
        maps: List[LoadableSegment] = None,
        set_thumb: bool = False,
        platform: Platform = None,
        endianness: Endian = Endian.LITTLE,
    ):
        super(RawBinaryLoader, self).__init__("")

        self._architecture = architecture
        self._platform = platform if platform else None
        self._cpustate = cpustate if cpustate else {}
        self.maps = maps
        self._arch_mode = ArchMode.THUMB if set_thumb else None
        self._endian = endianness
        if self._platform and (self._architecture, self._platform) in ARCHS:
            self._archinfo = ARCHS[(self._architecture, self._platform)]
        elif self._architecture in ARCHS:
            self._archinfo = ARCHS[self._architecture]
        else:
            logger.error("Unknown architecture")
            assert False

    @property
    def name(self) -> str:

        return f"Monolithic({self.bin_path})"

    @property
    def architecture(self) -> Architecture:

        return self._architecture

    @property
    def arch_mode(self) -> ArchMode:

        return self._arch_mode

    @property
    def entry_point(self) -> Addr:

        return self.cpustate[self._archinfo.pc_reg]

    def memory_segments(self) -> Generator[LoadableSegment, None, None]:

        yield from self.maps

    @property
    def cpustate(self) -> Dict[str, int]:

        return self._cpustate

    @property
    def platform(self) -> Optional[Platform]:

        return self._platform

    @property
    def endianness(self) -> Endian:

        return self._endian
