
import bisect
from typing import Optional, Union, Generator, List
from collections import namedtuple
import struct
from contextlib import contextmanager


from triton import TritonContext


from tritondse.types import Perm, Addr, ByteSize, Endian

MemMap = namedtuple('Map', "start size perm name")


class MapOverlapException(Exception):
    
    pass


class MemoryAccessViolation(Exception):
    
    def __init__(self, addr: Addr, access: Perm, map_perm: Perm = None, memory_not_mapped: bool = False, perm_error: bool = False):
        
        super(MemoryAccessViolation, self).__init__()
        self.address: Addr = addr
        
        self._is_mem_unmapped = memory_not_mapped
        self._is_perm_error = perm_error
        self.access: Perm = access
        
        self.map_perm: Optional[Perm] = map_perm
        

    def is_permission_error(self) -> bool:
        
        return self._is_perm_error

    def is_memory_unmapped_error(self) -> bool:
        
        return self._is_mem_unmapped

    def __str__(self) -> str:
        if self.is_permission_error():
            return f"(addr:{self.address:
        else:
            return f"({str(self.access)}: {self.address:

    def __repr__(self):
        return str(self)


STRUCT_MAP = {
    (True, 1): 'B',
    (False, 1): 'b',
    (True, 2): 'H',
    (False, 2): 'h',
    (True, 4): 'I',
    (False, 4): 'i',
    (True, 8): 'Q',
    (False, 8): 'q'
}

ENDIAN_MAP = {
    Endian.LITTLE: "<",
    Endian.BIG: ">"
}


class Memory(object):
    

    def __init__(self, ctx: TritonContext, endianness: Endian = Endian.LITTLE):
        
        self.ctx: TritonContext = ctx
        
        self._linear_map_addr = []  
        self._linear_map_map = []   
        self._segment_enabled = True
        self._endian = endianness
        self._endian_key = ENDIAN_MAP[self._endian]
        self._mem_cbs_enabled = True
        

    def set_endianness(self, en: Endian) -> None:
        
        self._endian = en
        self._endian_key = ENDIAN_MAP[self._endian]

    @property
    def _ptr_size(self) -> int:
        return self.ctx.getGprSize()

    @property
    def segmentation_enabled(self) -> bool:
        
        return self._segment_enabled

    def disable_segmentation(self) -> None:
        
        self._segment_enabled = False

    def enable_segmentation(self) -> None:
        
        self._segment_enabled = True

    def set_segmentation(self, enabled: bool) -> None:
        
        self._segment_enabled = enabled

    @contextmanager
    def without_segmentation(self, disable_callbacks=False) -> Generator['Memory', None, None]:
        
        previous = self._segment_enabled
        self.disable_segmentation()
        cbs = self._mem_cbs_enabled
        self._mem_cbs_enabled = not disable_callbacks
        yield self
        self._mem_cbs_enabled = cbs
        self.set_segmentation(previous)

    def callbacks_enabled(self) -> bool:
        
        return self._mem_cbs_enabled

    def get_maps(self) -> Generator[MemMap, None, None]:
        
        yield from (x for x in self._linear_map_map if x)

    def map(self, start, size, perm: Perm = Perm.R | Perm.W | Perm.X, name="") -> MemMap:
        
        def _map_idx(index):
            self._linear_map_addr.insert(index, start + size - 1)  
            self._linear_map_addr.insert(index, start)
            self._linear_map_map.insert(index, None)
            memmap = MemMap(start, size, perm, name)
            self._linear_map_map.insert(index, memmap)
            return memmap

        if not self._linear_map_addr:  
            return _map_idx(0)

        idx = bisect.bisect_left(self._linear_map_addr, start)

        if idx == len(self._linear_map_addr):  
            return _map_idx(idx)

        addr = self._linear_map_addr[idx]
        if (idx % 2) == 0:  
            if start < addr and start+size <= addr:  
                return _map_idx(idx)
            else:  
                raise MapOverlapException(f"0x{start:08x}:{size} overlap with map: 0x{addr:08x} (even)")
        else:  
            prev = self._linear_map_addr[idx-1]
            raise MapOverlapException(f"0x{start:08x}:{size} overlap with map: 0x{prev:08x} (odd)")

    def unmap(self, addr: Addr) -> None:
        
        def _unmap_idx(index):
            self._linear_map_addr.pop(index)  
            self._linear_map_addr.pop(index)  
            self._linear_map_map.pop(index)   
            self._linear_map_map.pop(index)   

        idx = bisect.bisect_left(self._linear_map_addr, addr)
        try:
            mapaddr = self._linear_map_addr[idx]
            if (idx % 2) == 0:  
                if addr == mapaddr:  
                    _unmap_idx(idx)
                else:
                    raise MemoryAccessViolation(addr, Perm(0), memory_not_mapped=True)
            else:  
                _unmap_idx(idx-1)
        except IndexError:
            raise MemoryAccessViolation(addr, Perm(0), memory_not_mapped=True)

    def mprotect(self, addr: Addr, perm: Perm) -> None:
        
        idx = bisect.bisect_left(self._linear_map_addr, addr)
        try:
            if (idx % 2) == 0:  
                mmap = self._linear_map_map[idx]
                self._linear_map_map[idx] = MemMap(mmap.start, mmap.size, perm, mmap.name)  
            else:  
                mmap = self._linear_map_map[idx-1]
                self._linear_map_map[idx-1] = MemMap(mmap.start, mmap.size, perm, mmap.name)  
        except IndexError:
            raise MemoryAccessViolation(addr, Perm(0), memory_not_mapped=True)

    def __setitem__(self, key: Addr, value: bytes) -> None:
        
        if isinstance(key, slice):
            raise TypeError("slice unsupported for __setitem__")
        else:
            self.write(key, value)

    def __getitem__(self, item: Union[Addr, slice]) -> bytes:
        
        if isinstance(item, slice):
            return self.read(item.start, item.stop)
        elif isinstance(item, int):
            return self.read(item, 1)

    def write(self, addr: Addr, data: bytes) -> None:
        
        if self._segment_enabled:
            mmap = self._get_map(addr, len(data))
            if mmap is None:
                raise MemoryAccessViolation(addr, Perm.W, memory_not_mapped=True)
            if Perm.W not in mmap.perm:
                raise MemoryAccessViolation(addr, Perm.W, map_perm=mmap.perm, perm_error=True)
        return self.ctx.setConcreteMemoryAreaValue(addr, data)

    def read(self, addr: Addr, size: ByteSize) -> bytes:
        
        if self._segment_enabled:
            mmap = self._get_map(addr, size)
            if mmap is None:
                raise MemoryAccessViolation(addr, Perm.R, memory_not_mapped=True)
            if Perm.R not in mmap.perm:
                raise MemoryAccessViolation(addr, Perm.R, map_perm=mmap.perm, perm_error=True)
        return self.ctx.getConcreteMemoryAreaValue(addr, size)

    def _get_map(self, ptr: Addr, size: ByteSize) -> Optional[MemMap]:
        
        idx = bisect.bisect_left(self._linear_map_addr, ptr)
        try:
            addr = self._linear_map_addr[idx]
            if (idx % 2) == 0:  
                end = self._linear_map_addr[idx+1]
                return self._linear_map_map[idx] if (ptr == addr and ptr+size <= end+1) else None
            else:  
                start = self._linear_map_addr[idx-1]
                return self._linear_map_map[idx-1] if (start <= addr and ptr+size <= addr+1) else None  
        except IndexError:
            return None  

    def get_map(self, addr: Addr, size: ByteSize = 1) -> Optional[MemMap]:
        
        return self._get_map(addr, size)

    def find_map(self, name: str) -> Optional[List[MemMap]]:
        
        mmaps = []
        for mmap in (x for x in self._linear_map_map if x):
            if mmap.name == name:
                mmaps.append(mmap)
        return mmaps

    def map_from_name(self, name: str) -> MemMap:
        
        for mmap in (x for x in self._linear_map_map if x):
            if mmap.name == name:
                return mmap
        assert False

    def is_mapped(self, ptr: Addr, size: ByteSize = 1) -> bool:
        
        return self._get_map(ptr, size) is not None

    def has_ever_been_written(self, ptr: Addr, size: ByteSize) -> bool:
        
        return self.ctx.isConcreteMemoryValueDefined(ptr, size)

    def read_uint(self, addr: Addr, size: ByteSize = 4):
        
        data = self.read(addr, size)
        return struct.unpack(self._endian_key+STRUCT_MAP[(True, size)], data)[0]

    def read_sint(self, addr: Addr, size: ByteSize = 4):
        
        data = self.read(addr, size)
        return struct.unpack(self._endian_key+STRUCT_MAP[(False, size)], data)[0]

    def read_ptr(self, addr: Addr) -> int:
        
        return self.read_uint(addr, self._ptr_size)

    def read_char(self, addr: Addr) -> int:
        
        return self.read_sint(addr, 1)

    def read_uchar(self, addr: Addr) -> int:
        
        return self.read_uint(addr, 1)

    def read_int(self, addr: Addr) -> int:
        
        return self.read_sint(addr, 4)

    def read_word(self, addr: Addr) -> int:
        
        return self.read_uint(addr, 2)

    def read_dword(self, addr: Addr) -> int:
        
        return self.read_uint(addr, 4)

    def read_qword(self, addr: Addr) -> int:
        
        return self.read_uint(addr, 8)

    def read_long(self, addr: Addr) -> int:
        
        return self.read_sint(addr, 4)

    def read_ulong(self, addr: Addr) -> int:
        
        return self.read_uint(addr, 4)

    def read_long_long(self, addr: Addr) -> int:
        
        return self.read_sint(addr, 8)

    def read_ulong_long(self, addr: Addr) -> int:
        
        return self.read_uint(addr, 8)

    def read_string(self, addr: Addr) -> str:
        
        s = ""
        index = 0
        while True:
            val = self.read_uint(addr+index, 1)
            if not val:
                return s
            s += chr(val)
            index += 1

    def write_int(self, addr: Addr, value: int, size: ByteSize = 4):
        
        self.write(addr, struct.pack(self._endian_key+STRUCT_MAP[(value >= 0, size)], value))

    def write_ptr(self, addr: Addr, value: int) -> None:
        
        self.write_int(addr, value, self._ptr_size)

    def write_char(self, addr: Addr, value: int) -> None:
        
        self.write_int(addr, value, 1)

    def write_word(self, addr: Addr, value: int) -> None:
        
        self.write_int(addr, value, 2)

    def write_dword(self, addr: Addr, value: int) -> None:
        
        self.write_int(addr, value, 4)

    def write_qword(self, addr: Addr, value: int) -> None:
        
        self.write_int(addr, value, 8)

    def write_long(self, addr: Addr, value: int) -> None:
        
        return self.write_int(addr, value, 4)

    def write_long_long(self, addr: Addr, value: int) -> None:
        
        return self.write_int(addr, value, 8)
