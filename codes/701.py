from tritondse.types import Addr, ByteSize, Perm
from tritondse.memory import Memory
from tritondse.exception import AllocatorException
import tritondse.logging

logger = tritondse.logging.get("heapallocator")


class HeapAllocator(object):

    def __init__(self, start: Addr, end: Addr, memory: Memory):

        self.start: Addr = start

        self.end: Addr = end

        self._curr_offset: Addr = self.start
        self._memory = memory

        self.alloc_pool = dict()
        self.free_pool = dict()

    def alloc(self, size: ByteSize) -> Addr:

        if size <= 0:
            logger.error(f"Heap: invalid allocation size {size}")
            return 0

        ptr = None
        for sz in sorted(x for x in self.free_pool if x >= size):

            ptr = self.free_pool[sz].pop().start

            if not self.free_pool[sz]:
                del self.free_pool[sz]
            break

        if ptr is None:
            ptr = self._curr_offset
            self._curr_offset += size

        mmap = self._memory.map(ptr, size, Perm.R | Perm.W, "heap")
        self.alloc_pool.update({ptr: mmap})

        return ptr

    def free(self, ptr: Addr) -> None:

        if self.is_ptr_freed(ptr):
            raise AllocatorException("Double free or corruption!")

        if not self.is_ptr_allocated(ptr):
            raise AllocatorException(f"Invalid pointer ({hex(ptr)})")

        memmap = self.alloc_pool[ptr]
        if memmap.size in self.free_pool:
            self.free_pool[memmap.size].add(memmap)
        else:
            self.free_pool[memmap.size] = {memmap}

        self._memory.unmap(ptr)
        del self.alloc_pool[ptr]

    def is_ptr_allocated(self, ptr: Addr) -> bool:

        return self._memory.is_mapped(ptr, 1)

    def is_ptr_freed(self, ptr: Addr) -> bool:

        for size, chunks in self.free_pool.items():
            for chunk in chunks:
                if chunk.start <= ptr < chunk.start + size:
                    return True
        return False
