from __future__ import annotations
import hashlib
import struct
from pathlib import Path
from typing import List, Generator, Tuple, Set, Union, Dict, Optional
from collections import Counter
from enum import IntFlag, Enum, auto
import pickle
import enum_tools.documentation


from triton import AST_NODE


from tritondse.types import (
    Addr,
    PathConstraint,
    PathBranch,
    SolverStatus,
    PathHash,
    Edge,
    SymExType,
)
import tritondse.logging

logger = tritondse.logging.get("coverage")

CovItem = Union[Addr, Edge, PathHash, Tuple[PathHash, Edge]]


@enum_tools.documentation.document_enum
class CoverageStrategy(str, Enum):

    BLOCK = "block"
    EDGE = "edge"
    PATH = "path"
    PREFIXED_EDGE = "PREFIXED_EDGE"


@enum_tools.documentation.document_enum
class BranchSolvingStrategy(IntFlag):

    ALL_NOT_COVERED = auto()
    FIRST_LAST_NOT_COVERED = auto()
    UNSAT_ONCE = auto()
    TIMEOUT_ONCE = auto()
    TIMEOUT_ALWAYS = auto()
    COVER_SYM_DYNJUMP = auto()
    COVER_SYM_READ = auto()
    COVER_SYM_WRITE = auto()
    SOUND_MEM_ACCESS = auto()
    MANUAL = auto()


class CoverageSingleRun(object):

    def __init__(self, strategy: CoverageStrategy):

        self.strategy: CoverageStrategy = strategy

        self.covered_instructions: Dict[Addr, int] = Counter()

        self.covered_items: Dict[CovItem, int] = Counter()

        self.not_covered_items: Set[CovItem] = set()
        self._not_covered_items_mirror: Dict[CovItem, List[str]] = {}

        self._current_path: List[Addr] = []

        self._current_path_hash = hashlib.md5()

    def add_covered_address(self, address: Addr) -> None:

        self.covered_instructions[address] += 1

    def add_covered_dynamic_branch(self, source: Addr, target: Addr) -> None:

        if self.strategy == CoverageStrategy.BLOCK:
            pass

        if self.strategy == CoverageStrategy.EDGE:
            self.covered_items[(source, target)] += 1
            self.not_covered_items.discard((source, target))

        if self.strategy == CoverageStrategy.PATH:
            self._current_path.append(target)
            self._current_path_hash.update(struct.pack("<Q", target))
            self.covered_items[self._current_path_hash.hexdigest()] += 1

        if self.strategy == CoverageStrategy.PREFIXED_EDGE:

            self.covered_items[("", (source, target))] += 1

            self._current_path.append(target)
            self._current_path_hash.update(struct.pack("<Q", target))

    def add_covered_branch(
        self, program_counter: Addr, taken_addr: Addr, not_taken_addr: Addr
    ) -> None:

        if self.strategy == CoverageStrategy.BLOCK:
            self.covered_items[taken_addr] += 1
            self.not_covered_items.discard(taken_addr)
            if not_taken_addr not in self.covered_items:
                self.not_covered_items.add(not_taken_addr)

        if self.strategy == CoverageStrategy.EDGE:
            taken_tuple, not_taken_tuple = (program_counter, taken_addr), (
                program_counter,
                not_taken_addr,
            )
            self.covered_items[taken_tuple] += 1
            self.not_covered_items.discard(taken_tuple)
            if not_taken_tuple not in self.covered_items:
                self.not_covered_items.add(not_taken_tuple)

        if self.strategy == CoverageStrategy.PATH:
            self._current_path.append(taken_addr)

            not_taken_path_hash = self._current_path_hash.copy()
            not_taken_path_hash.update(struct.pack("<Q", not_taken_addr))
            self.not_covered_items.add(not_taken_path_hash.hexdigest())

            self._current_path_hash.update(struct.pack("<Q", taken_addr))
            self.covered_items[self._current_path_hash.hexdigest()] += 1

        if self.strategy == CoverageStrategy.PREFIXED_EDGE:
            taken_tuple, not_taken_tuple = (program_counter, taken_addr), (
                program_counter,
                not_taken_addr,
            )
            _, not_taken = (self._current_path_hash.hexdigest(), taken_tuple), (
                self._current_path_hash.hexdigest(),
                not_taken_tuple,
            )
            gtaken, gnot_taken = ("", taken_tuple), ("", not_taken_tuple)

            self.covered_items[gtaken] += 1

            if taken_tuple in self._not_covered_items_mirror:
                for prefix in self._not_covered_items_mirror[taken_tuple]:
                    self.not_covered_items.discard((prefix, taken_tuple))
                self._not_covered_items_mirror.pop(taken_tuple)

            if gnot_taken not in self.covered_items:
                self.not_covered_items.add(not_taken)
                if not_taken[1] not in self._not_covered_items_mirror:
                    self._not_covered_items_mirror[not_taken[1]] = [not_taken[0]]
                else:
                    self._not_covered_items_mirror[not_taken[1]].append(not_taken[0])

            self._current_path.append(taken_addr)
            self._current_path_hash.update(struct.pack("<Q", taken_addr))

    @property
    def unique_instruction_covered(self) -> int:

        return len(self.covered_instructions)

    @property
    def unique_covitem_covered(self) -> int:

        return len(self.covered_items)

    @property
    def total_instruction_executed(self) -> int:

        return sum(self.covered_instructions.values())

    def post_execution(self) -> None:

        pass

    def is_covered(self, item: CovItem) -> bool:

        if self.strategy == CoverageStrategy.PREFIXED_EDGE:
            try:
                return ("", item[1]) in self.covered_items
            except TypeError:
                return False
        else:
            return item in self.covered_items

    def pp_item(self, covitem: CovItem) -> str:

        if self.strategy == CoverageStrategy.BLOCK:
            return f"0x{covitem:08x}"
        elif self.strategy == CoverageStrategy.EDGE:
            return f"(0x{covitem[0]:08x}-0x{covitem[1]:08x})"
        elif self.strategy == CoverageStrategy.PATH:
            return covitem[:10]
        elif self.strategy == CoverageStrategy.PREFIXED_EDGE:
            return f"({covitem[0][:6]}_0x{covitem[1][0]:08x}-0x{covitem[1][1]:08x})"

    def difference(self, other: CoverageSingleRun) -> Set[CovItem]:
        if self.strategy == other.strategy:
            return self.covered_items.keys() - other.covered_items.keys()
        else:
            logger.error(
                "Trying to make difference of coverage with different strategies"
            )
            return set()

    def __sub__(self, other) -> Set[CovItem]:
        return self.difference(other)


class GlobalCoverage(CoverageSingleRun):

    COVERAGE_FILE = "coverage.json"

    def __init__(
        self, strategy: CoverageStrategy, branch_strategy: BranchSolvingStrategy
    ):

        super().__init__(strategy)
        self.branch_strategy = branch_strategy

        self.pending_coverage: Set[CovItem] = set()

        self.uncoverable_items: Dict[CovItem, SolverStatus] = {}

        self.covered_symbolic_pointers: Set[Addr] = set()

    def iter_new_paths(self, path_constraints: List[PathConstraint]) -> Generator[
        Tuple[SymExType, List[PathConstraint], PathBranch, CovItem, int],
        Optional[SolverStatus],
        None,
    ]:

        if BranchSolvingStrategy.MANUAL in self.branch_strategy:
            logger.info(f"Branch solving strategy set to MANUAL.")
            return

        pending_csts = []
        current_hash = hashlib.md5()

        not_covered_items = self._get_items_trace(path_constraints)

        for i, pc in enumerate(path_constraints):
            if pc.isMultipleBranches():
                for branch in pc.getBranchConstraints():

                    if not branch["isTaken"]:
                        covitem = self._get_covitem(current_hash, branch)
                        generic_covitem = (
                            ("", covitem[1])
                            if self.strategy == CoverageStrategy.PREFIXED_EDGE
                            else covitem
                        )

                        if (
                            generic_covitem not in self.covered_items
                            and generic_covitem not in self.pending_coverage
                            and covitem not in self.uncoverable_items
                            and i in not_covered_items.get(covitem, [])
                        ):

                            res = (
                                yield SymExType.CONDITIONAL_JMP,
                                pending_csts,
                                branch,
                                covitem,
                                i,
                            )

                            if res == SolverStatus.SAT:
                                self.pending_coverage.add(generic_covitem)

                            elif res == SolverStatus.UNSAT:
                                if (
                                    BranchSolvingStrategy.UNSAT_ONCE
                                    in self.branch_strategy
                                ):
                                    self.uncoverable_items[covitem] = res
                                elif self.strategy in [
                                    CoverageStrategy.PATH,
                                    CoverageStrategy.PREFIXED_EDGE,
                                ]:
                                    self.uncoverable_items[covitem] = res

                            elif res == SolverStatus.TIMEOUT:
                                if (
                                    BranchSolvingStrategy.TIMEOUT_ONCE
                                    in self.branch_strategy
                                ):
                                    self.uncoverable_items[covitem] = res

                            elif res == SolverStatus.UNKNOWN:
                                pass

                            else:
                                logger.debug(f"Branch skipped!")

                            pending_csts = []

                    else:
                        pass

                pending_csts.append(pc)
                current_hash.update(struct.pack("<Q", pc.getTakenAddress()))

            else:
                cmt = pc.getComment()

                if (
                    (
                        cmt.startswith("dyn-jmp")
                        and BranchSolvingStrategy.COVER_SYM_DYNJUMP
                        in self.branch_strategy
                    )
                    or (
                        cmt.startswith("sym-read")
                        and BranchSolvingStrategy.COVER_SYM_READ in self.branch_strategy
                    )
                    or (
                        cmt.startswith("sym-write")
                        and BranchSolvingStrategy.COVER_SYM_WRITE
                        in self.branch_strategy
                    )
                ):
                    typ, offset, addr = cmt.split(":")
                    typ = SymExType(typ)
                    offset, addr = int(offset), int(addr)
                    if addr not in self.covered_symbolic_pointers:
                        pred = pc.getTakenPredicate()
                        if pred.getType() == AST_NODE.EQUAL:
                            p1, p2 = pred.getChildren()
                            if p2.getType() == AST_NODE.BV:
                                logger.info(
                                    f"Try to enumerate value {offset}:0x{addr:02x}: {p1}"
                                )
                                res = (
                                    yield typ,
                                    pending_csts,
                                    p1,
                                    (addr, p2.evaluate()),
                                    i,
                                )
                                self.covered_symbolic_pointers.add(addr)
                            else:
                                logger.warning(
                                    f"memory constraint unexpected pattern: {pred}"
                                )
                        else:
                            logger.warning(
                                f"memory constraint unexpected pattern: {pred}"
                            )

                    if BranchSolvingStrategy.SOUND_MEM_ACCESS in self.branch_strategy:
                        pending_csts.append(pc)

                else:
                    pending_csts.append(pc)

    def _get_covitem(self, path_hash, branch: PathBranch) -> CovItem:
        src, dst = branch["srcAddr"], branch["dstAddr"]

        if self.strategy == CoverageStrategy.BLOCK:
            return dst
        elif self.strategy == CoverageStrategy.EDGE:
            return src, dst
        elif self.strategy == CoverageStrategy.PATH:

            forked_hash = path_hash.copy()
            forked_hash.update(struct.pack("<Q", dst))
            return forked_hash.hexdigest()
        elif self.strategy == CoverageStrategy.PREFIXED_EDGE:
            return path_hash.hexdigest(), (src, dst)
        else:
            assert False

    def _get_items_trace(
        self, path_constraints: List[PathConstraint]
    ) -> Dict[CovItem, List[int]]:

        not_covered = {}
        current_hash = hashlib.md5()
        for i, pc in enumerate(path_constraints):
            if pc.isMultipleBranches():
                for branch in pc.getBranchConstraints():
                    if not branch["isTaken"]:
                        covitem = self._get_covitem(current_hash, branch)
                        if covitem in not_covered:
                            not_covered[covitem].append(i)
                        else:
                            not_covered[covitem] = [i]
                current_hash.update(struct.pack("<Q", pc.getTakenAddress()))
            else:
                pass

        if BranchSolvingStrategy.FIRST_LAST_NOT_COVERED in self.branch_strategy:
            if self.strategy == CoverageStrategy.PREFIXED_EDGE:

                m = {("", e): [] for h, e in not_covered.keys()}
                for (h, e), v in not_covered.items():
                    m[("", e)].extend(v)
                for k in m.keys():
                    idxs = m[k]
                    if len(idxs) > 2:
                        m[k] = [min(idxs), max(idxs)]
                for k in not_covered.keys():
                    not_covered[k] = m[("", k[1])]

            else:
                for k in not_covered.keys():
                    lst = not_covered[k]
                    if len(lst) > 2:
                        not_covered[k] = [lst[0], lst[-1]]
        else:
            pass
        return not_covered

    def merge(self, other: CoverageSingleRun) -> None:

        assert self.strategy == other.strategy

        self.covered_instructions.update(other.covered_instructions)

        self.pending_coverage.difference_update(other.covered_items)

        self.covered_items.update(other.covered_items)

        if self.strategy == CoverageStrategy.PREFIXED_EDGE:

            for _, edge in other.covered_items.keys():
                if edge in self._not_covered_items_mirror:
                    for prefix in self._not_covered_items_mirror[edge]:
                        self.not_covered_items.discard((prefix, edge))
                    self._not_covered_items_mirror.pop(edge)

            for prefix, edge in other.not_covered_items:
                if ("", edge) not in self.covered_items:
                    self.not_covered_items.add((prefix, edge))
                    if edge not in self._not_covered_items_mirror:
                        self._not_covered_items_mirror[edge] = [prefix]
                    else:
                        self._not_covered_items_mirror[edge].append(prefix)

        else:
            self.not_covered_items.update(
                other.not_covered_items - self.covered_items.keys()
            )

    def can_improve_coverage(self, other: CoverageSingleRun) -> bool:

        return bool(self.new_items_to_cover(other))

    def can_cover_symbolic_pointers(self, execution: "SymbolicExecutor") -> bool:

        path_constraints = execution.pstate.get_path_constraints()

        for pc in path_constraints:
            if not pc.isMultipleBranches():
                cmt = pc.getComment()

                if (
                    (
                        cmt.startswith("dyn-jmp")
                        and BranchSolvingStrategy.COVER_SYM_DYNJUMP
                        in self.branch_strategy
                    )
                    or (
                        cmt.startswith("sym-read")
                        and BranchSolvingStrategy.COVER_SYM_READ in self.branch_strategy
                    )
                    or (
                        cmt.startswith("sym-write")
                        and BranchSolvingStrategy.COVER_SYM_WRITE
                        in self.branch_strategy
                    )
                ):
                    typ, offset, addr = cmt.split(":")
                    typ = SymExType(typ)
                    offset, addr = int(offset), int(addr)
                    if addr not in self.covered_symbolic_pointers:
                        return True
        return False

    def new_items_to_cover(self, other: CoverageSingleRun) -> Set[CovItem]:

        assert self.strategy == other.strategy

        return (
            other.not_covered_items
            - self.covered_items.keys()
            - self.uncoverable_items.keys()
            - self.pending_coverage
        )

    def improve_coverage(self, other: CoverageSingleRun) -> bool:

        return bool(other.covered_items.keys() - self.covered_items.keys())

    @staticmethod
    def from_file(file: Union[str, Path]) -> "GlobalCoverage":
        with open(file, "rb") as f:
            obj = pickle.load(f)
        return obj

    def to_file(self, file: Union[str, Path]) -> None:
        copy = self._current_path_hash
        self._current_path_hash = None
        with open(file, "wb") as f:
            pickle.dump(self, f)
        self._current_path_hash = copy

    def post_exploration(self, workspace: "Workspace") -> None:

        self.to_file(workspace.get_metadata_file_path(self.COVERAGE_FILE))

    def clone(self) -> "GlobalCoverage":
        cov2 = GlobalCoverage(self.strategy, self.branch_strategy)

        cov2.covered_instructions = Counter(
            {k: v for k, v in self.covered_instructions.items()}
        )
        cov2.covered_items = Counter({k: v for k, v in self.covered_items.items()})
        cov2.not_covered_items = {x for x in self.not_covered_items}
        cov2._not_covered_items_mirror = {
            k: v for k, v in self._not_covered_items_mirror.items()
        }
        cov2._current_path = self._current_path[:]
        self._current_path: List[Addr] = []
        self._current_path_hash = self._current_path_hash.copy()

        cov2.pending_coverage = {x for x in self.pending_coverage}
        cov2.uncoverable_items = {k: v for k, v in self.uncoverable_items.items()}
        cov2.covered_symbolic_pointers = {x for x in self.covered_symbolic_pointers}
        return cov2
