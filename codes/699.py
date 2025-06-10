import logging
import json
from enum import Enum, IntFlag
from pathlib import Path
from typing import List
from functools import reduce


from tritondse.coverage import CoverageStrategy, BranchSolvingStrategy
from tritondse.types import SmtSolver
from tritondse.seed import SeedFormat
import tritondse.logging

logger = tritondse.logging.get("config")


class Config(object):

    def __init__(
        self,
        seed_format: SeedFormat = SeedFormat.RAW,
        pipe_stdout: bool = False,
        pipe_stderr: bool = False,
        skip_sleep_routine: bool = False,
        smt_solver: SmtSolver = SmtSolver.Z3,
        smt_timeout: int = 5000,
        execution_timeout: int = 0,
        exploration_timeout: int = 0,
        exploration_limit: int = 0,
        thread_scheduling: int = 200,
        smt_queries_limit: int = 1200,
        smt_enumeration_limit: int = 40,
        coverage_strategy: CoverageStrategy = CoverageStrategy.BLOCK,
        branch_solving_strategy: BranchSolvingStrategy = BranchSolvingStrategy.FIRST_LAST_NOT_COVERED,
        workspace: str = "",
        workspace_reset=False,
        program_argv: List[str] = None,
        time_inc_coefficient: float = 0.00001,
        skip_unsupported_import: bool = False,
        skip_unsupported_instruction: bool = False,
        memory_segmentation: bool = True,
    ):

        self.seed_format: SeedFormat = seed_format

        self.pipe_stdout: bool = pipe_stdout

        self.pipe_stderr: bool = pipe_stderr

        self.skip_sleep_routine: bool = skip_sleep_routine

        self.smt_solver: SmtSolver = smt_solver

        self.smt_timeout: int = smt_timeout

        self.execution_timeout: int = execution_timeout

        self.exploration_timeout: int = exploration_timeout

        self.exploration_limit: int = exploration_limit

        self.thread_scheduling: int = thread_scheduling

        self.smt_queries_limit: int = smt_queries_limit

        self.smt_enumeration_limit: int = smt_enumeration_limit

        self.coverage_strategy: CoverageStrategy = coverage_strategy

        self.branch_solving_strategy: BranchSolvingStrategy = branch_solving_strategy

        self.workspace: str = workspace

        self.workspace_reset: bool = workspace_reset

        self.program_argv: List[str] = [] if program_argv is None else program_argv

        self.time_inc_coefficient: float = time_inc_coefficient

        self.skip_unsupported_import: bool = skip_unsupported_import

        self.skip_unsupported_instruction: bool = skip_unsupported_instruction

        self.memory_segmentation: bool = memory_segmentation

        self.custom = {}

    def __str__(self):
        return "\n".join(f"{k.ljust(23)}= {v}" for k, v in self.__dict__.items())

    def to_file(self, file: str) -> None:

        with open(file, "w") as f:
            f.write(self.to_json())

    @staticmethod
    def from_file(file: str) -> "Config":

        raw = Path(file).read_text()
        return Config.from_json(raw)

    @staticmethod
    def from_json(s: str) -> "Config":

        data = json.loads(s)
        c = Config()
        for k, v in data.items():
            if hasattr(c, k):
                mapping = {
                    "coverage_strategy": CoverageStrategy,
                    "smt_solver": SmtSolver,
                    "seed_format": SeedFormat,
                }
                if k in mapping:
                    v = mapping[k][v]
                elif k == "branch_solving_strategy":
                    v = reduce(lambda acc, x: BranchSolvingStrategy[x] | acc, v, 0)
                setattr(c, k, v)
            else:
                logging.warning(f"config unknown parameter: {k}")
        return c

    def to_json(self) -> str:

        def to_str_list(value):
            return [x.name for x in list(BranchSolvingStrategy) if x in value]

        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, IntFlag):
                d[k] = to_str_list(v)
            elif isinstance(v, Enum):
                d[k] = v.name
            else:
                d[k] = v
        return json.dumps(d, indent=2)

    def is_format_composite(self) -> bool:

        return self.seed_format == SeedFormat.COMPOSITE

    def is_format_raw(self) -> bool:

        return self.seed_format == SeedFormat.RAW
