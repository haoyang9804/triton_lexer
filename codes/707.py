from tritondse import ProbeInterface, CbType, SymbolicExecutor, ProcessState
import tritondse.logging

logger = tritondse.logging.get("probe.basictrace")


class BasicDebugTrace(ProbeInterface):

    NAME = "debugtrace-probe"

    def __init__(self):
        super(BasicDebugTrace, self).__init__()
        self._add_callback(CbType.PRE_INST, self.trace_debug)

    def trace_debug(self, se: SymbolicExecutor, __: ProcessState, ins: "Instruction"):
        logger.debug(
            f"[tid:{ins.getThreadId()}] {se.trace_offset} [0x{ins.getAddress():x}]: {ins.getDisassembly()}"
        )


class BasicTextTrace(ProbeInterface):

    NAME = "txttrace-probe"

    def __init__(self):
        super(BasicTextTrace, self).__init__()
        self._add_callback(CbType.PRE_EXEC, self.pre_execution)
        self._add_callback(CbType.POST_EXEC, self.post_execution)
        self._add_callback(CbType.PRE_INST, self.trace_debug)

        self._file = None

    def pre_execution(self, executor: SymbolicExecutor, _: ProcessState):

        name = f"{executor.uid}-{executor.seed.hash}.txt"
        file = executor.workspace.get_metadata_file_path(f"{self.NAME}/{name}")
        self._file = open(file, "w")

    def post_execution(self, _: SymbolicExecutor, __: ProcessState):
        self._file.close()

    def trace_debug(self, se: SymbolicExecutor, __: ProcessState, ins: "Instruction"):

        self._file.write(
            f"[tid:{ins.getThreadId()}] {se.trace_offset} [0x{ins.getAddress():x}]: {ins.getDisassembly()}\n"
        )
