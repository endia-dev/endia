# Compile-only stub of MAX 25.3's `_mlir.builtin_types`.

from .ir import Context, MlirHandle, Type


struct FunctionType(Copyable, Movable):
    comptime cType = MlirHandle
    var c: Self.cType
    var inputs: List[Type]
    var results: List[Type]

    @implicit
    def __init__(out self, c: MlirHandle):
        self.c = c
        self.inputs = List[Type]()
        self.results = List[Type]()

    def __init__(out self, inputs: List[Type], results: List[Type]):
        self.c = MlirHandle()
        self.inputs = inputs.copy()
        self.results = results.copy()

    def __init__(out self, ctx: Context, inputs: List[Type], results: List[Type]):
        self = Self(inputs, results)

    @staticmethod
    def from_mlir(c: MlirHandle) -> FunctionType:
        return FunctionType(c)

    def to_mlir(self) -> MlirHandle:
        return self.c
