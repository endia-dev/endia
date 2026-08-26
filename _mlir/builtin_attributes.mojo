# Compile-only stub of MAX 25.3's `_mlir.builtin_attributes`.

from .ir import Attribute, Context, MlirHandle, Type


@fieldwise_init
struct StringAttr(TrivialRegisterPassable, ImplicitlyCopyable):
    comptime cType = MlirHandle
    var c: Self.cType

    def __init__(out self, ctx: Context, value: String):
        self.c = MlirHandle()

    @staticmethod
    def from_mlir(c: MlirHandle) -> StringAttr:
        return StringAttr(c)

    def to_mlir(self) -> MlirHandle:
        return self.c


@fieldwise_init
struct BoolAttr(TrivialRegisterPassable, ImplicitlyCopyable):
    comptime cType = MlirHandle
    var c: Self.cType

    def __init__(out self, ctx: Context, value: Bool):
        self.c = MlirHandle()

    def to_mlir(self) -> MlirHandle:
        return self.c


struct TypeAttr(TrivialRegisterPassable, ImplicitlyCopyable):
    comptime cType = MlirHandle
    var c: Self.cType
    var type: MlirHandle

    @implicit
    def __init__(out self, c: MlirHandle):
        self.c = c
        self.type = c

    @staticmethod
    def from_mlir(c: MlirHandle) -> TypeAttr:
        return TypeAttr(c)

    def to_mlir(self) -> MlirHandle:
        return self.c
