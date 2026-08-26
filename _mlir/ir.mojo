# Compile-only stub of MAX 25.3's `_mlir.ir`. See package docstring.

from std.os import abort


@fieldwise_init
struct MlirHandle(TrivialRegisterPassable, ImplicitlyCopyable, Writable):
    """Opaque stand-in for every MLIR C API handle type."""

    var ptr: Int

    def __init__(out self):
        self.ptr = 0

    def write_to[W: Writer](self, mut writer: W):
        writer.write("<mlir-stub>")


comptime _fail = "the _mlir stub cannot build graphs: MAX 25.3's Mojo graph runtime is unavailable on Mojo 1.0"


@always_inline("nodebug")
def _unavailable() -> MlirHandle:
    abort(_fail)
    return MlirHandle()


@fieldwise_init
struct Context(TrivialRegisterPassable, ImplicitlyCopyable):
    comptime cType = MlirHandle
    var c: Self.cType

    def __init__(out self):
        self.c = MlirHandle()

    def load_modular_dialects(self):
        pass

    def load_all_available_dialects(self):
        pass

    def diagnostic_error(self) -> Context:
        return self

    def __enter__(self) -> Context:
        return self

    def __exit__(self):
        pass


@fieldwise_init
struct Location(TrivialRegisterPassable, ImplicitlyCopyable):
    comptime cType = MlirHandle
    var c: Self.cType

    def __init__(out self, ctx: Context, name: String, line: Int, col: Int):
        self.c = MlirHandle()

    @staticmethod
    def unknown(ctx: Context) -> Location:
        return Location(MlirHandle())


struct Identifier(TrivialRegisterPassable, ImplicitlyCopyable, Writable):
    comptime cType = MlirHandle
    var c: Self.cType

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.c)

    @implicit
    def __init__(out self, c: MlirHandle):
        self.c = c

    def __init__(out self, ctx: Context, name: String):
        self.c = MlirHandle()


struct Attribute(TrivialRegisterPassable, ImplicitlyCopyable, Writable):
    comptime cType = MlirHandle
    var c: Self.cType

    @implicit
    def __init__(out self, c: MlirHandle):
        self.c = c

    @staticmethod
    def parse(ctx: Context, repr: String) -> Attribute:
        return Attribute(_unavailable())

    @staticmethod
    def from_mlir(c: MlirHandle) -> Attribute:
        return Attribute(c)

    def to_mlir(self) -> MlirHandle:
        return self.c

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.c)


@fieldwise_init
struct NamedAttribute(TrivialRegisterPassable, ImplicitlyCopyable):
    comptime cType = MlirHandle
    var name: Identifier
    var attr: Attribute

    @implicit
    def __init__(out self, c: MlirHandle):
        self.name = Identifier(c)
        self.attr = Attribute(c)

    def to_mlir(self) -> MlirHandle:
        return MlirHandle()


struct Type(TrivialRegisterPassable, ImplicitlyCopyable, Writable):
    comptime cType = MlirHandle
    var c: Self.cType

    @implicit
    def __init__(out self, c: MlirHandle):
        self.c = c

    @staticmethod
    def parse(ctx: Context, repr: String) -> Type:
        return Type(_unavailable())

    @staticmethod
    def from_mlir(c: MlirHandle) -> Type:
        return Type(c)

    def to_mlir(self) -> MlirHandle:
        return self.c

    def context(self) -> Context:
        return Context(MlirHandle())

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.c)


@fieldwise_init
struct Value(TrivialRegisterPassable, ImplicitlyCopyable, Writable):
    comptime cType = MlirHandle
    var c: Self.cType

    def __init__(out self):
        self.c = MlirHandle()

    def type(self) -> MlirHandle:
        return MlirHandle()

    def replace_all_uses_with(self, other: Value):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.c)


@fieldwise_init
struct Block(TrivialRegisterPassable, ImplicitlyCopyable):
    comptime cType = MlirHandle
    var c: Self.cType

    def __init__(out self):
        self.c = MlirHandle()

    def argument(self, idx: Int) -> Value:
        return Value(MlirHandle())

    def num_arguments(self) -> Int:
        return 0

    def first_operation(self) -> Operation:
        return Operation(MlirHandle())

    def terminator(self) -> Optional[Operation]:
        return None

    def insert_before(self, anchor: Operation, op: Operation):
        pass

    def append(self, op: Operation):
        pass


@fieldwise_init
struct Region(TrivialRegisterPassable, ImplicitlyCopyable):
    comptime cType = MlirHandle
    var c: Self.cType

    def first_block(self) -> Block:
        return Block(MlirHandle())


struct Operation(TrivialRegisterPassable, ImplicitlyCopyable, Writable):
    comptime cType = MlirHandle
    var c: Self.cType

    @implicit
    def __init__(out self, c: MlirHandle):
        self.c = c

    def __init__(out self):
        self.c = MlirHandle()

    def __init__(
        out self,
        *,
        name: String,
        location: Location,
        operands: List[Value],
        results: List[Type],
        attributes: List[NamedAttribute] = List[NamedAttribute](),
        enable_result_type_inference: Bool = False,
    ):
        self.c = MlirHandle()

    def verify(self) -> Bool:
        return False

    def num_results(self) -> Int:
        return 0

    def result(self, idx: Int) -> Value:
        return Value(_unavailable())

    def first_block(self) -> Block:
        return Block(MlirHandle())

    def set_inherent_attr(self, name: String, attr: MlirHandle):
        pass

    def set_inherent_attr(self, name: String, attr: Attribute):
        pass

    def get_inherent_attr(self, name: String) -> MlirHandle:
        return MlirHandle()

    def set_discardable_attr(self, name: String, attr: MlirHandle):
        pass

    def set_discardable_attr(self, name: String, attr: Attribute):
        pass

    def parent(self) -> Operation:
        return Operation(MlirHandle())

    def region(self, idx: Int) -> Region:
        return Region(MlirHandle())

    def destroy(self):
        pass

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.c)


@fieldwise_init
struct Module(TrivialRegisterPassable, ImplicitlyCopyable, Writable):
    comptime cType = MlirHandle
    var c: Self.cType

    def __init__(out self, location: Location):
        self.c = MlirHandle()

    @staticmethod
    def from_op(op: Operation) -> Module:
        return Module(op.c)

    @staticmethod
    def parse(ctx: Context, repr: String) -> Module:
        return Module(_unavailable())

    def body(self) -> Block:
        return Block(MlirHandle())

    def as_op(self) -> Operation:
        return Operation(self.c)

    def context(self) -> Context:
        return Context(MlirHandle())

    def verify(self) -> Bool:
        return False

    def to_mlir(self) -> MlirHandle:
        return self.c

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.c)
