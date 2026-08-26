# ===----------------------------------------------------------------------=== #
# Mojo 1.0 compatibility package: minimal re-implementation of the MAX 25.3
# `tensor` module surface that nabla's vendored compiler wrappers consume
# (Tensor, TensorShape, TensorSpec, RuntimeTensorSpec). Pure Mojo — these are
# plain host-memory containers with no MAX runtime dependency.
# ===----------------------------------------------------------------------=== #

from std.memory import alloc, memcpy, memset_zero
from std.memory.unsafe_pointer import UnsafePointer
from std.utils.index import IndexList

from . import _indexing as _indexing



def _dtype_bytes(d: DType) -> Int:
    """Byte width of a runtime DType (Mojo 1.0 dropped DType.sizeof)."""
    if d == DType.bool or d == DType.int8 or d == DType.uint8 or d.is_float8():
        return 1
    if d == DType.int16 or d == DType.uint16 or d.is_half_float():
        return 2
    if d == DType.int64 or d == DType.uint64 or d == DType.float64:
        return 8
    return 4


struct TensorShape(Copyable, Movable, Writable):
    """The shape of a tensor: a list of dimension sizes."""

    var _dims: List[Int]

    def __init__(out self):
        self._dims = List[Int]()

    @implicit
    def __init__(out self, *dims: Int):
        self._dims = List[Int]()
        for d in dims:
            self._dims.append(d)

    @implicit
    def __init__(out self, dims: List[Int]):
        self._dims = dims.copy()

    def rank(self) -> Int:
        return len(self._dims)

    def num_elements(self) -> Int:
        var n = 1
        for d in self._dims:
            n *= d
        return n

    def __getitem__(self, idx: Int) -> Int:
        if idx < 0:
            return self._dims[len(self._dims) + idx]
        return self._dims[idx]

    def __len__(self) -> Int:
        return len(self._dims)

    def __eq__(self, other: Self) -> Bool:
        if len(self._dims) != len(other._dims):
            return False
        for i in range(len(self._dims)):
            if self._dims[i] != other._dims[i]:
                return False
        return True

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def __str__(self) -> String:
        return String.write(self)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(")
        for i in range(len(self._dims)):
            if i > 0:
                writer.write(", ")
            writer.write(self._dims[i])
        writer.write(")")


struct TensorSpec(Copyable, Movable, Writable):
    """A tensor's dtype together with its shape."""

    var _dtype: DType
    var _shape: TensorShape

    def __init__(out self):
        self._dtype = DType.uint8
        self._shape = TensorShape()

    def __init__(out self, type: DType, *dims: Int):
        self._dtype = type
        var shape = List[Int]()
        for d in dims:
            shape.append(d)
        self._shape = TensorShape(shape)

    def __init__(out self, type: DType, shape: TensorShape):
        self._dtype = type
        self._shape = shape.copy()

    def __init__(out self, type: DType, shape: List[Int]):
        self._dtype = type
        self._shape = TensorShape(shape)

    def __init__[t: DType, r: Int](out self, spec: RuntimeTensorSpec[t, r]):
        self._dtype = t
        var dims = List[Int]()
        for i in range(r):
            dims.append(spec.shape[i])
        self._shape = TensorShape(dims)

    def dtype(self) -> DType:
        return self._dtype

    def shape(self) -> TensorShape:
        return self._shape.copy()

    def rank(self) -> Int:
        return self._shape.rank()

    def num_elements(self) -> Int:
        return self._shape.num_elements()

    def bytecount(self) -> Int:
        return self.num_elements() * _dtype_bytes(self._dtype)

    def __getitem__(self, idx: Int) -> Int:
        return self._shape[idx]

    def __eq__(self, other: Self) -> Bool:
        return self._dtype == other._dtype and self._shape == other._shape

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def __str__(self) -> String:
        return String.write(self)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._shape, "x", String(self._dtype))


struct Tensor[dtype: DType](Copyable, Movable, Writable):
    """An owning host-memory tensor of `dtype` scalars."""

    var _spec: TensorSpec
    var _ptr: UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]

    def __init__(out self):
        self._spec = TensorSpec(Self.dtype, TensorShape())
        self._ptr = alloc[Scalar[Self.dtype]](1)

    @implicit
    def __init__(out self, shape: TensorShape):
        self._spec = TensorSpec(Self.dtype, shape)
        var n = max(1, shape.num_elements())
        self._ptr = alloc[Scalar[Self.dtype]](n)
        memset_zero(self._ptr, n)

    def __init__(out self, shape: TensorShape, value: Scalar[Self.dtype]):
        self = Self(shape)
        for i in range(self.num_elements()):
            self._ptr[i] = value

    def __init__(out self, shape: List[Int], value: Scalar[Self.dtype]):
        self = Self(TensorShape(shape), value)

    @implicit
    def __init__(out self, spec: TensorSpec):
        self = Self(spec.shape())

    def __init__(out self, shape: TensorShape, var ptr: UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]):
        """Takes ownership of `ptr`, which must hold `shape.num_elements()`
        scalars allocated with `UnsafePointer.alloc`."""
        self._spec = TensorSpec(Self.dtype, shape)
        self._ptr = ptr

    def copy(self) -> Self:
        var n = max(1, self.num_elements())
        var ptr = alloc[Scalar[Self.dtype]](n)
        memcpy(dest=ptr, src=self._ptr, count=self.num_elements())
        return Self(self._spec.shape(), ptr)

    def __init__(out self, *, deinit existing: Self):
        self._spec = existing._spec.copy()
        self._ptr = existing._ptr

    def __deinit__(deinit self):
        self._ptr.free()

    def spec(self) -> TensorSpec:
        return self._spec.copy()

    def shape(self) -> TensorShape:
        return self._spec.shape()

    def rank(self) -> Int:
        return self._spec.rank()

    def num_elements(self) -> Int:
        return self._spec.num_elements()

    def bytecount(self) -> Int:
        return self._spec.bytecount()

    def unsafe_ptr(self) -> UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]:
        return self._ptr

    def unsafe_uint8_ptr(self) -> UnsafePointer[Scalar[DType.uint8], MutUntrackedOrigin]:
        return self._ptr.bitcast[Scalar[DType.uint8]]()

    def data(self) -> UnsafePointer[Scalar[dtype], MutUntrackedOrigin]:
        return self._ptr

    def load(self, idx: Int) -> Scalar[dtype]:
        return self._ptr[idx]

    def store(mut self, idx: Int, value: Scalar[dtype]):
        self._ptr[idx] = value

    def __getitem__(self, idx: Int) -> Scalar[dtype]:
        return self._ptr[idx]

    def __setitem__(mut self, idx: Int, value: Scalar[dtype]):
        self._ptr[idx] = value

    def _take_data_ptr(deinit self) -> UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]:
        return self._ptr

    def _steal_ptr(deinit self) -> UnsafePointer[Scalar[Self.dtype], MutUntrackedOrigin]:
        return self._ptr

    def __str__(self) -> String:
        return String.write(self)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Tensor[", String(Self.dtype), "](", self._spec.shape(), ")")


struct RuntimeTensorSpec[type: DType, rank: Int](Copyable, ImplicitlyCopyable, Movable):
    """A rank-parameterized tensor spec, as used by the MAX driver API."""

    var shape: IndexList[Self.rank]

    @implicit
    def __init__(out self, shape: IndexList[Self.rank]):
        self.shape = shape

    @implicit
    def __init__(out self, spec: TensorSpec):
        self.shape = IndexList[Self.rank]()
        for i in range(Self.rank):
            self.shape[i] = spec[i]

    def bytecount(self) -> Int:
        var n = 1
        for i in range(Self.rank):
            n *= self.shape[i]
        return n * _dtype_bytes(Self.type)

    def __getitem__(self, idx: Int) -> Int:
        return self.shape[idx]
