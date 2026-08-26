# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements type erased generic tensor and memory types.

For example you can use AnyTensor if you don't know the dtype of tensor in
advance or if you don't know the tensor is DeviceTensor or Tensor:

```mojo
from nabla.compiler.driver import Tensor, AnyTensor
from nabla.compiler.tensor import TensorShape

@fieldwise_init
struct Container(Copyable, Movable):
    var _tensor: AnyTensor

def main():
    tensor = Tensor[DType.float32, rank=1](TensorShape(1,))
    container = Container(tensor^)
```
"""
from std.collections import Optional
from std.sys import align_of as alignof, size_of as sizeof
from std.sys.info import CompilationTarget
from std.ffi import external_call

from nabla.compiler._utils import null_ptr, exchange
from nabla.compiler.tensor import TensorSpec
from std.memory import UnsafePointer

from std.utils import Variant

from .device import Device
from .device_memory import DeviceMemory
from .tensor import Tensor
from std.os import abort


struct AnyTensor(Writable, Copyable, Movable):
    """A type erased tensor representation that is useful
    for situations where we need variadics of tensors."""

    var _data: UnsafePointer[UInt8, MutUntrackedOrigin]
    var _spec: TensorSpec
    var _device: Device
    var _name: Optional[String]
    var _device_memory_impl_ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    def __init__(out self) raises:
        """Default constructor for AnyTensor."""
        self._device = Device()
        self._spec = TensorSpec(DType.uint8, 0)
        self._name = None
        self._data = null_ptr[UInt8]()
        self._device_memory_impl_ptr = null_ptr[NoneType]()

    @implicit
    def __init__(out self, var device_tensor: DeviceTensor):
        """Creates AnyTensor from a DeviceTensor.

        Args:
            device_tensor: DeviceTensor to construct AnyTensor from.
        """
        self._device = device_tensor.device()
        self._spec = device_tensor.spec.copy()
        self._name = device_tensor.name()
        self._data = device_tensor.unsafe_ptr()
        var tmp = device_tensor^
        var tmp_dm = tmp._storage^
        tmp._storage = DeviceMemory()  # Make sure the destructor can run.
        _ = tmp^
        self._device_memory_impl_ptr = tmp_dm^._steal_impl_ptr()


    @always_inline
    def copy(self) -> Self:
        """Explicit copies are unsupported for this resource type in the
        Mojo 1.0 port (the 25.3 original trapped at compile time)."""
        abort("copy() is not supported on this type")

    def __init__(out self, *, deinit existing: Self):
        """Move constructor for AnyTensor.

        Args:
            existing: Instance to move from.
        """
        self._device = existing._device^
        self._spec = existing._spec^
        self._name = existing._name^
        self._data = existing._data
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr

    @implicit
    def __init__[
        type: DType, rank: Int
    ](out self, var tensor: Tensor[type, rank]) raises:
        """Creates AnyTensor from a Tensor.

        Args:
            tensor: Tensor to construct AnyTensor from.
        """
        self = Self(tensor^.to_device_tensor())

    def get_rank(self) -> Int:
        """Gets rank of the tensor.

        Returns:
            Rank of the tensor.
        """
        return self._spec.rank()

    def spec(self) -> TensorSpec:
        """Gets the spec of the tensor.

        Returns:
            Spec of the tensor.
        """
        return self._spec

    def _steal_ptr(var self) -> UnsafePointer[UInt8]:
        var ptr = self._data
        self._data = null_ptr[UInt8]()
        return ptr

    def to_device_tensor(var self) raises -> DeviceTensor:
        """Consumes this AnyTensor and converts it into a device tensor.

        Returns:
            DeviceTensor representation of AnyTensor.
        """
        var spec = self._spec.copy()
        return DeviceTensor(DeviceMemory(self^), spec)

    def to_tensor[
        type: DType, rank: Int
    ](var self) raises -> Tensor[type, rank]:
        """Consumes this anytensor and convert it into a tensor.

        Parameters:
            type: Type of tensor.
            rank: Rank of tensor.

        Returns:
            Tensor representation of AnyTensor.
        """
        return self^.to_device_tensor().to_tensor[type, rank]()

    def take(mut self) raises -> Self:
        """The returned value takes self's resources and replaces them with default
        initialized values.

        Returns:
            Newly constructed anytensor that takes storage from this.
        """
        var tmp = Self()
        swap(self, tmp)
        return tmp.copy()

    def __deinit__(deinit self):
        """Destructor for AnyTensor."""
        _ = DeviceMemory(
            self._device_memory_impl_ptr, self._spec.bytecount(), self._device
        )

    @no_inline
    def __str__(self) -> String:
        """Gets the tensor as a string.

        Returns:
          A compact string of the tensor.
        """

        return String(self)

    def write_to[W: Writer](self, mut writer: W):
        """
        Formats this Tensor to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write("Tensor(")

        @parameter
        def write_dtype_and_shape():
            writer.write("dtype=")
            writer.write(self._spec.dtype())
            writer.write(", ")
            writer.write("shape=")
            for i in range(self.get_rank()):
                if i > 0:
                    writer.write("x")
                writer.write(self._spec.shape()[i])

        var device_str = String(self._device)
        if "cpu" not in device_str:
            writer.write("<Unable to print device tensor>, ")
            writer.write(device_str)
            writer.write(", ")
            write_dtype_and_shape()
            writer.write(")")
            return

        # Mojo 1.0 port: element-level serialization relied on
        # std.utils._serialize and DType._dispatch_custom, which are gone.
        # Summarize instead.
        write_dtype_and_shape()
        writer.write(")")


@fieldwise_init
struct _CMojoValue(TrivialRegisterPassable, ImplicitlyCopyable):
    var _ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    comptime _destroy_func_type = def (UnsafePointer[NoneType, MutUntrackedOrigin]) thin -> None
    var _destroy_func: Self._destroy_func_type

    def __init__(out self):
        self._ptr = null_ptr[NoneType]()
        self._destroy_func = Self._destroy_pointee_wrapper[NoneType]

    def __init__[T: Movable & Deinitable](out self, ptr: UnsafePointer[T, MutUntrackedOrigin]):
        self._ptr = ptr.bitcast[NoneType]()
        self._destroy_func = Self._destroy_pointee_wrapper[T]

    @staticmethod
    def _destroy_pointee_wrapper[T: Deinitable](ptr: UnsafePointer[NoneType, MutUntrackedOrigin]):
        ptr.bitcast[T]().destroy_pointee()

    @staticmethod
    def _no_op_destructor[T: AnyType](ptr: UnsafePointer[NoneType]):
        pass

    @staticmethod
    def _free(ptr: UnsafePointer[NoneType, MutUntrackedOrigin]):
        external_call["KGEN_CompilerRT_MojoValueFreeBuffer", NoneType](ptr)

    def destroy(self):
        if Int(self._ptr) != 0:
            self._destroy_func(self._ptr)
            self._free(self._ptr)


struct AnyMojoValue(Copyable, Movable):
    """Type erased representation of a mojo object. This is useful for passing
    opaque type as input for graph executution.

    CAUTION: Experimental API.
    """

    comptime c_type = _CMojoValue
    """Internal representation of Mojo object."""

    var _impl: Self.c_type

    def __init__(out self):
        """Default constructor for MojoValue."""
        self._impl = _CMojoValue()

    @implicit
    def __init__(out self, impl: _CMojoValue):
        self._impl = impl.copy()

    def __init__[T: Movable & Deinitable](out self, var val: T):
        """Creates Type erased Mojo Value from T.

        Args:
            val: Object to type erase.
        """
        var ptr = external_call[
            "KGEN_CompilerRT_MojoValueAllocateBuffer", UnsafePointer[T, MutUntrackedOrigin]
        ](sizeof[T](), alignof[T]())
        ptr.init_pointee_move(val^)
        self._impl = _CMojoValue(ptr)


    @always_inline
    def copy(self) -> Self:
        """Explicit copies are unsupported for this resource type in the
        Mojo 1.0 port (the 25.3 original trapped at compile time)."""
        abort("copy() is not supported on this type")

    def __init__(out self, *, deinit existing: Self):
        """Move constructor for AnyMojoValue.

        Args:
            existing: Instance to move from.
        """
        self._impl = existing._impl.copy()

    def take(mut self) -> Self:
        """Returns the current value and initializes this object to default
        state.

        Returns:
            An instance of AnyMojoValue.
        """
        var tmp = Self()
        swap(tmp, self)
        return tmp^

    def release(var self) -> Self.c_type:
        """Release the underlying Mojo Value pointer. Caller is responsible for
        destroying the object."""
        var impl = exchange(self._impl, _CMojoValue())
        return impl

    def to[T: Movable](var self) -> T:
        """Consume this object and produces an instance of T. This doesn't do
        any type check and assumes this AnyMojoValue was created from T.

        Returns:
            Instance of type T.
        """
        var value = self._impl._ptr.bitcast[T]().take_pointee()
        self._impl._destroy_func = _CMojoValue._no_op_destructor[T]
        return value^

    def __deinit__(deinit self):
        """Destructor for AnyMojoValue."""
        self._impl.destroy()


@fieldwise_init
struct AnyMemory(Copyable, Movable, Writable):
    """A generic representation which can either be a Driver Tensor or Mojo object.
    """

    var _value: Variant[AnyTensor, AnyMojoValue]

    def __init__(out self):
        "Default constructor for AnyMemory."
        self._value = AnyMojoValue()

    @implicit
    def __init__(out self, var device_tensor: DeviceTensor):
        """Creates AnyMemory from a DeviceTensor.

        Args:
            device_tensor: DeviceTensor to construct AnyMemory from.
        """
        self._value = AnyTensor(device_tensor^)

    @implicit
    def __init__[
        type: DType, rank: Int
    ](out self, var tensor: Tensor[type, rank]) raises:
        """Creates AnyMemory from a Tensor.

        Args:
            tensor: Tensor to construct AnyMemory from.
        """
        self._value = AnyTensor(tensor^)

    @implicit
    def __init__(out self, var tensor: AnyTensor):
        """Creates AnyMemory from a AnyTensor.

        Args:
            tensor: AnyTensor to construct AnyMemory from.
        """
        self._value = tensor^

    @implicit
    def __init__(out self, var value: AnyMojoValue):
        """Creates AnyMemory from AnyMojoValue.

        Args:
            value: AnyMojoValue to construct AnyMemory from.
        """
        self._value = value^

    def is_tensor(self) -> Bool:
        """Check whether this contains a tensor.

        Returns:
            True if contains tensor.
        """
        return self._value.isa[AnyTensor]()

    def take_tensor(mut self) raises -> AnyTensor:
        """Take tensor from object. Further access to this object is
            undefined behavior.

        Returns:
            The tensor inside the memory as AnyTensor.
        """
        return self._value[AnyTensor].take()

    def take(mut self) -> Self:
        """The returned value takes self's resources and replaces them with
        default initialized values.

        Returns:
            Newly constructed AnyMemory that takes storage from this.
        """
        var tmp = Self()
        swap(tmp, self)
        return tmp^

    def to_device_tensor(var self) raises -> DeviceTensor:
        """Consume this object and produces and instance of DeviceTensor.
        Only valid if this was created from DeviceTensor.

        Returns:
            DeviceTensor representation of AnyMemory.
        """
        var tmp = self^
        return tmp.take_tensor().to_device_tensor()

    def to[T: Movable](var self) -> T:
        """Consume this object and produces an instance of T. This doesn't do
        any type check beyond whether this is a AnyTensor or not,
        and if not assume this was created from T.

        Returns:
            An instance of type T.
        """
        var tmp = self^
        var value = tmp.take_value()
        return value.to[T]()

    def take_value(mut self) -> AnyMojoValue:
        """Take value from object. Further access to this object is undefined
        behavior.

        Returns:
            The value inside the memory as AnyMojoValue.
        """
        return self._value[AnyMojoValue].take()

    @no_inline
    def __str__(self) -> String:
        """Gets this value as a string."""
        return String(self)

    def write_to[W: Writer](self, mut writer: W):
        """
        Formats the string representation of this value to the provided
        Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        if self._value.isa[AnyTensor]():
            return writer.write(self._value[AnyTensor])
        else:
            # TODO: Implement print for AnyMojoValue.
            return writer.write("AnyMojoValue")
