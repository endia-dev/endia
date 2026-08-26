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
"""Defines tensor type, which is an owned, indexible buffer allocated on a given
device.

For example, a tensor can be created and used like this:

```mojo
from nabla.compiler.driver import Tensor
from nabla.compiler.tensor import TensorShape

def main():
    tensor = Tensor[DType.float32, rank=2](TensorShape(1, 2))
    tensor[0, 0] = 1.0
```

"""

from std.buffer.dimlist import DimList
from std.collections import InlineArray, Optional

from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout
from nabla.compiler._tensor_utils import _indexing
from nabla.compiler.tensor import Tensor as OldTensor
from nabla.compiler.tensor import TensorShape, TensorSpec
from std.memory import UnsafePointer

from std.utils import IndexList

from ._utils import _convert_from
from .device import Device, DeviceMemory, DeviceTensor
from nabla.compiler._utils import null_ptr
from std.os import abort


struct Tensor[type: DType, rank: Int](
    Copyable, Movable, Equatable, Writable
):
    """An owned, indexible buffer type."""

    var _ptr: UnsafePointer[Scalar[Self.type], MutUntrackedOrigin]
    var _spec: RuntimeTensorSpec[Self.type, Self.rank]
    var _strides: IndexList[Self.rank]
    var _device: Device
    var name: Optional[String]

    # TODO: We should be able to hold DeviceMemory here. Revisit
    # after DeviceMemory/DeviceTensor work.
    # this is needed because DeviceMemory may have a custom free
    # function set on the cpp side.
    var _device_memory_impl_ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    comptime layout_tensor = LayoutTensor[
        type,
        Layout(
            IntTuple(DimList.create_unknown[rank]()),
            IntTuple(DimList.create_unknown[rank]()),
        ),
        MutableAnyOrigin,
    ]
    """The corresponding layout tensor type which acts as a structured view
    into the underlying data."""

    def __init__(out self) raises:
        """Default constructor for Tensor. Accessing the elements of default
        constructed tensor is undefined behavior.
        """
        self._ptr = null_ptr[Scalar[Self.type]]()
        self._spec = RuntimeTensorSpec[Self.type, Self.rank](IndexList[Self.rank]())
        self._strides = IndexList[Self.rank]()
        self._device = Device()
        self.name = None
        self._device_memory_impl_ptr = null_ptr[NoneType]()

    def __init__(out self, *, var device_tensor: DeviceTensor) raises:
        """Creates a tensor from DeviceTensor.

        Args:
            device_tensor: DeviceTensor to create tensor from.
        """
        self._device = device_tensor.device()
        self.name = device_tensor.name()
        self._spec = device_tensor.spec
        self._strides = _indexing._row_major_strides(self._spec.shape)
        self._ptr = device_tensor.unsafe_ptr().bitcast[Scalar[Self.type]]()
        var tmp = device_tensor._storage^
        device_tensor._storage = DeviceMemory()
        self._device_memory_impl_ptr = tmp^._steal_impl_ptr()

    def __init__(
        out self, shape: TensorShape, device: Optional[Device] = None
    ) raises:
        """Creates tensor with given shape on the given device. If device is
        not given tensor will be created on cpu.

        Args:
            shape: Shape of the tensor.
            device: Device on which tensor is to be allocated.
        """
        var spec = TensorSpec(Self.type, shape)
        var dev = device.value().copy() if device else cpu()
        var dt = dev.allocate(spec)
        self = Self(device_tensor=dt.copy())

    def __init__(out self, tensor: OldTensor[Self.type]) raises:
        """Converts max.tensor to max.driver.Tensor. This creates tensor on
        the CPU.

        Args:
            tensor: Tensor to copy from.
        """
        self = _convert_from[rank = Self.rank](tensor)

    def __init__(out self, *, deinit existing: Self):
        """Move constructor for Tensor.

        Args:
            existing: Instance to move from.
        """
        self._ptr = existing._ptr
        self._spec = existing._spec
        self._strides = existing._strides
        self._device = existing._device^
        self.name = existing.name^
        self._device_memory_impl_ptr = existing._device_memory_impl_ptr


    @always_inline
    def copy(self) -> Self:
        """Explicit copies are unsupported for this resource type in the
        Mojo 1.0 port (the 25.3 original trapped at compile time)."""
        abort("copy() is not supported on this type")

    def spec(self) -> RuntimeTensorSpec[Self.type, Self.rank]:
        """Gets the spec of tensor.

        Returns
            Spec of the tensor.
        """
        return self._spec

    @always_inline
    def __getitem__(mut self, *indices: Int) -> ref [self] Scalar[type]:
        """Gets the value at the specified indices.

        Args:
          indices: The indices of the value to retrieve.

        Returns:
          The value at the specified indices.
        """
        debug_assert(
            len(indices) == rank, "mismatch between requested index and rank"
        )

        @always_inline
        @parameter
        def _is_cpu() -> Bool:
            return "cpu" in String(self._device)

        debug_assert[_is_cpu](
            "Cannot index into non-CPU Tensor from host",
        )

        var offset = _indexing._dot_prod(indices, self._strides)
        return self._ptr[offset]

    @always_inline
    def to_layout_tensor(
        self,
        out result: Self.layout_tensor,
    ) raises:
        """Returns a view of the tensor conforming to given slices. If given
        a single slice `:` the view would point to the entire tensor. The caller
        is responsible to make sure tensor outlives the returned slice.

        Args:

        Returns:
            View of the tensor according to given slices.
        """
        return type_of(result)(
            self.unsafe_ptr(),
            type_of(result.runtime_layout)(self._spec.shape, self._strides),
        )

    def _steal_ptr(var self) -> UnsafePointer[Scalar[type]]:
        var tmp = self._ptr
        self._ptr = null_ptr[Scalar[Self.type]]()
        return tmp

    def _get_device(self) -> Device:
        return self._device.copy()

    def to_device_tensor(var self) raises -> DeviceTensor:
        """Converts the tensor to a DeviceTensor.

        Returns:
            DeviceTensor pointing to the memory var by tensor.
        """
        var spec = self.spec()
        return DeviceTensor(DeviceMemory(self^), TensorSpec(spec))

    def __deinit__(deinit self):
        """Destructor for the tensor."""
        _ = DeviceMemory(
            self._device_memory_impl_ptr,
            self.spec().bytecount(),
            self._device,
        )

    def unsafe_ptr[__type: DType = Self.type](self) -> UnsafePointer[Scalar[__type], MutUntrackedOrigin]:
        """Gets a pointer to the underlying memory.

        Note: The caller is responsible for ensuring that the returned pointer
        is not used after it's owner is last used.

        Parameters:
            __type: If given the pointer will be rebound to this type. Defaulted
                    to type of tensor.

        Returns:
           Pointer to the beginning of tensor data.
        """
        return rebind[UnsafePointer[Scalar[__type], MutUntrackedOrigin]](self._ptr)

    def take(mut self) raises -> Self:
        """Takes self's resources and replaces them with default
        initialized values.

        Returns:
            An instance of tensor.
        """
        var tmp = Self()
        swap(tmp, self)
        return tmp

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
            writer.write(Self.type)
            writer.write(", ")
            writer.write("shape=")
            for i in range(Self.rank):
                if i > 0:
                    writer.write("x")
                writer.write(self._spec.shape[i])

        var device_str = String(self._device)
        if "cpu" not in device_str:
            writer.write("<Unable to print device tensor>, ")
            writer.write(device_str)
            writer.write(", ")
            write_dtype_and_shape()
            writer.write(")")
            return

        # Mojo 1.0 port: element serialization used std.utils._serialize,
        # which is gone. Summarize instead.
        write_dtype_and_shape()
        writer.write(")")

    def move_to(var self, device: Device) raises -> Self:
        """Returns self if already allocated on device, otherwise copy the contents
        of self to device.

        Args:
            device: The Device of the returned buffer.

        Returns:
            Instance of Tensor allocated on given device.
        """
        return self^.to_device_tensor().move_to(device).to_tensor[type, rank]()

    def __eq__(self, other: Self) -> Bool:
        """Check if two tensors are equal. Note that only host tensors can be
        compared. If either tensor is on an accelerator device, the result is False.

        Args:
            other: The tensor to compare with.

        Returns:
            True if the tensors have the same shape and all elements are equal, False otherwise.
        """

        # First, check that both tensors on on the host device. We can't compare
        # tensors on accelerator devices.
        if not (
            "cpu" in String(self._device) and "cpu" in String(other._device)
        ):
            return False

        # Next check if they point to the same memory
        if self._ptr == other._ptr:
            return True

        # Check if shapes match
        if self._spec.shape != other._spec.shape:
            return False

        # Now compare element by element
        var self_ptr = self.unsafe_ptr()
        var other_ptr = other.unsafe_ptr()
        var size = self._spec.shape[0] * self._spec.shape[1]
        for i in range(size):
            if self_ptr[i] != other_ptr[i]:
                return False
        return True

    def __ne__(self, other: Self) -> Bool:
        """Check if two tensors are not equal. Note that only host tensors can be
        compared. If either tensor is on an accelerator device, the result is True.

        Args:
            other: The tensor to compare with.

        Returns:
            True if the tensors are not equal, False otherwise.
        """
        return not self == other
