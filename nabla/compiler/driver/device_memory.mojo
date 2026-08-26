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

from std.collections import Optional

from nabla.compiler.tensor import TensorShape, TensorSpec
from std.memory import UnsafePointer

from ._driver_library import DriverLibrary
from ._status import Status
from .anytensor import AnyTensor
from .device import Device, _CDevice
from .tensor import Tensor
from nabla.compiler._utils import null_ptr
from std.os import abort


trait DeviceBuffer:
    def copy_to(self, dev: Device, name: Optional[String]) raises -> Self:
        """Copies the contents of self into DeviceBuffer allocated on dev.
        Note: this function allocates memory on dev.

        Args:
            dev: The Device on which to allocate the new DeviceBuffer.
            name: Optional name of the new DeviceBuffer.

        Returns:
            Newly allocated DeviceBuffer containing a copy of self's contents.

        Raises:
            If the DeviceBuffer is backed by the same Device object as dev.
        """
        ...

    def copy_into(self, mut dst_memory: Self) raises:
        """Copies the contents of self into a preallocated DeviceBuffer.

        Args:
            dst_memory: The destination DeviceBuffer of the copy.
        """
        ...

    def move_to(var self, dev: Device) raises -> Self:
        """Returns self if already allocated on dev, otherwise copy the contents
        of self to dev.

        Args:
            dev: The Device of the returned buffer.

        Returns:
            A DeviceBuffer located in dev's address space.
        """
        ...

    def unsafe_ptr(self) -> UnsafePointer[UInt8, MutUntrackedOrigin]:
        """Returns a pointer to the DeviceBuffer's storage in device memory."""
        ...

    def device(self) -> Device:
        """Returns the Device on which the DeviceBuffer was allocated."""
        ...

    def bytecount(self) -> Int:
        """Returns the size of the DeviceBuffer in bytes."""
        ...


struct DeviceMemory(
    DeviceBuffer, Copyable, Movable, Writable
):
    """DeviceMemory is an owning buffer allocated on a (possibly non-CPU) Device.
    """

    var _impl_ptr: UnsafePointer[NoneType, MutUntrackedOrigin]
    var _device: Device
    var name: Optional[String]
    var num_bytes: Int

    def __init__(out self):
        """Constructs a DeviceMemory object in a state that is only valid for deletion.
        Can be used to represent a `moved from` state.
        """
        self = Self(
            null_ptr[NoneType](),
            0,
            Device(),
        )

    def __init__(
        out self,
        num_bytes: Int,
        device: Device,
        name: Optional[String] = None,
    ) raises:
        """Allocates DeviceMemory from the Device's address space.

        Args:
            num_bytes: Size of the DeviceMemory buffer to allocate in bytes.
            device: Device on which to perform the allocation.
            name: Optional name for the DeviceMemory.

        """
        self._device = device.copy()
        var tmp_spec = TensorSpec(DType.uint8, num_bytes)
        var status = Status(device._lib.value())
        # CAUTION: this assumes that TensorSpec is bitwise identical in mojo and cpp
        self._impl_ptr = device._lib.value().create_device_memory_fn(
            UnsafePointer(to=tmp_spec).unsafe_origin_cast[MutUntrackedOrigin](),
            self._device._cdev._ptr,
            status.impl,
        )
        if status:
            raise String(status)
        self.name = name
        self.num_bytes = num_bytes

    def __init__(
        out self,
        owned_impl_ptr: UnsafePointer[NoneType, MutUntrackedOrigin],
        num_bytes: Int,
        device: Device,
        name: Optional[String] = None,
    ):
        self._device = device.copy()
        self._impl_ptr = owned_impl_ptr
        self.name = name
        self.num_bytes = num_bytes


    @always_inline
    def copy(self) -> Self:
        """Explicit copies are unsupported for this resource type in the
        Mojo 1.0 port (the 25.3 original trapped at compile time)."""
        abort("copy() is not supported on this type")

    def __init__[
        type: DType, rank: Int
    ](out self, var tensor: Tensor[type, rank]) raises:
        """Creates a DeviceMemory from the existing `tensor` storage.

        Args:
            tensor: Tensor whose storage to use.
        """

        self._device = tensor._get_device()
        self.name = tensor.name
        self.num_bytes = tensor.spec().bytecount()
        self._impl_ptr = tensor._device_memory_impl_ptr
        tensor._device_memory_impl_ptr = null_ptr[NoneType]()

    def __init__(out self, var anytensor: AnyTensor) raises:
        """Creates a device tensor the existing `anytensor` storage.

        Args:
            anytensor: AnyTensor whose storage to use.

        """

        self._device = anytensor._device.copy()
        self.name = anytensor._name
        self._impl_ptr = anytensor._device_memory_impl_ptr
        self.num_bytes = anytensor._spec.bytecount()
        anytensor._device_memory_impl_ptr = null_ptr[NoneType]()

    def __deinit__(deinit self):
        """De-allocate and destroy the DeviceMemory.

        Note: this will also decrement the refcount on the Device used to allocate
        the DeviceMemory.
        """
        if Int(self._impl_ptr) == 0:
            return
        self._device._lib.value().destroy_device_memory_fn(self._impl_ptr)

    def bytecount(self) -> Int:
        """Returns the number of bytes in the DeviceMemory."""
        return self.num_bytes

    def get_device(self) -> Device:
        """Returns the device on which the DeviceMemory was allocated."""

        return self._device.copy()

    def device(self) -> Device:
        """Returns the device on which the DeviceMemory was allocated."""

        return self._device.copy()

    def __init__(out self, *, deinit existing: Self):
        self._impl_ptr = existing._impl_ptr
        self._device = existing._device^
        self.name = existing.name^
        self.num_bytes = existing.num_bytes

    def __str__(self) raises -> String:
        """Returns a description of the DeviceMemory."""
        return String(self)

    def write_to[W: Writer](self, mut writer: W):
        """
        Formats a description of the DeviceMemory to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write(
            "DeviceMemory(",
            self.name.value() if self.name else "",
            "," if self.name else "",
            self.get_device(),
            ",Bytecount(",
            self.bytecount(),
            "))",
        )

    def _steal_impl_ptr(var self) -> UnsafePointer[NoneType, MutUntrackedOrigin]:
        var tmp = self._impl_ptr
        self._impl_ptr = null_ptr[NoneType]()
        return tmp

    def _steal_ptr(var self) -> UnsafePointer[UInt8]:
        comptime func_name_take_data = "M_takeDataFromDeviceMemory"
        var take_data_func = self._device._lib.value().get_handle().get_function[
            def (UnsafePointer[NoneType]) thin abi("C") -> UnsafePointer[UInt8]
        ](
            func_name_take_data
        )
        var data = take_data_func(self._impl_ptr)
        # Extend lifetime of self to avoid the C funtion working on invalid pointer.
        _ = self^
        return data

    def copy_into(self, mut dst_memory: DeviceMemory) raises:
        """Copies the contents of self into preallocated DeviceMemory.

        Args:
            dst_memory: The destination DeviceMemory of the copy.
        """
        if self.bytecount() != dst_memory.bytecount():
            raise String(
                "source bytecount({}) does not match destination bytecount({})"
            ).format(self.bytecount(), dst_memory.bytecount())

        var status = Status(self._device._lib.value())

        self._device._lib.value().copy_device_memory_fn(
            dst_memory._impl_ptr, self._impl_ptr, status.impl
        )
        if status:
            raise String(status)

    def copy_to(
        self, dev: Device, name: Optional[String] = None
    ) raises -> DeviceMemory:
        """Copies the contents of self into DeviceMemory allocated on dev.
        Note: this function allocates memory on dev.

        Args:
            dev: The Device on which to allocate the new DeviceMemory.
            name: Optional name of the new DeviceMemory.

        Returns:
            Newly allocated DeviceMemory containing a copy of self's contents.

        Raises:
            If the DeviceMemory is backed by the same Device object as dev.
        """
        if dev == self._device:
            raise Error(self, "is already allocated on ", dev)

        var dst = dev.allocate(self.bytecount(), name)
        self.copy_into(dst)
        return dst^

    def move_to(var self, dev: Device) raises -> Self:
        """Returns self if already allocated on dev, otherwise copy the contents
        of self to dev.

        Args:
            dev: The Device on which the returned buffer is allocated.

        Returns:
            DeviceMemory located in dev's address space.
        """
        if dev == self._device:
            return self^
        else:
            return self.copy_to(dev)

    def unsafe_ptr(self) -> UnsafePointer[UInt8, MutUntrackedOrigin]:
        """Returns a pointer to the underlying device memory.

        Note: The caller is responsible for ensuring that the returned pointer
        is not used after its owner is last used.
        """

        return self._device._lib.value().get_data_fn(self._impl_ptr)

    def take(mut self) raises -> Self:
        """Takes and returns the contents of `self`, leaving `self` in an empty but destructible state.
        """
        var tmp = Self()
        swap(tmp, self)
        return tmp.copy()


struct DeviceTensor(
    DeviceBuffer, Copyable, Movable, Writable
):
    var _storage: DeviceMemory
    var spec: TensorSpec

    def __init__(
        out self, spec: TensorSpec, device: Device, name: Optional[String]
    ) raises:
        """Allocates a DeviceTensor in the Device's address space.

        Args:
            spec: TensorSpec describing the dtype and shape of the DeviceTensor.
            device: Device on which to perform the allocation.
            name: Optional name for the DeviceMemory.

        """
        self.spec = spec.copy()
        self._storage = DeviceMemory(
            spec.bytecount(),
            device,
            name,
        )

    def __init__(out self):
        """Constructs a DeviceTensor in a state that is only valid for deletion.
        Can be used to represent a `moved from` state.
        """
        self.spec = TensorSpec()
        self._storage = DeviceMemory()

    def __init__(out self, var storage: DeviceMemory, spec: TensorSpec) raises:
        """Constructs a DeviceTensor from an existing storage buffer and spec.

        Args:
            storage: The storage backing the DeviceTensor.
            spec: TensorSpec describing the type and shape of the DeviceTensor.
        """
        self._storage = storage^
        self.spec = spec.copy()

        if self.bytecount() != self.bytecount():
            raise "DeviceMemory size does not match DeviceTensor requirements"

    def copy_to(self, dev: Device, name: Optional[String] = None) raises -> Self:
        """Copies the contents of self into a DeviceTensor allocated on dev.
        Note: this function allocates memory on dev.

        Args:
            dev: The Device on which to allocate the new DeviceTensor.
            name: Optional name of the new DeviceTensor.

        Returns:
            Newly allocated DeviceTensor containing a copy of self's contents.

        Raises:
            If the DeviceTensor is backed by the same Device object as dev.
        """
        var t = Self(self._storage.copy_to(dev, name), self.spec)
        return t.copy()

    def copy_into(self, mut dst_tensor: Self) raises:
        """Copies the contents of self into a preallocated DeviceTensor.

        Args:
            dst_tensor: The destination DeviceTensor of the copy.
        """
        if dst_tensor.spec != self.spec:
            raise String(
                "source({}) and destination({}) specs do not match"
            ).format(self.spec, dst_tensor.spec)
        self._storage.copy_into(dst_tensor._storage)

    def move_to(var self, dev: Device) raises -> Self:
        """Returns self if already allocated on dev, otherwise copy the contents
        of self to dev.

        Args:
            dev: The Device on which the returned buffer is allocated.

        Returns:
            A DeviceTensor located in dev's address space.
        """
        if dev == self._storage._device:
            return self^
        else:
            return self.copy_to(dev)

    def unsafe_ptr(self) -> UnsafePointer[UInt8, MutUntrackedOrigin]:
        """Returns a pointer to the DeviceTensor's storage in device memory."""
        return self._storage.unsafe_ptr()

    def to_tensor[
        type: DType, rank: Int
    ](var self) raises -> Tensor[type, rank]:
        """Returns a Tensor created using the DeviceTensor's shape and storage.
        """
        if rank != self.spec.rank():
            raise "requested rank does not match existing rank"

        if type != self.spec.dtype():
            raise "requested dtype does not match existing type."

        return Tensor[type, rank](device_tensor=self^)

    def device(self) -> Device:
        """Returns the Device on which the DeviceTensor was allocated."""
        return self._storage.device()

    def name(self) -> Optional[String]:
        """Returns the name of the DeviceTensor."""
        return self._storage.name


    @always_inline
    def copy(self) -> Self:
        """Explicit copies are unsupported for this resource type in the
        Mojo 1.0 port (the 25.3 original trapped at compile time)."""
        abort("copy() is not supported on this type")

    def __init__(out self, *, deinit existing: Self):
        self._storage = existing._storage^
        self.spec = existing.spec^

    def __str__(self) raises -> String:
        """Returns a descriptor for the DeviceTensor."""
        return String(self)

    def write_to[W: Writer](self, mut writer: W):
        """
        Formats a description of the DeviceTensor to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write(
            "DeviceTensor(",
            self._storage.name.value() if self._storage.name else "",
            "," if self._storage.name else "",
            self.device(),
            ",Spec(",
            self.spec,
            "))",
        )

    def bytecount(self) -> Int:
        """Returns the number of bytes in the DeviceTensor."""
        return self.spec.bytecount()

    def take(mut self) -> Self:
        """Takes and returns the contents of `self`, leaving `self` in an empty but destructible state.
        """
        var tmp = Self()
        swap(tmp, self)
        return tmp
