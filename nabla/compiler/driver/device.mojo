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
# """
# Defines the types and functions to interact with hardware devices.

# For example, you can create a CPU device like this:

# ```mojo
# from nabla.compiler.driver import cpu

# def main():
#     device = cpu()
# ```
# """


from std.collections import Optional
from std.collections.string import StaticString
from std.pathlib import Path

from nabla.compiler._utils import null_ptr, call_dylib_func, get_lib_path_from_cfg
from nabla.compiler.tensor import TensorSpec
from std.memory import UnsafePointer

from ._driver_library import DriverLibrary
from ._status import Status, _CStatus
from .device_memory import DeviceMemory, DeviceTensor
from std.runtime.asyncrt import DeviceContextPtr


struct _CPUDescriptor:
    var numa_id: Int

    def __init__(out self, *, numa_id: Optional[Int] = None):
        self.numa_id = numa_id.value() if numa_id else -1


def _get_driver_path() raises -> String:
    return get_lib_path_from_cfg(".driver_lib", "MAX Driver")


struct _CDevice(TrivialRegisterPassable, ImplicitlyCopyable):
    var _ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    @implicit
    def __init__(out self, ptr: UnsafePointer[NoneType, MutUntrackedOrigin]):
        self._ptr = ptr

    def copy(self, lib: Optional[DriverLibrary]) -> Self:
        if not lib:
            return self
        return lib.value().copy_device_fn(self._ptr)

    def free_data(self, lib: DriverLibrary, data: UnsafePointer[UInt8]) raises:
        var status = Status(lib)
        lib.free_device_data_fn(self._ptr, data, status.impl)
        if status:
            raise String(status)

    def __eq__(self, other: Self) -> Bool:
        return self._ptr == other._ptr


# @deprecated("use gpu.host.DeviceContext() instead")
struct Device(Copyable, Movable, Writable):
    """Represents a logical instance of a device, for eg: CPU. This
    can be used to allocate and manage memory in a device's address space,
    and to compile and execute models and graphs on a device.
    """

    var _lib: Optional[DriverLibrary]
    var _cdev: _CDevice

    def __init__(out self):
        """Constructs a default initialized Device in a state that is only valid
        for deletion. Can be used to represent a 'moved from' state.


        Use cpu() or accelerator() to create a CPU or GPU Device.
        """

        self._lib = None
        self._cdev = _CDevice(null_ptr[NoneType]())

    def __init__(
        out self, lib: DriverLibrary, *, var owned_ptr: _CDevice
    ) raises:
        self._lib = Optional[DriverLibrary](lib.copy())
        self._cdev = owned_ptr

    def copy(self) -> Self:
        """Explicitly construct a copy of self (bumping a refcount on the
        underlying Device).

        Returns:
            A copy of this value.
        """
        var res = Self()
        res._lib = self._lib.copy()
        res._cdev = self._cdev.copy(self._lib)
        return res^

    def __init__(out self, *, deinit existing: Self):
        """Create a new Device and consume `existing`.

        Args:
            existing: Instance from which to move from.
        """
        self._lib = existing._lib^
        self._cdev = existing._cdev

    def allocate(
        self, spec: TensorSpec, name: Optional[String] = None
    ) raises -> DeviceTensor:
        """Creates tensor allocated in the Device's address space.

        Args:
            spec: TensorSpec descripting the shape and type of the tensor to allocate.
            name: An optional name for the DeviceTensor.
        Returns:
            DeviceTensor allocated in Device's address space.
        """

        return DeviceTensor(spec, self, name)

    def allocate(
        self, bytecount: Int, name: Optional[String] = None
    ) raises -> DeviceMemory:
        """Allocates a DeviceMemory object in the Device's address space.

        Args:
            bytecount: The size of the memory to allocate in bytes.
            name: An optional name for the DeviceMemory.

        Returns:
            A DeviceMemory object allocated in the Device's address space.
        """

        return DeviceMemory(bytecount, self, name)

    def unsafe_ptr(self) -> UnsafePointer[NoneType]:
        """Gets the underlying pointer to the Device.

        Returns:
          The underlying pointer of the Device.
        """
        return self._cdev._ptr

    def _free(self, data: UnsafePointer[UInt8]) raises:
        self._cdev.free_data(self._lib.value(), data)

    def __str__(self) -> String:
        """Returns a descriptor of the device.

        Returns:
            String representation of device.
        """
        return String(self)

    def write_to[W: Writer](self, mut writer: W):
        """
        Formats this Device to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write(
            String(
                unsafe_from_utf8_ptr=self._lib.value().get_device_desc_fn(
                    self._cdev._ptr
                )
            )
        )

    def __deinit__(deinit self):
        """Destroys the device.

        Note that any DeviceBuffer allocated on the Device will contain a reference
        to the Device, and the Device will only be de-allocated when all of its
        DeviceBuffers have also been destroyed.
        """
        if Int(self._cdev._ptr) == 0:
            return
        self._lib.value().destroy_device_fn(self._cdev._ptr)

    def __eq__(self, other: Self) -> Bool:
        """Check if `self` and `other` point to the same underlying Device.

        Args:
            other: Instance to compare against.
        Returns:
            True if they are the same logical device.
        """
        return self._cdev == other._cdev

    @staticmethod
    def wait_for(device: Device) raises:
        """Blocks until all enqueued, asynchronous calls on the device have completed.
        """
        var device_context = call_dylib_func[DeviceContextPtr](
            device._lib.value().get_handle(), "M_getDeviceContext", device._cdev
        )

        device_context[].synchronize()


# @deprecated('use gpu.host.DeviceContext(api="cpu") instead')
def cpu() raises -> Device:
    """Creates a CPU Device.

    Returns:
        A logical device representing CPU.
    """
    var lib = DriverLibrary()
    var descriptor = _CPUDescriptor()
    var status = Status(lib)
    var device = lib.create_cpu_device_fn(descriptor.numa_id, status.impl)
    if status:
        raise String(status)
    return Device(lib, owned_ptr=device)
