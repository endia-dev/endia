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

from nabla.compiler._dlhandle import DLHandle

from nabla.compiler._utils import call_dylib_func, get_lib_path_from_cfg
from nabla.compiler.tensor import TensorSpec
from std.memory import ArcPointer, UnsafePointer

from ._status import _CStatus


struct ManagedDLHandle(Movable):
    var lib: DLHandle

    @implicit
    def __init__(out self, lib: DLHandle):
        self.lib = lib

    def __init__(out self, *, deinit existing: Self):
        self.lib = existing.lib

    def get_handle(self) -> DLHandle:
        return self.lib

    def __deinit__(deinit self):
        if self.lib:
            self.lib.close()


@fieldwise_init
struct DriverLibrary(Copyable, Movable):
    var lib: ArcPointer[ManagedDLHandle]

    comptime device_type = UnsafePointer[NoneType, MutUntrackedOrigin]
    comptime device_memory_type = UnsafePointer[NoneType, MutUntrackedOrigin]

    comptime destroy_device_fn_sig = def (Self.device_type) thin abi("C") -> None
    var destroy_device_fn: Self.destroy_device_fn_sig

    comptime create_cpu_device_fn_sig = def (Int, _CStatus) thin abi("C") -> Self.device_type
    var create_cpu_device_fn: Self.create_cpu_device_fn_sig

    comptime create_accelerator_device_fn_sig = def (
        Int, _CStatus
    ) thin abi("C") -> Self.device_type
    var create_accelerator_device_fn: Self.create_accelerator_device_fn_sig

    comptime copy_device_fn_sig = def (Self.device_type) thin abi("C") -> Self.device_type
    var copy_device_fn: Self.copy_device_fn_sig

    comptime free_device_data_fn_sig = def (
        Self.device_type, UnsafePointer[UInt8, MutUntrackedOrigin], _CStatus
    ) thin abi("C") -> None
    var free_device_data_fn: Self.free_device_data_fn_sig

    comptime get_device_desc_fn_sig = def (Self.device_type) thin abi("C") -> UnsafePointer[UInt8, MutUntrackedOrigin]
    var get_device_desc_fn: Self.get_device_desc_fn_sig

    comptime create_device_memory_fn_sig = def (
        UnsafePointer[TensorSpec, MutUntrackedOrigin], Self.device_type, _CStatus
    ) thin abi("C") -> Self.device_memory_type
    var create_device_memory_fn: Self.create_device_memory_fn_sig

    comptime destroy_device_memory_fn_sig = def (Self.device_memory_type) thin abi("C") -> None
    var destroy_device_memory_fn: Self.destroy_device_memory_fn_sig

    comptime copy_device_memory_fn_sig = def (
        Self.device_memory_type, Self.device_memory_type, _CStatus
    ) thin abi("C") -> None
    var copy_device_memory_fn: Self.copy_device_memory_fn_sig

    comptime get_data_fn_sig = def (Self.device_memory_type) thin abi("C") -> UnsafePointer[UInt8, MutUntrackedOrigin]
    var get_data_fn: Self.get_data_fn_sig

    comptime accelerator_count_fn_sig = def () thin abi("C") -> Int
    var accelerator_count_fn: Self.accelerator_count_fn_sig

    def __init__(out self) raises:
        var lib = DLHandle(_get_driver_path())
        self.destroy_device_fn = lib.get_function[Self.destroy_device_fn_sig](
            "M_destroyDevice"
        )
        self.create_cpu_device_fn = lib.get_function[
            Self.create_cpu_device_fn_sig
        ]("M_createCPUDevice")
        self.create_accelerator_device_fn = lib.get_function[
            Self.create_accelerator_device_fn_sig
        ]("M_createAcceleratorDevice")
        self.copy_device_fn = lib.get_function[Self.copy_device_fn_sig](
            "M_copyDevice"
        )
        self.free_device_data_fn = lib.get_function[
            Self.free_device_data_fn_sig
        ]("M_freeDeviceData")
        self.get_device_desc_fn = lib.get_function[Self.get_device_desc_fn_sig](
            "M_getDeviceDesc"
        )
        self.create_device_memory_fn = lib.get_function[
            Self.create_device_memory_fn_sig
        ]("M_createDeviceMemory")
        self.destroy_device_memory_fn = lib.get_function[
            Self.destroy_device_memory_fn_sig
        ]("M_destroyDeviceMemory")
        self.copy_device_memory_fn = lib.get_function[
            Self.copy_device_memory_fn_sig
        ]("M_copyDeviceMemory")
        self.get_data_fn = lib.get_function[Self.get_data_fn_sig]("M_getData")
        self.accelerator_count_fn = lib.get_function[
            Self.accelerator_count_fn_sig
        ]("M_getAcceleratorCount")
        self.lib = ArcPointer[ManagedDLHandle](lib)

    def get_handle(self) -> DLHandle:
        return self.lib[].get_handle()


def _get_driver_path() raises -> String:
    return get_lib_path_from_cfg(".driver_lib", "MAX Driver")
