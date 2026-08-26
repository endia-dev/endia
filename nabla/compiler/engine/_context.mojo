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

from std.collections.optional import Optional
from nabla.compiler._dlhandle import DLHandle
from std.sys.param_env import is_defined

from nabla.compiler._utils import null_ptr, call_dylib_func, exchange
from nabla.compiler.driver import Device
from std.memory import UnsafePointer

from ._status import Status

comptime MODULAR_PRODUCTION = is_defined["MODULAR_PRODUCTION"]()


struct AllocatorType(TrivialRegisterPassable, ImplicitlyCopyable):
    var value: Int32
    # This needs to map M_AllocatorType enum on the C API side.
    comptime SYSTEM = Int32(0)
    comptime CACHING = Int32(1)

    @always_inline("nodebug")
    @implicit
    def __init__(out self, value: Int32):
        self.value = value

    @always_inline("nodebug")
    def __ne__(self, rhs: Int32) -> Bool:
        return self.value != rhs


struct CRuntimeConfig(TrivialRegisterPassable, ImplicitlyCopyable):
    var ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    comptime FreeRuntimeConfigFnName = "M_freeRuntimeConfig"
    comptime SetAllocatorTypeFnName = "M_setAllocatorType"
    comptime SetDeviceFnName = "M_setDevice"
    comptime SetMaxContextFnName = "M_setMaxContext"
    comptime SetAPILanguageFnName = "M_setAPILanguage"

    @implicit
    def __init__(out self, ptr: UnsafePointer[NoneType, MutUntrackedOrigin]):
        self.ptr = ptr

    def free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeRuntimeConfigFnName, self)

    def set_device(self, lib: DLHandle, device: Device):
        call_dylib_func(lib, Self.SetDeviceFnName, self, device._cdev)

    def set_api_language(self, lib: DLHandle, source: String):
        call_dylib_func(
            lib, Self.SetAPILanguageFnName, self, source.unsafe_ptr()
        )

    def set_allocator_type(self, lib: DLHandle, allocator_type: AllocatorType):
        call_dylib_func(lib, Self.SetAllocatorTypeFnName, self, allocator_type)

    def set_max_context(
        self, lib: DLHandle, max_context: UnsafePointer[NoneType, MutUntrackedOrigin]
    ) -> None:
        call_dylib_func(lib, Self.SetMaxContextFnName, self, max_context)


struct RuntimeConfig:
    var ptr: CRuntimeConfig
    var lib: DLHandle

    comptime NewRuntimeConfigFnName = "M_newRuntimeConfig"

    def __init__(
        out self,
        lib: DLHandle,
        device: Device,
        allocator_type: AllocatorType = AllocatorType.CACHING,
        max_context: OptionalPointer[NoneType, MutUntrackedOrigin] = {},
    ):
        self.ptr = call_dylib_func[CRuntimeConfig](
            lib, Self.NewRuntimeConfigFnName
        )

        if max_context:
            # `mojo-run` already has an existing `M::Context`.
            # Set the runtime config to reuse this existing context, rather
            # than trying to recreate a new one.
            self.ptr.set_max_context(lib, max_context.value())

        self.lib = lib

        if allocator_type != AllocatorType.CACHING:
            self.ptr.set_allocator_type(self.lib, allocator_type)

        self.ptr.set_device(self.lib, device)

        self.ptr.set_api_language(self.lib, "mojo")

    def __init__(out self, *, deinit existing: Self):
        self.ptr = exchange[CRuntimeConfig](
            existing.ptr, null_ptr[NoneType]()
        )
        self.lib = existing.lib

    def borrow_ptr(self) -> CRuntimeConfig:
        """
        Borrow the underlying C ptr.
        """
        return self.ptr

    def __deinit__(deinit self):
        self.ptr.free(self.lib)


struct CRuntimeContext(TrivialRegisterPassable, ImplicitlyCopyable):
    var ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    @implicit
    def __init__(out self, ptr: UnsafePointer[NoneType, MutUntrackedOrigin]):
        self.ptr = ptr

    comptime FreeRuntimeContextFnName = "M_freeRuntimeContext"

    def free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeRuntimeContextFnName, self)


struct RuntimeContext:
    var ptr: CRuntimeContext
    var lib: DLHandle

    comptime NewRuntimeContextFnName = "M_newRuntimeContext"
    comptime SetDebugPrintOptionsFnName = "M_setDebugPrintOptions"

    def __init__(out self, var config: RuntimeConfig, lib: DLHandle):
        var status = Status(lib)
        self.ptr = call_dylib_func[CRuntimeContext](
            lib,
            Self.NewRuntimeContextFnName,
            config.borrow_ptr(),
            status.borrow_ptr(),
        )
        if status:
            print(String(status))
            self.ptr = null_ptr[NoneType]()
        _ = config^
        self.lib = lib

    def __init__(out self, *, deinit existing: Self):
        self.ptr = exchange[CRuntimeContext](
            existing.ptr, null_ptr[NoneType]()
        )
        self.lib = existing.lib

    def borrow_ptr(self) -> CRuntimeContext:
        return self.ptr

    def __deinit__(deinit self):
        self.ptr.free(self.lib)
        _ = self.lib

    def set_debug_print_options(
        mut self,
        style: PrintStyle,
        precision: UInt,
        var output_directory: String,
    ):
        _ = call_dylib_func[CRuntimeContext](
            self.lib,
            Self.SetDebugPrintOptionsFnName,
            self.ptr,
            style.style,
            precision,
            output_directory.as_c_string_slice().unsafe_ptr(),
        )


@fieldwise_init
struct PrintStyle(TrivialRegisterPassable, ImplicitlyCopyable):
    var style: Int32

    comptime COMPACT = PrintStyle(0)
    comptime FULL = PrintStyle(1)
    comptime BINARY = PrintStyle(2)
    comptime NONE = PrintStyle(3)
