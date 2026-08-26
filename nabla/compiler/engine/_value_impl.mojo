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

from nabla.compiler._utils import call_dylib_func
from std.memory import UnsafePointer

from ._tensor_impl import CTensor

comptime CMojoVal = UnsafePointer[UInt8]


struct CValue(TrivialRegisterPassable, ImplicitlyCopyable):
    """Represents an AsyncValue pointer from Engine."""

    var ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    comptime _GetTensorFnName = "M_getTensorFromValue"
    comptime _GetBoolFnName = "M_getBoolFromValue"
    comptime _GetListFnName = "M_getListFromValue"
    comptime _TakeMojoValueFnName = "M_takeMojoValueFromValue"
    comptime _FreeValueFnName = "M_freeValue"

    @implicit
    def __init__(out self, ptr: UnsafePointer[NoneType, MutUntrackedOrigin]):
        self.ptr = ptr

    def get_c_tensor(self, lib: DLHandle) -> CTensor:
        """Get CTensor within value."""
        return call_dylib_func[CTensor](lib, Self._GetTensorFnName, self)

    def get_bool(self, lib: DLHandle) -> Bool:
        """Get bool within value."""
        return call_dylib_func[Bool](lib, Self._GetBoolFnName, self)

    def get_list(self, lib: DLHandle) -> CList:
        """Get list within value."""
        return call_dylib_func[CList](lib, Self._GetListFnName, self)

    def take_mojo_value(self, lib: DLHandle) -> CMojoVal:
        """Take ownership of mojo_val within value."""
        return call_dylib_func[CMojoVal](lib, Self._TakeMojoValueFnName, self)

    def free(self, lib: DLHandle):
        """Free value."""
        call_dylib_func(lib, Self._FreeValueFnName, self)


@fieldwise_init
struct CList(TrivialRegisterPassable, ImplicitlyCopyable):
    """Represents an AsyncList pointer from Engine."""

    var ptr: UnsafePointer[NoneType]

    comptime _GetSizeFnName = "M_getListSize"
    comptime _GetValueFnName = "M_getListValue"
    comptime _AppendFnName = "M_appendToList"
    comptime _FreeFnName = "M_freeList"

    @implicit
    def __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    def get_size(self, lib: DLHandle) -> Int:
        """Get number of elements in list."""
        return call_dylib_func[Int](lib, Self._GetSizeFnName, self)

    def get_value(self, lib: DLHandle, index: Int) -> CValue:
        """Get value by index in list."""
        return call_dylib_func[CValue](lib, Self._GetValueFnName, self, index)

    def append(self, lib: DLHandle, value: CValue):
        """Get value by index in list."""
        return call_dylib_func(lib, Self._AppendFnName, self, value)

    def free(self, lib: DLHandle):
        """Free list."""
        call_dylib_func(lib, Self._FreeFnName, self)
