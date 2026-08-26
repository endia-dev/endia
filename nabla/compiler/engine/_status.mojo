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

from nabla.compiler._utils import null_ptr, CString, call_dylib_func, exchange
from std.memory import UnsafePointer


struct CStatus(TrivialRegisterPassable, ImplicitlyCopyable):
    """Represents Status ptr from Engine."""

    var ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    comptime IsErrorFnName = "M_isError"
    comptime GetErrorFnName = "M_getError"
    comptime FreeStatusFnName = "M_freeStatus"

    @implicit
    def __init__(out self, ptr: UnsafePointer[NoneType, MutUntrackedOrigin]):
        self.ptr = ptr

    def is_error(self, lib: DLHandle) -> Bool:
        """
        Check if status is error.

        Returns:
            True if error.
        """
        return call_dylib_func[Bool](lib, Self.IsErrorFnName, self)

    def get_error(self, lib: DLHandle) -> String:
        """
        Get Error String from Engine library.
        """
        var error = call_dylib_func[CString](lib, Self.GetErrorFnName, self)
        return String(error)

    def free(self, lib: DLHandle):
        """
        Free the status ptr.
        """
        call_dylib_func(lib, Self.FreeStatusFnName, self)


struct Status(Writable):
    var ptr: CStatus
    var lib: DLHandle

    comptime NewStatusFnName = "M_newStatus"

    @implicit
    def __init__(out self, lib: DLHandle):
        self.ptr = call_dylib_func[CStatus](lib, self.NewStatusFnName)
        self.lib = lib

    def __init__(out self, *, deinit existing: Self):
        self.ptr = exchange[CStatus](existing.ptr, null_ptr[NoneType]())
        self.lib = existing.lib

    def __bool__(self) -> Bool:
        """
        Check if status is error.

        Returns:
            True if error.
        """
        return self.ptr.is_error(self.lib)

    def __str__(self) -> String:
        """
        Get Error String.

        Returns:
            Error string if there is an error. Else empty.
        """
        if self:
            return self.ptr.get_error(self.lib)
        return ""

    def borrow_ptr(self) -> CStatus:
        """
        Borrow the underlying C ptr.
        """
        return self.ptr

    def __deinit__(deinit self):
        self.ptr.free(self.lib)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())
