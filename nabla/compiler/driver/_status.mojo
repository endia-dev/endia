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


from nabla.compiler._utils import CString, call_dylib_func
from std.memory import UnsafePointer

from ._driver_library import DriverLibrary


@fieldwise_init
struct _CStatus(TrivialRegisterPassable, ImplicitlyCopyable):
    var _ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    def is_error(self, lib: DriverLibrary) -> Bool:
        comptime is_error_func = "M_isError"
        return call_dylib_func[Int](lib.get_handle(), is_error_func, self) != 0

    def get_error(self, lib: DriverLibrary) -> String:
        comptime get_error_func = "M_getError"
        var err = call_dylib_func[CString](
            lib.get_handle(), get_error_func, self
        )
        return String(err)

    def free(self, lib: DriverLibrary):
        comptime free_func = "M_deleteStatus"
        call_dylib_func(lib.get_handle(), free_func, self)


struct Status(Writable):
    var impl: _CStatus
    var lib: DriverLibrary

    @implicit
    def __init__(out self, lib: DriverLibrary):
        self.impl = call_dylib_func[_CStatus](lib.get_handle(), "M_newStatus")
        self.lib = lib.copy()

    def __bool__(self) -> Bool:
        return self.impl.is_error(self.lib)

    def __str__(self) -> String:
        return self.impl.get_error(self.lib)

    def __deinit__(deinit self):
        self.impl.free(self.lib)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())
