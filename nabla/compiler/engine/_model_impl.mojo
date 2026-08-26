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

from ._compilation import CCompiledModel
from ._status import Status
from .session import InferenceSession


struct CModel(TrivialRegisterPassable, ImplicitlyCopyable):
    """Mojo representation of Engine's AsyncModel pointer.
    Useful for C inter-op.
    """

    var ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    comptime FreeModelFnName = "M_freeModel"
    comptime WaitForModelFnName = "M_waitForModel"

    @implicit
    def __init__(out self, ptr: UnsafePointer[NoneType, MutUntrackedOrigin]):
        self.ptr = ptr

    def await_model(self, lib: DLHandle) raises:
        var status = Status(lib)
        call_dylib_func(
            lib, Self.WaitForModelFnName, self.ptr, status.borrow_ptr()
        )
        if status:
            raise String(status)

    def free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeModelFnName, self)
