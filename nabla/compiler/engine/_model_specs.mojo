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

from std.collections.string import StaticString
from nabla.compiler._dlhandle import DLHandle

from nabla.compiler._utils import null_ptr, CString, call_dylib_func, exchange
from std.memory import UnsafePointer

from ._compilation import CCompiledModel
from ._status import Status


@fieldwise_init
struct CTensorNameArray(TrivialRegisterPassable, ImplicitlyCopyable):
    """Mojo representation of Engine's TensorArray pointer.
    This doesn't free the memory on destruction.
    """

    var ptr: UnsafePointer[NoneType]

    comptime FreeTensorNameArrayFnName = "M_freeTensorNameArray"
    comptime GetTensorNameAtFnName = "M_getTensorNameAt"

    @implicit
    def __init__(out self, ptr: UnsafePointer[NoneType]):
        self.ptr = ptr

    def get_name_at(self, idx: Int, lib: DLHandle) raises -> String:
        if not self.ptr:
            raise "failed to get tensor name"
        var name = call_dylib_func[CString](
            lib, Self.GetTensorNameAtFnName, self, idx
        )
        return String(name)

    def free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeTensorNameArrayFnName, self)


struct TensorNamesIterator(Sized):
    var ptr: CTensorNameArray
    var current: Int
    var length: Int
    var lib: DLHandle

    def __init__(out self, ptr: CTensorNameArray, length: Int, lib: DLHandle):
        self.ptr = ptr
        self.current = 0
        self.length = length
        self.lib = lib

    def __next__(mut self) raises -> String:
        var next = self.ptr.get_name_at(self.current, self.lib)
        self.current += 1
        return next

    @always_inline
    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    def __len__(self) -> Int:
        if self.current == self.length:
            return 0
        return 1


struct TensorNames(Sized):
    var ptr: CTensorNameArray
    var lib: DLHandle
    var length: Int

    def __init__(
        out self,
        fn_name: String,
        ptr: CCompiledModel,
        length: Int,
        lib: DLHandle,
    ):
        var status = Status(lib)
        self.ptr = call_dylib_func[CTensorNameArray](
            lib,
            StaticString(ptr=fn_name.unsafe_ptr(), length=len(fn_name)),
            ptr,
            status.borrow_ptr(),
        )
        if status:
            print(String(status))
            self.ptr = null_ptr[NoneType]()
        self.length = length
        self.lib = lib

    def __init__(out self, *, deinit existing: Self):
        self.ptr = exchange[CTensorNameArray](
            existing.ptr, null_ptr[NoneType]()
        )
        self.length = existing.length
        self.lib = existing.lib

    def __getitem__(self, idx: Int) raises -> String:
        return self.ptr.get_name_at(idx, self.lib)

    def __len__(self) -> Int:
        return self.length

    def __deinit__(deinit self):
        self.ptr.free(self.lib)


struct InputTensorNames(Sized):
    """Collection of model input names."""

    var names: TensorNames

    comptime GetInputTensorNamesFnName = "M_getInputNames"

    def __init__(
        out self,
        ptr: CCompiledModel,
        length: Int,
        lib: DLHandle,
    ):
        self.names = TensorNames(
            Self.GetInputTensorNamesFnName, ptr, length, lib
        )

    def __init__(out self, *, deinit existing: Self):
        self.names = existing.names^

    def __getitem__(self, idx: Int) raises -> String:
        return self.names[idx]

    def __len__(self) -> Int:
        return len(self.names)


struct OutputTensorNames(Sized):
    """Collection of model output names."""

    var names: TensorNames

    comptime GetOutputTensorNamesFnName = "M_getOutputNames"

    def __init__(
        out self,
        ptr: CCompiledModel,
        length: Int,
        lib: DLHandle,
    ):
        self.names = TensorNames(
            Self.GetOutputTensorNamesFnName, ptr, length, lib
        )

    def __init__(out self, *, deinit existing: Self):
        self.names = existing.names^

    def __getitem__(self, idx: Int) raises -> String:
        return self.names[idx]

    def __len__(self) -> Int:
        return len(self.names)
