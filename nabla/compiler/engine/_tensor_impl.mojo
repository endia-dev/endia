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

from std.sys import sizeof
from nabla.compiler._dlhandle import DLHandle

from std.buffer import NDBuffer
from nabla.compiler._utils import null_ptr, call_dylib_func, exchange
from nabla.compiler.tensor import Tensor
from std.memory import UnsafePointer, memcpy
from std.memory.unsafe import bitcast
from std.python import Python, PythonObject
from std.collections.string import StaticString

from ._tensor_spec_impl import CTensorSpec
from .session import InferenceSession
from .tensor_spec import TensorSpec
import tensor


struct CTensor(TrivialRegisterPassable, ImplicitlyCopyable):
    """Represents AsyncTensor ptr from Engine."""

    var ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    comptime GetTensorNumElementsFnName = "M_getTensorNumElements"
    comptime GetTensorDTypeFnName = "M_getTensorType"
    comptime GetTensorDataFnName = "M_getTensorData"
    comptime GetTensorSpecFnName = "M_getTensorSpec"
    comptime FreeTensorFnName = "M_freeTensor"

    @implicit
    def __init__(out self, ptr: UnsafePointer[NoneType, MutUntrackedOrigin]):
        self.ptr = ptr

    def size(self, lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetTensorNumElementsFnName, self)

    def dtype(self, lib: DLHandle) -> DType:
        return call_dylib_func[DType](lib, Self.GetTensorDTypeFnName, self)

    def unsafe_ptr(self, lib: DLHandle) -> UnsafePointer[NoneType, MutUntrackedOrigin]:
        return call_dylib_func[UnsafePointer[NoneType, MutUntrackedOrigin]](
            lib, Self.GetTensorDataFnName, self
        )

    def get_tensor_spec(
        self, lib: DLHandle, var session: InferenceSession
    ) -> EngineTensorSpec:
        var spec = call_dylib_func[CTensorSpec](
            lib, Self.GetTensorSpecFnName, self
        )
        return EngineTensorSpec(spec, lib, session^)

    def free(self, lib: DLHandle):
        """
        Free the status ptr.
        """
        call_dylib_func(lib, Self.FreeTensorFnName, self)


struct EngineTensor(Sized):
    var ptr: CTensor
    var lib: DLHandle
    var session: InferenceSession

    def __init__(
        out self,
        ptr: CTensor,
        lib: DLHandle,
        var session: InferenceSession,
    ):
        self.ptr = ptr
        self.lib = lib
        self.session = session^

    def __init__(out self, *, deinit existing: Self):
        self.ptr = exchange[CTensor](existing.ptr, null_ptr[NoneType]())
        self.lib = existing.lib
        self.session = existing.session^

    def __len__(self) -> Int:
        return self.ptr.size(self.lib)

    def unsafe_ptr(self) -> UnsafePointer[NoneType, MutUntrackedOrigin]:
        return self.ptr.unsafe_ptr(self.lib)

    def data[type: DType](self) raises -> UnsafePointer[Scalar[type], MutUntrackedOrigin]:
        var ptr = self.unsafe_ptr()
        return ptr.bitcast[Scalar[type]]()

    def dtype(self) -> DType:
        return self.ptr.dtype(self.lib)

    def spec(self) raises -> TensorSpec:
        return self.ptr.get_tensor_spec(
            self.lib, self.session
        ).get_as_tensor_spec()

    def buffer[type: DType](self) raises -> NDBuffer[type, 1, MutableAnyOrigin]:
        return NDBuffer[type, 1](self.data[type](), len(self))

    def buffer(self) -> NDBuffer[DType.invalid, 1, MutableAnyOrigin]:
        return NDBuffer[DType.invalid, 1](
            rebind[UnsafePointer[Scalar[DType.invalid]]](self.unsafe_ptr()),
            len(self) * tensor._dtype_bytes(self.dtype()),
        )

    def tensor[type: DType](self) raises -> Tensor[type]:
        var tensor = Tensor[type](self.spec())
        memcpy(
            dest=tensor.unsafe_ptr(),
            src=self.data[type](),
            count=len(self),
        )
        return tensor^

    def __deinit__(deinit self):
        self.ptr.free(self.lib)
        _ = self.session^


@fieldwise_init
struct _Numpy(RegisterPassable, ImplicitlyCopyable):
    var np: PythonObject

    def __init__(out self) raises:
        self.np = Python.import_module("numpy")

    def __getattr__(self, attr: StaticString) raises -> PythonObject:
        return self.np.__getattr__(String(attr))
