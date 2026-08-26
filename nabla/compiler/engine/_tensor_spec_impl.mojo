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

from nabla.compiler._utils import CString, call_dylib_func
from std.memory import UnsafePointer


@fieldwise_init
struct CTensorSpec(TrivialRegisterPassable, ImplicitlyCopyable):
    """Mojo representation of Engine's TensorSpec pointer.
    This doesn't free the memory on destruction.
    """

    comptime ptr_type = UnsafePointer[NoneType, MutUntrackedOrigin]
    var ptr: Self.ptr_type

    comptime FreeTensorSpecFnName = "M_freeTensorSpec"
    comptime GetDimAtFnName = "M_getDimAt"
    comptime GetRankFnName = "M_getRank"
    comptime GetNameFnName = "M_getName"
    comptime GetDTypeFnName = "M_getDtype"
    comptime IsDynamicallyRankedFnName = "M_isDynamicRanked"
    comptime GetDynamicRankValueFnName = "M_getDynamicRankValue"
    comptime GetDynamicDimensionValueFnName = "M_getDynamicDimensionValue"

    def get_dim_at(self, idx: Int, lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetDimAtFnName, self, idx)

    def get_rank(self, lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetRankFnName, self)

    def get_name(self, lib: DLHandle) -> String:
        var name = call_dylib_func[CString](lib, Self.GetNameFnName, self)
        return String(name)

    def get_dtype(self, lib: DLHandle) -> DType:
        return call_dylib_func[DType](lib, Self.GetDTypeFnName, self)

    def is_dynamically_ranked(self, lib: DLHandle) -> Bool:
        var is_dynamic = call_dylib_func[Int](
            lib, Self.IsDynamicallyRankedFnName, self
        )
        return is_dynamic == 1

    @staticmethod
    def get_dynamic_rank_value(lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetDynamicRankValueFnName)

    @staticmethod
    def get_dynamic_dimension_value(lib: DLHandle) -> Int:
        return call_dylib_func[Int](lib, Self.GetDynamicDimensionValueFnName)

    def free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeTensorSpecFnName, self)
