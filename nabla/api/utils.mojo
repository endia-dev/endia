# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from std.memory import ArcPointer
from std.collections import Dict
import nabla.compiler as compiler
from nabla.engine.utils import TrafoMeta, GraphRepr, Callable
from nabla.api.array import Array
from nabla.api.functional import jit


comptime none: Int = -55555


@fieldwise_init
struct ExecutionContext(Copyable, ImplicitlyCopyable, Movable):
    var dict: ArcPointer[Dict[Int, ArcPointer[compiler.engine.Model]]]

    def __init__(out self):
        self.dict = ArcPointer(Dict[Int, ArcPointer[compiler.engine.Model]]())

    def __getitem__(self, key: Int) raises -> ArcPointer[compiler.engine.Model]:
        return self.dict[][key]

    def __setitem__(
        mut self, key: Int, value: ArcPointer[compiler.engine.Model]
    ) -> None:
        if key in self.dict[]:
            print("Warning: key-value pair alrey present in model cache.")
            return
        self.dict[][key] = value

    def __contains__(self, key: Int) -> Bool:
        return key in self.dict[]

    def clear(mut self) -> None:
        self.dict[].clear()


def xpr(callable: Callable) raises -> GraphRepr:
    return GraphRepr(callable)


def xpr(func: def (List[Array]) raises thin -> List[Array]) raises -> GraphRepr:
    return GraphRepr(jit(func))
