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

import std.math as math
from std.time import perf_counter
import nabla


def test_vmap() raises:
    def dot(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var res: List[nabla.Array] = [nabla.sum(args[0] * args[1], axis=[0])]
        return res.copy()

    def mv_prod(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var res = nabla.vmap(dot, [nabla.none, 1])(args)
        return res.copy()

    def mm_prod(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var res = nabla.vmap(mv_prod, [0, nabla.none])(args)
        return res.copy()

    def batched_matmul(args: List[nabla.Array]) raises -> List[nabla.Array]:
        var res = nabla.vmap(mm_prod, [0, nabla.none])([args[0], args[1]])[0]
        return [res]

    var batch_a = nabla.arange((2, 3, 4), DType.float32)
    var mat_b = nabla.arange((4, 5), DType.float32)

    print(nabla.xpr(batched_matmul)([batch_a, mat_b]))
    var res = batched_matmul([batch_a, mat_b])
    print(res[0])


def test_vmap2() raises:
    def vv(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return [nabla.sum(args[0] * args[1])]

    def mv(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return nabla.vmap(vv, [0, nabla.none])(args)

    def mm(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return nabla.vmap(mv, [nabla.none, 1], [1])(args)

    var a = nabla.arange((2, 3), DType.float32)
    var b = nabla.arange((3, 4), DType.float32)

    var res = mm([a, b])[0]
    print(res)


def test_vmap3() raises:
    def br_foo(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return [nabla.broadcast_to(args[0], (1, 3, 9))]

    var res = nabla.vmap(br_foo)([nabla.arange((2, 9), DType.float32)])[0]
    print(res)


def test_vmap4() raises:
    def dot(args: List[nabla.Array]) raises -> List[nabla.Array]:
        return [nabla.sum(args[0] * args[1], axis=[0])]

    var mv_prod = nabla.vmap(dot, [nabla.none, 1])
    var mm_prod = nabla.vmap(mv_prod, [0, nabla.none])
    var batched_matmul = nabla.vmap(mm_prod, [0, nabla.none])

    var batch_a = nabla.arange((2, 3, 4), DType.float32)
    var mat_b = nabla.arange((4, 5), DType.float32)

    var res = batched_matmul([batch_a, mat_b])[0]
    print(res)
