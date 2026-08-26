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

import nabla.compiler as compiler
from std.collections import Dict

from nabla.core.device_array import DeviceArray, ArrayImpl
from nabla.ops.utils import register_any_op, RuntimeInfo
from nabla.ops.view_ops import squeeze, broadcast_to

comptime BATCH_DIM_CTR = 0
comptime ORIGINALshape = 2
comptime AXES = 3
comptime KEEP_DIM = 4
comptime ORIGINAL_AXES = 5
comptime ACT_ON_BATCH_DIMS = 6


struct Sum:
    @staticmethod
    def maxpr(
        args: List[compiler.graph.Symbol], array: DeviceArray
    ) raises -> compiler.graph.Symbol:
        var axes = array.impl[].runtime_info[AXES].copy()
        var originalshape = array.impl[].runtime_info[ORIGINALshape].copy()
        var symbol = args[0]

        for i in range(len(axes)):
            var axis = axes[i]
            symbol = compiler.graph.ops.mean(symbol, axis) * originalshape[axis]

        return symbol

    @staticmethod
    def eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Sum"

    @staticmethod
    def vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var act_on_batch_dims = True if array.impl[].runtime_info[
            ACT_ON_BATCH_DIMS
        ][0] == 1 else False
        var originalshape = array.impl[].runtime_info[ORIGINALshape].copy()
        var target_shape = originalshape.copy()
        return [
            broadcast_to(
                tangent,
                target_shape,
                act_on_batch_dims=act_on_batch_dims,
                expand_dims=False,
            )
        ]

    @staticmethod
    def jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var runtime_info = array.impl[].runtime_info.copy()
        var act_on_batch_dims = True if runtime_info[ACT_ON_BATCH_DIMS][
            0
        ] == 1 else False
        var axes = array.impl[].runtime_info[ORIGINAL_AXES].copy()
        return sum(
            tangents[0],
            axes,
            keep_dim=True,
            act_on_batch_dims=act_on_batch_dims,
        )


def sum(
    arg: DeviceArray,
    _axis: List[Int] = List[Int](),
    keep_dim: Bool = False,
    act_on_batch_dims: Bool = False,
) raises -> DeviceArray:
    var batch_dim_ctr = arg.batch_dim_ctr()
    if act_on_batch_dims:
        batch_dim_ctr = 0

    if arg.batch_dim_ctr() == len(arg.shape()):
        return arg

    var arg_shape = arg.shape()[batch_dim_ctr:]
    var axes = _axis.copy()

    if len(axes) == 0:
        for i in range(-len(arg_shape), 0):
            axes.append(i)
    else:
        for i in range(len(axes)):
            axes[i] = axes[i] if axes[i] < 0 else -len(arg_shape) + axes[i]
    sort(axes)

    var target_shape = List(arg.shape()[:batch_dim_ctr])
    for i in range(-len(arg_shape), 0):
        if i not in axes:
            target_shape.append(arg_shape[len(arg_shape) + i])
        else:
            target_shape.append(1)

    if len(target_shape) == 0:
        target_shape.append(1)

    var runtime_info = RuntimeInfo(7)
    runtime_info[ORIGINAL_AXES] = _axis.copy()
    runtime_info[AXES] = axes.copy()
    runtime_info[ORIGINALshape] = List(arg_shape)
    runtime_info[KEEP_DIM] = [1] if keep_dim else [0]
    runtime_info[BATCH_DIM_CTR] = [batch_dim_ctr]
    runtime_info[ACT_ON_BATCH_DIMS] = [1] if act_on_batch_dims else [0]
    var name = "sum(" + String(axes) + ")"
    var res = register_any_op[Sum.maxpr, Sum.vjp, Sum.jvp, Sum.eagerxpr](
        [arg], name, target_shape, runtime_info=runtime_info
    )
    if not keep_dim:
        if len(axes) == len(target_shape):
            _ = axes.pop()
        for i in range(len(axes)):
            res = squeeze(res, [axes[i]], act_on_batch_dims)

    return res
