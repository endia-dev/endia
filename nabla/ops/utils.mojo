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
from std.collections import Dict, Optional
from nabla.core.device_array import DeviceArray, ArrayImpl
from nabla.ops.view_ops import broadcast_to, unsqueeze
from nabla.ops.unary_ops import incr_batch_dim_ctr, decr_batch_dim_ctr
from nabla.api.utils import ExecutionContext
from nabla.api.utils import none


def generic_setup(args: List[DeviceArray], name: String) raises -> DeviceArray:
    var dtype = args[0].impl[].spec.dtype()
    var diffable = False
    var execution_context = Optional[ExecutionContext](None)
    var batch_dim_ctr = 0
    var arg_string: String = ""

    for arg in args:
        arg_string += String(arg.dtype()) + String(arg.shape()) + ","
        diffable = diffable or arg.impl[]._diffable
        if arg.impl[].spec.dtype() != dtype:
            raise "DType mismatch in arguments when registering op:" + name
        if arg.batch_dim_ctr() > batch_dim_ctr:
            batch_dim_ctr = arg.batch_dim_ctr()
        if arg.impl[].execution_context:
            execution_context = arg.impl[].execution_context

    var _shape0: List[Int] = [0]
    var res = DeviceArray(
        shape=_shape0.copy(),
        dtype=dtype,
        requires_pullback=diffable,
        execution_context=execution_context,
        name="{" + String(batch_dim_ctr) + "}" + name + "(" + arg_string + ")",
    )
    res.batch_dim_ctr_(batch_dim_ctr)

    for arg in args:
        res.impl[]._args.append(arg.impl)

    return res


def register_any_op[
    maxpr: def (
        List[compiler.graph.Symbol], DeviceArray
    ) raises thin -> compiler.graph.Symbol,
    vjp: def (List[DeviceArray], DeviceArray, DeviceArray) raises thin -> List[
        DeviceArray
    ],
    jvp: def (
        List[DeviceArray], List[DeviceArray], DeviceArray
    ) raises thin -> DeviceArray,
    eagerxpr: def (mut DeviceArray, List[DeviceArray]) raises thin -> None,
](
    args: List[DeviceArray],
    name: String,
    targetshape: List[Int],
    runtime_info: List[List[Int]] = List[List[Int]](),
) raises -> DeviceArray:
    var res = generic_setup(args, name)
    res.shape_(targetshape)
    res.impl[].runtime_info = runtime_info.copy()

    res.impl[]._maxpr = maxpr
    res.impl[]._vjp = vjp
    res.impl[]._jvp = jvp
    res.impl[]._eagerxpr = eagerxpr

    var arg_shapes: String = ""
    for arg in args:
        arg_shapes += String(arg.shape()) + ", "

    # print(
    #     "   ",
    #     res.impl[].name,
    #     String(targetshape),
    #     " args:",
    #     arg_shapes,
    #     "batch_dim_ctr:",
    #     res.batch_dim_ctr(),
    # )

    return res


def get_broadcasted_axis(
    argshape: List[Int], targetshape: List[Int]
) raises -> List[Int]:
    var broadcasted_axis = List[Int]()
    var rank = len(targetshape)
    var i = len(argshape) - 1
    var j = len(targetshape) - 1
    while j >= 0:
        if i >= 0 and argshape[i] == targetshape[j]:
            i -= 1
            j -= 1
        elif i >= 0 and argshape[i] == 1 and targetshape[j] > 1:
            broadcasted_axis.append(-rank + j)
            i -= 1
            j -= 1
        elif i < 0:
            broadcasted_axis.append(-rank + j)
            j -= 1
        else:
            raise "Invalid broadcast, trying to broadcast from " + String(argshape) + " to " + String(targetshape)

    return broadcasted_axis.copy()


def get_broadcastedshape(
    arg0: List[Int], arg1: List[Int], right_offset: Int = 0
) raises -> List[Int]:
    var newshape = List[Int]()
    var i = len(arg0) - 1 - right_offset
    var j = len(arg1) - 1 - right_offset

    # the ignored shape elements must be set manually
    for _ in range(right_offset):
        newshape.append(-1)

    if len(arg0) == 0:
        return arg1.copy()
    if len(arg1) == 0:
        return arg0.copy()

    while i >= 0 or j >= 0:
        if i >= 0 and j >= 0:
            if arg0[i] == arg1[j]:
                newshape.append(arg0[i])
                i -= 1
                j -= 1
            elif arg0[i] == 1:
                newshape.append(arg1[j])
                j -= 1
                i -= 1
            elif arg1[j] == 1:
                newshape.append(arg0[i])
                i -= 1
                j -= 1
            else:
                raise "Invalid broadcast, when finding the brshape for: " + String(arg0) + " and " + String(arg1)
        elif i >= 0:
            newshape.append(arg0[i])
            i -= 1
        elif j >= 0:
            newshape.append(arg1[j])
            j -= 1

    newshape.reverse()

    return newshape.copy()


def register_binary_op[
    maxpr: def (
        List[compiler.graph.Symbol], DeviceArray
    ) raises thin -> compiler.graph.Symbol,
    vjp: def (List[DeviceArray], DeviceArray, DeviceArray) raises thin -> List[
        DeviceArray
    ],
    jvp: def (
        List[DeviceArray], List[DeviceArray], DeviceArray
    ) raises thin -> DeviceArray,
    eagerxpr: def (mut DeviceArray, List[DeviceArray]) raises thin -> None,
](
    read _arg0: DeviceArray,
    read _arg1: DeviceArray,
    name: String,
) raises -> DeviceArray:
    var arg0 = _arg0
    var arg1 = _arg1

    var arg0_offset = arg0.batch_dim_ctr() if arg0.batch_dim_ctr() != none else 0
    var arg1_offset = arg1.batch_dim_ctr() if arg1.batch_dim_ctr() != none else 0

    var arg0_batch_dims = List(arg0.shape()[:arg0_offset])
    var arg1_batch_dims = List(arg1.shape()[:arg1_offset])
    var arg0_true_dims = List(arg0.shape()[arg0_offset:])
    var arg1_true_dims = List(arg1.shape()[arg1_offset:])

    var res_batch_dim = get_broadcastedshape(
        arg0_batch_dims,
        arg1_batch_dims,
    )
    var res_true_dim = get_broadcastedshape(
        arg0_true_dims,
        arg1_true_dims,
    )
    var new_shape = res_batch_dim + res_true_dim.copy()

    arg0 = broadcast_to(arg0, res_true_dim)
    arg1 = broadcast_to(arg1, res_true_dim)
    arg0 = broadcast_to(arg0, new_shape, act_on_batch_dims=True)
    arg1 = broadcast_to(arg1, new_shape, act_on_batch_dims=True)

    return register_any_op[maxpr, vjp, jvp, eagerxpr](
        [arg0, arg1],
        name,
        new_shape,
    )


def register_unary_op[
    maxpr: def (
        List[compiler.graph.Symbol], DeviceArray
    ) raises thin -> compiler.graph.Symbol,
    vjp: def (List[DeviceArray], DeviceArray, DeviceArray) raises thin -> List[
        DeviceArray
    ],
    jvp: def (
        List[DeviceArray], List[DeviceArray], DeviceArray
    ) raises thin -> DeviceArray,
    eagerxpr: def (mut DeviceArray, List[DeviceArray]) raises thin -> None,
](arg: DeviceArray, name: String,) raises -> DeviceArray:
    return register_any_op[maxpr, vjp, jvp, eagerxpr](
        [arg], name, arg.shape()
    )


def RuntimeInfo(cap: Int) raises -> List[List[Int]]:
    var runtime_info = List[List[Int]]()
    for _ in range(cap):
        runtime_info.append(List[Int]())
    return runtime_info.copy()
