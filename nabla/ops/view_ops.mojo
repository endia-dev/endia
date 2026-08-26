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

from nabla.core.device_array import DeviceArray, ArrayImpl, zeros
from nabla.core.utils import getshape, ShapeType

from nabla.ops.unary_ops import incr_batch_dim_ctr, decr_batch_dim_ctr
from nabla.ops.utils import get_broadcasted_axis, register_any_op, RuntimeInfo
from nabla.api.utils import ExecutionContext, none
from nabla.ops.reduce_ops import sum


####################################################################################################
# View OPERATIONS
####################################################################################################


comptime BATCH_DIM_CTR = 0
comptime PERM = 1
comptime TARGET_shape = 1
comptime ORIGINALshape = 2
comptime FULL_TARGET_SHAPE = 3


# general permute transform function
struct Permute:
    @staticmethod
    def maxpr(
        args: List[compiler.graph.Symbol], array: DeviceArray
    ) raises -> compiler.graph.Symbol:
        var batch_dim_ctr = array.impl[].runtime_info[BATCH_DIM_CTR][0]
        var perm = array.impl[].runtime_info[PERM].copy()

        var target_perm = List[Int]()
        for i in range(batch_dim_ctr):
            target_perm.append(i)
        for i in range(len(perm)):
            target_perm.append(perm[i])

        var num_dims = len(target_perm)
        var current_axis_order = List[Int]()
        for i in range(-num_dims, 0):
            current_axis_order.append(i)

        var out_symbol = args[0]

        for x in range(num_dims):
            var target_axis = target_perm[x]
            var y = current_axis_order[target_axis]
            if x == y:
                continue

            out_symbol = compiler.graph.ops.transpose(out_symbol, x, y)
            current_axis_order[x] = y
            current_axis_order[y] = x

        return out_symbol

    @staticmethod
    def eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Permute"

    @staticmethod
    def vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var perm = array.impl[].runtime_info[PERM].copy()
        return [permute(tangent, perm)]

    @staticmethod
    def jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var perm = array.impl[].runtime_info[PERM].copy()
        return permute(tangents[0], perm)


def permute(arg: DeviceArray, perm: List[Int]) raises -> DeviceArray:
    var runtime_info = RuntimeInfo(2)
    var batch_dim_ctr = arg.batch_dim_ctr()

    var argshape = arg.shape()
    var axes = List[Int]()

    if len(perm) != len(arg.shape()) - batch_dim_ctr:
        raise "The permutation must be the same length as the number of dimensions in the array"

    for dim in perm:
        if dim >= 0:
            axes.append(-len(argshape[batch_dim_ctr:]) + dim)
        else:
            axes.append(dim)

    var target_shape = argshape.copy()
    for i in range(len(axes)):
        target_shape[i + batch_dim_ctr] = argshape[axes[i]]

    var name = "permute(" + String(axes) + ")"

    runtime_info[PERM] = axes.copy()
    runtime_info[BATCH_DIM_CTR] = [batch_dim_ctr]

    return register_any_op[
        Permute.maxpr, Permute.vjp, Permute.jvp, Permute.eagerxpr
    ]([arg], name, target_shape, runtime_info=runtime_info)


def transpose(arg: DeviceArray, x: Int, y: Int) raises -> DeviceArray:
    return permute(arg, [x, y])


struct Reshape:
    @staticmethod
    def maxpr(
        args: List[compiler.graph.Symbol], array: DeviceArray
    ) raises -> compiler.graph.Symbol:
        var dims = List[compiler.graph.Dim]()
        for s in array.shape():
            dims.append(compiler.graph.Dim(s))
        return compiler.graph.ops.reshape(args[0], dims)

    @staticmethod
    def eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Reshape"

    @staticmethod
    def vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var originalshape = array.impl[].runtime_info[ORIGINALshape].copy()
        return [reshape(tangent, originalshape)]

    @staticmethod
    def jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var target_shape = array.impl[].runtime_info[TARGET_shape].copy()
        return reshape(tangents[0], target_shape)


def reshape(arg: DeviceArray, shape: List[Int]) raises -> DeviceArray:
    var runtime_info = RuntimeInfo(4)
    var batch_dim_ctr = arg.batch_dim_ctr()
    var arg_shape = arg.shape()
    var arg_num_elements = 1
    var target_num_elements = 1
    for i in range(batch_dim_ctr, len(arg_shape)):
        arg_num_elements *= arg_shape[i]
    for i in range(len(shape)):
        target_num_elements *= shape[i]
    if arg_num_elements != target_num_elements:
        raise "The number of elements in the target shape must be equal to the number of elements in the original shape. " + String(arg_shape[
            batch_dim_ctr:
        ]) + " vs " + String(shape)

    var target_shape = List(arg_shape[:batch_dim_ctr]) + shape.copy()
    runtime_info[TARGET_shape] = shape.copy()
    runtime_info[ORIGINALshape] = List(arg_shape[batch_dim_ctr:])
    var name = "reshape(" + String(arg.shape()) + " -> " + String(target_shape) + ")"

    return register_any_op[
        Reshape.maxpr, Reshape.vjp, Reshape.jvp, Reshape.eagerxpr
    ]([arg], name, target_shape, runtime_info=runtime_info)


def flatten(arg: DeviceArray) raises -> DeviceArray:
    var batch_dim_ctr = arg.batch_dim_ctr()
    var shape = arg.shape()[batch_dim_ctr:]
    var num_elements = 1
    for i in range(0, len(shape)):
        num_elements *= shape[i]
    return reshape(arg, [num_elements])


comptime BROADCASTED_AXES = 3
comptime ACT_ON_BATCH_DIMS = 4
comptime FULL_TARGET_shape = TARGET_shape


struct BroadcastTo:
    @staticmethod
    def maxpr(
        args: List[compiler.graph.Symbol], array: DeviceArray
    ) raises -> compiler.graph.Symbol:
        var runtime_info = array.impl[].runtime_info.copy()
        var full_target_shape = runtime_info[FULL_TARGET_shape].copy()
        var pre_shape = List(array.shape()[: -len(full_target_shape)])
        var dims = List[compiler.graph.Dim]()
        for s in pre_shape + full_target_shape.copy():
            dims.append(compiler.graph.Dim(s))

        return compiler.graph.ops.broadcast_to(args[0], dims)

    @staticmethod
    def eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for BroadcastTo"

    @staticmethod
    def vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var axes = array.impl[].runtime_info[BROADCASTED_AXES].copy()
        var act_on_batch_dims = True if array.impl[].runtime_info[
            ACT_ON_BATCH_DIMS
        ][0] == 1 else False

        return [
            sum(
                tangent,
                axes,
                keep_dim=True,
                act_on_batch_dims=act_on_batch_dims,
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

        var target_shape = runtime_info[FULL_TARGET_shape].copy()
        var tangent_shape = tangents[0].shape()
        var primal_shape = primals[0].shape()
        var offset = 0
        if len(tangent_shape) > len(primal_shape):
            offset = len(tangent_shape) - len(primal_shape)
        var _pre = List(tangent_shape[:offset])
        _pre += target_shape.copy()
        target_shape = _pre^

        return broadcast_to(
            tangents[0], target_shape, act_on_batch_dims, expand_dims=False
        )


def broadcast_to(
    _arg: DeviceArray,
    _shape: List[Int],
    act_on_batch_dims: Bool = False,
    expand_dims: Bool = True,
) raises -> DeviceArray:
    var shape = _shape.copy()
    var arg = _arg
    var argshape = arg.shape()
    var batch_dim_ctr = arg.batch_dim_ctr() if arg.batch_dim_ctr() >= 0 else 0

    if act_on_batch_dims:
        var len_true_dims = len(argshape) - arg.batch_dim_ctr()
        if argshape[len(argshape) - len_true_dims:] != shape[len(shape) - len_true_dims:]:
            raise "Error in broadcast_to: When acting on batch dimensions, the non-batch dims must be equal. Currently:" + String(argshape[
                len(argshape) - len_true_dims:
            ]) + " vs " + String(shape[
                len(shape) - len_true_dims:
            ])
        batch_dim_ctr = 0

    if argshape[batch_dim_ctr:] == shape:
        return arg

    if expand_dims:
        for _ in range(len(shape) - len(argshape[batch_dim_ctr:])):
            arg = unsqueeze(arg, [0], act_on_batch_dims)
        argshape = arg.shape()

    var target_shape = List(argshape[:batch_dim_ctr]) + shape.copy()

    if len(shape) < len(argshape[batch_dim_ctr:]):
        raise "Error in setting up broadcast op: The target shape must be greater than or equal to the original shape. trying to broadcast from " + String(argshape[
            batch_dim_ctr:
        ]) + " to " + String(shape)

    var runtime_info = RuntimeInfo(5)
    var broadcasted_axis = get_broadcasted_axis(List(argshape[batch_dim_ctr:]), shape)
    runtime_info[ORIGINALshape] = List(argshape[batch_dim_ctr:])
    runtime_info[BROADCASTED_AXES] = broadcasted_axis.copy()
    runtime_info[FULL_TARGET_shape] = shape.copy()
    runtime_info[BATCH_DIM_CTR] = [batch_dim_ctr]
    runtime_info[ACT_ON_BATCH_DIMS] = [1] if act_on_batch_dims else [0]
    var name = "broadcast_to(" + String(target_shape) + ")" + " {" + String(arg.batch_dim_ctr()) + "}"

    return register_any_op[
        BroadcastTo.maxpr,
        BroadcastTo.vjp,
        BroadcastTo.jvp,
        BroadcastTo.eagerxpr,
    ]([arg], name, target_shape, runtime_info=runtime_info)


comptime SLICES = 1
comptime RED_SLICES = 2


struct ArraySlice:
    @staticmethod
    def maxpr(
        args: List[compiler.graph.Symbol], array: DeviceArray
    ) raises -> compiler.graph.Symbol:
        var list_slices = array.impl[].runtime_info[SLICES].copy()

        var slices = List[Slice]()

        for i in range(0, len(list_slices) // 3):
            slices.append(
                Slice(
                    list_slices[i * 3],
                    list_slices[i * 3 + 1],
                    list_slices[i * 3 + 2],
                )
            )

        # print("\nIn MAX slice:", String(slices))
        # print("arg_shape:", String(array.args()[0][].shape))
        # print("target_shape:", String(array.shape()))

        if len(slices) == 1:
            return compiler.graph.ops.slice(args[0], slices[0])
        elif len(slices) == 2:
            return compiler.graph.ops.slice(args[0], slices[0], slices[1])
        elif len(slices) == 3:
            return compiler.graph.ops.slice(
                args[0], slices[0], slices[1], slices[2]
            )
        elif len(slices) == 4:
            return compiler.graph.ops.slice(
                args[0], slices[0], slices[1], slices[2], slices[3]
            )
        elif len(slices) == 5:
            return compiler.graph.ops.slice(
                args[0], slices[0], slices[1], slices[2], slices[3], slices[4]
            )
        elif len(slices) == 6:
            return compiler.graph.ops.slice(
                args[0],
                slices[0],
                slices[1],
                slices[2],
                slices[3],
                slices[4],
                slices[5],
            )
        elif len(slices) == 7:
            return compiler.graph.ops.slice(
                args[0],
                slices[0],
                slices[1],
                slices[2],
                slices[3],
                slices[4],
                slices[5],
                slices[6],
            )
        elif len(slices) == 8:
            return compiler.graph.ops.slice(
                args[0],
                slices[0],
                slices[1],
                slices[2],
                slices[3],
                slices[4],
                slices[5],
                slices[6],
                slices[7],
            )
        elif len(slices) == 9:
            return compiler.graph.ops.slice(
                args[0],
                slices[0],
                slices[1],
                slices[2],
                slices[3],
                slices[4],
                slices[5],
                slices[6],
                slices[7],
                slices[8],
            )
        elif len(slices) == 10:
            return compiler.graph.ops.slice(
                args[0],
                slices[0],
                slices[1],
                slices[2],
                slices[3],
                slices[4],
                slices[5],
                slices[6],
                slices[7],
                slices[8],
                slices[9],
            )
        else:
            raise "Slicing more than 10 dimensions is not supported"

    @staticmethod
    def eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for ArraySlice"

    @staticmethod
    def vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var list_slices = array.impl[].runtime_info[RED_SLICES].copy()
        var primal = primals[0]
        var primal_batch_ctr = primal.batch_dim_ctr()
        var primal_shape = primal.shape()[primal_batch_ctr:]
        var tangent_batch_ctr = tangent.batch_dim_ctr()
        var slice_ctr = 0
        var slice_axis = 0

        for i in range(len(primal_shape)):
            if list_slices[3 * i + 2] != 1:
                raise "Error: Cannot compute VJP for array_slice operation if step size is greater than 1."
            var red_dim = (
                list_slices[3 * i + 1] - list_slices[3 * i]
            ) // list_slices[3 * i + 2]
            if red_dim != primal_shape[i]:
                slice_ctr += 1
                slice_axis = i

        if slice_ctr > 1:
            raise "VJP for Slice not implemented! This would require a padding operation in the MAX Engine which currently does not exist."

        var front_shape = tangent.shape()
        front_shape[slice_axis + tangent_batch_ctr] = list_slices[
            3 * slice_axis
        ]

        var back_shape = tangent.shape()
        back_shape[slice_axis + tangent_batch_ctr] = (
            primal_shape[slice_axis] - list_slices[3 * slice_axis + 1]
        ) // list_slices[3 * slice_axis + 2]

        var array_stack = List[DeviceArray]()
        if front_shape[slice_axis + tangent_batch_ctr] > 0:
            var front_zeros = zeros(front_shape.copy(), tangent.dtype())
            front_zeros.batch_dim_ctr_(tangent_batch_ctr)
            array_stack.append(front_zeros)

        array_stack.append(tangent)

        if back_shape[slice_axis + tangent_batch_ctr] > 0:
            var back_zeros = zeros(back_shape.copy(), tangent.dtype())
            back_zeros.batch_dim_ctr_(tangent_batch_ctr)
            array_stack.append(back_zeros)

        var new_contanget = concat(
            array_stack, axis=slice_axis - primal.batch_dim_ctr()
        )
        # print("\nprimal_shape:", String(primal.shape()))
        # print("tangent_shape:", String(tangent.shape()))
        # print("front_shape:", String(front_shape))
        # print("tangent_shape:", String(tangent.shape()))
        # print("back_shape:", String(back_shape))

        if len(primal.impl[].cotangent) == 1:
            var old_cotangent = DeviceArray(primal.impl[].cotangent[0])
            return [old_cotangent + new_contanget]
        else:
            return [new_contanget]

    @staticmethod
    def jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var list_slices = array.impl[].runtime_info[RED_SLICES].copy()
        var slices = List[Slice]()
        for i in range(0, len(list_slices) // 3):
            slices.append(
                Slice(
                    list_slices[i * 3],
                    list_slices[i * 3 + 1],
                    list_slices[i * 3 + 2],
                )
            )

        return array_slice(tangents[0], slices)


def array_slice(arg: DeviceArray, slices: List[Slice]) raises -> DeviceArray:
    var batch_dim_ctr = arg.batch_dim_ctr()
    if batch_dim_ctr == none:
        batch_dim_ctr = 0

    var name = "array_slice" + String(slices) + ")"
    var runtime_info = RuntimeInfo(3)
    var shape = arg.shape()

    var list_slices = List[Int]()

    var new_shape = List[Int]()

    for i in range(batch_dim_ctr):
        list_slices.append(0)
        list_slices.append(shape[i])
        list_slices.append(1)
        new_shape.append(shape[i])

    for i in range(len(slices)):
        # start value
        if slices[i].start:
            list_slices.append(slices[i].start.value())
        else:
            list_slices.append(0)

        # end value
        if slices[i].end:
            list_slices.append(slices[i].end.value())
        else:
            list_slices.append(arg.shape()[i])

        # step size
        if slices[i].step:
            list_slices.append(slices[i].step.value())
        else:
            list_slices.append(1)

        # update shape
        new_shape.append((list_slices[len(list_slices) - 2] - list_slices[len(list_slices) - 3]) // list_slices[len(list_slices) - 1])

    for i in range(batch_dim_ctr + len(slices), len(shape), 1):
        list_slices.append(0)
        list_slices.append(shape[i])
        list_slices.append(1)
        new_shape.append(shape[i])

    runtime_info[BATCH_DIM_CTR] = [arg.batch_dim_ctr()]
    runtime_info[SLICES] = list_slices.copy()
    runtime_info[RED_SLICES] = List(list_slices[3 * batch_dim_ctr :])

    return register_any_op[
        ArraySlice.maxpr, ArraySlice.vjp, ArraySlice.jvp, ArraySlice.eagerxpr
    ]([arg], name, new_shape, runtime_info=runtime_info)


comptime AXIS = 1
comptime SIZES = 2


struct Stack:
    @staticmethod
    def maxpr(
        args: List[compiler.graph.Symbol], array: DeviceArray
    ) raises -> compiler.graph.Symbol:
        var batch_dim_ctr = array.impl[].runtime_info[BATCH_DIM_CTR][0]
        var axis = array.impl[].runtime_info[AXIS][0] + batch_dim_ctr
        return compiler.graph.ops.stack(args, axis)

    @staticmethod
    def eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Stack"

    @staticmethod
    def vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var axis = array.impl[].runtime_info[AXIS][0]
        var sizes = array.impl[].runtime_info[SIZES]
        var primal_tangents = split(tangent, sizes, axis)
        for i in range(len(primal_tangents)):
            primal_tangents[i] = primal_tangents[i].reshape(primals[0].shape())
        return primal_tangents

    @staticmethod
    def jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var axis = array.impl[].runtime_info[AXIS][0]
        return stack(tangents, axis)


def stack(args: List[DeviceArray], axis: Int = 0) raises -> DeviceArray:
    var batch_dim_ctr = args[0].batch_dim_ctr()
    var sizes = List[Int]()
    var shape = List[Int]()
    var refshape = args[0].shape()
    for arg in args:
        if arg.shape() != refshape:
            raise "All input arrays must have the same shape"
        sizes.append(1)

    for i in range(axis + batch_dim_ctr):
        shape.append(args[0].shape()[i])
    shape.append(len(args))
    for i in range(axis + batch_dim_ctr, len(args[0].shape())):
        shape.append(args[0].shape()[i])

    var runtime_info = RuntimeInfo(3)
    runtime_info[AXIS] = [axis]
    runtime_info[SIZES] = sizes
    runtime_info[BATCH_DIM_CTR] = [batch_dim_ctr]
    var name = "stack(" + String(axis) + ")"

    return register_any_op[Stack.maxpr, Stack.vjp, Stack.jvp, Stack.eagerxpr](
        args, name, shape, runtime_info=runtime_info
    )


struct Concat:
    @staticmethod
    def maxpr(
        args: List[compiler.graph.Symbol], array: DeviceArray
    ) raises -> compiler.graph.Symbol:
        var batch_dim_ctr = array.impl[].runtime_info[BATCH_DIM_CTR][0]
        var axis = array.impl[].runtime_info[AXIS][0] + batch_dim_ctr
        return compiler.graph.ops.concat(args, axis)

    @staticmethod
    def eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Concat"

    @staticmethod
    def vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var axis = array.impl[].runtime_info[AXIS][0]
        var sizes = array.impl[].runtime_info[SIZES].copy()
        var primal_tangents = split(tangent, sizes, axis)
        for i in range(len(primal_tangents)):
            primal_tangents[i] = primal_tangents[i].reshape(primals[0].shape())
        return primal_tangents.copy()

    @staticmethod
    def jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var axis = array.impl[].runtime_info[AXIS][0]
        return concat(tangents, axis)


def concat(args: List[DeviceArray], axis: Int = 0) raises -> DeviceArray:
    var batch_dim_ctr = args[0].batch_dim_ctr()
    var sizes = List[Int]()
    var new_dim_size = 0

    for arg in args:
        var batch_dim_ctr = arg.batch_dim_ctr()
        new_dim_size += arg.shape()[batch_dim_ctr:][axis]
        sizes.append(arg.shape()[batch_dim_ctr:][axis])

    var shape = args[0].shape()
    shape[axis + batch_dim_ctr] = new_dim_size

    var runtime_info = RuntimeInfo(3)
    runtime_info[AXIS] = [axis]
    runtime_info[SIZES] = sizes.copy()
    runtime_info[BATCH_DIM_CTR] = [batch_dim_ctr]
    var name = "concat(" + String(axis) + ")"

    return register_any_op[
        Concat.maxpr, Concat.vjp, Concat.jvp, Concat.eagerxpr
    ](args, name, shape, runtime_info=runtime_info)


def split(
    arg: DeviceArray, sizes: List[Int], axis: Int
) raises -> List[DeviceArray]:
    var slices = List[Slice]()
    var batch_dim_ctr = arg.batch_dim_ctr()
    var argshape = arg.shape()[batch_dim_ctr:]
    for i in range(len(argshape)):
        slices.append(Slice(0, argshape[i], 1))

    var idx = 0
    var results = List[DeviceArray]()

    for i in range(len(sizes)):
        slices[axis] = Slice(idx, idx + sizes[i], 1)
        results.append(array_slice(arg, slices))
        idx += sizes[i]

    return results.copy()


struct Squeeze:
    @staticmethod
    def maxpr(
        args: List[compiler.graph.Symbol], array: DeviceArray
    ) raises -> compiler.graph.Symbol:
        var axes = array.impl[].runtime_info[AXIS].copy()
        if len(axes) > 1:
            raise "Squeeze only supports a single axis at the moment"

        var symbol = compiler.graph.ops.squeeze(args[0], axes[0])
        return symbol

    @staticmethod
    def eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Squeeze"

    @staticmethod
    def vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var axis = array.impl[].runtime_info[AXIS].copy()
        var act_on_batch_dims = True if array.impl[].runtime_info[
            ACT_ON_BATCH_DIMS_SQ
        ][0] == 1 else False
        return [
            unsqueeze(tangent, axis, act_on_batch_dims, incr_batch_dims=False)
        ]

    @staticmethod
    def jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var axis = array.impl[].runtime_info[AXIS].copy()
        var act_on_batch_dims = True if array.impl[].runtime_info[
            ACT_ON_BATCH_DIMS_SQ
        ][0] == 1 else False
        return squeeze(
            tangents[0], axis, act_on_batch_dims, dec_batch_dims=False
        )


comptime ACT_ON_BATCH_DIMS_SQ = 2


def squeeze(
    arg: DeviceArray,
    axis: List[Int],
    act_on_batch_dims: Bool = False,
    dec_batch_dims: Bool = True,
) raises -> DeviceArray:
    var batch_dim_ctr = arg.batch_dim_ctr()
    if len(axis) > 1:
        raise "Squeeze only supports a single axis at the moment"
    if act_on_batch_dims:
        batch_dim_ctr = 0

    var argshape = arg.shape()[batch_dim_ctr:]
    var target_shape = List(arg.shape()[:batch_dim_ctr])

    if len(axis) == len(arg.shape()):
        raise "Cannot squeeze all dimensions of an array with size > 1"

    var axes = axis.copy()
    for i in range(len(axis)):
        if axis[i] >= 0:
            axes[i] = -len(argshape) + axis[i]

    for i in range(-len(argshape), 0):
        if i not in axes:
            target_shape.append(argshape[len(argshape) + i])
        elif argshape[len(argshape) + i] != 1:
            raise "Cannot squeeze dimension " + String(
                i
            ) + " with size " + String(arg.shape()[len(arg.shape()) + i])

    var runtime_info = RuntimeInfo(3)
    runtime_info[AXIS] = axes.copy()
    runtime_info[BATCH_DIM_CTR] = [batch_dim_ctr]
    runtime_info[ACT_ON_BATCH_DIMS_SQ] = [1] if act_on_batch_dims else [
        0
    ]
    var name = "squeeze(" + String(axes) + ")"
    var res = register_any_op[
        Squeeze.maxpr, Squeeze.vjp, Squeeze.jvp, Squeeze.eagerxpr
    ]([arg], name, target_shape, runtime_info=runtime_info)
    if act_on_batch_dims and dec_batch_dims:
        var diff = len(argshape) - len(target_shape)
        for _ in range(diff):
            res = decr_batch_dim_ctr(res)

    return res


struct Unsqueeze:
    @staticmethod
    def maxpr(
        args: List[compiler.graph.Symbol], array: DeviceArray
    ) raises -> compiler.graph.Symbol:
        var axes = array.impl[].runtime_info[AXIS].copy()
        var symbol = args[0]

        for i in range(len(axes)):
            symbol = compiler.graph.ops.unsqueeze(symbol, axes[i])

        return symbol

    @staticmethod
    def eagerxpr(mut curr: DeviceArray, args: List[DeviceArray]) raises -> None:
        raise "Eager execution is not supported for Unsqueeze"

    @staticmethod
    def vjp(
        primals: List[DeviceArray], tangent: DeviceArray, array: DeviceArray
    ) raises -> List[DeviceArray]:
        var axes = array.impl[].runtime_info[AXIS].copy()
        var act_on_batch_dims = True if array.impl[].runtime_info[
            ACT_ON_BATCH_DIMS_SQ
        ][0] == 1 else False
        return [
            squeeze(tangent, axes, act_on_batch_dims, dec_batch_dims=False)
        ]

    @staticmethod
    def jvp(
        primals: List[DeviceArray],
        tangents: List[DeviceArray],
        array: DeviceArray,
    ) raises -> DeviceArray:
        var axes = array.impl[].runtime_info[AXIS].copy()
        var act_on_batch_dims = True if array.impl[].runtime_info[
            ACT_ON_BATCH_DIMS_SQ
        ][0] == 1 else False
        var res = unsqueeze(
            tangents[0], axes, act_on_batch_dims, incr_batch_dims=False
        )
        return res


def unsqueeze(
    arg: DeviceArray,
    axes: List[Int],
    act_on_batch_dims: Bool = False,
    incr_batch_dims: Bool = True,
) raises -> DeviceArray:
    var batch_dim_ctr = arg.batch_dim_ctr()
    if len(axes) > 1:
        raise "Unsqueeze only supports a single axis at the moment"
    if act_on_batch_dims:
        batch_dim_ctr = 0

    var argshape = arg.shape()
    var target_rank = len(argshape[batch_dim_ctr:]) + len(axes)
    var shape = List(argshape[:batch_dim_ctr])
    for _ in range(target_rank):
        shape.append(1)

    var sorted_axes = List[Int]()
    for axis in axes:
        var actual_axis = axis if axis < 0 else -target_rank + axis
        sorted_axes.append(actual_axis)

    sort(sorted_axes)

    var arg_idx = -len(argshape[batch_dim_ctr:])
    for target_idx in range(-target_rank, 0):
        if target_idx not in sorted_axes:
            shape[len(shape) + target_idx] = argshape[len(argshape) + arg_idx]
            arg_idx += 1

    var runtime_info = RuntimeInfo(3)
    runtime_info[AXIS] = sorted_axes.copy()
    runtime_info[BATCH_DIM_CTR] = [batch_dim_ctr]
    runtime_info[ACT_ON_BATCH_DIMS_SQ] = [1] if act_on_batch_dims else [
        0
    ]
    var name = "unsqueeze(" + String(sorted_axes) + ")"
    var res = register_any_op[
        Unsqueeze.maxpr, Unsqueeze.vjp, Unsqueeze.jvp, Unsqueeze.eagerxpr
    ]([arg], name, shape, runtime_info=runtime_info)
    if act_on_batch_dims and incr_batch_dims:
        for _ in range(len(axes)):
            res = incr_batch_dim_ctr(res)
    return res
