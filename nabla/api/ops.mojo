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


from nabla.api.array import Array
from ..ops import binary_ops as binary_ops
from ..ops import reduce_ops as reduce_ops
from ..ops import unary_ops as unary_ops
from ..ops import view_ops as view_ops
from nabla.core.utils import ShapeType, getshape
from nabla.core.device_array import DeviceArray
from ..engine.trafos import vjp_trafo as vjp_trafo


def add(x: Array, y: Array) raises -> Array:
    return Array(binary_ops.add(x.device_array[], y.device_array[]))


def mul(x: Array, y: Array) raises -> Array:
    return Array(binary_ops.mul(x.device_array[], y.device_array[]))


def sub(x: Array, y: Array) raises -> Array:
    return Array(binary_ops.sub(x.device_array[], y.device_array[]))


def div(x: Array, y: Array) raises -> Array:
    return Array(binary_ops.div(x.device_array[], y.device_array[]))


def matmul(x: Array, y: Array) raises -> Array:
    return Array(binary_ops.matmul(x.device_array[], y.device_array[]))


def gt(x: Array, y: Array) raises -> Array:
    return Array(binary_ops.gt(x.device_array[], y.device_array[]))


def pow(x: Array, y: Array) raises -> Array:
    return Array(binary_ops.pow(x.device_array[], y.device_array[]))


def sum(
    x: Array,
    axis: List[Int] = List[Int](),
    keep_dim: Bool = False,
    act_on_batch_dims: Bool = False,
) raises -> Array:
    return Array(
        reduce_ops.sum(x.device_array[], axis, keep_dim, act_on_batch_dims)
    )


def sin(x: Array) raises -> Array:
    return Array(unary_ops.sin(x.device_array[]))


def cast(x: Array, dtype: DType) raises -> Array:
    return Array(unary_ops.cast(x.device_array[], dtype))


def negate(x: Array) raises -> Array:
    return Array(unary_ops.negate(x.device_array[]))


def cos(x: Array) raises -> Array:
    return Array(unary_ops.cos(x.device_array[]))


def relu(x: Array) raises -> Array:
    return Array(unary_ops.relu(x.device_array[]))


def log(x: Array) raises -> Array:
    return Array(unary_ops.log(x.device_array[]))


def gt_zero(x: Array) raises -> Array:
    return Array(unary_ops.gt_zero(x.device_array[]))


def incr_batch_dim_ctr(x: Array) raises -> Array:
    return Array(unary_ops.incr_batch_dim_ctr(x.device_array[]))


def decr_batch_dim_ctr(x: Array) raises -> Array:
    return Array(unary_ops.decr_batch_dim_ctr(x.device_array[]))


def permute(arg: Array, perm: List[Int]) raises -> Array:
    return Array(view_ops.permute(arg.device_array[], perm))


def transpose(arg: Array, x: Int, y: Int) raises -> Array:
    return Array(view_ops.transpose(arg.device_array[], x, y))


def reshape(x: Array, shape: ShapeType) raises -> Array:
    return Array(view_ops.reshape(x.device_array[], getshape(shape)))


def flatten(x: Array) raises -> Array:
    return Array(view_ops.flatten(x.device_array[]))


def broadcast_to(
    x: Array, shape: ShapeType, act_on_batch_dims: Bool = False
) raises -> Array:
    return Array(
        view_ops.broadcast_to(
            x.device_array[], getshape(shape), act_on_batch_dims
        )
    )


def stack(args: List[Array], axis: Int = 0) raises -> Array:
    var device_arrays = List[DeviceArray]()
    for arg in args:
        device_arrays.append(arg.device_array[])
    return Array(view_ops.stack(device_arrays, axis))


def array_slice(arg: Array, slices: List[Slice]) raises -> Array:
    return Array(view_ops.array_slice(arg.device_array[], slices))


def concat(args: List[Array], axis: Int = 0) raises -> Array:
    var device_arrays = List[DeviceArray]()
    for arg in args:
        device_arrays.append(arg.device_array[])
    return Array(view_ops.concat(device_arrays, axis))


def split(arg: Array, sizes: List[Int], axis: Int) raises -> List[Array]:
    var device_arrays = view_ops.split(arg.device_array[], sizes, axis)
    var results = List[Array]()
    for device_array in device_arrays:
        results.append(Array(device_array))
    return results.copy()


def squeeze(
    arg: Array, axis: List[Int], act_on_batch_dims: Bool = False
) raises -> Array:
    return Array(
        view_ops.squeeze(arg.device_array[], axis, act_on_batch_dims)
    )


def unsqueeze(
    arg: Array, axes: List[Int], act_on_batch_dims: Bool = False
) raises -> Array:
    return Array(
        view_ops.unsqueeze(arg.device_array[], axes, act_on_batch_dims)
    )


def backward(array: Array, remat: Bool = False) raises -> None:
    vjp_trafo.backward(array.device_array[], remat)
