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
from std.memory import ArcPointer
from std.collections import Optional
from nabla.core.device_array import DeviceArray
from nabla.api.utils import ExecutionContext
from nabla.core.utils import ShapeType
from ..core import device_array as devar


struct Array(Copyable, ImplicitlyCopyable, Movable, Writable):
    var device_array: ArcPointer[DeviceArray]

    def __init__(out self, device_array: ArcPointer[DeviceArray]) raises:
        self.device_array = device_array
        self.device_array[].not_to_be_materialized_(False)

    def __init__(out self, read device_array: DeviceArray) raises:
        self.device_array = ArcPointer(device_array)
        self.device_array[].not_to_be_materialized_(False)

    def __deinit__(deinit self):
        if self.device_array.count() == 1:
            self.device_array[].not_to_be_materialized_(True)

    def to_max[dtype: DType](self) raises -> compiler.tensor.Tensor[dtype]:
        return self.device_array[].to_max[dtype]()

    def tangent(self) raises -> Array:
        return Array(self.device_array[].tangent())

    def cotangent(self) raises -> Array:
        return Array(self.device_array[].cotangent())

    def grad(self) raises -> Array:
        return Array(self.device_array[].grad())

    def zero_tangent(mut self) raises -> None:
        self.device_array[].zero_tangent()

    def zero_cotangent(mut self) raises -> None:
        self.device_array[].zero_cotangent()

    def zero_grad(mut self) raises -> None:
        self.device_array[].zero_grad()

    def __str__(self) -> String:
        return String(self.device_array[])

    def write_to[W: Writer](self, mut writer: W):
        writer.write(String(self))

    def no_tangent(mut self) raises -> None:
        self.device_array[].no_tangent()

    def checkpoint(mut self, value: Bool = True) raises -> None:
        self.device_array[].checkpoint(value)

    def requires_pullback(self) raises -> Bool:
        return self.device_array[].requires_pullback()

    def requires_pullback_(mut self, value: Bool = True) raises -> None:
        self.device_array[].requires_pullback_(value)

    def requires_grad(self) raises -> Bool:
        return self.device_array[].requires_grad()

    def requires_grad_(mut self, value: Bool = True) raises -> None:
        self.device_array[].requires_grad_(value)

    def shape(self) raises -> List[Int]:
        return self.device_array[].shape()

    def shape_(mut self, shape: List[Int]) raises -> None:
        self.device_array[].shape_(shape)

    def dtype(self) raises -> DType:
        return self.device_array[].dtype()

    def batch_dim_ctr(self) raises -> Int:
        return self.device_array[].batch_dim_ctr()

    def batch_dim_ctr_(mut self, value: Int) raises -> None:
        self.device_array[].batch_dim_ctr_(value)

    def backward(mut self, remat: Bool = False) raises -> None:
        self.device_array[].backward(remat)

    def item[
        type: DType = DType.float32
    ](
        self, execution_context: Optional[ExecutionContext] = None
    ) raises -> Scalar[type]:
        _execution_context = (
            execution_context.value() if execution_context else self.device_array[]
            .impl[]
            .execution_context
        )
        return self.device_array[].item[type](_execution_context)

    def load[
        type: DType = DType.float32, width: Int = 1
    ](
        self, idx: Int, execution_context: Optional[ExecutionContext] = None
    ) raises -> SIMD[type, width]:
        _execution_context = (
            execution_context.value() if execution_context else self.device_array[]
            .impl[]
            .execution_context
        )
        return self.device_array[].load[type, width](idx, _execution_context)

    def store[
        type: DType, width: Int
    ](
        mut self,
        idx: Int,
        value: SIMD[type, width],
        execution_context: Optional[ExecutionContext] = None,
    ) raises -> None:
        _execution_context = (
            execution_context.value() if execution_context else self.device_array[]
            .impl[]
            .execution_context
        )
        self.device_array[].store[type, width](idx, value, _execution_context)

    def __getitem__(self, *slices: Slice) raises -> Array:
        var slice_list = List[Slice]()
        for slice in slices:
            slice_list.append(slice)
        return Array(self.device_array[].__getitem__(slice_list))

    def __add__(self, other: Array) raises -> Array:
        return Array(self.device_array[] + other.device_array[])

    def __add__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(self.device_array[] + other)

    def __add__(self, other: Int) raises -> Array:
        return Array(self.device_array[] + other)

    def __radd__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(other + self.device_array[])

    def __radd__(self, other: Int) raises -> Array:
        return Array(other + self.device_array[])

    def __iadd__(mut self, other: Array) raises -> None:
        self = Array(self.device_array[] + other.device_array[])

    def __iadd__(mut self, other: SIMD[_, 1]) raises -> None:
        self = Array(self.device_array[] + other)

    def __iadd__(mut self, other: Int) raises -> None:
        self = Array(self.device_array[] + other)

    def __mul__(self, other: Array) raises -> Array:
        return Array(self.device_array[] * other.device_array[])

    def __mul__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(self.device_array[] * other)

    def __mul__(self, other: Int) raises -> Array:
        return Array(self.device_array[] * other)

    def __rmul__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(other * self.device_array[])

    def __rmul__(self, other: Int) raises -> Array:
        return Array(other * self.device_array[])

    def __imul__(mut self, other: Array) raises -> None:
        self = Array(self.device_array[] * other.device_array[])

    def __imul__(mut self, other: SIMD[_, 1]) raises -> None:
        self = Array(self.device_array[] * other)

    def __imul__(mut self, other: Int) raises -> None:
        self = Array(self.device_array[] * other)

    def __sub__(self, other: Array) raises -> Array:
        return Array(self.device_array[] - other.device_array[])

    def __sub__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(self.device_array[] - other)

    def __sub__(self, other: Int) raises -> Array:
        return Array(self.device_array[] - other)

    def __rsub__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(other - self.device_array[])

    def __rsub__(self, other: Int) raises -> Array:
        return Array(other - self.device_array[])

    def __isub__(mut self, other: Array) raises -> None:
        self = Array(self.device_array[] - other.device_array[])

    def __isub__(mut self, other: SIMD[_, 1]) raises -> None:
        self = Array(self.device_array[] - other)

    def __isub__(mut self, other: Int) raises -> None:
        self = Array(self.device_array[] - other)

    def __truediv__(self, other: Array) raises -> Array:
        return Array(self.device_array[] / other.device_array[])

    def __truediv__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(self.device_array[] / other)

    def __truediv__(self, other: Int) raises -> Array:
        return Array(self.device_array[] / other)

    def __rtruediv__(self, other: SIMD[_, 1]) raises -> Array:
        return Array(other / self.device_array[])

    def __rtruediv__(self, other: Int) raises -> Array:
        return Array(other / self.device_array[])

    def __itruediv__(mut self, other: Array) raises -> None:
        self = Array(self.device_array[] / other.device_array[])

    def __itruediv__(mut self, other: SIMD[_, 1]) raises -> None:
        self = Array(self.device_array[] / other)

    def __itruediv__(mut self, other: Int) raises -> None:
        self = Array(self.device_array[] / other)

    def __neg__(self) raises -> Array:
        return Array(-self.device_array[])

    def __matmul__(self, other: Array) raises -> Array:
        return Array(self.device_array[] @ other.device_array[])

    def T(self, x: Int = -2, y: Int = -1) raises -> Array:
        return Array(self.device_array[].T(x, y))

    def reshape(self, shape: List[Int]) raises -> Array:
        return Array(self.device_array[].reshape(shape))

    def __pow__(self, exp: DeviceArray) raises -> Array:
        return Array(self.device_array[] ** exp)

    def __pow__(self, exp: SIMD[_, 1]) raises -> Array:
        return Array(self.device_array[] ** exp)

    def __pow__(self, exp: Int) raises -> Array:
        return Array(self.device_array[] ** exp)

    def __rpow__(self, exp: SIMD[_, 1]) raises -> Array:
        return Array(exp ** self.device_array[])

    def __rpow__(self, exp: Int) raises -> Array:
        return Array(exp ** self.device_array[])


comptime Tensor = Array


def ones(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(devar.ones(shape, dtype, requires_grad, execution_context))


def ones_like(
    array: Array,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(
        devar.ones_like(
            array.device_array[], dtype, requires_pullback, execution_context
        )
    )


def full(
    shape: ShapeType,
    fill_value: SIMD[_, 1],
    dtype: DType = fill_value.dtype,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(
        devar.full(
            shape, fill_value, dtype, requires_pullback, execution_context
        )
    )


def arange(
    start: Float32,
    end: Float32,
    step: Float32,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(
        devar.arange(
            start, end, step, dtype, requires_pullback, execution_context
        )
    )


def arange(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(devar.arange(shape, dtype, requires_grad, execution_context))


def zeros(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(devar.zeros(shape, dtype, requires_grad, execution_context))


def zeros_like(
    array: Array,
    dtype: DType = DType.float32,
    requires_pullback: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
) raises -> Array:
    return Array(
        devar.zeros_like(
            array.device_array[], dtype, requires_pullback, execution_context
        )
    )


def randn(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
    seed: Optional[Int] = None,
    mean: Float64 = Float64(0.0),
    variance: Float64 = Float64(1.0),
) raises -> Array:
    return Array(
        devar.randn(
            shape,
            dtype,
            requires_grad,
            execution_context,
            seed,
            mean,
            variance,
        )
    )


def rand(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
    seed: Optional[Int] = None,
    min: Float64 = Float64(0.0),
    max: Float64 = Float64(1.0),
) raises -> Array:
    return Array(
        devar.rand(
            shape, dtype, requires_grad, execution_context, seed, min, max
        )
    )


def he_normal(
    shape: ShapeType,
    dtype: DType = DType.float32,
    requires_grad: Bool = False,
    execution_context: Optional[ExecutionContext] = None,
    seed: Optional[Int] = None,
) raises -> Array:
    return Array(
        devar.he_normal(shape, dtype, requires_grad, execution_context, seed)
    )
