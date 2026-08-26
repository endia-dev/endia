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

from std.collections import Optional
from std.utils import Variant
from std.collections import Dict
from std.memory import ArcPointer

from nabla.api.array import Array
from nabla.core.device_array import DeviceArray
from nabla.engine.trafos.vjp_trafo import compute_cotangent
from nabla.api.utils import ExecutionContext
from nabla.engine.executor import Executor
from nabla.api.array import zeros
from nabla.api.utils import none


comptime axes_type = Variant[
    Int,
    List[Int],
    List[Optional[Int]],
    Tuple[Int],
    Tuple[Int, Int],
    Tuple[Int, Int, Int],
    Tuple[Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int, Int, Int],
    Tuple[Int, Int, Int, Int, Int, Int, Int, Int],
]


def get_axes(axes: axes_type) -> List[Int]:
    if axes.isa[Int]():
        var res = List[Int]()
        res.append(axes[Int])
        return res
    elif axes.isa[List[Int]]():
        var res = List[Int]()
        for axis in axes[List[Int]]:
            res.append(axis)
        return res
    elif axes.isa[List[Optional[Int]]]():
        var res = List[Int]()
        for axis in axes[List[Optional[Int]]]:
            if axis:
                res.append(axis.value())
            else:
                res.append(none)
        return res
    elif axes.isa[Tuple[Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int]]
        res.append(tuple[0])
        return res
    elif axes.isa[Tuple[Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        return res
    elif axes.isa[Tuple[Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        return res

    elif axes.isa[Tuple[Int, Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        res.append(tuple[3])
        return res

    elif axes.isa[Tuple[Int, Int, Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        res.append(tuple[3])
        res.append(tuple[4])
        return res

    elif axes.isa[Tuple[Int, Int, Int, Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int, Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        res.append(tuple[3])
        res.append(tuple[4])
        res.append(tuple[5])
        return res

    elif axes.isa[Tuple[Int, Int, Int, Int, Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int, Int, Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        res.append(tuple[3])
        res.append(tuple[4])
        res.append(tuple[5])
        res.append(tuple[6])
        return res

    elif axes.isa[Tuple[Int, Int, Int, Int, Int, Int, Int, Int]]():
        var res = List[Int]()
        var tuple = axes[Tuple[Int, Int, Int, Int, Int, Int, Int, Int]]
        res.append(tuple[0])
        res.append(tuple[1])
        res.append(tuple[2])
        res.append(tuple[3])
        res.append(tuple[4])
        res.append(tuple[5])
        res.append(tuple[6])
        res.append(tuple[7])
        return res

    else:
        raise "Error: Invalid axes type. Use _None comptime to use Tuples."


@fieldwise_init
struct TrafoMeta(Copyable, Movable):
    var data: ArcPointer[Dict[String, List[Int]]]

    def __init__(out self) raises:
        self.data = ArcPointer(Dict[String, List[Int]]())

    def __getitem__(self, key: String) raises -> List[Int]:
        return self.data[][key].copy()

    def __setitem__(mut self, key: String, value: List[Int]) raises:
        self.data[][key] = value.copy()

    def __contains__(self, key: String) raises -> Bool:
        return key in self.data[]


def reset_full_trace_recursively_jvp(mut array: DeviceArray) raises -> None:
    if not array.impl[]._compute_jvp:
        return

    array.impl[]._compute_jvp = False

    for arg in array.args():
        var parent = arg
        reset_full_trace_recursively_jvp(parent)


def get_full_trace_recursively_jvp(
    mut trace: List[DeviceArray], mut array: DeviceArray
) raises -> None:
    if array.impl[]._compute_jvp or not array.impl[]._jvp:
        return

    array.impl[]._compute_jvp = True

    for arg in array.args():
        var parent = arg
        get_full_trace_recursively_jvp(trace, parent)

    trace.append(array)


def default_start_rule(
    mut args: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    return args.copy()


def default_call(
    meta: TrafoMeta,
    args: List[Array],
) raises -> List[Array]:
    return args.copy()


def default_end_rule(
    mut args: List[Array],
    mut res: List[Array],
    mut meta: TrafoMeta,
) raises -> List[Array]:
    return res.copy()


@fieldwise_init
struct Callable(Copyable, Movable):
    var func: Optional[def (List[Array]) raises thin -> List[Array]]
    var pre: Optional[def (mut List[Array], mut TrafoMeta) raises thin -> List[Array]]
    var call: Optional[
        def (
            TrafoMeta,
            List[Array],
        ) raises thin -> List[Array]
    ]
    var post: Optional[
        def (
            mut List[Array], mut List[Array], mut TrafoMeta
        ) raises thin -> List[Array]
    ]
    var meta: TrafoMeta
    var trafos: List[ArcPointer[Self]]
    var const_args: List[Array]
    var execution_context: Optional[ExecutionContext]

    def __init__(
        out self,
        callable: Self,
        mut meta: TrafoMeta,
        pre: def (mut List[Array], mut TrafoMeta) raises thin -> List[
            Array
        ] = default_start_rule,
        call: def (
            TrafoMeta,
            List[Array],
        ) raises thin -> List[Array] = default_call,
        post: def (
            mut List[Array], mut List[Array], mut TrafoMeta
        ) raises thin -> List[Array] = default_end_rule,
        const_args: List[Array] = List[Array](),
    ) raises:
        self.func = None
        self.pre = pre
        self.call = call
        self.post = post
        self.meta = meta.copy()
        self.trafos = [ArcPointer[Self](callable.copy())]
        self.const_args = const_args.copy()
        self.execution_context = None

    def __init__(
        out self,
        func: def (List[Array]) raises thin -> List[Array],
        mut meta: TrafoMeta,
        pre: def (mut List[Array], mut TrafoMeta) raises thin -> List[
            Array
        ] = default_start_rule,
        call: def (
            TrafoMeta,
            List[Array],
        ) raises thin -> List[Array] = default_call,
        post: def (
            mut List[Array], mut List[Array], mut TrafoMeta
        ) raises thin -> List[Array] = default_end_rule,
        const_args: List[Array] = List[Array](),
    ) raises:
        self.func = func
        self.pre = pre
        self.call = call
        self.post = post
        self.meta = meta.copy()
        self.trafos = List[ArcPointer[Self]]()
        self.const_args = const_args.copy()
        self.execution_context = None

    def __call__(self, args: List[Array] = List[Array]()) raises -> List[Array]:
        var adapted_args = self.const_args.copy() + args.copy()
        var meta = self.meta.copy()
        var res: List[Array]

        if self.pre:
            adapted_args = self.pre.value()(adapted_args, meta)

        if self.execution_context:
            for arg in adapted_args:
                arg.device_array[].impl[].execution_context = (
                    self.execution_context.value()
                )

        if self.func:
            var _adapted_args = self.call.value()(meta, adapted_args)
            res = self.func.value()(_adapted_args)

        elif len(self.trafos) == 1:
            var child_trafo = self.trafos[0]
            var _adapted_args = self.call.value()(meta, adapted_args)
            res = child_trafo[](_adapted_args)
        else:
            raise "Error in Callable struct."

        if self.post:
            res = self.post.value()(adapted_args, res, meta)

        return res^

    def __call__(self, arg0: Array, arg1: Array) raises -> List[Array]:
        return self([arg0, arg1])

    def __call__(
        self, arg0: List[Array], arg1: List[Array]
    ) raises -> List[Array]:
        return self(arg0.copy() + arg1.copy())


@fieldwise_init
struct GraphRepr(Copyable, Movable):
    var callable: ArcPointer[Callable]

    def __init__(out self, callable: Callable) raises:
        self.callable = ArcPointer(callable.copy())

    def __call__(self, args: List[Array]) raises -> String:
        var res = self.callable[](args)
        var device_arrays = List[DeviceArray]()
        for array in res:
            device_arrays.append(array.device_array[])
        var ctx = ExecutionContext()
        var executor = Executor(device_arrays, ctx)
        return String(executor)


def callable(
    func: Variant[def (List[Array]) raises thin -> List[Array], Callable],
) raises -> Callable:
    if func.isa[Callable]():
        return func[Callable].copy()
    else:
        var meta = TrafoMeta()
        return Callable(
            func[def (List[Array]) raises thin -> List[Array]],
            meta,
            pre=default_start_rule,
            call=default_call,
            post=default_end_rule,
        )


def std_basis(
    args: List[Array],
) raises -> Tuple[List[Int], List[Array]]:
    var num_total_arg_elements = 0
    var max_rank = 0
    for arg in args:
        var num_elements = 1
        var batch_dim_ctr = arg.batch_dim_ctr()
        batch_dim_ctr = batch_dim_ctr if batch_dim_ctr != none else 0
        for dim in arg.shape()[batch_dim_ctr:]:
            num_elements *= dim
        num_total_arg_elements += num_elements
        var rank = len(arg.shape()[batch_dim_ctr:])
        if rank > max_rank:
            max_rank = rank

    var batch_ctr = 0
    var sizes = List[Int]()

    var tangents = List[Array]()

    for arg in args:
        var num_elements = 1

        var batch_dim_ctr = arg.batch_dim_ctr()
        batch_dim_ctr = batch_dim_ctr if batch_dim_ctr != none else 0
        for dim in arg.shape()[batch_dim_ctr:]:
            num_elements *= dim

        arg.device_array[].impl[]._compute_jvp = True
        var arg_batch_ctr = arg.batch_dim_ctr()
        arg_batch_ctr = arg_batch_ctr if arg_batch_ctr != none else 0
        var batched_shape = arg.shape()
        for _ in range(max_rank - len(batched_shape)):
            batched_shape.insert(0, 1)

        var _bs = List(batched_shape[:arg_batch_ctr])
        _bs.append(num_total_arg_elements)
        _bs += List(batched_shape[arg_batch_ctr:])
        batched_shape = _bs^
        var dtype = arg.dtype()
        var tangent = zeros(batched_shape.copy(), dtype)

        var total_elements = 1
        for dim in batched_shape:
            total_elements *= dim

        var num_els_batch_dims = 1
        for dim in arg.shape()[:batch_dim_ctr]:
            num_els_batch_dims *= dim

        for i in range(num_els_batch_dims):
            var offset = batch_ctr + num_total_arg_elements * num_elements * i

            for j in range(num_elements):
                var idx = offset + j
                tangent.store[DType.float32, 1](idx, Float32(1.0))
                offset += num_elements

        batch_ctr += num_elements * num_elements
        tangent.batch_dim_ctr_((arg.batch_dim_ctr()))
        tangents.append(tangent)
        sizes.append(num_elements)

    return sizes^, tangents^


def get_full_trace_recursively(
    mut trace: List[DeviceArray], mut array: DeviceArray
) raises -> None:
    # TODO: The folloing loop is to maintain materialization of certain arrays,
    # however in fact it should NOT be needed, if we remove it the code breaks
    # when we compute the loss after the updates of the weights and biases.
    if not array.not_to_be_materialized():
        array.is_tmp_output_(True)

    if array.visited() or not array.impl[]._diffable:
        return

    array.visited_(True)

    for arg in array.args():
        var parent = arg
        get_full_trace_recursively(trace, parent)

    array.id_(len(trace))
    trace.append(array)


def reset_visited(mut trace: List[DeviceArray]) raises -> None:
    for ref array in trace:
        array.visited_(False)


def cotangent_with_remat(
    outs: List[DeviceArray], keep_graph: Bool = True
) raises -> List[DeviceArray]:
    var trace = List[DeviceArray]()

    for output in outs:
        var parent = output
        get_full_trace_recursively(trace, parent)

    reset_visited(trace)

    for ref array in trace:
        if (
            not array.impl[].is_checkpoint
            and (not array.impl[].requires_pullback)
            and array.impl[]._diffable
            and (not array.is_tmp_output())
        ):
            var dual_args = array.args()

            for ref i in range(len(dual_args)):
                var arg = dual_args[i]
                if arg.has_dual():
                    dual_args[i] = arg.dual()

            var dual = DeviceArray(ArcPointer(array.impl[].copy()))
            dual.name_("dual_" + array.name())
            dual.args_(dual_args)
            array.dual_(dual)

    var cotangents = List[DeviceArray]()

    for i in range(len(trace) - 1, -1, -1):
        var p_array = trace[i]
        var array = p_array

        if (
            len(array.args()) == 0
            or not array.impl[]._diffable
            or array.impl[].requires_pullback
        ):
            continue

        if len(array.impl[]._dual) == 1:
            array = DeviceArray(p_array.impl[]._dual[0])
            if (
                len(p_array.impl[].cotangent) == 1
                and len(array.impl[].cotangent) == 0
            ):
                array.impl[].cotangent = p_array.impl[].cotangent.copy()

        compute_cotangent(array, cotangents)
        array.impl[].cotangent.clear()
        p_array.impl[].cotangent.clear()

    for array in trace:
        array.impl[]._dual.clear()

    if not keep_graph:
        for ref cotangent in cotangents:
            cotangent.requires_pullback_(False)

    return cotangents.copy()
