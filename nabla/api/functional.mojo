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


from nabla.engine.utils import (
    TrafoMeta,
    std_basis,
    get_full_trace_recursively_jvp,
    Callable,
    callable,
)
from nabla.api.utils import ExecutionContext
from nabla.api.array import Array
from nabla.engine.trafos.jacfwd_trafo import jacfwd_end_rule
from nabla.engine.trafos.jvp_trafo import jvp_call, jvp_end_rule
from nabla.engine.trafos.jacrev_trafo import (
    jacrev_start_rule,
    jacrev_call,
    jacrev_end_rule,
)
from nabla.engine.trafos.vjp_trafo import vjp_call, vjp_end_rule, backward
from nabla.engine.trafos.jit_trafo import set_execution_context_recursively
from nabla.engine.trafos.vmap_trafo import vmap_start_rule, vmap_end_rule
from nabla.engine.trafos.grad_trafo import grad_call, grad_end_rule
from nabla.engine.utils import GraphRepr
from std.memory import ArcPointer


def jacfwd(
    callable: Callable,
) raises -> Callable:
    var meta = TrafoMeta()
    return Callable(
        callable,
        meta,
        post=jacfwd_end_rule,
    )


def jacfwd(
    func: def (List[Array]) raises thin -> List[Array],
) raises -> Callable:
    return jacfwd(callable(func))


def jacrev(callable: Callable, remat: Bool = False) raises -> Callable:
    var meta = TrafoMeta()
    meta["with_remat"] = [Int(remat)]
    return Callable(
        callable,
        meta,
        jacrev_start_rule,
        jacrev_call,
        jacrev_end_rule,
    )


def jacrev(
    func: def (List[Array]) raises thin -> List[Array],
    remat: Bool = False,
) raises -> Callable:
    return jacrev(callable(func), remat)


def grad(callable: Callable, remat: Bool = False) raises -> Callable:
    var meta = TrafoMeta()
    meta["with_remat"] = [Int(remat)]
    return Callable(
        callable,
        meta,
        call=grad_call,
        post=grad_end_rule,
    )


def grad(
    func: def (List[Array]) raises thin -> List[Array], remat: Bool = False
) raises -> Callable:
    return grad(callable(func), remat)


def jit(func: Callable) raises -> Callable:
    var meta = TrafoMeta()
    var execution_context = ExecutionContext()
    callable_ref = ArcPointer(Callable(func, meta=meta))
    set_execution_context_recursively(callable_ref, execution_context)
    return callable_ref[].copy()


def jit(func: def (List[Array]) raises thin -> List[Array]) raises -> Callable:
    return jit(callable(func))


def jvp(
    func: Callable,
    primals: List[Array],
    tangents: List[Array],
) raises -> Tuple[List[Array], List[Array]]:
    var meta = TrafoMeta()
    var res = Callable(
        func,
        meta,
        call=jvp_call,
        post=jvp_end_rule,
        const_args=primals + tangents.copy(),
    )()
    var num_res = meta["num_res"][0]
    return List(res[:num_res]), List(res[num_res:])


def jvp(
    func: def (List[Array]) raises thin -> List[Array],
    primals: List[Array],
    tangents: List[Array],
) raises -> Tuple[List[Array], List[Array]]:
    return jvp(callable(func), primals, tangents)


def vjp(
    func: Callable,
    primals: List[Array],
    remat: Bool = False,
) raises -> Tuple[List[Array], Callable]:
    var meta = TrafoMeta()
    meta["with_remat"] = [Int(remat)]
    meta["num_primals"] = [len(primals)]
    return func(primals), Callable(
        func,
        meta,
        call=vjp_call,
        post=vjp_end_rule,
        const_args=primals,
    )


def vjp(
    func: def (List[Array]) raises thin -> List[Array],
    primals: List[Array],
    remat: Bool = False,
) raises -> Tuple[List[Array], Callable]:
    return vjp(callable(func), primals, remat)


def vmap(
    func: Callable,
    in_axes: List[Int] = List[Int](),
    out_axes: List[Int] = List[Int](),
) raises -> Callable:
    var meta = TrafoMeta()
    meta["in_axes"] = in_axes
    meta["out_axes"] = out_axes
    return Callable(
        func,
        meta,
        pre=vmap_start_rule,
        post=vmap_end_rule,
    )


def vmap(
    func: def (List[Array]) raises thin -> List[Array],
    in_axes: List[Int] = List[Int](),
    out_axes: List[Int] = List[Int](),
) raises -> Callable:
    return vmap(callable(func), in_axes, out_axes)
