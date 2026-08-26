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


def test_motree() raises:
    var tree = nabla.motree(
        (
            String("params"),
            nabla.motree(
                (String("weight_1"), nabla.arange((2, 4))),
                (String("bias_1"), nabla.arange((2, 5))),
            ),
        ),
        (
            String("velocities"),
            nabla.motree(
                (String("weight_1"), nabla.arange((2, 4))),
                (String("bias_1"), nabla.arange((2, 5))),
            ),
        ),
        (
            String("tangents"),
            nabla.motree(
                (String("weight_1"), nabla.arange((2, 4))),
                (String("bias_1"), nabla.arange((2, 5))),
            ),
        ),
    )

    var flattened_tree = tree.flatten()
    for leaf in flattened_tree:
        print(leaf[])


def test_motree_func() raises:
    def foo(args: nabla.MoTree) raises -> nabla.MoTree:
        var params = args["params"][List[nabla.Array]].copy()
        var grads = args["grads"][List[nabla.Array]].copy()

        var updated_params = List[nabla.Array]()
        for i in range(len(params)):
            var updated_param = params[i] + 0.1 * grads[i]
            updated_params.append(updated_param)

        var outputs = nabla.motree(
            (String("params"), updated_params.copy()),
        )
        return outputs.copy()

    var params = List[nabla.Array]()
    var grads = List[nabla.Array]()

    for _ in range(3):
        params.append(nabla.arange((2, 4)))
        grads.append(nabla.arange((2, 4)))

    var args = nabla.motree(
        (String("params"), params.copy()),
        (String("grads"), grads.copy()),
    )

    var outputs = foo(args)
    var flattened_outputs = outputs.flatten()
    for i in range(len(flattened_outputs)):
        print("\nparam", i)
        print(flattened_outputs[i])
