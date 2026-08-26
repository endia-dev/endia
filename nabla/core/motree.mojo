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
#
# Mojo 1.0 port note: the original stored
# `Variant[Array, List[Array], Self, List[Self]]` directly, but Mojo 1.0 no
# longer accepts a self-recursive Variant as a struct field. Subtrees are now
# held behind `ArcPointer[MoTree]`; the public overloads keep the old
# call-sites (tuple-variadic `motree(...)`, `tree[key][List[Array]]`) working.

from std.collections import Dict, Optional
from std.memory import ArcPointer
from std.utils import Variant
from nabla.api.array import Array


struct MoTree(Copyable, Movable):
    comptime Stored = Variant[
        Array, List[Array], ArcPointer[Self], List[ArcPointer[Self]]
    ]

    var data: List[Dict[String, Self.Stored]]

    def __init__(out self) raises:
        self.data = List[Dict[String, Self.Stored]]()

    def __init__(out self, var data: List[Dict[String, Self.Stored]]):
        self.data = data^

    def _ensure_root(mut self) raises:
        if len(self.data) == 0:
            self.data.append(Dict[String, Self.Stored]())

    def insert(mut self, key: String, value: Array) raises:
        self._ensure_root()
        self.data[0][key] = Self.Stored(value)

    def insert(mut self, key: String, value: List[Array]) raises:
        self._ensure_root()
        self.data[0][key] = Self.Stored(value.copy())

    def insert(mut self, key: String, value: Self) raises:
        self._ensure_root()
        self.data[0][key] = Self.Stored(ArcPointer(value.copy()))

    def insert(mut self, key: String, value: List[Self]) raises:
        self._ensure_root()
        var subtrees = List[ArcPointer[Self]]()
        for tree in value:
            subtrees.append(ArcPointer(tree.copy()))
        self.data[0][key] = Self.Stored(subtrees^)

    def __setitem__(mut self, key: String, value: Array) raises:
        self.insert(key, value)

    def __setitem__(mut self, key: String, value: List[Array]) raises:
        self.insert(key, value)

    def __setitem__(mut self, key: String, value: Self) raises:
        self.insert(key, value)

    def __setitem__(mut self, key: String, value: List[Self]) raises:
        self.insert(key, value)

    def __getitem__(self, key: String = "") raises -> Self.Stored:
        if len(self.data) != 1 or key not in self.data[0]:
            raise "Key not found" + key
        return self.data[0][key].copy()

    def retreive_leaves(self, mut leaves: List[Array], curr: Self.Stored) raises:
        if curr.isa[Array]():
            leaves.append(curr[Array])
        elif curr.isa[List[Array]]():
            for val in curr[List[Array]]:
                leaves.append(val)
        elif curr.isa[ArcPointer[Self]]():
            for key in curr[ArcPointer[Self]][].data[0].keys():
                self.retreive_leaves(
                    leaves, curr[ArcPointer[Self]][].data[0][key]
                )
        elif curr.isa[List[ArcPointer[Self]]]():
            for subtree in curr[List[ArcPointer[Self]]]:
                for key in subtree[].data[0].keys():
                    self.retreive_leaves(leaves, subtree[].data[0][key])

    def flatten(self) raises -> List[Array]:
        var leaves = List[Array]()
        for key in self.data[0].keys():
            self.retreive_leaves(leaves, self.data[0][key])
        return leaves^


def motree(key: String, value: Array) raises -> MoTree:
    var new_tree = MoTree()
    new_tree.insert(key, value)
    return new_tree^


def motree(key: String, value: List[Array]) raises -> MoTree:
    var new_tree = MoTree()
    new_tree.insert(key, value)
    return new_tree^


def motree(key: String, value: MoTree) raises -> MoTree:
    var new_tree = MoTree()
    new_tree.insert(key, value)
    return new_tree^


def motree(key: String, value: List[MoTree]) raises -> MoTree:
    var new_tree = MoTree()
    new_tree.insert(key, value)
    return new_tree^


def motree(*key_value_pairs: Tuple[String, Array]) raises -> MoTree:
    var new_tree = MoTree()
    for key_value_pair in key_value_pairs:
        new_tree.insert(key_value_pair[0], key_value_pair[1])
    return new_tree^


def motree(*key_value_pairs: Tuple[String, List[Array]]) raises -> MoTree:
    var new_tree = MoTree()
    for key_value_pair in key_value_pairs:
        new_tree.insert(key_value_pair[0], key_value_pair[1])
    return new_tree^


def motree(*key_value_pairs: Tuple[String, MoTree]) raises -> MoTree:
    var new_tree = MoTree()
    for key_value_pair in key_value_pairs:
        new_tree.insert(key_value_pair[0], key_value_pair[1])
    return new_tree^


def motree(*key_value_pairs: Tuple[String, List[MoTree]]) raises -> MoTree:
    var new_tree = MoTree()
    for key_value_pair in key_value_pairs:
        new_tree.insert(key_value_pair[0], key_value_pair[1])
    return new_tree^


def motree(*trees: MoTree) raises -> MoTree:
    var new_tree = MoTree()
    var subtrees = List[MoTree]()
    for tree in trees:
        subtrees.append(tree.copy())
    new_tree.insert("_", subtrees)
    return new_tree^
