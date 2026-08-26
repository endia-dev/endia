# ===----------------------------------------------------------------------=== #
# Nabla 2025
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or beautiful, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Structs used for maintaining a collection of tensors."""
from std.collections import Dict
from std.collections.dict import _DictEntryIter, _DictKeyIter

from nabla.compiler.tensor import Tensor, TensorSpec
from std.memory import alloc, UnsafePointer, memcpy
from nabla.compiler._utils import null_ptr


@fieldwise_init
struct _CheckpointTensor(Copyable, Movable):
    """A wrapper around a Tensor pointer that can be saved/loaded from disk."""

    var ptr: UnsafePointer[UInt8]
    var spec: TensorSpec

    def __init__(
        out self,
        var ptr: UnsafePointer[UInt8],
        var spec: TensorSpec,
    ):
        """Creates a _CheckpointTensor.

        Args:
            ptr: UnsafePointer to tensor data.
            spec: Tensor's spec.
        """
        self.ptr = ptr
        self.spec = spec^

    def copy_to_tensor[T: DType](var self) -> Tensor[T]:
        """Returns a deep copy of the Tensor data."""
        var num_elements = self.spec.num_elements()
        var spec = self.spec
        var self_ptr = self.ptr.bitcast[Scalar[T]]()
        var ptr = alloc[Scalar[T]](num_elements)
        memcpy(ptr, self_ptr, num_elements)
        return Tensor[T](spec, ptr)

    def to_tensor[T: DType](var self) -> Tensor[T]:
        """Converts this object to a Tensor."""
        var spec = self.spec^
        var ptr = self.ptr.bitcast[Scalar[T]]()
        self.spec = TensorSpec()
        self.ptr = null_ptr[UInt8]()
        return Tensor[T](spec, ptr)

    @staticmethod
    def from_tensor(var tensor: Tensor) -> Self:
        """Creates a _CheckpointTensor from a Tensor."""
        var spec = tensor.spec()
        var ptr = tensor._steal_ptr().bitcast[Scalar[DType.uint8]]()
        return Self(ptr, spec)


struct TensorDict(Sized, Writable):
    """A collection of keyed `Tensor` values used with checkpoint files.

    This is the type accepted by
    [`save()`](/max/api/mojo/graph/checkpoint/save_load/save) and
    returned by
    [`load()`](/max/api/mojo/graph/checkpoint/save_load/load).

    For example:

    ```mojo
    from nabla.compiler.graph.checkpoint import load, save, TensorDict
    from nabla.compiler.tensor import Tensor, TensorShape

    def write_to_disk():
        tensors = TensorDict()
        tensors.set("x", Tensor[DType.int32](TensorShape(1, 2, 2), 1, 2, 3, 4))
        tensors.set("y", Tensor[DType.float32](TensorShape(10, 5), -1.23))
        save(tensors, "/path/to/checkpoint.maxckpt")

    def read_from_disk():
        tensors = load("/path/to/checkpoint.maxckpt")
        x = tensors.get[DType.int32]("x")
    ```
    """

    var _items: Dict[String, _CheckpointTensor]

    def __init__(out self):
        self._items = Dict[String, _CheckpointTensor]()

    def __setitem__[T: DType](mut self, key: String, value: Tensor[T]):

        self._items._insert(key, _CheckpointTensor.from_tensor(value))

    def set[T: DType](mut self, key: String, value: Tensor[T]):
        """Adds or updates a tensor in the dictionary.

        Args:
            key: The name of the tensor.
            value: The tensor to add.
        """
        self._items._insert(key, _CheckpointTensor.from_tensor(value))

    def get[type: DType](self, key: String) raises -> Tensor[type]:
        """Gets a tensor from the dictionary.

        Currently, this returns a copy of the tensor. For better performance,
        use `Dict.pop()`.

        This method may change in the future to return an immutable reference
        instead of a mutable tensor copy.

        Args:
            key: The name of the tensor.

        Returns:
            A copy of the tensor.
        """
        try:
            return self._items[key].copy_to_tensor[type]()
        except e:
            raise Error("Error when getting key '", key, "': ", e)

    def _set(mut self, key: String, value: _CheckpointTensor):
        """Adds or updates a tensor in the dictionary.

        Args:
            key: The name of the tensor.
            value: The tensor to add.
        """
        self._items._insert(key, value)

    def _get(self, key: String) raises -> _CheckpointTensor:
        """Gets a raw `CheckpointTensor` value from the dictionary.

        Args:
            key: The name of the tensor.
        """
        return self._items[key]

    def pop[type: DType](mut self, key: String) raises -> Tensor[type]:
        """Removes a tensor from the dictionary.

        This function moves the Tensor pointer out of the dictionary and returns
        it to the caller.

        Args:
            key: The name of the tensor.

        Returns:
            The tensor.
        """
        try:
            return self._items.pop(key).to_tensor[type]()
        except e:
            raise Error("Error when getting key '", key, "': ", e)

    def __len__(self) -> Int:
        return len(self._items)

    def items(
        ref self,
    ) -> _DictEntryIter[String, _CheckpointTensor, origin_of(self._items)]:
        """Gets an iterable view of all elements in the dictionary."""
        return _DictEntryIter(0, 0, self._items)

    def keys(
        ref self,
    ) -> _DictKeyIter[String, _CheckpointTensor, origin_of(self._items)]:
        """Gets an iterable view of all keys in the dictionary."""
        return _DictKeyIter(_DictEntryIter(0, 0, self._items))

    def __iter__(
        ref self,
    ) -> _DictKeyIter[String, _CheckpointTensor, origin_of(self._items)]:
        return _DictKeyIter(_DictEntryIter(0, 0, self._items))

    def __init__(out self, *, copy: Self):
        """Copies a dictionary.

        Args:
            existing: The existing dict.
        """
        self._items = existing._items

    def __init__(out self, *, deinit existing: Self):
        """Moves data of an existing dictionary into a new one.

        Args:
            existing: The existing dict.
        """
        self._items = existing._items^

    def __str__(self) -> String:
        return String(self)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("TensorDict(")
        var first = True
        for key_ref in self._items.keys():
            var key = key_ref
            if first:
                first = False
            else:
                writer.write(", ")
            writer.write(key, ": ")
            try:
                writer.write(self._items[key].spec)
            except:
                # Should never happen.
                writer.write("(contents could not be read)")
        writer.write(")")
