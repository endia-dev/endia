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

from std.collections.string import StaticString, StringSlice
from std.os import abort
from std.pathlib import Path
from std.ffi import c_char, external_call
from nabla.compiler._dlhandle import DLHandle

from std.memory.unsafe_pointer import *
from std.memory import alloc


struct CString(TrivialRegisterPassable, ImplicitlyCopyable, Writable):
    """Represents `const char*` in C. Useful for binding with C APIs."""

    var ptr: UnsafePointer[c_char, MutUntrackedOrigin]

    @implicit
    def __init__(out self, ptr: UnsafePointer[c_char, MutUntrackedOrigin]):
        """
        Construct a `CString` from a C string data pointer.

        Args:
            ptr: The string data pointer to wrap.
        """
        self.ptr = ptr.bitcast[c_char]()

    def get_as_string_ref(self) -> StaticString:
        """
        Get the `CString` as `StringRef`. Origin is tied to C API.
        For owning version use `__str__()`.
        """
        return StaticString(unsafe_from_utf8_ptr=self.ptr)

    def __str__(self) -> String:
        """
        Get `CString` as a owning `String`.
        """
        return String(unsafe_from_utf8_ptr=self.ptr)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())


@always_inline("nodebug")
def exchange[T: TrivialRegisterPassable](mut old_var: T, var new_value: T) -> T:
    """
    Assign `new_value` to `old_var` and returns the value previously
    contained in `old_var`.
    """
    var old = old_var
    old_var = new_value
    return old




@always_inline("nodebug")
def mut_ptr[T: AnyType](p: UnsafePointer[T, ImmUntrackedOrigin]) -> UnsafePointer[T, MutUntrackedOrigin]:
    """Reinterpret an immutable untracked pointer as mutable (FFI plumbing)."""
    var addr = Int(p)
    return UnsafePointer(to=addr).bitcast[UnsafePointer[T, MutUntrackedOrigin]]()[]


@always_inline("nodebug")
def null_ptr[T: AnyType = NoneType]() -> UnsafePointer[T, MutUntrackedOrigin]:
    """A NULL pointer value. Mojo 1.0's `UnsafePointer` has no null
    constructor (non-null by design), but the vendored MAX FFI wrappers use
    NULL as a moved-from/absent sentinel that the C side also produces."""
    var zero: Int = 0
    return UnsafePointer(to=zero).bitcast[UnsafePointer[T, MutUntrackedOrigin]]()[]


# ======================================================================#
#                                                                       #
# Utility structs and functions to interact with dylibs.                #
#                                                                       #
# ======================================================================#


@always_inline("nodebug")
def call_dylib_func[
    ReturnType: RegisterPassable = NoneType,
    *Args: AnyType,
](lib: DLHandle, name: StringSlice, *args: *Args) -> ReturnType:
    var func_ptr = lib.get_function[
        def (*a: *Args) thin abi("C") -> ReturnType
    ](String(name))

    return func_ptr(*args)


struct OwningVector[T: Movable & Deinitable](Sized):
    var ptr: UnsafePointer[Self.T, MutUntrackedOrigin]
    var size: Int

    comptime initial_capacity = 5
    var capacity: Int

    def __init__(out self):
        var ptr = alloc[Self.T](Self.initial_capacity)
        self.ptr = ptr
        self.size = 0
        self.capacity = Self.initial_capacity

    def __init__(out self, *, deinit existing: Self):
        self.ptr = existing.ptr
        self.size = existing.size
        self.capacity = existing.capacity

    def emplace_back(mut self, var value: Self.T):
        if self.size < self.capacity:
            (self.ptr + self.size).init_pointee_move(value^)
            self.size += 1
            return

        self.capacity = self.capacity * 2
        var new_ptr = alloc[Self.T](self.capacity)
        for i in range(self.size):
            (new_ptr + i).init_pointee_move((self.ptr + i).take_pointee())
        self.ptr.free()
        self.ptr = new_ptr
        self.emplace_back(value^)

    def get(self, idx: Int) raises -> UnsafePointer[Self.T, MutUntrackedOrigin]:
        if idx >= self.size:
            raise Error(
                "requested index(",
                idx,
                ") exceeds size of vector(",
                self.size,
                ")",
            )
        return self.ptr + idx

    def __len__(self) -> Int:
        return self.size

    def __deinit__(deinit self):
        for i in range(self.size):
            (self.ptr + i).destroy_pointee()
        self.ptr.free()


def get_lib_path_from_cfg(
    name: StringSlice, err_name: StaticString
) raises -> String:
    # TODO: Move KGEN_CompilerRT_getMAXConfigValue to a helper somewhere.
    var lib_path_str_ptr = external_call[
        "KGEN_CompilerRT_getMAXConfigValue", UnsafePointer[UInt8, MutUntrackedOrigin]
    ](name.unsafe_ptr(), name.byte_length())

    if Int(lib_path_str_ptr) == 0:
        raise Error(
            "cannot get the location of ", name, " library from modular.cfg"
        )

    var lib_path = String(unsafe_from_utf8_ptr=lib_path_str_ptr)
    lib_path_str_ptr.free()

    if not Path(lib_path).exists():
        raise String(err_name) + " not found at " + lib_path
    return lib_path
