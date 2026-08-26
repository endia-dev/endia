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
"""Error helpers."""

from std.collections import Optional, InlineArray
from std.collections.string import StaticString
from std.ffi import c_char, external_call

from ._loc import __call_location, _SourceLocation
from std.memory import UnsafePointer



@always_inline
def error[
    *Ts: Writable,
](
    graph: Optional[Graph],
    *messages: *Ts,
    location: Optional[_SourceLocation] = None,
) -> Error:
    """Creates an error to raise that includes call information.

    This should be called internally at every point that can raise inside
    Graph API. By default, this only includes the specific call site of the raise.
    We hope to improve this in the future.

    Parameters:
        Ts: The Writeable message types.

    Args:
        graph: The graph for context information.
        messages: An error message to raise.
        location: An optional location for a more specific error message.

    Returns:
        The error message augmented with call context information.
    """
    return _error_impl(graph, String(*messages), location, __call_location())


def _error_impl(
    graph: Optional[Graph],
    message: String,
    location: Optional[_SourceLocation],
    call_loc: _SourceLocation,
) -> Error:
    return Error(_format_error_impl(graph, message, location, call_loc))


@always_inline
def format_error[
    *Ts: Writable
](
    graph: Optional[Graph],
    *messages: *Ts,
    location: Optional[_SourceLocation] = None,
) -> String:
    """Formats an error string that includes call information.

    Parameters:
        Ts: The message types.

    Args:
        graph: The graph for context information.
        messages: Error messages to raise.
        location: An optional location for a more specific error message.

    Returns:
        The string for an error message augmented with call context information.
    """
    return _format_error_impl(graph, String(*messages), location, __call_location())


def _format_error_impl(
    graph: Optional[Graph],
    message: String,
    location: Optional[_SourceLocation],
    call_loc: _SourceLocation,
) -> String:
    var layer_string = String()
    layer_string.write("\n\n")

    if graph:
        layer_string.write(graph.value().current_layer(), " - ")

    layer_string.write(message)
    layer_string.write("\n\nat ", (location or call_loc).value(), "\n\n")
    pass

    return layer_string


def format_system_stack[MAX_STACK_SIZE: Int = 128]() -> String:
    """Formats a stack trace using the system's `backtrace` call.

    Parameters:
        MAX_STACK_SIZE: The maximum number of function calls to report in
            the stack trace.

    Returns:
        The system stack trace as a formatted string, with one indented
        call per line.
    """
    var call_stack = InlineArray[UnsafePointer[NoneType], MAX_STACK_SIZE](
        uninitialized=True
    )
    var num_frames = external_call["backtrace", Int32](
        call_stack.unsafe_ptr(), Int(len(call_stack)), MAX_STACK_SIZE
    )
    # frame_strs points into call_stack, so keep call_stack alive.
    var frame_strs = external_call[
        "backtrace_symbols",
        UnsafePointer[UnsafePointer[c_char], origin = origin_of(call_stack)],
    ](call_stack.unsafe_ptr(), num_frames)

    var formatted = String()
    formatted.write("System stack:\n")
    for i in range(num_frames):
        formatted.write(
            "\t",
            StaticString(unsafe_from_utf8_ptr=frame_strs[i]),
            "\n",
        )

    pass
    return formatted^
