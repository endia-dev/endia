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
from std.ffi import external_call
from nabla.compiler._utils import mut_ptr

import _mlir
from std.memory import UnsafePointer

# ===-----------------------------------------------------------------------===#
# Library Load
# ===-----------------------------------------------------------------------===#

# Mojo 1.0 port: the MAX graph library and its loader
# (modular.cfg + _Global dylib caching) do not exist; cfunc aborts on use.


@always_inline
def cfunc[func_name: StaticString, T: TrivialRegisterPassable]() -> T:
    abort(
        "the MAX graph library is unavailable on Mojo 1.0 (needed symbol: "
        + String(func_name)
        + ")"
    )


# Note: Keep sections below in sync with max_graph.cpp, including order, grouping
# and naming.

# Note: Please keep the following naming convention: conept_function. For
# example graph_new, etc.


# ===-----------------------------------------------------------------------===#
# Op factories
# ===-----------------------------------------------------------------------===#


def graph_new(
    module: _mlir.Module,
    loc: _mlir.Location,
    name: String,
    signature: _mlir.builtin_types.FunctionType,
) -> _mlir.Operation:
    return cfunc[
        "MAXG_graphNew",
        def (
            _mlir.Module.cType,
            _mlir.Location.cType,
            StringSlice[origin_of(name)],
            _mlir.Type.cType,
        ) thin -> _mlir.Operation.cType,
    ]()(module.c, loc.c, name, signature.to_mlir())


# ===-----------------------------------------------------------------------===#
# Attribute factories
# ===-----------------------------------------------------------------------===#


def attr_new_tensor[
    T: Copyable & Movable
](
    name: String,
    data: List[T],
    type: _mlir.Type,
    is_owned: Bool,
) -> _mlir.NamedAttribute:
    return cfunc[
        "MAXG_attrNewTensor",
        def (
            StringSlice[origin_of(name)],
            UnsafePointer[T, MutUntrackedOrigin],
            _mlir.Type.cType,
            Bool,
        ) thin -> _mlir.NamedAttribute.cType,
    ]()(name, mut_ptr(data.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]()), type.c, is_owned)


def attr_new_tensor(
    name: String,
    data: UnsafePointer[NoneType, MutUntrackedOrigin],
    type: _mlir.Type,
    is_owned: Bool,
) -> _mlir.NamedAttribute:
    return cfunc[
        "MAXG_attrNewTensor",
        def (
            StringSlice[origin_of(name)],
            UnsafePointer[NoneType, MutUntrackedOrigin],
            _mlir.Type.cType,
            Bool,
        ) thin -> _mlir.NamedAttribute.cType,
    ]()(name, data, type.c, is_owned)


def attr_new_tensor_from_file(
    name: String, file_name: String, type: _mlir.Type
) -> _mlir.NamedAttribute:
    return cfunc[
        "MAXG_attrNewTensorFromFile",
        def (
            StringSlice[origin_of(name)],
            StringSlice[origin_of(file_name)],
            _mlir.Type.cType,
        ) thin -> _mlir.NamedAttribute.cType,
    ]()(name, file_name, type.c)


def attr_new_dim_param_decl(
    ctx: _mlir.Context,
    name: String,
) -> _mlir.Attribute:
    var result = cfunc[
        "MAXG_attrNewDimParamDecl",
        def (
            _mlir.Context.cType, StringSlice[origin_of(name)]
        ) thin -> _mlir.Attribute.cType,
    ]()(ctx.c, name)
    return result


def attr_new_param_decl_array(
    ctx: _mlir.Context,
    params: List[_mlir.Attribute],
) -> _mlir.Attribute:
    var result = cfunc[
        "MAXG_attrNewParamDeclArray",
        def (
            _mlir.Context.cType,
            UnsafePointer[_mlir.Attribute.cType, MutUntrackedOrigin],
            Int32,
        ) thin -> _mlir.Attribute.cType,
    ]()(
        ctx.c,
        mut_ptr(params.unsafe_ptr().bitcast[_mlir.Attribute.cType]().unsafe_origin_cast[ImmUntrackedOrigin]()),
        Int32(len(params)),
    )
    return result


def attr_new_shape(
    ctx: _mlir.Context,
    dims: List[_mlir.Attribute],
) -> _mlir.Attribute:
    var result = cfunc[
        "MAXG_attrNewShape",
        def (
            _mlir.Context.cType,
            UnsafePointer[_mlir.Attribute.cType, MutUntrackedOrigin],
            Int32,
        ) thin -> _mlir.Attribute.cType,
    ]()(
        ctx.c,
        mut_ptr(dims.unsafe_ptr().bitcast[_mlir.Attribute.cType]().unsafe_origin_cast[ImmUntrackedOrigin]()),
        Int32(len(dims)),
    )
    return result


# ===-----------------------------------------------------------------------===#
# Type helpers
# ===-----------------------------------------------------------------------===#


def dtype_new(ctx: _mlir.Context, dtype: DType) -> _mlir.Type:
    return cfunc[
        "MAXG_dTypeNew", def (_mlir.Context.cType, UInt8) thin -> _mlir.Type.cType
    ]()(ctx.c, UInt8(mlir_value=dtype._as_ui8()))


def dim_type_new_dynamic() -> Int64:
    return cfunc["MAXG_dimTypeNewDynamic", def () thin -> Int64]()()


def tensor_type_new(
    ctx: _mlir.Context,
    dtype: _mlir.Type,
    dims: List[_mlir.Attribute],
    ranked: Bool,
) -> _mlir.Type:
    var result = cfunc[
        "MAXG_tensorTypeNew",
        def (
            _mlir.Context.cType,
            _mlir.Type.cType,
            Bool,
            UnsafePointer[_mlir.Attribute.cType, MutUntrackedOrigin],
            Int32,
        ) thin -> _mlir.Type.cType,
    ]()(
        ctx.c,
        dtype.c,
        ranked,
        mut_ptr(dims.unsafe_ptr().bitcast[_mlir.Attribute.cType]().unsafe_origin_cast[ImmUntrackedOrigin]()),
        Int32(len(dims)),
    )
    return result


def tensor_type_get_dtype(v: _mlir.Type) -> DType:
    var dtype = cfunc[
        "MAXG_tensorTypeGetDType", def (_mlir.Type.cType) thin -> UInt8
    ]()(v.c)
    _ = dtype
    return DType.uint8


def tensor_type_is_ranked(v: _mlir.Type) -> Bool:
    return cfunc["MAXG_tensorTypeIsRanked", def (_mlir.Type.cType) thin -> Bool]()(
        v.c
    )


def tensor_type_get_rank(t: _mlir.Type) -> Int64:
    return cfunc["MAXG_tensorTypeGetRank", def (_mlir.Type.cType) thin -> Int64]()(
        t.c
    )


def tensor_type_get_dim(t: _mlir.Type, dim: Int64) -> _mlir.Attribute:
    return cfunc[
        "MAXG_tensorTypeShapeGetDim",
        def (_mlir.Type.cType, Int64) thin -> _mlir.Attribute.cType,
    ]()(t.c, dim)


def dim_new_dynamic(ctx: _mlir.Context) -> _mlir.Attribute:
    return cfunc[
        "MAXG_dimNewDynamic",
        def (_mlir.Context.cType) thin -> _mlir.Attribute.cType,
    ]()(ctx.c)


def dim_new_static(ctx: _mlir.Context, dim: Int64) -> _mlir.Attribute:
    return cfunc[
        "MAXG_dimNewStatic",
        def (_mlir.Context.cType, Int64) thin -> _mlir.Attribute.cType,
    ]()(ctx.c, dim)


def dim_new_symbolic(ctx: _mlir.Context, name: String) -> _mlir.Attribute:
    return cfunc[
        "MAXG_dimNewSymbolic",
        def (
            _mlir.Context.cType, StringSlice[origin_of(name)]
        ) thin -> _mlir.Attribute.cType,
    ]()(ctx.c, name)


def dim_is_dynamic(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsDynamic", def (_mlir.Attribute.cType) thin -> Bool]()(a.c)


def dim_is_static(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsStatic", def (_mlir.Attribute.cType) thin -> Bool]()(a.c)


def dim_is_symbolic(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsSymbolic", def (_mlir.Attribute.cType) thin -> Bool]()(
        a.c
    )


def dim_is_algebraic(a: _mlir.Attribute) -> Bool:
    return cfunc["MAXG_dimIsAlgebraic", def (_mlir.Attribute.cType) thin -> Bool]()(
        a.c
    )


def dim_static_value(a: _mlir.Attribute) -> Int64:
    return cfunc["MAXG_dimStaticValue", def (_mlir.Attribute.cType) thin -> Int64]()(
        a.c
    )


def dim_symbolic_name(a: _mlir.Attribute) -> _mlir.Identifier:
    return cfunc[
        "MAXG_dimSymbolicName",
        def (_mlir.Attribute.cType) thin -> _mlir.Identifier.cType,
    ]()(a.c)


def list_type_new(ctx: _mlir.Context, eltype: _mlir.Type) -> _mlir.Type:
    return cfunc[
        "MAXG_listTypeNew",
        def (_mlir.Context.cType, _mlir.Type.cType) thin -> _mlir.Type.cType,
    ]()(ctx.c, eltype.c)


def list_type_element_type(t: _mlir.Type) -> _mlir.Type:
    return cfunc[
        "MAXG_listTypeElementType", def (_mlir.Type.cType) thin -> _mlir.Type.cType
    ]()(t.c)


def type_is_list(t: _mlir.Type) -> Bool:
    return cfunc["MAXG_typeIsList", def (_mlir.Type.cType) thin -> Bool]()(t.c)


def type_is_tensor(t: _mlir.Type) -> Bool:
    return cfunc["MAXG_typeIsTensor", def (_mlir.Type.cType) thin -> Bool]()(t.c)


def type_is_opaque(t: _mlir.Type) -> Bool:
    return cfunc["MAXG_typeIsOpaque", def (_mlir.Type.cType) thin -> Bool]()(t.c)


def opaque_type_new(ctx: _mlir.Context, name: String) -> _mlir.Type:
    return cfunc[
        "MAXG_opaqueTypeNew",
        def (
            _mlir.Context.cType, StringSlice[origin_of(name)]
        ) thin -> _mlir.Type.cType,
    ]()(ctx.c, name)


def opaque_type_name(t: _mlir.Type) -> StaticString:
    return cfunc[
        "MAXG_opaqueTypeName", def (_mlir.Type.cType) thin -> StaticString
    ]()(t.c)
