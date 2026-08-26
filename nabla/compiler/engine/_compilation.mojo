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

from std.collections import List, Optional
from std.collections.string import StringSlice
from std.pathlib import Path
from std.ffi import external_call
from std.ffi import c_char
from nabla.compiler._dlhandle import DLHandle

from nabla.compiler._utils import mut_ptr, null_ptr, OwningVector, call_dylib_func, exchange
from nabla.compiler.tensor import TensorSpec
from std.memory import OwnedPointer, UnsafePointer

from ._model_specs import InputTensorNames, OutputTensorNames
from ._status import Status
from ._tensor_spec_impl import CTensorSpec
from .session import InferenceSession


@fieldwise_init
struct FrameworkFormat(TrivialRegisterPassable, ImplicitlyCopyable):
    """Enum-like struct indicating the model framework."""

    comptime MAXGraph = FrameworkFormat(0)
    comptime TorchscriptModule = FrameworkFormat(1)
    comptime TorchscriptFunction = FrameworkFormat(2)
    comptime TorchMLIR = FrameworkFormat(3)

    var value: UInt8


@fieldwise_init
struct ModelSource(TrivialRegisterPassable, ImplicitlyCopyable, Copyable, Movable):
    """Model source representation that is ABI compatible with the C API's `M_ModelSource`.
    """

    var source: UnsafePointer[NoneType, MutUntrackedOrigin]
    var format: FrameworkFormat


@fieldwise_init
struct CCompileConfig(TrivialRegisterPassable, ImplicitlyCopyable):
    """Mojo representation of Engine's CompileConfig pointer.
    This doesn't free the memory on destruction. For memory managed
    option see CompileConfig.
    """

    var ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    comptime FreeCompileConfigFnName = "M_freeCompileConfig"
    comptime SetModelSourceFnName = "M_setModelSourceInternal"
    comptime SetPipelineNameFnName = "M_setPipelineName"
    comptime SetModelPathFnName = "M_setModelPath"
    comptime ReplaceOpsFnName = "M_useKernelsFrom"
    comptime SetTorchInputSpecsFnName = "M_setTorchInputSpecs"

    def set_model_source(self, model_source: ModelSource, lib: DLHandle):
        call_dylib_func(lib, Self.SetModelSourceFnName, self, model_source)

    def set_pipeline_name(self, name: String, lib: DLHandle):
        call_dylib_func(
            lib, Self.SetPipelineNameFnName, self, name.unsafe_ptr()
        )

    def set_model_path(self, var path: String, lib: DLHandle):
        """Sets the path of model to compile."""
        call_dylib_func(
            lib, Self.SetModelPathFnName, self, path.as_c_string_slice().unsafe_ptr()
        )

    def replace_ops(self, var path: String, lib: DLHandle) raises:
        var status = Status(lib)
        call_dylib_func(
            lib, Self.ReplaceOpsFnName, self, path.as_c_string_slice().unsafe_ptr(), status.ptr
        )
        if status:
            raise Error(String(status))

    def set_torch_input_specs(
        self,
        torch_lib: DLHandle,
        specs_ptr: List[CTorchInputSpec],
    ):
        call_dylib_func(
            torch_lib,
            Self.SetTorchInputSpecsFnName,
            self,
            specs_ptr.unsafe_ptr(),
            len(specs_ptr),
        )

    def free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeCompileConfigFnName, self)


@fieldwise_init
struct CTorchInputSpec(TrivialRegisterPassable, ImplicitlyCopyable, Copyable, Movable):
    """C API ABI compatible M_TorchInputSpec."""

    comptime ptr_type = UnsafePointer[NoneType, MutUntrackedOrigin]
    var ptr: Self.ptr_type

    comptime FreeTorchInputSpecFnName = "M_freeTorchInputSpec"

    def free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeTorchInputSpecFnName, self)


struct TorchInputSpec(Movable):
    comptime shape_type = List[Int64]
    var shape: Self.shape_type
    var dtype: DType
    var ptr: CTorchInputSpec
    var torch_lib: DLHandle

    comptime NewTorchInputSpecFnName = "M_newTorchInputSpec"

    def __init__(out self, spec: TensorSpec, lib: DLHandle) raises:
        var shape = Self.shape_type()
        shape.reserve(spec.rank())
        for i in range(spec.rank()):
            shape.append(Int64(spec[i]))
        self.shape = shape.copy()
        self.dtype = spec.dtype()
        var status = Status(lib)
        var ptr = call_dylib_func[CTorchInputSpec](
            lib,
            Self.NewTorchInputSpecFnName,
            self.shape.unsafe_ptr(),
            null_ptr[NoneType](),
            len(self.shape),
            self.dtype,
            null_ptr[NoneType](),
            status.ptr,
        )
        if status:
            raise Error(String(status))
        self.ptr = ptr
        self.torch_lib = lib

    def __init__(
        out self,
        shape: List[ShapeElement],
        dtype: DType,
        lib: DLHandle,
        engine_lib: DLHandle,
    ) raises:
        var converted_shape = Self.shape_type()
        var converted_dim_names = List[UnsafePointer[c_char, ImmUntrackedOrigin]]()
        converted_shape.reserve(len(shape))

        var strs = List[String]()
        for dim in shape:
            if dim.is_static():
                converted_shape.append(Int64(dim.static_value()))
                converted_dim_names.append(null_ptr[c_char]().as_imm())
            else:
                converted_shape.append(
                    Int64(CTensorSpec.get_dynamic_dimension_value(engine_lib))
                )
                var str = dim._name
                strs.append(str^)  # Keep the string alive.
                var c_str = strs[len(strs) - 1].as_c_string_slice().unsafe_ptr()
                converted_dim_names.append(c_str.unsafe_origin_cast[ImmUntrackedOrigin]())

        self.shape = converted_shape^
        self.dtype = dtype
        var status = Status(lib)
        var ptr = call_dylib_func[CTorchInputSpec](
            lib,
            Self.NewTorchInputSpecFnName,
            self.shape.unsafe_ptr(),
            converted_dim_names.unsafe_ptr(),
            len(self.shape),
            self.dtype,
            null_ptr[NoneType](),
            status.ptr,
        )

        if status:
            raise Error(String(status))
        self.ptr = ptr
        self.torch_lib = lib

    def __init__(
        out self,
        shape: NoneType,
        dtype: DType,
        lib: DLHandle,
        engine_lib: DLHandle,
    ) raises:
        self.shape = Self.shape_type()
        self.dtype = dtype
        var status = Status(lib)
        var ptr = call_dylib_func[CTorchInputSpec](
            lib,
            Self.NewTorchInputSpecFnName,
            null_ptr[NoneType](),
            null_ptr[NoneType](),
            CTensorSpec.get_dynamic_rank_value(engine_lib),
            self.dtype,
            status.ptr,
        )
        if status:
            raise Error(String(status))
        self.ptr = ptr
        self.torch_lib = lib

    def __init__(out self, *, deinit existing: Self):
        self.shape = existing.shape^
        self.dtype = existing.dtype
        self.ptr = existing.ptr.copy()
        self.torch_lib = existing.torch_lib

    def __deinit__(deinit self):
        self.ptr.free(self.torch_lib)


struct CompileConfig:
    """Memory managed version of Engine's Compile Config."""

    var _ptr: OwnedPointer[CCompileConfig]
    var lib: DLHandle
    var torch_lib: Optional[DLHandle]
    var input_specs: OwningVector[TorchInputSpec]

    comptime NewCompileConfigFnName = "M_newCompileConfig"

    @implicit
    def __init__(out self, lib: DLHandle):
        self._ptr = OwnedPointer(
            call_dylib_func[CCompileConfig](lib, Self.NewCompileConfigFnName)
        )
        self.lib = lib
        self.input_specs = OwningVector[TorchInputSpec]()
        self.torch_lib = Self._get_torch_lib()

    @staticmethod
    def _get_torch_lib() -> Optional[DLHandle]:
        # Since we only need to open this library for this case we
        # can lazy load it here.
        comptime key = StaticString(".torch_ext_lib")

        # TODO: Move KGEN_CompilerRT_getMAXConfigValue to a helper somewhere.
        var torch_ext_lib_path_str_ptr = external_call[
            "KGEN_CompilerRT_getMAXConfigValue", UnsafePointer[UInt8, MutUntrackedOrigin]
        ](key.unsafe_ptr(), key.byte_length())

        if Int(torch_ext_lib_path_str_ptr) == 0:
            return None

        var torch_ext_lib_path = String(
            unsafe_from_utf8_ptr=torch_ext_lib_path_str_ptr
        )
        torch_ext_lib_path_str_ptr.free()

        # If the path does not exist, swallow the error and return None.
        try:
            return DLHandle(torch_ext_lib_path)
        except:
            return None

    def __init__(out self, *, deinit existing: Self):
        self._ptr = existing._ptr^
        self.lib = existing.lib
        self.input_specs = existing.input_specs^
        self.torch_lib = existing.torch_lib

    def set_model_source(self, model_source: ModelSource):
        self._ptr[].set_model_source(model_source, self.lib)

    def set_pipeline_name(self, name: String):
        self._ptr[].set_pipeline_name(name, self.lib)

    def set_model_path(self, path: String):
        """Sets the path of model to compile."""
        self._ptr[].set_model_path(path, self.lib)

    def set_replace_ops_path(self, path: String) raises:
        """Replace Modular kernels with user-defined kernels."""
        self._ptr[].replace_ops(path, self.lib)

    def set_torch_input_specs(self) raises:
        if len(self.input_specs) == 0:
            return

        if not self.torch_lib:
            raise "cannot find torch extension libraries"

        var inner_spec = List[CTorchInputSpec]()
        for i in range(len(self.input_specs)):
            var spec_ptr = self.input_specs.get(i)
            inner_spec.append(spec_ptr[].ptr)
        self._ptr[].set_torch_input_specs(self.torch_lib.value(), inner_spec)

    def add_input_spec(mut self, spec: TensorSpec) raises:
        self.input_specs.emplace_back(
            TorchInputSpec(spec, self.torch_lib.value())
        )

    def add_input_spec(
        mut self,
        shape_or: Optional[List[ShapeElement]],
        dtype: DType,
    ) raises:
        if not shape_or:
            self.input_specs.emplace_back(
                TorchInputSpec(None, dtype, self.torch_lib.value(), self.lib)
            )
            return
        self.input_specs.emplace_back(
            TorchInputSpec(
                shape_or.value(),
                dtype,
                self.torch_lib.value(),
                self.lib,
            )
        )

    def borrow_ptr(self) -> UnsafePointer[CCompileConfig, MutUntrackedOrigin]:
        return mut_ptr(self._ptr.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin]())

    def __deinit__(deinit self):
        if self.torch_lib:
            var torch = self.torch_lib.value()
            torch.close()

        self._ptr[].free(self.lib)


struct CCompiledModel(TrivialRegisterPassable, ImplicitlyCopyable):
    """Mojo representation of Engine's AsyncCompiledModel pointer.
    Useful for C inter-op.
    """

    var ptr: UnsafePointer[NoneType, MutUntrackedOrigin]

    comptime FreeCompiledModelFnName = "M_freeCompiledModel"
    comptime GetModelInputSpecByNameFnName = "M_getModelInputSpecByName"
    comptime GetModelOutputSpecByNameFnName = "M_getModelOutputSpecByName"
    comptime GetNumInputsFnName = "M_getNumModelInputs"
    comptime GetNumOutputsFnName = "M_getNumModelOutputs"
    comptime ExportModelFnName = "M_exportCompiledModel"

    @implicit
    def __init__(out self, ptr: UnsafePointer[NoneType, MutUntrackedOrigin]):
        self.ptr = ptr

    def num_model_inputs(self, lib: DLHandle) raises -> Int:
        """Gets the number of inputs of the model."""

        var status = Status(lib)
        var num_inputs = call_dylib_func[Int](
            lib, Self.GetNumInputsFnName, self, status.ptr
        )
        if status:
            raise Error(String(status))
        return num_inputs

    def num_model_outputs(self, lib: DLHandle) raises -> Int:
        """Gets the number of outputs of the model."""

        var status = Status(lib)
        var num_outputs = call_dylib_func[Int](
            lib, Self.GetNumOutputsFnName, self, status.ptr
        )
        if status:
            raise Error(String(status))
        return num_outputs

    def get_model_input_spec_by_name(
        self,
        var tensor_name: String,
        lib: DLHandle,
        var session: InferenceSession,
    ) raises -> EngineTensorSpec:
        """Gets the input spec of the model by name."""
        var status = Status(lib)
        var input_spec = call_dylib_func[CTensorSpec](
            lib,
            Self.GetModelInputSpecByNameFnName,
            self,
            tensor_name.as_c_string_slice().unsafe_ptr(),
            status.ptr,
        )
        if status:
            raise Error(String(status))
        return EngineTensorSpec(input_spec, lib, session)

    def get_model_output_spec_by_name(
        self,
        var tensor_name: String,
        lib: DLHandle,
        var session: InferenceSession,
    ) raises -> EngineTensorSpec:
        """Gets the output spec of the model by name."""
        var status = Status(lib)
        var output_spec = call_dylib_func[CTensorSpec](
            lib,
            Self.GetModelOutputSpecByNameFnName,
            self,
            tensor_name.as_c_string_slice().unsafe_ptr(),
            status.ptr,
        )
        if status:
            raise Error(String(status))
        return EngineTensorSpec(output_spec, lib, session)

    def export_compiled_model(self, lib: DLHandle, var path: String) raises:
        var status = Status(lib)
        call_dylib_func(
            lib,
            Self.ExportModelFnName,
            self,
            path.as_c_string_slice().unsafe_ptr(),
            status.ptr,
        )
        if status:
            raise Error(String(status))

    def free(self, lib: DLHandle):
        call_dylib_func(lib, Self.FreeCompiledModelFnName, self)


@fieldwise_init
struct CompiledModel(Copyable, Movable):
    """Memory managed CompiledModel pointer."""

    var ptr: CCompiledModel
    var lib: DLHandle
    var session: InferenceSession

    comptime CompileModelFnName = "M_compileModelSync"

    def __init__(out self, *, deinit existing: Self):
        self.ptr = exchange[CCompiledModel](
            existing.ptr, null_ptr[NoneType]()
        )
        self.lib = existing.lib
        self.session = existing.session^

    def num_model_inputs(self) raises -> Int:
        """Gets the number of inputs of the model."""

        return self.ptr.num_model_inputs(self.lib)

    def get_model_input_names(self) raises -> List[String]:
        """Gets the names of model inputs."""

        var names = InputTensorNames(
            self.ptr, self.num_model_inputs(), self.lib
        )
        var name_vec = List[String]()
        name_vec.reserve(len(names))
        for i in range(len(names)):
            name_vec.append(names[i])
        return name_vec

    def num_model_outputs(self) raises -> Int:
        """Gets the number of outputs of the model."""

        return self.ptr.num_model_outputs(self.lib)

    def get_model_output_names(self) raises -> List[String]:
        """Gets the names of model outputs."""

        var names = OutputTensorNames(
            self.ptr, self.num_model_outputs(), self.lib
        )
        var name_vec = List[String]()
        name_vec.reserve(len(names))
        for i in range(len(names)):
            name_vec.append(names[i])
        return name_vec

    def get_model_input_metadata(self) raises -> List[EngineTensorSpec]:
        """Get the metadata for inputs of the model."""
        var input_metadata = List[EngineTensorSpec]()
        var input_tensor_names = self.get_model_input_names()
        input_metadata.reserve(len(input_tensor_names))

        for input_tensor_name in input_tensor_names:
            var input_spec = self.ptr.get_model_input_spec_by_name(
                input_tensor_name, self.lib, self.session
            )
            input_metadata.append(input_spec^)
        return input_metadata

    def get_model_output_metadata(self) raises -> List[EngineTensorSpec]:
        """Get the metadata for outputs of the model."""
        var output_metadata = List[EngineTensorSpec]()
        var output_tensor_names = self.get_model_output_names()
        output_metadata.reserve(len(output_tensor_names))

        for output_tensor_name in output_tensor_names:
            var output_spec = self.ptr.get_model_output_spec_by_name(
                output_tensor_name, self.lib, self.session
            )
            output_metadata.append(output_spec^)
        return output_metadata

    def borrow_ptr(self) -> CCompiledModel:
        return self.ptr

    def export_compiled_model(self, lib: DLHandle, path: String) raises:
        self.ptr.export_compiled_model(lib, path)

    def __deinit__(deinit self):
        self.ptr.free(self.lib)
        _ = self.session^
