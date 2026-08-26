# ===----------------------------------------------------------------------=== #
# Mojo 1.0 compatibility stub for MAX 25.3's closed-source `_mlir` package.
#
# The real package wrapped the MLIR C API shipped inside the MAX graph
# library. That library (and the Mojo graph API it powered) does not exist
# in the Mojo 1.0 toolchain, so this stub only provides the type surface the
# vendored `nabla.compiler.graph` code needs to COMPILE. Every handle is an
# opaque null; any attempt to actually build or run a graph aborts earlier,
# when the MAX driver/engine dylibs fail to load.
# ===----------------------------------------------------------------------=== #

from .ir import (
    Attribute,
    Block,
    Context,
    Identifier,
    Location,
    Module,
    NamedAttribute,
    Operation,
    Region,
    Type,
    Value,
)
from . import builtin_attributes as builtin_attributes
from . import builtin_types as builtin_types
from . import ir as ir
