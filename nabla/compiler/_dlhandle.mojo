# Mojo 1.0 compatibility: the trivially-copyable `DLHandle` the vendored
# MAX wrappers rely on survives in std.ffi as `_DLHandle` (same surface:
# init-from-path, get_function, check_symbol, close). Function-pointer
# types must be spelled with `thin abi("C")`.

from std.ffi import _DLHandle

comptime DLHandle = _DLHandle
