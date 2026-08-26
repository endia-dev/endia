# Mojo 1.0 compatibility: `builtin._location` (__call_location /
# _SourceLocation) is not part of the 1.0 stdlib. Graph code only used it to
# decorate error messages, so a dummy location is fine.


@fieldwise_init
struct _SourceLocation(Copyable, ImplicitlyCopyable, Movable, Writable):
    var line: Int
    var col: Int
    var file_name: StaticString

    def __init__(out self):
        self.line = 0
        self.col = 0
        self.file_name = ""

    def __str__(self) -> String:
        return String.write(self)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.file_name, ":", self.line, ":", self.col)


@always_inline("nodebug")
def __call_location() -> _SourceLocation:
    return _SourceLocation()
