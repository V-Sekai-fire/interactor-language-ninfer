"""The Jinja error type. C++ throws several exception classes; a typed `raises JinjaError`
carries the class in `kind`, so no code has to peek at the first byte of a message."""

from ninfer.text.unicode import CodepointSpan, append_utf8, utf8_codepoints

comptime K_ERROR = 0  # an ordinary `std::exception`; the nearest `execute` adds source context
comptime K_BREAK = 1  # loop control, never shown to a caller
comptime K_CONTINUE = 2  # loop control, never shown to a caller
comptime K_RETHROWN = 3  # a message that already carries its source context
comptime K_NOTIMPL = 4  # a NotImplemented error that `execute` must not decorate


@fieldwise_init
struct JinjaError(Copyable, Writable):
    var kind: Int
    var message: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.message)


def jinja_error(message: String) -> JinjaError:
    return JinjaError(K_ERROR, message)


# The unicode layer and the standard library raise `Error`. These wrappers are the one place
# where that becomes a `JinjaError`, so the message text is unchanged.
def codepoints(text: List[UInt8], context: String) raises JinjaError -> List[CodepointSpan]:
    try:
        return utf8_codepoints(text, context)
    except e:
        raise jinja_error(String(e))


def push_utf8(mut out: List[UInt8], codepoint: Int) raises JinjaError:
    try:
        append_utf8(out, codepoint)
    except e:
        raise jinja_error(String(e))


def parse_float(text: String) raises JinjaError -> Float64:
    try:
        return atof(text)
    except e:
        raise jinja_error(String(e))


def parse_int(text: String) raises JinjaError -> Int:
    try:
        return atol(text)
    except e:
        raise jinja_error(String(e))
