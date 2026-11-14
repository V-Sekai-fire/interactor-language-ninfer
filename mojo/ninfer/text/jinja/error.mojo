"""The Jinja error type. C++ throws several exception classes; a typed `raises JinjaError`
carries the class in `kind`, so no code has to peek at the first byte of a message."""

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
