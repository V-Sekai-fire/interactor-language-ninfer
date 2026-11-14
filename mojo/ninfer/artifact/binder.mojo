"""Logical parameters over stored objects; mirrors the Host-side half of `src/artifact/binder.cpp`.

Ported here: binding lookup and shape/format checks, `Use` lookup, component resources, and
owning Host values. Device, pinned and registered placement, transcode demands and
`MaterializationPlan` belong to the materializer port and are not part of this file yet.
"""

from std.collections import Dict, List, Optional

from ninfer.artifact.framing import read_u32_le
from ninfer.artifact.reader import Reader
from ninfer.artifact.schema import Binding, Use, checked_add, checked_mul
from ninfer.core.weight import (
    BF16,
    CONTIGUOUS,
    FP32,
    INT32,
    weight_element_count,
)

comptime RESIDENCY_DEVICE = 0
comptime RESIDENCY_HOST = 1
comptime RESIDENCY_VALUES = 2
comptime RESIDENCY_PINNED = 3
comptime RESIDENCY_REGISTERED = 4


struct ParameterReference(Copyable, Movable):
    var name: String
    var shape: List[UInt64]
    var binding: Binding
    var residency: Int

    def __init__(
        out self, name: String, shape: List[UInt64], binding: Binding, residency: Int
    ):
        self.name = name
        self.shape = shape.copy()
        self.binding = binding.copy()
        self.residency = residency


struct HostValues(Copyable, Movable):
    var format: Int
    var elements: UInt64
    var data: List[UInt8]

    def __init__(out self):
        self.format = FP32
        self.elements = 0
        self.data = List[UInt8]()

    def scalar_f32(self) raises -> Float32:
        if self.format != FP32 or self.elements != 1 or len(self.data) != 4:
            raise Error("auxiliary requires one represented FP32 value")
        var bits = read_u32_le(self.data, 0)
        return UnsafePointer(to=bits).bitcast[Float32]()[]

    def integers(self) raises -> List[Int32]:
        if self.format != INT32 or UInt64(len(self.data)) != checked_mul(
            self.elements, 4, "integer values"
        ):
            raise Error("semantic table requires INT32 values")
        var out = List[Int32]()
        for i in range(Int(self.elements)):
            var bits = read_u32_le(self.data, i * 4)
            out.append(UnsafePointer(to=bits).bitcast[Int32]()[])
        return out^


struct Binder(Movable):
    var host_index: Dict[Int, Int]  # object handle -> slot in host_data
    var host_data: List[List[UInt8]]
    var read_bytes: UInt64
    var owned_value_bytes: UInt64

    def __init__(out self):
        self.host_index = Dict[Int, Int]()
        self.host_data = List[List[UInt8]]()
        self.read_bytes = 0
        self.owned_value_bytes = 0

    def contains(self, reader: Reader, parameter: String) -> Bool:
        return parameter in reader.directory.bindings

    def parameter(
        mut self,
        mut reader: Reader,
        name: String,
        shape: List[UInt64],
        residency: Int = RESIDENCY_DEVICE,
        exact_format: Optional[Int] = None,
    ) raises -> ParameterReference:
        if name not in reader.directory.bindings:
            raise Error("missing logical parameter " + name)
        var binding = reader.directory.bindings[name].copy()
        return self.bind(reader, name, binding, shape, residency, exact_format)

    def bind(
        mut self,
        mut reader: Reader,
        name: String,
        binding: Binding,
        shape: List[UInt64],
        residency: Int,
        exact_format: Optional[Int],
    ) raises -> ParameterReference:
        var mismatch = weight_element_count(shape) != binding.elements
        if not mismatch and binding.whole_object:
            var stored = reader.directory.objects[binding.parts[0].object].tensor.shape.copy()
            if len(stored) != len(shape):
                mismatch = True
            else:
                for i in range(len(shape)):
                    if stored[i] != shape[i]:
                        mismatch = True
        if mismatch:
            raise Error(name + ": logical shape differs from Binding coverage")
        for part in binding.parts:
            var geometry = reader.geometry(part.object)
            if exact_format and geometry.format != exact_format.value():
                raise Error(
                    name + ": representation does not match its mathematical value type"
                )
            if residency == RESIDENCY_HOST or residency == RESIDENCY_REGISTERED:
                _ = self.host_object(reader, part.object)
        return ParameterReference(name, shape, binding, residency)

    def use(self, reader: Reader, parameter: String, input: String) raises -> Use:
        var key = parameter + "\0" + input
        if key not in reader.directory.uses:
            raise Error("missing Use " + parameter + "@" + input)
        return reader.directory.uses[key].copy()

    def host_object(mut self, mut reader: Reader, handle: Int) raises -> Int:
        """Read an object once and keep its bytes; returns the slot in `host_data`."""
        reader.validate_object(handle)
        var found = self.host_index.get(handle)
        if found:
            return found.value()
        var data = reader.read_object(handle)
        self.read_bytes = checked_add(self.read_bytes, UInt64(len(data)), "Host read bytes")
        self.host_data.append(data^)
        self.host_index[handle] = len(self.host_data) - 1
        return len(self.host_data) - 1

    def resource(
        mut self, mut reader: Reader, component: String, role: String
    ) raises -> Int:
        reader.directory.check_component(component)
        ref c = reader.directory.components[component]
        var handle = -1
        for i in range(len(c.resource_roles)):
            if c.resource_roles[i] == role:
                handle = c.resources[i]
        if handle < 0:
            raise Error(component + ": missing resource " + role)
        _ = self.host_object(reader, handle)
        return handle

    def values(
        mut self,
        mut reader: Reader,
        binding: Binding,
        format: Optional[Int] = None,
    ) raises -> HostValues:
        var out = HostValues()
        out.elements = binding.elements
        if len(binding.parts) == 0:
            raise Error("value Binding is empty")
        out.format = (
            format.value() if format else reader.geometry(binding.parts[0].object).format
        )
        var word_bytes = UInt64(0)
        if out.format == BF16:
            word_bytes = 2
        elif out.format == FP32 or out.format == INT32:
            word_bytes = 4
        else:
            raise Error("owning Host values require a direct numeric format")
        var total = checked_mul(out.elements, word_bytes, "Host value bytes")
        for part in binding.parts:
            var geometry = reader.geometry(part.object)
            if (
                geometry.layout != CONTIGUOUS
                or geometry.format != out.format
                or part.begin >= part.end
                or part.end > geometry.elements
            ):
                raise Error("Host value Binding has an incompatible representation or range")
            var bytes = checked_mul(part.end - part.begin, word_bytes, "value range")
            if bytes > total - UInt64(len(out.data)):
                raise Error("Host value coverage exceeds declared size")
            var source = checked_mul(part.begin, word_bytes, "value offset")
            var slot = self.host_index.get(part.object)
            if slot:
                ref cached = self.host_data[slot.value()]
                for i in range(Int(bytes)):
                    out.data.append(cached[Int(source) + i])
            else:
                var offset = reader.directory.objects[part.object].offset
                var chunk = reader.read_range(
                    checked_add(offset, source, "value file offset"), bytes
                )
                self.read_bytes = checked_add(self.read_bytes, bytes, "Host read bytes")
                out.data.extend(chunk^)
        if UInt64(len(out.data)) != total:
            raise Error("Host value coverage is incomplete")
        self.owned_value_bytes = checked_add(
            self.owned_value_bytes, UInt64(len(out.data)), "owning value bytes"
        )
        return out^
