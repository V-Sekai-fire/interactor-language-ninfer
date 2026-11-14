"""Artifact directory model and validation; mirrors `src/artifact/schema.{h,cpp}`.

Every validation rule and error message of the C++ parser is kept so a malformed directory is
rejected by both readers for the same reason. Component configs, metadata and provenance stay
as nodes of `Directory.doc`; architecture parsers read them from there.
"""

from std.collections import Dict, List, Optional

from ninfer.core.weight import weight_element_count
from ninfer.artifact.json import (
    Json,
    NULL,
    OBJECT,
    ARRAY,
    STRING,
    parse_json,
)


comptime MAX_RANK = 16
comptime MAX_U64 = UInt64.MAX


def checked_add(a: UInt64, b: UInt64, label: String) raises -> UInt64:
    if b > MAX_U64 - a:
        raise Error(label + " overflows u64")
    return a + b


def checked_mul(a: UInt64, b: UInt64, label: String) raises -> UInt64:
    if a != 0 and b > MAX_U64 // a:
        raise Error(label + " overflows u64")
    return a * b


def align_up(value: UInt64, alignment: UInt64, label: String) raises -> UInt64:
    if alignment == 0:
        raise Error(label + ": zero alignment")
    return checked_add(value, alignment - 1, label) // alignment * alignment


def require_members(
    doc: Json,
    value: Int,
    required: List[String],
    optional: List[String],
    label: String,
) raises:
    if not doc.is_object(value):
        raise Error(label + " must be an object")
    for key in required:
        if not doc.contains(value, key):
            raise Error(label + ": missing " + key)
    ref node = doc.nodes[value]
    for i in range(len(node.keys)):
        var k = node.keys[i]
        var known = False
        for r in required:
            if r == k:
                known = True
        for o in optional:
            if o == k:
                known = True
        if not known:
            raise Error(label + ": unknown member " + k)


def require_id(doc: Json, value: Int, label: String) raises -> String:
    if not doc.is_string(value):
        raise Error(label + " must be a string")
    var result = doc.string(value)
    if result.byte_length() == 0 or result.find("\0") >= 0:
        raise Error(label + " must be nonempty without NUL")
    return result


def require_u64(
    doc: Json, value: Int, label: String, positive: Bool = False
) raises -> UInt64:
    if not doc.is_integer(value) or (
        not doc.is_unsigned(value) and doc.nodes[value].u != 0
    ):
        raise Error(label + " must be a nonnegative integer")
    var result = doc.nodes[value].u
    if positive and result == 0:
        raise Error(label + " must be positive")
    return result


struct TensorObject(Copyable, Movable):
    var shape: List[UInt64]
    var format: String
    var layout: String
    var divisors: UInt64  # source matrices of a stacked plane; 1 otherwise

    def __init__(out self):
        self.shape = List[UInt64]()
        self.format = String()
        self.layout = String()
        self.divisors = 1


struct Object(Copyable, Movable):
    """Tensor or resource. `is_tensor` selects which fields are meaningful."""

    var id: String
    var is_tensor: Bool
    var offset: UInt64
    var bytes: UInt64
    var tensor: TensorObject
    var encoding: String  # resource only

    def __init__(out self):
        self.id = String()
        self.is_tensor = False
        self.offset = 0
        self.bytes = 0
        self.tensor = TensorObject()
        self.encoding = String()


struct Part(Copyable, Movable):
    var object: Int
    var begin: UInt64
    var end: UInt64

    def __init__(out self, object: Int, begin: UInt64, end: UInt64):
        self.object = object
        self.begin = begin
        self.end = end


struct Binding(Copyable, Movable):
    var whole_object: Bool
    var parts: List[Part]
    var elements: UInt64

    def __init__(out self):
        self.whole_object = False
        self.parts = List[Part]()
        self.elements = 0


comptime ACT_NONE = -1
comptime ACT_A16_ONLY = 0
comptime ACT_ALLOW_A8 = 1
comptime ACT_ALLOW_A4 = 2


struct Use(Copyable, Movable):
    var parameter: String
    var input: String
    var activation_policy: Int  # ACT_*
    var auxiliary_roles: List[String]
    var auxiliaries: List[Binding]

    def __init__(out self):
        self.parameter = String()
        self.input = String()
        self.activation_policy = ACT_NONE
        self.auxiliary_roles = List[String]()
        self.auxiliaries = List[Binding]()


struct Proposal(Copyable, Movable):
    var indexed: Bool
    var rows: UInt64

    def __init__(out self, indexed: Bool = False, rows: UInt64 = 0):
        self.indexed = indexed
        self.rows = rows


struct Component(Copyable, Movable):
    var config: Int  # node in Directory.doc
    var target: Optional[String]
    var resource_roles: List[String]
    var resources: List[Int]  # object handles, parallel to resource_roles
    var proposal: Optional[Proposal]

    def __init__(out self):
        self.config = -1
        self.target = None
        self.resource_roles = List[String]()
        self.resources = List[Int]()
        self.proposal = None


struct FileRecord(Copyable, Movable):
    var path: Optional[String]  # None for the entry file
    var payload_bytes: UInt64
    var logical_begin: UInt64

    def __init__(out self):
        self.path = None
        self.payload_bytes = 0
        self.logical_begin = 0


struct Directory(Movable):
    var doc: Json
    var component_names: List[String]  # sorted, like std::map
    var components: Dict[String, Component]
    var objects: List[Object]
    var object_index: Dict[String, Int]
    var binding_names: List[String]  # sorted
    var bindings: Dict[String, Binding]
    var uses: Dict[String, Use]  # key: parameter + "\0" + input
    var files: List[FileRecord]
    var payload_bytes: UInt64
    var metadata: Int
    var provenance: Int

    def __init__(out self):
        self.doc = Json()
        self.component_names = List[String]()
        self.components = Dict[String, Component]()
        self.objects = List[Object]()
        self.object_index = Dict[String, Int]()
        self.binding_names = List[String]()
        self.bindings = Dict[String, Binding]()
        self.uses = Dict[String, Use]()
        self.files = List[FileRecord]()
        self.payload_bytes = 0
        self.metadata = -1
        self.provenance = -1

    def check_handle(self, handle: Int) raises:
        if handle < 0 or handle >= len(self.objects):
            raise Error("invalid object handle")

    def check_tensor(self, handle: Int) raises:
        self.check_handle(handle)
        if not self.objects[handle].is_tensor:
            raise Error("object does not reference a tensor")

    def check_component(self, name: String) raises:
        if name not in self.components:
            raise Error("missing component " + name)


def _sorted(var names: List[String]) -> List[String]:
    # Insertion sort on byte order; directories hold hundreds to thousands of names, and the
    # names arrive nearly sorted from the writer.
    for i in range(1, len(names)):
        var j = i
        while j > 0 and names[j] < names[j - 1]:
            var tmp = names[j]
            names[j] = names[j - 1]
            names[j - 1] = tmp^
            j -= 1
    return names^


def _parse_shape(doc: Json, value: Int, label: String) raises -> List[UInt64]:
    if not doc.is_array(value) or doc.size(value) > MAX_RANK:
        raise Error(label + ": shape must have rank 0 through 16")
    var out = List[UInt64]()
    var elements = UInt64(1)
    for i in range(doc.size(value)):
        var dim = require_u64(doc, doc.at(value, i), label, True)
        elements = checked_mul(elements, dim, label)
        out.append(dim)
    return out^


def _reference(
    d: Directory, value: Int, label: String, tensor: Bool
) raises -> Int:
    var name = require_id(d.doc, value, label)
    var found = d.object_index.get(name)
    if not found:
        raise Error(label + ": missing object " + name)
    var handle = found.value()
    if d.objects[handle].is_tensor != tensor:
        raise Error(label + ": wrong object kind for " + name)
    return handle


def _parse_binding(d: Directory, value: Int, label: String) raises -> Binding:
    if not d.doc.is_object(value):
        raise Error(label + ": expected Binding object")
    var out = Binding()
    if d.doc.size(value) == 1 and d.doc.contains(value, "object"):
        var handle = _reference(d, d.doc.find(value, "object"), label, True)
        out.whole_object = True
        out.elements = weight_element_count(d.objects[handle].tensor.shape)
        out.parts.append(Part(handle, 0, out.elements))
        return out^
    require_members(d.doc, value, ["parts"], [], label)
    var parts = d.doc.find(value, "parts")
    if not d.doc.is_array(parts) or d.doc.size(parts) == 0:
        raise Error(label + ": parts must be a nonempty array")
    for i in range(d.doc.size(parts)):
        var part = d.doc.at(parts, i)
        require_members(d.doc, part, ["object", "range"], [], label)
        var handle = _reference(d, d.doc.find(part, "object"), label, True)
        var bounds = d.doc.find(part, "range")
        if not d.doc.is_array(bounds) or d.doc.size(bounds) != 2:
            raise Error(label + ": range must contain two integers")
        var begin = require_u64(d.doc, d.doc.at(bounds, 0), label)
        var end = require_u64(d.doc, d.doc.at(bounds, 1), label)
        if begin >= end or end > weight_element_count(d.objects[handle].tensor.shape):
            raise Error(label + ": Part exceeds source logical elements")
        out.elements = checked_add(out.elements, end - begin, label)
        out.parts.append(Part(handle, begin, end))
    return out^


def _parse_files(mut d: Directory, files: Int, entry_name: String) raises:
    if not d.doc.is_array(files) or d.doc.size(files) == 0:
        raise Error("files must be a nonempty array")
    var names = Dict[String, Bool]()
    names[entry_name] = True
    for i in range(d.doc.size(files)):
        var value = d.doc.at(files, i)
        require_members(d.doc, value, ["path", "payload_bytes"], [], "file")
        var file = FileRecord()
        var path_node = d.doc.find(value, "path")
        if len(d.files) == 0:
            if not d.doc.is_null(path_node):
                raise Error("files[0].path must be null")
        else:
            var path = require_id(d.doc, path_node, "file path")
            if (
                path == "."
                or path == ".."
                or path.find("/") >= 0
                or path.find("\\") >= 0
                or path in names
            ):
                raise Error("invalid or duplicate sibling filename: " + path)
            names[path] = True
            file.path = path^
        file.payload_bytes = require_u64(
            d.doc, d.doc.find(value, "payload_bytes"), "file payload bytes", True
        )
        file.logical_begin = d.payload_bytes
        d.payload_bytes = checked_add(
            d.payload_bytes, file.payload_bytes, "total payload bytes"
        )
        d.files.append(file^)


def _parse_objects(mut d: Directory, objects: Int) raises:
    if not d.doc.is_array(objects) or d.doc.size(objects) == 0:
        raise Error("objects must be a nonempty array")
    var previous_end = UInt64(0)
    for i in range(d.doc.size(objects)):
        var value = d.doc.at(objects, i)
        if not d.doc.is_object(value) or not d.doc.contains(value, "kind"):
            raise Error("invalid object descriptor")
        var kind = require_id(d.doc, d.doc.find(value, "kind"), "object kind")
        if kind == "tensor":
            require_members(
                d.doc,
                value,
                ["id", "kind", "shape", "format", "layout", "offset", "bytes"],
                ["divisors"],
                "tensor",
            )
        elif kind == "resource":
            require_members(
                d.doc,
                value,
                ["id", "kind", "encoding", "offset", "bytes"],
                [],
                "resource",
            )
        else:
            raise Error("unknown object kind " + kind)
        var id = require_id(d.doc, d.doc.find(value, "id"), "object id")
        var offset = require_u64(d.doc, d.doc.find(value, "offset"), id)
        var bytes = require_u64(d.doc, d.doc.find(value, "bytes"), id, True)
        var end = checked_add(offset, bytes, id)
        if offset < previous_end or end > d.payload_bytes:
            raise Error(id + ": object overlaps, is out of order or exceeds payload")
        if id in d.object_index:
            raise Error("duplicate object id " + id)
        d.object_index[id] = len(d.objects)
        previous_end = end
        var obj = Object()
        obj.id = id
        obj.offset = offset
        obj.bytes = bytes
        if kind == "tensor":
            obj.is_tensor = True
            var divisors = UInt64(1)
            if d.doc.contains(value, "divisors"):
                divisors = require_u64(d.doc, d.doc.find(value, "divisors"), id, True)
            obj.tensor.shape = _parse_shape(d.doc, d.doc.find(value, "shape"), id)
            obj.tensor.format = require_id(d.doc, d.doc.find(value, "format"), id)
            obj.tensor.layout = require_id(d.doc, d.doc.find(value, "layout"), id)
            obj.tensor.divisors = divisors
        else:
            obj.encoding = require_id(d.doc, d.doc.find(value, "encoding"), id)
        d.objects.append(obj^)


def _parse_components(mut d: Directory, values: Int) raises:
    if not d.doc.is_object(values) or d.doc.size(values) == 0:
        raise Error("components must be a nonempty object")
    for i in range(d.doc.size(values)):
        var name = d.doc.key(values, i)
        var value = d.doc.at(values, i)
        if name.byte_length() == 0 or name.find("\0") >= 0:
            raise Error("component id must be nonempty without NUL")
        require_members(
            d.doc, value, ["config"], ["target", "resources", "proposal"], name
        )
        var component = Component()
        component.config = d.doc.find(value, "config")
        if not d.doc.is_object(component.config):
            raise Error(name + ": config must be an object")
        if d.doc.contains(value, "target"):
            var target = require_id(d.doc, d.doc.find(value, "target"), name + " target")
            if not d.doc.contains(values, target):
                raise Error(name + ": missing target " + target)
            component.target = target^
        if d.doc.contains(value, "resources"):
            var resources = d.doc.find(value, "resources")
            if not d.doc.is_object(resources):
                raise Error(name + ": resources must be an object")
            for r in range(d.doc.size(resources)):
                var role = d.doc.key(resources, r)
                if role.byte_length() == 0 or role.find("\0") >= 0:
                    raise Error("resource role must be nonempty without NUL")
                var handle = _reference(
                    d, d.doc.at(resources, r), name + "/" + role, False
                )
                component.resource_roles.append(role^)
                component.resources.append(handle)
        if d.doc.contains(value, "proposal"):
            if name != "text":
                raise Error("only text may declare proposal")
            var p = d.doc.find(value, "proposal")
            require_members(d.doc, p, ["domain"], ["rows"], "proposal")
            var domain = require_id(d.doc, d.doc.find(p, "domain"), "proposal domain")
            if domain == "full" and not d.doc.contains(p, "rows"):
                component.proposal = Proposal()
            elif domain == "indexed" and d.doc.contains(p, "rows"):
                component.proposal = Proposal(
                    True, require_u64(d.doc, d.doc.find(p, "rows"), "proposal rows", True)
                )
            else:
                raise Error(
                    "proposal must be full without rows or indexed with positive rows"
                )
        d.component_names.append(name)
        d.components[name] = component^
    d.component_names = _sorted(d.component_names.copy())


def _parse_uses(mut d: Directory, values: Int) raises:
    if not d.doc.is_array(values):
        raise Error("uses must be an array")
    for i in range(d.doc.size(values)):
        var value = d.doc.at(values, i)
        require_members(
            d.doc,
            value,
            ["parameter", "input"],
            ["activation_policy", "auxiliaries"],
            "Use",
        )
        var use = Use()
        use.parameter = require_id(d.doc, d.doc.find(value, "parameter"), "Use parameter")
        use.input = require_id(d.doc, d.doc.find(value, "input"), "Use input")
        var label = use.parameter + "@" + use.input
        if use.parameter not in d.bindings:
            raise Error(label + ": missing parameter")
        if d.doc.contains(value, "activation_policy"):
            var policy = require_id(d.doc, d.doc.find(value, "activation_policy"), label)
            if policy == "A16Only":
                use.activation_policy = ACT_A16_ONLY
            elif policy == "AllowA8":
                use.activation_policy = ACT_ALLOW_A8
            elif policy == "AllowA4":
                use.activation_policy = ACT_ALLOW_A4
            else:
                raise Error(label + ": unknown activation policy " + policy)
        if d.doc.contains(value, "auxiliaries"):
            var aux = d.doc.find(value, "auxiliaries")
            if not d.doc.is_object(aux):
                raise Error(label + ": auxiliaries must be an object")
            var roles = List[String]()
            for r in range(d.doc.size(aux)):
                var role = d.doc.key(aux, r)
                if role.byte_length() == 0 or role.find("\0") >= 0:
                    raise Error("auxiliary role must be nonempty without NUL")
                roles.append(role)
            # std::map order: sorted by role.
            roles = _sorted(roles^)
            for role in roles:
                var node = d.doc.find(aux, role)
                use.auxiliary_roles.append(role)
                use.auxiliaries.append(_parse_binding(d, node, label + "/" + role))
        var key = use.parameter + "\0" + use.input
        if key in d.uses:
            raise Error(label + ": duplicate Use")
        d.uses[key] = use^


def parse_directory(var text: List[UInt8], entry_name: String) raises -> Directory:
    """Parse and validate the entry directory JSON."""
    var d = Directory()
    var root = parse_json(text^, "artifact directory", d.doc)
    require_members(
        d.doc,
        root,
        ["components", "objects", "bindings", "uses", "files"],
        ["metadata", "provenance"],
        "directory",
    )
    _parse_files(d, d.doc.find(root, "files"), entry_name)
    _parse_objects(d, d.doc.find(root, "objects"))
    _parse_components(d, d.doc.find(root, "components"))
    var bindings = d.doc.find(root, "bindings")
    if not d.doc.is_object(bindings):
        raise Error("bindings must be an object")
    for i in range(d.doc.size(bindings)):
        var name = d.doc.key(bindings, i)
        if name.byte_length() == 0 or name.find("\0") >= 0:
            raise Error("parameter name must be nonempty without NUL")
        var b = _parse_binding(d, d.doc.at(bindings, i), name)
        d.binding_names.append(name)
        d.bindings[name] = b^
    d.binding_names = _sorted(d.binding_names.copy())
    _parse_uses(d, d.doc.find(root, "uses"))
    if d.doc.contains(root, "metadata"):
        d.metadata = d.doc.find(root, "metadata")
    if d.doc.contains(root, "provenance"):
        d.provenance = d.doc.find(root, "provenance")
    if (d.metadata >= 0 and not d.doc.is_object(d.metadata)) or (
        d.provenance >= 0 and not d.doc.is_object(d.provenance)
    ):
        raise Error("metadata and provenance must be objects")
    if d.metadata >= 0 and d.doc.contains(d.metadata, "name"):
        _ = require_id(d.doc, d.doc.find(d.metadata, "name"), "metadata.name")
    return d^
