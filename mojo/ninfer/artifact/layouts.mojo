"""Tensor interpretation; mirrors `src/artifact/layouts.cpp`."""

from ninfer.artifact.formats import parse_format, parse_layout
from ninfer.artifact.schema import Object
from ninfer.core.weight import WeightGeometry, weight_geometry


def describe_tensor(object: Object) raises -> WeightGeometry:
    """Interpret and validate a requested tensor's complete encoded parent."""
    try:
        var geometry = weight_geometry(
            parse_format(object.tensor.format),
            parse_layout(object.tensor.layout),
            object.tensor.shape,
            object.tensor.divisors,
        )
        if geometry.bytes != object.bytes or object.offset % geometry.alignment != 0:
            raise Error("encoded size or object alignment differs from layout")
        return geometry^
    except e:
        raise Error(object.id + ": " + String(e))
