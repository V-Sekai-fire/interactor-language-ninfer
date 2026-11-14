"""`std::sort` as libstdc++ implements it (introsort, median of three, heap fallback, final
insertion sort), so ties land where they do in the C++ engine built on Linux."""

from ninfer.text.jinja.error import JinjaError, jinja_error
from std.collections import List

from ninfer.text.jinja.value import (
    Heap, V_ARRAY, V_TUPLE, V_OBJECT, OP_GT, OP_LT, vtype, value_compare, obj_at_strict, is_array_kind,
)

comptime _THRESHOLD = 16


struct SortSpec(Copyable, Movable):
    """What the comparator looks at. `mode` 0: array elements; 1: dictsort by key; 2: dictsort by value
    (elements are then indices into `keys`/`vals`)."""

    var mode: Int
    var reverse: Bool
    var attribute: Int
    var attr_is_int: Bool
    var attr_int: Int
    var keys: List[Int]
    var vals: List[Int]

    def __init__(out self, mode: Int, reverse: Bool):
        self.mode = mode
        self.reverse = reverse
        self.attribute = -1
        self.attr_is_int = False
        self.attr_int = 0
        self.keys = List[Int]()
        self.vals = List[Int]()


def _array_at(h: Heap, arr: Int, index_in: Int) raises JinjaError -> Int:
    var index = index_in
    var n = len(h.nodes[arr].arr)
    if index < 0:
        index += n
    if index < 0 or index >= n:
        raise jinja_error("Index " + String(index) + " out of bounds for array of size " + String(n))
    return h.nodes[arr].arr[index]


def _less(mut h: Heap, spec: SortSpec, a: Int, b: Int) raises JinjaError -> Bool:
    var op = OP_GT if spec.reverse else OP_LT
    if spec.mode == 1:
        return value_compare(h, spec.keys[a], spec.keys[b], op)
    if spec.mode == 2:
        return value_compare(h, spec.vals[a], spec.vals[b], op)
    var va = a
    var vb = b
    if h.nodes[spec.attribute].kind != 1:  # not undefined
        if spec.attr_is_int and is_array_kind(h.nodes[a].kind) and is_array_kind(h.nodes[b].kind):
            va = _array_at(h, a, spec.attr_int)
            vb = _array_at(h, b, spec.attr_int)
        elif (not spec.attr_is_int) and h.nodes[a].kind == V_OBJECT and h.nodes[b].kind == V_OBJECT:
            va = obj_at_strict(h, a, spec.attribute)
            vb = obj_at_strict(h, b, spec.attribute)
        else:
            raise jinja_error(
                "sort: unsupported object attribute comparison between " + vtype(h, a) + " and " + vtype(h, b)
            )
    return value_compare(h, va, vb, op)


def _swap(mut v: List[Int], i: Int, j: Int):
    var t = v[i]
    v[i] = v[j]
    v[j] = t


def _lg(n: Int) -> Int:
    var r = 0
    var x = n
    while x > 1:
        x >>= 1
        r += 1
    return r


def _adjust_heap(
    mut h: Heap, spec: SortSpec, mut v: List[Int], first: Int, hole_in: Int, length: Int, value: Int
) raises JinjaError:
    var top = hole_in
    var hole = hole_in
    var second = hole
    while second < (length - 1) // 2:
        second = 2 * (second + 1)
        if _less(h, spec, v[first + second], v[first + second - 1]):
            second -= 1
        v[first + hole] = v[first + second]
        hole = second
    if (length & 1) == 0 and second == (length - 2) // 2:
        second = 2 * (second + 1)
        v[first + hole] = v[first + second - 1]
        hole = second - 1
    var parent = (hole - 1) // 2
    while hole > top and _less(h, spec, v[first + parent], value):
        v[first + hole] = v[first + parent]
        hole = parent
        parent = (hole - 1) // 2
    v[first + hole] = value


def _make_heap(mut h: Heap, spec: SortSpec, mut v: List[Int], first: Int, last: Int) raises JinjaError:
    var length = last - first
    if length < 2:
        return
    var parent = (length - 2) // 2
    while True:
        _adjust_heap(h, spec, v, first, parent, length, v[first + parent])
        if parent == 0:
            return
        parent -= 1


def _sort_heap(mut h: Heap, spec: SortSpec, mut v: List[Int], first: Int, last_in: Int) raises JinjaError:
    var last = last_in
    while last - first > 1:
        last -= 1
        var value = v[last]
        v[last] = v[first]
        _adjust_heap(h, spec, v, first, 0, last - first, value)


def _move_median_to_first(
    mut h: Heap, spec: SortSpec, mut v: List[Int], result: Int, a: Int, b: Int, c: Int
) raises JinjaError:
    if _less(h, spec, v[a], v[b]):
        if _less(h, spec, v[b], v[c]):
            _swap(v, result, b)
        elif _less(h, spec, v[a], v[c]):
            _swap(v, result, c)
        else:
            _swap(v, result, a)
    elif _less(h, spec, v[a], v[c]):
        _swap(v, result, a)
    elif _less(h, spec, v[b], v[c]):
        _swap(v, result, c)
    else:
        _swap(v, result, b)


def _unguarded_partition(
    mut h: Heap, spec: SortSpec, mut v: List[Int], first_in: Int, last_in: Int, pivot: Int
) raises JinjaError -> Int:
    var first = first_in
    var last = last_in
    while True:
        while _less(h, spec, v[first], v[pivot]):
            first += 1
        last -= 1
        while _less(h, spec, v[pivot], v[last]):
            last -= 1
        if not (first < last):
            return first
        _swap(v, first, last)
        first += 1


def _introsort_loop(
    mut h: Heap, spec: SortSpec, mut v: List[Int], first: Int, last_in: Int, depth_in: Int
) raises JinjaError:
    var last = last_in
    var depth = depth_in
    while last - first > _THRESHOLD:
        if depth == 0:
            _make_heap(h, spec, v, first, last)
            _sort_heap(h, spec, v, first, last)
            return
        depth -= 1
        var mid = first + (last - first) // 2
        _move_median_to_first(h, spec, v, first, first + 1, mid, last - 1)
        var cut = _unguarded_partition(h, spec, v, first + 1, last, first)
        _introsort_loop(h, spec, v, cut, last, depth)
        last = cut


def _unguarded_linear_insert(mut h: Heap, spec: SortSpec, mut v: List[Int], last_in: Int) raises JinjaError:
    var last = last_in
    var value = v[last]
    var nxt = last - 1
    while _less(h, spec, value, v[nxt]):
        v[last] = v[nxt]
        last = nxt
        nxt -= 1
    v[last] = value


def _insertion_sort(mut h: Heap, spec: SortSpec, mut v: List[Int], first: Int, last: Int) raises JinjaError:
    if first == last:
        return
    for i in range(first + 1, last):
        if _less(h, spec, v[i], v[first]):
            var value = v[i]
            var j = i
            while j > first:
                v[j] = v[j - 1]
                j -= 1
            v[first] = value
        else:
            _unguarded_linear_insert(h, spec, v, i)


def std_sort(mut h: Heap, spec: SortSpec, mut v: List[Int]) raises JinjaError:
    var n = len(v)
    if n == 0:
        return
    _introsort_loop(h, spec, v, 0, n, _lg(n) * 2)
    if n > _THRESHOLD:
        _insertion_sort(h, spec, v, 0, _THRESHOLD)
        for i in range(_THRESHOLD, n):
            _unguarded_linear_insert(h, spec, v, i)
    else:
        _insertion_sort(h, spec, v, 0, n)
