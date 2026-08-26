# Row-major stride helpers, replacing MAX 25.3's tensor._indexing.

from std.utils.index import IndexList


def _row_major_strides[rank: Int](shape: IndexList[rank]) -> IndexList[rank]:
    var strides = IndexList[rank]()
    var stride = 1
    for i in range(rank - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return strides


def _dot_prod[rank: Int](a: IndexList[rank], b: IndexList[rank]) -> Int:
    var total = 0
    for i in range(rank):
        total += a[i] * b[i]
    return total
