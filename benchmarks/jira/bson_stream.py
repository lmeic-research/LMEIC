"""
Minimal streaming BSON reader for the Apache Jira MSR dumps (no bsons/pymongo available).

Supports the element types these dumps use: double, string, embedded document, array,
binary, ObjectId, bool, UTC datetime, null, int32, int64. Unknown types are skipped by their
length prefix, which keeps the stream aligned.

A .bson.gz file is a concatenation of length-prefixed BSON documents:
  int32 total_len | elements | 0x00
"""
from __future__ import annotations

import gzip
import struct
import datetime as _dt

_INT32 = 16
_INT64 = 18
_DOUBLE = 1
_STRING = 2
_DOCUMENT = 3
_ARRAY = 4
_BINARY = 5
_OBJECTID = 7
_BOOL = 8
_DATETIME = 9
_NULL = 10


def _cstring(buf: bytes, i: int) -> tuple[str, int]:
    j = buf.index(b"\x00", i)
    return buf[i:j].decode("utf-8", "replace"), j + 1


def _doc(buf: bytes, start: int) -> tuple[dict, int]:
    (total,) = struct.unpack_from("<i", buf, start)
    end = start + total
    i = start + 4
    out: dict = {}
    while i < end - 1:
        t = buf[i]
        i += 1
        key, i = _cstring(buf, i)
        if t == _DOUBLE:
            (out[key],) = struct.unpack_from("<d", buf, i); i += 8
        elif t == _STRING:
            (n,) = struct.unpack_from("<i", buf, i)
            out[key] = buf[i + 4:i + 4 + n - 1].decode("utf-8", "replace"); i += 4 + n
        elif t in (_DOCUMENT, _ARRAY):
            sub, i = _doc(buf, i)
            out[key] = list(sub.values()) if t == _ARRAY else sub
        elif t == _BINARY:
            (n,) = struct.unpack_from("<i", buf, i)
            out[key] = buf[i + 5:i + 5 + n]; i += 5 + n
        elif t == _OBJECTID:
            out[key] = buf[i:i + 12].hex(); i += 12
        elif t == _BOOL:
            out[key] = buf[i] != 0; i += 1
        elif t == _DATETIME:
            (ms,) = struct.unpack_from("<q", buf, i); i += 8
            out[key] = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(milliseconds=ms)
        elif t == _NULL:
            out[key] = None
        elif t == _INT32:
            (out[key],) = struct.unpack_from("<i", buf, i); i += 4
        elif t == _INT64:
            (out[key],) = struct.unpack_from("<q", buf, i); i += 8
        else:  # skip by declared length for types not used here
            (n,) = struct.unpack_from("<i", buf, i)
            out[key] = None; i += 4 + n
    return out, end


def iter_docs(path: str, batch: int = 2000):
    """Yield documents from a gzipped BSON stream, `batch` documents at a time from the reader."""
    with gzip.open(path, "rb") as f:
        buf = b""
        pos = 0
        while True:
            chunk = f.read(1 << 22)
            if not chunk:
                break
            buf = buf[pos:] + chunk
            pos = 0
            while True:
                if len(buf) - pos < 4:
                    break
                (total,) = struct.unpack_from("<i", buf, pos)
                if len(buf) - pos < total:
                    break
                doc, pos = _doc(buf, pos)
                yield doc
