"""Patch codec.py in the installed michelangelo package.

Filters the kwargs dict to only include keys accepted by cls.__init__ before
calling cls(**dct) in DataclassCodec.decode. This fixes TypeError when the
serialized form contains _io_metadata but the target class __init__ predates
that field being added as a parameter.
"""
import sys

path = "/app/michelangelo/uniflow/core/codec.py"
with open(path) as f:
    code = f.read()

old = "        del dct[self._ATTR_CLASS]\n        return cls(**dct)"
new = (
    "        del dct[self._ATTR_CLASS]\n"
    "        import inspect as _i\n"
    "        _p = set(_i.signature(cls.__init__).parameters) - {'self'}\n"
    "        return cls(**{k: v for k, v in dct.items() if k in _p})"
)

if old not in code:
    print("ERROR: patch target not found in codec.py", file=sys.stderr)
    sys.exit(1)

with open(path, "w") as f:
    f.write(code.replace(old, new, 1))

print("codec.py patched OK")
