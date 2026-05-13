"""Top-level package init.

Compatibility shims that must run before any numpy-using submodule imports.
Keep this file tiny — it's loaded by every `from indextts...` import path.
"""
from __future__ import annotations

# --- numpy 2.x compatibility shims ---
# numpy 2.0 removed several legacy attributes that older third-party packages
# still reference at module-import time:
#   - `np.bool8`: referenced by tensorboard's tensorflow_stub.dtypes (line 326),
#     pulled in via audiotools → torch.utils.tensorboard.
#   - `np.sctypes`: referenced by NeMo's preprocessing.segment when normalizing
#     audio dtypes.
# Restore the names so those packages can still import. Purely additive; if a
# future numpy ever re-introduces them, our shims are no-ops.
import numpy as _np
if not hasattr(_np, "bool8"):
    _np.bool8 = _np.bool_
if not hasattr(_np, "sctypes"):
    _np.sctypes = {
        "int":     [_np.int8, _np.int16, _np.int32, _np.int64],
        "uint":    [_np.uint8, _np.uint16, _np.uint32, _np.uint64],
        "float":   [_np.float16, _np.float32, _np.float64],
        "complex": [_np.complex64, _np.complex128],
        "others":  [bool, object, bytes, str, _np.void],
    }
# Legacy scalar aliases that tensorboard / older audio libs sometimes reach for
for _legacy, _modern in (
    ("object0", "object_"),
    ("str0",    "str_"),
    ("bytes0",  "bytes_"),
    ("int0",    "intp"),
    ("uint0",   "uintp"),
):
    if not hasattr(_np, _legacy) and hasattr(_np, _modern):
        setattr(_np, _legacy, getattr(_np, _modern))
del _np
