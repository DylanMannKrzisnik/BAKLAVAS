# harmony_patch.py
import snapatac2.preprocessing._harmony as _hmod

_orig_worker = _hmod._harmony

def harmony_worker_fixed(*args, **kwargs):
    Z = _orig_worker(*args, **kwargs)
    # Collector expects (cells, dims); if we got (dims, cells) transpose.
    if getattr(Z, "ndim", None) == 2 and Z.shape[0] < Z.shape[1]:
        return Z.T
    return Z

def apply():
    _hmod._harmony = harmony_worker_fixed
