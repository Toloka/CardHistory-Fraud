"""Save and load the released weights: one .safetensors file per model, with its configuration in the file's metadata.

The files follow the safetensors format (https://github.com/huggingface/safetensors): 8 bytes of header length, a JSON header
(tensor names, dtypes, shapes, byte offsets, and a "__metadata__" map of strings), then the raw little-endian tensor bytes.
This module reads and writes that format with numpy only, so the models need no extra package; the `safetensors` package
reads the same files.
"""
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import SeqModel  # noqa: E402

_DT = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
       "U8": np.uint8, "BOOL": np.bool_}
_NAME = {np.dtype(v): k for k, v in _DT.items()}


def save_file(tensors, path, metadata=None):
    header, chunks, offset = {}, [], 0
    for name in sorted(tensors):
        a = tensors[name].detach().cpu().contiguous().numpy() if torch.is_tensor(tensors[name]) else np.ascontiguousarray(tensors[name])
        b = a.astype(a.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {"dtype": _NAME[a.dtype], "shape": list(a.shape), "data_offsets": [offset, offset + len(b)]}
        chunks.append(b); offset += len(b)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)                                 # pad the header to a multiple of 8 bytes
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for b in chunks:
            f.write(b)
    os.replace(tmp, path)


def load_file(path):
    """(dict of torch tensors, metadata dict)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        buf = f.read()
    meta = header.pop("__metadata__", {})
    out = {}
    for name, info in header.items():
        lo, hi = info["data_offsets"]
        a = np.frombuffer(buf, dtype=np.dtype(_DT[info["dtype"]]).newbyteorder("<"), count=(hi - lo) // np.dtype(_DT[info["dtype"]]).itemsize,
                          offset=lo).reshape(info["shape"])
        out[name] = torch.from_numpy(a.astype(a.dtype.newbyteorder("="), copy=True))
    return out, meta


def build_model(cfg):
    m = cfg["model"]
    return SeqModel(m["vocab_sizes"], d_model=m["d_model"], n_layers=m["n_layers"], n_heads=m["n_heads"], d_field=m["d_field"],
                    field_layers=m["field_layers"], L_max=m["L_max"], emb_dim=m["emb_dim"], n_mcc=m["n_mcc"], dropout=m["dropout"],
                    next_from_emb=m["next_from_emb"], n_amt=m["n_amt"], n_num=m["n_num"], d_ff=m["d_ff"])


def load_model(path, device="cpu"):
    """(model in eval mode, its configuration) from a released .safetensors file."""
    sd, meta = load_file(path)
    cfg = json.loads(meta["config"])
    model = build_model(cfg)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), cfg


def save_model(model, cfg, path):
    save_file(model.state_dict(), path, {"config": json.dumps(cfg), "format": "pt"})
