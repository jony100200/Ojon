"""Repair Ojon P-H3 GGUF Q4_K 1D norms without loading entire checkpoints.

The original stable-diffusion.cpp conversion quantized RMSNorm vectors.
ComfyUI-GGUF treats those packed Q4_K bytes as plain normalization weights.
This tool copies all other GGUF tensor bytes unchanged, while restoring only
1D Q4_K normalization tensors as F32 from the original BF16 safetensors.

Default: validation/dry run. --confirm: create distinct atomic output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

GGUF_PREFIX = "model.diffusion_model."
Q4_K = 12
F32 = 0
ALIGN = 32
CHUNK = 4 * 1024 * 1024

def u32(f):
    b = f.read(4)
    if len(b) != 4: raise ValueError("Truncated u32")
    return struct.unpack("<I", b)[0]

def u64(f):
    b = f.read(8)
    if len(b) != 8: raise ValueError("Truncated u64")
    return struct.unpack("<Q", b)[0]

def align(n):
    return (n + ALIGN - 1) & ~(ALIGN - 1)

def gguf_info(path):
    with path.open("rb") as f:
        if f.read(4) != b"GGUF": raise ValueError("Expected GGUF")
        ver, count, nkv = u32(f), u64(f), u64(f)
        if ver != 3 or not 1 <= count <= 100000 or nkv != 1:
            raise ValueError(f"Unsupported header: GGUF {ver}, {count} tensors, {nkv} metadata")
        meta_begin = f.tell()
        key_len = u64(f)
        key = f.read(key_len).decode("utf-8")
        vtype = u32(f)
        value_len = u64(f)
        value = f.read(value_len).decode("utf-8")
        if (key, vtype, value) != ("general.architecture", 8, "wan"):
            raise ValueError(f"Unexpected metadata: {key} {vtype} {value}")
        meta_end = f.tell()
        f.seek(meta_begin)
        metadata = f.read(meta_end - meta_begin)
        rows = []
        for _ in range(count):
            name_len = u64(f)
            if not 1 <= name_len <= 4096: raise ValueError("Bad tensor name length")
            name = f.read(name_len).decode("utf-8")
            ndim = u32(f)
            if not 1 <= ndim <= 4: raise ValueError("Unexpected tensor rank")
            dims = [u64(f) for _ in range(ndim)]
            qtype, offset = u32(f), u64(f)
            rows.append(dict(name=name,dims=dims,qtype=qtype,offset=offset))
        data_start = align(f.tell())
    file_size = path.stat().st_size
    if not rows or rows[0]["offset"] != 0:
        raise ValueError("Bad first tensor offset")
    for a,b in zip(rows,rows[1:]):
        if a["offset"] >= b["offset"] or b["offset"] % ALIGN:
            raise ValueError("Invalid tensor offsets/alignment")
    if data_start + rows[-1]["offset"] >= file_size:
        raise ValueError("Tensor payload outside file")
    return metadata,rows,data_start,file_size

def safetensors_info(path):
    with path.open("rb") as f:
        n = u64(f)
        if not 1 <= n <= 32 * 1024 * 1024: raise ValueError("Bad safetensors header")
        meta = json.loads(f.read(n))
    return meta,8+n

def restore_bytes(source, source_data, sd):
    name = source["name"][len(GGUF_PREFIX):]
    s = sd.get(name)
    if s is None: raise ValueError(f"Missing BF16 tensor: {name}")
    if s["dtype"] != "BF16" or s["shape"] != list(reversed(source["dims"])):
        raise ValueError(f"Mismatch for {name}: {s.get('dtype')}, {s.get('shape')}, {source['dims']}")
    a,b = s["data_offsets"]
    if b-a != source["dims"][0]*2: raise ValueError(f"Wrong source length: {name}")
    # BF16 => exact F32 promotion via shifting 16 bits, no torch or precision loss.
    with source_data.open("rb") as f:
        f.seek(sd["_data_start"]+a)
        raw = f.read(b-a)
    import numpy as np
    out = (np.frombuffer(raw,dtype="<u2").astype("<u4") << 16).tobytes()
    if len(out) != source["dims"][0]*4: raise AssertionError("Unexpected restored length")
    return out

def prepare(src, checkpoint):
    meta,rows,old_data_start,old_file_size = gguf_info(src)
    sd,sd_data_start=safetensors_info(checkpoint)
    sd["_data_start"]=sd_data_start
    repairs=[]
    for row in rows:
        if row["qtype"] == Q4_K and len(row["dims"]) == 1:
            if not row["name"].startswith(GGUF_PREFIX) or "norm" not in row["name"]:
                raise ValueError(f"Unexpected quantized 1D tensor: {row['name']}")
            # validate matching original shape without reading its values
            orig=sd.get(row["name"][len(GGUF_PREFIX):])
            if not orig or orig["dtype"] != "BF16" or orig["shape"] != list(reversed(row["dims"])):
                raise ValueError(f"No matching BF16 reference for {row['name']}")
            repairs.append(row["name"])
    if len(repairs) != 105:
        raise ValueError(f"Expected 105 P-H3 Q4_K 1D norms, found {len(repairs)}")
    cursor = 0
    for i,row in enumerate(rows):
        row["new_offset"] = align(cursor)
        old_next=rows[i+1]["offset"] if i+1<len(rows) else old_file_size-old_data_start
        old_span=old_next-row["offset"]
        if old_span <= 0: raise ValueError("Invalid tensor span")
        row["old_span"]=old_span
        row["repair"]=row["name"] in repairs
        row["new_type"]=F32 if row["repair"] else row["qtype"]
        row["new_size"]=row["dims"][0]*4 if row["repair"] else old_span
        cursor=row["new_offset"]+row["new_size"]
    raw_header=bytearray()
    raw_header += b"GGUF"+struct.pack("<IQQ",3,len(rows),1)
    raw_header += meta
    for row in rows:
        name=row["name"].encode("utf-8")
        raw_header += struct.pack("<Q",len(name))+name+struct.pack("<I",len(row["dims"]))
        for dim in row["dims"]: raw_header += struct.pack("<Q",dim)
        raw_header += struct.pack("<IQ",row["new_type"],row["new_offset"])
    new_data_start=align(len(raw_header))
    return sd,rows,old_data_start,old_file_size,bytes(raw_header),new_data_start,cursor

def copy_span(src,dst,amount):
    while amount:
        block=src.read(min(amount,CHUNK))
        if not block: raise EOFError("Unexpected GGUF EOF")
        dst.write(block)
        amount-=len(block)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("reference",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--confirm",action="store_true")
    a=p.parse_args()
    if a.source.resolve()==a.output.resolve() or a.reference.resolve()==a.output.resolve():
        raise ValueError("Refusing to overwrite input")
    sd,rows,old_start,old_size,head,new_start,new_end=prepare(a.source,a.reference)
    repairs=sum(r["repair"] for r in rows)
    print(f"VALIDATED tensors={len(rows)} restored_norms={repairs} input_bytes={old_size} output_bytes={new_start+new_end}",flush=True)
    if not a.confirm:
        print("DRY RUN; add --confirm to write separate GGUF",flush=True)
        return
    if a.output.exists(): raise FileExistsError(str(a.output))
    temp=a.output.with_suffix(a.output.suffix+".partial")
    if temp.exists(): raise FileExistsError(str(temp))
    with a.source.open("rb") as src, temp.open("xb") as dst:
        dst.write(head)
        dst.write(b"\0"*(new_start-dst.tell()))
        for i,row in enumerate(rows):
            position=dst.tell()-new_start
            if position>row["new_offset"]: raise ValueError("Computed GGUF offsets overlap")
            dst.write(b"\0"*(row["new_offset"]-position))
            if row["repair"]:
                dst.write(restore_bytes(row,a.reference,sd))
            else:
                src.seek(old_start+row["offset"])
                copy_span(src,dst,row["old_span"])
            if i%75==74:
                print(f"tensors_processed={i+1}/{len(rows)}",flush=True)
        dst.flush()
        os.fsync(dst.fileno())
    if temp.stat().st_size != new_start+new_end:
        raise AssertionError("Output file size mismatch")
    os.replace(temp,a.output)
    print(f"FINISHED {a.output} size={a.output.stat().st_size}",flush=True)

if __name__=="__main__": main()
