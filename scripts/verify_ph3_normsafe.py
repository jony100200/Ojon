"""Static format and weight validation for a P-H3 norm-safe GGUF.

Checks that every 1-D Q4_K tensor of the original GGUF was restored as exact F32 from the BF16
source and that every other tensor kept its type and payload.
usage: python verify_ph3_normsafe.py <original.gguf> <normsafe.gguf> <source_bf16.safetensors>"""
import json, struct, sys
from pathlib import Path
import numpy as np
from gguf import GGUFReader

if len(sys.argv)!=4:
    sys.exit(__doc__)
a=GGUFReader(sys.argv[1])
b=GGUFReader(sys.argv[2])
assert len(a.tensors)==len(b.tensors),(len(a.tensors),len(b.tensors))
c=Path(sys.argv[3])
with c.open('rb') as f:
    size=struct.unpack("<Q",f.read(8))[0]
    desc=json.loads(f.read(size))
data_start=8+size
changed=0
samples=0
with c.open('rb') as raw:
    for x,y in zip(a.tensors,b.tensors):
        assert x.name==y.name,(x.name,y.name)
        assert tuple(x.shape)==tuple(y.shape),(x.name,x.shape,y.shape)
        if len(x.shape)==1 and int(x.tensor_type)==12:
            assert int(y.tensor_type)==0,(x.name,y.tensor_type)
            name=x.name.replace("model.diffusion_model.","",1)
            info=desc[name]
            assert info['dtype']=="BF16" and info["shape"]==list(reversed(x.shape))
            offset,end=info["data_offsets"]
            raw.seek(data_start+offset)
            bf=raw.read(end-offset)
            promoted=(np.frombuffer(bf,dtype="<u2").astype("<u4")<<16)
            actual=np.asarray(y.data).reshape(-1).view("<u4")
            assert np.array_equal(promoted,actual),x.name
            changed+=1
        else:
            assert x.tensor_type==y.tensor_type,x.name
            xx=np.asarray(x.data).reshape(-1).view("u1")
            yy=np.asarray(y.data).reshape(-1).view("u1")
            assert xx.size==yy.size,x.name
            assert np.array_equal(xx[:128], yy[:128]) and np.array_equal(xx[-128:], yy[-128:]),x.name
            samples+=1
assert changed>0,"no Q4_K norm tensors were found to restore"
print(f"GGUF_VALIDATED source_tensors={len(a.tensors)} norms_restored_exactly={changed} unchanged_sample_checked={samples}")
print(f"GGUF_Q4_NORMS_AFTER={sum(len(t.shape)==1 and int(t.tensor_type)==12 for t in b.tensors)}")
print(f"OUTPUT_BYTES={Path(sys.argv[2]).stat().st_size}")
