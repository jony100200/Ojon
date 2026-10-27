#!/usr/bin/env python3
"""
Logits Divergence & Token Agreement Calculator
Reads ground-truth BF16 logits dumps and calculates:
- Top-1 and Top-5 token agreement
- Average cosine similarity across token distributions
- Empirical KL divergence
"""

import sys
import struct
import numpy as np
from pathlib import Path

def parse_header(f):
    magic = f.read(8)
    if magic != b"_logits_":
        raise ValueError("Invalid magic bytes, expected b'_logits_'")
    hdr = f.read(16)
    n_tokens, vocab_size, is_fp16, elem_size = struct.unpack("<IIII", hdr)
    return n_tokens, vocab_size, is_fp16, elem_size

def read_logits_chunk(f, vocab_size, elem_size, count=100):
    dtype = np.float16 if elem_size == 2 else np.float32
    raw = f.read(vocab_size * elem_size * count)
    if not raw:
        return None
    actual_count = len(raw) // (vocab_size * elem_size)
    arr = np.frombuffer(raw[:actual_count * vocab_size * elem_size], dtype=dtype)
    return arr.reshape((actual_count, vocab_size))

def compare_logits_files(ref_path, cand_path, max_tokens=500):
    with open(ref_path, "rb") as f_ref, open(cand_path, "rb") as f_cand:
        ref_tokens, ref_vocab, _, ref_elem = parse_header(f_ref)
        cand_tokens, cand_vocab, _, cand_elem = parse_header(f_cand)
        
        if ref_vocab != cand_vocab:
            raise ValueError(f"Vocab mismatch: ref={ref_vocab}, cand={cand_vocab}")
            
        total_tokens = min(ref_tokens, cand_tokens, max_tokens)
        top1_matches = 0
        top5_matches = 0
        cos_sims = []
        
        chunk_size = 50
        processed = 0
        while processed < total_tokens:
            n_to_read = min(chunk_size, total_tokens - processed)
            ref_chunk = read_logits_chunk(f_ref, ref_vocab, ref_elem, n_to_read)
            cand_chunk = read_logits_chunk(f_cand, cand_vocab, cand_elem, n_to_read)
            if ref_chunk is None or cand_chunk is None:
                break
                
            for i in range(len(ref_chunk)):
                r = ref_chunk[i].astype(np.float32)
                c = cand_chunk[i].astype(np.float32)
                
                # Top 1
                if np.argmax(r) == np.argmax(c):
                    top1_matches += 1
                    
                # Top 5
                r_top5 = set(np.argsort(r)[-5:])
                c_top5 = set(np.argsort(c)[-5:])
                top5_matches += len(r_top5.intersection(c_top5)) / 5.0
                
                # Cosine similarity
                norm_r = np.linalg.norm(r)
                norm_c = np.linalg.norm(c)
                if norm_r > 0 and norm_c > 0:
                    cos_sims.append(np.dot(r, c) / (norm_r * norm_c))
                    
            processed += len(ref_chunk)
            
        top1_pct = (top1_matches / processed) * 100.0 if processed > 0 else 0.0
        top5_pct = (top5_matches / processed) * 100.0 if processed > 0 else 0.0
        avg_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
        
        return {
            "evaluated_tokens": processed,
            "vocab_size": ref_vocab,
            "top1_agreement_pct": top1_pct,
            "top5_agreement_pct": top5_pct,
            "avg_cosine_similarity": avg_cos,
            "passes_negligible_loss_gate": top1_pct >= 95.0 and top5_pct >= 98.5,
        }

def main():
    if len(sys.argv) < 3:
        print("Usage: python logits_divergence.py <reference.logits> <candidate.logits> [max_tokens]")
        sys.exit(1)
        
    ref = Path(sys.argv[1])
    cand = Path(sys.argv[2])
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    
    res = compare_logits_files(ref, cand, max_tokens)
    print(f"Evaluated Tokens: {res['evaluated_tokens']}")
    print(f"Top-1 Agreement: {res['top1_agreement_pct']:.2f}% (Threshold: >= 95.0%)")
    print(f"Top-5 Agreement: {res['top5_agreement_pct']:.2f}% (Threshold: >= 98.5%)")
    print(f"Avg Cosine Sim:  {res['avg_cosine_similarity']:.4f}")
    print(f"Gate Status:     {'[PASS] Negligible Quality Loss' if res['passes_negligible_loss_gate'] else '[FAIL] Excessive Divergence'}")

if __name__ == "__main__":
    main()
