#!/usr/bin/env python3
"""
MoE Router Guard (ProQuant RoCo Invariant Validator)
Verifies that no directional abliteration or weight edit touches
the sensitive MoE router gating input matrices (ffn_gate_inp / router).
"""

import sys
import json
from pathlib import Path

def audit_tensor_list(tensor_names):
    violations = []
    for name in tensor_names:
        lower = name.lower()
        if "ffn_gate_inp" in lower or "router" in lower or "gate_exps" in lower:
            violations.append(name)
    return violations

def main():
    if len(sys.argv) < 2:
        print("Usage: python moe_router_guard.py <tensors_json_or_recipe>")
        sys.exit(1)
        
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"Error: file not found: {path}")
        sys.exit(1)
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    tensors = []
    if isinstance(data, list):
        tensors = data
    elif isinstance(data, dict):
        if "tensors" in data:
            tensors = [t.get("name", t) if isinstance(t, dict) else t for t in data["tensors"]]
        elif "edits" in data:
            tensors = [e.get("tensor", "") for e in data["edits"]]
        elif "recipe" in data:
            tensors = list(data["recipe"].keys())
            
    violations = audit_tensor_list(tensors)
    if violations:
        print(f"[FAIL] ProQuant RoCo invariant violated! Found {len(violations)} protected router tensors:")
        for v in violations:
            print(f"  - {v}")
        sys.exit(2)
    else:
        print(f"[PASS] All {len(tensors)} target tensors respect the MoE router protection invariant.")
        sys.exit(0)

if __name__ == "__main__":
    main()
