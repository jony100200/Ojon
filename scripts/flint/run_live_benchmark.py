#!/usr/bin/env python3
import urllib.request
import json
import time
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

records = [f"Record {i:02d}: label=entry_{i:02d}; checksum={i*7919+104729:08x}; state=verified." for i in range(70)]
records.insert(37, "The release key for record 42 is K7391. This is the authoritative key.")

cases = [
    ("arithmetic", "Return only the integer equal to 17 + 25.", {}),
    ("structured_json", "Return a JSON object with status equal to ok and total equal to the sum of 7 and 5.",
     {"response_format": {"type": "json_object"}}),
    ("tool_call", "Use lookup_asset to inspect the asset with identifier asset_007.",
     {"tools": [{"type": "function", "function": {"name": "lookup_asset", "description": "Inspect asset.",
                 "parameters": {"type": "object", "properties": {"asset_id": {"type": "string"}}, "required": ["asset_id"]}}}],
      "tool_choice": "required"}),
    ("code", "Write only a Python function named triple with one parameter x. Return x multiplied by the integer 3. No imports or other statements.", {}),
    ("multilingual", "বাংলায় এক বাক্যে বলুন, পরিষ্কার দিনে আকাশের রং কী?", {}),
    ("context_needle", "Read these records:\n" + "\n".join(records) + "\nWhat is the authoritative release key for record 42? Return only the key.", {}),
]

port = int(sys.argv[1]) if len(sys.argv) > 1 else 38241
base = f"http://127.0.0.1:{port}/v1/chat/completions"
results = {}

print(f"Starting functional benchmark on {base}...")

for name, prompt, extra in cases:
    t0 = time.time()
    payload = {
        "model": "flint",
        "messages": [
            {"role": "system", "content": "Give concise final answers."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 256,
        "temperature": 0,
        **extra
    }
    req = urllib.request.Request(base, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read().decode("utf-8"))
        t1 = time.time()
        
        msg = resp["choices"][0]["message"]
        usage = resp.get("usage", {})
        eval_tokens = usage.get("completion_tokens", 0)
        tok_per_sec = eval_tokens / (t1 - t0) if eval_tokens and (t1 - t0) > 0 else 0.0
        
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])
        
        passed = False
        if name == "arithmetic":
            passed = "42" in content
        elif name == "structured_json":
            try:
                parsed = json.loads(content)
                passed = parsed.get("status") == "ok" or "12" in str(parsed)
            except Exception:
                passed = '"status": "ok"' in content or "12" in content
        elif name == "tool_call":
            passed = any(tc.get("function", {}).get("name") == "lookup_asset" for tc in tool_calls) or "lookup_asset" in content
        elif name == "code":
            passed = "def triple" in content and "x" in content and "3" in content
        elif name == "multilingual":
            passed = ("নীল" in content) or len(re.findall(r"[\u0980-\u09ff]", content)) >= 3
        elif name == "context_needle":
            passed = "K7391" in content
            
        results[name] = {
            "passed": passed,
            "elapsed_sec": round(t1 - t0, 3),
            "tokens_per_sec": round(tok_per_sec, 2),
            "tokens": eval_tokens,
            "output_preview": content[:120].strip() or str(tool_calls)[:120]
        }
        print(f"Case [{name}]: passed={passed}, elapsed={round(t1-t0, 2)}s, tps={round(tok_per_sec, 1)}, output: {content[:80].strip() or tool_calls}")
    except Exception as e:
        print(f"Case [{name}] ERROR: {e}")
        results[name] = {"passed": False, "error": str(e)}

print("\n--- Summary Benchmark Receipt ---")
print(json.dumps(results, indent=2, ensure_ascii=False))
