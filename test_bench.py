import urllib.request
import json
import time

payload = {
    "model": "qwen3.8:27b-unlimited",
    "prompt": "Count from 1 to 100:",
    "stream": False,
    "options": {
        "num_predict": 128,
        "temperature": 0.0,
        "num_gpu": 99,
        "num_thread": 8
    }
}

req = urllib.request.Request(
    "http://127.0.0.1:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    eval_count = res.get("eval_count", 0)
    eval_dur = res.get("eval_duration", 1)
    prompt_count = res.get("prompt_eval_count", 0)
    prompt_dur = res.get("prompt_eval_duration", 1)
    
    print(f"Prompt: {prompt_count} tokens in {prompt_dur / 1e9:.3f}s -> {prompt_count / (prompt_dur / 1e9):.2f} tok/s")
    print(f"Eval: {eval_count} tokens in {eval_dur / 1e9:.3f}s -> {eval_count / (eval_dur / 1e9):.2f} tok/s")
    print(f"Total: {res.get('total_duration', 0) / 1e9:.3f}s")
