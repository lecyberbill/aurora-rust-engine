import urllib.request
import json
import time

def test_config(name, options):
    payload = {
        "model": "qwen3.8:27b-unlimited",
        "prompt": "Explain quantum computing in 10 concise bullet points:",
        "stream": False,
        "options": options
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        ec = res.get("eval_count", 0)
        ed = res.get("eval_duration", 1)
        pc = res.get("prompt_eval_count", 0)
        pd = res.get("prompt_eval_duration", 1)
        prompt_spd = pc / (pd / 1e9) if pd > 0 else 0
        eval_spd = ec / (ed / 1e9) if ed > 0 else 0
        print(f"[{name}] Prompt: {pc} tok @ {prompt_spd:.1f} tok/s | Eval: {ec} tok @ {eval_spd:.1f} tok/s (eval_dur: {ed/1e9:.2f}s)")

# Warmup
test_config("Default (Warmup)", {"num_predict": 128})
test_config("num_gpu=99, num_thread=8", {"num_predict": 128, "num_gpu": 99, "num_thread": 8})
test_config("num_gpu=99, num_thread=16", {"num_predict": 128, "num_gpu": 99, "num_thread": 16})
test_config("num_gpu=99, num_batch=512", {"num_predict": 128, "num_gpu": 99, "num_batch": 512})
test_config("num_gpu=99, num_batch=1024", {"num_predict": 128, "num_gpu": 99, "num_batch": 1024})
test_config("num_gpu=99, num_batch=2048", {"num_predict": 128, "num_gpu": 99, "num_batch": 2048})
test_config("num_gpu=99, low_vram=False", {"num_predict": 128, "num_gpu": 99, "low_vram": False})
