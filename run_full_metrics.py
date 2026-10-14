import urllib.request
import json
import time
import subprocess

def run_metric(prompt, max_tokens, name):
    payload = {
        "model": "qwen3.8:27b-unlimited",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.0,
            "num_gpu": 99,
            "num_thread": 8,
            "low_vram": False
        }
    }
    
    t0 = time.time()
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    
    total_time = time.time() - t0
    
    # Extraire métriques
    eval_count = res.get("eval_count", 0)
    eval_dur = res.get("eval_duration", 1) / 1e9
    eval_spd = eval_count / eval_dur if eval_dur > 0 else 0
    
    prompt_count = res.get("prompt_eval_count", 0)
    prompt_dur = res.get("prompt_eval_duration", 1) / 1e9
    prompt_spd = prompt_count / prompt_dur if prompt_dur > 0 else 0
    
    # Extraire mémoire via /api/ps
    req_ps = urllib.request.Request("http://127.0.0.1:11434/api/ps")
    with urllib.request.urlopen(req_ps) as resp_ps:
        ps_data = json.loads(resp_ps.read().decode("utf-8"))
    
    vram_model_gb = 0.0
    for m in ps_data.get("models", []):
        if "qwen3.8:27b" in m.get("name", ""):
            vram_model_gb = m.get("size_vram", 0) / (1024**3)
            
    # Extraire VRAM totale système
    smi = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram"]).decode("utf-8")
    used_vram_gb = 0.0
    for line in smi.splitlines():
        if "GPU[0]" in line and "Used Memory" in line:
            bytes_val = int(line.split(":")[-1].strip())
            used_vram_gb = bytes_val / (1024**3)
            
    print(f"=== [MÉTRIQUE: {name}] ===")
    print(f"  * Prompt (Prefill)    : {prompt_count} tokens en {prompt_dur:.3f}s -> {prompt_spd:.2f} tok/s")
    print(f"  * Génération (Decode) : {eval_count} tokens en {eval_dur:.3f}s -> {eval_spd:.2f} tok/s")
    print(f"  * Temps total requête : {total_time:.3f}s")
    print(f"  * VRAM Modèle (Poids) : {vram_model_gb:.2f} Go")
    print(f"  * VRAM Totale Allouée : {used_vram_gb:.2f} Go / 31.86 Go (avec KV Cache)")
    print()

run_metric("Explain what makes Rust unique in 3 sentences:", 64, "Génération Courte (64 tokens)")
run_metric("Count from 1 to 200:", 128, "Génération Moyenne Déterministe (128 tokens)")
run_metric("List 10 major algorithms in computer science with a 2-sentence description for each:", 256, "Génération Longue Raisonnement (256 tokens)")
