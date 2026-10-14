import urllib.request
import json
import base64
import sys

data = {
    "data": [
        "Krea 2 Turbo (Qwen3-VL + VAE)",
        "1024×1024",
        "a majestic white wolf on a snowy cliff at golden hour, photorealistic, 8k",
        "",
        8,
        1.0,
        42,
        "data:image/png;base64,",
        []
    ]
}

req = urllib.request.Request(
    "http://127.0.0.1:7860/api/predict",
    data=json.dumps(data).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("Sending predict request to http://127.0.0.1:7860/api/predict ...")
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("Response keys:", list(res.keys()))
        outputs = res.get("outputs", res.get("data", {}))
        print("Outputs:", outputs if isinstance(outputs, list) else list(outputs.keys()))
        
        # Check if t2i_output in outputs or res
        img_b64 = ""
        if isinstance(outputs, dict):
            img_b64 = outputs.get("t2i_output", "")
        elif isinstance(outputs, list):
            for item in outputs:
                if isinstance(item, str) and item.startswith("data:image"):
                    img_b64 = item
                    break
        
        if img_b64.startswith("data:image/"):
            header, payload = img_b64.split(",", 1)
            raw = base64.b64decode(payload)
            with open("output_test_krea2.png", "wb") as f:
                f.write(raw)
            print(f"SUCCESS: Saved output image to output_test_krea2.png ({len(raw)} bytes)")
        else:
            print("No base64 image in response, dumping full response:")
            print(json.dumps(res, indent=2)[:500])
except Exception as e:
    print(f"Error during request: {e}")
    sys.exit(1)
