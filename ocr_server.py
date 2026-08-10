"""
EZI OCR Backend Server — Trilo Automation
Model: Qwen2-VL-2B-Instruct on RTX 4050/4060/5060

Endpoints:
  GET  /health  — connectivity check from the EZI app
  POST /ocr     — receives cropped plate JPEG, returns serial + part
  GET  /        — browser UI for manual testing (legacy)
  POST /scan    — legacy browser UI endpoint

Run:
    pip install fastapi uvicorn python-multipart pillow torch transformers qwen-vl-utils
    python ocr_server.py
"""

import io
import re
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── Load Model ────────────────────────────────────────────────────────────────
device = "cuda"
torch_dtype = torch.float16
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print("=" * 60)
print("   LOADING QWEN2-VL-2B INTO GPU VRAM — EZI OCR SERVER")
print("=" * 60)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch_dtype, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("[✓] Model loaded and ready.\n")

# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="EZI OCR Server", version="2.0.0")

# Allow requests from the Android WebView (Capacitor uses https://localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def run_model(pil_image: Image.Image) -> str:
    """Run Qwen2-VL inference and return raw output text."""
    pil_image.thumbnail((1024, 1024))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {
                    "type": "text",
                    "text": (
                        "This is an embossed industrial machine plate. "
                        "Read the text on the plate carefully. "
                        "The plate has exactly two lines of text: "
                        "the top line is the SERIAL NUMBER and the bottom line is the PART NUMBER. "
                        "Respond in this exact format only, with no extra words:\n"
                        "SERIAL: <value>\n"
                        "PART: <value>"
                    ),
                },
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def parse_output(raw: str) -> dict:
    """
    Parse Qwen output into serial / part / confidence.
    Expected format:
        SERIAL: CM1 05 26 0017
        PART: 668 9100 02
    Falls back to line splitting if the model doesn't follow the format exactly.
    """
    serial = ""
    part = ""

    # Try structured format first
    serial_match = re.search(r"SERIAL[:\s]+([A-Z0-9 \-]+)", raw, re.IGNORECASE)
    part_match   = re.search(r"PART[:\s]+([A-Z0-9 \-]+)",   raw, re.IGNORECASE)

    if serial_match:
        serial = serial_match.group(1).strip().upper()
    if part_match:
        part = part_match.group(1).strip().upper()

    # Fallback: just take the first two non-empty lines
    if not serial and not part:
        lines = [
            re.sub(r"[^A-Z0-9 ]", " ", l).strip().upper()
            for l in raw.splitlines()
            if re.sub(r"[^A-Z0-9 ]", "", l).strip()
        ]
        if lines:
            serial = lines[0]
        if len(lines) > 1:
            part = lines[1]

    # Estimate confidence based on how well the output matched
    confidence = 90.0 if (serial_match and part_match) else 65.0

    return {"serial": serial, "part": part, "confidence": confidence}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Connectivity check — pinged by the EZI app settings screen."""
    return {"status": "ok", "service": "EZI OCR Server", "model": MODEL_ID}


@app.post("/ocr")
async def ocr_endpoint(image: UploadFile = File(...)):
    """
    Main endpoint called by the EZI Android app after the user crops the plate.

    Accepts:  multipart/form-data  field: 'image'  (JPEG)
    Returns:  { serial: str, part: str, confidence: float }
    """
    print(f"\n[→] /ocr received: {image.filename}")

    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    raw = run_model(pil_image)
    print(f"[←] Raw model output: {raw!r}")

    result = parse_output(raw)
    print(f"[✓] serial={result['serial']!r}  part={result['part']!r}  conf={result['confidence']}")

    return result


# ── Legacy browser UI (kept for manual testing) ───────────────────────────────

HTML_UI = """<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EZI OCR Server</title>
  <style>
    body{font-family:sans-serif;text-align:center;padding:20px;background:#121212;color:#fff}
    .btn{background:#00ffcc;color:#000;padding:20px;font-size:18px;font-weight:bold;border:none;
         border-radius:10px;width:100%;margin-top:20px;cursor:pointer}
    #result{margin-top:30px;text-align:left;background:#1e1e1e;padding:15px;border-radius:10px;
            border:1px solid #333;white-space:pre-wrap;font-family:monospace}
  </style>
</head>
<body>
  <h2>EZI OCR Server — Test UI</h2>
  <input type="file" accept="image/*" capture="environment" id="f" style="display:none">
  <button class="btn" onclick="document.getElementById('f').click()">📷 Scan Plate</button>
  <div id="result">Awaiting image...</div>
  <script>
    document.getElementById('f').addEventListener('change', async e => {
      const file = e.target.files[0]; if (!file) return;
      document.getElementById('result').innerText = 'Processing...';
      const fd = new FormData(); fd.append('image', file);
      try {
        const r = await fetch('/ocr', {method:'POST', body:fd});
        const d = await r.json();
        document.getElementById('result').innerText =
          'Serial:     ' + (d.serial || '—') + '\\nPart:       ' + (d.part || '—') +
          '\\nConfidence: ' + (d.confidence || '—') + '%';
      } catch(err) {
        document.getElementById('result').innerText = 'Error: ' + err;
      }
    });
  </script>
</body>
</html>"""

@app.get("/")
async def home():
    return HTMLResponse(content=HTML_UI)

@app.post("/scan")
async def scan_legacy(file: UploadFile = File(...)):
    """Legacy endpoint — kept for backward compatibility with the old HTML UI."""
    image_bytes = await file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    raw = run_model(pil_image)
    result = parse_output(raw)
    return {
        "text": f"SERIAL: {result['serial']}\nPART: {result['part']}",
        **result,
    }


if __name__ == "__main__":
    import os
    port = int(os.environ.get("EZI_OCR_PORT", "8000"))
    print(f"[*] Server starting on http://0.0.0.0:{port}")
    print(f"[*] EZI app: tap ⚙ gear → enter http://<THIS-MACHINE-IP>:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
