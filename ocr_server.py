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
import json
import os
import re
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, CLIPModel, CLIPProcessor
from qwen_vl_utils import process_vision_info

import barcode
from barcode.writer import ImageWriter
import win32print
import win32ui
from PIL import ImageWin

# ── API Key ───────────────────────────────────────────────────────────────────
# Change this to any secret string — must match the key baked into the app
API_KEY = "EZI-TRILO-OCR-2025"

# ── Printer configuration (Brother QL-800, via the normal Windows printer
# queue — confirmed working) ─────────────────────────────────────────────────
# Earlier assumption that this printer only worked through P-touch Editor
# Lite turned out to be wrong — it IS installed as a normal Windows printer
# and prints correctly through it. If printer_name isn't set, falls back to
# whatever Windows has as the default printer.
PRINT_CONFIG_FILE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "EZI OCR Server" / "printer.json"
DEFAULT_PRINTER_NAME = "Brother QL-800"

# ── Shape library storage ────────────────────────────────────────────────────
# Same persistent-data convention as the launcher (%LOCALAPPDATA%\EZI OCR
# Server\) so reference photos/embeddings survive reinstalling the app.
SHAPE_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "EZI OCR Server" / "shape_library"
SHAPE_IMAGES_DIR = SHAPE_DIR / "images"
SHAPE_INDEX_FILE = SHAPE_DIR / "index.json"
SHAPE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Cosine similarity below this is treated as "no confident match" rather than
# guessing. This is a starting point — tune it against your real parts once
# you have a working reference library; too low and unrelated parts will
# match each other, too high and legitimate matches get rejected.
SHAPE_MATCH_THRESHOLD = 0.80

# ── Load Models ───────────────────────────────────────────────────────────────
device = "cuda"
torch_dtype = torch.float16
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

print("=" * 60)
print("   LOADING QWEN2-VL-2B INTO GPU VRAM — EZI OCR SERVER")
print("=" * 60)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch_dtype, device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("[✓] OCR model loaded and ready.\n")

print("=" * 60)
print("   LOADING CLIP INTO GPU VRAM — SHAPE RECOGNITION")
print("=" * 60)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, torch_dtype=torch_dtype).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
print("[✓] Shape-recognition model loaded and ready.\n")

# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="EZI OCR Server", version="2.0.0")

# Allow requests from the Android WebView (Capacitor uses https://localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Reject any request that doesn't carry the correct API key.
    /health is exempt so the settings screen can ping without a key."""
    if request.method == "OPTIONS" or request.url.path in ("/health", "/", "/printer/last-label"):
        return await call_next(request)
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        print(f"[!] Unauthorized request from {request.client.host} — key: {key!r}")
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


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


# ── Shape recognition (CLIP embeddings + cosine similarity) ─────────────────

def compute_shape_embedding(pil_image: Image.Image) -> list[float]:
    """Return a normalized CLIP image embedding as a plain Python list
    (JSON-serializable, easy to store alongside the reference library)."""
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features[0].float().cpu().tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot  # both vectors are already L2-normalized, so dot == cosine sim


def load_printer_name() -> str:
    try:
        return json.loads(PRINT_CONFIG_FILE.read_text()).get("printer_name", "") or DEFAULT_PRINTER_NAME
    except Exception:
        return DEFAULT_PRINTER_NAME


def build_label_image(part_number: str, serial_number: str, part_name: str, width_px: int, height_px: int) -> Image.Image:
    """
    Render a label as just a barcode (Code128), scaled to fill as much of
    the label as possible, centered. part_name is accepted for signature
    compatibility with the caller but no longer used — text was dropped
    per request, barcode only.
    """
    label = Image.new("RGB", (width_px, height_px), "white")

    # Encode both numbers in the barcode when both exist, not just the part
    # number — a scanner reading this barcode later should get the full
    # picture, not just half of it.
    if part_number and serial_number:
        code_value = f"{part_number}|{serial_number}"
    else:
        code_value = part_number or serial_number or "UNKNOWN"

    # Render the barcode natively at (very close to) its final size by
    # computing the DPI needed, rather than rendering small and resizing up
    # — any resize/interpolation step blends adjacent black/white pixels at
    # bar edges, which is invisible on screen but shifts effective bar
    # widths just enough to break real barcode scanners. This renders each
    # bar at its correct pixel width directly, no interpolation involved.
    barcode_class = barcode.get_barcode_class("code128")
    bc_opts = {"module_height": 15.0, "font_size": 0, "text_distance": 0, "quiet_zone": 1}

    max_w = int(width_px * (1 - 0.06 * 2))
    max_h = int(height_px * (1 - 0.06 * 2))

    ref_dpi = 300
    ref_bc = barcode_class(code_value, writer=ImageWriter(dpi=ref_dpi))
    ref_img = ref_bc.render(writer_options=bc_opts)

    scale_needed = min(max_w / ref_img.width, max_h / ref_img.height)
    final_dpi = max(72, int(ref_dpi * scale_needed))

    if final_dpi == ref_dpi:
        bc_final = ref_img
    else:
        bc = barcode_class(code_value, writer=ImageWriter(dpi=final_dpi))
        bc_final = bc.render(writer_options=bc_opts)

    x = (width_px - bc_final.width) // 2
    y = (height_px - bc_final.height) // 2
    label.paste(bc_final, (x, y))

    return label


def print_label(part_number: str, serial_number: str, part_name: str) -> None:
    """Send a label straight to the Windows-installed printer via GDI —
    confirmed working through the normal Windows print system."""
    printer_name = load_printer_name()

    hDC = win32ui.CreateDC()
    hDC.CreatePrinterDC(printer_name)
    hDC.StartDoc("EZI Label")
    hDC.StartPage()

    # HORZRES=8, VERTRES=10 — standard Windows GDI DeviceCaps indices for
    # the printable area in pixels at the printer's current resolution.
    printable_w = hDC.GetDeviceCaps(8)
    printable_h = hDC.GetDeviceCaps(10)
    print(f"[i] Printer reports printable area: {printable_w} x {printable_h} px")

    label = build_label_image(part_number, serial_number, part_name, printable_w, printable_h)

    # Save every generated label so it can be inspected/verified — this is
    # what actually gets sent to the printer, so if nothing is coming out
    # physically, checking this image first tells us whether the problem
    # is "the image is blank/wrong" vs "the image is fine but printing it
    # fails somewhere in GDI/the driver."
    debug_path = PRINT_CONFIG_FILE.parent / "last_label_preview.png"
    try:
        PRINT_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        label.save(debug_path, "PNG")
    except Exception as e:
        print(f"[!] Couldn't save label preview: {e}")

    dib = ImageWin.Dib(label)
    dib.draw(hDC.GetHandleOutput(), (0, 0, printable_w, printable_h))

    hDC.EndPage()
    hDC.EndDoc()
    hDC.DeleteDC()


def load_shape_library() -> list[dict]:
    if not SHAPE_INDEX_FILE.exists():
        return []
    try:
        return json.loads(SHAPE_INDEX_FILE.read_text())
    except Exception:
        return []


def save_shape_library(entries: list[dict]) -> None:
    SHAPE_INDEX_FILE.write_text(json.dumps(entries))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/printer/last-label")
async def last_label_preview():
    """View the actual image generated for the most recent print attempt —
    open this in a browser to check whether it has visible content or is
    blank/malformed, without needing physical access to the printer."""
    debug_path = PRINT_CONFIG_FILE.parent / "last_label_preview.png"
    if not debug_path.exists():
        return JSONResponse({"error": "No label has been generated yet."}, status_code=404)
    return FileResponse(debug_path, media_type="image/png")


@app.get("/printer")
async def get_printer_config():
    """Show the configured printer name and what Windows currently has installed."""
    try:
        installed = [p[2] for p in win32print.EnumPrinters(win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS)]
    except Exception as e:
        installed = []
        print(f"[!] Couldn't enumerate printers: {e}")
    return {
        "configured": load_printer_name(),
        "windows_default": win32print.GetDefaultPrinter(),
        "installed": installed,
    }


@app.post("/printer")
async def set_printer_config(printer_name: str = Form(...)):
    PRINT_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    PRINT_CONFIG_FILE.write_text(json.dumps({"printer_name": printer_name}))
    return {"printer_name": printer_name}


@app.post("/print")
async def print_label_endpoint(
    part_number: str = Form(...),
    serial_number: str = Form(""),
    part_name: str = Form(""),
):
    """
    Tablet calls this when the operator taps Print. Generates a label
    (barcode + text) and sends it through the normal Windows print system
    to whatever printer is configured (confirmed working: Brother QL-800).
    """
    try:
        print_label(part_number, serial_number, part_name)
        print(f"[✓] Printed label: P/N={part_number!r} S/N={serial_number!r}")
        return {"status": "printed"}
    except Exception as e:
        print(f"[!] Print failed: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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


# ── Shape recognition endpoints ─────────────────────────────────────────────

@app.post("/shapes/register")
async def register_shape(part_number: str = Form(...), image: UploadFile = File(...)):
    """
    Admin panel calls this to add a reference photo for a part number.
    Multiple photos can be registered for the same part_number (e.g. from a
    couple of angles) — identification matches against all of them and takes
    the best score.
    """
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    entry_id = uuid.uuid4().hex
    image_path = SHAPE_IMAGES_DIR / f"{entry_id}.jpg"
    pil_image.save(image_path, "JPEG", quality=90)

    embedding = compute_shape_embedding(pil_image)

    library = load_shape_library()
    library.append({
        "id": entry_id,
        "part_number": part_number.strip().upper(),
        "embedding": embedding,
    })
    save_shape_library(library)

    print(f"[✓] Registered shape reference for part {part_number!r} (id={entry_id})")
    return {"id": entry_id, "part_number": part_number.strip().upper()}


@app.post("/shapes/identify")
async def identify_shape(image: UploadFile = File(...)):
    """
    Tablet calls this with a photo of an unidentified small part.
    Returns the best-matching part_number and its similarity score, or
    part_number: null if nothing in the library is a confident enough match.
    """
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    query_embedding = compute_shape_embedding(pil_image)

    library = load_shape_library()
    if not library:
        return {"part_number": None, "confidence": 0.0, "reason": "No reference shapes registered yet."}

    scored = sorted(
        (
            {"part_number": entry["part_number"], "score": cosine_similarity(query_embedding, entry["embedding"])}
            for entry in library
        ),
        key=lambda r: r["score"],
        reverse=True,
    )
    best = scored[0]
    print(f"[shape] best match {best['part_number']!r} score={best['score']:.3f} (threshold {SHAPE_MATCH_THRESHOLD})")

    if best["score"] < SHAPE_MATCH_THRESHOLD:
        return {"part_number": None, "confidence": round(best["score"], 3), "reason": "No confident match."}

    return {"part_number": best["part_number"], "confidence": round(best["score"], 3)}


@app.get("/shapes")
async def list_shapes():
    """Admin panel calls this to show what's registered, grouped by part number."""
    library = load_shape_library()
    grouped: dict[str, list[str]] = {}
    for entry in library:
        grouped.setdefault(entry["part_number"], []).append(entry["id"])
    return {"parts": [{"part_number": pn, "reference_ids": ids} for pn, ids in grouped.items()]}


@app.delete("/shapes/{entry_id}")
async def delete_shape(entry_id: str):
    library = load_shape_library()
    remaining = [e for e in library if e["id"] != entry_id]
    if len(remaining) == len(library):
        return JSONResponse({"error": "Not found"}, status_code=404)
    save_shape_library(remaining)
    image_path = SHAPE_IMAGES_DIR / f"{entry_id}.jpg"
    image_path.unlink(missing_ok=True)
    return {"deleted": entry_id}


if __name__ == "__main__":
    print("[*] Server starting on http://0.0.0.0:8000")
    print("[*] EZI app: tap ⚙ gear → enter http://<THIS-MACHINE-IP>:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
