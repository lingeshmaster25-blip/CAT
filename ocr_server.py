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
import math
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
import win32con
import win32gui
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

# Physical label stock — 12 mm wide CONTINUOUS tape (8 m or 30 m roll).
# The tape width is fixed; the length of each printed label is computed per
# print job from the barcode content (a longer value => a longer label), then
# forced via DEVMODE for that job.
#   LABEL_WIDTH_MM   = tape width (short side, fixed)
#   MODULE_WIDTH_MM  = width of the thinnest barcode bar. 0.17 mm = 2 printer dots at 300 dpi (matches the sample label); raise to 0.254 (3 dots) if scans fail. 0.25 mm scans
#                      reliably; do not go much below ~0.2 mm.
#   MIN_LENGTH_MM    = shortest label that will ever be printed
#   SIDE_MARGIN_MM   = blank tape at each end, beyond the barcode quiet zone
LABEL_WIDTH_MM = 12
MODULE_WIDTH_MM = 0.17
MIN_LENGTH_MM = 30
SIDE_MARGIN_MM = 1.0
QUIET_ZONE_MM = 1.5

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
                        "This is an embossed industrial machine plate with exactly two lines "
                        "of text. Transcribe each line exactly as embossed — character by "
                        "character, top to bottom. Do not interpret, classify, correct, or "
                        "reorder anything; just read what is physically there.\n"
                        "Respond in this exact format only, with no extra words:\n"
                        "LINE1: <first line, top to bottom, exactly as embossed>\n"
                        "LINE2: <second line, top to bottom, exactly as embossed>"
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


def looks_like_serial(value: str) -> bool:
    """
    Best-effort heuristic for this business's serial-number convention: a
    short leading letter code (e.g. 'CM1', 'IMOP') followed by space-
    separated digit groups, one of which looks like a two-digit month
    (01-12) — matching the "plant code + date + sequence" pattern.

    Deliberately NOT given to the model as a prompt example (that caused
    the model to literally echo example values instead of reading the
    plate — see run_model). This lives in code instead, where it's
    inspectable and tunable against real scans without risking the model
    copying it.

    Tuned against real examples so far:
      'IMOP 09 26 0500' -> True  (serial: IMOP + month 09 + 26 + 0500)
      'CM1 05 26 0017'  -> True  (serial: CM1 + month 05 + 26 + 0017)
      '658 7404 08'     -> False (part: no letters at all)
      '9R0314'          -> False (part: letter embedded mid-digits, no
                            leading letter block, no space-separated
                            month-like group)
    """
    v = value.strip().upper()
    m = re.match(r"^([A-Z]+[0-9]?)\s+(.*)$", v)
    if not m:
        return False
    for group in m.group(2).split():
        if re.fullmatch(r"0[1-9]|1[0-2]", group):
            return True
    return False


def classify_serial_and_part(line1: str, line2: str) -> tuple[str, str]:
    """
    Decide which transcribed line is the serial number and which is the
    part number. The model no longer attempts this classification itself
    (see run_model's prompt) — it's done here deterministically so it can
    be tuned against real plates without touching model behavior.
    """
    l1, l2 = line1.strip(), line2.strip()
    if not l1 and not l2:
        return "", ""
    if not l2:
        return (l1, "") if (looks_like_serial(l1) or re.search(r"[A-Z]", l1)) else ("", l1)
    if not l1:
        return (l2, "") if (looks_like_serial(l2) or re.search(r"[A-Z]", l2)) else ("", l2)

    l1_serial, l2_serial = looks_like_serial(l1), looks_like_serial(l2)
    if l1_serial and not l2_serial:
        return l1, l2
    if l2_serial and not l1_serial:
        return l2, l1

    # Neither matched the serial date-pattern confidently. Fall back to:
    # a line with letters is more likely the serial in this convention
    # (part numbers here are pure digits, or short codes with an embedded
    # rather than leading letter) — prefer whichever line has letters.
    l1_has_letters, l2_has_letters = bool(re.search(r"[A-Z]", l1)), bool(re.search(r"[A-Z]", l2))
    if l1_has_letters and not l2_has_letters:
        return l1, l2
    if l2_has_letters and not l1_has_letters:
        return l2, l1

    # Truly ambiguous — keep the model's own line order.
    return l1, l2


def parse_output(raw: str) -> dict:
    """
    Parse Qwen's raw transcription (LINE1/LINE2, see run_model) into
    serial / part / confidence. Falls back to naive line splitting if the
    model doesn't follow the LINE1:/LINE2: format exactly.
    """
    line1 = ""
    line2 = ""

    line1_match = re.search(r"LINE1[:\s]+([A-Z0-9 \-]+)", raw, re.IGNORECASE)
    line2_match = re.search(r"LINE2[:\s]+([A-Z0-9 \-]+)", raw, re.IGNORECASE)

    if line1_match:
        line1 = line1_match.group(1).strip().upper()
    if line2_match:
        line2 = line2_match.group(1).strip().upper()

    # Fallback: just take the first two non-empty lines
    if not line1 and not line2:
        lines = [
            re.sub(r"[^A-Z0-9 ]", " ", l).strip().upper()
            for l in raw.splitlines()
            if re.sub(r"[^A-Z0-9 ]", "", l).strip()
        ]
        if lines:
            line1 = lines[0]
        if len(lines) > 1:
            line2 = lines[1]

    serial, part = classify_serial_and_part(line1, line2)

    # Estimate confidence based on how well the output matched
    confidence = 90.0 if (line1_match and line2_match) else 65.0

    return {"serial": serial, "part": part, "confidence": confidence}


# ── Shape recognition (CLIP embeddings + cosine similarity) ─────────────────

def compute_shape_embedding(pil_image: Image.Image) -> list[float]:
    """Return a normalized CLIP image embedding as a plain Python list
    (JSON-serializable, easy to store alongside the reference library).

    Deliberately calls clip_model.vision_model + clip_model.visual_projection
    directly instead of clip_model.get_image_features(...) — on the
    transformers version installed on the production server,
    get_image_features() returns the raw vision-tower output
    (a BaseModelOutputWithPooling) instead of the projected embedding
    tensor, which crashed every /shapes/register call with:
    "AttributeError: 'BaseModelOutputWithPooling' object has no attribute
    'norm'". Going straight to vision_model + visual_projection sidesteps
    that entirely and is correct on every transformers version.
    """
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        pooled_output = vision_outputs.pooler_output
        features = clip_model.visual_projection(pooled_output)
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


def compute_barcode_pieces(part_number: str, serial_number: str) -> list[tuple[str, str]]:
    """Separate barcodes as (label, value) pairs — serial number on top,
    part number below. The barcode itself encodes just the raw value; the
    label is only used for the human-readable caption. If only one is
    present (e.g. this part has no serial), returns just that one piece —
    build_label_image() centers a single piece automatically."""
    # One combined barcode "SERIAL|PART" (whitespace stripped) — same value
    # the app shows on screen. Its own value is printed underneath; no
    # separate caption line above it.
    s = re.sub(r"\s+", "", serial_number or "")
    p = re.sub(r"\s+", "", part_number or "")
    value = f"{s}|{p}" if (s and p) else (s or p or "UNKNOWN")
    return [("", value)]


def _render_piece(value: str, dpi: float, module_width_mm: float, module_height_mm: float, quiet_zone_mm: float) -> Image.Image:
    """Bars only, no baked-in text — text is added separately in
    build_label_image() AFTER any rotation, so it always stays upright and
    readable regardless of how the bars are oriented on the label."""
    barcode_class = barcode.get_barcode_class("code128")
    bc = barcode_class(value, writer=ImageWriter(dpi=dpi))
    opts = {
        "module_width": module_width_mm,
        "module_height": module_height_mm,
        "font_size": 0,
        "text_distance": 0,
        "quiet_zone": quiet_zone_mm,
    }
    return bc.render(writer_options=opts)


def _compose_with_text(
    bars_img: Image.Image, caption_text: str, dpi: float, max_width_px: int,
    font_pt: float = 24, gap_mm: float = 3,
) -> Image.Image:
    """Stack upright, horizontal caption text below the (possibly
    already-rotated) bars image. Always horizontal regardless of the bars'
    own orientation — this is what keeps text readable after rotation.
    Font size is scaled to fit max_width_px (the label's short/fixed
    dimension) so longer captions don't overflow or get clipped."""
    draw_tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def measure(font_size: int):
        try:
            f = ImageFont.truetype("arial.ttf", size=font_size)
        except Exception:
            f = ImageFont.load_default()
        bbox = draw_tmp.textbbox((0, 0), caption_text, font=f)
        return f, bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Start from a reasonable size and scale down to fit if needed.
    font_size = max(8, int(dpi * font_pt / 72))
    font, text_w, text_h = measure(font_size)
    max_text_w = int(max_width_px * 0.92)
    if text_w > max_text_w:
        font_size = max(8, int(font_size * max_text_w / text_w))
        font, text_w, text_h = measure(font_size)

    text_margin = int(dpi * gap_mm / 25.4)  # gap between bars and text
    out_w = max(bars_img.width, text_w)
    out_h = bars_img.height + text_margin + text_h + 4
    out = Image.new("RGB", (out_w, out_h), "white")
    out.paste(bars_img, ((out_w - bars_img.width) // 2, 0))

    draw = ImageDraw.Draw(out)
    draw.text(
        (out_w // 2, bars_img.height + text_margin),
        caption_text, fill="black", font=font, anchor="ma",
    )
    return out


def compute_label_length_mm(value: str) -> float:
    """Label length needed for this barcode value at MODULE_WIDTH_MM bars,
    including quiet zones and end margins (never below MIN_LENGTH_MM)."""
    ref_dpi = 300
    img = _render_piece(value, ref_dpi, MODULE_WIDTH_MM, 5.0, QUIET_ZONE_MM)
    bars_mm = img.width / ref_dpi * 25.4
    return float(max(MIN_LENGTH_MM, math.ceil(bars_mm + 2 * SIDE_MARGIN_MM)))


def build_label_image(pieces: list[tuple[str, str]], width_px: int, height_px: int) -> Image.Image:
    """
    One combined barcode (bars) with its value printed underneath, centred
    on a continuous-tape label LABEL_WIDTH_MM (12 mm) wide and as long as
    compute_label_length_mm() says (the DEVMODE length for this job).

    Orientation-agnostic: the design is built landscape (long side
    horizontal). If the driver reports a portrait printable area
    (height > width), the finished landscape image is rotated 90 degrees to
    fill it, so it works whichever way the driver feeds the tape.
    """
    portrait = height_px > width_px
    long_px, short_px = (height_px, width_px) if portrait else (width_px, height_px)

    value = pieces[0][1]
    length_mm = compute_label_length_mm(value)

    label = Image.new("RGB", (long_px, short_px), "white")
    dpi = max(72, round(long_px / (length_mm / 25.4)))

    margin_mm = 0.8
    font_pt = 6
    gap_mm = 0.6
    # Bars take what's left of the short side after margins, text and gap.
    text_mm = font_pt * 25.4 / 72 * 1.25
    module_height_mm = max(2.0, LABEL_WIDTH_MM - 2 * margin_mm - text_mm - gap_mm)

    bars = _render_piece(value, dpi, MODULE_WIDTH_MM, module_height_mm, QUIET_ZONE_MM)
    piece_img = _compose_with_text(bars, value, dpi, long_px, font_pt=font_pt, gap_mm=gap_mm)

    # Clamp defensively (driver snapping the length/width a little) so
    # nothing sits flush against, or runs off, the physical edge.
    safe_w = int(long_px * 0.99)
    safe_h = int(short_px * 0.96)
    if piece_img.height > safe_h or piece_img.width > safe_w:
        shrink = min(safe_h / piece_img.height, safe_w / piece_img.width)
        piece_img = piece_img.resize(
            (max(1, int(piece_img.width * shrink)), max(1, int(piece_img.height * shrink))),
            Image.NEAREST,
        )

    x = (long_px - piece_img.width) // 2
    y = (short_px - piece_img.height) // 2
    label.paste(piece_img, (x, y))

    if portrait:
        label = label.rotate(90, expand=True)
    return label


def get_forced_label_devmode(printer_name: str, length_mm: float):
    """
    Build a DEVMODE forcing the exact tape width (LABEL_WIDTH_MM, fixed)
    and a per-job length (continuous roll — computed from the barcode).

    NOTE: this is the one piece of this file I could not test against real
    hardware — win32print/win32ui only run on Windows, and I have no
    Windows machine available. This follows the standard, widely-used
    pywin32 pattern for setting a custom paper size, but if it throws or
    the driver rejects it, print_label() falls back to the previous
    (confirmed-working) ambient-settings behavior rather than failing the
    print entirely.
    """
    hPrinter = win32print.OpenPrinter(printer_name)
    try:
        properties = win32print.GetPrinter(hPrinter, 2)
        devmode = properties["pDevMode"]

        # DEVMODE paper width/length are in tenths of a millimetre.
        # PaperSize must be DMPAPER_USER (256) to tell the driver "use the
        # custom PaperWidth/PaperLength below" — 0 is NOT the correct
        # sentinel for this and likely caused the driver's validation step
        # to snap back to the nearest REGISTERED form it knew about
        # instead (e.g. an old "EZI Label" 29x90mm form from earlier
        # testing), which is exactly the mismatch you saw.
        DMPAPER_USER = getattr(win32con, "DMPAPER_USER", 256)
        devmode.PaperSize = DMPAPER_USER
        devmode.PaperWidth = int(LABEL_WIDTH_MM * 10)
        devmode.PaperLength = int(length_mm * 10)
        devmode.Fields |= win32con.DM_PAPERSIZE | win32con.DM_PAPERWIDTH | win32con.DM_PAPERLENGTH

        # Let the driver validate/normalize our requested settings — some
        # drivers adjust custom sizes to the nearest size they actually
        # support, which is exactly why this step matters.
        win32print.DocumentProperties(
            0, hPrinter, printer_name, devmode, devmode,
            win32con.DM_IN_BUFFER | win32con.DM_OUT_BUFFER,
        )
        print(
            f"[i] After driver validation: PaperSize={devmode.PaperSize} "
            f"PaperWidth={devmode.PaperWidth} (={devmode.PaperWidth/10}mm) "
            f"PaperLength={devmode.PaperLength} (={devmode.PaperLength/10}mm)"
        )
        return devmode
    finally:
        win32print.ClosePrinter(hPrinter)


def print_label(part_number: str, serial_number: str, part_name: str) -> None:
    """Send a label straight to the Windows-installed printer via GDI —
    confirmed working through the normal Windows print system."""
    printer_name = load_printer_name()
    pieces = compute_barcode_pieces(part_number, serial_number)
    # Continuous tape: length follows the barcode content.
    length_mm = compute_label_length_mm(pieces[0][1])
    print(f"[i] Label length for this print: {length_mm} mm (tape {LABEL_WIDTH_MM} mm)")

    devmode = get_forced_label_devmode(printer_name, length_mm)
    try:
        # win32ui's PyCDC has no generic CreateDC(driver, device, output,
        # devmode) method — that was my earlier incorrect guess. The
        # actual way to apply a custom DEVMODE is the lower-level
        # win32gui.CreateDC(driver, device, devmode) — note: only 3 args,
        # it drops the rarely-used "output" parameter the raw Win32 API
        # has. Returns a raw HDC handle; wrap that into a PyCDC via
        # CreateDCFromHandle to get the StartDoc/StartPage etc. methods
        # the rest of this function uses.
        raw_hdc = win32gui.CreateDC("WINSPOOL", printer_name, devmode)
        hDC = win32ui.CreateDCFromHandle(raw_hdc)
        print(f"[✓] Forced label size to {LABEL_WIDTH_MM}mm x {length_mm}mm via DEVMODE.")
    except Exception as e:
        # Fall back to the previously-working ambient approach rather than
        # failing the print entirely — but note this means the custom
        # length won't be applied, so continuous-roll jobs may paginate.
        print(f"[!] Couldn't force label size via DEVMODE ({e}); falling back to ambient printer settings.")
        hDC = win32ui.CreateDC()
        hDC.CreatePrinterDC(printer_name)

    hDC.StartDoc("EZI Label")
    hDC.StartPage()

    # HORZRES=8, VERTRES=10 — standard Windows GDI DeviceCaps indices for
    # the printable area in pixels at the printer's current resolution.
    printable_w = hDC.GetDeviceCaps(8)
    printable_h = hDC.GetDeviceCaps(10)
    print(f"[i] Printer reports printable area: {printable_w} x {printable_h} px")

    label = build_label_image(pieces, printable_w, printable_h)

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

    # Explicitly convert to pure 1-bit black/white with NO dithering before
    # handing off to GDI. Everything drawn in build_label_image() is
    # already solid black or white (no anti-aliasing), so this conversion
    # is lossless — but it also means Windows' driver has no ambiguous
    # grayscale/color content left to dither on its own. Some label
    # printer drivers apply automatic dithering when converting color/gray
    # images down to 1-bit for the thermal head, which would corrupt sharp
    # barcode edges even though the source image is already clean.
    label_bw = label.convert("1", dither=Image.NONE)
    dib = ImageWin.Dib(label_bw)
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
