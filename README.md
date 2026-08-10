# EZI OCR Server Launcher

A one-button, self-contained front end for `ocr_server.py`. The server
script and its dependency list are bundled inside the app — the customer
never sees a file picker, a Python path, or any settings screen.

## What the customer sees

1. Double-click the app.
2. Click **Start Server**.
3. First time only: a "Setting up…" status while the app downloads its own
   private Python environment and the ML dependencies (PyTorch,
   transformers, etc.) — this needs internet access and can take a while
   depending on connection speed. Progress streams into the Log panel.
4. Once ready, the status turns green and shows the URL to enter in the EZI
   app's Settings (⚙) screen, e.g. `http://192.168.1.23:8000`. The Qwen2-VL
   model itself (~4 GB) also downloads automatically the first time the
   server actually starts — same one-time cost, visible in the log.
5. Every run after that skips straight to step 2 — nothing is re-downloaded.

Nothing is installed system-wide. The private environment lives under
`%LOCALAPPDATA%\EZI OCR Server\`, isolated from anything else on the
customer's machine, and is unaffected by re-installing or replacing the
launcher `.exe` itself.

## How it works (for you, not the customer)

- `ocr_server.py` and `requirements.txt` are bundled into the `.exe` as
  data files via PyInstaller's `--add-data`.
- On first "Start Server" click, `server_launcher.py`:
  1. Downloads the official Windows *embeddable* Python distribution (not a
     system install — just a folder of files) into
     `%LOCALAPPDATA%\EZI OCR Server\python_env\`.
  2. Bootstraps `pip` into it via `get-pip.py`.
  3. Installs the CUDA build of `torch` (see `TORCH_INDEX_URL` in
     `server_launcher.py` — set this to match the CUDA version your
     customers' NVIDIA drivers support), then the rest of
     `requirements.txt`.
  4. Marks setup complete and launches `ocr_server.py` using that private
     Python.
- On every later run, setup is skipped and it goes straight to launching
  the server.

### If you need to change the port

Ship-time default is `8000`. It's stored in
`%LOCALAPPDATA%\EZI OCR Server\port.json` — edit that once on the target
machine if a customer's setup needs a different port; there is deliberately
no in-app UI for this so the customer never has to think about it.

## Building the Windows .exe

Same pattern as the main EZI app: GitHub Actions builds it on a Windows
runner, no local Windows machine needed.

1. Push this folder to GitHub.
2. Actions tab → **Build OCR Server Launcher** → Run workflow (or push to
   `main`).
3. Download the **EZI-OCR-Server-Launcher-Windows** artifact once green —
   GitHub zips it automatically. Unzip and hand the customer the whole
   `EZI OCR Server Launcher` folder; the rest happens on their machine the
   first time they click Start.

### Build locally instead

```bash
pip install -r requirements-build.txt
pyinstaller --onedir --windowed --name "EZI OCR Server Launcher" ^
  --add-data "ocr_server.py;." ^
  --add-data "requirements.txt;." ^
  server_launcher.py
```

Output lands in `dist/EZI OCR Server Launcher/` — it's a folder, not a
single file (this build faster and starts faster than a one-file exe).
Hand the customer the whole folder (zipped); they run the `.exe` inside it.
