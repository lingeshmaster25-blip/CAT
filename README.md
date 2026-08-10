# EZI OCR Server Launcher

A one-button front end for `backend/ocr_server.py`. It doesn't bundle
Python, torch, transformers, or the model — it launches the Python
environment you already have working on this PC (the same one you use to
run `ocr_server.py` today) as a background process, and shows its status
and log output in a clean window instead of an IDE console.

## First run

1. Open the app, click the ⚙ (gear) in the top right.
2. **Python executable** — browse to the `python.exe` inside the
   environment/venv that already has `torch`, `transformers`,
   `fastapi`, etc. installed (the one you currently use to run the server).
3. **ocr_server.py location** — browse to `backend/ocr_server.py`.
4. **Port** — leave as `8000` unless you changed it.
5. Save.

These are remembered in `server_launcher_config.json`, saved next to the
app — you only need to do this once per machine.

## Using it

- **Start Server** — launches `ocr_server.py` in the background. The log
  panel shows the same output you'd see in a terminal (model loading,
  request logs, etc.). Once ready, the status turns green and shows the
  URL to enter in the EZI app's Settings (⚙) screen, e.g.
  `http://192.168.1.23:8000`.
- **Stop Server** — same button, now labeled Stop — cleanly terminates it.
- Closing the window while the server is running asks for confirmation and
  stops it first (no orphaned background process).

## Building the Windows .exe

Same pattern as the main EZI app: GitHub Actions builds it on a Windows
runner, no local Windows machine needed.

1. Push this folder to GitHub (as part of the same repo, or its own).
2. Actions tab → **Build OCR Server Launcher** → Run workflow (or push to
   `main` — it runs automatically for changes under `server-launcher/`).
3. Download the **EZI-OCR-Server-Launcher-Windows** artifact once green —
   it's a single portable `.exe`, no installer needed.

### Build locally instead

```bash
pip install -r requirements-build.txt
pyinstaller --onefile --windowed --name "EZI OCR Server Launcher" server_launcher.py
```
Output lands in `dist/`.
