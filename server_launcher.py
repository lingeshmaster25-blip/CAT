"""
EZI OCR Server Launcher — Trilo Automation

Self-contained: the customer never picks a Python interpreter or a script
file. This app carries ocr_server.py and its dependency list with it.

First run on a machine:
  1. Downloads a small, private, portable Python (not the system Python —
     nothing is installed system-wide, nothing conflicts with anything else
     on the machine).
  2. Installs pip into it, then installs torch (CUDA build) + fastapi +
     transformers + everything ocr_server.py needs. This step needs an
     internet connection and downloads a few GB — progress is streamed into
     the Log panel.
  3. Starts ocr_server.py using that private Python.

Every run after that: the private env already exists, so it skips straight
to step 3 — the model weights (~4 GB) still download once, automatically,
the first time the server actually starts, same as step 2.

Everything lives under %LOCALAPPDATA%\\EZI OCR Server\\ so re-installing or
replacing the launcher .exe doesn't force a re-download.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path
from tkinter import messagebox, ttk
import tkinter as tk

APP_TITLE = "EZI OCR Server"

# ── Where the bundled server script lives ───────────────────────────────────
# When built with PyInstaller (--add-data "ocr_server.py;."), it's extracted
# next to this file at runtime under sys._MEIPASS. When run directly as a
# .py during development, it's just next to this script.
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
BUNDLED_SERVER_SCRIPT = BUNDLE_DIR / "ocr_server.py"
BUNDLED_REQUIREMENTS = BUNDLE_DIR / "requirements.txt"

# ── Where the private, per-machine runtime lives (NOT inside the bundle —
# this has to be writable and persist across app updates) ──────────────────
APP_DATA_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "EZI OCR Server"
ENV_DIR = APP_DATA_DIR / "python_env"
SETUP_MARKER = ENV_DIR / ".setup_complete"
PORT_CONFIG_FILE = APP_DATA_DIR / "port.json"

PYTHON_VERSION = "3.11.9"
EMBED_PYTHON_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
    f"python-{PYTHON_VERSION}-embed-amd64.zip"
)
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
# cu128 supports Blackwell (RTX 50-series, sm_120) as well as everything
# older down through Ampere/Turing. If you ever ship to a fleet of much
# older cards only, a smaller/older index would also work, but cu128 is the
# safe default for current-gen NVIDIA hardware.
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

# Bump this whenever requirements.txt or the torch/torchvision install line
# changes. Machines that already ran setup under an older version will
# automatically redo just the pip-install steps (not the ~100MB Python
# download) instead of silently keeping stale/missing packages.
SETUP_VERSION = "6"

DEFAULT_PORT = 8000


def load_port() -> int:
    try:
        return json.loads(PORT_CONFIG_FILE.read_text())["port"]
    except Exception:
        return DEFAULT_PORT


def save_port(port: int) -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PORT_CONFIG_FILE.write_text(json.dumps({"port": port}))


def local_ip() -> str:
    """Best-effort LAN IP so the user knows what to type into the EZI app."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def python_exe_path() -> Path:
    return ENV_DIR / "python.exe"


# ─────────────────────────────────────────────────────────────────────── UI ──
class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("560x420")
        self.minsize(480, 360)
        self.configure(bg="#0b0b0c")

        self.port = load_port()
        self.process: subprocess.Popen | None = None
        self.running = False
        self.busy = False  # true during first-run environment setup

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if not BUNDLED_SERVER_SCRIPT.exists():
            # This is a packaging problem, not something the customer can
            # fix — surface it plainly rather than pretending.
            self.after(200, lambda: messagebox.showerror(
                APP_TITLE,
                "ocr_server.py is missing from this build.\n\n"
                "This copy of the app was packaged incorrectly. "
                "Please contact support.",
            ))

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self):
        FG = "#e7e7e8"
        SUB = "#9a9a9d"
        CARD = "#17171a"

        top = tk.Frame(self, bg="#0b0b0c")
        top.pack(fill="x", padx=20, pady=(18, 8))
        tk.Label(top, text="EZI OCR Server", font=("Segoe UI", 15, "bold"),
                 fg=FG, bg="#0b0b0c").pack(side="left")

        card = tk.Frame(self, bg=CARD, highlightbackground="#2a2a2e", highlightthickness=1)
        card.pack(fill="x", padx=20, pady=8)

        self.status_dot = tk.Canvas(card, width=14, height=14, bg=CARD, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(16, 8), pady=16)
        self._dot = self.status_dot.create_oval(2, 2, 12, 12, fill="#6b6b6e", outline="")

        status_col = tk.Frame(card, bg=CARD)
        status_col.pack(side="left", pady=14)
        self.status_label = tk.Label(status_col, text="Stopped", font=("Segoe UI", 12, "bold"),
                                      fg=FG, bg=CARD, anchor="w")
        self.status_label.pack(anchor="w")
        self.url_label = tk.Label(status_col, text="", font=("Consolas", 10),
                                   fg=SUB, bg=CARD, anchor="w")
        self.url_label.pack(anchor="w")

        self.toggle_btn = tk.Button(
            self, text="Start Server", font=("Segoe UI", 13, "bold"),
            bg="#e8c44d", fg="#1a1400", activebackground="#f4d35e",
            relief="flat", height=2, command=self.toggle_server,
        )
        self.toggle_btn.pack(fill="x", padx=20, pady=(6, 12))

        tk.Label(self, text="Log", font=("Segoe UI", 9, "bold"), fg=SUB, bg="#0b0b0c",
                 anchor="w").pack(fill="x", padx=20)
        log_frame = tk.Frame(self, bg="#0b0b0c")
        log_frame.pack(fill="both", expand=True, padx=20, pady=(4, 18))
        self.log_text = tk.Text(log_frame, bg="#0f0f11", fg="#c8c8ca", insertbackground="#c8c8ca",
                                 font=("Consolas", 9), relief="flat", wrap="word")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set, state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # ------------------------------------------------------------ logging ----
    def log(self, line: str):
        def _do():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line if line.endswith("\n") else line + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.after(0, _do)

    def set_status(self, text: str, color: str, url: str = ""):
        self.status_label.configure(text=text)
        self.status_dot.itemconfig(self._dot, fill=color)
        self.url_label.configure(text=url)

    # ------------------------------------------------------------- control ----
    def toggle_server(self):
        if self.running:
            self.stop_server()
        elif not self.busy:
            self.start_server()

    def start_server(self):
        if not BUNDLED_SERVER_SCRIPT.exists():
            return
        self.toggle_btn.configure(state="disabled")
        threading.Thread(target=self._start_flow, daemon=True).start()

    def _start_flow(self):
        """Runs off the UI thread: provision the env if needed, then launch."""
        if not self._setup_is_current():
            self.busy = True
            self.after(0, lambda: self.toggle_btn.configure(text="Setting up…"))
            self.set_status("Setting up (first run only)…", "#e8c44d")
            ok = self._provision_environment()
            self.busy = False
            if not ok:
                self.after(0, lambda: self.toggle_btn.configure(text="Start Server", state="normal"))
                self.set_status("Setup failed", "#e05a4e")
                return
        self._launch_server_process()

    def _setup_is_current(self) -> bool:
        try:
            return SETUP_MARKER.read_text().strip() == SETUP_VERSION
        except Exception:
            return False

    # -------------------------------------------------- first-run setup ----
    def _run_step(self, cmd: list[str], cwd: Path | None = None) -> bool:
        """Run a subprocess, streaming its output into the log. Returns success."""
        self.log("$ " + " ".join(str(c) for c in cmd))
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except Exception as e:
            self.log(f"[!] {e}")
            return False
        for line in proc.stdout:
            self.log(line.rstrip())
        return proc.wait() == 0

    def _download(self, url: str, dest: Path) -> bool:
        self.log(f"Downloading {url}")
        try:
            def _progress(block_num, block_size, total_size):
                if total_size > 0:
                    pct = min(100, block_num * block_size * 100 // total_size)
                    if block_num % 200 == 0:
                        self.log(f"  … {pct}%")
            urllib.request.urlretrieve(url, dest, reporthook=_progress)
            return True
        except Exception as e:
            self.log(f"[!] Download failed: {e}")
            return False

    def _provision_environment(self) -> bool:
        """One-time: download a private Python, install pip + all deps."""
        ENV_DIR.mkdir(parents=True, exist_ok=True)
        self.log("=" * 50)
        self.log("First run: setting up the server environment.")
        self.log("This needs an internet connection and downloads several")
        self.log("GB of files (PyTorch, the AI model, etc). One-time only.")
        self.log("=" * 50)

        # 1. Download + extract embeddable Python
        if not python_exe_path().exists():
            zip_path = ENV_DIR / "python-embed.zip"
            if not self._download(EMBED_PYTHON_URL, zip_path):
                return False
            import zipfile
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(ENV_DIR)
            zip_path.unlink(missing_ok=True)

            # Enable site-packages (embeddable python ships with it disabled)
            pth_files = list(ENV_DIR.glob("python*._pth"))
            if pth_files:
                pth = pth_files[0]
                text = pth.read_text()
                text = text.replace("#import site", "import site")
                if "import site" not in text:
                    text += "\nimport site\n"
                pth.write_text(text)

        # 2. Install pip
        get_pip = ENV_DIR / "get-pip.py"
        if not self._download(GET_PIP_URL, get_pip):
            return False
        if not self._run_step([str(python_exe_path()), str(get_pip), "--no-warn-script-location"]):
            return False

        # 3. Install torch (CUDA build) then the rest of the requirements
        self.log("Installing PyTorch (CUDA build) — this is the biggest download…")
        if not self._run_step([
            str(python_exe_path()), "-m", "pip", "install",
            "--upgrade", "--force-reinstall",
            "torch", "torchvision", "--index-url", TORCH_INDEX_URL,
        ]):
            return False

        self.log("Installing remaining dependencies…")
        if not self._run_step([
            str(python_exe_path()), "-m", "pip", "install",
            "-r", str(BUNDLED_REQUIREMENTS),
        ]):
            return False

        SETUP_MARKER.write_text(SETUP_VERSION)
        self.log("[✓] Environment ready.")
        return True

    # -------------------------------------------------------- run server ----
    def _launch_server_process(self):
        self.log(f"Starting server on port {self.port}…")
        self.after(0, lambda: self.set_status("Starting…", "#e8c44d"))
        self.after(0, lambda: self.toggle_btn.configure(text="Starting…", state="disabled"))

        env = os.environ.copy()
        env["EZI_OCR_PORT"] = str(self.port)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        try:
            self.process = subprocess.Popen(
                [str(python_exe_path()), str(BUNDLED_SERVER_SCRIPT)],
                cwd=str(BUNDLED_SERVER_SCRIPT.parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace", env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except Exception as e:
            self.log(f"[!] Failed to start: {e}")
            self.after(0, lambda: self.set_status("Error", "#e05a4e"))
            self.after(0, lambda: self.toggle_btn.configure(text="Start Server", state="normal"))
            return

        self.running = True
        self.after(0, lambda: self.toggle_btn.configure(
            text="Stop Server", state="normal", bg="#3a3a3e", fg="#e7e7e8"))
        threading.Thread(target=self._pump_output, daemon=True).start()

    def _pump_output(self):
        seen_ready = False
        for line in self.process.stdout:
            self.log(line.rstrip())
            if not seen_ready and ("Uvicorn running" in line or "Application startup complete" in line):
                seen_ready = True
                url = f"http://{local_ip()}:{self.port}"
                self.after(0, lambda: self.set_status("Running", "#4caf6d", f"Enter this in the app: {url}"))
        self.after(0, self._on_process_ended)

    def _on_process_ended(self):
        was_running = self.running
        self.running = False
        self.process = None
        self.toggle_btn.configure(text="Start Server", state="normal", bg="#e8c44d", fg="#1a1400")
        if was_running:
            self.set_status("Stopped (server exited)", "#6b6b6e")
            self.log("[i] Server process ended.")

    def stop_server(self):
        if self.process and self.process.poll() is None:
            self.log("Stopping server…")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.running = False
        self.set_status("Stopped", "#6b6b6e")
        self.toggle_btn.configure(text="Start Server", bg="#e8c44d", fg="#1a1400")

    def _on_close(self):
        if self.running:
            if not messagebox.askyesno(APP_TITLE, "Server is running. Stop it and quit?"):
                return
            self.stop_server()
        self.destroy()


if __name__ == "__main__":
    LauncherApp().mainloop()
