"""
EZI OCR Server Launcher — Trilo Automation

A tiny one-button front end for backend/ocr_server.py. It doesn't run the
model itself — it launches the same Python environment you already use
(the one with torch/transformers/etc. already installed and working) as a
background process, and shows its status/log output in a clean window
instead of a raw terminal/IDE console.

First run: click the gear icon once to point it at:
  - Python executable  (the interpreter that already runs ocr_server.py today)
  - ocr_server.py      (the script itself, inside backend/)
Both are remembered afterwards in a small config file next to this program.
"""

import json
import socket
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

APP_TITLE = "EZI OCR Server"
CONFIG_FILE = Path(sys.argv[0]).resolve().parent / "server_launcher_config.json"

DEFAULT_CONFIG = {
    "python_path": "",
    "script_path": "",
    "port": 8000,
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            cfg = dict(DEFAULT_CONFIG)
            cfg.update(data)
            return cfg
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


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


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, cfg: dict, on_save):
        super().__init__(parent)
        self.title("Server Settings")
        self.resizable(False, False)
        self.cfg = cfg
        self.on_save = on_save
        self.transient(parent)
        self.grab_set()

        pad = {"padx": 12, "pady": 6}

        tk.Label(self, text="Python executable (the one with torch/transformers installed):",
                 anchor="w").grid(row=0, column=0, columnspan=2, sticky="w", **pad)
        self.python_var = tk.StringVar(value=cfg["python_path"])
        tk.Entry(self, textvariable=self.python_var, width=52).grid(row=1, column=0, **pad)
        tk.Button(self, text="Browse…", command=self.browse_python).grid(row=1, column=1, **pad)

        tk.Label(self, text="ocr_server.py location:", anchor="w").grid(
            row=2, column=0, columnspan=2, sticky="w", **pad)
        self.script_var = tk.StringVar(value=cfg["script_path"])
        tk.Entry(self, textvariable=self.script_var, width=52).grid(row=3, column=0, **pad)
        tk.Button(self, text="Browse…", command=self.browse_script).grid(row=3, column=1, **pad)

        tk.Label(self, text="Port:", anchor="w").grid(row=4, column=0, sticky="w", **pad)
        self.port_var = tk.StringVar(value=str(cfg["port"]))
        tk.Entry(self, textvariable=self.port_var, width=10).grid(row=4, column=1, sticky="w", **pad)

        btns = tk.Frame(self)
        btns.grid(row=5, column=0, columnspan=2, pady=(10, 12))
        tk.Button(btns, text="Save", width=10, command=self.save).pack(side="left", padx=6)
        tk.Button(btns, text="Cancel", width=10, command=self.destroy).pack(side="left", padx=6)

    def browse_python(self):
        path = filedialog.askopenfilename(
            title="Select Python executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
        )
        if path:
            self.python_var.set(path)

    def browse_script(self):
        path = filedialog.askopenfilename(
            title="Select ocr_server.py",
            filetypes=[("Python file", "*.py"), ("All files", "*.*")],
        )
        if path:
            self.script_var.set(path)

    def save(self):
        try:
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror(APP_TITLE, "Port must be a number.")
            return
        if not self.python_var.get() or not self.script_var.get():
            messagebox.showerror(APP_TITLE, "Both fields are required.")
            return
        self.cfg["python_path"] = self.python_var.get()
        self.cfg["script_path"] = self.script_var.get()
        self.cfg["port"] = port
        save_config(self.cfg)
        self.on_save(self.cfg)
        self.destroy()


class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("560x420")
        self.minsize(480, 360)
        self.configure(bg="#0b0b0c")

        self.cfg = load_config()
        self.process: subprocess.Popen | None = None
        self.running = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if not self.cfg["python_path"] or not self.cfg["script_path"]:
            self.after(300, self.open_settings)

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self):
        FG = "#e7e7e8"
        SUB = "#9a9a9d"
        CARD = "#17171a"

        top = tk.Frame(self, bg="#0b0b0c")
        top.pack(fill="x", padx=20, pady=(18, 8))
        tk.Label(top, text="EZI OCR Server", font=("Segoe UI", 15, "bold"),
                 fg=FG, bg="#0b0b0c").pack(side="left")
        tk.Button(top, text="⚙", font=("Segoe UI", 12), width=3,
                  command=self.open_settings).pack(side="right")

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

    def open_settings(self):
        SettingsDialog(self, self.cfg, on_save=lambda cfg: setattr(self, "cfg", cfg))

    # ------------------------------------------------------------ logging ----
    def log(self, line: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line if line.endswith("\n") else line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def set_status(self, text: str, color: str, url: str = ""):
        self.status_label.configure(text=text)
        self.status_dot.itemconfig(self._dot, fill=color)
        self.url_label.configure(text=url)

    # ------------------------------------------------------------- control ----
    def toggle_server(self):
        if self.running:
            self.stop_server()
        else:
            self.start_server()

    def start_server(self):
        if not self.cfg["python_path"] or not self.cfg["script_path"]:
            messagebox.showwarning(APP_TITLE, "Set the Python path and ocr_server.py location first (⚙).")
            self.open_settings()
            return

        script = Path(self.cfg["script_path"])
        if not script.exists():
            messagebox.showerror(APP_TITLE, f"ocr_server.py not found at:\n{script}")
            return

        self.log(f"Starting: {self.cfg['python_path']} {script.name}")
        self.set_status("Starting…", "#e8c44d")
        self.toggle_btn.configure(text="Starting…", state="disabled")

        try:
            self.process = subprocess.Popen(
                [self.cfg["python_path"], str(script)],
                cwd=str(script.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except Exception as e:
            self.log(f"[!] Failed to start: {e}")
            self.set_status("Error", "#e05a4e")
            self.toggle_btn.configure(text="Start Server", state="normal")
            return

        self.running = True
        self.toggle_btn.configure(text="Stop Server", state="normal", bg="#3a3a3e", fg="#e7e7e8")
        threading.Thread(target=self._pump_output, daemon=True).start()

    def _pump_output(self):
        port = self.cfg["port"]
        seen_ready = False
        for line in self.process.stdout:
            self.log(line.rstrip())
            if not seen_ready and ("Uvicorn running" in line or "Application startup complete" in line):
                seen_ready = True
                url = f"http://{local_ip()}:{port}"
                self.after(0, lambda: self.set_status("Running", "#4caf6d", f"Enter this in the app: {url}"))
        # process ended (crashed, or stopped)
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
