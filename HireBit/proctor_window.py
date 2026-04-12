import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import mediapipe as mp
import time
import pyperclip
import ctypes
import numpy as np
import argparse
import sys
import threading

# Import project database utility
try:
    from database import save_flag_payload
except ImportError:
    # Minimal mock if database.py is missing or failing
    def save_flag_payload(payload):
        print(f"DATABASE_OFFLINE: {payload}")
        return False

# Windows API for window tracking
class WindowTracker:
    @staticmethod
    def get_active_window_text():
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
            return buff.value
        except Exception:
            return "Unknown"

class ProctorApp:
    def __init__(self, window, session_id="dev_session"):
        self.window = window
        self.session_id = session_id
        
        # UI CONSTANTS
        self.BG_COLOR = "#0c0c0c"
        self.HUD_ACCENT = "#00d9ff"
        self.HUD_WARNING = "#ff2d55"
        
        # UI SETUP
        self.window.overrideredirect(True)
        self.window.attributes('-topmost', True)
        self.window.attributes('-alpha', 0.78)
        self.window.configure(bg=self.BG_COLOR)
        
        self.w, self.h = 420, 200
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        x = screen_w - self.w - 20
        y = screen_h - self.h - 60
        self.window.geometry(f"{self.w}x{self.h}+{x}+{y}")

        self.window.bind("<Button-1>", self.start_move)
        self.window.bind("<B1-Motion>", self.do_move)

        # Detection Modules
        self.face_mesh = None
        self.using_mediapipe = False
        try:
            if hasattr(mp, 'solutions'):
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5)
                self.using_mediapipe = True
        except: pass

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # State & Analytics
        self.face_count = 0
        self.eyes_detected = 0
        self.active_window = "HireBit Badge"
        self.clipboard_content = ""
        try: self.clipboard_content = pyperclip.paste()
        except: pass
        
        self.is_running = True
        self.cheating_pulse = 0
        self.last_sync_time = time.time()

        self.setup_ui()

        # Video source
        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        self.update_video()
        self.check_os_events()
        self.heartbeat_loop()

    def start_move(self, event):
        self.x = event.x
        self.y = event.y

    def do_move(self, event):
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.window.winfo_x() + deltax
        y = self.window.winfo_y() + deltay
        self.window.geometry(f"+{x}+{y}")

    def setup_ui(self):
        header = tk.Frame(self.window, bg="#151515", height=20)
        header.pack(fill=tk.X)
        self.window.bind("<Button-1>", self.start_move, add="+")
        
        tk.Label(header, text=f"PROCTOR // {self.session_id}", font=("Verdana", 7, "bold"), bg="#151515", fg=self.HUD_ACCENT).pack(side=tk.LEFT, padx=8)
        tk.Button(header, text="×", font=("Arial", 10, "bold"), bg="#151515", fg="#555555", bd=0, command=self.on_closing).pack(side=tk.RIGHT, padx=5)

        self.content = tk.Frame(self.window, bg=self.BG_COLOR)
        self.content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.vid_box = tk.Frame(self.content, bg=self.HUD_ACCENT, bd=1)
        self.vid_box.pack(side=tk.LEFT)
        self.canvas = tk.Canvas(self.vid_box, width=200, height=150, bg="black", highlightthickness=0)
        self.canvas.pack()

        self.meter_panel = tk.Frame(self.content, bg=self.BG_COLOR)
        self.meter_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.monitors = {}
        self.add_badge_meter("OBJ_DET", "Face")
        self.add_badge_meter("EYE_TRK", "Eye")
        self.add_badge_meter("FOCUS", "Window")
        self.add_badge_meter("CLOUD", "Sync")
        
        self.status_bar = tk.Label(self.meter_panel, text="LIVE // SECURE", font=("Arial", 8, "bold"), bg="#1a1a1a", fg=self.HUD_ACCENT, pady=5)
        self.status_bar.pack(fill=tk.X, pady=(5, 0))

    def add_badge_meter(self, label, key):
        f = tk.Frame(self.meter_panel, bg=self.BG_COLOR)
        f.pack(fill=tk.X, pady=1)
        tk.Label(f, text=label, font=("Arial", 6, "bold"), bg=self.BG_COLOR, fg="#444444").pack(side=tk.LEFT)
        indicator = tk.Label(f, text="●", font=("Arial", 9), bg=self.BG_COLOR, fg=self.HUD_ACCENT)
        indicator.pack(side=tk.RIGHT)
        self.monitors[key] = indicator

    def sync_to_cloud(self, event_type, details, score_impact=10):
        """Pushes a proctoring flag to Supabase using established database utility."""
        payload = {
            "session_id": self.session_id,
            "fraud_score": score_impact,
            "risk_level": "PROCTOR_EVENT",
            "flag": {
                "triggered": True,
                "reasoning_note": f"Proctor HUD: {event_type} - {details}"
            },
            "signals": {
                "source": "standalone_hud",
                "event": event_type,
                "details": details,
                "timestamp": time.time()
            }
        }
        # Run in background to avoid UI stutter
        threading.Thread(target=save_flag_payload, args=(payload,), daemon=True).start()
        self.monitors["Sync"].config(fg=self.HUD_ACCENT)
        self.window.after(1000, lambda: self.monitors["Sync"].config(fg="#444444"))

    def check_os_events(self):
        focus = WindowTracker.get_active_window_text()
        if focus and focus != self.active_window:
            if "HireBit" not in focus and focus != "":
                self.sync_to_cloud("WINDOW_FOCUS_LOST", focus[:30])
            self.active_window = focus
            
        try:
            clip = pyperclip.paste()
            if clip != self.clipboard_content and clip.strip() != "":
                self.sync_to_cloud("CLIPBOARD_CHANGE", "Text copied to clipboard", score_impact=15)
                self.clipboard_content = clip
        except: pass

        if self.is_running:
            self.window.after(1000, self.check_os_events)

    def heartbeat_loop(self):
        # Every 60 seconds, send a 'heartbeat' payload to show the proctor is active
        if self.is_running:
            self.sync_to_cloud("PROCTOR_HEARTBEAT", "Active and Monitoring", score_impact=0)
            self.window.after(60000, self.heartbeat_loop)

    def update_video(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (200, 150))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            self.face_count = 0
            self.eyes_detected = 0
            
            if self.using_mediapipe:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.face_mesh.process(rgb)
                if res.multi_face_landmarks:
                    self.face_count = len(res.multi_face_landmarks)
                    self.eyes_detected = 2
            
            if self.face_count == 0:
                faces = self.face_cascade.detectMultiScale(gray, 1.2, 4)
                self.face_count = len(faces)
                for (x,y,w,h) in faces:
                    roi = gray[y:y+h, x:x+w]
                    self.eyes_detected += len(self.eye_cascade.detectMultiScale(roi))

            # Update Mini Meters
            is_warn = False
            self.monitors["Face"].config(fg=self.HUD_ACCENT if self.face_count == 1 else self.HUD_WARNING)
            self.monitors["Eye"].config(fg=self.HUD_ACCENT if self.eyes_detected >= 1 else self.HUD_WARNING)
            self.monitors["Window"].config(fg=self.HUD_ACCENT if "HireBit" in self.active_window else self.HUD_WARNING)
            
            if self.face_count != 1 or self.eyes_detected < 1 or "HireBit" not in self.active_window:
                is_warn = True

            if is_warn:
                self.status_bar.config(text="⚠ BREACH DETECTED", fg=self.HUD_WARNING)
                self.vid_box.config(bg=self.HUD_WARNING)
                self.cheating_pulse = (self.cheating_pulse + 1) % 6
                self.window.configure(bg="#200000" if self.cheating_pulse > 3 else self.BG_COLOR)
            else:
                self.status_bar.config(text="LIVE // SECURE", fg=self.HUD_ACCENT)
                self.vid_box.config(bg=self.HUD_ACCENT)
                self.window.configure(bg=self.BG_COLOR)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk

        if self.is_running:
            self.window.after(30, self.update_video)

    def on_closing(self):
        self.is_running = False
        if self.vid.isOpened(): self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, default=None, help="Session ID to link data")
    args = parser.parse_args()
    
    root = tk.Tk()
    
    # If no session provided, ask via simple dialog
    session_id = args.session
    if not session_id:
        from tkinter import simpledialog
        root.withdraw() # Hide root temporarily
        session_id = simpledialog.askstring("Session Required", "Enter Candidate Session ID:", initialvalue="dev_session_1")
        if not session_id: # User cancelled
            sys.exit()
        root.deiconify() # Show root back
    
    app = ProctorApp(root, session_id=session_id)
    root.mainloop()
