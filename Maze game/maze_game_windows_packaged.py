"""
Maze Game - four control modes
==============================
Modes:
  1. Auto-solve  - BFS animates a red path from start to finish
  2. Hand gesture - MediaPipe Hands + trainable KNN classifier
  3. Voice command - speech_recognition listens for up/down/left/right
  4. Keyboard    - arrow keys

Difficulty:
  Easy   = 4x4
  Medium = 8x8
  Hard   = 16x16

Training data for hand gestures lives in `gesture_training_data.json`
(beside this file). Use the "Train Gestures" button to record samples
for each label (up / down / left / right / none) and retrain the model.

Dependencies:
  pip install opencv-python mediapipe scikit-learn numpy SpeechRecognition pyaudio
"""

import base64
import importlib
import json
import os
import queue
import random
import sys
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import messagebox, ttk

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies - the app degrades gracefully if any are missing.
# ---------------------------------------------------------------------------
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    mp = None
    HAS_MEDIAPIPE = False

HAS_VISION = HAS_CV2

try:
    from sklearn.neighbors import KNeighborsClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import speech_recognition as sr
    HAS_SPEECH = True
except ImportError:
    HAS_SPEECH = False


# ---------------------------------------------------------------------------
# Maze generation (recursive backtracker - guarantees a solvable maze)
# ---------------------------------------------------------------------------
class Maze:
    """A grid maze where each cell tracks which of its 4 walls are open."""

    DIRS = {
        "N": (-1, 0),
        "S": (1, 0),
        "E": (0, 1),
        "W": (0, -1),
    }
    OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

    def __init__(self, size: int):
        self.size = size
        self.cells = [
            [{"N": True, "S": True, "E": True, "W": True} for _ in range(size)]
            for _ in range(size)
        ]
        self._generate()
        self.start = (0, 0)
        self.end = (size - 1, size - 1)

    def _generate(self):
        """Recursive backtracker - carves a perfect maze."""
        visited = [[False] * self.size for _ in range(self.size)]
        stack = [(0, 0)]
        visited[0][0] = True

        while stack:
            r, c = stack[-1]
            neighbors = []
            for d, (dr, dc) in self.DIRS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and not visited[nr][nc]:
                    neighbors.append((d, nr, nc))

            if not neighbors:
                stack.pop()
                continue

            d, nr, nc = random.choice(neighbors)
            self.cells[r][c][d] = False
            self.cells[nr][nc][self.OPPOSITE[d]] = False
            visited[nr][nc] = True
            stack.append((nr, nc))

    def can_move(self, r: int, c: int, direction: str) -> bool:
        """True if movement from (r,c) in direction ('N'/'S'/'E'/'W') is open."""
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        return not self.cells[r][c][direction]

    def solve(self) -> list:
        """BFS from start to end. Returns list of (r,c) cells (inclusive)."""
        start, end = self.start, self.end
        prev = {start: None}
        q = deque([start])
        while q:
            cell = q.popleft()
            if cell == end:
                break
            r, c = cell
            for d, (dr, dc) in self.DIRS.items():
                if self.can_move(r, c, d):
                    nxt = (r + dr, c + dc)
                    if nxt not in prev:
                        prev[nxt] = cell
                        q.append(nxt)

        if end not in prev:
            return []
        path = []
        node = end
        while node is not None:
            path.append(node)
            node = prev[node]
        return list(reversed(path))


# ---------------------------------------------------------------------------
# Gesture classifier - trainable KNN over MediaPipe hand landmarks
# ---------------------------------------------------------------------------
def app_data_path(filename: str) -> str:
    """Use a writable folder when running from a packaged Windows executable."""
    if getattr(sys, "frozen", False):
        base = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "MazeGame")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def bundled_data_path(filename: str) -> str:
    """Find read-only files bundled by PyInstaller."""
    if getattr(sys, "frozen", False):
        bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        return os.path.join(bundle_dir, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


TRAINING_FILE = app_data_path("gesture_training_data.json")
DEFAULT_TRAINING_FILE = bundled_data_path("gesture_training_data.json")
LABELS = ["up", "down", "left", "right", "none"]


def landmarks_to_features(landmarks) -> np.ndarray:
    """
    Convert 21 hand landmarks (each x,y,z) into a 63-dim feature vector,
    normalized so position and scale don't matter - only finger shape does.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts.flatten()


def open_camera():
    """Open a webcam reliably on Windows, falling back to the default backend."""
    if not HAS_CV2:
        return None

    backends = [
        getattr(cv2, "CAP_DSHOW", 0),
        getattr(cv2, "CAP_MSMF", 0),
        0,
    ]
    seen = set()
    for backend in backends:
        if backend in seen:
            continue
        seen.add(backend)
        for index in range(3):
            cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ok, _frame = cap.read()
                if ok:
                    return cap
            cap.release()
    return None


def get_mediapipe_modules():
    """Return (hands_module, drawing_utils, error_message)."""
    if not HAS_MEDIAPIPE:
        return None, None, "mediapipe is not installed."

    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            return mp.solutions.hands, mp.solutions.drawing_utils, None
    except Exception:
        pass

    # Some installs do not expose mp.solutions but still have importable modules.
    candidates = [
        ("mediapipe.solutions.hands", "mediapipe.solutions.drawing_utils"),
        ("mediapipe.python.solutions.hands", "mediapipe.python.solutions.drawing_utils"),
    ]
    for hands_name, draw_name in candidates:
        try:
            hands_module = importlib.import_module(hands_name)
            drawing_utils = importlib.import_module(draw_name)
            return hands_module, drawing_utils, None
        except Exception:
            continue

    module_path = getattr(mp, "__file__", "unknown location")
    return (
        None,
        None,
        "Camera is working, but this mediapipe install has no Hands API. "
        f"Loaded mediapipe from: {module_path}",
    )


def frame_to_tk_data(frame, width=480) -> str | None:
    """Convert an OpenCV BGR frame into PNG data that Tk PhotoImage can display."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > width:
        scale = width / w
        frame = cv2.resize(frame, (width, max(1, int(h * scale))))
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def fallback_hand_features(frame):
    """
    Lightweight OpenCV fallback when MediaPipe Hands is unavailable.
    Returns (features, box, mask), where features is 63 dims or None.
    """
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, mask

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 1800:
        return None, None, mask

    x, y, w, h = cv2.boundingRect(contour)
    if w < 25 or h < 25:
        return None, None, mask

    roi = mask[y:y + h, x:x + w]
    roi = cv2.resize(roi, (7, 8), interpolation=cv2.INTER_AREA)
    roi_features = (roi.flatten().astype(np.float32) / 255.0)

    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten().astype(np.float32)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-6)
    hu = np.clip(hu / 10.0, -1.0, 1.0)

    features = np.concatenate([roi_features, hu]).astype(np.float32)
    return features, (x, y, w, h), mask


class GestureClassifier:
    """Wraps a KNN over saved landmark samples. Trains on demand."""

    def __init__(self):
        self.data = {label: [] for label in LABELS}
        self.model = None
        self.load()

    def load(self):
        source_file = TRAINING_FILE if os.path.exists(TRAINING_FILE) else DEFAULT_TRAINING_FILE
        if not os.path.exists(source_file):
            return
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for label in LABELS:
                self.data[label] = [np.array(x, dtype=np.float32) for x in raw.get(label, [])]
        except (json.JSONDecodeError, OSError):
            pass

    def save(self):
        serializable = {label: [v.tolist() for v in vecs] for label, vecs in self.data.items()}
        with open(TRAINING_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f)

    def add_sample(self, label: str, features: np.ndarray):
        if label not in self.data:
            self.data[label] = []
        self.data[label].append(features)

    def sample_count(self, label: str) -> int:
        return len(self.data.get(label, []))

    def total_samples(self) -> int:
        return sum(len(v) for v in self.data.values())

    def clear_label(self, label: str):
        self.data[label] = []

    def train(self) -> bool:
        if not HAS_SKLEARN:
            return False
        X, y = [], []
        for label, samples in self.data.items():
            for s in samples:
                X.append(s)
                y.append(label)
        if len(X) < 5 or len(set(y)) < 2:
            self.model = None
            return False
        k = min(3, len(X))
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(np.array(X), y)
        return True

    def predict(self, features: np.ndarray):
        """Returns (label, confidence). Confidence = fraction of k neighbors voting for label."""
        if self.model is None:
            return None, 0.0
        try:
            proba = self.model.predict_proba([features])[0]
            classes = self.model.classes_
            idx = int(np.argmax(proba))
            return classes[idx], float(proba[idx])
        except Exception:
            label = self.model.predict([features])[0]
            return label, 1.0


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
CELL_PX = 40
WALL_PX = 3
PLAYER_R = 12
DIR_FROM_KEY = {
    "Up": "N", "Down": "S", "Left": "W", "Right": "E",
    "w": "N", "s": "S", "a": "W", "d": "E",
    "up": "N", "down": "S", "left": "W", "right": "E",
}
VOICE_ALIASES = {
    "up": "up", "above": "up", "app": "up",
    "down": "down", "done": "down",
    "left": "left", "lift": "left",
    "right": "right", "write": "right", "ride": "right",
}


class StableGestureFilter:
    """Require a gesture to win several recent frames before firing a move."""

    def __init__(self, window=9, required=6, conf_min=0.65, cooldown=1.05):
        self.window = window
        self.required = required
        self.conf_min = conf_min
        self.cooldown = cooldown
        self.samples = deque(maxlen=window)
        self.last_fire_label = None
        self.last_fire_time = 0.0

    def update(self, label, confidence):
        clean_label = label if label and label != "none" and confidence >= self.conf_min else None
        self.samples.append((clean_label, confidence))
        counts = {}
        confidences = {}
        for sample_label, sample_conf in self.samples:
            if sample_label is None:
                continue
            counts[sample_label] = counts.get(sample_label, 0) + 1
            confidences.setdefault(sample_label, []).append(sample_conf)
        if not counts:
            return None, 0.0, 0

        stable_label = max(counts, key=counts.get)
        votes = counts[stable_label]
        stable_conf = sum(confidences[stable_label]) / len(confidences[stable_label])
        if votes < self.required:
            return None, stable_conf, votes

        now = time.time()
        if stable_label == self.last_fire_label and (now - self.last_fire_time) < self.cooldown:
            return None, stable_conf, votes

        self.last_fire_label = stable_label
        self.last_fire_time = now
        return stable_label, stable_conf, votes


class MazeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Maze Game")
        self.root.resizable(False, False)

        self.maze: Maze | None = None
        self.player = (0, 0)
        self.trail: list = []
        self.win = False
        self.canvas: tk.Canvas | None = None
        self.info_label: ttk.Label | None = None
        self.gesture_indicator: tk.Canvas | None = None
        self.gesture_indicator_text: ttk.Label | None = None
        self.gesture_window_pos = (40, 80)

        self.classifier = GestureClassifier()
        self.classifier.train()

        self.gesture_thread: threading.Thread | None = None
        self.gesture_stop = threading.Event()
        self.voice_thread: threading.Thread | None = None
        self.voice_stop = threading.Event()
        self.cmd_queue: "queue.Queue[str]" = queue.Queue()

        self._build_menu()
        self.root.bind("<Key>", self._on_key)

    def _build_menu(self):
        self.menu_frame = ttk.Frame(self.root, padding=20)
        self.menu_frame.grid(row=0, column=0)

        ttk.Label(self.menu_frame, text="Maze Game", font=("Helvetica", 20, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 16)
        )

        ttk.Label(self.menu_frame, text="Difficulty:").grid(row=1, column=0, sticky="e", padx=4)
        self.difficulty = tk.StringVar(value="Easy (4x4)")
        ttk.Combobox(
            self.menu_frame,
            textvariable=self.difficulty,
            values=["Easy (4x4)", "Medium (8x8)", "Hard (16x16)"],
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(self.menu_frame, text="Control mode:").grid(row=2, column=0, sticky="e", padx=4, pady=(8, 0))
        self.mode = tk.StringVar(value="Keyboard")
        ttk.Combobox(
            self.menu_frame,
            textvariable=self.mode,
            values=["Auto-solve", "Hand Gesture", "Voice", "Keyboard"],
            state="readonly",
            width=18,
        ).grid(row=2, column=1, sticky="w", padx=4, pady=(8, 0))

        ttk.Button(self.menu_frame, text="Start Maze", command=self.start_maze).grid(
            row=3, column=0, columnspan=2, pady=(16, 4), sticky="ew"
        )
        ttk.Button(self.menu_frame, text="Train Gestures", command=self.open_training).grid(
            row=4, column=0, columnspan=2, pady=4, sticky="ew"
        )
        ttk.Button(self.menu_frame, text="How to Play", command=self.show_how_to_play).grid(
            row=5, column=0, columnspan=2, pady=(4, 0), sticky="ew"
        )
        ttk.Button(self.menu_frame, text="Quit", command=self._quit).grid(
            row=6, column=0, columnspan=2, pady=(4, 0), sticky="ew"
        )

        status_lines = []
        if not HAS_CV2:
            status_lines.append("opencv-python not installed (no camera)")
        if not HAS_MEDIAPIPE:
            status_lines.append("mediapipe not installed (no hand detection)")
        if not HAS_SKLEARN:
            status_lines.append("scikit-learn not installed (no gesture model)")
        if not HAS_SPEECH:
            status_lines.append("SpeechRecognition not installed (no voice)")
        if status_lines:
            ttk.Label(
                self.menu_frame,
                text="\n".join(status_lines),
                foreground="#a33",
                font=("Helvetica", 9),
                justify="left",
            ).grid(row=7, column=0, columnspan=2, pady=(12, 0))

    def show_how_to_play(self):
        messagebox.showinfo(
            "How to Play",
            "Goal\n"
            "Move the blue player from the green start square to the gold finish square.\n\n"
            "Keyboard\n"
            "Use arrow keys or W A S D.\n\n"
            "Auto-solve\n"
            "Choose Auto-solve and start a maze to watch the red path solve it.\n\n"
            "Voice\n"
            "Say go up, go down, go left, or go right. The game moves when the command is heard.\n\n"
            "Hand Gesture\n"
            "First click Train Gestures. Pick a label, hold that gesture clearly in the camera, "
            "capture samples, then Train & save. In gesture mode, the camera window opens beside "
            "the maze and the green indicator shows when a gesture is recognized.\n\n"
            "Difficulty\n"
            "Easy is 4x4, Medium is 8x8, and Hard is 16x16.",
        )

    def _difficulty_size(self) -> int:
        return {"Easy (4x4)": 4, "Medium (8x8)": 8, "Hard (16x16)": 16}[self.difficulty.get()]

    def start_maze(self):
        size = self._difficulty_size()
        mode = self.mode.get()

        self._stop_background_threads()

        self.maze = Maze(size)
        self.player = self.maze.start
        self.trail = [self.maze.start]
        self.win = False

        if self.canvas:
            self.canvas.master.destroy()

        win = tk.Toplevel(self.root)
        win.title(f"Maze - {self.difficulty.get()} - {mode}")
        win.protocol("WM_DELETE_WINDOW", lambda w=win: self._close_maze(w))
        win.resizable(False, False)

        px = size * CELL_PX + WALL_PX * 2
        self.canvas = tk.Canvas(win, width=px, height=px, bg="white", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)

        self.info_label = ttk.Label(win, text=f"Mode: {mode}    (close window to return to menu)")
        self.info_label.pack(pady=(0, 8))
        indicator_frame = ttk.Frame(win)
        indicator_frame.pack(pady=(0, 8))
        self.gesture_indicator = tk.Canvas(indicator_frame, width=18, height=18, highlightthickness=0)
        self.gesture_indicator.grid(row=0, column=0, padx=(0, 6))
        self.gesture_indicator.create_oval(3, 3, 15, 15, fill="#999999", outline="")
        self.gesture_indicator_text = ttk.Label(indicator_frame, text="Gesture: idle")
        self.gesture_indicator_text.grid(row=0, column=1, sticky="w")

        self._draw_maze()
        win.update_idletasks()
        x = win.winfo_rootx() + win.winfo_width() + 20
        y = win.winfo_rooty()
        if x + 660 > win.winfo_screenwidth():
            x = max(0, win.winfo_rootx() - 660)
        self.gesture_window_pos = (x, max(0, y))

        if mode == "Auto-solve":
            self.root.after(400, self._auto_solve)
        elif mode == "Hand Gesture":
            self._start_gesture_mode()
        elif mode == "Voice":
            self._start_voice_mode()

        self._poll_commands()

    def _close_maze(self, win):
        self._stop_background_threads()
        win.destroy()
        self.canvas = None
        self.info_label = None
        self.gesture_indicator = None
        self.gesture_indicator_text = None

    def _set_status(self, text: str):
        if self.info_label:
            self.info_label.config(text=text)

    def _set_gesture_indicator(self, active: bool, text: str):
        if self.gesture_indicator:
            self.gesture_indicator.delete("all")
            fill = "#22a447" if active else "#999999"
            self.gesture_indicator.create_oval(3, 3, 15, 15, fill=fill, outline="")
        if self.gesture_indicator_text:
            self.gesture_indicator_text.config(text=text)

    def _show_gesture_frame(self, frame, positioned: bool) -> bool:
        cv2.imshow("Hand Gesture - Maze", frame)
        if not positioned:
            x, y = self.gesture_window_pos
            cv2.moveWindow("Hand Gesture - Maze", int(x), int(y))
        return True

    def _draw_maze(self):
        if not self.maze or not self.canvas:
            return
        c = self.canvas
        c.delete("all")
        n = self.maze.size
        ox = oy = WALL_PX

        sx, sy = self.maze.start
        ex, ey = self.maze.end
        c.create_rectangle(
            oy + sy * CELL_PX, ox + sx * CELL_PX,
            oy + (sy + 1) * CELL_PX, ox + (sx + 1) * CELL_PX,
            fill="#cdeccd", outline="",
        )
        c.create_rectangle(
            oy + ey * CELL_PX, ox + ex * CELL_PX,
            oy + (ey + 1) * CELL_PX, ox + (ex + 1) * CELL_PX,
            fill="#fde9a8", outline="",
        )

        for i in range(len(self.trail) - 1):
            (r1, col1), (r2, col2) = self.trail[i], self.trail[i + 1]
            x1 = oy + col1 * CELL_PX + CELL_PX // 2
            y1 = ox + r1 * CELL_PX + CELL_PX // 2
            x2 = oy + col2 * CELL_PX + CELL_PX // 2
            y2 = ox + r2 * CELL_PX + CELL_PX // 2
            c.create_line(x1, y1, x2, y2, fill="#e23b3b", width=4, capstyle="round")

        for r in range(n):
            for col in range(n):
                cell = self.maze.cells[r][col]
                x0 = oy + col * CELL_PX
                y0 = ox + r * CELL_PX
                x1 = x0 + CELL_PX
                y1 = y0 + CELL_PX
                if cell["N"]:
                    c.create_line(x0, y0, x1, y0, width=WALL_PX)
                if cell["S"]:
                    c.create_line(x0, y1, x1, y1, width=WALL_PX)
                if cell["W"]:
                    c.create_line(x0, y0, x0, y1, width=WALL_PX)
                if cell["E"]:
                    c.create_line(x1, y0, x1, y1, width=WALL_PX)

        pr, pc = self.player
        cx = oy + pc * CELL_PX + CELL_PX // 2
        cy = ox + pr * CELL_PX + CELL_PX // 2
        c.create_oval(
            cx - PLAYER_R, cy - PLAYER_R, cx + PLAYER_R, cy + PLAYER_R,
            fill="#1f77b4", outline="",
        )

        if self.win:
            px_center = (n * CELL_PX + 2 * WALL_PX) // 2
            c.create_text(
                px_center,
                px_center,
                text="SOLVED!", fill="#2a7a2a", font=("Helvetica", 28, "bold"),
            )

    def _try_move(self, direction: str):
        if not self.maze or self.win:
            return
        r, c = self.player
        if not self.maze.can_move(r, c, direction):
            self._set_status(f"Blocked: {direction}")
            return
        dr, dc = Maze.DIRS[direction]
        self.player = (r + dr, c + dc)
        self.trail.append(self.player)
        if self.player == self.maze.end:
            self.win = True
            self._set_status("Solved!")
        self._draw_maze()

    def _on_key(self, event):
        if not self.maze or self.win:
            return
        d = DIR_FROM_KEY.get(event.keysym)
        if d:
            self._try_move(d)

    def _poll_commands(self):
        """Pull commands posted from voice/gesture threads into the Tk thread."""
        try:
            while True:
                cmd = self.cmd_queue.get_nowait()
                if cmd.startswith("__STATUS__:"):
                    self._set_status(cmd.split(":", 1)[1])
                    continue
                if cmd.startswith("__GESTURE__:"):
                    _tag, label, conf = cmd.split(":", 2)
                    self._set_gesture_indicator(True, f"Gesture: {label} ({float(conf):.2f})")
                    self.root.after(450, lambda: self._set_gesture_indicator(False, "Gesture: listening"))
                    continue
                if cmd.startswith("__HAND__:"):
                    seen = cmd.split(":", 1)[1] == "1"
                    self._set_gesture_indicator(seen, "Gesture: hand seen" if seen else "Gesture: no hand")
                    continue
                if cmd == "__ERR_CAM__":
                    messagebox.showwarning("Camera", "Could not open your webcam. Close other camera apps and try again.")
                    self._set_status("Camera unavailable")
                    continue
                if cmd == "__ERR_MIC__":
                    messagebox.showwarning("Voice", "Could not open your microphone.")
                    self._set_status("Microphone unavailable")
                    continue
                if cmd.startswith("__ERR_VOICE__:"):
                    self._set_status(cmd.split(":", 1)[1])
                    continue

                d = DIR_FROM_KEY.get(cmd)
                if d:
                    self._set_status(f"Command: {cmd}")
                    self._try_move(d)
        except queue.Empty:
            pass
        if self.canvas:
            self.root.after(60, self._poll_commands)

    def _auto_solve(self):
        if not self.maze:
            return
        path = self.maze.solve()
        if not path:
            messagebox.showerror("Maze", "No solution found.")
            return
        self.trail = []
        self._step_path(path, 0)

    def _step_path(self, path, i):
        if not self.canvas or i >= len(path):
            if i >= len(path):
                self.win = True
                self._draw_maze()
            return
        self.player = path[i]
        self.trail.append(path[i])
        self._draw_maze()
        delay = max(40, 220 - self.maze.size * 8)
        self.root.after(delay, lambda: self._step_path(path, i + 1))

    def _start_gesture_mode(self):
        if not HAS_CV2 or not HAS_SKLEARN:
            messagebox.showwarning(
                "Hand Gesture",
                "Install opencv-python and scikit-learn to use hand gestures.",
            )
            return
        if self.classifier.model is None:
            messagebox.showwarning(
                "Hand Gesture",
                "No trained gesture model. Open 'Train Gestures' from the menu first.",
            )
            return
        self.gesture_stop.clear()
        self.gesture_thread = threading.Thread(target=self._gesture_loop, daemon=True)
        self.gesture_thread.start()
        self._set_status("Gesture mode listening")

    def _gesture_loop(self):
        """Camera + MediaPipe loop - pushes movement commands onto cmd_queue."""
        cap = None
        try:
            cap = open_camera()
            if cap is None:
                self.cmd_queue.put("__ERR_CAM__")
                return
            last_cmd = None
            last_cmd_time = 0.0
            last_hand_report = 0.0
            cooldown = 0.7
            conf_min = 0.6
            stabilizer = StableGestureFilter(window=9, required=6, conf_min=0.65, cooldown=1.05)
            window_positioned = False
            mp_hands, mp_draw, mp_error = get_mediapipe_modules()
            if mp_error:
                self.cmd_queue.put("__STATUS__:Using OpenCV fallback hand detector")
                while not self.gesture_stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    feats, box, _mask = fallback_hand_features(frame)
                    label, conf = None, 0.0
                    if feats is not None:
                        label, conf = self.classifier.predict(feats)
                        x, y, w, h = box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        self.cmd_queue.put("__HAND__:1")
                    else:
                        self.cmd_queue.put("__HAND__:0")

                    stable_label, stable_conf, votes = stabilizer.update(label, conf)
                    if stable_label:
                        self.cmd_queue.put(f"__GESTURE__:{stable_label}:{stable_conf}")
                        self.cmd_queue.put(stable_label)

                    txt = f"raw: {label or '-'} ({conf:.2f})"
                    cv2.putText(frame, txt, (12, 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, f"stable votes: {votes}/6",
                                (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    window_positioned = self._show_gesture_frame(frame, window_positioned)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                return

            with mp_hands.Hands(
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as hands:
                while not self.gesture_stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    res = hands.process(rgb)
                    rgb.flags.writeable = True

                    label, conf = None, 0.0
                    hand_seen = False
                    if res.multi_hand_landmarks:
                        hand_seen = True
                        lm = res.multi_hand_landmarks[0]
                        feats = landmarks_to_features(lm.landmark)
                        label, conf = self.classifier.predict(feats)
                        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                    now = time.time()
                    if now - last_hand_report > 0.25:
                        self.cmd_queue.put("__HAND__:1" if hand_seen else "__HAND__:0")
                        last_hand_report = now

                    stable_label, stable_conf, votes = stabilizer.update(label, conf)
                    if stable_label:
                        self.cmd_queue.put(f"__GESTURE__:{stable_label}:{stable_conf}")
                        self.cmd_queue.put(stable_label)

                    txt = f"{label or '-'} ({conf:.2f})" if label else "no hand"
                    cv2.putText(frame, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, f"stable votes: {votes}/6    Press Q to stop",
                                (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    window_positioned = self._show_gesture_frame(frame, window_positioned)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except Exception as exc:
            self.cmd_queue.put(f"__STATUS__:Gesture error: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if HAS_CV2:
                try:
                    cv2.destroyWindow("Hand Gesture - Maze")
                except cv2.error:
                    pass

    def _start_voice_mode(self):
        if not HAS_SPEECH:
            messagebox.showwarning(
                "Voice",
                "Install SpeechRecognition and pyaudio to use voice control.",
            )
            return
        self.voice_stop.clear()
        self.voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
        self.voice_thread.start()
        self._set_status("Voice mode listening")

    def _voice_loop(self):
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.5
        try:
            mic = sr.Microphone()
        except OSError:
            self.cmd_queue.put("__ERR_MIC__")
            return

        try:
            with mic as source:
                self.cmd_queue.put("__STATUS__:Calibrating microphone...")
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
            self.cmd_queue.put("__STATUS__:Say up, down, left, or right")
        except Exception as exc:
            self.cmd_queue.put(f"__ERR_VOICE__:Microphone setup failed: {exc}")
            return

        while not self.voice_stop.is_set():
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=4, phrase_time_limit=2)
                text = recognizer.recognize_google(audio, language="en-US").lower()
                self.cmd_queue.put(f"__STATUS__:Heard: {text}")
                normalized = text.replace(",", " ").replace(".", " ")
                for phrase, cmd in {
                    "go up": "up",
                    "go down": "down",
                    "go left": "left",
                    "go right": "right",
                }.items():
                    if phrase in normalized:
                        self.cmd_queue.put(cmd)
                        break
                else:
                    for word in normalized.split():
                        cmd = VOICE_ALIASES.get(word)
                        if cmd:
                            self.cmd_queue.put(cmd)
                            break
            except sr.WaitTimeoutError:
                self.cmd_queue.put("__STATUS__:Listening...")
            except sr.UnknownValueError:
                self.cmd_queue.put("__STATUS__:Could not understand that")
            except sr.RequestError as exc:
                self.cmd_queue.put(f"__ERR_VOICE__:Speech service error: {exc}")
                time.sleep(1.0)
            except Exception as exc:
                self.cmd_queue.put(f"__ERR_VOICE__:Voice error: {exc}")
                time.sleep(0.5)

    def _stop_background_threads(self):
        self.gesture_stop.set()
        self.voice_stop.set()

    def _quit(self):
        self._stop_background_threads()
        self.root.after(150, self.root.destroy)

    def open_training(self):
        if not HAS_CV2:
            messagebox.showwarning(
                "Training",
                "opencv-python is required for the camera preview.",
            )
            return
        TrainingWindow(self.root, self.classifier)


# ---------------------------------------------------------------------------
# Training window - separate class for clarity
# ---------------------------------------------------------------------------
class TrainingWindow:
    """
    Live camera preview. Pick a label, hold your gesture, click 'Capture sample'
    to add one frame to that label's training set. Click 'Train & save' to
    retrain the KNN and persist the data to disk.
    """

    def __init__(self, parent, classifier: GestureClassifier):
        self.classifier = classifier
        self.win = tk.Toplevel(parent)
        self.win.title("Train Gestures")
        self.win.protocol("WM_DELETE_WINDOW", self.close)

        self.label_var = tk.StringVar(value="up")
        ttk.Label(self.win, text="Label:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
        ttk.Combobox(
            self.win, textvariable=self.label_var,
            values=LABELS, state="readonly", width=14,
        ).grid(row=0, column=1, padx=6, pady=6, sticky="w")

        self.capture_button = ttk.Button(self.win, text="Capture sample", command=self.capture, state="disabled")
        self.capture_button.grid(
            row=1, column=0, columnspan=2, padx=6, pady=4, sticky="ew",
        )
        self.burst_button = ttk.Button(
            self.win,
            text="Capture 10 over 3s",
            command=lambda: self.capture_burst(10, 3.0),
            state="disabled",
        )
        self.burst_button.grid(
            row=2, column=0, columnspan=2, padx=6, pady=4, sticky="ew",
        )
        ttk.Button(self.win, text="Clear this label", command=self.clear_label).grid(
            row=3, column=0, columnspan=2, padx=6, pady=4, sticky="ew",
        )
        ttk.Button(self.win, text="Train & save", command=self.train_and_save).grid(
            row=4, column=0, columnspan=2, padx=6, pady=(10, 4), sticky="ew",
        )

        self.preview_label = ttk.Label(self.win, text="Starting camera...", anchor="center")
        self.preview_label.grid(row=5, column=0, columnspan=2, padx=6, pady=(8, 4))
        self.preview_image = None
        self.last_status_text = ""

        detect_frame = ttk.Frame(self.win)
        detect_frame.grid(row=6, column=0, columnspan=2, padx=6, pady=(0, 6))
        self.detect_indicator = tk.Canvas(detect_frame, width=18, height=18, highlightthickness=0)
        self.detect_indicator.grid(row=0, column=0, padx=(0, 6))
        self.detect_indicator.create_oval(3, 3, 15, 15, fill="#999999", outline="")
        self.detect_label = ttk.Label(detect_frame, text="Waiting for hand")
        self.detect_label.grid(row=0, column=1, sticky="w")

        self.counts_label = ttk.Label(self.win, text="", justify="left", font=("Courier", 10))
        self.counts_label.grid(row=7, column=0, columnspan=2, padx=6, pady=8)
        self.status_label = ttk.Label(self.win, text="Opening camera...", foreground="#555")
        self.status_label.grid(row=8, column=0, columnspan=2, padx=6, pady=(0, 6))
        self._refresh_counts()

        ttk.Label(
            self.win,
            text=("Hold your gesture in front of the camera and click Capture.\n"
                  "Aim for 15-30 samples per label, varying angle and distance.\n"
                  "The capture buttons unlock when a hand is detected."),
            foreground="#555", font=("Helvetica", 9), justify="left",
        ).grid(row=9, column=0, columnspan=2, padx=6, pady=(4, 8))

        self.stop = threading.Event()
        self.latest_features: np.ndarray | None = None
        self.latest_preview_data: str | None = None
        self.hand_seen_history = deque(maxlen=5)
        self.latest_lock = threading.Lock()
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()
        self._update_preview()

    def _refresh_counts(self):
        lines = []
        for label in LABELS:
            lines.append(f"{label:>6}: {self.classifier.sample_count(label):>4} samples")
        lines.append(f"{'total':>6}: {self.classifier.total_samples():>4}")
        self.counts_label.config(text="\n".join(lines))

    def _set_status(self, text: str):
        if text == self.last_status_text:
            return
        self.last_status_text = text
        if self.win.winfo_exists():
            def update():
                if self.win.winfo_exists():
                    self.status_label.config(text=text)

            self.win.after(0, update)

    def _set_detection_ui(self, hand_detected: bool):
        self.detect_indicator.delete("all")
        fill = "#22a447" if hand_detected else "#999999"
        self.detect_indicator.create_oval(3, 3, 15, 15, fill=fill, outline="")
        self.detect_label.config(text="Hand detected - ready to capture" if hand_detected else "Waiting for hand")
        state = "normal" if hand_detected else "disabled"
        self.capture_button.config(state=state)
        self.burst_button.config(state=state)

    def _update_preview(self):
        if self.stop.is_set() or not self.win.winfo_exists():
            return
        with self.latest_lock:
            preview_data = self.latest_preview_data
            raw_hand_detected = self.latest_features is not None
        self.hand_seen_history.append(raw_hand_detected)
        hand_detected = sum(1 for seen in self.hand_seen_history if seen) >= 3
        if preview_data:
            try:
                self.preview_image = tk.PhotoImage(data=preview_data, format="png")
                self.preview_label.config(image=self.preview_image, text="")
            except tk.TclError:
                try:
                    self.preview_image = tk.PhotoImage(data=preview_data)
                    self.preview_label.config(image=self.preview_image, text="")
                except tk.TclError:
                    self.preview_label.config(text="Camera preview unavailable")
        self._set_detection_ui(hand_detected)
        self.win.after(80, self._update_preview)

    def capture(self):
        with self.latest_lock:
            feats = None if self.latest_features is None else self.latest_features.copy()
        if feats is None:
            messagebox.showinfo(
                "Training",
                "No hand detected yet. Keep your whole hand in the camera view, palm facing the camera, with good light.",
            )
            return
        self.classifier.add_sample(self.label_var.get(), feats)
        self._refresh_counts()

    def capture_burst(self, n: int, seconds: float):
        """Capture n samples evenly spaced over seconds."""
        label = self.label_var.get()
        interval_ms = max(50, int(seconds * 1000 / n))
        captured = [0]

        def step():
            if captured[0] >= n or self.stop.is_set():
                self._refresh_counts()
                return
            with self.latest_lock:
                feats = None if self.latest_features is None else self.latest_features.copy()
            if feats is not None:
                self.classifier.add_sample(label, feats)
                captured[0] += 1
                self._refresh_counts()
            self.win.after(interval_ms, step)

        step()

    def clear_label(self):
        label = self.label_var.get()
        if messagebox.askyesno("Clear", f"Delete all samples for '{label}'?"):
            self.classifier.clear_label(label)
            self._refresh_counts()

    def train_and_save(self):
        ok = self.classifier.train()
        self.classifier.save()
        if ok:
            messagebox.showinfo("Training", "Model trained and data saved.")
        else:
            messagebox.showwarning(
                "Training",
                "Need at least 5 samples across at least 2 labels to train. Data was saved.",
            )

    def _camera_loop(self):
        cap = None
        try:
            cap = open_camera()
            if cap is None:
                self._set_status("Could not open webcam. Close other camera apps and try again.")
                return
            mp_hands, mp_draw, mp_error = get_mediapipe_modules()
            if mp_error:
                self._set_status("Using OpenCV fallback hand detector - capture when the dot turns green")
                while not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        self._set_status("Camera opened, but no frames are arriving.")
                        continue
                    frame = cv2.flip(frame, 1)
                    feats, box, _mask = fallback_hand_features(frame)
                    if feats is not None:
                        x, y, w, h = box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        with self.latest_lock:
                            self.latest_features = feats
                        self._set_status("Fallback hand detected - capture when ready")
                    else:
                        with self.latest_lock:
                            self.latest_features = None
                        self._set_status("Fallback detector waiting for hand")

                    cv2.putText(frame, f"Label: {self.label_var.get()}",
                                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, "OpenCV fallback detector",
                                (12, frame.shape[0] - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    preview_data = frame_to_tk_data(frame)
                    with self.latest_lock:
                        self.latest_preview_data = preview_data
                return

            with mp_hands.Hands(
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as hands:
                while not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        self._set_status("Camera opened, but no frames are arriving.")
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    res = hands.process(rgb)
                    rgb.flags.writeable = True

                    if res.multi_hand_landmarks:
                        lm = res.multi_hand_landmarks[0]
                        feats = landmarks_to_features(lm.landmark)
                        with self.latest_lock:
                            self.latest_features = feats
                        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                        self._set_status("Hand detected - capture when ready")
                    else:
                        with self.latest_lock:
                            self.latest_features = None
                        self._set_status("No hand detected")

                    cv2.putText(frame, f"Label: {self.label_var.get()}",
                                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, "Show your full hand clearly",
                                (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    preview_data = frame_to_tk_data(frame)
                    with self.latest_lock:
                        self.latest_preview_data = preview_data
        except Exception as exc:
            self._set_status(f"Camera/MediaPipe error: {exc}")
        finally:
            if cap is not None:
                cap.release()

    def close(self):
        self.stop.set()
        self.win.destroy()


# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    MazeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
