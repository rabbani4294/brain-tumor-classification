"""
╔══════════════════════════════════════════════════════════════════════════╗
║         AI Brain Tumor Detection System  —  Desktop Application         ║
║         TensorFlow + Tkinter + OpenCV  |  Production Grade              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, threading, datetime, collections, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk, ImageDraw

# ── heavy imports (graceful fallback) ─────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════
MODEL_PATH        = r"D:\Brain_tumor_project\models\brain_tumor_model.h5"
CLASSES           = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMG_SIZE          = (224, 224)
VIDEO_FRAME_SKIP  = 10
CONFIDENCE_THRESH = 70.0
HISTORY_FILE      = "prediction_history.txt"

# palette
C = dict(
    bg        = "#0D1117",
    panel     = "#161B22",
    card      = "#1C2128",
    input     = "#21262D",
    border    = "#30363D",
    blue      = "#58A6FF",
    cyan      = "#39D0D8",
    green     = "#3FB950",
    red       = "#F85149",
    yellow    = "#E3B341",
    purple    = "#BC8CFF",
    text      = "#E6EDF3",
    muted     = "#8B949E",
    dim       = "#484F58",
)

FT = dict(
    title   = ("Consolas", 18, "bold"),
    head    = ("Consolas", 11, "bold"),
    body    = ("Consolas", 10),
    small   = ("Consolas",  9),
    mono    = ("Courier New", 9),
    big     = ("Consolas", 26, "bold"),
    mid     = ("Consolas", 13, "bold"),
    tab     = ("Consolas", 11, "bold"),
)

# ══════════════════════════════════════════════════════════════════════════
#  BACKEND FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def load_model_from_disk(log_fn=None):
    def _l(m):
        if log_fn: log_fn(m)
    if not TF_AVAILABLE:
        _l("[ERROR] TensorFlow not installed."); return None
    if not os.path.exists(MODEL_PATH):
        _l(f"[ERROR] Model not found: {MODEL_PATH}"); return None
    try:
        _l("[INFO]  Loading model …")
        m = tf.keras.models.load_model(MODEL_PATH)
        _l(f"[OK]    Model ready  ➜  {MODEL_PATH}")
        return m
    except Exception as e:
        _l(f"[ERROR] {e}"); return None


def preprocess_pil(pil_image):
    img = pil_image.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
    return preprocess_input(arr)


def run_image_inference(model, pil_image):
    probs = model.predict(preprocess_pil(pil_image), verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])*100, [float(p)*100 for p in probs]


def run_video_inference(model, path, prog_cb=None, frame_cb=None, done_cb=None, log_fn=None):
    def _l(m):
        if log_fn: log_fn(m)

    if not CV2_AVAILABLE:
        _l("[ERROR] OpenCV not installed.")
        if done_cb: done_cb(None, 0, {})
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        _l(f"[ERROR] Cannot open video: {path}")
        if done_cb: done_cb(None, 0, {})
        return

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    votes  = []
    acc    = collections.defaultdict(list)
    fi     = 0
    _l(f"[INFO]  {total} frames  |  sampling every {VIDEO_FRAME_SKIP}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        if fi % VIDEO_FRAME_SKIP == 0:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb)
            lbl, conf, _ = run_image_inference(model, pil)
            votes.append(lbl); acc[lbl].append(conf)
            if frame_cb: frame_cb(pil, f"{lbl}  {conf:.1f}%")
            if prog_cb:  prog_cb(min(int(fi/max(total,1)*100), 99))
            _l(f"[FRAME {fi:05d}]  {lbl}  ({conf:.1f}%)")
        fi += 1
    cap.release()

    if not votes:
        _l("[WARN]  No frames processed.")
        if done_cb: done_cb(None, 0, {})
        return

    winner = collections.Counter(votes).most_common(1)[0][0]
    avg    = float(np.mean(acc[winner]))
    dist   = {c: round(float(np.mean(acc[c])),2) for c in acc}
    _l(f"[RESULT] {winner}  avg {avg:.1f}%")
    if prog_cb:  prog_cb(100)
    if done_cb:  done_cb(winner, avg, dist)


def append_history(filename, prediction, confidence, mode):
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = f"{ts} | {mode:5s} | {os.path.basename(filename):40s} | {prediction:12s} | {confidence:.2f}%\n"
    try:
        with open(HISTORY_FILE, "a") as f: f.write(row)
    except Exception: pass


def placeholder(w=500, h=380, msg="Drop a file or click Upload"):
    img  = Image.new("RGB", (w, h), C["card"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([3,3,w-4,h-4], outline=C["border"], width=2)
    # dashed cross hint
    cx, cy = w//2, h//2
    draw.line([(cx-30,cy),(cx+30,cy)], fill=C["dim"], width=2)
    draw.line([(cx,cy-30),(cx,cy+30)], fill=C["dim"], width=2)
    draw.text((cx, cy+55), msg, fill=C["dim"], anchor="mm")
    return img

# ══════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Brain Tumor Detection System")
        self.configure(bg=C["bg"])
        self.minsize(1200, 780)
        self._center(1260, 800)

        # state
        self.model         = None
        self._img_pil      = None
        self._img_path     = None
        self._vid_path     = None
        self._vid_thread   = None
        self._active_tab   = "image"   # "image" | "video"

        self._build()
        threading.Thread(target=self._load_model_bg, daemon=True).start()

    # ──────────────────────────────────────────────────────────────────
    #  LAYOUT
    # ──────────────────────────────────────────────────────────────────
    def _build(self):
        self._build_header()
        self._build_body()
        self._build_log()

    # ── Header ────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg="#090D13", height=58)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⬡  AI BRAIN TUMOR DETECTION SYSTEM",
                 font=FT["title"], fg=C["blue"], bg="#090D13"
                 ).pack(side="left", padx=20, pady=10)

        self._status_var = tk.StringVar(value="● Initialising …")
        self._status_lbl = tk.Label(hdr, textvariable=self._status_var,
                                    font=FT["small"], fg=C["yellow"], bg="#090D13")
        self._status_lbl.pack(side="right", padx=20)

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

    # ── Body (left tabs | right results) ──────────────────────────────
    def _build_body(self):
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=10, pady=8)

        self._build_left(body)
        self._build_right(body)

    # ── LEFT PANEL ────────────────────────────────────────────────────
    def _build_left(self, parent):
        left = tk.Frame(parent, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0,6))

        # ── TAB BAR ───────────────────────────────────────────────────
        tab_bar = tk.Frame(left, bg=C["panel"])
        tab_bar.pack(fill="x")

        self._tab_img_btn = tk.Button(
            tab_bar, text="🖼   IMAGE PREDICTION",
            font=FT["tab"], relief="flat", bd=0,
            padx=20, pady=10, cursor="hand2",
            command=self._show_image_tab,
        )
        self._tab_img_btn.pack(side="left")

        self._tab_vid_btn = tk.Button(
            tab_bar, text="🎬   VIDEO PREDICTION",
            font=FT["tab"], relief="flat", bd=0,
            padx=20, pady=10, cursor="hand2",
            command=self._show_video_tab,
        )
        self._tab_vid_btn.pack(side="left")

        tk.Frame(tab_bar, bg=C["border"], width=1).pack(side="left", fill="y")

        # Clear + Exit on right of tab bar
        tk.Button(tab_bar, text="🗑  Clear", font=FT["small"],
                  fg=C["yellow"], bg=C["panel"], activeforeground=C["yellow"],
                  activebackground=C["input"], relief="flat", bd=0,
                  padx=12, pady=10, cursor="hand2",
                  command=self._clear).pack(side="right")
        tk.Button(tab_bar, text="✕  Exit", font=FT["small"],
                  fg=C["red"], bg=C["panel"], activeforeground=C["red"],
                  activebackground=C["input"], relief="flat", bd=0,
                  padx=12, pady=10, cursor="hand2",
                  command=self._exit).pack(side="right")

        tk.Frame(left, bg=C["border"], height=1).pack(fill="x")

        # ── CANVAS (shared) ───────────────────────────────────────────
        canvas_frame = tk.Frame(left, bg=C["card"],
                                highlightthickness=1,
                                highlightbackground=C["border"])
        canvas_frame.pack(fill="both", expand=True, pady=(6,4))

        self._canvas = tk.Canvas(canvas_frame, bg=C["bg"],
                                 highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._ph_img = None      # keeps reference to current PhotoImage
        self._pil_display = placeholder()
        self._redraw_canvas()

        # frame label under canvas
        self._frame_var = tk.StringVar(value="")
        tk.Label(canvas_frame, textvariable=self._frame_var,
                 font=FT["small"], fg=C["cyan"], bg=C["card"]
                 ).pack(pady=(0,4))

        # ── IMAGE TAB CONTROLS ────────────────────────────────────────
        self._img_tab = tk.Frame(left, bg=C["bg"])

        # big upload button
        img_up_frame = tk.Frame(self._img_tab, bg=C["card"],
                                highlightthickness=1,
                                highlightbackground=C["border"])
        img_up_frame.pack(fill="x", pady=(0,4))

        tk.Label(img_up_frame, text="SELECT MRI IMAGE",
                 font=FT["small"], fg=C["dim"], bg=C["card"],
                 anchor="w").pack(fill="x", padx=10, pady=(6,2))
        tk.Frame(img_up_frame, bg=C["border"], height=1).pack(fill="x", padx=10)

        up_row = tk.Frame(img_up_frame, bg=C["card"])
        up_row.pack(fill="x", padx=10, pady=8)

        self._img_path_var = tk.StringVar(value="No image selected")
        tk.Label(up_row, textvariable=self._img_path_var,
                 font=FT["small"], fg=C["muted"], bg=C["card"],
                 anchor="w", width=45).pack(side="left", padx=(0,8))

        self._btn_browse_img = self._mkbtn(up_row, "📂  Browse Image …",
                                            C["blue"], self._browse_image)
        self._btn_browse_img.pack(side="left")

        # predict button for image
        img_pred_row = tk.Frame(self._img_tab, bg=C["bg"])
        img_pred_row.pack(fill="x", pady=2)

        self._btn_predict_img = self._mkbtn(
            img_pred_row, "🔬  RUN IMAGE PREDICTION",
            C["green"], self._predict_image,
            big=True, state="disabled")
        self._btn_predict_img.pack(fill="x", ipady=6)

        # ── VIDEO TAB CONTROLS ────────────────────────────────────────
        self._vid_tab = tk.Frame(left, bg=C["bg"])

        vid_up_frame = tk.Frame(self._vid_tab, bg=C["card"],
                                highlightthickness=1,
                                highlightbackground=C["border"])
        vid_up_frame.pack(fill="x", pady=(0,4))

        tk.Label(vid_up_frame, text="SELECT MRI VIDEO",
                 font=FT["small"], fg=C["dim"], bg=C["card"],
                 anchor="w").pack(fill="x", padx=10, pady=(6,2))
        tk.Frame(vid_up_frame, bg=C["border"], height=1).pack(fill="x", padx=10)

        vid_row = tk.Frame(vid_up_frame, bg=C["card"])
        vid_row.pack(fill="x", padx=10, pady=8)

        self._vid_path_var = tk.StringVar(value="No video selected")
        tk.Label(vid_row, textvariable=self._vid_path_var,
                 font=FT["small"], fg=C["muted"], bg=C["card"],
                 anchor="w", width=45).pack(side="left", padx=(0,8))

        self._btn_browse_vid = self._mkbtn(vid_row, "📂  Browse Video …",
                                            C["purple"], self._browse_video)
        self._btn_browse_vid.pack(side="left")

        # progress
        prog_frame = tk.Frame(self._vid_tab, bg=C["bg"])
        prog_frame.pack(fill="x", pady=(2,2))

        self._progress = ttk.Progressbar(prog_frame, orient="horizontal",
                                         mode="determinate")
        self._progress.pack(fill="x")
        self._prog_var = tk.StringVar(value="")
        tk.Label(prog_frame, textvariable=self._prog_var,
                 font=FT["small"], fg=C["muted"], bg=C["bg"]).pack(anchor="w")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Horizontal.TProgressbar",
                         troughcolor=C["input"], background=C["purple"],
                         bordercolor=C["input"], lightcolor=C["purple"],
                         darkcolor=C["purple"])

        # predict button for video
        vid_pred_row = tk.Frame(self._vid_tab, bg=C["bg"])
        vid_pred_row.pack(fill="x", pady=2)

        self._btn_predict_vid = self._mkbtn(
            vid_pred_row, "🔬  RUN VIDEO PREDICTION",
            C["green"], self._predict_video,
            big=True, state="disabled")
        self._btn_predict_vid.pack(fill="x", ipady=6)

        # show image tab by default
        self._show_image_tab()

    # ── RIGHT PANEL ───────────────────────────────────────────────────
    def _build_right(self, parent):
        right = tk.Frame(parent, bg=C["bg"], width=460)
        right.pack(side="right", fill="both", padx=(6,0))
        right.pack_propagate(False)

        # ── Result card ───────────────────────────────────────────────
        rc = self._card(right, "PREDICTION RESULT")

        self._res_var = tk.StringVar(value="—")
        self._res_lbl = tk.Label(rc, textvariable=self._res_var,
                                 font=FT["big"], fg=C["dim"], bg=C["card"],
                                 wraplength=400)
        self._res_lbl.pack(pady=(14,2))

        self._conf_var = tk.StringVar(value="")
        self._conf_lbl = tk.Label(rc, textvariable=self._conf_var,
                                  font=FT["mid"], fg=C["muted"], bg=C["card"])
        self._conf_lbl.pack(pady=(0,2))

        self._warn_var = tk.StringVar(value="")
        tk.Label(rc, textvariable=self._warn_var,
                 font=FT["small"], fg=C["yellow"], bg=C["card"],
                 wraplength=400).pack(pady=(0,6))

        tk.Frame(rc, bg=C["border"], height=1).pack(fill="x", padx=12, pady=4)

        # probability bars
        bar_frame = tk.Frame(rc, bg=C["card"])
        bar_frame.pack(fill="x", padx=16, pady=8)

        bar_colors = {"glioma": C["red"], "meningioma": C["yellow"],
                      "no_tumor": C["green"], "pituitary": C["purple"]}
        self._bars  = {}
        self._blbls = {}
        for cls in CLASSES:
            row = tk.Frame(bar_frame, bg=C["card"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=cls.replace("_"," ").title(),
                     font=FT["body"], fg=C["muted"], bg=C["card"],
                     width=11, anchor="w").pack(side="left")
            bg = tk.Frame(row, bg=C["input"], height=14, width=220)
            bg.pack(side="left", padx=6)
            bg.pack_propagate(False)
            fill = tk.Frame(bg, bg=C["dim"], height=14, width=0)
            fill.place(x=0,y=0,height=14,width=0)
            self._bars[cls] = (bg, fill, bar_colors[cls])
            pct = tk.Label(row, text="0.0%", font=FT["mono"],
                           fg=C["muted"], bg=C["card"], width=7, anchor="w")
            pct.pack(side="left")
            self._blbls[cls] = pct

        tk.Frame(rc, bg=C["border"], height=1).pack(fill="x", padx=12, pady=4)

        self._file_lbl_var = tk.StringVar(value="No file analysed")
        tk.Label(rc, textvariable=self._file_lbl_var,
                 font=FT["small"], fg=C["muted"], bg=C["card"],
                 wraplength=400).pack(pady=(0,2))
        self._time_var = tk.StringVar(value="")
        tk.Label(rc, textvariable=self._time_var,
                 font=FT["small"], fg=C["dim"], bg=C["card"]).pack(pady=(0,10))

        # ── Video distribution card ───────────────────────────────────
        dc = self._card(right, "VIDEO FRAME DISTRIBUTION")
        self._dist_text = tk.Text(dc, height=6, bg=C["input"],
                                  fg=C["muted"], font=FT["mono"],
                                  bd=0, relief="flat", state="disabled")
        self._dist_text.pack(fill="x", padx=8, pady=8)

    # ── LOG BAR ───────────────────────────────────────────────────────
    def _build_log(self):
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")
        log_outer = tk.Frame(self, bg=C["panel"], height=130)
        log_outer.pack(fill="x", side="bottom")
        log_outer.pack_propagate(False)

        hdr = tk.Frame(log_outer, bg=C["panel"])
        hdr.pack(fill="x", padx=10, pady=(5,0))
        tk.Label(hdr, text="SYSTEM LOG", font=FT["head"],
                 fg=C["cyan"], bg=C["panel"]).pack(side="left")
        tk.Button(hdr, text="Clear Log", font=FT["small"],
                  fg=C["muted"], bg=C["input"], relief="flat", bd=0,
                  padx=6, pady=2, cursor="hand2",
                  command=self._clear_log).pack(side="right")

        inner = tk.Frame(log_outer, bg=C["panel"])
        inner.pack(fill="both", expand=True, padx=10, pady=(2,6))
        sb = tk.Scrollbar(inner, bg=C["panel"], troughcolor=C["panel"])
        sb.pack(side="right", fill="y")
        self._log = tk.Text(inner, bg=C["panel"], fg=C["muted"],
                            font=FT["mono"], bd=0, relief="flat",
                            state="disabled", yscrollcommand=sb.set)
        self._log.pack(side="left", fill="both", expand=True)
        sb.config(command=self._log.yview)
        for tag, col in [("ok",C["green"]),("err",C["red"]),
                         ("warn",C["yellow"]),("info",C["cyan"]),
                         ("frame",C["muted"])]:
            self._log.tag_configure(tag, foreground=col)

    # ──────────────────────────────────────────────────────────────────
    #  TAB SWITCHING
    # ──────────────────────────────────────────────────────────────────
    def _show_image_tab(self):
        self._active_tab = "image"
        self._vid_tab.pack_forget()
        self._img_tab.pack(fill="x", pady=(4,0))
        self._tab_img_btn.configure(fg=C["blue"],   bg=C["card"])
        self._tab_vid_btn.configure(fg=C["muted"],  bg=C["panel"])
        # reset canvas if no image loaded
        if self._img_pil is None:
            self._pil_display = placeholder(msg="Browse an image to begin")
            self._redraw_canvas()
        else:
            self._pil_display = self._img_pil
            self._redraw_canvas()

    def _show_video_tab(self):
        self._active_tab = "video"
        self._img_tab.pack_forget()
        self._vid_tab.pack(fill="x", pady=(4,0))
        self._tab_vid_btn.configure(fg=C["purple"], bg=C["card"])
        self._tab_img_btn.configure(fg=C["muted"],  bg=C["panel"])
        self._pil_display = placeholder(msg="Browse a video to begin")
        self._redraw_canvas()

    # ──────────────────────────────────────────────────────────────────
    #  CANVAS HELPERS
    # ──────────────────────────────────────────────────────────────────
    def _redraw_canvas(self):
        self._canvas.update_idletasks()
        cw = max(self._canvas.winfo_width(),  1)
        ch = max(self._canvas.winfo_height(), 1)
        img = self._pil_display.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)
        padded = Image.new("RGB", (cw, ch), C["bg"])
        px = (cw - img.width)  // 2
        py = (ch - img.height) // 2
        padded.paste(img, (px, py))
        self._ph_img = ImageTk.PhotoImage(padded)
        self._canvas.delete("all")
        self._canvas.create_image(cw//2, ch//2, image=self._ph_img, anchor="center")

    def _on_canvas_resize(self, event):
        self._redraw_canvas()

    # ──────────────────────────────────────────────────────────────────
    #  WIDGET FACTORIES
    # ──────────────────────────────────────────────────────────────────
    def _card(self, parent, title=""):
        f = tk.Frame(parent, bg=C["card"],
                     highlightthickness=1, highlightbackground=C["border"])
        f.pack(fill="both", expand=True, pady=5)
        if title:
            tk.Label(f, text=title, font=FT["small"],
                     fg=C["dim"], bg=C["card"], anchor="w",
                     padx=10).pack(fill="x", pady=(6,2))
            tk.Frame(f, bg=C["border"], height=1).pack(fill="x", padx=10)
        return f

    def _mkbtn(self, parent, text, color, cmd, big=False, state="normal"):
        font = FT["head"] if big else FT["body"]
        b = tk.Button(parent, text=text, font=font,
                      fg=color, bg=C["input"],
                      activeforeground=color, activebackground=C["card"],
                      relief="flat", bd=0, padx=10, pady=6,
                      state=state, command=cmd, cursor="hand2")
        b.bind("<Enter>", lambda e: b.configure(bg=C["card"]))
        b.bind("<Leave>", lambda e: b.configure(bg=C["input"]))
        return b

    def _center(self, w, h):
        self.update_idletasks()
        x = (self.winfo_screenwidth()  - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    # ──────────────────────────────────────────────────────────────────
    #  LOGGING
    # ──────────────────────────────────────────────────────────────────
    def log(self, msg: str):
        self.after(0, self._append_log, msg)

    def _append_log(self, msg: str):
        tag = ("ok"    if "[OK]"    in msg else
               "err"   if "[ERROR]" in msg else
               "warn"  if "[WARN]"  in msg else
               "frame" if "[FRAME]" in msg else "info")
        ts   = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self._log.configure(state="normal")
        self._log.insert("end", line, tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0","end")
        self._log.configure(state="disabled")

    # ──────────────────────────────────────────────────────────────────
    #  MODEL LOAD (background)
    # ──────────────────────────────────────────────────────────────────
    def _load_model_bg(self):
        self.model = load_model_from_disk(log_fn=self.log)
        if self.model:
            self.after(0, lambda: (
                self._status_var.set("● Model ready"),
                self._status_lbl.configure(fg=C["green"])
            ))
        else:
            self.after(0, lambda: (
                self._status_var.set("● Model NOT loaded  —  check path"),
                self._status_lbl.configure(fg=C["red"])
            ))

    # ──────────────────────────────────────────────────────────────────
    #  IMAGE WORKFLOW
    # ──────────────────────────────────────────────────────────────────
    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("All","*.*")])
        if not path: return
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open image:\n{e}")
            self.log(f"[ERROR] {e}"); return

        self._img_pil  = pil
        self._img_path = path
        self._img_path_var.set(os.path.basename(path))
        self._pil_display = pil
        self._redraw_canvas()
        self._frame_var.set("")
        self._btn_predict_img.configure(state="normal")
        self._reset_result()
        self.log(f"[INFO]  Image loaded: {path}")

    def _predict_image(self):
        if self.model is None:
            messagebox.showerror("Model Error",
                "Model is not loaded.\nVerify the model path and restart.")
            return
        if self._img_pil is None:
            messagebox.showwarning("No Image", "Please browse and select an image first.")
            return
        try:
            self.log("[INFO]  Running image inference …")
            lbl, conf, probs = run_image_inference(self.model, self._img_pil)
            self._show_result(lbl, conf, probs)
            append_history(self._img_path, lbl, conf, "image")
            self.log(f"[OK]    {lbl}  |  {conf:.2f}%")
        except Exception as e:
            self.log(f"[ERROR] {e}")
            messagebox.showerror("Prediction Error", str(e))

    # ──────────────────────────────────────────────────────────────────
    #  VIDEO WORKFLOW
    # ──────────────────────────────────────────────────────────────────
    def _browse_video(self):
        if not CV2_AVAILABLE:
            messagebox.showwarning("Missing Library",
                "OpenCV is not installed.\n\npip install opencv-python")
            return
        path = filedialog.askopenfilename(
            title="Select MRI Video",
            filetypes=[("Videos","*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All","*.*")])
        if not path: return

        self._vid_path = path
        self._vid_path_var.set(os.path.basename(path))
        self._progress["value"] = 0
        self._prog_var.set("")
        self._btn_predict_vid.configure(state="normal")
        self._reset_result()
        self.log(f"[INFO]  Video selected: {path}")

        # preview first frame
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self._pil_display = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self._redraw_canvas()
            self._frame_var.set("First frame preview  —  press Run Video Prediction")

    def _predict_video(self):
        if self.model is None:
            messagebox.showerror("Model Error",
                "Model is not loaded.\nVerify the model path and restart.")
            return
        if not self._vid_path:
            messagebox.showwarning("No Video", "Please browse and select a video first.")
            return
        if self._vid_thread and self._vid_thread.is_alive():
            self.log("[WARN]  Video analysis already running."); return

        self._btn_predict_vid.configure(state="disabled")
        self._progress["value"] = 0
        self._prog_var.set("Starting …")
        self.log("[INFO]  Video analysis started …")

        def _run():
            run_video_inference(
                self.model, self._vid_path,
                prog_cb  = lambda p: self.after(0, self._set_prog, p),
                frame_cb = lambda img, lbl: self.after(0, self._vid_frame, img, lbl),
                done_cb  = lambda c, cf, d: self.after(0, self._vid_done, c, cf, d),
                log_fn   = self.log,
            )

        self._vid_thread = threading.Thread(target=_run, daemon=True)
        self._vid_thread.start()

    def _set_prog(self, pct):
        self._progress["value"] = pct
        self._prog_var.set(f"Processing … {pct}%")

    def _vid_frame(self, pil, lbl):
        self._pil_display = pil
        self._redraw_canvas()
        self._frame_var.set(lbl)

    def _vid_done(self, cls, conf, dist):
        self._prog_var.set("Analysis complete ✓")
        self._btn_predict_vid.configure(state="normal")
        if cls is None:
            self.log("[ERROR] No results from video."); return
        probs = [dist.get(c, 0.0) for c in CLASSES]
        self._show_result(cls, conf, probs)
        append_history(self._vid_path, cls, conf, "video")
        # fill distribution panel
        self._dist_text.configure(state="normal")
        self._dist_text.delete("1.0","end")
        self._dist_text.insert("end","Class            Avg Conf\n")
        self._dist_text.insert("end","─"*34+"\n")
        for c, v in sorted(dist.items(), key=lambda x:-x[1]):
            mark = "  ◀ WINNER" if c == cls else ""
            self._dist_text.insert("end", f"{c:<16s}  {v:6.2f}%{mark}\n")
        self._dist_text.configure(state="disabled")
        self.log(f"[OK]    Video result → {cls}  avg {conf:.2f}%")

    # ──────────────────────────────────────────────────────────────────
    #  RESULT DISPLAY
    # ──────────────────────────────────────────────────────────────────
    def _show_result(self, label, conf, probs):
        tumor  = label != "no_tumor"
        color  = C["red"] if tumor else C["green"]
        name   = label.replace("_"," ").title()

        self._res_var.set(name)
        self._res_lbl.configure(fg=color)
        self._conf_var.set(f"Confidence: {conf:.2f}%")
        self._conf_lbl.configure(fg=color)
        self._warn_var.set(
            "⚠  Low confidence — manual review recommended"
            if conf < CONFIDENCE_THRESH else "")
        self._time_var.set(
            datetime.datetime.now().strftime("Analysed: %Y-%m-%d  %H:%M:%S"))
        self._file_lbl_var.set(
            os.path.basename(self._img_path or self._vid_path or ""))

        bar_colors = {"glioma":C["red"],"meningioma":C["yellow"],
                      "no_tumor":C["green"],"pituitary":C["purple"]}
        for i, cls in enumerate(CLASSES):
            pct = probs[i]
            bg, fill, _ = self._bars[cls]
            bg.update_idletasks()
            w = int(pct / 100 * bg.winfo_width())
            fill.place(x=0,y=0,height=14,width=max(w,0))
            fill.configure(bg=bar_colors[cls])
            self._blbls[cls].configure(text=f"{pct:.1f}%",
                                        fg=bar_colors[cls])

    def _reset_result(self):
        self._res_var.set("—")
        self._res_lbl.configure(fg=C["dim"])
        self._conf_var.set("")
        self._warn_var.set("")
        self._time_var.set("")
        self._file_lbl_var.set("No file analysed")
        for cls in CLASSES:
            _, fill, _ = self._bars[cls]
            fill.place(x=0,y=0,height=14,width=0)
            self._blbls[cls].configure(text="0.0%", fg=C["muted"])

    # ──────────────────────────────────────────────────────────────────
    #  GLOBAL ACTIONS
    # ──────────────────────────────────────────────────────────────────
    def _clear(self):
        self._img_pil   = None
        self._img_path  = None
        self._vid_path  = None
        self._img_path_var.set("No image selected")
        self._vid_path_var.set("No video selected")
        self._frame_var.set("")
        self._prog_var.set("")
        self._progress["value"] = 0
        self._btn_predict_img.configure(state="disabled")
        self._btn_predict_vid.configure(state="disabled")
        self._pil_display = placeholder(
            msg="Browse an image or video to begin")
        self._redraw_canvas()
        self._reset_result()
        self._dist_text.configure(state="normal")
        self._dist_text.delete("1.0","end")
        self._dist_text.configure(state="disabled")
        self.log("[INFO]  Screen cleared.")

    def _exit(self):
        if messagebox.askyesno("Exit","Exit the application?"):
            self.destroy()

# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()
