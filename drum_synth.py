#!/usr/bin/env python3
"""
Random Drum Synth
-----------------
Generate random synthesized drum sounds (kick, snare, hi-hat, perc), tweak them,
and export as 16/24-bit WAV.

Install:
    pip install numpy sounddevice
    (Linux only, if tkinter is missing:  sudo apt install python3-tk libportaudio2)

Run:
    python drum_synth.py

Shortcuts:  G = generate   R = replay   Ctrl+S = export WAV
"""

import io
import queue
import threading
import tkinter as tk
import wave
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    import sounddevice as sd
except Exception:  # PortAudio missing, etc.
    sd = None

try:  # Windows-only fallback player
    import winsound
except Exception:
    winsound = None

SR = 44100  # sample rate

# --------------------------------------------------------------------------
#  Synthesis engine (pure numpy, no GUI code)
# --------------------------------------------------------------------------

CATEGORIES = ["Kick", "Snare", "Hi-Hat", "Perc/Noise"]

# Slider presets per category (what the sliders jump to when you pick one)
PRESETS = {
    "Kick":       dict(fmin=100, fmax=200,  drop=60, mix=5,  dur=450, dist=15),
    "Snare":      dict(fmin=250, fmax=400,  drop=30, mix=60, dur=220, dist=10),
    "Hi-Hat":     dict(fmin=300, fmax=500,  drop=0,  mix=85, dur=70,  dist=5),
    "Perc/Noise": dict(fmin=300, fmax=900,  drop=45, mix=25, dur=250, dist=10),
}

# Envelope time constants (fraction of total duration unless noted)
ENV = {
    "Kick":       dict(tone=0.28, noise=None, noise_abs=0.008),  # noise = short click
    "Snare":      dict(tone=0.12, noise=0.22),
    "Hi-Hat":     dict(tone=0.15, noise=0.17),
    "Perc/Noise": dict(tone=0.20, noise=0.20),
}

METAL_RATIOS = np.array([1.0, 1.4471, 1.6170, 1.9265, 2.5028, 2.6637])


def _filter(x, kind, fc):
    """Zero-phase 2nd-order-ish FFT filter ('hp' or 'lp') or pink-noise tilt ('pink')."""
    n = len(x)
    nfft = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / SR)
    if kind == "hp":
        H = 1.0 / np.sqrt(1.0 + (fc / np.maximum(f, 1e-9)) ** 4)
    elif kind == "lp":
        H = 1.0 / np.sqrt(1.0 + (f / fc) ** 4)
    elif kind == "pink":
        H = 1.0 / np.sqrt(np.maximum(f, 20.0))
    else:
        return x
    return np.fft.irfft(X * H, nfft)[:n]


def _osc(phase, shape):
    s = np.sin(phase)
    if shape == "tri":
        return (2.0 / np.pi) * np.arcsin(s)
    if shape == "square":
        return np.tanh(4.0 * s)  # slightly rounded square
    return s


def _unit_rms(x, target=0.7):
    return x / (np.std(x) + 1e-12) * target


def synthesize(category, base, jitter, chaos, seed):
    """
    Deterministically render a drum hit.

    category : one of CATEGORIES
    base     : dict of slider values (fmin, fmax, drop, mix, dur, dist)
    jitter   : 7 random numbers in [-1, 1] (index 6 in [0, 1]) frozen per sound
    chaos    : 0..1 – how far jitter pushes the parameters
    seed     : seed for the noise / waveform choices
    returns  : (float32 audio in [-0.95, 0.95], info dict)
    """
    rng = np.random.default_rng(seed)
    u = jitter
    c = float(chaos)

    # ---- effective parameters (slider value + frozen randomness * chaos)
    fmin = np.clip(base["fmin"] * 2 ** (u[0] * c * 2.0), 20, 12000)
    fmax = np.clip(base["fmax"] * 2 ** (u[1] * c * 2.0), 20, 12000)
    lo, hi = min(fmin, fmax), max(fmin, fmax)
    f_start = lo * (hi / lo) ** u[6]
    drop = float(np.clip(base["drop"] + u[2] * c * 50, 0, 100))
    mix = float(np.clip(base["mix"] + u[3] * c * 50, 0, 100))
    dur_ms = float(np.clip(base["dur"] * 2 ** (u[4] * c * 1.2), 10, 2500))
    dist = float(np.clip(base["dist"] + u[5] * c * 50, 0, 100))

    n = max(int(SR * dur_ms / 1000.0), 64)
    t = np.arange(n) / SR
    dur = n / SR

    # ---- pitch envelope: exponential sweep from f_start down to f_end
    octaves = drop / 100.0 * 3.0
    f_end = min(max(f_start * 2 ** (-octaves), 25.0), f_start)
    tau_p = 0.100 - 0.088 * drop / 100.0  # 100 ms (slow) .. 12 ms (snappy)
    freq = f_end + (f_start - f_end) * np.exp(-t / tau_p)
    phase = 2 * np.pi * np.cumsum(freq) / SR

    # ---- tonal part
    shape_pool = ["sine"] if c < 0.15 else ["sine", "sine", "tri", "square"]
    shape = shape_pool[rng.integers(len(shape_pool))]
    fm_idx = c * rng.uniform(0, 5 if category == "Perc/Noise" else 2)
    fm_ratio = rng.choice([0.5, 1.0, 1.41, 2.0, 3.0, 3.73])
    mod = fm_idx * np.sin(fm_ratio * phase) * np.exp(-t / (0.3 * dur))

    if category == "Hi-Hat":
        # six detuned square waves = classic metallic cluster
        tone = np.zeros(n)
        for r in METAL_RATIOS:
            tone += np.sign(np.sin(r * phase + mod))
        tone = _filter(tone / 6.0, "hp", 3500.0)
    elif category == "Snare":
        tone = 0.7 * _osc(phase + mod, shape) + 0.4 * _osc(1.58 * phase, "sine")
    else:
        tone = _osc(phase + mod, shape)
    tone = _unit_rms(tone)

    # ---- noise part
    noise = rng.standard_normal(n)
    pink_prob = 0.2 if category == "Hi-Hat" else 0.4
    if rng.random() < pink_prob:
        noise = _filter(noise, "pink", 0)
    if category == "Hi-Hat":
        noise = _filter(noise, "hp", rng.uniform(6000, 9000))
    elif category == "Snare":
        noise = _filter(noise, "hp", rng.uniform(900, 2500))
        noise = _filter(noise, "lp", rng.uniform(8000, 12000))
    elif category == "Kick":
        noise = _filter(noise, "lp", rng.uniform(1500, 4000))
    else:
        hp = rng.uniform(200, 3000)
        noise = _filter(noise, "hp", hp)
        noise = _filter(noise, "lp", min(hp * rng.uniform(2, 8), 12000))
    noise = _unit_rms(noise)

    # ---- amplitude envelopes
    env = ENV[category]
    tone_env = np.exp(-t / (env["tone"] * dur))
    if env["noise"] is None:
        noise_env = np.exp(-t / env["noise_abs"])
    else:
        noise_env = np.exp(-t / (env["noise"] * dur))
    attack = np.minimum(1.0, t / 0.0008)  # 0.8 ms anti-click ramp

    # ---- equal-power tone/noise crossfade
    a = mix / 100.0 * np.pi / 2
    x = np.cos(a) * tone * tone_env + np.sin(a) * noise * noise_env
    x *= attack
    x /= np.max(np.abs(x)) + 1e-12

    # ---- saturation
    if dist > 0:
        drive = 1.0 + dist / 100.0 * 24.0
        x = np.tanh(drive * x)

    # ---- fade-out tail + normalise
    tail = int(min(0.006 * SR, n * 0.2))
    if tail > 1:
        x[-tail:] *= np.linspace(1.0, 0.0, tail)
    x = x / (np.max(np.abs(x)) + 1e-12) * 0.95

    info = dict(category=category, f_start=f_start, f_end=f_end,
                dur_ms=dur_ms, mix=mix, drop=drop, dist=dist, shape=shape)
    return x.astype(np.float32), info


def write_wav(path, audio, bits=16):
    """Write mono PCM WAV (16 or 24-bit) using only the stdlib wave module."""
    x = np.clip(audio.astype(np.float64), -1.0, 1.0)
    if bits == 16:
        dither = (np.random.random(len(x)) - np.random.random(len(x)))  # TPDF
        data = np.round(x * 32767 + dither).clip(-32768, 32767).astype("<i2").tobytes()
        width = 2
    else:
        i32 = np.round(x * 8388607).astype("<i4")
        data = i32.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()  # keep low 3 bytes
        width = 3
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(SR)
        w.writeframes(data)


# --------------------------------------------------------------------------
#  GUI
# --------------------------------------------------------------------------

BG, PANEL, FG, MUTED = "#080F0F", "#080F0F", "#F1FFFA", "#FF0000"
ACCENT, ACCENT_HOT, WAVE = "#FF0000", "#FF0000", "#FF0000"


class Param(ttk.Frame):
    """Label + live value readout + slider (optionally log-scaled)."""

    def __init__(self, master, label, lo, hi, init, fmt, log=False, command=None):
        super().__init__(master, style="Panel.TFrame")
        self.lo, self.hi, self.fmt, self.log, self.command = lo, hi, fmt, log, command
        self.var = tk.DoubleVar(value=self._to_pos(init))
        head = ttk.Frame(self, style="Panel.TFrame")
        head.pack(fill="x")
        ttk.Label(head, text=label, style="Panel.TLabel").pack(side="left")
        self.readout = ttk.Label(head, text="", style="Muted.TLabel")
        self.readout.pack(side="right")
        self.scale = ttk.Scale(self, from_=self._to_pos(lo), to=self._to_pos(hi),
                               variable=self.var, command=self._on_move)
        self.scale.pack(fill="x", pady=(2, 8))
        self._refresh()

    def _to_pos(self, v):
        return float(np.log10(v)) if self.log else float(v)

    def _from_pos(self, p):
        return float(10 ** p) if self.log else float(p)

    def _refresh(self):
        self.readout.config(text=self.fmt(self.get()))

    def _on_move(self, _=None):
        self._refresh()
        if self.command:
            self.command()

    def get(self):
        return self._from_pos(self.var.get())

    def set(self, v):  # programmatic: does not fire command
        self.var.set(self._to_pos(v))
        self._refresh()


class DrumSynthApp:
    def __init__(self, root):
        self.root = root
        root.title("Random Drum Synth")
        root.geometry("940x580")
        root.minsize(820, 520)
        root.configure(bg=BG)
        self._style()

        self.audio = None
        self.sound = None            # dict(category, seed, jitter)
        self._gen_id = 0
        self._pending_play = False
        self._debounce = None
        self._loading = False
        self.results = queue.Queue()

        self._build_ui()
        self._bind_keys()
        self.root.after(25, self._poll_results)

        if sd is None and winsound is None:
            self.status.config(text="Playback unavailable (pip install sounddevice) – export still works.")

        self.generate(play=False)   # silent first sound so the canvas isn't empty

    # ---------- styling
    def _style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=PANEL, font=("Segoe UI", 10))
        s.configure("TFrame", background=BG)
        s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=FG)
        s.configure("Panel.TLabel", background=PANEL, foreground=FG)
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        s.configure("Title.TLabel", background=PANEL, foreground=FG, font=("Segoe UI", 13, "bold"))
        s.configure("Status.TLabel", background=BG, foreground=MUTED)
        s.configure("Horizontal.TScale", background=PANEL, troughcolor="#343746")
        s.configure("TButton", background="#343746", foreground=FG, padding=(14, 8), borderwidth=0)
        s.map("TButton", background=[("active", "#444759")])
        s.configure("Accent.TButton", background=ACCENT, foreground="white",
                    font=("Segoe UI", 11, "bold"), padding=(18, 10))
        s.map("Accent.TButton", background=[("active", ACCENT_HOT)])
        s.configure("TCombobox", arrowcolor=FG, background="#343746", foreground=FG)
        s.configure("TCheckbutton", background=BG, foreground=FG)
        s.configure("TRadiobutton", background=BG, foreground=FG)
        s.map("TCheckbutton", background=[("active", BG)])
        s.map("TRadiobutton", background=[("active", BG)])

    # ---------- layout
    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # ---- left: controls
        left = ttk.Frame(outer, style="Panel.TFrame", padding=16)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        ttk.Label(left, text="Synthesis", style="Title.TLabel").pack(anchor="w", pady=(0, 10))

        ttk.Label(left, text="Sound category", style="Panel.TLabel").pack(anchor="w")
        self.cat_var = tk.StringVar(value="Kick")
        cb = ttk.Combobox(left, textvariable=self.cat_var, state="readonly", width=24,
                          values=CATEGORIES + ["Fully Random"])
        cb.pack(fill="x", pady=(2, 12))
        cb.bind("<<ComboboxSelected>>", self._on_category)

        hz = lambda v: f"{v:,.0f} Hz"
        self.p_fmin = Param(left, "Min start frequency", 20, 4000, 100, hz, log=True, command=self._on_param)
        self.p_fmax = Param(left, "Max start frequency", 20, 4000, 200, hz, log=True, command=self._on_param)
        self.p_drop = Param(left, "Pitch decay / drop", 0, 100, 60, lambda v: f"{v:.0f}%", command=self._on_param)
        self.p_mix = Param(left, "Tone ↔ Noise mix", 0, 100, 5, lambda v: f"{v:.0f}% noise", command=self._on_param)
        self.p_dur = Param(left, "Envelope / duration", 10, 1500, 450, lambda v: f"{v:,.0f} ms", log=True, command=self._on_param)
        self.p_dist = Param(left, "Distortion / saturation", 0, 100, 15, lambda v: f"{v:.0f}%", command=self._on_param)
        self.p_chaos = Param(left, "Chaos (random intensity)", 0, 100, 35, lambda v: f"{v:.0f}%", command=self._on_param)
        for p in (self.p_fmin, self.p_fmax, self.p_drop, self.p_mix, self.p_dur, self.p_dist, self.p_chaos):
            p.pack(fill="x")

        # ---- right: waveform + actions
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(right, bg=PANEL, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self._draw_wave())

        self.info = ttk.Label(right, text="", style="Status.TLabel")
        self.info.grid(row=1, column=0, sticky="w", pady=(8, 8))

        bar = ttk.Frame(right)
        bar.grid(row=2, column=0, sticky="ew")
        ttk.Button(bar, text="Generate Random Sound", style="Accent.TButton",
                   command=lambda: self.generate(play=True)).pack(side="left")
        ttk.Button(bar, text="▶ Replay", command=self.replay).pack(side="left", padx=8)
        ttk.Button(bar, text="Export WAV", command=self.export).pack(side="left")

        opts = ttk.Frame(bar)
        opts.pack(side="right")
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Auto-preview on tweak", variable=self.auto_var).pack(side="left", padx=(0, 12))
        self.bits_var = tk.IntVar(value=24)
        ttk.Radiobutton(opts, text="16-bit", variable=self.bits_var, value=16).pack(side="left")
        ttk.Radiobutton(opts, text="24-bit", variable=self.bits_var, value=24).pack(side="left")

        self.status = ttk.Label(right, text="Keys: G generate · R replay · Ctrl+S export", style="Status.TLabel")
        self.status.grid(row=3, column=0, sticky="w", pady=(8, 0))

    def _bind_keys(self):
        self.root.bind("<Key-g>", lambda e: self.generate(play=True))
        self.root.bind("<Key-r>", lambda e: self.replay())
        self.root.bind("<Control-s>", lambda e: self.export())

    # ---------- parameter handling
    def _slider_values(self):
        return dict(fmin=self.p_fmin.get(), fmax=self.p_fmax.get(), drop=self.p_drop.get(),
                    mix=self.p_mix.get(), dur=self.p_dur.get(), dist=self.p_dist.get())

    def _load_preset(self, cat):
        pr = PRESETS[cat]
        self._loading = True
        self.p_fmin.set(pr["fmin"]); self.p_fmax.set(pr["fmax"]); self.p_drop.set(pr["drop"])
        self.p_mix.set(pr["mix"]); self.p_dur.set(pr["dur"]); self.p_dist.set(pr["dist"])
        self._loading = False

    def _on_category(self, _=None):
        if self.cat_var.get() != "Fully Random":
            self._load_preset(self.cat_var.get())
        self.generate(play=True)

    def _on_param(self):
        if self._loading or self.sound is None:
            return
        if self._debounce:
            self.root.after_cancel(self._debounce)
        self._debounce = self.root.after(120, self._tweak_render)

    def _tweak_render(self):
        self._debounce = None
        self._request_render(play=self.auto_var.get())

    # ---------- generate / render (worker thread)
    def generate(self, play=True):
        rng = np.random.default_rng()
        choice = self.cat_var.get()
        if choice == "Fully Random":
            cat = CATEGORIES[rng.integers(len(CATEGORIES))]
            self._load_preset(cat)
        else:
            cat = choice
        jitter = rng.uniform(-1, 1, 7)
        jitter[6] = rng.uniform(0, 1)
        self.sound = dict(category=cat, seed=int(rng.integers(1, 2**31 - 1)), jitter=jitter)
        self._request_render(play=play)

    def _request_render(self, play):
        self._gen_id += 1
        gid = self._gen_id
        self._pending_play = self._pending_play or play
        snap = dict(sound=dict(self.sound), base=self._slider_values(),
                    chaos=self.p_chaos.get() / 100.0)

        def work():
            audio, info = synthesize(snap["sound"]["category"], snap["base"],
                                     snap["sound"]["jitter"], snap["chaos"], snap["sound"]["seed"])
            info["seed"] = snap["sound"]["seed"]
            self.results.put((gid, audio, info))

        threading.Thread(target=work, daemon=True).start()

    def _poll_results(self):
        try:
            while True:
                gid, audio, info = self.results.get_nowait()
                if gid != self._gen_id:      # stale render, a newer one is coming
                    continue
                self.audio = audio
                self._draw_wave()
                self.info.config(text=(
                    f"{info['category']} · {info['f_start']:.0f} → {info['f_end']:.0f} Hz · "
                    f"{info['dur_ms']:.0f} ms · {info['shape']} · "
                    f"noise {info['mix']:.0f}% · drive {info['dist']:.0f}% · seed {info['seed']}"))
                if self._pending_play:
                    self._pending_play = False
                    self.replay()
        except queue.Empty:
            pass
        self.root.after(25, self._poll_results)

    # ---------- playback (non-blocking)
    def replay(self):
        if self.audio is None:
            return
        try:
            if sd is not None:
                sd.stop()
                sd.play(self.audio, SR)           # returns immediately
            elif winsound is not None:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                    w.writeframes((self.audio * 32767).astype("<i2").tobytes())
                winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY | winsound.SND_ASYNC)
        except Exception as exc:
            self.status.config(text=f"Playback error: {exc}")

    # ---------- export
    def export(self):
        if self.audio is None:
            messagebox.showinfo("Nothing to export", "Generate a sound first.")
            return
        cat = (self.sound["category"] if self.sound else "drum").lower().replace("/", "-")
        path = filedialog.asksaveasfilename(
            title="Export WAV", defaultextension=".wav",
            initialfile=f"{cat}_{self.sound['seed'] % 100000}.wav",
            filetypes=[("WAV audio", "*.wav")])
        if not path:
            return
        try:
            write_wav(path, self.audio, self.bits_var.get())
            self.status.config(text=f"Saved {path} ({self.bits_var.get()}-bit, {SR/1000:.1f} kHz)")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    # ---------- waveform canvas
    def _draw_wave(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 10:
            return
        mid = h / 2
        for frac in (0.25, 0.5, 0.75):   # faint grid
            c.create_line(0, h * frac, w, h * frac, fill="#2d3040")
        if self.audio is None:
            return
        a = self.audio
        edges = np.linspace(0, len(a), w + 1).astype(int)
        for x in range(w):
            seg = a[edges[x]:max(edges[x + 1], edges[x] + 1)]
            if seg.size:
                c.create_line(x, mid - seg.max() * mid * 0.92, x, mid - seg.min() * mid * 0.92, fill=WAVE)
        c.create_text(8, h - 8, anchor="sw", fill=MUTED, text="0 ms")
        c.create_text(w - 8, h - 8, anchor="se", fill=MUTED, text=f"{len(a) / SR * 1000:.0f} ms")


def main():
    root = tk.Tk()
    DrumSynthApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
