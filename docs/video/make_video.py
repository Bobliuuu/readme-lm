#!/usr/bin/env python3
"""Regenerate explainer.mp4, poster.png, preview.gif and script.md.   Usage: python3 docs/video/make_video.py
Needs: macOS `say` (voice Samantha), ffmpeg, Playwright + installed Google Chrome, numpy, Pillow.
Pipeline: narration.json -> per-sentence TTS (exact durations) -> scene timeline -> scenes.html driven frame by frame
(window.setT(t)) through headless Chrome -> ffmpeg (H.264 + AAC). Scratch files live in a temp dir that is deleted."""
import json, os, subprocess, sys, shutil, tempfile, wave, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
VOICE, RATE, FPS, SR = "Samantha", 220, 20, 44100
LEAD, GAP, TAIL = 0.4, 0.15, 0.5
ap = argparse.ArgumentParser(); ap.add_argument("--keep", action="store_true"); ap.add_argument("--only-plan", action="store_true"); ap.add_argument("--shots", help="scene:sentence:offset,... only write test PNGs to ./_shots"); A = ap.parse_args()
tmp = tempfile.mkdtemp(prefix="rlmvideo_")
sh = lambda *a, **k: subprocess.run(a, check=True, **k)

# 1. narration audio, one clip per sentence
nar = json.load(open(os.path.join(HERE, "narration.json")))
def wav_of(i, j, text):
    aiff, wv = f"{tmp}/s{i}_{j}.aiff", f"{tmp}/s{i}_{j}.wav"
    sh("say", "-v", VOICE, "-r", str(RATE), "-o", aiff, text)
    sh("ffmpeg", "-loglevel", "error", "-y", "-i", aiff, "-ar", str(SR), "-ac", "1", wv)
    w = wave.open(wv); d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32); return d / 32768
# 2. timeline
scenes, t, chunks = [], 0.0, []
for i, sc in enumerate(nar):
    st, ends, caps, lt = [], [], [], LEAD
    for j, s in enumerate(sc["sents"]):
        a = wav_of(i, j, s.get("tts", s["cap"])); st.append(lt); ends.append(lt + len(a) / SR); caps.append(s["cap"])
        chunks.append((t + lt, a)); lt += len(a) / SR + GAP
    dur = ends[-1] + TAIL
    scenes.append(dict(id=sc["id"], start=t, end=t + dur, st=st, ends=ends, caps=caps)); t += dur
TOTAL = t
print("scene durations:", [round(s["end"] - s["start"], 1) for s in scenes], "total", round(TOTAL, 1))
for s in scenes: print(s["id"], [round(x,1) for x in s["st"]], round(s["end"]-s["start"],1))
if A.only_plan: shutil.rmtree(tmp); sys.exit()
mix = np.zeros(int(TOTAL * SR) + SR, dtype=np.float32)
for t0, a in chunks: k = int(t0 * SR); mix[k:k + len(a)] += a
mix = np.clip(mix * 0.95, -1, 1)
w = wave.open(f"{tmp}/audio.wav", "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()

# 3. assets: live SVG rendered by Chrome at exactly 640x492, DPR 1 (as in demos/svg/harness.py); GPT-2 gif frames
sys.path.insert(0, os.path.join(ROOT, "demos", "svg")); import harness
from PIL import Image
arr, _ = harness.render(os.path.join(ROOT, "demos", "svg", "latest.svg"), 640, 492, img=True)
Image.fromarray(arr).convert("RGB").save(f"{tmp}/latest.png")
harness._br.close(); harness._pw.stop()
os.makedirs(f"{tmp}/gif")
sh("ffmpeg", "-loglevel", "error", "-y", "-i", os.path.join(ROOT, "docs", "img", "gpt2-generation.gif"), "-vsync", "0", "-q:v", "3", f"{tmp}/gif/%03d.jpg")
gif = sorted(os.listdir(f"{tmp}/gif"))
# work dir for the page: scenes + assets side by side
page = f"{tmp}/page"; os.makedirs(page)
for f in ("scenes.html", "scenes.css", "scenes.js"): shutil.copy(os.path.join(HERE, f), page)
os.makedirs(f"{page}/live")
sh("ffmpeg", "-loglevel", "error", "-y", "-t", "70", "-i", os.path.join(ROOT, "docs", "gifs", "issue-flow.mp4"), "-vf", "crop=954:945:0:135,scale=528:523,fps=12", "-q:v", "3", f"{page}/live/%04d.jpg")
LF = sorted(f[:-4] for f in os.listdir(f"{page}/live"))
shutil.copy(f"{tmp}/latest.png", page); shutil.copytree(f"{tmp}/gif", f"{page}/gif")

# 4. frame-by-frame render
from playwright.sync_api import sync_playwright
out = os.path.join(HERE, "explainer.mp4")
with sync_playwright() as pw:
    br = pw.chromium.launch(channel="chrome", headless=True, args=["--hide-scrollbars"])
    pg = br.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1).new_page()
    pg.goto("file://" + page + "/scenes.html"); pg.wait_for_load_state("load")
    pg.evaluate("async c=>await init(c)", dict(scenes=scenes, gifFrames=["gif/" + g for g in gif], liveFrames=["live/" + f + ".jpg" for f in LF]))
    if A.shots:
        os.makedirs(os.path.join(HERE, "_shots"), exist_ok=True)
        for spec in A.shots.split(","):
            sid, si, off = spec.split(":"); s_ = next(x for x in scenes if x["id"] == sid)
            pg.evaluate("t=>setT(t)", s_["start"] + s_["st"][int(si)] + float(off)); pg.screenshot(path=os.path.join(HERE, "_shots", f"{sid}_{si}_{off}.png"))
        br.close(); sys.exit()
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", str(FPS), "-i", "-", "-i", f"{tmp}/audio.wav",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-t", f"{TOTAL:.3f}", out], stdin=subprocess.PIPE)
    n = int(TOTAL * FPS)
    for f in range(n):
        pg.evaluate("t=>setT(t)", f / FPS)
        ff.stdin.write(pg.screenshot(type="jpeg", quality=92))
        if f % 200 == 0: print("frame", f, "/", n, flush=True)
    ff.stdin.close(); ff.wait()
    # poster: the three-versions overview, once everything is on screen
    s3 = next(s for s in scenes if s["id"] == "three"); pg.evaluate("t=>setT(t)", s3["start"] + s3["st"][2] + 1.8)
    pg.screenshot(path=os.path.join(HERE, "poster.png"), type="png"); br.close()

# 5. preview gif: highlights (scene id, sentence index, offset s, length s)
HL = [("title", 0, 0.5, 1.6), ("puzzle", 1, 0.2, 2.2), ("three", 0, 0.4, 2.0), ("a", 3, 0.2, 2.6), ("b", 1, 0.3, 2.0), ("c", 4, 0.4, 1.8), ("live", 1, 0.3, 2.0), ("tried", 2, 1.2, 1.6), ("outro", 0, 0.3, 1.2)]
sc = {s["id"]: s for s in scenes}; parts = []
for k, (sid, si, off, ln) in enumerate(HL):
    s0 = sc[sid]["start"] + sc[sid]["st"][si] + off; parts.append(f"[0:v]trim=start={s0:.3f}:duration={ln},setpts=PTS-STARTPTS[v{k}]")
fc = ";".join(parts) + ";" + "".join(f"[v{k}]" for k in range(len(HL))) + f"concat=n={len(HL)}:v=1:a=0,fps=10,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
sh("ffmpeg", "-loglevel", "error", "-y", "-i", out, "-filter_complex", fc, os.path.join(HERE, "preview.gif"))

# 6. script.md
L = ["# README-LM explainer: script", "", f"Narration: macOS `say`, voice {VOICE}. Total {TOTAL:.0f} s. Generated by make_video.py from narration.json.", ""]
for i, (n, s) in enumerate(zip(nar, scenes), 1):
    L += [f"## Scene {i}: {n['id']} ({s['start']:.0f}-{s['end']:.0f} s)", "", f"**On screen:** {n['onscreen']}", "", "**Narration (also burned in as captions):**", ""]
    L += [f"{j}. {x['cap']}" for j, x in enumerate(n["sents"], 1)] + [""]
open(os.path.join(HERE, "script.md"), "w").write("\n".join(L))
if not A.keep: shutil.rmtree(tmp)
print("done", TOTAL)
