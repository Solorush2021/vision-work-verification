"""
Mini-Optifye demo v2: YOLOE text-prompted object detection + full body skeleton
+ MediaPipe finger keypoints + zone/anti-gaming state machine + annotated render.
"""
import argparse, os, time
import cv2
import numpy as np

OBJ_CLASSES = ["bottle", "flask", "can", "cup", "mug", "container", "thermos"]
WRIST_IDX = (9, 10)
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
ZONE_A, ZONE_B = (0.06, 0.44), (0.56, 0.94)
STABLE_FRAMES = 3
EMA_ALPHA = 0.35
FAKE_COOLDOWN = 2.5

C_GREEN, C_RED, C_ORANGE, C_BLUE, C_MAGENTA, C_CYAN, C_WHITE, C_BLACK, C_GRAY = (
    (60, 220, 60), (60, 60, 235), (30, 150, 255), (235, 160, 40), (230, 80, 230),
    (230, 230, 80), (255, 255, 255), (0, 0, 0), (170, 170, 170))


class Pipe:
    def __init__(self):
        self.cycles = self.fakes = self.reverses = 0
        self.zone, self.cand, self.cnt = None, None, 0
        self.work_dir = None  # learned from first deliberate carry: +1 = A->B is work
        self.bx = None
        self.bspeed = self.wspeed = 0.0
        self.prev_wr = None
        self.fake = False
        self.fake_t = -10.0
        self.banner, self.bcolor, self.bttl = "", C_WHITE, 0
        self.events = []  # (t, text)

    def step(self, t, bx_raw, wrists):
        if bx_raw is not None:
            if self.bx is None:
                self.bx = bx_raw
            self.bspeed = 0.4 * abs(bx_raw - self.bx) + 0.6 * self.bspeed
            self.bx = EMA_ALPHA * bx_raw + (1 - EMA_ALPHA) * self.bx
        if wrists:
            if self.prev_wr is not None and len(wrists) == len(self.prev_wr):
                d = np.mean([abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(wrists, self.prev_wr)])
                self.wspeed = 0.35 * d + 0.65 * self.wspeed
            self.prev_wr = wrists
        else:
            self.prev_wr = None
            self.wspeed *= 0.92
        was = self.fake
        self.fake = self.wspeed > 0.030 and self.bspeed < 0.006
        if self.fake and not was and t - self.fake_t > FAKE_COOLDOWN:
            self.fakes += 1
            self.fake_t = t
            self.events.append((t, "FAKE ATTEMPT"))
        if self.bx is not None:
            for name, (lo, hi) in (("A", ZONE_A), ("B", ZONE_B)):
                if lo <= self.bx <= hi:
                    self.cnt = self.cnt + 1 if self.cand == name else 1
                    self.cand = name
                    if self.cnt >= STABLE_FRAMES and self.zone != name:
                        if self.zone is None:
                            self.zone = name  # first stabilization: no event
                        else:
                            direction = 1 if (self.zone == "A" and name == "B") else -1
                            if self.work_dir is None:
                                self.work_dir = direction  # first deliberate carry defines work direction
                            if direction == self.work_dir:
                                self.cycles += 1
                                self.banner, self.bcolor, self.bttl = "CYCLE COUNTED  +1", C_GREEN, 40
                                self.events.append((t, "CYCLE"))
                            else:
                                self.reverses += 1
                                self.banner, self.bcolor, self.bttl = "REVERSE TRANSFER - NOT COUNTED", C_ORANGE, 40
                                self.events.append((t, "REVERSE"))
                            self.zone = name
                    break
        if self.bttl > 0:
            self.bttl -= 1
        return self


def draw_skeleton(img, kpts, confs=None):
    if kpts is None or not len(kpts):
        return
    pts = [(float(k[0]), float(k[1])) for k in kpts]
    for a, b in SKELETON:
        if pts[a][0] > 0 and pts[b][0] > 0:
            cv2.line(img, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), C_CYAN, 2)
    for i, (x, y) in enumerate(pts):
        if x > 0:
            col = C_MAGENTA if i in WRIST_IDX else C_CYAN
            cv2.circle(img, (int(x), int(y)), 5 if i in WRIST_IDX else 3, col, -1)


def draw_hands(img, hands_lms):
    for lm in hands_lms or []:
        for x, y in lm:
            cv2.circle(img, (int(x), int(y)), 3, C_MAGENTA, -1)


def card(img, lines, W, H):
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (W, H), C_BLACK, -1)
    img = cv2.addWeighted(ov, 0.78, img, 0.22, 0)
    y = H // 2 - 22 * len(lines) + 10
    for i, (txt, scale, col) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(txt, 2, scale, 2)
        cv2.putText(img, txt, (W // 2 - tw // 2, y + i * 40), 2, scale, col, 2)
    return img


def llm_summary(events, stats):
    import json, urllib.request
    sys_p = ("You are a factory floor supervisor. You receive JSON event data from a vision system that "
             "counts completed work cycles and detects fake work (hand motion without part movement). "
             "Reply with exactly 2 short plain sentences: what happened, and one recommendation.")
    payload = {"model": "qwen3.5-0.8b-mtp", "messages": [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": json.dumps({"events": [(round(t, 1), e) for t, e in events], "totals": stats})}],
        "max_tokens": 1500, "temperature": 0.3}
    req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    msg = r["choices"][0]["message"]
    text = (msg.get("content") or "").strip()
    if not text:
        # thinking-model fallback: the drafted answer appears in the reasoning tail
        import re as _re
        rsn = msg.get("reasoning_content") or ""
        quotes = _re.findall(r'"([^"\n]{25,300})"', rsn)
        if quotes:
            text = quotes[-1].strip()
        else:
            text = " ".join(rsn.split())[-220:]
    return text


def wrap_lines(s, width=52):
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out[:6]


def summary_clip(path, text, W, H, fps, secs=5):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for _ in range(int(fps * secs)):
        img = np.zeros((H, W, 3), np.uint8)
        cv2.putText(img, "SUPERVISOR - local 0.8B model", (W // 2 - 230, 90), 2, 0.9, C_WHITE, 2)
        cv2.putText(img, "input: JSON event stream from the vision pipeline", (W // 2 - 260, 130), 2, 0.58, C_GRAY, 1)
        y = H // 2 - 30
        for ln in wrap_lines(text):
            (tw, _), _ = cv2.getTextSize(ln, 2, 0.85, 2)
            cv2.putText(img, ln, (W // 2 - tw // 2, y), 2, 0.85, C_WHITE, 2)
            y += 42
        vw.write(img)
    vw.release()


def concat(out, parts):
    c0 = cv2.VideoCapture(parts[0])
    fps, W, H = c0.get(5) or 30, int(c0.get(3)), int(c0.get(4))
    c0.release()
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for p in parts:
        c = cv2.VideoCapture(p)
        while True:
            ok, f = c.read()
            if not ok:
                break
            vw.write(f)
        c.release()
    vw.release()


def run(inp, out, previews=None):
    from ultralytics import YOLO, YOLOE
    import mediapipe as mp_pkg
    obj_model = YOLOE("yoloe-11s-seg.pt")
    obj_model.set_classes(OBJ_CLASSES, obj_model.get_text_pe(OBJ_CLASSES))
    pose_model = YOLO("yolo11n-pose.pt")
    hands = mp_pkg.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4)

    cap = cv2.VideoCapture(inp)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    pipe = Pipe()
    times, idx, t0 = [], 0, time.time()
    title_n, end_n = int(fps * 2.5), int(fps * 3.0)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = idx / fps
        ts = time.time()
        ores = obj_model.predict(f, imgsz=640, conf=0.3, verbose=False)[0]
        pres = pose_model(f, imgsz=320, verbose=False)[0]
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        hres = hands.process(rgb)
        times.append(time.time() - ts)

        bx_raw, obox, olabel = None, None, ""
        best = 0
        for x1, y1, x2, y2, cf, c in ores.boxes.data.tolist():
            if cf > best:
                best, obox, olabel = cf, tuple(map(int, (x1, y1, x2, y2))), OBJ_CLASSES[int(c)]
        if obox:
            bx_raw = (obox[0] + obox[2]) / 2 / W
        wrists = []
        kpts = pres.keypoints.xy if pres.keypoints is not None and len(pres.keypoints.xy) else None
        if kpts is not None:
            for kp in kpts:
                for ki in WRIST_IDX:
                    x, y = float(kp[ki][0]), float(kp[ki][1])
                    if x > 0 and y > 0:
                        wrists.append((x / W, y / H))
        hands_lms = []
        if hres.multi_hand_landmarks:
            for hl in hres.multi_hand_landmarks:
                hands_lms.append([(p.x * W, p.y * H) for p in hl.landmark])
        pipe.step(t, bx_raw, wrists if wrists else None)

        # ---- render ----
        zone_ov = f.copy()
        for name, (lo, hi), col in (("A", ZONE_A, (90, 160, 90)), ("B", ZONE_B, (90, 120, 190))):
            cv2.rectangle(zone_ov, (int(W * lo), 36), (int(W * hi), H - 36), col, -1)
        f = cv2.addWeighted(zone_ov, 0.18, f, 0.82, 0, f)  # ~80% translucent zones
        for name, (lo, hi) in (("ZONE A", ZONE_A), ("ZONE B", ZONE_B)):
            cv2.putText(f, name, (int(W * lo) + 8, 56), 1, 1.0, C_GRAY, 2)
            cv2.line(f, (int(W * lo), 36), (int(W * lo), H - 36), C_GRAY, 1)
            cv2.line(f, (int(W * hi), 36), (int(W * hi), H - 36), C_GRAY, 1)
        if kpts is not None:
            for kp in kpts:
                draw_skeleton(f, kp)
        draw_hands(f, hands_lms)
        if obox:
            cv2.rectangle(f, (obox[0], obox[1]), (obox[2], obox[3]), C_GREEN, 3)
            cv2.putText(f, f"{olabel} {best:.2f}", (obox[0], obox[1] - 8), 2, 0.8, C_GREEN, 2)
        # stacked color-coded counters, top-left, translucent backing
        hud_ov = f.copy()
        cv2.rectangle(hud_ov, (0, 0), (185, 108), C_BLACK, -1)
        f = cv2.addWeighted(hud_ov, 0.55, f, 0.45, 0, f)
        cv2.putText(f, f"CYCLES  {pipe.cycles}", (12, 26), 2, 0.8, C_GREEN, 2)
        cv2.putText(f, f"FAKES   {pipe.fakes}", (12, 52), 2, 0.8, C_RED, 2)
        cv2.putText(f, f"REVERSE {pipe.reverses}", (12, 78), 2, 0.8, C_ORANGE, 2)
        cv2.putText(f, f"ZONE {pipe.zone or '-'}", (12, 100), 2, 0.65, C_WHITE, 2)
        pfps = (idx + 1) / (time.time() - t0)
        fps_ov = f.copy()
        cv2.rectangle(fps_ov, (W - 108, 0), (W, 26), C_BLACK, -1)
        f = cv2.addWeighted(fps_ov, 0.55, f, 0.45, 0, f)
        cv2.putText(f, f"{pfps:.1f} FPS", (W - 100, 20), 2, 0.6, C_GRAY, 2)
        bar = int(min(1.0, pipe.wspeed / 0.08) * 100)
        mb_ov = f.copy()
        cv2.rectangle(mb_ov, (12, H - 24), (116, H - 12), C_BLACK, -1)
        f = cv2.addWeighted(mb_ov, 0.5, f, 0.5, 0, f)
        cv2.rectangle(f, (14, H - 22), (14 + max(0, bar), H - 14), C_MAGENTA if pipe.wspeed > 0.03 else C_GRAY, -1)
        if pipe.fake:
            cv2.rectangle(f, (W // 2 - 250, H // 2 - 62), (W // 2 + 250, H // 2 - 8), C_BLACK, -1)
            cv2.putText(f, "FAKING TASK DETECTED", (W // 2 - 222, H // 2 - 26), 2, 1.05, C_RED, 3)
        elif pipe.bttl > 0:
            (tw, _), _ = cv2.getTextSize(pipe.banner, 2, 1.05, 3)
            cv2.rectangle(f, (W // 2 - tw // 2 - 18, H // 2 - 62), (W // 2 + tw // 2 + 18, H // 2 - 8), C_BLACK, -1)
            cv2.putText(f, pipe.banner, (W // 2 - tw // 2, H // 2 - 26), 2, 1.05, pipe.bcolor, 3)
        if idx < title_n:
            f = card(f, [("CYCLE VERIFICATION PIPELINE", 1.25, C_WHITE),
                         ("text-prompted detection  |  body pose  |  hand keypoints", 0.62, C_GRAY),
                         ("counts real work, blocks fake work - no faces, no biometrics", 0.62, C_WHITE)], W, H)
        if total and idx >= total - end_n:
            ms = 1000 * np.mean(times)
            f = card(f, [("RESULTS", 1.15, C_WHITE),
                         (f"real cycles: {pipe.cycles}", 0.95, C_GREEN),
                         (f"fake attempts blocked: {pipe.fakes}", 0.95, C_RED),
                         (f"reverse transfers (not counted): {pipe.reverses}", 0.95, C_ORANGE),
                         (f"yoloe-11s + yolo11n-pose + mediapipe | {ms:.0f} ms/frame | CPU", 0.55, C_GRAY)], W, H)
        vw.write(f)
        if previews and idx in previews:
            cv2.imwrite(previews[idx], f)
        idx += 1
    cap.release(); vw.release(); hands.close()
    print(f"[done] {idx} frames -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    print(f"[stats] cycles={pipe.cycles} fakes={pipe.fakes} reverses={pipe.reverses} | {1000*np.mean(times):.0f} ms/frame | events: {pipe.events}")
    try:
        s = llm_summary(pipe.events, {"cycles": pipe.cycles, "fakes": pipe.fakes, "reverses": pipe.reverses})
        import re as _re
        s = _re.sub(r"<think>.*?</think>", "", s, flags=_re.S).strip() or s.strip()
        print("[llm]", s)
        summary_clip("_supervisor.mp4", s, W, H, fps)
        concat("_final.mp4", [out, "_supervisor.mp4"])
        os.replace("_final.mp4", out)
        os.remove("_supervisor.mp4")
        print("[llm] supervisor segment appended")
    except Exception as e:
        print("[llm] unavailable, skipping supervisor segment:", e)
    return pipe


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="out_v2.mp4")
    a = ap.parse_args()
    total_est = int(cv2.VideoCapture(a.input).get(cv2.CAP_PROP_FRAME_COUNT) or 248)
    pv = {int(total_est * 0.25): "preview_1.png", int(total_est * 0.5): "preview_2.png",
          int(total_est * 0.75): "preview_3.png", max(0, total_est - 60): "preview_4.png"}
    run(a.input, a.output, previews=pv)
