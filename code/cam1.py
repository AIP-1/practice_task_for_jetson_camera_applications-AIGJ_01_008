#!/usr/bin/env python3
"""카메라 과제 1 — 카메라 연동과 기준선 측정 도구 (키트 없이 직접 만든 버전)

  # ① 입력별 측정: 첫 프레임 지연 · 실측 FPS · 해상도 · CPU 사용률
  python cam1.py probe --src 0                                  # USB 웹캠 (/dev/video0)
  python cam1.py probe --src rtsp://<맥 주소>:8554/mac      # 맥 카메라 (IP 카메라 역할)

  # ② 수신 방식 비교: 0.2초 처리 지연을 넣고 2분 뒤 지연이 얼마나 쌓이는지
  python cam1.py lag --src <입력> --reader a      # (a) cap.read()를 그대로 반복
  python cam1.py lag --src <입력> --reader b      # (b) 별도 스레드가 읽고 최신 1장만 남김

  # ③④⑤ 브라우저로 보기 + 자동 재연결 + 화면→화면 지연 측정용 시계
  python cam1.py view --src <입력> --port 8080    # 맥 브라우저: http://<젯슨 IP>:8080

결과는 runs/cam1_log.jsonl 에 한 줄씩 쌓이고, 마지막에 표로도 출력된다.

지연(lag) 계산 방법
  프레임마다 '그 프레임이 찍힌 시각'(스트림 시각, CAP_PROP_POS_MSEC)을 읽는다.
  지연 증가량 = (첫 프레임 이후 흐른 실제 시간) − (첫 프레임 이후 흐른 스트림 시각)
  오래된 프레임이 줄 서서 밀려 나오면 스트림 시각이 실제 시간보다 뒤처져 이 값이 커진다.
  첫 프레임에 이미 있던 지연(네트워크·압축 등)은 포함되지 않으므로, 절대 지연은 view 의
  화면→화면 측정으로 따로 잰다.
"""
import argparse
import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2  # noqa: E402

LOG = Path("runs/cam1_log.jsonl")


# ------------------------------------------------------------------ 공통
def open_src(src, w, h, fps):
    """USB 번호면 V4L2(MJPG), 아니면 FFmpeg(RTSP·파일)로 연다. 실패하면 None."""
    if str(src).isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
    else:
        cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def kind(src):
    s = str(src)
    return "USB 웹캠" if s.isdigit() else ("RTSP" if s.startswith("rtsp") else "파일")


class Cpu:
    """이 프로세스의 CPU 사용률(%)과 보드 전체 CPU 사용률(%)"""
    def __init__(self):
        self.t, self.p, self.sys = time.time(), time.process_time(), self._sys()

    @staticmethod
    def _sys():
        with open("/proc/stat") as f:
            v = list(map(int, f.readline().split()[1:]))
        return sum(v), v[3] + v[4]           # 전체, idle+iowait

    def read(self):
        t, p, (tot, idle) = time.time(), time.process_time(), self._sys()
        proc = (p - self.p) / max(t - self.t, 1e-6) * 100
        dt, di = tot - self.sys[0], idle - self.sys[1]
        board = (1 - di / dt) * 100 if dt else 0.0
        self.t, self.p, self.sys = t, p, (tot, idle)
        return round(proc, 1), round(board, 1)


def log(row):
    row = {"시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **row}
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("기록:", json.dumps(row, ensure_ascii=False))


class LatestReader(threading.Thread):
    """(b) 별도 스레드가 계속 읽고 가장 최신 1장만 남긴다. 끊기면 스스로 다시 연다."""
    def __init__(self, args, reconnect=False):
        super().__init__(daemon=True)
        self.args, self.reconnect = args, reconnect
        self.lock = threading.Lock()
        self.frame, self.pos, self.t_recv, self.n = None, None, 0.0, 0
        self.status, self.events = "연결 중", []
        self.cap = open_src(args.src, args.width, args.height, args.fps)

    def run(self):
        down_at = None
        while True:
            if self.cap is None:
                time.sleep(1.0)
                self.cap = open_src(self.args.src, self.args.width, self.args.height, self.args.fps)
                continue
            ok, f = self.cap.read()
            now = time.time()
            if ok:
                with self.lock:
                    self.frame, self.pos, self.t_recv = f, self.cap.get(cv2.CAP_PROP_POS_MSEC), now
                    self.n += 1
                if down_at is not None:
                    sec = now - down_at
                    self.events.append(f"{datetime.now():%H:%M:%S} 복구 ({sec:.1f}초 만에)")
                    log({"시험": "재연결", "입력": kind(self.args.src), "복구_초": round(sec, 1)})
                    down_at = None
                self.status = "수신 중"
                continue
            # 읽기 실패: 끊김
            if not self.reconnect:
                time.sleep(0.01)
                continue
            if down_at is None:
                down_at = now
                self.events.append(f"{datetime.now():%H:%M:%S} 끊김 → 다시 연결 시도")
                print("끊김 감지, 다시 연결 시도")
            self.status = "다시 연결 중"
            self.cap.release()
            self.cap = None

    def get(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy()), self.pos, self.t_recv, self.n


# ------------------------------------------------------------------ ① probe
def cmd_probe(args):
    cpu = Cpu()
    t0 = time.time()
    cap = open_src(args.src, args.width, args.height, args.fps)
    if cap is None:
        raise SystemExit(f"입력을 열 수 없음: {args.src}")
    t_open = time.time() - t0
    ok, f = cap.read()
    while not ok:
        ok, f = cap.read()
    t_first = time.time() - t0
    h, w = f.shape[:2]
    cpu.read()
    n, t1 = 0, time.time()
    while time.time() - t1 < args.secs:
        ok, _ = cap.read()
        n += ok
    el = time.time() - t1
    proc, board = cpu.read()
    cap.release()
    log({"시험": "입력 측정", "입력": kind(args.src), "src": str(args.src),
         "해상도": f"{w}x{h}", "실측_FPS": round(n / el, 1), "열기_ms": round(t_open * 1000),
         "첫프레임_ms": round(t_first * 1000), "프로세스_CPU_%": proc, "보드_CPU_%": board,
         "측정_초": args.secs})


# ------------------------------------------------------------------ ② lag
def cmd_lag(args):
    rows, t_start = [], None
    if args.reader == "a":
        cap = open_src(args.src, args.width, args.height, args.fps)
        if cap is None:
            raise SystemExit(f"입력을 열 수 없음: {args.src}")

        def grab():
            ok, f = cap.read()
            return (f if ok else None), cap.get(cv2.CAP_PROP_POS_MSEC)
    else:
        rd = LatestReader(args)
        if rd.cap is None:
            raise SystemExit(f"입력을 열 수 없음: {args.src}")
        rd.start()

        def grab():
            f, pos, _, _ = rd.get()
            return f, pos

    cpu = Cpu()
    wall0 = pos0 = None
    nxt = 10
    print(f"({args.reader}) 방식, 처리 지연 {args.work}초, {args.secs}초 동안 측정. 10초마다 출력")
    while True:
        f, pos = grab()
        now = time.time()
        if f is None or pos is None:
            time.sleep(0.01)
            continue
        if wall0 is None:
            wall0, pos0, t_start = now, pos, now
        el = now - wall0
        lag = el * 1000 - (pos - pos0)
        rows.append((el, lag))
        if el >= nxt:
            print(f"  {el:5.0f}초  지연 증가 {lag:7.0f} ms")
            nxt += 10
        if el >= args.secs:
            break
        time.sleep(args.work)                      # 처리(추론)에 걸리는 시간 흉내
    proc, board = cpu.read()
    last = [l for e, l in rows if e >= args.secs - 10]
    log({"시험": "수신 방식 비교", "입력": kind(args.src), "리더": args.reader,
         "처리지연_초": args.work, "측정_초": args.secs,
         "2분뒤_지연증가_ms": round(sum(last) / len(last)), "처리한_프레임": len(rows),
         "프로세스_CPU_%": proc, "보드_CPU_%": board})


# ------------------------------------------------------------------ ③④⑤ view
PAGE = """<!doctype html><meta charset=utf-8><title>카메라 과제 1</title>
<style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}
#clock{font:bold 64px monospace;color:#0f0}img{max-width:100%;border:1px solid #444}</style>
<h2>카메라 과제 1 · 실시간 보기</h2>
<p>화면→화면 지연: 카메라가 아래 시계를 찍게 하고, 화면 전체를 스크린숏으로 찍어 <b>위 시계 − 영상 속 시계</b>를 계산한다.</p>
<div id=clock>-</div>
<img src="/stream"><pre id=st>-</pre>
<script>
function tick(){const d=new Date();document.getElementById('clock').textContent=
 d.toTimeString().slice(0,8)+'.'+String(d.getMilliseconds()).padStart(3,'0');requestAnimationFrame(tick)}tick();
setInterval(async()=>{document.getElementById('st').textContent=await (await fetch('/status')).text()},1000);
</script>"""


def cmd_view(args):
    rd = LatestReader(args, reconnect=True)
    rd.start()
    t0, n0 = time.time(), 0

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            nonlocal t0, n0
            if self.path == "/":
                b = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b)
            elif self.path == "/status":
                _, _, _, n = rd.get()
                now = time.time()
                fps = (n - n0) / max(now - t0, 1e-6)
                t0, n0 = now, n
                txt = f"입력: {kind(args.src)} {args.src}\n상태: {rd.status}  수신 FPS: {fps:.1f}\n" + \
                      "\n".join(rd.events[-10:])
                b = txt.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                last = 0
                try:
                    while True:
                        f, _, tr, n = rd.get()
                        if f is None or n == last:
                            time.sleep(0.005)
                            continue
                        last = n
                        jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    print(f"브라우저에서 http://<젯슨 IP>:{args.port} 로 접속. 끝내려면 Ctrl+C")
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="카메라 과제 1 측정 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "lag", "view"):
        p = sub.add_parser(name)
        p.add_argument("--src", required=True, help="USB 번호(예: 0) 또는 rtsp://... 또는 동영상 파일")
        p.add_argument("--width", type=int, default=1280)
        p.add_argument("--height", type=int, default=720)
        p.add_argument("--fps", type=int, default=30)
        if name == "probe":
            p.add_argument("--secs", type=float, default=10, help="FPS를 잴 시간(초)")
        if name == "lag":
            p.add_argument("--reader", choices=["a", "b"], required=True)
            p.add_argument("--work", type=float, default=0.2, help="프레임마다 넣을 처리 지연(초)")
            p.add_argument("--secs", type=float, default=120)
        if name == "view":
            p.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    {"probe": cmd_probe, "lag": cmd_lag, "view": cmd_view}[args.cmd](args)


if __name__ == "__main__":
    main()
