#!/usr/bin/env python3
"""카메라 과제 6 — 재실 인원 서비스 v1 (키트 없이 직접 만든 버전)

카메라 여러 대에서 최신 프레임만 받아 1초마다 사람 수를 세고, 시간 평활한 값을
10초마다 MQTT(retain)로 발행한다. 같은 값을 REST로도 조회할 수 있다.
영상·사진·크롭은 어디에도 저장하지 않는다 (숫자만 내보냄).

  python occupancy_service.py --cam lab1=vigi --cam test=file:/home/team2-2/eg2-lab/media/test.mp4

  카메라 지정 (이름=입력)
    vigi              Jetson 유선 포트의 VIGI (환경 변수 CAM_PW, VIGI_IP, VIGI_STREAM)
    rtsp://...        다른 RTSP 카메라
    0, 2 ...          USB 웹캠 번호
    file:경로         동영상 파일을 실제 속도로 반복 재생 (가짜 카메라)

  결과
    MQTT   aiot/space/<이름>/occupancy  (retain, 10초마다, mosquitto_pub 로 발행)
           aiot/service/occupancy/status (online/offline, retain)
    REST   http://<젯슨>:8090/occupancy   현재 값
           http://<젯슨>:8090/health      카메라 상태·재연결 횟수·자원
    기록   logs/service_YYYYMMDD.jsonl   1분마다 값·상태·CPU·메모리·GPU
           logs/events.jsonl             시작·끊김·복구·프레임 지연(camera_stale, camera_fresh)·종료
"""
import argparse
import glob
import json
import os
import signal
import statistics
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2  # noqa: E402

STOP = threading.Event()
LOGDIR = Path("logs")
T0 = time.time()


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def event(kind, **kw):
    row = {"ts": now_iso(), "event": kind, **kw}
    LOGDIR.mkdir(exist_ok=True)
    with (LOGDIR / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("[이벤트]", json.dumps(row, ensure_ascii=False), flush=True)


def src_url(spec):
    if spec == "vigi":
        pw = os.environ.get("CAM_PW")
        if not pw:
            raise SystemExit('CAM_PW 없음: read -s -p "카메라 관리자 비밀번호: " CAM_PW; echo; export CAM_PW')
        ip = os.environ.get("VIGI_IP", "10.42.0.112")
        stream = os.environ.get("VIGI_STREAM", "stream2")
        return f"rtsp://admin:{quote(pw, safe='')}@{ip}:554/{stream}"
    return spec


# ------------------------------------------------------------------ 카메라
class Camera(threading.Thread):
    """최신 프레임 1장만 보관. 끊기면 2초마다 다시 연결 (프로세스 재시작 없음)"""

    def __init__(self, name, spec):
        super().__init__(daemon=True, name=f"cam-{name}")
        self.name, self.spec = name, spec
        self.lock = threading.Lock()
        self.frame, self.t_frame, self.t_read, self.n = None, 0.0, 0.0, 0
        self.status, self.reconnects, self.down_since = "연결 중", 0, None

    def _open(self):
        s = self.spec
        if s.isdigit():
            cap = cv2.VideoCapture(int(s), cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        elif s.startswith("file:"):
            cap = cv2.VideoCapture(s[5:])
        else:
            cap = cv2.VideoCapture(src_url(s), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _down(self, why):
        if self.down_since is None:
            self.down_since = time.time()
            self.status = "끊김(재연결 중)"
            event("camera_down", cam=self.name, why=why)

    def run(self):
        cap, is_file, delay = None, self.spec.startswith("file:"), 0.0
        while not STOP.is_set():
            if cap is None:
                cap = self._open()
                if cap is None:
                    self._down("열 수 없음")
                    STOP.wait(2.0)
                    continue
                if is_file:
                    delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 15.0)
            ok, f = cap.read()
            if not ok and is_file:                       # 파일 끝 → 처음부터 다시
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = cap.read()
            if not ok:
                cap.release()
                cap = None
                self._down("읽기 실패")
                continue
            t = time.time()
            gap, self.t_read = t - self.t_read, t
            with self.lock:
                if gap < 2.0:                            # 끊긴 동안 드문드문 오는 프레임은 살아 있는 영상으로 치지 않음
                    self.frame, self.t_frame = f, t
                self.n += 1
            if self.down_since is not None:
                sec = round(time.time() - self.down_since, 1)
                self.reconnects += 1
                event("camera_recovered", cam=self.name, after_s=sec)
                self.down_since = None
            self.status = "정상"
            if is_file:
                time.sleep(delay)
        if cap is not None:
            cap.release()

    def latest(self):
        with self.lock:
            return self.frame, self.t_frame

    def state_now(self, stale):
        """랜선이 빠지면 읽기가 실패 없이 멈추므로 마지막 프레임 시각도 함께 본다"""
        with self.lock:
            age = time.time() - self.t_frame
        if self.status == "정상" and age > stale:
            return "끊김(프레임 없음)"
        return self.status


# ------------------------------------------------------------------ 자원
class Res:
    def __init__(self):
        self.prev = self._cpu()
        self.gpu_paths = [p for p in ["/sys/devices/platform/gpu.0/load", "/sys/devices/gpu.0/load"]
                          + glob.glob("/sys/devices/platform/bus@0/*.gpu/load") if os.path.exists(p)]

    @staticmethod
    def _cpu():
        with open("/proc/stat") as f:
            v = list(map(int, f.readline().split()[1:]))
        return sum(v), v[3] + v[4]

    def read(self):
        tot, idle = self._cpu()
        dt, di = tot - self.prev[0], idle - self.prev[1]
        self.prev = (tot, idle)
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                mem[k] = int(v.split()[0])
        gpu = None
        if self.gpu_paths:
            try:
                gpu = round(int(open(self.gpu_paths[0]).read()) / 10, 1)
            except (OSError, ValueError):
                pass
        return {"cpu_pct": round((1 - di / dt) * 100, 1) if dt else None,
                "mem_used_mb": round((mem["MemTotal"] - mem["MemAvailable"]) / 1024),
                "gpu_pct": gpu}


# ------------------------------------------------------------------ 메인
def main():
    ap = argparse.ArgumentParser(description="재실 인원 서비스 v1")
    ap.add_argument("--cam", action="append", required=True, help="이름=입력 (여러 번 지정 가능)")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--period", type=float, default=1.0, help="카메라마다 몇 초에 한 번 셀지")
    ap.add_argument("--window", type=float, default=30.0, help="시간 평활 창(초)")
    ap.add_argument("--stat", choices=["median", "p75", "max"], default="median")
    ap.add_argument("--publish", type=float, default=10.0, help="MQTT 발행 주기(초)")
    ap.add_argument("--stale", type=float, default=5.0, help="이 초보다 오래된 프레임은 세지 않음")
    ap.add_argument("--mqtt", default="localhost:1883")
    ap.add_argument("--topic", default="aiot/space")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--tegrastats", action="store_true", help="logs/tegrastats.log 에 1분마다 전력·GPU 기록")
    args = ap.parse_args()

    cams = []
    for c in args.cam:
        name, spec = c.split("=", 1)
        cams.append(Camera(name, spec))
    hist = {c.name: deque() for c in cams}
    state = {c.name: {"count": None, "raw": None, "age_s": None, "stale": False} for c in cams}
    infer_ms = deque(maxlen=300)

    from ultralytics import YOLO
    model = YOLO(args.model)
    import numpy as np
    model.predict(np.zeros((480, 640, 3), np.uint8), classes=[0], verbose=False)   # 예열

    # MQTT: paho-mqtt 가 있으면 쓰고, 없으면 mosquitto_pub 명령으로 발행 (pip 설치가 안 되는 환경 대비)
    host, port = args.mqtt.split(":")
    status_topic = "aiot/service/occupancy/status"
    try:
        import paho.mqtt.client as mqtt
        mq = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"occupancy-{os.getpid()}")
        mq.will_set(status_topic, json.dumps({"status": "offline"}), qos=1, retain=True)
        mq.reconnect_delay_set(1, 30)
        mq.connect_async(host, int(port), keepalive=30)
        mq.loop_start()
        print("MQTT: paho-mqtt 사용", flush=True)
    except ImportError:
        class CliPub:
            """mosquitto_pub 로 한 건씩 발행. 마지막 성공 여부로 연결 상태를 판단"""
            ok = False

            def publish(self, topic, payload, qos=1, retain=True):
                cmd = ["mosquitto_pub", "-h", host, "-p", port, "-q", str(qos), "-t", topic, "-m", payload]
                if retain:
                    cmd.insert(1, "-r")
                try:
                    self.ok = subprocess.run(cmd, timeout=5, capture_output=True).returncode == 0
                except (OSError, subprocess.TimeoutExpired):
                    self.ok = False

            def is_connected(self):
                return self.ok

            def loop_stop(self):
                pass
        mq = CliPub()
        print("MQTT: paho-mqtt 없음 → mosquitto_pub 명령으로 발행", flush=True)
    mq.publish(status_topic, json.dumps({"status": "online", "ts": now_iso()}), qos=1, retain=True)

    tstats = None
    if args.tegrastats:
        LOGDIR.mkdir(exist_ok=True)
        tstats = subprocess.Popen(["tegrastats", "--interval", "60000", "--logfile", str(LOGDIR / "tegrastats.log")])

    for c in cams:
        c.start()
    res = Res()
    event("start", cams=[f"{c.name}={c.spec if c.spec != 'vigi' else 'vigi'}" for c in cams],
          model=args.model, imgsz=args.imgsz, window=args.window, stat=args.stat)

    def snapshot():
        out = {}
        for c in cams:
            s = state[c.name]
            out[c.name] = {"ts": now_iso(), "space": c.name, "count": s["count"], "raw": s["raw"],
                           "unit": "명", "window_s": args.window, "stat": args.stat,
                           "age_s": s["age_s"], "camera": c.state_now(args.stale), "reconnects": c.reconnects}
        return out

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            b = json.dumps(obj, ensure_ascii=False, indent=1).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/occupancy"):
                self._json(snapshot())
            elif self.path.startswith("/health"):
                self._json({"uptime_s": round(time.time() - T0), "mqtt_connected": mq.is_connected(),
                            "cams": {c.name: {"status": c.state_now(args.stale), "reconnects": c.reconnects,
                                              "frames": c.n} for c in cams},
                            "infer_ms_avg": round(statistics.mean(infer_ms), 1) if infer_ms else None,
                            "saved_images": 0, **res.read()})
            else:
                self.send_error(404)

    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever, daemon=True).start()

    def stop(*_):
        STOP.set()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    next_pub = next_log = time.time()
    print(f"서비스 시작: REST http://<젯슨>:{args.port}/occupancy , MQTT {args.mqtt} {args.topic}/<이름>/occupancy", flush=True)
    while not STOP.is_set():
        t_loop = time.time()
        for c in cams:
            f, tf = c.latest()
            s = state[c.name]
            if f is None or t_loop - tf > args.stale:
                s["age_s"] = None if f is None else round(t_loop - tf, 1)
                if f is not None and not s["stale"]:
                    s["stale"] = True
                    event("camera_stale", cam=c.name, age_s=s["age_s"])
                continue
            if s["stale"]:
                s["stale"] = False
                event("camera_fresh", cam=c.name)
            t1 = time.time()
            r = model.predict(f, classes=[0], conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            infer_ms.append((time.time() - t1) * 1000)
            raw = int(len(r.boxes))
            h = hist[c.name]
            h.append((t_loop, raw))
            while h and t_loop - h[0][0] > args.window:
                h.popleft()
            vals = sorted(v for _, v in h)
            if args.stat == "median":
                cnt = statistics.median(vals)
            elif args.stat == "p75":
                cnt = vals[min(len(vals) - 1, int(round(0.75 * (len(vals) - 1))))]
            else:
                cnt = vals[-1]
            s.update(count=int(round(cnt)), raw=raw, age_s=round(t_loop - tf, 1))
        now = time.time()
        if now >= next_pub:
            for name, msg in snapshot().items():
                mq.publish(f"{args.topic}/{name}/occupancy", json.dumps(msg, ensure_ascii=False))
            next_pub = now + args.publish
        if now >= next_log:
            row = {"ts": now_iso(), "uptime_s": round(now - T0), "mqtt": mq.is_connected(),
                   "infer_ms_avg": round(statistics.mean(infer_ms), 1) if infer_ms else None,
                   "cams": snapshot(), **res.read()}
            LOGDIR.mkdir(exist_ok=True)
            with (LOGDIR / f"service_{datetime.now():%Y%m%d}.jsonl").open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            next_log = now + 60
        STOP.wait(max(0.0, args.period - (time.time() - t_loop)))

    event("stop", uptime_s=round(time.time() - T0))
    mq.publish(status_topic, json.dumps({"status": "offline", "ts": now_iso()}))
    if tstats:
        tstats.terminate()


if __name__ == "__main__":
    main()
