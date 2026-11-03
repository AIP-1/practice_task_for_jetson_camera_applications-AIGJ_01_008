#!/usr/bin/env python3
"""과제 2 선 통과 계수 (키트 없이 만든 버전): 박스 하단 중앙이 세로선을 넘으면 IN/OUT 1회
  python line_count.py --src 0 --line 0.5 0 0.5 1 --seconds 180
  python line_count.py --src vigi --mqtt-host 127.0.0.1
화면 왼쪽→오른쪽 = IN (위→아래로 그은 선 기준). 사진은 저장하지 않는다.
"""
import argparse, json, os, subprocess, time
from urllib.parse import quote
import cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="0", help="USB 번호, vigi, 또는 동영상 파일 경로")
ap.add_argument("--model", default="yolo11n.pt")
ap.add_argument("--line", type=float, nargs=4, default=[0.5, 0.0, 0.5, 1.0], metavar=("X0", "Y0", "X1", "Y1"))
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--seconds", type=float, default=180)
ap.add_argument("--mqtt-host", default=None)
ap.add_argument("--topic", default="aiot/vision/test")
ap.add_argument("--tracker", default="bytetrack.yaml", help="ultralytics 추적기 설정 (bytetrack.yaml / botsort.yaml)")
a = ap.parse_args()

if a.src == "vigi":
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    ip = os.environ.get("VIGI_IP", "10.42.0.111")
    cap = cv2.VideoCapture(f"rtsp://admin:{quote(os.environ['CAM_PW'], safe='')}@{ip}:554/stream2", cv2.CAP_FFMPEG)
elif not a.src.isdigit():
    cap = cv2.VideoCapture(a.src)                       # 동영상 파일: 같은 계수 로직을 녹화본에 돌린다
else:
    cap = cv2.VideoCapture(int(a.src), cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
assert cap.isOpened(), "카메라를 열 수 없음"
model = YOLO(a.model)
side, ins, outs, t0, n = {}, 0, 0, time.time(), 0

def publish(kind, tid):
    if a.mqtt_host:
        msg = json.dumps({"ts": round(time.time(), 2), "dir": kind, "id": tid, "in": ins, "out": outs})
        subprocess.Popen(["mosquitto_pub", "-h", a.mqtt_host, "-t", f"{a.topic}/event", "-m", msg])

while time.time() - t0 < a.seconds:
    ok, f = cap.read()
    if not ok:
        if not a.src.isdigit() and a.src != "vigi":
            break                                       # 동영상 끝
        time.sleep(0.05); continue
    h, w = f.shape[:2]; n += 1
    ax, ay, bx, by = a.line[0] * w, a.line[1] * h, a.line[2] * w, a.line[3] * h
    r = model.track(f, persist=True, classes=[0], conf=a.conf, tracker=a.tracker, verbose=False)[0]
    if r.boxes.id is None:
        continue
    for (x1, y1, x2, y2), tid in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist()):
        px, py = (x1 + x2) / 2, y2                         # 박스 하단 중앙 = 발
        s = (bx - ax) * (py - ay) - (by - ay) * (px - ax)  # 외적 부호로 선의 어느 쪽인지 판단
        cur = 1 if s > 0 else -1
        if tid in side and side[tid] != cur:
            if cur < 0:
                ins += 1; publish("IN", tid); print(f"IN  #{tid}  합계 IN {ins} / OUT {outs}", flush=True)
            else:
                outs += 1; publish("OUT", tid); print(f"OUT #{tid}  합계 IN {ins} / OUT {outs}", flush=True)
        side[tid] = cur
print(f"끝: IN {ins} / OUT {outs}, {n / (time.time() - t0):.1f} fps")
