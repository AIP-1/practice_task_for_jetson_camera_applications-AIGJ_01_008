#!/usr/bin/env python3
"""과제 2 조명×거리 사람 검출 신뢰도: 한 칸(조명, 거리)을 N초 동안 재서 CSV에 한 줄 추가

  set -a; . ~/.config/pose.env; set +a
  python light_dist.py --light 밝음 --dist 2 --secs 30

프레임마다 사람(class 0) 박스 중 가장 높은 신뢰도를 기록한다(없으면 0). 검출률은 신뢰도 0.4 이상인
프레임 비율이다. 화면 밝기(회색 평균 0~255)를 함께 남겨 조명 조건을 숫자로 적는다. 사진은 저장하지 않는다.
"""
import argparse, csv, os, threading, time
from urllib.parse import quote
import cv2
import numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--light", required=True, help="예: 밝음, 어두움")
ap.add_argument("--dist", required=True, help="카메라에서 사람까지 거리(m)")
ap.add_argument("--secs", type=float, default=30)
ap.add_argument("--model", default="../yolo11n.pt")
ap.add_argument("--th", type=float, default=0.4, help="검출로 셀 신뢰도")
ap.add_argument("--out", default="../runs/light_dist.csv")
a = ap.parse_args()

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
url = f"rtsp://admin:{quote(os.environ['CAM_PW'], safe='')}@{os.environ.get('VIGI_IP', '10.42.0.111')}:554/stream2"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
assert cap.isOpened(), "카메라를 열 수 없음"
latest = [None]

def reader():                       # 최신 1장만 남긴다 (처리가 느려도 지연이 쌓이지 않게)
    while True:
        ok, f = cap.read()
        if ok:
            latest[0] = f

threading.Thread(target=reader, daemon=True).start()
model = YOLO(a.model)
while latest[0] is None:
    time.sleep(0.05)
model(latest[0], classes=[0], conf=0.05, verbose=False)   # 첫 실행 준비

confs, bright, t0 = [], [], time.time()
while time.time() - t0 < a.secs:
    f = latest[0]
    r = model(f, classes=[0], conf=0.05, verbose=False)[0]
    c = r.boxes.conf.tolist()
    confs.append(max(c) if c else 0.0)
    bright.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
c = np.array(confs)
row = {"time": time.strftime("%H:%M:%S"), "light": a.light, "dist_m": a.dist, "frames": len(c),
       "det_rate": round(float((c >= a.th).mean()), 3), "conf_mean": round(float(c.mean()), 3),
       "conf_median": round(float(np.median(c)), 3), "brightness": round(float(np.mean(bright)), 1)}
new = not os.path.exists(a.out)
with open(a.out, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
print(row)
os._exit(0)
