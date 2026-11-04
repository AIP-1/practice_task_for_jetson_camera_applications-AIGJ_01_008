#!/usr/bin/env python3
"""과제 2 벤치마크 전력 칸: 엔진마다 1280x720 임의 영상을 반복 추론하는 동안 tegrastats(0.2초 간격)로 보드 전력(VDD_IN)을 잰다

  cd ~/eg2-lab/trt_b && python ../power_bench.py yolo11s.engine yolo11n.engine

- 앞선 FP16 yolo11n 전력(10.9 W)과 같은 조건이 되도록 서비스 2개는 켠 채로 잰다.
- 엔진마다 추론 전 10초 대기(유휴 전력), 추론 --secs초 동안 평균·최대 전력, 평균 추론 시간을 낸다.
"""
import argparse, re, subprocess, sys, threading, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("engines", nargs="+")
ap.add_argument("--secs", type=float, default=30)
a = ap.parse_args()

samples = []          # (시각, mW)
def reader(proc):
    for line in proc.stdout:
        m = re.search(r"VDD_IN (\d+)mW", line)
        if m:
            samples.append((time.time(), int(m.group(1))))

ts_proc = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE, text=True)
threading.Thread(target=reader, args=(ts_proc,), daemon=True).start()

def watts(t0, t1):
    v = [mw for t, mw in samples if t0 <= t <= t1]
    return (round(np.mean(v) / 1000, 1), round(max(v) / 1000, 1), len(v)) if v else (None, None, 0)

from ultralytics import YOLO
img = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8)
print("| 엔진 | 유휴 평균 W | 추론 중 평균 W | 추론 중 최대 W | 평균 ms | 횟수 |")
print("| --- | --- | --- | --- | --- | --- |")
for path in a.engines:
    m = YOLO(path, task="detect")
    for _ in range(20):
        m(img, imgsz=640, verbose=False)            # 예열
    t_idle = time.time(); time.sleep(10)
    idle = watts(t_idle, time.time())
    t0, ms = time.time(), []
    while time.time() - t0 < a.secs:
        t = time.perf_counter(); m(img, imgsz=640, verbose=False); ms.append((time.perf_counter() - t) * 1000)
    avg, peak, n = watts(t0, time.time())
    print(f"| {path} | {idle[0]} | {avg} | {peak} | {np.mean(ms):.2f} | {len(ms)} |", flush=True)
    del m
ts_proc.terminate()
