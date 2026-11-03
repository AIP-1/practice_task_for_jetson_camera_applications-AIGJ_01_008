#!/usr/bin/env python3
"""과제 2 벤치마크 나머지 두 줄: yolo11n INT8 엔진, yolo11s FP16 엔진을 만들고 같은 방식으로 잰다

  cd ~/eg2-lab/trt_b && python ../trt_extra.py

- 변환은 하나씩 자식 프로세스로 돌리고, 남은 메모리가 --guard MB 아래로 내려가면 그 변환을 멈춘다(공용 보드 보호).
- 속도: 1280x720 임의 영상 200회(앞 20회 버림)의 평균·p95 (앞선 bench_yolo.py와 같은 방식).
- 정확도: coco128 검증의 mAP50-95와 사진 1장당 추론 시간(ms).
"""
import argparse, csv, subprocess, sys, threading, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--guard", type=int, default=400, help="이 MB 아래로 남으면 변환 중단")
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--out", default="../runs/bench_trt_extra.csv")
ap.add_argument("--only", choices=["int8", "large"], default=None, help="하나만 할 때")
ap.add_argument("--timeout", type=int, default=0, help="변환을 이 초 넘으면 중단 (0 = 제한 없음)")
a = ap.parse_args()

JOBS = [("yolo11n INT8", "yolo11n.pt", {"int8": True, "data": "coco128.yaml"}),
        ("yolo11s FP16", "yolo11s.pt", {"half": True})]
if a.only:
    JOBS = [JOBS[0] if a.only == "int8" else JOBS[1]]

def avail_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024

def export(pt, kw):
    """자식 프로세스에서 변환. 반환: (엔진 경로 또는 None, 변환 중 최소 남은 메모리, 걸린 초, 메모)"""
    code = ("from ultralytics import YOLO; import sys; "
            f"p = YOLO({pt!r}).export(format='engine', imgsz=640, workspace=0.5, **{kw!r}); print('ENGINE=' + str(p))")
    t0, low = time.time(), avail_mb()
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines, stop = [], {"why": ""}
    def watch():
        nonlocal low
        while proc.poll() is None:
            m = avail_mb(); low = min(low, m)
            if m < a.guard:
                stop["why"] = f"남은 메모리 {m} MB로 중단"; proc.kill(); return
            if a.timeout and time.time() - t0 > a.timeout:
                stop["why"] = f"{a.timeout}초 넘어 중단"; proc.kill(); return
            time.sleep(1)
    threading.Thread(target=watch, daemon=True).start()
    for line in proc.stdout:
        lines.append(line.rstrip())
    proc.wait()
    eng = next((l[7:] for l in lines if l.startswith("ENGINE=")), None)
    note = stop["why"] or ("" if eng else (lines[-1] if lines else "실패"))
    return eng, low, round(time.time() - t0), note

def bench(path):
    from ultralytics import YOLO
    m = YOLO(path, task="detect")
    img = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8)
    ts = []
    for i in range(a.frames + 20):
        t = time.perf_counter(); m(img, imgsz=640, verbose=False); ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts[20:])
    v = m.val(data="coco128.yaml", imgsz=640, verbose=False, plots=False)
    return round(ts.mean(), 2), round(float(np.percentile(ts, 95)), 2), round(1000 / ts.mean(), 1), \
        round(float(v.box.map), 3), round(float(v.speed["inference"]), 1)

rows = []
for name, pt, kw in JOBS:
    print(f"== {name}: 변환 시작 (남은 메모리 {avail_mb()} MB)", flush=True)
    eng, low, secs, note = export(pt, kw)
    row = {"model": name, "export_s": secs, "min_avail_mb": low, "note": note}
    if eng:
        mean, p95, fps, mp, inf = bench(eng)
        row.update(mean_ms=mean, p95_ms=p95, fps=fps, map50_95=mp, infer_ms=inf)
    rows.append(row); print(row, flush=True)

keys = ["model", "mean_ms", "p95_ms", "fps", "map50_95", "infer_ms", "export_s", "min_avail_mb", "note"]
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("저장:", a.out)
for r in rows:
    print(f"| {r['model']} | {r.get('mean_ms','-')} | {r.get('p95_ms','-')} | {r.get('fps','-')} | {r.get('map50_95','-')} | "
          f"추론 {r.get('infer_ms','-')} ms | 변환 {r['export_s']}초, 최소 {r['min_avail_mb']} MB {r['note']} |")
