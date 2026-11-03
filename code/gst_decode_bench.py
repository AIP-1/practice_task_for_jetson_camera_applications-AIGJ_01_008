#!/usr/bin/env python3
"""카메라 과제 1 심화: 같은 RTSP 입력을 하드웨어(NVDEC)와 CPU(openh264)로 디코딩해 CPU 사용률·FPS 비교

  set -a; . ~/.config/pose.env; set +a
  /usr/bin/python3 gst_decode_bench.py --stream stream1 --secs 60

시스템 파이썬으로 실행한다(venv에는 gi가 없음). 카메라 주소는 환경 변수(CAM_PW, VIGI_IP)로
프로세스 안에서 만들어, 비밀번호가 명령줄(ps)에 보이지 않게 한다. 디코딩한 화면은 버린다(fakesink).
"""
import argparse
import os
import resource
import time
from urllib.parse import quote

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

DECODERS = {"hw": "nvv4l2decoder", "cpu": "openh264dec"}


def board_cpu():
    """/proc/stat 의 (전체, 쉰 시간)"""
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]


def run(kind, url, secs, warmup):
    p = Gst.parse_launch(
        "rtspsrc name=src protocols=tcp latency=200 ! rtph264depay ! h264parse ! "
        f"{DECODERS[kind]} ! fakesink name=sink sync=false")
    p.get_by_name("src").set_property("location", url)
    n = [0]

    def probe(pad, info):
        n[0] += 1
        return Gst.PadProbeReturn.OK

    p.get_by_name("sink").get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe)
    bus = p.get_bus()
    t_open = time.monotonic()
    p.set_state(Gst.State.PLAYING)
    first = None
    while time.monotonic() - t_open < warmup:         # 연결과 첫 화면, 안정화
        m = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.ERROR)
        if m:
            err, dbg = m.parse_error()
            p.set_state(Gst.State.NULL)
            return {"decoder": kind, "error": err.message}
        if first is None and n[0]:
            first = time.monotonic() - t_open
    r0, b0, n0, t0 = resource.getrusage(resource.RUSAGE_SELF), board_cpu(), n[0], time.monotonic()
    while time.monotonic() - t0 < secs:
        m = bus.timed_pop_filtered(200 * Gst.MSECOND, Gst.MessageType.ERROR)
        if m:
            break
    r1, b1, n1, t1 = resource.getrusage(resource.RUSAGE_SELF), board_cpu(), n[0], time.monotonic()
    p.set_state(Gst.State.NULL)
    wall = t1 - t0
    cpu_s = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    tot, idle = b1[0] - b0[0], b1[1] - b0[1]
    return {"decoder": kind, "element": DECODERS[kind], "secs": round(wall, 1),
            "fps": round((n1 - n0) / wall, 1), "frames": n1 - n0,
            "first_frame_s": round(first, 2) if first is not None else None,
            "proc_cpu_pct_1core": round(100 * cpu_s / wall, 1),
            "board_cpu_pct": round(100 * (tot - idle) / tot, 1) if tot else None,
            "max_rss_mb": round(r1.ru_maxrss / 1024)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="stream1", help="stream1 = 메인, stream2 = 서브")
    ap.add_argument("--secs", type=float, default=60)
    ap.add_argument("--warmup", type=float, default=8)
    ap.add_argument("--only", choices=list(DECODERS), help="한 쪽만 잴 때")
    a = ap.parse_args()
    Gst.init(None)
    url = (f"rtsp://admin:{quote(os.environ['CAM_PW'], safe='')}@"
           f"{os.environ.get('VIGI_IP', '10.42.0.111')}:554/{a.stream}")
    for kind in ([a.only] if a.only else ["hw", "cpu"]):
        print(f"{kind} ({DECODERS[kind]}) {a.stream} {a.secs:.0f}초 측정 중...", flush=True)
        print(run(kind, url, a.secs, a.warmup), flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
