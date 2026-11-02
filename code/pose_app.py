import os, sys, time, threading, collections, argparse, csv, json, re, subprocess
from urllib.parse import quote
import cv2, numpy as np, uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
from pose_model import (KP_CONF, NOSE, LS, RS, LW, RW, body, facing_score,
                        frame_features, load_model, predict)

# ---------------- 실행 옵션 ----------------
# --raw          : 프레임마다 판정한 결과를 그대로 표시 (9-1 문제 확인용)
# (기본)         : 0.6초 다수결 안정화 + 손 흔들기(시간 패턴) 판정 (9-2 개선)
# --record F.csv : 관절 특징을 CSV로 저장 (9-3 학습 데이터 수집)
# --behavior     : 정면 응시·이동 속도로 행동 패턴(지나감/힐끗 봄/관심 있음/머무름) 판정 (9-5)
# --model M.pt   : 학습한 GRU 모델로 판정 (9-6). 여러 번 지정 가능 (동작용, 행동용)
# --mqtt-host H  : 이벤트·상태·집계를 MQTT로 발행 (응용 서비스 A3)
# --stats-log F  : 1분마다 fps·사람 수·재연결·메모리를 CSV로 기록 (12시간 가동 기록)
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="rtsp",
                help="rtsp(기본, VIGI 카메라), USB 장치 번호(예: 0), 또는 영상 파일(예: media/pose_test.mp4)")
ap.add_argument("--stream", default="stream2", help="RTSP 스트림: stream1(2560x1440) / stream2(848x480)")
ap.add_argument("--conf", type=float, default=0.5)
ap.add_argument("--raw", action="store_true")
ap.add_argument("--record", default=None, help="예: data/wave.csv")
ap.add_argument("--behavior", action="store_true")
ap.add_argument("--model", action="append", default=[], help="예: models/action.pt")
ap.add_argument("--port", type=int, default=8001, help="팀원 app.py가 8000을 쓰므로 기본 8001")
ap.add_argument("--mqtt-host", default=None, help="MQTT 브로커 주소. 없으면 발행하지 않음")
ap.add_argument("--mqtt-port", type=int, default=1883)
ap.add_argument("--topic", default="aiot/kiosk/lab1", help="발행 토픽 앞부분 (/event, /status, /summary)")
ap.add_argument("--stats-log", default=None, help="예: runs/pose_stats.csv")
ap.add_argument("--watchdog", type=float, default=120.0, help="이 초 동안 프레임이 없으면 프로세스 종료 (systemd가 다시 띄움). 0이면 끔")
args = ap.parse_args()
MODE = "raw" if args.raw else "smooth"
FILE = args.src != "rtsp" and not args.src.isdigit()   # 영상 파일로 같은 장면을 반복 비교

CAM_IP = os.environ.get("VIGI_IP", "10.42.0.111")   # Jetson 유선 포트의 VIGI 카메라. 주소가 바뀌면 find_cam_ip()가 다시 찾는다
CAM_MAC = os.environ.get("VIGI_MAC", "")   # 카메라 MAC 주소 (저장소에 올리지 않음)
W, H = 1280, 720         # USB 웹캠일 때만 사용
SMOOTH_S = 0.6        # 다수결 창(초)
WAVE_S = 1.5          # 손 흔들기 판정 창(초)
BEH_WIN = 15.0        # 행동 패턴 판정 창(초)
FACE_TH = 0.6         # 정면으로 볼 facing 점수
LOOK_S, INTEREST_S, LINGER_S = 1.0, 3.0, 10.0
MOVE_TH = 0.5         # 이동 판정 속도 (어깨 너비/초)
ML_TH = 0.7           # 모델 확률 기준
ML_HANDS = 0.7        # 판정 구간 중 두 손목이 보이는 프레임 비율이 이보다 낮으면 모델 대신 규칙 사용
EN = {"만세": "BOTH ARMS UP", "한팔들기": "ONE ARM UP", "손흔들기": "WAVE", "악수": "HANDSHAKE",
      "기본": "-", "지나감": "PASSING", "힐끗 봄": "GLANCE", "관심 있음": "INTERESTED",
      "머무름": "LINGER", "관찰 중": "..."}
BEH_RANK = {"관찰 중": 0, "지나감": 1, "머무름": 2, "힐끗 봄": 3, "관심 있음": 4}
LE, RE = 7, 8         # COCO 관절: 왼쪽·오른쪽 팔꿈치 (어깨 높이 손 흔들기 판정용)
FACE_R = 0.5          # 손목이 코에서 어깨 너비의 이 배수 이내면 손들기·흔들기에서 제외

if args.src == "rtsp" and "CAM_PW" not in os.environ:
    raise SystemExit('CAM_PW가 없음: read -s -p "카메라 관리자 비밀번호: " CAM_PW; echo; export CAM_PW')

det = YOLO("yolo11n-pose.pt")         # 첫 실행 시 자동 다운로드
models = [load_model(p) for p in args.model]
SEQ_LEN = max([m["window"] for m in models], default=1)
print(f"[pose mode={MODE}] conf={args.conf} behavior={args.behavior} src={args.src} port={args.port} "
      f"models={[(m['task'], m['labels']) for m in models]}")

frame_lock, ev_lock = threading.Lock(), threading.Lock()
latest_jpeg = None
tracks = {}
pairs = {}            # (tid_a, tid_b) -> deque[(t, bool)]
pair_on = set()
events = collections.deque(maxlen=50)
counts = collections.Counter()
beh_counts = collections.Counter()   # 사람(추적 ID) 단위 최종 행동
stats = {"fps": 0.0, "people": 0}
rec = None
rec_tid = None        # 녹화 대상: 매 프레임 가장 큰 박스의 추적 ID
if args.record:
    os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
    rec_f = open(args.record, "a", newline="")
    rec = csv.writer(rec_f)

def new_track(now):
    return {"raw": collections.deque(maxlen=90), "wave": collections.deque(maxlen=90),
            "beh": collections.deque(maxlen=600), "feat": collections.deque(maxlen=SEQ_LEN),
            "label": "기본", "bstate": "관찰 중", "best": "관찰 중", "ml": {},
            "first": now, "seen": now, "prev": None, "look": 0.0, "n": 0}

def near_face(k, w, sw):
    """손목이 코에서 어깨 너비 FACE_R배 이내면 얼굴 근처 (턱 괴기·얼굴 만지기·전화 받기)"""
    return k[NOSE, 2] > KP_CONF and np.linalg.norm(k[w, :2] - k[NOSE, :2]) < FACE_R * sw

def pose_label(k, b):
    (cx, cy), sw = b
    raised = lambda w: k[w, 2] > KP_CONF and k[w, 1] < cy - 0.3 * sw and not near_face(k, w, sw)
    lu, ru = raised(LW), raised(RW)
    if lu and ru:
        return "만세"
    if lu or ru:
        return "한팔들기"
    return "기본"

def wave_wrist(k, b):
    """흔드는 손 후보: 손목이 팔꿈치보다 어깨 너비 0.1배 이상 위, 가슴(어깨+0.5배)보다 위.
    한팔들기 기준(어깨보다 0.3배 위)보다 낮아서 어깨 높이에서 팔꿈치를 굽혀 흔드는 동작도 포함한다"""
    (cx, cy), sw = b
    best = None
    for w, e in ((LW, LE), (RW, RE)):
        if (k[w, 2] > KP_CONF and k[e, 2] > KP_CONF and k[w, 1] < k[e, 1] - 0.1 * sw
                and k[w, 1] < cy + 0.5 * sw and not near_face(k, w, sw)):
            if best is None or k[w, 1] < k[best, 1]:
                best = w
    return best

def is_wave(hist):
    """올린 손목의 x 좌표(어깨 너비로 정규화)가 좌우로 2번 이상 방향을 바꾸면 손 흔들기"""
    xs = [x for _, x in hist]
    if len(xs) < 8:
        return False
    changes, last_dir, anchor = 0, 0, xs[0]
    for x in xs[1:]:
        d = x - anchor
        if abs(d) > 0.08:
            cur = 1 if d > 0 else -1
            if last_dir and cur != last_dir:
                changes += 1
            last_dir, anchor = cur, x
    return changes >= 2

def behavior(st, now):
    """정면 연속 시간·누적 시간, 이동 여부, 체류 시간으로 행동 패턴 판정"""
    h = [x for x in st["beh"] if now - x[0] <= BEH_WIN]
    if len(h) < 5:
        return "관찰 중"
    run = 0.0                                   # 끝에서부터 정면 연속 시간
    for i in range(len(h) - 1, 0, -1):
        if h[i][1] < FACE_TH:
            break
        run = h[-1][0] - h[i - 1][0]
    look = sum(h[i][0] - h[i - 1][0] for i in range(1, len(h)) if h[i][1] >= FACE_TH)
    moving = np.median([s for t, _, s in h if now - t < 2.0]) > MOVE_TH
    if run >= INTEREST_S and not moving:
        return "관심 있음"
    if moving and LOOK_S <= look < INTEREST_S:
        return "힐끗 봄"
    if moving:
        return "지나감"
    if now - st["first"] >= LINGER_S:
        return "머무름"
    return "관찰 중"

def majority(dq, now, window):
    vals = [v for t, v in dq if now - t <= window]
    if not vals:
        return None, 0.0
    v, n = collections.Counter(vals).most_common(1)[0]
    return v, n / len(vals)

def log_event(now, who, label):
    with ev_lock:
        events.appendleft({"ts": now, "who": who, "label": label})
        counts[label] += 1
    publish("event", {"ts": round(now, 2), "who": who, "label": label})

def close_track(tid, st, now):
    """추적이 끝난 사람의 요약을 남긴다"""
    if not args.behavior or st["best"] == "관찰 중":
        return
    dwell = st["seen"] - st["first"]
    with ev_lock:
        events.appendleft({"ts": now, "who": f"#{tid}",
                           "label": f"요약: 체류 {dwell:.0f}초, 정면 {st['look']:.0f}초 → {st['best']}"})
        beh_counts[st["best"]] += 1

def handshake_pairs(kps, ids, now):
    """두 사람의 손목이 가까이(어깨 너비 0.5배 이내) 있고 높이가 가슴~허리 사이면 악수 후보"""
    found = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = kps[i], kps[j]
            ba, bb = body(a), body(b)
            if ba is None or bb is None:
                continue
            sw = (ba[1] + bb[1]) / 2
            best = 1e9
            for wa in (LW, RW):
                for wb in (LW, RW):
                    if a[wa, 2] > KP_CONF and b[wb, 2] > KP_CONF:
                        mid_y = (a[wa, 1] + b[wb, 1]) / 2
                        if ba[0][1] < mid_y < ba[0][1] + 2.5 * ba[1]:
                            best = min(best, np.linalg.norm(a[wa, :2] - b[wb, :2]))
            key = tuple(sorted((ids[i], ids[j])))
            pairs.setdefault(key, collections.deque(maxlen=60)).append((now, best < 0.5 * sw))
            found.add(key)
    return found

def update_person(tid, k, now):
    st = tracks.setdefault(tid, new_track(now))
    dt = now - st["seen"]
    st["seen"] = now
    b = body(k)
    if b is None:
        return st
    raw = pose_label(k, b)
    fa = facing_score(k)
    spd = 0.0 if st["prev"] is None else float(np.linalg.norm(b[0] - st["prev"]) / b[1] / max(dt, 1e-3))
    st["prev"] = b[0]
    st["raw"].append((now, raw))
    st["beh"].append((now, fa, spd))
    if fa >= FACE_TH and st["n"] > 0:
        st["look"] += dt
    st["n"] += 1
    w = wave_wrist(k, b)                   # 손 흔들기용: 팔꿈치보다 올라간 손목의 x (어깨 중심 기준, 어깨 너비 단위)
    if w is not None:
        st["wave"].append((now, (k[w, 0] - b[0][0]) / b[1]))
    feat = frame_features(k, b, fa, spd)
    st["feat"].append(feat)
    if rec and tid == rec_tid:            # 화면에서 가장 크게 보이는 사람만 기록 (옆 사람이 같은 라벨로 섞이지 않게)
        rec.writerow([round(now, 3), tid, raw, round(fa, 3), round(spd, 3)] + feat[:51].round(3).tolist())

    # ---- 동작 (규칙) ----
    if MODE == "raw":
        label = raw
    else:
        v, frac = majority(st["raw"], now, SMOOTH_S)
        label = v if frac >= 0.6 else st["label"]
        recent = [(t, x) for t, x in st["wave"] if now - t <= WAVE_S]
        still_up = any(now - t <= 0.5 for t, _ in recent)   # 최근 0.5초 안에 손목이 팔꿈치 위에 있었음
        # 어깨 높이에서 흔들어도 잡히도록 "기본"에서도 판정. 한 번 판정되면 손을 내릴 때까지 유지 (깜빡임 방지)
        if label in ("기본", "한팔들기") and (is_wave(recent) or (st["label"] == "손흔들기" and still_up)):
            label = "손흔들기"

    # ---- 학습 모델 (5프레임마다) ----
    if models and st["n"] % 5 == 0:
        seq = list(st["feat"])
        for m in models:
            if len(seq) >= m["window"]:
                win = np.asarray(seq[-m["window"]:])
                # 두 손목이 보이는 프레임이 적으면 (손이 화면 밖·가려짐) 모델을 쓰지 않고 규칙 판정에 맡긴다.
                # 안 보이는 손목은 좌표가 0(어깨 중심)으로 지워져 팔짱처럼 보이기 때문
                hands = np.mean((win[:, 34 + LW] > KP_CONF) & (win[:, 34 + RW] > KP_CONF))
                if hands < ML_HANDS:
                    st["ml"][m["task"]] = None
                    continue
                lab, p = predict(m, win)
                st["ml"][m["task"]] = (lab, p) if p >= ML_TH else None
    ml_act = st["ml"].get("action")
    if ml_act:
        label = "기본" if ml_act[0] == "idle" else ml_act[0]

    if label != st["label"]:
        if label != "기본":
            log_event(now, f"#{tid}", label)
        st["label"] = label

    # ---- 행동 패턴 ----
    if args.behavior:
        bs = behavior(st, now)
        ml_beh = st["ml"].get("behavior")
        if ml_beh:
            bs = ml_beh[0]
        if bs != st["bstate"]:
            if bs != "관찰 중":
                log_event(now, f"#{tid}", bs)
            st["bstate"] = bs
        if BEH_RANK.get(bs, 1) > BEH_RANK.get(st["best"], 0):
            st["best"] = bs
    return st

def publish(name, data, retain=False):
    """보드에 설치된 mosquitto_pub으로 발행 (브로커가 꺼져 있어도 앱은 멈추지 않는다)"""
    if not args.mqtt_host:
        return
    cmd = ["mosquitto_pub", "-h", args.mqtt_host, "-p", str(args.mqtt_port), "-q", "1",
           "-t", f"{args.topic}/{name}", "-m", json.dumps(data, ensure_ascii=False)]
    subprocess.Popen(cmd + (["-r"] if retain else []), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def find_cam_ip():
    """DHCP로 카메라 주소가 바뀌어도 MAC 주소로 다시 찾는다 (재부팅 뒤 자동 복귀용)"""
    pings = [subprocess.Popen(["ping", "-c1", "-W1", f"10.42.0.{i}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for i in range(100, 131)]
    for p in pings:
        p.wait()
    out = subprocess.run(["ip", "neigh", "show", "dev", "enP8p1s0"], capture_output=True, text=True).stdout
    m = re.search(rf"^(\S+) lladdr {CAM_MAC}", out, re.M)
    return m.group(1) if m else None

def status_loop():
    """1분마다 상태 발행·CSV 기록, 15분마다 집계 발행. 값이 얼마나 오래됐는지 frame_age_s로 알린다"""
    rec_f = open(args.stats_log, "a", newline="") if args.stats_log else None
    rec_w = csv.writer(rec_f) if rec_f else None
    if rec_f and rec_f.tell() == 0:
        rec_w.writerow(["ts", "fps", "people", "reconnects", "frame_age_s", "mem_avail_mb"])
    t_sum, prev = time.time(), collections.Counter()
    while True:
        time.sleep(60)
        now = time.time()
        mem = int(re.search(r"MemAvailable:\s+(\d+)", open("/proc/meminfo").read()).group(1)) // 1024
        age = round(now - stats.get("last_frame", now), 1)
        row = {"ts": round(now), "fps": round(stats["fps"], 1), "people": stats["people"],
               "reconnects": stats.get("reconnects", 0), "frame_age_s": age, "mem_avail_mb": mem}
        publish("status", row, retain=True)
        if rec_w:
            rec_w.writerow(list(row.values())); rec_f.flush()
        if args.watchdog and age > args.watchdog:     # 읽기 호출이 멈추면 스레드로는 풀 수 없어 프로세스째 다시 시작
            print(f"프레임이 {age:.0f}초 동안 없음. 종료 후 재시작", flush=True)
            os._exit(1)
        if now - t_sum >= 900:
            with ev_lock:
                cur = counts + beh_counts
            publish("summary", {"ts": round(now), "window_s": 900, "counts": dict(cur - prev)}, retain=True)
            t_sum, prev = now, cur

def open_camera():
    if args.src == "rtsp":
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        url = f"rtsp://admin:{quote(os.environ['CAM_PW'], safe='')}@{CAM_IP}:554/{args.stream}"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(int(args.src), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    return cap if cap.isOpened() else None

class LatestFrame(threading.Thread):
    """별도 스레드가 카메라를 계속 읽고 최신 프레임만 보관한다 (RTSP 지연 누적 방지).
    처음에 못 열거나 도중에 끊기면 몇 초마다 다시 연결을 시도한다."""
    def __init__(self):
        super().__init__(daemon=True)
        self.cap, self.lock, self.frame, self.seq, self.t_ok = None, threading.Lock(), None, 0, 0.0

    def run(self):
        global CAM_IP
        while True:
            if self.cap is None:
                self.cap = open_camera()
                if self.cap is None:
                    print(f"카메라를 열 수 없음 ({CAM_IP}). 3초 뒤 다시 시도")
                    if args.src == "rtsp":
                        CAM_IP = find_cam_ip() or CAM_IP
                    time.sleep(3)
                    continue
                self.t_ok = time.time()
            ok, f = self.cap.read()
            if not ok:
                if time.time() - self.t_ok > 2.5:  # 실패 한 번이 30초씩 걸릴 수 있어 횟수가 아니라 시간으로 판단
                    print("카메라 재연결 중...")
                    stats["reconnects"] = stats.get("reconnects", 0) + 1
                    self.cap.release()
                    self.cap = None
                time.sleep(0.05)
                continue
            self.t_ok = time.time()
            with self.lock:
                self.frame, self.seq = f, self.seq + 1

    def get(self, last_seq):
        with self.lock:
            return (self.frame, self.seq) if self.seq != last_seq else (None, last_seq)

def finish_file():
    """영상 파일 입력이 끝나면 남은 사람을 정리하고 횟수를 출력한 뒤 종료"""
    now = max([s["seen"] for s in tracks.values()], default=time.time()) + 10
    for tid in list(tracks):
        close_track(tid, tracks.pop(tid), now)
    if rec:
        rec_f.flush()
    print(f"\n===== 영상 끝 [{MODE}{' +behavior' if args.behavior else ''}] {args.src} =====")
    print("동작별 횟수:", dict(counts))
    if args.behavior:
        print("행동 패턴 (사람 단위):", dict(beh_counts))
    sys.stdout.flush()
    os._exit(0)

def capture_loop():
    global latest_jpeg
    if FILE:   # 영상 파일: 프레임을 빠짐없이 순서대로 읽고, 시간은 영상 속 시간으로 계산
        cap = cv2.VideoCapture(args.src)
        assert cap.isOpened(), f"영상을 열 수 없음: {args.src}"
        vfps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        print(f"영상 입력: {args.src} ({vfps:.1f} fps, {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}프레임)")
        t_base = time.time()
    else:
        reader = LatestFrame()
        reader.start()
    t0, n, last = time.time(), 0, 0
    while True:
        if FILE:
            ok, frame = cap.read()
            if not ok:
                finish_file()
            last += 1
            now = t_base + last / vfps
        else:
            frame, last = reader.get(last)
            if frame is None:
                time.sleep(0.01); continue
            now = time.time()
        r = det.track(frame, persist=True, conf=args.conf, verbose=False)[0]
        vis = r.plot(labels=False)        # 박스 + 관절 뼈대
        ids, kps, boxes = [], [], []
        if r.boxes.id is not None and r.keypoints is not None:
            ids = r.boxes.id.int().tolist()
            kps = np.concatenate([r.keypoints.xy.cpu().numpy(),
                                  r.keypoints.conf.cpu().numpy()[..., None]], axis=-1)
            boxes = r.boxes.xyxy.int().tolist()
        stats["people"], stats["last_frame"] = len(ids), now
        global rec_tid
        rec_tid = ids[int(np.argmax([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]))] if ids else None

        for tid, k, (x1, y1, x2, y2) in zip(ids, kps, boxes):
            st = update_person(tid, k, now)
            txt = f"#{tid} {EN.get(st['label'], st['label'])}"
            if args.behavior:
                txt += f" | {EN.get(st['bstate'], st['bstate'])}"
            color = (0, 200, 255) if st["label"] != "기본" or st["bstate"] == "관심 있음" else (0, 255, 0)
            cv2.putText(vis, txt, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ---- 2인 동작 (악수) ----
        active = handshake_pairs(kps, ids, now) if len(ids) >= 2 else set()
        for key in list(pairs):
            if key not in active:
                pairs.pop(key); pair_on.discard(key); continue
            if MODE == "raw":
                on = pairs[key][-1][1]
            else:
                v, frac = majority(pairs[key], now, SMOOTH_S)
                on = bool(v) and frac >= 0.6
            if on and key not in pair_on:
                log_event(now, f"#{key[0]}+#{key[1]}", "악수")
            (pair_on.add if on else pair_on.discard)(key)
        for a, b_ in pair_on:
            if a in ids and b_ in ids:
                ba, bb = boxes[ids.index(a)], boxes[ids.index(b_)]
                x, y = (ba[0] + bb[2]) // 2 - 80, min(ba[1], bb[1]) + 30
                cv2.putText(vis, EN["악수"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

        # 3초 이상 안 보인 사람은 추적 종료 + 요약
        for tid in [t for t, s in tracks.items() if now - s["seen"] > 3]:
            close_track(tid, tracks.pop(tid), now)

        n += 1
        if now - t0 >= 1:
            stats["fps"], t0, n = n / (now - t0), now, 0
            if rec:
                rec_f.flush()
        cv2.putText(vis, f"{stats['fps']:.1f} fps | {MODE} | people {stats['people']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with frame_lock:
                latest_jpeg = buf.tobytes()

app = FastAPI()

@app.get("/stream")
def stream():
    def gen():
        while True:
            with frame_lock:
                f = latest_jpeg
            if f:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/events")
def get_events():
    with ev_lock:
        total = sum(beh_counts.values())
        return {"mode": MODE + (" +behavior" if args.behavior else "") + (" +model" if models else ""),
                "stats": stats, "counts": dict(counts), "events": list(events)[:20],
                "beh": dict(beh_counts),
                "interest_rate": round(beh_counts["관심 있음"] / total, 2) if total else None,
                "now": {f"#{t}": s["label"] + (f" / {s['bstate']}" if args.behavior else "")
                        for t, s in tracks.items()}}

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Jetson 동작 인식</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
img{width:100%;max-width:960px;border-radius:8px}table{border-collapse:collapse;margin-top:8px}
td,th{border:1px solid #444;padding:4px 10px;font-size:14px}.chip{display:inline-block;background:#333;
border-radius:12px;padding:4px 10px;margin:2px}</style>
<h2>실시간 영상 <span id=mode></span></h2><img src="/stream">
<h2>현재 상태</h2><div id=now></div>
<h2>동작별 횟수</h2><div id=cnt></div>
<h2>관심도 통계 (사람 단위)</h2><div id=beh></div>
<h2>동작 기록</h2><table><thead><tr><th>시각</th><th>대상</th><th>동작</th></tr></thead><tbody id=ev></tbody></table>
<script>
const chips=o=>Object.entries(o).map(([k,v])=>`<span class=chip>${k} ${v}</span>`).join('')||'-';
async function poll(){const r=await (await fetch('/events')).json();
 document.getElementById('mode').textContent=`[${r.mode}] ${r.stats.fps.toFixed(1)} fps, ${r.stats.people}명`;
 document.getElementById('now').innerHTML=Object.entries(r.now).map(([k,v])=>`<span class=chip>${k}: ${v}</span>`).join('')||'사람 없음';
 document.getElementById('cnt').innerHTML=chips(r.counts);
 document.getElementById('beh').innerHTML=chips(r.beh)+(r.interest_rate!==null?` <b>관심 비율 ${(r.interest_rate*100).toFixed(0)}%</b>`:'');
 document.getElementById('ev').innerHTML=r.events.map(e=>`<tr><td>${new Date(e.ts*1000).toLocaleTimeString()}</td><td>${e.who}</td><td>${e.label}</td></tr>`).join('');}
setInterval(poll,1000);poll();
</script>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    if args.mqtt_host or args.stats_log:
        threading.Thread(target=status_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=args.port)
