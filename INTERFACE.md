# 결과 형식 문서 — 교육장 카메라 서비스 v1

AIGJ_01_008 박민수 · 팀 Jetson(team2)에서 가동 중. 다른 팀이 이 문서만 보고 구독·조회할 수 있게 쓴다. 영상과 사진은 보드 밖으로 내보내지 않고, 숫자와 이벤트만 낸다.

## 서비스 목록

| 서비스 | 하는 일 | systemd 유닛 | 조회 포트 | 카메라 |
| --- | --- | --- | --- | --- |
| A1 재실 인원 | 방에 몇 명이 있는지 | `occupancy` | 8095 | VIGI C540-W (RTSP 서브 스트림) |
| A3 안내판 행동 | 앞의 사람이 지나가는지, 멈춰 보는지, 손짓하는지 | `pose-service` | 8001 | VIGI C540-W (RTSP 서브 스트림 848x480) |

브로커는 보드의 mosquitto(포트 1883)다. 보드 주소는 팀 내부 공지로만 알린다(아래 `<보드 주소>`). 두 서비스는 부팅 때 자동으로 시작한다.

## MQTT 토픽

| 토픽 | 서비스 | 내용 | 갱신 주기 | retain |
| --- | --- | --- | --- | --- |
| `aiot/space/lab1/occupancy` | A1 | 현재 재실 인원 | 10초 | 예 |
| `aiot/service/occupancy/status` | A1 | 서비스 온라인·오프라인 | 시작·종료 때 | 예 |
| `aiot/kiosk/lab1/event` | A3 | 행동·동작 이벤트 | 일어날 때 즉시 | 아니오 |
| `aiot/kiosk/lab1/status` | A3 | 처리 상태 | 60초 | 예 |
| `aiot/kiosk/lab1/summary` | A3 | 최근 구간 집계 | 900초 | 예 |
| `aiot/vision/test/event` | 선 통과 도구 (시험용) | IN/OUT 1회마다 `ts`, `dir`, `id`, `in`, `out` | 통과할 때 즉시 | 아니오 |

retain 토픽은 새로 구독하면 바로 마지막 값을 받는다.

## 필드

**`aiot/space/lab1/occupancy`**

| 필드 | 단위·형식 | 뜻 |
| --- | --- | --- |
| `ts` | ISO 8601, KST | 값을 만든 시각 |
| `space` | 문자열 | 공간 이름 (`lab1`) |
| `count` | 명 | 최근 30초 중간값. **제어에는 이 값을 쓴다** |
| `raw` | 명 | 그 순간의 검출 수. 진단용 |
| `unit` | 문자열 | `명` |
| `window_s`, `stat` | 초, 문자열 | 평활 창(30)과 방식(`median`) |
| `age_s` | 초 | 마지막 프레임이 얼마나 오래됐는지 |
| `camera` | 문자열 | `정상` 또는 `끊김(프레임 없음)` |
| `reconnects` | 회 | 시작 뒤 카메라 재연결 횟수 |

`camera`가 `정상`이 아니거나 `age_s`가 5초를 넘으면 `count`를 믿지 않는다. 사람이 들어오고 약 15초 뒤에 `count`가 바뀐다.

**`aiot/kiosk/lab1/event`**

| 필드 | 단위·형식 | 뜻 |
| --- | --- | --- |
| `ts` | 유닉스 초 | 이벤트 시각 |
| `who` | 문자열 | 추적 번호 (`#16`, 두 사람이면 `#16+#17`) |
| `label` | 문자열 | 아래 라벨 중 하나, 또는 `요약: 체류 N초, 정면 M초 → 판정` |

라벨: 행동 `지나감`, `힐끗 봄`, `관심 있음`, `머무름` / 동작 `만세`, `한팔들기`, `손흔들기`, `악수`. 추적 번호는 서비스가 다시 시작하면 처음부터 다시 매긴다. 사람을 식별하는 값이 아니다.

**`aiot/kiosk/lab1/status`**: `ts`(유닉스 초), `fps`(초당 프레임), `people`(화면 속 사람 수), `reconnects`(회), `frame_age_s`(초), `mem_avail_mb`(보드 남은 메모리, MB).

**`aiot/kiosk/lab1/summary`**: `ts`, `window_s`(900), `counts`(라벨별 횟수).

## 예시

```
aiot/space/lab1/occupancy {"ts": "2026-11-03T14:46:45+09:00", "space": "lab1", "count": 1, "raw": 1, "unit": "명", "window_s": 30.0, "stat": "median", "age_s": 0.0, "camera": "정상", "reconnects": 0}
aiot/kiosk/lab1/event {"ts": 1793692476.19, "who": "#10", "label": "지나감"}
```

두 줄 모두 11/3 실제 기록에서 가져왔다(재실 14:46, 이벤트 16:54). 재실 줄은 서비스 기록 파일의 같은 필드다.

## 조회 주소 (REST)

| 주소 | 돌려주는 것 |
| --- | --- |
| `http://<보드 주소>:8095/occupancy` | 위 occupancy 메시지와 같은 JSON |
| `http://<보드 주소>:8095/health` | 가동 시간, MQTT 연결, 카메라 상태, 추론 시간, CPU·메모리·GPU |
| `http://<보드 주소>:8001/events` | A3 처리 상태, 라벨별 누적 횟수, 최근 이벤트 20개 |

## 구독 예

```bash
mosquitto_sub -h <보드 주소> -t 'aiot/space/lab1/occupancy' -v
mosquitto_sub -h <보드 주소> -t 'aiot/kiosk/lab1/#' -v
curl -s http://<보드 주소>:8095/occupancy
```

## 장애 때 동작

카메라가 끊기면 A1은 약 6초 안에 `camera`를 `끊김(프레임 없음)`으로 바꾸고, 카메라가 돌아오면 재시작 없이 다시 연결한다. A3는 2.5초 동안 프레임이 없으면 다시 연결하고, 120초 넘게 없으면 스스로 종료해 systemd가 다시 띄운다. 보드를 재부팅하면 A1은 약 1분, A3는 2분 안에 돌아왔다(11/3 시험). MJPEG 뷰어(8001)는 시험용이라 운영 때는 끄거나 실습망 안으로 제한한다.
