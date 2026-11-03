# Jetson 카메라 응용 실습

AIGJ_01_008 박민수 · 측정 11/2~11/3 · 트랙 기본(A), 일부 심화(B)

## 제출물

| 제출물 | 파일 | 내용 |
| --- | --- | --- |
| 기능 시험 기록 | [runs/test_log.md](runs/test_log.md) | T0~T4 항목별 조건·결과·판정 |
| 결과 보고서 | [REPORT.md](REPORT.md) | 과제 1~6 산출물 요약 (표·사진 중심) |
| 확인 문제 답안 | [ANSWERS.md](ANSWERS.md) | 1-1~6-5 |
| 응용 서비스 | 팀 Jetson에서 가동 중 | A1 재실 인원(`occupancy`, 8095), A3 안내판 행동(`pose-service`, 8001). 토픽·조회 주소는 결과 형식 문서 |
| 결과 형식 문서 | [INTERFACE.md](INTERFACE.md) | 다른 팀용 MQTT 토픽·필드·단위·예시·조회 주소 |
| 상세 기록 | [docs/DETAILS.md](docs/DETAILS.md) | 시험마다의 전체 과정, 재시험, 실패 기록 |
| 발표 | (자료 별도) | 핵심 결과와 "설계 변수 하나를 바꿔 가장 크게 달라진 것"은 REPORT.md 7장 |

## 과제별 상태

| 과제 | 상태 |
| --- | --- |
| 1. 카메라 연동과 기준선 | 완료 |
| 2. 물체 인식 | 완료 (FP16 추론 5.1 ms, 선 통과 10/10과 MQTT, 물품 3방식 비교) |
| 3. 재실 인원 카운터 | 완료 (빈 방 오판 0%, MAE 0.00~0.40) |
| 4. 행동 패턴 인식 | 완료 (만세 10/10, 한팔 정지 오판 1회, 행동 11/12) |
| 5. 거리 측정 | 완료 (웹캠 1대 이동 방식, 1.4 m −3.6%) |
| 6. 응용 서비스 | 완료 (12시간, 카메라 분리, 재부팅 2분 안 복귀) |

## 코드

| 파일 | 용도 |
| --- | --- |
| `code/occupancy_service.py`, `code/occupancy.service` | A1 재실 인원 서비스 |
| `code/pose_app.py`, `code/pose_model.py`, `code/pose-service.service` | A3 행동 패턴 서비스 |
| `code/cam1.py` | 과제 1 수신·지연 측정 |
| `code/gst_decode_bench.py` | 과제 1 심화, NVDEC와 CPU 디코딩 비교 |
| `code/light_dist.py` | 과제 2 조명×거리 측정 |

카메라 비밀번호는 Jetson의 설정 파일에서 읽으며 저장소에 넣지 않는다. 영상은 넣지 않았고, 사람이 나온 사진은 흐림 처리했다.
