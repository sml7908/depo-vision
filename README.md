# Depo-Vision

PVD AlCu 증착 품질 예측 및 공정 모니터링 대시보드 (제조 AX 아카데미 AX 2조)

## 바로 가기

**https://depo-vision-a20kyll.streamlit.app**

<img src="qr.png" width="220" alt="Depo-Vision 접속 QR">

## 화면 구성

| 화면 | 내용 |
|---|---|
| 설비 SPC 모니터링 | 설비별 주간 평균 관리도, 이탈·치우침 감지 |
| Thickness · Rs 예측 | 계측 전 17개 지점 예측, 비저항 위험 등급 4단계 |
| What-if 시뮬레이션 | Target Life · Deposition Time 변경 시 결과 예측, 보정값 제안 |

## 구성 파일

- `app.py` : Streamlit 화면
- `logic.py` : 예측·SPC 계산
- `data/` : 원본 데이터, 전처리 데이터
- `models/` : 두께·면저항 17개 지점 예측 모델 (LightGBM)
