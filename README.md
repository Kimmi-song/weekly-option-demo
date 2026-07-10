# 📊 위클리 옵션 변동성 예측 대시보드

> 아시아권 옵션 시장 크로스마켓 전이학습 · 다중 시간 지평 모델 · 백테스팅 전략 검증

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://weekly-option-demo-nfawmucc4cwkyjgonouqqy.streamlit.app/)

---

## 🔍 프로젝트 개요

2026년 6월 국내 개별주식 위클리 옵션 도입을 위한 파생상품시장 업무규정이 시행되었으나, 삼성전자·SK하이닉스·현대차·LG에너지솔루션 4개 종목의 위클리 옵션 상장은 **시장 변동성 확대 우려**로 연기된 상태입니다.

본 프로젝트는 이 문제를 해결하기 위해:

- 위클리 옵션이 이미 정착된 **아시아권 시장(홍콩·일본·대만)의 변동성 패턴을 학습**
- 국내 도입 예정 4개 종목에 적용하여 **변동성을 사전 예측**
- **백테스팅 기반 투자 전략 검증**까지 수행하는 변동성 조기경보 모델을 제안합니다.

> "예측할 데이터가 없어서 상장이 미뤄지고, 상장이 미뤄져서 데이터가 쌓이지 않는" 악순환을 끊기 위한 **크로스마켓 전이학습 접근**입니다.

---

## 🚀 데모

👉 **[라이브 대시보드 보기](https://weekly-option-demo-nfawmucc4cwkyjgonouqqy.streamlit.app/)**

> ⚠️ 현재 데모는 공모전 시연용 프로토타입입니다. 실제 모델 학습 결과는 최종 보고서를 참조하세요.

---

## 📁 구성

```
weekly-option-demo/
├── weekly_option_demo.py   # Streamlit 대시보드 메인 코드
└── requirements.txt        # 패키지 의존성
```

---

## 🗂 대시보드 탭 구성

| 탭                      | 내용                                                            |
| ----------------------- | --------------------------------------------------------------- |
| 🌏 아시아권 변동성 비교 | VKOSPI vs VHSI·VJPX·VTWN 비교, 상관관계 히트맵, VIX 선행 시그널 |
| 🤖 모델 예측 결과       | XGBoost·TabNet 고변동성 예측 시그널, 모델 성능 비교(AUC·F1)     |
| ⏱ 시간 지평 비교        | 주간 vs 월간 예측력 비교, 만기일 효과 분석(H3)                  |
| 💰 백테스팅 결과        | 모델 전략 vs Buy&Hold vs 이동평균 누적 수익률·샤프비율·MDD 비교 |

---

## 🔬 핵심 가설

| 가설 | 내용                                                          |
| ---- | ------------------------------------------------------------- |
| H1   | 아시아권 학습 모델이 국내 지수형 위클리옵션에 이전 가능       |
| H2   | 중장기(월간) 모델의 예측력이 단기(주간)보다 높음              |
| H3   | 만기일 직전 주간의 변동성이 일반 주간보다 유의미하게 높음     |
| H4   | PCR 임계값 이상 주간에 다음 주 변동성이 높음                  |
| H5   | 모델 기반 전략이 Buy&Hold 대비 위험 대비 수익 우위            |
| H6   | 소규모 데이터 환경에서 TabNet이 일반 딥러닝보다 과적합에 강함 |

---

## 🛠 기술 스택

- **Frontend**: Streamlit, Plotly
- **Data**: yfinance, pykrx, KRX 정보데이터시스템
- **Model**: XGBoost, TabNet
- **Language**: Python 3.10+

---

## ⚙️ 로컬 실행

```bash
# 패키지 설치
pip install -r requirements.txt

# 실행
streamlit run weekly_option_demo.py
```

> 로컬 실행 시 yfinance를 통해 실제 VIX·VHSI 데이터가 자동으로 로드됩니다.

---

## 📊 데이터 출처

- [KRX 정보데이터시스템](https://data.krx.co.kr) — VKOSPI, 옵션 거래·미결제약정
- [Yahoo Finance](https://finance.yahoo.com) — VHSI(홍콩), VIX(미국)
- [pykrx](https://github.com/sharebook-kr/pykrx) — 국내 종목 주가·수급 데이터

---

_본 프로젝트는 iM:POSSIBLE Challenger 공모전 출품작입니다._
