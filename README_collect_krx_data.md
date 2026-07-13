# KRX 데이터 자동 수집 스크립트 (collect_krx_data.py)

위클리옵션 프로젝트에서 필요한 국내 현물·지수 데이터를 pykrx로 자동 수집하는 스크립트입니다.

## 이 스크립트가 받는 데이터

1. **4개 종목 현물 일별 OHLCV** — 삼성전자·SK하이닉스·현대차·LG에너지솔루션 (H1 핵심: 위클리옵션이 본주식 변동성에 미치는 영향 검증용 라벨 데이터)
2. **4개 종목 투자자별(개인/외국인/기관) 순매수 거래량** — H3·H7 수급 집중도 분석용
3. **4개 종목 외국인 보유비중 추이** — H7 보조 피처
4. **코스피200 지수 시세** — 기준 모델 학습 데이터

결과는 `krx_data/` 폴더 안에 `stock_ohlcv.csv`, `investor_trading.csv`, `foreign_ownership.csv`, `kospi200_index.csv` 4개 파일로 저장됩니다.

## ⚠️ 이 스크립트가 못 받는 것

옵션·선물(파생상품) 데이터, 파생상품 투자자별 거래실적, PCR, VKOSPI 등은 pykrx가 지원하지 않습니다. 이 데이터들은 KRX Open API(openapi.krx.co.kr) 또는 정보데이터시스템(data.krx.co.kr)에서 수동으로 받아야 합니다.

## 실행 방법

### 1. 사전 준비물 확인

```
python --version
```

버전이 안 뜨면 [python.org](https://www.python.org)에서 파이썬을 먼저 설치하세요. 설치 시 **"Add to PATH"** 체크박스를 꼭 체크해야 합니다.

### 2. 필요한 패키지 설치

```
pip install pykrx pandas
```

이미 설치돼 있다면 최신 버전으로 업데이트하는 걸 권장합니다.

```
pip install --upgrade pykrx
```

### 3. 스크립트 실행

`collect_krx_data.py` 파일이 있는 폴더로 이동한 뒤 실행합니다.

```
cd Downloads
python collect_krx_data.py
```

실행하면 진행 상황이 콘솔에 출력되고, 완료되면 같은 폴더 안에 `krx_data/` 폴더가 생기고 그 안에 CSV 4개가 저장됩니다.

## 알려진 이슈

KRX 서버가 짧은 시간에 연속 요청을 받으면 빈 응답(`Expecting value: line 1 column 1 (char 0)`)을 주면서 일부 요청을 막는 경우가 있습니다. 스크립트에 재시도(최대 4회) + 대기시간 증가 로직이 내장돼 있지만, 그래도 계속 실패하면 해당 데이터(투자자별 거래실적, 외국인 보유비중, 코스피200 지수)는 data.krx.co.kr 통계 메뉴에서 수동으로 받아도 됩니다 — 이 스크립트의 핵심 목적인 4종목 현물 주가(`stock_ohlcv.csv`) 수집은 정상적으로 동작합니다.

## 설정 변경

스크립트 상단의 다음 값들을 필요에 따라 수정할 수 있습니다.

- `TICKERS`: 수집할 종목 코드/이름
- `START`, `END`: 수집 기간 (YYYYMMDD)
- `SLEEP_SEC`, `MAX_RETRY`: 요청 간 대기시간과 재시도 횟수
