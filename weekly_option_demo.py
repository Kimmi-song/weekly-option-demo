import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(
    page_title="위클리 옵션 변동성 예측 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 16px 20px; margin: 4px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
    .metric-label { font-size: 0.85rem; color: #8b949e; margin-bottom: 4px; }
    .metric-delta-pos { color: #3fb950; font-size: 0.85rem; }
    .metric-delta-neg { color: #f85149; font-size: 0.85rem; }
    .section-header {
        border-left: 3px solid #58a6ff; padding-left: 12px;
        margin: 24px 0 16px 0; font-size: 1.1rem;
        font-weight: 600; color: #e6edf3;
    }
</style>
""", unsafe_allow_html=True)

# ─── 더미 데이터 생성 ──────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="W-THU")
    n = len(dates)

    vkospi = 20 + np.cumsum(np.random.randn(n) * 0.8) + np.sin(np.arange(n) * 0.3) * 5
    vkospi = np.clip(vkospi, 10, 50)

    # 아시아권 국가별 변동성 (각각 다른 시드로 변화 패턴 차별화)
    np.random.seed(10)
    vhsi = 22 + np.cumsum(np.random.randn(n) * 0.9) + np.sin(np.arange(n) * 0.25) * 6
    vhsi = np.clip(vhsi, 12, 55)

    np.random.seed(20)
    vjpx = 18 + np.cumsum(np.random.randn(n) * 0.7) + np.sin(np.arange(n) * 0.35) * 4
    vjpx = np.clip(vjpx, 8, 45)

    np.random.seed(30)
    vtwn = 21 + np.cumsum(np.random.randn(n) * 0.85) + np.sin(np.arange(n) * 0.2) * 5
    vtwn = np.clip(vtwn, 10, 50)

    pcr = np.clip(0.8 + np.random.randn(n) * 0.15, 0.4, 1.5)
    oi_change = np.random.randn(n) * 0.05
    high_vol = (vkospi > 28).astype(int)

    pred_xgb = high_vol.copy()
    pred_xgb[np.random.choice(n, size=int(n * 0.12), replace=False)] ^= 1

    pred_tab = high_vol.copy()
    pred_tab[np.random.choice(n, size=int(n * 0.15), replace=False)] ^= 1

    return pd.DataFrame({
        "date": dates,
        "vkospi": vkospi, "vhsi": vhsi, "vjpx": vjpx, "vtwn": vtwn,
        "pcr": pcr, "oi_change": oi_change,
        "high_vol": high_vol, "pred_xgb": pred_xgb, "pred_tab": pred_tab,
    })

@st.cache_data
def generate_backtest():
    np.random.seed(99)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="W-THU")
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "model": np.cumprod(1 + np.random.randn(n) * 0.012 + 0.003),
        "buy_hold": np.cumprod(1 + np.random.randn(n) * 0.015 + 0.001),
        "moving_avg": np.cumprod(1 + np.random.randn(n) * 0.013 + 0.0015),
    })

df = generate_data()
bt = generate_backtest()

LAYOUT = dict(
    plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
    font=dict(color="#e6edf3"),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    hovermode="x unified", margin=dict(t=20, b=40)
)

# ─── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 분석 설정")
    st.markdown("---")

    ticker = st.selectbox("📌 종목 선택", [
        "삼성전자 (005930)", "SK하이닉스 (000660)",
        "현대차 (005380)", "LG에너지솔루션 (373220)"
    ])

    horizon = st.radio("📅 시간 지평", ["단기 (주간)", "중장기 (월간)"])

    model_choice = st.multiselect(
        "🤖 모델 선택", ["XGBoost", "TabNet"],
        default=["XGBoost", "TabNet"]
    )

    countries = st.multiselect(
        "🌏 비교 국가 선택",
        ["🇭🇰 홍콩 (VHSI)", "🇯🇵 일본 (VJPX)", "🇹🇼 대만 (VTWN)"],
        default=["🇭🇰 홍콩 (VHSI)", "🇯🇵 일본 (VJPX)", "🇹🇼 대만 (VTWN)"]
    )

    st.markdown("---")
    st.markdown("**데이터 출처**")
    st.caption("KRX 정보데이터시스템")
    st.caption("Yahoo Finance (yfinance)")
    st.caption("pykrx")

# ─── 헤더 ─────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:1.8rem; margin-bottom:4px;'>
📊 위클리 옵션 변동성 예측 대시보드
</h1>
<p style='color:#8b949e; margin-bottom:24px;'>
아시아권 옵션 시장 크로스마켓 전이학습 · 다중 시간 지평 모델 · 백테스팅 전략 검증
</p>
""", unsafe_allow_html=True)

# ─── KPI 카드 ─────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, label, val, delta in zip(
    [c1, c2, c3, c4],
    ["XGBoost AUC", "TabNet AUC", "모델 전략 샤프비율", "최대 낙폭 (MDD)"],
    ["0.81", "0.77", "1.24", "-8.3%"],
    ["▲ +0.06 vs 기준모델", "▲ +0.02 vs 기준모델", "▲ +0.41 vs Buy&Hold", "▼ -6.2%p vs Buy&Hold"]
):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{val}</div>
        <div class='metric-delta-pos'>{delta}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── 탭 ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🌏 아시아권 변동성 비교",
    "🤖 모델 예측 결과",
    "⏱ 시간 지평 비교",
    "💰 백테스팅 결과"
])

# ── Tab 1: 아시아권 비교 ────────────────────────────────────
with tab1:
    st.markdown("<div class='section-header'>국가별 변동성 지수 비교 — VKOSPI (한국) 기준</div>", unsafe_allow_html=True)

    # 전체 비교 차트
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["date"], y=df["vkospi"],
        name="🇰🇷 VKOSPI (한국)",
        line=dict(color="#58a6ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.06)"
    ))
    country_map = {
        "🇭🇰 홍콩 (VHSI)": ("vhsi", "#f0b429", "VHSI"),
        "🇯🇵 일본 (VJPX)": ("vjpx", "#3fb950", "VJPX"),
        "🇹🇼 대만 (VTWN)": ("vtwn", "#bc8cff", "VTWN"),
    }
    for c in countries:
        col_key, color, label = country_map[c]
        fig1.add_trace(go.Scatter(
            x=df["date"], y=df[col_key],
            name=f"{c.split(' ')[0]} {label}",
            line=dict(color=color, width=1.8, dash="dot")
        ))
    fig1.add_hline(y=28, line_dash="dash", line_color="#f85149", opacity=0.5,
                   annotation_text="고변동성 임계값 (28)", annotation_font_color="#f85149")
    fig1.update_layout(**{**LAYOUT, "height": 360, "yaxis": {**LAYOUT["yaxis"], "title": "변동성 지수"}})
    st.plotly_chart(fig1, use_container_width=True)

    # 국가별 서브탭
    st.markdown("<div class='section-header'>국가별 상세 분석</div>", unsafe_allow_html=True)
    sub_hk, sub_jp, sub_tw = st.tabs(["🇭🇰 홍콩", "🇯🇵 일본", "🇹🇼 대만"])

    def country_detail(subtab, col_key, label, color, corr_val, avg_val, threshold):
        with subtab:
            col_left, col_right = st.columns([2, 1])
            with col_left:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df["vkospi"],
                    name="VKOSPI (한국)", line=dict(color="#58a6ff", width=1.5)
                ))
                fig.add_trace(go.Scatter(
                    x=df["date"], y=df[col_key],
                    name=f"{label}", line=dict(color=color, width=2)
                ))
                fig.add_hline(y=threshold, line_dash="dash", line_color="#f85149",
                              opacity=0.5, annotation_text=f"고변동성 임계값 ({threshold})")
                fig.update_layout(**{**LAYOUT, "height": 300,
                                     "yaxis": {**LAYOUT["yaxis"], "title": "변동성 지수"}})
                st.plotly_chart(fig, use_container_width=True)
            with col_right:
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric(f"{label} 평균", f"{avg_val:.1f}")
                st.metric("VKOSPI 상관계수", f"{corr_val:.2f}",
                          "높을수록 학습 전이 유효")
                high_pct = (df[col_key] > threshold).mean() * 100
                st.metric("고변동성 빈도", f"{high_pct:.1f}%")

    country_detail(sub_hk, "vhsi", "VHSI (홍콩)", "#f0b429",
                   np.corrcoef(df["vkospi"], df["vhsi"])[0,1],
                   df["vhsi"].mean(), 28)
    country_detail(sub_jp, "vjpx", "VJPX (일본)", "#3fb950",
                   np.corrcoef(df["vkospi"], df["vjpx"])[0,1],
                   df["vjpx"].mean(), 25)
    country_detail(sub_tw, "vtwn", "VTWN (대만)", "#bc8cff",
                   np.corrcoef(df["vkospi"], df["vtwn"])[0,1],
                   df["vtwn"].mean(), 27)

    # 상관관계 히트맵
    st.markdown("<div class='section-header'>국가간 변동성 상관관계 히트맵</div>", unsafe_allow_html=True)
    corr_df = df[["vkospi", "vhsi", "vjpx", "vtwn"]].rename(columns={
        "vkospi": "🇰🇷 한국", "vhsi": "🇭🇰 홍콩",
        "vjpx": "🇯🇵 일본", "vtwn": "🇹🇼 대만"
    }).corr()
    fig_corr = px.imshow(
        corr_df, text_auto=".2f", color_continuous_scale="Blues",
        zmin=0, zmax=1
    )
    fig_corr.update_layout(
        plot_bgcolor="#161b22", paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3"), height=320, margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-header'>PCR (풋/콜 비율)</div>", unsafe_allow_html=True)
        fig_pcr = go.Figure()
        fig_pcr.add_trace(go.Bar(
            x=df["date"], y=df["pcr"],
            marker_color=np.where(df["pcr"] > 1.0, "#f85149", "#3fb950")
        ))
        fig_pcr.add_hline(y=1.0, line_dash="dash", line_color="#8b949e",
                           opacity=0.7, annotation_text="PCR = 1.0")
        fig_pcr.update_layout(**{**LAYOUT, "height": 260, "showlegend": False})
        st.plotly_chart(fig_pcr, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-header'>미결제약정 주간 변화율</div>", unsafe_allow_html=True)
        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(
            x=df["date"], y=df["oi_change"] * 100,
            marker_color=np.where(df["oi_change"] > 0, "#58a6ff", "#f85149")
        ))
        fig_oi.update_layout(**{**LAYOUT, "height": 260, "showlegend": False,
                                 "yaxis": {**LAYOUT["yaxis"], "title": "변화율 (%)"}})
        st.plotly_chart(fig_oi, use_container_width=True)

# ── Tab 2: 모델 예측 결과 ───────────────────────────────────
with tab2:
    st.markdown(f"<div class='section-header'>고변동성 예측 — {ticker.split(' ')[0]} · {horizon}</div>",
                unsafe_allow_html=True)

    recent = df.tail(40).copy()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=recent["date"], y=recent["vkospi"],
        name="실제 VKOSPI", line=dict(color="#8b949e", width=1.5)
    ))
    if "XGBoost" in model_choice:
        mask = recent["pred_xgb"] == 1
        fig2.add_trace(go.Scatter(
            x=recent[mask]["date"], y=recent[mask]["vkospi"],
            mode="markers", name="XGBoost 고변동성 예측",
            marker=dict(symbol="triangle-up", size=10, color="#58a6ff")
        ))
    if "TabNet" in model_choice:
        mask2 = recent["pred_tab"] == 1
        fig2.add_trace(go.Scatter(
            x=recent[mask2]["date"], y=recent[mask2]["vkospi"],
            mode="markers", name="TabNet 고변동성 예측",
            marker=dict(symbol="diamond", size=8, color="#f0b429")
        ))
    fig2.add_hline(y=28, line_dash="dash", line_color="#f85149",
                   opacity=0.5, annotation_text="고변동성 임계값")
    fig2.update_layout(**{**LAYOUT, "height": 360,
                           "yaxis": {**LAYOUT["yaxis"], "title": "VKOSPI"}})
    st.plotly_chart(fig2, use_container_width=True)

    perf_df = pd.DataFrame({
        "모델": ["XGBoost", "TabNet", "기준모델 (Logistic)"],
        "정밀도": [0.83, 0.79, 0.71],
        "재현율": [0.78, 0.74, 0.68],
        "F1-Score": [0.80, 0.76, 0.69],
        "AUC": [0.81, 0.77, 0.75],
    })
    st.markdown("<div class='section-header'>모델 성능 비교</div>", unsafe_allow_html=True)
    st.dataframe(
        perf_df.style
        .background_gradient(subset=["AUC"], cmap="Blues")
        .format({"정밀도": "{:.2f}", "재현율": "{:.2f}", "F1-Score": "{:.2f}", "AUC": "{:.2f}"}),
        use_container_width=True, hide_index=True
    )

# ── Tab 3: 시간 지평 비교 ───────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>단기(주간) vs 중장기(월간) 예측력 비교</div>",
                unsafe_allow_html=True)

    h_data = pd.DataFrame({
        "모델": ["XGBoost", "XGBoost", "TabNet", "TabNet"],
        "시간지평": ["주간 (단기)", "월간 (중장기)", "주간 (단기)", "월간 (중장기)"],
        "AUC": [0.72, 0.81, 0.69, 0.77],
    })
    fig3 = px.bar(h_data, x="모델", y="AUC", color="시간지평", barmode="group",
                  color_discrete_map={"주간 (단기)": "#58a6ff", "월간 (중장기)": "#3fb950"},
                  text="AUC")
    fig3.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig3.update_layout(**{**LAYOUT, "height": 360,
                           "yaxis": {**LAYOUT["yaxis"], "range": [0.5, 0.95]}})
    st.plotly_chart(fig3, use_container_width=True)
    st.info("💡 H2 검증: 중장기(월간) 모델 AUC가 단기(주간) 대비 0.08~0.09 높음. 주간 노이즈로 인한 단기 예측력 한계가 정량적으로 확인됨.")

    st.markdown("<div class='section-header'>만기일 효과 분석 (H3)</div>", unsafe_allow_html=True)
    exp_data = pd.DataFrame({
        "구분": ["만기일 직전 주간", "일반 주간"],
        "평균 VKOSPI": [29.4, 22.1],
        "표준편차": [6.2, 4.1]
    })
    c1, c2 = st.columns(2)
    with c1:
        fig_e = px.bar(exp_data, x="구분", y="평균 VKOSPI", color="구분",
                       color_discrete_map={"만기일 직전 주간": "#f85149", "일반 주간": "#58a6ff"},
                       error_y="표준편차")
        fig_e.update_layout(**{**LAYOUT, "height": 280, "showlegend": False})
        st.plotly_chart(fig_e, use_container_width=True)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("만기일 직전 평균 VKOSPI", "29.4", "+7.3 vs 일반 주간")
        st.metric("t-검정 p-value", "0.003", "✅ 유의수준 0.05 이하")
        st.success("H3 채택: 만기일 직전 주간 변동성이 유의미하게 높음")

# ── Tab 4: 백테스팅 ─────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-header'>누적 수익률 비교 (2024.01 ~ 2025.12)</div>",
                unsafe_allow_html=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=bt["date"], y=(bt["model"] - 1) * 100,
        name="모델 기반 전략", line=dict(color="#3fb950", width=2.5),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.06)"
    ))
    fig4.add_trace(go.Scatter(
        x=bt["date"], y=(bt["buy_hold"] - 1) * 100,
        name="Buy & Hold", line=dict(color="#8b949e", width=1.5, dash="dot")
    ))
    fig4.add_trace(go.Scatter(
        x=bt["date"], y=(bt["moving_avg"] - 1) * 100,
        name="이동평균 전략", line=dict(color="#f0b429", width=1.5, dash="dash")
    ))
    fig4.add_hline(y=0, line_color="#30363d")
    fig4.update_layout(**{**LAYOUT, "height": 380,
                           "yaxis": {**LAYOUT["yaxis"], "title": "누적 수익률 (%)"}})
    st.plotly_chart(fig4, use_container_width=True)

    risk_df = pd.DataFrame({
        "전략": ["🟢 모델 기반 전략", "⚪ Buy & Hold", "🟡 이동평균 전략"],
        "누적 수익률": ["18.4%", "9.2%", "11.7%"],
        "연환산 수익률": ["9.1%", "4.6%", "5.8%"],
        "샤프비율": ["1.24", "0.83", "0.91"],
        "최대 낙폭(MDD)": ["-8.3%", "-14.5%", "-11.2%"],
        "승률": ["64.2%", "52.1%", "55.8%"]
    })
    st.markdown("<div class='section-header'>전략별 리스크/수익 지표</div>", unsafe_allow_html=True)
    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    st.success("✅ H5 검증: 모델 전략이 Buy&Hold 대비 누적 수익률 +9.2%p, 샤프비율 +0.41, MDD -6.2%p 개선.")

st.markdown("---")
st.caption("⚠️ 본 대시보드는 공모전 시연용 프로토타입입니다. 실제 데이터 기반 결과는 최종 보고서를 참조하세요.")
st.caption("📁 데이터 출처: KRX 정보데이터시스템 · Yahoo Finance · pykrx | 모델: XGBoost, TabNet")