import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from datetime import datetime, timedelta

# --- 頁面設定 ---
st.set_page_config(page_title="美股主力高勝率分析系統", layout="wide")

# --- 側邊欄：使用者輸入 ---
st.sidebar.title("🛠️ 參數設定")
ticker_symbol = st.sidebar.text_input("輸入美股代號 (例如: NVDA, TSLA, AAPL)", "NVDA").upper()
time_period = st.sidebar.selectbox("分析週期", ["6mo", "1y", "2y", "5y"], index=1)
ma_window_short = st.sidebar.slider("短期均線 (日)", 5, 50, 20)
ma_window_long = st.sidebar.slider("長期均線 (日)", 50, 200, 50)

# --- 核心功能函數 ---

def get_stock_data(ticker, period):
    """獲取股價數據"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None, None
        return df, stock
    except Exception as e:
        st.error(f"數據獲取失敗: {e}")
        return None, None

def analyze_sentiment(news_list):
    """新聞情緒分析 (簡易 NLP)"""
    sentiment_score = 0
    analyzed_news = []
    
    if not news_list:
        return 0, []

    for news in news_list[:5]: # 只分析最新的 5 則新聞
        title = news.get('title', '')
        link = news.get('link', '#')
        publisher = news.get('publisher', 'Unknown')
        
        # 使用 TextBlob 進行情緒分析 (-1 到 1)
        blob = TextBlob(title)
        polarity = blob.sentiment.polarity
        sentiment_score += polarity
        
        sentiment_label = "⚪ 中性"
        if polarity > 0.1: sentiment_label = "🟢 正面"
        elif polarity < -0.1: sentiment_label = "🔴 負面"
        
        analyzed_news.append({
            "title": title,
            "link": link,
            "publisher": publisher,
            "sentiment": sentiment_label,
            "score": polarity
        })
    
    # 正規化總分 (-100 到 100)
    count = len(news_list[:5])
    if count == 0:
        return 0, []
    final_score = (sentiment_score / count) * 100
    return final_score, analyzed_news

def calculate_smart_money(df):
    """主力追蹤邏輯 (基於成交量與價格行為)"""
    # 1. 計算相對成交量 (RVOL)
    df['Vol_SMA'] = df['Volume'].rolling(50).mean()
    df['RVOL'] = df['Volume'] / df['Vol_SMA']
    
    # 2. OBV (能量潮指標 - 判斷資金流向)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBV_EMA'] = ta.ema(df['OBV'], length=20)
    
    # 3. MFI (資金流量指標)
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)

    # 判斷最近一天的狀態
    latest = df.iloc[-1]
    
    signals = []
    score = 50 # 基礎分
    
    # 主力進出判斷
    if latest['RVOL'] > 1.5 and latest['Close'] > latest['Open']:
        score += 15
        signals.append("🔥 爆量上漲 (主力搶籌跡象)")
    elif latest['RVOL'] > 1.5 and latest['Close'] < latest['Open']:
        score -= 15
        signals.append("⚠️ 爆量下跌 (主力出貨跡象)")
        
    if latest['OBV'] > latest['OBV_EMA']:
        score += 10
        signals.append("📈 OBV 位於均線上方 (資金持續流入)")
        
    if latest['MFI'] > 80:
        score -= 10
        signals.append("⚠️ MFI 過熱 (>80)，資金可能短期撤離")
    elif latest['MFI'] < 20:
        score += 10
        signals.append("🛒 MFI 過冷 (<20)，資金可能回流")
        
    return max(0, min(100, score)), signals

def calculate_technical_strategy(df):
    """高級交易員技術分析 (高勝率策略)"""
    # 1. 趨勢指標
    df['SMA_Short'] = ta.sma(df['Close'], length=ma_window_short)
    df['SMA_Long'] = ta.sma(df['Close'], length=ma_window_long)
    
    # 2. 動能指標 (RSI, MACD)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    # pandas_ta 的 macd 欄位命名通常是 MACD_12_26_9, MACDh_12_26_9 (Hist), MACDs_12_26_9 (Signal)
    df = pd.concat([df, macd], axis=1) 
    
    # 3. 通道指標 (Bollinger Bands)
    bb = ta.bbands(df['Close'], length=20)
    df = pd.concat([df, bb], axis=1)

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 50
    reasons = []
    
    # A. 趨勢判斷
    if latest['SMA_Long'] is not None and latest['Close'] > latest['SMA_Long']:
        score += 10
        reasons.append("✅ 價格位於長期均線之上 (多頭趨勢)")
    
    # B. RSI 高勝率區間
    if latest['RSI'] < 30:
        score += 20
        reasons.append("💎 RSI 超賣 (<30)，高勝率反彈點")
    elif latest['RSI'] > 70:
        score -= 20
        reasons.append("⚠️ RSI 超買 (>70)，回調風險高")
        
    # C. MACD 金叉/死叉 (確認欄位名稱存在)
    if 'MACD_12_26_9' in df.columns:
        macd_line = latest['MACD_12_26_9']
        macd_signal = latest['MACDs_12_26_9']
        prev_macd_line = prev['MACD_12_26_9']
        prev_macd_signal = prev['MACDs_12_26_9']
        
        if macd_line > macd_signal and prev_macd_line <= prev_macd_signal:
            score += 15
            reasons.append("🚀 MACD 黃金交叉 (買入訊號)")
        
    # D. 布林帶策略
    if 'BBL_20_2.0' in df.columns and latest['Close'] < latest['BBL_20_2.0']:
        score += 15
        reasons.append("🛡️ 跌破布林下軌 (超跌回歸)")
        
    return max(0, min(100, score)), reasons, df

# --- 主程式邏輯 ---

st.title(f"🇺🇸 美股深度分析軟體: {ticker_symbol}")

# 1. 獲取數據
data, stock_obj = get_stock_data(ticker_symbol, time_period)

if data is not None:
    # --- 2. 計算各項指標 ---
    try:
        tech_score, tech_reasons, data = calculate_technical_strategy(data)
        sm_score, sm_signals = calculate_smart_money(data)
        
        # 新聞獲取與分析
        news = stock_obj.news
        sent_score, analyzed_news = analyze_sentiment(news)
        
        # --- 3. 綜合儀表板 (上方 KPI) ---
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        price_change = current_price - prev_close
        pct_change = (price_change / prev_close) * 100
        
        # 綜合評分權重: 技術(50%) + 主力(30%) + 新聞(20%)
        total_score = (tech_score * 0.5) + (sm_score * 0.3) + (sent_score + 50) * 0.2 
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最新股價", f"${current_price:.2f}", f"{pct_change:.2f}%")
        col2.metric("綜合推薦評分", f"{total_score:.1f}/100", delta_color="normal")
        col3.metric("主力活躍度", f"{sm_score:.1f}", delta_color="off")
        col4.metric("市場情緒", f"{sent_score:.1f}", "Sentiment")

        # --- 4. 建議與進場點 ---
        st.divider()
        st.subheader("🎯 交易決策與進場分析")
        
        col_rec, col_entry = st.columns(2)
        
        with col_rec:
            if total_score >= 70:
                st.success(f"### 🚀 強力買入訊號 \n這支股票目前技術面強勢，且主力資金流入。")
            elif total_score >= 50:
                st.warning(f"### ⚖️ 中性 / 觀望 \n多空訊號混雜，建議等待更明確的回調或突破。")
            else:
                st.error(f"### 🛑 賣出 / 避免進場 \n技術面轉弱或主力出貨中。")
                
            st.write("**綜合分析理由:**")
            for r in tech_reasons + sm_signals:
                st.write(f"- {r}")

        with col_entry:
            # 計算支撐與壓力 (20日極值)
            recent_high = data['High'].tail(20).max()
            recent_low = data['Low'].tail(20).min()
            
            st.info("### 📉 合理進場/出場點位")
            st.write(f"**短期壓力位 (目標價):** ${recent_high:.2f}")
            st.write(f"**目前價格:** ${current_price:.2f}")
            st.write(f"**短期支撐位 (安全進場):** ${recent_low:.2f}")
            
            if total_score >= 60:
                buy_zone = current_price * 0.98
                st.write(f"💡 **策略建議:** 若價格回調至 **${buy_zone:.2f}** 附近可分批佈局。")

        # --- 5. 專業互動圖表 (Plotly) ---
        st.divider()
        st.subheader("📊 高級交易員視圖")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=('價格與均線 (Price)', '成交量與主力 (Volume)'), 
                            row_width=[0.2, 0.7])

        # K線圖
        fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                                     low=data['Low'], close=data['Close'], name='K線'), row=1, col=1)
        # 均線
        if 'SMA_Short' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_Short'], line=dict(color='orange', width=1), name='短期均線'), row=1, col=1)
        if 'SMA_Long' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_Long'], line=dict(color='blue', width=1), name='長期均線'), row=1, col=1)
        # 布林帶
        if 'BBU_20_2.0' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['BBU_20_2.0'], line=dict(color='gray', width=0.5, dash='dot'), name='布林上軌'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['BBL_20_2.0'], line=dict(color='gray', width=0.5, dash='dot'), name='布林下軌'), row=1, col=1)

        # 成交量
        colors = ['green' if row['Open'] - row['Close'] >= 0 else 'red' for index, row in data.iterrows()]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color=colors, name='成交量'), row=2, col=1)

        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # --- 6. 最新消息與情緒 ---
        st.divider()
        st.subheader("📰 即時新聞與情緒解讀")
        
        if analyzed_news:
            for news_item in analyzed_news:
                with st.expander(f"{news_item['sentiment']} | {news_item['title']} ({news_item['publisher']})"):
                    st.write(f"情緒分數: {news_item['score']:.2f}")
                    st.write(f"[閱讀全文]({news_item['link']})")
        else:
            st.write("暫無最新相關新聞。")
            
    except Exception as e:
        st.error(f"分析過程中發生錯誤: {e}")
        st.write("建議檢查股票代碼是否正確，或稍後再試。")
            
else:
    st.info("請在左側輸入有效的股票代碼並按 Enter (例如: NVDA)。")
