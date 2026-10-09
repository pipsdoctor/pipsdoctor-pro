from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import urllib.request
import json

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

app = FastAPI(title="PipsDoctor Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Candle(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float

class NewsItem(BaseModel):
    title: str
    pub_date: str
    impact: str

class TradeAnalysis(BaseModel):
    symbol: str
    timeframe: str
    current_price: float
    source: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confidence_score: int
    technical_reasons: list[str] = []
    price_action_reasons: list[str] = []
    news_events: list[NewsItem] = []
    candles: list[Candle] = []

class AccountConnectRequest(BaseModel):
    login: int
    password: str
    server: str

class ExecuteTradeRequest(BaseModel):
    symbol: str
    action: str
    volume: float = 0.01
    sl: float = 0.0
    tp: float = 0.0

class ChatRequest(BaseModel):
    message: str
    symbol: str = "EURUSD"
    current_bias: str = "NEUTRAL"

class ChatResponse(BaseModel):
    reply: str

def fetch_economic_news() -> list[NewsItem]:
    try:
        req = urllib.request.Request(
            "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            items = []
            for ev in data[:5]:
                impact = ev.get("impact", "Low").upper()
                items.append(NewsItem(
                    title=f"{ev.get('country', '')} - {ev.get('title', '')}",
                    pub_date=ev.get("date", "Live")[:16],
                    impact=impact if impact in ["HIGH", "MEDIUM"] else "LOW"
                ))
            if items:
                return items
    except Exception:
        pass

    return [
        NewsItem(title="USD - Federal Reserve Liquidity Operations", pub_date="Live Flow", impact="HIGH"),
        NewsItem(title="EUR - ECB Core Inflation Outlook", pub_date="Live Flow", impact="HIGH"),
        NewsItem(title="XAU - Gold Institutional Reserve Flows", pub_date="Live Flow", impact="MEDIUM"),
    ]

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df['ema_fast'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema_slow'] = df['Close'].ewm(span=50, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean().bfill()
    return df

@app.post("/api/account/connect")
def connect_account(req: AccountConnectRequest):
    if not MT5_AVAILABLE:
        raise HTTPException(status_code=500, detail="MetaTrader5 Python module not installed. Run 'pip install MetaTrader5'.")

    if not mt5.initialize():
        raise HTTPException(status_code=500, detail="Could not initialize MT5. Ensure MetaTrader 5 is open on your laptop.")

    authorized = mt5.login(login=int(req.login), password=str(req.password), server=str(req.server).strip())
    if not authorized:
        err = mt5.last_error()
        raise HTTPException(status_code=401, detail=f"MT5 login failed: {err}")

    acc = mt5.account_info()
    return {
        "status": "connected",
        "login": req.login,
        "balance": acc.balance if acc else 0.0,
        "equity": acc.equity if acc else 0.0,
        "currency": acc.currency if acc else "USD"
    }

@app.post("/api/trade/execute")
def execute_trade(req: ExecuteTradeRequest):
    if not MT5_AVAILABLE or not mt5.terminal_info():
        raise HTTPException(status_code=400, detail="MetaTrader 5 terminal is not connected.")

    action_type = req.action.upper()
    sym = req.symbol.upper()

    matched_sym = sym
    if not mt5.symbol_select(sym, True):
        for s in mt5.symbols_get():
            if sym in s.name:
                matched_sym = s.name
                mt5.symbol_select(matched_sym, True)
                break

    tick = mt5.symbol_info_tick(matched_sym)
    if not tick:
        raise HTTPException(status_code=400, detail=f"No tick data available for {matched_sym}")

    price = tick.ask if action_type == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if action_type == "BUY" else mt5.ORDER_TYPE_SELL

    trade_req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": matched_sym,
        "volume": float(req.volume),
        "type": order_type,
        "price": price,
        "sl": float(req.sl),
        "tp": float(req.tp),
        "deviation": 20,
        "magic": 202610,
        "comment": "PipsDoctor Pro",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(trade_req)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise HTTPException(status_code=500, detail=f"Order rejected ({result.retcode}): {result.comment}")

    return {"status": "success", "ticket": result.order, "price": result.price}

@app.get("/api/analyze", response_model=TradeAnalysis)
def analyze(symbol: str = "EURUSD", timeframe: str = "1h"):
    raw_df = pd.DataFrame()
    data_source = "Global Web Aggregator"

    if MT5_AVAILABLE and mt5.terminal_info():
        tf_map = {
            "15m": mt5.TIMEFRAME_M15,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1
        }
        mt_tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
        matched_sym = symbol
        if not mt5.symbol_select(symbol, True):
            for s in mt5.symbols_get():
                if symbol in s.name:
                    matched_sym = s.name
                    mt5.symbol_select(matched_sym, True)
                    break

        rates = mt5.copy_rates_from_pos(matched_sym, mt_tf, 0, 100)
        if rates is not None and len(rates) > 30:
            raw_df = pd.DataFrame(rates)
            raw_df['Open'] = raw_df['open']
            raw_df['High'] = raw_df['high']
            raw_df['Low'] = raw_df['low']
            raw_df['Close'] = raw_df['close']
            raw_df.index = pd.to_datetime(raw_df['time'], unit='s')
            data_source = "Live MT5 Broker Feed"

    if raw_df.empty:
        import yfinance as yf
        pair_sym = "GC=F" if symbol == "XAUUSD" else f"{symbol}=X"
        valid_intervals = {"15m": ("5d", "15m"), "1h": ("1mo", "1h"), "1d": ("1y", "1d")}
        period, interval = valid_intervals.get(timeframe, ("1mo", "1h"))
        raw_df = yf.Ticker(pair_sym).history(period=period, interval=interval)
        if raw_df.empty or len(raw_df) < 30:
            raise HTTPException(status_code=500, detail="Data feed unavailable.")

    df = calculate_technical_features(raw_df.copy())
    candles = [
        Candle(
            time=int(idx.timestamp()),
            open=round(float(row['Open']), 5),
            high=round(float(row['High']), 5),
            low=round(float(row['Low']), 5),
            close=round(float(row['Close']), 5)
        )
        for idx, row in df.iterrows()
    ]

    last = df.iloc[-1]
    prev1 = df.iloc[-2]
    curr_price = float(last['Close'])
    atr = float(last['atr'])

    tech_reasons = []
    pa_reasons = []
    bullish = 0
    bearish = 0

    if last['ema_fast'] > last['ema_slow']:
        bullish += 2
        tech_reasons.append("EMA 20 > EMA 50: Bullish structural trend established.")
    else:
        bearish += 2
        tech_reasons.append("EMA 20 < EMA 50: Bearish structural trend in control.")

    if float(last['rsi']) > 55:
        bullish += 1
        tech_reasons.append("RSI shows sustained buyer momentum.")
    elif float(last['rsi']) < 45:
        bearish += 1
        tech_reasons.append("RSI highlights prevailing selling pressure.")

    high_20 = df['High'].iloc[-21:-1].max()
    low_20 = df['Low'].iloc[-21:-1].min()

    if prev1['High'] > high_20 and last['Close'] < prev1['High']:
        bearish += 3
        pa_reasons.append("CRT Liquidity Sweep: Price pierced swing high and rejected into range.")
    elif prev1['Low'] < low_20 and last['Close'] > prev1['Low']:
        bullish += 3
        pa_reasons.append("CRT Liquidity Sweep: Price swept swing low and reclaimed range.")

    digits = 2 if "XAU" in symbol else (3 if "JPY" in symbol else 5)
    if bullish > bearish and bullish >= 3:
        direction = "BUY"
        entry_price = curr_price
        stop_loss = round(max(curr_price - (1.5 * atr), low_20), digits)
        sl_dist = abs(entry_price - stop_loss)
        take_profit = round(entry_price + (sl_dist * 2.0), digits)
        confidence = min(94, 55 + (bullish * 6))
    elif bearish > bullish and bearish >= 3:
        direction = "SELL"
        entry_price = curr_price
        stop_loss = round(min(curr_price + (1.5 * atr), high_20), digits)
        sl_dist = abs(stop_loss - entry_price)
        take_profit = round(entry_price - (sl_dist * 2.0), digits)
        confidence = min(94, 55 + (bearish * 6))
    else:
        direction = "NEUTRAL"
        entry_price = curr_price
        stop_loss = curr_price
        take_profit = curr_price
        confidence = 50
        pa_reasons.append("Equilibrium: No directional sweep edge detected. Await session open.")

    sl_dist = abs(entry_price - stop_loss)
    tp_dist = abs(take_profit - entry_price)
    rr = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0

    return TradeAnalysis(
        symbol=symbol,
        timeframe=timeframe,
        current_price=round(curr_price, digits),
        source=data_source,
        direction=direction,
        entry_price=round(entry_price, digits),
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=rr,
        confidence_score=confidence,
        technical_reasons=tech_reasons,
        price_action_reasons=pa_reasons,
        news_events=fetch_economic_news(),
        candles=candles
    )

@app.post("/api/chat", response_model=ChatResponse)
def bot_chat(req: ChatRequest):
    q = req.message.lower()
    if any(k in q for k in ["buy", "sell", "trend", "direction"]):
        reply = f"Current bias for {req.symbol} is {req.current_bias}. Look for pullbacks into discount/premium value areas before executing."
    elif any(k in q for k in ["crt", "range"]):
        reply = "Candle Range Theory: Observe previous high/low boundaries. Enter when wicks sweep liquidity and candle bodies close back inside the range."
    elif any(k in q for k in ["lot", "risk"]):
        reply = "Lot Size Formula: Lot Size = (Account Balance × Risk %) / (SL Pips × Pip Value). Keep risk strictly at 1% per trade."
    elif any(k in q for k in ["gold", "xau"]):
        reply = "Gold (XAUUSD): 1 pip = 0.10 points ($1 move = 10 pips). London Open (07:00 UTC) and NY Open (12:00 UTC) offer highest volume."
    else:
        reply = f"PipsDoctor Bot active on {req.symbol} ({req.current_bias}). Ask me about CRT, lot sizing, or trading strategy!"
    return ChatResponse(reply=reply)

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)