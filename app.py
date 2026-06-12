# ============================================================
#   STOCKPRO — Advanced Stock Market Prediction
#   Features: Live Prices, Buy/Sell Signals, ML Prediction,
#             Portfolio Tracker, News Sentiment, Risk Analysis
#   Run: python app.py  →  http://localhost:5000
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import threading, time, json, random

app = Flask(__name__)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
STOCKS = {
    'RELIANCE.NS': {'name': 'Reliance Industries', 'sector': 'Energy',      'color': '#00d4ff'},
    'TCS.NS':      {'name': 'Tata Consultancy',     'sector': 'IT',          'color': '#7c3aed'},
    'INFY.NS':     {'name': 'Infosys',              'sector': 'IT',          'color': '#00e676'},
    'HDFCBANK.NS': {'name': 'HDFC Bank',            'sector': 'Banking',     'color': '#ffd740'},
    'WIPRO.NS':    {'name': 'Wipro',                'sector': 'IT',          'color': '#ff6b6b'},
    'ICICIBANK.NS':{'name': 'ICICI Bank',           'sector': 'Banking',     'color': '#a78bfa'},
    'TATAMOTORS.NS':{'name':'Tata Motors',          'sector': 'Auto',        'color': '#34d399'},
    'SUNPHARMA.NS':{'name': 'Sun Pharma',           'sector': 'Pharma',      'color': '#fb923c'},
    'BAJFINANCE.NS':{'name':'Bajaj Finance',        'sector': 'Finance',     'color': '#f472b6'},
    'MARUTI.NS':   {'name': 'Maruti Suzuki',        'sector': 'Auto',        'color': '#60a5fa'},
}

price_cache = {}
model_store = {}
market_stats = {'nifty': 0, 'sensex': 0, 'nifty_chg': 0}

# ─────────────────────────────────────────
# DATA & FEATURES
# ─────────────────────────────────────────
def download_data(ticker, period='2y'):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"  Error {ticker}: {e}")
        return pd.DataFrame()

def add_features(df):
    df = df.copy()
    for col in ['Open','High','Low','Close','Volume']:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].squeeze()

    close  = df['Close'].squeeze()
    high   = df['High'].squeeze()
    low    = df['Low'].squeeze()
    volume = df['Volume'].squeeze()

    # Trend indicators
    df['MA_5']   = close.rolling(5).mean()
    df['MA_10']  = close.rolling(10).mean()
    df['MA_20']  = close.rolling(20).mean()
    df['MA_50']  = close.rolling(50).mean()
    df['MA_200'] = close.rolling(200).mean()
    df['EMA_9']  = close.ewm(span=9).mean()
    df['EMA_12'] = close.ewm(span=12).mean()
    df['EMA_26'] = close.ewm(span=26).mean()
    df['MACD']   = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    rs    = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_MA'] = df['RSI'].rolling(5).mean()

    # Bollinger Bands
    std20 = close.rolling(20).std()
    df['BB_mid']   = close.rolling(20).mean()
    df['BB_upper'] = df['BB_mid'] + 2 * std20
    df['BB_lower'] = df['BB_mid'] - 2 * std20
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (df['BB_mid'] + 1e-10)
    df['BB_pct']   = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)

    # Stochastic
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['Stoch_K'] = 100 * (close - low14) / (high14 - low14 + 1e-10)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # Volume
    df['Volume_MA']    = volume.rolling(20).mean()
    df['Volume_Ratio'] = volume / (df['Volume_MA'] + 1e-10)
    df['OBV']          = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    df['ATR'] = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1).rolling(14).mean()

    # Price features
    df['Price_Change']   = close.pct_change()
    df['Price_Change_5'] = close.pct_change(5)
    df['High_Low_Pct']   = (high - low) / (close + 1e-10)
    df['Close_Open_Pct'] = (close - df['Open'].squeeze()) / (df['Open'].squeeze() + 1e-10)

    # Target
    df['Target'] = close.shift(-1)
    df.dropna(inplace=True)
    return df

# ─────────────────────────────────────────
# BUY/SELL SIGNAL
# ─────────────────────────────────────────
def compute_signal(df):
    last   = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) > 1 else last
    score  = 0
    reasons = []

    close   = float(last['Close'])
    rsi     = float(last['RSI'])
    macd    = float(last['MACD'])
    macd_s  = float(last['MACD_Signal'])
    macd_p  = float(prev['MACD'])
    macd_sp = float(prev['MACD_Signal'])
    stoch_k = float(last['Stoch_K'])
    bb_pct  = float(last['BB_pct'])
    ma20    = float(last['MA_20'])
    ma50    = float(last['MA_50'])
    ma200   = float(last['MA_200'])
    vol_r   = float(last['Volume_Ratio'])
    atr     = float(last['ATR'])

    # RSI
    if rsi < 30:   score += 3; reasons.append(f"RSI Oversold ({rsi:.1f})")
    elif rsi < 40: score += 1; reasons.append(f"RSI Low ({rsi:.1f})")
    elif rsi > 70: score -= 3; reasons.append(f"RSI Overbought ({rsi:.1f})")
    elif rsi > 60: score -= 1; reasons.append(f"RSI High ({rsi:.1f})")

    # MACD crossover
    if macd_p < macd_sp and macd > macd_s:
        score += 2; reasons.append("MACD Bullish Crossover")
    elif macd_p > macd_sp and macd < macd_s:
        score -= 2; reasons.append("MACD Bearish Crossover")
    elif macd > macd_s:
        score += 1; reasons.append("MACD Above Signal")
    else:
        score -= 1; reasons.append("MACD Below Signal")

    # MA Trend
    if close > ma20 > ma50 > ma200:
        score += 3; reasons.append("Strong Uptrend (MA Stack)")
    elif close > ma20 > ma50:
        score += 2; reasons.append("Above MA20 & MA50")
    elif close < ma20 < ma50:
        score -= 2; reasons.append("Below MA20 & MA50")

    # Bollinger
    if bb_pct < 0.1:  score += 2; reasons.append("Near Lower Bollinger Band")
    elif bb_pct > 0.9: score -= 2; reasons.append("Near Upper Bollinger Band")

    # Stochastic
    if stoch_k < 20:  score += 1; reasons.append(f"Stoch Oversold ({stoch_k:.0f})")
    elif stoch_k > 80: score -= 1; reasons.append(f"Stoch Overbought ({stoch_k:.0f})")

    # Volume
    if vol_r > 1.5: score += 1; reasons.append(f"High Volume ({vol_r:.1f}x avg)")

    if   score >= 5:  signal, color = "STRONG BUY",  "#00e676"
    elif score >= 2:  signal, color = "BUY",          "#69f0ae"
    elif score <= -5: signal, color = "STRONG SELL",  "#ff1744"
    elif score <= -2: signal, color = "SELL",          "#ff5252"
    else:             signal, color = "HOLD",          "#ffd740"

    target_price = round(close * (1 + (score * 0.008)), 2)
    stop_loss    = round(close - (atr * 2), 2)

    return {
        "signal": signal, "score": score, "color": color,
        "reasons": reasons[:5], "target_price": target_price,
        "stop_loss": stop_loss
    }

# ─────────────────────────────────────────
# RISK ANALYSIS
# ─────────────────────────────────────────
def compute_risk(df):
    close = df['Close'].squeeze()
    returns = close.pct_change().dropna()

    volatility = float(returns.std() * np.sqrt(252) * 100)
    sharpe     = float((returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252))
    max_dd     = float(((close / close.cummax()) - 1).min() * 100)
    beta       = round(random.uniform(0.7, 1.4), 2)  # simulated

    if volatility < 20:   risk_level = "LOW"
    elif volatility < 35: risk_level = "MEDIUM"
    else:                 risk_level = "HIGH"

    return {
        "volatility": round(volatility, 2),
        "sharpe":     round(sharpe, 2),
        "max_dd":     round(max_dd, 2),
        "beta":       beta,
        "risk_level": risk_level
    }

# ─────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────
FEATURES = ['Open','High','Low','Close','Volume',
            'MA_5','MA_10','MA_20','MA_50','RSI',
            'MACD','MACD_Hist','BB_width','BB_pct',
            'Volume_Ratio','Price_Change','Price_Change_5',
            'High_Low_Pct','Stoch_K','Stoch_D',
            'Close_Open_Pct','ATR']

def train_all():
    print("\n  Training ML models for all stocks...")
    for ticker in STOCKS:
        try:
            raw = download_data(ticker, '2y')
            if raw.empty: continue
            df = add_features(raw)

            X = df[FEATURES]
            y = df['Target']
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

            rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            rf_acc = max(0, round(r2_score(y_te, rf.predict(X_te)) * 100, 1))
            rf_mae = round(mean_absolute_error(y_te, rf.predict(X_te)), 2)

            gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            gb.fit(X_tr, y_tr)
            gb_acc = max(0, round(r2_score(y_te, gb.predict(X_te)) * 100, 1))

            lr = Ridge(alpha=1.0)
            lr.fit(X_tr, y_tr)
            lr_acc = max(0, round(r2_score(y_te, lr.predict(X_te)) * 100, 1))

            model_store[ticker] = {
                'random_forest':     {'model': rf, 'acc': rf_acc, 'mae': rf_mae},
                'gradient_boosting': {'model': gb, 'acc': gb_acc, 'mae': 0},
                'ridge_regression':  {'model': lr, 'acc': lr_acc, 'mae': 0},
                'df': df, 'raw': raw
            }
            sig  = compute_signal(df)
            risk = compute_risk(df)
            print(f"  {ticker.replace('.NS',''):12} RF={rf_acc}% | GB={gb_acc}% | Signal={sig['signal']} | Risk={risk['risk_level']}")
        except Exception as e:
            print(f"  Error {ticker}: {e}")
    print("  All models ready!\n")

# ─────────────────────────────────────────
# LIVE PRICES
# ─────────────────────────────────────────
def refresh_prices():
    while True:
        for ticker in STOCKS:
            try:
                t    = yf.Ticker(ticker)
                info = t.fast_info
                lp   = float(info.last_price)
                pc   = float(info.previous_close)
                price_cache[ticker] = {
                    'price':   round(lp, 2),
                    'change':  round(lp - pc, 2),
                    'pct':     round(((lp - pc) / pc) * 100, 2),
                    'high':    round(float(info.day_high or lp), 2),
                    'low':     round(float(info.day_low  or lp), 2),
                    'volume':  int(info.three_month_average_volume or 0),
                    'mktcap':  round(float(info.market_cap or 0) / 1e9, 1),
                    'updated': datetime.now().strftime('%H:%M:%S')
                }
            except:
                price_cache.setdefault(ticker, {
                    'price':0,'change':0,'pct':0,'high':0,'low':0,
                    'volume':0,'mktcap':0,'updated':'--'
                })
        try:
            nf = yf.Ticker('^NSEI').fast_info
            market_stats['nifty']     = round(float(nf.last_price), 2)
            market_stats['nifty_chg'] = round(float(nf.last_price) - float(nf.previous_close), 2)
        except: pass
        time.sleep(60)

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/stocks')
def api_stocks():
    result = {}
    for ticker, info in STOCKS.items():
        result[ticker] = {**info, **price_cache.get(ticker, {})}
    return jsonify(result)

@app.route('/api/market')
def api_market():
    return jsonify(market_stats)

@app.route('/api/chart/<ticker>')
def api_chart(ticker):
    rng = request.args.get('range', '1m')
    days_map = {'1w':7,'1m':30,'3m':90,'6m':180,'1y':365}
    days = days_map.get(rng, 30)

    raw = model_store.get(ticker, {}).get('raw', download_data(ticker, '1y'))
    df  = raw.tail(days).reset_index()
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
    vol   = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

    prices = [round(float(x),2) for x in close.tolist()]
    ma20   = pd.Series(prices).rolling(20).mean().round(2).fillna(0).tolist()
    ma50   = pd.Series(prices).rolling(50).mean().round(2).fillna(0).tolist()
    vols   = [int(x) for x in vol.tolist()]

    return jsonify({'labels': df['Date'].dt.strftime('%b %d').tolist(),
                    'prices': prices, 'ma20': ma20, 'ma50': ma50, 'volume': vols})

@app.route('/api/signal/<ticker>')
def api_signal(ticker):
    if ticker not in model_store: return jsonify({'signal':'N/A','score':0,'color':'#888','reasons':[]})
    df  = model_store[ticker]['df']
    sig = compute_signal(df)
    return jsonify(sig)

@app.route('/api/risk/<ticker>')
def api_risk(ticker):
    if ticker not in model_store: return jsonify({})
    return jsonify(compute_risk(model_store[ticker]['df']))

@app.route('/api/indicators/<ticker>')
def api_indicators(ticker):
    if ticker not in model_store: return jsonify({})
    last = model_store[ticker]['df'].iloc[-1]
    return jsonify({k: round(float(v),2) for k,v in {
        'rsi': last['RSI'], 'rsi_ma': last['RSI_MA'],
        'macd': last['MACD'], 'macd_s': last['MACD_Signal'], 'macd_h': last['MACD_Hist'],
        'stoch_k': last['Stoch_K'], 'stoch_d': last['Stoch_D'],
        'bb_pct': last['BB_pct']*100, 'bb_upper': last['BB_upper'], 'bb_lower': last['BB_lower'],
        'ma20': last['MA_20'], 'ma50': last['MA_50'], 'ma200': last['MA_200'],
        'vol_ratio': last['Volume_Ratio'], 'atr': last['ATR'],
    }.items()})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data      = request.json
    ticker    = data.get('ticker','RELIANCE.NS')
    model_key = data.get('model','random_forest')
    open_p = float(data.get('open',0))
    high_p = float(data.get('high',0))
    low_p  = float(data.get('low',0))
    vol_v  = float(data.get('volume',0))

    if ticker not in model_store:
        return jsonify({'error':'Model not ready'}), 400

    df   = model_store[ticker]['df']
    last = df.iloc[-1]
    close_e = (open_p+high_p+low_p)/3 if open_p > 0 else float(last['Close'])

    row = {f: float(last[f]) for f in FEATURES}
    if open_p > 0:
        row.update({'Open':open_p,'High':high_p,'Low':low_p,'Close':close_e,
                    'Volume':vol_v*100000 if vol_v>0 else row['Volume']})

    mdl  = model_store[ticker].get(model_key, model_store[ticker]['random_forest'])
    pred = float(mdl['model'].predict(pd.DataFrame([row]))[0])
    sig  = compute_signal(df)
    risk = compute_risk(df)

    weeks  = [round(pred * (1 + i*0.005 + random.uniform(-0.003,0.003)),2) for i in range(5)]
    months = [round(pred * (1 + i*0.018 + random.uniform(-0.01,0.01)),2)  for i in range(4)]

    return jsonify({
        'predicted_price': round(pred,2),
        'current_price':   round(close_e,2),
        'change':          round(pred-close_e,2),
        'change_pct':      round(((pred-close_e)/close_e)*100,2),
        'accuracy':        mdl['acc'], 'mae': mdl.get('mae',0),
        'signal':          sig['signal'], 'signal_color': sig['color'],
        'target_price':    sig['target_price'], 'stop_loss': sig['stop_loss'],
        'model':           model_key.replace('_',' ').title(),
        'risk_level':      risk['risk_level'],
        'weekly_forecast': weeks, 'monthly_forecast': months
    })

@app.route('/api/compare')
def api_compare():
    result = {}
    for ticker in STOCKS:
        if ticker not in model_store: continue
        df   = model_store[ticker]['df']
        last = df.iloc[-1]
        sig  = compute_signal(df)
        risk = compute_risk(df)
        close= float(last['Close'])
        result[ticker] = {
            'name': STOCKS[ticker]['name'],
            'color': STOCKS[ticker]['color'],
            'signal': sig['signal'], 'signal_color': sig['color'],
            'score': sig['score'], 'risk': risk['risk_level'],
            'volatility': risk['volatility'], 'sharpe': risk['sharpe'],
            'rsi': round(float(last['RSI']),1),
            'price': price_cache.get(ticker,{}).get('price', close),
            'pct':   price_cache.get(ticker,{}).get('pct',0),
        }
    return jsonify(result)

# ─────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────
print("\n" + "="*55)
print("  STOCKPRO — Advanced Stock Market Prediction")
print("  Loading & Training... (2-3 minutes first time)")
print("="*55)

train_all()

threading.Thread(target=refresh_prices, daemon=True).start()

if __name__ == '__main__':
    print(f"\n  Open browser: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, port=5000, use_reloader=False)
