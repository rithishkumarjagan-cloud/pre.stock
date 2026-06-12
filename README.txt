╔═══════════════════════════════════════════════════╗
║        STOCKPRO — ADVANCED TRADING DASHBOARD     ║
╚═══════════════════════════════════════════════════╝

FEATURES:
  ✅ 10 NSE Stocks (Reliance, TCS, Infosys, HDFC, Wipro,
                    ICICI, Tata Motors, Sun Pharma, 
                    Bajaj Finance, Maruti)
  ✅ Live Price Auto-Refresh (every 60 seconds)
  ✅ Buy / Sell / Strong Buy / Strong Sell Signals
  ✅ Bloomberg Dark Terminal UI
  ✅ 3 ML Models (Random Forest, Gradient Boosting, Ridge)
  ✅ Technical Indicators (RSI, MACD, Stochastic, BB, ATR, OBV)
  ✅ Price Chart with MA20, MA50 overlay (toggleable)
  ✅ 5-Day Price Forecast
  ✅ Risk Analysis (Volatility, Sharpe, Max Drawdown, Beta)
  ✅ All Stocks Comparison Table
  ✅ Portfolio Tracker with P&L
  ✅ News Sentiment Analysis

PAGES:
  Dashboard  - Live prices, signals, chart, indicators
  Predict    - ML prediction, risk analysis, forecast
  Compare    - All 10 stocks side by side
  Portfolio  - Track your investments
  News       - Market news with sentiment

HOW TO RUN:

Step 1 - Install libraries (only once):
  pip install flask yfinance pandas numpy scikit-learn matplotlib

Step 2 - Run the app:
  python app.py

Step 3 - Open browser:
  http://localhost:5000

NOTE: First run takes 3-5 minutes to download and 
      train models for all 10 stocks.
