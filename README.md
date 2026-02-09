# Crypto Trading Website - EMA & Candlestick Chart

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip3 install flask flask-cors pandas numpy requests
```

### 2. Start Server
```bash
python3 backend_api.py
```

### 3. Open Browser
```
http://localhost:5000
```

## 📁 Files

- `backend_api.py` - Flask backend API
- `fetch_trading_data.py` - Crypto API client
- `static/index.html` - Frontend website
- `backtest_strategy.py` - Complete trading strategy backtesting script

## 🎯 Features

- ✅ Candlestick Charts (Open, High, Low, Close)
- ✅ EMA Lines (9, 21, 50 periods)
- ✅ Multiple Symbols (BTC, ETH, BNB, etc.)
- ✅ Multiple Intervals (1m, 5m, 1h, 4h, 1d)
- ✅ Market Info (Price, 24h change, high/low)

## 📡 API Endpoints

- `GET /api/candles` - Candle data with EMA
- `GET /api/market-info` - Market information
- `GET /api/health` - Health check

## 📊 Backtesting Strategy

### Running the Backtest

The `backtest_strategy.py` script provides a complete backtesting solution for the EMA crossover strategy:

```bash
python3 backtest_strategy.py
```

### Strategy Rules

- **Indicators**: EMA 9, EMA 21, EMA 50
- **Entry Rules**:
  - BUY: EMA 9 crosses above BOTH EMA 21 and EMA 50 (after candle close)
  - SELL: EMA 9 crosses below BOTH EMA 21 and EMA 50 (after candle close)
- **Exit Rules**:
  - Fixed Stop Loss: 400 points
  - Fixed Target: 800 points
  - Risk:Reward = 1:2
  - Only one trade at a time (no overlapping trades)

### Backtesting Periods

The script automatically tests the strategy over:
- Last 3 months
- Last 6 months
- Last 1 year
- Last 2 years

### Output Metrics

For each period, the script provides:
- Total number of trades
- Number of winning trades
- Number of losing trades
- Win rate (%)
- Net points gained or lost
- Maximum drawdown
- Profit factor
- Average win/loss
- Total profit/loss

### Configuration

You can easily configure the strategy by modifying the `DEFAULT_CONFIG` dictionary in `backtest_strategy.py`:

```python
DEFAULT_CONFIG = {
    'symbol': 'BTCUSDT',      # Trading symbol
    'timeframe': '5m',         # Timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
    'exchange': 'binance',      # Exchange (binance or delta)
    'ema_periods': {
        'ema9': 9,
        'ema21': 21,
        'ema50': 50
    },
    'stop_loss_points': 400,   # Stop loss in points
    'target_points': 800       # Target in points
}
```

### Results

Results are displayed in the console and saved to `backtest_results.json` for further analysis.

## ⚠️ Important

API credentials `backend_api.py` and `backtest_strategy.py` mein update karein:
```python
API_KEY = "your_api_key"
SECRET_KEY = "your_secret_key"
```
