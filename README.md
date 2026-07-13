# 📈 Nasdaq 100 Direction Predictor

A machine learning web application that predicts whether the **Nasdaq-100 (NQ)** will finish **Bullish, Bearish, or Neutral** based on the pre-market returns of its top 10 constituent stocks.

Built with **Django + XGBoost** and deployed free on Render.

---

## 🚀 Live Demo

> Deploy your own free instance on [Render.com](https://render.com) — see deployment section below.

---

## 🧠 How It Works

The Nasdaq-100 is a **market-cap-weighted index**. Larger companies like NVDA, MSFT, and AAPL have a much bigger influence on index direction than smaller ones.

Instead of predicting NQ directly from its own candles, this model learns:
> *"How do the major constituent stocks move before the market opens, and what does that imply for NQ direction?"*

### Feature Engineering Levels

| Level | Feature Type | Description |
|-------|-------------|-------------|
| 1 | Raw Returns | Individual stock returns per timeframe |
| 2 | Weighted Returns | Each stock × its index weight |
| 3 | Sector Averages | Tech, Comm, Discretionary, Staples, Semi |
| 4 | Market Breadth | Positive/negative count, avg/median/max/min return |
| 5 | Agreement Score | % of stocks moving in the same direction |
| 6 | Momentum Spread | 1H−4H and 1H−1D return differences |

### Prediction Modes

- **1H** — Uses only 1-hour pre-market returns (8AM–9:30AM ET)
- **4H** — Uses only 4-hour pre-market returns (5AM–9:30AM ET)
- **1D** — Uses overnight gap returns (previous close → today's open)
- **General** — Uses all timeframes combined

### Top 10 Stocks Used

| Stock | Weight |
|-------|--------|
| MSFT  | 17%    |
| AAPL  | 17%    |
| NVDA  | 16%    |
| AMZN  | 10%    |
| AVGO  | 10%    |
| META  | 9%     |
| TSLA  | 7%     |
| GOOGL | 5%     |
| GOOG  | 5%     |
| COST  | 4%     |

---

## 🛠️ Tech Stack

- **Backend**: Python, Django 5.2
- **ML**: XGBoost, Scikit-Learn (GridSearchCV)
- **Frontend**: Vanilla HTML/CSS/JS, Chart.js
- **Static Files**: WhiteNoise
- **Deployment**: Render.com (free tier)

---

## 📦 Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/trading.git
cd trading

# Install dependencies
pip install -r requirements.txt

# Run server
cd trading_project
python manage.py runserver
```

Open `http://127.0.0.1:8000/` in your browser.

---

## ☁️ Free Deployment on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set these settings:
   - **Root Directory**: *(leave blank)*
   - **Build Command**: `./build.sh`
   - **Start Command**: `cd trading_project && gunicorn trading_project.wsgi --log-file -`
5. Add environment variable:
   - `SECRET_KEY` → any long random string
   - `DEBUG` → `False`
6. Click **Deploy** ✅

---

## 📊 Model Performance

| Model | Best Params | Test Accuracy |
|-------|-------------|---------------|
| 1H    | depth=3, lr=0.05, n=50  | ~56% |
| 4H    | depth=2, lr=0.03, n=100 | ~54% |
| 1D    | depth=3, lr=0.01, n=100 | ~51% |
| General | depth=2, lr=0.05, n=50 | ~57% |

> Note: Financial markets are inherently noisy. 50–58% accuracy on a 3-class problem (vs 33% random) represents meaningful predictive signal.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not financial advice. Do not use this to make real trading decisions.
