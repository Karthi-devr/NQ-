import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Define configuration
STOCKS = ["MSFT", "AAPL", "NVDA", "AMZN", "META", "AVGO", "TSLA", "COST", "GOOGL", "GOOG"]
WEIGHTS = {
    'MSFT': 0.17, 'AAPL': 0.17, 'NVDA': 0.16, 'AMZN': 0.10, 'META': 0.09,
    'AVGO': 0.10, 'TSLA': 0.07, 'COST': 0.04, 'GOOGL': 0.05, 'GOOG': 0.05
}
DATA_DIR = "data"

def generate_trading_days(start_date, num_days):
    dates = []
    curr = start_date
    while len(dates) < num_days:
        if curr.weekday() < 5:  # Monday to Friday
            dates.append(curr)
        curr += timedelta(days=1)
    return dates

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    np.random.seed(42)
    
    num_days = 1000
    start_date = datetime(2022, 1, 1)
    trading_dates = generate_trading_days(start_date, num_days)
    
    print(f"Generating {num_days} days of synthetic trading data...")
    
    # 1. Initialize stock prices and base paths
    prices = {
        'MSFT': 300.0, 'AAPL': 150.0, 'NVDA': 100.0, 'AMZN': 120.0, 'META': 250.0,
        'AVGO': 400.0, 'TSLA': 200.0, 'COST': 500.0, 'GOOGL': 100.0, 'GOOG': 100.0
    }
    
    # Initialize daily storage dataframes
    stock_daily_data = {symbol: [] for symbol in STOCKS}
    stock_hourly_data = {symbol: [] for symbol in STOCKS}
    index_daily_data = []
    
    nq_price = 15000.0
    
    # 2. Iterate daily and generate data
    for idx, date in enumerate(trading_dates):
        date_str = date.strftime('%Y-%m-%d')
        
        # We need a previous close for the overnight return, so on day 0 we just record close
        if idx == 0:
            for s in STOCKS:
                stock_daily_data[s].append({
                    'Date': f"{date_str} 00:00:00-05:00",
                    'Open': prices[s],
                    'High': prices[s] * 1.01,
                    'Low': prices[s] * 0.99,
                    'Close': prices[s],
                    'Volume': 1000000
                })
            index_daily_data.append({
                'Date': f"{date_str} 00:00:00-05:00",
                'Open': nq_price,
                'High': nq_price * 1.01,
                'Low': nq_price * 0.99,
                'Close': nq_price,
                'Volume': 500000000
            })
            continue
            
        # Daily factors
        market_trend = np.random.normal(0.0002, 0.01) # global drift
        
        stock_rets_1h = {}
        stock_rets_4h = {}
        stock_rets_1d = {}
        stock_rets_intraday = {}
        
        for s in STOCKS:
            # Generate realistic returns
            # Overnight return (1D)
            ret_1d = np.random.normal(market_trend + 0.0001, 0.008)
            # 4H premarket return
            ret_4h = np.random.normal(market_trend * 0.8, 0.006)
            # 1H premarket return (correlated with 4H return)
            ret_1h = ret_4h * 0.6 + np.random.normal(0, 0.003)
            # Intraday return (from open to close, correlated with pre-market moves)
            ret_intraday = (ret_1h * 0.3 + ret_4h * 0.2 + ret_1d * 0.1) + np.random.normal(0, 0.009)
            
            stock_rets_1h[s] = ret_1h
            stock_rets_4h[s] = ret_4h
            stock_rets_1d[s] = ret_1d
            stock_rets_intraday[s] = ret_intraday
            
            # Calculate actual prices
            prev_close = stock_daily_data[s][-1]['Close']
            today_open = prev_close * (1 + ret_1d)
            today_close = today_open * (1 + ret_intraday)
            
            # Store daily
            stock_daily_data[s].append({
                'Date': f"{date_str} 00:00:00-05:00",
                'Open': today_open,
                'High': max(today_open, today_close) * (1 + abs(np.random.normal(0, 0.004))),
                'Low': min(today_open, today_close) * (1 - abs(np.random.normal(0, 0.004))),
                'Close': today_close,
                'Volume': int(np.random.normal(2000000, 500000))
            })
            
            # Generate pre-market hourly bars (4:00, 5:00, 6:00, 7:00, 8:00, 9:00)
            # We want: 5:00 AM Open to today's open to match ret_4h
            # 8:00 AM Open to today's open to match ret_1h
            open_5 = today_open / (1 + ret_4h)
            open_8 = today_open / (1 + ret_1h)
            
            hours = [4, 5, 6, 7, 8, 9]
            h_prices = {
                4: (open_5 * 0.995, open_5 * 0.998),
                5: (open_5, open_5 * 1.001),
                6: (open_5 * 1.001, open_5 * 1.002),
                7: (open_5 * 1.002, open_8),
                8: (open_8, open_8 * 1.001),
                9: (open_8 * 1.001, today_open)
            }
            
            for h in hours:
                h_open, h_close = h_prices[h]
                stock_hourly_data[s].append({
                    'Datetime': f"{date_str} {h:02d}:00:00-05:00",
                    'Open': h_open,
                    'High': max(h_open, h_close) * 1.001,
                    'Low': min(h_open, h_close) * 0.999,
                    'Close': h_close,
                    'Volume': int(np.random.normal(50000, 10000))
                })
                
        # Generate NQ index return based on stock returns and weights
        nq_weighted_ret = sum(stock_rets_intraday[s] * WEIGHTS[s] for s in STOCKS)
        nq_ret = nq_weighted_ret + np.random.normal(0.0001, 0.002) # index return has minor residual noise
        
        prev_nq_close = index_daily_data[-1]['Close']
        # overnight index gap return matches stocks average overnight
        nq_overnight_ret = sum(stock_rets_1d[s] * WEIGHTS[s] for s in STOCKS)
        nq_open = prev_nq_close * (1 + nq_overnight_ret)
        nq_close = nq_open * (1 + nq_ret)
        
        index_daily_data.append({
            'Date': f"{date_str} 00:00:00-05:00",
            'Open': nq_open,
            'High': max(nq_open, nq_close) * 1.002,
            'Low': min(nq_open, nq_close) * 0.998,
            'Close': nq_close,
            'Volume': int(np.random.normal(500000000, 50000000))
        })
        
    # Save all dataframes to CSV
    pd.DataFrame(index_daily_data).to_csv(os.path.join(DATA_DIR, "index_daily.csv"), index=False)
    
    for s in STOCKS:
        pd.DataFrame(stock_daily_data[s]).to_csv(os.path.join(DATA_DIR, f"{s}_daily.csv"), index=False)
        pd.DataFrame(stock_hourly_data[s]).to_csv(os.path.join(DATA_DIR, f"{s}_hourly.csv"), index=False)
        
    print("\nSynthetic trading dataset successfully generated in 'data/'!")

if __name__ == "__main__":
    main()
