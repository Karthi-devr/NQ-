import yfinance as yf
import pandas as pd

def main():
    print("Testing yfinance for NVDA daily and hourly data...")
    nvda = yf.Ticker("NVDA")
    
    # 1. Fetch daily data
    print("\n--- Daily Data (last 5 days) ---")
    df_daily = nvda.history(period="5d", interval="1d")
    print(df_daily)
    
    # 2. Fetch hourly data with pre-market
    print("\n--- Hourly Data with Prepost (last 2 days) ---")
    df_hourly = nvda.history(period="2d", interval="1h", prepost=True)
    print(df_hourly.head(20))
    
    # 3. Fetch NQ index daily data
    print("\n--- NQ Index (NDX) Daily Data (last 5 days) ---")
    ndx = yf.Ticker("^NDX")
    df_ndx = ndx.history(period="5d", interval="1d")
    print(df_ndx)

if __name__ == "__main__":
    main()
