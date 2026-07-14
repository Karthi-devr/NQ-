import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configurations
STOCKS = ["MSFT", "NVDA", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "TSLA", "COST", "AMD"]
INDEX = "^NDX"
DATA_DIR = "data"
MODELS_DIR = "models"

WEIGHTS = {
    'MSFT': 0.16, 'NVDA': 0.15, 'AAPL': 0.14, 'AMZN': 0.09, 'META': 0.08,
    'AVGO': 0.08, 'GOOGL': 0.05, 'TSLA': 0.04, 'COST': 0.03, 'AMD': 0.03
}

def load_index_data():
    path = os.path.join(DATA_DIR, "index_daily.csv")
    df = pd.read_csv(path)
    # Parse dates and normalize to YYYY-MM-DD (handling timezones)
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    df = df.set_index('Date')
    
    # Calculate intraday return: (Close - Open) / Open
    df['nq_ret'] = (df['Close'] - df['Open']) / df['Open']
    
    # Target labeling: Bullish (2), Bearish (0), Neutral (1)
    # Threshold is 0.25% (0.0025)
    def get_target(ret):
        if ret >= 0.0025:
            return 2  # Bullish
        elif ret <= -0.0025:
            return 0  # Bearish
        else:
            return 1  # Neutral
            
    df['target'] = df['nq_ret'].apply(get_target)
    return df[['Open', 'Close', 'nq_ret', 'target']]

def extract_stock_features(symbol):
    print(f"Processing features for {symbol}...")
    
    # Load daily data
    daily_path = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    df_daily = pd.read_csv(daily_path)
    df_daily['Date'] = pd.to_datetime(df_daily['Date'], utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    df_daily = df_daily.set_index('Date')
    
    # Load hourly data
    hourly_path = os.path.join(DATA_DIR, f"{symbol}_hourly.csv")
    df_hourly = pd.read_csv(hourly_path)
    df_hourly['Datetime'] = pd.to_datetime(df_hourly['Datetime'], utc=True).dt.tz_convert('America/New_York')
    df_hourly['Date'] = df_hourly['Datetime'].dt.strftime('%Y-%m-%d')
    df_hourly['Hour'] = df_hourly['Datetime'].dt.hour
    
    features = []
    dates = df_daily.index.tolist()
    
    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        
        # Guard if dates are not found (e.g. index errors)
        try:
            today_open = df_daily.loc[date, 'Open']
            prev_close = df_daily.loc[prev_date, 'Close']
            prev_open = df_daily.loc[prev_date, 'Open']
        except KeyError:
            continue
            
        # Level 6: Momentum (1D overnight gap, and previous day return)
        ret_1d = (today_open - prev_close) / prev_close
        ret_prev_day = (prev_close - prev_open) / prev_open
        
        # Filter hourly premarket bars for today
        df_today_h = df_hourly[df_hourly['Date'] == date]
        
        # 1H pre-market open (Hour == 8 (8 AM ET), fallback to Hour == 9, then Hour == 7)
        open_8 = None
        for hr in [8, 9, 7]:
            bars = df_today_h[df_today_h['Hour'] == hr]
            if not bars.empty:
                open_8 = bars.iloc[0]['Open']
                break
        
        if open_8 is not None:
            ret_1h = (today_open - open_8) / open_8
        else:
            ret_1h = 0.0
            
        # 4H pre-market open (Hour == 5 (5 AM ET), fallback to 6, 4, 7, 8)
        open_5 = None
        for hr in [5, 6, 4, 7, 8]:
            bars = df_today_h[df_today_h['Hour'] == hr]
            if not bars.empty:
                open_5 = bars.iloc[0]['Open']
                break
                
        if open_5 is not None:
            ret_4h = (today_open - open_5) / open_5
        else:
            ret_4h = 0.0
            
        features.append({
            'Date': date,
            f'{symbol}_1h_ret': float(ret_1h),
            f'{symbol}_4h_ret': float(ret_4h),
            f'{symbol}_1d_ret': float(ret_1d),
            f'{symbol}_prev_day_ret': float(ret_prev_day)
        })
        
    return pd.DataFrame(features).set_index('Date')

def engineer_features(df_master):
    print("Engineering features across multiple levels...")
    timeframes = ['1h', '4h', '1d']
    df_eng = df_master.copy()
    
    for tf in timeframes:
        # Level 2: Weighted Importance Features
        weighted_rets = []
        for symbol, weight in WEIGHTS.items():
            df_eng[f'{symbol}_weighted_{tf}_ret'] = df_eng[f'{symbol}_{tf}_ret'] * weight
            weighted_rets.append(df_eng[f'{symbol}_weighted_{tf}_ret'])
        
        df_eng[f'market_weighted_avg_{tf}_ret'] = sum(weighted_rets)
        
        # Level 3: Sector Strength
        df_eng[f'sector_tech_{tf}_ret'] = df_eng[[f'MSFT_{tf}_ret', f'AAPL_{tf}_ret']].mean(axis=1)
        df_eng[f'sector_comm_{tf}_ret'] = df_eng[[f'META_{tf}_ret', f'GOOGL_{tf}_ret']].mean(axis=1)
        df_eng[f'sector_disc_{tf}_ret'] = df_eng[[f'AMZN_{tf}_ret', f'TSLA_{tf}_ret']].mean(axis=1)
        df_eng[f'sector_staples_{tf}_ret'] = df_eng[f'COST_{tf}_ret']
        df_eng[f'sector_semi_{tf}_ret'] = df_eng[[f'NVDA_{tf}_ret', f'AVGO_{tf}_ret', f'AMD_{tf}_ret']].mean(axis=1)
        
        # Level 4: Market Breadth
        ret_cols = [f'{symbol}_{tf}_ret' for symbol in WEIGHTS.keys()]
        df_eng[f'breadth_pos_count_{tf}'] = (df_eng[ret_cols] > 0).sum(axis=1)
        df_eng[f'breadth_neg_count_{tf}'] = (df_eng[ret_cols] < 0).sum(axis=1)
        df_eng[f'breadth_avg_ret_{tf}'] = df_eng[ret_cols].mean(axis=1)
        df_eng[f'breadth_median_ret_{tf}'] = df_eng[ret_cols].median(axis=1)
        df_eng[f'breadth_max_ret_{tf}'] = df_eng[ret_cols].max(axis=1)
        df_eng[f'breadth_min_ret_{tf}'] = df_eng[ret_cols].min(axis=1)
        
        # Level 5: Agreement Score
        df_eng[f'agreement_score_{tf}'] = df_eng[[f'breadth_pos_count_{tf}', f'breadth_neg_count_{tf}']].max(axis=1) / 10.0 * 100.0

    # Level 6: Momentum Spreads
    for symbol in WEIGHTS.keys():
        df_eng[f'{symbol}_mom_1h_4h'] = df_eng[f'{symbol}_1h_ret'] - df_eng[f'{symbol}_4h_ret']
        df_eng[f'{symbol}_mom_1h_1d'] = df_eng[f'{symbol}_1h_ret'] - df_eng[f'{symbol}_1d_ret']

    return df_eng

def get_feature_lists():
    stocks = list(WEIGHTS.keys())
    
    features_1h = [f'{s}_1h_ret' for s in stocks] + \
                  [f'{s}_weighted_1h_ret' for s in stocks] + \
                  ['market_weighted_avg_1h_ret', 
                   'sector_tech_1h_ret', 'sector_comm_1h_ret', 'sector_disc_1h_ret', 'sector_staples_1h_ret', 'sector_semi_1h_ret',
                   'breadth_pos_count_1h', 'breadth_neg_count_1h', 'breadth_avg_ret_1h', 'breadth_median_ret_1h', 'breadth_max_ret_1h', 'breadth_min_ret_1h',
                   'agreement_score_1h']
                  
    features_4h = [f'{s}_4h_ret' for s in stocks] + \
                  [f'{s}_weighted_4h_ret' for s in stocks] + \
                  ['market_weighted_avg_4h_ret', 
                   'sector_tech_4h_ret', 'sector_comm_4h_ret', 'sector_disc_4h_ret', 'sector_staples_4h_ret', 'sector_semi_4h_ret',
                   'breadth_pos_count_4h', 'breadth_neg_count_4h', 'breadth_avg_ret_4h', 'breadth_median_ret_4h', 'breadth_max_ret_4h', 'breadth_min_ret_4h',
                   'agreement_score_4h']
                  
    features_1d = [f'{s}_1d_ret' for s in stocks] + \
                  [f'{s}_prev_day_ret' for s in stocks] + \
                  [f'{s}_weighted_1d_ret' for s in stocks] + \
                  ['market_weighted_avg_1d_ret', 
                   'sector_tech_1d_ret', 'sector_comm_1d_ret', 'sector_disc_1d_ret', 'sector_staples_1d_ret', 'sector_semi_1d_ret',
                   'breadth_pos_count_1d', 'breadth_neg_count_1d', 'breadth_avg_ret_1d', 'breadth_median_ret_1d', 'breadth_max_ret_1d', 'breadth_min_ret_1d',
                   'agreement_score_1d']
                  
    features_general = features_1h + features_4h + features_1d + \
                       [f'{s}_mom_1h_4h' for s in stocks] + \
                       [f'{s}_mom_1h_1d' for s in stocks]
                       
    return features_1h, features_4h, features_1d, features_general

def train_and_evaluate(df, features, target_col, model_name):
    # Filter rows where features aren't all NaN (e.g. initial rows or missing data days)
    df_clean = df.dropna(subset=features + [target_col])
    
    X = df_clean[features]
    y = df_clean[target_col]
    
    print(f"\nTraining {model_name} with {len(X)} samples and {len(features)} features...")
    
    # Chronological Split (Time-based to prevent look-ahead bias)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Distribution check
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Train XGBoost with Hyperparameter Tuning (GridSearchCV)
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.03, 0.05],
        'n_estimators': [50, 100, 150]
    }
    
    xgb = XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f"[{model_name}] Best parameters: {grid_search.best_params_}")
    
    # Predict and evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    print(f"Train Accuracy: {acc_train:.4f} | Test Accuracy: {acc_test:.4f}")
    
    # Classification metrics
    report = classification_report(y_test, y_pred_test, target_names=["Bearish", "Neutral", "Bullish",], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test).tolist()
    
    # Feature importances
    importances = model.feature_importances_.tolist()
    feature_importance_dict = sorted(
        [{"feature": f, "importance": float(imp)} for f, imp in zip(features, importances)],
        key=lambda x: x["importance"],
        reverse=True
    )
    
    # Save the model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, f"{model_name.lower().replace(' ', '_')}.pkl"))
    
    # Prepare metadata
    metrics = {
        "train_accuracy": float(acc_train),
        "test_accuracy": float(acc_test),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importances": feature_importance_dict[:20]  # Top 20 features
    }
    
    return metrics

def main():
    print("Loading Nasdaq 100 Index data...")
    df_index = load_index_data()
    print(f"Index data loaded: {len(df_index)} days. Target distribution:")
    print(df_index['target'].value_counts())
    
    # Process features for all stocks
    stock_dfs = []
    for symbol in STOCKS:
        stock_dfs.append(extract_stock_features(symbol))
        
    print("Merging stock data...")
    # Join all stocks on Date
    df_master = stock_dfs[0]
    for df in stock_dfs[1:]:
        df_master = df_master.join(df, how='outer')
        
    # Clean index timezone mismatch or alignment
    df_merged = df_index.join(df_master, how='inner')
    print(f"Merged dataset shape (before feature engineering): {df_merged.shape}")
    
    # Perform feature engineering
    df_engineered = engineer_features(df_merged)
    print(f"Dataset shape after feature engineering: {df_engineered.shape}")
    
    # Get feature lists for the 4 models
    f_1h, f_4h, f_1d, f_gen = get_feature_lists()
    
    # Train and evaluate the models
    metadata = {}
    metadata["1h"] = train_and_evaluate(df_engineered, f_1h, 'target', "Model 1H")
    metadata["4h"] = train_and_evaluate(df_engineered, f_4h, 'target', "Model 4H")
    metadata["1d"] = train_and_evaluate(df_engineered, f_1d, 'target', "Model 1D")
    metadata["general"] = train_and_evaluate(df_engineered, f_gen, 'target', "Model General")
    
    # Save metadata JSON
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"\nAll models trained and metadata successfully saved to '{metadata_path}'!")

if __name__ == "__main__":
    main()
