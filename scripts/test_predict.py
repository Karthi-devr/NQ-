import os
import joblib
import pandas as pd
import numpy as np

# Configurations
STOCKS = ["MSFT", "AAPL", "NVDA", "AMZN", "META", "AVGO", "TSLA", "COST", "GOOGL", "GOOG"]
WEIGHTS = {
    'MSFT': 0.17, 'AAPL': 0.17, 'NVDA': 0.16, 'AMZN': 0.10, 'META': 0.09,
    'AVGO': 0.10, 'TSLA': 0.07, 'COST': 0.04, 'GOOGL': 0.05, 'GOOG': 0.05
}
MODELS_DIR = "models"

def load_model(mode):
    path = os.path.join(MODELS_DIR, f"model_{mode}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)

def engineer_features_single(input_data, mode):
    data = {k: v / 100.0 for k, v in input_data.items()}
    features = {}
    
    # 1. 1H features
    if mode in ['1h', 'general']:
        weighted_1h = []
        for s in STOCKS:
            val = data.get(f'{s}_1h_ret', 0.0)
            features[f'{s}_1h_ret'] = val
            features[f'{s}_weighted_1h_ret'] = val * WEIGHTS[s]
            weighted_1h.append(features[f'{s}_weighted_1h_ret'])
            
        features['market_weighted_avg_1h_ret'] = sum(weighted_1h)
        features['sector_tech_1h_ret'] = np.mean([features['MSFT_1h_ret'], features['AAPL_1h_ret'], features['AVGO_1h_ret']])
        features['sector_comm_1h_ret'] = np.mean([features['META_1h_ret'], features['GOOGL_1h_ret'], features['GOOG_1h_ret']])
        features['sector_disc_1h_ret'] = np.mean([features['AMZN_1h_ret'], features['TSLA_1h_ret']])
        features['sector_staples_1h_ret'] = features['COST_1h_ret']
        features['sector_semi_1h_ret'] = np.mean([features['NVDA_1h_ret'], features['AVGO_1h_ret']])
        
        ret_cols_1h = [features[f'{s}_1h_ret'] for s in STOCKS]
        pos_count = sum(1 for r in ret_cols_1h if r > 0)
        neg_count = sum(1 for r in ret_cols_1h if r < 0)
        features['breadth_pos_count_1h'] = float(pos_count)
        features['breadth_neg_count_1h'] = float(neg_count)
        features['breadth_avg_ret_1h'] = float(np.mean(ret_cols_1h))
        features['breadth_median_ret_1h'] = float(np.median(ret_cols_1h))
        features['breadth_max_ret_1h'] = float(np.max(ret_cols_1h))
        features['breadth_min_ret_1h'] = float(np.min(ret_cols_1h))
        features['agreement_score_1h'] = float(max(pos_count, neg_count) / 10.0 * 100.0)

    # 2. 4H features
    if mode in ['4h', 'general']:
        weighted_4h = []
        for s in STOCKS:
            val = data.get(f'{s}_4h_ret', 0.0)
            features[f'{s}_4h_ret'] = val
            features[f'{s}_weighted_4h_ret'] = val * WEIGHTS[s]
            weighted_4h.append(features[f'{s}_weighted_4h_ret'])
            
        features['market_weighted_avg_4h_ret'] = sum(weighted_4h)
        features['sector_tech_4h_ret'] = np.mean([features['MSFT_4h_ret'], features['AAPL_4h_ret'], features['AVGO_4h_ret']])
        features['sector_comm_4h_ret'] = np.mean([features['META_4h_ret'], features['GOOGL_4h_ret'], features['GOOG_4h_ret']])
        features['sector_disc_4h_ret'] = np.mean([features['AMZN_4h_ret'], features['TSLA_4h_ret']])
        features['sector_staples_4h_ret'] = features['COST_4h_ret']
        features['sector_semi_4h_ret'] = np.mean([features['NVDA_4h_ret'], features['AVGO_4h_ret']])
        
        ret_cols_4h = [features[f'{s}_4h_ret'] for s in STOCKS]
        pos_count = sum(1 for r in ret_cols_4h if r > 0)
        neg_count = sum(1 for r in ret_cols_4h if r < 0)
        features['breadth_pos_count_4h'] = float(pos_count)
        features['breadth_neg_count_4h'] = float(neg_count)
        features['breadth_avg_ret_4h'] = float(np.mean(ret_cols_4h))
        features['breadth_median_ret_4h'] = float(np.median(ret_cols_4h))
        features['breadth_max_ret_4h'] = float(np.max(ret_cols_4h))
        features['breadth_min_ret_4h'] = float(np.min(ret_cols_4h))
        features['agreement_score_4h'] = float(max(pos_count, neg_count) / 10.0 * 100.0)

    # 3. 1D features
    if mode in ['1d', 'general']:
        weighted_1d = []
        for s in STOCKS:
            val = data.get(f'{s}_1d_ret', 0.0)
            features[f'{s}_1d_ret'] = val
            features[f'{s}_prev_day_ret'] = data.get(f'{s}_prev_day_ret', 0.0)
            features[f'{s}_weighted_1d_ret'] = val * WEIGHTS[s]
            weighted_1d.append(features[f'{s}_weighted_1d_ret'])
            
        features['market_weighted_avg_1d_ret'] = sum(weighted_1d)
        features['sector_tech_1d_ret'] = np.mean([features['MSFT_1d_ret'], features['AAPL_1d_ret'], features['AVGO_1d_ret']])
        features['sector_comm_1d_ret'] = np.mean([features['META_1d_ret'], features['GOOGL_1d_ret'], features['GOOG_1d_ret']])
        features['sector_disc_1d_ret'] = np.mean([features['AMZN_1d_ret'], features['TSLA_1d_ret']])
        features['sector_staples_1d_ret'] = features['COST_1d_ret']
        features['sector_semi_1d_ret'] = np.mean([features['NVDA_1d_ret'], features['AVGO_1d_ret']])
        
        ret_cols_1d = [features[f'{s}_1d_ret'] for s in STOCKS]
        pos_count = sum(1 for r in ret_cols_1d if r > 0)
        neg_count = sum(1 for r in ret_cols_1d if r < 0)
        features['breadth_pos_count_1d'] = float(pos_count)
        features['breadth_neg_count_1d'] = float(neg_count)
        features['breadth_avg_ret_1d'] = float(np.mean(ret_cols_1d))
        features['breadth_median_ret_1d'] = float(np.median(ret_cols_1d))
        features['breadth_max_ret_1d'] = float(np.max(ret_cols_1d))
        features['breadth_min_ret_1d'] = float(np.min(ret_cols_1d))
        features['agreement_score_1d'] = float(max(pos_count, neg_count) / 10.0 * 100.0)

    # 4. Spreads (General mode only)
    if mode == 'general':
        for s in STOCKS:
            features[f'{s}_mom_1h_4h'] = features[f'{s}_1h_ret'] - features[f'{s}_4h_ret']
            features[f'{s}_mom_1h_1d'] = features[f'{s}_1h_ret'] - features[f'{s}_1d_ret']

    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Sort columns in exact order
    ordered_cols = []
    if mode == '1h':
        ordered_cols = [f'{s}_1h_ret' for s in STOCKS] + [f'{s}_weighted_1h_ret' for s in STOCKS] + [
            'market_weighted_avg_1h_ret', 'sector_tech_1h_ret', 'sector_comm_1h_ret', 'sector_disc_1h_ret', 'sector_staples_1h_ret', 'sector_semi_1h_ret',
            'breadth_pos_count_1h', 'breadth_neg_count_1h', 'breadth_avg_ret_1h', 'breadth_median_ret_1h', 'breadth_max_ret_1h', 'breadth_min_ret_1h', 'agreement_score_1h'
        ]
    elif mode == '4h':
        ordered_cols = [f'{s}_4h_ret' for s in STOCKS] + [f'{s}_weighted_4h_ret' for s in STOCKS] + [
            'market_weighted_avg_4h_ret', 'sector_tech_4h_ret', 'sector_comm_4h_ret', 'sector_disc_4h_ret', 'sector_staples_4h_ret', 'sector_semi_4h_ret',
            'breadth_pos_count_4h', 'breadth_neg_count_4h', 'breadth_avg_ret_4h', 'breadth_median_ret_4h', 'breadth_max_ret_4h', 'breadth_min_ret_4h', 'agreement_score_4h'
        ]
    elif mode == '1d':
        ordered_cols = [f'{s}_1d_ret' for s in STOCKS] + [f'{s}_prev_day_ret' for s in STOCKS] + [f'{s}_weighted_1d_ret' for s in STOCKS] + [
            'market_weighted_avg_1d_ret', 'sector_tech_1d_ret', 'sector_comm_1d_ret', 'sector_disc_1d_ret', 'sector_staples_1d_ret', 'sector_semi_1d_ret',
            'breadth_pos_count_1d', 'breadth_neg_count_1d', 'breadth_avg_ret_1d', 'breadth_median_ret_1d', 'breadth_max_ret_1d', 'breadth_min_ret_1d', 'agreement_score_1d'
        ]
    elif mode == 'general':
        cols_1h = [f'{s}_1h_ret' for s in STOCKS] + [f'{s}_weighted_1h_ret' for s in STOCKS] + [
            'market_weighted_avg_1h_ret', 'sector_tech_1h_ret', 'sector_comm_1h_ret', 'sector_disc_1h_ret', 'sector_staples_1h_ret', 'sector_semi_1h_ret',
            'breadth_pos_count_1h', 'breadth_neg_count_1h', 'breadth_avg_ret_1h', 'breadth_median_ret_1h', 'breadth_max_ret_1h', 'breadth_min_ret_1h', 'agreement_score_1h'
        ]
        cols_4h = [f'{s}_4h_ret' for s in STOCKS] + [f'{s}_weighted_4h_ret' for s in STOCKS] + [
            'market_weighted_avg_4h_ret', 'sector_tech_4h_ret', 'sector_comm_4h_ret', 'sector_disc_4h_ret', 'sector_staples_4h_ret', 'sector_semi_4h_ret',
            'breadth_pos_count_4h', 'breadth_neg_count_4h', 'breadth_avg_ret_4h', 'breadth_median_ret_4h', 'breadth_max_ret_4h', 'breadth_min_ret_4h', 'agreement_score_4h'
        ]
        cols_1d = [f'{s}_1d_ret' for s in STOCKS] + [f'{s}_prev_day_ret' for s in STOCKS] + [f'{s}_weighted_1d_ret' for s in STOCKS] + [
            'market_weighted_avg_1d_ret', 'sector_tech_1d_ret', 'sector_comm_1d_ret', 'sector_disc_1d_ret', 'sector_staples_1d_ret', 'sector_semi_1d_ret',
            'breadth_pos_count_1d', 'breadth_neg_count_1d', 'breadth_avg_ret_1d', 'breadth_median_ret_1d', 'breadth_max_ret_1d', 'breadth_min_ret_1d', 'agreement_score_1d'
        ]
        cols_spreads = [f'{s}_mom_1h_4h' for s in STOCKS] + [f'{s}_mom_1h_1d' for s in STOCKS]
        ordered_cols = cols_1h + cols_4h + cols_1d + cols_spreads

    return df[ordered_cols]

def run_test():
    print("Running Model Load & Inference Verification Test...")
    
    # 1. Mock inputs (Bullish scenario: all stocks positive return)
    bullish_inputs = {}
    for s in STOCKS:
        bullish_inputs[f'{s}_1h_ret'] = 0.5   # +0.5% return
        bullish_inputs[f'{s}_4h_ret'] = 1.2   # +1.2% return
        bullish_inputs[f'{s}_1d_ret'] = 0.4   # +0.4% overnight
        bullish_inputs[f'{s}_prev_day_ret'] = 0.8 # +0.8% previous day
        
    modes = ['1h', '4h', '1d', 'general']
    class_labels = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}
    
    for mode in modes:
        print(f"\n--- Testing Mode: {mode.upper()} ---")
        try:
            # Load model
            model = load_model(mode)
            print("Model loaded successfully.")
            
            # Engineer single row
            df_row = engineer_features_single(bullish_inputs, mode)
            print(f"Feature engineering completed. Row shape: {df_row.shape}")
            
            # Run prediction
            pred = model.predict(df_row)[0]
            proba = model.predict_proba(df_row)[0]
            
            print(f"Prediction: {class_labels[pred]} (Class {pred})")
            print(f"Probabilities: Bearish: {proba[0]*100:.1f}%, Neutral: {proba[1]*100:.1f}%, Bullish: {proba[2]*100:.1f}%")
            
        except Exception as e:
            print(f"Failed to test mode {mode}: {e}")
            
    print("\nVerification Test Completed!")

if __name__ == "__main__":
    run_test()
