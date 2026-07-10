import os
import json
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse

# Configuration (must match training script)
STOCKS = ["MSFT", "AAPL", "NVDA", "AMZN", "META", "AVGO", "TSLA", "COST", "GOOGL", "GOOG"]
WEIGHTS = {
    'MSFT': 0.17, 'AAPL': 0.17, 'NVDA': 0.16, 'AMZN': 0.10, 'META': 0.09,
    'AVGO': 0.10, 'TSLA': 0.07, 'COST': 0.04, 'GOOGL': 0.05, 'GOOG': 0.05
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

# Cache for models and metadata to avoid reloading on every request
_models = {}
_metadata = None

def load_models_and_metadata():
    global _metadata
    # Load metadata
    if _metadata is None:
        metadata_path = os.path.join(MODELS_DIR, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                raw_meta = json.load(f)
            
            # Preprocess metadata for template rendering
            _metadata = {}
            for mode, data in raw_meta.items():
                _metadata[mode] = data.copy()
                processed_imp = []
                for item in data.get('feature_importances', []):
                    feat_display = item['feature'].replace('_ret', '').replace('_', ' ').upper()
                    processed_imp.append({
                        'feature': feat_display,
                        'importance': item['importance'],
                        'importance_pct': round(item['importance'] * 100, 1)
                    })
                _metadata[mode]['feature_importances_display'] = processed_imp
        else:
            _metadata = {}
            
    # Load specific models as needed
    modes = ['1h', '4h', '1d', 'general']
    for mode in modes:
        if mode not in _models:
            model_path = os.path.join(MODELS_DIR, f"model_{mode}.pkl")
            if os.path.exists(model_path):
                _models[mode] = joblib.load(model_path)
            else:
                _models[mode] = None



# Single-row Feature Engineering
def engineer_features_single(input_data, mode):
    # input_data is a dict containing raw returns (in %)
    # Convert returns back to decimals for model feature calculations
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
            features[f'{s}_prev_day_ret'] = data.get(f'{s}_prev_day_ret', 0.0) # we assume 0 or fetched value
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
    
    # Sort columns in exact order as the models expect
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

def dashboard(request):
    load_models_and_metadata()
    
    # Check if models are available. If not, notify user
    models_ready = all(_models.values())
    
    context = {
        'stocks': STOCKS,
        'weights': WEIGHTS,
        'models_ready': models_ready,
        'selected_mode': 'general',
        'prediction': None,
        'probabilities': None,
        'agreement_score': None,
        'market_breadth': None,
        'sector_strength': None,
        'input_values': {}
    }
    
    # Default returns to 0.0 for manual input
    default_returns = {
        '1h': {s: 0.0 for s in STOCKS},
        '4h': {s: 0.0 for s in STOCKS},
        '1d': {s: 0.0 for s in STOCKS},
        'prev_day_ret': {s: 0.0 for s in STOCKS}
    }
    
    # Set default values in context
    for s in STOCKS:
        context['input_values'][f'{s}_1h'] = default_returns['1h'][s]
        context['input_values'][f'{s}_4h'] = default_returns['4h'][s]
        context['input_values'][f'{s}_1d'] = default_returns['1d'][s]
        context['input_values'][f'{s}_prev_day_ret'] = default_returns['prev_day_ret'][s]
        
    if request.method == 'POST' and models_ready:
        mode = request.POST.get('mode', 'general')
        context['selected_mode'] = mode
        
        def get_float(key, default=0.0):
            val = request.POST.get(key, '')
            if val is None or str(val).strip() == '':
                return default
            try:
                return float(val)
            except ValueError:
                return default
                
        # Read form returns
        raw_inputs = {}
        for s in STOCKS:
            if mode in ['1h', 'general']:
                raw_inputs[f'{s}_1h_ret'] = get_float(f'{s}_1h')
            if mode in ['4h', 'general']:
                raw_inputs[f'{s}_4h_ret'] = get_float(f'{s}_4h')
            if mode in ['1d', 'general']:
                raw_inputs[f'{s}_1d_ret'] = get_float(f'{s}_1d')
                raw_inputs[f'{s}_prev_day_ret'] = get_float(f'{s}_prev_day_ret')
                
        # Update input values in context so form stays populated with submitted inputs
        for s in STOCKS:
            if mode in ['1h', 'general']:
                context['input_values'][f'{s}_1h'] = raw_inputs[f'{s}_1h_ret']
            if mode in ['4h', 'general']:
                context['input_values'][f'{s}_4h'] = raw_inputs[f'{s}_4h_ret']
            if mode in ['1d', 'general']:
                context['input_values'][f'{s}_1d'] = raw_inputs[f'{s}_1d_ret']
                context['input_values'][f'{s}_prev_day_ret'] = raw_inputs[f'{s}_prev_day_ret']

        try:
            # Run feature engineering
            df_feat = engineer_features_single(raw_inputs, mode)
            
            # Predict
            model = _models[mode]
            pred_class = int(model.predict(df_feat)[0])
            pred_proba = model.predict_proba(df_feat)[0].tolist() # [bearish_prob, neutral_prob, bullish_prob]
            
            # Class labels
            class_labels = {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}
            original_prediction = class_labels[pred_class]
            
            # Confidence Thresholding: override if direction prediction has probability < 50%
            CONFIDENCE_THRESHOLD = 0.50
            if pred_class in [0, 2] and pred_proba[pred_class] < CONFIDENCE_THRESHOLD:
                context['prediction'] = 'Neutral (Low Confidence)'
            else:
                context['prediction'] = original_prediction
                
            context['probabilities'] = {
                'Bearish': round(pred_proba[0] * 100, 1),
                'Neutral': round(pred_proba[1] * 100, 1),
                'Bullish': round(pred_proba[2] * 100, 1)
            }
            
            # Local Feature Impact: Calculate weighted return contributions
            contributions = []
            for s in STOCKS:
                tf_key = '1h' if mode == '1h' else ('4h' if mode == '4h' else '1d')
                ret_val = raw_inputs.get(f'{s}_{tf_key}_ret', 0.0)
                weight = WEIGHTS[s]
                impact = ret_val * weight
                contributions.append({
                    'symbol': s,
                    'return': round(ret_val, 2),
                    'impact': round(impact, 3)
                })
                
            # Separate into contributors and detractors
            context['top_contributors'] = [c for c in sorted(contributions, key=lambda x: x['impact'], reverse=True) if c['impact'] > 0][:3]
            context['top_detractors'] = [c for c in sorted(contributions, key=lambda x: x['impact']) if c['impact'] < 0][:3]
            
            # Extract additional visual context metrics from engineered row
            # Use whichever timeframe is relevant (or general -> use 1h as proxy)
            tf_prefix = '1h' if mode == '1h' else ('4h' if mode == '4h' else '1d')
            
            context['agreement_score'] = round(df_feat.loc[0, f'agreement_score_{tf_prefix}'], 1)
            context['market_breadth'] = {
                'positive': int(df_feat.loc[0, f'breadth_pos_count_{tf_prefix}']),
                'negative': int(df_feat.loc[0, f'breadth_neg_count_{tf_prefix}']),
                'average': round(df_feat.loc[0, f'breadth_avg_ret_{tf_prefix}'] * 100, 2),
                'max': round(df_feat.loc[0, f'breadth_max_ret_{tf_prefix}'] * 100, 2),
                'min': round(df_feat.loc[0, f'breadth_min_ret_{tf_prefix}'] * 100, 2),
            }
            context['sector_strength'] = {
                'Tech': round(df_feat.loc[0, f'sector_tech_{tf_prefix}_ret'] * 100, 2),
                'Comm': round(df_feat.loc[0, f'sector_comm_{tf_prefix}_ret'] * 100, 2),
                'Discretionary': round(df_feat.loc[0, f'sector_disc_{tf_prefix}_ret'] * 100, 2),
                'Staples': round(df_feat.loc[0, f'sector_staples_{tf_prefix}_ret'] * 100, 2),
                'Semiconductor': round(df_feat.loc[0, f'sector_semi_{tf_prefix}_ret'] * 100, 2)
            }
            
            # Include feature importances for Chart.js
            if _metadata and mode in _metadata:
                context['feature_importances'] = _metadata[mode]['feature_importances']
                
        except Exception as e:
            context['error'] = f"Prediction failed: {e}"
            
    # Include feature importances on GET as well if metadata exists
    elif _metadata and 'general' in _metadata:
        context['feature_importances'] = _metadata['general']['feature_importances']
        
    # Build a structured list for easy template rendering
    stock_data = []
    for s in STOCKS:
        stock_data.append({
            'symbol': s,
            'weight': WEIGHTS[s],
            'val_1h': context['input_values'][f'{s}_1h'],
            'val_4h': context['input_values'][f'{s}_4h'],
            'val_1d': context['input_values'][f'{s}_1d'],
            'val_prev_day': context['input_values'][f'{s}_prev_day_ret'],
        })
    context['stock_data'] = stock_data
    
    return render(request, 'predictor/dashboard.html', context)

def performance(request):
    load_models_and_metadata()
    
    context = {
        'metadata': _metadata,
        'models_ready': _metadata is not None and len(_metadata) > 0
    }
    
    return render(request, 'predictor/performance.html', context)
