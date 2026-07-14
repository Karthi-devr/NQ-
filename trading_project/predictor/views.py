import os
import json
import joblib
import datetime
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib import messages

# Configuration (must match training script)
STOCKS = ["MSFT", "NVDA", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "TSLA", "COST", "AMD"]
WEIGHTS = {
    'MSFT': 0.16, 'NVDA': 0.15, 'AAPL': 0.14, 'AMZN': 0.09, 'META': 0.08,
    'AVGO': 0.08, 'GOOGL': 0.05, 'TSLA': 0.04, 'COST': 0.03, 'AMD': 0.03
}

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # trading_project/predictor → trading_project
REPO_ROOT = BASE_DIR.parent                          # trading_project → repo root
MODELS_DIR = str(REPO_ROOT / 'models')              # repo_root/models


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
        features['sector_tech_1h_ret'] = np.mean([features['MSFT_1h_ret'], features['AAPL_1h_ret']])
        features['sector_comm_1h_ret'] = np.mean([features['META_1h_ret'], features['GOOGL_1h_ret']])
        features['sector_disc_1h_ret'] = np.mean([features['AMZN_1h_ret'], features['TSLA_1h_ret']])
        features['sector_staples_1h_ret'] = features['COST_1h_ret']
        features['sector_semi_1h_ret'] = np.mean([features['NVDA_1h_ret'], features['AVGO_1h_ret'], features['AMD_1h_ret']])
        
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
        features['sector_tech_4h_ret'] = np.mean([features['MSFT_4h_ret'], features['AAPL_4h_ret']])
        features['sector_comm_4h_ret'] = np.mean([features['META_4h_ret'], features['GOOGL_4h_ret']])
        features['sector_disc_4h_ret'] = np.mean([features['AMZN_4h_ret'], features['TSLA_4h_ret']])
        features['sector_staples_4h_ret'] = features['COST_4h_ret']
        features['sector_semi_4h_ret'] = np.mean([features['NVDA_4h_ret'], features['AVGO_4h_ret'], features['AMD_4h_ret']])
        
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
        features['sector_tech_1d_ret'] = np.mean([features['MSFT_1d_ret'], features['AAPL_1d_ret']])
        features['sector_comm_1d_ret'] = np.mean([features['META_1d_ret'], features['GOOGL_1d_ret']])
        features['sector_disc_1d_ret'] = np.mean([features['AMZN_1d_ret'], features['TSLA_1d_ret']])
        features['sector_staples_1d_ret'] = features['COST_1d_ret']
        features['sector_semi_1d_ret'] = np.mean([features['NVDA_1d_ret'], features['AVGO_1d_ret'], features['AMD_1d_ret']])
        
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


# ─────────────────────────────────────────────────────────────────────────────
# REAL TRAINING DATA VIEWS
# ─────────────────────────────────────────────────────────────────────────────

def save_prediction(request):
    """Save today's prediction + inputs to the database for later retraining."""
    from .models import TradingDay
    if request.method == 'POST':
        mode             = request.POST.get('mode', '1h')
        prediction       = request.POST.get('prediction', '')
        confidence       = float(request.POST.get('confidence', 0))
        trade_date_str   = request.POST.get('trade_date', '')

        # Parse raw inputs back from hidden fields
        raw_inputs = {}
        for s in STOCKS:
            for tf in ['1h', '4h', '1d', 'prev_day_ret']:
                key = f'{s}_{tf}'
                val = request.POST.get(key, '')
                try:
                    raw_inputs[f'{s}_{tf}_ret' if tf != 'prev_day_ret' else f'{s}_prev_day_ret'] = float(val) if val.strip() != '' else 0.0
                except ValueError:
                    raw_inputs[f'{s}_{tf}_ret' if tf != 'prev_day_ret' else f'{s}_prev_day_ret'] = 0.0

        # Parse date
        try:
            trade_date = datetime.date.fromisoformat(trade_date_str)
        except (ValueError, TypeError):
            trade_date = datetime.date.today()

        # Allow multiple saves per day (e.g. 1H data at 9:00, 9:15, 9:20)
        TradingDay.objects.create(
            date=trade_date,
            mode=mode,
            raw_inputs=raw_inputs,
            model_prediction=prediction,
            model_confidence=confidence,
            actual_outcome=''
        )
        time_now = datetime.datetime.now().strftime('%H:%M')
        messages.success(request, f'✅ Saved at {time_now}! Enter the actual NQ outcome after market close.')

    return redirect('training_data')


def training_data(request):
    """Show all saved trading days. User can set actual outcome and retrain."""
    from .models import TradingDay

    days        = TradingDay.objects.all()
    total       = days.count()
    complete    = days.filter(actual_outcome__in=['Bullish', 'Bearish', 'Neutral']).count()
    correct     = sum(1 for d in days if d.was_correct)
    accuracy    = round((correct / complete * 100), 1) if complete > 0 else 0

    can_retrain = complete >= 20   # Need at least 20 real days

    context = {
        'days': days,
        'total': total,
        'complete': complete,
        'correct': correct,
        'accuracy': accuracy,
        'can_retrain': can_retrain,
        'min_days': 20,
    }
    return render(request, 'predictor/training_data.html', context)


@require_POST
def update_outcome(request, day_id):
    """Update the actual outcome for a specific saved trading day."""
    from .models import TradingDay
    try:
        day = TradingDay.objects.get(id=day_id)
        outcome = request.POST.get('actual_outcome', '')
        if outcome in ['Bullish', 'Bearish', 'Neutral']:
            day.actual_outcome = outcome
            day.save()
            messages.success(request, f'✅ Outcome updated for {day.date}.')
        else:
            messages.error(request, 'Invalid outcome selected.')
    except TradingDay.DoesNotExist:
        messages.error(request, 'Record not found.')
    return redirect('training_data')


def retrain_model(request):
    """Retrain XGBoost models using all saved real trading days with outcomes."""
    from .models import TradingDay
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    if request.method != 'POST':
        return redirect('training_data')

    # Get all complete days
    days = TradingDay.objects.filter(actual_outcome__in=['Bullish', 'Bearish', 'Neutral'])
    if days.count() < 20:
        messages.error(request, f'Need at least 20 complete days. Currently have {days.count()}.')
        return redirect('training_data')

    # ── Build feature matrix ──────────────────────────────────────────────────
    label_map = {'Bearish': 0, 'Neutral': 1, 'Bullish': 2}
    rows = []
    labels = []

    for day in days:
        inp = {k: v / 100.0 for k, v in day.raw_inputs.items()}   # % → decimal
        feat = {}

        # 1H features
        for s in STOCKS:
            val = inp.get(f'{s}_1h_ret', 0.0)
            feat[f'{s}_1h_ret'] = val
            feat[f'{s}_weighted_1h_ret'] = val * WEIGHTS[s]
        vals_1h = [inp.get(f'{s}_1h_ret', 0.0) for s in STOCKS]
        feat['market_weighted_avg_1h_ret'] = sum(feat[f'{s}_weighted_1h_ret'] for s in STOCKS)
        feat['sector_tech_1h_ret']  = np.mean([feat['MSFT_1h_ret'], feat['AAPL_1h_ret']])
        feat['sector_comm_1h_ret']  = np.mean([feat['META_1h_ret'], feat['GOOGL_1h_ret']])
        feat['sector_disc_1h_ret']  = np.mean([feat['AMZN_1h_ret'], feat['TSLA_1h_ret']])
        feat['sector_staples_1h_ret'] = feat['COST_1h_ret']
        feat['sector_semi_1h_ret']  = np.mean([feat['NVDA_1h_ret'], feat['AVGO_1h_ret'], feat['AMD_1h_ret']])
        pos = sum(1 for r in vals_1h if r > 0)
        neg = sum(1 for r in vals_1h if r < 0)
        feat['breadth_pos_count_1h']  = float(pos)
        feat['breadth_neg_count_1h']  = float(neg)
        feat['breadth_avg_ret_1h']    = float(np.mean(vals_1h))
        feat['breadth_median_ret_1h'] = float(np.median(vals_1h))
        feat['breadth_max_ret_1h']    = float(np.max(vals_1h))
        feat['breadth_min_ret_1h']    = float(np.min(vals_1h))
        feat['agreement_score_1h']    = float(max(pos, neg) / 10.0 * 100.0)

        # 4H features
        for s in STOCKS:
            val = inp.get(f'{s}_4h_ret', 0.0)
            feat[f'{s}_4h_ret'] = val
            feat[f'{s}_weighted_4h_ret'] = val * WEIGHTS[s]
        vals_4h = [inp.get(f'{s}_4h_ret', 0.0) for s in STOCKS]
        feat['market_weighted_avg_4h_ret'] = sum(feat[f'{s}_weighted_4h_ret'] for s in STOCKS)
        feat['sector_tech_4h_ret']  = np.mean([feat['MSFT_4h_ret'], feat['AAPL_4h_ret']])
        feat['sector_comm_4h_ret']  = np.mean([feat['META_4h_ret'], feat['GOOGL_4h_ret']])
        feat['sector_disc_4h_ret']  = np.mean([feat['AMZN_4h_ret'], feat['TSLA_4h_ret']])
        feat['sector_staples_4h_ret'] = feat['COST_4h_ret']
        feat['sector_semi_4h_ret']  = np.mean([feat['NVDA_4h_ret'], feat['AVGO_4h_ret'], feat['AMD_4h_ret']])
        pos4 = sum(1 for r in vals_4h if r > 0)
        neg4 = sum(1 for r in vals_4h if r < 0)
        feat['breadth_pos_count_4h']  = float(pos4)
        feat['breadth_neg_count_4h']  = float(neg4)
        feat['breadth_avg_ret_4h']    = float(np.mean(vals_4h))
        feat['breadth_median_ret_4h'] = float(np.median(vals_4h))
        feat['breadth_max_ret_4h']    = float(np.max(vals_4h))
        feat['breadth_min_ret_4h']    = float(np.min(vals_4h))
        feat['agreement_score_4h']    = float(max(pos4, neg4) / 10.0 * 100.0)

        # 1D features
        for s in STOCKS:
            val1d = inp.get(f'{s}_1d_ret', 0.0)
            valpd = inp.get(f'{s}_prev_day_ret', 0.0)
            feat[f'{s}_1d_ret'] = val1d
            feat[f'{s}_prev_day_ret'] = valpd
            feat[f'{s}_weighted_1d_ret'] = val1d * WEIGHTS[s]
        vals_1d = [inp.get(f'{s}_1d_ret', 0.0) for s in STOCKS]
        feat['market_weighted_avg_1d_ret'] = sum(feat[f'{s}_weighted_1d_ret'] for s in STOCKS)
        feat['sector_tech_1d_ret']  = np.mean([feat['MSFT_1d_ret'], feat['AAPL_1d_ret']])
        feat['sector_comm_1d_ret']  = np.mean([feat['META_1d_ret'], feat['GOOGL_1d_ret']])
        feat['sector_disc_1d_ret']  = np.mean([feat['AMZN_1d_ret'], feat['TSLA_1d_ret']])
        feat['sector_staples_1d_ret'] = feat['COST_1d_ret']
        feat['sector_semi_1d_ret']  = np.mean([feat['NVDA_1d_ret'], feat['AVGO_1d_ret'], feat['AMD_1d_ret']])
        pos1d = sum(1 for r in vals_1d if r > 0)
        neg1d = sum(1 for r in vals_1d if r < 0)
        feat['breadth_pos_count_1d']  = float(pos1d)
        feat['breadth_neg_count_1d']  = float(neg1d)
        feat['breadth_avg_ret_1d']    = float(np.mean(vals_1d))
        feat['breadth_median_ret_1d'] = float(np.median(vals_1d))
        feat['breadth_max_ret_1d']    = float(np.max(vals_1d))
        feat['breadth_min_ret_1d']    = float(np.min(vals_1d))
        feat['agreement_score_1d']    = float(max(pos1d, neg1d) / 10.0 * 100.0)

        # Momentum spreads
        for s in STOCKS:
            feat[f'{s}_mom_1h_4h'] = feat[f'{s}_1h_ret'] - feat[f'{s}_4h_ret']
            feat[f'{s}_mom_1h_1d'] = feat[f'{s}_1h_ret'] - feat[f'{s}_1d_ret']

        rows.append(feat)
        labels.append(label_map[day.actual_outcome])

    df_all = pd.DataFrame(rows)
    y = np.array(labels)

    # ── Feature sets ─────────────────────────────────────────────────────────
    stocks = list(WEIGHTS.keys())
    f_1h  = [f'{s}_1h_ret' for s in stocks] + [f'{s}_weighted_1h_ret' for s in stocks] + \
            ['market_weighted_avg_1h_ret', 'sector_tech_1h_ret', 'sector_comm_1h_ret',
             'sector_disc_1h_ret', 'sector_staples_1h_ret', 'sector_semi_1h_ret',
             'breadth_pos_count_1h', 'breadth_neg_count_1h', 'breadth_avg_ret_1h',
             'breadth_median_ret_1h', 'breadth_max_ret_1h', 'breadth_min_ret_1h', 'agreement_score_1h']
    f_4h  = [f'{s}_4h_ret' for s in stocks] + [f'{s}_weighted_4h_ret' for s in stocks] + \
            ['market_weighted_avg_4h_ret', 'sector_tech_4h_ret', 'sector_comm_4h_ret',
             'sector_disc_4h_ret', 'sector_staples_4h_ret', 'sector_semi_4h_ret',
             'breadth_pos_count_4h', 'breadth_neg_count_4h', 'breadth_avg_ret_4h',
             'breadth_median_ret_4h', 'breadth_max_ret_4h', 'breadth_min_ret_4h', 'agreement_score_4h']
    f_1d  = [f'{s}_1d_ret' for s in stocks] + [f'{s}_prev_day_ret' for s in stocks] + \
            [f'{s}_weighted_1d_ret' for s in stocks] + \
            ['market_weighted_avg_1d_ret', 'sector_tech_1d_ret', 'sector_comm_1d_ret',
             'sector_disc_1d_ret', 'sector_staples_1d_ret', 'sector_semi_1d_ret',
             'breadth_pos_count_1d', 'breadth_neg_count_1d', 'breadth_avg_ret_1d',
             'breadth_median_ret_1d', 'breadth_max_ret_1d', 'breadth_min_ret_1d', 'agreement_score_1d']
    f_gen = f_1h + f_4h + f_1d + [f'{s}_mom_1h_4h' for s in stocks] + [f'{s}_mom_1h_1d' for s in stocks]

    # ── Train each model ──────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    metadata = {}
    label_names = ['Bearish', 'Neutral', 'Bullish']

    for mode_key, features in [('1h', f_1h), ('4h', f_4h), ('1d', f_1d), ('general', f_gen)]:
        X = df_all[features].fillna(0.0)

        if len(X) < 10:
            continue

        split = max(1, int(len(X) * 0.8))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]

        param_grid = {'max_depth': [2, 3], 'learning_rate': [0.03, 0.05], 'n_estimators': [50, 100]}
        xgb = XGBClassifier(subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss')
        gs  = GridSearchCV(xgb, param_grid, cv=min(3, split), scoring='accuracy', n_jobs=-1)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_

        y_pred = model.predict(X_test) if len(X_test) > 0 else model.predict(X_train)
        y_true = y_test if len(X_test) > 0 else y_train
        acc    = float(accuracy_score(y_true, y_pred))
        report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
        imps   = sorted([{'feature': f, 'importance': float(i)} for f, i in zip(features, model.feature_importances_)],
                        key=lambda x: x['importance'], reverse=True)

        joblib.dump(model, os.path.join(MODELS_DIR, f'model_{mode_key}.pkl'))
        metadata[mode_key] = {
            'train_accuracy': float(accuracy_score(y_train, model.predict(X_train))),
            'test_accuracy': acc,
            'precision_macro': float(report['macro avg']['precision']),
            'recall_macro': float(report['macro avg']['recall']),
            'f1_macro': float(report['macro avg']['f1-score']),
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importances': imps[:20],
            'trained_on_real_days': int(days.count()),
            'best_params': gs.best_params_,
        }

    # Save metadata & clear model cache so next request reloads fresh models
    global _models, _metadata
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    _models = {}
    _metadata = None

    messages.success(request, f'✅ Models retrained on {days.count()} real trading days! Accuracy: {round(metadata.get("1h", {}).get("test_accuracy", 0)*100, 1)}%')
    return redirect('training_data')
