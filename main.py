# app.py
import os
import sys
import time
import logging
from pathlib import Path

# Ensure required directories exist
Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# ---- Infinity Bot Code----

### Fully improved trading bot code

```python
import os
import time
import threading
from collections import defaultdict, deque
from pathlib import Path
import yaml
import requests
import websocket
import numpy as np
import pandas as pd
import joblib
import ta
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from hmmlearn import hmm
from sklearn.decomposition import PCA
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedTradingBot")

# Load config
CONFIG_PATH = Path("config.yaml")
if not CONFIG_PATH.exists():
    raise SystemExit("Missing config.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Constants
API_TOKEN = os.getenv("DERIV_API_TOKEN", CONFIG.get("app", {}).get("api_token"))
APP_ID = CONFIG.get("app", {}).get("app_id", 1089)
DERIV_MODE = os.getenv("DERIV_MODE", CONFIG.get("app", {}).get("env", "DEMO")).upper()
SYMBOLS = CONFIG.get("symbols", ["EURUSD", "GBPUSD", "XAUUSD", "R_75"])
TRADE_COOLDOWN = int(CONFIG.get("app", {}).get("trade_cooldown", 30))
MIN_STAKE = float(CONFIG.get("app", {}).get("min_stake", 1))
MAX_DAILY_LOSS = float(CONFIG.get("app", {}).get("max_daily_loss", 0.1))
FEATURES = [
    "rsi", "ema_fast", "ema_slow", "macd", "signal", "bollinger_hi", "bollinger_lo", "atr",
    "lag1", "lag2", "return_1m", "return_5m", "return_15m", "trend_slope",
    "bb_width", "price_deviation", "corr_EURUSD", "corr_GBPUSD",
    "volume_spike", "macd_cross", "rsi_boll_width", "market_regime",
    "volatility", "order_book_imbalance", "sentiment_score",
    "macro_event", "market_trend", "pca1", "pca2"
]
WS_BASE = (f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"
           if DERIV_MODE == "DEMO" else f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}")

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Global state
last_trade_time = {}
recent_trades = deque(maxlen=2000)
symbol_perf = defaultdict(lambda: {'wins': 0, 'losses': 0, 'accuracy': 0, 'enabled': True})
symbol_models = {}
symbol_scalers = {}
symbol_meta_models = {}
symbol_meta_scalers = {}
symbol_iso_regressors = {}
symbol_meta_regressors = {}
last_retrain_time = 0

# --- Data fetching and feature engineering ---

def get_market_data(symbol, count=5000):
    url = f"https://deriv-api.io/api/v1/ohlc?symbol={symbol}&granularity=60&count={count}"
    res = requests.get(url).json()
    df = pd.DataFrame(res.get("candles", []))
    if df.empty:
        return pd.DataFrame()
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["volume"] = df.get("volume", pd.Series([1] * len(df)))
    df.index = pd.to_datetime(df["epoch"], unit='s')
    df.sort_index(inplace=True)
    # Indicators
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=30).ema_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df["close"])
    df["bollinger_hi"] = bb.bollinger_hband()
    df["bollinger_lo"] = bb.bollinger_lband()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"])
    df["atr"] = atr.average_true_range()

    # Placeholder: real order book, sentiment, macro data
    df["order_book_imbalance"] = get_order_book_imbalance(symbol)
    df["sentiment_score"] = get_market_sentiment(symbol)
    df["macro_event"] = get_macro_event_score()

    # Market regime via HMM
    df["market_regime"] = classify_market_regime(df)

    # Cross-symbol correlations
    for other_symbol in SYMBOLS:
        if other_symbol != symbol:
            if other_symbol not in symbol_data_cache:
                symbol_data_cache[other_symbol] = get_market_data(other_symbol)
            other_df = symbol_data_cache[other_symbol]
            common_idx = df.index.intersection(other_df.index)
            if not common_idx.empty:
                corr_series = df.loc[common_idx, "close"].rolling(window=60).corr(other_df.loc[common_idx, "close"])
                df.loc[common_idx, f"corr_{other_symbol}"] = corr_series

    # Volatility interaction
    df["volatility"] = df["atr"] / df["close"]

    # Add more features
    df["lag1"] = df["close"].shift(1)
    df["lag2"] = df["close"].shift(2)
    df["return_1m"] = df["close"].pct_change(1)
    df["return_5m"] = df["close"].pct_change(5)
    df["return_15m"] = df["close"].pct_change(15)
    df["trend_slope"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator().diff()

    # Add PCA features for dimensionality reduction
    pca_features = df[["close", "volume", "rsi", "macd", "bollinger_hi", "bollinger_lo", "atr"]].dropna()
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(pca_features)
    df["pca1"] = np.nan
    df["pca2"] = np.nan
    df.loc[pca_features.index, "pca1"] = pca_components[:,0]
    df.loc[pca_features.index, "pca2"] = pca_components[:,1]

    # Normalize features
    for col in FEATURES:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

    df.dropna(inplace=True)
    return df

def get_order_book_imbalance(symbol):
    # TODO: replace with real order book data
    return np.random.uniform(-1, 1)

def get_market_sentiment(symbol):
    # TODO: sentiment analysis
    return np.random.uniform(-1, 1)

def get_macro_event_score():
    # TODO: macro news impact
    return np.random.uniform(-1, 1)

def classify_market_regime(df):
    # Using HMM on ATR and Bollinger width
    features = np.vstack([
        df["atr"].fillna(0).values,
        (df["bollinger_hi"] - df["bollinger_lo"]).fillna(0).values
    ]).T
    try:
        model = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=100)
        regimes = model.fit(features).predict(features)
        return regimes
    except:
        return np.zeros(len(df))

# --- Model training ---

def save_model(symbol, models_dict, scaler, meta_model, iso_regressor, meta_scaler):
    joblib.dump(models_dict['xgb'], MODELS_DIR / f"{symbol}_xgb.pkl")
    joblib.dump(models_dict['lgb'], MODELS_DIR / f"{symbol}_lgb.pkl")
    models_dict['lstm'].save(str(MODELS_DIR / f"{symbol}_lstm.h5"))
    joblib.dump(scaler, MODELS_DIR / f"{symbol}_scaler.pkl")
    joblib.dump(meta_model, MODELS_DIR / f"{symbol}_meta.pkl")
    joblib.dump(iso_regressor, MODELS_DIR / f"{symbol}_iso.pkl")
    joblib.dump(meta_scaler, MODELS_DIR / f"{symbol}_meta_scaler.pkl")

def load_model(symbol):
    try:
        xgb_model = joblib.load(MODELS_DIR / f"{symbol}_xgb.pkl")
        lgb_model = joblib.load(MODELS_DIR / f"{symbol}_lgb.pkl")
        lstm_model = tf.keras.models.load_model(str(MODELS_DIR / f"{symbol}_lstm.h5"))
        scaler = joblib.load(MODELS_DIR / f"{symbol}_scaler.pkl")
        meta_model = joblib.load(MODELS_DIR / f"{symbol}_meta.pkl")
        iso_regressor = joblib.load(MODELS_DIR / f"{symbol}_iso.pkl")
        meta_scaler = joblib.load(MODELS_DIR / f"{symbol}_meta_scaler.pkl")
        return {'xgb': xgb_model, 'lgb': lgb_model, 'lstm': lstm_model, 'scaler': scaler,
                'meta': meta_model, 'iso': iso_regressor, 'meta_scaler': meta_scaler}
    except:
        return None

def build_lstm(seq_len, feature_dim):
    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_len, feature_dim), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def add_features(df):
    # Additional feature engineering can be added here
    return df

# --- Training ---

def train_models(df, symbol):
    df = add_features(df)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    X = df[FEATURES]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hyperparameter tuning for XGB
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': trial.suggest_int("max_depth", 3, 10),
            'learning_rate': trial.suggest_float("lr", 0.01, 0.3),
            'n_estimators': trial.suggest_int("n_estimators", 50, 200),
            'subsample': trial.suggest_float("subsample", 0.6, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_scaled, y)
        preds = model.predict_proba(X_scaled)[:, 1]
        loss = np.mean(np.abs(y - preds))
        return loss

    import optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    xgb_model = xgb.XGBClassifier(**best_params).fit(X_scaled, y)
    lgb_model = lgb.LGBMClassifier().fit(X_scaled, y)

    # Sequence for LSTM
    seq_len = 50
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i + seq_len])
        y_seq.append(y.iloc[i + seq_len])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Build & train LSTM
    lstm_model = build_lstm(seq_len, len(FEATURES))
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    lstm_model.fit(X_seq, y_seq, epochs=50, batch_size=32, verbose=0, validation_split=0.2, callbacks=[early_stop])

    # Generate predictions for stacking
    preds_xgb = xgb_model.predict_proba(X_scaled)[:, 1]
    preds_lgb = lgb_model.predict_proba(X_scaled)[:, 1]
    lstm_preds = []
    for i in range(len(X_seq)):
        seq_input = X_seq[i].reshape(1, seq_len, len(FEATURES))
        pred = lstm_model.predict(seq_input, verbose=0)[0][0]
        lstm_preds.append(pred)
    lstm_preds = np.array(lstm_preds)

    # Align predictions
    valid_idx = range(seq_len, len(X_scaled))
    meta_X = np.vstack([preds_xgb[valid_idx], preds_lgb[valid_idx], lstm_preds]).T
    meta_y = y.iloc[valid_idx]

    # Calibration with isotonic regression
    iso_regressor = IsotonicRegression(out_of_bounds='clip')
    meta_preds = np.mean(meta_X, axis=1)
    iso_regressor.fit(meta_preds, meta_y)

    # Meta-model (neural network)
    meta_input = meta_X
    meta_target = meta_y
    meta_scaler = StandardScaler()
    meta_input_scaled = meta_scaler.fit_transform(meta_input)
    meta_nn = Sequential()
    meta_nn.add(Dense(8, activation='relu', input_shape=(meta_input_scaled.shape[1],)))
    meta_nn.add(Dense(4, activation='relu'))
    meta_nn.add(Dense(1, activation='sigmoid'))
    meta_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    meta_nn.fit(meta_input_scaled, meta_target, epochs=20, verbose=0, validation_split=0.2,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])

    # Save models
    save_model(symbol, {'xgb': xgb_model, 'lgb': lgb_model, 'lstm': lstm_model},
               scaler, meta_nn, iso_regressor, meta_scaler)

    # Save for use
    symbol_models[symbol] = {'xgb': xgb_model, 'lgb': lgb_model, 'lstm': lstm_model}
    symbol_scalers[symbol] = scaler
    symbol_meta_models[symbol] = meta_nn
    symbol_meta_scalers[symbol] = meta_scaler
    symbol_iso_regressors[symbol] = iso_regressor

    return {
        'xgb': xgb_model,
        'lgb': lgb_model,
        'lstm': lstm_model,
        'scaler': scaler,
        'meta': meta_nn,
        'iso': iso_regressor,
        'meta_scaler': meta_scaler
    }

# --- Prediction ---

def ensemble_predict(models, df):
    X = df[FEATURES].values[-1].reshape(1, -1)
    scaler = models['scaler']
    X_scaled = scaler.transform(X)

    # Base predictions
    pred_xgb = models['xgb'].predict_proba(X_scaled)[:, 1][0]
    pred_lgb = models['lgb'].predict_proba(X_scaled)[:, 1][0]
    seq_len = 50

    try:
        recent_X = df[FEATURES].values[-seq_len:]
        recent_X_scaled = scaler.transform(recent_X)
        recent_X_seq = recent_X_scaled.reshape((1, seq_len, len(FEATURES)))
        pred_lstm = models['lstm'].predict(recent_X_seq)[0][0]
    except:
        pred_lstm = 0.5

    base_preds = np.array([[pred_xgb, pred_lgb, pred_lstm]])
    # Meta-model prediction
    meta_input = base_preds
    meta_input_scaled = models['meta_scaler'].transform(meta_input)
    conf = models['meta'].predict(meta_input_scaled)[0]
    # Calibrate confidence
    conf = models['iso'].predict([conf])[0]
    decision = 'call' if conf >= 0.5 else 'put'
    return decision, conf

# --- Risk & position sizing ---

def check_risk():
    total_loss = sum(r['payout'] - r['stake'] for r in recent_trades if r['result'] == 'loss')
    balance = getattr(check_risk, 'account_balance', 1)
    # Add more risk metrics as needed
    if abs(total_loss) / max(1, abs(balance)) > MAX_DAILY_LOSS:
        return False
    return True

def update_performance(symbol, result):
    perf = symbol_perf[symbol]
    if result == 'win':
        perf['wins'] += 1
    elif result == 'loss':
        perf['losses'] += 1
    total = perf['wins'] + perf['losses']
    perf['accuracy'] = perf['wins'] / total if total else 0
    perf['enabled'] = perf['accuracy'] >= 0.65

def get_threshold(symbol):
    perf = symbol_perf[symbol]
    if perf['accuracy'] > 0.75:
        return 0.7
    elif perf['accuracy'] > 0.65:
        return 0.8
    else:
        return 0.85

def compute_stake(confidence, symbol, current_drawdown=0):
    # Dynamic stake based on confidence and drawdown
    base_stake = MIN_STAKE
    stake = base_stake * (confidence + 0.1) * max(0.5, 1 - current_drawdown)
    return max(MIN_STAKE, stake)

# --- Placeholder for trade execution ---

def execute_trade(symbol, decision, stake, confidence):
    print(f"Trade: {decision} {symbol} stake {stake:.2f} conf {confidence:.2f}")

def update_trade_result(symbol, trade_id):
    # Placeholder: replace with real API call
    return 'win' if np.random.rand() > 0.5 else 'loss'

# --- Main trading loop ---

def maybe_retrain_models():
    global last_retrain_time
    current_time = time.time()
    if current_time - last_retrain_time > 86400:  # retrain daily
        for s in SYMBOLS:
            df = get_market_data(s)
            train_models(df, s)
        last_retrain_time = current_time

def trade_loop(symbol):
    while True:
        maybe_retrain_models()
        # Fetch all symbol data
        all_data = {s: get_market_data(s) for s in SYMBOLS}
        df = all_data.get(symbol)
        if df is None or df.empty:
            time.sleep(5)
            continue
        df_feat = add_features(df)

        # Load or train models
        models = load_model(symbol)
        if not models:
            models = train_models(df, symbol)

        # Prediction
        decision, conf = ensemble_predict(models, df_feat)
        threshold = get_threshold(symbol)
        if conf < threshold:
            time.sleep(5)
            continue

        now = time.time()
        if last_trade_time.get(symbol, 0) + TRADE_COOLDOWN > now:
            time.sleep(5)
            continue

        if not check_risk():
            time.sleep(30)
            continue

        # Stake calculation
        current_drawdown = 0  # Placeholder, implement tracking
        stake = compute_stake(conf, symbol, current_drawdown)
        execute_trade(symbol, decision, stake, conf)

        last_trade_time[symbol] = now

        # Simulate or get real outcome
        trade_result = update_trade_result(symbol, None)
        update_performance(symbol, trade_result)

        time.sleep(5)

# --- WebSocket handling & main ---

def run():
    def on_open(ws):
        logger.info("WebSocket opened.")

    def on_close(ws):
        logger.info("WebSocket closed.")

    def on_message(ws, message):
        pass  # TODO: implement message handling for real-time updates

    ws = websocket.WebSocketApp(WS_BASE, on_open=on_open, on_close=on_close, on_message=on_message)
    threading.Thread(target=ws.run_forever, daemon=True).start()

    # Launch trading threads
    for s in SYMBOLS:
        threading.Thread(target=trade_loop, args=(s,), daemon=True).start()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    run()
```

# ----------------------------

if __name__ == "__main__":
    print("🚀 Infinity Deriv Bot starting inside Cloud Run...")
    run()