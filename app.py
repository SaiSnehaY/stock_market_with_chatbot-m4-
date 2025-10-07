from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
import requests
import os

app = Flask(__name__)
CORS(app)

# Cache for trained models: { symbol: { 'timestamp': ..., 'lr': ..., 'rf': ..., 'svr': ..., 'scaler': ..., 'lstm': ..., 'hist': DataFrame, 'arima_pred': float, 'model_metrics': dict }}
MODEL_CACHE = {}
MODEL_TTL_SECONDS = 60 * 15  # 15 minutes


def is_cache_valid(entry):
    return entry and (time.time() - entry.get('timestamp', 0) < MODEL_TTL_SECONDS)


# --- Machine Learning Models ---

def train_models(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Close']]
    # Prepare data for supervised learning
    df['Prediction'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    X = np.array(df.drop(['Prediction'], axis=1))
    y = np.array(df['Prediction'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_score = lr_model.score(X_test, y_test)
    lr_pred = lr_model.predict(X_test)
    lr_mse = np.mean((y_test - lr_pred) ** 2)
    lr_rmse = np.sqrt(lr_mse)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=120, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_score = rf_model.score(X_test, y_test)
    rf_pred = rf_model.predict(X_test)
    rf_mse = np.mean((y_test - rf_pred) ** 2)
    rf_rmse = np.sqrt(rf_mse)
    
    # SVR
    svr_model = SVR(kernel='rbf', C=100, gamma='auto')
    svr_model.fit(X_train, y_train)
    svr_score = svr_model.score(X_test, y_test)
    svr_pred = svr_model.predict(X_test)
    svr_mse = np.mean((y_test - svr_pred) ** 2)
    svr_rmse = np.sqrt(svr_mse)
    
    # LSTM (optional, may be skipped later if insufficient data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    lstm_model = None
    lstm_metrics = {}
    
    try:
        if len(scaled_data) >= 100:
            train_data = scaled_data[0:int(len(scaled_data)*0.8), :]
            x_train_lstm = []
            y_train_lstm = []
            for i in range(30, len(train_data)):
                x_train_lstm.append(train_data[i-30:i, 0])
                y_train_lstm.append(train_data[i, 0])
            x_train_lstm, y_train_lstm = np.array(x_train_lstm), np.array(y_train_lstm)
            x_train_lstm = np.reshape(x_train_lstm, (x_train_lstm.shape[0], x_train_lstm.shape[1], 1))
            lstm_model = Sequential()
            lstm_model.add(LSTM(32, return_sequences=False, input_shape=(x_train_lstm.shape[1], 1)))
            lstm_model.add(Dropout(0.1))
            lstm_model.add(Dense(16))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            lstm_model.fit(x_train_lstm, y_train_lstm, batch_size=8, epochs=1, verbose=0)
            
            # LSTM metrics
            lstm_pred = lstm_model.predict(x_train_lstm, verbose=0)
            lstm_pred_original = scaler.inverse_transform(lstm_pred)
            lstm_metrics = {
                'mse': float(np.mean((y_train_lstm - lstm_pred_original.flatten()) ** 2)),
                'rmse': float(np.sqrt(np.mean((y_train_lstm - lstm_pred_original.flatten()) ** 2))),
                'r2': float(1 - np.sum((y_train_lstm - lstm_pred_original.flatten()) ** 2) / np.sum((y_train_lstm - np.mean(y_train_lstm)) ** 2))
            }
    except Exception:
        lstm_model = None
        lstm_metrics = {}
    
    # Store metrics for each model
    model_metrics = {
        'lr': {
            'mse': float(lr_mse),
            'rmse': float(lr_rmse),
            'r2': float(lr_score)
        },
        'rf': {
            'mse': float(rf_mse),
            'rmse': float(rf_rmse),
            'r2': float(rf_score)
        },
        'svr': {
            'mse': float(svr_mse),
            'rmse': float(svr_rmse),
            'r2': float(svr_score)
        },
        'lstm': lstm_metrics
    }
    
    return lr_model, rf_model, svr_model, lstm_model, scaler, model_metrics


def arima_next_prediction(close_series: pd.Series) -> float | None:
    try:
        values = pd.Series(close_series).astype(float).dropna()
        if len(values) < 30:
            return None
        model = ARIMA(values, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return float(forecast.iloc[0])
    except Exception:
        return None


@app.route('/stock_info', methods=['POST'])
def stock_info():
    data = request.get_json(silent=True) or {}
    stock_symbol = data.get('stock_symbol') or data.get('stock')
    if not stock_symbol:
        return jsonify({"error": "Missing stock_symbol"}), 400
    try:
        ticker = yf.Ticker(stock_symbol)
        hist = ticker.history(period="3mo").reset_index()
        if hist.empty:
            return jsonify({"error": "No data for symbol"}), 404
        name = None
        try:
            name = ticker.get_info().get('longName')
        except Exception:
            name = None
        latest = hist.iloc[-1]
        prev_close = None
        change_pct = None
        if len(hist) >= 2:
            prev_close = float(hist.iloc[-2]['Close'])
            try:
                change_pct = round((float(latest['Close']) - prev_close) / prev_close * 100.0, 2) if prev_close else None
            except Exception:
                change_pct = None
        info = {
            "symbol": stock_symbol,
            "name": name or stock_symbol,
            "latest_close": round(float(latest['Close']), 2),
            "latest_date": pd.to_datetime(latest['Date']).strftime('%Y-%m-%d'),
            "day_high": round(float(latest.get('High', latest['Close'])), 2),
            "day_low": round(float(latest.get('Low', latest['Close'])), 2),
            "volume": int(latest.get('Volume', 0)),
            "prev_close": round(prev_close, 2) if prev_close is not None else None,
            "change_pct": change_pct
        }
        chart_data = hist[['Date','Close']].tail(30)
        chart_data['Date'] = pd.to_datetime(chart_data['Date']).dt.strftime('%Y-%m-%d')
        return jsonify({
            "info": info,
            "historical_prices": chart_data.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    stock_symbol = data.get('stock_symbol') or data.get('stock')
    prediction_date_str = data.get('prediction_date') or data.get('date')
    if not stock_symbol or not prediction_date_str:
        return jsonify({"error": "Missing required fields: stock_symbol and prediction_date"}), 400
    try:
        cache_entry = MODEL_CACHE.get(stock_symbol)
        if not is_cache_valid(cache_entry):
            ticker = yf.Ticker(stock_symbol)
            hist = ticker.history(period="3mo")
            if hist.empty:
                return jsonify({"error": "Could not fetch data for the given stock symbol."}), 404
            hist = hist.reset_index()
            hist['Date'] = pd.to_datetime(hist['Date'])
            lr, rf, svr, lstm, scaler, model_metrics = train_models(hist.copy())
            arima_pred = arima_next_prediction(hist['Close'])
            MODEL_CACHE[stock_symbol] = {
                "timestamp": time.time(),
                "lr": lr,
                "rf": rf,
                "svr": svr,
                "lstm": lstm,
                "scaler": scaler,
                "hist": hist,
                "arima_pred": arima_pred,
                "model_metrics": model_metrics
            }
        else:
            lr = cache_entry['lr']
            rf = cache_entry['rf']
            svr = cache_entry['svr']
            lstm = cache_entry['lstm']
            scaler = cache_entry['scaler']
            hist = cache_entry['hist']
            arima_pred = cache_entry.get('arima_pred')
            model_metrics = cache_entry.get('model_metrics', {})

        last_close_price = hist['Close'].iloc[-1]
        input_for_prediction = np.array([[last_close_price]])
        lr_prediction = lr.predict(input_for_prediction)[0]
        rf_prediction = rf.predict(input_for_prediction)[0]
        svr_prediction = svr.predict(input_for_prediction)[0]

        lstm_prediction = None
        try:
            if lstm is not None and len(hist) >= 60:
                last_30_days = hist['Close'].tail(30).values
                last_30_scaled = scaler.transform(last_30_days.reshape(-1, 1))
                X_test_lstm = np.array([last_30_scaled])
                X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
                lp_scaled = lstm.predict(X_test_lstm, verbose=0)
                lstm_prediction = scaler.inverse_transform(lp_scaled)[0][0]
        except Exception:
            lstm_prediction = None

        # Choose model predictions for chart (use all available)
        model_to_value = {
            "lr": lr_prediction,
            "rf": rf_prediction,
            "svr": svr_prediction,
            "lstm": lstm_prediction,
            "arima": arima_pred
        }
        predictions_for_chart = {k: float(v) for k, v in model_to_value.items() if v is not None}

        # Minimal ranking visualization (bar chart)
        ranking_sorted = sorted(predictions_for_chart.items(), key=lambda kv: kv[1], reverse=True)
        highest_model = ranking_sorted[0][0].upper() if ranking_sorted else None
        lowest_model = ranking_sorted[-1][0].upper() if ranking_sorted else None
        fig_rank, ax_rank = plt.subplots(figsize=(5.5, 2.6))
        fig_rank.patch.set_facecolor('#eef3fb')
        ax_rank.set_facecolor('#f8fbff')
        for spine in ax_rank.spines.values():
            spine.set_color('#cfd9ea'); spine.set_linewidth(0.8)
        ax_rank.grid(True, axis='y', color='#d9e2f3', linewidth=0.7, alpha=0.7)
        labels = [k.upper() for k,_ in ranking_sorted]
        values = [v for _,v in ranking_sorted]
        bar_colors = ['#7c3aed','#f59e0b','#10b981','#ef4444','#a855f7'][:len(values)]
        ax_rank.bar(labels, values, color=bar_colors, alpha=0.9)
        for i, v in enumerate(values):
            ax_rank.text(i, v, f" {v:.2f}", va='bottom', fontsize=8)
        ax_rank.set_ylabel('Predicted Price')
        ax_rank.set_title('Model predicted prices (High → Low)', fontsize=10)
        fig_rank.tight_layout()
        buf2 = io.BytesIO(); fig_rank.savefig(buf2, format='png', dpi=120); plt.close(fig_rank); buf2.seek(0)
        img_rank_b64 = base64.b64encode(buf2.read()).decode('utf-8'); img_rank_url = f"data:image/png;base64,{img_rank_b64}"

        # Render chart based on requested chart_type (no date labels)
        recent = hist.tail(60).copy()
        fig, ax = plt.subplots(figsize=(6, 3))
        # Professional soft backgrounds and grid styling
        fig.patch.set_facecolor('#eef3fb')
        ax.set_facecolor('#f8fbff')
        for spine in ax.spines.values():
            spine.set_color('#cfd9ea')
            spine.set_linewidth(0.8)
        ax.grid(True, axis='y', color='#d9e2f3', linewidth=0.7, alpha=0.7)
        ct = (data.get('chart_type') or 'histogram').lower().strip()
        # Color mapping for models
        model_colors = {
            'lr': '#7c3aed',
            'rf': '#f59e0b',
            'svr': '#10b981',
            'lstm': '#ef4444',
            'arima': '#a855f7'
        }
        if ct in ('histogram', 'bar'):
            # Histogram of recent closes with vertical lines for each model prediction
            ax.hist(recent['Close'].values.astype(float), bins=12, color='#a78bfa', alpha=0.9)
            ylim = ax.get_ylim()
            for i, (m, val) in enumerate(predictions_for_chart.items()):
                c = model_colors.get(m, '#333333')
                ax.axvline(val, color=c, linewidth=2)
                ax.text(val, ylim[1]*0.92, f"{m.upper()} price: {val:.2f}", rotation=90, fontsize=7, color=c, ha='left', va='top')
            ax.set_xlabel('Price'); ax.set_ylabel('Frequency')
            ax.set_title('Distribution of recent closes with model predictions', fontsize=10)
        elif ct == 'line':
            # Line chart of recent closes over index (no dates) and predicted points jittered to the right
            x = np.arange(len(recent))
            ax.plot(x, recent['Close'].values.astype(float), color='#8b5cf6', linewidth=2)
            # place predictions at future x positions with slight offsets
            x_base = len(x)
            offsets = np.linspace(0.2, 1.0, num=max(1, len(predictions_for_chart)))
            for (m, val), dx in zip(predictions_for_chart.items(), offsets):
                c = model_colors.get(m, '#333333')
                xp = x_base + dx
                ax.scatter([xp], [val], color=c, s=36, zorder=3)
                ax.text(xp, val, f" {m.upper()} price: {val:.2f}", fontsize=7, color=c, ha='left', va='bottom')
            ax.set_xticks([])
            ax.set_ylabel('Price')
            ax.set_title('Recent closes with model predicted prices', fontsize=10)
        elif ct == 'area':
            # Area chart (filled line) without dates
            x = np.arange(len(recent))
            y = recent['Close'].values.astype(float)
            ax.plot(x, y, color='#8b5cf6', linewidth=1.8)
            ax.fill_between(x, y, color='#d8b4fe', alpha=0.5)
            x_base = len(x)
            offsets = np.linspace(0.2, 1.0, num=max(1, len(predictions_for_chart)))
            for (m, val), dx in zip(predictions_for_chart.items(), offsets):
                c = model_colors.get(m, '#333333')
                xp = x_base + dx
                ax.scatter([xp], [val], color=c, s=38, zorder=3)
                ax.text(xp, val, f" {m.upper()} price: {val:.2f}", fontsize=7, color=c, ha='left', va='bottom')
            ax.set_xticks([]); ax.set_ylabel('Price')
            ax.set_title('Area chart of recent closes with predictions', fontsize=10)
        elif ct == 'box':
            # Box plot of recent closes with overlays
            bp = ax.boxplot(recent['Close'].values.astype(float), vert=True, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#e6f0ff')
                patch.set_edgecolor('#9fb7e8')
            for (m, val) in predictions_for_chart.items():
                c = model_colors.get(m, '#333333')
                ax.scatter([1.15], [val], color=c, s=38, zorder=3)
                ax.text(1.18, val, f" {m.upper()} price: {val:.2f}", fontsize=7, color=c, ha='left', va='center')
            ax.set_xticks([]); ax.set_ylabel('Price')
            ax.set_title('Box plot of recent closes with predictions', fontsize=10)
        elif ct == 'violin':
            # Violin plot of recent closes with overlays
            vp = ax.violinplot(recent['Close'].values.astype(float), showmeans=True, showextrema=False, showmedians=True)
            for body in vp['bodies']:
                body.set_facecolor('#e9d5ff'); body.set_edgecolor('#c4b5fd'); body.set_alpha(0.75)
            if 'cmeans' in vp:
                vp['cmeans'].set_color('#8b5cf6')
            for (m, val) in predictions_for_chart.items():
                c = model_colors.get(m, '#333333')
                ax.scatter([1.05], [val], color=c, s=38, zorder=3)
                ax.text(1.08, val, f" {m.upper()} price: {val:.2f}", fontsize=7, color=c, ha='left', va='center')
            ax.set_xticks([]); ax.set_ylabel('Price')
            ax.set_title('Violin plot of recent closes with predictions', fontsize=10)
        else:
            # Fallback: histogram behavior
            ax.hist(recent['Close'].values.astype(float), bins=12, color='#a78bfa', alpha=0.9)
            ylim = ax.get_ylim()
            for i, (m, val) in enumerate(predictions_for_chart.items()):
                c = model_colors.get(m, '#333333')
                ax.axvline(val, color=c, linewidth=2)
                ax.text(val, ylim[1]*0.92, f"{m.upper()} price: {val:.2f}", rotation=90, fontsize=7, color=c, ha='left', va='top')
            ax.set_xlabel('Price'); ax.set_ylabel('Frequency')
            ax.set_title('Distribution of recent closes with model predictions', fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8'); img_url = f"data:image/png;base64,{img_b64}"

        chart_data = hist[['Date', 'Close']].tail(90).to_dict(orient='records')
        for item in chart_data:
            item['Date'] = item['Date'].strftime('%Y-%m-%d')
        
        # Prepare historical data for comparison tool
        historical_data = {
            'dates': [item['Date'] for item in chart_data],
            'prices': [float(item['Close']) for item in chart_data]
        }
        
        # Prepare predictions in a structured format
        predictions = []
        if lr_prediction is not None:
            predictions.append({
                'name': 'Linear Regression',
                'value': round(float(lr_prediction), 2)
            })
        if rf_prediction is not None:
            predictions.append({
                'name': 'Random Forest',
                'value': round(float(rf_prediction), 2)
            })
        if svr_prediction is not None:
            predictions.append({
                'name': 'SVR',
                'value': round(float(svr_prediction), 2)
            })
        if lstm_prediction is not None:
            predictions.append({
                'name': 'LSTM',
                'value': round(float(lstm_prediction), 2)
            })
        if arima_pred is not None:
            predictions.append({
                'name': 'ARIMA',
                'value': round(float(arima_pred), 2)
            })
        
        response = {
            "lr_prediction": round(float(lr_prediction), 2),
            "rf_prediction": round(float(rf_prediction), 2),
            "svr_prediction": round(float(svr_prediction), 2),
            "arima_prediction": round(float(arima_pred), 2) if arima_pred is not None else None,
            "lstm_prediction": round(float(lstm_prediction), 2) if lstm_prediction is not None else None,
            "historical_prices": chart_data,
            "chart_image": img_url,
            "ranking_image": img_rank_url,
            "highest_model": highest_model,
            "lowest_model": lowest_model,
            "predictions": predictions,
            "historical_data": historical_data,
            "model_metrics": model_metrics
        }
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/ollama_chat', methods=['POST'])
def ollama_chat():
    return jsonify({'error': 'Chatbot has been removed'}), 404


@app.route('/ollama_health', methods=['GET'])
def ollama_health():
    return jsonify({'ok': False, 'error': 'Chatbot has been removed'}), 404


@app.route('/chatbot_predict', methods=['POST'])
def chatbot_predict():
    """
    Lightweight chatbot-style prediction: parse free-form 'message' to extract
    stock symbol, date, and chart type, then run the same prediction pipeline
    and return a concise text plus chart/ranking images.
    """
    return jsonify({'error': 'Chatbot has been removed'}), 404
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    # Defaults
    from datetime import datetime
    today_str = datetime.now().strftime('%Y-%m-%d')
    stock_symbol = None
    prediction_date_str = today_str
    chart_type = 'bar'

    # Very simple parsing from message
    msg_lower = message.lower()
    # Chart type detection
    if 'line' in msg_lower:
        chart_type = 'line'
    elif 'area' in msg_lower:
        chart_type = 'area'
    elif 'box' in msg_lower:
        chart_type = 'box'
    elif 'violin' in msg_lower:
        chart_type = 'violin'
    else:
        chart_type = 'bar'

    # Date detection: look for YYYY-MM-DD
    import re
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", message)
    if m:
        prediction_date_str = m.group(1)

    # Stock detection: take first all-caps token of 1-5 chars or common tickers in message
    tokens = re.findall(r"\b[A-Z]{1,5}\b", message)
    if tokens:
        stock_symbol = tokens[0]
    # Fallback popular list if mentioned in any case
    popular = ['AAPL','MSFT','GOOGL','AMZN','TSLA','FB','NVDA','JPM','V','PG']
    if not stock_symbol:
        for t in popular:
            if t.lower() in msg_lower:
                stock_symbol = t; break
    if not stock_symbol:
        stock_symbol = 'AAPL'

    # Reuse the existing prediction pipeline (duplicate minimal logic)
    try:
        cache_entry = MODEL_CACHE.get(stock_symbol)
        if not is_cache_valid(cache_entry):
            ticker = yf.Ticker(stock_symbol)
            hist = ticker.history(period="3mo")
            if hist.empty:
                return jsonify({"error": "Could not fetch data for the given stock symbol."}), 404
            hist = hist.reset_index()
            hist['Date'] = pd.to_datetime(hist['Date'])
            lr, rf, svr, lstm, scaler, model_metrics = train_models(hist.copy())
            arima_pred = arima_next_prediction(hist['Close'])
            MODEL_CACHE[stock_symbol] = {
                "timestamp": time.time(),
                "lr": lr,
                "rf": rf,
                "svr": svr,
                "lstm": lstm,
                "scaler": scaler,
                "hist": hist,
                "arima_pred": arima_pred,
                "model_metrics": model_metrics
            }
        else:
            lr = cache_entry['lr']
            rf = cache_entry['rf']
            svr = cache_entry['svr']
            lstm = cache_entry['lstm']
            scaler = cache_entry['scaler']
            hist = cache_entry['hist']
            arima_pred = cache_entry.get('arima_pred')
            model_metrics = cache_entry.get('model_metrics', {})

        last_close_price = hist['Close'].iloc[-1]
        input_for_prediction = np.array([[last_close_price]])
        lr_prediction = lr.predict(input_for_prediction)[0]
        rf_prediction = rf.predict(input_for_prediction)[0]
        svr_prediction = svr.predict(input_for_prediction)[0]

        lstm_prediction = None
        try:
            if lstm is not None and len(hist) >= 60:
                last_30_days = hist['Close'].tail(30).values
                last_30_scaled = scaler.transform(last_30_days.reshape(-1, 1))
                X_test_lstm = np.array([last_30_scaled])
                X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
                lp_scaled = lstm.predict(X_test_lstm, verbose=0)
                lstm_prediction = scaler.inverse_transform(lp_scaled)[0][0]
        except Exception:
            lstm_prediction = None

        model_to_value = {
            "lr": lr_prediction,
            "rf": rf_prediction,
            "svr": svr_prediction,
            "lstm": lstm_prediction,
            "arima": arima_pred
        }
        predictions_for_chart = {k: float(v) for k, v in model_to_value.items() if v is not None}

        # Ranking image
        ranking_sorted = sorted(predictions_for_chart.items(), key=lambda kv: kv[1], reverse=True)
        highest_model = ranking_sorted[0][0].upper() if ranking_sorted else None
        lowest_model = ranking_sorted[-1][0].upper() if ranking_sorted else None
        fig_rank, ax_rank = plt.subplots(figsize=(5.5, 2.6))
        fig_rank.patch.set_facecolor('#eef3fb')
        ax_rank.set_facecolor('#f8fbff')
        for spine in ax_rank.spines.values():
            spine.set_color('#cfd9ea'); spine.set_linewidth(0.8)
        ax_rank.grid(True, axis='y', color='#d9e2f3', linewidth=0.7, alpha=0.7)
        labels = [k.upper() for k,_ in ranking_sorted]
        values = [v for _,v in ranking_sorted]
        bar_colors = ['#7c3aed','#f59e0b','#10b981','#ef4444','#a855f7'][:len(values)]
        ax_rank.bar(labels, values, color=bar_colors, alpha=0.9)
        for i, v in enumerate(values):
            ax_rank.text(i, v, f" {v:.2f}", va='bottom', fontsize=8)
        ax_rank.set_ylabel('Predicted Price')
        ax_rank.set_title('Model predicted prices (High → Low)', fontsize=10)
        fig_rank.tight_layout()
        buf2 = io.BytesIO(); fig_rank.savefig(buf2, format='png', dpi=120); plt.close(fig_rank); buf2.seek(0)
        img_rank_b64 = base64.b64encode(buf2.read()).decode('utf-8'); img_rank_url = f"data:image/png;base64,{img_rank_b64}"

        # Chart image
        recent = hist.tail(60).copy()
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#eef3fb'); ax.set_facecolor('#f8fbff')
        for spine in ax.spines.values(): spine.set_color('#cfd9ea'); spine.set_linewidth(0.8)
        ax.grid(True, axis='y', color='#d9e2f3', linewidth=0.7, alpha=0.7)
        ct = (chart_type or 'bar').lower().strip()
        model_colors = {'lr': '#7c3aed','rf': '#f59e0b','svr': '#10b981','lstm': '#ef4444','arima': '#a855f7'}
        if ct in ('histogram','bar'):
            ax.hist(recent['Close'].values.astype(float), bins=12, color='#a78bfa', alpha=0.9)
            ylim = ax.get_ylim()
            for (m, val) in predictions_for_chart.items():
                c = model_colors.get(m, '#333'); ax.axvline(val, color=c, linewidth=2)
                ax.text(val, ylim[1]*0.92, f"{m.upper()} price: {val:.2f}", rotation=90, fontsize=7, color=c, ha='left', va='top')
            ax.set_xlabel('Price'); ax.set_ylabel('Frequency'); ax.set_title('Distribution of recent closes with model predictions', fontsize=10)
        elif ct == 'line':
            x = np.arange(len(recent)); ax.plot(x, recent['Close'].values.astype(float), color='#8b5cf6', linewidth=2)
            x_base = len(x); offsets = np.linspace(0.2, 1.0, num=max(1, len(predictions_for_chart)))
            for (m, val), dx in zip(predictions_for_chart.items(), offsets):
                c = model_colors.get(m, '#333'); xp = x_base + dx
                ax.scatter([xp], [val], color=c, s=36, zorder=3); ax.text(xp, val, f" {m.upper()} price: {val:.2f}", fontsize=7, color=c, ha='left', va='bottom')
            ax.set_xticks([]); ax.set_ylabel('Price'); ax.set_title('Recent closes with model predicted prices', fontsize=10)
        elif ct == 'area':
            x = np.arange(len(recent)); y = recent['Close'].values.astype(float)
            ax.plot(x, y, color='#8b5cf6', linewidth=1.8); ax.fill_between(x, y, color='#d8b4fe', alpha=0.5)
            x_base = len(x); offsets = np.linspace(0.2, 1.0, num=max(1, len(predictions_for_chart)))
            for (m, val), dx in zip(predictions_for_chart.items(), offsets):
                c = model_colors.get(m, '#333'); xp = x_base + dx
                ax.scatter([xp], [val], color=c, s=38, zorder=3); ax.text(xp, val, f" {m.upper()} price: {val:.2f}", fontsize=7, color=c, ha='left', va='bottom')
            ax.set_xticks([]); ax.set_ylabel('Price'); ax.set_title('Area chart of recent closes with predictions', fontsize=10)
        elif ct == 'box':
            bp = ax.boxplot(recent['Close'].values.astype(float), vert=True, patch_artist=True)
            for patch in bp['boxes']: patch.set_facecolor('#e6f0ff'); patch.set_edgecolor('#9fb7e8')
            for (m, val) in predictions_for_chart.items():
                c = model_colors.get(m, '#333'); ax.scatter([1.15], [val], color=c, s=38, zorder=3); ax.text(1.18, val, f" {m.upper()} price: {val:.2f}", fontsize=7, color=c, ha='left', va='center')
            ax.set_xticks([]); ax.set_ylabel('Price'); ax.set_title('Box plot of recent closes with predictions', fontsize=10)
        elif ct == 'violin':
            vp = ax.violinplot(recent['Close'].values.astype(float), showmeans=True, showextrema=False, showmedians=True)
            for body in vp['bodies']: body.set_facecolor('#e9d5ff'); body.set_edgecolor('#c4b5fd'); body.set_alpha(0.75)
            if 'cmeans' in vp: vp['cmeans'].set_color('#8b5cf6')
            for (m, val) in predictions_for_chart.items():
                c = model_colors.get(m, '#333'); ax.scatter([1.05], [val], color=c, s=38, zorder=3); ax.text(1.08, val, f" {m.upper()} price: {val:.2f}", fontsize=7, color=c, ha='left', va='center')
            ax.set_xticks([]); ax.set_ylabel('Price'); ax.set_title('Violin plot of recent closes with predictions', fontsize=10)
        else:
            ax.hist(recent['Close'].values.astype(float), bins=12, color='#a78bfa', alpha=0.9)
            ylim = ax.get_ylim()
            for (m, val) in predictions_for_chart.items():
                c = model_colors.get(m, '#333'); ax.axvline(val, color=c, linewidth=2)
                ax.text(val, ylim[1]*0.92, f"{m.upper()} price: {val:.2f}", rotation=90, fontsize=7, color=c, ha='left', va='top')
            ax.set_xlabel('Price'); ax.set_ylabel('Frequency'); ax.set_title('Distribution of recent closes with model predictions', fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8'); img_url = f"data:image/png;base64,{img_b64}"

        # Build predictions list for UI
        predictions_list = []
        if lr_prediction is not None:
            predictions_list.append({'name': 'Linear Regression', 'value': round(float(lr_prediction), 2)})
        if rf_prediction is not None:
            predictions_list.append({'name': 'Random Forest', 'value': round(float(rf_prediction), 2)})
        if svr_prediction is not None:
            predictions_list.append({'name': 'SVR', 'value': round(float(svr_prediction), 2)})
        if lstm_prediction is not None:
            predictions_list.append({'name': 'LSTM', 'value': round(float(lstm_prediction), 2)})
        if arima_pred is not None:
            predictions_list.append({'name': 'ARIMA', 'value': round(float(arima_pred), 2)})

        # Basic stock info
        latest = hist.iloc[-1]
        info = {
            'symbol': stock_symbol,
            'latest_close': round(float(latest['Close']), 2),
            'latest_date': pd.to_datetime(latest['Date']).strftime('%Y-%m-%d')
        }

        # Build simple text reply
        parts = [
            f"Stock: {stock_symbol}",
            f"Date: {prediction_date_str}",
        ]
        for p in predictions_list:
            parts.append(f"{p['name']}: {p['value']:.2f}")
        if highest_model:
            parts.append(f"Top: {highest_model}")
        reply = " | ".join(parts)

        return jsonify({
            'content': reply,
            'stock_symbol': stock_symbol,
            'prediction_date': prediction_date_str,
            'chart_type': ct,
            'chart_image': img_url,
            'ranking_image': img_rank_url,
            'highest_model': highest_model,
            'lowest_model': lowest_model,
            'predictions': predictions_list,
            'info': info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)