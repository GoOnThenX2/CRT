import os
import json
import requests
import pandas as pd
import numpy as np
import mplfinance as mpf
import yfinance as yf

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ALERTS_FILE = "sent_alerts.json"

TICKER_MAPPING = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDCAD": "USDCAD=X", 
    "AUDUSD": "AUDUSD=X", "NZDUSD": "NZDUSD=X", "USDJPY": "USDJPY=X", 
    "USDCHF": "USDCHF=X", "JP225": "^N225", "US30": "^DJI", 
    "NAS100": "^NDX", "SPX500": "^GSPC", "GER40": "^GDAXI", 
    "AUS200": "^AXJO", "FRA40": "^FCHI", "XAGUSD": "SI=F", 
    "XAUUSD": "GC=F"
}

TIMEFRAMES = {
    "Monthly": {"interval": "1mo", "period": "10y"},
    "Weekly": {"interval": "1wk", "period": "5y"},
    "Daily": {"interval": "1d", "period": "1y"}
}

def load_sent_alerts():
    if os.path.exists(ALERTS_FILE):
        try:
            with open(ALERTS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_sent_alerts(alerts):
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f, indent=4)

def fetch_yahoo_data(ticker_symbol, interval, period):
    try:
        df = yf.download(ticker_symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"Error fetching {ticker_symbol}: {e}")
        return None

def send_telegram_alert(photo_path, caption):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing in environment variables.")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
        files = {'photo': photo}
        response = requests.post(url, data=payload, files=files)
        if not response.ok:
            print(f"Failed to send Telegram alert: {response.text}")

def get_weekly_trend(df_weekly):
    if df_weekly is None or len(df_weekly) < 200:
        return "UNKNOWN"
    
    df_weekly = df_weekly.copy()
    df_weekly['EMA200'] = df_weekly['Close'].ewm(span=200, adjust=False).mean()
    current_close = df_weekly.iloc[-1]['Close']
    current_ema = df_weekly.iloc[-1]['EMA200']
    
    recent_hh = df_weekly.iloc[-5]['High'] < df_weekly.iloc[-1]['High']
    recent_hl = df_weekly.iloc[-5]['Low'] < df_weekly.iloc[-1]['Low']
    
    is_uptrend = (current_close > current_ema) and (recent_hh or recent_hl)
    is_downtrend = (current_close < current_ema) and not (recent_hh or recent_hl)
    
    if is_uptrend:
        return "Uptrend"
    elif is_downtrend:
        return "Downtrend"
    return "Consolidation"

def scan_key_levels(df):
    key_levels = []
    if df is None or len(df) < 15:
        return key_levels
    
    for i in range(2, len(df) - 1):
        if df.iloc[i-1]['Low'] > df.iloc[i+1]['High']:
            fvg_top = df.iloc[i-1]['Low']
            fvg_bottom = df.iloc[i+1]['High']
            mitigated = any(df.iloc[j]['Low'] <= fvg_bottom for j in range(i+2, len(df)))
            if not mitigated:
                key_levels.append({"type": "FVG", "top": fvg_top, "bottom": fvg_bottom})
        elif df.iloc[i-1]['High'] < df.iloc[i+1]['Low']:
            fvg_top = df.iloc[i+1]['High']
            fvg_bottom = df.iloc[i-1]['Low']
            mitigated = any(df.iloc[j]['High'] >= fvg_top for j in range(i+2, len(df)))
            if not mitigated:
                key_levels.append({"type": "FVG", "top": fvg_top, "bottom": fvg_bottom})
                
        if df.iloc[i]['Close'] < df.iloc[i]['Open'] and df.iloc[i+1]['Close'] > df.iloc[i+1]['Open'] * 1.002:
            ob_top = df.iloc[i]['High']
            ob_bottom = df.iloc[i]['Low']
            mitigated = any(df.iloc[j]['Low'] <= ob_bottom for j in range(i+2, len(df)))
            if not mitigated:
                key_levels.append({"type": "OrderBlock", "top": ob_top, "bottom": ob_bottom})
        elif df.iloc[i]['Close'] > df.iloc[i]['Open'] and df.iloc[i+1]['Close'] < df.iloc[i+1]['Open'] * 1.002:
            ob_top = df.iloc[i]['High']
            ob_bottom = df.iloc[i]['Low']
            mitigated = any(df.iloc[j]['High'] >= ob_top for j in range(i+2, len(df)))
            if not mitigated:
                key_levels.append({"type": "OrderBlock", "top": ob_top, "bottom": ob_bottom})

        if i > 5 and df.iloc[i]['High'] > df.iloc[i-1]['High'] and df.iloc[i]['Close'] < df.iloc[i-1]['Low']:
            bb_top = df.iloc[i]['High']
            bb_bottom = df.iloc[i]['Low']
            key_levels.append({"type": "BreakerBlock", "top": bb_top, "bottom": bb_bottom})

        bar_open = df.iloc[i]['Open']
        bar_close = df.iloc[i]['Close']
        oc_top = max(bar_open, bar_close)
        oc_bottom = min(bar_open, bar_close)
        oc_mitigated = any(df.iloc[j]['Low'] <= oc_bottom and df.iloc[j]['High'] >= oc_top for j in range(i+1, len(df)))
        if not oc_mitigated:
            key_levels.append({"type": "OpenCloseKL", "top": oc_top, "bottom": oc_bottom})
                
    return key_levels

def check_kl_confluence(price_point, key_levels):
    for kl in key_levels:
        if kl['bottom'] <= price_point <= kl['top']:
            return True, f"Tapped {kl['type']} ({kl['bottom']:.5f} - {kl['top']:.5f})"
    return False, "No Key Level Intersected"

def generate_crt_chart(symbol, timeframe_name, df, ref_high, ref_low, setup_type):
    plot_df = df.tail(40).copy()
    
    tv_style = mpf.make_mpf_style(
        base_mpf_style='charles',
        marketcolors=mpf.make_marketcolors(
            up='white', down='black',
            edge={'up':'black', 'down':'black'},
            wick={'up':'black', 'down':'black'},
            volume='in'
        ),
        facecolor='#D1D4DC',
        figcolor='#D1D4DC',
        gridcolor='#C5C8CE',
        gridstyle='--',
        y_on_right=True
    )
    
    hline_props = dict(hlines=[ref_high, ref_low], colors=['#1E1B4B', '#1E1B4B'], linestyle='-', linewidths=1.2)
    
    filename = f"{symbol}_{timeframe_name}_CRT.png"
    mpf.plot(
        plot_df,
        type='candle',
        style=tv_style,
        hlines=hline_props,
        title=f"\n{symbol} - {timeframe_name} | {setup_type}",
        savefig=dict(fname=filename, dpi=300, bbox_inches='tight')
    )
    return filename

def evaluate_crt(symbol, timeframe_name, df, df_weekly):
    if df is None or len(df) < 5:
        return None
        
    ref_high = df.iloc[-2]['High']
    ref_low = df.iloc[-2]['Low']
    
    curr_low = df.iloc[-1]['Low']
    curr_high = df.iloc[-1]['High']
    curr_close = df.iloc[-1]['Close']
    
    prev_high = df.iloc[-2]['High']
    prev_low = df.iloc[-2]['Low']
    
    weekly_trend = get_weekly_trend(df_weekly)
    all_kls = scan_key_levels(df)
    
    setup_model = None
    setup_type_str = None
    
    is_inside_bar = (prev_high <= df.iloc[-3]['High']) and (prev_low >= df.iloc[-3]['Low'])
    if is_inside_bar and curr_low < prev_low and curr_close > prev_low:
        setup_model = "Inside Bar CRT"
        setup_type_str = "Bullish Inside Bar Sweep"
        sweep_extreme = curr_low
    elif is_inside_bar and curr_high > prev_high and curr_close < prev_high:
        setup_model = "Inside Bar CRT"
        setup_type_str = "Bearish Inside Bar Sweep"
        sweep_extreme = curr_high
    elif curr_low < ref_low and curr_close > ref_low and (curr_close - curr_low) > (curr_high - curr_close):
        setup_model = "2-Candle CRT"
        setup_type_str = "Bullish 2-Candle Rejection"
        sweep_extreme = curr_low
    elif curr_high > ref_high and curr_close < ref_high and (curr_high - curr_close) > (curr_close - curr_low):
        setup_model = "2-Candle CRT"
        setup_type_str = "Bearish 2-Candle Rejection"
        sweep_extreme = curr_high
    elif curr_low < df['Low'].iloc[-6:-1].min() and curr_close > df['Low'].iloc[-6:-1].min():
        setup_model = "Multiple-Candle CRT"
        setup_type_str = "Bullish Multi-Bar Sweep"
        sweep_extreme = curr_low
        ref_low = df['Low'].iloc[-6:-1].min()
    elif curr_high > df['High'].iloc[-6:-1].max() and curr_close < df['High'].iloc[-6:-1].max():
        setup_model = "Multiple-Candle CRT"
        setup_type_str = "Bearish Multi-Bar Sweep"
        sweep_extreme = curr_high
        ref_high = df['High'].iloc[-6:-1].max()
    elif curr_low < ref_low and curr_close > ref_low:
        setup_model = "Classic 3-Candle CRT"
        setup_type_str = "Bullish Classic Sweep"
        sweep_extreme = curr_low
    elif curr_high > ref_high and curr_close < ref_high:
        setup_model = "Classic 3-Candle CRT"
        setup_type_str = "Bearish Classic Sweep"
        sweep_extreme = curr_high

    if not setup_model:
        return None

    is_bullish = "Bullish" in setup_type_str
    is_aligned = (weekly_trend == "Uptrend" if is_bullish else weekly_trend == "Downtrend")
    
    has_kl, kl_note = check_kl_confluence(sweep_extreme, all_kls)
    classification = "High Probability CRT (Trend-Following)" if is_aligned else "Reversal CRT"
    classification += f" + {kl_note}"

    setup = {
        "symbol": symbol,
        "timeframe": timeframe_name,
        "model": setup_model,
        "type": setup_type_str,
        "classification": classification,
        "rc_high": ref_high,
        "rc_low": ref_low,
        "sweep_extreme": sweep_extreme,
        "weekly_trend": weekly_trend,
        "kl_note": kl_note
    }
    
    chart_path = generate_crt_chart(symbol, timeframe_name, df, ref_high, ref_low, setup_type_str)
    setup['chart_path'] = chart_path
    
    return setup

def run_bot_loop():
    print("Starting CRT Yahoo Finance Scanner...")
    sent_alerts = load_sent_alerts()
    new_alerts_found = False
    
    for symbol, ticker in TICKER_MAPPING.items():
        print(f"Scanning {symbol} ({ticker})...")
        df_weekly = fetch_yahoo_data(ticker, TIMEFRAMES["Weekly"]["interval"], TIMEFRAMES["Weekly"]["period"])
        
        if df_weekly is None:
            continue
        
        for tf_name, tf_config in TIMEFRAMES.items():
            df_tf = fetch_yahoo_data(ticker, tf_config["interval"], tf_config["period"])
            setup = evaluate_crt(symbol, tf_name, df_tf, df_weekly)
            
            if setup:
                alert_id = f"{symbol}_{tf_name}_{setup['model']}_{setup['type']}_{setup['rc_high']}_{setup['rc_low']}"
                
                if alert_id in sent_alerts:
                    print(f"Skipping already sent alert for {symbol} [{tf_name}] ({setup['model']})")
                    continue
                
                caption = (
                    f"🚨 *CRT ALERT: {setup['classification']}*\n\n"
                    f"• *Asset:* `{setup['symbol']}`\n"
                    f"• *Timeframe:* {setup['timeframe']}\n"
                    f"• *CRT Model:* `{setup['model']}`\n"
                    f"• *Setup Variant:* {setup['type']}\n"
                    f"• *Weekly Trend:* {setup['weekly_trend']}\n"
                    f"• *Key Level (KL):* {setup['kl_note']}\n"
                    f"• *CRT High:* `{setup['rc_high']:.5f}`\n"
                    f"• *CRT Low:* `{setup['rc_low']:.5f}`\n"
                    f"• *Sweep Extreme:* `{setup['sweep_extreme']:.5f}`"
                )
                send_telegram_alert(setup['chart_path'], caption)
                print(f"Alert sent for {setup['symbol']} [{setup['timeframe']}] - {setup['model']}")
                
                sent_alerts.append(alert_id)
                new_alerts_found = True

    if new_alerts_found:
        save_sent_alerts(sent_alerts)

if __name__ == "__main__":
    run_bot_loop()
