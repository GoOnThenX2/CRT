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
                data = json.load(f)
                return list(dict.fromkeys(data)) if isinstance(data, list) else []
        except Exception:
            return []
    return []

def save_sent_alerts(alerts):
    deduped = list(dict.fromkeys(alerts))
    with open(ALERTS_FILE, 'w') as f:
        json.dump(deduped, f, indent=4)

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
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
        files = {'photo': photo}
        response = requests.post(url, data=payload, files=files)
        if not response.ok:
            print(f"Failed to send Telegram alert: {response.text}")
            return False
        return True

def get_weekly_trend(df_weekly):
    if df_weekly is None or len(df_weekly) < 201:
        return "UNKNOWN"

    df_weekly = df_weekly.copy()
    df_weekly['EMA200'] = df_weekly['Close'].ewm(span=200, adjust=False).mean()
    last_closed = df_weekly.iloc[-2]
    prior_closed = df_weekly.iloc[-6]

    current_close = last_closed['Close']
    current_ema = last_closed['EMA200']

    recent_hh = prior_closed['High'] < last_closed['High']
    recent_hl = prior_closed['Low'] < last_closed['Low']

    is_uptrend = (current_close > current_ema) and (recent_hh or recent_hl)
    is_downtrend = (current_close < current_ema) and not (recent_hh or recent_hl)

    if is_uptrend:
        return "Uptrend"
    elif is_downtrend:
        return "Downtrend"
    return "Consolidation"

def scan_key_levels(df_closed):
    key_levels = []
    if df_closed is None or len(df_closed) < 15:
        return key_levels

    for i in range(2, len(df_closed) - 1):
        # Unmitigated Fair Value Gaps (FVG) only
        if df_closed.iloc[i-1]['Low'] > df_closed.iloc[i+1]['High']:
            fvg_top = df_closed.iloc[i-1]['Low']
            fvg_bottom = df_closed.iloc[i+1]['High']
            mitigated = any(df_closed.iloc[j]['Low'] <= fvg_bottom for j in range(i+2, len(df_closed)))
            if not mitigated:
                key_levels.append({"type": "FVG", "top": fvg_top, "bottom": fvg_bottom})
        elif df_closed.iloc[i-1]['High'] < df_closed.iloc[i+1]['Low']:
            fvg_top = df_closed.iloc[i+1]['High']
            fvg_bottom = df_closed.iloc[i-1]['Low']
            mitigated = any(df_closed.iloc[j]['High'] >= fvg_top for j in range(i+2, len(df_closed)))
            if not mitigated:
                key_levels.append({"type": "FVG", "top": fvg_top, "bottom": fvg_bottom})

    return key_levels

def check_kl_confluence(price_point, key_levels):
    for kl in key_levels:
        if kl['bottom'] <= price_point <= kl['top']:
            return True, f"Tapped {kl['type']} ({kl['bottom']:.5f} - {kl['top']:.5f})"
    return False, "No FVG Intersected"

def generate_crt_chart(symbol, timeframe_name, df_closed, ref_high, ref_low, setup_type):
    plot_df = df_closed.tail(40).copy()

    tv_style = mpf.make_mpf_style(
        base_mpf_style='charles',
        marketcolors=mpf.make_marketcolors(
            up='white', down='black',
            edge={'up': 'black', 'down': 'black'},
            wick={'up': 'black', 'down': 'black'},
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

def closed_inside_range(close_price, low_bound, high_bound):
    return (low_bound <= close_price) and (close_price <= high_bound)

def normalize_timestamp(ts):
    return pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def evaluate_crt(symbol, timeframe_name, df, df_weekly):
    if df is None or len(df) < 8:
        return None

    df_closed = df.iloc[:-1]

    sweep_bar = df_closed.iloc[-1]
    sweep_time = normalize_timestamp(df_closed.index[-1])
    sweep_low = sweep_bar['Low']
    sweep_high = sweep_bar['High']
    sweep_close = sweep_bar['Close']

    ref_bar = df_closed.iloc[-2]
    ref_high = ref_bar['High']
    ref_low = ref_bar['Low']

    pre_ref_bar = df_closed.iloc[-3]

    multi_low = df_closed['Low'].iloc[-6:-1].min()
    multi_high = df_closed['High'].iloc[-6:-1].max()

    weekly_trend = get_weekly_trend(df_weekly)
    all_kls = scan_key_levels(df_closed)

    setup_model = None
    setup_type_str = None
    sweep_extreme = None

    # 1. Inside Bar CRT Check (closed inside reference candle's range)
    is_inside_bar = (ref_high <= pre_ref_bar['High']) and (ref_low >= pre_ref_bar['Low'])
    if is_inside_bar and sweep_low < ref_low and closed_inside_range(sweep_close, ref_low, ref_high):
        setup_model = "Inside Bar CRT"
        setup_type_str = "Bullish Inside Bar Sweep"
        sweep_extreme = sweep_low
    elif is_inside_bar and sweep_high > ref_high and closed_inside_range(sweep_close, ref_low, ref_high):
        setup_model = "Inside Bar CRT"
        setup_type_str = "Bearish Inside Bar Sweep"
        sweep_extreme = sweep_high

    # 2. 2-Candle CRT Check (closed inside reference candle's range with rejection)
    elif sweep_low < ref_low and closed_inside_range(sweep_close, ref_low, ref_high) and (sweep_close - sweep_low) > (sweep_high - sweep_close):
        setup_model = "2-Candle CRT"
        setup_type_str = "Bullish 2-Candle Rejection"
        sweep_extreme = sweep_low
    elif sweep_high > ref_high and closed_inside_range(sweep_close, ref_low, ref_high) and (sweep_high - sweep_close) > (sweep_close - sweep_low):
        setup_model = "2-Candle CRT"
        setup_type_str = "Bearish 2-Candle Rejection"
        sweep_extreme = sweep_high

    # 3. Multiple-Candle CRT Check (closed inside multi-bar reference range)
    elif sweep_low < multi_low and closed_inside_range(sweep_close, multi_low, multi_high):
        setup_model = "Multiple-Candle CRT"
        setup_type_str = "Bullish Multi-Bar Sweep"
        sweep_extreme = sweep_low
        ref_low = multi_low
        ref_high = multi_high
    elif sweep_high > multi_high and closed_inside_range(sweep_close, multi_low, multi_high):
        setup_model = "Multiple-Candle CRT"
        setup_type_str = "Bearish Multi-Bar Sweep"
        sweep_extreme = sweep_high
        ref_low = multi_low
        ref_high = multi_high

    # 4. Classic 3-Candle CRT Check (closed inside reference candle's range)
    elif sweep_low < ref_low and closed_inside_range(sweep_close, ref_low, ref_high):
        setup_model = "Classic 3-Candle CRT"
        setup_type_str = "Bullish Classic Sweep"
        sweep_extreme = sweep_low
    elif sweep_high > ref_high and closed_inside_range(sweep_close, ref_low, ref_high):
        setup_model = "Classic 3-Candle CRT"
        setup_type_str = "Bearish Classic Sweep"
        sweep_extreme = sweep_high

    if not setup_model:
        return None

    is_bullish = "Bullish" in setup_type_str
    is_aligned = (weekly_trend == "Uptrend" if is_bullish else weekly_trend == "Downtrend")

    _, kl_note = check_kl_confluence(sweep_extreme, all_kls)
    classification = "High Probability CRT (Trend-Following)" if is_aligned else "Reversal CRT"
    classification += f" + {kl_note}"

    setup = {
        "symbol": symbol,
        "timeframe": timeframe_name,
        "sweep_time": sweep_time,
        "model": setup_model,
        "type": setup_type_str,
        "classification": classification,
        "rc_high": ref_high,
        "rc_low": ref_low,
        "sweep_extreme": sweep_extreme,
        "weekly_trend": weekly_trend,
        "kl_note": kl_note
    }

    chart_path = generate_crt_chart(symbol, timeframe_name, df_closed, ref_high, ref_low, setup_type_str)
    setup['chart_path'] = chart_path

    return setup

def run_bot_loop():
    print("Starting CRT Yahoo Finance Scanner (Daily 11:01 PM WAT | Closed Sweep + Inside Range + FVG Only)...")
    sent_alerts = load_sent_alerts()
    sent_set = set(sent_alerts)
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
                alert_id = (
                    f"{symbol}_{tf_name}_{setup['sweep_time']}_{setup['model']}_"
                    f"{setup['type']}_{setup['rc_high']:.5f}_{setup['rc_low']:.5f}"
                )

                if alert_id in sent_set:
                    print(f"Skipping already sent alert for {symbol} [{tf_name}] ({setup['model']})")
                    continue

                caption = (
                    f"🚨 *CRT ALERT (CLOSED INSIDE): {setup['classification']}*\n\n"
                    f"• *Asset:* `{setup['symbol']}`\n"
                    f"• *Timeframe:* {setup['timeframe']}\n"
                    f"• *Sweep Close:* `{setup['sweep_time']}`\n"
                    f"• *CRT Model:* `{setup['model']}`\n"
                    f"• *Setup Variant:* {setup['type']}\n"
                    f"• *Weekly Trend:* {setup['weekly_trend']}\n"
                    f"• *Key Level (FVG):* {setup['kl_note']}\n"
                    f"• *CRT High:* `{setup['rc_high']:.5f}`\n"
                    f"• *CRT Low:* `{setup['rc_low']:.5f}`\n"
                    f"• *Sweep Extreme:* `{setup['sweep_extreme']:.5f}`"
                )
                if send_telegram_alert(setup['chart_path'], caption):
                    print(f"Alert sent for {setup['symbol']} [{setup['timeframe']}] - {setup['model']}")
                    sent_alerts.append(alert_id)
                    sent_set.add(alert_id)
                    new_alerts_found = True

    if new_alerts_found:
        save_sent_alerts(sent_alerts)

if __name__ == "__main__":
    run_bot_loop()
