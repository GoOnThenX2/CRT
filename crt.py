import os
import requests
import pandas as pd
import numpy as np
import mplfinance as mpf
import yfinance as yf

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TICKER_MAPPING = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", 
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "NZDUSD": "NZDUSD=X", "USDCHF": "USDCHF=X",
    "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X", "EURAUD": "EURAUD=X", "EURNZD": "EURNZD=X", 
    "EURCAD": "EURCAD=X", "EURCHF": "EURCHF=X", "GBPJPY": "GBPJPY=X", "GBPAUD": "GBPAUD=X", 
    "GBPNZD": "GBPNZD=X", "GBPCAD": "GBPCAD=X", "GBPCHF": "GBPCHF=X",
    "AUDJPY": "AUDJPY=X", "AUDNZD": "AUDNZD=X", "AUDCAD": "AUDCAD=X", "AUDCHF": "AUDCHF=X",
    "NZDJPY": "NZDJPY=X", "NZDCAD": "NZDCAD=X", "NZDCHF": "NZDCHF=X",
    "CADJPY": "CADJPY=X", "CADCHF": "CADCHF=X", "CHFJPY": "CHFJPY=X",
    "NAS100": "^NDX", "SPX500": "^GSPC", "US30": "^DJI", 
    "JP225": "^N225", "GER40": "^GDAXI", "EURO50": "^STOXX50E",
    "Gold": "GC=F", "Silver": "SI=F",
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD"
}

TIMEFRAMES = {
    "Monthly": {"interval": "1mo", "period": "10y"},
    "Weekly": {"interval": "1wk", "period": "5y"},
    "Daily": {"interval": "1d", "period": "1y"}
}

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

        if i > 4 and df.iloc[i]['High'] > df.iloc[i-2]['High'] and df.iloc[i]['Low'] < df.iloc[i-3]['Low']:
            qm_level = df.iloc[i-2]['High']
            key_levels.append({"type": "Quasimodo", "top": qm_level * 1.001, "bottom": qm_level * 0.999})
                
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
    if df is None or len(df) < 3:
        return None
        
    ref_high = df.iloc[-2]['High']
    ref_low = df.iloc[-2]['Low']
    
    curr_low = df.iloc[-1]['Low']
    curr_high = df.iloc[-1]['High']
    curr_close = df.iloc[-1]['Close']
    
    weekly_trend = get_weekly_trend(df_weekly)
    all_kls = scan_key_levels(df)
    
    setup = None
    
    if curr_low < ref_low and curr_close > ref_low:
        has_kl, kl_note = check_kl_confluence(curr_low, all_kls)
        is_aligned = (weekly_trend == "Uptrend")
        
        classification = "High Probability CRT (Trend-Following)" if is_aligned else "Reversal CRT"
        classification += f" + {kl_note}"
            
        setup = {
            "symbol": symbol,
            "timeframe": timeframe_name,
            "type": "Bullish CRT (Low Sweep)",
            "classification": classification,
            "rc_high": ref_high,
            "rc_low": ref_low,
            "sweep_extreme": curr_low,
            "weekly_trend": weekly_trend,
            "kl_note": kl_note
        }
        
    elif curr_high > ref_high and curr_close < ref_high:
        has_kl, kl_note = check_kl_confluence(curr_high, all_kls)
        is_aligned = (weekly_trend == "Downtrend")
        
        classification = "High Probability CRT (Trend-Following)" if is_aligned else "Reversal CRT"
        classification += f" + {kl_note}"
            
        setup = {
            "symbol": symbol,
            "timeframe": timeframe_name,
            "type": "Bearish CRT (High Sweep)",
            "classification": classification,
            "rc_high": ref_high,
            "rc_low": ref_low,
            "sweep_extreme": curr_high,
            "weekly_trend": weekly_trend,
            "kl_note": kl_note
        }
        
    if setup:
        chart_path = generate_crt_chart(symbol, timeframe_name, df, ref_high, ref_low, setup['type'])
        setup['chart_path'] = chart_path
        
    return setup

def run_bot_loop():
    print("Starting CRT Yahoo Finance Scanner...")
    
    for symbol, ticker in TICKER_MAPPING.items():
        print(f"Scanning {symbol} ({ticker})...")
        df_weekly = fetch_yahoo_data(ticker, TIMEFRAMES["Weekly"]["interval"], TIMEFRAMES["Weekly"]["period"])
        
        if df_weekly is None:
            continue
        
        for tf_name, tf_config in TIMEFRAMES.items():
            df_tf = fetch_yahoo_data(ticker, tf_config["interval"], tf_config["period"])
            setup = evaluate_crt(symbol, tf_name, df_tf, df_weekly)
            
            if setup:
                caption = (
                    f"🚨 *CRT ALERT: {setup['classification']}*\n\n"
                    f"• *Asset:* `{setup['symbol']}`\n"
                    f"• *Timeframe:* {setup['timeframe']}\n"
                    f"• *Setup Type:* {setup['type']}\n"
                    f"• *Weekly Trend:* {setup['weekly_trend']}\n"
                    f"• *Key Level (KL):* {setup['kl_note']}\n"
                    f"• *CRT High:* `{setup['rc_high']:.5f}`\n"
                    f"• *CRT Low:* `{setup['rc_low']:.5f}`\n"
                    f"• *Sweep Extreme:* `{setup['sweep_extreme']:.5f}`"
                )
                send_telegram_alert(setup['chart_path'], caption)
                print(f"Alert sent for {setup['symbol']} [{setup['timeframe']}]")

if __name__ == "__main__":
    run_bot_loop()
