import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import pytz
from datetime import datetime, timedelta, time
import numpy as np
import seaborn as sns
import os
from itertools import product
from typing import Dict, Optional, Tuple, Any

# Streamlit app configuration
st.set_page_config(layout="wide", page_title="Rebound Strategy Analyzer")

# Main title
st.title("Rebound Strategy Analyzer")
st.markdown("""
This app analyzes a rebound trading strategy that trades the reversal after opening range breakouts.
""")

# Sidebar for user inputs
with st.sidebar:

    if st.button("📄 Download Documentation"):
          # Direct link to raw PDF on GitHub
          pdf_url = "https://github.com/ethosconsulting/ORR/raw/main/ORRDocumentation.pdf"
          
          try:
              # Fetch the PDF from GitHub
              response = requests.get(pdf_url)
              response.raise_for_status()  # Check for errors
              
              # Create download button with the PDF content
              st.download_button(
                  label="⬇️ Save PDF Now",
                  data=response.content,
                  file_name="ORR_Documentation.pdf",
                  mime="application/pdf"
              )
          except requests.exceptions.RequestException as e:
              st.error(f"Failed to download documentation: {e}")
              
    st.header("Strategy Parameters")

    # Ticker symbol
    ticker = st.text_input("Ticker Symbol", value='GC=F')

    # Timezone selection
    timezone = st.selectbox(
        "Timezone",
        ['Europe/London', 'America/New_York', 'Asia/Tokyo'],
        index=0
    )

    # Trading session hours - using string options but converting to int later
    st.subheader("Trading Session Hours")
    col1, col2 = st.columns(2)
    with col1:
        start_hour = st.number_input("Start Hour (0-23)", min_value=0, max_value=23, value=12)
        start_minute = st.selectbox("Start Minute", options=['00', '15', '30', '45', '59'], index=0)
    with col2:
        end_hour = st.number_input("End Hour (0-23)", min_value=0, max_value=23, value=18)
        end_minute = st.selectbox("End Minute", options=['00', '15', '30', '45','59'], index=0)

    # Opening range configuration
    st.subheader("Opening Range Configuration")
    col1, col2 = st.columns(2)
    with col1:
        or_start_hour = st.number_input("OR Start Hour (0-23)", min_value=0, max_value=23, value=14)
        or_start_minute = st.selectbox("OR Start Minute", options=['00', '15', '30', '45'], index=2)
    with col2:
        or_end_hour = st.number_input("OR End Hour (0-23)", min_value=0, max_value=23, value=15)
        or_end_minute = st.selectbox("OR End Minute", options=['00', '15', '30', '45'], index=0)

    # Rebound confirmation parameters
    st.subheader("Rebound Confirmation")
    confirmation_bars = st.number_input("Number of Confirmation Bars", min_value=1, max_value=5, value=1)
    close_inside = st.checkbox("Require Close Inside OR", value=True)
    buffer_multiplier = st.number_input("Buffer Multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.1)

    # Date range
    st.subheader("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        # Set start_date to 50 days before today
        start_date = st.date_input("Start Date", 
                                  value=datetime.now() - timedelta(days=55),
                                  max_value=datetime.now() - timedelta(days=1))
    with col2:
        # Set end_date to today
        end_date = st.date_input("End Date", 
                                value=datetime.now(),
                                max_value=datetime.now())

    st.subheader("TP-SL Range (Absolute Points)")
    col1, col2 = st.columns(2)
    with col1:
        tp_min = st.number_input("TP Min (points)", min_value=0.1, max_value=100.0, value=4.0, step=0.5)
        tp_max = st.number_input("TP Max (points)", min_value=0.1, max_value=100.0, value=10.0, step=0.5)
        tp_step = st.number_input("TP Step", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    with col2:
        sl_min = st.number_input("SL Min (points)", min_value=0.1, max_value=100.0, value=4.0, step=0.5)
        sl_max = st.number_input("SL Max (points)", min_value=0.1, max_value=100.0, value=10.0, step=0.5)
        sl_step = st.number_input("SL Step", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    # Optimization Settings
    st.subheader("Optimization Settings")
    skip_poor_rr = st.checkbox("Skip poor risk-reward (SL < TP)", value=True)

    # Input: Friction parameters
    st.subheader("Friction parameters")
    buffer_pts = st.sidebar.number_input("Buffer (pts)", min_value=0.0, max_value=10.0, value=0.5, step=0.05)
    cost_pts = st.sidebar.number_input("Commission + Slippage (pts)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Helper functions (same as before)
def safe_float(value) -> float:
    """Type-safe conversion for pandas/float inputs"""
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return float(value.iloc[0])
    return float(value)

def get_trading_days_data(start_date, end_date, ticker, timezone) -> pd.DataFrame:
    """Fetch 5-minute data for the specified date range"""
    # Convert string dates to datetime objects
    start_dt = pd.to_datetime(start_date).tz_localize(timezone)
    end_dt = pd.to_datetime(end_date).tz_localize(timezone) + timedelta(days=1)  # Include full end date

    # Download data with buffer days
    data = yf.download(
        tickers=ticker,
        interval='5m',
        start=start_dt - timedelta(days=2),  # Buffer for timezone issues
        end=end_dt + timedelta(days=2),      # Buffer for timezone issues
        progress=False
    )

    # Convert to specified timezone
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert(timezone)

    # Filter for the exact date range
    data = data[(data.index >= start_dt) & (data.index <= end_dt)]

    return data

def format_duration_minutes(td):
    """Helper function to format timedelta as minutes"""
    if pd.isna(td):
        return "N/A"
    # If it's already a number (from analyze_strategy_performance conversion)
    if isinstance(td, (int, float)):
        return f"{td:.1f} minutes"
    # If it's a timedelta object
    return f"{td.total_seconds()/60:.1f} minutes"

def split_data_by_day(data: pd.DataFrame, start_hour, start_minute, end_hour, end_minute) -> Dict[datetime.date, pd.DataFrame]:
    """Split the data into separate days"""
    daily_data = {}
    for date, group in data.groupby(data.index.date):
        # Filter for trading hours
        trading_day = group[
            (group.index.time >= pd.to_datetime(f'{start_hour}:{start_minute}').time()) &
            (group.index.time <= pd.to_datetime(f'{end_hour}:{end_minute}').time())
        ]
        if not trading_day.empty:
            daily_data[date] = trading_day
    return daily_data

def get_opening_range_levels(data: pd.DataFrame, or_start_hour: int, or_start_minute: str,
                           or_end_hour: int, or_end_minute: str,
                           start_hour: int, start_minute: str,
                           end_hour: int, end_minute: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int]]:
    """Returns (high, low, atr, expected_candles) with proper type handling"""
    # Convert string minutes to integers
    or_start_min = int(or_start_minute)
    or_end_min = int(or_end_minute)

    start_time = time(or_start_hour, or_start_min)
    end_time = time(or_end_hour, or_end_min)

    mask = (data.index.time >= start_time) & (data.index.time <= end_time)
    opening_range = data.loc[mask]

    if opening_range.empty:
        return None, None, None, None

    # Calculate expected candles for the full session
    session_start = time(start_hour, int(start_minute))
    session_end = time(end_hour, int(end_minute))
    session_date = data.index[0].date()
    session_duration = (datetime.combine(session_date, session_end) -
                      datetime.combine(session_date, session_start)).total_seconds() / 60
    expected_candles = int(session_duration // 5)

    high = safe_float(opening_range['High'].max())
    low = safe_float(opening_range['Low'].min())
    close = safe_float(opening_range['Close'].iloc[-1])

    tr = max(
        high - low,
        abs(high - close),
        abs(low - close)
    )

    return high, low, tr, expected_candles

def analyze_strategy_performance(trades_df: pd.DataFrame) -> Dict:
    """Calculate strategy performance metrics from the trades DataFrame"""
    if trades_df.empty:
        return {}

    metrics = {
        'total_trades': len(trades_df),
        'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
        'losing_trades': len(trades_df[trades_df['pnl'] < 0]),
        'win_rate': len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100,
        'total_pnl': trades_df['pnl'].sum(),
        'avg_pnl': trades_df['pnl'].mean(),
        'avg_trade_duration': trades_df['position_duration'].mean(),
        'max_win': trades_df['pnl'].max(),
        'max_loss': trades_df['pnl'].min(),
        'profit_factor': abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() /
                           trades_df[trades_df['pnl'] < 0]['pnl'].sum()),
        'long_trades': len(trades_df[trades_df['direction'] == 'long']),
        'short_trades': len(trades_df[trades_df['direction'] == 'short']),
        'long_win_rate': len(trades_df[(trades_df['direction'] == 'long') & (trades_df['pnl'] > 0)]) /
                        max(1, len(trades_df[trades_df['direction'] == 'long'])) * 100,
        'short_win_rate': len(trades_df[(trades_df['direction'] == 'short') & (trades_df['pnl'] > 0)]) /
                         max(1, len(trades_df[trades_df['direction'] == 'short'])) * 100,
        'up_breakout_trades': len(trades_df[trades_df['breakout_direction'] == 'up']),
        'down_breakout_trades': len(trades_df[trades_df['breakout_direction'] == 'down']),
        'up_breakout_win_rate': len(trades_df[(trades_df['breakout_direction'] == 'up') & (trades_df['pnl'] > 0)]) /
                               max(1, len(trades_df[trades_df['breakout_direction'] == 'up'])) * 100,
        'down_breakout_win_rate': len(trades_df[(trades_df['breakout_direction'] == 'down') & (trades_df['pnl'] > 0)]) /
                                max(1, len(trades_df[trades_df['breakout_direction'] == 'down'])) * 100
    }

    # Convert timedelta to minutes for display
    if not pd.isna(metrics['avg_trade_duration']):
        metrics['avg_trade_duration'] = metrics['avg_trade_duration'].total_seconds() / 60

    return metrics

def plot_single_day(data: pd.DataFrame, date: datetime.date,
                   opening_high: Optional[float], opening_low: Optional[float],
                   trade: Optional[Dict] = None, timezone='Europe/London'):
    """Plot single trading day with VWAP, EMAs, volume, and ATR"""
    # Calculate all indicators
    hlc3 = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (hlc3 * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['Volume_EMA21'] = data['Volume'].ewm(span=21, adjust=False).mean()

    # Prepare OHLC data with color coding
    ohlc = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    ohlc['Color'] = np.where(ohlc['Close'] >= ohlc['Open'], 'green', 'red')
    ohlc = ohlc.reset_index()
    ohlc['Date_mpl'] = ohlc['Datetime'].apply(lambda x: mdates.date2num(x.to_pydatetime()))

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1])
    ax1 = fig.add_subplot(gs[0])  # Price with EMAs
    ax2 = fig.add_subplot(gs[1])  # Volume
    ax3 = fig.add_subplot(gs[2])  # ATR

    # --- Price Chart ---
    # Candlesticks
    candlestick_ohlc(
        ax=ax1,
        quotes=ohlc[['Date_mpl', 'Open', 'High', 'Low', 'Close']].values,
        width=0.0015,
        colorup='g',
        colordown='r',
        alpha=1
    )

    # Moving Averages
    ax1.plot(ohlc['Date_mpl'], data['EMA21'], color='blue', linewidth=1.5, label='EMA 21')
    ax1.plot(ohlc['Date_mpl'], data['EMA200'], color='red', linewidth=1.5, label='EMA 200')

    # VWAP line (yellow)
    ax1.plot(ohlc['Date_mpl'], data['VWAP'], color='yellow', linewidth=2, label='VWAP (HLC3)')

    # Opening range levels
    if opening_high is not None:
        ax1.axhline(y=opening_high, color='blue', linestyle='--', label=f'OR High: {opening_high:.2f}')
    if opening_low is not None:
        ax1.axhline(y=opening_low, color='purple', linestyle='--', label=f'OR Low: {opening_low:.2f}')

    # Trade annotations
    if trade is not None:
        entry_time = mdates.date2num(trade['entry_time'].to_pydatetime())
        exit_time = mdates.date2num(trade['exit_time'].to_pydatetime())

        # Highlight breakout area
        breakout_time = mdates.date2num(
            data[(data.index.time > time(or_end_hour, int(or_end_minute))) &
            (data.index.time <= trade['entry_time'].time())].index[0].to_pydatetime()
        )

        ax1.axvspan(breakout_time, entry_time, alpha=0.2, color='orange', label='Breakout Confirmation')

        ax1.plot(entry_time, trade['entry_price'], 'bo', markersize=10, label='Entry')
        ax1.plot(exit_time, trade['exit_price'], 'ro', markersize=10, label='Exit')
        ax1.plot([entry_time, exit_time],
                [trade['entry_price'], trade['exit_price']],
                'k-', linewidth=2)

        trade_summary = (
            f"Trade Executed:\n"
            f"Direction: {trade['direction']}\n"
            f"Breakout: {trade['breakout_direction']}\n"
            f"Entry Time: {trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"Entry Price: ${trade['entry_price']:.2f}\n"
            f"Exit Time: {trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"Exit Price: ${trade['exit_price']:.2f}\n"
            f"Exit Reason: {trade['exit_reason']}\n"
            f"P/L: {trade['pnl']:.2f} points"
        )

        ax1.annotate(trade_summary,
                    xy=(exit_time, trade['exit_price']),
                    xytext=(50, 30),
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='black'))

    # --- Volume Chart ---
    # Colored volume bars
    for idx, row in ohlc.iterrows():
        ax2.bar(row['Date_mpl'], row['Volume'], width=0.001, color=row['Color'], alpha=0.7)

    # Volume EMA 21
    ax2.plot(ohlc['Date_mpl'], data['Volume_EMA21'], color='blue', linewidth=1.5, label='Volume EMA 21')
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper left')

    # --- ATR Chart ---
    if trade is not None:
        ax3.axhline(y=trade['atr'], color='green', linestyle='-', label='ATR')
        ax3.axhline(y=trade['tp_distance'], color='blue', linestyle='--', label='TP Distance')
        ax3.axhline(y=trade['sl_distance'], color='red', linestyle='--', label='SL Distance')
        ax3.set_ylabel('Points')
        ax3.legend()
        ax3.set_title('ATR and Stop Levels')

    # Formatting
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(timezone)))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))

    start_dt = pytz.timezone(timezone).localize(
        datetime.combine(date, datetime.strptime(f'{start_hour}:00', '%H:%M').time()))
    end_dt = pytz.timezone(timezone).localize(
        datetime.combine(date, datetime.strptime(f'{end_hour}:{end_minute}', '%H:%M').time()))

    margin = timedelta(minutes=5)
    x_min = mdates.date2num(start_dt - margin)
    x_max = mdates.date2num(end_dt + margin)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(x_min, x_max)

    ax1.grid(True, which='major', linestyle='-', alpha=0.7)
    ax1.set_title(f'{ticker} 5-minute Chart ({date}) - Rebound Strategy')
    ax1.legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_optimization_heatmap(results_df: pd.DataFrame):
    """Create a heatmap of absolute TP/SL optimization results"""
    heatmap_data = results_df.pivot(index='sl_value', columns='tp_value', values='total_pnl')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Total P/L (points)'},
        ax=ax
    )
    ax.set_title('TP/SL Optimization Results (Absolute Values)')
    ax.set_xlabel('Take Profit (points)')
    ax.set_ylabel('Stop Loss (points)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_win_rate_heatmap(results_df: pd.DataFrame):
    """Create a heatmap of win rate optimization results"""
    heatmap_data = results_df.pivot(index='sl_value', columns='tp_value', values='win_rate')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".1f",
        cmap='RdYlGn',
        vmin=0,  # Win rate can't be negative
        vmax=100,  # Maximum win rate is 100%
        cbar_kws={'label': 'Win Rate (%)'},
        ax=ax
    )
    ax.set_title('Win Rate Optimization Results')
    ax.set_xlabel('Take Profit (points)')
    ax.set_ylabel('Stop Loss (points)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def plot_efficiency_frontier(results_df: pd.DataFrame):
    """Create an efficiency frontier plot showing win rate vs total P/L"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create color mapping based on risk-reward ratio (TP/SL)
    results_df['risk_reward'] = results_df['tp_value'] / results_df['sl_value']
    colors = results_df['risk_reward']

    # Plot all points with color indicating risk-reward ratio
    scatter = ax.scatter(
        x=results_df['win_rate'],
        y=results_df['total_pnl'],
        c=colors,
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='w',
        linewidths=0.5
    )

    # Highlight Pareto efficient points (frontier)
    sorted_df = results_df.sort_values('win_rate')
    pareto_front = []
    current_max = -np.inf
    for _, row in sorted_df.iterrows():
        if row['total_pnl'] > current_max:
            pareto_front.append(row)
            current_max = row['total_pnl']
    pareto_df = pd.DataFrame(pareto_front)

    ax.plot(
        pareto_df['win_rate'],
        pareto_df['total_pnl'],
        'r--',
        alpha=0.7,
        label='Efficiency Frontier'
    )

    # Add labels and annotations for interesting points
    max_pnl_idx = results_df['total_pnl'].idxmax()
    max_pnl_point = results_df.loc[max_pnl_idx]
    ax.scatter(
        x=max_pnl_point['win_rate'],
        y=max_pnl_point['total_pnl'],
        c='red',
        s=200,
        marker='s',
        label=f"Max P/L (TP={max_pnl_point['tp_value']}, SL={max_pnl_point['sl_value']})"
    )

    max_win_rate_idx = results_df['win_rate'].idxmax()
    max_win_rate_point = results_df.loc[max_win_rate_idx]
    ax.scatter(
        x=max_win_rate_point['win_rate'],
        y=max_win_rate_point['total_pnl'],
        c='blue',
        s=200,
        marker='o',
        label=f"Max Win Rate (TP={max_win_rate_point['tp_value']}, SL={max_win_rate_point['sl_value']})"
    )

    # Add colorbar for risk-reward ratio
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Risk-Reward Ratio (TP/SL)')

    # Formatting
    ax.set_xlabel('Win Rate (%)')
    ax.set_ylabel('Total P/L (points)')
    ax.set_title('Efficiency Frontier: Win Rate vs Total P/L')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def optimize_tp_sl(daily_data: Dict[datetime.date, pd.DataFrame], tp_range, sl_range, buffer: float, cost: float) -> pd.DataFrame:
    """Grid search optimization using absolute TP/SL values"""
    results = []

    for tp_val, sl_val in product(tp_range, sl_range):
        # Skip poor risk-reward combinations
        if skip_poor_rr and tp_val < sl_val:
            continue

        metrics = {
            'tp_value': tp_val,
            'sl_value': sl_val,
            'total_pnl': 0,
            'win_rate': 0,
            'trades': 0,
            'avg_pnl': 0
        }

        for date, day_data in daily_data.items():
            high, low, tr, expected_candles = get_opening_range_levels(
                day_data, or_start_hour, or_start_minute, or_end_hour, or_end_minute,
                start_hour, start_minute, end_hour, end_minute
            )
            if None in (high, low, tr):
                continue

            trade = simulate_rebound_trade(
                data=day_data,
                opening_high=high,
                opening_low=low,
                atr=tr,
                expected_candles=expected_candles,
                tp_value=tp_val,
                sl_value=sl_val,
                or_start_hour=or_start_hour,
                or_start_minute=or_start_minute,
                or_end_hour=or_end_hour,
                or_end_minute=or_end_minute,
                buffer=buffer,
                cost=cost,
                confirmation_bars=confirmation_bars,
                close_inside=close_inside,
                buffer_multiplier=buffer_multiplier
            )

            if trade:
                metrics['total_pnl'] += trade['pnl']
                metrics['win_rate'] += 1 if trade['pnl'] > 0 else 0
                metrics['trades'] += 1

        if metrics['trades'] > 0:
            metrics['win_rate'] = metrics['win_rate'] / metrics['trades'] * 100
            metrics['avg_pnl'] = metrics['total_pnl'] / metrics['trades']
            results.append(metrics)

    return pd.DataFrame(results).sort_values('total_pnl', ascending=False)

def simulate_rebound_trade(
    data: pd.DataFrame,
    opening_high: float,
    opening_low: float,
    atr: float,
    expected_candles: int,
    tp_value: float,
    sl_value: float,
    or_start_hour: int,
    or_start_minute: str,
    or_end_hour: int,
    or_end_minute: str,
    buffer: float,
    cost: float,
    confirmation_bars: int = 1,
    close_inside: bool = True,
    buffer_multiplier: float = 1.0
) -> Optional[Dict[str, Any]]:
    """Execute rebound trade simulation with absolute TP/SL values"""
    # Convert string minutes to integers
    or_start_min = int(or_start_minute)
    or_end_min = int(or_end_minute)

    opening_range_end_time = time(or_end_hour, or_end_min)
    post_opening_data = data[data.index.time > opening_range_end_time]

    trade = {
        'date': data.index[0].date(),
        'candles': len(data),
        'expected_candles': expected_candles,
        'opening_high': opening_high,
        'opening_low': opening_low,
        'direction': None,
        'entry_time': None,
        'entry_price': None,
        'exit_time': None,
        'exit_price': None,
        'exit_reason': None,
        'pnl': None,
        'trade_taken': False,
        'position_duration': None,
        'atr': round(atr, 2),
        'tp_value': round(tp_value, 2),
        'sl_value': round(sl_value, 2),
        'tp_distance': round(tp_value, 2),
        'sl_distance': round(sl_value, 2),
        'breakout_direction': None
    }

    # Track breakout direction and confirmation
    breakout_confirmed = False
    breakout_direction = None
    confirmation_count = 0

    for idx, row in post_opening_data.iterrows():
        current_high = safe_float(row['High'])
        current_low = safe_float(row['Low'])
        current_close = safe_float(row['Close'])

        # Check for breakout if not already confirmed
        if not breakout_confirmed:
            if current_high > opening_high + (buffer * buffer_multiplier):
                breakout_direction = 'up'
                confirmation_count += 1
            elif current_low < opening_low - (buffer * buffer_multiplier):
                breakout_direction = 'down'
                confirmation_count += 1
            else:
                confirmation_count = 0

            # Check if we have enough confirmation bars
            if confirmation_count >= confirmation_bars:
                breakout_confirmed = True
                trade['breakout_direction'] = breakout_direction

        # Entry logic (after breakout is confirmed)
        if breakout_confirmed and trade['entry_time'] is None:
            if breakout_direction == 'up' and (current_close < opening_high if close_inside else True):
                # Take short position after upside breakout
                entry_price = opening_high - buffer - cost
                trade.update({
                    'entry_time': idx,
                    'entry_price': entry_price,
                    'direction': 'short',
                    'tp_price': entry_price - tp_value,
                    'sl_price': entry_price + sl_value,
                    'trade_taken': True
                })
            elif breakout_direction == 'down' and (current_close > opening_low if close_inside else True):
                # Take long position after downside breakout
                entry_price = opening_low + buffer + cost
                trade.update({
                    'entry_time': idx,
                    'entry_price': entry_price,
                    'direction': 'long',
                    'tp_price': entry_price + tp_value,
                    'sl_price': entry_price - sl_value,
                    'trade_taken': True
                })
            continue

        # Exit logic
        if trade['direction'] == 'long':
            if current_low <= trade['sl_price']:
                trade.update({
                    'exit_time': idx,
                    'exit_price': trade['sl_price'],
                    'exit_reason': 'SL',
                    'pnl': -sl_value
                })
                break
            elif current_high >= trade['tp_price']:
                trade.update({
                    'exit_time': idx,
                    'exit_price': trade['tp_price'],
                    'exit_reason': 'TP',
                    'pnl': tp_value
                })
                break
        elif trade['direction'] == 'short':
            if current_high >= trade['sl_price']:
                trade.update({
                    'exit_time': idx,
                    'exit_price': trade['sl_price'],
                    'exit_reason': 'SL',
                    'pnl': -sl_value
                })
                break
            elif current_low <= trade['tp_price']:
                trade.update({
                    'exit_time': idx,
                    'exit_price': trade['tp_price'],
                    'exit_reason': 'TP',
                    'pnl': tp_value
                })
                break

    # EOD exit
    if trade['entry_time'] and not trade['exit_time']:
        last_close = safe_float(post_opening_data['Close'].iloc[-1])
        trade.update({
            'exit_time': post_opening_data.index[-1],
            'exit_price': last_close,
            'exit_reason': 'EOD',
            'pnl': (last_close - trade['entry_price']) if trade['direction'] == 'long'
                   else (trade['entry_price'] - last_close)
        })

    if trade['trade_taken']:
        trade['position_duration'] = trade['exit_time'] - trade['entry_time']

    return trade if trade['trade_taken'] else None

# [Rest of the functions remain the same: plot_single_day, plot_optimization_heatmap,
# plot_win_rate_heatmap, plot_efficiency_frontier, analyze_strategy_performance,
# format_duration_minutes, calculate_vwap]

# Main execution block
if st.sidebar.button("Run Analysis"):
    with st.spinner("Running analysis... This may take a few minutes"):
        # Create parameter ranges from absolute values
        tp_range = np.round(np.arange(tp_min, tp_max + tp_step, tp_step), 2)
        sl_range = np.round(np.arange(sl_min, sl_max + sl_step, sl_step), 2)

        # Get and prepare data
        full_data = get_trading_days_data(start_date, end_date, ticker, timezone)
        if full_data.empty:
            st.error("No data available for the specified date range!")
            st.stop()

        daily_data = split_data_by_day(full_data, start_hour, start_minute, end_hour, end_minute)
        st.success(f"Found {len(daily_data)} trading days in the date range")

        # Run optimization with absolute values
        optimization_results = optimize_tp_sl(
            daily_data,
            tp_range,
            sl_range,
            buffer=buffer_pts,
            cost=cost_pts
        )

        if optimization_results.empty:
            st.error("No valid optimization results found!")
            st.stop()

        # Display results
        st.subheader("Optimization Results")
        st.dataframe(optimization_results.head(10))

        # Plot heatmap
        st.subheader("Parameter Optimization Heatmap")
        plot_optimization_heatmap(optimization_results)

        st.subheader("Win Rate Optimization Heatmap")
        plot_win_rate_heatmap(optimization_results)

        st.subheader("Efficiency Frontier Analysis")
        plot_efficiency_frontier(optimization_results)

        # Get best parameters
        best_params = optimization_results.iloc[0]
        tp_opt = best_params['tp_value']
        sl_opt = best_params['sl_value']
        st.success(f"Optimal parameters found: TP = {tp_opt:.2f} points, SL = {sl_opt:.2f} points")

        # Generate trades with optimal parameters
        all_trades = []
        for date, day_data in daily_data.items():
            high, low, tr, expected_candles = get_opening_range_levels(
                day_data, or_start_hour, or_start_minute, or_end_hour, or_end_minute,
                start_hour, start_minute, end_hour, end_minute
            )
            if None in (high, low, tr):
                continue

            trade = simulate_rebound_trade(
                data=day_data,
                opening_high=high,
                opening_low=low,
                atr=tr,
                expected_candles=expected_candles,
                tp_value=tp_opt,
                sl_value=sl_opt,
                or_start_hour=or_start_hour,
                or_start_minute=or_start_minute,
                or_end_hour=or_end_hour,
                or_end_minute=or_end_minute,
                buffer=buffer_pts,
                cost=cost_pts,
                confirmation_bars=confirmation_bars,
                close_inside=close_inside,
                buffer_multiplier=buffer_multiplier
            )

            if trade:
                all_trades.append(trade)

        # Create trades DataFrame
        if all_trades:
            trades_df = pd.DataFrame(all_trades)

            # Convert and format columns
            if 'date' in trades_df.columns and trades_df['date'].dtype == object:
                trades_df['date'] = pd.to_datetime(trades_df['date']).dt.date

            if 'position_duration' in trades_df.columns:
                trades_df['position_duration'] = pd.to_timedelta(trades_df['position_duration'])

            # Display trade details
            st.subheader("Trade Details")
            st.dataframe(trades_df)

            # Performance analysis
            metrics = analyze_strategy_performance(trades_df)

            st.subheader("Strategy Performance")

            # Overall Performance
            st.markdown("### Overall Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Trades", metrics.get('total_trades', 0))
                st.metric("Winning Trades", metrics.get('winning_trades', 0))
                st.metric("Losing Trades", metrics.get('losing_trades', 0))
                st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
            with col2:
                st.metric("Total PNL", f"{metrics.get('total_pnl', 0):.2f}")
                st.metric("Average PNL", f"{metrics.get('avg_pnl', 0):.2f}")
                st.metric("Average Trade Duration", format_duration_minutes(metrics.get('avg_trade_duration')))
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

            # Long Positions Performance
            st.markdown("### Long Positions Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Long Trades", metrics.get('long_trades', 0))
                st.metric("Winning Long Trades", len(trades_df[(trades_df['direction'] == 'long') & (trades_df['pnl'] > 0)]))
                st.metric("Losing Long Trades", len(trades_df[(trades_df['direction'] == 'long') & (trades_df['pnl'] < 0)]))
                st.metric("Long Win Rate", f"{metrics.get('long_win_rate', 0):.1f}%")
            with col2:
                long_trades = trades_df[trades_df['direction'] == 'long']
                long_pnl = long_trades['pnl'].sum() if not long_trades.empty else 0
                st.metric("Long Total PNL", f"{long_pnl:.2f}")
                st.metric("Long Average PNL", f"{long_trades['pnl'].mean() if not long_trades.empty else 0:.2f}")
                st.metric("Long Average Duration", format_duration_minutes(long_trades['position_duration'].mean() if not long_trades.empty else None))
                long_profit_factor = abs(long_trades[long_trades['pnl'] > 0]['pnl'].sum() / long_trades[long_trades['pnl'] < 0]['pnl'].abs().sum()) if not long_trades.empty and len(long_trades[long_trades['pnl'] < 0]) > 0 else 0
                st.metric("Long Profit Factor", f"{long_profit_factor:.2f}")

            # Short Positions Performance
            st.markdown("### Short Positions Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Short Trades", metrics.get('short_trades', 0))
                st.metric("Winning Short Trades", len(trades_df[(trades_df['direction'] == 'short') & (trades_df['pnl'] > 0)]))
                st.metric("Losing Short Trades", len(trades_df[(trades_df['direction'] == 'short') & (trades_df['pnl'] < 0)]))
                st.metric("Short Win Rate", f"{metrics.get('short_win_rate', 0):.1f}%")
            with col2:
                short_trades = trades_df[trades_df['direction'] == 'short']
                short_pnl = short_trades['pnl'].sum() if not short_trades.empty else 0
                st.metric("Short Total PNL", f"{short_pnl:.2f}")
                st.metric("Short Average PNL", f"{short_trades['pnl'].mean() if not short_trades.empty else 0:.2f}")
                st.metric("Short Average Duration", format_duration_minutes(short_trades['position_duration'].mean() if not short_trades.empty else None))
                short_profit_factor = abs(short_trades[short_trades['pnl'] > 0]['pnl'].sum() / short_trades[short_trades['pnl'] < 0]['pnl'].abs().sum()) if not short_trades.empty and len(short_trades[short_trades['pnl'] < 0]) > 0 else 0
                st.metric("Short Profit Factor", f"{short_profit_factor:.2f}")

            # Plot cumulative PNL
            st.subheader("Cumulative PNL Curve")
            fig, ax = plt.subplots(figsize=(12, 6))

            # Calculate cumulative PNL for all trades
            trades_df['cum_pnl'] = trades_df['pnl'].cumsum()
            ax.plot(pd.to_datetime(trades_df['date']), trades_df['cum_pnl'], 'b-', linewidth=2, label='All Trades')

            # Calculate cumulative PNL for long trades only
            long_trades = trades_df[trades_df['direction'] == 'long'].copy()
            if not long_trades.empty:
                long_trades['cum_pnl'] = long_trades['pnl'].cumsum()
                ax.plot(pd.to_datetime(long_trades['date']), long_trades['cum_pnl'], 'g-', linewidth=2, label='Long Trades Only')

            # Calculate cumulative PNL for short trades only
            short_trades = trades_df[trades_df['direction'] == 'short'].copy()
            if not short_trades.empty:
                short_trades['cum_pnl'] = short_trades['pnl'].cumsum()
                ax.plot(pd.to_datetime(short_trades['date']), short_trades['cum_pnl'], 'r-', linewidth=2, label='Short Trades Only')

            ax.set_title('Cumulative PNL Curve')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative PNL')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            plt.close()

            # Display all trades in full size, newest first
            st.subheader("All Trade Executions")

            # Sort trades newest to oldest
            trades_df = trades_df.sort_values('date', ascending=False)

            # Create a container for all plots
            plots_container = st.container()

            with plots_container:
                for _, trade_row in trades_df.iterrows():
                    trade = trade_row.to_dict()
                    date = trade['date']

                    # Get corresponding day data
                    if isinstance(date, pd.Timestamp):
                        date = date.date()
                    day_data = daily_data[date]

                    # Recompute opening range for plotting
                    high, low, tr, expected_candles = get_opening_range_levels(
                        day_data, or_start_hour, or_start_minute, or_end_hour, or_end_minute,
                        start_hour, start_minute, end_hour, end_minute
                    )

                    # Display date and P/L as a header
                    st.markdown(f"### {date.strftime('%Y-%m-%d')} - P/L: {trade['pnl']:.2f} points - Breakout: {trade['breakout_direction']}")

                    # Plot the trade with original full-size function
                    plot_single_day(day_data, date, high, low, trade, timezone)

                    # Add some spacing between plots
                    st.write("---")
