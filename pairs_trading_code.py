# ============================================================
# Pairs Trading Strategy with Cointegration
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import requests
from io import StringIO

# Define the parameters for the pairs trading strategy
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
IN_SAMPLE_END = "2020-12-31"   
OUT_SAMPLE_START = "2021-01-01" 

COINT_PVALUE_THRESHOLD = 0.05
ENTRY_Z = 2.0
EXIT_Z  = 0.0
LOOKBACK = 60          # rolling window for z-score calculation
STOP_LOSS_Z = 3.5      # z-score threshold for stop loss
MAX_HOLDING_DAYS = 30  # maximum holding period for a trade
TRANSACTION_COST = 0  # transaction cost per trade (0.1%)

CORR_THRESHOLD = 0.8 # minimum correlation threshold for filtering pairs

# function to get S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text), header=0) 
    df = tables[0]
    print(df.columns.tolist())
    print(df.head())
    tickers = df["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers if isinstance(t, str)]
    return tickers

TICKERS = get_sp500_tickers()
print(f"gained {len(TICKERS)} tickers from S&P 500")
print(TICKERS[:10])

# Download historical price data for the tickers
def download_prices(tickers, start, end, batch_size=50):
    all_data = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(
                batch, start=start, end=end,
                auto_adjust=True,
                multi_level_index=False,
                progress=False
            )["Close"]
            all_data.append(data)
            print(f"downloaded {min(i+batch_size, len(tickers))}/{len(tickers)}")
        except Exception as e:
            print(f"batch {i} failed to download: {e}")
            continue

    prices = pd.concat(all_data, axis=1)
    prices = prices.dropna(axis=1, how="any")  # drop tickers with any missing data
    return prices

prices = download_prices(TICKERS, START_DATE, END_DATE)
print(f"cleaned price matrix shape: {prices.shape}")

log_prices = np.log(prices)
returns = prices.pct_change().dropna()

# Split the data into in-sample and out-of-sample periods
in_sample_prices = prices.loc[:IN_SAMPLE_END]
out_sample_prices = prices.loc[OUT_SAMPLE_START:]

# cointegration test function
def find_cointegrated_pairs(prices, corr_threshold=0.8, pvalue_threshold=0.05):
    """
    2.1. Find cointegrated pairs of stocks based on price data.
    2.2. Filter pairs based on correlation and cointegration p-value thresholds.
    """
    n = prices.shape[1]
    cols = prices.columns

    # use percentage change to compute correlation matrix
    ret = prices.pct_change().dropna()
    corr_matrix = ret.corr()

    pairs = []
    tested = 0
    total_possible = n * (n - 1) // 2

    # test all pairs of stocks for cointegration
    for i in range(n):
        for j in range(i+1, n):
            c = corr_matrix.iloc[i, j]
            if abs(c) < corr_threshold:
                continue
            s1 = prices.iloc[:, i]
            s2 = prices.iloc[:, j]
            try:
                _, pvalue, _ = coint(s1, s2)
                tested += 1
                if pvalue < pvalue_threshold:
                    pairs.append((cols[i], cols[j], pvalue, c))
            except Exception:
                continue

    print(f"total possible pairs: {total_possible}, pairs tested after correlation filter: {tested}")
    pairs_df = pd.DataFrame(pairs, columns=["Stock_A", "Stock_B", "pvalue", "correlation"])
    pairs_df = pairs_df.sort_values("pvalue").reset_index(drop=True)
    return pairs_df

pairs_df = find_cointegrated_pairs(
    in_sample_prices,
    corr_threshold=CORR_THRESHOLD,
    pvalue_threshold=COINT_PVALUE_THRESHOLD
)
print(f"found {len(pairs_df)} candidate pairs")
print(pairs_df.head(10))

# Compute the spread and z-score for a given pair of stocks
def compute_spread(prices, stock_a, stock_b, lookback=60):
    y = prices[stock_a]
    x = prices[stock_b]
    hedge_ratios = []
    spreads = []
    dates = prices.index

    for i in range(len(prices)):
        if i < lookback:
            hedge_ratios.append(np.nan)
            spreads.append(np.nan)
            continue
        y_win = y.iloc[i-lookback:i]
        x_win = x.iloc[i-lookback:i]
        X = add_constant(x_win)
        model = OLS(y_win, X).fit()
        beta = model.params.iloc[1]
        alpha = model.params.iloc[0]
        spread = y.iloc[i] - (alpha + beta * x.iloc[i])
        hedge_ratios.append(beta)
        spreads.append(spread)

    spread_series = pd.Series(spreads, index=dates)
    hedge_series = pd.Series(hedge_ratios, index=dates)

    mean = spread_series.rolling(lookback).mean()
    std = spread_series.rolling(lookback).std()
    zscore = (spread_series - mean) / std
    zscore = zscore.replace([np.inf, -np.inf], np.nan)
    zscore = zscore.clip(-10, 10) 

    return spread_series, zscore, hedge_series


# Backtest the pairs trading strategy
def backtest_pair(prices, stock_a, stock_b, lookback=60,
                  entry_z=2.0, exit_z=0.0, stop_loss_z=3.0,
                  max_holding=20, tc=0.000):

    spread, zscore, hedge = compute_spread(prices, stock_a, stock_b, lookback)

    # zscore clipping to avoid extreme values
    zscore = zscore.replace([np.inf, -np.inf], np.nan).clip(-10, 10)

    dates = prices.index
    ret_a = prices[stock_a].pct_change()
    ret_b = prices[stock_b].pct_change()

    position = 0
    entry_date = None
    entry_idx = None
    trades = []
    equity = [1.0]
    daily_returns = []

    for i in range(1, len(dates)):
        z = zscore.iloc[i]
        if np.isnan(z) or np.isnan(ret_a.iloc[i]) or np.isnan(ret_b.iloc[i]):
            daily_returns.append(0)
            equity.append(equity[-1])
            continue

        ret = 0.0

        # calculate daily return based on the current position
        if position != 0:
            ret = position * (ret_a.iloc[i] - hedge.iloc[i] * ret_b.iloc[i])

        # judge whether to close the position
        if position != 0:
            holding_days = i - entry_idx
            exit_condition = (
                (position == 1 and z >= exit_z) or
                (position == -1 and z <= exit_z) or
                (abs(z) > stop_loss_z) or
                (holding_days > max_holding)
            )
            if exit_condition:
                ret -= tc * 2   # cost for closing the position
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": dates[i],
                    "position": position,
                    "entry_z": zscore.iloc[entry_idx],
                    "exit_z": z,
                    "holding_days": holding_days,
                    "return": equity[-1] / equity[entry_idx] - 1
                })
                position = 0
                entry_date = None
                entry_idx = None

        # judge whether to open a new position
        if position == 0:
            if z > entry_z:
                position = -1
                entry_date = dates[i]
                entry_idx = i
                ret -= tc   # cost for opening the position
            elif z < -entry_z:
                position = 1
                entry_date = dates[i]
                entry_idx = i
                ret -= tc

        daily_returns.append(ret)
        equity.append(equity[-1] * (1 + ret))

    equity_curve = pd.Series(equity[1:], index=dates[1:])
    trades_df = pd.DataFrame(trades)
    return equity_curve, trades_df

# Performance metrics calculation
def performance_metrics(equity_curve, trades_df, freq=252):
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return {k: np.nan for k in [
            "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
            "Max Drawdown", "Win Rate", "Average Trade Return",
            "Turnover", "Number of Trades"
        ]}

    ann_return = (equity_curve.iloc[-1]) ** (freq / len(equity_curve)) - 1
    ann_vol = returns.std() * np.sqrt(freq)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()

    # Calculate win rate and average trade return
    n_trades = len(trades_df) if trades_df is not None else 0

    if n_trades > 0 and "return" in trades_df.columns:
        win_rate = (trades_df["return"] > 0).mean()
        avg_trade = trades_df["return"].mean()
    else:
        win_rate = np.nan
        avg_trade = np.nan

    turnover = n_trades / len(equity_curve) * freq if len(equity_curve) > 0 else np.nan

    return {
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Average Trade Return": avg_trade,
        "Turnover": turnover,
        "Number of Trades": n_trades
    }


# Backtest the top N pairs and create a portfolio equity curve
def portfolio_backtest(prices, pairs_df, top_n=10, **kwargs):
    """
    backtest the top N cointegrated pairs and combine their equity curves into a portfolio.
    """
    all_equity = []
    all_trades = []

    for i in range(min(top_n, len(pairs_df))):
        a = pairs_df.iloc[i]["Stock_A"]
        b = pairs_df.iloc[i]["Stock_B"]

        # ensure both stocks are present in the price data
        if a not in prices.columns or b not in prices.columns:
            print(f"skip pairing {a} vs {b}: stock not in price data")
            continue

        eq, tr = backtest_pair(prices, a, b, **kwargs)
        print(f"pair {i+1}: {a} vs {b}, trades {len(tr)}, final equity {eq.iloc[-1]:.4f}")

        all_equity.append(eq)
        if len(tr) > 0:
            tr = tr.copy()
            tr["pair"] = f"{a}-{b}"
            all_trades.append(tr)

    if not all_equity:
        print("No valid pairs to backtest.")
        return pd.Series(dtype=float), pd.DataFrame()

    daily_returns = pd.concat(
        [eq.pct_change() for eq in all_equity], axis=1
    ).fillna(0)
    portfolio_equity = (1 + daily_returns.mean(axis=1)).cumprod()

    # Combine all trades into a single DataFrame
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return portfolio_equity, trades_df

# In-sample portfolio backtest
portfolio_eq, portfolio_trades = portfolio_backtest(
    in_sample_prices, pairs_df, top_n=10,
    lookback=LOOKBACK, entry_z=ENTRY_Z, exit_z=EXIT_Z,
    stop_loss_z=STOP_LOSS_Z, max_holding=MAX_HOLDING_DAYS,
    tc=TRANSACTION_COST
)

print("\nIn-Sample Portfolio Performance:")
print(performance_metrics(portfolio_eq, portfolio_trades))

# Plot the in-sample portfolio equity curve
plt.figure(figsize=(12, 5))
plt.plot(portfolio_eq, label="Portfolio Equity")
plt.title("In-Sample Portfolio Cumulative Return")
plt.legend()
plt.show()

# Out-of-sample backtest for the top N pairs
oos_equity_list = []
for i in range(min(10, len(pairs_df))):
    a = pairs_df.iloc[i]["Stock_A"]
    b = pairs_df.iloc[i]["Stock_B"]
    # ensure both stocks are present in the out-of-sample prices
    if a not in out_sample_prices.columns or b not in out_sample_prices.columns:
        continue
    eq, tr = backtest_pair(
        out_sample_prices, a, b,
        lookback=LOOKBACK, entry_z=ENTRY_Z, exit_z=EXIT_Z,
        stop_loss_z=STOP_LOSS_Z, max_holding=MAX_HOLDING_DAYS,
        tc=TRANSACTION_COST
    )
    oos_equity_list.append(eq)

if oos_equity_list:
    oos_eq, oos_trades = portfolio_backtest(
    out_sample_prices, pairs_df, top_n=10,
    lookback=LOOKBACK, entry_z=ENTRY_Z, exit_z=EXIT_Z,
    stop_loss_z=STOP_LOSS_Z, max_holding=MAX_HOLDING_DAYS,
    tc=TRANSACTION_COST
)

print("\nOut-of-Sample Portfolio Performance:")
print(performance_metrics(oos_eq, oos_trades))

plt.figure(figsize=(12, 5))
plt.plot(oos_eq, label="Out-of-Sample Equity")
plt.title("Out-of-Sample Portfolio Cumulative Return")
plt.legend()
plt.show()

# ============================================================
# Transaction cost sensitivity analysis
# ============================================================
cost_list = [0.0, 0.0002, 0.0005, 0.001, 0.002, 0.005]
cost_results = []

for tc in cost_list:
    eq, tr = portfolio_backtest(
        in_sample_prices, pairs_df, top_n=10,
        lookback=LOOKBACK, entry_z=ENTRY_Z, exit_z=EXIT_Z,
        stop_loss_z=STOP_LOSS_Z, max_holding=MAX_HOLDING_DAYS,
        tc=tc
    )
    m = performance_metrics(eq, tr)
    cost_results.append({
        "tc": tc,
        "sharpe": m["Sharpe Ratio"],
        "ann_return": m["Annualized Return"],
        "n_trades": m["Number of Trades"],
        "turnover": m["Turnover"]
    })
    print(f"tc={tc:.4f} -> Sharpe={m['Sharpe Ratio']:.3f}, Return={m['Annualized Return']:.4f}, Trades={m['Number of Trades']}")

cost_df = pd.DataFrame(cost_results)
print(cost_df)

# Plot the sensitivity of Sharpe ratio and annualized return to transaction costs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(cost_df["tc"] * 10000, cost_df["sharpe"], marker="o", color="steelblue")
axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
axes[0].set_xlabel("Transaction Cost (bps, one-way)")
axes[0].set_ylabel("Sharpe Ratio")
axes[0].set_title("Sharpe vs Transaction Cost (In-Sample)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(cost_df["tc"] * 10000, cost_df["ann_return"] * 100, marker="o", color="darkorange")
axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
axes[1].set_xlabel("Transaction Cost (bps, one-way)")
axes[1].set_ylabel("Annualized Return (%)")
axes[1].set_title("Annualized Return vs Transaction Cost (In-Sample)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cost_sensitivity.png", dpi=150)
plt.show()

# ============================================================
# Entry Z threshold sensitivity analysis
# ============================================================
entry_z_list = [1.0, 1.5, 2.0, 2.5, 3.0]
entry_results = []

for ez in entry_z_list:
    eq, tr = portfolio_backtest(
        in_sample_prices, pairs_df, top_n=10,
        lookback=LOOKBACK, entry_z=ez, exit_z=EXIT_Z,
        stop_loss_z=STOP_LOSS_Z, max_holding=MAX_HOLDING_DAYS,
        tc=TRANSACTION_COST
    )
    m = performance_metrics(eq, tr)
    entry_results.append({
        "entry_z": ez,
        "sharpe": m["Sharpe Ratio"],
        "ann_return": m["Annualized Return"],
        "n_trades": m["Number of Trades"],
        "turnover": m["Turnover"]
    })
    print(f"entry_z={ez} -> Sharpe={m['Sharpe Ratio']:.3f}, Return={m['Annualized Return']:.4f}, Trades={m['Number of Trades']}")

entry_df = pd.DataFrame(entry_results)
print(entry_df)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(entry_df["entry_z"], entry_df["sharpe"], marker="o", color="steelblue")
axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
axes[0].set_xlabel("Entry Z threshold")
axes[0].set_ylabel("Sharpe Ratio")
axes[0].set_title("Sharpe vs Entry Z (In-Sample)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(entry_df["entry_z"], entry_df["ann_return"] * 100, marker="o", color="darkorange")
axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
axes[1].set_xlabel("Entry Z threshold")
axes[1].set_ylabel("Annualized Return (%)")
axes[1].set_title("Return vs Entry Z (In-Sample)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(entry_df["entry_z"], entry_df["n_trades"], marker="o", color="seagreen")
axes[2].set_xlabel("Entry Z threshold")
axes[2].set_ylabel("Number of Trades")
axes[2].set_title("Trades vs Entry Z (In-Sample)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("entry_z_sensitivity.png", dpi=150)
plt.show()

# ============================================================
# Compare In-Sample vs Out-of-Sample Equity Curves
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(portfolio_eq, label="In-Sample (tc=0)", color="steelblue")
ax.plot(oos_eq, label="Out-of-Sample (tc=0)", color="darkorange")
ax.set_title("In-Sample vs Out-of-Sample Equity Curve (tc=0)")
ax.set_ylabel("Cumulative Return")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("is_vs_oos_equity.png", dpi=150)
plt.show()

# ============================================================
# Visualize Spread and Z-score for a specific pair
# ============================================================
spread, zscore, hedge = compute_spread(in_sample_prices, "GL", "JPM", LOOKBACK)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(spread, color="purple")
axes[0].set_title("Spread: GL vs JPM")
axes[0].grid(True, alpha=0.3)

axes[1].plot(zscore, color="orange")
axes[1].axhline(ENTRY_Z, color="red", linestyle="--")
axes[1].axhline(-ENTRY_Z, color="green", linestyle="--")
axes[1].axhline(EXIT_Z, color="black", linestyle="--")
axes[1].set_title("Z-score with Entry/Exit Thresholds")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("spread_zscore_example.png", dpi=150)
plt.show()

# ============================================================
# Analyze the contribution of top 10% trades to total profit
# ============================================================
if len(portfolio_trades) > 0:
    trades_sorted = portfolio_trades.sort_values("return", ascending=False)
    top_10pct = int(len(trades_sorted) * 0.1)
    top_profit = trades_sorted.head(top_10pct)["return"].sum()
    total_profit = trades_sorted["return"].sum()
    print(f"top 10% profit: {top_profit:.4f}")
    print(f"total profit: {total_profit:.4f}")
    print(f"percentage: {top_profit / total_profit * 100:.2f}%")

