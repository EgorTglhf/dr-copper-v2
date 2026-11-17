import numpy as np
import pandas as pd


def long_cash_targets(predictions, rule="daily"):
    up = np.r_[False, np.asarray(predictions)[:-1] == 1]
    if rule == "daily":
        targets = up.astype(np.int8)
    else:
        targets = np.zeros(len(up), dtype=np.int8)
        long = False
        for row in range(1, len(up)):
            if not up[row] and not up[row - 1]:
                long = False
            elif up[row] and up[row - 1]:
                long = True
            targets[row] = long
    targets[-1] = 0
    return targets


def backtest_long_cash(
    frame,
    initial_cash=100_000.0,
    commission_per_share=0.005,
    slippage_bps=5.0,
):
    prices = frame["close"].to_numpy()
    targets = frame["target_long"].to_numpy()
    cash_returns = frame["cash_return"].to_numpy()

    slippage_rate = slippage_bps / 10_000
    cash = float(initial_cash)
    shares = 0
    orders = np.zeros(len(frame), dtype=np.int64)
    holdings = np.zeros(len(frame), dtype=np.int64)
    balances = np.zeros(len(frame))
    fees = np.zeros(len(frame))
    fills = np.full(len(frame), np.nan)
    slippage = np.zeros(len(frame))
    for row, (price, target) in enumerate(zip(prices, targets, strict=True)):
        cash += cash * cash_returns[row]
        if target == 1 and shares == 0:
            fill_price = price * (1 + slippage_rate)
            quantity = int(np.floor(cash / (fill_price + commission_per_share)))
            if (
                quantity > 0
                and quantity * fill_price + quantity * commission_per_share > cash
            ):
                quantity -= 1
            if quantity > 0:
                orders[row] = quantity
                fills[row] = fill_price
                fees[row] = quantity * commission_per_share
                slippage[row] = quantity * (fill_price - price)
                cash -= quantity * fill_price + fees[row]
                shares = quantity
        elif target == 0 and shares > 0:
            orders[row] = -shares
            fills[row] = price * (1 - slippage_rate)
            fees[row] = shares * commission_per_share
            slippage[row] = shares * (price - fills[row])
            cash += shares * fills[row] - fees[row]
            shares = 0
        holdings[row] = shares
        balances[row] = cash

    share_value = holdings * prices
    equity = balances + share_value
    return pd.DataFrame(
        {
            "close": prices,
            "target_long": targets,
            "order_shares": orders,
            "shares": holdings,
            "cash": balances,
            "fill_price": fills,
            "commission": fees,
            "slippage": slippage,
            "share_value": share_value,
            "equity": equity,
            "net_return": equity / np.r_[initial_cash, equity[:-1]] - 1,
        },
        index=frame.index,
    )


def backtest_summary(results, initial_cash=100_000.0):
    """Return a formatted comparison table for notebook display."""
    return pd.DataFrame.from_dict(
        {
            name: {
                "Net return": f"{result['equity'].iloc[-1] / initial_cash - 1:.2%}",
                "Max drawdown": f"{(result['equity'] / result['equity'].cummax() - 1).min():.2%}",
                "Fills": f"{(result['order_shares'] != 0).sum():.0f}",
                "Commission (USD)": f"{result['commission'].sum():,.2f}",
                "Slippage (USD)": f"{result['slippage'].sum():,.2f}",
                "Final equity (USD)": f"{result['equity'].iloc[-1]:,.2f}",
            }
            for name, result in results.items()
        },
        orient="index",
    ).rename_axis("Strategy")
