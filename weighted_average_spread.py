import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.orderbook import random_orderbook


def _avg_exec_price(book: pd.DataFrame, side: str, qty: float) -> float:
    """
    Walk the book and compute the volume‑weighted average execution
    price for either a buy ('ask' side) or sell ('bid' side) order of size 'qty'.
    """
    side_book = book.query("side == @side").copy()

    # For buys, we start at BEST ASK (lowest); for sells at BEST BID (highest)
    side_book = side_book.sort_values(
        "price", ascending=(side == "ask")  # True→asks low→high ; False→bids high→low
    ).reset_index(drop=True)

    cum = side_book["size"].cumsum()
    if qty > cum.iat[-1]:
        raise ValueError("Requested quantity exceeds book depth.")

    take_full = side_book.loc[cum < qty, ["price", "size"]]
    take_part = side_book.loc[cum >= qty].iloc[0]

    filled_qty = take_full["size"].sum()
    remaining = qty - filled_qty

    vwap_numer = (take_full["price"] * take_full["size"]).sum()
    vwap_numer += take_part["price"] * remaining

    return vwap_numer / qty


def wa_spread(book: pd.DataFrame, q: float) -> float:
    """
    Weighted‑average bid‑ask spread divided by mid-price.
    """
    a_q = _avg_exec_price(book, "ask", q)   # cost to BUY  q
    b_q = _avg_exec_price(book, "bid", q)   # proceeds to SELL q
    mid = (book.query("side == 'ask'")["price"].min() +
           book.query("side == 'bid'")["price"].max()) / 2
    return (a_q - b_q) / mid                # relative form (dimensionless)


if __name__ == "__main__":
    orderbook = random_orderbook()

    # Prepare data for Plot 1 (Order Book Depth)
    asks = orderbook[orderbook["side"] == "ask"].sort_values("price", ascending=True)
    bids = orderbook[orderbook["side"] == "bid"].sort_values("price", ascending=False)
    asks['cum_size'] = asks['size'].cumsum()
    bids['cum_size'] = bids['size'].cumsum()

    # Prepare data for Plot 2 (Weighted-Average Spread)
    max_size = min(asks['size'].sum(), bids['size'].sum()) * 0.95
    q_values = np.linspace(100, max_size, 100)
    spreads_bps = [wa_spread(orderbook, q) * 10000 for q in q_values]  # Convert to basis points

    # Create a figure with 2 subplots (stacked vertically)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=False)

    # --- Top Plot: Cumulative Order Book Depth ---
    ax1.step(bids['price'], bids['cum_size'], color='green', label='Bids (Buy)', where='post', linewidth=2)
    ax1.step(asks['price'], asks['cum_size'], color='red', label='Asks (Sell)', where='pre', linewidth=2)
    ax1.fill_between(bids['price'], bids['cum_size'], color='green', alpha=0.2, step='post')
    ax1.fill_between(asks['price'], asks['cum_size'], color='red', alpha=0.2, step='pre')

    ax1.set_title("Cumulative Order Book Depth", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Price ($)", fontsize=12)
    ax1.set_ylabel("Cumulative Size", fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # --- Bottom Plot: Weighted-Average Spread vs. Size ---
    ax2.plot(q_values, spreads_bps, color='purple', linewidth=2.5)

    ax2.set_title("Weighted-Average Spread vs. Order Size ($q$)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Order Size ($q$)", fontsize=12)
    ax2.set_ylabel("WA-Spread (Basis Points)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()