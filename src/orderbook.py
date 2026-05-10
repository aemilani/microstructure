import numpy as np
import pandas as pd


def random_orderbook(
    mid: float = 100.0,               # central price around which we build the book
    tick: float = 0.01,               # price granularity
    spread_ticks: int = 2,            # best‑bid/ask gap in ticks
    depth_levels: int = 20,           # levels per side
    base_size: float = 1_000.0,       # expected size at the best bid/ask
    depth_decay: float = 0.15,        # how quickly size grows deeper in book
    sigma_vol: float = 0.5,           # randomness in size (log‑normal std‑dev)
    rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """
    Build a one‑shot synthetic order book with realistic features:

    • tick‑aligned prices, symmetric around 'mid'
    • quoted spread = spread_ticks * tick
    • depth increases (on average) as we move away from the top
    • log‑normal noise to avoid perfectly smooth shapes
    """
    rng = rng or np.random.default_rng()
    half_spread = (spread_ticks * tick) / 2

    # --- price ladders -------------------------------------------------------
    ask_px = mid + half_spread + tick * np.arange(depth_levels)
    bid_px = mid - half_spread - tick * np.arange(depth_levels)

    # --- sizes: grow with depth + randomness ---------------------------------
    vol_multiplier = rng.lognormal(mean=0.0, sigma=sigma_vol, size=depth_levels)
    depth_factor = np.exp(depth_decay * np.arange(depth_levels))
    ask_sz = base_size * depth_factor * vol_multiplier
    bid_sz = base_size * depth_factor * vol_multiplier      # symmetric book

    asks = pd.DataFrame({"side": "ask", "price": ask_px, "size": ask_sz})
    bids = pd.DataFrame({"side": "bid", "price": bid_px, "size": bid_sz})

    return pd.concat([bids, asks], ignore_index=True)
