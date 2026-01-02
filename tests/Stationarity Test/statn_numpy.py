#!/usr/bin/env python3
"""STATN - NumPy-accelerated stationarity test

Usage: python statn_numpy.py Lookback Fractile Version Filename
E.G. python statn_numpy.py 100 0.5 1 path/to/marketdata.txt
"""

from __future__ import annotations
import sys
import math
from typing import List
import numpy as np


def find_slope(lookback: int, x: np.ndarray, idx: int) -> float:
    start = idx - lookback + 1
    window = x[start : start + lookback]
    coefs = np.arange(lookback, dtype=float) - 0.5 * (lookback - 1)
    denom = np.dot(coefs, coefs)
    slope = float(np.dot(coefs, window))
    return slope / denom


def atr(lookback: int, high: np.ndarray, low: np.ndarray, close: np.ndarray, idx: int) -> float:
    start = idx - lookback + 1
    hi = high[start : start + lookback]
    lo = low[start : start + lookback]
    cl = close[start : start + lookback]

    # TR = max(hi-lo, hi - prev_close, prev_close - lo) per bar
    tr = hi - lo
    # for elements i>=1, consider hi[i]-cl[i-1] and cl[i-1]-lo[i]
    if lookback > 1:
        prev = cl[:-1]
        # compute comparisons for positions 1..end
        comp1 = hi[1:] - prev
        comp2 = prev - lo[1:]
        # combine into tr[1:]
        tr[1:] = np.maximum(tr[1:], comp1)
        tr[1:] = np.maximum(tr[1:], comp2)

    return float(np.mean(tr))


def gap_analyze(n: int, x: np.ndarray, thresh: float, ngaps: int, gap_size: List[int], gap_count: List[int]) -> None:
    for i in range(ngaps):
        gap_count[i] = 0

    count = 1
    above_below = 1 if x[0] >= thresh else 0

    for i in range(1, n + 1):
        if i == n:
            new_above_below = 1 - above_below
        else:
            new_above_below = 1 if x[i] >= thresh else 0

        if new_above_below == above_below:
            count += 1
        else:
            for j in range(ngaps - 1):
                if count <= gap_size[j]:
                    break
            gap_count[j] += 1
            count = 1
            above_below = new_above_below


def usage_and_exit() -> None:
    print("\nUsage: STATN_NUMPY.py Lookback Fractile Version Filename")
    sys.exit(1)


def parse_market_file(filename: str):
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []

    with open(filename, 'rt') as fp:
        for lineno, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            if len(line) < 8:
                raise ValueError(f"Invalid date reading line {lineno} of file {filename}")
            date_str = line[:8]
            if not date_str.isdigit():
                raise ValueError(f"Invalid date reading line {lineno} of file {filename}")
            full_date = int(date_str)
            rest = line[8:]
            parts = [p for p in rest.replace(',', ' ').split() if p]
            if len(parts) < 4:
                raise ValueError(f"Invalid price line {lineno} in file {filename}")
            o = float(parts[0]); h = float(parts[1]); l = float(parts[2]); c = float(parts[3])
            if o > 0.0: o = math.log(o)
            if h > 0.0: h = math.log(h)
            if l > 0.0: l = math.log(l)
            if c > 0.0: c = math.log(c)
            if l > o or l > c or h < o or h < c:
                raise ValueError(f"Invalid open/high/low/close reading line {lineno} of file {filename}")
            dates.append(full_date)
            opens.append(o); highs.append(h); lows.append(l); closes.append(c)

    return np.array(dates, dtype=int), np.array(opens, dtype=float), np.array(highs, dtype=float), np.array(lows, dtype=float), np.array(closes, dtype=float)


def main(argv: List[str]) -> int:
    if len(argv) != 5:
        usage_and_exit()
    lookback = int(argv[1])
    fractile = float(argv[2])
    version = int(argv[3])
    filename = argv[4]

    if lookback < 2:
        print("Lookback must be at least 2")
        return 1

    if version == 0:
        full_lookback = lookback
    elif version == 1:
        full_lookback = 2 * lookback
    elif version > 1:
        full_lookback = version * lookback
    else:
        print("Version cannot be negative")
        return 1

    try:
        date, open_p, high, low, close = parse_market_file(filename)
    except Exception as e:
        print(e)
        return 1

    nprices = len(date)
    print(f"Market price history read ({nprices} lines)")
    print(f"\n\nIndicator version {version}")

    NGAPS = 11
    ngaps = NGAPS
    gap_size = [0] * (ngaps - 1)
    gap_count = [0] * ngaps
    k = 1
    for i in range(ngaps - 1):
        gap_size[i] = k
        k *= 2

    nind = nprices - full_lookback + 1
    if nind <= 0:
        print("Not enough data for indicators")
        return 1

    trend = np.empty(2 * nind, dtype=float)
    trend_sorted = trend[nind:]

    trend_min = 1e60
    trend_max = -1e60
    for i in range(nind):
        kk = full_lookback - 1 + i
        if version == 0:
            trend[i] = find_slope(lookback, close, kk)
        elif version == 1:
            trend[i] = find_slope(lookback, close, kk) - find_slope(lookback, close, kk - lookback)
        else:
            trend[i] = find_slope(lookback, close, kk) - find_slope(full_lookback, close, kk)
        trend_sorted[i] = trend[i]
        if trend[i] < trend_min:
            trend_min = trend[i]
        if trend[i] > trend_max:
            trend_max = trend[i]

    trend_sorted = np.sort(trend_sorted[:nind])
    idx = int(fractile * (nind + 1)) - 1
    if idx < 0:
        idx = 0
    trend_quantile = float(trend_sorted[idx])

    print(f"\n\nTrend  min={trend_min:.4f}  max={trend_max:.4f}  {fractile:.3f} quantile={trend_quantile:.4f}")

    print(f"\n\nGap analysis for trend with lookback={lookback}")
    print("  Size   Count")

    gap_analyze(nind, trend[:nind], trend_quantile, ngaps, gap_size, gap_count)
    for i in range(ngaps):
        if i < ngaps - 1:
            print(f" {gap_size[i]:5d} {gap_count[i]:7d}")
        else:
            print(f">{gap_size[ngaps-2]:5d} {gap_count[i]:7d}")

    volatility = np.empty(2 * nind, dtype=float)
    volatility_sorted = volatility[nind:]

    volatility_min = 1e60
    volatility_max = -1e60
    for i in range(nind):
        kk = full_lookback - 1 + i
        if version == 0:
            volatility[i] = atr(lookback, high, low, close, kk)
        elif version == 1:
            volatility[i] = atr(lookback, high, low, close, kk) - atr(lookback, high, low, close, kk - lookback)
        else:
            volatility[i] = atr(lookback, high, low, close, kk) - atr(full_lookback, high, low, close, kk)
        volatility_sorted[i] = volatility[i]
        if volatility[i] < volatility_min:
            volatility_min = volatility[i]
        if volatility[i] > volatility_max:
            volatility_max = volatility[i]

    volatility_sorted = np.sort(volatility_sorted[:nind])
    idx = int(fractile * (nind + 1)) - 1
    if idx < 0:
        idx = 0
    volatility_quantile = float(volatility_sorted[idx])

    print(f"\n\nVolatility  min={volatility_min:.4f}  max={volatility_max:.4f}  {fractile:.3f} quantile={volatility_quantile:.4f}")

    print(f"\n\nGap analysis for volatility with lookback={lookback}")
    print("  Size   Count")
    gap_analyze(nind, volatility[:nind], volatility_quantile, ngaps, gap_size, gap_count)
    for i in range(ngaps):
        if i < ngaps - 1:
            print(f" {gap_size[i]:5d} {gap_count[i]:7d}")
        else:
            print(f">{gap_size[ngaps-2]:5d} {gap_count[i]:7d}")

    input("\n\nPress Enter to continue...")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
