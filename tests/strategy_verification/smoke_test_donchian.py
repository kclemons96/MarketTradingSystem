import sys
import os
import pandas as pd

# Ensure strategy folder is importable (folder name contains a space)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\\..'))
STRAT_DIR = os.path.join(ROOT, 'Strategies', 'Donchian Breakout')
sys.path.insert(0, STRAT_DIR)

from donchian import donchian_breakout


def make_sample_df():
    # simple price series with an up move then down then up
    close = [100, 101, 102, 103, 104, 103, 102, 101, 100, 101, 102, 103, 104]
    return pd.DataFrame({'close': close})


def run_smoke_test():
    df = make_sample_df()
    sig = donchian_breakout(df, lookback=3)

    assert len(sig) == len(df), 'signal length mismatch'
    vals = set(sig.dropna().astype(float).unique())
    assert vals.issubset({1.0, -1.0}), f'unexpected signal values: {vals}'

    print('SMOKE TEST PASSED: donchian_breakout produced valid signals')


if __name__ == '__main__':
    try:
        run_smoke_test()
        sys.exit(0)
    except AssertionError as e:
        print('SMOKE TEST FAILED:', e)
        sys.exit(2)
