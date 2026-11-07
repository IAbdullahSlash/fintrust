"""Download the Bitstamp BTC/USD 1-minute bars used in the paper (Section IX, BTC_PREREGISTRATION.md).

Source: github.com/ff137/bitstamp-btcusd-minute-data (Bitstamp public API, CC BY 4.0).
The archive keeps growing, so rows after the paper's download time (2026-11-05 03:45 UTC) are dropped.
The trimmed file should match the SHA-256 recorded in the pre-registration.

usage: python data/download_btc.py            # writes data/raw/btcusd_bitstamp_1min.csv
"""
import hashlib, sys, urllib.request
from pathlib import Path

URL = ('https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/main/'
       'data/updates/btcusd_bitstamp_1min_latest.csv')
LAST_TS = 1793850300          # 2026-11-05 03:45:00 UTC, last row of the paper's copy
SHA256 = '72b11143c6ddba74dd2604167bd1232b34e5bcbe5b44ed2af1f2b59247d24ddf'
OUT = Path(__file__).resolve().parent / 'raw' / 'btcusd_bitstamp_1min.csv'

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print('downloading', URL)
    lines = urllib.request.urlopen(URL, timeout=600).read().decode().splitlines(keepends=True)
    keep = [lines[0]] + [l for l in lines[1:] if l.strip() and int(l.split(',', 1)[0]) <= LAST_TS]
    data = ''.join(keep).encode()
    OUT.write_bytes(data)
    h = hashlib.sha256(data).hexdigest()
    print(f'wrote {OUT} ({len(keep) - 1} rows), sha256 {h}')
    if h != SHA256:
        print('WARNING: checksum differs from the paper\'s copy (the upstream archive may have been revised). '
              'Results can differ slightly.', file=sys.stderr)

if __name__ == '__main__':
    main()
