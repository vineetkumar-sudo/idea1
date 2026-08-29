#!/usr/bin/env python
"""Download CADC and CADC-clear without pulling archives you don't need.

The two halves ship in different formats, so they need different tricks:

  CADC        labeled.zip      -> remotezip: HTTP range requests for just the members
                                  we want (the server returns 206 Partial Content).
  CADC-clear  labeled.tar.zst  -> pipe the stream through `zstd -dc | tar -x <globs>`,
                                  so the archive never lands on disk. For --parts gps a
                                  short prefix of the stream suffices, since novatel sits
                                  near the front of the tar.

Sizes (measured 2026-08-29, ~90s/seq for CADC and ~34s/seq for CADC-clear):

    mode     transferred   on disk    contents
    gps        ~4 GiB        61 MB    per-frame GPS only
    lidar    ~110 GiB      ~10 GiB    LiDAR + GPS + calib
    full     ~195 GiB     ~195 GiB    adds all 8 cameras

The asymmetry in `lidar` is the archive format, not the payload. CADC's .zip supports
range requests, so remotezip pulls only the LiDAR members (~5 GiB total). CADC-clear's
.tar.zst is a single zstd stream with no random access, so all 105.6 GiB must flow past
even though 94% is discarded unread. Disk is what selective extraction saves there.

Layout written (matches what pcdet's CadcDataset expects):
    --parts gps            data/gps/<cadc|cadc-clear>/<date>/<seq>/{timestamps.txt,data/}
    --parts lidar|full     data/<cadc|cadc-clear>/<date>/<seq>/labeled/...
                           data/<cadc|cadc-clear>/<date>/calib/

Resumable: sequences already complete on disk are skipped.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HALVES = {
    'cadc':       ('https://wiselab.uwaterloo.ca/cadc/',       'labeled.zip',      'calib.zip'),
    'cadc-clear': ('https://wiselab.uwaterloo.ca/cadc-clear/', 'labeled.tar.zst',  'calib.tar.zst'),
}
# zip member substrings / tar globs per --parts mode
MEMBERS = {'gps': ['/novatel/'], 'lidar': ['/novatel/', '/lidar_points/'], 'full': None}
GLOBS = {'gps': ['labeled/novatel/*'],
         'lidar': ['labeled/novatel/*', 'labeled/lidar_points/*'],
         'full': None}
GPS_PREFIXES_MB = [40, 128, 384, None]   # None = stream the whole archive
TIMEOUT = 600


def listdir(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        html = r.read().decode('utf8', 'ignore')
    return [h for h in re.findall(r'href="([^"]+)"', html) if not h.startswith(('?', '/'))]


def sequences(base):
    out = []
    for date in (d for d in listdir(base) if d.endswith('/')):
        for seq in (s for s in listdir(base + date) if s.endswith('/')):
            if seq.strip('/') != 'calib':          # calib is not a sequence
                out.append((date.strip('/'), seq.strip('/')))
    return out


def _count_ok(ts, data, ext):
    if not ts.exists() or not data.is_dir():
        return False
    n = len([ln for ln in ts.read_text().splitlines() if ln.strip()])
    return n > 0 and len(list(data.glob(f'*{ext}'))) == n


def complete(dest, parts):
    """Frame count under data/ must match timestamps.txt."""
    if parts == 'gps':
        return _count_ok(dest / 'timestamps.txt', dest / 'data', '.txt')
    lp = dest / 'labeled' / 'lidar_points'
    if not _count_ok(lp / 'timestamps.txt', lp / 'data', '.bin'):
        return False
    if parts == 'full':
        return (dest / 'labeled' / 'image_00' / 'data').is_dir()
    return True


def _place(src_root, dest, parts):
    """Move freshly extracted `labeled/...` into its final home."""
    if parts == 'gps':
        got = src_root / 'labeled' / 'novatel'
        if not got.is_dir():
            return False
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        got.rename(dest)
        return True
    got = src_root / 'labeled'
    if not got.is_dir():
        return False
    target = dest / 'labeled'
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    got.rename(target)
    return True


def fetch_zip(url, dest, parts):
    from remotezip import RemoteZip
    tmp = Path(tempfile.mkdtemp(prefix='_tmp_', dir=dest.parent))
    try:
        if parts == 'full':
            arc = tmp / 'a.zip'
            subprocess.run(['curl', '-s', '--max-time', str(TIMEOUT), '-o', str(arc), url], check=True)
            with zipfile.ZipFile(arc) as z:
                z.extractall(tmp)
            arc.unlink()
        else:
            with RemoteZip(url) as z:
                want = [n for n in z.namelist() if any(m in n for m in MEMBERS[parts])]
                if not want:
                    return False
                z.extractall(tmp, members=want)
        _place(tmp, dest, parts)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return complete(dest, parts)


def fetch_tar_zst(url, dest, parts):
    globs = GLOBS[parts]
    sel = ' '.join(f"'{g}'" for g in globs) if globs else ''
    wild = '--wildcards ' if globs else ''
    prefixes = GPS_PREFIXES_MB if parts == 'gps' else [None]
    for mb in prefixes:
        tmp = Path(tempfile.mkdtemp(prefix='_tmp_', dir=dest.parent))
        try:
            rng = f'-r 0-{mb * 1048576 - 1} ' if mb else ''
            subprocess.run(
                f"curl -s --max-time {TIMEOUT} {rng}'{url}' "
                f"| zstd -dc 2>/dev/null | tar -x {wild}{sel} 2>/dev/null",
                shell=True, cwd=tmp, executable='/bin/bash',
            )
            if _place(tmp, dest, parts) and complete(dest, parts):
                return True
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return complete(dest, parts)


def fetch_calib(half, base, calib_arc, date, parts):
    if parts == 'gps':
        return 'skip'
    out = ROOT / 'data' / half / date / 'calib'
    if out.is_dir() and any(out.glob('*.yaml')):
        return 'skip'
    out.mkdir(parents=True, exist_ok=True)
    url = f'{base}{date}/{calib_arc}'
    tmp = Path(tempfile.mkdtemp(prefix='_tmp_', dir=out.parent))
    try:
        if calib_arc.endswith('.zip'):
            arc = tmp / 'c.zip'
            subprocess.run(['curl', '-s', '--max-time', '300', '-o', str(arc), url], check=True)
            with zipfile.ZipFile(arc) as z:
                z.extractall(tmp)
        else:
            subprocess.run(f"curl -s --max-time 300 '{url}' | zstd -dc | tar -x",
                           shell=True, cwd=tmp, executable='/bin/bash')
        for y in tmp.rglob('*.yaml'):
            shutil.copy2(y, out / y.name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 'ok' if any(out.glob('*.yaml')) else 'INCOMPLETE'


def one(half, base, archive, date, seq, parts):
    dest = (ROOT / 'data' / 'gps' / half / date / seq if parts == 'gps'
            else ROOT / 'data' / half / date / seq)
    if complete(dest, parts):
        return half, date, seq, 'skip'
    dest.mkdir(parents=True, exist_ok=True)
    url = f'{base}{date}/{seq}/{archive}'
    try:
        fn = fetch_zip if archive.endswith('.zip') else fetch_tar_zst
        ok = fn(url, dest, parts)
    except Exception as e:
        return half, date, seq, f'ERR {type(e).__name__}: {str(e)[:60]}'
    return half, date, seq, 'ok' if ok else 'INCOMPLETE'


def verify(parts, halves):
    bad_total = 0
    for half in halves:
        root = ROOT / ('data/gps/' + half if parts == 'gps' else 'data/' + half)
        seqs = sorted(p for p in root.glob('*/*') if p.is_dir() and p.name != 'calib')
        frames, bad = 0, []
        for d in seqs:
            sub = d if parts == 'gps' else d / 'labeled' / 'lidar_points'
            ext = '.txt' if parts == 'gps' else '.bin'
            n = len(list((sub / 'data').glob(f'*{ext}'))) if (sub / 'data').is_dir() else 0
            frames += n
            if not complete(d, parts):
                bad.append(d.name)
        print(f'{half:11s} {len(seqs):3d} seqs  {frames:6d} frames  mismatched={len(bad)} {bad[:5]}')
        bad_total += len(bad)
    print('ALL COMPLETE' if bad_total == 0 else f'{bad_total} INCOMPLETE')
    return bad_total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parts', choices=['gps', 'lidar', 'full'], default='lidar',
                    help='gps: per-frame GPS only. lidar: LiDAR+GPS (default). full: +cameras')
    ap.add_argument('--only', choices=['cadc', 'clear'], help='restrict to one half')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--verify-only', action='store_true', help='check what is on disk, download nothing')
    args = ap.parse_args()

    halves = [h for h in HALVES
              if not (args.only == 'cadc' and h != 'cadc')
              and not (args.only == 'clear' and h != 'cadc-clear')]

    if args.verify_only:
        return 1 if verify(args.parts, halves) else 0

    targets = []
    for half in halves:
        base, archive, calib = HALVES[half]
        seqs = sequences(base)
        print(f'{half}: {len(seqs)} sequences', flush=True)
        for date in sorted({d for d, _ in seqs}):
            print(f'  calib {half}/{date}: {fetch_calib(half, base, calib, date, args.parts)}', flush=True)
        targets += [(half, base, archive, d, s, args.parts) for d, s in seqs]

    counts = {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, *t) for t in targets]
        for i, f in enumerate(as_completed(futs), 1):
            half, date, seq, status = f.result()
            counts[status.split()[0]] = counts.get(status.split()[0], 0) + 1
            if status != 'skip':
                print(f'[{i}/{len(futs)}] {half}/{date}/{seq}: {status}', flush=True)
    print('\nsummary:', counts)
    print()
    return 1 if verify(args.parts, halves) else 0


if __name__ == '__main__':
    sys.exit(main())
