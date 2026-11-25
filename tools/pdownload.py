"""Resumable multi-connection HTTP downloader.

The DSEC server limits each connection to ~1 MB/s, but 4 parallel range requests reach ~15 MB/s.
The file is split into 64 MB chunks fetched by worker threads and written in place; finished chunk
indices are saved to <out>.progress.json so an interrupted download resumes where it stopped.
A pre-existing partial file without a progress file (e.g. left by curl, which writes sequentially)
is reused: chunks lying entirely inside its current size are treated as done.

Usage:  python pdownload.py <url> <out_file> [--workers 4]
Exit code 0 on success.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.request

CHUNK = 64 << 20


def head_size(url):
    """File size from HEAD; falls back to a 1-byte range GET (Content-Range) when HEAD gives no usable length,
    as Google Drive sometimes does. Raises instead of returning 0, which would mark an empty file complete."""
    for attempt in range(5):
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=60) as r:
            size = int(r.headers.get('Content-Length') or 0)
        if size > 0:
            return size
        req = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get('Content-Range') or ''
        if '/' in cr and cr.rsplit('/', 1)[1].isdigit() and int(cr.rsplit('/', 1)[1]) > 0:
            return int(cr.rsplit('/', 1)[1])
        time.sleep(10 * (attempt + 1))
    raise IOError(f'could not get a file size for {url}')


class QuotaError(IOError):
    """The server answered a range request with a full page instead of the bytes (e.g. Google Drive's
    'Quota exceeded' page). Retrying soon does not help, so the whole download stops with exit code 3."""


def fetch_chunk(url, f, lock, start, end):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end - 1}'})
    pos = start
    with urllib.request.urlopen(req, timeout=60) as r:
        whole = r.status == 200 and start == 0 and int(r.headers.get('Content-Length') or -1) == end
        if (r.status != 206 and not whole) or 'text/html' in (r.headers.get('Content-Type') or ''):
            raise QuotaError(f'HTTP {r.status} {r.headers.get("Content-Type")} instead of a byte range')
        while pos < end:
            b = r.read(min(1 << 20, end - pos))
            if not b:
                raise IOError(f'connection closed at {pos}')
            with lock:
                f.seek(pos)
                f.write(b)
            pos += len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('url')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    size = head_size(a.url)
    n_chunks = (size + CHUNK - 1) // CHUNK
    prog_file = a.out + '.progress.json'
    done = set()
    if os.path.exists(prog_file):
        done = set(json.load(open(prog_file))['done'])
    elif os.path.exists(a.out):
        # Count only bytes actually on disk: a run that stopped before finishing any chunk leaves a sparse file
        # already truncated to the full size, which must not look complete (st_blocks: allocated 512 B blocks).
        st = os.stat(a.out)
        prefix = min(st.st_size, getattr(st, 'st_blocks', st.st_size // 512 + 1) * 512)
        done = {i for i in range(n_chunks) if min((i + 1) * CHUNK, size) <= prefix}
    mode = 'r+b' if os.path.exists(a.out) else 'w+b'
    f = open(a.out, mode)
    f.truncate(size)
    with open(prog_file, 'w') as pf:   # written before any chunk, so the truncated file always has a progress file
        json.dump({'size': size, 'done': sorted(done)}, pf)

    todo = queue.Queue()
    for i in range(n_chunks):
        if i not in done:
            todo.put(i)
    lock, state_lock = threading.Lock(), threading.Lock()
    errors = []
    t0, done0 = time.time(), len(done)

    def save():
        with open(prog_file, 'w') as pf:
            json.dump({'size': size, 'done': sorted(done)}, pf)

    quota = []

    def worker():
        while not quota:
            try:
                i = todo.get_nowait()
            except queue.Empty:
                return
            start, end = i * CHUNK, min((i + 1) * CHUNK, size)
            for attempt in range(10):
                try:
                    fetch_chunk(a.url, f, lock, start, end)
                    break
                except QuotaError as e:
                    print(f'  chunk {i}: {e}; stopping', flush=True)
                    quota.append(i)
                    return
                except Exception as e:  # network hiccup: retry this chunk
                    print(f'  chunk {i} attempt {attempt + 1}: {type(e).__name__}: {str(e)[:120]}', flush=True)
                    time.sleep(min(60, 5 * (attempt + 1)))
            else:
                errors.append(i)
                continue
            with state_lock:
                done.add(i)
                save()
                el = time.time() - t0
                rate = (len(done) - done0) * CHUNK / max(el, 1e-6) / 1e6
                print(f'  {len(done)}/{n_chunks} chunks  {len(done) * CHUNK / 1e9:.1f}/{size / 1e9:.1f} GB  '
                      f'{rate:.1f} MB/s', flush=True)

    print(f'{a.url}\n  size {size / 1e9:.2f} GB, {n_chunks} chunks, {len(done)} already done, {a.workers} workers',
          flush=True)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(a.workers)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    f.close()
    if quota:
        print('QUOTA: server refused byte ranges; progress kept, retry later', flush=True)
        sys.exit(3)
    if errors:
        print(f'FAILED chunks: {errors}', flush=True)
        sys.exit(1)
    if os.path.exists(prog_file):   # absent when every chunk was already present on entry
        os.remove(prog_file)
    print('  complete', flush=True)


if __name__ == '__main__':
    main()
