"""Resumable download of Ca II H & K cut-outs from the ESO HARPS archive.

Instead of downloading full HARPS 1D spectra (~5 MB each) and discarding most of
the wavelength range, this script asks the ESO SODA cut-out service for only the
two windows of interest (Ca II K and H, +/- half_window_nm each, as in
Bendahan-West et al. 2025) and stores them as compact .npz files.

Commands
    python harps_fetch.py manifest   # list every HARPS spectrum from ESO TAP
    python harps_fetch.py run        # download/extract; safe to stop and rerun
    python harps_fetch.py status     # print the counters
    python harps_fetch.py retry      # reset failed spectra so `run` tries them again

Progress lives in <data_dir>/state.sqlite. Every finished spectrum is committed
immediately, so stopping (Ctrl+C), a crash or a power cut loses at most the
spectra that were in flight; rerunning `run` continues where it stopped.

Settings come from harps_fetch.json, next to this script (set data_dir to where the data should go).
Disk use for the whole archive: about 24 GB stored (202 GB would be downloaded as full spectra; only the two
2 nm windows are kept). Requirements: numpy, pandas, requests, astropy.
"""

import argparse
import gzip
import io
import json
import logging
import random
import signal
import sqlite3
import sys
import threading
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits

CONFIG_PATH = Path(__file__).resolve().with_name("harps_fetch.json")

# Header keywords copied into the database for fast queries. The full primary
# header is also kept, gzipped, inside each .npz file.
HEADER_KEYS = [
    "OBJECT", "RA", "DEC", "MJD-OBS", "DATE-OBS", "EXPTIME", "SNR", "SPEC_RES",
    "SPEC_BIN", "SPECSYS", "PROG_ID", "ARCFILE",
    "HIERARCH ESO OBS TARG NAME", "HIERARCH ESO OBS PROG ID",
    "HIERARCH ESO DRS BERV", "HIERARCH ESO DRS VERSION",
    "HIERARCH ESO TEL TARG ALPHA", "HIERARCH ESO TEL TARG DELTA",
    "HIERARCH ESO TEL TARG PMA", "HIERARCH ESO TEL TARG PMD",
    "HIERARCH ESO TEL TARG EPOCH", "HIERARCH ESO TEL TARG EPOCHSYSTEM",
    "HIERARCH ESO TEL TARG EQUINOX", "HIERARCH ESO TEL TARG RADVEL",
    "HIERARCH ESO TEL TARG PARALLAX",
]

# A window "covers" a line when at least this fraction of its pixels are finite
# and the window reaches within one bin of both requested edges.
MIN_FINITE_FRACTION = 0.95

log = logging.getLogger("harps_fetch")
_thread_local = threading.local()
_stop = threading.Event()


# --------------------------------------------------------------------------- setup

def load_config():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    cfg["data_dir"] = Path(cfg["data_dir"])
    return cfg


def setup_logging(data_dir):
    (data_dir / "logs").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(data_dir / "logs" / "fetch.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def connect(data_dir):
    db = sqlite3.connect(data_dir / "state.sqlite", timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("""
        CREATE TABLE IF NOT EXISTS spectra (
            dp_id        TEXT PRIMARY KEY,
            target_name  TEXT,
            s_ra         REAL,
            s_dec        REAL,
            t_min        REAL,
            snr          REAL,
            est_kb       REAL,
            proposal_id  TEXT,
            priority     INTEGER,
            status       TEXT DEFAULT 'pending',  -- pending | done | failed | restricted
            attempts     INTEGER DEFAULT 0,
            last_error   TEXT,
            covers_K     INTEGER,
            covers_H     INTEGER,
            finite_K     REAL,
            finite_H     REAL,
            file         TEXT,
            bytes_downloaded INTEGER,
            bytes_stored INTEGER,
            header_json  TEXT,
            finished_at  TEXT
        )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_status ON spectra(status, priority)")
    db.commit()
    return db


# ------------------------------------------------------------------------ manifest

def tap_query(cfg, adql):
    r = requests.post(cfg["tap_url"], timeout=600, data={
        "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
        "MAXREC": "2000000", "QUERY": adql,
    })
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def build_manifest(cfg):
    """List every public HARPS 1D spectrum, one year at a time, into the database."""
    data_dir = cfg["data_dir"]
    db = connect(data_dir)
    cols = "dp_id, target_name, s_ra, s_dec, t_min, snr, access_estsize, proposal_id"
    frames = []
    # MJD 52640 = 2003-01-01; one chunk per ~year keeps each TAP response small.
    for start in range(52640, 62000, 366):
        df = tap_query(cfg, f"SELECT {cols} FROM ivoa.ObsCore "
                            f"WHERE obs_collection='HARPS' AND dataproduct_type='spectrum' "
                            f"AND t_min >= {start} AND t_min < {start + 366}")
        log.info("manifest: MJD %d-%d -> %d spectra", start, start + 366, len(df))
        frames.append(df)
    df = pd.concat(frames).drop_duplicates("dp_id")
    total = tap_query(cfg, "SELECT COUNT(*) AS n FROM ivoa.ObsCore "
                           "WHERE obs_collection='HARPS' AND dataproduct_type='spectrum'")["n"][0]
    if len(df) != total:
        log.warning("manifest has %d spectra but the archive reports %d", len(df), total)

    priority_names = {normalise(n) for n in cfg["priority_targets"]}
    rng = random.Random(42)  # random order so a partial run is a fair sample
    df["priority"] = [0 if normalise(str(t)) in priority_names else 1 + rng.randrange(10**6)
                      for t in df["target_name"]]

    (data_dir / "manifest").mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "manifest" / "obscore_harps.csv", index=False)

    rows = [(r.dp_id, r.target_name, r.s_ra, r.s_dec, r.t_min, r.snr, r.access_estsize,
             None if pd.isna(r.proposal_id) else str(r.proposal_id), int(r.priority))
            for r in df.itertuples()]
    # INSERT OR IGNORE keeps progress of spectra already processed.
    before = db.execute("SELECT COUNT(*) FROM spectra").fetchone()[0]
    db.executemany("INSERT OR IGNORE INTO spectra (dp_id, target_name, s_ra, s_dec, t_min, "
                   "snr, est_kb, proposal_id, priority) VALUES (?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    added = db.execute("SELECT COUNT(*) FROM spectra").fetchone()[0] - before
    log.info("manifest: %d spectra listed, %d new in the database", len(df), added)
    print_status(cfg, db)


def normalise(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


# -------------------------------------------------------------------------- worker

def session():
    if not hasattr(_thread_local, "s"):
        s = requests.Session()
        s.headers["User-Agent"] = "harps_fetch"
        _thread_local.s = s
    return _thread_local.s


def out_path(data_dir, dp_id):
    # dp_id looks like ADP.2014-10-01T10:21:19.907; ':' is not allowed in Windows filenames.
    safe = dp_id.replace(":", "_")
    return data_dir / "spectra" / dp_id[4:11] / f"{safe}.npz"


def window_quality(wave, flux, centre, half):
    """wave in nm. Returns (covers, finite_fraction)."""
    finite = np.isfinite(flux) & (flux != 0)
    frac = float(finite.mean()) if flux.size else 0.0
    reaches = flux.size > 0 and wave.min() <= centre - half + 0.002 and wave.max() >= centre + half - 0.002
    return int(reaches and frac >= MIN_FINITE_FRACTION), frac


def fetch_one(cfg, dp_id):
    """Download the K and H cut-outs for one spectrum and save them. Returns a result dict."""
    lines, half = cfg["lines_nm"], cfg["half_window_nm"]
    bands = [(name, centre) for name, centre in lines.items()]
    params = [("ID", f"ivo://eso.org/ID?{dp_id}")]
    for _, centre in bands:
        params.append(("BAND", f"{(centre - half) * 1e-9:.6E} {(centre + half) * 1e-9:.6E}"))

    r = session().get(cfg["soda_url"], params=params, timeout=cfg["request_timeout_s"])
    if r.status_code in (401, 403):
        return {"dp_id": dp_id, "status": "restricted", "error": f"HTTP {r.status_code}"}
    r.raise_for_status()
    content = r.content

    # One band returns a FITS file; several bands return a zip of FITS files (cut0, cut1, ...).
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            names = sorted(z.namelist())
            parts = [z.read(n) for n in names]
    else:
        parts = [content]
    if len(parts) != len(bands):
        raise ValueError(f"expected {len(bands)} cut-outs, got {len(parts)}")

    arrays, quality, header = {}, {}, None
    for (name, centre), blob in zip(bands, parts):
        with fits.open(io.BytesIO(blob), memmap=False) as hdul:
            if header is None:
                header = hdul[0].header
            ext = hdul["SPECTRUM"]
            row = ext.data[0]
            wave = np.asarray(row["WAVE"], dtype=np.float64)
            # HARPS ADP spectra store WAVE in Angstrom; keep everything in nm.
            unit = (ext.columns["WAVE"].unit or "").strip().lower()
            if unit.startswith("angstrom") or (not unit and np.nanmedian(wave) > 1000):
                wave = wave / 10.0
            flux = np.asarray(row["FLUX"], dtype=np.float32)
            err = np.asarray(row["ERR"], dtype=np.float32)
        arrays[f"wave_{name}"], arrays[f"flux_{name}"], arrays[f"err_{name}"] = wave, flux, err
        quality[name] = window_quality(wave, flux, centre, half)

    path = out_path(cfg["data_dir"], dp_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, header_gz=np.frombuffer(gzip.compress(header.tostring().encode()), dtype=np.uint8),
                        **arrays)
    tmp.replace(path)  # atomic: a half-written file is never left under the final name

    hdr = {k: header.get(k.replace("HIERARCH ", "")) for k in HEADER_KEYS}
    return {
        "dp_id": dp_id, "status": "done", "error": None,
        "covers_K": quality["K"][0], "finite_K": quality["K"][1],
        "covers_H": quality["H"][0], "finite_H": quality["H"][1],
        "file": str(path.relative_to(cfg["data_dir"])),
        "bytes_downloaded": len(content), "bytes_stored": path.stat().st_size,
        "header_json": json.dumps(hdr, default=str),
    }


# --------------------------------------------------------------------------- run

def run(cfg):
    data_dir = cfg["data_dir"]
    db = connect(data_dir)
    if db.execute("SELECT COUNT(*) FROM spectra").fetchone()[0] == 0:
        log.error("database is empty: run `manifest` first")
        return

    signal.signal(signal.SIGINT, lambda *_: (_stop.set(), log.warning(
        "stop requested: finishing spectra in flight, then exiting (press Ctrl+C again to force)"),
        signal.signal(signal.SIGINT, signal.SIG_DFL)))

    # Each pass retries what failed in the previous one (temporary server errors are common).
    for pass_no in range(1, cfg["max_attempts"] + 1):
        processed = run_pass(cfg, db, pass_no)
        if _stop.is_set() or processed == 0 or cfg.get("_limit"):
            break
        log.info("pass %d complete; retrying failed spectra in 60 s", pass_no)
        for _ in range(60):
            if _stop.is_set():
                break
            time.sleep(1)

    log.info("run finished%s", " (stopped early)" if _stop.is_set() else "")
    print_status(cfg, db)


def run_pass(cfg, db, pass_no):
    workers, max_attempts = cfg["workers"], cfg["max_attempts"]
    todo = [r[0] for r in db.execute(
        "SELECT dp_id FROM spectra WHERE status IN ('pending','failed') AND attempts < ? "
        "ORDER BY priority", (max_attempts,))]
    if cfg.get("_limit"):
        todo = todo[:cfg["_limit"]]
    log.info("pass %d: %d spectra to process with %d parallel downloads", pass_no, len(todo), workers)
    if not todo:
        return 0

    started, done_now, failed_now, bytes_now = time.time(), 0, 0, 0
    last_report, consecutive_errors = time.time(), 0
    queue = iter(todo)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        in_flight = {}

        def submit_next():
            dp = next(queue, None)
            if dp is not None and not _stop.is_set():
                in_flight[pool.submit(fetch_one, cfg, dp)] = dp

        for _ in range(workers * 2):
            submit_next()

        while in_flight:
            finished, _ = wait(in_flight, timeout=30, return_when=FIRST_COMPLETED)
            for fut in finished:
                dp = in_flight.pop(fut)
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                try:
                    res = fut.result()
                except Exception as exc:  # network error, bad file, server error...
                    failed_now += 1
                    consecutive_errors += 1
                    db.execute("UPDATE spectra SET status='failed', attempts=attempts+1, "
                               "last_error=?, finished_at=? WHERE dp_id=?",
                               (f"{type(exc).__name__}: {exc}"[:500], now, dp))
                else:
                    consecutive_errors = 0
                    if res["status"] == "restricted":
                        db.execute("UPDATE spectra SET status='restricted', attempts=attempts+1, "
                                   "last_error=?, finished_at=? WHERE dp_id=?",
                                   (res["error"], now, dp))
                    else:
                        done_now += 1
                        bytes_now += res["bytes_downloaded"]
                        db.execute(
                            "UPDATE spectra SET status='done', attempts=attempts+1, last_error=NULL, "
                            "covers_K=?, finite_K=?, covers_H=?, finite_H=?, file=?, "
                            "bytes_downloaded=?, bytes_stored=?, header_json=?, finished_at=? "
                            "WHERE dp_id=?",
                            (res["covers_K"], res["finite_K"], res["covers_H"], res["finite_H"],
                             res["file"], res["bytes_downloaded"], res["bytes_stored"],
                             res["header_json"], now, dp))
                db.commit()  # every spectrum is saved before moving on
                submit_next()

            # If the archive is having trouble, back off instead of hammering it.
            if consecutive_errors >= workers * 3:
                log.warning("%d errors in a row: pausing 5 minutes", consecutive_errors)
                for _ in range(300):
                    if _stop.is_set():
                        break
                    time.sleep(1)
                consecutive_errors = 0

            if time.time() - last_report > 60:
                last_report = time.time()
                elapsed_h = (time.time() - started) / 3600
                rate = done_now / elapsed_h if elapsed_h > 0 else 0
                remaining = db.execute("SELECT COUNT(*) FROM spectra WHERE status IN "
                                       "('pending','failed') AND attempts < ?", (max_attempts,)).fetchone()[0]
                eta = remaining / rate if rate else float("nan")
                log.info("progress: +%d done, +%d failed this run | %.0f spectra/h | %.1f GB downloaded "
                         "| %d remaining | ETA %.1f h", done_now, failed_now, rate, bytes_now / 1e9,
                         remaining, eta)

    return len(todo)


# ------------------------------------------------------------------------- status

def print_status(cfg, db=None):
    db = db or connect(cfg["data_dir"])
    q = lambda sql: db.execute(sql).fetchone()[0] or 0
    listed = q("SELECT COUNT(*) FROM spectra")
    done = q("SELECT COUNT(*) FROM spectra WHERE status='done'")
    stats = {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spectra_listed_in_archive": listed,
        "spectra_downloaded": done,
        "spectra_covering_K": q("SELECT COUNT(*) FROM spectra WHERE status='done' AND covers_K=1"),
        "spectra_covering_H": q("SELECT COUNT(*) FROM spectra WHERE status='done' AND covers_H=1"),
        "spectra_covering_both": q("SELECT COUNT(*) FROM spectra WHERE status='done' AND covers_K=1 AND covers_H=1"),
        "pending": q("SELECT COUNT(*) FROM spectra WHERE status='pending'"),
        "failed": q("SELECT COUNT(*) FROM spectra WHERE status='failed'"),
        "restricted_proprietary": q("SELECT COUNT(*) FROM spectra WHERE status='restricted'"),
        "unique_target_names_downloaded": q("SELECT COUNT(DISTINCT target_name) FROM spectra WHERE status='done'"),
        "gb_downloaded": round(q("SELECT SUM(bytes_downloaded) FROM spectra") / 1e9, 2),
        "gb_stored": round(q("SELECT SUM(bytes_stored) FROM spectra") / 1e9, 2),
    }
    (cfg["data_dir"] / "summary.json").write_text(json.dumps(stats, indent=2))
    width = max(len(k) for k in stats)
    print()
    for k, v in stats.items():
        print(f"  {k.ljust(width)}  {v}")
    print("  (target names are raw archive names, not unique stars: grouping comes later)\n")


def retry_failed(cfg):
    db = connect(cfg["data_dir"])
    n = db.execute("UPDATE spectra SET status='pending', attempts=0 WHERE status='failed'").rowcount
    db.commit()
    log.info("retry: %d failed spectra reset to pending", n)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["manifest", "run", "status", "retry"])
    p.add_argument("--limit", type=int, help="run: only process this many spectra (for testing)")
    p.add_argument("--workers", type=int, help="override the number of parallel downloads")
    args = p.parse_args()

    cfg = load_config()
    if args.workers:
        cfg["workers"] = args.workers
    cfg["data_dir"].mkdir(parents=True, exist_ok=True)
    setup_logging(cfg["data_dir"])

    if args.command == "manifest":
        build_manifest(cfg)
    elif args.command == "run":
        if args.limit:
            cfg["_limit"] = args.limit
        run(cfg)
    elif args.command == "status":
        print_status(cfg)
    elif args.command == "retry":
        retry_failed(cfg)


if __name__ == "__main__":
    main()
