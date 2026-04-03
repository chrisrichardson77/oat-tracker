#!/usr/bin/env python3
"""
find-passes.py — list upcoming satellite passes visible from an observer location.

Downloads a TLE catalog from Celestrak, propagates every satellite over the next
<lookahead> minutes, and prints a table of passes that rise above <min-alt> degrees,
sorted by rise time.

Usage example
-------------
    venv/bin/python find-passes.py --lat 51.5 --lon -0.1 --min-alt 20
"""

import argparse
import sys
import time
import math
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import requests
from sgp4.api import Satrec, SatrecArray, jday

# Catalog URLs from Celestrak (FORMAT=TLE).  Try each in order.
CATALOGS = [
    ("Visual (brightest ~100)",  "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=TLE"),
    ("Active satellites",        "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=TLE"),
    ("Space stations",           "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=TLE"),
]

# Earth radius (km) for approximate AltAz from TEME ECEF
EARTH_R_KM = 6371.0

CACHE_DIR = Path.home() / ".cache" / "oat-tracker"
OBSERVER_CACHE_FILE = CACHE_DIR / "observer.json"


# ── TLE loading ───────────────────────────────────────────────────────────────

def load_cached_observer() -> dict | None:
    try:
        data = json.loads(OBSERVER_CACHE_FILE.read_text())
        return {
            "lat": float(data["lat"]),
            "lon": float(data["lon"]),
            "elev": float(data.get("elev", 0.0)),
        }
    except Exception:
        return None


def save_cached_observer(lat: float, lon: float, elev: float) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        OBSERVER_CACHE_FILE.write_text(
            json.dumps({"lat": lat, "lon": lon, "elev": elev}, indent=2)
        )
    except Exception:
        pass

def fetch_tle_catalog(url: str) -> list[tuple[str, str, str]]:
    """Download a TLE catalog and return list of (name, tle1, tle2) tuples."""
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    lines = [ln.rstrip() for ln in resp.text.splitlines()]
    # Strip blank lines
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"Catalog too short ({len(lines)} lines)")
    sats = []
    for i in range(0, len(lines) - 2, 3):
        name = lines[i].strip()
        tle1 = lines[i + 1].strip()
        tle2 = lines[i + 2].strip()
        if tle1.startswith("1 ") and tle2.startswith("2 "):
            sats.append((name, tle1, tle2))
    return sats


# ── Coordinate helpers ────────────────────────────────────────────────────────

def observer_ecef_km(lat_deg: float, lon_deg: float, elev_m: float) -> np.ndarray:
    """Return observer ECEF position in km (WGS-84 approximation)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = EARTH_R_KM + elev_m / 1000.0
    return np.array([
        r * math.cos(lat) * math.cos(lon),
        r * math.cos(lat) * math.sin(lon),
        r * math.sin(lat),
    ])


def altaz_from_teme(sat_r_km: np.ndarray,  # shape (N, 3)
                    obs_ecef: np.ndarray,   # shape (3,)
                    gst_rad: float,         # Greenwich sidereal time in radians
                    lat_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute approximate altitude and azimuth (degrees) for N satellites.

    We rotate the TEME position into ECEF using GST, compute the range vector
    from observer to satellite in the SEZ (South-East-Zenith) frame, then
    derive alt/az.  This is the same geometry used inside sgp4's own examples.
    """
    # TEME→ECEF rotation around Z by -GST
    cos_g = math.cos(gst_rad)
    sin_g = math.sin(gst_rad)
    # rotate each row
    x_e = sat_r_km[:, 0] * cos_g + sat_r_km[:, 1] * sin_g
    y_e = -sat_r_km[:, 0] * sin_g + sat_r_km[:, 1] * cos_g
    z_e = sat_r_km[:, 2]

    # Range vector from observer (ECEF)
    dx = x_e - obs_ecef[0]
    dy = y_e - obs_ecef[1]
    dz = z_e - obs_ecef[2]

    # Rotate into SEZ (topocentric) frame
    lat = math.radians(lat_deg)
    lon = math.radians(0.0)  # The longitude is already baked into obs_ecef via GST

    # Observer unit normal (zenith direction in ECEF at this instant = rotated by GST)
    # obs ECEF longitudinal angle
    lon_obs = math.atan2(obs_ecef[1], obs_ecef[0])

    cos_lat = math.cos(lat)
    sin_lat = math.sin(lat)
    cos_lon = math.cos(lon_obs)
    sin_lon = math.sin(lon_obs)

    # SEZ rotation matrix rows:
    # S (south):  [ sin_lat*cos_lon, sin_lat*sin_lon, -cos_lat ]
    # E (east):   [ -sin_lon,         cos_lon,          0       ]
    # Z (zenith): [ cos_lat*cos_lon, cos_lat*sin_lon,  sin_lat  ]
    rho_s = sin_lat * cos_lon * dx + sin_lat * sin_lon * dy - cos_lat * dz
    rho_e = -sin_lon * dx + cos_lon * dy
    rho_z = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    rho = np.sqrt(rho_s ** 2 + rho_e ** 2 + rho_z ** 2)
    rho = np.where(rho == 0, 1e-9, rho)  # guard against zero

    el_rad = np.arcsin(rho_z / rho)
    az_rad = np.arctan2(rho_e, -rho_s)

    alt_deg = np.degrees(el_rad)
    az_deg = (np.degrees(az_rad) + 360.0) % 360.0
    return alt_deg, az_deg


def gmst_rad(jd_ut1: float) -> float:
    """Approximate Greenwich Mean Sidereal Time in radians for a JD(UT1)."""
    # IAU 1982 GMST formula (sufficient accuracy for pass prediction)
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd_ut1 - 2451545.0)
        + 0.000387933 * T * T
        - T * T * T / 38710000.0
    )
    return math.radians(gmst_deg % 360.0)


# ── Pass prediction ───────────────────────────────────────────────────────────

def predict_passes(
    sats: list[tuple[str, str, str]],
    lat_deg: float,
    lon_deg: float,
    elev_m: float,
    min_alt_deg: float,
    lookahead_min: int,
    step_sec: int = 15,
) -> list[dict]:
    """
    Predict upcoming passes for all satellites in the catalog.

    Returns a list of pass dicts sorted by rise time:
        name, rise_utc, max_alt, az_at_rise, az_at_max
    """
    obs_ecef = observer_ecef_km(lat_deg, lon_deg, elev_m)

    # Build SatrecArray for vectorized propagation
    sat_objects = []
    names = []
    for name, tle1, tle2 in sats:
        try:
            sat = Satrec.twoline2rv(tle1, tle2)
            sat_objects.append(sat)
            names.append(name)
        except Exception:
            continue

    if not sat_objects:
        return []

    sat_array = SatrecArray(sat_objects)
    n_sats = len(sat_objects)

    # Time grid: now → now + lookahead, every step_sec seconds
    now_unix = time.time()
    steps = range(0, lookahead_min * 60 + 1, step_sec)
    n_steps = len(steps)

    # Pre-compute jd arrays
    jd_arr = np.empty(n_steps)
    fr_arr = np.empty(n_steps)
    gst_arr = np.empty(n_steps)
    for i, offset in enumerate(steps):
        t = now_unix + offset
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
        jd_arr[i] = jd
        fr_arr[i] = fr
        gst_arr[i] = gmst_rad(jd + fr)

    # Propagate: shape (n_sats, n_steps, 3)
    e_arr, r_arr, _ = sat_array.sgp4(jd_arr, fr_arr)
    # e_arr: (n_sats, n_steps) error codes; r_arr: (n_sats, n_steps, 3) km TEME

    # For each time step compute alt/az for all sats at once
    # alt_grid[i, j] = altitude of sat i at time j
    alt_grid = np.full((n_sats, n_steps), -90.0)
    az_grid  = np.full((n_sats, n_steps),   0.0)

    for j in range(n_steps):
        valid_mask = e_arr[:, j] == 0
        if not np.any(valid_mask):
            continue
        pos = r_arr[:, j, :]  # (n_sats, 3)
        alt, az = altaz_from_teme(pos, obs_ecef, gst_arr[j], lat_deg)
        alt_grid[valid_mask, j] = alt[valid_mask]
        az_grid[valid_mask, j]  = az[valid_mask]

    # Detect passes: rising edge into min_alt cone within lookahead
    passes = []
    step_offsets = np.array(list(steps))
    for i in range(n_sats):
        alts = alt_grid[i]
        azs  = az_grid[i]

        # Find contiguous blocks above min_alt
        above = alts >= min_alt_deg
        if not np.any(above):
            continue

        # Find rising edge: index where above becomes True
        rise_indices = np.where(np.diff(above.astype(int)) == 1)[0] + 1
        # Also include index 0 if already above (satellite currently visible)
        if above[0]:
            rise_indices = np.concatenate([[0], rise_indices])

        for ri in rise_indices:
            # Find set index for this pass
            above_from_rise = above[ri:]
            set_offsets_rel = np.where(~above_from_rise)[0]
            si = ri + set_offsets_rel[0] - 1 if len(set_offsets_rel) > 0 else n_steps - 1

            # Max alt in this window
            pass_alts = alts[ri:si + 1]
            max_alt = float(np.max(pass_alts)) if len(pass_alts) > 0 else float(alts[ri])
            max_idx = ri + int(np.argmax(pass_alts))

            rise_unix = now_unix + float(step_offsets[ri])
            rise_dt = datetime.fromtimestamp(rise_unix, tz=timezone.utc)
            max_dt  = datetime.fromtimestamp(now_unix + float(step_offsets[max_idx]), tz=timezone.utc)

            already_up = (ri == 0 and above[0])

            passes.append({
                "name":        names[i],
                "rise_utc":    rise_dt,
                "max_utc":     max_dt,
                "max_alt":     max_alt,
                "az_rise":     float(azs[ri]),
                "az_max":      float(azs[max_idx]),
                "already_up":  already_up,
            })

    passes.sort(key=lambda p: p["rise_utc"])
    return passes


# ── Formatting ────────────────────────────────────────────────────────────────

def format_table(passes: list[dict], now_utc: datetime, max_rows: int) -> str:
    if not passes:
        return "  (no passes found above threshold)"

    header = f"{'Satellite':<30}  {'Rise (UTC)':<20}  {'In':>7}  {'Max alt':>7}  {'Az rise':>7}  {'Az@max':>7}"
    sep    = "-" * len(header)
    rows   = [header, sep]

    for p in passes[:max_rows]:
        dt_min = (p["rise_utc"] - now_utc).total_seconds() / 60.0
        if p["already_up"]:
            in_str = "NOW"
        else:
            in_str = f"{dt_min:+.0f} min"

        rise_str = p["rise_utc"].strftime("%H:%M:%S")
        rows.append(
            f"{p['name']:<30}  {rise_str:<20}  {in_str:>7}  {p['max_alt']:>6.1f}°"
            f"  {p['az_rise']:>6.1f}°  {p['az_max']:>6.1f}°"
        )

    return "\n".join(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="List upcoming satellite passes visible from your location.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    obs = p.add_argument_group("Observer location (optional if cached)")
    obs.add_argument("--lat",  type=float, default=None, metavar="DEG",
                     help="Latitude in decimal degrees (N positive). If provided, overrides cached latitude")
    obs.add_argument("--lon",  type=float, default=None, metavar="DEG",
                     help="Longitude in decimal degrees (E positive). If provided, overrides cached longitude")
    obs.add_argument("--elev", type=float, default=None,  metavar="M",
                     help="Elevation above sea level in metres. If provided, overrides cached elevation")
    obs.add_argument("--no-save-location", action="store_true",
                     help="Do not save/update cached observer location (overrides still apply for this run only)")

    p.add_argument("--min-alt",    type=float, default=20.0, metavar="DEG",
                   help="Minimum altitude threshold (degrees)")
    p.add_argument("--lookahead",  type=int,   default=90,   metavar="MIN",
                   help="Prediction window in minutes")
    p.add_argument("--step",       type=int,   default=15,   metavar="SEC",
                   help="Propagation time step in seconds (smaller = more accurate)")
    p.add_argument("--catalog",    type=int,   default=0,    metavar="N",
                   choices=range(len(CATALOGS)),
                   help="; ".join(f"{i}={name}" for i, (name, _) in enumerate(CATALOGS)))
    p.add_argument("--rows",       type=int,   default=30,   metavar="N",
                   help="Maximum number of passes to show")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    cached = load_cached_observer()
    lat = args.lat if args.lat is not None else (cached["lat"] if cached else None)
    lon = args.lon if args.lon is not None else (cached["lon"] if cached else None)
    elev = args.elev if args.elev is not None else (cached["elev"] if cached else 0.0)

    if lat is None or lon is None:
        print(
            "ERROR: --lat and --lon are required on first run, or after deleting cache.",
            file=sys.stderr,
        )
        print(f"Expected cache file: {OBSERVER_CACHE_FILE}", file=sys.stderr)
        sys.exit(2)

    if not args.no_save_location and (
        args.lat is not None or args.lon is not None or args.elev is not None or cached is None
    ):
        save_cached_observer(lat, lon, elev)

    cat_name, cat_url = CATALOGS[args.catalog]
    print(f"Catalog    : {cat_name}")
    print(f"Observer   : {lat:.4f}°N  {lon:.4f}°E  elev {elev:.0f} m")
    print(f"Min alt    : {args.min_alt}°   Lookahead: {args.lookahead} min   Step: {args.step} s")
    print()

    print(f"Downloading TLEs from Celestrak …", end="", flush=True)
    try:
        sats = fetch_tle_catalog(cat_url)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f" {len(sats)} satellites loaded.")

    print(f"Propagating {len(sats)} satellites over {args.lookahead} min …", end="", flush=True)
    t0 = time.time()
    passes = predict_passes(
        sats,
        lat_deg=lat,
        lon_deg=lon,
        elev_m=elev,
        min_alt_deg=args.min_alt,
        lookahead_min=args.lookahead,
        step_sec=args.step,
    )
    print(f" done in {time.time() - t0:.1f} s — {len(passes)} passes found.")

    now_utc = datetime.now(tz=timezone.utc)
    print(f"\nUpcoming passes as of {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    print(format_table(passes, now_utc, args.rows))
    print()

    if passes:
        first = passes[0]
        if first["already_up"]:
            in_str = "NOW"
        else:
            in_str = f"{(first['rise_utc'] - now_utc).total_seconds() / 60:.0f} min"
        print(
            f"Next pass: \033[1m{first['name']}\033[0m  "
            f"rises in {in_str}  max alt {first['max_alt']:.1f}°"
        )
        print(
            f"To track it:  venv/bin/python oat-tracker.py "
            f"--lat {lat} --lon {lon} --elev {elev:.0f} "
            f"--device 'LX200 OpenAstroTech' --target '{first['name']}'"
        )


if __name__ == "__main__":
    main()
