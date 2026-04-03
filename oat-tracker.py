#!/usr/bin/env python3
"""
oat-tracker.py — Track satellites, planets, and stars with an INDI-connected mount.

Downloads TLE data from Celestrak, propagates the ISS position with SGP4,
and drives an INDI mount via the INDI protocol to follow the ISS across the sky.

The position pipeline is:
    SGP4 (TEME frame)  →  AltAz (topocentric, observer-fixed)
        →  TETE apparent / epoch-of-date  →  EQUATORIAL_EOD_COORD (INDI)

Going through AltAz is essential: TEME→GCRS gives *geocentric* coordinates,
but the ISS is only ~400 km up so the parallax offset can be tens of degrees.
The AltAz transform naturally applies the observer's topocentric correction.

Dependencies
------------
    pip install requests sgp4 astropy pyindi-client

Quick start
-----------
    python oat-tracker.py \\
        --lat 51.5074 --lon -0.1278 --elev 10 \\
        --host localhost --port 7624 \\
        --device "EQMod Mount"
"""

import sys
import time
import json
import argparse
import logging
import difflib
import subprocess
import os
import math
import socket
import threading
from pathlib import Path
from datetime import datetime, timezone

import requests
from sgp4.api import Satrec

from astropy.time import Time
from astropy.coordinates import (
    TEME,
    AltAz,
    TETE,
    get_body,
    CartesianRepresentation,
    EarthLocation,
    SkyCoord,
)
import astropy.units as u

try:
    import PyIndi
except ImportError:
    sys.exit(
        "pyindi-client is not installed. Install it with:\n"
        "    pip install pyindi-client"
    )

# ── Constants ─────────────────────────────────────────────────────────────────

ISS_NORAD_ID = 25544
CELESTRAK_BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"

TLE_REFRESH_INTERVAL = 3600    # seconds — refresh TLE data every hour
DEFAULT_UPDATE_INTERVAL = 2    # seconds — position command rate
DEFAULT_SAT_INTERVAL = 0.3     # seconds — satellite command update rate
RATE_ESTIMATE_DT = 1.0         # seconds — finite-difference horizon for rate estimates
DEFAULT_MIN_ALT = 10.0         # degrees — don't track below this elevation
DEFAULT_PREPOSITION_LOOKAHEAD = 5400.0  # seconds — search next 90 minutes
DEFAULT_PREPOSITION_ALT = 0.0            # degrees — slew to apparent rise point
PREDICTION_RETRY_INTERVAL = 60.0         # seconds
DISPLAY_RISE_LOOKAHEAD = 12 * 3600       # seconds — longer horizon for user-facing rise ETA
DISPLAY_RISE_LOOKAHEAD_MAX = 24 * 3600   # seconds — extended fallback horizon for user-facing rise ETA
DEFAULT_CONNECT_TIMEOUT = 8.0            # seconds — INDI handshake/device stage timeout

CACHE_DIR = Path.home() / ".cache" / "oat-tracker"
OBSERVER_CACHE_FILE = CACHE_DIR / "observer.json"
TLE_CACHE_TTL = 6 * 3600          # seconds — how long a cached TLE stays fresh
STAR_CACHE_TTL = 30 * 24 * 3600   # seconds — 30 days for SIMBAD star lookups
SUPPORTED_PLANETS = {
    "mercury",
    "venus",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
}


_ORIGINAL_STDERR_FD: int | None = None


# ── INDI client ───────────────────────────────────────────────────────────────

class _IndiClient(PyIndi.BaseClient):
    """Thin INDI client wrapper with no-op callbacks."""

    def newDevice(self, d): pass        # noqa: N802
    def newProperty(self, p): pass      # noqa: N802
    def removeProperty(self, p): pass   # noqa: N802
    def newBLOB(self, bp): pass         # noqa: N802
    def newSwitch(self, svp): pass      # noqa: N802
    def newNumber(self, nvp): pass      # noqa: N802
    def newText(self, tvp): pass        # noqa: N802
    def newLight(self, lvp): pass       # noqa: N802
    def newMessage(self, d, m): pass    # noqa: N802
    def serverConnected(self): pass     # noqa: N802
    def serverDisconnected(self, code): pass  # noqa: N802


def configure_native_stderr(show_native_stderr: bool) -> None:
    """
    Suppress noisy stderr lines emitted by native INDI/PyIndi layers.

    Python logging still goes to stdout, so normal tracker status remains visible.
    """
    global _ORIGINAL_STDERR_FD

    if show_native_stderr:
        return

    if _ORIGINAL_STDERR_FD is not None:
        return

    try:
        _ORIGINAL_STDERR_FD = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
    except OSError:
        # If redirection fails, continue with default stderr behavior.
        _ORIGINAL_STDERR_FD = None


def _fatal_exit(message: str) -> None:
    """Emit fatal errors to stdout/logging so they remain visible when stderr is hidden."""
    if logging.getLogger().handlers:
        logging.error("%s", message)
    else:
        print(message, file=sys.stdout, flush=True)
    raise SystemExit(1)


# ── Disk cache helpers ───────────────────────────────────────────────────────

def _cache_path(subdir: str, key: str, suffix: str) -> Path:
    safe_key = key.replace("/", "_").replace(" ", "_")
    p = CACHE_DIR / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{safe_key}{suffix}"


def _cache_age(path: Path) -> float:
    """Return seconds since path was last modified, or infinity if absent."""
    try:
        return time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return float("inf")


def _read_tle_cache(key: str) -> tuple[str, str, str] | None:
    """Return (display_name, tle1, tle2) from cache, or None if stale/missing."""
    p = _cache_path("tle", key, ".txt")
    if _cache_age(p) > TLE_CACHE_TTL:
        return None
    try:
        lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        if len(lines) >= 3:
            return lines[0], lines[1], lines[2]
    except OSError:
        pass
    return None


def _write_tle_cache(key: str, display_name: str, tle1: str, tle2: str) -> None:
    p = _cache_path("tle", key, ".txt")
    try:
        p.write_text(f"{display_name}\n{tle1}\n{tle2}\n")
    except OSError:
        pass  # cache write failure is non-fatal


def _read_star_cache(star_name: str) -> tuple[float, float] | None:
    """Return (ra_deg, dec_deg) in ICRS from cache, or None if stale/missing."""
    p = _cache_path("stars", star_name.lower(), ".json")
    if _cache_age(p) > STAR_CACHE_TTL:
        return None
    try:
        data = json.loads(p.read_text())
        return float(data["ra_deg"]), float(data["dec_deg"])
    except (OSError, KeyError, ValueError):
        pass
    return None


def _write_star_cache(star_name: str, ra_deg: float, dec_deg: float) -> None:
    p = _cache_path("stars", star_name.lower(), ".json")
    try:
        p.write_text(json.dumps({"ra_deg": ra_deg, "dec_deg": dec_deg}))
    except OSError:
        pass


def _read_observer_cache() -> tuple[float, float, float] | None:
    """Return (lat_deg, lon_deg, elev_m) from disk cache, or None if unavailable."""
    try:
        data = json.loads(OBSERVER_CACHE_FILE.read_text())
        lat_deg = float(data["lat"])
        lon_deg = float(data["lon"])
        elev_m = float(data.get("elev", 0.0))
    except (OSError, KeyError, ValueError):
        return None
    return lat_deg, lon_deg, elev_m


def _write_observer_cache(lat_deg: float, lon_deg: float, elev_m: float) -> None:
    """Persist observer coordinates for future runs."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        OBSERVER_CACHE_FILE.write_text(
            json.dumps({"lat": lat_deg, "lon": lon_deg, "elev": elev_m}, indent=2)
        )
    except OSError:
        pass


# ── TLE helpers ───────────────────────────────────────────────────────────────

def fetch_tle(url: str) -> tuple[str, str]:
    """
    Download a TLE from *url* and return (line1, line2).

    Celestrak's FORMAT=TLE response has the layout:
        ISS (ZARYA)
        1 25544U ...
        2 25544 ...
    """
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"Unexpected TLE payload (only {len(lines)} lines)")
    # lines[0] = object name, lines[1] = TLE line 1, lines[2] = TLE line 2
    return lines[1], lines[2]


def build_satellite(tle1: str, tle2: str) -> Satrec:
    return Satrec.twoline2rv(tle1, tle2)


def _tle_url_for_norad_id(norad_id: int) -> str:
    return f"{CELESTRAK_BASE_URL}?CATNR={norad_id}&FORMAT=TLE"


def resolve_satellite_tle(
    target: str,
) -> tuple[str, str, str, str]:
    """
    Resolve a satellite target to (display_name, tle_line1, tle_line2, refresh_url).

    Accepts:
    - "iss"          — the ISS (NORAD 25544)
    - All-digit str  — a NORAD catalog number
    - Name fragment  — searched on Celestrak by name (first/best match returned)

    Results are cached in ~/.cache/oat-tracker/tle/ for TLE_CACHE_TTL seconds.

    Raises
    ------
    ValueError
        If the satellite cannot be found.
    """
    if target == "iss":
        norad_id = ISS_NORAD_ID
        cache_key = str(norad_id)
        url = _tle_url_for_norad_id(norad_id)
    elif target.isdigit():
        norad_id = int(target)
        cache_key = str(norad_id)
        url = _tle_url_for_norad_id(norad_id)
    else:
        # Name search — cache keyed on the normalised name fragment
        cache_key = f"name_{target.lower().replace(' ', '_')}"
        cached = _read_tle_cache(cache_key)
        if cached is not None:
            display_name, tle1, tle2 = cached
            logging.debug("TLE cache hit for '%s' (%s)", target, display_name)
            norad_id = int(tle2.split()[1])
            return display_name, tle1, tle2, _tle_url_for_norad_id(norad_id)

        import urllib.parse
        url = f"{CELESTRAK_BASE_URL}?NAME={urllib.parse.quote(target.upper())}&FORMAT=TLE"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            raw_lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
        except Exception as exc:
            raise ValueError(f"Celestrak name search for '{target}' failed: {exc}")

        if len(raw_lines) < 3:
            raise ValueError(f"No satellite found on Celestrak matching '{target}'")

        names = [raw_lines[i] for i in range(0, len(raw_lines) - 2, 3)]
        upper_names = [n.upper() for n in names]
        matches = difflib.get_close_matches(target.upper(), upper_names, n=1, cutoff=0.0)
        idx = upper_names.index(matches[0]) if matches else 0

        display_name = names[idx]
        tle1 = raw_lines[idx * 3 + 1]
        tle2 = raw_lines[idx * 3 + 2]
        norad_id = int(tle2.split()[1])
        # Cache under both the name key and the NORAD ID for future lookups
        _write_tle_cache(cache_key, display_name, tle1, tle2)
        _write_tle_cache(str(norad_id), display_name, tle1, tle2)
        return display_name, tle1, tle2, _tle_url_for_norad_id(norad_id)

    # NORAD-ID path (iss or numeric) — check cache first
    cached = _read_tle_cache(cache_key)
    if cached is not None:
        display_name, tle1, tle2 = cached
        logging.debug("TLE cache hit for NORAD %s (%s)", cache_key, display_name)
        return display_name, tle1, tle2, url

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        raw_lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
    except Exception as exc:
        raise ValueError(f"Could not fetch TLE for '{target}': {exc}")

    if len(raw_lines) < 3:
        raise ValueError(f"No TLE returned for '{target}'")

    display_name, tle1, tle2 = raw_lines[0], raw_lines[1], raw_lines[2]
    _write_tle_cache(cache_key, display_name, tle1, tle2)
    return display_name, tle1, tle2, url


# ── Coordinate computation ────────────────────────────────────────────────────

def iss_apparent_eod(
    satellite: Satrec,
    t: Time,
    location: EarthLocation,
) -> tuple[float, float, float]:
    """
    Compute the topocentric apparent position of the ISS at astropy Time *t*.

    Returns
    -------
    (ra_hours, dec_deg, alt_deg)
        RA and Dec in the current epoch-of-date (for EQUATORIAL_EOD_COORD),
        and altitude above the observer's horizon for visibility gating.

    Raises
    ------
    RuntimeError
        If SGP4 propagation fails (usually means a stale TLE).
    """
    # ── SGP4 propagation ─────────────────────────────────────────────────────
    error_code, r_km, _v = satellite.sgp4(t.jd1, t.jd2)
    if error_code != 0:
        raise RuntimeError(
            f"SGP4 error code {error_code} — TLE may be too old to propagate"
        )

    # ── Build TEME SkyCoord ───────────────────────────────────────────────────
    r_teme = CartesianRepresentation(
        r_km[0] * u.km, r_km[1] * u.km, r_km[2] * u.km
    )
    iss_teme = SkyCoord(r_teme, frame=TEME(obstime=t))

    # ── Topocentric AltAz ────────────────────────────────────────────────────
    # Going through AltAz is *required* for LEO objects: TEME→GCRS would give
    # geocentric RA/Dec, but at ~400 km altitude the parallax is enormous.
    # AltAz() built with an EarthLocation applies the full topocentric correction.
    altaz_frame = AltAz(obstime=t, location=location)
    iss_altaz = iss_teme.transform_to(altaz_frame)

    # ── Epoch-of-date RA/Dec ─────────────────────────────────────────────────
    # INDI's EQUATORIAL_EOD_COORD expects topocentric apparent RA/Dec in the
    # current epoch. TETE matches the direct observed alt/az -> RA/Dec solution.
    iss_eod = iss_altaz.transform_to(TETE(obstime=t, location=location))

    return iss_eod.ra.hour, iss_eod.dec.deg, float(iss_altaz.alt.deg)


def planet_apparent_eod(
    body_name: str,
    t: Time,
    location: EarthLocation,
) -> tuple[float, float, float]:
    """
    Compute topocentric apparent RA/Dec (epoch-of-date) for a solar-system body.

    Returns
    -------
    (ra_hours, dec_deg, alt_deg)
    """
    body = get_body(body_name, t, location=location)

    altaz_frame = AltAz(obstime=t, location=location)
    body_altaz = body.transform_to(altaz_frame)
    body_eod = body_altaz.transform_to(TETE(obstime=t, location=location))

    return body_eod.ra.hour, body_eod.dec.deg, float(body_altaz.alt.deg)


def resolve_star(star_name: str) -> SkyCoord:
    """
    Resolve a star name to RA/Dec coordinates via SIMBAD.

    Results are cached in ~/.cache/oat-tracker/stars/ for STAR_CACHE_TTL seconds.

    Raises
    ------
    ValueError
        If the name cannot be resolved.
    """
    cached = _read_star_cache(star_name)
    if cached is not None:
        ra_deg, dec_deg = cached
        logging.debug("Star cache hit for '%s'  RA=%.4f° Dec=%.4f°", star_name, ra_deg, dec_deg)
        return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    try:
        coord = SkyCoord.from_name(star_name)
    except Exception as exc:
        raise ValueError(f"Could not resolve star '{star_name}' (tried SIMBAD): {exc}")

    _write_star_cache(star_name, float(coord.ra.deg), float(coord.dec.deg))
    return coord


def star_apparent_eod(
    star_name: str,
    t: Time,
    location: EarthLocation,
) -> tuple[float, float, float]:
    """
    Compute topocentric apparent RA/Dec (epoch-of-date) for a star.

    Resolves the star name via SIMBAD, then applies topocentric transformation.

    Returns
    -------
    (ra_hours, dec_deg, alt_deg)
    """
    star_icrs = resolve_star(star_name)

    # Apply topocentric correction via AltAz → TETE
    altaz_frame = AltAz(obstime=t, location=location)
    star_altaz = star_icrs.transform_to(altaz_frame)
    star_eod = star_altaz.transform_to(TETE(obstime=t, location=location))

    return star_eod.ra.hour, star_eod.dec.deg, float(star_altaz.alt.deg)



def predict_next_threshold_crossing(
    satellite: Satrec,
    start_time: Time,
    location: EarthLocation,
    min_alt_deg: float,
    lookahead_sec: float,
    coarse_step_sec: float = 10.0,
) -> tuple[Time, float, float, float] | None:
    """
    Predict the next upward crossing of min_alt_deg within lookahead_sec.

    Returns
    -------
    (crossing_time, ra_hours, dec_deg, alt_deg_at_crossing) or None.
    """
    try:
        _ra0, _dec0, prev_alt = iss_apparent_eod(satellite, start_time, location)
        have_prev = True
    except RuntimeError:
        prev_alt = float("nan")
        have_prev = False

    prev_t = start_time
    steps = int(max(1.0, lookahead_sec / coarse_step_sec))

    for step_idx in range(1, steps + 1):
        t = start_time + (step_idx * coarse_step_sec) * u.s
        try:
            _ra, _dec, alt = iss_apparent_eod(satellite, t, location)
        except RuntimeError:
            continue

        if have_prev and prev_alt < min_alt_deg <= alt:
            lo = prev_t
            hi = t
            for _ in range(12):
                mid = Time((lo.jd + hi.jd) / 2.0, format="jd", scale="utc")
                try:
                    _mra, _mdec, mid_alt = iss_apparent_eod(satellite, mid, location)
                except RuntimeError:
                    # If refinement fails at this sample, keep searching coarsely.
                    mid_alt = min_alt_deg - 1.0
                if mid_alt >= min_alt_deg:
                    hi = mid
                else:
                    lo = mid

            try:
                rise_ra, rise_dec, rise_alt = iss_apparent_eod(satellite, hi, location)
            except RuntimeError:
                continue
            return hi, rise_ra, rise_dec, rise_alt

        prev_t = t
        prev_alt = alt
        have_prev = True

    return None


# ── INDI helpers ──────────────────────────────────────────────────────────────

def _poll_switch(
    client: _IndiClient,
    device_name: str,
    prop_name: str,
    timeout: float = 10.0,
) -> "PyIndi.ISwitchVectorProperty":
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        dev = client.getDevice(device_name)
        if dev:
            prop = dev.getSwitch(prop_name)
            if prop:
                return prop
        time.sleep(0.25)
    raise TimeoutError(
        f"Switch '{prop_name}' on '{device_name}' not available after {timeout:.0f}s"
    )


def _poll_number(
    client: _IndiClient,
    device_name: str,
    prop_name: str,
    timeout: float = 10.0,
) -> "PyIndi.INumberVectorProperty":
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        dev = client.getDevice(device_name)
        if dev:
            prop = dev.getNumber(prop_name)
            if prop:
                return prop
        time.sleep(0.25)
    raise TimeoutError(
        f"Number '{prop_name}' on '{device_name}' not available after {timeout:.0f}s"
    )


def _poll_text(
    client: _IndiClient,
    device_name: str,
    prop_name: str,
    timeout: float = 10.0,
) -> "PyIndi.ITextVectorProperty":
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        dev = client.getDevice(device_name)
        if dev:
            prop = dev.getText(prop_name)
            if prop:
                return prop
        time.sleep(0.25)
    raise TimeoutError(
        f"Text '{prop_name}' on '{device_name}' not available after {timeout:.0f}s"
    )


def _vector_count(vec) -> int:
    count_member = getattr(vec, "count", None)
    if callable(count_member):
        return int(count_member())
    return int(count_member)


def _probe_device_names_with_indi_getprop(host: str, port: int) -> list[str]:
    """Best-effort device discovery via indi_getprop."""
    try:
        result = subprocess.run(
            ["indi_getprop", "-h", host, "-p", str(port), "-t", "3"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []

    names: set[str] = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "." not in line:
            continue
        names.add(line.split(".", 1)[0].strip())

    return sorted((name for name in names if name), key=str.lower)


def _available_device_names(
    client: _IndiClient,
    host: str | None = None,
    port: int | None = None,
) -> list[str]:
    names: list[str] = []
    devices = client.getDevices()
    if not devices:
        devices = []

    for i in range(len(devices)):
        try:
            name = devices[i].getDeviceName()
        except Exception:
            continue
        if name:
            names.append(name)

    if host is not None and port is not None:
        names.extend(_probe_device_names_with_indi_getprop(host, port))

    return sorted(set(names), key=str.lower)


def _device_not_found_message(requested_name: str, available_names: list[str]) -> str:
    lines = [
        f"Device '{requested_name}' not found on the INDI server.",
    ]

    if available_names:
        close = difflib.get_close_matches(requested_name, available_names, n=3, cutoff=0.25)
        if close:
            lines.append(f"Did you mean: {', '.join(close)}")
        lines.append("Available devices:")
        lines.extend([f"  - {name}" for name in available_names])
    else:
        lines.append("No devices were reported by the INDI server.")

    lines.append("Tip: run 'indi_getprop' to verify driver/device names.")
    return "\n".join(lines)


def _indi_longitude_deg(lon_deg: float) -> float:
    """Convert east-positive longitude to INDI's 0..360 east-positive form."""
    return lon_deg % 360.0


def _call_with_timeout(func, timeout: float, action: str):
    """Run a potentially blocking call in a daemon thread with a hard timeout."""
    result: dict[str, object] = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            result["value"] = func()
        except BaseException as exc:
            result["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    if not done.wait(timeout):
        raise TimeoutError(f"{action} timed out after {timeout:.1f}s")

    if "error" in result:
        raise RuntimeError(f"{action} raised an exception") from result["error"]

    return result.get("value")


def _tcp_server_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """Quick TCP probe to distinguish network reachability from protocol issues."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_device(client: _IndiClient, device_name: str, timeout: float = 10.0):
    """Wait for a device to appear in the INDI device tree."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        device = client.getDevice(device_name)
        if device is not None:
            return device
        time.sleep(0.25)
    return None


def sync_mount_context(
    client: _IndiClient,
    device_name: str,
    lat_deg: float,
    lon_deg: float,
    elev_m: float,
    prop_timeout: float = 30.0,
) -> None:
    """Push current UTC time and observer location into the mount driver."""
    geographic = _poll_number(client, device_name, "GEOGRAPHIC_COORD", timeout=prop_timeout)
    for i in range(_vector_count(geographic)):
        elem = geographic[i]
        if elem.name == "LAT":
            elem.value = lat_deg
        elif elem.name == "LONG":
            elem.value = _indi_longitude_deg(lon_deg)
        elif elem.name == "ELEV":
            elem.value = elev_m
    client.sendNewNumber(geographic)

    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    time_utc = _poll_text(client, device_name, "TIME_UTC", timeout=prop_timeout)
    for i in range(_vector_count(time_utc)):
        elem = time_utc[i]
        if elem.name == "UTC":
            elem.text = utc_now
        elif elem.name == "OFFSET":
            elem.text = "0"
    client.sendNewText(time_utc)

    logging.info(
        "Synced mount context  lat=%.4f° lon=%.4f° elev=%.1fm utc=%s",
        lat_deg,
        lon_deg,
        elev_m,
        utc_now,
    )


def connect_indi(
    host: str,
    port: int,
    device_name: str,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
) -> _IndiClient:
    """Connect to the INDI server and bring up the named mount device."""
    client = _IndiClient()
    client.setServer(host, port)
    logging.info("Connecting to INDI server %s:%d …", host, port)
    tcp_probe_timeout = min(2.0, max(0.5, connect_timeout / 2.0))
    if not _tcp_server_reachable(host, port, timeout=tcp_probe_timeout):
        _fatal_exit(
            f"Cannot reach INDI server at {host}:{port} (TCP connect failed). "
            "Check host/port and that indiserver is running."
        )

    try:
        connected = bool(_call_with_timeout(
            client.connectServer,
            timeout=connect_timeout,
            action="connectServer()",
        ))
    except TimeoutError:
        _fatal_exit(
            f"Timed out waiting for INDI handshake at {host}:{port}. "
            "TCP is reachable, but connectServer() did not return. "
            "Try restarting indiserver/driver and rerun with --show-indi-stderr for native logs."
        )
    except RuntimeError as exc:
        _fatal_exit(f"Failed during connectServer(): {exc.__cause__ or exc}")

    if not connected:
        _fatal_exit(f"INDI refused connection at {host}:{port} (connectServer() returned False)")

    logging.info("INDI server handshake complete")

    client.watchDevice(device_name)
    logging.info("Watching device '%s' …", device_name)

    device = _wait_for_device(client, device_name, timeout=connect_timeout)
    if device is None:
        available = _available_device_names(client, host=host, port=port)
        _fatal_exit(_device_not_found_message(device_name, available))

    logging.info("Device '%s' discovered", device_name)

    # Some pyindi-client builds can crash when mutating the CONNECTION switch
    # vector via sendNewSwitch(). BaseClient.connectDevice() is the safer path.
    try:
        connect_device_ok = bool(_call_with_timeout(
            lambda: client.connectDevice(device_name),
            timeout=connect_timeout,
            action=f"connectDevice('{device_name}')",
        ))
    except TimeoutError:
        logging.warning(
            "connectDevice('%s') timed out; proceeding to property checks",
            device_name,
        )
        connect_device_ok = False
    except RuntimeError as exc:
        logging.warning(
            "connectDevice('%s') raised %s; proceeding to property checks",
            device_name,
            exc.__cause__ or exc,
        )
        connect_device_ok = False

    if not connect_device_ok:
        # connectDevice() can fail silently on remote INDI servers (BaseClientQt
        # can't locate the driver locally).  Fall back to sending the CONNECTION
        # switch property directly — this is the standard INDI protocol approach.
        logging.info(
            "connectDevice('%s') unavailable for remote server; sending CONNECTION switch …",
            device_name,
        )
        try:
            conn_svp = _poll_switch(client, device_name, "CONNECTION", timeout=connect_timeout)
            for i in range(_vector_count(conn_svp)):
                sw = conn_svp[i]
                if sw.name == "CONNECT":
                    sw.s = PyIndi.ISS_ON
                elif sw.name == "DISCONNECT":
                    sw.s = PyIndi.ISS_OFF
            client.sendNewSwitch(conn_svp)
            logging.info("CONNECTION → CONNECT sent to '%s'", device_name)
        except TimeoutError:
            logging.warning(
                "CONNECTION switch not available on '%s' after %.0fs; "
                "proceeding — properties may take longer to appear",
                device_name,
                connect_timeout,
            )
    else:
        logging.info("connectDevice('%s') sent", device_name)

    # Wait for the hardware connection to come up.
    # We check the CONNECTION vector *state* (IPS_OK / IPS_ALERT) rather than
    # the element value, because sendNewSwitch() already mutates the local
    # switch object to CONNECT=On before the driver has confirmed anything.
    # IPS_BUSY means the driver is still opening the serial port.
    # IPS_OK + CONNECT=On  → success.
    # IPS_ALERT            → driver reported an error.
    logging.info("Waiting for '%s' hardware connection …", device_name)
    hw_deadline = time.monotonic() + connect_timeout
    hw_connected = False
    hw_alert = False
    while time.monotonic() < hw_deadline:
        dev = client.getDevice(device_name)
        if dev is not None:
            conn_svp = dev.getSwitch("CONNECTION")
            if conn_svp is not None:
                vec_state = conn_svp.getState() if hasattr(conn_svp, "getState") else getattr(conn_svp, "s", None)
                if vec_state == PyIndi.IPS_OK:
                    for i in range(_vector_count(conn_svp)):
                        if conn_svp[i].name == "CONNECT" and conn_svp[i].s == PyIndi.ISS_ON:
                            hw_connected = True
                            break
                    break  # settled (Ok) — exit regardless
                elif vec_state == PyIndi.IPS_ALERT:
                    hw_alert = True
                    break
                # IPS_BUSY or IPS_IDLE → still connecting, keep polling
        time.sleep(0.5)

    # Gather port/baud info for diagnostics
    port_info = ""
    baud_info = ""
    dev = client.getDevice(device_name)
    if dev is not None:
        port_svp = dev.getText("DEVICE_PORT")
        if port_svp is not None:
            for i in range(_vector_count(port_svp)):
                if port_svp[i].name == "PORT":
                    port_info = port_svp[i].text
                    break
        baud_svp = dev.getSwitch("DEVICE_BAUD_RATE")
        if baud_svp is not None:
            for i in range(_vector_count(baud_svp)):
                if baud_svp[i].s == PyIndi.ISS_ON:
                    baud_info = baud_svp[i].name
                    break

    if hw_connected:
        logging.info(
            "Device '%s' hardware connection confirmed (port=%s baud=%s)",
            device_name, port_info, baud_info,
        )
    else:
        reason = "timed out" if not hw_alert else "driver reported connection error (IPS_ALERT)"
        _fatal_exit(
            f"Driver for '{device_name}' did not establish a hardware connection "
            f"({reason}) within {connect_timeout:.0f}s.\n"
            f"  Serial port : {port_info or '(unknown)'}\n"
            f"  Baud rate   : {baud_info or '(unknown)'}\n"
            "Possible causes:\n"
            "  • USB/serial cable not plugged into the Pi\n"
            "  • Wrong serial port — check available ports shown in the driver\n"
            "  • Baud rate mismatch — verify your firmware serial speed and match it in INDI\n"
            "    Example fixes:\n"
            "      indi_setprop -h {host} \"LX200 OpenAstroTech.DEVICE_BAUD_RATE.19200=On\"\n"
            "      indi_setprop -h {host} \"LX200 OpenAstroTech.DEVICE_BAUD_RATE.57600=On\"\n"
            "  • Pi user not in the 'dialout' group — fix: sudo adduser $USER dialout\n"
            "  • Mount is powered off"
        )

    return client


def set_on_coord_set_mode(client: _IndiClient, device_name: str, mode_name: str) -> None:
    """
    Configure ON_COORD_SET to one of the available modes (typically SLEW/TRACK/SYNC).
    """
    svp = _poll_switch(client, device_name, "ON_COORD_SET")
    mode_upper = mode_name.upper()
    found = False
    for i in range(_vector_count(svp)):
        sw = svp[i]
        is_target = sw.name.upper() == mode_upper
        sw.s = PyIndi.ISS_ON if is_target else PyIndi.ISS_OFF
        found = found or is_target

    if not found:
        available = ", ".join(svp[i].name for i in range(_vector_count(svp)))
        raise RuntimeError(
            f"ON_COORD_SET mode '{mode_name}' not available. Found: {available}"
        )

    client.sendNewSwitch(svp)
    logging.info("ON_COORD_SET → %s", mode_upper)


def set_tracking_state(client: _IndiClient, device_name: str, enabled: bool) -> None:
    """Best-effort control of TELESCOPE_TRACK_STATE (TRACK_ON / TRACK_OFF)."""
    try:
        svp = _poll_switch(client, device_name, "TELESCOPE_TRACK_STATE", timeout=3)
    except TimeoutError:
        logging.info("TELESCOPE_TRACK_STATE not available on this driver")
        return

    wanted = "TRACK_ON" if enabled else "TRACK_OFF"
    opposite = "TRACK_OFF" if enabled else "TRACK_ON"

    found_wanted = False
    for i in range(_vector_count(svp)):
        sw = svp[i]
        name_upper = sw.name.upper()
        if name_upper == wanted:
            sw.s = PyIndi.ISS_ON
            found_wanted = True
        elif name_upper == opposite:
            sw.s = PyIndi.ISS_OFF

    if not found_wanted:
        logging.info("TELESCOPE_TRACK_STATE missing %s switch; leaving unchanged", wanted)
        return

    client.sendNewSwitch(svp)
    logging.info("TELESCOPE_TRACK_STATE → %s", wanted)


def set_tracking_mode_sidereal(client: _IndiClient, device_name: str) -> None:
    """Best-effort set TELESCOPE_TRACK_MODE to sidereal."""
    try:
        svp = _poll_switch(client, device_name, "TELESCOPE_TRACK_MODE", timeout=3)
    except TimeoutError:
        logging.info("TELESCOPE_TRACK_MODE not available on this driver")
        return

    sidereal_names = {"TRACK_SIDEREAL", "SIDEREAL", "TRACKMODE_SIDEREAL"}
    found = False
    for i in range(_vector_count(svp)):
        sw = svp[i]
        is_sidereal = sw.name.upper() in sidereal_names
        sw.s = PyIndi.ISS_ON if is_sidereal else PyIndi.ISS_OFF
        found = found or is_sidereal

    if not found:
        logging.info("No sidereal mode switch found in TELESCOPE_TRACK_MODE")
        return

    client.sendNewSwitch(svp)
    logging.info("TELESCOPE_TRACK_MODE → SIDEREAL")


def set_tracking_mode_custom(client: _IndiClient, device_name: str) -> bool:
    """Best-effort set TELESCOPE_TRACK_MODE to custom. Returns True if applied."""
    try:
        svp = _poll_switch(client, device_name, "TELESCOPE_TRACK_MODE", timeout=3)
    except TimeoutError:
        logging.info("TELESCOPE_TRACK_MODE not available on this driver")
        return False

    custom_names = {"TRACK_CUSTOM", "CUSTOM", "TRACKMODE_CUSTOM"}
    found = False
    for i in range(_vector_count(svp)):
        sw = svp[i]
        is_custom = sw.name.upper() in custom_names
        sw.s = PyIndi.ISS_ON if is_custom else PyIndi.ISS_OFF
        found = found or is_custom

    if not found:
        logging.info("No custom mode switch found in TELESCOPE_TRACK_MODE")
        return False

    client.sendNewSwitch(svp)
    logging.info("TELESCOPE_TRACK_MODE → CUSTOM")
    return True


def _find_number_element_by_names(
    nvp: "PyIndi.INumberVectorProperty",
    candidate_names: tuple[str, ...],
) -> "PyIndi.INumber | None":
    for name in candidate_names:
        for i in range(_vector_count(nvp)):
            elem = nvp[i]
            if elem.name.upper() == name.upper():
                return elem
    return None


def detect_non_sidereal_rate_vector(
    client: _IndiClient,
    device_name: str,
) -> tuple[str, str, str] | None:
    """
    Detect a writable INDI number vector for RA/Dec tracking rates.

    Returns (property_name, ra_element_name, dec_element_name), or None.
    """
    dev = client.getDevice(device_name)
    if dev is None:
        return None

    candidates: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        ("TELESCOPE_TRACK_RATE", ("TRACK_RATE_RA", "RA_RATE", "RA"), ("TRACK_RATE_DEC", "TRACK_RATE_DE", "DEC_RATE", "DEC")),
        ("EQUATORIAL_TRACK_RATE", ("TRACK_RATE_RA", "RA_RATE", "RA"), ("TRACK_RATE_DEC", "TRACK_RATE_DE", "DEC_RATE", "DEC")),
        ("TRACK_RATE", ("TRACK_RATE_RA", "RA_RATE", "RA"), ("TRACK_RATE_DEC", "TRACK_RATE_DE", "DEC_RATE", "DEC")),
    ]

    for prop_name, ra_names, dec_names in candidates:
        nvp = dev.getNumber(prop_name)
        if nvp is None:
            continue
        ra_elem = _find_number_element_by_names(nvp, ra_names)
        dec_elem = _find_number_element_by_names(nvp, dec_names)
        if ra_elem is not None and dec_elem is not None:
            return prop_name, ra_elem.name, dec_elem.name

    return None


def _safe_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _encode_rate_for_element(
    rate_deg_per_sec: float,
    elem: "PyIndi.INumber",
) -> float:
    """
    Encode a physical angular rate for an INDI element.

    Heuristic:
    - wide element ranges (|max| > 5): expect arcsec/sec
    - narrow ranges: expect sidereal multiples
    """
    rate_arcsec_per_sec = rate_deg_per_sec * 3600.0
    elem_min = _safe_float(getattr(elem, "min", None), -1.0e9)
    elem_max = _safe_float(getattr(elem, "max", None), 1.0e9)
    max_abs = max(abs(elem_min), abs(elem_max))

    if max_abs <= 5.0:
        sidereal_arcsec_per_sec = 15.041067
        value = rate_arcsec_per_sec / sidereal_arcsec_per_sec
    else:
        value = rate_arcsec_per_sec

    return max(elem_min, min(elem_max, value))


def set_non_sidereal_rates(
    client: _IndiClient,
    device_name: str,
    rate_vector: tuple[str, str, str],
    ra_rate_deg_per_sec: float,
    dec_rate_deg_per_sec: float,
) -> bool:
    """Set non-sidereal RA/Dec tracking rates. Returns False if unavailable."""
    prop_name, ra_name, dec_name = rate_vector
    dev = client.getDevice(device_name)
    if dev is None:
        return False

    nvp = dev.getNumber(prop_name)
    if nvp is None:
        return False

    ra_set = False
    dec_set = False
    for i in range(_vector_count(nvp)):
        elem = nvp[i]
        if elem.name == ra_name:
            elem.value = _encode_rate_for_element(ra_rate_deg_per_sec, elem)
            ra_set = True
        elif elem.name == dec_name:
            elem.value = _encode_rate_for_element(dec_rate_deg_per_sec, elem)
            dec_set = True

    if not (ra_set and dec_set):
        return False

    client.sendNewNumber(nvp)
    return True


def _wrapped_ra_rate_deg_per_sec(
    ra_now_hours: float,
    ra_future_hours: float,
    dt_sec: float,
) -> float:
    ra_now_deg = ra_now_hours * 15.0
    ra_future_deg = ra_future_hours * 15.0
    diff = (ra_future_deg - ra_now_deg + 180.0) % 360.0 - 180.0
    return diff / dt_sec


def stop_mount_motion(client: _IndiClient, device_name: str) -> None:
    """
    Best-effort stop command for mount shutdown.

    Tries TELESCOPE_ABORT_MOTION first (if exposed by the driver), then turns
    tracking off via TELESCOPE_TRACK_STATE.
    """
    try:
        svp = _poll_switch(client, device_name, "TELESCOPE_ABORT_MOTION", timeout=2)
        sent_abort = False
        for i in range(_vector_count(svp)):
            sw = svp[i]
            if "ABORT" in sw.name.upper():
                sw.s = PyIndi.ISS_ON
                sent_abort = True
            else:
                sw.s = PyIndi.ISS_OFF

        if sent_abort:
            client.sendNewSwitch(svp)
            logging.info("TELESCOPE_ABORT_MOTION sent")
        else:
            logging.info("TELESCOPE_ABORT_MOTION has no ABORT switch; skipping")
    except TimeoutError:
        logging.info("TELESCOPE_ABORT_MOTION not available on this driver")
    except Exception as exc:
        logging.warning("Failed to send TELESCOPE_ABORT_MOTION: %s", exc)

    try:
        set_tracking_state(client, device_name, enabled=False)
    except Exception as exc:
        logging.warning("Failed to set tracking OFF during shutdown: %s", exc)


def send_eod_coordinates(
    client: _IndiClient,
    device_name: str,
    ra_hours: float,
    dec_deg: float,
) -> bool:
    """
    Send a new RA/Dec (epoch-of-date) to the mount.

    Returns False if the property is momentarily unavailable.
    """
    dev = client.getDevice(device_name)
    if dev is None:
        return False
    coord = dev.getNumber("EQUATORIAL_EOD_COORD")
    if coord is None:
        return False
    for i in range(_vector_count(coord)):
        elem = coord[i]
        if elem.name == "RA":
            elem.value = ra_hours
        elif elem.name == "DEC":
            elem.value = dec_deg
    client.sendNewNumber(coord)
    return True


def get_mount_reported_eod_coordinates(
    client: _IndiClient,
    device_name: str,
) -> tuple[float, float] | None:
    """Read the mount's currently reported epoch-of-date RA/Dec."""
    dev = client.getDevice(device_name)
    if dev is None:
        return None

    coord = dev.getNumber("EQUATORIAL_EOD_COORD")
    if coord is None:
        return None

    ra_hours = None
    dec_deg = None
    for i in range(_vector_count(coord)):
        elem = coord[i]
        if elem.name == "RA":
            ra_hours = float(elem.value)
        elif elem.name == "DEC":
            dec_deg = float(elem.value)

    if ra_hours is None or dec_deg is None:
        return None
    return ra_hours, dec_deg


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Track the ISS, a planet, or a star with an INDI-connected telescope mount.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    obs = p.add_argument_group("Observer location")
    obs.add_argument(
        "--lat", type=float, default=None,
        metavar="DEG",
        help="Latitude in decimal degrees (N positive). Optional if cached from a previous run; if provided, overrides cached value.",
    )
    obs.add_argument(
        "--lon", type=float, default=None,
        metavar="DEG",
        help="Longitude in decimal degrees (E positive). Optional if cached from a previous run; if provided, overrides cached value.",
    )
    obs.add_argument(
        "--elev", type=float, default=None,
        metavar="M",
        help="Elevation above sea level in metres. Optional if cached from a previous run; if provided, overrides cached value.",
    )

    indi = p.add_argument_group("INDI connection")
    indi.add_argument("--host", default="localhost", help="INDI server hostname or IP")
    indi.add_argument("--port", type=int, default=7624, help="INDI server port")
    indi.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT,
        metavar="SEC",
        help="Timeout for INDI handshake/device discovery stages",
    )
    indi.add_argument(
        "--device", required=True,
        help="INDI mount device name (e.g. 'EQMod Mount', 'Telescope Simulator')",
    )

    track = p.add_argument_group("Tracking behaviour")
    track.add_argument(
        "--target",
        type=str,
        default="iss",
        metavar="NAME",
        help="Object to track: iss (default), a planet name, or a star name (resolved via SIMBAD)",
    )
    track.add_argument(
        "--interval", type=float, default=DEFAULT_SAT_INTERVAL,
        metavar="SEC",
        help="Seconds between position update commands",
    )
    track.add_argument(
        "--min-alt", type=float, default=DEFAULT_MIN_ALT,
        dest="min_alt",
        metavar="DEG",
        help="Do not command the mount below this altitude",
    )
    track.add_argument(
        "--preposition-lookahead", type=float, default=DEFAULT_PREPOSITION_LOOKAHEAD,
        metavar="SEC",
        help="When ISS is below min-alt, predict next rise within this lookahead and pre-slew",
    )
    track.add_argument(
        "--preposition-alt", type=float, default=DEFAULT_PREPOSITION_ALT,
        metavar="DEG",
        help="Altitude used for pre-positioning before live tracking starts",
    )

    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    p.add_argument(
        "--show-indi-stderr",
        action="store_true",
        help="Show raw native INDI/PyIndi stderr messages",
    )
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_arg_parser().parse_args()

    cached_observer = _read_observer_cache()
    lat_deg = args.lat if args.lat is not None else (cached_observer[0] if cached_observer else None)
    lon_deg = args.lon if args.lon is not None else (cached_observer[1] if cached_observer else None)
    elev_m = args.elev if args.elev is not None else (cached_observer[2] if cached_observer else 0.0)

    if lat_deg is None or lon_deg is None:
        _fatal_exit(
            "Observer location is required on first run. Provide --lat and --lon at least once; "
            f"they will be cached in {OBSERVER_CACHE_FILE}."
        )

    _write_observer_cache(lat_deg, lon_deg, elev_m)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    configure_native_stderr(show_native_stderr=args.show_indi_stderr)

    if args.connect_timeout <= 0:
        _fatal_exit("--connect-timeout must be greater than 0 seconds")

    target = args.target.strip().lower()
    # Allow any target name at this point; we'll validate planets/stars during mode selection

    location = EarthLocation(
        lat=lat_deg * u.deg,
        lon=lon_deg * u.deg,
        height=elev_m * u.m,
    )

    # ── INDI connection ───────────────────────────────────────────────────────
    client = connect_indi(
        args.host,
        args.port,
        args.device,
        connect_timeout=args.connect_timeout,
    )

    try:
        sync_mount_context(
            client=client,
            device_name=args.device,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            elev_m=elev_m,
            prop_timeout=max(30.0, args.connect_timeout),
        )
    except TimeoutError as exc:
        client.disconnectServer()
        _fatal_exit(
            f"Could not sync mount context: {exc}\n"
            "The driver connected to the INDI server but mount hardware properties are not available.\n"
            "Check that the mount is powered on and its USB/serial cable is plugged into the Pi."
        )

    # Verify the EQUATORIAL_EOD_COORD property exists before we start the loop
    logging.info("Waiting for EQUATORIAL_EOD_COORD property …")
    try:
        _poll_number(client, args.device, "EQUATORIAL_EOD_COORD", timeout=15)
    except TimeoutError as exc:
        available = _available_device_names(client, host=args.host, port=args.port)
        client.disconnectServer()
        _fatal_exit(
            f"{exc}\n"
            f"Either the mount is not connected yet or '--device' is incorrect.\n"
            f"Known devices: {', '.join(available) if available else '(none reported)'}"
        )

    # ── Target resolution ────────────────────────────────────────────────────
    # Priority order: planet → explicit NORAD ID → Celestrak name search → SIMBAD star
    is_planet = target in SUPPORTED_PLANETS
    is_explicit_satellite = target == "iss" or target.isdigit()

    # For non-planets and non-explicit satellites, probe Celestrak first
    sat_display_name: str | None = None
    tle_url: str | None = None
    tle1_initial: str | None = None
    tle2_initial: str | None = None
    mode: str | None = None  # "planet" | "satellite" | "star"

    if is_planet:
        mode = "planet"
    elif is_explicit_satellite:
        mode = "satellite"
    else:
        # Try Celestrak name search
        try:
            sat_display_name, tle1_initial, tle2_initial, tle_url = resolve_satellite_tle(target)
            mode = "satellite"
            logging.info("Resolved '%s' as satellite: %s", target, sat_display_name)
        except ValueError:
            pass  # not found on Celestrak — try SIMBAD next

        if mode is None:
            # Try SIMBAD
            t_probe = Time(datetime.now(timezone.utc), scale="utc")
            try:
                _ra_probe, _dec_probe, _alt_probe = star_apparent_eod(target, t_probe, location)
                mode = "star"
            except ValueError:
                pass

        if mode is None:
            client.disconnectServer()
            _fatal_exit(
                f"Could not resolve '{target}' as a planet, satellite, or star.\n"
                f"  Planets: {', '.join(sorted(SUPPORTED_PLANETS))}\n"
                f"  Satellites: use 'iss', a NORAD ID, or a Celestrak name fragment\n"
                f"  Stars: any SIMBAD-resolvable name"
            )

    try:
        if mode == "planet":
            t = Time(datetime.now(timezone.utc), scale="utc")
            logging.info("Planet mode: target = '%s'", target)
            try:
                ra_h, dec_d, alt_d = planet_apparent_eod(target, t, location)
            except Exception as exc:
                client.disconnectServer()
                _fatal_exit(f"Failed to compute planet position: {exc}")

        elif mode == "star":
            t = Time(datetime.now(timezone.utc), scale="utc")
            logging.info("Star mode: resolving '%s' via SIMBAD …", target)
            try:
                ra_h, dec_d, alt_d = star_apparent_eod(target, t, location)
            except ValueError as exc:
                client.disconnectServer()
                _fatal_exit(str(exc))

        if mode in ("planet", "star"):

            logging.info(
                "Computed %s position  RA=%7.4fh  Dec=%+7.3f°  Alt=%+.2f°",
                target,
                ra_h,
                dec_d,
                alt_d,
            )

            if alt_d < args.min_alt:
                logging.warning(
                    "%s is below threshold  alt=%+.1f°  (threshold %.1f°)",
                    target.capitalize(),
                    alt_d,
                    args.min_alt,
                )

            set_on_coord_set_mode(client, args.device, "SLEW")
            ok = send_eod_coordinates(client, args.device, ra_h, dec_d)
            if not ok:
                logging.error("Could not send slew coordinates — property unavailable")
            else:
                reported = get_mount_reported_eod_coordinates(client, args.device)
                if reported is not None:
                    reported_ra_h, reported_dec_d = reported
                    logging.info(
                        "Mount reported  RA=%7.4fh  Dec=%+7.3f°",
                        reported_ra_h,
                        reported_dec_d,
                    )

            set_tracking_mode_sidereal(client, args.device)
            set_tracking_state(client, args.device, enabled=True)
            logging.info("Slew complete; sidereal tracking enabled on '%s'", target)
            while True:
                time.sleep(60)
                logging.info("Holding sidereal tracking on '%s'", target)

        # ── Satellite continuous tracking loop ───────────────────────────────
        # Start in SLEW mode so pre-positioning moves decisively to the rise point.
        # Switch to TRACK only once live updates begin above the threshold.
        set_on_coord_set_mode(client, args.device, "SLEW")
        set_tracking_state(client, args.device, enabled=False)

        rate_vector = detect_non_sidereal_rate_vector(client, args.device)
        use_rate_tracking = rate_vector is not None
        live_tracking_armed = False
        last_wait_log_time: float = 0.0
        if use_rate_tracking:
            prop_name, ra_name, dec_name = rate_vector
            logging.info(
            "Will use non-sidereal rate tracking via %s (%s,%s) once live tracking starts",
                prop_name,
                ra_name,
                dec_name,
            )
        else:
            logging.info(
                "No non-sidereal rate vector found; falling back to streamed coordinate tracking"
            )

        if is_explicit_satellite or tle_url is None:
            logging.info("Resolving satellite TLE for '%s' from Celestrak …", target)
            try:
                sat_display_name, tle1_initial, tle2_initial, tle_url = resolve_satellite_tle(target)
            except ValueError as exc:
                client.disconnectServer()
                _fatal_exit(str(exc))

        tle1, tle2 = tle1_initial, tle2_initial
        satellite = build_satellite(tle1, tle2)
        tle_fetched_at = time.monotonic()
        logging.info(
            "TLE loaded for %s — jdsatepoch=%.6f", sat_display_name, satellite.jdsatepoch
        )

        logging.info(
            "Satellite tracking loop started  target=%s  interval=%.3fs  min_alt=%.1f°  preposition_alt=%.1f°",
            sat_display_name,
            args.interval,
            args.min_alt,
            args.preposition_alt,
        )

        pending_rise: tuple[Time, float, float] | None = None
        next_display_rise: tuple[Time, float, float] | None = None
        next_tracking_rise_t: Time | None = None
        last_prediction_attempt = 0.0

        while True:
            loop_start = time.monotonic()

            if loop_start - tle_fetched_at >= TLE_REFRESH_INTERVAL:
                try:
                    tle1, tle2 = fetch_tle(tle_url)
                    satellite = build_satellite(tle1, tle2)
                    tle_fetched_at = loop_start
                    logging.info("TLE refreshed for %s", sat_display_name)
                except Exception as exc:
                    logging.warning("TLE refresh failed (%s) — using cached data", exc)

            t = Time(datetime.now(timezone.utc), scale="utc")

            try:
                ra_h, dec_d, alt_d = iss_apparent_eod(satellite, t, location)
            except RuntimeError as exc:
                logging.error("Position error: %s", exc)
                time.sleep(args.interval)
                continue

            if alt_d < args.min_alt:
                if pending_rise is None and (loop_start - last_prediction_attempt) >= PREDICTION_RETRY_INTERVAL:
                    last_prediction_attempt = loop_start

                    tracking_predicted = predict_next_threshold_crossing(
                        satellite=satellite,
                        start_time=t,
                        location=location,
                        min_alt_deg=args.min_alt,
                        lookahead_sec=args.preposition_lookahead,
                    )
                    next_tracking_rise_t = tracking_predicted[0] if tracking_predicted is not None else None

                    display_predicted = predict_next_threshold_crossing(
                        satellite=satellite,
                        start_time=t,
                        location=location,
                        min_alt_deg=args.preposition_alt,
                        lookahead_sec=max(args.preposition_lookahead, DISPLAY_RISE_LOOKAHEAD),
                        coarse_step_sec=30.0,
                    )
                    if display_predicted is None:
                        display_predicted = predict_next_threshold_crossing(
                            satellite=satellite,
                            start_time=t,
                            location=location,
                            min_alt_deg=args.preposition_alt,
                            lookahead_sec=max(args.preposition_lookahead, DISPLAY_RISE_LOOKAHEAD_MAX),
                            coarse_step_sec=60.0,
                        )
                    if display_predicted is not None:
                        display_rise_t, display_rise_ra_h, display_rise_dec_d, _display_rise_alt = display_predicted
                        next_display_rise = (display_rise_t, display_rise_ra_h, display_rise_dec_d)
                    else:
                        next_display_rise = None

                    predicted = predict_next_threshold_crossing(
                        satellite=satellite,
                        start_time=t,
                        location=location,
                        min_alt_deg=args.preposition_alt,
                        lookahead_sec=args.preposition_lookahead,
                    )
                    if predicted is not None:
                        rise_t, rise_ra_h, rise_dec_d, _rise_alt = predicted
                        eta_sec = float((rise_t - t).to_value(u.s))
                        logging.info(
                            "Pre-positioning mount for %s appearance at %.1f°  ETA=%.0fs  target RA=%7.4fh Dec=%+7.3f°",
                            sat_display_name,
                            args.preposition_alt,
                            eta_sec,
                            rise_ra_h,
                            rise_dec_d,
                        )
                        ok = send_eod_coordinates(client, args.device, rise_ra_h, rise_dec_d)
                        if ok:
                            logging.info(
                                "Preposition slew sent for %s  RA=%7.4fh  Dec=%+7.3f°",
                                sat_display_name,
                                rise_ra_h,
                                rise_dec_d,
                            )
                            reported = get_mount_reported_eod_coordinates(client, args.device)
                            if reported is not None:
                                reported_ra_h, reported_dec_d = reported
                                logging.info(
                                    "Mount reported  RA=%7.4fh  Dec=%+7.3f°",
                                    reported_ra_h,
                                    reported_dec_d,
                                )
                            pending_rise = (rise_t, rise_ra_h, rise_dec_d)
                        else:
                            logging.warning("Could not pre-position mount — property unavailable")
                    else:
                        logging.info(
                            "No %s rise above %.1f° found in next %.0fs",
                            sat_display_name,
                            args.preposition_alt,
                            args.preposition_lookahead,
                        )
                        pending_rise = None

                threshold_eta_sec: float | None = None
                if next_tracking_rise_t is not None:
                    threshold_eta_sec = float((next_tracking_rise_t - t).to_value(u.s))
                    if threshold_eta_sec < -120:
                        next_tracking_rise_t = None
                        threshold_eta_sec = None

                preposition_eta_sec: float | None = None
                if pending_rise is not None:
                    rise_t, _rise_ra_h, _rise_dec_d = pending_rise
                    preposition_eta_sec = float((rise_t - t).to_value(u.s))
                    if preposition_eta_sec < -120:
                        pending_rise = None
                        preposition_eta_sec = None

                display_eta_sec: float | None = None
                if next_display_rise is not None:
                    display_rise_t, _display_rise_ra_h, _display_rise_dec_d = next_display_rise
                    display_eta_sec = float((display_rise_t - t).to_value(u.s))
                    if display_eta_sec < -120:
                        next_display_rise = None
                        display_eta_sec = None

                _now_mono = time.monotonic()
                if _now_mono - last_wait_log_time >= 10.0:
                    last_wait_log_time = _now_mono
                    if threshold_eta_sec is not None:
                        if pending_rise is not None:
                            rise_t, rise_ra_h, rise_dec_d = pending_rise
                            logging.info(
                                "%s below threshold  alt=%+.1f°  waiting for %.1f° rise ETA=%.0fs  preposition ETA=%.0fs (RA=%7.4fh Dec=%+7.3f°)",
                                sat_display_name,
                                alt_d,
                                args.min_alt,
                                max(0.0, threshold_eta_sec),
                                max(0.0, preposition_eta_sec),
                                rise_ra_h,
                                rise_dec_d,
                            )
                        else:
                            logging.info(
                                "%s below threshold  alt=%+.1f°  waiting for %.1f° rise ETA=%.0fs",
                                sat_display_name,
                                alt_d,
                                args.min_alt,
                                max(0.0, threshold_eta_sec),
                            )
                    elif pending_rise is not None and preposition_eta_sec is not None:
                        rise_t, rise_ra_h, rise_dec_d = pending_rise
                        logging.info(
                            "%s below threshold  alt=%+.1f°  waiting for rise ETA=%.0fs  preposition target RA=%7.4fh Dec=%+7.3f°",
                            sat_display_name,
                            alt_d,
                            max(0.0, preposition_eta_sec),
                            rise_ra_h,
                            rise_dec_d,
                        )
                    elif next_display_rise is not None and display_eta_sec is not None:
                        display_rise_t, display_rise_ra_h, display_rise_dec_d = next_display_rise
                        logging.info(
                            "%s below threshold  alt=%+.1f°  waiting for rise ETA=%.0fs  target RA=%7.4fh Dec=%+7.3f°",
                            sat_display_name,
                            alt_d,
                            max(0.0, display_eta_sec),
                            display_rise_ra_h,
                            display_rise_dec_d,
                        )
                    else:
                        logging.info(
                            "%s below threshold  alt=%+.1f°  waiting for %.1f° rise (ETA unknown)",
                            sat_display_name,
                            alt_d,
                            args.min_alt,
                        )
                time.sleep(args.interval)
                continue

            if pending_rise is not None:
                logging.info("%s reached tracking threshold; switching to live updates", sat_display_name)
                pending_rise = None
            next_display_rise = None
            next_tracking_rise_t = None

            if not live_tracking_armed:
                set_on_coord_set_mode(client, args.device, "TRACK")
                if use_rate_tracking:
                    set_tracking_mode_custom(client, args.device)
                    set_tracking_state(client, args.device, enabled=True)
                else:
                    set_tracking_state(client, args.device, enabled=False)
                live_tracking_armed = True
                logging.info("Live satellite tracking armed on '%s'", sat_display_name)

            logging.info(
                "Commanding mount  RA=%7.4fh  Dec=%+7.3f°  Alt=%+.2f°",
                ra_h,
                dec_d,
                alt_d,
            )

            ok = send_eod_coordinates(client, args.device, ra_h, dec_d)
            if not ok:
                logging.warning("Could not send coordinates — property unavailable")
            else:
                reported = get_mount_reported_eod_coordinates(client, args.device)
                if reported is not None:
                    reported_ra_h, reported_dec_d = reported
                    logging.info(
                        "Mount reported  RA=%7.4fh  Dec=%+7.3f°",
                        reported_ra_h,
                        reported_dec_d,
                    )

            if use_rate_tracking and rate_vector is not None:
                t_future = t + (RATE_ESTIMATE_DT * u.s)
                try:
                    ra_future_h, dec_future_d, _alt_future = iss_apparent_eod(satellite, t_future, location)
                    ra_rate_deg_s = _wrapped_ra_rate_deg_per_sec(
                        ra_now_hours=ra_h,
                        ra_future_hours=ra_future_h,
                        dt_sec=RATE_ESTIMATE_DT,
                    )
                    dec_rate_deg_s = (dec_future_d - dec_d) / RATE_ESTIMATE_DT
                    if not set_non_sidereal_rates(
                        client=client,
                        device_name=args.device,
                        rate_vector=rate_vector,
                        ra_rate_deg_per_sec=ra_rate_deg_s,
                        dec_rate_deg_per_sec=dec_rate_deg_s,
                    ):
                        logging.warning("Rate vector became unavailable; continuing with coordinate streaming")
                        use_rate_tracking = False
                        set_tracking_state(client, args.device, enabled=False)
                except Exception as exc:
                    logging.warning("Rate update failed (%s); continuing with coordinate streaming", exc)
                    use_rate_tracking = False
                    set_tracking_state(client, args.device, enabled=False)

            elapsed = time.monotonic() - loop_start
            remaining = args.interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        logging.info("Interrupted by user — stopping tracker")
    finally:
        stop_mount_motion(client, args.device)
        client.disconnectServer()
        logging.info("INDI connection closed")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001
        # Ensure unhandled exceptions are always visible even when stderr is
        # redirected to /dev/null by configure_native_stderr().
        import traceback as _tb
        _msg = "".join(_tb.format_exception(type(_exc), _exc, _exc.__traceback__))
        if logging.getLogger().handlers:
            logging.critical("Unhandled exception:\n%s", _msg)
        else:
            print(_msg, file=sys.stdout, flush=True)
        raise SystemExit(1)
