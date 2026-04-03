# OAT Tracker

Python tools for OpenAstroTracker mount control and satellite pass planning.

This project contains:

- `oat-tracker.py`: live tracking of satellites, planets, and stars via INDI
- `find-passes.py`: quick upcoming-pass finder using Celestrak TLE data

## Features

- INDI mount control using `EQUATORIAL_EOD_COORD`
- Satellite tracking with continuous coordinate streaming
- Pre-position slew before satellite rise
- Observer location cache (`~/.cache/oat-tracker/observer.json`)
- TLE and star lookup caching
- Clear connection diagnostics (timeouts, device mismatch, hardware link state)

## Requirements

- Python 3.10+
- An INDI server and mount driver (tested with `indi_lx200_OpenAstroTech`)
- Dependencies listed in `requirements.txt`

Install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1) Track ISS through INDI

```bash
venv/bin/python oat-tracker.py \
  --lat <lat> --lon <lon> --elev <elev> \
  --host <host> --port 7624 \
  --device "LX200 OpenAstroTech" \
  --target iss
```

After first run, location is cached, so later runs can omit `--lat/--lon/--elev`.

### 2) Find upcoming passes

```bash
venv/bin/python find-passes.py --lat <lat> --lon <lon> --elev <elev> --min-alt 20
```

## Main CLI Options (`oat-tracker.py`)

- `--target`: `iss` (default), a NORAD ID, a Celestrak name fragment, a planet, or a SIMBAD-resolvable star
- `--host`, `--port`, `--device`: INDI connection settings
- `--connect-timeout`: timeout (seconds) for INDI handshake/device stages
- `--interval`: update period in seconds (default satellite cadence: `0.3`)
- `--min-alt`: do not command mount below this altitude
- `--preposition-lookahead`: search window for next rise when below threshold
- `--preposition-alt`: altitude used for pre-position slew
- `--show-indi-stderr`: show native INDI/PyIndi stderr output
- `--verbose`: debug logging

## Driver Serial Settings (Pi)

For the OAT setup validated in this repo, these settings were working:

- Port: `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0` (or `/dev/ttyUSB0`)
- Baud: `19200`

Set through INDI tools:

```bash
indi_setprop -h localhost -p 7624 "LX200 OpenAstroTech.DEVICE_PORT.PORT=/dev/ttyUSB0"
indi_setprop -h localhost -p 7624 "LX200 OpenAstroTech.DEVICE_BAUD_RATE.19200=On"
indi_setprop -h localhost -p 7624 "LX200 OpenAstroTech.CONNECTION.CONNECT=On"
indi_setprop -h localhost -p 7624 "LX200 OpenAstroTech.CONFIG_PROCESS.CONFIG_SAVE=On"
```

## Troubleshooting

### Connect succeeds but script exits early

Run with:

```bash
venv/bin/python oat-tracker.py ... --show-indi-stderr
```

This reveals native driver errors hidden in quiet mode.

### `Device ... did not establish a hardware connection`

Check:

- Mount power is on
- USB cable and port are correct
- Driver port and baud match firmware
- User is in `dialout` group on Indi server


## Notes

- Cache files live under `~/.cache/oat-tracker/`.
