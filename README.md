# F1 Lap Time Simulator – Spa-Francorchamps 🏎️

A Python-based simulator that visualizes the fastest lap of an F1 driver using official FastF1 telemetry data. This project animates the lap on the track, showing speed, sector, and cumulative distance information in real-time.

---

## Features

- Animate the fastest lap of any F1 driver
- Display live telemetry data: speed, lap time, sector, and distance
- Color-coded sectors for better visualization (for Belgium):
  - Sector 1: Red
  - Sector 2: Blue
  - Sector 3: Yellow
- Scales GPS data to official track length
- Built with Python and FastF1 API

---

## Requirements

- Python 3.8+
- `fastf1`
- `numpy`
- `matplotlib`

You can install the required packages via pip:

```bash
pip install fastf1 numpy matplotlib
