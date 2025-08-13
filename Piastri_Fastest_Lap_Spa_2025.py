"""
F1 Fastest Lap Visualization – Spa-Francorchamps

This script uses FastF1 telemetry data to animate the fastest lap of a given driver,
displaying speed, sector, and distance information on a track map.

Requirements:
    - fastf1
    - numpy
    - matplotlib

Data Source:
    Official F1 timing data provided via the FastF1 API.

Author: (Malek El Kahza)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use 'Agg' for export without GUI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.collections as mcoll
import fastf1
from fastf1 import plotting

# =============================================================================
# 1) Session Setup
# =============================================================================
plotting.setup_mpl()
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')  # Enable local caching to reduce API calls

YEAR = 2025
GRAND_PRIX = 'Belgium'
SESSION_TYPE = 'Q'
DRIVER = 'PIA'  # Driver code according to official F1 timing (e.g., 'PIA' = Oscar Piastri)

# =============================================================================
# 2) Load Session Data
# =============================================================================
print(f"Loading session: {YEAR} {GRAND_PRIX} {SESSION_TYPE}...")

session = fastf1.get_session(YEAR, GRAND_PRIX, SESSION_TYPE)
session.load()

laps = session.laps.pick_drivers([DRIVER])
if laps.empty:
    raise ValueError(f"No laps found for driver '{DRIVER}'.")

# Get fastest lap telemetry
lap = laps.pick_fastest()
tel = lap.get_telemetry()

# =============================================================================
# 3) GPS & Telemetry Preprocessing
# =============================================================================
mask = ~np.isnan(tel['X']) & ~np.isnan(tel['Y'])
x_orig = tel['X'][mask].to_numpy()
y_orig = tel['Y'][mask].to_numpy()
speed = tel['Speed'][mask].to_numpy()
time = tel['Time'][mask].to_numpy()

# Convert absolute time to seconds from lap start
time_seconds = (time - time[0]).astype('timedelta64[ms]').astype(float) / 1000
if len(x_orig) == 0:
    raise ValueError("No valid GPS data found.")

# Rotate track coordinates by 90° CCW for better display
x = -y_orig
y = x_orig

# Convert mm → m
x_m = x / 1000
y_m = y / 1000

# Compute cumulative distance along lap
distances_xy = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
cumulative_distance = np.insert(np.cumsum(distances_xy), 0, 0)

# Scale to official track length (Spa: 7004 m)
scale_factor = 7004 / cumulative_distance[-1]
cumulative_distance *= scale_factor

# =============================================================================
# 4) Sector Information
# =============================================================================
sector1_end = lap['Sector1Time'].total_seconds()
sector2_end = lap['Sector2Time'].total_seconds()

# Spa-specific GPS coordinate for sector 2 end
sector2_end_x_m = 1.444e+04 / 1000
sector2_end_y_m = -3.62e+03 / 1000
distances = np.sqrt((x_m - sector2_end_x_m)**2 + (y_m - sector2_end_y_m)**2)
idx_sector2_end = np.argmin(distances)

# Assign colors by sector
colors = np.empty(len(x), dtype=object)
for i in range(len(x)):
    if time_seconds[i] <= sector1_end:
        colors[i] = '#FF4136'  # Sector 1 – Red
    elif i <= idx_sector2_end:
        colors[i] = '#0074D9'  # Sector 2 – Blue
    else:
        colors[i] = '#FFDC00'  # Sector 3 – Yellow

# =============================================================================
# 5) Downsampling for Performance
# =============================================================================
downsample_factor = 2
x = x[::downsample_factor]
y = y[::downsample_factor]
speed = speed[::downsample_factor]
time_seconds = time_seconds[::downsample_factor]
cumulative_distance = cumulative_distance[::downsample_factor]
colors = colors[::downsample_factor]
idx_sector2_end //= downsample_factor

# =============================================================================
# 6) Plot Setup
# =============================================================================
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, color='lightgray', alpha=0.3)
lc = mcoll.LineCollection(segments, colors=colors[:-1], linewidths=2)
ax.add_collection(lc)

point, = ax.plot([], [], 'go', markersize=8, label=f'{DRIVER} – Fastest Lap')

# On-screen telemetry texts
speed_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, va='top')
lap_time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10, va='top')
sector_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=10, va='top')
track_text = ax.text(0.02, 0.80, '', transform=ax.transAxes, fontsize=10, va='top')

# Formatting
ax.set_xlim(min(x) - 50, max(x) + 50)
ax.set_ylim(min(y) - 50, max(y) + 50)
ax.set_title(f"{DRIVER} – Fastest Lap ({GRAND_PRIX} {YEAR} {SESSION_TYPE})")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend()

# =============================================================================
# 7) Animation Update Function
# =============================================================================
def format_time(t):
    """Format time in M:SS.ss or SS.ss format."""
    if t is None:
        return "---"
    m = int(t // 60)
    s = t % 60
    return f"{m}:{s:05.2f}" if m > 0 else f"{s:05.2f}"

def update(frame):
    """Animation frame update."""
    point.set_data([x[frame]], [y[frame]])
    speed_text.set_text(f"Speed: {speed[frame]:.0f} km/h")
    lap_time_text.set_text(f"Lap Time: {format_time(time_seconds[frame])}")

    if time_seconds[frame] <= sector1_end:
        sector_text.set_text("Sector: 1")
    elif frame <= idx_sector2_end:
        sector_text.set_text("Sector: 2")
    else:
        sector_text.set_text("Sector: 3")

    track_text.set_text(f"Distance: {cumulative_distance[frame]:.1f} m")
    return point, speed_text, lap_time_text, sector_text, track_text

# =============================================================================
# 8) Run Animation in Matplotlib Window
# =============================================================================
fps_display = 14
speed_factor = 1.0
interval_ms = (60 / fps_display) / speed_factor

ani = FuncAnimation(fig, update, frames=len(x), interval=interval_ms, blit=True)
plt.show()
