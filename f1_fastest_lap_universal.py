"""
F1 Fastest Lap Visualization – Universal Circuit Support with Real-Time Option

This script uses FastF1 telemetry data to animate the fastest lap of any F1 driver
on any circuit, displaying real-time speed, sector timing, and distance information
on an interactive track map with dynamic sector boundaries and official track lengths.

Features:
    - Support for all F1 circuits with official track lengths
    - Dynamic sector boundary calculation using multi-driver data
    - Interactive session selection (year, circuit, session type, driver)
    - Real-time telemetry animation with speed and sector information
    - Automatic coordinate rotation based on circuit layout
    - Performance optimization with intelligent downsampling
    - Choice between real-time and accelerated animation playback

Requirements:
    - fastf1
    - numpy
    - matplotlib

Data Source:
    Official F1 timing data provided via the FastF1 API.

Important Note:
    Sector boundaries are calculated using averaged timing data from multiple drivers
    and may not match the exact official FIA sector timing points. The visualization
    provides an approximation for educational and entertainment purposes.

Original Author: Malek El Kahza
Enhanced by: Víctor Vega Sobral (@VforVitorio)
Contributions: Universal circuit support, dynamic sector calculation, 
               official track lengths integration, improved user interface,
               real-time playback option
"""

from fastf1 import plotting
import fastf1
import matplotlib.collections as mcoll
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use 'Agg' for export without GUI

# =============================================================================
# Official Track Lengths (meters)
# =============================================================================
OFFICIAL_TRACK_LENGTHS = {
    'Belgium': 7004,      # Spa-Francorchamps
    'Monaco': 3337,       # Circuit de Monaco
    'Italy': 5793,        # Monza
    'Bahrain': 5412,      # Bahrain International Circuit
    'Spain': 4675,        # Circuit de Barcelona-Catalunya
    'Austria': 4318,      # Red Bull Ring
    'Britain': 5891,      # Silverstone
    'Hungary': 4381,      # Hungaroring
    'Netherlands': 4259,  # Circuit Zandvoort
    'Singapore': 5063,    # Marina Bay Street Circuit
    'Japan': 5807,        # Suzuka International Racing Course
    'Qatar': 5380,        # Lusail International Circuit
    'United States': 5513,  # Circuit of the Americas
    'Mexico': 4304,       # Autódromo Hermanos Rodríguez
    'Brazil': 4309,       # Interlagos
    'Las Vegas': 6201,    # Las Vegas Strip Circuit
    'Abu Dhabi': 5281,    # Yas Marina Circuit
    'Australia': 5278,    # Albert Park
    'Saudi Arabia': 6174,  # Jeddah Corniche Circuit
    'Miami': 5412,        # Miami International Autodrome
    'Emilia Romagna': 4909,  # Imola
    'Canada': 4361,       # Circuit Gilles Villeneuve
    'Azerbaijan': 6003,   # Baku City Circuit
    'China': 5451,        # Shanghai International Circuit
}

# =============================================================================
# 1) Session Setup
# =============================================================================
plotting.setup_mpl()
os.makedirs('cache', exist_ok=True)
fastf1.Cache.enable_cache('cache')  # Enable local caching to reduce API calls

# Interactive input for session parameters
print("=== F1 Fastest Lap Visualization Setup ===")
YEAR = int(input("Enter year (e.g., 2025): "))
GRAND_PRIX = input("Enter Grand Prix name (e.g., Belgium): ")
SESSION_TYPE = input(
    "Enter session type (Q for Qualifying, R for Race): ").upper()
DRIVER = input("Enter driver code (e.g., PIA, VER, HAM): ").upper()

# New: Animation speed selection
print("\n=== Animation Speed Options ===")
print("1. Real-time (same duration as actual lap)")
print("2. Accelerated (faster playback for quick viewing)")
speed_choice = input("Choose animation speed (1 or 2): ").strip()
REAL_TIME_MODE = speed_choice == "1"

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

# Display lap time for reference
lap_time_str = str(lap['LapTime']).split()[-1]  # Extract time portion
print(f"Fastest lap time: {lap_time_str}")

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

# Get circuit info for dynamic rotation
circuit_info = session.get_circuit_info()
rotation_angle = circuit_info.rotation if hasattr(
    circuit_info, 'rotation') else 0
print(f"Applying circuit rotation: {rotation_angle}°")

# Apply dynamic rotation based on circuit info
angle_rad = np.radians(rotation_angle)
cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

# Rotate coordinates
x = x_orig * cos_a - y_orig * sin_a
y = x_orig * sin_a + y_orig * cos_a

# Convert mm → m
x_m = x / 1000
y_m = y / 1000

# Compute cumulative distance along lap
distances_xy = np.sqrt(np.diff(x_m)**2 + np.diff(y_m)**2)
cumulative_distance = np.insert(np.cumsum(distances_xy), 0, 0)

# Get official track length (replaces hardcoded 7004)
try:
    track_len = OFFICIAL_TRACK_LENGTHS[GRAND_PRIX]
    print(f"Official track length for {GRAND_PRIX}: {track_len} m")
except KeyError:
    # Fallback to calculated length if circuit not in dict
    track_len = cumulative_distance[-1]
    print(
        f"Warning: {GRAND_PRIX} not in official lengths. Using calculated: {track_len:.1f} m")

# Scale to official track length
scale_factor = track_len / cumulative_distance[-1]
cumulative_distance *= scale_factor

# =============================================================================
# 4) Sector Information (Dynamic calculation using averaged timing data)
# =============================================================================
# NOTE: Future Enhancement Opportunity
# The most accurate approach would be to create a dictionary with exact GPS coordinates
# for sector start/end points for each circuit (similar to OFFICIAL_TRACK_LENGTHS).
# This would eliminate timing-based approximations and provide 100% accuracy.
# However, FIA does not publicly release these coordinates, and alternative sources
# (OpenF1, FastF1, GitHub repos) only provide track outlines or marshal zones,
# not the precise timing sector boundaries used for official lap timing.
# Current time-ratio approach provides ~95% accuracy as a practical compromise.

# Calculate sector distances using average ratios from all drivers
print("Calculating sector boundaries from all drivers...")

# Get fastest lap per driver to avoid outliers
# NOTE: Filter only drivers with valid lap times to prevent empty sequence errors
# Some drivers may not have completed valid laps in the session
valid_drivers = session.laps.dropna(subset=['LapTime']).groupby('Driver')
fastest_laps = valid_drivers.apply(
    lambda x: x.loc[x['LapTime'].idxmin()]).reset_index(drop=True)

# Filter valid laps with complete sector times
valid_laps = fastest_laps.dropna(
    subset=['Sector1Time', 'Sector2Time', 'LapTime'])

if len(valid_laps) > 0:
    # Calculate average sector time ratios across all drivers
    sector1_times = valid_laps['Sector1Time'].dt.total_seconds()
    sector2_times = valid_laps['Sector2Time'].dt.total_seconds()
    lap_times = valid_laps['LapTime'].dt.total_seconds()

    # Average ratios
    avg_sector1_ratio = (sector1_times / lap_times).mean()
    avg_sector2_ratio = ((sector1_times + sector2_times) / lap_times).mean()

    print(
        f"Average sector ratios: S1={avg_sector1_ratio:.3f}, S2={avg_sector2_ratio:.3f}")
else:
    # Fallback to single lap if no valid multi-driver data
    print("Using single lap data as fallback...")
    sector1_time = lap['Sector1Time'].total_seconds()
    sector2_time = lap['Sector2Time'].total_seconds()
    total_time = lap['LapTime'].total_seconds()

    avg_sector1_ratio = sector1_time / total_time
    avg_sector2_ratio = (sector1_time + sector2_time) / total_time

# Convert to actual distances using averaged ratios
sector1_distance = avg_sector1_ratio * track_len
sector2_distance = avg_sector2_ratio * track_len

print(
    f"Sector boundaries: S1={sector1_distance:.1f}m, S2={sector2_distance:.1f}m, S3={track_len:.1f}m")

# Assign colors by distance using averaged boundaries
colors = np.empty(len(cumulative_distance), dtype=object)
for i in range(len(cumulative_distance)):
    if cumulative_distance[i] <= sector1_distance:
        colors[i] = '#FF4136'  # Sector 1 – Red
    elif cumulative_distance[i] <= sector2_distance:
        colors[i] = '#0074D9'  # Sector 2 – Blue
    else:
        colors[i] = '#FFDC00'  # Sector 3 – Yellow

# Update sector display logic for animation
idx_sector1_end = np.where(cumulative_distance <= sector1_distance)[
    0][-1] if len(np.where(cumulative_distance <= sector1_distance)[0]) > 0 else 0
idx_sector2_end = np.where(cumulative_distance <= sector2_distance)[
    0][-1] if len(np.where(cumulative_distance <= sector2_distance)[0]) > 0 else 0

# =============================================================================
# 5) Intelligent Downsampling and Frame Interpolation
# =============================================================================
# Reduce downsampling for smoother animation (was 2, now adaptive)
min_frames = 1500  # Minimum frames for smooth animation
original_frames = len(x)

if original_frames > min_frames:
    downsample_factor = max(1, original_frames // min_frames)
else:
    downsample_factor = 1  # No downsampling if already low frame count

print(
    f"Original frames: {original_frames}, Downsample factor: {downsample_factor}")

x = x[::downsample_factor]
y = y[::downsample_factor]
speed = speed[::downsample_factor]
time_seconds = time_seconds[::downsample_factor]
cumulative_distance = cumulative_distance[::downsample_factor]
colors = colors[::downsample_factor]
idx_sector2_end //= downsample_factor

# =============================================================================
# 6) Frame Interpolation for Smooth Real-Time Animation
# =============================================================================


def interpolate_data_for_smooth_animation(x, y, speed, time_seconds, cumulative_distance, colors, target_fps=30):
    """Interpolate frames to achieve target FPS while maintaining accurate timing."""
    total_time = time_seconds[-1]
    current_frames = len(x)
    needed_frames = int(total_time * target_fps)

    if needed_frames <= current_frames:
        return x, y, speed, time_seconds, cumulative_distance, colors

    print(
        f"Interpolating from {current_frames} to {needed_frames} frames for {target_fps} FPS")

    # Create new time array with target FPS
    new_time = np.linspace(0, total_time, needed_frames)

    # Interpolate all data arrays
    new_x = np.interp(new_time, time_seconds, x)
    new_y = np.interp(new_time, time_seconds, y)
    new_speed = np.interp(new_time, time_seconds, speed)
    new_distance = np.interp(new_time, time_seconds, cumulative_distance)

    # Interpolate colors (assign nearest color)
    color_indices = np.searchsorted(time_seconds, new_time, side='left')
    color_indices = np.clip(color_indices, 0, len(colors) - 1)
    new_colors = colors[color_indices]

    return new_x, new_y, new_speed, new_time, new_distance, new_colors


# =============================================================================
# 7) Plot Setup
# =============================================================================
# Apply frame interpolation for real-time mode to ensure smooth animation
if REAL_TIME_MODE:
    x, y, speed, time_seconds, cumulative_distance, colors = interpolate_data_for_smooth_animation(
        x, y, speed, time_seconds, cumulative_distance, colors, target_fps=30)
    # Recalculate sector indices after interpolation
    idx_sector1_end = np.where(cumulative_distance <= sector1_distance)[
        0][-1] if len(np.where(cumulative_distance <= sector1_distance)[0]) > 0 else 0
    idx_sector2_end = np.where(cumulative_distance <= sector2_distance)[
        0][-1] if len(np.where(cumulative_distance <= sector2_distance)[0]) > 0 else 0

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, color='lightgray', alpha=0.3)
lc = mcoll.LineCollection(segments, colors=colors[:-1], linewidths=2)
ax.add_collection(lc)

point, = ax.plot([], [], 'go', markersize=8, label=f'{DRIVER} – Fastest Lap')

# On-screen telemetry texts
speed_text = ax.text(
    0.02, 0.95, '', transform=ax.transAxes, fontsize=10, va='top')
lap_time_text = ax.text(
    0.02, 0.90, '', transform=ax.transAxes, fontsize=10, va='top')
sector_text = ax.text(
    0.02, 0.85, '', transform=ax.transAxes, fontsize=10, va='top')
track_text = ax.text(
    0.02, 0.80, '', transform=ax.transAxes, fontsize=10, va='top')

# Add mode indicator
mode_text = ax.text(
    0.98, 0.95, f"Mode: {'Real-time' if REAL_TIME_MODE else 'Accelerated'}",
    transform=ax.transAxes, fontsize=9, va='top', ha='right',
    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

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
# 8) Animation Timing Calculation - Accurate Real-Time with Interpolation
# =============================================================================
# Calculate animation parameters based on user choice
total_lap_time = time_seconds[-1]  # Real lap duration in seconds
total_frames = len(x)

if REAL_TIME_MODE:
    # Real-time: EXACT timing with smooth 30 FPS thanks to interpolation
    interval_ms = (total_lap_time * 1000) / total_frames
    actual_fps = 1000 / interval_ms
    print(
        f"Real-time mode: Animation will take {total_lap_time:.2f} seconds (actual lap time)")
    print(f"Frame rate: {actual_fps:.1f} FPS (interpolated for smoothness)")
else:
    # Accelerated mode: use original fast playback speed
    fps_display = 14
    speed_factor = 1.0
    interval_ms = (60 / fps_display) / speed_factor
    animation_duration = (interval_ms * total_frames) / 1000
    print(
        f"Accelerated mode: Animation will take {animation_duration:.1f} seconds (vs {total_lap_time:.1f}s real lap)")
    print(f"Frame rate: Original accelerated speed")

print(f"Final settings: {total_frames} frames, {interval_ms:.1f} ms interval")

# =============================================================================
# 8) Animation Update Function
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

    # Determine sector based on distance instead of time/coordinates
    if frame <= idx_sector1_end:
        sector_text.set_text("Sector: 1")
    elif frame <= idx_sector2_end:
        sector_text.set_text("Sector: 2")
    else:
        sector_text.set_text("Sector: 3")

    track_text.set_text(f"Distance: {cumulative_distance[frame]:.1f} m")
    return point, speed_text, lap_time_text, sector_text, track_text


# =============================================================================
# 9) Run Animation in Matplotlib Window
# =============================================================================
ani = FuncAnimation(fig, update, frames=total_frames,
                    interval=interval_ms, blit=True)

# Show additional info before starting
print(f"\n=== Starting Animation ===")
print(f"Driver: {DRIVER}")
print(f"Circuit: {GRAND_PRIX}")
print(f"Lap time: {lap_time_str}")
print(f"Animation mode: {'Real-time' if REAL_TIME_MODE else 'Accelerated'}")
print("Close the window to exit.")

plt.show()
