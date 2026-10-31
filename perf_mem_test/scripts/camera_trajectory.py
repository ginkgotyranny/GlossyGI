"""
Camera trajectory module for Falcor benchmark.

Defines a Catmull-Rom spline through recorded waypoints and exposes
a per-frame callback that drives the camera along the trajectory.

Waypoints are loaded from camera_waypoints.txt (format produced by
record_waypoints.py). Both position and target are interpolated so
the camera pans smoothly between viewpoints.

Usage (inside a render graph script or benchmark.py):
    from camera_trajectory import build_callback
    m.sceneUpdateCallback = build_callback(total_duration=20.0, loops=1)
"""

import math

# float3 is a Falcor type injected by the embedded Python environment.
# Import defensively so this module works both when loaded via m.script()
# (where falcor is already in the namespace) and when imported standalone.
try:
    from falcor import float3
except ImportError:
    try:
        float3  # already in globals (e.g. loaded via `from falcor import *`)
    except NameError:
        def float3(x, y, z):
            return (x, y, z)

# Waypoints recorded by record_waypoints.py
# Each tuple: (px, py, pz, tx, ty, tz, dwell_seconds)
WAYPOINTS = [
    (-0.0384, 0.7121, -0.6126,  -1.0226, 0.7627, -0.7817,  0.5),
    (-0.5758, 0.7832, -1.0940,  -1.3766, 0.9156, -0.5099,  0.5),
    (-0.9711, 0.9239, -0.9387,  -1.9276, 1.1207, -0.7236,  0.5),
    (-1.2780, 1.0022, -0.9206,  -2.2187, 0.9863, -1.2596,  0.5),
    (-1.5453, 1.0382, -1.2433,  -2.1464, 1.2219, -2.0210,  0.5),
    (-1.5453, 1.0382, -1.2433,  -1.7276, 1.2464, -2.2042,  0.5),
    (-1.2987, 1.0382, -1.2864,  -1.1503, 1.2464, -2.2532,  0.5),
    (-0.0467, 0.8673, -1.0145,  -0.6279, 1.1488, -1.7780,  0.5),
    (0.2293, 0.8357, -1.2033,  -0.7387, 0.9996, -1.3930,  0.5),
    (0.2205, 0.9211, -0.5782,  -0.7312, 1.0163, -0.8701,  0.5),
]

# ---- interpolation ---------------------------------------------------------

def _catmull_rom(p0, p1, p2, p3, t):
    """Catmull-Rom spline interpolation for a single float."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
        (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )

def _interp_segment(points, i, t):
    """Interpolate a 3D point on segment i (between points[i] and points[i+1])."""
    n = len(points)
    p0 = points[(i - 1) % n]
    p1 = points[i]
    p2 = points[(i + 1) % n]
    p3 = points[(i + 2) % n]
    return (
        _catmull_rom(p0[0], p1[0], p2[0], p3[0], t),
        _catmull_rom(p0[1], p1[1], p2[1], p3[1], t),
        _catmull_rom(p0[2], p1[2], p2[2], p3[2], t),
    )

def _build_segment_durations(waypoints, segment_time):
    """Duration of each spline segment = travel time + dwell at endpoint."""
    n = len(waypoints)
    return [segment_time + waypoints[i][6] for i in range(n)]

def _trajectory_at(local_time, waypoints, seg_durations, total_duration):
    """
    Given elapsed time in [0, total_duration), return (pos, target, finished).
    pos = (x,y,z), target = (x,y,z).
    """
    if local_time >= total_duration:
        wp = waypoints[0]
        return (wp[0:3], wp[3:6], True)

    # find which segment we're in
    n = len(waypoints)
    positions = [(wp[0], wp[1], wp[2]) for wp in waypoints]
    targets   = [(wp[3], wp[4], wp[5]) for wp in waypoints]

    elapsed = 0.0
    for i in range(n):
        seg_dur = seg_durations[i]
        if local_time < elapsed + seg_dur:
            local_t = local_time - elapsed
            travel_dur = seg_dur - waypoints[i][6]
            if local_t < travel_dur:
                # interpolating along the spline
                u = local_t / travel_dur if travel_dur > 0 else 0.0
                pos = _interp_segment(positions, i, u)
                tgt = _interp_segment(targets, i, u)
            else:
                # dwelling at waypoint i+1
                pos = positions[(i + 1) % n]
                tgt = targets[(i + 1) % n]
            return (pos, tgt, False)
        elapsed += seg_dur

    wp = waypoints[0]
    return (wp[0:3], wp[3:6], True)

# ---- public API ------------------------------------------------------------

def build_callback(total_duration=20.0, loops=1, segment_time=None):
    """
    Build a sceneUpdateCallback that drives the camera along the trajectory.

    Args:
        total_duration: total time in seconds for one loop (ignored if
                        segment_time is given).
        loops: number of times to repeat the trajectory.
        segment_time: explicit travel time per segment in seconds. If None,
                      derived from total_duration minus dwell times.

    Returns:
        A function(scene, current_time) suitable for m.sceneUpdateCallback.
    """
    n = len(WAYPOINTS)
    if segment_time is not None:
        seg_durations = _build_segment_durations(WAYPOINTS, segment_time)
    else:
        total_dwell = sum(wp[6] for wp in WAYPOINTS)
        seg_time = max(0.5, (total_duration - total_dwell) / n)
        seg_durations = _build_segment_durations(WAYPOINTS, seg_time)

    one_loop = sum(seg_durations)
    full_duration = one_loop * loops

    print(f"[trajectory] {n} waypoints, {loops} loop(s), "
          f"one loop = {one_loop:.1f}s, total = {full_duration:.1f}s")

    def callback(scene, current_time):
        t = current_time % one_loop if loops > 1 else min(current_time, one_loop)
        pos, tgt, finished = _trajectory_at(t, WAYPOINTS, seg_durations, one_loop)
        scene.camera.position = float3(pos[0], pos[1], pos[2])
        scene.camera.target = float3(tgt[0], tgt[1], tgt[2])

    return callback, full_duration
