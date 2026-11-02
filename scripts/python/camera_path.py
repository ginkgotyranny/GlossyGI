"""Keyframe camera paths for Falcor pyscene scripts.

Each pose is a dict with position, target, up, and optionally focalLength,
focalDistance, and apertureRadius. The animation linearly interpolates the
view transform. Playback loops over the duration, so the first pose is
repeated at the end when the sequence does not already close.
"""


def _same_vec(a, b):
    return a.x == b.x and a.y == b.y and a.z == b.z


def _same_pose(a, b):
    return _same_vec(a["position"], b["position"]) and _same_vec(a["target"], b["target"]) and _same_vec(a["up"], b["up"])


def addCameraPath(sceneBuilder, camera, poses, duration):
    """Animate camera through poses over duration seconds, then loop."""
    from falcor import Animation, Transform

    if len(poses) < 2:
        raise ValueError("addCameraPath needs at least two poses")
    if duration <= 0.0:
        raise ValueError("addCameraPath duration must be positive")

    keys = [dict(pose) for pose in poses]
    if not _same_pose(keys[0], keys[-1]):
        keys.append(dict(keys[0]))

    first = keys[0]
    for name in ("focalLength", "focalDistance", "apertureRadius"):
        if name not in first:
            continue
        if any(pose.get(name) != first[name] for pose in keys):
            print("Warning: %s differs between camera poses and is not keyframed; using the first pose." % name)
        setattr(camera, name, first[name])

    step = duration / (len(keys) - 1)
    animation = sceneBuilder.createAnimation(camera, "CameraPath", duration)
    # The scene clock wraps at the longest animation. A shorter path holds its
    # last key after that unless it cycles on its own duration.
    animation.postInfinityBehavior = Animation.Behavior.Cycle
    animation.preInfinityBehavior = Animation.Behavior.Cycle
    for index, pose in enumerate(keys):
        transform = Transform()
        transform.lookAt(pose["position"], pose["target"], pose["up"])
        animation.addKeyframe(index * step, transform)
    return animation
