"""Mogwai entry script of run.bat / run.sh for launcher options that act on the loaded scene.

Mogwai runs --script before it loads --scene, so a render-graph script cannot reach the scene. When an
option needs the scene, the launchers start Mogwai with --deferred and this script: --deferred runs it
at the first frame, after the scene has loaded and before the scene's first update. It runs the
render-graph script the user chose, then applies the options; in that order they also reach a scene the
graph script loads itself (WARDiffPathTracer.py does).

Environment (set by the launchers):
  NCR_GRAPH_SCRIPT     path of the render-graph script to run
  NCR_CAMERA_ANIMATED  1 or 0: whether the cameras follow their animation path, the "Animated" box under
                       Scene Settings > Camera (also shown there as "Animate Camera"). Unset: the scene's
                       own setting, which is on.
"""
import os

graph_script = os.environ["NCR_GRAPH_SCRIPT"]
exec(compile(open(graph_script).read(), graph_script, "exec"), globals())

camera_animated = os.environ.get("NCR_CAMERA_ANIMATED")
if camera_animated is not None:
    if m.scene is None:
        print("[run_launch] no scene loaded: NCR_CAMERA_ANIMATED is ignored")
    else:
        for camera in m.scene.cameras:
            camera.animated = camera_animated == "1"
        print(f"[run_launch] camera animation {'on' if camera_animated == '1' else 'off'}")
