/**
 * The native side of the app (entry/src/main/cpp/app.cpp).
 * init: call before the XComponent is built; the render thread waits for it.
 */
export const init: (resourceManager: object, filesDir: string, args: string) => void;
export const setForeground: (foreground: boolean) => void;
/**
 * What the panel shows, as JSON: the status line, the scenes on the phone, and for the running scene
 * its settings, capabilities, training schedule, animation time and frame times (pages/Index.ets
 * declares the shape).
 */
export const getState: () => string;
/** A setting of the panel or of the training schedule by name (Runtime::settingNames), as a number. */
export const set: (name: string, value: number) => void;
/**
 * scene "<scene>/<variant>", rescan, resetSchedule, resetCamera, saveModel, loadModel, resetModel.
 * Applied by the render thread at the start of its next frame.
 */
export const command: (name: string, argument: string) => void;
/**
 * The camera controls, used while the camera does not follow its path. The stick moves the camera as
 * a gamepad's left stick does in Falcor: x to the right, y down as the screen counts (pushing up moves
 * forward), within the unit circle; it stays where it was set until set again (0, 0 to stop).
 */
export const cameraStick: (x: number, y: number) => void;
/** Up (1), down (-1) or neither (0), until set again. */
export const cameraLift: (direction: number) => void;
/** Turns the view by these angles in radians: yaw to the right and pitch down for positive values. */
export const cameraLook: (yaw: number, pitch: number) => void;
