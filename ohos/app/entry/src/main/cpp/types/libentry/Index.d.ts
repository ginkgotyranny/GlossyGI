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
