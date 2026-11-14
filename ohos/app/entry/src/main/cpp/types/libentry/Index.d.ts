/**
 * The native side of the app (entry/src/main/cpp/app.cpp).
 * init: call before the XComponent is built; the render thread waits for it.
 */
export const init: (resourceManager: object, filesDir: string, args: string) => void;
export const setForeground: (foreground: boolean) => void;
/** One line about what the native side is doing: progress while loading, frame times, or the error. */
export const getStatus: () => string;
